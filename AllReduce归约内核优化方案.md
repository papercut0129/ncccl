# AllReduce 归约内核优化方案

> 项目定位：单卡环境下，以 NCCL 归约内核（`reduceCopyPacks` / `Apply_Reduce`）为参照，
> 在可 benchmark 的 element-wise 归约场景上复现并验证其优化手段。
> 本文只做分析设计，不改动 NCCL 源码。

---

## 1. 为什么优化

AllReduce 在分布式训练里做梯度同步，本质是两件事：**通信搬运 + 归约计算**。
其中归约计算（把各卡对应位置的梯度逐元素相加）是纯访存 + 计算密集的部分：

1. **数据量巨大**：梯度数组和模型参数一样多，动辄几亿到几百亿个元素。
2. **频率极高**：每步训练都做一次 AllReduce，一次训练几百万步。
3. **访存是瓶颈**：归约基本是"读两个数、加一下、写一个数"，计算很轻，性能完全卡在显存带宽上。

因此优化的核心目标是：**让"逐元素归约"尽可能吃满显存带宽**。
手段就是减少访存指令数、提高每线程有效工作量、消除循环开销。

---

## 2. 优化前是什么样（baseline）

如果不用任何优化，最直接的 element-wise 归约是"标量逐个加，一个线程处理一个元素"：

```c
// 优化前：naive 标量归约
__global__ void naive_reduce(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];   // 每个线程只处理 1 个 float，标量访存
  }
}
```

**存在的问题**：

- 每个线程只搬 3 个 `float`（读 2 个 + 写 1 个），访存粒度太小；
- 一次只能处理 4 字节，无法利用 128 位访存（一条指令读 16 字节）;
- 线程数量等于元素数量，线程调度开销大；
- 没有合并访存和向量化，带宽利用率低。

---

## 3. 优化了哪里（4 个优化点，对应 NCCL 源码）

### 优化 1：标量访存 → 16 字节向量化（float4）

**NCCL 源码依据**：[common_kernel.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/common_kernel.h:210) 里 `reduceCopy` 优先用 16 字节大包：

```c
// NCCL：优先 16 字节打包（= float4），指针不对齐才退回标量
constexpr int BigPackSize = 16;
if (aligned) {
  reduceCopyPacks<RedFn, T, Unroll, BigPackSize, ...>(...);  // 16B 一次
}
```

[common_kernel.h:40](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/common_kernel.h:40) 里用 16 字节向量化加载/存储：

```c
acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);   // 一次读 16 字节
...
st_global<BytePerPack>(minDsts[d], acc[u]);              // 一次写 16 字节
```

**优化前**：`float` 标量，一条访存指令只搬 4 字节。
**优化后**：`float4`，一条访存指令搬 16 字节（4 个 float）。

**收益**：访存指令数减少到 1/4，显存带宽利用率大幅提升。

```c
// 优化后：float4 向量化
__global__ void vec_reduce(const float4* a, const float4* b, float4* c, int n4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    c[i].x = a[i].x + b[i].x;
    c[i].y = a[i].y + b[i].y;
    c[i].z = a[i].z + b[i].z;
    c[i].w = a[i].w + b[i].w;
  }
}
```

### 优化 2：每线程 1 元素 → 每线程多元素（Unroll）

**NCCL 源码依据**：[common_kernel.h:40](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/common_kernel.h:40) 的 `reduceCopyPacks` 用 `Unroll` 模板参数让每个线程一次处理多个包：

```c
BytePack<BytePerPack> acc[Unroll];     // 每个线程的累加器数组，一次处理 Unroll 个包
NVCC_PRAGMA_UNROLL(Unroll)
for (int u = 0; u < Unroll; u++) {
  acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);
  minSrcs[0] += WARP_SIZE * BytePerPack;
}
```

**优化前**：一个线程只处理 1 个元素，线程数 = 元素数。
**优化后**：一个线程连续处理多个元素（grid-stride 或固定 Unroll），线程数 = 元素数 / Unroll。

**收益**：

- 减少线程数量和调度开销；
- 每个线程连续访问，合并访存更友好；
- 多个 load 之间可以重叠，隐藏访存延迟。

### 优化 3：运行时循环 → 编译期展开

**NCCL 源码依据**：[reduce_kernel.h:320](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce_kernel.h:320) 的 `Apply_Reduce` 用**编译期递归模板**，把 16 字节包拆成逐元素归约，而不是运行时循环：

```c
// 编译期递归：把 BytePack 拆成两半分别归约，直到 EltPerPack=1
template <typename Fn, int EltPerPack>
struct Apply_Reduce {
  static BytePack<Size> reduce(Fn fn, BytePack<Size> a, BytePack<Size> b) {
    a.half[0] = Apply_Reduce<Fn, EltPerPack / 2>::reduce(fn, a.half[0], b.half[0]);
    a.half[1] = Apply_Reduce<Fn, EltPerPack / 2>::reduce(fn, a.half[1], b.half[1]);
    return a;
  }
};
```

**优化前**：用 `for` 循环在运行时遍历元素，每次迭代有循环判断和跳转。
**优化后**：编译器在编译期把循环完全展开成顺序指令，无循环开销。

**收益**：消除循环判断/跳转指令，减少分支，利于指令流水和寄存器分配。

### 优化 4：单源归约 → 多源归约

**NCCL 源码依据**：[prims_simple.h:261](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/prims_simple.h:261) 里 `reduceCopy` 的 `nSrcs` 参数支持一次归约多个源：

```c
// AllReduce 归约时：源 = 1 个对端 + 1 个本地 = 2 个源
reduceCopy<Unroll, RedOp, T, ...>(
  tid, nworkers, redOpArgs, postOp,
  Recv * fan.nrecv() + Src,   // nSrcs：多个源一起归约
  srcs,                        // 源指针数组
  Send * fan.nsend() + Dst,   // nDsts
  dsts, ...);
```

**优化前**：一次只归约两个数（对端 + 本地），多个 rank 的数据要多次调用。
**优化后**：把"所有源"一次性加载、归约到累加器，再统一写回。

**收益**：减少中间结果的写回和重复读取，归约链路更短、更省带宽。

---

## 4. 优化后提升了什么

| 优化项 | 优化前 | 优化后 | 主要收益 |
|---|---|---|---|
| 向量化 | `float` 标量（4B/指令） | `float4`（16B/指令） | 访存指令数降到 1/4，带宽↑ |
| 多元素 Unroll | 1 线程 1 元素 | 1 线程 N 元素 | 减少线程切换，隐藏延迟 |
| 编译期展开 | 运行时 for 循环 | 递归模板展开 | 消除循环/分支开销 |
| 多源归约 | 单源多次调用 | 多源一次归约 | 减少中间读写，链路更短 |

预期总效果：从"标量逐加、线程爆炸"的 naive 版，到"16B 向量化 + 多元素 + 编译期展开"的优化版，
有效带宽通常能从远低于峰值提升到接近显存带宽上限，这也是 NCCL 归约内核能跑满带宽的原因。

---

## 5. 如何验证

**正确性**：与 CPU 串行结果、`cub::DeviceReduce` 或 `thrust::transform` 对比，误差在浮点精度内。

**性能 benchmark**：

- 对不同规模（1M / 16M / 64M 个 float）跑 element-wise 归约（`c[i] = a[i] + b[i]`）；
- 用 CUDA event 计时，计算有效带宽 `GB/s = 读写字节数 / 耗时`；
- 对比 naive 标量版 vs float4 向量化版 vs +Unroll 版 vs +编译期展开版，画提升曲线。

**硬件指标（nsight compute）**：

- `memory throughput`：是否接近显存带宽上限；
- `L1/L2 hit rate`、`global load/store transactions`：验证向量化后访存事务减少；
- `instructions issued`：验证编译期展开后指令数下降；
- `achieved occupancy`：验证 Unroll 后线程配置是否合理。

---

## 6. 简历项目表述

> 以 NCCL AllReduce 的归约内核（`reduceCopyPacks` / `Apply_Reduce`）为参照，
> 在单卡上复现并验证 element-wise 归约的四项优化：16 字节向量化（float4）、
> 每线程多元素（Unroll）、编译期递归展开、多源归约。
> 通过 nsight compute 定位访存瓶颈，优化后有效带宽相对 naive 标量实现提升 X 倍。

---

## 7. 相关源码索引

| 优化点 | NCCL 源码位置 |
|---|---|
| 16B 向量化 | [common_kernel.h:210](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/common_kernel.h:210) `BigPackSize = 16` |
| 向量化加载/存储 | [common_kernel.h:40](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/common_kernel.h:40) `reduceCopyPacks` |
| 多元素 Unroll | [common_kernel.h:40](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/common_kernel.h:40) `acc[Unroll]` |
| 编译期展开 | [reduce_kernel.h:320](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce_kernel.h:320) `Apply_Reduce` 递归 |
| 多源归约 | [prims_simple.h:261](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/prims_simple.h:261) `nSrcs` |
| 最终加法 | [reduce_kernel.h:338](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce_kernel.h:338) `FuncSum` |
