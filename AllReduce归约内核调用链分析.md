# AllReduce 归约内核：函数调用链与来龙去脉

> 目标：从最顶层 API 到最底层“加法”，梳理 AllReduce 归约内核的完整调用链，
> 说明每个函数的作用，以及归约内核的来龙去脉。
> 基于 NCCL 源码 `src/` 目录分析。

---

## 一、完整函数调用链（5 层）

```
【第 1 层：Host 入口】
ncclAllReduce()                                        src/collectives.cc
  └─ ncclAllReduceConfigImpl()   构造 ncclInfo
       └─ ncclEnqueueCheck()      算法/协议选择、启动 kernel    src/enqueue/enqueue.cc

【第 2 层：Kernel 调度】
ncclDevKernel_AllReduce_Sum_f32_RING_SIMPLE()          src/device/generate.py 生成
  └─ ncclKernelMain()                                   src/device/common.h
       └─ RunWorkBatch<AllReduce,...>::run()            src/device/common.h
            └─ RunWorkColl<AllReduce,...,RING,SIMPLE>::run()   src/device/all_reduce.h

【第 3 层：Ring 算法】
                 └─ runRing()                           src/device/all_reduce.h
                      └─ prims.directRecvReduceDirectSend()      // 收+归约+发
                      └─ prims.directRecvReduceCopyDirectSend()  // 收+归约+存+发

【第 4 层：搬运原语】
                           └─ genericOp<1,1,1,1,Input,-1>()      src/device/prims_simple.h
                                └─ reduceCopy()                  src/device/common_kernel.h

【第 5 层：归约内核】
                                     └─ reduceCopyPacks()        src/device/common_kernel.h
                                          └─ applyReduce()
                                               └─ Apply_Reduce<>::reduce()   src/device/reduce_kernel.h
                                                    └─ FuncSum: a + b        src/device/reduce_kernel.h
```

---

## 二、每层函数的作用

### 第 1 层：Host 入口

| 函数 | 作用 |
|---|---|
| `ncclAllReduce` | 公开 API，只做 NVTX 性能埋点 |
| `ncclAllReduceConfigImpl` | 把参数（sendbuff/recvbuff/count/datatype/op）打包成 `ncclInfo` |
| `ncclEnqueueCheck` | 校验参数、用代价模型选 Ring/Tree/NVLS 和 LL/LL128/SIMPLE、切分 work、启动 kernel |

### 第 2 层：Kernel 调度

| 函数 | 作用 |
|---|---|
| `ncclDevKernel_*` | 真正的 `__global__` 核函数（由 generate.py 按“算子×类型×算法×协议”生成） |
| `ncclKernelMain` | 核函数主循环：加载 work、按 funcId 分发 |
| `RunWorkBatch::run` | 批处理入口，逐个 work 调用 `RunWorkColl` |
| `RunWorkColl::run` | 具体算子入口，转发给算法函数 |

### 第 3 层：Ring 算法

| 函数 | 作用 |
|---|---|
| `runRing` | Ring AllReduce 两阶段（scatter-reduce + all-gather），决定“每一步收发哪块数据” |
| `directRecvReduceDirectSend` | **收对端 + 归约 + 转发**（阶段 1 中间步） |
| `directRecvReduceCopyDirectSend` | **收对端 + 归约 + 存本地 + 转发**（阶段 1 最后一步） |

### 第 4 层：搬运原语（关键桥梁）

| 函数 | 作用 |
|---|---|
| `genericOp` | 所有 send/recv/reduce 原语的统一入口，把“收几个、发几个、从哪读、写哪”转成 `reduceCopy` 的参数 |

`directRecvReduceDirectSend` 在 [prims_simple.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/prims_simple.h:1000) 里就是一行：

```c
// 模板参数 <DirectRecv, DirectSend, Recv, Send, SrcBuf, DstBuf>
genericOp<1, 1, 1, 1, Input, -1>(inpIx, outIx, eltN, postOp);
//        ↑  ↑  ↑  ↑   ↑      ↑
//       直收 直发 收1 发1  读本地sendbuff  不写本地输出
```

### 第 5 层：归约内核（核心）

| 函数 | 作用 |
|---|---|
| `reduceCopy` | 归约入口，判断 16 字节对齐，选大包/标量路径 |
| `reduceCopyPacks` | 真正的“多源归约 + 多目标写出”，16B 向量化 + Unroll |
| `Apply_Reduce` | 编译期递归，把 16 字节包拆成逐元素归约 |
| `FuncSum` | 最终执行 `a + b` |

---

## 三、归约内核的“来龙去脉”

### 第 1 步：genericOp 把“归约”翻译成 reduceCopy 的参数

在 [prims_simple.h:247](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/prims_simple.h:247) 附近，`genericOp` 根据 `Recv/Send/Src/Dst` 算出源/目的数量：

```c
// 源数量 = 收的对端数 + 本地源数
// （AllReduce 归约时 = 1 个对端 + 1 个本地 = 2 个源）
reduceCopy<Unroll, RedOp, T, ...>(
  tid, nworkers, redOpArgs, postOp,
  Recv * fan.nrecv() + Src,   // nSrcs：几个源要归约
  srcs,                        // 源指针数组（对端数据 + 本地数据）
  Send * fan.nsend() + Dst,   // nDsts：几个目的要写
  dsts,                        // 目的指针数组（本地输出 + 发给对端）
  workSize);                   // 本次要归约的元素数
```

**对 AllReduce 的 `directRecvReduceDirectSend` 来说**：

- 2 个源 = 对端传来的梯度块 + 本地对应位置的梯度；
- 归约 = 把这两个源**逐元素相加**；
- 2 个目的 = 写回本地 + 发给下一个 rank。

### 第 2 步：reduceCopy 决定用多大粒度搬

在 [common_kernel.h:210](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/common_kernel.h:210)：

```c
constexpr int BigPackSize = 16;  // 优先 16 字节（= float4）
// 指针对齐就用 16B 大包，否则退回 sizeof(T) 标量
```

### 第 3 步：reduceCopyPacks 做向量化归约

在 [common_kernel.h:40](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/common_kernel.h:40)：

```c
// 加载第一个源到累加器（16 字节一次）
acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);
// 其余源逐个加载并归约
acc[u] = applyReduce(redFn, acc[u], tmp[u]);
// 写回所有目的
st_global<BytePerPack>(minDsts[d], acc[u]);
```

### 第 4 步：Apply_Reduce 把“包”拆成“元素”

在 [reduce_kernel.h:320](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce_kernel.h:320)：

```c
// 编译期递归：16 字节包 → 8 → 4 → 2 → 1，拆到单个元素
a.half[0] = Apply_Reduce<Fn, EltPerPack/2>::reduce(fn, a.half[0], b.half[0]);
a.half[1] = Apply_Reduce<Fn, EltPerPack/2>::reduce(fn, a.half[1], b.half[1]);
```

### 第 5 步：FuncSum 做真正的加法

在 [reduce_kernel.h:338](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce_kernel.h:338)：

```c
// 递归到底，EltPerPack=1，最终就是一次 a + b
return toPack<T>(fromPack<T>(a) + fromPack<T>(b));
```

---

## 四、一句话总结来龙去脉

```
AllReduce 要“把各卡梯度对应相加”
  → Ring 算法决定“哪块数据什么时候去哪”（runRing）
  → 原语决定“收几个、发几个”（directRecvReduceDirectSend → genericOp）
  → reduceCopy 决定“用 16 字节还是标量搬”（对齐判断）
  → reduceCopyPacks 用向量化 + 多元素搬运并归约
  → Apply_Reduce 把 16 字节包编译期拆成单个元素
  → FuncSum 执行最终的 a + b
```

**通信部分（Ring 怎么转、数据从哪来）在上半段，归约部分（搬来之后怎么加）在下半段。**

归约内核 = `reduceCopy → reduceCopyPacks → Apply_Reduce → FuncSum` 这条从“判断粒度”到“真正加法”的链路。
其优化核心是：**16 字节向量化（少读少写）+ 每线程多元素（Unroll）+ 编译期展开（避免运行时循环）**。

---

## 五、相关源码文件索引

| 文件 | 关键内容 |
|---|---|
| [collectives.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:204) | `ncclAllReduce` 公开入口与 `ncclInfo` 构造 |
| [enqueue.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/enqueue/enqueue.cc:3383) | `ncclEnqueueCheck` 入队与调度 |
| [common.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/common.h:395) | `ncclKernelMain`、`RunWorkBatch` |
| [all_reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_reduce.h:23) | `runRing`、`runTreeUpDown`、`runTreeSplit` |
| [prims_simple.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/prims_simple.h:929) | `genericOp`、各种 send/recv/reduce 原语 |
| [common_kernel.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/common_kernel.h:40) | `reduceCopy`、`reduceCopyPacks` |
| [reduce_kernel.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce_kernel.h:320) | `Apply_Reduce`、`FuncSum` 等归约 functor |
