# NCCL 算子总结与横向对比

本文档回答一个问题：NCCL 里有哪些“算子”，它们分别做什么，哪些地方是共用的，哪些地方又彼此不同。

源码根目录：`C:/Users/79811/Desktop/nccl/nccl-master/nccl-master`

关键源码入口：

- 内部算子枚举：[nccl_common.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/include/nccl_common.h:72)
- 对外 API 声明：[nccl.h.in](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/nccl.h.in:517)
- 对外 API 入队实现：[collectives.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:22)
- 设备端公共执行框架：[common.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/common.h:434)
- 底层传输原语：[primitives.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/primitives.h:26)
- 内核生成算法矩阵：[generate.py](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/generate.py:20)

## 1. NCCL 算子全景

NCCL 中的“算子”不是只有一个层次，而是可以分为四类：

1. **核心集合通信算子**：Broadcast、Reduce、AllGather、ReduceScatter、AllReduce。
   - 这是 NCCL 最核心的设备端内核，集中在 `src/device/` 下的同名 `.h` 文件中。
   - 它们具备独立的 `RunWorkColl` 模板特化，按“算子 + 数据类型 + 归约操作 + 算法 + 协议”编译成大量内核。

2. **点对点算子**：Send、Recv，以及内部统一使用的 SendRecv。
   - 对外 API 是 `ncclSend` / `ncclRecv`。
   - 设备端统一由 `src/device/sendrecv.h` 的 `RunWorkBatch<ncclFuncSendRecv, ...>` 执行。

3. **组合型集合算子**：AlltoAll、Gather、Scatter。
   - 它们没有独立的设备端集合内核，而是在任务分类阶段被拆成多个 Send/Recv 任务。
   - 拆分逻辑见 [task_classify.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/enqueue/task_prep/task_classify.cc:58)。

4. **单边 / 信号算子**：PutSignal、Signal、WaitSignal。
   - 对应 `ncclPutSignal`、`ncclSignal`、`ncclWaitSignal`，走 RMA 任务队列，而不是传统集合内核。

此外还有内部优化算子 `AllGatherV`，它是变长 AllGather / 批量化 Broadcast 的内部实现，不是公开 API，设备端见 [all_gather_v.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_gather_v.h:16)。

## 2. 核心集合算子语义对比

下表先给出五种核心集合算子最本质的区别。其中 `count` 表示“每个 rank 的单份元素数”。

| 算子 | 数据流 | 是否归约 | root 含义 | 每个 rank 发送量 | 每个 rank 接收量 | 原地条件 |
|---|---|---|---|---|---|---|
| Broadcast | root -> 所有 rank | 否 | 数据来源 rank | root 发 `count`，其余不发 | 每 rank 收 `count` | `sendbuff == recvbuff` |
| Reduce | 所有 rank -> root | 是 | 结果落点 rank | 每 rank 发 `count` | 仅 root 收 `count` | `sendbuff == recvbuff`，非 root 的 `recvbuff` 可为 NULL |
| AllGather | 所有 rank -> 所有 rank | 否 | 无，固定 0 | 每 rank 发 `count` | 每 rank 收 `nranks * count` | `sendbuff == recvbuff + rank * count` |
| ReduceScatter | 所有 rank -> 所有 rank，但结果分散 | 是 | 无，固定 0 | 每 rank 发 `nranks * count` | 每 rank 收 `count` | `recvbuff == sendbuff + rank * count` |
| AllReduce | 所有 rank -> 所有 rank | 是 | 无，固定 0 | 每 rank 发 `count` | 每 rank 收 `count` | `sendbuff == recvbuff` |

语义来源：

- API 注释见 [nccl.h.in](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/nccl.h.in:517)。
- 入队包装见 [collectives.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:153)。

### 2.1 直观理解

- Broadcast 是“一对多复制”。
- Reduce 是“多对一归约，结果只在 root”。
- AllGather 是“人人贡献一份，最后人人拥有全部”，不归约。
- ReduceScatter 是“先全局归约，再把结果切成 n 块分散到各 rank”。
- AllReduce 是“ReduceScatter + AllGather”的合并：先归约分散，再收集广播，最终人人得到完整结果。

如果把 AllReduce 看成复合算子，那么：

- Reduce 是 AllReduce 去掉广播阶段，只保留“归约到 root”。
- Broadcast 是 AllReduce 去掉归约阶段，只保留“从 root 扩散”。
- AllGather 是纯数据传输，不改变数值。
- ReduceScatter 是 AllReduce 的前半段。

## 3. 公共执行框架：所有算子的相同点

NCCL 之所以能高效实现这么多算子，是因为它们共享同一套执行框架。

### 3.1 Host 侧统一入口

每个公开集合算子都遵循“薄封装”模式：

1. `ncclXxxConfigImpl()` 构造一个 `ncclInfo` 描述对象。
2. 解析 `ncclCollConfig_t`。
3. 调用统一的 `ncclEnqueueCheck()` 完成参数校验、算法/协议选择、任务调度和内核启动。

例如 [collectives.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:234) 中的 `ncclAllReduceConfigImpl` 只是把参数填进 `ncclInfo` 后交给统一入口。所有集合算子都是异步入队，函数返回只代表“已经提交到 CUDA stream”，不代表通信完成。

### 3.2 Device 侧统一分发

设备端主循环在 [common.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/common.h:434) 的 `ncclKernelMain`：

1. 把 kernel 参数、comm、channel 加载到共享内存。
2. 通过 `loadWorkBatchToShmem` 加载 work。
3. 调用 `RunWorkBatch::run()`，内部遍历多个 work。
4. 对集合 work，调用 `RunWorkColl<Fn, T, RedOp, Algo, Proto>::run()`。

每个算子的 `.h` 文件只负责特化 `RunWorkColl`，公共循环、work 加载、profiler、abort 检查全部复用。

### 3.3 Channel / CBD 数据划分

所有集合算子都使用 channel 并行，并由 [ncclCollCbdPart](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/include/device.h:357) 计算：

- `partOffset`：当前 channel 在全局 buffer 中的起始元素偏移。
- `partCount`：当前 channel 负责的元素数量。
- `chunkCount`：当前 channel 的 chunk 大小。

这样同一个算子在数据规模变化时，不用改算法结构，只需调整 channel 数和 chunk 大小。

### 3.4 三种传输协议

协议定义在 [primitives.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/primitives.h:37)：

- **LL**：低延迟，16 字节线路里 8 字节数据 + 8 字节 flag，适合极小消息。
- **LL128**：128 字节对齐，中等消息。
- **SIMPLE**：直接用 load/store 搬运，适合大消息。

协议是模板参数，因此同一套算法代码可以参数化复用。

### 3.5 Fan 与 Primitives

`FanAsymmetric` / `FanSymmetric` 描述一个算子在当前节点上的“接收邻居数 / 发送邻居数”：

- Ring 通常是 `FanSymmetric<1>`，即 1 个前驱 + 1 个后继。
- Tree 上行是 `FanAsymmetric<3, 1>`，最多 3 个孩子、1 个父。
- Tree 下行是 `FanAsymmetric<1, 3>`，1 个父、最多 3 个孩子。

所有算子最终都通过 `Primitives` 的 `send` / `recv` / `directSend` / `recvReduceSend` 等方法完成数据搬运和归约。

### 3.6 归约统一走 reduceCopy

凡是涉及归约的算子（Reduce、AllReduce、ReduceScatter），都调用 `reduceCopy`。`reduceCopy` 负责：

- 多源数据合并。
- `preOp`：输入前的局部缩放，例如 `ncclAvg` 在浮点类型下映射为 `PreMulSum`，先乘 `1/nranks`。
- `postOp`：最终结果后处理，例如 `SumPostDiv` 在整数类型下除以 `nranks`。

这是所有归约算子共有的数值语义基础，详见 [reduce_kernel.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce_kernel.h:204)。

## 4. 不同点一：数据流与 root

五种核心算子最核心的差异是数据流方向。

| 算子 | 起点 | 终点 | 中间节点的动作 |
|---|---|---|---|
| Broadcast | root | 所有节点 | 边接收边转发，不归约 |
| Reduce | 所有节点 | root | 边接收边归约，向 root 方向传递 |
| AllGather | 所有节点 | 所有节点 | 边接收边转发，不归约 |
| ReduceScatter | 所有节点 | 所有节点，但每个节点只收一块 | 边接收边归约，结果切块 |
| AllReduce | 所有节点 | 所有节点 | 先归约扩散，再收集广播 |

只有 Broadcast、Reduce、Gather、Scatter 有真正的 `root` 参数。AllGather、ReduceScatter、AllReduce 没有 root，`ncclInfo.root` 固定填 0，或直接忽略。

## 5. 不同点二：是否做归约

按是否改变数值，可以把算子分成两组：

**不归约的搬运类算子：**

- Broadcast
- AllGather
- AlltoAll
- Gather
- Scatter
- Send / Recv

这些算子在构造 `ncclInfo` 时，`op` 只是用 `ncclSum` 占位，设备端只做 copy/send/recv，不调用真正的归约逻辑。

**归约类算子：**

- Reduce
- ReduceScatter
- AllReduce

它们必须携带真实的 `ncclRedOp_t op`，并经过 [enqueue.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/enqueue/enqueue.cc:2480) 把公开 `ncclRedOp_t` 映射成设备端 `ncclDevRedOp_t`：

- `ncclSum` -> `ncclDevSum`
- `ncclProd` -> `ncclDevProd`
- `ncclMin` / `ncclMax` -> `ncclDevMinMax`
- `ncclAvg` -> 浮点为 `ncclDevPreMulSum`，整数为 `ncclDevSumPostDiv`
- 用户自定义 PreMulSum -> `ncclDevPreMulSum`

## 6. 不同点三：算法支持矩阵

算法支持由 [generate.py](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/generate.py:74) 中的 `algos_of_coll` 决定：

| 算子 | Ring | Tree | CollNet Direct | CollNet Chain | NVLS | NVLS_TREE | PAT |
|---|---|---|---|---|---|---|---|
| Broadcast | ✅ | — | — | — | — | — | — |
| Reduce | ✅ | — | — | — | — | — | — |
| AllGather | ✅ | — | ✅ | — | ✅ | — | ✅ |
| ReduceScatter | ✅ | — | ✅ | — | ✅ | — | ✅ |
| AllReduce | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| Send / Recv | 专用 P2P 入口 | — | — | — | — | — | — |

说明：

- Broadcast 和 Reduce 在当前设备内核中只有 Ring 实现。
- AllReduce 的算法最丰富，是 NCCL 调优空间最大的算子。
- AllGather / ReduceScatter 在 NVLink + NVSwitch 场景多了 PAT、NVLS 选择。
- CollNet 系列依赖集合网络硬件，如 IB SHARP。

## 7. 不同点四：协议支持

协议与算法组合由 [generate.py](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/generate.py:83) 的 `required_cuda` 约束：

- RING 和 TREE 通常支持 LL、LL128、SIMPLE 三种协议。
- PAT、NVLS、NVLS_TREE、COLLNET_DIRECT、COLLNET_CHAIN 只支持 SIMPLE。
- 原因是这些硬件加速算法本身面向大消息 / 高带宽，小消息低延迟不划算。

因此：

| 算子 | LL | LL128 | SIMPLE |
|---|---|---|---|
| Broadcast | ✅ | ✅ | ✅ |
| Reduce | ✅ | ✅ | ✅ |
| AllGather | ✅ | ✅ | ✅ |
| ReduceScatter | ✅ | ✅ | ✅ |
| AllReduce | Ring/Tree 全支持；硬件算法仅 SIMPLE | Ring/Tree 全支持 | 全部算法 |
| Send/Recv | 可选 LL | — | ✅ |

## 8. 不同点五：实现文件

| 算子 | Host API | 设备内核 |
|---|---|---|
| Broadcast | [collectives.cc:271](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:271) | [broadcast.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/broadcast.h:23) |
| Reduce | [collectives.cc:354](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:354) | [reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce.h:23) |
| AllGather | [collectives.cc:153](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:153) | [all_gather.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_gather.h:20) |
| ReduceScatter | [collectives.cc:393](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:393) | [reduce_scatter.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce_scatter.h:20) |
| AllReduce | [collectives.cc:234](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:234) | [all_reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_reduce.h:25) |
| Send / Recv | [collectives.cc:469](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:469) | [sendrecv.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/sendrecv.h:18) |

## 9. 组合算子与单边算子的不同实现方式

### 9.1 AlltoAll、Gather、Scatter 拆成 P2P

这三个算子没有独立集合内核。`task_classify.cc` 会把它们转成一组 Send/Recv：

- AlltoAll：对每个 `r`，发 `sendbuff + r*count`，收 `recvbuff + r*count`。
- Gather：所有 rank 向 root 发；root 再按 rank 顺序收。
- Scatter：root 按 rank 顺序发；所有 rank 从 root 收。

见 [task_classify.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/enqueue/task_prep/task_classify.cc:58)。

### 9.2 PutSignal / Signal / WaitSignal 走 RMA

这三者是单边 / 信号操作，不参与传统集合内核调度：

- `ncclPutSignal`：向对端注册窗口写数据并携带信号。
- `ncclSignal`：只发信号，不传数据。
- `ncclWaitSignal`：等待指定 peer/context 上的信号。

它们在 `ncclTaskClassification` 中被归入 `rmaTaskQueue`，见 [task_classify.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/enqueue/task_prep/task_classify.cc:130)。

## 10. 使用场景速查

| 场景 | 推荐算子 / 算法倾向 |
|---|---|
| 梯度同步，所有 GPU 都要完整梯度 | AllReduce |
| 参数服务器，梯度汇聚到单点 | Reduce |
| 从主节点分发模型 / 超参 | Broadcast |
| 多卡并行前向，切分 token 后收集完整序列 | AllGather |
| 把全局归约结果按 rank 切块，后续每卡只处理一块 | ReduceScatter |
| 每卡向其他卡发送不同数据，做转置 / 置换 | AlltoAll |
| 收集所有卡数据到单卡 | Gather |
| 从单卡分发不同数据到各卡 | Scatter |
| 需要精确控制单个发送 / 接收顺序 | Send / Recv，必要时配合 ncclGroupStart/End |

消息大小层面的通用倾向：

- 极小消息：优先低延迟，LL 协议，算法选择偏重启动开销。
- 中等消息：LL128。
- 大消息：SIMPLE，带宽优先。
- 单机多卡：Ring/Tree 通用；有 NVSwitch 时可考虑 NVLS、NVLS_TREE、PAT。
- 多机：Ring、CollNet Direct/Chain；有 SHARP 等集合网络硬件时 CollNet 常更优。

## 11. 一句话总结

NCCL 算子的“同”在于统一 host 入队、统一 channel/CBD 划分、统一协议与 Primitives/reduceCopy；算子的“异”在于数据流方向、是否归约、是否有 root、结果是否复制或分散，以及可选算法和适用硬件。掌握“谁发、谁收、谁归约、结果放在哪”，就能把五种核心集合算子及组合算子清楚地区分开。
