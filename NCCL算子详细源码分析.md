# NCCL 算子详细源码分析

本文档逐算子分析 NCCL 的通信算子：作用、公开 API、host 入队路径、设备端内核算法、协议/算法变体、原地条件与适用场景。

源码根目录：`C:/Users/79811/Desktop/nccl/nccl-master/nccl-master`

## 0. 阅读指南与公共基础

在深入每个算子前，先明确 NCCL 的公共执行模型：

### 0.1 Host 侧统一入队

所有集合算子都遵循相同的三层结构：

```text
ncclXxx() -> ncclXxxConfigImpl() -> ncclEnqueueCheck()
```

- `ncclXxx()`：公开 API，加 NVTX 埋点，`config == NULL`。
- `ncclXxxConfigImpl()`：构造 `ncclInfo`，解析 `ncclCollConfig_t`。
- `ncclEnqueueCheck()`：参数校验、算法选择、协议选择、任务分类、内核启动。

入口实现在 [collectives.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:22)。这些函数只负责异步入队，真正通信在 CUDA stream 上完成。

### 0.2 Device 侧统一分发

设备端主循环是 [ncclKernelMain](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/common.h:434)：

1. 把 kernel 参数、comm、channel 复制到共享内存 `ncclShmem`。
2. 用 `loadWorkBatchToShmem` 加载 work。
3. 循环调用 `RunWorkBatch::run()`。
4. 对集合 work，逐项调用 `RunWorkColl<Fn, T, RedOp, Algo, Proto>::run()`。

每个算子的设备端 `.h` 只提供 `RunWorkColl` 或 `RunWorkBatch` 的特化。

### 0.3 channel 与 chunk

每个集合算子都调用 [ncclCollCbdPart](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/include/device.h:357)，得到：

- `gridOffset` / `partOffset`：当前 channel 负责的数据起点。
- `channelCount` / `partCount`：当前 channel 负责的数据量。
- `chunkCount`：当前 channel 的 chunk 步长。

数据量很大时，channel 和 chunk 共同提供并行度与流水深度。

### 0.4 协议与 Primitives

协议类：

- `ProtoLL`：16 字节线 = 8 字节数据 + 8 字节 flag。
- `ProtoLL128`：128 字节线，数据和 flag 交错。
- `ProtoSimple<SlicePerChunk, StepPerSlice, Unroll>`：直接 load/store。

见 [primitives.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/primitives.h:37)。

`Primitives` 提供 send/recv/directSend/directRecv/recvReduceSend 等底层动作。归约统一由 `reduceCopy` 完成。

## 1. Broadcast

### 1.1 作用

Broadcast 把 root 上的一块数据复制到所有 rank。典型场景：

- 从主节点广播模型参数、随机种子、配置张量。
- AllReduce 的“广播回结果”阶段本质上也是 Broadcast。

### 1.2 公开 API 与语义

```c
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream);
```

- 只有 root 的 `sendbuff` 有意义。
- 所有 rank 的 `recvbuff` 都会得到 root 的数据。
- 原地操作发生在 `sendbuff == recvbuff`。
- 废弃接口 `ncclBcast` 等价于原地 Broadcast。

声明见 [nccl.h.in](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/nccl.h.in:559)。

### 1.3 Host 入队

[ncclBroadcastConfigImpl](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:271) 构造：

```c
struct ncclInfo info = {ncclFuncBroadcast, "Broadcast",
                        sendbuff, recvbuff, count, datatype, ncclSum, root,
                        comm, stream, BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS};
```

Broadcast 不归约，`op` 用 `ncclSum` 占位；`root` 是数据来源 rank。

### 1.4 设备端算法：Ring

当前 Broadcast 设备内核只有 Ring 算法，见 [broadcast.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/broadcast.h:23)。

核心函数 `runRing` 的逻辑是：

```text
for 每个 chunk:
    if rank == root:
        原地则 directSend
        非原地则 directCopySend
    else if nextRank == root:
        directRecv               // 环上最后一个节点，只收不转发
    else:
        directRecvCopyDirectSend // 中间节点，边收边转发
```

解释：

- root 是数据源，先把数据发给下一个 rank。
- 中间 rank 收到上一跳的数据后，立即转发给下一跳。
- `nextRank == root` 的节点是 root 的前驱，说明数据已经绕环一周，只需接收，不需要再转发。

### 1.5 协议特化

[broadcast.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/broadcast.h:85) 提供三种特化：

- `NCCL_PROTO_SIMPLE`
- `NCCL_PROTO_LL`
- `NCCL_PROTO_LL128`

三种协议共用同一个 `runRing<T, RedOp, Proto>`，区别只在于模板参数 `Proto`。

### 1.6 网络卸载与本地拷贝

当 `isNetOffload == true` 时，只用一个 warp 驱动网络通信，其余 warp 负责 root 非原地情况下的本地拷贝。实现见 [broadcast.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/broadcast.h:30)。

### 1.7 适用场景

- 单条消息广播，尤其是消息不大、对延迟敏感时。
- root 需要被多个 GPU 共享的参数或元数据。
- 大消息时走 SIMPLE 提高带宽，小消息走 LL 降低延迟。

## 2. Reduce

### 2.1 作用

Reduce 把所有 rank 的 `count` 个元素按 `op` 归约，结果只放到 root 的 `recvbuff`。典型场景：

- 参数服务器模式，梯度汇聚到一台 master。
- 需要单点保存全局统计量。

### 2.2 公开 API 与语义

```c
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream);
```

- 非 root 的 `recvbuff` 可以传 NULL。
- 原地操作发生在 `sendbuff == recvbuff`。

声明见 [nccl.h.in](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/nccl.h.in:530)。

### 2.3 Host 入队

[ncclReduceConfigImpl](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:354) 构造 `ncclInfo`，其中 `op` 是真实归约操作，`root` 是结果落点 rank。

### 2.4 设备端算法：Ring

当前 Reduce 也只有 Ring 算法，见 [reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce.h:23)。

核心逻辑：

```text
for 每个 chunk:
    if prevRank == root:
        send()                     // root 的下一个节点：起点，只发本地数据
    else if rank == root:
        recvReduceCopy(..., postOp=true) // 终点：接收并归约出最终结果
    else:
        recvReduceSend()           // 中间节点：接收 + 归约 + 继续传递
```

解释：

- 数据从 root 的下一个节点出发，沿环向 root 方向归约推进。
- 每经过一个节点，就把该节点的本地数据合并进去。
- 到达 root 时，完整结果写回 root 的输出缓冲区。

### 2.5 协议特化

[reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce.h:72) 提供 SIMPLE、LL、LL128 三个特化。

### 2.6 适用场景

- 多对一归约，且结果只需要在一处存在。
- 与 AllReduce 相比，Reduce 省掉了广播阶段，适合参数服务器或日志聚合。
- 如果后续每个 rank 都需要完整结果，应直接用 AllReduce。

## 3. AllGather

### 3.1 作用

每个 rank 贡献 `sendcount` 个元素，最终每个 rank 都得到 `nranks * sendcount` 个元素。rank i 的数据位于 `recvbuff + i * sendcount`。

典型场景：

- 多卡并行的序列 / 张量拼接。
- MoE 路由、all-to-all 前对 token 的收集。
- 数据并行中收集每卡处理的 batch 切片。

### 3.2 公开 API 与语义

```c
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                           ncclDataType_t datatype, ncclComm_t comm,
                           cudaStream_t stream);
```

- 接收缓冲区至少 `nranks * sendcount` 个元素。
- 原地条件：`sendbuff == recvbuff + rank * sendcount`。

声明见 [nccl.h.in](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/nccl.h.in:606)。

### 3.3 Host 入队

[ncclAllGatherConfigImpl](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:153) 构造 `ncclInfo`，不归约，`op` 用 `ncclSum` 占位。

### 3.4 设备端算法

AllGather 支持四种算法，见 [all_gather.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_gather.h:20)：

- Ring
- PAT
- NVLS
- CollNet Direct

#### 3.4.1 Ring AllGather

`runRing` 的流程：

```text
for 每个 chunk:
    step 0: 发送本地自己的块给下一个 rank
    中间 nranks-2 步: 接收上一跳的块并转发给下一跳
    最后 1 步: 接收最后一块，落到 recvbuff
```

经过 `nranks - 1` 跳后，所有 rank 都拥有完整数据。

实现见 [all_gather.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_gather.h:20)。

Ring 的三种协议特化见 [all_gather.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_gather.h:102)。

#### 3.4.2 PAT AllGather

PAT 是 Parallel Aggregated Tree。设备端用专门的“算法计算线程”生成 `ncclPatStep`，worker 线程按步骤搬运。

实现见 [all_gather.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_gather.h:131)。

#### 3.4.3 NVLS AllGather

利用 NVSwitch 的 scatter/gather 能力，将数据按 rail 切分并完成汇聚。

实现见 [all_gather.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_gather.h:290)。

#### 3.4.4 CollNet Direct AllGather

分三阶段流水：

1. 发送到集合网络。
2. 网络返回并广播。
3. 接收广播并落地。

实现见 [all_gather.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_gather.h:520)。

### 3.5 适用场景

- 数据量大、拓扑支持 NVSwitch / SHARP 时，优先 NVLS / CollNet / PAT。
- 普通拓扑用 Ring，实现简单且可流水。
- 纯数据搬运，没有归约，因此带宽利用率通常是主要优化目标。

## 4. ReduceScatter

### 4.1 作用

ReduceScatter 先把所有 rank 的 `nranks * recvcount` 个元素做全局归约，再把结果切成 n 块，rank i 只得到第 i 块 `recvcount` 个元素。

典型场景：

- 大规模参数 / 梯度分片：先全局归约，再把参数分片放到不同 GPU。
- 后续每卡只处理自己那块结果时，ReduceScatter 比 AllReduce 少一次收集。

### 4.2 公开 API 与语义

```c
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream);
```

- 参数是 `recvcount`。
- `sendbuff` 至少 `nranks * recvcount` 个元素。
- 原地条件：`recvbuff == sendbuff + rank * recvcount`。

声明见 [nccl.h.in](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/nccl.h.in:589)。

### 4.3 Host 入队

[ncclReduceScatterConfigImpl](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:393) 构造 `ncclInfo`，`count` 字段填的是 `recvcount`。

注意 `ncclFuncSendCount` 对 ReduceScatter 会返回 `nRanks * count`，用于计算实际发送缓冲大小。

### 4.4 设备端算法

ReduceScatter 支持四种算法，见 [reduce_scatter.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce_scatter.h:20)：

- Ring
- PAT
- NVLS
- CollNet Direct

#### 4.4.1 Ring ReduceScatter

`runRing` 的流程：

```text
for 每个 chunk:
    step 0: 发送本地对应的块
    中间 nranks-2 步: 接收上一跳的块 + 归约 + 继续发送
    最后 1 步: 接收 + 归约，结果写回本地 recvbuff
```

实现见 [reduce_scatter.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce_scatter.h:20)。

#### 4.4.2 PAT ReduceScatter

与 AllGather 类似，PAT ReduceScatter 使用算法计算线程调度 worker 线程，按 `PatRSAlgorithm` 生成归约步骤。

实现见 [reduce_scatter.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce_scatter.h:95)。

#### 4.4.3 NVLS ReduceScatter

利用 NVSwitch 的 scatter + reduce 能力，避免 GPU 之间逐跳归约。

实现见 [reduce_scatter.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce_scatter.h:247)。

#### 4.4.4 CollNet Direct ReduceScatter

分三阶段：

1. 分散输入。
2. 从本地 peer 归约后发送到集合网络。
3. 从网络接收最终块。

实现见 [reduce_scatter.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/reduce_scatter.h:460)。

### 4.5 适用场景

- 归约后结果按 rank 切分，后续无需完整结果。
- 在数据并行 / 张量并行 / 流水并行的混合策略中非常常见。
- 有 NVSwitch 或 SHARP 时，硬件归约可以显著减少传输量。

## 5. AllReduce

### 5.1 作用

AllReduce 对所有 rank 的 `count` 个元素做全局归约，最终每个 rank 都得到完全相同的完整结果。它是分布式训练梯度同步的核心算子。

### 5.2 公开 API 与语义

```c
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream);
```

- 原地条件：`sendbuff == recvbuff`。
- 没有 root。

声明见 [nccl.h.in](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/nccl.h.in:575)。

### 5.3 Host 入队

[ncclAllReduceConfigImpl](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:234) 构造 `ncclInfo`。AllReduce 是算法空间最大的集合算子，调优阶段会根据消息大小、拓扑、硬件选择算法和协议。

### 5.4 设备端算法

AllReduce 支持六种算法，见 [all_reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_reduce.h:25)：

- Ring
- Tree
- CollNet Direct
- CollNet Chain
- NVLS
- NVLS_TREE

#### 5.4.1 Ring AllReduce

Ring AllReduce 是 NCCL 最经典的实现：

```text
第一阶段：scatter-reduce
    step 0: 发送本地 chunk 给下一个 rank
    中间 nranks-2 步: 接收 + 归约 + 发送
    最后 1 步: 接收 + 归约，得到最终块

第二阶段：all-gather
    中间 nranks-2 步: 接收 + 拷贝 + 发送
    最后 1 步: 接收最终块
```

每个 rank 在环上转 `2 * (nranks - 1)` 步后，拥有完整归约结果。

实现见 [all_reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_reduce.h:25)。

Ring 的三个协议特化见 [all_reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_reduce.h:296) 和文件末尾的 LL/LL128 特化。

#### 5.4.2 Tree AllReduce

Tree 算法按逻辑树组织 rank：

- 上行 Reduce：叶子发送，中间节点接收并归约后上传，根节点得到完整结果。
- 下行 Broadcast：根节点把结果沿树广播，叶子接收。

NCCL 有两种 Tree 实现：

- `runTreeUpDown`：上行、下行严格分离，先 Reduce 再 Broadcast。
- `runTreeSplit`：把线程分成两组，一组做上行归约，一组做下行广播，二者并行推进，提高大消息带宽。

实现见 [all_reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_reduce.h:120) 和 [all_reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_reduce.h:193)。

Tree + SIMPLE 的入口见 [all_reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_reduce.h:306)，Tree + LL/LL128 见文件末尾。

#### 5.4.3 CollNet Direct / CollNet Chain

- CollNet Direct：借助集合网络硬件，按 Scatter / Gather / Reduce / Broadcast 四段流水完成，见 [all_reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_reduce.h:319)。
- CollNet Chain：链式上行归约、下行广播，见 [all_reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_reduce.h:712)。

二者都依赖集合网络，如 InfiniBand SHARP。

#### 5.4.4 NVLS 与 NVLS_TREE

- NVLS：使用 NVSwitch 的硬件归约能力完成 scatter/gather/reduce/broadcast，见 [all_reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_reduce.h:462)。
- NVLS_TREE：NVLS 负责归约，tree 负责多节点分发/汇聚，见 [all_reduce.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_reduce.h:600)。

### 5.5 适用场景

- 数据并行梯度同步的首选算子。
- 小消息：Ring/Tree + LL，降低启动延迟。
- 中等消息：Ring/Tree + LL128。
- 大消息：Ring/Tree + SIMPLE，或 NVLS / CollNet 等硬件加速。
- 单机多卡且带 NVSwitch：NVLS、NVLS_TREE 常优于 Ring/Tree。
- 多机带 SHARP：CollNet Direct/Chain 常优于纯 Ring。

## 6. Send / Recv

### 6.1 作用

Send / Recv 是点对点通信原语：

```c
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);
```

双方必须使用相同的 `count` 和 `datatype` 配对。它们是 GPU 阻塞式操作；多个 Send/Recv 需要并发推进时，应放进 `ncclGroupStart()` / `ncclGroupEnd()`。

### 6.2 Host 入队

[collectives.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:469) 中：

- `ncclSend` 构造 `ncclFuncSend` 的 `ncclInfo`，`root` 字段复用为 peer。
- `ncclRecv` 构造 `ncclFuncRecv` 的 `ncclInfo`，同样复用 `root` 为 peer。

### 6.3 设备端实现

设备端统一使用 [sendrecv.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/sendrecv.h:18) 的：

```c
RunWorkBatch<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE>
```

实现特点：

- `T` 必须是 1 字节类型，因为 P2P 按字节搬运。
- 一个 work 可以同时包含发送和接收，warp 被分配去执行 send、recv 或本地 copy。
- 根据 `sendProtoLL` / `recvProtoLL` 选择 LL 或 SIMPLE 协议。
- 全 NVLink 或网络注册时，允许使用更大 chunk。

### 6.4 适用场景

- 自定义通信模式，例如稀疏通信、邻居交换。
- 组合算子的底层实现。
- 需要精确控制发送/接收顺序时，配合 group semantics 使用。

## 7. AlltoAll、Gather、Scatter

### 7.1 作用

- AlltoAll：每个 rank 向其他每个 rank 发送不同数据，同时从每个 rank 接收不同数据。
- Gather：所有 rank 把数据汇聚到 root。
- Scatter：root 把不同数据分发到各 rank。

公开 API 见 [collectives.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:197)、[collectives.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:317)、[collectives.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:431)。

### 7.2 实现方式：拆成 P2P

这三个算子没有独立集合内核。任务分类阶段会拆成 Send/Recv：

[task_classify.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/enqueue/task_prep/task_classify.cc:58)：

- AlltoAll：对每个 rank r，生成一个 Send 和一个 Recv。
- Gather：每个非 root rank 发一个 Send；root 生成 n 个 Recv。
- Scatter：root 生成 n 个 Send；每个 rank 生成一个 Recv。

这意味着它们的性能取决于 P2P 调度，而不是专用集合算法。

### 7.3 适用场景

- AlltoAll：MoE token 分发、矩阵转置、FFT 等全交换通信。
- Gather：日志、调试信息、检查点汇聚到单卡。
- Scatter：主节点分发不同切片给 worker。

## 8. PutSignal、Signal、WaitSignal

### 8.1 作用

这是 NCCL 的单边 / 信号算子：

- `ncclPutSignal`：把本地数据写入对端注册窗口，并携带信号。
- `ncclSignal`：只发送信号，不传数据。
- `ncclWaitSignal`：等待指定 peer / sigIdx / ctx 上的信号。

公开 API 见 [collectives.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:521)、[collectives.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:554)、[collectives.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/collectives.cc:584)。

### 8.2 实现方式：RMA 队列

这些算子被 `ncclTaskClassification` 归入 `rmaTaskQueue`，不进入传统集合内核调度，见 [task_classify.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/enqueue/task_prep/task_classify.cc:130)。

### 8.3 适用场景

- 单边通信：目标进程不需要显式参与数据传输。
- 细粒度同步：用 Signal/WaitSignal 表示依赖关系。
- PutSignal 适合远程写加同步标记，替代“Put + Signal”的两次操作。

## 9. 内部算子 AllGatherV

AllGatherV 是 NCCL 内部使用的变长 AllGather / 批量化 Broadcast。

设备端见 [all_gather_v.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/device/all_gather_v.h:16)：

- 使用 `RunWorkBatch`，一次 kernel 处理多个 work。
- 每个 work 有 `bytes`、`bytes_done`、`chunkSize`、`ringDepth`。
- `ringDepth == 0` 的 rank 是数据源，发本地数据；中间 rank 接收并转发；`ringDepth == nranks - 1` 的 rank 只接收。

它主要用于某些 Broadcast 路径的批量化优化，不直接作为公开 API 暴露。

## 10. 归约操作符补充

NCCL 中除了通信算子，还有“归约操作符”这一层概念。

公开归约操作符见 [nccl.h.in](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/nccl.h.in:448)：

```c
ncclSum, ncclProd, ncclMax, ncclMin, ncclAvg
```

设备端归约操作符见 [device.h](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/include/device.h:60)：

```c
ncclDevSum, ncclDevProd, ncclDevMinMax, ncclDevPreMulSum, ncclDevSumPostDiv
```

映射关系在 [enqueue.cc](C:/Users/79811/Desktop/nccl/nccl-master/nccl-master/src/enqueue/enqueue.cc:2480)：

- `ncclSum` -> `ncclDevSum`
- `ncclProd` -> `ncclDevProd`
- `ncclMin` / `ncclMax` -> `ncclDevMinMax`
- `ncclAvg` 对浮点类型 -> `ncclDevPreMulSum`（先乘 1/nranks 再求和）
- `ncclAvg` 对整数类型 -> `ncclDevSumPostDiv`（先求和再除 nranks）

这些设备端归约操作是 Reduce、ReduceScatter、AllReduce 三个算子的数值基础，搬运类算子不使用。

## 11. 总结

NCCL 的算子体系可以这样记忆：

```text
核心五种集合算子：
  Broadcast    = 一对多复制
  Reduce       = 多对一归约
  AllGather    = 人人贡献、人人拥有、不归约
  ReduceScatter = 先归约、再分块
  AllReduce    = ReduceScatter + AllGather

P2P：
  Send / Recv，以及由它们组合出的 AlltoAll / Gather / Scatter

单边信号：
  PutSignal / Signal / WaitSignal
```

从源码角度看，五种核心集合算子的差异主要体现在 `RunWorkColl` 特化中的循环逻辑；共同点则体现在统一的 `ncclInfo`、`ncclEnqueueCheck`、`ncclKernelMain`、`ncclCollCbdPart`、`Primitives` 与 `reduceCopy`。理解这一套公共框架后，再读任意一个算子的设备端 `.h`，就能很快抓住它的数据流和算法要点。
