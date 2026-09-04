# Detailed Source-Code Analysis of NCCL Operators

This document analyzes NCCL communication operators one by one: their purpose, public APIs, host enqueue paths, device-side kernel algorithms, protocol/algorithm variants, in-place conditions, and applicable scenarios.

Source root: the root directory of this repository.

## 0. Reading guide and common foundation

Before diving into each operator, first clarify NCCL's common execution model.

### 0.1 Unified host-side enqueue

All collective operators follow the same three-layer structure:

```text
ncclXxx() -> ncclXxxConfigImpl() -> ncclEnqueueCheck()
```

- `ncclXxx()`: public API, adds NVTX instrumentation, with `config == NULL`.
- `ncclXxxConfigImpl()`: constructs `ncclInfo` and parses `ncclCollConfig_t`.
- `ncclEnqueueCheck()`: parameter validation, algorithm selection, protocol selection, task classification, and kernel launch.

The entry implementations are in [collectives.cc](src/collectives.cc#L22). These functions only enqueue asynchronously; the actual communication is completed on the CUDA stream.

### 0.2 Unified device-side dispatch

The device-side main loop is [ncclKernelMain](src/device/common.h#L434):

1. Copies kernel parameters, comm, and channel into shared memory `ncclShmem`.
2. Uses `loadWorkBatchToShmem` to load work.
3. Calls `RunWorkBatch::run()` in a loop.
4. For collective work, calls `RunWorkColl<Fn, T, RedOp, Algo, Proto>::run()` item by item.

Each operator's device-side `.h` file only provides a specialization of `RunWorkColl` or `RunWorkBatch`.

### 0.3 Channels and chunks

Every collective operator calls [ncclCollCbdPart](src/include/device.h#L357) and obtains:

- `gridOffset` / `partOffset`: the starting position of the data handled by the current channel.
- `channelCount` / `partCount`: the amount of data handled by the current channel.
- `chunkCount`: the chunk stride of the current channel.

When the data volume is large, channels and chunks together provide parallelism and pipeline depth.

### 0.4 Protocols and primitives

Protocol classes:

- `ProtoLL`: 16-byte line = 8 bytes data + 8 bytes flag.
- `ProtoLL128`: 128-byte line, with data and flag interleaved.
- `ProtoSimple<SlicePerChunk, StepPerSlice, Unroll>`: direct load/store.

See [primitives.h](src/device/primitives.h#L37).

`Primitives` provides low-level actions such as send/recv/directSend/directRecv/recvReduceSend. Reduction is uniformly handled by `reduceCopy`.

## 1. Broadcast

### 1.1 Purpose

Broadcast copies a block of data on the root to all ranks. Typical scenarios:

- Broadcasting model parameters, random seeds, or configuration tensors from a master node.
- The "broadcast the result back" phase of AllReduce is essentially a Broadcast.

### 1.2 Public API and semantics

```c
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream);
```

- Only the root's `sendbuff` is meaningful.
- Every rank's `recvbuff` receives the root's data.
- The in-place case occurs when `sendbuff == recvbuff`.
- The deprecated `ncclBcast` is equivalent to in-place Broadcast.

The declaration is in [nccl.h.in](src/nccl.h.in#L559).

### 1.3 Host enqueue

[ncclBroadcastConfigImpl](src/collectives.cc#L271) constructs:

```c
struct ncclInfo info = {ncclFuncBroadcast, "Broadcast",
                        sendbuff, recvbuff, count, datatype, ncclSum, root,
                        comm, stream, BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS};
```

Broadcast does not reduce, so `op` uses `ncclSum` as a placeholder; `root` is the data-source rank.

### 1.4 Device-side algorithm: Ring

In the current source tree, the Broadcast device kernel only generates the Ring algorithm, as shown in [broadcast.h](src/device/broadcast.h#L23). This describes the device-side kernel matrix of this particular checkout; classical NCCL versions may also have a Tree implementation for Broadcast, but in this repository [generate.py](src/device/generate.py#L104) keeps only `["RING"]` for Broadcast.

The core logic of `runRing` is:

```text
For each chunk:
    if rank == root:
        directSend if in-place
        directCopySend if not in-place
    else if nextRank == root:
        directRecv               // Last node on the ring; receive only, no forward
    else:
        directRecvCopyDirectSend // Intermediate node; receive and forward
```

Explanation:

- root is the data source and first sends data to the next rank.
- Intermediate ranks receive the previous hop's data and immediately forward it to the next hop.
- The node where `nextRank == root` is the predecessor of root, meaning the data has already traveled around the ring once, so it only needs to receive, without forwarding again.

### 1.5 Protocol specializations

[broadcast.h](src/device/broadcast.h#L85) provides three specializations:

- `NCCL_PROTO_SIMPLE`
- `NCCL_PROTO_LL`
- `NCCL_PROTO_LL128`

All three protocols share the same `runRing<T, RedOp, Proto>`; the only difference is the template parameter `Proto`.

### 1.6 Network offload and local copy

When `isNetOffload == true`, only one warp drives network communication, while the remaining warps handle the local copy for the non-in-place root case. The implementation is in [broadcast.h](src/device/broadcast.h#L30).

### 1.7 Applicable scenarios

- Broadcasting a single message, especially when the message is small and latency-sensitive.
- Parameters or metadata that must be shared by multiple GPUs from a root.
- For large messages, use SIMPLE to improve bandwidth; for small messages, use LL to reduce latency.

## 2. Reduce

### 2.1 Purpose

Reduce combines the `count` elements from all ranks using `op` and places the result only in the root's `recvbuff`. Typical scenarios:

- Parameter-server mode, where gradients converge to a single master.
- Storing global statistics on a single node.

### 2.2 Public API and semantics

```c
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream);
```

- Non-root `recvbuff` may be NULL.
- The in-place case occurs when `sendbuff == recvbuff`.

The declaration is in [nccl.h.in](src/nccl.h.in#L530).

### 2.3 Host enqueue

[ncclReduceConfigImpl](src/collectives.cc#L354) constructs `ncclInfo`, where `op` is the real reduction operation and `root` is the rank where the result lands.

### 2.4 Device-side algorithm: Ring

In the current source tree, the Reduce device kernel also only generates the Ring algorithm, as shown in [reduce.h](src/device/reduce.h#L23). Again, this is specific to this checkout: classical NCCL versions may have a Tree implementation for Reduce, but this repository's [generate.py](src/device/generate.py#L104) keeps only `["RING"]` for Reduce.

The core logic is:

```text
For each chunk:
    if prevRank == root:
        send()                     // Node after root: starting point; send local data only
    else if rank == root:
        recvReduceCopy(..., postOp=true) // Endpoint: receive and reduce to final result
    else:
        recvReduceSend()           // Intermediate node: receive + reduce + continue forwarding
```

Explanation:

- Data starts from the node after root and moves around the ring toward root while being reduced.
- At each node, that node's local data is merged in.
- When it reaches root, the complete result is written back to root's output buffer.

### 2.5 Protocol specializations

[reduce.h](src/device/reduce.h#L72) provides three specializations: SIMPLE, LL, and LL128.

### 2.6 Applicable scenarios

- Many-to-one reduction where the result only needs to exist in one place.
- Compared with AllReduce, Reduce omits the broadcast phase, making it suitable for parameter servers or log aggregation.
- If every rank needs the full result later, AllReduce should be used directly.

## 3. AllGather

### 3.1 Purpose

Each rank contributes `sendcount` elements, and every rank ends up with `nranks * sendcount` elements. Rank i's data is located at `recvbuff + i * sendcount`.

Typical scenarios:

- Concatenating sequences or tensors processed by multiple GPUs in parallel.
- Token collection before all-to-all in MoE routing.
- Collecting per-GPU batch slices in data parallelism.

### 3.2 Public API and semantics

```c
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                           ncclDataType_t datatype, ncclComm_t comm,
                           cudaStream_t stream);
```

- The receive buffer must contain at least `nranks * sendcount` elements.
- In-place condition: `sendbuff == recvbuff + rank * sendcount`.

The declaration is in [nccl.h.in](src/nccl.h.in#L606).

### 3.3 Host enqueue

[ncclAllGatherConfigImpl](src/collectives.cc#L153) constructs `ncclInfo`; there is no reduction, so `op` uses `ncclSum` as a placeholder.

### 3.4 Device-side algorithms

AllGather supports four algorithms, as shown in [all_gather.h](src/device/all_gather.h#L20):

- Ring
- PAT
- NVLS
- CollNet Direct

#### 3.4.1 Ring AllGather

The `runRing` flow is:

```text
For each chunk:
    step 0: send the local block to the next rank
    middle nranks-2 steps: receive the previous hop's block and forward it to the next hop
    last 1 step: receive the final block and store it in recvbuff
```

After `nranks - 1` hops, all ranks have the complete data.

The implementation is in [all_gather.h](src/device/all_gather.h#L20).

The three Ring protocol specializations are in [all_gather.h](src/device/all_gather.h#L102).

#### 3.4.2 PAT AllGather

PAT is Parallel Aggregated Tree. On the device, a dedicated "algorithm computation thread" generates `ncclPatStep`, and worker threads move data according to those steps.

The implementation is in [all_gather.h](src/device/all_gather.h#L131).

#### 3.4.3 NVLS AllGather

It uses the scatter/gather capability of NVSwitch to split data by rail and complete the aggregation.

The implementation is in [all_gather.h](src/device/all_gather.h#L290).

#### 3.4.4 CollNet Direct AllGather

It is a three-stage pipeline:

1. Send to the collective network.
2. The network returns and broadcasts.
3. Receive the broadcast and store it.

The implementation is in [all_gather.h](src/device/all_gather.h#L520).

### 3.5 Applicable scenarios

- When the data volume is large and the topology supports NVSwitch / SHARP, prefer NVLS / CollNet / PAT.
- On ordinary topologies, use Ring, which is simple and pipeline-friendly.
- This is pure data movement without reduction, so bandwidth utilization is usually the main optimization target.

## 4. ReduceScatter

### 4.1 Purpose

ReduceScatter first performs a global reduction on the `nranks * recvcount` elements from all ranks, then splits the result into n blocks. Rank i receives only block i, consisting of `recvcount` elements.

Typical scenarios:

- Large-scale parameter/gradient sharding: perform global reduction first, then place parameter shards on different GPUs.
- When each GPU subsequently processes only its own result block, ReduceScatter has one fewer collection step than AllReduce.

### 4.2 Public API and semantics

```c
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream);
```

- The parameter is `recvcount`.
- `sendbuff` must contain at least `nranks * recvcount` elements.
- In-place condition: `recvbuff == sendbuff + rank * recvcount`.

The declaration is in [nccl.h.in](src/nccl.h.in#L589).

### 4.3 Host enqueue

[ncclReduceScatterConfigImpl](src/collectives.cc#L393) constructs `ncclInfo`, with the `count` field set to `recvcount`.

Note that `ncclFuncSendCount` returns `nRanks * count` for ReduceScatter, which is used to calculate the actual send buffer size.

### 4.4 Device-side algorithms

ReduceScatter supports four algorithms, as shown in [reduce_scatter.h](src/device/reduce_scatter.h#L20):

- Ring
- PAT
- NVLS
- CollNet Direct

#### 4.4.1 Ring ReduceScatter

The `runRing` flow is:

```text
For each chunk:
    step 0: send the corresponding local block
    middle nranks-2 steps: receive the previous hop's block + reduce + continue sending
    last 1 step: receive + reduce, and write the result to local recvbuff
```

The implementation is in [reduce_scatter.h](src/device/reduce_scatter.h#L20).

#### 4.4.2 PAT ReduceScatter

Similar to AllGather, PAT ReduceScatter uses an algorithm computation thread to schedule worker threads, generating reduction steps according to `PatRSAlgorithm`.

The implementation is in [reduce_scatter.h](src/device/reduce_scatter.h#L95).

#### 4.4.3 NVLS ReduceScatter

It uses the scatter + reduce capability of NVSwitch to avoid hop-by-hop reduction between GPUs.

The implementation is in [reduce_scatter.h](src/device/reduce_scatter.h#L247).

#### 4.4.4 CollNet Direct ReduceScatter

It has three stages:

1. Scatter the input.
2. Reduce from local peers, then send to the collective network.
3. Receive the final block from the network.

The implementation is in [reduce_scatter.h](src/device/reduce_scatter.h#L460).

### 4.5 Applicable scenarios

- After reduction, the result is split by rank, and no complete result is needed afterward.
- Very common in hybrid data-parallel / tensor-parallel / pipeline-parallel strategies.
- With NVSwitch or SHARP, hardware reduction can significantly reduce traffic.

## 5. AllReduce

### 5.1 Purpose

AllReduce performs a global reduction over the `count` elements from all ranks, and every rank obtains exactly the same complete result. It is the core operator for gradient synchronization in distributed training.

### 5.2 Public API and semantics

```c
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream);
```

- In-place condition: `sendbuff == recvbuff`.
- There is no root.

The declaration is in [nccl.h.in](src/nccl.h.in#L575).

### 5.3 Host enqueue

[ncclAllReduceConfigImpl](src/collectives.cc#L234) constructs `ncclInfo`. AllReduce has the largest algorithm space among collective operators, and the tuning phase selects algorithms and protocols based on message size, topology, and hardware.

### 5.4 Device-side algorithms

AllReduce supports six algorithms, as shown in [all_reduce.h](src/device/all_reduce.h#L25):

- Ring
- Tree
- CollNet Direct
- CollNet Chain
- NVLS
- NVLS_TREE

#### 5.4.1 Ring AllReduce

Ring AllReduce is NCCL's most classic implementation:

```text
Phase 1: scatter-reduce
    step 0: send the local chunk to the next rank
    middle nranks-2 steps: receive + reduce + send
    last 1 step: receive + reduce, obtaining the final block

Phase 2: all-gather
    middle nranks-2 steps: receive + copy + send
    last 1 step: receive the final block
```

After `2 * (nranks - 1)` ring steps, every rank has the complete reduced result.

The implementation is in [all_reduce.h](src/device/all_reduce.h#L25).

The three Ring protocol specializations are in [all_reduce.h](src/device/all_reduce.h#L296) and the LL/LL128 specializations at the end of the file.

#### 5.4.2 Tree AllReduce

The Tree algorithm organizes ranks into a logical tree:

- Upward Reduce: leaves send, intermediate nodes receive, reduce, and upload, and the root obtains the complete result.
- Downward Broadcast: the root broadcasts the result down the tree, and leaves receive it.

NCCL has two Tree implementations:

- `runTreeUpDown`: strictly separates upward and downward phases, first Reduce and then Broadcast.
- `runTreeSplit`: splits threads into two groups, one performing upward reduction and one performing downward broadcast, running in parallel to improve large-message bandwidth.

The implementations are in [all_reduce.h](src/device/all_reduce.h#L120) and [all_reduce.h](src/device/all_reduce.h#L193).

The Tree + SIMPLE entry is in [all_reduce.h](src/device/all_reduce.h#L306); Tree + LL/LL128 are at the end of the file.

#### 5.4.3 CollNet Direct / CollNet Chain

- CollNet Direct: uses collective-network hardware and completes through a four-stage pipeline of Scatter / Gather / Reduce / Broadcast, as shown in [all_reduce.h](src/device/all_reduce.h#L319).
- CollNet Chain: chain-style upward reduction and downward broadcast, as shown in [all_reduce.h](src/device/all_reduce.h#L712).

Both rely on collective network hardware such as InfiniBand SHARP.

#### 5.4.4 NVLS and NVLS_TREE

- NVLS: uses NVSwitch hardware reduction to perform scatter/gather/reduce/broadcast, as shown in [all_reduce.h](src/device/all_reduce.h#L462).
- NVLS_TREE: NVLS handles reduction, while the tree handles multi-node distribution/aggregation, as shown in [all_reduce.h](src/device/all_reduce.h#L600).

### 5.5 Applicable scenarios

- The preferred operator for data-parallel gradient synchronization.
- Small messages: Ring/Tree + LL, to reduce startup latency.
- Medium messages: Ring/Tree + LL128.
- Large messages: Ring/Tree + SIMPLE, or hardware acceleration such as NVLS / CollNet.
- Single machine with multiple GPUs and NVSwitch: NVLS and NVLS_TREE are often better than Ring/Tree.
- Multiple machines with SHARP: CollNet Direct/Chain are often better than pure Ring.

## 6. Send / Recv

### 6.1 Purpose

Send / Recv are point-to-point communication primitives:

```c
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);
```

Both sides must use the same `count` and `datatype` in a matched pair. They are GPU-blocking operations; when multiple Send/Recv calls need to progress concurrently, they should be placed inside `ncclGroupStart()` / `ncclGroupEnd()`.

### 6.2 Host enqueue

In [collectives.cc](src/collectives.cc#L469):

- `ncclSend` constructs an `ncclInfo` of type `ncclFuncSend`, reusing the `root` field as the peer.
- `ncclRecv` constructs an `ncclInfo` of type `ncclFuncRecv`, also reusing `root` as the peer.

### 6.3 Device-side implementation

The device side uniformly uses the following from [sendrecv.h](src/device/sendrecv.h#L18):

```c
RunWorkBatch<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE>
```

Implementation characteristics:

- `T` must be a 1-byte type because P2P moves data by bytes.
- One work item can contain both send and receive; warps are assigned to execute send, recv, or local copy.
- LL or SIMPLE protocol is selected according to `sendProtoLL` / `recvProtoLL`.
- Larger chunks are allowed when all links are NVLink or the network is registered.

### 6.4 Applicable scenarios

- Custom communication patterns, such as sparse communication or neighbor exchange.
- The low-level implementation of composite operators.
- When precise control over send/receive order is needed, used together with group semantics.

## 7. AlltoAll, Gather, Scatter

### 7.1 Purpose

- AlltoAll: each rank sends different data to every other rank and receives different data from every rank.
- Gather: all ranks gather data to the root.
- Scatter: the root distributes different data to each rank.

The public APIs are in [collectives.cc](src/collectives.cc#L197), [collectives.cc](src/collectives.cc#L317), and [collectives.cc](src/collectives.cc#L431).

### 7.2 Implementation: decomposed into P2P

These three operators do not have dedicated collective kernels. During task classification they are decomposed into Send/Recv:

[task_classify.cc](src/enqueue/task_prep/task_classify.cc#L58):

- AlltoAll: for each rank r, generate one Send and one Recv.
- Gather: each non-root rank issues one Send; root generates n Recvs.
- Scatter: root generates n Sends; each rank generates one Recv.

This means their performance depends on P2P scheduling rather than dedicated collective algorithms.

### 7.3 Applicable scenarios

- AlltoAll: MoE token distribution, matrix transpose, FFT, and other full-exchange communication.
- Gather: logs, debug information, or checkpoint data collected to a single GPU.
- Scatter: a master distributes different slices to workers.

## 8. PutSignal, Signal, WaitSignal

### 8.1 Purpose

These are NCCL's one-sided / signaling operators:

- `ncclPutSignal`: writes local data into a peer-registered window and carries a signal.
- `ncclSignal`: sends only a signal, without data.
- `ncclWaitSignal`: waits for a signal on the specified peer / sigIdx / ctx.

The public APIs are in [collectives.cc](src/collectives.cc#L521), [collectives.cc](src/collectives.cc#L554), and [collectives.cc](src/collectives.cc#L584).

### 8.2 Implementation: RMA queue

These operators are classified into `rmaTaskQueue` by `ncclTaskClassification` and do not enter the traditional collective-kernel scheduler. See [task_classify.cc](src/enqueue/task_prep/task_classify.cc#L130).

### 8.3 Applicable scenarios

- One-sided communication: the target process does not need to explicitly participate in data transfer.
- Fine-grained synchronization: use Signal/WaitSignal to represent dependencies.
- PutSignal is suitable for remote writes with a synchronization marker, replacing the two operations "Put + Signal."

## 9. Internal operator AllGatherV

AllGatherV is a variable-length AllGather / batched Broadcast used internally by NCCL.

The device side is in [all_gather_v.h](src/device/all_gather_v.h#L16):

- It uses `RunWorkBatch`, with one kernel handling multiple work items.
- Each work item has `bytes`, `bytes_done`, `chunkSize`, and `ringDepth`.
- A rank with `ringDepth == 0` is the data source and sends local data; intermediate ranks receive and forward; a rank with `ringDepth == nranks - 1` only receives.

It is mainly used for batched optimization in certain Broadcast paths and is not directly exposed as a public API.

## 10. Supplementary notes on reduction operators

In addition to communication operators, NCCL has the concept of "reduction operators."

Public reduction operators are in [nccl.h.in](src/nccl.h.in#L448):

```c
ncclSum, ncclProd, ncclMax, ncclMin, ncclAvg
```

Device-side reduction operators are in [device.h](src/include/device.h#L60):

```c
ncclDevSum, ncclDevProd, ncclDevMinMax, ncclDevPreMulSum, ncclDevSumPostDiv
```

The mapping is in [enqueue.cc](src/enqueue/enqueue.cc#L2480):

- `ncclSum` -> `ncclDevSum`
- `ncclProd` -> `ncclDevProd`
- `ncclMin` / `ncclMax` -> `ncclDevMinMax`
- `ncclAvg` for floating-point types -> `ncclDevPreMulSum` (multiply by 1/nranks first, then sum)
- `ncclAvg` for integer types -> `ncclDevSumPostDiv` (sum first, then divide by nranks)

These device-side reduction operators are the numerical foundation for Reduce, ReduceScatter, and AllReduce; data-movement operators do not use them.

## 11. Summary

NCCL's operator system can be remembered as:

```text
Five core collective operators:
  Broadcast      = one-to-many copy
  Reduce         = many-to-one reduction
  AllGather      = everyone contributes, everyone owns, no reduction
  ReduceScatter  = reduce first, then split into blocks
  AllReduce      = ReduceScatter + AllGather

P2P:
  Send / Recv, and AlltoAll / Gather / Scatter composed from them

One-sided signaling:
  PutSignal / Signal / WaitSignal
```

From a source-code perspective, the differences among the five core collective operators are mainly reflected in the loop logic inside their `RunWorkColl` specializations; the commonalities are reflected in the unified `ncclInfo`, `ncclEnqueueCheck`, `ncclKernelMain`, `ncclCollCbdPart`, `Primitives`, and `reduceCopy`. After understanding this common framework, reading any operator's device-side `.h` file makes it easy to quickly identify its data flow and algorithm essentials.
