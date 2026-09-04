# NCCL Operator Summary and Side-by-Side Comparison

This document answers one question: what "operators" exist in NCCL, what they do, which parts are shared, and which parts differ.

Source root: the root directory of this repository.

Key source-code entry points:

- Internal operator enumeration: [nccl_common.h](src/include/nccl_common.h#L72)
- Public API declarations: [nccl.h.in](src/nccl.h.in#L517)
- Public API enqueue implementations: [collectives.cc](src/collectives.cc#L22)
- Common device-side execution framework: [common.h](src/device/common.h#L434)
- Low-level transport primitives: [primitives.h](src/device/primitives.h#L26)
- Kernel-generation algorithm matrix: [generate.py](src/device/generate.py#L20)

## 1. The full landscape of NCCL operators

"Operators" in NCCL are not a single layer; they can be divided into four categories:

1. **Core collective-communication operators**: Broadcast, Reduce, AllGather, ReduceScatter, and AllReduce.
   - These are NCCL's most important device-side kernels, concentrated in the same-named `.h` files under `src/device/`.
   - They have independent `RunWorkColl` template specializations and are compiled into many kernels by "operator + data type + reduction operation + algorithm + protocol."

2. **Point-to-point operators**: Send, Recv, and the internally unified SendRecv.
   - The public APIs are `ncclSend` / `ncclRecv`.
   - On the device, they are uniformly executed by `RunWorkBatch<ncclFuncSendRecv, ...>` in `src/device/sendrecv.h`.

3. **Composite collective operators**: AlltoAll, Gather, and Scatter.
   - They do not have dedicated device-side collective kernels; instead, they are decomposed into multiple Send/Recv tasks during task classification.
   - The decomposition logic is in [task_classify.cc](src/enqueue/task_prep/task_classify.cc#L58).

4. **One-sided / signaling operators**: PutSignal, Signal, and WaitSignal.
   - Corresponding to `ncclPutSignal`, `ncclSignal`, and `ncclWaitSignal`, they use the RMA task queue rather than traditional collective kernels.

There is also the internal optimization operator `AllGatherV`, which is an internal implementation of variable-length AllGather / batched Broadcast. It is not a public API; the device side is in [all_gather_v.h](src/device/all_gather_v.h#L16).

## 2. Semantic comparison of the core collective operators

The following table gives the most essential differences among the five core collective operators. Here `count` means "the number of elements per rank in a single contribution."

| Operator | Data flow | Reduces? | Meaning of root | Per-rank send volume | Per-rank receive volume | In-place condition |
|---|---|---|---|---|---|---|
| Broadcast | root -> all ranks | No | Data-source rank | root sends `count`; others send nothing | Every rank receives `count` | `sendbuff == recvbuff` |
| Reduce | all ranks -> root | Yes | Result-destination rank | Every rank sends `count` | Only root receives `count` | `sendbuff == recvbuff`; non-root `recvbuff` may be NULL |
| AllGather | all ranks -> all ranks | No | None, fixed 0 | Every rank sends `count` | Every rank receives `nranks * count` | `sendbuff == recvbuff + rank * count` |
| ReduceScatter | all ranks -> all ranks, but the result is scattered | Yes | None, fixed 0 | Every rank sends `nranks * count` | Every rank receives `count` | `recvbuff == sendbuff + rank * count` |
| AllReduce | all ranks -> all ranks | Yes | None, fixed 0 | Every rank sends `count` | Every rank receives `count` | `sendbuff == recvbuff` |

Semantics sources:

- API comments: [nccl.h.in](src/nccl.h.in#L517).
- Enqueue wrappers: [collectives.cc](src/collectives.cc#L153).

### 2.1 Intuitive understanding

- Broadcast is "one-to-many copy."
- Reduce is "many-to-one reduction, with the result only on root."
- AllGather is "everyone contributes one part, and everyone ends up owning everything," with no reduction.
- ReduceScatter is "first perform a global reduction, then split the result into n blocks and scatter them to the ranks."
- AllReduce is the combination of "ReduceScatter + AllGather": first reduce and scatter, then gather and broadcast, so everyone ends up with the complete result.

If AllReduce is viewed as a composite operator, then:

- Reduce is AllReduce without the broadcast phase, keeping only "reduce to root."
- Broadcast is AllReduce without the reduction phase, keeping only "spread from root."
- AllGather is pure data transfer and does not change numerical values.
- ReduceScatter is the first half of AllReduce.

## 3. Common execution framework: what all operators share

NCCL can implement so many operators efficiently because they share the same execution framework.

### 3.1 Unified host-side entry point

Every public collective operator follows a "thin wrapper" pattern:

1. `ncclXxxConfigImpl()` constructs an `ncclInfo` description object.
2. Parses `ncclCollConfig_t`.
3. Calls the unified `ncclEnqueueCheck()` to complete parameter validation, algorithm/protocol selection, task scheduling, and kernel launch.

For example, `ncclAllReduceConfigImpl` in [collectives.cc](src/collectives.cc#L234) only fills parameters into `ncclInfo` and passes them to the unified entry point. All collective operators enqueue asynchronously; a function return only means "submitted to the CUDA stream," not that communication has completed.

### 3.2 Unified device-side dispatch

The device-side main loop is `ncclKernelMain` in [common.h](src/device/common.h#L434):

1. Loads kernel parameters, comm, and channel into shared memory.
2. Loads work through `loadWorkBatchToShmem`.
3. Calls `RunWorkBatch::run()`, which iterates over multiple work items internally.
4. For collective work, calls `RunWorkColl<Fn, T, RedOp, Algo, Proto>::run()`.

Each operator's `.h` file is only responsible for specializing `RunWorkColl`; the common loop, work loading, profiler, and abort checks are all reused.

### 3.3 Channel / CBD data partitioning

All collective operators use channel parallelism, and [ncclCollCbdPart](src/include/device.h#L357) computes:

- `partOffset`: the starting element offset of the current channel in the global buffer.
- `partCount`: the number of elements handled by the current channel.
- `chunkCount`: the chunk size of the current channel.

This means the same operator does not need to change its algorithm structure when the data size changes; it only adjusts the channel count and chunk size.

### 3.4 Three transport protocols

The protocols are defined in [primitives.h](src/device/primitives.h#L37):

- **LL**: low latency; a 16-byte line contains 8 bytes of data + 8 bytes of flag, suitable for very small messages.
- **LL128**: 128-byte alignment, for medium messages.
- **SIMPLE**: direct load/store, suitable for large messages.

Protocols are template parameters, so the same algorithm code can be reused parametrically.

### 3.5 Fan and Primitives

`FanAsymmetric` / `FanSymmetric` describe an operator's "number of receive neighbors / send neighbors" at the current node:

- Ring is usually `FanSymmetric<1>`: one predecessor + one successor.
- Tree upward is `FanAsymmetric<3, 1>`: up to 3 children and 1 parent.
- Tree downward is `FanAsymmetric<1, 3>`: 1 parent and up to 3 children.

All operators ultimately use `Primitives` methods such as `send` / `recv` / `directSend` / `recvReduceSend` to move data and perform reductions.

### 3.6 Reductions uniformly use reduceCopy

All operators that perform reduction (Reduce, AllReduce, ReduceScatter) call `reduceCopy`. `reduceCopy` is responsible for:

- Merging data from multiple sources.
- `preOp`: local scaling before input. For example, `ncclAvg` on floating-point types maps to `PreMulSum`, multiplying by `1/nranks` first.
- `postOp`: post-processing the final result. For example, `SumPostDiv` divides by `nranks` for integer types.

This is the shared numerical-semantics foundation of all reduction operators. See [reduce_kernel.h](src/device/reduce_kernel.h#L204).

## 4. Difference 1: data flow and root

The most essential difference among the five core operators is the direction of data flow.

| Operator | Start | End | Action of intermediate nodes |
|---|---|---|---|
| Broadcast | root | all nodes | Receive and forward, without reduction |
| Reduce | all nodes | root | Receive and reduce while moving toward root |
| AllGather | all nodes | all nodes | Receive and forward, without reduction |
| ReduceScatter | all nodes | all nodes, but each node receives only one block | Receive and reduce, and the result is split into blocks |
| AllReduce | all nodes | all nodes | First reduce and spread, then gather and broadcast |

Only Broadcast, Reduce, Gather, and Scatter have a real `root` parameter. AllGather, ReduceScatter, and AllReduce do not; `ncclInfo.root` is fixed to 0 or simply ignored.

## 5. Difference 2: whether reduction is performed

Based on whether numerical values are changed, operators can be divided into two groups:

**Non-reducing data-movement operators:**

- Broadcast
- AllGather
- AlltoAll
- Gather
- Scatter
- Send / Recv

When these operators construct `ncclInfo`, `op` is only a placeholder using `ncclSum`; the device side only performs copy/send/recv and does not invoke real reduction logic.

**Reducing operators:**

- Reduce
- ReduceScatter
- AllReduce

They must carry a real `ncclRedOp_t op` and pass through [enqueue.cc](src/enqueue/enqueue.cc#L2480) to map public `ncclRedOp_t` to device-side `ncclDevRedOp_t`:

- `ncclSum` -> `ncclDevSum`
- `ncclProd` -> `ncclDevProd`
- `ncclMin` / `ncclMax` -> `ncclDevMinMax`
- `ncclAvg` -> `ncclDevPreMulSum` for floating point, `ncclDevSumPostDiv` for integers
- User-defined PreMulSum -> `ncclDevPreMulSum`

## 6. Difference 3: algorithm support matrix

Algorithm support is determined by `algos_of_coll` in [generate.py](src/device/generate.py#L74):

| Operator | Ring | Tree | CollNet Direct | CollNet Chain | NVLS | NVLS_TREE | PAT |
|---|---|---|---|---|---|---|---|
| Broadcast | ✅ | — | — | — | — | — | — |
| Reduce | ✅ | — | — | — | — | — | — |
| AllGather | ✅ | — | ✅ | — | ✅ | — | ✅ |
| ReduceScatter | ✅ | — | ✅ | — | ✅ | — | ✅ |
| AllReduce | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| Send / Recv | Dedicated P2P entry | — | — | — | — | — | — |

Notes:

- The table above reflects the **device-side kernel matrix of this repository's current source code**: Broadcast and Reduce generate only Ring kernels and have no independent Tree kernels.
- This is not the same as saying "in all official NCCL versions only AllReduce supports Tree." In classical NCCL versions and many references, Tree is also used for Broadcast / Reduce; this repository fixes the algorithm list of these two operators to `["RING"]` through `algos_of_coll` in [generate.py](src/device/generate.py#L104), and hard-disables the Tree model for non-AllReduce operations in [tree.cc](src/tuning/tree.cc#L22).
- AllReduce has the richest algorithm set and is the operator with the largest tuning space in NCCL.
- AllGather / ReduceScatter gain additional PAT and NVLS choices in NVLink + NVSwitch scenarios.
- The CollNet family depends on collective-network hardware such as IB SHARP.

## 7. Difference 4: protocol support

Algorithm-protocol combinations are constrained by `required_cuda` in [generate.py](src/device/generate.py#L83):

- In this repository's current source code, RING supports LL, LL128, and SIMPLE; TREE exists only for AllReduce and also supports LL, LL128, and SIMPLE.
- PAT, NVLS, NVLS_TREE, COLLNET_DIRECT, and COLLNET_CHAIN support only SIMPLE.
- The reason is that these hardware-accelerated algorithms target large messages / high bandwidth, so low-latency protocols for small messages are not worthwhile.

Therefore:

| Operator | LL | LL128 | SIMPLE |
|---|---|---|---|
| Broadcast | ✅ | ✅ | ✅ |
| Reduce | ✅ | ✅ | ✅ |
| AllGather | ✅ | ✅ | ✅ |
| ReduceScatter | ✅ | ✅ | ✅ |
| AllReduce | Ring/Tree fully supported; hardware algorithms SIMPLE only | Ring/Tree fully supported | All algorithms |
| Send/Recv | Optional LL | — | ✅ |

## 8. Difference 5: implementation files

| Operator | Host API | Device kernel |
|---|---|---|
| Broadcast | [collectives.cc:271](src/collectives.cc#L271) | [broadcast.h](src/device/broadcast.h#L23) |
| Reduce | [collectives.cc:354](src/collectives.cc#L354) | [reduce.h](src/device/reduce.h#L23) |
| AllGather | [collectives.cc:153](src/collectives.cc#L153) | [all_gather.h](src/device/all_gather.h#L20) |
| ReduceScatter | [collectives.cc:393](src/collectives.cc#L393) | [reduce_scatter.h](src/device/reduce_scatter.h#L20) |
| AllReduce | [collectives.cc:234](src/collectives.cc#L234) | [all_reduce.h](src/device/all_reduce.h#L25) |
| Send / Recv | [collectives.cc:469](src/collectives.cc#L469) | [sendrecv.h](src/device/sendrecv.h#L18) |

## 9. Different implementation approaches for composite and one-sided operators

### 9.1 AlltoAll, Gather, Scatter decomposed into P2P

These three operators have no dedicated collective kernels. `task_classify.cc` converts them into a set of Send/Recv operations:

- AlltoAll: for each `r`, send `sendbuff + r*count` and receive `recvbuff + r*count`.
- Gather: all ranks send to root; root receives in rank order.
- Scatter: root sends in rank order; all ranks receive from root.

See [task_classify.cc](src/enqueue/task_prep/task_classify.cc#L58).

### 9.2 PutSignal / Signal / WaitSignal use RMA

These three are one-sided / signaling operations and do not participate in traditional collective-kernel scheduling:

- `ncclPutSignal`: writes data into a peer-registered window and carries a signal.
- `ncclSignal`: sends only a signal, without data.
- `ncclWaitSignal`: waits for a signal on the specified peer/context.

They are classified into `rmaTaskQueue` in `ncclTaskClassification`; see [task_classify.cc](src/enqueue/task_prep/task_classify.cc#L130).

## 10. Quick reference for usage scenarios

| Scenario | Recommended operator / algorithm tendency |
|---|---|
| Gradient synchronization where every GPU needs the full gradient | AllReduce |
| Parameter server, gradients aggregated to a single point | Reduce |
| Distributing a model / hyperparameters from a master node | Broadcast |
| Multi-GPU parallel forward pass, splitting tokens and then collecting the complete sequence | AllGather |
| Splitting a global reduction result by rank, with each GPU processing only one block later | ReduceScatter |
| Each GPU sends different data to every other GPU for transpose / permutation | AlltoAll |
| Collecting data from all GPUs to one GPU | Gather |
| Distributing different data from one GPU to each GPU | Scatter |
| Precisely controlling individual send/receive order | Send / Recv, with ncclGroupStart/End when needed |

General tendencies by message size:

- Very small messages: prioritize low latency, use LL, and bias algorithm choice toward startup overhead.
- Medium messages: LL128.
- Large messages: SIMPLE, bandwidth first.
- Single machine with multiple GPUs: Ring/Tree are general-purpose; with NVSwitch, consider NVLS, NVLS_TREE, and PAT.
- Multiple machines: Ring, CollNet Direct/Chain; with collective-network hardware such as SHARP, CollNet is often better.

## 11. One-sentence summary

The "sameness" of NCCL operators lies in unified host enqueue, unified channel/CBD partitioning, unified protocols, and Primitives/reduceCopy. The "differences" lie in data-flow direction, whether reduction is performed, whether there is a root, whether the result is replicated or scattered, and which algorithms and hardware are available. Once you understand "who sends, who receives, who reduces, and where the result goes," you can clearly distinguish the five core collective operators and their composite variants.
