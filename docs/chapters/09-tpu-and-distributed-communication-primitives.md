# TPU 与分布式通信原语

之前的内容都是在单张 GPU 上展开的，但实际 LLM 时代，分布式训练/推理已经成为绕不开的话题。本节以 TPU 为切入点，介绍 TPU 在分布式场景下常用的分布式通信原语实现，以及其应用。其思路和实现原理同样适用于 GPU。

## TPU 的硬件模型

TPU 通常的物理拓扑是 mesh。比如 2*2 的mesh，组成一个ring。

TPU 每条链路单向带宽大约为 92 GB/s，单 hop 延迟大约 1 微秒。

每个 TPU v5p 设备有两个 core，每个 core 是单线程的 VLIW 风格处理器，但 SIMD 很宽：对 fp32 而言，原生处理宽度是 1024 元素；TPU通常把它组织成一个 `(8, 128)` 的 tile。 Pallas 对齐、切片、RDMA 连续性基本都围着这个最小高效单位在转。

TPU 的内存层级和 GPU 很不一样。TPU 最重要的是 VMEM，也就是每个 core 拥有的 64 MiB 高带宽 SRAM scratchpad。与 H100 相比，这个 scratchpad 容量非常大。除此之外，每个 device 还有 95 GB 的 HBM2e；但和 GPU 复杂的 cache hierarchy 相比，TPU 更像“HBM + 大片上 scratchpad”的体系。很多实现策略，本质上都是在问：哪些数据值得常驻 VMEM，哪些访问必须走 RDMA，哪些布局才能让编译器和 ICI 都满意。

## TPU Pallas

Pallas 是一个很低层、很贴近硬件的系统，但它采用 tracing-based compilation model，所以它只是“发射指令的模板”，不是直接在 TPU 上跑的动态程序。

我觉得最值得记住的是下面四句话。

1. **Python control flow 不是 kernel control flow。** 在 traced kernel 里写 `if`、`while`，本质发生在 trace time，而不是 runtime。只有用 `pl.when`、`lax.cond`、`lax.fori_loop` 之类 traced 控制流，才是真正把分支或循环发到 TPU 上执行。
2. **Python print 不是 kernel print。** 想看 runtime 的动态值，必须用 `jax.debug.print`，而且还要打开 starter code 里的 debug 开关。
3. **性能只取决于最终发射出的指令序列。** 你在 Python 里如何组织代码、是否用了字典或 helper function，并不直接决定 runtime 性能；真正决定性能的是 tracing 之后的底层指令。
4. **Array ref 和 array value 要分清。** ref 更像带 shape 信息的指针，value 才是被临时放进向量寄存器里的数值。RDMA 操作的是 ref，数学运算操作的是 value。

## TPU RDMA

Pallas 会保证各设备上 kernel address space 布局一致，所以“我本地这个 buffer 的 ref”也可以用来指向远端设备上对应位置的 buffer。

RDMA 是异步的，因此必须配套 semaphore 和 wait 语义来管理生命周期。

示例 TPU RDMA 代码：

```python
import time

import jax
import numpy as np
from jax import lax, numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec

AXIS_NAME = "i"
N_DEVICES = 4

ENABLE_DEBUG = False


def pallas_get_my_device_id():
    return lax.axis_index(AXIS_NAME)


def pallas_rdma_start(*, src_ref, dst_ref, dst_device_id, src_send_sem, dst_recv_sem):
    pltpu.make_async_remote_copy(
        src_ref=src_ref,
        dst_ref=dst_ref,
        send_sem=src_send_sem,
        recv_sem=dst_recv_sem,
        device_id=dst_device_id,
        device_id_type=pltpu.DeviceIdType.LOGICAL,
    ).start()


def pallas_rdma_wait_send(*, src_ref, src_send_sem):
    pltpu.make_async_remote_copy(
        src_ref=src_ref,
        dst_ref=src_ref,  # ignored by 'wait_send'
        send_sem=src_send_sem,
        recv_sem=src_send_sem,  # ignored by 'wait_send'
        device_id=0,  # ignored by 'wait_send'
        device_id_type=pltpu.DeviceIdType.LOGICAL,
    ).wait_send()


def pallas_rdma_wait_recv(*, dst_ref, dst_recv_sem):
    pltpu.make_async_remote_copy(
        src_ref=dst_ref,  # ignored by 'wait_recv'
        dst_ref=dst_ref,
        send_sem=dst_recv_sem,  # ignored by 'wait_recv'
        recv_sem=dst_recv_sem,
        device_id=0,  # ignored by 'wait_recv'
        device_id_type=pltpu.DeviceIdType.LOGICAL,
    ).wait_recv()


def exchange_with_neighbor_pallas_scratch_specs(x):
    return {
        "send_sem": pltpu.SemaphoreType.DMA,
        "recv_sem": pltpu.SemaphoreType.DMA,
    }


def exchange_with_neighbor_pallas_kernel(x_ref, out_ref, scratch_refs):
    my_device_id = pallas_get_my_device_id()
    paired_device_id = my_device_id + 1 - 2 * lax.rem(my_device_id, 2)

    send_sem = scratch_refs["send_sem"]
    recv_sem = scratch_refs["recv_sem"]

    pallas_rdma_start(
        src_ref=x_ref,
        dst_ref=out_ref,
        dst_device_id=paired_device_id,
        src_send_sem=send_sem,
        dst_recv_sem=recv_sem,
    )
    pallas_rdma_wait_send(src_ref=x_ref, src_send_sem=send_sem)
    pallas_rdma_wait_recv(dst_ref=out_ref, dst_recv_sem=recv_sem)

```

## reduce-scatter 与 all-gather 

- reduce-scatter：“先全局求和，再把结果按 device 切开”；
- all-gather：“先把各 device 的 shard 拼起来，再在每个 device 上都放一份完整结果”。

如果只说语义，这两个 collective 很简单；真正的难点是怎样在 ring 上把它们写成高吞吐的 RDMA schedule。对于 4 个设备的 ring，实现思路可以概括成下面两句：

1. **all-gather**：每个设备先把自己的 shard 写到输出中对应的位置，然后不断把“已经拥有的一块”发给下一个设备，同时从上一个设备收一块，直到拼齐完整结果。
2. **reduce-scatter**：每个设备先把某一块发出去、收回来某一块 partial，然后把收到的块和自己本地对应块相加，再把新的 partial sum 继续往 ring 里传，最后留下属于自己的那一块。

如果把 ring 想成一个流动的管道，那么 all-gather 传播的是“原始 chunk”，reduce-scatter 传播的是“不断累加的 partial chunk”。两者看上去很像，但一个的 payload 是原始值，一个的 payload 是归约中的中间值。

参考reduce-scatter 与 all-gather实现：

```python
def reduce_scatter_pallas_scratch_specs(x):
    shard_shape = (x.shape[0] // N_DEVICES, x.shape[1], x.shape[2])
    return {
        "carry_buf": pltpu.VMEM(shape=shard_shape, dtype=x.dtype),
        "recv_buf": pltpu.VMEM(shape=shard_shape, dtype=x.dtype),
        "send_sem": pltpu.SemaphoreType.DMA,
        "recv_sem": pltpu.SemaphoreType.DMA,
    }


def reduce_scatter_pallas_kernel(x_ref, out_ref, scratch_refs):
    my_device_id = pallas_get_my_device_id()
    next_device_id = lax.rem(my_device_id + 1, N_DEVICES)
    shard_len = out_ref.shape[0]

    carry_buf = scratch_refs["carry_buf"]
    recv_buf = scratch_refs["recv_buf"]
    send_sem = scratch_refs["send_sem"]
    recv_sem = scratch_refs["recv_sem"]

    def x_chunk_start(chunk_idx):
        return chunk_idx * shard_len

    def x_chunk_ref(chunk_idx):
        return x_ref.at[pl.ds(x_chunk_start(chunk_idx), shard_len)]

    def x_chunk_val(chunk_idx):
        return x_ref[pl.ds(x_chunk_start(chunk_idx), shard_len)]

    first_send_idx = lax.rem(my_device_id + N_DEVICES - 1, N_DEVICES)
    first_recv_idx = lax.rem(my_device_id + N_DEVICES - 2, N_DEVICES)

    pallas_rdma_start(
        src_ref=x_chunk_ref(first_send_idx),
        dst_ref=recv_buf,
        dst_device_id=next_device_id,
        src_send_sem=send_sem,
        dst_recv_sem=recv_sem,
    )
    pallas_rdma_wait_recv(dst_ref=recv_buf, dst_recv_sem=recv_sem)
    carry_buf[...] = recv_buf[...] + x_chunk_val(first_recv_idx)
    pallas_rdma_wait_send(src_ref=x_chunk_ref(first_send_idx), src_send_sem=send_sem)

    for step in range(1, N_DEVICES - 1):
        recv_idx = lax.rem(my_device_id + N_DEVICES - step - 2, N_DEVICES)

        pallas_rdma_start(
            src_ref=carry_buf,
            dst_ref=recv_buf,
            dst_device_id=next_device_id,
            src_send_sem=send_sem,
            dst_recv_sem=recv_sem,
        )
        pallas_rdma_wait_recv(dst_ref=recv_buf, dst_recv_sem=recv_sem)
        pallas_rdma_wait_send(src_ref=carry_buf, src_send_sem=send_sem)

        reduced_chunk = recv_buf[...] + x_chunk_val(recv_idx)
        if step == N_DEVICES - 2:
            out_ref[...] = reduced_chunk
        else:
            carry_buf[...] = reduced_chunk


def all_gather_pallas_scratch_specs(x):
    return {
        "send_sem": pltpu.SemaphoreType.DMA,
        "recv_sem": pltpu.SemaphoreType.DMA,
    }


def all_gather_pallas_kernel(x_ref, out_ref, scratch_refs):
    my_device_id = pallas_get_my_device_id()
    next_device_id = lax.rem(my_device_id + 1, N_DEVICES)
    shard_len = x_ref.shape[0]

    send_sem = scratch_refs["send_sem"]
    recv_sem = scratch_refs["recv_sem"]

    def out_chunk_start(chunk_idx):
        return chunk_idx * shard_len

    def out_chunk_ref(chunk_idx):
        return out_ref.at[pl.ds(out_chunk_start(chunk_idx), shard_len)]

    out_ref[pl.ds(out_chunk_start(my_device_id), shard_len)] = x_ref[...]

    for step in range(N_DEVICES - 1):
        send_idx = lax.rem(my_device_id + N_DEVICES - step, N_DEVICES)
        recv_idx = lax.rem(my_device_id + N_DEVICES - step - 1, N_DEVICES)

        send_ref = out_chunk_ref(send_idx)
        recv_ref = out_chunk_ref(recv_idx)

        pallas_rdma_start(
            src_ref=send_ref,
            dst_ref=recv_ref,
            dst_device_id=next_device_id,
            src_send_sem=send_sem,
            dst_recv_sem=recv_sem,
        )
        pallas_rdma_wait_recv(dst_ref=recv_ref, dst_recv_sem=recv_sem)
        pallas_rdma_wait_send(src_ref=send_ref, src_send_sem=send_sem)


```

## collective matmul：通信与计算重叠

假设我们需要实现 tensor parallel 版本的：`X = gelu(X @ W1) @ W2`。

张量并行的关键是：把 `W1` 按列切，把 `W2` 按行切以后，每个设备都可以独立算出一份贡献，然后在层尾把这些贡献求和。也就是说，原问题被改写成了“局部 matmul + collective”。如果把 all-reduce 展开成 reduce-scatter 再接 all-gather，那么单层从单个 device 的视角就会变成下面这个模式。

```text
单层逻辑：
  local0 = X @ W1^i
  local1 = gelu(local0)
  local2 = local1 @ W2^i
  out    = All-Gather(Reduce-Scatter(local2))

多层串联时，中间层可以改写成：
  out_shard = Reduce-Scatter(gelu(All-Gather(prev_shard) @ W1^i) @ W2^i)
```

### 为什么需要通信与计算重叠

这两个 collective matmul 到底更像 compute-bound 还是 comm-bound？如果这个问题不先想清楚，后面的优化很容易没有方向。

> 假设 x 为 [256, 1024]； W1 为 [1024, 16384]； W2 为[16384，1024]。

先看计算量。对 all-gather-matmul 而言，gather 后的输入是 `[256, 1024]`，本地权重是 `[1024, 4096]`；对 matmul-reduce-scatter 而言，输入是 `[256, 4096]`，权重是 `[4096, 1024]`。两次矩阵乘的 FLOPs 恰好相同：

```text
All-Gather-Matmul FLOPs = 2 * 256 * 1024 * 4096 = 2,147,483,648
Matmul-Reduce-Scatter FLOPs = 2 * 256 * 4096 * 1024 = 2,147,483,648
```

TPU v5p bfloat16 matmul 峰值大约是整芯片 459 TFLOP/s，也就是单 core 大约 230 TFLOP/s。假设完全只受算力约束，两次 matmul 的理论最短时间都约为 **9.34 微秒**。

再看通信量。每个设备手里的 shard 大小是 `[256, 256]`，也就是 `256 * 256 * 2 B = 131072 B = 128 KiB`。在 4-device ring 中，不论 all-gather 还是 reduce-scatter，每个设备都需要在一个方向上发送 3 个 shard，并在另一个方向上接收 3 个 shard，因此每个方向的数据量都是 **384 KiB**。如果只受 ICI 带宽限制，以单向 92 GB/s 估算，理论最短时间大约是 **4.27 微秒**。

把两者放在一起比较，就能得到一个非常重要的结论：**这两个 hybrid primitive 的理论下界里，compute 时间大于 communication 时间。** 也就是说，如果你能把通信较好地藏在计算后面，那么理想状态下性能更可能被 matmul 而不是 ICI 限制。换句话说，**优化目标不是“减少通信到零”，而是“尽量让通信别裸露在 critical path 上”**。这就是 overlap 的价值所在。

| 操作 | 理论计算下界 | 理论通信下界 | 结论 |
|-|-|-|-|
| All-Gather-Matmul | 约 9.34 us | 约 4.27 us | 更偏 compute-bound，可通过 overlap 隐藏通信 |
| Matmul-Reduce-Scatter | 约 9.34 us | 约 4.27 us | 同样更偏 compute-bound |

collective matmul 实现上述tensor parallel 版本的 `X = gelu(X @ W1) @ W2`：

```python
import time

import jax
import numpy as np
from jax import lax, numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec

AXIS_NAME = "i"
N_DEVICES = 4

ENABLE_DEBUG = False

N_BATCH = 256
K1 = 1024
K2 = 4096 


def pallas_get_my_device_id():
    return lax.axis_index(AXIS_NAME)


def pallas_rdma_start(*, src_ref, dst_ref, dst_device_id, src_send_sem, dst_recv_sem):
    pltpu.make_async_remote_copy(
        src_ref=src_ref,
        dst_ref=dst_ref,
        send_sem=src_send_sem,
        recv_sem=dst_recv_sem,
        device_id=dst_device_id,
        device_id_type=pltpu.DeviceIdType.LOGICAL,
    ).start()


def pallas_rdma_wait_send(*, src_ref, src_send_sem):
    pltpu.make_async_remote_copy(
        src_ref=src_ref,
        dst_ref=src_ref,  # ignored by 'wait_send'
        send_sem=src_send_sem,
        recv_sem=src_send_sem,  # ignored by 'wait_send'
        device_id=0,  # ignored by 'wait_send'
        device_id_type=pltpu.DeviceIdType.LOGICAL,
    ).wait_send()


def pallas_rdma_wait_recv(*, dst_ref, dst_recv_sem):
    pltpu.make_async_remote_copy(
        src_ref=dst_ref,  # ignored by 'wait_recv'
        dst_ref=dst_ref,
        send_sem=dst_recv_sem,  # ignored by 'wait_recv'
        recv_sem=dst_recv_sem,
        device_id=0,  # ignored by 'wait_recv'
        device_id_type=pltpu.DeviceIdType.LOGICAL,
    ).wait_recv()


def _matmul_bf16(x, w):
    return jnp.astype(pl.dot(x, w), jnp.bfloat16)


def _full_to_chunk_major(x, shard_width):
    return jnp.transpose(x.reshape(x.shape[0], N_DEVICES, shard_width), (1, 0, 2))


def _chunk_major_to_full(x):
    return jnp.transpose(x, (1, 0, 2)).reshape(x.shape[1], x.shape[0] * x.shape[2])


def _chunk_ref(buf_ref, chunk_idx):
    return buf_ref.at[chunk_idx, :, :]


def _chunk_val(buf_ref, chunk_idx):
    return buf_ref[chunk_idx, :, :]


def _all_gather_ring(local_ref, gathered_ref, send_sem, recv_sem):
    my_device_id = pallas_get_my_device_id()
    next_device_id = lax.rem(my_device_id + 1, N_DEVICES)

    _chunk_ref(gathered_ref, my_device_id)[...] = local_ref[...]

    for step in range(N_DEVICES - 1):
        send_idx = lax.rem(my_device_id + N_DEVICES - step, N_DEVICES)
        recv_idx = lax.rem(my_device_id + N_DEVICES - step - 1, N_DEVICES)

        send_ref = _chunk_ref(gathered_ref, send_idx)
        recv_ref = _chunk_ref(gathered_ref, recv_idx)

        pallas_rdma_start(
            src_ref=send_ref,
            dst_ref=recv_ref,
            dst_device_id=next_device_id,
            src_send_sem=send_sem,
            dst_recv_sem=recv_sem,
        )
        pallas_rdma_wait_recv(dst_ref=recv_ref, dst_recv_sem=recv_sem)
        pallas_rdma_wait_send(src_ref=send_ref, src_send_sem=send_sem)


def _reduce_scatter_ring(chunked_ref, out_ref, carry_ref, recv_ref, send_sem, recv_sem):
    my_device_id = pallas_get_my_device_id()
    next_device_id = lax.rem(my_device_id + 1, N_DEVICES)

    first_send_idx = lax.rem(my_device_id + N_DEVICES - 1, N_DEVICES)
    first_recv_idx = lax.rem(my_device_id + N_DEVICES - 2, N_DEVICES)

    first_send_ref = _chunk_ref(chunked_ref, first_send_idx)
    pallas_rdma_start(
        src_ref=first_send_ref,
        dst_ref=recv_ref,
        dst_device_id=next_device_id,
        src_send_sem=send_sem,
        dst_recv_sem=recv_sem,
    )
    pallas_rdma_wait_recv(dst_ref=recv_ref, dst_recv_sem=recv_sem)
    carry_ref[...] = recv_ref[...] + _chunk_val(chunked_ref, first_recv_idx)
    pallas_rdma_wait_send(src_ref=first_send_ref, src_send_sem=send_sem)

    for step in range(1, N_DEVICES - 1):
        recv_idx = lax.rem(my_device_id + N_DEVICES - step - 2, N_DEVICES)

        pallas_rdma_start(
            src_ref=carry_ref,
            dst_ref=recv_ref,
            dst_device_id=next_device_id,
            src_send_sem=send_sem,
            dst_recv_sem=recv_sem,
        )
        pallas_rdma_wait_recv(dst_ref=recv_ref, dst_recv_sem=recv_sem)
        pallas_rdma_wait_send(src_ref=carry_ref, src_send_sem=send_sem)

        reduced_chunk = recv_ref[...] + _chunk_val(chunked_ref, recv_idx)
        if step == N_DEVICES - 2:
            out_ref[...] = reduced_chunk
        else:
            carry_ref[...] = reduced_chunk


def matmul_pallas_scratch_specs(x, w):
    return {}


def matmul_pallas_kernel(x_ref, w_ref, out_ref, scratch_refs):
    out_ref[...] = _matmul_bf16(x_ref[...], w_ref[...])


def all_gather_matmul_pallas_scratch_specs(x):
    gathered_shape = (N_DEVICES, x.shape[0], x.shape[1])
    return {
        "gathered_x": pltpu.VMEM(shape=gathered_shape, dtype=x.dtype),
        "send_sem": pltpu.SemaphoreType.DMA,
        "recv_sem": pltpu.SemaphoreType.DMA,
    }


def all_gather_matmul_pallas_kernel(x_ref, w1_ref, out_ref, scratch_refs):
    gathered_x = scratch_refs["gathered_x"]
    send_sem = scratch_refs["send_sem"]
    recv_sem = scratch_refs["recv_sem"]

    _all_gather_ring(x_ref, gathered_x, send_sem, recv_sem)
    out_ref[...] = _matmul_bf16(_chunk_major_to_full(gathered_x[...]), w1_ref[...])


def matmul_reduce_scatter_pallas_scratch_specs(x):
    shard_width = K1 // N_DEVICES
    return {
        "partial_chunks": pltpu.VMEM(
            shape=(N_DEVICES, x.shape[0], shard_width), dtype=x.dtype
        ),
        "carry_buf": pltpu.VMEM(shape=(x.shape[0], shard_width), dtype=x.dtype),
        "recv_buf": pltpu.VMEM(shape=(x.shape[0], shard_width), dtype=x.dtype),
        "send_sem": pltpu.SemaphoreType.DMA,
        "recv_sem": pltpu.SemaphoreType.DMA,
    }


def matmul_reduce_scatter_pallas_kernel(x_ref, w2_ref, out_ref, scratch_refs):
    partial_chunks = scratch_refs["partial_chunks"]
    carry_buf = scratch_refs["carry_buf"]
    recv_buf = scratch_refs["recv_buf"]
    send_sem = scratch_refs["send_sem"]
    recv_sem = scratch_refs["recv_sem"]

    partial_chunks[...] = _full_to_chunk_major(
        _matmul_bf16(x_ref[...], w2_ref[...]), out_ref.shape[1]
    )
    _reduce_scatter_ring(
        partial_chunks, out_ref, carry_buf, recv_buf, send_sem, recv_sem
    )


def neural_network_pallas_scratch_specs(x, w1_refs, w2_refs):
    shard_width = x.shape[1] // N_DEVICES
    return {
        "matmul1_out": pltpu.VMEM(shape=(x.shape[0], K2), dtype=x.dtype),
        "chunk_buf": pltpu.VMEM(
            shape=(N_DEVICES, x.shape[0], shard_width), dtype=x.dtype
        ),
        "shard_buf": pltpu.VMEM(shape=(x.shape[0], shard_width), dtype=x.dtype),
        "carry_buf": pltpu.VMEM(shape=(x.shape[0], shard_width), dtype=x.dtype),
        "recv_buf": pltpu.VMEM(shape=(x.shape[0], shard_width), dtype=x.dtype),
        "send_sem": pltpu.SemaphoreType.DMA,
        "recv_sem": pltpu.SemaphoreType.DMA,
    }


def neural_network_pallas_kernel(init_x_ref, w1_refs, w2_refs, out_ref, scratch_refs):
    matmul1_out = scratch_refs["matmul1_out"]
    chunk_buf = scratch_refs["chunk_buf"]
    shard_buf = scratch_refs["shard_buf"]
    carry_buf = scratch_refs["carry_buf"]
    recv_buf = scratch_refs["recv_buf"]
    send_sem = scratch_refs["send_sem"]
    recv_sem = scratch_refs["recv_sem"]

    for layer_idx in range(w1_refs.shape[0]):
        w1_ref = w1_refs.at[layer_idx]
        w2_ref = w2_refs.at[layer_idx]
        x_val = init_x_ref[...] if layer_idx == 0 else out_ref[...]

        matmul1_out[...] = jnp.astype(
            jax.nn.gelu(_matmul_bf16(x_val, w1_ref[...])), matmul1_out.dtype
        )
        chunk_buf[...] = _full_to_chunk_major(
            _matmul_bf16(matmul1_out[...], w2_ref[...]), shard_buf.shape[1]
        )

        _reduce_scatter_ring(
            chunk_buf, shard_buf, carry_buf, recv_buf, send_sem, recv_sem
        )
        _all_gather_ring(shard_buf, chunk_buf, send_sem, recv_sem)
        out_ref[...] = _chunk_major_to_full(chunk_buf[...])


```

---

最后一次更新时间：`2026-08-05 14:01:27 CST`
