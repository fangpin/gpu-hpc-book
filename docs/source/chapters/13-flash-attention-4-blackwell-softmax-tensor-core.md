# Flash Attention 4：在 Blackwell 上把 Softmax 接进 Tensor Core 流水线

```{contents} 本页目录
---
depth: 2
local: true
---
```

原文来自 MLC.AI 的 [Flash Attention 4](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_flash_attention/index.html)。前面 GEMM 章节讨论的是一个相对纯粹的数据路径：operand 从 GMEM 进入 SMEM，Tensor Core 把累加结果放进 TMEM，最后再写回 GMEM。Flash Attention 4 的难点在于，它不是一次单纯的矩阵乘，而是把 `QK^T`、row-wise softmax、`PV` 三段计算接成一条流水线，同时避免把完整 attention score matrix 写到全局内存。

这篇技术文档不逐段翻译原文，而是从工程视角重写 FA4 的核心逻辑：在线 softmax 如何维持数值稳定，为什么 conditional rescaling 可以减少 TMEM 往返，`S`、`P`、`O` 三类 tile 怎样在 TMEM、寄存器和 SMEM 之间流动，以及 warpgroup 如何分工来让 QK、softmax、PV 和输出修正互相衔接。

> 核心观点：FA4 的性能关键不只是“少写一个中间矩阵”，而是把 attention 的依赖链拆成可交接的 tile primitive。`QK^T` 产生 `S`，softmax 把 `S` 转成 `P`，`PV` 更新 `O`；这些阶段由不同 warpgroup 执行，并通过 TMEM、SMEM mailbox 和 barrier 精确交接。

## Attention 的真实瓶颈：不是公式，而是中间矩阵

Attention 前向计算可以写成：

```text
O = softmax(QK^T / sqrt(d)) V
```

其中 `QK^T` 生成 query 与 key 之间的 score matrix。对于序列长度为 `L` 的单个 head，这个矩阵大小是 `L x L`。如果直接把它完整物化到 GMEM，再读回来做 softmax 和第二次矩阵乘，访存量会随序列长度平方增长。FlashAttention 的基本思想就是按 K/V block 流式处理，只保留当前 tile 和每一行的在线 softmax 状态。

对固定 query 行 `i`，kernel 不需要保存所有 `s_ij = q_i · k_j`。它只需要维护三类逐行状态：

- `row_max`：当前指数参考值，也就是 softmax 中被减掉的 reference。
- `row_sum`：相对于当前 reference 的未归一化 softmax 分母。
- `O`：相对于同一个 reference 累加的加权 value 向量，最后再除以 `row_sum`。

因此，FA4 的核心不是把 softmax “搬到片上”这么简单，而是让每个 K/V block 消费完之后，score tile 可以立刻丢弃，只留下可继续更新的 `row_max`、`row_sum` 和 `O`。

## 在线 Softmax 与 Conditional Rescaling

为了数值稳定，softmax 通常会减去一行中的最大值。在线处理 K/V block 时，新的 block 可能带来更大的 score。标准 online softmax 会把 reference 更新到新的最大值，并把旧的 `row_sum` 和 `O` 重新缩放到新 reference 下。问题是，`O` 存在 TMEM 中，缩放它需要走一遍 `TMEM -> registers -> TMEM`，这会消耗额外带宽和指令。

FA4 使用 base-2 exponent，并定义：

```text
scale_log2 = log2(e) / sqrt(d)
```

设旧 reference 为 `row_max`，当前 block 和旧 reference 的候选最大值为 `candidate_max`。代码计算：

```text
delta = (row_max - candidate_max) * scale_log2
```

由于 `candidate_max >= row_max`，`delta` 不会大于 0。当前实现使用阈值 8：当 `delta >= -8` 时，继续沿用旧 reference，`acc_scale = 1`，不需要修正旧的 `O`；当 `delta < -8` 时，才切换到新的 reference，并设置 `acc_scale = exp2(delta)`，把旧状态缩放到新 reference 下。

| 情况 | reference 选择 | `acc_scale` | 对 `O` 的影响 |
|-|-|-|-|
| 第一个 K/V block | 采用当前 block 的最大值 | 1 | 直接初始化 `O` |
| `delta >= -8` | 保留旧 `row_max` | 1 | 旧 `O` 不需要从 TMEM 读出重写 |
| `delta < -8` | 采用 `candidate_max` | `exp2(delta)` | 旧 `O` 需要乘以 `acc_scale` 后再继续累加 |

阈值 8 的直观含义是允许最多 `2^8 = 256` 倍的指数尺度差。在这个范围内，kernel 宁愿保持旧 reference，让当前 block 的 unnormalized weight 变大一些，也不立刻付出 rescale `O` 的代价。超过这个范围时，再把 reference 切到新的最大值，保证数值范围仍受控。

## 一个 K/V Block 的数据流

把算法放到 Blackwell 的硬件路径上，FA4 的一个 K/V block 可以拆成下面几条 tile 数据流：

```text
Q, K:  GMEM --TMA load--> SMEM --QK^T MMA--> S in TMEM
S:     TMEM --tcgen05.ld--> registers --softmax--> P in registers
P:     registers --TMEM store--> P in TMEM
V:     GMEM --TMA load--> SMEM
P, V:  P in TMEM + V in SMEM --PV MMA--> O in TMEM

when needed: O in TMEM --tcgen05.ld--> registers --rescale--> O in TMEM
at the end:  O in TMEM --tcgen05.ld--> registers --normalize/cast--> SMEM --TMA store--> GMEM
```

这条路径里有三个关键中间 tile。`S` 是 `QK^T` 的 score tile，由 `tcgen05.mma` 写入 TMEM。`P` 是 softmax 后的未归一化权重，先在寄存器中算出来，再以 fp16 view 写回 TMEM。`O` 是 `P @ V` 的输出累加器，同样放在 TMEM 中。

| 阶段 | Scope | Layout / 存储位置 | Dispatch / 硬件路径 |
|-|-|-|-|
| 加载 Q/K/V | WG3 的 TMA load warp | GMEM tile 进入 SMEM stage | `Tx.copy_async(..., dispatch="tma_auto")` |
| QK^T MMA | WG3 的 MMA warp | SMEM 中的 Q/K 到 TMEM 中的 `S` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` |
| Softmax | WG0 或 WG1 | `S` 从 TMEM 读到寄存器，`P` 再写回 TMEM | `tcgen05.ld` + CUDA core softmax + TMEM store |
| PV MMA | WG3 的 MMA warp | TMEM 中的 `P` 与 SMEM 中的 `V` 更新 TMEM 中的 `O` | `tcgen05.mma`，其中一个 operand 来自 TMEM |
| Correction | WG2 | `O` 从 TMEM 到寄存器，再写回 TMEM | `tcgen05.ld` / TMEM store / register multiply |
| Epilogue | 非 causal 下主要由 WG2 完成 | 最终 `O` 从 TMEM 到寄存器，再经 SMEM 写回 GMEM | `tcgen05.ld` + TMA store |

和前面 GEMM 相比，FA4 多了两个高成本交接：softmax 必须把 `S` 从 TMEM 读到寄存器，又要把 `P` 写回 TMEM；如果 reference 变化过大，`O` 还要额外经历一次 TMEM 读写。FA4 的很多优化，本质上都在减少这些交接的等待和无效搬运。

## Warpgroup 分工：四个 WG 各管一段依赖链

FA4 的一个 CTA 包含四个 warpgroups，每个 warpgroup 128 个线程。当前实现保留两个 Q tile in flight，也就是两个 Q stage：stage 0 由 WG0 做 softmax，stage 1 由 WG1 做 softmax。WG3 负责 TMA 和 MMA 的发起，WG2 负责 `O` correction，并在非 causal 路径上负责最终 epilogue。

| 执行角色 | 职责 | 为什么这样拆 |
|-|-|-|
| WG3, warp 1 | 发起 Q/K/V 的 TMA load | 一个 warp 足够提交异步搬运，避免让所有线程执行 copy |
| WG3, warp 0 | 发起 QK^T MMA 和 PV MMA | 同一个 issuer 串起 Tensor Core 操作，减少调度复杂度 |
| WG3, warp 2 | 发起最终 O 的 TMA store | 把写回和主计算流水线分离 |
| WG0 | 处理 Q stage 0 的 softmax | 保留一整行 128 个 fp32 score 和 softmax 临时量 |
| WG1 | 处理 Q stage 1 的 softmax | 和 WG0 对称，让两个 Q stage 交替推进 |
| WG2 | 按需 rescale `O`，非 causal 下执行 epilogue | 把 TMEM 中 `O` 的修正从 softmax 路径中拆出去 |

这种拆分也影响寄存器预算。Softmax 需要每个线程持有一行 128 个 fp32 score 以及临时变量，因此 WG0 和 WG1 的寄存器需求最高。当前实现通过 `setmaxnreg` 动态调节每个角色的上限：

```python
if wg_id == 3:
    T.ptx.setmaxnreg(False, 48)
elif wg_id < 2:
    T.ptx.setmaxnreg(True, 200)
elif wg_id == 2:
    T.ptx.setmaxnreg(False, 64)
```

对应的总预算是 `128 * (200 + 200 + 64 + 48) = 65,536` 个寄存器。对比如果四个 warpgroup 都按 200 个寄存器配置，则需要 `128 * 4 * 200 = 102,400` 个寄存器。FA4 通过让 WG3 和 WG2 释放寄存器，把资源集中给真正需要大寄存器文件的 softmax 角色。

## Softmax、Correction 与 Barrier 的交接

FA4 中最容易出错的不是单个算子，而是交接顺序。QK^T MMA 写完 `S` 之后，softmax 才能读；softmax 把一部分 `P` 写回 TMEM 之后，PV MMA 可以先消费第一段；如果旧 `O` 需要 rescale，WG2 必须在 PV MMA 累加前完成修正。

关键 barrier 可以这样理解：

- `s_ready`：QK^T MMA 完成，`S` 可以被 softmax 读取。
- `p_o_rescale`：softmax 已经写好第一段 `P`，并且 WG2 已经让 `O` 处于可累加状态。
- `p_ready_2`：softmax 已经写好剩余 `P`，PV MMA 可以执行第二段。
- `o_ready`：PV MMA 对 `O` 的更新完成，后续 correction 或 epilogue 可以读取 `O`。
- `softmax_corr.empty`：WG2 已经读走 softmax 写入 SMEM mailbox 的 `acc_scale` 或 `row_sum`，该 mailbox 可以复用。

`p_o_rescale` 的 arrival count 是 256，因为它合并了两个条件：softmax warpgroup 的 128 个线程报告第一段 `P` 已写好，WG2 的 128 个线程报告 `O` 已经完成 rescale 或确认无需 rescale。只有这两个条件同时满足，WG3 才能让第一段 PV MMA 消费 `P` 并更新 `O`。

这里的一个重要优化是 PV MMA 分段。非 causal 路径使用 96+32 的 split，causal 路径使用 64+64 的 split。这样，softmax 不必等 128 列 `P` 全部写完，PV MMA 就能先消费第一段，后续 softmax 再把剩余列交给第二段 PV MMA。这个 split 把 softmax 写回和 Tensor Core 消费做成了更细粒度的 overlap。

## Conditional Rescaling 为什么有效

`row_sum` 留在 softmax warpgroup 的寄存器中，乘以 `acc_scale` 是本地操作；`O` 在 TMEM 中，修正它要付出 TMEM load、register multiply、TMEM store 三步。conditional rescaling 的收益就来自这里：如果多数行的 `acc_scale = 1`，WG2 就可以跳过实际数据路径，只保留同步 arrival。

```text
should_rescale = acc_scale < 1.0
any_needs_rescale = any_sync(should_rescale)

if any_needs_rescale:
    O_row = load_from_tmem()
    O_row = O_row * acc_scale
    store_to_tmem(O_row)

arrive(p_o_rescale)
arrive(softmax_corr.empty)
```

注意，跳过数据修正不等于跳过同步。WG2 即使发现当前 warp 负责的 32 行都不需要 rescale，也必须继续对 `p_o_rescale` 和 `softmax_corr.empty` 做 arrival。否则 PV MMA 会一直等待 `O` 可用，softmax 也无法复用 mailbox。

这也是 FA4 里“优化”和“正确性协议”分得很清楚的地方：conditional rescaling 可以减少 TMEM 往返，但不能改变 barrier contract。只要某个消费者还依赖这个阶段给出的 readiness 信号，即使没有实际数据操作，也要维持同样的同步行为。

## Causal Attention 与 GQA：同一条数据路径上的两个特化

在 causal attention 中，每个 query 只能访问自己之前的位置。当前实现使用 bottom-right-aligned causal mask：当 `SEQ_LEN_Q` 与 `SEQ_LEN_KV` 不相等时，query 位置 `i` 最多能看见 key 位置 `i + SEQ_LEN_KV - SEQ_LEN_Q`。kernel 会跳过完全无效的 K/V block，并在跨越边界的 block 中把无效列设为 `-inf`，使其 softmax 权重变为 0。

Causal 路径仍然是 QK^T MMA、softmax、PV MMA、correction 和 writeback 这条链，但它会改变几个工程细节：K/V block 的访问数量不再均匀，softmax 里要做 register-level mask，PV split 从非 causal 的 96+32 变成 64+64，最终 epilogue 也从 WG2 移到 WG0/WG1，减少最后一次 `row_sum` mailbox 往返。

GQA，也就是 Grouped Query Attention，则改变 Q/K/V head 的映射方式。设 query head 数为 `num_qo_heads`，K/V head 数为 `num_kv_heads`，则：

```python
GQA_RATIO = num_qo_heads // num_kv_heads
SEQ_Q_PER_TILE = BLK_M // GQA_RATIO
```

当 `GQA_RATIO = 4` 且 `BLK_M = 128` 时，一个 Q tile 的 128 行可以解释为 32 个 sequence position 乘以 4 个 query head。第 `row` 行对应：

```text
seq_offset    = row // GQA_RATIO
q_head_offset = row % GQA_RATIO
q_head        = kv_head_idx * GQA_RATIO + q_head_offset
```

K/V 不需要为每个 query head 复制。打包进 128 行的多个 query head 共享同一个 `kv_head_idx` 对应的 K/V tile。对 Tensor Core 来说，Q operand 仍是普通的 `128 x HEAD_DIM` tile；4D view 只是在 TMA load 和最终 O store 时，把物理行解释回 `(sequence, query head)` 坐标。

## 调度策略：non-causal 和 causal 的任务形状不同

FA4 scheduler 把一个 CTA 映射到 `(batch, kv_head, m_block)` 任务。一个 `m_block` 内部包含两个 Q stage，因此一个任务会同时推进两个 query tile。non-causal 下，每个任务访问同样数量的 K/V block，工作量均匀；causal 下，靠前的 query block 访问的 K/V block 更少，靠后的 query block 更重。

| 模式 | Scheduler | 调度动机 |
|-|-|-|
| Non-causal | `FlashAttentionLinearScheduler` | 任务成本均匀，persistent CTA 完成一个 tile 后按线性顺序领取下一个任务。 |
| Causal | `FlashAttentionLPTScheduler` | 任务成本不均匀，优先调度更重的后部 query block，并通过 `L2_SWIZZLE` 控制活跃 K/V working set。 |

原文中的 `max_ctas=148` 和 `L2_SIZE=50 MiB` 是针对书中 B200 配置选择的调度常量，不应该理解成所有 Blackwell GPU 的通用参数。换到不同 SM 数或 cache 配置时，这些值需要重新调优，或者由目标硬件信息推导。

## 当前实现与论文描述的差异

原文特别提醒：代码基于当前 `flash_attention4.py` 的默认路径，不能把论文中的每个设计都当成已经启用的实现细节。两个差异尤其重要。

- 论文中会错开 WG0 和 WG1 的 exponential-heavy 区域，避免两个 softmax warpgroup 同时争用指数单元。当前实现保留了相关同步分支，但默认 `USE_S0_S1_BARRIER=False`。
- 论文中使用空闲 TMEM 传递 correction statistics；当前 TIRx 实现使用 SMEM 中的 `sScale` 作为 mailbox，通过 named barrier 通知 WG2 读取 `acc_scale` 或最终 `row_sum`。

这两个点决定了阅读源码时的边界：本文解释的是当前 TIRx kernel 默认路径，而不是把 FA4 paper 的所有策略都合并成一个泛化版本。

## 编译与验证约束

当前 `flash_attention4.py` 不是通用 attention API，而是围绕固定 tile shape 写出的专用 kernel。运行示例前需要满足这些约束：

- `NUM_QO_HEADS % NUM_KV_HEADS == 0`，保证 `GQA_RATIO` 是整数。
- `GQA_RATIO` 必须整除 `BLK_M = 128`，否则 128 个 packed Q 行无法均匀映射回 sequence position。
- `HEAD_DIM == 128`，因为 TMEM region、PV MMA 和 epilogue 都围绕这个宽度组织。
- non-causal 路径要求 `SEQ_LEN_KV` 能被 `BLK_N = 128` 整除；代码会向上取 K/V block 数，但没有为 non-causal 的最后一个 partial block 做 tail mask。

```python
import torch
import torch.nn.functional as F
import tvm
from tirx_kernels.attention.flash_attention4 import get_flash_attention4_kernel

B, S, Hq, Hkv, D = 1, 1024, 32, 8, 128
assert Hq % Hkv == 0
assert 128 % (Hq // Hkv) == 0
assert D == 128
assert S % 128 == 0

Q = torch.randn(B, S, Hq, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
O = torch.empty(B, S, Hq, D, dtype=torch.float16, device="cuda")

kernel = get_flash_attention4_kernel(B, S, S, Hq, Hkv, D, is_causal=False)
target = tvm.target.Target("cuda")
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

ex.mod(Q, K, V, O)
torch.cuda.synchronize()

qt, kt, vt = (x.transpose(1, 2).float() for x in (Q, K, V))
ref = F.scaled_dot_product_attention(qt, kt, vt, enable_gqa=True).transpose(1, 2).half()
torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)
```

验证使用 `torch.nn.functional.scaled_dot_product_attention(..., enable_gqa=True)` 作为参考，容忍度为 `rtol=1e-2, atol=1e-2`。这个误差范围反映了 fp16 存储、hardware `exp2` 与 FMA polynomial approximation、blockwise accumulation 顺序和最终 fp16 cast 带来的有限精度差异。如果误差显著变大，更常见的根因是 handoff 错了：例如漏等 `s_ready`、`p_o_rescale` 或 `p_ready_2`，或者 `row_max`/`row_sum` 没有正确传到 correction path。

## 总结

Flash Attention 4 把前面 GEMM 章节里的 TMA、`tcgen05`、TMEM 和 barrier 机制组合成了更长的依赖链。QK^T MMA 只负责生成 score tile，softmax 在寄存器中做 row-wise 归一化准备，PV MMA 再把权重和 V 结合起来更新输出。整个过程中，完整 score matrix 从未进入 GMEM。

真正值得关注的是它的工程边界：`S`、`P`、`O` 都在片上流转，但每次跨角色交接都必须有明确的 readiness 证明；conditional rescaling 可以减少 `O` 的 TMEM 往返，但不能省略同步协议；GQA 和 causal mask 改变的是坐标解释和任务成本，而不是 attention 主数据路径。理解这些边界，才能把 FA4 看成一个可维护的 GPU kernel，而不是一组难以追踪的异步指令堆叠。
