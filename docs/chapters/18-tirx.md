# TIRx 视角：这章改的是执行路径，不是数学定义

从 TIRx 的 scope/layout/dispatch 三元组看，这一章的演进非常规整。数学上仍然是 $D = A B^\top$，tile shape 仍然是 `128 x 128 x 64`，MMA 仍然 dispatch 到 `tcgen05`。变化发生在执行路径和调度层。

| 维度 | Step 3 baseline | Step 4 | Step 5 | Step 6 |
|-|-|-|-|-|
| Scope | 二维 grid，一 CTA 一输出 tile | 同上，单线程发起 TMA | 同上，单 warpgroup 管理 stage ring | 一维 persistent CTA pool，通过 scheduler 领取 tile |
| Layout | 单份 A/B SMEM tile | 增加 Dsmem 作为 TMA store staging | A/B SMEM 增加 `PIPE_DEPTH` stage 维度 | 每个 persistent CTA 复用同一套 per-tile layout |
| Dispatch | 线程 copy + `tcgen05` | TMA load/store + `tcgen05` | TMA prefetch + staged `tcgen05` | 同 Step 5，外层调度改为 scheduler |

这个视角能帮助我们避免把代码看成一堆 intrinsic。`Tx.copy_async(..., dispatch="tma")` 是 dispatch 变化；`Asmem[stage, :, :]` 是 layout 和 storage lifetime 变化；`ClusterPersistentScheduler2D` 是 scope 和 tile ownership 变化。每个变化都对应一个新的同步或生命周期约束。

---

最后一次更新时间：`2026-08-05 16:12:21 CST`
