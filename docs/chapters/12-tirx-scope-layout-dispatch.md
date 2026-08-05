# TIRx 视角：scope、layout、dispatch 是可读性的核心

这篇原文真正有价值的地方，不只是给出三个 kernel 版本，而是展示了 TIRx 如何把低层硬件契约写成相对稳定的程序结构。**Scope** 决定谁参与操作：CTA copy、warpgroup copy、单个 elected issuer、二维 CTA grid，都是 scope 的变化。**Layout** 决定数据如何被硬件解释：A/B 在 SMEM 中使用 swizzled layout，TMEM accumulator 使用 `TLane/TCol` 视图，寄存器写回时又用 `tid_in_wg` 把 128 行分配给 128 个线程。**Dispatch** 决定 tile primitive 走哪条硬件路径：这个 baseline 里，MMA 明确走 `tcgen05`。

把这三个词放在一起，kernel 的演进就不再像一串神秘 intrinsic。Step 1 的主要目标是建立 layout 与 dispatch 的最小闭环；Step 2 主要是在同一 layout 上复用 accumulator，并补上 barrier phase 这个同步状态；Step 3 则把 scope 从单 CTA 扩展到二维 grid，同时保持 CTA 内的数据路径不变。

---

最后一次更新时间：`2026-08-05 14:01:27 CST`
