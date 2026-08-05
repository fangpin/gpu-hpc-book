# 给 MLSys 工程的几个启发

**异步不是把 sync 删掉，而是换一种更精确的等待。**TMA load 之后不能靠 `cta_sync()` 判断完成，而要用 mbarrier 追踪 pending bytes。TMA store 之后也不能直接复用 `Dsmem`，而要 commit/wait bulk group。异步路径需要更具体的完成条件，而不是更少的正确性约束。

**Pipeline 首先是存储生命周期设计。**双缓冲不只是“多分配一份 SMEM”，而是明确每个 stage 在某个时间点归 load path、compute path 还是 free path。没有这个生命周期，overlap 会变成数据覆盖 bug。

**调度顺序会影响 cache 复用。**Step 6 的 scheduler 没有改变每个 CTA 内部如何计算一个 tile，却改变了 tile 被领取的顺序。对于大矩阵 GEMM，相邻输出 tile 对 A/B operand 有天然复用关系；把这种关系暴露给 L2，是 persistent scheduling 的核心收益之一。

**Barrier phase 是可维护性问题。**当 barrier 只服务一个 K-loop 时，phase 已经容易出错；当 barrier 被 persistent CTA 跨多个 tile 复用时，phase parity 更需要被参数约束或显式维护。工程上，像 `K_TILES % (2 * PIPE_DEPTH) == 0` 这样的 assert 不是保守，而是在保护异步协议的前提。

---

最后一次更新时间：`2026-08-05 16:12:21 CST`
