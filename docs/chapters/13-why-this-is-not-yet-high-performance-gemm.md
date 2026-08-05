# 为什么这还不是高性能 GEMM

到 Step 3 为止，kernel 已经“能算对”，但离“跑满硬件”还很远。最明显的问题是 GMEM 到 SMEM 的加载仍由线程同步执行，Tensor Core 在等待数据时没有被充分隐藏；K-loop 每轮都按顺序 load、sync、compute、wait，没有软件流水；CTA grid 虽然覆盖了空间 tiling，却没有把相邻 CTA 会复用的 A/B tile 留在更近的位置；角色分工也还没有做 warp specialization。

这也是原文把 GEMM 分成三章讲的原因。基础章负责让每个 memory hop、每个同步点和每个 tile 坐标都可验证；下一章开始把 copy 路径替换成 TMA，并用多级 SMEM stage 做 pipeline；再往后，persistent scheduling、warp specialization、CTA cluster 和 multi-consumer execution 才会逐步进入。换句话说，性能优化不是推翻 baseline，而是在同一条数据路径上减少等待、减少重复搬运、提高并行密度。

---

最后一次更新时间：`2026-08-05 14:01:27 CST`
