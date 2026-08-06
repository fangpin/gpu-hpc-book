# Step 6：Persistent kernel 让 CTA 持续领取 tile

Step 4 和 Step 5 主要优化一个输出 tile 内部的 K-loop。Step 6 则换了一个层面：跨输出 tile 的调度。原来的二维 grid 是一个 CTA 对应一个 `128 x 128` 输出 tile。对于 `M=N=4096` 且 `BLK_M=BLK_N=128` 的矩阵，一共有 `32 x 32 = 1024` 个输出 tile，也就是 1024 个 CTA。每个 CTA 初始化资源、计算一个 tile，然后退出。

persistent kernel 的思路是：只启动一个固定大小的 CTA 池，让每个 CTA 在 kernel 内通过 scheduler 领取多个 tile。原文示例设置 `SM_COUNT=148`，用一维 grid 启动 148 个 persistent CTA。每个 CTA 初始化一次 TMEM、mbarrier 和 scheduler 状态，然后在 `while tile_scheduler.valid()` 循环里处理多个 tile。

```python
SM_COUNT = 148

bx = T.cta_id([SM_COUNT])

tile_scheduler = ClusterPersistentScheduler2D(
    "ts",
    num_m_tiles=M // BLK_M,
    num_n_tiles=N // BLK_N,
    l2_group_size=8,
    num_clusters=SM_COUNT
)
tile_scheduler.init(bx)

while tile_scheduler.valid():
    m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
    n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

    # reuse the same Step 5 staged K-loop for this output tile
    ...

    tile_scheduler.next_tile()
```

persistent scheduling 的收益有两层。第一是摊薄初始化成本：TMEM 分配、barrier 初始化、scheduler 初始化不再每个输出 tile 都重新发生。第二是改善 L2 局部性：`l2_group_size=8` 让 scheduler 在 M 方向把 8 行 tile 组织成一组，在同一个 N tile column 内优先沿这组 M tile 推进。这样，一批相邻工作更可能复用相同的 B tile，也会在较短时间窗口内回访相近的 A tile。

需要注意的是，`SM_COUNT` 并不表示 CTA 被永久绑定到某个 SM。它只是 launch 的 persistent CTA 数量。实际哪些 CTA 同时 resident、由哪些 SM 执行，仍由硬件调度决定。这里的“persistent”指的是 CTA 生命周期变长：它不再算完一个 tile 就退出，而是在同一份本地资源上持续处理 tile。

---

最后一次更新时间：`2026-08-05 16:12:21 CST`

原文链接：[https://fangpin.github.io/gpu-hpc-book/#/chapters/16-step-6-persistent-kernel-cta-tile.md](https://fangpin.github.io/gpu-hpc-book/#/chapters/16-step-6-persistent-kernel-cta-tile.md)
