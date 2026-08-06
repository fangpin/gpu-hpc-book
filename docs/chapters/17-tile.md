# 跨 tile 复用带来的新正确性约束

Step 6 把同一组 barrier 复用于多个输出 tile。这样做有一个隐含前提：每个输出 tile 结束时，各 barrier 的 phase parity 必须回到下一轮代码假设的状态。原文参数下，这个条件成立：`K=4096`、`BLK_K=64`，所以每个输出 tile 有 64 个 K iteration。`mma_bar` 使用 64 次，每个 TMA stage barrier 使用 32 次，都是偶数，因此一轮 tile 结束后 barrier 回到初始 parity，下一轮可以把本地 `phase_tma` 和 `phase_mma` 重新设为 0。

```python
assert K % BLK_K == 0, "K must be divisible by BLK_K"
K_TILES = K // BLK_K
assert K_TILES % (2 * PIPE_DEPTH) == 0, (
    "K_TILES must be divisible by 2 * PIPE_DEPTH"
)
```

如果改了 `K`、`BLK_K` 或 `PIPE_DEPTH`，让某个 barrier 在一个输出 tile 内经历奇数轮，那么简单地在下一轮 tile 开头把本地 phase 置零就不再安全。这个问题和上一章的 `phase_mma ^= 1` 一样，本质都是“异步协议状态也是程序状态”。高性能 GEMM 中的很多 bug 不会表现为崩溃，而是表现为偶发错误或 silent corruption。

---

最后一次更新时间：`2026-08-05 16:12:21 CST`

原文链接：[https://fangpin.github.io/gpu-hpc-book/#/chapters/17-tile.md](https://fangpin.github.io/gpu-hpc-book/#/chapters/17-tile.md)
