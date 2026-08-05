# Step 5：双缓冲 SMEM stage 是 pipeline 的前提

如果只有一份 `Asmem` 和 `Bsmem`，下一轮 TMA load 不能提前启动，因为它会覆盖当前 MMA 正在读取的 operand tile。Step 5 解决这个存储冲突：给 A/B 的 SMEM buffer 增加一个 `PIPE_DEPTH` 维度，`PIPE_DEPTH=2` 时就是双缓冲。

```python
PIPE_DEPTH = 2

tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)

for s in range(PIPE_DEPTH):
    T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
```

双缓冲后，kernel 的结构变成 ring buffer。启动阶段先 prefetch 前两个 K tile；主循环中，`stage = k % PIPE_DEPTH` 选择当前要消费的 stage。等当前 stage 的 TMA load 完成后，在这个 stage 上执行 MMA；MMA 完成后，如果还存在 `k + PIPE_DEPTH` 这个未来 tile，就把刚刚消费完的 stage 复用给下一次 prefetch。

```python
if tid == 0:
    for s in range(min(PIPE_DEPTH, K_TILES)):
        tma_load(s, s * BLK_K)

for k in range(K_TILES):
    stage = k % PIPE_DEPTH

    T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)

    if tid == 0:
        mma(stage, accum=(k != 0))

    T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
    phase_mma ^= 1

    next_k = k + PIPE_DEPTH
    if next_k < K_TILES:
        if tid == 0:
            tma_load(stage, next_k * BLK_K)

    if stage == PIPE_DEPTH - 1:
        phase_tma ^= 1
```

这里的 `phase_mma` 和 `phase_tma` 很容易混淆。`mma_bar` 只有一个，因为所有 K chunk 都累加到同一个 TMEM accumulator；每轮 MMA 都会让它完成一轮，因此 `phase_mma` 每次 K iteration 都翻转。`tma_bar` 则是每个 stage 一个：stage 0 的 barrier 只在 ring 回到 stage 0 时进入下一轮，stage 1 也是如此。因此在 `PIPE_DEPTH=2` 的情况下，只有当 `stage == 1`、ring 即将回绕时，统一的 `phase_tma` 才翻转。

| 变量 | 跟踪对象 | 翻转频率 | 错误后果 |
|-|-|-|-|
| `phase_mma` | 同一个 TMEM accumulator 的 MMA 完成 barrier | 每个 K tile 翻转一次 | 可能在 MMA 未完成时读取 accumulator，产生 silent wrong result |
| `phase_tma` | SMEM ring 中每个 stage 的 TMA load barrier | 每轮 ring wrap 翻转一次 | 可能提前消费未完成的 TMA load，或永远等不到下一轮 |

**图示参考：**[Target schedule with PIPE_DEPTH=2](https://mlc.ai/modern-gpu-programming-for-mlsys/_images/pipe_depth2.png) 展示了双缓冲希望支持的目标节奏：一个 stage 被 MMA 消费时，另一个 stage 可以承接未来的 TMA load。原文也强调，当前单 warpgroup 版本还没有完全达到这个重叠；真正的 producer/consumer 角色拆分在下一章完成。

---

最后一次更新时间：`2026-08-05 16:12:21 CST`
