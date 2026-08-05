# 从顺序 GEMM 到异步 GEMM 的瓶颈

上一章的基础 kernel 已经能覆盖完整矩阵：每个 CTA 负责一个 `128 x 128` 输出 tile，在 K-loop 中反复把 A/B 的 `128 x 64` chunk 搬到 SMEM，然后发起 `tcgen05` MMA，最后从 TMEM 读回并写出结果。这个版本的好处是清晰，坏处也很直接：load 和 compute 完全串行。

```python
Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])
Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])
T.cuda.cta_sync()

if tid == 0:
    Tx.gemm_async(
        tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
        accum=(i != 0), dispatch="tcgen05", cta_group=1
    )
```

这段代码里有两个问题。第一，GMEM 到 SMEM 的地址生成和搬运由 CTA 线程执行，线程要为 copy 付出指令和同步成本。第二，即使 copy 完成，MMA 也必须等在后面；下一轮 K tile 的 load 不能提前开始，因为只有一份 `Asmem` / `Bsmem`，提前 load 会覆盖当前 MMA 还在读取的数据。

这一章的三个步骤分别处理这些问题。Step 4 用 TMA 替换线程 copy；Step 5 给 A/B operand 各增加 `PIPE_DEPTH=2` 的 stage 维度，建立双缓冲；Step 6 让固定数量的 CTA 在 kernel 内持续领取 tile，从而摊薄初始化成本并改善 operand 在 L2 中的复用机会。

| 步骤 | 主要变化 | 解决的问题 | 还没解决的问题 |
|-|-|-|-|
| Step 4: TMA Async Load | GMEM -> SMEM load 由 TMA engine 执行 | 减少 CTA 线程执行 copy 的负担，建立异步 load/store 协议 | 仍然每次 load 完就立刻 wait，load 与 MMA 还不重叠 |
| Step 5: Software Pipeline | SMEM operand 变成双缓冲 ring | 为 prefetch 和后续 overlap 提供独立存储 stage | 单 warpgroup 版本仍在 MMA 后发起下一次 TMA |
| Step 6: Persistent Kernel | 固定数量 CTA 通过 scheduler 处理多个输出 tile | 摊薄初始化成本，并让 tile 顺序更 L2-friendly | 真正的 producer/consumer 并行要到下一章的 warp specialization |

---

最后一次更新时间：`2026-08-05 16:12:21 CST`
