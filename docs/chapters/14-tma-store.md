# TMA store：写回也变成异步协议

Step 4 不只替换 load，也把输出写回改成 TMA store。写回路径变成 `TMEM -> register -> Dsmem -> GMEM`：先从 TMEM 读出 fp32 accumulator 到寄存器，cast 成 fp16，写入 shared memory 中的 `Dsmem`，最后让 TMA 把整个输出 tile 从 SMEM 搬回 GMEM。

```python
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
T.cuda.cta_sync()

Tx.cast(Dreg_f16[:], Dreg[:])
Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
T.ptx.fence.proxy_async("shared::cta")
T.cuda.warpgroup_sync(10)

if tid == 0:
    Tx.copy_async(
        D[m_st:m_st + BLK_M, n_st:n_st + BLK_N],
        Dsmem[:, :],
        dispatch="tma"
    )
    T.ptx.cp_async.bulk.commit_group()
    T.ptx.cp_async.bulk.wait_group(0)
```

TMA load 和 TMA store 的等待机制不一样。load 通过 mbarrier 报告完成，因为消费者是后续 MMA，需要知道 SMEM 中的 A/B tile 何时可读。store 使用 `commit_group()` 和 `wait_group(0)`：前者把已发起但尚未提交的 TMA store 收进一个 bulk async group，后者要求之前提交的 group 全部完成。这里的 `0` 意味着不能留下任何 pending group；在它返回前，`Dsmem` 不能被覆盖或复用。

这个版本已经把数据搬运切到硬件路径上，但调度仍然保守：每轮 K-loop 发起 TMA load，马上等待，等待完成后才执行 MMA。换句话说，Step 4 建立的是正确的异步协议，不是完整的重叠执行。

---

最后一次更新时间：`2026-08-05 16:12:21 CST`
