# Step 4：TMA 接管 GMEM 到 SMEM 的搬运

TMA，全称 Tensor Memory Accelerator，是 Blackwell/Hopper 这类现代 NVIDIA GPU 中用于 tile 级数据搬运的硬件路径。线程不再逐元素或逐向量计算地址、发出 load/store，而是由一个线程发起 tile copy，TMA engine 负责后续地址生成和数据搬运。

在 TIRx 里，这个变化非常集中：`Tx.cta.copy` 变成 `Tx.copy_async(..., dispatch="tma")`。同时，异步 load 的完成不能靠 `cta_sync()` 判断，因为 `cta_sync()` 只同步 CTA 线程本身，不知道 TMA engine 是否已经把数据搬完。因此 Step 4 引入了一个 TMA mbarrier。

```python
tid = T.meta_var(warp_id * 32 + lane_id)

if tid == 0:
    Tx.copy_async(
        Asmem[:, :],
        A[m_st:m_st + BLK_M, k_st:k_st + BLK_K],
        dispatch="tma", cta_group=1, mbar=tma_bar.ptr_to([0])
    )
    Tx.copy_async(
        Bsmem[:, :],
        B[n_st:n_st + BLK_N, k_st:k_st + BLK_K],
        dispatch="tma", cta_group=1, mbar=tma_bar.ptr_to([0])
    )
    T.ptx.mbarrier.arrive.expect_tx(
        tma_bar.ptr_to([0]),
        (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE
    )

T.ptx.mbarrier.try_wait(tma_bar.ptr_to([0]), phase_tma)
```

这里有两个容易忽略的细节。第一，`tid == 0` 是为了确保只有一个线程发起 TMA。如果直接在每个 warp 内调用 `elect_sync()`，四个 warp 会各选出一个 lane，结果变成四个线程重复发起 copy。第二，`arrive.expect_tx` 不是普通的 arrival，它同时告诉 barrier：除了发起线程到达之外，还有多少字节的异步传输需要完成。

以原文参数为例，A tile 和 B tile 都是 `128 x 64` 的 fp16，总字节数是 `(128 * 64 + 128 * 64) * 2 = 32768`。mbarrier 的状态可以理解成两个条件：线程 arrival 归零，TMA pending bytes 也归零。只有这两个条件都满足，后面的 `try_wait` 才能通过，MMA 才能安全读取 SMEM。

**图示参考：**[TMA Async Load: Synchronization Flow](https://mlc.ai/modern-gpu-programming-for-mlsys/_images/tma_sync_flow.svg) 展示了 issuing thread、TMA engine、mbarrier 和 MMA 之间的交接关系。为了避免飞书创建时远程图片拉取超时，这里保留为可点击原图链接。

---

最后一次更新时间：`2026-08-05 16:12:21 CST`
