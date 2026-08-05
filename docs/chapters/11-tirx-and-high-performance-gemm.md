# TIRx 与高性能GEMM

GEMM 是现代 MLSys 里最值得认真对待的基础 kernel。线性层、attention projection，以及不少 convolution lowering 之后的算子，最终都会把 GPU 时间花在矩阵乘上。也正因为它重要，GEMM 的优化不能一开始就把 TMA、pipeline、warp specialization、cluster 和 Tensor Core 调度全部混在一起。更稳妥的路线，是先建立一个可信的正确版本，再让每一轮优化只改变一个明确的契约。

## 从 GEMM 公式到数据路径

对于矩阵乘 $D = A B^\top$。其中 `A` 的形状是 `M x K`，`B` 的形状是 `N x K`，输出 `D` 的形状是 `M x N`。这里的转置不是额外做一次数据搬运，而是来自权重矩阵的存储方式：`B` 被组织成 `N` 行、每行长度为 `K`，沿着 `K` contraction 时，自然就是在计算 `A @ B.T`。

真正决定 kernel 写法的，不只是数学公式，而是数据会经过哪些硬件层级。Blackwell 上的基础 GEMM 路径可以概括为：operand tile 从 GMEM 进入 SMEM，Tensor Core MMA 从 SMEM 读取 A/B 并把累加结果写入 TMEM，最后 epilogue 再把 TMEM 结果读到寄存器，转换精度后写回 GMEM。

| 阶段 | 数据移动 | TIRx / PTX 关键点 | 正确性约束 |
|-|-|-|-|
| 加载 operand | GMEM -> SMEM | `Tx.cta.copy` | 所有线程写完 SMEM 后，MMA 才能读取完整 tile。 |
| 执行 MMA | SMEM -> TMEM | `Tx.gemm_async`，`dispatch="tcgen05"` | 只有一个 elected issuer 发起 tile op，但硬件执行的是 cooperative MMA。 |
| 等待完成 | TMEM accumulator readiness | `tcgen05.commit` 与 `mbarrier.try_wait` | 复用 barrier 时必须维护 phase，否则 wait 可能过早返回。 |
| 写回输出 | TMEM -> register -> GMEM | `Tx.wg.copy_async`，`tcgen05.wait.ld`，`Tx.copy` | TMEM load 完成后才能读寄存器，并在写回前完成 fp32 到 fp16 的 cast。 |

## 第一步：单 tile GEMM，先把完整路径跑通

最小但仍然覆盖完整硬件路径的版本，是计算一个 `128 x 128` 输出 tile，并让 `K = 64`。这个版本没有 K-loop，也没有二维 grid；它只是让一个 warpgroup 顺序完成分配、加载、MMA、等待、写回和释放。它的价值不在性能，而在于它把后续所有优化都会复用的数据路径固定下来。

**共享内存与 TMEM 元数据分配。**Kernel 首先用 `T.SMEMPool` 分配 TMEM 地址槽、mbarrier，以及 A/B 两个 shared-memory tile。`pool.move_base_to(1024)` 把大块 operand tile 放到更干净的地址边界上，而 `tma_shared_layout` 生成的 layout 则让 SMEM 中的数据能被后续 TMA 和 `tcgen05.mma` 直接消费。

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg

pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")
mma_bar = pool.alloc((1,), "uint64", align=8)
pool.move_base_to(1024)
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
pool.commit()
```

**线程驱动的同步加载。**第一版没有使用 TMA，而是让 CTA 内线程合作把 A/B 从 GMEM 拷到 SMEM。这里的 `T.cuda.cta_sync()` 不能省略：它既等待所有线程完成拷贝，也保证这些 shared-memory 写入对后续 MMA 可见。

```python
Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
T.cuda.cta_sync()
```

**一个线程发起，不等于一个线程计算。**MMA dispatch 由 warp 0 中一个被 `T.ptx.elect_sync()` 选出的 lane 发起。这一点很容易误读：单个 issuer 只是在发起 tile operation，真正的乘加仍然由硬件以 cooperative 方式完成。`Tx.gemm_async` 会根据 operand 与 accumulator 的 tile 形状 lowering 到一组 `tcgen05.mma` 指令；如果 128 个线程都发起同一个 op，只会把同一份工作重复提交 128 次。

```python
if warp_id == 0:
    if T.ptx.elect_sync():
        Tx.gemm_async(
            tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
            accum=False, dispatch="tcgen05", cta_group=1
        )
        T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
```

**从 TMEM 到寄存器，再写回 GMEM。**MMA 结果先以 fp32 accumulator 形式留在 TMEM。写回时，warpgroup 通过 `Tx.wg.copy_async` 把一个 `128 x 128` tile 读到寄存器视图中，再等待 `tcgen05.wait.ld()`。每个线程持有输出 tile 的一行，完成 fp32 到 fp16 的转换后写入自己的全局输出行。

```python
Dreg = T.alloc_local((BLK_N,), acc_type)
Dreg_f16 = T.alloc_local((BLK_N,), d_type)
Dreg_wg = Dreg.view(
    128, BLK_N,
    layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)])
)

Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
Tx.cast(Dreg_f16[:], Dreg[:])
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

完整的kernel

```python
def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    # MMA_M/MMA_N/MMA_K document the underlying hardware MMA tile; they are not
    # passed to gemm_async (which derives the MMA shape from the operand and
    # accumulator tiles), so the later steps omit them.
    MMA_M, MMA_N, MMA_K = 128, 128, 16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # Step 1 is a single-tile kernel: M = BLK_M and N = BLK_N, so the grid
        # is 1x1. Starting with a 1x1 grid keeps the per-CTA tile offsets
        # (m_st, n_st) trivially zero; Steps 3+ generalise this to larger M / N.
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])      # single warpgroup, so wg_id is always 0 (unused below)
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
    
        # --- SMEM allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()
    
        # --- Barrier + TMEM init (warp 0 only) ---
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
    
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
    
        tmem = T.decl_buffer(
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )
    
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_mma: T.int32 = 0
    
        # --- Load: all threads copy global -> shared (synchronous).
        # With M=BLK_M and N=BLK_N the slices below cover the full matrices;
        # the slice form is kept so the diff to Step 3 (multi-tile) is minimal.
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()
    
        # --- Compute: single elected thread issues MMA ---
        if warp_id == 0:
            if T.ptx.elect_sync():
                Tx.gemm_async(
                    tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                    accum=False, dispatch="tcgen05", cta_group=1
                )
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
    
        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
    
        # --- Writeback: TMEM -> RF -> GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
    
        # --- Deallocate TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

这个版本的限制也非常清楚：它只能处理一个 K tile、一个输出 tile；数据加载是同步的，compute 与 memory movement 没有重叠。但它已经建立了最重要的 baseline：GMEM、SMEM、TMEM、register、GMEM 之间的路径全部跑通，而且每个同步点都有明确原因。

## 第二步：加入 K-loop，真正的难点是 barrier phase

真实矩阵的 K 通常远大于 64。第二步不改变输出 tile 的空间范围，只把 K 拆成多个 `BLK_K=64` 的 chunk：每轮加载下一片 A/B，发起一次 MMA，并把结果累加进同一个 TMEM accumulator。

这里的关键开关是 `accum`。第一轮 `accum=False`，表示覆盖 TMEM accumulator；之后每轮 `accum=True`，表示把当前 K chunk 的乘积加到已有部分和上。数学上这是普通的 dot product，硬件上则是一次次复用同一个 accumulator slot。

```python
def hgemm_v2(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])  # still one output tile (M=N=128)
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # === K-loop: iterate over K in chunks of BLK_K ===
        for i in T.serial(K_TILES):   # serial device loop (keeps the full-K A/B parameters correctly shaped)
            # Load the i-th K chunk
            Tx.cta.copy(Asmem[:, :], A[:, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[:, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            # MMA: accum=False for first tile, True for rest
            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            # Wait for MMA, then flip phase
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # === Writeback (same as Step 1) ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

其中值得注意的是对 mbarrier phase的管理  `phase_mma ^= 1` 。mbarrier 的 phase 是 1 bit 状态，每次期望的 arrival 到达后都会翻转。`try_wait(bar, phase)` 等待的不是“barrier 等于 phase”，而是等待 barrier 离开当前 phase。因此本地变量 `phase_mma` 表示的是“我正在等待它离开的旧 phase”。

| K 迭代 | wait 前的 `phase_mma` | 等待条件 | wait 后更新 |
|-|-|-|-|
| 0 | 0 | barrier 从 0 翻到 1 | `phase_mma = 1` |
| 1 | 1 | barrier 从 1 翻到 0 | `phase_mma = 0` |
| 2 | 0 | barrier 从 0 翻到 1 | `phase_mma = 1` |

如果忘了这次 phase flip，第二轮仍会调用 `try_wait(bar, 0)`。但第一轮结束后 barrier 已经处在 phase 1，此时 wait 看到“当前 phase 不等于 0”就会立即返回，哪怕第二轮 MMA 还没完成。这个 bug 很危险：它不会阻止 kernel 编译，也不一定触发显式异常，只会让结果 silently wrong。

## 第三步：空间 tiling，把单 CTA 扩展成二维 grid

K-loop 解决的是 contraction 维度，M/N 方向仍然只覆盖一个 `128 x 128` 输出 tile。第三步把 grid 扩展为 `[M // BLK_M, N // BLK_N]`，让每个 CTA 负责一个输出 tile。这样，CTA `(bx, by)` 负责输出区域 `D[bx*BLK_M:(bx+1)*BLK_M, by*BLK_N:(by+1)*BLK_N]`。

```python
bx, by = T.cta_id([M // BLK_M, N // BLK_N])
m_st = T.meta_var(bx * BLK_M)
n_st = T.meta_var(by * BLK_N)

for i in T.serial(K_TILES):
    Tx.cta.copy(
        Asmem[:, :],
        A[m_st:m_st + BLK_M, i*BLK_K:(i+1)*BLK_K]
    )
    Tx.cta.copy(
        Bsmem[:, :],
        B[n_st:n_st + BLK_N, i*BLK_K:(i+1)*BLK_K]
    )
```

这里的索引关系正好对应 `D = A @ B.T`：`bx` 选择 A 的 row band，也就是 D 的 row band；`by` 选择 B 的 row band，而这些 B row 在转置语义下会成为 D 的 column band。kernel 的内部 SMEM/TMEM/register 路径并没有变，变的是 CTA scope 与每个 CTA 看到的全局切片。

一步到这里，kernel 已经能覆盖完整矩阵，但还没有真正利用跨 CTA 的数据复用。同一行 CTA 会反复从 GMEM 加载相同 A tile，同一列 CTA 会反复加载相同 B tile。原文把这个浪费留给后续章节处理：TMA、software pipeline、persistent scheduling、warp specialization 和 CTA cluster 都是在这个正确 baseline 上继续降低数据移动成本、提高硬件占用和 compute density。

| 阶段 | Scope 变化 | 复用对象 | 新增正确性契约 | 仍未解决的问题 |
|-|-|-|-|-|
| Step 1: 单 tile | 一个 CTA / 一个 warpgroup | 无循环复用 | `cta_sync`、mbarrier、`wait.ld` | 只支持一个 K tile 和一个输出 tile。 |
| Step 2: K-loop | 仍然是单输出 tile | 复用 SMEM tile buffer 与 TMEM accumulator | `accum` 标志与 barrier phase flip | M/N 仍然没有空间 tiling。 |
| Step 3: 2D grid | 每个 CTA 负责一个输出 tile | 每个 CTA 内部复用同一路径 | 正确计算 `m_st`、`n_st` 与写回行列 | 相邻 CTA 的 A/B tile 复用尚未利用。 |

---

最后一次更新时间：`2026-08-05 14:01:27 CST`
