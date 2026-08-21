# TIRx 基础

GPU 编程中经常遇到的一个现实问题是：到了 Tensor Core、TMEM、barrier、TMA 这一层，真正决定一个 kernel 行为的关键信息，往往散落在 intrinsic 选择、地址计算和线程协作约定里。TIRx 想解决的，不是“再发明一种更高级的 CUDA 语法”，而是把这些分散的信息重新组织成编译器可见、程序员也更容易推理的结构化 IR。

## 为什么是 TIRx

现代 GPU kernel 的难点，从来不只是“把一个算子写出来”。真正麻烦的是，程序里有很多重要决定并不直接写在算法本身里，而是隐含在实现细节中。比如，哪一组线程负责发起某个 tile operation，某个逻辑 tile 在 shared memory 或 tensor memory 里究竟怎么排布，最终又会走哪条硬件数据通路。这些信息在 CUDA 或 PTX 里当然也存在，但往往分散在不同层次的代码细节里，不容易被统一检查和变换。

TIRx 的切入点很直接：把这些决定显式放进 IR。它仍然直接面对 GPU 的硬件概念，例如线程层级、SMEM、TMEM、barrier 和 Tensor Core 指令；不同之处在于，这些概念不再只是“代码里碰巧这样写了”，而是成为编译器可以看见、理解并进一步 lowering 的结构化信息。

## 从一个最小 GEMM 开始，TIRx 在做什么

这个 kernel 计算 `D = A * B^T`，其中 `A` 和 `B` 的 shape 都是 `128x64`，输出 `D` 是 `128x128`。由于它只覆盖一个 `128x128x64` tile，所以 grid 中只有一个 CTA。它的数据路径非常清楚：

```text
A/B: GMEM -> SMEM -> tcgen05.mma
D:   tcgen05.mma -> TMEM -> registers -> GMEM
```

这段例子值得看的地方，不是它做了一个多复杂的矩阵乘，而是它把现代 GPU kernel 的几个关键阶段压缩进了一个最小可运行单元里：

1. 先分配 SMEM 和 TMEM，并准备同步所需的 barrier。
2. 用 `Tx.cta.copy` 把输入 tile 从 GMEM (global memory) 搬到 SMEM (shared memory)。
3. 用 `Tx.gemm_async` 发起真正的 tile GEMM。
4. 再把结果从 TMEM (Tensor memory) 分发到寄存器，最后写回 GMEM。

其中最值得注意的一点是，`Tx.gemm_async` 这类 tile primitive 表达的是一个完整的 tile 级操作，而不是某一条具体的底层指令。编译器会结合 tile 的 shape、layout 和 dispatch 信息，把它展开成真正的硬件序列。例如在这个例子里，完整的 `128x128x64` 计算最终会被 lowering 成多条 `tcgen05.mma` 指令，因为底层的 MMA 沿 K 维每次只推进 16 个元素。

## 理解 TIRx，关键是看三个决定

TIRx 的核心抽象，其实可以压缩成三个问题：谁执行、数据怎么摆、走哪条硬件路径。几乎所有 tile operation 都可以沿着这三个维度来理解。

| 维度 | 它回答的问题 | 在入门 GEMM 里的例子 |
|-|-|-|
| **Scope** | 这件事由谁来做？哪些线程需要参与？ | `Tx.cta.copy` 由整个 CTA 协作完成；`Tx.gemm_async` 由被选中的线程发起；`Tx.wg.copy_async` 由整个 warpgroup 协作执行。 |
| **Layout** | 逻辑 tile 的元素，分别落在什么物理位置？ | A/B 在 SMEM 中采用 128B swizzle；TMEM accumulator 使用 `TLane` 和 `TCol` 描述；寄存器视图用 `tid_in_wg` 分配结果行。 |
| **Dispatch** | 最终由哪条硬件实现路径来完成？ | `Tx.gemm_async` 配合 `dispatch="tcgen05"`，明确选择 Blackwell 的 `tcgen05.mma` 路径。 |

这三个维度之所以重要，是因为它们合在一起，才真正定义了一个 tile operation 的语义。缺了 scope，你不知道谁负责执行；缺了 layout，你不知道 producer 和 consumer 是否指向同一个物理元素；缺了 dispatch，你也无法判断最终会 lower 到哪类硬件指令。TIRx 的价值，正是把这三件事放进统一模型里，而不是让它们继续散落在实现细节里。

### Layout API

如果说 `scope` 和 `dispatch` 比较像“给操作补充上下文”，那么 `Layout API` 则是 TIRx 真正带有方法论色彩的部分。它试图回答一个看似基础、但在现代 GPU 上其实非常复杂的问题：一个逻辑 tile，到底如何映射到具体硬件坐标。

TIRx 用 `TileLayout` 来描述这个问题，最核心的写法是：

```text
TileLayout(S[shape:strides] + R[replica_shape:replica_stride] + offset)
```

其中，`S[...]` 表示 **shard**，负责给出基础映射；`R[...]` 表示 **replica**，负责描述额外的物理副本；而 offset 则是在整体上做统一平移。把三者合在一起，可以写成一个很紧凑的表达式：

```text
L(x) = { D(x) + r + O | r in R }
```

这里的 `D(x)` 是 shard 对逻辑坐标 `x` 生成的基础物理坐标，`r` 是 replica 贡献的额外偏移，`O` 则是固定 offset。这个表达式的重要性在于，它把“逻辑元素在哪里”这件事，从一堆地址计算技巧里抽出来，变成了一个可以明确讨论和组合的对象。

这里还有一个特别值得注意的细节：`replica` 表达的是“一个逻辑元素对应多个物理位置”，也就是 one-to-many 的物理复制；这和把某个维度写成 stride 0 的 many-to-one 逻辑别名并不是一回事。现代 GPU 上，广播和跨 warp window 复用往往真的需要物理副本，TIRx 把这件事当成一等公民。

### Named Axes：Layout 不一定产生线性地址

理解 TIRx Layout API 的另一个关键，是接受一个看似反直觉的事实：layout 的输出不一定是普通线性地址。它完全可以是一组具名硬件坐标，例如 `laneid`、`warpid`、`tid_in_wg`、`TLane`、`TCol`。

这也是为什么文章反复强调 **named axes**。在 TIRx 里，`1@laneid` 和 `1@TLane` 虽然数值相同，但表示的不是同一个物理位置，因为前者是 warp 内线程 lane，后者是 TMEM 的 Lane 方向。轴名本身就是 layout 语义的一部分。

基于这套设计，`layout.apply()` 的工作流程也就很自然了。它先把逻辑坐标 flatten 成线性索引，再按照 shard 的 extents 拆分成各个迭代分量，随后把每个分量按 `ck * sk @ ak` 的形式加到对应 axis 上，最后再加 offset。需要注意的是，`apply()` 返回的是基础坐标加 offset，不会主动把 replica 展开枚举出来；replica 仍然保存在 layout 里，等具体 tile operation 消费。



一旦接受“layout 可以产生具名硬件坐标”这件事，TMEM 的表达就会变得非常自然。一个例子是：

```text
TileLayout(
  S[(2, 128, 112):(112@TCol, 1@TLane, 1@TCol)]
)
```

这个 layout 直接说明：逻辑元素 `(a, l, c)` 会映射到 `TLane = l`、`TCol = 112 * a + c`。它不需要先“伪装成线性地址”，再由读者自己把含义还原成 TMEM 坐标。更重要的是，这里还能看到一个很实用的设计点：TMEM 的布局维度并不需要硬凑成 2 的幂。列维度用 112 也完全成立，只要它符合真实的数据组织方式。

这种表达方式在 block-scaled MMA 的 scale factor 场景里会更有价值。scale factors 往往需要被多个 warp window 读取，因此它们不是简单的一对一映射，而需要显式复制：

```text
TileLayout(
  S[(32, sf_per_mma):(1@TLane, 1@TCol)] +
  R[4:32@TLane]
)
```

这意味着同一组 32 行 scale factor 会被复制到四个 `TLane` 窗口中，分别落在 `0-31`、`32-63`、`64-95` 和 `96-127`。这正是“一个逻辑元素对应多个物理位置”的典型场景，也是为什么前面说 replica 不能简单等同于 stride-0 alias。

###  Swizzle 单独建模

`TileLayout` 表达的是 affine mapping：stride、replica、offset 都是线性的；而 shared-memory swizzle 本质上是 XOR 地址变换，不是 affine 结构。也正因为如此，TIRx 没有强行把 swizzle 塞进 `TileLayout`，而是把它单独建模成 `SwizzleLayout`。当一个普通 tile layout 先产生线性 `m` 轴地址，再需要叠加 shared-memory swizzle 时，就用 `ComposeLayout(swizzle, tile)` 来组合两者。

这个设计的好处在于非常诚实。它没有试图用一种统一对象掩盖两种本质不同的变换，而是承认“affine 布局”和“非 affine 地址置换”就是两类不同问题。

## TIRx 真正重要的地方，不在于语法

回过头看，这两篇文章真正想传达的其实不是“如何记住 TIRx 的几个 API”，而是一个更深的思路：现代 GPU kernel 的关键语义，应该被提升为 IR 里可见、可推理、可 lowering 的对象。`scope`、`layout` 和 `dispatch` 并不是三组平行的术语，而是定义 tile operation 的三个核心维度；`TileLayout` 也不是“换一种写地址计算的方法”，而是在重新定义程序员和编译器如何共同理解数据摆放这件事。

从这个角度看，TIRx 最吸引人的地方并不只是“它能写 Blackwell 上的 kernel”，而是它展示了一条很清晰的路线：当硬件越来越复杂时，好的抽象不是把硬件藏起来，而是把那些真正重要的硬件语义显式化、结构化，并交给编译器系统认真处理。

最后给出使用 TIRx 实现的`128x128x64` GEMM参考实现，应重点理解其中各个部分对应 scope，layout，dispatch中的那一块。

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import (
    tma_shared_layout,
    SwizzleMode,
)
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
import torch


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
        wg_id = T.warpgroup_id(
            [1]
        )  # single warpgroup, so wg_id is always 0 (unused below)
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
            (128, 512),
            "float32",
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)]),
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_mma: T.int32 = 0

        # --- Load: all threads copy global -> shared (synchronous).
        # With M=BLK_M and N=BLK_N the slices below cover the full matrices;
        # the slice form is kept so the diff to Step 3 (multi-tile) is minimal.
        Tx.cta.copy(Asmem[:, :], A[m_st : m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st : n_st + BLK_N, :])
        T.cuda.cta_sync()

        # --- Compute: single elected thread issues MMA ---
        if warp_id == 0:
            if T.ptx.elect_sync():
                Tx.gemm_async(
                    tmem[:, :BLK_N],
                    Asmem[:, :],
                    Bsmem[:, :],
                    accum=False,
                    dispatch="tcgen05",
                    cta_group=1,
                )
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)

        # --- Writeback: TMEM -> RF -> GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(
            128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1 @ tid_in_wg, 1)])
        )
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


target = tvm.target.Target("cuda")
device = torch.device("cuda")  # gpu(0)

M, N, K = 128, 128, 64
kernel = hgemm_v1(M, N, K)
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

torch.cuda.empty_cache()
torch.cuda.synchronize()
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

# ex.mod(...) takes torch tensors directly, the same call form used in every chapter.
ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")

```

---

最后一次更新时间：`2026-08-21 21:11:44 CST`

原文链接：[https://fangpin.github.io/gpu-hpc-book/#/chapters/09-tirx-basics.md](https://fangpin.github.io/gpu-hpc-book/#/chapters/09-tirx-basics.md)
