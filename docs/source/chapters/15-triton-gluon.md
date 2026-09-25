# Triton GLUON

```{contents} 本页目录
---
depth: 2
local: true
---
```

## GLUON 简介

Gluon 是一门建立在 Triton 编译栈之上的 GPU 编程语言。它沿用了 Triton 熟悉的 Python DSL、JIT 编译基础设施和 tile-based SPMD 编程模型，因此写过 Triton kernel 的开发者不需要重新学习一套 host 侧启动协议。真正变化发生在 device code：Triton 通常让编译器决定 tile layout、内存分配、数据搬运与异步执行，Gluon 则把这些选择显式交给开发者。

这项取舍直接决定了它的定位。Triton 适合让编译器在较大的算子范围内生成稳定高效的代码；当性能瓶颈落到布局、搬运路径或流水线细节时，隐藏的决策会限制继续优化的空间。Gluon 允许开发者继续向下控制这些细节，同时也要求开发者理解 CTA、warp、寄存器、共享内存和异步执行之间的关系。

| 维度 | Triton | Gluon |
|-|-|-|
| 编程入口 | Python DSL 与 JIT kernel | Python DSL 与 JIT kernel |
| 并行模型 | tile-based SPMD | tile-based SPMD |
| host 侧启动 | `kernel[grid](...)` | `kernel[grid](...)` |
| tile layout | 多数情况下由编译器管理 | 由开发者显式选择和控制 |
| 内存与数据搬运 | 编译器承担较多决策 | 开发者承担更多决策 |
| 适用目标 | 用较少底层细节获得高性能 | 针对硬件路径继续手工压榨性能 |

这里先从1个很小的 kernel 入手。一维 `memcpy`，用实际带宽暴露“任务已经分块，但 CTA 内部还没有并行布局”的问题。

Gluon 当前位于 Triton 的实验命名空间中。kernel 使用 `@gluon.jit` 声明，设备侧操作从 `triton.experimental.gluon.language` 导入：

```python
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

第一步是按 `XBLOCK` 切分输入。`XBLOCK` 使用 `gl.constexpr` 声明，这意味着它在编译期可知，能够参与特化和后续调优。第 `pid` 个 program 处理从 `pid * XBLOCK` 开始的一段数据，尾部用 `min` 截断，避免越过 `xnumel`。

```python
@gluon.jit
def memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK
    end = min(start + XBLOCK, xnumel)
    for i in range(start, end):
        value = gl.load(in_ptr + i)
        gl.store(out_ptr + i, value)


def memcpy(input, output, XBLOCK):
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK),)
    memcpy_kernel[grid](
        input, output, xnumel, XBLOCK, num_warps=1
    )
```

grid 使用向上取整：

$$N_{CTA}=\left\lceil\frac{xnumel}{XBLOCK}\right\rceil$$

例如 `xnumel=500`、`XBLOCK=64` 时会启动 8 个 CTA。前 7 个 CTA 各处理 64 个元素，最后一个 CTA 处理剩余 52 个元素。任务已经在 CTA 之间切开，但 CTA 内的循环仍逐元素执行。`XBLOCK` 此时只定义了每个 CTA 的任务区间，还没有定义这些元素怎样分布到 CTA 内的线程和寄存器。



Gluon 可以直接复用 Triton 的 autotune 机制。下面把 `XBLOCK` 的候选范围设为 `2^8` 到 `2^13`，并用 `xnumel` 作为调优缓存的 key：

```python
@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 2**i}, num_warps=1)
        for i in range(8, 14)
    ],
    key=["xnumel"],
)
@gluon.jit
def memcpy_kernel_autotune(
    in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr
):
    memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK)


def memcpy_autotune(input, output):
    xnumel = input.numel()

    def grid(META):
        return (triton.cdiv(xnumel, META["XBLOCK"]),)

    memcpy_kernel_autotune[grid](input, output, xnumel)
```

`grid` 接收 `META`，所以每个候选配置都会用自己的 `XBLOCK` 重新计算 CTA 数量。运行下面的命令可以看到最终选择：

```bash
TRITON_PRINT_AUTOTUNING=1 python 01-intro.py
```

在 GB200 上复制 8 GB 输入时，最优候选为 `XBLOCK=2048`，耗时约 24.00 ms，按读写总流量计算得到约 666.24 GB/s：

```text
Time:        24.00 ms
Throughput: 666.24 GB/s
```

这里的带宽计算口径需要说清。示例令 `xnumel = 2 << 30`，即 `2^31` 个 `float32` 元素，输入大小为 8 GiB。一次复制既要读取输入，也要写出结果，因此统计的数据移动量为 16 GiB：

$$B_{traffic}=2\times xnumel\times sizeof(float32)=16\;GiB$$

$$BW=\frac{16\;GiB}{24\;ms}\approx 666.7\;GiB/s$$

它远低于教程给出的 GB200 峰值 8 TB/s，约为峰值的 8.3%。这里更重要的是诊断：改变 `XBLOCK` 确实会改变表现，但没有修复 CTA 内部缺少并行分工这一根本问题。



tile-based SPMD 的关键对象是 tile。tile 是一个 N 维数组，由一个 program 共同处理；layout 进一步说明 tile 中的每个元素由哪些线程负责，以及值放在哪些寄存器位置。只有给数据选择 layout，Gluon 才能把一次标量 `load/store` 扩展成 CTA 内多个线程协作的 tile 访问。

当前 `memcpy_kernel` 有跨 CTA 的并行，每个 CTA 却在循环里逐个复制元素。继续优化时需要构造一维 offset tile，为它选择与 warp 数量和硬件访存模式相匹配的 layout，再用 tile 形式执行 `gl.load` 与 `gl.store`。layout 选择会同时影响线程利用率、地址合并、寄存器占用和可调度的 CTA 数量，因此它既是语义的一部分，也是性能模型的一部分。

这也解释了 Gluon 为什么要求更深的硬件知识。Triton 会替开发者处理大量 layout 推导；Gluon 允许直接干预这一步。额外控制力只有在 layout 与实际硬件路径匹配时才会转化为性能，否则同样可能得到低利用率、过高寄存器压力或不理想的访存指令。



把代码按 Scope、Layout、Dispatch 拆开，可以快速看出性能问题落在哪一层：

| 观察层 | `copy_scalar_kernel` | 当前 `memcpy_kernel` | 下一步的 tile memcpy |
|-|-|-|-|
| Scope | 1 个 program，也就是 1 个 CTA | 多个 CTA，每个 CTA 负责一个 `XBLOCK` 区间 | 多个 CTA，各自负责一个 tile |
| Layout | 标量，没有线程间的数据分布 | 仍是标量循环，没有显式 tile layout | 明确把 tile 元素分配到 warp、lane 和寄存器 |
| Dispatch | 1 次标量 load 和 1 次标量 store | 每个 CTA 重复执行标量 load/store | 让多个线程共同发出可合并的 tile load/store |

Scope 已经通过 grid 把大数组切给多个 CTA。性能缺口集中在 Layout 和 Dispatch：数据还没有在 CTA 内分布，硬件也就无法用合适的并行访存路径处理一个 tile。单纯增大 `XBLOCK` 或扩大 autotune 搜索范围，只是在调整每个 CTA 串行处理多少工作。



下一步的重点已经很明确：把一维标量循环改写成带 layout 的 tile load/store。完成这一步后，`XBLOCK` 才同时具有任务分块和 CTA 内数据分布的含义，性能分析也会从“启动了多少 CTA”进入“每个 CTA 如何使用 warp、寄存器与内存事务”的层次。

## GLUON Tensor Layouts

上一节的 `memcpy` 已经用 grid 把输入切给多个 CTA，却仍让每个 CTA 在循环里逐元素复制。缺少的环节是 layout：它规定一个 tile 里的逻辑元素分别由哪个 warp、哪个 lane 和该 lane 的哪个寄存器持有。Gluon 要求 tensor 携带 layout，正是为了让这种映射成为程序语义的一部分。

显式 layout 带来两件事。开发者可以按照全局内存的连续方向组织线程访问，也可以让归约、扫描和矩阵乘法看到更合适的数据分布；与此同时，布局选错会直接表现为访存不合并、跨线程通信增多、共享内存占用上升或寄存器压力过大。



Gluon 按 GPU 的执行层级分配 tensor：先确定 thread block，也就是 CTA；再确定 CTA 内的 warp；接着落到 warp 内的 lane；最后落到每个 lane 的寄存器。可以把 layout 写成下面这类映射：

$$(warp, lane, register)\longrightarrow tensor\ index$$

同一个 tensor 的元素会均匀分给所有线程，因此每个线程拥有相同数量的元素。Triton tile 的各维大小要求是 2 的幂，这进一步保证每线程持有的元素数也是 2 的幂。

这里要区分逻辑形状和物理分布。tensor shape 描述程序看到的 N 维索引空间；layout 描述这些索引由硬件线程和寄存器怎样承载。两个 tensor 可以有相同 shape，却因为 layout 不同而生成完全不同的 load、store、shuffle 或 shared-memory 指令。



`BlockedLayout` 是 Gluon 最常用的布局。下面这个二维例子同时给出了线程内、warp 内和 CTA 内的分块方式：

```python
gl.BlockedLayout(
    size_per_thread=[2, 4],
    threads_per_warp=[16, 2],
    warps_per_cta=[2, 2],
    order=[1, 0],
)
```

| 参数 | 含义 | 当前取值对应的结构 |
|-|-|-|
| `size_per_thread` | 每个线程在各维持有的元素数 | 每个线程持有一个 `2×4` 子块，共 8 个值 |
| `threads_per_warp` | 一个 warp 的线程怎样铺到各维 | `16×2=32` 个 lane |
| `warps_per_cta` | 一个 CTA 的 warp 怎样铺到各维 | `2×2=4` 个 warp |
| `order` | 各维从内到外的铺设顺序 | `[1, 0]` 对应 row-major 铺设 |

三层形状逐维相乘，就得到 layout 的基本 block shape：

$$S_{block}=S_{thread}\odot S_{lane}\odot S_{warp} =[2\times16\times2,\;4\times2\times2]=[64,16]$$

`size_per_thread=[2, 4]` 表示每个线程拥有 8 个寄存器值。`order=[1, 0]` 时，寄存器编号先沿内层维度增加：

```text
[[T:0, T:1, T:2, T:3],
 [T:4, T:5, T:6, T:7]]
```

改成 `order=[0, 1]` 后，寄存器编号改为先沿另一维增加：

```text
[[T:0, T:2, T:4, T:6],
 [T:1, T:3, T:5, T:7]]
```

`threads_per_warp=[16, 2]` 再把 32 个 lane 组织成 `16×2` 的 warp tile。把每个 lane 替换成它自己的 `2×4` 寄存器子块，就得到一个 warp 覆盖的逻辑元素。`warps_per_cta=[2, 2]` 用相同方法继续扩展，最终形成整个 CTA 的 `[64,16]` block。

### Tensor 比 block 大时平铺，比 block 小时广播

当 tensor shape 与 block shape 相同，元素直接按 block layout 分布。形状不同时，Gluon 会平铺 block 或广播较小的 tensor。两种情况都会影响实际寄存器用量。

对于 `128×128×f32` tensor，`[64,16]` block 需要沿两维分别重复 `[2,8]` 次，共 16 个 block 副本。每个线程在一个 block 中持有 8 个值，因此整个 tensor 让每个线程持有：

$$R_{thread}=8\times2\times8=128$$

这 128 个值会形成很高的寄存器压力。kernel 设计不能只看 tile shape，还要算清每个线程最后承担多少寄存器值。

较小的 tensor 会触发广播。以 `32×8×f32` 为例，它只有 256 个逻辑元素，却要适配 `[64,16]` block。`tensor shape / block shape` 在两维上都对应 2 倍的广播关系，恰好匹配 `warps_per_cta=[2,2]`，于是四个 warp 各自持有一份 tensor 副本。整个 program 因此占用 1024 个标量寄存器槽位。逻辑元素少，不代表物理存储一定少。

### memcpy 的 tile layout 开始参与执行

带 layout 的一维复制不再用 Python 风格循环逐元素访问，而是先生成带布局的索引 tile。Gluon 的类型推导会把 `indices` 的 layout 向后传播到 offsets、pointer tensor、mask 和 load 结果，因此通常只需在数据流入口指定一次：

```python
@gluon.jit
def memcpy_1d_kernel(
    in_ptr, out_ptr, xnumel,
    XBLOCK: gl.constexpr, layout: gl.constexpr
):
    pid = gl.program_id(0)
    start = pid * XBLOCK

    indices = gl.arange(0, XBLOCK, layout=layout)
    offsets = start + indices
    mask = offsets < xnumel

    value = gl.load(in_ptr + offsets, mask=mask)
    gl.store(out_ptr + offsets, value, mask=mask)
```

当 `num_warps=4` 时，一维 blocked layout 可以写成：

```python
gl.BlockedLayout(
    size_per_thread=[R],
    threads_per_warp=[32],
    warps_per_cta=[4],
    order=[0],
)
```

这里共有 128 个线程，每线程持有 `R` 个元素，基本 block 包含 `128R` 个元素。沿用上一节的 `XBLOCK=2048`，为了避免 layout 的基本 block 大于 tile 并产生冗余值，`R` 最大取 16。

GB200 上的测量说明 layout 已经直接影响吞吐：

| R | 吞吐量 |
|-|-|
| 1 | 6.574 TB/s |
| 2 | 6.476 TB/s |
| 4 | 6.474 TB/s |
| 8 | 6.502 TB/s |
| 16 | 6.214 TB/s |

与上一节约 666.24 GB/s 的标量循环相比，tile 化后已经进入 6 TB/s 以上的区间。不同 `R` 之间仍有明显差异，说明“让所有线程参与”只是第一步，线程各自持有多少连续元素同样会改变指令形态。



检查生成的 SASS，可以看到 `R` 同时改变 load/store 的向量宽度、指令数量与地址步长：

| R | width | vec_len | `LDG/STG` 次数 | stride |
|-|-|-|-|-|
| 1 | 32 | 32 | 1 | `0x00` |
| 2 | 64 | 64 | 1 | `0x00` |
| 4 | 128 | 128 | 1 | `0x00` |
| 8 | 256 | 128 | 2 | `0x10` |
| 16 | 512 | 128 | 4 | `0x10` |

现代 NVIDIA GPU 的 cache line 是 128 byte，并拆成 4 个 32-byte sector。全局内存以 sector 为访问粒度，warp 的连续地址会被合并。`R=1` 时，一个 warp 的 `LDG.E` 恰好读取连续 128 byte，正好覆盖一个 cache line；PyTorch 分配的 tensor 又按 256 byte 对齐，边界条件也比较理想。

`R` 增加到 2 或 4 后，单条指令变宽，但访问的 32-byte sector 数没有减少，吞吐反而略降。SASS 左侧的调度注释给出了一条线索：

```text
wait_mask : read_barrier : write_barrier : yield : stall
```

load 指令向寄存器写值，因此会设置 `write_barrier`；依赖这些寄存器的后续 `STG.E` 通过 `wait_mask` 等待 barrier 清除。较小粒度的 load 可能让一部分 store 更早开始执行。不过，教程明确指出，仅靠这段 SASS 还不能解释所有现象，例如 `R=8` 在特定配置下为什么快于 `R=2` 和 `R=4`，仍需 profiler 证据。

联合扫描 `XBLOCK` 与 `R` 后，当前测试中最好的组合是 `XBLOCK=8192、R=1`，吞吐为 6.606 TB/s。它同时说明最优 layout 会受到 tile 大小影响，不能把某个单独参数的局部最优直接推广到所有配置。



二维 strided tensor 的地址由行列索引共同决定：

$$offset=x\times stride_x+y\times stride_y$$

`indices_x` 和 `indices_y` 都是一维 tensor，但相加广播后要形成同一个二维 layout。`SliceLayout` 从父 layout 中删去一个维度，让一维索引保留与二维结果兼容的 lane 和寄存器映射：

```python
indices_x = start_x + gl.arange(
    0, XBLOCK,
    layout=gl.SliceLayout(dim=1, parent=layout),
)
indices_y = start_y + gl.arange(
    0, YBLOCK,
    layout=gl.SliceLayout(dim=0, parent=layout),
)

in_offsets = (
    xstride_in * indices_x[:, None]
    + ystride_in * indices_y[None, :]
)
```

`indices_x[:, None]` 和 `indices_y[None, :]` 扩维后恢复父 layout，再广播成 `[XBLOCK,YBLOCK]`。这类广播发生在虚拟寄存器映射层，只复制元素归属关系，不需要真的搬运数据，因此可以是零成本操作。



对连续二维 tensor，可令 `XBLOCK=1、YBLOCK=2048`，并使用下列 layout，让 32 个 lane 沿连续的内层维度取数：

```python
gl.BlockedLayout(
    [1, 1], [1, 32], [1, 4], [1, 0]
)
```

GB200 上该二维 memcpy 达到 6.260 TB/s，只比一维版本慢约 5%。如果保持同一个 layout，却把 tensor 转置，内层维度不再连续，warp 访问无法合并，吞吐会降到 0.774 TB/s。交换 block 方向并改用下面的 layout 后，连续访问转到另一维，吞吐恢复到 6.590 TB/s：

```python
gl.BlockedLayout(
    [1, 1], [32, 1], [4, 1], [0, 1]
)
```

| 场景 | layout 与连续维度是否一致 | 吞吐量 |
|-|-|-|
| 连续二维 tensor | 一致 | 6.260 TB/s |
| 转置 tensor，沿用原 layout | 不一致 | 0.774 TB/s |
| 转置 tensor，同时交换 block 与 layout | 一致 | 6.590 TB/s |

三种情况的差别来自同一个原则：相邻 lane 应尽量访问同一组连续 sector。前后两个高性能版本在单个 program 内产生相同的内存访问形态，剩余差异可能来自 program 在 GPU 上的调度位置、TLB 和分区式 L2 cache 的局部性。

一维 memcpy 的表现更稳定，但要求输入和输出都能视为连续内存。二维实现可以保留显式 stride，因此更适合非连续 view。教程用一个“每隔一行取一次”的 8 GB 输入测试 `contiguous()` 场景，得到下面的结果：

| 实现 | 吞吐量 |
|-|-|
| 2D memcpy | 6.258 TB/s |
| `torch.Tensor.contiguous()` | 2.946 TB/s |
| 转置视角下的 2D memcpy | 6.398 TB/s |

这些结果来自 GB200 上的教程环境。它们说明显式 stride 与匹配的 layout 可以在非连续输入上保持合并访存，不代表所有 PyTorch 版本和 GPU 都会得到相同比例。



layout 还决定归约需要多少跨线程通信。对 `128×128×f32` tensor 沿内层维度归约时，如果使用面向全局内存连续访问的布局：

```python
gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
```

同一行的元素分散在不同线程中。编译器会先用 butterfly shuffle 在 warp 内归约，再选择 leader warp，通过 shared memory 合并每行剩余的 4 个部分结果。

如果改用下面的布局，每个线程恰好拥有完整一行，归约就不需要线程间通信：

```python
gl.BlockedLayout([1, 128], [32, 1], [4, 1], [0, 1])
```

这并不意味着应该为每个归约都先转换 layout。Gluon 编译器通常能针对不同输入布局生成有效的 reduction 和 scan；如果先转换再归约，转换成本可能高于省下的通信。只有多个候选 layout 获取成本相近时，归约方向才应成为选择依据。

shared memory 访问还同时受 shared-memory layout 和 register layout 影响。shared memory 按 bank 组织，一个 warp 在一个周期内访问同一 bank 的不同地址会产生冲突。编译器会尽量减少 bank conflict，但最终冲突数量仍取决于两侧布局怎样组合。



Gluon 没有唯一的 canonical layout。不同对象可以表达相同的元素映射，例如下面两个 layout 等价：

```python
gl.BlockedLayout([1], [32], [4], [0])
gl.SliceLayout(
    1,
    gl.BlockedLayout([1, 1], [32, 1], [4, 1], [1, 0]),
)
```

已知转换前后映射等价，或只需在线程内部重排寄存器时，可以加上 `assert_trivial=True`：

```python
y = gl.convert_layout(x, target_layout, assert_trivial=True)
```

这个断言把性能假设变成编译期约束。一旦转换需要跨线程通信，编译就会暴露问题，避免一次原本预期为零成本的操作悄悄使用 shuffle 或 shared memory。

所有 Gluon layout 最终都可以表示成 linear layout。前面的一维映射可写成：

```python
gl.DistributedLinearLayout(
    reg_bases=[],
    lane_bases=[[1], [2], [4], [8], [16]],
    warp_bases=[[32], [64]],
    block_bases=[],
    shape=[128],
)
```

它相当于 tensor 一维索引 bit 上的 `7×7` 单位映射：低 5 bit 选择 32 个 lane，高 2 bit 选择 4 个 warp。linear layout 可以统一表达 split、join、reshape 和 permute 等零成本变换，也能处理 5D、7D 等高维 tensor。代价是表示更接近位级映射，阅读和手工构造都比 `BlockedLayout` 困难。
