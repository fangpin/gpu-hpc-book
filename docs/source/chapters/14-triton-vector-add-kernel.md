# Triton Vector Add：从一个向量加法 kernel 看懂编程模型

```{contents} 本页目录
---
depth: 2
local: true
---
```

Triton 的入门示例从向量加法开始：给定两个长度相同的一维张量 `x` 和 `y`，计算 `z = x + y`。这个计算本身不复杂，但它正好暴露 Triton 编程模型的几个核心问题：kernel 如何定义，program 如何映射到数据块，为什么需要 mask，元参数如何在 JIT 编译期参与 shape 推导，以及如何用 benchmark 判断自定义算子的真实吞吐。

这篇文章围绕官方 Vector Addition tutorial 展开。重点不是把 Python 代码逐行翻译，而是把一个最小 Triton kernel 拆成 Scope、Layout、Dispatch 三个层次：Scope 说明一个 Triton program 负责哪段数据，Layout 说明 lane 看到的 offsets 如何形成向量化访问，Dispatch 说明 Python 调用如何变成 JIT 编译后的 GPU kernel。

> 核心观点：Triton 的 `program` 不是 CUDA thread，而是更粗粒度的 SPMD 实例。一个 program 通常处理一个 block 的元素，block 内部用 `tl.arange` 形成向量化 lanes；边界由 mask 保护，launch grid 决定需要多少个 program 覆盖整个张量。

## 问题本身很小，但模型很完整

向量加法的数学形式是：

$$z_i=x_i+y_i,\quad 0\le i<n$$

每个元素只做一次加法，计算量很低，主要成本来自内存访问。对 `float32` 输入来说，每个元素需要读 `x_i` 4 字节、读 `y_i` 4 字节、写 `z_i` 4 字节，所以理论带宽统计常按每元素 12 字节估算：

$$GB/s=\frac{3\times n\times sizeof(float32)}{time}$$

这解释了为什么 tutorial 的 benchmark 用 `3 * x.numel() * x.element_size()` 来计算吞吐。这里没有复杂的算术强度分析，瓶颈基本是从 DRAM 到 kernel 再写回 DRAM 的路径是否高效。

## Compute Kernel：一个 program 处理一个连续 block

Triton kernel 用 `@triton.jit` 修饰。被 JIT 编译的函数参数可以分成两类：运行时参数，例如输入输出指针和元素个数；编译期元参数，例如 `BLOCK_SIZE: tl.constexpr`。后者必须在编译期可见，因为它会决定 `tl.arange(0, BLOCK_SIZE)` 的向量长度，也影响生成代码的形状。

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

代码里最关键的是 `pid` 和 `offsets`。`tl.program_id(axis=0)` 取出当前 program 在 1D grid 中的编号。假设 `BLOCK_SIZE = 1024`，那么 `pid = 0` 处理 `[0, 1024)`，`pid = 1` 处理 `[1024, 2048)`。`tl.arange(0, BLOCK_SIZE)` 不是 Python list，而是 Triton IR 中的一组向量 lane，后续 `tl.load` 和 `tl.store` 会按这些 offsets 发起向量化内存访问。

| 概念 | 在代码中的位置 | 含义 |
|-|-|-|
| Program | `tl.program_id(axis=0)` | 一个 SPMD kernel 实例，负责一个 block 的元素。 |
| Block | `BLOCK_SIZE` | 每个 program 处理的元素数量，是编译期元参数。 |
| Offsets | `block_start + tl.arange(...)` | 当前 program 覆盖的全局元素下标。 |
| Mask | `offsets < n_elements` | 保护最后一个不满 block 的尾部访问。 |

## 为什么 mask 是必需的

如果 `n_elements` 正好能被 `BLOCK_SIZE` 整除，最后一个 program 的 offsets 都合法。但真实输入长度通常不是 block size 的整数倍。tutorial 里的测试长度是 `98432`，而 launch 时 `BLOCK_SIZE = 1024`。需要的 program 数是：

$$\lceil 98432 / 1024 \rceil = 97$$

前 96 个 program 覆盖 `0` 到 `98303`，第 97 个 program 的 offsets 覆盖 `98304` 到 `99327`。其中 `98432` 之后的下标已经越界，如果没有 mask，load/store 会访问非法地址。`mask = offsets < n_elements` 让 Triton 只对合法 lane 执行内存操作，不合法 lane 被屏蔽掉。

这也是 Triton 和普通 Python 向量表达的差别之一。Triton 让 kernel 作者显式处理 block 边界，换来的是对内存访问形状和编译期展开的控制。对于更复杂的矩阵乘、softmax 或 layer norm，mask 同样会出现在 tile 边界、causal mask、padding 或 ragged batch 中。

## Launch Wrapper：Python 张量如何进入 GPU kernel

kernel 本身只描述单个 program 要做什么；还需要一个 Python wrapper 负责分配输出、计算 grid，并用 Triton 的 launch 语法提交 GPU kernel。

```python
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

`grid` 是一个根据 meta-parameters 计算 launch shape 的函数。这里使用 `triton.cdiv` 做向上取整，确保最后一个 partial block 也有 program 负责。调用 `add_kernel[grid](...)` 时，Triton 会把 `torch.Tensor` 隐式转换为指向首元素的 device pointer，并把 `BLOCK_SIZE=1024` 作为编译期元参数传入。

返回 `output` 时，kernel 可能仍在 GPU 上异步执行。这个行为和 CUDA kernel launch 一致：Python 调用把工作提交到设备队列，不会默认阻塞等待完成。只有后续需要同步结果、计时或跨设备依赖时，才需要显式同步。Triton 的 benchmark 工具内部会处理计时所需的同步。

## Scope / Layout / Dispatch：用三个问题读这段代码

把这个最小例子放进 Scope / Layout / Dispatch 框架，会比只记 API 更稳。

| 问题 | 向量加法中的答案 | 为什么重要 |
|-|-|-|
| Scope | 一个 Triton program 处理 `BLOCK_SIZE` 个连续元素。 | 决定 launch grid 大小，也决定每个 program 的工作量。 |
| Layout | `offsets = pid * BLOCK_SIZE + arange`，形成连续一维访问。 | 连续访问有利于合并访存；更复杂 kernel 会把 layout 扩展到二维 tile。 |
| Dispatch | `@triton.jit` 把 Python DSL 编译成 GPU kernel，`tl.load/store` 变成设备内存访问。 | 决定 Python 表达式是否只是在描述 IR，而不是立即在 CPU 上执行。 |

这个例子没有 shared memory、Tensor Core 或 warp specialization，但它已经包含 Triton 的基本抽象边界。到了矩阵乘法，`tl.arange` 会从一维 offsets 扩展成二维 tile 坐标；到了 fused softmax，mask 不只保护尾部，还保护每行长度；到了 attention，program id 可能同时编码 batch、head 和 block 坐标。

## 正确性验证：先和 PyTorch reference 对齐

自定义 kernel 第一件事不是跑 benchmark，而是确认输出和参考实现一致。tutorial 使用固定随机种子构造两个输入张量，用 PyTorch 的 `x + y` 作为 reference，再比较 Triton 输出。

```python
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)

print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```

示例输出中的最大差异是 `0.0`。对向量加法来说，这很合理：两边都是 `float32` 的逐元素加法，没有改变运算顺序，也没有引入近似函数。对于后续 softmax、matmul 或 attention，验证标准就不能机械套用 `0.0`，通常要结合数据类型、归约顺序和近似数学函数设置 `rtol` / `atol`。

## Benchmark：为什么用 GB/s 而不是只看耗时

向量加法是典型 memory-bound 操作。单个元素只做一次加法，算术开销很小；性能更接近“每秒能搬多少字节”。因此 tutorial 用 `triton.testing.perf_report` 扫描不同输入长度，从 $2^{12}$ 到 $2^{27}$，分别测 Triton 和 Torch 的吞吐。

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
```

这里的 `quantiles = [0.5, 0.2, 0.8]` 表示返回中位数、较快分位和较慢分位，用来画出吞吐曲线的不确定范围。示例输出中，从 `4096` 到 `134217728` 个元素，吞吐从 `8 GB/s` 上升到约 `1684 GB/s`。这个趋势比单个耗时更有解释力：小输入主要受 launch overhead 影响，大输入逐渐接近设备内存带宽上限。

下图来自 tutorial 运行结果，展示了不同 size 下 Triton 与 Torch 的向量加法吞吐对比。

![Triton Vector Add 与 Torch 在不同向量长度下的吞吐对比（来源：Triton 官方教程）](../assets/images/14-triton-vector-add-kernel/image-01.png)

## 这个例子能迁移到哪些 Triton kernel

向量加法看起来简单，但它提供了后续 Triton kernel 的基本骨架。任何 tile kernel 都需要回答：当前 program id 对应哪块输出，如何从 program id 推导 input/output offsets，哪些 lane 是合法的，哪些参数必须是 `tl.constexpr`，以及如何建立 reference 和 benchmark。

把向量加法推广到二维矩阵时，`tl.program_id(axis=0)` 和 `tl.program_id(axis=1)` 可以分别表示 row block 和 column block；`tl.arange` 可以构造 row offsets 与 column offsets 的组合；mask 从一维边界扩展成二维边界。再往后到 matmul，program 会负责一个 $M\times N$ 输出 tile，并沿 $K$ 维循环加载 A/B tile。

因此，这个入门示例真正要掌握的是执行模型，而不是向量加法本身。只要能稳定解释 `pid`、`BLOCK_SIZE`、`offsets`、`mask`、`grid` 和 `@triton.jit` 的关系，就已经具备阅读更复杂 Triton tutorial 的基础。

## 总结

Triton 用 Python 语法描述 GPU kernel，但它不是把普通 Python 函数搬到 GPU 上执行。`@triton.jit` 标记的是一段可编译的 kernel IR；`tl.program_id` 给出 SPMD program 的身份；`tl.arange` 构造 block 内的向量 lanes；`mask` 负责边界安全；launch grid 决定有多少 program 并行覆盖输入。

Vector Add 的价值在于把这些概念压缩到几十行代码里。读懂这段代码以后，再看 fused softmax、matmul、attention 等 tutorial，就可以沿着同一条路径继续追问：每个 program 负责哪个 tile，tile 内的 offsets 如何组织，边界和同步如何处理，benchmark 指标到底反映计算吞吐还是内存带宽。

**参考：**[Triton Tutorial - Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)；[triton-lang/triton](https://github.com/triton-lang/triton)。
