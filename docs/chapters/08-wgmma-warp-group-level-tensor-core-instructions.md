# **`wgmma`：warp-group level tensor core instructions**

Wgmma 是利用warp group level的tensor cores进行高性能矩阵乘法的关键。以H100为例，tensor core的 bf16 计算能力约为 1000 TFLOPs；而普通的FMA的 bf16 计算能力是 120 TFLOPs，大约只有tensor core的十分之一。因此使用 mma 进行矩阵运算几乎是必选项。

相比mma，wgmma有几个重要不同：

<table><colgroup><col/><col/><col/></colgroup><tbody><tr><td>不同点</td><td>mma</td><td>wgmma</td></tr><tr><td>工作单元</td><td>warp</td><td>Warp group，通常4warps</td></tr><tr><td>同步方式</td><td>同步</td><td>异步执行，可以结合TMA实现计算和存储交叉</td></tr><tr><td>计算逻辑</td><td>D=A*B+C</td><td>D=A*B+D。<ul><li>支持A，B转制，取相反数；</li><li>支持D 乘 0 或 1</li></ul></td></tr><tr><td>数据位置</td><td>均在寄存器内</td><td><ul><li>A 寄存器或shared memory</li><li>B shared memory</li><li>D 寄存器</li></ul></td></tr><tr><td>Tile 大小</td><td><code>M = 16</code>, <code>N = 8</code>, and <code>K = 16</code></td><td><code>M = 64</code>, <code>K = 16</code>, and <code>N</code> ranging from 8 to 256 in steps of 8</td></tr><tr><td>是否支持swizzling pattern</td><td>否</td><td>是</td></tr></tbody></table>

---

最后一次更新时间：`2026-08-05 16:12:21 CST`

原文链接：[https://fangpin.github.io/gpu-hpc-book/#/chapters/08-wgmma-warp-group-level-tensor-core-instructions.md](https://fangpin.github.io/gpu-hpc-book/#/chapters/08-wgmma-warp-group-level-tensor-core-instructions.md)
