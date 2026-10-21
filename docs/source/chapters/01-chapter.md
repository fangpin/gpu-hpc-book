# 计算的硬件加速思路

```{contents} 本页目录
---
depth: 2
local: true
---
```

在硬件层面，计算通常可以分为几个阶段：

- 取指令
- 指令解码
- 取数
- 计算
- 写回

为了加速计算，整体有两个大的思路：

1. 思路一：缩短指令的执行时间
2. 思路二：提升单位时间内指令的吞吐量。



## 思路1: 缩短指令的执行时间

常见的方法包括：

- 指令流水线，以及对应的各种tricky技术，比如

  - 指令预取
  - 内存预取
  - 分支预测
- 更大的core，更快的主频等



## 思路2: 提升单位时间内指令的吞吐量

常见的方法包括：

- 多核心
- 硬件复用后的虚拟多核心：只将部分关键硬件复制几分，底层的存储等仍然复用，这样能做到有效控制成本的同时提升单位时间内的指令吞吐。



现代处理器，无论是CPU/GPU，通常会同时使用以上两种加速思路。对于 **提升单位时间内指令的吞吐量，**在现代CPU，有 SIMD 指令。

### 对于cpu scalar 实现

```java
void mandelbrot_cpu_scalar(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    for (uint64_t i = 0; i < img_size; ++i) {
        for (uint64_t j = 0; j < img_size; ++j) {
            // Get the plane coordinate X for the image pixel.
            float cx = (float(j) / float(img_size)) * 2.5f - 2.0f;
            float cy = (float(i) / float(img_size)) * 2.5f - 1.25f;

            // Innermost loop: start the recursion from z = 0.
            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;
            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float x = x2 - y2 + cx;
                float y = w - x2 - y2 + cy;
                x2 = x * x;
                y2 = y * y;
                float z = x + y;
                w = z * z;
                ++iters;
            }

            // Write result.
            out[i * img_size + j] = iters;
        }
    }
}
```

有：

```bash
g++ -march=native -O3 -Wall -Wextra -o mandelbrot mandelbrot_cpu.cc

Testing with image size 320x320 and 320 max iterations.
Running mandelbrot_cpu_scalar ...
  Runtime: 57.518 ms
```

### CPU 使用 SIMD 实现 CPU 的向量化加速：

```c++
void mandelbrot_cpu_vector(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    const __m512 v_img_size = _mm512_set1_ps((float)img_size);
    const __m512 v_2_5 = _mm512_set1_ps(2.5f);
    const __m512 v_4_0 = _mm512_set1_ps(4.0f);
    const __m512i v_max_iters = _mm512_set1_epi32(max_iters);
    const __m512i v_step = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

    for (int i = 0; i < img_size; ++i) {
        float cy_val = (float(i) / img_size) * 2.5f - 1.25f;
        __m512 v_cy = _mm512_set1_ps(cy_val);

        for (int j = 0; j < img_size; j += 16) {
            __m512i v_j = _mm512_add_epi32(_mm512_set1_epi32(j), v_step);
            __m512 v_cx = _mm512_sub_ps(_mm512_mul_ps(_mm512_div_ps(_mm512_cvtepi32_ps(v_j), v_img_size), v_2_5), _mm512_set1_ps(2.0f));

            __m512 v_x = _mm512_set1_ps(0.0f);
            __m512 v_y = _mm512_set1_ps(0.0f);
            __m512i v_iters = _mm512_set1_epi32(0);

            for (int n = 0; n < max_iters; ++n) {
                __m512 x2 = _mm512_mul_ps(v_x, v_x);
                __m512 y2 = _mm512_mul_ps(v_y, v_y);
                
                __mmask16 mask = _mm512_cmp_ps_mask(_mm512_add_ps(x2, y2), v_4_0, _CMP_LE_OS);
                if (!mask) break;

                v_iters = _mm512_mask_add_epi32(v_iters, mask, v_iters, _mm512_set1_epi32(1));

                __m512 xy = _mm512_mul_ps(v_x, v_y);
                v_y = _mm512_add_ps(_mm512_add_ps(xy, xy), v_cy);
                v_x = _mm512_add_ps(_mm512_sub_ps(x2, y2), v_cx);
            }
            _mm512_storeu_si512((__m512i *)(out + i * img_size + j), v_iters);
        }
    }
}
```

#### AVX-512 一次计算 16 个像素

标量版本每次只计算一个像素，向量版本把同一行连续的 16 个像素装进一条 512 位寄存器。一个 **float** 占 32 位，因此 **__m512** 正好容纳 16 个浮点数；每个位置称为一个 lane，对应一个像素。

Mandelbrot 迭代可以写成 $z_{n+1}=z_n^2+c$。拆成实部和虚部后：

$$x_{n+1}=x_n^2-y_n^2+c_x$$

$$y_{n+1}=2x_ny_n+c_y$$

代码中的 **v_x**、**v_y** 和 **v_iters** 分别保存 16 个像素当前的实部、虚部和迭代次数。

#### 像素坐标如何装进 16 个 lane

外层循环按行遍历。同行像素的纵坐标相同，因此先计算一个 **cy_val**，再用 **_mm512_set1_ps** 广播到 16 个 lane。

内层循环每次令 **j** 增加 16。**v_step** 在低 lane 到高 lane 中保存 0 到 15，和广播后的 **j** 相加后得到连续的 16 个列号。例如 **j=32** 时，**v_j** 表示第 32 到 47 列。

**_mm512_set_epi32** 的参数按高 lane 到低 lane 书写，所以源码中的参数看起来从 15 倒排到 0，得到的 lane 顺序仍然是 0 到 15。

随后，这 16 个列号一起映射到复平面的横坐标：

$$c_x=(j/img_size)\times 2.5-2.0$$

#### mask 处理不同像素的逃逸速度

每轮迭代先计算 16 个点各自的 $x^2$ 和 $y^2$，再比较 $x^2+y^2\leq 4$。比较结果写入 16 位的 **__mmask16**：

- 某一位为 1，表示对应像素仍在逃逸半径内，需要继续计数。
- 某一位为 0，表示对应像素已经逃逸，保留当前迭代次数。

**_mm512_mask_add_epi32** 只给 mask 为 1 的 lane 增加一次迭代计数。mask 全为 0 时，说明这组 16 个像素已经全部逃逸，可以提前结束循环。

后面的 **v_x** 和 **v_y** 更新仍会对全部 lane 执行，但已经逃逸的 lane 不再增加 **v_iters**。Mandelbrot 的逃逸性质保证点越过逃逸半径后不会重新变成有效点，因此这里不需要让 lane 再次加入计数。

#### 迭代公式和写回过程

**xy**、**x2** 和 **y2** 都在更新寄存器前算好，因此先给 **v_y** 赋值不会污染 **v_x** 使用的旧值。两行更新代码正好对应：

```c++
new_y = 2 * x * y + cy;
new_x = x * x - y * y + cx;
```

迭代结束后，**_mm512_storeu_si512** 一次把 16 个计数写入 **out[i * img_size + j]** 到 **out[i * img_size + j + 15]**。函数名中的 **u** 表示目标地址不要求按 64 字节对齐。

#### 标量实现和向量实现的对应关系

| 标量实现 | AVX-512 实现 |
|-|-|
| 一个 cx | 16 个 cx |
| 一个 cy | 广播成 16 个相同的 cy |
| 一组 x、y | 16 组 x、y |
| while 条件 | 16 位 mask |
| iters 自增 | mask 控制的向量加法 |
| 写回一个结果 | 一次写回 16 个结果 |

#### 使用前提和边界

<callout emoji="💡">
当前实现要求 **img_size** 是 16 的倍数。否则最后一轮仍会写回 16 个结果，可能跨过本行，最后一行还会越过 **out** 的分配边界。尾部不足 16 个像素时，应改用 masked store，或者退回标量循环。
</callout>

- **v_max_iters** 当前没有被使用；迭代上限由 **n < max_iters** 控制。
- 运行机器必须支持 AVX-512，编译时也要启用相应指令集，例如 **-mavx512f**。
- 向量宽度提供的是并行上限。实际加速比还会受到分支分化、指令吞吐、访存和编译器生成代码的影响。

性能提升19倍，约等于 avx 并行数。但可以发现 cpu SIMD 编程相比下面的cuda 编程，心智负担需要更高，因为需要自行计算和使用mask，确保vector中不同位置(lane)执行(可能)不同的分支。

```bash
g++ -O3 -mavx512f -mavx512vl -o mandelbrot_avx512 mandelbrot_cpu.cpp
./mandelbrot_avx512 -r 320 -b 320 -i vector
```

```bash
Testing with image size 320x320 and 320 max iterations.
Running mandelbrot_cpu_vector ...
  Runtime: 3.09283 ms
  Correctness: average output difference from reference = 0.000207642
```

### Cuda SPMD

```c++

__global__ void mandelbrot_gpu_vector(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    for (uint64_t i = 0; i < img_size; ++i) {
        for (uint64_t j = 0; j < img_size; j+=blockDim.x) {
            // Get the plane coordinate X for the image pixel.
            int jx = j + threadIdx.x;
            float cx = (float(jx) / float(img_size)) * 2.5f - 2.0f;
            float cy = (float(i) / float(img_size)) * 2.5f - 1.25f;

            // Innermost loop: start the recursion from z = 0.
            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;
            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float x = x2 - y2 + cx;
                float y = w - x2 - y2 + cy;
                x2 = x * x;
                y2 = y * y;
                float z = x + y;
                w = z * z;
                ++iters;
            }

            // Write result.
            out[i * img_size + jx] = iters;
        }
    }
}

void launch_mandelbrot_gpu_vector(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    mandelbrot_gpu_vector<<<1,32>>>(img_size, max_iters, out);
}
```

Gpu vector parallel

```bash
nvcc -O3  -o mandelbrot mandelbrot_gpu.cu
./mandelbrot -r 320 -b 320 -i vector
```

```bash
Testing with image size 320x320 and 320 max iterations.
Running launch_mandelbrot_gpu_vector ...
  Runtime: 10.7224 ms
  Correctness: average output difference from reference 0.000191223
```

可见从单核心性能上，gpu 单核通常慢于cpu 单核
