# 计算的硬件加速思路

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

```java
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

```cpp

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

---

最后一次更新时间：`2026-08-05 16:12:21 CST`
