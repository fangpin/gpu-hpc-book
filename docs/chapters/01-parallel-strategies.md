# 并行策略

无论是 CPU 还是 GPU 从并行策略上都分为以下几类。理解不同并行方式有助于理解如何实现高性能计算。

## 指令并行 ILP (Instuction Level Parallel)

指令并行是指同一线程(同一程序流)中，在一个周期内同时发射多条无相互依赖的指令到cpu进行执行。

现代 CPU 通常包含多套独立ALU/FPU等硬件设备，加上流水线和乱序执行等技术，支持了指令并行的实现。

### CPU 指令并行示例

```java
void mandelbrot_cpu_vector_ilp(uint32_t img_size,
                               uint32_t max_iters,
                               uint32_t *out) {
    const __m512 v_img_size = _mm512_set1_ps((float)img_size);
    const __m512 v_zoom = _mm512_set1_ps(window_zoom);
    const __m512 v_4 = _mm512_set1_ps(4.0f);
    const __m512 v_wx = _mm512_set1_ps(window_x);
    const __m512i v_one = _mm512_set1_epi32(1);

    const __m512i v_step =
        _mm512_set_epi32(15,14,13,12,11,10,9,8,
                        7,6,5,4,3,2,1,0);

    for (int i = 0; i < img_size; ++i) {
        float cy = (float(i) / img_size) * window_zoom + window_y;
        __m512 v_cy = _mm512_set1_ps(cy);

        for (int j = 0; j < img_size; j += 32) {
            __m512i v_j0 = _mm512_add_epi32(_mm512_set1_epi32(j), v_step);
            __m512i v_j1 = _mm512_add_epi32(_mm512_set1_epi32(j + 16), v_step);

            __m512 v_cx0 = _mm512_add_ps(
                _mm512_mul_ps(_mm512_div_ps(_mm512_cvtepi32_ps(v_j0), v_img_size), v_zoom),
                v_wx
            );

            __m512 v_cx1 = _mm512_add_ps(
                _mm512_mul_ps(_mm512_div_ps(_mm512_cvtepi32_ps(v_j1), v_img_size), v_zoom),
                v_wx
            );

            __m512 v_x0 = _mm512_setzero_ps();
            __m512 v_y0 = _mm512_setzero_ps();
            __m512i v_it0 = _mm512_setzero_si512();

            __m512 v_x1 = _mm512_setzero_ps();
            __m512 v_y1 = _mm512_setzero_ps();
            __m512i v_it1 = _mm512_setzero_si512();

            for (int n = 0; n < max_iters; ++n) {
                __m512 x2_0 = _mm512_mul_ps(v_x0, v_x0);
                __m512 y2_0 = _mm512_mul_ps(v_y0, v_y0);
                __mmask16 m0 =
                    _mm512_cmp_ps_mask(_mm512_add_ps(x2_0, y2_0), v_4, _CMP_LE_OS);

                __m512 x2_1 = _mm512_mul_ps(v_x1, v_x1);
                __m512 y2_1 = _mm512_mul_ps(v_y1, v_y1);
                __mmask16 m1 =
                    _mm512_cmp_ps_mask(_mm512_add_ps(x2_1, y2_1), v_4, _CMP_LE_OS);

                if (!(m0 | m1)) break;

                v_it0 = _mm512_mask_add_epi32(v_it0, m0, v_it0, v_one);
                v_it1 = _mm512_mask_add_epi32(v_it1, m1, v_it1, v_one);

                __m512 xy0 = _mm512_mul_ps(v_x0, v_y0);
                __m512 xy1 = _mm512_mul_ps(v_x1, v_y1);

                v_y0 = _mm512_add_ps(_mm512_add_ps(xy0, xy0), v_cy);
                v_y1 = _mm512_add_ps(_mm512_add_ps(xy1, xy1), v_cy);

                v_x0 = _mm512_add_ps(_mm512_sub_ps(x2_0, y2_0), v_cx0);
                v_x1 = _mm512_add_ps(_mm512_sub_ps(x2_1, y2_1), v_cx1);
            }

            _mm512_storeu_si512((__m512i *)(out + i * img_size + j), v_it0);
            _mm512_storeu_si512((__m512i *)(out + i * img_size + j + 16), v_it1);
        }
    }
}
```

这里实现了两路并行的指令并行，在循环中，每次计算两个不相关的像素点。整体性能能够带来约两倍的提升。不足两倍的主要原因应该是每次循环中迭代次数相差较大，导致负载不均，需要更好的负载均衡策略，才能更逼近理论两倍提升值。

```bash
g++ -march=native -O3 -Wall -Wextra -o mandelbrot mandelbrot_cpu_2.cpp 
```

```bash
./mandelbrot -r 320 -b 320 -i scalar
Testing with image size 320x320 and 320 max iterations.

Running mandelbrot_cpu_scalar ...
  Runtime:  70.93 ms
  
  
./mandelbrot -r 320 -b 320 -i vector
Testing with image size 320x320 and 320 max iterations.

Running mandelbrot_cpu_vector ...
  Runtime:   7.45 ms
  
  
/mandelbrot -r 320 -b 320 -i vector_ilp
Testing with image size 320x320 and 320 max iterations.

Running mandelbrot_cpu_vector_ilp ...
  Runtime:   4.73 ms
```

### GPU 指令并行示例

```cpp
__global__ void mandelbrot_gpu_vector_ilp(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out)
{
    for (uint64_t i = 0; i < img_size; ++i) {
        for (uint64_t j = 0; j < img_size; j += 2 * blockDim.x) {

            int jx0 = j + threadIdx.x;
            int jx1 = j + threadIdx.x + blockDim.x;

            float cx0 = (float(jx0) / img_size) * 2.5f - 2.0f;
            float cx1 = (float(jx1) / img_size) * 2.5f - 2.0f;
            float cy  = (float(i) / img_size) * 2.5f - 1.25f;

            float x2_0 = 0.0f, y2_0 = 0.0f, w0 = 0.0f;
            float x2_1 = 0.0f, y2_1 = 0.0f, w1 = 0.0f;

            uint32_t iters0 = 0, iters1 = 0;

            while (true) {
                bool active0 = (x2_0 + y2_0 <= 4.0f) && (iters0 < max_iters);
                bool active1 = (x2_1 + y2_1 <= 4.0f) && (iters1 < max_iters);

                if (!active0 && !active1) break;

                if (active0) {
                    float x = x2_0 - y2_0 + cx0;
                    float y = w0 - x2_0 - y2_0 + cy;
                    x2_0 = x * x;
                    y2_0 = y * y;
                    float z = x + y;
                    w0 = z * z;
                    iters0++;
                }

                if (active1) {
                    float x = x2_1 - y2_1 + cx1;
                    float y = w1 - x2_1 - y2_1 + cy;
                    x2_1 = x * x;
                    y2_1 = y * y;
                    float z = x + y;
                    w1 = z * z;
                    iters1++;
                }
            }

            out[i * img_size + jx0] = iters0;
            out[i * img_size + jx1] = iters1;
        }
    }
}

void launch_mandelbrot_gpu_vector_ilp(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    return mandelbrot_gpu_vector_ilp<<<1, 32>>>(img_size, max_iters, out);
}

```

GPU同样可以使用指令并行的方式来提升执行速度，且有与CPU相似的结论

```bash
nvcc -O3  -o mandelbrot_gpu mandelbrot_gpu_2.cu 
```

```bash
 ./mandelbrot_gpu -r 320 -b 320 -i vector
Testing with image size 320x320 and 320 max iterations.

Running launch_mandelbrot_gpu_vector ...
  Runtime:  52.10 ms
  
  
./mandelbrot_gpu -r 320 -b 320 -i vector_ilp
Testing with image size 320x320 and 320 max iterations.

Running launch_mandelbrot_gpu_vector_ilp ...
  Runtime:  17.59 ms
```

## 多核并行 Multiple core parallelism

多核并行将任务分发到不同核心来实现并行计算。

### CPU 多核并行示例

使用与CPU 逻辑核相同核心数目

```cpp
void mandelbrot_cpu_vector_thread(int n_threads, int id, uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    const __m512 v_img_size = _mm512_set1_ps((float)img_size);
    const __m512 v_window_zoom = _mm512_set1_ps(window_zoom);
    const __m512 v_4_0 = _mm512_set1_ps(4.0f);
    const __m512i v_max_iters = _mm512_set1_epi32(max_iters);
    const __m512i v_step = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

    const int chunk_size = img_size / n_threads;
    const int start_i = id * chunk_size, end_i = min((id + 1) * chunk_size, img_size);

    for (int i = start_i; i < end_i; ++i) {
        float cy_val = (float(i) / img_size) * window_zoom + window_y;
        __m512 v_cy = _mm512_set1_ps(cy_val);

        for (int j = 0; j < img_size; j += 16) {
            __m512i v_j = _mm512_add_epi32(_mm512_set1_epi32(j), v_step);
            __m512 v_cx = _mm512_add_ps(_mm512_mul_ps(_mm512_div_ps(_mm512_cvtepi32_ps(v_j), v_img_size), v_window_zoom), _mm512_set1_ps(window_x));

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

void mandelbrot_cpu_vector_multicore(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {
    const int n_threads = 8;
    std::vector<std::thread> threads;
    for (int i=0; i<n_threads; ++i) {
        threads.emplace_back(std::thread([=] {
            mandelbrot_cpu_vector_thread(n_threads, i, img_size, max_iters, out);
        }));
    }
    for (auto &t: threads) {
        t.join();
    }
}
```

```bash
g++ -march=native -O3 -Wall -Wextra -o mandelbrot mandelbrot_cpu_2.cpp -lpthread
```

```bash
./mandelbrot -r 320 -b 320 -i vector_multicore

Running mandelbrot_cpu_vector_multicore ...
  Runtime:   3.54 ms
```

### GPU 多核并行示例

```cpp
__global__ void mandelbrot_gpu_vector_multicore(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    for (uint64_t i = 0; i < img_size; i+=gridDim.x) {
        int ix = i + blockIdx.x;
        for (uint64_t j = 0; j < img_size; j+=blockDim.x) {
            // Get the plane coordinate X for the image pixel.
            int jx = j + threadIdx.x;
            float cx = (float(jx) / float(img_size)) * window_zoom + window_x;
            float cy = (float(ix) / float(img_size)) * window_zoom + window_y;

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
            out[ix * img_size + jx] = iters;
        }
    }
}

void launch_mandelbrot_gpu_vector_multicore(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    return mandelbrot_gpu_vector_multicore<<<80, 32 * 4>>>(img_size, max_iters, out);
}

```

注意使用了80个SM(我手上gpu卡的全部sm)，每个sm共4个warp scheduler。类似于cpu的思路，使用了全部可用的硬件核心。

```bash
nvcc -O3  -o mandelbrot_gpu mandelbrot_gpu_2.cu 
```

```bash
./mandelbrot_gpu -r 320 -b 320 -i vector
Testing with image size 320x320 and 320 max iterations.

Running launch_mandelbrot_gpu_vector ...
  Runtime:  52.63 ms
  
#<<<80,32*4>>>
./mandelbrot_gpu -r 320 -b 320 -i vector_multicore
Testing with image size 320x320 and 320 max iterations.

Running launch_mandelbrot_gpu_vector_multicore ...
  Runtime:   0.10 ms
  
#<<<160,32*2>>>
./mandelbrot_gpu -r 320 -b 320 -i vector_multicore
Testing with image size 320x320 and 320 max iterations.

Running launch_mandelbrot_gpu_vector_multicore ...
  Runtime:   0.09 ms
  
  
#<<<40,32*8>>> sm未充分使用
./mandelbrot_gpu -r 320 -b 320 -i vector_multicore
Testing with image size 320x320 and 320 max iterations.

Running launch_mandelbrot_gpu_vector_multicore ...
  Runtime:   0.14 ms
```

## Multi-threaded parallelism

整体思路类似Multiple core parallelism，区别是并发数目大于硬件核心数。



示例代码与多核并行相同，只是增加了 thread 数量。可以发现Multi-threaded parallelism中，随着并发数(threaded数目)增加，性能先明显提升，后提升放缓，最后不再提升。通常的原因是一个thread中会因为cache miss等原因，让硬件stall。

- **而更多的threads可以通过thread的调度来隐藏硬件latency。**
- 通常你的 ILP 做的越好，任务越是compute-bound，这个提升也就越小。
- 当硬件资源(比如ALU没有闲置，内存带宽没有闲置等)被打满后，再提升 thread 数目就不再能取得正向效果。

---

最后一次更新时间：`2026-08-05 16:12:21 CST`

原文链接：[https://fangpin.github.io/gpu-hpc-book/#/chapters/01-parallel-strategies.md](https://fangpin.github.io/gpu-hpc-book/#/chapters/01-parallel-strategies.md)
