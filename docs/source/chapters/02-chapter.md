# 并行策略

```{contents} 本页目录
---
depth: 2
local: true
---
```

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

#### 两条向量链让处理器有更多可发射指令

基础 AVX-512 版本每次维护一个 16-lane 向量。这段 ILP 版本同时维护后缀为 **0** 和 **1** 的两套状态，每套仍包含 16 个 lane，因此每个 **j** 批次一共处理 32 个像素。

| 状态组 | 像素列 | 独立状态 |
|-|-|-|
| 向量链 0 | j 到 j + 15 | v_x0、v_y0、v_it0、v_cx0 |
| 向量链 1 | j + 16 到 j + 31 | v_x1、v_y1、v_it1、v_cx1 |

两条链计算不同像素，数据之间没有依赖。第一条链等待乘法结果时，乱序执行器可以发射第二条链中已经就绪的指令，用更多独立指令填充执行流水线。

#### 两组 mask 分别跟踪逃逸状态

**m0** 和 **m1** 分别记录两组 16 个像素是否仍满足 $x^2+y^2\leq 4$。**_mm512_mask_add_epi32** 只给对应 mask 为 1 的 lane 增加迭代次数。

退出条件是 **!(m0 | m1)**。只有两组共 32 个像素全部逃逸，循环才会结束；任意一组还有有效 lane，另一组的寄存器仍会参与后续向量运算，但其无效 lane 不再增加迭代次数。

#### ILP 隐藏的是依赖延迟

单个像素的下一轮必须等待上一轮产生新的 **x** 和 **y**，这是一条循环携带依赖。增加第二条独立链不会扩大一条 AVX-512 指令的 16-lane 宽度，却能在第一条依赖链等待结果时提供另一批可执行指令。

因此，ILP 的收益取决于指令延迟、执行端口吞吐、mask 分布和寄存器压力，并不保证固定获得两倍加速。这里的并行结构是“2 条独立向量链 × 每条 16 个 lane”，而不是两个标量像素。

#### 当前实现的边界

<callout emoji="💡">
循环每次固定写回 32 个结果，因此当前实现要求 **img_size** 是 32 的倍数。否则第二组向量可能跨过行尾，最后一行还会越过 **out** 的分配边界。尾部应使用 masked store，或者退回标量循环。
</callout>

- **window_zoom**、**window_x** 和 **window_y** 在当前文件片段中没有定义，需要由外部提供。
- 如需与使用固定坐标范围的 GPU 版本逐像素比较，应确认 **window_zoom=2.5**、**window_x=-2.0**、**window_y=-1.25**。
- 一组向量已经全部逃逸、另一组仍在计算时，代码仍会执行两组更新指令。这会消耗一部分吞吐，但简化了控制流。

下面的 benchmark 中，基础 AVX-512 版本从 7.45 ms 降到 4.73 ms，约为 1.58 倍加速。这个结果说明第二条独立向量链提高了流水线利用率，但实际收益仍受依赖延迟、mask 分布和寄存器压力影响，并不存在固定的两倍加速。

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

```c++
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

#### 一个线程同时推进两个像素

kernel 仍然启动 32 个线程，但每个线程保存两套 Mandelbrot 状态。对于线程 **t=threadIdx.x**：

| 状态链 | 像素列 | 状态变量 |
|-|-|-|
| 链 0 | j + t | x2_0、y2_0、w0、iters0 |
| 链 1 | j + t + blockDim.x | x2_1、y2_1、w1、iters1 |

在 **blockDim.x=32** 时，一个 warp 的第一条链覆盖第 **j** 到 **j+31** 列，第二条链覆盖第 **j+32** 到 **j+63** 列。两个写回指令各自访问连续地址，因此仍能形成合并访存。

#### 两条递推链可以独立退出

**active0** 和 **active1** 分别判断两个像素是否还在逃逸半径内、是否达到最大迭代次数。一个像素先结束后，对应的 **if** 分支不再更新它；线程要等到两个像素都结束才退出 **while**。

两套递推都利用 $(x+y)^2-x^2-y^2=2xy$ 计算虚部。它们共享同一行的 **cy**，除此之外没有数据依赖，编译器可以在保证语义的前提下交错安排两条链的指令。

#### 只有一个 warp 时更依赖 ILP 隐藏延迟

启动配置 **<<<1, 32>>>** 只创建一个 warp。常规 GPU kernel 会在一个 warp 等待结果时切换到其他 warp，而这里没有其他 warp 可切换。增加第二条独立递推链后，scoreboard 等待链 0 的操作数时，链 1 可能仍有就绪指令，从而减少整个 warp 的空转时间。

源码只是表达出两条独立计算链，最终能否交错发射取决于编译器生成的指令和 GPU 调度。实测加速比还混合了循环控制开销等因素，不能直接当成通用的 ILP 倍率。

#### ILP 会增加分支和寄存器成本

- 每个线程需要保存两套坐标、中间值和计数器，会增加寄存器占用；在正常的多 block kernel 中，寄存器压力过高可能降低 occupancy。
- 两个像素的逃逸次数差异较大时，一个分支已经停止，另一个分支仍要继续。warp 内不同线程的 **active0** 和 **active1** 也可能不同，从而产生分支分化。
- 这段代码只启动一个 warp，ILP 能缓解依赖停顿，却不能替代足够的 block 和 warp 并行度。

#### 当前实现的边界

<callout emoji="💡">
内层循环每次覆盖 64 列，却没有检查 **jx0** 和 **jx1** 是否小于 **img_size**。当前实现要求 **img_size** 是 64 的倍数，否则最后一批线程会越界写入。
</callout>

- 索引没有使用 **blockIdx.x**。直接增加 block 数量会让多个 block 计算并写入同一批像素。
- kernel launch 相对 host 异步。包装函数中的 **return** 不会等待 GPU 完成，也不会返回计算结果；调用方仍需同步并检查启动错误。
- **out** 必须指向 GPU 可访问的内存。

下面的 benchmark 中，单像素链版本从 52.10 ms 降到双像素链版本的 17.59 ms。由于两个版本都只启动一个 warp，这组数据同时受到依赖停顿、循环控制开销和编译器调度影响，更适合说明当前示例中的 ILP 收益，不能直接外推到常规多 warp kernel。

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

这版实现固定创建 8 个 CPU 线程，并未读取机器的逻辑核数。线程之间按行分块，每个线程内部再用 AVX-512 一次处理 16 个像素；下方 benchmark 中的 3.54 ms 只代表当前环境和参数。

```c++
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

#### 多核并行和 SIMD 分别切分两个维度

这段实现把图像的行分给 8 个 CPU 线程，每个线程再用 AVX-512 按 16 列一组计算。两层并行处理的是不同维度：

| 并行层级 | 切分对象 | 当前宽度 |
|-|-|-|
| CPU 多线程 | 图像行 i | 固定 8 个线程 |
| AVX-512 SIMD | 同行中的列 j | 每条指令 16 个像素 |

线程只写入自己负责的完整行，因此不同线程访问的 **out** 区间互不重叠，不需要互斥锁。每个像素内部仍使用前文相同的 Mandelbrot 向量递推和 mask 计数。

#### 线程创建和 join 保证任务完整结束

循环中的 lambda 使用 **[=]** 按值捕获当前线程编号 **i**、线程总数、图像参数和输出指针。每个 lambda 随后调用一次 **mandelbrot_cpu_vector_thread**。

**join** 会等待对应线程计算结束。主函数只有在 8 个线程全部完成后才返回，这也保证调用方读取 **out** 时结果已经写完。**threads.emplace_back(std::thread(...))** 会先构造临时线程再移动进容器，可以简写成直接向 **emplace_back** 传入 lambda，但当前写法在语义上成立。

#### 当前行分块会遗漏余数

<callout emoji="💡">
**chunk_size=img_size/n_threads** 使用整数除法。图像高度不能被线程数整除时，余数行不会分给任何线程。例如 **img_size=322**、**n_threads=8** 时，最后一个线程只处理到第 319 行，第 320 和 321 行会被遗漏；当 **img_size<8** 时，所有线程都没有工作。
</callout>

可以用按比例计算边界的方式覆盖全部行，并让各线程的行数最多相差 1：

```c++
int start_i = id * int(img_size) / n_threads;
int end_i = (id + 1) * int(img_size) / n_threads;
```

#### 静态分块还会受到负载不均影响

Mandelbrot 每个像素的逃逸次数不同，各行计算量也不同。连续行静态分块虽然没有调度开销，但某些线程可能提前完成，剩余线程继续处理高迭代区域。可以改用交错分行，或者用线程池动态领取行任务，以改善负载均衡。

线程数也被固定为 8，只有运行机器恰好有 8 个合适的逻辑执行单元时才与硬件匹配。实际程序通常根据 **std::thread::hardware_concurrency()**、物理核心数和 benchmark 选择线程数。

#### SIMD 尾部仍需单独处理

- 列循环每次固定写回 16 个结果，因此 **img_size** 不是 16 的倍数时会越界写入。尾部应使用 masked store，或者退回标量循环。
- **v_max_iters** 在当前函数中没有被使用，最大迭代次数由 **for (n < max_iters)** 控制。
- **window_zoom**、**window_x** 和 **window_y** 仍需要由当前文件之外的代码提供。

```bash
g++ -march=native -O3 -Wall -Wextra -o mandelbrot mandelbrot_cpu_2.cpp -lpthread
```

```bash
./mandelbrot -r 320 -b 320 -i vector_multicore

Running mandelbrot_cpu_vector_multicore ...
  Runtime:   3.54 ms
```

### GPU 多核并行示例

```c++
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

#### block 切分行，thread 切分列

这个 kernel 使用二维任务的手工映射，但仍采用一维 grid 和一维 block：

| CUDA 层级 | 负责的图像维度 | 索引关系 |
|-|-|-|
| block | 图像行 | ix = i + blockIdx.x |
| thread | 同行中的列 | jx = j + threadIdx.x |

外层每次让 **i** 增加 **gridDim.x**，所以 block **b** 负责第 **b**、**b+gridDim.x**、**b+2×gridDim.x** 等行。内层同理，线程 **t** 负责第 **t**、**t+blockDim.x**、**t+2×blockDim.x** 等列。

在索引合法时，每个 **(ix, jx)** 只会由一个线程写入，不需要原子操作。一个 warp 中相邻线程写入同行的连续列，也符合合并访存的地址模式。

#### 80 个 block 不等于固定绑定 80 个 SM

启动配置 **<<<80, 32×4>>>** 创建 80 个 block，每个 block 有 128 个线程，也就是 4 个 warp。block 由 GPU 运行时动态调度到 SM，一个 block 不会跨 SM，但代码不能指定某个 block 固定运行在哪个 SM，也不能保证 4 个 warp 分别固定绑定 4 个 warp scheduler。

当设备恰有 80 个 SM 时，80 个 block 最多可以让每个 SM 获得一个 block。这个数字仍然是设备相关的硬编码。实际配置应结合 SM 数量、寄存器和共享内存占用、每个 SM 可驻留的 block 数，以及 workload 的负载差异来选择。

#### 两层 grid-stride 循环都缺少尾部保护

<callout emoji="💡">
循环条件只检查基础索引 **i** 和 **j**，没有检查加上 **blockIdx.x**、**threadIdx.x** 后的 **ix** 和 **jx**。图像尺寸不能整除 grid 或 block 宽度时会越界。示例中的 **img_size=320** 不能被 **blockDim.x=128** 整除，最后一轮会有 64 个线程写到行尾之外。
</callout>

至少需要在两层映射后分别增加边界判断：

```c++
int ix = i + blockIdx.x;
if (ix >= img_size) break;

int jx = j + threadIdx.x;
if (jx >= img_size) break;
```

#### 更多 block 通常比按 SM 数量硬配更稳妥

Mandelbrot 不同行、不同像素的迭代次数差异很大。只创建与 SM 数量相同的 block 时，一个耗时较长的 block 可能拖住收尾阶段。创建更多、粒度更小的 block，可以让硬件调度器更灵活地平衡工作，但 block 太少或每个 block 资源占用太高都会降低并行度。

每个线程内部仍是标量 Mandelbrot 循环，因此 warp 内像素逃逸时间不同会产生分支分化。增加 block 数量可以改善 SM 级负载均衡，不能消除同一 warp 内的分化。

#### 包装函数只提交异步任务

- **return** 不会把 kernel 结果返回给 host，也不会等待 GPU 完成；调用方仍需同步并检查启动错误。
- **out** 必须指向 GPU 可访问的内存。
- 亚毫秒 benchmark 应使用 CUDA Event，或者在 host 计时边界显式同步，否则可能只测到异步提交时间。
- **window_zoom**、**window_x** 和 **window_y** 需要由当前文件之外的代码提供。

下方 benchmark 比较三种启动配置。80 个 block、每个 block 128 个线程，只说明这次提交了 80 个 block；CUDA 会把 block 动态调度到可用 SM，代码没有把 80 个 block 固定绑定到 80 个 SM，也没有把 4 个 warp 固定绑定到 4 个 warp scheduler。0.09～0.14 ms 只代表当前设备和参数，且计时是否包含 GPU 同步会直接影响结果。

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
