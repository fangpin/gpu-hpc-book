# 高级 cuda 特性 —— 从矩阵乘法(gemm)说起

```{contents} 本页目录
---
depth: 2
local: true
---
```

以优化矩阵乘法 general matrix multiplication (gemm) 为切入点，介绍高级 cuda 特性在性能优化中的使用。

## 使用Roofline Model对矩阵乘法进行性能分析。

矩阵乘法 C = AB（其中 A, B, C 均为 n 行 n 列）的计算量是固定的，但**访存量**取决于内存访问的实现方式（即数据复用程度）。

- **计算量 (FLOPS)**：总计约 2n^3 次浮点运算。
- **数据大小**：假设使用 `float`（4 字节）。

### 情况 A：无缓存/无复用的实现 (Naive)

每个线程计算 $C_{ij}$ 时都从全局内存读取 $A$ 的一行和 $B$ 的一列。

- **访存量**：计算每个 C_ij 需要读取 2n 个元素。总访存为 2n^3 个元素。
- **字节数**： 8n^3 Bytes。
- $\text{OI}_{naive} = \frac{2n^3}{8n^3} = 0.25 \text{ FLOPs/Byte}$

> **结论**：如果不使用缓存，无论 n 多大，OI 始终极低，永远是 **Memory-bound**。

### 情况 B：理想状态/完全复用 (Ideal Cache)

假设 $A$ 和 $B$ 只从全局内存加载一次并完全留在缓存中。

- **访存量**：读取 A, B 并写入 C，总共 3n^2 个元素。
- **字节数**：12n^2 Bytes。
- $\text{OI}_{ideal} = \frac{2n^3}{12n^2} = \frac{n}{6} \text{ FLOPs/Byte}$

> **结论**：OI 随 $n$ 线性增长。这意味着随规模扩大，算法会从内存受限转为计算受限。

### 从 Memory-bound 到 Compute-bound 的转变

要找到这个转折点，我们需要对比**算法的 OI** 和 **GPU 的硬件拐点 (Ridge Point)**。

以 RTX 4000 Ada 为例.

- **Peak FP32 Performance**: 26.7 TFLOPS
- **Peak Memory Bandwidth**: 360 GB/s
- **硬件拐点:** $\text{OI}_{gpuhw} = \frac{26700 \text{ GFLOPS}}{360 \text{ GB/s}} \approx 74.17 \text{ FLOPs/Byte}$

$$\frac{n}{6} \ge 74.17$$

$$n \ge 74.17 \times 6$$

$$n \ge 445$$

> 结论：理论上当 n>=445 时，程序进入 **Compute-bound** 区域。

**你想让我帮你推导一下使用 Shared Memory Tiling 时具体的 OI 公式吗？这能帮你更精确地预测性能。**



## 显性的矩阵乘法性能分析

接着对 3072 * 3072 矩阵乘法（GEMM）在 **RTX 4000 Ada** 上的性能分析，涵盖了从理论峰值到不同缓存层级限制下的深度计算。

### 基本数据

- **计算峰值 (Peak FP32)**: 26.7 \text TFLOPS}$ (忽略 Tensor Cores)
- **内存带宽 (Peak DRAM Bandwidth)**: $360 \text{ GB/s}$
- **L2 缓存带宽**: $\approx 2.5 \text{ TB/s}$
- **矩阵规模**: $n = 3072$
- **总运算量**: $2n^3 = 2 \times (3072)^3 \approx 5.798 \times 10^{10} \text{ FLOPs}$

---

### 理论极限分析

#### (1) 如果是计算受限 (Compute-bound)，最快运行时间？

假设程序能完美跑满 26.7 TFLOPS：

$$\text{Time}_{comp} = \frac{2n^3}{\text{Peak GFLOPS}} = \frac{57.98 \text{ GFLOPs}}{26700 \text{ GFLOPS}} \approx \mathbf{2.17 \text{ ms}}$$

#### (2) 如果是 DRAM 受限且具备理想复用，最快运行时间？

理想情况下，每个元素A, B, C 仅从 DRAM 读写一次。总数据量为 3n^2 个 `float`：

$$\text{Bytes} = 3 \times (3072)^2 \times 4 \text{ Bytes} \approx 113.25 \text{ MB} = 0.11325 \text{ GB}$$

$$\text{Time}_{ideal_dram} = \frac{0.11325 \text{ GB}}{360 \text{ GB/s}} \approx \mathbf{0.31 \text{ ms}}$$

#### (3) 结论：谁统治了运行时间？

比较 (1) 和 (2)：**计算时间 2.17 ms) 远大于 DRAM 搬运时间 (0.31 ms)**。

这意味着，在经过良好优化（如 Shared Memory Tiling）并实现高数据复用的情况下，该工作负载的运行时间应由**计算（Compute）主导**。此时，数据的搬运被隐藏在计算背后。

### 缓存缺失与带宽约束分析

#### (4) 若完全不复用数据，总访存量是多少？

如果不利用缓存，每次 $C_{i,j} = \sum A_{i,k} \times B_{k,j}$ 都要从 DRAM 重新读取 $A$ 和 B：

- 每次乘加操作加载 2 个元素（8 字节）。
- 总操作次数为 $n^3$（乘加对）。
- $\text{Total Bytes} = 8 \times n^3 = 8 \times (3072)^3 \approx \mathbf{231.87 \text{ GB}}$

#### (5) 无复用情况下，DRAM 带宽约束下的最快时间？

$$\text{Time}_{no_reuse_dram} = \frac{231.87 \text{ GB}}{360 \text{ GB/s}} \approx \mathbf{644.08 \text{ ms}}$$

这比计算受限的理论时间慢了近 **300 倍**，说明了不进行 Tiling 优化的代价。

#### (6) 若所有操作直接命中 L2 缓存，最快运行时间？

假设 L2 提供 $2.5 \text{ TB/s}$ 2500  GB/s) 的带宽：

$$\text{Time}_{L2_bound} = \frac{231.87 \text{ GB}}{2500 \text{ GB/s}} \approx \mathbf{92.75 \text{ ms}}$$

---

### 综合对比总结


| 场景 | 限制因素 | 理论最快耗时 | 与计算极限 (2.17ms) 相比 |
| --- | --- | --- | --- |
| 理想 Tiling 优化 | Compute | 2.17 ms | 1.0x (瓶颈在 ALU) |
| 无复用 (Naive) | DRAM Bandwidth | 644.08 ms | ~296x 慢 |
| 无复用 (L2 命中) | L2 Bandwidth | 92.75 ms | ~42x 慢 |


**核心结论：**

- (5) 和 (6) 均远慢于 (1)。这证明了**内存层次结构（Memory Hierarchy）**存在的意义。
- 即使是极快的 L2 缓存，如果在算法层面上不通过 **Shared Memory** 或 **Register Tiling** 来实现寄存器级别的数据复用，其带宽依然无法支撑 GPU 庞大的算力。
- **优化目标**：将访存模式从场景 (5) 推动到场景 (2)，使得最终瓶颈落在场景 (1) 的计算曲线上。



## 矩阵乘法中的分块与数据复用

- **核心挑战**：矩阵乘法的计算复杂度是 O(N^3)，而数据量是 N^2。如果直接从全局内存（Global Memory）读取数据，程序将受限于内存带宽（Memory Bound），而非算力。
- **优化思路**：利用 **Shared Memory (L1 SRAM)** 和 **Registers** 进行数据复用，减少对低速全局内存的访问。

两个层级的复用：

### Shared Memory Tiling (块级复用)

- **原理**：将大矩阵划分为较小的 Tile（例如 16 * 16） 。每个 Thread Block 负责计算 C 的一个 Tile。
- **操作**：

  1. 从全局内存加载 A 的一个小块和 B 的一个小块到 Shared Memory。
  2. 调用 `__syncthreads()` 同步，确保块内所有线程都完成了加载。
  3. 在 Shared Memory 中进行计算。
  4. 重复上述过程，直到完成 K 维的遍历。
- **收益**：全局内存的访问次数减少了 Tile_Size 倍 (从 2N^3 下降到 2N^3/Tile_size)。例如tile为16*16时，内存防卫此时减少约93%。

**参考实现：**

```c++

__global__ void matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
        extern __shared__ float sh[];
        float *sh_a = sh;
        float *sh_b = sh + blockDim.x * blockDim.y;

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float vc = 0.0f;

        for (int k=0; k<(size_k + blockDim.x - 1)/blockDim.x; ++k) {
            int gy_a = blockIdx.y * blockDim.y + threadIdx.y;
            int gx_a = k * blockDim.x + threadIdx.x;
            if (gy_a < size_i && gx_a < size_k) {
                sh_a[threadIdx.y * blockDim.x + threadIdx.x] = a[gy_a * size_k + gx_a];
            } else {
                sh_a[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
            }

            int gy_b = k * blockDim.y + threadIdx.y;
            int gx_b = blockIdx.x * blockDim.x + threadIdx.x;
            if (gy_b < size_k && gx_b < size_j) {
                sh_b[threadIdx.y * blockDim.x + threadIdx.x] = b[gy_b * size_j + gx_b];
            } else {
                sh_b[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
            }

            __syncthreads();

            #pragma unroll
            for (int t_i=0; t_i < blockDim.x; ++t_i) {
                vc += sh_a[threadIdx.y * blockDim.x + t_i] * sh_b[t_i * blockDim.x + threadIdx.x];
            }
            __syncthreads();
        }

        if (row < size_i && col < size_j) {
            c[row * size_j + col] = vc;
        }
}

void launch_matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
        static constexpr int32_t block_size = 32;
        int32_t grid_y = (size_i + block_size - 1) / block_size;
        int32_t grid_x = (size_j + block_size - 1) / block_size;

        static constexpr int32_t share_mem_bytes = 2 * block_size * block_size * sizeof(float);

        CUDA_CHECK(cudaFuncSetAttribute(
            matmul_l1,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            share_mem_bytes));

        dim3 block(block_size, block_size);
        dim3 grid(grid_x, grid_y);
        matmul_l1<<<grid, block, share_mem_bytes>>>(size_i, size_j, size_k, a, b, c);
        CUDA_CHECK(cudaGetLastError());
}
```

**性能测试：**

```yaml
matmul_l1:

  size  256 *  256 *  256:
    correctness: 0.00e+00 relative RMSE
    run time:   0.02 ms
    throughput:  1.82 TFLOP/s

  size 3072 * 3072 * 3072:
    correctness: 1.03e-06 relative RMSE
    run time:  18.93 ms
    throughput:  3.06 TFLOP/s
```

### Register Tiling (线程级复用)

- **原理**：在 Thread Block 内部，进一步让每个线程负责输出 C 矩阵中的多个元素（例如 8 * 8 的微块）。
- **操作**：每个线程维护一组寄存器作为累加器（Accumulators）。在计算时，将 Shared Memory 中的数据加载到寄存器中进行多次计算。
- **收益**：

  - 极大减少了对 Shared Memory 的访问频率(寄存器的访问延迟比shared memory更低)
  - 通过增加每个线程的计算量（Compute Intensity），更好地隐藏指令延迟。也就是提高算术强度(参考roofline model章节)，降低memory bound。

**参考实现：**

```c++
template <int MB, int TS>
__global__ void matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
        const int block_size = blockDim.x;

        extern __shared__ float sh[];
        float* sh_a = sh;
        float* sh_b = sh + TS * block_size;

        int row_start = blockIdx.y * TS + threadIdx.y * MB;
        int col_start = blockIdx.x * TS + threadIdx.x * MB;

        float vc[MB][MB];
        #pragma unroll
        for (int i=0; i<MB; ++i) {
            #pragma unroll
            for (int j=0; j<MB; ++j) {
                vc[i][j] = 0.0f;
            }
        }

        for (int k=0; k<(size_k + blockDim.x - 1)/blockDim.x; ++k) {
            #pragma unroll
            for (int i=0; i<MB; ++i) {
                int gy_a = row_start + i;
                int gx_a = k * block_size + threadIdx.x;
                int sh_idx = threadIdx.y * TS + i * block_size + threadIdx.x;
                if (gy_a < size_i && gx_a < size_k) {
                    sh_a[sh_idx] = a[gy_a * size_k + gx_a];
                } else {
                    sh_a[sh_idx] = 0.0f;
                }
            }
            #pragma unroll
            for (int i=0; i<MB; ++i) {
                int gy_b = k * block_size + threadIdx.y;
                int gx_b = col_start + i;
                int sh_idx = threadIdx.y * TS + threadIdx.x * MB + i;
                if (gy_b < size_k && gx_b < size_j) {
                    sh_b[sh_idx] = b[gy_b * size_j + gx_b];
                } else {
                    sh_b[sh_idx] = 0.0f;
                }
            }
            __syncthreads();

            float va[MB], vb[MB];
            for (int i=0; i<blockDim.x; ++i) {
                #pragma unroll
                for (int j=0; j<MB; ++j) {
                    va[j] = sh_a[threadIdx.y * TS + j * block_size + i];
                    vb[j] = sh_b[i * TS + threadIdx.x * MB + j];
                }
                #pragma unroll
                for (int j=0;j<MB; ++j) {
                    #pragma unroll
                    for (int k=0;k<MB; ++k) {
                        vc[j][k] += va[j] * vb[k];
                    }
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int i=0; i<MB; ++i) {
            int row = row_start + i;
            #pragma unroll
            for (int j=0; j<MB; ++j) {
                int col = col_start + j;
                if (row < size_i && col < size_j) {
                    c[row * size_j + col] = vc[i][j];
                }
            }
        }
}

void launch_matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
        constexpr int block_size = 16;
        constexpr int micro_block_size = 8;
        constexpr int tile_size = block_size * micro_block_size;

        const int grid_y = (size_i + tile_size - 1) / tile_size;
        const int grid_x = (size_j + tile_size - 1) / tile_size;

        const int share_mem_bytes = 2 * tile_size * block_size * sizeof(float);
        CUDA_CHECK(cudaFuncSetAttribute(
            matmul_l1_reg<micro_block_size, tile_size>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            share_mem_bytes));

        dim3 block(block_size, block_size);
        dim3 grid(grid_x, grid_y);
        matmul_l1_reg<micro_block_size, tile_size><<<grid, block, share_mem_bytes>>>(size_i, size_j, size_k, a, b, c);
        CUDA_CHECK(cudaGetLastError());
}
```

**性能测试：（对比Shared Memory Tiling）**

寄存器复用比shared memory复用，带来了2.88x倍性能提升。

```yaml
./matmul 
matmul_l1:

  size  256 *  256 *  256:
    correctness: 0.00e+00 relative RMSE
    run time:   0.02 ms
    throughput:  1.82 TFLOP/s

  size 3072 * 3072 * 3072:
    correctness: 1.03e-06 relative RMSE
    run time:  18.93 ms
    throughput:  3.06 TFLOP/s

matmul_l1_reg:

  size  256 *  256 *  256:
    correctness: 0.00e+00 relative RMSE
    run time:   0.09 ms
    throughput:  0.37 TFLOP/s

  size 3072 * 3072 * 3072:
    correctness: 1.03e-06 relative RMSE
    run time:   6.57 ms
    throughput:  8.82 TFLOP/s

speedups on largest problem size:

  speedup matmul_l1 -> matmul_l1_reg: 2.88x
```



## 利特尔法则与占用率控制

无论是多线程并行还是指令级并行（ILP），其核心目标都是让硬件保持忙碌，不让计算单元因等待内存数据而“空转”。

### **利特尔法则 (Little's Law)：**

要实现峰值吞吐量，我们需要多少并行度？利特尔法则给出了简洁的答案：

$$\text{Required Parallelism} = \text{Latency} \times \text{Throughput}$$

**应用举例：**计算 DRAM 饱和所需的在途字节数

**已知参数：**

- **DRAM 延迟**: 800 cycles
- **GPU 时钟频率**: 2175 MHz
- **DRAM 带宽**: 360 GB/s

**推导过程：**

1. 首先计算每个时钟周期的理想吞吐量（Throughput per cycle）：
2. $\frac{360 \times 10^9 \text{ bytes/s}}{2.175 \times 10^9 \text{ cycles/s}} \approx 165.52 \text{ bytes/cycle}$
3. 根据利特尔法则，计算在途（In-flight）字节数：
4. $\text{Required Parallelism} = 800 \text{ cycles} \times 165.52 \text{ bytes/cycle} \approx 132,414 \text{ bytes}$

**结论：** 硬件必须在任何时刻都有约 **132.4 KB** 的内存请求正在处理中，才能填满 DRAM 带宽的流水线。

### 占用率 (Occupancy) 的计算

占用率衡量了 SM（流式多处理器）的利用程度。

假设：

- **Max Warps/SM**: 48
- **Max Shared Memory/SM**: 100 KB
- 每个 Block 运行 64 线程（即 2 个 Warps），不限制寄存器。



**应用举例：共享内存对活跃 Block 数的影响**

活跃 Block 数受限于 min(线程总数限制, Shared Memory 限制)。


| 每个 Block 共享内存 | 基于 Shmem 的限制 (102400 / x) | 活跃 Block 数 (上限 24) | 活跃 Warps 数 (Block × 2) | 占用率 (Warps / 48) |
| --- | --- | --- | --- | --- |
| 1,000 字节 | 102 | 24 | 48 | 100% |
| 10,000 字节 | 10.24 | 10 | 20 | 41.70% |
| 30,000 字节 | 3.41 | 3 | 6 | 12.50% |


### 人工控制占用率

通过控制每个block中的共享内存大小，block中thread数目，可以决定gpu的占用率。如果是memory bound型计算，占用率过低将导致延迟无法被并行掩盖。



**达到峰值带宽所需的最小占用率：**

在内存密集型任务（如 `memcpy`）中，通常 **33% - 50%** 的占用率就足以达到峰值带宽。这是因为一旦“在途数据”满足了利特尔法则计算出的 132 KB 阈值，进一步增加线程只会增加排队时间，而不会提升吞吐量。



### 总结

1. **并行不只是越多越好**：超过饱和点后的占用率提升对带宽没有贡献。
2. **利特尔法则是调优的指南针**：它告诉了我们硬件流水线的“深度”。
3. **资源权衡**：通过 Shared Memory 或寄存器限制 Occupancy 是调试性能瓶颈的常用技巧。

在实际开发中，建议优先使用 `float4` 等向量化手段增加每个线程的指令级并行度（ILP），这样可以在较低的占用率下就实现带宽饱和，从而为其他计算密集型指令留出更多的寄存器空间。



在高性能计算（HPC）与 GPU 编程领域，矩阵乘法（Matrix Multiplication）的优化始终是衡量硬件利用率的核心基准。在完成基础的数据复用优化后，进一步的性能提升往往取决于对硬件调度（Scheduling）**与**占用率（Occupancy）的深度掌控。

本篇笔记整理自针对矩阵乘法改进调度的技术实验，重点探讨如何通过改进调度策略来最大化硬件资源的并行潜力。

## 调度优化

通过优化GPU调度可以进一步优化矩阵乘法。以下是三种进阶调度方案：

### 隐藏访存延迟（Hiding Latency）

- **批处理访存（Batching）：** 避免创建细碎的访存与计算串行链。与其一次加载单行/单列，不如在 k 维度上一次性发射（issue）一批内存请求，随后集中处理计算。
- **访存层次结构：** Global Memory 延迟约 700-800 周期，而 L1/Shared Memory 仅需约 35 周期。优化目标是尽量让计算逻辑在等待 Global Data 到达时有其他任务可做。

### 重叠计算与搬运（Overlapping Data Movement & Computation）

这是提升硬件利用率最高效的手段，主要有三种实现路径：

- **路径 A：多 Block 调度（Multi-block Scheduling）**

  - 通过控制每个 Block 的资源占用，使得多个 Block 能同时驻留在单个 SM 上。当一个 Block 在等待内存同步（`__syncthreads()`）时，硬件会自动切换到另一个 Block 的 Warp 执行计算。开发者可以使用 `launch_bounds` 提示编译器进行寄存器优化。
- **路径 B：Warp 特化（Warp Specialization）**

  - 在同一个 Block 内手动划分 Warp 的职责：一部分 Warp 专门负责从 Global Memory 搬运数据到 L1，另一部分 Warp 专门负责执行计算逻辑。这种设计通常需要双缓冲（Double Buffering）技术，以避免数据在计算完成前被覆盖。
- **路径 C：异步拷贝（Async Copy）**

  - 利用 Ampere 架构引入的 `cp.async` 指令，数据可以直接从 Global Memory 传输到 Shared Memory，无需经过寄存器中转。
  
    - **优势：** 减少了寄存器压力，且 Warp 在拷贝期间不会被阻塞。
    - **管理：** 通过“提交组”（Commit Group）批量管理异步任务，并使用显式的等待（Wait）指令确保数据就绪。

## Tensor Cores

### 什么是 Tensor Core？

- **非传统核心**：它不是像 CPU/GPU 核心那样拥有独立指令流的处理器，而是一个**专用数学函数单元**（类似于高级 FPU/ALU）。
- **协作执行**：Tensor Core 指令以 **Warp（线程束，32 个线程）** 为基本单位。单线程无法独立调用，必须由 Warp 内的所有线程协同完成一个矩阵块的操作。
- **计算模式**：主要执行 D = A * B + C 的矩阵乘加运算。

### TF32 (TensorFloat-32) 精度

Tensor core 支持使用 TF32，它是专门为保持 FP32 兼容性而设计的格式：

- **特点**：接受 FP32 输入，但在内部计算时，将尾数（Mantissa）截断为 10 位（与 FP16 精度相当），同时保持 8 位指数（与 FP32 范围相当）。
- **权衡**：计算吞吐量远高于 FP32 FMA，但会有不可避免的精度损失。

### 关键指令详解：`mma.sync`

主要使用以下两条指令：

- **`mma.sync.aligned.m16n8k8...`**：

  - **矩阵尺寸**：A(16 * 8) * B(8 * 8) + C(16 * 8)$。
  - **寄存器消耗**：
  
    - **A**：由 Warp 分担，每线程需 4 个寄存器。
    - **B**：每线程需 2 个寄存器。
    - **C/D**：每线程需 4 个寄存器。

### 寄存器布局（Register Layout）—— 最难点

Tensor Core 要求数据在 Warp 的 32 个线程的寄存器中以**特定布局**排列。

- **A 矩阵 (Row-Major)**：被划分为四个象限，每个象限映射到一个特定的寄存器组。
- **B 矩阵 (Col-Major)**：垂直划分为两半，按列主序排列在寄存器中。
- **数据重排（Swizzling）**：由于内存中的数据通常是行主序，在载入寄存器之前或之后，必须通过代码（如 `__shfl_sync` 或特定的 Shared Memory 索引）进行重排，以满足 Tensor Core 的布局要求。

### **`wgmma`：warp-group level tensor core instructions**

Wgmma 是利用warp group level的tensor cores进行高性能矩阵乘法的关键。以H100为例，tensor core的 bf16 计算能力约为 1000 TFLOPs；而普通的FMA的 bf16 计算能力是 120 TFLOPs，大约只有tensor core的十分之一。因此使用 mma 进行矩阵运算几乎是必选项。

相比mma，wgmma有几个重要不同：

<table><colgroup><col/><col/><col/></colgroup><tbody><tr><td>不同点</td><td>mma</td><td>wgmma</td></tr><tr><td>工作单元</td><td>warp</td><td>Warp group，通常4warps</td></tr><tr><td>同步方式</td><td>同步</td><td>异步执行，可以结合TMA实现计算和存储交叉</td></tr><tr><td>计算逻辑</td><td>D=A*B+C</td><td>D=A*B+D。<ul><li>支持A，B转制，取相反数；</li><li>支持D 乘 0 或 1</li></ul></td></tr><tr><td>数据位置</td><td>均在寄存器内</td><td><ul><li>A 寄存器或shared memory</li><li>B shared memory</li><li>D 寄存器</li></ul></td></tr><tr><td>Tile 大小</td><td><code>M = 16</code>, <code>N = 8</code>, and <code>K = 16</code></td><td><code>M = 64</code>, <code>K = 16</code>, and <code>N</code> ranging from 8 to 256 in steps of 8</td></tr><tr><td>是否支持swizzling pattern</td><td>否</td><td>是</td></tr></tbody></table>
