# GPU 上动态的调度 —— 从run-length-compression说起

之前的例子中，我们对GPU的HBM的访问模式是静态可确定的，比如访问多少数据，访问的频率等在代码编译阶段就已经确定。

现实中另外一类常见的case是，GPU对HBM的访问模式需要在runtime时，根据输入的计算结果，动态决定，也就是GPU的动态调度问题。

接下来我们用run-length encoding（RLE，游程压缩）的实现来说明这里动态调度的高性能计算如何在GPU上高效实现。

## 并行扫描问题

RLE 依赖于前缀和的计算，前缀和问题可以归纳为扫描问题：

- 从前往后扫描数据，形成最终的累计数据。比如前缀和，最大值等。

GPU如何实现高效的扫描呢？

- 整体使用分治思想
- 独立完成每个block的scan，再将每个 block 中元素加上上一个block最后元素位置的前缀和
- 每个block scan通过实现warp scan后，再将每个warp 中的元素加上上一个warp的最后元素位置的前缀和进行实现 

### Warp scan

使用 `__shfl_up_sync`，每一轮从 `delta` 个 lane 之前拿一个值过来，然后与当前线程的局部前缀合并：

```cpp
template <typename Word, typename T>
__device__ __forceinline__ Word pack_bits(const T &v) {
    static_assert(
        sizeof(Word) >= sizeof(T),
        "Word size must be greater than or equal to size of T");
    static_assert(std::is_trivially_copyable_v<T>, "type T should be trivally copyable");
    Word word;
    std::memcpy(&word, &v, sizeof(T));
    return word;
}

template <typename Word, typename T>
__device__ __forceinline__ T unpack_bits(const Word &bits) {
    static_assert(
        sizeof(Word) >= sizeof(T),
        "Word size must be greater than or equal to size of T");
    static_assert(std::is_trivially_copyable_v<T>, "type T should be trivally copyable");
    T v;
    std::memcpy(&v, &bits, sizeof(T));
    return v;
}

template <typename T>
__device__ __forceinline__ T shuffle_up_any(unsigned int mask, T v, int delta) {
    static_assert(
        sizeof(T) <= sizeof(uint64_t),
        "do not support type that size is greater than 8");
    if constexpr (sizeof(T) <= sizeof(int)) {
        auto bits = pack_bits<int, T>(v);
        auto other_bits = __shfl_up_sync(mask, bits, delta);
        return unpack_bits<int, T>(other_bits);
    }
    auto bits = pack_bits<long long, T>(v);
    auto other_bits = __shfl_up_sync(mask, bits, delta);
    return unpack_bits<long long, T>(other_bits);
}

template <typename Op> __device__ typename Op::Data warp_scan(typename Op::Data v) {
    using Data = typename Op::Data;
    unsigned int mask = 0xffffffff;

    int lane = threadIdx.x & (WARP_SIZE - 1);
#pragma unroll
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        Data other = Op::identity();
        other = shuffle_up_any(mask, v, i);
        if (lane >= i) {
            v = Op::combine(other, v);
        }
    }

    return v;
}
```

这里的 `shuffle_up_any` 也值得一提。因为 `__shfl_up_sync` 天然只擅长处理标量寄存器，代码里通过 `pack_bits` 和 `unpack_bits` 把任意不超过 8 字节、且 trivially copyable 的类型打包成 `int` 或 `long long`，从而支持 `uint32_t` 之外的 `DebugRange`。这正好对应了课程网页里“scan 应该是泛型”的要求。



### Block scan

只做 warp 内 scan 还不够，因为一个 block (通常)有 1024 个线程，也就是 32 个 warp。当前实现的 `block_scan` 使用了标准的两级结构：

1. 每个 warp 先独立做自己的 inclusive scan。
2. 每个 warp 的最后一个 lane 把该 warp 的总和写入 shared memory。
3. 第 0 个 warp 再对这些 warp sums 做一次 scan。
4. 除了第 0 个 warp 之外，其他 warp 再把自己的 warp prefix 加回本 warp 内的局部结果。

这就是一个很典型的 hierarchical scan。

```cpp
template <typename Op>
__device__ typename Op::Data
block_scan(typename Op::Data v, typename Op::Data *shared_warp_sums) {
    using Data = typename Op::Data;

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;

    v = warp_scan<Op>(v);
    if (lane == WARP_SIZE - 1) {
        shared_warp_sums[warp_id] = v;
    }

    __syncthreads();

    Data *scaned_warp_sums = shared_warp_sums;

    if (warp_id == 0) {
        Data warp_sum =
            lane < NUM_WARPS_PER_BLOCK ? shared_warp_sums[lane] : Op::identity();
        Data sum = warp_scan<Op>(warp_sum);
        scaned_warp_sums[lane] = sum;
    }

    __syncthreads();

    if (warp_id > 0) {
        Data prefix = scaned_warp_sums[warp_id - 1];
        v = Op::combine(prefix, v);
    }

    return v;
}
```



### grid 级 scan：递归扫描 block sums

单个 block 的结果还只是局部前缀和。要把整个数组拼起来，当前代码进一步引入了两个全局缓冲区：

- `block_sums`：每个 block 的总和
- `scanned_block_sums`：对 `block_sums` 再做一次 scan 的结果

整个流程是：

1. `blocks_scan_no_fixup` 对每个 block 内做扫描，并写出 `output` 和 `block_sums`。
2. `launch_scan_recursive` 递归地对 `block_sums` 做同样的扫描，得到 `scanned_block_sums`。
3. `fixup_kernel` 把前面所有 block 的前缀值加回当前 block 的每个元素。

```cpp
template <typename Op>
__global__ void blocks_scan_no_fixup(
    const typename Op::Data *input,
    size_t n,
    typename Op::Data *output,
    typename Op::Data *block_sums) {

    using Data = typename Op::Data;
    extern __shared__ unsigned char sh_mem_raw[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    Data v = tid < n ? input[tid] : Op::identity();

    Data *shared_warp_sums = reinterpret_cast<Data *>(sh_mem_raw);
    v = block_scan<Op>(v, shared_warp_sums);

    if (tid < n) {
        output[tid] = v;
    }

    if (threadIdx.x == blockDim.x - 1 && block_sums != nullptr) {
        block_sums[blockIdx.x] = v;
    }
}

template <typename Op>
__global__ void
fixup_kernel(typename Op::Data *output, typename Op::Data *scanned_block_sums, size_t n) {
    using Data = typename Op::Data;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n) {
        return;
    }

    if (blockIdx.x == 0) {
        return;
    }

    Data prefix = scanned_block_sums[blockIdx.x - 1];
    output[tid] = Op::combine(prefix, output[tid]);
}
```



### 整体scan 流程

```cpp
template <typename Op>
void launch_scan_recursive(
    typename Op::Data *input,
    size_t n,
    typename Op::Data *output,
    unsigned char *workspace,
    size_t workspace_offset) {
    using Data = typename Op::Data;
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (n == 0) {
        return;
    }

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shared_bytes = 2 * NUM_WARPS_PER_BLOCK * sizeof(Data);
    if (num_blocks == 1) {
        blocks_scan_no_fixup<Op>
            <<<1, BLOCK_SIZE, shared_bytes>>>(input, n, output, nullptr);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    Data *block_sums = reinterpret_cast<Data *>(workspace + workspace_offset);
    workspace_offset += num_blocks * sizeof(Data);
    workspace_offset = (workspace_offset + 256) & ~255;

    Data *scanned_block_sums = reinterpret_cast<Data *>(workspace + workspace_offset);
    workspace_offset += num_blocks * sizeof(Data);
    workspace_offset = (workspace_offset + 256) & ~255;

    blocks_scan_no_fixup<Op>
        <<<num_blocks, BLOCK_SIZE, shared_bytes>>>(input, n, output, block_sums);
    CUDA_CHECK(cudaGetLastError());

    launch_scan_recursive<Op>(
        block_sums,
        num_blocks,
        scanned_block_sums,
        workspace,
        workspace_offset);
    CUDA_CHECK(cudaGetLastError());

    fixup_kernel<Op>
        <<<num_blocks, BLOCK_SIZE, shared_bytes>>>(output, scanned_block_sums, n);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Op>
typename Op::Data *launch_scan(
    size_t n,
    typename Op::Data *x, // pointer to GPU memory
    void *workspace       // pointer to GPU memory
) {
    using Data = typename Op::Data;
    Data *output = reinterpret_cast<Data *>(workspace);
    size_t workspace_offset = n * sizeof(Data);
    launch_scan_recursive<Op>(
        x,
        n,
        output,
        reinterpret_cast<unsigned char *>(workspace),
        workspace_offset);
    return output;
}

```



### workspace size



实际需要使用 scratch buffer，来存储中间结果这里需要两个 buffer：

1. `block_sums`
2. `scanned_block_sums`

每一层递归都需要存 block sums。为了简单，可以按 `2 * num_blocks * sizeof(Data)` 分配。



但递归扫描 block sums 时也需要 workspace。最安全的做法是把每一层需要的 block sums 加起来。

```cpp
template <typename Op> size_t get_workspace_size(size_t n) {
    using Data = typename Op::Data;

    size_t total = n * sizeof(Data);
    while (n > 1) {
        size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        total += 2 * num_blocks * sizeof(Data);
        n = num_blocks;
    }
    total += 4096;
    return total;
}
```



## Run length compression

run-length compression 可以很好展示了“如何把一个动态调度问题变成 scan 能解决的问题”。

run-length compression本身很简单，对应的cpu实现如下：

```cpp
void rle_compress_cpu(
    uint32_t raw_count,
    char const *raw,
    std::vector<char> &compressed_data,
    std::vector<uint32_t> &compressed_lengths) {
    compressed_data.clear();
    compressed_lengths.clear();

    uint32_t i = 0;
    while (i < raw_count) {
        char c = raw[i];
        uint32_t run_length = 1;
        i++;
        while (i < raw_count && raw[i] == c) {
            run_length++;
            i++;
        }
        compressed_data.push_back(c);
        compressed_lengths.push_back(run_length);
    }
}


```

这正是 GPU 最头疼的一类问题：不规则输出定位。

下面介绍一种并行算法设计中的经典套路。它并不直接问“每个 run 应该写到哪里”，而是先构造一个辅助视角：哪些位置是一个 run 的起点？

如果当前位置是某个新 run 的开始，就标记为 1；否则标记为 0。这样一来，原始问题就变成了一个 0/1 数组上的计数问题。而一旦是计数问题，scan 就能派上用场了。

对这个 0/1 数组做前缀和后，每个位置都会知道“到我这里为止，已经出现了多少个 run”。于是，如果某个位置恰好是 run 的起点，那么它在压缩输出中的下标就等于这个前缀和减一。原本那个“必须等前面都做完才知道写哪里”的动态问题，就这样被转换成了一个可以并行计算的索引问题。

这一步转化非常关键，也非常有代表性。它说明很多不规则问题的难点，不在于“GPU 不能算”，而在于你有没有找到一个合适的中间表示。这里的中间表示就是 run boundary，也就是“run 起点标记”。一旦找到它，后面的工作就重新回到了 GPU 擅长的轨道上：局部判断、前缀扫描、按索引写入。

事实上，这种“标记 + scan + 压缩写出”的模式在 GPU 编程里极其常见。比如过滤满足条件的元素、删除无效项、紧凑存储、生成输出索引，背后都常常能看到 scan 的影子。scan 并不是一个孤立的算法题，而更像是并行程序设计中的一种通用基础设施。

完整的实现可以参考：

```cpp
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 1024;
constexpr int NUM_WARPS_PER_BLOCK = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;

__global__ void mark_flags(char const *raw, uint32_t *flags, uint32_t raw_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= raw_count) {
        return;
    }
    flags[tid] = (tid == 0 || raw[tid] != raw[tid - 1]) ? 1u : 0u;
}

__global__ void scatter_run_starts(
    char const *raw,
    uint32_t const *flags,
    uint32_t const *scanned_flags,
    uint32_t *run_starts,
    char *compressed_data,
    uint32_t raw_count) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= raw_count || flags[tid] == 0) {
        return;
    }

    uint32_t run_idx = scanned_flags[tid] - 1;
    run_starts[run_idx] = tid;
    compressed_data[run_idx] = raw[tid];
}

__global__ void finalize_run_lengths(
    uint32_t const *run_starts,
    uint32_t compressed_count,
    uint32_t raw_count,
    uint32_t *compressed_lengths) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= compressed_count) {
        return;
    }

    uint32_t start = run_starts[tid];
    uint32_t end = tid + 1 < compressed_count ? run_starts[tid + 1] : raw_count;
    compressed_lengths[tid] = end - start;
}

// Returns desired size of scratch buffer in bytes.
size_t get_workspace_size(uint32_t raw_count) {
    size_t total = 0;

    total += raw_count * sizeof(uint32_t); // flags
    total = (total + 255) & ~size_t(255);

    total += raw_count * sizeof(uint32_t); // scanned_flags
    total = (total + 255) & ~size_t(255);

    total += raw_count * sizeof(uint32_t); // run_starts
    total = (total + 255) & ~size_t(255);

    uint32_t n = raw_count;
    while (n > 1) {
        size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        total += 2 * num_blocks * sizeof(uint32_t);
        total = (total + 255) & ~size_t(255);
        n = num_blocks;
    }

    total += 4096;
    return total;
}

template <typename Word, typename T>
__device__ __forceinline__ Word pack_bits(const T &v) {
    static_assert(
        sizeof(Word) >= sizeof(T),
        "Word size must be greater than or equal to size of T");
    static_assert(std::is_trivially_copyable_v<T>, "type T should be trivally copyable");
    Word word;
    std::memcpy(&word, &v, sizeof(T));
    return word;
}

template <typename Word, typename T>
__device__ __forceinline__ T unpack_bits(const Word &bits) {
    static_assert(
        sizeof(Word) >= sizeof(T),
        "Word size must be greater than or equal to size of T");
    static_assert(std::is_trivially_copyable_v<T>, "type T should be trivally copyable");
    T v;
    std::memcpy(&v, &bits, sizeof(T));
    return v;
}

template <typename T>
__device__ __forceinline__ T shuffle_up_any(unsigned int mask, T v, int delta) {
    static_assert(
        sizeof(T) <= sizeof(uint64_t),
        "do not support type that size is greater than 8");
    if constexpr (sizeof(T) <= sizeof(int)) {
        auto bits = pack_bits<int, T>(v);
        auto other_bits = __shfl_up_sync(mask, bits, delta);
        return unpack_bits<int, T>(other_bits);
    }
    auto bits = pack_bits<long long, T>(v);
    auto other_bits = __shfl_up_sync(mask, bits, delta);
    return unpack_bits<long long, T>(other_bits);
}

template <typename Op> __device__ typename Op::Data warp_scan(typename Op::Data v) {
    using Data = typename Op::Data;
    unsigned int mask = 0xffffffff;

    int lane = threadIdx.x & (WARP_SIZE - 1);
#pragma unroll
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        Data other = Op::identity();
        other = shuffle_up_any(mask, v, i);
        if (lane >= i) {
            v = Op::combine(other, v);
        }
    }

    return v;
}

template <typename Op>
__device__ typename Op::Data
block_scan(typename Op::Data v, typename Op::Data *shared_warp_sums) {
    using Data = typename Op::Data;

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;

    v = warp_scan<Op>(v);
    if (lane == WARP_SIZE - 1) {
        shared_warp_sums[warp_id] = v;
    }

    __syncthreads();

    Data *scaned_warp_sums = shared_warp_sums;

    if (warp_id == 0) {
        Data warp_sum =
            lane < NUM_WARPS_PER_BLOCK ? shared_warp_sums[lane] : Op::identity();
        Data sum = warp_scan<Op>(warp_sum);
        scaned_warp_sums[lane] = sum;
    }

    __syncthreads();

    if (warp_id > 0) {
        Data prefix = scaned_warp_sums[warp_id - 1];
        v = Op::combine(prefix, v);
    }

    return v;
}

template <typename Op>
__global__ void blocks_scan_no_fixup(
    const typename Op::Data *input,
    size_t n,
    typename Op::Data *output,
    typename Op::Data *block_sums) {

    using Data = typename Op::Data;
    extern __shared__ unsigned char sh_mem_raw[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    Data v = tid < n ? input[tid] : Op::identity();

    Data *shared_warp_sums = reinterpret_cast<Data *>(sh_mem_raw);
    v = block_scan<Op>(v, shared_warp_sums);

    if (tid < n) {
        output[tid] = v;
    }

    if (threadIdx.x == blockDim.x - 1 && block_sums != nullptr) {
        block_sums[blockIdx.x] = v;
    }
}

template <typename Op>
__global__ void
fixup_kernel(typename Op::Data *output, typename Op::Data *scanned_block_sums, size_t n) {
    using Data = typename Op::Data;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n) {
        return;
    }

    if (blockIdx.x == 0) {
        return;
    }

    Data prefix = scanned_block_sums[blockIdx.x - 1];
    output[tid] = Op::combine(prefix, output[tid]);
}

template <typename Op>
void launch_scan_recursive(
    typename Op::Data *input,
    size_t n,
    typename Op::Data *output,
    unsigned char *workspace,
    size_t workspace_offset) {
    using Data = typename Op::Data;
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (n == 0) {
        return;
    }

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shared_bytes = 2 * NUM_WARPS_PER_BLOCK * sizeof(Data);
    if (num_blocks == 1) {
        blocks_scan_no_fixup<Op>
            <<<1, BLOCK_SIZE, shared_bytes>>>(input, n, output, nullptr);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    Data *block_sums = reinterpret_cast<Data *>(workspace + workspace_offset);
    workspace_offset += num_blocks * sizeof(Data);
    workspace_offset = (workspace_offset + 256) & ~255;

    Data *scanned_block_sums = reinterpret_cast<Data *>(workspace + workspace_offset);
    workspace_offset += num_blocks * sizeof(Data);
    workspace_offset = (workspace_offset + 256) & ~255;

    blocks_scan_no_fixup<Op>
        <<<num_blocks, BLOCK_SIZE, shared_bytes>>>(input, n, output, block_sums);
    CUDA_CHECK(cudaGetLastError());

    launch_scan_recursive<Op>(
        block_sums,
        num_blocks,
        scanned_block_sums,
        workspace,
        workspace_offset);
    CUDA_CHECK(cudaGetLastError());

    fixup_kernel<Op>
        <<<num_blocks, BLOCK_SIZE, shared_bytes>>>(output, scanned_block_sums, n);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Op>
typename Op::Data *launch_scan(
    size_t n,
    typename Op::Data *x, // pointer to GPU memory
    typename Op::Data *output,
    void *workspace,
    size_t workspace_offset) {

    using Data = typename Op::Data;
    launch_scan_recursive<Op>(
        x,
        n,
        output,
        reinterpret_cast<unsigned char *>(workspace),
        workspace_offset);
    return output;
}

__forceinline__ size_t align256(const size_t &offset) {
    return (offset + 255) & ~size_t(255);
}

struct SumOp {
    using Data = uint32_t;

    static __host__ __device__ __forceinline__ Data identity() { return 0; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }

    static std::string to_string(Data d) { return std::to_string(d); }
};

// 'launch_rle_compress'
//
// Input:
//
//   'raw_count': Number of bytes in the input buffer 'raw'.
//
//   'raw': Uncompressed bytes in GPU memory.
//
//   'workspace': Scratch buffer in GPU memory. The size of the scratch buffer
//   in bytes is determined by 'get_workspace_size'.
//
// Output:
//
//   Returns: 'compressed_count', the number of runs in the compressed data.
//
//   'compressed_data': Output buffer of size 'raw_count' in GPU memory. The
//   function should fill the first 'compressed_count' bytes of this buffer
//   with the compressed data.
//
//   'compressed_lengths': Output buffer of size 'raw_count' in GPU memory. The
//   function should fill the first 'compressed_count' integers in this buffer
//   with the lengths of the runs in the compressed data.
//
uint32_t launch_rle_compress(
    uint32_t raw_count,
    char const *raw,             // pointer to GPU buffer
    void *workspace,             // pointer to GPU buffer
    char *compressed_data,       // pointer to GPU buffer
    uint32_t *compressed_lengths // pointer to GPU buffer
) {
    if (raw_count == 0) {
        return 0;
    }

    size_t offset = 0;
    unsigned char *ws = reinterpret_cast<unsigned char *>(workspace);

    uint32_t *flags = reinterpret_cast<uint32_t *>(ws + offset);
    offset += raw_count * sizeof(uint32_t);
    offset = align256(offset);

    uint32_t *scanned_flags = reinterpret_cast<uint32_t *>(ws + offset);
    offset += raw_count * sizeof(uint32_t);
    offset = align256(offset);

    uint32_t *run_starts = reinterpret_cast<uint32_t *>(ws + offset);
    offset += raw_count * sizeof(uint32_t);
    offset = align256(offset);

    int num_blocks = (raw_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    mark_flags<<<num_blocks, BLOCK_SIZE>>>(raw, flags, raw_count);
    CUDA_CHECK(cudaGetLastError());

    launch_scan<SumOp>(raw_count, flags, scanned_flags, workspace, offset);
    CUDA_CHECK(cudaGetLastError());

    uint32_t compressed_count = 0;
    CUDA_CHECK(cudaMemcpy(
        &compressed_count,
        scanned_flags + (raw_count - 1),
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost));

    scatter_run_starts<<<num_blocks, BLOCK_SIZE>>>(
        raw,
        flags,
        scanned_flags,
        run_starts,
        compressed_data,
        raw_count);
    CUDA_CHECK(cudaGetLastError());

    int run_blocks = (compressed_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    finalize_run_lengths<<<run_blocks, BLOCK_SIZE>>>(
        run_starts,
        compressed_count,
        raw_count,
        compressed_lengths);
    CUDA_CHECK(cudaGetLastError());

    return compressed_count;
}

```



## Run length decompression

Run length decompression则是Run length compression的逆过程。主要过程如下：

1. Scan compress_length 前缀和生成run_starts
2. 并写写数据：对第 i 个run知道了其在原始数据的起点为 run_starts[i], 重点为run_starts[i+1]。

其中并写数据的方案有多种：

1. 每个 thread 负责写一个run
2. 每个 warp 写一个run
3. 每个block写一个run
4. tiled：每个 warp 负责原始数据的一块连续区域

前三种方式很难单独使用，因为run的长度很不均匀，会导致gpu thread负载严重不平衡，通常需要先预先扫描每个run，根据run的工作量来进行灵活的分配。

  
下面给出第4方案的参考代码：

```cpp

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 1024;
constexpr int NUM_WARPS_PER_BLOCK = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;

template <typename Word, typename T>
__device__ __forceinline__ Word pack_bits(const T &v) {
    static_assert(
        sizeof(Word) >= sizeof(T),
        "Word size must be greater than or equal to size of T");
    static_assert(std::is_trivially_copyable_v<T>, "type T should be trivally copyable");
    Word word;
    std::memcpy(&word, &v, sizeof(T));
    return word;
}

template <typename Word, typename T>
__device__ __forceinline__ T unpack_bits(const Word &bits) {
    static_assert(
        sizeof(Word) >= sizeof(T),
        "Word size must be greater than or equal to size of T");
    static_assert(std::is_trivially_copyable_v<T>, "type T should be trivally copyable");
    T v;
    std::memcpy(&v, &bits, sizeof(T));
    return v;
}

template <typename T>
__device__ __forceinline__ T shuffle_up_any(unsigned int mask, T v, int delta) {
    static_assert(
        sizeof(T) <= sizeof(uint64_t),
        "do not support type that size is greater than 8");
    if constexpr (sizeof(T) <= sizeof(int)) {
        auto bits = pack_bits<int, T>(v);
        auto other_bits = __shfl_up_sync(mask, bits, delta);
        return unpack_bits<int, T>(other_bits);
    }
    auto bits = pack_bits<long long, T>(v);
    auto other_bits = __shfl_up_sync(mask, bits, delta);
    return unpack_bits<long long, T>(other_bits);
}

template <typename Op> __device__ typename Op::Data warp_scan(typename Op::Data v) {
    using Data = typename Op::Data;
    unsigned int mask = 0xffffffff;

    int lane = threadIdx.x & (WARP_SIZE - 1);
#pragma unroll
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        Data other = Op::identity();
        other = shuffle_up_any(mask, v, i);
        if (lane >= i) {
            v = Op::combine(other, v);
        }
    }

    return v;
}

template <typename Op>
__device__ typename Op::Data
block_scan(typename Op::Data v, typename Op::Data *shared_warp_sums) {
    using Data = typename Op::Data;

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;

    v = warp_scan<Op>(v);
    if (lane == WARP_SIZE - 1) {
        shared_warp_sums[warp_id] = v;
    }

    __syncthreads();

    Data *scaned_warp_sums = shared_warp_sums;

    if (warp_id == 0) {
        Data warp_sum =
            lane < NUM_WARPS_PER_BLOCK ? shared_warp_sums[lane] : Op::identity();
        Data sum = warp_scan<Op>(warp_sum);
        scaned_warp_sums[lane] = sum;
    }

    __syncthreads();

    if (warp_id > 0) {
        Data prefix = scaned_warp_sums[warp_id - 1];
        v = Op::combine(prefix, v);
    }

    return v;
}

template <typename Op>
__global__ void blocks_scan_no_fixup(
    const typename Op::Data *input,
    size_t n,
    typename Op::Data *output,
    typename Op::Data *block_sums) {

    using Data = typename Op::Data;
    extern __shared__ unsigned char sh_mem_raw[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    Data v = tid < n ? input[tid] : Op::identity();

    Data *shared_warp_sums = reinterpret_cast<Data *>(sh_mem_raw);
    v = block_scan<Op>(v, shared_warp_sums);

    if (tid < n) {
        output[tid] = v;
    }

    if (threadIdx.x == blockDim.x - 1 && block_sums != nullptr) {
        block_sums[blockIdx.x] = v;
    }
}

template <typename Op>
__global__ void
fixup_kernel(typename Op::Data *output, typename Op::Data *scanned_block_sums, size_t n) {
    using Data = typename Op::Data;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n) {
        return;
    }

    if (blockIdx.x == 0) {
        return;
    }

    Data prefix = scanned_block_sums[blockIdx.x - 1];
    output[tid] = Op::combine(prefix, output[tid]);
}

template <typename Op>
void launch_scan_recursive(
    typename Op::Data *input,
    size_t n,
    typename Op::Data *output,
    unsigned char *workspace,
    size_t workspace_offset) {
    using Data = typename Op::Data;
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (n == 0) {
        return;
    }

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shared_bytes = 2 * NUM_WARPS_PER_BLOCK * sizeof(Data);
    if (num_blocks == 1) {
        blocks_scan_no_fixup<Op>
            <<<1, BLOCK_SIZE, shared_bytes>>>(input, n, output, nullptr);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    Data *block_sums = reinterpret_cast<Data *>(workspace + workspace_offset);
    workspace_offset += num_blocks * sizeof(Data);
    workspace_offset = (workspace_offset + 256) & ~255;

    Data *scanned_block_sums = reinterpret_cast<Data *>(workspace + workspace_offset);
    workspace_offset += num_blocks * sizeof(Data);
    workspace_offset = (workspace_offset + 256) & ~255;

    blocks_scan_no_fixup<Op>
        <<<num_blocks, BLOCK_SIZE, shared_bytes>>>(input, n, output, block_sums);
    CUDA_CHECK(cudaGetLastError());

    launch_scan_recursive<Op>(
        block_sums,
        num_blocks,
        scanned_block_sums,
        workspace,
        workspace_offset);
    CUDA_CHECK(cudaGetLastError());

    fixup_kernel<Op>
        <<<num_blocks, BLOCK_SIZE, shared_bytes>>>(output, scanned_block_sums, n);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Op>
typename Op::Data *launch_scan(
    size_t n,
    typename Op::Data *x, // pointer to GPU memory
    typename Op::Data *output,
    void *workspace,
    size_t workspace_offset) {

    using Data = typename Op::Data;
    launch_scan_recursive<Op>(
        x,
        n,
        output,
        reinterpret_cast<unsigned char *>(workspace),
        workspace_offset);
    return output;
}

__forceinline__ size_t align256(const size_t &offset) {
    return (offset + 255) & ~size_t(255);
}

struct SumOp {
    using Data = uint32_t;

    static __host__ __device__ __forceinline__ Data identity() { return 0; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }

    static std::string to_string(Data d) { return std::to_string(d); }
};

size_t get_scan_workspace_size(size_t n) {
    size_t total = n * sizeof(uint32_t);
    while (n > 1) {
        size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        total += 2 * num_blocks * sizeof(uint32_t);
        n = num_blocks;
    }
    total += 4096;
    return total;
}

template <std::totally_ordered T>
__device__ size_t upper_bound(const T *data, const size_t length, const T &key) {
    int l = 0, r = length;
    while (l < r) {
        int m = l + (r - l) / 2;
        auto candidate = data[m];
        if (candidate <= key) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

__global__ void decompress_kernel(
    char const *compressed_data,
    uint32_t compressed_count,
    const uint32_t decompressed_count,
    char *const decompressed_data,
    const uint32_t *run_starts) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= decompressed_count) {
        return;
    }

    size_t run_id = upper_bound<uint32_t>(
                        run_starts,
                        static_cast<size_t>(compressed_count) + 1,
                        static_cast<uint32_t>(tid)) -
        1;
    decompressed_data[tid] = compressed_data[run_id];
}

// 'launch_rle_decompress'
//
// Input:
//
//   'compressed_count': Number of runs in the compressed data.
//
//   'compressed_data': Array of size 'compressed_count' in GPU memory,
//   containing the byte value for each run.
//
//   'compressed_lengths': Array of size 'compressed_count' in GPU memory,
//    containing the length of each run.
//
//   'workspace_alloc_1', 'workspace_alloc_2': 'GpuAllocCache' objects each of
//   which can be used to allocate a single GPU buffer of arbitrary size.
//
// Output:
//
//   Returns a 'Decompressed' struct containing the following:
//
//     'count': Number of bytes in the decompressed data.
//
//     'data': Pointer to the decompressed data in GPU memory. May point to a
//     buffer allocated using 'workspace_alloc_1' or 'workspace_alloc_2'.
//
Decompressed launch_rle_decompress(
    uint32_t compressed_count,
    char const *compressed_data,
    uint32_t const *compressed_lengths,
    GpuAllocCache &workspace_alloc_1,
    GpuAllocCache &workspace_alloc_2) {

    if (compressed_count == 0) {
        return {0, nullptr};
    }

    size_t run_starts_bytes =
        align256((static_cast<size_t>(compressed_count) + 1) * sizeof(uint32_t));
    size_t scan_workspace_bytes = get_scan_workspace_size(compressed_count);
    size_t total_workspace_bytes = run_starts_bytes + scan_workspace_bytes;

    auto *workspace =
        reinterpret_cast<unsigned char *>(workspace_alloc_1.alloc(total_workspace_bytes));
    auto *run_starts = reinterpret_cast<uint32_t *>(workspace);
    auto *scan_workspace = workspace + run_starts_bytes;
    auto *scan_input = reinterpret_cast<uint32_t *>(scan_workspace);

    CUDA_CHECK(cudaMemset(run_starts, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(
        scan_input,
        compressed_lengths,
        compressed_count * sizeof(uint32_t),
        cudaMemcpyDeviceToDevice));
    launch_scan<SumOp>(
        compressed_count,
        scan_input,
        run_starts + 1,
        scan_workspace,
        compressed_count * sizeof(uint32_t));
    CUDA_CHECK(cudaGetLastError());

    uint32_t decompressed_count = 0;
    CUDA_CHECK(cudaMemcpy(
        &decompressed_count,
        run_starts + compressed_count,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost));

    if (decompressed_count == 0) {
        return {0, nullptr};
    }

    char *decompressed_data =
        reinterpret_cast<char *>(workspace_alloc_2.alloc(decompressed_count));
    CUDA_CHECK(cudaGetLastError());

    const int n_blocks = (decompressed_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    decompress_kernel<<<n_blocks, BLOCK_SIZE>>>(
        compressed_data,
        compressed_count,
        decompressed_count,
        decompressed_data,
        run_starts);
    CUDA_CHECK(cudaGetLastError());

    return {decompressed_count, decompressed_data};
}

```

---

最后一次更新时间：`2026-08-05 16:12:21 CST`

原文链接：[https://fangpin.github.io/gpu-hpc-book/#/chapters/06-dynamic-scheduling-run-length-compression.md](https://fangpin.github.io/gpu-hpc-book/#/chapters/06-dynamic-scheduling-run-length-compression.md)
