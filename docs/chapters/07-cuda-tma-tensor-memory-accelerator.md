# Cuda TMA （Tensor Memory Accelerator）

 Cuda 从 Hopper 架构引入了 Tensor Memory Accelerator，简称 TMA。它是一个专门负责在shared memory和 global memory 之间搬运数据的专用协处理器。通过专用协处理器可以将数据拷贝从 warp 同步调度中分解出来，实现异步数据拷贝，并最终通过重叠计算和内存访问，提高GPU的内存带宽利用率。

## 基础介绍

TMA 可以执行四类操作：

1. global memory 到 shared memory 的 load；
2. shared memory 到 global memory 的 store；
3. shared 到 global 的 reduce，也就是边写回边做逐元素归约；
4. multicast，把同一份数据广播到多个 block 所在的 SM。



`cp.async 本质上还是“由 warp 驱动的细粒度异步拷贝指令”，而 TMA 本质上是“由描述符驱动的、独立硬件引擎执行的块级/张量级搬运”。 `

传统 `cp.async` 里，每个线程还是要自己参与这件事。通常是一个 warp 的 32 个 lane，各自给出 global address 和 shared memory address，每个 lane 搬一小段数据，典型是 4/8/16B。于是整个 warp 合起来完成一块数据的搬运。虽然它是异步的，但它仍然是“线程指令流的一部分”：warp 要发出这些 `cp.async` 指令，硬件按 lane 收集地址、形成内存事务，再把结果写进 shared memory。

TMA 则完全换了思路。程序先在 host 端构造一个 `CUtensorMap` 描述符，把张量的基地址、维度、stride、tile 大小、swizzle 等信息编码好。到 kernel 里，往往只需要一个 lane 发起一次 TMA 指令，并告诉硬件“我要第几个 tile”。之后真正的地址生成、多维步长展开、tile 内各元素的定位、实际内存请求的组织，都是 TMA 引擎自己完成的。也就是说，线程不再逐小块参与搬运过程，而是把“搬哪一块”这件事提交给硬件协处理器。

`TMA 和 cp.async` 相比，它的几个关键特点是：

- 搬运粒度从小块连续字节提升到 **tile 粒度**。
- 可以处理多维张量布局，而不是手写复杂地址计算。
- 一般只需要 **单个 lane 发起**，后面的搬运由 TMA 引擎独立完成。
- 这样可以减少指令发射压力，也能减少寄存器压力。

这意味着：

- 线程不用再为每个元素做大量 index arithmetic。
- 数据搬运和有用计算更容易重叠。
- 后面的多 warp pipeline 才有成立的基础。

## Tensor descriptor：`CUtensorMap`

TMA 不能凭空知道一块数据在 global memory 里长什么样，所以必须先在 host 端创建一个 tensor descriptor，再传给 kernel。Tensor descriptor 即`CUtensorMap`通常包含：

- `tensorDataType`：元素类型。这个 lab 里重点是 `bf16`。
- `tensorRank`：张量维度。
- `globalDim`：整个 tensor 每一维多大。
- `globalStrides`：global memory 中跨维移动时要跳过多少 **字节**。
- `boxDim`：一次 TMA 操作搬多大的 tile。
- `elementStrides`：写入 shared memory 时，各维元素的步长，单位是 **元素**，不是字节。

## TMA 完成异步数据搬运

因为 TMA 是异步的，所以 issue 一次拷贝之后，线程并不能立刻假设数据已经到达 shared memory。这个时候就必须有一种同步机制来判断“数据什么时候真正可用”。

对于 global-to-shared 的 TMA load，Hopper 使用的是 split barrier，也叫 `mbarrier`。这和我们平时熟悉的 `__syncthreads()` 很不一样。

`__syncthreads()` 是“到达和等待”绑定在一起的 join barrier。线程一旦到这里，就一边声明自己到了，一边阻塞等待所有人都到齐。

而 split barrier 把“arrival”和“wait”拆开了。线程可以先到达 barrier，之后去做别的事，过一会儿再来等待它完成。更重要的是，arrival 数量可以在运行时配置，所以你可以只同步某几个 lane，而不是整个 block。

在 TMA 场景下，`mbarrier` 不仅可以统计有多少线程到达，还能统计有多少字节的异步数据已经到达。换句话说，barrier 释放的条件是双重的：参与同步的 lane 已经满足 arrival 条件，而且 TMA 搬运的字节也真的完成了。这样 consumer 才不会过早读取尚未到达的 shared memory 数据。

TMA 的异步性必须配合 barrier 的状态管理，否则就会读到错误数据。

### 单 block、单 lane、单 tile 的 TMA load

一个 block，一个 lane，一次 TMA，把一块 tile 从 global memory 搬到 shared memory，再用普通 CUDA store 写回 global memory。

TMA load 用的是 `mbarrier`，因为 barrier 和目标 shared memory 在同一侧，硬件容易在数据到达 shared memory 时顺便更新 barrier 状态。但 TMA store 是 shared 到 global，写完成的信息如果还要再反向通知 shared memory 中的 barrier，就会涉及一次不自然的“往返确认”。因此硬件没有沿用 `mbarrier` 方案，而是采用 commit group 机制来追踪 store 是否完成。

示例代码：

```cpp
__device__ static __forceinline__ void async_proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ static __forceinline__ void init_barrier(uint64_t *bar,
                                                    int arrival_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(arrival_count)
               : "memory");
}

__device__ static __forceinline__ void arrive(uint64_t *bar, uint32_t count) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0],  %1;\n"
               :
               : "r"(mbar_ptr), "r"(count)
               : "memory");
}

__device__ static __forceinline__ int try_wait(uint64_t *bar, int phaseParity) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  int result;
  asm volatile("{\n"
               ".reg .pred P1;\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
               "selp.u32 %0,1,0,P1;"
               "}\n"
               : "=r"(result)
               : "r"(mbar_ptr), "r"(phaseParity));
  return result;
}

__device__ static __forceinline__ int test_wait(uint64_t *bar,
                                                int phaseParity) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  int result;
  asm volatile("{\n"
               ".reg .pred P1;\n"
               "mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
               "selp.u32 %0,1,0,P1;"
               "}\n"
               : "=r"(result)
               : "r"(mbar_ptr), "r"(phaseParity));
  return result;
}

__device__ static __forceinline__ void wait(uint64_t *bar, int phaseParity) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@P1                       bra.uni DONE;\n"
               "bra.uni                   LAB_WAIT;\n"
               "DONE:\n"
               "}\n" ::"r"(mbar_ptr),
               "r"(phaseParity));
}

__device__ static __forceinline__ void expect_bytes(uint64_t *bar,
                                                    uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n"
      :
      : "r"(bar_ptr), "r"(bytes)
      : "memory");
}

__device__ static __forceinline__ void expect_bytes_and_arrive(uint64_t *bar,
                                                               uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm("mbarrier.arrive.expect_tx.release.cta.shared.b64 _, [%0], %1;\n "
      :
      : "r"(bar_ptr), "r"(bytes)
      : "memory");
}

__device__ static __forceinline__ void tma_commit_group() {
  asm volatile("cp.async.bulk.commit_group;");
}

template <int N>
__device__ static __forceinline__ void tma_wait_until_pending() {
  asm volatile("cp.async.bulk.wait_group %0;" : : "n"(N) : "memory");
}

__device__ static __forceinline__ void cp_async_bulk_tensor_1d_global_to_shared(
    void *smem_dest, const CUtensorMap *tensor_map, int c0, uint64_t *bar) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes "
      "[%0], [%1, {%2}], [%3];\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(mbar_ptr)
      : "memory");
}

/**
 * @brief Asynchronously copy a 2D tensor tile from global to shared memory.
 *
 * @param smem_dest Destination address in shared memory.
 * @param tensor_map Tensor map descriptor for the source tensor.
 * @param c0 Coordinate in the first dimension.
 * @param c1 Coordinate in the second dimension.
 * @param bar Pointer to mbarrier for completion tracking.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void
cp_async_bulk_tensor_2d_global_to_shared(void *smem_dest,
                                         const CUtensorMap *tensor_map, int c0,
                                         int c1, uint64_t *bar) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(c1), "r"(mbar_ptr)
      : "memory");
}

/**
 * @brief Asynchronously copy a 3D tensor tile from global to shared memory.
 *
 * @param smem_dest Destination address in shared memory.
 * @param tensor_map Tensor map descriptor for the source tensor.
 * @param c0 Coordinate in the first dimension.
 * @param c1 Coordinate in the second dimension.
 * @param c2 Coordinate in the third dimension.
 * @param bar Pointer to mbarrier for completion tracking.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void
cp_async_bulk_tensor_3d_global_to_shared(void *smem_dest,
                                         const CUtensorMap *tensor_map, int c0,
                                         int c1, int c2, uint64_t *bar) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes "
      "[%0], [%1, {%2, %3, %4}], [%5];\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(mbar_ptr)
      : "memory");
}

__device__ static __forceinline__ void cp_async_bulk_tensor_4d_global_to_shared(
    void *smem_dest, const CUtensorMap *tensor_map, int c0, int c1, int c2,
    int c3, uint64_t *bar) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes "
      "[%0], [%1, {%2, %3, %4, %5}], [%6];\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar_ptr)
      : "memory");
}

__device__ static __forceinline__ void cp_async_bulk_tensor_5d_global_to_shared(
    void *smem_dest, const CUtensorMap *tensor_map, int c0, int c1, int c2,
    int c3, int c4, uint64_t *bar) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes "
      "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4),
        "r"(mbar_ptr)
      : "memory");
}

__device__ static __forceinline__ void
cp_async_bulk_tensor_1d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         const void *src) {
  asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group "
               "[%0, {%1}], [%2];\n"
               :
               : "l"(tensor_map), "r"(c0),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}

__device__ static __forceinline__ void
cp_async_bulk_tensor_2d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         int c1, const void *src) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
               "[%0, {%1, %2}], [%3];\n"
               :
               : "l"(tensor_map), "r"(c0), "r"(c1),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}

__device__ static __forceinline__ void
cp_async_bulk_tensor_3d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         int c1, int c2, const void *src) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group "
               "[%0, {%1, %2, %3}], [%4];\n"
               :
               : "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}

__device__ static __forceinline__ void
cp_async_bulk_tensor_4d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         int c1, int c2, int c3,
                                         const void *src) {
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group "
               "[%0, {%1, %2, %3, %4}], [%5];\n"
               :
               : "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}

__device__ static __forceinline__ void
cp_async_bulk_tensor_5d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         int c1, int c2, int c3, int c4,
                                         const void *src) {
  asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group "
               "[%0, {%1, %2, %3, %4, %5}], [%6];\n"
               :
               : "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}


template <int TILE_M, int TILE_N>
__global__ void single_tma_load(__grid_constant__ const CUtensorMap src_map, bf16 *dest) {
    __shared__ alignas(128) bf16 share_mem[TILE_M * TILE_N];
    __shared__ alignas(8) uint64_t barrier;

    constexpr int tileSize = TILE_M * TILE_N * sizeof(bf16);
    init_barrier(&barrier, 1);
    async_proxy_fence();

    expect_bytes(&barrier, tileSize);

    cp_async_bulk_tensor_2d_global_to_shared(share_mem, &src_map, 0, 0, &barrier);
    arrive(&barrier, 1);
    wait(&barrier, 0);

    for (int i = 0; i < TILE_M; ++i) {
        int base = i * TILE_N;
        for (int j = 0; j < TILE_N; ++j) {
            int idx = base + j;
            dest[idx] = share_mem[idx];
        }
    }
}

template <int TILE_M, int TILE_N> void launch_single_tma_load(bf16 *src, bf16 *dest) {
    alignas(128) CUtensorMap tensor_map;
    constexpr cuuint32_t RANK = 2;

    const cuuint64_t global_dim[RANK] = {
        static_cast<cuuint64_t>(TILE_N),
        static_cast<cuuint64_t>(TILE_M)};

    const cuuint64_t global_stride[RANK - 1] = {
        static_cast<cuuint64_t>(TILE_N * sizeof(bf16))};

    const cuuint32_t box_dim[RANK] = {
        static_cast<cuuint32_t>(TILE_N),
        static_cast<cuuint32_t>(TILE_M),
    };

    const cuuint32_t element_stride[RANK] = {1, 1};

    CUDA_CHECK(cuInit(0));
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        RANK,
        src,
        global_dim,
        global_stride,
        box_dim,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    single_tma_load<TILE_M, TILE_N><<<1, 1>>>(tensor_map, dest);
}

```

### TMA store

TMA Store：即“shared memory 写回 global memory”。看起来只是方向跟TMS load反了，但同步机制其实完全不同。

TMA load 用的是 `mbarrier`，因为 barrier 和目标 shared memory 在同一侧，硬件容易在数据到达 shared memory 时顺便更新 barrier 状态。但 TMA store 是 shared 到 global，写完成的信息如果还要再反向通知 shared memory 中的 barrier，就会涉及一次不自然的“往返确认”。因此硬件没有沿用 `mbarrier` 方案，而是采用 commit group 机制来追踪 store 是否完成。

示例代码：

```cpp
typedef __nv_bfloat16 bf16;

template <int TILE_M, int TILE_N>
__global__ void single_tma_store(__grid_constant__ const CUtensorMap src_map,
                                 __grid_constant__ const CUtensorMap dest_map) {
    __shared__ alignas(128) bf16 share_mem[TILE_M * TILE_N];
    __shared__ alignas(8) uint64_t barrier;

    constexpr int tile_size = TILE_M * TILE_N * sizeof(bf16);
    init_barrier(&barrier, 1);
    async_proxy_fence();

    expect_bytes(&barrier, tile_size);
    cp_async_bulk_tensor_2d_global_to_shared(share_mem, &src_map, 0, 0,
                                             &barrier);
    arrive(&barrier, 1);
    wait(&barrier, 0);

    cp_async_bulk_tensor_2d_shared_to_global(&dest_map, 0, 0, share_mem);
    tma_commit_group();
    tma_wait_until_pending<0>();
}

template <int TILE_M, int TILE_N>
void launch_single_tma_store(bf16 *src, bf16 *dest) {
    alignas(128) CUtensorMap src_map;
    alignas(128) CUtensorMap dest_map;
    constexpr cuuint32_t RANK = 2;

    const cuuint64_t global_dim[RANK] = {
        static_cast<cuuint64_t>(TILE_N),
        static_cast<cuuint64_t>(TILE_M)};

    const cuuint64_t global_stride[RANK - 1] = {
        static_cast<cuuint64_t>(TILE_N * sizeof(bf16))};

    const cuuint32_t box_dim[RANK] = {
        static_cast<cuuint32_t>(TILE_N),
        static_cast<cuuint32_t>(TILE_M),
    };

    const cuuint32_t element_stride[RANK] = {1, 1};

    CUDA_CHECK(cuInit(0));
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &src_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        RANK,
        src,
        global_dim,
        global_stride,
        box_dim,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &dest_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        RANK,
        dest,
        global_dim,
        global_stride,
        box_dim,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    single_tma_store<TILE_M, TILE_N><<<1, 1>>>(src_map, dest_map);
}

```

### TMA reduce，把“搬运”和“归约”合并

TMA 还有更强的能力：边搬运边做 reduce。也就是把 shared memory 中的数据写回 global memory 的同时，与 global memory 里原有数据做逐元素归约，例如 add、min 之类。

`add` 版本的 TMA reduce 接口示例代码如下：

```cpp
__device__ static __forceinline__ void
cp_async_reduce_add_bulk_tensor_2d_shared_to_global(
    const CUtensorMap *tensor_map, int c0, int c1, const void *src) {
    asm volatile(
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group "
        "[%0, {%1, %2}], [%3];\n"
        :
        : "l"(tensor_map), "r"(c0), "r"(c1),
          "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
        : "memory");
}

template <int TILE_M, int TILE_N>
__global__ void
single_tma_reduce(__grid_constant__ const CUtensorMap src_map,
                  __grid_constant__ const CUtensorMap dest_map) {
    __shared__ alignas(128) bf16 share_mem[TILE_M * TILE_N];
    __shared__ alignas(8) uint64_t barrier;

    constexpr int tile_size = TILE_M * TILE_N * sizeof(bf16);
    init_barrier(&barrier, 1);
    async_proxy_fence();

    expect_bytes(&barrier, tile_size);
    cp_async_bulk_tensor_2d_global_to_shared(share_mem, &src_map, 0, 0,
                                             &barrier);
    arrive(&barrier, 1);
    wait(&barrier, 0);

    cp_async_reduce_add_bulk_tensor_2d_shared_to_global(&dest_map, 0, 0,
                                                        share_mem);
    tma_commit_group();
    tma_wait_until_pending<0>();
}

template <int TILE_M, int TILE_N>
void launch_single_tma_reduce(bf16 *src, bf16 *dest) {
    alignas(128) CUtensorMap src_map;
    alignas(128) CUtensorMap dest_map;
    constexpr cuuint32_t RANK = 2;

    const cuuint64_t global_dim[RANK] = {
        static_cast<cuuint64_t>(TILE_N),
        static_cast<cuuint64_t>(TILE_M)};

    const cuuint64_t global_stride[RANK - 1] = {
        static_cast<cuuint64_t>(TILE_N * sizeof(bf16))};

    const cuuint32_t box_dim[RANK] = {
        static_cast<cuuint32_t>(TILE_N),
        static_cast<cuuint32_t>(TILE_M),
    };

    const cuuint32_t element_stride[RANK] = {1, 1};

    CUDA_CHECK(cuInit(0));
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &src_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        RANK,
        src,
        global_dim,
        global_stride,
        box_dim,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &dest_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        RANK,
        dest,
        global_dim,
        global_stride,
        box_dim,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    single_tma_reduce<TILE_M, TILE_N><<<1, 1>>>(src_map, dest_map);
}

```

## TMA 中使用 swizzling pattern

我们知道 shared memory 的 bank conflict 会严重影响性能。如果一个 warp 中多个 lane 访问的地址落在同一个 bank，就会发生串行化。过去通常靠 padding、转置或手工重排来规避。但在 Hopper 上，TMA 可以在写入 shared memory 时直接按某种 swizzle pattern 排列数据，从而降低或避免 bank conflict。

具体来说：TMA 把一个 tile 从 global memory 搬到 shared memory 时，不按线性地址写入 shared memory，而是按 tensor map descriptor 里的 `CUtensorMapSwizzle` 做重排，目的是让后续 warp 对 shared memory 的访问分散到不同 bank。NVIDIA 的文档把 TMA 描述为 Hopper 引入的 global/shared bulk copy 机制；driver API 里 `cuTensorMapEncodeTiled` / `cuTensorMapEncodeIm2col*` 都显式带 `CUtensorMapSwizzle swizzle` 参数。

---

最后一次更新时间：`2026-08-12 20:23:36 CST`

原文链接：[https://fangpin.github.io/gpu-hpc-book/#/chapters/07-cuda-tma-tensor-memory-accelerator.md](https://fangpin.github.io/gpu-hpc-book/#/chapters/07-cuda-tma-tensor-memory-accelerator.md)
