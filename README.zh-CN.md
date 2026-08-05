# GPU 高性能计算

[English](README.md)

在线阅读：[https://fangpin.github.io/gpu-hpc-book/](https://fangpin.github.io/gpu-hpc-book/)

这是一本面向现代加速器的高性能计算实践指南。内容从硬件加速的基本思路讲起，逐步进入 GPU 编程模型、性能建模、内存优化、高级 CUDA 特性、TPU 通信原语以及 TIRx。

这本书想解决的不是“记住更多 CUDA 技巧”，而是建立一条可解释的性能优化主线：数据在哪里、如何移动、硬件能在数据移动时并行做什么、程序结构如何把这些能力表达出来。

## 你会读到什么

- 硬件加速如何改变我们对指令吞吐、并行度和数据复用的理解。
- 如何用 Roofline Model 判断一个优化是否值得做。
- GPU 编程模型如何把计算映射到线程、warp、block、SM 和内存层次。
- 为什么内存合并访问、bank conflict、shared memory tiling 和 register tiling 会主导很多真实 kernel 的性能。
- GEMM 如何串起 Tensor Cores、occupancy 控制、计算与搬运重叠、TMA 和 WGMMA 等高级机制。
- TPU 的 reduce-scatter、all-gather、collective matmul 如何和数据搬运、重叠执行这些核心原则相连。
- TIRx 如何表达 scope、layout、dispatch 和面向硬件的张量程序。

## 本地阅读

- GitHub Pages 内容位于 `docs/`。
- 章节 Markdown 文件位于 `docs/chapters/`。
- 运行 `npm run serve` 后打开 <http://127.0.0.1:4193/>。
- 私有内容更新后运行 `LARK_DOC_URL=<url> npm run sync` 或 `npm run sync -- --doc <url>` 重新生成章节。

## 站点

- 站点框架：Docsify + GitHub Pages
- 发布方式：GitHub Actions 发布 `docs/` 目录

最后一次更新时间：`2026-08-05 16:12:21 CST`
