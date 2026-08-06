# GPU 高性能计算

[中文](README.zh-CN.md)

Read online: [https://fangpin.github.io/gpu-hpc-book/](https://fangpin.github.io/gpu-hpc-book/)

A hands-on, book-length guide to high-performance computing on modern accelerators. The chapters move from the first principles of hardware acceleration to practical GPU programming, performance modeling, memory optimization, advanced CUDA features, TPU communication, and TIRx.

The goal is to make performance feel explainable. Instead of treating CUDA kernels, Tensor Cores, TMA, WGMMA, and TPU collectives as separate tricks, the book connects them through one question: where does the data live, how does it move, and what work can the hardware overlap while it moves?

## What You Will Learn

- How hardware acceleration changes the way we think about instruction throughput, parallelism, and data reuse.
- How to use the Roofline model to decide whether an optimization is likely to help.
- How the GPU programming model maps work onto threads, warps, blocks, SMs, and memory hierarchy.
- Why memory coalescing, bank conflicts, shared memory tiling, and register tiling dominate many real kernels.
- How GEMM exposes the essential ideas behind Tensor Cores, occupancy control, overlapping compute and memory movement, TMA, and WGMMA.
- How TPU collectives such as reduce-scatter, all-gather, and collective matmul relate to the same performance principles.
- How TIRx represents scope, layout, dispatch, and hardware-aware tensor programs.

## Read the Book

- GitHub Pages content lives in `docs/`.
- Chapter Markdown files live in `docs/chapters/`.
- Read the book locally with `npm run serve`, then open <http://127.0.0.1:4193/>.
- Regenerate the generated files with `LARK_DOC_URL=<url> npm run sync` or `npm run sync -- --doc <url>` after private content changes.

## Site

- Generated site shell: Docsify + GitHub Pages
- Deployment: GitHub Actions publishes the `docs/` directory

Last updated: `2026-08-05 16:12:21 CST`

原文链接：[https://fangpin.github.io/gpu-hpc-book/](https://fangpin.github.io/gpu-hpc-book/)
