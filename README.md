# GPU 高性能计算

[中文](README.zh-CN.md)

Read online: [https://fangpin.github.io/gpu-hpc-book/](https://fangpin.github.io/gpu-hpc-book/)

GitHub: [fangpin/gpu-hpc-book](https://github.com/fangpin/gpu-hpc-book)

A hands-on, book-length guide to high-performance computing on modern accelerators. The chapters move from the first principles of hardware acceleration to practical GPU programming, performance modeling, memory optimization, advanced CUDA features, TPU communication, and TIRx.

The goal is to make performance feel explainable. Instead of treating CUDA kernels, Tensor Cores, TMA, WGMMA, and TPU collectives as separate tricks, the book connects them through one question: where does the data live, how does it move, and what work can the hardware overlap while it moves?

## What You Will Learn

- How hardware acceleration changes the way we think about instruction throughput, parallelism, and data reuse.
- How to use the Roofline model to decide whether an optimization is likely to help.
- How the GPU programming model maps work onto threads, warps, blocks, SMs, and the memory hierarchy.
- Why memory coalescing, bank conflicts, shared memory tiling, and register tiling dominate many real kernels.
- How GEMM exposes the essential ideas behind Tensor Cores, occupancy control, overlapping compute and memory movement, TMA, and WGMMA.
- How TPU collectives such as reduce-scatter, all-gather, and collective matmul relate to the same performance principles.
- How TIRx represents scope, layout, dispatch, and hardware-aware tensor programs.

## Project Layout

- `docs/source/` — Sphinx source tree (`conf.py`, `index.md`, `chapters/`, `assets/`).
- `docs/source/chapters/` — chapter Markdown (MyST), generated from the Feishu source doc.
- `docs/source/assets/images/` — images downloaded from the Feishu doc.
- `doc.mk` — Make targets for install / sync / build / serve.
- `doc_scripts/sync_lark_doc.py` — fetches the Feishu doc via `lark-cli` and regenerates the source tree.
- `requirements-docs.txt` — Python toolchain (Sphinx, sphinx-rtd-theme, myst-parser).
- `.github/workflows/docs.yml` — builds and deploys the site to GitHub Pages on push to `main`/`master`.

## Build the Book Locally

The docs toolchain needs Python ≥ 3.11 (Sphinx 8.x requirement). On macOS the system `python3` is usually 3.9, so `doc.mk` auto-selects the first `python3.13` / `python3.12` / `python3.11` it finds; override with `make PYTHON=/path/to/python3.12 ...` if needed.

```bash
# 1. Install the Python toolchain into .venv (and lark-cli if missing)
make -f doc.mk docs-install

# 2. Sync chapters + images from the Feishu source doc
#    Uses the URL recorded in docs/project.json, or pass --doc:
make -f doc.mk docs-sync
make -f doc.mk docs-sync DOC="<feishu-doc-url>"

# 3. Build the HTML site
make -f doc.mk docs-html

# 4. Serve it locally, then open http://127.0.0.1:8000/
make -f doc.mk docs-serve
```

`docs-sync` requires `lark-cli` to be authenticated (`lark-cli auth login`) so it can download the doc text and the images hosted at `feishu.cn/file/<token>` — those return an SSO login page to anonymous requests.

## Site

- Generated site: Sphinx + sphinx-rtd-theme + MyST
- Deployment: GitHub Actions builds `docs/source/` and publishes the HTML to GitHub Pages

原文链接：[https://fangpin.github.io/gpu-hpc-book/](https://fangpin.github.io/gpu-hpc-book/)
