# GPU 高性能计算

[English](README.md)

在线阅读：[https://fangpin.github.io/gpu-hpc-book/](https://fangpin.github.io/gpu-hpc-book/)

GitHub：[fangpin/gpu-hpc-book](https://github.com/fangpin/gpu-hpc-book)

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

## 项目结构

- `docs/source/` — Sphinx 源码树（`conf.py`、`index.md`、`chapters/`、`assets/`）。
- `docs/source/chapters/` — 章节 Markdown（MyST），从飞书源文档生成。
- `docs/source/assets/images/` — 从飞书文档下载的图片。
- `doc.mk` — 安装 / 同步 / 构建 / 本地预览的 Make 目标。
- `doc_scripts/sync_lark_doc.py` — 通过 `lark-cli` 拉取飞书文档并重新生成源码树。
- `requirements-docs.txt` — Python 工具链（Sphinx、sphinx-rtd-theme、myst-parser）。
- `.github/workflows/docs.yml` — push 到 `main`/`master` 时构建并发布到 GitHub Pages。

## 本地构建

文档工具链需要 Python ≥ 3.11（Sphinx 8.x 的要求）。macOS 自带的 `python3` 通常是 3.9，所以 `doc.mk` 会自动挑选第一个可用的 `python3.13` / `python3.12` / `python3.11`；需要时可用 `make PYTHON=/path/to/python3.12 ...` 指定。

```bash
# 1. 安装 Python 工具链到 .venv（缺失时也会装 lark-cli）
make -f doc.mk docs-install

# 2. 从飞书源文档同步章节和图片
#    默认使用 docs/project.json 里记录的 URL，也可传 --doc：
make -f doc.mk docs-sync
make -f doc.mk docs-sync DOC="<feishu-doc-url>"

# 3. 构建 HTML 站点
make -f doc.mk docs-html

# 4. 本地预览，打开 http://127.0.0.1:8000/
make -f doc.mk docs-serve
```

`docs-sync` 需要 `lark-cli` 已登录（`lark-cli auth login`），才能下载文档正文以及 `feishu.cn/file/<token>` 托管的图片 —— 匿名请求只会拿到一整页 SSO 登录页。

## 站点

- 站点框架：Sphinx + sphinx-rtd-theme + MyST
- 发布方式：GitHub Actions 构建 `docs/source/` 并把 HTML 发布到 GitHub Pages

原文链接：[https://fangpin.github.io/gpu-hpc-book/](https://fangpin.github.io/gpu-hpc-book/)
