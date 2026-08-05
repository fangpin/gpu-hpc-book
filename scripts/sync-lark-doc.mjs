import { spawnSync } from 'node:child_process';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const DEFAULT_DOCS_DIR = 'docs';
const DEFAULT_SITE_URL = 'https://fangpin.github.io/gpu-coding/';

const TITLE_SLUGS = new Map([
  ['高性能计算', 'high-performance-computing'],
  ['计算的硬件加速思路', 'computing-hardware-acceleration-ideas'],
  ['并行策略', 'parallel-strategies'],
  ['GPU 编程模型', 'gpu-programming-model'],
  ['Roofline Model —— 性能优化方向的理论指导', 'roofline-model'],
  ['内存访问密集型高性能计算', 'memory-intensive-high-performance-computing'],
  ['高级 cuda 特性 —— 从矩阵乘法(gemm)说起', 'advanced-cuda-features-gemm'],
  ['GPU 上动态的调度 —— 从run-length-compression说起', 'dynamic-scheduling-run-length-compression'],
  ['Cuda TMA （Tensor Memory Accelerator）', 'cuda-tma-tensor-memory-accelerator'],
  ['wgmma：warp-group level tensor core instructions', 'wgmma-warp-group-level-tensor-core-instructions'],
  ['TPU 与分布式通信原语', 'tpu-and-distributed-communication-primitives'],
  ['TIRx 基础', 'tirx-basics'],
  ['ex.mod(...) takes torch tensors directly, the same call form used in every chapter.', 'ex-mod-takes-torch-tensors-directly'],
  ['TIRx 与高性能GEMM', 'tirx-and-high-performance-gemm'],
  ['TIRx 视角：scope、layout、dispatch 是可读性的核心', 'tirx-scope-layout-dispatch'],
  ['为什么这还不是高性能 GEMM', 'why-this-is-not-yet-high-performance-gemm'],
  ['给 MLSys 工程的几个启发', 'mlsys-engineering-takeaways'],
  ['参考', 'references'],
]);

const IMAGE_EXTENSIONS = new Map([
  ['image/apng', '.apng'],
  ['image/avif', '.avif'],
  ['image/gif', '.gif'],
  ['image/jpeg', '.jpg'],
  ['image/jpg', '.jpg'],
  ['image/png', '.png'],
  ['image/svg+xml', '.svg'],
  ['image/webp', '.webp'],
]);

const CODE_FENCE_LANGUAGES = new Map([
  ['c++', 'cpp'],
  ['cpp', 'cpp'],
  ['py', 'python'],
  ['python', 'python'],
  ['bash', 'bash'],
  ['shell', 'bash'],
  ['sh', 'bash'],
  ['java', 'java'],
  ['yaml', 'yaml'],
  ['yml', 'yaml'],
  ['text', 'text'],
  ['txt', 'text'],
]);

const PRISM_LANGUAGES = [
  'python',
  'c',
  'cpp',
  'bash',
  'java',
  'yaml',
];

const DOCSIFY_SLUG_PUNCTUATION = /[\u2000-\u206F\u2E00-\u2E7F\\'!"#$%&()*+,./:;<=>?@[\]^`{|}~]/g;

export function cleanTitle(title) {
  return title
    .replace(/<[^>]+>/g, '')
    .replace(/!\[[^\]]*\]\([^)]+\)/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[`*_~]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

export function chapterSlug(title, index, seen = new Set()) {
  const cleaned = cleanTitle(title);
  const mapped = TITLE_SLUGS.get(cleaned);
  const base = mapped ?? cleaned
    .normalize('NFKD')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  const prefix = String(index).padStart(2, '0');
  let candidate = `${prefix}-${base || 'chapter'}`;
  let suffix = 2;

  while (seen.has(candidate)) {
    candidate = `${prefix}-${base || 'chapter'}-${suffix}`;
    suffix += 1;
  }

  seen.add(candidate);
  return candidate;
}

export function splitIntoChapters(markdown) {
  const titleMatch = markdown.match(/^<title>([\s\S]*?)<\/title>\s*/);
  const documentTitle = titleMatch ? cleanTitle(titleMatch[1]) : 'Introduction';
  const body = titleMatch ? markdown.slice(titleMatch[0].length) : markdown;
  const lines = body.split(/\r?\n/);
  const starts = [];
  let inFence = false;

  for (let index = 0; index < lines.length; index += 1) {
    if (/^\s*(```|~~~)/.test(lines[index])) {
      inFence = !inFence;
    }

    if (!inFence && /^#\s+\S/.test(lines[index])) {
      starts.push(index);
    }
  }

  const seen = new Set();
  const chapters = [];
  const firstHeading = starts[0] ?? lines.length;
  const preface = lines.slice(0, firstHeading).join('\n').trim();

  if (preface) {
    chapters.push({
      index: 0,
      title: documentTitle || 'Introduction',
      slug: '00-introduction',
      content: [`# ${documentTitle || 'Introduction'}`, preface].filter(Boolean).join('\n\n').trimEnd() + '\n',
    });
  }

  for (let i = 0; i < starts.length; i += 1) {
    const start = starts[i];
    const end = starts[i + 1] ?? lines.length;
    const rawTitle = lines[start].replace(/^#\s+/, '');
    const title = cleanTitle(rawTitle);
    const index = chapters.length;

    chapters.push({
      index,
      title,
      slug: chapterSlug(title, index, seen),
      content: lines.slice(start, end).join('\n').trimEnd() + '\n',
    });
  }

  return chapters;
}

export function extractImages(markdown) {
  const images = [];
  const imagePattern = /!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)/g;
  let match;

  while ((match = imagePattern.exec(markdown)) !== null) {
    const [, alt, url] = match;

    if (/^https?:\/\//.test(url)) {
      images.push({ alt, url });
    }
  }

  return images;
}

export function rewriteImageUrls(markdown, urlToPath) {
  return markdown.replace(/!\[([^\]]*)\]\(([^)\s]+)(\s+\"[^\"]*\")?\)/g, (full, alt, url, title = '') => {
    const replacement = urlToPath.get(url);
    return replacement ? `![${alt}](${replacement}${title})` : full;
  });
}

function cleanMarkdownLine(line) {
  return line
    .replace(/\\([*_])/g, '$1')
    .replace(/\\\$/g, '$')
    .replace(/\\([[\]()])/g, '$1')
    .replace(/(^|:\s*)([0-9]+(?:\.[0-9]+)?\s+\\text\s+([^}$]+)})\$/g, '$1$$$2$$')
    .replace(/\\text\s+([^}$]+)}/g, '\\text{ $1}');
}

function normalizeCodeFenceLine(line) {
  return line.replace(/^(\s*)(`{3,}|~{3,})(\S+)(.*)$/, (full, indent, fence, language, suffix) => {
    const normalized = CODE_FENCE_LANGUAGES.get(language.toLowerCase());
    return normalized ? `${indent}${fence}${normalized}${suffix}` : full;
  });
}

export function cleanMarkdownForRendering(markdown) {
  const lines = markdown.split(/\r?\n/);
  let inFence = false;

  return lines.map((line) => {
    const openingFence = !inFence && /^\s*(`{3,}|~{3,})/.test(line);
    const closingFence = inFence && /^\s*(`{3,}|~{3,})\s*$/.test(line);

    if (openingFence) {
      inFence = true;
      return normalizeCodeFenceLine(line);
    }

    if (closingFence) {
      inFence = false;
      return line;
    }

    return inFence ? line : cleanMarkdownLine(line);
  }).join('\n');
}

export function docsifyHeadingId(title, seen = new Map()) {
  const base = cleanTitle(title)
    .trim()
    .replace(/[A-Z]+/g, (match) => match.toLowerCase())
    .replace(/<[^>]+>/g, '')
    .replace(DOCSIFY_SLUG_PUNCTUATION, '')
    .replace(/\s/g, '-')
    .replace(/-+/g, '-')
    .replace(/^(\d)/, '_$1');
  const count = seen.has(base) ? seen.get(base) + 1 : 0;
  seen.set(base, count);

  return count ? `${base}-${count}` : base;
}

export function extractSidebarHeadings(markdown, maxDepth = 3) {
  const headings = [];
  const seen = new Map();
  const lines = markdown.split(/\r?\n/);
  let inFence = false;

  for (const line of lines) {
    if (!inFence && /^\s*(`{3,}|~{3,})/.test(line)) {
      inFence = true;
      continue;
    }

    if (inFence) {
      if (/^\s*(`{3,}|~{3,})\s*$/.test(line)) {
        inFence = false;
      }

      continue;
    }

    const match = line.match(/^(#{2,})(?:\s+)(.+?)\s*#*\s*$/);

    if (!match) {
      continue;
    }

    const level = match[1].length;

    if (level > maxDepth) {
      continue;
    }

    const title = cleanTitle(match[2]);

    if (!title) {
      continue;
    }

    headings.push({
      level,
      title,
      id: docsifyHeadingId(match[2], seen),
    });
  }

  return headings;
}

function sidebarLinkPath(fileName) {
  return `/${fileName.replace(/^\/+/, '')}`;
}

function buildChapterList(chapters, rootRelative = false) {
  const lines = [];

  for (const chapter of chapters) {
    const chapterLink = rootRelative ? chapter.fileName : sidebarLinkPath(chapter.fileName);
    lines.push(`- [${chapter.title}](${chapterLink})`);

    for (const heading of extractSidebarHeadings(chapter.content ?? '')) {
      const indent = '  '.repeat(heading.level - 1);
      lines.push(`${indent}- [${heading.title}](${chapterLink}?id=${heading.id})`);
    }
  }

  return lines.join('\n');
}

export function buildSidebar(chapters) {
  const lines = ['- [首页](/)', buildChapterList(chapters)];

  lines.push('');
  return lines.join('\n');
}

function parseJsonOutput(output) {
  const start = output.indexOf('{');
  const end = output.lastIndexOf('}');

  if (start === -1 || end === -1 || end <= start) {
    throw new Error(`Expected JSON output, got: ${output.slice(0, 200)}`);
  }

  return JSON.parse(output.slice(start, end + 1));
}

async function fetchLarkMarkdown(docUrl) {
  if (!docUrl) {
    throw new Error('Missing source document URL. Set LARK_DOC_URL=<url> or pass --doc <url>.');
  }

  const result = spawnSync('lark-cli', [
    'docs',
    '+fetch',
    '--doc',
    docUrl,
    '--doc-format',
    'markdown',
    '--format',
    'json',
  ], {
    encoding: 'utf8',
    env: {
      ...process.env,
      LARKSUITE_CLI_NO_SKILLS_NOTIFIER: '1',
      LARKSUITE_CLI_NO_UPDATE_NOTIFIER: '1',
    },
    maxBuffer: 64 * 1024 * 1024,
  });

  if (result.status !== 0) {
    throw new Error(`lark-cli failed with status ${result.status}:\n${result.stderr || result.stdout}`);
  }

  const json = parseJsonOutput(result.stdout);

  if (!json.ok) {
    throw new Error(`lark-cli returned an error:\n${JSON.stringify(json, null, 2)}`);
  }

  return {
    markdown: json.data.document.content,
    documentId: json.data.document.document_id,
    revisionId: json.data.document.revision_id,
  };
}

export function extensionFromBytes(bytes) {
  if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return '.jpg';
  }

  if (bytes.length >= 8
    && bytes[0] === 0x89
    && bytes[1] === 0x50
    && bytes[2] === 0x4e
    && bytes[3] === 0x47
    && bytes[4] === 0x0d
    && bytes[5] === 0x0a
    && bytes[6] === 0x1a
    && bytes[7] === 0x0a) {
    return '.png';
  }

  if (bytes.length >= 6 && bytes.slice(0, 6).toString('ascii').startsWith('GIF')) {
    return '.gif';
  }

  if (bytes.length >= 12
    && bytes.slice(0, 4).toString('ascii') === 'RIFF'
    && bytes.slice(8, 12).toString('ascii') === 'WEBP') {
    return '.webp';
  }

  return undefined;
}

export function extensionFromResponse(url, response, bytes) {
  const bytesExtension = extensionFromBytes(bytes);

  if (bytesExtension) {
    return bytesExtension;
  }

  const contentType = response.headers.get('content-type')?.split(';')[0]?.trim().toLowerCase();

  if (contentType && IMAGE_EXTENSIONS.has(contentType)) {
    return IMAGE_EXTENSIONS.get(contentType);
  }

  if (contentType && !contentType.startsWith('image/')) {
    throw new Error(`Expected image response, got ${contentType}`);
  }

  try {
    const ext = path.extname(new URL(url).pathname);
    return ext || '.bin';
  } catch {
    return '.bin';
  }
}

async function downloadChapterImages(chapter, imagesRoot) {
  const images = extractImages(chapter.content);
  const uniqueUrls = [...new Set(images.map((image) => image.url))];
  const replacements = new Map();

  if (uniqueUrls.length === 0) {
    return replacements;
  }

  const chapterImageDir = path.join(imagesRoot, chapter.slug);
  await mkdir(chapterImageDir, { recursive: true });

  for (let index = 0; index < uniqueUrls.length; index += 1) {
    const url = uniqueUrls[index];

    try {
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const bytes = Buffer.from(await response.arrayBuffer());
      const extension = extensionFromResponse(url, response, bytes);
      const fileName = `image-${String(index + 1).padStart(2, '0')}${extension}`;
      const filePath = path.join(chapterImageDir, fileName);
      await writeFile(filePath, bytes);
      replacements.set(url, `../assets/images/${chapter.slug}/${fileName}`);
    } catch (error) {
      console.warn(`Warning: could not download image for ${chapter.slug}: ${url} (${error.message})`);
    }
  }

  return replacements;
}

export function buildDocsReadme({
  title,
  syncedAt,
  lastUpdated,
  siteUrl = DEFAULT_SITE_URL,
  chapters,
  language = 'zh',
}) {
  const displayUpdatedAt = lastUpdated ?? syncedAt;
  const chapterLinks = buildChapterList(chapters, true);
  const isChinese = language === 'zh';

  if (isChinese) {
    return [
      `# ${title}`,
      '',
      '[English](README.en.md)',
      '',
      `在线阅读：[${siteUrl}](${siteUrl})`,
      '',
      '从硬件加速、并行编程到 GPU/TPU 性能优化，系统梳理高性能计算的核心概念、分析方法与工程实践。',
      '',
      '这本书面向希望真正理解高性能计算的工程师和学习者：不仅知道 CUDA、Tensor Core、TMA、WGMMA、TPU collective 这些名词，更能把硬件能力、程序结构、数据搬运和性能模型连成一条清晰的优化路径。',
      '',
      '## 阅读方式',
      '',
      '- 打开左侧目录按章节阅读。',
      '- 从 Roofline、内存层次和并行策略开始，再进入 GEMM、TMA、WGMMA、TPU 通信和 TIRx。',
      '- 每个章节的 Markdown 源文件保存在 `docs/chapters/`。',
      '- 内容更新后，运行 `npm run sync` 重新生成章节。',
      '',
      '## 章节',
      '',
      chapterLinks,
      '',
      `最后一次更新时间：\`${displayUpdatedAt}\``,
      '',
    ].join('\n');
  }

  return [
    `# ${title}`,
    '',
    '[中文](README.md)',
    '',
    `Read online: [${siteUrl}](${siteUrl})`,
    '',
    'This book is a practical tour of high-performance computing, from hardware acceleration and parallel programming to GPU/TPU performance engineering.',
    '',
    'It is written for readers who want to understand how modern accelerators really deliver speed: how work is mapped to threads and warps, how memory movement shapes performance, how GEMM becomes a playground for advanced CUDA features, and how newer ideas such as TMA, WGMMA, TPU collectives, and TIRx fit into one optimization story.',
    '',
    '## Why Read This Book',
    '',
    '- Build a hardware-first mental model for performance instead of memorizing isolated CUDA tricks.',
    '- Use Roofline analysis to reason about compute bounds, memory bounds, and when an optimization should matter.',
    '- Follow the path from basic parallel strategies to memory coalescing, shared memory tiling, register tiling, Tensor Cores, TMA, and WGMMA.',
    '- See how distributed communication on TPU connects back to the same ideas of data movement and overlap.',
    '- Read TIRx as a modern way to express layouts, scopes, dispatch, and hardware-aware tensor programs.',
    '',
    '## How To Read',
    '',
    '- Start with the sidebar, or use the chapter outline below.',
    '- Read the early chapters for the performance model, then jump into GEMM and accelerator-specific chapters when you want concrete mechanisms.',
    '- The Markdown source for every chapter lives in `docs/chapters/`.',
    '',
    '## Chapters',
    '',
    chapterLinks,
    '',
    `Last updated: \`${displayUpdatedAt}\``,
    '',
  ].join('\n');
}

export function buildChapterPage(content, lastUpdated) {
  return [
    content.trimEnd(),
    '',
    '---',
    '',
    `最后一次更新时间：\`${lastUpdated}\``,
    '',
  ].join('\n');
}

export function formatLastUpdated(date = new Date()) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).formatToParts(date);
  const byType = Object.fromEntries(parts.map((part) => [part.type, part.value]));

  return `${byType.year}-${byType.month}-${byType.day} ${byType.hour}:${byType.minute}:${byType.second} CST`;
}

export function buildSourceMetadata({ title, syncedAt, lastUpdated, chapters = [] }) {
  return `${JSON.stringify({
    title,
    syncedAt,
    lastUpdated,
    chapterCount: chapters.length,
    chapters,
  }, null, 2)}\n`;
}

export function buildIndexHtml(title) {
  return `<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>${title}</title>
    <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/docsify@4/lib/themes/vue.css">
    <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <link rel="stylesheet" href="assets/site.css">
  </head>
  <body>
    <div id="app">加载中...</div>
    <script>
      window.$docsify = {
        name: '${title}',
        repo: '',
        loadSidebar: true,
        subMaxLevel: 3,
        auto2top: true,
        relativePath: true,
        search: {
          noData: '没有结果',
          paths: 'auto',
          placeholder: '搜索文档'
        }
      };
    </script>
    <script src="//cdn.jsdelivr.net/npm/docsify@4"></script>
    <script src="//cdn.jsdelivr.net/npm/docsify@4/lib/plugins/search.min.js"></script>
${PRISM_LANGUAGES.map((language) => `    <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-${language}.min.js"></script>`).join('\n')}
    <script src="//cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script src="//cdn.jsdelivr.net/npm/docsify-latex@0/dist/docsify-latex.min.js"></script>
  </body>
</html>
`;
}

function buildSiteCss() {
  return `:root {
  --theme-color: #2563eb;
}

body {
  letter-spacing: 0;
}

.markdown-section {
  max-width: 980px;
}

.markdown-section pre,
.markdown-section code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}

.markdown-section img {
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  max-height: 680px;
  object-fit: contain;
}

.sidebar {
  width: 320px;
}

.content {
  left: 320px;
}

@media (max-width: 768px) {
  .content {
    left: 0;
  }
}
`;
}

export function buildRootReadmes({
  title,
  siteUrl = DEFAULT_SITE_URL,
  syncedAt,
  lastUpdated,
}) {
  const displayUpdatedAt = lastUpdated ?? syncedAt;
  const english = [
    `# ${title}`,
    '',
    '[中文](README.zh-CN.md)',
    '',
    `Read online: [${siteUrl}](${siteUrl})`,
    '',
    'A hands-on, book-length guide to high-performance computing on modern accelerators. The chapters move from the first principles of hardware acceleration to practical GPU programming, performance modeling, memory optimization, advanced CUDA features, TPU communication, and TIRx.',
    '',
    'The goal is to make performance feel explainable. Instead of treating CUDA kernels, Tensor Cores, TMA, WGMMA, and TPU collectives as separate tricks, the book connects them through one question: where does the data live, how does it move, and what work can the hardware overlap while it moves?',
    '',
    '## What You Will Learn',
    '',
    '- How hardware acceleration changes the way we think about instruction throughput, parallelism, and data reuse.',
    '- How to use the Roofline model to decide whether an optimization is likely to help.',
    '- How the GPU programming model maps work onto threads, warps, blocks, SMs, and memory hierarchy.',
    '- Why memory coalescing, bank conflicts, shared memory tiling, and register tiling dominate many real kernels.',
    '- How GEMM exposes the essential ideas behind Tensor Cores, occupancy control, overlapping compute and memory movement, TMA, and WGMMA.',
    '- How TPU collectives such as reduce-scatter, all-gather, and collective matmul relate to the same performance principles.',
    '- How TIRx represents scope, layout, dispatch, and hardware-aware tensor programs.',
    '',
    '## Read the Book',
    '',
    '- GitHub Pages content lives in `docs/`.',
    '- Chapter Markdown files live in `docs/chapters/`.',
    '- Read the book locally with `npm run serve`, then open <http://127.0.0.1:4193/>.',
    '- Regenerate the generated files with `LARK_DOC_URL=<url> npm run sync` after private content changes.',
    '',
    '## Site',
    '',
    '- Generated site shell: Docsify + GitHub Pages',
    '- Deployment: GitHub Actions publishes the `docs/` directory',
    ...(displayUpdatedAt ? ['', `Last updated: \`${displayUpdatedAt}\``] : []),
    '',
  ].join('\n');

  const chinese = [
    `# ${title}`,
    '',
    '[English](README.md)',
    '',
    `在线阅读：[${siteUrl}](${siteUrl})`,
    '',
    '这是一本面向现代加速器的高性能计算实践指南。内容从硬件加速的基本思路讲起，逐步进入 GPU 编程模型、性能建模、内存优化、高级 CUDA 特性、TPU 通信原语以及 TIRx。',
    '',
    '这本书想解决的不是“记住更多 CUDA 技巧”，而是建立一条可解释的性能优化主线：数据在哪里、如何移动、硬件能在数据移动时并行做什么、程序结构如何把这些能力表达出来。',
    '',
    '## 你会读到什么',
    '',
    '- 硬件加速如何改变我们对指令吞吐、并行度和数据复用的理解。',
    '- 如何用 Roofline Model 判断一个优化是否值得做。',
    '- GPU 编程模型如何把计算映射到线程、warp、block、SM 和内存层次。',
    '- 为什么内存合并访问、bank conflict、shared memory tiling 和 register tiling 会主导很多真实 kernel 的性能。',
    '- GEMM 如何串起 Tensor Cores、occupancy 控制、计算与搬运重叠、TMA 和 WGMMA 等高级机制。',
    '- TPU 的 reduce-scatter、all-gather、collective matmul 如何和数据搬运、重叠执行这些核心原则相连。',
    '- TIRx 如何表达 scope、layout、dispatch 和面向硬件的张量程序。',
    '',
    '## 本地阅读',
    '',
    '- GitHub Pages 内容位于 `docs/`。',
    '- 章节 Markdown 文件位于 `docs/chapters/`。',
    '- 运行 `npm run serve` 后打开 <http://127.0.0.1:4193/>。',
    '- 私有内容更新后运行 `LARK_DOC_URL=<url> npm run sync` 重新生成章节。',
    '',
    '## 站点',
    '',
    '- 站点框架：Docsify + GitHub Pages',
    '- 发布方式：GitHub Actions 发布 `docs/` 目录',
    ...(displayUpdatedAt ? ['', `最后一次更新时间：\`${displayUpdatedAt}\``] : []),
    '',
  ].join('\n');

  return { english, chinese };
}

async function writeRootReadmes({
  title,
  siteUrl,
  syncedAt,
  lastUpdated,
}) {
  const readmes = buildRootReadmes({
    title,
    siteUrl,
    syncedAt,
    lastUpdated,
  });
  await writeFile('README.md', readmes.english);
  await writeFile('README.zh-CN.md', readmes.chinese);
}

async function syncDocs({
  docUrl = process.env.LARK_DOC_URL,
  docsDir = DEFAULT_DOCS_DIR,
  fixturePath,
  downloadImages = true,
} = {}) {
  const source = fixturePath
    ? JSON.parse(await readFile(fixturePath, 'utf8'))
    : await fetchLarkMarkdown(docUrl);
  const markdown = source.markdown ?? source.data?.document?.content;
  const revisionId = source.revisionId ?? source.data?.document?.revision_id;

  if (!markdown) {
    throw new Error('No markdown content found in source payload.');
  }

  const titleMatch = markdown.match(/^<title>([\s\S]*?)<\/title>/);
  const title = titleMatch ? cleanTitle(titleMatch[1]) : '高性能计算';
  const syncedAt = new Date().toISOString();
  const lastUpdated = formatLastUpdated(new Date(syncedAt));
  const chapters = splitIntoChapters(cleanMarkdownForRendering(markdown)).map((chapter) => ({
    ...chapter,
    fileName: `chapters/${chapter.slug}.md`,
  }));

  await mkdir(docsDir, { recursive: true });
  await rm(path.join(docsDir, 'chapters'), { recursive: true, force: true });
  await rm(path.join(docsDir, 'assets', 'images'), { recursive: true, force: true });
  await mkdir(path.join(docsDir, 'chapters'), { recursive: true });
  await mkdir(path.join(docsDir, 'assets'), { recursive: true });

  for (const chapter of chapters) {
    let content = chapter.content;

    if (downloadImages) {
      const replacements = await downloadChapterImages(chapter, path.join(docsDir, 'assets', 'images'));
      content = rewriteImageUrls(content, replacements);
    }

    await writeFile(path.join(docsDir, chapter.fileName), buildChapterPage(content, lastUpdated));
  }

  await writeFile(path.join(docsDir, 'README.md'), buildDocsReadme({
    title,
    language: 'zh',
    syncedAt,
    lastUpdated,
    siteUrl: DEFAULT_SITE_URL,
    chapters,
  }));
  await rm(path.join(docsDir, 'README.zh-CN.md'), { force: true });
  await writeFile(path.join(docsDir, 'README.en.md'), buildDocsReadme({
    title,
    language: 'en',
    syncedAt,
    lastUpdated,
    siteUrl: DEFAULT_SITE_URL,
    chapters,
  }));
  await writeFile(path.join(docsDir, '_sidebar.md'), buildSidebar(chapters));
  await writeFile(path.join(docsDir, 'index.html'), buildIndexHtml(title));
  await writeFile(path.join(docsDir, 'assets', 'site.css'), buildSiteCss());
  await writeFile(path.join(docsDir, '.nojekyll'), '');
  await writeFile(path.join(docsDir, 'source.json'), buildSourceMetadata({
    title,
    syncedAt,
    lastUpdated,
    chapters: chapters.map(({ title: chapterTitle, fileName, slug }) => ({ title: chapterTitle, fileName, slug })),
  }));
  await writeRootReadmes({
    title,
    siteUrl: DEFAULT_SITE_URL,
    syncedAt,
    lastUpdated,
  });

  return { title, revisionId, chapters };
}

function parseArgs(argv) {
  const options = {};

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === '--doc') {
      options.docUrl = argv[++index];
    } else if (arg === '--docs-dir') {
      options.docsDir = argv[++index];
    } else if (arg === '--fixture') {
      options.fixturePath = argv[++index];
    } else if (arg === '--no-images') {
      options.downloadImages = false;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const result = await syncDocs(options);
  console.log(`Synced ${result.chapters.length} chapters from revision ${result.revisionId}.`);
}

const isEntrypoint = process.argv[1]
  ? import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
  : false;

if (isEntrypoint) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
