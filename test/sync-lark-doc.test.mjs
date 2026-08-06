import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  buildChapterPage,
  buildIndexHtml,
  buildDocsReadme,
  buildRootReadmes,
  buildSourceMetadata,
  buildSidebar,
  buildSiteCss,
  chapterSlug,
  cleanMarkdownForRendering,
  extractImages,
  extensionFromBytes,
  extensionFromResponse,
  formatLastUpdated,
  parseArgs,
  rewriteImageUrls,
  splitIntoChapters,
} from '../scripts/sync-lark-doc.mjs';

test('parses a positional source URL forwarded by npm', () => {
  assert.deepEqual(
    parseArgs(['https://bytedance.larkoffice.com/wiki/CMQxwBKEXi3wJAkDthhcCHyanPd']),
    { docUrl: 'https://bytedance.larkoffice.com/wiki/CMQxwBKEXi3wJAkDthhcCHyanPd' },
  );
});

test('splits markdown into introduction and top-level heading chapters', () => {
  const markdown = [
    '<title>GPU Notes</title>',
    '',
    'preface line',
    '',
    '# Hardware',
    '',
    'body',
    '',
    '```',
    '# not a chapter',
    '```',
    '',
    '# Memory',
    '',
    'memory body',
  ].join('\n');

  const chapters = splitIntoChapters(markdown);

  assert.deepEqual(
    chapters.map((chapter) => chapter.title),
    ['GPU Notes', 'Hardware', 'Memory'],
  );
  assert.equal(chapters[0].slug, '00-introduction');
  assert.equal(chapters[1].slug, '01-hardware');
  assert.match(chapters[1].content, /# Hardware/);
  assert.match(chapters[1].content, /# not a chapter/);
  assert.equal(chapters[2].slug, '02-memory');
});

test('does not create an empty introduction chapter when source only has a document title', () => {
  const markdown = [
    '<title>GPU Notes</title>',
    '',
    '# Hardware',
    '',
    'body',
  ].join('\n');

  const chapters = splitIntoChapters(markdown);

  assert.deepEqual(
    chapters.map((chapter) => chapter.title),
    ['Hardware'],
  );
  assert.equal(chapters[0].slug, '00-hardware');
});

test('chapter slugs are stable, readable, and unique', () => {
  const seen = new Set();

  assert.equal(chapterSlug('计算的硬件加速思路', 1, seen), '01-computing-hardware-acceleration-ideas');
  assert.equal(chapterSlug('**`wgmma`：warp-group level tensor core instructions**', 9, seen), '09-wgmma-warp-group-level-tensor-core-instructions');
  assert.equal(chapterSlug('计算的硬件加速思路', 10, seen), '10-computing-hardware-acceleration-ideas');
});

test('extracts and rewrites markdown image URLs', () => {
  const markdown = [
    '![arch](https://example.com/a.png)',
    '![](https://example.com/b)',
    '![local](assets/local.png)',
  ].join('\n');

  assert.deepEqual(extractImages(markdown), [
    { alt: 'arch', url: 'https://example.com/a.png' },
    { alt: '', url: 'https://example.com/b' },
  ]);

  const rewritten = rewriteImageUrls(markdown, new Map([
    ['https://example.com/a.png', 'assets/images/chapter-01/image-01.png'],
    ['https://example.com/b', 'assets/images/chapter-01/image-02.png'],
  ]));

  assert.ok(rewritten.includes('![arch](assets/images/chapter-01/image-01.png)'));
  assert.ok(rewritten.includes('![](assets/images/chapter-01/image-02.png)'));
  assert.ok(rewritten.includes('![local](assets/local.png)'));
});

test('rewrites downloaded chapter images relative to chapter markdown files', () => {
  const markdown = '![arch](https://example.com/a.png)';
  const rewritten = rewriteImageUrls(markdown, new Map([
    ['https://example.com/a.png', '../assets/images/chapter-01/image-01.jpg'],
  ]));

  assert.equal(rewritten, '![arch](../assets/images/chapter-01/image-01.jpg)');
});

test('cleans common Lark markdown escaping without changing code fences', () => {
  const markdown = [
    '接着对 3072 \\* 3072 矩阵乘法（GEMM）在 \\*\\*RTX 4000 Ada\\*\\* 上分析。',
    '',
    'OI 随 \\$n\\$ 线性增长，带宽是 \\$360 \\text{ GB/s}\\$。',
    '时间是 \\$\\text{Time}_{ideal\\_dram}\\$。',
    '- **计算峰值 (Peak FP32)**: 26.7 \\text TFLOPS}\\$ (忽略 Tensor Cores)',
    '',
    '```text',
    '\\*\\*literal\\*\\* and \\$literal\\$',
    '```',
  ].join('\n');

  assert.equal(cleanMarkdownForRendering(markdown), [
    '接着对 3072 * 3072 矩阵乘法（GEMM）在 **RTX 4000 Ada** 上分析。',
    '',
    'OI 随 $n$ 线性增长，带宽是 $360 \\text{ GB/s}$。',
    '时间是 $\\text{Time}_{ideal_dram}$。',
    '- **计算峰值 (Peak FP32)**: $26.7 \\text{ TFLOPS}$ (忽略 Tensor Cores)',
    '',
    '```text',
    '\\*\\*literal\\*\\* and \\$literal\\$',
    '```',
  ].join('\n'));
});

test('normalizes code fence languages for Prism without changing code contents', () => {
  const markdown = [
    '```C++',
    'std::cout << "```Python";',
    '```',
    '',
    '```Python',
    'print("hi")',
    '```',
    '',
    '```Bash',
    'echo ok',
    '```',
    '',
    '```Java',
    'class Main {}',
    '```',
    '',
    '```YAML',
    'name: gpu-hpc-book',
    '```',
    '',
    '```text',
    'plain',
    '```',
  ].join('\n');

  assert.equal(cleanMarkdownForRendering(markdown), [
    '```cpp',
    'std::cout << "```Python";',
    '```',
    '',
    '```python',
    'print("hi")',
    '```',
    '',
    '```bash',
    'echo ok',
    '```',
    '',
    '```java',
    'class Main {}',
    '```',
    '',
    '```yaml',
    'name: gpu-hpc-book',
    '```',
    '',
    '```text',
    'plain',
    '```',
  ].join('\n'));
});

test('builds docsify sidebar from generated chapters', () => {
  const sidebar = buildSidebar([
    {
      title: 'GPU Notes',
      fileName: 'chapters/00-introduction.md',
      content: [
        '# GPU Notes',
        '',
        '## Overview',
        '',
        '```',
        '## not a heading',
        '```',
      ].join('\n'),
    },
    {
      title: 'Hardware',
      fileName: 'chapters/01-hardware.md',
      content: [
        '# Hardware',
        '',
        '## GPU 编程模型',
        '',
        '### CUDA Core & SM',
        '',
        '#### hidden detail',
        '',
        '## GPU 编程模型',
      ].join('\n'),
    },
  ]);

  assert.equal(sidebar, [
    '- [首页](/)',
    '- [GPU Notes](/chapters/00-introduction.md)',
    '  - [Overview](/chapters/00-introduction.md?id=overview)',
    '- [Hardware](/chapters/01-hardware.md)',
    '  - [GPU 编程模型](/chapters/01-hardware.md?id=gpu-编程模型)',
    '    - [CUDA Core & SM](/chapters/01-hardware.md?id=cuda-core-sm)',
    '  - [GPU 编程模型](/chapters/01-hardware.md?id=gpu-编程模型-1)',
    '',
    '<!-- 原文链接：[https://fangpin.github.io/gpu-hpc-book/#/_sidebar.md](https://fangpin.github.io/gpu-hpc-book/#/_sidebar.md) -->',
    '',
  ].join('\n'));
});

test('docs readme shows only the last updated time instead of sync details', () => {
  const readme = buildDocsReadme({
    title: 'GPU Notes',
    language: 'zh',
    siteUrl: 'https://fangpin.github.io/gpu-hpc-book/',
    repoUrl: 'https://github.com/fangpin/gpu-hpc-book',
    sourceUrl: 'https://example.com/source',
    documentId: 'doc-token',
    revisionId: '123',
    lastUpdated: '2026-08-05 18:00:00 CST',
    chapters: [
      { title: 'Hardware', fileName: 'chapters/01-hardware.md' },
    ],
  });

  assert.match(readme, /在线阅读：\[https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\]\(https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\)/);
  assert.match(readme, /GitHub：\[fangpin\/gpu-hpc-book\]\(https:\/\/github\.com\/fangpin\/gpu-hpc-book\)/);
  assert.match(readme, /最后一次更新时间：`2026-08-05 18:00:00 CST`/);
  assert.doesNotMatch(readme, /## 同步信息/);
  assert.doesNotMatch(readme, /飞书源文档/);
  assert.doesNotMatch(readme, /Doc token/);
  assert.doesNotMatch(readme, /Revision/);
  assert.doesNotMatch(readme, /Synced at/);
});

test('docs readme is Chinese by default and links to the English version', () => {
  const readme = buildDocsReadme({
    title: 'GPU Notes',
    siteUrl: 'https://fangpin.github.io/gpu-hpc-book/',
    repoUrl: 'https://github.com/fangpin/gpu-hpc-book',
    lastUpdated: '2026-08-05 18:00:00 CST',
    chapters: [
      {
        title: 'Hardware',
        fileName: 'chapters/00-hardware.md',
        content: [
          '# Hardware',
          '',
          '## GPU 编程模型',
        ].join('\n'),
      },
    ],
  });

  assert.match(readme, /^# GPU Notes/);
  assert.match(readme, /\[English\]\(README\.en\.md\)/);
  assert.match(readme, /在线阅读：\[https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\]\(https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\)/);
  assert.match(readme, /GitHub：\[fangpin\/gpu-hpc-book\]\(https:\/\/github\.com\/fangpin\/gpu-hpc-book\)/);
  assert.match(readme, /从硬件加速、并行编程到 GPU\/TPU 性能优化/);
  assert.match(readme, /## 章节/);
  assert.match(readme, /- \[Hardware\]\(chapters\/00-hardware\.md\)/);
  assert.match(readme, /  - \[GPU 编程模型\]\(chapters\/00-hardware\.md\?id=gpu-编程模型\)/);
  assert.match(readme, /最后一次更新时间：`2026-08-05 18:00:00 CST`/);
  assert.match(readme, /\n原文链接：\[https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\]\(https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\)\n$/);
  assert.doesNotMatch(readme, /Why Read This Book/);
});

test('docs English readme links back to the Chinese homepage', () => {
  const readme = buildDocsReadme({
    title: 'GPU Notes',
    language: 'en',
    siteUrl: 'https://fangpin.github.io/gpu-hpc-book/',
    repoUrl: 'https://github.com/fangpin/gpu-hpc-book',
    lastUpdated: '2026-08-05 18:00:00 CST',
    chapters: [
      {
        title: 'Hardware',
        fileName: 'chapters/00-hardware.md',
        content: [
          '# Hardware',
          '',
          '## GPU 编程模型',
        ].join('\n'),
      },
    ],
  });

  assert.match(readme, /^# GPU Notes/);
  assert.match(readme, /\[中文\]\(README\.md\)/);
  assert.match(readme, /Read online: \[https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\]\(https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\)/);
  assert.match(readme, /GitHub: \[fangpin\/gpu-hpc-book\]\(https:\/\/github\.com\/fangpin\/gpu-hpc-book\)/);
  assert.match(readme, /This book is a practical tour of high-performance computing/);
  assert.match(readme, /## Chapters/);
  assert.match(readme, /- \[Hardware\]\(chapters\/00-hardware\.md\)/);
  assert.match(readme, /  - \[GPU 编程模型\]\(chapters\/00-hardware\.md\?id=gpu-编程模型\)/);
  assert.match(readme, /Last updated: `2026-08-05 18:00:00 CST`/);
  assert.match(readme, /\n原文链接：\[https:\/\/fangpin\.github\.io\/gpu-hpc-book\/#\/README\.en\.md\]\(https:\/\/fangpin\.github\.io\/gpu-hpc-book\/#\/README\.en\.md\)\n$/);
  assert.doesNotMatch(readme, /阅读方式/);
});

test('docs readme chapter list includes chapter subsections', () => {
  const readme = buildDocsReadme({
    title: 'GPU Notes',
    language: 'zh',
    syncedAt: '2026-08-05T10:00:00.000Z',
    chapters: [
      {
        title: 'Hardware',
        fileName: 'chapters/00-hardware.md',
        content: [
          '# Hardware',
          '',
          '## GPU 编程模型',
          '',
          '### CUDA Core & SM',
          '',
          '#### hidden detail',
        ].join('\n'),
      },
    ],
  });

  assert.match(readme, /- \[Hardware\]\(chapters\/00-hardware\.md\)/);
  assert.match(readme, /  - \[GPU 编程模型\]\(chapters\/00-hardware\.md\?id=gpu-编程模型\)/);
  assert.match(readme, /    - \[CUDA Core & SM\]\(chapters\/00-hardware\.md\?id=cuda-core-sm\)/);
  assert.doesNotMatch(readme, /hidden detail/);
});

test('chapter pages include the last updated time', () => {
  const page = buildChapterPage(
    '# Hardware\n\nBody\n',
    '2026-08-05 18:00:00 CST',
    'https://fangpin.github.io/gpu-hpc-book/#/chapters/00-hardware.md',
  );

  assert.equal(page, [
    '# Hardware',
    '',
    'Body',
    '',
    '---',
    '',
    '最后一次更新时间：`2026-08-05 18:00:00 CST`',
    '',
    '原文链接：[https://fangpin.github.io/gpu-hpc-book/#/chapters/00-hardware.md](https://fangpin.github.io/gpu-hpc-book/#/chapters/00-hardware.md)',
    '',
  ].join('\n'));
});

test('formats last updated time in Beijing time for readers', () => {
  assert.equal(
    formatLastUpdated(new Date('2026-08-05T10:00:00.000Z')),
    '2026-08-05 18:00:00 CST',
  );
});

test('root readmes are bilingual and link to each other', () => {
  const { english, chinese } = buildRootReadmes({
    title: 'GPU Notes',
    sourceUrl: 'https://example.com/source',
    siteUrl: 'https://fangpin.github.io/gpu-hpc-book/',
    lastUpdated: '2026-08-05 18:00:00 CST',
  });

  assert.match(english, /^# GPU Notes/);
  assert.match(english, /\[中文\]\(README\.zh-CN\.md\)/);
  assert.match(english, /Read online: \[https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\]\(https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\)/);
  assert.match(english, /What You Will Learn/);
  assert.match(english, /GPU programming model/);
  assert.match(english, /Read the book locally/);
  assert.match(english, /npm run serve/);
  assert.match(english, /Last updated: `2026-08-05 18:00:00 CST`/);
  assert.match(english, /\n原文链接：\[https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\]\(https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\)\n$/);
  assert.doesNotMatch(english, /Source document/);
  assert.doesNotMatch(english, /https:\/\/example\.com\/source/);

  assert.match(chinese, /^# GPU Notes/);
  assert.match(chinese, /\[English\]\(README\.md\)/);
  assert.match(chinese, /在线阅读：\[https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\]\(https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\)/);
  assert.match(chinese, /你会读到什么/);
  assert.match(chinese, /GPU 编程模型/);
  assert.match(chinese, /本地阅读/);
  assert.match(chinese, /最后一次更新时间：`2026-08-05 18:00:00 CST`/);
  assert.match(chinese, /\n原文链接：\[https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\]\(https:\/\/fangpin\.github\.io\/gpu-hpc-book\/\)\n$/);
  assert.doesNotMatch(chinese, /源文档/);
  assert.doesNotMatch(chinese, /https:\/\/example\.com\/source/);
});

test('public source metadata does not expose private document source details', () => {
  const metadata = buildSourceMetadata({
    title: 'GPU Notes',
    sourceUrl: 'https://example.com/source',
    documentId: 'doc-token',
    revisionId: '123',
    syncedAt: '2026-08-05T10:00:00.000Z',
    chapters: [
      { title: 'Hardware', fileName: 'chapters/00-hardware.md', slug: '00-hardware' },
    ],
  });

  assert.match(metadata, /"title": "GPU Notes"/);
  assert.match(metadata, /"chapterCount": 1/);
  assert.doesNotMatch(metadata, /sourceUrl/);
  assert.doesNotMatch(metadata, /documentId/);
  assert.doesNotMatch(metadata, /revisionId/);
  assert.doesNotMatch(metadata, /example\.com/);
  assert.doesNotMatch(metadata, /doc-token/);
});

test('detects image extension from file bytes before response headers', () => {
  assert.equal(extensionFromBytes(Buffer.from([0xff, 0xd8, 0xff, 0xe0])), '.jpg');
  assert.equal(extensionFromBytes(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])), '.png');
  assert.equal(extensionFromBytes(Buffer.from('GIF89a')), '.gif');
  assert.equal(extensionFromBytes(Buffer.from('RIFFxxxxWEBP')), '.webp');
  assert.equal(extensionFromBytes(Buffer.from('not an image')), undefined);
});

test('rejects html responses when downloading images', () => {
  const response = {
    headers: new Map([
      ['content-type', 'text/html; charset=utf-8'],
    ]),
  };

  assert.throws(
    () => extensionFromResponse('https://example.com/image', response, Buffer.from('<!doctype html>')),
    /Expected image response/,
  );
});

test('site shell loads docsify latex support', () => {
  const html = buildIndexHtml('GPU Notes');

  assert.match(html, /katex\.min\.css/);
  assert.match(html, /docsify-latex/);
  assert.match(html, /relativePath: true/);
  assert.match(html, /window\.\$docsify/);
});

test('site shell marks only the current sidebar chapter as expanded', () => {
  const html = buildIndexHtml('GPU Notes');

  assert.match(html, /function updateSidebarCurrentChapter/);
  assert.match(html, /is-current-chapter/);
  assert.match(html, /querySelectorAll\(':scope > ul > li'\)/);
  assert.match(html, /hook\.doneEach\(updateSidebarCurrentChapter\)/);
  assert.match(html, /hook\.mounted\(updateSidebarCurrentChapter\)/);
});

test('site css collapses subsection lists outside the current sidebar chapter', () => {
  const css = buildSiteCss();

  assert.match(css, /\.sidebar-nav > ul > li > ul/);
  assert.match(css, /display: none;/);
  assert.match(css, /\.sidebar-nav > ul > li\.is-current-chapter > ul/);
  assert.match(css, /display: block;/);
});

test('site css makes docs comfortable to read on phones', () => {
  const css = buildSiteCss();

  assert.match(css, /@media \(max-width: 768px\)/);
  assert.match(css, /\.markdown-section\s*{[\s\S]*padding: 22px 18px 40px;/);
  assert.match(css, /\.markdown-section\s*{[\s\S]*font-size: 16px;/);
  assert.match(css, /\.markdown-section\s*{[\s\S]*line-height: 1\.72;/);
  assert.match(css, /\.content\s*{[\s\S]*left: 0;/);
  assert.match(css, /\.sidebar-toggle\s*{[\s\S]*position: fixed;/);
  assert.match(css, /\.markdown-section pre\s*{[\s\S]*overflow-x: auto;/);
  assert.match(css, /\.markdown-section table\s*{[\s\S]*display: block;/);
  assert.match(css, /\.markdown-section img\s*{[\s\S]*max-width: 100%;/);
  assert.match(css, /\.app-nav\s*{[\s\S]*display: none;/);
});

test('site shell links back to the GitHub repository', () => {
  const html = buildIndexHtml('GPU Notes', 'https://github.com/fangpin/gpu-hpc-book');

  assert.match(html, /repo: 'https:\/\/github\.com\/fangpin\/gpu-hpc-book'/);
});

test('site shell loads Prism languages used by generated chapters', () => {
  const html = buildIndexHtml('GPU Notes');

  assert.match(html, /prism-python\.min\.js/);
  assert.match(html, /prism-c\.min\.js/);
  assert.match(html, /prism-cpp\.min\.js/);
  assert.match(html, /prism-bash\.min\.js/);
  assert.match(html, /prism-java\.min\.js/);
  assert.match(html, /prism-yaml\.min\.js/);
});

test('serve script uses UTF-8 response headers for markdown previews', async () => {
  const { Utf8DocsHandler } = await import('../scripts/serve-docs.mjs');

  assert.match(Utf8DocsHandler.extensions_map['.md'], /charset=utf-8/);
});
