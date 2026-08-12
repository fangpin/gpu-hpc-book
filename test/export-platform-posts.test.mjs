import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  buildMarkdownDraft,
  buildPlainTextDraft,
  buildPlatformExportReadme,
  renderMarkdownForRichText,
  resolveAssetUrl,
} from '../scripts/export-platform-posts.mjs';

const siteUrl = 'https://fangpin.github.io/gpu-hpc-book/';

test('rewrites relative docs assets to public GitHub Pages URLs', () => {
  assert.equal(
    resolveAssetUrl('../assets/images/03-roofline-model/image-01.jpg', {
      fileName: 'chapters/03-roofline-model.md',
      siteUrl,
    }),
    'https://fangpin.github.io/gpu-hpc-book/assets/images/03-roofline-model/image-01.jpg',
  );

  assert.equal(
    resolveAssetUrl('https://example.com/image.png', {
      fileName: 'chapters/03-roofline-model.md',
      siteUrl,
    }),
    'https://example.com/image.png',
  );
});

test('builds a markdown draft with public image URLs', () => {
  const markdown = [
    '# Roofline Model',
    '',
    '![roofline](../assets/images/03-roofline-model/image-01.jpg)',
    '',
    '```cpp',
    'float x = 1.0f;',
    '```',
  ].join('\n');

  const draft = buildMarkdownDraft(markdown, {
    fileName: 'chapters/03-roofline-model.md',
    siteUrl,
  });

  assert.match(draft, /!\[roofline\]\(https:\/\/fangpin\.github\.io\/gpu-hpc-book\/assets\/images\/03-roofline-model\/image-01\.jpg\)/);
  assert.match(draft, /```cpp\nfloat x = 1\.0f;\n```/);
});

test('builds a markdown draft with a labeled source footer link', () => {
  const draft = buildMarkdownDraft(
    '原文链接：[https://fangpin.github.io/gpu-hpc-book/#/chapters/03-roofline-model.md](https://fangpin.github.io/gpu-hpc-book/#/chapters/03-roofline-model.md)',
    {
      fileName: 'chapters/03-roofline-model.md',
      siteUrl,
    },
  );

  assert.equal(
    draft,
    '[原文链接](https://fangpin.github.io/gpu-hpc-book/#/chapters/03-roofline-model.md)',
  );
});

test('renders markdown as copy-friendly rich text html', () => {
  const markdown = [
    '# Roofline Model',
    '',
    '程序性能 $P$ 由 **带宽** 和 `FLOPs` 决定。',
    '',
    '![roofline](../assets/images/03-roofline-model/image-01.jpg)',
    '',
    '| 操作 | 结论 |',
    '|-|-|',
    '| GEMM | compute-bound |',
    '',
    '```cpp',
    'float x = 1.0f;',
    '```',
  ].join('\n');

  const html = renderMarkdownForRichText(markdown, {
    fileName: 'chapters/03-roofline-model.md',
    siteUrl,
  });

  assert.match(html, /<h1 style="/);
  assert.match(html, /<strong>带宽<\/strong>/);
  assert.match(html, /<code style="[^"]*">FLOPs<\/code>/);
  assert.match(html, /class="katex"/);
  assert.match(html, /src="https:\/\/fangpin\.github\.io\/gpu-hpc-book\/assets\/images\/03-roofline-model\/image-01\.jpg"/);
  assert.match(html, /<table style="/);
  assert.match(html, /<th style="[^"]*">操作<\/th>/);
  assert.match(html, /<pre style="/);
  assert.doesNotMatch(html, /\|-+\|/);
});

test('renders inline math as KaTeX in rich text html', () => {
  const markdown = '程序性能 $P = OI \\times BW$ 由带宽决定。';

  const html = renderMarkdownForRichText(markdown, {
    fileName: 'chapters/03-roofline-model.md',
    siteUrl,
  });

  assert.match(html, /class="katex"/);
  assert.match(html, /aria-hidden="true"/);
  assert.doesNotMatch(html, /\$P = OI \\times BW\$/);
});

test('renders single-line display math without swallowing following blocks', () => {
  const markdown = [
    '# Formula',
    '',
    '$$P = \\min(peak, oi \\times bandwidth)$$',
    '',
    'after formula',
  ].join('\n');

  const html = renderMarkdownForRichText(markdown, {
    fileName: 'chapters/03-roofline-model.md',
    siteUrl,
  });

  assert.match(html, /class="katex-display"/);
  assert.doesNotMatch(html, /\$\$P = \\min\(peak, oi \\times bandwidth\)\$\$/);
  assert.match(html, /<p style="[^"]*">after formula<\/p>/);
  assert.doesNotMatch(html, /<code>[\s\S]*after formula[\s\S]*<\/code>/);
});

test('renders source footer as a labeled link without showing the raw URL', () => {
  const html = renderMarkdownForRichText(
    '原文链接：[https://fangpin.github.io/gpu-hpc-book/#/chapters/03-roofline-model.md](https://fangpin.github.io/gpu-hpc-book/#/chapters/03-roofline-model.md)',
    {
      fileName: 'chapters/03-roofline-model.md',
      siteUrl,
    },
  );

  assert.match(html, /<a style="[^"]*" href="https:\/\/fangpin\.github\.io\/gpu-hpc-book\/#\/chapters\/03-roofline-model\.md">原文链接<\/a>/);
  assert.doesNotMatch(html, />https:\/\/fangpin\.github\.io\/gpu-hpc-book\/#\/chapters\/03-roofline-model\.md<\/a>/);
});

test('builds a plain text draft for social text editors', () => {
  const markdown = [
    '# Roofline Model',
    '',
    '参考 [论文](https://example.com/paper)，程序性能 $P$ 由 **带宽** 决定。',
    '',
    '![roofline](../assets/images/03-roofline-model/image-01.jpg)',
  ].join('\n');

  const text = buildPlainTextDraft(markdown, {
    fileName: 'chapters/03-roofline-model.md',
    siteUrl,
  });

  assert.match(text, /^Roofline Model/);
  assert.match(text, /参考 论文 \(https:\/\/example\.com\/paper\)，程序性能 \$P\$ 由 带宽 决定。/);
  assert.match(text, /图片：roofline https:\/\/fangpin\.github\.io\/gpu-hpc-book\/assets\/images\/03-roofline-model\/image-01\.jpg/);
  assert.doesNotMatch(text, /\*\*/);
});

test('export readme explains recommended workflows by platform family', () => {
  const readme = buildPlatformExportReadme({
    title: 'GPU 高性能计算',
    siteUrl,
    outputDir: 'dist/platform-posts',
    chapters: [
      {
        title: 'Roofline Model',
        slug: '03-roofline-model',
        fileName: 'chapters/03-roofline-model.md',
      },
    ],
  });

  assert.match(readme, /微信公众号、知乎：打开 `rich.html`/);
  assert.match(readme, /CSDN：优先使用 `markdown.md`/);
  assert.match(readme, /小红书、抖音长文、头条号、百家号：使用 `plain.txt`/);
  assert.match(readme, /dist\/platform-posts\/03-roofline-model\/rich\.html/);
});
