import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import katex from 'katex';

const DEFAULT_DOCS_DIR = 'docs';
const DEFAULT_OUTPUT_DIR = 'dist/platform-posts';
const DEFAULT_SITE_URL = 'https://fangpin.github.io/gpu-hpc-book/';

const RICH_STYLE = {
  article: [
    'box-sizing:border-box',
    'max-width:760px',
    'margin:0 auto',
    'padding:32px 20px 56px',
    'color:#1f2937',
    'font-size:16px',
    'line-height:1.78',
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Hiragino Sans GB','Microsoft YaHei',Arial,sans-serif",
  ].join(';'),
  h1: 'margin:0 0 24px;color:#111827;font-size:28px;line-height:1.28;font-weight:700',
  h2: 'margin:34px 0 14px;color:#111827;font-size:23px;line-height:1.36;font-weight:700;border-left:4px solid #2563eb;padding-left:12px',
  h3: 'margin:26px 0 12px;color:#111827;font-size:19px;line-height:1.42;font-weight:700',
  h4: 'margin:22px 0 10px;color:#111827;font-size:17px;line-height:1.45;font-weight:700',
  p: 'margin:12px 0;color:#1f2937;font-size:16px;line-height:1.78',
  a: 'color:#2563eb;text-decoration:none;border-bottom:1px solid rgba(37,99,235,.28)',
  code: "font-family:SFMono-Regular,Consolas,'Liberation Mono',Menlo,monospace;background:#f3f4f6;color:#be123c;border-radius:4px;padding:1px 5px;font-size:88%",
  math: "font-family:SFMono-Regular,Consolas,'Liberation Mono',Menlo,monospace;background:#f8fafc;color:#334155;border-radius:4px;padding:1px 5px;font-size:92%",
  displayMath: 'margin:18px 0;text-align:center;overflow-x:auto;overflow-y:hidden',
  pre: "margin:16px 0;padding:14px 16px;background:#0f172a;color:#e5e7eb;border-radius:6px;overflow-x:auto;font-size:13px;line-height:1.62;font-family:SFMono-Regular,Consolas,'Liberation Mono',Menlo,monospace;white-space:pre",
  blockquote: 'margin:16px 0;padding:10px 14px;border-left:4px solid #93c5fd;background:#eff6ff;color:#1e3a8a',
  ul: 'margin:12px 0 12px 1.2em;padding:0;color:#1f2937;font-size:16px;line-height:1.78',
  ol: 'margin:12px 0 12px 1.2em;padding:0;color:#1f2937;font-size:16px;line-height:1.78',
  li: 'margin:4px 0',
  table: 'width:100%;border-collapse:collapse;margin:18px 0;font-size:14px;line-height:1.6',
  th: 'border:1px solid #d1d5db;background:#f3f4f6;color:#111827;padding:8px 10px;text-align:left;font-weight:700',
  td: 'border:1px solid #d1d5db;color:#1f2937;padding:8px 10px;vertical-align:top',
  img: 'display:block;max-width:100%;height:auto;margin:16px auto;border:1px solid #e5e7eb;border-radius:6px',
  figure: 'margin:18px 0;text-align:center',
  figcaption: 'margin-top:6px;color:#6b7280;font-size:13px;line-height:1.5',
  hr: 'border:0;border-top:1px solid #e5e7eb;margin:28px 0',
};

function normalizeSiteUrl(siteUrl = DEFAULT_SITE_URL) {
  return siteUrl.endsWith('/') ? siteUrl : `${siteUrl}/`;
}

function splitUrlSuffix(url) {
  const match = url.match(/^([^?#]*)([?#].*)?$/);
  return {
    pathPart: match?.[1] ?? url,
    suffix: match?.[2] ?? '',
  };
}

function encodeUrlPath(urlPath) {
  return urlPath
    .split('/')
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join('/');
}

function isExternalUrl(url) {
  return /^(?:https?:|data:|mailto:|tel:)/i.test(url) || url.startsWith('//');
}

function resolveRelativeDocsPath(relativeUrl, fileName) {
  const baseDir = path.posix.dirname(fileName.replace(/^\/+/, ''));
  return path.posix
    .normalize(path.posix.join(baseDir === '.' ? '' : baseDir, relativeUrl.replace(/^\/+/, '')))
    .replace(/^(\.\.\/)+/, '');
}

function buildGithubPagesUrl(fileName = '', siteUrl = DEFAULT_SITE_URL, suffix = '') {
  const normalizedSiteUrl = normalizeSiteUrl(siteUrl);
  const normalizedFileName = fileName.replace(/^\/+/, '');

  if (!normalizedFileName || normalizedFileName === 'README.md') {
    return `${normalizedSiteUrl}${suffix}`;
  }

  return `${normalizedSiteUrl}#/${normalizedFileName}${suffix}`;
}

export function resolveAssetUrl(url, { fileName, siteUrl = DEFAULT_SITE_URL }) {
  const trimmed = url.trim();

  if (!trimmed || isExternalUrl(trimmed) || trimmed.startsWith('#')) {
    return trimmed.startsWith('//') ? `https:${trimmed}` : trimmed;
  }

  const { pathPart, suffix } = splitUrlSuffix(trimmed);
  const resolvedPath = resolveRelativeDocsPath(pathPart, fileName);
  return `${normalizeSiteUrl(siteUrl)}${encodeUrlPath(resolvedPath)}${suffix}`;
}

function resolveLinkUrl(url, { fileName, siteUrl = DEFAULT_SITE_URL }) {
  const trimmed = url.trim();

  if (!trimmed || isExternalUrl(trimmed)) {
    return trimmed.startsWith('//') ? `https:${trimmed}` : trimmed;
  }

  if (trimmed.startsWith('#')) {
    return buildGithubPagesUrl(fileName, siteUrl, trimmed);
  }

  const { pathPart, suffix } = splitUrlSuffix(trimmed);
  const resolvedPath = resolveRelativeDocsPath(pathPart, fileName);

  if (/\.md$/i.test(pathPart)) {
    return buildGithubPagesUrl(resolvedPath, siteUrl, suffix);
  }

  return `${normalizeSiteUrl(siteUrl)}${encodeUrlPath(resolvedPath)}${suffix}`;
}

export function buildMarkdownDraft(markdown, { fileName, siteUrl = DEFAULT_SITE_URL }) {
  const withLabeledSourceLinks = markdown.replace(
    /^原文链接：\[(https?:\/\/[^\]]+)]\((https?:\/\/[^)]+)\)$/gm,
    (full, label, url) => (label === url ? `[原文链接](${url})` : full),
  );

  const withPublicImages = withLabeledSourceLinks.replace(
    /!\[([^\]]*)\]\(([^)\s]+)(\s+"[^"]*")?\)/g,
    (full, alt, url, title = '') => `![${alt}](${resolveAssetUrl(url, { fileName, siteUrl })}${title})`,
  );

  return withPublicImages.replace(
    /(^|[^!])\[([^\]]+)\]\(([^)\s]+)(\s+"[^"]*")?\)/g,
    (full, prefix, label, url, title = '') => `${prefix}[${label}](${resolveLinkUrl(url, { fileName, siteUrl })}${title})`,
  );
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function escapeAttribute(value) {
  return escapeHtml(value).replace(/'/g, '&#39;');
}

function restoreTokens(text, tokens) {
  return text.replace(/\u0000(\d+)\u0000/g, (full, index) => tokens[Number(index)] ?? full);
}

function renderMath(formula, { displayMode = false } = {}) {
  try {
    const rendered = katex.renderToString(formula.trim(), {
      displayMode,
      output: 'html',
      strict: false,
      throwOnError: false,
      trust: false,
    });

    if (displayMode) {
      return `<div style="${RICH_STYLE.displayMath}">${rendered}</div>`;
    }

    return rendered;
  } catch {
    const fallback = displayMode ? `$$${formula.trim()}$$` : `$${formula.trim()}$`;
    return `<span style="${RICH_STYLE.math}">${escapeHtml(fallback)}</span>`;
  }
}

function renderInline(text) {
  const tokens = [];
  const put = (html) => {
    tokens.push(html);
    return `\u0000${tokens.length - 1}\u0000`;
  };

  let protectedText = text
    .replace(/`([^`]+)`/g, (full, code) => put(`<code style="${RICH_STYLE.code}">${escapeHtml(code)}</code>`))
    .replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (full, alt, url) => put(renderImage(url, alt)))
    .replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (full, label, url) => put(`<a style="${RICH_STYLE.a}" href="${escapeAttribute(url)}">${escapeHtml(label)}</a>`))
    .replace(/(^|[^$\\])\$([^$\n]+)\$/g, (full, prefix, formula) => `${prefix}${put(renderMath(formula))}`);

  protectedText = escapeHtml(protectedText)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_]+)__/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/(^|[^_])_([^_\n]+)_/g, '$1<em>$2</em>')
    .replace(/&lt;br\s*\/?&gt;/gi, '<br>');

  return restoreTokens(protectedText, tokens);
}

function isFenceStart(line) {
  return /^(```|~~~)/.test(line.trim());
}

function isHorizontalRule(line) {
  return /^-{3,}\s*$/.test(line.trim());
}

function isTableSeparator(line) {
  const trimmed = line.trim();
  return /^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$/.test(trimmed);
}

function isPotentialTableRow(line) {
  return line.trim().includes('|') && !isFenceStart(line);
}

function splitTableRow(line) {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim());
}

function renderImage(url, alt = '') {
  const caption = alt
    ? `<figcaption style="${RICH_STYLE.figcaption}">${escapeHtml(alt)}</figcaption>`
    : '';
  return `<figure style="${RICH_STYLE.figure}"><img style="${RICH_STYLE.img}" src="${escapeAttribute(url)}" alt="${escapeAttribute(alt)}">${caption}</figure>`;
}

function renderSourceLinkParagraph(line) {
  const match = line.trim().match(/^原文链接：\[(https?:\/\/[^\]]+)]\((https?:\/\/[^)]+)\)$/);

  if (!match || match[1] !== match[2]) {
    return undefined;
  }

  return `<p style="${RICH_STYLE.p}"><a style="${RICH_STYLE.a}" href="${escapeAttribute(match[2])}">原文链接</a></p>`;
}

function renderCodeBlock(code, language = '') {
  const label = language ? `<div style="color:#93c5fd;margin-bottom:8px;font-size:12px">${escapeHtml(language)}</div>` : '';
  return `<pre style="${RICH_STYLE.pre}">${label}<code>${escapeHtml(code.replace(/\n$/, ''))}</code></pre>`;
}

function renderTable(lines) {
  const header = splitTableRow(lines[0]);
  const rows = lines.slice(2).map(splitTableRow);
  const thead = `<thead><tr>${header.map((cell) => `<th style="${RICH_STYLE.th}">${renderInline(cell)}</th>`).join('')}</tr></thead>`;
  const tbody = rows.length
    ? `<tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td style="${RICH_STYLE.td}">${renderInline(cell)}</td>`).join('')}</tr>`).join('')}</tbody>`
    : '';
  return `<table style="${RICH_STYLE.table}">${thead}${tbody}</table>`;
}

function isListLine(line) {
  return /^(\s*)(?:[-*+]\s+|\d+\.\s+)/.test(line);
}

function renderList(lines) {
  const ordered = /^\s*\d+\.\s+/.test(lines[0]);
  const tag = ordered ? 'ol' : 'ul';
  const items = lines.map((line) => {
    const match = line.match(/^(\s*)(?:[-*+]\s+|\d+\.\s+)(.*)$/);
    const indent = Math.floor((match?.[1].length ?? 0) / 2);
    const margin = indent > 0 ? `margin-left:${indent * 18}px;` : '';
    return `<li style="${RICH_STYLE.li};${margin}">${renderInline(match?.[2] ?? line.trim())}</li>`;
  });
  return `<${tag} style="${RICH_STYLE[tag]}">${items.join('')}</${tag}>`;
}

function isImageOnlyLine(line) {
  return /^!\[[^\]]*]\([^)]+\)\s*$/.test(line.trim());
}

function isBlockBoundary(lines, index) {
  const line = lines[index] ?? '';
  const next = lines[index + 1] ?? '';

  return !line.trim()
    || isFenceStart(line)
    || /^#{1,6}\s+\S/.test(line)
    || isHorizontalRule(line)
    || isImageOnlyLine(line)
    || isListLine(line)
    || /^>\s?/.test(line)
    || line.trim().startsWith('$$')
    || (isPotentialTableRow(line) && isTableSeparator(next));
}

export function renderMarkdownForRichText(markdown, { fileName, siteUrl = DEFAULT_SITE_URL }) {
  const publicMarkdown = buildMarkdownDraft(markdown, { fileName, siteUrl });
  const lines = publicMarkdown.replace(/\r\n/g, '\n').split('\n');
  const html = [];

  for (let index = 0; index < lines.length;) {
    const line = lines[index];

    if (!line.trim()) {
      index += 1;
      continue;
    }

    const sourceLinkParagraph = renderSourceLinkParagraph(line);
    if (sourceLinkParagraph) {
      html.push(sourceLinkParagraph);
      index += 1;
      continue;
    }

    if (isFenceStart(line)) {
      const language = line.trim().replace(/^(```|~~~)/, '').trim();
      const codeLines = [];
      index += 1;

      while (index < lines.length && !isFenceStart(lines[index])) {
        codeLines.push(lines[index]);
        index += 1;
      }

      if (index < lines.length) {
        index += 1;
      }

      html.push(renderCodeBlock(codeLines.join('\n'), language));
      continue;
    }

    if (line.trim().startsWith('$$')) {
      const singleLineFormula = line.trim().match(/^\$\$([\s\S]*?)\$\$$/);

      if (singleLineFormula) {
        html.push(renderMath(singleLineFormula[1], { displayMode: true }));
        index += 1;
        continue;
      }

      const formulaLines = [line.trim().replace(/^\$\$/, '')];
      index += 1;

      while (index < lines.length && !lines[index].trim().endsWith('$$')) {
        formulaLines.push(lines[index]);
        index += 1;
      }

      if (index < lines.length) {
        formulaLines.push(lines[index].trim().replace(/\$\$$/, ''));
        index += 1;
      }

      html.push(renderMath(formulaLines.join('\n'), { displayMode: true }));
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      const level = Math.min(heading[1].length, 4);
      html.push(`<h${level} style="${RICH_STYLE[`h${level}`]}">${renderInline(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }

    if (isHorizontalRule(line)) {
      html.push(`<hr style="${RICH_STYLE.hr}">`);
      index += 1;
      continue;
    }

    if (isImageOnlyLine(line)) {
      const image = line.trim().match(/^!\[([^\]]*)]\(([^)\s]+)(?:\s+"[^"]*")?\)$/);
      html.push(renderImage(image?.[2] ?? '', image?.[1] ?? ''));
      index += 1;
      continue;
    }

    if (isPotentialTableRow(line) && isTableSeparator(lines[index + 1] ?? '')) {
      const tableLines = [line, lines[index + 1]];
      index += 2;

      while (index < lines.length && isPotentialTableRow(lines[index]) && lines[index].trim()) {
        tableLines.push(lines[index]);
        index += 1;
      }

      html.push(renderTable(tableLines));
      continue;
    }

    if (/^>\s?/.test(line)) {
      const quoteLines = [];

      while (index < lines.length && /^>\s?/.test(lines[index])) {
        quoteLines.push(lines[index].replace(/^>\s?/, ''));
        index += 1;
      }

      html.push(`<blockquote style="${RICH_STYLE.blockquote}">${quoteLines.map((quoteLine) => `<p style="${RICH_STYLE.p}">${renderInline(quoteLine)}</p>`).join('')}</blockquote>`);
      continue;
    }

    if (isListLine(line)) {
      const listLines = [];

      while (index < lines.length && isListLine(lines[index])) {
        listLines.push(lines[index]);
        index += 1;
      }

      html.push(renderList(listLines));
      continue;
    }

    const paragraph = [];
    while (index < lines.length && !isBlockBoundary(lines, index)) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    html.push(`<p style="${RICH_STYLE.p}">${renderInline(paragraph.join(' '))}</p>`);
  }

  return html.join('\n');
}

function stripInlineMarkdown(text) {
  return text
    .replace(/!\[([^\]]*)]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (full, alt, url) => `图片：${alt || '未命名图片'} ${url}`)
    .replace(/\[([^\]]+)]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, '$1 ($2)')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1$2')
    .replace(/(^|[^_])_([^_\n]+)_/g, '$1$2')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<[^>]+>/g, '')
    .trimEnd();
}

export function buildPlainTextDraft(markdown, { fileName, siteUrl = DEFAULT_SITE_URL }) {
  const publicMarkdown = buildMarkdownDraft(markdown, { fileName, siteUrl });
  const lines = publicMarkdown.replace(/\r\n/g, '\n').split('\n');
  const output = [];
  let inFence = false;

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();

    if (isFenceStart(line)) {
      inFence = !inFence;
      continue;
    }

    if (inFence) {
      output.push(line);
      continue;
    }

    if (isTableSeparator(line)) {
      continue;
    }

    if (isHorizontalRule(line)) {
      output.push('');
      continue;
    }

    const heading = line.match(/^#{1,6}\s+(.+?)\s*#*\s*$/);
    if (heading) {
      output.push(stripInlineMarkdown(heading[1]));
      continue;
    }

    const list = line.match(/^\s*(?:[-*+]\s+|\d+\.\s+)(.*)$/);
    if (list) {
      output.push(`- ${stripInlineMarkdown(list[1])}`);
      continue;
    }

    if (/^>\s?/.test(line)) {
      output.push(stripInlineMarkdown(line.replace(/^>\s?/, '')));
      continue;
    }

    if (isPotentialTableRow(line)) {
      output.push(splitTableRow(line).map(stripInlineMarkdown).join(' | '));
      continue;
    }

    output.push(stripInlineMarkdown(line));
  }

  return output.join('\n').replace(/\n{3,}/g, '\n\n').trimEnd() + '\n';
}

function buildRichTextDocument({ title, bodyHtml, plainText }) {
  return `<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>${escapeHtml(title)}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.18.2/dist/katex.min.css">
  </head>
  <body style="margin:0;background:#f9fafb">
    <div style="position:sticky;top:0;z-index:2;background:#ffffff;border-bottom:1px solid #e5e7eb;padding:10px 14px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',Arial,sans-serif">
      <button id="copy-rich-text" style="border:1px solid #2563eb;background:#2563eb;color:#fff;border-radius:6px;padding:7px 12px;cursor:pointer">复制富文本</button>
      <span id="copy-status" style="margin-left:10px;color:#6b7280;font-size:13px">也可以只选中文章区域手动复制</span>
    </div>
    <main id="article" style="${RICH_STYLE.article};background:#ffffff">
${bodyHtml}
    </main>
    <script>
      const button = document.getElementById('copy-rich-text');
      const status = document.getElementById('copy-status');
      button.addEventListener('click', async () => {
        const article = document.getElementById('article');
        const html = article.innerHTML;
        const text = ${JSON.stringify(plainText)};
        try {
          if (window.ClipboardItem) {
            await navigator.clipboard.write([
              new ClipboardItem({
                'text/html': new Blob([html], { type: 'text/html' }),
                'text/plain': new Blob([text], { type: 'text/plain' }),
              }),
            ]);
          } else {
            await navigator.clipboard.writeText(text);
          }
          status.textContent = '已复制';
        } catch (error) {
          status.textContent = '复制失败，请手动选中文章区域复制';
        }
      });
    </script>
  </body>
</html>
`;
}

function chapterSlugFromFileName(fileName) {
  return path.posix.basename(fileName, path.posix.extname(fileName));
}

function titleFromMarkdown(markdown, fallback) {
  const heading = markdown.match(/^#\s+(.+?)\s*#*\s*$/m);
  return heading ? stripInlineMarkdown(heading[1]).trim() : fallback;
}

async function loadChapters(docsDir) {
  try {
    const metadata = JSON.parse(await readFile(path.join(docsDir, 'source.json'), 'utf8'));
    return metadata.chapters.map((chapter) => ({
      ...chapter,
      slug: chapter.slug ?? chapterSlugFromFileName(chapter.fileName),
    }));
  } catch {
    throw new Error(`Missing ${path.join(docsDir, 'source.json')}. Run npm run sync first or restore generated docs metadata.`);
  }
}

function matchesChapter(chapter, selector) {
  if (!selector) {
    return true;
  }

  return chapter.slug === selector
    || chapter.fileName === selector
    || chapter.fileName.endsWith(selector)
    || chapter.title === selector;
}

export function buildPlatformExportReadme({
  title,
  siteUrl = DEFAULT_SITE_URL,
  outputDir = DEFAULT_OUTPUT_DIR,
  chapters = [],
}) {
  const lines = [
    `# ${title} 平台发布包`,
    '',
    `生成来源：${siteUrl}`,
    '',
    '## 推荐用法',
    '',
    '- 微信公众号、知乎：打开 `rich.html`，点击“复制富文本”或手动选中文章区域复制，再粘贴到平台编辑器。',
    '- CSDN：优先使用 `markdown.md`。如果平台 Markdown 渲染不理想，再改用 `rich.html` 复制富文本。',
    '- 小红书、抖音长文、头条号、百家号：使用 `plain.txt`，图片按文中给出的 URL 或本地素材手动上传。',
    '- 图片链接已经改写为 GitHub Pages 公网 URL，避免从本地文件粘贴时丢图。',
    '',
    '## 章节',
    '',
  ];

  for (const chapter of chapters) {
    const base = `${outputDir}/${chapter.slug}`;
    lines.push(`- ${chapter.title}`);
    lines.push(`  - rich: \`${base}/rich.html\``);
    lines.push(`  - markdown: \`${base}/markdown.md\``);
    lines.push(`  - plain: \`${base}/plain.txt\``);
  }

  lines.push('');
  return lines.join('\n');
}

export async function exportPlatformPosts({
  docsDir = DEFAULT_DOCS_DIR,
  outputDir = DEFAULT_OUTPUT_DIR,
  siteUrl = DEFAULT_SITE_URL,
  chapter: chapterSelector,
} = {}) {
  const resolvedOutputDir = path.resolve(outputDir);
  if (resolvedOutputDir === process.cwd()) {
    throw new Error('Refusing to export into the repository root. Choose a subdirectory with --out-dir.');
  }

  const chapters = (await loadChapters(docsDir)).filter((chapter) => matchesChapter(chapter, chapterSelector));

  if (chapters.length === 0) {
    throw new Error(`No chapter matched: ${chapterSelector}`);
  }

  await rm(outputDir, { recursive: true, force: true });
  await mkdir(outputDir, { recursive: true });

  const written = [];
  for (const chapter of chapters) {
    const sourcePath = path.join(docsDir, chapter.fileName);
    const markdown = await readFile(sourcePath, 'utf8');
    const title = titleFromMarkdown(markdown, chapter.title);
    const slug = chapter.slug ?? chapterSlugFromFileName(chapter.fileName);
    const targetDir = path.join(outputDir, slug);
    const context = { fileName: chapter.fileName, siteUrl };
    const markdownDraft = buildMarkdownDraft(markdown, context);
    const plainText = buildPlainTextDraft(markdown, context);
    const richHtml = buildRichTextDocument({
      title,
      bodyHtml: renderMarkdownForRichText(markdown, context),
      plainText,
    });

    await mkdir(targetDir, { recursive: true });
    await writeFile(path.join(targetDir, 'markdown.md'), markdownDraft);
    await writeFile(path.join(targetDir, 'plain.txt'), plainText);
    await writeFile(path.join(targetDir, 'rich.html'), richHtml);
    written.push({ ...chapter, title, slug });
  }

  const sourceMetadata = JSON.parse(await readFile(path.join(docsDir, 'source.json'), 'utf8'));
  await writeFile(path.join(outputDir, 'README.md'), buildPlatformExportReadme({
    title: sourceMetadata.title ?? 'GitHub Pages',
    siteUrl,
    outputDir,
    chapters: written,
  }));

  return { outputDir, chapters: written };
}

export function parseArgs(argv) {
  const options = {};

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === '--docs-dir') {
      options.docsDir = argv[++index];
    } else if (arg === '--out-dir') {
      options.outputDir = argv[++index];
    } else if (arg === '--site-url') {
      options.siteUrl = argv[++index];
    } else if (arg === '--chapter') {
      options.chapter = argv[++index];
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

async function main() {
  const result = await exportPlatformPosts(parseArgs(process.argv.slice(2)));
  console.log(`Exported ${result.chapters.length} platform draft(s) to ${result.outputDir}.`);
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
