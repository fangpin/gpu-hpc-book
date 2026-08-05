import http from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 4193;
const DEFAULT_DOCS_DIR = 'docs';

const TEXT_CONTENT_TYPES = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.md', 'text/markdown; charset=utf-8'],
  ['.txt', 'text/plain; charset=utf-8'],
]);

const BINARY_CONTENT_TYPES = new Map([
  ['.apng', 'image/apng'],
  ['.avif', 'image/avif'],
  ['.gif', 'image/gif'],
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
  ['.png', 'image/png'],
  ['.svg', 'image/svg+xml; charset=utf-8'],
  ['.webp', 'image/webp'],
]);

export class Utf8DocsHandler {
  static extensions_map = Object.fromEntries([...TEXT_CONTENT_TYPES, ...BINARY_CONTENT_TYPES]);
}

function contentTypeFor(filePath) {
  return Utf8DocsHandler.extensions_map[path.extname(filePath).toLowerCase()] ?? 'application/octet-stream';
}

function safePathFromUrl(urlPath, rootDir) {
  const decoded = decodeURIComponent(urlPath.split('?')[0]);
  const relativePath = decoded === '/' ? 'index.html' : decoded.replace(/^\/+/, '');
  const resolved = path.resolve(rootDir, relativePath);
  const root = path.resolve(rootDir);

  if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) {
    return undefined;
  }

  return resolved;
}

export function createDocsServer({ rootDir = DEFAULT_DOCS_DIR } = {}) {
  return http.createServer(async (request, response) => {
    const filePath = safePathFromUrl(request.url ?? '/', rootDir);

    if (!filePath) {
      response.writeHead(403, { 'content-type': 'text/plain; charset=utf-8' });
      response.end('Forbidden\n');
      return;
    }

    try {
      const fileStat = await stat(filePath);

      if (!fileStat.isFile()) {
        throw new Error('Not a file');
      }

      response.writeHead(200, {
        'content-length': fileStat.size,
        'content-type': contentTypeFor(filePath),
      });
      createReadStream(filePath).pipe(response);
    } catch {
      response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
      response.end('Not found\n');
    }
  });
}

function parseArgs(argv) {
  const options = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    rootDir: DEFAULT_DOCS_DIR,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === '--host') {
      options.host = argv[++index];
    } else if (arg === '--port') {
      options.port = Number(argv[++index]);
    } else if (arg === '--directory') {
      options.rootDir = argv[++index];
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

const isEntrypoint = process.argv[1]
  ? import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
  : false;

if (isEntrypoint) {
  const options = parseArgs(process.argv.slice(2));
  const server = createDocsServer({ rootDir: options.rootDir });

  server.listen(options.port, options.host, () => {
    console.log(`Serving ${path.resolve(options.rootDir)} at http://${options.host}:${options.port}/`);
  });
}
