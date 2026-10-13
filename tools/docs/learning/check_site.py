"""Check generated document links and assets without network access."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import json
import argparse

class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []
        self.images = []
        self.remote_assets = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag in ('a', 'img', 'script', 'link'):
            field = 'href' if tag in ('a', 'link') else 'src'
            if attrs.get(field):
                self.links.append((tag, attrs[field]))
                if tag in ('img', 'script') and attrs[field].startswith(('https:', 'http:', '//')):
                    self.remote_assets.append(attrs[field])
        if tag == 'img':
            self.images.append(attrs)

arguments = argparse.ArgumentParser(description=__doc__)
arguments.add_argument('root', type=Path)
arguments.add_argument('--allow-remote-asset', action='append', default=[],
                       help='Exact URL of an expected external script or image')
args = arguments.parse_args()
root = args.root.resolve()
pages = {}
for file in root.rglob('*.html'):
    if file.relative_to(root).parts[0] == '_static':
        continue  # Theme source templates are assets, not generated document pages.
    parser = Page()
    parser.feed(file.read_text())
    pages[file] = parser
missing = []
for file, page in pages.items():
    for tag, href in page.links:
        parts = urlsplit(href)
        if parts.scheme or parts.netloc or not parts.path and not parts.fragment:
            continue
        target = ((root / unquote(parts.path).lstrip('/')) if parts.path.startswith('/')
                  else file.parent / unquote(parts.path)).resolve() if parts.path else file
        if target.is_dir():
            target /= 'index.html'
        if not target.exists():
            missing.append([str(file.relative_to(root)), href, 'file'])
        elif parts.fragment and target in pages and unquote(parts.fragment) not in pages[target].ids:
            missing.append([str(file.relative_to(root)), href, 'anchor'])
report = {'html_pages': len(pages), 'missing_local_links': missing,
          'remote_scripts_or_images': sorted({x for p in pages.values() for x in p.remote_assets}),
          'lesson_pages': sum(1 for p in pages if p.relative_to(root).parts[0] in ('learn', 'nvf')),
          'lesson_images_without_alt': [str(p.relative_to(root)) for p,page in pages.items()
                                        if p.relative_to(root).parts[0] in ('learn','nvf')
                                        for img in page.images if not img.get('alt')]}
report['unexpected_remote_assets'] = [url for url in report['remote_scripts_or_images']
                                     if url not in args.allow_remote_asset]
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(bool(missing or report['unexpected_remote_assets'] or report['lesson_images_without_alt']))
