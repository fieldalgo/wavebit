"""Full-page Korean source overlays with explicit English fallback.

The canonical document names are shared by both builds. Overlay files and source
assets are excluded from source discovery, never published as competing pages.
"""
from __future__ import annotations

from pathlib import Path
import posixpath
import re


NAV_LABELS = {
    'getting-started/index': '시작하기', 'learn/index': '배우기',
    'guide/index': '사용자 가이드', 'reference/index': 'API 레퍼런스',
    'theory/index': '이론과 검증', 'development/index': '개발자 가이드',
}


def _after_title(text, note):
    match = re.search(r'^# .+\n', text, re.MULTILINE)
    if match:
        return text[:match.end()] + '\n' + note + '\n' + text[match.end():]
    return text


def _source_read(app, docname, source):
    if app.config.language != 'ko':
        return
    root = Path(app.srcdir)
    overlay = root / 'locales/ko/pages' / (docname + '.md')
    if overlay.is_file():
        source[0] = overlay.read_text(encoding='utf-8')
        app.env.note_dependency(str(overlay))
    elif not (root / 'locales/ko/LC_MESSAGES' / (docname + '.po')).is_file():
        source[0] = _after_title(source[0], ':::{note}\n이 페이지는 아직 한국어로 번역되지 않아 영어 원문을 표시합니다. 언어 메뉴에서 같은 페이지의 영어 문서로 이동할 수 있습니다.\n:::')
    if docname in NAV_LABELS:
        source[0] = re.sub(r'^# .+$', '# ' + NAV_LABELS[docname], source[0], count=1, flags=re.MULTILINE)
    # MyST figure/image directives refer to the canonical source paths. A Korean
    # SVG can use translated labels without modifying the English source or data.
    def localized(match):
        original = match.group(2)
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(docname), original))
        alternate = root / 'locales/ko/assets' / resolved
        if alternate.is_file():
            app.env.note_dependency(str(alternate))
            return match.group(1) + posixpath.relpath(alternate.relative_to(root).as_posix(), posixpath.dirname(docname) or '.')
        return match.group(0)
    source[0] = re.sub(r'((?:`{3,}|:{3,})\{(?:figure|image)\}\s+)([^\s]+)', localized, source[0])
    source[0] = re.sub(r'(!\[[^\]]*\]\()([^\s)]+)', localized, source[0])


def _page_context(app, pagename, templatename, context, doctree):
    root = Path(app.srcdir)
    overlay = root / 'locales/ko/pages' / (pagename + '.md')
    context['wavebit_translation_kind'] = ('korean-source' if overlay.is_file() else 'gettext-or-english-fallback')


def _configure_assets(app, config):
    if config.language == 'ko':
        config.html_js_files = [
            (name.replace('learning/', 'learning/ko/') if name in (
                'learning/learn-interactions.js', 'learning/nvf-interactions.js',
                'learning/reader.js') else name, attributes)
            for name, attributes in config.html_js_files]


def _localize_navigation(app, doctree):
    if app.config.language != 'ko':
        return
    from sphinx import addnodes
    for tree in doctree.findall(addnodes.toctree):
        tree['entries'] = [(NAV_LABELS.get(target, title), target) for title, target in tree['entries']]


def setup(app):
    # Run before the external toctree injects its navigation into the source.
    app.connect('config-inited', _configure_assets)
    app.connect('source-read', _source_read, priority=100)
    app.connect('html-page-context', _page_context)
    app.connect('doctree-read', _localize_navigation)
    return {'version': '1', 'parallel_read_safe': True, 'parallel_write_safe': True}
