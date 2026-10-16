"""The documentation site.

    sphinx-build -W -b html docs docs/_build/html

`-W` is how CI runs it, and the reason is `docs/reference/api.md`: it names objects by hand,
and autodoc reports one it cannot import as a *warning*. Without `-W` a renamed public symbol
leaves a quietly shorter page and nothing fails.

Two choices here are load-bearing rather than taste, and both serve the same rule -- **a
document in `docs/` reads the same on GitHub as it does on the site**:

* The navigation is in `_toc.yml`, through `sphinx-external-toc`, so no `toctree` directive
  goes into a page. A directive renders here and arrives as its own source text there.
* `myst_fence_as_directive` maps a plain ```mermaid fence, so the diagrams stay the fenced
  blocks GitHub also draws rather than `{mermaid}` directives only this build reads.

The one place the rule is broken on purpose is the card grid on `index.md`, which is the site's
front door and has no GitHub reader -- `docs/README.md` is the folder's face there, and is
excluded from this build.

Sphinx rather than MkDocs because the docstrings cross-reference themselves in Sphinx's own
notation -- ``:func:``, ``:class:``, 760 of them. Under MkDocs those arrived as literal text
and a hook stripped them; here they resolve to links, and `intersphinx` carries them out to
numpy, jax and torch.
"""
from datetime import date
import pathlib
import posixpath
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "_ext"))
import wavebit

# An editable environment must point at the same checkout as these documents.
# Refuse a stale/shared installation rather than documenting another worktree.
expected_package = pathlib.Path(__file__).resolve().parents[1] / 'src/wavebit/__init__.py'
if pathlib.Path(wavebit.__file__).resolve() != expected_package:
    raise RuntimeError('Documentation must import this checkout: install it with '
                       "python -m pip install -e '.[docs]' or set PYTHONPATH to its src directory.")

project = 'Wavebit'
author = 'FieldAlgo'
copyright = f'{date.today().year}, FieldAlgo'
# The version the site shows is the version the package it imports declares; nothing to update at a release.
release = wavebit.__version__
version = release
language = 'en'
locale_dirs = ['locales/']
gettext_compact = False

extensions = [
    'wavebit_i18n',           # full Korean chapters and explicit fallback
    'myst_parser',              # the docs are markdown, and stay markdown
    'sphinx.ext.autodoc',       # the API page, read off the live objects
    'sphinx.ext.napoleon',      # Google-style Args:/Returns: in those docstrings
    'sphinx.ext.intersphinx',   # :class:`numpy.ndarray` becomes a link to numpy's own docs
    'sphinx.ext.viewcode',
    'sphinx_design',            # the card grid on the landing page
    'sphinx_external_toc',      # the nav as data, in _toc.yml
    'sphinxcontrib.mermaid',
]

# The nav lives in `_toc.yml` rather than in `toctree` directives inside the pages. That is the
# whole point of the extension here: a `toctree` block is a directive, and a directive does not
# render on GitHub, so putting the nav in the pages would break the property every document in
# this folder is written to hold -- that it reads the same in both places.
external_toc_path = '_toc.yml'
external_toc_exclude_missing = False

myst_enable_extensions = ['colon_fence', 'deflist', 'attrs_inline', 'dollarmath']

# A ```mermaid fence becomes the mermaid directive, so the diagrams stay plain fenced blocks
# that GitHub also draws -- rather than `{mermaid}` directives, which only this build reads.
myst_fence_as_directive = ['mermaid']

# Heading anchors for the same reason the MkDocs side uses GitHub's slugifier: `#some-heading`
# links are written once and followed in both readers.
myst_heading_anchors = 3

# Every inventory here is fetched at build time, and a host that does not answer is a
# WARNING -- which `-W` turns into a failed build. So an entry has to earn the risk it adds.
#
# `scipy` is deliberately absent, and this note is here so that it is not helpfully restored.
# It earned nothing: the handful of scipy names in this codebase's prose are written as
# literals (``scipy.ndimage.gaussian_filter``), not as roles, so the inventory produced zero
# links -- while `docs.scipy.org` was the one host of the five that did not respond, failing
# the build twice on the day this was set up. An unreachable mapping that also links nothing
# is a build risk with no upside.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://docs.jax.dev/en/latest/', None),
    'torch': ('https://docs.pytorch.org/docs/stable/', None),
}

# Fail a dead host fast rather than hanging the build on a connect that never returns --
# the scipy timeout above had no bound at all.
intersphinx_timeout = 15

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

exclude_patterns = ['_build', '_templates', 'README.md', 'figures', 'locales', '_ext', '**/learning-provenance', '**/learning-records']
templates_path = ['_templates']

html_theme = 'pydata_sphinx_theme'
html_title = 'Wavebit'
html_favicon = '_static/wavebit-favicon.ico'
# `../assets/branding` puts the Wavebit logo files at _static/ for the navbar logo below.
html_static_path = ['_static', '../assets/branding']
# wavebit-site.css last: it aligns the docs with wavebit.fieldalgo.com (colours, faces, header,
# footer) and overrides learning.css where both speak.
html_css_files = ['learning/learning.css', 'wavebit-site.css', 'header-controls.css']
html_js_files = [(f'learning/{name}', {'defer': 'defer'}) for name in (
    'learn-interactions.js', 'nvf-interactions.js', 'reader.js', 'light-section-links.js')]
html_js_files.append(('wavebit-site.js', {'defer': 'defer'}))

html_theme_options = {
    # The product site's logo, linking to the product site, as its header does; the docs'
    # own home stays one click away in the breadcrumbs.
    'logo': {
        'image_light': '_static/wavebit_logo.svg',
        'image_dark': '_static/wavebit_logo_dark.svg',
        'alt_text': 'Wavebit',
        'link': 'https://wavebit.fieldalgo.com/',
    },
    'github_url': 'https://github.com/fieldalgo/wavebit',
    'show_prev_next': True,
    'navbar_align': 'left',
    'navbar_end': ['theme-switcher', 'version-switcher.html', 'navbar-icon-links', 'company-link.html'],
    'navbar_persistent': ['search-button.html', 'language-switcher.html'],
    'show_toc_level': 2,
}


def _language_links(app, pagename, templatename, context, doctree):
    """Link matching HTML pages without assuming the site's hosting prefix."""
    context['wavebit_language_links'] = []
    if not app.config.wavebit_bilingual or app.builder.name != 'html':
        return
    language = app.config.language
    if language not in ('en', 'ko'):
        raise ValueError('The bilingual site supports language=en or language=ko.')
    page = app.builder.get_target_uri(pagename)
    current = posixpath.join('ko' if language == 'ko' else '', page)
    directory = posixpath.dirname(current) or '.'
    for code, label in (('en', 'English'), ('ko', '한국어')):
        target = posixpath.join('ko' if code == 'ko' else '', page)
        context['wavebit_language_links'].append({
            'code': code, 'label': label,
            'href': posixpath.relpath(target, directory), 'active': code == language,
        })


def setup(app):
    # Opt in only when both language trees will be present in the same artifact.
    app.add_config_value('wavebit_bilingual', False, 'html', types=[bool])
    app.connect('html-page-context', _language_links)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
