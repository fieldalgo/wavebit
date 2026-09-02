"""Sphinx build, as a trial against the MkDocs one that sits beside it.

Both read the same `docs/` markdown. What differs, and the reason this exists:

* The docstrings cross-reference themselves in Sphinx's own notation -- ``:func:``,
  ``:class:`` -- 760 times. MkDocs cannot interpret those and `mkdocs_hooks.py` strips them;
  here they resolve to links, and `intersphinx` carries them out to numpy, jax and torch.
* `pydata_sphinx_theme` is the design the MkDocs side was reconstructing in 78 lines of CSS.

Run it with `sphinx-build -b html docs docs/_build/html`.
"""
import pathlib
import sys

# The package is not pip-installed in the development environment -- every `import wavebit` in
# this repository works because it is run from the repository root, with the root on sys.path
# implicitly. `sphinx-build` runs with `docs/` as the source dir, so it does not get that.
#
# Resolved from `__file__` rather than from the working directory on purpose: several worktrees
# share this clone, and a cwd-relative path would let a build in one of them document the
# checkout in another -- silently, since the module would import fine and be the wrong copy.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

project = 'Wavebit'
author = 'FieldAlgo'
copyright = '2026, FieldAlgo'
release = '0.1.0'

extensions = [
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

myst_enable_extensions = ['colon_fence', 'deflist', 'attrs_inline']

# A ```mermaid fence becomes the mermaid directive, so the diagrams stay plain fenced blocks
# that GitHub also draws -- rather than `{mermaid}` directives, which only this build reads.
myst_fence_as_directive = ['mermaid']

# Heading anchors for the same reason the MkDocs side uses GitHub's slugifier: `#some-heading`
# links are written once and followed in both readers.
myst_heading_anchors = 3

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'jax': ('https://docs.jax.dev/en/latest/', None),
    'torch': ('https://docs.pytorch.org/docs/stable/', None),
}

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

exclude_patterns = ['_build', 'README.md', 'figures']

html_theme = 'pydata_sphinx_theme'
html_title = 'Wavebit'
html_theme_options = {
    'github_url': 'https://github.com/fieldalgo/wavebit',
    'show_prev_next': True,
    'navbar_align': 'left',
    'show_toc_level': 2,
}
