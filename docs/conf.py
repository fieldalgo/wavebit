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

exclude_patterns = ['_build', 'README.md', 'figures']

html_theme = 'pydata_sphinx_theme'
html_title = 'Wavebit'
html_theme_options = {
    'github_url': 'https://github.com/fieldalgo/wavebit',
    'show_prev_next': True,
    'navbar_align': 'left',
    'show_toc_level': 2,
}
