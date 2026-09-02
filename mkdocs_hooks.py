"""Build hooks for the documentation site.

One hook, and it exists because the docstrings were written for a reader of the source.

The codebase cross-references itself in Sphinx's notation -- ``:func:`solve_conv```,
``:class:`~wavebit.Material``` -- which reads fine in a terminal and in an editor's tooltip.
mkdocstrings does not interpret those roles, so each one reaches the page as the literal text
``:func:`` in front of a code span: 132 of them on ``docs/reference/api.md`` alone.

Rewriting the docstrings instead was the alternative and is the worse trade. The roles are
correct for the reader they were written for, they are used 760 times across the package, and
the site is a second consumer of prose whose first consumer is fine -- so the second consumer
adapts. Nothing here edits a source file.
"""
import re

# The role, and the code span that Python-Markdown has already made out of its backticks.
# Anchored on the ``<code`` so that a colon-delimited word in ordinary prose is left alone --
# what is being removed is only ever a role attached to a reference.
_ROLE = re.compile(r':(?:class|func|attr|meth|mod|data|exc|obj):\s*(?=<code)')

# Sphinx's "show the last component only" prefix, which survives into the span's text.
# Whitespace is allowed inside the path because a long reference wraps in the source, and the
# line break comes through the span -- one of them does exactly that today.
_TILDE = re.compile(r'(<code[^>]*>)~([\w.\s]+?)(</code>)')


def _shorten(match):
    open_tag, dotted, close_tag = match.groups()
    dotted = ''.join(dotted.split())
    return f'{open_tag}{dotted.rsplit(".", 1)[-1]}{close_tag}'


def on_page_content(html, page, config, files):
    """Strip the Sphinx roles, and honour the ``~`` that goes with them."""
    return _TILDE.sub(_shorten, _ROLE.sub('', html))
