"""Let Sphinx's source analyzer read a ``**``-unpacked dict literal.

``sphinx.pycode`` parses each documented module to find where its classes and functions
start and end -- what ``sphinx.ext.viewcode`` turns into the ``[source]`` pages -- and to
read ``#:`` attribute comments for autodoc. To decide whether a class is ``@final`` it
unparses every decorator, and its unparser drops the ``None`` key of a ``**mapping`` entry
while keeping the value, so ``{**a, **b}`` raises ``ValueError: zip() argument 2 is longer
than argument 1`` (``_UnparseVisitor.visit_Dict``; Sphinx 9.1.1, and the same code on its
main branch). The analyzer then gives up on the whole module. viewcode records it as
unreadable, still lists it in ``_modules/index.html``, and the link is dead.
``wavebit.observations.contracts`` is such a module: ``RawDiffractionSpectrum`` is declared
with ``@spec(renamed={**_RAW_ORDER_RENAMED, **_RAW_FIELDS_RENAMED})``.

This extension renders such an entry as ``**value``, as :func:`ast.unparse` does. It patches
only while the installed Sphinx still fails on the construct, so a Sphinx that has fixed it
is left alone.
"""
import ast

from sphinx.errors import ExtensionError
from sphinx.pycode import ast as pycode_ast


def _unparses_unpacked_dict():
    try:
        pycode_ast.unparse(ast.parse('{**a, "b": 1}', mode='eval').body)
    except Exception:
        return False
    return True


def _visit_dict(self, node):
    items = []
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None:
            items.append('**' + self.visit(value))
        else:
            items.append(self.visit(key) + ': ' + self.visit(value))
    return '{' + ', '.join(items) + '}'


def setup(app):
    if not _unparses_unpacked_dict():
        visitor = getattr(pycode_ast, '_UnparseVisitor', None)
        if visitor is None or not hasattr(visitor, 'visit_Dict'):
            raise ExtensionError('wavebit_pycode: this Sphinx still fails to unparse {**a}, '
                                 'but sphinx.pycode.ast._UnparseVisitor.visit_Dict is not where '
                                 'the patch goes; update or drop docs/_ext/wavebit_pycode.py')
        visitor.visit_Dict = _visit_dict
        if not _unparses_unpacked_dict():
            raise ExtensionError('wavebit_pycode: the patch did not make {**a} unparse')
    return {'version': '1', 'parallel_read_safe': True, 'parallel_write_safe': True}
