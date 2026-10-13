"""Lazy dataclass pytree registration with numerical metadata kept as children.

No backend is imported here. Strings and structural keys are static; arrays and
numeric values are children even when nested in reference metadata mappings.
"""
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
from enum import Enum
from numbers import Number
from types import MappingProxyType
import sys

_REGISTERED = set()
_STATIC_FIELDS = {}


def _pack(value, leaves):
    if value is None or isinstance(value, (str, bytes, Enum)):
        return ('static', value)
    if isinstance(value, Number) or hasattr(value, 'dtype'):
        leaves.append(value)
        return ('leaf', len(leaves)-1)
    if isinstance(value, Mapping):
        return ('mapping', tuple(value.keys()), tuple(_pack(v, leaves) for v in value.values()))
    if isinstance(value, (tuple, list)):
        return ('sequence', type(value), tuple(_pack(v, leaves) for v in value))
    if is_dataclass(value):
        names = tuple(f.name for f in fields(value))
        static = _STATIC_FIELDS.get(type(value), ())
        return ('dataclass', type(value), names, tuple(('static', getattr(value,n)) if n in static else _pack(getattr(value,n), leaves) for n in names))
    # A registered backend-aware object may define its own tree representation.
    leaves.append(value)
    return ('leaf', len(leaves)-1)


def _unpack(spec, leaves):
    kind = spec[0]
    if kind == 'static': return spec[1]
    if kind == 'leaf': return leaves[spec[1]]
    if kind == 'mapping': return MappingProxyType(dict(zip(spec[1], (_unpack(v, leaves) for v in spec[2]))))
    if kind == 'sequence': return spec[1](_unpack(v, leaves) for v in spec[2])
    if kind == 'dataclass':
        obj = object.__new__(spec[1])
        for name, val in zip(spec[2], spec[3]): object.__setattr__(obj, name, _unpack(val, leaves))
        return obj
    raise TypeError(f'unknown tree specification {kind}')


def register_dataclass(cls, *, static_fields=()):
    """Register an immutable dataclass if JAX is already present.

    Static fields must be hashable and contain no live arrays. Unflatten bypasses
    constructor validation, as JAX legitimately uses abstract placeholders there.
    """
    _STATIC_FIELDS[cls] = tuple(static_fields)
    jax = sys.modules.get('jax')
    tree = getattr(jax, 'tree_util', None)
    if tree is None or cls in _REGISTERED: return False
    names = tuple(f.name for f in fields(cls))
    def flatten(obj):
        leaves=[]
        specs=tuple(('static',getattr(obj,n)) if n in static_fields else _pack(getattr(obj,n),leaves) for n in names)
        return tuple(leaves), specs
    def unflatten(specs, leaves):
        obj=object.__new__(cls)
        for name,spec in zip(names,specs): object.__setattr__(obj,name,_unpack(spec,leaves))
        return obj
    tree.register_pytree_node(cls,flatten,unflatten)
    _REGISTERED.add(cls)
    return True
