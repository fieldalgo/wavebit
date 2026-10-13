"""Data-only serialization for measured observations; loading never imports types."""
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import math
from types import MappingProxyType

import numpy as np

from wavebit.errors import ValidationError


def record_error(message, path='record'):
    return ValidationError(message, code='record.format', path=path)


@dataclass(frozen=True)
class RecordedFields(Mapping):
    """Named fields of an archived declaration, without its executable methods."""
    type_name: str
    fields: Mapping

    def __post_init__(self):
        object.__setattr__(self, 'fields', MappingProxyType(dict(self.fields)))

    def __getitem__(self, key):
        return self.fields[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self):
        return len(self.fields)


def _type_name(value):
    cls = type(value)
    return f'{cls.__module__}.{cls.__qualname__}'


def _host_array(value, path):
    # Saving is an explicit host snapshot, outside jit/grad. Never import an
    # optional backend merely to read an archive made with that backend.
    try:
        if type(value).__module__.split('.')[0] == 'torch':
            value = value.detach().resolve_conj().resolve_neg().cpu().numpy()
        array = np.asarray(value)
    except Exception as exc:
        raise record_error('Expected concrete numeric values; save outside jit/grad.', path) from exc
    if array.dtype.kind not in 'biufc' or array.dtype.hasobject:
        raise record_error(f'Unsupported array dtype {array.dtype}; numeric arrays are required.', path)
    return np.array(array, copy=True, order='C')


def encode(value, arrays, path='observations'):
    """Encode public data and owned host arrays, refusing hidden executable state."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {'kind': 'float', 'value': repr(value)}
    if type(value) is complex:
        return {'kind': 'complex', 'real': encode(float(value.real), arrays, path),
                'imag': encode(float(value.imag), arrays, path)}
    if isinstance(value, Enum):
        return {'kind': 'record', 'type': _type_name(value),
                'fields': {'value': encode(value.value, arrays, path+'.value')}}
    if isinstance(value, np.dtype):
        if value.kind not in 'biufc' or value.hasobject:
            raise record_error('Only numeric dtypes can be saved.', path)
        return {'kind': 'dtype', 'value': value.str}
    if isinstance(value, (np.ndarray, np.generic)) or (
            hasattr(value, 'shape') and hasattr(value, 'dtype')):
        array = _host_array(value, path)
        name = f'a{len(arrays):06d}'
        arrays[name] = array
        return {'kind': 'array', 'name': name, 'dtype': array.dtype.str, 'shape': list(array.shape)}
    if is_dataclass(value) and not isinstance(value, type):
        private = [f.name for f in fields(value) if f.name.startswith('_')]
        if private:
            raise record_error('This value retains executable/private state. Save explicitly sampled public data.', path)
        return {'kind': 'record', 'type': _type_name(value),
                'fields': {f.name: encode(getattr(value, f.name), arrays, f'{path}.{f.name}')
                           for f in fields(value)}}
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return {'kind': 'record', 'type': _type_name(value),
                'fields': {name: encode(getattr(value, name), arrays, f'{path}.{name}') for name in value._fields}}
    if isinstance(value, Mapping):
        def portable_key(key):
            return key is None or type(key) in (str, bool, int, float, complex) or (
                type(key) is tuple and all(portable_key(part) for part in key))
        if not all(portable_key(key) for key in value):
            raise record_error('Mapping keys must be Python scalar values or tuples of them.', path)
        return {'kind': 'mapping', 'items': [[encode(k, arrays, path+'.key'), encode(v, arrays, f'{path}[{k!r}]')]
                                            for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {'kind': 'tuple' if isinstance(value, tuple) else 'list',
                'items': [encode(v, arrays, f'{path}[{i}]') for i, v in enumerate(value)]}
    raise record_error(f'{type(value).__name__} is not archive data; sample or convert it explicitly.', path)


def decode(value, arrays, used, path='observations'):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if not isinstance(value, dict):
        raise record_error('Malformed observation value.', path)
    kind = value.get('kind')
    if kind == 'array' and set(value) == {'kind', 'name', 'dtype', 'shape'}:
        name = value['name']
        if not isinstance(name, str) or name not in arrays:
            raise record_error('Missing numeric array.', path)
        array = arrays[name]
        if array.dtype.kind not in 'biufc' or array.dtype.hasobject:
            raise record_error('Only numeric arrays can be loaded.', path)
        shape = value['shape']
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or list(array.shape) != shape or array.dtype.str != value['dtype']):
            raise record_error('Array shape or dtype differs from its declaration.', path)
        used.add(name)
        array = array.copy()
        array.setflags(write=False)
        return array
    if kind == 'float' and set(value) == {'kind', 'value'} and value['value'] in ('nan', 'inf', '-inf'):
        return float(value['value'])
    if kind == 'complex' and set(value) == {'kind', 'real', 'imag'}:
        real = decode(value['real'], arrays, used, path)
        imag = decode(value['imag'], arrays, used, path)
        if type(real) not in (float, int) or type(imag) not in (float, int):
            raise record_error('Invalid complex components.', path)
        return complex(real, imag)
    if kind == 'dtype' and set(value) == {'kind', 'value'}:
        try:
            dtype = np.dtype(value['value'])
        except (TypeError, ValueError) as exc:
            raise record_error('Invalid dtype.', path) from exc
        if dtype.kind not in 'biufc' or dtype.hasobject:
            raise record_error('Only numeric dtypes can be loaded.', path)
        return dtype
    if kind == 'record' and set(value) == {'kind', 'type', 'fields'}:
        if not isinstance(value['type'], str) or not isinstance(value['fields'], dict):
            raise record_error('Invalid data record.', path)
        return RecordedFields(value['type'], {k: decode(v, arrays, used, f'{path}.{k}')
                                             for k, v in value['fields'].items()})
    if kind in ('list', 'tuple') and set(value) == {'kind', 'items'} and isinstance(value['items'], list):
        items = [decode(v, arrays, used, f'{path}[{i}]') for i, v in enumerate(value['items'])]
        return tuple(items) if kind == 'tuple' else items
    if kind == 'mapping' and set(value) == {'kind', 'items'} and isinstance(value['items'], list):
        out = {}
        for pair in value['items']:
            if not isinstance(pair, list) or len(pair) != 2:
                raise record_error('Invalid mapping entry.', path)
            key, entry = (decode(v, arrays, used, path) for v in pair)
            try:
                if key in out:
                    raise record_error('Duplicate mapping key.', path)
                out[key] = entry
            except TypeError as exc:
                raise record_error('Mapping keys must be hashable.', path) from exc
        return MappingProxyType(out)
    raise record_error('Unknown or malformed observation encoding.', path)
