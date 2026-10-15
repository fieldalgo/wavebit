"""Portable, versioned jobs with explicit solver dispatch.

Legacy RCWA JobSpec files are handled by wavebit.compat.legacy; changing a schema
number is not a coordinate or material migration.
"""
import base64
import dataclasses
import hashlib
import importlib
import json
import math
from collections.abc import Mapping

import numpy as np

from ._common.declarations import spec
from ._common.execution import Execution
from .errors import ValidationError, UnsupportedFeatureError

_SCHEMA = 'wavebit.job/v1'

# Explicit allowlist: deserializing a job never imports a class named by its input.
_CLASS_MODULES = {
    'wavebit.units': ('Units',),
    'wavebit.lattice': ('Lattice',),
    'wavebit.materials': ('Material', 'ConductiveMaterial', 'DispersiveMaterial',
                         'TabulatedMaterial', 'Drude', 'Lorentz', 'Debye'),
    'wavebit.rcwa': ('Problem', 'Stack', 'Layer', 'Region', 'Rectangle', 'Circle',
                     'Ellipse', 'Polygon', 'Excitation', 'Outputs', 'Numerics',
                     'PlaneWave', 'OrderExcitation', 'GaussianBeam', 'ElectricDipole'),
    'wavebit.fdtd': ('Problem', 'Scene', 'Domain', 'Structure', 'Slab', 'Box', 'Sphere',
                     'Cylinder', 'Prism', 'Plane', 'Region', 'Boundaries', 'Periodic', 'Bloch',
                     'Open', 'PEC', 'PMC', 'Symmetry', 'Axis1D', 'PlaneWave', 'Dipole', 'TFSF', 'InitialField',
                     'GaussianPulse', 'Outputs', 'Record', 'Field', 'Flux',
                     'Diffraction', 'FarField', 'Numerics', 'UniformGrid', 'Duration', 'PML', 'Absorber', 'Decay'),
    'wavebit._common.execution': ('Execution',),
}


def _registry():
    out = {}
    for module, names in _CLASS_MODULES.items():
        namespace = importlib.import_module(module)
        for name in names:
            cls = getattr(namespace, name, None)
            if cls is not None and dataclasses.is_dataclass(cls):
                out[f'{module}.{name}'] = cls
    return out


def _error(message, path):
    return ValidationError(message, code='job.encoding', path=path)


def _encode(value, registry, path):
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _error('Portable scalar values must be finite.', path)
        return value
    if isinstance(value, complex):
        return {'type': 'complex', 'real': _encode(float(value.real), registry, path),
                'imag': _encode(float(value.imag), registry, path)}
    if isinstance(value, np.generic):
        return _encode(np.asarray(value), registry, path)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.kind not in 'biufc':
            raise _error('Only numeric or boolean arrays are portable job data.', path)
        a = np.ascontiguousarray(value)
        return {'type': 'array', 'dtype': value.dtype.str, 'shape': list(value.shape),
                'data': base64.b64encode(a.tobytes()).decode('ascii')}
    from .materials import TabulatedMaterial
    if isinstance(value, TabulatedMaterial):
        return {'type': 'table', 'wavelengths': _encode(value.wavelengths, registry, path+'.wavelengths'),
                'values': _encode(value.values, registry, path+'.values'),
                'mu': _encode(value.mu, registry, path+'.mu'),
                'wavelength_unit': value.wavelength_unit, 'interpolation': value.interpolation}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        tag = next((tag for tag, cls in registry.items() if type(value) is cls), None)
        if tag is None:
            raise _error(f'{type(value).__name__} is not a portable public job declaration.', path)
        return {'type': tag, 'fields': {f.name: _encode(getattr(value, f.name), registry, f'{path}.{f.name}')
                                      for f in dataclasses.fields(value) if f.init and not f.name.startswith('_')}}
    if isinstance(value, tuple):
        return {'type': 'tuple', 'items': [_encode(v, registry, f'{path}[{i}]') for i, v in enumerate(value)]}
    if isinstance(value, list):
        return {'type': 'list', 'items': [_encode(v, registry, f'{path}[{i}]') for i, v in enumerate(value)]}
    if isinstance(value, Mapping):
        pairs = [(_encode(k, registry, path), _encode(v, registry, f'{path}[{k!r}]')) for k, v in value.items()]
        pairs.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, allow_nan=False))
        return {'type': 'mapping', 'items': pairs}
    if callable(value):
        raise _error('Callables cannot be saved as jobs; first construct a concrete Problem.', path)
    raise _error(f'{type(value).__name__} is not portable job data. Convert concrete device arrays to explicit host arrays before saving.', path)


def _decode(value, registry, path):
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _error('Nonfinite scalar in saved job.', path)
        return value
    if not isinstance(value, dict) or 'type' not in value:
        raise _error('Expected a tagged job value.', path)
    tag = value['type']
    if not isinstance(tag, str):
        raise _error('A job type tag must be a string.', path)
    if tag == 'table' and set(value) == {'type', 'wavelengths', 'values', 'mu', 'wavelength_unit', 'interpolation'}:
        from .materials import TabulatedMaterial
        if value['interpolation'] not in ('index', 'epsilon'):
            raise _error('Unknown table interpolation quantity.', path)
        constructor = TabulatedMaterial.from_refractive_index if value['interpolation'] == 'index' else TabulatedMaterial
        return constructor(_decode(value['wavelengths'], registry, path+'.wavelengths'),
                           _decode(value['values'], registry, path+'.values'),
                           mu=_decode(value['mu'], registry, path+'.mu'), wavelength_unit=value['wavelength_unit'])
    if tag == 'complex' and set(value) == {'type', 'real', 'imag'}:
        real, imag = value['real'], value['imag']
        if type(real) not in (int, float) or type(imag) not in (int, float) or not math.isfinite(real) or not math.isfinite(imag):
            raise _error('Complex parts must be finite real numbers.', path)
        return complex(real, imag)
    if tag == 'array' and set(value) == {'type', 'dtype', 'shape', 'data'}:
        try:
            dtype = np.dtype(value['dtype'])
            shape = value['shape']
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError('invalid shape')
            if dtype.hasobject or dtype.kind not in 'biufc':
                raise ValueError('non-numeric dtype')
            raw = base64.b64decode(value['data'], validate=True)
            if math.prod(shape) * dtype.itemsize != len(raw):
                raise ValueError('shape and byte count disagree')
            return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        except (TypeError, ValueError, OverflowError) as exc:
            raise _error(f'Invalid array encoding: {exc}', path) from exc
    if tag in ('tuple', 'list', 'mapping') and set(value) == {'type', 'items'}:
        if not isinstance(value['items'], list):
            raise _error('items must be a list.', path)
        if tag == 'mapping':
            result = {}
            for i, item in enumerate(value['items']):
                if not isinstance(item, list) or len(item) != 2:
                    raise _error('Each mapping item must be a key/value pair.', path)
                key, entry = (_decode(v, registry, f'{path}[{i}]') for v in item)
                try:
                    if key in result:
                        raise _error('Duplicate mapping key.', path)
                    result[key] = entry
                except TypeError as exc:
                    raise _error('Mapping keys must be immutable hashable values.', path) from exc
            return result
        items = [_decode(v, registry, f'{path}[{i}]') for i, v in enumerate(value['items'])]
        return tuple(items) if tag == 'tuple' else items
    if tag in registry and set(value) == {'type', 'fields'} and isinstance(value['fields'], dict):
        cls = registry[tag]
        known = {f.name for f in dataclasses.fields(cls) if f.init and not f.name.startswith('_')}
        extra = set(value['fields']) - known
        if extra:
            raise _error(f'Unknown declaration fields: {sorted(extra)}.', path)
        kwargs = {k: _decode(v, registry, f'{path}.{k}') for k, v in value['fields'].items()}
        try:
            return cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise _error(f'Invalid {tag}: {exc}', path) from exc
    raise _error(f'Unknown or malformed job type {tag!r}.', path)


@spec
class Job:
    """A solver-tagged physical problem, numerical policy, and execution settings."""
    solver: str
    problem: object
    numerics: object
    execution: Execution = dataclasses.field(default_factory=Execution)

    def __post_init__(self):
        if self.solver not in ('rcwa', 'fdtd'):
            raise _error("solver must be 'rcwa' or 'fdtd'.", 'solver')
        namespace = importlib.import_module(f'wavebit.{self.solver}')
        if not isinstance(self.problem, namespace.Problem) or not isinstance(self.numerics, namespace.Numerics):
            raise _error('Problem and Numerics must belong to the selected solver.', 'problem')
        if not isinstance(self.execution, Execution):
            raise _error('execution must be Execution.', 'execution')

    def to_json(self, *, indent=None):
        registry = _registry()
        body = {'schema': _SCHEMA, 'solver': self.solver,
                'problem': _encode(self.problem, registry, 'problem'),
                'numerics': _encode(self.numerics, registry, 'numerics'),
                'execution': _encode(self.execution, registry, 'execution')}
        return json.dumps(body, sort_keys=True, indent=indent, allow_nan=False,
                          separators=(',', ':') if indent is None else None)

    @classmethod
    def from_json(cls, text):
        try:
            body = json.loads(text, parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)))
        except (TypeError, ValueError) as exc:
            raise _error(f'Invalid JSON: {exc}', 'job') from exc
        if not isinstance(body, dict) or body.get('schema') != _SCHEMA:
            raise UnsupportedFeatureError('Expected wavebit.job/v1. Legacy JobSpec requires explicit migration; do not edit its version number.',
                                          code='job.schema', path='schema')
        if set(body) != {'schema', 'solver', 'problem', 'numerics', 'execution'}:
            raise _error('The job envelope contains missing or unknown fields.', 'job')
        registry = _registry()
        return cls(solver=body['solver'], problem=_decode(body['problem'], registry, 'problem'),
                   numerics=_decode(body['numerics'], registry, 'numerics'),
                   execution=_decode(body['execution'], registry, 'execution'))

    @property
    def key(self):
        """SHA-256 of the complete canonical portable envelope."""
        return hashlib.sha256(self.to_json().encode('utf-8')).hexdigest()


from ._common.backend import eager_under_torch_compile


@eager_under_torch_compile
def run(job):
    """Execute one explicit Job; sweeps must expand into separate Jobs first."""
    if not isinstance(job, Job):
        raise _error('wavebit.run accepts a Job; use rcwa.run/fdtd.run for a Problem.', 'job')
    namespace = importlib.import_module(f'wavebit.{job.solver}')
    return namespace.run(job.problem, numerics=job.numerics, backend=job.execution.backend,
                         device=job.execution.device, precision=job.execution.precision)
