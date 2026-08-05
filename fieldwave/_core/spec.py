"""A run described as data — ``JobSpec`` — and the stateless entry point that executes it.

:func:`fieldwave.rcwa` builds a *solver object* from Python values: arrays, enums, floats.
That is the right shape for a script, and the wrong shape for anything that has to cross a
process, a queue, or a network. This module adds the other shape:

    JobSpec (plain data)  ──▶  run(spec)  ──▶  result

``JobSpec`` is JSON-round-trippable and fully determines a run, which buys three things a
solver object cannot:

* **A contract.** One schema is shared by every caller — a script, a distributed worker,
  an HTTP client — so they cannot drift apart.
* **Reproducibility.** :meth:`JobSpec.key` hashes the normalised spec, so an identical run
  is identifiable as identical without re-solving it.
* **Safety.** Accepting a declarative spec is not accepting arbitrary Python. A service
  that takes specs is not executing its callers' code.

The physical scene reuses the existing :class:`~fieldwave._core.domain.Stack` /
:class:`~fieldwave._core.domain.Incidence` objects; ``numerics`` holds the discretisation
choices, which are not part of the scene.

Sweeps are declared *inside* the spec, so "501 wavelengths" travels as one small object
rather than 501 near-identical ones. :meth:`JobSpec.expand` turns that back into one
sweep-free spec per grid point:

    >>> for point in spec.expand():        # doctest: +SKIP
    ...     result = run(point)
"""
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Optional
import hashlib
import json

from .domain import Layer, Stack, Incidence
from .enums import Backend, Pol, FourierType

SPEC_VERSION = '1'

#: Incidence fields a sweep may vary. Geometry sweeps wait for the parametric geometry
#: layer — sweeping a raster grid point-by-point is not a thing anyone wants to declare.
SWEEPABLE = ('wavelength', 'theta', 'phi', 'pol', 'psi')


class SpecError(ValueError):
    """A spec that cannot be executed as written."""


# ------------------------------------------------------------------ coercion helpers

def _reject_unknown(got, allowed, where):
    unknown = sorted(set(got) - set(allowed))
    if unknown:
        raise SpecError(
            f"{where}: unknown field(s) {unknown}. Allowed: {sorted(allowed)}")


def _enum(value, enum_cls, where):
    """Accept an enum, its integer value, or its name ('numpy' / 'NUMPY' / 0)."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls[value.upper()]
        except KeyError:
            raise SpecError(
                f"{where}: {value!r} is not a {enum_cls.__name__}. "
                f"Expected one of {[m.name.lower() for m in enum_cls]}") from None
    try:
        return enum_cls(value)
    except ValueError:
        raise SpecError(
            f"{where}: {value!r} is not a {enum_cls.__name__}. "
            f"Expected one of {[m.name.lower() for m in enum_cls]} "
            f"or {[int(m) for m in enum_cls]}") from None


def _jsonable(value):
    """Plain-Python form of a value, so ``to_dict`` output survives ``json.dumps``."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (Backend, Pol, FourierType)):
        return value.name.lower()
    if isinstance(value, (str, int, float)):
        return value
    if hasattr(value, 'tolist'):                     # ndarray / tensor
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    raise SpecError(f"value of type {type(value).__name__} is not JSON-representable")


# ------------------------------------------------------------------ numerics

@dataclass
class Numerics:
    """Discretisation and backend choices — everything that is not the physical scene.

    ``type_complex`` is the bit width (128 / 64) rather than a dtype object, so the spec
    stays backend-neutral; :func:`run` maps it onto the solver's code.
    """
    backend: Backend = Backend.NUMPY
    fto: Any = (0, 0)
    fourier_type: FourierType = FourierType.DFS
    type_complex: int = 128
    connecting_algo: str = 'TMM'
    enhanced_dfs: bool = True
    use_pinv: bool = False
    perturbation: float = 1e-20
    analytic_normal_grid: int = 1024

    _FIELDS = ('backend', 'fto', 'fourier_type', 'type_complex', 'connecting_algo',
               'enhanced_dfs', 'use_pinv', 'perturbation', 'analytic_normal_grid')

    @classmethod
    def from_dict(cls, d) -> 'Numerics':
        d = dict(d or {})
        _reject_unknown(d, cls._FIELDS, 'numerics')
        if 'backend' in d:
            d['backend'] = _enum(d['backend'], Backend, 'numerics.backend')
        if 'fourier_type' in d:
            d['fourier_type'] = _enum(d['fourier_type'], FourierType, 'numerics.fourier_type')
        if 'fto' in d:
            fto = d['fto']
            fto = (fto, 0) if isinstance(fto, int) else tuple(fto)
            if len(fto) != 2 or not all(isinstance(v, int) and v >= 0 for v in fto):
                raise SpecError(
                    f"numerics.fto: expected two non-negative ints (x, y), got {d['fto']!r}")
            d['fto'] = fto
        if d.get('type_complex', 128) not in (128, 64):
            raise SpecError(
                f"numerics.type_complex: expected 128 or 64, got {d['type_complex']!r}")
        return cls(**d)

    def to_dict(self) -> dict:
        return {f: _jsonable(getattr(self, f)) for f in self._FIELDS}

    def as_kwargs(self) -> dict:
        """Solver keyword arguments. ``type_complex`` becomes the solver's 0/1 code."""
        return dict(
            backend=int(self.backend),
            fto=list(self.fto),
            fourier_type=int(self.fourier_type),
            type_complex=0 if self.type_complex == 128 else 1,
            connecting_algo=self.connecting_algo,
            enhanced_dfs=self.enhanced_dfs,
            use_pinv=self.use_pinv,
            perturbation=self.perturbation,
            analytic_normal_grid=self.analytic_normal_grid,
        )


# ------------------------------------------------------------------ sweep

@dataclass
class Sweep:
    """Declared axes to vary, as ``{field: [values]}`` after resolution.

    Each axis is written either as an explicit list or as a ``linspace``; both land here as
    a concrete list. Multiple axes form the full Cartesian product — an outer product, not
    a zip, because that is what a parameter sweep means.
    """
    axes: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d) -> 'Sweep':
        d = dict(d or {})
        _reject_unknown(d, SWEEPABLE, 'sweep')
        axes = {}
        for name, axis in d.items():
            if isinstance(axis, (list, tuple)):       # bare list is shorthand for `values`
                axis = {'values': list(axis)}
            if not isinstance(axis, dict):
                raise SpecError(
                    f"sweep.{name}: expected a list, {{'values': [...]}} or "
                    f"{{'linspace': [start, stop, num]}}, got {axis!r}")
            _reject_unknown(axis, ('values', 'linspace'), f'sweep.{name}')
            if ('values' in axis) == ('linspace' in axis):
                raise SpecError(
                    f"sweep.{name}: give exactly one of 'values' or 'linspace'")
            if 'values' in axis:
                values = list(axis['values'])
            else:
                spec = axis['linspace']
                if len(spec) != 3 or not isinstance(spec[2], int) or spec[2] < 1:
                    raise SpecError(
                        f"sweep.{name}.linspace: expected [start, stop, num] with an "
                        f"integer num >= 1, got {spec!r}")
                start, stop, num = spec
                step = 0.0 if num == 1 else (stop - start) / (num - 1)
                values = [start + i * step for i in range(num)]
            if not values:
                raise SpecError(f"sweep.{name}: no values — an empty axis has no points")
            axes[name] = values
        return cls(axes=axes)

    def to_dict(self) -> dict:
        return {name: {'values': _jsonable(v)} for name, v in self.axes.items()}

    def points(self):
        """One ``{field: value}`` mapping per grid point, in declaration order."""
        if not self.axes:
            return []
        names = list(self.axes)
        return [dict(zip(names, combo)) for combo in product(*(self.axes[n] for n in names))]

    def __len__(self) -> int:
        n = 1
        for values in self.axes.values():
            n *= len(values)
        return n if self.axes else 0

    def __bool__(self) -> bool:
        return bool(self.axes)


# ------------------------------------------------------------------ the spec

@dataclass
class JobSpec:
    """A complete, serialisable description of one run (or one sweep of runs)."""
    stack: Stack = field(default_factory=Stack)
    incidence: Incidence = field(default_factory=Incidence)
    numerics: Numerics = field(default_factory=Numerics)
    sweep: Sweep = field(default_factory=Sweep)
    outputs: tuple = ('de_ri', 'de_ti')
    solver: str = 'rcwa'
    version: str = SPEC_VERSION

    _FIELDS = ('version', 'solver', 'numerics', 'stack', 'incidence', 'sweep', 'outputs')

    # -------------------------------------------------------------- construction

    @classmethod
    def from_dict(cls, d) -> 'JobSpec':
        """Build from a plain mapping, rejecting anything that will not execute.

        Validation is deliberately eager and specific: a spec that reaches a worker is far
        more expensive to debug than one rejected at submission.
        """
        if not isinstance(d, dict):
            raise SpecError(f"spec: expected a mapping, got {type(d).__name__}")
        d = dict(d)
        _reject_unknown(d, cls._FIELDS, 'spec')

        version = d.get('version', SPEC_VERSION)
        if str(version) != SPEC_VERSION:
            raise SpecError(
                f"spec.version: this build understands version {SPEC_VERSION!r}, "
                f"got {version!r}")
        solver = d.get('solver', 'rcwa')
        if solver != 'rcwa':
            raise SpecError(f"spec.solver: only 'rcwa' exists so far, got {solver!r}")

        outputs = tuple(d.get('outputs', ('de_ri', 'de_ti')))
        if not outputs:
            raise SpecError("spec.outputs: no outputs requested — the run would return nothing")

        return cls(
            stack=_stack_from_dict(d.get('stack')),
            incidence=_incidence_from_dict(d.get('incidence')),
            numerics=Numerics.from_dict(d.get('numerics')),
            sweep=Sweep.from_dict(d.get('sweep')),
            outputs=outputs,
            solver=solver,
            version=SPEC_VERSION,
        )

    @classmethod
    def from_json(cls, text) -> 'JobSpec':
        return cls.from_dict(json.loads(text))

    # -------------------------------------------------------------- serialisation

    def to_dict(self) -> dict:
        return {
            'version': self.version,
            'solver': self.solver,
            'numerics': self.numerics.to_dict(),
            'stack': _stack_to_dict(self.stack),
            'incidence': _incidence_to_dict(self.incidence),
            'sweep': self.sweep.to_dict(),
            'outputs': list(self.outputs),
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def key(self) -> str:
        """Content hash of the normalised spec — equal specs give equal keys.

        Sort order and separators are pinned so the digest does not depend on how the
        caller happened to spell the input.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()

    # -------------------------------------------------------------- sweeps

    def expand(self) -> list:
        """One sweep-free :class:`JobSpec` per grid point (``[self]`` if there is no sweep).

        Each point carries the same stack and numerics with the swept incidence fields
        overridden, so every returned spec is independently executable — which is what makes
        a sweep distributable, and a retry cheap.
        """
        if not self.sweep:
            return [self]
        return [replace(self, incidence=replace(self.incidence, **point), sweep=Sweep())
                for point in self.sweep.points()]

    def __len__(self) -> int:
        """Number of solves this spec describes."""
        return len(self.sweep) if self.sweep else 1


# ------------------------------------------------------------------ scene (de)serialisation

def _stack_from_dict(d) -> Stack:
    d = dict(d or {})
    _reject_unknown(d, ('layers', 'n_top', 'n_bot', 'period'), 'stack')
    layers = []
    for i, raw in enumerate(d.get('layers', [])):
        _reject_unknown(raw, ('thickness', 'eps'), f'stack.layers[{i}]')
        if 'thickness' not in raw or 'eps' not in raw:
            raise SpecError(f"stack.layers[{i}]: needs both 'thickness' and 'eps'")
        thickness = raw['thickness']
        if not isinstance(thickness, (int, float)) or thickness < 0:
            raise SpecError(
                f"stack.layers[{i}].thickness: expected a non-negative number, "
                f"got {thickness!r}")
        layers.append(Layer(thickness=thickness, eps=_eps_from_json(raw['eps'], i)))
    period = d.get('period', (1.0, 1.0))
    return Stack(layers=layers, n_top=d.get('n_top', 1.0), n_bot=d.get('n_bot', 1.0),
                 period=list(period) if isinstance(period, (list, tuple)) else period)


def _stack_to_dict(stack: Stack) -> dict:
    return {
        'n_top': _jsonable(stack.n_top),
        'n_bot': _jsonable(stack.n_bot),
        'period': _jsonable(stack.period),
        'layers': [{'thickness': _jsonable(l.thickness), 'eps': _eps_to_json(l.eps)}
                   for l in stack.layers],
    }


def _eps_from_json(eps, index):
    """A layer's permittivity, as written in JSON.

    A nested list is a raster grid and becomes an ndarray. Vector / analytic recipes pass
    through as-is — they are already plain data.

    A raster inlined into a spec is convenient and does not scale: a 512x512 grid is
    megabytes of JSON per layer. Large rasters belong in object storage, referenced rather
    than embedded; that path arrives with the parametric geometry layer.
    """
    if isinstance(eps, dict) and 'raster_ref' in eps:
        raise SpecError(
            f"stack.layers[{index}].eps: 'raster_ref' is not resolvable here — "
            f"this entry point takes self-contained specs. Inline the grid, or resolve "
            f"the reference before calling.")
    if isinstance(eps, list) and eps and isinstance(eps[0], (list, tuple)):
        import numpy as np
        return np.asarray(eps, dtype=float)
    return eps


def _eps_to_json(eps):
    return _jsonable(eps)


def _incidence_from_dict(d) -> Incidence:
    d = dict(d or {})
    _reject_unknown(d, ('wavelength', 'theta', 'phi', 'pol', 'psi'), 'incidence')
    if 'pol' in d:
        d['pol'] = _normalise_pol(d['pol'])
    return Incidence(**d)


def _normalise_pol(pol):
    """``pol`` as a number, whichever way it was written.

    It is the TM *fraction*, not a two-valued flag: ``Pol.TE`` (0) and ``Pol.TM`` (1) are the
    named ends of a continuum, and 0.5 is a legitimate value in between. So the numeric form
    is the general one and the names are sugar — normalising the other way would have to
    reject the interior of the range. Storing one canonical form is what lets ``pol: 'te'``
    and ``pol: 0`` hash to the same key.
    """
    if isinstance(pol, str):
        return int(_enum(pol, Pol, 'incidence.pol'))
    if isinstance(pol, Pol):
        return int(pol)
    if isinstance(pol, bool) or not isinstance(pol, (int, float)):
        raise SpecError(
            f"incidence.pol: expected a number in [0, 1] (the TM fraction) or one of "
            f"{[m.name.lower() for m in Pol]}, got {pol!r}")
    return pol


def _incidence_to_dict(inc: Incidence) -> dict:
    return {'wavelength': _jsonable(inc.wavelength), 'theta': _jsonable(inc.theta),
            'phi': _jsonable(inc.phi), 'pol': _jsonable(inc.pol), 'psi': _jsonable(inc.psi)}


# ------------------------------------------------------------------ execution

def run(spec, geometry=None):
    """Execute one :class:`JobSpec` and return the solver's result.

    ``spec`` may be a :class:`JobSpec`, a mapping, or a JSON string. The call is stateless:
    no solver object outlives it, so the same spec run twice — here, or on two different
    machines — gives the same answer.

    A spec that declares a sweep is refused rather than silently collapsed. Call
    :meth:`JobSpec.expand` and run the points; that is the seam a distributed runner needs,
    and hiding it would only hide it from the caller who most needs it.
    """
    if isinstance(spec, str):
        spec = JobSpec.from_json(spec)
    elif isinstance(spec, dict):
        spec = JobSpec.from_dict(spec)
    elif not isinstance(spec, JobSpec):
        raise SpecError(
            f"run: expected a JobSpec, mapping or JSON string, got {type(spec).__name__}")

    if spec.sweep:
        raise SpecError(
            f"run: this spec declares a sweep of {len(spec)} points. Expand it first — "
            f"`[run(point) for point in spec.expand()]` — so each point is retried, "
            f"distributed and cached on its own.")

    from ..main import rcwa
    solver = rcwa(**spec.numerics.as_kwargs(),
                  **spec.stack.as_kwargs(),
                  **spec.incidence.as_kwargs())
    return solver.conv_solve(geometry=geometry)
