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
import math

from .domain import Layer, Stack, Incidence
from .enums import Backend, Factorization, FourierCoefficients, Pol

SPEC_VERSION = '2'   # '2': result fields renamed de_ri/de_ti -> R/T (2026-08-12)

#: Incidence fields a sweep may vary. Geometry sweeps wait for the parametric geometry
#: layer — sweeping a raster grid point-by-point is not a thing anyone wants to declare.
SWEEPABLE = ('wavelength', 'theta', 'phi', 'psi', 'chi')


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
        # Only the integer-coded enums have a numeric spelling to suggest.
        codes = ('' if not all(isinstance(m.value, int) for m in enum_cls)
                 else f" or {[int(m) for m in enum_cls]}")
        raise SpecError(
            f"{where}: {value!r} is not a {enum_cls.__name__}. "
            f"Expected one of {[m.name.lower() for m in enum_cls]}{codes}") from None


def _jsonable(value):
    """Plain-Python form of a value, so ``to_dict`` output survives ``json.dumps``."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (Backend, Pol, FourierCoefficients, Factorization)):
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
    #: ``None`` = unset, for the same reason as ``factorization`` below: the solver resolves an
    #: unset series against the modeling path (sampled on a raster, integrated on both shape
    #: paths, which is the only series either can form) and refuses one the path cannot form.
    #: Defaulting this to ``FourierCoefficients.SAMPLED`` would hand every spec an explicit
    #: request it never made, so a spec carrying shape layers would be refused for asking for
    #: the sampled series.
    fourier_coefficients: Optional[FourierCoefficients] = None
    #: ``None`` = unset, and it has to stay spellable: the solver resolves an unset
    #: factorization against the modeling path (Li on a raster, NVF on analytic shapes) and
    #: refuses a rule the path cannot express. Defaulting this to ``Factorization.LI`` would
    #: hand every spec an explicit request it never made, so a spec carrying analytic layers
    #: would be refused for asking for Li.
    factorization: Optional[Factorization] = None
    type_complex: int = 128
    connecting_algo: str = 'ETM'
    perturbation: float = 1e-20
    normal_grid: int = 1024

    _FIELDS = ('backend', 'fto', 'fourier_coefficients', 'factorization', 'type_complex',
               'connecting_algo', 'perturbation', 'normal_grid')

    @classmethod
    def from_dict(cls, d) -> 'Numerics':
        d = dict(d or {})
        _reject_unknown(d, cls._FIELDS, 'numerics')
        if 'backend' in d:
            d['backend'] = _enum(d['backend'], Backend, 'numerics.backend')
        if d.get('fourier_coefficients') is not None:
            d['fourier_coefficients'] = _enum(d['fourier_coefficients'], FourierCoefficients,
                                              'numerics.fourier_coefficients')
        if d.get('factorization') is not None:
            d['factorization'] = _enum(d['factorization'], Factorization,
                                       'numerics.factorization')
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
        """Solver keyword arguments. ``type_complex`` becomes the solver's 0/1 code.

        An unset ``fourier_coefficients`` or ``factorization`` is *omitted* rather than filled
        in, so that the solver sees the same "nobody asked" it would from a bare ``rcwa()``
        call and resolves each per modeling path.
        """
        return dict(
            backend=int(self.backend),
            fto=list(self.fto),
            type_complex=0 if self.type_complex == 128 else 1,
            connecting_algo=self.connecting_algo,
            perturbation=self.perturbation,
            normal_grid=self.normal_grid,
            **({} if self.fourier_coefficients is None
               else {'fourier_coefficients': self.fourier_coefficients.value}),
            **({} if self.factorization is None
               else {'factorization': self.factorization.value}),
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
            if name == 'theta':
                # Swept angles reach the solver through expand(), bypassing the incidence
                # check, so the same bound is applied where the values are declared.
                for v in values:
                    _check_polar_angle(v)
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
    outputs: tuple = ('R', 'T')
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

        outputs = tuple(d.get('outputs', ('R', 'T')))
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
    _reject_unknown(d, ('wavelength', 'theta', 'phi', 'tetm', 'psi', 'chi',
                        'incidence_side', 'polarization'),
                    'incidence')
    if 'tetm' in d:
        d['tetm'] = _normalise_tetm(d['tetm'])
    if 'theta' in d:
        _check_polar_angle(d['theta'])
    if d.get('incidence_side', 'top') not in ('top', 'bottom'):
        raise SpecError(
            f"incidence.incidence_side: expected 'top' or 'bottom', "
            f"got {d['incidence_side']!r}")
    if d.get('polarization') is not None:
        d['polarization'] = _polarization_from_json(d['polarization'])
    return Incidence(**d)


def _polarization_from_json(value):
    """The amplitude pair with each entry as a ``complex``, whichever way it was written.

    JSON has no complex type, so a component is written as a real number or as a two-list
    ``[re, im]``; a spec built in-process rather than parsed may also carry Python complex
    values, which pass through. One canonical parsed form is what lets ``(1, -1j)`` and
    ``[[1, 0], [0, -1]]`` hash to the same :meth:`JobSpec.key`.
    """
    if isinstance(value, str) or not hasattr(value, '__iter__'):
        raise SpecError(
            f'incidence.polarization: expected a pair (a_s, a_p), got {value!r}')
    entries = list(value)
    if len(entries) != 2:
        raise SpecError(
            f'incidence.polarization: expected 2 amplitudes (a_s, a_p), '
            f'got {len(entries)}')
    pair = []
    for name, v in zip(('a_s', 'a_p'), entries):
        if isinstance(v, (list, tuple)):
            ok = (len(v) == 2 and all(isinstance(x, (int, float))
                                      and not isinstance(x, bool) for x in v))
            if not ok:
                raise SpecError(
                    f'incidence.polarization: {name} written as a list must be the pair '
                    f'[re, im], got {v!r}')
            pair.append(complex(v[0], v[1]))
        elif isinstance(v, complex) or (isinstance(v, (int, float))
                                        and not isinstance(v, bool)):
            pair.append(complex(v))
        else:
            raise SpecError(
                f'incidence.polarization: {name} must be a number or [re, im], got {v!r}')
    return tuple(pair)


def _check_polar_angle(theta):
    """Reject a polar angle that cannot be one — which in practice means degrees.

    Angles here are **radians**, and nothing in the arithmetic objects to being handed
    degrees: ``theta=10`` is 573 degrees, and the solver returns finite numbers for it that
    conserve no energy. Worse, they look like absorption, so the solver's own physical check
    passes them — it declines to flag ``R + T < 1`` precisely because a lossy stack does that
    legitimately, and it cannot tell the two apart.

    Nothing downstream can catch this, so it is caught here. A polar angle at or beyond
    pi/2 is light arriving from behind the stack, so the bound costs no legitimate use.
    """
    try:
        magnitude = abs(float(theta))
    except (TypeError, ValueError):
        return                                       # arrays and tensors are checked by use
    if magnitude >= math.pi / 2:
        raise SpecError(
            f'incidence.theta: {theta} is not a polar angle — angles are in RADIANS, and '
            f'this is {math.degrees(magnitude):.0f} degrees. Did you mean '
            f'{math.radians(magnitude):.4f}? (Nothing further downstream can catch this: '
            f'the solve returns finite numbers that merely look absorbing.)')


def _normalise_tetm(tetm):
    """``tetm`` as ``'te'`` / ``'tm'`` / ``None``, whichever way it was written.

    The canonical form is the *word*, not a number. The retired ``pol`` normalised the other
    way -- to a TM fraction, because it accepted the interior of ``[0, 1]`` -- and that
    interior was the problem: a fractional ``pol`` was a linear angle in a bad unit, so one
    key meant two different kinds of thing. ``tetm`` is a declaration with exactly two
    values, and a spec that wants an angle in between writes ``psi``.
    """
    if tetm is None:
        return None
    if isinstance(tetm, Pol):
        return tetm.name.lower()
    if isinstance(tetm, str) and tetm.lower() in ('te', 'tm'):
        return tetm.lower()
    raise SpecError(
        f"incidence.tetm: expected 'te', 'tm' or null, got {tetm!r}. It is a declaration "
        f'that the polarization is not a degree of freedom here, not a quantity; for a '
        f'linear angle write incidence.psi.')


def _incidence_to_dict(inc: Incidence) -> dict:
    d = {'wavelength': _jsonable(inc.wavelength), 'theta': _jsonable(inc.theta),
         'phi': _jsonable(inc.phi), 'psi': _jsonable(inc.psi),
         'incidence_side': _jsonable(inc.incidence_side)}
    if inc.chi is not None:
        # Emitted only when named, so a spec that stays on the linear angle keeps its key set.
        d['chi'] = _jsonable(inc.chi)
    if inc.tetm is not None:
        # Emitted only when declared, like `polarization` -- a spec that names no declaration
        # keeps the key set it had.
        d['tetm'] = _normalise_tetm(inc.tetm)
    if inc.polarization is not None:
        # Written as [re, im] pairs — JSON has no complex type — and only when set, so the
        # key() of every pre-existing spec is unchanged. `+ 0.0` flushes the negative zero
        # complex(-1j).real produces, which would otherwise split one run into two keys.
        d['polarization'] = [[v.real + 0.0, v.imag + 0.0]
                             for v in _polarization_from_json(inc.polarization)]
    return d


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
