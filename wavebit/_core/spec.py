"""A run described as data — ``JobSpec`` — and the stateless entry point that executes it.

:func:`wavebit.rcwa` builds a *solver object* from Python values: arrays, enums, floats.
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

The physical scene reuses the existing :class:`~wavebit._core.domain.Stack` /
:class:`~wavebit._core.domain.Incidence` objects; ``numerics`` holds the discretisation
choices, which are not part of the scene.

**Every material number in a spec is a relative permittivity** -- ``stack.top``, ``stack.bot``,
each layer's ``structure``, and the material slots inside a shape recipe. That is what
``SPEC_VERSION`` 3 marks, and it is why a version 2 document is refused rather than read: the
keys moved (``eps`` -> ``structure``, ``n_top`` / ``n_bot`` -> ``top`` / ``bot``) but so did
the quantity, and a v2 spec whose keys are renamed and whose numbers are kept describes a
different structure that solves and conserves energy. See ``docs/conventions.md``.

Two consequences of that, both load-bearing:

* **A parsed document must state its version.** :meth:`JobSpec.from_dict` refuses one that
  does not, rather than reading the absence as "current" -- see the refusal itself for why
  defaulting made the gate a no-op for exactly the documents most likely to be stale. Building
  a spec in Python is the unceremonious path: ``JobSpec(...)`` is versioned by the build it is
  constructed in, because a spec constructed against this module's classes cannot be stale.
* **A material has a JSON form**, ``{"epsilon": e}`` (plus ``"mu"`` when one is named), so a
  :class:`~wavebit._core.domain.Material` survives ``to_dict`` / ``from_dict`` instead of
  making the object that carries it unserialisable. A bare number is still the scalar
  shorthand and stays a bare number, since a spec that respelled it would change its
  :meth:`JobSpec.key`.

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

from .conventions import parse_order_pair
from .domain import Layer, Material, Stack, Incidence
from .enums import (Answers, Backend, Factorization, FourierCoefficients, NormalMethod,
                    Tetm, answers_of)

SPEC_VERSION = '6'   # '2': result fields renamed de_ri/de_ti -> R/T (2026-08-12)
#                     '3': the material convention. `stack.layers[i].eps` -> `.structure`,
#                     `stack.n_top` / `.n_bot` -> `.top` / `.bot`, and the *quantity* every
#                     one of them holds went from refractive index to relative permittivity
#                     (2026-08-20). A version bump rather than a compatible read: a v2 spec
#                     re-keyed to v3 spellings and left at its old numbers is a legal,
#                     energy-conserving, different structure, so it must be refused rather
#                     than migrated by guess.
#                     '4': the pre-release rename set (2026-08-20). `numerics.fto` ->
#                     `numerics.truncation_order`, `.type_complex` -> `.complex_dtype`,
#                     `.connecting_algo` -> `.layer_algorithm`; and `spec.outputs` is gone
#                     rather than renamed. Unlike '3' none of this can be got wrong quietly
#                     -- an unknown key is refused by `_reject_unknown` and a missing one
#                     takes the default -- but the document's shape changed, so the version
#                     says so rather than leaving two spellings both looking current.
#                     '5': `numerics.max_order` -> `numerics.truncation_order` (2026-08-22).
#                     The name stopped being true when `incidence.excitation` arrived: the
#                     retained window widens around the lit orders, so the field is how many
#                     orders each lit one keeps on either side and not the highest order kept.
#                     Same shape and same values -- a v4 document re-keyed is a correct v5
#                     one -- so unlike '3' this is a spelling change, and the version says so
#                     because `_reject_unknown` would otherwise report the old key as simply
#                     unknown rather than as renamed.
#                     '6': `stack.lattice`, the cell basis (2026-08-23). Additive and
#                     optional -- a v5 document is a correct v6 one -- but the bump is not
#                     ceremony: it runs the other way. A v6 document carrying a lattice,
#                     read by a build that predates the key, loses the skew and solves the
#                     rectangle `period` spells, quietly and conserving energy. The version
#                     is what makes that old build refuse instead.

#: Refusal for a document that states no version. Its own justification, because the rule
#: reads as gratuitous ceremony until you see which case it is for.
_MISSING_VERSION = (
    f"spec.version: missing. A parsed document has to say which schema it is written in, and "
    f"absence is not a statement that it is the current one -- it is the absence of a "
    f"statement. This used to default to {SPEC_VERSION!r}, which made the version gate a "
    f"no-op for exactly the documents most likely to be stale: version 3 renamed the material "
    f"keys AND changed the quantity they hold from refractive index to relative "
    f"permittivity, so a pre-3 document whose keys were mechanically renamed and whose "
    f"numbers were left alone passes every other check in this module. It has valid keys, it "
    f"solves, it conserves energy, and it describes a different structure (n = 3.5 is "
    f"eps = 12.25). The version is the only field that can tell the two apart, so it has to "
    f"be written rather than assumed. Add 'version': {SPEC_VERSION!r} once you have checked "
    f"the numbers are permittivities -- wavebit.eps_from_index converts -- or build the spec "
    f"in Python instead: JobSpec(...) takes the version of the build that constructs it, "
    f"which is the one case where absence really does mean current.")

#: Incidence fields a sweep may vary. Geometry sweeps wait for the parametric geometry
#: layer — sweeping a raster grid point-by-point is not a thing anyone wants to declare.
SWEEPABLE = ('wavelength', 'theta', 'phi', 'psi', 'chi')


class SpecError(ValueError):
    """A spec that cannot be executed as written."""


# ------------------------------------------------------------------ coercion helpers

def _reject_unknown(got, allowed, where):
    """Refuse fields this schema does not read.

    Took a ``retired`` set until 2026-08-26, which named the pre-rename material spellings and
    said that the *quantity* had changed with each. Gone with the rest of that machinery: the
    reader it addressed cannot exist before a first release, and the renames belong in the
    release notes. A retired spelling is now reported as what it is here -- an unknown field.
    """
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
    """Plain-Python form of a value, so ``to_dict`` output survives ``json.dumps``.

    A :class:`~wavebit._core.domain.Material` is handled here rather than only at the two
    half-space slots because it can sit at any depth -- an analytic layer's background, an
    analytic shape's ``material`` -- and its JSON form is *self-identifying* (a dict with an
    ``epsilon`` key), so it can be recognised again on the way back in without the reader
    having to know which position in which recipe it came from. That is exactly the property a
    bare ``complex`` does not have, which is why complex is spelled out only where the slot is
    named; see :func:`_material_to_json`.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (Answers, Backend, Tetm, FourierCoefficients, Factorization)):
        return value.name.lower()
    if isinstance(value, Material):
        return _material_to_json(value)
    if isinstance(value, (str, int, float)):
        return value
    if hasattr(value, 'tolist'):                     # ndarray / tensor
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    raise SpecError(f"value of type {type(value).__name__} is not JSON-representable")


# ------------------------------------------------------------------ materials

def _material_to_json(value):
    """A material slot's value as JSON: ``{'epsilon': e}``, a bare number, or ``[re, im]``.

    Three spellings because a material slot holds three different kinds of thing, and the
    spec's job is to write each one back as itself:

    * a :class:`~wavebit._core.domain.Material` becomes an **object**, and carries ``mu`` as a
      key only when one was named. ``Material(4.0)`` and ``Material(4.0, mu=1.0)`` are
      different materials -- the second requests the permeability path -- so the key has to be
      present exactly when :attr:`~wavebit._core.domain.Material.is_magnetic` is true, or the
      round trip would quietly move a scene from one to the other.
    * a bare number stays a bare number. It is the scalar shorthand, and respelling it as
      ``{'epsilon': x}`` on the way out would change :meth:`JobSpec.key` for every spec written
      before materials had a JSON form -- and would also claim the author said "a medium" where
      they said "a number", a distinction :func:`~wavebit._core.domain.as_material` keeps.
    * a bare ``complex`` becomes ``[re, im]``, the same pair spelling
      :func:`_polarization_from_json` uses, because JSON has no complex type and a lossy
      half-space is not exotic.

    The last one only works where the reader knows it is looking at a material -- ``stack.top``
    and ``stack.bot``, and the ``epsilon`` / ``mu`` entries of the object form. Inside a shape
    recipe a two-list is not distinguishable from a recipe fragment, so a bare
    complex there is still refused by :func:`_jsonable` rather than written in a spelling
    nothing could read back. Wrapping it in a ``Material`` is the way to put one anywhere.
    """
    if isinstance(value, Material):
        out = {'epsilon': _scalar_to_json(value.epsilon)}
        if value.is_magnetic:
            out['mu'] = _scalar_to_json(value.mu)
        return out
    return _scalar_to_json(value)


def _scalar_to_json(value):
    """One material number: a real stays itself, a complex becomes ``[re, im]``.

    ``+ 0.0`` flushes the negative zero that ``complex(2.25, -0.0)`` and the arithmetic behind
    it produce, for the reason :func:`_incidence_to_dict` gives about ``polarization``: an
    imaginary part of ``-0.0`` serialises to ``-0.0``, hashes differently from ``0.0``, and
    would split one run into two :meth:`JobSpec.key` values.
    """
    if isinstance(value, complex):
        return [value.real + 0.0, value.imag + 0.0]
    return _jsonable(value)


def _material_from_json(value, where):
    """The inverse of :func:`_material_to_json`, for a slot known to hold a material.

    Parsing is deliberately narrow. A dict is the object form and its keys are checked, so
    ``{'epsilon': 4.0, 'my': 1.0}`` is a refusal rather than a silently non-magnetic medium --
    the same reason the analytic path funnels its keys: an unread key here leaves a scene that
    still builds and still conserves energy.

    ``'mu': null`` reads as *absent*, giving the non-magnetic ``Material(epsilon)``. JSON's
    null and a missing key are the same statement ("no permeability named"), and collapsing
    them here is what gives one canonical parsed form -- so the two spellings hash to the same
    :meth:`JobSpec.key`, exactly as the two spellings of a polarization component do.

    Everything else goes through untouched: a :class:`~wavebit._core.domain.Material` built in
    process, a plain number, and equally a tracer or an array that a caller assembled without
    going through JSON. The domain layer passes half-space values through unread on purpose,
    and a spec layer that coerced them would become the second reader of the material
    convention that :class:`~wavebit._core.domain.Stack` says must not exist. Untouched is not
    unchecked, though: the two spellings that are *legible but not numeric* -- a string and a
    bool -- are refused here rather than left to become ``Material('Si')`` or a medium of
    permittivity ``True``, which is 1 and solves.
    """
    if isinstance(value, Material):
        return value
    if isinstance(value, dict):
        _reject_unknown(value, ('epsilon', 'mu'), where)
        if 'epsilon' not in value:
            raise SpecError(
                f"{where}: a material object needs 'epsilon' -- it is written "
                f"{{'epsilon': e}} for a plain medium and {{'epsilon': e, 'mu': m}} for a "
                f"magnetic one, both relative and both permittivity/permeability rather than "
                f"an index. Got {sorted(value)}.")
        epsilon = _scalar_from_json(value['epsilon'], f'{where}.epsilon')
        mu = value.get('mu')
        return Material(epsilon) if mu is None else Material(
            epsilon, _scalar_from_json(mu, f'{where}.mu'))
    if isinstance(value, (str, bytes)):
        raise SpecError(
            f'{where}: expected a permittivity, [re, im], or {{"epsilon": e}}, got '
            f'{value!r}. Materials are numeric; wavebit has no lookup from a material name.')
    return _scalar_from_json(value, where)


def _scalar_from_json(value, where):
    """One material number as written: a real, or the ``[re, im]`` pair JSON needs for complex.

    A two-list is unambiguous here because the caller has already decided this slot holds a
    single number; that is the whole reason the complex spelling is confined to named slots.
    """
    if isinstance(value, (list, tuple)):
        ok = (len(value) == 2 and all(isinstance(x, (int, float))
                                      and not isinstance(x, bool) for x in value))
        if not ok:
            raise SpecError(
                f'{where}: a complex material is written as the pair [re, im], got {value!r}')
        return complex(value[0], value[1])
    if isinstance(value, bool):
        raise SpecError(f'{where}: expected a number, got {value!r}')
    return value


def _revive_materials(value, where='material'):
    """Turn every material *object* inside a shape recipe back into a ``Material``.

    :func:`_jsonable` writes a ``Material`` as ``{'epsilon': ...}`` at whatever depth it finds
    one, so the reader has to meet it at the same depths or ``to_dict`` would emit a document
    its own ``from_dict`` reads as something else. Recognition is by shape rather than by
    position -- a dict carrying ``epsilon`` and at most ``mu`` -- which is safe because the two
    recipe schemas funnel their keys strictly (``background`` / ``shapes`` on a
    layer, and a fixed set on a shape), so no legitimate dict in here has an ``epsilon`` key.

    Rasters never reach this: they are arrays of numbers, not containers of assignments, and
    :func:`_structure_from_json` has already taken that branch.
    """
    if isinstance(value, dict):
        if 'epsilon' in value and set(value) <= {'epsilon', 'mu'}:
            return _material_from_json(value, where)
        return {k: _revive_materials(v, f'{where}.{k}') for k, v in value.items()}
    if isinstance(value, list):
        return [_revive_materials(v, f'{where}[{i}]') for i, v in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(_revive_materials(v, f'{where}[{i}]') for i, v in enumerate(value))
    return value


# ------------------------------------------------------------------ numerics

@dataclass
class Numerics:
    """Discretisation and backend choices — everything that is not the physical scene.

    ``complex_dtype`` is the bit width (128 / 64) rather than a dtype object, so the spec
    stays backend-neutral; :func:`run` maps it onto the solver's code.
    """
    backend: Backend = Backend.NUMPY
    truncation_order: Any = (0, 0)
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
    #: ``None`` = unset, same shape as the two above though the reason is simpler: the
    #: default ('distance') is path-independent, but an explicit value is *refused* on any
    #: build that constructs no raster normal field, so a spec that never asked must not be
    #: made to ask.
    normal_method: Optional[NormalMethod] = None
    complex_dtype: int = 128
    layer_algorithm: str = 'ETM'
    normal_grid: int = 1024
    #: What the solve keeps the state to answer. Unlike the three above this has a real
    #: default rather than ``None``: it resolves against nothing -- no modeling path, no
    #: geometry -- so ``'amplitude'`` here is the same claim a bare ``rcwa()`` makes, and
    #: writing it into every spec costs a spec nothing. It lives in ``numerics`` and not in
    #: ``incidence`` because it describes the job, not the light.
    answers: Answers = Answers.AMPLITUDE

    _FIELDS = ('backend', 'truncation_order', 'fourier_coefficients', 'factorization', 'normal_method',
               'complex_dtype', 'layer_algorithm', 'normal_grid', 'answers')

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
        if d.get('normal_method') is not None:
            d['normal_method'] = _enum(d['normal_method'], NormalMethod,
                                       'numerics.normal_method')
        if 'answers' in d:
            d['answers'] = _enum(d['answers'], Answers, 'numerics.answers')
        if 'truncation_order' in d:
            # The solver's own parser, not a second one. This was an inline check that refused
            # five spellings `rcwa()` accepts -- `2.0`, `[3]`, both mappings, and a NumPy
            # integer -- while catching two the parser missed. Both halves now live in
            # `parse_order_pair`, so the two doors agree by construction rather than by anyone
            # remembering to update both. A mapping is canonicalised to a pair here, which
            # costs a spec written that way the deferred `x`/`y`-on-a-skewed-cell *warning*;
            # it is a warning rather than a refusal, and it still fires for a caller who hands
            # the mapping straight to `rcwa()`.
            try:
                d['truncation_order'] = tuple(
                    parse_order_pair(d['truncation_order'], 'numerics.truncation_order')[0])
            except ValueError as exc:
                raise SpecError(str(exc)) from None
        if d.get('complex_dtype', 128) not in (128, 64):
            raise SpecError(
                f"numerics.complex_dtype: expected 128 or 64, got {d['complex_dtype']!r}")
        return cls(**d)

    def check(self):
        """Every reason these numerics cannot be used, as messages, or an empty tuple.

        Same contract as :meth:`~wavebit._core.domain.Stack.check`. ``truncation_order`` goes
        through ``parse_order_pair`` -- the solver's own parser, which this class stopped
        second-guessing in `0dde6c5` -- and ``complex_dtype`` is checked in the bit-width
        spelling this class uses (128 / 64), not the solver's 0 / 1 code. That difference is
        real and is why the width is not simply handed to `_resolve_precision`: they are two
        vocabularies for one choice, and `as_kwargs` is where one becomes the other.
        """
        problems = []
        if self.truncation_order is not None:
            try:
                parse_order_pair(self.truncation_order, 'truncation_order')
            except ValueError as exc:
                problems.append(str(exc))
        if self.complex_dtype not in (128, 64):
            problems.append(
                f'complex_dtype: expected 128 or 64, got {self.complex_dtype!r}. It is the '
                f'width in bits; the solver spells the same choice 0 and 1.')
        if self.normal_grid is not None:
            try:
                grid = int(self.normal_grid)
                bad = grid != self.normal_grid or grid <= 0
            except (TypeError, ValueError):
                bad = True
            if bad:
                problems.append(
                    f'normal_grid: expected a positive whole number of samples, got '
                    f'{self.normal_grid!r}.')
        return tuple(problems)

    def to_dict(self) -> dict:
        return {f: _jsonable(getattr(self, f)) for f in self._FIELDS}

    def as_kwargs(self) -> dict:
        """Solver keyword arguments. ``complex_dtype`` becomes the solver's 0/1 code.

        An unset ``fourier_coefficients`` or ``factorization`` is *omitted* rather than filled
        in, so that the solver sees the same "nobody asked" it would from a bare ``rcwa()``
        call and resolves each per modeling path.
        """
        return dict(
            backend=int(self.backend),
            truncation_order=list(self.truncation_order),
            complex_dtype=0 if self.complex_dtype == 128 else 1,
            layer_algorithm=self.layer_algorithm,
            normal_grid=self.normal_grid,
            answers=answers_of(self.answers).value,
            **({} if self.fourier_coefficients is None
               else {'fourier_coefficients': self.fourier_coefficients.value}),
            **({} if self.factorization is None
               else {'factorization': self.factorization.value}),
            **({} if self.normal_method is None
               else {'normal_method': self.normal_method.value}),
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
    solver: str = 'rcwa'
    version: str = SPEC_VERSION

    _FIELDS = ('version', 'solver', 'numerics', 'stack', 'incidence', 'sweep')

    #: Removed in SPEC_VERSION 4, and refused by name rather than as an unknown field. It
    #: declared which result quantities to return and **nothing ever read it** -- `run`
    #: returns the whole `Result` either way -- while `key()` hashed it, so two specs with
    #: byte-identical results got different cache keys whenever one spelled it differently.
    #: An unread knob is the same defect as an unread `normal_method`, and this library
    #: refuses those rather than ignoring them; the difference here is that it was silent
    #: for as long as it existed, so the removal has to say what a caller who wrote it was
    #: promised. A `Result` carries `R`, `T` and the Fresnel amplitudes together; select
    #: from it after the run.
    _RETIRED_OUTPUTS = (
        "spec.outputs: removed in spec version 4. It was never read -- `run` returns the "
        "whole Result, with R, T and the Fresnel amplitudes on it -- while `JobSpec.key()` "
        "hashed it, so two specs that produce identical results got different cache keys. "
        "Drop the field and select the quantities you want off the returned Result.")

    # -------------------------------------------------------------- construction

    @classmethod
    def from_dict(cls, d) -> 'JobSpec':
        """Build from a plain mapping, rejecting anything that will not execute.

        Validation is deliberately eager and specific: a spec that reaches a worker is far
        more expensive to debug than one rejected at submission.

        This is the *parse* boundary and it cannot see where the mapping came from -- a JSON
        payload the caller decoded itself arrives here exactly like a dict literal typed two
        lines up -- so it holds every mapping to the same contract, including the mandatory
        ``version``. Constructing ``JobSpec(...)`` directly is the path with no ceremony, and
        it is the honest one: a spec assembled out of this module's own classes is versioned
        by the build that assembled it.
        """
        if not isinstance(d, dict):
            raise SpecError(f"spec: expected a mapping, got {type(d).__name__}")
        d = dict(d)
        if 'outputs' in d:
            raise SpecError(cls._RETIRED_OUTPUTS)
        _reject_unknown(d, cls._FIELDS, 'spec')

        if 'version' not in d:
            raise SpecError(_MISSING_VERSION)
        version = d['version']
        if str(version) != SPEC_VERSION:
            raise SpecError(
                f"spec.version: this build understands version {SPEC_VERSION!r}, "
                f"got {version!r}")
        solver = d.get('solver', 'rcwa')
        if solver != 'rcwa':
            raise SpecError(f"spec.solver: only 'rcwa' exists so far, got {solver!r}")

        return cls(
            stack=_stack_from_dict(d.get('stack')),
            incidence=_incidence_from_dict(d.get('incidence')),
            numerics=Numerics.from_dict(d.get('numerics')),
            sweep=Sweep.from_dict(d.get('sweep')),
            solver=solver,
            version=SPEC_VERSION,
        )

    @classmethod
    def from_json(cls, text) -> 'JobSpec':
        return cls.from_dict(json.loads(text))

    # -------------------------------------------------------------- serialisation

    def check(self):
        """Every reason this job cannot be submitted, as messages, or an empty tuple.

        The three carriers' checks, each prefixed with where it came from, in the order a
        reader would look: the scene, then the illumination, then the numerics. Nothing is
        raised -- `run` and the solver still refuse, one problem at a time, which is right
        for them; this is for the caller who would rather fix all four at once than learn
        about them across four submissions.

        Deliberately not called by `from_dict`. That door parses JSON and refuses what cannot
        become a spec at all; this one asks whether a spec that parsed can be solved, and the
        two are different questions with different right answers for a caller who is, say,
        generating a sweep and wants to report every bad point rather than stop at the first.
        """
        problems = []
        for where, carrier in (('stack', self.stack), ('incidence', self.incidence),
                               ('numerics', self.numerics)):
            if carrier is None:
                continue
            # One prefix shape, always. Guessing whether a message already names its field
            # produced `incidence.psi and polarization both given`, which reads as a field
            # called `psi and polarization`.
            problems.extend(f'[{where}] {message}' for message in carrier.check())
        return tuple(problems)

    def to_dict(self) -> dict:
        return {
            'version': self.version,
            'solver': self.solver,
            'numerics': self.numerics.to_dict(),
            'stack': _stack_to_dict(self.stack),
            'incidence': _incidence_to_dict(self.incidence),
            'sweep': self.sweep.to_dict(),
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
    _reject_unknown(d, ('layers', 'top', 'bot', 'period', 'lattice'), 'stack')
    layers = []
    for i, raw in enumerate(d.get('layers', [])):
        _reject_unknown(raw, ('thickness', 'structure', 'mu'), f'stack.layers[{i}]')
        if 'thickness' not in raw or 'structure' not in raw:
            raise SpecError(f"stack.layers[{i}]: needs both 'thickness' and 'structure'")
        thickness = raw['thickness']
        if not isinstance(thickness, (int, float)) or thickness < 0:
            raise SpecError(
                f"stack.layers[{i}].thickness: expected a non-negative number, "
                f"got {thickness!r}")
        layers.append(Layer(
            thickness=thickness,
            structure=_structure_from_json(raw['structure'], i, 'structure'),
            mu=(None if raw.get('mu') is None
                else _structure_from_json(raw['mu'], i, 'mu'))))
    period = d.get('period', (1.0, 1.0))
    lattice = d.get('lattice')
    if lattice is not None:
        lattice = [list(row) for row in lattice]
    return Stack(layers=layers,
                 top=_material_from_json(d.get('top', 1.0), 'stack.top'),
                 bot=_material_from_json(d.get('bot', 1.0), 'stack.bot'),
                 period=list(period) if isinstance(period, (list, tuple)) else period,
                 lattice=lattice)


def _stack_to_dict(stack: Stack) -> dict:
    """The inverse of :func:`_stack_from_dict` -- and it has to be a real inverse.

    ``top`` / ``bot`` go through :func:`_material_to_json` rather than :func:`_jsonable`
    because they are *named* material slots, which is what makes the bare-complex ``[re, im]``
    spelling readable there. A ``Material`` needs no such help at either end: its object form
    identifies itself, so :func:`_jsonable` writes one wherever it appears, including inside a
    shape recipe.
    """
    out = {
        'top': _material_to_json(stack.top),
        'bot': _material_to_json(stack.bot),
        'period': _jsonable(stack.period),
        **({} if stack.lattice is None else {'lattice': _jsonable(stack.lattice)}),
        'layers': [{'thickness': _jsonable(l.thickness),
                    'structure': _jsonable(l.structure),
                    **({} if l.mu is None else {'mu': _jsonable(l.mu)})}
                   for l in stack.layers],
    }
    return out


def _structure_from_json(value, index, key):
    """A layer's structure (or its ``mu`` field), as written in JSON.

    A nested list is a raster grid and becomes an ndarray -- of **relative permittivity**, the
    quantity the whole spec is stated in since ``SPEC_VERSION`` 3. Shape recipes
    keep their shape and their numbers; the one thing done to them is that a material written
    in its object form (``{'epsilon': ...}``) becomes a ``Material`` again, since that is what
    :func:`_jsonable` emitted for one. A bare number in a material slot stays a bare number --
    the scalar shorthand is an input convenience, and turning it into a ``Material`` here would
    make the spec layer the second place that decides what a bare number means.

    A raster inlined into a spec is convenient and does not scale: a 512x512 grid is
    megabytes of JSON per layer. Large rasters belong in object storage, referenced rather
    than embedded; that path arrives with the parametric geometry layer.
    """
    if isinstance(value, dict) and 'raster_ref' in value:
        raise SpecError(
            f"stack.layers[{index}].{key}: 'raster_ref' is not resolvable here — "
            f"this entry point takes self-contained specs. Inline the grid, or resolve "
            f"the reference before calling.")
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        import numpy as np
        return np.asarray(value, dtype=float)
    return _revive_materials(value, f'stack.layers[{index}].{key}')


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
    if isinstance(tetm, Tetm):
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
    return solver.solve(geometry=geometry)
