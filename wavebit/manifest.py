"""What this solver supports, in a form another program can read.

Wavebit already knows this about itself, in pieces: ``_core/enums.py`` holds the legal
values, the ``structure`` setter decides which modeling path a structure takes, and
``_core/engine.py`` carries the tables that refuse an option a path cannot express. This
module collects those into one JSON-round-trippable document, so a caller outside the
process — documentation, a configuration UI, an assistant that recommends conditions — can
read the same facts the solver enforces instead of restating them.

**Capabilities and constraints only.** What each path implements, what is refused and why,
and which spelling of ``structure`` selects which path. Deliberately not here:

* *Recommendations.* "PyTorch below ``truncation_order≈5``" is a measurement about a workload, not a
  property of the solver, and it belongs to whoever is giving advice. The manifest says
  which backends exist; picking one is someone else's job.
* *A vocabulary for describing experiments.* Fields like "is the boundary curved" describe
  a user's problem, not this library.

The line matters because of what it buys: everything here is checked against the code it
describes by ``tests/test_manifest.py``, so a path that grows or loses an option fails the
suite rather than going stale in a consumer. A fact that cannot be checked that way does
not belong in this module.

Regenerate the JSON with::

    python -m wavebit.manifest > wavebit-manifest.json
"""
import json

from ._core.domain import RETIRED_MATERIAL_NAMES
from ._core.enums import (Answers, Backend, DEFAULT_ANSWERS, Factorization,
                          FourierCoefficients, NormalMethod, Tetm)
from ._core.engine import (_COEFFICIENTS_IMPLEMENTED_BY, _COEFFICIENTS_REASON,
                           _FACTORIZATION_IMPLEMENTED_BY, _FACTORIZATION_REASON)
from ._core.spec import SPEC_VERSION

#: Bumped when the document's *shape* changes, so a consumer can refuse one it cannot read.
#: Independent of the Wavebit version, which moves whenever the content does.
#: '2': the ``results`` section — the solve's output contract joined the document when the
#: result surface was renamed for the first release (2026-08-12).
#: '3': the ``normal_method`` axis — per-path availability (with the contourpy probe) and
#: reasons, the ``enums`` entry, and ``normal_methods_refused_on_oblique`` in ``lattice``
#: (2026-08-17).
#: '4': the ``answers`` axis — the construction-time declaration of what a solve is for, its
#: ladder and default, and the ``enums`` entry (2026-08-20).
#: '5': the material convention — ``material_quantity``, and the ``structure`` key on every
#: path entry where ``ucell`` used to be. The *value* a caller writes changed with the name
#: (refractive index -> relative permittivity), so a consumer that recognised the old shape
#: and kept generating old numbers would generate legal, energy-conserving, wrong scenes:
#: this bump is what lets it refuse instead (2026-08-20).
#: '6': the pre-release rename set (2026-08-20). What a consumer sees move:
#: ``results.incidences.fields`` goes ``('res', 'res_te_inc', 'res_tm_inc')`` ->
#: ``('given', 's_inc', 'p_inc')``, and the constructor keywords named in ``why`` and
#: ``meaning`` prose move with them (``fto`` -> ``max_order`` -> ``truncation_order``, ``type_complex`` ->
#: ``complex_dtype``, ``connecting_algo`` -> ``layer_algorithm``). ``docs/capability-manifest.md``
#: says a rename alone does not earn a bump, because a consumer reading unknown keys degrades
#: gracefully -- but ``fields`` is a *recognisable* key whose contents are attribute names a
#: consumer indexes a result object with, so the old contents keep their shape while ceasing
#: to resolve. That is the same failure the '5' bump exists for, one level down.
MANIFEST_VERSION = '7'


# --------------------------------------------------------------- how a structure is spelled

#: What a caller passes as ``structure``, and the modeling path it selects. There is no
#: separate mode switch — the type of the object *is* the choice, which is why a consumer that
#: wants to tell a user how to spell their structure has to be told this rather than inferring
#: it from a keyword.
#:
#: ``material_form`` is the second half of the same fact and is what :data:`MATERIAL_QUANTITY`
#: is stated against: three of the four paths name a material per region ('assignment') and
#: one samples a field ('field'), which is why only the raster takes a ``mu=`` array while the
#: other three carry permeability inside each ``Material``.
SPELLINGS = (
    {
        'path': 'raster',
        'modeling_type': 0,
        'structure': 'a NumPy array of relative permittivity, one 2D grid per layer',
        'material_form': 'field',
        'mu': 'a second array of the same shape, passed as mu=',
        'summary': 'Permittivity as samples. The only path that implements both Fourier '
                   'series and both factorization rules, and the one topology optimisation '
                   'uses.',
    },
    {
        'path': 'block',
        'modeling_type': 1,
        'structure': 'a list of [background, [shape, ...]] layers, each shape a list '
                     "starting with 'rectangle' or 'ellipse' and carrying its Material in "
                     'position 5',
        'material_form': 'assignment',
        'mu': 'inside each Material',
        'summary': 'Shapes decomposed into axis-aligned blocks, painted in order. Edges that '
                   'are not axis-aligned become a staircase.',
    },
    {
        'path': 'smooth',
        'modeling_type': 2,
        'structure': "a list of dict(background=..., shapes=[...], combine='add') layers, "
                     "each shape dict carrying material=",
        'material_form': 'assignment',
        'mu': 'inside each Material',
        'summary': 'Shape edges enter the Fourier transform in closed form, so a tilted or '
                   'curved boundary carries no staircase error. Overlapping shapes are summed.',
    },
    {
        'path': 'boundary',
        'modeling_type': 2,
        'structure': "the smooth spelling with combine='paint'",
        'material_form': 'assignment',
        'mu': 'inside each Material',
        'summary': 'The smooth path with overlaps resolved into disjoint polygons first, so '
                   'later shapes cover earlier ones without losing edge exactness.',
    },
)


#: **Which quantity a material number is.** Machine-readable, because the answer changed:
#: every material slot took a refractive index until 2026-08-20 and takes relative
#: permittivity now. A consumer that generated scenes against the old answer would keep
#: generating numbers that are still *legal* -- ``3.5`` is a fine permittivity -- so nothing
#: downstream raises and no energy check fires. Prose cannot be tested for; this can, which is
#: the whole reason it is a dict of names rather than a sentence in ``summary``.
#:
#: ``retired`` maps each old spelling to the new one, and is *read from the table the
#: refusals themselves use* (:data:`wavebit._core.domain.RETIRED_MATERIAL_NAMES`) rather than
#: restated here. Every entry is refused **by name** at the boundary rather than accepted, so
#: a consumer can rely on an old document failing loudly instead of being reinterpreted -- and
#: deriving the list is what keeps that promise true of the list rather than of a copy of it.
MATERIAL_QUANTITY = {
    'quantity': 'permittivity',
    'relative': True,
    'complex': True,
    'loss_sign': 'Im(epsilon) > 0',
    'time_convention': 'exp(-i*omega*t)',
    'type': 'wavebit.Material(epsilon, mu=None)',
    'scalar_shorthand': 'a bare number means Material(epsilon=x)',
    'magnetic_slot': 'mu',
    # Absence and the value one are different inputs, and a consumer generating scenes has to
    # know which it is writing: mu=None (the default, and what the scalar shorthand produces)
    # says no permeability was named and keeps the solve off the permeability path entirely,
    # while mu=1.0 names it and builds that path for the same numbers. The distinction is
    # structural so that it can be read inside a jax.jit trace, where a value cannot be.
    'non_magnetic_spelling': 'mu=None',
    'conversions': ('eps_from_index', 'index_from_eps'),
    'retired': {old: new for old, (new, _what) in RETIRED_MATERIAL_NAMES.items()},
    'source': 'docs/conventions.md',
}

#: Shape lengths on both parametric paths are absolute, in the units of ``period`` — not
#: fractions of it. Stated here because passing a fraction is not an error: it solves a
#: different, much smaller structure and nothing raises.
SHAPE_COORDINATES = 'absolute'


# ------------------------------------------------------------------ per-path option support

def _options_for(modeling_type):
    """The two Fourier axes for one modeling path, as implemented rather than as documented.

    Read from :mod:`wavebit._core.engine`'s own tables. A path absent from a table
    implements every value of that axis and takes the option as given; a path present in one
    implements exactly the value it names, and the reason says whether the others are unbuilt
    or unbuildable.
    """
    factorization = _FACTORIZATION_IMPLEMENTED_BY.get(modeling_type)
    coefficients = _COEFFICIENTS_IMPLEMENTED_BY.get(modeling_type)
    return {
        'factorization': (
            [m.value for m in Factorization] if factorization is None
            else [factorization.value]),
        'factorization_reason': (
            None if factorization is None else _FACTORIZATION_REASON[modeling_type]),
        'fourier_coefficients': (
            [m.value for m in FourierCoefficients] if coefficients is None
            else [coefficients.value]),
        'fourier_coefficients_reason': (
            None if coefficients is None else _COEFFICIENTS_REASON[modeling_type]),
        # The third axis exists only where a raster normal field is built. On the raster
        # path the constructions are probed rather than assumed -- 'contour' needs the
        # optional contourpy dependency, and offering an option this environment cannot
        # build is exactly the drift the manifest tests exist to prevent -- and the axis is
        # read only under factorization='nvf' (explicitly set anywhere it has no reader, it
        # is refused rather than ignored). The block path applies Li's rules and builds no
        # normal field at all; the smooth shape path builds its normal field from the
        # geometry itself. Neither has a raster construction to select.
        'normal_method': _raster_normal_methods() if modeling_type == 0 else [],
        'normal_method_reason': (
            "read only under factorization='nvf'" if modeling_type == 0
            else ("the block path applies Li's rules only and builds no normal field"
                  if modeling_type == 1
                  else 'the smooth shape path builds its normal field from the geometry '
                       'itself') + '; there is no raster construction to select'),
    }


def _raster_normal_methods():
    """The constructions this environment can actually build, probed not assumed."""
    import importlib.util
    methods = [m.value for m in NormalMethod]
    if importlib.util.find_spec('contourpy') is None:
        methods.remove(NormalMethod.CONTOUR.value)
    return methods


# ------------------------------------------------------------------ lattice

#: What a non-orthogonal (skewed) cell refuses, mirrored from
#: :meth:`wavebit._core.engine.CoreSolveMixin._reject_unsupported_lattice`. Both of these
#: *run* on a skewed cell and return a plausible, wrong number — the raster is still a
#: raster, the transforms are still transforms, and NVF's assembly stays Hermitian so energy
#: conserves to 1e-15 either way. Nothing downstream can notice, which is why the refusal is
#: a fact worth publishing rather than an implementation detail.
#:
#: A third refusal used to live here, ``normal_method`` — see the note on the empty tuple
#: below. It was the one entry that named an *accuracy* limit rather than a wrong answer, and
#: that is what made it removable.
LATTICE = {
    'default': 'period, meaning diag(Lx, Ly)',
    'oblique': 'pass the two lattice vectors as the columns of `lattice`; `period` then '
               'reports their lengths',
    'raster_coordinates': 'lattice',
    #: The same refusals as ``refused_on_oblique`` below, named rather than described.
    #: A consumer deciding *what to recommend* needs to test membership, and prose is not a
    #: membership test — these are what makes the constraint machine-checkable outside this
    #: process. Kept beside the prose so the two cannot be updated apart; a test asserts they
    #: agree.
    #:
    #: ``normal_methods_refused_on_oblique`` is empty as of 2026-08-21 and kept rather than
    #: dropped: the key is the membership test a consumer already asks, and an empty answer
    #: is the answer. All four constructions carry the cell metric now — see
    #: ``docs/nvf-normal-methods.md``; ``'contour'`` is the one to recommend on a skewed cell
    #: as on any other.
    'paths_refused_on_oblique': ('block',),
    'factorizations_refused_on_oblique': ('li',),
    'normal_methods_refused_on_oblique': (),
    'refused_on_oblique': (
        {
            'what': 'the block path (modeling_type=1)',
            'why': 'its modeler rasterises shapes onto a rectangular grid of its own, which '
                   'is not ported to a skewed cell',
        },
        {
            'what': "factorization='li'",
            'why': 'the inverse rule is applied along the raster axes, which are a1 and a2 '
                   'rather than x and y once the cell is skewed. Use NVF.',
        },
    ),
}


# ------------------------------------------------------------------ materials

#: The permittivity tensor classes the mode operator can be given. ``supported`` is the
#: solver's own answer, not a roadmap: a false here means a run cannot be done today, and
#: ``why`` says what would have to change.
MATERIALS = (
    {
        'case': 'isotropic',
        'supported': True,
        'why': 'a scalar relative permittivity per material, complex for a lossy one '
               '(Im(eps) > 0); see material_quantity for what a material number means',
        'source': 'wavebit/_core/solve_2d.py',
    },
    {
        'case': 'uniform_anisotropic',
        'supported': True,
        'why': 'solve_conv reads nine permittivity slots, so a tensor filling a whole '
               'layer works, z row included. Note that every off-diagonal slot holds the '
               'negative of its component: -eps_xy, and equally -eps_xz.',
        'source': 'docs/off-diagonal-slot-convention.md',
    },
    {
        'case': 'patterned_anisotropic',
        'supported': True,
        'why': 'the in-plane tensor -- the five slots epx, epy, epz_i, epxy, epyx -- through '
               'the Popov-Neviere rearrangement: finish the projector algebra pointwise, '
               'transform once. Both solvers are unchanged, as it was scoped. A patterned '
               'layer that also fills the z row is patterned_z_coupled below, not this.',
        'source': 'docs/patterned-anisotropic-factorization.md',
    },
    {
        'case': 'magnetic',
        'supported': True,
        'why': 'mu is the same five slots on the other side of the (E, H, eps, mu) -> '
               '(H, -E, mu, eps) duality: layers as the E/H dual, half-spaces as one pair '
               'of admittances, patterned layers through the same Popov-Neviere '
               'rearrangement. Written as Material(epsilon, mu) on the three assignment '
               'paths and as a mu= array beside the raster; the same builders run twice, so '
               'both quantities land on one geometry. The 2D path only -- the 1D/conical '
               'operators and the first-order 4ff solve take no mu, and refuse rather than '
               'ignore one. Field reconstruction carries mu since 2026-08-20: H_z is '
               'Faraday z row, mu_zz^-1 (Kx Ey - Ky Ex), and before that date it came back '
               'scaled by mu_zz while every other component was exact.',
        'source': 'docs/magnetic-materials.md',
    },
    {
        'case': 'z_coupled',
        'supported': True,
        'why': 'with eps_xz or eps_yz the second-order reduction fails, so the solve takes '
               'the first-order (4ff) formulation instead -- selected automatically by the '
               'presence of the four z-row slots, or by formulation=4ff, at about 8x. Two '
               'limits ride with it: layer_algorithm must be ETM, and mu is not implemented '
               'there. Field reconstruction is (2026-08-14). Uniform layers only; a '
               'patterned z row is patterned_z_coupled below.',
        'source': 'wavebit/_core/solve_2d_4ff.py',
    },
    {
        'case': 'patterned_z_coupled',
        'supported': False,
        'why': 'patterned *and* z-coupled: the 3x3 closing step. The two halves that exist '
               'reach this from opposite sides and neither extends to it -- the in-plane '
               'factorization has no z row, and the 4ff solve takes no pattern. No oracle '
               'exists for it either, which is the harder half.',
        'source': 'docs/anisotropic-materials.md',
    },
)


# ------------------------------------------------------------------ what a solve returns

#: The result contract, restated as data for consumers that build UIs over the output.
#: This is the half of ``docs/conventions.md`` a program can hold: field names, their kind,
#: and the layout rule. It passes the same admission test as everything else here — every
#: claim is checked against a *real* result object by ``tests/test_manifest.py``, so a
#: rename that misses this table fails the suite instead of going stale in a consumer.
RESULTS = {
    'convention': 'uppercase is power, lowercase is amplitude; the normative statement '
                  'is docs/conventions.md',
    'efficiencies': {
        'total': ('R', 'T'),
        'by_outgoing_polarization': ('R_s', 'R_p', 'T_s', 'T_p'),
        'layout': 'arrays of shape (n_orders_y, n_orders_x) — y first, the transpose of the '
                  'truncation_order=[x, y] input order. Centred on the incident order unless '
                  'an excitation widened the retained window; result.order_window says which '
                  'orders these are, and result.order(m, n) spares you the arithmetic',
        'accessor': 'result.order(m, n) indexes every per-order quantity by the order '
                    'indices themselves',
        'caveat': 'with an absorbing superstrate the incident-power normalization is '
                  'convention-dependent, so these are not comparable to 1; use the '
                  'amplitudes there',
    },
    'amplitudes': {
        'fields': ('r_s', 'r_p', 't_s', 't_p'),
        'meaning': 'complex Fresnel amplitudes in e^{-iwt}: E-field ratios against a unit '
                   'incident E-field, comparable to the textbook coefficients',
        'p_sign': 'the convention where a normal-incidence interface gives r_p = -r_s',
        'per_order_frame': 'each order is resolved in its own s/p frame, s_hat = zhat x '
                           'khat_par; at theta = 0 the zeroth order has no in-plane '
                           'direction and phi supplies it',
    },
    'incidences': {
        'fields': ('given', 's_inc', 'p_inc'),
        'meaning': 'the solve for the incidence as specified — psi, or the complex '
                   'amplitude pair polarization=(a_s, a_p) — plus the pure-s and pure-p '
                   'variants; top-level R/T average the two variants when no specified '
                   'solve exists',
        'polarization_input': 'polarization=(a_s, a_p) supersedes psi (naming both raises); '
                              'psi = v is exactly (sin(v), cos(v)), and the pair adds the '
                              'relative phase psi cannot spell — (1, -1j)/sqrt(2) and '
                              '(1, +1j)/sqrt(2) are the circular states. The pair is '
                              'normalized to unit incident flux (overall scale is free, '
                              'overall phase is kept)',
    },
    'incidence_side': {
        'values': ('top', 'bottom'),
        'default': 'top',
        'meaning': "which medium the illumination arrives from ('bottom' = the `bot` "
                   "material). top / bot / structure order are scene labels and never move; "
                   'results '
                   'are incidence-relative — R returns into the incidence medium, T exits '
                   'the far side, theta/phi are measured against the incidence medium\'s '
                   'normal — so post-processing is identical for both sides',
    },
    'answers': {
        'values': tuple(m.value for m in Answers),
        'default': DEFAULT_ANSWERS.value,
        'ladder': tuple(m.value for m in Answers),
        'meaning': 'what a solve keeps the state to answer, declared at construction. A '
                   'ladder, not a menu: each rung answers everything below it. '
                   "'efficiency' is R/T per order; 'amplitude' adds the Fresnel amplitudes, "
                   'their s/p-resolved efficiencies and the pure-TE/pure-TM incidence '
                   "variants; 'field' adds the per-layer modes and amplitudes "
                   'reconstruct_field walks. Below that rung reconstruct_field is refused '
                   'rather than served, because the state is kept while the solve runs or not at '
                   'all -- it is 4.25 matrices of side 2*n_orders_x*n_orders_y per layer on the '
                   '2D path '
                   '(3.75 conical, 5 pure 1D), against boundary vectors for the amplitudes',
    },
    'jobspec': {
        #: No `outputs_default` since spec version 4: `JobSpec.outputs` was removed rather
        #: than made to work. It was never read -- a run returns the whole Result -- while
        #: `key()` hashed it, so two specs with identical results got different cache keys.
        'spec_version': SPEC_VERSION,
    },
    'source': 'docs/conventions.md',
}

# ------------------------------------------------------------------ the document

def manifest() -> dict:
    """Everything above, as one JSON-round-trippable document."""
    from . import __version__

    return {
        'manifest_version': MANIFEST_VERSION,
        'wavebit_version': __version__,
        'enums': {
            'backend': [m.name.lower() for m in Backend],
            'tetm': [m.name.lower() for m in Tetm],
            'factorization': [m.value for m in Factorization],
            'fourier_coefficients': [m.value for m in FourierCoefficients],
            'normal_method': [m.value for m in NormalMethod],
            'answers': [m.value for m in Answers],
        },
        'shape_coordinates': SHAPE_COORDINATES,
        'material_quantity': _jsonable_material_quantity(),
        'paths': [
            dict(spelling, **_options_for(spelling['modeling_type']))
            for spelling in SPELLINGS
        ],
        'lattice': {
            'default': LATTICE['default'],
            'oblique': LATTICE['oblique'],
            'raster_coordinates': LATTICE['raster_coordinates'],
            'paths_refused_on_oblique': list(LATTICE['paths_refused_on_oblique']),
            'factorizations_refused_on_oblique': list(
                LATTICE['factorizations_refused_on_oblique']),
            'normal_methods_refused_on_oblique': list(
                LATTICE['normal_methods_refused_on_oblique']),
            'refused_on_oblique': [dict(r) for r in LATTICE['refused_on_oblique']],
        },
        'materials': [dict(m) for m in MATERIALS],
        'results': _jsonable_results(),
    }


def _jsonable_material_quantity() -> dict:
    """:data:`MATERIAL_QUANTITY` with its tuples as lists, so it round-trips through JSON."""
    out = dict(MATERIAL_QUANTITY)
    out['conversions'] = list(out['conversions'])
    out['retired'] = dict(out['retired'])
    return out


def _jsonable_results() -> dict:
    """``RESULTS`` with tuples as lists, so the document round-trips through JSON."""
    def conv(v):
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, tuple):
            return list(v)
        return v
    return conv(RESULTS)


def cited_sources() -> set:
    """Every repository path the manifest points at, for the staleness check."""
    return {m['source'] for m in MATERIALS} | {MATERIAL_QUANTITY['source']}


def main():
    print(json.dumps(manifest(), indent=2, sort_keys=False))


if __name__ == '__main__':
    main()
