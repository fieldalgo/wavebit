"""What this solver supports, in a form another program can read.

Wavebit already knows this about itself, in pieces: ``_core/enums.py`` holds the legal
values, the ``ucell`` setter decides which modeling path a structure takes, and
``_core/engine.py`` carries the tables that refuse an option a path cannot express. This
module collects those into one JSON-round-trippable document, so a caller outside the
process — documentation, a configuration UI, an assistant that recommends conditions — can
read the same facts the solver enforces instead of restating them.

**Capabilities and constraints only.** What each path implements, what is refused and why,
and which spelling of ``ucell`` selects which path. Deliberately not here:

* *Recommendations.* "PyTorch below ``fto≈5``" is a measurement about a workload, not a
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

from ._core.enums import Backend, Factorization, FourierCoefficients, NormalMethod, Pol
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
MANIFEST_VERSION = '3'


# ------------------------------------------------------------------ how a ucell is spelled

#: What a caller passes as ``ucell``, and the modeling path it selects. There is no separate
#: mode switch — the type of the object *is* the choice, which is why a consumer that wants
#: to tell a user how to spell their structure has to be told this rather than inferring it
#: from a keyword.
SPELLINGS = (
    {
        'path': 'raster',
        'modeling_type': 0,
        'ucell': 'a NumPy array of refractive indices, one 2D grid per layer',
        'summary': 'Permittivity as samples. The only path that implements both Fourier '
                   'series and both factorization rules.',
    },
    {
        'path': 'block',
        'modeling_type': 1,
        'ucell': 'a list of [n_base, [shape, ...]] layers, each shape a list starting with '
                 "'rectangle' or 'ellipse'",
        'summary': 'Shapes decomposed into axis-aligned blocks, painted in order. Edges that '
                   'are not axis-aligned become a staircase.',
    },
    {
        'path': 'smooth',
        'modeling_type': 2,
        'ucell': "a list of dict(n_base=..., shapes=[...], combine='add') layers",
        'summary': 'Shape edges enter the Fourier transform in closed form, so a tilted or '
                   'curved boundary carries no staircase error. Overlapping shapes are summed.',
    },
    {
        'path': 'boundary',
        'modeling_type': 2,
        'ucell': "the smooth spelling with combine='paint'",
        'summary': 'The smooth path with overlaps resolved into disjoint polygons first, so '
                   'later shapes cover earlier ones without losing edge exactness.',
    },
)

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
LATTICE = {
    'default': 'period, meaning diag(Lx, Ly)',
    'oblique': 'pass the two lattice vectors as the columns of `lattice`; `period` then '
               'reports their lengths',
    'raster_coordinates': 'lattice',
    #: The same two refusals as ``refused_on_oblique`` below, named rather than described.
    #: A consumer deciding *what to recommend* needs to test membership, and prose is not a
    #: membership test — these are what makes the constraint machine-checkable outside this
    #: process. Kept beside the prose so the two cannot be updated apart; a test asserts they
    #: agree.
    'paths_refused_on_oblique': ('block',),
    'factorizations_refused_on_oblique': ('li',),
    'normal_methods_refused_on_oblique': ('contour', 'spectral', 'smooth'),
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
        {
            'what': "normal_method='contour' / 'spectral' / 'smooth'",
            'why': "only 'distance' carries the Jacobian pullback that makes an index-space "
                   'gradient a Cartesian normal on a skewed cell; the smoothing '
                   'constructions are not ported there',
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
        'why': 'a scalar index per material, complex for a lossy one',
        'source': 'wavebit/_core/solve_2d.py',
    },
    {
        'case': 'uniform_anisotropic',
        'supported': True,
        'why': 'solve_for_conv reads nine permittivity slots, so a tensor filling a whole '
               'layer works, z row included. Note that every off-diagonal slot holds the '
               'negative of its component: -eps_xy, and equally -eps_xz.',
        'source': 'docs/off-diagonal-slot-convention.md',
    },
    {
        'case': 'patterned_anisotropic',
        'supported': False,
        'why': 'the solver is unchanged — what is missing is a factorization that produces '
               'the five slots for a patterned tensor',
        'source': 'docs/anisotropic-materials.md',
    },
    {
        'case': 'magnetic',
        'supported': False,
        'why': 'mu enters both the layer operator and the half-space boundary, so it is not '
               'a material substitution',
        'source': 'docs/anisotropic-materials.md',
    },
    {
        'case': 'z_coupled',
        'supported': True,
        'why': 'with eps_xz or eps_yz the second-order reduction fails, so the solve takes '
               'the first-order (4ff) formulation instead -- selected automatically by the '
               'presence of the four z-row slots, or by formulation=4ff, at about 8x. Two '
               'limits ride with it: connecting_algo must be ETM, and mu is not implemented '
               'there. Field reconstruction is (2026-08-14). Uniform layers only, for the '
               'same reason as patterned_anisotropic.',
        'source': 'wavebit/_core/solve_2d_4ff.py',
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
        'layout': 'arrays of shape (ff_y, ff_x) — y first, the transpose of the '
                  'fto=[x, y] input order — with the incident order at the centre',
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
    },
    'incidences': {
        'fields': ('res', 'res_te_inc', 'res_tm_inc'),
        'meaning': 'the solve for the incidence as specified — psi, or the complex '
                   'amplitude pair polarization=(a_s, a_p) — plus the pure-TE and pure-TM '
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
        'meaning': "which medium the illumination arrives from ('bottom' = n_bot). "
                   'n_top / n_bot / ucell order are scene labels and never move; results '
                   'are incidence-relative — R returns into the incidence medium, T exits '
                   'the far side, theta/phi are measured against the incidence medium\'s '
                   'normal — so post-processing is identical for both sides',
    },
    'jobspec': {
        'outputs_default': ('R', 'T'),
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
            'tetm': [m.name.lower() for m in Pol],
            'factorization': [m.value for m in Factorization],
            'fourier_coefficients': [m.value for m in FourierCoefficients],
            'normal_method': [m.value for m in NormalMethod],
        },
        'shape_coordinates': SHAPE_COORDINATES,
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
    return {m['source'] for m in MATERIALS}


def main():
    print(json.dumps(manifest(), indent=2, sort_keys=False))


if __name__ == '__main__':
    main()
