"""What this solver supports, in a form another program can read.

FieldWave already knows this about itself, in pieces: ``_core/enums.py`` holds the legal
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

    python -m fieldwave.manifest > fieldwave-manifest.json
"""
import json

from ._core.enums import Backend, Factorization, FourierCoefficients, Pol
from ._core.engine import (_COEFFICIENTS_IMPLEMENTED_BY, _COEFFICIENTS_REASON,
                           _FACTORIZATION_IMPLEMENTED_BY, _FACTORIZATION_REASON)

#: Bumped when the document's *shape* changes, so a consumer can refuse one it cannot read.
#: Independent of the FieldWave version, which moves whenever the content does.
MANIFEST_VERSION = '1'


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

    Read from :mod:`fieldwave._core.engine`'s own tables. A path absent from a table
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
    }


# ------------------------------------------------------------------ lattice

#: What a non-orthogonal (skewed) cell refuses, mirrored from
#: :meth:`fieldwave._core.engine.CoreSolveMixin._reject_unsupported_lattice`. Both of these
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
        'why': 'a scalar index per material, complex for a lossy one',
        'source': 'fieldwave/_core/solve_2d.py',
    },
    {
        'case': 'uniform_anisotropic',
        'supported': True,
        'why': 'the mode operator reads five permittivity slots, so an in-plane tensor '
               'filling a whole layer already works. Note the off-diagonal slot holds '
               '-eps_xy, not eps_xy.',
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
        'supported': False,
        'why': 'with eps_xz or eps_yz the elimination of E_z leaves a first-derivative term '
               'and the second-order reduction fails. The first-order (4ff) operator that '
               'would handle it is written and pinned, but nothing dispatches to it, its '
               'stack solve is not yet numerically stable, and there is no field '
               'reconstruction.',
        'source': 'fieldwave/_core/solve_2d_4ff.py',
    },
)


# ------------------------------------------------------------------ the document

def manifest() -> dict:
    """Everything above, as one JSON-round-trippable document."""
    from . import __version__

    return {
        'manifest_version': MANIFEST_VERSION,
        'fieldwave_version': __version__,
        'enums': {
            'backend': [m.name.lower() for m in Backend],
            'pol': [m.name.lower() for m in Pol],
            'factorization': [m.value for m in Factorization],
            'fourier_coefficients': [m.value for m in FourierCoefficients],
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
            'refused_on_oblique': [dict(r) for r in LATTICE['refused_on_oblique']],
        },
        'materials': [dict(m) for m in MATERIALS],
    }


def cited_sources() -> set:
    """Every repository path the manifest points at, for the staleness check."""
    return {m['source'] for m in MATERIALS}


def main():
    print(json.dumps(manifest(), indent=2, sort_keys=False))


if __name__ == '__main__':
    main()
