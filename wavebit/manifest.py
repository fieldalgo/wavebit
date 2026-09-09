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

from ._core.enums import (Answers, Backend, DEFAULT_ANSWERS, DIFFERENTIABLE_BACKENDS,
                          Factorization,
                          FourierCoefficients, ModelingPath, NormalMethod, Projector, Tetm)
from ._core.engine import (_COEFFICIENTS_IMPLEMENTED_BY, _COEFFICIENTS_REASON,
                           _FACTORIZATION_IMPLEMENTED_BY, _FACTORIZATION_REASON)
from ._core.spec import SPEC_VERSION

#: Bumped when the document's *shape* changes, so a consumer can refuse one it cannot read.
#: An **added** key is not that: a consumer reading unknown keys degrades gracefully, and
#: nothing it already reads changes meaning. ``backends`` (2026-08-26) went in without a
#: bump for exactly the reason ``docs/capability-manifest.md`` gives for why a rename alone
#: would not earn one.
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
#: '7': ``max_order`` -> ``truncation_order`` (2026-08-22), the last of the rename set the
#: '6' entry lists; it took its own number because it landed on its own commit.
#: '8': the shape origin (2026-08-31). ``cx`` / ``cy`` are measured from the centre of the cell
#: rather than from its corner, and ``shape_coordinates`` says so -- it was the bare string
#: ``'absolute'``, which said only half of what a generator has to know. Same argument as '5':
#: a consumer that kept writing ``cx = Px/2`` for a centred pillar now places it half a period
#: off, and the scene it builds solves and conserves energy. The version is what lets it
#: refuse rather than quietly answer a different question.
#:
#: The ``projector`` axis (2026-09-01) did **not** earn a bump, and the reason is the rule
#: above read the other way. It adds a key to ``paths[*]`` and to ``enums`` and changes the
#: meaning of nothing already published: its default is what the solver has always done, and a
#: consumer that never learns the key keeps generating exactly the scenes it generated before.
#: That is the "reading unknown keys degrades gracefully" case the '5' bump was contrasted
#: with. ``normal_method`` took '3' and ``answers`` took '4' on the same shape of change, but
#: both predate the rule being written down, and a bump a consumer must react to for an
#: addition it can ignore is the false alarm the rule exists to prevent.
#: '9': two modeling paths instead of three (2026-09-03). ``smooth`` and ``boundary`` were one
#: path published twice -- identical option lists, one ``structure`` spelling, and a choice
#: taken per layer rather than per solve -- so they became ``parametric`` with an ``overlap``
#: property, and ``raster`` kept its name. The ``modeling_type`` integer went with them.
#:
#: A rename alone would not earn this, and the argument that it *is* a value a consumer
#: branches on is not the one that carries it: ``path`` is a label, not something fed back to
#: the solver, so a branch that stops matching falls through rather than solving the wrong
#: thing. What carries it is ``shape_coordinates.applies_to``, which is a **membership test
#: keyed on these names** and gates a coordinate convention. A consumer holding ``'smooth'``
#: in its own source now gets ``False``, concludes absolute lengths and a cell-centre origin
#: do not apply here, and emits fractions or a corner-relative centre. That scene solves and
#: conserves energy -- the '8' failure exactly, on the convention '8' was about.
#: The ``solvers`` key (2026-09-04) did **not** earn a bump, for the reason the paragraph above
#: gives from the other side. It adds a top-level key and changes the meaning of nothing already
#: published: a consumer that never learns it keeps generating exactly the RCWA scenes it
#: generated before, and cannot reach ``wavebit.fdtd`` without reading the key describing it.
#: The near miss is ``backends``, which says jax and torch differentiate and now shares a
#: document with a NumPy-only door -- but that entry is *about the backends*, and the fdtd door
#: carries its own ``backends`` and ``differentiable`` rather than leaning on it, so nothing
#: already read acquires a second meaning. What WOULD earn a bump is the other placement:
#: folding ``paths`` / ``lattice`` / ``backends`` under ``solvers.rcwa``, which moves keys a
#: consumer already indexes.
#: Version 10 scopes FDTD capability flags to available routes, and names the
#: default/explicit-backend selectors. Consumers must select a differentiable
#: route before applying AD; backend omission still selects legacy NumPy.
#: Version 11 distinguishes fixed mesh geometry from moving internal layer
#: interfaces, whose thickness derivatives require a fixed positive total depth.
#: Version 12 distinguishes default piecewise thickness derivatives from optional
#: C1 internal-interface averaging, including its static width and clearance.
#: Version 13 adds shared axis-aligned rectangle recipes on the explicit FDTD
#: route, including moving transverse faces and optional translation averaging.
#: Version 14 adds opt-in exact area integration and rotated rectangle gradients.
#: Version 15 extends area integration with exact translated-polygon moments.
#: Version 16 permits live total depth with fixed laboratory meshes and live
#: reference-plane phases, including reflected incidence and exterior smoothing.
MANIFEST_VERSION = '16'


# --------------------------------------------------------------- how a structure is spelled

#: What a caller passes as ``structure``, and the modeling path it selects. There is no
#: separate mode switch — the type of the object *is* the choice, which is why a consumer that
#: wants to tell a user how to spell their structure has to be told this rather than inferring
#: it from a keyword.
#:
#: ``material_form`` is the second half of the same fact and is what :data:`MATERIAL_QUANTITY`
#: is stated against: one spelling names a material per region ('assignment') and one samples a
#: field ('field'), which is why only the raster takes a ``mu=`` array while the parametric
#: path carries permeability inside each ``Material``.
SPELLINGS = (
    {
        'path': 'raster',
        'structure': 'a NumPy array of relative permittivity, one 2D grid per layer',
        'material_form': 'field',
        'mu': 'a second array of the same shape, passed as mu=',
        'overlap': None,
        'summary': 'Permittivity as samples. The only path that implements both Fourier '
                   'series and both factorization rules, and the one topology optimisation '
                   'uses.',
    },
    {
        'path': 'parametric',
        'structure': 'a list of dict(background=..., shapes=[...]) layers, each shape '
                     'dict carrying material=',
        'material_form': 'assignment',
        'mu': 'inside each Material',
        #: **Not a third path.** Overlap handling used to be published as one, under the name
        #: `boundary`, beside a `smooth` that was the same path without an overlap. They
        #: carried identical option lists and one `structure` spelling, and the choice between
        #: them is taken *per layer* from the shapes' own coordinates -- so one structure can
        #: have its first layer resolved and its second transformed in closed form, which two
        #: sibling path entries cannot express. It is a property of this path instead.
        'overlap': {
            'allowed': True,
            'decided': 'per layer, from the shapes\' own coordinates',
            'selected_by': 'the overlap itself, not a keyword',
            'resolution': 'shapes that intersect are resolved into disjoint polygons before '
                          'the transform, so a later shape covers an earlier one and the '
                          'polygon transform keeps every edge exact',
            'why_not_summed': "summing each shape's transform would give the overlap both "
                              'materials, which is not a material',
            'observed_at': 'geometry.boundary_plans -- a layer\'s slot holds a plan if it '
                           'resolved, None if it stayed on the closed form',
            'source': 'docs/boundary-modeling.md',
        },
        'summary': 'Shape edges enter the Fourier transform in closed form, so a tilted or '
                   'curved boundary carries no staircase error at any truncation order. '
                   'Shapes are transformed one by one, or, where they overlap, as the '
                   'disjoint polygons that overlap resolves into.',
    },
)


#: **Which quantity a material number is.** Machine-readable, because the answer changed:
#: every material slot took a refractive index until 2026-08-20 and takes relative
#: permittivity now. A consumer that generated scenes against the old answer would keep
#: generating numbers that are still *legal* -- ``3.5`` is a fine permittivity -- so nothing
#: downstream raises and no energy check fires. Prose cannot be tested for; this can, which is
#: the whole reason it is a dict of names rather than a sentence in ``summary``.
#:
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
    #: No `retired` map since 2026-08-26: the retired-spelling machinery it was derived
    #: from is gone. wavebit has never been released, so the reader those messages
    #: addressed does not exist yet; the renames belong in the first release's notes.
    #: Removed without a MANIFEST_VERSION bump, following `c08e682`, which took
    #: `outputs_default` out of `results.jobspec` and left the version at '6'.
    'source': 'docs/conventions.md',
}

#: How a shape dict's numbers are read, on the parametric path. It says two things, and each
#: is here because getting it wrong is not an error: a fraction where a length belongs solves a
#: different, much smaller structure, and a corner-relative centre solves a correctly sized one
#: in the wrong place. Both raise nothing.
#:
#: ``lengths``: absolute, in the units of ``period``, never fractions of it.
#: ``origin``: what ``cx`` / ``cy`` are measured from -- the centre of the cell,
#: ``0.5*(a1 + a2)``, which is also what an omitted centre means. Corner-relative until
#: 2026-08-31; moved to match S4, the only other solver that takes parametric shapes.
#: The **raster** path has no such coordinate and did not move: an array is the geometry.
SHAPE_COORDINATES = {
    'lengths': 'absolute',
    'origin': 'cell_centre',
    'origin_expression': '0.5*(a1 + a2)',
    'omitted_centre': 'cx = cy = 0',
    'applies_to': ('parametric',),
}


# ------------------------------------------------------------------ per-path option support

def _options_for(path):
    """The two Fourier axes for one modeling path, as implemented rather than as documented.

    Read from :mod:`wavebit._core.engine`'s own tables. A path absent from a table
    implements every value of that axis and takes the option as given; a path present in one
    implements exactly the value it names, and the reason says whether the others are unbuilt
    or unbuildable.
    """
    factorization = _FACTORIZATION_IMPLEMENTED_BY.get(path)
    coefficients = _COEFFICIENTS_IMPLEMENTED_BY.get(path)
    return {
        'factorization': (
            [m.value for m in Factorization] if factorization is None
            else [factorization.value]),
        'factorization_reason': (
            None if factorization is None else _FACTORIZATION_REASON[path]),
        'fourier_coefficients': (
            [m.value for m in FourierCoefficients] if coefficients is None
            else [coefficients.value]),
        'fourier_coefficients_reason': (
            None if coefficients is None else _COEFFICIENTS_REASON[path]),
        # The third axis exists only where a raster normal field is built. On the raster
        # path the constructions are probed rather than assumed -- 'contour' needs the
        # optional contourpy dependency, and offering an option this environment cannot
        # build is exactly the drift the manifest tests exist to prevent -- and the axis is
        # read only under factorization='nvf' (explicitly set anywhere it has no reader, it
        # is refused rather than ignored). The parametric path builds its normal field
        # from the geometry itself, so it has no raster construction to select.
        'normal_method': _raster_normal_methods() if path == ModelingPath.RASTER else [],
        'normal_method_reason': (
            "read only under factorization='nvf'" if path == ModelingPath.RASTER
            else 'the parametric path builds its normal field from the geometry itself; '
                 'there is no raster construction to select'),
        # A fourth axis, and deliberately *not* four more `normal_method` values. That one
        # says how the field is built; this says how the projector is formed from it, so the
        # two compose -- publishing them as one list of eight would tell a consumer there are
        # eight independent choices where there are two.
        'projector': ([m.value for m in Projector] if path == ModelingPath.RASTER else []),
        'projector_reason': (
            "read only under factorization='nvf', beside normal_method"
            if path == ModelingPath.RASTER
            else 'the parametric path builds its normal field from the geometry, which '
                 'carries no magnitude for the circular limit to read'),
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
    #: Empty since the block path was removed: it was the only one refused here, because its
    #: modeller rasterised shapes onto a rectangular grid of its own. Both surviving paths
    #: take a skewed cell.
    'paths_refused_on_oblique': (),
    'factorizations_refused_on_oblique': ('li',),
    'normal_methods_refused_on_oblique': (),
    'refused_on_oblique': (
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
               'both quantities land on one geometry. The 1D and conical paths carry it too '
               'since 2026-09-01, where the same duality maps the TE branch onto the TM one; '
               'those operators take the diagonal three slots and refuse mu_xy / mu_yx by '
               'name, having no off-diagonal position for them. The first-order 4ff solve '
               'takes no mu at all and likewise refuses rather than ignores. Field '
               'reconstruction carries mu on every path: H_z is Faraday z row, so it divides '
               'by mu_zz^-1, and on the 2D path it came back scaled by mu_zz until '
               '2026-08-20 while every other component was exact.',
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
               'exists for it either, which is the harder half. Refused rather than dropped '
               'since 2026-08-27: the Popov-Neviere builder used to return matrices '
               'bit-identical to the same layer with its z row zeroed.',
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
        'layout': 'arrays of shape (n_orders_v, n_orders_u) — y first, the transpose of the '
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
                   'all -- it is 4.25 matrices of side 2*n_orders_u*n_orders_v per layer on the '
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

# ------------------------------------------------------------------ the two doors

#: **Two solvers, one document.** Everything above is the RCWA solve's parameter space, because
#: for most of this document's life that was the only solve there was. ``wavebit.fdtd`` is a
#: second door: it reads the same ``Stack`` and ``Incidence``, returns the same ``Result``, and
#: takes almost none of the axes ``paths``, ``lattice`` and ``enums`` publish. Published as one
#: key naming BOTH doors rather than as a bare ``fdtd`` entry, because what a bare entry would
#: leave a consumer to guess is which of the sections above apply to which door -- ``backends``
#: says jax and torch differentiate, and beside an unscoped entry that reads as a promise about
#: every solve here.
#:
#: Same admission test as everything else: each claim is held against runtime by
#: ``tests/test_manifest.py``. Both entry points are called, every name in ``refused`` is asked
#: for and raises, and ``sections`` names keys of this very document.
SOLVERS = {
    'rcwa': {
        'entry_point': 'wavebit.rcwa',
        'domain': 'frequency',
        'returns': 'a solver object; call .solve()',
        #: Derived from the same enum ``enums.backend`` is, so the two spellings cannot drift.
        'backends': tuple(m.name.lower() for m in Backend),
        'differentiable': True,
        'sections': ('paths', 'shape_coordinates', 'lattice', 'materials', 'enums',
                     'backends', 'material_quantity', 'results'),
    },
    'fdtd': {
        'entry_point': 'wavebit.fdtd',
        'domain': 'time',
        'returns': 'a Result directly, or a list for a wavelength vector; optional diagnostics '
                   'return a (result, diagnostics) pair',
        #: These flags describe available routes. The selector below is part
        #: of the contract: omission does not select the common NumPy engine.
        'backends': tuple(m.name.lower() for m in Backend),
        'differentiable': True,
        'default_route': 'legacy_numpy',
        'routes': {
            'legacy_numpy': {
                'selector': 'backend omitted or None',
                'backends': ('numpy',),
                'differentiable': False,
                'incidence_sides': ('top',),
                'source': 'docs/fdtd.md',
            },
            'common': {
                'selector': 'explicit backend=0/1/2 or numpy/jax/torch',
                'backends': tuple(m.name.lower() for m in Backend),
                'differentiable_backends': ('jax', 'torch'),
                'differentiable': True,
                'incidence_sides': ('top', 'bottom'),
                'required': ('resolution',),
                'differentiable_parameters': ('theta', 'phi', 'raster epsilon', 'raster mu',
                                              'electric pole coefficients', 'half-space epsilon',
                                              'half-space mu', 'fixed-total layer thickness', 'independent layer thickness (variable_depth)',
                                              'rectangle centre', 'rectangle size', 'rectangle angle (area)', 'shape epsilon/mu'),
                'static_parameters': ('periods', 'shape types/count/paint order', 'nominal mesh depth', 'wavelength',
                                      'mesh_indices', 'mesh_thickness', 'variable_depth', 'layer_smoothing', 'shape_smoothing', 'shape_integration', 'truncation_order'),
                'parametric_shapes': ('rectangle', 'square'),
                'parametric_requires': ('two rectangular periods', 'subpixel=True'),
                'shape_integration': ('separable', 'area'),
                'separable_requires': 'static angle=0',
                'area_requires': 'at most one rectangle per layer; projected extents within periods',
                'area_smoothing': 'exact bilinear polygon moments for box translations; no quadrature order',
                'area_averaging': 'exact polygon/cell areas; arithmetic transverse material averages; full y sampling',
                'parametric_refused': ('circles/ellipses', 'raster mu', 'dispersion labels'),
                'shape_constraint': 'positive lengths no larger than periods; periodic later-shape painting; '
                                    'gradients piecewise across topology changes',
                'shape_smoothing_constraint': 'static scalar or (x,y) widths in [0,period]; '
                                              'averages translations before sampling; changes the sampled profile',
                'array_angles_require': ('truncation_order',),
                'array_phi_requires': 'Jones or psi polarization; full y-period sampling including zero',
                'array_half_spaces_require': ('mesh_indices', 'truncation_order'),
                'backend_thickness_requires': ('mesh_thickness', 'subpixel=True'),
                'thickness_constraint': 'positive lengths with the same total as mesh_thickness; '
                                        'variable_depth=True instead permits a changing total with live reference-plane phases',
                'variable_depth_requires': ('mesh_thickness', 'subpixel=True', 'interfaces and smoothing supports clear of fixed diffraction stencils'),
                'variable_depth_coordinates': 'fixed laboratory mesh/source/monitors/records; reference faces at 0 and sum(thickness) for both incidence sides',
                'positive_layer_smoothing_requires': ('mesh_thickness', 'subpixel=True'),
                'layer_smoothing_constraint': 'static nonnegative physical width; end layers at least half its width; '
                                              'averages internal faces and the moving exterior face with variable_depth; changes the material profile; '
                                              'nonuniform reference-phase slopes remain piecewise',
                'results_incidences': ('given', 's_inc', 'p_inc'),
                'verified_devices': ('cpu',),
                'source': 'docs/guide/time-domain.md',
            },
        },
        'sections': ('material_quantity', 'results'),
        'structure': 'an array of relative permittivity, shape (n_layers, Ny, Nx); '
                     'Ny = 1 is a grating invariant along y and anything larger is a crossed '
                     'one. The explicit common route also accepts rectangle/square recipes, with '
                     'rotation through opt-in area integration, using the shared parametric conventions; the default legacy route refuses them',
        'answers': ('pure te/tm at any theta in the classical mount (phi = 0)',
                    'a conical mount through psi/chi or a complex Jones pair',
                    'a crossed grating, the raster periodic along both axes',
                    'drude/lorentz dispersion and permeability, on all of those',
                    'a band of wavelengths covered by one run'),
        #: Names, not prose: a consumer deciding what to send here needs a membership test, and
        #: prose is not one -- the same argument ``normal_methods_refused_on_oblique`` is here
        #: for. Each is refused rather than approximated because each returns a *plausible*
        #: number if it is quietly ignored, and a plausible wrong number from the cross-check
        #: solver is worse than no cross-check at all.
        'refused': ('a complex theta', 'a complex permittivity', 'Re(eps) <= 0',
                    'lattice'),
        #: ``given`` is guaranteed on both routes. The common route additionally
        #: guarantees both pure-incidence columns for every solve.
        'results_incidences': ('given',),
        'source': 'docs/fdtd-alpha.md',
    },
}

# ------------------------------------------------------------------ the document

#: What each backend can do that is not the same everywhere. One entry so far, and it is the
#: one a caller has to know *before* building: only two of the three differentiate, and asking
#: after the fact means an exception from a method that exists only to say no.
#:
#: Read off `_DIFFERENTIABLE` rather than restated here, and
#: `test_manifest.py::test_differentiability_claims_match_the_solver` holds the published answer
#: against `hasattr(solver, 'grad')` on a real solver of each backend -- so a backend that gains
#: or loses autodiff fails the suite instead of going stale in a consumer.
def _backend_capabilities() -> dict:
    import importlib.util

    out = {}
    for member in Backend:
        # `find_spec` rather than an import: reading this document must not pull an optional
        # library into the caller's process, and importing the JAX backend lifts
        # `jax_enable_x64` process-wide. The manifest is what a caller reads to decide whether
        # to install JAX; it cannot be the thing that changes their JAX.
        module = {Backend.NUMPY: 'numpy', Backend.JAX: 'jax', Backend.TORCH: 'torch'}[member]
        out[member.name.lower()] = {
            'available': importlib.util.find_spec(module) is not None,
            'differentiable': member in DIFFERENTIABLE_BACKENDS,
        }
    return out


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
            'projector': [m.value for m in Projector],
            'answers': [m.value for m in Answers],
        },
        'backends': _backend_capabilities(),
        'shape_coordinates': SHAPE_COORDINATES,
        'material_quantity': _jsonable_material_quantity(),
        'paths': [
            dict(spelling, **_options_for(ModelingPath(spelling['path'])))
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
        'solvers': _jsonable_solvers(),
    }


def _jsonable_material_quantity() -> dict:
    """:data:`MATERIAL_QUANTITY` with its tuples as lists, so it round-trips through JSON."""
    out = dict(MATERIAL_QUANTITY)
    out['conversions'] = list(out['conversions'])
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


def _jsonable_solvers() -> dict:
    """``SOLVERS`` with its tuples as lists, so the document round-trips through JSON."""
    return {door: {k: list(v) if isinstance(v, tuple) else v for k, v in entry.items()}
            for door, entry in SOLVERS.items()}


def cited_sources() -> set:
    """Every repository path the manifest points at, for the staleness check.

    Read from the published structures rather than listed here, so a citation added to one of
    them joins the check by being added -- which is what a hand-kept list would not do.
    """
    return ({m['source'] for m in MATERIALS} | {MATERIAL_QUANTITY['source']}
            | {s['overlap']['source'] for s in SPELLINGS if s['overlap'] is not None}
            | {e['source'] for e in SOLVERS.values() if 'source' in e})


def main():
    print(json.dumps(manifest(), indent=2, sort_keys=False))


if __name__ == '__main__':
    main()
