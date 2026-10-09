"""Typed enumerations for the public API.

``Backend`` and ``Tetm`` are ``IntEnum``s equal to the integer codes the solver already uses.
``Tetm`` names the two states the ``tetm`` declaration selects, and its integer value is the
index the pure 1D solves dispatch on (``CoreSolveMixin.tetm_index``). Users write the word --
``tetm='te'`` or ``tetm=Tetm.TE`` -- because the declaration has exactly two values and a
number invited a third meaning: the retired ``pol`` accepted the interior of ``[0, 1]``,
where it was a linear angle in a bad unit rather than a declaration at all.

``ModelingPath`` is the one nobody sets. It is reported rather than passed -- the type of
``structure`` selects it -- and it is what the other axes are conditioned on, which is why it
is a member of this module rather than an integer on the solver.

``FourierCoefficients`` and ``Factorization`` are ``str`` enums, equal to the lowercase
strings the solver stores, so ``factorization=Factorization.NVF`` and ``factorization='nvf'``
are the same thing.

The two are independent choices, and keeping them as separate keywords is what makes all four
of their pairings reachable — see :func:`resolve_fourier_options`.

``NormalMethod`` and ``Projector`` are the same shape of pair one level down, over the raster
NVF normal field: the first says how the field is *built*, the second what is done *with* it.
Neither is a value of the other, so the eight pairings are reachable and the two lists stay
four long and two long instead of one list of eight.

``Nesting`` sits one level down on the other side: it is read only under
``factorization='li'``, where it says in which order Li's one-dimensional rules are nested for
each field component. See :class:`Nesting`.
"""
import re
from enum import Enum, IntEnum


class Backend(IntEnum):
    """Array backend for :func:`wavebit.rcwa`."""
    NUMPY = 0
    JAX = 1
    TORCH = 2


#: Which backends differentiate through a solve. The one table: the backend classes set their
#: own ``_DIFFERENTIABLE`` from it, and ``manifest.py`` publishes it -- so the document can be
#: built **without importing an optional backend to ask one**. That matters more than it
#: sounds: importing the JAX backend lifts ``jax_enable_x64`` process-wide, and the manifest is
#: exactly what a caller reads to decide whether to install JAX at all. Reading it must not
#: change their process.
#:
#: ``tests/test_manifest.py::test_differentiability_claims_match_the_solver`` holds this against
#: a real solver of each backend, so the table cannot drift from the classes it describes.
DIFFERENTIABLE_BACKENDS = frozenset({Backend.JAX, Backend.TORCH})


class Tetm(IntEnum):
    """The two states of the ``tetm`` declaration; the value is ``tetm_index``."""
    TE = 0
    TM = 1


class ModelingPath(str, Enum):
    """How a layer is described -- the choice every other axis is conditioned on.

    There is no keyword to set this. The *type of the object* passed as ``structure`` is the
    choice: an array selects :attr:`RASTER`, a list of layer dicts selects :attr:`PARAMETRIC`.
    :attr:`~wavebit._core.engine.SolverBase.modeling_path` reports which was read.

    The two names sit on one axis -- what the input *is* -- because that is the only thing a
    caller picks. ``'parametric'`` rather than ``'analytic'`` for that reason: closed form is
    not exclusive to it (``fourier_coefficients='integrated'`` is closed form over a raster's
    constant blocks), and the pair as spelled is the one the inverse-design literature already
    uses -- a permittivity field on a grid for topology optimization, a handful of geometric
    parameters for parametric optimization.

    **An overlap is not a third path.** Where a parametric layer's shapes intersect, the layer
    is resolved into disjoint polygons first (:mod:`wavebit._core.boundary`) because summing
    each shape's transform would give the overlap both materials. That choice is taken **per
    layer** from the shapes' own coordinates, so one ``structure`` can have its first layer
    resolved and its second transformed in closed form -- which is why it cannot be a member
    here, and why what records it is ``geometry.boundary_plans`` rather than this. See
    ``docs/boundary-modeling.md``.

    Replaced a bare ``modeling_type`` integer whose values were ``0`` and ``2``: ``1`` was the
    block path, removed in ``ad415b4``, and the hole outlived it.
    """
    RASTER = 'raster'          # a 2D array of sampled permittivity, one grid per layer
    PARAMETRIC = 'parametric'  # a background material plus a list of shapes carrying material=


class FourierCoefficients(str, Enum):
    """How a layer's Fourier coefficients are obtained from the raster."""
    SAMPLED = 'sampled'      # discrete Fourier series off the (optionally upsampled) raster
    INTEGRATED = 'integrated'  # closed-form series over the compressed constant blocks


class Factorization(str, Enum):
    """Which rule closes the products of discontinuous functions.

    ``'laurent'`` is the direct rule for every product, and it is opt-in: nothing resolves to it.
    Each ``eps * E`` is closed with the plain convolution matrix ``[[eps]]`` whatever is
    continuous across the interface -- the formulation RCWA started from (Moharam & Gaylord,
    JOSA 71, 811, 1981) and the one Li showed converges slowly wherever a discontinuous field
    component meets a discontinuous permittivity (L. Li, JOSA A 13, 1870, 1996). It is offered
    so that a result can be compared like for like with a solver that uses it, fmmax's
    ``Formulation.FFT`` among them; see :func:`wavebit._core.factorization.laurent_conv_matrices`
    for the slots it fills.
    """
    LI = 'li'    # Li's rules on the x/y axes
    NVF = 'nvf'  # normal vector field (Schuster et al. 2007); a superset of Li
    LAURENT = 'laurent'  # the direct rule for every product (Moharam & Gaylord 1981); opt-in


class NormalMethod(str, Enum):
    """How the raster NVF normal field is constructed from the pixel grid.

    Only the raster path with ``factorization='nvf'`` reads this — the shape paths build
    their normal fields from the geometry itself and have nothing to choose here.

    All four run eagerly in NumPy/SciPy and hand the solve a constant field, whatever the
    backend (the normal field is frozen geometry — see
    :func:`wavebit._core.factorization.normal_vector_field`). What separates them is how the
    staircase is read: the raw distance-transform gradient quantizes to the {0, 45} degree
    directions integer central differences can spell, and the other three smooth the
    *indicator* before differentiating, which is the one move every raster-native
    construction in the literature shares (``docs/nvf-normal-methods.md`` records the
    measurements that rank them; ``docs/nvf-upsampling.md`` diagnosed the defect).
    """
    DISTANCE = 'distance'  # unsigned distance-to-interface gradient (the original)
    CONTOUR = 'contour'    # subpixel level-set polylines, exact distance-to-curve direction
    SPECTRAL = 'spectral'  # band-limited spectral gradient at the interface, bulk-propagated
    SMOOTH = 'smooth'      # signed distance, derivative-of-Gaussian (rounds corners; see docs)


class Projector(str, Enum):
    """How the NVF projector ``P`` is formed *from* the normal field — a separate axis.

    :class:`NormalMethod` chooses how the field is **built**; this chooses what is done
    **with** it, so the two compose rather than multiply into one list of eight.

    ``'linear'`` is ``P = n n^T``, built from the field's *direction* alone. It is what NVF
    has always used and what every number in the repository was measured on.

    ``'circular'`` builds ``P = J J^H`` from a Jones vector that carries the field's
    *magnitude* as well::

        J = (n cos(phi) + i n_perp sin(phi)),   phi = (pi/8) (1 + cos(pi |n|))

    with ``|n|`` scaled so its maximum over the cell is 1 and ``n_perp = (-n_y, n_x)``. At
    full magnitude ``phi = 0`` and ``P`` is exactly ``n n^T`` again; as ``|n| -> 0`` the
    Jones vector becomes exactly circular and ``P`` **loses its direction**:
    ``P_xx = P_yy = 1/2``, ``P_xy = -i/2``, ``P_yx = +i/2``.

    That limit is the point. On the medial axis — the ridge where every boundary point is
    equidistant — the direction is not merely hard to compute, it is undefined, and both
    values ``'linear'`` can report there are arbitrary. ``'circular'`` reaches a
    direction-independent answer as a *limit*, so there is no threshold anywhere and no
    "is this the medial axis?" test to get wrong.

    Raster NVF supports both. Shape NVF also supports circular
    projection through a positive normal-dyad lift, rather than the raster
    magnitude mapping above. Disjoint circles and ellipses use a periodic
    level-set dyad with an isotropic extension away from their boundaries.
    Overlapping curves correct that extension near their polygon chords so
    the normal matches the geometry used by the material transform.
    All projector selections require ``factorization='nvf'``.

    **The scalar NVF path only.** The Popov-Neviere tensor builders
    (:func:`~wavebit._core.factorization.pn_eps_conv_matrices` and its ``mu`` twin) never form
    a ``C_xx``-style projector series to deform: they build ``n n^T`` *pointwise* inside a
    different rearrangement and transform once at the end, because doing it any other way
    costs a uniform slab 0.5 (see :func:`~wavebit._core.factorization._pn_fields`). Carrying
    the limit there is a separate derivation, not this one with an argument threaded through,
    and no solver path reaches those builders today. So they take no ``nvf_projector``, which is
    also why nothing has to refuse one.
    """
    LINEAR = 'linear'      # P = n n^T, direction only (the original)
    CIRCULAR = 'circular'  # P = J J^H, Antos' complex basis; direction-free as |n| -> 0


class Nesting(str, Enum):
    """The order Li's rules are nested in, per field component — scalar ``factorization='li'``.

    A crossed grating needs a rule along each lattice axis, and applying two one-dimensional
    rules means transforming along one axis *inside* the other (L. Li, JOSA A 14, 2758, 1997).
    With ``T_u`` / ``T_v`` the Toeplitz (convolution) matrix along each axis and ``epx`` the slot
    acting on ``E_x``::

        slot    'fixed'                        'symmetric'
        epx     ( T_u( [T_v(eps)]^-1 ) )^-1    T_v( [T_u(1/eps)]^-1 )
        epy     T_u( [T_v(1/eps)]^-1 )         T_u( [T_v(1/eps)]^-1 )     (the same)
        epz_i   ( T_uv(eps) )^-1               ( T_uv(eps) )^-1           (the same)

    ``'fixed'`` nests ``v`` inside ``u`` for every slot. It is what the raster path has always
    built, and the order the Cartesian shape and tensor Li builds were written in --
    RETICOLO's ``parm.res1.li=1`` -- and it is not covariant under relabelling x and y: on a
    structure with ``eps(x, y) == eps(y, x)`` the two polarizations disagree at finite
    truncation. ``'symmetric'`` puts the inverse-rule axis inside for each slot (Li's own
    construction, RETICOLO's ``li=0``), so ``epx`` is the x<->y mirror image of ``epy`` and the
    relabelling holds to round-off. Both are Li's rules on a staircase and converge to the same
    answer; they differ in how they get there. On a y-invariant (1D) structure both reduce to
    Li's 1D inverse rule exactly. See ``docs/factorization-terms.md`` §6 for what separates
    them, measured.

    Read by the two scalar Li builds: a raster under ``factorization='li'`` (either coefficient
    path, the permeability included) and the parametric path's axis-aligned rectangles under
    ``factorization='li'``. Explicitly set anywhere else it is refused rather than ignored, as
    ``raster_normal_method`` is off raster NVF and ``nvf_projector`` off NVF: under
    ``factorization='nvf'`` or ``'laurent'`` any value (neither nests rules), and on a tensor
    material
    ``'symmetric'`` (tensor Li, :mod:`wavebit._core.tensor_li`, builds the fixed nesting only).
    The solver option that takes it is ``li_nesting``.
    """
    FIXED = 'fixed'          # inner v, outer u for every slot (the original; RETICOLO li=1)
    SYMMETRIC = 'symmetric'  # the inverse-rule axis inside, per slot (Li 1997; RETICOLO li=0)


class Formulation(str, Enum):
    """Which layer formulation the 2D solve uses — how many fields per order it carries.

    ``'2ff'`` reduces a layer to ``d2E_t/dz2 = Omega2 E_t``, which exists only while the
    permittivity's ``z`` row and column are empty. ``'4ff'`` carries the full transverse state
    ``(Ex, Ey, Hx, Hy)`` as a first-order system, which is what a z-coupled (magneto-optic,
    Voigt) tensor needs — at roughly 8x the cost, because the eigenproblem doubles.

    The members are named for the *value* they carry, not for the ODE order: ``TWO_FIELD``
    is the second-order form in two field components and ``FOUR_FIELD`` the first-order form
    in four, so the pair used to read inverted as ``SECOND = '2ff'`` / ``FIRST = '4ff'``.

    ``'auto'`` is the default and reads the problem: a solve handed any of the four z-row
    convolution slots takes ``'4ff'``, everything else ``'2ff'``. The override exists because
    forcing ``'4ff'`` on a problem that does not need it is how the two are cross-checked
    (``docs/anisotropic-materials.md`` §8); forcing ``'2ff'`` on a z-coupled tensor raises
    rather than dropping the row, because dropping it answers a different problem while
    conserving energy to machine precision.
    """
    AUTO = 'auto'          # read the slots and decide
    TWO_FIELD = '2ff'      # second-order ODE, two transverse E components; refuses a z row
    FOUR_FIELD = '4ff'     # first-order ODE, four transverse components; always possible, ~8x


class Answers(str, Enum):
    """What a solve keeps the state to answer afterwards — a *declaration*, not a value.

    Like ``tetm``, this says what is not a question of this problem rather than what its
    answer happens to be. ``tetm`` declares a degree of freedom away; this declares a
    *question* away, and the state that would have answered it is never kept.

    The three are a ladder, not a menu — each rung answers everything below it:

    ==================  ==========================================================
    ``'efficiency'``    ``R`` / ``T`` per order, and nothing else
    ``'amplitude'``     **default** — the Fresnel amplitudes ``r_s``/``r_p``/``t_s``/
                        ``t_p``, their s/p-resolved efficiencies, and the pure-TE /
                        pure-TM incidence variants ``s_inc`` / ``p_inc``
    ``'field'``         the per-layer modes and amplitudes ``reconstruct_field`` walks
    ==================  ==========================================================

    The seam that matters is between the last two, and it is not a matter of taste. The
    amplitudes are boundary data — a few vectors, already solved for — while the field state is
    ``4.25`` matrices of side ``dim`` *per layer* on the 2D path (``3.75`` conical, ``5`` pure
    1D), with ``dim = 2*n_orders_u*n_orders_v`` on the coupled paths. At ``truncation_order=(15, 15)``
    that is one 56 MB block per matrix. The 2D figure is the one to quote: ``3.75`` is what you
    get by assuming that path's ``W`` is the conical one's half-height block, and it is not. A
    phase-target optimization needs the amplitudes and never calls ``reconstruct_field``;
    putting both behind one flag would have made it buy the second to get the first.

    Why a declaration rather than a runtime argument to the solve: it is known when the
    problem is set up, which is when the caller knows what they are solving for; it lands in
    the JAX pytree's ``aux_data``, where it is static under ``jit`` rather than a traced
    value that cannot branch; and asking for something it excludes is a contradiction the
    code can refuse by name (see :meth:`wavebit._core.engine.CoreSolveMixin.answers`) instead
    of silently satisfying at every scale.
    """
    EFFICIENCY = 'efficiency'
    AMPLITUDE = 'amplitude'
    FIELD = 'field'

    def covers(self, other):
        """Does a solve declared for this rung also answer ``other``?

        The ladder is ordered by what each rung *retains*, so this is the whole of the
        containment relation — a solve declared for the field answers amplitudes too, and no
        caller has to spell that out.
        """
        return _ANSWERS_LADDER.index(self) >= _ANSWERS_LADDER.index(answers_of(other))


#: The ladder, cheapest first. A new rung is added here and to :class:`Answers` together;
#: ``covers`` reads the order from this tuple and nothing else does.
#:
#: A ``'scattering'`` rung was drafted here and then not built, which is worth recording
#: because the reasoning is what keeps the ladder three rungs long. The SMM cascade builds
#: ``S``, the per-layer ``S_layer``, the below-plane reflections ``gamma`` and the gap<->layer
#: coefficients ``AB`` (:func:`~wavebit._core.scattering._ab_matrices`) -- the same "computed
#: and thrown away" shape the field state had before ``answers`` existed. What shipped instead
#: is :class:`~wavebit._core.scattering.ScatteringMatrix` on ``solver.scattering``, riding
#: ``AMPLITUDE`` rather than a rung of its own, because the whole-stack ``S`` is four matrices
#: whatever the depth and an ``S`` *is* amplitude information. The rest stayed unshipped:
#:
#: * **The gap basis is a gauge.** Its waves are chosen per order
#:   (:func:`~wavebit._core.scattering._gap_waves`); re-run a solve with them scaled by 2.7
#:   and the observables hold (``T`` moves 0.9e-15 to 4.1e-15, ``r_s`` 0.0 to 2.0e-15, on a TM,
#:   a conical and a 2D stack) while ``S_layers[0].S11`` moves 0.44 to 0.80 and ``gamma[0]``
#:   0.44 to 0.80, each against its own magnitude. The whole-stack blocks move 3.6e-15 to
#:   7.0e-15 -- they are gauge-*free*, because the entry and exit interface factors cancel the
#:   gap between them and leave the two half-spaces as the ports. That is the whole reason
#:   ``ScatteringMatrix`` is directly readable and the per-layer blocks would not be. (Measured
#:   2026-10-09 on the gap of that date; the 2026-08-22 measurement, on the gap of
#:   ``eps = 1 + k_par^2``, ``kz = 1`` swapped for ``2.7 + k_par^2``, found the same.)
#: * **So ``AB`` would have to ship with the per-layer blocks, or they do not ship.**
#:   ``_ab_matrices`` is the gap -> half-space conversion, and exposing gauge numbers while
#:   withholding it would document a footgun and withhold the safety.
#: * **``S_ref`` / ``S_trn`` should not ship at all.** They *are* that conversion -- converted
#:   to a physical (E, H) transfer ``S_ref`` is exactly the identity -- and their ``S11`` moved
#:   0.34 to 0.81 and 0.48 to 0.82 under the same gauge change while encoding nothing.
_ANSWERS_LADDER = (Answers.EFFICIENCY, Answers.AMPLITUDE, Answers.FIELD)


DEFAULT_FOURIER_COEFFICIENTS = FourierCoefficients.SAMPLED
DEFAULT_FACTORIZATION = Factorization.LI
DEFAULT_FORMULATION = Formulation.AUTO
DEFAULT_NORMAL_METHOD = NormalMethod.DISTANCE
#: ``n n^T``. Not a taste call: every measurement in ``docs/`` and every golden was taken on
#: it, and the circular limit is a *trade* rather than a free win -- it removes the medial
#: axis's arbitrariness and costs accuracy against an exact circle (3.18e-5 -> 3.95e-5 at
#: 512^2). A default that moved would move both.
DEFAULT_PROJECTOR = Projector.LINEAR
#: The nesting every raster ``'li'`` number in the repository was measured on, and every golden
#: was taken with. Kept rather than flipped to ``'symmetric'`` so that nothing moves; which of
#: the two should be the default is the maintainer's call -- ``docs/factorization-terms.md`` §6
#: has the measurements it would rest on.
DEFAULT_NESTING = Nesting.FIXED
#: Amplitudes, not fields. The field state is three orders of magnitude larger and most
#: large runs -- an optimization sweep against a fixed objective -- never reconstruct one.
DEFAULT_ANSWERS = Answers.AMPLITUDE


def _one_of(enum_cls, value, keyword):
    """Coerce ``value`` to a member of ``enum_cls``, case-insensitively."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value).lower())
    except (ValueError, AttributeError):
        raise ValueError(
            f"{keyword}={value!r} is not valid; "
            f"expected one of {[m.value for m in enum_cls]}") from None


def coefficients_of(value):
    """Validate a ``fourier_coefficients`` value."""
    return _one_of(FourierCoefficients, value, 'fourier_coefficients')


def factorization_of(value):
    """Validate a ``factorization`` value."""
    return _one_of(Factorization, value, 'factorization')


def formulation_of(value):
    """Validate a ``formulation`` value."""
    return _one_of(Formulation, value, 'formulation')


_NVF_OPTION_RENAMES = {
    'normal_method': 'raster_normal_method',
    'normal_grid': 'shape_nvf_grid_size',
    'projector': 'nvf_projector',
}


def nvf_option_rename_hint(names):
    """Explain the pre-release NVF rename without accepting an obsolete spelling."""
    renamed = [f'{name!r} -> {_NVF_OPTION_RENAMES[name]!r}'
               for name in sorted(names) if name in _NVF_OPTION_RENAMES]
    return (' Renamed NVF option(s): ' + ', '.join(renamed) + '.') if renamed else ''


#: The slots that take a material: a :class:`~wavebit._core.domain.Material`, or its scalar
#: shorthand, which is **relative permittivity** (``manifest.MATERIAL_QUANTITY``) -- the two
#: half-spaces, ``structure`` in both encodings, and the parametric layer's ``background`` and
#: shape's ``material`` (the manifest's ``automatic_slots``, which ``tests/test_spec.py`` holds
#: this set to). They read as permittivity by what they hold, since their names do not say so.
_MATERIAL_SLOTS = frozenset({'top', 'bot', 'structure', 'background', 'material'})

#: The words an index symbol that also means something else has to stand beside to be read as
#: an index: the material slots, the other names a half-space or a background goes by, and the
#: words that name a part of an index (``n_eff``, ``n_real``). ``n_top``, ``n_sub``, ``n_bg`` and
#: ``nd_top`` are an index; ``n_orders``, ``n_threads``, ``nG`` and ``nd_array`` are a count or a
#: dimension, and a count is matched like any other name. ``core``, ``host``, ``env``, ``mat``
#: and ``layer`` are left out on purpose: ``n_cores``/``ncore``, ``nhost``, ``nmat`` and
#: ``n_layer`` are counts at least as often as indices.
_SLOT_WORDS = _MATERIAL_SLOTS | frozenset({
    'bottom', 'sub', 'substrate', 'sup', 'super', 'superstrate', 'cover', 'clad', 'cladding',
    'ambient', 'air', 'medium', 'bg', 'film', 'inc', 'incident', 'ref', 'trn', 'struct'})
_INDEX_PART_WORDS = frozenset({'eff', 'real', 'imag', 're', 'im'})

#: What one *word* of a name denotes, where it denotes a material quantity on its own. Words,
#: not names: ``top_index`` is ``top`` + ``index`` and ``refractiveIndex`` is ``refractive`` +
#: ``Index``, so the rule covers every spelling assembled from these rather than a list of the
#: ones seen so far. ``ri`` is the abbreviation (``RI_top``); ``n`` with a number (``n1``) is an
#: index however it is used.
_QUANTITY_WORDS = (
    ('index', re.compile(r'n\d+|nk|neff|ri|index(es)?|indices|ior|refract\w*')),
    ('permeability', re.compile(r'mu\d*|mur|permeab\w*')),
    ('permittivity', re.compile(r'eps\d*|epsr|epsilon\d*|permittiv\w*|dielectric')),
)
#: The index symbols that are also "number of" (``n_orders``, ``nG``, ``nr_steps``) or "n-D"
#: (``nd_array``): an index only standing alone (``n``), next to a number (``n_1``), or next to
#: one of :data:`_SLOT_WORDS` / :data:`_INDEX_PART_WORDS` (``n_top``, ``top_n``, ``nr_sub``,
#: ``nd_top``, ``n_eff``). Elsewhere they say nothing, so a count keeps its ordinary guess.
_CONTEXT_INDEX_WORDS = frozenset({'n', 'nr', 'nd'})
#: Each quantity's symbol, for the spellings that glue it to a slot: ``ntop``, ``nsub``,
#: ``ritop``, ``mubot``. Not ``nr`` / ``nd``: ``bg`` + ``nd`` is ``bgnd``, a background.
_QUANTITY_SYMBOLS = {'n': 'index', 'ri': 'index', 'mu': 'permeability', 'eps': 'permittivity'}
#: What to say instead of a guess, per quantity, where nothing on the door takes it.
_QUANTITY_HOME = {
    'index': ("a refractive index",
              "every material slot -- `top`, `bot`, `structure`, a layer's `background`, a "
              "shape's `material` -- takes relative permittivity (eps = n**2) or a Material; "
              "wavebit.eps_from_index(n) converts"),
    'permeability': ("a permeability",
                     "a permeability is `mu` beside a raster `structure`, or the `mu` of a "
                     "Material(epsilon, mu) in any material slot"),
    'permittivity': ("a permittivity",
                     "a permittivity goes in a material slot -- `top`, `bot`, `structure`, a "
                     "layer's `background`, a shape's `material` -- as a number or a Material"),
}


def _reads_as_index_beside(word):
    """Whether a neighbouring ``word`` makes an ``n`` / ``nr`` / ``nd`` beside it an index."""
    return (word.isdigit() or word in _INDEX_PART_WORDS
            or word.rstrip('0123456789') in _SLOT_WORDS)


def name_quantities(name):
    """The material quantities ``name`` is spelled as, as a frozenset -- empty for most names.

    A material slot reads as permittivity; any other name reads as whatever its words say
    (:data:`_QUANTITY_WORDS`, an ``n`` / ``nr`` / ``nd`` in an index's company
    (:data:`_CONTEXT_INDEX_WORDS`), and a symbol glued to a slot), so ``n_top``, ``nTop``,
    ``ntop``, ``n_sub``, ``n1``, ``n``, ``RI_top``, ``nr_top``, ``top_index`` and
    ``refractive_index`` are all an index, ``mu_top`` a permeability, ``eps_top`` a
    permittivity, and ``thet``, ``n_orders`` and ``nG`` nothing. Read off the spelling alone,
    because that is all a refused name has: it names no slot, so no slot can say what it holds.
    """
    if name.lower() in _MATERIAL_SLOTS:
        return frozenset({'permittivity'})
    words = [w.lower() for w in re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z0-9])', name)]
    found = set()
    for i, word in enumerate(words):
        found.update(q for q, pattern in _QUANTITY_WORDS if pattern.fullmatch(word))
        if word in _CONTEXT_INDEX_WORDS and (
                len(words) == 1 or any(map(_reads_as_index_beside, words[max(i - 1, 0):i]
                                           + words[i + 1:i + 2]))):
            found.add('index')
        found.update(q for symbol, q in _QUANTITY_SYMBOLS.items() for slot in _SLOT_WORDS
                     if word in (symbol + slot, slot + symbol))
    return frozenset(found)


def near_match(name, known, cutoff=0.6):
    """The one near name to offer for a refused ``name``, or ``None``; never across quantities.

    Every "did you mean" in the package comes from here, because a guess is an instruction the
    reader follows, and one that crosses quantities is a wrong answer rather than a wrong name.
    ``stack.n_top = 1.5`` is one letter from ``top``, and the plain nearest match said so: the
    user who follows it hands a refractive index to a slot that reads relative permittivity,
    and ``1.5`` is a fine permittivity -- the solve conserves energy and warns about nothing
    (:class:`~wavebit._core.domain.Material` says why the slots are permittivity). So a name
    spelled like a quantity (:func:`name_quantities`) is matched only against the names spelled
    like the same one, and a name spelled like none is matched against them all (``tpo`` is
    still a typo of ``top``). ``cutoff`` is difflib's; the solver's keyword door keeps 0.7.
    """
    import difflib
    quantity = name_quantities(name)
    candidates = sorted(k for k in known if not quantity or name_quantities(k) == quantity)
    near = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
    return near[0] if near else None


def quantity_note(names, known):
    """What a refused name spelled like a quantity is told when nothing in ``known`` takes it.

    The other half of :func:`near_match`'s rule: withholding the guess leaves ``n_top`` with
    nothing to go on, and the one thing that reader needs is that the material slots are not
    stated in index. Said only where no name in ``known`` is spelled like that quantity -- a
    dispersion table takes ``index``, and a note saying nothing here does would be false there.
    Returns ``''`` when there is nothing to say.
    """
    taken = frozenset().union(*(name_quantities(k) for k in known))
    spelled = {}
    for name in sorted(names):
        for quantity in sorted(name_quantities(name) - taken):
            spelled.setdefault(quantity, []).append(name)
    notes = []
    for quantity, refused in spelled.items():
        what, home = _QUANTITY_HOME[quantity]
        notes.append(f" {', '.join(map(repr, refused))} {'is' if len(refused) == 1 else 'are'} "
                     f"spelled like {what}, and no name here takes one, so no near name is "
                     f"guessed: {home}.")
    return ''.join(notes)


def near_match_hint(names, known):
    """The hint a refused name carries: its NVF rename, else its :func:`near_match`, and the
    :func:`quantity_note` of a name spelled like a quantity nothing in ``known`` takes.

    Shared by the two ways a name can be unknown -- an attribute assigned on an object
    (:func:`wavebit._core.engine._unknown_attribute`, for the solver, the excitation facade and
    the spec dataclasses) and a key in a parsed document (``_core/spec.py``'s
    ``_reject_unknown``) -- so ``inc.thet = 0.2`` and ``{'incidence': {'thet': 0.2}}`` are
    answered alike. A rename is offered only when its target is in ``known``: the owner must be
    able to take the name it is pointed at, so ``Stack.projector`` is not sent to a slot that
    lives on ``Numerics``. ``names`` are strings; a parsed key that is not one is refused
    before it gets here. Returns ``''`` when there is nothing to suggest.
    """
    known = sorted(known)
    renamed = [n for n in names if _NVF_OPTION_RENAMES.get(n) in known]
    rest = sorted(n for n in names if n not in renamed)
    near = [(n, meant) for n in rest for meant in [near_match(n, known)] if meant is not None]
    if len(names) == 1:
        guesses = ''.join(f' Did you mean {meant!r}?' for _, meant in near)
    else:
        guesses = (' Did you mean ' + ', '.join(f'{meant!r} for {n!r}' for n, meant in near)
                   + '?') if near else ''
    return nvf_option_rename_hint(renamed) + guesses + quantity_note(rest, known)


def normal_method_of(value):
    """Validate a ``raster_normal_method`` value."""
    return _one_of(NormalMethod, value, 'raster_normal_method')


def projector_of(value):
    """Validate a ``nvf_projector`` value."""
    return _one_of(Projector, value, 'nvf_projector')


def nesting_of(value):
    """Validate a ``li_nesting`` value."""
    return _one_of(Nesting, value, 'li_nesting')


def answers_of(value):
    """Validate an ``answers`` value."""
    return _one_of(Answers, value, 'answers')


def resolve_fourier_options(fourier_coefficients=None, factorization=None, keep_unset=False):
    """Settle the two Fourier axes from the keywords a caller passed.

    The two axes are independent:

    ==========================  ==============================================================
    ``fourier_coefficients``    how a layer's Fourier coefficients are obtained
    ``'sampled'``               discrete series off the raster, coefficients exact
    ``'integrated'``            closed-form series over the compressed constant blocks
    --------------------------  --------------------------------------------------------------
    ``factorization``           which rule closes the products of discontinuous functions
    ``'li'``                    Li's rules on the x/y axes
    ``'nvf'``                   normal vector field (a superset of Li: a constant normal
                                collapses the NVF projector onto Li's rules exactly)
    ``'laurent'``               the direct rule for every product -- opt-in, for comparing
                                with solvers that use it; never a default
    ==========================  ==============================================================

    The default is ``('sampled', 'li')`` and must stay there. That is a gradient-safety
    property, not a taste call: topology optimization needs one gradient per raster cell, and
    only the sampled path gives that — the integrated path compresses the raster into blocks,
    so the interior cells genuinely stop being variables. See
    ``tests/test_gradients.py::test_raster_per_cell_gradient_reaches_every_cell``.

    ``keep_unset=True`` returns ``None`` for an axis the caller left out, instead of that
    axis's default. Keeping the two apart is what lets a caller tell *"the user asked for
    Li"* from *"Li is the default nobody touched"* — a distinction the defaults erase, and
    the only thing that makes refusing a rule a build path cannot express safe. See
    :attr:`wavebit._core.engine.CoreFactorizationMixin.factorization`.
    """
    unset_coefficients = None if keep_unset else DEFAULT_FOURIER_COEFFICIENTS
    unset_factorization = None if keep_unset else DEFAULT_FACTORIZATION
    return (unset_coefficients if fourier_coefficients is None
            else coefficients_of(fourier_coefficients),
            unset_factorization if factorization is None
            else factorization_of(factorization))
