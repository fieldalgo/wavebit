"""Typed enumerations for the public API.

``Backend`` and ``Pol`` are ``IntEnum``s equal to the integer codes the solver already uses.
``Pol`` names the two states the ``tetm`` declaration selects, and its integer value is the
index the pure 1D solves dispatch on (``CoreSolveMixin.tetm_index``). Users write the word --
``tetm='te'`` or ``tetm=Pol.TE`` -- because the declaration has exactly two values and a
number invited a third meaning: the retired ``pol`` accepted the interior of ``[0, 1]``,
where it was a linear angle in a bad unit rather than a declaration at all.

``FourierCoefficients`` and ``Factorization`` are ``str`` enums, equal to the lowercase
strings the solver stores, so ``factorization=Factorization.NVF`` and ``factorization='nvf'``
are the same thing.

The two are independent choices, and keeping them as separate keywords is what makes all four
of their pairings reachable — see :func:`resolve_fourier_options`.
"""
from enum import Enum, IntEnum


class Backend(IntEnum):
    """Array backend for :func:`wavebit.rcwa`."""
    NUMPY = 0
    JAX = 1
    TORCH = 2


class Pol(IntEnum):
    """The two states of the ``tetm`` declaration; the value is ``tetm_index``."""
    TE = 0
    TM = 1


class FourierCoefficients(str, Enum):
    """How a layer's Fourier coefficients are obtained from the raster."""
    SAMPLED = 'sampled'      # discrete Fourier series off the (optionally upsampled) raster
    INTEGRATED = 'integrated'  # closed-form series over the compressed constant blocks


class Factorization(str, Enum):
    """Which rule closes the products of discontinuous functions."""
    LI = 'li'    # Li's rules on the x/y axes
    NVF = 'nvf'  # normal vector field (Schuster et al. 2007); a superset of Li


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


class Formulation(str, Enum):
    """Which layer formulation the 2D solve uses — how many fields per order it carries.

    ``'2ff'`` reduces a layer to ``d2E_t/dz2 = Omega2 E_t``, which exists only while the
    permittivity's ``z`` row and column are empty. ``'4ff'`` carries the full transverse state
    ``(Ex, Ey, Hx, Hy)`` as a first-order system, which is what a z-coupled (magneto-optic,
    Voigt) tensor needs — at roughly 8x the cost, because the eigenproblem doubles.

    ``'auto'`` is the default and reads the problem: a solve handed any of the four z-row
    convolution slots takes ``'4ff'``, everything else ``'2ff'``. The override exists because
    forcing ``'4ff'`` on a problem that does not need it is how the two are cross-checked
    (``docs/anisotropic-materials.md`` §8); forcing ``'2ff'`` on a z-coupled tensor raises
    rather than dropping the row, because dropping it answers a different problem while
    conserving energy to machine precision.
    """
    AUTO = 'auto'    # read the slots and decide
    SECOND = '2ff'   # second-order; refuses a z row
    FIRST = '4ff'    # first-order; always possible, ~8x


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
                        pure-TM incidence variants ``res_te_inc`` / ``res_tm_inc``
    ``'field'``         the per-layer modes and amplitudes ``reconstruct_field`` walks
    ==================  ==========================================================

    The seam that matters is between the last two, and it is not a matter of taste. The
    amplitudes are boundary data — a few vectors, already solved for — while the field state is
    ``4.25`` matrices of side ``dim`` *per layer* on the 2D path (``3.75`` conical, ``5`` pure
    1D), with ``dim = 2*n_orders_x*n_orders_y`` on the coupled paths. At ``max_order=(15, 15)``
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
#: The rung this is waiting for is ``'scattering'``, above ``'field'``: the SMM cascade builds
#: ``S``, the per-layer ``S_layer``, the below-plane reflections ``gamma`` and the gap<->layer
#: interface coefficients ``AB`` (:func:`~wavebit._core.scattering._ab_matrices`), uses them,
#: and drops them -- the same "computed and thrown away" shape the field state had before
#: ``answers`` existed. Two constraints go with it, written here because here is where someone
#: adding the rung will be standing:
#:
#: * **``AB`` ships with the blocks or the blocks do not ship.** ``S_layers`` and ``gamma`` are
#:   expressed in the gap basis (``eps_gap`` chosen per order so ``kz_gap == 1``), and that
#:   basis is a gauge: change ``eps_gap`` and the physical observables hold at 1e-16 while the
#:   raw blocks move by O(1) against their own magnitude. What they encode is gauge-invariant;
#:   their numbers are not readable without the gap -> half-space conversion, and
#:   ``_ab_matrices`` is that conversion. Exposing the blocks while withholding it would
#:   document a footgun and withhold the safety. (Measured in a separate session; the raw
#:   record is not in this repository.)
#: * **``S_ref`` / ``S_trn`` should not ship.** They are gauge like the rest, and converted to a
#:   physical (E, H) transfer ``S_ref`` is exactly the identity -- it encodes nothing -- while
#:   the two of them are most of the fixed matrices such a mapping would retain.
_ANSWERS_LADDER = (Answers.EFFICIENCY, Answers.AMPLITUDE, Answers.FIELD)


DEFAULT_FOURIER_COEFFICIENTS = FourierCoefficients.SAMPLED
DEFAULT_FACTORIZATION = Factorization.LI
DEFAULT_FORMULATION = Formulation.AUTO
DEFAULT_NORMAL_METHOD = NormalMethod.DISTANCE
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


def normal_method_of(value):
    """Validate a ``normal_method`` value."""
    return _one_of(NormalMethod, value, 'normal_method')


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
