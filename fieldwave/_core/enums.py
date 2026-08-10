"""Typed enumerations for the public API.

``Backend`` and ``Pol`` are ``IntEnum``s equal to the integer codes the solver already uses,
so they are a drop-in, non-breaking upgrade: ``Pol.TE == 0`` is true and the existing
``if self.pol == 0`` comparisons keep working unchanged. Users may pass either the enum
(``pol=Pol.TE``) or the raw int (``pol=0``).

``FourierCoefficients`` and ``Factorization`` are ``str`` enums, equal to the lowercase
strings the solver stores, so ``factorization=Factorization.NVF`` and ``factorization='nvf'``
are the same thing.

The two are independent choices, and keeping them as separate keywords is what makes all four
of their pairings reachable — see :func:`resolve_fourier_options`.
"""
from enum import Enum, IntEnum


class Backend(IntEnum):
    """Array backend for :func:`fieldwave.rcwa`."""
    NUMPY = 0
    JAX = 1
    TORCH = 2


class Pol(IntEnum):
    """Incident polarization (1D / conical)."""
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


DEFAULT_FOURIER_COEFFICIENTS = FourierCoefficients.SAMPLED
DEFAULT_FACTORIZATION = Factorization.LI


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


def resolve_fourier_options(fourier_coefficients=None, factorization=None, keep_unset=False):
    """Settle the two Fourier axes from the keywords a caller passed.

    The two axes are independent:

    ==========================  ==============================================================
    ``fourier_coefficients``    how a layer's Fourier coefficients are obtained
    ``'sampled'``               discrete series off the raster (DFS), coefficients exact
    ``'integrated'``            closed-form series over the compressed constant blocks (CFS)
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
    :attr:`fieldwave._core.engine.CoreFactorizationMixin.factorization`.
    """
    unset_coefficients = None if keep_unset else DEFAULT_FOURIER_COEFFICIENTS
    unset_factorization = None if keep_unset else DEFAULT_FACTORIZATION
    return (unset_coefficients if fourier_coefficients is None
            else coefficients_of(fourier_coefficients),
            unset_factorization if factorization is None
            else factorization_of(factorization))
