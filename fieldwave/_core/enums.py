"""Typed enumerations for the public API.

``Backend`` and ``Pol`` are ``IntEnum``s equal to the integer codes the solver already uses,
so they are a drop-in, non-breaking upgrade: ``Pol.TE == 0`` is true and the existing
``if self.pol == 0`` comparisons keep working unchanged. Users may pass either the enum
(``pol=Pol.TE``) or the raw int (``pol=0``).

``FourierCoefficients`` and ``Factorization`` are ``str`` enums, equal to the lowercase
strings the solver stores, so ``factorization=Factorization.NVF`` and ``factorization='nvf'``
are the same thing.

They replace the single ``FourierType`` integer, which conflated two independent choices —
see :func:`resolve_fourier_options`.
"""
import warnings
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


class FourierType(IntEnum):
    """Deprecated: the single integer that used to select both axes at once.

    Superseded by ``fourier_coefficients`` + ``factorization``; see
    :func:`resolve_fourier_options` for the mapping. Kept so existing code and pickles keep
    importing.
    """
    DFS = 0         # discrete Fourier series (standard Li rules on a raster)
    CONTINUOUS = 1  # closed-form Fourier of axis-aligned raster rectangles
    NVF = 2         # normal vector field (also accepts the strings 'nvf' / 'NVF')


# The deprecated integer stood for a *pair* of choices, which is exactly why it had to go:
# only three of the four combinations were spellable, and the missing one --
# ('integrated', 'nvf') -- is the one a shape/OCD fit of a tilted sidewall wants.
_FROM_FOURIER_TYPE = {
    0: (FourierCoefficients.SAMPLED, Factorization.LI),
    1: (FourierCoefficients.INTEGRATED, Factorization.LI),
    2: (FourierCoefficients.SAMPLED, Factorization.NVF),
}

# Reverse map, for the deprecated ``solver.fourier_type`` attribute. Deliberately partial.
_TO_FOURIER_TYPE = {pair: code for code, pair in _FROM_FOURIER_TYPE.items()}

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


def from_fourier_type(fourier_type):
    """The ``(fourier_coefficients, factorization)`` pair a legacy ``fourier_type`` meant.

    Emits a ``DeprecationWarning``. Accepts the integers 0/1/2 (or :class:`FourierType`) and
    the strings ``'nvf'`` / ``'NVF'`` the solver used to special-case.
    """
    key = 2 if isinstance(fourier_type, str) and fourier_type.lower() == 'nvf' else fourier_type
    try:
        pair = _FROM_FOURIER_TYPE[key]
    except (KeyError, TypeError):
        raise ValueError(
            f"fourier_type={fourier_type!r} is not valid; expected 0, 1, 2 or 'nvf'") from None

    warnings.warn(
        f"'fourier_type' is deprecated and will be removed; it conflated two independent "
        f"choices. Use fourier_coefficients={pair[0].value!r}, factorization={pair[1].value!r}.",
        DeprecationWarning, stacklevel=3)
    return pair


def to_fourier_type(fourier_coefficients, factorization):
    """The legacy integer for a pair, or ``None`` when the pair has no legacy spelling."""
    return _TO_FOURIER_TYPE.get((coefficients_of(fourier_coefficients),
                                 factorization_of(factorization)))


def resolve_fourier_options(fourier_coefficients=None, factorization=None, fourier_type=None):
    """Settle the two Fourier axes from the new keywords or the deprecated alias.

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

    ``fourier_type`` is the deprecated alias, mapping ``0 -> ('sampled', 'li')``,
    ``1 -> ('integrated', 'li')``, ``2 -> ('sampled', 'nvf')``. Passing it alongside either
    new keyword is an error rather than a silent precedence rule.
    """
    if fourier_type is not None:
        given = [k for k, v in (('fourier_coefficients', fourier_coefficients),
                                ('factorization', factorization)) if v is not None]
        if given:
            raise ValueError(
                f"fourier_type is the deprecated alias for {' and '.join(given)}; "
                f"pass one or the other, not both.")
        return from_fourier_type(fourier_type)

    return (DEFAULT_FOURIER_COEFFICIENTS if fourier_coefficients is None
            else coefficients_of(fourier_coefficients),
            DEFAULT_FACTORIZATION if factorization is None
            else factorization_of(factorization))
