"""Common half-plane certificates for the experimental finite pair cells.

The phase is used only to certify coercivity. Physical permittivities, weak
equations, energy coefficients and their derivatives are never rotated.
"""

import numpy as np

from wavebit.rcwa._engine.conforming.cartesian_pair import _require


def phase_from_angle(angle):
    """Validate a static angle in radians and return exp(-1j*angle)."""
    value = np.asarray(angle)
    if (value.shape != () or value.dtype.kind not in "iuf"
            or not np.isfinite(value) or not 0 <= value <= np.pi):
        raise ValueError("coercivity_angle must be a finite real scalar in [0, pi]")
    return complex(np.exp(-1j * float(value)))


def passive_pair_materials(epsilon, background, phase=1.0, xp=np):
    """Physical passive scalars inside one certified open half-plane.

    At the default phase, retain the original strictly positive-real domain.
    A rotated certificate needs a 64-epsilon angular margin so trigonometric
    roundoff cannot admit opposed lossless signs on a half-plane boundary.
    """
    try:
        e, b = xp.asarray(epsilon), xp.asarray(background)
    except (TypeError, ValueError) as error:
        raise ValueError("materials must be numeric scalars") from error
    if (e.shape != (2,) or b.shape != ()
            or e.dtype.kind not in "iufc" or b.dtype.kind not in "iufc"):
        raise ValueError("materials must be two numeric scalar permittivities and a scalar background")
    values = xp.asarray(xp.concatenate([xp.reshape(b, (1,)), e]), dtype=xp.complex128)
    finite = xp.all(xp.isfinite(values))
    # Invalid traced data must not overflow while constructing the certificate.
    safe = xp.where(finite, values, xp.ones_like(values))
    if phase == 1.0:
        # Preserve the original domain even for extremely unequal components;
        # division by |epsilon| could underflow a strictly positive real part.
        inside = xp.real(safe) > 0
    else:
        magnitude = xp.abs(safe)
        unit = safe / xp.where(magnitude > 0, magnitude, 1.0)
        inside = (magnitude > 0) & (xp.real(phase * unit) > 64 * np.finfo(np.float64).eps)
    valid = finite & xp.all(xp.imag(safe) >= 0) & xp.all(inside)
    _require(valid, "materials must be finite, passive and inside the certified coercivity half-plane")
    return values, valid


def minimum_rotated_storage(matrix, phase=1.0, xp=np):
    rotated = phase * matrix
    return xp.linalg.eigvalsh((rotated + rotated.conj().T) / 2)[0]
