"""Boundary power for isotropic half-spaces and real transverse wavevectors.

An s-channel amplitude is electric and a p-channel boundary amplitude is
magnetic. Their admittances are respectively kz/mu and kz/epsilon, with kz
normalized by the vacuum wavenumber. A common factor 1/(2 Z0) cancels from
the efficiency ratios. ``wave_power`` evaluates a wave in isolation;
``net_power`` includes incident/reflected interference at the same boundary.
"""
from wavebit._common.backend import _concrete
from wavebit._common.backend import as_array


def wave_power(xp, amplitude, admittance):
    """Compute scaled normal power from wave amplitudes at the boundary.

    Return one value per diffraction order for a single propagation direction.
    Incident and reflected waves are evaluated separately, without their
    interference term. The common factor 1/(2 Z0) is omitted because it
    cancels in efficiency ratios.
    """
    return xp.real(amplitude * xp.conj(amplitude)) * xp.real(admittance)


def net_power(xp, incident, reflected, admittance):
    """Net normal power from incident and reflected amplitudes at the boundary.

    Add the primary fields and subtract their opposite-direction partners
    before taking the real Poynting flux. This retains their interference.
    The primary is E for s and H for p; the real product is the same in both
    cases. Return one scaled power per order, positive into the structure,
    with the same common factor omitted as in ``wave_power``.
    """
    primary = incident + reflected
    partner = admittance * (incident - reflected)
    return xp.real(primary * xp.conj(partner))


def transverse_k_is_real(xp, kx, ky=None):
    """Whether the transverse wavevector components are real up to floating-point roundoff.

    The one test for "real transverse k": :func:`power_for_real_k` masks the cell powers on
    it, and the engine's complex-incidence warning fires on it, so the two cannot disagree
    about which inputs they describe.
    """
    valid = None
    for k in (kx, ky):
        if k is None:
            continue
        tolerance = 32 * xp.finfo(xp.real(k).dtype).eps * (1 + xp.abs(k))
        real = xp.all(xp.abs(xp.imag(k)) <= tolerance)
        valid = real if valid is None else valid & real
    return valid


def power_for_real_k(xp, value, kx, ky=None):
    """Mark cell-power results unavailable for complex transverse wavevectors.

    A real transverse wavevector makes distinct diffraction orders orthogonal
    over a unit cell. A complex one needs a different spatial integral; return
    NaN rather than presenting a sum of diagonal order powers as that integral.
    Allow only floating-point roundoff in the imaginary components.
    """
    valid = transverse_k_is_real(xp, kx, ky)
    return xp.where(valid, value, xp.full_like(value, float('nan')))


def incident_power(xp, C, lit, admittance):
    """Power for the declared order amplitudes of one polarization channel.

    With no explicit excitation the central order has unit amplitude. For a
    superposition, each illuminated order has its own real power weight.
    This function does not convert electric p amplitudes to magnetic ones;
    callers include that scale when forming the actual incident power.
    """
    if lit is None:
        return xp.real(admittance[admittance.shape[0] // 2])
    total = None
    for one in lit:
        weight = xp.real(admittance[one.index])
        if _concrete(weight > 0) is False:
            raise ValueError(
                f'the excitation lights order {one.order!r}, which is evanescent '
                'or carries no positive incident power in this medium; '
                'a zero-power near-field excitation cannot be normalized as '
                'an incident plane wave.')
        a = as_array(xp, one.amplitude, dtype=C)
        term = xp.real(a * xp.conj(a)) * weight
        total = term if total is None else total + term
    return total


def sp_incident_power(xp, C, lit, kz, n, epsilon, mu):
    """Separate powers for unit electric s and p input coefficients."""
    s_power = incident_power(xp, C, lit, kz / mu)
    h_per_e = as_array(xp, n / mu, dtype=C)
    p_power = xp.abs(h_per_e)**2 * incident_power(xp, C, lit, kz / epsilon)
    return s_power, p_power
