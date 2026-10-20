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


def _normal_power_roundoff(xp, admittance):
    """The roundoff floor below which ``Re(admittance)`` is no normal power at all.

    The tolerance of :func:`transverse_k_is_real`, applied to the admittance, and absolute as
    well as relative for the same reason. An evanescent incident wave written as a complex
    angle, ``theta = pi/2 + 1j*alpha`` from a lossless medium, has ``kx = n*cosh(alpha)`` in
    exact arithmetic, but ``cos(pi/2)`` is ``6.1e-17`` in double precision, so ``kx`` keeps an
    imaginary part of about ``6.1e-17*n*sinh(alpha)``. That is roundoff by the transverse test,
    and it reaches ``Re(kz)`` as about ``6.1e-17*n*cosh(alpha)``, which does not shrink with
    ``|kz|``: a relative test would call the wave propagating close to the light line.
    """
    return 32 * xp.finfo(xp.real(admittance).dtype).eps * (1 + xp.abs(admittance))


def carries_normal_power(xp, admittance):
    """Whether a wave with this admittance carries normal power beyond floating-point roundoff.

    False for an evanescent wave in a lossless medium, whose ``kz`` is imaginary, and for a wave
    at grazing to roundoff; true wherever ``Re(admittance)`` is a power of either sign. An
    efficiency divided by a power this test rejects is a ratio of roundoff: measured on air
    onto glass at ``theta = pi/2 +- 1j*acosh(1.2)``, ``T = -+1.7e16``, its sign set by the sign
    of ``alpha`` (#84). Traceable: a traced admittance gives a traced flag, read by no one.
    """
    return xp.abs(xp.real(admittance)) > _normal_power_roundoff(xp, admittance)


def incident_power(xp, C, lit, admittance):
    """Power for the declared order amplitudes of one polarization channel.

    With no explicit excitation the central order has unit amplitude. For a
    superposition, each illuminated order has its own real power weight.
    This function does not convert electric p amplitudes to magnetic ones;
    callers include that scale when forming the actual incident power.

    An explicitly lit order whose weight is not a positive power beyond roundoff
    (:func:`carries_normal_power`) is refused wherever its value can be read. Under
    ``jax.jit`` it cannot, and the solver marks what this power normalizes as NaN instead
    (:meth:`~wavebit.rcwa._engine.engine.CoreFactorizationMixin._incident_power_defined`). With no
    excitation an evanescent incident order is not refused: its amplitudes are a well-posed
    answer, and only the ratios to its power are undefined.
    """
    if lit is None:
        return xp.real(admittance[admittance.shape[0] // 2])
    total = None
    for one in lit:
        weight = xp.real(admittance[one.index])
        if _concrete(weight > _normal_power_roundoff(xp, admittance[one.index])) is False:
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
