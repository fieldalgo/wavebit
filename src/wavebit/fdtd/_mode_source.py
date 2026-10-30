"""A waveguide mode source: the public :class:`ModeSource` lowered to a one-sided mode sheet.

The sheet is the kernel's TFSF construction with a mode of the discrete system for its
incident field (``src/wavebit/fdtd/_kernel/mode_sheet.py``); the mode is solved on the source plane
at the pulse's centre wavelength by ``src/wavebit/fdtd/_modes.py``. At that wavelength the sheet
injects the mode one way only, exactly. At every other wavelength it injects exactly what
:func:`injected_amplitudes` computes, by the following argument.

**What a TFSF plane injects.** The sheet's corrections read only the incident field's two
planes: the transverse E on the node plane ``k0`` and the transverse H on the half plane
upstream. Any such pair of transverse fields is uniquely a sum of the discrete modes travelling
both ways, ``E_t(k0) = sum (alpha_m + gamma_m) e_m`` and ``H_t = sum (alpha_m exp(-i theta_m) -
gamma_m exp(+i theta_m)) h_m`` (``theta = beta dx/2``; the half plane upstream of a ``+w``
sheet). That sum is an exact source-free solution with the same two planes, so the sheet
radiates what it would for it: the total-field side holds the whole sum and the scattered side
nothing. That field brings ``gamma`` in from infinity on the total-field side; the causal,
outgoing field subtracts the backward part everywhere. Downstream there is ``alpha_m``, and
upstream ``-gamma_m``: the injection and the leakage. The modes' exact unconjugated
orthogonality gives both from two overlaps ``X = sum (E_u h_v - E_v h_u) dA`` and ``Y = sum (e_u
H_v - e_v H_u) dA``:

    alpha = (X exp(i theta) + Y) / (2 N cos theta),     gamma = (X exp(-i theta) - Y) / (2 N cos theta),

with ``N = sum e . h dA``. With the centre frequency's mode as the incident field, ``alpha`` is the
pulse and ``gamma`` zero, which is the exactness stated above. (For a ``-w`` sheet the half plane
is downstream in ``w``; the roles of ``alpha`` and ``gamma`` swap.)
"""
import math
from dataclasses import dataclass

import numpy as np

from wavebit.errors import UnsupportedFeatureError, ValidationError

__all__ = ['ModeSheet', 'mode_sheet', 'injected_amplitudes']


@dataclass(frozen=True, eq=False)
class ModeSheet:
    """The engine's mode source: a lowered :class:`ModeSource`, in the engine's units.

    ``band`` and the waveform leaves (``amplitude``, ``phase``, ``delay``) are the engine
    source's usual ones; ``k`` is the sheet's node plane along ``axis`` and ``sign`` the
    direction. ``window`` holds the transverse cell ranges, ``patterns`` the four incident
    patterns (``'Eu'``, ``'Ev'``, ``'Hu'``, ``'Hv'``, local ``(N_u, N_v)`` arrays, scaled so that a
    unit waveform is a 1 W mode), ``row_offsets`` the waveform rows' offsets, and ``reference``
    what :func:`injected_amplitudes` needs.
    """

    name: str
    band: object
    amplitude: object
    phase: object
    delay: object
    axis: int
    k: int
    sign: int
    window: tuple
    patterns: dict
    row_offsets: tuple
    reference: dict
    #: Each incident component's ``((pattern label, row), ...)``; the rows' time shifts.
    components: dict = None
    shifts: tuple = (0.0, 0.5)
    kind: str = 'ModeSource'

    def __post_init__(self):
        if self.components is None:
            object.__setattr__(self, 'components', {'Eu': (('Eu', 0),), 'Ev': (('Ev', 0),),
                                                    'Hu': (('Hu', 1),), 'Hv': (('Hv', 1),)})

    @property
    def position(self):
        return self.reference['position']


def _group_velocity(window, wavelength, index):
    """``d omega / d beta`` of mode ``index`` at ``wavelength``, by a central difference."""
    from ._modes import solve_window
    h = 1e-4
    betas = []
    for lam in (wavelength * (1 - h), wavelength * (1 + h)):
        betas.append(solve_window(window, lam, index + 1)[index][1])
    omegas = (2 * math.pi / (wavelength * (1 - h)), 2 * math.pi / (wavelength * (1 + h)))
    return (omegas[0] - omegas[1]) / (betas[0] - betas[1])


def mode_sheet(source, problem, numerics, units):
    """The :class:`ModeSheet` of public ``source``: solve its mode and lay out the sheet."""
    from ._modes import prepare_window, solve_window, window_plane
    from ._lower_public import _pulse
    from . import _geometry as geometry
    path = 'sources.' + source.name
    if not isinstance(source.plane, geometry.Plane):
        raise ValidationError('expected Plane', code='source.plane', path=path + '.plane')
    axis = source.plane.axis
    direction = source.direction
    if direction not in ('+' + 'xyz'[axis], '-' + 'xyz'[axis]):
        raise ValidationError(f"ModeSource(direction={direction!r}): '+{'xyz'[axis]}' or "
                              f"'-{'xyz'[axis]}', along the plane's normal",
                              code='source.direction', path=path + '.direction')
    sign = 1 if direction[0] == '+' else -1
    band = _pulse(source.pulse, units)
    from ._modes import _check_wavelength
    wavelength = _check_wavelength(source.pulse.wavelength, path + '.pulse.wavelength')
    plane = window_plane(problem, numerics, source.plane)
    window = prepare_window(problem, numerics, plane, wavelength)
    index = source.mode
    modes = solve_window(window, wavelength, index + 1)
    q, beta, f, residual, edge = modes[index]
    v_g = _group_velocity(window, wavelength, index)
    if not (math.isfinite(v_g) and v_g > 0):
        raise UnsupportedFeatureError(f'mode {index} has no positive group velocity at the '
                                      f'centre wavelength; a mode source launches a forward mode',
                                      code='source.mode', path=path + '.mode')
    dx = window.dx
    omega0 = 2 * math.pi / wavelength
    delta = dx / (2 * v_g)
    # The H row: the carrier advanced by beta dx/2, the envelope by the half cell's group delay.
    row_offsets = ((0.0, 0.0), (beta * dx / 2 - omega0 * delta, -delta))
    scale = 1.0 / math.sqrt(window.measure)            # unit flux -> 1 W of the mode
    patterns = {'Eu': f['Eu'].real * scale, 'Ev': f['Ev'].real * scale,
                'Hu': sign * f['Hu'].real * scale, 'Hv': sign * f['Hv'].real * scale}
    w, u, v = window.w, window.u, window.v
    reference = dict(wavelength=wavelength, beta=beta, q=q, group_velocity=v_g, index=index,
                     residual=residual, edge_ratio=edge, measure=window.measure,
                     position=window.low[w] + window.k * dx, window=window)
    return ModeSheet(name=source.name, band=band, amplitude=source.amplitude,
                     phase=source.phase, delay=units._time_to_internal(source.delay_seconds),
                     axis=w, k=window.k, sign=sign,
                     window=(window.ranges[u], window.ranges[v]), patterns=patterns,
                     row_offsets=row_offsets, reference=reference)


def waveform_spectra(sheet, plan, rt, wavelengths):
    """One ``(nf,)`` array per waveform row (a mode sheet's ``(G_E, G_H)``): its DFT at its own times.

    Built as the kernel builds the rows (``waveform_table``), from the declaration's leaves, so a
    traced amplitude, phase or delay stays traced.
    """
    xp = rt.xp
    band = sheet.band
    tau = 1.0 / ((2.0 * math.pi) * band.df)
    t0 = band.sigmas * tau
    f0 = 1.0 / band.wavelength
    n = np.arange(plan.steps, dtype=np.float64)
    freqs = np.asarray([1.0 / w for w in wavelengths], dtype=np.float64)
    out = []
    for shift, (dphi, ddelay) in zip(sheet.shifts, sheet.row_offsets):
        t = (n + shift) * plan.dt
        s = rt.array(t, 'float64') - t0 - rt.array(sheet.delay, 'float64') - ddelay
        w = (rt.array(sheet.amplitude, 'float64')
             * xp.exp(-0.5 * (s / tau) ** 2)
             * xp.sin((2.0 * math.pi) * f0 * s + rt.array(sheet.phase, 'float64') + dphi))
        kernel = np.exp(1j * 2.0 * math.pi * freqs[:, None] * t[None, :]) * plan.dt
        out.append(rt.array(kernel, 'complex128') @ (w * rt.array(1.0 + 0.0j, 'complex128')))
    return tuple(out)


def injected_amplitudes(sheet, plan, rt, wavelengths, index=None):
    """``(injected, leakage)``, ``(nf,)``: the source mode's amplitude launched and leaked.

    In the units :class:`ModeMonitor` reports before its spectral scaling (the DFT amplitude of
    the unit-flux mode in the kernel's units), at the sheet's own node plane: ``injected`` is
    the amplitude of the mode travelling the source's way on its far side, ``leakage`` of the
    same mode travelling the other way on its near side. Each wavelength's mode is solved on the
    source's window; ``index`` is the mode (the source's own by default).
    """
    from ._modes import solve_window
    xp = rt.xp
    window = sheet.reference['window']
    index = sheet.reference['index'] if index is None else index
    dA = window.dx * window.dx
    G_E, G_H = waveform_spectra(sheet, plan, rt, wavelengths)[:2]
    p = sheet.patterns                                  # the incident field per unit waveform
    injected, leakage = [], []
    for i, lam in enumerate(wavelengths):
        q, beta, f, residual, edge = solve_window(window, float(lam), index + 1)[index]
        theta = beta * window.dx / 2.0
        two_n_cos = 4.0                                 # 2 N cos(theta), N = 2/cos(theta)
        # The sheet's incident planes at this wavelength, against the mode at this wavelength.
        X0 = float(np.sum(p['Eu'] * f['Hv'].real - p['Ev'] * f['Hu'].real) * dA)
        Y0 = float(np.sum(f['Eu'].real * p['Hv'] - f['Ev'].real * p['Hu']) * dA)
        X, Y = X0 * G_E[i], Y0 * G_H[i]
        if sheet.sign > 0:
            alpha = (X * np.exp(1j * theta) + Y) / two_n_cos
            gamma = (X * np.exp(-1j * theta) - Y) / two_n_cos
            injected.append(alpha)
            leakage.append(-gamma)
        else:
            # The half plane is downstream in w: there the forward mode's H is exp(+i theta) h
            # and the backward one's -exp(-i theta) h, which swaps the two solutions; the sheet
            # launches the backward mode.
            alpha = (X * np.exp(-1j * theta) + Y) / two_n_cos
            gamma = (X * np.exp(1j * theta) - Y) / two_n_cos
            injected.append(gamma)
            leakage.append(-alpha)
    return xp.stack(injected), xp.stack(leakage)
