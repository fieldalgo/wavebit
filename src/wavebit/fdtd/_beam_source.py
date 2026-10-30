"""A focused Gaussian beam: the public :class:`GaussianBeam` lowered to a one-sided sheet.

The sheet is the mode source's (``src/wavebit/fdtd/_kernel/mode_sheet.py``) with a beam for its
incident field. The beam is built from the discrete plane waves of the Yee grid in the
homogeneous medium at the sheet, so at the pulse's centre frequency it is an exact solution of
the grid and the sheet launches it one way only.

**The beam.** On the focal plane the polarized transverse component, sampled at its own Yee
points, is ``exp(-r^2 / waist^2)`` about the focus (``waist`` is the 1/e amplitude radius, as
for ``rcwa.GaussianBeam``); the other transverse component is zero. That plane's discrete
Fourier transform is taken over a zero-padded transverse grid wide enough that the beam is
negligible at its edges. Each transverse wavenumber ``k_t`` is one discrete plane wave: with
``q_a = (2/dx) sin(k_a dx/2)`` (the staggered difference of ``exp(i k_a a)``) and the leapfrog's
``Omega = (2/dt) sin(omega dt/2)``, it propagates along the normal with ``q_w`` from
``q_u^2 + q_v^2 + q_w^2 = Omega^2 eps mu``, i.e. ``beta = (2/dx) asin(q_w dx/2)``. Components
with no real ``beta`` (evanescent ones, or beyond the grid's band) are dropped: the beam is the
propagating part of the angular spectrum. The longitudinal E follows from the discrete Gauss
law ``q . E = 0`` and H from the discrete Faraday law ``H = q x E / (Omega mu)`` (phasors
``exp(-i omega t)``, each component at its own Yee point and, for H, its own half step). Each
plane wave then advances from the focal plane to the sheet's node plane (E) and to the half
plane upstream (H) by ``exp(i beta d)``. The angular-spectrum representation of a beam is
standard (Goodman, *Introduction to Fourier Optics*, ch. 3; Mandel and Wolf, *Optical
Coherence and Quantum Optics*, sec. 3.2); its discrete form here uses the grid's own dispersion.

**The rows.** The incident patterns are complex. Each field is ``Re(P exp(-i psi))`` with
``psi = 2 pi f0 s + phase``: ``Re(P)`` on a cosine row and ``Im(P)`` on a sine row. The H rows
are read at ``(n + 1/2) dt`` with the envelope advanced by the axial group delay over half a
cell (the carrier's half-cell phase is in the H patterns). At the centre frequency the sheet's
two planes are the beam's; away from it each plane wave's half-cell phase and profile are the
centre's, and the sheet launches a little of the beam backward. :func:`launched_power`
computes both parts exactly, per plane wave, from the actual rows.

**Power.** The patterns are scaled so that the kernel flux through the sheet's window at the
centre frequency is one unit; with the window's watts per unit flux, a continuous wave of
amplitude 1 carries 1 W (per metre of each invariant axis).
"""
import math

import numpy as np

from wavebit.errors import UnsupportedFeatureError, ValidationError

__all__ = ['beam_sheet', 'beam_fields', 'launched_power']

#: The beam's transverse E on the window's outermost cells, relative to its peak, at most this
#: (the sheet stops at the window; what lies beyond is not launched). As for a mode's window.
EDGE = 1e-3
#: The beam on the padded transform grid's edges, relative to its peak, at most this: what
#: wraps around the periodic transform is of that order. Dropping the evanescent part cuts the
#: spectrum at the light line, which leaves a ripple decaying only as 1/r (about 1e-10 of the
#: peak at 25 um for a waist of 1.5 wavelengths, more for a tighter one), so the padding grows
#: until the ripple is below this, up to ``TRANSFORM_POINTS``.
PADDED = 1e-7
#: The padded transform grid's largest number of points; a beam that needs more is refused.
TRANSFORM_POINTS = 1 << 24
UNIFORM = 1e-12


def _local_axis(name, w, path):
    axes = 'xyz'
    if name not in axes or axes.index(name) == w:
        raise ValidationError(f"GaussianBeam(polarization={name!r}): the transverse axis the E "
                              f"field points along, one of "
                              f"{[a for a in axes if axes.index(a) != w]}",
                              code='source.polarization', path=path)
    a = axes.index(name)
    return 'u' if a == (w + 1) % 3 else 'v'


def _concrete(value, path, what):
    from ._modes import _traced
    if _traced(value):
        raise UnsupportedFeatureError(f'{what} is traced or carries a gradient; the beam is '
                                      f'built on the host from concrete values. The amplitude, '
                                      f'phase and delay may be traced.', code='source.traced',
                                      path=path)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f'{what} must be a real number', code='source.value',
                              path=path) from exc


def _medium(window, path):
    """The window's one relative permittivity and permeability, or a refusal."""
    values = []
    for store in (window.eps, window.mu):
        arrays = np.concatenate([np.asarray(a).ravel() for a in store.values()])
        ref = float(arrays[0])
        if not np.allclose(arrays, ref, rtol=UNIFORM, atol=0.0):
            raise UnsupportedFeatureError(
                'the medium on the beam plane is not uniform; a Gaussian beam is launched in a '
                'homogeneous medium (the one that fills its plane). Move the plane into a '
                'uniform region.', code='source.medium', path=path)
        values.append(ref)
    return values


def _sides(window, path):
    for pair in window.sides:
        for side in pair:
            if side.kind == 'invariant' or (side.kind == 'wall' and side.truncates):
                continue
            raise UnsupportedFeatureError(
                'a Gaussian beam needs Open (or InvariantAxis) faces across its plane: it is a '
                'beam in open space, and a metal, mirror or periodic face would turn it into '
                'the modes of that guide or lattice.', code='source.boundary', path=path)


def beam_fields(window, wavelength, waist, focus, polarization, sign, planes, pad=4):
    """The beam's complex phasors on the window, per plane, before normalization.

    ``planes`` maps a label to ``(field, w_position)``: ``field`` ``'E'`` or ``'H'``, the plane's
    position along the normal. Returns ``{label: {'u': (Nu, Nv), 'v': (Nu, Nv)}}`` (the
    transverse components at their Yee points on that plane), the axial ``beta`` and the
    padded grid's edge ratio.
    """
    from ._kernel.grid import OFFSETS
    w, u, v = window.w, window.u, window.v
    dx, dt = window.dx, window.dt
    eps, mu = _medium(window, 'sources')
    omega = 2.0 * math.pi / wavelength
    Omega = 2.0 * math.sin(omega * dt / 2.0) / dt
    (u0, u1), (v0, v1) = window.ranges[u], window.ranges[v]
    Nu, Nv = u1 - u0, v1 - v0
    invariant = [s[0].kind == 'invariant' for s in window.sides]
    sizes, starts = [], []
    for n, lo, inv in ((Nu, u0, invariant[0]), (Nv, v0, invariant[1])):
        if inv:
            sizes.append(1)
            starts.append(lo)
            continue
        m = 1 << int(math.ceil(math.log2(pad * n)))
        sizes.append(m)
        starts.append(lo - (m - n) // 2)
    Mu, Mv = sizes
    ku = 2.0 * np.pi * np.fft.fftfreq(Mu, dx)[:, None]
    kv = 2.0 * np.pi * np.fft.fftfreq(Mv, dx)[None, :]
    qu = (2.0 / dx) * np.sin(ku * dx / 2.0)
    qv = (2.0 / dx) * np.sin(kv * dx / 2.0)
    qw2 = Omega ** 2 * eps * mu - qu ** 2 - qv ** 2
    keep = (qw2 > 0) & (np.sqrt(np.maximum(qw2, 0.0)) * dx / 2.0 < 1.0)
    qw = np.where(keep, np.sqrt(np.maximum(qw2, 0.0)), 1.0)
    beta = np.where(keep, (2.0 / dx) * np.arcsin(np.minimum(qw * dx / 2.0, 1.0)), 0.0)
    qw, beta = sign * qw, sign * beta

    def positions(field, local):
        c = {'u': u, 'v': v, 'w': w}[local]
        off = OFFSETS[field][c]
        cu = window.low[u] + (starts[0] + np.arange(Mu) + 0.5 * off[u]) * dx
        cv = window.low[v] + (starts[1] + np.arange(Mv) + 0.5 * off[v]) * dx
        return cu, cv, off

    # The focal plane: the polarized component sampled at its own points.
    cu, cv, off_p = positions('E', polarization)
    fu, fv, fw = focus[u], focus[v], focus[w]
    du = (cu[:, None] - fu) if not invariant[0] else np.zeros((1, 1))
    dv = (cv[None, :] - fv) if not invariant[1] else np.zeros((1, 1))
    g = np.exp(-(du ** 2 + dv ** 2) / waist ** 2)
    # Amplitudes referred to each component's own sampling: A_c(k) exp(i k (n + o_c/2) dx).
    A = {}
    G = np.fft.fft2(g)
    shift_p = np.exp(-1j * (ku * off_p[u] + kv * off_p[v]) * dx / 2.0)
    A['Eu' if polarization == 'u' else 'Ev'] = np.where(keep, G * shift_p, 0.0)
    A['Ev' if polarization == 'u' else 'Eu'] = np.zeros_like(A['Eu' if polarization == 'u' else 'Ev'])
    A['Ew'] = np.where(keep, -(qu * A['Eu'] + qv * A['Ev']) / qw, 0.0)
    A['Hu'] = (qv * A['Ew'] - qw * A['Ev']) / (Omega * mu)
    A['Hv'] = (qw * A['Eu'] - qu * A['Ew']) / (Omega * mu)
    out = {}
    edge = 0.0
    for label, (field, position) in planes.items():
        phase = np.exp(1j * beta * (position - fw))
        comps = {}
        for local in ('u', 'v'):
            name = field + local
            _, _, off = positions(field, local)
            spatial = np.exp(1j * (ku * off[u] + kv * off[v]) * dx / 2.0)
            full = np.fft.ifft2(A[name] * phase * spatial)
            if field == 'E':
                peak = float(np.max(np.abs(full)))
                if peak > 0:
                    rim = max(float(np.max(np.abs(full[[0, -1], :]))) if Mu > 1 else 0.0,
                              float(np.max(np.abs(full[:, [0, -1]]))) if Mv > 1 else 0.0)
                    edge = max(edge, rim / peak)
            iu = slice(u0 - starts[0], u0 - starts[0] + Nu)
            iv = slice(v0 - starts[1], v0 - starts[1] + Nv)
            comps[local] = full[iu, iv]
        out[label] = comps
    axial = float((2.0 / dx) * math.asin(min(Omega * math.sqrt(eps * mu) * dx / 2.0, 1.0)))
    return out, axial, edge


def _axial_group_velocity(window, wavelength):
    eps, mu = _medium(window, 'sources')
    dx, dt = window.dx, window.dt

    def beta(lam):
        Om = 2.0 * math.sin(math.pi * dt / lam) / dt
        return (2.0 / dx) * math.asin(Om * math.sqrt(eps * mu) * dx / 2.0)
    h = 1e-4
    l1, l2 = wavelength * (1 - h), wavelength * (1 + h)
    return (2 * math.pi / l1 - 2 * math.pi / l2) / (beta(l1) - beta(l2))


def beam_sheet(source, problem, numerics, units):
    """The engine sheet (a ``ModeSheet``) of public :class:`GaussianBeam` ``source``."""
    from ._modes import prepare_window, window_plane, _check_wavelength, _masks
    from ._lower_public import _pulse
    from ._mode_source import ModeSheet
    from . import _geometry as geometry
    path = 'sources.' + source.name
    if not isinstance(source.plane, geometry.Plane):
        raise ValidationError('expected Plane', code='source.plane', path=path + '.plane')
    w = source.plane.axis
    direction = source.direction
    if direction not in ('+' + 'xyz'[w], '-' + 'xyz'[w]):
        raise ValidationError(f"GaussianBeam(direction={direction!r}): '+{'xyz'[w]}' or "
                              f"'-{'xyz'[w]}', along the plane's normal",
                              code='source.direction', path=path + '.direction')
    sign = 1 if direction[0] == '+' else -1
    polarization = _local_axis(source.polarization, w, path + '.polarization')
    waist = _concrete(source.waist, path + '.waist', 'the waist')
    if not (math.isfinite(waist) and waist > 0):
        raise ValidationError('the waist is a positive length', code='source.waist',
                              path=path + '.waist', actual=source.waist)
    if not isinstance(source.focus, tuple) or len(source.focus) != 3:
        raise ValidationError('the focus is a point (x, y, z)', code='source.focus',
                              path=path + '.focus')
    focus = tuple(_concrete(c, path + '.focus', 'the focus') for c in source.focus)
    band = _pulse(source.pulse, units)
    wavelength = _check_wavelength(source.pulse.wavelength, path + '.pulse.wavelength')
    window = prepare_window(problem, numerics, window_plane(problem, numerics, source.plane),
                            wavelength)
    _sides(window, path + '.plane')
    eps, mu = _medium(window, path + '.plane')
    dx = window.dx
    w_node = window.low[w] + window.k * dx
    h_plane = w_node - sign * dx / 2.0
    planes = {'E': ('E', w_node), 'H': ('H', h_plane),
              'H_other': ('H', w_node + sign * dx / 2.0)}
    fields, axial, padded = beam_fields(window, wavelength, waist, focus, polarization, sign,
                                        planes, pad=4)
    pad = 4
    while padded > PADDED:
        pad *= 2
        points = 1
        for n, s in zip(window.eps['u'].shape, window.sides):
            points *= 1 if s[0].kind == 'invariant' else (1 << int(math.ceil(math.log2(pad * n))))
        if points > TRANSFORM_POINTS:
            raise UnsupportedFeatureError(
                f'the beam does not fit a transform grid of {TRANSFORM_POINTS} points (it still '
                f'reaches {padded:.2g} of its peak at the padded edge, above {PADDED:g}). A '
                f'beam focused this tightly has a large part of its angular spectrum at the '
                f'light line, and the propagating part rings far out; widen the waist.',
                code='source.beam_size', path=path)
        fields, axial, padded = beam_fields(window, wavelength, waist, focus, polarization,
                                            sign, planes, pad=pad)
    masks = _masks(*window.eps['u'].shape, window.sides)
    E = {c: fields['E'][c] * masks['E' + c] for c in 'uv'}
    Ht = {c: fields['H'][c] * masks['H' + c] for c in 'uv'}
    peak = max(float(np.max(np.abs(E['u']))), float(np.max(np.abs(E['v']))))
    edge = _edge(E, window.sides) / peak
    if edge > EDGE:
        raise UnsupportedFeatureError(
            f'the beam reaches {edge:.2g} of its peak on the outermost cells of the plane '
            f'(at most {EDGE:g}); the sheet stops at the window, so the cross-section must '
            f'hold the beam. Widen the domain, or move the plane nearer the focus.',
            code='source.beam_size', path=path)
    # Kernel flux at the centre frequency: node-plane E against the mean of the two H planes.
    Hm = {c: 0.5 * (fields['H'][c] + fields['H_other'][c]) * masks['H' + c] for c in 'uv'}
    flux = 0.5 * float(np.sum((E['u'] * np.conj(Hm['v']) - E['v'] * np.conj(Hm['u'])).real)) * dx * dx
    if not sign * flux > 0:
        raise ValidationError('the beam carries no power through its window',
                              code='source.beam_size', path=path)
    scale = 1.0 / math.sqrt(abs(flux) * window.measure)
    patterns = {}
    for c in 'uv':
        for part, f in (('re', np.real), ('im', np.imag)):
            patterns[f'E{c}_{part}'] = f(E[c]) * scale
            patterns[f'H{c}_{part}'] = f(Ht[c]) * scale
    components = {f'E{c}': ((f'E{c}_re', 0), (f'E{c}_im', 1)) for c in 'uv'}
    components.update({f'H{c}': ((f'H{c}_re', 2), (f'H{c}_im', 3)) for c in 'uv'})
    omega0 = 2.0 * math.pi / wavelength
    delta = dx / (2.0 * _axial_group_velocity(window, wavelength))
    half = 0.5 * math.pi
    row_offsets = ((half, 0.0), (0.0, 0.0), (half - omega0 * delta, -delta),
                   (-omega0 * delta, -delta))
    reference = dict(wavelength=wavelength, beta=axial, window=window, eps=eps, mu=mu,
                     waist=waist, focus=focus, polarization=polarization, scale=scale,
                     edge_ratio=edge, position=w_node, measure=window.measure, pad=pad)
    u, v = window.u, window.v
    return ModeSheet(name=source.name, band=band, amplitude=source.amplitude,
                     phase=source.phase, delay=units._time_to_internal(source.delay_seconds),
                     axis=w, k=window.k, sign=sign,
                     window=(window.ranges[u], window.ranges[v]), patterns=patterns,
                     row_offsets=row_offsets, reference=reference, components=components,
                     shifts=(0.0, 0.0, 0.5, 0.5), kind='GaussianBeam')


def _edge(E, sides):
    rim = 0.0
    for c in 'uv':
        a = np.abs(E[c])
        (ul, uh), (vl, vh) = sides
        if ul.kind != 'invariant':
            rim = max(rim, float(np.max(a[:2, :])), float(np.max(a[-2:, :])))
        if vl.kind != 'invariant':
            rim = max(rim, float(np.max(a[:, :2])), float(np.max(a[:, -2:])))
    return rim


def launched_power(sheet, plan, rt, wavelengths):
    """``(forward, backward)``, ``(nf,)``: the spectral energy the sheet launches each way.

    In the units a :class:`FluxMonitor` reports before its spectral scaling (kernel flux times
    the DFT's squared magnitude). The sheet's incident planes, truncated to its window, are
    decomposed on the padded transform grid into the grid's discrete plane waves; for each
    transverse wavenumber and each of its two polarizations, the pair (node-plane E, upstream
    H) is uniquely a forward plus a backward wave, and their powers add.
    """
    from ._mode_source import waveform_spectra
    from ._kernel.grid import OFFSETS
    window = sheet.reference['window']
    eps, mu = sheet.reference['eps'], sheet.reference['mu']
    w, u, v = window.w, window.u, window.v
    dx, dt = window.dx, window.dt
    sign = sheet.sign
    pad = sheet.reference['pad']
    spectra = [np.asarray(_np(g)) for g in waveform_spectra(sheet, plan, rt, wavelengths)]
    (u0, u1), (v0, v1) = window.ranges[u], window.ranges[v]
    Nu, Nv = u1 - u0, v1 - v0
    invariant = [s[0].kind == 'invariant' for s in window.sides]
    Mu = 1 if invariant[0] else 1 << int(math.ceil(math.log2(pad * Nu)))
    Mv = 1 if invariant[1] else 1 << int(math.ceil(math.log2(pad * Nv)))
    ku = 2.0 * np.pi * np.fft.fftfreq(Mu, dx)[:, None]
    kv = 2.0 * np.pi * np.fft.fftfreq(Mv, dx)[None, :]
    qu = (2.0 / dx) * np.sin(ku * dx / 2.0)
    qv = (2.0 / dx) * np.sin(kv * dx / 2.0)
    p = sheet.patterns

    def spectrum(field, c, values):
        padded = np.zeros((Mu, Mv), dtype=complex)
        padded[:Nu, :Nv] = values
        off = OFFSETS[field][{'u': u, 'v': v}[c]]
        return np.fft.fft2(padded) * np.exp(-1j * (ku * off[u] + kv * off[v]) * dx / 2.0)

    forward, backward = [], []
    for i, lam in enumerate(wavelengths):
        omega = 2.0 * math.pi / float(lam)
        Omega = 2.0 * math.sin(omega * dt / 2.0) / dt
        qw2 = Omega ** 2 * eps * mu - qu ** 2 - qv ** 2
        prop = (qw2 > 0) & (np.sqrt(np.maximum(qw2, 0.0)) * dx / 2.0 < 1.0)
        qw = np.where(prop, np.sqrt(np.maximum(qw2, 0.0)), 1.0)
        theta = np.where(prop, np.arcsin(np.minimum(qw * dx / 2.0, 1.0)), 0.0)
        Gc_E, Gs_E, Gc_H, Gs_H = (s[i] for s in spectra)
        E = {c: spectrum('E', c, p[f'E{c}_re'] * Gc_E + p[f'E{c}_im'] * Gs_E) for c in 'uv'}
        H = {c: spectrum('H', c, p[f'H{c}_re'] * Gc_H + p[f'H{c}_im'] * Gs_H) for c in 'uv'}
        # Admittance of a wave travelling +-w: H_t = Y E_t, from H = q x E / (Omega mu) with
        # E_w = -(q_u E_u + q_v E_v) / q_w.
        f_tot, b_tot = 0.0, 0.0
        Y = {}
        for s_dir in (1.0, -1.0):
            q = s_dir * qw
            # H_u = (q_v E_w - q E_v)/(Om mu), H_v = (q E_u - q_u E_w)/(Om mu)
            Y[s_dir] = (((-qv * qu / q), (-qv * qv / q - q)),
                        ((q + qu * qu / q), (qu * qv / q)))
        # Upstream half plane: the travelling-with wave lags by theta there, the other leads.
        up = np.exp(-1j * theta)
        dn = np.exp(1j * theta)
        a = {1.0: up, -1.0: dn} if sign > 0 else {1.0: dn, -1.0: up}
        # Solve E = Ef + Eb, H = Y+ Ef a+ + Y- Eb a- per wavenumber (2x2 for (u, v)).
        M = np.zeros(qu.shape + (4, 4), dtype=complex)
        rhs = np.zeros(qu.shape + (4,), dtype=complex)
        scale = 1.0 / (Omega * mu)
        M[..., 0, 0] = M[..., 1, 1] = M[..., 0, 2] = M[..., 1, 3] = 1.0
        for j, comp in enumerate(('u', 'v')):
            for col_base, s_dir in ((0, 1.0), (2, -1.0)):
                for k_e in (0, 1):
                    M[..., 2 + j, col_base + k_e] = Y[s_dir][j][k_e] * scale * a[s_dir]
        rhs[..., 0], rhs[..., 1] = E['u'], E['v']
        rhs[..., 2], rhs[..., 3] = H['u'], H['v']
        sol = np.zeros_like(rhs)
        sol[prop] = np.linalg.solve(M[prop], rhs[prop][..., None])[..., 0]
        cos = np.cos(theta)
        for col_base, s_dir, acc in ((0, 1.0, 'f'), (2, -1.0, 'b')):
            Eu_, Ev_ = sol[..., col_base], sol[..., col_base + 1]
            Hu_ = (Y[s_dir][0][0] * Eu_ + Y[s_dir][0][1] * Ev_) * scale
            Hv_ = (Y[s_dir][1][0] * Eu_ + Y[s_dir][1][1] * Ev_) * scale
            # Parseval over the padded grid (paired components share their transverse points):
            # the kernel flux 0.5 Re sum E_t x conj(H_mean) dx^2, H_mean = cos(theta) H at the
            # node plane.
            flux = (Eu_ * np.conj(Hv_) - Ev_ * np.conj(Hu_)).real * cos
            power = 0.5 * float(np.sum(np.where(prop, flux, 0.0))) * dx * dx / (Mu * Mv)
            if acc == 'f':
                f_tot = power
            else:
                b_tot = -power
        forward.append(f_tot if sign > 0 else b_tot)
        backward.append(b_tot if sign > 0 else f_tot)
    return np.asarray(forward), np.asarray(backward)


def _np(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    return np.asarray(value)
