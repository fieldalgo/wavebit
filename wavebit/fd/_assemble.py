"""Stage 3: turn a finished run's accumulators and source series into the monitors' results.

``_run.py`` calls :func:`monitor_results` and :func:`decay_report` once per run, and the
``SourceSeries`` methods call :func:`source_power`, :func:`source_intensity` and
:func:`incident_amplitude`; the arithmetic is ``_post.py``'s and every derivation lives there.
This module only knows which accumulator belongs to which monitor, which source normalises
which result, and when a finished run was stopped too early (``UnsettledFieldWarning``). Every
refusal a scene can be given is raised at plan time instead (``_lower/monitors.py``).
"""
import math
import warnings

import numpy as np

from . import _post
from ._lower.monitors import (FACE_SEP, box_faces, diffraction_layout, diffraction_window,
                              farfield_checks, field_points, plane_layout)
from .diagnostics import UnsettledFieldWarning
from .monitors import Diffraction, FarField, Field, Flux, Record

__all__ = ['monitor_results', 'source_power', 'source_intensity', 'incident_amplitude',
           'decay_report', 'UNSETTLED_THRESHOLD']

#: The decay diagnostic's line: the change of the region energy over the last window, over its
#: peak (``_post.decay_ratio``). The plan's starting point was Tidy3D's 1e-5 field-energy
#: shutoff; measured 2026-09-19 on two quick scenes, a flux box (0.5 on a side) around an ``Ez``
#: dipole at 20 cells per wavelength, the flux at wavelengths 0.9 / 1.0 / 1.1 against a long
#: reference run of the same scene:
#:
#:     open PML box (1.0 interior)       ratio      flux error
#:        7 periods                      5.5e-2     1.0e-2
#:        8 periods                      8.4e-4     4.7e-4
#:        9 periods                      2.6e-6     4.8e-6
#:       10 / 12 / 16 periods            2.5e-9 .. 7e-11   4e-8 .. 7e-8   (the floor)
#:     4-cell Absorber box (1.2)
#:        8 periods                      3.7e-4     1.1e-4
#:       12 .. 80 periods                2.4e-10 .. 6e-12  2.6e-7 .. 1.2e-8
#:
#: so the flux error runs at 0.3 to 2 times the ratio, a settled run sits at or below 2.5e-9,
#: and one period short of settled is already 3.7e-4. 1e-5 separates the two populations by
#: more than a decade each way and keeps the truncation error of a run that passes near 1e-5;
#: it is kept.
UNSETTLED_THRESHOLD = 1e-5


def _xp_freqs(wavelengths):
    return np.asarray([1.0 / w for w in wavelengths], dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------------------------

def source_power(series, wavelengths):
    """``(nf,)``: the power a Dipole delivered, ``-0.5 Re (conj(w^) F^)`` (``_post`` module).

    ``w^`` is the drive's phasor at its own times (the current's), ``F^`` the stamp-weighted
    field's at the field's: the frequency-domain ``-0.5 Re sum J^* . E^ dV`` of (*) exactly, which
    the flux out of any closed surface around the dipole equals in a lossless medium whatever
    the medium is. Not the phasor of the ``work`` series: that product averages the two E levels
    around the current's time, which in the frequency domain is ``E^ cos(w dt/2)`` -- a 1e-3
    bias at 20 cells per wavelength that the flux does not have.
    """
    ctx = series.context
    rt = ctx.rt
    freqs = _xp_freqs(wavelengths)
    dt = ctx.plan.dt
    w = _post.dft(rt, series.drive, series.t, dt, freqs)
    F = _post.dft(rt, series.field, series.t_field, dt, freqs, leapfrog=True)
    return -0.5 * rt.xp.real(rt.xp.conj(w) * F)


def _column_frame(series):
    """``(P, u, v, sign, source)`` of a PlaneWave or TFSF source's column."""
    ctx = series.context
    term = ctx.term
    src = next(s for s in ctx.scene.sources if s.name == series.name)
    P = term.axis
    return P, (P + 1) % 3, (P + 2) % 3, src.sign, src


def _incident_split(series, wavelengths):
    """The incident-direction wave at the column's entry plane, per frequency.

    Returns a dict with the tangential ``Eu``, ``Ev`` of the wave travelling the source's way
    (at the entry node plane, the stagger removed and the transverse phase referred to the
    origin), the wave's ``KP``, ``kappa``, ``cos``, ``Omega``, the medium and ``K_u``, ``K_v``.
    A TFSF column over layers holds the layers' reflection too; the E/H split of ``_post``
    removes it. A PlaneWave column holds the incident wave alone and the split changes nothing.
    """
    ctx = series.context
    rt, plan, scene = ctx.rt, ctx.plan, ctx.scene
    xp = rt.xp
    grid = plan.grid
    dx, dt = grid.dx, plan.dt
    P, u, v, sign, src = _column_frame(series)
    freqs = _xp_freqs(wavelengths)
    omega = 2.0 * math.pi * freqs
    Omega = rt.array(2.0 * np.sin(omega * dt / 2.0) / dt, 'float64')
    E = _post.dft(rt, series.incident, series.t, dt, freqs, leapfrog=True)       # (nf, 3)
    H = _post.dft(rt, series.incident_H, series.t_H, dt, freqs, leapfrog=True)   # (nf, 3)
    entry = int(round((series.plane - grid.low[P]) / dx))
    eps, mu = _post.medium_at(scene, grid, P, (entry - 1, entry),
                              f'{type(src).__name__}({src.name!r}) normalisation')
    eps, mu = rt.array(eps, 'float64'), rt.array(mu, 'float64')
    if type(src).__name__ == 'PlaneWave':
        ku, kv = _post.bloch_k(rt, scene, grid, u), _post.bloch_k(rt, scene, grid, v)
    else:
        ku = kv = rt.array(0.0, 'float64')
    Ku = _post.discrete_k(rt, scene, u, ku, dx)
    Kv = _post.discrete_k(rt, scene, v, kv, dx)
    KP, kappa, cos, Minv = _post.plane_wave_matrices(rt, Ku, Kv, Omega, eps, mu, dx)
    # The column is cell (0, 0): E_u sits half a cell along u, E_v half a cell along v; H_u
    # half along v, H_v half along u (both also half along P -- handled by theta below).
    # Referred to the cell's low corner. An Axis1D axis has no stagger: its field is
    # exp(i k a) with one phase reference for every component.
    hu = 0.0 if _post.is_axis1d(scene, u) else dx / 2.0
    hv = 0.0 if _post.is_axis1d(scene, v) else dx / 2.0
    ph_u = xp.exp(-1j * (ku * hu))
    ph_v = xp.exp(-1j * (kv * hv))
    Eu, Ev = E[:, u] * ph_u, E[:, v] * ph_v
    Hu, Hv = H[:, u] * ph_v, H[:, v] * ph_u
    # H is on the half plane plane_H, sigma half a cell from the node: H_rec = M(+)[e^{i th} E+ -
    # e^{-i th} E-] with th = kappa sigma dx/2, so D = M^-1 H_rec = e^{i th} E+ - e^{-i th} E- and
    # E+ = (D + e^{-i th} E) / (2 cos th).
    sigma = 1.0 if series.plane_H > series.plane else -1.0
    th = kappa * (sigma * dx / 2.0)
    Du = Minv['uu'] * Hu + Minv['uv'] * Hv
    Dv = Minv['vu'] * Hu + Minv['vv'] * Hv
    back = xp.exp(-1j * th)
    Eu_p, Ev_p = (Du + back * Eu) / (2.0 * xp.cos(th)), (Dv + back * Ev) / (2.0 * xp.cos(th))
    if sign > 0:
        Iu, Iv = Eu_p, Ev_p
    else:
        Iu, Iv = Eu - Eu_p, Ev - Ev_p
    return dict(Eu=Iu, Ev=Iv, KP=KP, kappa=kappa, cos=cos, Omega=Omega, eps=eps, mu=mu,
                Ku=Ku, Kv=Kv, ku=ku, kv=kv, sign=sign, P=P, u=u, v=v, src=src,
                omega=rt.array(omega, 'float64'))


def source_intensity(series, wavelengths):
    """``(nf,)``: the incident wave's flux per unit area along its direction, at the entry plane.

    ``0.5 Re (E_t x conj(H_mean)) . P_hat`` of the incident-direction wave alone, with ``H_mean``
    the mean of the two half planes about the entry node plane (``_post``) -- the same pairing a
    ``Flux`` plane makes, so an empty box's face flux over its area is this number.
    """
    w = _incident_split(series, wavelengths)
    return _post.wave_flux(series.context.rt, w['Eu'], w['Ev'], w['Ku'], w['Kv'], w['KP'],
                           w['Omega'], w['mu'], w['cos'], w['sign'])


def incident_amplitude(series, wavelengths, plane=None):
    """``(nf,)`` complex: the incident wave's amplitude along its polarisation unit vector.

    A ``PlaneWave``'s polarisation is s or p in the frame of ``_post`` at its own ``K_par``; a
    TFSF's is its ``polarization`` axis. Referred to the scene's transverse origin and to the
    entry plane, or to ``plane`` (a position along the propagation axis) by the Yee kz.
    """
    ctx = series.context
    rt = ctx.rt
    xp = rt.xp
    w = _incident_split(series, wavelengths)
    src = w['src']
    if type(src).__name__ == 'PlaneWave':
        s_u, s_v, q_u, q_v = _post.frame(rt, w['Ku'], w['Kv'], math.radians(src.phi))
        Knorm = w['Omega'] * xp.sqrt(w['eps'] * w['mu'])
        Es, Ep = _post.sp_amplitudes(w['Eu'], w['Ev'], s_u, s_v, q_u, q_v, w['KP'], Knorm,
                                     w['sign'])
        A = Es if src.polarization == 's' else Ep
    else:
        A = w['Eu'] if src.pol == w['u'] else w['Ev']
    if plane is not None:
        kz = _post.order_kz(rt, w['kappa'], w['ku'], w['kv'],
                            w['omega'] * xp.sqrt(w['eps'] * w['mu']))
        A = A * xp.exp(1j * (w['sign'] * kz) * (plane - series.plane))
    return A


# ---------------------------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------------------------

def consistent_accumulators(plan, state, rt, params=None):
    """``{region: {component: (nf, ...)}}``: every DFT accumulator with its end correction.

    ``_post.end_correction`` with the final field in the region (``state``'s E at ``N dt``, H at
    ``(N - 1/2) dt``) and the initial one (an ``InitialField``'s, masked as the kernel masks it;
    zero for a source run). The phasors every result is built from.
    """
    dt, N = plan.dt, plan.steps
    E0 = H0 = None
    if params is not None and (plan.has_E0 or plan.has_H0):
        from ._kernel.step import initial_state
        start = initial_state(plan, params, rt)
        E0 = start['E'] if plan.has_E0 else None
        H0 = start['H'] if plan.has_H0 else None
    out = {}
    for o in plan.observers:
        if not hasattr(o, 'freqs'):
            continue
        acc = state['obs'][o.name]
        fixed = {}
        for label, field, slab in o.parts:
            last = (state['E'] if field == 'E' else state['H'])[slab.slices]
            first = E0 if field == 'E' else H0
            first = None if first is None else first[slab.slices]
            t_last = N * dt if field == 'E' else (N - 0.5) * dt
            t_first = 0.0 if field == 'E' else -0.5 * dt
            fixed[label] = acc[label] + _post.end_correction(rt, last, t_last, dt, o.freqs,
                                                             first=first, t_first=t_first)
        out[o.name] = fixed
    return out


def monitor_results(scene, plan, state, rt, sources, params=None):
    """``{monitor name: result}`` for every stage-3 monitor; records are left to the caller."""
    from .results import DiffractionResult, FarFieldResult, FieldResult, FluxResult
    grid = plan.grid
    obs = consistent_accumulators(plan, state, rt, params)
    out = {}
    for m in scene.monitors:
        if isinstance(m, Record):
            continue
        if isinstance(m, FarField):
            where = f'FarField({m.name!r})'
            eps, mu = farfield_checks(m, scene, grid)
            faces = box_faces(where, grid, m.box)
            accs = {face: obs[m.name + FACE_SEP + face] for face, _, _ in faces}
            net, parts = _post.box_flux(rt, accs, faces, grid.dx)
            out[m.name] = FarFieldResult(m.name, m.wavelengths, rt, accs, faces, grid, eps, mu,
                                         net, parts)
        elif isinstance(m, Flux):
            where = f'Flux({m.name!r})'
            if m.box is not None:
                faces = box_faces(where, grid, m.box)
                accs = {face: obs[m.name + FACE_SEP + face] for face, _, _ in faces}
                net, parts = _post.box_flux(rt, accs, faces, grid.dx)
                out[m.name] = FluxResult(name=m.name, wavelengths=m.wavelengths, flux=net,
                                         faces=parts)
            else:
                extents = tuple(None if a == m.plane.axis else m.plane.extent(a)
                                for a in range(3))
                layout = plane_layout(where, grid, m.plane.axis, m.plane.position, extents)
                flux = _post.plane_flux(rt, obs[m.name], layout, grid.dx)
                out[m.name] = FluxResult(name=m.name, wavelengths=m.wavelengths, flux=flux,
                                         faces=None)
        elif isinstance(m, Field):
            points = field_points(m, grid)
            out[m.name] = FieldResult(
                name=m.name, wavelengths=m.wavelengths,
                fields={c: obs[m.name][c] for c in m.components},
                coords={c: points[c][1] for c in m.components})
        elif isinstance(m, Diffraction):
            given = _diffraction_sub(m, scene, plan, obs[m.name], rt, sources)
            out[m.name] = DiffractionResult(monitor=m, given=given['sub'], extras=given)
    return out


def _diffraction_sub(monitor, scene, plan, acc, rt, sources):
    """The ``ResultSub`` of one Diffraction monitor for the run's own incidence, and extras."""
    from wavebit._core.result import ResultSub
    xp = rt.xp
    grid = plan.grid
    dx, dt = grid.dx, plan.dt
    layout, pw = diffraction_layout(monitor, scene, grid)
    P, u, v = layout.P, layout.u, layout.v
    where = f'Diffraction({monitor.name!r})'
    eps, mu = _post.medium_at(scene, grid, P, (layout.k - 1, layout.k), where)
    series = sources[pw.name]
    inc = _incident_split(series, monitor.wavelengths)
    # Which way the monitored waves go.
    out_dir = -pw.sign if monitor.side == 'reflect' else pw.sign
    z_m = grid.low[P] + layout.k * dx
    z_ref = z_m if monitor.reference_plane is None else monitor.reference_plane
    z_inc = z_ref if monitor.incident_plane is None else monitor.incident_plane
    Mu, Mv = diffraction_window(monitor, scene, grid, eps, mu)
    ku0, kv0 = inc['ku'], inc['kv']
    m_u = np.arange(-Mu, Mu + 1)
    m_v = np.arange(-Mv, Mv + 1)
    ku = ku0 + rt.array(2.0 * math.pi * m_u / grid.size[u], 'float64')
    kv = kv0 + rt.array(2.0 * math.pi * m_v / grid.size[v], 'float64')
    # The orders' amplitudes of the four tangential components, (nf, n_v, n_u). An Axis1D
    # axis carries no stagger (its field is exp(i k a) with one phase reference).
    su = 0 if _post.is_axis1d(scene, u) else 1
    sv = 0 if _post.is_axis1d(scene, v) else 1
    Eu = _post.order_coefficients(rt, acc['Eu'], P, u, v, ku, kv, grid, (su, 0))
    Ev = _post.order_coefficients(rt, acc['Ev'], P, u, v, ku, kv, grid, (0, sv))
    Hu = _post.order_coefficients(rt, _post._mean_planes(acc['Hu'], P), P, u, v, ku, kv, grid,
                                  (0, sv))
    Hv = _post.order_coefficients(rt, _post._mean_planes(acc['Hv'], P), P, u, v, ku, kv, grid,
                                  (su, 0))
    Omega = inc['Omega'][:, None, None]
    eps_a, mu_a = rt.array(eps, 'float64'), rt.array(mu, 'float64')
    Ku = _post.discrete_k(rt, scene, u, ku, dx)[None, None, :]
    Kv = _post.discrete_k(rt, scene, v, kv, dx)[None, :, None]
    KP, kappa, cos, Minv = _post.plane_wave_matrices(rt, Ku, Kv, Omega, eps_a, mu_a, dx)
    plus, minus = _post.split_directions(Eu, Ev, Hu, Hv, Minv, cos)
    Eo_u, Eo_v = plus if out_dir > 0 else minus
    # Flux of each order, split s/p, at the monitor plane.
    s_u, s_v, q_u, q_v = _post.frame(rt, Ku, Kv, math.radians(pw.phi))
    Es_m = Eo_u * s_u + Eo_v * s_v
    Sp = _post.wave_flux(rt, Eo_u - Es_m * s_u, Eo_v - Es_m * s_v, Ku, Kv, KP, Omega, mu_a,
                         cos, out_dir)
    Ss = _post.wave_flux(rt, Es_m * s_u, Es_m * s_v, Ku, Kv, KP, Omega, mu_a, cos, out_dir)
    S_inc = _post.wave_flux(rt, inc['Eu'], inc['Ev'], inc['Ku'], inc['Kv'], inc['KP'],
                            inc['Omega'], inc['mu'], inc['cos'], inc['sign'])[:, None, None]
    # De-embedding: each order by its own kz, from the monitor plane to the reference plane.
    omega = inc['omega'][:, None, None]
    n_omega = omega * xp.sqrt(eps_a * mu_a)
    kz = _post.order_kz(rt, kappa, ku[None, None, :], kv[None, :, None], n_omega)
    shift = xp.exp(1j * (out_dir * kz) * (z_ref - z_m))
    Knorm = Omega * xp.sqrt(eps_a * mu_a)
    Es, Ep = _post.sp_amplitudes(Eo_u * shift, Eo_v * shift, s_u, s_v, q_u, q_v, KP, Knorm,
                                 out_dir)
    A_inc = incident_amplitude(series, monitor.wavelengths, plane=z_inc)[:, None, None]
    r_s, r_p = Es / A_inc, Ep / A_inc
    R_s, R_p = Ss / S_inc, Sp / S_inc
    R = R_s + R_p
    if monitor.side == 'reflect':
        sub = ResultSub(r_s, r_p, None, None, R, R_s, R_p, None, None, None)
    else:
        sub = ResultSub(None, None, r_s, r_p, None, None, None, R, R_s, R_p)
    n_inc = xp.sqrt(inc['eps'] * inc['mu'])
    kpar = xp.sqrt(ku0 ** 2 + kv0 ** 2)
    sin_theta = kpar / (n_inc * inc['omega'])
    theta_at = xp.asin(sin_theta) * (180.0 / math.pi)
    return dict(sub=sub, order_window=((0, int(Mu)), (0, int(Mv))), theta_at=theta_at,
                lateral_origin=(grid.low[u], grid.low[v]),
                reference_plane=z_ref, incident_plane=z_inc, plane=z_m,
                kz=kz, polarization=pw.polarization, source=pw.name)


# ---------------------------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------------------------

def _np(rt, x):
    """``x`` as NumPy, or ``None`` when it is a tracer (a run under ``jax.jit``)."""
    try:
        return np.asarray(rt.numpy(x))
    except Exception:
        return None


def decay_report(scene, plan, state, rt, sources):
    """``{name: ratio}`` for every frequency-domain monitor, and a warning over the line.

    The ratio is how much the region energy ``sum |F|^2`` over the monitor's region changes in
    the run's **last window** -- one period of the longest wavelength it asks for -- divided by
    its largest value anywhere in the run (``_post.decay_ratio``: the change and not the level,
    because a static residue is exact after ``_post.end_correction``). The measured relation to
    the truncation error of a flux is at :data:`UNSETTLED_THRESHOLD`, where the warning starts.
    A region that never saw a field (a scattered-field box around an empty TFSF, peak energy
    below 1e-20 of the loudest monitor's) reports 0. Records and sources are reported too (``records``,
    ``sources``: the last twentieth of the run against the peak), for reading, and never warn:
    a time series is not a spectral claim. Under ``jax.jit`` the series are tracers and the
    ratios are ``None`` -- the diagnostic reads values, and a trace has none yet. Under a bare
    ``jax.grad`` (no jit) the energy series are not on the differentiated path and come back
    concrete, so the ratios are read and the warning fires exactly as on NumPy
    (``tests/test_fd_decay.py`` pins both).
    """
    obs = state['obs']
    dt = plan.dt
    tail = max(1, plan.steps // 20)
    monitors, records, srcs = {}, {}, {}
    for m in scene.monitors:
        if isinstance(m, Record):
            v = _np(rt, obs[m.name])
            records[m.name] = None if v is None else _post.decay_ratio(np.abs(v) ** 2, tail)
            continue
        window = int(math.ceil(max(m.wavelengths) / dt))
        energy = None
        for o in plan.observers:
            if getattr(o, 'monitor', None) == m.name:
                e = _np(rt, obs[o.name]['energy'])
                if e is None:
                    energy = None
                    break
                energy = e if energy is None else energy + e
        monitors[m.name] = None if energy is None else (window, energy)
    loudest = max((float(np.max(e)) for w, e in (v for v in monitors.values() if v is not None)
                   if e.size), default=0.0)
    for name, entry in monitors.items():
        if entry is None:
            continue
        window, energy = entry
        quiet = not float(np.max(energy)) > 1e-20 * loudest if energy.size else True
        monitors[name] = 0.0 if quiet else _post.decay_ratio(energy, window)
    for name, series in (sources or {}).items():
        values = series.incident if series.incident is not None else series.field
        v = None if values is None else _np(rt, values)
        if v is None:
            continue
        v = np.abs(v)
        e = v ** 2 if v.ndim == 1 else np.sum(v ** 2, axis=1)
        srcs[name] = _post.decay_ratio(e, tail)
    for name, ratio in monitors.items():
        if ratio is not None and ratio > UNSETTLED_THRESHOLD:
            warnings.warn(
                f'monitor {name!r}: the field in its region was still {ratio:.2e} of its peak '
                f'energy in the last period of its longest wavelength (the line is '
                f'{UNSETTLED_THRESHOLD:.0e}). Its DFT is of a signal cut off before it '
                f'decayed, so the phasors -- and every flux, order and normalisation built on '
                f'them -- carry that truncation. Run longer (Run(steps=), Run(periods=), '
                f'Run(time=)).', UnsettledFieldWarning, stacklevel=4)
    return dict(decay=monitors, records=records, sources=srcs,
                threshold=UNSETTLED_THRESHOLD)
