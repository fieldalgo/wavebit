"""Stage 3: turn a finished run's accumulators and source series into the monitors' results.

``_run.py`` calls :func:`monitor_results` and :func:`decay_report` once per run, and the
``SourceSeries`` methods call :func:`source_power`, :func:`source_intensity` and
:func:`incident_amplitude`; the arithmetic is ``_post.py``'s and every derivation lives there.
This module only knows which accumulator belongs to which monitor, which source normalises
which result, and when a finished run was stopped too early (``UnsettledFieldWarning``) or
still grows after its sources ended (``GrowingFieldWarning``). Every refusal a scene can be
given is raised at plan time instead (``src/wavebit/fdtd/_prepare/monitors.py``).
"""
import math
import warnings

import numpy as np

import wavebit.fdtd._post as _post
from wavebit.fdtd._prepare.cpml import MEASURED_STABILIZE
from wavebit.fdtd._prepare.cpml import spell
from wavebit.fdtd._prepare.cpml import sufficient_stabilize
from wavebit.fdtd._spelling import spoken
from wavebit.fdtd._prepare.monitors import FACE_SEP
from wavebit.fdtd._prepare.monitors import box_faces
from wavebit.fdtd._prepare.monitors import diffraction_layout
from wavebit.fdtd._prepare.monitors import diffraction_window
from wavebit.fdtd._prepare.monitors import farfield_checks
from wavebit.fdtd._prepare.monitors import field_points
from wavebit.fdtd._prepare.monitors import plane_layout
from wavebit.fdtd._public_names import public_name
from wavebit.fdtd.diagnostics import GrowingFieldWarning
from wavebit.fdtd.diagnostics import UnsettledFieldWarning
from wavebit.fdtd._legacy_monitors import Diffraction
from wavebit.fdtd._legacy_monitors import FarField
from wavebit.fdtd._legacy_monitors import Field
from wavebit.fdtd._legacy_monitors import Flux
from wavebit.fdtd._legacy_monitors import Record

__all__ = ['monitor_results', 'source_power', 'source_intensity', 'incident_amplitude',
           'decay_report', 'sources_end', 'column_growth', 'interior_energy', 'UNSETTLED_THRESHOLD',
           'COLUMN_LEVEL', 'GROWTH_THRESHOLD', 'GROWTH_STEADY', 'GROWTH_FLOOR', 'GROWTH_WINDOWS']

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

#: The growth diagnostic's line: the interior's energy
#: (``src/wavebit/fdtd/_kernel/observe.py::InteriorEnergy``) averaging more than this many times
#: higher over the last quarter of the span after every source has ended than over the
#: quarter before (``_post.growth_ratio``), and rising as an exponential does
#: (:data:`GROWTH_STEADY`), is growing. A passive scene's reads at most 1, up to what a PML
#: hands back -- what the reading itself is not is gone, it being the leapfrog's exactly
#: conserved energy. Measured 2026-09-20 (NumPy, the readings of this module and
#: ``InteriorEnergy`` as they stand) over the interior of every ``fd.run`` in the fd suite --
#: 1552 series in the default tier and 157 in the slow one, of which 365 and 134 are judged at
#: their own run length (the rest end within :data:`GROWTH_WINDOWS` periods of their sources,
#: under :data:`GROWTH_FLOOR`, or are mutants and declared growths) -- and over scenes built to
#: grow or to hold, each judged at every run length its one run holds (a run's prefix is the
#: shorter run). **Passive**, the largest ratio anything reached, and (after the arrow) what
#: the verdict reads once the spread is asked too:
#:
#:     the suite, at each run's own length: 1.0006 (the stabilized, PEC and slab-less control
#:       columns of ``test_fd_pml_stability.py``); at every shorter length 1.0019; nothing over
#:       the line at any length, and nothing warns
#:     the slow tier: 1.156 (the conical grating at 160 periods, spread 2.43) and 1.085 (the
#:       crossed grating at 480, spread inf) -- both of the growing family, missed at that
#:       length, not false; everything else 1.0000
#:     lossless Lorentz (10, 40 omega0^2) and Drude (3, 6 omega) cavities, courant 0.5 and 0.7:
#:       1.0000 at every length; a Debye cavity 0.86
#:     the review's TFSF echo (glass 3.5 below a vacuum box) and the same with a glass layer in
#:       the box, PlaneWave columns ended by ``Absorber(12 .. 64)``: never judged at any length
#:       to 30 .. 40 (their end is read off the column, ``sources_end``, and the interior is
#:       under the floor after it); a pulse kicked into ``Absorber(12)`` and ``(32)`` columns:
#:       0.99, 0.96 (the review read 2.57 with the layer left out)
#:     a dipole in ``PML(24, 1e-3)``, ``(12, 1e-3)``, ``(8, 1e-2)``; a pulse kicked into a 3-D
#:       box under ``PML(6, 1e-2)`` and ``(12, 1e-3)``: 1.0000 .. 1.0003
#:     a pulse kicked into a vacuum column under a deep weak layer, ``PML(64, 1e-3)``,
#:       ``(48, 1e-2)``, ``(32, 1e-2)``: the wall's echo after a quiet gap, 1.8e4, 210 and 3.6
#:       -> under the line (spreads 4.2, 7.8 and 18)
#:     the exact Yee modes (14, 12, 12) and (14, 13, 11) of a closed 16^3 PEC box, which the
#:       first version's reading aliased into 1.165 and 1.38 (``ENERGY_STRIDE``): 1.0000 at
#:       2000 and 3000 steps, courant 0.5 and 0.9
#:     earlier, with the first version's statistic: a glass slab 8 thick, free or PEC-backed
#:       (1.0000, where the monitors read 1.3e9), PEC boxes with two near-degenerate modes and
#:       half-cavities coupled through a membrane (1.0000, Records to 2.8), the closed slab
#:       column (15 .. 400), ``pec_cavity_dipole`` (20 .. 160 periods), pulses through lossless
#:       Lorentz and Drude slabs (0.84)
#:
#: **Growing** (the pure stretch; the factor rises with the run, since a quarter is longer; the
#: spread in brackets, and "missed" where it is over :data:`GROWTH_STEADY` at that length):
#:
#:     slab column (eps 1.5, 1e-8)      t = 50: 1.06 (1.35)  100: 1.34 (1.64)  150: 2.19 (1.52)
#:                                      200: 3.95 (1.28)
#:     the same at reflection=1e-4      t = 50: 1.17 (1.32)  100: 1.99 (1.42)  150: 4.00 (1.18)
#:                                      200: 7.41 (1.06)
#:     eps 4, 0.08 thick, kx = 2pi/1.6  t = 100: 1.02 (1.46)  200: 1.27 (2.73, missed)
#:                                      300: 3.13 (2.18, missed)  400: 9.67 (1.34)
#:     the same at kx = 2pi/1.2         t = 100: 1.04 (1.43)  200: 1.35 (2.12, missed)
#:                                      300: 2.83 (1.74)  400: 6.65 (1.27); at kx = 2pi/0.8 it
#:                                      does not grow (0.99 at 400)
#:     eps 2.25, 0.24 thick, kx=2pi/1.2 t = 100: 1.02 (1.49)  200: 1.04  300: 1.08  400: 1.12
#:                                      (1.23); the other two kx do not grow
#:     lamellar grating, p, 12 cells    96 periods: 0.48, 192: 0.41 (still ringing down),
#:                                      288: 1.42 (its third quarter still rings down, so the
#:                                      last half does not rise throughout: missed),
#:                                      384: 2.63 (1.21), 576: 4.69 (1.003), 768: 7.94 (1.003)
#:     the same slab column driven by    t = 100: 1.89 (1.02)  200: 4.57 (1.05)  300: 9.85
#:     an oblique PlaneWave, whose       (1.02)  400: 22.0 (1.01) -- judged from the envelope's
#:     column holds (``sources_end``)    end, and warning from t = 31 on
#:     conical grating, p, 960 periods  160: 0.95  480: 1.09 (1.21)  960: 3.52 (4.14, missed:
#:                                      its last half beats, and the eighths do not rise
#:                                      evenly -- the one growth measured that a longer run
#:                                      does not obviously mend)
#:     crossed grating, s, PML(24,1e-12) 160: 0.59  480: 1.09 (inf, not rising throughout)
#:                                      960: 1.73 (1.011)
#:
#: 1.1 is 55x over the largest excess over 1 that a passive scene showed at its own length
#: (1.0006, and 1.0019 at any length) and under every growing run from the slab column at
#: t = 100 on. Missed are slow growths at the lengths where a quarter holds less than a 1.1x
#: rise (the column at t = 50, the conical grating to 480 periods, the crossed one at 480, the
#: eps-4 column at t = 100, the slowest slab families: eps 1.05 stays under 1.011 up to
#: t = 400) and a growth whose last half is still partly a ring-down or a beat (the grating at
#: 288, the eps-4 column at 200 .. 300, the conical grating at 960) -- each caught at a longer run, where the plain
#: ``UnsettledFieldWarning`` has been saying the run is not done. A false alarm on a passive
#: scene would tell its user that a longer run cannot help. **What still reads as growth**: an
#: ``InitialField`` with content near zero frequency under a PML -- a Gaussian kick
#: ``exp(-(z/0.15)^2) cos(2 pi z)`` in a vacuum column, whose zero-frequency part the layer's
#: CFS term leaves unabsorbed, returns off the wall behind it over some twenty time units and
#: reads up to 34 (spread 1.17 at its tightest), warning at 8 of the 53 lengths a 16-unit run
#: holds under the default layer, 19 under ``PML(12, 1e-4)`` and 2 under ``PML(6, 1e-3)`` (the
#: same kick 1 wide, its zero-frequency content 5e-5, never warns, and neither does it under an
#: ``Absorber``). The exact reading does not change that: the energy really does return. A
#: source cannot put it in (its band stops above zero frequency).
GROWTH_THRESHOLD = 1.1

#: How steadily a growing interior's energy must rise: the spread of ``_post.growth_ratio`` --
#: the last half of the judged span cut into four eighths, the largest of the three
#: eighth-to-eighth rises of the log of their means over the smallest -- at most this. A linear
#: instability's energy is a sum of exponentials that the fastest comes to dominate, so its log
#: rises in equal steps; energy handed back arrives with its pulse's shape, a quiet floor and
#: then a steepening edge. Measured (the table above): the growing runs are caught with spreads
#: 1.00 .. 1.64, falling toward 1 as the fastest mode takes over; the passive runs over the
#: line are the deep weak layer's echoes at 4.2, 7.8 and 18. 2 sits 1.22x over the first and
#: 1.9x under the second. **Why not 3**, which the numbers would also carry: it would catch the
#: eps-4 slab column at t = 200 .. 300 (spreads 2.1 .. 2.7, a growth still mixed with its
#: ring-down) and the suite's conical grating at 160 periods (2.43) one run length sooner -- but it
#: more than doubles the zero-frequency kick's false warnings above (8 of 53 lengths -> 21,
#: and 19 -> 36 under ``PML(12, 1e-4)``), and a missed growth is caught at a longer run where a
#: false alarm is never taken back. Suggested by the review (2026-09-20: "a single returning
#: echo must not trigger"); its first proposal, requiring the third quarter to be over the
#: second too, missed the grating at 384 periods, whose second quarter still rings down (0.79).
GROWTH_STEADY = 2.0

#: The level under which the growth ratio is not read: the last quarter's peak over the
#: interior energy's largest value in the run. A field that has decayed to its floor --
#: round-off, or the last digits of a static residue -- wanders by factors of a few from
#: quarter to quarter: over the suite, every passive run whose ratio passed the line sat at or
#: under 7.3e-10 of its peak (a TFSF box over a substrate, a PlaneWave cell's Diffraction
#: scene: ratios to 2.7). A growing interior sits higher: the lowest the growing runs above
#: reached is 1.0e-5 (the lamellar grating at 384 periods, whose growing mode is still 1e-5 of
#: what its pulse left behind). 1e-7 is 137x over the first and 100x under the second.
GROWTH_FLOOR = 1e-7

#: The fewest periods (``_scene_period``: the longest wavelength the scene names) a judged span
#: may have, so that a quarter of it holds a whole one and is longer than what crosses the
#: scene. Measured over the suite (2026-09-20, both tiers, every run length): judged from one
#: period of span, a passive scene reaches 1211 -- a TFSF box's echo caught whole inside a
#: single period, before its column's end is known -- and ordinary passive runs 1.017 .. 1.12;
#: from 4 periods up, 1.0019 and nothing over the line. A growth needs no longer span to show:
#: the interior's energy of a passive scene cannot rise over any span.
GROWTH_WINDOWS = 4


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
                              f'{public_name(src)}({src.name!r}) normalisation')
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
        from wavebit.fdtd._kernel.step import initial_state
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
    from wavebit.fdtd._legacy_results import DiffractionResult
    from wavebit.fdtd._legacy_results import FarFieldResult
    from wavebit.fdtd._legacy_results import FieldResult
    from wavebit.fdtd._legacy_results import FluxResult
    grid = plan.grid
    obs = consistent_accumulators(plan, state, rt, params)
    out = {}
    for m in scene.monitors:
        if isinstance(m, Record):
            continue
        if isinstance(m, FarField):
            where = f'FarFieldMonitor({m.name!r})'
            eps, mu = farfield_checks(m, scene, grid)
            faces = box_faces(where, grid, m.box)
            accs = {face: obs[m.name + FACE_SEP + face] for face, _, _ in faces}
            net, parts = _post.box_flux(rt, accs, faces, grid.dx)
            out[m.name] = FarFieldResult(m.name, m.wavelengths, rt, accs, faces, grid, eps, mu,
                                         net, parts, dt=plan.dt, propagation=m.propagation)
        elif isinstance(m, Flux):
            where = f'FluxMonitor({m.name!r})'
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
            incident = _diffraction_sub(m, scene, plan, obs[m.name], rt, sources)
            out[m.name] = DiffractionResult(monitor=m, incident=incident['sub'], extras=incident)
    return out


def _diffraction_sub(monitor, scene, plan, acc, rt, sources):
    """The ``ResultSub`` of one Diffraction monitor for the run's own incidence, and extras."""
    from wavebit.observations.diffraction import ResultSub
    xp = rt.xp
    grid = plan.grid
    dx, dt = grid.dx, plan.dt
    layout, pw = diffraction_layout(monitor, scene, grid)
    P, u, v = layout.P, layout.u, layout.v
    where = f'DiffractionMonitor({monitor.name!r})'
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
    # Raw plane coefficients and the incident reference both use the cell's low corner.
    # Refer their ratio to the public cell centre: the common Bloch carrier cancels and
    # only exp(i*pi*(m+n)) remains. An exact sign selection also preserves symmetry zeros.
    odd = rt.array((m_v[:, None] + m_u[None, :]) % 2 != 0)
    r_s = xp.where(odd, -Es / A_inc, Es / A_inc)
    r_p = xp.where(odd, -Ep / A_inc, Ep / A_inc)
    # The Cartesian electric vector follows the discrete, not continuum, wavevector.
    # Use the extracted tangential fields directly to avoid a singular s/p frame at kpar=0.
    e_u, e_v = Eo_u * shift / A_inc, Eo_v * shift / A_inc
    e_P = -(Ku * e_u + Kv * e_v) / (out_dir * KP)
    components = [None, None, None]
    for axis, value in ((u, e_u), (v, e_v), (P, e_P)):
        components[axis] = xp.where(odd, -value, value)
    xyz = xp.stack(components, axis=-1)
    R_s, R_p = Ss / S_inc, Sp / S_inc
    R = R_s + R_p
    if monitor.side == 'reflect':
        # This monitor may retain only a subset of orders or lie inside the structure.
        # It does not establish a complete illuminated port, so entering power is absent.
        sub = ResultSub(r_s, r_p, None, None, R, R_s, R_p, None, None, None, r_xyz=xyz)
    else:
        # A transmission monitor alone cannot infer entering power or absorption.
        sub = ResultSub(None, None, r_s, r_p, None, None, None, R, R_s, R_p, t_xyz=xyz)
    n_inc = xp.sqrt(inc['eps'] * inc['mu'])
    kpar = xp.sqrt(ku0 ** 2 + kv0 ** 2)
    sin_theta = kpar / (n_inc * inc['omega'])
    theta_at = xp.asin(sin_theta) * (180.0 / math.pi)
    return dict(sub=sub, order_window=((0, int(Mu)), (0, int(Mv))), theta_at=theta_at,
                lateral_origin=(grid.low[u] + grid.size[u] / 2.0,
                                grid.low[v] + grid.size[v] / 2.0),
                reference_plane=z_ref, incident_plane=z_inc, plane=z_m,
                kz=kz, polarization=pw.polarization, source=pw.name,
                # Keep raw observations independently of incident normalization. In
                # particular, a zero incident source still has a finite zero raw DFT.
                raw_s=xp.where(odd, -Es, Es)*xp.exp(1j*(ku0*grid.size[u]/2+kv0*grid.size[v]/2)),
                raw_p=xp.where(odd, -Ep, Ep)*xp.exp(1j*(ku0*grid.size[u]/2+kv0*grid.size[v]/2)),
                raw_flux=Ss+Sp, cell_area=grid.size[u]*grid.size[v], epsilon=eps, mu=mu,
                k_parallel=xp.stack((xp.ones_like(kz)*ku[None,None,:],
                                     xp.ones_like(kz)*kv[None,:,None]),axis=-1),
                tangent_axes=('xyz'[u],'xyz'[v]),
                outward_normal=tuple(out_dir if a==P else 0 for a in range(3)))


# ---------------------------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------------------------

def _np(rt, x):
    """``x`` as NumPy, or ``None`` when it is a tracer (a run under ``jax.jit``)."""
    try:
        return np.asarray(rt.numpy(x))
    except Exception:
        return None


def sources_end(scene, sources=None, rt=None):
    """The time after which no source drives the run: 0 with none, ``None`` if unknown.

    A source's waveform (:class:`wavebit.fdtd._legacy.Band`) is ``exp(-s^2 / (2 tau^2))`` about
    ``t0 + delay``, with ``tau = 1/(2 pi df)`` and ``t0 = k tau`` where ``exp(-k^2/2)`` is the
    band's ``cutoff``; it starts at ``cutoff`` of its peak at ``t = delay`` and, the envelope
    being symmetric, is back at ``cutoff`` at ``delay + 2 t0`` and below it after. A ``Dipole``
    stops driving there. With no source -- an ``InitialField`` kick -- the whole excitation is
    the state at ``t = 0``, so the end is 0.

    **A ``PlaneWave`` sheet and a ``TFSF`` box drive as long as their column's field on their
    faces changes**, which is not when their waveform ends: they add and remove the column's
    field wherever it is, and the column holds more than the pulse. A layered TFSF column holds
    every echo of its layers: with glass 3.5 below a vacuum box, the reflection (R = 0.04) comes
    back up the column at t = 9 .. 13 and the box re-injected it into an interior the pulse had
    left, which read 4.04 and warned (review, 2026-09-20; the closed-form end this replaced,
    ``delay + 2 t0 + L n_max``, was 6.7). A PlaneWave's column holds what its ends return: an
    ``Absorber`` end returns 5% of the amplitude, which the sheet injects again. So a column
    source's end is read off its own series (``SourceSeries.incident`` and ``incident_H``, the
    tangential E on the entry plane and the H half a cell inside it -- exactly what a PlaneWave's
    sheet and a TFSF's entry face inject): the last time the change of either over one step,
    ``|x(t + dt) - x(t)|^2`` summed over components, is above :data:`COLUMN_LEVEL` of its own
    peak, or ``delay + 2 t0`` if that is later. ``None`` when the series has no value (a trace).

    *Why the entry plane sees every echo.* A wave that re-enters the box span comes in through
    the entry plane (a layer above the box sends it down across it) or through the exit plane (a
    layer below sends it up), and one that came in through the exit plane crosses the box and
    leaves through the entry plane: its energy rises in the box while it crosses and is seen at
    the entry plane when it leaves, so the last time the entry plane sees it is after the last
    rise it caused. A layer inside the box lets through what it does not reflect, and the entry
    plane reads that (``tests/fdtd/test_fd_decay.py``: a glass layer inside the box and a substrate
    below it). What the plane under-reads is a wave held between a near-total reflector inside
    the box and a layer below it, by the reflector's transmission. *Why the change and not the
    level.* A column's static residue is injected at constant energy -- it feeds nothing -- and
    it never ends: an oblique PlaneWave's column holds H at 1e-7 .. 1.3e-7 of its peak for good
    (``fd_helpers.bloch_plane_wave`` at 30 degrees, ``bloch_stack``), so a level would never
    let those scenes be judged. The change of a wave at frequency ``f`` is ``2 sin(pi f dt)``
    times its amplitude, so an echo at the carrier is read at its energy, and one inside the
    band (``f0 +- 2 df``, where nearly all of an echo is) at ``((f0 - 2 df)/f0)^2`` or more of
    it: 0.36 at ``df = 0.2 f0``, 0.25 at ``df = 0.25 f0``.

    **A column that never goes quiet** (``_column_state``). The change after the envelope's end
    is judged with the interior's statistic (``_post.growth_ratio``, floor
    :data:`COLUMN_LEVEL`), over quarters of at least one period and one round trip of the
    column (``_round_trip``: ``2 L / v`` at the slowest axial group speed ``v`` of the scene's
    materials), since an echo arrives within a round trip and a quarter that holds one sees it
    come and go. Three outcomes:

        growing  the change rises as an exponential does (over GROWTH_THRESHOLD, spread under
                 GROWTH_STEADY): the column holds the scene's instability, and the scene is
                 growing -- ``decay_report`` warns naming the source's column
                 (:func:`column_growth`), and the interior is judged from the envelope's end
        holding  the four eighths of the last half within GROWTH_THRESHOLD of each other: a
                 steady oscillation, injected at constant energy, which can lift a passive
                 interior only toward a level and never as an exponential -- the interior is
                 judged from the envelope's end
        neither  falling, or an echo passing: the column end above stands

    Why this exists (review, 2026-09-20): the growing slab column of
    ``tests/fdtd/test_fd_pml_stability.py`` driven by an oblique PlaneWave. Its column is the
    uniform background under the same pure-stretch faces, and the stretch holds an evanescent
    field at the entry plane forever -- amplitude 1.5e-4, a one-step change at 1.8e-8 of its
    peak, just over the level, flat to 1% from t = 20 to 400 -- so the column end was the run's
    end, the interior (growing 14000-fold by t = 400) was never judged, and the run said only
    "run longer". Judged from the envelope's end it reads 1.89 at t = 100 and 4.57 at 200
    (spread 1.02, 1.05) and warns from t = 31 on; its stabilized twin's column falls silent at
    34.7. The round trip in the quarter is what keeps an echo from reading as a growing
    column: without it, the review's TFSF echo (glass 3.5 below the box) arriving at t = 10
    read 379 with spread 1.6 over quarters of a period, a steeper rise than any growth.
    Measured on every column the fast tier judges (724 series) and on the review's scenes at
    every run length: holding only in the growing PlaneWave scene, growing in none (no scene
    built so far grows in its own column -- a PlaneWave's column is uniform, and a TFSF's is at
    normal incidence, where a layered background has no bound mode), and neither in every
    TFSF, Absorber-ended and oblique column (``tests/fdtd/test_fd_decay.py``).
    """
    ends = _source_ends(scene, sources, rt)
    return None if ends is None else max((end for end, _ in ends.values()), default=0.0)


def column_growth(scene, sources=None, rt=None):
    """``{name: (ratio, lo, mid, hi, spread)}`` for every ``PlaneWave`` and ``TFSF`` whose own
    column is growing (:func:`sources_end`), or ``None`` when an end is unknown."""
    ends = _source_ends(scene, sources, rt)
    return None if ends is None else {name: judged for name, (_, judged) in ends.items()
                                      if judged is not None}


def _source_ends(scene, sources, rt):
    """``{name: (end, growing column's judgement or None)}``, or ``None`` if any end is
    unknown (:func:`sources_end`)."""
    from wavebit.fdtd._legacy_sources import concrete_or_none
    out = {}
    for src in scene.sources:
        df = concrete_or_none(src.band.df)
        delay = concrete_or_none(src.delay)
        if df is None or delay is None:
            return None
        stop = delay + 2.0 * src.band.sigmas / (2.0 * math.pi * df)
        judged = None
        if type(src).__name__ in ('PlaneWave', 'TFSF'):
            series = None if sources is None else sources.get(src.name)
            column = None if series is None else _column_end(series, rt)
            if column is None:
                return None
            end, change = column
            kind, judged = _column_state(change, stop, concrete_or_none(src.band.wavelength),
                                         _round_trip(scene, src, series))
            if kind is None:
                stop = max(stop, end)
            if kind != 'growing':
                judged = None
        out[src.name] = (stop, judged)
    return out


#: The level a column source's series must change by, over one step, for the source to be
#: still driving (:func:`sources_end`): of the largest change in the run. An echo of amplitude
#: ``a`` re-injects about ``a^2`` of the peak's energy; the growth ratio reads a last quarter at
#: :data:`GROWTH_FLOOR` of the peak or more, so an echo under ``(GROWTH_THRESHOLD - 1)
#: GROWTH_FLOOR`` of the peak lifts it by less than the line's margin even at the floor. Derived,
#: 1e-8. Measured (2026-09-20), against the envelope's end (7.89 at the helpers' band): the
#: review's echo scene (glass 3.5 below a vacuum TFSF box, ``tests/fdtd/test_fd_decay.py``) ends at
#: 15.6, after its echo at 9 .. 13; ``fd_helpers.tfsf_box`` in vacuum at 8.90 and over glass at
#: 10.50; a lamellar grating's PlaneWave at 9.97; oblique PlaneWaves (``bloch_plane_wave`` at 30
#: degrees, ``bloch_stack`` at 25), whose columns keep a static H at 1e-7 of its peak, at 11.02
#: and 10.53; a PlaneWave column ended by ``Absorber(12)`` at 17.7 against its own band's 4.73
#: (its ends return 5%).
COLUMN_LEVEL = (GROWTH_THRESHOLD - 1.0) * GROWTH_FLOOR


def _column_end(series, rt):
    """``(end, (t, change))``: the last time a column source's series changes by more than
    :data:`COLUMN_LEVEL` of its largest change over one step (:func:`sources_end`), and that
    change -- E's and H's, each over its own largest, summed -- at the E times; ``None``
    without values."""
    end, total = 0.0, None
    for values, t in ((series.incident, series.t), (series.incident_H, series.t_H)):
        v = None if values is None else _np(rt, values)
        if v is None:
            return None
        change = np.sum(np.abs(np.diff(v, axis=0)) ** 2, axis=1)
        if not change.size or not np.all(np.isfinite(change)):
            return None
        top = float(np.max(change))
        if top > 0.0:
            last = int(np.nonzero(change > COLUMN_LEVEL * top)[0][-1])
            end = max(end, float(np.asarray(t)[last + 1]))
            change = change / top
        total = change if total is None else total[:change.size] + change[:total.size]
    return end, (np.asarray(series.t, dtype=np.float64)[1:1 + total.size], total)


def _column_state(change, start, wavelength, round_trip):
    """``(kind, judgement)`` of a column's one-step change after its pulse's envelope ended at
    ``start`` (:func:`sources_end`): ``'growing'`` (it rises as an exponential does, by the
    interior's rule), ``'holding'`` (the four eighths of the last half within
    :data:`GROWTH_THRESHOLD` of each other) or ``None`` (falling, an echo passing, or not
    judged). Judged with quarters of at least one period and one round trip of the column."""
    if wavelength is None or round_trip is None:
        return None, None
    judged = _post.growth_ratio(*change, start, max(wavelength, round_trip), GROWTH_WINDOWS,
                                COLUMN_LEVEL)
    if judged is None:
        return None, None
    ratio, lo, _, hi, spread = judged
    if ratio > GROWTH_THRESHOLD and spread <= GROWTH_STEADY:
        return 'growing', judged
    t, e = change
    edges = np.linspace(lo, hi, 5)
    means = [float(np.mean(e[(t >= a) & (t <= b)])) for a, b in zip(edges[:-1], edges[1:])]
    if min(means) > 0.0 and max(means) <= GROWTH_THRESHOLD * min(means):
        return 'holding', judged
    return None, judged


def _round_trip(scene, src, series):
    """The time a wave takes down the source's column and back, at its slowest: ``2 L / v``
    for the column's length ``L`` and the slowest axial group speed ``v = sqrt(n^2 - s^2) /
    n^2`` over the scene's materials that carry the wave (``n > s``, ``s = n_background sin
    theta``); ``None`` when a material is traced."""
    from wavebit.fdtd._prepare.checks import _index_at
    from wavebit.materials._time_domain import as_material
    from wavebit.fdtd._legacy_sources import concrete_or_none
    wavelength = concrete_or_none(src.band.wavelength)
    if wavelength is None:
        return None
    background = as_material(getattr(scene, 'background', None), 'Scene(background=)')
    materials = [background] + [st.material for st in (getattr(scene, 'structures', ()) or ())]
    n = [_index_at(m, wavelength)[0] for m in materials]
    if any(v is None for v in n):
        return None
    s = n[0] * math.sin(math.radians(float(getattr(src, 'theta', 0.0) or 0.0)))
    speeds = [math.sqrt(v * v - s * s) / (v * v) for v in n if v > s]
    term = getattr(getattr(series, 'context', None), 'term', None)
    if term is not None:
        sub = term.sub
        length = (sub.nx, sub.ny, sub.nz)[term.axis] * sub.dx
    else:
        length = float(scene.domain.size[src.axis])
    return 2.0 * length / min(speeds) if speeds else 2.0 * length


def _scene_period(scene, plan):
    """The growth judgement's time unit: the longest wavelength the scene names, or ``2 L``.

    The longest of the frequency-domain monitors' wavelengths, the sources' band centres and
    the PML faces' wavelengths -- the slowest oscillation the scene declares, so that a
    quarter of the judged span holds whole periods of it. A scene that names none (a closed
    box kicked by an ``InitialField``) takes ``2 L`` for its longest side ``L``: the period of
    the slowest standing wave a vacuum box of that side holds along one axis.
    """
    from wavebit.fdtd._legacy_sources import concrete_or_none
    lengths = [max(m.wavelengths) for m in scene.monitors if not isinstance(m, Record)]
    lengths += [w for w in (concrete_or_none(s.band.wavelength) for s in scene.sources)
                if w is not None]
    lengths += [w for _, w in plan.pml_wavelengths if w is not None]
    return max(lengths) if lengths else 2.0 * max(plan.grid.size)


def interior_energy(plan, state, rt):
    """``(t, W)`` of the interior's energy as NumPy, or ``None`` under a trace
    (``src/wavebit/fdtd/_kernel/observe.py::InteriorEnergy``)."""
    acc = _np(rt, state['energy'])
    return None if acc is None else plan.energy.series(plan, acc)


def _knob(scene, faces):
    """``PML(stabilize=...)`` with the measured value for the weakest target among ``faces``."""
    targets = [scene.boundaries.face(name).reflection for name in faces]
    rho = [sufficient_stabilize(r) for r in targets]
    if any(v is None for v in rho):
        return (f'PML(stabilizing_loss=) (no value was measured at a reflection target '
                f'weaker than {spell(MEASURED_STABILIZE[-1][0])}; '
                f'{spell(MEASURED_STABILIZE[-1][1])} sufficed there)')
    return f'PML(stabilizing_loss={spell(max(rho))})'


def _growth_cause(scene, plan):
    """What can feed a growing field in ``scene``, and the knob that stops it."""
    cause = _pml_cause(scene, plan)
    if cause is not None and getattr(scene, 'initial', None) is not None:
        cause += (' This scene starts from an InitialField, and one passive case reads the same: '
                  'its content near zero frequency, which a PML does not absorb (its CFS term '
                  'makes it transparent there), returns off the wall behind the layer as a slow, '
                  'steady rise that stabilizing_loss does not change '
                  '(src/wavebit/fdtd/_assemble.py::GROWTH_THRESHOLD). A kick with no '
                  'zero-frequency content is not affected.')
    return cause if cause is not None else _no_pml_cause()


def _pml_cause(scene, plan):
    """The PML faces' part of :func:`_growth_cause`, or ``None`` with no PML face."""
    faces = [(face.name, face.stabilize) for face in plan.cpml]
    pure = [name for name, rho in faces if not rho]
    if pure:
        return (
            f'The PML on {pure} is the pure stretch (the default, stabilizing_loss=None), and '
            f'a stretch backed by its PEC wall feeds an evanescent field that reaches it: a '
            f'lossless structure with a bound mode just past the light line (a grating\'s '
            f'guided resonance at a Rayleigh wavelength, a slab under a fixed transverse k) '
            f'grows under it (src/wavebit/fdtd/_prepare/cpml.py::stabilizing_rate). '
            f'{_knob(scene, pure)} on those faces adds the loss that stopped it on every slab '
            f'family it was measured on, at the price of a reflection at oblique incidence '
            f'(2.1e-4 at 30 degrees on 12 cells at 3e-3).')
    if faces:
        parts = []
        for name, rho in faces:
            target = scene.boundaries.face(name).reflection
            enough = sufficient_stabilize(target)
            measured = ('no value was measured at a target that weak' if enough is None
                        else f'{spell(enough)} was measured to suffice there')
            parts.append(f'{name!r} stabilizing_loss={spell(rho)} at '
                         f'reflection={spell(target)} ({measured})')
        return (
            f'Every PML face is stabilized -- {"; ".join(parts)} -- and the loss is a measured '
            f'balance on sampled slab families, not passivity '
            f'(src/wavebit/fdtd/_prepare/cpml.py::stabilizing_rate): this structure lies outside the '
            f'families it was measured on. Raise stabilizing_loss on those faces (it costs '
            f'reflection at oblique incidence in proportion).')
    return None


def _no_pml_cause():
    return ('No face of this scene is a PML, and nothing else in a passive scene can feed a '
            'field once its sources have ended (every material is passive: gain is refused '
            'where a material is declared): the energy this reads is the leapfrog\'s own, '
            'exactly conserved in a closed lossless box whatever the field is made of '
            '(src/wavebit/fdtd/_kernel/observe.py::InteriorEnergy). So this is a defect of the '
            'solver -- please report it with the scene.')


def decay_report(scene, plan, state, rt, sources, stable=True):
    """``{name: ratio}`` for every frequency-domain monitor, the growth, and the warnings.

    **Decay.** The ratio is how much the region energy ``sum |F|^2`` over the monitor's region
    changes in the run's **last window** -- one period of the longest wavelength it asks for --
    divided by its largest value anywhere in the run (``_post.decay_ratio``: the change and not
    the level, because a static residue is exact after ``_post.end_correction``). The measured
    relation to the truncation error of a flux is at :data:`UNSETTLED_THRESHOLD`, where the
    ``UnsettledFieldWarning`` starts ("run longer"). A region that never saw a field (a
    scattered-field box around an empty TFSF, peak energy below 1e-20 of the loudest monitor's)
    reports 0. Records and sources are reported too (``records``, ``sources``: the last twentieth
    of the run against the peak), for reading, and never warn unsettled: a time series is not a
    spectral claim.

    **Growth** is judged on the scene, not on a monitor: ``interior_growth`` is
    ``_post.growth_ratio`` of the energy of everything but the PML layers
    (``src/wavebit/fdtd/_kernel/observe.py::InteriorEnergy``) after every source has ended
    (:func:`sources_end`, ``sources_end`` in the result) -- its mean over the last quarter of
    that span over its mean over the quarter before -- and ``interior_spread`` how steadily the
    last half rose (``inf`` unless it rose throughout). A passive scene's interior energy cannot
    rise once nothing drives it, except by what a PML hands back, which arrives as a pulse and
    not as an exponential; so over :data:`GROWTH_THRESHOLD` with a spread under
    :data:`GROWTH_STEADY` the field is being fed -- in this package, by a PEC-backed PML
    stretch (``src/wavebit/fdtd/_prepare/cpml.py::stabilizing_rate``). ``None`` where it cannot judge:
    a span of fewer than :data:`GROWTH_WINDOWS` periods (``_scene_period``), a last quarter
    under :data:`GROWTH_FLOOR` of the peak, a source end that is not known (a traced band, or a
    PlaneWave or TFSF whose column series has no value). **A column source's own field is
    judged too** (:func:`column_growth`, ``column_growth`` in the result): a ``PlaneWave``'s or
    ``TFSF``'s column that rises as an exponential does after its pulse has ended is a growing
    scene by itself, whatever the interior reads, because what the source injects is that
    column; and one that neither rises nor falls -- a steady oscillation the stretch holds --
    injects at constant energy, which cannot lift a passive interior steadily, so the interior
    is judged from the envelope's end instead of waiting for a column that never goes quiet
    (:func:`sources_end`). ``growth`` says, per monitor (Records
    included), how its own series moved over the same two quarters (``_post.quarter_ratio``;
    ``None`` when the interior was not judged, for a quiet region, or for a traced series). A
    single series cannot judge growth -- energy moves through a region, and an echo or a slow
    beat lifts it by any factor in a passive scene -- but once the interior is growing it says
    which monitors see it. **An interior energy that is not finite** is the largest growth there
    is: ``interior_growth`` is ``inf`` and one ``GrowingFieldWarning`` says the field
    overflowed, and from when, in place of every other warning -- unless the run's stability
    guard voided it on purpose (``stable`` False, every array ``nan``).

    **The warnings.** In a growing scene -- the interior over the line, or a column growing,
    or both -- every monitor that sees it (its own peak rose over the two quarters, or it is
    unsettled) warns ``GrowingFieldWarning`` **instead of** its ``UnsettledFieldWarning``: a
    longer run does not settle a growing field, so no monitor is told to run longer; the
    warning says which of the two was seen (with its numbers, and the source's name for a
    column) and names what can feed it and the knob (``PML(stabilize=)`` with the value
    :func:`sufficient_stabilize` measured for the face's target). When no monitor
    sees it, one ``GrowingFieldWarning`` names the interior, so a growth is never silent. A
    ``Diffraction(polarizations='both')`` scene is two solves, and each reports its own.

    Under ``jax.jit``, ``jax.vmap`` or any other transform that traces the run every series is
    a tracer and every ratio and growth is ``None``, and nothing warns -- the diagnostic reads
    values, and a trace has none yet. Under a bare ``jax.grad`` (no jit) the energy series (the
    interior's and the monitors' regions') are not on the differentiated path and come back
    concrete, so the decay and the growth are judged and warn exactly as on NumPy; a Record's
    own values are on that path, so its entry in ``growth`` is ``None`` there, and a growing
    scene whose only monitor is a Record warns for the interior (``tests/fdtd/test_fd_decay.py`` pins
    these). A PlaneWave's or a TFSF's column series is on that path too, so such a scene has no
    known end under ``jax.grad`` and its growth is not judged.
    """
    obs = state['obs']
    dt = plan.dt
    tail = max(1, plan.steps // 20)
    monitors, records, srcs = {}, {}, {}
    series = {}                                 # name: (t, energy), for "does it see it"
    for m in scene.monitors:
        if isinstance(m, Record):
            v = _np(rt, obs[m.name])
            records[m.name] = None if v is None else _post.decay_ratio(np.abs(v) ** 2, tail)
            series[m.name] = None if v is None else (
                (np.arange(plan.steps) + 1.0) * dt, np.abs(v).astype(np.float64) ** 2)
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
        series[m.name] = None if energy is None else ((np.arange(plan.steps) + 1.0) * dt,
                                                      energy)
    loudest = max((float(np.max(e)) for w, e in (v for v in monitors.values() if v is not None)
                   if e.size), default=0.0)
    quiet = set()
    for name, entry in monitors.items():
        if entry is None:
            continue
        window, energy = entry
        if not float(np.max(energy)) > 1e-20 * loudest if energy.size else True:
            quiet.add(name)
        monitors[name] = 0.0 if name in quiet else _post.decay_ratio(energy, window)
    for name, series_ in (sources or {}).items():
        values = series_.incident if series_.incident is not None else series_.field
        v = None if values is None else _np(rt, values)
        if v is None:
            continue
        v = np.abs(v)
        e = v ** 2 if v.ndim == 1 else np.sum(v ** 2, axis=1)
        srcs[name] = _post.decay_ratio(e, tail)

    end = sources_end(scene, sources, rt)
    columns = column_growth(scene, sources, rt) or {}
    interior = interior_energy(plan, state, rt)
    judged = None
    overflow = None
    if interior is not None and stable is not False:
        bad = ~np.isfinite(interior[1])
        if np.any(bad):
            overflow = float(interior[0][np.argmax(bad)])
    if interior is not None and end is not None and overflow is None:
        judged = _post.growth_ratio(*interior, end, _scene_period(scene, plan),
                                    GROWTH_WINDOWS, GROWTH_FLOOR)
    interior_growth = math.inf if overflow is not None else None if judged is None else judged[0]
    spread = None if judged is None else judged[4]
    inside = (judged is not None and interior_growth > GROWTH_THRESHOLD
              and spread <= GROWTH_STEADY)
    growing = inside or bool(columns)
    # The two quarters a monitor's own rise is read over: the interior's, or a growing
    # column's when the interior is not judged growing.
    quarters = (judged[1:4] if inside or (judged is not None and not columns)
                else next(iter(columns.values()))[1:4] if columns else None)
    growth = {}
    for name, entry in series.items():
        growth[name] = (None if quarters is None or entry is None or name in quiet
                        else _post.quarter_ratio(*entry, *quarters))
    seen = []
    if overflow is not None:
        # Every monitor's series is garbage past the overflow: one warning, and no other.
        seen = list(series)
        warnings.warn(spoken(
            f"the field OVERFLOWED -- the energy in the scene's interior was no longer finite "
            f'from t = {overflow:.4g} on. It grew past the range of the floating-point type, '
            f"and a passive scene's energy cannot grow at all once nothing drives it. "
            + _growth_cause(scene, plan)), GrowingFieldWarning, stacklevel=4)
    elif growing:
        seen = [name for name in series
                if (growth[name] is not None and growth[name] > 1.0)
                or (monitors.get(name) is not None and monitors[name] > UNSETTLED_THRESHOLD)]
        heads = []
        if inside:
            lo, mid, hi = judged[1:4]
            heads.append(
                f"the energy in the scene's interior (every cell outside the PML layers) "
                f'averaged {interior_growth:.3g} times higher over t = {mid:.4g} .. {hi:.4g} than '
                f'over t = {lo:.4g} .. {mid:.4g}, after every source had ended, rising steadily '
                f"as an exponential does (the line is {GROWTH_THRESHOLD:g}; a passive scene's "
                f'energy can only hold or fall once nothing drives it)')
        for name, (ratio, lo, mid, hi, _) in columns.items():
            heads.append(
                f"the field in source {name!r}'s own column (the incident field it injects, "
                f"stepped with the scene's layers and PML faces) kept changing more after its "
                f'pulse had ended: its one-step change averaged {ratio:.3g} times higher over '
                f't = {mid:.4g} .. {hi:.4g} than over t = {lo:.4g} .. {mid:.4g}, rising steadily '
                f'as an exponential does, so the source never stops driving the scene')
        head = '; and '.join(heads)
        cause = (' A longer run does not settle it, it makes it worse: the DFT of a growing '
                 'signal has no limit to converge to. ' + _growth_cause(scene, plan))
        for name in seen:
            own = growth[name]
            sees = ('' if own is None else
                    f' -- and this monitor sees it: its own peak rose {own:.3g} times')
            warnings.warn(spoken(f'monitor {name!r}: the field is GROWING -- {head}{sees}.' + cause),
                          GrowingFieldWarning, stacklevel=4)
        if not seen:
            warnings.warn(spoken(f'the field is GROWING, though no monitor sees it yet -- {head}.'
                                 + cause), GrowingFieldWarning, stacklevel=4)
    for name, ratio in monitors.items():
        if name in seen:
            continue
        if ratio is not None and ratio > UNSETTLED_THRESHOLD:
            warnings.warn(spoken(
                f'monitor {name!r}: the field in its region was still {ratio:.2e} of its peak '
                f'energy in the last period of its longest wavelength (the line is '
                f'{UNSETTLED_THRESHOLD:.0e}). Its DFT is of a signal cut off before it '
                f'decayed, so the phasors -- and every flux, order and normalisation built on '
                f'them -- carry that truncation. Run longer (Run(steps=), Run(periods=), '
                f'Run(time=)).'),
                UnsettledFieldWarning, stacklevel=4)
    return dict(decay=monitors, records=records, sources=srcs, growth=growth,
                interior_growth=interior_growth, interior_spread=spread,
                interior_energy=interior, sources_end=end,
                column_growth={name: v[0] for name, v in columns.items()},
                threshold=UNSETTLED_THRESHOLD,
                growth_threshold=GROWTH_THRESHOLD)
