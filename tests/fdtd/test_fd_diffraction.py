"""Stage 3, gate 3 -- diffraction orders against ``wavebit.rcwa``, the engine's first direct oracle.

Every scene here is a ``fd.PlaneWave`` in a Bloch (or periodic) cell with ``fd.Diffraction``
monitors, and every number is compared with ``wavebit.rcwa`` solving the **identical** geometry
at the run's own angle: a PlaneWave has a fixed transverse ``k``, so the angle moves across the
band and each wavelength is compared at ``res['r'].theta_at[i]`` (degrees; RCWA takes radians).
The amplitudes are compared as complex numbers -- amplitude and phase at once -- in the frame
``wavebit.observations.diffraction`` defines (each order's own ``s_hat = z_hat x k_hat_par``, the p sign
for which ``r_p = -r_s`` at normal incidence, ``r`` referred to the top interface, ``t`` to the
bottom one, the lateral phase origin at the cell centre). The scenes:

* **(a) a uniform stack**: a 0.3 slab of ``eps = 2.25`` on an ``eps = 1.44^2`` substrate, theta
  25 degrees, s and p (``fd_helpers.bloch_stack`` in a two-cell transverse cell);
* **(b) a lamellar grating**: a cell-snapped ``fd.Region`` tooth (``eps = 2.25``, 0.8 wide, 0.2
  high) that spans y, period 1.6 at normal incidence (s and p) and period 1.2 at 20 degrees
  (s, wavelengths 1.0 and 1.1 -- at 0.9 the ``(+1, 0)`` order sits on the slab's guided
  resonance, where a half-cell of staircase moves ``R_0`` from 0.30 to 0.13);
* **(c) a crossed grating**: a square pillar 0.6 on a side in a 1.2 square cell, normal
  incidence, s, wavelengths 0.95 and 1.05 (clear of the ``(1, 1)`` Rayleigh anomaly at 0.849
  and of the grazing rule: every propagating order leaves at ``cos theta >= 0.48``).

**RCWA is converged first.** Its truncation is part of the comparison, so each scene shows it:
the uniform stack is exact at truncation 0 (a higher one changes nothing, to 1e-12); the lamellar
grating at truncation 20 is within 1.7e-5 (s) / 4.4e-5 (p) / 4e-6 (20 degrees) of truncation 30;
the crossed grating at 4 is within 8.1e-4 of 10 (the single-resolution test, 33x below its
2.7e-2) and at 6 within 4.1e-4 (the sweep, 32x below its finest 1.3e-2) -- in every case far
below the FDTD error it is used to judge
(:func:`test_rcwa_truncation_is_converged_below_the_fdtd_error` and the slow sweeps).

**What was measured (2026-09-19, NumPy, 160 periods for the gratings)** -- the worst ``|fd -
rcwa|`` over every propagating order's complex amplitudes (co- and cross-polarised) at every
wavelength, reflect and transmit sides, and the fitted order ``p`` of ``error ~ dx^p``:

    scene                              cells per wavelength   worst |d amplitude|        p
    uniform stack, s                   20 / 40 / 80           2.6e-2 / 6.5e-3 / 1.6e-3   2.0
    uniform stack, p                   20 / 40 / 80           3.3e-2 / 9.8e-3 / 3.3e-3   1.7
      (p: magnitude 5.4e-2 / 1.3e-2 / 3.2e-3, order 2.0; phase 7.8e-2 / 4.0e-2 / 2.0e-2 rad, 1.0)
    lamellar, normal, s                20 / 40 / 80           2.9e-2 / 1.2e-2 / 5.5e-3   1.2
    lamellar, normal, p                20 / 40 / 80           1.2e-2 / 3.1e-3 / 1.5e-3   1.5
    lamellar, 20 degrees, s            20 / 40 / 80           2.1e-2 / 1.1e-2 / 5.7e-3   0.95
    crossed, normal, s                 20 / 30 / 40           2.7e-2 / 1.8e-2 / 1.3e-2   1.05

The s stack converges second order (both the amplitudes and ``R``, ``T``); everything that has
an interface a component sees half a cell away from where another component sees it converges
first order -- the p stack's normal E sees the snapped z interface on the cell boundary and its
tangential E half a cell lower (``tests/fdtd/test_fd_airy.py`` derives the half cell), and every
``Region`` edge across x or y does the same to ``E_y`` or ``E_x``: the grid's cell-snapped tooth is
the RCWA tooth moved by ``-dx/2`` for the component parallel to that edge, which shows as the
``+-1`` orders' phases leaning opposite ways (``r_s(+1) != -r_s(-1)`` by ``~ 2 k_G dx/2``) while
their efficiencies stay equal. That is the staircase stage 4's subpixel averaging removes; the
gate therefore states the error at each resolution and fits its order rather than holding one
number to a second-order tolerance. The resolution sweeps of the gratings are ``slow`` (3 min
for the crossed one, 2 min each for the others, at ``-n 4``).

**A defect this gate exposed, fixed 2026-09-19**
(:func:`test_a_conical_plane_wave_is_launched_without_the_other_polarisation`, an
``xfail(strict=True)`` until then): off the planes ``phi = 0`` and ``90`` the PlaneWave column
launched a p part with an s wave (and the reverse). Its hard source wrote the continuum
polarisation into the stored ``E_x`` and ``E_y``, each of which sits half a cell along its own
axis, so the launched wave was elliptical by ``~ (e . k_hat_par)(k_x - k_y) dx/2``: a uniform
stack at theta 25, phi 60 transmitted ``|t_p| = 9.5e-3 / 4.7e-3 / 2.4e-3`` for an s wave at 20 /
40 / 80 cells per wavelength (and ``|t_s| = 7.8e-3 / 3.9e-3 / 2.0e-3`` for a p wave), first
order. The drive now writes the grid's own s or p wave -- stagger-removed amplitudes transverse
to the discrete ``K_par``, times each component's stagger phase ``exp(i k_c dx/2)``
(``src/wavebit/fdtd/_prepare/sources.py::plane_wave_drive`` derives it) -- which is the frame
``_post.frame`` reads the orders in, and the cross-polarised amplitude is round-off: 1.2e-16 to
6.4e-16 at 20 / 40 / 80, where RCWA gives 1e-16. (Build B's probe, with the stagger phase and
the continuum direction, left 3e-5: the direction is second order.) At phi 60 the worst
``|fd - rcwa|`` over co- and cross-polarised amplitudes is unchanged at 20 and 40 cells per
wavelength (s 2.65e-2 / 6.45e-3, p 3.33e-2 / 9.78e-3: the co-polarised error dominates there),
and at 80 the s stack's fell from 2.4e-3 to 1.6e-3, the leak having been its largest part.
Every gate-3 scene above is at ``phi = 0``, where the leak vanished (an s wave's E is along y,
which carries no k; a p wave's along x carries one common phase, which the normalisation by the
column divides out), so the table above did not move.

**Mutants** (each re-assembles the *same* finished run under a module switch -- the post-
processing is all that differs, so the difference is the mutant's alone):

* ``_post.DIFFRACTION_KZ = 'continuum'`` (``kz = sqrt(n^2 w^2 - k_par^2)`` instead of the Yee
  ``kappa``): the de-embedding is no longer exact, so the same order read on two planes at
  different heights disagrees -- by ``(kz_c - kappa) * dz``, which the test predicts from the
  discrete dispersion relation and finds to 1e-9 rad. The Yee reading agrees across the planes
  to the run's unsettled level (2e-5). **Measured, and against the plan's premise:** the
  mismatch does *not* grow monotonically with the order's angle on this grid. To second order
  ``kappa^2 - kz_c^2 ~ k^4 dx^2 (cos^4 t + sin^4 t - (dt/dx)^2)/12``, so ``kz_c - kappa``
  scales as ``(cos^4 t + sin^4 t - (dt/dx)^2)/cos t``: it *falls* from 0 to about 45 degrees (the
  Yee grid is most isotropic along the diagonal) and only then grows, without bound toward
  grazing. The ``+-1`` orders here, at 34-43 degrees, carry 0.61-0.64 of the zeroth order's
  error -- which the closed form predicts exactly.
* The s/p frame of the ``-m`` orders reversed (``s_hat, k_hat_par -> -s_hat, -k_hat_par`` on the
  ``Re K_u < 0`` half plane -- ``wavebit.rcwa``'s own 2026-08-16 bug): ``R`` does not move to
  the bit (``|r|^2`` cannot see a reversed frame), and the ``-1`` order's ``r_s`` misses RCWA by
  twice its size -- the worst ``-1`` amplitude (``t_s``) by 0.43, ten times the gate's 4.5e-2.
* The incident amplitude taken from the **analytic** pulse spectrum (the Fourier transform of
  the band's waveform, carried with the continuum ``kz``) instead of the PlaneWave's column: it
  is off by a phase first order in ``dx`` (0.28 / 0.14 / 0.07 rad at 20 / 40 / 80 cells per
  wavelength -- the sheet's own half cell and half step) and, for p, by ``cos(theta_0)/cos
  (theta)`` in amplitude (the column drives the tangential E, fixed k moves the angle), so the
  stack's amplitudes miss RCWA by 0.14 (s) / 0.12 (p) at 40 cells per wavelength, 8-14x the
  gate's tolerance, while ``R`` and ``T`` (normalised by the column's flux) do not move.
"""
import functools
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.observations.diffraction import Result
from wavebit.fdtd import _assemble, _post
from wavebit.fdtd._prepare import lower
from wavebit.fdtd._execution import _results, execute
from wavebit.fdtd._runtime import runtime

from tests.fixtures.fdtd.fd_helpers import bloch_stack, box_grating, order_value, to_numpy

pytestmark = [pytest.mark.filterwarnings('ignore::wavebit.fdtd._legacy.CoarseMeshWarning'),
              pytest.mark.filterwarnings('ignore::wavebit.fdtd._legacy.AbsorberProximityWarning'),
              pytest.mark.filterwarnings('ignore::wavebit.fdtd._legacy.UnsettledFieldWarning')]

_QUIET = (fd.CoarseMeshWarning, fd.AbsorberProximityWarning, fd.UnsettledFieldWarning)

#: The uniform stack: a 0.3 slab of eps 2.25 on a substrate of index 1.44, at 25 degrees.
STACK = dict(theta=25.0, layers=((2.25, 0.3),), substrate=1.44 ** 2)
SUBSTRATE_INDEX = 1.44

#: The lamellar grating at normal incidence: fd_helpers.box_grating's default cell (period
#: 1.6, tooth 0.8 x 0.2), with the monitors 1.5 away from the sheet and the tooth so that a
#: second pair of planes near them can test the de-embedding (``gap``).
LAMELLAR = dict(kind='lamellar', gap=1.5)
#: The lamellar grating at 20 degrees: period 1.2 (tooth 0.6) keeps every propagating order
#: at exit cos theta >= 0.82 over these wavelengths (with 1.6 the (+1, 0) order grazes at 0.25).
OBLIQUE = dict(kind='lamellar', theta=20.0, period=1.2, wavelengths=(1.0, 1.1))
#: The crossed grating: a 0.6 square pillar in a 1.2 square cell, monitors 0.3 off.
CROSSED = dict(kind='crossed', period=1.2, wavelengths=(0.95, 1.05), sheet=0.3, gap=0.3)

#: RCWA truncations the comparisons use (converged: see the truncation test). The crossed
#: grating's sweep, whose finest error is 1.3e-2, uses 6 (a 2-D solve at 10 is ~20 s).
TRUNCATION = {'lamellar': 20, 'crossed': 4}
CROSSED_SWEEP_TRUNCATION = 6

#: The decay ratio each scene's monitors must be under (``res.diagnostics['decay']``): every
#: comparison is of a settled run, or of one whose unsettled part is shown not to be what the
#: gate reads. The stack and the lamellar gratings settle under ``UNSETTLED_THRESHOLD`` (1e-5;
#: measured 6e-9 .. 9.9e-7). The crossed grating does not, and cannot be made to: its ``(1, 0)``
#: Rayleigh wavelength (1.2, the period) is at the band's edge, where an order leaves at
#: grazing and never quite does -- 3.0e-3 at 160 periods and 20 cells per wavelength (2.7e-3 at
#: 30, 5.0e-3 at 40). A longer run does not settle it, even under a 24-cell 1e-12 layer: 3.2e-3
#: / 1.4e-3 / 2.3e-3 / 5.2e-3 at 160 / 320 / 480 / 960 periods (rising again -- the late-time
#: growth ``test_fd_decay.py`` pins under the default layer is not gone under that one either).
#: What the orders at 0.95 and 1.05 read of it: they move up to 6.0e-3 between 160 and 480
#: periods and 4e-3 between 480 and 960, so every crossed comparison carries a run-length floor
#: of about 5e-3 -- 9x below the single-resolution gate's 4.5e-2, and 0.4x the finest sweep
#: error, which is why the sweep is read as a trend and not as a clean order
#: (:func:`test_the_crossed_gratings_ringing_is_not_what_its_orders_read`, slow, holds the
#: floor). So its line is 1e-2, over the finest sweep's 5.0e-3.
DECAY_LINE = {'stack': _assemble.UNSETTLED_THRESHOLD, 'lamellar': _assemble.UNSETTLED_THRESHOLD,
              'oblique': _assemble.UNSETTLED_THRESHOLD, 'crossed': 1e-2}


def assert_settled(solved):
    """Every Diffraction monitor of ``solved`` under its scene's :data:`DECAY_LINE`."""
    line = DECAY_LINE[solved.key[0]]
    ratios = solved.res.diagnostics['decay']
    for name, ratio in ratios.items():
        assert ratio is not None and ratio < line, (solved.key, name, ratio, line)
    return ratios


def _quiet():
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    for w in _QUIET:
        warnings.simplefilter('ignore', w)
    return ctx


class Solved:
    """One finished NumPy run, kept so that the post-processing can be re-run under a mutant."""

    def __init__(self, scene, info, key):
        self.scene, self.info, self.key = scene, info, key
        ctx = _quiet()
        try:
            self.plan = fd.plan(scene)
            self.rt = runtime('numpy')
            self.params = lower(scene, self.plan, self.rt)
            self.state = execute(self.plan, self.params, self.rt)
        finally:
            ctx.__exit__(None, None, None)
        self.res = self.results()

    def results(self):
        """The Results of this run, assembled now (so a monkeypatched ``_post`` applies)."""
        ctx = _quiet()
        try:
            return _results(self.scene, self.plan, self.state, self.rt, self.params)
        finally:
            ctx.__exit__(None, None, None)


@functools.lru_cache(maxsize=None)
def stack(polarization, cpw, theta=STACK['theta']):
    dx = 1.0 / cpw
    kwargs = dict(STACK, theta=theta)
    scene, info = bloch_stack(polarization=polarization, cells_per_wavelength=cpw,
                              transverse=(2 * dx, 2 * dx), **kwargs)
    return Solved(scene, info, ('stack', polarization, cpw, theta))


def _grating(key, polarization, cpw, extra=()):
    kwargs = dict(key)
    scene, info = box_grating(polarization=polarization, cells_per_wavelength=cpw, **kwargs)
    if extra:
        scene = scene.replace(monitors=scene.monitors + tuple(extra(info, scene)))
    return scene, info


def _lamellar_extra(info, scene):
    """Near planes for the de-embedding test, and a wider order window on the reflect plane."""
    ws = scene.monitors[0].wavelengths
    r = scene.monitors[0]
    return (fd.Diffraction('r_near', z=0.6 + 0.2, side='reflect', wavelengths=ws,
                           reference_plane=info['top']),
            fd.Diffraction('r_total', z=0.3, side='reflect', wavelengths=ws,
                           reference_plane=info['top']),
            fd.Diffraction('t_near', z=-0.2 - 0.2, side='transmit', wavelengths=ws,
                           reference_plane=info['bottom'], incident_plane=info['top']),
            fd.Diffraction('r_wide', z=r.z, side='reflect', wavelengths=ws,
                           reference_plane=info['top'], orders=(2, 0)))


@functools.lru_cache(maxsize=None)
def lamellar(polarization, cpw):
    scene, info = _grating(tuple(LAMELLAR.items()), polarization, cpw, _lamellar_extra)
    return Solved(scene, info, ('lamellar', polarization, cpw, 0.0))


@functools.lru_cache(maxsize=None)
def oblique(cpw):
    scene, info = _grating(tuple(OBLIQUE.items()), 's', cpw)
    return Solved(scene, info, ('oblique', 's', cpw, 20.0))


@functools.lru_cache(maxsize=None)
def crossed(cpw):
    scene, info = _grating(tuple(CROSSED.items()), 's', cpw)
    return Solved(scene, info, ('crossed', 's', cpw, 0.0))


#: RCWA solves, shared by every test (and every resolution: the twin's geometry is whole
#: cells at each resolution used here, so it does not depend on ``dx``).
_RCWA = {}


def rcwa_of(solved, index, truncation=None):
    """RCWA at the run's own angle for wavelength ``index`` (``theta_at``, fixed k)."""
    theta = float(to_numpy(solved.res['r'].theta_at)[index])
    if truncation is None:
        truncation = 0 if solved.key[0] == 'stack' else TRUNCATION[
            'crossed' if solved.key[0] == 'crossed' else 'lamellar']
    key = (solved.key[0], solved.key[1], solved.key[3], index, round(theta, 12), truncation)
    if key not in _RCWA:
        _RCWA[key] = solved.info['rcwa'](index, theta, truncation_order=truncation)
    return _RCWA[key]


def _propagating(solved, side, index, m, n):
    """Whether order ``(m, n)`` propagates on ``side`` at wavelength ``index``."""
    scene = solved.scene
    pw = next(s for s in scene.sources if isinstance(s, fd.PlaneWave))
    wl = solved.res[side].wavelengths[index]
    theta = math.radians(float(to_numpy(solved.res['r'].theta_at)[index]))
    phi = math.radians(pw.phi)
    k0 = 2 * math.pi / wl
    size = solved.plan.grid.size
    kx = k0 * math.sin(theta) * math.cos(phi) + 2 * math.pi * m / size[0]
    ky = k0 * math.sin(theta) * math.sin(phi) + 2 * math.pi * n / size[1]
    n_out = SUBSTRATE_INDEX if (side == 't' and solved.key[0] == 'stack') else 1.0
    return kx * kx + ky * ky < (n_out * k0) ** 2


def order_errors(solved, sides=('r', 't'), truncation=None, res=None):
    """``(worst |d amplitude|, worst |d efficiency|, rows)`` against RCWA over every
    propagating order, every wavelength and both outgoing polarisations."""
    if res is None:
        assert_settled(solved)
    res = solved.res if res is None else res
    amp = eff = 0.0
    rows = []
    for side in sides:
        d = res[side]
        names = ('r_s', 'r_p') if side == 'r' else ('t_s', 't_p')
        power = 'R' if side == 'r' else 'T'
        for i in range(len(d.wavelengths)):
            ref = rcwa_of(solved, i, truncation)
            for n in d.orders.orders_v:
                for m in d.orders.orders_u:
                    if not _propagating(solved, side, i, int(m), int(n)):
                        continue
                    e = abs(order_value(d.orders, power, m, n)[i]
                            - order_value(ref, power, m, n).real)
                    a = max(abs(order_value(d.orders, nm, m, n)[i] - order_value(ref, nm, m, n))
                            for nm in names)
                    rows.append((side, d.wavelengths[i], int(m), int(n), a, e))
                    amp, eff = max(amp, a), max(eff, e)
    return amp, eff, rows


def _table(rows):
    return '\n'.join(f'  {s} wl={w} ({m},{n}) |d amp|={a:.2e} |d eff|={e:.2e}'
                     for s, w, m, n, a, e in rows)


def fitted_order(cpws, errors):
    """The ``p`` of ``error ~ dx^p``, least squares in log-log over the resolutions."""
    return float(np.polyfit(np.log(1.0 / np.asarray(cpws, float)), np.log(errors), 1)[0])


# ---------------------------------------------------------------------------------------------
# (a) the uniform stack
# ---------------------------------------------------------------------------------------------

def stack_errors(solved):
    """``(|d|, |d magnitude| relative, |d phase| rad, |d efficiency|)``, the worst over r and t
    and the three wavelengths, of the run's own polarisation against RCWA."""
    assert_settled(solved)
    pol = solved.key[1]
    d = mag = phase = 0.0
    for side in ('r', 't'):
        name = f'{side}_{pol}'
        for i in range(len(solved.res[side].wavelengths)):
            f = order_value(solved.res[side].orders, name, 0)[i]
            g = order_value(rcwa_of(solved, i), name, 0)
            d = max(d, abs(f - g))
            mag = max(mag, abs(abs(f) / abs(g) - 1.0))
            phase = max(phase, abs(float(np.angle(f / g))))
    return d, mag, phase, order_errors(solved)[1]


@pytest.mark.parametrize('pol, amp_tol, mag_tol, phase_tol',
                         [('s', 1.0e-2, 1.5e-2, 1.2e-2), ('p', 1.5e-2, 2.0e-2, 6.0e-2)])
def test_a_uniform_stack_matches_rcwa_in_amplitude_and_phase(pol, amp_tol, mag_tol, phase_tol):
    """r and t, s and p, at 40 cells per wavelength and at each wavelength's own angle.

    Measured: s 6.5e-3 complex, 9.5e-3 in magnitude, 7.9e-3 rad in phase; p 9.8e-3, 1.25e-2 and
    4.0e-2 rad; ``R`` and ``T`` 7.7e-4 / 6.5e-4, and ``R + T = 1`` to 2.6e-5. The other
    polarisation's amplitude is zero.
    """
    solved = stack(pol, 40)
    d, mag, phase, eff = stack_errors(solved)
    assert d < amp_tol and mag < mag_tol and phase < phase_tol, (d, mag, phase)
    assert eff < 1.2e-3, eff
    res = solved.res
    other = 'p' if pol == 's' else 's'
    for side in ('r', 't'):
        cross = order_value(res[side].orders, f'{side}_{other}', 0)
        assert np.max(np.abs(cross)) < 1e-12, cross
    total = to_numpy(res['r'].orders.incident.R + res['t'].orders.incident.T).ravel()
    assert np.max(np.abs(total - 1.0)) < 1e-4, total


def test_theta_at_is_the_fixed_k_angle():
    """``sin(theta_at) / wavelength`` is the band centre's, at every wavelength (fixed k)."""
    res = stack('s', 20).res
    th = np.radians(to_numpy(res['r'].theta_at))
    ws = np.asarray(res['r'].wavelengths)
    expected = math.sin(math.radians(STACK['theta'])) / 1.0
    assert np.allclose(np.sin(th) / ws, expected, rtol=0, atol=1e-13), np.degrees(th)
    assert abs(np.degrees(th[1]) - STACK['theta']) < 1e-12
    assert np.allclose(to_numpy(res['t'].theta_at), to_numpy(res['r'].theta_at), atol=0)


def test_the_p_sign_is_results_at_normal_incidence():
    """``r_p = -r_s`` and ``t_p = t_s`` at normal incidence -- ``wavebit.observations.diffraction``'s p sign
    -- from the two runs of ``polarizations='both'``, and each run matches RCWA's."""
    scene, info = bloch_stack(theta=0.0, polarization='s', layers=STACK['layers'],
                              substrate=STACK['substrate'], cells_per_wavelength=40,
                              transverse=(2 / 40, 2 / 40), both=True)
    ctx = _quiet()
    try:
        res = fd.run(scene)
    finally:
        ctx.__exit__(None, None, None)
    assert res.diagnostics['runs'] == 2 and res['r'].runs == 2
    r, t = res['r'].orders, res['t'].orders
    assert r.s_inc is r.incident                   # the scene's own run is the s one
    rs, rp = to_numpy(r.s_inc.r_s)[:, 0, 0], to_numpy(r.p_inc.r_p)[:, 0, 0]
    ts, tp = to_numpy(t.s_inc.t_s)[:, 0, 0], to_numpy(t.p_inc.t_p)[:, 0, 0]
    assert np.max(np.abs(rp + rs)) < 1e-12 * np.max(np.abs(rs)), (rs, rp)
    assert np.max(np.abs(tp - ts)) < 1e-12 * np.max(np.abs(ts)), (ts, tp)
    assert np.max(np.abs(to_numpy(r.s_inc.r_p))) < 1e-12
    twin_p = bloch_stack(theta=0.0, polarization='p', layers=STACK['layers'],
                         substrate=STACK['substrate'], cells_per_wavelength=40,
                         transverse=(2 / 40, 2 / 40))[1]['rcwa']
    for i in range(3):
        ref_s, ref_p = info['rcwa'](i, 0.0).order(0, 0), twin_p(i, 0.0).order(0, 0)
        assert abs(ref_p.r_p + ref_s.r_s) < 1e-12              # RCWA's own sign, the same
        assert abs(ref_p.t_p - ref_s.t_s) < 1e-12
        assert abs(rs[i] - ref_s.r_s) < 1e-2 and abs(rp[i] - ref_p.r_p) < 1e-2
        assert abs(ts[i] - ref_s.t_s) < 1e-2 and abs(tp[i] - ref_p.t_p) < 1e-2


@pytest.mark.parametrize('phi', [30.0, 90.0])
def test_at_normal_incidence_the_frame_is_the_plane_waves_own_phi(phi):
    """At ``K_par = 0`` no order has a plane of incidence, so ``_post.frame`` takes the s/p
    frame from the PlaneWave's ``phi`` -- as ``wavebit.rcwa`` does at the same ``phi``. The
    two runs of ``polarizations='both'`` must then read each incidence whole in its own
    polarisation (the other one zero), ``R_s + R_p = R``, ``r_p = -r_s``, and each amplitude
    must match RCWA's at that ``phi``. With the frame taken at ``phi = 0`` instead (review
    mutant M16) the s amplitude at ``phi = 90`` divides by zero (``nan``) and at 30 it is
    ``cos 30`` of itself, while ``R`` and ``T`` do not move -- which is why this test reads
    amplitudes."""
    kwargs = dict(theta=0.0, phi=phi, layers=STACK['layers'], substrate=STACK['substrate'],
                  cells_per_wavelength=40, transverse=(2 / 40, 2 / 40))
    scene, info = bloch_stack(polarization='s', both=True, **kwargs)
    ctx = _quiet()
    try:
        res = fd.run(scene)
    finally:
        ctx.__exit__(None, None, None)
    r, t = res['r'].orders, res['t'].orders
    rs, rp = to_numpy(r.s_inc.r_s)[:, 0, 0], to_numpy(r.p_inc.r_p)[:, 0, 0]
    ts, tp = to_numpy(t.s_inc.t_s)[:, 0, 0], to_numpy(t.p_inc.t_p)[:, 0, 0]
    assert np.all(np.isfinite(rs)) and np.all(np.isfinite(ts)), (rs, ts)
    for cross in (r.s_inc.r_p, t.s_inc.t_p, r.p_inc.r_s, t.p_inc.t_s):
        assert np.max(np.abs(to_numpy(cross))) < 1e-12, cross
    assert np.max(np.abs(rp + rs)) < 1e-12 * np.max(np.abs(rs)), (rs, rp)
    for run in (r.s_inc, r.p_inc):
        total = to_numpy(run.R_s) + to_numpy(run.R_p)
        assert np.max(np.abs(total - to_numpy(run.R))) < 1e-14, (total, run.R)
    twin_p = bloch_stack(polarization='p', **kwargs)[1]['rcwa']
    for i in range(3):
        ref_s, ref_p = info['rcwa'](i, 0.0).order(0, 0), twin_p(i, 0.0).order(0, 0)
        assert abs(ref_p.r_p + ref_s.r_s) < 1e-12              # RCWA's frame turns with phi
        assert abs(rs[i] - ref_s.r_s) < 1e-2 and abs(rp[i] - ref_p.r_p) < 1e-2, (rs, ref_s.r_s)
        assert abs(ts[i] - ref_s.t_s) < 1e-2 and abs(tp[i] - ref_p.t_p) < 1e-2, (ts, ref_s.t_s)


def test_a_uniform_stack_converges_second_order_except_the_p_phase():
    """20 / 40 / 80 cells per wavelength (each run is seconds). Measured:

        s  magnitude 4.1e-2 / 9.5e-3 / 2.4e-3 (p = 2.0)   phase 3.2e-2 / 7.9e-3 / 2.0e-3 (2.0)
        p  magnitude 5.4e-2 / 1.3e-2 / 3.2e-3 (2.0)       phase 7.8e-2 / 4.0e-2 / 2.0e-2 (0.98)
        R, T  3.4e-3 / 7.7e-4 / 2.0e-4 (s), 2.9e-3 / 6.5e-4 / 1.7e-4 (p): second order

    The p wave's error is a phase, first order: its normal E sees the snapped interfaces where
    the cells are and its tangential E half a cell lower (``tests/fdtd/test_fd_airy.py``), so no one
    reference plane is right for both -- stage 4's subpixel averaging is what closes it.
    """
    cpws = (20, 40, 80)
    table = {pol: np.array([stack_errors(stack(pol, c)) for c in cpws]) for pol in 'sp'}
    orders = {pol: [fitted_order(cpws, table[pol][:, j]) for j in range(4)] for pol in 'sp'}
    msg = '; '.join(f'{pol}: {table[pol].tolist()} orders {np.round(orders[pol], 2)}'
                    for pol in 'sp')
    for pol in 'sp':
        assert np.all(np.diff(table[pol], axis=0) < 0), msg
    s_d, s_mag, s_phase, s_eff = orders['s']
    p_d, p_mag, p_phase, p_eff = orders['p']
    assert min(s_d, s_mag, s_phase, s_eff) > 1.8, msg
    assert min(p_mag, p_eff) > 1.8, msg
    assert 0.8 < p_phase < 1.3, msg                  # first order, and stated as such
    assert table['s'][2, 0] < 2.5e-3 and table['p'][2, 0] < 5e-3, msg


# ---------------------------------------------------------------------------------------------
# (b, c) the gratings at one resolution
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('pol, amp_tol, eff_tol, asym_tol',
                         [('s', 4.5e-2, 7.5e-3, 1e-5), ('p', 2.0e-2, 1.0e-2, 1.2e-2)])
def test_a_lamellar_grating_at_normal_incidence_matches_rcwa(pol, amp_tol, eff_tol, asym_tol):
    """Every propagating order (0, +-1) at 0.9 / 1.0 / 1.1, 20 cells per wavelength.

    Measured: s 2.9e-2 in amplitude, 4.8e-3 in efficiency; p 1.2e-2, 6.8e-3; ``R + T = 1`` to
    the run's settling. The +-1 efficiencies of this symmetric tooth: for s (``E_y`` alone,
    which sees the RCWA tooth moved by ``-dx/2`` along x -- a shift ``|r|`` cannot see) they are
    equal to 1e-7 of the largest; for p the ``E_x`` sees the tooth where it is and the ``E_z``
    half a cell over, which is not a shift but an asymmetric tooth, and they differ by 6.7e-3
    of it (the staircase, first order, stage 4).
    """
    solved = lamellar(pol, 20)
    amp, eff, rows = order_errors(solved)
    assert amp < amp_tol and eff < eff_tol, _table(rows)
    assert len(rows) == 2 * 3 * 3
    res = solved.res
    for side, power in (('r', 'R'), ('t', 'T')):
        v = to_numpy(getattr(res[side].orders.incident, power))
        assert np.max(np.abs(v[:, 0, 0] - v[:, 0, 2])) < asym_tol * np.max(v), v
    total = (to_numpy(res['r'].orders.incident.R).sum(axis=(1, 2))
             + to_numpy(res['t'].orders.incident.T).sum(axis=(1, 2)))
    assert np.max(np.abs(total - 1.0)) < 2e-3, total


def test_a_lamellar_grating_at_20_degrees_matches_rcwa():
    """Orders 0 and -1 at 1.0 / 1.1 (period 1.2, s), 20 cells per wavelength.

    Measured 2.1e-2 in amplitude and 1.3e-2 in efficiency (``R_0``: the slab's guided
    resonance is near, and a half-cell staircase moves it).
    """
    solved = oblique(20)
    amp, eff, rows = order_errors(solved)
    assert amp < 3.5e-2 and eff < 2.0e-2, _table(rows)
    assert {(m, n) for _, _, m, n, _, _ in rows} == {(0, 0), (-1, 0)}
    th = to_numpy(solved.res['r'].theta_at)
    assert abs(th[0] - 20.0) < 1e-12 and th[1] > th[0]


def test_a_crossed_grating_matches_rcwa_with_the_orders_laid_out_v_first():
    """Every propagating order of a square pillar (the 3 x 3 block less the corners) at 0.95 and
    1.05, 20 cells per wavelength -- and fd's ``(nf, n_v, n_u)`` arrays lie over RCWA's
    ``(n_v, n_u)`` ones entry for entry, which their transpose does not.

    Measured: 2.7e-2 in amplitude, 5.8e-3 in efficiency (``T_00``), and the reflect efficiencies
    to 1e-3 -- against 7.5e-3 between ``R(1, 0)`` and ``R(0, 1)`` for an s wave along y, which is
    what a transposed layout would be read against.
    """
    solved = crossed(20)
    amp, eff, rows = order_errors(solved)
    assert amp < 4.5e-2 and eff < 1.0e-2, _table(rows)
    # The run's unsettled ringing (DECAY_LINE) is a tenth of the tolerance: measured 3.0e-3.
    assert max(solved.res.diagnostics['decay'].values()) < 4.5e-2 / 10
    # RCWA's truncation 4 is converged far below the tolerance: within 8.1e-4 of 10, and here
    # held against 6 (a 2-D solve at 6 is ~1 s) over every compared value.
    gap = 0.0
    for i in range(2):
        a, b = rcwa_of(solved, i, TRUNCATION['crossed']), rcwa_of(solved, i, 6)
        for _, _, m, n, _, _ in rows:
            for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
                gap = max(gap, abs(order_value(a, name, m, n) - order_value(b, name, m, n)))
    assert gap < 4.5e-2 / 20, gap
    assert {(m, n) for _, _, m, n, _, _ in rows} == {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}
    orders = solved.res['r'].orders
    N = TRUNCATION['crossed']
    for i in range(2):
        mine = to_numpy(orders.at(i).R)
        ref = np.real(np.asarray(rcwa_of(solved, i).R))[N - 1:N + 2, N - 1:N + 2]
        assert mine.shape == ref.shape == (3, 3)
        assert np.max(np.abs(mine - ref)) < 1.5e-3, (mine, ref)
        assert np.max(np.abs(mine.T - ref)) > 5e-3, 'the check cannot tell a transpose'


# ---------------------------------------------------------------------------------------------
# The Result contract: order_window, .order(m, n), orders_u / orders_v, .at(i)
# ---------------------------------------------------------------------------------------------

def test_the_orders_are_a_result_indexed_as_rcwa_indexes_its_own():
    solved = lamellar('s', 20)
    res = solved.res
    orders = res['r'].orders
    assert isinstance(orders, Result)
    assert orders.order_window == res['r'].order_window == ((0, 1), (0, 0))
    assert list(orders.orders_u) == [-1, 0, 1] and list(orders.orders_v) == [0]
    nf = len(res['r'].wavelengths)
    assert to_numpy(orders.R).shape == (nf, 1, 3)
    ref = rcwa_of(solved, 1)
    small = solved.info['rcwa'](1, 0.0, truncation_order=1)
    for i in range(nf):
        one = orders.at(i)
        assert type(one) is Result
        assert to_numpy(one.R).shape == np.asarray(small.R).shape == (1, 3)
        assert list(one.orders_u) == list(small.orders_u)
        assert list(one.orders_v) == list(small.orders_v)
        for m in (-1, 0, 1):
            view = orders.order(m, 0)
            assert (view.m, view.n) == (m, 0)
            assert to_numpy(view.R).shape == (nf,)
            assert to_numpy(view.R)[i] == to_numpy(one.order(m, 0).R)
            assert to_numpy(view.r_s)[i] == to_numpy(one.order(m, 0).r_s)
            assert to_numpy(view.R)[i] == to_numpy(orders.R)[i, 0, m + 1]
            assert view.T is None and view.t_s is None            # a reflect monitor
    for bad in ((2, 0), (0, 1), (-2, 0)):
        with pytest.raises(IndexError):
            orders.order(*bad)
        with pytest.raises(IndexError):
            small.order(*bad)
    assert ref.order(2, 0) is not None                       # RCWA's window is wider


def test_a_wider_order_window_adds_the_evanescent_orders_and_moves_nothing_else():
    """``orders=(2, 0)`` on the same plane: the (-1..1) orders are the default window's to
    round-off, and the +-2 orders -- evanescent at every wavelength -- carry no power, as in
    RCWA."""
    res = lamellar('s', 20).res
    wide, narrow = res['r_wide'].orders, res['r'].orders
    assert wide.order_window == ((0, 2), (0, 0))
    assert list(wide.orders_u) == [-2, -1, 0, 1, 2]
    for m in (-1, 0, 1):
        for name in ('R', 'r_s', 'r_p'):
            a, b = order_value(wide, name, m), order_value(narrow, name, m)
            assert np.max(np.abs(a - b)) < 1e-12 * max(1.0, np.max(np.abs(b))), (m, name)
    for m in (-2, 2):
        assert np.max(np.abs(order_value(wide, 'R', m))) < 1e-12


def test_rcwa_truncation_is_converged_below_the_fdtd_error():
    """Each reference is RCWA's converged answer, not an artefact of its truncation.

    The uniform stack is exact at truncation 0; the lamellar grating at 20 is within 1.7e-5 (s)
    and 4.4e-5 (p) of 30, over 100x below the finest FDTD error it judges (5.5e-3). The crossed
    grating's (and the 20-degree lamellar's) are in their slow sweeps -- a 2-D truncation 10
    solve is ~20 s.
    """
    s = stack('s', 20)
    for i in range(3):
        a, b = rcwa_of(s, i, 0), rcwa_of(s, i, 3)
        for name in ('r_s', 't_s', 'R', 'T'):
            assert abs(order_value(a, name, 0) - order_value(b, name, 0)) < 1e-12
    worst = 0.0
    for pol in 'sp':
        lam = lamellar(pol, 20)
        for i in range(3):
            a, b = rcwa_of(lam, i, TRUNCATION['lamellar']), rcwa_of(lam, i, 30)
            for m in (-1, 0, 1):
                for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
                    worst = max(worst, abs(order_value(a, name, m) - order_value(b, name, m)))
    assert worst < 5.5e-3 / 20, worst


# ---------------------------------------------------------------------------------------------
# Mutants
# ---------------------------------------------------------------------------------------------

def _yee_and_continuum_kz(solved, monitor):
    """``(kappa, kz_c)`` per order of the lamellar cell at normal incidence, ``(nf, 3)``, from
    the discrete dispersion relation -- independent of the engine's own arithmetic."""
    plan = solved.plan
    dx, dt = plan.grid.dx, plan.dt
    L = plan.grid.size[0]
    omega = 2 * np.pi / np.asarray(solved.res[monitor].wavelengths)[:, None]
    ku = 2 * np.pi * np.arange(-1, 2)[None, :] / L
    Omega = 2 * np.sin(omega * dt / 2) / dt
    Ku = 2 * np.sin(ku * dx / 2) / dx
    kappa = (2 / dx) * np.arcsin(np.sqrt(Omega ** 2 - Ku ** 2) * dx / 2)
    return kappa, np.sqrt(omega ** 2 - ku ** 2)


def test_the_de_embedding_is_exact_and_the_continuum_kz_is_not(monkeypatch):
    """The same orders read on planes 1.3 apart agree with the Yee kz and disagree with the
    continuum one by exactly ``(kz_c - kappa) * dz`` -- predicted, per order, from the discrete
    dispersion relation.

    Measured (20 cells per wavelength): Yee planes agree to 2e-5 (1.4e-4 for the plane between
    the sheet and the tooth, in the total field); the continuum's disagree by 3e-3 to 4e-2, and
    the +-1 orders (34-43 degrees) carry 0.61-0.64 of the zeroth order's phase error -- the
    mismatch falls toward the grid's diagonal and grows only past it (module docstring).
    """
    solved = lamellar('s', 20)
    good = solved.res
    monkeypatch.setattr(_post, 'DIFFRACTION_KZ', 'continuum')
    mutant = solved.results()
    kappa, kz_c = _yee_and_continuum_kz(solved, 'r')
    assert np.allclose(to_numpy(good['r'].kz)[:, 0, :], kappa, rtol=0, atol=1e-12)
    assert np.allclose(to_numpy(mutant['r'].kz)[:, 0, :], kz_c, rtol=0, atol=1e-12)
    shift = kz_c - kappa
    for far, near, name, direction in (('r', 'r_near', 'r_s', 1.0), ('r', 'r_total', 'r_s', 1.0),
                                       ('t', 't_near', 't_s', -1.0)):
        dz = good[near].plane - good[far].plane
        tol = 5e-4 if near == 'r_total' else 1e-4
        for j, m in enumerate((-1, 0, 1)):
            g_far, g_near = order_value(good[far].orders, name, m), order_value(
                good[near].orders, name, m)
            m_far, m_near = order_value(mutant[far].orders, name, m), order_value(
                mutant[near].orders, name, m)
            assert np.max(np.abs(g_far - g_near)) < tol, (far, near, m)
            assert np.min(np.abs(m_far - m_near)) > 2.5e-3, (far, near, m)
            measured = np.angle((m_far / g_far) / (m_near / g_near))
            predicted = direction * shift[:, j] * dz
            assert np.allclose(measured, predicted, rtol=0, atol=1e-9), (measured, predicted)
    ratio = shift[:, 0] / shift[:, 1]
    assert np.all((ratio > 0.6) & (ratio < 0.75)), ratio      # 34-43 deg: less than normal


def _frame_reversed_on_negative_orders(original):
    def frame(rt, Ku, Kv, phi):
        s_u, s_v, q_u, q_v = original(rt, Ku, Kv, phi)
        xp = rt.xp
        flip = xp.where(xp.real(Ku * (1.0 + 0.0j)) < 0.0, -1.0, 1.0)
        return s_u * flip, s_v * flip, q_u * flip, q_v * flip
    return frame


def test_a_reversed_frame_on_the_negative_orders_is_caught(monkeypatch):
    """``|r|^2`` cannot see a reversed frame: R, T move by nothing; the -1 order's complex r_s
    and t_s miss RCWA by twice their size (up to 0.43), far past the gate's 4.5e-2."""
    solved = lamellar('s', 20)
    good_amp, _, _ = order_errors(solved)
    monkeypatch.setattr(_post, 'frame', _frame_reversed_on_negative_orders(_post.frame))
    mutant = solved.results()
    for side, power in (('r', 'R'), ('t', 'T')):
        assert np.array_equal(to_numpy(getattr(mutant[side].orders.incident, power)),
                              to_numpy(getattr(solved.res[side].orders.incident, power)))
    amp, _, rows = order_errors(solved, res=mutant)
    worst_minus = max(a for _, _, m, _, a, _ in rows if m == -1)
    worst_rest = max(a for _, _, m, _, a, _ in rows if m != -1)
    assert good_amp < 4.5e-2
    assert worst_minus > 0.15, _table(rows)
    assert worst_rest < 4.5e-2                             # only the -m orders move


def _analytic_incident_amplitude(series, wavelengths, plane=None):
    """The legacy normalisation: the pulse's analytic spectrum, carried with the continuum kz."""
    ctx = series.context
    src = next(s for s in ctx.scene.sources if s.name == series.name)
    band = src.band
    f = 1.0 / np.asarray(wavelengths, dtype=float)
    t = np.linspace(0.0, ctx.plan.steps * ctx.plan.dt, 400001)
    w = band.waveform(t, amplitude=float(src.amplitude), phase=float(src.phase),
                      delay=float(src.delay))
    A = np.trapezoid(w[None, :] * np.exp(2j * np.pi * f[:, None] * t[None, :]), t, axis=1)
    if plane is not None:
        k_par = 2 * np.pi / band.concrete_wavelength * math.sin(math.radians(src.theta))
        kz = np.sqrt((2 * np.pi * f) ** 2 - k_par ** 2 + 0j)
        A = A * np.exp(1j * src.sign * kz * (plane - series.plane))
    return ctx.rt.array(A, 'complex128')


@pytest.mark.parametrize('pol', ['s', 'p'])
def test_an_analytic_pulse_normalisation_is_caught(monkeypatch, pol):
    """Normalised by the pulse's analytic spectrum instead of the column, the stack's r and t
    miss RCWA by 0.14 (s) / 0.12 (p) at 40 cells per wavelength (the gate: 1e-2 / 1.5e-2), and R, T
    -- normalised by the column's flux -- do not move."""
    solved = stack(pol, 40)
    good_amp, good_eff, _ = order_errors(solved)
    monkeypatch.setattr(_assemble, 'incident_amplitude', _analytic_incident_amplitude)
    mutant = solved.results()
    amp, eff, rows = order_errors(solved, res=mutant)
    assert good_amp < 1.5e-2
    assert amp > 3 * 1.5e-2, _table(rows)
    assert eff == good_eff


# ---------------------------------------------------------------------------------------------
# A defect (kernel, stage 2): the conical launch
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('pol', ['s', 'p'])
def test_a_conical_plane_wave_is_launched_without_the_other_polarisation(pol):
    """A uniform stack is polarisation-preserving at any angle: at theta 25, phi 60 an s wave
    reflects and transmits no p, and a p wave no s (RCWA: 1e-16). On the grid too: with the
    stagger removed its equations are the continuum's with ``k -> K``, so the grid's own s and
    p waves do not couple in a layered cell, and the PlaneWave launches exactly one of them.
    Measured at 40 cells per wavelength: 6.4e-16 (s) / 3.9e-16 (p); before the drive carried
    the stagger phase, 4.7e-3 / 3.9e-3. The co-polarised amplitudes still match RCWA as
    :func:`test_a_uniform_stack_matches_rcwa_in_amplitude_and_phase` holds them at phi 0."""
    scene, info = bloch_stack(theta=25.0, phi=60.0, polarization=pol, layers=STACK['layers'],
                              substrate=STACK['substrate'], cells_per_wavelength=40,
                              transverse=(2 / 40, 2 / 40))
    ctx = _quiet()
    try:
        res = fd.run(scene)
    finally:
        ctx.__exit__(None, None, None)
    other = 'p' if pol == 's' else 's'
    th = to_numpy(res['r'].theta_at)
    d = 0.0
    for i in range(3):
        ref = info['rcwa'](i, th[i])
        assert abs(order_value(ref, f't_{other}', 0)) < 1e-12
        for side in ('r', 't'):
            f = complex(to_numpy(order_value(res[side].orders, f'{side}_{pol}', 0))[i])
            d = max(d, abs(f - complex(order_value(ref, f'{side}_{pol}', 0))))
    leak = max(np.max(np.abs(order_value(res['r'].orders, f'r_{other}', 0))),
               np.max(np.abs(order_value(res['t'].orders, f't_{other}', 0))))
    assert leak < 1e-13, leak
    assert d < 1.5e-2, d                                        # measured 6.5e-3 / 9.8e-3


# ---------------------------------------------------------------------------------------------
# The resolution sweeps (slow): the error falls toward RCWA's converged answer
# ---------------------------------------------------------------------------------------------

def _sweep(build, cpws, truncation=None):
    amps, effs, tables = [], [], []
    for cpw in cpws:
        a, e, rows = order_errors(build(cpw), truncation=truncation)
        amps.append(a)
        effs.append(e)
        tables.append(f'{cpw} cells per wavelength:\n{_table(rows)}')
    return amps, effs, '\n'.join(tables)


@pytest.mark.slow
@pytest.mark.parametrize('pol, need, finest', [('s', 0.9, 8e-3), ('p', 0.9, 5e-3)])
def test_a_lamellar_grating_converges_toward_rcwa(pol, need, finest):
    """20 / 40 / 80 cells per wavelength at normal incidence (~45 s). Measured s: 2.9e-2 /
    1.2e-2 / 5.5e-3, first order (1.2); p: 1.2e-2 / 3.0e-3 / see the message."""
    cpws = (20, 40, 80)
    amps, effs, table = _sweep(lambda c: lamellar(pol, c), cpws)
    p = fitted_order(cpws, amps)
    msg = f'amp {amps} (p={p:.2f}) eff {effs} (p={fitted_order(cpws, effs):.2f})\n{table}'
    assert amps[0] > amps[1] > amps[2], msg
    assert p > need and amps[2] < finest, msg
    assert effs[2] < effs[0], msg


@pytest.mark.slow
def test_an_oblique_lamellar_grating_converges_toward_rcwa():
    """20 / 40 / 80 cells per wavelength at 20 degrees (~85 s). Measured 2.1e-2 / 1.1e-2 /
    5.7e-3: first order (0.95). RCWA at truncation 20 is within 4e-6 of 30."""
    cpws = (20, 40, 80)
    amps, effs, table = _sweep(oblique, cpws)
    p = fitted_order(cpws, amps)
    msg = f'amp {amps} (p={p:.2f}) eff {effs} (p={fitted_order(cpws, effs):.2f})\n{table}'
    assert amps[0] > amps[1] > amps[2], msg
    assert p > 0.8 and amps[2] < 8e-3, msg
    assert effs[2] < effs[0], msg
    solved = oblique(20)
    worst = 0.0
    for i in range(2):
        a, b = rcwa_of(solved, i), rcwa_of(solved, i, 30)
        for m in (-1, 0):
            for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
                worst = max(worst, abs(order_value(a, name, m) - order_value(b, name, m)))
    assert worst < amps[2] / 20, (worst, amps[2])


@pytest.mark.slow
def test_the_crossed_gratings_ringing_is_not_what_its_orders_read():
    """The crossed grating is compared unsettled (:data:`DECAY_LINE`): its ``(1, 0)`` Rayleigh
    wavelength 1.2 sits at the band's edge, whose grazing content leaves the cell too slowly for
    any run length. Under a ``PML(24, 1e-12)`` layer the ratio is 3.2e-3 at 160 periods and
    still 2.3e-3 at 480, and the compared orders at 0.95 and 1.05 move up to 6.0e-3 between the
    two (the ``t_p`` of the ``(0, +-1)`` orders and the ``t_s`` of ``(1, 0)`` at 1.05, nearest
    the anomaly). That floor is what the ringing costs the gate: held here under a fifth of the
    single-resolution tolerance 4.5e-2, it is 9x below it. (Measured 2026-09-19, 20 cells per
    wavelength, ~1 min.)"""
    runs = {}
    for periods in (160, 480):
        scene, info = _grating(tuple(dict(CROSSED, pml=24).items()), 's', 20)
        scene = scene.replace(run=fd.Run(periods=periods), boundaries=fd.Boundaries(
            x=scene.boundaries.face('x-'), y=scene.boundaries.face('y-'),
            z=fd.PML(cells=24, reflection=1e-12)))
        runs[periods] = Solved(scene, info, ('crossed', 's', 20, 0.0)).res
    ratios = [max(runs[p].diagnostics['decay'].values()) for p in (160, 480)]
    assert all(1e-3 < r < 1e-2 for r in ratios), ratios           # it does not settle
    move = 0.0
    for side, names in (('r', ('r_s', 'r_p', 'R')), ('t', ('t_s', 't_p', 'T'))):
        for m, n in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            for name in names:
                a = order_value(runs[160][side].orders, name, m, n)
                b = order_value(runs[480][side].orders, name, m, n)
                move = max(move, float(np.max(np.abs(a - b))))
    assert move < 4.5e-2 / 5, (move, ratios)


@pytest.mark.slow
def test_a_crossed_grating_converges_toward_rcwa():
    """20 / 30 / 40 cells per wavelength (the pillar and the cell are whole cells at each;
    ~3 min, the 40 run is 212k cells x 22k steps). Measured 2.7e-2 / 1.8e-2 / 1.3e-2: first
    order (1.05). RCWA's own truncation, 6 against 10, moves the reference by at most 4.1e-4 --
    26x below the finest FDTD error. The runs are not settled (:data:`DECAY_LINE`: the band-edge
    Rayleigh ringing, a run-length floor of about 5e-3 on every order), which is 0.4x the finest
    error: the fall is real, the fitted order is a trend with that floor in it."""
    cpws = (20, 30, 40)
    amps, effs, table = _sweep(crossed, cpws, CROSSED_SWEEP_TRUNCATION)
    p = fitted_order(cpws, amps)
    msg = f'amp {amps} (p={p:.2f}) eff {effs} (p={fitted_order(cpws, effs):.2f})\n{table}'
    assert amps[0] > amps[1] > amps[2], msg
    assert p > 0.8 and amps[2] < 1.8e-2, msg
    solved = crossed(20)
    worst = 0.0
    for i in range(2):
        a = rcwa_of(solved, i, CROSSED_SWEEP_TRUNCATION)
        b = rcwa_of(solved, i, 10)
        for m, n in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
                worst = max(worst, abs(order_value(a, name, m, n) - order_value(b, name, m, n)))
    assert worst < amps[2] / 20, (worst, amps[2])


@pytest.mark.parametrize('pol', ('s', 'p'))
def test_cartesian_orders_preserve_spectral_indexing_and_report_only_measured_power(pol):
    """Cartesian output is the same resolved electric wave, with a separate component axis."""
    solved = stack(pol, 20)
    for side, direction in (('r', 1), ('t', -1)):
        monitor = solved.res[side]
        orders = monitor.orders
        assert not hasattr(orders, 'given')
        assert orders.incident is not None
        assert orders.power_entering is None and orders.absorption is None
        grid = solved.plan.grid
        np.testing.assert_allclose(monitor.lateral_origin,
                                   [grid.low[a] + grid.size[a] / 2 for a in (0, 1)],
                                   rtol=0, atol=0)
        vector = to_numpy(getattr(orders, f'{side}_xyz'))
        assert vector.shape == to_numpy(orders.R if side == 'r' else orders.T).shape + (3,)
        view = orders.order(0, 0)
        np.testing.assert_array_equal(getattr(view, f'{side}_xyz'), vector[:, 0, 0, :])
        for i in range(len(monitor.wavelengths)):
            sliced = orders.at(i)
            np.testing.assert_array_equal(getattr(sliced.order(0, 0), f'{side}_xyz'),
                                          vector[i, 0, 0, :])
            assert sliced.power_entering is None and sliced.absorption is None
        # This fixture is in the x-z incidence plane. Its y component is s and its
        # x/z components form the p wave perpendicular to the discrete wavevector.
        s_amp = to_numpy(getattr(orders, f'{side}_s'))
        p_amp = to_numpy(getattr(orders, f'{side}_p'))
        np.testing.assert_allclose(vector[..., 1], s_amp, rtol=1e-13, atol=1e-14)
        kp = 2 * np.sin(to_numpy(monitor.kz) * grid.dx / 2) / grid.dx
        ku = 2 * np.sin(math.sin(math.radians(STACK['theta'])) * 2 * math.pi * grid.dx / 2) / grid.dx
        np.testing.assert_allclose(ku * vector[..., 0] + direction * kp * vector[..., 2],
                                   0, rtol=0, atol=1e-13)
        knorm = np.sqrt(kp ** 2 + ku ** 2)
        np.testing.assert_allclose(vector[..., 0], p_amp * direction * kp / knorm,
                                   rtol=1e-13, atol=1e-14)
        missing = 't_xyz' if side == 'r' else 'r_xyz'
        assert getattr(orders, missing) is None and getattr(view, missing) is None


@pytest.mark.parametrize('axis', ('x', 'y', 'z'))
@pytest.mark.parametrize('direction', (-1, 1))
@pytest.mark.parametrize('pol', ('s', 'p'))
def test_cartesian_diffraction_uses_global_axes_for_both_propagation_directions(axis, direction, pol):
    """A normally incident vacuum plane wave has one known global electric component."""
    P = 'xyz'.index(axis)
    size = [0.1, 0.1, 0.1]
    size[P] = 4.0
    boundaries = {a: fd.Periodic() for a in 'xyz'}
    boundaries[axis] = fd.PML(cells=8, wavelength=1.0, allow_thin=True)
    src = fd.PlaneWave('pw', **{axis: -0.5 * direction},
                       direction=('+' if direction > 0 else '-') + axis,
                       polarization=pol, band=fd.Band(wavelength=1.0, df=0.15))
    monitor = fd.Diffraction('t', **{axis: 0.5 * direction}, side='transmit',
                             wavelengths=(0.9, 1.1), orders=0)
    scene = fd.Scene(domain=fd.Domain(size=tuple(size)), boundaries=fd.Boundaries(**boundaries),
                     resolution=fd.Resolution(dx=0.05), run=fd.Run(steps=800),
                     sources=(src,), monitors=(monitor,))
    ctx = _quiet()
    try:
        res = fd.run(scene)
    finally:
        ctx.__exit__(None, None, None)
    order = res['t'].orders.order(0, 0)
    expected = np.zeros((2, 3), dtype=complex)
    component = (P + (2 if pol == 's' else 1)) % 3
    amplitude = order.t_s if pol == 's' else order.t_p
    expected[:, component] = amplitude * (1 if pol == 's' else direction)
    assert np.min(np.abs(amplitude)) > 0.5
    np.testing.assert_allclose(order.t_xyz, expected, rtol=1e-13, atol=1e-14)
