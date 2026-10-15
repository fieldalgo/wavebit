"""Gate 6 of stage 3 -- ``fd.Axis1D``: a one-cell axis whose derivative is ``i k`` exactly.

On an ``Axis1D`` axis the grid holds one cell of a field ``F exp(i k y)``; the curl's difference
along it is replaced by the product ``i k dx F`` (``src/wavebit/fdtd/_kernel/curl.py``), forward and
backward alike, so the leapfrog's dispersion is

    sin^2(w dt/2) / dt^2 = (K_x^2 + k^2 + K_z^2) / 4,    K = 2 sin(k dx/2) / dx on a differenced axis

with the *exact* ``k`` on the Axis1D axis. What is pinned here, each against something the
engine did not compute itself:

* **The dispersion is exact along the axis.** A discrete plane wave (periodic 16 x 1 x 16, y an
  ``Axis1D``) recurs at the frequency above to round-off (measured <= 5.6e-16 for k = 3 .. 12,
  on NumPy, JAX and Torch) and *not* at the one-cell Bloch frequency (``K_y = 2 sin(k dx/2)/dx``:
  8.6e-6 at k = 3, 6.9e-4 at k = 9), so the check can tell the two apart.
* **A conical 1-D lamellar grating matches ``wavebit.rcwa`` at matched geometry**
  (``fd_helpers.conical_grating``: theta 15, phi 60, a 0.8 x 0.2 tooth of eps 2.25 in a 1.6
  period, y the Axis1D; RCWA truncation 15, each wavelength at the run's own ``theta_at``).
  Measured at 20 cells per wavelength, 160 periods, NumPy, the worst over every propagating
  order and the three wavelengths:

      incidence   |r_s, r_p, t_s, t_p| amplitude   |R|, |T| per order   |R + T - 1|
      s           1.79e-2                          6.4e-3               1.5e-4
      p           3.45e-2                          5.7e-3               6.2e-4

  and at 40 cells per wavelength 7.3e-3 (s) and 1.38e-2 (p) in amplitude (slow test): first
  order in ``dx``, the cell-snapped staircase of the tooth (``fd_helpers.box_grating``), not
  the Axis1D. (Until the launch fix below: 1.96e-2 / 3.60e-2 at 20, 8.2e-3 / 1.46e-2 at 40.)
* **A Bloch phase on a one-cell user axis is refused, naming Axis1D** (the artificial
  dispersion above), and the spelling it suggests runs.
* **The CFL bound carries ``k_max^2/4``.** With two differenced axes at courant 0.5 the line is
  ``k_max dx = 2 sqrt(10) = 6.325``. The bound is the real stability line, not a convention: at
  ``k dx = 6.30`` a Nyquist-seeded field stays bounded over 600 steps; with ``k dx = 6.35`` on a
  face that understates ``k_max`` as 6.30 (the plan then accepts it) the same field grows by
  1.2e43 (measured). A traced ``k`` past ``k_max`` under ``jax.jit`` gives ``nan`` and
  ``diagnostics['stable'] == False`` instead; one inside gives the NumPy run's field.

**Mutant.** The algebraic derivative replaced by the one-cell Bloch difference (``exp(i k dx) -
1`` forward, ``1 - exp(-i k dx)`` backward, patched into ``curl._diff``): the recurrence residual
at the exact frequency appears and grows as ``k^2`` -- 4.0e-4, 1.6e-3, 7.1e-3 at k = 3, 6, 12
(measured), 0 at k = 0.

**A launch defect, fixed 2026-09-19** (``xfail(strict=True)`` until then): a conical
``PlaneWave`` (phi off 0 and 90 on a two-axis transverse plane) was launched with a p part. The
column's hard source wrote the continuum ``e_c`` into the stored ``E_x`` and ``E_y``, but each
sits half a cell along its own axis, so the injected tangential E was elliptical: in an empty
cell at theta 15, phi 60, 20 cells per wavelength, s incidence transmitted ``|t_p| = 9.1e-3``
where it must be 0. The drive now writes the grid's own wave -- stagger-removed amplitudes
transverse to the discrete ``K_par`` (here ``K_x = 2 sin(k_x dx/2)/dx`` and the exact ``k_y`` of
the Axis1D, which has no stagger), times ``exp(i k_x dx/2)`` on ``E_x``
(``src/wavebit/fdtd/_prepare/sources.py::plane_wave_drive``) -- and the leak is round-off: 2.4e-16 (s)
/ 2.8e-16 (p) at phi 60, 1.9e-16 / 3.6e-16 at phi 75, at 20 and 40 cells per wavelength alike,
with ``|t| = 1`` to 1e-7 (the 40-period run's settling). At phi 90 (E_s along x, where the
Axis1D ``k_y`` is the whole ``K_par``) it was and is exactly 0 -- the control.
"""

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._kernel import curl as kernel_curl
from tests.fixtures.fdtd.fd_helpers import backends, conical_grating, recurrence_residual, to_numpy

# ---------------------------------------------------------------------------------------------
# The exact dispersion
# ---------------------------------------------------------------------------------------------

N, L = 16, 1.0
DX = L / N
KX, KZ = 2 * math.pi / L, 4 * math.pi / L


def _K(k):
    return 2 * math.sin(k * DX / 2) / DX


def _plane_wave_scene(ky, steps=300):
    """A discrete plane wave ``exp(i (kx x + ky y + kz z))`` in a periodic x-z cell, y Axis1D.

    The polarisation is discretely divergence-free with the exact ``ky`` (no stagger along y):
    ``K_x A + ky B + K_z C = 0``.
    """
    A, B = 1.0, 0.7
    amps = (A, B, -(A * _K(KX) + B * ky) / _K(KZ))

    def comp(c):
        return lambda x, y, z: amps[c] * np.exp(1j * (KX * x + ky * y + KZ * z))
    return fd.Scene(domain=fd.Domain(size=(L, DX, L)),
                    boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Axis1D(k=ky),
                                             z=fd.Periodic()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=steps),
                    initial=fd.InitialField(E=tuple(comp(c) for c in range(3))),
                    monitors=(fd.Record('p', at=(0.1, 0.0, -0.05), component='Ex'),
                              fd.Record('q', at=(0.3, 0.0, 0.2), component='Hy')))


def _omega(ky_eff, dt):
    s = (_K(KX) ** 2 + ky_eff ** 2 + _K(KZ) ** 2) / 4
    return 2 / dt * math.asin(dt * math.sqrt(s))


def _residuals(res, ky, name='p'):
    """``(against the exact dispersion, against the one-cell Bloch one)``."""
    dt = res.plan.dt
    v = to_numpy(res[name].values)
    return (recurrence_residual(v, _omega(ky, dt), dt),
            recurrence_residual(v, _omega(_K(ky), dt), dt))


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('ky', [3.0, 9.0])
def test_a_plane_wave_along_axis1d_recurs_at_the_exact_dispersion(backend, ky):
    res = fd.run(_plane_wave_scene(ky), backend=backend)
    assert res.plan.ny == 1 and res.plan.dtype == 'complex128'
    for name in ('p', 'q'):
        exact, bloch = _residuals(res, ky, name)
        assert exact < 1e-13, (name, exact)                     # measured <= 5.6e-16
        assert bloch > 5e-6, (name, bloch)                      # 8.6e-6 at 3, 6.9e-4 at 9


def _one_cell_bloch_difference(original):
    """``curl._diff`` with the Axis1D product replaced by the one-cell Bloch difference."""
    def mutant(F, comp, axis, side, field, plan, params, psi, updates, rt, out):
        name = kernel_curl.AXIS_NAMES[axis]
        if plan.pad(name + side, comp).kind == 'axis1d':
            f = params['axis1d'][name]                          # i k dx
            return F[comp] * ((np.exp(f) - 1) if side == '+' else (1 - np.exp(-f)))
        return original(F, comp, axis, side, field, plan, params, psi, updates, rt, out)
    return mutant


def test_mutant_the_one_cell_bloch_difference_errs_and_the_error_grows_with_k(monkeypatch):
    monkeypatch.setattr(kernel_curl, '_diff', _one_cell_bloch_difference(kernel_curl._diff))
    errors = [_residuals(fd.run(_plane_wave_scene(ky)), ky)[0] for ky in (0.0, 3.0, 6.0, 12.0)]
    assert errors[0] < 1e-13                                    # k = 0: nothing to differ
    assert errors[1] > 1e-4                                     # measured 4.0e-4
    assert errors[1] < errors[2] < errors[3]                    # 1.6e-3, 7.1e-3
    assert errors[3] / errors[1] > 8                            # ~ k^2 (16x in k^2)


# ---------------------------------------------------------------------------------------------
# The conical grating against wavebit.rcwa
# ---------------------------------------------------------------------------------------------

#: The worst ``|fd - rcwa|`` allowed at 20 cells per wavelength, (amplitude, efficiency), with
#: the measured value it sits above. The amplitude lines sit ~3% above the measured values and
#: below what the pre-fix PlaneWave drive measured, so the old conical-launch leak fails here
#: too. That is the most this gate can say about the launch: the error it reads is the
#: tooth's first-order staircase, and a leak much smaller than the old one hides under it.
#: The launch itself is guarded by the leak tests, which read the cross-polarised amplitude
#: directly: :func:`test_an_empty_conical_cell_transmits_no_p`,
#: ``test_fd_diffraction.py::test_a_conical_plane_wave_is_launched_without_the_other_polarisation``
#: and ``test_fd_stage3_backends.py::test_a_layered_cell_carries_no_cross_polarisation_on_any_backend``.
CONICAL_TOL = {'s': (1.85e-2, 1e-2),       # measured 1.79e-2, 6.4e-3 (1.96e-2 before the fix)
               'p': (3.55e-2, 1e-2)}       # measured 3.45e-2, 5.7e-3 (3.60e-2 before the fix)


#: RCWA's truncation for the conical twin, and the one it is checked against.
CONICAL_TRUNCATION, CONICAL_TRUNCATION_CHECK = 15, 30


def _conical_errors(polarization, cells_per_wavelength, truncation_gap=None):
    """``(worst amplitude error, worst efficiency error, worst |R + T - 1|, res, scene)``.

    With ``truncation_gap`` a list, the worst ``|rcwa(15) - rcwa(30)|`` over the same values is
    appended to it: the oracle's own truncation, measured where it is used.
    """
    scene, info = conical_grating(polarization=polarization,
                                  cells_per_wavelength=cells_per_wavelength)
    with warnings.catch_warnings():
        # The grating rings at its in-band Rayleigh wavelengths (see test_fd_decay.py's xfail):
        # the reflect monitor reads 7.2e-5 (s) / 1.7e-3 (p), and the orders move 8.9e-4 /
        # 3.3e-3 between 160 and 480 periods -- inside the tolerances below.
        warnings.simplefilter('ignore', fd.UnsettledFieldWarning)
        # The tooth's glass is 8.6 cells per wavelength at the band's short edge (20 at 1.0).
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        res = fd.run(scene)
    r, t = res['r'], res['t']
    theta_at = to_numpy(r.theta_at)
    amp = eff = closure = 0.0
    compared = 0
    for i, wavelength in enumerate(scene.monitors[0].wavelengths):
        ref = info['rcwa'](i, theta_at[i], truncation_order=CONICAL_TRUNCATION)
        check = (None if truncation_gap is None else
                 info['rcwa'](i, theta_at[i], truncation_order=CONICAL_TRUNCATION_CHECK))
        m_max = r.order_window[0][1]
        for m in range(-m_max, m_max + 1):
            for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
                got = complex(to_numpy(getattr((r if name[0] in 'rR' else t).orders.order(m, 0),
                                               name))[i])
                want = complex(getattr(ref.order(m, 0), name))
                if check is not None:
                    gap = abs(complex(getattr(check.order(m, 0), name)) - want)
                    truncation_gap.append(gap)
                if name in ('R', 'T'):
                    eff = max(eff, abs(got - want))
                else:
                    amp = max(amp, abs(got - want))
                compared += 1
        total = (float(np.sum(to_numpy(r.orders.at(i).R)).real)
                 + float(np.sum(to_numpy(t.orders.at(i).T)).real))
        closure = max(closure, abs(total - 1.0))
    assert compared == 3 * 3 * 6                    # three wavelengths, orders -1..1, six values
    return amp, eff, closure, res, scene


@pytest.mark.parametrize('polarization', ['s', 'p'])
def test_a_conical_lamellar_grating_matches_rcwa_at_matched_geometry(polarization):
    gaps = []
    amp, eff, closure, res, scene = _conical_errors(polarization, 20, truncation_gap=gaps)
    tol_amp, tol_eff = CONICAL_TOL[polarization]
    assert amp < tol_amp, amp
    assert eff < tol_eff, eff
    # The oracle is converged far below the tolerance it judges (measured 6.7e-5 / 6.3e-5).
    assert max(gaps) < tol_amp / 20, max(gaps)
    # The run is not settled (the in-band Rayleigh ringing, test_fd_decay.py's xfail); what is
    # left is held at a tenth of the amplitude tolerance so it cannot be what the gate reads.
    ratios = res.diagnostics['decay']
    assert max(ratios['r'], ratios['t']) < tol_amp / 10, ratios
    assert closure < 1e-3, closure                              # measured 1.5e-4 / 6.2e-4
    # Fixed k: the angle moves across the band as sin(theta) = wavelength * sin(15 deg).
    expected = np.degrees(np.arcsin(np.asarray(scene.monitors[0].wavelengths)
                                    * math.sin(math.radians(15.0))))
    assert np.max(np.abs(to_numpy(res['r'].theta_at) - expected)) < 1e-9
    assert res.plan.ny == 1
    assert isinstance(scene.boundaries.face('y-'), fd.Axis1D)


@pytest.mark.slow
@pytest.mark.parametrize('polarization', ['s', 'p'])
def test_the_conical_grating_converges_toward_rcwa_with_resolution(polarization):
    """Measured: 1.79e-2 -> 7.3e-3 (s), 3.45e-2 -> 1.38e-2 (p) from 20 to 40 cells/wavelength
    (1.96e-2 -> 8.2e-3 and 3.60e-2 -> 1.46e-2 while the launch carried its first-order p part)."""
    coarse = _conical_errors(polarization, 20)[0]
    fine = _conical_errors(polarization, 40)[0]
    assert fine < coarse / 1.8, (coarse, fine)


# ---------------------------------------------------------------------------------------------
# A Bloch phase on a one-cell axis
# ---------------------------------------------------------------------------------------------

def test_a_bloch_phase_on_a_one_cell_user_axis_is_refused_naming_axis1d():
    scene, _ = conical_grating(periods=1)
    pw = scene.sources[0]
    bloch = scene.replace(boundaries=fd.Boundaries(**fd.Bloch.from_source(pw),
                                                   z=scene.boundaries.face('z-')))
    assert isinstance(bloch.boundaries.face('y-'), fd.Bloch)
    with pytest.raises(ValueError, match='fd.Axis1D') as e:
        fd.plan(bloch)
    assert "fd.Bloch.from_source(pw, axis1d='y')" in str(e.value)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = fd.run(scene)                                     # the spelling it names runs
    assert res.diagnostics['stable'] is True


# ---------------------------------------------------------------------------------------------
# The CFL bound with k_max
# ---------------------------------------------------------------------------------------------

CFL_N = 8
CFL_DX = 1.0 / CFL_N
#: ``k_max dx`` on the stability line at courant 0.5 with two differenced axes: 2 sqrt(10).
LINE = 2 * math.sqrt(10.0)


def _nyquist(x, y, z):
    """The checkerboard in x and z (``K = 2/dx``, the fastest differenced mode) and a smooth part."""
    return np.exp(1j * np.pi * (x + z) / CFL_DX) + 0.3 * np.cos(2 * np.pi * x) + 0.0 * y


def _cfl_scene(face, steps=600):
    return fd.Scene(domain=fd.Domain(size=(1.0, CFL_DX, 1.0)),
                    boundaries=fd.Boundaries(x=fd.Periodic(), y=face, z=fd.Periodic()),
                    resolution=fd.Resolution(dx=CFL_DX), run=fd.Run(steps=steps),
                    initial=fd.InitialField(E=(None, _nyquist, None), H=(_nyquist, None, None)),
                    monitors=(fd.Record('p', at=(0.1, 0.0, 0.1), component='Ey'),))


def _growth(res):
    v = np.abs(to_numpy(res['p'].values))
    return float(v[-50:].max() / v[:50].max())


def test_the_cfl_line_with_k_max_is_the_stability_line():
    assert abs(LINE - 6.3246) < 1e-4
    with pytest.raises(ValueError, match='k_max'):
        fd.plan(_cfl_scene(fd.Axis1D(k=6.35 / CFL_DX)))           # just past the line
    inside = fd.run(_cfl_scene(fd.Axis1D(k=6.30 / CFL_DX)))      # just inside it
    assert _growth(inside) < 2.0
    assert np.all(np.isfinite(to_numpy(inside.final.E)))
    # The same field past the line, on a face that understates k_max (a lie the plan cannot
    # see): the bound is what kept the run bounded.
    liar = fd.Axis1D(k=6.30 / CFL_DX)
    object.__setattr__(liar, 'k', 6.35 / CFL_DX)
    with np.errstate(over='ignore', invalid='ignore'):
        past = fd.run(_cfl_scene(liar))
    v = np.abs(to_numpy(past['p'].values))
    assert not np.all(np.isfinite(v)) or v[-1] > 1e30 * v[0]     # measured 1.2e43


def test_a_traced_k_past_k_max_is_nan_and_unstable_and_inside_it_is_the_numpy_run():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit_enable_x64()

    def f(k):
        res = fd.run(_cfl_scene(fd.Axis1D(k=k, k_max=6.30 / CFL_DX), steps=200), backend='jax')
        return res['p'].values, res.diagnostics['stable']
    g = jax.jit(f)
    values, stable = g(jnp.asarray(6.30 / CFL_DX, dtype=jnp.float64))
    assert bool(stable)
    want = to_numpy(fd.run(_cfl_scene(fd.Axis1D(k=6.30 / CFL_DX), steps=200))['p'].values)
    assert np.max(np.abs(np.asarray(values) - want)) < 1e-12 * np.max(np.abs(want))
    values, stable = g(jnp.asarray(6.35 / CFL_DX, dtype=jnp.float64))
    assert not bool(stable)
    assert np.all(np.isnan(np.asarray(values)))


# ---------------------------------------------------------------------------------------------
# A kernel defect: the conical PlaneWave's launched p part
# ---------------------------------------------------------------------------------------------

def _empty_conical_t_p(phi):
    """``|t_p|`` of the zero order, s incidence, in an empty cell (the tooth is vacuum)."""
    scene, _ = conical_grating(fill=1.0, eps=1.0, periods=40, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = fd.run(scene)
    order = res['t'].orders.order(0, 0)
    assert np.all(np.abs(np.abs(to_numpy(order.t_s)) - 1.0) < 1e-6)
    return float(np.max(np.abs(to_numpy(order.t_p))))


def test_an_empty_cell_at_phi_90_transmits_no_p():
    assert _empty_conical_t_p(90.0) < 1e-12                     # measured 0.0


@pytest.mark.parametrize('phi', [60.0, 75.0])
def test_an_empty_conical_cell_transmits_no_p(phi):
    """Measured 2.4e-16 at phi 60 and 1.9e-16 at 75; 9.1e-3 at 60 before the drive carried
    the stagger phase and the discrete ``K_par`` direction."""
    assert _empty_conical_t_p(phi) < 1e-13
