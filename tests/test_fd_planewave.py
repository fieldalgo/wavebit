"""Gate 3 of stage 2 -- the PlaneWave: a one-directional Huygens sheet in a periodic cell.

The sheet is the one face of a TFSF box that crosses the propagation axis, with the transverse
box unbounded: an electric current on the E node plane ``k0`` (``before_E``) and a magnetic one
on the H half plane half a cell upstream (``before_H``), both read from a one-cell transverse
Bloch column of the same z grid (``src/wavebit/fd/_kernel/sources.py``). What is pinned here, each
against something the engine did not compute itself:

* **The backward wave is at round-off.** In a uniform Periodic cell (theta 0) and in Bloch cells
  (theta 20 and 40, s and p, and one wave off both transverse axes), PML on z, both
  directions, and in glass: the peak of every component upstream of the sheet over the whole
  run is <= 1e-13 of the peak downstream (measured 0 on the real theta-0 run, 5e-16 to 7e-16
  on the Bloch runs). **Why round-off and not the discretisation error:** the J/M ratio is the
  column's own discrete impedance at every frequency at once. The continuum's ratio would not
  do it -- with the J current scaled by rho = (continuum ratio)/(discrete ratio) at the band
  centre (``rho_continuum``, below), the backward wave is |rho - 1|/|rho + 1| of the forward
  one: **measured 9.3e-6 at 20 degrees and 4.8e-4 at 40 degrees** (s and p alike, at 20 cells
  per wavelength; predicted 9.2e-6 and 4.77e-4), and exactly 0 at theta 0, where the 1-D Yee
  impedance is exactly the continuum's (``Z0 H = -E`` at the staggered points).
* **The forward field is the closed form at every frequency in the band.** The DFT of E at a
  probe plane downstream equals ``e_t A W(w) exp(i kz(w) d)``, ``e_t`` the stored tangential E
  of the grid's own s or p wave (transverse to the discrete ``K_par``, each component with its
  half-cell stagger phase ``exp(i k_c dx/2)``, written out in ``_grid_polarisation``; the
  continuum ``e`` missed by 4e-2 to 6e-2 off normal incidence): ``W`` the DFT of the waveform
  written out here from ``fd.Band``'s definition at the hard source's times ``(n + 1) dt``, ``d``
  the distance from the column's hard-source plane to the probe, and ``kz`` the **discrete**
  propagation constant of the fixed-k Bloch wave, ``(2/dx) sin(kz dx/2) = sqrt(W^2 - Kx^2 -
  Ky^2)`` with ``W = (2/dt) sin(w dt/2)`` and ``K = (2/dx) sin(k_par dx/2)``. At real
  frequencies (theta 0 and 20, a 24-cell PML) it holds to 2e-6 of the peak over ``f0 +- k df``
  (measured <= 6e-7; the 12-cell default layer's reflection puts 2e-4 there instead, which is
  why the layer is 24 cells here). At 40 degrees the band crosses the fixed-k cutoff
  ``f = sin(40) f0 = 0.64 f0``, where the group velocity is zero and the field never leaves:
  a real-frequency DFT over any window leaks that ring-down into the band (1e-3 at 1100 steps,
  still 3e-5 at 4000). So every case is also checked on the contour ``w + i gamma``
  (``gamma = 0.4``, a Laplace transform: the linear relation holds there exactly and the
  window's truncation is damped by ``exp(-gamma T)``), where it holds to 3e-6 over the whole
  band, the evanescent part below the cutoff included (measured <= 5.2e-7).
* **The reference series is the grid's incident field.** In an empty cell the full grid's
  total field at cell (0, 0) of the entry plane -- the tangential E on the node plane, the
  normal E on the half cell inside, every H at that half cell's index -- equals
  ``res.sources['pw'].incident`` / ``.incident_H`` to 1e-13 (measured <= 1e-15), every step.
* **``Bloch.from_source`` is the fixed-k phase** ``k_par = n (2 pi/lambda0) sin(theta) (cos
  phi, sin phi)`` -- the same for every frequency of the band -- and returns ``Periodic`` on
  both transverse axes for a concrete theta = 0 (a real run). In the run, the field of one
  cell is the field of its neighbour times ``exp(+i k dx)`` at every step (``e^{-i w t}``: a
  positive phase is a wave toward +x).
* **s and p are transverse, and E x H points along k.** ``PlaneWave.vectors()`` is checked for
  the identities; the run's own incident field at the band centre (phases of the staggered
  points removed) is checked for the discrete ones: ``K . E = 0`` and ``W H = K x E`` (so
  ``Re(E x H*)`` is along ``K``), s with E alone on ``z_hat x K_hat_par`` and p with H alone
  there, at every ``phi`` (the conical case included: the grid's s and p, not the continuum's,
  are what the sheet launches). At real
  frequency the energy flow is along ``K`` to 1e-5 and within 5e-5 (measured) of the
  continuum ``k`` at 20 degrees -- the discrete kz is not the continuum one.

* **A reflecting face along the axis is scattered field, not incident.** In a Periodic cell
  backed by a PEC face, the mirror's reflection crosses the sheet: the peak above the sheet is
  ``|r| = 1`` of the incident peak (measured 1.009, the pulse's own dispersion), and the
  column's reference series decays to the PML floor after the pulse (measured 1.3e-6 of its
  peak). The column carries a PML of its own on padded cells where the scene's face does not
  absorb (``column_pads``); before it did, the column copied the PEC, the hard source and the
  wall made it a lossless cavity, the mirror's reflection was counted as incident and
  cancelled by the sheet -- 0.0 above the sheet, and a reference ringing at 0.60 forever.

**Mutants.** (1) M on the wrong half plane (the ``before_H`` slabs moved one cell across the E
plane): the backward wave is 0.12 to 0.16 of the forward one (a half-plane of phase error,
``~ sin(kz dx/2)``, at 20 cells per wavelength) -- the gate's 1e-13 is 1e12 below it. (2) The
impedance ratio taken from the continuum (above): 9.3e-6 / 4.8e-4. (3) The Bloch phase sign
flipped: ``from_source`` with ``-k`` is refused at plan by the consistency check; the phase
conjugated on the main grid, on the sheet's transverse pattern, or in the column breaks the
reference-equals-grid gate at O(1) (0.2 to 2). The backward-wave gate alone would not see the
column flip -- the tangential impedance depends on ``K_par^2`` only -- which is why both gates
exist.

Every run is NumPy (the three-backend comparison and the gradients are gate 6), one to four
thousand steps on 5 x 4 x ~100 cells, under a second each.
"""
import dataclasses
import functools
import math

import numpy as np
import pytest

import wavebit.fd as fd
from fd_helpers import bloch_plane_wave, peak_of, ring_monitors, run_mutated, to_numpy

#: Backward over forward peak, and the reference against the grid: round-off.
ROUND_OFF = 1e-13

#: The closed form at real frequencies (24-cell PML, 1100 steps): measured <= 6e-7.
CLOSED_TOL = 2e-6

#: The closed form on the damped contour w + i GAMMA (1600 steps): measured <= 5.2e-7.
DAMPED_TOL = 3e-6
GAMMA = 0.4

#: The discrete plane-wave relations of the incident field on the damped contour: <= 1.4e-6.
RELATION_TOL = 1e-5

#: Amplitude, phase and delay that are not the defaults, so the closed form pins all three.
A, PHASE, DELAY = 1.3, 0.4, 0.2

#: (theta, polarization, phi): the cases every physics gate runs.
CASES = [(0.0, 's', 0.0), (20.0, 's', 0.0), (20.0, 'p', 0.0), (40.0, 's', 0.0),
         (40.0, 'p', 0.0), (30.0, 'p', 35.0)]


def _id(case):
    theta, pol, phi = case
    return f'theta{theta:g}-{pol}' + (f'-phi{phi:g}' if phi else '')


# ---------------------------------------------------------------------------------------------
# Scenes and readers
# ---------------------------------------------------------------------------------------------

def _scene(theta, pol, phi=0.0, direction='-z', monitors=None, **kw):
    """``bloch_plane_wave`` with ``monitors(scene, pw, plan)`` built on the scene's own grid."""
    scene, pw = bloch_plane_wave(theta=theta, polarization=pol, phi=phi, direction=direction,
                                 **kw)
    if monitors is not None:
        scene, pw = bloch_plane_wave(theta=theta, polarization=pol, phi=phi,
                                     direction=direction,
                                     monitors=monitors(scene, pw, fd.plan(scene)), **kw)
    return scene, pw


def _cell0(plan, F, c, index):
    """The ``(x, y, z)`` of component ``F c`` at cell (0, 0) across and node ``index`` along z."""
    xs, ys, zs = plan.grid.yee_points(F, c)
    return (float(xs[0]), float(ys[0]), float(zs[index]))


def _entry_monitors(prefix='r'):
    """Records of all six components at cell (0, 0) where the reference series is read."""
    def make(scene, pw, plan):
        term = plan.terms[0]
        out = []
        for F in 'EH':
            for c in range(3):
                index = term.entry_E if (F == 'E' and c != 2) else term.entry_H
                out.append(fd.Record(f'{prefix}:{F}{"xyz"[c]}', at=_cell0(plan, F, c, index),
                                     component=f'{F}{"xyz"[c]}'))
        return tuple(out)
    return make


def _probe_monitors(offset):
    """Tangential E at cell (0, 0) on the node plane ``offset`` downstream of the sheet."""
    def make(scene, pw, plan):
        index = int(round((pw.position + pw.sign * offset - plan.grid.low[2]) / plan.dx))
        return tuple(fd.Record(f'f:E{"xyz"[c]}', at=_cell0(plan, 'E', c, index),
                               component=f'E{"xyz"[c]}') for c in (0, 1))
    return make


def _both_sides(distance=0.6):
    """Six components at two transverse points, ``distance`` upstream (b) and downstream (f)."""
    def make(scene, pw, plan):
        z_back = pw.position - pw.sign * distance
        z_fwd = pw.position + pw.sign * distance
        return (ring_monitors([(0.0, 0.0, z_back), (0.1, 0.05, z_back)], 'b')
                + ring_monitors([(0.0, 0.0, z_fwd), (0.1, 0.05, z_fwd)], 'f'))
    return make


def _backward_ratio(res):
    return peak_of(res, 'b') / peak_of(res, 'f')


def _with_corrections(plan, change):
    """``plan`` with its PlaneWave term's corrections passed through ``change(correction)``."""
    term = plan.terms[0]
    term = dataclasses.replace(term, corrections=tuple(change(c) for c in term.corrections))
    return dataclasses.replace(plan, terms=(term,) + plan.terms[1:])


def _dft(t, values, w):
    """``sum_n f(t_n) exp(+i w t_n)`` for each ``w`` (complex allowed): the e^{-iwt} amplitude."""
    f = np.asarray(to_numpy(values))
    return np.exp(1j * np.outer(w, np.asarray(t))) @ f


def _waveform(pw, t):
    """``fd.Band``'s waveform written out from its definition, independently of the engine."""
    b = pw.band
    tau = 1.0 / (2.0 * math.pi * b.df)
    t0 = math.sqrt(2.0 * math.log(1.0 / b.cutoff)) * tau
    s = np.asarray(t) - t0 - pw.delay
    return pw.amplitude * np.exp(-0.5 * (s / tau) ** 2) * np.sin(
        2.0 * math.pi / b.wavelength * s + pw.phase)


def _k_par(pw):
    k, _, _ = pw.vectors()
    return tuple(2.0 * math.pi / pw.band.wavelength * k[a] for a in (0, 1))


def _discrete_kz(pw, plan, w):
    """The fixed-k Bloch wave's discrete propagation constant at (complex) ``w``, outgoing."""
    dx, dt = plan.dx, plan.dt
    Om = 2.0 * np.sin(np.asarray(w) * dt / 2.0) / dt
    K2 = sum((2.0 * math.sin(k * dx / 2.0) / dx) ** 2 for k in _k_par(pw))
    kz = (2.0 / dx) * np.arcsin((dx / 2.0) * np.sqrt(Om ** 2 - K2 + 0j))
    return np.where(kz.imag < 0, -kz, np.where((kz.imag == 0) & (kz.real < 0), -kz, kz))


def rho_continuum(theta, pol, dx, dt, wavelength=1.0):
    """(continuum H/E ratio) / (discrete H/E ratio) of the tangential fields at the band centre.

    s: ``Z0 Hx = -(kz/w) Ey`` in the continuum, ``-(Kz/W) Ey`` on the grid (Faraday with the
    forward difference); p: ``Z0 Hy = (w/kz) Ex`` against ``(W/Kz) Ex`` (Ampere). ``k_par`` is the
    Bloch k, the same on both; the continuum ``kz = sqrt(w^2 - k_par^2)`` and the grid's
    ``Kz = sqrt(W^2 - K_par^2)``.
    """
    w = 2.0 * math.pi / wavelength
    k_par = w * math.sin(math.radians(theta))
    kz = math.sqrt(w * w - k_par * k_par)
    Om = 2.0 * math.sin(w * dt / 2.0) / dt
    K = 2.0 * math.sin(k_par * dx / 2.0) / dx
    Kz = math.sqrt(Om * Om - K * K)
    r = (kz / w) / (Kz / Om)
    return r if pol == 's' else 1.0 / r


@functools.lru_cache(maxsize=None)
def _backward_run(theta, pol, phi, direction, glass=False):
    kw = {}
    if glass:
        kw = dict(background=fd.Material(eps=2.25), cells_per_wavelength=30)
    scene, pw = _scene(theta, pol, phi, direction, monitors=_both_sides(), steps=900, **kw)
    plan = fd.plan(scene)
    return scene, pw, plan, run_mutated(scene, plan)


@functools.lru_cache(maxsize=None)
def _reference_run(theta, pol, phi, direction='-z'):
    def monitors(scene, pw, plan):
        return _entry_monitors()(scene, pw, plan) + _both_sides()(scene, pw, plan)
    scene, pw = _scene(theta, pol, phi, direction, monitors=monitors, steps=900,
                       amplitude=A, phase=PHASE, delay=DELAY)
    return scene, pw, fd.plan(scene)


@functools.lru_cache(maxsize=None)
def _long_run(theta, pol, phi):
    """1600 steps behind a 24-cell PML: the incident field at the band centre, twice read."""
    scene, pw = _scene(theta, pol, phi, '-z', steps=1600, pml=24)
    plan = fd.plan(scene)
    return pw, plan, run_mutated(scene, plan)


def _reference_error(res):
    """max over steps and components |grid - reference| / peak reference."""
    src = res.sources['pw']
    inc, inc_H = to_numpy(src.incident), to_numpy(src.incident_H)
    peak = max(np.abs(inc).max(), np.abs(inc_H).max())
    err = 0.0
    for c in range(3):
        err = max(err, np.abs(to_numpy(res[f'r:E{"xyz"[c]}'].values) - inc[:, c]).max(),
                  np.abs(to_numpy(res[f'r:H{"xyz"[c]}'].values) - inc_H[:, c]).max())
    return err / peak


# ---------------------------------------------------------------------------------------------
# The backward wave
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('direction', ['-z', '+z'])
@pytest.mark.parametrize('case', CASES, ids=_id)
def test_backward_wave_is_at_round_off(case, direction):
    theta, pol, phi = case
    _, _, plan, res = _backward_run(theta, pol, phi, direction)
    assert plan.dtype == ('float64' if theta == 0.0 else 'complex128')
    forward = peak_of(res, 'f')
    assert forward > 0.5, f'the forward wave did not arrive (peak {forward:.3g})'
    assert _backward_ratio(res) <= ROUND_OFF


@pytest.mark.parametrize('theta', [0.0, 20.0])
def test_backward_wave_is_at_round_off_in_glass(theta):
    """In a background of index 1.5, with ``from_source(pw, background=glass)``."""
    _, _, plan, res = _backward_run(theta, 's', 0.0, '-z', glass=True)
    assert peak_of(res, 'f') > 0.3
    assert _backward_ratio(res) <= ROUND_OFF


# ---------------------------------------------------------------------------------------------
# The forward field against the closed form
# ---------------------------------------------------------------------------------------------

def _grid_polarisation(pw, plan):
    """The stored tangential E of the grid's own s or p wave at cell (0, 0), written out here.

    With each component referred to its own Yee point the grid's plane wave is the
    continuum's with ``k -> K = (2/dx) sin(k dx/2)``, so s is ``z_hat x K_hat_par`` and p's
    tangential part is along ``K_hat_par`` (its length, ``sigma cos(theta)``, is the drive's);
    stored at cell (0, 0), ``E_x`` sits half a cell along x and carries ``exp(i k_x dx/2)``,
    ``E_y`` ``exp(i k_y dx/2)``. At ``K_par = 0`` it is ``pw.vectors()``'s ``e``.
    """
    kx, ky = _k_par(pw)
    dx = plan.dx
    K = np.array([2.0 * math.sin(kx * dx / 2.0) / dx, 2.0 * math.sin(ky * dx / 2.0) / dx])
    _, e, _ = pw.vectors()
    if np.linalg.norm(K) == 0.0:
        return np.array([e[0], e[1]], dtype=complex)
    q = K / np.linalg.norm(K)
    if pw.polarization == 's':
        t = np.array([-q[1], q[0]])
    else:
        t = pw.sign * math.cos(math.radians(pw.theta)) * q
    return t * np.exp(0.5j * dx * np.array([kx, ky]))


def _closed_form_error(theta, pol, phi, direction, steps, gamma, offset=0.8):
    scene, pw = _scene(theta, pol, phi, direction, monitors=_probe_monitors(offset),
                       steps=steps, pml=24, amplitude=A, phase=PHASE, delay=DELAY)
    plan = fd.plan(scene)
    res = run_mutated(scene, plan)
    term = plan.terms[0]
    b = pw.band
    k = math.sqrt(2.0 * math.log(1.0 / b.cutoff))
    f0 = 1.0 / b.wavelength
    w = 2.0 * math.pi * np.linspace(f0 - k * b.df, f0 + k * b.df, 41) + 1j * gamma
    t = (np.arange(plan.steps) + 1.0) * plan.dt          # the hard source's and E's times
    W = _dft(t, _waveform(pw, t), w)
    probe = int(round((pw.position + pw.sign * offset - plan.grid.low[2]) / plan.dx))
    distance = abs(probe - term.ks) * plan.dx
    e = _grid_polarisation(pw, plan)
    worst = 0.0
    for c in (0, 1):
        want = e[c] * W * np.exp(1j * _discrete_kz(pw, plan, w) * distance)
        got = _dft(res[f'f:E{"xyz"[c]}'].t, res[f'f:E{"xyz"[c]}'].values, w)
        assert np.all(np.isfinite(got))
        scale = max(np.abs(e[0] * W).max(), np.abs(e[1] * W).max())
        worst = max(worst, float(np.max(np.abs(got - want)) / scale))
    return worst


@pytest.mark.parametrize('direction', ['-z', '+z'])
@pytest.mark.parametrize('case', [(0.0, 's', 0.0), (20.0, 's', 0.0), (20.0, 'p', 0.0)],
                         ids=_id)
def test_forward_field_is_the_closed_form_at_real_frequencies(case, direction):
    assert _closed_form_error(*case, direction, steps=1100, gamma=0.0) <= CLOSED_TOL


@pytest.mark.parametrize('case', CASES, ids=_id)
def test_forward_field_is_the_closed_form_on_the_damped_contour(case):
    """Every case, the grazing and evanescent part of the 40-degree band included."""
    assert _closed_form_error(*case, '-z', steps=1600, gamma=GAMMA) <= DAMPED_TOL


def test_closed_form_sees_a_wrong_propagation_constant():
    """The comparison is not blind to kz: the continuum kz instead misses by far more."""
    scene, pw = _scene(20.0, 's', 0.0, '-z', monitors=_probe_monitors(0.8), steps=1100, pml=24)
    plan = fd.plan(scene)
    res = run_mutated(scene, plan)
    w = np.array([2.0 * math.pi])
    t = (np.arange(plan.steps) + 1.0) * plan.dt
    W = _dft(t, _waveform(pw, t), w)
    probe = int(round((pw.position - 0.8 - plan.grid.low[2]) / plan.dx))
    d = abs(probe - plan.terms[0].ks) * plan.dx
    got = _dft(res['f:Ey'].t, res['f:Ey'].values, w)
    kz_continuum = 2.0 * math.pi * math.cos(math.radians(20.0))
    discrete = abs(got - W * np.exp(1j * _discrete_kz(pw, plan, w) * d))[0] / abs(W[0])
    continuum = abs(got - W * np.exp(1j * kz_continuum * d))[0] / abs(W[0])
    assert discrete <= CLOSED_TOL
    assert continuum > 1e3 * CLOSED_TOL


# ---------------------------------------------------------------------------------------------
# The reference series
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('case', CASES, ids=_id)
def test_reference_equals_the_grid_incident_field(case):
    scene, pw, plan = _reference_run(*case)
    res = run_mutated(scene, plan)
    src = res.sources['pw']
    assert src.kind == 'PlaneWave'
    assert np.asarray(to_numpy(src.incident)).shape == (plan.steps, 3)
    assert src.plane == pytest.approx(plan.grid.low[2] + plan.terms[0].entry_E * plan.dx)
    assert np.max(np.abs(to_numpy(src.incident))) > 0.5
    assert _reference_error(res) <= ROUND_OFF


def test_reference_equals_the_grid_going_up():
    scene, pw = _scene(40.0, 'p', 0.0, '+z', monitors=_entry_monitors(), steps=900,
                       amplitude=A, phase=PHASE, delay=DELAY)
    res = run_mutated(scene)
    assert _reference_error(res) <= ROUND_OFF


# ---------------------------------------------------------------------------------------------
# Bloch.from_source
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('direction', ['-z', '+z', '-x', '+y'])
def test_from_source_is_periodic_at_normal_incidence(direction):
    axis = direction[1]
    pw = fd.PlaneWave('pw', direction=direction, band=fd.Band(wavelength=1.0, df=0.15),
                      **{axis: 0.0})
    faces = fd.Bloch.from_source(pw)
    assert set(faces) == set('xyz') - {axis}
    assert all(isinstance(f, fd.Periodic) for f in faces.values())


@pytest.mark.parametrize('theta,phi,eps', [(20.0, 0.0, 1.0), (40.0, 0.0, 1.0),
                                           (30.0, 35.0, 1.0), (20.0, 0.0, 2.25),
                                           (30.0, 90.0, 1.0)])
def test_from_source_is_the_fixed_k_phase(theta, phi, eps):
    wavelength = 0.8
    pw = fd.PlaneWave('pw', z=0.0, direction='-z', theta=theta, phi=phi,
                      band=fd.Band(wavelength=wavelength, df=0.1))
    faces = fd.Bloch.from_source(pw, background=None if eps == 1.0 else fd.Material(eps=eps))
    k_par = math.sqrt(eps) * 2.0 * math.pi / wavelength * math.sin(math.radians(theta))
    want = {'x': k_par * math.cos(math.radians(phi)), 'y': k_par * math.sin(math.radians(phi))}
    for axis, k in want.items():
        if abs(k) < 1e-9:
            assert isinstance(faces[axis], fd.Periodic)
        else:
            assert isinstance(faces[axis], fd.Bloch)
            assert faces[axis].k == pytest.approx(k, rel=1e-14)
    # Fixed k: the band's width does not enter -- every frequency shares the one k_par.
    wide = fd.Bloch.from_source(pw.replace(band=fd.Band(wavelength=wavelength, df=0.2)),
                                background=None if eps == 1.0 else fd.Material(eps=eps))
    assert wide == faces


@pytest.mark.parametrize('case', [(20.0, 's', 0.0), (30.0, 'p', 35.0)], ids=_id)
def test_the_grid_carries_the_fixed_k_phase_cell_to_cell(case):
    """``F(cell + 1) = exp(+i k dx) F(cell)`` along each transverse axis, mid-pulse, everywhere."""
    theta, pol, phi = case
    scene, pw = _scene(theta, pol, phi, '-z', steps=350)
    plan = fd.plan(scene)
    res = run_mutated(scene, plan)
    kx, ky = _k_par(pw)
    for F in (to_numpy(res.final.E), to_numpy(res.final.H)):
        assert np.max(np.abs(F)) > 0.1
        for array_axis, k in ((3, kx), (2, ky)):
            lead = np.take(F, range(1, F.shape[array_axis]), axis=array_axis)
            trail = np.take(F, range(0, F.shape[array_axis] - 1), axis=array_axis)
            err = np.max(np.abs(lead - np.exp(1j * k * plan.dx) * trail))
            assert err <= 1e-13 * np.max(np.abs(F))


# ---------------------------------------------------------------------------------------------
# s and p: transverse to k, E x H along k
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('direction', ['-z', '+z', '-x', '+y'])
@pytest.mark.parametrize('case', CASES, ids=_id)
def test_vectors_are_transverse_and_right_handed(case, direction):
    theta, pol, phi = case
    axis = 'xyz'.index(direction[1])
    pw = fd.PlaneWave('pw', direction=direction, theta=theta, phi=phi, polarization=pol,
                      band=fd.Band(wavelength=1.0, df=0.15), **{direction[1]: 0.0})
    k, e, h = (np.array(v) for v in pw.vectors())
    for v in (k, e, h):
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-15)
    assert abs(k @ e) <= 1e-15 and abs(k @ h) <= 1e-15 and abs(e @ h) <= 1e-15
    np.testing.assert_allclose(np.cross(e, h), k, atol=1e-15)
    assert np.sign(k[axis]) == (1 if direction[0] == '+' else -1)
    assert k[axis] == pytest.approx((1 if direction[0] == '+' else -1)
                                    * math.cos(math.radians(theta)), abs=1e-15)
    normal = np.cross(k, np.eye(3)[axis])        # normal to the plane of incidence
    if theta > 0:
        normal /= np.linalg.norm(normal)
        if pol == 's':
            assert abs(abs(e @ normal) - 1.0) <= 1e-15
        else:
            assert abs(e @ normal) <= 1e-15 and abs(abs(h @ normal) - 1.0) <= 1e-15


def _incident_amplitudes(res, pw, plan, w):
    """E and H of the reference at ``w``, each with its own Yee point's plane-wave phase removed."""
    src, term = res.sources['pw'], plan.terms[0]
    kx, ky = _k_par(pw)
    kz = pw.sign * complex(_discrete_kz(pw, plan, np.array([w]))[0])
    E, H = np.zeros(3, complex), np.zeros(3, complex)
    for F, values, t, out in (('E', src.incident, src.t, E), ('H', src.incident_H, src.t_H, H)):
        for c in range(3):
            index = term.entry_E if (F == 'E' and c != 2) else term.entry_H
            x, y, z = _cell0(plan, F, c, index)
            out[c] = (_dft(t, to_numpy(values)[:, c], np.array([w]))[0]
                      * np.exp(-1j * (kx * x + ky * y + kz * z)))
    Om = 2.0 * np.sin(w * plan.dt / 2.0) / plan.dt
    K = np.array([2.0 * math.sin(kx * plan.dx / 2.0) / plan.dx,
                  2.0 * math.sin(ky * plan.dx / 2.0) / plan.dx,
                  2.0 * np.sin(kz * plan.dx / 2.0) / plan.dx])
    return E, H, K, Om


@pytest.mark.parametrize('case', CASES, ids=_id)
def test_incident_field_is_a_discrete_transverse_plane_wave(case):
    """``K . E = 0``, ``W H = K x E``; s has E on the normal alone, p has H there alone."""
    theta, pol, phi = case
    pw, plan, res = _long_run(theta, pol, phi)
    E, H, K, Om = _incident_amplitudes(res, pw, plan, 2.0 * math.pi + 1j * GAMMA)
    size = np.linalg.norm(E)
    assert abs(K @ E) / np.linalg.norm(K) <= RELATION_TOL * size
    assert np.linalg.norm(Om * H - np.cross(K, E)) <= RELATION_TOL * abs(Om) * size
    if theta > 0:
        # The grid's s and p are normal to the plane of the *discrete* K and z (off the
        # coordinate planes K_par is not parallel to k_par); the PlaneWave launches exactly
        # one of them, so the other is absent to round-off at every phi.
        normal = np.cross(K.real, [0.0, 0.0, 1.0])
        normal /= np.linalg.norm(normal)
        on, rest = (E, H) if pol == 's' else (H, E)
        # The field on the normal to the plane of incidence is all of it for s's E / p's H,
        # and none of it for the other.
        assert abs(abs(on @ normal) - np.linalg.norm(on)) <= 1e-12 * np.linalg.norm(on)
        assert abs(rest @ normal) <= 1e-12 * np.linalg.norm(rest)


@pytest.mark.parametrize('case', [(0.0, 's', 0.0), (20.0, 's', 0.0), (20.0, 'p', 0.0)],
                         ids=_id)
def test_energy_flows_along_k(case):
    """``Re(E x H*)`` at the band centre is along the grid's ``K`` and within 1e-4 of ``k``."""
    theta, pol, phi = case
    pw, plan, res = _long_run(theta, pol, phi)
    E, H, K, _ = _incident_amplitudes(res, pw, plan, 2.0 * math.pi)
    S = np.real(np.cross(E, np.conj(H)))
    S = S / np.linalg.norm(S)
    K = np.real(K) / np.linalg.norm(np.real(K))
    k = np.array(pw.vectors()[0])
    assert np.linalg.norm(S - K) <= 1e-5
    assert np.linalg.norm(S - k) <= 1e-4
    assert S @ k > 0.999


# ---------------------------------------------------------------------------------------------
# Mutants
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('direction', ['-z', '+z'])
@pytest.mark.parametrize('case', CASES[:5], ids=_id)
def test_mutant_M_on_the_wrong_half_plane(case, direction):
    """The ``before_H`` slabs moved one cell across the E plane: the backward wave is O(k dx)."""
    theta, pol, phi = case
    scene, pw, plan, _ = _backward_run(theta, pol, phi, direction)

    def move(c):
        if c.slot != 'H':
            return c
        z0, z1 = c.slab.z
        return dataclasses.replace(c, slab=dataclasses.replace(
            c.slab, z=(z0 + pw.sign, z1 + pw.sign)))
    res = run_mutated(scene, _with_corrections(plan, move))
    ratio = _backward_ratio(res)
    assert 0.05 < ratio < 0.3, ratio          # measured 0.116 to 0.161


@pytest.mark.parametrize('case', [(20.0, 's', 0.0), (20.0, 'p', 0.0), (40.0, 's', 0.0),
                                  (40.0, 'p', 0.0)], ids=_id)
def test_mutant_impedance_from_the_continuum(case):
    """J scaled to the continuum's H/E ratio at the band centre: |rho-1|/|rho+1| comes back."""
    theta, pol, phi = case
    scene, pw, plan, _ = _backward_run(theta, pol, phi, '-z')
    rho = rho_continuum(theta, pol, plan.dx, plan.dt)
    res = run_mutated(scene, _with_corrections(
        plan, lambda c: dataclasses.replace(c, sign=c.sign * rho) if c.slot == 'E' else c))
    predicted = abs(rho - 1.0) / abs(rho + 1.0)
    measured = _backward_ratio(res)
    assert measured == pytest.approx(predicted, rel=0.05)
    assert measured > 1e7 * ROUND_OFF              # 9.3e-6 at 20 degrees, 4.8e-4 at 40
    assert predicted == pytest.approx({20.0: 9.16e-6, 40.0: 4.77e-4}[theta], rel=0.01)


def test_continuum_impedance_is_the_discrete_one_at_normal_incidence():
    """Why the continuum mutant needs an angle: in 1-D the Yee impedance is exactly Z0."""
    assert rho_continuum(0.0, 's', 0.05, 0.05 * 0.5 / math.sqrt(3.0)) == pytest.approx(
        1.0, abs=1e-15)


def _conj_grid(params, scene, plan, rt):
    return dict(params, bloch={a: np.conj(v) for a, v in params['bloch'].items()})


def _conj_source(key):
    def mutate(params, scene, plan, rt):
        own = dict(params['sources']['pw'])
        if key == 'pattern':
            own['pattern'] = np.conj(own['pattern'])
        else:
            column = dict(own['column'])
            column['bloch'] = {a: np.conj(v) for a, v in column['bloch'].items()}
            own['column'] = column
        return dict(params, sources=dict(params['sources'], pw=own))
    return mutate


@pytest.mark.parametrize('where', ['grid', 'pattern', 'column'])
@pytest.mark.parametrize('case', [(20.0, 's', 0.0), (40.0, 'p', 0.0)], ids=_id)
def test_mutant_bloch_phase_sign_flipped(case, where):
    """Each of the three places the phase lives, conjugated: the reference gate fails at O(1)."""
    scene, pw, plan = _reference_run(*case)
    mutate = {'grid': _conj_grid, 'pattern': _conj_source('pattern'),
              'column': _conj_source('column')}[where]
    res = run_mutated(scene, plan, mutate_params=mutate)
    assert _reference_error(res) > 0.1               # measured 0.2 to 2
    if where == 'column':
        # The sheet's tangential impedance depends on K_par^2 alone, so the backward-wave
        # gate cannot see this one; the reference gate above is the one that must.
        assert _backward_ratio(res) <= ROUND_OFF


def test_mutant_from_source_sign_flipped_is_refused():
    scene, pw = bloch_plane_wave(theta=20.0)
    faces = fd.Bloch.from_source(pw)
    flipped = {a: fd.Bloch(k=-f.k) if isinstance(f, fd.Bloch) else f for a, f in faces.items()}
    bad = scene.replace(boundaries=fd.Boundaries(**flipped, z=fd.PML(cells=12)))
    with pytest.raises(ValueError, match=r'carries k = -2\.1489.*transverse wavevector at the '
                                         r'band centre is 2\.1489'):
        fd.plan(bad)
    fd.plan(scene)                                    # the accepted spelling


# ---------------------------------------------------------------------------------------------
# A non-absorbing face along the propagation axis
# ---------------------------------------------------------------------------------------------

def _mirror_backed(low_face, band_=None, resolution=None):
    """A 2 x 2 Periodic cell, a sheet at z = 0.8 going down onto ``low_face``, PML above."""
    from fd_helpers import band
    b = band() if band_ is None else band_
    dx = 1.0 / 20
    pw = fd.PlaneWave('pw', z=0.8, direction='-z', polarization='s', band=b)
    monitors = (fd.Record('up', at=(0.0, 0.0, 1.3), component='Ey'),
                fd.Record('down', at=(0.0, 0.0, 0.3), component='Ey'))
    return fd.Scene(domain=fd.Domain(size=(2 * dx, 2 * dx, 4.0)),
                    boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                             z=(low_face, fd.PML(cells=12, wavelength=1.0))),
                    resolution=resolution or fd.Resolution(dx=dx), run=fd.Run(steps=2400),
                    sources=(pw,), monitors=monitors)


@pytest.mark.parametrize('wall', ['PEC', 'PMC'])
def test_a_mirror_behind_the_cell_reflects_through_the_sheet(wall):
    scene = _mirror_backed(getattr(fd, wall)())
    plan = fd.plan(scene)
    (term,) = plan.terms
    assert term.sub.grid.n[2] == plan.grid.n[2] + 12            # a PML padded below
    res = fd.run(scene)
    incident = np.abs(to_numpy(res.sources['pw'].incident))
    up = np.abs(to_numpy(res['up'].values))
    assert np.all(np.isfinite(up)) and np.all(np.isfinite(incident))
    # |r| = 1 at a PEC or PMC wall: the whole pulse comes back up through the sheet.
    assert 0.95 < up.max() / incident.max() < 1.05, (up.max(), incident.max())
    # the reference is the free incident wave: it passes once and is gone
    assert incident[-400:].max() <= 1e-5 * incident.max()


def test_an_absorbing_cell_has_no_reflection_above_the_sheet():
    """The control for the gate above: the same cell with PML below reflects nothing."""
    res = fd.run(_mirror_backed(fd.PML(cells=12, wavelength=1.0)))
    incident = np.abs(to_numpy(res.sources['pw'].incident))
    assert np.abs(to_numpy(res['up'].values)).max() <= ROUND_OFF * incident.max()


def test_a_padded_column_needs_a_concrete_wavelength():
    """The column's own PML is graded at a wavelength; a traced band centre has none to give."""
    torch = pytest.importorskip('torch')
    from fd_helpers import band
    traced = band(wavelength=torch.tensor(1.0, dtype=torch.float64, requires_grad=True))
    with pytest.raises(ValueError, match='needs a concrete wavelength'):
        fd.plan(_mirror_backed(fd.PEC(), band_=traced))
    # the accepted spelling: the scene names the wavelength its layers are graded at (the
    # mesh check still cannot read a traced band, and says so)
    with pytest.warns(fd.CoarseMeshWarning, match='band is traced'):
        fd.plan(_mirror_backed(fd.PEC(), band_=traced, resolution=fd.Resolution(
            cells_per_wavelength=20, wavelength=1.0)))
