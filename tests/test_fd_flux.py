"""Stage-3 gates 1 and 4: flux closure and the sources' spectral normalisation.

Every number a flux or a normalisation returns is post-processing of the DFT accumulators
(``wavebit/fd/_post.py``), so each gate below runs a scene **once** and reads it several ways:
the right way, and each mutant's way (``_post`` and ``_assemble`` functions swapped for the
read only, on the same final state -- the mutant changes what is computed, never what ran).

What is gated, and what was measured (NumPy, 2026-09-19, 20 cells per wavelength unless said):

* **Closure, source outside.** The net flux of an empty closed box beside a dipole is
  ~1e-9 of the dipole's power and ~1e-7 of the box's largest face. That is *not* the DFT's
  round-off, and running longer does not take it there: 10 / 14 / 20 / 30 / 45 periods give
  2.5e-8 / 4e-9 / 4e-9 / 3e-9 / 1e-9 of the power, with an ``fd.Absorber`` box as with a PML.
  The residue is the late-time field left in the domain -- the final ``|H|`` is still 1e-5,
  2e-6, 9e-7 of its peak at 10, 14, 30 periods, relaxing slowly -- and the end correction
  (``_post.end_correction``) is exact only for a static, curl-free residue: for an oscillating
  one the E and H identities hold for phasors that differ by that residue. Without the
  correction the same box is 4e-5 of its largest face, so the correction buys 2.5 decades and
  what is left follows the residue. The gate holds 1e-6 of the largest face (and 1e-7 of the
  power), a decade over the floor.
* **Closure, source inside.** The net outward flux of a box around a dipole equals
  ``res.sources['d'].power()`` to 7e-8 (the same floor), for an electric and a magnetic dipole,
  and at every resolution. The flux-power identity is exact on the grid, so what converges with
  resolution is the power itself: against the continuum radiation of the same current
  moment, ``k^2 |w^|^2 / (12 pi)`` (``c = eta = 1``), it is -2.11e-2 / -1.08e-2 / -5.29e-3 at 20
  / 28 / 40 cells per wavelength at the band centre -- fitted order 2.00 (1.99 and 2.00 at 0.9
  and 1.1). The B-spline stamp's spread and the grid's dispersion are both second order.
* **Plane flux.** A PlaneWave's flux through a full plane below its sheet is ``-I A`` (``I``
  the source's ``intensity()``, ``A`` the cell area, the minus the ``-z`` travel) to 1e-7 at
  normal incidence, 1.5e-6 (s) and 1.2e-5 (p, at wavelength 1.1) at 25 degrees after 1200
  steps; the gate runs 1600.
* **TFSF.** An empty TFSF box's total-field region holds the incident wave alone, so a flux
  box inside it has an entry face of ``-I A`` and an exit face of ``+I A``: 1 to 1e-6 either
  way, net 1e-7 of ``I A``, for ``-z`` and ``+x`` incidence.
* **Incident amplitude.** In an empty Bloch cell the ``Diffraction`` zero order of a plane
  below the sheet is the incident wave itself, so ``t`` of order ``(0, 0)`` -- the main grid's
  outgoing amplitude over ``incident_amplitude()`` read from the source's column -- is 1: to
  1.3e-7 at normal incidence, 1.5e-6 (s) and 8.8e-6 (p) at 25 degrees, with the reference
  planes moved or not; ``T`` is 1 as closely, and a reflect monitor in the scattered-field
  region reads ``R`` of 1e-32.

The mutants, and how each fails (the gate's tolerance is in brackets):

* **H not collocated.** (a) The tangential H of one half plane (the low one) paired with the
  node-plane E instead of the mean of the two: the dipole's box misses its power by -1.58e-3
  / -1.18e-3 / -8.0e-4 at 20 / 28 / 40 cells per wavelength -- **first order in dx**, the
  O(k dx) leak the design predicted [1e-6] -- and the empty box is 1.4e-3 of the power [1e-7].
  It is invisible to a single travelling plane wave: there E and H are in phase and one half
  plane carries ``exp(+-i kappa dx/2)`` where the mean carries ``cos(kappa dx/2)``, the same
  real part, so the PlaneWave's plane flux is unchanged to round-off -- which is why the
  closure gate is a near field. (b) The mean of the two H planes moved one cell along ``u``
  (a transverse misplacement): -3.7e-2 / -2.4e-2 / -1.5e-2 on the fixed 0.4 box (order 1.3).
* **A box face with the wrong outward sign** (``x+`` flipped): the dipole's box reads -0.45
  of the power off (twice that face's share) and the empty box 1.9e-2 of the power.
* **The dipole power from the H time level**: the stamp-weighted E read as if it were sampled
  half a step earlier, at the H times -- the power a pairing of the current with the wrong
  time level gives. The dipole's near field is almost all reactive (``Q/P`` 13 to 25), so the
  phase error ``w dt/2`` moves ``Re`` by +64% / +82% / +101% at the three wavelengths at 20
  cells per wavelength and grows as the grid refines (+315% .. +476% at 40).

Also found while measuring (not gated here): a flux box whose face cuts a dipole's stamp is
accepted without a word and misses part of the power (-1.6e-3 at 13 cells per wavelength with
a 0.3 box, the stamp's half width being two cells); the boxes below are whole cells and clear
the stamp by two cells at every resolution.
"""
import dataclasses
import math
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd import _assemble, _post
from wavebit.fd._lower import lower, plan as make_plan
from wavebit.fd._run import _results, execute
from wavebit.fd._runtime import runtime

from fd_helpers import WAVELENGTHS, bloch_plane_wave, dipole_box, tfsf_box, to_numpy

WS = WAVELENGTHS
#: Where the unmutated closure must land: a decade over the measured late-time floor.
CLOSURE_OF_POWER = 1e-6
EMPTY_OF_POWER = 1e-7
EMPTY_OF_FACE = 1e-6


# ---------------------------------------------------------------------------------------------
# One run, read many ways
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Solved:
    scene: object
    plan: object
    params: object
    state: object
    rt: object

    def read(self):
        """The Results of the finished state, assembled with whatever ``_post`` holds now."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return _results(self.scene, self.plan, self.state, self.rt, self.params)


def solve(scene):
    rt = runtime('numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = make_plan(scene)
        params = lower(scene, p, rt)
        state = execute(p, params, rt)
    return Solved(scene, p, params, state, rt)


def dipole_scene(cells_per_wavelength=20, kind='electric', size=0.8, periods=12, empty=True,
                 fixed=False):
    """A dipole in an open PML box: ``'in'`` a box of 8 cells around it, ``'empty'`` one beside.

    Both boxes are whole cells; ``'in'`` clears the B-spline stamp (half width two cells) by two
    cells on every side at every resolution, ``'empty'`` (4 cells at 20 cells per wavelength)
    holds no part of it. ``fixed=True`` adds ``'fixed'``, a box 0.4 on a side whatever the
    resolution -- the one a leak's order in ``dx`` is read on (``'in'`` shrinks with the grid,
    so its faces sit in a near field that changes with it).
    """
    dx = 1.0 / cells_per_wavelength
    monitors = [fd.Flux('in', box=fd.Box(size=(8 * dx,) * 3), wavelengths=WS)]
    if fixed:
        monitors.append(fd.Flux('fixed', box=fd.Box(size=(0.4,) * 3), wavelengths=WS))
    if empty:
        monitors.append(fd.Flux('empty', box=fd.Box(size=(0.2,) * 3,
                                                     center=(0.25, 0.25, -0.25)),
                                wavelengths=WS))
    component = 'Ez' if kind == 'electric' else 'Hz'
    return dipole_box(center=(0.013, -0.02, 0.007), component=component, kind=kind, size=size,
                      cells_per_wavelength=cells_per_wavelength, periods=periods,
                      monitors=tuple(monitors))


@pytest.fixture(scope='module')
def dipole():
    return solve(dipole_scene())


RESOLUTIONS = (20, 28, 40)


@pytest.fixture(scope='module')
def ladder():
    """The same dipole at three resolutions, a smaller box (0.6 of interior): ~10 s together."""
    return {cpw: solve(dipole_scene(cpw, size=0.6, empty=False, fixed=True))
            for cpw in RESOLUTIONS}


def continuum_power(series, plan):
    """``k^2 |w^|^2 / (12 pi)``: a current moment ``w`` radiating in vacuum (``c = eta = 1``)."""
    freqs = np.array([1.0 / w for w in WS])
    w = to_numpy(_post.dft(series.context.rt, series.drive, series.t, plan.dt, freqs))
    k = 2.0 * math.pi * freqs
    return k ** 2 * np.abs(w) ** 2 / (12.0 * math.pi)


def fitted_order(dxs, errors):
    return float(np.polyfit(np.log(dxs), np.log(np.abs(errors)), 1)[0])


def largest_face(result):
    return max(float(np.max(np.abs(to_numpy(v)))) for v in result.faces.values())


# ---------------------------------------------------------------------------------------------
# Gate 1: closure
# ---------------------------------------------------------------------------------------------

def test_an_empty_box_beside_a_dipole_has_no_net_flux(dipole):
    res = dipole.read()
    P = to_numpy(res.sources['d'].power())
    net = to_numpy(res['empty'].flux)
    assert np.all(P > 0)
    assert np.all(np.abs(net) < EMPTY_OF_POWER * P), net / P
    assert np.all(np.abs(net) < EMPTY_OF_FACE * largest_face(res['empty'])), net
    # The faces themselves carry flux -- the box is not in a dead spot.
    assert largest_face(res['empty']) > 1e-3 * float(np.max(P))


def test_a_box_around_a_dipole_carries_its_power(dipole):
    res = dipole.read()
    P = to_numpy(res.sources['d'].power())
    flux = to_numpy(res['in'].flux)
    np.testing.assert_allclose(flux / P, 1.0, rtol=0, atol=CLOSURE_OF_POWER)
    assert res.diagnostics['decay']['in'] < _assemble.UNSETTLED_THRESHOLD


def test_a_box_around_a_magnetic_dipole_carries_its_power():
    res = solve(dipole_scene(kind='magnetic', empty=False)).read()
    P = to_numpy(res.sources['d'].power())
    assert np.all(P > 0)
    np.testing.assert_allclose(to_numpy(res['in'].flux) / P, 1.0, rtol=0,
                               atol=CLOSURE_OF_POWER)


def test_the_dipole_power_converges_second_order_to_the_continuum(ladder):
    errors, dxs = [], []
    for cpw, solved in ladder.items():
        res = solved.read()
        series = res.sources['d']
        P = to_numpy(series.power())
        # Exact on the grid at every resolution: the box closes on the power.
        for box in ('in', 'fixed'):
            np.testing.assert_allclose(to_numpy(res[box].flux) / P, 1.0, rtol=0,
                                       atol=CLOSURE_OF_POWER)
        errors.append(P / continuum_power(series, solved.plan) - 1.0)
        dxs.append(1.0 / cpw)
    errors = np.array(errors)                   # (resolution, wavelength)
    assert np.all(errors < 0)                   # the grid radiates less, from below
    assert np.all(np.abs(errors[0]) < 3e-2)     # measured 1.7e-2 .. 2.6e-2 at 20
    for j in range(len(WS)):
        order = fitted_order(dxs, errors[:, j])
        assert 1.85 < order < 2.15, (WS[j], order, errors[:, j])


# ---------------------------------------------------------------------------------------------
# Plane flux, TFSF intensity, incident amplitude
# ---------------------------------------------------------------------------------------------

PLANE_CASES = ((0.0, 's'), (25.0, 's'), (25.0, 'p'))


@pytest.fixture(scope='module')
def plane_waves():
    """An empty Bloch cell per case: a flux plane and three Diffraction monitors below/above."""
    out = {}
    for theta, pol in PLANE_CASES:
        monitors = (fd.Flux('plane', plane=fd.Plane(z=-0.5), wavelengths=WS),
                    fd.Diffraction('t', z=-0.5, side='transmit', wavelengths=WS),
                    fd.Diffraction('t_moved', z=-0.5, side='transmit', wavelengths=WS,
                                   reference_plane=-0.2, incident_plane=-0.2),
                    fd.Diffraction('r', z=0.8, side='reflect', wavelengths=WS))
        scene, _ = bloch_plane_wave(theta=theta, polarization=pol, at=0.5, steps=1600,
                                    length=2.0, monitors=monitors)
        out[theta, pol] = solve(scene)
    return out


@pytest.mark.parametrize('theta,pol', PLANE_CASES)
def test_a_plane_wave_plane_flux_is_its_intensity_times_the_area(plane_waves, theta, pol):
    solved = plane_waves[theta, pol]
    res = solved.read()
    size = solved.scene.domain.size
    I = to_numpy(res.sources['pw'].intensity(WS))
    assert np.all(I > 0)
    ratio = to_numpy(res['plane'].flux) / (I * size[0] * size[1])
    np.testing.assert_allclose(ratio, -1.0, rtol=0, atol=2e-5)      # travelling -z


@pytest.mark.parametrize('theta,pol', PLANE_CASES)
def test_the_incident_amplitude_is_the_empty_cells_zero_order(plane_waves, theta, pol):
    res = plane_waves[theta, pol].read()
    amp = 't_s' if pol == 's' else 't_p'
    for name in ('t', 't_moved'):
        orders = res[name].orders
        assert orders.order_window == ((0, 0), (0, 0))
        t = to_numpy(getattr(orders.order(0, 0), amp))
        np.testing.assert_allclose(t, 1.0, rtol=0, atol=2e-5)
        np.testing.assert_allclose(to_numpy(orders.order(0, 0).T), 1.0, rtol=0, atol=3e-5)
    # The scattered-field side of an empty cell holds nothing at all.
    assert np.all(np.abs(to_numpy(res['r'].orders.order(0, 0).R)) < 1e-20)
    # The amplitude itself: the band's spectrum, nonzero at every wavelength.
    A = to_numpy(res.sources['pw'].incident_amplitude(WS))
    assert np.all(np.abs(A) > 0.1 * np.max(np.abs(A)))


@pytest.mark.parametrize('direction,pol', (('-z', 'x'), ('+x', 'y')))
def test_the_tfsf_intensity_is_an_empty_boxs_entry_face_flux(direction, pol):
    side = 0.3
    monitors = (fd.Flux('inner', box=fd.Box(size=(side,) * 3), wavelengths=WS),)
    res = solve(tfsf_box(direction=direction, polarization=pol, margin=0.3, steps=700,
                         monitors=monitors)).read()
    I = to_numpy(res.sources['t'].intensity(WS))
    assert np.all(I > 0)
    axis, going = direction[1], direction[0]
    entry = axis + ('+' if going == '-' else '-')
    leave = axis + ('-' if going == '-' else '+')
    IA = I * side * side
    faces = res['inner'].faces
    # Outward flux: the wave comes in through the entry face and leaves through the other.
    np.testing.assert_allclose(to_numpy(faces[entry]) / IA, -1.0, rtol=0, atol=1e-5)
    np.testing.assert_allclose(to_numpy(faces[leave]) / IA, 1.0, rtol=0, atol=1e-5)
    np.testing.assert_allclose(to_numpy(res['inner'].flux) / IA, 0.0, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------------------------
# Mutants
# ---------------------------------------------------------------------------------------------

def _one_half_plane(H, P):
    """Mutant: the low half plane's H alone, half a cell off the E node plane."""
    return H[_post._index(4, _post.AX[P] + 1, slice(0, 1))]


def _shifted_one_cell(H, P):
    """Mutant: the mean of the two half planes, moved one cell along ``u``."""
    mean = 0.5 * (H[_post._index(4, _post.AX[P] + 1, slice(0, 1))]
                  + H[_post._index(4, _post.AX[P] + 1, slice(1, 2))])
    return np.roll(mean, 1, axis=_post.AX[(P + 1) % 3] + 1)


def test_mutant_h_from_one_half_plane_leaks_first_order(ladder, dipole, monkeypatch):
    monkeypatch.setattr(_post, '_mean_planes', _one_half_plane)
    leaks, dxs = [], []
    for cpw, solved in ladder.items():
        res = solved.read()
        P = to_numpy(res.sources['d'].power())
        leaks.append(to_numpy(res['fixed'].flux) / P - 1.0)
        dxs.append(1.0 / cpw)
    leaks = np.array(leaks)
    # Measured -1.58e-3 / -1.18e-3 / -8.0e-4: a thousand times the gate's tolerance, and O(dx).
    assert np.all(np.abs(leaks) > 300 * CLOSURE_OF_POWER), leaks
    for j in range(len(WS)):
        assert 0.8 < fitted_order(dxs, leaks[:, j]) < 1.2, leaks[:, j]
    res = dipole.read()
    P = to_numpy(res.sources['d'].power())
    assert np.all(np.abs(to_numpy(res['empty'].flux)) > 1e3 * EMPTY_OF_POWER * P)


def test_mutant_h_from_one_half_plane_is_invisible_to_one_plane_wave(plane_waves, monkeypatch):
    # Why the closure gate needs a near field: E and H of a travelling wave are in phase, and
    # one half plane's exp(+-i kappa dx/2) has the real part the mean's cos(kappa dx/2) has.
    solved = plane_waves[25.0, 'p']
    right = to_numpy(solved.read()['plane'].flux)
    monkeypatch.setattr(_post, '_mean_planes', _one_half_plane)
    mutant = to_numpy(solved.read()['plane'].flux)
    np.testing.assert_allclose(mutant, right, rtol=1e-6)


def test_mutant_h_moved_one_cell_along_the_plane_leaks(dipole, monkeypatch):
    monkeypatch.setattr(_post, '_mean_planes', _shifted_one_cell)
    res = dipole.read()
    P = to_numpy(res.sources['d'].power())
    leak = to_numpy(res['in'].flux) / P - 1.0
    assert np.all(np.abs(leak) > 1e4 * CLOSURE_OF_POWER), leak      # measured -3.6e-2 .. -3.9e-2


def test_mutant_a_box_face_with_the_wrong_outward_sign(dipole, monkeypatch):
    right = _assemble.box_faces

    def flipped(where, grid, box):
        return tuple((face, -sign if face == 'x+' else sign, layout)
                     for face, sign, layout in right(where, grid, box))
    monkeypatch.setattr(_assemble, 'box_faces', flipped)
    res = dipole.read()
    P = to_numpy(res.sources['d'].power())
    # Twice the x+ face's share: measured -0.45 of the power, and 1.9e-2 for the empty box.
    np.testing.assert_allclose(to_numpy(res['in'].flux) / P - 1.0, -0.45, atol=0.02)
    assert np.all(np.abs(to_numpy(res['empty'].flux)) > 1e4 * EMPTY_OF_POWER * P)


def test_mutant_the_dipole_power_from_the_h_time_level(dipole):
    res = dipole.read()
    series = res.sources['d']
    P = to_numpy(series.power())
    # The stamp-weighted E read as if it were sampled at the H times, half a step early.
    moved = dataclasses.replace(series, t_field=series.t_field - 0.5 * dipole.plan.dt)
    wrong = to_numpy(moved.power())
    # Measured +64% / +82% / +101%: the reactive near field turned by w dt/2.
    assert np.all(wrong / P - 1.0 > 0.5), wrong / P
    assert np.all(np.abs(to_numpy(res['in'].flux) / wrong - 1.0) > 0.3)
