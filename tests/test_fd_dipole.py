"""Gate 2 of stage 2 -- the point dipole: its radiated energy, its dual, its moments, its gradient.

**The closed form, in these units** (``c = eps0 = mu0 = 1``, H stored as ``Z0 H``, so the wave
impedance is 1). A time-harmonic current element ``J dV = Il z e^{-iwt}`` radiates the far field
``|E_theta| = |H_phi| = k |Il| sin(theta) / (4 pi r)``; the time-averaged Poynting flux is
``|E|^2 / 2``, and over the sphere (``int sin^2 dOmega = 8 pi / 3``)

    P = k^2 |Il|^2 / (12 pi),        so  R(w) = 2 P / |Il|^2 = w^2 / (6 pi)   (k = w).

A ``fd.Dipole`` is driven by a pulse, and its moment is ``sum J dV = w(t)`` -- the stamp sums to
1 -- so the quantity with a closed form is the *energy* it delivers over the pulse. With
``p(w) = int w(t) e^{iwt} dt`` and ``E`` at the source ``= -Z(w) p``, Parseval gives
``U = (1/pi) int_0^inf R(w) |p(w)|^2 dw``, the band integral of ``P`` above; with ``R = w^2/(6 pi)``
that is ``(1/(6 pi^2)) int_0^inf w^2 |p|^2 dw = (1/(6 pi)) int (dw/dt)^2 dt`` -- Larmor's formula
for the dipole moment ``d`` with ``dd/dt = Il = w``. For ``w = A exp(-s^2/(2 tau^2))
sin(w0 s + phi)`` the integral is Gaussian:

    U = A^2 sqrt(pi) tau (w0^2 + 1/(2 tau^2)) / (12 pi),

dropping terms of order ``exp(-w0^2 tau^2)`` (``exp(-44)`` at the default band) and the energy of
the envelope's start at ``cutoff`` (the table begins at ``t = 0``). The grid's ``U`` is its own
work series summed, ``dt * sum(res.sources['d'].work)``, the Poynting ``-sum J.E dV`` per step
-- the source's own measurement, with no monitor and no flux box.

What is pinned:

* **Second order in dx.** A ``Ez`` dipole in a vacuum PML box at 14, 20 and 28 cells per centre
  wavelength. The fitted exponent of ``|U/U_c - 1|`` against ``dx`` is 2 (measured 1.98), and the
  Richardson extrapolation of the two finest runs lands on the closed form to 1e-3 (measured
  -7.3e-5; the errors themselves are -4.7%, -2.3%, -1.2%, which is the grid's dispersion and the
  stamp's width, both O(dx^2)). The limit is what pins the *constant*: a ``12 pi`` for ``6 pi``
  or a current off by ``dx`` would converge, to the wrong number.
* **Electric/magnetic duality**, ``E -> H, H -> -E, J -> M, M -> -J``. On the Yee lattice the
  dual of an E component sits half a cell up the diagonal from it, and the dual of the
  ``before_E`` half step is the ``before_H`` one, half a step earlier. So a magnetic dipole
  moved by ``dx/2 (1, 1, 1)`` and delayed by ``-dt/2`` is the electric one's exact dual: in a
  periodic box (a lattice with no edge) its field series, its work and its whole H field equal
  the electric run's to round-off (measured 2e-15). In the PML box the CPML's half-cell
  profiles are not self-dual, and the two energies agree to 1e-4 (measured 2.7e-5).
* **The moments.** One step from rest leaves the injected current alone in the field
  (``E(1) = -dt J / eps`` for J, ``H(1/2) = -dt M / mu`` for M), so the kernel's actual
  injection is read off the final field. Its sum is the requested moment -- the drive, which
  the delay and phase put at exactly the requested amplitude -- and its centroid is the
  requested ``center``, both to round-off, for both stamps, both kinds and all components.
* **The position differentiates.** ``d U / d center_x`` within ``range=``, on JAX and Torch,
  against central differences of the NumPy run, for both stamps. ``U`` there is the energy
  delivered over the first 200 steps -- mid-pulse, so it still holds the reactive near-field
  energy, which depends on the stamp's sub-cell position enough to give a derivative of order
  one (0.84 for the B-spline, 1.17 for the tent; central differences at h = 1e-4, 1e-5, 1e-6
  agree to 1e-7), where the full pulse's radiated energy would give one of order 1e-3.

**Mutants**, each run through the check that exists to catch it:

* the stamp normalised by its sum of squares instead of its sum -- the moment gate (the sum is
  15.9, not 1) and the energy gate (U is 75x the closed form);
* the stamp built at ``center + dx/2`` -- the centroid gate (the energy cannot see it: the
  B-spline's radiated power moves by 1e-4 over a whole cell, which is why it was chosen);
* the work series computed with the stamp-weighted H instead of E -- the energy gate (U ~ 0).
"""
import math

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._kernel.sources import DipoleTerm, stamp
from wavebit.fd._runtime import runtime
from fd_helpers import backends, band, dipole_box, run_mutated, to_numpy

pytestmark = pytest.mark.filterwarnings('ignore::wavebit.fd.CoarseMeshWarning')

#: Off every node, every half node and every symmetry plane of the boxes below.
CENTER = (0.013, -0.021, 0.007)
#: The small open box: 0.6 wavelengths of interior, a 12-cell PML all round.
SIZE = 0.6
RESOLUTIONS = (14, 20, 28)


def closed_form_energy(amplitude=1.0, wavelength=1.0, df=0.15):
    """``U = (1/(6 pi)) int (dw/dt)^2 dt`` for the band's pulse (module docstring)."""
    tau = 1.0 / (2.0 * math.pi * df)
    w0 = 2.0 * math.pi / wavelength
    return amplitude ** 2 * math.sqrt(math.pi) * tau * (w0 ** 2 + 0.5 / tau ** 2) / (12 * math.pi)


def delivered_energy(res, name='d'):
    """``dt * sum(work)``: the energy the source put into the field, from its own series."""
    return res.plan.dt * float(np.sum(to_numpy(res.sources[name].work)))


def energy_error(cpw, backend='numpy', mutate_params=None, plan_hook=None, **kwargs):
    """``U / U_c - 1`` for a dipole in the small open box at ``cpw`` cells per wavelength."""
    scene = dipole_box(center=CENTER, size=SIZE, cells_per_wavelength=cpw, periods=8, **kwargs)
    plan = fd.plan(scene)
    if plan_hook is not None:
        plan = plan_hook(plan)
    res = run_mutated(scene, plan=plan, backend=backend, mutate_params=mutate_params)
    return delivered_energy(res) / closed_form_energy(kwargs.get('amplitude', 1.0)) - 1.0


#: At 14 cells per wavelength the grid delivers 4.7% less than the continuum (measured).
COARSE_ERROR = (-0.06, -0.03)


def check_coarse_energy(err):
    """The single-resolution energy gate: the measured deficit, not the mutants' O(1)."""
    return COARSE_ERROR[0] < err < COARSE_ERROR[1]


# ---------------------------------------------------------------------------------------------
# The energy, converging at second order
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def convergence():
    return [energy_error(cpw) for cpw in RESOLUTIONS]


def test_the_closed_form_is_the_band_integral_of_k2_il2_over_12pi():
    """The Gaussian closed form equals ``(1/pi) int R |p|^2 dw`` with ``R = w^2/(6 pi)``, numerically."""
    A, df = 1.3, 0.15
    tau = 1.0 / (2 * math.pi * df)
    s = np.linspace(-12 * tau, 12 * tau, 20001)
    w = A * np.exp(-0.5 * (s / tau) ** 2) * np.sin(2 * math.pi * s + 0.4)
    om = np.linspace(0.0, 4 * math.pi, 1601)
    p = (np.exp(1j * np.outer(om, s)) @ w) * (s[1] - s[0])
    U = np.trapezoid(om ** 2 / (6 * math.pi) * np.abs(p) ** 2, om) / math.pi
    assert math.isclose(U, closed_form_energy(A, 1.0, df), rel_tol=1e-9)


def test_radiated_energy_converges_to_the_closed_form_at_second_order(convergence):
    """Exponent 2 in dx, and the Richardson limit of the two finest runs is the closed form."""
    errors = np.array(convergence)
    assert np.all(errors < 0), errors                       # a deficit at every resolution
    dx = 1.0 / np.array(RESOLUTIONS)
    order = np.polyfit(np.log(dx), np.log(np.abs(errors)), 1)[0]
    assert 1.9 < order < 2.1, (order, errors)
    ratio = (RESOLUTIONS[2] / RESOLUTIONS[1]) ** 2
    limit = (ratio * errors[2] - errors[1]) / (ratio - 1)
    assert abs(limit) < 1e-3, (limit, errors)
    assert check_coarse_energy(errors[0]), errors


def test_mutant_the_stamp_normalised_by_its_sum_of_squares_is_seen():
    """``S / sum S^2``: the energy gate sees U at 75x (measured); the moment gate sees it too."""
    def mutate(params, scene, plan, rt):
        entry = params['sources']['d']
        S = entry['stamp']
        return dict(params, sources=dict(params['sources'],
                                         d=dict(entry, stamp=S / rt.xp.sum(S * S))))
    err = energy_error(RESOLUTIONS[0], mutate_params=mutate)
    assert not check_coarse_energy(err) and err > 10, err


class _WorkWithH(DipoleTerm):
    """The mutant: the work series reads the stamp-weighted H, not the mean of the two E."""

    def advance(self, mem, E, H, n, params, rt):
        out = super().advance(mem, E, H, n, params, rt)
        fH = rt.xp.sum(H[self.slab.slices] * params['sources'][self.name]['stamp'])
        w = params['waveform'][self.row, n]
        return dict(out, work=rt.record(out['work'], n, -w * fH))


def test_mutant_the_work_series_computed_with_H_is_seen():
    """The energy gate sees a work series that reads H: it delivers ~nothing."""
    def hook(plan):
        terms = tuple(_WorkWithH(**{f: getattr(t, f) for f in ('name', 'field', 'slab', 'row',
                                                               'dx', 'stamp')})
                      if isinstance(t, DipoleTerm) else t for t in plan.terms)
        return plan.with_static(terms=terms)
    err = energy_error(RESOLUTIONS[0], plan_hook=hook)
    assert not check_coarse_energy(err) and abs(err + 1) < 0.1, err


# ---------------------------------------------------------------------------------------------
# Duality
# ---------------------------------------------------------------------------------------------

def _periodic_dipole(kind, component, center, delay, dx=1 / 14, cells=16, steps=150):
    periodic = fd.Periodic()
    return fd.Scene(domain=fd.Domain(size=(cells * dx,) * 3),
                    boundaries=fd.Boundaries(x=periodic, y=periodic, z=periodic),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps),
                    sources=(fd.Dipole('d', center=center, component=component, kind=kind,
                                       band=band(), delay=delay),))


@pytest.mark.parametrize('axis', 'xyz')
def test_the_magnetic_dipole_is_the_exact_dual_on_the_lattice(axis):
    """M at ``center + dx/2 (1,1,1)``, delayed ``-dt/2``: field, work and H equal J's E."""
    dx = 1 / 14
    electric = _periodic_dipole('electric', 'E' + axis, CENTER, 0.0)
    dt = fd.plan(electric).dt
    moved = tuple(v + dx / 2 for v in CENTER)
    magnetic = _periodic_dipole('magnetic', 'H' + axis, moved, -dt / 2)
    re, rm = fd.run(electric), fd.run(magnetic)
    a, b = re.sources['d'], rm.sources['d']
    scale = np.max(np.abs(to_numpy(a.field)))
    assert np.max(np.abs(to_numpy(a.drive) - to_numpy(b.drive))) < 1e-13
    assert np.max(np.abs(to_numpy(a.field) - to_numpy(b.field))) < 1e-13 * scale
    assert math.isclose(delivered_energy(re), delivered_energy(rm), rel_tol=1e-13)
    assert delivered_energy(re) > 0
    # The whole field. The magnetic run's clock is half a step ahead, so its final H (at
    # (N - 1/2) dt) is the dual of the electric run's final E (at N dt). E_k at (i + 1/2, j, k)
    # moved by dx/2 (1, 1, 1) is (i + 1, j + 1/2, k + 1/2), which is H_k's point at index i + 1:
    # one index up the component's own axis, and nothing else.
    E = to_numpy(re.final.E)
    H = to_numpy(rm.final.H)
    for k in range(3):
        shifted = np.roll(E[k], 1, axis=2 - k)
        assert np.max(np.abs(H[k] - shifted)) < 1e-13 * np.max(np.abs(E)), (axis, k)


def test_magnetic_and_electric_energies_agree_in_the_open_box():
    """In the PML box the dual pair delivers the same energy to 1e-4 (the CPML's half cells)."""
    dx = 1 / RESOLUTIONS[0]
    electric = dipole_box(center=CENTER, size=SIZE, cells_per_wavelength=RESOLUTIONS[0],
                          periods=8)
    dt = fd.plan(electric).dt
    magnetic = dipole_box(center=tuple(v + dx / 2 for v in CENTER), size=SIZE,
                          cells_per_wavelength=RESOLUTIONS[0], periods=8, kind='magnetic',
                          component='Hz', delay=-dt / 2)
    Ue, Um = delivered_energy(fd.run(electric)), delivered_energy(fd.run(magnetic))
    assert math.isclose(Ue, Um, rel_tol=1e-4), (Ue, Um)
    assert check_coarse_energy(Um / closed_form_energy() - 1)


# ---------------------------------------------------------------------------------------------
# The moments: what the kernel injects, read off the field after one step
# ---------------------------------------------------------------------------------------------

#: Yee offsets in half cells, written from the lattice: E_c is half a cell along c, H_c half a
#: cell along the other two.
YEE = {'E': lambda c: tuple(int(a == c) for a in range(3)),
       'H': lambda c: tuple(int(a != c) for a in range(3))}


def injected(kind, component, center, amplitude=1.7, stamp_kind='bspline', mutate_params=None):
    """``(moment, centroid, drive)`` of the current the kernel injects in step 0.

    The delay puts the envelope's peak and the phase ``pi/2`` the carrier's at the drive's
    first sample, so the drive there is the requested amplitude.
    """
    b = band()
    field = 'E' if kind == 'electric' else 'H'
    probe = _periodic_dipole(kind, component, center, 0.0, steps=1)
    dt = fd.plan(probe).dt
    tau = 1.0 / (2 * math.pi * b.df)
    first = 0.5 * dt if field == 'E' else 0.0
    source = fd.Dipole('d', center=center, component=component, kind=kind, band=b,
                       amplitude=amplitude, phase=math.pi / 2,
                       delay=first - b.sigmas * tau, stamp=stamp_kind)
    scene = probe.replace(sources=(source,))
    res = run_mutated(scene, mutate_params=mutate_params)
    plan = res.plan
    F = to_numpy(res.final.E if field == 'E' else res.final.H)
    c = 'xyz'.index(component[1])
    q = -F[c] * plan.dx ** 3 / plan.dt                      # J dV (or M dV) per Yee point
    others = [k for k in range(3) if k != c]
    assert np.max(np.abs(F[others])) == 0.0                 # nothing else moved in one step
    offset = YEE[field](c)
    axes = [plan.low[a] + (np.arange(n) + 0.5 * offset[a]) * plan.dx
            for a, n in enumerate((plan.nx, plan.ny, plan.nz))]
    Z, Y, X = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
    moment = q.sum()
    centroid = tuple(float((q * r).sum() / moment) for r in (X, Y, Z))
    drive = float(to_numpy(res.sources['d'].drive)[0])
    return float(moment), centroid, drive


MOMENT_CASES = [(kind, f'{"E" if kind == "electric" else "H"}{a}', s)
                for kind in ('electric', 'magnetic') for a in 'xyz'
                for s in ('bspline', 'tent')]


@pytest.mark.parametrize('kind, component, stamp_kind', MOMENT_CASES)
def test_the_moment_and_centroid_are_the_requested_ones(kind, component, stamp_kind):
    """``sum J dV`` is the amplitude and its centroid is ``center``, to round-off."""
    for center in (CENTER, (-0.031, 0.044, 0.0), (0.0357, 0.0, -0.0178)):
        moment, centroid, drive = injected(kind, component, center, stamp_kind=stamp_kind)
        assert math.isclose(drive, 1.7, rel_tol=1e-14), drive
        assert math.isclose(moment, drive, rel_tol=1e-13), (moment, drive)
        assert np.allclose(centroid, center, rtol=0, atol=1e-13), (centroid, center)


def _mutated_stamp(transform):
    def mutate(params, scene, plan, rt):
        entry = params['sources']['d']
        S = transform(entry['stamp'], scene, plan, rt)
        return dict(params, sources=dict(params['sources'], d=dict(entry, stamp=S)))
    return mutate


def test_mutant_sum_of_squares_fails_the_moment():
    mutate = _mutated_stamp(lambda S, scene, plan, rt: S / rt.xp.sum(S * S))
    moment, centroid, drive = injected('electric', 'Ez', CENTER, mutate_params=mutate)
    assert moment / drive > 5, moment / drive
    assert np.allclose(centroid, CENTER, atol=1e-13)        # the centroid cannot see it


def test_mutant_a_centre_half_a_cell_off_fails_the_centroid():
    def shifted(S, scene, plan, rt):
        term = next(t for t in plan.terms if isinstance(t, DipoleTerm))
        c = scene.sources[0].center
        moved = (c[0] + plan.dx / 2, c[1], c[2])
        return stamp(term, tuple(rt.array(v, 'float64') for v in moved), plan.grid, rt)

    moment, centroid, drive = injected('electric', 'Ez', CENTER,
                                       mutate_params=_mutated_stamp(shifted))
    assert math.isclose(moment, drive, rel_tol=1e-13)      # the moment cannot see it
    dx = 1 / 14
    assert abs(centroid[0] - CENTER[0] - dx / 2) < 1e-12, centroid


# ---------------------------------------------------------------------------------------------
# The position differentiates
# ---------------------------------------------------------------------------------------------

RANGE = ((-0.03, 0.05), (-0.03, 0.0), (0.0, 0.03))
GRAD_STEPS = 200


def _energy_of_x(x, backend, stamp_kind):
    scene = dipole_box(center=(x, CENTER[1], CENTER[2]), rng=RANGE, size=SIZE,
                       cells_per_wavelength=RESOLUTIONS[0], steps=GRAD_STEPS, stamp=stamp_kind)
    res = fd.run(scene, backend=backend)
    work = res.sources['d'].work
    return res.plan.dt * work.sum()


@pytest.mark.parametrize('stamp_kind', ('bspline', 'tent'))
@pytest.mark.parametrize('backend', [b for b in backends() if b.values[0] != 'numpy'])
def test_the_energy_differentiates_in_the_centre(backend, stamp_kind):
    """``dU/dx`` within ``range=`` is finite and equals the central difference of NumPy runs."""
    x0, h = CENTER[0], 1e-5
    reference = (float(_energy_of_x(x0 + h, 'numpy', stamp_kind))
                 - float(_energy_of_x(x0 - h, 'numpy', stamp_kind))) / (2 * h)
    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        runtime('jax')                  # x64 lifted before the leaf is built
        got = float(jax.grad(lambda x: _energy_of_x(x, 'jax', stamp_kind))(jnp.asarray(x0)))
    else:
        import torch
        x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        _energy_of_x(x, 'torch', stamp_kind).backward()
        got = float(x.grad)
    assert math.isfinite(got) and got != 0.0, got
    assert math.isclose(got, reference, rel_tol=1e-5), (got, reference)
