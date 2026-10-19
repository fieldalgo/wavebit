"""Gate 2 of stage 3 -- the far field: a point dipole's pattern, its dual, its sphere, its size.

**The closed form, in these units** (``c = eps0 = mu0 = 1``, H stored as ``Z0 H``, so ``eta =
1``; ``e^{-iwt}``). A current element ``J dV = Il z_hat`` at ``r0`` radiates, far away,

    r e^{-ikr} E_theta = -i k eta Il sin(theta) e^{-ik r_hat.r0} / (4 pi),

and a magnetic current element ``M dV = Kl z_hat`` (the dual: ``E -> H, H -> -E, J -> M``)

    r e^{-ikr} E_phi   = +i k Kl sin(theta) e^{-ik r_hat.r0} / (4 pi),

both from ``src/wavebit/fdtd/_post.py``'s ``E_theta = (ik/4pi)(eta N_theta + L_phi)``, ``E_phi =
(ik/4pi)(eta N_phi - L_theta)`` with ``N = Il z_hat`` or ``L = Kl z_hat``. So the intensity is
``U = k^2 |Il|^2 sin^2(theta) / (32 pi^2)`` and the power ``k^2 |Il|^2 / (12 pi)``. A
``fd.Dipole``'s moment ``sum J dV`` is its drive ``w(t)`` (the stamp sums to 1), so ``Il`` at a
wavelength is the drive's phasor ``w^ = sum w(t_n) e^{i w t_n} dt`` at the current's own times
(``res.sources['d'].drive`` and ``.t``) -- no fitted constant anywhere.

**The scene** (``fd_helpers.farfield_dipole``): an open vacuum box of 1.4 wavelengths, a 12-cell
PML all round, the dipole at ``(0.013, -0.02, 0.007)`` (off every node, half node and symmetry
plane), a ``FarField`` box of side 0.5 around it, wavelengths 0.9 / 1.0 / 1.1 of a band centred on
1.0, 14 periods (settled by 10: the decay ratio is 6e-11 and below). 52^3 cells, 4 s on NumPy at
20 cells per wavelength; 26 s at 40, which is the slow tier.

**What is pinned, and the tolerances** (measured 2026-09-19, NumPy, the three wavelengths):

* **Pattern.** The ``Ez`` dipole's ``U(theta, phi)`` over a 37 x 24 grid (every 5 degrees in
  theta, 15 in phi) against ``max(U) sin^2(theta)``: 1.4e-2 / 9.9e-3 / 6.9e-3 of the peak at 20
  cells per wavelength, held to 2.5e-2. That is the Yee grid's anisotropy -- ``U(90, 45) /
  U(90, 0)`` is 0.987 -- which is ``O((k dx)^2)`` and falls 4x at 40. The dual ``Hz`` dipole: 3.5e-3,
  and ``Hx`` against ``1 - (r_hat.x_hat)^2``: 3.3e-3, both held to 1e-2. The cross polarisation
  (``E_phi`` of the electric dipole, ``E_theta`` of the ``Hz`` one) is 5e-7 of the peak intensity
  and held to 1e-5: the dual is a *different* field, not the same pattern.
* **The sphere.** ``U`` integrated over the sphere (Gauss-Legendre 48 x 64) is the box's net flux
  to 1.1e-3 / 1.3e-3 / 2.7e-3 (``Ez``) and 8.5e-3 / 6.1e-3 / 4.4e-3 (``Hz``, ``Hx``) at 20 cells
  per wavelength, held to 1.5e-2; the transform's discretisation (the face quadrature at the
  continuum ``k``), second order: 4x smaller at 40 (the slow test). The flux itself is the
  dipole's ``power()`` to 4e-8, held to 1e-6 (the PML's reflection floor; ``_post`` derives why
  both sides are exact phasor identities).
* **Absolute size and phase.** ``fields()`` against the closed form above, complex: the ratio is
  real and below 1 by 0.8-1.5% (``Ez``) and 1.1-1.9% (``Hz``) at 20 cells per wavelength --
  the grid's dispersion and the stamp's width, the same deficit the dipole's own power has
  against ``k^2 |Il|^2/(12 pi)`` (-2.1% at the centre) -- held to 3e-2 in modulus and 1e-2 in the
  imaginary part (measured 4e-3: a wrong ``e^{+-ik r_hat.r0}`` or time level would be 0.1 rad).
  A single resolution cannot pin a constant, so the slow test does: at 20 and 40 cells per
  wavelength the deviation falls 4.0x (held to 3.2-5), and the Richardson limit of the two is
  the closed form to 2e-4 (held to 2e-3).

**Mutants**, each judged by the same check functions the gates call:

* the equivalent magnetic current's sign flipped (``_post.FARFIELD_M_SIGN = -1``): the pattern
  misses ``sin^2`` by 27-38% of its own peak, the sphere misses the flux by 30-39% and the
  absolute field is off by 125-166%;
* one face's outward normal flipped (each of the six in turn, the result rebuilt with that
  face's sign negated): the pattern misses by 31-98%, the sphere by 18-36%, the field by
  42-155% (the z faces, which the ``z`` dipole's lobe hardly crosses, are the smallest);
* the retarded phase ``e^{-ik r_hat.r'}`` dropped (``_post._surface_transform`` called with
  ``r_hat = 0``): the sphere misses by 74-134% and the field by 93-141%, but the pattern only
  by 5e-3 to 8e-3, *inside* its tolerance -- a 0.5-wavelength box round a near-central dipole
  sums to nearly the point moment, so dropping the phase rescales the lobe without bending it.
  The pattern check alone is blind to that mutant; the sphere and absolute checks are why the
  gate has three.
"""
import math

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd import _post
from wavebit.fdtd._legacy_results import FarFieldResult
from tests.fixtures.fdtd.fd_helpers import WAVELENGTHS, farfield_dipole, sphere_integral, to_numpy

pytestmark = pytest.mark.filterwarnings('ignore::wavebit.fdtd._legacy.CoarseMeshWarning')

CENTER = np.array((0.013, -0.02, 0.007))
#: The pattern grid: every 5 degrees in theta, every 15 in phi.
THETA = np.linspace(0.0, 180.0, 37)
PHI = np.arange(0.0, 360.0, 15.0)
#: Where the complex field is compared: off the axis, where sin(theta) is not small.
FIELD_THETA = np.array((30.0, 60.0, 90.0, 120.0, 150.0))
FIELD_PHI = np.array((0.0, 70.0, 200.0))

PATTERN_TOL = {'Ez': 2.5e-2, 'Hz': 1e-2, 'Hx': 1e-2}
CROSS_TOL = 1e-5
SPHERE_TOL = 1.5e-2
FLUX_POWER_TOL = 1e-6
FIELD_TOL = 3e-2
PHASE_TOL = 1e-2


def _run(component, cells_per_wavelength=20):
    kind = 'electric' if component[0] == 'E' else 'magnetic'
    res = fd.run(farfield_dipole(component=component, kind=kind,
                                 cells_per_wavelength=cells_per_wavelength))
    assert res.diagnostics['decay']['ff'] < 1e-8, res.diagnostics['decay']
    return res


@pytest.fixture(scope='module')
def runs():
    return {c: _run(c) for c in ('Ez', 'Hz', 'Hx')}


def drive_phasor(res, wavelengths=WAVELENGTHS):
    """``(nf,)``: the dipole moment's phasor ``sum w(t_n) e^{i w t_n} dt`` at the drive's times."""
    src = res.sources['d']
    f = 1.0 / np.asarray(wavelengths)
    t = to_numpy(src.t)
    return (np.exp(1j * 2 * math.pi * f[:, None] * t[None, :]) @ to_numpy(src.drive)) * res.plan.dt


def _rhat(T, F):
    t, p = np.radians(T), np.radians(F)
    return np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def _shape(component, T, F):
    """``1 - (r_hat . d_hat)^2``: every dipole's pattern, ``d_hat`` along its component."""
    return 1.0 - _rhat(T, F)['xyz'.index(component[1])] ** 2


# ---------------------------------------------------------------------------------------------
# The check functions: the gates and the mutants go through these and nothing else
# ---------------------------------------------------------------------------------------------

def pattern_error(ff, component):
    """``(nf,)``: ``max |U - max(U) shape|`` over the grid, relative to ``max(U)``."""
    T, F = np.meshgrid(THETA, PHI, indexing='ij')
    U = to_numpy(ff.intensity(T, F))
    peak = U.max(axis=(1, 2))
    ref = peak[:, None, None] * _shape(component, T, F)[None]
    return np.max(np.abs(U - ref), axis=(1, 2)) / peak


def cross_polarization(ff, component):
    """``(nf,)``: the peak of the component the dipole must not radiate, over the peak ``U``."""
    T, F = np.meshgrid(THETA, PHI, indexing='ij')
    E_t, E_p = (to_numpy(a) for a in ff.fields(T, F))
    wrong = E_p if component == 'Ez' else E_t            # the Hz dual radiates E_phi only
    U = to_numpy(ff.intensity(T, F))
    return np.max(np.abs(wrong) ** 2 / 2.0, axis=(1, 2)) / U.max(axis=(1, 2))


def sphere_error(ff):
    """``(nf,)``: the sphere integral of ``U`` over the box's net flux, minus 1."""
    return sphere_integral(ff) / to_numpy(ff.flux) - 1.0


def field_ratio(res, component, ff=None):
    """``(nf, nt, np)``: ``fields()`` over the closed form of the module docstring (complex).

    ``ff`` is ``res['ff']`` unless given (a mutant's transform of the same run)."""
    ff = res['ff'] if ff is None else ff
    T, F = np.meshgrid(FIELD_THETA, FIELD_PHI, indexing='ij')
    E_t, E_p = (to_numpy(a) for a in ff.fields(T, F))
    k = 2 * math.pi / np.asarray(WAVELENGTHS)
    kk = k[:, None, None]
    retard = np.exp(-1j * kk * np.tensordot(CENTER, _rhat(T, F), 1)[None])
    base = 1j * kk * drive_phasor(res)[:, None, None] * np.sin(np.radians(T))[None] \
        / (4 * math.pi) * retard
    return E_t / (-base) if component == 'Ez' else E_p / base


def field_error(res, component, ff=None):
    """``(nf,)``: the largest ``|ratio - 1|`` of :func:`field_ratio`."""
    return np.abs(field_ratio(res, component, ff) - 1.0).max(axis=(1, 2))


def gates_pass(res, component, ff=None):
    """What the pattern, sphere and absolute-field gates below assert, as one verdict."""
    ff = res['ff'] if ff is None else ff
    return bool(np.all(pattern_error(ff, component) < PATTERN_TOL[component])
                and np.all(np.abs(sphere_error(ff)) < SPHERE_TOL)
                and np.all(field_error(res, component, ff) < FIELD_TOL))


# ---------------------------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------------------------

def test_electric_dipole_pattern_is_sin_squared(runs):
    """An ``Ez`` dipole: ``U`` is ``sin^2(theta)`` over the whole grid, and ``E_phi`` is absent."""
    ff = runs['Ez']['ff']
    err = pattern_error(ff, 'Ez')
    assert np.all(err < PATTERN_TOL['Ez']), err
    cross = cross_polarization(ff, 'Ez')
    assert np.all(cross < CROSS_TOL), cross


def test_the_grid_anisotropy_is_what_the_pattern_tolerance_covers(runs):
    """``U(90, 45)/U(90, 0)`` is the Yee grid's anisotropy (0.987 at the centre), below 1 and
    within ``PATTERN_TOL`` of it -- so the tolerance is the grid's, not the transform's."""
    ff = runs['Ez']['ff']
    ratio = to_numpy(ff.intensity(90.0, 45.0)) / to_numpy(ff.intensity(90.0, 0.0))
    assert np.all(ratio < 1.0) and np.all(ratio > 1.0 - PATTERN_TOL['Ez']), ratio


@pytest.mark.parametrize('component', ('Hz', 'Hx'))
def test_magnetic_dipole_radiates_the_dual_pattern(runs, component):
    """A magnetic dipole: ``1 - (r_hat.d_hat)^2`` about its own axis; ``Hz`` radiates ``E_phi``
    only, where the electric ``Ez`` radiates ``E_theta`` only."""
    ff = runs[component]['ff']
    err = pattern_error(ff, component)
    assert np.all(err < PATTERN_TOL[component]), err
    if component == 'Hz':
        cross = cross_polarization(ff, 'Hz')
        assert np.all(cross < CROSS_TOL), cross


@pytest.mark.parametrize('component', ('Ez', 'Hz', 'Hx'))
def test_sphere_integral_equals_box_flux_and_dipole_power(runs, component):
    """``int U dOmega`` = the FarField box's flux = the dipole's ``power()``."""
    res = runs[component]
    ff = res['ff']
    power = to_numpy(res.sources['d'].power())
    flux = to_numpy(ff.flux)
    assert np.all(np.abs(flux / power - 1.0) < FLUX_POWER_TOL), flux / power - 1.0
    err = sphere_error(ff)
    assert np.all(np.abs(err) < SPHERE_TOL), err
    assert np.all(np.abs(sphere_integral(ff) / power - 1.0) < SPHERE_TOL)


@pytest.mark.parametrize('component', ('Ez', 'Hz'))
def test_absolute_far_field_is_the_current_elements(runs, component):
    """``r e^{-ikr} E`` against ``-+ i k Il sin(theta) e^{-ik r_hat.r0}/(4 pi)``, complex: the
    grid's deficit (a ratio just below 1) and nothing else -- no phase, no stray constant."""
    ratio = field_ratio(runs[component], component)
    assert np.all(np.abs(ratio - 1.0) < FIELD_TOL), np.abs(ratio - 1.0).max()
    assert np.all(ratio.real < 1.0), ratio.real.max()
    assert np.all(np.abs(ratio.imag) < PHASE_TOL), np.abs(ratio.imag).max()
    # And the intensity: |ratio|^2 of the closed-form U = k^2 |Il|^2 sin^2 / (32 pi^2).
    res = runs[component]
    T, F = np.meshgrid(FIELD_THETA, FIELD_PHI, indexing='ij')
    k = 2 * math.pi / np.asarray(WAVELENGTHS)
    Uc = (k ** 2 * np.abs(drive_phasor(res)) ** 2)[:, None, None] \
        * np.sin(np.radians(T))[None] ** 2 / (32 * math.pi ** 2)
    U = to_numpy(res['ff'].intensity(T, F))
    # (to the cross polarisation's 5e-7, which U holds and the ratio does not).
    np.testing.assert_allclose(U / Uc, np.abs(ratio) ** 2, rtol=1e-5)


@pytest.mark.slow
def test_far_field_converges_to_the_closed_form_at_second_order():
    """20 and 40 cells per wavelength: the field's deviation and the sphere's miss fall 4x, and
    the Richardson limit of the field ratio is the closed form -- which pins the constant."""
    for component in ('Ez', 'Hz'):
        coarse, fine = _run(component, 20), _run(component, 40)
        r20, r40 = field_ratio(coarse, component), field_ratio(fine, component)
        fall = np.abs(r20 - 1.0).max(axis=(1, 2)) / np.abs(r40 - 1.0).max(axis=(1, 2))
        assert np.all((fall > 3.2) & (fall < 5.0)), (component, fall)
        limit = (4.0 * r40 - r20) / 3.0
        assert np.all(np.abs(limit - 1.0) < 2e-3), (component, np.abs(limit - 1.0).max())
        s_fall = np.abs(sphere_error(coarse['ff'])) / np.abs(sphere_error(fine['ff']))
        assert np.all((s_fall > 3.2) & (s_fall < 5.0)), (component, s_fall)


# ---------------------------------------------------------------------------------------------
# The mutants
# ---------------------------------------------------------------------------------------------

def test_the_gates_pass_the_unmutated_transform(runs):
    """The control for every mutant below: the same verdict function, on the real transform."""
    assert gates_pass(runs['Ez'], 'Ez')
    assert gates_pass(runs['Hz'], 'Hz')


def test_mutant_magnetic_current_sign_flipped_is_caught(runs, monkeypatch):
    """``M = +n x E``: every check fails -- pattern, sphere and absolute field."""
    monkeypatch.setattr(_post, 'FARFIELD_M_SIGN', -1.0)
    for component in ('Ez', 'Hz'):
        res = runs[component]
        assert np.all(pattern_error(res['ff'], component) > 0.2), component
        assert np.all(np.abs(sphere_error(res['ff'])) > 0.2), component
        assert np.all(field_error(res, component) > 1.0), component
        assert not gates_pass(res, component)


def _one_face_flipped(ff, index):
    """``ff`` rebuilt with face ``index``'s outward sign negated in the transform only (the
    box's ``flux`` is carried over untouched: the mutant is the transform's normal)."""
    rt, accs, faces, grid, _freqs, eps, mu = ff._args
    faces = list(faces)
    name, sign, layout = faces[index]
    faces[index] = (name, -sign, layout)
    return FarFieldResult(ff.name, ff.wavelengths, rt, accs, tuple(faces), grid, eps, mu,
                          ff.flux, ff.faces), name


@pytest.mark.parametrize('component', ('Ez', 'Hz'))
def test_mutant_one_face_normal_flipped_is_caught(runs, component):
    """Each of the six faces in turn: every check fails."""
    res = runs[component]
    names = []
    for index in range(len(res['ff']._args[2])):
        bad, name = _one_face_flipped(res['ff'], index)
        names.append(name)
        assert np.all(pattern_error(bad, component) > 0.2), name
        assert np.all(np.abs(sphere_error(bad)) > 0.1), name
        assert np.all(field_error(res, component, bad) > 0.3), name
        assert not gates_pass(res, component, bad), name
    assert sorted(names) == ['x+', 'x-', 'y+', 'y-', 'z+', 'z-'], names


def test_mutant_retarded_phase_dropped_is_caught(runs, monkeypatch):
    """``e^{-ik r_hat.r'} -> 1``: the sphere and the absolute field fail; the pattern *cannot*
    see it (a 0.5-wavelength box round a near-central dipole: dropping the phase rescales the
    lobe and hardly bends it, 5e-3 to 8e-3), which is why the gate has all three checks."""
    original = _post._surface_transform

    def no_retardation(rt, F, us, vs, wu, wv, k, ru, rv, rp, p, dx):
        return original(rt, F, us, vs, wu, wv, k, 0.0 * ru, 0.0 * rv, 0.0 * rp, p, dx)
    monkeypatch.setattr(_post, '_surface_transform', no_retardation)
    for component in ('Ez', 'Hz'):
        res = runs[component]
        assert np.all(pattern_error(res['ff'], component) < PATTERN_TOL[component]), component
        assert np.all(np.abs(sphere_error(res['ff'])) > 0.5), component
        assert np.all(field_error(res, component) > 0.5), component
        assert not gates_pass(res, component)
