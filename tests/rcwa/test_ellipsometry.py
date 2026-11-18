"""``rcwa.run_ellipsometry``: Jones, Psi/Delta and Mueller outputs, pinned to closed forms.

The conventions under test are stated in ``src/wavebit/ellipsometry.py``: the ``(s, p)`` Jones
basis indexed ``J[out, in]``, the reflected ``p_hat`` that makes ``r_p = -r_s`` at normal
incidence (Nebraska), and ``delta`` reported in the instrument convention,
``tan(psi) exp(i delta) = conj(j_pp / j_ss)``. Each check here is against something outside the
solver: the Airy recursion of ``test_lossy_media``, the Fresnel coefficients written out
below, the RETICOLO efficiencies frozen in ``golden_reticolo.npz``, or the algebraic identities
a pure Mueller matrix satisfies.

Angles are compared through ``exp(i delta)`` rather than ``delta``: a lossless stack has
``delta`` exactly ``0`` or ``pi``, and rounding may put it on either side of ``0 = 2 pi``.
"""
import sys as _repository_sys
from contextlib import nullcontext
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path

import numpy as np
import pytest

from wavebit import Material, Units, rcwa, eps_from_refractive_index
from wavebit.ellipsometry import cloude_eigenvalues, depolarization_index, mueller_from_jones, psi_delta
from wavebit.errors import ValidationError
from tests.rcwa.test_lossy_media import analytic_r

# 633 nm values, held constant over the short wavelength lists below: the oracle takes an index.
N_SIO2, N_SI, N_AU = 1.457, 3.882 + 0.019j, 0.197 + 3.09j
WAVELENGTHS = (600., 633., 700.)


def _backend(name):
    """Import the backend or skip, and the scope its double-precision run needs."""
    if name == 'numpy':
        return nullcontext()
    module = pytest.importorskip(name)
    return module.enable_x64(True) if name == 'jax' else nullcontext()


def stack(ns, thicknesses, *, theta=np.radians(70.), phi=0., wavelength=WAVELENGTHS, side='top',
          polarization='s', period=(300., 300.)):
    """``ns = [n_top, *layers, n_bottom]`` as ``analytic_r`` takes it; thicknesses top-down."""
    layers = tuple(rcwa.Layer(thickness=d, background=Material(epsilon=eps_from_refractive_index(n)))
                   for n, d in zip(ns[1:-1][::-1], thicknesses[::-1]))
    return rcwa.Problem(units=Units(length='nm'),
                        stack=rcwa.Stack(period=period, top=Material(epsilon=eps_from_refractive_index(ns[0])),
                                         bottom=Material(epsilon=eps_from_refractive_index(ns[-1])),
                                         layers=layers),
                        excitation=rcwa.Excitation(wavelength=wavelength,
                                                   sources=(rcwa.PlaneWave(side=side, polarization=polarization,
                                                                           theta=theta, phi=phi),)))


UNIFORM = rcwa.Numerics(orders=(0, 0))


def analytic_t(ns, thicknesses, wavelength, theta, tetm):
    """Transmitted field amplitude of a uniform stack, Nebraska Fresnel coefficients.

    The same recursion as :func:`analytic_r`, carried for ``t`` as well: ``t_p`` is the
    *electric* field ratio ``2 n_i cos_i / (n_j cos_i + n_i cos_j)``, which equals ``t_s`` at
    normal incidence -- the one statement the solver's transmitted ``p_hat`` makes.
    """
    ns = [complex(n) for n in ns]
    sin_t = ns[0] * np.sin(theta)
    cos = [np.sqrt(1 - (sin_t / n) ** 2 + 0j) for n in ns]
    cos = [c if (n * c).imag >= 0 else -c for n, c in zip(ns, cos)]

    def r_t(i, j):
        ni, nj, ci, cj = ns[i], ns[j], cos[i], cos[j]
        if tetm == 'te':
            return (ni * ci - nj * cj) / (ni * ci + nj * cj), 2 * ni * ci / (ni * ci + nj * cj)
        return (nj * ci - ni * cj) / (nj * ci + ni * cj), 2 * ni * ci / (nj * ci + ni * cj)

    r, t = r_t(len(ns) - 2, len(ns) - 1)
    for i in range(len(ns) - 2, 0, -1):
        beta = 2 * np.pi / wavelength * ns[i] * cos[i] * thicknesses[i - 1]
        r_up, t_up = r_t(i - 1, i)
        phase = np.exp(1j * beta)
        denominator = 1 + r_up * r * phase ** 2
        r, t = (r_up + r * phase ** 2) / denominator, t_up * t * phase / denominator
    return r, t


# ------------------------------------------------------------------ (a) closed-form films

@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_film_jones_matches_the_airy_recursion(backend):
    """The two diagonal entries are the Airy ``r_s`` and ``r_p``; the off-diagonal ones are zero."""
    ns, ds = [1., N_SIO2, N_SI], [100.]
    with _backend(backend):
        result = rcwa.run_ellipsometry(stack(ns, ds), numerics=UNIFORM, backend=backend)
    jones = np.asarray(result.reflection.jones())
    assert jones.shape == (3, 2, 2)
    for i, wavelength in enumerate(WAVELENGTHS):
        np.testing.assert_allclose(jones[i, 0, 0], analytic_r(ns, ds, wavelength, np.radians(70.), 'te'), rtol=0, atol=1e-12)
        np.testing.assert_allclose(jones[i, 1, 1], analytic_r(ns, ds, wavelength, np.radians(70.), 'tm'), rtol=0, atol=1e-12)
    np.testing.assert_allclose(jones[:, 0, 1], 0, atol=1e-14)
    np.testing.assert_allclose(jones[:, 1, 0], 0, atol=1e-14)


@pytest.mark.parametrize('ns,ds', [([1., 1.5, 2.2, N_AU], [20., 35.]),
                                   ([1., N_SI, N_SIO2, 1.8, N_SI], [30., 120., 45.])])
def test_multilayer_reflection_and_transmission_match_the_closed_form(ns, ds):
    """Several layers, absorbing ones included, in reflection and transmission at 1e-12."""
    result = rcwa.run_ellipsometry(stack(ns, ds), numerics=UNIFORM)
    reflected, transmitted = result.reflection.jones(), result.transmission.jones()
    for i, wavelength in enumerate(WAVELENGTHS):
        for k, tetm in enumerate(('te', 'tm')):
            r, t = analytic_t(ns, ds, wavelength, np.radians(70.), tetm)
            np.testing.assert_allclose(reflected[i, k, k], r, rtol=0, atol=1e-12)
            np.testing.assert_allclose(transmitted[i, k, k], t, rtol=0, atol=1e-12)


def test_psi_delta_of_a_film_match_the_closed_form_in_the_instrument_convention():
    """``psi = atan|r_p / r_s|`` and ``exp(i delta) = conj(r_p / r_s) / |r_p / r_s|``."""
    ns, ds = [1., N_SIO2, N_SI], [100.]
    angles = rcwa.run_ellipsometry(stack(ns, ds), numerics=UNIFORM).reflection.ellipsometry()
    for i, wavelength in enumerate(WAVELENGTHS):
        rho = analytic_r(ns, ds, wavelength, np.radians(70.), 'tm') / analytic_r(ns, ds, wavelength, np.radians(70.), 'te')
        np.testing.assert_allclose(angles.psi[i], np.arctan(abs(rho)), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.exp(1j * angles.delta[i]), np.conj(rho) / abs(rho), rtol=0, atol=1e-12)
        np.testing.assert_allclose(angles.rho[i], rho, rtol=0, atol=1e-12)
    assert np.all(angles.delta >= 0) and np.all(angles.delta < 2 * np.pi)
    # 633 nm, 100 nm SiO2 on Si at 70 deg: the textbook (41.0 deg, 79.8 deg) in this convention.
    np.testing.assert_allclose(np.degrees([angles.psi[1], angles.delta[1]]), [41.04, 79.79], atol=.01)


# ------------------------------------------------------------- (b) Brewster and a metal

def test_psi_vanishes_at_the_brewster_angle_and_delta_steps_through_it():
    """A bare lossless interface: ``r_p = 0`` at ``atan(n)``, ``delta = pi`` below it, ``0`` above."""
    brewster = np.arctan(N_SIO2)
    at = rcwa.run_ellipsometry(stack([1., N_SIO2], [], theta=brewster, wavelength=633.), numerics=UNIFORM)
    assert float(at.reflection.ellipsometry().psi[0]) < 1e-12
    below = rcwa.run_ellipsometry(stack([1., N_SIO2], [], theta=brewster - .05, wavelength=633.), numerics=UNIFORM)
    above = rcwa.run_ellipsometry(stack([1., N_SIO2], [], theta=brewster + .05, wavelength=633.), numerics=UNIFORM)
    np.testing.assert_allclose(np.exp(1j * below.reflection.ellipsometry().delta[0]), -1, atol=1e-12)
    np.testing.assert_allclose(np.exp(1j * above.reflection.ellipsometry().delta[0]), 1, atol=1e-12)


@pytest.mark.parametrize('n,theta_deg,psi_deg,delta_deg', [
    (N_AU, 70., 43.371, 102.831),     # gold at 633 nm, 70 deg
    (N_AU, 45., 44.265, 154.932),
    (N_SI, 70., 10.573, 179.230),     # silicon, 70 deg: delta just under 180 deg
])
def test_a_metal_substrate_gives_the_literature_shaped_angles(n, theta_deg, psi_deg, delta_deg):
    """An absorber has ``0 < delta < pi`` in the instrument convention, falling from ``pi``.

    The numbers are the bare-interface Fresnel values of the stated constant indices, rounded
    to the ``0.001 deg`` of an ellipsometry table.
    """
    result = rcwa.run_ellipsometry(stack([1., n], [], theta=np.radians(theta_deg), wavelength=633.), numerics=UNIFORM)
    angles = result.reflection.ellipsometry()
    np.testing.assert_allclose(np.degrees([angles.psi[0], angles.delta[0]]), [psi_deg, delta_deg], atol=5e-4)
    assert 0 < float(angles.delta[0]) < np.pi


# ------------------------------------------------------------- (c) Mueller physicality

def grating(phi=.4, wavelength=(600., 633., 700.), side='top'):
    """A 2D crossed grating lit off its symmetry planes, so s and p mix in every order."""
    return rcwa.Problem(units=Units(length='nm'),
                        stack=rcwa.Stack(period=(500., 400.), bottom=Material(epsilon=eps_from_refractive_index(N_SI)),
                                         layers=(rcwa.Layer(thickness=150., background=Material(epsilon=1.), regions=(
                                             rcwa.Region(geometry=rcwa.Rectangle(size=(220., 160.), center=(40., -30.)),
                                                         material=Material(epsilon=eps_from_refractive_index(N_SIO2) + .5)),)),)),
                        excitation=rcwa.Excitation(wavelength=wavelength,
                                                   sources=(rcwa.PlaneWave(side=side, theta=.35, phi=phi),)))


GRATING = rcwa.Numerics(orders=(3, 2))


def test_a_jones_derived_mueller_matrix_is_pure_on_every_order():
    """Depolarization index 1 and a rank-one coherency matrix with a nonnegative spectrum."""
    result = rcwa.run_ellipsometry(grating(), numerics=GRATING)
    for port in (result.reflection, result.transmission):
        jones = port.jones(None)
        assert jones.shape == (3, 5, 7, 2, 2)
        assert np.abs(jones[..., 0, 1]).max() > 1e-3, 'premise: the grating cross-polarizes'
        mueller = port.mueller(None)
        np.testing.assert_allclose(depolarization_index(mueller), 1., rtol=0, atol=1e-11)
        eigenvalues = cloude_eigenvalues(mueller)
        assert eigenvalues.min() > -1e-12
        np.testing.assert_allclose(eigenvalues[..., -1], 1., rtol=0, atol=1e-11)
        np.testing.assert_allclose(mueller[..., 0, 0], 1.)


def test_an_isotropic_film_gives_the_standard_mueller_form():
    """``[[1, -N, 0, 0], [-N, 1, 0, 0], [0, 0, C, S], [0, 0, -S, C]]`` with ``S = sin 2psi sin delta``."""
    port = rcwa.run_ellipsometry(stack([1., N_SIO2, N_SI], [100.]), numerics=UNIFORM).reflection
    angles, mueller = port.ellipsometry(), port.mueller()
    for i in range(3):
        n, c, s = np.cos(2 * angles.psi[i]), np.sin(2 * angles.psi[i]) * np.cos(angles.delta[i]), np.sin(2 * angles.psi[i]) * np.sin(angles.delta[i])
        want = np.array([[1, -n, 0, 0], [-n, 1, 0, 0], [0, 0, c, s], [0, 0, -s, c]])
        np.testing.assert_allclose(mueller[i], want, rtol=0, atol=1e-13)
    # The unnormalized m_11 is the unpolarized reflectance.
    raw = port.mueller(normalize=False)
    np.testing.assert_allclose(raw[:, 0, 0], np.asarray(port.s.efficiency[:, 0, 0] + port.p.efficiency[:, 0, 0]) / 2, rtol=0, atol=1e-13)


# ------------------------------------------------------------ (d) a stored grating oracle

GOLDEN = repository_path('tests/references/golden_reticolo.npz')

# The scenes of ``tests/references/test_reticolo_golden.py``, in the current API; permittivity
# ``9`` on five of ten pixels, a 150 nm layer on glass, lit at 0.2 rad: pure TE/TM at ``phi = 0``
# and conical at ``phi = 0.3``, where s and p mix. Lengths in micrometres as the golden has them.
_HALF = np.array([[9.] * 5 + [1.] * 5])


def reticolo_scene(phi):
    return rcwa.Problem(units=Units(length='um'),
                        stack=rcwa.Stack(period=(.7, .7), bottom=Material(epsilon=2.25),
                                         layers=(rcwa.Layer(thickness=.15, epsilon=_HALF),)),
                        excitation=rcwa.Excitation(wavelength=.9, sources=(rcwa.PlaneWave(theta=.2, phi=phi),)))


@pytest.mark.parametrize('phi,s_case,p_case', [(0., 'te_oblique', 'tm_oblique'), (.3, 'conical_te', 'conical_tm')])
def test_the_zeroth_order_jones_columns_reproduce_reticolo_efficiencies(phi, s_case, p_case):
    """``|j_ss|^2 + |j_ps|^2`` is the s-incidence specular reflectance RETICOLO froze, and the
    p column likewise: reflected into the incidence medium, the zeroth order's efficiency is
    exactly the squared amplitude ratios summed over both outgoing polarizations."""
    if not GOLDEN.exists():
        pytest.skip(f'{GOLDEN.name} is missing')
    golden = np.load(GOLDEN)
    jones = rcwa.run_ellipsometry(reticolo_scene(phi), numerics=rcwa.Numerics(orders=(10, 0), fourier_coefficients='integrated')).reflection.jones()
    for column, case in ((0, s_case), (1, p_case)):
        specular = list(golden[case + '__idx_ri']).index(10)      # order index 10 is m = 0
        np.testing.assert_allclose(np.sum(np.abs(jones[0, :, column]) ** 2), golden[case + '__de_ri'][specular], rtol=0, atol=1e-11)
    if phi:
        assert np.abs(jones[0, 1, 0]) > 1e-2, 'premise: conical incidence cross-polarizes'


# ---------------------------------------------- (e) backends agree, and AD matches FD

@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_the_backends_agree_on_jones_angles_and_mueller(backend):
    reference = rcwa.run_ellipsometry(grating(), numerics=GRATING)
    with _backend(backend):
        other = rcwa.run_ellipsometry(grating(), numerics=GRATING, backend=backend)
        for port, port_ref in ((other.reflection, reference.reflection), (other.transmission, reference.transmission)):
            np.testing.assert_allclose(np.asarray(port.jones(None)), port_ref.jones(None), rtol=0, atol=1e-11)
            # Unnormalized: dividing an evanescent order's entries by its tiny m_11 amplifies round-off.
            np.testing.assert_allclose(np.asarray(port.mueller(None, normalize=False)), port_ref.mueller(None, normalize=False), rtol=0, atol=1e-11)
            angles, angles_ref = port.ellipsometry(), port_ref.ellipsometry()
            np.testing.assert_allclose(np.asarray(angles.psi), angles_ref.psi, rtol=0, atol=1e-11)
            np.testing.assert_allclose(np.exp(1j * np.asarray(angles.delta)), np.exp(1j * angles_ref.delta), rtol=0, atol=1e-11)


def _film(thickness, wavelength=633.):
    return stack([1., N_SIO2, N_SI], [thickness], wavelength=wavelength)


def _finite_difference(read, thickness=100., step=1e-3):
    def value(h):
        return float(read(rcwa.run_ellipsometry(_film(h), numerics=UNIFORM)))
    return (value(thickness + step) - value(thickness - step)) / (2 * step)


_READINGS = {'psi': lambda r: r.reflection.ellipsometry().psi[0],
             'delta': lambda r: r.reflection.ellipsometry().delta[0],
             'm34': lambda r: r.reflection.mueller()[0, 2, 3]}


@pytest.mark.parametrize('reading', list(_READINGS))
def test_jax_gradients_of_the_angles_match_finite_differences(reading):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    read = _READINGS[reading]
    with jax.enable_x64(True):
        def value(h):
            return read(rcwa.run_ellipsometry(_film(h), numerics=UNIFORM, backend='jax'))
        derivative = float(jax.grad(value)(jnp.asarray(100.)))
    np.testing.assert_allclose(derivative, _finite_difference(read), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize('reading', list(_READINGS))
def test_torch_gradients_of_the_angles_match_finite_differences(reading):
    torch = pytest.importorskip('torch')
    read = _READINGS[reading]
    thickness = torch.tensor(100., dtype=torch.float64, requires_grad=True)
    value = read(rcwa.run_ellipsometry(_film(thickness), numerics=UNIFORM, backend='torch'))
    derivative = float(torch.autograd.grad(value, thickness)[0])
    np.testing.assert_allclose(derivative, _finite_difference(read), rtol=1e-6, atol=1e-12)


# --------------------------------------------------------------- (f) the sign conventions

def test_normal_incidence_pins_the_reflected_p_axis_and_the_transmitted_one():
    """Isotropic, normal incidence: ``r_p = -r_s`` (``delta = pi``) and ``t_p = t_s`` (``delta = 0``).

    This is the Nebraska choice of the reflected ``p_hat``, and the whole reason an ellipsometer's
    ``delta`` of a bare dielectric reads 180 deg and not 0.
    """
    for side in ('top', 'bottom'):
        result = rcwa.run_ellipsometry(stack([1., N_SIO2, N_SI], [100.], theta=0., side=side), numerics=UNIFORM)
        reflected, transmitted = result.reflection.jones(), result.transmission.jones()
        np.testing.assert_allclose(reflected[:, 1, 1], -reflected[:, 0, 0], rtol=0, atol=1e-13)
        np.testing.assert_allclose(transmitted[:, 1, 1], transmitted[:, 0, 0], rtol=0, atol=1e-13)
        np.testing.assert_allclose(np.exp(1j * result.reflection.ellipsometry().delta), -1, atol=1e-12)
        np.testing.assert_allclose(np.exp(1j * result.transmission.ellipsometry().delta), 1, atol=1e-12)
        np.testing.assert_allclose(result.reflection.ellipsometry().psi, np.pi / 4, atol=1e-12)


def test_the_jones_matrix_is_indexed_out_in_in_the_s_p_basis():
    """Column 0 is the s-polarized solve's ``(amplitude_ratio_s, amplitude_ratio_p)``, column 1 the
    p-polarized one's, untouched: ``j_ps = J[1, 0]`` is the p amplitude for s incidence."""
    result = rcwa.run_ellipsometry(grating(), numerics=GRATING)
    jones = result.reflection.jones(None)
    np.testing.assert_array_equal(jones[..., 0, 0], result.s.reflection.amplitude_ratio_s)
    np.testing.assert_array_equal(jones[..., 1, 0], result.s.reflection.amplitude_ratio_p)
    np.testing.assert_array_equal(jones[..., 0, 1], result.p.reflection.amplitude_ratio_s)
    np.testing.assert_array_equal(jones[..., 1, 1], result.p.reflection.amplitude_ratio_p)
    assert not np.allclose(jones[..., 1, 0], jones[..., 0, 1]), 'premise: the two cross terms differ'
    # One order, by its labels, is the same slice of the full table.
    j, i = result.reflection.orders_v.index(-1), result.reflection.orders_u.index(2)
    np.testing.assert_array_equal(result.reflection.jones((2, -1)), jones[:, j, i])
    with pytest.raises(KeyError):
        result.reflection.jones((9, 0))


def test_the_delta_convention_is_the_conjugate_of_the_native_ratio():
    """``rho`` keeps the ``exp(-i omega t)`` phasor; ``delta`` is minus its argument, in ``[0, 2 pi)``."""
    angles = rcwa.run_ellipsometry(grating(), numerics=GRATING).reflection.ellipsometry(None)
    np.testing.assert_allclose(angles.rho, angles.rho_ps * 0 + angles.rho)       # shape check
    np.testing.assert_allclose(np.exp(1j * angles.delta), np.conj(angles.rho) / np.abs(angles.rho), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.exp(1j * angles.delta_sp), np.conj(angles.rho_sp) / np.abs(angles.rho_sp), rtol=0, atol=1e-12)
    for delta in (angles.delta, angles.delta_ps, angles.delta_sp):
        assert np.all(delta >= 0) and np.all(delta < 2 * np.pi)
    assert np.abs(angles.rho_sp).max() > 1e-2 and np.abs(angles.rho_ps).max() > 1e-2


def test_generalized_ratios_vanish_on_an_isotropic_stack_and_not_on_a_rotated_crystal():
    isotropic = rcwa.run_ellipsometry(stack([1., N_SIO2, N_SI], [100.]), numerics=UNIFORM).reflection.ellipsometry()
    np.testing.assert_allclose(isotropic.rho_ps, 0, atol=1e-14)
    np.testing.assert_allclose(isotropic.rho_sp, 0, atol=1e-14)
    crystal = Material(epsilon=(2.25, 4., 2.25)).rotated(np.radians(30.), axis='z')
    problem = rcwa.Problem(units=Units(length='nm'),
                           stack=rcwa.Stack(period=(300., 300.), layers=(rcwa.Layer(thickness=220., background=crystal),)),
                           excitation=rcwa.Excitation(wavelength=600., sources=(rcwa.PlaneWave(theta=.3),)))
    anisotropic = rcwa.run_ellipsometry(problem, numerics=UNIFORM).reflection.ellipsometry()
    assert float(np.abs(anisotropic.rho_ps[0])) > 1e-2 and float(np.abs(anisotropic.rho_sp[0])) > 1e-2


# ----------------------------------------------------------------- the result's surface

def test_the_result_carries_both_plain_results_and_names_its_conventions():
    result = rcwa.run_ellipsometry(stack([1., N_SIO2, N_SI], [100.]), numerics=UNIFORM)
    assert isinstance(result.s, rcwa.Result) and isinstance(result.p, rcwa.Result)
    np.testing.assert_array_equal(result.wavelengths, WAVELENGTHS)
    assert result.reflection.s is result.s.reflection and result.reflection.p is result.p.reflection
    assert result.transmission.orders_u == (0,) and result.transmission.orders_v == (0,)
    reference = result.reflection.reference
    assert reference['wave'] == 'reflected' and reference['time_convention'] == 'exp(-i*omega*t)'
    for key in ('jones_basis', 'delta_convention', 'mueller_basis'):
        assert key in reference
    assert 'conj(j_pp / j_ss)' in reference['delta_convention']
    # The plain results are the pure-state solves, with everything a single run has.
    np.testing.assert_allclose(result.s.R.sum(axis=(1, 2)) + result.s.T.sum(axis=(1, 2)) + result.s.absorption, 1., atol=1e-12)


def test_the_two_solves_share_one_preparation(monkeypatch):
    from wavebit.rcwa import _lifecycle, _ellipsometry
    calls = []
    original = _lifecycle.prepare

    def counting(problem, **kw):
        calls.append(problem)
        return original(problem, **kw)
    monkeypatch.setattr(_lifecycle, 'prepare', counting)
    rcwa.run_ellipsometry(stack([1., N_SIO2, N_SI], [100.]), numerics=UNIFORM)
    assert len(calls) == 1
    assert calls[0].excitation.sources[0].polarization == _ellipsometry._S_PAIR


def test_a_pair_two_sources_or_no_diffraction_are_refused_by_name():
    problem = stack([1., N_SIO2, N_SI], [100.])
    pair = problem.replace(excitation=problem.excitation.replace(sources=(problem.excitation.sources[0].replace(polarization=(1., 1j)),)))
    with pytest.raises(ValidationError, match='polarization') as info:
        rcwa.run_ellipsometry(pair, numerics=UNIFORM)
    assert info.value.code == 'ellipsometry.polarization'
    two = problem.replace(excitation=problem.excitation.replace(sources=problem.excitation.sources * 2))
    with pytest.raises(ValidationError) as info:
        rcwa.run_ellipsometry(two, numerics=UNIFORM)
    assert info.value.code == 'ellipsometry.source'
    silent = problem.replace(outputs=rcwa.Outputs(diffraction=False, power_region='cell'))
    with pytest.raises(ValidationError) as info:
        rcwa.run_ellipsometry(silent, numerics=UNIFORM)
    assert info.value.code == 'ellipsometry.outputs'


def test_the_pure_functions_take_the_assembled_jones_matrix_directly():
    port = rcwa.run_ellipsometry(grating(), numerics=GRATING).reflection
    jones = port.jones(None)
    np.testing.assert_array_equal(psi_delta(jones).psi, port.ellipsometry(None).psi)
    np.testing.assert_array_equal(mueller_from_jones(jones), port.mueller(None))


# ------------------------------------------------------------------------ the example

def test_the_film_example_runs_and_keeps_its_own_checks():
    import importlib.util
    path = repository_path('examples/rcwa/ellipsometry_film.py')
    spec = importlib.util.spec_from_file_location('ellipsometry_film_example', path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    angles = example.main()
    assert angles.psi.shape == example.WAVELENGTHS.shape
    assert np.all(angles.delta >= 0) and np.all(angles.delta < 2 * np.pi)
