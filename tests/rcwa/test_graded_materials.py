"""Materials that change with height: ``rcwa.Graded`` and Bruggeman films in a ``ProfiledLayer``.

The pre-registered criteria (2026-11-24), each held by a test below:

(e) In the fine-slice limit the solver agrees with closed forms to 1e-6: a film whose
    permittivity is linear in height against the exact Airy-function solution of
    ``E'' + k0^2 (eps(z) - sin^2(theta) eps_c) E = 0``; a Bruggeman roughness layer, which is
    uniform, against the thin-film formula to 1e-12; a graded Bruggeman interface layer against
    an independent characteristic-matrix calculation with 20 000 sublayers.
(c) Automatic derivatives with respect to a roughness or interface thickness and a volume
    fraction agree with central differences on the same preparation to 1e-6, relative.
(f) NumPy, PyTorch and JAX agree to 1e-12.

The closed forms are written here, from M. Born and E. Wolf, *Principles of Optics*, sec. 1.6
(the characteristic matrix of a stratified medium), and an independent Bruggeman root from
``numpy.roots``; none of them calls the solver.
"""
import math

import numpy as np
import pytest

from wavebit import Material, Units, ValidationError, rcwa
from wavebit.materials import Bruggeman

WAVELENGTH = 0.6
SUBSTRATE, COVER = 2.1025, 1.0
SILICON = 15.2 + 0.16j


def _problem(layers, theta=0., polarization='s'):
    return rcwa.Problem(units=Units(length='um'),
                        stack=rcwa.Stack(period=(1., 1.), bottom=Material(epsilon=SUBSTRATE), top=Material(epsilon=COVER),
                                         layers=tuple(layers)),
                        excitation=rcwa.Excitation(wavelength=WAVELENGTH, sources=(
                            rcwa.PlaneWave(theta=theta, phi=0., polarization=polarization),)))


def _numerics(count=None):
    slicing = rcwa.Slicing(count=count, estimate=None) if count else None
    return rcwa.Numerics(orders=(0, 0), slicing=slicing)


def _reflectance(result):
    return float(np.sum(np.asarray(result.R)))


def _bruggeman(host, inclusion, fraction):
    b = (2 - 3*fraction)*host + (3*fraction - 1)*inclusion
    roots = np.roots([2, -b, -host*inclusion])
    return max(roots, key=lambda r: (round(r.imag, 12), r.real))


def _characteristic_matrix_reflectance(epsilons, thicknesses, theta):
    """TE reflectance of a stack of uniform films, light from the cover (Born and Wolf, 1.6)."""
    k0 = 2*math.pi/WAVELENGTH
    kx2 = COVER*math.sin(theta)**2
    def q(eps):
        root = np.sqrt(complex(eps - kx2))
        return root if root.imag >= 0 else -root
    m = np.eye(2, dtype=complex)
    for eps, d in zip(epsilons[::-1], thicknesses[::-1]):            # cover side first
        qj = q(eps)
        delta = k0*qj*d
        m = m @ np.array([[np.cos(delta), -1j*np.sin(delta)/qj], [-1j*qj*np.sin(delta), np.cos(delta)]])
    qc, qs = q(COVER), q(SUBSTRATE)
    r = ((m[0, 0] + m[0, 1]*qs)*qc - (m[1, 0] + m[1, 1]*qs))/((m[0, 0] + m[0, 1]*qs)*qc + (m[1, 0] + m[1, 1]*qs))
    return abs(r)**2


def _airy_reflectance(e0, e1, height, theta):
    """TE reflectance of a film with eps = e0 + (e1 - e0) z/height, exact, by Airy functions."""
    from scipy.special import airy
    k0 = 2*math.pi/WAVELENGTH
    kx2 = COVER*math.sin(theta)**2
    slope = (e1 - e0)/height
    c = (k0**2*slope)**(1/3 + 0j)

    def basis(z):                                    # rows: E, dE/dz, for the Ai and Bi solutions
        ai, aip, bi, bip = airy(-c*(z + (e0 - kx2)/slope))
        return np.array([[ai, bi], [-c*aip, -c*bip]])
    kc = k0*np.sqrt(COVER - kx2 + 0j)
    ks = k0*np.sqrt(SUBSTRATE - kx2 + 0j)
    inside = np.linalg.solve(basis(0.), np.array([1, -1j*ks]))      # a wave t*exp(-i ks z) below
    e_top, de_top = basis(height) @ inside
    t, r = np.linalg.solve(np.array([[e_top, -1], [de_top, -1j*kc]]), np.array([1, -1j*kc]))
    return abs(r)**2


# (e) -----------------------------------------------------------------------------------------

@pytest.mark.parametrize('theta', (0., math.radians(30.)))
def test_a_linearly_graded_film_reaches_the_airy_solution(theta):
    e0, e1, height = 2.25, 4.0 + 0.1j, 0.3
    film = rcwa.ProfiledLayer(thickness=height, background=rcwa.Graded(
        material=lambda u: Material(epsilon=e0 + (e1 - e0)*u), relative=True))
    exact = _airy_reflectance(e0, e1, height, theta)
    values = [_reflectance(rcwa.run(_problem([film], theta), numerics=_numerics(n))) for n in (128, 256, 512)]
    assert abs(values[-1] - exact) < 1e-6
    order = math.log2(abs(values[1] - values[0])/abs(values[2] - values[1]))
    assert 1.8 <= order <= 2.2


@pytest.mark.parametrize('fraction', (0.3, 0.5))
def test_a_roughness_layer_is_the_thin_film_of_its_effective_medium(fraction):
    rough = rcwa.Layer(thickness=0.004, background=Bruggeman(
        host=Material(epsilon=SILICON), inclusion=Material(epsilon=1.0), fraction=fraction))
    exact = _characteristic_matrix_reflectance([SILICON, _bruggeman(SILICON, 1.0, fraction)], [0.08, 0.004], 0.)
    silicon = rcwa.Layer(thickness=0.08, background=Material(epsilon=SILICON))
    assert abs(_reflectance(rcwa.run(_problem([silicon, rough]), numerics=_numerics())) - exact) < 1e-12


def test_a_graded_interface_layer_reaches_its_fine_staircase():
    height = 0.01
    interface = rcwa.ProfiledLayer(thickness=height, background=rcwa.Graded.interface(
        below=Material(epsilon=SILICON), above=Material(epsilon=2.1025)))
    fine = 20000
    epsilons = [_bruggeman(SILICON, 2.1025, (k + 0.5)/fine) for k in range(fine)]
    exact = _characteristic_matrix_reflectance(epsilons, [height/fine]*fine, math.radians(20.))
    result = rcwa.run(_problem([interface], math.radians(20.)), numerics=_numerics(512))
    assert abs(_reflectance(result) - exact) < 1e-6


# (c) -----------------------------------------------------------------------------------------

def _stack(thickness=0.004, fraction=0.5, interface=0.01):
    rough = rcwa.Layer(thickness=thickness, background=Bruggeman(
        host=Material(epsilon=SILICON), inclusion=Material(epsilon=1.0), fraction=fraction))
    mixed = rcwa.ProfiledLayer(thickness=interface, background=rcwa.Graded.interface(
        below=Material(epsilon=2.1025), above=Material(epsilon=SILICON)))
    return _problem([mixed, rcwa.Layer(thickness=0.08, background=Material(epsilon=SILICON)), rough],
                    math.radians(20.), 'p')


_POINT = {'thickness': 0.004, 'fraction': 0.5, 'interface': 0.01}


@pytest.mark.parametrize('name', tuple(_POINT))
def test_torch_derivatives_match_central_differences(name):
    torch = pytest.importorskip('torch')
    numerics = _numerics(16)
    prepared = rcwa.prepare(_stack(), numerics=numerics, backend='torch')
    value = torch.tensor(_POINT[name], dtype=torch.float64, requires_grad=True)
    torch.sum(prepared.run(_stack(**{name: value})).R).backward()
    reference_prepared = rcwa.prepare(_stack(), numerics=numerics)
    step = 1e-6
    def total(v):
        return _reflectance(reference_prepared.run(_stack(**{name: v})))
    reference = (total(_POINT[name] + step) - total(_POINT[name] - step))/(2*step)
    assert abs(float(value.grad) - reference) <= 1e-6*abs(reference)


@pytest.mark.slow
def test_jax_fraction_derivative_matches_a_central_difference():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    numerics = _numerics(4)
    prepared = rcwa.prepare(_stack(), numerics=numerics, backend='jax')
    gradient = float(jax.grad(lambda f: jax.numpy.sum(prepared.run(_stack(fraction=f)).R))(0.5))
    reference_prepared = rcwa.prepare(_stack(), numerics=numerics)
    reference = (_reflectance(reference_prepared.run(_stack(fraction=0.5 + 1e-6)))
                 - _reflectance(reference_prepared.run(_stack(fraction=0.5 - 1e-6))))/2e-6
    assert abs(gradient - reference) <= 1e-6*abs(reference)


# (f) -----------------------------------------------------------------------------------------

def test_torch_and_numpy_agree():
    pytest.importorskip('torch')
    reference = rcwa.run(_stack(), numerics=_numerics(8))
    result = rcwa.run(_stack(), numerics=_numerics(8), backend='torch')
    assert abs(float(result.R.sum()) - _reflectance(reference)) < 1e-12
    assert abs(float(result.T.sum()) - float(np.sum(reference.T))) < 1e-12


@pytest.mark.slow
def test_jax_and_numpy_agree():
    pytest.importorskip('jax')
    reference = rcwa.run(_stack(), numerics=_numerics(4))
    result = rcwa.run(_stack(), numerics=_numerics(4), backend='jax')
    assert abs(_reflectance(result) - _reflectance(reference)) < 1e-12


# The contract ----------------------------------------------------------------------------------

def test_a_constant_grading_is_the_uniform_layer_to_the_bit():
    uniform = rcwa.run(_problem([rcwa.Layer(thickness=0.2, background=Material(epsilon=2.25))]), numerics=_numerics(8))
    graded = rcwa.run(_problem([rcwa.ProfiledLayer(thickness=0.2, background=rcwa.Graded(
        material=lambda z: Material(epsilon=2.25)))]), numerics=_numerics(8))
    assert np.array_equal(uniform.R, graded.R) and graded.diagnostics['z_slicing'][0]['merged']


def test_a_graded_background_under_a_tapered_line():
    period = 0.5
    line = rcwa.ProfiledLayer(thickness=0.2, background=rcwa.Graded.interface(
        below=Material(epsilon=1.0), above=Material(epsilon=2.25)), regions=(
        rcwa.Region(geometry=rcwa.Trapezoid(cd=0.2, swa=1.45), material=Material(epsilon=12.25)),))
    problem = rcwa.Problem(units=Units(length='um'), stack=rcwa.Stack(period=(period, period), layers=(line,)),
                           excitation=rcwa.Excitation(wavelength=0.6, sources=(rcwa.PlaneWave(theta=0.2, phi=0.),)))
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(8, 0), factorization='li',
                                                      slicing=rcwa.Slicing(count=8, estimate=None)))
    assert abs(float(np.sum(result.R) + np.sum(result.T)) - 1.) < 1e-10      # lossless: power balance


def test_a_graded_material_outside_a_profiled_layer_is_refused():
    graded = rcwa.Graded(material=lambda z: Material(epsilon=2.25))
    with pytest.raises(ValidationError) as caught:
        rcwa.Layer(thickness=0.1, background=graded)
    assert caught.value.issue.code == 'layer.profile_material'
    with pytest.raises(ValidationError) as caught:
        rcwa.Stack(top=graded)
    assert caught.value.issue.code == 'stack.graded_half_space'
    with pytest.raises(ValidationError) as caught:
        rcwa.Graded(material=2.25)
    assert caught.value.issue.code == 'material.graded'


def test_a_roughness_layer_saves_as_a_job_and_a_graded_one_does_not():
    from wavebit.jobs import Job
    rough = rcwa.Layer(thickness=0.004, background=Bruggeman(
        host=Material(epsilon=SILICON), inclusion=Material(epsilon=1.0), fraction=0.5))
    job = Job(solver='rcwa', problem=_problem([rough]), numerics=_numerics())
    assert Job.from_json(job.to_json()) == job
    graded = rcwa.ProfiledLayer(thickness=0.01, background=rcwa.Graded.interface(
        below=Material(epsilon=SILICON), above=Material(epsilon=1.0)))
    with pytest.raises(ValidationError, match='Callables'):
        Job(solver='rcwa', problem=_problem([graded]), numerics=_numerics(4)).to_json()
