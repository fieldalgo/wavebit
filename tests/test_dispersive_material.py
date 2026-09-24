"""Dispersion is reevaluated, serialized and differentiated rather than frozen."""
import json

import numpy as np
import pytest

import wavebit
from wavebit import Incidence, JobSpec, Layer, Material, Stack, Sweep, TabulatedMaterial


def scalar(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    return np.asarray(value).item()


def film(material, backend=0, **kwargs):
    return wavebit.rcwa(backend=backend, wavelength=900., period=(600., 600.),
                        structure=[dict(background=material, shapes=[])], thickness=[120.],
                        truncation_order=(0, 0), polarization=(1., 0.), factorization="li", **kwargs)


def airy(wavelength, epsilon):
    n = np.sqrt(complex(epsilon))
    r01, r12 = (1-n)/(1+n), (n-1)/(n+1)
    phase = np.exp(2j*np.pi*n*120/wavelength)
    den = 1 + r01*r12*phase**2
    r = (r01+r12*phase**2)/den
    t = 4*n/(1+n)**2*phase/den
    return abs(r)**2, abs(t)**2


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_dispersive_film_against_airy_and_repeated_wavelengths(backend):
    if backend:
        pytest.importorskip('jax' if backend == 1 else 'torch')
    m = TabulatedMaterial([800., 1000.], [4.+.1j, 9.+.3j])
    solver = film(m, backend)
    for w in (800., 875., 1000., 900.):
        result = solver.solve(wavelength=w, auto_detune=False)
        epsilon = 4.+.1j + (w-800.)/200.*(5.+.2j)
        expected = airy(w, epsilon)
        np.testing.assert_allclose([scalar(result.R.sum()), scalar(result.T.sum())],
                                   expected, rtol=1e-11, atol=1e-12)
        assert scalar(solver.structure[0]['background'].epsilon) == pytest.approx(epsilon)


def test_imported_index_interpolation_and_explicit_snapshot():
    wavelengths = np.array([800., 1000.])
    index = np.array([2.+.1j, 3.+.3j])
    model = TabulatedMaterial.from_index(wavelengths, index)
    snapshot = model.at(900.)
    assert snapshot.epsilon == pytest.approx((2.5+.2j)**2)
    assert snapshot.at(950.) is snapshot
    wavelengths[:] = 0
    index[:] = 0
    assert model.at(900.).epsilon == snapshot.epsilon
    with pytest.raises(ValueError, match='range'):
        model.at(1001.)


@pytest.mark.parametrize('slot', ['top', 'bot'])
def test_half_space_refresh_and_atomic_range_error(slot):
    m = TabulatedMaterial([800., 1000.], [4., 9.])
    solver = wavebit.rcwa(period=600., wavelength=900., structure=np.ones((1, 1, 1)),
                          thickness=[0.], tetm='te', **{slot: m})
    for w, n in ((800., 2.), (900., np.sqrt(6.5)), (1000., 3.)):
        result = solver.solve(wavelength=w, auto_detune=False)
        assert scalar(result.R.sum()) == pytest.approx(((n-1)/(n+1))**2)
    before = (solver.wavelength, solver.top, solver.bot, solver.n_top, solver.n_bot)
    with pytest.raises(ValueError, match='range'):
        solver.wavelength = 1200.
    assert (solver.wavelength, solver.top, solver.bot, solver.n_top, solver.n_bot) == before
    setattr(solver, slot, Material(2.25))
    solver.wavelength = 1200.  # replacing the material also removes dispersion
    assert getattr(solver, slot).epsilon == 2.25


def test_shape_material_is_updated_without_mutating_input_and_can_be_replaced():
    material = TabulatedMaterial([800., 1000.], [4., 9.])
    shape = dict(type='circle', cx=0., cy=0., lx=180., material=material)
    structure = [dict(background=1., shapes=[shape])]
    solver = wavebit.rcwa(wavelength=900., period=(600., 600.), thickness=[100.],
                          structure=structure, truncation_order=(1, 1),
                          shape_nvf_grid_size=32, polarization=(1., 0.))
    for w in (850., 950.):
        actual = solver.solve(wavelength=w, auto_detune=False)
        manual = wavebit.rcwa(wavelength=w, period=(600., 600.), thickness=[100.],
                              structure=[dict(background=1., shapes=[dict(shape, material=material.at(w))])],
                              truncation_order=(1, 1), shape_nvf_grid_size=32,
                              polarization=(1., 0.)).solve(auto_detune=False)
        np.testing.assert_allclose(actual.R, manual.R, atol=1e-13)
        np.testing.assert_allclose(actual.T, manual.T, atol=1e-13)
    assert shape['material'] is material
    solver.structure = [dict(background=2.25, shapes=[])]
    solver.wavelength = 1100.
    assert solver.structure[0]['background'] == 2.25


def test_jobspec_round_trip_retains_model_and_sweep():
    m = TabulatedMaterial([800., 1000.], [4., 9.])
    spec = JobSpec(stack=Stack([Layer(120., dict(background=m, shapes=[]))], period=(600., 600.)),
                   incidence=Incidence(wavelength=900., polarization=(1., 0.)),
                   sweep=Sweep({"wavelength": [800., 900., 1000.]}))
    restored = JobSpec.from_json(spec.to_json())
    assert restored.key() == spec.key()
    assert restored.stack.check() == ()
    for point in restored.expand():
        w = point.incidence.wavelength
        result = wavebit.run(point)
        np.testing.assert_allclose([result.R.sum(), result.T.sum()],
                                   airy(w, 4.+(w-800.)/200.*5.), atol=1e-12)
    obj = json.loads(spec.to_json())
    obj['stack']['layers'][0]['structure']['background']['dispersion']['unused'] = 1
    with pytest.raises(wavebit.SpecError):
        JobSpec.from_dict(obj)


def test_index_table_and_magnetic_table_json_round_trip():
    for m in (TabulatedMaterial.from_index([800., 1000.], [2.+.1j, 3.+.2j]),
              TabulatedMaterial([800., 1000.], [4., 9.], mu=[1.1, 1.2])):
        spec = JobSpec(stack=Stack([Layer(0., np.ones((1, 1)))], top=m, period=(600., 600.)))
        restored = JobSpec.from_json(spec.to_json())
        assert restored.stack.top == m
        assert restored.key() == spec.key()
        assert restored.stack.check() == ()


@pytest.mark.parametrize('w,eps,mu', [([1.], [2.], None), ([2., 1.], [2., 3.], None),
                                    ([1., 1.], [2., 3.], None), ([-1., 2.], [2., 3.], None),
                                    ([1., np.inf], [2., 3.], None), ([1., 2.], [2.], None),
                                    ([1., 2.], [2., np.nan], None), ([1., 2.], [2., 3.], [1.])])
def test_invalid_tables_are_refused(w, eps, mu):
    with pytest.raises(ValueError):
        TabulatedMaterial(w, eps, mu)


def test_jax_wavelength_gradient_jit_and_solver_pytree_keep_dispersion():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()
    m = TabulatedMaterial([800., 1000.], [4., 9.])
    solver = film(m, backend=1)
    geometry = solver.freeze_geometry()
    fn = jax.jit(lambda w: solver.solve(wavelength=w, geometry=geometry,
                                        auto_detune=False).R.sum())
    w = 910.
    finite = (airy(w+.01, 4.+(w+.01-800.)*.025)[0] -
              airy(w-.01, 4.+(w-.01-800.)*.025)[0])/.02
    assert float(jax.grad(fn)(w)) == pytest.approx(finite, rel=2e-7, abs=1e-9)
    assert float(fn(950.)) == pytest.approx(airy(950., 7.75)[0])
    assert float(solver.wavelength) == 900.  # traced state must not leak
    assert float(solver.solve(auto_detune=False).R.sum()) == pytest.approx(airy(900., 6.5)[0])
    leaves, tree = jax.tree_util.tree_flatten(solver)
    rebuilt = jax.tree_util.tree_unflatten(tree, leaves)
    assert float(rebuilt.solve(wavelength=950., auto_detune=False).R.sum()) == pytest.approx(airy(950., 7.75)[0])
    assert np.isnan(np.asarray(jax.jit(lambda w: m.at(w).epsilon)(jnp.asarray(1100.))))


def test_torch_wavelength_derivative_keeps_graph():
    torch = pytest.importorskip('torch')
    m = TabulatedMaterial([800., 1000.], [4., 9.])
    w = torch.tensor(910., dtype=torch.float64, requires_grad=True)
    solver = film(m, backend=2)
    r = solver.solve(wavelength=w, auto_detune=False).R.sum()
    r.backward()
    finite = (airy(910.01, 6.75025)[0]-airy(909.99, 6.74975)[0])/.02
    assert w.grad.item() == pytest.approx(finite, rel=2e-7, abs=1e-9)
    w2 = torch.tensor(920., dtype=torch.float64, requires_grad=True)
    m.at(w2).epsilon.real.backward()
    assert w2.grad.item() == pytest.approx(.025)


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_precision_change_keeps_material_definitions(backend):
    if backend:
        pytest.importorskip('jax' if backend == 1 else 'torch')
    m = TabulatedMaterial([800., 1000.], [4., 9.])
    solver = film(m, backend, bot=m)
    solver.complex_dtype = 1
    solver.wavelength = 950.
    assert scalar(solver.bot.epsilon) == pytest.approx(7.75)
    assert scalar(solver.structure[0]['background'].epsilon) == pytest.approx(7.75)
    assert '64' in str(solver.bot.epsilon.dtype)
    solver.complex_dtype = 0
    solver.wavelength = 850.
    assert scalar(solver.bot.epsilon) == pytest.approx(5.25)
    assert '128' in str(solver.bot.epsilon.dtype)


def test_range_failure_does_not_partially_update_other_materials():
    wide = TabulatedMaterial([700., 1100.], [2., 3.])
    narrow = TabulatedMaterial([800., 1000.], [4., 9.])
    solver = film(narrow, top=wide)
    before = (solver.wavelength, solver.top, solver.n_top, solver.structure)
    with pytest.raises(ValueError, match='range'):
        solver.wavelength = 1050.
    assert solver.wavelength == before[0]
    assert solver.top is before[1]
    assert solver.n_top is before[2]
    assert solver.structure is before[3]


def test_manifest_and_schema_declare_dispersion():
    from wavebit.manifest import manifest
    info = manifest()
    assert info['material_dispersion']['automatic_slots'] == [
        'top', 'bot', 'layer.background', 'shape.material']
    assert info['results']['jobspec']['spec_version'] == '10'
    with pytest.raises(wavebit.SpecError, match="version='10'"):
        JobSpec.from_dict({'version': '8'})


def test_jit_assignment_restores_definition_with_its_evaluated_values():
    jax = pytest.importorskip('jax')
    wavebit.enable_x64()
    old = TabulatedMaterial([800., 1000.], [4., 9.])
    new = TabulatedMaterial([800., 1000.], [16., 25.])
    solver = wavebit.rcwa(backend=1, wavelength=900., period=600.,
                          structure=np.ones((1, 1, 1)), thickness=[0.],
                          top=old, tetm='te', fourier_coefficients='integrated')
    geometry = solver.freeze_geometry()
    run = jax.jit(lambda w: solver.solve(wavelength=w, top=new, geometry=geometry,
                                         auto_detune=False).R.sum())
    n = np.sqrt(20.5)
    assert float(run(900.)) == pytest.approx(((n-1)/(n+1))**2)
    assert scalar(solver.top.epsilon) == pytest.approx(6.5)
    result = solver.solve(wavelength=950., auto_detune=False)
    n = np.sqrt(7.75)
    assert float(result.R.sum()) == pytest.approx(((n-1)/(n+1))**2)


@pytest.mark.parametrize('backend', [0, 1, 2])
@pytest.mark.parametrize('slot', ['top', 'bot'])
def test_permeability_dispersion_reaches_interface_impedance(backend, slot):
    """Changing mu alone changes Fresnel reflection; index-only ingest cannot pass."""
    if backend:
        pytest.importorskip('jax' if backend == 1 else 'torch')
    m = TabulatedMaterial([800., 1000.], [4., 4.], mu=[1., 4.])
    solver = wavebit.rcwa(backend=backend, wavelength=800., period=600.,
                          structure=np.ones((1, 1, 1)), thickness=[0.],
                          truncation_order=0, tetm='te', **{slot: m})
    for w in (800., 900., 1000.):
        result = solver.solve(wavelength=w, auto_detune=False)
        mu = 1.+(w-800.)/200.*3.
        impedance = np.sqrt(mu/4.)
        reflection = ((impedance-1)/(impedance+1))**2
        assert scalar(result.R.sum()) == pytest.approx(reflection, abs=1e-12)
        assert scalar(result.T.sum()) == pytest.approx(1-reflection, abs=1e-12)
        assert scalar(getattr(solver, slot).mu) == pytest.approx(mu)
