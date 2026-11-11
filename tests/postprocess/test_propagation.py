"""Independent plane-wave, interference, unit and derivative contracts."""
import numpy as np
import pytest

from wavebit import Units, Material, rcwa
from wavebit.observations import DiffractionOrders
from wavebit.postprocess.propagation import Plane, propagate
from wavebit.errors import ValidationError


def orders(sign=1, unit='um', evanescent=False):
    scale = Units(length=unit).length_unit_meters/1e-6
    return DiffractionOrders(wavelengths=np.array([1./scale]), orders_u=(0, 1), orders_v=(0,),
        amplitude_ratio_xyz=np.array([[[[1, 0, 0], [1j, 0, 0]]]], complex),
        k_parallel=np.array([[[[0, 0], [0, 2]]]], complex)*scale,
        k_normal=np.array([[[3, 2j if evanescent else 2]]], complex)*sign*scale,
        reference=dict(wave='transmitted', plane=(0.,), lateral_origin=(0., 0.),
                       outward_normal=(0., 0., sign), epsilon=(1.,), mu=(1.,),
                       length_units=unit, wavevector_units='rad/'+unit,
                       incident_field=1., incident_field_units='V/m'),
        metadata=dict(propagation='continuum', time_convention='exp(-i*omega*t)'))


def target(d=.3, unit='um'):
    return Plane(x=[0., .2], y=[0., .1, .3], distance=d, units=Units(length=unit))


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('evanescent', [False, True])
@pytest.mark.parametrize('chunk', [1, 4, 256])
def test_analytic_interference_and_decay(sign, evanescent, chunk):
    got = propagate(orders(sign, evanescent=evanescent), target=target(), chunk_size=chunk)
    y = np.array([0., .1, .3])
    expected = np.exp(.9j)+1j*np.exp(2j*y)* (np.exp(-.6) if evanescent else np.exp(.6j))
    np.testing.assert_allclose(got.field.values[0, :, :, 0], np.repeat(expected[:, None], 2, axis=1), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(got.field.values[..., 1:], 0)
    assert got.field.units['Ex'] == 'V/m'
    np.testing.assert_allclose(got.field.coordinates['z'], sign*.3)


@pytest.mark.parametrize('unit', ['m', 'um', 'nm'])
def test_units_zero_distance_and_composition(unit):
    o = orders(unit=unit)
    f = Units(length=unit).length_unit_meters/1e-6
    t = Plane(x=[0., .2/f], y=[0., .1/f, .3/f], distance=.3/f, units=Units(length=unit))
    np.testing.assert_allclose(propagate(o, target=t).field.values, propagate(orders(), target=target()).field.values, rtol=1e-10)
    zero = propagate(orders(), target=target(0.)).field.values
    np.testing.assert_allclose(zero[0, :, 0, 0], 1+1j*np.exp(2j*np.array([0., .1, .3])), rtol=1e-10)
    shifted = o.replace(amplitude_ratio_xyz=o.amplitude_ratio_xyz*np.exp(1j*o.k_normal[..., None]*.1/f))
    np.testing.assert_allclose(propagate(shifted, target=t.replace(distance=.2/f)).field.values, propagate(o, target=t).field.values, rtol=1e-10)


@pytest.mark.parametrize('change', ['missing_k', 'pulse', 'discrete', 'incoming', 'wrong_units', 'backward', 'lossy', 'negative', 'empty', 'asm'])
def test_refusals(change):
    o, t, method = orders(), target(), 'orders'
    if change == 'missing_k': o = o.replace(k_normal=None)
    if change == 'pulse': o = o.replace(reference={k:v for k,v in o.reference.items() if not k.startswith('incident_field')})
    if change == 'discrete': o = o.replace(metadata={**o.metadata, 'propagation':'yee'})
    if change == 'incoming': o = o.replace(reference={**o.reference, 'wave':'incoming'})
    if change == 'wrong_units': o = o.replace(reference={**o.reference, 'wavevector_units':'rad/m'})
    if change == 'backward': o = o.replace(k_normal=-o.k_normal)
    if change == 'lossy': o = o.replace(reference={**o.reference, 'epsilon':(1+.1j,)})
    if change == 'negative': t = t.replace(distance=-1.)
    if change == 'empty': t = t.replace(x=[])
    if change == 'asm': method = 'asm'
    with pytest.raises(ValidationError, match='') as exc: propagate(o, target=t, method=method)
    assert exc.value.code.startswith('propagation.')


def test_reference_origin_and_nan_provenance():
    o = orders()
    shifted = o.replace(reference={**o.reference, 'lateral_origin':(.4, .5), 'plane':(2.,)})
    a = propagate(shifted, target=Plane(x=[.4, .6], y=[.5, .6, .8], distance=.3, units=Units(length='um')))
    np.testing.assert_allclose(a.field.values, propagate(o, target=target()).field.values, atol=1e-12)
    np.testing.assert_allclose(a.field.coordinates['z'], 2.3)
    e = o.amplitude_ratio_xyz.copy(); e[0, 0, 0] = np.nan
    bad = propagate(o.replace(amplitude_ratio_xyz=e), target=target())
    assert np.isnan(bad.field.values).all()
    assert bad.diagnostics['source_metadata'] == o.metadata


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_backend_forward_and_amplitude_distance_gradient(backend):
    o = orders()
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as xp
        conv = xp.asarray
    else:
        xp = pytest.importorskip('torch')
        conv = lambda v: xp.as_tensor(v, dtype=xp.complex128 if np.iscomplexobj(v) else xp.float64)
    native = o.replace(amplitude_ratio_xyz=conv(o.amplitude_ratio_xyz), k_parallel=conv(o.k_parallel), k_normal=conv(o.k_normal))
    np.testing.assert_allclose(np.asarray(propagate(native, target=target()).field.values), propagate(o, target=target()).field.values, rtol=1e-10)
    def objective(p):
        amplitude = native.amplitude_ratio_xyz*(p[0]+1j*p[1])
        v = propagate(native.replace(amplitude_ratio_xyz=amplitude), target=target(p[2])).field.values
        return xp.sum(xp.abs(v)**2)
    p = np.array([1.2, .4, .3])
    if backend == 'jax': grad = np.asarray(jax.grad(objective)(conv(p)))
    else:
        v = xp.tensor(p, requires_grad=True); objective(v).backward(); grad = v.grad.numpy()
    for h in [1e-3, 3e-4, 1e-4]:
        fd = np.array([(float(objective(conv(p+np.eye(3)[i]*h)))-float(objective(conv(p-np.eye(3)[i]*h))))/(2*h) for i in range(3)])
        assert np.max(np.abs(grad-fd)/np.maximum(1, np.maximum(abs(grad),abs(fd)))) <= 1e-6


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_rcwa_boundary_fields(side):
    problem = rcwa.Problem(units=Units(length='um'),
        stack=rcwa.Stack(period=(1.,1.), top=Material(1.), bottom=Material(2.25), layers=()),
        excitation=rcwa.Excitation(wavelength=[.8], sources=(rcwa.PlaneWave(side=side, polarization='s', theta=.2),)),
        outputs=rcwa.Outputs(fields=True))
    result = rcwa.run(problem, numerics=rcwa.Numerics())
    for o in (result.reflection, result.transmission):
        got = propagate(o, target=target(0.)).field
        expected = result.boundary_fields(side=o.reference['port'], x=[0., .2], y=[0., .1, .3], part='outgoing')
        np.testing.assert_allclose(got.values, expected.values[..., :3], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_rcwa_design_to_observation_gradient(backend):
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as xp
        conv = xp.asarray
    else:
        xp = pytest.importorskip('torch')
        conv = lambda v: xp.as_tensor(v, dtype=xp.complex128 if np.iscomplexobj(v) else xp.float64)
    def objective(thickness):
        problem = rcwa.Problem(units=Units(length='um'),
            stack=rcwa.Stack(period=(1., 1.), top=Material(1.), bottom=Material(1.),
                            layers=(rcwa.Layer(thickness=thickness, background=Material(2.25)),)),
            excitation=rcwa.Excitation(wavelength=[.8], sources=(rcwa.PlaneWave(polarization='s'),)))
        result = rcwa.run(problem, numerics=rcwa.Numerics(), backend=backend)
        values = propagate(result.transmission, target=target()).field.values
        return xp.sum(xp.abs(values)**2)
    if backend == 'jax': grad = float(jax.grad(objective)(conv(.23)))
    else:
        t = xp.tensor(.23, dtype=xp.float64, requires_grad=True)
        objective(t).backward(); grad = float(t.grad)
    # Smaller dimensionless thickness steps are needed for an oscillatory film.
    errors=[]
    for h in [1e-3, 3e-4, 1e-4]:
        step = .01*h
        fd = (float(objective(conv(.23+step)))-float(objective(conv(.23-step))))/(2*step)
        errors.append(abs(grad-fd)/max(1,abs(grad),abs(fd)))
    assert sum(e <= 1e-6 for e in errors) >= 2


def test_complex_coordinates_are_not_silently_discarded():
    with pytest.raises(ValidationError, match='real coordinates'):
        propagate(orders(), target=target().replace(x=[1+1j]))


def test_same_efficiencies_different_phase_give_different_fields():
    o = orders()
    e = o.amplitude_ratio_xyz.copy(); e[..., 1, :] *= -1
    a = propagate(o, target=target()).field.values
    b = propagate(o.replace(amplitude_ratio_xyz=e), target=target()).field.values
    assert not np.allclose(abs(a)**2, abs(b)**2)


def test_jit_refuses_abstract_distance_validation():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    o = orders()
    o = o.replace(amplitude_ratio_xyz=jnp.asarray(o.amplitude_ratio_xyz),
                  k_parallel=jnp.asarray(o.k_parallel), k_normal=jnp.asarray(o.k_normal))
    with pytest.raises(ValidationError, match='Abstract validation'):
        jax.jit(lambda d: propagate(o, target=target(d)).field.values)(.3)


def test_multiple_wavelengths_and_reference_planes():
    o = orders()
    o = o.replace(wavelengths=np.array([1., 2.]),
                  amplitude_ratio_xyz=np.repeat(o.amplitude_ratio_xyz, 2, axis=0),
                  k_parallel=np.concatenate([o.k_parallel, o.k_parallel/2]),
                  k_normal=np.concatenate([o.k_normal, o.k_normal/2]),
                  reference={**o.reference, 'plane':(0., 1.), 'epsilon':(1.,1.), 'mu':(1.,1.)})
    got = propagate(o, target=target(), chunk_size=2).field
    for i in range(2):
        expected = propagate(o.at_wavelength_index(i), target=target()).field
        np.testing.assert_allclose(got.values[i:i+1], expected.values, rtol=1e-10)
    np.testing.assert_allclose(got.coordinates['z'], [.3, 1.3])
