"""Physical constitutive invariants independent of either Maxwell solver."""
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import numpy as np
import pytest

from wavebit.materials import Material, MaterialResponse, ConductiveMaterial, DispersiveMaterial, Drude, Lorentz, Debye, TabulatedMaterial
from wavebit.materials._lower import lower_material
from wavebit.materials import _time_domain as time_domain
from wavebit.units import Units, C0, EPSILON0
from wavebit.errors import UnsupportedFeatureError, ValidationError
from wavebit.observations import DiffractionOrders


@pytest.mark.parametrize('length,wavelength', [('m',800e-9),('mm',800e-6),('um',.8),('nm',800.)])
def test_si_poles_are_independent_of_geometry_unit(length,wavelength):
    model=DispersiveMaterial(2.,(Drude(2e15,1e13),Lorentz(3e15,2e13,1e30),Debye(1e-15,.4)))
    omega=2*np.pi*C0/(800e-9)
    expected=2.-(2e15)**2/(omega**2+1j*1e13*omega)+1e30/((3e15)**2-omega**2-1j*2e13*omega)+.4/(1-1j*omega*1e-15)
    units=Units(length=length)
    response=model.at(wavelength,units=units)
    assert isinstance(response,MaterialResponse)
    assert response.epsilon==pytest.approx(expected,rel=2e-14)
    legacy=lower_material(model,units=units)
    assert time_domain.eps_at(legacy,wavelength)==pytest.approx(expected,rel=2e-14)


@pytest.mark.parametrize('length,wavelength', [('nm',800.),('um',.8)])
def test_conductivity_is_si_and_lowered_once(length,wavelength):
    model=ConductiveMaterial(epsilon=2.,conductivity=1e4,mu=1.2)
    units=Units(length=length)
    expected=2.+1j*1e4/(EPSILON0*(2*np.pi*C0/(800e-9)))
    assert model.at(wavelength,units=units).epsilon==pytest.approx(expected)
    old=lower_material(model,units=units)
    assert old.sigma[0]==pytest.approx(1e4*units.length_unit_meters/(EPSILON0*C0))
    assert old.mu==(1.2,)*3


def test_isotropic_pole_adds_only_tensor_diagonal():
    epsilon=np.array([[2.,.1,0.],[.1,3.,0.],[0.,0.,4.]])
    model=DispersiveMaterial(epsilon,(Debye(1e-15,.4),))
    response=model.at(np.array([700.,800.]),units=Units(length='nm'))
    assert response.epsilon.shape==(2,3,3)
    np.testing.assert_array_equal(response.epsilon[:,0,1],(.1,.1))
    np.testing.assert_array_equal(response.epsilon[:,0,2],(0.,0.))


def test_constants_preserve_structural_mu_and_explicit_spectral_axis():
    units=Units(length='nm')
    assert Material(2.).at(800.,units=units).mu is None
    assert Material(2.,mu=1.).at(800.,units=units).mu is not None
    result=Material([2.,3.,4.]).at(np.array([700.,800.]),units=units)
    assert result.epsilon.shape==(2,3)
    assert lower_material(Material(np.array([2.,3.,4.])),units=units).eps==(2.,3.,4.)


def test_tables_retain_interpolation_and_source_units():
    model=TabulatedMaterial.from_refractive_index((500.,1000.),(2.,3.),wavelength_unit='nm')
    expected=2.6**2
    assert model.at(.8,units=Units(length='um')).epsilon==pytest.approx(expected)
    assert model.interpolated_quantity=='index'
    assert model.replace(wavelength_unit='um')!=model
    with pytest.raises(ValidationError):model.at(-1.,units=Units(length='nm'))
    with pytest.raises(UnsupportedFeatureError,match='causal'):lower_material(model,units=Units(length='nm'))


@pytest.mark.parametrize('model', [Material(2.+.1j),Material(np.eye(3)),ConductiveMaterial(2.,-1.),DispersiveMaterial(2.,(Lorentz(1e15,1e13,-1e29),))])
def test_fdtd_admission_refuses_unsupported_constitutive_cases(model):
    with pytest.raises(UnsupportedFeatureError):lower_material(model,units=Units(length='nm'))


def test_jax_entire_observation_and_nested_references_are_pytrees():
    jax=pytest.importorskip('jax')
    import jax.numpy as jnp
    def response(plasma):
        epsilon=DispersiveMaterial(2.,(Drude(plasma,1e13),)).at(jnp.asarray(800.),units=Units(length='nm')).epsilon
        return DiffractionOrders(wavelengths=jnp.array([800.]),orders_u=(0,),orders_v=(0,),amplitude_s=epsilon.reshape((1,1,1)),reference={'frame':'s/p','medium':epsilon,'source':{'amplitude':plasma}},metadata={'solver':'test'})
    with jax.enable_x64(True):
        result=jax.jit(response)(2e15)
        assert result.reference['frame']=='s/p'
        gradient=jax.grad(lambda p:jnp.real(response(p).amplitude_s).sum())(2e15)
        assert gradient!=0
        assert result.order(0).amplitude_s.shape==(1,)
        assert result.at_wavelength(0).amplitude_s.shape==(1,1,1)


def test_torch_material_values_keep_their_graph():
    torch=pytest.importorskip('torch')
    plasma=torch.tensor(2e15,dtype=torch.float64,requires_grad=True)
    value=DispersiveMaterial(2.,(Drude(plasma,1e13),)).at(800.,units=Units(length='nm')).epsilon
    value.real.backward()
    assert plasma.grad is not None and plasma.grad!=0


def test_import_and_new_execution_do_not_mutate_jax_precision():
    pytest.importorskip('jax')
    code='''
import sys, jax
jax.config.update('jax_enable_x64',False)
import wavebit
assert not jax.config.x64_enabled
from wavebit._common.execution import Execution
from wavebit._common.backend import lift_jax_x64
from wavebit.errors import UnsupportedFeatureError
try: Execution(backend='jax',precision='double').validate()
except UnsupportedFeatureError: pass
else: raise AssertionError('double was silently downgraded')
with Execution(backend='jax',precision='single')._scope():
    lift_jax_x64(required=True)
    assert not jax.config.x64_enabled
assert not jax.config.x64_enabled
assert 'torch' not in sys.modules
'''
    import wavebit
    env=dict(os.environ,PYTHONPATH=str(Path(wavebit.__file__).parents[1]))
    subprocess.run([sys.executable,'-c',code],check=True,env=env,cwd=tempfile.gettempdir())


def test_numpy_scalar_wavelength_does_not_compete_with_torch_material_backend():
    torch = pytest.importorskip('torch')
    epsilon = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    response = Material(epsilon).at(np.float64(800.), units=Units(length='nm'))
    response.epsilon.real.backward()
    assert isinstance(response.epsilon, torch.Tensor)
    assert epsilon.grad.item() == pytest.approx(1.)


@pytest.mark.parametrize('pole', [lambda x: Drude(1e15,x), lambda x: Lorentz(3e15,x,1e29), lambda x: Debye(x,.4)])
def test_invalid_pole_rates_are_never_silently_admitted_under_jit(pole):
    jax = pytest.importorskip('jax')
    with jax.enable_x64(True):
        def response(rate):
            return DispersiveMaterial(2., (pole(rate),)).at(800., units=Units(length='nm')).epsilon
        with pytest.raises(ValidationError):
            response(-1.)
        assert np.isnan(jax.jit(response)(-1.))
        assert np.isfinite(jax.jit(response)(1.))


def test_table_accepts_a_wavelength_sequence_with_the_same_unit_contract_as_other_models():
    from wavebit.materials import TabulatedMaterial
    table = TabulatedMaterial((500., 1000.), (2., 4.), wavelength_unit='nm')
    values = table.at([.5, .75, 1.], units=Units(length='um')).epsilon
    np.testing.assert_allclose(values, [2., 3., 4.], rtol=1e-14)
    with pytest.raises(ValidationError):
        table.at(.499, units=Units(length='um'))
