"""Physical geometry and material oracles for current-API inspection."""
import subprocess
import sys

import numpy as np
import pytest

from wavebit import Material, TabulatedMaterial, Units, rcwa
from wavebit.errors import ValidationError, UnsupportedFeatureError


def problem(layer, *, unit='nm', wavelength=850., period=(600., 400.)):
    return rcwa.Problem(units=Units(length=unit),
        stack=rcwa.Stack(period=period, layers=(layer,)),
        excitation=rcwa.Excitation(wavelength=wavelength, sources=(rcwa.PlaneWave(),)))


@pytest.mark.parametrize('quantity', ['epsilon', 'mu'])
def test_raster_is_owned_and_keeps_yx_orientation_and_resolution(quantity):
    epsilon = np.array([[1., 2., 3.], [4., 5., 6.]])
    mu = epsilon / 10 + 1
    p = problem(rcwa.Layer(thickness=100., epsilon=epsilon, mu=mu))
    expected = epsilon if quantity == 'epsilon' else mu
    actual = rcwa.sample_structure(p, quantity=quantity, n=73)
    np.testing.assert_array_equal(actual, expected)
    actual[0, 0] = 99
    np.testing.assert_array_equal(rcwa.sample_structure(p, quantity=quantity), expected)


def test_region_painting_matches_independent_rectangle_membership():
    p = problem(rcwa.Layer(thickness=100., background=Material(2., mu=1.1), regions=(
        rcwa.Region(geometry=rcwa.Rectangle(size=(300., 200.), center=(-61., 21.)), material=Material(4., mu=1.3)),
        rcwa.Region(geometry=rcwa.Rectangle(size=(180., 120.), center=(41., -31.)), material=Material(7., mu=1.6)),
    )))
    n = 60
    x, y = np.meshgrid((np.arange(n)+.5)*600/n-300, (np.arange(n)+.5)*400/n-200)
    # No sample lies on an interface: its exact boundary ownership is not a
    # material-interior oracle and depends on the renderer's half-open convention.
    first = (abs(x+61)<150) & (abs(y-21)<100)
    last = (abs(x-41)<90) & (abs(y+31)<60)
    for quantity, values in [('epsilon', (2.,4.,7.)), ('mu', (1.1,1.3,1.6))]:
        expected = np.where(last, values[2], np.where(first, values[1], values[0]))
        np.testing.assert_allclose(rcwa.sample_structure(p, quantity=quantity, n=n), expected)


@pytest.mark.parametrize('unit,scale', [('nm', 1.), ('um', .001)])
def test_tabulated_materials_use_physical_units_and_selected_wavelength(unit, scale):
    model = TabulatedMaterial(wavelengths=[800.,900.], wavelength_unit='nm',
                             epsilon=[4.+.1j,6.+.3j], mu=[1.2,1.6])
    p = problem(rcwa.Layer(thickness=100.*scale, background=model), unit=unit,
                wavelength=np.array([800.,850.,900.])*scale, period=(600.*scale,400.*scale))
    np.testing.assert_allclose(rcwa.sample_structure(p, n=7, wavelength_index=1), 5.+.2j)
    np.testing.assert_allclose(rcwa.sample_structure(p, n=7, wavelength_index=2, quantity='mu'), 1.6)


@pytest.mark.parametrize('raster', [False, True])
def test_missing_permeability_is_one(raster):
    layer = (rcwa.Layer(thickness=100., epsilon=np.full((2,3),4.)) if raster
             else rcwa.Layer(thickness=100., background=Material(4.)))
    np.testing.assert_array_equal(rcwa.sample_structure(problem(layer), quantity='mu', n=8),
                                  np.ones((2,3) if raster else (8,8)))


def test_tensor_components_are_physical_and_default_mu_is_scalar():
    tensor = np.array([[3.,.2,0.],[.2,4.,0.],[0.,0.,5.]])
    for layer in (rcwa.Layer(thickness=100., background=Material(tensor)),
                  rcwa.Layer(thickness=100., epsilon=np.broadcast_to(tensor,(2,4,3,3)))):
        p = problem(layer)
        np.testing.assert_allclose(rcwa.sample_structure(p, n=8, component='xy'), .2)
        np.testing.assert_allclose(rcwa.sample_structure(p, n=8, quantity='mu'), 1.)


def test_layer_selection_and_no_scattering_or_compat_import():
    # A fresh process also catches accidental import-time precision/backend effects.
    code = '''
import sys
import numpy as np
from wavebit import rcwa, Material, Units
from wavebit.rcwa._engine.engine import CoreFactorizationMixin
def forbidden(*a, **kw):
    raise AssertionError("Sampling must not solve scattering")
CoreFactorizationMixin.solve = forbidden
CoreFactorizationMixin.solve_conv = forbidden
CoreFactorizationMixin.eps_conv_matrices = forbidden
p = rcwa.Problem(units=Units(length='nm'), stack=rcwa.Stack(period=(600.,400.), layers=(
    rcwa.Layer(thickness=50.,background=Material(2.)),
    rcwa.Layer(thickness=75.,epsilon=np.full((2,3),7.)),
)), excitation=rcwa.Excitation(wavelength=900.,sources=(rcwa.PlaneWave(),)))
np.testing.assert_array_equal(rcwa.sample_structure(p,layer=0,n=8),np.full((8,8),2.))
np.testing.assert_array_equal(rcwa.sample_structure(p,layer=1,n=8),np.full((2,3),7.))
assert not any(m.startswith('wavebit.compat') for m in sys.modules)
assert 'jax' not in sys.modules and 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, '-I', '-c', code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('kwargs,code', [
    ({'layer': -1}, 'structure.layer'), ({'layer': 1}, 'structure.layer'),
    ({'layer': True}, 'structure.layer'), ({'n': 0}, 'structure.samples'),
    ({'n': 1.5}, 'structure.samples'), ({'quantity': 'n'}, 'structure.quantity'),
    ({'wavelength_index': 1}, 'structure.wavelength'),
])
def test_invalid_sampling_requests_are_refused(kwargs, code):
    with pytest.raises(ValidationError) as error:
        rcwa.sample_structure(problem(rcwa.Layer(thickness=100.)), **kwargs)
    assert error.value.code == code


def test_nonfinite_raster_is_refused():
    with pytest.raises((ValidationError, ValueError)):
        rcwa.sample_structure(problem(rcwa.Layer(thickness=100., epsilon=np.array([[np.nan]]))))


def test_transformed_geometry_is_explicitly_refused():
    jax = pytest.importorskip('jax')
    p = problem(rcwa.Layer(thickness=100., regions=(
        rcwa.Region(geometry=rcwa.Circle(radius=70.), material=Material(4.)),)))
    def sample(radius):
        region = p.stack.layers[0].regions[0]
        layer = p.stack.layers[0].replace(regions=(region.replace(geometry=region.geometry.replace(radius=radius)),))
        return rcwa.sample_structure(p.replace(stack=p.stack.replace(layers=(layer,))), n=8)
    with pytest.raises(UnsupportedFeatureError) as error:
        jax.jit(sample)(70.)
    assert error.value.code == 'structure.concrete'
