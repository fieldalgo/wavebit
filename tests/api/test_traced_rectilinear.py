"""Cartesian Li replay checks live edge order before exposing traced observations."""
import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.errors import IncompatibleProblemError, UnsupportedFeatureError


def pair(gap=0., epsilon=9., width=300.):
    return rcwa.Problem(units=Units(length='nm'), stack=rcwa.Stack(period=(700.,700.),
        bottom=Material(2.25), layers=(rcwa.Layer(thickness=300., background=Material(1.), regions=(
            rcwa.Region(geometry=rcwa.Rectangle(size=(300.,300.), center=(-150.-gap/2,0.)), material=Material(4.)),
            rcwa.Region(geometry=rcwa.Rectangle(size=(width,300.), center=(150.+gap/2,0.)), material=Material(epsilon)))),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(theta=.2, phi=0., polarization='p'),)),
        outputs=rcwa.Outputs(fields=True, scattering=True))


NUMERICS = rcwa.Numerics(orders=(1,1), factorization='li')


def test_jitted_geometry_and_material_derivatives_match_fresh_solve_differences():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64',True)
    jnp = pytest.importorskip('jax.numpy')
    prepared = rcwa.prepare(pair(), numerics=NUMERICS, backend='jax')
    def loss(v): return prepared.run(pair(v[0],v[1])).T.sum()
    value, derivative = jax.jit(jax.value_and_grad(loss))(jnp.asarray([1.,9.]))
    def fresh(gap, epsilon): return rcwa.run(pair(gap,epsilon), numerics=NUMERICS).T.sum()
    np.testing.assert_allclose(value, fresh(1.,9.), rtol=2e-13)
    h=1e-4
    expected=((fresh(1.+h,9.)-fresh(1.-h,9.))/(2*h), (fresh(1.,9.+h)-fresh(1.,9.-h))/(2*h))
    np.testing.assert_allclose(derivative, expected, rtol=2e-5, atol=2e-9)
    assert np.all(np.abs(expected)>1e-6)


def test_vmap_masks_overlap_and_preserves_valid_gap_and_contact():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64',True)
    jnp = pytest.importorskip('jax.numpy')
    prepared = rcwa.prepare(pair(), numerics=NUMERICS, backend='jax')
    def run(gap):
        result = prepared.run(pair(gap))
        return (result.validity.checks['rectilinear_topology'], result.T,
                result.field(x=[0.],y=[0.],z=[150.]).values,
                result.boundary_fields(side='top',x=[0.],y=[0.]).values,
                tuple(getattr(result.scattering[0], k) for k in ('S11','S12','S21','S22')))
    flags, powers, fields, boundary, scattering = jax.jit(jax.vmap(run))(jnp.asarray([1.,0.,-1.]))
    np.testing.assert_array_equal(flags, [True,True,False])
    for values in (powers, fields, boundary, *scattering):
        assert np.isfinite(np.asarray(values[:2])).all()
        assert np.isnan(np.asarray(values[2])).all()
    for i,gap in enumerate((1.,0.)):
        np.testing.assert_allclose(powers[i], rcwa.run(pair(gap),numerics=NUMERICS).T,rtol=2e-13)
    with pytest.raises(IncompatibleProblemError): prepared.run(pair(-1.))


def test_contact_derivative_agrees_with_the_gap_opening_side():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64',True)
    prepared = rcwa.prepare(pair(),numerics=NUMERICS,backend='jax')
    derivative = jax.jit(jax.grad(lambda g: prepared.run(pair(g)).T.sum()))(0.)
    fresh=lambda g: rcwa.run(pair(g),numerics=NUMERICS).T.sum()
    h=1e-4
    expected=(-3*fresh(0.)+4*fresh(h)-fresh(2*h))/(2*h)
    assert abs(expected)>1e-6
    np.testing.assert_allclose(derivative,expected,rtol=2e-5,atol=2e-9)


def test_size_guard_rejects_collapse_or_escape_from_the_prepared_cell():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64',True)
    prepared = rcwa.prepare(pair(gap=10.),numerics=NUMERICS,backend='jax')
    def calculate(width):
        result=prepared.run(pair(gap=10.,width=width))
        return result.validity.checks['rectilinear_topology'], result.R
    compiled=jax.jit(calculate)
    assert bool(compiled(290.)[0])
    for width in (0.,-1.,800.):
        valid,power=compiled(width)
        assert not bool(valid) and np.isnan(power).all()


def test_period_and_orientation_transforms_still_require_their_own_guard():
    jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
    p=pair();prepared=rcwa.prepare(p,numerics=NUMERICS,backend='jax')
    with pytest.raises(UnsupportedFeatureError,match='topology guard'):
        jax.jit(lambda period: prepared.run(p.replace(stack=p.stack.replace(period=(period,700.)))).T)(700.)
