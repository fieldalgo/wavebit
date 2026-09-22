"""Independent geometric and algebraic checks of the constitutive comparator."""
from types import SimpleNamespace
import numpy as np
import pytest
from wavebit.fd._runtime import runtime
from wavebit.fd._shapes import Sphere
from studies.fdtd_subpixel_constitutive import (
    sphere_surface_moments, cross_action, spectral_enclosure, _TensorRuntime,
)
from fd_helpers import to_numpy


def evaluate(backend, function, *values):
    rt = runtime('jax' if backend == 'jit' else backend)
    arrays = [rt.array(v, 'float64') for v in values]
    if backend == 'jit':
        import jax
        function = jax.jit(function, static_argnums=0)
    return function(rt, *arrays)


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'jit', 'torch'))
def test_spherical_cap_moments_follow_independent_polar_integrals(backend):
    # Integrate z=R*cos(theta), dA=R dz dphi over z>=a:
    # S=2*pi*R*(R-a), Mz=2*pi*(R^3-a^3)/(3R), Mx=My=(S-Mz)/2.
    # A cube of side 4R contains the entire cap and clips only its lower z face.
    # Measured arbitrary-patch quadrature error/dx^2 <=1.33e-13 at R/dx<=20;
    # this cap fixture's four-backend maximum is 3.56e-15.
    for a in (-1., -.25, .6):
        center = np.array([.13, -.21, .07])
        point = center+np.array([0., 0., a+2.])
        expected_area = 2*np.pi*(1-a)
        mz = 2*np.pi*(1-a**3)/3
        expected = np.array([(expected_area-mz)/2]*2+[mz])
        area, moment = evaluate(backend, lambda rt, p: sphere_surface_moments(
            Sphere(1., tuple(center)), p, 4.), point[None])
        assert np.max(abs(to_numpy(moment)[0]-expected)) < 3e-12
        assert abs(float(to_numpy(area)[0])-expected_area) < 3e-12


def test_curved_cap_keeps_a_surface_when_the_tangent_plane_misses_it():
    body, h = Sphere(.25), .05
    point = np.array([[.225, 0., 0.]])
    fill = body.exact_fill(point, h)[0]
    area, moment = sphere_surface_moments(body, point, h)
    assert 0 < fill < 1
    # This patch projects across the whole yz square; area >= its projection.
    assert area[0] >= h*h
    assert moment[0, 0] > moment[0, 1]+moment[0, 2]
    # Strictly interior and exterior cubes contain no surface, even if an
    # inclusion/exclusion volume leaves a floating-point cancellation residue.
    area, moment = sphere_surface_moments(body, np.array([[0.,0.,0.], [.7,0.,0.]]), h)
    assert np.array_equal(area, np.zeros(2))
    assert np.array_equal(moment, np.zeros((2,3)))


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'jit', 'torch'))
def test_cross_action_matches_the_staggered_fourier_symbol(backend):
    # A smooth Fourier wave on each component's OWN half-cell position. The
    # node/edge averages multiply its amplitude by cos(k_j h/2) cos(k_i h/2).
    n = 7
    z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    wave = 2*np.pi*np.array([1, 2, 3])/n
    phase = wave[0]*x+wave[1]*y+wave[2]*z
    amplitude = np.array([.7, -.3, 1.2])
    vector = np.stack([amplitude[i]*np.cos(phase+wave[i]/2) for i in range(3)])
    matrix = np.array([[0., .2, -.13], [.2, 0., .07], [-.13, .07, 0.]])
    cross = np.array([matrix[0,1],matrix[0,2],matrix[1,2]])[:,None,None,None]*np.ones_like(vector)
    c = np.cos(wave/2)
    expected = np.stack([np.cos(phase+wave[i]/2)*c[i]*np.dot(matrix[i], c*amplitude)
                         for i in range(3)])
    actual = evaluate(backend, lambda rt, v, k: cross_action(v, k, rt.xp), vector, cross)
    # Four-backend Fourier construction/averaging gap measured at 3.44e-16.
    assert np.max(abs(to_numpy(actual)-expected)) < 3e-14


def test_inhomogeneous_cross_action_is_self_adjoint():
    rng = np.random.default_rng(21)
    cross, u, v = rng.normal(size=(3,3,4,5,6))
    lhs = np.sum(u*cross_action(v, cross, np))
    rhs = np.sum(cross_action(u, cross, np)*v)
    # Exact adjoint identity, measured floating-point relative gap 1.26e-17.
    assert abs(lhs-rhs) < 3e-14*np.linalg.norm(u)*np.linalg.norm(v)


def test_runtime_adds_cross_terms_to_only_the_electric_update():
    # Deliberately equal-valued, distinct E/H coefficient objects. The adapter
    # must select by the kernel's static object, not by shape or array value.
    shape = (3,4,5,6)
    be, bh = np.ones(shape), np.ones(shape)
    base = SimpleNamespace(xp=np, update=lambda u,c,a,b:a*u+b*c)
    cross = np.array([.2,-.13,.07])[:,None,None,None]*np.ones(shape)
    rt = _TensorRuntime(base, be, cross, .4)
    vector = np.array([.7,-.3,1.2])[:,None,None,None]*np.ones(shape)
    expected_cross = np.array([.2*(-.3)-.13*1.2, .2*.7+.07*1.2, -.13*.7+.07*(-.3)])
    e = rt.update(np.zeros(shape), vector, 1., be)
    h = rt.update(np.zeros(shape), vector, 1., bh)
    assert np.max(abs(e-(vector+.4*expected_cross[:,None,None,None]))) < 2e-15
    assert np.array_equal(h, vector)


def test_full_flat_interface_response_preserves_normal_and_tangential_fields():
    from wavebit.fd._lower.coefficients import mix
    # Continuity of D_normal and E_tangent independently gives the effective
    # normal inverse epsilon <1/eps> and tangential inverse epsilon 1/<eps>.
    eps, f = 2.25, .5
    n = np.array([1.,1.,0.])/np.sqrt(2)
    t = np.array([-1.,1.,0.])/np.sqrt(2)
    normal, tangent = f/eps+(1-f), 1/(f*eps+(1-f))
    diagonal = 1/mix(np, np.ones(3), eps, f, n*n)
    cross = np.array([(normal-tangent)*n[i]*n[j] for i,j in ((0,1),(0,2),(1,2))])
    shape = (3,4,5,6)
    cross = cross[:,None,None,None]*np.ones(shape)
    enclosure = spectral_enclosure(diagonal[:,None,None,None]*np.ones(shape), cross)
    assert np.max(abs(np.array(enclosure)-[tangent, normal])) < 2e-15
    for vector, value in ((n,normal),(t,tangent)):
        d = vector[:,None,None,None]*np.ones(shape)
        electric = diagonal[:,None,None,None]*d+cross_action(d,cross,np)
        assert np.max(abs(electric-value*d)) < 2e-15
        # Diagonal truncation has a finite local defect even with an exact
        # flat normal: 7.40% normal, 8.68% tangential for this fixture.
        assert np.linalg.norm(diagonal*vector-value*vector)/value > .07
    # For n=(1,1,1)/sqrt(3), every diagonal is b+(a-b)/3 and
    # each row has two cross entries (a-b)/3. Unlike the 2-D normal,
    # no uncoupled z row hides an omitted lower-bound radius.
    delta = normal-tangent
    diagonal3 = np.full(shape, tangent+delta/3)
    cross3 = np.full(shape, delta/3)
    assert np.max(abs(np.array(spectral_enclosure(diagonal3, cross3))
                      -[tangent-delta/3, normal])) < 2e-15
