"""Exact Maxwell fields isolate the production radiation transform from time stepping."""
import numpy as np
import pytest
from references.mie import sphere_efficiencies
from references.mie_fields import exterior_scattered, far_amplitude
from studies import fdtd_subpixel_mie_transform as study


def points():
    rng = np.random.default_rng(23924)  # Reproducible independent test inputs.
    p = rng.normal(size=(40, 3))
    p /= np.linalg.norm(p, axis=-1)[:, None]
    return p*np.linspace(.3, 1.3, len(p))[:, None]


def test_exact_exterior_fields_satisfy_both_maxwell_curls_and_divergences():
    p = points()
    # A 1e-2..1e-5 sweep gave its smallest maximum at h=1e-4:
    # 1.46423e-12 normalized residual. Allow >10x this measured floor.
    h = 1e-4
    for eps in (1.1, 2.25, 12.):
        E, H = exterior_scattered(p, eps)
        derivatives = []
        for a in range(3):
            step = np.eye(3)[a]*h
            fm2, fm = np.array(exterior_scattered(p-2*step, eps)), np.array(exterior_scattered(p-step, eps))
            fp, fp2 = np.array(exterior_scattered(p+step, eps)), np.array(exterior_scattered(p+2*step, eps))
            derivatives.append((fm2-8*fm+8*fp-fp2)/(12*h))
        D = np.array(derivatives)
        curl = np.stack((D[1, :, :, 2]-D[2, :, :, 1], D[2, :, :, 0]-D[0, :, :, 2],
                         D[0, :, :, 1]-D[1, :, :, 0]), axis=-1)
        divergence = D[0, :, :, 0]+D[1, :, :, 1]+D[2, :, :, 2]
        scale = 2*np.pi*max(np.max(abs(E)), np.max(abs(H)))
        assert np.max(abs(curl-np.array((2j*np.pi*H, -2j*np.pi*E))))/scale < 2e-11
        assert np.max(abs(divergence))/scale < 2e-11


def test_exterior_series_tail_power_and_far_limit():
    t = np.radians(np.array((0., 13., 37., 90., 133., 177., 180.)))[:, None]
    p = np.radians(np.array((0., 31., 90., 201.)))[None, :]
    rhat = np.stack(np.broadcast_arrays(np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)), axis=-1)
    that = np.stack(np.broadcast_arrays(np.cos(t)*np.cos(p), np.cos(t)*np.sin(p), -np.sin(t)), axis=-1)
    phat = np.stack(np.broadcast_arrays(-np.sin(p), np.cos(p), 0*t), axis=-1)
    for eps in (1.1, 2.25, 12.):
        a = np.array(exterior_scattered(.250001*rhat, eps))
        b = np.array(exterior_scattered(.250001*rhat, eps, extra_orders=16))
        # +8 versus +16 terms: <=7.15e-17 in the preliminary near-surface sweep.
        assert np.max(abs(a-b))/np.max(abs(b)) < 1e-14
        expected = np.array(far_amplitude(np.degrees(t), np.degrees(p), eps))
        values = []
        for r in (16., 32., 64., 128.):
            E, _ = exterior_scattered(r*rhat, eps)
            values.append(np.array((np.sum(E*that, axis=-1), np.sum(E*phat, axis=-1)))
                          *r*np.exp(-2j*np.pi*r))
        # Cubic interpolation to 1/r=0: coefficients from these four radii.
        limit = (-values[0]+14*values[1]-56*values[2]+64*values[3])/21
        # Measured maximum 2.192e-10 from the remaining O(r^-4) term.
        assert np.max(abs(limit-expected))/np.max(abs(expected)) < 1e-9
        for r in (.3, .65, 1.1):
            mu, w = np.polynomial.legendre.leggauss(16)
            ph = 2*np.pi*np.arange(32)/32
            n = np.stack(np.broadcast_arrays(np.sqrt(1-mu[:, None]**2)*np.cos(ph),
                np.sqrt(1-mu[:, None]**2)*np.sin(ph), mu[:, None]), axis=-1)
            E, H = exterior_scattered(r*n, eps)
            power = .5*np.sum(np.real(np.cross(E, np.conj(H)))*n, axis=-1)
            measured = np.sum(power*w[:, None])*2*np.pi/32*r**2
            expected_power = .5*np.pi*.25**2*sphere_efficiencies(np.sqrt(eps), np.pi/2)[1]
            # 16/32/64 quadrature orders measured maximum 4.56e-15.
            assert abs(measured/expected_power-1) < 3e-14


def test_translation_preserves_global_incident_phase():
    p = points()
    center = np.array((.0125, .02, -.013))
    original = np.array(exterior_scattered(p, 2.25))
    translated = np.array(exterior_scattered(p+center, 2.25, center=center))
    # Measured maximum translation discrepancy 7.84327e-17; allow >10x roundoff.
    np.testing.assert_allclose(translated, original*np.exp(2j*np.pi*center[2]), rtol=0, atol=2e-14)
    theta, phi = np.array((0., 37., 90., 133., 180.)), 73.
    t, ph = np.radians(theta), np.radians(phi)
    direction = np.stack((np.sin(t)*np.cos(ph), np.sin(t)*np.sin(ph), np.cos(t)), axis=-1)
    original = np.array(far_amplitude(theta, phi, 2.25))
    translated = np.array(far_amplitude(theta, phi, 2.25, center=center))
    np.testing.assert_allclose(translated, original*np.exp(2j*np.pi*(center[2]-direction@center)), rtol=0, atol=2e-14)


def test_independent_continuum_surface_recovers_mie_phase_and_polarization():
    theta, phi = np.array((0., 31., 90., 147., 180.))[:, None], np.array((0., 73., 90., 215.))[None, :]
    for eps in (1.1, 2.25, 12.):
        for center in ((0., 0., 0.), (.0125, .02, -.013)):
            actual = np.array(study.gauss_surface(eps, .65, theta, phi, 24, center))
            expected = np.array(far_amplitude(theta, phi, eps, center=center))
            # Orders 12/24/48 gave <=7.25e-7/7.49e-15/1.02e-14.
            assert np.max(abs(actual-expected))/np.max(abs(expected)) < 5e-14


def test_sampler_reads_the_physical_staggered_component_positions():
    grid, faces, accs = study.records(20, 2.25, .65, study.PLACEMENTS[1])
    center = np.array(study.PLACEMENTS[1])/20
    for face, _, layout in faces:
        for field in ('E', 'H'):
            for suffix, component in (('u', layout.u), ('v', layout.v)):
                array = accs[face][field+suffix]
                # Independently reconstruct one corner sample on every stored
                # normal plane from the documented Yee placement, not parts().
                for plane in range(1 if field == 'E' else 2):
                    point = []
                    for axis in range(3):
                        if axis == layout.P:
                            pos = layout.k if field == 'E' else layout.k-.5+plane
                        else:
                            half = (axis == component) if field == 'E' else (axis != component)
                            pos = (layout.half[axis][0]+.5) if half else layout.nodes[axis][0]
                        point.append(grid.low[axis]+pos*grid.dx)
                    expected = exterior_scattered(point, 2.25, center=center)[field == 'H'][component]
                    index = [0, 0, 0, 0]
                    index[3-layout.P] = plane
                    # Scalar versus vectorized maximum: 2.79716e-17.
                    assert abs(array[tuple(index)]-expected) < 2e-14


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'jax.jit', 'torch'))
def test_exact_surface_transform_backends_and_absolute_fields(backend):
    rt = study.runtime(backend.split('.')[0])
    theta, phi = np.array((0., 31., 90., 147., 180.))[:, None], np.array((0., 73., 90., 215.))[None, :]
    for offset in study.PLACEMENTS:
        grid, faces, accs = study.records(20, 1.1, .65, offset)
        reference = study.far_field(study.runtime('numpy'), accs, faces, grid, (1.,), 1., 1., theta, phi)
        arrays = {key: {name: rt.array(val, 'complex128') for name, val in part.items()} for key, part in accs.items()}
        def transform(a):
            return study.far_field(rt, a, faces, grid, (1.,), 1., 1., theta, phi)
        if backend == 'jax.jit':
            import jax
            transform = jax.jit(transform)
        actual = transform(arrays)
        actual = [np.asarray(v.detach() if hasattr(v, 'detach') else v) for v in actual]
        for a, b in zip(actual, reference):
            # Twelve backend/placement probes measured <=1.073e-17 absolute.
            np.testing.assert_allclose(a, b, rtol=0, atol=5e-16)
        exact = np.array(far_amplitude(theta, phi, 1.1, center=np.array(offset)/20))
        # Same fixture's full two angular cuts measured <0.9% complex peak
        # error at 20 cells. This absolute oracle also catches shared defects.
        assert np.max(abs(np.array(actual[:2])[:, 0]-exact))/np.max(abs(exact)) < .01


def test_report_normalization_and_collocated_diagnostic():
    staggered = study.measure(40, 1.1)
    collocated = study.measure(40, 1.1, collocated=True)
    # Initial exact-data probe: power -0.20659%, optical +0.25017%,
    # closure +0.45770%; collocated power -0.003473%, optical -0.07653%.
    assert abs(staggered['power_vs_exact']) < .003
    assert abs(staggered['optical_vs_exact']) < .003
    assert abs(staggered['optical_closure']) < .005
    assert abs(collocated['power_vs_exact']) < .00005
    assert abs(collocated['optical_vs_exact']) < .001
    assert abs(staggered['forward_error']) < .0007


def test_exterior_oracle_refuses_invalid_samples_before_evaluation():
    with pytest.raises(ValueError, match='ExteriorMieSurfaceRefused'):
        study.records(20, half=.2)
    with pytest.raises(ValueError, match='ExteriorMieSampleRefused'):
        exterior_scattered([[0., 0., .25]])
    with pytest.raises(ValueError, match='ExteriorMieInputRefused'):
        exterior_scattered([[0., 0., np.nan]])
