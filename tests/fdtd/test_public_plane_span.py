"""Public full-cell ports must not silently discard finite Plane spans."""
import pytest

from wavebit import Units, fdtd as fd
from wavebit.errors import UnsupportedFeatureError
from wavebit.fdtd._lower_public import lower_problem

NUMERICS = fd.Numerics(grid=fd.UniformGrid(spacing=.025),
                       duration=fd.Duration(steps=2), absorber=fd.PML(cells=12))


def problem(axis='z', span=None, target='source'):
    transverse = [a for a in 'xyz' if a != axis]
    source_plane = fd.Plane(**{axis: .5}, **(span if target == 'source' else {}))
    output_plane = fd.Plane(**{axis: .3}, **(span if target == 'output' else {}))
    return fd.Problem(units=Units(length='um'), scene=fd.Scene(
        domain=fd.Domain(size=tuple(1.6 if a == axis else .2 for a in 'xyz')),
        boundaries=fd.Boundaries(**{axis: fd.Open(),
                                    **{a: fd.Periodic() for a in transverse}})),
        sources=(fd.PlaneWave('light', plane=source_plane, direction='-'+axis,
                              pulse=fd.GaussianPulse(wavelength=.8, frequency_sigma_hz=4e13)),),
        outputs=fd.Outputs(monitors=(fd.DiffractionMonitor('reflection', plane=output_plane,
            wavelengths=(.8,), source='light', wave='reflected'),)))


@pytest.mark.parametrize('axis', 'xyz')
@pytest.mark.parametrize('target', ['source', 'output'])
@pytest.mark.parametrize('extent', ['first', 'second', 'both', 'full'])
def test_finite_port_span_is_structured_at_every_lifecycle_door(axis, target, extent):
    transverse = [a for a in 'xyz' if a != axis]
    axes = transverse if extent in ('both', 'full') else [transverse[extent == 'second']]
    span = {a: (-.1, .1) if extent == 'full' else (-.05, .05) for a in axes}
    p = problem(axis, span, target)
    path = 'sources.light.plane' if target == 'source' else 'outputs.reflection.plane'
    report = fd.check(p, numerics=NUMERICS)
    assert not report.ok
    (issue,) = report.issues
    assert (issue.code, issue.path) == ('plane_span', path)
    assert issue.actual == span
    assert issue.expected == 'no transverse spans'
    for door in (fd.prepare, fd.run):
        with pytest.raises(UnsupportedFeatureError) as caught:
            door(p, numerics=NUMERICS)
        assert (caught.value.code, caught.value.path) == ('plane_span', path)
        assert caught.value.actual == issue.actual
        assert caught.value.expected == issue.expected


@pytest.mark.parametrize('axis', 'xyz')
def test_unbounded_ports_and_finite_field_flux_remain_supported(axis):
    p = problem(axis, {}, 'source')
    transverse = [a for a in 'xyz' if a != axis]
    plane = fd.Plane(**{axis: 0., **{a: (-.05, .05) for a in transverse}})
    p = p.replace(outputs=fd.Outputs(monitors=p.outputs.monitors + (
        fd.FieldMonitor('field', plane=plane, wavelengths=(.8,)),
        fd.FluxMonitor('flux', plane=plane, wavelengths=(.8,)))))
    scene, _ = lower_problem(p, NUMERICS)
    assert scene.monitors[1].plane == plane
    assert scene.monitors[2].plane == plane
    assert fd.check(p, numerics=NUMERICS).ok
    result = fd.run(p, numerics=NUMERICS)
    assert result is not None


@pytest.mark.parametrize('axis', 'xyz')
@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_field_samples_and_flux_integral_keep_the_requested_aperture(axis, backend):
    """A uniform nonzero field gives an independent area oracle for the flux."""
    import numpy as np
    from wavebit.units import Z0

    if backend != 'numpy':
        module = pytest.importorskip(backend)
        if backend == 'jax':
            module.config.update('jax_enable_x64', True)
    normal = 'xyz'.index(axis)
    electric = (normal + 1) % 3
    magnetic = (normal + 2) % 3
    spans = {a: (-.05, .05) for a in 'xyz' if a != axis}
    finite = fd.Plane(**{axis: 0., **spans})
    full = fd.Plane(**{axis: 0.})
    p = fd.Problem(units=Units(length='um'), scene=fd.Scene(
        domain=fd.Domain(size=(.2, .2, .2)),
        boundaries=fd.Boundaries(**{a: fd.Periodic() for a in 'xyz'})),
        initial=fd.InitialField(E=tuple(1. if a == electric else 0. for a in range(3)),
                              H=tuple(1./Z0 if a == magnetic else 0. for a in range(3))),
        outputs=fd.Outputs(monitors=(
            fd.FieldMonitor('field', plane=finite, wavelengths=(.8,)),
            fd.FluxMonitor('finite', plane=finite, wavelengths=(.8,)),
            fd.FluxMonitor('full', plane=full, wavelengths=(.8,)))))
    r = fd.run(p, numerics=NUMERICS, backend=backend)

    def array(value):
        return value.detach().cpu().numpy() if backend == 'torch' else np.asarray(value)

    for component, coords in r['field'].coordinates.items():
        for a in range(3):
            if a == normal:
                assert len(coords[a]) == 1
            else:
                # Four Yee cell centres or five nodes on this 0.1-wide aperture.
                assert len(coords[a]) in (4, 5)
                assert np.min(coords[a]) >= -.05 - 1e-14
                assert np.max(coords[a]) <= .05 + 1e-14
        values = array(r['field'].values[component])
        assert values.shape == (1, *(len(c) for c in reversed(coords)))
    partial = array(r['finite'].energy_density)
    whole = array(r['full'].energy_density)
    assert partial.shape == whole.shape == (1,)
    assert np.all(np.isfinite(whole)) and np.all(whole > 0)
    # The finite aperture has half the width on each transverse axis.
    np.testing.assert_allclose(partial, whole / 4, rtol=2e-14, atol=0.)
