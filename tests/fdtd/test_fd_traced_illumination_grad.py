"""Public geometry derivatives under plane-wave illumination, read by frequency-domain monitors.

The scenes and objectives are the measured ones in ``studies/fdtd/traced_illumination/fdtd_traced_illumination_probe.py``
(results in ``studies/fdtd/traced_illumination/records/fdtd_traced_illumination_probe.json``). The plane-wave medium
certificate is ``src/wavebit/fdtd/_prepare/shape_ranges.py::plane_media``.
"""
import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd import _post
from wavebit.fdtd._kernel.grid import Grid
from wavebit.fdtd._prepare.shape_ranges import planning_scene
from wavebit.fdtd._runtime import runtime
from studies.fdtd.traced_illumination.fdtd_traced_illumination_probe import POINTS, objectives, scene
from tests.fdtd.test_fd_sphere_range import quiet


#: Central-difference steps, measured by the probe's largest relative Jacobian error at
#: 1e-4, 1e-5, 3e-6 and 1e-6: pw 2.02e-6, 2.02e-8, 1.82e-9, 1.22e-8 (cancellation below its
#: valley at 3e-6); tfsf 5.82e-7, 5.82e-9, 5.16e-10, 4.42e-11 (still truncation, h^2).
#: The existing 2e-6 relative budget is kept, with no absolute allowance.
STEPS = {'pw': 3e-6, 'tfsf': 1e-6}


def _jacobian(family, backend, p):
    runtime(backend)
    if backend == 'jax':
        import jax
        import jax.numpy as jnp

        def fn(v):
            return jnp.stack(quiet(objectives, family, v, 'jax', p))
        x = jnp.asarray(POINTS[family])
        return np.asarray(fn(x)), np.asarray(quiet(jax.jacrev(fn), x))
    import torch
    v = torch.tensor(POINTS[family], dtype=torch.float64, requires_grad=True)
    out = quiet(objectives, family, v, 'torch', p)
    assert all(o.requires_grad for o in out), \
        'An illuminated objective lost its geometry derivative graph.'
    rows = [torch.autograd.grad(o, v, retain_graph=True)[0].detach().numpy() for o in out]
    return np.array([float(o.detach()) for o in out]), np.array(rows)


def _check(family, backend):
    p = quiet(fd.plan, scene(family))
    h = STEPS[family]
    point = POINTS[family]

    def values(v):
        return np.array([float(x) for x in quiet(objectives, family, v, 'numpy', p)])
    expected = np.array([(values(point + d) - values(point - d)) / (2 * h)
                         for d in np.eye(len(point)) * h]).T
    want = values(point)
    got, jacobian = _jacobian(family, backend, p)
    np.testing.assert_allclose(got, want, rtol=128 * np.finfo(float).eps, atol=0.)
    assert np.all(expected != 0)
    np.testing.assert_allclose(jacobian, expected, rtol=2e-6, atol=0.)
    if backend == 'torch':
        # Forward mode is a separate route: a tangent that a host conversion drops gives 0.
        import torch
        x = torch.tensor(point, dtype=torch.float64)
        _, tangents = quiet(torch.func.jvp, lambda v: torch.stack(
            quiet(objectives, family, v, 'torch', p)), (x,), (torch.ones_like(x),))
        np.testing.assert_allclose(tangents.numpy(), expected.sum(axis=1), rtol=2e-6, atol=0.)


@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_a_pillar_under_a_bloch_plane_wave_differentiates_flux_and_orders(backend):
    """Radius and height of a ranged pillar in an oblique (complex) Bloch cell: the normalised
    transmitted flux, the transmitted and the reflected zero order."""
    _check('pw', backend)


@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_a_body_inside_a_tfsf_box_differentiates_cross_section_and_far_field(backend):
    """The three sizes of a turned, ranged Box in an open scene: the scattered flux through a
    closed box over the incident intensity, and a FarField box's intensity."""
    _check('tfsf', backend)


def _band():
    return fd.Band(wavelength=1., df=.15)


def stack(sheet=.9, plane=-.6, v=(.12, .11)):
    """A ranged pillar (structures[0]) above a ranged ball (structures[1]) in a Bloch cell.

    60 cells along z (low -1.5, dx .05). Reserved supports: pillar z cells 31..40, ball
    19..28 (half-open halos (30, 42) and (18, 30) meet without sharing a cell). The sheet at
    ``z = sheet`` is read at layers ``(k - 1, k)``, ``k = (sheet + 1.5) / .05``, and the
    transmit plane likewise; the reflect plane at z = 1.0 reads (49, 50). Only planned here.
    """
    pw = fd.PlaneWave('pw', z=sheet, direction='-z', theta=25., phi=30., polarization='s',
                      band=_band())
    pillar = fd.Cylinder(radius=v[0], height=.3, center=(.01, -.007, .3),
                         range=((-.145, .145), (-.145, .145), (.105, .495)))
    ball = fd.Sphere(radius=v[1], center=(0., .004, -.3),
                     range=((-.145, .145), (-.145, .145), (-.495, -.105)))
    return fd.Scene(
        domain=fd.Domain(size=(.6, .6, 3.)), resolution=fd.Resolution(dx=.05),
        boundaries=fd.Boundaries(**fd.Bloch.from_source(pw),
                                 z=fd.PML(cells=8, wavelength=1., allow_thin=True)),
        run=fd.Run(steps=1, courant=.5), sources=(pw,),
        structures=tuple(fd.Structure(s, fd.Material(eps=2.25)) for s in (pillar, ball)),
        monitors=(fd.Diffraction('T', z=plane, side='transmit', wavelengths=(1.,), orders=0),
                  fd.Diffraction('R', z=1., side='reflect', wavelengths=(1.,), orders=0)))


def _traced(make, index, **kwargs):
    import torch
    values = [.12, .11]
    values[index] = torch.tensor(values[index], dtype=torch.float64, requires_grad=True)
    return make(v=values, **kwargs)


def _admitted(traced, p):
    """``planning_scene`` returned a concrete scene; a refusal here is the failed assertion."""
    try:
        concrete = planning_scene(traced, p)
    except ValueError as error:
        raise AssertionError(f'refused a certified traced scene: {error}') from error
    assert concrete is not traced and concrete.sources == traced.sources
    return concrete


def test_plane_wave_media_must_lie_outside_every_traced_support():
    """The layers a PlaneWave normalisation and a Diffraction plane read must not be in a traced
    body's reserved support, even where the concrete seed leaves them in the background.

    The sheet at .55 reads layers (40, 41) and the pillar's support ends at 40 (its sheet door,
    on cells 41..42, passes); the transmit plane at -.55 reads (18, 19) and the ball's support
    starts at 19. Each concrete scene plans, both bodies being absent from those layers; a
    concrete partner's support may reach them, and a support ending at the read layer may not.
    """
    cases = ((dict(sheet=.55), 0, "PlaneWave('pw') normalisation", '(40, 41)'),
             (dict(plane=-.55), 1, "Diffraction('T')", '(18, 19)'))
    for kwargs, index, label, cells in cases:
        p = quiet(fd.plan, stack(**kwargs))
        with pytest.raises(ValueError, match=r'traced structures\[%d\]' % index) as refused:
            planning_scene(_traced(stack, index, **kwargs), p)
        assert label in str(refused.value) and cells in str(refused.value)
        _admitted(_traced(stack, 1 - index, **kwargs), p)       # the concrete partner reaches
    for kwargs in (dict(), dict(sheet=.6), dict(plane=-.6)):   # (41, 42) and (17, 18): clear
        p = quiet(fd.plan, stack(**kwargs))
        for index in (0, 1):
            concrete = _admitted(_traced(stack, index, **kwargs), p)
            assert concrete == stack(**kwargs)


def beside(v=(.1,)):
    """A ranged ball outside an axis-aligned TFSF box on x, level with its entry face.

    44 cells a side (low -1.1). The box's nodes are 18..26, so a '-z' wave enters at node 26
    and its normalisation reads layers (25, 26); the ball's support is cells 27..33 on x (clear
    of the box shell) and 22..29 on z.
    """
    tfsf = fd.TFSF('t', box=fd.Region(size=(.4,) * 3), direction='-z', polarization='x',
                   band=_band())
    ball = fd.Sphere(radius=v[0], center=(.42, 0., .2),
                     range=((.305, .535), (-.115, .115), (.095, .305)))
    return fd.Scene(domain=fd.Domain(size=(2.2,) * 3), resolution=fd.Resolution(dx=.05),
                    boundaries=fd.Boundaries.open(fd.PML(cells=8, wavelength=1.,
                                                         allow_thin=True)),
                    run=fd.Run(steps=1, courant=.5), sources=(tfsf,),
                    structures=(fd.Structure(ball, fd.Material(eps=2.25)),))


def test_a_tfsf_normalisation_plane_must_lie_outside_a_traced_support():
    """A body outside a TFSF box's shell but level with its entry face changes the medium the
    incident wave's normalisation reads. The concrete scene plans (its normalisation refuses
    only when asked, after the run); the traced one is refused before any step."""
    import torch
    p = quiet(fd.plan, beside())
    traced = beside(v=(torch.tensor(.1, dtype=torch.float64, requires_grad=True),))
    with pytest.raises(ValueError, match=r"TFSF\('t'\) normalisation") as refused:
        planning_scene(traced, p)
    assert '(25, 26)' in str(refused.value) and 'cells 22..29' in str(refused.value)


def test_a_plane_wave_medium_is_never_read_through_a_traced_shape():
    """``layer_share`` reads fills on the host: a Torch forward tangent would vanish there."""
    import torch
    import torch.autograd.forward_ad as forward
    grid = Grid.from_domain((1.6,) * 3, (0., 0., 0.), .1)
    with forward.dual_level():
        radius = forward.make_dual(torch.tensor(.3, dtype=torch.float64),
                                   torch.tensor(1., dtype=torch.float64))
        ball = fd.Sphere(radius=radius, center=(0., 0., 0.), range=((-.5, .5),) * 3)
        with pytest.raises(ValueError, match='depends on a traced Sphere'):
            _post.layer_share(ball, grid, 2, 8)
    assert _post.layer_share(ball.replace(radius=.3), grid, 2, 8) == 'partial'
