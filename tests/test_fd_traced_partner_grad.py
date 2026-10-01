"""Public geometry derivatives of ranged bodies beside concrete partners without a range.

The scenes and objectives are the measured ones in ``studies/fdtd_traced_partner_probe.py``
(results in ``studies/results/fdtd_traced_partner_probe.json``). The partner certificate is
``wavebit/fd/_lower/shape_ranges.py::planning_scene``.
"""
import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._lower.shape_ranges import planning_scene
from wavebit.fd._runtime import runtime
from studies.fdtd_traced_partner_probe import POINTS, objectives, partners_scene, scene
from test_fd_sphere_range import quiet


#: Central-difference steps, measured by the probe's largest relative Jacobian error at
#: 1e-4, 1e-5, 3e-6 and 1e-6: substrate 2.63e-7, 2.63e-9, 5.71e-10, 8.73e-10; mirror
#: 2.58e-5, 3.49e-7, 3.37e-8, 4.76e-8 -- both valleys at 3e-6. The existing 2e-6 relative
#: budget is kept, with no absolute allowance.
STEPS = {'substrate': 3e-6, 'mirror': 3e-6}


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
        'An objective beside a concrete partner lost its geometry derivative graph.'
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


@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_a_pillar_above_a_snapped_substrate_differentiates_its_orders(backend):
    """Radius and height of a ranged pillar three cells above a cell-snapped glass Slab that
    runs into the bottom PML: the zero order transmitted into the glass and the reflected one."""
    _check('substrate', backend)


@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_a_ball_above_a_dispersive_mirror_differentiates_reflection_and_leakage(backend):
    """Radius and height of a ranged ball above a Drude-metal Slab film in an oblique Bloch cell:
    the reflected zero order and the normalised flux leaking through the film."""
    _check('mirror', backend)


def _lowered(scene_, plan, backend):
    """``fd.lower`` of a certified traced scene; any refusal or error is the failed assertion."""
    try:
        return quiet(fd.lower, scene_, plan, backend)
    except Exception as error:
        raise AssertionError(f'lowering a certified traced scene failed: {error!r}') from error


def test_a_traced_ball_beside_three_partners_lowers_to_the_sum_of_their_changes():
    """An unranged smoothed Sphere, a Region and a Slab beside a traced ball: the combined
    coefficients are the background plus each structure's own change, as for separated
    ranged bodies, and the ball's change carries its derivative."""
    import torch
    p = quiet(fd.plan, partners_scene())
    radius = torch.tensor(.237, dtype=torch.float64, requires_grad=True)
    combined = _lowered(partners_scene(radius), p, 'torch')
    assert combined['be'].requires_grad and combined['bh'].requires_grad
    current = partners_scene(.237)

    def lower(s):
        return quiet(fd.lower, s, quiet(fd.plan, s))
    base = lower(current.replace(structures=()))
    singles = [lower(current.replace(structures=(s,))) for s in current.structures]
    for key in ('be', 'bh'):
        want = base[key] + sum(single[key] - base[key] for single in singles)
        np.testing.assert_allclose(combined[key].detach().numpy(), want, rtol=0,
                                   atol=128 * np.finfo(float).eps * np.max(abs(want)))


def _traced_pillar(**slab):
    import torch
    v = (torch.tensor(.12, dtype=torch.float64, requires_grad=True), .3)
    current = scene('substrate', v)
    if slab:
        current = current.replace(structures=(
            current.structures[0].replace(shape=fd.Slab(**slab)),) + current.structures[1:])
    return current


def _admitted(traced, p):
    try:
        concrete = planning_scene(traced, p)
    except ValueError as error:
        raise AssertionError(f'refused a certified traced scene: {error}') from error
    assert concrete is not traced and concrete.structures[0] == traced.structures[0]
    return concrete


def test_a_concrete_partner_must_stay_a_cell_off_every_traced_halo():
    """The substrate's last cell is 14 and the pillar's halo starts at 16: widened by one cell
    the substrate meets the halo at an index boundary and is admitted, running into the PML
    on its other side. Raised by one cell it shares cell 15 with the halo and is refused, as
    is an unranged smoothed Sphere moved next to the ball of the partners scene."""
    import torch
    p = quiet(fd.plan, scene('substrate'))
    _admitted(_traced_pillar(), p)
    raised = scene('substrate').replace(structures=(
        fd.Structure(fd.Slab(z=(-1.5, -.2)), fd.Material(eps=2.1)),
        scene('substrate').structures[1]))
    q = quiet(fd.plan, raised)
    with pytest.raises(ValueError, match=r'structures\[0\] \(z=.*reaches the interpolation '
                                         r'halo of the traced structures\[1\]'):
        planning_scene(_traced_pillar(z=(-1.5, -.2)), q)
    near = partners_scene().structures
    moved = near[1].replace(shape=near[1].shape.replace(center=(-.05, -.01, .02)))
    concrete = partners_scene().replace(structures=(near[0], moved) + near[2:])
    r = torch.tensor(.237, dtype=torch.float64, requires_grad=True)
    traced = partners_scene(r).replace(structures=(partners_scene(r).structures[0], moved)
                                       + near[2:])
    with pytest.raises(ValueError, match=r'structures\[1\] \(Sphere.*reaches the '
                                         r'interpolation halo'):
        planning_scene(traced, quiet(fd.plan, concrete))


def test_a_traced_partner_material_in_a_pml_or_with_poles_is_refused():
    """A partner's traced eps is planned at the plan's value and lowered live
    (tests/test_fd_traced_material.py), but not this substrate's: it fills the bottom PML,
    whose grazing survey is judged on the host only. A traced pole parameter has no seed."""
    import torch
    p = quiet(fd.plan, scene('substrate'))
    eps = torch.tensor(2.1, dtype=torch.float64, requires_grad=True)
    omega = torch.tensor(20., dtype=torch.float64, requires_grad=True)
    for material, message in ((fd.Material(eps=eps), r'structures\[0\] has a traced material '
                                                     r'value and fills a PML layer'),
                              (fd.Dispersive(eps_inf=1., poles=(fd.Drude(omega_p=omega,
                                                                         gamma=.5),)),
                               r'structures\[0\] has a traced value in an fd.Dispersive')):
        current = _traced_pillar()
        current = current.replace(structures=(current.structures[0].replace(
            material=material),) + current.structures[1:])
        with pytest.raises(ValueError, match=message):
            planning_scene(current, p)


def junction_scene(radius=.2):
    """A concrete pillar flush on a concrete unturned Box substrate (a 'split' junction) and a
    ranged ball clear of both, in a 32 x 16 x 20 cell with periodic x and y."""
    substrate = fd.Box(size=(10., 10., .5), center=(0., 0., -.45))
    pillar = fd.Cylinder(radius=.2, height=.4, center=(0., 0., 0.))
    ball = fd.Sphere(radius=radius, center=(1., 0., .45),
                     range=((.75, 1.25), (-.22, .22), (.22, .68)))
    return fd.Scene(domain=fd.Domain(size=(3.2, 1.6, 2.)), resolution=fd.Resolution(dx=.1),
                    boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.PEC()),
                    run=fd.Run(steps=1),
                    structures=(fd.Structure(substrate, fd.Material(eps=2.25)),
                                fd.Structure(pillar, fd.Material(eps=4.)),
                                fd.Structure(ball, fd.Material(eps=2.25))))


def test_a_flush_junction_beside_a_traced_body_lowers_as_its_concrete_scene():
    """The ball is certified clear of both junction partners. The junction's signature covers
    all scene geometry, so it is compared through the concrete planning scene the plan was
    built from (it was refused outright before tests/test_fd_traced_junction.py); the traced
    lowering at another radius equals the concrete scene planned at that radius."""
    import torch
    p = quiet(fd.plan, junction_scene())
    assert len(p.junctions) == 1
    traced = junction_scene(torch.tensor(.21, dtype=torch.float64, requires_grad=True))
    assert planning_scene(traced, p).structures[:2] == traced.structures[:2]
    got = _lowered(traced, p, 'torch')
    assert got['be'].requires_grad
    concrete = junction_scene(.21)
    want = quiet(fd.lower, concrete, quiet(fd.plan, concrete))
    np.testing.assert_allclose(got['be'].detach().numpy(), want['be'], rtol=0,
                               atol=128 * np.finfo(float).eps * np.max(abs(want['be'])))


def beside_box(v=.2):
    """A ranged Cylinder beside a concrete unranged, unturned Box, far from flush."""
    pillar = fd.Cylinder(radius=v, height=.4, axis='z', center=(-.6, 0., 0.),
                         range=((-.95, -.25), (-.3, .3), (-.3, .3)))
    box = fd.Box(size=(.4, .4, .4), center=(.6, 0., 0.))
    return fd.Scene(domain=fd.Domain(size=(3., 1.6, 1.6)), resolution=fd.Resolution(dx=.1),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    run=fd.Run(steps=1),
                    structures=(fd.Structure(pillar, fd.Material(eps=2.)),
                                fd.Structure(box, fd.Material(eps=2.))))


def test_jitted_lowering_beside_an_unturned_box_partner_reads_no_traced_face():
    """Under ``jax.jit`` no host test may read a traced length: the flush-pair search is not
    asked about a traced body, and the runtime guard skips a partner with no range."""
    runtime('jax')
    import jax
    import jax.numpy as jnp
    p = quiet(fd.plan, beside_box())

    def lower(v):
        params = quiet(fd.lower, beside_box(v), p, 'jax')
        return params['stable'], jnp.sum(params['be'])
    try:
        compiled = jax.jit(lower)
        stable, value = compiled(jnp.asarray(.2))
        bad, _ = compiled(jnp.asarray(.1))      # two cells across: the guard's NaN
    except Exception as error:
        raise AssertionError(f'jitted lowering failed: {error!r}') from error
    assert float(stable) == 1. and np.isfinite(float(value))
    assert np.isnan(float(bad))
