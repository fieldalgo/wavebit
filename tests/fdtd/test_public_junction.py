"""Public junction policy belongs to flush standing bodies, not every structure."""
import pytest

from wavebit import Material, Units, fdtd as fd
from wavebit.errors import ValidationError
from wavebit.fdtd._lower_public import lower_problem
from wavebit.fdtd import _legacy as legacy
from tests.fdtd.test_fd_junction import quiet


def example(rule='staircase', reverse=False, continuation=False, extra=True):
    plane = .0125
    sub = fd.Structure('substrate', shape=fd.Box(size=(2., 2., 1.2+plane),
        center=(0., 0., (plane-1.2)/2)), material=Material(epsilon=2.25),
        continuation=('z-',) if continuation else ())
    pillar = fd.Structure('pillar', shape=fd.Cylinder(radius=.15, height=.4,
        center=(.017, -.013, plane+.2)), material=Material(epsilon=4.))
    structures = (pillar, sub) if reverse else (sub, pillar)
    if extra:
        structures += (fd.Structure('ball', shape=fd.Sphere(radius=.11, center=(.5,.3,.2)), material=Material(epsilon=3.)),)
    p = fd.Problem(units=Units(length='um'), scene=fd.Scene(
        domain=fd.Domain(size=(1.6,1.6,1.2)), structures=structures,
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
            z=(fd.Open(), fd.PEC()) if continuation else fd.PEC())))
    n = fd.Numerics(grid=fd.UniformGrid(spacing=.05), duration=fd.Duration(steps=2), junction=rule, absorber=fd.PML(wavelength=.8))
    return p, n


@pytest.mark.parametrize('kind', ['cylinder', 'box'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('continuation', [False, True])
def test_only_standing_body_gets_staircase(reverse, continuation, kind):
    p, n = example(reverse=reverse, continuation=continuation)
    if kind == 'box':
        p = p.replace(scene=p.scene.replace(structures=tuple(
            s.replace(shape=fd.Box(size=(.3,.3,.4), center=s.shape.center))
            if s.name == 'pillar' else s for s in p.scene.structures)))
    scene, _ = lower_problem(p, n)
    assert [s.junction for s in scene.structures] == (['staircase','split','split'] if reverse else ['split','staircase','split'])
    plan = quiet(legacy.plan, scene)
    assert len(plan.junctions) == 1
    assert plan.junctions[0].rule == 'staircase'
    assert fd.check(p, numerics=n).ok


def test_refuse_reaches_actual_pair_instead_of_rejecting_unrelated_sphere():
    p, n = example('refuse')
    scene, _ = lower_problem(p, n)
    assert scene.structures[1].junction == 'refuse'
    with pytest.raises(ValueError, match="asked for junction='refuse'"):
        quiet(legacy.plan, scene)


@pytest.mark.parametrize('rule', ['bad', None, 1, [], 'Staircase'])
def test_invalid_policy_has_public_path(rule):
    p, n = example(rule)
    with pytest.raises(ValidationError) as error:
        lower_problem(p, n)
    assert error.value.path == 'numerics.junction'
    assert fd.check(p, numerics=n).issues[0].path == 'numerics.junction'


@pytest.mark.parametrize('rule', ['staircase','refuse'])
def test_unused_policy_has_public_path(rule):
    p, n = example(rule)
    p = p.replace(scene=p.scene.replace(structures=(p.scene.structures[-1],)))
    with pytest.raises(ValidationError, match='no supported flush pair') as error:
        lower_problem(p, n)
    assert error.value.path == 'numerics.junction'
    assert fd.check(p, numerics=n).issues[0].path == 'numerics.junction'


@pytest.mark.parametrize('continuation', [False, True])
def test_two_standing_bodies_share_one_substrate(continuation):
    p, n = example(extra=False, continuation=continuation)
    sub, pillar = p.scene.structures
    left = pillar.replace(name='left', shape=pillar.shape.replace(center=(-.3,0.,.2125)))
    right = pillar.replace(name='right', shape=pillar.shape.replace(center=(.3,0.,.2125)))
    p = p.replace(scene=p.scene.replace(structures=(left, sub, right)))
    scene, _ = lower_problem(p, n)
    assert [s.junction for s in scene.structures] == ['staircase','split','staircase']
    assert len(quiet(legacy.plan, scene).junctions) == 2
    prepared = quiet(fd.prepare, p, numerics=n)
    reused, _ = prepared._lower(p)
    assert [s.junction for s in reused.structures] == ['staircase','split','staircase']


def test_prepared_split_junction_keeps_traced_radius_and_continuation():
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    p, n = example('split', continuation=True, extra=False)
    n = n.replace(geometry_ranges={'pillar': ((-.16,.19),(-.19,.16),(.0115,.4135))})
    prepared = quiet(fd.prepare, p, numerics=n, backend='jax')
    def lower(radius):
        sub, pillar = p.scene.structures
        live = p.replace(scene=p.scene.replace(structures=(sub,
            pillar.replace(shape=pillar.shape.replace(radius=radius)))))
        scene, _ = prepared._lower(live)
        return scene.structures[1].shape.radius
    assert float(jax.jit(lower)(jnp.asarray(.15))) == .15
    assert float(jax.grad(lower)(jnp.asarray(.15))) == 1.


def test_default_without_pairs_does_not_inspect_host_roles(monkeypatch):
    from wavebit.fdtd._prepare import junction
    p, n = example('split')
    p = p.replace(scene=p.scene.replace(structures=(p.scene.structures[-1],)))
    def forbidden(*args):
        pytest.fail('default policy must not compare host geometry')
    monkeypatch.setattr(junction, '_roles', forbidden)
    scene, _ = lower_problem(p, n)
    assert scene.structures[0].junction == 'split'


def test_nonflush_pair_does_not_consume_policy():
    p, n = example(extra=False)
    sub, pillar = p.scene.structures
    pillar = pillar.replace(shape=pillar.shape.replace(center=(.017,-.013,.2225)))
    p = p.replace(scene=p.scene.replace(structures=(sub,pillar)))
    with pytest.raises(ValidationError, match='no supported flush pair') as error:
        lower_problem(p, n)
    assert error.value.path == 'numerics.junction'
    assert fd.check(p, numerics=n).issues[0].path == 'numerics.junction'


def test_prepared_staircase_rejects_traced_standing_body_explicitly():
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from wavebit.errors import UnsupportedFeatureError
    p, n = example('staircase', continuation=True, extra=False)
    n = n.replace(geometry_ranges={'pillar': ((-.16,.19),(-.19,.16),(.0115,.4135))})
    prepared = quiet(fd.prepare, p, numerics=n, backend='jax')
    scene, _ = prepared._lower(p)
    assert scene.structures[1].junction == 'staircase'
    def lower(radius):
        sub, pillar = p.scene.structures
        live = p.replace(scene=p.scene.replace(structures=(sub,
            pillar.replace(shape=pillar.shape.replace(radius=radius)))))
        return prepared._lower(live)[0].structures[1].shape.radius
    with pytest.raises(UnsupportedFeatureError, match="traced standing body supports junction='split' only") as error:
        jax.jit(lower)(jnp.asarray(.15))
    assert error.value.path == 'numerics.junction'


def test_refuse_cannot_prepare_a_ranged_junction_as_split():
    import jax
    jax.config.update('jax_enable_x64', True)
    p, n = example('refuse', continuation=True, extra=False)
    n = n.replace(geometry_ranges={'pillar': ((-.16,.19),(-.19,.16),(.0115,.4135))})
    with pytest.raises(ValidationError, match="asked for .*refuse"):
        quiet(fd.prepare, p, numerics=n, backend='jax')


def test_static_staircase_survives_an_unrelated_traced_body():
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    p, n = example('staircase', continuation=True)
    sub, pillar, ball = p.scene.structures
    sub = sub.replace(shape=fd.Box(size=(2.,2.,2.0125), center=(0.,0.,-.99375)))
    ball = ball.replace(shape=ball.shape.replace(radius=.1, center=(.45,.3,.8)))
    p = p.replace(scene=p.scene.replace(domain=fd.Domain(size=(1.6,1.6,2.4)),
                                      structures=(sub,pillar,ball)))
    n = n.replace(geometry_ranges={'ball': ((.33,.57),(.18,.42),(.68,.92))})
    prepared = quiet(fd.prepare, p, numerics=n, backend='jax')
    def lower(radius):
        live = p.replace(scene=p.scene.replace(structures=(sub,pillar,
            ball.replace(shape=ball.shape.replace(radius=radius)))))
        scene, _ = prepared._lower(live)
        assert [s.junction for s in scene.structures] == ['split','staircase','split']
        return scene.structures[2].shape.radius
    assert float(jax.jit(lower)(jnp.asarray(.1))) == .1
    assert float(jax.grad(lower)(jnp.asarray(.1))) == 1.
