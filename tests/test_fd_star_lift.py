"""The star lift (``wavebit/fd/_lower/stars.py``): the smoothed operator is positive by
construction, and the lift changes nothing it does not have to.

The references here are written from the definitions and share no code with the module: the
star sum is enumerated edge by edge against ``cross_action``'s own dense matrix, and the least
move is found with ``numpy.linalg.eigvalsh`` on each star's explicit matrix (up to 6x6), not with
the module's harmonic-mean test.
"""
import itertools
import os
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._kernel.constitutive import cross_action
from wavebit.fd._lower import stars
from wavebit.fd._lower.coefficients import build
from wavebit.fd._runtime import runtime
from fd_helpers import to_numpy

PAIR = {(0, 1): 0, (0, 2): 1, (1, 2): 2}


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


def _nodes(field, shape):
    """Every node whose star holds an edge of the region: one layer past the far end (the high
    end for E, the low end for H), where there is no cross coefficient and no move."""
    return itertools.product(*[range(0, n + 1) if field == 'E' else range(-1, n) for n in shape])


def _star_edges(field, p, shape):
    """``(axis, (z, y, x))`` of each edge of ``p``'s star inside the region: the own-index edge
    and the other one, ``p - e_a`` for E and ``p + e_a`` for H."""
    out = []
    for a in range(3):
        for step in (0, -1 if field == 'E' else 1):
            q = list(p)
            q[2 - a] += step
            if all(0 <= v < n for v, n in zip(q, shape)):
                out.append((a, tuple(q)))
    return out


def _inside(p, shape):
    return all(0 <= v < n for v, n in zip(p, shape))


def _star_matrix(edges, diagonal, cross, node, p, t):
    """The explicit star: each edge's half at ``t``, every coupling through ``p`` at ``c/4``."""
    m = np.diag([(diagonal[(a,) + q] + t * (node[(a,) + p] - diagonal[(a,) + q])) / 2
                 for a, q in edges])
    for (i, (a, _)), (j, (b, _)) in itertools.combinations(list(enumerate(edges)), 2):
        if a != b:
            m[i, j] = m[j, i] = cross[(PAIR[tuple(sorted((a, b)))],) + p] / 4
    return m


def _dense(diagonal, cross, field):
    """``diag + cross_action`` as a dense matrix, one unit vector at a time."""
    n = diagonal.size
    out = np.empty((n, n))
    for i in range(n):
        v = np.zeros(n)
        v[i] = 1.
        x = v.reshape(diagonal.shape)
        out[:, i] = (diagonal * x + cross_action(x, cross, np, field, bounded=True)).ravel()
    return out


def _tensors(rng, shape, contrast=1.5):
    """Node Kottke tensors (``inv I + contrast n n^T``) and edge diagonals that make stars fail."""
    normal = rng.normal(size=(3,) + shape)
    normal /= np.linalg.norm(normal, axis=0)
    inverse = rng.uniform(.05, 1., size=shape)
    strength = rng.uniform(0., contrast, size=shape)
    node = inverse + strength * normal ** 2
    cross = np.stack([strength * normal[i] * normal[j] for i, j in stars.PAIRS])
    diagonal = rng.uniform(.02, 1., size=(3,) + shape)
    return diagonal, cross, node, inverse


@pytest.mark.parametrize('field', ('E', 'H'))
def test_the_stars_sum_to_the_operator(field):
    """The premise of the proof: ``sum_p P^T S_p(t) P = diag(e + delta) + C`` for any ``t``."""
    rng = np.random.default_rng(71)
    shape = (3, 4, 5)
    diagonal, cross, node, inverse = _tensors(rng, shape)
    moves = rng.uniform(size=shape)
    delta = stars._assemble(field, diagonal, cross, node, inverse, runtime('numpy'), stars.THETA,
                            shape, np.ones(shape, bool), moves,
                            [np.full(shape, float(a)) for a in range(3)], np.zeros(shape, bool))
    size = diagonal.size
    expected = np.zeros((size, size))
    for p in _nodes(field, shape):
        edges = _star_edges(field, p, shape)
        inside = _inside(p, shape)
        rows = [np.ravel_multi_index((a,) + q, diagonal.shape) for a, q in edges]
        star = (_star_matrix(edges, diagonal, cross, node, p, moves[p]) if inside else
                np.diag([diagonal[(a,) + q] / 2 for a, q in edges]))
        expected[np.ix_(rows, rows)] += star
    np.testing.assert_allclose(_dense(diagonal + delta, cross, field), expected,
                               rtol=0, atol=1e-14)


def _least(star, floor):
    """The least ``t`` in ``[0, 1]`` with ``eigvalsh(star(t))[0] > floor / 2``, by bisection."""
    if np.linalg.eigvalsh(star(0.))[0] > floor / 2:
        return 0.
    lo, hi = 0., 1.
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if np.linalg.eigvalsh(star(mid))[0] > floor / 2 else (mid, hi)
    return hi


@pytest.mark.parametrize('field', ('E', 'H'))
def test_a_star_moves_the_least_t_above_its_floor(field):
    """Each star moves the least ``t`` that puts its explicit matrix above half the floor
    ``THETA * min(1/<eps>, its edges' diagonals)``, and each edge gets half of each of its two
    stars' moves."""
    rng = np.random.default_rng(72)
    shape = (3, 4, 5)
    diagonal, cross, node, inverse = _tensors(rng, shape)
    expected, moved = np.zeros_like(diagonal), 0
    for p in np.ndindex(*shape):
        edges = _star_edges(field, p, shape)
        floor = stars.THETA * min([inverse[p]] + [diagonal[(a,) + q] for a, q in edges])
        t = _least(lambda t: _star_matrix(edges, diagonal, cross, node, p, t), floor)
        moved += t > 0
        for a, q in edges:
            expected[(a,) + q] += t * (node[(a,) + p] - diagonal[(a,) + q]) / 2
    out = stars.lift(field, diagonal, cross, node, inverse, runtime('numpy'))
    # The fixture moves some stars and not others, or the comparison proves little.
    assert 0 < moved < np.prod(shape) and out.moved == moved and out.certified
    np.testing.assert_allclose(out.delta, expected, rtol=0, atol=1e-13)


def test_a_star_without_cross_coefficients_never_moves():
    """No cross coefficient, no move: even an edge diagonal far below ``THETA / <eps>``
    (an ``eps = 20`` edge beside a vacuum node) is the star's own smallest eigenvalue."""
    shape = (2, 3, 4)
    diagonal = np.full((3,) + shape, 1 / 20)
    out = stars.lift('E', diagonal, np.zeros((3,) + shape), np.ones((3,) + shape),
                     np.ones(shape), runtime('numpy'))
    assert out.delta is None and out.moved == 0 and out.certified


def _cavity(eps, mu=1.5, center=(.012, .007, -.009)):
    return fd.Scene(domain=fd.Domain(size=(.8,) * 3), resolution=fd.Resolution(dx=.05),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    run=fd.Run(steps=8), structures=(fd.Structure(
                        fd.Sphere(radius=.15, center=center), fd.Material(eps=eps, mu=mu)),))


def _region(scene, plan, field, lifted):
    """The term's inverse-material diagonal over its region, with or without the lift."""
    from wavebit.fd._kernel.observe import InteriorEnergy
    from wavebit.fd._lower import constitutive
    rt = runtime('numpy')
    params = dict(build(scene, plan, rt))
    nodes = {}
    params['constitutive'] = constitutive.lower(scene, plan, rt, nodes=nodes)
    if lifted:
        assert constitutive.lift(plan, params, nodes, rt)
    term = next(t for t in plan.constitutive if t.field == field)
    key = 'be' if field == 'E' else 'bh'
    cross = params['constitutive'][term.name]
    return np.array(np.broadcast_to(InteriorEnergy._cut(params[key], slice(None), term.box)
                                    / term.ratio, cross.shape)), cross


@pytest.mark.parametrize('field', ('E', 'H'))
def test_the_lifted_operator_is_positive_definite(field):
    """The claim itself, on the kernel's own operator: ``eps = mu = 100`` in the 16^3 cavity is
    indefinite unlifted and positive definite lifted. Measured 2026-09-27: smallest eigenvalue
    times 100, E -1.549 -> +0.518 and H -1.831 -> +0.391 at THETA 0.02 (+0.564 and +0.448 at
    0.1)."""
    scene = _cavity(100., 100.)
    plan = quiet(fd.plan, scene)
    before = np.linalg.eigvalsh(_dense(*_region(scene, plan, field, False), field))[0]
    after = np.linalg.eigvalsh(_dense(*_region(scene, plan, field, True), field))[0]
    assert before < 0 < after


def test_an_eps_12_sphere_moves_no_star():
    """Where the operator is comfortably positive the star leaves it alone: an ``eps = 12``
    sphere lowers to build's bits. WBC's eight blocks per node, the lift's first form, moved
    1.3-3.3% of such a sphere's blocks and added an error that did not shrink with the cell
    (``studies/fdtd_subpixel_lift.py``)."""
    scene = _cavity(12., 1.)
    plan = quiet(fd.plan, scene)
    rt = runtime('numpy')
    params, built = fd.lower(scene, plan, rt), build(scene, plan, rt)
    for key in ('be', 'bh', 'minus_bh'):
        assert np.array_equal(params[key], built[key])


@pytest.mark.parametrize('radius', (.25, .1))
def test_the_c3_spheres_move_no_star(radius):
    """The C3 Mie evidence was measured before the lift, and stands: the epsilon-12 sphere of
    radius .25 wavelength and the small high-index control of radius .1, at 40 cells per wavelength
    on their own grid (``studies/fdtd_subpixel_mie_sphere.py``'s box, walled here: the lift reads
    only the structures and the grid), move no star. Checked 2026-09-27 over all 37 C3 scenes --
    the three spheres at 20-80 cells per wavelength and three placements, the control, and the TE
    cylinder at 40/56/80 -- with none moved; these two are the cheapest that bind."""
    from wavebit.fd._lower.constitutive import lifted_on_host
    dx = 1 / 40
    scene = fd.Scene(domain=fd.Domain(size=(2 * (.75 + .2 + 12 * dx),) * 3),
                     boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                     resolution=fd.Resolution(dx=dx), run=fd.Run(steps=4),
                     structures=(fd.Structure(fd.Sphere(radius=radius), 12.),))
    plan = quiet(fd.plan, scene)
    lifted = lifted_on_host(scene, plan)[1]
    assert lifted and all(out.moved == 0 and out.stars > 0 for out in lifted.values())


@pytest.mark.parametrize('field', ('E', 'H'))
def test_the_node_tensor_is_the_kottke_tensor_at_each_node(field):
    """What the lift moves a star toward, from the definitions: ``1/<eps> + (<1/eps> -
    1/<eps>) n_a^2`` on the diagonal and ``1/<eps>``, at E nodes and H cell centres, with the
    sphere's exact fill and radial normal."""
    from wavebit.fd._lower import constitutive
    from wavebit.fd._shapes import Sphere
    scene = _cavity(12., 5.)
    plan = quiet(fd.plan, scene)
    nodes = {}
    constitutive.lower(scene, plan, runtime('numpy'), nodes=nodes)
    term = next(t for t in plan.constitutive if t.field == field)
    tensor, inverse = nodes[term.name]
    z, y, x = np.meshgrid(*(np.arange(a, b) for a, b in term.box), indexing='ij')
    points = np.stack([x, y, z], axis=-1).astype(float)
    points = plan.low + plan.dx * (points + (0 if field == 'E' else .5))
    body = Sphere(.15, (.012, .007, -.009))
    fill = body.exact_fill(points, plan.dx)
    normal = points - np.array(body.center)
    normal /= np.linalg.norm(normal, axis=-1)[..., None]
    value = 12. if field == 'E' else 5.
    arithmetic = fill * value + 1 - fill
    contrast = fill / value + 1 - fill - 1 / arithmetic
    # A fill of -6e-14 is kept by design (coefficients._mix); the painted tensor keeps the
    # background's there, which differs from this formula by that much.
    np.testing.assert_allclose(inverse, 1 / arithmetic, rtol=0, atol=1e-12)
    np.testing.assert_allclose(tensor, np.stack([1 / arithmetic + contrast * normal[..., a] ** 2
                                                 for a in range(3)]), rtol=0, atol=1e-12)
    assert np.ptp(tensor) > .1                  # the fixture has a surface in it


def test_a_glass_sphere_lowers_to_the_same_bits():
    """Nothing to repair at glass contrast: ``be`` and ``bh`` are build's, to the bit."""
    scene = _cavity(2.25)
    plan = quiet(fd.plan, scene)
    rt = runtime('numpy')
    params, built = fd.lower(scene, plan, rt), build(scene, plan, rt)
    for key in ('be', 'bh', 'minus_bh'):
        assert np.array_equal(params[key], built[key])


def test_the_lift_is_continuous_where_a_star_starts_to_move():
    """``t*`` grows from zero where a star reaches the floor. A trigger at zero with a lift to
    the floor would jump there (the first prototype did)."""
    rng = np.random.default_rng(73)
    shape = (2, 2, 2)
    diagonal, cross, node, inverse = _tensors(rng, shape, contrast=3.)
    rt = runtime('numpy')
    moves = lambda scale: stars.lift('E', diagonal, scale * cross, node, inverse, rt)
    lo, hi = 0., 1.
    assert moves(lo).delta is None and moves(hi).delta is not None
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if moves(mid).delta is None else (lo, mid)
    assert np.max(np.abs(moves(hi).delta)) < 1e-9


def test_an_extreme_contrast_is_certified():
    """``eps = 1e10``: the stars lift almost to the node tensor and are certified. The unscaled
    leading minors, the test's first spelling, refused pieces here (the design review measured
    331 of the first form's blocks at 1e10)."""
    from wavebit.fd._lower import constitutive
    scene = _cavity(1e10, None)
    plan = quiet(fd.plan, scene)
    rt, nodes, lifted = runtime('numpy'), {}, {}
    params = dict(build(scene, plan, rt))
    params['constitutive'] = constitutive.lower(scene, plan, rt, nodes=nodes)
    assert constitutive.lift(plan, params, nodes, rt, lifted)
    assert all(out.moved > 0 for out in lifted.values())


def test_plan_and_check_say_where_the_lift_moved_stars():
    """A scene the Gershgorin certificate used to refuse now plans with a modified operator; the
    plan says so, with each term's count, and ``fd.check`` prints it under the body. A glass
    sphere, which moves nothing, hears neither."""
    from wavebit.fd._lower.constitutive import lifted_on_host
    scene = _cavity(40., 30.)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        plan = fd.plan(scene)
    said = [str(w.message) for w in caught if 'star lift moved' in str(w.message)]
    assert len(said) == 1
    report = str(quiet(fd.check, scene))
    for name, out in lifted_on_host(scene, plan)[1].items():
        assert out.moved > 0 and f'{name} {out.moved} of {out.stars}' in said[0]
        assert f'lift     {name}: {out.moved} of {out.stars} stars moved' in report
    glass = _cavity(2.25)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fd.plan(glass)
    assert not any('star lift' in str(w.message) for w in caught)
    assert 'lift     ' not in str(quiet(fd.check, glass))


def _npy(x):
    return to_numpy(x) if hasattr(x, 'shape') else np.asarray(x)


def test_every_backend_and_the_traced_path_lift_alike():
    """NumPy, JAX, ``jax.jit`` (which has no value to read, so it lifts every star in the
    runtime's own arrays) and Torch, on a sphere whose E and H stars both move."""
    import jax
    scene = _cavity(40., 30.)
    plan = quiet(fd.plan, scene)
    reference = fd.lower(scene, plan, runtime('numpy'))
    built = build(scene, plan, runtime('numpy'))
    full = (3,) + plan.shape[1:]
    for key in ('be', 'bh'):
        assert not np.array_equal(np.broadcast_to(reference[key], full),
                                  np.broadcast_to(built[key], full))
    # The H update reads minus_bh; it has to be the negation of the lifted bh.
    assert np.array_equal(reference['minus_bh'], -reference['bh'])
    jit_rt = runtime('jax')
    lowered = {'jax': fd.lower(scene, plan, jit_rt),
               'torch': fd.lower(scene, plan, runtime('torch')),
               'jit': jax.jit(lambda e, m: fd.lower(_cavity(e, m), plan, jit_rt))(
                   jit_rt.array(40.), jit_rt.array(30.))}
    for name, params in lowered.items():
        assert float(_npy(params['stable'])) == 1.
        for key in ('be', 'bh', 'minus_bh'):
            np.testing.assert_allclose(np.broadcast_to(_npy(params[key]), full),
                                       np.broadcast_to(reference[key], full), rtol=1e-12, atol=0,
                                       err_msg=f'{name} {key}')


def test_an_uncertified_lift_is_refused_and_poisons_stable(monkeypatch):
    """The certificate is read, not assumed. A star cannot stay indefinite at its lift for
    positive finite phases, so this forces one: the plan refuses by name, and a lowering made
    against an earlier plan reads ``stable = NaN``."""
    scene = _cavity(40., 30.)
    plan = quiet(fd.plan, scene)
    real = stars.lift
    monkeypatch.setattr(stars, 'lift', lambda *a, **k: real(*a, **k)._replace(certified=False))
    with pytest.raises(ValueError, match='not positive definite'):
        quiet(fd.plan, scene)
    assert np.isnan(float(fd.lower(scene, plan, runtime('numpy'))['stable']))


def _wall_cavity(eps, mu, center):
    """A small sphere against the PEC walls: its term's region starts (E) or ends (H) at the
    domain, so stars there have a ghost edge and cross coefficients."""
    return fd.Scene(domain=fd.Domain(size=(.8,) * 3), resolution=fd.Resolution(dx=.05),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    run=fd.Run(steps=8), structures=(fd.Structure(
                        fd.Sphere(radius=.12, center=center), fd.Material(eps=eps, mu=mu)),))


def _ghost_stars_moved(scene, plan, field):
    """How many stars with a ghost edge move, read with the module's own test."""
    from wavebit.fd._kernel.observe import InteriorEnergy
    from wavebit.fd._lower import constitutive
    rt = runtime('numpy')
    params = dict(build(scene, plan, rt))
    nodes = {}
    cross = constitutive.lower(scene, plan, rt, nodes=nodes)
    term = next(t for t in plan.constitutive if t.field == field)
    c = np.asarray(cross[term.name])
    shape = c.shape[1:]
    d = np.broadcast_to(InteriorEnergy._cut(np.asarray(params['be' if field == 'E' else 'bh']),
                                            slice(None), term.box) / term.ratio, c.shape)
    k, inv = (np.asarray(v) for v in nodes[term.name])
    g = [np.broadcast_to(stars._ghost(shape, a, field), shape) for a in range(3)]
    own, other = stars._star(np, field, d, k, g)
    floor = stars._floor(np, stars.THETA, inv, own, other, g)
    fire = ~stars._test(np, own, other, g, k, c, np.zeros(shape), floor)[0]
    return int((fire & (g[0] | g[1] | g[2])).sum())


@pytest.mark.parametrize('field, center', [('E', (-.27,) * 3), ('H', (.255,) * 3)])
def test_the_lift_differentiates_through_a_star_at_a_wall(field, center):
    """A star with a ghost edge reads its lone edge as ``2 x``, and its rate as ``2 x'``. The
    centred cavity moves no such star, so the review of the star found that branch untested:
    here 12 E stars (and 24 H stars) with a ghost edge move. Central differences against
    jax.grad at eps = mu = 1000, relative (eps, mu), measured 2026-09-27: E wall h 1 2.3e-8,
    7.1e-7; h 0.1 2.4e-10, 7.2e-9; h 0.01 5.6e-10, 1.7e-9. H wall h 1 5.4e-7, 4.8e-7; h 0.1
    6.1e-9, 4.7e-9; h 0.01 1.3e-8, 1.3e-10. So h = 0.1."""
    import jax
    scene = _wall_cavity(1000., 1000., center)
    plan = quiet(fd.plan, scene)
    assert _ghost_stars_moved(scene, plan, field) > 0
    full = (3,) + plan.shape[1:]
    weight = np.random.default_rng(75).normal(size=full)
    rt, host = runtime('jax'), runtime('numpy')

    def loss(eps, mu, rt):
        params = fd.lower(_wall_cavity(eps, mu, center), plan, rt)
        w = rt.array(weight)
        return sum(rt.xp.sum(w * rt.xp.broadcast_to(params[k], full)) for k in ('be', 'bh'))

    gradient = [float(v) for v in
                jax.grad(loss, argnums=(0, 1))(rt.array(1000.), rt.array(1000.), rt)]
    h = .1
    finite = [(loss(1000. + h, 1000., host) - loss(1000. - h, 1000., host)) / (2 * h),
              (loss(1000., 1000. + h, host) - loss(1000., 1000. - h, host)) / (2 * h)]
    np.testing.assert_allclose(gradient, finite, rtol=1e-7)


@pytest.mark.parametrize('theta', (0., 1., -.1))
def test_a_floor_outside_the_unit_interval_is_refused(theta):
    shape = (2, 2, 2)
    with pytest.raises(ValueError, match='must lie in'):
        stars.lift('E', np.ones((3,) + shape), np.zeros((3,) + shape), np.ones((3,) + shape),
                   np.ones(shape), runtime('numpy'), theta=theta)


def test_a_value_read_back_in_another_shape_is_not_a_value():
    """A batch wrapper can read back with its batch axis; the host path must not take it."""
    class Batched:
        def constant(self, x):
            return x

        def numpy(self, x):
            return np.stack([x, x])
    x = np.ones((3, 2, 2, 2))
    assert stars._read(Batched(), (x,)) is None
    assert stars._read(runtime('numpy'), (x,))[0].shape == x.shape


@pytest.mark.parametrize('eps, said', [(1e16, 'contrast of 1e+16'),
                                       (1e200, 'could not solve its eigenproblem')])
def test_past_float64s_reach_the_plan_says_so(eps, said):
    """Real inputs, not a monkeypatch: past about 1e12 a fill's round-off times the contrast
    breaks the mix, the lift cannot repair it, and the plan says that -- not that a value is
    not a finite positive number, nor (at 1e200, where the eigenproblem fails) that the scene
    is not concrete."""
    with pytest.raises(ValueError) as caught:
        quiet(fd.plan, _cavity(eps, None))
    assert said in str(caught.value) and 'concrete' not in str(caught.value)


def test_a_shared_term_says_so_on_its_lift_line():
    """Two bodies in one merged term print its lift under both; each says it is shared, as the
    cross line does, so that nobody sums the two."""
    scene = fd.Scene(domain=fd.Domain(size=(1.0,) * 3), resolution=fd.Resolution(dx=.05),
                     boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                     run=fd.Run(steps=2), structures=tuple(
                         fd.Structure(fd.Sphere(radius=.15, center=(s * .2, .01, .02)), 40.)
                         for s in (-1, 1)))
    lines = [l for l in str(quiet(fd.check, scene)).splitlines() if l.startswith('  lift')]
    assert len(lines) == 2 and all('shared with structures[' in l for l in lines)


def test_a_run_lowers_its_scene_on_the_host_once(monkeypatch):
    """fd.run plans (which lifts on the host, to certify) and reports; the report reuses the
    plan's lift for the same scene. It lowered it again, twice per run, before the review of
    the star."""
    from wavebit.fd._lower import constitutive
    calls = []
    real = constitutive.lifted_on_host
    monkeypatch.setattr(constitutive, 'lifted_on_host',
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    quiet(fd.run, _cavity(40., 30.))
    assert len(calls) == 1


def test_a_floor_is_never_negative():
    """At eps 1e13 a cell-centred sphere 12 cells across reads a fill of -7e-14 and an edge
    diagonal of -15 (the review of the star); the floor went negative with it, the test became
    weaker than positivity, and the plan refused a scene the lift can repair. Clamped at zero,
    it plans."""
    dx = .05
    scene = fd.Scene(domain=fd.Domain(size=(18 * dx,) * 3), resolution=fd.Resolution(dx=dx),
                     boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                     run=fd.Run(steps=2), structures=(fd.Structure(
                         fd.Sphere(radius=6 * dx, center=(dx / 2,) * 3), 1e13),))
    quiet(fd.plan, scene)


def test_the_plans_lift_is_reused_for_its_own_scene_only():
    """A plan can lower another scene of the same structure; the notes then read that scene's
    lift, not the one the plan kept."""
    from wavebit.fd._lower.checks import notes
    from wavebit.fd._lower.constitutive import lifted_on_host
    plan = quiet(fd.plan, _cavity(40., 30.))
    other = _cavity(100., 30.)
    lifted = {}
    quiet(notes, other, plan, lifted=lifted)
    expected = lifted_on_host(other, plan)[1]
    assert {k: v.moved for k, v in lifted.items()} == {k: v.moved for k, v in expected.items()}
    assert lifted['constitutive[E0]'].moved != plan._lifted[1]['constitutive[E0]'].moved


def test_torch_lowers_a_lifted_scene_without_a_warning():
    """The ghost masks went to Torch as read-only broadcast views, and every Torch lowering that
    moved a star printed "The given NumPy array is not writable" (the review of the star)."""
    scene = _cavity(40., 30.)
    plan = quiet(fd.plan, scene)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fd.lower(scene, plan, runtime('torch'))
    assert not [w for w in caught if 'not writable' in str(w.message)]


#: ``(center, radius, eps, mu)`` of the Hessian scenes: a sphere off a node, on a node, on a cell,
#: and against the low and high PEC walls -- where 3 E and 3 H stars have a double root, whose
#: root function is ``d2`` (stars.TAU).
HESSIAN_SCENES = {'off-node': ((.012, .007, -.009), .15, 40., 30.),
                  'node': ((0., 0., 0.), .15, 40., 30.),
                  'cell': ((.025,) * 3, .15, 40., 30.),
                  'E wall': ((-.27,) * 3, .12, 1000., 1000.),
                  'H wall': ((.255,) * 3, .12, 1000., 1000.)}


def _hessian_loss(name):
    center, radius, eps, mu = HESSIAN_SCENES[name]
    scene = lambda e, m: fd.Scene(
        domain=fd.Domain(size=(.8,) * 3), resolution=fd.Resolution(dx=.05),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()), run=fd.Run(steps=8),
        structures=(fd.Structure(fd.Sphere(radius=radius, center=center),
                                 fd.Material(eps=e, mu=m)),))
    plan = quiet(fd.plan, scene(eps, mu))
    full = (3,) + plan.shape[1:]
    weight = np.random.default_rng(74).normal(size=full)

    def loss(p, rt):
        params = fd.lower(scene(p[0], p[1]), plan, rt)
        w = rt.array(weight)
        return sum(rt.xp.sum(w * rt.xp.broadcast_to(params[k], full)) for k in ('be', 'bh'))
    return loss, np.array([eps, mu])


@pytest.mark.parametrize('backend, name', [('jax', 'off-node'), ('jit', 'off-node'),
                                           ('torch', 'off-node'), ('jax', 'node'),
                                           ('jax', 'cell'), ('jax', 'E wall'), ('jax', 'H wall')])
def test_the_lift_has_exact_second_derivatives(backend, name):
    """The root's series reversion at ``t*``: the Hessian of a lowered sum is the derivative of its
    gradient, on whole scenes where the oracle's fixtures are small. Against central differences
    of jax.grad (whose values the finite-difference tests above pin) at a step of ``1e-4`` of each
    input, measured 2026-09-27: 3.8e-8 off a node, 4.6e-8 on a node, 2.6e-8 on a cell, 1.1e-8 at
    either wall, on JAX, ``jax.jit`` and Torch alike -- and 100 times more at ``1e-3``, the step
    squared, so the residual is the differences' own. The eigenvector correction it replaced read
    188% off."""
    import jax
    loss, p0 = _hessian_loss(name)
    rj = runtime('jax')
    grad = jax.grad(lambda p: loss(p, rj))
    h = 1e-4 * p0
    expected = np.stack([(np.asarray(grad(rj.array(p0 + h[i] * np.eye(2)[i])))
                          - np.asarray(grad(rj.array(p0 - h[i] * np.eye(2)[i])))) / (2 * h[i])
                         for i in range(2)])
    if backend == 'torch':
        import torch
        rt = runtime('torch')
        hessian = torch.autograd.functional.hessian(
            lambda p: loss(p, rt), torch.tensor(p0, dtype=torch.float64)).numpy()
    else:
        function = jax.hessian(lambda p: loss(p, rj))
        function = jax.jit(function) if backend == 'jit' else function
        hessian = np.asarray(function(rj.array(p0)))
    assert np.abs(hessian - expected).max() < 1e-6 * np.abs(expected).max()


#: The 50-digit oracle, frozen by ``references/gen_star_lift_oracle.py``: four small regions, each
#: with two directions, and the lift's ``delta`` with its first and second derivatives along them.
ORACLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'references',
                      'star_lift_oracle.npz')


def _nested(backend, field, inputs, u, w):
    """``delta``, its derivatives along ``u`` and ``w``, and along ``u`` then ``w``: forward mode
    inside forward mode, as a Hessian-vector product takes it."""
    if backend == 'torch':
        import torch
        rt, jvp = runtime('torch'), torch.func.jvp
        scalar = lambda v: torch.tensor(v, dtype=torch.float64)
    else:
        import jax
        rt, jvp = runtime('jax'), jax.jvp
        scalar = rt.array

    def run(base, du, dw):
        def delta(a, b):
            point = [x + a * p + b * q for x, p, q in zip(base, du, dw)]
            return stars.lift(field, *point, rt).delta
        inner = lambda a: jvp(lambda b: delta(a, b), (scalar(0.),), (scalar(1.),))
        return jvp(inner, (scalar(0.),), (scalar(1.),))

    if backend == 'jit':
        run = jax.jit(run)
    (value, along_w), (along_u, second) = run(*([rt.array(x) for x in arrays]
                                                 for arrays in (inputs, u, w)))
    return [to_numpy(x) for x in (value, along_u, along_w, second)]


@pytest.mark.parametrize('backend', ('jax', 'jit', 'torch'))
@pytest.mark.parametrize('fixture', ('random', 'ghost', 'symmetric', 'decoupled'))
def test_the_lift_agrees_with_its_oracle_to_second_order(backend, fixture):
    """Against 50-digit central differences of the definition (``references/
    gen_star_lift_oracle.py``, which shares no arithmetic with the module), entry by entry: the
    value, both first derivatives, and the three second ones. ``random`` and ``ghost`` move stars
    with and without ghost edges; ``symmetric`` has a double root, along directions that keep it;
    ``decoupled`` a star nearly uncoupled along z, where a fixed ``x, y, z`` order loses its
    middle pivot, and whose largest eigenvector points along z, so neither the fixed order nor one
    read off the wrong eigenvector will do. Measured 2026-09-27: 6.7e-16 at worst, every fixture
    and backend."""
    data = np.load(ORACLE)
    field = str(data[f'{fixture}_field'])
    inputs = [data[f'{fixture}_input{i}'] for i in range(4)]
    u, w = ([data[f'{fixture}_dir{tag}{i}'] for i in range(4)] for tag in 'uw')
    for (a, b), (da, db) in {'uu': (u, u), 'ww': (w, w), 'uw': (u, w)}.items():
        value, along_a, along_b, second = _nested(backend, field, inputs, da, db)
        for got, key in ((value, 'delta'), (along_a, f'first_{a}'), (along_b, f'first_{b}'),
                         (second, f'second_{a}{b}')):
            expected = data[f'{fixture}_{key}']
            scale = max(np.abs(expected).max(), 1.)
            assert np.abs(got - expected).max() < 1e-12 * scale, (key, np.abs(got - expected).max())


#: Off a node, on a node (the 16-cell box's origin is node 8) and on a cell centre. The two
#: symmetric placements put stars with a double smallest eigenvalue on the body diagonals,
#: where a determinant's derivative is noise (the design review measured 44% on the first,
#: eight-block form).
PLACEMENTS = {'off-node': (.012, .007, -.009), 'node': (0., 0., 0.), 'cell': (.025, .025, .025)}


@pytest.mark.parametrize('backend, placement', [('jax', 'off-node'), ('jit', 'off-node'),
                                                ('torch', 'off-node'), ('jax', 'node'),
                                                ('jax', 'cell')])
def test_the_lift_differentiates_through_t(backend, placement):
    """The implicit gradient of ``t*``: without it a gradient holds every ``t`` fixed."""
    center = PLACEMENTS[placement]
    scene = _cavity(40., 30., center)
    plan = quiet(fd.plan, scene)
    rt = runtime('jax' if backend == 'jit' else backend)
    full = (3,) + plan.shape[1:]
    weight = np.random.default_rng(74).normal(size=full)

    def loss(eps, mu, rt=rt):
        params = fd.lower(_cavity(eps, mu, center), plan, rt)
        w = rt.array(weight)
        return sum(rt.xp.sum(w * rt.xp.broadcast_to(params[k], full)) for k in ('be', 'bh'))

    if backend == 'torch':
        import torch
        values = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (40., 30.)]
        loss(*values).backward()
        gradient = [float(v.grad) for v in values]
    else:
        import jax
        grad = jax.grad(loss, argnums=(0, 1))
        grad = jax.jit(grad) if backend == 'jit' else grad
        gradient = [float(v) for v in grad(rt.array(40.), rt.array(30.))]
    # Central differences against jax.grad, relative, (eps, mu), measured 2026-09-27 (THETA 0.02):
    #   h 1e-2: off-node 1.8e-8, 2.8e-8; node 4.0e-8, 9.2e-7; cell 1.8e-7, 6.9e-8
    #   h 1e-3: off-node 2.3e-10, 2.6e-10; node 1.0e-9, 4.1e-9; cell 2.3e-9, 5.2e-10
    #   h 1e-4: off-node 4.8e-10, 7.4e-10; node 1.0e-9, 6.3e-8; cell 2.2e-8, 9.0e-13
    # so 1e-3, below the curvature and above the loss's round-off. (On the first, eight-block
    # form a determinant test scattered the cell's differences by 6% at 1e-6: its t* was
    # resolved to 1e-8 at a double root.)
    h, host = 1e-3, runtime('numpy')
    finite = [(loss(40. + h, 30., host) - loss(40. - h, 30., host)) / (2 * h),
              (loss(40., 30. + h, host) - loss(40., 30. - h, host)) / (2 * h)]
    np.testing.assert_allclose(gradient, finite, rtol=1e-7)
