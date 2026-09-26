"""The per-block lift (``wavebit/fd/_lower/blocks.py``): the smoothed operator is positive by
construction, and the lift changes nothing it does not have to.

The references here are written from the definitions and share no code with the module: the
block sum is enumerated edge by edge against ``cross_action``'s own dense matrix, and the least
move is found with ``numpy.linalg.eigvalsh`` on the principal sub-block, not with the module's
leading minors.
"""
import itertools
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._kernel.constitutive import cross_action
from wavebit.fd._lower import blocks
from wavebit.fd._lower.coefficients import build
from wavebit.fd._runtime import runtime
from fd_helpers import to_numpy


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


def _edges(field, p, s, n):
    """The ``(z, y, x)`` index of each axis's edge in block ``(p, s)``, or ``None`` outside."""
    out = []
    for a in range(3):
        q = list(p)
        q[2 - a] += s[a] * (-1 if field == 'E' else 1)
        out.append(tuple(q) if -1 <= q[2 - a] <= n[2 - a] else None)
    return out


def _off(cross, p):
    xy, xz, yz = cross[(slice(None),) + p]
    return np.array([[0., xy, xz], [xy, 0., yz], [xz, yz, 0.]])


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


def _tensors(rng, shape, contrast=1.):
    """Node Kottke tensors (``inv I + contrast n n^T``) and edge diagonals that make blocks fail."""
    normal = rng.normal(size=(3,) + shape)
    normal /= np.linalg.norm(normal, axis=0)
    inverse = rng.uniform(.05, 1., size=shape)
    strength = rng.uniform(0., contrast, size=shape)
    node = inverse + strength * normal ** 2
    cross = np.stack([strength * normal[i] * normal[j] for i, j in blocks.PAIRS])
    diagonal = rng.uniform(.02, 1., size=(3,) + shape)
    return diagonal, cross, node, inverse


@pytest.mark.parametrize('field', ('E', 'H'))
def test_the_blocks_average_to_the_operator(field):
    """The premise of the proof: ``(1/8) sum P^T B(t) P = diag(e + delta) + C`` for any ``t``."""
    rng = np.random.default_rng(71)
    shape = (3, 4, 5)
    diagonal, cross, node, inverse = _tensors(rng, shape)
    rt = runtime('numpy')
    moves = rng.uniform(size=(8,) + shape)
    t = dict(zip(blocks.ORIENTATIONS, moves))
    unit = [np.ones((8,) + shape), np.zeros((8,) + shape), np.zeros((8,) + shape)]
    delta = blocks._assemble(field, diagonal, cross, node, inverse, rt, blocks.THETA, shape,
                             np.ones((8,) + shape, bool), moves, unit)
    size = int(np.prod(diagonal.shape))
    expected = np.zeros((size, size))
    # Every node whose blocks reach the region's edges: one layer past its high end for E
    # (the far node of the last edge), before its low end for H. Outside, no cross and t = 0.
    reach = [range(0, n + 1) if field == 'E' else range(-1, n) for n in shape]
    for p in itertools.product(*reach):
        inside = all(0 <= v < n for v, n in zip(p, shape))
        for s in blocks.ORIENTATIONS:
            edges = _edges(field, p, s, shape)
            axes = [a for a in range(3) if edges[a] is not None
                    and all(0 <= v < n for v, n in zip(edges[a], shape))]
            rows = [np.ravel_multi_index((a,) + edges[a], diagonal.shape) for a in axes]
            off = _off(cross, p) if inside else np.zeros((3, 3))
            block = off[np.ix_(axes, axes)] + np.diag(
                [diagonal[(a,) + edges[a]] + (t[s][p] * (node[(a,) + p] - diagonal[(a,) + edges[a]])
                                              if inside else 0.) for a in axes])
            expected[np.ix_(rows, rows)] += block / 8
    np.testing.assert_allclose(_dense(diagonal + delta, cross, field), expected, rtol=0, atol=1e-14)


def _least(block, floor):
    """The least ``t`` in ``[0, 1]`` with ``eigvalsh(block(t))[0] > floor``, by bisection."""
    if np.linalg.eigvalsh(block(0.))[0] > floor:
        return 0.
    lo, hi = 0., 1.
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if np.linalg.eigvalsh(block(mid))[0] > floor else (mid, hi)
    return hi


@pytest.mark.parametrize('field', ('E', 'H'))
def test_the_lift_is_the_least_move_above_the_floor(field):
    """Each block moves the least ``t`` that puts its principal sub-block above
    ``THETA * min(1/<eps>, its edges' diagonals)``, and each edge gets the eighth of it."""
    rng = np.random.default_rng(72)
    shape = (3, 4, 5)
    diagonal, cross, node, inverse = _tensors(rng, shape)
    expected, moved = np.zeros_like(diagonal), 0
    for p in np.ndindex(*shape):
        for s in blocks.ORIENTATIONS:
            edges = _edges(field, p, s, shape)
            axes = [a for a in range(3) if edges[a] is not None
                    and all(0 <= v < n for v, n in zip(edges[a], shape))]
            if not axes:
                continue
            e = np.array([diagonal[(a,) + edges[a]] for a in axes])
            k = np.array([node[(a,) + p] for a in axes])
            off = _off(cross, p)[np.ix_(axes, axes)]
            floor = blocks.THETA * min([inverse[p]] + list(e))
            t = _least(lambda t: off + np.diag(e + t * (k - e)), floor)
            moved += t > 0
            for i, a in enumerate(axes):
                expected[(a,) + edges[a]] += t * (k[i] - e[i]) / 8
    out = blocks.lift(field, diagonal, cross, node, inverse, runtime('numpy'))
    # The fixture moves some blocks and not others, or the comparison proves little.
    assert 0 < moved < 8 * np.prod(shape) and out.moved == moved and out.certified
    np.testing.assert_allclose(out.delta, expected, rtol=0, atol=1e-13)


def test_a_block_without_cross_coefficients_never_moves():
    """No cross coefficient, no move: even an edge diagonal far below ``THETA / <eps>``
    (an ``eps = 20`` edge beside a vacuum node) is the block's own smallest eigenvalue."""
    shape = (2, 3, 4)
    diagonal = np.full((3,) + shape, 1 / 20)
    out = blocks.lift('E', diagonal, np.zeros((3,) + shape), np.ones((3,) + shape),
                      np.ones(shape), runtime('numpy'))
    assert out.delta is None and out.moved == 0 and out.certified


def test_an_extreme_contrast_is_certified():
    """``eps = 1e10``: every block lifts to the node tensor almost whole and is certified. The
    unscaled leading minors, the test's first spelling, refused blocks here (the design review
    measured 331 at 1e10)."""
    from wavebit.fd._lower import constitutive
    scene = _cavity(1e10, None)
    plan = quiet(fd.plan, scene)
    rt, nodes, lifted = runtime('numpy'), {}, {}
    params = dict(build(scene, plan, rt))
    params['constitutive'] = constitutive.lower(scene, plan, rt, nodes=nodes)
    assert constitutive.lift(plan, params, nodes, rt, lifted)
    assert all(out.moved > 0 for out in lifted.values())


def _cavity(eps, mu=1.5, center=(.012, .007, -.009)):
    return fd.Scene(domain=fd.Domain(size=(.8,) * 3), resolution=fd.Resolution(dx=.05),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    run=fd.Run(steps=8), structures=(fd.Structure(
                        fd.Sphere(radius=.15, center=center), fd.Material(eps=eps, mu=mu)),))


@pytest.mark.parametrize('field', ('E', 'H'))
def test_the_node_tensor_is_the_kottke_tensor_at_each_node(field):
    """What the lift moves a block toward, from the definitions: ``1/<eps> + (<1/eps> -
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


def test_the_lift_is_continuous_where_a_block_starts_to_move():
    """``t*`` grows from zero where a block reaches the floor. A trigger at zero with a lift to
    the floor would jump there (the first prototype did)."""
    rng = np.random.default_rng(73)
    shape = (2, 2, 2)
    diagonal, cross, node, inverse = _tensors(rng, shape, contrast=2.)
    rt = runtime('numpy')
    moves = lambda scale: blocks.lift('E', diagonal, scale * cross, node, inverse, rt)
    lo, hi = 0., 1.
    assert moves(lo).delta is None and moves(hi).delta is not None
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if moves(mid).delta is None else (lo, mid)
    assert np.max(np.abs(moves(hi).delta)) < 1e-9


def _npy(x):
    return to_numpy(x) if hasattr(x, 'shape') else np.asarray(x)


def test_every_backend_and_the_traced_path_lift_alike():
    """NumPy, JAX, ``jax.jit`` (which has no value to read, so it lifts every block in the
    runtime's own arrays) and Torch, on a sphere whose E and H blocks both move."""
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
    """The certificate is read, not assumed. A block cannot stay indefinite at its lift for
    positive finite phases, so this forces one: the plan refuses by name, and a lowering made
    against an earlier plan reads ``stable = NaN``."""
    scene = _cavity(40., 30.)
    plan = quiet(fd.plan, scene)
    real = blocks.lift
    monkeypatch.setattr(blocks, 'lift', lambda *a, **k: real(*a, **k)._replace(certified=False))
    with pytest.raises(ValueError, match='not positive definite'):
        quiet(fd.plan, scene)
    assert np.isnan(float(fd.lower(scene, plan, runtime('numpy'))['stable']))


#: Off a node, on a node (the 16-cell box's origin is node 8) and on a cell centre. The two
#: symmetric placements put blocks with a double smallest eigenvalue on the body diagonals,
#: where the determinant's derivative is noise (the design review measured 44% on them).
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
    # Central differences against jax.grad, relative, (eps, mu), measured 2026-09-27:
    #   h 1e-2: off-node 2e-9, 2.8e-4; node 1.9e-8, 5.4e-7; cell 4.0e-7, 6.3e-8
    #   h 1e-3: off-node 1.7e-10, 5.4e-10; node 4.1e-10, 4.7e-9; cell 1.5e-8, 7.7e-10
    #   h 1e-4: off-node 7.9e-10, 1.4e-10; node 8.3e-9, 2.9e-8; cell 2.1e-7, 9.7e-10
    # so 1e-3, below the curvature and above the loss's round-off, cancelled on the cell's
    # small eps gradient. (A determinant test for the pivots scattered the cell's differences
    # by 6% at 1e-6 and 8e-5 at 1e-4: its t* was resolved to 1e-8 at a double root.)
    h, host = 1e-3, runtime('numpy')
    finite = [(loss(40. + h, 30., host) - loss(40. - h, 30., host)) / (2 * h),
              (loss(40., 30. + h, host) - loss(40., 30. - h, host)) / (2 * h)]
    np.testing.assert_allclose(gradient, finite, rtol=1e-7)
