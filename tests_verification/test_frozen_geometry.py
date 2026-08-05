"""Frozen geometry makes the public API traceable (fieldwave.StaticGeometry).

Two pipeline steps decide something structural rather than numeric — the NVF normal field and
the CFS block compression. Both branch on values, so neither survives a JAX trace. (There was
a third, the same-material short-circuit; removing it is what left DFS with nothing to
freeze.) ``freeze_geometry()`` takes those decisions once against the concrete
structure; handing the result back to ``conv_solve`` makes the whole call jit- and
grad-traceable without changing a single number.
"""
import importlib

import numpy as np
import pytest

import fieldwave
from fieldwave import FourierType

UCELL = np.random.default_rng(0).uniform(1.0, 4.0, (1, 5, 5))
KW = dict(n_top=1.0, n_bot=1.0, theta=0.1, phi=0.2, psi=0.3, fto=[2, 2],
          period=[1.0, 1.0], wavelength=0.9, thickness=[0.2])
STRATEGIES = [FourierType.DFS, FourierType.CONTINUOUS, FourierType.NVF]
IDS = ['dfs', 'cfs', 'nvf']


@pytest.mark.parametrize('fourier_type', STRATEGIES, ids=IDS)
def test_frozen_geometry_does_not_change_the_answer(fourier_type):
    """Freezing is an optimization, not a different computation."""
    solver = fieldwave.rcwa(backend=0, ucell=UCELL, fourier_type=fourier_type, **KW)
    auto = solver.conv_solve()
    frozen = solver.conv_solve(geometry=solver.freeze_geometry())
    assert np.array_equal(np.asarray(auto.res.de_ri), np.asarray(frozen.res.de_ri))
    assert np.array_equal(np.asarray(auto.res.de_ti), np.asarray(frozen.res.de_ti))


@pytest.mark.parametrize('fourier_type', STRATEGIES, ids=IDS)
def test_jit_and_grad_through_the_public_api(fourier_type):
    """jax.jit and jax.grad over rcwa/conv_solve, for every Fourier strategy."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    solver = fieldwave.rcwa(backend=1, ucell=UCELL, fourier_type=fourier_type, **KW)
    geometry = solver.freeze_geometry()

    def loss(u):
        return jnp.sum(solver.conv_solve(ucell=u, geometry=geometry).res.de_ti).real

    u = jnp.asarray(UCELL)
    eager = float(loss(u))
    assert np.isclose(float(jax.jit(loss)(u)), eager, atol=1e-12, rtol=0)

    g = np.asarray(jax.grad(loss)(u))
    assert g.shape == UCELL.shape and np.isfinite(g).all()

    # finite-difference spot check along the all-ones direction
    d = np.ones_like(UCELL)
    h = 1e-6
    fd = (float(loss(jnp.asarray(UCELL + h * d))) - float(loss(jnp.asarray(UCELL - h * d)))) / (2 * h)
    assert abs(float(np.sum(g * d)) - fd) < 1e-5, (float(np.sum(g * d)), fd)


def _jit_loss(fourier_type):
    import jax.numpy as jnp
    solver = fieldwave.rcwa(backend=1, ucell=UCELL, fourier_type=fourier_type, **KW)
    return lambda u: jnp.sum(solver.conv_solve(ucell=u).res.de_ti).real


@pytest.mark.parametrize('fourier_type', [FourierType.CONTINUOUS, FourierType.NVF],
                         ids=['cfs', 'nvf'])
def test_auto_detection_still_blocks_jit(fourier_type):
    """Pins *why* the API exists: for these strategies, tracing fails without it.

    Auto-detection is the eager path; the normal field and the CFS block compression both
    branch on concrete values, and the compression's output shape depends on them.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    with pytest.raises(Exception):
        jax.jit(_jit_loss(fourier_type))(jnp.asarray(UCELL))


def test_dfs_needs_no_frozen_geometry_at_all():
    """DFS decides nothing, so it jits straight off the public API.

    It used to raise here. The one discrete decision on this path was the same-material
    short-circuit, and removing that left nothing between ``conv_solve`` and a trace --
    ``freeze_geometry()`` is now optional for DFS rather than the price of admission.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    got = float(jax.jit(_jit_loss(FourierType.DFS))(jnp.asarray(UCELL)))
    want = float(np.sum(np.asarray(
        fieldwave.rcwa(backend=0, ucell=UCELL, fourier_type=FourierType.DFS,
                       **KW).conv_solve().res.de_ti).real))
    assert got == pytest.approx(want, abs=1e-10)


def test_freeze_geometry_reports_what_the_strategy_needs():
    """Only the fields the active strategy consumes are populated.

    DFS populates nothing: it has no discrete decision left to take. (It had one -- the
    same-material short-circuit -- until that turned out to be a workaround for a singularity
    the convention merge had already removed.)
    """
    def geo(fourier_type):
        return fieldwave.rcwa(backend=0, ucell=UCELL, fourier_type=fourier_type,
                               **KW).freeze_geometry()

    dfs = geo(FourierType.DFS)
    assert dfs.normal_field is None and dfs.compression is None

    assert geo(FourierType.NVF).normal_field is not None
    assert geo(FourierType.CONTINUOUS).compression is not None


@pytest.mark.parametrize('name,make,start', [
    ('ellipse lx', lambda v: [[3.5, [['ellipse', 140.0, 160.0, v, 55.0, 2.0, 0.27, 9, 9]]]], 85.0),
    ('rectangle lx', lambda v: [[3.5, [['rectangle', 140.0, 160.0, v, 55.0, 2.0, 0.27, 4, 4]]]],
     85.0),
    ('rectangle angle',
     lambda v: [[3.5, [['rectangle', 140.0, 160.0, 85.0, 55.0, 2.0, v, 4, 4]]]], 0.27),
    ('overlapping pair',
     lambda v: [[1.0, [['rectangle', 130.0, 150.0, v, 120.0, 2.0, 0.0, 0, 0],
                       ['rectangle', 190.0, 150.0, 120.0, 120.0, 3.0, 0.0, 0, 0]]]], 120.0),
])
def test_a_frozen_geometry_makes_the_vector_path_traceable(name, make, start):
    """``jax.jit`` over a shape parameter, which the vector path could not do before.

    What stopped it was never arithmetic: it was the comparisons that decide which edge becomes
    which grid line, which two share one, and which strip bounds each of a rectangle's slabs.
    Those are discrete, so ``freeze_geometry`` records them and the traced call replays them —
    the same bargain the normal field and the same-material flags already make, and with the
    same caveat, that a parameter moved far enough to reorder the edges leaves the record
    describing a layer that is no longer there.

    The overlapping pair is here because it is the case that has no alternative: the analytic
    path superposes shapes rather than painting them (see test_analytic.py), so a layer whose
    objects overlap can only be built this way.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    importlib.import_module('fieldwave.on_jax')

    cfg = dict(period=[300.0, 300.0], thickness=[205.0], fto=[3, 3], n_top=1.0, n_bot=1.0,
               theta=0.15, phi=0.35, pol=0, wavelength=900.0)
    x = jnp.asarray(start)
    frozen = fieldwave.rcwa(backend=1, ucell=make(x), **cfg).freeze_geometry()
    assert frozen.vector_grid is not None, 'the grid plan was not recorded'

    def total(v):
        return jnp.sum(fieldwave.rcwa(backend=1, ucell=make(v), **cfg)
                       .conv_solve(geometry=frozen).de_ti).real

    eager = float(total(x))
    assert float(jax.jit(total)(x)) == pytest.approx(eager, rel=1e-11)

    grad = float(jax.grad(total)(x))
    assert grad != 0.0
    assert float(jax.jit(jax.grad(total))(x)) == pytest.approx(grad, rel=1e-11)

    step = 1e-3
    fd = (float(total(jnp.asarray(start + step))) - float(total(jnp.asarray(start - step)))) \
        / (2 * step)
    assert grad == pytest.approx(fd, rel=2e-3), (grad, fd)


def test_a_stale_frozen_geometry_is_refused_rather_than_answered():
    """Reusing a frozen plan too far from where it was taken has to fail loudly.

    The plan records which block edge becomes which grid line. Move the shape parameters far
    enough and the edges reorder, so the plan describes a different layer — and the answer that
    comes back is wrong without looking wrong, which is the whole hazard. Measured before the
    check existed: an ellipse frozen at ``lx`` 85 and replayed at 120 was 1.5e-4 off.

    The tell is free, because grid lines are sorted by construction: a replayed list that is
    not sorted is a plan that no longer fits. This pins both halves — that it is refused where
    it does not fit, and that where it *is* accepted the answer matches a rebuild exactly, so
    the check is not merely conservative.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    importlib.import_module('fieldwave.on_jax')
    from fieldwave._core.modeler import StaleGeometryError

    cfg = dict(period=[300.0, 300.0], thickness=[205.0], fto=[3, 3], n_top=1.0, n_bot=1.0,
               theta=0.15, phi=0.35, pol=0, wavelength=900.0)
    make = lambda lx: [[3.5, [['ellipse', 150.0, 150.0, lx, 55.0, 2.0, 0.27, 9, 9]]]]

    def total(lx, geometry=None):
        return float(jnp.sum(fieldwave.rcwa(backend=1, ucell=make(lx), **cfg)
                             .conv_solve(geometry=geometry).de_ti).real)

    frozen = fieldwave.rcwa(backend=1, ucell=make(jnp.asarray(85.0)), **cfg).freeze_geometry()

    for lx in (85.0, 100.0):                       # near the freeze: accepted, and exact
        assert total(jnp.asarray(lx), frozen) == total(lx)

    for lx in (120.0, 200.0):                      # far from it: refused, not answered
        with pytest.raises(StaleGeometryError, match='no longer fits'):
            total(jnp.asarray(lx), frozen)
