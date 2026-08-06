"""Frozen geometry makes the public API traceable (fieldwave.StaticGeometry).

Two pipeline steps decide something structural rather than numeric — the NVF normal field and
the CFS block compression. Both branch on values, so neither survives ``jax.jit``. (There was
a third, the same-material short-circuit; removing it is what left DFS with nothing to
freeze.) ``freeze_geometry()`` takes those decisions once against the concrete
structure; handing the result back to ``conv_solve`` makes the whole call jit- and
grad-traceable without changing a single number.

``jax.grad`` on its own is the weaker bar and the two steps part company there: the
compression decides itself inside a gradient, the normal field cannot. That boundary is worth
pinning, because on the near side a frozen record is optional — and an optional record is one
that cannot go stale.
"""
import importlib

import numpy as np
import pytest

import fieldwave

UCELL = np.random.default_rng(0).uniform(1.0, 4.0, (1, 5, 5))
KW = dict(n_top=1.0, n_bot=1.0, theta=0.1, phi=0.2, psi=0.3, fto=[2, 2],
          period=[1.0, 1.0], wavelength=0.9, thickness=[0.2])

# Every point of the two-axis cross product, including ('integrated', 'nvf') -- the one the
# deprecated `fourier_type` integer had no code for. It needs both frozen fields at once,
# which nothing else exercises.
STRATEGIES = [dict(fourier_coefficients=c, factorization=f)
              for c in ('sampled', 'integrated') for f in ('li', 'nvf')]
IDS = ['sampled-li', 'sampled-nvf', 'integrated-li', 'integrated-nvf']
DEFAULT_STRATEGY = dict(fourier_coefficients='sampled', factorization='li')


@pytest.mark.parametrize('strategy', STRATEGIES, ids=IDS)
def test_frozen_geometry_does_not_change_the_answer(strategy):
    """Freezing is an optimization, not a different computation."""
    solver = fieldwave.rcwa(backend=0, ucell=UCELL, **strategy, **KW)
    auto = solver.conv_solve()
    frozen = solver.conv_solve(geometry=solver.freeze_geometry())
    assert np.array_equal(np.asarray(auto.res.de_ri), np.asarray(frozen.res.de_ri))
    assert np.array_equal(np.asarray(auto.res.de_ti), np.asarray(frozen.res.de_ti))


@pytest.mark.parametrize('strategy', STRATEGIES, ids=IDS)
def test_jit_and_grad_through_the_public_api(strategy):
    """jax.jit and jax.grad over rcwa/conv_solve, for every Fourier strategy."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    solver = fieldwave.rcwa(backend=1, ucell=UCELL, **strategy, **KW)
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


def _jit_loss(**strategy):
    import jax.numpy as jnp
    solver = fieldwave.rcwa(backend=1, ucell=UCELL, **strategy, **KW)
    return lambda u: jnp.sum(solver.conv_solve(ucell=u).res.de_ti).real


# The default pairing is left out on purpose: it decides nothing, so it traces unfrozen.
NEEDS_FREEZE = [s for s in STRATEGIES if s != DEFAULT_STRATEGY]
NEEDS_FREEZE_IDS = [i for i, s in zip(IDS, STRATEGIES) if s != DEFAULT_STRATEGY]


@pytest.mark.parametrize('strategy', NEEDS_FREEZE, ids=NEEDS_FREEZE_IDS)
def test_auto_detection_still_blocks_jit(strategy):
    """Pins *why* the API exists: for these strategies, tracing fails without it.

    Auto-detection is the eager path; the normal field and the CFS block compression both
    branch on concrete values, and the compression's output shape depends on them.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    with pytest.raises(Exception):
        jax.jit(_jit_loss(**strategy))(jnp.asarray(UCELL))


def test_the_default_pairing_needs_no_frozen_geometry_at_all():
    """('sampled', 'li') decides nothing, so it jits straight off the public API.

    It used to raise here. The one discrete decision on this path was the same-material
    short-circuit, and removing that left nothing between ``conv_solve`` and a trace --
    ``freeze_geometry()`` is now optional for DFS rather than the price of admission.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    got = float(jax.jit(_jit_loss(**DEFAULT_STRATEGY))(jnp.asarray(UCELL)))
    want = float(np.sum(np.asarray(
        fieldwave.rcwa(backend=0, ucell=UCELL, **DEFAULT_STRATEGY,
                       **KW).conv_solve().res.de_ti).real))
    assert got == pytest.approx(want, abs=1e-10)


def test_freeze_geometry_reports_what_the_strategy_needs():
    """Only the fields the active strategy consumes are populated.

    The default pairing populates nothing: it has no discrete decision left to take. (It had
    one -- the same-material short-circuit -- until that turned out to be a workaround for a
    singularity the convention merge had already removed.)
    """
    def geo(**strategy):
        return fieldwave.rcwa(backend=0, ucell=UCELL, **strategy, **KW).freeze_geometry()

    default = geo()
    assert default.normal_field is None and default.compression is None

    # Each axis pulls in its own decision, and they compose.
    assert geo(factorization='nvf').normal_field is not None
    assert geo(fourier_coefficients='integrated').compression is not None

    both = geo(fourier_coefficients='integrated', factorization='nvf')
    assert both.normal_field is not None and both.compression is not None


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


# --------------------------------------------------------------- the raster records

RASTER_KW = dict(n_top=1.0, n_bot=1.0, theta=0.1, phi=0.2, psi=0.3, fto=[2, 2],
                 period=[1000.0, 1000.0], wavelength=900.0, thickness=[300.0])


def _rod(width, height=10, n_rod=3.5, n_bg=1.0, n=20):
    """One layer: a centred rectangular rod, ``width`` x ``height`` cells of a 20x20 raster."""
    cell = np.ones((1, n, n)) * n_bg
    r0, c0 = (n - height) // 2, (n - width) // 2
    cell[0, r0:r0 + height, c0:c0 + width] = n_rod
    return cell


def _de_ti(ucell, strategy, geometry=None):
    solver = fieldwave.rcwa(backend=0, ucell=ucell, **strategy, **RASTER_KW)
    return np.asarray(solver.conv_solve(geometry=geometry).res.de_ti).real


@pytest.mark.parametrize('strategy', [dict(factorization='nvf'),
                                      dict(fourier_coefficients='integrated')],
                         ids=['nvf', 'cfs'])
def test_a_stale_raster_record_is_refused_rather_than_answered(strategy):
    """A record frozen on one raster must not be replayed onto a raster that has moved.

    Both raster records are sampled from, or indexed against, one concrete pattern — the NVF
    normal field says which way each interface points, the CFS compression says which rows and
    columns end a constant block. Move the pattern and each describes a layer that is no longer
    there, and the answer comes back wrong without looking wrong. Measured on a 40x40 rod frozen
    at 12 cells wide and replayed wider, before either check existed: NVF read a total of 0.710
    against a true 0.346, with single orders 0.55 off; CFS returned the *same* total (0.591) for
    widths 13, 16, 20 and 24 while the true answers ran 0.733 down to 0.496 — the replay had
    collapsed the layer back onto the one it was frozen on.

    This pins both halves, as the vector-path test does: refused where the record no longer
    fits, and exact where it is accepted, so the check is not merely conservative.
    """
    frozen = fieldwave.rcwa(backend=0, ucell=_rod(6), **strategy,
                            **RASTER_KW).freeze_geometry()

    for width in (7, 8, 10, 14):                   # the pattern moved: refused, not answered
        with pytest.raises(fieldwave.StaleGeometryError, match='no longer fits'):
            _de_ti(_rod(width), strategy, frozen)

    for n_rod in (3.5, 3.0, 2.0, 1.4):             # material only: accepted, and bit-exact
        ucell = _rod(6, n_rod=n_rod)
        assert np.array_equal(_de_ti(ucell, strategy, frozen),
                              _de_ti(ucell, strategy))


def test_a_compression_that_still_separates_the_blocks_is_kept():
    """The CFS check asks whether the record still fits, not whether it is the one re-derived.

    A cut with a line the layer no longer needs splits one constant block into two of the same
    value — the same profile written differently, not a different layer. Insisting the record
    match a re-derivation exactly refused 19 of 300 randomised replays that a rebuild agreed
    with to 0.0e+00, nearly all of them material sweeps that drove two neighbouring blocks to
    one index, which is the sweep freezing exists to serve. Asking only that every step still
    falls on a cut refused none of them, and admitted nothing worse than a reordered summation
    (3.6e-15 on the convolution matrices, 7.3e-15 on the efficiencies).
    """
    def stripes(values, n=20, split=10):
        cell = np.ones((1, n, n))
        width = n // len(values)
        for k, v in enumerate(values):
            cell[0, :split, k * width:(k + 1) * width] = v
            cell[0, split:, k * width:(k + 1) * width] = 0.5 * v + 0.5
        return cell

    frozen = fieldwave.rcwa(backend=0, ucell=stripes([1.0, 2.0, 3.0, 4.0]),
                            fourier_coefficients='integrated',
                            **RASTER_KW).freeze_geometry()

    for values in ([1.0, 2.0, 2.0, 4.0], [1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0]):
        ucell = stripes(values)                    # blocks merged: the frozen cut still fits
        replayed = _de_ti(ucell, dict(fourier_coefficients='integrated'), frozen)
        assert replayed == pytest.approx(_de_ti(ucell, dict(fourier_coefficients='integrated')), abs=1e-13)

    for values in ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]):
        with pytest.raises(fieldwave.StaleGeometryError, match='no longer fits'):
            _de_ti(stripes(values), dict(fourier_coefficients='integrated'), frozen)


def test_a_one_cell_axis_makes_every_same_shape_replay_valid():
    """On a 1D grating the normal field is constant, so the check must not reject a moved rod.

    ``normal_vector_field`` short-circuits a one-cell axis: the structure cannot vary along it,
    so every normal points along the other one whatever the pattern is. The tell has to know
    that, or NVF on 1D gratings — the case the short-circuit was added for — would refuse every
    replay that changed anything.
    """
    cfg = dict(n_top=1.0, n_bot=1.0, theta=0.2, phi=0.0, pol=0, fto=[5, 0],
               period=[700.0], wavelength=900.0, thickness=[200.0])

    def grating(width):
        cell = np.ones((1, 1, 20))
        cell[0, 0, :width] = 3.5
        return cell

    frozen = fieldwave.rcwa(backend=0, ucell=grating(8), factorization='nvf',
                            **cfg).freeze_geometry()
    for width in (8, 10, 14):
        solvers = [fieldwave.rcwa(backend=0, ucell=grating(width), factorization='nvf',
                                  **cfg) for _ in range(2)]
        replayed = np.asarray(solvers[0].conv_solve(geometry=frozen).res.de_ti)
        assert np.array_equal(replayed, np.asarray(solvers[1].conv_solve().res.de_ti))


def test_a_traced_replay_carries_no_check():
    """Under a trace the comparison is the thing the record exists to avoid, so it is skipped.

    This is the documented limit, pinned so it stays a decision rather than an accident: a
    jitted call replays the record on trust. A long fit should re-freeze on a schedule rather
    than wait to be told.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    solver = fieldwave.rcwa(backend=0, ucell=_rod(6), factorization='nvf', **RASTER_KW)
    frozen = solver.freeze_geometry()
    assert frozen.normal_field_source is not None, 'the tell was not recorded'

    jax_solver = fieldwave.rcwa(backend=1, ucell=_rod(6), factorization='nvf',
                                **RASTER_KW)

    def total(u):
        return jnp.sum(jax_solver.conv_solve(ucell=u, geometry=frozen).res.de_ti).real

    moved = jnp.asarray(_rod(14))
    with pytest.raises(fieldwave.StaleGeometryError):
        total(moved)                               # eager: refused
    assert np.isfinite(float(jax.jit(total)(moved)))   # traced: answered, unchecked


CFS_STRATEGY = dict(fourier_coefficients='integrated', factorization='li')


def test_the_compression_decides_itself_inside_a_gradient():
    """``('integrated', 'li')`` differentiates with no frozen geometry at all.

    The compression branches on values, so it cannot be traced by ``jax.jit`` — but ``jax.grad``
    is the weaker bar. A grad tracer carries a concrete primal, so each comparison still narrows
    to a Python ``bool`` one value at a time, which is all the decision needs. The legacy
    ``meent`` builder had this property by accident, being written in ``jnp``; the port lost it
    by routing the decision through ``np.asarray``, which no tracer will answer.

    Getting it back matters for one workflow in particular. A fit whose raster *pattern* moves
    every step — a binary topology optimization, where the design variable is which cell is
    which material — invalidates a frozen record every step, and under a trace nothing can say
    so. Deciding inside the gradient removes the record, and with it the whole failure mode.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def total(u):
        return jnp.sum(fieldwave.rcwa(backend=1, ucell=u, **CFS_STRATEGY, **KW)
                       .conv_solve().res.de_ti).real

    grad = np.asarray(jax.grad(total)(jnp.asarray(UCELL)))
    assert grad.shape == UCELL.shape and np.isfinite(grad).all()

    d = np.zeros_like(UCELL)
    d[0, 2, 3] = 1.0
    h = 1e-6
    fd = (float(total(jnp.asarray(UCELL + h * d))) -
          float(total(jnp.asarray(UCELL - h * d)))) / (2 * h)
    assert float(np.sum(grad * d)) == pytest.approx(fd, rel=1e-4), (np.sum(grad * d), fd)


def test_a_moving_pattern_needs_no_record_on_the_compression_path():
    """The payoff: a gradient stays correct while the raster pattern moves under it.

    Frozen, this is the case that goes stale every step and is refused (or, under a trace,
    answered wrongly). Unfrozen, the compression is re-decided per call, so there is nothing to
    be stale — checked here against a finite difference at each moved pattern.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    rng = np.random.default_rng(3)
    design = rng.random((1, 6, 6)) < 0.4

    def total(u):
        return jnp.sum(fieldwave.rcwa(backend=1, ucell=u, **CFS_STRATEGY, **KW)
                       .conv_solve().res.de_ti).real

    for flips in (0, 2, 4):
        pattern = design.copy()
        for k in range(flips):
            i, j = divmod(k * 7 % 36, 6)
            pattern[0, i, j] = ~pattern[0, i, j]
        ucell = np.where(pattern, 3.5, 1.0)

        grad = np.asarray(jax.grad(total)(jnp.asarray(ucell)))
        assert np.isfinite(grad).all()

        d = np.zeros_like(ucell)
        d[0, 1, 4] = 1.0
        h = 1e-6
        fd = (float(total(jnp.asarray(ucell + h * d))) -
              float(total(jnp.asarray(ucell - h * d)))) / (2 * h)
        assert float(np.sum(grad * d)) == pytest.approx(fd, rel=1e-4, abs=1e-9)


def test_the_normal_field_still_needs_a_freeze_to_differentiate():
    """The other side of the boundary, pinned so it stays a known limit.

    The normal field comes off SciPy's distance transform, which wants the whole layer as a
    NumPy array — a grad tracer will not give one, primal or not. So NVF has no unfrozen
    gradient, and a fit whose pattern moves must re-freeze rather than drop ``geometry=``.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    nvf = dict(fourier_coefficients='sampled', factorization='nvf')

    def total(u):
        return jnp.sum(fieldwave.rcwa(backend=1, ucell=u, **nvf, **KW)
                       .conv_solve().res.de_ti).real

    with pytest.raises(Exception):
        jax.grad(total)(jnp.asarray(UCELL))
