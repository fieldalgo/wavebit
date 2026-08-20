"""Frozen geometry makes the public API traceable (wavebit.StaticGeometry).

Two pipeline steps decide something structural rather than numeric — the NVF normal field and
the integrated block compression. Both branch on values, so neither survives ``jax.jit``. (There was
a third, the same-material short-circuit; removing it is what left the sampled path with nothing to
freeze.) ``freeze_geometry()`` takes those decisions once against the concrete
structure; handing the result back to ``solve`` makes the whole call jit- and
grad-traceable without changing a single number.

``jax.grad`` on its own is the weaker bar and the two steps part company there: the
compression decides itself inside a gradient, the normal field cannot. That boundary is worth
pinning, because on the near side a frozen record is optional — and an optional record is one
that cannot go stale.
"""
import importlib

import numpy as np
import pytest

import wavebit

#: A permittivity raster. Drawn as an index and squared, so it is the same layer this file
#: measured before `structure` started holding permittivity.
STRUCTURE = np.random.default_rng(0).uniform(1.0, 4.0, (1, 5, 5)) ** 2
KW = dict(top=1.0, bot=1.0, theta=0.1, phi=0.2, psi=0.3, max_order=[2, 2],
          period=[1.0, 1.0], wavelength=0.9, thickness=[0.2])

# Every point of the two-axis cross product, including ('integrated', 'nvf') -- the pairing
# that only became reachable once the two axes were named independently. It needs both frozen
# fields at once, which nothing else exercises.
STRATEGIES = [dict(fourier_coefficients=c, factorization=f)
              for c in ('sampled', 'integrated') for f in ('li', 'nvf')]
IDS = ['sampled-li', 'sampled-nvf', 'integrated-li', 'integrated-nvf']
DEFAULT_STRATEGY = dict(fourier_coefficients='sampled', factorization='li')


@pytest.mark.parametrize('strategy', STRATEGIES, ids=IDS)
def test_frozen_geometry_does_not_change_the_answer(strategy):
    """Freezing is an optimization, not a different computation."""
    solver = wavebit.rcwa(backend=0, structure=STRUCTURE, **strategy, **KW)
    auto = solver.solve()
    frozen = solver.solve(geometry=solver.freeze_geometry())
    assert np.array_equal(np.asarray(auto.res.R), np.asarray(frozen.res.R))
    assert np.array_equal(np.asarray(auto.res.T), np.asarray(frozen.res.T))


@pytest.mark.parametrize('strategy', STRATEGIES, ids=IDS)
def test_jit_and_grad_through_the_public_api(strategy):
    """jax.jit and jax.grad over rcwa/solve, for every Fourier strategy."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    solver = wavebit.rcwa(backend=1, structure=STRUCTURE, **strategy, **KW)
    geometry = solver.freeze_geometry()

    def loss(u):
        return jnp.sum(solver.solve(structure=u, geometry=geometry).res.T).real

    u = jnp.asarray(STRUCTURE)
    eager = float(loss(u))
    assert np.isclose(float(jax.jit(loss)(u)), eager, atol=1e-12, rtol=0)

    g = np.asarray(jax.grad(loss)(u))
    assert g.shape == STRUCTURE.shape and np.isfinite(g).all()

    # finite-difference spot check along the all-ones direction
    d = np.ones_like(STRUCTURE)
    h = 1e-6
    fd = (float(loss(jnp.asarray(STRUCTURE + h * d))) - float(loss(jnp.asarray(STRUCTURE - h * d)))) / (2 * h)
    assert abs(float(np.sum(g * d)) - fd) < 1e-5, (float(np.sum(g * d)), fd)


def _jit_loss(**strategy):
    import jax.numpy as jnp
    solver = wavebit.rcwa(backend=1, structure=STRUCTURE, **strategy, **KW)
    return lambda u: jnp.sum(solver.solve(structure=u).res.T).real


# The default pairing is left out on purpose: it decides nothing, so it traces unfrozen.
NEEDS_FREEZE = [s for s in STRATEGIES if s != DEFAULT_STRATEGY]
NEEDS_FREEZE_IDS = [i for i, s in zip(IDS, STRATEGIES) if s != DEFAULT_STRATEGY]


@pytest.mark.parametrize('strategy', NEEDS_FREEZE, ids=NEEDS_FREEZE_IDS)
def test_auto_detection_still_blocks_jit(strategy):
    """Pins *why* the API exists: for these strategies, tracing fails without it.

    Auto-detection is the eager path; the normal field and the integrated block compression both
    branch on concrete values, and the compression's output shape depends on them.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    with pytest.raises(Exception):
        jax.jit(_jit_loss(**strategy))(jnp.asarray(STRUCTURE))


def test_the_default_pairing_needs_no_frozen_geometry_at_all():
    """('sampled', 'li') decides nothing, so it jits straight off the public API.

    It used to raise here. The one discrete decision on this path was the same-material
    short-circuit, and removing that left nothing between ``solve`` and a trace --
    ``freeze_geometry()`` is now optional for the sampled path rather than the price of admission.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    got = float(jax.jit(_jit_loss(**DEFAULT_STRATEGY))(jnp.asarray(STRUCTURE)))
    want = float(np.sum(np.asarray(
        wavebit.rcwa(backend=0, structure=STRUCTURE, **DEFAULT_STRATEGY,
                       **KW).solve().res.T).real))
    assert got == pytest.approx(want, abs=1e-10)


def test_freeze_geometry_reports_what_the_strategy_needs():
    """Only the fields the active strategy consumes are populated.

    The default pairing populates nothing: it has no discrete decision left to take. (It had
    one -- the same-material short-circuit -- until that turned out to be a workaround for a
    singularity the convention merge had already removed.)
    """
    def geo(**strategy):
        return wavebit.rcwa(backend=0, structure=STRUCTURE, **strategy, **KW).freeze_geometry()

    default = geo()
    assert default.normal_field is None and default.compression is None

    # Each axis pulls in its own decision, and they compose.
    assert geo(factorization='nvf').normal_field is not None
    assert geo(fourier_coefficients='integrated').compression is not None

    both = geo(fourier_coefficients='integrated', factorization='nvf')
    assert both.normal_field is not None and both.compression is not None


@pytest.mark.parametrize('name,make,start', [
    ('ellipse lx', lambda v: [[12.25, [['ellipse', 140.0, 160.0, v, 55.0, 4.0, 0.27, 9, 9]]]], 85.0),
    ('rectangle lx', lambda v: [[12.25, [['rectangle', 140.0, 160.0, v, 55.0, 4.0, 0.27, 4, 4]]]],
     85.0),
    ('rectangle angle',
     lambda v: [[12.25, [['rectangle', 140.0, 160.0, 85.0, 55.0, 4.0, v, 4, 4]]]], 0.27),
    ('overlapping pair',
     lambda v: [[1.0, [['rectangle', 130.0, 150.0, v, 120.0, 4.0, 0.0, 0, 0],
                       ['rectangle', 190.0, 150.0, 120.0, 120.0, 9.0, 0.0, 0, 0]]]], 120.0),
])
def test_a_frozen_geometry_makes_the_vector_path_traceable(name, make, start):
    """``jax.jit`` over a shape parameter, which the block path could not do before.

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
    importlib.import_module('wavebit.on_jax')

    cfg = dict(period=[300.0, 300.0], thickness=[205.0], max_order=[3, 3], top=1.0, bot=1.0,
               theta=0.15, phi=0.35, psi=np.pi / 2, wavelength=900.0)
    x = jnp.asarray(start)
    frozen = wavebit.rcwa(backend=1, structure=make(x), **cfg).freeze_geometry()
    assert frozen.block_grid is not None, 'the grid plan was not recorded'

    def total(v):
        return jnp.sum(wavebit.rcwa(backend=1, structure=make(v), **cfg)
                       .solve(geometry=frozen).T).real

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
    importlib.import_module('wavebit.on_jax')
    from wavebit._core.modeler import StaleGeometryError

    cfg = dict(period=[300.0, 300.0], thickness=[205.0], max_order=[3, 3], top=1.0, bot=1.0,
               theta=0.15, phi=0.35, psi=np.pi / 2, wavelength=900.0)
    make = lambda lx: [[12.25, [['ellipse', 150.0, 150.0, lx, 55.0, 4.0, 0.27, 9, 9]]]]

    def total(lx, geometry=None):
        return float(jnp.sum(wavebit.rcwa(backend=1, structure=make(lx), **cfg)
                             .solve(geometry=geometry).T).real)

    frozen = wavebit.rcwa(backend=1, structure=make(jnp.asarray(85.0)), **cfg).freeze_geometry()

    for lx in (85.0, 100.0):                       # near the freeze: accepted, and exact
        assert total(jnp.asarray(lx), frozen) == total(lx)

    for lx in (120.0, 200.0):                      # far from it: refused, not answered
        with pytest.raises(StaleGeometryError, match='no longer fits'):
            total(jnp.asarray(lx), frozen)


# --------------------------------------------------------------- the raster records

RASTER_KW = dict(top=1.0, bot=1.0, theta=0.1, phi=0.2, psi=0.3, max_order=[2, 2],
                 period=[1000.0, 1000.0], wavelength=900.0, thickness=[300.0])


def _rod(width, height=10, eps_rod=12.25, eps_bg=1.0, n=20):
    """One layer: a centred rectangular rod, ``width`` x ``height`` cells of a 20x20 raster.

    ``eps_rod`` / ``eps_bg`` are permittivities -- 12.25 is the index 3.5 this rod was before
    the material flip, so the layer is unchanged.
    """
    cell = np.ones((1, n, n)) * eps_bg
    r0, c0 = (n - height) // 2, (n - width) // 2
    cell[0, r0:r0 + height, c0:c0 + width] = eps_rod
    return cell


def _de_ti(structure, strategy, geometry=None):
    solver = wavebit.rcwa(backend=0, structure=structure, **strategy, **RASTER_KW)
    return np.asarray(solver.solve(geometry=geometry).res.T).real


@pytest.mark.parametrize('strategy', [dict(factorization='nvf'),
                                      dict(fourier_coefficients='integrated')],
                         ids=['nvf', 'cfs'])
def test_a_stale_raster_record_is_refused_rather_than_answered(strategy):
    """A record frozen on one raster must not be replayed onto a raster that has moved.

    Both raster records are sampled from, or indexed against, one concrete pattern — the NVF
    normal field says which way each interface points, the integrated compression says which rows and
    columns end a constant block. Move the pattern and each describes a layer that is no longer
    there, and the answer comes back wrong without looking wrong. Measured on a 40x40 rod frozen
    at 12 cells wide and replayed wider, before either check existed: NVF read a total of 0.710
    against a true 0.346, with single orders 0.55 off; the integrated path returned the *same* total (0.591) for
    widths 13, 16, 20 and 24 while the true answers ran 0.733 down to 0.496 — those four sample
    to the same three values at the frozen indices, so they all build one profile that is
    neither the true layer nor the frozen one (whose own answer is 0.490).

    This pins both halves, as the vector-path test does: refused where the record no longer
    fits, and exact where it is accepted, so the check is not merely conservative.
    """
    frozen = wavebit.rcwa(backend=0, structure=_rod(6), **strategy,
                            **RASTER_KW).freeze_geometry()

    for width in (7, 8, 10, 14):                   # the pattern moved: refused, not answered
        with pytest.raises(wavebit.StaleGeometryError, match='no longer fits'):
            _de_ti(_rod(width), strategy, frozen)

    # 3.5, 3.0, 2.0 and 1.4 squared: material only, so accepted, and bit-exact.
    for eps_rod in (12.25, 9.0, 4.0, 1.9599999999999997):
        cell = _rod(6, eps_rod=eps_rod)
        assert np.array_equal(_de_ti(cell, strategy, frozen),
                              _de_ti(cell, strategy))


def test_a_compression_that_still_separates_the_blocks_is_kept():
    """The integrated check asks whether the record still fits, not whether it is the one re-derived.

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

    frozen = wavebit.rcwa(backend=0, structure=stripes([1.0, 2.0, 3.0, 4.0]),
                            fourier_coefficients='integrated',
                            **RASTER_KW).freeze_geometry()

    for values in ([1.0, 2.0, 2.0, 4.0], [1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0]):
        structure = stripes(values)                    # blocks merged: the frozen cut still fits
        replayed = _de_ti(structure, dict(fourier_coefficients='integrated'), frozen)
        assert replayed == pytest.approx(_de_ti(structure, dict(fourier_coefficients='integrated')), abs=1e-13)

    for values in ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]):
        with pytest.raises(wavebit.StaleGeometryError, match='no longer fits'):
            _de_ti(stripes(values), dict(fourier_coefficients='integrated'), frozen)


def test_a_one_cell_axis_makes_every_same_shape_replay_valid():
    """On a 1D grating the normal field is constant, so the check must not reject a moved rod.

    ``normal_vector_field`` short-circuits a one-cell axis: the structure cannot vary along it,
    so every normal points along the other one whatever the pattern is. The tell has to know
    that, or NVF on 1D gratings — the case the short-circuit was added for — would refuse every
    replay that changed anything.
    """
    cfg = dict(top=1.0, bot=1.0, theta=0.2, phi=0.0, psi=np.pi / 2, max_order=[5, 0],
               period=[700.0], wavelength=900.0, thickness=[200.0])

    def grating(width):
        cell = np.ones((1, 1, 20))
        cell[0, 0, :width] = 12.25
        return cell

    frozen = wavebit.rcwa(backend=0, structure=grating(8), factorization='nvf',
                            **cfg).freeze_geometry()
    for width in (8, 10, 14):
        solvers = [wavebit.rcwa(backend=0, structure=grating(width), factorization='nvf',
                                  **cfg) for _ in range(2)]
        replayed = np.asarray(solvers[0].solve(geometry=frozen).res.T)
        assert np.array_equal(replayed, np.asarray(solvers[1].solve().res.T))


def test_a_traced_replay_carries_no_check():
    """Under a trace the comparison is the thing the record exists to avoid, so it is skipped.

    This is the documented limit, pinned so it stays a decision rather than an accident: a
    jitted call replays the record on trust. A long fit should re-freeze on a schedule rather
    than wait to be told.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    solver = wavebit.rcwa(backend=0, structure=_rod(6), factorization='nvf', **RASTER_KW)
    frozen = solver.freeze_geometry()
    assert frozen.normal_field_source is not None, 'the tell was not recorded'

    jax_solver = wavebit.rcwa(backend=1, structure=_rod(6), factorization='nvf',
                                **RASTER_KW)

    def total(u):
        return jnp.sum(jax_solver.solve(structure=u, geometry=frozen).res.T).real

    moved = jnp.asarray(_rod(14))
    with pytest.raises(wavebit.StaleGeometryError):
        total(moved)                               # eager: refused
    assert np.isfinite(float(jax.jit(total)(moved)))   # traced: answered, unchecked


INTEGRATED_STRATEGY = dict(fourier_coefficients='integrated', factorization='li')


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
        return jnp.sum(wavebit.rcwa(backend=1, structure=u, **INTEGRATED_STRATEGY, **KW)
                       .solve().res.T).real

    grad = np.asarray(jax.grad(total)(jnp.asarray(STRUCTURE)))
    assert grad.shape == STRUCTURE.shape and np.isfinite(grad).all()

    d = np.zeros_like(STRUCTURE)
    d[0, 2, 3] = 1.0
    h = 1e-6
    fd = (float(total(jnp.asarray(STRUCTURE + h * d))) -
          float(total(jnp.asarray(STRUCTURE - h * d)))) / (2 * h)
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
        return jnp.sum(wavebit.rcwa(backend=1, structure=u, **INTEGRATED_STRATEGY, **KW)
                       .solve().res.T).real

    for flips in (0, 2, 4):
        pattern = design.copy()
        for k in range(flips):
            i, j = divmod(k * 7 % 36, 6)
            pattern[0, i, j] = ~pattern[0, i, j]
        structure = np.where(pattern, 12.25, 1.0)

        grad = np.asarray(jax.grad(total)(jnp.asarray(structure)))
        assert np.isfinite(grad).all()

        d = np.zeros_like(structure)
        d[0, 1, 4] = 1.0
        h = 1e-6
        fd = (float(total(jnp.asarray(structure + h * d))) -
              float(total(jnp.asarray(structure - h * d)))) / (2 * h)
        assert float(np.sum(grad * d)) == pytest.approx(fd, rel=1e-4, abs=1e-9)


def test_the_normal_field_decides_itself_inside_a_gradient_too():
    """NVF differentiates unfrozen as well, so no raster strategy is forced to hold a record.

    It could not before, and the reason was narrow: ``normal_vector_field`` handed the whole
    *upsampled* layer to SciPy in one NumPy conversion, which no tracer answers. But the field
    reads nothing except the interface mask, and the upsample is a block repeat -- so the mask
    follows from the base raster, which narrows value by value like the compression's
    comparisons do, with ``repeat_y * repeat_x`` fewer of them.

    What this buys is the same thing the compression got: a fit whose raster pattern moves
    every step carries no record, so it cannot be served a stale one.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    for strategy in (dict(fourier_coefficients='sampled', factorization='nvf'),
                     dict(fourier_coefficients='integrated', factorization='nvf')):
        def total(u):
            return jnp.sum(wavebit.rcwa(backend=1, structure=u, **strategy, **KW)
                           .solve().res.T).real

        grad = np.asarray(jax.grad(total)(jnp.asarray(STRUCTURE)))
        assert grad.shape == STRUCTURE.shape and np.isfinite(grad).all()

        # Deriving the mask instead of reading it is meant to be the same mask, so the frozen
        # and unfrozen paths must not merely agree -- they must be the same computation.
        solver = wavebit.rcwa(backend=1, structure=STRUCTURE, **strategy, **KW)
        frozen = solver.freeze_geometry()

        def total_frozen(u):
            return jnp.sum(wavebit.rcwa(backend=1, structure=u, **strategy, **KW)
                           .solve(geometry=frozen).res.T).real

        assert np.array_equal(grad, np.asarray(jax.grad(total_frozen)(jnp.asarray(STRUCTURE))))


def test_the_nvf_gradient_holds_the_interfaces_fixed():
    """Pins what the NVF gradient *is*, because a finite difference does not measure it.

    The normal field is geometry: the gradient is taken with the interfaces held where they
    are. On a raster whose cells take two values that makes a finite difference the wrong
    instrument -- nudging a cell that shares a value with a neighbour *creates* an interface
    between them, so the difference quotient straddles two different structures and disagrees
    by 3-37%. Nudge a cell that already differs from all four neighbours, where the mask cannot
    move, and it agrees to 5e-8.

    Worth pinning because the disagreement looks exactly like a broken gradient and is not one.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from wavebit._core import factorization as _fac

    strategy = dict(fourier_coefficients='sampled', factorization='nvf')
    rng = np.random.default_rng(0)
    structure = np.where(rng.random((1, 8, 8)) < 0.35, 12.25, 1.0)

    def total(u):
        return jnp.sum(wavebit.rcwa(backend=1, structure=u, **strategy, **KW)
                       .solve().res.T).real

    def mask_size(u):
        return int(_fac.normal_field_source_for_raster(u, KW['max_order'][0], KW['max_order'][1])[0][1].sum())

    grad = np.asarray(jax.grad(total)(jnp.asarray(structure)))
    step = 1e-4

    frozen_interfaces = [(r, c) for r in range(1, 7) for c in range(1, 7)
                         if all(structure[0, r + dr, c + dc] != structure[0, r, c]
                                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
    assert frozen_interfaces, 'no cell isolated enough to probe'

    r, c = frozen_interfaces[0]
    d = np.zeros_like(structure)
    d[0, r, c] = 1.0
    assert mask_size(structure + step * d) == mask_size(structure), 'the probe moved an interface'

    fd = (float(total(jnp.asarray(structure + step * d))) -
          float(total(jnp.asarray(structure - step * d)))) / (2 * step)
    assert float(np.sum(grad * d)) == pytest.approx(fd, rel=1e-5)



def test_a_length_that_moves_without_reordering_reuses_the_compilation():
    """Re-freezing every step costs a recompilation only when the *ordering* changes.

    The vector record is which block edge becomes which grid line, in what order — the edge
    coordinates stay traced, which is what carries a gradient to a length. So a length can move
    a long way with the record unchanged: swept over ``lx`` 85 to 125 below, the plan is
    identical until 115, where two edges cross.

    That is what makes re-freezing every step affordable, and re-freezing every step is what
    makes a stale record impossible — the fit never carries one forward. Compared against
    identity comparison, which recompiled on every freeze regardless, this sweep goes from ten
    compilations to two.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    importlib.import_module('wavebit.on_jax')

    cfg = dict(period=[300.0, 300.0], thickness=[205.0], max_order=[3, 3], top=1.0, bot=1.0,
               theta=0.15, phi=0.35, psi=np.pi / 2, wavelength=900.0)
    make = lambda lx: [[12.25, [['ellipse', 150.0, 150.0, lx, 55.0, 4.0, 0.27, 4, 4]]]]

    def freeze_at(lx):
        return wavebit.rcwa(backend=1, structure=make(jnp.asarray(lx)), **cfg).freeze_geometry()

    keys = {}
    for lx in (85.0, 90.0, 100.0, 106.0, 110.0, 115.0, 125.0):
        keys[lx] = freeze_at(lx)

    # Same ordering: a fresh freeze is the same static argument, hash included.
    for lx in (90.0, 100.0, 106.0, 110.0):
        assert keys[lx] == keys[85.0], lx
        assert hash(keys[lx]) == hash(keys[85.0]), lx

    # The ordering does change eventually, and then it must not be reused.
    assert keys[115.0] != keys[85.0]
    assert keys[125.0] == keys[115.0]

    # And the gradient is the one for the structure actually in front of it, each time.
    def total(lx, geometry):
        return jnp.sum(wavebit.rcwa(backend=1, structure=make(lx), **cfg)
                       .solve(geometry=geometry).T).real

    for lx in (100.0, 120.0):
        grad = float(jax.grad(total)(jnp.asarray(lx), freeze_at(lx)))
        step = 1e-3
        fd = (float(total(jnp.asarray(lx + step), freeze_at(lx + step))) -
              float(total(jnp.asarray(lx - step), freeze_at(lx - step)))) / (2 * step)
        assert grad == pytest.approx(fd, rel=2e-3), (lx, grad, fd)


PAINT_KW = dict(period=(1., 1.), thickness=(0.32,), wavelength=0.94, top=1., bot=2.1025,
                theta=0.2, phi=0.3, psi=0.7, max_order=(4, 4), normal_grid=256)


#: 3.4**2 and 2.1**2 -- the materials these two rectangles always had, as permittivity.
EPS_A, EPS_B = 11.559999999999999, 4.41


def _painted(cx, material=EPS_A):
    """Two painted rectangles; the first crosses the second as ``cx`` runs past ~0.52."""
    return [dict(background=1.0, combine='paint', shapes=[
        dict(type='rectangle', cx=cx, cy=0.5, lx=0.34, ly=0.26, angle=0.3, material=material),
        dict(type='rectangle', cx=0.55, cy=0.55, lx=0.30, ly=0.24, angle=0.0,
             material=EPS_B)])]


def _painted_t(cx, frozen=None, material=EPS_A):
    solver = wavebit.rcwa(backend=0, structure=_painted(cx, material), **PAINT_KW)
    return np.asarray(solver.solve(geometry=frozen).T)


def test_a_stale_painting_plan_is_refused_rather_than_answered():
    """A painting plan survives a shape *moving*; it does not survive the overlap changing.

    This record is unlike the others here. It holds no coordinates at all -- only which loop
    holds which vertex and which edges cross -- and that is deliberate: it is what lets a replay
    carry a gradient to vertices that have moved, which is the whole reason the boundary path
    can be traced. So the usual reading of "stale" is wrong for it. Moving a shape is *exactly*
    what it is built to survive, and the first loop below pins that, bit-for-bit.

    What it cannot survive is the combinatorics changing underneath it. Slide the first
    rectangle far enough and it begins to overlap the second; the recorded loops then describe a
    painting nobody asked for. Before the check, the replay took it: bit-identical through
    ``cx = 0.50`` and then 3.9e-4, 2.3e-2, 3.3e-2 wrong at 0.55, 0.62, 0.75, with no step in
    between to notice. Re-planning to compare costs 0.2% of a frozen ``solve`` on this layer
    (1.2% on one of ten overlapping shapes, less as ``max_order`` rises) -- the freeze here buys
    traceability rather than a speed plateau, so that is not what is being traded.
    """
    frozen = wavebit.rcwa(backend=0, structure=_painted(0.40), **PAINT_KW).freeze_geometry()
    assert frozen.boundary_plans is not None, 'the plan was not recorded'

    for cx in (0.40, 0.405, 0.42, 0.45, 0.50):     # moved, same overlap: accepted, and exact
        assert np.array_equal(_painted_t(cx, frozen), _painted_t(cx))

    for cx in (0.55, 0.62, 0.75):                  # the overlap changed: refused, not answered
        with pytest.raises(wavebit.StaleGeometryError, match='no longer fits'):
            _painted_t(cx, frozen)

    for eps in (9.0, 4.0, 1.9599999999999997):     # material only: never moves a vertex
        assert np.array_equal(_painted_t(0.40, frozen, eps),
                              _painted_t(0.40, material=eps))


def test_the_painting_plan_check_is_skipped_under_a_trace_too():
    """Same documented limit as the raster records, reached a different way.

    A tracer has no coordinates to re-plan from, so there is nothing to compare and the replay
    goes through on trust -- including onto an overlap the plan does not describe. Pinned so it
    stays a decision: ``docs/boundary-modeling.md`` tells a fit to re-freeze every step, and
    this is why that instruction is not optional.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def total(cx, frozen):
        solver = wavebit.rcwa(backend=1, structure=_painted(cx), **PAINT_KW)
        return jnp.sum(solver.solve(geometry=frozen).T).real

    frozen = wavebit.rcwa(backend=1, structure=_painted(0.40), **PAINT_KW).freeze_geometry()

    with pytest.raises(wavebit.StaleGeometryError):        # eagerly, jax converts: checked
        total(0.75, frozen)

    jitted = jax.jit(total, static_argnums=1)              # traced: not checked, and finite
    assert np.isfinite(float(jitted(jnp.asarray(0.75), frozen)))
    assert np.isfinite(float(jax.grad(total)(jnp.asarray(0.40), frozen)))


def _hidden(cx, material=4.0):
    """A small circle painted *under* a larger one — completely covered, so it has no loops."""
    return [dict(background=1.0, combine='paint', shapes=[
        dict(type='circle', cx=cx, cy=0.43, lx=0.12, material=material),
        dict(type='circle', cx=0.35, cy=0.45, lx=0.25, material=9.0)])]


def test_a_shape_the_plan_recorded_as_covered_cannot_be_slid_out_from_under_it():
    """The sharpest reading of a stale plan: a shape the record says is not there.

    The first circle sits entirely inside the second (centres 0.036 apart, radii 0.12 and 0.25),
    and the second is painted over it, so the plan gives shape 0 **zero** loops. Slide it out and
    the replay keeps every loop it recorded, which is the same structure it always was: before
    the check it returned 0.929791164 for *every* position tried — 0.42, 0.55, 0.60, 0.70 — while
    the true answers ran 0.9295, 0.9536, 0.9483, 0.9382. The shape had simply left the structure.

    This case also pins *what* is compared. From ``cx = 0.55`` on, the crossing table is
    identical and only the loop assignment differs, so a check that compared the table alone
    would pass all three of those and answer with a circle that is not in the layer.
    """
    frozen = wavebit.rcwa(backend=0, structure=_hidden(0.38), **PAINT_KW).freeze_geometry()
    assert frozen.boundary_plans[0][1][0] == [], 'the covered shape was expected to have no loops'

    def total(cx, geometry=None, material=4.0):
        solver = wavebit.rcwa(backend=0, structure=_hidden(cx, material), **PAINT_KW)
        return np.asarray(solver.solve(geometry=geometry).T)

    for cx in (0.38, 0.40):                        # still covered: accepted, and exact
        assert np.array_equal(total(cx, frozen), total(cx))

    for eps in (6.760000000000001, 2.25):          # material only: never uncovers anything
        assert np.array_equal(total(0.38, frozen, eps), total(0.38, material=eps))

    for cx in (0.42, 0.55, 0.60, 0.70):            # out from under it: refused, not answered
        with pytest.raises(wavebit.StaleGeometryError, match='no longer fits'):
            total(cx, frozen)


def _disjoint(cx):
    """A rectangle and a circle that do not touch until ``cx`` is dragged down onto the circle."""
    return [dict(background=1.0, combine='paint', shapes=[
        dict(type='rectangle', cx=cx, cy=0.407, lx=0.30, ly=0.24, angle=0.3, material=EPS_A),
        dict(type='circle', cx=0.225, cy=0.354, lx=0.18, material=EPS_B)])]


def test_a_frozen_no_overlap_is_a_claim_about_the_layer_and_is_checked_too():
    """``plan is None`` is one of the record's two answers, not the absence of one.

    A painted layer with nothing overlapping is summed, which is exact and cheaper, and the
    freeze records that decision as a ``None`` plan. It is a claim about the geometry, and it
    goes stale the same way a plan does — but for a long time it went stale silently in a
    different place: the dispatch took the summing branch on the record's word, and the summing
    builder refused with a ``ValueError`` naming *itself* and pointing at the boundary path the
    caller was already inside. Nothing in it mentioned the frozen record or said to re-freeze,
    and the same call without ``geometry=`` returns 0.933226529 quite happily.
    """
    frozen = wavebit.rcwa(backend=0, structure=_disjoint(0.629), **PAINT_KW).freeze_geometry()
    assert frozen.boundary_plans == [None], 'the layer was expected to be recorded as disjoint'

    def total(cx, geometry=None):
        solver = wavebit.rcwa(backend=0, structure=_disjoint(cx), **PAINT_KW)
        return np.asarray(solver.solve(geometry=geometry).T)

    for cx in (0.629, 0.60, 0.70):                 # still disjoint: accepted, and exact
        assert np.array_equal(total(cx, frozen), total(cx))

    with pytest.raises(wavebit.StaleGeometryError, match='not overlapping'):
        total(0.374, frozen)                       # dragged onto the circle
    assert np.isfinite(total(0.374).sum()), 'the unfrozen call solves this layer fine'

    # And the re-asking has to stay eager. A tracer cannot answer the overlap question at all --
    # `layer_needs_resolving` raises rather than guess, which is the whole reason the record
    # carries the answer -- so a check that re-asked unconditionally would take the one path this
    # record exists to enable and break it, with an error reading "needs a frozen geometry" at a
    # call that was given one.
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def traced_total(cx):
        solver = wavebit.rcwa(backend=1, structure=_disjoint(cx), **PAINT_KW)
        return jnp.sum(solver.solve(geometry=frozen_jax).T).real

    frozen_jax = wavebit.rcwa(backend=1, structure=_disjoint(0.629), **PAINT_KW).freeze_geometry()
    assert frozen_jax.boundary_plans == [None]
    assert np.isfinite(float(jax.jit(traced_total)(jnp.asarray(0.629))))
    assert np.isfinite(float(jax.grad(traced_total)(jnp.asarray(0.629))))
