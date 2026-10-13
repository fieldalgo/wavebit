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

from wavebit.compat import legacy as _legacy_wavebit

import numpy as np
import pytest

import wavebit

#: A permittivity raster. Drawn as an index and squared, so it is the same layer this file
#: measured before `structure` started holding permittivity.
STRUCTURE = np.random.default_rng(0).uniform(1.0, 4.0, (1, 5, 5)) ** 2
KW = dict(top=1.0, bot=1.0, theta=0.1, phi=0.2, psi=0.3, truncation_order=[2, 2],
          period=[1.0, 1.0], wavelength=0.9, thickness=[0.2])

# Every point of the two-axis cross product, including ('integrated', 'nvf') -- the pairing
# that only became reachable once the two axes were named independently. It needs both frozen
# fields at once, which nothing else exercises.
STRATEGIES = [dict(fourier_coefficients=c, factorization=f)
              for c in ('sampled', 'integrated') for f in ('li', 'nvf', 'laurent')]
IDS = [f"{s['fourier_coefficients']}-{s['factorization']}" for s in STRATEGIES]
DEFAULT_STRATEGY = dict(fourier_coefficients='sampled', factorization='li')
#: The pairings with no discrete decision to take: the sampled series under a rule that reads
#: no normal field. They trace straight off the public API, frozen or not.
DECIDES_NOTHING = [DEFAULT_STRATEGY, dict(fourier_coefficients='sampled', factorization='laurent')]


@pytest.mark.parametrize('strategy', STRATEGIES, ids=IDS)
def test_frozen_geometry_does_not_change_the_answer(strategy):
    """Freezing is an optimization, not a different computation."""
    solver = _legacy_wavebit.rcwa(backend='numpy', structure=STRUCTURE, **strategy, **KW)
    auto = solver.solve()
    frozen = solver.solve(geometry=solver.freeze_geometry())
    assert np.array_equal(np.asarray(auto.incident.R), np.asarray(frozen.incident.R))
    assert np.array_equal(np.asarray(auto.incident.T), np.asarray(frozen.incident.T))


@pytest.mark.parametrize('strategy', STRATEGIES, ids=IDS)
def test_jit_and_grad_through_the_public_api(strategy):
    """jax.jit and jax.grad over rcwa/solve, for every Fourier strategy."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    solver = _legacy_wavebit.rcwa(backend='jax', structure=STRUCTURE, **strategy, **KW)
    geometry = solver.freeze_geometry()

    def loss(u):
        return jnp.sum(solver.solve(structure=u, geometry=geometry).incident.T).real

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
    solver = _legacy_wavebit.rcwa(backend='jax', structure=STRUCTURE, **strategy, **KW)
    return lambda u: jnp.sum(solver.solve(structure=u).incident.T).real


# The pairings that decide nothing are left out on purpose: they trace unfrozen.
NEEDS_FREEZE = [s for s in STRATEGIES if s not in DECIDES_NOTHING]
NEEDS_FREEZE_IDS = [i for i, s in zip(IDS, STRATEGIES) if s not in DECIDES_NOTHING]


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


@pytest.mark.parametrize('strategy', DECIDES_NOTHING,
                         ids=[s['factorization'] for s in DECIDES_NOTHING])
def test_the_pairings_that_decide_nothing_need_no_frozen_geometry_at_all(strategy):
    """('sampled', 'li') decides nothing, so it jits straight off the public API -- and so does
    ('sampled', 'laurent'), which reads the same staircase and builds no normal field.

    It used to raise here. The one discrete decision on this path was the same-material
    short-circuit, and removing that left nothing between ``solve`` and a trace --
    ``freeze_geometry()`` is now optional for the sampled path rather than the price of admission.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    got = float(jax.jit(_jit_loss(**strategy))(jnp.asarray(STRUCTURE)))
    want = float(np.sum(np.asarray(
        _legacy_wavebit.rcwa(backend='numpy', structure=STRUCTURE, **strategy,
                       **KW).solve().incident.T).real))
    assert got == pytest.approx(want, abs=1e-10)


def test_freeze_geometry_reports_what_the_strategy_needs():
    """Only the fields the active strategy consumes are populated.

    The default pairing populates nothing: it has no discrete decision left to take. (It had
    one -- the same-material short-circuit -- until that turned out to be a workaround for a
    singularity the convention merge had already removed.)
    """
    def geo(**strategy):
        return _legacy_wavebit.rcwa(backend='numpy', structure=STRUCTURE, **strategy, **KW).freeze_geometry()

    default = geo()
    assert default.normal_field is None and default.compression is None

    # Each axis pulls in its own decision, and they compose.
    assert geo(factorization='nvf').normal_field is not None
    assert geo(fourier_coefficients='integrated').compression is not None

    both = geo(fourier_coefficients='integrated', factorization='nvf')
    assert both.normal_field is not None and both.compression is not None

    # Laurent's rule reads the series and no normal field, so it records what the series needs.
    assert geo(factorization='laurent').normal_field is None
    assert geo(factorization='laurent').compression is None
    laurent = geo(fourier_coefficients='integrated', factorization='laurent')
    assert laurent.normal_field is None and laurent.compression is not None


@pytest.mark.parametrize('name,make,start', [
    ('ellipse lx',
     lambda v: [dict(background=12.25, shapes=[dict(type='ellipse', cx=-10.0, cy=10.0,
                                                    lx=v, ly=55.0, angle=0.27, material=4.0)])],
     85.0),
    ('rectangle lx',
     lambda v: [dict(background=12.25, shapes=[dict(type='rectangle', cx=-10.0, cy=10.0,
                                                    lx=v, ly=55.0, angle=0.27, material=4.0)])],
     85.0),
    ('rectangle angle',
     lambda v: [dict(background=12.25, shapes=[dict(type='rectangle', cx=-10.0, cy=10.0,
                                                    lx=85.0, ly=55.0, angle=v, material=4.0)])],
     0.27),
    ('overlapping pair',
     lambda v: [dict(background=1.0, shapes=[
         dict(type='rectangle', cx=-20.0, cy=0.0, lx=v, ly=120.0, angle=0.0, material=4.0),
         dict(type='rectangle', cx=40.0, cy=0.0, lx=120.0, ly=120.0, angle=0.0,
              material=9.0)])],
     120.0),
])
def test_a_frozen_geometry_makes_a_shape_parameter_traceable(name, make, start):
    """``jax.jit`` over a shape parameter, which no shape path could do before freezing.

    What stops it is never arithmetic: it is the comparisons that decide which shapes overlap
    and how a painting resolves into disjoint polygons. Those are discrete, so
    ``freeze_geometry`` records them and the traced call replays them -- the same bargain the
    normal field already makes, and with the same caveat, that a parameter moved far enough to
    change the combinatorics leaves the record describing a layer that is no longer there.

    The overlapping pair is here because it is the case that has no alternative: the closed-form
    builder superposes rather than paints and refuses an overlap outright (see test_parametric.py),
    so a layer whose objects cover each other can only be built through the boundary sub-path.

    These four were block instructions until that path was removed. The theorem is the same one
    and the spelling is the surviving shape one.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit.enable_x64()

    cfg = dict(period=[300.0, 300.0], thickness=[205.0], truncation_order=[3, 3], top=1.0, bot=1.0,
               theta=0.15, phi=0.35, psi=np.pi / 2, wavelength=900.0)
    x = jnp.asarray(start)
    frozen = _legacy_wavebit.rcwa(backend='jax', structure=make(x), **cfg).freeze_geometry()
    # A painted, overlapping layer records its polygon plan; a disjoint one has no discrete
    # decision in it and records nothing. Either way the freeze is what makes the trace legal.
    if 'overlapping' in name:
        assert frozen.boundary_plans is not None, 'the polygon plan was not recorded'

    def total(v):
        return jnp.sum(_legacy_wavebit.rcwa(backend='jax', structure=make(v), **cfg)
                       .solve(geometry=frozen).T).real

    eager = float(total(x))
    assert float(jax.jit(total)(x)) == pytest.approx(eager, rel=1e-11)

    grad = float(jax.grad(total)(x))
    assert grad != 0.0
    assert float(jax.jit(jax.grad(total))(x)) == pytest.approx(grad, rel=1e-11)

    # 1e-5, not 1e-3: the rectangle's normal field is sampled on a grid, and a rotation of
    # 1e-3 rad sweeps its face lines across enough samples that the secant averages a
    # sawtooth in (2.6e-3 off autograd on the angle row with the piece field of 2026-09-06,
    # 1.5e-5 at 1e-5; the other three rows sit within 3e-5 at either step). The derivative
    # itself is not in question -- it agrees with the difference once the difference is
    # taken below the sawtooth's scale.
    step = 1e-5
    fd = (float(total(jnp.asarray(start + step))) - float(total(jnp.asarray(start - step))))\
        / (2 * step)
    assert grad == pytest.approx(fd, rel=2e-3), (grad, fd)


def test_a_stale_frozen_geometry_is_refused_rather_than_answered():
    """Reusing a frozen plan too far from where it was taken has to fail loudly.

    The plan records the *combinatorics* of the painting -- which loop holds which vertex,
    which edges cross. A shape moving is exactly what it is built to survive, since that is
    what carries the gradient; what makes it stale is the combinatorics changing, which here
    is the two rectangles ceasing to overlap. Replay it past that point and the plan describes
    a different layer, and the answer comes back wrong without looking wrong.

    This pins both halves — that it is refused where it does not fit, and that where it *is*
    accepted the answer matches a rebuild exactly, so the check is not merely conservative.
    The docstring described the removed block path's grid plan, and a sortedness tell that
    went with it, until 2026-08-27; the body has used the boundary plan throughout.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit.enable_x64()
    from wavebit.rcwa._engine.geometry import StaleGeometryError

    cfg = dict(period=[300.0, 300.0], thickness=[205.0], truncation_order=[3, 3], top=1.0, bot=1.0,
               theta=0.15, phi=0.35, psi=np.pi / 2, wavelength=900.0)
    # Two rectangles whose overlap is the discrete decision. Rectangles rather than an
    # ellipse: an ellipse's loop is discretized, so its vertex indices move with `lx` and any
    # change at all goes stale. A rectangle has four vertices wherever it is, so what the plan
    # records stays valid until the *combinatorics* change -- here, until the two stop
    # overlapping, which the centres 80 apart and the second's width 100 put at lx = 60.
    make = lambda lx: [dict(background=12.25, shapes=[
        dict(type='rectangle', cx=-30.0, cy=0.0, lx=lx, ly=100.0, angle=0.0, material=4.0),
        dict(type='rectangle', cx=50.0, cy=0.0, lx=100.0, ly=100.0, angle=0.0,
             material=9.0)])]

    def total(lx, geometry=None):
        return float(jnp.sum(_legacy_wavebit.rcwa(backend='jax', structure=make(lx), **cfg)
                             .solve(geometry=geometry).T).real)

    frozen = _legacy_wavebit.rcwa(backend='jax', structure=make(jnp.asarray(80.0)), **cfg).freeze_geometry()

    for lx in (75.0, 65.0, 62.0):                  # still overlapping: accepted, and exact
        assert total(jnp.asarray(lx), frozen) == total(lx)

    for lx in (60.0, 40.0):                        # no longer overlapping: refused
        with pytest.raises(StaleGeometryError, match='no longer fits'):
            total(jnp.asarray(lx), frozen)


# --------------------------------------------------------------- the raster records

RASTER_KW = dict(top=1.0, bot=1.0, theta=0.1, phi=0.2, psi=0.3, truncation_order=[2, 2],
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
    solver = _legacy_wavebit.rcwa(backend='numpy', structure=structure, **strategy, **RASTER_KW)
    return np.asarray(solver.solve(geometry=geometry).incident.T).real


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

    This pins both halves, as `test_a_stale_frozen_geometry_is_refused_rather_than_answered`
    does: refused where the record no longer
    fits, and exact where it is accepted, so the check is not merely conservative.
    """
    frozen = _legacy_wavebit.rcwa(backend='numpy', structure=_rod(6), **strategy,
                            **RASTER_KW).freeze_geometry()

    for width in (7, 8, 10, 14):                   # the pattern moved: refused, not answered
        with pytest.raises(_legacy_wavebit.StaleGeometryError, match='no longer fits'):
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

    frozen = _legacy_wavebit.rcwa(backend='numpy', structure=stripes([1.0, 2.0, 3.0, 4.0]),
                            fourier_coefficients='integrated',
                            **RASTER_KW).freeze_geometry()

    for values in ([1.0, 2.0, 2.0, 4.0], [1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0]):
        structure = stripes(values)                    # blocks merged: the frozen cut still fits
        replayed = _de_ti(structure, dict(fourier_coefficients='integrated'), frozen)
        assert replayed == pytest.approx(_de_ti(structure, dict(fourier_coefficients='integrated')), abs=1e-13)

    for values in ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]):
        with pytest.raises(_legacy_wavebit.StaleGeometryError, match='no longer fits'):
            _de_ti(stripes(values), dict(fourier_coefficients='integrated'), frozen)


def test_a_one_cell_axis_makes_every_same_shape_replay_valid():
    """On a 1D grating the normal field is constant, so the check must not reject a moved rod.

    ``normal_vector_field`` short-circuits a one-cell axis: the structure cannot vary along it,
    so every normal points along the other one whatever the pattern is. The tell has to know
    that, or NVF on 1D gratings — the case the short-circuit was added for — would refuse every
    replay that changed anything.
    """
    cfg = dict(top=1.0, bot=1.0, theta=0.2, phi=0.0, psi=np.pi / 2, truncation_order=[5, 0],
               period=[700.0], wavelength=900.0, thickness=[200.0])

    def grating(width):
        cell = np.ones((1, 1, 20))
        cell[0, 0, :width] = 12.25
        return cell

    frozen = _legacy_wavebit.rcwa(backend='numpy', structure=grating(8), factorization='nvf',
                            **cfg).freeze_geometry()
    for width in (8, 10, 14):
        solvers = [_legacy_wavebit.rcwa(backend='numpy', structure=grating(width), factorization='nvf',
                                  **cfg) for _ in range(2)]
        replayed = np.asarray(solvers[0].solve(geometry=frozen).incident.T)
        assert np.array_equal(replayed, np.asarray(solvers[1].solve().incident.T))


def test_a_traced_replay_carries_no_check():
    """Under a trace the comparison is the thing the record exists to avoid, so it is skipped.

    This is the documented limit, pinned so it stays a decision rather than an accident: a
    jitted call replays the record on trust. A long fit should re-freeze on a schedule rather
    than wait to be told.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    solver = _legacy_wavebit.rcwa(backend='numpy', structure=_rod(6), factorization='nvf', **RASTER_KW)
    frozen = solver.freeze_geometry()
    assert frozen.normal_field_source is not None, 'the tell was not recorded'

    jax_solver = _legacy_wavebit.rcwa(backend='jax', structure=_rod(6), factorization='nvf',
                                **RASTER_KW)

    def total(u):
        return jnp.sum(jax_solver.solve(structure=u, geometry=frozen).incident.T).real

    moved = jnp.asarray(_rod(14))
    with pytest.raises(_legacy_wavebit.StaleGeometryError):
        total(moved)                               # eager: refused
    assert np.isfinite(float(jax.jit(total)(moved)))   # traced: answered, unchecked


@pytest.mark.parametrize('strategy', [dict(factorization='nvf'),
                                      dict(fourier_coefficients='integrated'),
                                      dict(fourier_coefficients='integrated', factorization='nvf')],
                         ids=['nvf', 'cfs', 'cfs-nvf'])
def test_the_jitted_field_door_refuses_a_stale_raster_record_too(strategy):
    """JAX's jitted ``solve_field`` compares the records with the live layer, as ``solve()`` does.

    It builds only under its trace, where a layer has no value, so until 2026-09-25 it replayed
    every record on trust -- and once keywords that move the pattern or the cell reached it, a
    frozen normal field replayed on another oblique cell answered ``sum(T)`` wrong by up to
    0.86 with nothing said. It now runs the builders' own checks on the live solver before its
    compiled call, without the build. Refused where the record no longer fits; accepted, and
    the eager answer to 1e-12, where only a material moved; and under an outer ``jax.jit``,
    where the pattern itself is traced, replayed as frozen -- the limit
    ``test_a_traced_replay_carries_no_check`` pins for ``solve``.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit.enable_x64()

    frozen = _legacy_wavebit.rcwa(backend='jax', structure=_rod(6), **strategy,
                          **RASTER_KW).freeze_geometry()
    solver = _legacy_wavebit.rcwa(backend='jax', structure=_rod(6), answers='field', **strategy,
                          **RASTER_KW)

    for width in (8, 14):
        with pytest.raises(_legacy_wavebit.StaleGeometryError, match='no longer fits'):
            solver.solve_field(4, 4, 2, geometry=frozen, structure=_rod(width))

    cell = _rod(6, eps_rod=4.0)
    got, _ = solver.solve_field(4, 4, 2, geometry=frozen, structure=cell)
    want = np.asarray(_legacy_wavebit.rcwa(backend='jax', structure=cell, **strategy,
                                   **RASTER_KW).solve().incident.T)
    assert np.abs(np.asarray(got.incident.T) - want).max() < 1e-12

    def total(u):
        result, _ = solver.solve_field(4, 4, 2, geometry=frozen, structure=u)
        return jnp.sum(result.incident.T).real
    assert np.isfinite(float(jax.jit(total)(jnp.asarray(_rod(14)))))


@pytest.mark.parametrize('transform', ['jit', 'grad'])
@pytest.mark.parametrize('quantity', ['structure', 'mu'])
@pytest.mark.parametrize('strategy', [dict(factorization='nvf'),
                                      dict(fourier_coefficients='integrated'),
                                      dict(fourier_coefficients='integrated', factorization='nvf')],
                         ids=['nvf', 'cfs', 'cfs-nvf'])
def test_retained_solve_tracers_do_not_hide_a_concrete_stale_record(transform, quantity, strategy):
    """An earlier traced solve cannot exempt the next concrete pattern from validation.

    The conditions are restored after jit/grad, but the retained modes and amplitudes can
    still hold tracers. Scanning every solver attribute skipped the stale-record check on
    that history, although the raster and cell being checked were concrete. Both material
    quantities and both kinds of frozen raster record must still refuse a moved interface.
    Re-freezing the changed pattern must then recover the fresh eager result and field.
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit.enable_x64()
    kw = dict(RASTER_KW, period=[700.0, 700.0], thickness=[200.0],
              truncation_order=[1, 1], answers='field', structure=_rod(2, height=4, n=8))
    material = 12.25 if quantity == 'structure' else 1.7
    if quantity == 'mu':
        kw['mu'] = _rod(2, height=4, n=8, eps_rod=material)
    solver = _legacy_wavebit.rcwa(backend='jax', **kw, **strategy)
    frozen = solver.freeze_geometry()

    def total(wavelength):
        return solver.solve(wavelength=wavelength, geometry=frozen).T.sum().real

    assert np.isfinite(float(getattr(jax, transform)(total)(900.0)))
    moved = {quantity: _rod(5, height=4, n=8, eps_rod=material)}
    with pytest.raises(_legacy_wavebit.StaleGeometryError, match='no longer fits'):
        solver.solve_field(2, 2, 2, geometry=frozen, **moved)

    result, field = solver.solve_field(2, 2, 2, geometry=solver.freeze_geometry())
    fresh = _legacy_wavebit.rcwa(backend='jax', **(kw | moved), **strategy)
    expected, expected_field = fresh.solve_field_eager(2, 2, 2)
    np.testing.assert_allclose(result.T, expected.T, atol=1e-11, rtol=0)
    np.testing.assert_allclose(field, expected_field, atol=1e-10, rtol=0)


def test_the_jitted_field_door_checks_the_permeability_records_beside_the_permittivity():
    """A raster ``mu`` carries records of its own, and the door compares them too.

    A magnetic pattern need not sit where the dielectric one does, so the freeze keeps a
    normal field for each; the check mirrors the build, which replays the permeability's own.
    """
    pytest.importorskip('jax')
    _legacy_wavebit.enable_x64()
    kw = dict(structure=_rod(6), factorization='nvf', answers='field', **RASTER_KW)
    mu = _rod(6, eps_rod=1.7)
    frozen = _legacy_wavebit.rcwa(backend='jax', mu=mu, **kw).freeze_geometry()
    assert frozen.mu_normal_field_source is not None
    solver = _legacy_wavebit.rcwa(backend='jax', mu=mu, **kw)
    with pytest.raises(_legacy_wavebit.StaleGeometryError, match='no longer fits'):
        solver.solve_field(4, 4, 2, geometry=frozen, mu=_rod(12, eps_rod=1.7))
    solver.solve_field(4, 4, 2, geometry=frozen, mu=_rod(6, eps_rod=1.3))


def test_a_normal_field_frozen_on_one_size_of_a_cell_replays_on_another():
    """One cell shape at two sizes is one normal field, and the replay is accepted.

    The record keys the basis scaled to ``|a1| = 1``, because a uniform rescale leaves the
    field unchanged, and compared that key exactly. The scaling rounds: ``hexagonal(640.)`` and
    ``hexagonal(700.)`` keyed ``-0.49999999999999983`` and ``-0.4999999999999998`` for the same
    entry, and the replay was refused as a moved cell (measured 2026-09-25). It is compared to
    rounding now. The accepted replay is the fresh freeze's answer, and a different shape is
    still refused.
    """
    eps = np.where(np.random.default_rng(0).random((1, 8, 8)) < 0.4, 5.76, 1.0)

    def solver(cell):
        return _legacy_wavebit.rcwa(backend='numpy', lattice=cell, structure=eps, truncation_order=[2, 2],
                            thickness=[300.0], wavelength=900.0, theta=0.2, phi=0.1, psi=0.3,
                            factorization='nvf')

    frozen = solver(wavebit.Lattice.hexagonal(640.0)).freeze_geometry()
    bigger = wavebit.Lattice.hexagonal(700.0)
    replayed = np.asarray(solver(bigger).solve(geometry=frozen).T)
    assert np.abs(replayed - np.asarray(solver(bigger).solve().T)).max() < 1e-12
    with pytest.raises(_legacy_wavebit.StaleGeometryError, match='cell basis'):
        solver(wavebit.Lattice.oblique(640.0, 640.0, np.radians(75.0))).solve(geometry=frozen)


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
        return jnp.sum(_legacy_wavebit.rcwa(backend='jax', structure=u, **INTEGRATED_STRATEGY, **KW)
                       .solve().incident.T).real

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
        return jnp.sum(_legacy_wavebit.rcwa(backend='jax', structure=u, **INTEGRATED_STRATEGY, **KW)
                       .solve().incident.T).real

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
    comparisons do, with ``repeat_v * repeat_u`` fewer of them.

    What this buys is the same thing the compression got: a fit whose raster pattern moves
    every step carries no record, so it cannot be served a stale one.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    for strategy in (dict(fourier_coefficients='sampled', factorization='nvf'),
                     dict(fourier_coefficients='integrated', factorization='nvf')):
        def total(u):
            return jnp.sum(_legacy_wavebit.rcwa(backend='jax', structure=u, **strategy, **KW)
                           .solve().incident.T).real

        grad = np.asarray(jax.grad(total)(jnp.asarray(STRUCTURE)))
        assert grad.shape == STRUCTURE.shape and np.isfinite(grad).all()

        # Deriving the mask instead of reading it is meant to be the same mask, so the frozen
        # and unfrozen paths must not merely agree -- they must be the same computation.
        solver = _legacy_wavebit.rcwa(backend='jax', structure=STRUCTURE, **strategy, **KW)
        frozen = solver.freeze_geometry()

        def total_frozen(u):
            return jnp.sum(_legacy_wavebit.rcwa(backend='jax', structure=u, **strategy, **KW)
                           .solve(geometry=frozen).incident.T).real

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
    from wavebit.rcwa._engine import factorization as _fac

    strategy = dict(fourier_coefficients='sampled', factorization='nvf')
    rng = np.random.default_rng(0)
    structure = np.where(rng.random((1, 8, 8)) < 0.35, 12.25, 1.0)

    def total(u):
        return jnp.sum(_legacy_wavebit.rcwa(backend='jax', structure=u, **strategy, **KW)
                       .solve().incident.T).real

    def mask_size(u):
        return int(_fac.normal_field_source_for_raster(u, KW['truncation_order'][0], KW['truncation_order'][1])[0][1].sum())

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



def test_a_length_that_moves_without_changing_the_overlaps_reuses_the_compilation():
    """Re-freezing every step costs a recompilation only when the *combinatorics* change.

    The record is which shapes overlap and how the painting resolves into disjoint polygons --
    the vertex coordinates stay traced, which is what carries a gradient to a length. So a
    length can move a long way with the record unchanged: swept over ``lx`` 80 to 180 below,
    the plan is identical until 122, where the middle rectangle begins to overlap its
    neighbours.

    That is what makes re-freezing every step affordable, and re-freezing every step is what
    makes a stale record impossible -- the fit never carries one forward.

    Three shapes, because two cannot show it: a pair of rectangles keeps the same plan at every
    width, since growing one never changes *which* shapes overlap. The transition needs a third
    for the middle one to reach.

    **Exactly 120 is its own record** since the interface set joined the freeze (2026-09-06,
    ``StaticGeometry.gamma_plans``). There the middle rectangle *touches* both neighbours --
    its faces lie on theirs -- and touching is not overlapping for the painting plan, but it
    is a combinatorial event for the interface set: two cross pairs appear, of different
    materials, so both seams are interfaces the field co-orients. So 118 and 120 are two
    records, 120 and 122 are two records, and the sweep recompiles twice around the contact
    rather than once. That is the contract, not a cost to trim: a record replayed across
    contact would cancel a seam or keep one the material runs through.

    The same record also holds each piece's list of periodic instances, bounded by the
    shape's *size* alone -- every lattice offset within ``diam + 2R`` of the fold -- so a
    length can grow across one of those thresholds too: the middle rectangle's list goes from
    9 to 13 instances near ``lx = 145`` (its bounding radius reaches 88, where the ``(2, 0)``
    family enters at 600 on this 300 cell). 140 and 122 are one record; 180 is a third, and a
    record frozen at 122 is refused at 180 rather than replayed with a list that could lack a
    nearest instance. Positions never enter the list, which is what a moving fit needs.
    """
    pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit.enable_x64()

    cfg = dict(period=[300.0, 300.0], thickness=[205.0], truncation_order=[3, 3], top=1.0,
               bot=1.0, theta=0.15, phi=0.35, psi=np.pi / 2, wavelength=900.0)
    make = lambda lx: [dict(background=12.25, shapes=[
        dict(type='rectangle', cx=-90.0, cy=0.0, lx=60.0, ly=100.0, angle=0.0, material=4.0),
        dict(type='rectangle', cx=0.0, cy=0.0, lx=lx, ly=100.0, angle=0.0, material=9.0),
        dict(type='rectangle', cx=90.0, cy=0.0, lx=60.0, ly=100.0, angle=0.0,
             material=6.25)])]

    def freeze_at(lx):
        return _legacy_wavebit.rcwa(backend='jax', structure=make(jnp.asarray(lx)), **cfg).freeze_geometry()

    keys = {lx: freeze_at(lx) for lx in (80.0, 100.0, 118.0, 120.0, 122.0, 140.0, 180.0)}

    # Same overlaps and no contact: a fresh freeze is the same static argument, hash included.
    for lx in (100.0, 118.0):
        assert keys[lx] == keys[80.0], lx
        assert hash(keys[lx]) == hash(keys[80.0]), lx

    # Contact: the interface set changes (two seams appear) while the painting plan does not.
    assert keys[120.0] != keys[80.0]
    assert keys[120.0].boundary_plans == keys[80.0].boundary_plans
    assert keys[120.0].gamma_plans != keys[80.0].gamma_plans
    assert sum(len(plan.pairs) for plan in keys[120.0].gamma_plans) == 2
    assert not any(plan.pairs for plan in keys[80.0].gamma_plans)

    # They do change eventually, and then the record must not be reused.
    assert keys[122.0] != keys[80.0] and keys[122.0] != keys[120.0]
    assert keys[140.0] == keys[122.0]
    # ... and once more when the middle rectangle's image list grows with its size.
    assert keys[180.0] != keys[122.0]
    assert keys[180.0].boundary_plans == keys[122.0].boundary_plans
    assert len(keys[180.0].gamma_plans[0].pieces[4].imgs) > len(keys[122.0].gamma_plans[0].pieces[4].imgs)

    # And the answer is the one for the structure actually in front of it, each time.
    def total(lx, geometry):
        return float(jnp.sum(_legacy_wavebit.rcwa(backend='jax', structure=make(lx), **cfg)
                             .solve(geometry=geometry).T).real)

    for lx, key in ((100.0, keys[80.0]), (140.0, keys[122.0])):
        assert total(jnp.asarray(lx), key) == total(lx, None)


PAINT_KW = dict(period=(1., 1.), thickness=(0.32,), wavelength=0.94, top=1., bot=2.1025,
                theta=0.2, phi=0.3, psi=0.7, truncation_order=(4, 4), shape_nvf_grid_size=256)


#: 3.4**2 and 2.1**2 -- the materials these two rectangles always had, as permittivity.
EPS_A, EPS_B = 11.559999999999999, 4.41


def _painted(cx, material=EPS_A):
    """Two painted rectangles; the first crosses the second as ``cx`` runs past ~0.02."""
    return [dict(background=1.0, shapes=[
        dict(type='rectangle', cx=cx, cy=0.0, lx=0.34, ly=0.26, angle=0.3, material=material),
        dict(type='rectangle', cx=0.05, cy=0.05, lx=0.30, ly=0.24, angle=0.0,
             material=EPS_B)])]


def _painted_t(cx, frozen=None, material=EPS_A):
    solver = _legacy_wavebit.rcwa(backend='numpy', structure=_painted(cx, material), **PAINT_KW)
    return np.asarray(solver.solve(geometry=frozen).T)


def test_a_stale_painting_plan_is_refused_rather_than_answered():
    """A painting plan survives a shape *moving*; it does not survive the overlap changing.

    This record is unlike the others here. It holds no coordinates at all -- only which loop
    holds which vertex and which edges cross -- and that is deliberate: it is what lets a replay
    carry a gradient to vertices that have moved, which is the whole reason the boundary route
    can be traced. So the usual reading of "stale" is wrong for it. Moving a shape is *exactly*
    what it is built to survive, and the first loop below pins that, bit-for-bit.

    What it cannot survive is the combinatorics changing underneath it. Slide the first
    rectangle far enough and it begins to overlap the second; the recorded loops then describe a
    painting nobody asked for. Before the check, the replay took it: bit-identical through
    ``cx = 0.00`` and then 3.9e-4, 2.3e-2, 3.3e-2 wrong at 0.05, 0.12, 0.25, with no step in
    between to notice. Re-planning to compare costs 0.2% of a frozen ``solve`` on this layer
    (1.2% on one of ten overlapping shapes, less as ``truncation_order`` rises) -- the freeze here buys
    traceability rather than a speed plateau, so that is not what is being traded.
    """
    frozen = _legacy_wavebit.rcwa(backend='numpy', structure=_painted(-0.10), **PAINT_KW).freeze_geometry()
    assert frozen.boundary_plans is not None, 'the plan was not recorded'

    for cx in (-0.10, -0.095, -0.08, -0.05, 0.00):  # moved, same overlap: accepted, and exact
        assert np.array_equal(_painted_t(cx, frozen), _painted_t(cx))

    for cx in (0.05, 0.12, 0.25):                  # the overlap changed: refused, not answered
        with pytest.raises(_legacy_wavebit.StaleGeometryError, match='no longer fits'):
            _painted_t(cx, frozen)

    for eps in (9.0, 4.0, 1.9599999999999997):     # material only: never moves a vertex
        assert np.array_equal(_painted_t(-0.10, frozen, eps),
                              _painted_t(-0.10, material=eps))


def test_the_painting_plan_check_is_skipped_under_a_trace_too():
    """Same documented limit as the raster records, reached a different way.

    A tracer has no coordinates to re-plan from, so there is nothing to compare and the replay
    goes through on trust -- including onto an overlap the plan does not describe. Pinned so it
    stays a decision: ``docs/guide/rcwa/boundary-modeling.md`` tells a fit to re-freeze every step, and
    this is why that instruction is not optional.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def total(cx, frozen):
        solver = _legacy_wavebit.rcwa(backend='jax', structure=_painted(cx), **PAINT_KW)
        return jnp.sum(solver.solve(geometry=frozen).T).real

    frozen = _legacy_wavebit.rcwa(backend='jax', structure=_painted(-0.10), **PAINT_KW).freeze_geometry()

    with pytest.raises(_legacy_wavebit.StaleGeometryError):        # eagerly, jax converts: checked
        total(0.25, frozen)

    jitted = jax.jit(total, static_argnums=1)              # traced: not checked, and finite
    assert np.isfinite(float(jitted(jnp.asarray(0.25), frozen)))
    assert np.isfinite(float(jax.grad(total)(jnp.asarray(-0.10), frozen)))


def _hidden(cx, material=4.0):
    """A small circle painted *under* a larger one — completely covered, so it has no loops."""
    return [dict(background=1.0, shapes=[
        dict(type='circle', cx=cx, cy=-0.07, lx=0.12, material=material),
        dict(type='circle', cx=-0.15, cy=-0.05, lx=0.25, material=9.0)])]


def test_a_shape_the_plan_recorded_as_covered_cannot_be_slid_out_from_under_it():
    """The sharpest reading of a stale plan: a shape the record says is not there.

    The first circle sits entirely inside the second (centres 0.036 apart, radii 0.12 and 0.25),
    and the second is painted over it, so the plan gives shape 0 **zero** loops. Slide it out and
    the replay keeps every loop it recorded, which is the same structure it always was: before
    the check it returned 0.929791164 for *every* position tried — -0.08, 0.05, 0.10, 0.20 —
    while the true answers ran 0.9295, 0.9536, 0.9483, 0.9382. The shape had simply left the
    structure.

    This case also pins *what* is compared. From ``cx = 0.05`` on, the crossing table is
    identical and only the loop assignment differs, so a check that compared the table alone
    would pass all three of those and answer with a circle that is not in the layer.
    """
    frozen = _legacy_wavebit.rcwa(backend='numpy', structure=_hidden(-0.12), **PAINT_KW).freeze_geometry()
    assert frozen.boundary_plans[0][1][0] == [], 'the covered shape was expected to have no loops'

    def total(cx, geometry=None, material=4.0):
        solver = _legacy_wavebit.rcwa(backend='numpy', structure=_hidden(cx, material), **PAINT_KW)
        return np.asarray(solver.solve(geometry=geometry).T)

    for cx in (-0.12, -0.10):                      # still covered: accepted, and exact
        assert np.array_equal(total(cx, frozen), total(cx))

    for eps in (6.760000000000001, 2.25):          # material only: never uncovers anything
        assert np.array_equal(total(-0.12, frozen, eps), total(-0.12, material=eps))

    for cx in (-0.08, 0.05, 0.10, 0.20):           # out from under it: refused, not answered
        with pytest.raises(_legacy_wavebit.StaleGeometryError, match='no longer fits'):
            total(cx, frozen)


def _disjoint(cx):
    """A rectangle and a circle that do not touch until ``cx`` is dragged down onto the circle."""
    return [dict(background=1.0, shapes=[
        dict(type='rectangle', cx=cx, cy=-0.093, lx=0.30, ly=0.24, angle=0.3, material=EPS_A),
        dict(type='circle', cx=-0.275, cy=-0.146, lx=0.18, material=EPS_B)])]


def test_a_frozen_no_overlap_is_a_claim_about_the_layer_and_is_checked_too():
    """``plan is None`` is one of the record's two answers, not the absence of one.

    A painted layer with nothing overlapping is summed, which is exact and cheaper, and the
    freeze records that decision as a ``None`` plan. It is a claim about the geometry, and it
    goes stale the same way a plan does — but for a long time it went stale silently in a
    different place: the dispatch took the summing branch on the record's word, and the summing
    builder refused with a ``ValueError`` naming *itself* and pointing at the boundary route the
    caller was already inside. Nothing in it mentioned the frozen record or said to re-freeze,
    and the same call without ``geometry=`` returns 0.933226529 quite happily.
    """
    frozen = _legacy_wavebit.rcwa(backend='numpy', structure=_disjoint(0.129), **PAINT_KW).freeze_geometry()
    assert frozen.boundary_plans == [None], 'the layer was expected to be recorded as disjoint'

    def total(cx, geometry=None):
        solver = _legacy_wavebit.rcwa(backend='numpy', structure=_disjoint(cx), **PAINT_KW)
        return np.asarray(solver.solve(geometry=geometry).T)

    for cx in (0.129, 0.10, 0.20):                 # still disjoint: accepted, and exact
        assert np.array_equal(total(cx, frozen), total(cx))

    with pytest.raises(_legacy_wavebit.StaleGeometryError, match='not overlapping'):
        total(-0.126, frozen)                      # dragged onto the circle
    assert np.isfinite(total(-0.126).sum()), 'the unfrozen call solves this layer fine'

    # And the re-asking has to stay eager. A tracer cannot answer the overlap question at all --
    # `layer_needs_resolving` raises rather than guess, which is the whole reason the record
    # carries the answer -- so a check that re-asked unconditionally would take the one path this
    # record exists to enable and break it, with an error reading "needs a frozen geometry" at a
    # call that was given one.
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def traced_total(cx):
        solver = _legacy_wavebit.rcwa(backend='jax', structure=_disjoint(cx), **PAINT_KW)
        return jnp.sum(solver.solve(geometry=frozen_jax).T).real

    frozen_jax = _legacy_wavebit.rcwa(backend='jax', structure=_disjoint(0.129), **PAINT_KW).freeze_geometry()
    assert frozen_jax.boundary_plans == [None]
    assert np.isfinite(float(jax.jit(traced_total)(jnp.asarray(0.129))))
    assert np.isfinite(float(jax.grad(traced_total)(jnp.asarray(0.129))))


# ------------------------------------------- the freeze is a check, not only a recording

_REFUSE_KW = dict(period=(300.0, 300.0), thickness=(205.0,), truncation_order=(6, 6),
                  top=1.0, bot=1.0, wavelength=900.0, psi=np.pi / 2, theta=0.0, phi=0.0)


def _oversized(lx=340.0):
    return [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0, lx=lx,
                                              ly=300.0, angle=0.0, material=4.0)])]


def test_freeze_geometry_runs_the_layer_refusals_it_is_named_for():
    """A record taken over a refused layer is a record that lets the refusal be replayed past.

    ``check_layer_shapes`` steps aside under a trace -- it compares numbers -- and both its
    docstring and the analytic module's say ``freeze_geometry`` is the eager call that still
    sees them. Until 2026-08-27 it did not make that call at all, and the gap was not academic:
    freeze a rod of ``lx = 340`` on a period of 300, hand the record to a traced solve, and it
    builds. ``T`` came back 0.9715 where the slab that rod actually describes gives 0.9590, and
    ``dT/dlx`` came back 9.8e-4 where the true derivative is **exactly zero** -- every ``lx``
    past the period is the same full slab. A phantom descent direction is worse than a wrong
    forward number, because an optimiser follows it.

    Pinned on the eager call, since that is the one that can raise. The traced replay is what
    it protects and cannot be asserted directly: with the freeze refusing, no record exists to
    hand it.
    """
    solver = _legacy_wavebit.rcwa(backend='numpy', structure=_oversized(), **_REFUSE_KW)
    with pytest.raises(ValueError, match='larger than the period'):
        solver.freeze_geometry()

    ok = _legacy_wavebit.rcwa(backend='numpy', structure=_oversized(lx=120.0), **_REFUSE_KW)
    assert ok.freeze_geometry().boundary_plans == [None]


def test_the_freeze_also_refuses_a_layer_key_nobody_reads():
    """Same hole, milder consequence, and the one a migrating recipe hits first.

    ``combine='paint'`` was a layer key until 2026-08-27. ``solve`` refuses it; the freeze used
    to accept it, so a recipe carrying it could take a record, then be refused by the solve it
    was taken for -- or, worse, not be refused at all on the paths that only read the record.
    """
    stale = [dict(background=1.0, combine='paint',
                  shapes=[dict(type='rectangle', cx=0.0, cy=0.0, lx=120.0, ly=120.0,
                               angle=0.0, material=4.0)])]
    solver = _legacy_wavebit.rcwa(backend='numpy', structure=stale, **_REFUSE_KW)
    with pytest.raises(ValueError, match="'combine'"):
        solver.freeze_geometry()


def test_the_renderer_refuses_what_the_solver_refuses():
    """A picture of a structure the solver will not build is the worst kind of documentation.

    ``slice_structure`` reaches the same layer dicts by a different door, and it is exactly
    where someone goes to find out what a refusal was about -- so it has to agree about which
    layers exist. Only the layer asked for is checked: drawing layer 0 must not be stopped by
    something wrong with layer 3.
    """
    solver = _legacy_wavebit.rcwa(backend='numpy', structure=_oversized(), **_REFUSE_KW)
    with pytest.raises(ValueError, match='larger than the period'):
        solver.slice_structure(n=32)

    good = dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0, lx=120.0,
                                             ly=120.0, angle=0.0, material=4.0)])
    mixed = _legacy_wavebit.rcwa(backend='numpy', structure=[good, _oversized()[0]],
                         **dict(_REFUSE_KW, thickness=(205.0, 205.0)))
    assert mixed.slice_structure(layer=0, n=32).real.mean() > 1.0
    with pytest.raises(ValueError, match='larger than the period'):
        mixed.slice_structure(layer=1, n=32)


# ------------------------------------------------------- a record belongs to its rule

_BOX = [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.1, cy=-0.05, lx=0.4, ly=0.3,
                                          material=6.25)])]


@pytest.mark.parametrize('coefficients', ['sampled', 'integrated'])
@pytest.mark.parametrize('held, now', [('li', 'nvf'), ('nvf', 'li'), ('laurent', 'li'),
                                       ('laurent', 'nvf'), ('li', 'laurent'),
                                       ('nvf', 'laurent')])
def test_a_raster_record_is_refused_under_another_rule(held, now, coefficients):
    """A record holds what its own rule decides: Li's integrated one the compression and no
    normal field, NVF's both. Replayed under the other rule, the build found its slot empty and
    rebuilt it eagerly -- or, under ``jax.jit``, had nothing to build from. It is stamped with
    its rule now, and a mismatch is refused by name, on the solve and on the per-call door."""
    record = _legacy_wavebit.rcwa(backend='numpy', structure=STRUCTURE, fourier_coefficients=coefficients,
                          factorization=held, **KW).freeze_geometry()
    assert record.factorization == held
    solver = _legacy_wavebit.rcwa(backend='numpy', structure=STRUCTURE, fourier_coefficients=coefficients,
                          factorization=now, **KW)
    with pytest.raises(_legacy_wavebit.StaleGeometryError, match=rf"taken under factorization='{held}' and is "
                                                 rf"being replayed under factorization='{now}'"):
        solver.solve(geometry=record)
    same = _legacy_wavebit.rcwa(backend='numpy', structure=STRUCTURE, fourier_coefficients=coefficients,
                        factorization=held, **KW)
    with pytest.raises(_legacy_wavebit.StaleGeometryError, match=f"replayed under factorization='{now}'"):
        same.eps_conv_matrices(geometry=record, factorization=now)


@pytest.mark.parametrize('held, now', [('li', 'nvf'), ('nvf', 'li'), ('laurent', 'li'),
                                       ('laurent', 'nvf'), ('li', 'laurent'),
                                       ('nvf', 'laurent')])
def test_a_shape_record_is_refused_under_another_rule(held, now):
    """The parametric path refused this already, by testing ``rectilinear_plans`` for ``None``;
    the stamp says it in one place for both paths and names both rules."""
    record = _legacy_wavebit.rcwa(backend='numpy', structure=_BOX, factorization=held,
                          **KW).freeze_geometry()
    with pytest.raises(_legacy_wavebit.StaleGeometryError, match=rf"taken under factorization='{held}'"):
        _legacy_wavebit.rcwa(backend='numpy', structure=_BOX, factorization=now, **KW).solve(geometry=record)


def test_the_stamp_is_checked_inside_a_trace_too():
    """A string, not a value: ``jax.jit`` cannot hide it, so a traced replay is refused as an
    eager one is -- which the slot tests the stamp replaces could not do."""
    jax = pytest.importorskip('jax')
    record = _legacy_wavebit.rcwa(backend='jax', structure=STRUCTURE, fourier_coefficients='integrated',
                          **KW).freeze_geometry()
    solver = _legacy_wavebit.rcwa(backend='jax', structure=STRUCTURE, fourier_coefficients='integrated',
                          factorization='nvf', **KW)
    with pytest.raises(_legacy_wavebit.StaleGeometryError, match="factorization='li'"):
        jax.jit(lambda s: s.solve(geometry=record).T)(solver)


def test_a_hand_built_record_carries_no_stamp_and_replays_on_trust():
    from wavebit.rcwa._engine.geometry import StaticGeometry
    solver = _legacy_wavebit.rcwa(backend='numpy', structure=STRUCTURE, **KW)
    assert StaticGeometry().factorization is None
    assert np.array_equal(np.asarray(solver.solve(geometry=StaticGeometry()).incident.T),
                          np.asarray(solver.solve().incident.T))
