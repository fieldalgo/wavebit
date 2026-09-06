"""The interface set as a frozen record: replayed under ``jit``, checked eagerly, keyed by content.

``freeze_geometry`` records each layer's :class:`~wavebit._core.parametric.GammaPlan` beside its
painting plan (``StaticGeometry.gamma_plans``), and the replay reads it instead of planning: a
jitted solve with a record builds the *planned* field -- pairs, windows, ties -- where one
without a record gets the blind plan (every piece, no pairs). The record holds no coordinate
and no material, so a shape moving or a material sweeping never invalidates it; what does is
the combinatorics changing, which ``check_gamma_plan_still_fits`` re-derives eagerly and
refuses by name. The projector cache keys on the plan's structure and, for a layer with a
cross pair, on the concrete equality bits of its materials, so a material sweep on a pair-free
layer still hits and one that crosses equality on a seam rebuilds.

Every jitted solve here builds its solver **outside** the trace and hands ``structure=`` and
``geometry=`` to ``solve``: a solver built inside the traced function never crosses the jit
boundary, and a test written that way passes without exercising it.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core import parametric as P
from wavebit._core.geometry import StaleGeometryError, StaticGeometry
from wavebit._core.lattice import as_lattice

PERIOD = 700.0
SQUARE = np.array([[PERIOD, 0.0], [0.0, PERIOD]])
SHEAR = np.array([[PERIOD, 200.0], [0.0, PERIOD]])

KW = dict(wavelength=900.0, top=1.0, bot=2.25, theta=0.2, phi=0.0, psi=0.0,
          thickness=[300.0], truncation_order=dict(u=4, v=4), normal_grid=128)


def _box(**geom):
    return dict(dict(type='rectangle', cx=0.0, cy=0.0, angle=0.0, material=4.0), **geom)


def _b_layer(lx):
    """B: a box spanning the sheared cell; its two v faces meet their images over ``lx - 200``."""
    return [dict(background=1.0, shapes=[_box(lx=lx, ly=PERIOD, material=12.25)])]


def _k_layer(left=4.0, right=9.0):
    """K: two 300 x 300 boxes abutting inside the cell, materials ``left | right``."""
    return [dict(background=1.0, shapes=[_box(cx=-150.0, lx=300.0, ly=300.0, material=left),
                                         _box(cx=150.0, lx=300.0, ly=300.0, material=right)])]


def _d_layer():
    return [dict(background=1.0, shapes=[_box(lx=350.0, ly=350.0, cy=-175.0),
                                         _box(lx=350.0, ly=350.0, cy=175.0)])]


def _total(structure, lattice, backend=0, geometry=None, solver=None):
    P._PROJECTOR_CACHE.clear()
    if solver is None:
        solver = wavebit.rcwa(backend=backend, lattice=lattice, structure=structure, **KW)
    return float(np.sum(np.asarray(solver.solve(structure=structure, geometry=geometry).T)))


# ------------------------------------------------------------------------ the record

def test_the_freeze_records_a_gamma_plan_for_every_layer():
    """Beside the painting plan, for every layer, whether or not the closed form builds it."""
    solver = wavebit.rcwa(backend=0, lattice=SHEAR, structure=_b_layer(500.0), **KW)
    frozen = solver.freeze_geometry()
    assert frozen.gamma_plans == [P.plan_gamma(_b_layer(500.0)[0]['shapes'],
                                               as_lattice(lattice=SHEAR))]
    assert frozen.boundary_plans == [None]
    assert frozen.gamma_plans[0].pairs                      # B has its run


def test_the_record_hashes_by_content_with_its_gamma_plans():
    """Two freezes of one combinatorics are one jit key; a different plan is another.

    ``StaticGeometry`` compares by content while no array-bearing slot is filled, and the
    gamma plans are pure Python -- so the record a moving fit re-takes every step hits the
    compilation cache until a pair appears or vanishes.
    """
    def freeze(structure, lattice=SHEAR):
        return wavebit.rcwa(backend=0, lattice=lattice, structure=structure, **KW).freeze_geometry()

    assert freeze(_b_layer(500.0)) == freeze(_b_layer(450.0))          # same pair, moved
    assert hash(freeze(_b_layer(500.0))) == hash(freeze(_b_layer(450.0)))
    assert freeze(_b_layer(500.0)) != freeze(_b_layer(150.0))          # the pair is gone
    assert freeze(_k_layer(4.0, 9.0)) == freeze(_k_layer(9.0, 9.0))    # materials never enter
    assert StaticGeometry(gamma_plans=[None]) != freeze(_b_layer(500.0))


# ------------------------------------------------------------------- replay under jit

@pytest.mark.parametrize('name, lattice, make, start', [
    ('B', SHEAR, _b_layer, 500.0),
    ('K', SQUARE, lambda lx: _k_layer(4.0, 9.0) if lx is None else
     [dict(_k_layer(4.0, 9.0)[0], shapes=[dict(_k_layer()[0]['shapes'][0], lx=lx),
                                          dict(_k_layer()[0]['shapes'][1], cx=lx / 2.0)])],
     300.0),
])
def test_a_jitted_solve_with_a_record_equals_the_eager_one(name, lattice, make, start):
    """B and K under ``jax.jit`` with a frozen record, the solver built outside the trace.

    The planned field -- the pair's exact cancellation on B, the seam's sign on K -- replays
    from the record inside the trace, where nothing can be read off a tracer; ``jax.grad``
    reads the primal and plans per call, and the two agree with the eager solve to 1e-12.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    x = jnp.asarray(start)
    solver = wavebit.rcwa(backend=1, lattice=lattice, structure=make(start), **KW)
    frozen = solver.freeze_geometry()
    assert frozen.gamma_plans[0].pairs, name

    def total(v):
        return jnp.sum(solver.solve(structure=make(v), geometry=frozen).T).real

    eager = float(total(x))
    assert float(jax.jit(total)(x)) == pytest.approx(eager, rel=1e-12, abs=1e-12), name
    grad = float(jax.grad(total)(x))
    assert np.isfinite(grad) and grad != 0.0, name
    assert float(jax.jit(jax.grad(total))(x)) == pytest.approx(grad, rel=1e-9), name
    P._PROJECTOR_CACHE.clear()
    plain = float(np.sum(np.asarray(wavebit.rcwa(backend=0, lattice=lattice,
                                                 structure=make(start), **KW).solve().T)))
    assert eager == pytest.approx(plain, abs=1e-12), name


@pytest.mark.parametrize('name, lattice', [
    ('shear', SHEAR),
    # `a2 = 10 a1 + (50, 700)`: the cell of `a2 = (50, 700)` on an unreduced basis, which
    # `as_lattice` accepts as it is. The planned list per piece is 479 offsets out to ring 14.
    ('unreduced, 10 a1', np.array([[PERIOD, 7050.0], [0.0, PERIOD]])),
])
def test_a_traced_basis_with_no_record_is_refused_and_with_one_replays(name, lattice):
    """``jax.jit`` over the lattice, lone shape, no record: refused; with a record: the eager answer.

    The blind plan used to answer a traced basis with a fixed 7x7 ring of images and no
    argument that the ring held every instance the planned list holds. On the unreduced
    basis here it did not -- 460 of the 479 planned offsets lay outside it -- and the jitted
    solve came back 6e-3 in ``sum(T)`` from the eager one, unit-length and energy-conserving.
    Now the case raises with the message every traced decision raises with, naming the
    freeze; the record taken outside the trace replays the planned list under it, to 1e-12
    of the eager solve. ``jax.grad`` needs no record: its tracer carries the primal and the
    basis is planned from it per call (``d/d(a2x)`` finite, and the AD-vs-FD row of the
    lattice gradient lives in ``test_lattice_pytree.py``).

    The freeze is taken *before* the refused call on purpose: ``solve(lattice=...)`` assigns
    the design variable to the solver, so a jitted call leaves a tracer in ``solver.lattice``
    and a freeze taken after it would record ``None`` for the plan.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    structure = [dict(background=1.0, shapes=[_box(lx=500.0, ly=PERIOD, material=12.25)])]
    solver = wavebit.rcwa(backend=1, lattice=lattice, structure=structure, **KW)
    frozen = solver.freeze_geometry()
    assert frozen.gamma_plans[0].pairs, name

    def total(L, geometry=None):
        return jnp.sum(solver.solve(lattice=L, geometry=geometry).T).real

    with pytest.raises(ValueError, match='freeze_geometry'):
        jax.jit(total)(jnp.asarray(lattice))
    eager = float(total(jnp.asarray(lattice)))
    replayed = float(jax.jit(lambda L: total(L, frozen))(jnp.asarray(lattice)))
    assert replayed == pytest.approx(eager, abs=1e-12), name
    P._PROJECTOR_CACHE.clear()
    plain = float(np.sum(np.asarray(wavebit.rcwa(backend=0, lattice=lattice, structure=structure,
                                                 **KW).solve().T)))
    assert eager == pytest.approx(plain, abs=1e-12), name
    assert np.isfinite(np.asarray(jax.grad(total)(jnp.asarray(lattice)))).all(), name


def test_plan_gamma_runs_once_at_the_freeze_and_never_in_the_trace(monkeypatch):
    """A call counter on the planner: the freeze plans, the trace replays.

    Eagerly with a record the staleness check re-plans once -- that is the check. Inside
    ``jax.jit`` there is no primal, the concreteness probe answers first, and the planner is
    not called at all: a plan taken from a tracer would be the blind one, silently.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    calls = []
    real = P.plan_gamma

    def counting(shapes, lattice):
        calls.append(len(shapes))
        return real(shapes, lattice)

    monkeypatch.setattr(P, 'plan_gamma', counting)
    solver = wavebit.rcwa(backend=1, lattice=SHEAR, structure=_b_layer(500.0), **KW)
    frozen = solver.freeze_geometry()
    assert calls == [1]

    def total(v):
        return jnp.sum(solver.solve(structure=_b_layer(v), geometry=frozen).T).real

    float(jax.jit(total)(jnp.asarray(500.0)))
    assert calls == [1], 'the trace planned'
    float(total(jnp.asarray(500.0)))
    assert calls == [1, 1], 'the eager replay re-derives the plan once, for the check'


def test_the_gradient_across_a_coverage_breakpoint_is_finite():
    """``d/dlx`` at the onset of B's run, ``lx = 200`` on the sheared cell, is finite and right.

    Below 200 the two v faces slide past each other and no pair exists; at 200 the run has
    zero length; above it the plan carries the pair. Under ``jax.grad`` the plan is taken from
    the primal per call, so each value gets the plan of its own geometry, and on either side
    autograd is the central difference to 1e-6 (``h = 1e-5``, ``normal_grid`` 128):

        lx      AD              FD h=1e-3 (rel)   h=1e-5 (rel)
        199.9   8.184202e-03    +4.2e-9           +2.6e-7
        200.1   8.252927e-03    +2.4e-8           +4.6e-7
        350.0   8.518871e-04    -7.1e-7           -8.1e-7

    The 350 row was 5.2e-3 off at every ``h`` until the weight of a piece *tied* with the
    nearest one kept its derivative (see the weights in ``_piece_field``); that was a wrong
    derivative, constant in ``h``, and the fix moved no value.

    **What the onset itself is: a kink, not a step.** The field is continuous through the
    contact (the ladder in ``test_gamma_field.py``), but its extension near the two face ends
    changes character there -- below the onset the gap between the ends reads the faces'
    residual y-normal, above it the run cancels exactly and reads the u faces -- and the
    one-sided slopes of ``sum(T)`` differ by a relative +4.2e-4 / -1.1e-3 / -1.6e-3 / -1.3e-3
    / -1.5e-3 at orders 4 / 6 / 8 / 10 / 12 (secants of 1e-2 either side). That does not
    decay with order, so by this project's rule it is a limit of the fixed-normal extension
    at a run's end rather than a residual; it is in the slope, an optimiser crossing contact
    sees no cliff, and it is recorded in ``docs/normal-field-known-limits.md`` (case J).
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    def total(v):
        P._PROJECTOR_CACHE.clear()
        solver = wavebit.rcwa(backend=1, lattice=SHEAR, structure=_b_layer(v), **KW)
        return jnp.sum(solver.solve().T).real

    for lx in (199.9, 200.0, 200.1, 350.0):
        assert np.isfinite(float(jax.grad(total)(jnp.asarray(lx)))), lx
    h = 1e-5
    for lx in (200.1, 350.0):
        fd = (float(total(jnp.asarray(lx + h))) - float(total(jnp.asarray(lx - h)))) / (2 * h)
        assert float(jax.grad(total)(jnp.asarray(lx))) == pytest.approx(fd, rel=1e-5), lx


# ------------------------------------------------------------------- what the record survives

def test_a_frozen_material_sweep_is_accepted_and_the_seam_sign_follows():
    """The record holds no material, so a sweep through equality replays -- and flips the sign.

    Frozen at ``4 | 9`` and replayed at ``9 | 9`` the seam pair cancels, live: the answer is
    the eager same-material solve's to 1e-12, eagerly and under ``jax.jit`` with the
    materials traced. Nothing was invalidated and nothing had to be.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    solver = wavebit.rcwa(backend=1, lattice=SQUARE, structure=_k_layer(4.0, 9.0), **KW)
    frozen = solver.freeze_geometry()
    same = _total(_k_layer(9.0, 9.0), SQUARE)                          # eager, planned afresh
    P._PROJECTOR_CACHE.clear()
    replayed = float(np.sum(np.asarray(solver.solve(structure=_k_layer(9.0, 9.0),
                                                    geometry=frozen).T)))
    assert replayed == pytest.approx(same, abs=1e-12)

    def total(left):
        return jnp.sum(solver.solve(structure=_k_layer(left, 9.0), geometry=frozen).T).real

    assert float(jax.jit(total)(jnp.asarray(9.0))) == pytest.approx(same, abs=1e-12)
    differ = _total(_k_layer(4.0, 9.0), SQUARE)
    assert float(jax.jit(total)(jnp.asarray(4.0))) == pytest.approx(differ, abs=1e-12)
    assert abs(differ - same) > 1e-4                                      # the sign did something


def test_a_pair_that_appears_or_vanishes_is_refused_by_name():
    """B frozen with its run and replayed short of the onset, and the reverse: refused, eagerly.

    A shape moving never trips this; a face leaving its partner's line does, because the
    recorded pair would then cancel two faces that are interfaces (or fail to cancel two that
    are not). The message names the pair. Under a trace nothing checks, as for every record.
    """
    def frozen_at(lx):
        return wavebit.rcwa(backend=0, lattice=SHEAR, structure=_b_layer(lx), **KW).freeze_geometry()

    with_run = frozen_at(500.0)
    assert _total(_b_layer(450.0), SHEAR, geometry=with_run) == pytest.approx(
        _total(_b_layer(450.0), SHEAR), abs=1e-13)                     # still a pair: accepted
    with pytest.raises(StaleGeometryError, match='pair'):
        _total(_b_layer(150.0), SHEAR, geometry=with_run)
    without = frozen_at(150.0)
    with pytest.raises(StaleGeometryError, match='pair'):
        _total(_b_layer(500.0), SHEAR, geometry=without)
    # a translation is what the record is built to survive
    moved = [dict(_b_layer(500.0)[0], shapes=[dict(_b_layer(500.0)[0]['shapes'][0], cx=123.0)])]
    assert _total(moved, SHEAR, geometry=with_run) == pytest.approx(_total(moved, SHEAR), abs=1e-13)


def test_a_face_that_becomes_dead_or_a_list_that_grows_is_refused_too():
    """The record's other content: a face's coverage and a piece's image list.

    The image lists are bounded by size alone, so a shape growing across a ring boundary
    changes them and is refused (positions never do); the 1-D grating's dead faces flip to
    live one unit short of the period and are refused with them.
    """
    grating = [dict(background=1.0, shapes=[_box(lx=350.0, ly=PERIOD)])]
    frozen = wavebit.rcwa(backend=0, lattice=SQUARE, structure=grating, **KW).freeze_geometry()
    short = [dict(background=1.0, shapes=[_box(lx=350.0, ly=PERIOD - 1.0)])]
    with pytest.raises(StaleGeometryError, match='no longer fits'):
        _total(short, SQUARE, geometry=frozen)
    small = [dict(background=1.0, shapes=[_box(lx=100.0, ly=100.0)])]
    frozen_small = wavebit.rcwa(backend=0, lattice=SQUARE, structure=small, **KW).freeze_geometry()
    big = [dict(background=1.0, shapes=[_box(lx=600.0, ly=600.0)])]
    with pytest.raises(StaleGeometryError, match='image'):
        _total(big, SQUARE, geometry=frozen_small)


# ------------------------------------------------------------------------------ the cache

def test_the_projector_cache_reads_the_equality_bits_of_a_cross_pair():
    """A material sweep rebuilds a layer with a seam only when it crosses equality.

    The key holds the plan's structure and, for a cross pair, whether its two materials are
    equal -- never the values. So ``4 | 9``, ``4 | 9.5`` and ``4 | 12`` are one entry, ``4 | 4``
    a second; a pair-free layer swept the same way stays at one, as it always did.
    """
    P._PROJECTOR_CACHE.clear()
    for right in (9.0, 9.5, 12.0, 4.0, 4.0, 9.0):
        wavebit.rcwa(backend=0, lattice=SQUARE, structure=_k_layer(4.0, right), **KW).solve()
    assert len(P._PROJECTOR_CACHE) == 2

    P._PROJECTOR_CACHE.clear()
    apart = lambda right: [dict(background=1.0, shapes=[
        _box(cx=-200.0, lx=200.0, ly=300.0, material=4.0),
        _box(cx=200.0, lx=200.0, ly=300.0, material=right)])]
    for right in (9.0, 9.5, 4.0, 12.0):
        wavebit.rcwa(backend=0, lattice=SQUARE, structure=apart(right), **KW).solve()
    assert len(P._PROJECTOR_CACHE) == 1


def test_a_traced_material_on_a_seam_is_never_served_from_the_cache():
    """The equality bit of a tracer is unknown, so the key is ``None`` and the field is rebuilt."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    P._PROJECTOR_CACHE.clear()
    wavebit.rcwa(backend=1, lattice=SQUARE, structure=_k_layer(4.0, 9.0), **KW).solve()
    assert len(P._PROJECTOR_CACHE) == 1

    def total(right):
        return jnp.sum(wavebit.rcwa(backend=1, lattice=SQUARE, structure=_k_layer(4.0, right),
                                    **KW).solve().T).real

    g = float(jax.grad(total)(jnp.asarray(9.0)))
    assert np.isfinite(g) and len(P._PROJECTOR_CACHE) == 1     # built, not cached


# ---------------------------------------------------------------------------- backends

@pytest.mark.parametrize('name, lattice, structure', [
    ('B', SHEAR, _b_layer(500.0)),
    ('D', SQUARE, _d_layer()),
    ('K', SQUARE, _k_layer(4.0, 9.0)),
])
def test_every_backend_builds_the_same_field(name, lattice, structure):
    """NumPy, JAX and Torch through three namespaces, one plan, one answer to 1e-12."""
    totals = [_total(structure, lattice, backend=0)]
    for backend in (1, 2):
        pytest.importorskip({1: 'jax', 2: 'torch'}[backend])
        if backend == 1:
            wavebit.enable_x64()
        totals.append(_total(structure, lattice, backend=backend))
    assert max(abs(t - totals[0]) for t in totals) < 1e-12, (name, totals)
