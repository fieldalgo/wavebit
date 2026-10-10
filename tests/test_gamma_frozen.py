"""The interface set as a frozen record: replayed under ``jit``, checked eagerly, keyed by content.

``freeze_geometry`` records each layer's :class:`~wavebit._core.parametric.GammaPlan` beside its
painting plan (``StaticGeometry.gamma_plans``), and the replay reads it instead of planning: a
jitted solve with a record builds the *planned* field -- pairs, windows, ties -- where one
without a record gets the blind plan (every piece, no pairs). The record holds no coordinate
and no material, so a shape moving or a material sweeping never invalidates it; what does is
the combinatorics changing, which ``check_gamma_plan_still_fits`` re-derives eagerly and
refuses by name. The projector cache keys on the plan's structure and, for a layer of several
shapes, on every shape's concrete material -- the seam contrast reads the values, since
2026-09-11 -- so a lone shape's material sweep still hits and a multi-shape sweep rebuilds per
value.

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
          thickness=[300.0], truncation_order=dict(u=4, v=4), shape_nvf_grid_size=128)


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


def _total(structure, lattice, backend='numpy', geometry=None, solver=None):
    P._PROJECTOR_CACHE.clear()
    if solver is None:
        solver = wavebit.rcwa(backend=backend, lattice=lattice, structure=structure, **KW)
    return float(np.sum(np.asarray(solver.solve(structure=structure, geometry=geometry).T)))


# ------------------------------------------------------------------------ the record

def test_the_freeze_records_a_gamma_plan_for_every_layer():
    """Beside the painting plan, for every layer, whether or not the closed form builds it."""
    solver = wavebit.rcwa(backend='numpy', lattice=SHEAR, structure=_b_layer(500.0), **KW)
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
        return wavebit.rcwa(backend='numpy', lattice=lattice, structure=structure, **KW).freeze_geometry()

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

    The planned field -- the pair's exact cancellation on B, the seam's contrast on K -- replays
    from the record inside the trace, where nothing can be read off a tracer; ``jax.grad``
    reads the primal and plans per call, and the two agree with the eager solve to 1e-12.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    x = jnp.asarray(start)
    solver = wavebit.rcwa(backend='jax', lattice=lattice, structure=make(start), **KW)
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
    plain = float(np.sum(np.asarray(wavebit.rcwa(backend='numpy', lattice=lattice,
                                                 structure=make(start), **KW).solve().T)))
    assert eager == pytest.approx(plain, abs=1e-12), name


@pytest.mark.parametrize('name, lattice', [
    ('shear', SHEAR),
    # `a2 = 10 a1 + (50, 700)`: the cell of `a2 = (50, 700)` on an unreduced basis, which
    # `as_lattice` accepts as it is. It planned 479 offsets per piece out to ring 14 on the
    # basis as given; since 2026-09-06 it plans as its reduction, 21 offsets on ring 3.
    ('unreduced, 10 a1', np.array([[PERIOD, 7050.0], [0.0, PERIOD]])),
])
def test_a_traced_basis_with_no_record_is_refused_and_with_one_replays(name, lattice):
    """``jax.jit`` over the lattice, lone shape, no record: refused; with a record: the eager answer.

    The blind plan used to answer a traced basis with a fixed 7x7 ring of images and no
    argument that the ring held every instance the planned list holds. On the unreduced
    basis here it did not -- 460 of the 479 offsets it then planned lay outside it -- and the
    jitted solve came back 6e-3 in ``sum(T)`` from the eager one, unit-length and
    energy-conserving. Now the case raises with the message every traced decision raises
    with, naming the freeze (a traced basis cannot be reduced either); the record taken
    outside the trace replays the planned list under it, to 1e-12 of the eager solve --
    reducing the traced basis with the record's integers, since the offsets count cells of
    the reduced spelling. ``jax.grad`` needs no record: its tracer carries the primal and the
    basis is planned from it per call (``d/d(a2x)`` finite, and the AD-vs-FD row of the
    lattice gradient lives in ``test_lattice_pytree.py``).

    The freeze is taken before the refused call here; it may be taken after it too, since
    2026-09-06 -- ``solve()`` puts the concrete cell back once a traced call returns, which
    is what ``test_the_refusal_recipe_works_in_the_order_it_is_read`` pins. Until then the
    jitted call left its tracer in ``solver.lattice`` and a freeze taken after it recorded
    ``None`` for the plan.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    structure = [dict(background=1.0, shapes=[_box(lx=500.0, ly=PERIOD, material=12.25)])]
    solver = wavebit.rcwa(backend='jax', lattice=lattice, structure=structure, **KW)
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
    plain = float(np.sum(np.asarray(wavebit.rcwa(backend='numpy', lattice=lattice, structure=structure,
                                                 **KW).solve().T)))
    assert eager == pytest.approx(plain, abs=1e-12), name
    assert np.isfinite(np.asarray(jax.grad(total)(jnp.asarray(lattice)))).all(), name


@pytest.mark.parametrize('name, lattice, structure, traced', [
    ('shear 500 x 700 box, lattice traced', SHEAR, _b_layer(500.0), 'lattice'),
    ('lone 350 box, lattice traced', SQUARE,
     [dict(background=1.0, shapes=[_box(lx=350.0, ly=350.0)])], 'lattice'),
    ('720 ellipse, structure traced', SQUARE, None, 'structure'),
])
def test_the_refusal_recipe_works_in_the_order_it_is_read(name, lattice, structure, traced):
    """Jit refused -> ``freeze_geometry()`` -> jitted replay -> eager replay, on one solver.

    The recipe the traced-decision refusal prints, followed literally *after* the refusal.
    It did not work until 2026-09-06: ``solve(lattice=L)`` assigned the tracer to the solver
    and left it there, so the freeze taken next recorded ``gamma_plans = [None]`` in silence,
    the jitted replay with that record was refused again with the same message, and every
    eager solve on the solver raised ``UnexpectedTracerError`` until the lattice was
    reassigned by hand -- measured on exactly these two boxes. Now ``solve()`` puts the
    concrete cell (and structure) back once the traced call returns, so the solver the freeze
    sees is the one that was built: the record carries the plan, the replay equals the eager
    solve to 1e-12, and the eager solves after it are the fresh solve's to the bit.

    The ellipse row is the same story through ``structure=``: the blind plan refuses a traced
    ellipse by name, and a freeze taken off the leaked structure would have recorded nothing.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    if structure is None:
        structure = _l_ellipse_layer(720.0)
    solver = wavebit.rcwa(backend='jax', lattice=lattice, structure=structure, **KW)
    before = np.asarray(solver.lattice)
    reference = float(jnp.sum(solver.solve().T).real)

    if traced == 'lattice':
        def total(L, geometry=None):
            return jnp.sum(solver.solve(lattice=L, geometry=geometry).T).real
        x = jnp.asarray(lattice)
    else:
        def total(lx, geometry=None):
            return jnp.sum(solver.solve(structure=_l_ellipse_layer(lx), geometry=geometry).T).real
        x = jnp.asarray(720.0)

    with pytest.raises(ValueError, match='freeze_geometry'):           # step 1
        jax.jit(total)(x)
    assert np.array_equal(np.asarray(solver.lattice), before), name    # nothing traced stayed
    frozen = solver.freeze_geometry()                                  # step 2
    assert frozen.gamma_plans[0] is not None, name
    if 'box' in name and 'shear' in name:
        assert frozen.gamma_plans[0].pairs
    replayed = float(jax.jit(lambda v: total(v, frozen))(x))          # step 3
    assert replayed == pytest.approx(reference, abs=1e-12), name
    assert float(jnp.sum(solver.solve(geometry=frozen).T).real) == reference, name   # 3b
    assert float(jnp.sum(solver.solve().T).real) == reference, name                  # step 4
    assert np.isfinite(np.asarray(jax.grad(total)(x))).all(), name
    assert np.array_equal(np.asarray(solver.lattice), before), name


@pytest.mark.parametrize('name, value', [
    ('wavelength', 900.0), ('thickness', [300.0]), ('theta', 0.2)])
def test_a_traced_scalar_condition_does_not_outlive_its_solve(name, value):
    """A jitted sweep over a wavelength, a thickness or an angle leaves nothing traced behind.

    The first restore (2026-09-06) put back the cell and the structure slots by name and left
    the scalars, on the argument that no eager decision reads one before the next keyword
    overwrites it. The eager ``solve()`` is that decision: after ``jax.jit(lambda w:
    solver.solve(geometry=g, wavelength=w))(900.0)`` the plain ``solver.solve()`` raised
    ``UnexpectedTracerError`` from the tracer left in ``solver.wavelength`` -- the failure
    class the recipe test above names, on a different slot -- and ``thickness`` and
    ``theta`` did the same. ``_conditions`` now restores every slot a keyword's setter left
    holding a tracer, read off the instance dictionary rather than a list of names. So the
    slot reads its number after the jitted call, the eager solve is the fresh solve's to the
    bit, and a freeze taken afterwards records a plan. The second half is the value that
    arrives *concrete* and is stored traced: ``thickness=[310.0]`` under the trace is staged
    by the JAX setter's ``as_array`` into a tracer, and is put back as well.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    structure = [dict(background=1.0, shapes=[_box(lx=350.0, ly=350.0, material=12.25)])]
    solver = wavebit.rcwa(backend='jax', lattice=SQUARE, structure=structure, **KW)
    reference = float(jnp.sum(solver.solve().T).real)
    frozen = solver.freeze_geometry()

    def total(v):
        return jnp.sum(solver.solve(geometry=frozen, **{name: v}).T).real

    replayed = float(jax.jit(total)(jnp.asarray(value)))
    assert replayed == pytest.approx(reference, abs=1e-12)
    assert np.allclose(np.asarray(getattr(solver, name)), np.asarray(value))   # a number, not a tracer
    assert float(jnp.sum(solver.solve().T).real) == reference
    assert solver.freeze_geometry().gamma_plans[0] is not None
    assert np.isfinite(np.asarray(jax.grad(total)(jnp.asarray(value, dtype=float)))).all()
    assert float(jnp.sum(solver.solve().T).real) == reference

    def mixed(w):
        return jnp.sum(solver.solve(geometry=frozen, wavelength=w, thickness=[310.0]).T).real

    jax.jit(mixed)(jnp.asarray(900.0))
    assert np.allclose(np.asarray(solver.thickness), [300.0])


@pytest.mark.parametrize('name, value', [('wavelength', 900.0), ('thickness', [300.0]),
                                         ('theta', 0.2)])
def test_a_jitted_first_solve_leaves_no_tracer_in_the_projector_cache(name, value):
    """The cold order: no eager solve before the jit, so the projector key is a miss in the trace.

    The test above warms the key at its first line, and that warm order is what hid this: a
    miss *inside* a jit trace builds the coefficient grids under omnistaging -- tracers, from
    concrete inputs -- and stored them under the concrete key. The next eager ``solve()`` on
    that key, on any solver in the process, popped them, put them back, and raised
    ``UnexpectedTracerError`` at ``c * phase``; it did not heal until the cache was cleared by
    hand or evicted by eight other keys. So: clear the cache, build, freeze, jit as the FIRST
    solve, then an eager ``solve()`` must equal a fresh solver's value, and no entry in
    ``_PROJECTOR_CACHE`` may hold a tracer.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from wavebit._core import parametric as P
    wavebit.enable_x64()

    structure = [dict(background=1.0, shapes=[_box(lx=350.0, ly=350.0, material=12.25)])]
    fresh = wavebit.rcwa(backend='jax', lattice=SQUARE, structure=structure, **KW)
    reference = float(jnp.sum(fresh.solve().T).real)

    P._PROJECTOR_CACHE.clear()
    solver = wavebit.rcwa(backend='jax', lattice=SQUARE, structure=structure, **KW)
    frozen = solver.freeze_geometry()                      # no eager solve: the key is cold

    def total(v):
        return jnp.sum(solver.solve(geometry=frozen, **{name: v}).T).real

    assert float(jax.jit(total)(jnp.asarray(value))) == pytest.approx(reference, abs=1e-12)
    assert not any(isinstance(c, jax.core.Tracer)
                   for coeffs in P._PROJECTOR_CACHE.values() for c in coeffs)
    assert float(jnp.sum(solver.solve().T).real) == reference     # raised before the guard
    assert float(jnp.sum(fresh.solve().T).real) == reference      # ... on every solver
    assert float(jnp.sum(solver.solve().T).real) == reference


#: A two-row raster on the square cell, for the companion-slot cases below.
COMPANION_RASTER = np.array([[[1.0, 4.0, 4.0, 1.0], [4.0, 1.0, 1.0, 4.0]]])


#: What a solve writes on the solver, and so what a traced call may leave changed: its state.
SOLVE_STATE = frozenset({'_formulation_assigned', '_frozen_geometry', '_grating_type',
                         '_solved_scene', 'layer_info_list', 'T1'})


def _companion_raster_solver(**over):
    kw = dict(backend='jax', structure=COMPANION_RASTER, lattice=SQUARE, truncation_order=[1, 1],
              thickness=[300.0], wavelength=900.0, top=1.0, bot=1.0)
    kw.update(over)
    return wavebit.rcwa(**kw)


def test_a_traced_setter_is_put_back_whole_with_the_slots_it_wrote_beside_the_tracer():
    """Three setters write a concrete companion next to the value, and it goes back with it.

    Measured before 2026-09-24, when only the tracer-holding slots were put back:

    * ``psi`` latches ``_psi_given``. After a jitted ``solve(psi=p)`` the angle read its default
      while the flag said it was given, and ``solver.polarization = (1, 1j)`` was refused as
      naming the incidence twice.
    * ``structure`` selects ``_modeling_path``. A ``jax.grad`` of ``solve(structure=shapes)``
      on a raster solver put the raster back under the parametric path, and every eager
      ``solve()`` after it raised ``TypeError``.
    * ``structure`` clears ``_structure_drawn_on`` for a bare raster. A traced raster through a
      solver whose structure came from ``Lattice.drawn`` came back undeclared, so a moved cell
      would be answered where it is refused.

    The traced cases compare the whole instance dictionary, less the slots a solve writes,
    since a companion slot is exactly what a list of names would miss.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    absent = object()

    def moved_slots(solver, before):
        now = vars(solver)
        return sorted(k for k in before.keys() | now.keys() if k not in SOLVE_STATE
                      and now.get(k, absent) is not before.get(k, absent))

    solver = _companion_raster_solver()
    frozen = solver.freeze_geometry()
    before = dict(vars(solver))
    jax.jit(lambda p: jnp.sum(solver.solve(geometry=frozen, psi=p).T).real)(0.3)
    assert moved_slots(solver, before) == []
    solver.polarization = (1, 1j)                        # refused while the flag was latched

    solver = _companion_raster_solver()
    reference = float(jnp.sum(solver.solve().T).real)

    def total(lx):
        shapes = [dict(background=1.0, shapes=[_box(lx=lx, ly=300.0)])]
        return jnp.sum(solver.solve(structure=shapes).T).real

    assert np.isfinite(float(jax.grad(total)(300.0)))
    assert solver.modeling_path == _companion_raster_solver().modeling_path
    assert float(jnp.sum(solver.solve().T).real) == reference

    cell = wavebit.Lattice.rectangular(PERIOD, PERIOD)
    solver = _companion_raster_solver(structure=cell.drawn(COMPANION_RASTER))
    frozen = solver.freeze_geometry()
    before = dict(vars(solver))
    jax.jit(lambda r: jnp.sum(solver.solve(geometry=frozen, structure=r).T).real)(
        jnp.asarray(COMPANION_RASTER))
    assert moved_slots(solver, before) == []
    assert solver.structure_drawn_on is not None


@pytest.mark.parametrize('side, value', [('top', 1.0), ('bot', 2.25)])
def test_a_traced_material_condition_does_not_outlive_its_solve(side, value):
    """A half-space swept as a ``Material`` is put back whole, not only the index derived from it.

    The setter stores two slots: the index the solve reads, which is itself a tracer and is
    caught as one, and the ``Material`` the property returns, which is a plain object holding
    one. Only ``_holds_tracer``'s step into a ``Material`` finds the second. Without that step
    the eager solve afterwards still agrees -- it reads the restored index -- which is what
    makes the leak quiet: ``solver.top`` hands back a ``Material`` around a dead tracer.
    Asserted on the object itself: the one the solver was built with, concrete, and a freeze
    that records a plan.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    structure = [dict(background=1.0, shapes=[_box(lx=350.0, ly=350.0, material=12.25)])]
    solver = wavebit.rcwa(backend='jax', lattice=SQUARE, structure=structure, **KW)
    reference = float(jnp.sum(solver.solve().T).real)
    before = getattr(solver, side)
    frozen = solver.freeze_geometry()

    def total(e):
        return jnp.sum(solver.solve(geometry=frozen, **{side: wavebit.Material(e)}).T).real

    assert float(jax.jit(total)(jnp.asarray(value))) == pytest.approx(reference, abs=1e-12)
    assert getattr(solver, side) is before
    assert not isinstance(getattr(solver, side).epsilon, jax.core.Tracer)
    assert solver.freeze_geometry().gamma_plans[0] is not None
    assert float(jnp.sum(solver.solve().T).real) == reference


def test_a_setter_that_raises_after_a_traced_assignment_still_puts_it_back():
    """The keywords are assigned inside the ``try``, so a bad one cannot leak a good one's tracer.

    ``solve(lattice=L, geometry=record, structure='junk')`` inside ``jax.jit``: the lattice
    setter took the tracer, the structure setter raised, and -- with the assignment outside
    the ``try`` -- the ``finally`` never ran, so ``solver.lattice`` held the tracer and every
    eager solve after it raised ``UnexpectedTracerError``. The same through
    ``thickness='junk'``, and through a misspelled keyword, which is refused before anything
    is assigned. Now the restore covers the assignment: the setter's own error comes out,
    and the cell is the one the solver was built with.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    solver = wavebit.rcwa(backend='jax', lattice=SHEAR, structure=_b_layer(500.0), **KW)
    frozen = solver.freeze_geometry()
    before = np.asarray(solver.lattice)
    reference = float(jnp.sum(solver.solve().T).real)

    for bad, error in ((dict(structure='junk'), ValueError), (dict(thickness='junk'), ValueError),
                       (dict(thicknes=[1.0]), TypeError)):
        seen = []

        def traced(L):
            try:
                solver.solve(lattice=L, geometry=frozen, **bad)
            except error as e:
                seen.append(e)
            return jnp.sum(L)

        jax.jit(traced)(jnp.asarray(SHEAR))
        assert len(seen) == 1, bad
        assert np.array_equal(np.asarray(solver.lattice), before), bad
        assert float(jnp.sum(solver.solve().T).real) == reference, bad


def test_a_freeze_on_a_solver_holding_a_tracer_is_refused_by_name():
    """A tracer assigned to the solver directly, inside a trace: the freeze refuses, naming the slot.

    ``solve()`` never leaves one behind, so this is the one way a solver holds a tracer at
    the freeze -- and a record taken then held ``gamma_plans = [None]`` and replayed blind.
    Both slots are asked: the cell, and a shape layer whose parameter is traced. A
    ``jax.grad`` tracer is not refused, since its primal plans the same plan the eager call
    would; the message names ``jax.jit`` and says to freeze on a concrete solver.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    solver = wavebit.rcwa(backend='jax', lattice=SHEAR, structure=_b_layer(500.0), **KW)
    seen = []

    def traced_cell(L):
        solver.lattice = L
        try:
            solver.freeze_geometry()
        except ValueError as e:
            seen.append(str(e))
        return jnp.sum(L)

    jax.jit(traced_cell)(jnp.asarray(SHEAR))
    assert len(seen) == 1 and 'cell basis' in seen[0] and 'concrete solver' in seen[0]
    solver.lattice = SHEAR                                   # assigned directly: put back by hand

    def traced_shape(lx):
        solver.structure = _b_layer(lx)
        try:
            solver.freeze_geometry()
        except ValueError as e:
            seen.append(str(e))
        return lx

    jax.jit(traced_shape)(jnp.asarray(500.0))
    assert len(seen) == 2 and 'layer 0' in seen[1] and 'concrete solver' in seen[1]
    solver.structure = _b_layer(500.0)
    assert solver.freeze_geometry().gamma_plans[0].pairs      # concrete again: a record

    # under grad the primal is read and the plan is the eager one
    def under_grad(lx):
        solver.structure = _b_layer(lx)
        return jnp.sum(solver.solve(geometry=solver.freeze_geometry()).T).real

    assert np.isfinite(float(jax.grad(under_grad)(jnp.asarray(500.0))))
    solver.structure = _b_layer(500.0)


def test_a_reconstruction_after_a_traced_solve_is_refused_rather_than_read_off_the_wrong_cell():
    """The modes belong to the traced cell; the solver holds the concrete one again. Refuse.

    Putting the cell back after a jitted ``solve(lattice=L)`` opens one door: a
    reconstruction that reads ``solver.lattice`` would read the put-back cell against modes
    computed on the tracer, and the old scene check skipped exactly that pairing (one key
    ``None``). It refuses now, naming ``solve_field`` -- which solves and reconstructs in one
    traced call -- and the eager solve; the eager solve then reconstructs as before.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    solver = wavebit.rcwa(backend='jax', lattice=SHEAR, structure=_b_layer(500.0), answers='field',
                          **KW)
    frozen = solver.freeze_geometry()
    jax.jit(lambda L: jnp.sum(solver.solve(lattice=L, geometry=frozen).T).real)(jnp.asarray(SHEAR))
    with pytest.raises(RuntimeError, match='solve_field'):
        solver.reconstruct_field(4, 4, 3)
    solver.solve(geometry=frozen)
    assert np.isfinite(np.asarray(solver.reconstruct_field(4, 4, 3))).all()


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
    solver = wavebit.rcwa(backend='jax', lattice=SHEAR, structure=_b_layer(500.0), **KW)
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
    autograd is the central difference to 1e-6 (``h = 1e-5``, ``shape_nvf_grid_size`` 128):

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
    / -1.5e-3 at orders 4 / 6 / 8 / 10 / 12 (secants of 1e-2 either side).
    These finite-order secants do not establish the infinite-order limit.
    The measurements are recorded in ``docs/normal-field-known-limits.md``
    (case J), beside the large closing-gap slopes measured by
    ``test_gamma_field.py::test_closing_gap_has_large_slopes_on_the_documented_grid``.
    That sampled ladder does not establish an unbounded physical derivative.
    An onset is not a merge: the faces here slide past each other's ends, and the value's
    slope is finite on both sides.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    def total(v):
        P._PROJECTOR_CACHE.clear()
        solver = wavebit.rcwa(backend='jax', lattice=SHEAR, structure=_b_layer(v), **KW)
        return jnp.sum(solver.solve().T).real

    for lx in (199.9, 200.0, 200.1, 350.0):
        assert np.isfinite(float(jax.grad(total)(jnp.asarray(lx)))), lx
    h = 1e-5
    for lx in (200.1, 350.0):
        fd = (float(total(jnp.asarray(lx + h))) - float(total(jnp.asarray(lx - h)))) / (2 * h)
        assert float(jax.grad(total)(jnp.asarray(lx))) == pytest.approx(fd, rel=1e-5), lx


# ------------------------------------------------------------------- what the record survives

def test_a_frozen_material_sweep_is_accepted_and_the_seam_contrast_follows():
    """The record holds no material: a sweep through equality replays, and the contrast follows.

    Frozen at ``4 | 9`` and replayed at ``9 | 9`` the seam's contrast is zero, live: the answer
    is the eager same-material solve's to 1e-12, eagerly and under ``jax.jit`` with the
    materials traced. Nothing was invalidated and nothing had to be.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    solver = wavebit.rcwa(backend='jax', lattice=SQUARE, structure=_k_layer(4.0, 9.0), **KW)
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
    assert abs(differ - same) > 1e-4                             # the contrast did something


def test_a_pair_that_appears_or_vanishes_is_refused_by_name():
    """B frozen with its run and replayed short of the onset, and the reverse: refused, eagerly.

    A shape moving never trips this; a face leaving its partner's line does, because the
    recorded pair would then cancel two faces that are interfaces (or fail to cancel two that
    are not). The message names the pair. Under a trace nothing checks, as for every record.
    """
    def frozen_at(lx):
        return wavebit.rcwa(backend='numpy', lattice=SHEAR, structure=_b_layer(lx), **KW).freeze_geometry()

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
    frozen = wavebit.rcwa(backend='numpy', lattice=SQUARE, structure=grating, **KW).freeze_geometry()
    short = [dict(background=1.0, shapes=[_box(lx=350.0, ly=PERIOD - 1.0)])]
    with pytest.raises(StaleGeometryError, match='no longer fits'):
        _total(short, SQUARE, geometry=frozen)
    small = [dict(background=1.0, shapes=[_box(lx=100.0, ly=100.0)])]
    frozen_small = wavebit.rcwa(backend='numpy', lattice=SQUARE, structure=small, **KW).freeze_geometry()
    big = [dict(background=1.0, shapes=[_box(lx=600.0, ly=600.0)])]
    with pytest.raises(StaleGeometryError, match='image'):
        _total(big, SQUARE, geometry=frozen_small)


#: Li's checkerboard primitive cell, on which a 2.5-wide bar tiles the plane at ``ly = 1.25``
#: (each v face covered half by ``a1`` and half by ``-a2``) and does not at ``ly = 0.8``.
CHECKER = np.array([[1.25, 1.25], [1.25, -1.25]])


def test_a_layer_that_starts_or_stops_tiling_the_plane_is_refused_by_that_name():
    """The record's ``empty`` flag has its own clause in the refusal, and each direction says which.

    Growing the bar to ``ly = 1.25`` also makes two pairs appear and both v faces dead, and the
    message names those too; what only this clause says is the consequence -- the layer is now
    uniform, and a record that still describes interfaces would put a normal field where the
    material runs on. Shrinking it back says the reverse.
    """
    span = [dict(background=1.0, shapes=[_box(lx=2.5, ly=0.8)])]
    bar = [dict(background=1.0, shapes=[_box(lx=2.5, ly=1.25)])]
    kw = dict(KW, wavelength=1.6)
    for held, replayed, says in ((span, bar, 'the layer tiles the plane now'),
                                 (bar, span, 'the layer no longer tiles the plane')):
        solver = wavebit.rcwa(backend='numpy', lattice=CHECKER, structure=held, **kw)
        frozen = solver.freeze_geometry()
        assert frozen.gamma_plans[0].empty is (held is bar)
        with pytest.raises(StaleGeometryError, match=says):
            solver.solve(structure=replayed, geometry=frozen)


def test_a_record_replayed_on_another_spelling_of_its_lattice_is_refused_by_the_reduction():
    """``a2 = (50, 700)`` and ``a2 = 2 a1 + (50, 700)`` are one lattice, and not one record.

    Every offset a plan holds counts cells of the Lagrange-reduced basis, and the plan carries
    the integers that reduce the basis as given. The two spellings reduce to the same basis by
    different integers, so the plans agree on every pair, piece and ring -- only the transform
    differs -- and replayed on the other spelling each recorded offset names a different image.
    That leaves the reduction as the *only* clause that can say what is wrong; without it the
    refusal would fall back to "the recorded plan differs from a fresh one". A freeze taken on
    the spelling in use answers.
    """
    near = np.array([[PERIOD, 50.0], [0.0, PERIOD]])
    far = np.array([[PERIOD, 1450.0], [0.0, PERIOD]])
    structure = _b_layer(500.0)
    for held, replayed in ((near, far), (far, near)):
        frozen = wavebit.rcwa(backend='numpy', lattice=held, structure=structure,
                              **KW).freeze_geometry()
        solver = wavebit.rcwa(backend='numpy', lattice=replayed, structure=structure, **KW)
        fresh = solver.freeze_geometry()
        assert frozen.gamma_plans[0].reduction != fresh.gamma_plans[0].reduction
        assert (frozen.gamma_plans[0]._replace(reduction=None)
                == fresh.gamma_plans[0]._replace(reduction=None))
        with pytest.raises(StaleGeometryError, match='the basis reduces differently now') as e:
            solver.solve(geometry=frozen)
        assert 'pair' not in str(e.value).split('. The recorded')[0], str(e.value)
        assert np.isfinite(float(np.sum(np.asarray(solver.solve(geometry=fresh).T))))


# ------------------------------------------------------------------------------ the cache

def _apart_layer(right):
    """Two boxes 200 apart, ``4 | right``: no pair, and the seam contrast reads both."""
    return [dict(background=1.0, shapes=[_box(cx=-200.0, lx=200.0, ly=300.0, material=4.0),
                                         _box(cx=200.0, lx=200.0, ly=300.0, material=right)])]


def _lone_layer(material):
    return [dict(background=1.0, shapes=[_box(lx=350.0, ly=350.0, material=material)])]


def _cache_sizes(make, values):
    """``len(_PROJECTOR_CACHE)`` after each solve of ``make(v)``, from a cleared cache."""
    P._PROJECTOR_CACHE.clear()
    sizes = []
    for v in values:
        wavebit.rcwa(backend='numpy', lattice=SQUARE, structure=make(v), **KW).solve()
        sizes.append(len(P._PROJECTOR_CACHE))
    return sizes


def test_the_projector_cache_reads_every_material_of_a_multishape_layer():
    """A material sweep on a multi-shape layer rebuilds per value; on a lone shape it hits.

    The seam contrast reads the *values* (``|eps_a - eps_b| / (|eps_a| + |eps_b|)``), so the
    key of a multi-shape layer holds every shape's material (:func:`parametric._material_key`):
    the K sweep ``4 | 9, 9.5, 12, 4, 4, 9`` is one entry per distinct pair, the repeats served
    -- 1, 2, 3, 4, 4, 4 against the cache's maximum of 8. The formula reads no record, so a
    pair-free layer reads the same contrast between its facing faces at any gap and the
    'apart' sweep rebuilds per value too (1, 2, 3, 4), where until 2026-09-11 it stayed at one
    -- the key then held only each cross pair's equality bit, and a pair-free layer's held
    nothing. A lone shape's field reads no material, and its sweep is one entry as it always
    was.
    """
    assert P._material_key(_k_layer(4.0, 9.0)[0]['shapes']) == (4.0 + 0j, 9.0 + 0j)
    assert _cache_sizes(lambda right: _k_layer(4.0, right),
                        (9.0, 9.5, 12.0, 4.0, 4.0, 9.0)) == [1, 2, 3, 4, 4, 4]

    assert not P.plan_gamma(_apart_layer(9.0)[0]['shapes'], as_lattice(lattice=SQUARE)).pairs
    assert _cache_sizes(_apart_layer, (9.0, 9.5, 4.0, 12.0)) == [1, 2, 3, 4]

    assert P._material_key(_lone_layer(9.0)[0]['shapes']) == ()
    assert _cache_sizes(_lone_layer, (9.0, 9.5, 4.0, 12.0)) == [1, 1, 1, 1]


def _keys_asked_for(monkeypatch):
    """A spy on the key: every key a build asked the cache for, in order."""
    keys = []
    real = P._projector_cache_key

    def spy(*args, **kwargs):
        keys.append(real(*args, **kwargs))
        return keys[-1]

    monkeypatch.setattr(P, '_projector_cache_key', spy)
    return keys


def test_a_traced_material_on_a_seam_is_never_served_from_the_cache(monkeypatch):
    """A tracer has no concrete value, so the key is ``None`` and the field is rebuilt."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    P._PROJECTOR_CACHE.clear()
    wavebit.rcwa(backend='jax', lattice=SQUARE, structure=_k_layer(4.0, 9.0), **KW).solve()
    assert len(P._PROJECTOR_CACHE) == 1

    def total(right):
        return jnp.sum(wavebit.rcwa(backend='jax', lattice=SQUARE, structure=_k_layer(4.0, right),
                                    **KW).solve().T).real

    keys = _keys_asked_for(monkeypatch)
    g = float(jax.grad(total)(jnp.asarray(9.0)))
    assert np.isfinite(g) and g != 0.0
    assert keys == [None] and len(P._PROJECTOR_CACHE) == 1     # built, not cached


def test_a_traced_material_on_a_pair_free_layer_is_never_served_from_the_cache(monkeypatch):
    """``jax.grad`` over one material of the 'apart' layer: key ``None``, the field rebuilt.

    A pair-free layer's key held no material until 2026-09-11, so a gradient with respect to
    one was served the coefficients built at the first value -- right then, since the field
    read no material, and wrong now that the seam contrast reads every shape's. The key of a
    layer of several shapes holds the values, a tracer has none, and the key is ``None``: the
    one key the solve under ``grad`` asks for, one entry before and after, and the gradient is
    the central difference. Measured at order 4, N = 128: AD 2.078100e-2 against
    2.078106e-2 at ``h = 1e-4`` (``h = 1e-5`` agrees with it to 7e-8), 3.2e-6 apart; served
    from a material-blind key -- the key a pair-free layer had -- the same gradient reads
    2.056730e-2, 1.0e-2 apart, which is what the 1e-5 tolerance separates.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    def total(right):
        return jnp.sum(wavebit.rcwa(backend='jax', lattice=SQUARE, structure=_apart_layer(right),
                                    **KW).solve().T).real

    h = 1e-4
    fd = (float(total(jnp.asarray(9.0 + h))) - float(total(jnp.asarray(9.0 - h)))) / (2 * h)
    P._PROJECTOR_CACHE.clear()
    wavebit.rcwa(backend='jax', lattice=SQUARE, structure=_apart_layer(9.0), **KW).solve()
    assert len(P._PROJECTOR_CACHE) == 1

    keys = _keys_asked_for(monkeypatch)
    g = float(jax.grad(total)(jnp.asarray(9.0)))
    assert keys == [None] and len(P._PROJECTOR_CACHE) == 1     # built, not cached
    assert g == pytest.approx(fd, rel=1e-5)


# ---------------------------------------------------------------------------- backends

def _l_ellipse_layer(lx):
    """L: an ellipse of ``lx`` x 200 at 10 deg; past 709.9 its support leaves its Wigner-Seitz cell."""
    return [dict(background=1.0, shapes=[dict(type='ellipse', cx=0.0, cy=0.0, lx=lx, ly=200.0,
                                             angle=np.deg2rad(10.0), material=12.25)])]


def test_an_ellipse_over_its_ring_replays_under_jit_and_is_refused_across_the_edge():
    """The arc's instance list is a plan fact: replayed from the record, re-derived eagerly.

    Frozen at 720 (eleven instances) the jitted solve equals the eager one; frozen at 690
    (one instance) and replayed at 720 the record is refused, naming the Wigner-Seitz cell,
    since the recorded list would read the fold's field on a shape whose neighbours' boundaries
    are inside the cell. The gradient in ``lx`` through the ring is the central difference to
    1e-6 (``h = 1e-3``, order 4, N = 128): the exact distance is stationary in its nearest
    point, so the bisection that finds it costs the derivative nothing.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    solver = wavebit.rcwa(backend='jax', lattice=SQUARE, structure=_l_ellipse_layer(720.0), **KW)
    frozen = solver.freeze_geometry()
    assert len(frozen.gamma_plans[0].pieces[0].imgs) == 11

    def total(lx, geometry=frozen):
        return jnp.sum(solver.solve(structure=_l_ellipse_layer(lx), geometry=geometry).T).real

    eager = float(total(jnp.asarray(720.0)))
    assert float(jax.jit(total)(jnp.asarray(720.0))) == pytest.approx(eager, abs=1e-12)
    h = 1e-3
    fd = (float(total(jnp.asarray(720.0 + h), None)) - float(total(jnp.asarray(720.0 - h), None))) / (2 * h)
    assert float(jax.grad(lambda v: total(v, None))(jnp.asarray(720.0))) == pytest.approx(fd, rel=1e-5)

    inside = wavebit.rcwa(backend='numpy', lattice=SQUARE, structure=_l_ellipse_layer(690.0), **KW)
    record = inside.freeze_geometry()
    assert record.gamma_plans[0].pieces[0].imgs == ((0, 0),)
    with pytest.raises(StaleGeometryError, match='Wigner-Seitz'):
        _total(_l_ellipse_layer(720.0), SQUARE, geometry=record)
    assert _total(_l_ellipse_layer(700.0), SQUARE, geometry=record) == pytest.approx(
        _total(_l_ellipse_layer(700.0), SQUARE), abs=1e-13)             # still inside: accepted


def test_a_jitted_ellipse_with_no_record_is_refused_and_a_circle_builds():
    """The blind plan reads a shape's *type*, not its size -- so an ellipse is refused, a circle not.

    Whether a round shape leaves its Wigner-Seitz cell is decided by its size
    (:func:`~wavebit._core.parametric._arc_offsets`), which a ``jax.jit`` tracer cannot
    give. Until 2026-09-06 the blind plan answered every round shape with its single
    instance, so the 720 x 200 ellipse under ``jax.jit`` with no record was built on the
    fold's field the ring had replaced: -8.222e-3 / -7.058e-3 in ``sum(T)`` from the eager
    solve at orders 4 / 6 (``shape_nvf_grid_size`` 128 / 256), the same digits as a NumPy solve with
    the plan forced to ``None``, and nothing said so. It is refused now with the message every
    traced decision raises with, naming the freeze; with a record taken outside the trace the
    eleven-instance list replays to 1e-12 of the eager solve. A circle of any accepted size
    never leaves its cell, so its single instance *is* the planned list and the jitted circle
    builds without a record, to 1e-12 of the eager solve. "Accepted" is the eager refusal's
    word, and that refusal is what bounds the circle: it cannot fire on a traced ``lx``, so a
    jitted sweep carried past ``P`` builds an overlapping structure in silence (750 on the
    700 square: ``sum(T)`` 0.928891 where the eager solve raises) -- a sweep must not carry
    it past ``P``, and re-taking the freeze every step (README) is what catches it.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()

    solver = wavebit.rcwa(backend='jax', lattice=SQUARE, structure=_l_ellipse_layer(720.0), **KW)
    frozen = solver.freeze_geometry()
    assert len(frozen.gamma_plans[0].pieces[0].imgs) == 11

    def total(lx, geometry=None):
        return jnp.sum(solver.solve(structure=_l_ellipse_layer(lx), geometry=geometry).T).real

    eager = float(total(jnp.asarray(720.0)))
    with pytest.raises(ValueError, match='freeze_geometry'):
        jax.jit(total)(jnp.asarray(720.0))
    replayed = float(jax.jit(lambda lx: total(lx, frozen))(jnp.asarray(720.0)))
    assert replayed == pytest.approx(eager, abs=1e-12)
    P._PROJECTOR_CACHE.clear()
    plain = float(np.sum(np.asarray(wavebit.rcwa(backend='numpy', lattice=SQUARE,
                                                 structure=_l_ellipse_layer(720.0),
                                                 **KW).solve().T)))
    assert eager == pytest.approx(plain, abs=1e-12)

    def circle_layer(lx):
        return [dict(background=1.0, shapes=[dict(type='circle', cx=0.0, cy=0.0, lx=lx,
                                                 material=12.25)])]

    disc = wavebit.rcwa(backend='jax', lattice=SQUARE, structure=circle_layer(693.0), **KW)

    def disc_total(lx):
        return jnp.sum(disc.solve(structure=circle_layer(lx)).T).real

    eager_disc = float(disc_total(jnp.asarray(693.0)))
    assert float(jax.jit(disc_total)(jnp.asarray(693.0))) == pytest.approx(eager_disc, abs=1e-12)
    assert np.isfinite(float(jax.jit(jax.grad(disc_total))(jnp.asarray(693.0))))


@pytest.mark.parametrize('name, lattice, structure', [
    ('B', SHEAR, _b_layer(500.0)),
    ('D', SQUARE, _d_layer()),
    ('K', SQUARE, _k_layer(4.0, 9.0)),
    ('L ellipse', SQUARE, _l_ellipse_layer(720.0)),
])
def test_every_backend_builds_the_same_field(name, lattice, structure):
    """NumPy, JAX and Torch through three namespaces, one plan, one answer to 1e-12."""
    totals = [_total(structure, lattice, backend='numpy')]
    for backend in ('jax', 'torch'):
        pytest.importorskip(backend)
        if backend == 'jax':
            wavebit.enable_x64()
        totals.append(_total(structure, lattice, backend=backend))
    assert max(abs(t - totals[0]) for t in totals) < 1e-12, (name, totals)
