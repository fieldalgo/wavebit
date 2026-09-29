"""``SolverJax.solve_field`` — the jit-compiled solve-and-reconstruct entry point.

This method jits over ``self``, which is what makes it different from ``solve``: the whole
structure arrives as tracers, so every discrete decision downstream — the grating-type dispatch,
the NVF and integrated-path geometry — has nothing concrete to branch on. It
therefore *requires* a ``StaticGeometry`` from ``freeze_geometry()``, and says so rather than
letting a raw ``TracerBoolConversionError`` escape.

The regression these tests pin: ``tetm`` and ``psi`` are one degree of freedom carried twice —
``psi`` as a traced pytree child, ``tetm`` as static aux data because the solve *dispatches* on
it. Unflattening applied ``psi`` last, whose setter recomputes ``tetm``, so the static dispatch
key silently became a tracer and every caller of this method died on the first ``if tetm == 'te'``.
"""
import numpy as np
import pytest

import wavebit

RNG = np.random.default_rng(0)
# Permittivity rasters -- an index draw, squared, so these are the same two layers this file
# traced before the material convention flipped.
UCELL_1D = RNG.uniform(1.0, 4.0, (1, 1, 8)) ** 2
UCELL_2D = RNG.uniform(1.0, 4.0, (1, 5, 5)) ** 2

BASE = dict(top=1.0, bot=1.0, theta=0.1, wavelength=900.0, thickness=[500.0])

# (kwargs, (sample_x, sample_y, sample_z)) per grating type the dispatcher can pick.
CASES = {
    '1d_tetm': (dict(tetm='te', phi=None, truncation_order=[3], period=[1000.0], structure=UCELL_1D), (8, 1, 4)),
    '1d_tm': (dict(tetm='tm', phi=None, truncation_order=[3], period=[1000.0], structure=UCELL_1D), (8, 1, 4)),
    '1d_conical': (dict(psi=0.35 * np.pi, phi=0.2, truncation_order=[3], period=[1000.0], structure=UCELL_1D), (8, 1, 4)),
    '2d': (dict(psi=0.35 * np.pi, phi=0.2, truncation_order=[2, 2], period=[1000.0, 1000.0], structure=UCELL_2D), (4, 4, 4)),
}


def _solver(backend, case, **extra):
    kwargs, _ = CASES[case]
    extra.setdefault('answers', 'field')      # every solver in this file is reconstructed from
    return wavebit.rcwa(backend=backend, **BASE, **kwargs, **extra)


def _res(case):
    return CASES[case][1]


@pytest.fixture(autouse=True)
def _require_jax():
    pytest.importorskip('jax')


@pytest.mark.parametrize('case', list(CASES))
def test_jit_matches_the_eager_path(case):
    """The jitted method reproduces ``solve_field_eager`` exactly, for every grating type.

    Freezing the geometry is an optimization, not a different computation — same contract as
    ``solve(geometry=...)``.
    """
    solver = _solver(1, case)
    sample_x, sample_y, sample_z = _res(case)

    jitted, field = solver.solve_field(sample_x=sample_x, sample_y=sample_y, sample_z=sample_z,
                                            geometry=solver.freeze_geometry())
    eager, field_eager = solver.solve_field_eager(sample_x=sample_x, sample_y=sample_y, sample_z=sample_z)

    assert np.allclose(np.asarray(field), np.asarray(field_eager), rtol=0, atol=1e-10)
    assert np.allclose(np.asarray(jitted.T), np.asarray(eager.T), rtol=0, atol=1e-12)
    assert np.allclose(np.asarray(jitted.R), np.asarray(eager.R), rtol=0, atol=1e-12)


@pytest.mark.parametrize('case', list(CASES))
def test_jit_matches_the_numpy_backend(case):
    """Physics check: the jitted JAX field agrees with NumPy, not merely with itself."""
    sample_x, sample_y, sample_z = _res(case)

    jax_solver = _solver(1, case)
    _, jax_field = jax_solver.solve_field(sample_x=sample_x, sample_y=sample_y, sample_z=sample_z,
                                               geometry=jax_solver.freeze_geometry())
    _, np_field = _solver(0, case).solve_field(sample_x=sample_x, sample_y=sample_y, sample_z=sample_z)

    assert np.asarray(jax_field).shape == np.asarray(np_field).shape
    assert np.allclose(np.asarray(jax_field), np.asarray(np_field), rtol=1e-8, atol=1e-8)


def test_omitting_the_geometry_names_freeze_geometry():
    """The whole point of the guard: a directive, not a TracerBoolConversionError.

    Before this was added, every call to ``solve_field`` on the JAX backend died inside the
    trace with a message about ``self[4]``, which points at nothing a caller can act on.
    """
    solver = _solver(1, '1d_conical')

    with pytest.raises(ValueError, match=r'freeze_geometry\(\)'):
        solver.solve_field(sample_x=8, sample_y=1, sample_z=4)

    # ...and it stays raised — a failed trace caches nothing, so the second call is not silently
    # served from the jit cache.
    with pytest.raises(ValueError, match=r'solve_field_eager'):
        solver.solve_field(sample_x=8, sample_y=1, sample_z=4)


def test_no_jit_still_auto_detects_without_a_geometry():
    """``solve_field_eager`` keeps ``geometry`` optional: eager code needs no freeze."""
    solver = _solver(1, '1d_conical')
    sample_x, sample_y, sample_z = _res('1d_conical')

    auto, auto_field = solver.solve_field_eager(sample_x=sample_x, sample_y=sample_y, sample_z=sample_z)
    frozen, frozen_field = solver.solve_field_eager(sample_x=sample_x, sample_y=sample_y, sample_z=sample_z,
                                                          geometry=solver.freeze_geometry())

    assert np.array_equal(np.asarray(auto_field), np.asarray(frozen_field))
    assert np.array_equal(np.asarray(auto.T), np.asarray(frozen.T))


@pytest.mark.parametrize('transform', ['jit', 'grad'])
@pytest.mark.parametrize('door', ['solve_field', 'solve_field_eager'])
def test_no_traced_condition_outlives_a_field_call(door, transform):
    """A wavelength traced into either field door is the solver's for the call only.

    Both doors under an outer ``jax.jit`` or ``jax.grad``, which is how an optimizer reaches
    them. Until 2026-09-24 the eager one assigned its keywords and kept every one, tracers
    included: ``solver.wavelength`` was left a dead tracer, and every eager ``solve()`` after it
    raised ``UnexpectedTracerError`` -- the leak ``solve`` closed on 2026-09-06, still open one
    door over. The jitted one never leaked, because it assigned inside its own trace, onto a
    copy -- which is also why it kept nothing concrete either. Both now assign on the live
    solver under ``solve``'s rule, so this pins the half of that rule a sweep cannot see: the
    slot reads its number afterwards, and the eager solve is the untouched one's to the bit.

    The traced value is 700, not the solver's own 900, and the answer is compared -- value
    under ``jit``, derivative under ``grad`` -- with a solver *built* at the traced wavelength.
    Traced at 900 this passed for a door that dropped the keyword: the value was the solver's
    own either way, and nothing read the derivative, which such a door returns as exactly 0.
    """
    import jax
    import jax.numpy as jnp
    wavebit.enable_x64()

    solver = _solver(1, '2d')
    geometry = solver.freeze_geometry()
    reference = float(jnp.sum(solver.solve().T).real)

    def total(w):
        result, _ = getattr(solver, door)(sample_x=4, sample_y=4, sample_z=4,
                                          geometry=geometry, wavelength=w)
        return jnp.sum(result.T).real

    def built_at(w):
        built = wavebit.rcwa(backend=1, **{**BASE, 'wavelength': w}, **CASES['2d'][0],
                             answers='field')
        result, _ = built.solve_field_eager(sample_x=4, sample_y=4, sample_z=4,
                                            geometry=geometry)
        return jnp.sum(result.T).real

    w = jnp.asarray(700.0)
    got = float(getattr(jax, transform)(total)(w))
    want = float(getattr(jax, transform)(built_at)(w))
    assert want != 0.0 and abs(got - want) < 1e-10 * abs(want), (door, got, want)
    assert not isinstance(solver.wavelength, jax.core.Tracer), door
    assert float(solver.wavelength) == 900.0
    assert float(jnp.sum(solver.solve().T).real) == reference


@pytest.mark.parametrize('fourier_coefficients', ['sampled', 'integrated'])
def test_every_traceable_fourier_strategy_runs_under_jit(fourier_coefficients):
    """The integrated series included: with its block compression frozen, it no longer bails.

    The old implementation printed a notice and returned ``(None, None, None)`` for the
    closed-form path — a silent wrong-arity return that no caller could unpack.
    """
    solver = _solver(1, '2d', fourier_coefficients=fourier_coefficients)
    result, field = solver.solve_field(sample_x=4, sample_y=4, sample_z=4,
                                            geometry=solver.freeze_geometry())

    assert np.asarray(field).shape == (1, 4, 4, 4, 6)
    assert np.isfinite(np.asarray(result.T)).all()


def test_a_freeze_recompiles_only_when_its_content_moves():
    """A frozen geometry is a static argument, so what it compares by decides recompilation.

    It used to compare by identity, which made *taking a freeze again* a recompilation even
    when it recorded exactly the same decisions. That is the wrong default for the loop this
    API is for: re-freezing every step is the only way a fit cannot be served a stale record,
    and identity comparison priced it at a full recompilation per step. Records that can say
    what they contain now do, so an unchanged structure re-freezes into the same cache entry.

    Records holding arrays still fall back to identity — value equality on an array is neither
    well defined nor hashable — so this pins both halves.

    The count is read off ``_solve_field_compiled``, the jitted body ``solve_field`` calls once
    it has assigned its keyword conditions on the live solver, outside the trace.
    """
    import jax
    from wavebit.backends.jax_backend import RCWAJax

    compiled = RCWAJax._solve_field_compiled
    jax.clear_caches()
    solver = _solver(1, '1d_conical')          # ('sampled', 'li'): records nothing at all
    geometry = solver.freeze_geometry()
    before = compiled._cache_size()

    solver.solve_field(sample_x=8, sample_y=1, sample_z=4, geometry=geometry)
    solver.solve_field(sample_x=8, sample_y=1, sample_z=4, geometry=geometry)
    assert compiled._cache_size() == before + 1

    # Freezing again on the same structure records the same thing, so it is the same program.
    solver.solve_field(sample_x=8, sample_y=1, sample_z=4, geometry=solver.freeze_geometry())
    assert compiled._cache_size() == before + 1

    # A record carrying arrays cannot compare by content, and says so by compiling again.
    nvf = _solver(1, '1d_conical', factorization='nvf')
    nvf.solve_field(sample_x=8, sample_y=1, sample_z=4, geometry=nvf.freeze_geometry())
    nvf.solve_field(sample_x=8, sample_y=1, sample_z=4, geometry=nvf.freeze_geometry())
    assert compiled._cache_size() == before + 3


def test_resolution_and_field_selection_are_static():
    """They set the output shape, so they must be compile-time constants, not traced."""
    solver = _solver(1, '2d')
    geometry = solver.freeze_geometry()

    _, one = solver.solve_field(sample_x=4, sample_y=4, sample_z=4, geometry=geometry)
    _, three = solver.solve_field(sample_x=4, sample_y=4, sample_z=4, geometry=geometry,
                                       incidences=('incident', 's', 'p'))

    assert np.asarray(one).shape == (1, 4, 4, 4, 6)
    assert np.asarray(three).shape == (3, 4, 4, 4, 6)


@pytest.mark.parametrize('case,slot,values', [
    ('2d', 'psi', (0.0, 0.35 * np.pi, np.pi / 2)),
    ('1d_tetm', 'tetm', ('te', 'tm')),
])
def test_changing_the_incidence_is_not_served_from_a_stale_trace(case, slot, values):
    """Pinning ``tetm`` static must not pin it *stale*.

    It is a jit cache key, so changing it has to recompile rather than silently reuse the trace
    built for the previous polarization — which would be a wrong answer, not a crash. Checked
    against the eager path at each value, and the values must genuinely differ from each other.

    The two rows are the two kinds of incidence, and they are exercised through the slot each
    one owns: ``psi`` is a traced child and varies freely, while ``tetm`` is the static
    declaration and has exactly two values. A 2D structure has no ``tetm`` to change at all —
    it is refused there — which is why that row sweeps the angle instead.
    """
    solver = _solver(1, case)
    sample_x, sample_y, sample_z = _res(case)
    geometry = solver.freeze_geometry()

    fields = {}
    for value in values:
        setattr(solver, slot, value)
        _, jitted = solver.solve_field(sample_x=sample_x, sample_y=sample_y, sample_z=sample_z,
                                            geometry=geometry)
        _, eager = solver.solve_field_eager(sample_x=sample_x, sample_y=sample_y, sample_z=sample_z)
        assert np.allclose(np.asarray(jitted), np.asarray(eager), rtol=0, atol=1e-10), \
            f'jit disagrees with eager at {slot}={value}'
        fields[value] = np.asarray(jitted)

    for a, b in zip(values, values[1:]):
        assert np.abs(fields[a] - fields[b]).max() > 1e-6, \
            f'{slot}={a} and {slot}={b} produced the same field — a stale trace was reused'


def test_the_declaration_is_static_and_the_angle_is_traced():
    """The root cause, pinned directly — and it is the same split the API is built on.

    ``tetm`` is aux data precisely because the solve dispatches on it, so it must come back
    from a trace concrete. ``psi`` is a value, so it must come back a tracer. What the two
    rows below add is that they are *exclusive*: a solver that declared its polarization away
    reads ``psi is None``, because a declared-away degree of freedom should be absent rather
    than present-and-ignored. It used to be present — a derived ``pi/2`` whose gradient was
    quietly zero on this very path, since the pure 1D solve never reads it.
    """
    import jax

    def peek_at(case):
        seen = {}

        def peek(s):
            seen['tetm'] = s.tetm
            seen['psi'] = s.psi
            return s.structure.sum()

        jax.jit(peek)(_solver(1, case))
        return seen

    declared = peek_at('1d_tetm')
    assert declared['tetm'] == 'te'
    assert not isinstance(declared['tetm'], jax.core.Tracer), 'tetm must survive concrete'
    assert declared['psi'] is None, 'a declared-away angle must be absent, not zero'

    angled = peek_at('1d_conical')
    assert angled['tetm'] is None
    assert isinstance(angled['psi'], jax.core.Tracer), 'psi must stay differentiable'


#: A sheared cell, the one whose basis is not a period in another spelling, and the square one a
#: `period=` sweep moves when the solver was built with `lattice=`.
SHEAR = np.array([[700.0, 200.0], [0.0, 700.0]])
SQUARE = np.diag([700.0, 700.0])


def _cell_solver(basis):
    """A raster on ``basis``, declared with ``lattice=`` -- the spelling a basis rides under."""
    eps = np.where(np.random.default_rng(0).random((1, 6, 6)) < 0.4, 5.76, 1.0)
    return wavebit.rcwa(backend=1, lattice=basis, structure=eps, truncation_order=[1, 1],
                        thickness=[300.0], wavelength=900.0, theta=0.2, phi=0.1, psi=0.3,
                        answers='field', factorization='nvf')


def _scene_2d(**over):
    """The '2d' case with any of its conditions -- `top` and `structure` included -- moved."""
    return wavebit.rcwa(backend=1, **{**BASE, **CASES['2d'][0], 'answers': 'field', **over})


@pytest.mark.parametrize('name, basis, value', [('lattice', SHEAR, SHEAR),
                                                ('period', SQUARE, np.array([700.0, 630.0]))],
                         ids=['lattice', 'period-on-a-lattice-built-solver'])
@pytest.mark.parametrize('transform', ['jit', 'grad'])
def test_a_traced_cell_crosses_the_jitted_door(name, basis, value, transform):
    """``jax.grad`` of ``solve_field(lattice=L)`` answers, and answers what the eager door does.

    Once the jitted door assigned its keywords on the live solver it flattened a *traced*
    basis, and the basis rode in aux data as a nested tuple: ``.tolist()`` on a tracer raised
    ``ConcretizationTypeError`` before anything was solved -- for ``lattice=``, and for
    ``period=`` on a solver built with ``lattice=``, whose setter re-derives the basis. The
    basis is a pytree child now. The value under ``jit`` is compared with ``solve_field_eager``'s
    on a second solver over the same frozen geometry (they agreed to 1e-15, measured
    2026-09-24); the derivative under ``grad`` with a central difference of the same door under
    ``jax.jit``, where the frozen record is replayed as frozen -- the function the gradient is
    of. Concretely, both doors re-check the record against every perturbed cell and refuse it.
    Measured: 3e-9 for the shear entry of the basis, 9e-10 for a period.
    """
    import jax
    import jax.numpy as jnp

    solver = _cell_solver(basis)
    geometry = solver.freeze_geometry()

    def through(door_solver, door):
        def total(x):
            result, field = getattr(door_solver, door)(3, 3, 2, geometry=geometry,
                                                       **{name: x})
            return jnp.sum(result.T).real + 1e-3 * jnp.sum(jnp.abs(field) ** 2)
        return total

    total = through(solver, 'solve_field')
    if transform == 'jit':
        got = float(jax.jit(total)(jnp.asarray(value)))
        want = float(jax.jit(through(_cell_solver(basis), 'solve_field_eager'))(
            jnp.asarray(value)))
        assert abs(got - want) < 1e-10 * abs(want)
        assert np.array_equal(np.asarray(solver.lattice), basis), 'a traced cell stayed'
    else:
        got = np.asarray(jax.grad(total)(jnp.asarray(value)))
        assert np.array_equal(np.asarray(solver.lattice), basis), 'a traced cell stayed'
        # The difference quotient is taken under `jax.jit` too: there the cell is traced and
        # the frozen record is replayed as frozen, which is the function the gradient is of.
        # Called concretely, each perturbed cell would be refused as a stale record by the
        # door's own check, as `solve()` refuses it -- the normal field depends on the cell.
        h, want = 1e-3, np.zeros(value.shape)
        replayed = jax.jit(total)
        for i in np.ndindex(value.shape):
            step = np.zeros(value.shape)
            step[i] = h
            want[i] = (float(replayed(jnp.asarray(value + step)))
                       - float(replayed(jnp.asarray(value - step)))) / (2 * h)
        assert np.abs(want).max() > 0
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-6 * np.abs(want).max())


#: Rectangular cells for the sweeps below, on the default Li / sampled path. That path freezes
#: nothing -- its record is empty -- so one freeze is a legitimate record of every cell in the
#: sweep, and compiling once is the correct outcome rather than a stale record replayed.
RECT_SWEEP = [np.diag([700.0, 700.0]), np.diag([707.0, 700.0]), np.diag([714.0, 690.0])]


def _li_raster_on(cell, radius=None):
    """A raster on ``cell``: a random mask, or a disc of fixed absolute ``radius`` drawn there."""
    if radius is None:
        return np.where(np.random.default_rng(0).random((1, 6, 6)) < 0.4, 5.76, 1.0)
    X, Y = (np.asarray(c) for c in wavebit.Lattice.from_columns(cell).meshgrid_xy((6, 6)))
    return np.where(X ** 2 + Y ** 2 < radius ** 2, 5.76, 1.0)[None]


def _li_solver(**cell):
    return wavebit.rcwa(backend=1, truncation_order=[1, 1], thickness=[300.0],
                        wavelength=900.0, theta=0.2, phi=0.1, psi=0.3, answers='field', **cell)


def _fresh_eager_T(**cell):
    """The answer a solver *built* on that cell gives eagerly -- nothing shared, no freeze."""
    return np.asarray(_li_solver(**cell).solve().T)


@pytest.mark.parametrize('name', ['lattice', 'period'],
                         ids=['lattice', 'period-on-a-lattice-built-solver'])
def test_a_cell_sweep_through_the_jitted_door_compiles_once(name):
    """A new cell is a new value of the compiled call, not a new program.

    The basis was aux data, hashed into the treedef, so each cell in a sweep -- ``lattice=``, or
    ``period=`` on a solver built with ``lattice=`` -- compiled again: measured three
    compilations for three values where a wavelength or thickness sweep takes one. As a child
    it is a leaf.

    Measured on rectangular cells with the default Li / sampled path, and compared with a solver
    built on each cell and solved eagerly with nothing shared. The first version of this test
    swept a sheared cell under NVF with one freeze and compared against the same freeze, so it
    pinned "compiles once" on a normal-field record replayed onto cells it no longer described
    -- the jitted door replays a record on trust, and such a replay answered wrongly by up to
    0.86 in ``sum(T)`` (measured 2026-09-25) while a comparison sharing the record could not see
    it. A correct oblique sweep re-freezes each cell, and an NVF record compares by identity, so
    it compiles per cell whatever the basis rides as.
    """
    import jax
    from wavebit.backends.jax_backend import RCWAJax

    compiled = RCWAJax._solve_field_compiled
    jax.clear_caches()
    raster = _li_raster_on(RECT_SWEEP[0])
    solver = _li_solver(lattice=RECT_SWEEP[0], structure=raster)
    geometry = solver.freeze_geometry()
    before = compiled._cache_size()
    answers = []
    for cell in RECT_SWEEP:
        value = cell if name == 'lattice' else np.diag(cell)
        result, _ = solver.solve_field(3, 3, 2, geometry=geometry, **{name: value})
        want = _fresh_eager_T(lattice=cell, structure=raster)
        assert np.abs(np.asarray(result.T) - want).max() < 1e-12, cell
        answers.append(float(np.sum(np.asarray(result.T)).real))
    assert compiled._cache_size() == before + 1
    assert len(set(answers)) == len(RECT_SWEEP), 'the sweep did not move the answer'


def test_a_declared_structure_cell_sweep_compiles_once():
    """Redrawing a ``Lattice.drawn`` raster on each cell of a sweep is a new value, not a program.

    A pattern of fixed absolute size -- a disc of radius 160 -- is redrawn on each cell, the
    workflow ``docs/oblique-lattice.md`` gives for the Cartesian frame, and declared with
    ``Lattice.drawn``; ``lattice=L_k, structure=L_k.drawn(raster_k)`` is the only legal spelling,
    since a declared raster is refused on any other cell. The declared cell rode in aux as a
    nested tuple, so every step was a new treedef: measured three compilations for three cells,
    about 4 s each, where the same sweep over the bare arrays took one. The numbers are leaves
    now. Each answer is a fresh eager solve's on that cell; a raster declared on one cell is
    still refused on another.
    """
    import jax
    from wavebit.backends.jax_backend import RCWAJax

    def drawn_on(cell):
        return wavebit.Lattice.from_columns(cell).drawn(_li_raster_on(cell, radius=160.0))

    compiled = RCWAJax._solve_field_compiled
    jax.clear_caches()
    solver = _li_solver(lattice=RECT_SWEEP[0], structure=drawn_on(RECT_SWEEP[0]))
    geometry = solver.freeze_geometry()
    before = compiled._cache_size()
    answers = []
    for cell in RECT_SWEEP:
        result, _ = solver.solve_field(3, 3, 2, geometry=geometry, lattice=cell,
                                       structure=drawn_on(cell))
        want = _fresh_eager_T(lattice=cell, structure=drawn_on(cell))
        assert np.abs(np.asarray(result.T) - want).max() < 1e-12, cell
        answers.append(float(np.sum(np.asarray(result.T)).real))
    assert compiled._cache_size() == before + 1
    assert len(set(answers)) == len(RECT_SWEEP), 'the sweep did not move the answer'
    with pytest.raises(ValueError, match='drawn on a different cell'):
        solver.solve_field(3, 3, 2, geometry=geometry, lattice=RECT_SWEEP[0],
                           structure=drawn_on(RECT_SWEEP[1]))


@pytest.mark.parametrize('name, value', [('top', 2.25),
                                         ('structure', UCELL_2D[:, ::-1] * 1.5),
                                         ('truncation_order', [1, 1])])
def test_a_kept_keyword_takes_the_old_solve_state_with_it(name, value):
    """After ``solve_field(top=...)`` the eager solve's modes are refused, not walked.

    The jitted door keeps a concrete keyword on the solver and solves on the copy its trace
    rebuilds, so the live solver's modes and amplitudes stayed those of the eager solve before
    it -- and ``_reconstruction_scene`` keys none of these three, so nothing noticed. Measured
    2026-09-24 on a 2D raster: ``reconstruct_field()`` after ``top=2.25`` returned a field 2.12
    from the moved scene's and 0.83 from the old one's, after a new structure the old scene's
    field to 3e-14, after a new truncation a reshape ``TypeError``. It now refuses, naming the
    door and ``solve()``, and one ``solve()`` later reconstructs the field ``solve_field``
    returned.
    """
    solver = _scene_2d()
    solver.solve()
    geometry = _scene_2d(**{name: value}).freeze_geometry()
    _, field = solver.solve_field(4, 4, 4, geometry=geometry, **{name: value})
    with pytest.raises(RuntimeError, match=r'jitted solve_field\(\.\.\.\)[\s\S]*Call solve\(\)'):
        solver.reconstruct_field(4, 4, 4)
    solver.solve()
    np.testing.assert_allclose(np.asarray(solver.reconstruct_field(4, 4, 4)),
                               np.asarray(field), rtol=0, atol=1e-10)


def test_a_call_that_keeps_nothing_keeps_the_solve_state():
    """No keyword, or only a traced one, leaves the scene where it was -- and the state valid.

    Dropping the state on every call would refuse a reconstruction that is right: the eager
    solve before the call is still a solve of the scene the solver reports.
    """
    import jax
    import jax.numpy as jnp

    solver = _scene_2d()
    geometry = solver.freeze_geometry()
    solver.solve()
    before = np.asarray(solver.reconstruct_field(4, 4, 4))

    solver.solve_field(4, 4, 4, geometry=geometry)

    def total(t):
        result, _ = solver.solve_field(4, 4, 4, geometry=geometry, thickness=t)
        return jnp.sum(result.T).real
    jax.grad(total)(jnp.asarray([610.0]))

    assert np.array_equal(np.asarray(solver.reconstruct_field(4, 4, 4)), before)


@pytest.mark.parametrize('name, value', [('top', 1.0), ('thickness', [500.0]), ('theta', 0.1)])
def test_a_keyword_equal_to_what_the_solver_holds_keeps_the_solve_state(name, value):
    """``solve_field(top=1.0)`` on a solver already at ``top=1.0`` has not moved the scene.

    Whether the call moved it was decided by object identity over the instance, and a concrete
    keyword stores a new object whatever its value, so this dropped a solve state that still
    described the scene -- ``reconstruct_field`` refused where it had answered exactly, and
    ``wavelength`` escaped only because its setter stores the object it is given. It is decided
    by the flattened solver now, the scene the compiled call reads.
    """
    solver = _scene_2d()
    geometry = solver.freeze_geometry()
    solver.solve()
    before = np.asarray(solver.reconstruct_field(4, 4, 4))
    solver.solve_field(4, 4, 4, geometry=geometry, **{name: value})
    assert np.array_equal(np.asarray(solver.reconstruct_field(4, 4, 4)), before)


def test_a_declaration_named_at_the_jitted_door_answers_as_one_built_with_it():
    """``solve_field(tetm='te')`` on a 1D solver built without it: the TE answer, and kept.

    It raised inside JAX's tree machinery -- the rebuild handed the default angle in the `psi`
    slot back as a *given* `psi` beside `tetm`, which is refused as naming the incidence twice
    -- after `tetm` had been assigned on the live solver, which kept it.
    """
    kwargs = dict(CASES['1d_tetm'][0])
    del kwargs['tetm']
    solver = wavebit.rcwa(backend=1, **BASE, **kwargs, answers='field')
    built = _solver(1, '1d_tetm')
    geometry = built.freeze_geometry()
    got = solver.solve_field(*_res('1d_tetm'), geometry=geometry, tetm='te')
    want = built.solve_field(*_res('1d_tetm'), geometry=geometry)
    assert np.array_equal(np.asarray(got[0].R), np.asarray(want[0].R))
    assert np.array_equal(np.asarray(got[1]), np.asarray(want[1]))
    assert solver.tetm == 'te'


def test_the_scattering_matrix_a_kept_keyword_drops_says_where_it_went():
    """After the jitted door kept a keyword, ``solver.scattering`` names that door.

    The door drops the matrix with the rest of the solve state, since it belonged to the scene
    before the keyword. The message it fell through to said there was no scattering matrix
    *yet* -- that no solve had run -- when one had, on a scene the solver no longer reports.
    One ``solve()`` later the matrix is back, and it is the moved scene's.
    """
    solver = _scene_2d(layer_algorithm='SMM')
    solver.solve()
    assert solver.scattering.S11 is not None
    geometry = _scene_2d(layer_algorithm='SMM', top=2.25).freeze_geometry()
    solver.solve_field(4, 4, 4, geometry=geometry, top=2.25)
    with pytest.raises(RuntimeError, match=r'jitted solve_field\(\.\.\.\)[\s\S]*Call solve\(\)'):
        solver.scattering
    solver.solve()
    moved = _scene_2d(layer_algorithm='SMM', top=2.25)
    moved.solve()
    assert np.array_equal(np.asarray(solver.scattering.S11), np.asarray(moved.scattering.S11))


#: A lone 350 box on the 700 square, the scene ``docs/normal-field-known-limits.md`` 8d names.
BOX_LAYER = [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0, lx=350.0,
                                               ly=350.0, angle=0.0, material=12.25)])]


def test_parametric_structures_cross_the_compiled_door_and_keyword_assignment():
    """Shape strings are static declarations; the retained keyword door keeps that support."""
    kw = dict(backend=1, lattice=SQUARE, thickness=[300.0], wavelength=900.0,
              truncation_order=[2, 2], answers='field')
    parametric = wavebit.rcwa(structure=BOX_LAYER, **kw)
    geometry = parametric.freeze_geometry()
    compiled, field = parametric.solve_field(3, 3, 2, geometry=geometry)
    eager, eager_field = parametric.solve_field_eager(3, 3, 2)
    np.testing.assert_allclose(np.asarray(compiled.T), np.asarray(eager.T), atol=1e-11)
    np.testing.assert_allclose(np.asarray(field), np.asarray(eager_field), atol=1e-11)

    solver = wavebit.rcwa(structure=np.ones((1, 3, 3)), **kw)
    assigned, assigned_field = solver.solve_field(3, 3, 2, geometry=geometry,
                                                 structure=BOX_LAYER)
    np.testing.assert_allclose(np.asarray(assigned.T), np.asarray(compiled.T), atol=1e-11)
    np.testing.assert_allclose(np.asarray(assigned_field), np.asarray(field), atol=1e-11)
    assert solver.structure is BOX_LAYER
