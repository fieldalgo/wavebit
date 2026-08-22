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
    """
    import jax
    from wavebit.on_jax.emsolver.rcwa import RCWAJax

    jax.clear_caches()
    solver = _solver(1, '1d_conical')          # ('sampled', 'li'): records nothing at all
    geometry = solver.freeze_geometry()
    before = RCWAJax.solve_field._cache_size()

    solver.solve_field(sample_x=8, sample_y=1, sample_z=4, geometry=geometry)
    solver.solve_field(sample_x=8, sample_y=1, sample_z=4, geometry=geometry)
    assert RCWAJax.solve_field._cache_size() == before + 1

    # Freezing again on the same structure records the same thing, so it is the same program.
    solver.solve_field(sample_x=8, sample_y=1, sample_z=4, geometry=solver.freeze_geometry())
    assert RCWAJax.solve_field._cache_size() == before + 1

    # A record carrying arrays cannot compare by content, and says so by compiling again.
    nvf = _solver(1, '1d_conical', factorization='nvf')
    nvf.solve_field(sample_x=8, sample_y=1, sample_z=4, geometry=nvf.freeze_geometry())
    nvf.solve_field(sample_x=8, sample_y=1, sample_z=4, geometry=nvf.freeze_geometry())
    assert RCWAJax.solve_field._cache_size() == before + 3


def test_resolution_and_field_selection_are_static():
    """They set the output shape, so they must be compile-time constants, not traced."""
    solver = _solver(1, '2d')
    geometry = solver.freeze_geometry()

    _, one = solver.solve_field(sample_x=4, sample_y=4, sample_z=4, geometry=geometry)
    _, three = solver.solve_field(sample_x=4, sample_y=4, sample_z=4, geometry=geometry,
                                       incidences=('given', 's', 'p'))

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
