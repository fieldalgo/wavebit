"""``SolverJax.conv_solve_field`` — the jit-compiled solve-and-reconstruct entry point.

This method jits over ``self``, which is what makes it different from ``conv_solve``: the whole
structure arrives as tracers, so every discrete decision downstream — the grating-type dispatch,
the NVF and integrated-path geometry — has nothing concrete to branch on. It
therefore *requires* a ``StaticGeometry`` from ``freeze_geometry()``, and says so rather than
letting a raw ``TracerBoolConversionError`` escape.

The regression these tests pin: ``pol`` and ``psi`` are one degree of freedom carried twice —
``psi`` as a traced pytree child, ``pol`` as static aux data because the solve *dispatches* on
it. Unflattening applied ``psi`` last, whose setter recomputes ``pol``, so the static dispatch
key silently became a tracer and every caller of this method died on the first ``if pol == 0``.
"""
import numpy as np
import pytest

import fieldwave

RNG = np.random.default_rng(0)
UCELL_1D = RNG.uniform(1.0, 4.0, (1, 1, 8))
UCELL_2D = RNG.uniform(1.0, 4.0, (1, 5, 5))

BASE = dict(n_top=1.0, n_bot=1.0, theta=0.1, wavelength=900.0, thickness=[500.0])

# (kwargs, (res_x, res_y, res_z)) per grating type the dispatcher can pick.
CASES = {
    '1d_tetm': (dict(pol=0, phi=None, fto=[3], period=[1000.0], ucell=UCELL_1D), (8, 1, 4)),
    '1d_tm': (dict(pol=1, phi=None, fto=[3], period=[1000.0], ucell=UCELL_1D), (8, 1, 4)),
    '1d_conical': (dict(pol=0.3, phi=0.2, fto=[3], period=[1000.0], ucell=UCELL_1D), (8, 1, 4)),
    '2d': (dict(pol=0.3, phi=0.2, fto=[2, 2], period=[1000.0, 1000.0], ucell=UCELL_2D), (4, 4, 4)),
}


def _solver(backend, case, **extra):
    kwargs, _ = CASES[case]
    return fieldwave.rcwa(backend=backend, **BASE, **kwargs, **extra)


def _res(case):
    return CASES[case][1]


@pytest.fixture(autouse=True)
def _require_jax():
    pytest.importorskip('jax')


@pytest.mark.parametrize('case', list(CASES))
def test_jit_matches_the_eager_path(case):
    """The jitted method reproduces ``conv_solve_field_no_jit`` exactly, for every grating type.

    Freezing the geometry is an optimization, not a different computation — same contract as
    ``conv_solve(geometry=...)``.
    """
    solver = _solver(1, case)
    res_x, res_y, res_z = _res(case)

    jitted, field = solver.conv_solve_field(res_x=res_x, res_y=res_y, res_z=res_z,
                                            geometry=solver.freeze_geometry())
    eager, field_eager = solver.conv_solve_field_no_jit(res_x=res_x, res_y=res_y, res_z=res_z)

    assert np.allclose(np.asarray(field), np.asarray(field_eager), rtol=0, atol=1e-10)
    assert np.allclose(np.asarray(jitted.T), np.asarray(eager.T), rtol=0, atol=1e-12)
    assert np.allclose(np.asarray(jitted.R), np.asarray(eager.R), rtol=0, atol=1e-12)


@pytest.mark.parametrize('case', list(CASES))
def test_jit_matches_the_numpy_backend(case):
    """Physics check: the jitted JAX field agrees with NumPy, not merely with itself."""
    res_x, res_y, res_z = _res(case)

    jax_solver = _solver(1, case)
    _, jax_field = jax_solver.conv_solve_field(res_x=res_x, res_y=res_y, res_z=res_z,
                                               geometry=jax_solver.freeze_geometry())
    _, np_field = _solver(0, case).conv_solve_field(res_x=res_x, res_y=res_y, res_z=res_z)

    assert np.asarray(jax_field).shape == np.asarray(np_field).shape
    assert np.allclose(np.asarray(jax_field), np.asarray(np_field), rtol=1e-8, atol=1e-8)


def test_omitting_the_geometry_names_freeze_geometry():
    """The whole point of the guard: a directive, not a TracerBoolConversionError.

    Before this was added, every call to ``conv_solve_field`` on the JAX backend died inside the
    trace with a message about ``self[4]``, which points at nothing a caller can act on.
    """
    solver = _solver(1, '1d_conical')

    with pytest.raises(ValueError, match=r'freeze_geometry\(\)'):
        solver.conv_solve_field(res_x=8, res_y=1, res_z=4)

    # ...and it stays raised — a failed trace caches nothing, so the second call is not silently
    # served from the jit cache.
    with pytest.raises(ValueError, match=r'conv_solve_field_no_jit'):
        solver.conv_solve_field(res_x=8, res_y=1, res_z=4)


def test_no_jit_still_auto_detects_without_a_geometry():
    """``conv_solve_field_no_jit`` keeps ``geometry`` optional: eager code needs no freeze."""
    solver = _solver(1, '1d_conical')
    res_x, res_y, res_z = _res('1d_conical')

    auto, auto_field = solver.conv_solve_field_no_jit(res_x=res_x, res_y=res_y, res_z=res_z)
    frozen, frozen_field = solver.conv_solve_field_no_jit(res_x=res_x, res_y=res_y, res_z=res_z,
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
    result, field = solver.conv_solve_field(res_x=4, res_y=4, res_z=4,
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
    from fieldwave.on_jax.emsolver.rcwa import RCWAJax

    jax.clear_caches()
    solver = _solver(1, '1d_conical')          # ('sampled', 'li'): records nothing at all
    geometry = solver.freeze_geometry()
    before = RCWAJax.conv_solve_field._cache_size()

    solver.conv_solve_field(res_x=8, res_y=1, res_z=4, geometry=geometry)
    solver.conv_solve_field(res_x=8, res_y=1, res_z=4, geometry=geometry)
    assert RCWAJax.conv_solve_field._cache_size() == before + 1

    # Freezing again on the same structure records the same thing, so it is the same program.
    solver.conv_solve_field(res_x=8, res_y=1, res_z=4, geometry=solver.freeze_geometry())
    assert RCWAJax.conv_solve_field._cache_size() == before + 1

    # A record carrying arrays cannot compare by content, and says so by compiling again.
    nvf = _solver(1, '1d_conical', factorization='nvf')
    nvf.conv_solve_field(res_x=8, res_y=1, res_z=4, geometry=nvf.freeze_geometry())
    nvf.conv_solve_field(res_x=8, res_y=1, res_z=4, geometry=nvf.freeze_geometry())
    assert RCWAJax.conv_solve_field._cache_size() == before + 3


def test_resolution_and_field_selection_are_static():
    """They set the output shape, so they must be compile-time constants, not traced."""
    solver = _solver(1, '2d')
    geometry = solver.freeze_geometry()

    _, one = solver.conv_solve_field(res_x=4, res_y=4, res_z=4, geometry=geometry)
    _, three = solver.conv_solve_field(res_x=4, res_y=4, res_z=4, geometry=geometry,
                                       set_field_input=(True, True, True))

    assert np.asarray(one).shape == (1, 4, 4, 4, 6)
    assert np.asarray(three).shape == (3, 4, 4, 4, 6)


@pytest.mark.parametrize('case,pols', [('2d', (0.0, 0.3, 1.0)), ('1d_tetm', (0, 1))])
def test_changing_pol_is_not_served_from_a_stale_trace(case, pols):
    """Pinning ``pol`` static must not pin it *stale*.

    It is a jit cache key, so changing it has to recompile rather than silently reuse the trace
    built for the previous polarization — which would be a wrong answer, not a crash. Checked
    against the eager path at each value, and the values must genuinely differ from each other.
    """
    solver = _solver(1, case)
    res_x, res_y, res_z = _res(case)
    geometry = solver.freeze_geometry()

    fields = {}
    for pol in pols:
        solver.pol = pol
        _, jitted = solver.conv_solve_field(res_x=res_x, res_y=res_y, res_z=res_z,
                                            geometry=geometry)
        _, eager = solver.conv_solve_field_no_jit(res_x=res_x, res_y=res_y, res_z=res_z)
        assert np.allclose(np.asarray(jitted), np.asarray(eager), rtol=0, atol=1e-10), \
            f'jit disagrees with eager at pol={pol}'
        fields[pol] = np.asarray(jitted)

    for a, b in zip(pols, pols[1:]):
        assert np.abs(fields[a] - fields[b]).max() > 1e-6, \
            f'pol={a} and pol={b} produced the same field — a stale trace was reused'


def test_pol_stays_concrete_through_the_pytree_round_trip():
    """The root cause, pinned directly.

    ``pol`` is aux data precisely because the solve dispatches on it; ``psi`` stays a traced
    child so gradients still flow through the incident amplitudes. Both must hold at once.
    """
    import jax

    solver = _solver(1, '1d_tetm')
    seen = {}

    def peek(s):
        seen['pol'] = s.pol
        seen['psi'] = s.psi
        return s.ucell.sum()

    jax.jit(peek)(solver)

    assert seen['pol'] == 0
    assert not isinstance(seen['pol'], jax.core.Tracer), 'pol must survive the trace concrete'
    assert isinstance(seen['psi'], jax.core.Tracer), 'psi must stay differentiable'
