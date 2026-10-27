"""An eager JAX time loop compiles each scan program once, not once per call.

Measured 2026-09-11 at 154faa2: every eager ``wavebit.fdtd(backend=1)`` scene solve ran XLA 8
times -- two source bases, each with a reference and an actual ``fdtd_grid`` run, each a
checkpointed chunk scan plus a tail scan -- and ran all 8 again on every later call, though the
solve holds 2 distinct programs. An eager ``jax.lax.scan`` caches its executable on the traced
body, which hashes by identity, and ``fdtd_grid`` built a new ``step`` closure per call.
``fdtd_grid`` now hands the loop a ``runtime.Step`` -- a module-level builder, hashable settings
and the call's arrays -- and ``JaxTimeLoop._run`` runs it through a plain ``jax.jit`` keyed on
static values, shapes and dtypes alone, with no reading of JAX's traced program.
"""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd

from wavebit.fdtd._periodic.grid import fdtd_grid as _legacy_wavebit_fdtd_grid
import gc
import subprocess
import sys
import textwrap
import weakref

import numpy as np
import pytest

from wavebit.fdtd._periodic import runtime as time_domain

jax = pytest.importorskip('jax')

BACKEND_COMPILE = '/jax/core/compile/backend_compile_duration'


class CompileLog:
    """Names of the XLA compilations run inside the block, read from ``jax.monitoring``."""

    def __enter__(self):
        self.names = []

        def listener(event, duration, fun_name=None, **kwargs):
            if event == BACKEND_COMPILE:
                self.names.append(fun_name)
        self._listener = listener
        jax.monitoring.register_event_duration_secs_listener(listener)
        return self

    def __exit__(self, *exc):
        jax.monitoring.unregister_event_duration_listener(self._listener)


def scene(**extra):
    kw = dict(structure=np.ones((1, 1, 2)), thickness=[.1], top=1., bot=1.,
              period=[.2], wavelength=.9, resolution=20., n_periods=20, backend='jax')
    kw.update(extra)
    return kw


def test_a_repeated_scene_solve_runs_no_compiler():
    """The measured case: nothing about the second solve is new except the permittivity."""
    _legacy_wavebit_fdtd(**scene())
    with CompileLog() as log:
        _legacy_wavebit_fdtd(**scene(structure=np.full((1, 1, 2), 2.25)))
    assert log.names == [], log.names


def test_one_solve_compiles_each_distinct_program_once():
    """Two source bases times reference/actual is four runs, each a chunk scan plus a tail scan.
    The two structure runs share one shape and the two uniform-reference runs share another --
    the reference is stepped on a two-cell transverse column (``scene_fdtd._uniform_reference``)
    -- so a solve at sizes nothing has compiled needs 4 programs, two per shape.
    The program cache is emptied first, so the count cannot come from a warm one; it used to
    rest on no other test in the session choosing resolution 23."""
    time_domain._SCAN_PROGRAMS.clear()
    with CompileLog() as log:
        _legacy_wavebit_fdtd(**scene(resolution=23.))
    assert log.names.count('jit(scan_program)') == 4, log.names


E0 = np.zeros((3, 16, 2, 2))
E0[1] = np.exp(-((np.arange(16)-8)/2.)**2)[:, None, None]


def test_a_repeated_recorded_subsampled_solve_runs_no_compiler():
    """A recorded solve runs its first step eagerly, with a Python index. With ``dft_subsample``
    above one that step's monitor update was an eager ``lax.cond`` on new branch closures, so
    every solve compiled ``jit(cond)`` again although the predicate was a known Python bool."""
    def solve(eps):
        return _legacy_wavebit_fdtd_grid(np.full((16, 2, 2), eps), spacing=(1., 1., 1.), dt=.2,
                                         steps=57, electric=E0, wavelengths=[20.], monitor_z=[3],
                                         checkpoint_steps=7, record_field=4, dft_subsample=3,
                                         backend='jax')
    solve(2.)
    with CompileLog() as log:
        solve(2.5)
    assert log.names == [], log.names


def grid(dt, eps):
    out = _legacy_wavebit_fdtd_grid(np.full((16, 2, 2), eps), spacing=(1., 1., 1.), dt=dt, steps=43,
                            electric=E0, wavelengths=[20., 24.], monitor_z=[3],
                            conductivity=.015, checkpoint_steps=7, backend='jax')
    return [np.asarray(out[key]) for key in ('E', 'H', 'electric_spectrum')]


def test_a_shared_program_returns_the_bytes_of_a_cold_compile():
    """Sharing is allowed only where it is exact, and bytes are the only proof of that.

    The third case differs from the first only in array values, so it runs the first case's
    program. The second differs only in ``dt``, a Python scalar the step bakes in as a literal:
    a key blind to literal values hands it the first case's program, and its fields come back
    wrong while every shape and dtype is right.
    """
    cases = [(.2, 2.), (.19, 2.), (.2, 2.5)]
    shared = [grid(*case) for case in cases]
    for case, hot in zip(cases, shared):
        time_domain._SCAN_PROGRAMS.clear()
        cold = grid(*case)
        for a, b in zip(hot, cold):
            assert (a.dtype, a.shape) == (b.dtype, b.shape) and a.tobytes() == b.tobytes(), case
    assert shared[0][0].tobytes() != shared[1][0].tobytes()


def test_changed_bloch_factors_remain_live_in_a_shared_scan_program():
    """Hoisted forward/inverse phases are this call's constants, never cached values."""
    phases = ((.1, -.2, .3), (.4, .15, -.25))

    def run(phase):
        out = _legacy_wavebit_fdtd_grid(np.full((16, 2, 2), 2.),
            spacing=(1.,)*3, dt=.2, steps=23, electric=E0,
            wavelengths=[20.], monitor_z=[0, 3], bloch_phase=np.array(phase),
            checkpoint_steps=7, backend='jax')
        return [np.asarray(out[key]) for key in ('E', 'H', 'electric_spectrum', 'magnetic_spectrum')]

    time_domain._SCAN_PROGRAMS.clear()
    first = run(phases[0])
    programs = set(time_domain._SCAN_PROGRAMS)
    assert programs
    shared = run(phases[1])
    assert set(time_domain._SCAN_PROGRAMS) == programs
    assert first[0].tobytes() != shared[0].tobytes()
    time_domain._SCAN_PROGRAMS.clear()
    cold = run(phases[1])
    for hot, fresh in zip(shared, cold):
        assert (hot.dtype, hot.shape) == (fresh.dtype, fresh.shape)
        assert hot.tobytes() == fresh.tobytes()


def _scale(runtime, settings, arrays):
    """A minimal step builder: ``carry * factor + bias + n``, with a static ``shift``."""
    def step(carry, n):
        return carry * arrays['factor'] + arrays['bias'] + n + settings, None
    return step


def _run(arrays, settings=0., init=None, length=3, kind='steps'):
    import jax.numpy as jnp
    loop = time_domain.JaxTimeLoop()
    step = time_domain.Step(_scale, loop, settings, arrays)
    init = jnp.zeros(2) if init is None else init
    xs = jnp.arange(float(length))
    return loop._run(kind, step, init, xs)[0]


def _arrays(factor=.5, bias=(1., 2.)):
    import jax.numpy as jnp
    return dict(factor=factor, bias=jnp.asarray(bias))


def test_a_plain_function_runs_the_plain_scan_and_caches_nothing():
    """Only a ``Step`` is shared. Any other function -- a user's closure, a custom_jvp or a
    callback in the body -- runs ``lax.scan`` as it always did: exact, and compiled per call."""
    import jax.numpy as jnp

    @jax.custom_jvp
    def g(x):
        return jnp.sin(x)
    g.defjvp(lambda primal, tangent: (g(primal[0]), jnp.cos(primal[0]) * tangent[0]))

    def step(c, n):
        c = c + jax.pure_callback(lambda x: np.asarray(x) * 2,
                                  jax.ShapeDtypeStruct(c.shape, c.dtype), c)
        return c + g(c + n), None
    time_domain._SCAN_PROGRAMS.clear()
    loop = time_domain.JaxTimeLoop()
    got = loop._run('steps', step, jnp.ones(3), jnp.arange(2.))[0]
    want = jax.lax.scan(step, jnp.ones(3), jnp.arange(2.))[0]
    assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
    assert not time_domain._SCAN_PROGRAMS


def test_a_step_built_with_a_custom_rule_or_callback_is_shared_exactly():
    """With the jaxpr no longer read, a body holding a custom_jvp or a callback is no longer a
    special case: the builder and settings are the key, and the rule is rebuilt inside the jit."""
    import jax.numpy as jnp

    def build(runtime, settings, arrays):
        import jax.numpy as jnp

        @jax.custom_jvp
        def g(x):
            return jnp.sin(x)
        g.defjvp(lambda primal, tangent: (g(primal[0]), jnp.cos(primal[0]) * tangent[0]))

        def step(c, n):
            c = c + jax.pure_callback(lambda x: np.asarray(x) * 2,
                                      jax.ShapeDtypeStruct(c.shape, c.dtype), c)
            return c * arrays['a'] + g(c + n), None
        return step
    time_domain._SCAN_PROGRAMS.clear()
    loop = time_domain.JaxTimeLoop()
    for a in (jnp.full(3, .5), jnp.full(3, .25)):
        step = time_domain.Step(build, loop, None, dict(a=a))
        got = loop._run('steps', step, jnp.ones(3), jnp.arange(2.))[0]
        want = jax.lax.scan(build(loop, None, dict(a=a)), jnp.ones(3), jnp.arange(2.))[0]
        assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
    assert len(time_domain._SCAN_PROGRAMS) == 1


def test_a_scalar_leaf_or_setting_is_part_of_the_program():
    """A Python scalar is baked into the program, as the closure it replaced baked it in. So it
    is in the key, by type and bit pattern: ``-0.0`` is not ``0.0``, and ``1``, ``1.0`` and
    ``True`` -- equal and equally hashed in Python -- are three programs, not one."""
    key = time_domain._static_key
    assert key(0.) != key(-0.)
    assert len({key(1), key(1.), key(True)}) == 3
    assert key((1., 2)) == key((1., 2)) and key((1., 2)) != key([1., 2])
    with pytest.raises(TypeError):
        key(np.array([1., 2.]))
    with pytest.raises(TypeError):
        key({})
    time_domain._SCAN_PROGRAMS.clear()
    results = [np.asarray(_run(_arrays(factor=f), settings=s)) for f, s in
               ((.5, 0.), (.25, 0.), (.5, -0.), (.5, 1.))]
    assert len(time_domain._SCAN_PROGRAMS) == 4
    assert results[0].tobytes() != results[1].tobytes()
    assert results[0].tobytes() != results[3].tobytes()


def test_unkeyable_settings_run_the_plain_scan():
    """Settings that cannot be a key cost the sharing, never the answer."""
    import jax.numpy as jnp

    def build(runtime, settings, arrays):
        return lambda c, n: (c + settings['shift'] + n, None)
    time_domain._SCAN_PROGRAMS.clear()
    loop = time_domain.JaxTimeLoop()
    step = time_domain.Step(build, loop, {'shift': 1.}, {})
    got = loop._run('steps', step, jnp.zeros(2), jnp.arange(3.))[0]
    np.testing.assert_array_equal(np.asarray(got), [6., 6.])
    assert not time_domain._SCAN_PROGRAMS


def test_a_builder_closing_over_a_value_is_not_shared():
    """The key sees a builder by identity, not what it closes over. One builder reading an array
    from a closure cell would hand a second call the first call's array -- the old structural key
    measured [4, 8, 12] where [40, 80, 120] was asked (2026-09-12) -- so it runs unshared."""
    import jax.numpy as jnp
    loop = time_domain.JaxTimeLoop()
    held = [None]

    def build(runtime, settings, arrays):
        value = held[0]
        return lambda carry, n: (carry + value, None)
    time_domain._SCAN_PROGRAMS.clear()
    for value, want in (([1., 2., 3.], [4., 8., 12.]), ([10., 20., 30.], [40., 80., 120.])):
        held[0] = jnp.array(value)
        step = time_domain.Step(build, loop, None, {})
        got = loop._run('steps', step, jnp.zeros(3), jnp.arange(4.))[0]
        np.testing.assert_array_equal(np.asarray(got), want)
    assert not time_domain._SCAN_PROGRAMS


def test_each_scan_length_is_its_own_entry():
    """One entry per trip count. Without the length in the key, a sweep of tail lengths grew a
    compiled copy per length inside one entry's jit, where the cache's bound never saw it."""
    time_domain._SCAN_PROGRAMS.clear()
    for length in (3, 5, 3):
        _run(_arrays(), length=length)
    assert len(time_domain._SCAN_PROGRAMS) == 2


def test_the_cache_stays_within_its_bound():
    """Least-recently-used eviction holds the cache at its bound. A hit keeps its entry -- the
    same compiled program, not a recompiled copy, which is what FIFO eviction would leave -- and
    the newest entry is never the one evicted, which is what MRU eviction would do."""
    time_domain._SCAN_PROGRAMS.clear()
    _run(_arrays(), settings=0.)
    first = next(iter(time_domain._SCAN_PROGRAMS))
    first_program = time_domain._SCAN_PROGRAMS[first]
    last = None
    for k in range(time_domain._SCAN_PROGRAMS_MAX + 3):
        before = set(time_domain._SCAN_PROGRAMS)
        _run(_arrays(), settings=k + 1.)
        added = set(time_domain._SCAN_PROGRAMS) - before
        assert len(added) == 1, f'call {k} left no entry of its own: the newest was evicted'
        (last,) = added
        _run(_arrays(), settings=0.)   # keep the hot entry recent
    assert len(time_domain._SCAN_PROGRAMS) == time_domain._SCAN_PROGRAMS_MAX
    assert time_domain._SCAN_PROGRAMS.get(first) is first_program
    assert last in time_domain._SCAN_PROGRAMS


def test_a_program_keeps_none_of_the_arrays_it_ran_on():
    """Arrays are arguments of the compiled program, never part of it or of its key."""
    import jax.numpy as jnp
    time_domain._SCAN_PROGRAMS.clear()
    big = jnp.arange(1000.)
    ref = weakref.ref(big)
    loop = time_domain.JaxTimeLoop()
    step = time_domain.Step(lambda rt, s, a: (lambda c, n: (c + a['big'][:2], None)),
                            loop, None, dict(big=big))
    loop._run('steps', step, jnp.zeros(2), jnp.arange(2.))
    del big, step
    gc.collect()
    assert len(time_domain._SCAN_PROGRAMS) == 1
    assert ref() is None, 'the cached program still holds an array it ran on'


def test_a_shifted_step_counts_from_its_offset():
    """``Step.shifted`` is ``step(state, n + start)`` -- the record path's final tail -- with
    ``start`` in the key, so two offsets are two programs and neither sees the other's."""
    import jax.numpy as jnp
    time_domain._SCAN_PROGRAMS.clear()
    loop = time_domain.JaxTimeLoop()
    step = time_domain.Step(_scale, loop, 0., dict(factor=1., bias=jnp.zeros(2)))
    for start in (0, 4):
        moved = step.shifted(start)
        got = loop._run('steps', moved, jnp.zeros(2), jnp.arange(3.))[0]
        np.testing.assert_array_equal(np.asarray(got), [3. + 3 * start] * 2)
        assert np.asarray(moved(jnp.zeros(2), 1.)[0]).tolist() == [1. + start] * 2
    assert len(time_domain._SCAN_PROGRAMS) == 2


def test_the_cache_is_changed_under_its_lock(monkeypatch):
    """Threads share the cache. Without the lock, six threads over a two-entry cache raised
    `dictionary changed size during iteration` within a second (measured 2026-09-13). A race is
    a poor test, so what is pinned is that the shared path takes the lock, exactly once."""
    class Counting:
        def __init__(self):
            self.entered = 0

        def __enter__(self):
            self.entered += 1

        def __exit__(self, *exc):
            return False
    lock = Counting()
    monkeypatch.setattr(time_domain, '_SCAN_LOCK', lock)
    time_domain._SCAN_PROGRAMS.clear()
    _run(_arrays())
    assert len(time_domain._SCAN_PROGRAMS) == 1
    assert lock.entered == 1


def test_a_jitted_first_solve_traces_once_and_repeats_without_compiling():
    """From jax 0.12.0, importing ``jax.extend`` grows JAX's tracing context; the structural
    key used to import it during the first solve, so a user's ``jax.jit`` around that solve
    traced twice and the next eager solve compiled everything again (measured 2026-10-19). The
    loop now reads no JAX internals. A fresh process is the only place this can be seen."""
    script = textwrap.dedent('''
        import jax, numpy as np
        jax.config.update('jax_enable_x64', True)
        from wavebit.fdtd._periodic.grid import fdtd_grid
        names, traces = [], []
        jax.monitoring.register_event_duration_secs_listener(
            lambda event, duration, fun_name=None, **kw:
            names.append(fun_name) if event == '/jax/core/compile/backend_compile_duration' else None)
        E0 = np.zeros((3, 8, 2, 2)); E0[1, 4] = 1.
        def solve(eps):
            traces.append(None)
            return fdtd_grid(eps, spacing=(1., 1., 1.), dt=.2, steps=9, electric=E0,
                             checkpoint_steps=4, backend='jax')['E']
        run = jax.jit(solve)
        run(np.full((8, 2, 2), 2.)); run(np.full((8, 2, 2), 2.5))
        assert len(traces) == 1, traces
        solve(np.full((8, 2, 2), 2.)); before = len(names)
        solve(np.full((8, 2, 2), 3.))
        assert names[before:] == [], names[before:]
        print('ok')
    ''')
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                         encoding='utf-8', timeout=600)
    assert out.returncode == 0 and out.stdout.strip() == 'ok', out.stderr[-3000:]


def test_a_solve_inside_a_trace_runs_the_plain_scan():
    """Inside a user's ``jax.jit`` every leaf is a tracer and nothing is dispatched, so the plain
    scan runs and nothing is cached: a nested jit there would change the user's program (same
    bytes, a few percent more temp memory on jax 0.12.0). The answer is the eager one."""
    jax.config.update('jax_enable_x64', True)
    def solve(eps):
        return _legacy_wavebit_fdtd_grid(eps, spacing=(1., 1., 1.), dt=.2, steps=43,
                                         electric=E0, wavelengths=[20., 24.], monitor_z=[3],
                                         checkpoint_steps=7, backend='jax')['electric_spectrum']
    eps = np.full((16, 2, 2), 2.)
    time_domain._SCAN_PROGRAMS.clear()
    traced = np.asarray(jax.jit(solve)(eps))
    assert not time_domain._SCAN_PROGRAMS
    np.testing.assert_allclose(traced, np.asarray(solve(eps)), rtol=1e-13, atol=1e-15)
