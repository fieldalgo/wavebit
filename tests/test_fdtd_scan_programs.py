"""An eager JAX time loop compiles each scan program once, not once per call.

Measured 2026-09-11 at 154faa2: every eager ``wavebit.fdtd(backend=1)`` scene solve ran XLA 8
times -- two source bases, each with a reference and an actual ``fdtd_grid`` run, each a
checkpointed chunk scan plus a tail scan -- and ran all 8 again on every later call, though the
solve holds 2 distinct programs. An eager ``jax.lax.scan`` caches its executable on the traced
body jaxpr, which hashes by identity, and ``fdtd_grid`` builds a new ``step`` closure per call.
``backends.time_domain._eager_scan`` keys the compiled program on the jaxpr's structure
(``_program_key``) and the scan's length.
"""
import gc
import warnings
import weakref

import numpy as np
import pytest

import wavebit
from wavebit.backends import time_domain

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
              period=[.2], wavelength=.9, resolution=20., n_periods=20, backend=1)
    kw.update(extra)
    return kw


def test_a_repeated_scene_solve_runs_no_compiler():
    """The measured case: nothing about the second solve is new except the permittivity."""
    wavebit.fdtd(**scene())
    with CompileLog() as log:
        wavebit.fdtd(**scene(structure=np.full((1, 1, 2), 2.25)))
    assert log.names == [], log.names


def test_one_solve_compiles_each_distinct_program_once():
    """Two source bases times reference/actual is four runs of one step program, and each run
    is a chunk scan plus a tail scan: a solve at a size nothing has compiled needs 2 programs.
    The program cache is emptied first, so the count cannot come from a warm one; it used to
    rest on no other test in the session choosing resolution 23."""
    time_domain._SCAN_PROGRAMS.clear()
    with CompileLog() as log:
        wavebit.fdtd(**scene(resolution=23.))
    assert log.names.count('jit(scan_program)') == 2, log.names


E0 = np.zeros((3, 16, 2, 2))
E0[1] = np.exp(-((np.arange(16)-8)/2.)**2)[:, None, None]


def grid(dt, eps):
    out = wavebit.fdtd_grid(np.full((16, 2, 2), eps), spacing=(1., 1., 1.), dt=dt, steps=43,
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


def _inner_jit_scan(c):
    """An eager scan whose body calls an inner ``jax.jit`` that closes over the array ``c``."""
    import jax.numpy as jnp
    inner = jax.jit(lambda x: x * c)
    return time_domain._eager_scan(lambda carry, n: (carry + inner(jnp.ones(3)), None),
                                   jnp.zeros(3), jnp.arange(4))[0]


def test_an_inner_jit_closing_over_an_array_is_not_shared():
    """jax 0.11 made ``ClosedJaxpr`` and ``Jaxpr`` one class, so the constants an inner
    ``jax.jit`` carries arrive on a nested *Jaxpr*. A key that reads them only on a
    ``ClosedJaxpr`` branch never reads them, and a second call runs on the first call's array:
    measured 2026-09-12, [4, 8, 12] came back where [40, 80, 120] was asked."""
    import jax.numpy as jnp
    time_domain._SCAN_PROGRAMS.clear()
    first = _inner_jit_scan(jnp.array([1., 2., 3.]))
    second = _inner_jit_scan(jnp.array([10., 20., 30.]))
    np.testing.assert_array_equal(np.asarray(first), [4., 8., 12.])
    np.testing.assert_array_equal(np.asarray(second), [40., 80., 120.])
    assert not time_domain._SCAN_PROGRAMS


def test_a_function_valued_parameter_is_not_shared():
    """A custom_jvp's rule is rebuilt by every trace and hashes by identity: keyed, every call
    would compile anew and push a program nothing asks for again into the bounded cache."""
    import jax.numpy as jnp

    @jax.custom_jvp
    def g(x):
        return jnp.sin(x)
    g.defjvp(lambda primal, tangent: (g(primal[0]), jnp.cos(primal[0]) * tangent[0]))

    def step(c, n):
        return c + g(c + n), None
    time_domain._SCAN_PROGRAMS.clear()
    got = time_domain._eager_scan(step, jnp.zeros(3), jnp.arange(2.))[0]
    want = jax.lax.scan(step, jnp.zeros(3), jnp.arange(2.))[0]
    assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
    assert not time_domain._SCAN_PROGRAMS


def test_a_weakly_typed_carry_runs_unshared():
    """``lax.scan`` promotes a weakly typed carry and retraces; a shared program would not."""
    time_domain._SCAN_PROGRAMS.clear()
    out = time_domain._eager_scan(lambda c, n: (c + n, None), 0., np.arange(3.))[0]
    assert float(out) == 3.
    assert not time_domain._SCAN_PROGRAMS


def test_each_scan_length_is_its_own_entry():
    """One entry per trip count. Without the length in the key, a sweep of tail lengths grew a
    compiled copy per length inside one entry's jit, where the cache's bound never saw it."""
    import jax.numpy as jnp

    def step(c, n):
        return c * .5 + n, None
    time_domain._SCAN_PROGRAMS.clear()
    for length in (3, 5, 3):
        time_domain._eager_scan(step, jnp.ones(2), jnp.arange(length, dtype=float))
    assert len(time_domain._SCAN_PROGRAMS) == 2


def test_the_cache_stays_within_its_bound():
    """Least-recently-used eviction holds the cache at its bound. A hit keeps its entry -- the
    same compiled program, not a recompiled copy, which is what FIFO eviction would leave -- and
    the newest entry is never the one evicted, which is what MRU eviction would do."""
    import jax.numpy as jnp

    def hot(c, n):
        return c + 0., None
    time_domain._SCAN_PROGRAMS.clear()
    time_domain._eager_scan(hot, jnp.zeros(1), jnp.arange(1.))
    first = next(iter(time_domain._SCAN_PROGRAMS))
    first_program = time_domain._SCAN_PROGRAMS[first]
    last = None
    for k in range(time_domain._SCAN_PROGRAMS_MAX + 3):
        before = set(time_domain._SCAN_PROGRAMS)
        time_domain._eager_scan(lambda c, n, k=k: (c + (k + 1.), None), jnp.zeros(1),
                                jnp.arange(1.))
        added = set(time_domain._SCAN_PROGRAMS) - before
        assert len(added) == 1, f'call {k} left no entry of its own: the newest was evicted'
        (last,) = added
        time_domain._eager_scan(hot, jnp.zeros(1), jnp.arange(1.))   # keep the hot entry recent
    assert len(time_domain._SCAN_PROGRAMS) == time_domain._SCAN_PROGRAMS_MAX
    assert time_domain._SCAN_PROGRAMS.get(first) is first_program
    assert last in time_domain._SCAN_PROGRAMS


def _body_closed_over_a_big_array():
    """``(weak reference, step function)``. The array lives only in the closure, so deleting
    the returned function drops the last reference this test holds."""
    import jax.numpy as jnp
    c = jnp.arange(1000.)
    return weakref.ref(c), lambda carry, n: (carry + c[:2], None)


def test_a_program_keeps_none_of_the_arrays_it_was_first_traced_with():
    """jax 0.11's ``closed.jaxpr`` is ``closed`` itself, constants included, and the cached
    program closed over it: up to 64 programs, each holding a first call's arrays."""
    import jax.numpy as jnp
    time_domain._SCAN_PROGRAMS.clear()
    ref, body = _body_closed_over_a_big_array()
    time_domain._eager_scan(body, jnp.zeros(2), jnp.arange(2.))
    del body
    gc.collect()
    assert len(time_domain._SCAN_PROGRAMS) == 1
    assert ref() is None, 'the cached program still holds the array it was traced with'


def test_a_jax_that_moved_a_name_the_key_reads_costs_the_sharing_not_the_solve(monkeypatch):
    """Every eager solve used to depend on ``jax.extend.core``, ``jax.typeof`` and
    ``jax.core.Tracer`` staying where they are; a JAX that moves one now warns once and scans."""
    import jax.numpy as jnp
    monkeypatch.setattr(time_domain, '_jax_api', lambda: None)
    monkeypatch.setattr(time_domain, '_API_WARNED', [])
    time_domain._SCAN_PROGRAMS.clear()

    def step(c, n):
        return c + n, None
    with pytest.warns(RuntimeWarning, match='eager-scan program cache'):
        out = time_domain._eager_scan(step, jnp.zeros(()), jnp.arange(3.))[0]
    assert float(out) == 3.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        time_domain._eager_scan(step, jnp.zeros(()), jnp.arange(3.))
    assert not time_domain._SCAN_PROGRAMS


def test_a_parameter_enters_the_key_by_its_bytes():
    """An array or float parameter is compared by bit pattern -- ``-0.0`` is not ``0.0``, one
    element changed is another program -- and a function-valued one cannot be keyed at all."""
    key = time_domain._param_key
    assert key(np.array([1., 2.])) == key(np.array([1., 2.]))
    assert key(np.array([1., 2.])) != key(np.array([1., 3.]))
    assert key(0.) != key(-0.)
    assert key(np.float32(1)) != key(np.float64(1))
    with pytest.raises(time_domain._Unshareable):
        key(lambda x: x)


def test_the_resolver_turns_a_missing_name_into_the_fallback(monkeypatch):
    """The resolver itself has to catch a moved name. The test above replaces it wholesale, so
    its ``except`` never ran, and a resolver catching only ``ImportError`` passed every test
    while a JAX without ``jax.typeof`` crashed every eager solve. ``__wrapped__`` bypasses the
    resolver's cache."""
    monkeypatch.delattr(jax, 'typeof')
    assert time_domain._jax_api.__wrapped__() is None


def test_the_cache_is_changed_under_its_lock(monkeypatch):
    """Threads share the cache. Without the lock, six threads over a two-entry cache raised
    `dictionary changed size during iteration` within a second (measured 2026-09-13). A race is
    a poor test, so what is pinned is that the shared path takes the lock, exactly once."""
    import jax.numpy as jnp

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
    time_domain._eager_scan(lambda c, n: (c * 2., None), jnp.ones(2), jnp.arange(3.))
    assert len(time_domain._SCAN_PROGRAMS) == 1
    assert lock.entered == 1


def test_a_callback_in_the_body_is_not_shared():
    """A callback primitive carries its function in a frozen dataclass that hashes and compares
    the function by identity; keyed, it would split every call apart like a bare function."""
    import jax.numpy as jnp

    def host(x):
        return np.asarray(x) * 2

    def step(c, n):
        return c + jax.pure_callback(host, jax.ShapeDtypeStruct(c.shape, c.dtype), c), None
    time_domain._SCAN_PROGRAMS.clear()
    got = time_domain._eager_scan(step, jnp.ones(2), jnp.arange(2.))[0]
    want = jax.lax.scan(step, jnp.ones(2), jnp.arange(2.))[0]
    assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
    assert not time_domain._SCAN_PROGRAMS
