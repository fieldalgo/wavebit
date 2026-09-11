"""An eager JAX time loop compiles each scan program once, not once per call.

Measured 2026-09-11 at 154faa2: every eager ``wavebit.fdtd(backend=1)`` scene solve ran XLA 8
times -- two source bases, each with a reference and an actual ``fdtd_grid`` run, each a
checkpointed chunk scan plus a tail scan -- and ran all 8 again on every later call, though the
solve holds 2 distinct programs. An eager ``jax.lax.scan`` caches its executable on the traced
body jaxpr, which hashes by identity, and ``fdtd_grid`` builds a new ``step`` closure per call.
``backends.time_domain._eager_scan`` keys the compiled program on the jaxpr's printed form.
"""
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
    The resolution is one no other test uses, so the count cannot come from a warm cache."""
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
