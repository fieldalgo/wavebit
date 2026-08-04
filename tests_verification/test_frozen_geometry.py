"""Frozen geometry makes the public API traceable (fieldwave.StaticGeometry).

Three pipeline steps decide something structural rather than numeric — the same-material
short-circuit, the NVF normal field, the CFS block compression. All branch on values, so none
survives a JAX trace. ``freeze_geometry()`` takes those decisions once against the concrete
structure; handing the result back to ``conv_solve`` makes the whole call jit- and
grad-traceable without changing a single number.
"""
import numpy as np
import pytest

import fieldwave
from fieldwave import FourierType

UCELL = np.random.default_rng(0).uniform(1.0, 4.0, (1, 5, 5))
KW = dict(n_top=1.0, n_bot=1.0, theta=0.1, phi=0.2, psi=0.3, fto=[2, 2],
          period=[1.0, 1.0], wavelength=0.9, thickness=[0.2])
STRATEGIES = [FourierType.DFS, FourierType.CONTINUOUS, FourierType.NVF]
IDS = ['dfs', 'cfs', 'nvf']


@pytest.mark.parametrize('fourier_type', STRATEGIES, ids=IDS)
def test_frozen_geometry_does_not_change_the_answer(fourier_type):
    """Freezing is an optimization, not a different computation."""
    solver = fieldwave.rcwa(backend=0, ucell=UCELL, fourier_type=fourier_type, **KW)
    auto = solver.conv_solve()
    frozen = solver.conv_solve(geometry=solver.freeze_geometry())
    assert np.array_equal(np.asarray(auto.res.de_ri), np.asarray(frozen.res.de_ri))
    assert np.array_equal(np.asarray(auto.res.de_ti), np.asarray(frozen.res.de_ti))


@pytest.mark.parametrize('fourier_type', STRATEGIES, ids=IDS)
def test_jit_and_grad_through_the_public_api(fourier_type):
    """jax.jit and jax.grad over rcwa/conv_solve, for every Fourier strategy."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    solver = fieldwave.rcwa(backend=1, ucell=UCELL, fourier_type=fourier_type, **KW)
    geometry = solver.freeze_geometry()

    def loss(u):
        return jnp.sum(solver.conv_solve(ucell=u, geometry=geometry).res.de_ti).real

    u = jnp.asarray(UCELL)
    eager = float(loss(u))
    assert np.isclose(float(jax.jit(loss)(u)), eager, atol=1e-12, rtol=0)

    g = np.asarray(jax.grad(loss)(u))
    assert g.shape == UCELL.shape and np.isfinite(g).all()

    # finite-difference spot check along the all-ones direction
    d = np.ones_like(UCELL)
    h = 1e-6
    fd = (float(loss(jnp.asarray(UCELL + h * d))) - float(loss(jnp.asarray(UCELL - h * d)))) / (2 * h)
    assert abs(float(np.sum(g * d)) - fd) < 1e-5, (float(np.sum(g * d)), fd)


@pytest.mark.parametrize('fourier_type', STRATEGIES, ids=IDS)
def test_auto_detection_still_blocks_jit(fourier_type):
    """Pins *why* the API exists: without frozen geometry, tracing fails.

    Auto-detection is the eager path that reproduces the legacy result; it is not traceable.
    (DFS survives ``jax.grad`` because its detection narrows a tracer whose primal is
    concrete, but no strategy survives ``jax.jit``.)
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    solver = fieldwave.rcwa(backend=1, ucell=UCELL, fourier_type=fourier_type, **KW)

    def loss(u):
        return jnp.sum(solver.conv_solve(ucell=u).res.de_ti).real

    with pytest.raises(Exception):
        jax.jit(loss)(jnp.asarray(UCELL))


def test_freeze_geometry_reports_what_the_strategy_needs():
    """Only the fields the active strategy consumes are populated."""
    def geo(fourier_type):
        return fieldwave.rcwa(backend=0, ucell=UCELL, fourier_type=fourier_type,
                               **KW).freeze_geometry()

    dfs = geo(FourierType.DFS)
    assert dfs.same_material is not None
    assert dfs.normal_field is None and dfs.compression is None

    assert geo(FourierType.NVF).normal_field is not None
    assert geo(FourierType.CONTINUOUS).compression is not None


def test_same_material_flags_are_correct():
    """A uniform bottom layer matching the substrate is detected; a patterned one is not."""
    ucell = np.array([[[3., 3., 1., 1., 3., 1.]], [[1., 1., 1., 1., 1., 1.]]])
    solver = fieldwave.rcwa(backend=0, ucell=ucell, pol=0, phi=None, n_top=1.0, n_bot=1.0,
                          theta=0.0, fto=[5, 0], period=[0.7], wavelength=0.9,
                          thickness=[0.2, 0.1])
    assert solver.freeze_geometry().same_material == [False, True]
