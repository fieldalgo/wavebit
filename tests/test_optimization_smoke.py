"""An optimizer loop has to run, reduce the loss, and do it the same way on both backends.

``test_gradients.py`` already checks single gradients against finite differences, which is the
harder numerical question. What it does not check is the loop: repeated ``solve`` calls
through a real optimizer, where state is carried between steps and a NaN or a silently
detached graph shows up as "it optimized nothing" rather than as a wrong number.

That is what ``QA/autodiff_raster1.py``, ``autodiff_raster2.py`` and ``autodiff_vector.py``
were for. They ran 10-20 steps and printed the loss for a human to glance at; nothing recorded
what counted as progress, and nothing ran them.

The third test here is the one worth having: the same structure, the same initial condition
and the same optimizer settings on JAX and PyTorch must produce the same loss *trajectory*.
A single solve agreeing across backends is already covered; a trajectory agreeing means the
gradients agree too, step after step, which is where a backend-specific defect would show.
"""
import numpy as np
import pytest

import wavebit

CFG = dict(max_order=[2, 2], wavelength=900.0, period=[1000.0, 1000.0], thickness=[300.0],
           top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=np.pi / 2)

# Small enough that this stays a smoke test. The grid is a smooth ramp rather than random so
# the starting point is reproducible without seeding anything.
# A permittivity ramp: the squares of the index ramp 1.0 .. 3.5 this used to start from.
UCELL0 = np.linspace(1.0, 3.5, 6 * 6).reshape(1, 6, 6) ** 2

STEPS = 4
LR = 0.1


def _jax_losses():
    import jax
    import jax.numpy as jnp
    import optax

    solver = wavebit.rcwa(backend=1, **CFG)

    def loss(u):
        return -jnp.sum(solver.solve(structure=u).given.T)

    u = jnp.asarray(UCELL0)
    opt = optax.adam(LR)
    state = opt.init(u)

    out = []
    for _ in range(STEPS):
        value, grad = jax.value_and_grad(loss)(u)
        assert jnp.all(jnp.isfinite(grad)), 'jax gradient went non-finite'
        out.append(float(value))
        updates, state = opt.update(grad, state)
        u = optax.apply_updates(u, updates)
    return out


def _torch_losses():
    import torch

    solver = wavebit.rcwa(backend=2, **CFG)

    u = torch.tensor(UCELL0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([u], lr=LR)

    out = []
    for _ in range(STEPS):
        opt.zero_grad()
        value = -torch.sum(solver.solve(structure=u).given.T)
        value.backward()
        assert torch.all(torch.isfinite(u.grad)), 'torch gradient went non-finite'
        out.append(float(value.detach()))
        opt.step()
    return out


@pytest.mark.parametrize('name', ['jax', 'torch'])
def test_the_optimizer_actually_reduces_the_loss(name):
    """Finite gradients are not enough — the loop has to make progress.

    A detached graph, a zeroed gradient or an optimizer fed the wrong tree all leave the loss
    flat while every individual number stays perfectly finite.
    """
    pytest.importorskip(name)
    if name == 'jax':
        pytest.importorskip('optax')
    losses = _jax_losses() if name == 'jax' else _torch_losses()

    assert len(losses) == STEPS
    assert all(np.isfinite(losses)), losses
    assert losses[-1] < losses[0] - 1e-6, losses


def test_jax_and_torch_take_the_same_trajectory():
    """Same start, same optimizer, same structure — the two loops must agree step by step.

    This is stronger than agreeing on one solve: it only holds if the gradients agree at every
    step, so a backend-specific defect in the backward pass cannot hide behind a matching
    forward value.
    """
    pytest.importorskip('jax')
    pytest.importorskip('optax')
    pytest.importorskip('torch')

    jax_losses = _jax_losses()
    torch_losses = _torch_losses()

    assert np.allclose(jax_losses, torch_losses, atol=1e-9), (jax_losses, torch_losses)
