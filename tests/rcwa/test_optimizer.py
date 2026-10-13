"""The optimizer layer: `solver.grad` and `solver.fit`, on the backends that have them.

This file exists because there was nothing. `solver.grad` and `solver.fit` ship on JAX and
PyTorch and **no test called either** -- the only caller in the repository was
`tutorials/rcwa/02-autograd-and-optimization-jax.ipynb`, JAX only. Torch's `fit` had never been run
by anything here. Two defects were sitting in that gap, one per backend, and both answered the
gradient question with a number rather than an error:

* JAX returned the **conjugate**. `jax.value_and_grad` of a real loss over a complex parameter
  gives `dL/dx - i dL/dy`, and `fit` conjugated it in its own `step` while `grad` handed the
  raw value back. So the two shipped answers to one question were conjugates of each other, and
  `grad`'s -- the one the tutorial prints for a reader to step along -- went uphill.
* Torch **accumulated**. `.grad` was never cleared, so the same question asked twice answered
  twice the truth.

Neither shows up on a real-valued parameter (`conj` of a real array is itself; and nobody asked
twice), which is why a thickness sweep never caught them.

The oracle is finite differences on the NumPy backend, which shares no code with either
autodiff path. Both partials are measured separately, because the whole defect on JAX lives in
the sign of the imaginary one.
"""

from wavebit.compat import legacy as _legacy_wavebit
import functools

import numpy as np
import pytest


#: A lossy grating -- complex permittivity, which is the ordinary case for a metal or an
#: absorber and the only case either defect is visible in.
LOSSY = np.array([[[1.0, 1.0, 4.0 + 0.6j, 4.0 + 0.6j, 4.0 + 0.6j, 1.0, 1.0, 1.0]]])
SCENE = dict(period=[1500.0, 1500.0], truncation_order=[2, 0], thickness=[150.0],
             wavelength=900.0, top=1.0, bot=1.0)
PIXEL = (0, 0, 2)


def _loss_at(value):
    """`sum(T)` with one pixel's permittivity replaced -- the finite-difference oracle."""
    eps = LOSSY.copy()
    eps[PIXEL] = value
    return float(np.asarray(_legacy_wavebit.rcwa(backend='numpy', structure=eps, **SCENE).solve().T).sum())


def _true_gradient(step=1e-6):
    """`dL/dx + i dL/dy` at the base point, the steepest-ascent direction."""
    z = LOSSY[PIXEL]
    return complex((_loss_at(z + step) - _loss_at(z - step)) / (2 * step),
                   (_loss_at(z + 1j * step) - _loss_at(z - 1j * step)) / (2 * step))


def _solver_and_gradient(backend):
    if backend == 'jax':
        jnp = pytest.importorskip('jax.numpy')
        _legacy_wavebit.enable_x64()
        solver = _legacy_wavebit.rcwa(backend='jax', structure=LOSSY, **SCENE)
        solver.structure = jnp.asarray(LOSSY)
        loss_fn = (lambda result: jnp.sum(result.T).real)
    else:
        torch = pytest.importorskip('torch')
        solver = _legacy_wavebit.rcwa(backend='torch', structure=LOSSY, **SCENE)
        solver.structure = torch.tensor(LOSSY, dtype=torch.complex128)
        loss_fn = (lambda result: torch.sum(result.T).real)
    grads = solver.grad(['structure'], functools.partial(solver.solve), loss_fn)
    return solver, complex(np.asarray(_detach(grads['structure']))[PIXEL])


def _detach(v):
    return v.detach() if hasattr(v, 'detach') else v


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_the_gradient_of_a_complex_permittivity_points_downhill(backend):
    """Against finite differences, both partials -- and then a step, which is the whole point.

    Asserting the value alone would pass for the conjugate on the real part and fail only on
    the imaginary one, so the step is asserted too: moving `-lr * grad` must *lower* the loss.
    That is the sentence a caller actually relies on, and it was false on JAX.
    """
    truth = _true_gradient()
    _, got = _solver_and_gradient(backend)

    assert got.real == pytest.approx(truth.real, rel=1e-4, abs=1e-9), (backend, got, truth)
    assert got.imag == pytest.approx(truth.imag, rel=1e-4, abs=1e-9), (
        f'{got.imag:+.9f} against {truth.imag:+.9f} -- a sign here is the conjugate, which '
        f'reads as a plausible gradient and steps the wrong way')

    base = _loss_at(LOSSY[PIXEL])
    stepped = _loss_at(LOSSY[PIXEL] - 1e-3 * got)
    assert stepped < base, (
        f'stepping -lr * grad raised the loss ({base:.12f} -> {stepped:.12f}); the gradient '
        f'points uphill')


def test_the_two_backends_return_the_same_gradient():
    """They were exact conjugates of each other, agreeing to 1e-13 after conjugation.

    Cross-backend agreement is the cheap check that would have caught this without an oracle,
    and it is the one this suite runs everywhere else. It was never run here.
    """
    pytest.importorskip('jax')
    pytest.importorskip('torch')
    _, from_jax = _solver_and_gradient('jax')
    _, from_torch = _solver_and_gradient('torch')
    assert from_jax == pytest.approx(from_torch, rel=1e-9, abs=1e-12), (
        f'jax {from_jax} against torch {from_torch}; conjugates differ by '
        f'{abs(from_jax - from_torch.conjugate()):.2e}, which is the tell')


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_asking_for_the_same_gradient_twice_gives_the_same_answer(backend):
    """Torch accumulates into `.grad`, and nothing used to clear it: 1x, 2x, 3x.

    Silent, and it reads as a learning rate that is merely too high -- every step of a
    hand-written loop moves further than the last.
    """
    if backend == 'jax':
        pytest.importorskip('jax')
    else:
        pytest.importorskip('torch')

    solver, first = _solver_and_gradient(backend)
    if backend == 'jax':
        import jax.numpy as jnp
        loss_fn = (lambda result: jnp.sum(result.T).real)
    else:
        import torch
        loss_fn = (lambda result: torch.sum(result.T).real)

    for call in (2, 3):
        again = solver.grad(['structure'], functools.partial(solver.solve), loss_fn)
        value = complex(np.asarray(_detach(again['structure']))[PIXEL])
        assert value == pytest.approx(first, rel=1e-12, abs=1e-15), (
            f'call {call} answered {value} where call 1 answered {first} '
            f'(ratio {abs(value / first):.4f})')


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_fit_lowers_the_loss(backend):
    """`fit` is the other half of this surface and had no test either.

    On JAX it conjugated in its own `step`, which is how the asymmetry with `grad` survived --
    `fit` was right and `grad` was wrong, so a user who only ever called `fit` saw nothing. The
    conjugation now happens once, in `_grad`, and this asserts `fit` still descends after that
    line moved.
    """
    if backend == 'jax':
        jnp = pytest.importorskip('jax.numpy')
        pytest.importorskip('optax')
        import optax
        _legacy_wavebit.enable_x64()
        solver = _legacy_wavebit.rcwa(backend='jax', structure=LOSSY, **SCENE)
        solver.structure = jnp.asarray(LOSSY)
        loss_fn = (lambda result: jnp.sum(result.T).real)
        before = float(loss_fn(solver.solve()))
        solver.fit(['structure'], functools.partial(solver.solve), loss_fn,
                   optax.sgd(1e-2), iteration=5)
    else:
        torch = pytest.importorskip('torch')
        solver = _legacy_wavebit.rcwa(backend='torch', structure=LOSSY, **SCENE)
        solver.structure = torch.tensor(LOSSY, dtype=torch.complex128)
        loss_fn = (lambda result: torch.sum(result.T).real)
        before = float(loss_fn(solver.solve()))
        solver.fit(['structure'], functools.partial(solver.solve), loss_fn,
                   torch.optim.SGD, {'lr': 1e-2}, iteration=5)

    after = float(loss_fn(solver.solve()))
    assert after < before, f'{before:.12f} -> {after:.12f}'
