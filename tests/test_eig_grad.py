"""Gradients through ``eig``, including where the eigenvalues are degenerate.

``backend.eig`` supplies its own gradient rule rather than delegating to the framework's.
It has to: the eigenvector derivative carries a ``1 / (w_i - w_j)``, and both JAX's
``enable_eigvec_derivs`` and PyTorch's native rule return NaN — the *whole* gradient, not
only the degenerate directions — the moment two eigenvalues coincide.

That is not a corner case in RCWA. A **uniform layer at normal incidence** has
``q(+m) == q(-m)`` exactly for every retained order, so a spacer or buffer layer in an
otherwise ordinary stack is enough to produce it. meent carried a regularized rule for this
from 2023 (``on_torch/emsolver/primitives.py::Eig``, ``perturbation = 1E-10``); the
single-source rewrite replaced it with the frameworks' native rules in ``d1837ec`` and the
regularization was lost, which is what this file exists to prevent recurring.

Two things are pinned here, and they pull in opposite directions:

* where the native rules *are* defined, this one must agree with them (and with finite
  differences) — the regularization must not buy degeneracy at the cost of everything else;
* where they are not, this one must return the limit rather than NaN.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.backend import eig

pytest.importorskip('jax')
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import wavebit.on_jax                                       # noqa: E402,F401  (enables x64)


def _rand(rng, n):
    return rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))


def _matrix_function(A, xp):
    """A gauge-invariant scalar that depends on the *eigenvectors*, not just the values.

    ``V f(w) V^-1`` is a function of ``A`` alone, so its norm cannot depend on how LAPACK
    chose to scale or phase each eigenvector. A loss built only from ``w`` would leave the
    eigenvector branch of the rule — the part that carries the ``1/(w_i - w_j)`` — untested.
    """
    w, V = eig(A)
    P = V @ xp.diag(xp.exp(w * 0.3)) @ xp.linalg.inv(V)
    return xp.real(xp.sum(P * xp.conj(P)))


# ----------------------------------------------------------------------
# 1. agreement with the native rules, where those are defined
# ----------------------------------------------------------------------
def test_matches_the_jax_native_rule():
    """Non-degenerate: the custom rule must reproduce ``enable_eigvec_derivs`` exactly."""
    import jax.lax.linalg as jll

    A = jnp.asarray(_rand(np.random.default_rng(0), 5))

    def native(A):
        out = jll.eig(A, compute_left_eigenvectors=False, compute_right_eigenvectors=True,
                      enable_eigvec_derivs=True)
        w, V = out[0], out[-1]
        P = V @ jnp.diag(jnp.exp(w * 0.3)) @ jnp.linalg.inv(V)
        return jnp.real(jnp.sum(P * jnp.conj(P)))

    got = np.asarray(jax.grad(lambda A: _matrix_function(A, jnp))(A))
    want = np.asarray(jax.grad(native)(A))
    # 1e-9 relative is round-off between two orderings of the same expression, not slack:
    # every way this rule can be wrong (a dropped term, a wrong conjugation, a sign) moves
    # the answer by O(1) — the conjugation error this test was written after moved it by 5x.
    assert np.abs(got - want).max() < 1e-9 * np.abs(want).max()


def test_matches_the_torch_native_rule():
    """Same statement on PyTorch — and it is a separate statement.

    The two frameworks hand a custom rule cotangents in opposite complex conventions (JAX
    with respect to the parameters, PyTorch with respect to their conjugates). Taking that
    on faith is how the PyTorch side first came out wrong by a whole term while JAX was
    already exact, so neither backend's agreement implies the other's.
    """
    torch = pytest.importorskip('torch')
    A0 = _rand(np.random.default_rng(0), 5)

    def run(fn):
        At = torch.tensor(A0, dtype=torch.complex128, requires_grad=True)
        w, V = fn(At)
        P = V @ torch.diag(torch.exp(w * 0.3)) @ torch.linalg.inv(V)
        torch.real(torch.sum(P * torch.conj(P))).backward()
        return At.grad.numpy()

    got, want = run(eig), run(torch.linalg.eig)
    assert np.abs(got - want).max() < 1e-9 * np.abs(want).max()


@pytest.mark.parametrize('scale', [1.0, 1e-2, 1e-4], ids=['unit', 'small', 'tiny'])
def test_matches_finite_differences_at_any_operator_scale(scale):
    """And against finite differences, with the operator scaled down four decades.

    The scale sweep is the whole reason the broadening epsilon is relative to the eigenvalue
    range instead of a fixed constant. With meent's fixed ``1e-10`` this same check degrades
    from ~2e-10 to ~1e-7 across these three cases; it stays flat here. Physically it says the
    gradient does not depend on what units the structure is written in.
    """
    rng = np.random.default_rng(1)
    A0 = _rand(rng, 6) * scale
    d = rng.normal(size=(6, 6))

    g = np.asarray(jax.grad(lambda A: _matrix_function(A, jnp))(jnp.asarray(A0)))
    ad = float(np.real(np.sum(g * d)))

    h = 1e-6
    fd = float((_matrix_function(jnp.asarray(A0 + h * d), jnp)
                - _matrix_function(jnp.asarray(A0 - h * d), jnp)) / (2 * h))
    assert abs(ad - fd) < 1e-8 * abs(fd)


# ----------------------------------------------------------------------
# 2. degeneracy: finite, and equal to the limit
# ----------------------------------------------------------------------
DEGENERATE = [
    ('distinct', [1.0, 2.0, 3.0, 4.0, 5.0]),
    ('double', [2.0, 2.0, 3.0, 4.0, 5.0]),
    ('triple', [2.0, 2.0, 2.0, 4.0, 5.0]),
    ('two_pairs', [2.0, 2.0, 4.0, 4.0, 5.0]),
]


@pytest.mark.parametrize('name,diag', DEGENERATE, ids=[c[0] for c in DEGENERATE])
def test_degenerate_eigenvalues_give_a_finite_gradient(name, diag):
    """The native rules return NaN for every one of these but the first."""
    B = jnp.asarray(_rand(np.random.default_rng(2), 5))
    M = jnp.asarray(np.diag(np.array(diag, dtype=complex)))

    g = float(jax.grad(lambda s: _matrix_function(M + s * B, jnp))(jnp.array(0.0)))
    assert np.isfinite(g), (name, g)

    torch = pytest.importorskip('torch')
    Mt = torch.tensor(np.diag(np.array(diag, dtype=complex)), dtype=torch.complex128,
                      requires_grad=True)
    w, V = eig(Mt)
    torch.real(torch.sum(w * torch.conj(w))).backward()
    assert bool(torch.isfinite(Mt.grad).all()), (name, Mt.grad)


# ----------------------------------------------------------------------
# 3. the RCWA case that motivates all of it
# ----------------------------------------------------------------------
UNIFORM = np.full((1, 1, 8), 2.5)
STACK = dict(n_top=1.0, n_bot=1.0, phi=None, tetm='te', fto=[4, 0],
             period=[0.7], wavelength=0.9, thickness=[0.2])


def _ucell_grad(theta, backend):
    """d(sum T) / d(ucell) — a derivative that has to cross the eigendecomposition."""
    if backend == 1:
        def loss(u):
            s = wavebit.rcwa(backend=1, ucell=u, theta=theta,
                               fourier_coefficients='sampled', factorization='li', **STACK)
            return jnp.real(jnp.sum(s.conv_solve().T))
        return np.asarray(jax.grad(loss)(jnp.asarray(UNIFORM)))

    torch = pytest.importorskip('torch')
    u = torch.tensor(UNIFORM, dtype=torch.float64, requires_grad=True)
    s = wavebit.rcwa(backend=2, ucell=u, theta=theta,
                       fourier_coefficients='sampled', factorization='li', **STACK)
    s.conv_solve().T.real.sum().backward()
    return u.grad.numpy()


@pytest.mark.parametrize('backend', [1, 2], ids=['jax', 'torch'])
def test_uniform_layer_at_normal_incidence_is_differentiable(backend):
    """The ordinary structure that used to return an all-NaN gradient.

    A uniform layer, normal incidence, sampled coefficients: ``q(+m) == q(-m)`` exactly for all nine retained
    orders. Every element of the gradient came back NaN, on both autodiff backends — and
    tilting the source by 1e-9 was enough to make it finite and well behaved, so this is a
    knife-edge that a sweep steps onto rather than a region of bad conditioning.

    The assertion is the limit, not merely finiteness: the value at ``theta = 0`` must equal
    what approaching it gives. Broadening that only avoided the NaN would fail this.
    """
    at_zero = _ucell_grad(0.0, backend)
    assert np.all(np.isfinite(at_zero)), at_zero

    approaching = _ucell_grad(1e-9, backend)
    assert np.allclose(at_zero, approaching, rtol=1e-6, atol=1e-12), (at_zero, approaching)


def test_the_degeneracy_is_real():
    """Guard the guard: none of the above means anything if the modes are not degenerate.

    Counted pairwise rather than between sorted neighbours. ``sort_complex`` orders by real
    part then imaginary, and these modes are a mix of real and purely imaginary ones, so a
    degenerate pair need not land adjacent — an earlier version of this check counted three
    pairs instead of four for that reason alone.
    """
    import array_api_compat.numpy as xnp
    from wavebit._core.factorization import to_conv_mat_raster_sampled
    from wavebit._core.solve_1d import layer_modes_1d

    epx, epy, epz = to_conv_mat_raster_sampled(UNIFORM ** 2, 4, 0)
    fto = 4
    kx = np.asarray([0.0 + m * 0.9 / 0.7 for m in range(-fto, fto + 1)], dtype=complex)
    I = np.eye(2 * fto + 1, dtype=complex)
    _, _, q = layer_modes_1d(xnp, 0, kx, I, epx[0], epy[0], epz[0], complex)

    q = np.asarray(q)
    nearest = (np.abs(q[:, None] - q[None, :]) + np.eye(len(q)) * 1e9).min(axis=1)
    # m = 0 stands alone; +m and -m coincide for every other order.
    assert int((nearest < 1e-12).sum()) == 2 * fto, nearest
    assert len({round(v.real, 12) + 1j * round(v.imag, 12) for v in q}) == fto + 1
