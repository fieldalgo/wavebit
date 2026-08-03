"""Backend adapter — the only code that dispatches on the array framework.

The single-source core (migration Phase 3+) calls these; the current per-backend solvers do
not yet. Keeping the surface tiny is the point: everything divergent between NumPy / JAX /
PyTorch lives here, so it can be tested in one place.

``eig`` is the one primitive that genuinely differs. JAX >= 0.11 supports gradients through
non-symmetric complex ``eig`` natively via ``enable_eigvec_derivs`` (assumes distinct
eigenvalues); PyTorch's ``torch.linalg.eig`` is differentiable; NumPy needs no gradient.
"""
import array_api_compat as ac


def array_namespace(*xs):
    """Return the Array-API namespace for the given array(s) (numpy / jax.numpy / torch)."""
    return ac.array_namespace(*xs)


def eig(A):
    """Eigendecomposition of a (possibly non-Hermitian) complex matrix.

    Returns ``(eigenvalues, right_eigenvectors)`` and is differentiable on JAX and PyTorch.
    """
    xp = ac.array_namespace(A)
    name = xp.__name__
    if 'torch' in name:
        import torch
        w, V = torch.linalg.eig(A)
        return w, V
    if 'jax' in name:
        import jax.lax.linalg as jll
        out = jll.eig(
            A,
            compute_left_eigenvectors=False,
            compute_right_eigenvectors=True,
            enable_eigvec_derivs=True,
        )
        return out[0], out[-1]
    import numpy as np
    w, V = np.linalg.eig(A)
    return w, V
