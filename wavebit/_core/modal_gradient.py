"""First material derivatives in a fixed numerical modal basis.

Individual eigenvectors have no unique derivative at a repeated eigenvalue.
The physical layer instead needs the complete variation of its generator.
We stop the eigensolve and represent that variation in the computed basis;
modal admittances and exponentials then consume its full matrix variation.
This is a first-order construction, not a differentiable eigendecomposition.
"""
from .backend import array_namespace


def _stop(value):
    xp = array_namespace(value)
    if 'jax' in xp.__name__:
        import jax
        return jax.lax.stop_gradient(value)
    if 'torch' in xp.__name__:
        return value.detach()
    return value


def _traced(value):
    xp = array_namespace(value)
    if 'jax' in xp.__name__:
        import jax
        return isinstance(value, jax.core.Tracer)
    return bool(getattr(value, 'requires_grad', False))


_JAX_DELTA = None
_TORCH_DELTA = None


def _variation(operator, basis):
    """Zero value whose first variation is basis^-1 d(operator) basis."""
    xp = array_namespace(operator)
    if 'jax' in xp.__name__:
        global _JAX_DELTA
        if _JAX_DELTA is None:
            import jax
            import jax.numpy as jnp
            @jax.custom_jvp
            def first_order_only(value):
                return value
            @first_order_only.defjvp
            def no_second_order(primals, tangents):
                raise NotImplementedError('modal generator derivatives support first order only')
            @jax.custom_vjp
            def variation(a, v):
                return jnp.zeros_like(v)
            def forward(a, v):
                return jnp.zeros_like(v), (a, v)
            def backward(saved, cotangent):
                a, v = saved
                first_order_only(a)
                gradient = jnp.linalg.solve(v.T, cotangent) @ v.T
                if not jnp.iscomplexobj(a):
                    gradient = jnp.real(gradient)
                return gradient, None
            variation.defvjp(forward, backward)
            _JAX_DELTA = variation
        return _JAX_DELTA(operator, basis)
    if 'torch' in xp.__name__:
        global _TORCH_DELTA
        if _TORCH_DELTA is None:
            import torch
            class Variation(torch.autograd.Function):
                @staticmethod
                def forward(ctx, a, v):
                    ctx.save_for_backward(v)
                    ctx.real_input = not torch.is_complex(a)
                    return torch.zeros_like(v)
                @staticmethod
                def backward(ctx, cotangent):
                    # once_differentiable alone is insufficient: a constant
                    # cotangent can still leave a graph through a nonlinear
                    # upstream parameterization and return a partial Hessian.
                    if torch.is_grad_enabled():
                        raise RuntimeError(
                            'modal generator derivatives support first order only; '
                            'create_graph=True is not supported')
                    (v,) = ctx.saved_tensors
                    gradient = torch.linalg.solve(v.mH, cotangent) @ v.mH
                    if ctx.real_input:
                        gradient = gradient.real
                    return gradient, None
            _TORCH_DELTA = Variation.apply
        return _TORCH_DELTA(operator, basis)
    return xp.zeros_like(operator)


def generator(xp, operator, w, W, values, *, square_root):
    """Represent the full first variation of a matrix or its selected square root.

    For Q**2 = A, the Sylvester equation Q dQ + dQ Q = dA gives
    dQ_ij = (W**-1 dA W)_ij / (q_i + q_j). Repeated q values are
    harmless; zero/opposing roots are outside this differentiable branch.
    """
    if not _traced(operator):
        return values
    n = w.shape[-1]
    eye = xp.eye(n, dtype=operator.dtype, device=getattr(operator, 'device', None))
    basis = _stop(W)
    residual = _variation(operator, basis)
    if square_root:
        total = values[:, None]+values[None, :]
        valid = xp.all(total != 0)
        _require_separated(valid, 'zero or opposing square-root modes; use auto_detune',
                           error=FloatingPointError)
        total = xp.where(total != 0, total, xp.ones_like(total))
        residual = residual / _stop(total)
        residual = residual*xp.where(valid, xp.ones_like(residual),
                                     xp.full_like(residual, complex('nan')))
    return eye*values + residual


def split_generator(xp, operator, values, basis, forward):
    """Differentiate separated directional subspaces, not individual degenerate modes."""
    if not _traced(operator):
        return values, basis
    n = values.shape[0]
    eye = xp.eye(n, dtype=operator.dtype, device=getattr(operator, 'device', None))
    residual = _variation(operator, basis)
    separate = forward[:, None] != forward[None, :]
    gap = values[None, :]-values[:, None]
    # A directional invariant subspace has a derivative only while separated
    # from the opposite one. Do not invent a finite derivative at a collision.
    magnitude = xp.max(xp.abs(values))
    tolerance = 64*xp.finfo(magnitude.dtype).eps*xp.maximum(xp.ones_like(magnitude), magnitude)
    valid = xp.all(~separate | (xp.abs(gap) > tolerance))
    _require_separated(valid, 'coincident forward and backward modes')
    residual = residual*xp.where(valid, xp.ones_like(residual),
                                     xp.full_like(residual, complex('nan')))
    denominator = xp.where(separate & (gap != 0), gap, xp.ones_like(gap))
    change = xp.where(separate, residual/denominator, xp.zeros_like(residual))
    vectors = basis @ (eye+change)
    modal = eye*values + xp.where(separate, xp.zeros_like(residual), residual)
    return modal, vectors


def _require_separated(valid, description, *, error=ValueError):
    """Raise eagerly; an abstract JAX trace is marked nonfinite by the caller."""
    try:
        concrete = bool(valid)
    except Exception:
        return
    if not concrete:
        raise error(f'modal derivatives are undefined for {description}')


def exponential(xp, q, scale):
    """The diagonal modal exponential with its complete first matrix variation."""
    scale = xp.asarray(scale, dtype=q.dtype, device=getattr(q, 'device', None))
    if q.ndim == 1:
        eye = xp.eye(q.shape[0], dtype=q.dtype, device=getattr(q, 'device', None))
        return eye*xp.exp(scale[..., None]*q)[..., None, :]
    values = xp.linalg.diagonal(q)
    eye = xp.eye(q.shape[0], dtype=q.dtype, device=getattr(q, 'device', None))
    scaled = scale[..., None]*values
    phases = xp.exp(scaled)
    difference = values[:, None]-values[None, :]
    argument = scale[..., None, None]*difference
    right = xp.real(argument) >= 0
    numerator = xp.where(right,
                         phases[..., :, None]*(-xp.expm1(-xp.where(right, argument, 0))),
                         phases[..., None, :]*xp.expm1(xp.where(right, 0, argument)))
    denominator = xp.where(difference != 0, difference, xp.ones_like(difference))
    divided = xp.where(difference != 0, numerator/denominator,
                       scale[..., None, None]*phases[..., None, :])
    return eye*phases[..., None, :] + (q-eye*values)*divided


def combine(xp, first, second):
    if first.ndim == second.ndim == 1:
        return xp.concat([first, second])
    a = xp.diag(first) if first.ndim == 1 else first
    b = xp.diag(second) if second.ndim == 1 else second
    off = xp.zeros((a.shape[0], b.shape[0]), dtype=a.dtype, device=getattr(a, 'device', None))
    return xp.concat([xp.concat([a, off], axis=1),
                      xp.concat([xp.matrix_transpose(off), b], axis=1)], axis=0)


def inverse(xp, modal):
    if modal.ndim == 1:
        return xp.diag(1/modal)
    values = xp.linalg.diagonal(modal)
    eye = xp.eye(modal.shape[0], dtype=modal.dtype, device=getattr(modal, 'device', None))
    return eye/values - (modal-eye*values)/(values[:, None]*values[None, :])
