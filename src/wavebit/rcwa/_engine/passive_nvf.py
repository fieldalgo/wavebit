"""Experimental passive closure of compressed scalar NVF projectors.

This matrix-level construction is explicit; ordinary NVF dispatch is unchanged.
It preserves passive constitutive forms, not a certified optical error bound.
"""
from wavebit.rcwa._engine.factorization import ConvMatrices


def _adjoint(xp, matrix):
    return xp.matrix_transpose(xp.conj(matrix))


def _block(xp, a, b, c, d):
    return xp.concat([xp.concat([a, b], axis=-1), xp.concat([c, d], axis=-1)], axis=-2)


def passive_nvf_assemble(xp, ep_dir, ep_inv, c_xx, c_xy, c_yy, c_yx=None):
    """Return scalar NVF slots using a normalized energy form.

    E=ep_dir and A=inv(ep_inv) are the direct and inverse material rules.
    G is the compressed unit normal projector, Q=I-G, and K=Q^H Q+G^H G.
    With Z=K^(-1/2), the transverse constitutive matrix is

        Z^H [Q^H diag(E,E) Q + G^H diag(A,A) G] Z.

    Passive E and A give a passive result by congruence, including negative
    real material parts. The same argument preserves positive storage when
    both input rules have positive storage. Uniform materials give epsilon*I,
    and a constant normal gives the exact one-dimensional Li factorization.

    The coefficient inputs must be the Fourier compression of a Hermitian
    unit normal projector on a sufficiently resolved grid, so 0 <= G <= I
    and I/2 <= K <= I. Seven coupled Newton-Schulz steps evaluate Z without
    eigenvectors or square roots at zero eigenvalues of G. Their derivative
    stays live at a constant-normal projector. This is an experimental finite
    closure: passivity does not establish Fourier convergence or contact
    accuracy, and it does not replace the existing off-interface field.
    """
    n = ep_dir.shape[-1]
    if c_yx is None:
        c_yx = c_xy
    g = _block(xp, c_xx, c_xy, c_yx, c_yy)
    dtype = xp.result_type(ep_dir, ep_inv, g)
    ep_dir, ep_inv, g = (xp.astype(value, dtype) for value in (ep_dir, ep_inv, g))
    device = ep_dir.device if hasattr(ep_dir, 'device') else None
    eye = xp.eye(2*n, dtype=ep_dir.dtype, device=device)
    q = eye-g
    gh, qh = _adjoint(xp, g), _adjoint(xp, q)
    gram = qh@q+gh@g
    # The spectrum is in [1/2, 1], so a fixed iteration count has a uniform
    # convergence bound. No material/geometry-dependent stopping decision is
    # hidden from differentiation.
    y, z = gram, eye
    for _ in range(7):
        step = (3*eye-z@y)/2
        y, z = y@step, step@z
    zero = xp.zeros_like(ep_dir)
    direct = _block(xp, ep_dir, zero, zero, ep_dir)
    inverse = xp.linalg.inv(ep_inv)
    normal = _block(xp, inverse, zero, zero, inverse)
    energy = qh@direct@q+gh@normal@g
    effective = _adjoint(xp, z)@energy@z
    return ConvMatrices(effective[:n, :n], effective[n:, n:], xp.linalg.inv(ep_dir),
                        -effective[:n, n:], -effective[n:, :n])
