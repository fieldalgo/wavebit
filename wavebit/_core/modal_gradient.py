"""First material derivatives in a fixed numerical modal basis.

Individual eigenvectors have no unique derivative at a repeated eigenvalue.
The physical layer instead needs the complete variation of its generator.
We stop the eigensolve and represent that variation in the computed basis;
modal admittances and exponentials then consume its full matrix variation.
This is a first-order construction, not a differentiable eigendecomposition.
"""
from .backend import array_namespace, carries_autodiff, forward_tangent, in_namespace


def _stop(value):
    xp = array_namespace(value)
    if 'jax' in xp.__name__:
        import jax
        return jax.lax.stop_gradient(value)
    if 'torch' in xp.__name__:
        return value.detach()
    return value


#: What a second derivative through the modal operator raises with.
_SECOND_ORDER = 'modal generator derivatives support first order only'

#: What ``create_graph=True`` adds: ``torch.autograd.functional.jvp`` is a first derivative that
#: PyTorch computes by a double backward, so it lands on this refusal too.
_CREATE_GRAPH = ('; create_graph=True is not supported (torch.autograd.functional.jvp and hvp '
                 'take that route too: for a forward derivative use torch.autograd.forward_ad)')

#: What a backward pass through a node built from a dual adds. The dual is the signal, so a
#: first-order backward pass through a graph built inside a dual level lands here as well, even
#: after the level has closed.
_DUAL = (' is not supported: it is a mixed second order, or a first-order backward pass through a '
         'graph built while a forward_ad dual level was open -- build the reverse pass outside the '
         'dual level')

#: What a ``torch.func`` transform through the modal operator raises with.
_FUNCTORCH = ('modal generator derivatives are not differentiable under torch.func transforms '
              '(grad, jvp, jacrev, jacfwd, vmap of a derivative): their torch.autograd.Function '
              'has no functorch rule. Use torch.autograd instead -- backward(), '
              'torch.autograd.grad or torch.autograd.forward_ad -- which they support to first '
              'order')

_JAX_DELTA = None
_TORCH_DELTA = None


def _variation(operator, basis):
    """Zero value whose first variation is basis^-1 d(operator) basis.

    The variation is linear in ``d(operator)``, with the basis held fixed, so it is the same
    rule in either mode: forward mode carries ``V^-1 dA V`` itself, reverse mode its transpose
    ``V^-T G V^T`` (the real part for a real operator). On JAX it is a ``custom_jvp``, so
    ``jax.jvp`` and ``jax.jacfwd`` take it as written and ``jax.grad`` by transposition; the
    solve is a ``custom_linear_solve`` whose transpose is spelled ``solve(V^T, G) @ V^T``, so a
    reverse-mode gradient is the arithmetic the reverse-only ``custom_vjp`` this replaced did.
    On PyTorch it is an ``autograd.Function`` with a ``backward`` and a ``jvp``, entered for a
    ``torch.autograd.forward_ad`` dual as well as for ``requires_grad``
    (:func:`~wavebit._core.backend.carries_autodiff`). Before, JAX's forward mode raised
    (``can't apply forward-mode autodiff (jvp) to a custom_vjp function``) and PyTorch's dual
    skipped the rule: ``d(R + T)/d(diameter)`` on dielectric disks at 81 harmonics read
    -5.48e-6 where reverse mode and a central difference read -7.35e-5, with no error.

    **First order only.** The basis is held fixed, so differentiating this rule again would
    drop the eigenvectors' own variation and return a partial Hessian. So a second derivative
    raises wherever it would differentiate the rule again, which is wherever the operator moves
    under both derivatives: on JAX ``NotImplementedError`` from ``first_order_only`` once the
    operator carries a tangent of an outer transformation (``jax.hessian``,
    ``jacfwd(jacrev)``, ``jvp(grad)``, ``grad(grad)``, ``grad`` of a ``jvp``, under
    ``jax.checkpoint`` as well); on PyTorch ``RuntimeError`` for ``create_graph=True`` and for
    a backward pass through a node built from a forward-mode dual, which is where both mixed
    orders reach it. A second derivative in which one parameter moves no operator -- a
    thickness beside a permittivity -- differentiates the rule once, in the other parameter,
    and is not refused: it is exact on JAX in every order and on PyTorch through
    ``create_graph=True`` taken in that parameter first (the other way round, the rule's own
    backward runs with grad mode on and refuses), while PyTorch's two mixed orders there
    carry ``torch.linalg.solve``'s own forward-over-reverse error wherever the stack solves
    through it (the autodiff guide, "Forward mode").
    """
    xp = array_namespace(operator)
    if 'jax' in xp.__name__:
        return _jax_variation()(operator, basis)
    if 'torch' in xp.__name__:
        try:
            return _torch_variation()(operator, basis, forward_tangent(operator))
        except RuntimeError as error:
            # `torch.func` refuses an `autograd.Function` without a functorch rule, naming
            # `setup_context`; that is said here in this rule's own terms. There is no rule to
            # offer: the first-order guard below reads `torch.is_grad_enabled()`, which
            # `torch.func.grad` turns on for every backward pass.
            if 'setup_context' not in str(error):
                raise
            raise NotImplementedError(_FUNCTORCH) from error
    return xp.zeros_like(operator)


def _jax_variation():
    """The JAX rule: a ``custom_jvp`` whose tangent is ``V^-1 dA V``, built on first use."""
    global _JAX_DELTA
    if _JAX_DELTA is None:
        import jax
        import jax.numpy as jnp

        @jax.custom_jvp
        def first_order_only(operator, basis):
            return basis

        @first_order_only.defjvp
        def no_second_order(primals, tangents):
            raise NotImplementedError(_SECOND_ORDER)

        @jax.custom_jvp
        def variation(a, v):
            return jnp.zeros_like(v)

        @variation.defjvp
        def tangent(primals, tangents):
            a, v = primals
            da = tangents[0]
            # The basis, passed through for what its own derivative does: it is differentiated
            # only when `a` carries the tangent of an outer transformation, which is a second
            # derivative, and then it raises. Its output is the basis the rule reads, so it is
            # load-bearing. A bare call whose result is dropped is dead code to JAX's partial
            # evaluation, which removes it under `jax.checkpoint` before the outer derivative
            # reaches it: `jax.hessian(jax.checkpoint(f))` of a 2D raster solve returned
            # 1.0385e-5 where a difference of `jax.grad` reads -6.352e-6, with no error.
            v = first_order_only(a, v)
            # `da` is cast to the basis' dtype first: a real operator's tangent is real, and
            # the transpose of that cast is the real part reverse mode takes.
            moved = jnp.astype(da, v.dtype) @ v
            # V^-1 (dA V), as a linear solve whose transpose is solve(V^T, G): the reverse rule
            # this replaced computed exactly `solve(V^T, G) @ V^T`, and `jnp.linalg.solve`'s
            # own transpose would reuse V's LU instead of factorizing V^T, changing rounding.
            return jnp.zeros_like(v), jax.lax.custom_linear_solve(
                lambda x: v @ x, moved,
                solve=lambda _, y: jnp.linalg.solve(v, y),
                transpose_solve=lambda _, y: jnp.linalg.solve(v.T, y))

        _JAX_DELTA = variation
    return _JAX_DELTA


def _torch_variation():
    """The PyTorch rule: an ``autograd.Function`` with a ``backward`` and a ``jvp``."""
    global _TORCH_DELTA
    if _TORCH_DELTA is None:
        import torch

        class Variation(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a, v, dual):
                ctx.save_for_backward(v)
                ctx.save_for_forward(v)
                ctx.real_input = not torch.is_complex(a)
                ctx.dual = dual
                return torch.zeros_like(v)

            @staticmethod
            def jvp(ctx, da, *constants):
                # Called with the operator's tangent: the basis and the flag carry none.
                (v,) = ctx.saved_tensors
                return torch.linalg.solve(v, da.to(v.dtype) @ v)

            @staticmethod
            def backward(ctx, cotangent):
                # once_differentiable alone is insufficient: a constant
                # cotangent can still leave a graph through a nonlinear
                # upstream parameterization and return a partial Hessian.
                if torch.is_grad_enabled():
                    raise RuntimeError(_SECOND_ORDER + _CREATE_GRAPH)
                if ctx.dual:
                    raise RuntimeError(_SECOND_ORDER + '; a backward pass through a modal '
                                                       'operator built from a forward-mode dual'
                                       + _DUAL)
                (v,) = ctx.saved_tensors
                gradient = torch.linalg.solve(v.mH, cotangent) @ v.mH
                if ctx.real_input:
                    gradient = gradient.real
                return gradient, None, None

        _TORCH_DELTA = Variation.apply
    return _TORCH_DELTA


def generator(xp, operator, w, W, values, *, square_root):
    """Represent the full first variation of a matrix or its selected square root.

    For Q**2 = A, the Sylvester equation Q dQ + dQ Q = dA gives
    dQ_ij = (W**-1 dA W)_ij / (q_i + q_j). Repeated q values are
    harmless; zero/opposing roots are outside this differentiable branch.
    """
    if not carries_autodiff(operator):
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
    if not carries_autodiff(operator):
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


def exponential(xp, q, scale, coupled=False):
    """The modal exponential ``exp(scale * q)`` with its complete first matrix variation.

    ``q`` is a vector of roots, or a matrix: a modal generator (:func:`generator`), whose value
    is diagonal, or a conical x-cutoff layer's propagator
    (:func:`~wavebit._core.solve_1d._regular_conical_modes`), whose value is diagonal plus one
    coupling block with disjoint rows and columns. For either value the divided differences
    below are the exact exponential: no product of two off-diagonal entries survives, so
    ``exp(s q)`` is the diagonal one plus ``q_jk (exp(s q_j) - exp(s q_k)) / (q_j - q_k)`` off
    it (Parlett's recurrence: B. N. Parlett, Linear Algebra Appl. 14, 117, 1976), and that holds
    for every ``scale``, so a ``scale`` derivative is exact too.

    What the formula does *not* carry exactly is a derivative of ``q`` beside a coupled value:
    a variation can put entries where two couplings chain, which the formula drops. So
    ``coupled=True`` -- the caller's *structural* word that ``q``'s value carries a coupling, set
    where the layer was rebuilt and never read off the value, since a pytree child cannot be
    branched on -- takes, when ``q`` carries autodiff, the value from the formula and the tangent
    from the backend's matrix exponential (:func:`_expm`), whose derivative is its Frechet
    derivative. Without autodiff, or with ``coupled=False`` (the default, and every other
    layer), the arithmetic is the formula alone, as it always was.
    """
    if 'jax' in xp.__name__ and in_namespace(xp, scale):
        # Already a JAX array: a cast, not a placement. JAX cannot batch `asarray(..., device=)`
        # of a tracer -- under `vmap`, and so under `jax.jacfwd` in a thickness, its
        # `convert_element_type` rule raised `AttributeError: 'SingleDeviceSharding' object has
        # no attribute 'spec'`. The values are the same either way.
        scale = xp.astype(scale, q.dtype)
    else:
        scale = xp.asarray(scale, dtype=q.dtype, device=getattr(q, 'device', None))
    if q.ndim == 1:
        eye = xp.eye(q.shape[0], dtype=q.dtype, device=getattr(q, 'device', None))
        return eye*xp.exp(scale[..., None]*q)[..., None, :]
    if coupled and carries_autodiff(q):
        exact = _expm(xp, scale[..., None, None]*q)
        return exponential(xp, _stop(q), _stop(scale)) + (exact - _stop(exact))
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


#: The largest 1-norm the degree-13 Pade approximant takes unscaled, ``theta_13`` (N. J. Higham,
#: SIAM J. Matrix Anal. Appl. 26, 1179, 2005).
_THETA_13 = 5.371920351148152

#: The most squarings :func:`_expm` takes: a 1-norm up to ``theta_13 * 2**64`` (about 1e20),
#: far past where ``exp`` of the layer overflows anyway. It bounds the eager count's loop and
#: is the traced count's fixed number of conditional steps.
_MAX_SQUARINGS = 64


def _squarings(norm):
    """The fewest ``s >= 0`` with ``norm <= theta_13 * 2**s``; ``None`` for an abstract ``norm``.

    Read one comparison at a time through :func:`~wavebit._core.backend._concrete`, the house's
    one door to a concrete value, so an abstract ``norm`` (``jax.jit``, ``jax.vmap``) answers
    ``None`` instead of raising. A NaN norm answers 0 (nothing to scale); a norm past the cap
    answers the cap.
    """
    from .backend import _concrete
    for s in range(_MAX_SQUARINGS):
        over = _concrete(norm > _THETA_13 * 2.0 ** s)
        if over is None:
            return None
        if not over:
            return s
    return _MAX_SQUARINGS


def _expm(xp, a):
    """``expm(a)`` (batched over leading axes) on the backend, for its Frechet derivative.

    Reached only from :func:`exponential` with ``coupled=True`` under autodiff, so only on JAX
    and PyTorch; NumPy carries no derivative. PyTorch's ``matrix_exp`` is used as it is.
    ``jax.scipy.linalg.expm`` takes ``floor(log2(|A|_1 / theta_13))`` squarings, so a norm up to
    ``2 theta_13`` reaches the Pade approximant unscaled: measured (jax 0.11.0) 4e-9 in the value
    and 8e-9 in the derivative on a cutoff layer's ``q`` at ``|A|_1 = 10.5``, where torch was
    3e-15. So it is scaled here to ``|A|_1 <= theta_13`` -- the *ceiling* of that ``log2``,
    Higham's count -- and squared back, and the squarings differentiate exactly.

    The count is read off the value where there is one (:func:`_squarings`), so eager arithmetic
    is ``s`` plain squarings. Where the norm is abstract -- a record built eagerly and then
    differentiated under ``jax.jit`` or ``jax.vmap`` with the record passed in, which makes its
    ``q`` a tracer -- the same count is formed traced and the squarings run as a fixed scan of
    ``_MAX_SQUARINGS`` conditional steps -- borrowed from ``jax.scipy.linalg.expm``, which undoes
    its own scaling that way, so a traced count differentiates -- rather than falling back to
    jax's floor count. Measured on a field's ``z`` derivative at a patterned
    exceptional point, ``jax.jit(jax.grad)`` against eager ``jax.grad``: 6.4e-11 with that
    fallback, 4e-16 .. 1.3e-15 with the traced count.
    """
    if 'jax' in xp.__name__:
        import jax
        norm = xp.max(xp.sum(xp.abs(_stop(a)), axis=-2))
        s = _squarings(norm)

        def expm(b):
            if b.ndim == 2:
                return jax.scipy.linalg.expm(b)
            r = jax.vmap(jax.scipy.linalg.expm)(xp.reshape(b, (-1,) + b.shape[-2:]))
            return xp.reshape(r, b.shape)
        if s is not None:
            r = expm(a / (2.0 ** s))
            for _ in range(s):
                r = r @ r
            return r
        tiny = xp.finfo(norm.dtype).tiny
        count = xp.clip(xp.ceil(xp.log2(xp.maximum(norm, tiny) / _THETA_13)), 0, _MAX_SQUARINGS)
        r = expm(a / (2.0 ** count).astype(a.dtype))

        def square(r, step):
            return jax.lax.cond(step < count, lambda m: m @ m, lambda m: m, r), None
        r, _ = jax.lax.scan(square, r, xp.arange(_MAX_SQUARINGS))
        return r
    if 'torch' in xp.__name__:
        import torch
        return torch.linalg.matrix_exp(a)
    import scipy.linalg
    return scipy.linalg.expm(a)


def two_face(xp, q, scale, top, bottom, keep, coupled=False):
    """A layer's ``-`` amplitude at its bottom face, read from whichever face keeps it accurate.

    ``top`` and ``bottom`` are the ``-`` (up-going) amplitudes at the layer's top and bottom
    faces, each read off its own face; in exact arithmetic ``top = exp(-scale*Q) @ bottom``, with
    ``scale = k0*d`` and ``Q`` the layer's roots. ``keep`` marks the modes to read from the top,
    ``exp(scale*q) * top``: those with ``|scale*q|`` small, for which that factor stays bounded
    (the caller's :data:`~wavebit._core.scattering._TOP_FACE`). Every other mode keeps
    ``bottom``.

    Why both faces. The ``+`` amplitude is read off the top face, and for a mode with
    ``|scale*q|`` small the two amplitudes nearly cancel in the field -- at a layer light line
    (``q -> 0``) each is ``O(1/q)`` while the field is ``O(1)``. Read off different faces, their
    rounding does not cancel with them, and the field lost ``log10(1/q**2)`` digits; read off one
    face it does. Any other mode's two amplitudes do not cancel, so each is read where it is
    computed, and ``exp(scale*q)`` -- which overflows for a mode that decays -- is never formed.

    With a modal generator ``q`` (autodiff, :func:`generator`) the answer carries the first
    variation of ``exp(scale*Q) @ top`` on the kept rows: divided differences of ``exp`` between
    two kept modes, as :func:`exponential` forms them, and for a kept mode ``j`` beside a dropped
    one ``k`` the exact identity ``exp(scale*q_k) top_k = bottom_k``, so the dropped mode's
    growing factor is never formed either. The value is the selection above.

    ``coupled=True`` is the caller's structural word that ``q``'s *value* carries a coupling
    block -- a conical x-cutoff layer (:func:`~wavebit._core.solve_1d._regular_conical_modes`),
    ``tau`` on rows ``J`` and columns ``K``, disjoint. The value above is still exact for every
    row outside ``J``: such a row of ``exp(scale*Q)`` is diagonal -- a mode in ``K`` included, its
    coupling sits in its *column* -- so the identity a dropped mode stands on still holds. A row
    in ``J`` is
    read off the bottom face (rows are found as the nonzero rows of the coupling's value: an
    exact test, no threshold). The first variation then misses one family of terms, the
    coupling chained behind a variation into a dropped ``J`` mode,
    ``d_ja tau_ab f[j, a, b] top_b``; with ``f[j, a, b] = (f[j, b] - f[a, b])/(q_j - q_a)`` and
    the ``f[a, b]`` half already inside ``bottom_a``, what is left is
    ``(d_ja/(q_j - q_a)) tau_ab f[j, b] top_b`` -- the ``beside`` term times ``tau`` times one
    more divided difference, written for a dropped ``b`` through ``bottom_b`` again -- which is
    added. Without it a field gradient at a patterned exceptional point was 2.6e-2 off; reading
    the whole layer off its bottom face instead was exact there, but took this protection away
    from every other mode of a rebuilt uniform layer, and a near-grazing order beside a cutoff
    lost 1.5e-2 in the field. ``coupled=False`` (the default, and every other layer) is the
    arithmetic above, unchanged.

    **What makes that first variation exact is partly the tangent, not this function.** A kept
    row ``k`` in ``K`` sits beside its cluster partner ``j`` in ``J``, a row the rule above drops
    whatever its ``|scale*q|``, and the two roots are a cutoff pair: ``q_k - q_j`` is the pair's
    root gap -- round-off at a patterned exceptional point (2.1e-15 .. 4.4e-15 over the test
    scenes), exactly 0 in a uniform layer. The ``beside`` term ``d_kj/(q_k - q_j)`` (``d_kj``
    over 1 at a zero gap) is then right only because a *physical* tangent's ``K x J`` entries
    vanish with that gap: family-1 modes stay eigenvectors of the layer operator under every
    variation that keeps the premise ``epz_i = inv(eps_y)``, ``muz_i = inv(mu_y)`` -- the one the
    family split already rests on, and every parameter of a 1D grating keeps it -- so
    ``d_ka = (q1_a - q2_k)(Z^-1 db1_a)_k``. For an arbitrary ``dq`` the term loses accuracy like
    ``u/gap``: measured on a synthetic coupled ``q`` with a random ``dq``, kept-row tangent
    errors of 7e-16 at a gap of 0.3, 9e-10 at 1e-8, 2.4e-5 at 1e-12, 8.5e-3 at 1e-15 and 0.30 at
    0. On the physical PA616 layer (gap 2.1e-15, its ``K`` row kept) SMM's field gradient is the
    ETM's to 2.4e-15 in the raster and 1.1e-14 in the wavelength.
    """
    if q.ndim == 1:
        e = xp.exp(xp.where(keep, scale * q, xp.zeros_like(q)))
        return xp.where(keep[:, None], e[:, None] * top, bottom)
    values = xp.linalg.diagonal(q)
    eye = xp.eye(q.shape[0], dtype=q.dtype, device=getattr(q, 'device', None))
    if coupled:
        tau = _stop(q) - eye * _stop(values)
        keep = xp.logical_and(keep, xp.logical_not(xp.any(tau != 0, axis=1)))
    e = xp.exp(xp.where(keep, scale * values, xp.zeros_like(values)))
    difference = values[:, None] - values[None, :]
    argument = scale * difference
    right = xp.real(argument) >= 0
    numerator = xp.where(right,
                         e[:, None] * (-xp.expm1(-xp.where(right, argument, 0))),
                         e[None, :] * xp.expm1(xp.where(right, 0, argument)))
    denominator = xp.where(difference != 0, difference, xp.ones_like(difference))
    divided = xp.where(difference != 0, numerator / denominator, scale * e[None, :])
    both = keep[:, None] & keep[None, :]
    across = keep[:, None] & ~keep[None, :]
    variation = q - eye * values
    zero = xp.zeros_like(variation)
    within = xp.where(both, variation * divided, zero)
    beside = xp.where(across, variation / denominator, zero)
    kept = e[:, None] * top + within @ top + e[:, None] * (beside @ top) - beside @ bottom
    if coupled:
        # f[j, b] top_b for a kept row j: by `divided` beside a kept b, through `bottom_b`
        # beside a dropped one (a mode in a coupling column has a diagonal row).
        ahead = xp.where(keep[None, :, None], divided[:, :, None] * top[None, :, :],
                         (e[:, None, None] * top[None, :, :] - bottom[None, :, :])
                         / denominator[:, :, None])
        kept = kept + xp.sum((beside @ tau)[:, :, None] * ahead, axis=1)
    return xp.where(keep[:, None], kept, bottom)


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
