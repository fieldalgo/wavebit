"""First material derivatives in a fixed numerical modal basis.

Individual eigenvectors have no unique derivative at a repeated eigenvalue.
The physical layer instead needs the complete variation of its generator.
We stop the eigensolve and represent that variation in the computed basis;
modal admittances and exponentials then consume its full matrix variation.
This is a first-order construction, not a differentiable eigendecomposition.
"""
from wavebit._common.backend import _concrete
from wavebit._common.backend import array_namespace
from wavebit._common.backend import carries_autodiff
from wavebit._common.backend import forward_tangent
from wavebit._common.backend import in_namespace


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

#: What a ``torch.func`` grad or jvp transform through the modal operator raises with. The
#: rule has a functorch ``vmap`` rule, so ``torch.func.vmap`` with ``torch.autograd`` is exact;
#: what it cannot do is tell a ``torch.func`` grad or jvp level from a second derivative (see
#: :func:`_variation`), so those are refused at its door where they differentiate the operator.
_FUNCTORCH = ('modal generator derivatives are not differentiable under torch.func transforms '
              '(grad, vjp, jvp, jacrev, jacfwd, hessian, or any of them composed with vmap): '
              'their first-order guard cannot tell such a level from a second derivative. Use '
              'torch.autograd instead -- backward(), torch.autograd.grad or '
              'torch.autograd.forward_ad, also through torch.func.vmap -- which they support to '
              'first order')

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
    (:func:`~wavebit._common.backend.carries_autodiff`). Before, JAX's forward mode raised
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
    and is not refused: it is exact on JAX in every order, and on PyTorch through
    ``create_graph=True`` taken in that parameter first (the other way round, the rule's own
    backward runs with grad mode on and refuses) and through its two mixed orders, because the
    stack solves through :func:`~wavebit._common.backend.linear_solve`, whose derivatives PyTorch
    can take again; through ``torch.linalg.solve`` itself they read 42% and 65% off on a 2D ETM
    raster, with no error (the autodiff guide, "Forward mode").

    **Under** ``torch.func``. The PyTorch Function carries a ``vmap`` rule, so
    ``torch.autograd`` through ``torch.func.vmap`` -- ``backward()``, ``torch.autograd.grad``,
    ``forward_ad``, a captured parameter beside a batched input -- takes this rule per lane and
    is the eager derivative. A ``torch.func`` grad or jvp level (``grad``, ``vjp``, ``jvp``,
    ``jacrev``, ``jacfwd``, ``hessian``, and each of them composed with ``vmap``) is refused at
    the door with ``NotImplementedError`` (:data:`_FUNCTORCH`) where it differentiates the
    operator: the transform's own level by its liveness, any level nested beneath it by its type
    (:func:`~wavebit._common.backend.functorch_differentiates`); a transform over something that
    moves no operator, such as a thickness, passes, also with a captured ``requires_grad``
    tensor in the operator, and the rule runs on that tensor's level only. The refusal is the
    guard's, not a missing rule: under ``torch.func.grad`` the backward runs with grad mode on,
    where the ``torch.is_grad_enabled()`` test cannot tell it from ``create_graph=True``, and
    under ``torch.func.jvp`` the ``jvp`` has no guard at all, so ``jvp(jvp)`` would
    differentiate the fixed-basis rule again and return a partial Hessian with no error.
    """
    xp = array_namespace(operator)
    if 'jax' in xp.__name__:
        return _jax_variation()(operator, basis)
    if 'torch' in xp.__name__:
        from wavebit._common.backend import functorch_differentiates
        if functorch_differentiates(operator):
            raise NotImplementedError(_FUNCTORCH)
        return _torch_variation()(operator, basis, forward_tangent(operator))
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
    """The PyTorch rule: an ``autograd.Function`` with a ``backward``, a ``jvp`` and a ``vmap``.

    ``forward`` takes no ``ctx`` and a ``setup_context`` saves what it did, so that
    ``torch.func.vmap`` can take the Function; its ``vmap`` rule moves each batched input's
    batch axis to the front, **expands an unbatched one** to the batch, and applies the
    Function once on the plain tensors, whose ``backward`` and ``jvp`` already work over
    leading batch axes. Both inputs are expanded because the output must be batched whenever
    either input is: a generated rule (``generate_vmap_rule = True``) returns ``zeros_like(v)``
    unbatched beside a batched operator, and the cotangent summed over the lanes then lands on
    every lane -- n times too large, silently (exactly 3 times on 3 lanes, the operator batched
    and the basis captured). This is the pattern of PyTorch's "Extending torch.func with
    autograd.Function" (docs.pytorch.org/docs/2.13/notes/extending.func.html), which
    ``backend._TorchBesselJ1`` follows with a generated rule -- right for an elementwise
    Function of one input, wrong for this one.
    """
    global _TORCH_DELTA
    if _TORCH_DELTA is None:
        import torch

        class Variation(torch.autograd.Function):
            @staticmethod
            def forward(a, v, dual):
                return torch.zeros_like(v)

            @staticmethod
            def setup_context(ctx, inputs, output):
                a, v, dual = inputs
                ctx.save_for_backward(v)
                ctx.save_for_forward(v)
                ctx.real_input = not torch.is_complex(a)
                ctx.dual = dual

            @staticmethod
            def vmap(info, in_dims, a, v, dual):
                a_dim, v_dim, _ = in_dims
                n = info.batch_size
                a = a.movedim(a_dim, 0) if a_dim is not None else a.expand(n, *a.shape)
                v = v.movedim(v_dim, 0) if v_dim is not None else v.expand(n, *v.shape)
                return Variation.apply(a, v, dual), 0

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
    """Differentiate separated directional subspaces, not individual degenerate modes.

    ``values`` is a vector of eigenvalues, or a matrix: a first-order layer whose defective pair
    was rebuilt in a regular basis (:func:`~wavebit.rcwa._engine.solve_2d_first_order._regular_pair`),
    diagonal plus a coupling on disjoint rows and columns, every coupling inside one direction.
    The value comes from stopped data, so the coupling carries no tangent of its own. Then
    ``basis^-1 dA basis = X values - values X + d(values)`` with ``X`` confined to the entries
    across the two directions, ``d(values)`` to those within one, and the across part is a
    Sylvester equation whose coupling is solved in four passes (:func:`_split_sylvester`).
    """
    if not carries_autodiff(operator):
        return values, basis
    n = values.shape[0]
    eye = xp.eye(n, dtype=operator.dtype, device=getattr(operator, 'device', None))
    residual = _variation(operator, basis)
    separate = forward[:, None] != forward[None, :]
    coupled = values.ndim == 2
    roots = xp.linalg.diagonal(values) if coupled else values
    gap = roots[None, :]-roots[:, None]
    # A directional invariant subspace has a derivative only while separated
    # from the opposite one. Do not invent a finite derivative at a collision.
    magnitude = xp.max(xp.abs(roots))
    tolerance = 64*xp.finfo(magnitude.dtype).eps*xp.maximum(xp.ones_like(magnitude), magnitude)
    valid = xp.all(~separate | (xp.abs(gap) > tolerance))
    _require_separated(valid, 'coincident forward and backward modes')
    residual = residual*xp.where(valid, xp.ones_like(residual),
                                     xp.full_like(residual, complex('nan')))
    denominator = xp.where(separate & (gap != 0), gap, xp.ones_like(gap))
    if coupled:
        change = _split_sylvester(xp, values - eye*roots, denominator, residual, separate)
        modal = values + xp.where(separate, xp.zeros_like(residual), residual)
        return modal, basis @ (eye+change)
    change = xp.where(separate, residual/denominator, xp.zeros_like(residual))
    vectors = basis @ (eye+change)
    modal = eye*values + xp.where(separate, xp.zeros_like(residual), residual)
    return modal, vectors


def _split_sylvester(xp, N, gap, R, separate):
    """``X`` on the ``separate`` entries with ``X (D + N) - (D + N) X = R`` there, ``0`` elsewhere.

    ``gap`` holds ``D_j - D_i`` (guarded to 1 off ``separate``), ``N`` the coupling, whose
    nonzero rows ``J`` and columns ``K`` are disjoint and lie inside one direction, so that a
    ``separate`` entry ``(i, j)`` reads ``(N X)_ij`` only through entries ``(k, j)`` with ``k`` in
    ``K`` -- across too -- and ``(X N)_ij`` only through ``(i, k)``, ``k`` in ``J``. Entries off
    ``J`` rows and ``K`` columns are a plain quotient; a ``J`` row reads the first pass, a ``K``
    column too, and a ``J x K`` entry reads the two before it: the four passes of
    :func:`~wavebit.rcwa._engine.solve_1d._coupled_sylvester`, for this equation. ``J`` and ``K`` are
    read off the value (``N != 0``: exact, no threshold), so an all-zero ``N`` -- a layer under a
    trace whose masks engaged nothing -- is the first pass alone. Linear in ``R``.
    """
    zero = xp.zeros_like(R)
    rows = xp.any(N != 0, axis=1)[:, None]
    cols = xp.any(N != 0, axis=0)[None, :]
    X1 = xp.where(separate, R/gap, zero)
    chain = xp.any(rows[:, 0] & cols[0, :])
    some = _concrete(chain)
    if some is not False:
        # A Jordan chain of length three in three levels (solve_2d._chain_split): a mode both a
        # coupled row and a coupled column, N**3 = 0. The passes below do not close; the series
        # in N does. A `separate` entry joins the two directions, and each holds its own chain
        # (an exceptional point of the first-order operator comes as +-q), so N reaches it from
        # both sides, each N raising a level of three: a term with three on one side vanishes,
        # and the first pass and four updates are exact. Disjoint rows and columns keep the
        # passes, to the bit: eagerly read off the value, under a trace both formed and one
        # selected.
        chained = X1
        for _ in range(4):
            chained = xp.where(separate, (R - chained @ N + N @ chained)/gap, zero)
        if some:
            return chained
    X2 = xp.where(separate, (R + N @ X1)/gap, zero)
    X3 = xp.where(separate, (R - X1 @ N)/gap, zero)
    Xa = xp.where(rows & ~cols, X2, xp.where(~rows & cols, X3, X1))
    X4 = xp.where(separate, (R - Xa @ N + N @ Xa)/gap, zero)
    X = xp.where(rows & cols, X4, Xa)
    return X if some is False else xp.where(chain, chained, X)


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
    (:func:`~wavebit.rcwa._engine.solve_1d._regular_conical_modes`), whose value is diagonal plus one
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
    from an exact Frechet derivative, by one of two rules:

    * **eager** (a concrete norm: an eager solve, ``jax.grad``, ``jvp``, ``jacfwd``, torch
      autograd and ``forward_ad``; on PyTorch, which reads no norm, torch autograd and
      ``forward_ad`` through ``torch.func.vmap`` as well): the backend's matrix exponential
      (:func:`_expm`), whose derivative is its Frechet derivative, with JAX's squaring count
      read off the value;
    * **traced** (an abstract norm on JAX: ``jax.jit``, ``jax.vmap``, a ``lax.map`` or ``scan``
      body, ``jax.checkpoint``, an eager record passed into a trace): a ``jax.custom_jvp``
      whose tangent is the closed-form Frechet derivative of the triangular exponential
      (:func:`_jax_coupled_exponential`, :func:`_coupled_frechet`) -- no ``expm``, no
      squarings, nothing formed in a value-only trace. An ``expm`` there needs its squaring
      count traced, and undoing a traced count takes a fixed scan of conditional squarings,
      which ``vmap`` turns into selects that run all of them: measured with that tangent, a
      jitted 16-wavelength ``vmap`` sweep cost 2.4x .. 3.2x per call and its gradient
      3.4x .. 5.9x of what they cost before traced layers were rebuilt, against 1.01x .. 1.20x
      and 1.12x .. 1.46x with the closed form.

    An integer ``coupled`` gives the structural pair capacity to the traced rule;
    ``True`` retains the conical layout's fixed capacity. The value is never inspected
    to decide whether a layer carries pairs.

    Both are exact: the closed form matched a 40-digit ``expm`` of ``[[A, E], [0, A]]`` to 4e-16
    of ``max|L|`` over confluent, near-confluent, widely spread and complex-scaled cases (SciPy's
    ``expm_frechet`` 1.8e-16 .. 2.8e-14), and with either rule the eager gradients at the
    rebuilt test points agree to 2.4e-14. They are kept apart so that an eager tangent stays the
    bits it was.

    PyTorch never takes the closed form: under ``torch.autograd`` and ``forward_ad``, through
    ``torch.func.vmap`` too, where it batches, ``matrix_exp`` is used as it is (PyTorch reads no
    norm). A ``torch.func.vmap`` with no derivative takes the formula. Without autodiff, or with
    ``coupled=False`` (the default, and every other layer), the arithmetic is the formula
    alone, as it always was.
    """
    if 'jax' in xp.__name__ and in_namespace(xp, scale):
        # Already a JAX array: a cast, not a placement. JAX cannot batch `asarray(..., device=)`
        # of a tracer -- under `vmap`, and so under `jax.jacfwd` in a thickness, its
        # `convert_element_type` rule raised `AttributeError: 'SingleDeviceSharding' object has
        # no attribute 'spec'`. The values are the same either way.
        scale = xp.astype(scale, q.dtype)
    else:
        # A Torch scale states requires_grad: unspecified, torch 2.14 follows the input (this
        # same value) and warns that it does.
        flag = {'requires_grad': scale.requires_grad} if hasattr(scale, 'requires_grad') else {}
        scale = xp.asarray(scale, dtype=q.dtype, device=getattr(q, 'device', None), **flag)
    if q.ndim == 1:
        eye = xp.eye(q.shape[0], dtype=q.dtype, device=getattr(q, 'device', None))
        return eye*xp.exp(scale[..., None]*q)[..., None, :]
    if coupled and carries_autodiff(q):
        a = scale[..., None, None]*q
        count = None
        if 'jax' in xp.__name__:
            count = _squarings(xp.max(xp.sum(xp.abs(_stop(a)), axis=-2)))
            if count is None:
                # Abstract norm (jit, vmap, a lax.map / scan body, checkpoint, an eager record
                # passed into a trace): the formula for the value, the closed-form Frechet
                # derivative for the tangent (:func:`_jax_coupled_exponential`). An integer
                # `coupled` is the most coupled rows (and columns) the caller can produce -- a 2D
                # layer's pair cap (`solve_2d._rebuild_defective_pairs`; a pair couples one row,
                # a jointly rebuilt cluster its rows J) -- `True` the 1D default.
                cap = _COUPLED_CAP if coupled is True else max(_COUPLED_CAP, int(coupled))
                return _jax_coupled_exponential(cap, _chain_capacity(coupled))(q, scale)
        exact = _expm(xp, a, count)
        # The value is the formula's (a torch.func.vmap reads no chain: it takes the capacity).
        return (_exp_formula(xp, _stop(q), _stop(scale), _chain_capacity(coupled))
                + (exact - _stop(exact)))
    return _exp_formula(xp, q, scale, _chain_capacity(coupled))


def _chain_capacity(coupled):
    """How many middle modes of Jordan chains a trace gathers (:func:`_chain_paths`, and the
    tangent's :func:`_frechet_chain_terms`), or ``None``.

    An integer ``coupled`` is the caller's count of the coupled rows its layer can hold -- a 2D
    layer's pair cap, a first-order layer's slots and cluster lanes -- and a chain's middle mode
    is one of them, so it bounds the middles too. ``True`` (a conical cutoff layer, whose
    coupling is one block on disjoint rows and columns) and ``False`` hold no chain: a trace
    forms no two-step term for them.
    """
    import numbers
    if isinstance(coupled, bool) or not isinstance(coupled, numbers.Integral):
        return None
    return int(coupled)


def _exp_formula(xp, q, scale, chains=None):
    """:func:`exponential`'s divided-difference formula for a matrix ``q``; ``chains`` is
    :func:`_chain_capacity`'s count, which only a trace reads."""
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
    out = eye*phases[..., None, :] + (q-eye*values)*divided
    N = q - eye*values
    chain = _chain_paths(xp, N, values, scale, chains)
    if chain is None:
        return out
    nz = _stop(N) != 0
    engaged = xp.any(xp.any(nz, axis=1) & xp.any(nz, axis=0))
    # A selection under a trace, so that a layer with no chain keeps its bits.
    return out + chain if _concrete(engaged) else xp.where(engaged, out + chain, out)


def _chain_paths(xp, N, values, scale, cap=None):
    """The paths of two coupling steps in ``exp(scale (diag(values) + N))``, or ``None``.

    A Jordan chain of length three rebuilt in three levels (``solve_2d._chain_split``) has a
    coupling with ``N**2 != 0`` (``N**3 = 0``), and the divided-difference form of a function of
    a triangular matrix (Opitz 1964; Davis 1973) gains the paths through a middle mode ``k``:
    ``sum_k N_ik N_kj f[x_i, x_k, x_j]`` with ``f(x) = exp(scale x)``. ``k`` ranges over the
    modes that are both a coupled row and a coupled column (an exact test on the value, no
    threshold); a coupling on disjoint rows and columns has none, and the exponential is the
    one-step formula alone, to the bit. Read eagerly off the stopped value. Under an abstract
    trace nothing can be read: with a capacity ``cap`` (:func:`_chain_capacity`) the paths are
    gathered to static sizes (:func:`_chain_paths_traced`), and without one there are none.
    """
    nz = _stop(N) != 0
    middle = xp.any(nz, axis=1) & xp.any(nz, axis=0)
    some = _concrete(xp.any(middle))
    if some is None and cap is not None:
        return _chain_paths_traced(xp, N, values, scale, nz, middle, cap)
    if not some:
        return None
    import numpy as np
    from wavebit._common.backend import _readable
    dev = getattr(N, 'device', None)
    m = N.shape[0]
    idx = xp.arange(m, device=dev)
    nz_np = np.asarray(_readable(nz, xp))
    out = None
    for k in np.nonzero(np.asarray(_readable(middle, xp)))[0].tolist():
        rows = xp.asarray(np.nonzero(nz_np[:, k])[0], device=dev)
        cols = xp.asarray(np.nonzero(nz_np[k, :])[0], device=dev)
        x_i = xp.take(values, rows)[:, None]
        x_j = xp.take(values, cols)[None, :]
        x_k = values[k]
        f2 = _exp_divided_2(xp, scale[..., None, None], x_i, x_k, x_j)
        weight = xp.take(N[:, k], rows)[:, None] * xp.take(N[k, :], cols)[None, :]
        Oi = xp.astype(idx[:, None] == rows[None, :], N.dtype)
        Oj = xp.astype(idx[:, None] == cols[None, :], N.dtype)
        term = Oi @ (weight * f2) @ xp.matrix_transpose(Oj)
        out = term if out is None else out + term
    return out


def _chain_paths_traced(xp, N, values, scale, nz, middle, cap):
    """:func:`_chain_paths` under an abstract trace, gathered to static sizes.

    The middle modes are gathered to ``cap`` (:func:`_chain_capacity`) and each one's coupled
    rows and columns to a cluster's lane count (``solve_2d._EP_CLUSTER_SIZE``: a chain's
    coupling stays inside its cluster), by a stable sort of the masks -- no shape depends on a
    value. The term is ``sum_c O_i^T (w o f[x_i, x_k, x_j]) O_j`` over the gathered middles
    ``k``, with one-hot maps ``O`` back to the modes; padded lanes carry a zero weight and are
    selected out. More middles, or more rows or columns on one, than the sizes hold multiply the
    term by NaN: loud.
    """
    from wavebit.rcwa._engine.solve_2d import _EP_CLUSTER_SIZE
    m = N.shape[0]
    dev = getattr(N, 'device', None)
    cm = max(1, min(int(cap), m))
    cl = min(_EP_CLUSTER_SIZE, m)

    def first(mask, c):
        # The positions of the first `c` True entries along the last axis, and which are real.
        pos = xp.argsort(xp.astype(~mask, xp.int32), axis=-1, stable=True)[..., :c]
        count = xp.sum(xp.astype(mask, xp.int32), axis=-1)
        return pos, xp.arange(c, device=dev) < count[..., None], count

    kk, vk, nk = first(middle, cm)                                       # (cm,)
    ii, vi, ni = first(xp.matrix_transpose(xp.take(nz, kk, axis=1)), cl)  # (cm, cl): N[i, k] != 0
    jj, vj, nj = first(xp.take(nz, kk, axis=0), cl)                       # (cm, cl): N[k, j] != 0
    over = (nk > cm) | xp.any(vk & ((ni > cl) | (nj > cl)))

    def gather(v, index):
        return xp.reshape(xp.take(v, xp.reshape(index, (-1,))), index.shape)

    Nf = xp.reshape(N, (-1,))
    w_i = gather(Nf, ii * m + kk[:, None])                                 # N[i, k]
    w_j = gather(Nf, kk[:, None] * m + jj)                                 # N[k, j]
    x_i, x_j, x_k = gather(values, ii), gather(values, jj), xp.take(values, kk)
    f2 = _exp_divided_2(xp, scale[..., None, None, None], x_i[:, :, None], x_k[:, None, None],
                        x_j[:, None, :])                                   # (..., cm, cl, cl)
    ok = vk[:, None, None] & vi[:, :, None] & vj[:, None, :]
    zero = xp.zeros_like(f2)
    term = xp.where(ok, (w_i[:, :, None] * w_j[:, None, :]) * f2, zero)
    # Block-diagonal over the middles, then back to the modes by one-hot maps.
    same = xp.arange(cm, device=dev)[:, None] == xp.arange(cm, device=dev)[None, :]
    block = xp.where(same[:, None, :, None], term[..., :, :, None, :], xp.zeros_like(term[..., :, :, None, :]))
    block = xp.reshape(block, block.shape[:-4] + (cm * cl, cm * cl))
    idx = xp.arange(m, device=dev)
    Oi = xp.astype((idx[:, None] == xp.reshape(ii, (-1,))[None, :])
                   & xp.reshape(vi & vk[:, None], (-1,))[None, :], N.dtype)   # (m, cm cl)
    Oj = xp.astype((idx[:, None] == xp.reshape(jj, (-1,))[None, :])
                   & xp.reshape(vj & vk[:, None], (-1,))[None, :], N.dtype)
    out = Oi @ block @ xp.matrix_transpose(Oj)
    return out * xp.where(over, float('nan'), 1.0)


def _exp_divided_2(xp, s, x0, x1, x2):
    """``f[x0, x1, x2]`` for ``f(x) = exp(s x)``, stably: ``s**2 exp[p0, p1, p2]`` at
    ``p = s x``. A tight set (diameter at most :data:`_DD_SERIES_DIAMETER`) takes the Taylor
    series about ``p0`` (:func:`_series_about_first`'s, through degree
    :data:`_DD_SERIES_TERMS`); a spread one the recurrence divided by its widest pair, each
    first divided difference spelled with ``expm1`` as :func:`exponential` spells its own -- the
    hybrid of McCurdy, Ng and Parlett (Math. Comp. 43, 501, 1984). Backend-generic; every
    divisor guarded."""
    import math
    p0, p1, p2 = s * x0, s * x1, s * x2

    def first(a, b):
        d = a - b
        right = xp.real(d) >= 0
        num = xp.where(right, xp.exp(a) * (-xp.expm1(-xp.where(right, d, 0))),
                       xp.exp(b) * xp.expm1(xp.where(right, 0, d)))
        return xp.where(d != 0, num / xp.where(d != 0, d, xp.ones_like(d)), xp.exp(a))

    pts = (p0, p1, p2)
    pairs = [(0, 1), (0, 2), (1, 2)]
    gaps = [xp.abs(pts[i] - pts[j]) for i, j in pairs]
    # f[P - p_j] - f[P - p_i] over (p_i - p_j), for each pair (i, j) as the divisor.
    omit = {0: first(p1, p2), 1: first(p0, p2), 2: first(p0, p1)}
    cand = []
    for i, j in pairs:
        diff = pts[i] - pts[j]
        cand.append((omit[j] - omit[i]) / xp.where(diff != 0, diff, xp.ones_like(diff)))
    widest = xp.where(gaps[0] >= gaps[1], xp.where(gaps[0] >= gaps[2], 0, 2),
                      xp.where(gaps[1] >= gaps[2], 1, 2))
    recurrence = xp.where(widest == 0, cand[0], xp.where(widest == 1, cand[1], cand[2]))
    y1, y2 = p1 - p0, p2 - p0
    h = [xp.ones_like(y1 + y2)]
    for _ in range(_DD_SERIES_TERMS):
        h.append(h[-1] * y1)
    for j in range(1, _DD_SERIES_TERMS + 1):
        h[j] = h[j] + y2 * h[j - 1]
    total = xp.zeros_like(h[0])
    for j in range(_DD_SERIES_TERMS, -1, -1):
        total = total + h[j] * (1.0 / math.factorial(j + 2))
    series = xp.exp(p0) * total
    diameter = xp.maximum(xp.maximum(gaps[0], gaps[1]), gaps[2])
    return s * s * xp.where(diameter > _DD_SERIES_DIAMETER, recurrence, series)


#: The largest 1-norm the degree-13 Pade approximant takes unscaled, ``theta_13`` (N. J. Higham,
#: SIAM J. Matrix Anal. Appl. 26, 1179, 2005).
_THETA_13 = 5.371920351148152

#: The most squarings :func:`_expm` takes: a 1-norm up to ``theta_13 * 2**64`` (about 1e20),
#: far past where ``exp`` of the layer overflows anyway. It bounds the count's loop.
_MAX_SQUARINGS = 64


def _squarings(norm):
    """The fewest ``s >= 0`` with ``norm <= theta_13 * 2**s``; ``None`` for an abstract ``norm``.

    Read one comparison at a time through :func:`~wavebit._common.backend._concrete`, the house's
    one door to a concrete value, so an abstract ``norm`` (``jax.jit``, ``jax.vmap``) answers
    ``None`` instead of raising -- which is what sends :func:`exponential` to the closed form.
    A NaN norm answers 0 (nothing to scale); a norm past the cap answers the cap.
    """
    from wavebit._common.backend import _concrete
    for s in range(_MAX_SQUARINGS):
        over = _concrete(norm > _THETA_13 * 2.0 ** s)
        if over is None:
            return None
        if not over:
            return s
    return _MAX_SQUARINGS


def _expm(xp, a, count):
    """``expm(a)`` (batched over leading axes) on the backend, for its Frechet derivative.

    Reached only from :func:`exponential` with ``coupled=True`` under autodiff and, on JAX, a
    concrete norm (PyTorch reads none, so ``torch.autograd`` through ``torch.func.vmap``,
    ``forward_ad`` included, reaches here, batched), so only on JAX and PyTorch; NumPy carries no
    derivative. PyTorch's ``matrix_exp`` is used as it is, and ``count`` is ignored there.
    ``jax.scipy.linalg.expm`` takes ``floor(log2(|A|_1 / theta_13))`` squarings, so a norm up to
    ``2 theta_13`` reaches the Pade approximant unscaled: measured (jax 0.11.0) 4e-9 in the value
    and 8e-9 in the derivative on a cutoff layer's ``q`` at ``|A|_1 = 10.5``, where torch was
    3e-15. So it is scaled here to ``|A|_1 <= theta_13`` -- the *ceiling* of that ``log2``,
    Higham's count, which the caller reads off the value (:func:`_squarings`) and passes as
    ``count`` -- and squared back in ``count`` plain squarings, which differentiate exactly.

    On JAX ``count`` must be that concrete integer. An abstract norm never comes here: it takes
    the closed form (:func:`_jax_coupled_exponential`), which needs no count. Until 2026-10-11 an
    abstract norm -- then only an eager record differentiated under a trace -- formed the count
    traced and ran a fixed scan of ``_MAX_SQUARINGS`` conditional squarings (and before that fell
    back to jax's own floor count, 6.4e-11 off on a field's ``z`` derivative at a patterned
    exceptional point). ``None`` raises, so that path cannot come back unnoticed.
    """
    if 'jax' in xp.__name__:
        import jax
        if count is None:
            raise RuntimeError('internal: _expm needs a concrete squaring count on JAX; an '
                               'abstract norm takes the closed form (_jax_coupled_exponential)')

        def expm(b):
            if b.ndim == 2:
                return jax.scipy.linalg.expm(b)
            r = jax.vmap(jax.scipy.linalg.expm)(xp.reshape(b, (-1,) + b.shape[-2:]))
            return xp.reshape(r, b.shape)
        r = expm(a / (2.0 ** count))
        for _ in range(count):
            r = r @ r
        return r
    if 'torch' in xp.__name__:
        import torch
        return torch.linalg.matrix_exp(a)
    import scipy.linalg
    return scipy.linalg.expm(a)


#: At most this many coupled rows, and as many columns, in one coupled ``q`` take the closed
#: form's higher terms (:func:`_coupled_frechet`); more gives a NaN tangent (loud), the value
#: untouched. A rebuilt layer couples one family-1 mode per cluster pair -- every patterned
#: point measured couples one -- and a uniform layer none (its coupling is exactly zero). A 2D
#: layer's jointly rebuilt cluster couples its rows J to its columns K (two and two on the
#: four-mode clusters measured, one and two on a three-mode one, four and four on the
#: eight-mode ones), counted with the layer's pairs against its integer cap.
_COUPLED_CAP = 4

#: At or below this diameter a set of points takes the Taylor series of its divided difference
#: (:func:`_series_about_first`); above it, the recurrence divided by its widest pair
#: (:func:`_dd_from_subsets`).
_DD_SERIES_DIAMETER = 0.5

#: The highest degree that series keeps: it sums ``j = 0 .. _DD_SERIES_TERMS``, 19 terms. Every
#: ``|p - p_0|`` is at most the diameter, 0.5, so the tail, from ``j = 19`` on, is below about
#: ``0.5**19/19!`` (1.6e-23) of the leading term.
_DD_SERIES_TERMS = 18


def _series_about_first(y, e0, k):
    """``exp[p_0, ..., p_k]`` as ``e0 sum_j h_j(y_1, ..., y_k)/(j + k)!``, ``y_r = p_r - p_0``.

    ``e0 = exp(p_0)`` and ``y`` is the list ``[y_1, ..., y_k]``; ``h_j`` is the complete
    homogeneous symmetric polynomial of degree ``j``, built one variable at a time
    (``h_j <- h_j + y_r h_{j-1}``). It is the Taylor series of the divided difference about
    ``p_0`` (``x**n[p_0, ..., p_k] = h_{n-k}(p_0, ..., p_k)``) -- the clustered-points half of
    the hybrid of A. McCurdy, K. C. Ng and B. N. Parlett (Math. Comp. 43, 501, 1984). Used where
    the points' diameter is at most :data:`_DD_SERIES_DIAMETER`, through degree
    :data:`_DD_SERIES_TERMS`.
    """
    import math
    import jax.numpy as jnp
    h = [jnp.ones_like(y[0])]
    for _ in range(_DD_SERIES_TERMS):
        h.append(h[-1] * y[0])
    for r in range(1, len(y)):
        for j in range(1, _DD_SERIES_TERMS + 1):
            h[j] = h[j] + y[r] * h[j - 1]
    total = jnp.zeros_like(y[0])
    for j in range(_DD_SERIES_TERMS, -1, -1):
        total = total + h[j] * (1.0 / math.factorial(j + k))
    return e0 * total


def _dd_from_subsets(points, subs, e0):
    """``exp[p_0, ..., p_k]`` from the ``k + 1`` divided differences of order ``k - 1`` that each
    omit one point (``subs[j]`` omits ``p_j``) -- gathered, never recomputed.

    A spread set takes the recurrence ``(f[P - p_j] - f[P - p_i]) / (p_i - p_j)`` with ``(i, j)``
    its widest pair, so the divisor is the diameter (above :data:`_DD_SERIES_DIAMETER`) and the
    subtraction loses about ``log2(1/diameter)`` bits at most; a tight set takes
    :func:`_series_about_first`, about ``p_0`` with ``e0 = exp(p_0)`` -- the hybrid of McCurdy,
    Ng and Parlett (Math. Comp. 43, 501, 1984). Both are formed for every lane and one is
    selected; every divisor is guarded, here and in ``F1`` (:func:`_coupled_frechet`), so no
    lane forms an inf a reverse pass could read. One does read them all: reverse over reverse,
    the inner variation in ``q`` and the outer in ``scale`` (a raster gradient differentiated in
    the thickness), NaN with either guard dropped
    (``test_second_derivatives_through_a_traced_cutoff``).
    """
    import jax.numpy as jnp
    k = len(points) - 1
    dist, cand = [], []
    for i in range(k + 1):
        for j in range(i + 1, k + 1):
            diff = points[i] - points[j]
            gap = jnp.abs(diff)
            dist.append(gap)
            cand.append((subs[j] - subs[i]) / jnp.where(gap > 0, diff, jnp.ones_like(diff)))
    dist = jnp.stack(dist, axis=-1)
    cand = jnp.stack(cand, axis=-1)
    widest = jnp.argmax(dist, axis=-1)
    recurrence = jnp.take_along_axis(cand, widest[..., None], axis=-1)[..., 0]
    series = _series_about_first([p - points[0] for p in points[1:]], e0, k)
    return jnp.where(jnp.max(dist, axis=-1) > _DD_SERIES_DIAMETER, recurrence, series)


def _coupled_frechet(q, scale, E, cap=None, chains=None):
    """``L(scale q, E)``: the Frechet derivative of ``exp`` at a coupled ``scale * q`` in direction
    ``E``, in closed form. JAX only; ``scale`` may carry leading (batch) axes, and ``E`` with it.

    ``q``'s value is diagonal plus a coupling ``tau`` whose nonzero rows ``J`` and columns ``K``
    are disjoint (a rebuilt conical layer, :func:`~wavebit.rcwa._engine.solve_1d._regular_conical_modes`).
    With ``A = scale q = diag(a) + n``, ``n = scale tau``, no product of two couplings survives
    (``n D n = 0`` for every diagonal ``D``), so

        exp(uA) = diag(e^{u a}) + n o exp_u[a_j, a_k]

    exactly: the divided-difference form of a function of a triangular matrix (G. Opitz, ZAMM 44,
    T52, 1964; C. Davis, Linear Algebra Appl. 6, 193, 1973), whose paths are one step long here
    -- what Parlett's recurrence gives too (:func:`exponential`). The integral form of the
    Frechet derivative, ``L(A, E) = int_0^1 exp((1-u)A) E exp(uA) du`` (N. J. Higham, Functions
    of Matrices, SIAM 2008, eq. (10.15)), then closes term by term into divided differences of
    ``exp``, each integral of a product of exponentials being one (the Hermite-Genocchi
    formula)::

        L_il = exp[a_i, a_l] E_il
             + sum_k n_ik exp[a_i, a_k, a_l] E_kl  +  sum_k E_ik n_kl exp[a_i, a_k, a_l]
             + sum_km n_ik E_km n_ml exp[a_i, a_k, a_m, a_l]

    The first term is the Daleckii-Krein kernel (Ju. L. Daleckii and S. G. Krein, Amer. Math.
    Soc. Transl. (2) 47, 1, 1965), all of ``L`` for a diagonal ``A``. The same four terms are the
    paths of the doubled matrix ``[[A, E], [0, A]]``, whose ``exp`` carries ``L(A, E)`` in its
    upper-right block (R. Mathias, SIAM J. Matrix Anal. Appl. 17, 610, 1996) -- the identity the
    tests check this against. Exact for any ``E``, and linear in it.

    Every divided difference is built from the one below it and gathered: the first order
    ``F1 = exp[a_i, a_l]`` (``m x m``, one ``expm1`` per pair, spelled as :func:`exponential`
    spells its ``divided``), the second on the triples that touch the coupling, ``(J, K, .)``
    and ``(., J, K)``, from ``F1``, and the third on ``(J, K, J, K)`` from the second
    (:func:`_dd_from_subsets`) -- so the only transcendentals are ``exp(a)`` and those
    ``expm1``. Measured against a 40-digit ``expm`` of the doubled matrix: <= 4e-16 of
    ``max|L|`` across the series/recurrence switch, ``J``/``K`` gaps of 1e-8, 1e-12 and exactly
    0, a decaying spread to ``|scale q| = 600``, complex scale and a tight cluster at
    ``|a| ~ 50``, where SciPy's ``expm_frechet`` scored 1.8e-16 .. 2.8e-14.

    ``J`` and ``K`` are read off the value (``tau != 0``: exact, no threshold) and gathered to a
    static size, :data:`_COUPLED_CAP` each, so no shape depends on a value. More coupled rows or
    columns than that multiply the result by NaN -- loud, and still linear in ``E``, as a
    tangent rule must be.

    A row that is also a column breaks the premise: a Jordan chain of length three rebuilt in
    three levels (``solve_2d._chain_split``), whose middle mode both receives a coupling and
    passes one on, so ``n**2 != 0`` (``n**3 = 0``: three levels) and a path on each side of
    ``E`` takes up to two steps. With ``chains`` (:func:`_chain_capacity`, a 2D or first-order
    layer's count) the five families with a two-step path are added
    (:func:`_frechet_chain_terms`), their tables (:func:`_frechet_chain_tables`) behind a branch
    on whether a middle is present, so a layer with none keeps these bits and forms no chain
    table (under ``vmap``, where the branch is a selection, every lane forms them). Only values
    enter the branch, ``E`` never; measured within 6.2e-16 of a 40-digit ``expm`` of the
    doubled matrix across splits of 0 to 0.7 and complex scales. Without ``chains`` (a conical
    layer, ``coupled=True``) such a row multiplies the result by NaN.
    """
    import jax.numpy as jnp
    m = q.shape[-1]
    v = jnp.diagonal(q)
    tau = q - jnp.diag(v)
    a = scale[..., None] * v                                         # (..., m)
    ea = jnp.exp(a)
    d = a[..., :, None] - a[..., None, :]
    right = jnp.real(d) >= 0
    x = jnp.expm1(jnp.where(right, -d, d))
    num = jnp.where(right, ea[..., :, None] * (-x), ea[..., None, :] * x)
    nz = d != 0
    F1 = jnp.where(nz, num / jnp.where(nz, d, jnp.ones_like(d)),
                   jnp.broadcast_to(ea[..., :, None], d.shape))
    L = F1 * E
    rows = jnp.any(tau != 0, axis=1)
    cols = jnp.any(tau != 0, axis=0)
    middle = rows & cols
    c = min(_COUPLED_CAP if cap is None else cap, m)
    nr, nc = jnp.sum(rows), jnp.sum(cols)
    ok = (nr <= c) & (nc <= c)
    if chains is None:
        ok = ok & ~jnp.any(middle)
    J = jnp.nonzero(rows, size=c, fill_value=0)[0]
    K = jnp.nonzero(cols, size=c, fill_value=0)[0]
    vJ = jnp.arange(c) < nr
    vK = jnp.arange(c) < nc
    t = jnp.where(vJ[:, None] & vK[None, :], tau[J][:, K], 0)
    n = scale[..., None, None] * t                                   # (..., c, c)
    idx = jnp.arange(m)

    def F2(i0, i1, i2):
        i0, i1, i2 = jnp.broadcast_arrays(i0, i1, i2)
        return _dd_from_subsets([a[..., i0], a[..., i1], a[..., i2]],
                                [F1[..., i1, i2], F1[..., i0, i2], F1[..., i0, i1]], ea[..., i0])
    F2a = F2(J[:, None, None], K[None, :, None], idx[None, None, :])   # (..., c, c, m): J_p K_q l
    F2b = F2(idx[:, None, None], J[None, :, None], K[None, None, :])   # (..., m, c, c): i J_r K_s
    p_, q_, r_, s_ = (jnp.arange(c).reshape(sh) for sh in ((c, 1, 1, 1), (1, c, 1, 1),
                                                          (1, 1, c, 1), (1, 1, 1, c)))
    p_, q_, r_, s_ = jnp.broadcast_arrays(p_, q_, r_, s_)
    F3 = _dd_from_subsets(
        [a[..., J[p_]], a[..., K[q_]], a[..., J[r_]], a[..., K[s_]]],
        [F2b[..., K[q_], r_, s_], F2b[..., J[p_], r_, s_], F2a[..., p_, q_, K[s_]],
         F2a[..., p_, q_, J[r_]]], ea[..., J[p_]])                   # (..., c, c, c, c)
    E_K = E[..., K, :]
    E_J = E[..., :, J]
    E_KJ = E_K[..., :, J]
    term2 = jnp.einsum('...pq,...pql,...ql->...pl', n, F2a, E_K)
    term3 = jnp.einsum('...ir,...rs,...irs->...is', E_J, n, F2b)
    term4 = jnp.einsum('...pq,...qr,...rs,...pqrs->...ps', n, E_KJ, n, F3)
    OJ = jnp.astype(vJ[:, None] & (idx[None, :] == J[:, None]), q.dtype)   # (c, m)
    OK = jnp.astype(vK[:, None] & (idx[None, :] == K[:, None]), q.dtype)
    L = L + OJ.T @ term2 + term3 @ OK + OJ.T @ term4 @ OK
    if chains is not None:
        import jax

        def tables():
            return _frechet_chain_tables(tau, a, ea, F1, scale, K, vK, middle, c, chains)

        def none():
            return (jnp.asarray(True),) + tuple(jnp.zeros(s.shape, s.dtype) for s in shapes[1:])
        # A branch on whether a middle is present, so a layer with none forms no chain table
        # (under `vmap`, where the branch becomes a selection, every lane forms them). Only values
        # enter it: `vmap`'s selection passes a branch's operands through `stop_gradient`, which
        # reverse mode cannot transpose on a tangent. The tangent is contracted outside, and where
        # no middle is, a negative zero is added, which leaves every bit of `L` as it was. The
        # contraction sits between optimization barriers: without them XLA rescheduled the one-step
        # form's sums around it, and a first-order layer with no chain moved its `jit(grad)` by
        # up to 9.2e-16 relative.
        shapes = jax.eval_shape(tables)
        middled = jnp.any(middle)
        fits, *found = jax.lax.cond(middled, tables, none)
        barrier = jax.lax.optimization_barrier
        chain = barrier(_frechet_chain_terms(barrier(E), K, OK, *found))
        L = L + jnp.where(middled, chain, -jnp.zeros((), L.dtype))
        ok = ok & fits
    return L * jnp.where(ok, 1.0, float('nan'))


def _frechet_chain_terms(E, K, OK, i2, s2, X, vT, vX, w, u, F20, F21, F22):
    """The families of :func:`_coupled_frechet` with a two-step path, in direction ``E``, from
    the paths and tables of :func:`_frechet_chain_tables`.

    A two-step path ``i -> k -> s`` runs through a middle mode ``k``, both a coupled row and a
    coupled column, with weight ``w = n_ik n_ks``. With ``F`` the divided difference of ``exp``
    over the points named, ``P = (i, k, s)`` a path and ``n_xK`` a one-step coupling into a
    coupled column::

        (2, 0)   L_il  += w E_sl F[P, l]           (0, 2)   L_is' += E_ii' w' F[i, P']
        (2, 1)   L_iK  += w E_sx n_xK F[P, x, K]   (1, 2)   L_xs' += n_xK E_Ki' w' F[x, K, P']
        (2, 2)   L_is' += w E_si' w' F[P, P']

    the paths of the doubled matrix ``[[A, E], [0, A]]`` with two coupling steps on a side.
    ``F`` is symmetric in its points, so (0, 2) reads (2, 0)'s table and (1, 2) reads
    (2, 1)'s. Linear in ``E``; ``E`` is read nowhere else.
    """
    import jax.numpy as jnp
    m = E.shape[-1]
    idx = jnp.arange(m)
    T20 = (w[..., :, None] * E[..., s2, :]) * F20
    T02 = (E[..., :, i2] * w[..., None, :]) * jnp.matrix_transpose(F20)
    T21 = jnp.einsum('...g,...gsx,...sx,...gsx->...gs', w, E[..., s2[:, None, None], X[None]], u, F21)
    T12 = jnp.einsum('...sx,...sh,...h,...hsx->...sxh', u, E[..., K[:, None], i2[None, :]], w, F21)
    T22 = (w[..., :, None] * E[..., s2[:, None], i2[None, :]] * w[..., None, :]) * F22
    OI = jnp.astype(vT[:, None] & (idx[None, :] == i2[:, None]), E.dtype)   # (cp, m)
    OS = jnp.astype(vT[:, None] & (idx[None, :] == s2[:, None]), E.dtype)
    rows = jnp.reshape(X, (-1,))
    OX = jnp.astype(jnp.reshape(vX, (-1,))[:, None] & (idx[None, :] == rows[:, None]), E.dtype)
    T12 = jnp.reshape(T12, T12.shape[:-3] + (-1, T12.shape[-1]))
    return OI.T @ (T20 + T21 @ OK + T22 @ OS) + (T02 + OX.T @ T12) @ OS


def _frechet_chain_tables(tau, a, ea, F1, scale, K, vK, middle, c, chains):
    """The two-step paths of :func:`_frechet_chain_terms` and their divided-difference tables,
    from the value alone, and whether they fit: ``(fits, i2, s2, X, vT, vX, w, u, F20, F21,
    F22)``.

    The middles are gathered to ``chains`` (:func:`_chain_capacity`), each one's coupled rows
    and columns to a cluster's lanes, as :func:`_chain_paths_traced` gathers them, and the
    paths ``(i2, k2, s2)``, listed flat, to ``4 c`` (at most the candidates), ``c`` the
    caller's count of coupled rows: a path starts on one, and a traced split puts at most two
    lanes in each upper level (``solve_2d._chain_split_traced_wide``), so a row reaches at most
    two middles and a middle at most two successors -- four paths a row (two chains beside a
    third mode: 12 paths on 5 rows). ``vT`` marks the real ones, ``w`` their weights. The one-step couplings
    ``x -> K`` are read by column: each of ``K``'s columns and its coupled rows ``X``, gathered
    to a cluster's lanes (``vX``), since a coupling stays inside its cluster; ``u`` their
    ``n_xK``. The three tables ``F[P, l]``, ``F[P, x, K]`` and ``F[P, P']`` are each built over
    every subset of their points from ``F1`` up (:func:`_dd_lattice`). More middles, rows or
    columns on one, or paths than the sizes hold, or a coupling between two middles (a path of
    three steps), answer ``fits = False``.
    """
    import jax.numpy as jnp
    from wavebit.rcwa._engine.solve_2d import _EP_CLUSTER_SIZE
    m = tau.shape[-1]
    nz = tau != 0
    cm = max(1, min(int(chains), m))
    cl = min(_EP_CLUSTER_SIZE, m)

    def first(mask, size):
        # The positions of the first `size` True entries along the last axis, and which are real.
        pos = jnp.argsort(jnp.astype(~mask, jnp.int32), axis=-1, stable=True)[..., :size]
        count = jnp.sum(jnp.astype(mask, jnp.int32), axis=-1)
        return pos, jnp.arange(size) < count[..., None], count

    M, vM, nm = first(middle, cm)                                       # (cm,)
    P, vP, npre = first(jnp.take(nz, M, axis=1).T, cl)                  # (cm, cl): tau[i, k] != 0
    S, vS, nsuc = first(jnp.take(nz, M, axis=0), cl)                    # (cm, cl): tau[k, s] != 0
    paths = vM[:, None, None] & vP[:, :, None] & vS[:, None, :]
    # Four paths a coupled row, but no more than there are candidates (a small layer).
    cp = min(4 * c, cm * cl * cl)
    pick, vT, npaths = first(jnp.reshape(paths, (-1,)), cp)             # (cp,)
    g = pick // (cl * cl)                                               # a path's middle
    i2, k2, s2 = P[g, (pick // cl) % cl], M[g], S[g, pick % cl]
    w = jnp.where(vT, (scale[..., None] * tau[i2, k2]) * (scale[..., None] * tau[k2, s2]), 0)
    fits = ((nm <= cm) & ~jnp.any(vM & ((npre > cl) | (nsuc > cl))) & (npaths <= cp)
            & ~jnp.any(middle[:, None] & middle[None, :] & nz))
    # The one-step couplings by column: each of K's columns and its coupled rows, gathered to a
    # cluster's lanes (a coupling stays inside its cluster).
    X, vX, nX = first(jnp.take(nz, K, axis=1).T, cl)                    # (c, cl): tau[x, K_s] != 0
    vX = vX & vK[:, None]
    u = jnp.where(vX, scale[..., None, None] * tau[X, K[:, None]], 0)   # (.., c, cl): n_{x, K_s}
    fits = fits & ~jnp.any(vK & (nX > cl))
    idx = jnp.arange(m)
    # Formed `c` paths at a time, and F22 `c**2 // cp` rows at a time: the batches of the tables
    # before the path capacity grew to `4 c`. At once, a 66-mode first-order layer's
    # `jit(grad(sum vmap))` held 8.9 GB (4.5 GB at capacity `c`), past what a CI runner holds
    # beside a second worker.
    F20 = _dd_lattice_rows(a, ea, F1, [i2[:, None], k2[:, None], s2[:, None], idx[None, :]],
                           c)                                                         # (.., cp, m)
    F21 = _dd_lattice_rows(a, ea, F1, [i2[:, None, None], k2[:, None, None], s2[:, None, None],
                                       X[None, :, :], K[None, :, None]], c)          # (.., cp, c, cl)
    F22 = _dd_lattice_rows(a, ea, F1, [i2[:, None], k2[:, None], s2[:, None],
                                       i2[None, :], k2[None, :], s2[None, :]],
                           max(1, c * c // cp))                                       # (.., cp, cp)
    return fits, i2, s2, X, vT, vX, w, u, F20, F21, F22


def _dd_lattice_rows(a, ea, F1, idx, block):
    """:func:`_dd_lattice` over the broadcast ``idx``, ``block`` of its leading rows at a time.

    The same table: each row's entries are :func:`_dd_lattice`'s, formed by ``jax.lax.map`` in
    batches of ``block`` rows, so the lattice's intermediates -- a set's ``_DD_SERIES_TERMS``
    series degrees for every subset of its points -- are held for one batch, not every row. A
    table of no more than ``block`` rows is formed at once, as before.
    """
    import jax
    import jax.numpy as jnp
    full = jnp.broadcast_arrays(*idx)
    if full[0].shape[0] <= block:
        return _dd_lattice(a, ea, F1, idx)
    stacked = jnp.stack(full, axis=-1)                                  # (rows, ..., r)
    out = jax.lax.map(lambda one: _dd_lattice(a, ea, F1, [one[..., j] for j in range(len(idx))]),
                      stacked, batch_size=block)                        # (rows, *batch, ...)
    return jnp.moveaxis(out, 0, a.ndim - 1)


def _dd_lattice(a, ea, F1, idx):
    """``exp[a_{i_0}, ..., a_{i_{r-1}}]`` for ``r >= 3`` index arrays ``idx`` (broadcast
    together), over every subset of the points: the pairs gathered from ``F1``, then each size
    from the one below, every subset of a size at once (:func:`_dd_level`)."""
    import itertools
    import numpy as np
    import jax.numpy as jnp
    r = len(idx)
    idx = jnp.stack(jnp.broadcast_arrays(*idx), axis=-1)                # (..., r)
    pts, e = a[..., idx], ea[..., idx]                                  # (..., S, r)
    below = list(itertools.combinations(range(r), 2))
    pair = np.array(below)
    cur = F1[..., idx[..., pair[:, 0]], idx[..., pair[:, 1]]]           # (..., S, C(r, 2))
    for size in range(3, r + 1):
        sets = list(itertools.combinations(range(r), size))
        at = {s: n for n, s in enumerate(below)}
        members = np.array(sets)                                        # (C, size)
        omit = np.array([[at[s[:t] + s[t + 1:]] for t in range(size)] for s in sets])
        cur = _dd_level(pts[..., members], cur[..., omit], e[..., members[:, 0]])
        below = sets
    return cur[..., 0]


def _dd_level(points, subs, e0):
    """``exp[p_0, ..., p_k]`` for the sets in ``points`` (``..., k + 1``) from the divided
    differences that omit one point (``subs[..., j]`` omits ``p_j``), with ``e0 = exp(p_0)``:
    :func:`_dd_from_subsets`'s hybrid, its pairs and its series each formed as one array. The
    series' ``h_j`` (:func:`_series_about_first`) are built degree by degree in a ``lax.scan``,
    ``h_j^(r) = sum_{s <= r} y_s h_{j-1}^(s)`` a cumulative sum over the variables, so a set of
    any size costs a fixed number of operations."""
    import itertools
    import math
    import numpy as np
    import jax
    import jax.numpy as jnp
    k = points.shape[-1] - 1
    pair = np.array(list(itertools.combinations(range(k + 1), 2)))
    diff = points[..., pair[:, 0]] - points[..., pair[:, 1]]
    gap = jnp.abs(diff)
    cand = (subs[..., pair[:, 1]] - subs[..., pair[:, 0]]) / jnp.where(gap > 0, diff,
                                                                       jnp.ones_like(diff))
    widest = jnp.argmax(gap, axis=-1)
    recurrence = jnp.take_along_axis(cand, widest[..., None], axis=-1)[..., 0]
    y = points[..., 1:] - points[..., :1]

    def degree(h, _):
        h = jnp.cumsum(y * h, axis=-1)
        return h, h[..., -1]
    _, hs = jax.lax.scan(degree, jnp.ones_like(y), None, length=_DD_SERIES_TERMS)
    # In the points' precision: a float64 array would lift a complex64 table to complex128, and
    # a custom JVP's tangent must keep its primal's dtype.
    weight = np.array([1.0 / math.factorial(j + k) for j in range(1, _DD_SERIES_TERMS + 1)],
                      dtype=jnp.finfo(hs.dtype).dtype)
    total = jnp.tensordot(weight[::-1], hs[::-1], axes=1) + 1.0 / math.factorial(k)
    series = e0 * total
    return jnp.where(jnp.max(gap, axis=-1) > _DD_SERIES_DIAMETER, recurrence, series)


_JAX_COUPLED_EXP = {}


def _jax_coupled_exponential(cap=None, chains=None):
    """``exp(scale * q)`` for a coupled ``q`` under an abstract trace, as a ``jax.custom_jvp``.

    The value is :func:`exponential`'s divided-difference formula, exact for a coupled value
    and all that a value-only trace forms -- with ``chains`` (:func:`_chain_capacity`) its
    two-step paths too, and the tangent's chain families with them; the tangent is exact for
    any variation::

        d exp(s q) = L(s q, s dq) + ds q exp(s q)

    with ``L`` the closed form (:func:`_coupled_frechet`) at the *stopped* ``q``, and the
    ``scale`` part through the commuting identity ``d/ds exp(s q) = q exp(s q)``. So ``L`` is
    only ever evaluated on its structured manifold, and an outer derivative never differentiates
    it off it: an outer ``scale`` moves ``L``'s ``s q`` along ``q`` (structured), and an outer
    ``q`` reaches ``q exp(s q)``, whose own tangent is this rule -- which keeps a thickness
    second derivative and a mixed thickness x permittivity one exact, in both orders. Without
    the ``ds`` term a rebuilt layer's thickness derivative is lost (measured under ``jax.jit``
    at a patterned exceptional point: relative error 1.0). Stopping ``q`` inside ``L`` is
    hygiene that no reachable transform observes: a second derivative that would see it needs
    two directions in ``q``, and those raise at the modal operator's first-order guard
    (measured bit-identical with ``L`` at the live ``q``).
    """
    cap = _COUPLED_CAP if cap is None else cap
    if (cap, chains) not in _JAX_COUPLED_EXP:
        import jax

        @jax.custom_jvp
        def coupled_exp(q, scale):
            return _exp_formula(array_namespace(q), q, scale, chains)

        @coupled_exp.defjvp
        def coupled_exp_tangent(primals, tangents):
            q, scale = primals
            dq, d_scale = tangents
            out = coupled_exp(q, scale)
            tangent = (_coupled_frechet(_stop(q), scale, scale[..., None, None] * dq,
                                        None if cap == _COUPLED_CAP else cap, chains)
                       + d_scale[..., None, None] * (q @ out))
            return out, tangent

        _JAX_COUPLED_EXP[(cap, chains)] = coupled_exp
    return _JAX_COUPLED_EXP[(cap, chains)]


def two_face(xp, q, scale, top, bottom, keep, coupled=False):
    """A layer's ``-`` amplitude at its bottom face, read from whichever face keeps it accurate.

    ``top`` and ``bottom`` are the ``-`` (up-going) amplitudes at the layer's top and bottom
    faces, each read off its own face; in exact arithmetic ``top = exp(-scale*Q) @ bottom``, with
    ``scale = k0*d`` and ``Q`` the layer's roots. ``keep`` marks the modes to read from the top,
    ``exp(scale*q) * top``: those with ``|scale*q|`` small, for which that factor stays bounded
    (the caller's :data:`~wavebit.rcwa._engine.scattering._TOP_FACE`). Every other mode keeps
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
    block -- a conical x-cutoff layer (:func:`~wavebit.rcwa._engine.solve_1d._regular_conical_modes`),
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

    ``coupled='pairs'`` also reads the coupling's columns from the bottom face.
    A rebuilt 2D pair has no conical constraint on its lower tangent entry, so
    keeping that column would divide an unrestricted tangent by the vanishing pair gap.

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
        if coupled == 'pairs':
            # A 2D layer's rebuilt pairs and clusters (`solve_2d._rebuild_defective_pairs`): the
            # coupling's columns are read off the bottom face too. Their tangent's lower entry
            # `d_ba` is not tied to the pair's gap, as a conical cutoff pair's is (below), so a
            # kept `b` beside its dropped `a` would divide it by that gap.
            keep = xp.logical_and(keep, xp.logical_not(xp.any(tau != 0, axis=0)))
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
        middle = xp.any(tau != 0, axis=1) & xp.any(tau != 0, axis=0)
        if _concrete(xp.any(middle)) is not False:
            # A Jordan chain of length three in three levels (solve_2d._chain_split): the mode
            # in its middle couples onward, so its `bottom_k` holds that coupling's paths too.
            # With h_k = sum over the paths from k of f[j, k, ...] top_end -- what a kept row j
            # needs beside a dropped k -- the identity f[j, k, R] = (f[j, R] - f[k, R]) / (q_j -
            # q_k) gives h_k = (e_j top_k + sum_m tau_km h_m - bottom_k) / (q_j - q_k): `ahead`
            # plus the onward term, h_m being `ahead` at the chain's end. Still no growing
            # exp(scale q_k) of a dropped mode. Placed on the middle modes only, so a layer
            # with none keeps its bits; under a trace it is formed for every coupled layer.
            onward = xp.matmul(tau[None, :, :], ahead)                 # [j, k] sum_m tau_km h_jm
            ahead = xp.where((middle & ~keep)[None, :, None],
                             ahead + onward / denominator[:, :, None], ahead)
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
