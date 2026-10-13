"""First-order Maxwell solve for full epsilon and permeability tensors.

The transverse state is (Ex, Ey, Hx, Hy), and dpsi/dz = i k0 M psi.
Eliminating the longitudinal fields gives

    Ez = ezz^-1 (Ky Hx - Kx Hy - ezx Ex - ezy Ey)
    Hz = mzz^-1 (Kx Ey - Ky Ex - mzx Hx - mzy Hy).

All coefficients are Fourier convolution operators; products retain their
order. Uniform tensors and patterned tensors use the same layer solver.
Both incidence sides, scalar magnetic half-spaces, and all six reconstructed
field components are supported with ETM or SMM. The unstabilized TMM helper
below is a diagnostic stack solver.

Forward and backward modes are separate eigenspaces, not generally sign
pairs. Evanescent propagation factors are referenced to opposite layer faces
in the stable recursion. Finite-order PN operators can introduce artificial
loss or gain even for lossless materials, so Fourier convergence must be
checked; a finite result alone is not an accuracy guarantee.

Private top/bottom denote propagation ports. The engine binds them to the
public bottom/top ports in the right-handed xyz frame (+z up).

Validation: tests/test_full_tensor_patterns.py (independent laminate
integrals and Maxwell transfer, magnetic duality, fields and gradients),
tests/test_anisotropic_berreman.py, tests/test_field_first_order.py, and the checked-in
RETICOLO/fmmax references. No external solver code is included here.
"""
from cmath import sqrt as _csqrt

import numpy as np

from .backend import (ExceptionalPointWarning, array_namespace, as_array, eig, in_namespace,
                      linear_solve)
from .modal_gradient import _stop, split_generator, exponential, combine
from .conventions import boundary_kz, half_space_medium, layer_runs, order_azimuth, run_k0_d
from .enums import Answers, answers_of
from .solve_1d import _block, _diag, _is_backside, _mirrored_scene, final_matching
from .solve_2d import order_grid_kxy, sp_modes_2d

# Flux below this, on a mode normalised to unit energy, is taken as none — the mode is
# evanescent and direction is decided by decay instead. See split_forward_backward.
_FLAT = 1e-12
#: float64's machine epsilon; ``_FLAT`` is scaled by the working dtype's against it.
_EPS64 = 2.220446049250313e-16

# The transfer inverts a matrix holding exp(+x) and exp(-x) at once, so it loses relative
# precision like eps*exp(2x) -- measured 1.2e-8 at x = 9.1 against a predicted 1.8e-8, and 8.5e-4
# by x = 15.9. This bound is where that prediction passes ~1e-9, which is where it stops being
# usable as a cross-check. It is deliberately far below the overflow point: the earlier value of
# 300 only caught the crash, and let three decades of silently wrong answers through first
# (5.1e+25 relative error at x = 34, no complaint).
_MAX_EXPONENT = 9.0


def layer_matrix_first_order(xp, Kx, Ky, exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz_i,
                             mu_blocks=None):
    """``M`` in ``dpsi/dz = i k0 M psi`` for ``psi = (Ex, Ey, Hx, Hy)``.

    All arguments are ``n_orders x n_orders`` convolution matrices except ``Kx`` / ``Ky``, which
    are the diagonal order matrices the second-order path builds. ``ezz_i`` is the *inverse* of
    the zz block, matching how :func:`wavebit._core.solve_2d.layer_modes_2d` already carries it.
    """
    n_orders = Kx.shape[0]
    I = xp.eye(n_orders, dtype=Kx.dtype, device=getattr(Kx, 'device', None))
    O = xp.zeros_like(I)
    m = (dict(exx=I,exy=O,exz=O,eyx=O,eyy=I,eyz=O,ezx=O,ezy=O,ezz_i=I)
         if mu_blocks is None else mu_blocks)
    # Longitudinal fields expressed in the transverse state (Ex, Ey, Hx, Hy).
    ez = [-(ezz_i @ ezx), -(ezz_i @ ezy), ezz_i @ Ky, -(ezz_i @ Kx)]
    hz = [-(m['ezz_i'] @ Ky), m['ezz_i'] @ Kx,
          -(m['ezz_i'] @ m['ezx']), -(m['ezz_i'] @ m['ezy'])]
    dx = [a + exz @ z for a, z in zip((exx, exy, O, O), ez)]
    dy = [a + eyz @ z for a, z in zip((eyx, eyy, O, O), ez)]
    bx = [a + m['exz'] @ z for a, z in zip((O, O, m['exx'], m['exy']), hz)]
    by = [a + m['eyz'] @ z for a, z in zip((O, O, m['eyx'], m['eyy']), hz)]
    return _block(xp, [
        [Kx @ z + b for z, b in zip(ez, by)],
        [Ky @ z - b for z, b in zip(ez, bx)],
        [Kx @ z - d for z, d in zip(hz, dy)],
        [Ky @ z + d for z, d in zip(hz, dx)],
    ])


def mode_flux_z(xp, psi, n_orders):
    """Time-averaged Poynting ``z`` component of each column of ``psi``.

    This is the raw flux carried by the column as given, which is what a mode *amplitude*
    multiplies — see :func:`solve_uniform_stack`. For deciding direction, divide by the column
    norm first; :func:`split_forward_backward` does.
    """
    Ex, Ey = psi[:n_orders], psi[n_orders:2 * n_orders]
    Hx, Hy = psi[2 * n_orders:3 * n_orders], psi[3 * n_orders:]
    return 0.5 * xp.real(xp.sum(Ex * xp.conj(Hy) - Ey * xp.conj(Hx), axis=0))


def split_forward_backward(xp, M, n_orders, k0_d=None, *, slots=1):
    """``(lam_f, psi_f, lam_b, psi_b)``: :func:`_layer_split` without its structural word.

    A rebuilt exceptional-point pair makes ``lam_f`` and ``lam_b`` matrix generators,
    including under an abstract trace. Propagation consumers that need the coupling
    flag should call :func:`_layer_split`. ``slots`` bounds traced pairs per direction;
    eagerly every eligible pair is rebuilt.
    """
    return _layer_split(xp, M, n_orders, k0_d, slots=slots)[:4]


#: A pair of same-direction modes counts as defective when the sine of the angle between their
#: two eigenvectors is below this (:func:`_pair_gate`). Measured, not derived. Near a patterned
#: conical x-cutoff (an exceptional point, ``docs/rayleigh-points.md`` section 7) the eigenvector
#: basis rounds like ``C_E u / s**2`` through ETM and ``C_S u / s**3`` through SMM (``s`` this
#: sine, ``u`` the unit round-off; fitted at relative 1e-4, largest at PL952: ``C_E = 0.087``,
#: ``C_S = 5.6e-3``), so at the gate's edge both are below 5e-15. Over 240 random first-order
#: scenes (conical, uniform, 2D rasters, tilted magnetic tensors) the smallest sine was 3.3e-2,
#: and one scene in 240 fell under the gate. The rebuild is an exact change of basis for any pair whose subspace is
#: separated from the rest, so a larger value costs time, never accuracy.
_PARALLEL_SIN = 0.05

#: A gated pair is skipped (kept in the eigenvector basis) when another root
#: of the layer lies within this of the pair's mean root, relative to ``max(1, |root|)``: the pair
#: then has no invariant subspace of its own to rebuild, and the bordered solve that refines one
#: loses what the third root's distance costs it. Measured on a rotated Jordan block beside a
#: third root at distance ``delta`` (refusal disabled): the rebuilt pair's residual
#: ``|M B - B T| / |M|`` was 1.5e-7, 2.7e-9 and 6.0e-11 at ``delta`` = 1e-7, 1e-6, 1e-5, and at
#: most 1.2e-12 from 1e-4 on. ``eig`` spreads a defective cluster's roots by about ``sqrt(u)``,
#: so a second Jordan block at the same root sits about that far away, not at zero. The nearest
#: other root at the five measured exceptional points is 2.6e-2 away.
_PAIR_SEPARATION = 1e-4

_REFUSED_PAIR = (
    'a first-order layer holds three or more same-direction modes in a defective eigenvalue '
    'cluster (including coincident exceptional points) that no regular pair rebuild spans: '
    'the eigenvector basis was kept for them, and the answer can be inaccurate. Move the '
    'wavelength or angle by more than a relative 1e-4 to check it.')

def _pair_gate(xp, lam, psi, forward, *, slots=1):
    """``(pairs, refused, overflow)``: defective same-direction pairs of one layer.

    Within each direction the pairs are taken greedily, most parallel first, by ``argmax`` over
    the normalised eigenvectors' Gram matrix with the modes already taken masked out -- static
    shapes, so the same expression serves an abstract trace. A candidate is *engaged* when its
    sine is below :data:`_PARALLEL_SIN`, and *refused* when a third mode of its direction is that
    parallel to either member (a cluster of three, which no pair basis spans) or another root sits
    within :data:`_PAIR_SEPARATION` of the pair's mean root. A layer can hold several: the x-cutoff
    condition does not depend on ``ky``, so a y-invariant raster reaches it in every row of orders
    at once, one pair per row and direction (three at ``truncation_order = (M, 1)``).

    Read through :func:`~wavebit._core.backend._concrete`, once per candidate: eagerly ``pairs``
    lists every engaged, unrefused pair as ``(a, b, True)``, ``a`` and ``b`` length-1 index arrays,
    and is empty off the band (the caller then takes the eigenvector basis, to the bit). Under an
    abstract trace -- ``jax.jit``, ``jax.vmap``, a ``lax.map`` or ``scan`` body,
    ``jax.checkpoint``, ``torch.func.vmap`` -- it is ``slots`` candidates per
    direction with their masks as data, ``engaged & ~refused``, and every traced layer takes the
    regular-pair layout (the 1D conical rebuild's rule, ``solve_1d._cutoff_cluster``).
    ``refused`` is eager-only (``None`` under a trace): ``True`` when a gated pair had to be kept.
    ``overflow`` is trace-only (``None`` eagerly): after the last slot, one more greedy
    candidate is checked, and an eligible remainder poisons the layer's generator.
    """
    from .backend import _concrete
    n = psi.shape[1]
    dev = getattr(psi, 'device', None)
    v = psi / xp.linalg.vector_norm(psi, axis=0)[None, :]
    C = xp.abs(xp.matrix_transpose(xp.conj(v)) @ v)
    eye = xp.eye(n, dtype=xp.bool, device=dev)
    cos = 1.0 - 0.5 * _PARALLEL_SIN ** 2      # sin < g  <=>  cos > sqrt(1 - g**2) ~ 1 - g**2/2
    index = xp.arange(n, device=dev)
    magnitude = xp.maximum(xp.ones_like(xp.abs(lam)), xp.abs(lam))
    out = []
    refused = xp.zeros((), dtype=xp.bool, device=dev)
    overflow = xp.zeros((), dtype=xp.bool, device=dev)
    traced = False
    for side in (forward, ~forward):
        Cs = xp.where(side[:, None] & side[None, :] & ~eye, C, xp.zeros_like(C))
        taken = xp.zeros((n,), dtype=xp.bool, device=dev)
        for k in range(n // 2 + 1):
            free = xp.where(~taken[:, None] & ~taken[None, :], Cs, xp.zeros_like(Cs))
            flat = xp.reshape(xp.argmax(xp.reshape(free, (-1,))), (1,))
            i, j = flat // n, flat % n
            a, b = xp.minimum(i, j), xp.maximum(i, j)  # 1-D, length 1: what every `take` reads
            engaged = xp.max(free) > cos
            # a third mode as parallel to either member
            third = (xp.sum(xp.astype(xp.take(Cs, a, axis=0) > cos, xp.int32))
                     + xp.sum(xp.astype(xp.take(Cs, b, axis=0) > cos, xp.int32))) > 2
            mean = 0.5 * (xp.take(lam, a) + xp.take(lam, b))
            others = (index != a) & (index != b)
            near = xp.any(others & (xp.abs(lam - mean) <= _PAIR_SEPARATION * magnitude))
            bad = third | near
            more = _concrete(engaged)
            if more is None:              # abstract: fixed slots, masks and overflow as data
                traced = True
                if k == slots:
                    overflow = overflow | (engaged & ~bad)
                    break
            elif not more:                # eager: the first unengaged candidate ends the side
                break
            out.append((a, b, engaged & ~bad))
            refused = refused | (engaged & bad)
            taken = taken | (engaged & ((index == a) | (index == b)))
    if traced:
        return out, None, overflow
    pairs = [(a, b, True) for a, b, ok in out if _concrete(ok)]
    return pairs, _concrete(refused), None


def _schur_2x2(xp, T):
    """Unitary ``U`` with ``U^H T U`` upper triangular, for a ``2 x 2`` ``T``; closed form.

    The first column is a unit eigenvector of ``T`` for ``mu = m + sqrt(((p - s)/2)**2 + q r)``,
    ``m = (p + s)/2``, read off whichever row of ``T - mu I`` is larger; the second its unitary
    complement (the complex Schur form, G. H. Golub and C. F. Van Loan, Matrix Computations, 4th
    ed., section 7.1.3). A scalar ``T`` has no larger row and takes ``e_1``. At a 2x2 Jordan block
    ``mu`` carries the ``sqrt(u)`` of any computed defective root, but ``U^H T U`` is still ``T``
    to ``O(u)``: the dropped ``(2, 1)`` entry is ``u2^H (T u1 - mu u1)``.
    """
    p, q, r, s = T[0, 0], T[0, 1], T[1, 0], T[1, 1]
    half = 0.5 * (p - s)
    mu = 0.5 * (p + s) + xp.sqrt(half * half + q * r)
    c1 = xp.stack([q, mu - p])
    c2 = xp.stack([mu - s, r])
    n1 = xp.sum(xp.abs(c1) ** 2)
    n2 = xp.sum(xp.abs(c2) ** 2)
    u = xp.where(n1 >= n2, c1, c2)
    norm = xp.sqrt(xp.maximum(n1, n2))
    e1 = xp.astype(xp.asarray([1.0, 0.0], device=getattr(T, 'device', None)), T.dtype)
    u = xp.where(norm > 0, u / xp.where(norm > 0, norm, xp.ones_like(norm)), e1)
    return xp.stack([xp.stack([u[0], -xp.conj(u[1])]), xp.stack([u[1], xp.conj(u[0])])])


def _regular_pair(xp, M, values, psi, a, b, engaged):
    """Rebuild modes ``a`` and ``b`` in a regular basis of their invariant subspace.

    ``values`` is the layer's generator so far (``n x n``, diagonal plus earlier couplings),
    ``psi`` its basis; both come back with columns ``a`` and ``b`` replaced where ``engaged``:
    ``psi[:, (a, b)] = B``, orthonormal with ``M B = B T``, ``T`` upper triangular, its diagonal on
    ``values``' and ``T[0, 1]`` coupling row ``a`` to column ``b`` -- disjoint rows and columns,
    which :func:`~wavebit._core.modal_gradient.exponential` takes exactly.

    **Why eig's pair cannot be kept.** At a 2x2 Jordan block ``eig`` returns two eigenvectors at
    an angle ``~ sqrt(u)``, and ``Psi Lam Psi^-1`` reproduces the layer only to ``u/s`` (4e-10 at
    the patterned conical x-cutoff). Orthonormalising the two loses the subspace the same way:
    their difference is ``O(s)`` and carries ``O(u)`` of rounding out of it.

    **The refinement.** One step of Newton's method for the invariant subspace through its
    Sylvester equation (G. W. Stewart, SIAM Rev. 15, 727, 1973; J. W. Demmel, Computing 38, 43,
    1987): with ``X`` orthonormal and ``S = X^H M X`` upper triangular (:func:`_schur_2x2`),
    ``R = M X - X S``, the correction ``Z``, ``X^H Z = 0``, solves
    ``(I - X X^H) M Z - Z S = -R`` through the bordered system ``[[M - sigma I, X], [X^H, 0]]``,
    both columns in one solve with one shift ``sigma``, the mean of ``S``'s diagonal. Two terms
    of the exact step are left out, both ``O(u)``: each column's own shift differs from
    ``sigma`` by ``|S_00 - S_11|``, which multiplies ``Z``; and the second column's coupling term
    ``S_01 z_1`` multiplies ``z_1``, the first column's correction -- and that column, after the
    Schur rotation, is ``eig``'s own eigenvector, which already lies in the pair's subspace to
    ``O(u / sep)`` (an eigenvector of a nearby operator); only the second direction carries the
    ``u / s`` error. Measured: one step puts both subspaces of all five patterned exceptional
    points within 1.7e-14 of a sorted complex Schur decomposition's (SciPy) -- that
    decomposition's own accuracy -- from 1e-13 .. 1.8e-9 before it; with the coupling term, or a
    second step, the same to the digit. ``|M B - B T| <= 4e-16 |M|`` after the step, against up
    to 5e-10 for the orthonormalised pair. The bordered matrix is regular while the pair is
    separated from the other roots (:data:`_PAIR_SEPARATION`), however defective the pair itself.

    Everything is on stopped values (the derivative is the modal rule's, in this basis:
    :func:`~wavebit._core.modal_gradient.split_generator`). A lane that is not ``engaged`` --
    under a trace, a layer whose masks engaged nothing -- returns ``values`` and ``psi`` as they
    were, and its bordered matrix is swapped for the identity so no lane forms an ``inf``.
    """
    n = psi.shape[0]
    dev = getattr(psi, 'device', None)
    if isinstance(engaged, bool):          # eager: a Python flag; PyTorch's `where` wants a tensor
        engaged = xp.asarray(engaged, device=dev)
    index = xp.arange(psi.shape[1], device=dev)
    at_a, at_b = index == a, index == b    # `a`, `b`: 1-D index arrays of length 1
    va = xp.take(psi, a, axis=1)[:, 0]
    vb = xp.take(psi, b, axis=1)[:, 0]

    def orthonormal(x1, x2):
        x1 = x1 / xp.linalg.vector_norm(x1)
        w = x2 - x1 * xp.sum(xp.conj(x1) * x2)
        size = xp.linalg.vector_norm(w)
        return xp.stack([x1, w / xp.where(size > 0, size, xp.ones_like(size))], axis=1)

    def project(X):
        return xp.matrix_transpose(xp.conj(X)) @ (M @ X)

    X = orthonormal(va, vb)
    U = _schur_2x2(xp, project(X))
    X = X @ U
    S = project(X)
    residual = M @ X - X @ S
    residual = residual - X @ (xp.matrix_transpose(xp.conj(X)) @ residual)
    sigma = 0.5 * (S[0, 0] + S[1, 1])
    eye = xp.eye(n, dtype=M.dtype, device=dev)
    bordered = _block(xp, [[M - sigma * eye, X],
                           [xp.matrix_transpose(xp.conj(X)), xp.zeros((2, 2), dtype=M.dtype, device=dev)]])
    bordered = xp.where(engaged, bordered, xp.eye(n + 2, dtype=M.dtype, device=dev))
    pad = xp.zeros((2, 2), dtype=M.dtype, device=dev)
    Z = linear_solve(xp, bordered, xp.concat([-residual, pad], axis=0))[:n]
    X = orthonormal(X[:, 0] + Z[:, 0], X[:, 1] + Z[:, 1])
    U = _schur_2x2(xp, project(X))
    B = X @ U
    T = project(B)

    psi = xp.where(engaged & at_a[None, :], B[:, 0:1],
                   xp.where(engaged & at_b[None, :], B[:, 1:2], psi))
    diagonal = xp.linalg.diagonal(values)
    diagonal = xp.where(engaged & at_a, T[0, 0], xp.where(engaged & at_b, T[1, 1], diagonal))
    unit = xp.eye(values.shape[0], dtype=values.dtype, device=dev)
    coupling = xp.where(engaged & at_a[:, None] & at_b[None, :], T[0, 1], values - unit * xp.linalg.diagonal(values))
    return unit * diagonal + coupling, psi


def _layer_split(xp, M, n_orders, k0_d=None, *, slots=1):
    """``(lam_f, psi_f, lam_b, psi_b, coupled)``: split modes into propagation towards +z and -z.

    For decay-dominated eigenvalues (|Im(lambda)| > |Re(lambda)|), attenuation
    selects the direction. Otherwise normalized real Poynting flux selects it;
    negligible flux falls back to attenuation. This prevents a small artificial
    flux in a truncated PN evanescent mode from reversing its radiation branch,
    while allowing a propagating negative-index mode to have negative phase
    velocity. Exact grazing, strongly active media and poorly converged
    operators require care; the split is not a universal gain-medium criterion.

    A stable sort keeps array sizes static under JAX tracing. Eager evaluation
    refuses a split with anything other than 2*n_orders modes in each direction.

    Rebuilt pairs use matrix generators for ``lam_f`` and ``lam_b`` and return
    ``coupled=True`` so propagation and fields retain their coupling. Under an
    abstract trace this layout is structural even when no pair engages. ``slots``
    bounds greedy candidates per direction; an eligible candidate beyond that
    capacity poisons both generators with NaN. Eagerly all eligible pairs are rebuilt.

    ``k0_d`` is ``k0`` times the thickness of the layer's run of identical layers
    (:func:`_run_k0_ds`), given by the two stack algorithms (ETM and SMM) and by the coherent
    sources' cascade (``illumination._cascade``, since 2026-10-10). With it the labels are capped
    by :func:`_cap_round_trips`; without it (``None``) the split is the uncapped one above, to
    the bit. Only the unstabilised transfer (:func:`_solve_naive`) passes none.
    """
    lam, psi = eig(_stop(M))
    flux = mode_flux_z(xp, psi, n_orders) / xp.sum(xp.abs(psi) ** 2, axis=0)
    # Finite-order PN operators need not be Hermitian even for lossless
    # materials. Their evanescent modes can carry a small truncation-induced
    # real flux with either sign. For decay-dominated modes use attenuation;
    # for propagation-dominated modes use energy flow (also allowing negative
    # phase velocity). The count check below still rejects ambiguous splits.
    decay_dominated = xp.abs(xp.imag(lam)) > xp.abs(xp.real(lam))
    # `_FLAT` is about 4500 ulp of float64; the same count of ulp at the working precision. A
    # mode that carries no flux -- one of a complex quadruplet (lam, -lam, conj) of a lossless
    # patterned negative-index layer, propagation-dominated by |Re| > |Im| -- comes out of a
    # single-precision eig with a flux of ~1e-7 of either sign, which 1e-12 read as a direction:
    # JAX and Torch split such a layer 17/19 and 16/20 at complex64 and refused it. Exactly
    # `_FLAT` in double precision, where the factor is one.
    flat = _FLAT * (float(xp.finfo(flux.dtype).eps) / _EPS64)
    forward = xp.where(decay_dominated | (xp.abs(flux) <= flat),
                       xp.imag(lam) > 0, flux > 0)
    if k0_d is not None:
        forward = _cap_round_trips(xp, _stop(M), lam, psi, forward, k0_d)

    # 1.0 forward / 0.0 backward, in `flux`'s own real dtype -- asking for int64 here is what
    # made JAX warn and silently narrow to int32 when x64 is off.
    key = xp.astype(forward, flux.dtype)
    n_f = _forward_count(xp, key)
    if n_f is not None and n_f != 2 * n_orders:
        raise ValueError(
            f'mode split found {n_f} forward modes, expected {2 * n_orders}. A degenerate layer can '
            'leave an ambiguous direction; a poorly resolved patterned tensor can also '
            'introduce spurious modal flux. Increase the Fourier order and NVF resolution '
            'and check convergence. This refuses rather than guessing a mode partition.')

    order = xp.argsort(-key, stable=True)
    fwd, bwd = order[:2 * n_orders], order[2 * n_orders:]
    # A defective pair -- a patterned conical x-cutoff is a 2x2 Jordan block -- is rebuilt in a
    # regular basis (:func:`_regular_pair`); off the band nothing here runs and the split is the
    # eigenvector one, to the bit. `coupled` is structural: True wherever the layout was taken.
    pairs, refused, overflow = _pair_gate(xp, lam, psi, forward, slots=slots)
    coupled = bool(pairs)
    eig_lam = lam
    if coupled:
        frozen = _stop(M)
        values = xp.eye(lam.shape[0], dtype=lam.dtype, device=getattr(lam, 'device', None)) * lam
        for a, b, engaged in pairs:
            values, psi = _regular_pair(xp, frozen, values, psi, a, b, engaged)
        if overflow is not None:
            # A trace must not leave an eligible pair in the ill-conditioned eigenvector basis.
            values = values * xp.where(overflow, float('nan'), 1.0)
        lam = values
    # A skipped pair (a third root or a third parallel mode beside it) is loud
    # only when the cluster it sits in is defective; a degenerate eigenspace in a nearly
    # parallel basis (uniform layers, C4 rasters at normal incidence) is exact as eig left it.
    # The same screen catches coincident exceptional points whose eig vectors no pair explains.
    from .solve_2d import _unresolved_defective_cluster
    if _unresolved_defective_cluster(xp, _stop(eig_lam), _stop(psi), lambda: _stop(M),
                                     _PARALLEL_SIN, group=forward):
        import warnings
        warnings.warn(_REFUSED_PAIR, ExceptionalPointWarning, stacklevel=3)
    modal, psi = split_generator(xp, M, lam, psi, forward)
    def selected(indices):
        if modal.ndim == 1:
            return xp.take(modal, indices)
        return xp.take(xp.take(modal, indices, axis=0), indices, axis=1)
    modes = (selected(fwd), xp.take(psi, fwd, axis=1),
             selected(bwd), xp.take(psi, bwd, axis=1))
    if n_f is None:
        # An abstract trace cannot raise on a value-dependent mode count. Mark
        # the invalid solve, rather than silently taking 2N of 2N+1 forward modes.
        valid = xp.sum(key) == 2 * n_orders
        modes = tuple(xp.where(valid, a, as_array(xp, complex('nan'), dtype=a.dtype,
                                                 device=getattr(a, 'device', None))) for a in modes)
    return (*modes, coupled)


def _cap_round_trips(xp, M, lam, psi, forward, k0_d):
    """The split's labels with every round trip across the layer capped at ``eps**-0.5``.

    The first-order twin of :func:`~wavebit._core.conventions.etm_mode_q`'s cap, and the reason
    it cannot be a per-mode rule here. Both stack algorithms carry a layer's reflection state as
    ``Y = X_b R X_f``: entry ``(b, a)`` is forward mode ``a`` going down, reflecting into backward
    mode ``b`` and coming back, and its size is ``exp(rt) |R_ba|`` with the round trip
    ``rt = k0 d (Im lam_b - Im lam_a)``. A growing round trip costs about ``eps*exp(rt)``; swapping
    the two labels costs about ``eps*exp(-rt)/|R_ba|``, at worst ``exp(-rt)``. The costs cross at
    ``rt = log(1/eps)/2``, which is ``etm_mode_q``'s ``G**2 = eps**-0.5`` -- in a layer whose modes
    come in mirror pairs ``lam_b = -lam_a`` the round trip of a pair is twice the growth of its
    label, and the rule below flips exactly the pairs that rule flips.

    Two things make it a rule about *sets*, not single modes. ETM's ``P`` and SMM's interface
    matrices need ``2N`` modes on each side, so a mode can only leave the forward set if one
    enters it. And the swap of forward ``a`` with backward ``b`` is a pivot on ``R_ba``
    (``P' = E P``, ``det E = R_ba``): where ``R_ba`` is structurally zero the swap is singular. A
    uniform layer reflects each order only into itself, and a y-invariant grating each row of
    orders only into itself, so ``R`` vanishes between the invariant blocks of ``M`` (connected
    components of its coupling graph) whenever what lies below shares that invariance -- a swap
    must stay inside one block.
    Within a block every swap is regular for a generic environment.

    So, per block: rank the forward modes by ``Im lam`` ascending and the backward modes by
    ``Im lam`` descending (most growing first) and swap the ``k``-th pair while its round trip
    exceeds the cap. The round trips of ranked pairs decrease with ``k``, so the swapped set is a
    prefix, and afterwards *every* forward/backward round trip in the block is within the cap
    (the swapped modes now decay, and an unswapped pair is bounded by the first unswapped rank).
    It is also the fewest swaps that achieve that: the cap is a band of width ``rt`` in
    ``Im lam``, and the prefix is the unique balanced set outside it. ``Im lam`` itself, not the
    growth of each label: a z-coupled tensor adds a common term to every ``lam`` of an order that
    cancels in every round trip, so a label can grow while its round trip does not, and the
    reverse.

    Within each block the ranking runs over the modes the block holds by their largest component.
    Where ``eig`` mixed two blocks that share an eigenvalue (orders ``(m, n)`` and ``(m, -n)`` at
    ``ky = 0``), that count can come out unbalanced; the modes left without a partner are then
    ranked and paired among themselves, as the one block they straddle.

    What the rule cannot see is a pivot that the *environment* makes zero inside a block: a mode
    the medium below receives without any reflection (a uniaxial layer whose ordinary index equals
    its substrate's). The swap then divides by that zero -- the first-order face of the case where
    ``etm_mode_q``'s cap meets an index-matched half-space. Such a scene is ill-conditioned in its
    own data (a 1e-16 perturbation of the layer moved the answer by 1e-3 to 1 from 20 um on), so
    no labels make it accurate; the cap makes it worse than the data warrant where it swaps.

    Static shapes throughout (two stable sorts, ``searchsorted``, ``where``): traceable under
    ``jax.jit`` with a traced thickness. The labels are discrete; the modal generator is handed
    the final ones. Every real ``k_par`` in a passive layer has ``rt <= 0`` up to round-off on
    every pair, so nothing swaps there and the split is the uncapped one to the bit.
    """
    nepers = -0.25 * float(np.log(float(xp.finfo(xp.real(lam).dtype).eps)))
    h = xp.imag(lam)
    real = h.dtype
    k0_d = xp.astype(k0_d, real) if in_namespace(xp, k0_d) else as_array(
        xp, k0_d, dtype=real, device=getattr(h, 'device', None))
    backward = ~forward
    block = _mode_blocks(xp, M, psi)
    swap, hit_f, hit_b = _ranked_swaps(xp, block, h, forward, backward, k0_d, nepers)
    # the leftovers of blocks eig mixed: ranked together, as the one block they straddle
    left_f, left_b = forward & ~hit_f, backward & ~hit_b
    loose = xp.zeros_like(block)
    swap_left, _, _ = _ranked_swaps(xp, loose, h, left_f, left_b, k0_d, nepers)
    return xp.where(swap | swap_left, backward, forward)


def _ranked_swaps(xp, block, h, forward, backward, k0_d, nepers):
    """Swaps of the ranked pairs within each block whose round trip exceeds ``2*nepers``.

    Returns ``(swap, paired_forward, paired_backward)``; ``forward`` and ``backward`` here are the
    members taking part, which need not cover every mode.
    """
    n = h.shape[0]
    rank_f = _rank_within(xp, block, h, forward)
    rank_b = _rank_within(xp, block, -h, backward)
    sentinel = -1
    code_f = xp.where(forward, block * n + rank_f, sentinel)
    code_b = xp.where(backward, block * n + rank_b, sentinel)
    partner_f, hit_f = _partner(xp, code_f, code_b)
    partner_b, hit_b = _partner(xp, code_b, code_f)
    trip_f = (xp.take(h, partner_f) - h) * k0_d          # forward a against its ranked partner
    trip_b = (h - xp.take(h, partner_b)) * k0_d          # backward b against its ranked partner
    swap = ((forward & hit_f & (trip_f > 2 * nepers))
            | (backward & hit_b & (trip_b > 2 * nepers)))
    return swap, hit_f, hit_b


def _mode_blocks(xp, M, psi):
    """Each mode's invariant block: a component of ``M``'s coupling graph, as one integer per mode.

    An entry counts as a coupling above ``4N*eps*max|M|``, the backward error ``eig`` itself works
    to: a coefficient that is zero in exact arithmetic -- ``sinc(pi*m)`` of a rectangle spanning
    the cell, ``1.2e-16`` -- is not one, and nothing below that can be told from zero by the
    eigensolver either. Components by minimum-label propagation with pointer jumping (the hook
    and shortcut steps of Y. Shiloach and U. Vishkin, J. Algorithms 3, 57 (1982)), for a fixed
    ``2*bit_length(4N)`` rounds. The graphs a layer operator has -- coordinates in order-grid
    order, coupled through the harmonics a pattern holds -- settle in at most 5 rounds up to
    ``truncation_order=(20, 20)`` (26 are run); a graph that has not settled (a long chain in a
    scrambled order would not) is taken as one block, which is the cap ranked over the whole
    layer. A mode belongs to the block of its largest component.
    """
    A = xp.abs(M)
    n = A.shape[0]
    tol = n * float(xp.finfo(A.dtype).eps) * xp.max(A)
    link = (A > tol) | (xp.matrix_transpose(A) > tol)
    label = xp.arange(n, device=getattr(A, 'device', None))
    big = xp.full_like(label, n)
    def hook_and_jump(label):
        label = xp.minimum(label, xp.min(xp.where(link, label[None, :], big[:, None]), axis=1))
        return xp.take(label, label)
    for _ in range(2 * n.bit_length()):
        label = hook_and_jump(label)
    settled = xp.all(hook_and_jump(label) == label)
    label = xp.where(settled, label, xp.zeros_like(label))
    return xp.take(label, xp.argmax(xp.abs(psi), axis=0))


def _rank_within(xp, block, value, member):
    """Each member's rank among its block's members, by ``value`` ascending (ties by index)."""
    n = block.shape[0]
    keyed = xp.where(member, block, xp.full_like(block, n))
    by_value = xp.argsort(value, stable=True)
    order = xp.take(by_value, xp.argsort(xp.take(keyed, by_value), stable=True))
    sorted_block = xp.take(keyed, order)
    start = xp.searchsorted(sorted_block, sorted_block, side='left')
    rank_sorted = xp.arange(n, device=getattr(block, 'device', None)) - start
    return xp.take(rank_sorted, xp.argsort(order, stable=True))


def _partner(xp, code, other):
    """For each non-negative ``code``: the index holding it in ``other``, and whether one does."""
    n = code.shape[0]
    order = xp.argsort(other, stable=True)
    ordered = xp.take(other, order)
    at = xp.clip(xp.searchsorted(ordered, code, side='left'), 0, n - 1)
    return xp.take(order, at), (code >= 0) & (xp.take(ordered, at) == code)


_SLOTS = ('exx', 'exy', 'exz', 'eyx', 'eyy', 'eyz', 'ezx', 'ezy', 'ezz_i')

#: What a layer record gains when its split took the regular-pair layout (:func:`_layer_split`):
#: a ninth slot, so the record's *length* tells :func:`~wavebit._core.fields.field_2d_first_order`
#: to pass ``coupled=True`` -- a structural word a traced record keeps (a pytree child could not
#: be branched on). Its content is unused; ``None`` adds no leaf.
_COUPLED_RECORD = [None]


def _run_k0_ds(xp, k0, layers):
    """``(k0_ds, links)``: each layer's ``k0_d`` for :func:`_cap_round_trips` -- ``k0`` times its
    run's thickness -- and which adjacent layers join
    (:func:`~wavebit._core.conventions.layer_runs`).

    A run is adjacent layers equal, to ``sqrt(eps)``, in all nine permittivity blocks and all
    nine permeability ones (absent on both, or equal); :func:`~wavebit._core.conventions.run_k0_d`
    says why it decides once. On this path the reason bites harder than on the second-order
    one: SMM has no gap medium to join two slices through, so a slice that swapped labels its
    neighbour kept made :func:`_interface_s` hold eigenvectors twice -- 29 of them, cond 5.5e17, for
    a gyrotropic layer cut 20 um + 1.5 um at ``truncation_order = (2, 2)`` -- and the answer finite
    garbage that differed between backends. The same material written twice must join too: a tilted
    uniaxial layer as 2.5 um + 1.5 um, the thin slice a rectangle of it drawn on itself (5.6e-17
    apart), put SMM 1.6 off at ``theta = 0.2+0.6j``.
    """
    def slots(blocks):
        mu = blocks.get('mu_blocks')
        return (tuple(blocks[k] for k in _SLOTS)
                + ((None,) * len(_SLOTS) if mu is None else tuple(mu[k] for k in _SLOTS)))
    links = layer_runs(xp, [slots(e) for e, _ in layers])
    return run_k0_d(xp, k0, [d for _, d in layers], links), links


def _run_pq(xp, psi_f, psi_b, G, below, join):
    """The ETM's ``[P; Q]``, the state ``G`` below a layer in that layer's modes; exact in a run.

    ``below = (psi_f_b, psi_b_b, Y)`` is what the layer stepped just before left, ``G =
    psi_f_b + psi_b_b Y``. Where this layer continues its run (``join``) the solve is taken on
    the difference of the two layers' modes, ``[I; Y] + [psi_f, psi_b]^-1 ((psi_f_b - psi_f) +
    (psi_b_b - psi_b) Y)``, the same algebra; slices of one layer have bit-equal modes, so it is
    ``[I; Y]`` exactly and a split layer steps as the uncut one -- the reason
    :func:`~wavebit._core.solve_1d._run_ab` gives on the second-order path.
    """
    basis = xp.concat([psi_f, psi_b], axis=1)
    if join is not True:
        PQ = linear_solve(xp, basis, G)
        if join is False:
            return PQ
    f_b, b_b, Y = below
    n = Y.shape[0]
    eye = xp.eye(n, dtype=Y.dtype, device=getattr(Y, 'device', None))
    run_PQ = xp.concat([eye, Y], axis=0) + linear_solve(xp, basis, (f_b - psi_f) + (b_b - psi_b) @ Y)
    if join is True:
        return run_PQ
    return xp.where(join, run_PQ, PQ)


def _forward_count(xp, key):
    """How many modes went forward, or ``None`` when nothing concrete can be read.

    ``None`` means a tracer: ``jax.jit`` traces with abstract values and ``int()`` on one
    raises rather than returning a number. ``jax.grad`` is *not* that case — its tracer carries
    a concrete primal, so the guard still runs under a gradient. That asymmetry is the same one
    every eager decision in this library leans on, and it is tabulated in
    ``docs/backend-choice.md`` §3: a grad tracer refuses a whole-array conversion but permits
    narrowing one value; a jit tracer refuses both.

    Catching broadly rather than naming the exception keeps this backend-agnostic, which is the
    same trade :func:`wavebit._core.engine._concrete_period` makes two modules over.
    """
    try:
        return int(xp.sum(key))
    except Exception:
        return None


def vacuum_sp_states(xp, kx, ky, n_orders, order=0):
    """``(psi_s, psi_p)``: unit-amplitude incident plane waves in vacuum, as first-order states.

    The incident field has to be named rather than picked out of the vacuum eigenbasis. In
    vacuum both forward modes share ``lam = kz``, so that eigenspace is degenerate and *any*
    basis of it is a valid answer from ``eig`` — LAPACK happens to return clean s and p at
    ``ky = 0`` because ``M`` separates there, which is an artefact of the block structure and
    not a promise. Off the plane of incidence, or on another backend, the two would mix and
    "the first forward mode" would silently stop meaning p.

    With ``khat = (kx, ky, kz)`` and ``shat`` the unit vector perpendicular to the plane of
    incidence, ``curl E = i k0 H`` gives ``H = khat x E``, so::

        s:  E = shat,          H = khat x shat
        p:  E = shat x khat,   H = shat

    At normal incidence the plane of incidence is undefined and ``shat`` is taken as ``y``,
    which is the limit the oblique cases approach.
    """
    # `complex`, not `float`. The cast that was here dropped `Im(kx)`, which is round-off at a
    # real angle and not remotely round-off at a complex one: it moved the returned state by
    # 0.12 at `theta = 0.4 + 0.3j` and by 0.47 at `0.2 + 0.9j`, and what came back was no longer
    # a plane wave at all -- `H != khat x E` by the same amount. Everything below is already
    # complex-clean, so keeping the imaginary part costs nothing and changes nothing at a real
    # angle. What a complex `kt` has no answer for is the guard, immediately below.
    kx0, ky0 = complex(xp.reshape(xp.asarray(kx), (-1,))[order]), \
        complex(xp.reshape(xp.asarray(ky), (-1,))[order])
    # `cmath.sqrt`, not `** 0.5`. These are Python `complex` scalars, and `complex.__pow__`
    # goes through `exp(0.5*log)`: past the light line `1 - kt**2` is a negative real, and the
    # spelling leaves `Re(kz)` at 6.1e-17 instead of a hard 0. Unlike the array paths there is
    # no backend to hide it -- this one leaked on every backend. It is what made the incident
    # z-flux of a cut-off order read as ~1e-17 rather than as the exact 0 it is: measured
    # Re(kz) = 2.7e-20 / 4.1e-17 / 1.1e-16 at kt = 1.0000001 / 1.2 / 2.0, which is twice the
    # flux figures those cases used to report.
    kt = _csqrt(kx0 ** 2 + ky0 ** 2)
    kz = _csqrt(1.0 - kt ** 2)

    # The guard asks about `order`, not about the angle. Order m carries `k_par + G_m`, and past
    # the light line that order is evanescent in vacuum, so there is no incident plane wave for
    # this function to name -- which is the one thing it exists to do. At a real angle
    # `kt ** 2 >= 1` states exactly that.
    #
    # Two different things get called evanescent here, and only the second is an open question.
    #
    # * Real `kx`, `kt > 1`, imaginary `kz`. Refused below, and the refusal is not a missing
    #   toggle: `solve_uniform_stack` normalises by `flux(incident)`, and a purely evanescent
    #   state carries no time-averaged z-flux in a lossless medium -- measured 4.3e-01 at
    #   `kt = 0.5` against 2.0e-17 at `kt = 1.2`. Lifting the guard gives 0/0, so R and T stop
    #   being defined before the solve is even reached. The band is worth reaching -- in an
    #   emitting stack it is where the substrate, waveguided and surface-plasmon channels live
    #   -- but reaching it means a source inside the stack and a power spectrum over `k_par`,
    #   not an incident plane wave and an efficiency. A separate feature, not this guard.
    #
    # * Complex `kx` (a complex incidence angle). TODO: undefined here. Under `e^{-iwt}` with
    #   `Im(n) > 0` a complex theta is an inhomogeneous wave by construction, so "evanescent,
    #   therefore refuse" stops being the rule -- `|kt| > 1` would reject the legitimate case
    #   rather than the unnameable one. `docs/first-order-formulation.md` does not scope complex
    #   incidence and no test reaches it, so this refuses rather than guessing. The `float()`
    #   cast that used to sit above admitted it silently and returned the wrong state, and it
    #   disarmed the line below too: `theta = 0.2 + 0.9j` has a true `|kt| = 1.046` and the
    #   cast read it as 0.285.
    #
    #   The S4 field oracle added in `dacc134` does not close this either, and it is worth
    #   saying so here because it now looks like it might: it excites propagating plane waves
    #   only. Scoping a complex angle would still mean scoping it with no external check
    #   behind it.
    if abs(kt.imag) > 1e-12 * max(abs(kt), 1.0):
        raise NotImplementedError(
            f'complex incidence angle (kt = {kt:.4g}): whether an order counts as evanescent is '
            'undefined here, so the vacuum s and p states have no scoped meaning yet.')
    if kt.real ** 2 >= 1.0:
        raise ValueError(f'order {order} is evanescent in vacuum (|kt| = {kt.real:.4f}); there '
                         'is no incident plane wave to define.')

    khat = (kx0, ky0, kz)
    shat = (-ky0 / kt, kx0 / kt, 0.0) if abs(kt) > 1e-14 else (0.0, 1.0, 0.0)
    cross = lambda a, b: (a[1] * b[2] - a[2] * b[1],   # noqa: E731
                          a[2] * b[0] - a[0] * b[2],
                          a[0] * b[1] - a[1] * b[0])

    out = []
    for E, H in ((shat, cross(khat, shat)), (cross(shat, khat), shat)):
        psi = xp.zeros((4 * n_orders,), dtype=xp.complex128)
        for slot, val in ((0, E[0]), (1, E[1]), (2, H[0]), (3, H[1])):
            psi[slot * n_orders + order] = val
        out.append(psi)
    return tuple(out)


def _interface_s(xp, psi1_f, psi1_b, psi2_f, psi2_b):
    """Scattering matrix of the boundary between two media, in their own mode bases.

    The whole transverse state is continuous across the interface, so with amplitudes ``a``,

        Psi1_f a1_f + Psi1_b a1_b = Psi2_f a2_f + Psi2_b a2_b

    and grouping the two *outgoing* sets on the left gives the scattering form directly. No
    ``kz`` sign flip and no gap medium is needed here — the second-order path needs both
    because its two directions share one eigenvector, and these do not.
    """
    n = psi1_f.shape[1]
    out = xp.concat([psi1_b, -psi2_f], axis=1)
    inn = xp.concat([-psi1_f, psi2_b], axis=1)
    S = linear_solve(xp, out, inn)
    return S[:n, :n], S[:n, n:], S[n:, :n], S[n:, n:]


def _propagate_s(xp, lam_f, lam_b, k0, d, n_orders, coupled=False):
    """Scattering matrix of a uniform layer, forward amplitudes referenced at its top.

    ``coupled`` is :func:`_layer_split`'s structural word that the generators may carry a
    regular pair's coupling, for :func:`~wavebit._core.modal_gradient.exponential`.

    **This is the whole reason for the direction split.** Forward modes decay towards ``+z``
    (``Im(lam_f) >= 0``) so ``|exp(i k0 lam_f d)| <= 1``, and backward modes decay towards ``-z``
    so their propagator is bounded too. Both entries are contractions for any thickness, which is
    precisely what the eigen-decomposition transfer cannot promise: it holds the growing and
    decaying halves in one matrix and inverts it, so the growing half sets the conditioning.
    """
    zero = xp.astype(xp.zeros((2 * n_orders, 2 * n_orders)), lam_f.dtype)
    X_f = exponential(xp, lam_f, 1j*k0*d, coupled=coupled)
    X_b = exponential(xp, lam_b, -1j*k0*d, coupled=coupled)
    return zero, X_b, X_f, zero


def _solve_smm(xp, k0, Kx, Ky, layers, n_orders, psi_vf, psi_vb, *, slots=1):
    """Cascade interface and propagation scattering matrices with the Redheffer star."""
    S, _, _ = _smm_cascade(xp, k0, Kx, Ky, layers, n_orders,
                           psi_vf, psi_vb, psi_vf, psi_vb, slots=slots)
    return S


def _smm_cascade(xp, k0, Kx, Ky, layers, n_orders,
                 entry_f, entry_b, exit_f, exit_b, keep_field=False, *, slots=1):
    """Cascade distinct forward/backward modes, optionally retaining field maps.

    Each backward amplitude is referenced at its layer's exit face. A suffix
    reflection maps the forward amplitude there to that backward amplitude;
    the next interface maps it to the next layer's entrance. These are the
    same two maps consumed by the ETM field reconstruction, recovered here
    from scattering blocks without an inverse propagation factor.
    """
    from .scattering import redheffer_star

    I = xp.eye(2 * n_orders, dtype=Kx.dtype, device=getattr(Kx, 'device', None))
    S, prev_f, prev_b = None, entry_f, entry_b
    modes, interfaces, propagators = [], [], []
    layers = list(layers)
    for (eps_blocks, d), k0_d in zip(layers, _run_k0_ds(xp, k0, layers)[0]):
        lam_f, psi_f, lam_b, psi_b, coupled = _layer_split(
            xp, layer_matrix_first_order(xp, Kx, Ky, **eps_blocks), n_orders, k0_d,
            slots=slots)
        interface = _interface_s(xp, prev_f, prev_b, psi_f, psi_b)
        propagation = _propagate_s(xp, lam_f, lam_b, k0, d, n_orders, coupled)
        for part in (interface, propagation):
            S = part if S is None else redheffer_star(xp, I, S, part)
        if keep_field:
            modes.append(([eps_blocks, psi_f, lam_f, psi_b, lam_b, d], _COUPLED_RECORD * coupled))
            interfaces.append(interface)
            propagators.append(propagation)
        prev_f, prev_b = psi_f, psi_b
    tail = _interface_s(xp, prev_f, prev_b, exit_f, exit_b)
    S = tail if S is None else redheffer_star(xp, I, S, tail)
    if not modes:
        return S, [], None

    reflection, advance = tail[0], tail[2]
    layer_info_list = []
    for (mode, marker), interface, propagation in reversed(list(zip(modes, interfaces, propagators))):
        layer_info_list.append(mode + [advance, reflection] + marker)
        # Reflection at the entrance of this layer, including both bounded
        # propagation factors. Close the loop across its entrance interface.
        reflected_at_entry = propagation[1] @ reflection @ propagation[2]
        advance = linear_solve(xp, I - interface[3] @ reflected_at_entry, interface[2])
        reflection = interface[0] + interface[1] @ reflected_at_entry @ advance
    return S, layer_info_list, advance


def _forward_amplitudes(xp, incident, psi_vf, psi_vb, n_orders):
    """Mode amplitudes of an incident state, and a check that it is purely incident.

    Decomposing over the *full* basis rather than solving against ``psi_vf`` alone keeps the
    system square and costs nothing, since the backward half is the thing worth testing: a state
    with any backward content is not an incident wave, and would otherwise be silently projected.
    """
    coef = linear_solve(xp, xp.concat([psi_vf, psi_vb], axis=1),
                            xp.reshape(incident, (-1, 1)))[:, 0]
    leak = float(xp.max(xp.abs(coef[2 * n_orders:])))
    if leak > 1e-10 * max(float(xp.max(xp.abs(coef[:2 * n_orders]))), 1e-300):
        raise ValueError(f'the incident state has backward content ({leak:.2e}); it is not a '
                         'wave arriving from the top. Build it with vacuum_sp_states.')
    return coef[:2 * n_orders]


def _solve_etm(xp, k0, Kx, Ky, layers, n_orders, incident, psi_vf, psi_vb, *, slots=1):
    """Enhanced transmittance matrix — the second-order path's stable transfer, carried over here.

    Same idea as :func:`wavebit._core.solve_2d.solve_2d`'s ``big_A_i`` / ``big_T`` recursion
    (Moharam et al. 1995), which is why the second-order transfer is stable while the naive one in
    :func:`_solve_naive` is not. The growth is not removed by better conditioning; it is arranged
    never to appear.

    Sweeping upward, suppose the state below a layer is ``psi = G t`` for an unknown ``t``.
    Decomposing in that layer's modes gives ``a_f = X_f^-1 P t`` — and ``X_f^-1`` is exactly the
    growing factor. ETM absorbs it into the unknown, ``t' = X_f^-1 P t``, leaving

        G_above = Psi_f + Psi_b @ X_b @ (Q P^-1) @ X_f

    in which ``X_f`` and ``X_b`` appear only as the bounded, decaying factors they are. The
    redefinition is bookkept separately as ``P^-1 X_f`` per layer, so the transmitted amplitude
    can be recovered at the end.
    """
    G, T_acc, _ = _etm_recursion(xp, k0, Kx, Ky, layers, n_orders, psi_vf,
                                keep_field=False, slots=slots)

    # incident + reflected on the top side must be whatever the recursion says arrives there
    sol = linear_solve(xp, xp.concat([psi_vb, -G], axis=1),
                           xp.reshape(-incident, (-1, 1)))[:, 0]
    return psi_vb @ sol[:2 * n_orders], psi_vf @ (T_acc @ sol[2 * n_orders:])


def _etm_recursion(xp, k0, Kx, Ky, layers, n_orders, G, keep_field=True, *, slots=1):
    """Sweep the ETM recursion up the stack: ``(G_above, T_acc, layer_info_list)``.

    Split out of :func:`_solve_etm` because the recursion is the part
    :func:`solve_2d_first_order` shares — it seeds a half-space of index ``n_bot`` rather than
    vacuum and hands the result to the s/p matching instead of matching against a vacuum basis here.
    The arithmetic is untouched by the split.

    ``G`` maps an unknown amplitude vector to the transverse state; whatever coordinates the
    seed's columns are in, ``T_acc`` maps the *returned* ``G``'s unknown back into them. The two
    are only meaningful together, which is why they are returned together: ``big_T`` feeds the
    transmitted amplitude directly, so a per-mode scale that cancels in the reflection does not
    cancel there.

    ``layer_info_list`` is what field reconstruction propagates — the counterpart of the
    ``[epz_conv_i, W, V, q, d, big_A_i, big_B]`` records
    :func:`wavebit._core.solve_2d.solve_2d` leaves behind, appended **bottom-to-top** for the
    same reason (:func:`wavebit._core.fields.field_2d_first_order` walks it in reverse). Each entry
    is ``[eps_blocks, psi_f, lam_f, psi_b, lam_b, d, P_i, QP_i]``, and the three pieces that are
    not simply the layer's modes are exactly the two amplitude maps this loop already forms:

    * ``P_i @ X_f`` carries the unknown *down* one layer (``t = P_i X_f t'``), which is what
      makes ``T_acc`` the product of them;
    * ``QP_i @ X_f`` turns the same unknown into the backward amplitudes referenced at the
      layer's **bottom** face, which is where ``X_b`` in the ``G`` update starts from.

    ``eps_blocks`` rides along because ``E_z`` needs ``ezz_i``, ``ezx`` and ``ezy`` — the one
    part of the reconstruction the second-order path cannot lend.
    """
    T_acc = xp.astype(xp.eye(2 * n_orders), Kx.dtype)
    layer_info_list = []
    layers = list(layers)
    run, links = _run_k0_ds(xp, k0, layers)
    below = None
    for index in range(len(layers) - 1, -1, -1):
        eps_blocks, d = layers[index]
        lam_f, psi_f, lam_b, psi_b, coupled = _layer_split(
            xp, layer_matrix_first_order(xp, Kx, Ky, **eps_blocks), n_orders, run[index],
            slots=slots)
        X_f = exponential(xp, lam_f, 1j*k0*d, coupled=coupled)
        X_b = exponential(xp, lam_b, -1j*k0*d, coupled=coupled)

        PQ = _run_pq(xp, psi_f, psi_b, G, below, False if below is None else links[index])
        P_i = xp.linalg.inv(PQ[:2 * n_orders])
        QP_i = PQ[2 * n_orders:] @ P_i
        G = psi_f + psi_b @ X_b @ QP_i @ X_f
        T_acc = T_acc @ (P_i @ X_f)
        # Kept only while the next layer up may continue this one's run: `G` in this layer's
        # modes, `[I; Y]`, for :func:`_run_pq`.
        below = (None if index == 0 or links[index - 1] is False
                 else (psi_f, psi_b, X_b @ QP_i @ X_f))

        # `keep_field=False` lets each layer's eight-slot state be collected as the loop
        # moves on; only `G` and `T_acc` have to survive an iteration. `_solve_etm` above
        # discards the list outright and so never builds it.
        if keep_field:
            layer_info_list.append([eps_blocks, psi_f, lam_f, psi_b, lam_b, d, P_i, QP_i]
                                   + _COUPLED_RECORD * coupled)
    return G, T_acc, layer_info_list


def _solve_naive(xp, k0, Kx, Ky, layers, n_orders, incident, psi_vf, psi_vb, *, slots=1):
    """Unstabilised eigen-decomposition transfer, kept only as a third opinion.

    Not ``layer_algorithm='ETM'``, which is the enhanced form in :func:`_solve_etm` and is what
    :mod:`wavebit._core.solve_2d` uses. This one exists so the stabilisation can be shown to
    be doing something, and refuses where it would be inaccurate.
    """
    P = xp.astype(xp.eye(4 * n_orders), Kx.dtype)
    for eps_blocks, d in layers:
        lam_f, psi_f, lam_b, psi_b, coupled = _layer_split(
            xp, layer_matrix_first_order(xp, Kx, Ky, **eps_blocks), n_orders, slots=slots)
        lam = combine(xp, lam_f, lam_b)
        exponent = k0 * float(xp.max(xp.abs(xp.imag(lam)))) * float(d)
        if exponent > _MAX_EXPONENT:
            raise ValueError(
                f'layer exponent {exponent:.1f} is past what the unstabilised transfer can '
                "carry in float64. Use layer_algorithm='ETM' or 'SMM', either of "
                'which is bounded for any thickness.')
        Psi = xp.concat([psi_f, psi_b], axis=1)
        E = exponential(xp, lam, 1j*k0*d, coupled=coupled)
        P = (Psi @ E @ xp.linalg.inv(Psi)) @ P

    # Matching at both interfaces at once: incident + reflected on the input side must transfer
    # to transmitted only on the output side, so P (a + psi_vb r) = psi_vf t.
    A = xp.concat([P @ psi_vb, -psi_vf], axis=1)
    sol = linear_solve(xp, A, xp.reshape(-(P @ incident), (-1, 1)))[:, 0]
    return psi_vb @ sol[:2 * n_orders], psi_vf @ sol[2 * n_orders:]


def solve_uniform_stack(wavelength, kx, ky, layers, n_orders, incident, layer_algorithm='ETM'):
    """``(R, T)`` for one incident state, through a stack of uniform anisotropic layers.

    ``layers`` is ``[(eps_blocks, thickness), ...]`` where ``eps_blocks`` is the mapping
    :func:`layer_matrix_first_order` takes. ``incident`` is a first-order state — see
    :func:`vacuum_sp_states`. Vacuum on both sides, and note it is *not* special-cased: vacuum
    is a layer with ``eps = I`` whose forward and backward halves come from the same mode split
    as any other, which is what would make an anisotropic substrate free here. The second-order path
    cannot say that, and neither can RETICOLO.

    ``R`` and ``T`` are fluxes of the *reconstructed* reflected and transmitted fields, not sums
    of per-mode fluxes. Those agree only when the modes are flux-orthogonal, which holds for a
    lossless medium and stops holding exactly where it would be least noticed.

    ``layer_algorithm`` selects how layers are joined. Three, and the choice matters here in a
    way it does not in the second-order path, where both shipped options are already stable:

    * ``'ETM'`` (default) is the enhanced-transmittance recursion — **the same option, under
      the same name, as the second-order path's default**. Growth never appears rather than being
      cancelled. Measured the cheapest of the three — 0.94x the plain transfer on a 40-layer
      stack, against SMM's 1.13x — and identical to SMM at 1.2e-15.
    * ``'SMM'`` cascades per-layer scattering matrices with the Redheffer star. Same stability,
      slightly dearer, and the formulation most of the literature states its results in. Its
      per-layer matrices are also independent of their neighbours, which is what would let them
      be cached or reordered.
    * ``'TMM'`` propagates ``Psi exp(i k0 lam d) Psi^-1``, holding the growing and decaying
      halves in one matrix and inverting it. Deliberately kept in its naive form as a third
      opinion, and it **raises** past the point where it is accurate — see ``_MAX_EXPONENT``.
    """
    xp = array_namespace(kx)
    k0 = 2 * 3.141592653589793 / wavelength
    Kx, Ky = _diag_of(xp, kx), _diag_of(xp, ky)

    vac = _identity_blocks(xp, n_orders, Kx.dtype)
    _, psi_vf, _, psi_vb = split_forward_backward(
        xp, layer_matrix_first_order(xp, Kx, Ky, **vac), n_orders)

    if layer_algorithm == 'SMM':
        S11, _, S21, _ = _solve_smm(xp, k0, Kx, Ky, layers, n_orders, psi_vf, psi_vb)
        a_inc = _forward_amplitudes(xp, incident, psi_vf, psi_vb, n_orders)
        reflected, transmitted = psi_vb @ (S11 @ a_inc), psi_vf @ (S21 @ a_inc)
    elif layer_algorithm == 'ETM':
        reflected, transmitted = _solve_etm(xp, k0, Kx, Ky, layers, n_orders, incident, psi_vf, psi_vb)
    elif layer_algorithm == 'TMM':
        reflected, transmitted = _solve_naive(xp, k0, Kx, Ky, layers, n_orders, incident,
                                              psi_vf, psi_vb)
    else:
        raise ValueError(
            f"layer_algorithm must be 'ETM', 'SMM' or 'TMM', got {layer_algorithm!r}")

    def flux(v):
        return float(mode_flux_z(xp, xp.reshape(v, (-1, 1)), n_orders)[0])

    inc = flux(incident)
    return -flux(reflected) / inc, flux(transmitted) / inc


def solve_2d_first_order(wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
                         epx_conv_all, epy_conv_all, epz_conv_i_all,
                         epxy_conv_all=None, epyx_conv_all=None,
                         epxz_conv_all=None, epyz_conv_all=None,
                         epzx_conv_all=None, epzy_conv_all=None,
                         *, phi, incidence_side='top', answers=Answers.EFFICIENCY, lit=None,
                         layer_algorithm='ETM', mu=None, mu_top=1.0, mu_bot=1.0,
                         eps_top=None, eps_bot=None):
    """Solve a 2D stack on the first-order formulation — the public route to it.

    Twin of :func:`wavebit._core.solve_2d.solve_2d`: same arguments, same returns, four more
    convolution slots. ``epxz_conv_all`` / ``epyz_conv_all`` / ``epzx_conv_all`` /
    ``epzy_conv_all`` are the ``z`` row and column the second-order form cannot express;
    ``None`` (the default) is zero, and a call giving none of them is the same physical problem
    :func:`~wavebit._core.solve_2d.solve_2d` solves, at about 8x the cost.

    **All four carry the same slot convention as the in-plane pair: the slot holds the
    negative of the tensor component.** ``epxz = -eps_xz``, and so on for the other three, so
    that one rule covers every off-diagonal slot in the signature rather than two rules split
    down the middle of it. ``epz_conv_i`` is still the *inverse* of the ``zz`` block. See
    ``docs/off-diagonal-slot-convention.md``.

    Three pieces are shared with the second-order path rather than re-derived, and this is where the
    mode correspondence at the top of this module is spent:

    * the **substrate seed** — :func:`_half_space_state` writes the same downward half-space
      wave the second-order seed
      ``(big_F, big_G) = ([[I, 0], [0, -i Kz/eps]], [[-i Kz, 0], [0, I]])`` spells, as a
      first-order state whose two columns are that path's ``(t_s, t_p)``;
    * the **s/p rotation** — :func:`~wavebit._core.solve_2d.sp_modes_2d`, given the
      recursion's ``G`` split as ``W = G[:2 * n_orders]`` and ``V = 1j*G[2 * n_orders:]``. That
      ``1j`` is the whole correspondence: with it the four boundary variables come out in the basis
      :func:`~wavebit._core.solve_1d.final_matching` already reads;
    * the **matching** — ``final_matching`` itself, so the incidence, the efficiencies, the
      Fresnel normalisation and the pure-s/pure-p variants are the same code the rest of the
      library uses, not a second opinion about them.

    ``big_T`` is ``T_acc`` unchanged, with no rescaling, *because* the seed's columns are
    already ``(t_s, t_p)``: the transmitted amplitude is read in the coordinates the seed
    named, so the per-mode scale that would cancel in ``R`` and survive in ``T`` never enters.
    Feeding ``eig``'s half-space modes instead — the obvious alternative — is what puts an
    arbitrary column scale between the two, and it is invisible in reflection.

    ``incidence_side='bottom'`` is the same mirrored scene the other solves use, **plus a sign
    flip on all four z-row slots**: ``z -> -z`` sends ``eps_xz -> -eps_xz`` (and likewise
    ``zx``, ``yz``, ``zy``) while leaving the in-plane block and ``eps_zz`` alone, which is why
    the shared :func:`~wavebit._core.solve_1d._mirrored_scene` cannot be the whole of it
    here. Pinned against a Berreman stack with no mirror in it at all
    (``tests/test_anisotropic_berreman.py``). This kernel ``'bottom'`` is the port, not the
    public side: since 2026-09-21 the engine sends public *top* incidence here, so the
    top-lit Berreman comparisons are the ones that reach this branch (measured with a call
    counter on ``_mirrored_scene``); the bottom-lit one reaches the unmirrored solve.

    ``ETM`` uses the shared final matching. ``SMM`` cascades interfaces and
    propagation in the separate forward/backward mode bases and uses the
    same s/p amplitude and power conversion as the second-order SMM. Both return
    the same field reconstruction layout; SMM also retains its scattering
    matrix from ``answers='amplitude'`` upward.
    """
    xp = array_namespace(epx_conv_all)
    if layer_algorithm not in ('ETM', 'SMM'):
        raise ValueError(
            f"layer_algorithm={layer_algorithm!r} is not available on the first-order "
            "path. Use layer_algorithm='ETM' or 'SMM'.")

    per_layer = [epx_conv_all, epy_conv_all, epz_conv_i_all, epxy_conv_all, epyx_conv_all,
                 epxz_conv_all, epyz_conv_all, epzx_conv_all, epzy_conv_all]
    mu_all = [None] * 9 if mu is None else list(mu)
    if len(mu_all) not in (3, 5, 9):
        raise ValueError('mu requires three, five or nine convolution slots')
    mu_all += [None] * (9 - len(mu_all))
    if _is_backside(incidence_side):
        n_top, n_bot, thickness, *stacks = _mirrored_scene(
            n_top, n_bot, thickness, *per_layer, *mu_all)
        per_layer, mu_all = stacks[:9], stacks[9:]
        # Reflection in z changes both epsilon and mu z off-diagonal slots.
        def mirror_z(slots):
            return slots[:5] + [None if a is None else [-m for m in a] for a in slots[5:]]
        per_layer, mu_all = mirror_z(per_layer), mirror_z(mu_all)
        mu_top, mu_bot = mu_bot, mu_top
        eps_top, eps_bot = eps_bot, eps_top
    # Read before the recovery below, which fills the permittivities in: only a caller that
    # held them can name a double negative (`half_space_medium`).
    medium_top = half_space_medium(eps_top, mu_top)
    medium_bot = half_space_medium(eps_bot, mu_bot)
    if eps_top is None:
        eps_top = n_top ** 2 / mu_top
    if eps_bot is None:
        eps_bot = n_bot ** 2 / mu_bot

    kx = as_array(xp, kx)
    ky = as_array(xp, ky)
    if not xp.isdtype(kx.dtype, 'complex floating'):
        kx = xp.astype(kx, xp.complex128)
        ky = xp.astype(ky, xp.complex128)
    C = kx.dtype

    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    n_orders_v, n_orders_u = kx_g.shape
    n_orders = n_orders_u * n_orders_v
    from .modal_gradient import _COUPLED_CAP
    # One exceptional-point pair may occur in every row of an invariant raster.
    # The traced exponential's capacity bounds how many each direction can retain.
    slots = min(max(n_orders_u, n_orders_v), _COUPLED_CAP)
    k0 = 2 * 3.141592653589793 / wavelength
    theta = 0.0 if theta is None else theta

    k_par_sq = kx_g ** 2 + ky_g ** 2
    kz_top = xp.reshape(boundary_kz(xp, n_top ** 2, k_par_sq, medium_top), (-1,))
    kz_bot = xp.reshape(boundary_kz(xp, n_bot ** 2, k_par_sq, medium_bot), (-1,))
    varphi = order_azimuth(xp, ky_g, kx_g, phi)

    Kx = _diag(xp, xp.reshape(kx_g, (-1,)))
    Ky = _diag(xp, xp.reshape(ky_g, (-1,)))
    F_c = _diag(xp, xp.cos(varphi))
    F_s = _diag(xp, xp.sin(varphi))

    layers = _conv_layers(xp, per_layer, thickness, n_orders, C)
    if any(a is not None for a in mu_all):
        for (eps, _), (magnetic, _) in zip(layers, _conv_layers(xp, mu_all, thickness, n_orders, C)):
            eps['mu_blocks'] = magnetic
    seed = _half_space_state(xp, _diag(xp, kz_bot), eps_bot, F_c, F_s, n_orders, C, mu_bot)
    keep_field = answers_of(answers).covers(Answers.FIELD)
    if layer_algorithm == 'SMM':
        from .scattering import _sp_scattering_result

        entry_f = _half_space_state(xp, _diag(xp, kz_top), eps_top,
                                    F_c, F_s, n_orders, C, mu_top)
        # In the shared s/p port convention the backward wave keeps its
        # tangential electric components and reverses its magnetic ones.
        def backward(state):
            return xp.concat([state[:2 * n_orders], -state[2 * n_orders:]], axis=0)

        S, layer_info_list, entry_map = _smm_cascade(
            xp, k0, Kx, Ky, layers, n_orders,
            entry_f, backward(entry_f), seed, backward(seed), keep_field, slots=slots)
        # The scene was already mirrored above when needed, including both
        # material tensors. Its ports are now incidence-relative.
        out, c_inc = _sp_scattering_result(
            xp, C, S, kx, ky, kz_top, kz_bot, n_top, n_bot, incidence, False,
            answers, mu_top, mu_bot, eps_top, eps_bot, lit)
        if keep_field:
            T1 = None if entry_map is None else xp.matrix_transpose(entry_map @ c_inc)[..., None]
            out.update(layer_info_list=layer_info_list, T1=T1)
        return out

    G, big_T, layer_info_list = _etm_recursion(xp, k0, Kx, Ky, layers, n_orders, seed,
                                               keep_field=keep_field, slots=slots)
    big_F, big_G = sp_modes_2d(xp, G[:2 * n_orders], 1j * G[2 * n_orders:], F_c, F_s, n_orders)

    out = final_matching(xp, C, big_F, big_G, big_T, kz_top, kz_bot, incidence, theta,
                         n_top, n_bot, n_orders_u, n_orders_v, answers=answers,
                         lit=lit, mu_top=mu_top, mu_bot=mu_bot,
                         eps_top=eps_top, eps_bot=eps_bot, kx=kx, ky=ky)
    if keep_field:
        # `sp_modes_2d` recombines the *rows* of `G` into the s/p boundary variables and leaves
        # its columns alone, so `final_matching`'s `T1` is the unknown of the `G` above the top
        # layer unchanged -- which is that layer's forward amplitudes at its own top face. That
        # is what `field_2d_first_order` starts from; nothing has to be undone here.
        out['layer_info_list'] = layer_info_list
    return out


def _half_space_state(xp, Kz, eps, F_c, F_s, n_orders, C, mu=1.0):
    """The downward half-space wave as a first-order state, columns ``(t_s, t_p)``.

    This is the second-order substrate seed, written in the variables the first-order system
    carries. Reading it off rather than deriving it: the seed must satisfy
    ``sp_modes_2d(seed[:2 * n_orders], 1j*seed[2 * n_orders:]) == (big_F, big_G)`` for the pair
    :func:`wavebit._core.solve_2d.solve_2d` starts from, which fixes the four transverse
    components per column with nothing left to choose ::

        E_s = t_s              E_p = -i kz t_p / eps
        H_p = -kz t_s          H_s = -i t_p

    and the ``(s, p) -> (x, y)`` rotation ``Ax = F_c A_p - F_s A_s``,
    ``Ay = F_s A_p + F_c A_s`` puts them in the state's own basis. ``Kz`` comes from
    :func:`~wavebit._core.conventions.boundary_kz`, so this path takes the half-space
    branch from the same place every other solve does rather than from the flux rule
    :func:`split_forward_backward` applies inside a layer.

    Building the seed instead of eigendecomposing the half-space is not only cheaper. An
    isotropic half-space has both forward modes at the same ``kz``, so that eigenspace is
    degenerate and ``eig`` may return any basis of it — including one that mixes s and p, and
    one whose column scales would land straight in ``big_T``.
    """
    I = xp.astype(xp.eye(n_orders), C)
    E_s, E_p = I, -1j * Kz / eps          # per unit t_s, per unit t_p
    H_p, H_s = -Kz / mu, -1j * I
    return _block(xp, [
        [-F_s @ E_s, F_c @ E_p],          # Ex = F_c E_p - F_s E_s
        [F_c @ E_s, F_s @ E_p],           # Ey = F_s E_p + F_c E_s
        [F_c @ H_p, -F_s @ H_s],          # Hx = F_c H_p - F_s H_s
        [F_s @ H_p, F_c @ H_s],           # Hy = F_s H_p + F_c H_s
    ])


def _conv_layers(xp, per_layer, thickness, n_orders, C):
    """``[(eps_blocks, d), ...]`` for :func:`layer_matrix_first_order`, from the nine slot stacks.

    Where the slot convention is undone: every off-diagonal slot holds the negative of its
    tensor component (``docs/off-diagonal-slot-convention.md``), and ``layer_matrix_first_order``
    reads the tensor itself. A missing stack is zero, matching how the second-order path treats
    ``epxy_conv_all`` / ``epyx_conv_all``.
    """
    epx, epy, epz_i, epxy, epyx, epxz, epyz, epzx, epzy = per_layer
    O = xp.astype(xp.zeros((n_orders, n_orders)), C)
    n_layers = len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]

    I = xp.eye(n_orders, dtype=C, device=getattr(O, "device", None))
    def diagonal(stack, i):
        return I if stack is None else stack[i]
    def slot(stack, i):
        return O if stack is None else -stack[i]

    return [(dict(exx=diagonal(epx, i), eyy=diagonal(epy, i), ezz_i=diagonal(epz_i, i),
                  exy=slot(epxy, i), eyx=slot(epyx, i),
                  exz=slot(epxz, i), eyz=slot(epyz, i),
                  ezx=slot(epzx, i), ezy=slot(epzy, i)),
             thickness[i])
            for i in range(n_layers)]


def _diag_of(xp, v):
    return _diag(xp, xp.reshape(xp.asarray(v), (-1,)))


def _identity_blocks(xp, n_orders, dtype):
    I = xp.astype(xp.eye(n_orders), dtype)
    O = xp.astype(xp.zeros((n_orders, n_orders)), dtype)
    return dict(exx=I, exy=O, exz=O, eyx=O, eyy=I, eyz=O, ezx=O, ezy=O, ezz_i=I)
