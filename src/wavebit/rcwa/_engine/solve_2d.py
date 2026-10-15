"""Single-source 2D RCWA solve — Phase 3c of the rewrite.

As in ``solve_1d``, private ``top``/``bottom`` port names mean the start/end of the
positive-z layer list. The public engine maps physical media and illumination onto them.

Port of the triplicated ``transfer_2d_1..4`` chain into one backend-agnostic solve against the
Array-API namespace, sharing the boundary/branch conventions and the final s/p matching with
:mod:`wavebit.rcwa._engine.solve_1d`. The only per-backend primitive is
:func:`wavebit._common.backend.eig`.

The 2D layer couples both polarizations *and* both in-plane directions, so per-layer modes and
the boundary system are ``2*n_orders`` block matrices (``n_orders = n_orders_u*n_orders_v``).
Optional off-axis convolution blocks (``epxy`` / ``epyx``, produced by the NVF factorization;
``None`` = zeros) enter the mode operator. ``transfer_2d_1`` and ``transfer_2d_4`` are identical
to the conical versions, so this module reuses ``solve_1d``'s boundary init and
``final_matching`` verbatim.
"""
from wavebit._common.backend import _concrete
from wavebit._common.backend import _readable
from wavebit._common.backend import array_namespace
from wavebit._common.backend import as_array
from wavebit._common.backend import carries_autodiff
from wavebit._common.backend import check_mode_cutoff
from wavebit._common.backend import checked_float
from wavebit._common.backend import division_at_a_cutoff
from wavebit._common.backend import eig
from wavebit._common.backend import linear_solve
from wavebit._common.conventions import boundary_kz
from wavebit._common.conventions import half_space_medium
from wavebit._common.conventions import layer_label_rule
from wavebit._common.conventions import layer_root
from wavebit._common.conventions import order_azimuth
from wavebit._common.conventions import power_down_labels
from wavebit.rcwa._engine.enums import Answers
from wavebit.rcwa._engine.enums import answers_of
from wavebit.rcwa._engine.solve_1d import _block
from wavebit.rcwa._engine.solve_1d import _diag
from wavebit.rcwa._engine.solve_1d import _env_decides
from wavebit.rcwa._engine.solve_1d import _is_backside
from wavebit.rcwa._engine.solve_1d import _keep_below
from wavebit.rcwa._engine.solve_1d import _mirrored_scene
from wavebit.rcwa._engine.solve_1d import _run_ab
from wavebit.rcwa._engine.solve_1d import _run_join
from wavebit.rcwa._engine.solve_1d import _run_k0_d
from wavebit.rcwa._engine.solve_1d import final_matching
from wavebit.rcwa._engine.solve_1d import mode_flux

# Engage the coupled-cluster rebuild (see `_rebuild_grazing_cluster`) below this
# min|w|/max|w| ratio. Well above `backend._CUTOFF_REL` (the warn threshold) on purpose,
# for two measured reasons: the product-eigvec construction degrades as ~eps/ratio through
# the whole band (2e-6 at 1e-12, per ``docs/theory/rcwa/rayleigh-points.md`` §3) while the rebuild is
# exact everywhere in it; and `solve`'s default detune step (1e-12 relative) lands a
# rescued solve at ratio ~1e-11 — with a 1e-12 gate the retry fell back into the shipped
# construction's degraded band and returned 1e-6..8e-6 errors on the harder graze stacks,
# with 1e-9 it lands in the rebuild's regime and matches the extrapolated exact-point
# oracle to its own uncertainty. Above the gate the ordinary product modes are returned;
# a random non-grazing solve reaches ratio 1e-9 with probability ~1e-8
# per solve (an order within 4e-9 of a cutoff), measured as 0 engagements over the
# 40-config validation sweep.
_COUPLED_REL = 1e-9

# Below this modal rate (relative to sqrt(max|w|), the layer's largest rate) a rebuilt
# cluster mode is numerically indistinguishable from the exactly-defective ``{1, z}`` pair,
# and no exponential mode basis represents the layer: the intrinsic error of any such basis
# is ~eps/(k0*d*|mu|). Measured on exact-graze stacks: |mu| = 2e-9 solves to 6.5e-9, while
# |mu| ~ 5e-16 is wrong at 1e-4..1e-3 across every basis choice tried (eigvec pairing,
# volume-maximal selection, |mu| floors — the floor repeats the epsilon-floor dead end of
# ``docs/theory/rcwa/rayleigh-points.md`` §5 in miniature: any q inconsistent with its own mode pair is
# amplified by exactly the conditioning that consistency lets cancel). 1e-10 puts the
# representable side at <= ~1e-6 for k0*d >~ 0.4; below it the only honest outcome is the
# loud refusal, which `solve`'s default ladder then retries off the condition.
_DEFECTIVE_REL = 1e-10


def order_grid_kxy(xp, kx, ky):
    """``kx`` / ``ky`` on the ``(n_orders_v, n_orders_u)`` order grid, from either spelling.

    Two ways in, because a lattice basis decides which one is even expressible:

    * **1-D**, lengths ``(n_orders_u,)`` and ``(n_orders_v,)`` — the separable form an
      axis-aligned cell
      admits, where ``kx`` depends only on ``m`` and ``ky`` only on ``n``. Broadcast to the
      grid here, which is what both solves used to do inline.
    * **2-D**, ``(n_orders_v, n_orders_u)`` each — the per-order Cartesian components, which is
      the only
      form a skewed cell has: with ``G = m*b1 + n*b2`` non-orthogonal, *both* components
      depend on *both* indices and no pair of 1-D vectors can hold them. See
      :func:`wavebit.lattice.order_kxy`.

    Everything downstream reads the grid, so the two spellings converge here and the algebra
    below never branches. The 1-D route broadcasts exactly as before, element for element.
    """
    if kx.ndim == 2:
        return kx, ky
    n_orders_u, n_orders_v = kx.shape[0], ky.shape[0]
    return (xp.broadcast_to(kx[None, :], (n_orders_v, n_orders_u)),
            xp.broadcast_to(xp.reshape(ky, (-1, 1)), (n_orders_v, n_orders_u)))


def mode_flux_2d(xp, W, V):
    """:func:`~wavebit.rcwa._engine.solve_1d.mode_flux` of 2D layer modes, from the unrotated pair.

    ``W = [Ex; Ey]`` and ``V`` its partner, which :func:`sp_modes_2d` turns into ``(E_s, F_p) =
    (Ey, Ex)`` and ``(H_s, H_p) = (V_x, V_y)`` at zero azimuth; the per-order rotation it
    applies otherwise leaves the flux as it is. So ``X = (V_x, Ex)`` against ``Y = (Ey, V_y)``.
    """
    n = W.shape[0] // 2
    return mode_flux(xp, xp.concat([V[:n], W[:n]], axis=0), xp.concat([W[n:], V[n:]], axis=0))


def _power_down_2d(xp, W, V, q, k0_d, patterned):
    """:func:`~wavebit._common.conventions.power_down_labels` on 2D modes; ``V`` turns with ``q``."""
    q, flip = power_down_labels(xp, q, *mode_flux_2d(xp, W, V), k0_d, patterned)
    return xp.where(flip, -V, V), q


def _rebuild_grazing_cluster(xp, w, W, V, q, Omega_L, Omega_R, k0_d=None, mirrored=None):
    """Re-derive the grazing eigenvalue cluster from the coupled first-order problem.

    Returns ``(W, V, q)`` as NumPy arrays with the cluster columns replaced, ``None`` to
    bail (caller keeps the shipped construction and its warning), or raises
    ``FloatingPointError`` when the cluster is numerically defective.

    Why the shipped construction cannot be repaired in place: the cluster eigvecs of the
    product ``Omega2 = Omega_L @ Omega_R`` are determined by round-off (the eigenvalues are
    separated by ~eps), so no choice of ``V`` makes the pair ``(W_j, V_j, q_j)`` satisfy
    both first-order relations ``Omega_R W = V q`` and ``Omega_L V = W q`` — and the
    interface algebra inverts ``big_W`` and ``big_V`` separately, so it amplifies whichever
    relation is broken by the cluster's 1/q conditioning. What does hold to eps is the
    *invariant subspace pair*: ``S_E`` (from the product's cluster, gap O(1) to the rest of
    the spectrum) and ``S_H`` (same for the swapped product ``Omega_R @ Omega_L``). Restrict
    the factors to those subspaces —

        ``R_r = Q_H^H Omega_R Q_E``,   ``L_r = Q_E^H Omega_L Q_H``      (k x k)

    — and take eigenpairs of the small first-order matrix ``[[0, L_r], [R_r, 0]]``: each
    eigenvector ``[gE; gH]`` with eigenvalue ``mu`` satisfies *both* relations to
    eps*||M_r|| by the standard eigenpair residual, so ``(Q_E gE, Q_H gH, mu)`` is a mode of
    a problem within eps of the requested one. Measured on the exact-graze repro stacks:
    the exact point lands inside the two-sided sqrt-extrapolated oracle bracket (~6e-9
    wide), where the shipped construction was off by 4.5e-3 .. 4.1e0.

    The eigenvectors come in ``+-mu`` pairs sharing one field pattern (one physical chain
    each), and near-degenerate chains make ``eig`` return an arbitrary — possibly nearly
    parallel — basis, so the k modes are chosen greedily to keep the E-parts *and* H-parts
    jointly independent (both matrices are inverted downstream); a ``+-`` twin scores ~0 on
    both sides once its partner is in and is never double-picked. Per column the member is
    branch-fixed to the root :func:`~wavebit._common.conventions.layer_root` takes for the rest
    of the layer (``k0_d`` and ``mirrored`` are passed through, so an ETM layer and a uniform
    double negative keep one rule), and kept at ``eig``'s joint normalization: a mode may then carry a tiny E-part (a pure-H graze)
    or a tiny H-part (the classic E-graze) but never both, which is the structure the
    interface inversions tolerate — normalizing either part to unit instead was measured to
    plant an inconsistent O(1/eps) column and lose three digits.

    Everything here runs in NumPy: the caller only enters when the eigenvalues were
    concretely readable, which is also the regime where no autograd graph is at stake.
    """
    import numpy as np

    w, W, V, q, Omega_L, Omega_R = (np.asarray(a) for a in (w, W, V, q, Omega_L, Omega_R))
    aw = np.abs(w)
    J = np.where(aw < _COUPLED_REL * aw.max())[0]
    wh, Vh = np.linalg.eig(Omega_R @ Omega_L)
    Jh = np.where(np.abs(wh) < _COUPLED_REL * np.abs(wh).max())[0]
    if len(Jh) != len(J):
        return None                              # structure not understood — detect, don't guess
    k = len(J)
    Q_E = np.linalg.qr(W[:, J])[0]
    Q_H = np.linalg.qr(Vh[:, Jh])[0]
    R_r = Q_H.conj().T @ (Omega_R @ Q_E)
    L_r = Q_E.conj().T @ (Omega_L @ Q_H)
    Z = np.zeros((k, k), dtype=W.dtype)
    mu_e, g = np.linalg.eig(np.block([[Z, L_r], [R_r, Z]]))
    g = g / np.linalg.norm(g, axis=0, keepdims=True)

    sel = []
    residE, residH = g[:k, :].copy(), g[k:, :].copy()
    for _ in range(k):
        score = (np.linalg.norm(residE, axis=0) ** 2
                 + np.linalg.norm(residH, axis=0) ** 2)
        score[sel] = -1.0
        pick = int(np.argmax(score))
        sel.append(pick)
        for resid in (residE, residH):
            col = resid[:, pick]
            n = np.linalg.norm(col)
            if n > 1e-150:
                d = col / n
                resid -= d[:, None] * (d.conj() @ resid)[None, :]

    if np.abs(mu_e[sel]).min() < _DEFECTIVE_REL * np.sqrt(aw.max()):
        raise FloatingPointError(
            'a 2D layer eigenmode pair is numerically defective at a grazing condition (an '
            'order sitting on the layer light line, so its q**2 is 0): the coupled mode '
            'rates split by less than 1e-10 of the layer modal scale, and the exponential '
            'mode basis this formulation uses does not exist there at working precision — '
            'the true limit basis is {1, z}, and any exponential spelling returns quietly '
            'wrong efficiencies (measured 1e-4 up to 1e0). Move off the condition: one ULP '
            'of wavelength, period or angle is enough, and solve(auto_detune=True), '
            'the default, does that retry for you.')

    Wc = np.empty((W.shape[0], k), dtype=W.dtype)
    Vc = np.empty_like(Wc)
    qc = np.empty(k, dtype=W.dtype)
    # Read as a number, the way the rest of this function reads its inputs: a thickness can be a
    # torch tensor that requires grad while the operator does not, and then this is entered.
    k0_d = None if k0_d is None else checked_float(k0_d)
    for i, col in enumerate(sel):
        mu_i = mu_e[col]
        q_i = layer_root(np, np.asarray([mu_i ** 2]), k0_d, mirrored)[0]
        gE, gH = g[:k, col], g[k:, col]
        if abs(q_i - mu_i) > abs(q_i + mu_i):
            gH = -gH                             # took the -q member; its H part flips
        Wc[:, i] = Q_E @ gE
        Vc[:, i] = Q_H @ gH
        qc[i] = q_i

    W, V, q = W.copy(), V.copy(), q.copy()
    W[:, J], V[:, J], q[J] = Wc, Vc, qc
    return W, V, q


#: The defective-pair gate (:func:`_defective_pairs`): two modes of a 2D layer that are each
#: other's nearest eigenvalue, and whose eigenvectors meet at a sine below this, are rebuilt as
#: an orthonormal Schur pair with an upper-triangular ``q`` (:func:`_rebuild_defective_pairs`).
#: Measured, not derived: the product construction's error follows ``K u / s**2`` (``u`` the
#: unit round-off, ``s`` the pair's sine, ``K <= 1e-2`` on SMM over the patterned scenes), and just
#: outside the edge (``s`` 1e-2 .. 1e-1) it was at its floor, 4e-15 .. 1.7e-13. The rebuild is an
#: exact change of basis for any pair, so a larger value costs time, never accuracy.
_EP_SINE = 1e-2

#: A pair root at or below this modulus is left to the construction: ``q -> 0`` is the grazing
#: cluster's (:func:`_rebuild_grazing_cluster`), where an up- and a down-going mode merge.
_EP_ROOT_FLOOR = 1e-9

#: A pair with an unpaired eigenvalue within this of either member (relative to
#: max(1, |w|)) is not rebuilt: the Newton step divides that row by its distance to the pair.
#: Measured on a Jordan pair beside a plain root at delta: the rebuilt pair's residual 2.0e-12
#: at 1e-8, 3.5e-13 at 2e-8, <= 8.1e-14 from 5e-8 on (three random bases). Another gated pair
#: is not counted: its rows take the 2x2 Sylvester closed form.
_EP_SEPARATION = 1e-7

#: The radius of the clusters _unresolved_defective_cluster reads.
_EP_CLUSTER = 1e-4

#: A cluster whose restriction T = Q^H A Q departs from normality by more than
#: this, ||T T^H - T^H T||_F / max(1, |lambda|)**2, is defective. Measured: degenerate clusters in
#: a nearly parallel eig basis (uniform layers, C4 rasters) <= 1.9e-19; exceptional points
#: (patterned rasters at offsets 0..1e-3, y-Littrow coincidences, 2D and first order) >= 1.4e-7.
_DEFECT_NU = 1e-10


def _unresolved_defective_cluster(xp, w, W, operator, sine, group=None):
    """Eagerly, whether ``eig`` left a defective cluster of three or more modes in a nearly
    dependent basis; ``None`` under an abstract trace.

    A cluster is a connected set of eigenvalues each within :data:`_EP_CLUSTER` of the next.
    It is unresolved when its unit columns' smallest singular value is below ``sine`` and its
    restriction ``T = Q^H A Q`` (``Q`` an orthonormal basis of the columns) is not normal:
    :data:`_DEFECT_NU`. A degenerate (diagonalizable) cluster passes whatever basis
    ``eig`` chose for it. ``operator()`` builds ``A`` only when needed. ``group`` (optional, one
    label per mode) keeps modes of different groups out of one cluster.
    """
    scale = xp.maximum(xp.ones_like(xp.abs(w)), xp.abs(w))
    near = xp.abs(w[:, None] - w[None, :]) <= _EP_CLUSTER * xp.maximum(scale[:, None],
                                                                          scale[None, :])
    if group is not None:
        # Modes of different groups never share a cluster: the first-order split's two
        # directions, so that an up- and a down-going mode merging at a graze (q -> 0, the
        # {1, z} regime, which this screen is not about) is not read as an exceptional point.
        near = near & (group[:, None] == group[None, :])
    count = xp.sum(xp.astype(near, xp.int32), axis=1)
    crowded = _concrete(xp.any(count >= 3))
    if not crowded:
        return None if crowded is None else False
    dev = getattr(W, 'device', None)
    Wn = W / xp.linalg.vector_norm(W, axis=0)[None, :]
    pending, A = count >= 3, None
    idx = xp.arange(w.shape[0], device=dev)
    while _concrete(xp.any(pending)):
        # Keep graph membership and indices as array data. Only Boolean reductions
        # cross to Python, through _concrete, even in this eager-only screen.
        comp = idx == xp.argmax(xp.astype(pending, xp.int32))
        frontier = comp
        while _concrete(xp.any(frontier)):
            neighbors = xp.any(xp.take(near, xp.nonzero(frontier)[0], axis=0), axis=0)
            frontier = neighbors & ~comp
            comp = comp | frontier
        pending = pending & ~comp
        members = xp.nonzero(comp)[0]
        cols = xp.take(Wn, members, axis=1)
        if _concrete(xp.linalg.svdvals(cols)[-1] >= sine):
            continue
        if A is None:
            A = operator()
        Q, _ = xp.linalg.qr(cols)
        T = xp.matrix_transpose(xp.conj(Q)) @ (A @ Q)
        k = members.shape[0]
        lam = xp.sum(xp.linalg.diagonal(T)) / k
        TH = xp.matrix_transpose(xp.conj(T))
        nu = (xp.linalg.matrix_norm(T @ TH - TH @ T)
              / xp.maximum(xp.ones_like(xp.abs(lam)), xp.abs(lam)) ** 2)
        # The floor is |E|**2 for a rounding E of T, so it scales with eps**2: in single
        # precision a normal cluster's measure reached the float64 threshold (4 of 60 uniform
        # first-order layers warned), so below double precision the screen stays silent.
        eps = float(xp.finfo(xp.real(T).dtype).eps) / 2.220446049250313e-16
        if _concrete(nu > _DEFECT_NU * eps * eps):
            return True
    return False


_UNRESOLVED_2D = (
    'a 2D layer holds three or more modes of one defective eigenvalue cluster (exceptional '
    'points that coincide, such as two order rows of a y-invariant layer at the same x-cutoff), '
    'which no pair rebuild spans: the eigenvector basis was kept, and the answer can be '
    'inaccurate. Move the angle or wavelength by more than a relative 1e-4 to check it.')


def _defective_pairs(xp, w, W, q, cap):
    """The near-defective mode pairs of a 2D layer, or ``None`` when there is none.

    A pair is two modes ``(a, b)``, ``a < b``, that are each other's nearest eigenvalue of the
    product operator and whose eigenvectors meet at a sine ``s`` below :data:`_EP_SINE`, with
    roots above :data:`_EP_ROOT_FLOOR` (their labels may disagree; the rebuild puts both on one
    side). That is
    what ``eig`` hands back at and near an exceptional point -- a 2x2 Jordan block of the layer
    operator, which on a patterned layer is reached wherever an order meets the conical x-cutoff
    (``docs/theory/rcwa/rayleigh-points.md`` section 7): the two computed eigenvectors are each accurate to
    ``u`` but meet at ``s ~ 1e-8 .. 1e-6``, the mode matrix's condition number is ``~1/s``, and the
    interface inverses amplify rounding by about ``1/s**2``. It is also what ``eig`` may hand
    back for an exactly degenerate (diagonalizable) pair, whose basis it chooses freely.

    Returns ``(a, b, mask, v, x0, overflow)``: index vectors of a static length, a mask of the
    pairs that are real, the unit eigenvector ``v`` of each ``a`` and the unit direction ``x0`` of
    its partner orthogonal to it -- ``(I - v v^H) W_b / s``, accurate to ``u / s`` -- and, under an
    abstract trace, whether more pairs qualified than ``cap`` (``None`` eagerly).

    Read once per layer through :func:`~wavebit._common.backend._concrete`. Eagerly the pairs are
    exactly the qualifying ones; under an abstract trace (``jax.jit``, ``jax.vmap``,
    ``torch.func.vmap``) there is nothing to read, so the ``cap`` smallest sines come back with
    their mask as data -- the 1D cutoff gate's arrangement (:func:`~wavebit.rcwa._engine.solve_1d.
    _cutoff_cluster`).
    """
    n2 = w.shape[0]
    dev = getattr(W, 'device', None)
    idx = xp.arange(n2, device=dev)
    d = xp.abs(w[:, None] - w[None, :])
    d = xp.where(idx[:, None] == idx[None, :], xp.full_like(d, float('inf')), d)
    nb = xp.argmin(d, axis=1)
    Wn = W / xp.linalg.vector_norm(W, axis=0)[None, :]
    Wp = xp.take(Wn, nb, axis=1)
    X = Wp - Wn * xp.sum(xp.conj(Wn) * Wp, axis=0)[None, :]
    s = xp.linalg.vector_norm(X, axis=0)
    qp = xp.take(q, nb)
    # no unpaired eigenvalue within _EP_SEPARATION of either member.
    member = ((xp.take(nb, nb) == idx) & (s < _EP_SINE)
              & (xp.minimum(xp.abs(q), xp.abs(qp)) > _EP_ROOT_FLOOR))
    scale = xp.maximum(xp.ones_like(xp.abs(w)), xp.abs(w))
    partner = idx[None, :] == nb[:, None]
    unpaired = (d <= _EP_SEPARATION * scale[:, None]) & ~partner & ~member[None, :]
    alone = ~xp.any(unpaired, axis=1)
    eligible = member & (idx < nb) & alone & xp.take(alone, nb)
    decided = _concrete(xp.any(eligible))
    if decided is False:
        return None
    overflow = None
    if decided:
        a = xp.nonzero(eligible)[0]
        mask = xp.ones(a.shape, dtype=xp.bool, device=dev)
    else:
        cap = max(1, min(cap, n2 // 2))
        order = xp.argsort(xp.where(eligible, s, xp.full_like(s, float('inf'))))
        a = order[:cap]
        mask = xp.take(eligible, a)
        overflow = xp.sum(xp.astype(eligible, xp.int32)) > cap
    b = xp.take(nb, a)
    sa = xp.take(s, a)
    x0 = xp.take(X, a, axis=1) / xp.where(sa > 0, sa, xp.ones_like(sa))[None, :]
    v = xp.take(Wn, a, axis=1)
    return a, b, mask, v, x0, overflow


def _one_hot(xp, n2, index, mask, dtype, dev):
    """``(n2, len(index))``: column ``p`` is the unit vector ``e_index[p]``, zero where unmasked."""
    idx = xp.arange(n2, device=dev)
    return xp.astype((idx[:, None] == index[None, :]) & mask[None, :], dtype)


def _rebuild_defective_pairs(xp, A, Omega_R, W, V, q, w, pairs):
    """Each near-defective pair of a 2D layer as an orthonormal Schur pair; values only.

    ``A`` is the layer's product operator ``Omega2 = Omega_L Omega_R``, ``(W, V, q)`` the modes
    the product construction built, ``w`` their eigenvalues and ``pairs``
    :func:`_defective_pairs`'s answer. Returns ``(W, V, q, a, b, mask)`` with every pair's two
    columns replaced and ``q`` the ``2n x 2n`` matrix ``diag(q) + tau``: ``tau[a, b] = t`` on each
    pair, zero elsewhere, so its coupled rows ``{a}`` and columns ``{b}`` are disjoint.

    **The subspace.** A pair's invariant subspace is well conditioned (its gap to the rest of the
    spectrum is O(1)) even where its eigenvectors are not, but the span ``eig`` hands back is not
    backward stable: ``v`` is an eigenvector to ``u``, its partner's direction ``x0`` only to
    ``u / s``. One Newton step for the invariant subspace (G. W. Stewart, SIAM J. Numer. Anal. 10,
    752, 1973; J. W. Demmel, Linear Algebra Appl. 79, 163, 1986) squares that error. It is taken
    in the basis ``W0`` -- the modes with every pair orthonormalised, well conditioned -- where
    the rest of the operator is already (block) diagonal: with ``M = Q^H A Q`` upper triangular
    (``M_10 = x0^H A v`` is ``O(u)``, ``v`` being an eigenvector, and is dropped), the residual
    ``R = A Q - Q M`` expands as ``R = W0 C`` by one solve for every pair at once, and the
    correction ``Q -> Q + W0 Y`` solves ``Lambda Y - Y M = -C`` row by row: a plain row
    ``y (lambda_i - M) = -c_i``, and another pair's two rows the 2x2 upper-triangular Sylvester
    equation with that pair's ``M``, in closed form. The pair's own rows are left at zero (a
    change of basis inside the subspace). Measured, the starting error ``u / s`` (1e-9 at the
    test points) lands at the solve's rounding: amplitudes within 5e-14 of the 1D conical rebuild,
    where the product construction was 1e-8 .. 4e-2 off.

    **The modes.** Re-orthonormalised to ``(v, x)``, the pair's restriction ``T = Q^H A Q`` is
    upper triangular to ``u`` (``T_10`` dropped: a backward error of ``u``), and
    ``q = [[alpha, t], [0, beta]]`` with ``alpha**2 = T_00``, ``beta**2 = T_11`` and
    ``t = T_01 / (alpha + beta)`` is the exact square root of ``T`` on one side: both roots take
    the side of the more decaying member's label, which is each member's own wherever the layer's
    rule agrees with itself (``alpha + beta`` is then at least ``2 * _EP_ROOT_FLOOR`` from zero).
    Then ``W = (v, x)`` and
    ``V = Omega_R W q^-1`` -- column ``a`` ``Omega_R v / alpha``, column ``b``
    ``(Omega_R x - V_a t) / beta`` -- satisfy both first-order relations ``Omega_R W = V q`` and
    ``Omega_L V = W q`` to ``u``: the second because ``Omega_L Omega_R = A`` and ``q**2 = T``.
    No division by a gap anywhere; ``alpha`` and ``beta`` may be equal.

    Every divisor is guarded where its lane is discarded, so an unmasked lane under a trace forms
    no ``inf``; such a lane keeps the construction's columns and a zero coupling.
    """
    a, b, mask, v, x0, overflow = pairs
    n2 = W.shape[0]
    C = W.dtype
    dev = getattr(W, 'device', None)
    idx = xp.arange(n2, device=dev)
    Oa = _one_hot(xp, n2, a, mask, C, dev)
    Ob = _one_hot(xp, n2, b, mask, C, dev)
    is_a = xp.any(Oa != 0, axis=1)
    is_b = xp.any(Ob != 0, axis=1)
    pa = xp.argmax(xp.astype(Oa != 0, xp.int32), axis=1)
    pb = xp.argmax(xp.astype(Ob != 0, xp.int32), axis=1)
    one = xp.ones_like

    def place(col_a, col_b, base):
        return xp.where(is_a[None, :], xp.take(col_a, pa, axis=1),
                        xp.where(is_b[None, :], xp.take(col_b, pb, axis=1), base))

    W0 = place(v, x0, W)
    Av, Ax = A @ v, A @ x0
    m00 = xp.sum(xp.conj(v) * Av, axis=0)
    m01 = xp.sum(xp.conj(v) * Ax, axis=0)
    m11 = xp.sum(xp.conj(x0) * Ax, axis=0)
    R = xp.concat([Av - v * m00[None, :], Ax - v * m01[None, :] - x0 * m11[None, :]], axis=1)
    eye = xp.astype(xp.eye(n2, device=dev), C)
    G = linear_solve(xp, xp.where(xp.any(mask), W0, eye), R)
    npair = a.shape[0]
    Ca, Cb = G[:, :npair], G[:, npair:]
    # Each row's own diagonal in the W0 basis: a pair's M_00 / M_11, else the eigenvalue.
    lam = xp.where(is_a, xp.take(m00, pa), xp.where(is_b, xp.take(m11, pb), w))
    own = ((idx[:, None] == a[None, :]) | (idx[:, None] == b[None, :]))
    den0 = xp.where(own, one(Ca), lam[:, None] - m00[None, :])
    den1 = xp.where(own, one(Ca), lam[:, None] - m11[None, :])
    Y0 = -Ca / den0
    Y1 = (-Cb + Y0 * m01[None, :]) / den1
    # Another pair's `a` row: its own M_01 couples it to its `b` row (upper-triangular Sylvester).
    m01_row = xp.where(is_a, xp.take(m01, pa), xp.zeros_like(w))
    partner = xp.where(is_a, xp.take(b, pa), idx)
    Y0 = xp.where(is_a[:, None], (-Ca - m01_row[:, None] * xp.take(Y0, partner, axis=0)) / den0, Y0)
    Y1 = xp.where(is_a[:, None], (-Cb - m01_row[:, None] * xp.take(Y1, partner, axis=0)
                                  + Y0 * m01[None, :]) / den1, Y1)
    zero = xp.zeros_like(Y0)
    Y0, Y1 = xp.where(own, zero, Y0), xp.where(own, zero, Y1)
    v1 = v + W0 @ Y0
    x1 = x0 + W0 @ Y1
    v1 = v1 / xp.linalg.vector_norm(v1, axis=0)[None, :]
    x1 = x1 - v1 * xp.sum(xp.conj(v1) * x1, axis=0)[None, :]
    x1 = x1 / xp.linalg.vector_norm(x1, axis=0)[None, :]
    Av1, Ax1 = A @ v1, A @ x1
    T00 = xp.sum(xp.conj(v1) * Av1, axis=0)
    T01 = xp.sum(xp.conj(v1) * Ax1, axis=0)
    T11 = xp.sum(xp.conj(x1) * Ax1, axis=0)
    ra, rb = xp.sqrt(T00), xp.sqrt(T11)
    qa, qb = xp.take(q, a), xp.take(q, b)
    # One side for the pair: the root nearer the label of its more decaying member. Where the
    # labels agree that is each member's own; where they disagree -- a complex pair on Maystre's
    # cut, a tie broken by the flux or by `Im(q) <= 0` -- the other member turns, so the `+` set
    # takes the whole Jordan block and stays complementary to the `-` set.
    ref = xp.where(xp.real(qa) >= xp.real(qb), qa, qb)
    alpha = xp.where(xp.real(xp.conj(ref) * ra) >= 0, ra, -ra)
    beta = xp.where(xp.real(xp.conj(ref) * rb) >= 0, rb, -rb)
    alpha = xp.where(mask, alpha, one(alpha))
    beta = xp.where(mask, beta, one(beta))
    t = xp.where(mask, T01 / (alpha + beta), xp.zeros_like(T01))
    Va = (Omega_R @ v1) / alpha[None, :]
    Vb = (Omega_R @ x1 - Va * t[None, :]) / beta[None, :]
    W = place(v1, x1, W)
    V = place(Va, Vb, V)
    q = xp.where(is_a, xp.take(alpha, pa), xp.where(is_b, xp.take(beta, pb), q))
    Q = _diag(xp, q) + Oa @ (t[:, None] * xp.matrix_transpose(Ob))
    if overflow is not None:
        # More qualifying pairs than the trace's static count: loud, never a silent remainder.
        Q = Q * xp.where(overflow, float('nan'), 1.0)
    return W, V, Q, a, b, mask


def _pair_coupling(xp, Q, a, b, mask):
    """``t_p = Q[a_p, b_p]`` (zero where unmasked) and the two one-hot maps, by gathers."""
    n2 = Q.shape[0]
    dev = getattr(Q, 'device', None)
    Oa = _one_hot(xp, n2, a, mask, Q.dtype, dev)
    Ob = _one_hot(xp, n2, b, mask, Q.dtype, dev)
    t = xp.sum(xp.take(Q, a, axis=0) * xp.matrix_transpose(Ob), axis=1)
    return t, Oa, Ob


def _pair_sylvester(xp, T, S, a, b, mask):
    """``X`` with ``T X + X T = S`` for ``T = diag(t) + coupling`` on the pairs ``(a_p, b_p)``.

    :func:`~wavebit.rcwa._engine.solve_1d._coupled_sylvester`'s four divided-difference passes, with the
    coupling's products taken by gathers (``O(n**2)`` per pair) rather than dense products: the
    coupling has one entry per pair. With no coupling every pass is ``S / (t_i + t_j)``, the
    arithmetic :func:`~wavebit.rcwa._engine.modal_gradient.generator` forms.
    """
    tv = xp.linalg.diagonal(T)
    total = tv[:, None] + tv[None, :]
    t, Oa, Ob = _pair_coupling(xp, T, a, b, mask)
    rows = xp.any(Oa != 0, axis=1)[:, None]
    cols = xp.any(Ob != 0, axis=1)[None, :]

    def left(X):                                 # N @ X: row a_p is t_p X[b_p, :]
        return Oa @ (t[:, None] * xp.take(X, b, axis=0))

    def right(X):                                # X @ N: column b_p is t_p X[:, a_p]
        return (xp.take(X, a, axis=1) * t[None, :]) @ xp.matrix_transpose(Ob)

    X1 = S / total
    X2 = (S - left(X1)) / total
    X3 = (S - right(X1)) / total
    Xa = xp.where(rows & ~cols, X2, xp.where(~rows & cols, X3, X1))
    X4 = (S - left(Xa) - right(Xa)) / total
    return xp.where(rows & cols, X4, Xa)


def _pair_generator(xp, operator, W, Q, a, b, mask):
    """:func:`~wavebit.rcwa._engine.modal_gradient.generator` for a layer with rebuilt pairs.

    ``Q`` is the rebuilt ``q`` (diagonal plus the pairs' coupling), the square root of
    ``W^-1 Omega2 W`` in the fixed basis ``W``; its first variation ``dQ`` solves
    ``Q dQ + dQ Q = W^-1 dOmega2 W`` (:func:`_pair_sylvester`). The value is ``Q``.
    """
    from wavebit.rcwa._engine.modal_gradient import _require_separated
    from wavebit.rcwa._engine.modal_gradient import _stop
    from wavebit.rcwa._engine.modal_gradient import _variation
    if not carries_autodiff(operator):
        return Q
    Q0 = _stop(Q)
    residual = _variation(operator, _stop(W))
    tv = xp.linalg.diagonal(Q0)
    total = tv[:, None] + tv[None, :]
    valid = xp.all(total != 0)
    _require_separated(valid, 'zero or opposing square-root modes; use auto_detune',
                       error=FloatingPointError)
    residual = residual * xp.where(valid, xp.ones_like(residual),
                                   xp.full_like(residual, complex('nan')))
    return Q0 + _pair_sylvester(xp, Q0, residual, a, b, mask)


def _pair_inverse(xp, Q, a, b, mask):
    """``Q^-1`` for a rebuilt ``q``, with its exact first variation ``-Q^-1 dQ Q^-1``.

    The value is :func:`~wavebit.rcwa._engine.modal_gradient.inverse`'s closed form, exact for a coupling
    with disjoint rows and columns (``D^-1 - D^-1 N D^-1``). Its own tangent drops every term in
    which the coupling ``N`` meets an off-diagonal ``dQ``; they are added here on a zero value,
    ``M Eoff D^-1 + D^-1 Eoff M - M E M`` with ``M = D^-1 N D^-1``, ``E = Q - stop(Q)`` and
    ``Eoff`` its off-diagonal part -- by gathers, ``M`` having one entry per pair. With no coupling
    they are zero and the arithmetic is :func:`~wavebit.rcwa._engine.modal_gradient.inverse`'s.
    """
    from wavebit.rcwa._engine.modal_gradient import _stop
    from wavebit.rcwa._engine.modal_gradient import inverse
    base = inverse(xp, Q)
    if not carries_autodiff(Q):
        return base
    Q0 = _stop(Q)
    tv = xp.linalg.diagonal(Q0)
    t, Oa, Ob = _pair_coupling(xp, Q0, a, b, mask)
    m = t / (xp.take(tv, a) * xp.take(tv, b))          # M[a_p, b_p]
    E = Q - Q0
    n2 = Q.shape[0]
    eye = xp.astype(xp.eye(n2, device=getattr(Q, 'device', None)), Q.dtype)
    Eoff = E - eye * E
    dinv = 1 / tv
    # M X: row a_p is m_p X[b_p, :];  X M: column b_p is m_p X[:, a_p].
    MEoffD = Oa @ (m[:, None] * xp.take(Eoff, b, axis=0)) * dinv[None, :]
    DEoffM = (dinv[:, None] * xp.take(Eoff, a, axis=1) * m[None, :]) @ xp.matrix_transpose(Ob)
    EM = xp.take(E, a, axis=1) * m[None, :]             # columns b_p of E M
    MEM = Oa @ (m[:, None] * xp.take(EM, b, axis=0)) @ xp.matrix_transpose(Ob)
    return base + (MEoffD + DEoffM - MEM)


def layer_modes_2d(xp, Kx, Ky, I, epx_conv, epy_conv, epz_conv_i, epxy_conv, epyx_conv, C,
                   mu=None, k0_d=None, dng=None, pair_cap=None, with_coupling=False, env=None):
    """Raw (unrotated) eigenmodes ``(W, V, q)`` of one 2D layer.

    ``W`` is the ``[Ex; Ey]`` mode matrix and ``q`` the modal decay constants; without ``k0_d``
    the root keeps ``Re(q) >= 0`` so ``exp(-k0*q*d)`` never grows. Shared by the ETM
    (:func:`solve_2d`) and SMM (:mod:`wavebit.rcwa._engine.scattering`) connecting algorithms, which
    differ only in which of ``±q`` they label "down": ETM passes the layer's optical thickness
    ``k0_d`` (:func:`~wavebit._common.conventions.layer_root`, as in
    :func:`~wavebit.rcwa._engine.solve_1d.layer_modes_1d`). ``dng`` is the engine's word on a
    double-negative region, read as :func:`~wavebit.rcwa._engine.solve_1d.layer_modes_1d` reads it.

    The two blocks are the factors of ``d2E/dz2 = P Q E``: ``Omega2_LR`` is the product written
    out, ``Omega_R`` the right factor kept so ``V`` can recover ``H``. **They must carry the
    off-diagonal pair in opposite orders** — ``Omega2_LR`` pairs ``epyx`` with the first column
    and ``epxy`` with the second, ``Omega_R`` the reverse. That is not a symmetry to tidy up: it
    is what makes ``Omega_L = Omega2_LR @ inv(Omega_R)`` come out as geometry alone,
    ``[[-Kx Ky epz_i, Kx^2 epz_i - 1], [-Ky^2 epz_i + 1, Kx Ky epz_i]]``, for an arbitrary
    tensor at any incidence. Align them and an asymmetric tensor takes its modes from one medium
    and its H-field from that medium's transpose, silently: every tensor the library builds is
    symmetric, so nothing in a normal run can tell the difference. See
    ``docs/theory/rcwa/off-diagonal-slot-convention.md`` §4, and ``tests/rcwa/test_anisotropic_berreman.py`` /
    ``tests/references/test_reticolo_anisotropic_golden.py`` for what pins it.

    ``mu`` is an optional ``(mux_conv, muy_conv, muz_conv_i, muxy_conv, muyx_conv)``
    permeability tuple in **the same five slots and the same convention as the
    permittivity** — ``muz_conv_i`` inverted, the off-diagonals negated. Left ``None`` the
    layer is non-magnetic and the same non-magnetic Maxwell equations apply. ``Kx`` and
    ``Ky`` are diagonal wavevector matrices; their action on the longitudinal inverse is
    evaluated by row and column scaling, including during a grazing-cluster rebuild.

    ``pair_cap`` bounds the number of exceptional-point pairs held under an abstract
    trace; callers with an order grid pass its larger axis count. An overflow poisons
    the generator with NaN. Eagerly every eligible pair is rebuilt. ``with_coupling``
    requests a fourth return value describing that layout for propagation and field
    consumers. Without it the three-value interface is retained, but ``q`` at an
    exceptional point is a matrix generator, not a vector of independent roots.

    Where it enters is a duality, not a patch. Under ``(E, H, eps, mu) -> (H, -E, mu, eps)``
    the two curl equations map onto each other, so ``Omega_L`` — which eliminates ``E_z``
    through ``eps_zz`` — carries the *transverse* ``mu``, and ``Omega_R`` — which eliminates
    ``H_z`` through ``mu_zz`` — carries the transverse ``eps``. Concretely the four blocks
    below each acquire one slot, uniformly, and ``Omega_R``'s three geometry products gain
    ``muz_conv_i`` between their ``K`` factors::

        B  = Kx epz_i Kx - muy          Cc = Kx epz_i Ky - muyx
        D  = Ky epz_i Ky - mux          E  = Ky epz_i Kx - muxy

    (Abbreviated on both sides — ``epz_i`` is ``epz_conv_i`` and ``muy`` is ``muy_conv``.
    Every one of these eight operands is a convolution matrix; the permeability is built by
    the same builder as the permittivity, which is what the shared ``_conv`` says.)

    Note the **cross** pairing: ``B`` takes ``mu_yy`` and ``D`` takes ``mu_xx``, not the other
    way round. ``docs/guide/rcwa/anisotropic-materials.md`` §5 pointed at the two ``- I`` terms and stopped
    there, which would have got the isotropic case right and silently transposed a uniaxial
    ``mu``. The placement was settled by requiring the derived ``Omega_L @ Omega_R`` to
    reproduce the shipped ``Omega2_LR`` term for term at ``mu = 1``.
    """
    mux_conv, muy_conv, muz_conv_i, muxy_conv, muyx_conv = (
        (I, I, None, None, None) if mu is None else mu)
    mirrored, patterned = layer_label_rule(
        xp, dng, (epx_conv, epy_conv, epz_conv_i),
        None if mu is None else (mux_conv, muy_conv, muz_conv_i),
        (epxy_conv, epyx_conv, muxy_conv, muyx_conv))
    kx, ky = xp.linalg.diagonal(Kx), xp.linalg.diagonal(Ky)

    def left_blocks():
        # diag(k) A diag(l) scales rows and columns. Keep that multiplication
        # order, but do not spend eight dense matrix products on diagonal factors.
        B = (kx[:, None] * epz_conv_i) * kx[None, :] - muy_conv
        D = (ky[:, None] * epz_conv_i) * ky[None, :] - mux_conv
        Cc = (kx[:, None] * epz_conv_i) * ky[None, :]
        E = (ky[:, None] * epz_conv_i) * kx[None, :]
        if muyx_conv is not None:
            Cc = Cc - muyx_conv
        if muxy_conv is not None:
            E = E - muxy_conv
        return B, D, Cc, E

    # `Omega_R`'s three geometry products carry `muz_conv_i` between the K factors, exactly as
    # `epz_conv_i` sits between them in `Omega_L`. Skipped entirely when non-magnetic rather
    # than multiplying an identity through, so the arithmetic stays the operations it was.
    def right_factor():
        if muz_conv_i is None:
            KxKy, KxKx, KyKy, KyKx = Kx @ Ky, Kx @ Kx, Ky @ Ky, Ky @ Kx
        else:
            KxKy, KxKx = Kx @ muz_conv_i @ Ky, Kx @ muz_conv_i @ Kx
            KyKy, KyKx = Ky @ muz_conv_i @ Ky, Ky @ muz_conv_i @ Kx
        return _block(xp, [
            [-KxKy + epyx_conv, KxKx - epy_conv],
            [epx_conv - KyKy, KyKx - epxy_conv],
        ])

    B, D, Cc, E = left_blocks()
    Omega_R = None
    if mu is None:
        # The hand-expanded product. Its four bare ``K @ K`` terms are not geometry that
        # survived the multiplication untouched — they are what two *other* terms collapse to
        # when the K factors commute past a `muz_conv_i` of one. Written out, the (0,0) block is
        #   Cc Kx muz_i Ky - B Ky muz_i Ky + B epx - Cc epyx,          (abbreviated, as above)
        # where the first pair leaves `muy Ky muz_i Ky + muyx Kx muz_i Ky` behind, which is
        # `Ky @ Ky` only when `muy = 1`, `muyx = 0` and `muz_i = 1`. So this form cannot be
        # reused with a permeability, and reusing it is not a small error: it costs energy
        # conservation outright on a gyromagnetic layer (R + T came out 1.035).
        Omega2_LR = _block(xp, [
            [Ky @ Ky + B @ epx_conv - Cc @ epyx_conv, Cc @ epy_conv - Kx @ Ky - B @ epxy_conv],
            [E @ epx_conv - Ky @ Kx - D @ epyx_conv, Kx @ Kx + D @ epy_conv - E @ epxy_conv],
        ])
    else:
        Omega_R = right_factor()
        Omega2_LR = _block(xp, [[-Cc, B], [-D, E]]) @ Omega_R

    # The eigensolve needs only the product. Reconstruct the left factor
    # later if the existing grazing guard requires it; ordinary modes do
    # not retain its four dense blocks or an unused right factor here.
    del B, D, Cc, E
    from wavebit.rcwa._engine.modal_gradient import _stop
    differentiated = carries_autodiff(Omega2_LR)
    w, W = eig(_stop(Omega2_LR))
    if not differentiated:
        # Only autodiff needs the original operator after the eigensolve.
        # Preserve the eager forward path's early release of this dense matrix.
        del Omega2_LR
    if Omega_R is None:
        Omega_R = right_factor()
    if _env_decides(env, mirrored, patterned):
        W_s, R_s = _stop(W), _stop(Omega_R)
        RW_s = R_s @ W_s

        def boundary(q_):
            with division_at_a_cutoff():
                return env.frame(W_s, RW_s / q_)
        q = env.choose(xp, _stop(w), C, boundary)
    else:
        q = layer_root(xp, xp.astype(w, C), k0_d, mirrored)
    V = Omega_R @ W
    with division_at_a_cutoff():                 # q == 0 on a light line; rebuilt or refused below
        V = V / q
    if patterned is not None:
        # Before the generator, which must see the labels the layer keeps.
        V, q = _power_down_2d(xp, W, V, q, k0_d, patterned)

    # At an exact Rayleigh point (an order on a layer's light line, q -> 0) this V is wrong
    # with no arithmetic to blame: the grazing cluster's eigvecs of the *product* Omega2 are
    # eps-unstable with respect to the factor split, so even the bit-exact `Omega_R @ W`
    # (measured: it agrees with 200-bit arithmetic to 1e-22 on the repro stack) yields a
    # column that satisfies `Omega_R W = V q` while violating `Omega_L V = W q` at order 1 —
    # the layer is then the mode set of *no* nearby problem, and the answers were quietly
    # wrong at 1e-5 (band edge) up to 1e0 (exact point). The rebuild below re-derives the
    # cluster from the coupled first-order problem, which restores both relations to eps;
    # where the cluster is numerically defective (the `{1, z}` regime) it refuses loudly
    # instead. Traced or otherwise unreadable values keep the old detect-and-warn contract.
    aw = xp.abs(_readable(w, xp))
    grazing = _concrete(xp.min(aw) < _COUPLED_REL * xp.max(aw))
    if grazing and not differentiated:
        B, D, Cc, E = left_blocks()
        Omega_L = _block(xp, [[-Cc, B], [-D, E]])
        del B, D, Cc, E
        try:
            rebuilt = _rebuild_grazing_cluster(xp, w, W, V, q, Omega_L, Omega_R, k0_d, mirrored)
        except FloatingPointError:
            raise                                # the defective-cluster refusal, on purpose
        except Exception:
            # Concrete but not materializable to NumPy (a torch tensor carrying grad, an
            # un-jitted jax.grad tracer): keep the detect-and-warn contract; the default
            # auto_detune ladder then re-solves off the point with the graph intact.
            rebuilt = None
        if rebuilt is not None:
            if env is not None:
                env.Wi = None
            W, V, q = (xp.astype(xp.asarray(a), C) for a in rebuilt)
            if patterned is not None:
                # The rebuilt cluster took `layer_root`'s labels; the per-mode rule reads them.
                V, q = _power_down_2d(xp, W, V, q, k0_d, patterned)
            return (W, V, q, False) if with_coupling else (W, V, q)
    check_mode_cutoff(xp, w)
    # A near-defective pair (an exceptional point of the layer operator, or a degenerate pair
    # whose basis `eig` chose nearly parallel): rebuilt as an orthonormal Schur pair with an
    # upper-triangular `q` (`_rebuild_defective_pairs`). Gated once per layer; under an abstract
    # trace every layer takes the coupled layout with its pairs' mask as data.
    if pair_cap is None:
        import math
        pair_cap = math.isqrt(W.shape[0] // 2) + 1
    pairs = _defective_pairs(xp, w, W, q, pair_cap)

    def _operator():
        if differentiated:
            return _stop(Omega2_LR)
        B_, D_, Cc_, E_ = left_blocks()
        return _block(xp, [[-Cc_, B_], [-D_, E_]]) @ Omega_R

    def _screen(W_final):
        # coincident exceptional points the pair gate cannot see -- eagerly loud.
        if _unresolved_defective_cluster(xp, _stop(w), _stop(W_final), _operator, _EP_SINE):
            import warnings
            from wavebit._common.backend import ExceptionalPointWarning
            warnings.warn(_UNRESOLVED_2D, ExceptionalPointWarning, stacklevel=4)
    if pairs is None:
        _screen(W)
    if pairs is not None:
        if differentiated:
            A = _stop(Omega2_LR)
        else:
            # Released above to keep the eager peak; formed again only for a gated layer.
            B, D, Cc, E = left_blocks()
            A = _block(xp, [[-Cc, B], [-D, E]]) @ Omega_R
            del B, D, Cc, E
        if env is not None:
            env.Wi = None
        W, V, Q, a, b, mask = _rebuild_defective_pairs(xp, A, _stop(Omega_R), W, V, q, w, pairs)
        del A
        _screen(W)
        if differentiated:
            Q = _pair_generator(xp, Omega2_LR, W, Q, a, b, mask)
            V = Omega_R @ W @ _pair_inverse(xp, Q, a, b, mask)
        return (W, V, Q, True) if with_coupling else (W, V, Q)
    if differentiated:
        from wavebit.rcwa._engine.modal_gradient import generator
        from wavebit.rcwa._engine.modal_gradient import inverse
        q = generator(xp, Omega2_LR, w, W, q, square_root=True)
        V = Omega_R @ W @ inverse(xp, q)
    return (W, V, q, False) if with_coupling else (W, V, q)


def sp_modes_2d(xp, W, V, F_c, F_s, n_orders):
    """Rotate 2D layer modes into the s/p boundary basis: ``(big_W, big_V)``.

    ``big_W`` maps mode amplitudes to the boundary variables ``(E_s, F_p)`` and ``big_V`` to
    ``(H_s, H_p)`` — the pair the substrate seed ``big_F`` / ``big_G`` is written in.
    ``F_c`` and ``F_s`` are per-order cosine/sine vectors or their diagonal matrices.
    Rotation only scales rows, so it needs neither dense products nor dense storage
    for those diagonals. The matrix spelling remains available to the full-tensor seed.
    """
    c = (F_c if F_c.ndim == 1 else xp.linalg.diagonal(F_c))[:, None]
    s = (F_s if F_s.ndim == 1 else xp.linalg.diagonal(F_s))[:, None]
    W_11 = W[:n_orders, :n_orders]
    W_12 = W[:n_orders, n_orders:]
    W_21 = W[n_orders:, :n_orders]
    W_22 = W[n_orders:, n_orders:]
    V_11 = V[:n_orders, :n_orders]
    V_12 = V[:n_orders, n_orders:]
    V_21 = V[n_orders:, :n_orders]
    V_22 = V[n_orders:, n_orders:]

    W_ss = c * W_21 - s * W_11
    W_sp = c * W_22 - s * W_12
    W_ps = c * W_11 + s * W_21
    W_pp = c * W_12 + s * W_22

    V_ss = c * V_11 + s * V_21
    V_sp = c * V_12 + s * V_22
    V_ps = c * V_21 - s * V_11
    V_pp = c * V_22 - s * V_12

    big_W = _block(xp, [[W_ss, W_sp], [W_ps, W_pp]])
    big_V = _block(xp, [[V_ss, V_sp], [V_ps, V_pp]])
    return big_W, big_V


def solve_2d(wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
             epx_conv_all, epy_conv_all, epz_conv_i_all,
             epxy_conv_all=None, epyx_conv_all=None,
             *, phi, incidence_side='top', answers=Answers.EFFICIENCY, lit=None,
             mu=None, mu_top=1.0, mu_bot=1.0, eps_top=None, eps_bot=None, dng=None):
    """Solve a 2D grating stack for the given incidence.

    ``incidence`` is the resolved incident amplitude pair ``(a_s, a_p)`` at unit flux — whatever
    when given — see :func:`~wavebit._common.conventions.incident_sp_amplitudes`.

    See :func:`wavebit.rcwa._engine.solve_1d.solve_1d` for the shared argument conventions —
    including ``incidence_side``, handled by the same mirror (the off-diagonal blocks
    reverse with the rest of the stack; a layer's in-plane tensor is untouched by ``z -> -z``).
    ``epxy_conv_all`` / ``epyx_conv_all`` are the optional off-diagonal permittivity blocks
    (``None`` -> zero).

    ``mu`` is the optional per-layer permeability, as five stacks
    ``(mux_conv_all, muy_conv_all, muz_conv_i_all, muxy_conv_all, muyx_conv_all)`` in the
    same slots and the same convention as the permittivity — see :func:`layer_modes_2d`. Any
    entry may be ``None``.

    ``mu_top`` / ``mu_bot`` are the **half-spaces'** scalar permeabilities. ``n_top`` / ``n_bot``
    keep meaning the *refractive index* ``sqrt(eps*mu)``, so nothing about ``kz``, Snell or the
    order grid changes and every existing call means what it did; ``mu`` enters only through
    the boundary admittances (:func:`~wavebit.rcwa._engine.solve_1d.sp_efficiencies`).

    ``eps_top`` / ``eps_bot`` are those half-spaces' permittivities. Optional, and recovered as
    ``n**2/mu`` when they are not given -- which is what this function did unconditionally, and
    is still right for a caller holding only an index. The engine passes them because it holds
    both numbers already: the recovery is a round trip through ``sqrt`` and back, exact only up
    to rounding, and there is no reason to take it when the value is in hand. Supply them with
    ``mu_top`` / ``mu_bot`` or with neither -- half a pair would describe a medium whose index
    and permittivity disagree.

    Each is read exactly once, as the p-channel admittance ``kz/eps`` of its own half-space:
    ``eps_bot`` in the substrate seed ``big_F`` below, ``eps_top`` in
    :func:`~wavebit.rcwa._engine.solve_1d.final_matching`'s top row. Both are the *same* two reads the
    SMM twin makes in ``W_bot`` / ``W_top``, which is the property that keeps the two
    connecting algorithms answering one question -- ``eps_top`` reached this signature before
    it reached the ETM matching, and for a while the two algorithms honoured different inputs.

    ``kx`` / ``ky`` are the retained in-plane orders, in either spelling :func:`order_grid_kxy`
    accepts: 1-D of lengths ``n_orders_u`` / ``n_orders_v`` for an axis-aligned cell, or
    ``(n_orders_v, n_orders_u)`` per-order Cartesian components for any lattice.

    Returns ``(de_ri, de_ti, power_entering)``: two efficiency arrays of shape
    ``(n_orders_v, n_orders_u)`` and a scalar net entering power fraction.
    """
    xp = array_namespace(epx_conv_all)
    mu_all = (None,) * 5 if mu is None else tuple(mu)
    if _is_backside(incidence_side):
        # mu reverses with the layers it belongs to. `z -> -z` leaves a layer's in-plane tensor
        # alone whether it is eps or mu, so the mirror is the same one, with five more stacks.
        (n_top, n_bot, thickness, epx_conv_all, epy_conv_all, epz_conv_i_all,
         epxy_conv_all, epyx_conv_all, *mu_all, dng) = _mirrored_scene(
            n_top, n_bot, thickness, epx_conv_all, epy_conv_all, epz_conv_i_all,
            epxy_conv_all, epyx_conv_all, *mu_all, dng)
        # The half-spaces swap with the indices they belong to -- all three quantities of
        # each, or the mirrored scene would take its permittivity from the far side.
        mu_top, mu_bot = mu_bot, mu_top
        eps_top, eps_bot = eps_bot, eps_top
    kx = as_array(xp, kx)
    ky = as_array(xp, ky)
    if not xp.isdtype(kx.dtype, 'complex floating'):
        kx = xp.astype(kx, xp.complex128)
        ky = xp.astype(ky, xp.complex128)
    C = kx.dtype

    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    n_orders_v, n_orders_u = kx_g.shape
    n_orders = n_orders_u * n_orders_v
    # Under an abstract trace a layer's near-defective pairs are rebuilt up to this many at once:
    # an invariant raster has one exceptional point per order row (or column) together.
    pair_cap = max(n_orders_u, n_orders_v)
    k0 = 2 * 3.141592653589793 / wavelength
    theta = 0.0 if theta is None else theta

    I = xp.astype(xp.eye(n_orders), C)
    O = xp.astype(xp.zeros((n_orders, n_orders)), C)

    # --- stage 1: boundary init (transfer_2d_1 == transfer_1d_conical_1) ----------------
    k_par_sq = kx_g ** 2 + ky_g ** 2
    kz_top = xp.reshape(boundary_kz(xp, n_top ** 2, k_par_sq,
                                    half_space_medium(eps_top, mu_top)), (-1,))
    kz_bot = xp.reshape(boundary_kz(xp, n_bot ** 2, k_par_sq,
                                    half_space_medium(eps_bot, mu_bot)), (-1,))
    varphi = order_azimuth(xp, ky_g, kx_g, phi)

    if eps_bot is None:
        eps_bot = n_bot ** 2 / mu_bot      # n stays the refractive index; eps carries the mu
    # The substrate's two admittances: kz/eps on the magnetic (p) amplitude, kz/mu on the
    # electric (s) one. Delay their dense storage until the first modes are
    # ready. The empty stack uses the same seed before final matching.
    def substrate_state():
        Kz_bot = _diag(xp, kz_bot)
        return (_block(xp, [[I, O], [O, -1j * Kz_bot / eps_bot]]),
                _block(xp, [[-1j * Kz_bot / mu_bot, O], [O, I]]),
                xp.astype(xp.eye(2 * n_orders), C))
    big_F = big_G = big_T = None

    # The order grid flattened row-major: m fastest, matching how the ``fourier2d_*`` builders
    # lay out the convolution matrices. Diagonal either way — the eigenproblem below reads
    # only these two diagonals, which is why a skewed lattice never reaches it.
    Kx = _diag(xp, xp.reshape(kx_g, (-1,)))
    Ky = _diag(xp, xp.reshape(ky_g, (-1,)))
    F_c, F_s = xp.cos(varphi), xp.sin(varphi)
    # --- stage 2/3: per-layer eig + coupled block transfer (transfer_2d_2 / transfer_2d_3)
    keep_field = answers_of(answers).covers(Answers.FIELD)
    layer_info_list = []
    n_layers = len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]
    run_k0_ds, run_links = _run_k0_d(xp, k0, thickness,
                                     (epx_conv_all, epy_conv_all, epz_conv_i_all,
                                      epxy_conv_all, epyx_conv_all, *mu_all), dng)
    below = None
    from wavebit.rcwa._engine import maxvol_labels
    use_maxvol = maxvol_labels.enabled(xp, kx, ky)
    flips_below = None
    for layer_index in range(n_layers)[::-1]:
        epx_conv = epx_conv_all[layer_index]
        epy_conv = epy_conv_all[layer_index]
        epz_conv_i = epz_conv_i_all[layer_index]
        epxy_conv = O if epxy_conv_all is None else epxy_conv_all[layer_index]
        epyx_conv = O if epyx_conv_all is None else epyx_conv_all[layer_index]
        d = thickness[layer_index]

        # `mux_conv`/`muy_conv` fall back to I and the other three to None, which is what
        # `layer_modes_2d` reads as non-magnetic -- so a stack with mu on some layers and not
        # others is expressible, and the layers without it take the original arithmetic.
        layer_mu = None if all(s is None for s in mu_all) else (
            I if mu_all[0] is None else mu_all[0][layer_index],
            I if mu_all[1] is None else mu_all[1][layer_index],
            *(None if s is None else s[layer_index] for s in mu_all[2:]))

        # The ETM's labels (:func:`~wavebit._common.conventions.etm_mode_q`): this recursion
        # inverts a transmission into the medium below, so a layer's `+` modes have to be
        # the waves that medium receives. Decided on the run's thickness
        # (:func:`~wavebit.rcwa._engine.solve_1d._run_k0_d`).
        env = None
        if use_maxvol is not False:
            from wavebit.rcwa._engine.modal_gradient import _stop
            if big_F is None:
                big_F, big_G, big_T = substrate_state()
            env = maxvol_labels.Environment(_stop(big_F), _stop(big_G), run_k0_ds[layer_index],
                                            flips_below, _run_join(run_links, layer_index, below),
                                            use_maxvol)
            env.frame = lambda W_, V_: sp_modes_2d(xp, W_, V_, F_c, F_s, n_orders)
        W, V, q, coupled = layer_modes_2d(xp, Kx, Ky, I, epx_conv, epy_conv, epz_conv_i,
                                          epxy_conv, epyx_conv, C, mu=layer_mu,
                                          k0_d=run_k0_ds[layer_index],
                                          dng=None if dng is None else dng[layer_index],
                                          pair_cap=pair_cap, with_coupling=True, env=env)
        flips_below = None if env is None else env.flips

        big_W, big_V = sp_modes_2d(xp, W, V, F_c, F_s, n_orders)
        if not keep_field:
            del W, V
        if big_F is None:
            big_F, big_G, big_T = substrate_state()

        big_I = xp.astype(xp.eye(2 * n_orders), C)
        reused = None
        if (env is not None and not coupled and not carries_autodiff(big_W)
                and not carries_autodiff(big_V)):
            reused = env.inverses(xp)
        if reused is None:
            big_W_i = xp.linalg.inv(big_W)
            big_V_i = xp.linalg.inv(big_V)
        else:
            big_W_i, big_V_i = reused
        # Exact inside a run, the plain form elsewhere
        # (:func:`~wavebit.rcwa._engine.solve_1d._run_ab`).
        big_A, big_B = _run_ab(xp, big_I, big_W_i, big_V_i, big_W, big_V, big_F, big_G, below,
                               _run_join(run_links, layer_index, below))
        big_A_i = xp.linalg.inv(big_A)
        del big_W_i, big_V_i, big_A

        from wavebit.rcwa._engine.modal_gradient import exponential
        # A layer with rebuilt pairs (`_rebuild_defective_pairs`) couples them in `q`'s value;
        # `coupled` is the builder's structural word, never read off the value.
        big_X = exponential(xp, q, -k0*d, coupled=pair_cap if coupled else False)

        Y = big_X @ big_B @ big_A_i @ big_X
        big_F = big_W @ (big_I + Y)
        big_G = big_V @ (big_I - Y)
        # Kept only while the next layer up may continue this one's run.
        below = _keep_below(run_links, layer_index, big_W, big_V, Y)
        del big_W, big_V, big_I, Y
        big_T = big_T @ big_A_i @ big_X

        # Bottom-to-top, matching the legacy append order; consumed by field reconstruction.
        # Conditional for the reason `solve_1d`'s append is -- and this is the path where it
        # costs the most: `W`, `V`, `big_A_i` and `big_B` are each a full `2*n_orders` square,
        # 4.25 of them per layer once `epz_conv_i` is counted at its quarter (4.5 on a
        # magnetic layer, where `muz_conv_i` is the second quarter).
        #
        # `muz_conv_i` rides in slot 7, and it is there for the same reason `epz_conv_i` is: the two
        # longitudinal components are algebraic rather than propagated, and each is recovered
        # by dividing out the z slot of its own tensor. `H_z = muz_conv_i (Kx Ey - Ky Ex)` is
        # Faraday's z row, dual to Ampere's z row that gives `E_z` -- see
        # :func:`wavebit.rcwa._engine.fields._field_coupled`. `None` is "non-magnetic", the spelling
        # `layer_mu` and `Material.mu` already use. The ninth slot is `None`, as in
        # :func:`~wavebit.rcwa._engine.solve_1d.solve_1d`'s record: only the SMM's carries one there.
        if keep_field:
            # A coupled layer's record carries a tenth slot (`None`): its presence is what tells
            # field reconstruction the layer's `q` couples its pairs -- a structural test that a
            # pytree round trip keeps.
            layer_info_list.append([epz_conv_i, W, V, q, d, big_A_i, big_B,
                                    None if layer_mu is None else layer_mu[2], None]
                                   + ([None] if coupled else []))
        # Field declarations own the saved references above. Other solves
        # release them before the next layer or final boundary elimination.
        del big_X, big_A_i, big_B

    # --- stage 3: final matching -> efficiency (transfer_2d_4 == transfer_1d_conical_4) --
    if big_F is None:
        big_F, big_G, big_T = substrate_state()
    out = final_matching(xp, C, big_F, big_G, big_T, kz_top, kz_bot, incidence, theta,
                         n_top, n_bot, n_orders_u, n_orders_v, answers=answers,
                         mu_top=mu_top, mu_bot=mu_bot, eps_top=eps_top, eps_bot=eps_bot,
                         lit=lit, kx=kx, ky=ky)
    if keep_field:
        out['layer_info_list'] = layer_info_list
    return out
