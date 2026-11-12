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
from wavebit._common.backend import _eig_raw
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
#: max(1, |w|)) is not rebuilt as a pair: the Newton step divides that row by its distance to
#: the pair. Measured on a Jordan pair beside a plain root at delta: the rebuilt pair's residual
#: 2.0e-12 at 1e-8, 3.5e-13 at 2e-8, <= 8.1e-14 from 5e-8 on (three random bases). The three
#: modes are a defective cluster instead, rebuilt jointly (:func:`_defective_clusters`). Another
#: gated pair is not counted: its rows take the 2x2 Sylvester closed form. That form also
#: divides by the two pairs' gap; two pairs within :data:`_EP_CLUSTER` of each other are one
#: cluster, rebuilt jointly, and the rest are screened after the rebuild (:data:`_EP_BACKWARD`).
_EP_SEPARATION = 1e-7

#: The radius of the clusters :func:`_defective_clusters` and the screens read.
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


#: Under an abstract trace the screens read, for every mode, the ball of at most this many nearest
#: modes within :data:`_EP_CLUSTER` instead of its whole connected cluster: twice the largest
#: cluster the joint rebuild takes, so a ball it leaves for being too large is still read whole up
#: to eight modes, and a larger one through its eight nearest.
_EP_SCREEN_BALL = 8


def _traced_balls(xp, w, group=None):
    """Every mode's ball for the traced screens: ``(idx, scale, near, count, nbr, valid)`` -- the
    mode indices, ``max(1, |w|)``, which modes lie within :data:`_EP_CLUSTER` of each (``group``
    restricting it) and how many, and each ball's at most :data:`_EP_SCREEN_BALL` nearest members,
    the mode itself first, with a lane mask (by argmin passes, as :func:`_closed_balls`)."""
    n2 = w.shape[0]
    m = min(_EP_SCREEN_BALL, n2)
    idx = xp.arange(n2, device=getattr(w, 'device', None))
    scale = xp.maximum(xp.ones_like(xp.abs(w)), xp.abs(w))
    d = xp.abs(w[:, None] - w[None, :])
    near = d <= _EP_CLUSTER * xp.maximum(scale[:, None], scale[None, :])
    if group is not None:
        near = near & (group[:, None] == group[None, :])
    count = xp.sum(xp.astype(near, xp.int32), axis=1)
    key = xp.where(near, d, xp.full_like(d, float('inf')))
    del d
    picks, found = [], []
    for _ in range(m):
        j = xp.argmin(key, axis=1)
        picks.append(j)
        found.append(xp.isfinite(xp.min(key, axis=1)))
        key = xp.where(idx[None, :] == j[:, None], xp.full_like(key, float('inf')), key)
    return idx, scale, near, count, xp.stack(picks, axis=1), xp.stack(found, axis=1)


def _one_seed_per_ball(xp, cand, near, nbr, valid, idx):
    """``cand`` less every mode whose whole ball -- every mode within the radius, not only the
    nearest the lanes hold -- is a lower-index candidate's (every member of a closed ball, say):
    one seed per ball, so a degenerate cluster larger than the lanes counts once."""
    same = xp.stack([xp.all(xp.take(near, nbr[:, l], axis=0) == near, axis=1)
                     for l in range(nbr.shape[1])], axis=1)
    lower = valid & (nbr < idx[:, None]) & same & xp.reshape(
        xp.take(cand, xp.reshape(nbr, (-1,))), nbr.shape)
    return cand & ~xp.any(lower, axis=1)


def _unresolved_cluster_traced(xp, w, W, operator, sine, group=None, slots=1):
    """Under an abstract trace, :func:`_unresolved_defective_cluster`'s question as data: a
    scalar boolean array, True where the layer must answer NaN.

    The eager screen walks each connected cluster of three or more modes; a trace cannot size a
    cluster, so it reads every mode's ball instead (:func:`_traced_balls`, ``group`` restricting
    it as eagerly). A ball with three or more modes whose unit columns' smallest singular value
    (from their Gram matrix) is below ``sine`` is a candidate, one per ball. The ``max(1,
    min(slots, n2 // 3))`` candidates with the smallest singular values take the defect measure
    (:data:`_DEFECT_NU`, scaled as eagerly), and the answer is True where one of them is
    defective, or where more candidates qualified than the slots hold (a degenerate ball in a
    nearly dependent basis counts toward that too, as in :func:`_defective_clusters`).

    A defective cluster's nearly dependent columns belong to one Jordan block, so they sit inside
    one ball; a chain's balls are read one by one, each the eager screen's measure on a part of
    the chain. ``operator()`` builds ``A`` once, for the slots.
    """
    n2 = w.shape[0]
    dev = getattr(W, 'device', None)
    C = W.dtype
    idx, _, near, count, nbr, valid = _traced_balls(xp, w, group)
    m = nbr.shape[1]
    Wn = W / xp.linalg.vector_norm(W, axis=0)[None, :]
    lane = xp.arange(m, device=dev)
    eye_m = xp.astype(lane[:, None] == lane[None, :], C)
    # The Gram matrix of each ball's unit columns, entry by entry, two column gathers alive at a
    # time (as the cluster gate builds it); padded lanes read the identity.
    G = xp.broadcast_to(eye_m[None], (n2, m, m))
    for i in range(m):
        Wi = xp.conj(xp.take(Wn, nbr[:, i], axis=1))
        for j in range(i + 1, m):
            g = xp.sum(Wi * xp.take(Wn, nbr[:, j], axis=1), axis=0)
            Eij = xp.astype((lane[:, None] == i) & (lane[None, :] == j), C)
            G = (G + g[:, None, None] * Eij[None]
                 + xp.conj(g)[:, None, None] * xp.matrix_transpose(Eij)[None])
        del Wi
    G = xp.where(valid[:, :, None] & valid[:, None, :], G, eye_m[None])
    low = xp.linalg.eigvalsh(G)[:, 0]
    sigma = xp.sqrt(xp.maximum(low, xp.zeros_like(low)))
    del G
    seed = _one_seed_per_ball(xp, (count >= 3) & (sigma < sine), near, nbr, valid, idx)
    del near
    S = max(1, min(slots, n2 // 3))
    order = xp.argsort(xp.where(seed, sigma, xp.full_like(sigma, float('inf'))), stable=True)
    sel = order[:S]
    live = xp.take(seed, sel)
    overflow = xp.sum(xp.astype(seed, xp.int32)) > S
    members = xp.take(nbr, sel, axis=0)
    mvalid = xp.take(valid, sel, axis=0) & live[:, None]
    # The restriction's departure from normality on the slots, as the eager screen reads it.
    A = operator()
    cols = _slot_columns(xp, Wn, members)
    Q0, _ = xp.linalg.qr(xp.where(mvalid[:, None, :], cols, xp.zeros_like(cols)))
    T = _conj_t(xp, Q0) @ (A @ Q0)
    T = xp.where(mvalid[:, :, None] & mvalid[:, None, :], T, xp.zeros_like(T))
    nv = xp.sum(xp.astype(mvalid, C), axis=1)
    lam = xp.sum(xp.linalg.diagonal(T), axis=1) / xp.where(nv != 0, nv, xp.ones_like(nv))
    TH = _conj_t(xp, T)
    nu = (xp.linalg.matrix_norm(T @ TH - TH @ T)
          / xp.maximum(xp.ones_like(xp.abs(lam)), xp.abs(lam)) ** 2)
    eps = float(xp.finfo(xp.real(T).dtype).eps) / 2.220446049250313e-16
    return xp.any(live & (nu > _DEFECT_NU * eps * eps)) | overflow


_UNRESOLVED_2D = (
    'a 2D layer holds three or more modes of one defective eigenvalue cluster (exceptional '
    'points that coincide) that no rebuild takes exactly -- more than eight modes, a chain of '
    'nearly equal eigenvalues, or a Jordan chain of length three: the eigenvector basis was '
    'kept, and the answer can be inaccurate. Move the angle or wavelength by more than a '
    'relative 1e-4 to check it.')

#: A cluster (:data:`_EP_CLUSTER`) holding two or more pairs rebuilt one at a time -- a cluster
#: the joint rebuild did not take; its own clusters answer to its check (:data:`_EP_RESIDUAL`)
#: -- warns when the layer model
#: its modes define misses the operator by more than this, in units of the cluster's own scale
#: ``max(1, |w|)`` and scaled by ``eps / eps64``: ``||(A W_c - W Q^2[:, c]) W^-1[c, :]||_2``,
#: ``c`` the cluster's columns. Two pairs a few 1e-8 .. 2e-6 apart -- exceptional points that
#: nearly coincide, such as two order rows of a y-invariant layer just off one x-cutoff -- leave
#: the one Newton step outside its convergence region: the 2x2 Sylvester rows between the pairs
#: divide by their gap, and which split of the pairs' joint subspace the step lands on depends
#: on the rounding. Measured on the y-Littrow scenes PL952 and PA616 at truncations 5, 10, 15
#: and 20, each at its own exceptional-point wavelength (2D, ETM and SMM, 36 offsets of either
#: sign, four CPU arithmetic paths; 2304 solves): every answer more than 1e-12 off that the
#: cluster screen let through missed by at least 1.43e-11, and every one at or below 3e-12 was
#: within 1.0e-13. Not relative to ``||A||_2``: that grows with the truncation through the
#: evanescent orders (11.6 .. 366 here) while the misses that matter do not, and a threshold
#: that separated them at truncation 5 left answers up to 3.4e-12 off silent at 15 and 20. Nor
#: the pairs' gap, which bounds the silent answers one-sidedly too but at a different spread per
#: scene: PA616's band ended at a gap of 8.5e-8, PL952's at 8.5e-7.
_EP_BACKWARD = 3e-12

#: The propagation sensitivity the threshold above was calibrated at: ``S = k0 d / (2 min|q|)``
#: over the cluster, the phase a miss in ``q**2`` buys across the layer. Every calibrated
#: cluster had ``S <= 3.47`` (PA616, 300 nm). A thicker layer, or a cluster whose roots are
#: small, turns the same miss into a larger error: a 10 um layer at ``|q| = 0.025``
#: (``S = 2.0e3``) was 1.1e-11 off with a miss below 3e-12. Above this the threshold is divided
#: by ``S / _EP_SENSITIVITY``, so the calibrated scenes read exactly as before.
_EP_SENSITIVITY = 3.5

_IMPRECISE_2D = (
    'a 2D layer holds two or more exceptional-point pairs whose eigenvalues nearly coincide '
    '(such as two order rows of a y-invariant layer just off the same x-cutoff), and the pair '
    'rebuild could not separate them: the rebuilt modes miss the layer operator by more than the '
    'layer can absorb, and the answer can be inaccurate. Move the angle or wavelength by more '
    'than a relative 1e-4 and compare the answers; in a thick layer whose roots are small this '
    'warning also fires on exact answers, out to angle moves of about a relative 1e-3.')


def _crowded_pairs_imprecise(xp, w, A, W, Q, a, b, mask, k0_d=None):
    """Eagerly, whether a cluster holding two or more rebuilt pairs has a backward error above
    :data:`_EP_BACKWARD`; ``None`` under an abstract trace. ``layer_modes_2d`` hands it only the
    pairs outside the jointly rebuilt clusters.

    ``(W, Q, a, b, mask)`` is :func:`_rebuild_defective_pairs`' answer for the product operator
    ``A``, with ``Q`` the ``2n x 2n`` matrix ``diag(q) + tau``. A cluster is a connected set of
    eigenvalues ``w`` each within :data:`_EP_CLUSTER` of the next, as in
    :func:`_unresolved_defective_cluster`. For its columns ``c`` the backward error of the layer
    model is ``||(A W_c - W Q^2[:, c]) W^-1[c, :]||_2``, the part of ``A - W Q^2 W^-1`` those
    columns carry, compared with the cluster's scale ``max(1, |w|)``. ``W^-1[c, :]`` comes from
    one solve with ``W^T``, and the 2-norm of the rank-``k`` product from a QR of that block:
    ``||R H||_2 = ||R S^H||_2`` for ``H^H = U S``. ``k0_d``, the layer's optical thickness
    (its run's, as the ETM labels read it), tightens the threshold where the cluster's
    propagation sensitivity exceeds :data:`_EP_SENSITIVITY`; ``None`` or an unreadable value
    keeps the calibrated threshold.
    """
    several = _concrete(xp.sum(xp.astype(mask, xp.int32)) >= 2)
    if not several:
        return None if several is None else False
    n2 = w.shape[0]
    dev = getattr(W, 'device', None)
    idx = xp.arange(n2, device=dev)
    scale = xp.maximum(xp.ones_like(xp.abs(w)), xp.abs(w))
    near = xp.abs(w[:, None] - w[None, :]) <= _EP_CLUSTER * xp.maximum(scale[:, None],
                                                                          scale[None, :])
    is_a = xp.any((idx[:, None] == a[None, :]) & mask[None, :], axis=1)
    pending = is_a
    eps = float(xp.finfo(xp.real(W).dtype).eps) / 2.220446049250313e-16
    half_k0_d = None
    if k0_d is not None:
        try:
            half_k0_d = checked_float(abs(k0_d)) / 2
        except Exception:
            half_k0_d = None
    roots = xp.abs(xp.linalg.diagonal(Q))
    while _concrete(xp.any(pending)):
        # Graph membership stays array data; only Boolean reductions cross to Python.
        comp = idx == xp.argmax(xp.astype(pending, xp.int32))
        frontier = comp
        while _concrete(xp.any(frontier)):
            neighbors = xp.any(xp.take(near, xp.nonzero(frontier)[0], axis=0), axis=0)
            frontier = neighbors & ~comp
            comp = comp | frontier
        pending = pending & ~comp
        if not _concrete(xp.sum(xp.astype(comp & is_a, xp.int32)) >= 2):
            continue
        members = xp.nonzero(comp)[0]
        Wc = xp.take(W, members, axis=1)
        R = A @ Wc - W @ (Q @ xp.take(Q, members, axis=1))
        E = xp.astype(idx[:, None] == members[None, :], W.dtype)
        Hc = linear_solve(xp, xp.matrix_transpose(W), E)      # W^-1[c, :] transposed
        _, S = xp.linalg.qr(xp.conj(Hc))
        miss = xp.linalg.svdvals(R @ xp.matrix_transpose(xp.conj(S)))[0]
        limit = _EP_BACKWARD * eps * xp.max(xp.take(scale, members))
        loud = miss > limit
        if half_k0_d is not None:
            # miss * max(1, S / S0) > limit, with S = k0 d / (2 min|q|) over the cluster
            loud = loud | (miss * half_k0_d
                           > limit * _EP_SENSITIVITY * xp.min(xp.take(roots, members)))
        if _concrete(loud):
            return True
    return False


def _crowded_pairs_traced(xp, w, A, W, Q, a, b, mask, k0_d=None, slots=1):
    """Under an abstract trace, :func:`_crowded_pairs_imprecise`'s question as data: a scalar
    boolean array, True where the layer must answer NaN.

    As :func:`_unresolved_cluster_traced` reads balls for clusters: a mode that is the first of a
    pair rebuilt one at a time (``mask``) and whose ball (:func:`_traced_balls`) holds another such
    pair is a candidate, one per ball. The first ``max(1, min(slots, n2 // 4))`` candidates take
    the eager screen's backward error on their ball's columns -- one solve with ``W^T`` for every
    slot's rows of ``W^-1`` together -- against the same threshold, with the propagation
    sensitivity read from ``k0_d`` as data; more candidates than slots answer True as well.
    """
    n2 = w.shape[0]
    C = W.dtype
    idx, scale, near, _, nbr, valid = _traced_balls(xp, w)
    m = nbr.shape[1]
    is_a = xp.any((idx[:, None] == a[None, :]) & mask[None, :], axis=1)
    held = xp.sum(xp.astype(valid & xp.reshape(xp.take(is_a, xp.reshape(nbr, (-1,))), nbr.shape),
                            xp.int32), axis=1)
    seed = _one_seed_per_ball(xp, is_a & (held >= 2), near, nbr, valid, idx)
    del near
    S = max(1, min(slots, n2 // 4))
    sel = xp.argsort(xp.astype(~seed, xp.int32), stable=True)[:S]          # seeds first
    live = xp.take(seed, sel)
    overflow = xp.sum(xp.astype(seed, xp.int32)) > S
    members = xp.take(nbr, sel, axis=0)
    mvalid = xp.take(valid, sel, axis=0) & live[:, None]

    def lanes(X):
        return xp.where(mvalid[:, None, :], X, xp.zeros_like(X))
    # The part of A - W Q^2 W^-1 each slot's columns c carry: (A W_c - W Q Q[:, c]) W^-1[c, :].
    R = A @ lanes(_slot_columns(xp, W, members)) - W @ lanes(Q @ _slot_columns(xp, Q, members))
    E = xp.astype((idx[None, :, None] == members[:, None, :]) & mvalid[:, None, :], C)
    H = linear_solve(xp, xp.matrix_transpose(W),
                     xp.reshape(xp.permute_dims(E, (1, 0, 2)), (n2, S * m)))
    H = xp.permute_dims(xp.reshape(H, (n2, S, m)), (1, 0, 2))              # W^-1[c, :] transposed
    _, Sr = xp.linalg.qr(xp.conj(H))
    miss = xp.linalg.svdvals(R @ _conj_t(xp, Sr))[:, 0]
    eps = float(xp.finfo(xp.real(W).dtype).eps) / 2.220446049250313e-16
    sc = xp.reshape(xp.take(scale, xp.reshape(members, (-1,))), members.shape)
    limit = _EP_BACKWARD * eps * xp.max(xp.where(mvalid, sc, xp.zeros_like(sc)), axis=1)
    loud = miss > limit
    if k0_d is not None:
        # miss * max(1, S / S0) > limit, with S = k0 d / (2 min|q|) over the slot's columns
        roots = xp.reshape(xp.take(xp.abs(xp.linalg.diagonal(Q)), xp.reshape(members, (-1,))),
                           members.shape)
        low = xp.min(xp.where(mvalid, roots, xp.full_like(roots, float('inf'))), axis=1)
        half_k0_d = xp.abs(xp.asarray(k0_d)) / 2
        loud = loud | (miss * half_k0_d > limit * _EP_SENSITIVITY * low)
    return xp.any(live & loud) | overflow


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


#: The most modes one defective cluster may hold for the joint rebuild
#: (:func:`_defective_clusters`): two exceptional-point pairs at or near one coincidence, or a
#: pair and the unpaired root beside it -- and, two parameters tuned at once, two such
#: coincidences at one point: an even and an odd mode of an x-symmetric raster at one x-cutoff
#: in two order rows, eight modes (2026-11-12; four left that loud and up to 1.5 off). Static, so
#: that every traced slot has one shape; a cluster of fewer modes pads its last lanes. Eagerly a
#: layer with no ball of five or more keeps four lanes a slot (:func:`_defective_clusters`).
_EP_CLUSTER_SIZE = 8


def _gather_pairs(xp, M, rows, cols):
    """``M[rows, cols]`` entry by entry (``rows`` and ``cols`` broadcast), by one flat take."""
    rows, cols = xp.broadcast_arrays(rows, cols)
    flat = xp.take(xp.reshape(M, (-1,)), xp.reshape(rows * M.shape[1] + cols, (-1,)))
    return xp.reshape(flat, rows.shape)


def _slot_columns(xp, W, members):
    """``(S, n, k)``: column ``members[s, l]`` of ``W`` at ``[s, :, l]``."""
    S, k = members.shape
    cols = xp.take(W, xp.reshape(members, (-1,)), axis=1)
    return xp.permute_dims(xp.reshape(cols, (W.shape[0], S, k)), (1, 0, 2))


def _conj_t(xp, X):
    return xp.conj(xp.matrix_transpose(X))


#: The joint rebuild reads balls of this radius first (relative to ``max(1, |w|)``), and the
#: :data:`_EP_CLUSTER` balls only where no wider one is taken (:func:`_defective_clusters`).
#: Off a symmetry coincidence -- a C4v pillar at normal incidence, where two exceptional points
#: of one degenerate pair of orders coincide -- the two split into degenerate doublets whose
#: spread grows like the square root of the offset, while their four columns stay nearly
#: dependent: measured spread 1.6e-4 .. 2.6e-4 with a smallest singular value 4e-5 .. 7e-4 at
#: relative offsets 1e-8 .. 1e-7. No pair is gated there (each mode's nearest eigenvalue is its
#: degenerate partner, at a sine above :data:`_EP_SINE`), so outside the :data:`_EP_CLUSTER`
#: ball the product construction was silent and up to 9.1e-12 off. With this radius every
#: offset measured (0 .. 1e-5 either side of five coincidences, Laurent and NVF, and 1e-4, 1e-3
#: above) was within
#: 5.4e-14 of a Cauchy oracle and quiet; a radius of 1e-3 left up to 3.2e-13. A wider ball
#: holding more than :data:`_EP_CLUSTER_SIZE` modes is left to
#: the narrower ones inside it, so clusters closer than this to other clusters or pairs keep
#: the :data:`_EP_CLUSTER` rebuild.
_EP_JOINT_RADIUS = 3e-3


def _closed_balls(xp, d, span, radius, rows_of, k, idx, dev):
    """The balls of ``radius`` (``d <= radius * span``) seeded at their lowest-index member:
    ``(rows, nbr, valid, closed, count_r)`` -- the seed rows read (eagerly the seeds only, under
    a trace every row), each ball's members nearest first with a lane mask, and whether the ball
    is closed (every member's own ball is the same set) and holds three to ``k`` modes."""
    near = d <= radius * span
    count = xp.sum(xp.astype(near, xp.int32), axis=1)
    lower = xp.any(near & (idx[None, :] < idx[:, None]), axis=1)
    seed = (count >= 3) & (count <= k) & ~lower
    rows = rows_of(seed)
    if rows is None:
        return None
    m = rows.shape[0]
    near_r = xp.take(near, rows, axis=0)
    key = xp.where(near_r, xp.take(d, rows, axis=0), xp.full(near_r.shape, float('inf'),
                                                               dtype=d.dtype, device=dev))
    # The ball's members first, nearest first, ties by index: k argmin passes. A pass that finds
    # no member left is a padded lane (its index is whatever argmin returns over infinities).
    picks, found = [], []
    for _ in range(k):
        j = xp.argmin(key, axis=1)
        picks.append(j)
        found.append(xp.isfinite(xp.min(key, axis=1)))
        key = xp.where(idx[None, :] == j[:, None], xp.full_like(key, float('inf')), key)
    del key, near_r
    nbr = xp.stack(picks, axis=1)                                         # (m, k)
    valid = xp.stack(found, axis=1)
    vv = valid[:, :, None] & valid[:, None, :]
    clique = xp.all(~vv | _gather_pairs(xp, near, nbr[:, :, None], nbr[:, None, :]), axis=(1, 2))
    count_r = xp.take(count, rows)
    same_ball = xp.all(~valid | (xp.reshape(xp.take(count, xp.reshape(nbr, (-1,))), nbr.shape)
                                 == count_r[:, None]), axis=1)
    del m
    return rows, nbr, valid, xp.take(seed, rows) & clique & same_ball, count_r


def _defective_clusters(xp, w, W, q, pairs, operator, slots, group=None):
    """The defective clusters of a 2D layer the joint rebuild takes, or ``None`` when none is.

    A cluster is a ball of the product operator's eigenvalues: the modes within a radius of its
    lowest-index member (relative to ``max(1, |w|)``), read at :data:`_EP_JOINT_RADIUS` and, where
    no ball of that radius is taken, at :data:`_EP_CLUSTER` (the radius
    :func:`_unresolved_defective_cluster` reads). It is taken when the ball is closed -- every
    member's own ball is the same set, so no chain runs out of it -- holds three to
    :data:`_EP_CLUSTER_SIZE` modes with roots above :data:`_EP_ROOT_FLOOR`, its unit columns are
    nearly dependent (smallest singular value below :data:`_EP_SINE`), and its restriction
    ``T = Q^H A Q`` departs from normality (:data:`_DEFECT_NU`). That is what ``eig`` hands back
    where exceptional points coincide or nearly coincide -- two order rows of a y-invariant layer
    at one x-cutoff -- and where a pair has an unpaired root within :data:`_EP_SEPARATION`, which
    :func:`_defective_pairs` refuses. A ball of exactly three that holds one gated pair is left to
    the pair rebuild: exact for a pair beside a root more than :data:`_EP_SEPARATION` away, not
    for a length-3 Jordan chain (which no disjoint split represents; the unresolved screen then
    warns). A degenerate (normal) cluster is left alone, whatever basis ``eig`` chose for it.

    Returns ``(members, valid, Q0, overflow)``: ``members`` ``(S, k)`` the clusters' mode
    indices, nearest the seed first; ``valid`` which lanes are members (a smaller cluster pads
    its last lanes, and under a trace a slot nothing took is all padding); ``Q0`` ``(S, n2, k)`` an
    orthonormal basis of each cluster's unit columns (zero-padded before the QR); and, under an
    abstract trace, whether more clusters qualified than the ``slots`` held (``None`` eagerly).
    ``operator()`` builds the product operator, only when a candidate needs it. ``group``
    (optional, one label per mode) keeps modes of different groups out of one ball: the
    first-order split's two directions (:func:`_unresolved_defective_cluster` reads it so too).

    Read through :func:`~wavebit._common.backend._concrete`, as :func:`_defective_pairs` is.
    Eagerly the answer is exactly the qualifying clusters, and a layer with none returns ``None``
    after an ``O(n2**2)`` look at the eigenvalues; under an abstract trace there is nothing to
    read, so ``max(1, min(slots, n2 // 3))`` slots come back, the qualifying balls with the
    smallest singular values, with their masks as data. The defect measure is read only on the
    slots, so a degenerate ball whose ``eig`` basis is nearly dependent counts toward the
    overflow too: a trace can return NaN for a layer whose eager solve takes no cluster.
    """
    n2 = w.shape[0]
    k = min(_EP_CLUSTER_SIZE, n2)
    dev = getattr(W, 'device', None)
    idx = xp.arange(n2, device=dev)
    scale = xp.maximum(xp.ones_like(xp.abs(w)), xp.abs(w))
    d = xp.abs(w[:, None] - w[None, :])
    if group is not None:
        d = xp.where(group[:, None] == group[None, :], d, xp.full_like(d, float('inf')))
    span = xp.maximum(scale[:, None], scale[None, :])
    # Eagerly a layer with no ball of five to `k` modes at either radius keeps four lanes a slot:
    # it runs as it did before balls of eight were taken, to the bit (a padded lane changes the
    # rounding of every product with the slot). Under a trace nothing is read, and every slot
    # has the full count.
    if k > 4:
        big = False
        for radius in (_EP_JOINT_RADIUS, _EP_CLUSTER):
            count = xp.sum(xp.astype(d <= radius * span, xp.int32), axis=1)
            some = _concrete(xp.any((count > 4) & (count <= k)))
            big = None if some is None else (big or some)
            if big is None or big:
                break
        if big is False:
            k = 4
    Wn = W / xp.linalg.vector_norm(W, axis=0)[None, :]
    C = W.dtype
    lane = xp.arange(k, device=dev)
    eye_k = xp.astype(lane[:, None] == lane[None, :], C)
    if pairs is not None:
        a, b, mask = pairs[:3]
        in_pair = xp.any(((idx[:, None] == a[None, :]) | (idx[:, None] == b[None, :]))
                         & mask[None, :], axis=1)
    else:
        in_pair = xp.zeros(n2, dtype=xp.bool, device=dev)

    def rows_of(seed):
        # Eagerly only the seeds' rows are read; under a trace every row, masked by `seed`.
        some = _concrete(xp.any(seed))
        if some is False:
            return None
        return xp.nonzero(seed)[0] if some else idx

    def candidates(ball):
        rows, nbr, valid, closed, count_r = ball
        m = rows.shape[0]
        qabs = xp.reshape(xp.abs(xp.take(q, xp.reshape(nbr, (-1,)))), nbr.shape)
        floor_ok = xp.all(~valid | (qabs > _EP_ROOT_FLOOR), axis=1)
        # The Gram matrix of the ball's unit columns, entry by entry: two n2-column gathers alive
        # at a time, the pair gate's peak. Padded lanes read the identity.
        vv = valid[:, :, None] & valid[:, None, :]
        G = xp.broadcast_to(eye_k[None], (m, k, k))
        for i in range(k):
            Wi = xp.conj(xp.take(Wn, nbr[:, i], axis=1))
            for j in range(i + 1, k):
                g = xp.sum(Wi * xp.take(Wn, nbr[:, j], axis=1), axis=0)
                Eij = xp.astype((lane[:, None] == i) & (lane[None, :] == j), C)
                G = (G + g[:, None, None] * Eij[None]
                     + xp.conj(g)[:, None, None] * xp.matrix_transpose(Eij)[None])
            del Wi
        G = xp.where(vv, G, eye_k[None])
        # Rounding can leave a rank-deficient Gram's smallest eigenvalue below zero.
        low = xp.linalg.eigvalsh(G)[:, 0]
        sigma = xp.sqrt(xp.maximum(low, xp.zeros_like(low)))
        held = xp.sum(xp.astype(valid & xp.reshape(xp.take(in_pair, xp.reshape(nbr, (-1,))),
                                                   nbr.shape), xp.int32), axis=1)
        lone_pair = (count_r == 3) & (held == 2)
        return nbr, valid, sigma, closed & floor_ok & (sigma < _EP_SINE) & ~lone_pair

    # The wide balls first; a narrow ball inside a wide one the wide level takes is its business
    # (a closed wide ball holds every mode within its radius of a member, so the narrow ball's
    # seed among its members means the whole narrow ball is).
    levels = []
    for radius in (_EP_JOINT_RADIUS, _EP_CLUSTER):
        ball = _closed_balls(xp, d, span, radius, rows_of, k, idx, dev)
        if ball is not None:
            levels.append((ball[0],) + candidates(ball))
    if not levels:
        return None
    if len(levels) == 2:
        (_, nbr_w, valid_w, _, cand_w), (rows_n, nbr_n, valid_n, sigma_n, cand_n) = levels
        taken = xp.any((nbr_w[:, :, None] == idx[None, None, :])
                       & (cand_w[:, None, None] & valid_w[:, :, None]), axis=(0, 1))
        levels[1] = (rows_n, nbr_n, valid_n, sigma_n, cand_n & ~xp.take(taken, rows_n))
    nbr = xp.concat([lv[1] for lv in levels], axis=0)
    valid = xp.concat([lv[2] for lv in levels], axis=0)
    sigma = xp.concat([lv[3] for lv in levels], axis=0)
    cand = xp.concat([lv[4] for lv in levels], axis=0)
    decided = _concrete(xp.any(cand))
    if decided is False:
        return None
    overflow = None
    if decided:
        sel = xp.nonzero(cand)[0]
        live = xp.ones(sel.shape, dtype=xp.bool, device=dev)
    else:
        S = max(1, min(slots, n2 // 3))
        order = xp.argsort(xp.where(cand, sigma, xp.full_like(sigma, float('inf'))), stable=True)
        sel = order[:S]
        live = xp.take(cand, sel)
        overflow = xp.sum(xp.astype(cand, xp.int32)) > S
    members = xp.take(nbr, sel, axis=0)
    mvalid = xp.take(valid, sel, axis=0) & live[:, None]
    # The restriction's departure from normality, as the unresolved screen reads it.
    A = operator()
    cols = _slot_columns(xp, Wn, members)
    Q0, _ = xp.linalg.qr(xp.where(mvalid[:, None, :], cols, xp.zeros_like(cols)))
    T = _conj_t(xp, Q0) @ (A @ Q0)
    T = xp.where(mvalid[:, :, None] & mvalid[:, None, :], T, xp.zeros_like(T))
    nv = xp.sum(xp.astype(mvalid, C), axis=1)
    lam = xp.sum(xp.linalg.diagonal(T), axis=1) / xp.where(nv != 0, nv, xp.ones_like(nv))
    TH = _conj_t(xp, T)
    nu = (xp.linalg.matrix_norm(T @ TH - TH @ T)
          / xp.maximum(xp.ones_like(xp.abs(lam)), xp.abs(lam)) ** 2)
    eps = float(xp.finfo(xp.real(T).dtype).eps) / 2.220446049250313e-16
    engaged = live & (nu > _DEFECT_NU * eps * eps)
    if decided:
        if not _concrete(xp.any(engaged)):
            return None
        keep = xp.nonzero(engaged)[0]
        return (xp.take(members, keep, axis=0), xp.take(mvalid, keep, axis=0),
                xp.take(Q0, keep, axis=0), None)
    return members, mvalid & engaged[:, None], Q0, overflow


def _split_candidates(k):
    """Every nonempty proper subset ``J`` of ``k`` lanes, grouped by size: for each size ``j``
    the ``(nc, j)`` lanes of ``J``, the ``(nc, k - j)`` lanes of its complement (ascending), the
    ``(nc, k)`` mask of ``J``, and the ``(nc, k)`` lane order of ``[J | K]``."""
    import itertools

    import numpy as np
    groups = []
    for j in range(1, k):
        combos = list(itertools.combinations(range(k), j))
        rest = [[x for x in range(k) if x not in J] for J in combos]
        mask = np.zeros((len(combos), k), dtype=bool)
        for c, J in enumerate(combos):
            mask[c, list(J)] = True
        order = np.array([list(J) + rest[c] for c, J in enumerate(combos)])
        groups.append((j, np.array(combos), np.array(rest), mask, order))
    return groups


def _dead_last(xp, lam, E, mu, ndead):
    """Eigenpairs with the ``ndead`` nearest ``mu`` moved to the last positions, the rest in
    front, each part in ``eig``'s order.

    Padded lanes are decoupled at the stand-in eigenvalue ``mu`` (a cluster of ``m`` members
    pads its last ``k - m`` lanes); ``eig`` returns their pairs in no promised position, so they
    are put back on their lanes explicitly, by a one-hot permutation -- the identity where
    ``eig`` already handed them back last. Ties in the distance go by position. ``ndead`` is
    clipped to ``k``, so the result is a permutation even for a quotient smaller than the
    padding (a split candidate whose ``J`` holds a padded lane, which the split discards).
    """
    k = lam.shape[-1]
    dev = getattr(lam, 'device', None)
    ar = xp.arange(k, device=dev)
    d = xp.abs(lam - mu[..., None])
    before = ar[None, :] < ar[:, None]                               # [p, q]: q before p
    closer = (d[..., None, :] < d[..., :, None]) | ((d[..., None, :] == d[..., :, None]) & before)
    nd = xp.where(ndead < k, ndead, k * xp.ones_like(ndead))[..., None]
    padded = xp.sum(xp.astype(closer, xp.int32), axis=-1) < nd       # (..., k)
    ahead_m = xp.sum(xp.astype(before & ~padded[..., None, :], xp.int32), axis=-1)
    ahead_p = xp.sum(xp.astype(before & padded[..., None, :], xp.int32), axis=-1)
    dest = xp.where(padded, k - nd + ahead_p, ahead_m)
    P = xp.astype(dest[..., :, None] == ar, E.dtype)                 # P[p, l]: l takes p
    return xp.sum(lam[..., :, None] * P, axis=-2), E @ P


#: :func:`_cluster_split` tries again with :func:`_spread_eigenvectors` where its best split's
#: ``cond(X)`` exceeds this. Splits from independent eigenvectors measured at most 20.0; the
#: check (:func:`_joint_cluster_bases`) reads ``cond(X)`` times a dropped part that itself grows
#: with ``cond(X)``: measured, it passed up to 146 and failed from 185 (2026-11-12). Below it the
#: split is ``eig``'s, to the bit.
_SPLIT_RETRY = 32.0

#: :func:`_spread_eigenvectors`: eigenvalues within this of each other, relative to
#: ``max(1, |lambda|)``, are one near set. Two Jordan pairs at one eigenvalue come back from
#: ``eig`` spread by about the square root of the rounding (1e-9 at ``DC612``); a near set wider
#: than the true multiplicity finds no eigenvector orthogonal to it, and the residual test keeps
#: ``eig``'s.
_SPREAD_NEAR = 1e-6

#: :func:`_spread_eigenvectors` keeps a re-derived vector only where its residual
#: ``||T x - rho x||`` is within this many ulp of ``||T||``: an eigenvector to rounding.
_SPREAD_RES = 16.0


def _spread_eigenvectors(xp, T, lam, E, live):
    """``eig``'s unit eigenvectors ``E`` of ``T`` with every collapsed one re-derived.

    At an eigenvalue of geometric multiplicity two or more -- two Jordan pairs at one eigenvalue,
    or a degenerate eigenvalue that is not defective -- ``eig``'s vectors are the eigenvectors of
    a rounding-sized perturbation within the eigenspace, and may come back nearly parallel
    (``DC612`` at ``theta`` 1e-4, first order, on some OpenBLAS kernels: the four vectors of one
    eigenvalue within a sine of 1.3e-3 of one another, and the best split's ``cond(X)`` 1833).
    For each live lane ``i`` in order, where an
    earlier live lane's eigenvalue is within :data:`_SPREAD_NEAR`, the lane takes the unit vector
    with the smallest residual against ``T - lambda_i`` orthogonal to those earlier lanes ``V``:
    the right singular vector of the smallest singular value of ``[T - lambda_i I; ||T|| V^H]``.
    It is kept only where it is an eigenvector to rounding (:data:`_SPREAD_RES`); a lane with no
    such vector -- the second member of a Jordan pair -- keeps ``eig``'s. Inverse iteration
    orthogonalises the vectors of close eigenvalues against each other in the same way (G. H.
    Golub and C. F. Van Loan, Matrix Computations, 4th ed., section 7.6.1; LAPACK's ``?stein``).
    The lane's own eigenvalue serves at a perturbed Jordan pair as well as the near set's mean:
    ``T - lambda_i`` there has a second singular value at rounding (measured on 900 random
    operators, the same verdicts).

    ``T`` is ``(S, k, k)``, ``lam`` and ``E`` ``eig``'s, ``live`` ``(S, k)`` the member lanes. Values
    only: ``T`` is stopped.
    """
    S, k, _ = T.shape
    C = T.dtype
    dev = getattr(T, 'device', None)
    rdt = xp.real(T).dtype
    eps = float(xp.finfo(rdt).eps)
    scale = xp.maximum(xp.ones_like(xp.abs(lam)), xp.abs(lam))
    near = ((xp.abs(lam[:, :, None] - lam[:, None, :]) <= _SPREAD_NEAR * scale[:, :, None])
            & live[:, :, None] & live[:, None, :])
    norm_T = xp.linalg.matrix_norm(T)
    eye = xp.astype(xp.eye(k, device=dev), C)
    cols = [E[:, :, 0]]
    for i in range(1, k):
        prev = near[:, i, :i]                                               # (S, i)
        V = xp.stack(cols, axis=-1) * xp.astype(prev, C)[:, None, :]
        A = xp.concat([T - lam[:, i][:, None, None] * eye,
                       xp.astype(norm_T, C)[:, None, None] * _conj_t(xp, V)], axis=-2)
        _, _, Vh = xp.linalg.svd(A, full_matrices=False)
        x = xp.conj(Vh[:, -1, :])
        Tx = (T @ x[:, :, None])[:, :, 0]
        rho = xp.sum(xp.conj(x) * Tx, axis=-1)
        res = xp.linalg.vector_norm(Tx - rho[:, None] * x, axis=-1)
        keep = xp.any(prev, axis=-1) & (res <= _SPREAD_RES * eps * norm_T)
        cols.append(xp.where(keep[:, None], x, E[:, :, i]))
    return xp.stack(cols, axis=-1)


def _cluster_split(xp, T, valid, pad):
    """A basis ``X`` with ``X^-1 T X`` diagonal but for a block on rows ``J``, columns ``K``.

    ``T`` is ``(S, k, k)``, each cluster's restriction with its padded lanes decoupled at the
    stand-in ``pad``. ``J`` ranges over every nonempty proper subset of ``eig(T)``'s unit
    eigenvectors that holds only member lanes; ``K``'s columns are the eigenvectors of the
    quotient ``C^H T C`` on the orthonormal complement ``C`` of ``span(E_J)`` (G. H. Golub and
    C. F. Van Loan, Matrix Computations, 4th ed., sections 7.1.4 and 7.6). Then
    ``X = [E_J | C Y]`` puts ``T`` in the block upper-triangular form ``[[L_J, B], [0, L_K]]``,
    whose coupling ``B`` has rows ``J`` and columns ``K`` disjoint: the form the exponential,
    inverse and derivative rules below take exactly. Where every Jordan chain of the cluster has
    length at most two, some ``J`` leaves no other off-diagonal entry; the split with the
    smallest ``cond(X)`` is taken (measured at most 8.14 on the y-Littrow scenes, 12.2 on a
    genuinely 2D one, 20.0 on eight modes at a double x-cutoff).

    Where that ``cond(X)`` exceeds :data:`_SPLIT_RETRY` -- ``eig`` handed back nearly parallel
    vectors at an eigenvalue of multiplicity two or more -- the split is tried again from
    :func:`_spread_eigenvectors`, and the one with the smaller ``cond(X)`` times its dropped part
    (what the check reads) is kept. Eagerly a layer whose splits are all below it never forms the
    second; under a trace both are formed and one is selected per slot.

    Returns ``(X, J, cond)``. A slot with no admissible split -- and under a trace a slot with no
    members -- takes ``X = I`` and ``cond = inf``; every padded row and column of ``X`` is the
    identity's, so no stand-in leaks into a member's column.
    """
    import numpy as np
    S, k, _ = T.shape
    C = T.dtype
    dev = getattr(T, 'device', None)
    lane = xp.arange(k, device=dev)
    eye = xp.astype(lane[:, None] == lane[None, :], C)
    ndead = k - xp.sum(xp.astype(valid, xp.int32), axis=1)          # the padded lanes, last
    vv = valid[:, :, None] & valid[:, None, :]
    mu = pad * xp.ones(S, dtype=C, device=dev)
    lam, E = _eig_raw(T)
    E = E / xp.linalg.vector_norm(E, axis=-2)[..., None, :]
    lam, E = _dead_last(xp, lam, E, mu, ndead)

    def best(E):
        Xs, conds, oks, masks = [], [], [], []
        for j, Jidx, Kidx, Jmask, order in _split_candidates(k):
            nc = Jidx.shape[0]
            Ji = xp.asarray(Jidx.reshape(-1), device=dev)
            EJ = xp.permute_dims(xp.reshape(xp.take(E, Ji, axis=2), (S, k, nc, j)), (0, 2, 1, 3))
            Qc, _ = xp.linalg.qr(EJ, mode='complete')
            Cc = Qc[..., j:]
            lq, Yq = _eig_raw(_conj_t(xp, Cc) @ T[:, None] @ Cc)
            Yq = Yq / xp.linalg.vector_norm(Yq, axis=-2)[..., None, :]
            # The padded lanes are K's last whenever J holds only members.
            _, Yq = _dead_last(xp, lq, Yq, mu[:, None], xp.broadcast_to(ndead[:, None], (S, nc)))
            # P[c, p, l] = 1 where position p of [J | K] is lane l: back to lane order, exactly.
            P = xp.astype(xp.asarray(order, device=dev)[:, :, None] == lane[None, None, :], C)
            X = xp.concat([EJ, Cc @ Yq], axis=-1) @ P[None]
            X = X / xp.linalg.vector_norm(X, axis=-2)[..., None, :]
            X = xp.where(vv[:, None], X, xp.broadcast_to(eye, X.shape))
            sv = xp.linalg.svdvals(X)
            pos = sv[..., -1] > 0
            conds.append(xp.where(pos, sv[..., 0] / xp.where(pos, sv[..., -1],
                                                              xp.ones_like(sv[..., -1])),
                                  xp.full_like(sv[..., 0], float('inf'))))
            vJ = xp.reshape(xp.take(valid, Ji, axis=1), (S, nc, j))
            vK = xp.reshape(xp.take(valid, xp.asarray(Kidx.reshape(-1), device=dev), axis=1),
                            (S, nc, k - j))
            oks.append(xp.all(vJ, axis=-1) & xp.any(vK, axis=-1))
            Xs.append(X)
            masks.append(Jmask)
        X = xp.concat(Xs, axis=1)                                     # (S, ncand, k, k)
        cond = xp.concat(conds, axis=1)
        ok = xp.concat(oks, axis=1) & xp.isfinite(cond)
        Jm = xp.asarray(np.concatenate(masks, axis=0), device=dev)    # (ncand, k)
        ncand = Jm.shape[0]
        pick_c = xp.argmin(xp.where(ok, cond, xp.full_like(cond, float('inf'))), axis=1)
        pick = xp.arange(S, device=dev) * ncand + pick_c
        has = xp.any(ok, axis=1)
        Xb = xp.take(xp.reshape(X, (S * ncand, k, k)), pick, axis=0)
        Xb = xp.where(has[:, None, None], Xb, xp.broadcast_to(eye, Xb.shape))
        cb = xp.take(xp.reshape(cond, (-1,)), pick)
        cb = xp.where(has, cb, xp.full_like(cb, float('inf')))
        J = xp.take(Jm, pick_c, axis=0) & valid & has[:, None]
        return Xb, J, cb

    def miss(X, J, cond):
        # What the check reads of a split: cond(X) times the part of X^-1 T X it drops.
        Mx = linear_solve(xp, X, T @ X)
        JK = J[:, :, None] & (~J & valid)[:, None, :]
        drop = vv & ~(JK | (eye != 0))
        dropped = xp.linalg.matrix_norm(xp.where(drop, Mx, xp.zeros_like(Mx)))
        finite = xp.isfinite(cond)
        return xp.where(finite, xp.where(finite, cond, xp.zeros_like(cond)) * dropped,
                        xp.full_like(cond, float('inf')))

    X, J, cond = best(E)
    again = cond > _SPLIT_RETRY
    if _concrete(xp.any(again)) is False:
        return X, J, cond
    live = lane[None, :] < (k - ndead)[:, None]
    X2, J2, cond2 = best(_spread_eigenvectors(xp, T, lam, E, live))
    take = again & (miss(X2, J2, cond2) < miss(X, J, cond))
    return (xp.where(take[:, None, None], X2, X), xp.where(take[:, None], J2, J),
            xp.where(take, cond2, cond))


def _rebuild_defective_pairs(xp, A, Omega_R, W, V, q, w, pairs, clusters=None):
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

    ``clusters`` (:func:`_defective_clusters`' answer, optional) hands the layer's defective
    clusters to :func:`_rebuild_defective_clusters`, which rebuilds each one jointly and the pairs
    outside them as here; without it this function is the pairs' rebuild alone.
    """
    if clusters is not None:
        return _rebuild_defective_clusters(xp, A, Omega_R, W, V, q, w, pairs, clusters)
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


#: The joint rebuild's check (:func:`_rebuild_defective_clusters`) reads the corrected
#: subspace's residual ``||A Q1 - Q1 T||_F`` against the larger of :data:`_EP_BACKWARD` at the
#: cluster's scale and this many units of ``eps ||A||_F``, the rounding a product with ``A``
#: leaves: the residual grows with the truncation through the evanescent orders while the
#: misses that matter do not.
_EP_RESIDUAL = 8.0

#: The joint rebuild's second step is kept for a cluster only where it leaves at most this
#: fraction of the one-step basis's miss of the check -- the larger of its subspace residual and
#: its split's dropped part times ``cond(X)``, each over its threshold
#: (:func:`_rebuild_defective_clusters`).
_EP_STEP_GAIN = 0.5

_JOINT_IMPRECISE_2D = (
    'a 2D layer holds a defective cluster of exceptional points (such as two order rows of a '
    'y-invariant layer at or near the same x-cutoff) whose joint rebuild failed its own check: '
    'the rebuilt modes miss the layer operator by more than rounding, and the answer can be '
    'inaccurate. Move the angle or wavelength by more than a relative 1e-4 to check it.')


def _joint_cluster_bases(xp, A, W0, Q0, Mc, Gc, members, cvalid, lam, is_a, m01_row, partner, F,
                         pair_terms=None):
    """Each defective cluster's corrected basis, split and check readings; values only.

    The part of the joint rebuild the two formulations share: :func:`_rebuild_defective_clusters`
    on the 2D path, :func:`~wavebit.rcwa._engine.solve_2d_first_order._regular_clusters` on the
    first-order one. ``A`` is the layer operator, ``Q0`` ``(S, n, k)`` each cluster's orthonormal
    start (padded lanes zero), ``W0`` the layer's basis with those columns on the members,
    ``Mc = Q0^H A Q0`` and ``Gc`` ``(n, S k)`` the clusters' residual ``A Q0 - Q0 Mc`` expanded in
    ``W0``. Off the clusters ``W0^-1 A W0`` is block diagonal: ``lam`` on the diagonal, and a row
    ``is_a`` coupled to its ``partner`` row by ``m01_row`` (a pair's upper-triangular block).
    ``pair_terms(HZ, Yc)``, where given, adds the pairs' own residual expansion to the chord
    step's residual: on the 2D path the pairs' columns in ``W0`` are corrected in the same step.
    ``F`` is the stand-in eigenvalue of padded lanes.

    One Newton step for each cluster's invariant subspace and a chord step after it, kept where
    it at least halves the check's miss (:data:`_EP_STEP_GAIN`), then :func:`_cluster_split`:
    the method :func:`_rebuild_defective_clusters` describes. Returns ``(Q1, X, Mx, dg, JK, res1,
    tau, tau_res, cond, dropped)`` for the basis each cluster keeps: the corrected orthonormal
    basis, the split ``X`` and ``X^-1 T X``, its diagonal, the coupled ``J x K`` entries, and the
    check's subspace residual, thresholds, ``cond(X)`` and dropped part.
    """
    n2 = W0.shape[0]
    C = W0.dtype
    dev = getattr(W0, 'device', None)
    idx = xp.arange(n2, device=dev)
    S, k = members.shape
    one = xp.ones_like
    lane = xp.arange(k, device=dev)
    eyek = xp.astype(lane[:, None] == lane[None, :], C)
    mflat = xp.reshape(members, (-1,))
    Oc = (idx[:, None] == mflat[None, :]) & xp.reshape(cvalid, (-1,))[None, :]   # (n2, S k)
    is_c = xp.any(Oc, axis=1)
    pc = xp.argmax(xp.astype(Oc, xp.int32), axis=1)                    # a member's flat lane
    slot_of = pc // k
    vv = cvalid[:, :, None] & cvalid[:, None, :]

    def padded(M, pad):
        return xp.where(vv, M, xp.where(eyek != 0, pad * one(M), xp.zeros_like(M)))

    def _correct(Gc):
        # A cluster's correction from the expansion Gc of its residual. Plain rows
        # y_i (lambda_i - M_c) = -g_i (a pair's `a` row also takes -m01 y_b); member rows below.
        Gs = xp.permute_dims(xp.reshape(Gc, (n2, S, k)), (1, 0, 2))               # (S, n2, k)
        Mrow = lam[None, :, None, None] * eyek - padded(Mc, -F)[:, None]          # (S, n2, k, k)
        Mrow = xp.matrix_transpose(xp.where(is_c[None, :, None, None], eyek * one(Mrow), Mrow))
        Yc = linear_solve(xp, Mrow, -Gs[..., None])[..., 0]
        rhs_a = -Gs - m01_row[None, :, None] * xp.take(Yc, partner, axis=1)
        Yc = xp.where(is_a[None, :, None], linear_solve(xp, Mrow, rhs_a[..., None])[..., 0], Yc)
        del Mrow, rhs_a
        if S > 1:
            # Another cluster's rows: M_r Y - Y M_s = -G, k*k unknowns (vec column-major), with
            # stand-ins of opposite sign on padded lanes so that no pivot is zero.
            Mr, Ms = padded(Mc, F), padded(Mc, -F)
            Kt = (eyek[None, None, :, None, :, None] * Mr[:, None, None, :, None, :]
                  - xp.permute_dims(Ms, (0, 2, 1))[None, :, :, None, :, None]
                  * eyek[None, None, None, :, None, :])
            Kt = xp.reshape(Kt, (S, S, k * k, k * k))
            GG = xp.reshape(xp.take(Gc, mflat, axis=0), (S, k, S, k))             # [r, i, s, j]
            rhs = -xp.reshape(xp.permute_dims(GG, (0, 2, 3, 1)), (S, S, k * k))
            rv = xp.reshape(xp.broadcast_to(cvalid[:, None, None, :], (S, S, k, k)), (S, S, k * k))
            same = xp.arange(S, device=dev)[:, None] == xp.arange(S, device=dev)[None, :]
            lane2 = xp.arange(k * k, device=dev)
            Kt = xp.where(same[:, :, None, None],
                          xp.astype(lane2[:, None] == lane2[None, :], C) * one(Kt), Kt)
            rhs = xp.where(rv & ~same[:, :, None], rhs, xp.zeros_like(rhs))
            sol = linear_solve(xp, Kt, rhs[..., None])[..., 0]
            Yrs = xp.reshape(xp.permute_dims(xp.reshape(sol, (S, S, k, k)), (0, 3, 1, 2)),
                             (S * k, S, k))
            Yc = xp.where(is_c[None, :, None], xp.permute_dims(xp.take(Yrs, pc, axis=0), (1, 0, 2)),
                          Yc)
            del Kt, GG, rhs, sol, Yrs
        ownc = is_c[None, :] & (slot_of[None, :] == xp.arange(S, device=dev)[:, None])
        Yc = xp.where(ownc[:, :, None] | ~cvalid[:, None, :], xp.zeros_like(Yc), Yc)
        return Yc

    Yc = _correct(Gc)
    # A second step, in W0's coordinates and with no new solve. There a cluster's subspace is
    # E_c + Y and H = W0^-1 A W0 is known from the first step: diagonal on the plain columns, the
    # pairs' and clusters' blocks plus their residual expansions G and Gc on theirs. The residual
    # H (E_c + Y) - (E_c + Y) T, T its own rows, takes the first step's correction again (the
    # chord method, Kelley 1995, ch. 5): one step leaves a remainder of second order in the start's
    # error over the gap, which reached the check's threshold where eig's start was poor or
    # another block close; this one leaves third order. Its residual is read from the first
    # step's expansions, so where the first step is already at rounding it only moves the basis
    # by rounding: below, each cluster takes it only where it at least halves the check's miss.
    ownc = is_c[None, :] & (slot_of[None, :] == xp.arange(S, device=dev)[:, None])
    Ecs = xp.astype(ownc[:, :, None] & ((pc - slot_of * k)[None, :, None] == lane[None, None, :])
                    & cvalid[:, None, :], C)                                  # (S, n2, k)
    HZ = Ecs @ Mc + xp.permute_dims(xp.reshape(Gc, (n2, S, k)), (1, 0, 2))
    HZ = HZ + lam[None, :, None] * Yc + xp.where(
        is_a[None, :, None], m01_row[None, :, None] * xp.take(Yc, partner, axis=1), xp.zeros_like(Yc))
    MY = xp.reshape(Mc[None] @ xp.reshape(xp.take(Yc, mflat, axis=1), (S, S, k, k)), (S, S * k, k))
    HZ = xp.where(is_c[None, :, None], HZ - lam[None, :, None] * Yc + xp.take(MY, pc, axis=1), HZ)
    if pair_terms is not None:
        HZ = pair_terms(HZ, Yc)
    HZ = HZ + Gc @ (xp.astype(xp.reshape(cvalid, (-1,)), C)[None, :, None] * xp.take(Yc, mflat, axis=1))
    F2 = HZ - (Ecs + Yc) @ (xp.matrix_transpose(Ecs) @ HZ)
    Yc2 = Yc + _correct(xp.reshape(xp.permute_dims(F2, (1, 0, 2)), (n2, S * k)))
    del Ecs, HZ, MY, F2
    # The clusters: the corrected bases, their restrictions, the disjoint split, and how far each
    # misses the check below (under 1 it passes). Each cluster keeps the basis after two steps
    # only where its miss is at most half the one-step basis's (sufficient decrease, as a line
    # search asks of a step: Kelley 1995, ch. 8); elsewhere the one-step basis stays, bit for bit.
    # The miss reads the split as well as the residual: at an exact coincidence eig(T)'s
    # eigenvectors are sensitive to rounding in T, and a basis with a smaller residual once gave
    # the split a cond(X) of 156 where the one-step basis gave 3.4.
    rdt = xp.real(Q0).dtype
    eps = float(xp.finfo(rdt).eps)
    norm_A = xp.linalg.matrix_norm(A)

    def corrected(Z):
        Q1, _ = xp.linalg.qr(Z)
        Q1 = xp.where(cvalid[:, None, :], Q1, Q0)
        AQ1 = A @ Q1
        T = xp.where(vv, _conj_t(xp, Q1) @ AQ1, xp.zeros_like(vv, dtype=C))
        res1 = xp.linalg.matrix_norm(xp.where(cvalid[:, None, :], AQ1 - Q1 @ T, xp.zeros_like(AQ1)))
        del AQ1
        T = padded(T, F)
        X, J, cond = _cluster_split(xp, T, cvalid, F)
        Mx = linear_solve(xp, X, T @ X)
        dg = xp.linalg.diagonal(Mx)
        JK = J[:, :, None] & (~J & cvalid)[:, None, :]
        dropped = xp.linalg.matrix_norm(xp.where(vv & ~(JK | (eyek != 0)), Mx, xp.zeros_like(Mx)))
        # The check's thresholds, at the cluster's scale.
        nvalid = xp.sum(xp.astype(cvalid, rdt), axis=1)
        lam_c = (xp.sum(xp.where(cvalid, dg, xp.zeros_like(dg)), axis=1)
                 / xp.maximum(nvalid, xp.ones_like(nvalid)))
        tau = (_EP_BACKWARD * (eps / 2.220446049250313e-16)
               * xp.maximum(xp.ones_like(xp.abs(lam_c)), xp.abs(lam_c)))
        tau_res = xp.maximum(tau, _EP_RESIDUAL * eps * norm_A * xp.ones_like(tau))
        finite = xp.isfinite(cond)
        miss = xp.maximum(res1 / tau_res, xp.where(finite, cond, xp.zeros_like(cond)) * dropped / tau)
        miss = xp.where(finite, miss, xp.full_like(miss, float('inf')))
        return (Q1, X, Mx, dg, JK, res1, tau, tau_res, cond, dropped), miss

    first, miss1 = corrected(Q0 + W0 @ Yc)
    second, miss2 = corrected(Q0 + W0 @ Yc2)
    two = miss2 <= _EP_STEP_GAIN * miss1
    del Yc, Yc2

    def chosen(x1, x2):
        return xp.where(xp.reshape(two, (S,) + (1,) * (len(x1.shape) - 1)), x2, x1)

    Q1, X, Mx, dg, JK, res1, tau, tau_res, cond, dropped = (
        chosen(x1, x2) for x1, x2 in zip(first, second))
    del first, second, miss1, miss2
    return Q1, X, Mx, dg, JK, res1, tau, tau_res, cond, dropped


def _rebuild_defective_clusters(xp, A, Omega_R, W, V, q, w, pairs, clusters):
    """:func:`_rebuild_defective_pairs` for a layer holding defective clusters; values only.

    ``clusters`` is :func:`_defective_clusters`' answer. Each cluster is rebuilt jointly, and
    every pair outside the clusters as :func:`_rebuild_defective_pairs` rebuilds it; a pair inside
    a cluster is switched off. Returns ``(W, V, Q, a, b, mask)`` in that function's shape, with
    ``(a, b, mask)`` an *entry list*: one lane per nonzero off-diagonal entry of ``Q``,
    ``Q[a_e, b_e]``, every entry listed once -- a pair's one entry, a cluster's block on its
    rows ``J`` and columns ``K``. The coupled rows and columns stay disjoint, so the exponential,
    the inverse and the derivative rules (:func:`_pair_sylvester`, :func:`_pair_inverse`), which
    sum over lanes, take it exactly as they take the pairs. Eagerly the list holds exactly the
    entries; under a trace every slot's ``k x k`` lanes, masked.

    **The subspace.** One Newton step for each cluster's invariant subspace, as for a pair
    (Stewart 1973; Demmel 1986), in the basis ``W0`` with every pair and every cluster
    orthonormalised. A cluster's restriction ``M_c = Q0^H A Q0`` is a dense ``k x k`` block, so
    its rows in another mode's correction solve ``k x k`` systems instead of dividing by a gap:
    a plain row ``y (lambda_i - M_c) = -g_i``, a pair's two rows the triangular Sylvester system
    with that pair's ``M``, another cluster's rows the Kronecker form of ``M_r Y - Y M_s = -G``
    (Golub and Van Loan, section 7.6.3), and the cluster's own rows zero. Two clusters, or a
    cluster and a pair, are at least :data:`_EP_CLUSTER` apart -- a cluster's ball is closed --
    so these systems are nonsingular; inside a cluster nothing is divided at all. The blocks are
    not normal, so one step leaves a remainder of second order in the start's error over the
    gap: it reached the check's threshold where ``eig``'s start was poor (smallest singular value
    2e-11) or another cluster 2e-4 away. A second step takes it out with no new solve, the chord
    method's (the first step's correction applied to the new residual, which leaves third order;
    C. T. Kelley, Iterative Methods for Linear and Nonlinear Equations, SIAM, 1995, ch. 5): in
    ``W0``'s coordinates the cluster's subspace is ``E_c + Y`` and ``H = W0^-1 A W0`` is known
    from the first step (diagonal on the plain columns; the pairs' and clusters' blocks plus their
    residual expansions on theirs), so the residual ``H (E_c + Y) - (E_c + Y) T`` takes the same
    correction again.

    **The modes.** With ``Q1`` the corrected orthonormal basis and ``T = Q1^H A Q1``,
    :func:`_cluster_split` gives ``X`` with ``X^-1 T X = L + B``: ``L`` diagonal and ``B`` on rows
    ``J``, columns ``K``, disjoint. Then ``q = diag(r) + B / (r_j + r_k)`` on ``J x K`` is the
    exact square root of ``L + B`` (``B**2 = 0``), every ``r = sqrt(L)`` taken on the side of
    the cluster member whose label decays most, as a pair's roots are. ``W = Q1 X`` and
    ``V = Omega_R W q^-1`` satisfy both first-order relations to the rounding of the split.

    **The check.** The corrected subspace's residual ``||A Q1 - Q1 T||_F`` (against
    :data:`_EP_RESIDUAL`) and the part of ``X^-1 T X`` the split drops, times ``cond(X)``,
    must stay below :data:`_EP_BACKWARD` at the cluster's scale. A start ``eig`` collapsed
    (a nearly rank-one basis for the cluster, beyond one Newton step) or a Jordan chain of length
    three (no disjoint split exists) fails it: eagerly an :class:`ExceptionalPointWarning`,
    under a trace a NaN answer, as an overflow of the trace's slots is.
    """
    members, cvalid, Q0, c_over = clusters
    n2 = W.shape[0]
    C = W.dtype
    dev = getattr(W, 'device', None)
    idx = xp.arange(n2, device=dev)
    S, k = members.shape
    one = xp.ones_like
    lane = xp.arange(k, device=dev)
    eyek = xp.astype(lane[:, None] == lane[None, :], C)
    mflat = xp.reshape(members, (-1,))
    Oc = (idx[:, None] == mflat[None, :]) & xp.reshape(cvalid, (-1,))[None, :]   # (n2, S k)
    is_c = xp.any(Oc, axis=1)
    pc = xp.argmax(xp.astype(Oc, xp.int32), axis=1)                    # a member's flat lane
    if pairs is not None:
        a, b, mask, v, x0, p_over = pairs
    else:
        # One discarded pair lane, so that the pairs' arithmetic keeps its shapes.
        a = b = xp.zeros(1, dtype=idx.dtype, device=dev)
        mask = xp.zeros(1, dtype=xp.bool, device=dev)
        e01 = xp.astype(xp.eye(n2, 2, device=dev), C)
        v, x0 = e01[:, :1], e01[:, 1:]
        p_over = None
    pmask = mask & ~xp.take(is_c, a) & ~xp.take(is_c, b)
    npair = a.shape[0]
    Oa = _one_hot(xp, n2, a, pmask, C, dev)
    Ob = _one_hot(xp, n2, b, pmask, C, dev)
    is_a = xp.any(Oa != 0, axis=1)
    is_b = xp.any(Ob != 0, axis=1)
    pa = xp.argmax(xp.astype(Oa != 0, xp.int32), axis=1)
    pb = xp.argmax(xp.astype(Ob != 0, xp.int32), axis=1)

    def place(col_a, col_b, base):
        return xp.where(is_a[None, :], xp.take(col_a, pa, axis=1),
                        xp.where(is_b[None, :], xp.take(col_b, pb, axis=1), base))

    def place_c(cols, base):                     # (S, n, k) -> the members' columns
        flat = xp.reshape(xp.permute_dims(cols, (1, 0, 2)), (cols.shape[1], S * k))
        return xp.where(is_c[None, :], xp.take(flat, pc, axis=1), base)

    vv = cvalid[:, :, None] & cvalid[:, None, :]
    # A stand-in eigenvalue for padded lanes, far from every mode's.
    F = 10.0 * (1.0 + xp.max(xp.abs(w)))

    def padded(M, pad):
        return xp.where(vv, M, xp.where(eyek != 0, pad * one(M), xp.zeros_like(M)))

    W0 = place_c(Q0, place(v, x0, W))
    Av, Ax = A @ v, A @ x0
    m00 = xp.sum(xp.conj(v) * Av, axis=0)
    m01 = xp.sum(xp.conj(v) * Ax, axis=0)
    m11 = xp.sum(xp.conj(x0) * Ax, axis=0)
    R = xp.concat([Av - v * m00[None, :], Ax - v * m01[None, :] - x0 * m11[None, :]], axis=1)
    AQ0 = A @ Q0
    Mc = xp.where(vv, _conj_t(xp, Q0) @ AQ0, xp.zeros_like(vv, dtype=C))
    Rc = xp.where(cvalid[:, None, :], AQ0 - Q0 @ Mc, xp.zeros_like(AQ0))
    del AQ0
    eye = xp.astype(xp.eye(n2, device=dev), C)
    W0s = xp.where(xp.any(pmask) | xp.any(cvalid), W0, eye)
    G = linear_solve(xp, W0s, R)
    # The clusters' residual takes a solve of its own, so the pairs' solve keeps its right-hand
    # sides: a solve's blocking, and with it its rounding, may depend on how many there are.
    Gc = linear_solve(xp, W0s, xp.reshape(xp.permute_dims(Rc, (1, 0, 2)), (n2, S * k)))
    del W0s, Rc
    Ca, Cb = G[:, :npair], G[:, npair:]
    lam = xp.where(is_a, xp.take(m00, pa), xp.where(is_b, xp.take(m11, pb), w))
    own = ((idx[:, None] == a[None, :]) | (idx[:, None] == b[None, :]))
    off = own | ~pmask[None, :]                  # own rows, and every row of a discarded lane
    den0 = xp.where(off, one(Ca), lam[:, None] - m00[None, :])
    den1 = xp.where(off, one(Ca), lam[:, None] - m11[None, :])
    Y0 = -Ca / den0
    Y1 = (-Cb + Y0 * m01[None, :]) / den1
    m01_row = xp.where(is_a, xp.take(m01, pa), xp.zeros_like(w))
    partner = xp.where(is_a, xp.take(b, pa), idx)
    Y0 = xp.where(is_a[:, None], (-Ca - m01_row[:, None] * xp.take(Y0, partner, axis=0)) / den0, Y0)
    Y1 = xp.where(is_a[:, None], (-Cb - m01_row[:, None] * xp.take(Y1, partner, axis=0)
                                  + Y0 * m01[None, :]) / den1, Y1)
    # A cluster's rows in a pair's correction: M_c Y - Y M_p = -C there, M_p the pair's
    # upper-triangular restriction -- two k x k solves per pair and cluster.
    Mp = padded(Mc, F)
    rows_c = cvalid[:, :, None]
    Ca_c = xp.reshape(xp.take(Ca, mflat, axis=0), (S, k, npair))
    Cb_c = xp.reshape(xp.take(Cb, mflat, axis=0), (S, k, npair))
    Ca_c = xp.where(rows_c, Ca_c, xp.zeros_like(Ca_c))
    Cb_c = xp.where(rows_c, Cb_c, xp.zeros_like(Cb_c))
    live_p = pmask[None, :, None, None]
    M0 = xp.where(live_p, Mp[:, None] - m00[None, :, None, None] * eyek, eyek * one(Mp[:, None]))
    M1 = xp.where(live_p, Mp[:, None] - m11[None, :, None, None] * eyek, eyek * one(Mp[:, None]))
    y0c = linear_solve(xp, M0, -xp.permute_dims(Ca_c, (0, 2, 1))[..., None])[..., 0]   # (S, P, k)
    y1c = linear_solve(xp, M1, (-xp.permute_dims(Cb_c, (0, 2, 1))
                               + m01[None, :, None] * y0c)[..., None])[..., 0]
    Y0 = xp.where(is_c[:, None], xp.take(xp.reshape(xp.permute_dims(y0c, (0, 2, 1)),
                                                    (S * k, npair)), pc, axis=0), Y0)
    Y1 = xp.where(is_c[:, None], xp.take(xp.reshape(xp.permute_dims(y1c, (0, 2, 1)),
                                                    (S * k, npair)), pc, axis=0), Y1)
    zero = xp.zeros_like(Y0)
    Y0, Y1 = xp.where(own, zero, Y0), xp.where(own, zero, Y1)
    pm = xp.astype(pmask, C)[None, :, None]

    def pair_terms(HZ, Yc):
        # The pairs' own residual expansion: their columns in W0 are corrected in this step too.
        return HZ + Ca @ (pm * xp.take(Yc, a, axis=1)) + Cb @ (pm * xp.take(Yc, b, axis=1))

    # The pairs, as in _rebuild_defective_pairs.
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
    ref = xp.where(xp.real(qa) >= xp.real(qb), qa, qb)
    alpha = xp.where(xp.real(xp.conj(ref) * ra) >= 0, ra, -ra)
    beta = xp.where(xp.real(xp.conj(ref) * rb) >= 0, rb, -rb)
    alpha = xp.where(pmask, alpha, one(alpha))
    beta = xp.where(pmask, beta, one(beta))
    t = xp.where(pmask, T01 / (alpha + beta), xp.zeros_like(T01))
    Va = (Omega_R @ v1) / alpha[None, :]
    Vb = (Omega_R @ x1 - Va * t[None, :]) / beta[None, :]
    Q1, X, Mx, dg, JK, res1, tau, tau_res, cond, dropped = _joint_cluster_bases(
        xp, A, W0, Q0, Mc, Gc, members, cvalid, lam, is_a, m01_row, partner, F, pair_terms)
    del W0, Gc
    # One side for the cluster: the root nearer the label of its most decaying member.
    qm = xp.reshape(xp.take(q, mflat), (S, k))
    key = xp.where(cvalid, xp.real(qm), xp.full_like(xp.real(qm), -float('inf')))
    cref = xp.sum(xp.astype(lane[None, :] == xp.argmax(key, axis=1)[:, None], C) * qm, axis=1)
    r = xp.sqrt(dg)
    r = xp.where(xp.real(xp.conj(cref)[:, None] * r) >= 0, r, -r)
    r = xp.where(cvalid, r, one(r))
    total = r[:, :, None] + r[:, None, :]
    N = xp.where(JK, Mx / xp.where(JK, total, one(total)), xp.zeros_like(Mx))
    Wcl = Q1 @ X
    Dinv = 1 / r
    Vcl = (Omega_R @ Wcl) @ (eyek * Dinv[:, None, :] - Dinv[:, :, None] * N * Dinv[:, None, :])
    W = place_c(Wcl, place(v1, x1, W))
    V = place_c(Vcl, place(Va, Vb, V))
    qd = xp.where(is_a, xp.take(alpha, pa), xp.where(is_b, xp.take(beta, pb), q))
    qd = xp.where(is_c, xp.take(xp.reshape(r, (-1,)), pc), qd)
    Q = _diag(xp, qd) + Oa @ (t[:, None] * xp.matrix_transpose(Ob))
    # The clusters' blocks by placement, never by a sum: Q stays exactly zero off each J x K.
    blk = xp.arange(S * k, device=dev) // k
    JKf = (xp.reshape(xp.broadcast_to(JK[:, :, None, :], (S, k, S, k)), (S * k, S * k))
           & (blk[:, None] == blk[None, :]))
    Nf = xp.reshape(xp.broadcast_to(N[:, :, None, :], (S, k, S, k)), (S * k, S * k))
    Ocf = xp.astype(Oc, C)
    OcT = xp.matrix_transpose(Ocf)
    pos = (Ocf @ xp.astype(JKf, C) @ OcT) != 0
    Q = xp.where(pos, Ocf @ xp.where(JKf, Nf, xp.zeros_like(Nf)) @ OcT, Q)
    # The check, at each cluster's scale (its thresholds from the kept basis).
    finite = xp.isfinite(cond)
    bad = xp.any(cvalid, axis=1) & (~finite | (res1 > tau_res)
                                    | (xp.where(finite, cond, xp.zeros_like(cond)) * dropped > tau))
    if c_over is None:
        if _concrete(xp.any(bad)):
            import warnings
            from wavebit._common.backend import ExceptionalPointWarning
            warnings.warn(_JOINT_IMPRECISE_2D, ExceptionalPointWarning, stacklevel=5)
        overflow = p_over
    else:
        overflow = c_over | xp.any(bad)
        if p_over is not None:
            overflow = overflow | p_over
    if overflow is not None:
        # More clusters or pairs than the trace's slots, or a failed check: loud, never silent.
        Q = Q * xp.where(overflow, float('nan'), 1.0)
    members = xp.astype(members, a.dtype)
    a_e = xp.concat([a, xp.reshape(xp.broadcast_to(members[:, :, None], (S, k, k)), (-1,))])
    b_e = xp.concat([b, xp.reshape(xp.broadcast_to(members[:, None, :], (S, k, k)), (-1,))])
    mask_e = xp.concat([pmask, xp.reshape(JK, (-1,))])
    if c_over is None:
        keep = xp.nonzero(mask_e)[0]
        a_e, b_e, mask_e = xp.take(a_e, keep), xp.take(b_e, keep), xp.take(mask_e, keep)
    return W, V, Q, a_e, b_e, mask_e


def _pair_coupling(xp, Q, a, b, mask):
    """``t_p = Q[a_p, b_p]`` (zero where unmasked) and the two one-hot maps, by gathers.

    Here and in the two rules below, ``(a, b, mask)`` lists the coupling entry by entry: a pair's
    one entry, or a jointly rebuilt cluster's ``J x K`` block one lane per entry
    (:func:`_rebuild_defective_clusters`), every nonzero entry once; a row may appear in several
    lanes, and the coupled rows and columns are disjoint. Every product below sums over lanes,
    so it is exact for such a list."""
    n2 = Q.shape[0]
    dev = getattr(Q, 'device', None)
    Oa = _one_hot(xp, n2, a, mask, Q.dtype, dev)
    Ob = _one_hot(xp, n2, b, mask, Q.dtype, dev)
    t = xp.sum(xp.take(Q, a, axis=0) * xp.matrix_transpose(Ob), axis=1)
    return t, Oa, Ob


def _pair_sylvester(xp, T, S, a, b, mask):
    """``X`` with ``T X + X T = S`` for ``T = diag(t) + coupling`` on the pairs ``(a_p, b_p)``.

    :func:`~wavebit.rcwa._engine.solve_1d._coupled_sylvester`'s four divided-difference passes, with the
    coupling's products taken by gathers (``O(n**2)`` per entry) rather than dense products: the
    coupling has a few entries (:func:`_pair_coupling`). With no coupling every pass is
    ``S / (t_i + t_j)``, the arithmetic :func:`~wavebit.rcwa._engine.modal_gradient.generator`
    forms.
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
    ``Eoff`` its off-diagonal part -- by gathers, ``M`` listed entry by entry. With no coupling
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


def _eig_or_diagonal(xp, A):
    """``eig(A)``, or ``A``'s own diagonal where ``eig`` fails on a matrix diagonal to round-off.

    A uniform isotropic layer's ``Omega2`` is ``diag(kx**2 + ky**2 - eps*mu)`` twice over in
    exact arithmetic; computed, its off-diagonal blocks hold the ``kx*ky - ky*kx`` cancellation,
    a few ulps scattered over a matrix of exactly repeated eigenvalues. LAPACK's ``geev`` can fail
    to converge on that (measured: ``info > 0`` on a 98 x 98 empty-interface operator, NumPy and
    SciPy alike) although the answer is analytic. Where it does, ``(diag(A), I)`` is the exact
    eigensystem of ``A`` less those round-off entries -- a nearby layer in the sense of
    ``docs/theory/rcwa/rayleigh-points.md`` section 1 -- so it is what the layer's plane-wave
    modes are, and the rest of the solve takes it as any computed basis. Only a failure is
    replaced: a converged ``eig`` keeps its bits, and a matrix off-diagonal beyond round-off
    still raises. JAX's ``eig`` reports the failure as NaN rather than raising; eagerly, a
    non-finite decomposition of a finite matrix is read as the same failure. Under a trace
    nothing can be read and the NaN goes on to the solve's finiteness check.
    """
    try:
        w, W = eig(A)
    except Exception as exc:
        if not _did_not_converge(exc):
            raise
        return _diagonal_modes(xp, A, exc)
    if 'jax' in xp.__name__:
        import numpy as np
        try:
            failed = not (np.all(np.isfinite(np.asarray(w))) and np.all(np.isfinite(np.asarray(W))))
            finite = bool(np.all(np.isfinite(np.asarray(A))))
        except Exception:
            failed = False
        if failed and finite:
            return _diagonal_modes(xp, A, None, (w, W))
    return w, W


def _diagonal_modes(xp, A, exc, failed=None):
    """``(diag(A), I)`` for :func:`_eig_or_diagonal`; its failure again where ``A`` is coupled.

    ``exc`` is the exception ``eig`` raised (re-raised), or ``None`` with ``failed`` the
    non-finite decomposition JAX returned (handed back unchanged).
    """
    import numpy as np
    try:
        host = np.asarray(_readable(A, xp))
    except Exception:
        host = None
    if host is not None:
        d = np.diagonal(host)
        scale = np.max(np.abs(host)) if host.size else 0.
        off = np.max(np.abs(host - np.diag(d)), initial=0.)
        tolerance = 4 * host.shape[-1] * np.finfo(host.real.dtype).eps * scale
        if np.all(np.isfinite(host)) and off <= tolerance:
            device = getattr(A, 'device', None)
            return (xp.asarray(d, dtype=A.dtype, device=device),
                    xp.asarray(np.eye(host.shape[-1], dtype=host.dtype), dtype=A.dtype,
                               device=device))
    if exc is not None:
        raise exc
    return failed


def _did_not_converge(exc):
    """Whether ``exc`` is a backend's eigensolver reporting non-convergence."""
    from numpy.linalg import LinAlgError
    if isinstance(exc, LinAlgError):
        return True
    return isinstance(exc, RuntimeError) and 'linalg' in type(exc).__name__.lower()


def layer_modes_2d(xp, Kx, Ky, I, epx_conv, epy_conv, epz_conv_i, epxy_conv, epyx_conv, C,
                   mu=None, k0_d=None, dng=None, pair_cap=None, with_coupling=False, env=None,
                   screen_k0_d=None):
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
    trace, and ``pair_cap // 2`` the defective clusters rebuilt jointly; callers with an order
    grid pass its larger axis count. An overflow poisons the generator with NaN. Eagerly every
    eligible pair and cluster is rebuilt. ``screen_k0_d`` is the
    optical thickness the eager screen of nearly coincident pairs reads
    (:func:`_crowded_pairs_imprecise`) and nothing else; it defaults to ``k0_d``, so SMM, which
    takes no labels from a thickness, passes it on its own. ``with_coupling``
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
    w, W = _eig_or_diagonal(xp, _stop(Omega2_LR))
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
    built = []

    def _operator():
        # Formed once, on first need: released above to keep the eager peak, and again below.
        if not built:
            if differentiated:
                built.append(_stop(Omega2_LR))
            else:
                B_, D_, Cc_, E_ = left_blocks()
                built.append(_block(xp, [[-Cc_, B_], [-D_, E_]]) @ Omega_R)
        return built[0]

    # Defective clusters of three to eight modes, rebuilt jointly (`_rebuild_defective_clusters`);
    # a trace holds `pair_cap // 2` of them, with their masks as data.
    clusters = _defective_clusters(xp, _stop(w), _stop(W), _stop(q), pairs, _operator,
                                   max(1, pair_cap // 2))

    def _screen(W_final, operator=_operator, rebuilt=None, joint=None):
        # Eagerly loud: defective clusters no rebuild took, and nearly coincident pairs one
        # rebuild cannot separate. A jointly rebuilt cluster answers to its own check instead.
        # One warning per layer. Under an abstract trace both questions come back as data (True:
        # the layer answers NaN); eagerly None.
        import warnings
        from wavebit._common.backend import ExceptionalPointWarning
        group = None
        if joint is not None:
            # Each rebuilt member in a group of its own: never in a cluster again.
            ii = xp.arange(w.shape[0], device=getattr(W_final, 'device', None))
            group = xp.where(joint, ii + 1, xp.zeros_like(ii))
            a_, b_, mask_ = rebuilt[2:]
            rebuilt = rebuilt[:2] + (a_, b_, mask_ & ~xp.take(joint, a_) & ~xp.take(joint, b_))
        unresolved = _unresolved_defective_cluster(xp, _stop(w), _stop(W_final), operator,
                                                   _EP_SINE, group)
        if unresolved is None:
            loud = _unresolved_cluster_traced(xp, _stop(w), _stop(W_final), operator, _EP_SINE,
                                              group, max(1, pair_cap // 2))
            if rebuilt is not None:
                loud = loud | _crowded_pairs_traced(
                    xp, _stop(w), operator(), *(_stop(x) for x in rebuilt),
                    k0_d=k0_d if screen_k0_d is None else screen_k0_d,
                    slots=max(1, pair_cap // 2))
            return loud
        if unresolved:
            warnings.warn(_UNRESOLVED_2D, ExceptionalPointWarning, stacklevel=4)
        elif rebuilt is not None and _crowded_pairs_imprecise(
                xp, _stop(w), operator(), *(_stop(x) for x in rebuilt),
                k0_d=k0_d if screen_k0_d is None else screen_k0_d):
            warnings.warn(_IMPRECISE_2D, ExceptionalPointWarning, stacklevel=4)
        return None
    if pairs is None and clusters is None:
        # Eager only: under an abstract trace both gates return their slots.
        _screen(W)
        built.clear()
    else:
        A = _operator()
        if env is not None:
            env.Wi = None
        if clusters is None:
            W, V, Q, a, b, mask = _rebuild_defective_pairs(xp, A, _stop(Omega_R), W, V, q, w,
                                                           pairs)
            joint = None
        else:
            W, V, Q, a, b, mask = _rebuild_defective_pairs(
                xp, A, _stop(Omega_R), _stop(W), _stop(V), _stop(q), _stop(w), pairs,
                clusters=clusters)
            ii = xp.arange(w.shape[0], device=getattr(W, 'device', None))
            joint = xp.any((ii[:, None] == xp.reshape(clusters[0], (-1,))[None, :])
                           & xp.reshape(clusters[1], (-1,))[None, :], axis=1)
        loud = _screen(W, lambda: A, (W, Q, a, b, mask), joint)
        if loud is not None:
            # A selection, not a product, so that a quiet layer keeps its bits.
            Q = xp.where(loud, xp.full_like(Q, float('nan')), Q)
        del A
        built.clear()
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
