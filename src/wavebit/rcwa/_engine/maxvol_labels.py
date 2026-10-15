"""ETM layer labels chosen by maximum volume, from the state below the layer.

Which of a mode's two roots ``+-q`` the ETM calls ``+`` moves no exact answer, only the
conditioning of the recursion (``docs/theory/rcwa/connecting-algorithms.md`` sections 9-10). At a
real ``k_par`` in passive media the rule of :func:`~wavebit._common.conventions.layer_root`
decides it from the layer alone, and does it well. At a complex ``k_par`` a propagating mode's
decaying root can travel up, the rule's decision is blind to the medium the layer's waves land in,
and an etched grating came back O(1) wrong. This module decides from that medium instead.

A layer's ETM state at its top face is the transmitted subspace ``span C``,
``C = [X^-1 a; X b]`` (rows: the ``+`` and ``-`` amplitudes of each mode pair, in the decaying
labels; ``a``, ``b`` the ETM's interface factors at the layer's bottom face, ``X`` the decaying
exponentials). ETM writes it as ``[I; Y]`` with the ``+`` rows as pivots, and a label flip of
mode ``j`` swaps rows ``j`` and ``N + j``. Labels are chosen as a locally maximum-volume pivot set
under single and 2x2 principal swaps. Pair normalisation (both rows of pair ``j`` times ``X_j``)
keeps every volume ratio and forms no growing factor: ``C~ = [a; X**2 b]``. The start is each
pair's larger leverage in one Householder QR of ``C~``; swaps run while one improves the volume by
more than :data:`TAU`, at most :data:`ITERATIONS` of them (6 was the most any of 3,000 layer
decisions took), as ``where`` selections so a trace runs the same program.

V. Mehrmann and F. Poloni, "Doubling algorithms with permuted Lagrangian graph bases",
SIAM J. Matrix Anal. Appl. 33(3), 780-805 (2012), doi:10.1137/110850773 (permuted Lagrangian
graph bases; for a symmetric ``Y`` a pivot set no single or 2x2 principal swap improves has
``|Y_jj| <= 1`` and ``|Y_jk| <= sqrt(2)``).

The decision reads stopped values only, so a derivative flows through the modes it labels and
never through the choice; and it is taken only where ``k_par`` is complex
(:func:`~wavebit.observations.power.transverse_k_is_real`), so every real-angle solve keeps the
bits it had. A layer that holds a double negative keeps its own rules, and a layer inside a run of
identical layers inherits the decision of the run's bottom slice.
"""
from wavebit._common.backend import linear_solve

#: ``'complex'`` -- maximum-volume labels where an in-plane wavevector is complex, the rule
#: elsewhere; ``'off'`` -- the rule everywhere. A module constant for studies and for tests that
#: need the rule's behaviour at a complex angle, not a user setting.
MODE = 'complex'
#: A swap is taken while it multiplies the pivot volume by more than this.
TAU = 1.5
#: The most swaps a decision takes; a trace runs all of them, masked.
ITERATIONS = 12


def _concrete(value):
    """``bool(value)``, or ``None`` where the value is abstract (a JAX tracer)."""
    try:
        return bool(value)
    except Exception:
        return None


def enabled(xp, kx, ky=None):
    """Whether this solve takes maxvol labels: ``True``, ``False`` or a traced boolean.

    Complex ``k_par`` exactly where :func:`~wavebit.observations.power.transverse_k_is_real` says
    so, the test the complex-incidence warning and the NaN powers use: a real angle in an absorbing
    medium leaves ``imag(kx)`` at round-off, and keeps the rule.
    """
    if MODE == 'off':
        return False
    from wavebit.observations.power import transverse_k_is_real
    real = transverse_k_is_real(xp, kx, ky)
    readable = _concrete(real)
    return xp.logical_not(real) if readable is None else not readable


def _onehot(xp, idx, j, dtype):
    return xp.astype(idx == j, dtype)


def _ppt_xp(xp, Y, j, k, pair):
    """Principal pivot transform on ``{j}`` (``pair`` false) or ``{j, k}``, by masks: traceable."""
    n = Y.shape[0]
    idx = xp.arange(n)
    ej = _onehot(xp, idx, j, Y.dtype)
    ek = _onehot(xp, idx, k, Y.dtype)
    cj, ck = Y @ ej, Y @ ek                       # columns j, k
    rj, rk = ej @ Y, ek @ Y                       # rows j, k
    yjj, yjk, ykj, ykk = rj @ ej, rj @ ek, rk @ ej, rk @ ek
    rowj = (idx == j)[:, None]
    rowk = (idx == k)[:, None]
    colj = (idx == j)[None, :]
    colk = (idx == k)[None, :]
    # single
    y = xp.where(pair, xp.ones_like(yjj), yjj)
    S = Y - (cj[:, None] * rj[None, :]) / y
    S = xp.where(rowj, -rj[None, :] / y, S)
    S = xp.where(colj, cj[:, None] / y, S)
    S = xp.where(rowj & colj, 1 / y, S)
    # pair
    det = yjj * ykk - yjk * ykj
    det = xp.where(pair, det, xp.ones_like(det))
    b00, b01, b10, b11 = ykk / det, -yjk / det, -ykj / det, yjj / det
    G0 = cj * b00 + ck * b10                      # (E Binv)[:, 0]
    G1 = cj * b01 + ck * b11
    H0 = b00 * rj + b01 * rk                      # (Binv F)[0, :]
    H1 = b10 * rj + b11 * rk
    P = Y - (G0[:, None] * rj[None, :] + G1[:, None] * rk[None, :])
    P = xp.where(rowj, -H0[None, :], P)
    P = xp.where(rowk, -H1[None, :], P)
    P = xp.where(colj, G0[:, None], P)
    P = xp.where(colk, G1[:, None], P)
    P = xp.where(rowj & colj, b00, P)
    P = xp.where(rowj & colk, b01, P)
    P = xp.where(rowk & colj, b10, P)
    P = xp.where(rowk & colk, b11, P)
    return xp.where(pair, P, S)


def maxvol_flips_xp(xp, a, b, x2, tau=None, iterations=None):
    """Per mode pair, ``True`` where the pivot is the decaying labels' ``-`` row.

    ``a``, ``b``: the ETM factors at the layer's bottom face in decaying labels; ``x2``:
    ``exp(-2 k0 q d)`` of those labels, ``|x2| <= 1``. Fixed iterations and where-selection;
    eagerly it stops once no swap passes ``tau``. Returns ``(flip, ok)``; ``ok`` is false where
    the start block was singular or the swaps left a non-finite ``Y``, and the caller then keeps
    its rule.
    """
    tau = TAU if tau is None else tau
    iterations = ITERATIONS if iterations is None else iterations
    n = a.shape[0]
    top = a
    bot = x2[:, None] * b
    Q, _ = xp.linalg.qr(xp.concat([top, bot], axis=0))
    lev = xp.sum(xp.real(Q * xp.conj(Q)), axis=1)
    flip = lev[n:] > lev[:n]
    piv = xp.where(flip[:, None], bot, top)
    non = xp.where(flip[:, None], top, bot)
    try:
        Y = xp.matrix_transpose(linear_solve(xp, xp.matrix_transpose(piv),
                                             xp.matrix_transpose(non)))
    except Exception:                             # an exactly singular start block, eagerly
        return flip, False
    idx = xp.arange(n)
    eye = idx[:, None] == idx[None, :]
    zero = xp.zeros((), dtype=xp.real(Y).dtype)
    for _ in range(iterations):
        d = xp.linalg.diagonal(Y)
        single = xp.abs(d)
        minors = xp.where(eye, zero, xp.abs(d[:, None] * d[None, :] - Y * xp.matrix_transpose(Y)))
        j1 = xp.argmax(single)
        v1 = xp.max(single)
        flat = xp.reshape(minors, (-1,))
        i2 = xp.argmax(flat)
        v2 = xp.max(flat)
        pair = v2 > v1
        do = xp.maximum(v1, v2) > tau
        stop = _concrete(do)
        if stop is False:
            break
        j = xp.where(pair, i2 // n, j1)
        k = xp.where(pair, i2 % n, j1)
        Y = xp.where(do, _ppt_xp(xp, Y, j, k, pair), Y)
        toggle = (idx == j) | ((idx == k) & pair)
        flip = xp.where(do & toggle, ~flip, flip)
    ok = xp.all(xp.isfinite(xp.real(Y)) & xp.isfinite(xp.imag(Y)))
    return flip, ok


class Environment:
    """What a layer's labels may depend on: the ETM state below it and its run.

    ``F``, ``G``: the state the layer below left, in the s/p boundary frame (stopped).
    ``k0_d``: the run's optical thickness. ``inherited``/``join``: the flip mask of the slice
    below, and whether this layer continues its run (``True``/``False``/traced). ``use``: the
    solve's gate (``True`` or traced; never ``False`` -- then there is no environment).
    ``frame``: raw -> boundary basis, as the solve loop rotates it. After :meth:`choose`,
    ``flips`` holds this layer's mask, relative to the decaying labels.
    """

    def __init__(self, F, G, k0_d, inherited=None, join=False, use=True):
        self.F = F
        self.G = G
        self.k0_d = k0_d
        self.inherited = inherited
        self.join = join
        self.use = use
        self.flips = None
        self.frame = None
        self.Wi = None          # inverses of the decaying-label boundary matrices, for reuse
        self.Vi = None

    def choose(self, xp, w, C, boundary):
        """Labels for a layer whose eigenvalues are ``w`` (stopped).

        ``boundary(q)`` returns the stopped boundary-frame ``(W, V)`` of the layer under labels
        ``q``. Develop's rule (:func:`~wavebit._common.conventions.layer_root` on ``k0_d``) is
        what a traced gate selects against, and what a failed decision falls back to.
        """
        from wavebit._common.conventions import layer_root, mode_q
        from wavebit.rcwa._engine.modal_gradient import _stop
        w = xp.astype(w, C)
        q_dec = mode_q(xp, w)
        q_rule = layer_root(xp, w, self.k0_d, None)
        flip_rule = q_rule != q_dec
        if self.join is True and self.inherited is not None:
            flips = self.inherited
        else:
            bW, bV = (_stop(m) for m in boundary(q_dec))   # the frame rotation may carry theta's
            self.Wi = xp.linalg.inv(bW)
            self.Vi = xp.linalg.inv(bV)
            P = self.Wi @ self.F
            Qm = self.Vi @ self.G
            a = 0.5 * (P + Qm)
            b = 0.5 * (P - Qm)
            k0_d = self.k0_d
            if hasattr(k0_d, 'shape'):
                # a thickness may be a design variable; the decision reads its value only
                k0_d = xp.astype(_stop(k0_d), xp.real(q_dec).dtype)
            x2 = xp.exp(-2.0 * k0_d * q_dec)
            flips, ok = maxvol_flips_xp(xp, a, b, x2)
            c = _concrete(ok)
            if c is False:
                flips = flip_rule
            elif c is None:
                flips = xp.where(ok, flips, flip_rule)
            if self.inherited is not None and self.join is not False:
                flips = xp.where(self.join, self.inherited, flips)
        if self.use is not True:
            flips = xp.where(self.use, flips, flip_rule)
        self.flips = flips
        q = xp.where(flips, -q_dec, q_dec)
        if self.use is not True:
            # a traced gate that turned out false keeps the rule's bits exactly
            q = xp.where(self.use, q, q_rule)
        return q

    def inverses(self, xp, flip_negates_w=None):
        """``(W^-1, V^-1)`` of the chosen labels from the decision's, or ``None``.

        A flip negates one column of ``V`` (or of ``W``, where ``flip_negates_w`` says so -- the
        conical path's first family), so the inverses differ by row signs only.
        """
        if self.Wi is None or self.flips is None:
            return None
        one = xp.ones_like(xp.real(self.Wi[:, 0]))
        neg = xp.where(self.flips, -one, one)
        if flip_negates_w is None:
            return self.Wi, xp.astype(neg, self.Vi.dtype)[:, None] * self.Vi
        sw = xp.where(flip_negates_w, neg, one)
        sv = xp.where(flip_negates_w, one, neg)
        return (xp.astype(sw, self.Wi.dtype)[:, None] * self.Wi,
                xp.astype(sv, self.Vi.dtype)[:, None] * self.Vi)
