"""Single-source 1D (TE / TM / conical) RCWA solve — Phase 3 of the rewrite.

One backend-agnostic implementation of the transfer-matrix solve for a 1D grating stack,
written against the Array-API namespace (``numpy`` / ``jax.numpy`` / ``torch``). It is the
port of the triplicated ``transfer_1d_*`` chains into a single source: the only per-backend
primitive it reaches for is :func:`wavebit._core.backend.eig`, and every sign/branch choice
comes from :mod:`wavebit._core.conventions`.

``solve_1d`` handles the pure TE / TM (no azimuth) path; ``solve_1d_conical`` handles the
general 1D grating (arbitrary ``psi`` / ``phi``) with its coupled s/p block system.

These are private propagation-frame kernels. Historical ``top``/``bottom`` names mean
the start/end ports of the positive-z layer list. The public engine binds them to physical
``bot``/``top`` and maps public incidence sides; see ``CoreSolveMixin._kernel_incidence_side``.
Directional descriptions below refer to this local frame, not the public upward height.

Each kernel consumes per-layer convolution matrices and incidence/geometry values.
At the efficiency level it returns ``(de_ri, de_ti, power_entering)``: two per-order
efficiency arrays and a scalar net entering power fraction. JAX / PyTorch arrays
run the same algebra with autodiff intact.

Stages (matching the legacy chain, kept as one function each to make bit-parity auditable):
  1. boundary init      — kz for efficiency, substrate admittance G, F = T = I
  2. per-layer eig+couple— Redheffer/transfer propagation from the bottom layer up
  3. final matching      — incident field -> R, T -> diffraction efficiency
"""
from .backend import (array_namespace, as_array, check_finite, division_at_a_cutoff,
                      eig)
from .conventions import (boundary_kz, half_space_medium, incident_column, incident_kz,
                          layer_label_rule, layer_root, order_azimuth, order_frame_sign,
                          layer_runs, power_down_labels, run_k0_d)
from .enums import Answers, answers_of
from .power import (incident_power, net_power, power_for_real_k,
                    sp_incident_power, wave_power)


def _diag(xp, v):
    """Construct a diagonal matrix from a 1D vector (Array-API ``diag`` build form)."""
    return xp.diag(v)


def _is_backside(incidence_side):
    """Whether ``incidence_side`` selects backside illumination, refusing anything else.

    The check lives at the core boundary rather than only in the solver constructors because
    the solves are also called directly; a typo'd side must not quietly illuminate from the
    top.
    """
    if incidence_side not in ('top', 'bottom'):
        raise ValueError(
            f"incidence_side must be 'top' or 'bottom', got {incidence_side!r}")
    return incidence_side == 'bottom'


def _mirrored_scene(n_top, n_bot, thickness, *per_layer):
    """The z-mirrored scene: reversed layers between swapped half-spaces.

    Backside illumination is solved as *top-side* illumination of this mirror. That is exact,
    not approximate: ``z -> -z`` is a symmetry of Maxwell's equations, it leaves every layer's
    in-plane pattern untouched (a layer is z-invariant inside itself), and it turns "arrive
    from ``n_bot`` travelling ``-z``" into "arrive from the mirror's top travelling ``+z``" —
    the one problem the solve already handles. ``theta`` / ``phi`` / ``psi`` pass through
    unchanged because the contract states them against the incidence medium's normal, which
    the mirror maps onto the top normal; ``kx`` / ``ky`` must already be built with the
    incidence medium's index (see :attr:`~wavebit._core.engine.CoreSolveMixin.n_inc`).

    Reversal is by Python-level indexing rather than ``xp.flip`` so it works identically on
    arrays, lists of per-layer matrices, and traced values — the indices are static, so
    nothing here blocks ``jit`` or gradients. ``per_layer`` entries may be ``None``
    (``epxy_conv_all`` / ``epyx_conv_all``) and pass through as ``None``.
    """
    n = len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]
    order = list(range(int(n)))[::-1]
    rev = lambda a: [a[i] for i in order]
    return (n_bot, n_top, rev(thickness),
            *(None if stack is None else rev(stack) for stack in per_layer))


def _block(xp, rows):
    """Assemble a block matrix from a 2D list of blocks (Array-API stand-in for ``np.block``).

    ``rows`` is a list of block-rows; each block-row is a list of equal-height matrices that
    concatenate horizontally, and the rows then concatenate vertically.
    """
    return xp.concat([xp.concat(r, axis=1) for r in rows], axis=0)


def mode_flux(xp, X, Y):
    """Each mode's time-averaged Poynting ``z`` component, and its energy ``sum(|psi|**2)``.

    ``-Im(sum(X*conj(Y)))/2`` and ``sum(|X|**2) + sum(|Y|**2)`` over each column. The pairs
    are the s/p boundary variables a substrate seed is written in, with ``X`` the one that
    carries the admittance: ``(X, Y) = (H_s, E_s)`` for an s wave and ``(F_p, H_p)`` for a p
    wave -- a pure path's partner and primary, either polarization. A downward wave of a
    lossless half-space then comes out positive, ``Re(kz/mu)/2`` and ``Re(kz/eps)/2``. The
    variables are tangential fields, matched across every interface, so the same form is the
    flux of a layer mode. What :func:`~wavebit._core.conventions.power_down_labels` reads.
    """
    flux = -0.5 * xp.imag(xp.sum(X * xp.conj(Y), axis=0))
    energy = xp.sum(xp.abs(X) ** 2, axis=0) + xp.sum(xp.abs(Y) ** 2, axis=0)
    return flux, energy


def _labels(xp, w, C, k0_d, mirrored, patterned, flux_of):
    """A layer's roots: :func:`~wavebit._core.conventions.layer_root`, then, where the layer holds
    a double-negative region, :func:`~wavebit._core.conventions.power_down_labels`.

    ``flux_of(q)`` returns ``(flux, energy)`` of the modes under the labels ``q``. Values only:
    the labels are decided before :func:`~wavebit._core.modal_gradient.generator` sees them.
    """
    q = layer_root(xp, xp.astype(w, C), k0_d, mirrored)
    if patterned is not None:
        q, _ = power_down_labels(xp, q, *flux_of(q), k0_d, patterned)
    return q


def layer_modes_1d(xp, tetm_index, kx, I, epx_conv, epy_conv, epz_conv_i, C, mu=None, k0_d=None,
                   dng=None):
    """Eigenmodes ``(W, V, q)`` of one 1D TE / TM layer.

    ``W`` expands the continuous primary field (``E_y`` for TE, ``H_y`` for TM), ``V`` its
    partner, and ``q`` are the modal decay constants used as ``exp(-k0*q*d)``. Without
    ``k0_d`` the root keeps ``Re(q) >= 0``, so that exponential never grows.

    Shared by the ETM (:func:`solve_1d`) and SMM (:mod:`wavebit._core.scattering`) connecting
    algorithms — the mode algebra is a property of the layer, not of how layers are joined.
    Which of ``±q`` is labelled "down" is the one exception: ETM passes the layer's optical
    thickness ``k0_d`` for the labels its recursion needs, and SMM's cascade passes nothing
    (:func:`~wavebit._core.conventions.layer_root`) -- its field state, being the ETM's, is
    rebuilt with ``k0_d`` where the two differ.

    ``dng`` is the engine's word on the layer's materials
    (:class:`~wavebit._core.conventions.DoubleNegativeLayer`), or ``None`` when it said nothing.
    A uniform double negative is labelled by the mirror of the plain rule, as a half-space of
    the same medium is; a patterned layer holding one is labelled per mode by its power flux
    (:func:`~wavebit._core.conventions.layer_label_rule`). No other layer is touched, and no
    label moves an exact answer.

    ``mu`` is an optional ``(mux_conv, muy_conv, muz_conv_i)`` permeability triple in the
    permittivity's own convention — ``muz_conv_i`` inverted — or ``None`` for a non-magnetic
    layer, where every expression below is the one that was here and the solve is
    bit-identical. Only three slots, not the 2D path's five: this operator has no off-diagonal
    position to put ``mu_xy`` / ``mu_yx`` in, exactly as it has none for ``eps_xy`` / ``eps_yx``,
    and :func:`~wavebit._core.engine.CoreSolveMixin.solve_conv` refuses those by name rather
    than dropping them here.

    **The two branches are each other under the duality**
    ``(E, H, eps, mu) -> (H, -E, mu, eps)`` (``docs/magnetic-materials.md`` §1), which is what
    places every slot below rather than a second derivation. Eliminating the longitudinal
    component of each polarization through the ``zz`` slot of *its own* tensor::

        TE:  d2(E_y)/dz~2 = mu_x  (Kx muz_i Kx - eps_y) E_y      H_z through mu_zz
        TM:  d2(H_y)/dz~2 = eps_x (Kx epz_i Kx - mu_y ) H_y      E_z through eps_zz

    so each line is the other with every symbol replaced. At ``mu = 1`` the TE prefactor
    ``mu_x`` is the identity and drops out, which is why that branch had no left factor and
    no inverse in ``V``; the TM branch already carried both, and its ``- I`` was ``- mu_y``
    all along. **A magnetic TE layer therefore costs what a TM layer always cost** — one more
    matmul and one more inverse — and that is the whole arithmetic price of this feature.

    Getting the pair uncrossed is the mistake to watch for, the same one
    ``docs/anisotropic-materials.md`` §5 made for the 2D operator: it is ``mu_y`` that joins
    the TM branch and ``mu_x`` the TE one, and reading ``- I`` as "the layer's own mu" puts
    ``mu_x`` in the TM line. A scalar ``mu`` cannot tell the two apart.
    """
    mux_conv, muy_conv, muz_conv_i = (I, I, None) if mu is None else mu
    mirrored, patterned = layer_label_rule(
        xp, dng, (epx_conv, epy_conv, epz_conv_i),
        None if mu is None else (mux_conv, muy_conv, muz_conv_i))
    if tetm_index == 0:  # TE
        Kx = _diag(xp, kx)
        # `muz_conv_i` sits between the K factors exactly as `epz_conv_i` does in the TM
        # branch below. Skipped where the layer has no mu_zz rather than multiplying an
        # identity through, so the non-magnetic arithmetic stays the two products it was.
        KxKx = Kx @ Kx if muz_conv_i is None else Kx @ muz_conv_i @ Kx
        A = KxKx - epy_conv
        if mu is not None:
            A = mux_conv @ A
        from .modal_gradient import _stop, generator
        w, W = eig(_stop(A))
        # `V = W * q` is the mu = 1 form; the general one divides out the same `mu_x` the
        # operator multiplied in, which is what the TM branch does with `eps_x`. Branched
        # rather than inverting an identity: `inv(I) @ W` is bit-identical but costs an
        # O(n**3) inverse a non-magnetic solve never paid.
        base = W if mu is None else xp.linalg.inv(mux_conv) @ W
        q = generator(xp, A, w, W,
                      _labels(xp, w, C, k0_d, mirrored, patterned,
                              lambda q: mode_flux(xp, base * q, W)),
                      square_root=True)
        V = base @ q if q.ndim == 2 else base * q
    elif tetm_index == 1:  # TM
        # `- muy_conv` is `- I` to the bit when non-magnetic: the fallback above binds
        # `muy_conv` to the very `I` this line used to name.
        B = kx[:, None] * epz_conv_i * kx[None, :] - muy_conv
        from .modal_gradient import _stop, generator
        operator = epx_conv @ B
        w, W = eig(_stop(operator))
        base = xp.linalg.inv(epx_conv) @ W
        # TM's primary is `H_y` (the `H_p` of the s/p pair) and its partner the `F_p`, so the
        # partner leads, as it does for TE's `H_s`.
        q = generator(xp, operator, w, W,
                      _labels(xp, w, C, k0_d, mirrored, patterned,
                              lambda q: mode_flux(xp, base * q, W)),
                      square_root=True)
        V = base @ q if q.ndim == 2 else base * q
    else:
        raise ValueError(f'tetm_index must be 0 (TE) or 1 (TM), got {tetm_index!r}')
    return W, V, q


def layer_modes_1d_conical(xp, Kx, Ky, Kx2, Ky2, I, epx_conv, epy_conv, epz_conv_i, C,
                           mu=None, k0_d=None, dng=None):
    """Raw (unrotated) eigenmodes of one conical 1D layer.

    Returns ``(W1, W2, q1, q2, V_11, V_12, V_21, V_22)`` — the two decoupled mode families and
    the four blocks that couple them into the s/p system built by :func:`sp_modes_1d_conical`.
    Shared by the ETM and SMM connecting algorithms.

    The ``V`` blocks are, on paper, functions of ``A**-1`` and ``B**-1`` — but neither inverse
    is formed. On this path ``Ky2 == ky**2 * I`` (``n_orders_v == 1``), so ``W1`` diagonalises
    ``A`` itself with eigenvalues ``w1 - ky**2``, and ``W2`` does the same for ``B @ epx_conv``;
    substituting those identities turns every block into an exact function of the computed
    eigensystem, plus one inverse each of the well-conditioned ``W1`` / ``W2``.

    Why that matters: when an order sits exactly on a layer's light line at ``ky == 0``
    (``kx_m == n``, a Rayleigh point — D#81), ``A`` and ``B`` are numerically singular, and
    ``inv(A)`` returns garbage of arbitrary size with nothing raised — measured answers ranged
    from 0.18 to 626 against a true 0.22, some passing every guard. Built from the eigensystem
    instead, the layer stays the exact mode set of a *nearby* problem (``eig`` is backward
    stable, and the physics is continuous across the point), and the solve at the exact
    Rayleigh point agrees with the one-ULP-detuned answer to 8 digits — the same stability
    the pure TE/TM path (``V = W * q``, nothing singular inverted) always had. Away from any
    degeneracy the two spellings agree to 8e-15 in efficiency over 40 random conical solves.

    The one case still refused: an eigenvalue that comes out *exactly* zero (e.g. a uniform
    layer whose FFT is noiseless, truncation_order=3 on the D#81 stack). Then ``q/w`` is 0/0 and the NaN
    reaches :func:`~wavebit._core.backend.check_finite`, whose message prescribes the one-ULP
    detune — where the old spelling died in LAPACK with no advice.

    ``mu`` is the optional ``(mux_conv, muy_conv, muz_conv_i)`` triple and ``k0_d`` / ``dng``
    the label choice :func:`layer_modes_1d` documents, ``None`` for a non-magnetic layer, for
    the bounded roots and for no word on a double negative respectively.

    **Why the two families still decouple with a permeability**, which is the one thing here
    that had to be checked rather than assumed. Written out, the coupled ``Omega_L @ Omega_R``
    of ``docs/magnetic-materials.md`` §1 has off-diagonal blocks proportional to
    ``Kx muz_i muy - epy epz_i Kx``, so the split into two ``n``-sized eigenproblems needs
    ``eps_y = eps_z`` **and** ``mu_y = mu_z`` — as matrices, not as materials. Both hold
    exactly on a 1D grating and for one reason: ``E_y``, ``E_z``, ``H_y`` and ``H_z`` are all
    *tangential* to a boundary that is perpendicular to x, so all four slots are built by the
    direct (Laurent) rule from one profile, and only the ``x`` slots take the inverse rule.
    Measured on a random 1D grating at ``truncation_order=5``: ``|epy - inv(epz_i)| = 9e-16``
    against a matrix of scale 1.9, while ``|epx - inv(epz_i)| = 0.16``. The permeability
    inherits it because the *same builder* runs a second time (§8), so a scene that breaks
    the premise for one tensor breaks it for both -- and the premise is the conical path's
    already, older than this feature.

    Under it the two operators are exact duals::

        family 1 (H_x):  ky**2 + (Kx muz_i Kx - eps_y) mu_x
        family 2 (E_x):  ky**2 + (Kx epz_i Kx - mu_y ) eps_x

    and so are the ``V`` blocks: the **E-side** pair ``V_11`` / ``V_12`` takes the ``mu``,
    the **H-side** pair ``V_21`` / ``V_22`` is untouched. That asymmetry is the duality
    showing, not an omission -- ``V_21`` already carried the ``epz_conv_i`` that ``V_12``
    now mirrors with ``muz_conv_i``, and the two lines have been each other's reflection
    since before there was a permeability to put in one of them.
    """
    mux_conv, muy_conv, muz_conv_i = (I, I, None) if mu is None else mu
    mirrored, patterned = layer_label_rule(
        xp, dng, (epx_conv, epy_conv, epz_conv_i),
        None if mu is None else (mux_conv, muy_conv, muz_conv_i))
    # `A` is Faraday's row and takes `muz_i` between the K factors; `B` is Ampere's and keeps
    # `epz_i` where it always was. `Kx2` is `Kx @ Kx` for a diagonal Kx, kept as the diagonal
    # build where there is no mu_zz so the non-magnetic product is the one it was.
    A = (Kx2 if muz_conv_i is None else Kx @ muz_conv_i @ Kx) - epy_conv
    B = Kx @ epz_conv_i @ Kx - muy_conv
    from .modal_gradient import _stop, generator, inverse
    operator1 = Ky2 + (A if mu is None else A @ mux_conv)
    operator2 = Ky2 + B @ epx_conv
    w1, W1 = eig(_stop(operator1))
    w2, W2 = eig(_stop(operator2))

    def family_flux(W_f, w_f, slot):
        # Unrotated, a family-1 mode is `H_s = W1` against `E_s = V_11 = mu_x W1 q1/wA`, and a
        # family-2 mode `F_p = W2` against `H_p = V_22 = eps_x W2 q2/wB`, the blocks built below;
        # the per-order s/p rotation leaves the flux as it is. Formed only when it is read.
        def of(q):
            with division_at_a_cutoff():
                other = (slot @ W_f) * (q / (w_f - Ky2[0, 0]))
            return mode_flux(xp, W_f, other)
        return of

    q1 = generator(xp, operator1, w1, W1,
                   _labels(xp, w1, C, k0_d, mirrored, patterned, family_flux(W1, w1, mux_conv)),
                   square_root=True)
    q2 = generator(xp, operator2, w2, W2,
                   _labels(xp, w2, C, k0_d, mirrored, patterned, family_flux(W2, w2, epx_conv)),
                   square_root=True)

    ky = Ky[0, 0]                    # Ky == ky * I on this path (single y order)
    ky_sq = Ky2[0, 0]
    wA = w1 - ky_sq                  # eigenvalues of A @ mu_x, by the shift identity
    wB = w2 - ky_sq                  # eigenvalues of B @ epx_conv
    W1_i = xp.linalg.inv(W1)
    W2_i = xp.linalg.inv(W2)

    if q1.ndim == 2 or q2.ndim == 2:
        Q1 = xp.diag(q1) if q1.ndim == 1 else q1
        Q2 = xp.diag(q2) if q2.ndim == 1 else q2
        modal1 = generator(xp, operator1, w1, W1, w1, square_root=False)
        modal2 = generator(xp, operator2, w2, W2, w2, square_root=False)
        modal1 = xp.diag(modal1) if modal1.ndim == 1 else modal1
        modal2 = xp.diag(modal2) if modal2.ndim == 1 else modal2
        invA = inverse(xp, modal1-ky_sq*I)
        invB = inverse(xp, modal2-ky_sq*I)
        V_11 = W1 @ Q1 @ invA
        middle = W1_i @ Kx @ W2 if muz_conv_i is None else W1_i @ Kx @ muz_conv_i @ W2
        V_12 = W1 @ (ky*invA) @ middle
        if mu is not None:
            V_11 = mux_conv @ V_11
            V_12 = mux_conv @ V_12
        V_21 = epx_conv @ W2 @ (ky*invB) @ (W2_i @ Kx @ epz_conv_i @ W1)
        V_22 = epx_conv @ W2 @ Q2 @ invB
    else:
        # The E-side blocks. `A**-1 = mu_x W1 diag(wA)**-1 W1_i` falls out of the eigenrelation
        # `A mu_x W1 = W1 diag(wA)`, which is where the leading `mu_x` comes from -- the same
        # substitution that keeps this function free of `inv(A)` at a Rayleigh point, one slot
        # richer. `Kx muz_i` inside `V_12` is the dual of the `Kx epz_conv_i` in `V_21`.
        # `wA` and `wB` are exactly zero where an order's `kx**2` meets the layer's permittivity.
        # With `ky == 0` that is a Rayleigh point and the quotients are 0/0; with `ky != 0` the
        # numerators stay nonzero, the quotients are inf, and the products below turn them into
        # NaN. Either way `check_finite` refuses the result downstream, so NumPy's own warnings
        # are silenced over the whole block (see `division_at_a_cutoff`).
        with division_at_a_cutoff():
            V_11 = W1 * (q1 / wA)
            V_12 = W1 @ ((ky / wA)[:, None] * (W1_i @ Kx @ W2 if muz_conv_i is None
                                               else W1_i @ Kx @ muz_conv_i @ W2))
            if mu is not None:
                V_11 = mux_conv @ V_11
                V_12 = mux_conv @ V_12
            # The H-side blocks are untouched by a permeability, and that is the duality rather
            # than an omission: they are built from `B**-1`, whose eigenrelation carries `eps_x`.
            V_21 = epx_conv @ (W2 * (ky / wB)) @ (W2_i @ Kx @ epz_conv_i @ W1)
            V_22 = epx_conv @ W2 * (q2 / wB)
    return W1, W2, q1, q2, V_11, V_12, V_21, V_22


def sp_modes_1d_conical(xp, F_c, F_s, W1, W2, V_11, V_12, V_21, V_22):
    """Rotate conical layer modes into the s/p boundary basis: ``(big_W, big_V)``.

    ``big_W`` maps mode amplitudes to the boundary variables ``(E_s, F_p)`` and ``big_V`` to
    ``(H_s, H_p)`` — the same pair the substrate seed ``big_F`` / ``big_G`` is written in.
    """
    V_ss = F_c @ V_11
    V_sp = F_c @ V_12 - F_s @ W2
    W_ss = F_c @ W1 + F_s @ V_21
    W_sp = F_s @ V_22
    W_ps = F_s @ V_11
    W_pp = F_c @ W2 + F_s @ V_12
    V_ps = F_c @ V_21 - F_s @ W1
    V_pp = F_c @ V_22

    big_W = _block(xp, [[V_ss, V_sp], [W_ps, W_pp]])
    big_V = _block(xp, [[W_ss, W_sp], [V_ps, V_pp]])
    return big_W, big_V


def _layer_mu(I, mu_all, layer_index):
    """One layer's ``(mux_conv, muy_conv, muz_conv_i)``, or ``None`` where it is non-magnetic.

    ``mu_all`` is the three per-layer stacks as ``solve_1d`` takes them, any of which may be
    ``None``. ``mux_conv`` / ``muy_conv`` fall back to the identity and ``muz_conv_i`` to
    ``None``, so a stack carrying ``mu`` on some layers and not others is expressible and the
    layers without it take the arithmetic they always had. The same fallback
    :func:`~wavebit._core.solve_2d.solve_2d` applies to its five, three slots narrower.
    """
    if all(s is None for s in mu_all):
        return None
    return (I if mu_all[0] is None else mu_all[0][layer_index],
            I if mu_all[1] is None else mu_all[1][layer_index],
            None if mu_all[2] is None else mu_all[2][layer_index])


def _run_k0_d(xp, k0, thickness, stacks, dng):
    """``(k0_ds, links)``: each layer's ``k0_d`` for the ETM's labels, and which layers join.

    ``stacks`` are the per-layer slot stacks the kernel builds its modes from (``None`` where
    absent) and ``dng`` the engine's per-layer word on double negatives, or ``None``. A run of
    identical layers (:func:`~wavebit._core.conventions.layer_runs`; ``links[i]`` joins layers
    ``i`` and ``i + 1``) decides its labels once on its whole thickness
    (:func:`~wavebit._core.conventions.run_k0_d`); every other layer reads ``k0*d`` as before.
    Shared with SMM's field-state rebuild, which has to take the very labels the ETM takes.
    """
    n_layers = len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]
    slots = [tuple(None if s is None else s[i] for s in stacks) for i in range(n_layers)]
    links = layer_runs(xp, slots, None if dng is None else [dng[i] for i in range(n_layers)])
    return run_k0_d(xp, k0, [thickness[i] for i in range(n_layers)], links), links


def _run_join(links, layer_index, below):
    """Whether the ETM's step into ``layer_index`` continues the run of the layer just below it.

    ``links`` from :func:`_run_k0_d`; ``below`` is what that layer left (``None`` for the first
    layer stepped, which rests on the half-space). ``True`` / ``False`` / traced.
    """
    if below is None or layer_index + 1 > len(links):
        return False
    return links[layer_index]


def _run_ab(xp, I, W_i, V_i, W, V, F, G, below, join):
    """The ETM's interface factors ``(a, b)`` for a layer, exact inside a run.

    ``a = (W^-1 F + V^-1 G)/2`` and ``b = (W^-1 F - V^-1 G)/2``, with ``F = W_b (I + Y)``,
    ``G = V_b (I - Y)`` the state the layer below left (``below = (W_b, V_b, Y)``). Where this
    layer continues the run of that one (``join``, :func:`_run_join`) the same two are written
    as ``a = I + (dW + dV)/2``, ``b = Y + (dW - dV)/2`` with ``dW = W^-1 (W_b - W) (I + Y)`` and
    ``dV = V^-1 (V_b - V) (I - Y)``: the same algebra, read off the difference of the two
    layers' modes. Slices of one layer have bit-equal modes, so the difference is zero and
    ``a = I``, ``b = Y`` exactly -- the plane between them reflects nothing, and the recursion
    now says so instead of reaching ``I`` through ``W^-1 W``. That rounding is what a run
    cannot afford: a run of the far half-space's own material whose thin slice touches the
    half-space carries, at the top of that slice, a ``Y`` of 4e17 (its capped label against
    the half-space pivots on 1.7e-18, which the uncut layer's own ``exp(-k0 q d)`` balances), and
    ``W^-1 W (I + Y)`` turned it into an exact zero on ``a``'s diagonal -- ``LinAlgError`` where
    the uncut layer is right to 2.4e-13 (TE, ``eps = 5.79`` on itself, 0.5 um + 60 um,
    ``theta = 0.43+0.067j``). Stepped this way a split layer is the uncut one: ``Y`` crosses
    the plane unchanged and is carried by the next slice's propagation alone. The derivative is
    exact too: along any direction that keeps the slices equal ``W_b - W`` stays zero, and along
    one that parts them it is ``W^-1 (dW_b - dW) (I + Y)``, as the plain form's.
    """
    if join is not True:
        a = 0.5 * (W_i @ F + V_i @ G)
        b = 0.5 * (W_i @ F - V_i @ G)
        if join is False:
            return a, b
    W_b, V_b, Y = below
    dW = W_i @ (W_b - W) @ (I + Y)
    dV = V_i @ (V_b - V) @ (I - Y)
    a_run = I + 0.5 * (dW + dV)
    b_run = Y + 0.5 * (dW - dV)
    if join is True:
        return a_run, b_run
    return xp.where(join, a_run, a), xp.where(join, b_run, b)


def _keep_below(links, layer_index, *state):
    """``state`` if the next layer up may continue this one's run, else ``None`` (nothing kept)."""
    if layer_index == 0 or links[layer_index - 1] is False:
        return None
    return state


def solve_1d(tetm_index, wavelength, kx, n_top, n_bot, theta, thickness,
             epx_conv_all, epy_conv_all, epz_conv_i_all,
             incidence_side='top', answers=Answers.EFFICIENCY, lit=None,
             mu=None, mu_top=1.0, mu_bot=1.0, eps_top=None, eps_bot=None, phi=None, dng=None):
    """Solve a 1D grating stack for reflection/transmission diffraction efficiency.

    Args:
        tetm_index: ``0`` (TE) or ``1`` (TM) — the two the ``tetm`` declaration selects.
        wavelength: incident wavelength (sets ``k0 = 2*pi/wavelength``).
        kx: in-plane wavevectors of the retained orders (length ``n_orders = 2*truncation_order+1``),
            built
            with the *incidence* medium's index (``n_bot`` for backside illumination).
        n_top, n_bot: start / end refractive indices of the private propagation frame.
        theta: incident angle metadata (``None`` -> 0); boundary k is taken from ``kx``,
            measured from the incidence medium's normal.
        thickness: per-layer thicknesses in positive-z list order (public bottom-to-top).
        epx_conv_all, epy_conv_all, epz_conv_i_all: per-layer convolution matrices,
            indexed ``[layer]`` on the first axis, as produced by the factorization stage.
        incidence_side: ``'top'`` (default) or ``'bottom'``. Backside illumination is solved
            as top-side illumination of the z-mirrored scene (see :func:`_mirrored_scene`);
            results are incidence-relative either way — ``de_ri`` is power back into the
            incidence medium, ``de_ti`` power through the far side.

        answers: how far up the ladder this solve goes -- see
            :class:`~wavebit._core.enums.Answers`. The core's default is the cheapest rung,
            so a caller that names nothing pays for nothing; the solver classes declare
            their own (``'amplitude'``).

        mu: optional per-layer permeability, three stacks
            ``(mux_conv_all, muy_conv_all, muz_conv_i_all)`` in the permittivity's own slots
            and convention -- see :func:`layer_modes_1d`, which also says why there are three
            here and five on the 2D path. Any entry may be ``None``; ``None`` throughout is a
            non-magnetic stack, which takes the arithmetic that existed before ``mu`` did, bit
            for bit.
        mu_top, mu_bot: the half-spaces' scalar permeabilities. ``n_top`` / ``n_bot`` keep
            meaning the refractive index ``sqrt(eps*mu)``, so ``kz``, Snell and the order grid
            are untouched and every existing call means what it did; ``mu`` enters only
            through the boundary admittances, ``kz/mu`` on the electric amplitude (TE) and
            ``kz/eps`` on the magnetic one (TM). That pair is the *whole* content of a
            magnetic half-space, and it is what a solver carrying only ``n`` cannot express:
            between vacuum and an impedance-mismatched medium of the same index it sees no
            interface at all.
        eps_top, eps_bot: those half-spaces' permittivities. Optional, recovered as
            ``n**2/mu`` when absent -- which is what this function did unconditionally and is
            still right for a caller holding only an index. Supply them with ``mu_top`` /
            ``mu_bot`` or with neither; half a pair describes a medium whose index and
            permittivity disagree. Same argument as
            :func:`~wavebit._core.solve_2d.solve_2d`'s, which see.
        phi: the frame's azimuth at ``kx == 0``, the only place this path reads an azimuth
            (:func:`~wavebit._core.conventions.order_frame_sign`): ``None`` or ``0`` (an order
            exactly at ``kx == 0`` has ``s_hat = +y``), or ``pi``, which the engine passes at
            normal incidence from a negative-index medium (``s_hat = -y`` there, the limit of
            ``theta -> 0+``).
        dng: per layer, in the same positive-z list order as ``thickness``, the engine's word
            on its materials (:class:`~wavebit._core.conventions.DoubleNegativeLayer`: does it
            hold a double-negative region, is it one medium), or ``None`` for no word. ``None``
            for the whole stack is a scene in which no layer can be one. Only the labels of a
            layer's modes read it (:func:`layer_modes_1d`), so it moves no exact answer, only
            how well the recursion is conditioned; it reverses with the layers under backside
            incidence.

    Returns:
        ``(de_ri, de_ti, power_entering)`` — reflected / transmitted efficiency arrays,
        shape ``(1, n_orders)``, and scalar net entering power divided by incident power.
        From ``answers='amplitude'`` up, the full result mapping instead (amplitudes and s/p
        split; the pure TE/TM path carries all of one polarization, the other side is zero).
        Every amplitude is quoted in its order's own s/p frame, as on the coupled paths --
        ``sigma_m = order_frame_sign(kx_m)`` times the Cartesian primary -- and the incidence
        (``lit`` included) is that frame's unit s (TE) or p (TM) wave. From ``'field'`` up
        that mapping also carries the ``layer_info_list`` / ``T1``
        state field reconstruction consumes -- for a unit incident **E** on both
        polarizations: the TM solve is posed in unit ``H_y``, and its ``T1`` is rescaled by the
        incidence medium's ``n/mu`` on the way out.
    """
    answers = answers_of(answers)
    keep_field = answers.covers(Answers.FIELD)
    xp = array_namespace(epx_conv_all)
    mu_all = (None,) * 3 if mu is None else tuple(mu)
    if _is_backside(incidence_side):
        # After the namespace read: the mirror rebinds the per-layer stacks to plain lists
        # (static indexing keeps them traceable), which array_namespace does not accept.
        # `mu` reverses with the layers it belongs to -- `z -> -z` leaves a layer's in-plane
        # tensor alone whether it is eps or mu -- so it is the same mirror, three stacks wider.
        (n_top, n_bot, thickness, epx_conv_all, epy_conv_all, epz_conv_i_all,
         *mu_all, dng) = _mirrored_scene(n_top, n_bot, thickness,
                                         epx_conv_all, epy_conv_all, epz_conv_i_all, *mu_all,
                                         dng)
        # The half-spaces swap with the indices they belong to -- all three quantities of
        # each, or the mirrored scene takes its permittivity from the far side.
        mu_top, mu_bot = mu_bot, mu_top
        eps_top, eps_bot = eps_bot, eps_top
    kx = as_array(xp, kx)
    if not xp.isdtype(kx.dtype, 'complex floating'):
        # Promote a real kx to the complex working dtype the algebra needs.
        kx = xp.astype(kx, xp.complex128)
    C = kx.dtype
    # Every order's own s/p frame on this row: `s_hat = sigma * y_hat`, `p_hat` turning over
    # with it (:func:`~wavebit._core.conventions.order_frame_sign`). `kx` is the incidence
    # medium's and the mirror above does not touch it, so one `sigma` serves both sides.
    sigma = order_frame_sign(xp, kx, phi)

    n_orders_u = kx.shape[0]
    k0 = 2 * 3.141592653589793 / wavelength
    theta = 0.0 if theta is None else theta

    I = xp.astype(xp.eye(n_orders_u), C)

    # --- stage 1: boundary init (port of transfer_1d_1) --------------------------------
    # One kz per half-space on the downward branch; it serves both the substrate admittance
    # and the efficiency factors. Scalars promote against the complex kx. A caller holding a
    # half-space's permittivity hands its `(eps, mu)` over, which is what tells a double
    # negative's root from an ordinary one (`boundary_kz`); without it the root is Maystre's.
    kz_top = boundary_kz(xp, n_top ** 2, kx ** 2, half_space_medium(eps_top, mu_top))
    kz_bot = boundary_kz(xp, n_bot ** 2, kx ** 2, half_space_medium(eps_bot, mu_bot))

    if eps_bot is None:
        eps_bot = n_bot ** 2 / mu_bot     # n stays the refractive index; eps carries the mu
    # The substrate admittance, and it is the one place a magnetic half-space is expressible
    # at all: `kz` depends only on the index, so `kz/mu` (TE, an electric amplitude) and
    # `kz/eps` (TM, a magnetic one) are what separate two media of the same `n`. Both are the
    # expression that was here at mu = 1 -- dividing by `mu_bot = 1.0` is exact, and
    # `n_bot**2/1.0` is `n_bot**2`.
    if tetm_index == 0:      # TE
        G = -1j * _diag(xp, kz_bot / mu_bot)
    elif tetm_index == 1:    # TM
        G = -1j * _diag(xp, kz_bot / eps_bot)
    else:
        raise ValueError(f'tetm_index must be 0 (TE) or 1 (TM), got {tetm_index!r}')

    F = I
    T = I

    # --- stage 2: per-layer eig + transfer coupling (transfer_1d_2 / transfer_1d_3) ----
    layer_info_list = []
    n_layers = len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]
    run_k0_ds, run_links = _run_k0_d(xp, k0, thickness,
                                     (epx_conv_all, epy_conv_all, epz_conv_i_all, *mu_all),
                                     dng)
    below = None
    for layer_index in range(n_layers)[::-1]:
        epx_conv = epx_conv_all[layer_index]
        epy_conv = epy_conv_all[layer_index]
        epz_conv_i = epz_conv_i_all[layer_index]
        d = thickness[layer_index]
        layer_mu = _layer_mu(I, mu_all, layer_index)

        # `k0_d` asks for the ETM's labels (:func:`~wavebit._core.conventions.etm_mode_q`):
        # this recursion inverts a transmission into the medium below, so a layer's `+`
        # modes have to be the waves that medium receives. It is the run's `k0*d`, which is
        # the layer's own unless an identical layer touches it (:func:`_run_k0_d`).
        W, V, q = layer_modes_1d(xp, tetm_index, kx, I, epx_conv, epy_conv, epz_conv_i, C,
                                 mu=layer_mu, k0_d=run_k0_ds[layer_index],
                                 dng=None if dng is None else dng[layer_index])

        from .modal_gradient import exponential
        X = exponential(xp, q, -k0*d)
        W_i = xp.linalg.inv(W)
        V_i = xp.linalg.inv(V)
        # Inside a run the factors are taken from the difference of the slices' modes, which
        # is zero: `a = I`, `b = Y` exactly (:func:`_run_ab`). Elsewhere the plain form, and a
        # layer of the same material as the half-space below still inverts its `a`, which is
        # exactly I there -- the branch that once skipped that inverse is gone; see the module
        # docstring.
        a, b = _run_ab(xp, I, W_i, V_i, W, V, F, G, below,
                       _run_join(run_links, layer_index, below))
        a_i = xp.linalg.inv(a)
        del a

        Y = X @ b @ a_i @ X
        F = W @ (I + Y)
        G = V @ (I - Y)
        T = T @ a_i @ X
        below = _keep_below(run_links, layer_index, W, V, Y)

        # Bottom-to-top, matching the legacy append order; consumed by field reconstruction.
        # Conditional, and that is a *peak* saving on top of the lifetime one: `W`, `V`, `q`,
        # `a_i` and `b` are the loop's own working set, and without this append the previous
        # layer's five become garbage as soon as the next iteration rebinds them. Only `F`,
        # `G` and `T` have to survive an iteration.
        #
        # `muz_conv_i` rides last, for the reason `epz_conv_i` rides first: the longitudinal
        # component of each polarization is algebraic rather than propagated, and is recovered
        # by dividing out the z slot of *its own* tensor. TE reconstructs `H_z` through
        # `mu_zz` (Faraday's z row) and TM reconstructs `E_z` through `eps_zz` (Ampere's), so
        # each branch reads one of the two and the slot is `None` on a non-magnetic layer --
        # the spelling `_layer_mu` and `Material.mu` already use. See
        # :func:`wavebit._core.fields.field_1d`.
        if keep_field:
            layer_info_list.append([epz_conv_i, W, V, q, d, a_i, b,
                                    None if layer_mu is None else layer_mu[2]])

    # --- stage 3: final matching -> efficiency (transfer_1d_4) -------------------------
    Kz_top = _diag(xp, kz_top)
    # The incident column in the primary's own Cartesian component (E_y for TE, H_y for TM),
    # which is what the matching below is written in. A weight means "this much of that order's
    # s_hat (TE) / p_hat (TM)", as on the coupled paths, and on this row that is `sigma_m` of
    # the Cartesian primary. The unlit column takes the scalar `sigma_inc` rather than the
    # elementwise product, so every other entry stays `+0.0` and a `sigma_inc = +1` solve is
    # the arithmetic it was, signed zeros included; `lit is None` is structural, so jit-safe.
    column = incident_column(xp, C, lit, n_orders_u, n_orders_u // 2)
    delta = column * (sigma[n_orders_u // 2] if lit is None else sigma)
    # The incident wave's own kz, read off the same vector the reflected orders use. It equals
    # `n_top * cos(theta)` for a real angle in a lossless medium, and is taken from `kz_top` rather
    # than recomputed so that a complex angle cannot put the two sides of the match on opposite
    # branches -- see :func:`~wavebit._core.conventions.incident_kz`.
    kz_inc = incident_kz(kz_top, n_orders_u // 2)

    # The superstrate's admittance, the same pair the substrate seed `G` was built from and
    # necessarily the same spelling: the two sides of this match have to see one medium law.
    # `Kz_top / mu_top` is `Kz_top` to the bit at mu_top = 1, and `n_top**2/mu_top` is
    # `n_top**2`, so the undeclared case keeps the arithmetic it had.
    if eps_top is None:
        eps_top = n_top ** 2 / mu_top
    if tetm_index == 0:  # TE
        Kz_scaled = Kz_top / mu_top
        inc = (-1j * kz_inc / mu_top * delta if lit is None
               else -1j * (Kz_scaled @ delta))
        T1 = xp.linalg.inv(G - 1j * Kz_scaled @ F) @ (-1j * Kz_scaled @ delta + inc)
    else:         # TM
        Kz_scaled = Kz_top / eps_top
        inc = (-1j * delta * kz_inc / eps_top if lit is None
               else -1j * (Kz_scaled @ delta))
        T1 = xp.linalg.inv(G - 1j * Kz_scaled @ F) @ (-1j * Kz_scaled @ delta + inc)

    R = (F @ T1 - delta)[None, :]
    Tt = (T @ T1)[None, :]

    # The pure TE channel carries E; pure TM carries H. Evaluate each wave's
    # real power before dividing, including each illuminated order's weight.
    admittance_top = kz_top / (mu_top if tetm_index == 0 else eps_top)
    admittance_bot = kz_bot / (mu_bot if tetm_index == 0 else eps_bot)
    incoming_power = incident_power(xp, C, lit, admittance_top)
    de_ri = wave_power(xp, R, admittance_top) / incoming_power
    de_ti = wave_power(xp, Tt, admittance_bot) / incoming_power
    entering = power_for_real_k(
        xp, xp.sum(net_power(xp, delta[None, :], R, admittance_top)) / incoming_power, kx)

    check_finite(xp, de_ri=de_ri, de_ti=de_ti)

    if not answers.covers(Answers.AMPLITUDE):
        return de_ri, de_ti, entering

    # TE carries everything on s, TM everything on p; the opposite side is identically zero.
    # The pure path's primary, taken into each order's own s/p frame by `sigma` (on this row an
    # order's s_hat, and the axis of its p wave's magnetic field, are `sigma * y_hat`), IS the
    # Fresnel amplitude on the reflection side: for TM the primary is H with a unit incident H,
    # and the H ratio is the E ratio in the convention where a normal-incidence interface
    # gives r_p = -r_s.
    # Transmission converts H back to E across the index step, t_E = t_H * n_top/n_bot.
    # Measured at 5.6e-16 (tests/test_lossy_media). Signed *after* that conversion, so the
    # product is the one it always was and only its sign moves; until 2026-09-27 this path
    # quoted the Cartesian primary on every order, which is `sigma_m * sigma_inc` of this.
    #
    # With a permeability the step is by **impedance**, `Z = sqrt(mu/eps) = mu/n`, so the
    # ratio is `(n_top/mu_top)/(n_bot/mu_bot)` -- the same `n/mu` per side that
    # :func:`final_matching`'s `fresnel` divides the coupled path's p amplitudes by, which is
    # what keeps one Fresnel convention across the two paths. It is `n_top/n_bot` to the bit
    # when neither half-space names a `mu`.
    zeros_amp = xp.zeros_like(R)
    zeros_eff = xp.zeros_like(de_ri)
    if tetm_index == 0:
        res = _pack(R * sigma, zeros_amp, Tt * sigma, zeros_amp, de_ri, zeros_eff, de_ti,
                    zeros_eff, power_entering=entering)
    else:
        # Parenthesised as one ratio, not `Tt * a / b`: the two associate differently and
        # differ by an ulp, and this spelling is `Tt * (n_top / n_bot)` exactly at mu = 1.
        res = _pack(zeros_amp, R * sigma, zeros_amp,
                    (Tt * ((n_top / mu_top) / (n_bot / mu_bot))) * sigma,
                    zeros_eff, de_ri, zeros_eff, de_ti, power_entering=entering)
    if not keep_field:
        return {'incident': res}
    if tetm_index == 1:
        # The coupled paths' p input is unit **E**; the TM solve above is posed in unit H (its
        # admittances and powers are written that way, and a Fresnel ratio does not care), so
        # the state the field walks is rescaled once, here, to the incidence that every other
        # path lights with. `h = n/mu` is the incidence medium's admittance in Z0 units -- the
        # |H| of a unit-E wave there -- and after the mirror `n_top` / `mu_top` are that medium
        # on both sides. Only the reconstruction state moves: `R`, `T` and the amplitudes above
        # keep their bits. Until 2026-09-27 this path's field was `mu_inc/n_inc` of the
        # conical p field's.
        T1 = T1 * (n_top / mu_top)
    return {'incident': res, 'layer_info_list': layer_info_list, 'T1': T1}


def solve_1d_conical(wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
                     epx_conv_all, epy_conv_all, epz_conv_i_all,
                     *, phi, incidence_side='top', answers=Answers.EFFICIENCY, lit=None,
                     mu=None, mu_top=1.0, mu_bot=1.0, eps_top=None, eps_bot=None, dng=None):
    """Solve a general 1D grating (arbitrary azimuth/polarization) for the given ``psi``.

    The conical case couples the s and p polarizations, so the per-layer modes and the boundary
    system are ``2*n_orders`` block matrices. Port of ``transfer_1d_conical_1..4``; returns the
    ``(de_ri, de_ti, power_entering)`` of the specified incidence: two efficiency arrays
    of shape ``(1, n_orders_u)`` and a scalar net entering power fraction.
    See :func:`solve_1d` for the argument conventions — including
    ``incidence_side``, handled by the same mirror; ``ky`` is the single out-of-plane order and
    ``psi`` the polarization angle. ``polarization`` is the complex amplitude pair ``(a_s,
    a_p)`` that supersedes ``psi`` when given — see
    :func:`~wavebit._core.conventions.incident_sp_amplitudes`.

    ``mu`` / ``mu_top`` / ``mu_bot`` / ``eps_top`` / ``eps_bot`` are the permeability and the
    half-space terms :func:`solve_1d` documents, and ``dng`` the per-layer word it documents,
    in the same three slots
    (:func:`layer_modes_1d_conical` says why the coupled 1D operator still takes three where
    the 2D path takes five). Non-magnetic throughout is bit-identical to the arithmetic that
    predates them.
    """
    xp = array_namespace(epx_conv_all)
    mu_all = (None,) * 3 if mu is None else tuple(mu)
    if _is_backside(incidence_side):
        # `mu` reverses with its layers and the half-space triples swap, exactly as in
        # :func:`solve_1d`; the mirror is one function and takes three more stacks.
        (n_top, n_bot, thickness, epx_conv_all, epy_conv_all, epz_conv_i_all,
         *mu_all, dng) = _mirrored_scene(n_top, n_bot, thickness,
                                         epx_conv_all, epy_conv_all, epz_conv_i_all, *mu_all,
                                         dng)
        mu_top, mu_bot = mu_bot, mu_top
        eps_top, eps_bot = eps_bot, eps_top
    kx = as_array(xp, kx)
    ky = as_array(xp, ky)
    if not xp.isdtype(kx.dtype, 'complex floating'):
        kx = xp.astype(kx, xp.complex128)
        ky = xp.astype(ky, xp.complex128)
    C = kx.dtype

    n_orders_u = kx.shape[0]
    n_orders_v = ky.shape[0]
    n_orders = n_orders_u * n_orders_v
    k0 = 2 * 3.141592653589793 / wavelength
    theta = 0.0 if theta is None else theta

    I = xp.astype(xp.eye(n_orders), C)
    O = xp.astype(xp.zeros((n_orders, n_orders)), C)
    ky_col = xp.reshape(ky, (-1, 1))

    # --- stage 1: boundary init (transfer_1d_conical_1) --------------------------------
    k_par_sq = kx ** 2 + ky_col ** 2
    kz_top = xp.reshape(boundary_kz(xp, n_top ** 2, k_par_sq,
                                    half_space_medium(eps_top, mu_top)), (-1,))
    kz_bot = xp.reshape(boundary_kz(xp, n_bot ** 2, k_par_sq,
                                    half_space_medium(eps_bot, mu_bot)), (-1,))
    varphi = order_azimuth(xp, ky_col, kx, phi)

    if eps_bot is None:
        eps_bot = n_bot ** 2 / mu_bot      # n stays the refractive index; eps carries the mu
    Kz_bot = _diag(xp, kz_bot)

    # The substrate's two admittances: `kz/eps` on the magnetic (p) amplitude, `kz/mu` on the
    # electric (s) one. Both are the previous expression at mu_bot = 1, and both are the same
    # pair :func:`~wavebit._core.solve_2d.solve_2d`'s seed carries -- one boundary law, two
    # formulations that must not spell it differently.
    big_F = _block(xp, [[I, O], [O, -1j * Kz_bot / eps_bot]])
    big_G = _block(xp, [[-1j * Kz_bot / mu_bot, O], [O, I]])
    big_T = xp.astype(xp.eye(2 * n_orders), C)

    # --- stage 2/3: per-layer eig + coupled s/p transfer (conical_2 / conical_3) --------
    Kx = _diag(xp, kx)
    ky_vec = ky[0] * xp.astype(xp.ones((n_orders_u,)), C)      # n_orders_v == 1: Ky = diag(ky0 * 1)
    Ky = _diag(xp, ky_vec)
    Kx2 = _diag(xp, kx * kx)
    Ky2 = _diag(xp, ky_vec * ky_vec)
    F_c = _diag(xp, xp.cos(varphi))
    F_s = _diag(xp, xp.sin(varphi))

    keep_field = answers_of(answers).covers(Answers.FIELD)
    layer_info_list = []
    n_layers = len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]
    run_k0_ds, run_links = _run_k0_d(xp, k0, thickness,
                                     (epx_conv_all, epy_conv_all, epz_conv_i_all, *mu_all),
                                     dng)
    below = None
    for layer_index in range(n_layers)[::-1]:
        epx_conv = epx_conv_all[layer_index]
        epy_conv = epy_conv_all[layer_index]
        epz_conv_i = epz_conv_i_all[layer_index]
        d = thickness[layer_index]
        layer_mu = _layer_mu(I, mu_all, layer_index)

        # The ETM's labels, on its run's thickness, for the reason :func:`solve_1d` gives.
        W1, W2, q1, q2, V_11, V_12, V_21, V_22 = layer_modes_1d_conical(
            xp, Kx, Ky, Kx2, Ky2, I, epx_conv, epy_conv, epz_conv_i, C, mu=layer_mu,
            k0_d=run_k0_ds[layer_index], dng=None if dng is None else dng[layer_index])

        from .modal_gradient import exponential, combine
        X_1 = exponential(xp, q1, -k0*d)
        X_2 = exponential(xp, q2, -k0*d)
        big_I = xp.astype(xp.eye(2 * n_orders), C)
        big_X = _block(xp, [[X_1, O], [O, X_2]])

        W = _block(xp, [[W1, W2]])
        V = _block(xp, [[V_11, V_12], [V_21, V_22]])
        q = combine(xp, q1, q2)

        big_W, big_V = sp_modes_1d_conical(xp, F_c, F_s, W1, W2,
                                           V_11, V_12, V_21, V_22)

        big_W_i = xp.linalg.inv(big_W)
        big_V_i = xp.linalg.inv(big_V)
        # Exact inside a run, the plain form elsewhere (:func:`_run_ab`).
        big_A, big_B = _run_ab(xp, big_I, big_W_i, big_V_i, big_W, big_V, big_F, big_G, below,
                               _run_join(run_links, layer_index, below))
        big_A_i = xp.linalg.inv(big_A)
        del big_A

        Y = big_X @ big_B @ big_A_i @ big_X
        big_F = big_W @ (big_I + Y)
        big_G = big_V @ (big_I - Y)
        big_T = big_T @ big_A_i @ big_X
        below = _keep_below(run_links, layer_index, big_W, big_V, Y)

        # Conditional for the reason `solve_1d`'s append is: the per-layer matrices are the
        # loop's working set, and only `big_F`, `big_G` and `big_T` have to outlive an
        # iteration.
        #
        # The eighth slot is `muz_conv_i` -- one entry layout for the one function that reads
        # both this list and `solve_2d`'s (:func:`wavebit._core.fields._field_coupled`),
        # rather than two lengths it would have to tell apart. It held a hard `None` while
        # this operator had no permeability; it now holds the layer's own `mu_zz^-1`, which
        # `H_z` divides out. A `None` left here on a magnetic layer would return an `H_z`
        # scaled by `mu_zz` with every other component exact -- the 2D path's 2026-08-20
        # defect, on this path.
        if keep_field:
            layer_info_list.append([epz_conv_i, W, V, q, d, big_A_i, big_B,
                                    None if layer_mu is None else layer_mu[2]])

    # --- stage 3: final matching -> efficiency (transfer_1d_conical_4) ------------------
    out = final_matching(xp, C, big_F, big_G, big_T, kz_top, kz_bot, incidence, theta,
                         n_top, n_bot, n_orders_u, n_orders_v, answers=answers,
                         mu_top=mu_top, mu_bot=mu_bot, eps_top=eps_top, eps_bot=eps_bot,
                         lit=lit, kx=kx, ky=ky)
    if keep_field:
        out['layer_info_list'] = layer_info_list
    return out


def _unit(xp, n, i):
    """Unit vector ``e_i`` of length ``n`` (real)."""
    v = [0.0] * n
    v[i] = 1.0
    return xp.asarray(v)


def final_matching(xp, C, big_F, big_G, big_T, kz_top, kz_bot, incidence, theta,
                   n_top, n_bot, n_orders_u, n_orders_v, answers=Answers.EFFICIENCY,
                   mu_top=1.0, mu_bot=1.0, eps_top=None, lit=None, eps_bot=None,
                   *, kx, ky):
    """Boundary match the assembled s/p system to the incident field.

    Shared boundary equations from ``transfer_1d_conical_4`` / ``transfer_2d_4``;
    used by both :func:`solve_1d_conical` and
    :func:`wavebit._core.solve_2d.solve_2d`.

    The incidence is ``psi`` — or the complex amplitude pair ``polarization = (a_s, a_p)``,
    which supersedes ``psi`` when given (see
    :func:`~wavebit._core.conventions.incident_sp_amplitudes`; ``psi = v`` and
    ``polarization = (sin(v), cos(v))`` are the same incidence).

    ``eps_top`` is the superstrate's permittivity, the one half-space quantity this function
    consumes that is not recoverable for free: ``n_top`` stays the refractive index
    ``sqrt(eps*mu)``, so the p-channel admittance ``kz/eps`` has to be spelled either as
    ``kz*mu/n**2`` -- a round trip through a square root and back, exact only to rounding --
    or from the number itself. ``None`` takes the round trip, which is what a caller holding
    only an index has and is what this always did; a caller holding both passes it. The
    argument exists at all because the SMM twin
    (:func:`wavebit._core.scattering.smm_2d`) already read the value, and a boundary condition
    that two connecting algorithms build from different inputs is not one boundary condition.
    ``test_half_space_permittivity_reaches_both_connecting_algorithms`` in
    ``tests/test_scattering_core.py`` is the check that keeps them reading it together.

    ``eps_bot`` supplies the exit-side permittivity to the power conversion.
    When absent it is recovered from ``n_bot**2/mu_bot``.

    Returns ``(de_ri, de_ti, power_entering)``: two efficiency arrays of shape
    ``(n_orders_v, n_orders_u)`` and a scalar net entering power fraction.
    From ``answers='amplitude'`` up it returns the full result mapping instead: the reflected
    and transmitted amplitudes and their s/p-resolved efficiencies for the given incidence,
    plus the pure-TE and pure-TM incidence variants. All requested incidence
    columns share one direct solve of ``final_A``; no full inverse is formed.

    ``T1`` joins that mapping only at ``'field'``. It is *built from* the variants' columns,
    which is why the two cannot be ordered the other way round: a solve that kept ``T1``
    while dropping ``s_inc`` would be doing the extra solve and then throwing away the
    cheap half of its result.
    """
    answers = answers_of(answers)
    n_orders = n_orders_u * n_orders_v
    I = xp.astype(xp.eye(n_orders), C)
    O = xp.astype(xp.zeros((n_orders, n_orders)), C)

    # See :func:`~wavebit._core.conventions.incident_kz`: an identity for a real angle, and
    # the difference between a solve and a degenerate R = -delta for a complex one.
    kz_inc = incident_kz(kz_top, n_orders // 2)

    Kz_top = _diag(xp, kz_top)
    kz_top_2d = xp.reshape(kz_top, (n_orders_v, n_orders_u))
    kz_bot_2d = xp.reshape(kz_bot, (n_orders_v, n_orders_u))
    a_s, a_p = as_array(xp, incidence[0], dtype=C), as_array(xp, incidence[1], dtype=C)

    delta = xp.reshape(incident_column(xp, C, lit, n_orders, n_orders // 2), (n_orders, 1))

    bF = [[big_F[:n_orders, :n_orders], big_F[:n_orders, n_orders:]],
          [big_F[n_orders:, :n_orders], big_F[n_orders:, n_orders:]]]
    bG = [[big_G[:n_orders, :n_orders], big_G[:n_orders, n_orders:]],
          [big_G[n_orders:, :n_orders], big_G[n_orders:, n_orders:]]]

    # `big_F` holds tangential E and `big_G` tangential H, each split s/p, so the coefficients
    # below are the half-space admittances: `kz/mu` where the amplitude is electric (s) and
    # `kz/eps` where it is magnetic (p). Both reduce to what was here at mu = 1: dividing by
    # `mu_top = 1.0` is exact, so this row is the `kz/n**2` the function computed before there
    # was a `mu` at all, to the bit.
    if eps_top is None:
        eps_top = n_top ** 2 / mu_top
    final_A = _block(xp, [
        [I, O, -bF[0][0], -bF[0][1]],
        [O, 1j * Kz_top / eps_top, -bF[1][0], -bF[1][1]],
        [1j * Kz_top / mu_top, O, -bG[0][0], -bG[0][1]],
        [O, I, -bG[1][0], -bG[1][1]],
    ])
    # The incident wave in the same four quantities. Its p component appears twice, as the
    # magnetic amplitude `h_p_inc` in the last row and as the tangential E that amplitude
    # carries in the second, and the two have to describe **one** wave: `E = (kz/eps) * H` is
    # the same top half-space relation the reflected column above is built from, so it reads
    # the same `eps_top`. Spelling the E row `kz_inc/n_top` instead -- algebraically identical
    # while `eps*mu == n**2`, which is the only way a caller should ever supply them -- made
    # the incident wave and the reflected one see different media the moment they disagreed.
    #
    # The magnetic scale is `n/mu`: the input pair holds electric coefficients.
    # Equal coefficients need not carry equal power in an absorbing medium;
    # their actual powers are evaluated separately below.
    #
    # `e_p_inc` is the one expression in this function that is *not* bit-identical to its
    # pre-`mu` form. `(n/mu)*kz/eps` and `kz/n` agree exactly whenever `n_top` is a power of
    # two and to within one ulp otherwise; measured end to end, 13 of 324 reported values move,
    # by at most 4e-16 relative. Taken deliberately: the alternative is to keep `kz_inc/n_top`
    # when `eps_top` was recovered and use the permittivity only when it was supplied, which
    # would put two spellings of one medium back in the file -- agreeing until they do not,
    # which is the shape of the bug this argument was added to close.
    h_p_inc = n_top / mu_top
    e_p_inc = h_p_inc * kz_inc / eps_top
    kz_col = None if lit is None else Kz_top @ delta

    def incident_column_sp(s, p):
        # All requested polarization columns use the same per-order wavevectors.
        if lit is None:
            return _block(xp, [[-s * delta], [p * e_p_inc * delta],
                               [1j * s * kz_inc / mu_top * delta],
                               [1j * h_p_inc * p * delta]])
        return _block(xp, [[-s * delta], [p * h_p_inc * kz_col / eps_top],
                           [1j * s * kz_col / mu_top],
                           [1j * h_p_inc * p * delta]])

    final_B = incident_column_sp(a_s, a_p)

    if answers.covers(Answers.AMPLITUDE):
        # Keep the given, pure-TE and pure-TM columns in one right-hand side,
        # so one factorization answers all three without forming a dense inverse.
        # The variant sources retain the same admittance and normalization as
        # before; the declared illumination's column remains first.
        half_pi = 3.141592653589793 / 2
        cols = [final_B]
        for psi_v in (half_pi, 0.0):
            s_p = xp.sin(xp.asarray(psi_v, dtype=C))
            c_p = xp.cos(xp.asarray(psi_v, dtype=C))
            cols.append(incident_column_sp(s_p, c_p))
        solved = xp.linalg.solve(final_A, xp.concat(cols, axis=1))
        final_RT, final_RT_tetm = solved[:, :1], solved[:, 1:]
    else:
        final_RT = xp.linalg.solve(final_A, final_B)
    del final_A, final_B
    R_s = xp.reshape(final_RT[:n_orders, :], (n_orders_v, n_orders_u))
    R_p = xp.reshape(final_RT[n_orders:2 * n_orders, :], (n_orders_v, n_orders_u))

    big_T1 = final_RT[2 * n_orders:, :]
    big_T_tetm = big_T
    big_T = big_T @ big_T1
    T_s = xp.reshape(big_T[:n_orders, :], (n_orders_v, n_orders_u))
    T_p = xp.reshape(big_T[n_orders:, :], (n_orders_v, n_orders_u))

    s_power, p_power = sp_incident_power(xp, C, lit, kz_top, n_top, eps_top, mu_top)
    denom = xp.abs(a_s)**2 * s_power + xp.abs(a_p)**2 * p_power

    def efficiencies(R_s_, R_p_, T_s_, T_p_, incoming):
        return sp_efficiencies(xp, kz_top_2d, kz_bot_2d, incoming, n_top, n_bot,
                               R_s_, R_p_, T_s_, T_p_, mu_top=mu_top, mu_bot=mu_bot,
                               eps_top=eps_top, eps_bot=eps_bot)

    de_ri_s, de_ri_p, de_ti_s, de_ti_p = efficiencies(R_s, R_p, T_s, T_p, denom)

    delta_2d = xp.reshape(delta, (n_orders_v, n_orders_u))

    def entering_power(R_s_, R_p_, s, p, incoming):
        flux = (net_power(xp, s * delta_2d, R_s_, kz_top_2d / mu_top)
                + net_power(xp, -1j * h_p_inc * p * delta_2d, R_p_,
                            kz_top_2d / eps_top))
        value = xp.sum(flux, axis=(-2, -1)) / incoming
        return power_for_real_k(xp, value, kx, ky)

    entering = entering_power(R_s, R_p, a_s, a_p, denom)
    check_finite(xp, de_ri=de_ri_s + de_ri_p, de_ti=de_ti_s + de_ti_p)

    if not answers.covers(Answers.AMPLITUDE):
        return de_ri_s + de_ri_p, de_ti_s + de_ti_p, entering

    # The coupled paths' p boundary variable is a magnetic amplitude; ``/(-i*n)`` on its own
    # side is what makes it the Fresnel E-ratio (measured, tests/test_lossy_media).
    def fresnel(R_s_, R_p_, T_s_, T_p_):
        return (R_s_, R_p_ / (-1j * n_top / mu_top),
                T_s_, T_p_ / (-1j * n_bot / mu_bot))

    res = _pack(*fresnel(R_s, R_p, T_s, T_p), de_ri_s, de_ri_p, de_ti_s, de_ti_p,
                power_entering=entering)

    def split2(M):
        """Two incidence columns -> (2, n_orders_v, n_orders_u), the legacy ``.T.reshape``."""
        return xp.reshape(xp.matrix_transpose(M), (2, n_orders_v, n_orders_u))

    R_s_2 = split2(final_RT_tetm[:n_orders, :])
    R_p_2 = split2(final_RT_tetm[n_orders:2 * n_orders, :])
    big_T1_tetm = final_RT_tetm[2 * n_orders:, :]
    big_T_tetm = big_T_tetm @ big_T1_tetm
    T_s_2 = split2(big_T_tetm[:n_orders, :])
    T_p_2 = split2(big_T_tetm[n_orders:, :])

    variant_power = xp.reshape(xp.stack([s_power, p_power]), (2, 1, 1))
    eff2 = efficiencies(R_s_2, R_p_2, T_s_2, T_p_2, variant_power)
    amps2 = fresnel(R_s_2, R_p_2, T_s_2, T_p_2)
    variants = []
    for i, psi_v in enumerate((half_pi, 0.0)):
        s = xp.sin(xp.asarray(psi_v, dtype=C))
        p = xp.cos(xp.asarray(psi_v, dtype=C))
        entry = entering_power(R_s_2[i], R_p_2[i], s, p, (s_power, p_power)[i])
        variants.append(_pack(*(a[i] for a in amps2), *(e[i] for e in eff2),
                              power_entering=entry))

    out = {'incident': res, 's_inc': variants[0], 'p_inc': variants[1]}
    if answers.covers(Answers.FIELD):
        # Field reconstruction consumes T1 for the given incidence plus the two variants.
        out['T1'] = xp.stack([big_T1, big_T1_tetm[:, 0:1], big_T1_tetm[:, 1:2]])
    return out


def sp_efficiencies(xp, kz_top_2d, kz_bot_2d, denom, n_top, n_bot, R_s, R_p, T_s, T_p,
                    mu_top=1.0, mu_bot=1.0, eps_top=None, eps_bot=None):
    """Power-normalized diffraction efficiencies from the s/p boundary amplitudes.

    ``denom`` is the real incident power for each input polarization column.
    The s channel carries E and the p channel carries H. Shared by ETM and SMM.

    A half-space's admittance is ``kz/mu`` for s and ``kz/eps`` for p, and with
    ``eps = n**2/mu`` the second is ``kz*mu/n**2``. That is the *whole* content of a magnetic
    half-space: ``kz`` itself depends only on the refractive index, so a solver carrying ``n``
    alone cannot express one — it sees no interface at all between vacuum and an
    impedance-mismatched medium of the same index.

    ``mu_top`` / ``mu_bot`` default to 1 for non-magnetic half-spaces.
    """
    if eps_top is None:
        eps_top = n_top**2 / mu_top
    if eps_bot is None:
        eps_bot = n_bot**2 / mu_bot
    return (wave_power(xp, R_s, kz_top_2d / mu_top) / denom,
            wave_power(xp, R_p, kz_top_2d / eps_top) / denom,
            wave_power(xp, T_s, kz_bot_2d / mu_bot) / denom,
            wave_power(xp, T_p, kz_bot_2d / eps_bot) / denom)


def _pack(r_s, r_p, t_s, t_p, R_s, R_p, T_s, T_p, *, power_entering):
    """Assemble one incidence's results in the ``ResultSub`` field layout.

    Amplitudes arrive **already Fresnel-normalized** and in each order's own s/p frame — each
    pack site owns the conversion from its boundary variables, because the factor differs
    between the pure-1D path (the primary itself is the amplitude, signed into the order's
    frame by ``order_frame_sign``; TM transmission needs ``n_top/n_bot``) and the coupled
    paths (the p boundary variable is a magnetic amplitude; ``/(-i*n)`` on its own side).
    Both conversions are measured against the Airy forms in ``tests/test_lossy_media.py``.
    """
    return {'r_s': r_s, 'r_p': r_p, 't_s': t_s, 't_p': t_p,
            'R': R_s + R_p, 'R_s': R_s, 'R_p': R_p,
            'T': T_s + T_p, 'T_s': T_s, 'T_p': T_p,
            'power_entering': power_entering}
