"""First-order (4ff) layer solve — the formulation a z-coupled tensor needs.

**Status: the operator, the mode split and the stack solve are done and pinned, and the solve is
stable for any thickness. Nothing dispatches to it, multi-order is unvalidated, and there is no
field reconstruction.** See "What is not here" at the bottom before building on it.

Pinned by ``tests/test_solve_2d_4ff.py`` against the 2ff path and a Berreman 4x4, and by
``tests/references/test_4ff_external_goldens.py`` against RETICOLO and fmmax — 21 cases, worst
gap 2.4e-15, no convention fitting. Note what that does *not* cover: every case in both goldens
has an empty z row, because both were frozen for a 2ff consumer. The geometry this module
exists for is still judged only by our own Berreman.

Why it exists. :mod:`fieldwave._core.solve_2d` reduces a layer to the second-order form
``d2E_t/dz2 = Omega2 E_t``, which exists only when the permittivity has an empty z row and
column. With ``eps_xz`` or ``eps_yz`` populated the elimination of ``E_z`` leaves a
first-derivative term, the reduction fails, and the 2ff mode operator has nothing to be —
``docs/anisotropic-materials.md`` §6. Physically that is the Voigt geometry: a magnetisation
lying in the plane rather than along the stack normal, which is what a magnetised film does on
its own unless it has perpendicular anisotropy or an applied field.

What the first-order form keeps instead is the full transverse state,

    d/dz (Ex, Ey, Hx, Hy) = i k0 M (Ex, Ey, Hx, Hy)

with ``M`` of size ``4*ff``. It costs about 8x the 2ff path (``eig`` is cubic and the problem
doubles), measured at 9.5x on RETICOLO, which is why choosing between them belongs to the
solver and not the caller — ``docs/anisotropic-materials.md`` §8.

The derivation, for ``e^{-iwt}`` with ``d/dx -> i k0 Kx`` and ``d/dy -> i k0 Ky``::

    curl E = i k0 H,   curl H = -i k0 eps E

    z components:   Hz = Kx Ey - Ky Ex,      Dz = Ky Hx - Kx Hy
    and Dz = ezx Ex + ezy Ey + ezz Ez, so
                    Ez = ezz^-1 (Ky Hx - Kx Hy - ezx Ex - ezy Ey)

    leaving         Ex' = i k0 (Kx Ez + Hy)
                    Ey' = i k0 (Ky Ez - Hx)
                    Hx' = i k0 (Kx Hz - Dy)
                    Hy' = i k0 (Ky Hz + Dx)

Every product is a convolution-matrix product and ``Kx`` / ``Ky`` are the same diagonals the 2ff
path uses, so this is the Fourier-space statement rather than a per-order one.

**Forward and backward modes are not a sign apart here.** In 2ff, ``exp(+q z)`` and
``exp(-q z)`` solve the same second-order equation with the *same* eigenvector, so the pair came
for free and :func:`fieldwave._core.conventions.mode_q` only had to choose a root. A
first-order system has ``4*ff`` eigenvalues that are not ``+-`` pairs in general, and the two
directions are genuinely different mode sets. Measured: for a tilted magnetisation at 35
degrees the backward set departs from the negated forward set by 9e-2, while at normal
incidence the two coincide to 1e-15 — so the pairing survives exactly where nobody would have
noticed it failing.

**What makes the stack solve stable is the direction split, not the algorithm on top of it.**
The split is what lets a layer be propagated by ``exp(i k0 lam_f d)`` forward and
``exp(-i k0 lam_b d)`` backward. **In a passive medium** both have modulus at most one for any
thickness — a mode carrying energy towards ``+z`` cannot gain amplitude on the way, so
``Im(lam_f) >= 0``, and the mirrored statement holds backwards. Two of the three connecting
algorithms are then arrangements that keep only those factors:

===========  ==================================  =====================  ===================
``algo``     form                                thickness              cost, 40 layers
===========  ==================================  =====================  ===================
``'ETM'``    enhanced transmittance (default)    unbounded              0.94x
``'SMM'``    scattering matrices, Redheffer      unbounded              1.13x
``'TMM'``    ``Psi exp(...) Psi^-1``, naive      refuses past x = 9     1.00x
===========  ==================================  =====================  ===================

ETM and SMM agree to 1.2e-15 over eight tensors, three angles and both polarisations, and both
hold ~1e-13 relative accuracy out to a modal exponent of 2500, where they reproduce the
closed-form single-interface Fresnel reflectance to 4e-16.

TMM is kept deliberately naive, as a third opinion whose failure mode is understood: it holds
the growing and decaying halves in one matrix and inverts it, so its relative error grows like
``eps*exp(2x)``. Its bound is set by *accuracy*, not by overflow — see ``_MAX_EXPONENT``.

**None of this covers a gain medium**, where a forward mode amplifies by construction
(``Im(lam_f) < 0``; measured -0.108 for ``n = 1.5 - 0.1j``) and the propagators stop being
contractions. Measured anyway: ETM and SMM still track the Airy closed form to ~1e-13 out to an
exponent of 113, and past roughly 700 return NaN. Nothing guards that, because there the growth
is the physics rather than an artefact — but the stability claim is about passive media and
should not be read further.

What is not here, in the order it will be wanted:

* **Fourier orders beyond one.** The algebra is written in blocks and should extend as it
  stands, but nothing has checked it against a grating.
* **Dispatch.** Nothing routes to this. ``solve_for_conv`` still refuses a z-coupled tensor.
* **Field reconstruction**, and the ``mu`` rows that
  ``tests/references/golden_fmmax_magnetic.npz`` is already holding.
"""
from .backend import array_namespace, eig

# Flux below this, on a mode normalised to unit energy, is taken as none — the mode is
# evanescent and direction is decided by decay instead. See split_forward_backward.
_FLAT = 1e-12

# The transfer inverts a matrix holding exp(+x) and exp(-x) at once, so it loses relative
# precision like eps*exp(2x) -- measured 1.2e-8 at x = 9.1 against a predicted 1.8e-8, and 8.5e-4
# by x = 15.9. This bound is where that prediction passes ~1e-9, which is where TMM stops being
# usable as a cross-check on SMM. It is deliberately far below the overflow point: the earlier
# value of 300 only caught the crash, and let three decades of silently wrong answers through
# first (5.1e+25 relative error at x = 34, no complaint).
_MAX_EXPONENT = 9.0


def layer_matrix_4ff(xp, Kx, Ky, exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz_i):
    """``M`` in ``dpsi/dz = i k0 M psi`` for ``psi = (Ex, Ey, Hx, Hy)``.

    All arguments are ``ff x ff`` convolution matrices except ``Kx`` / ``Ky``, which are the
    diagonal order matrices the 2ff path builds. ``ezz_i`` is the *inverse* of the zz block,
    matching how :func:`fieldwave._core.solve_2d.layer_modes_2d` already carries it.
    """
    ff = Kx.shape[0]
    I = xp.astype(xp.eye(ff), Kx.dtype)

    ez_ex, ez_ey = -(ezz_i @ ezx), -(ezz_i @ ezy)      # Ez in terms of the state
    ez_hx, ez_hy = ezz_i @ Ky, -(ezz_i @ Kx)
    hz_ex, hz_ey = -Ky, Kx                             # Hz likewise

    dx_ex, dx_ey = exx + exz @ ez_ex, exy + exz @ ez_ey
    dx_hx, dx_hy = exz @ ez_hx, exz @ ez_hy
    dy_ex, dy_ey = eyx + eyz @ ez_ex, eyy + eyz @ ez_ey
    dy_hx, dy_hy = eyz @ ez_hx, eyz @ ez_hy

    from .solve_1d import _block
    return _block(xp, [
        [Kx @ ez_ex, Kx @ ez_ey, Kx @ ez_hx, Kx @ ez_hy + I],
        [Ky @ ez_ex, Ky @ ez_ey, Ky @ ez_hx - I, Ky @ ez_hy],
        [Kx @ hz_ex - dy_ex, Kx @ hz_ey - dy_ey, -dy_hx, -dy_hy],
        [Ky @ hz_ex + dx_ex, Ky @ hz_ey + dx_ey, dx_hx, dx_hy],
    ])


def mode_flux_z(xp, psi, ff):
    """Time-averaged Poynting ``z`` component of each column of ``psi``.

    This is the raw flux carried by the column as given, which is what a mode *amplitude*
    multiplies — see :func:`solve_uniform_stack`. For deciding direction, divide by the column
    norm first; :func:`split_forward_backward` does.
    """
    Ex, Ey = psi[:ff], psi[ff:2 * ff]
    Hx, Hy = psi[2 * ff:3 * ff], psi[3 * ff:]
    return 0.5 * xp.real(xp.sum(Ex * xp.conj(Hy) - Ey * xp.conj(Hx), axis=0))


def split_forward_backward(xp, M, ff):
    """Eigenmodes of ``M`` split by direction: ``(lam_f, psi_f, lam_b, psi_b)``.

    Forward means energy flowing towards ``+z``, falling back to decay in ``+z`` where the mode
    carries no real flux. That is the same rule
    :func:`fieldwave._core.conventions.boundary_kz` applies to a half-space, and it has to be —
    ``psi`` goes as ``exp(i k0 lam z)``, so ``lam`` sits exactly where that function's ``kz``
    does.

    Flux is the primary test rather than ``Re(lam)`` because an absorbing or gyrotropic layer
    can put a propagating mode on the wrong side of the real axis; the ``Im`` fallback only
    decides modes that carry no flux to speak of.

    "No flux to speak of" is measured *per mode*, against its own norm. Scaling instead by the
    largest flux in the set — the obvious thing — holds up until every mode is evanescent, at
    which point that scale is itself noise and dust at 1e-17 is promoted to real energy flow.
    Every order above the light line is that case, so the set-relative version breaks exactly
    where the multi-order path will live.

    Boolean mask indexing here means this is numpy-only for now; the rest of ``_core`` is
    backend-neutral and this will have to be before it can be dispatched to.
    """
    lam, psi = eig(M)
    flux = mode_flux_z(xp, psi, ff) / xp.sum(xp.abs(psi) ** 2, axis=0)
    forward = xp.where(xp.abs(flux) > _FLAT, flux > 0, xp.imag(lam) > 0)
    backward = xp.logical_not(forward)

    n_f = int(xp.sum(xp.astype(forward, xp.int64)))
    if n_f != 2 * ff:
        raise ValueError(
            f'mode split found {n_f} forward modes, expected {2 * ff}. A degenerate layer can '
            'leave a mode with no flux and an ambiguous decay; this refuses rather than '
            'guessing which half it belongs to.')
    return lam[forward], psi[:, forward], lam[backward], psi[:, backward]


def vacuum_sp_states(xp, kx, ky, ff, order=0):
    """``(psi_s, psi_p)``: unit-amplitude incident plane waves in vacuum, as ``4ff`` states.

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
    kx0, ky0 = float(xp.reshape(xp.asarray(kx), (-1,))[order]), \
        float(xp.reshape(xp.asarray(ky), (-1,))[order])
    kt = (kx0 ** 2 + ky0 ** 2) ** 0.5
    kz = (1.0 - kt ** 2) ** 0.5
    if kt ** 2 >= 1.0:
        raise ValueError(f'order {order} is evanescent in vacuum (|kt| = {kt:.4f}); there is no '
                         'incident plane wave to define.')

    khat = (kx0, ky0, kz)
    shat = (-ky0 / kt, kx0 / kt, 0.0) if kt > 1e-14 else (0.0, 1.0, 0.0)
    cross = lambda a, b: (a[1] * b[2] - a[2] * b[1],   # noqa: E731
                          a[2] * b[0] - a[0] * b[2],
                          a[0] * b[1] - a[1] * b[0])

    out = []
    for E, H in ((shat, cross(khat, shat)), (cross(shat, khat), shat)):
        psi = xp.zeros((4 * ff,), dtype=xp.complex128)
        for slot, val in ((0, E[0]), (1, E[1]), (2, H[0]), (3, H[1])):
            psi[slot * ff + order] = val
        out.append(psi)
    return tuple(out)


def _interface_s(xp, psi1_f, psi1_b, psi2_f, psi2_b):
    """Scattering matrix of the boundary between two media, in their own mode bases.

    The whole transverse state is continuous across the interface, so with amplitudes ``a``,

        Psi1_f a1_f + Psi1_b a1_b = Psi2_f a2_f + Psi2_b a2_b

    and grouping the two *outgoing* sets on the left gives the scattering form directly. No
    ``kz`` sign flip and no gap medium is needed here — the 2ff path needs both because its two
    directions share one eigenvector, and these do not.
    """
    n = psi1_f.shape[1]
    out = xp.concat([psi1_b, -psi2_f], axis=1)
    inn = xp.concat([-psi1_f, psi2_b], axis=1)
    S = xp.linalg.solve(out, inn)
    return S[:n, :n], S[:n, n:], S[n:, :n], S[n:, n:]


def _propagate_s(xp, lam_f, lam_b, k0, d, ff):
    """Scattering matrix of a uniform layer, forward amplitudes referenced at its top.

    **This is the whole reason for the direction split.** Forward modes decay towards ``+z``
    (``Im(lam_f) >= 0``) so ``|exp(i k0 lam_f d)| <= 1``, and backward modes decay towards ``-z``
    so their propagator is bounded too. Both entries are contractions for any thickness, which is
    precisely what the eigen-decomposition transfer cannot promise: it holds the growing and
    decaying halves in one matrix and inverts it, so the growing half sets the conditioning.
    """
    zero = xp.astype(xp.zeros((2 * ff, 2 * ff)), lam_f.dtype)
    X_f = _diag_of(xp, xp.exp(1j * k0 * lam_f * d))
    X_b = _diag_of(xp, xp.exp(-1j * k0 * lam_b * d))
    return zero, X_b, X_f, zero


def _solve_smm(xp, k0, Kx, Ky, layers, ff, psi_vf, psi_vb):
    """Cascade interface and propagation scattering matrices with the Redheffer star."""
    from .scattering import redheffer_star

    I = xp.astype(xp.eye(2 * ff), Kx.dtype)
    S, prev_f, prev_b = None, psi_vf, psi_vb
    for eps_blocks, d in layers:
        lam_f, psi_f, lam_b, psi_b = split_forward_backward(
            xp, layer_matrix_4ff(xp, Kx, Ky, **eps_blocks), ff)
        for part in (_interface_s(xp, prev_f, prev_b, psi_f, psi_b),
                     _propagate_s(xp, lam_f, lam_b, k0, d, ff)):
            S = part if S is None else redheffer_star(xp, I, S, part)
        prev_f, prev_b = psi_f, psi_b
    tail = _interface_s(xp, prev_f, prev_b, psi_vf, psi_vb)
    return tail if S is None else redheffer_star(xp, I, S, tail)


def _forward_amplitudes(xp, incident, psi_vf, psi_vb, ff):
    """Mode amplitudes of an incident state, and a check that it is purely incident.

    Decomposing over the *full* basis rather than solving against ``psi_vf`` alone keeps the
    system square and costs nothing, since the backward half is the thing worth testing: a state
    with any backward content is not an incident wave, and would otherwise be silently projected.
    """
    coef = xp.linalg.solve(xp.concat([psi_vf, psi_vb], axis=1),
                           xp.reshape(incident, (-1, 1)))[:, 0]
    leak = float(xp.max(xp.abs(coef[2 * ff:])))
    if leak > 1e-10 * max(float(xp.max(xp.abs(coef[:2 * ff]))), 1e-300):
        raise ValueError(f'the incident state has backward content ({leak:.2e}); it is not a '
                         'wave arriving from the top. Build it with vacuum_sp_states.')
    return coef[:2 * ff]


def _solve_etm(xp, k0, Kx, Ky, layers, ff, incident, psi_vf, psi_vb):
    """Enhanced transmittance matrix — the 2ff path's stable transfer, carried over to 4ff.

    Same idea as :func:`fieldwave._core.solve_2d.solve_2d`'s ``big_A_i`` / ``big_T`` recursion
    (Moharam et al. 1995), which is why the 2ff transfer is stable while the naive one in
    :func:`_solve_tmm` is not. The growth is not removed by better conditioning; it is arranged
    never to appear.

    Sweeping upward, suppose the state below a layer is ``psi = G t`` for an unknown ``t``.
    Decomposing in that layer's modes gives ``a_f = X_f^-1 P t`` — and ``X_f^-1`` is exactly the
    growing factor. ETM absorbs it into the unknown, ``t' = X_f^-1 P t``, leaving

        G_above = Psi_f + Psi_b @ X_b @ (Q P^-1) @ X_f

    in which ``X_f`` and ``X_b`` appear only as the bounded, decaying factors they are. The
    redefinition is bookkept separately as ``P^-1 X_f`` per layer, so the transmitted amplitude
    can be recovered at the end.
    """
    G = psi_vf
    T_acc = xp.astype(xp.eye(2 * ff), Kx.dtype)
    for eps_blocks, d in reversed(list(layers)):
        lam_f, psi_f, lam_b, psi_b = split_forward_backward(
            xp, layer_matrix_4ff(xp, Kx, Ky, **eps_blocks), ff)
        X_f = _diag_of(xp, xp.exp(1j * k0 * lam_f * d))
        X_b = _diag_of(xp, xp.exp(-1j * k0 * lam_b * d))

        PQ = xp.linalg.solve(xp.concat([psi_f, psi_b], axis=1), G)
        P_i = xp.linalg.inv(PQ[:2 * ff])
        G = psi_f + psi_b @ X_b @ (PQ[2 * ff:] @ P_i) @ X_f
        T_acc = T_acc @ (P_i @ X_f)

    # incident + reflected on the top side must be whatever the recursion says arrives there
    sol = xp.linalg.solve(xp.concat([psi_vb, -G], axis=1),
                          xp.reshape(-incident, (-1, 1)))[:, 0]
    return psi_vb @ sol[:2 * ff], psi_vf @ (T_acc @ sol[2 * ff:])


def _solve_tmm(xp, k0, Kx, Ky, layers, ff, incident, psi_vf, psi_vb):
    """Eigen-decomposition transfer, kept as a second opinion. Refuses where it would blow up."""
    P = xp.astype(xp.eye(4 * ff), Kx.dtype)
    for eps_blocks, d in layers:
        lam_f, psi_f, lam_b, psi_b = split_forward_backward(
            xp, layer_matrix_4ff(xp, Kx, Ky, **eps_blocks), ff)
        lam = xp.concat([lam_f, lam_b])
        exponent = k0 * float(xp.max(xp.abs(xp.imag(lam)))) * float(d)
        if exponent > _MAX_EXPONENT:
            raise ValueError(
                f'layer exponent {exponent:.1f} is past what the eigen-decomposition transfer '
                "can carry in float64. Use connecting_algo='SMM', which is bounded for any "
                'thickness.')
        Psi = xp.concat([psi_f, psi_b], axis=1)
        E = _diag_of(xp, xp.exp(1j * k0 * lam * d))
        P = (Psi @ E @ xp.linalg.inv(Psi)) @ P

    # Matching at both interfaces at once: incident + reflected on the input side must transfer
    # to transmitted only on the output side, so P (a + psi_vb r) = psi_vf t.
    A = xp.concat([P @ psi_vb, -psi_vf], axis=1)
    sol = xp.linalg.solve(A, xp.reshape(-(P @ incident), (-1, 1)))[:, 0]
    return psi_vb @ sol[:2 * ff], psi_vf @ sol[2 * ff:]


def solve_uniform_stack(wavelength, kx, ky, layers, ff, incident, connecting_algo='ETM'):
    """``(R, T)`` for one incident state, through a stack of uniform anisotropic layers.

    ``layers`` is ``[(eps_blocks, thickness), ...]`` where ``eps_blocks`` is the mapping
    :func:`layer_matrix_4ff` takes. ``incident`` is a ``4ff`` state — see
    :func:`vacuum_sp_states`. Vacuum on both sides, and note it is *not* special-cased: vacuum
    is a layer with ``eps = I`` whose forward and backward halves come from the same mode split
    as any other, which is what would make an anisotropic substrate free here. The 2ff path
    cannot say that, and neither can RETICOLO.

    ``R`` and ``T`` are fluxes of the *reconstructed* reflected and transmitted fields, not sums
    of per-mode fluxes. Those agree only when the modes are flux-orthogonal, which holds for a
    lossless medium and stops holding exactly where it would be least noticed.

    ``connecting_algo`` selects how layers are joined. Three, and the choice matters here in a
    way it does not in the 2ff path, where both shipped options are already stable:

    * ``'ETM'`` (default) is the enhanced-transmittance recursion, the same technique that makes
      the 2ff transfer stable. Growth never appears rather than being cancelled. Measured the
      cheapest of the three — 0.94x the naive transfer on a 40-layer stack, against SMM's 1.13x
      — and identical to SMM at 1.2e-15, which is why it is the default.
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

    vac = _identity_blocks(xp, ff, Kx.dtype)
    _, psi_vf, _, psi_vb = split_forward_backward(xp, layer_matrix_4ff(xp, Kx, Ky, **vac), ff)

    if connecting_algo == 'SMM':
        S11, _, S21, _ = _solve_smm(xp, k0, Kx, Ky, layers, ff, psi_vf, psi_vb)
        a_inc = _forward_amplitudes(xp, incident, psi_vf, psi_vb, ff)
        reflected, transmitted = psi_vb @ (S11 @ a_inc), psi_vf @ (S21 @ a_inc)
    elif connecting_algo == 'ETM':
        reflected, transmitted = _solve_etm(xp, k0, Kx, Ky, layers, ff, incident, psi_vf, psi_vb)
    elif connecting_algo == 'TMM':
        reflected, transmitted = _solve_tmm(xp, k0, Kx, Ky, layers, ff, incident, psi_vf, psi_vb)
    else:
        raise ValueError(
            f"connecting_algo must be 'SMM', 'ETM' or 'TMM', got {connecting_algo!r}")

    def flux(v):
        return float(mode_flux_z(xp, xp.reshape(v, (-1, 1)), ff)[0])

    inc = flux(incident)
    return -flux(reflected) / inc, flux(transmitted) / inc


def _diag_of(xp, v):
    from .solve_1d import _diag
    return _diag(xp, xp.reshape(xp.asarray(v), (-1,)))


def _identity_blocks(xp, ff, dtype):
    I = xp.astype(xp.eye(ff), dtype)
    O = xp.astype(xp.zeros((ff, ff)), dtype)
    return dict(exx=I, exy=O, exz=O, eyx=O, eyy=I, eyz=O, ezx=O, ezy=O, ezz_i=I)
