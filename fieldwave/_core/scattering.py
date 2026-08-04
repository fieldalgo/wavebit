"""Scattering-matrix method (SMM) — the second connecting algorithm, single-source.

The *connecting algorithm* is how per-layer eigenmodes are joined into a stack response. TMM
(:mod:`fieldwave._core.solve_1d` / :mod:`fieldwave._core.solve_2d`) propagates an enhanced
transmittance matrix; SMM composes per-layer scattering matrices with the Redheffer star
product. Both consume the same modes — this module imports the mode builders from the TMM
solves rather than restating the eigen-algebra — and both use the same boundary conventions,
so ``connecting_algo='SMM'`` returns the same efficiencies as ``'TMM'``, not a second opinion.

Why have it: a layer's scattering matrix depends only on that layer and the shared gap medium,
and the star product is associative. Layer responses can therefore be built once and cascaded,
reused, or reordered without touching their neighbours — where TMM's recursion carries a state
that already contains everything below the current layer. SMM is also the formulation most of
the literature states its results in, which makes it the natural cross-check on the TMM path.

What it is *not*: a rescue from TMM instability. This TMM is the enhanced-transmittance form,
so it is already stable; both algorithms only ever evaluate ``exp(-k0*q*d)`` with ``Re(q) >= 0``
(principal root) and only inside bounded products. Measured on thick, strongly evanescent and
high-truncation stacks the two agree to ~1e-14 with no energy drift either way, and SMM costs
~1.1-1.4x TMM (the per-layer ``eig`` dominates both).

Formulation, in the conventions this codebase already uses (see :func:`solve_1d`):

* Every region's ``(W, V)`` describe its *downward* modes: a field with amplitudes ``c`` has
  primary variable ``W @ c`` and partner ``V @ c``; an upward wave flips the partner's sign.
  With that single convention the half-space scattering matrices below are derived, not
  transcribed — no per-region ``kz`` sign flip is needed.
* Layers are joined through a fictitious *gap medium* with ``eps_gap = 1 + kx^2 + ky^2``, so
  ``kz_gap == 1`` exactly and the gap's ``V`` can never be singular (the standard trick; the
  gap cancels out of the composed result whatever it is, as long as it is used everywhere).
* Incidence enters as mode amplitudes directly: ``E_s = sin(psi)`` and ``H_p = i*n_top*cos(psi)``
  for the coupled path, unit primary amplitude for 1D TE/TM. These are the same incident
  amplitudes the TMM final matching solves against.

Not supported: field reconstruction. ``calculate_field`` needs the per-layer amplitude
propagators that the TMM recursion produces as a by-product; the SMM path returns
``layer_info_list = []`` and :mod:`fieldwave._core.fields` raises a clear error on it.
"""
from .backend import array_namespace
from .conventions import boundary_kz
from .solve_1d import (_block, _diag, _pack, _unit, layer_modes_1d, layer_modes_1d_conical,
                       sp_efficiencies, sp_modes_1d_conical)
from .solve_2d import layer_modes_2d, sp_modes_2d

_PI = 3.141592653589793


# --------------------------------------------------------------------------- S-matrix algebra
# A scattering matrix is carried as the 4-tuple of blocks ``(S11, S12, S21, S22)`` relating
# the waves entering at each port to those leaving it:
#     [out_top ; out_bot] = S @ [in_top ; in_bot]
# where ``in_top`` travels down into the structure and ``in_bot`` up into it.

def _ab_matrices(inv, W_to, V_to, W_from, V_from):
    """Interface coefficients ``(A, B)`` for a boundary crossed *into* the ``to`` basis.

    From continuity of the primary field and its partner across the interface,
    ``c_to_down = (A @ c_from_down + B @ c_from_up)/2`` (and the mirrored relation for the
    upward amplitudes), with ``A = W_to^-1 W_from + V_to^-1 V_from`` and ``B`` the difference.
    """
    P = inv(W_to) @ W_from
    Q = inv(V_to) @ V_from
    return P + Q, P - Q


def _s_layer(inv, A, B, X):
    """Scattering matrix of one layer embedded between gap half-spaces.

    ``A``/``B`` describe the gap -> layer interface and ``X = diag(exp(-k0*q*d))`` the modal
    propagation across it. The layer is symmetric, so ``S22 == S11`` and ``S21 == S12``.
    """
    A_i = inv(A)
    XB = X @ B
    common = inv(A - XB @ A_i @ X @ B)
    S11 = common @ (XB @ A_i @ X @ A - B)
    S12 = common @ X @ (A - B @ A_i @ B)
    return S11, S12, S12, S11


def _s_reflection(inv, W_top, V_top, W_gap, V_gap):
    """Scattering matrix of the superstrate -> gap interface (port 1 in the superstrate)."""
    A, B = _ab_matrices(inv, W_gap, V_gap, W_top, V_top)
    A_i = inv(A)
    return -A_i @ B, 2 * A_i, 0.5 * (A - B @ A_i @ B), B @ A_i


def _s_transmission(inv, W_bot, V_bot, W_gap, V_gap):
    """Scattering matrix of the gap -> substrate interface (port 2 in the substrate)."""
    A, B = _ab_matrices(inv, W_gap, V_gap, W_bot, V_bot)
    A_i = inv(A)
    return B @ A_i, 0.5 * (A - B @ A_i @ B), 2 * A_i, -A_i @ B


def redheffer_star(inv, I, S_a, S_b):
    """Redheffer star product ``S_a * S_b`` — cascade two scattering matrices."""
    a11, a12, a21, a22 = S_a
    b11, b12, b21, b22 = S_b
    D_i = inv(I - b11 @ a22)
    F_i = inv(I - a22 @ b11)
    return (a11 + a12 @ D_i @ b11 @ a21,
            a12 @ D_i @ b12,
            b21 @ F_i @ a21,
            b22 + b21 @ F_i @ a22 @ b12)


def _n_layers(thickness):
    return len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]


def _prepare(epx_conv_all, kx, ky=None):
    """Namespace, complex working dtype and complex-promoted ``kx`` / ``ky``."""
    xp = array_namespace(epx_conv_all)
    kx = xp.asarray(kx)
    if not xp.isdtype(kx.dtype, 'complex floating'):
        kx = xp.astype(kx, xp.complex128)
        if ky is not None:
            ky = xp.astype(xp.asarray(ky), xp.complex128)
    elif ky is not None:
        ky = xp.asarray(ky)
    return xp, kx.dtype, kx, ky


# --------------------------------------------------------------------------------- 1D TE / TM

def smm_1d(pol, wavelength, kx, n_top, n_bot, theta, thickness,
           epx_conv_all, epy_conv_all, epz_conv_i_all,
           same_material=None, same_material_rtol=1e-3, use_pinv=False, full=False):
    """Solve a 1D grating stack by scattering matrices — SMM twin of :func:`solve_1d`.

    Same arguments and same returns as :func:`fieldwave._core.solve_1d.solve_1d`, so the two are
    interchangeable behind ``connecting_algo``. ``same_material`` / ``same_material_rtol`` are
    accepted and ignored: the S-matrix recursion never inverts an interface matrix that a
    substrate-matched layer would make singular, so the short-circuit they drive has no
    counterpart here.

    With ``full=True`` the returned mapping carries ``layer_info_list = []`` and ``T1 = None``:
    field reconstruction is a TMM-only capability (see the module docstring).
    """
    xp, C, kx, _ = _prepare(epx_conv_all, kx)
    _inv = xp.linalg.pinv if use_pinv else xp.linalg.inv

    ff_x = kx.shape[0]
    k0 = 2 * _PI / wavelength
    theta = 0.0 if theta is None else theta
    I = xp.astype(xp.eye(ff_x), C)

    # Boundary branches come from the same place the TMM path takes them, so the two
    # algorithms answer identically.
    kz_top = boundary_kz(xp, n_top ** 2, kx ** 2)
    kz_bot = boundary_kz(xp, n_bot ** 2, kx ** 2)
    eps_bot = n_bot ** 2

    # Gap medium: eps chosen per order so kz_gap == 1, which keeps V_gap trivially invertible.
    eps_gap = 1 + kx ** 2

    if pol == 0:      # TE — primary variable E_y, partner H_x
        V_top = -1j * _diag(xp, kz_top)
        V_bot = -1j * _diag(xp, kz_bot)
        V_gap = -1j * I
    elif pol == 1:    # TM — primary variable H_y, partner scaled by eps
        V_top = -1j * _diag(xp, kz_top / n_top ** 2)
        V_bot = -1j * _diag(xp, kz_bot / eps_bot)
        V_gap = -1j * _diag(xp, 1 / eps_gap)
    else:
        raise ValueError(f'pol must be 0 (TE) or 1 (TM), got {pol!r}')
    W_top = W_bot = W_gap = I    # homogeneous half-spaces are diagonal in this basis

    # --- cascade: superstrate interface, every layer top-to-bottom, substrate interface ----
    S = _s_reflection(_inv, W_top, V_top, W_gap, V_gap)
    for layer_index in range(_n_layers(thickness)):
        W, V, q = layer_modes_1d(xp, _inv, pol, kx, I, epx_conv_all[layer_index],
                                 epy_conv_all[layer_index], epz_conv_i_all[layer_index], C)
        X = _diag(xp, xp.exp(-k0 * q * thickness[layer_index]))
        A, B = _ab_matrices(_inv, W, V, W_gap, V_gap)
        S = redheffer_star(_inv, I, S, _s_layer(_inv, A, B, X))
    S = redheffer_star(_inv, I, S, _s_transmission(_inv, W_bot, V_bot, W_gap, V_gap))

    # --- incidence -> amplitudes -> efficiency --------------------------------------------
    delta = xp.reshape(xp.astype(_unit(xp, ff_x, ff_x // 2), C), (ff_x, 1))
    R = xp.reshape(S[0] @ delta, (1, ff_x))
    Tt = xp.reshape(S[2] @ delta, (1, ff_x))

    cos_t = xp.cos(xp.asarray(theta, dtype=C))
    de_ri = xp.real(R * xp.conj(R) * xp.real(kz_top / (n_top * cos_t)))
    if pol == 0:
        de_ti = xp.real(Tt * xp.conj(Tt) * xp.real(kz_bot / (n_top * cos_t)))
    else:
        de_ti = xp.real(Tt * xp.conj(Tt)
                        * xp.real(xp.conj(kz_bot) / n_bot ** 2 / (cos_t / n_top)))

    if not full:
        return de_ri, de_ti

    zeros_amp = xp.zeros_like(R)
    zeros_eff = xp.zeros_like(de_ri)
    if pol == 0:
        res = _pack(R, zeros_amp, Tt, zeros_amp, de_ri, zeros_eff, de_ti, zeros_eff)
    else:
        res = _pack(zeros_amp, R, zeros_amp, Tt, zeros_eff, de_ri, zeros_eff, de_ti)
    return {'res': res, 'layer_info_list': [], 'T1': None}


# ------------------------------------------------------------------- coupled s/p (conical, 2D)

def _smm_sp(xp, C, inv, wavelength, kx, ky, n_top, n_bot, theta, psi, thickness,
            layer_modes, full):
    """Shared SMM cascade + matching for the coupled s/p paths (conical and 2D).

    ``layer_modes(layer_index)`` returns that layer's ``(big_W, big_V, q)`` in the s/p boundary
    basis; everything else — half-spaces, gap, cascade, incidence, efficiencies — is common.
    """
    ff_x = kx.shape[0]
    ff_y = ky.shape[0]
    ff_xy = ff_x * ff_y
    k0 = 2 * _PI / wavelength
    theta = 0.0 if theta is None else theta

    I = xp.astype(xp.eye(ff_xy), C)
    O = xp.astype(xp.zeros((ff_xy, ff_xy)), C)
    big_I = xp.astype(xp.eye(2 * ff_xy), C)
    ky_col = xp.reshape(ky, (-1, 1))

    k_par_sq = kx ** 2 + ky_col ** 2
    kz_top = xp.reshape(boundary_kz(xp, n_top ** 2, k_par_sq), (-1,))
    kz_bot = xp.reshape(boundary_kz(xp, n_bot ** 2, k_par_sq), (-1,))
    eps_bot = n_bot ** 2
    eps_gap = xp.reshape(1 + kx ** 2 + ky_col ** 2, (-1,))    # kz_gap == 1 by construction

    Kz_top = _diag(xp, kz_top)
    Kz_bot = _diag(xp, kz_bot)

    # Half-space and gap modes in the (E_s, F_p) / (H_s, H_p) boundary basis — the same form
    # the TMM substrate seed (big_F, big_G) has.
    W_top = _block(xp, [[I, O], [O, -1j * Kz_top / n_top ** 2]])
    V_top = _block(xp, [[-1j * Kz_top, O], [O, I]])
    W_bot = _block(xp, [[I, O], [O, -1j * Kz_bot / eps_bot]])
    V_bot = _block(xp, [[-1j * Kz_bot, O], [O, I]])
    W_gap = _block(xp, [[I, O], [O, -1j * _diag(xp, 1 / eps_gap)]])
    V_gap = _block(xp, [[-1j * I, O], [O, I]])

    S = _s_reflection(inv, W_top, V_top, W_gap, V_gap)
    for layer_index in range(_n_layers(thickness)):
        big_W, big_V, q = layer_modes(layer_index)
        big_X = _diag(xp, xp.exp(-k0 * q * thickness[layer_index]))
        A, B = _ab_matrices(inv, big_W, big_V, W_gap, V_gap)
        S = redheffer_star(inv, big_I, S, _s_layer(inv, A, B, big_X))
    S = redheffer_star(inv, big_I, S, _s_transmission(inv, W_bot, V_bot, W_gap, V_gap))

    # --- incidence ------------------------------------------------------------------------
    # Amplitudes are (E_s, H_p) by construction of W/V above, so the incident column is the
    # incident field itself: E_s = sin(psi), H_p = i*n_top*cos(psi).
    delta = xp.reshape(xp.astype(_unit(xp, ff_xy, ff_xy // 2), C), (ff_xy, 1))

    def incident(psi_v):
        s_p = xp.sin(xp.asarray(psi_v, dtype=C))
        c_p = xp.cos(xp.asarray(psi_v, dtype=C))
        return xp.concat([s_p * delta, -1j * n_top * c_p * delta], axis=0)

    # psi as asked, then the pure-TE / pure-TM incidences (free: same scattering matrix).
    cols = [incident(psi)] + ([incident(_PI / 2), incident(0.0)] if full else [])
    c_inc = xp.concat(cols, axis=1)

    c_ref = S[0] @ c_inc
    c_trn = S[2] @ c_inc

    n_inc = len(cols)

    def as_orders(M):
        """(ff_xy, n_inc) amplitude columns -> (n_inc, ff_y, ff_x)."""
        return xp.reshape(xp.matrix_transpose(M), (n_inc, ff_y, ff_x))

    # The reflected p amplitude is reported as its magnetic field, which points the other way
    # for the upward wave; s and the transmitted pair need no such flip.
    R_s = as_orders(c_ref[:ff_xy, :])
    R_p = as_orders(-c_ref[ff_xy:, :])
    T_s = as_orders(c_trn[:ff_xy, :])
    T_p = as_orders(c_trn[ff_xy:, :])

    kz_top_2d = xp.reshape(kz_top, (ff_y, ff_x))
    kz_bot_2d = xp.reshape(kz_bot, (ff_y, ff_x))
    denom = n_top * xp.cos(xp.asarray(theta, dtype=C))
    eff = sp_efficiencies(xp, kz_top_2d, kz_bot_2d, denom, n_top, n_bot, R_s, R_p, T_s, T_p)

    if not full:
        return eff[0][0] + eff[1][0], eff[2][0] + eff[3][0]

    packed = [_pack(R_s[i], R_p[i], T_s[i], T_p[i], *(e[i] for e in eff)) for i in range(3)]
    return {'res': packed[0], 'res_te_inc': packed[1], 'res_tm_inc': packed[2],
            'layer_info_list': [], 'T1': None}


def smm_1d_conical(wavelength, kx, ky, n_top, n_bot, theta, psi, thickness,
                   epx_conv_all, epy_conv_all, epz_conv_i_all,
                   same_material=None, same_material_rtol=1e-3, use_pinv=False, full=False):
    """Solve a conical 1D grating by scattering matrices — SMM twin of :func:`solve_1d_conical`.

    Same arguments and returns; ``same_material`` is accepted and ignored (see :func:`smm_1d`).
    """
    xp, C, kx, ky = _prepare(epx_conv_all, kx, ky)
    _inv = xp.linalg.pinv if use_pinv else xp.linalg.inv

    ff_x = kx.shape[0]
    I = xp.astype(xp.eye(ff_x), C)
    ky_col = xp.reshape(ky, (-1, 1))
    varphi = xp.reshape(xp.atan(ky_col / kx), (-1,))

    Kx = _diag(xp, kx)
    ky_vec = ky[0] * xp.astype(xp.ones((ff_x,)), C)      # ff_y == 1: Ky = diag(ky0 * 1)
    Ky = _diag(xp, ky_vec)
    Kx2 = _diag(xp, kx * kx)
    Ky2 = _diag(xp, ky_vec * ky_vec)
    F_c = _diag(xp, xp.cos(varphi))
    F_s = _diag(xp, xp.sin(varphi))

    def layer_modes(layer_index):
        W1, W2, q1, q2, V_11, V_12, V_21, V_22 = layer_modes_1d_conical(
            xp, _inv, Kx, Ky, Kx2, Ky2, I, epx_conv_all[layer_index],
            epy_conv_all[layer_index], epz_conv_i_all[layer_index], C)
        big_W, big_V = sp_modes_1d_conical(xp, F_c, F_s, W1, W2, V_11, V_12, V_21, V_22)
        return big_W, big_V, xp.concat([q1, q2])

    return _smm_sp(xp, C, _inv, wavelength, kx, ky, n_top, n_bot, theta, psi, thickness,
                   layer_modes, full)


def smm_2d(wavelength, kx, ky, n_top, n_bot, theta, psi, thickness,
           epx_conv_all, epy_conv_all, epz_conv_i_all,
           epxy_conv_all=None, epyx_conv_all=None,
           same_material=None, same_material_rtol=1e-3, use_pinv=False, full=False):
    """Solve a 2D grating stack by scattering matrices — SMM twin of :func:`solve_2d`.

    Same arguments and returns; ``same_material`` is accepted and ignored (see :func:`smm_1d`).
    """
    xp, C, kx, ky = _prepare(epx_conv_all, kx, ky)
    _inv = xp.linalg.pinv if use_pinv else xp.linalg.inv

    ff_x = kx.shape[0]
    ff_y = ky.shape[0]
    ff_xy = ff_x * ff_y
    I = xp.astype(xp.eye(ff_xy), C)
    O = xp.astype(xp.zeros((ff_xy, ff_xy)), C)
    ky_col = xp.reshape(ky, (-1, 1))
    varphi = xp.reshape(xp.atan(ky_col / kx), (-1,))

    kx_full = xp.reshape(xp.broadcast_to(kx[None, :], (ff_y, ff_x)), (-1,))
    ky_full = xp.reshape(xp.broadcast_to(ky_col, (ff_y, ff_x)), (-1,))
    Kx = _diag(xp, kx_full)
    Ky = _diag(xp, ky_full)
    F_c = _diag(xp, xp.cos(varphi))
    F_s = _diag(xp, xp.sin(varphi))

    def layer_modes(layer_index):
        epxy_conv = O if epxy_conv_all is None else epxy_conv_all[layer_index]
        epyx_conv = O if epyx_conv_all is None else epyx_conv_all[layer_index]
        W, V, q = layer_modes_2d(xp, Kx, Ky, I, epx_conv_all[layer_index],
                                 epy_conv_all[layer_index], epz_conv_i_all[layer_index],
                                 epxy_conv, epyx_conv, C)
        big_W, big_V = sp_modes_2d(xp, W, V, F_c, F_s, ff_xy)
        return big_W, big_V, q

    return _smm_sp(xp, C, _inv, wavelength, kx, ky, n_top, n_bot, theta, psi, thickness,
                   layer_modes, full)
