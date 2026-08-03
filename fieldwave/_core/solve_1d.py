"""Single-source 1D (TE / TM) RCWA solve — Phase 3 of the rewrite (see ``<a scratch note outside this repository>``).

One backend-agnostic implementation of the transfer-matrix solve for a 1D grating stack,
written against the Array-API namespace (``numpy`` / ``jax.numpy`` / ``torch``). It is the
port of the three triplicated ``transfer_1d_1..4`` chains into a single source: the only
per-backend primitive it reaches for is :func:`fieldwave._core.backend.eig`, and every
sign/branch choice comes from :mod:`fieldwave._core.conventions`.

Boundary contract — it consumes what the legacy ``solve_1d`` consumes (the per-layer
convolution matrices already built by the factorization stage, plus the incidence/geometry
scalars) and returns the same ``(de_ri, de_ti)`` diffraction efficiencies. Feeding it the
exact inputs the legacy path uses reproduces the legacy result bit-for-bit; feeding it JAX /
PyTorch arrays runs the identical algebra with autodiff intact.

Stages (matching the legacy chain, kept as one function to make bit-parity auditable):
  1. boundary init      — kz for efficiency, substrate admittance G, F = T = I
  2. per-layer eig+couple— Redheffer/transfer propagation from the bottom layer up
  3. final matching      — incident field -> R, T -> diffraction efficiency
"""
from .backend import array_namespace, eig
from .conventions import outgoing_sqrt


def _diag(xp, v):
    """Construct a diagonal matrix from a 1D vector (Array-API ``diag`` build form)."""
    return xp.diag(v)


def _diagonal(xp, M):
    """Extract the diagonal of a 2D matrix as a 1D vector (Array-API standard)."""
    return xp.linalg.diagonal(M)


def solve_1d(pol, wavelength, kx, n_top, n_bot, theta, thickness,
             epx_conv_all, epy_conv_all, epz_conv_i_all,
             same_material=None, same_material_rtol=1e-3):
    """Solve a 1D grating stack for reflection/transmission diffraction efficiency.

    Args:
        pol: ``0`` (TE) or ``1`` (TM).
        wavelength: incident wavelength (sets ``k0 = 2*pi/wavelength``).
        kx: in-plane wavevectors of the retained orders (length ``ff = 2*fto+1``).
        n_top, n_bot: superstrate / substrate refractive indices.
        theta: incident polar angle (``None`` -> 0; matches the legacy ``cos(theta)`` factor).
        thickness: per-layer thicknesses, top-to-bottom (``ucell``/first-axis order).
        epx_conv_all, epy_conv_all, epz_conv_i_all: per-layer convolution matrices,
            indexed ``[layer]`` on the first axis, as produced by the factorization stage.
        same_material: optional per-layer bool sequence marking layers that are uniform and
            of the same material as the medium just below (interface matching skipped,
            propagation only). This is a discrete *geometry* fact, so supplying it keeps the
            solve traceable/jittable under JAX autodiff. When ``None`` it is auto-detected from
            the (concrete) convolution matrices — the eager NumPy path that reproduces the
            legacy result; do not rely on auto-detection while differentiating w.r.t. the
            permittivity, where the matrices are abstract tracers.
        same_material_rtol: tolerance for the auto-detection above.

    Returns:
        ``(de_ri, de_ti)`` — reflected / transmitted efficiency per order, shape ``(1, ff)``.
    """
    xp = array_namespace(epx_conv_all)
    kx = xp.asarray(kx)
    if not xp.isdtype(kx.dtype, 'complex floating'):
        # Promote a real kx to the complex working dtype the algebra needs.
        kx = xp.astype(kx, xp.complex128)
    C = kx.dtype

    ff_x = kx.shape[0]
    k0 = 2 * 3.141592653589793 / wavelength
    theta = 0.0 if theta is None else theta

    I = xp.astype(xp.eye(ff_x), C)

    # --- stage 1: boundary init (port of transfer_1d_1) --------------------------------
    # kz for the diffraction-efficiency factors: principal root, conjugated branch (raw n).
    # Scalars (n_top/n_bot) promote against the complex kx, so no explicit cast is needed.
    kz_top = xp.conj((n_top ** 2 - kx ** 2) ** 0.5)
    kz_bot = xp.conj((n_bot ** 2 - kx ** 2) ** 0.5)

    # Substrate admittance from the single outgoing-wave kz branch (lossless AND lossy).
    eps_bot = n_bot ** 2
    kz_sub = outgoing_sqrt(xp, eps_bot - kx ** 2 - 1e-20j)
    if pol == 0:      # TE
        G = 1j * _diag(xp, kz_sub)
    elif pol == 1:    # TM
        G = 1j * _diag(xp, kz_sub / eps_bot)
    else:
        raise ValueError(f'pol must be 0 (TE) or 1 (TM), got {pol!r}')

    F = I
    T = I

    # eps of the medium just below the current layer (substrate uses conj(n)^2 to match
    # the layer convention); drives the same-material short-circuit. Kept a plain scalar.
    eps_below = complex(n_bot).conjugate() ** 2

    # --- stage 2: per-layer eig + transfer coupling (transfer_1d_2 / transfer_1d_3) ----
    n_layers = len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]
    for layer_index in range(n_layers)[::-1]:
        epx_conv = epx_conv_all[layer_index]
        epy_conv = epy_conv_all[layer_index]
        epz_conv_i = epz_conv_i_all[layer_index]
        d = thickness[layer_index]

        if pol == 0:  # TE
            A = _diag(xp, kx) @ _diag(xp, kx) - epy_conv
            w, W = eig(A)
            q = xp.astype(w, C) ** 0.5
            V = W * q
        else:         # TM
            B = kx[:, None] * epz_conv_i * kx[None, :] - I
            w, W = eig(epx_conv @ B)
            q = xp.astype(w, C) ** 0.5
            V = xp.linalg.inv(epx_conv) @ W * q

        # Same-material short-circuit. A uniform layer of the medium below has diagonal
        # eps_conv ~= eps_below * I. The decision is discrete geometry: take it from the
        # supplied flags when given (keeps the trace intact under autodiff), else auto-detect
        # from the concrete matrices (eager NumPy path -> legacy parity).
        if same_material is None:
            layer_eps_diag = xp.mean(_diagonal(xp, epx_conv))
            is_same = bool(abs(complex(layer_eps_diag) - complex(eps_below))
                           <= same_material_rtol * abs(complex(eps_below)))
            eps_below = layer_eps_diag
        else:
            is_same = bool(same_material[layer_index])

        X = _diag(xp, xp.exp(-k0 * q * d))
        if is_same:
            # No interface reflection: propagate only, in the physical basis.
            W_i = xp.linalg.inv(W)
            T = T @ (W @ X @ W_i)
        else:
            W_i = xp.linalg.inv(W)
            V_i = xp.linalg.inv(V)
            a = 0.5 * (W_i @ F + V_i @ G)
            b = 0.5 * (W_i @ F - V_i @ G)
            a_i = xp.linalg.inv(a)
            F = W @ (I + X @ b @ a_i @ X)
            G = V @ (I - X @ b @ a_i @ X)
            T = T @ a_i @ X

    # --- stage 3: final matching -> efficiency (transfer_1d_4) -------------------------
    Kz_top = _diag(xp, kz_top)
    delta = xp.astype(_unit(xp, ff_x, ff_x // 2), C)
    cos_t = xp.cos(xp.astype(xp.asarray(theta), C))

    if pol == 0:  # TE
        inc = 1j * n_top * cos_t * delta
        T1 = xp.linalg.inv(G + 1j * Kz_top @ F) @ (1j * Kz_top @ delta + inc)
    else:         # TM
        Kz_scaled = Kz_top / (n_top ** 2)
        inc = 1j * delta * cos_t / n_top
        T1 = xp.linalg.inv(G + 1j * Kz_scaled @ F) @ (1j * Kz_scaled @ delta + inc)

    R = (F @ T1 - delta)[None, :]
    Tt = (T @ T1)[None, :]

    de_ri = xp.real(R * xp.conj(R) * xp.real(kz_top / (n_top * cos_t)))
    if pol == 0:  # TE
        de_ti = xp.real(Tt * xp.conj(Tt) * xp.real(kz_bot / (n_top * cos_t)))
    else:         # TM
        de_ti = xp.real(Tt * xp.conj(Tt)
                        * xp.real(xp.conj(kz_bot) / n_bot ** 2 / (cos_t / n_top)))

    return de_ri, de_ti


def _unit(xp, n, i):
    """Unit vector ``e_i`` of length ``n`` (real)."""
    v = [0.0] * n
    v[i] = 1.0
    return xp.asarray(v)
