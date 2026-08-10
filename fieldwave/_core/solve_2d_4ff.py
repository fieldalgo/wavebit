"""First-order (4ff) layer solve — the formulation a z-coupled tensor needs.

**Status: the operator and the mode split are done and pinned; the stack solve here is correct
but not yet numerically stable, nothing dispatches to it, and there is no field
reconstruction.** See "What is not here" at the bottom before building on it.

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

What is not here, in the order it will be wanted:

* **Numerical stability.** :func:`solve_uniform_stack` propagates a layer as
  ``Psi exp(i k0 lam d) Psi^-1``, which grows without bound for a thick layer — the classic
  failure the 2ff path avoids with its ``X`` / ``A`` / ``B`` recursion. The forward/backward
  split needed for a scattering-matrix recursion is already done here; the recursion is not.
  :func:`solve_uniform_stack` raises rather than returning a number it cannot stand behind.
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

# exp(i k0 lam d) beyond this is past the point where the eigen-decomposition transfer means
# anything in float64; a scattering-matrix recursion is the fix, not a larger bound.
_MAX_EXPONENT = 300.0


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


def solve_uniform_stack(wavelength, kx, ky, layers, ff):
    """``[(R, T), ...]`` per incident vacuum mode, for a stack of uniform anisotropic layers.

    ``layers`` is ``[(eps_blocks, thickness), ...]`` where ``eps_blocks`` is the mapping
    :func:`layer_matrix_4ff` takes. Vacuum on both sides — and note it is *not* special-cased:
    vacuum is a layer with ``eps = I`` and the mode split picks its forward and backward halves
    like any other, which is what would make an anisotropic substrate free here. The 2ff path
    cannot say that, and neither can RETICOLO.

    **Correct but not stable.** The transfer is ``Psi exp(i k0 lam d) Psi^-1`` per layer, which
    overflows for a thick or strongly evanescent layer. Rather than return a number that has
    lost its meaning, this raises when the exponent leaves float64's useful range. Fixing it is
    the scattering-matrix recursion, for which the split above is the prerequisite.
    """
    xp = array_namespace(kx)
    k0 = 2 * 3.141592653589793 / wavelength
    Kx, Ky = _diag_of(xp, kx), _diag_of(xp, ky)

    vac = _identity_blocks(xp, ff, Kx.dtype)
    _, psi_vf, _, psi_vb = split_forward_backward(xp, layer_matrix_4ff(xp, Kx, Ky, **vac), ff)

    P = xp.astype(xp.eye(4 * ff), Kx.dtype)
    for eps_blocks, d in layers:
        M = layer_matrix_4ff(xp, Kx, Ky, **eps_blocks)
        lam_f, psi_f, lam_b, psi_b = split_forward_backward(xp, M, ff)
        lam = xp.concat([lam_f, lam_b])
        exponent = k0 * float(xp.max(xp.abs(xp.imag(lam)))) * float(d)
        if exponent > _MAX_EXPONENT:
            raise ValueError(
                f'layer exponent {exponent:.1f} is past what the eigen-decomposition transfer '
                'can carry in float64. This needs the scattering-matrix recursion, which is '
                'not implemented here — see the module docstring.')
        Psi = xp.concat([psi_f, psi_b], axis=1)
        E = _diag_of(xp, xp.exp(1j * k0 * lam * d))
        P = (Psi @ E @ xp.linalg.inv(Psi)) @ P

    # Matching at the two interfaces at once: incident + reflected on the input side must
    # transfer to transmitted only on the output side, so P (a + psi_vb r) = psi_vf t.
    flux_f, flux_b = mode_flux_z(xp, psi_vf, ff), mode_flux_z(xp, psi_vb, ff)
    A = xp.concat([P @ psi_vb, -psi_vf], axis=1)

    out = []
    for j in range(2 * ff):
        sol = xp.linalg.solve(A, xp.reshape(-(P @ psi_vf[:, j]), (-1, 1)))[:, 0]
        r, t = sol[:2 * ff], sol[2 * ff:]
        inc = float(flux_f[j])
        out.append((float(-xp.sum(flux_b * xp.abs(r) ** 2) / inc),
                    float(xp.sum(flux_f * xp.abs(t) ** 2) / inc)))
    return out


def _diag_of(xp, v):
    from .solve_1d import _diag
    return _diag(xp, xp.reshape(xp.asarray(v), (-1,)))


def _identity_blocks(xp, ff, dtype):
    I = xp.astype(xp.eye(ff), dtype)
    O = xp.astype(xp.zeros((ff, ff)), dtype)
    return dict(exx=I, exy=O, exz=O, eyx=O, eyy=I, eyz=O, ezx=O, ezy=O, ezz_i=I)
