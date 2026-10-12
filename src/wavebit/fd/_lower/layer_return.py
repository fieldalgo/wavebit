"""The exact return of one PML face of this kernel: its complex reflection, at plan time.

**What it is.** A plane wave ``exp(i (k_u u + k_v v) - i omega t)`` meeting one ``PML`` face on
axis ``P`` sees a 1-D problem along ``P``, and on this kernel that problem is small and
exact: the leapfrog along ``P`` with the transverse differences replaced by the multipliers
they are, the CPML recursion per cell with ``b``, ``a`` and ``1/kappa`` exactly as
:func:`wavebit.fd._lower.cpml.profiles` lowers them, the stabilizing loss's ``keep``/``lift``
on the normal ``E`` and ``H`` exactly as :func:`~wavebit.fd._lower.cpml.stabilizing_factors`
lowers them, and the PEC wall where ``src/wavebit/fd/_kernel/faces.py`` puts it. Nothing here is
a continuum or a textbook formula: each line below is one of the kernel's update equations,
read at ``exp(-i omega dt)`` per step. ``tests/test_fd_layer_return.py`` holds it to the
kernel's own one-step matrix, driven at that frequency, to 1e-9.

**The kernel's equations, time-harmonic** (``src/wavebit/fd/_kernel/step.py``, ``curl.py``,
``cpml.py``). With ``F^n ~ exp(-i omega n dt)``, ``p = exp(-i omega dt / 2)``, and every
field referred to its own Yee point (so a transverse forward and backward difference are both
``i K dx``: ``K = 2 sin(k dx/2)/dx`` on a Bloch or Periodic axis, ``K = k`` on an ``Axis1D``
one, whose "difference" is the product ``i k dx`` -- ``faces.py``):

* the CPML memory ``psi <- b psi + a d; d <- d/kappa + psi`` is a multiplier on the bare
  difference, ``S = 1/kappa + a / (1 - b exp(i omega dt))`` -- ``node`` profile on the
  backward difference of ``curl H``, ``half`` on the forward one of ``curl E``;
* an update ``F <- keep (q F + lift g c)`` (``q = ae`` or ``ah``, ``g = be`` or ``bh``, the
  loss's ``keep = lift = 1`` off the normal component and outside the slab) is
  ``(p - keep q / p) F = keep lift g c``, so a component contributes ``t = (p - keep q/p) /
  (keep lift g)`` where the continuum has ``-i omega eps dx``;
* eliminating the normal components (``E_P`` on the half points, ``H_P`` on the nodes)
  leaves, per polarisation, one tangential ``e`` on the nodes and one ``h`` on the halves::

      S_half[k] (e[k+1] - e[k]) = alpha[k] h[k]
      S_node[k] (h[k] - h[k-1]) = beta[k] e[k]

      TE (E transverse to k_par):  alpha = t_H,               beta = t_E - j^2 / t_HP
      TM (E in the plane of k_par): alpha = -t_H + j^2 / t_EP,  beta = -t_E

  with ``j = i K dx``; the loss sits in ``t_HP`` (TE) and ``t_EP`` (TM) only, which is why
  normal incidence (``j = 0``) never sees it.

The transverse frame is rotated so that ``k_par`` lies along one axis; the equations are
covariant under that rotation because both transverse components share one ``S`` and one
medium. So the medium is **isotropic** here (``eps``, ``mu`` scalars); a caller with a diagonal
medium passes the index it judges the order in (``grazing.py`` takes the lowest). The return of
an isotropic medium depends on ``n = sqrt(eps mu)`` alone -- the matched loss has one rate for
``E`` and ``H`` and the impedance scales out of ``h`` -- so ``grazing.py`` passes ``eps = n^2``
(``eps = mu = 1.5`` and ``eps = 2.25, mu = 1`` agree to 1e-12 relative).

**The wall** (``faces.py``). On a high face the wall is the zero pad ``E_t[N] = 0``, half a
cell past the last half point; on a low face it is the node ``x_0``, where tangential ``E`` is
masked to zero. Both are marched from the wall to the layer's inner face, the depth-0 node
(``N - cells`` on a high face, ``cells`` on a low one), and the field there is split into the
interior's two discrete modes ``lambda^(+-k)``, ``4 sin^2(kappa/2) = dx^2 (n^2 W^2 - K^2)``,
``W = 2 sin(omega dt/2)/dt``. The return is ``r`` = the outgoing mode's tangential ``E`` over the
incoming one's, **referred to that inner face**. A propagating order's ``|r|`` is then the
amplitude ratio of any component, which is what the doubling probe of
``tests/test_fd_pml.py`` reads.

**Only a propagating order.** An order the grid makes evanescent (``n W < K``) is refused, not
modelled: the grazing rule never asks for one. On an ``Axis1D`` axis (``K = k``) the grid cuts
an order off at ``cos^2 theta = 1 - (W / omega)^2``, which is 0.130 at the coarsest grid the
plan takes (8 cells per wavelength, ``courant`` just below 1) -- below the rule's floor (0.204
at the design wavelength, 0.1875 as ``g -> 1``), and the rule does not model a ``'floor'``
row. On a Bloch or Periodic axis ``K = 2 sin(k dx/2)/dx`` falls below ``k`` faster than ``W``
falls below ``omega`` (``dx > dt``), so an order propagating in the continuum propagates on the
grid. (The branch that modelled one, ``kappa = 2i asinh(.)``, agreed with the kernel where the
review drove it, but no rule reached it, so no gate held it.) Nor is an order past the grid's
Nyquist along the axis (``dx^2 (n^2 W^2 - K^2) > 4``, under two cells per wavelength along it):
it was a bare 'math domain error' on one order and a NaN on an array, and is refused by name.
The plan refuses every wavelength the rule reads -- a ``Diffraction`` monitor's, and the
sources' band, which is what the incidence row is judged at -- below 8 cells per wavelength in
any material first (``checks.check_wavelength_resolution``, one door for both since a stabilized
column with no ``Diffraction`` monitor at 3.0 cells per wavelength reached this refusal), where
that product is at most ``(2 pi / 8)^2 = 0.62``.

**Cost**, measured 2026-09-20 (Windows, NumPy 2.5.1): one call, both polarisations, is 0.073 /
0.102 ms for the pure stretch at 12 / 24 cells and 0.105 / 0.131 ms with the loss (mean of
2000) -- the profile builders of ``cpml.py`` are most of it. ``K`` may be an array: every
per-cell coefficient is shared by a face's orders at one wavelength, so they march together,
0.25 / 0.31 / 0.76 ms for 1 / 100 / 1000 orders (best of 5 x 20), against 0.09 ms for one
order as a number. The grazing rule makes two calls per face and wavelength on a stabilized
face (the face, and its pure stretch beside it: it bounds the one by the other) and none on a
pure-stretch face.
"""
import cmath
import math

import numpy as np

from .cpml import profiles, stabilizing_factors

__all__ = ['grid_k', 'layer_return', 'layer_return_magnitude']


def grid_k(k, dx, axis1d=False):
    """``K``: what a transverse wavenumber ``k`` is to the kernel's transverse difference.

    ``2 sin(k dx / 2) / dx`` on a Bloch or Periodic axis (a one-cell difference of
    ``exp(i k x)``, position-referred), ``k`` itself on an ``Axis1D`` axis, whose derivative is
    the exact ``i k`` (``src/wavebit/fd/_kernel/faces.py``).
    """
    return float(k) if axis1d else 2.0 * math.sin(0.5 * k * dx) / dx


def layer_return(cells, dx, dt, design_wavelength, eps_r, side, reflection, rho, wavelength,
                 K, eps=1.0, mu=1.0):
    """``(r_TE, r_TM)``: the complex return of one PML face of this kernel.

    ``cells``, ``design_wavelength`` (the face's CFS pole is set from it), ``eps_r`` (what the
    stretch is matched to, :func:`~wavebit.fd._lower.cpml.face_eps_r`), ``side`` (``'-'`` or
    ``'+'``), ``reflection`` and ``rho`` (the stabilizing loss; 0 is the pure stretch) are the
    face's, exactly as the plan lowers them; ``dx`` and ``dt`` the plan's. ``wavelength`` is
    the wave's vacuum wavelength (``omega = 2 pi / wavelength``, ``c = 1``), ``K`` the magnitude
    of its transverse wavenumber on the grid (:func:`grid_k` per axis, then the root sum of
    squares), ``eps`` and ``mu`` the isotropic medium filling the layer and the space in front
    of it. ``r`` is the outgoing tangential ``E`` over the incoming one at the layer's inner
    face (the module docstring).

    ``K`` may be a 1-D array: the orders of one face at one wavelength share every per-cell
    coefficient, so the march runs once over all of them and ``r_TE``, ``r_TM`` are arrays of
    ``K``'s length (a scalar ``K`` gives two complex numbers, as before). Only an order the
    grid propagates is modelled: an order it makes evanescent is refused (the module
    docstring says why the grazing rule never asks for one).
    """
    omega = 2.0 * math.pi / wavelength
    table = profiles(cells, dx, dt, design_wavelength, eps_r, side, reflection)
    delay = cmath.exp(1j * omega * dt)
    S = {}
    for name in ('node', 'half'):
        b, a, ik = table[name]
        S[name] = (ik + a / (1.0 - b * delay)).tolist()
    if rho > 0.0:
        factors = stabilizing_factors(cells, dx, dt, eps_r, side, reflection, rho)
        keep_n, lift_n = factors['node']          # H_P sits on the node
        keep_h, lift_h = factors['half']          # E_P sits on the half point
    else:
        keep_n = lift_n = keep_h = lift_h = np.ones(cells)
    keep_n, lift_n, keep_h, lift_h = (x.tolist() for x in (keep_n, lift_n, keep_h, lift_h))
    p = cmath.exp(-0.5j * omega * dt)
    ce, ch = dt / (dx * eps), dt / (dx * mu)       # be and bh of the medium
    t_E, t_H = (p - 1.0 / p) / ce, (p - 1.0 / p) / ch
    Ks = np.asarray(K, dtype=np.float64)
    scalar = Ks.ndim == 0
    if Ks.ndim > 1:
        raise ValueError(f'layer_return: K is a number or a 1-D array, not shape {Ks.shape}')
    # j^2 = (i K dx)^2. A scalar stays a Python float (the one-order call keeps its speed); an
    # array broadcasts through every per-cell expression below unchanged.
    j2 = -(float(Ks) * dx) ** 2 if scalar else -(Ks * dx) ** 2
    t_EP = [(p - k / p) / (k * l * ce) for k, l in zip(keep_h, lift_h)]
    t_HP = [(p - k / p) / (k * l * ch) for k, l in zip(keep_n, lift_n)]
    # The interior's modes: lambda = exp(i kappa), the incoming one travelling toward the face.
    # 4 sin^2(kappa/2) = -alpha beta = dx^2 (n^2 W^2 - K^2), the same for TE and TM, and real
    # for the real medium this takes (t_E t_H and j^2 are real), so kappa >= 0 is chosen in
    # real arithmetic -- never on |lam| (a propagating lam sits on the unit circle to
    # round-off, and a test on its modulus flipped the direction at normal incidence).
    s = -np.real(t_H * t_E - j2)
    if np.any(s < 0.0):
        raise ValueError(
            f"layer_return: K = {np.max(Ks):.6g} is past the grid's own cutoff n W = "
            f"{math.sqrt(eps * mu) * 2.0 * math.sin(0.5 * omega * dt) / dt:.6g}: the order is "
            f"evanescent on this grid, and only a propagating order is modelled (the grazing "
            f"rule never asks for one: layer_return.py's module docstring)")
    if np.any(s > 4.0):
        # 4 sin^2(kappa/2) > 4: the wave is shorter along the axis than two cells, past the
        # grid's Nyquist there, and the grid carries it as a decaying sign-alternating field
        # -- not modelled either (asin of a number above 1 was a bare 'math domain error' on
        # one order and a silent NaN on an array).
        raise ValueError(
            f"layer_return: dx^2 (n^2 W^2 - K^2) = {float(np.max(s)):.6g} > 4: wavelength "
            f"{wavelength:g} in index {math.sqrt(eps * mu):g} is past the grid's Nyquist "
            f"limit along the face's axis at dx = {dx:g} (fewer than two cells per wavelength "
            f"along it), and only an order the grid propagates is modelled (the plan refuses "
            f"a Diffraction wavelength below 8 cells per wavelength in any material first: "
            f"checks.check_wavelength_resolution)")
    if scalar:
        lam = cmath.exp(2j * math.asin(math.sqrt(float(s)) / 2.0))
    else:
        lam = np.exp(2j * np.arcsin(np.sqrt(s) / 2.0))
    out = []
    for pol in ('TE', 'TM'):
        alpha, beta, alpha0 = _polarisation(pol, cells, t_E, t_H, t_EP, t_HP, j2)
        out.append(_march(side, cells, alpha, beta, S['half'], S['node'], alpha0,
                          _interior_beta(pol, t_E, t_H, j2), lam))
    return tuple(complex(r) for r in out) if scalar else tuple(np.asarray(r) for r in out)


def _renorm(e, h):
    """``(e, h)`` divided by the larger modulus: the march's scale, per order on an array."""
    if isinstance(e, np.ndarray) or isinstance(h, np.ndarray):
        scale = np.maximum(np.abs(e), np.abs(h))
    else:
        scale = max(abs(e), abs(h))
    return e / scale, h / scale


def _polarisation(pol, cells, t_E, t_H, t_EP, t_HP, j2):
    """``(alpha per half point, beta per node, alpha of the interior)`` of one polarisation.

    The normal components are eliminated here, and they are the only place the stabilizing
    loss enters: ``t_HP`` (on the nodes) in TE, ``t_EP`` (on the half points) in TM.
    """
    if pol == 'TE':
        return [t_H] * cells, [t_E - j2 / t for t in t_HP], t_H
    return [-t_H + j2 / t for t in t_EP], [-t_E] * cells, -t_H + j2 / t_E


def _interior_beta(pol, t_E, t_H, j2):
    """``beta`` of the interior (no loss, no stretch): the node in front of a low face."""
    return t_E - j2 / t_H if pol == 'TE' else -t_E


def _march(side, cells, alpha, beta, S_half, S_node, alpha0, beta0, lam):
    """``r`` of one polarisation: march from the wall to the inner face, split into modes."""
    if side == '+':
        # Local index i is array index N - cells + i. The wall is the pad e[N] = 0; march down.
        e, h = 0.0, 1.0                             # e[cells], h[cells - 1]
        for i in range(cells - 1, -1, -1):
            e = e - alpha[i] * h / S_half[i]        # e[i]
            h = h - beta[i] * e / S_node[i]         # h[i - 1]
            e, h = _renorm(e, h)
        # e = e[0] at the inner face, h = h[-1] half a cell in front of it. With the incoming
        # mode A lam^k and the outgoing B lam^-k (k from the face): e = A + B and
        # h[-1] = ((1 - 1/lam) A + (1 - lam) B) / alpha0.
        A_e, A_h = 1.0, (1.0 - 1.0 / lam) / alpha0
        B_e, B_h = 1.0, (1.0 - lam) / alpha0
        det = A_e * B_h - B_e * A_h
        A = (e * B_h - B_e * h) / det
        B = (A_e * h - e * A_h) / det
        return B / A
    # Low face: local index i is array index i. The wall is the node x_0, e[0] = 0; march up.
    e, h = 0.0, 1.0                                 # e[0], h[0]
    for i in range(cells):
        e = e + alpha[i] * h / S_half[i]            # e[i + 1]
        if i + 1 < cells:
            h = h + beta[i + 1] * e / S_node[i + 1]  # h[i + 1]
        else:
            h = h + beta0 * e                       # h[cells]: node `cells` is depth 0
        e, h = _renorm(e, h)
    # e = e[cells] at the inner face, h = h[cells] half a cell in front of it. The incoming
    # mode travels toward -P: B lam^-k, the outgoing A lam^k, and
    # h[cells] = ((lam - 1) A + (1/lam - 1) B) / alpha0.
    A_e, A_h = 1.0, (lam - 1.0) / alpha0
    B_e, B_h = 1.0, (1.0 / lam - 1.0) / alpha0
    det = A_e * B_h - B_e * A_h
    A = (e * B_h - B_e * h) / det
    B = (A_e * h - e * A_h) / det
    return A / B


def layer_return_magnitude(*args, **kwargs):
    """``max(|r_TE|, |r_TM|)`` of :func:`layer_return`: the worse polarisation's amplitude."""
    return float(max(abs(r) for r in layer_return(*args, **kwargs)))
