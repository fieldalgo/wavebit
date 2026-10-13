"""Stage 3's post-processing: what the DFT accumulators and the source series mean.

Everything here runs **after** the scan, in the backend's array module, on the accumulators
the kernel's one frequency-domain observer filled (``_kernel/observe.py::DFTRegion``) and on
the reference series the sources recorded (stage 2). Nothing is stepped and nothing is closed
form: a flux, an order, a normalisation is arithmetic on phasors the run produced, so it is
differentiable on JAX and Torch like the run itself.

Conventions, repeated because every derivation below uses them: ``e^{-i w t}``; ``c = eps0 =
mu0 = 1``; H is stored as ``Z0 H``; E is sampled at ``(n + 1) dt`` and H at ``(n + 1/2) dt``;
every phasor is ``F^ = sum_n F(t_n) exp(i w t_n) dt`` at its own component's times. With
``Omega = 2 sin(w dt/2)/dt`` and ``K_a = 2 sin(k_a dx/2)/dx`` the discrete wavenumbers, a run
that starts and ends at rest satisfies, point for point,

    -i Omega mu H^ = -curl E^ - M^,        -i Omega eps E^ = curl H^ - J^          (*)

with the curls the grid's own differences. (Summing ``H(n+1/2) - H(n-1/2) = -dt (curl E(n) +
M(n))`` against ``exp(i w n dt)``: the two H sums are ``H^ exp(-+i w dt/2)``, their difference
``-2i sin(w dt/2) H^``.) A run stopped early adds the boundary terms of those sums -- the field
still in the region at the end -- which is what the decay diagnostic reports.

**Flux, and the 2-cell face.** On the node plane ``k`` across ``P`` the tangential E lives on
the plane and the tangential H on the half planes ``k -+ 1/2``. Summation by parts of (*) over
a set of cells gives the discrete Poynting theorem: for ``E^`` on the nodes ``k0..k1`` and
``H^`` on the half planes between them,

    sum_h H^* . curl E^ - sum_e E^ . curl H^*  =  [E^ x H^*(k1 - 1/2)] - [E^ x H^*(k0 + 1/2)]

(plus ``-E.J* - H*.M``), and with the half planes just outside, ``[E x H*(k1 + 1/2)] - [E x
H*(k0 - 1/2)]``. The left side is ``i Omega (sum mu|H|^2 - sum eps|E|^2)`` in a lossless source-
free region -- purely imaginary -- so the real part of each bracket form closes **exactly**.
The average of the two forms pairs the node-plane E with the mean of the two H planes
straddling it, and closes exactly too, being the mean of two exact identities; it is also the
one that is centred, so its error on a single plane is second order. That average is what
:func:`plane_flux` computes: ``0.5 Re sum (E_u conj(H_v) - E_v conj(H_u)) dA`` with ``H`` the
mean of the two half planes and ``dA`` the quadrature weight of each point
(``src/wavebit/fdtd/_prepare/monitors.py::plane_layout``). Transversally ``E_u`` and ``H_v`` share their Yee
position, and so do ``E_v`` and ``H_u``; nothing is interpolated across the plane.

**One plane wave on the Yee grid.** For ``exp(i (k_u u + k_v v + kappa P))`` with the stagger
removed (each component referred to its own position), (*) gives ``H = K x E / (Omega mu)``
and ``K . E = 0``, ``K = (K_u, K_v, K_P)``, ``K_P = 2 sin(kappa dx/2)/dx``, and the dispersion
``K_u^2 + K_v^2 + K_P^2 = Omega^2 eps mu``. Eliminating ``E_P``, the tangential H at the node
is ``M(kappa) E_t`` with

    M = [[-K_u K_v, -(K_v^2 + K_P^2)], [K_u^2 + K_P^2, K_u K_v]] / (K_P Omega mu),
    M^-1 = [[K_u K_v, K_v^2 + K_P^2], [-(K_u^2 + K_P^2), -K_u K_v]] / (K_P Omega eps),

odd in ``K_P``. The mean of the two half planes is ``cos(kappa dx/2)`` times the node value,
so on a plane holding a wave each way, ``E_t = E+ + E-`` and ``H_mean = cos(kappa dx/2) M (E+ -
E-)``: **the two directions separate exactly**, ``E+- = (E_t +- M^-1 H_mean / cos)/2``, for
every order at once. That split is what lets a Diffraction plane sit in the total-field
region, and what isolates a TFSF column's incident wave from its layers' reflection.
``kappa = (2/dx) asin(K_P dx/2)`` is the order's **Yee kz**; the branch is ``Re(K_P) > 0`` (or
``Im > 0`` for an evanescent order), so ``+`` is travelling or decaying toward ``+P``.

**s and p.** Each order is resolved in its own frame ``s_hat = P_hat x K_hat_par`` (the
discrete transverse wavevector's direction, which is the frame the grid's own plane waves are
pure s or p in; the continuum's as ``dx -> 0``) and ``p_hat = s_hat x K_hat``, the
``wavebit.observations.diffraction`` convention (``r_p = -r_s`` off a single interface at normal
incidence). ``p_hat``'s tangential part is ``(K_P / |K|) K_hat_par`` with ``K_P`` signed by
the direction, so ``E_p = (E_t . K_hat_par) |K| / K_P``. At ``K_par = 0`` the frame is the
PlaneWave's own, from ``phi``. The flux of an order splits as ``S = S_s + S_p`` exactly (the
cross terms are ``s x s`` and ``p x p``).

**The far field** (:func:`far_field`, Taflove & Hagness, *Computational Electrodynamics*, 3rd
ed., ch. 8; the signs re-derived here for ``e^{-i w t}``). By the equivalence principle the
field outside a closed surface ``S`` around every source is radiated by the surface currents
``J = n x H`` and ``M = -n x E`` (``n`` outward) into the uniform medium outside. With the
outgoing Green's function ``exp(i k R)/(4 pi R)`` and ``R ~ r - r_hat . r'`` far away, the
radiation vectors are ``N = int J exp(-i k r_hat . r') dS'`` and ``L = int M exp(-i k r_hat . r')
dS'``, and from ``E = i w A - (1/eps) curl F`` with ``A = mu N G``, ``F = eps L G``, ``G =
exp(i k r)/(4 pi r)`` (only the transverse parts survive, ``grad div`` cancelling ``A``'s radial
part):

    E_theta = (i k / 4 pi) (eta N_theta + L_phi) exp(i k r)/r,
    E_phi   = (i k / 4 pi) (eta N_phi - L_theta) exp(i k r)/r,

``eta = sqrt(mu/eps)``, and the radiant intensity ``U = r^2 |E|^2 / (2 eta)`` (a check: a point
current ``J = I l z_hat`` gives ``E_theta = -i k eta I l sin(theta) exp(i k r)/(4 pi r)``, the
textbook dipole; and a plane wave leaving through a face adds ``eta N_theta`` and ``L_phi`` in
the forward direction and cancels them backward, which is Love's field-free interior).

On the grid the currents are the same face samples a ``Flux`` box pairs, and in the same
collocation: on a face across ``P`` at node plane ``k`` with outward ``sigma P_hat``,
``J_u = -sigma H_v``, ``J_v = sigma H_u``, ``M_u = sigma E_v``, ``M_v = -sigma E_u``, with ``H``
the mean of the two half planes about the node plane -- ``(J_u, M_v)`` at the ``E_u`` points
(half along ``u``, on the ``v`` nodes) and ``(J_v, M_u)`` at the ``E_v`` points -- integrated with
the flux's quadrature weights (``src/wavebit/fdtd/_prepare/monitors.py::plane_layout``) and the phase at each
point's own position. ``k = 2 pi f sqrt(eps mu)`` is the continuum wavenumber of the medium
outside: the transform is the free-space one, as it is in the reference. The integral of ``U``
over the sphere is the net flux of the same box to the discretisation of the transform
(``src/wavebit/fdtd/_prepare/monitors.py``); the flux itself is exact.
"""
import math

import numpy as np

__all__ = ['dft', 'plane_flux', 'box_flux', 'plane_wave_matrices', 'split_directions',
           'medium_at', 'order_kz', 'DIFFRACTION_KZ', 'decay_ratio', 'far_field',
           'FARFIELD_M_SIGN']

#: Position of axis ``a`` (0 = x, 1 = y, 2 = z) in a single-component ``(nz, ny, nx)`` array.
AX = {0: 2, 1: 1, 2: 0}

#: Which kz the de-embedding uses: ``'yee'`` (the discrete dispersion, the only right one) or
#: ``'continuum'`` (``sqrt(n^2 w^2 - k_par^2)``). A module switch and not a keyword, because
#: the continuum is not an option a user should have: it exists for the mutant that proves the
#: de-embedding gate can tell the two apart (a phase error growing with the order's angle).
DIFFRACTION_KZ = 'yee'

#: The sign the far-field transform gives the equivalent magnetic current, ``M = sign * (-n x
#: E)``: ``+1``, the only right one. A module switch for the same reason as
#: :data:`DIFFRACTION_KZ` -- it exists for the far-field gate's mutant (flipped, the forward
#: and backward lobes of a surface's field trade places and the pattern of a dipole is no
#: longer ``sin^2 theta``).
FARFIELD_M_SIGN = 1.0


def _cplx(rt, x):
    return x * rt.array(1.0 + 0.0j, 'complex128')


def dft(rt, values, t, dt, freqs, leapfrog=False):
    """``sum_n values[n] exp(i 2 pi f t_n) dt`` for each ``f``: ``(steps, ...) -> (nf, ...)``.

    ``leapfrog=True`` for a field series (not a drive): the end correction of
    :func:`end_correction` is added with the series' last value, and a first value (one step
    before ``t[0]``) of zero -- every source run starts at rest.
    """
    t = np.asarray(t, dtype=np.float64)
    W = np.exp(1j * 2.0 * math.pi * np.asarray(freqs, dtype=np.float64)[:, None] * t[None, :]) * dt
    shape = tuple(values.shape)
    flat = rt.xp.reshape(_cplx(rt, values), (shape[0], -1))
    out = rt.xp.reshape(rt.array(W, 'complex128') @ flat, (len(freqs),) + shape[1:])
    if leapfrog:
        out = out + end_correction(rt, values[-1], float(t[-1]), dt, freqs)
    return out


def end_correction(rt, last, t_last, dt, freqs, first=None, t_first=None):
    """What makes a truncated leapfrog DFT satisfy the discrete Maxwell equations exactly.

    For samples ``a_m`` at ``t_m = t_1 + (m - 1) dt``, ``m = 0 .. N``, and the accumulated
    ``A = sum_{m=1}^{N} a_m exp(i w t_m) dt``, summing the update ``a_{m+1} - a_m = dt (...)``
    against ``exp(i w (t_m + dt/2))`` gives, with nothing dropped,

        sum_m (a_{m+1} - a_m) exp(i w (t_m + dt/2)) = -i Omega A + exp(i w dt/2) (a_N exp(i w t_N)
                                                        - a_0 exp(i w t_0)),

    so the phasor for which ``-i Omega F^ = (the right-hand side)^`` holds exactly -- the one
    every identity of this module rests on -- is ``A + (i/Omega) exp(i w dt/2) (a_N e^{i w
    t_N} - a_0 e^{i w t_0})``. This returns that correction; ``last`` is ``a_N`` (the final
    field in the region) at ``t_last``, ``first`` is ``a_0`` at ``t_first`` (``None`` for
    zero -- a run with an ``InitialField`` has one). For a field that has
    decayed it is zero. **What it removes is the truncation of a static residue**: a dipole
    whose drive did not integrate to exactly zero (the band starts at its cutoff, 1e-3 of the
    peak) leaves a charge behind, whose near field is constant to the end of the run; the
    plain truncated DFT then carries ``c exp(i w T)/(i w)``, which moved a dipole's power by
    1e-3 as the run length changed (14 against 24 periods) and flux against power by the same.
    With the correction the power agrees with the flux of a box around it to 1e-8 at both
    lengths (an empty box's net flux: 5e-10 of the power), measured 2026-09-19. An
    *oscillating* residue is not removed -- that is the decay diagnostic's job -- and neither is
    a slowly *relaxing* one: every run the gates measured still holds a final ``|H|`` of 1e-6 to
    1e-5 of its peak after 10 to 45 periods (the CPML's shifted-pole memory or the dipole's
    leftover charge settling; not isolated), which is neither static nor curl-free, so flux
    closure stops near 1e-7 of a face's own flux (5e-10 of the power above is the same floor
    counted against the larger number) rather than at round-off. The gate tolerances sit a
    decade above it (``tests/fdtd/test_fd_flux.py``).
    """
    omega = 2.0 * math.pi * np.asarray(freqs, dtype=np.float64)
    Omega = 2.0 * np.sin(omega * dt / 2.0) / dt
    shape = (len(freqs),) + (1,) * len(tuple(getattr(last, 'shape', ())))
    gain = (1j / Omega) * np.exp(1j * omega * dt / 2.0)
    out = (rt.array((gain * np.exp(1j * omega * t_last)).reshape(shape), 'complex128')
           * _cplx(rt, last)[None])
    if first is not None:
        out = out - rt.array((gain * np.exp(1j * omega * t_first)).reshape(shape),
                             'complex128') * _cplx(rt, first)[None]
    return out


def _index(ndim, axis, sl):
    idx = [slice(None)] * ndim
    idx[axis] = sl
    return tuple(idx)


def _mean_planes(H, P):
    """The mean of the two half planes along ``P`` of an ``(nf, nz, ny, nx)`` accumulator."""
    ax = AX[P] + 1
    return 0.5 * (H[_index(4, ax, slice(0, 1))] + H[_index(4, ax, slice(1, 2))])


def _weights(rt, layout, a):
    """The node weights along transverse axis ``a``, shaped to broadcast, or ``None``."""
    w = layout.weights[a]
    if w is None:
        return None
    shape = [1, 1, 1, 1]
    shape[AX[a] + 1] = len(w)
    return rt.array(np.asarray(w).reshape(shape), 'float64')


def plane_flux(rt, acc, layout, dx):
    """``(nf,)``: ``0.5 Re sum (E_u conj(H_v) - E_v conj(H_u)) dA`` along ``+P`` (module doc)."""
    xp = rt.xp
    Hu, Hv = _mean_planes(acc['Hu'], layout.P), _mean_planes(acc['Hv'], layout.P)
    a = acc['Eu'] * xp.conj(Hv)
    b = acc['Ev'] * xp.conj(Hu)
    wv, wu = _weights(rt, layout, layout.v), _weights(rt, layout, layout.u)
    if wv is not None:
        a = a * wv                       # E_u, H_v sit on the v nodes
    if wu is not None:
        b = b * wu                       # E_v, H_u sit on the u nodes
    # Summed apart: on a finite face the two products live on different point sets (one more
    # node than half cell along each axis), so their arrays are not the same shape.
    total = (xp.sum(xp.reshape(a, (a.shape[0], -1)), axis=1)
             - xp.sum(xp.reshape(b, (b.shape[0], -1)), axis=1))
    return 0.5 * xp.real(total) * (dx * dx)


def box_flux(rt, accs, faces, dx):
    """``(net outward (nf,), {face: outward (nf,)})`` of a closed box's six faces."""
    parts, net = {}, None
    for face, sign, layout in faces:
        f = sign * plane_flux(rt, accs[face], layout, dx)
        parts[face] = f
        net = f if net is None else net + f
    return net, parts


# ---------------------------------------------------------------------------------------------
# The far field
# ---------------------------------------------------------------------------------------------

def _angles(rt, theta, phi):
    """``(theta, phi)`` in radians as flat backend arrays, and the broadcast shape."""
    xp = rt.xp
    th = rt.array(theta, 'float64') * (math.pi / 180.0)
    ph = rt.array(phi, 'float64') * (math.pi / 180.0)
    th, ph = th + 0.0 * ph, ph + 0.0 * th
    shape = tuple(th.shape)
    return xp.reshape(th, (-1,)), xp.reshape(ph, (-1,)), shape


def _surface_transform(rt, F, us, vs, wu, wv, k, ru, rv, rp, p, dx):
    """``sum_ij F[f, j, i] wu_i wv_j exp(-i k (ru u_i + rv v_j + rp p)) dx^2`` -> ``(nf, nd)``."""
    xp = rt.xp
    kk = k[:, None, None]
    a = rt.array(wu, 'float64')[None, None, :] * xp.exp(
        -1j * (kk * (ru[None, :, None] * rt.array(us, 'float64')[None, None, :])))
    b = rt.array(wv, 'float64')[None, None, :] * xp.exp(
        -1j * (kk * (rv[None, :, None] * rt.array(vs, 'float64')[None, None, :])))
    T = xp.matmul(a, xp.permute_dims(F, (0, 2, 1)))          # (nf, nd, Nv)
    S = xp.sum(T * b, axis=-1)
    return S * xp.exp(-1j * (k[:, None] * (rp[None, :] * p))) * (dx * dx)


def far_field(rt, accs, faces, grid, freqs, eps, mu, theta, phi):
    """``(E_theta, E_phi, U)`` at the directions ``(theta, phi)`` (degrees), each ``(nf, ...)``.

    ``E_theta, E_phi`` are ``r exp(-i k r) E`` and ``U`` the radiant intensity per unit solid
    angle; ``accs`` are a closed box's six face accumulators (end-corrected), ``faces`` its
    ``box_faces`` layouts. The module docstring derives every factor and sign.
    """
    xp = rt.xp
    th, ph, shape = _angles(rt, theta, phi)
    st, ct, sp, cp = xp.sin(th), xp.cos(th), xp.sin(ph), xp.cos(ph)
    rhat = (st * cp, st * sp, ct)
    that = (ct * cp, ct * sp, -st)
    phat = (-sp, cp, 0.0 * st)
    eps_a, mu_a = rt.array(eps, 'float64'), rt.array(mu, 'float64')
    n = xp.sqrt(eps_a * mu_a)
    eta = xp.sqrt(mu_a / eps_a)
    k = rt.array(2.0 * math.pi * np.asarray(freqs, dtype=np.float64), 'float64') * n
    dx = grid.dx
    N, L = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    m_sign = FARFIELD_M_SIGN
    for face, sign, layout in faces:
        acc = accs[face]
        P, u, v = layout.P, layout.u, layout.v
        p = grid.low[P] + layout.k * dx
        Eu = _vu(rt, acc['Eu'], P, u, v)
        Ev = _vu(rt, acc['Ev'], P, u, v)
        Hu = _vu(rt, _mean_planes(acc['Hu'], P), P, u, v)
        Hv = _vu(rt, _mean_planes(acc['Hv'], P), P, u, v)
        # The E_u points: half along u, on the v nodes; the E_v points the other way round.
        hu0, hu1 = layout.half[u]
        nu0, nu1 = layout.nodes[u]
        hv0, hv1 = layout.half[v]
        nv0, nv1 = layout.nodes[v]
        u_half = grid.low[u] + (np.arange(hu0, hu1) + 0.5) * dx
        u_node = grid.low[u] + np.arange(nu0, nu1) * dx
        v_half = grid.low[v] + (np.arange(hv0, hv1) + 0.5) * dx
        v_node = grid.low[v] + np.arange(nv0, nv1) * dx
        w_un = np.ones(nu1 - nu0) if layout.weights[u] is None else np.asarray(layout.weights[u])
        w_vn = np.ones(nv1 - nv0) if layout.weights[v] is None else np.asarray(layout.weights[v])
        w_uh, w_vh = np.ones(hu1 - hu0), np.ones(hv1 - hv0)
        ru, rv, rp = rhat[u], rhat[v], rhat[P]
        # (J_u, M_v) on the E_u points, (J_v, M_u) on the E_v points (module docstring).
        N[u] = N[u] + _surface_transform(rt, -sign * Hv, u_half, v_node, w_uh, w_vn, k, ru, rv,
                                         rp, p, dx)
        L[v] = L[v] + _surface_transform(rt, (-sign * m_sign) * Eu, u_half, v_node, w_uh, w_vn,
                                         k, ru, rv, rp, p, dx)
        N[v] = N[v] + _surface_transform(rt, sign * Hu, u_node, v_half, w_un, w_vh, k, ru, rv,
                                         rp, p, dx)
        L[u] = L[u] + _surface_transform(rt, (sign * m_sign) * Ev, u_node, v_half, w_un, w_vh,
                                         k, ru, rv, rp, p, dx)

    def dot(V, e):
        return V[0] * e[0][None, :] + V[1] * e[1][None, :] + V[2] * e[2][None, :]
    N_t, N_p, L_t, L_p = dot(N, that), dot(N, phat), dot(L, that), dot(L, phat)
    gain = (1j * k / (4.0 * math.pi))[:, None]
    E_t = gain * (eta * N_t + L_p)
    E_p = gain * (eta * N_p - L_t)
    U = (xp.abs(E_t) ** 2 + xp.abs(E_p) ** 2) / (2.0 * eta)
    out = (len(freqs),) + shape
    return xp.reshape(E_t, out), xp.reshape(E_p, out), xp.reshape(U, out)


# ---------------------------------------------------------------------------------------------
# Plane waves on the grid
# ---------------------------------------------------------------------------------------------

def _sqrt_branch(xp, z):
    """``sqrt(z)`` with ``Re > 0``, or ``Im > 0`` where the real part vanishes (evanescent)."""
    r = xp.sqrt(z)
    flip = (xp.real(r) < 0) | ((xp.abs(xp.real(r)) <= 1e-14 * xp.abs(r)) & (xp.imag(r) < 0))
    return xp.where(flip, -r, r)


def plane_wave_matrices(rt, Ku, Kv, Omega, eps, mu, dx):
    """``(K_P, kappa, cos(kappa dx/2), Minv)`` of the plane waves with transverse ``K_u, K_v``.

    ``K_P`` is the ``+P`` branch; ``Minv`` is ``M^-1`` of the module docstring as a dict of its
    four entries (each broadcast like the inputs), for the ``+P`` wave.
    """
    xp = rt.xp
    Ku, Kv = _cplx(rt, Ku), _cplx(rt, Kv)
    KP2 = Omega ** 2 * eps * mu - Ku ** 2 - Kv ** 2
    KP = _sqrt_branch(xp, _cplx(rt, KP2))
    half = KP * (dx / 2.0)
    kappa = (2.0 / dx) * xp.asin(half)
    cos = _sqrt_branch(xp, 1.0 - half ** 2)
    scale = 1.0 / (KP * Omega * eps)
    Minv = {'uu': Ku * Kv * scale, 'uv': (Kv ** 2 + KP2) * scale,
            'vu': -(Ku ** 2 + KP2) * scale, 'vv': -Ku * Kv * scale}
    return KP, kappa, cos, Minv


def split_directions(Eu, Ev, Hu, Hv, Minv, cos):
    """``((Eu+, Ev+), (Eu-, Ev-))`` from the node-plane E and the mean-plane H (module doc)."""
    du = (Minv['uu'] * Hu + Minv['uv'] * Hv) / cos
    dv = (Minv['vu'] * Hu + Minv['vv'] * Hv) / cos
    return ((0.5 * (Eu + du), 0.5 * (Ev + dv)), (0.5 * (Eu - du), 0.5 * (Ev - dv)))


def wave_flux(rt, Eu, Ev, Ku, Kv, KP, Omega, mu, cos, direction):
    """``0.5 Re (E_t x conj(H_mean)) . P_hat * direction`` of one wave travelling ``direction``."""
    xp = rt.xp
    KPs = KP * direction
    scale = 1.0 / (KPs * Omega * mu)
    # M E_t: H_u = -(K_u K_v E_u + (K_v^2 + K_P^2) E_v) / ..., H_v = ((K_u^2 + K_P^2) E_u + K_u K_v E_v) / ...
    Hu = -(Ku * Kv * Eu + (Kv ** 2 + KP ** 2) * Ev) * scale * cos
    Hv = ((Ku ** 2 + KP ** 2) * Eu + Ku * Kv * Ev) * scale * cos
    return 0.5 * direction * xp.real(Eu * xp.conj(Hv) - Ev * xp.conj(Hu))


def frame(rt, Ku, Kv, phi):
    """``(s_u, s_v, q_u, q_v)``: ``s_hat`` and ``K_hat_par`` per order; ``phi`` at ``K_par = 0``."""
    xp = rt.xp
    Ku, Kv = xp.real(_cplx(rt, Ku)), xp.real(_cplx(rt, Kv))
    sq = Ku ** 2 + Kv ** 2
    zero = sq <= 1e-24 * (1.0 + sq)
    safe = xp.sqrt(xp.where(zero, xp.ones_like(sq), sq))
    qu = xp.where(zero, math.cos(phi) + 0.0 * Ku, Ku / safe)
    qv = xp.where(zero, math.sin(phi) + 0.0 * Kv, Kv / safe)
    return -qv, qu, qu, qv


def sp_amplitudes(Eu, Ev, s_u, s_v, q_u, q_v, KP, Knorm, direction):
    """``(E_s, E_p)`` of a wave travelling ``direction`` along ``P`` (module doc)."""
    Es = Eu * s_u + Ev * s_v
    Ep = (Eu * q_u + Ev * q_v) * Knorm / (KP * direction)
    return Es, Ep


def order_kz(rt, kappa, Ku_c, Kv_c, n_omega, convention=None):
    """The kz the de-embedding uses: the Yee ``kappa``, or (mutant) the continuum one."""
    convention = DIFFRACTION_KZ if convention is None else convention
    if convention == 'yee':
        return kappa
    xp = rt.xp
    return _sqrt_branch(xp, _cplx(rt, n_omega ** 2 - Ku_c ** 2 - Kv_c ** 2))


# ---------------------------------------------------------------------------------------------
# The medium at a plane
# ---------------------------------------------------------------------------------------------

def layer_share(shape, grid, P, k):
    """``'full'``, ``'partial'`` or ``'absent'``: what ``shape`` is in cell layer ``k`` across ``P``.

    Asked only of a shape whose support reaches the layer. A cell-snapped shape's support *is*
    its filled cells, so it fills the layer when it is uniform across both transverse axes and
    cuts it otherwise -- the answer this module always gave. A smoothed shape's support carries
    a cell of margin and its fractional face cell, so it is asked by its own exact fill at all
    six components' Yee points in the layer: none anywhere is ``'absent'``, all of it across the
    whole layer is ``'full'``, anything else two media. Both ends are read against
    ``checks._CLIP_ROUNDOFF``, the exact clip's own round-off.

    The fill is read on the host, so a shape whose geometry carries a derivative is refused
    rather than read through ``numpy``, which would drop a Torch forward tangent silently.
    ``shape_ranges.plane_media`` keeps every layer a traced run reads outside such a shape's
    support, so the public paths never arrive here with one.
    """
    from wavebit.fdtd._prepare.shape_ranges import leaves
    from wavebit.fdtd._legacy_geometry import _shape_traced
    if any(_shape_traced(v) for v in leaves(shape)):
        raise ValueError(
            f'The medium of cell layer {k} across {"xyz"[P]} depends on a traced '
            f'{type(shape).__name__}; a plane-wave medium is read from concrete geometry '
            'only. Keep the plane outside the body\'s range and its one-cell margin.')
    body = shape.body()
    if body is None:
        full = all(shape.uniform_along(a, (0, grid.n[a]), grid) for a in range(3) if a != P)
        return 'full' if full else 'partial'
    import numpy as np
    from wavebit.fdtd._prepare.checks import _CLIP_ROUNDOFF
    from wavebit.fdtd._prepare.occupancy import yee_points
    from wavebit.fdtd._shapes import occupancy as body_occupancy
    box = list(shape.support(grid))
    box[P] = (k, k + 1)
    fills = [np.asarray(body_occupancy(body, np.asarray(yee_points(grid, tuple(box), field, c))
                                       .reshape(-1, 3), grid.dx)[0])
             for field in ('E', 'H') for c in range(3)]
    fill = np.concatenate(fills)
    if np.all(fill <= _CLIP_ROUNDOFF):
        return 'absent'
    whole = all(box[a] == (0, grid.n[a]) for a in range(3) if a != P)
    return 'full' if whole and np.all(fill >= 1.0 - _CLIP_ROUNDOFF) else 'partial'


def medium_at(scene, grid, P, cells, where):
    """The one plain material filling the cell layers ``cells`` across ``P``, or a refusal.

    A diffraction plane or a source normalisation reads a plane wave in a uniform medium, so
    every cell of each layer must belong to one material -- the background or a structure
    that covers the whole layer -- and the layers must agree. Structures are applied in order,
    so a shape that cuts the layer and is then overwritten by a later full layer is fine.

    The refusal names the structure that cut the layer, in that shape's own words
    (``Shape.describe``): it used to say "a Box structure cuts it" whatever the shape was,
    which is a sentence about a box in a message about a ball.

    A **smoothed** shape is asked what it is in the layer by its own fill
    (:func:`layer_share`), not by its support: the support carries a cell of margin and the
    fractional face cell, and reading it as "fills the layer" put a plane in the air under a
    smoothed ``fd.Box`` film into the film (``eps`` 4 where the voxels hold 1, found in the
    review of C4) and refused a plane a cell and a half clear of it.
    """
    from wavebit.materials.time_domain import Conductor
    from wavebit.materials.time_domain import Dispersive
    from wavebit.materials.time_domain import as_material
    from wavebit.materials.time_domain import concrete
    background = as_material(scene.background, 'Scene(background=)')
    owners = set()
    for k in cells:
        owner, partial = None, None
        for index, s in enumerate(scene.structures):
            box = s.shape.support(grid)        # anything that may reach this cell layer
            if not box[P][0] <= k < box[P][1]:
                continue
            share = layer_share(s.shape, grid, P, k)
            if share == 'absent':
                continue
            if share == 'full':
                owner, partial = index, None
            else:
                partial = index
        if partial is not None:
            cutter = scene.structures[partial].shape
            raise ValueError(
                f'{where}: the cell layer {k} across {"xyz"[P]} is not one medium '
                f'(structures[{partial}], {cutter.describe()}, cuts it). A plane wave -- an '
                f'order, an incident wave -- is defined in a uniform medium; move the plane '
                f'into one.')
        owners.add(owner)
    if len(owners) != 1:
        raise ValueError(
            f'{where}: the cells {tuple(cells)} either side of the plane are different media '
            f'(an interface on the plane). Move the plane at least a cell off it.')
    owner = owners.pop()
    material = background if owner is None else scene.structures[owner].material
    if isinstance(material, (Conductor, Dispersive)):
        raise ValueError(
            f'{where}: the medium at the plane is an fd.{type(material).__name__}; the orders '
            f'and the normalisations are plane waves of a real (eps, mu), and a lossy or '
            f'dispersive one is not in this build. Put the plane in a plain fd.Material.')
    eps, mu = material.eps, material.mu_value

    def iso(values, name):
        first = values[0]
        c = [concrete(v) for v in values]
        if all(v is first for v in values) or (None not in c and len(set(c)) == 1):
            return first
        raise ValueError(f'{where}: the medium at the plane is anisotropic in {name} '
                         f'{tuple(values)}; the plane-wave algebra here is isotropic.')
    return iso(eps, 'eps'), iso(mu, 'mu')


# ---------------------------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------------------------

def decay_ratio(energy, window):
    """How much the region energy still **changes** in the last window, over its peak.

    ``(max - min)(energy over the last window) / max(energy)``: 0 for a settled region, about
    1 for one still at its peak. The change and not the level, because a static residue -- the
    near field of the charge a dipole leaves when its drive does not integrate to exactly zero
    -- holds a constant energy to the end of every run (measured: 1.1e-8 to 1.4e-8 of the peak
    in a flux box around a dipole, the same at 10 and at 120 periods), and
    :func:`end_correction` makes the phasors exact for it; what it cannot mend is a field
    still oscillating, and that is what moves the energy within a period.
    """
    e = np.asarray(energy, dtype=np.float64)
    peak = float(np.max(e)) if e.size else 0.0
    if not peak > 0.0:
        return 0.0
    window = max(1, min(int(window), e.size))
    tail = e[-window:]
    return float((np.max(tail) - np.min(tail)) / peak)


def growth_ratio(t, energy, start, window, min_windows, floor):
    """Whether the interior's energy still **rises** once nothing drives it, and how steadily.

    ``(ratio, lo, mid, hi, spread)`` or ``None``. ``t`` and ``energy`` are the interior-energy
    series (``src/wavebit/fdtd/_kernel/observe.py::InteriorEnergy``), ``start`` the time every source
    has ended by (``_assemble.sources_end``). The judged span is ``[start, hi]``, ``hi`` the last
    sample; the ratio is the mean energy over its last quarter ``[mid, hi]`` over the mean over
    the quarter before ``[lo, mid)``. ``spread`` says whether that rise is an exponential's:
    the last half ``[lo, hi]`` cut into four eighths, it is the largest of the three
    eighth-to-eighth rises of the log of their means over the smallest -- ``inf`` unless all
    three rise. ``_assemble.decay_report`` calls a scene growing when the ratio is over
    ``GROWTH_THRESHOLD`` **and** the spread is under ``GROWTH_STEADY``.

    **Why the whole interior.** After the sources end, the energy of a passive scene's
    interior can only hold (closed and lossless) or fall (a loss, or an open face it leaves
    through): that is Poynting's theorem, and on the grid the leapfrog's own energy identity
    (``InteriorEnergy``). So for a passive scene the ratio is at most 1 up to what the sampled
    energy is not -- for any structure, whatever its modes do -- and up to what the PML hands
    back, which is not in the interior's account. A monitor's own series has no such bound:
    energy moves through its region, so a later echo of the same pulse (a slab's second
    reflection, a PEC-backed layer's return) or two modes beating slower than a quarter read as
    "growth" by any factor; a first version of this diagnostic judged each monitor's series
    and gave a passive glass slab 1.2e8 and a lossless PEC box with two near-degenerate modes
    2.8 (2026-09-20).

    **Why the means of two quarters.** Means rather than peaks: what a monitor's or a
    structure's field does inside a quarter is not the interior's energy, which the observer
    reads exactly (``InteriorEnergy``), but a mean is what a sampled series of anything else
    -- a first version's reading, a region's energy -- is read by, and it costs nothing here.
    Measured over the suite, the passive runs' largest ratio is 1.0006 by means (1.043 by
    peaks, measured on the first version's reading: a 4^3 box of random grid-scale field, whose
    sampled energy swung by 12-22% over a run). A growth is the same factor either way,
    ``exp(g T/4)`` for
    ``g`` e-folds per unit time over a judged span ``T``. Quarters rather than the span's early
    energy, because growth seeded under a ring-down starts far below the ring-down's level:
    the lamellar grating under the default layer grows 2.6x per quarter at 384 periods while
    its interior is at 1e-5 of the level it had when the source ended.

    **Why the spread.** Energy the PML took and hands back is a rise the interior's account
    cannot tell from growth by its size: a pulse kicked into a vacuum column under a deep weak
    layer (``PML(64, 1e-3)``) leaves, and the wall's echo comes back after the layer's round
    trip into an interior at 1e-8 of its peak -- a last quarter 1.8e4 times the one before. It
    can by its shape. A linear instability's energy is a sum of exponentials that the fastest
    comes to dominate, so over the last half of the span its log rises in equal steps: the
    spread of the growing runs measured is 1.06 .. 1.64 (the slab column at every length from
    50 to 200, the grating at 384), falling toward 1 as the fastest mode takes over. An echo
    arrives with its pulse's shape -- a quiet floor, then a leading edge whose log steepens --
    and its spread is 4.2 .. 18 (the deep-layer echoes). The table is at
    ``_assemble.GROWTH_STEADY``.

    ``None`` -- nothing judged -- when the span is shorter than ``min_windows`` windows of
    ``window`` (time units), when a quarter or an eighth holds no sample, when a sample is not
    finite, and when the last quarter's peak is not above ``floor`` of the run's (a field
    decayed to round-off, whose last digits wander).
    """
    t = np.asarray(t, dtype=np.float64)
    e = np.asarray(energy, dtype=np.float64)
    if t.size == 0 or not np.all(np.isfinite(e)):
        return None
    hi = float(t[-1])
    if not hi - start >= min_windows * window:
        return None
    q = (hi - start) / 4.0
    lo, mid = hi - 2.0 * q, hi - q
    last, before = e[t >= mid], e[(t >= lo) & (t < mid)]
    if last.size == 0 or before.size == 0:
        return None
    top = float(np.max(e))
    if not (top > 0.0 and float(np.max(last)) > floor * top):
        return None
    edges = [lo + i * q / 2.0 for i in range(4)] + [math.inf]
    eighths = [e[(t >= a) & (t < b)] for a, b in zip(edges[:-1], edges[1:])]
    if any(x.size == 0 for x in eighths):
        return None
    means = [float(np.mean(x)) for x in eighths]
    spread = math.inf
    if means[0] > 0.0 and all(b > a for a, b in zip(means[:-1], means[1:])):
        rises = [math.log(b / a) for a, b in zip(means[:-1], means[1:])]
        spread = max(rises) / min(rises)
    last, before = float(np.mean(last)), float(np.mean(before))
    return (last / before if before > 0.0 else math.inf), lo, mid, hi, spread


def quarter_ratio(t, energy, lo, mid, hi):
    """A monitor's own ``max over [mid, hi] / max over [lo, mid)``: whether it sees a growth.

    Read only once :func:`growth_ratio` has judged the interior growing, over the same two
    quarters, to name the monitors whose own field rises with it. ``None`` when a quarter holds
    no sample or both are zero.
    """
    t = np.asarray(t, dtype=np.float64)
    e = np.asarray(energy, dtype=np.float64)
    last, before = e[(t >= mid) & (t <= hi)], e[(t >= lo) & (t < mid)]
    if last.size == 0 or before.size == 0:
        return None
    last, before = float(np.max(last)), float(np.max(before))
    if before > 0.0:
        return last / before
    return math.inf if last > 0.0 else None


# ---------------------------------------------------------------------------------------------
# Transverse structure of a periodic plane
# ---------------------------------------------------------------------------------------------

def is_axis1d(scene, a):
    """Whether transverse axis ``a`` is an ``fd.Axis1D`` axis (one cell, derivative ``i k``)."""
    from wavebit.fdtd._legacy_boundaries import Axis1D
    return isinstance(scene.boundaries.face('xyz'[a] + '-'), Axis1D)


def discrete_k(rt, scene, a, k, dx):
    """The grid's wavenumber of ``exp(i k a)`` along ``a``: ``2 sin(k dx/2)/dx``, or ``k`` itself
    on an ``Axis1D`` axis, whose derivative is the exact ``i k`` (``_kernel/faces.py``)."""
    if is_axis1d(scene, a):
        return k
    return 2.0 * rt.xp.sin(k * dx / 2.0) / dx


def bloch_k(rt, scene, grid, a):
    """``k_par`` along axis ``a``: the Bloch face's ``k`` (or ``phase / L``), an Axis1D face's
    ``k``, 0 if Periodic."""
    from wavebit.fdtd._legacy_boundaries import Axis1D
    from wavebit.fdtd._legacy_boundaries import Bloch
    face = scene.boundaries.face('xyz'[a] + '-')
    if isinstance(face, Axis1D):
        return rt.array(face.k, 'float64')
    if not isinstance(face, Bloch):
        return rt.array(0.0, 'float64')
    if face.k is not None:
        return rt.array(face.k, 'float64')
    return rt.array(face.phase, 'float64') / grid.size[a]


def _vu(rt, F, P, u, v):
    """An ``(nf, nz, ny, nx)`` accumulator with one layer across ``P`` as ``(nf, n_v, n_u)``."""
    shape = tuple(F.shape)
    keep = [0, AX[v] + 1, AX[u] + 1, AX[P] + 1]
    G = rt.xp.permute_dims(F, tuple(keep))
    return rt.xp.reshape(G, (shape[0], shape[AX[v] + 1], shape[AX[u] + 1]))


def order_coefficients(rt, F, P, u, v, ku, kv, grid, offsets):
    """``(nf, n_v, n_u)``: the plane-wave amplitude of each order, at the cell's low corner.

    ``offsets`` are the component's half-cell offsets along ``u`` and ``v``, ``0`` along an
    ``Axis1D`` axis (whose field is ``exp(i k v)`` with no stagger: every component shares the
    one phase reference, the caller zeroes it there).
    ``F(i, j) = sum a_mn exp(i (k_u,m u_i + k_v,n v_j))`` with ``u_i = (i + o_u/2) dx`` the
    component's own positions **measured from the cell's low corner** -- the half-cell stagger is
    removed here, and ``a`` is the amplitude of ``exp(i k . (r - r_corner))``. This internal
    extraction convention matches the source-column reference. The public diffraction result
    divides by that incident amplitude and shifts to the cell centre in ``_assemble`` using
    ``exp(i (k_m - k_0) . P/2) = (-1)^(m + n)``, matching the current RCWA result contract.
    Keeping extraction in its original frame preserves raw field and accumulator semantics.
    """
    xp = rt.xp
    G = _vu(rt, F, P, u, v)
    Nu, Nv = grid.n[u], grid.n[v]
    pos_u = rt.array((np.arange(Nu) + 0.5 * offsets[0]) * grid.dx, 'float64')
    pos_v = rt.array((np.arange(Nv) + 0.5 * offsets[1]) * grid.dx, 'float64')
    Wu = xp.exp(-1j * (pos_u[:, None] * ku[None, :])) / Nu
    Wv = xp.exp(-1j * (kv[:, None] * pos_v[None, :])) / Nv
    return Wv @ G @ Wu


def default_orders(monitor, scene, grid, n_medium, kpar):
    """The half widths ``(M_u, M_v)``: every order propagating at any requested wavelength."""
    P = monitor.axis
    out = []
    for a, kp in zip(((P + 1) % 3, (P + 2) % 3), kpar):
        if is_axis1d(scene, a):
            out.append(0)                   # one cell, exp(i k a): the single order 0
            continue
        L = grid.size[a]
        k_max = n_medium * 2.0 * math.pi / min(monitor.wavelengths)
        m = 0
        while any(abs(kp + 2.0 * math.pi * s * (m + 1) / L) < k_max for s in (-1, 1)):
            m += 1
        out.append(m)
    return tuple(out)


def _concrete(v):
    from wavebit.materials.time_domain import concrete
    return concrete(v)
