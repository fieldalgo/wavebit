"""The canonical small scenes every ``tests/test_fd_*.py`` gate runs, and their closed forms.

One place, so that every gate drives the same boxes: a PEC box and a PMC box carrying one
exact Yee cavity mode, a periodic box and a Bloch box carrying one plane wave, and the
symmetry pair (a half box with a mirror, the full box it is half of). Grids are 16^3 or
smaller and runs are a few hundred steps, so a gate takes seconds on every backend.

**The discrete cavity mode.** On the Yee grid with ``k = (l pi/Lx, m pi/Ly, n pi/Lz)`` measured
from the walls, ``sin`` and ``cos`` at the staggered points are exact eigenfunctions of the
discrete differences with the *discrete* wavenumbers ``K_i = 2 sin(k_i dx/2)/dx``: a forward
difference of ``sin(k(x - w))`` at the nodes is ``K cos(k(x - w))`` at the half points, and so
on. A field ``E = (A c s s, B s c s, C s s c)`` (``s``/``c`` for ``sin``/``cos`` along x, y, z in
that order) with ``A Kx + B Ky + C Kz = 0`` is therefore discretely divergence-free and an
eigenvector of the discrete ``curl curl`` with eigenvalue ``Omega^2 = sum K_i^2``; the
leapfrog then advances it at the discrete frequency

    sin^2(w dt/2) / dt^2 = sum_i sin^2(k_i dx/2) / dx^2

and any linear functional of the state obeys ``f(n+1) + f(n-1) = 2 cos(w dt) f(n)``. That
is what the eigenmode gates check, to round-off. With ``H(-dt/2) = (dt/2) curl E0`` the phase
is exact too: ``E(n) = E0 cos(w n dt)`` and ``H(n + 1/2) = h curl E0 sin(w (n + 1/2) dt)`` with
``h = -dt / (2 sin(w dt/2))``. The PMC box is the dual: ``H`` carries the mode, measured from
the PMC walls half a cell outside the E nodes, ``H(-dt/2) = H0 cos(w dt/2)``, ``E(0) = 0``,
and ``E(n) = e curl H0 sin(w n dt)`` with ``e = +dt / (2 sin(w dt/2))``.

The recurrence residual is what makes a *wall position* testable: a wall one cell off
changes ``k`` by ``1/N`` and the residual by ~1e-2 against a round-off tolerance.
"""
import math

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._lower import lower
from wavebit.fd._run import _results, execute
from wavebit.fd._runtime import runtime

BACKENDS = ('numpy', 'jax', 'torch')

#: The probe every gate reads: off every node and every symmetry plane of a unit box.
PROBE = (0.1, 0.0, -0.05)


def backends():
    """``pytest.mark.parametrize`` values: every backend, each skipped if not installed."""
    return [pytest.param(b, marks=() if _installed(b) else pytest.mark.skip(f'{b} not installed'))
            for b in BACKENDS]


def _installed(backend):
    if backend == 'numpy':
        return True
    try:
        __import__(backend)
    except ImportError:
        return False
    return True


def to_numpy(x):
    """A NumPy view of a backend value (a JAX array, a Torch tensor, or NumPy already)."""
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def run_plan(scene, plan, backend='numpy'):
    """``fd.run`` on a plan that may have been mutated -- no skeleton check (tests only)."""
    rt = runtime(backend)
    return _results(scene, plan, execute(plan, lower(scene, plan, rt), rt), rt)


def discrete_omega(k, dx, dt):
    """The leapfrog frequency of wavevector ``k`` on a uniform Yee grid."""
    s = sum(math.sin(ki * dx / 2) ** 2 for ki in k) / dx ** 2
    return 2 / dt * math.asin(dt * math.sqrt(s))


def recurrence_residual(values, omega, dt):
    """max |f(n+1) + f(n-1) - 2 cos(w dt) f(n)| / max |f| over the series."""
    f = np.asarray(values)
    r = f[2:] + f[:-2] - 2 * math.cos(omega * dt) * f[1:-1]
    return float(np.max(np.abs(r)) / np.max(np.abs(f)))


# ---------------------------------------------------------------------------------------------
# Cavity modes
# ---------------------------------------------------------------------------------------------

def _K(k, dx):
    return tuple(2 * math.sin(ki * dx / 2) / dx for ki in k)


class CavityMode:
    """The mode ``(A c s s, B s c s, C s s c)`` from walls ``w`` with size ``L``, and its curl.

    ``field`` is the field that carries the pattern (``'E'`` in a PEC box, ``'H'`` in a PMC
    one); ``curl`` evaluates the other field's pattern at *its* Yee points.
    """

    def __init__(self, w, L, lmn, dx, A=1.0, B=0.5):
        self.w, self.L = tuple(w), tuple(L)
        self.k = tuple(li * math.pi / Li for li, Li in zip(lmn, L))
        self.K = _K(self.k, dx)
        Kx, Ky, Kz = self.K
        self.amp = (A, B, -(A * Kx + B * Ky) / Kz)      # discretely divergence-free
        self.omega2 = sum(K ** 2 for K in self.K)

    def _sc(self, x, y, z):
        u = [(c - w) * k for c, w, k in zip((x, y, z), self.w, self.k)]
        return [np.sin(v) for v in u], [np.cos(v) for v in u]

    def component(self, c):
        A = self.amp[c]

        def fn(x, y, z):
            s, cs = self._sc(x, y, z)
            f = [s[0], s[1], s[2]]
            f[c] = cs[c]
            return A * f[0] * f[1] * f[2]
        return fn

    def curl_component(self, c):
        """``(curl F)_c`` of the pattern, analytically, with the discrete wavenumbers."""
        a, b = (c + 1) % 3, (c + 2) % 3
        coef = self.amp[b] * self.K[a] - self.amp[a] * self.K[b]

        def fn(x, y, z):
            s, cs = self._sc(x, y, z)
            f = [cs[0], cs[1], cs[2]]
            f[c] = s[c]
            return coef * f[0] * f[1] * f[2]
        return fn

    def value(self, c, at):
        return float(self.component(c)(*at))

    def curl_value(self, c, at):
        return float(self.curl_component(c)(*at))


def cavity_box(kind, N=16, L=1.0, steps=300, lmn=(1, 1, 1), probes=None, courant=0.5):
    """A PEC (``kind='pec'``) or PMC box of ``N^3`` cells carrying one exact mode.

    Returns ``(scene, mode, omega)``. The walls are where the plan says they are: the box
    spans ``[-L/2, L/2]`` and a PEC wall is on the E nodes, a PMC wall half a cell outside.
    """
    dx = L / N
    dt = courant * dx / math.sqrt(3)
    face = fd.PEC() if kind == 'pec' else fd.PMC()
    lo = -L / 2
    w = (lo, lo, lo) if kind == 'pec' else (lo - dx / 2,) * 3
    mode = CavityMode(w, (L, L, L), lmn, dx)
    omega = discrete_omega(mode.k, dx, dt)
    if kind == 'pec':
        # E carries the mode; H half a step earlier is (dt/2) curl E0, exactly in phase.
        initial = fd.InitialField(
            E=tuple(mode.component(c) for c in range(3)),
            H=tuple((lambda c: (lambda x, y, z: (dt / 2) * mode.curl_component(c)(x, y, z)))(c)
                    for c in range(3)))
    else:
        cos_half = math.cos(omega * dt / 2)
        initial = fd.InitialField(
            H=tuple((lambda c: (lambda x, y, z: cos_half * mode.component(c)(x, y, z)))(c)
                    for c in range(3)))
    if probes is None:
        probes = (fd.Record('Ey', at=PROBE, component='Ey'),
                  fd.Record('Hz', at=PROBE, component='Hz'))
    scene = fd.Scene(domain=fd.Domain(size=(L, L, L)),
                     boundaries=fd.Boundaries(x=face, y=face, z=face),
                     resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps, courant=courant),
                     initial=initial, monitors=tuple(probes))
    return scene, mode, omega


def closed_form(kind, mode, omega, plan, series):
    """What the recorded ``series`` should be, exactly, for the cavity box of ``kind``."""
    dt = plan.dt
    o = next(o for o in plan.observers if o.name == series.name)
    at = tuple(v[i] for v, i in zip(plan.grid.yee_points(o.field, o.c), (o.i, o.j, o.k)))
    t = series.t
    if kind == 'pec':
        if o.field == 'E':
            return mode.value(o.c, at) * np.cos(omega * t)
        h = -dt / (2 * math.sin(omega * dt / 2))
        return h * mode.curl_value(o.c, at) * np.sin(omega * t)
    if o.field == 'H':
        return mode.value(o.c, at) * np.cos(omega * t)
    e = dt / (2 * math.sin(omega * dt / 2))
    return e * mode.curl_value(o.c, at) * np.sin(omega * t)


# ---------------------------------------------------------------------------------------------
# Plane waves in periodic and Bloch boxes
# ---------------------------------------------------------------------------------------------

def plane_wave_polarization(k, dx):
    """The unit E polarisation ``(Ky, -Kx, 0)/|(Kx, Ky)|`` of the plane wave with wavevector ``k``.

    Built from the *discrete* wavenumbers so that ``K . E0 = 0`` holds exactly on the grid,
    which is what makes the wave a discrete eigenmode. Every component of ``k`` is nonzero
    in the boxes below, so every axis carries a difference and every pad is read -- a wave
    with ``k_y = 0`` and ``E_y`` only never drives a y-varying field through the y faces,
    and a y-periodic pad reading the wrong layer passed the gate unseen (2026-09-18).
    """
    Kx, Ky, _ = _K(k, dx)
    norm = math.hypot(Kx, Ky)
    return (Ky / norm, -Kx / norm, 0.0)


def plane_wave_box(phase=None, N=16, L=1.0, steps=300, lmn=(1, 1, 1), courant=0.5):
    """A periodic box (``phase=None``) or a Bloch box with ``phase=(phi_x, phi_z)`` on x and z.

    The wave is ``E = A cos(k.r)`` (real, periodic) or ``A exp(i k.r)`` (complex, Bloch) with
    ``k_x = (phi_x + 2 pi l)/L``, ``k_y = 2 pi m/L``, ``k_z = (phi_z + 2 pi n)/L`` and the
    transverse polarisation ``A = plane_wave_polarization(k, dx)``, so that ``K . E0 = 0``
    and the wave is a discrete eigenmode. ``H(-dt/2) = (dt/2) curl E0`` as in the cavity:
    the forward difference of ``cos(k.r)`` lands ``-K sin(k.r)`` on the H points, so
    ``curl E0 = -(K x A) sin(k.r)`` for the real wave and ``i (K x A) exp(i k.r)`` for the
    complex one. Returns ``(scene, k, omega)``.
    """
    dx = L / N
    dt = courant * dx / math.sqrt(3)
    if phase is None:
        bx = bz = fd.Periodic()
        k = tuple(2 * math.pi * li / L for li in lmn)
    else:
        bx, bz = fd.Bloch(phase=phase[0]), fd.Bloch(phase=phase[1])
        k = ((phase[0] + 2 * math.pi * lmn[0]) / L, 2 * math.pi * lmn[1] / L,
             (phase[1] + 2 * math.pi * lmn[2]) / L)
    K = _K(k, dx)
    A = plane_wave_polarization(k, dx)
    KxA = tuple(K[(c + 1) % 3] * A[(c + 2) % 3] - K[(c + 2) % 3] * A[(c + 1) % 3] for c in range(3))
    omega = discrete_omega(k, dx, dt)

    def dot(x, y, z):
        return k[0] * x + k[1] * y + k[2] * z

    if phase is None:
        def E_c(c):
            return lambda x, y, z: A[c] * np.cos(dot(x, y, z))

        def H_c(c):
            return lambda x, y, z: (dt / 2) * (-KxA[c]) * np.sin(dot(x, y, z))
    else:
        def E_c(c):
            return lambda x, y, z: A[c] * np.exp(1j * dot(x, y, z))

        def H_c(c):
            return lambda x, y, z: (dt / 2) * (1j * KxA[c]) * np.exp(1j * dot(x, y, z))
    scene = fd.Scene(domain=fd.Domain(size=(L, L, L)),
                     boundaries=fd.Boundaries(x=bx, y=fd.Periodic(), z=bz),
                     resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps, courant=courant),
                     initial=fd.InitialField(E=(E_c(0), E_c(1), None), H=tuple(H_c(c) for c in range(3))),
                     monitors=(fd.Record('Ey', at=PROBE, component='Ey'),
                               fd.Record('Hx', at=PROBE, component='Hx')))
    return scene, k, omega


# ---------------------------------------------------------------------------------------------
# The symmetry pair
# ---------------------------------------------------------------------------------------------

def mirror_signs(parity, axis):
    """``(sign_E, sign_H)`` per component for a mirror across ``axis`` with ``parity``."""
    sE = tuple(-parity if c == axis else parity for c in range(3))
    sH = tuple(parity if c == axis else -parity for c in range(3))
    return sE, sH


#: Where axis ``a`` (0 = x, 1 = y, 2 = z) sits in a ``[component, z, y, x]`` array.
ARRAY_AXIS = {0: 3, 1: 2, 2: 1}


def symmetry_pair(side, parity, M=8, L_half=0.5, steps=200, far=None, seed=0, courant=0.5,
                  axis=0):
    """A half box with ``Symmetry(parity)`` on the face ``side`` of ``axis`` and the full box it is half of.

    Returns ``(half_scene, full_scene, take)`` where ``take(full_array)`` is the slice of a
    full-box ``(3, nz, ny, nx)`` array that corresponds to the half box, index for index.
    ``axis`` is 0, 1 or 2 for a mirror across x, y or z; the geometry below is written for
    x and the arrays are built with the mirrored axis last, then moved into place, so the
    three axes are the one construction and the x case is unchanged by the argument.

    ``side='-'``: the mirror is the NODE plane on the half box's low face, the full box's
    node ``x_M``. ``side='+'``: the mirror is the HALF plane on the high face, the centre of
    the full box's cell ``M``. How many cells the full box needs for its far walls to sit
    symmetrically about that plane depends on the wall type: PEC walls are on nodes
    (``x_0`` and ``x_N``), so ``N = 2M`` about a node and ``2M + 1`` about a half plane; PMC
    walls are half a cell outside (``x_{-1/2}`` and ``x_{N-1/2}``), so it is ``2M + 1`` about a
    node and ``2M + 2`` about a half plane. The half box is the cells from the plane up
    (``side='-'``) or up to and including the plane's cell (``side='+'``).

    The initial fields are random (seeded) on the half box, with the components that must
    vanish on the plane zeroed, and mirrored into the full box with the parity's signs --
    so the full field is exactly symmetric, bit for bit, and the two runs are the same
    arithmetic in mirror image. ``far`` is the face type on the far face of the axis and on
    the other two axes (PEC by default).
    """
    far = fd.PEC() if far is None else far
    dx = L_half / M
    pmc_far = isinstance(far, fd.PMC)
    if side == '-':
        n_full = 2 * M + (1 if pmc_far else 0)          # plane at node x_M
        n_half = n_full - M                              # cells M .. n_full-1
    else:
        n_full = 2 * M + (2 if pmc_far else 1)           # plane at x_{M+1/2}
        n_half = M + 1                                   # cells 0 .. M
    sx_half, sx_full = n_half * dx, n_full * dx
    nt = 6                                               # cells across the two other axes
    st = nt * dx
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((3, nt, nt, n_half))         # the mirrored axis last, for now
    H = rng.standard_normal((3, nt, nt, n_half))
    sE, sH = mirror_signs(parity, axis)
    tangential = tuple(c for c in range(3) if c != axis)
    # Components on the plane whose sign is odd vanish there. On the node plane those are
    # the components at a node along the axis: tangential E and the normal H; on the half
    # plane the ones half a cell along it: the normal E and tangential H.
    if side == '-':
        plane_layer, on_plane = 0, {'E': tangential, 'H': (axis,)}
    else:
        plane_layer, on_plane = n_half - 1, {'E': (axis,), 'H': tangential}
    for c in on_plane['E']:
        if sE[c] == -1:
            E[c, :, :, plane_layer] = 0.0
    for c in on_plane['H']:
        if sH[c] == -1:
            H[c, :, :, plane_layer] = 0.0
    # Mirror into the full box, index for index. A node component at x_j images to
    # x_{2P-j} about the node x_P and to x_{2P+1-j} about x_{P+1/2}; a half-cell component at
    # x_{j+1/2} images to x_{2P-1-j+1/2} and x_{2P-j+1/2}. An image beyond the far wall is the
    # wall's zero pad.
    fullE = np.zeros((3, nt, nt, n_full))
    fullH = np.zeros((3, nt, nt, n_full))
    # Half-cell offset along the mirrored axis: E_c is off-node along its own axis, H_c
    # along the other two.
    offsets = {'E': tuple(int(c == axis) for c in range(3)),
               'H': tuple(int(c != axis) for c in range(3))}
    start = M if side == '-' else 0                       # the half box's first full-box cell
    for name, half, full, signs in (('E', E, fullE, sE), ('H', H, fullH, sH)):
        for c in range(3):
            o = offsets[name][c]
            full[c, :, :, start:start + n_half] = half[c]
            for j in range(n_full):
                if start <= j < start + n_half:
                    continue
                if side == '-':
                    img = (2 * M - 1 - j) if o == 1 else (2 * M - j)
                else:
                    img = (2 * M - j) if o == 1 else (2 * M + 1 - j)
                if 0 <= img < n_full:
                    full[c, :, :, j] = signs[c] * half[c, :, :, img - start]
    pos = ARRAY_AXIS[axis]
    E, H, fullE, fullH = (np.ascontiguousarray(np.moveaxis(a, 3, pos)) for a in (E, H, fullE, fullH))
    mirror = fd.Symmetry(parity=parity)
    x_half = (mirror, far) if side == '-' else (far, mirror)
    common = dict(resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps, courant=courant))
    # Centre the boxes so that the half box's cells coincide with the full box's.
    cx_full = 0.0
    x_lo_full = cx_full - sx_full / 2
    x_lo_half = x_lo_full + (M * dx if side == '-' else 0.0)
    cx_half = x_lo_half + sx_half / 2
    probe_i = 1 if side == '-' else n_half - 2                 # a cell near the mirror
    probe_x_half = x_lo_half + (probe_i + 0.25) * dx

    def triple(along, others):
        """``(x, y, z)`` with ``along`` on the mirrored axis and ``others`` on the rest, in order."""
        rest = list(others)
        return tuple(along if a == axis else rest.pop(0) for a in range(3))

    probe = triple(probe_x_half, (0.05, -0.05))
    faces_half = {'xyz'[a]: x_half if a == axis else far for a in range(3)}
    half_scene = fd.Scene(domain=fd.Domain(size=triple(sx_half, (st, st)),
                                           center=triple(cx_half, (0, 0))),
                          boundaries=fd.Boundaries(**faces_half),
                          initial=fd.InitialField(E=E, H=H),
                          monitors=(fd.Record('Ey', at=probe, component='Ey'),
                                    fd.Record('Hx', at=probe, component='Hx')),
                          **common)
    full_scene = fd.Scene(domain=fd.Domain(size=triple(sx_full, (st, st)),
                                           center=triple(cx_full, (0, 0))),
                          boundaries=fd.Boundaries(x=far, y=far, z=far),
                          initial=fd.InitialField(E=fullE, H=fullH),
                          monitors=(fd.Record('Ey', at=probe, component='Ey'),
                                    fd.Record('Hx', at=probe, component='Hx')),
                          **common)
    index = [slice(None)] * 4
    index[pos] = slice(M, None) if side == '-' else slice(None, M + 1)
    index = tuple(index)
    take = lambda a: a[index]                               # noqa: E731
    return half_scene, full_scene, take
