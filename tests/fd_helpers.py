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

**Stage 1's scenes** are at the bottom of this file: a travelling Gaussian packet
(:class:`Packet`), the doubling pair the CPML gate measures with (:func:`doubling_pair`), a
slab scene (:func:`slab_scene`) and a half-space scene (:func:`half_space_scene`) for the
Airy, Fresnel and decay gates that come after it.

**Stage 2's scenes** follow them: a dipole in a PML box (:func:`dipole_box`), a PlaneWave in
a Bloch cell (:func:`bloch_plane_wave`), a TFSF box in vacuum or on a substrate
(:func:`tfsf_box`), and the readers the source gates share (:func:`scattered_region_peak`,
:func:`ring_monitors`, :func:`peak_of`).
"""
import math

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._lower import lower
from wavebit.fd._run import _results, execute
from wavebit.fd.results import Series
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


# ---------------------------------------------------------------------------------------------
# Stage 1: a travelling packet, the doubling pair, and the material scenes
# ---------------------------------------------------------------------------------------------

def run_mutated(scene, plan=None, backend='numpy', mutate_params=None):
    """``fd.run`` with the lowered params passed through ``mutate_params`` first.

    The CPML's mutants -- ``kappa = 1`` everywhere, ``alpha = 0`` -- are changes to the
    *profiles*, which are traced leaves and live in the params, not in the plan. This is the
    hook that lets a gate measure the mutant by exactly the method it measures the real
    absorber by, with nothing else moved. ``mutate_params(params, scene, plan, rt)``.
    """
    rt = runtime(backend)
    p = fd.plan(scene) if plan is None else plan
    params = lower(scene, p, rt)
    if mutate_params is not None:
        params = mutate_params(params, scene, p, rt)
    return _results(scene, p, execute(p, params, rt), rt, params)


class Packet:
    """A Gaussian wave packet travelling in +z with a fixed transverse wavevector.

    **Why the transverse box is one cell.** The transverse dependence is carried entirely by
    the Bloch phase, and the discrete transverse operator a one-cell period gives is exactly
    the one a many-cell period gives: the forward difference is ``(exp(i phi) - 1)/dx``, whose
    magnitude is ``2 sin(phi/2)/dx`` -- the same ``K`` as ``2 sin(k dx/2)/dx`` with
    ``phi = k dx``. So an oblique-incidence gate costs a 1 x 1 x N grid and not N^3, and
    nothing is approximated by that, which is what makes a 0-80 degree table affordable.

    **Why it is a sum over kz and not one plane wave with an envelope.** An envelope on a
    single plane wave is not a solution; the part of its spectrum with ``kz < 0`` travels the
    other way. At grazing that part is not small -- ``kz = k cos(theta)`` is 0.17 k at 80
    degrees -- so the packet is an explicit superposition of exact discrete eigenmodes, each
    with its own polarisation, frequency and H phase. Every member travels in +z, so the
    packet does; :func:`backward_content` is the check that it does.

    **The H phase, which the stage-0 helpers do not need.** ``plane_wave_box`` sets
    ``H(-dt/2) = (dt/2) curl E0``, which is a *standing* wave in time: it satisfies the
    recurrence and so passes the eigenmode gates, but it is the sum of the +w and -w modes. A
    travelling mode needs ``H(-dt/2) = h curl E0`` with ``h = dt exp(i w dt/2) /
    (2i sin(w dt/2))``, which is what ``H(1/2) - H(-1/2) = -dt curl E(0)`` together with
    ``H(1/2) = H(-1/2) exp(-i w dt)`` gives.

    ``real=True`` takes the real part of both fields, which is still a travelling solution
    (the update is real-linear) and is what a run with no Bloch face needs -- a zero Bloch
    phase being refused, normal incidence is exactly that run.
    """

    def __init__(self, wavelength, angle, dx, dt, z0, width=None, samples=71, span=3.5,
                 real=False, direction=1):
        self.wavelength, self.dx, self.dt, self.z0, self.real = wavelength, dx, dt, z0, real
        self.direction = direction
        k = 2 * math.pi / wavelength
        self.theta = math.radians(angle)
        self.kx, self.kz0 = k * math.sin(self.theta), k * math.cos(self.theta)
        # sigma_k = kz0/4 over +-3.5 sigma keeps the whole band clear of kz = 0 (the lowest
        # sample is kz0/8) at every angle; the 1/e half width of the z envelope is 2/sigma_k,
        # so the packet gets longer as the angle grazes -- 1.3 wavelengths at 0 degrees and
        # 7.3 at 80, which is what makes the grazing rows of a table the expensive ones.
        #
        # **The pedestal is subtracted, and that is not cosmetic.** A Gaussian truncated at
        # 3.5 sigma steps to 4.8e-6 at its edge, and a step in the spectrum is a sinc tail in
        # z: measured, a plain truncated sum at 2.5 sigma put 1.4e-4 of the peak five
        # wavelengths behind the packet, which is a floor ABOVE the reflection a 12-cell CPML
        # produces -- the first version of this helper measured its own window function and
        # called it a PML. Subtracting the pedestal makes the weight continuous at the edge
        # and drops the floor to ~1e-8.
        self.width = (8.0 / self.kz0) if width is None else width
        sigma_k = 2.0 / self.width
        u = np.linspace(-span, span, samples)
        self.weights = np.exp(-u ** 2) - math.exp(-span ** 2)
        # ``direction = -1`` negates every kz, which turns each member into the exact discrete
        # eigenmode travelling the other way -- the construction below is generic in k, and
        # ``discrete_omega`` is even in it. It is how a gate measures the layer on the LOW face
        # of an axis by the same method, with nothing else in the scene moved.
        self.kz = direction * (self.kz0 + sigma_k * u)
        self.modes = [self._mode(kz) for kz in self.kz]

    def _mode(self, kz):
        k = (self.kx, 0.0, kz)
        K = _K(k, self.dx)
        norm = math.hypot(K[0], K[2])
        # s (out of the x-z plane) and p (in it) summed, so all six components are nonzero
        # and both psi of a z face carry something.
        s, p = (0.0, 1.0, 0.0), (K[2] / norm, 0.0, -K[0] / norm)
        A = tuple((si + pi) / math.sqrt(2.0) for si, pi in zip(s, p))
        omega = discrete_omega(k, self.dx, self.dt)
        KxA = tuple(K[(c + 1) % 3] * A[(c + 2) % 3] - K[(c + 2) % 3] * A[(c + 1) % 3]
                    for c in range(3))
        h = self.dt * np.exp(1j * omega * self.dt / 2) / (2j * math.sin(omega * self.dt / 2))
        return k, A, KxA, omega, h

    def _sum(self, c, which):
        def fn(x, y, z):
            total = np.zeros(np.broadcast(x, y, z).shape, dtype=complex)
            for w, (k, A, KxA, omega, h) in zip(self.weights, self.modes):
                phase = np.exp(1j * (k[0] * x + k[2] * (z - self.z0)))
                total = total + w * (A[c] if which == 'E' else h * 1j * KxA[c]) * phase
            return total.real if self.real else total
        return fn

    def initial(self, amplitude=1.0):
        """The ``fd.InitialField`` of this packet: E at ``t = 0``, H at ``t = -dt/2``."""
        return fd.InitialField(E=tuple(self._sum(c, 'E') for c in range(3)),
                               H=tuple(self._sum(c, 'H') for c in range(3)),
                               amplitude=amplitude)

    @property
    def vz(self):
        """The group *speed* along z: ``cos(theta)`` in vacuum, with ``c = 1``.

        Unsigned, because every caller uses it to turn a distance into a step count;
        ``direction`` says which way the distance is measured.
        """
        return math.cos(self.theta)


def packet_layout(cells, angle, wavelength=1.0, cells_per_wavelength=20, courant=0.5,
                  extent=2.5, delay=5.0, tail=2.0):
    """The geometry and step count of one doubling measurement, as plain numbers.

    ``extent`` is the packet half-length in units of its 1/e width ``W`` (amplitude 2e-3
    there), ``delay`` sets how much further the reference box's layer sits (``delay * W``,
    which is what puts the reference's own echo outside the window), and ``tail`` is how long
    past the reflection the window runs. Everything scales with ``W = 0.955 / cos(theta)``
    wavelengths, so the grazing rows are the expensive ones: at 80 degrees the packet is 5.5
    wavelengths long, because a shorter one would carry negative-kz content.
    """
    dx = wavelength / cells_per_wavelength
    dt = courant * dx / math.sqrt(3.0)
    theta = math.radians(angle)
    W = 8.0 / (2 * math.pi / wavelength * math.cos(theta))
    w_cells = W / dx
    interior = int(math.ceil(2 * extent * w_cells)) + 12
    extra = int(math.ceil(delay * w_cells))
    nz = cells + interior + cells
    z0_index = cells + extent * w_cells
    probe_index = cells + interior - 3
    pml_index = cells + interior                      # the inner face of the z+ layer
    reach = ((pml_index - z0_index) + (pml_index - probe_index)) * dx + tail * W
    steps = int(math.ceil(reach / (dt * math.cos(theta))))
    k = 2 * math.pi / wavelength
    omega = discrete_omega((k * math.sin(theta), 0.0, k * math.cos(theta)), dx, dt)
    return dict(dx=dx, dt=dt, W=W, nz=nz, extra=extra, z0_index=z0_index,
                probe_index=probe_index, steps=steps, wavelength=wavelength, angle=angle,
                cells=cells, courant=courant, omega=omega)


def doubling_pair(cells=12, angle=0.0, layout=None, allow_thin=True, toward='+', **kwargs):
    """``(test_scene, reference_scene, layout)``: the same packet, the PML at two distances.

    The doubling method in the form an initial-field engine can run it: two boxes whose cells
    coincide one for one over the whole of the shorter, differing only in how far the layer
    under test is. Everything up to that layer -- the packet, the layer behind it, the probe's
    cell -- is the same problem on the same cells, so the difference at the probe is that
    layer's reflection and nothing else; round-off between two runs of different lengths is
    ~1e-14 of the peak, ten orders below the smallest number in the table. The reference's own
    reflection arrives ``2 * delay * W / cos(theta)`` later, which the window ends before.

    ``toward`` is **which z face is under test**, and it exists because the two are different
    code: ``_lower/cpml.profiles`` measures depth from the inner face with its own branch per
    side, and on a low face the PEC wall is the node ``x_0`` where on a high face it is the pad
    ``x_N``. With ``'+'`` (the default) the box runs from ``z = 0`` up and the packet travels
    in +z; with ``'-'`` it runs from ``z = 0`` down, the packet travels in -z, and the
    reference's extra cells are added at the low end -- the same geometry, mirrored, so the
    two calls differ in nothing but which branch of ``profiles`` is in the path.

    The transverse faces are ``Bloch`` (or ``Periodic`` at normal incidence, where a zero
    phase is refused and the run is real). The probe is three cells in front of the layer, so
    the reflection is read as it leaves rather than after another trip.
    """
    layout = packet_layout(cells, angle, **kwargs) if layout is None else layout
    dx, dt, nz = layout['dx'], layout['dt'], layout['nz']
    sign = 1 if toward == '+' else -1
    # Every position is measured from the face of the box the geometry is anchored at: the low
    # corner when the packet travels in +z, the high one when it travels in -z.
    packet = Packet(layout['wavelength'], angle, dx, dt, z0=sign * layout['z0_index'] * dx,
                    real=(angle == 0.0), direction=sign)
    phi = packet.kx * dx
    x_face = fd.Periodic() if phi == 0.0 else fd.Bloch(phase=phi)
    pml = fd.PML(cells=cells, wavelength=layout['wavelength'], allow_thin=allow_thin)
    probe_z = sign * layout['probe_index'] * dx
    monitors = tuple(fd.Record(f'E{a}', at=(0.0, 0.0, probe_z), component=f'E{a}')
                     for a in 'xyz')

    def scene(n):
        return fd.Scene(domain=fd.Domain(size=(dx, dx, n * dx),
                                         center=(0.0, 0.0, sign * n * dx / 2)),
                        boundaries=fd.Boundaries(x=x_face, y=fd.Periodic(), z=pml),
                        resolution=fd.Resolution(dx=dx),
                        run=fd.Run(steps=layout['steps'], courant=layout['courant']),
                        initial=packet.initial(), monitors=monitors)
    return scene(nz), scene(nz + layout['extra']), layout


def amplitude_at(series, omega):
    """``sum_n f(t_n) exp(+i omega t_n)``: the ``e^{-i w t}`` convention's positive-frequency amplitude.

    The transform is in the *test*, not in the engine -- stage 1 records and nothing else,
    and DFT monitors are stage 3.
    """
    f = np.asarray(to_numpy(series.values))
    return complex(np.sum(f * np.exp(1j * omega * np.asarray(series.t))))


def doubling_reflection(cells=12, angle=0.0, backend='numpy', mutate_params=None, **kwargs):
    """``(R at the carrier, max |E - E_ref| / max |E_ref|, layout)`` -- what the layer reflected.

    **Why the number that goes in a table is the one at a single frequency.** The packet has
    a fixed transverse wavevector and a band of ``kz``, so its *angle* varies across its band
    -- at 60 degrees nominal, ``kz`` runs over ``[0.06, 0.94] k`` and the angle over 20 to 86
    degrees. A worst-over-time figure is therefore the worst angle in the band and not the
    angle in the column heading, which is how the first version of this table came out
    *rising* with the layer depth: the grazing tail of every band dominates, and it dominates
    more the better the layer is at the angle asked for. One frequency pins one angle exactly
    (``kx`` is fixed, so ``sin theta = kx / w``), which is what an angle column has to mean.

    Both numbers are returned: the spectral one for the table, the worst-in-time one because
    it is the quantity a user actually suffers and is the one a mutant moves first.
    ``mutate_params`` is applied to both runs, so a mutant of the absorber (``kappa = 1``,
    ``alpha = 0``) is measured by exactly this method with nothing else moved.

    The returned ``layout`` also carries, of the reference probe's loudest component,
    ``reference_peak`` (the largest value it saw) and ``reference_initial`` (what it saw at
    ``t = 0``). Both ratios above are *relative*, so a measurement of exactly zero means either
    "the layer absorbed everything" or "nothing ever reached the probe", and those two must not
    look alike to a gate: an upper bound on a ratio alone is satisfied by an identity kernel
    (fields frozen, observers still recording), which returns 0.0 from both numbers here. The
    pair separates them by physics rather than by tolerance -- the probe starts in the packet's
    2e-3 tail and ends in its peak, so ``reference_peak / reference_initial`` is ~500 for a run
    that propagates and exactly 1 for one that does not.
    """
    test, reference, layout = doubling_pair(cells, angle, **kwargs)
    got, want = [run_mutated(s, backend=backend, mutate_params=mutate_params)
                 for s in (test, reference)]
    omega = layout['omega']
    spectral, worst, loudest = 0.0, 0.0, (0.0, 0.0)
    for name in got:
        a, b = got[name], want[name]
        values = to_numpy(b.values)
        peak = float(np.max(np.abs(values)))
        if peak == 0.0:
            continue
        difference = Series(name=name, component=b.component, t=b.t,
                            values=to_numpy(a.values) - values)
        reference_amplitude = abs(amplitude_at(b, omega))
        loudest = max(loudest, (peak, float(abs(values[0]))))
        if reference_amplitude > 0:
            spectral = max(spectral, abs(amplitude_at(difference, omega))
                           / reference_amplitude)
        worst = max(worst, float(np.max(np.abs(to_numpy(a.values) - values))) / peak)
    layout = dict(layout, reference_peak=loudest[0], reference_initial=loudest[1])
    return spectral, worst, layout


#: The packet's half-length in units of its 1/e width ``W``: the amplitude is 2e-3 of the peak
#: there, which is what :func:`packet_layout` calls ``extent`` and means the same thing.
PACKET_EXTENT = 2.5


def slab_scene(thickness=0.2, eps=2.25, material=None, cells=12, wavelength=1.0,
               cells_per_wavelength=20, angle=0.0, gap=1.0, courant=0.5, steps=None,
               probe_offset=0.5, half_space=False):
    """A packet in +z, a slab of ``material`` across the box, PML on z, Bloch across.

    The scene the Airy gate (a dielectric slab) and the conductive-decay gate drive, and --
    through :func:`half_space_scene` -- the Fresnel gate too. Returns
    ``(scene, packet, geometry)``.

    **The standoff is measured from the packet's ends, not from its centre.** The packet's own
    half-length is ``PACKET_EXTENT * W``, which is 2.4 wavelengths at normal incidence and
    grows as the angle grazes, so a clearance stated about the *centre* puts most of the packet
    inside the slab at ``t = 0``. Here the tail starts ``gap`` wavelengths past the ``z-``
    layer's inner face and the slab starts ``gap`` wavelengths past the head, so nothing is
    inside the structure at ``t = 0`` -- which is what makes a pole memory starting at rest
    exact rather than merely small, and why the pole term's one-step shift costs nothing.
    (An earlier version of this helper measured ``gap`` from the centre and said in its
    docstring that it did not. Two gate writers hit it and both wrote their own scene pair.)

    The two probes are ``probe_offset`` wavelengths in front of the slab and behind it, so a
    gate can take the incident, reflected and transmitted series by windowing them -- the
    engine only records, and the DFT belongs in the test (monitors are stage 3).

    ``half_space=True`` runs the medium out through the ``z+`` face instead of ending it, and
    makes that face an ``fd.Absorber`` -- see :func:`half_space_scene`. The second probe is
    then *inside* the medium rather than behind it, and ``geometry['back']`` is the box's end.
    """
    dx = wavelength / cells_per_wavelength
    dt = courant * dx / math.sqrt(3.0)
    material = fd.Material(eps=eps) if material is None else material
    W = 8.0 / (2 * math.pi / wavelength * math.cos(math.radians(angle)))
    half = PACKET_EXTENT * W
    # Two clearances and the whole packet between the layer's inner face and the slab.
    lead = int(math.ceil((2.0 * gap * wavelength + 2.0 * half) / dx))
    body = int(math.ceil(thickness / dx))
    nz = cells + lead + body + lead + cells
    z_lo = 0.0
    z0 = z_lo + cells * dx + gap * wavelength + half
    front = z_lo + (cells + lead) * dx
    back = z_lo + nz * dx if half_space else front + body * dx
    packet = Packet(wavelength, angle, dx, dt, z0=z0, real=(angle == 0.0))
    if steps is None:
        steps = int(math.ceil((2.0 * half + (2 * lead + body) * dx) / (dt * packet.vz)))
    phi = packet.kx * dx
    x_face = fd.Periodic() if phi == 0.0 else fd.Bloch(phase=phi)
    z_faces = (fd.PML(cells=cells, wavelength=wavelength),
               fd.Absorber(cells=cells) if half_space
               else fd.PML(cells=cells, wavelength=wavelength))
    monitors = tuple(fd.Record(name, at=(0.0, 0.0, z), component='Ey') for name, z in (
        ('incident', front - probe_offset * wavelength),
        ('transmitted', front + body * dx + probe_offset * wavelength)))
    scene = fd.Scene(domain=fd.Domain(size=(dx, dx, nz * dx),
                                      center=(0.0, 0.0, z_lo + nz * dx / 2)),
                     boundaries=fd.Boundaries(x=x_face, y=fd.Periodic(), z=z_faces),
                     resolution=fd.Resolution(dx=dx),
                     run=fd.Run(steps=steps, courant=courant),
                     initial=packet.initial(),
                     structures=(fd.Structure(fd.Slab(z=(front, back)), material),),
                     monitors=monitors)
    return scene, packet, dict(front=front, back=back, dx=dx, dt=dt, nz=nz,
                               steps=steps, cells=cells)


def half_space_scene(material=None, **kwargs):
    """A medium that runs out through the ``z+`` face: **one** interface, and no far one.

    What the Fresnel gate wants: ``R(w)`` is the bare Fresnel coefficient of ``eps(w)`` and not
    an Airy sum. So the medium is not a thick slab -- a thick slab has a back face whose echo
    arrives inside the window -- but a structure that reaches the boundary, which is exactly
    the case ``fd.Absorber`` exists for (a ``PML`` is matched to one ``(eps, mu)`` and refuses a
    conductive or dispersive medium crossing it by name).

    ``thickness`` still sets how much medium sits in front of the ``z+`` layer, and the second
    probe sits inside the medium rather than behind it.
    """
    kwargs.setdefault('thickness', 6.0 * kwargs.get('wavelength', 1.0))
    return slab_scene(material=material, half_space=True, **kwargs)


# ---------------------------------------------------------------------------------------------
# Stage 2: the source scenes
# ---------------------------------------------------------------------------------------------

#: The band every stage-2 helper uses unless told otherwise: centre wavelength 1, spectral
#: sigma 0.15 f0, so ``band.shortest_wavelength`` is 0.642 and 20 cells per centre wavelength is
#: 12.8 at the short edge -- above the CoarseMeshWarning line in vacuum.
BAND = dict(wavelength=1.0, df=0.15)

#: The six components, in the order ``outside_peak`` reads them.
COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')


def band(**changes):
    """``fd.Band`` with :data:`BAND` overridden by ``changes``."""
    return fd.Band(**dict(BAND, **changes))


def dipole_box(center=(0.0, 0.0, 0.0), component='Ez', kind='electric', size=1.5,
               cells_per_wavelength=20, pml=12, steps=None, periods=None, stamp='bspline',
               src_band=None, rng=None, amplitude=1.0, phase=0.0, delay=0.0, monitors=(),
               courant=0.5, grid_wavelength=None, pml_wavelength=None):
    """A point dipole at ``center`` in an open vacuum box: ``size`` of interior, PML all round.

    The PML's wavelength is left out on purpose -- it is inferred from the band centre, which
    is what stage 2 made legal. ``steps`` or ``periods`` (default: 8 periods of the band
    centre, which covers the pulse, ``2 t0 = 7.4 / (2 pi df)`` long at the default band, and
    a little of its ring-down). Returns the scene; ``res.sources['d']`` is the dipole's
    reference series.
    """
    b = band() if src_band is None else src_band
    # A traced band centre (the AD gate's) cannot size the grid or grade the PML, and a
    # central difference in it must not move either -- so ``grid_wavelength`` fixes dx and
    # ``pml_wavelength`` the layer whenever they are given (or the centre is traced).
    wavelength = grid_wavelength or b.concrete_wavelength or BAND['wavelength']
    dx = wavelength / cells_per_wavelength
    if pml_wavelength is None and b.concrete_wavelength is None:
        pml_wavelength = wavelength
    layer = fd.PML(cells=pml, wavelength=pml_wavelength)
    side = size + 2 * pml * dx
    run = (fd.Run(steps=steps, courant=courant) if steps is not None
           else fd.Run(periods=8 if periods is None else periods, courant=courant))
    source = fd.Dipole('d', center=center, component=component, kind=kind, band=b,
                       stamp=stamp, range=rng, amplitude=amplitude, phase=phase, delay=delay)
    return fd.Scene(domain=fd.Domain(size=(side, side, side)),
                    boundaries=fd.Boundaries.open(layer),
                    resolution=fd.Resolution(dx=dx), run=run, sources=(source,),
                    monitors=tuple(monitors))


def bloch_plane_wave(theta=0.0, phi=0.0, polarization='s', direction='-z', at=0.0,
                     cells_per_wavelength=20, length=3.0, transverse=(0.25, 0.2), pml=12,
                     steps=600, background=None, structures=(), monitors=(), src_band=None,
                     amplitude=1.0, phase=0.0, delay=0.0):
    """A PlaneWave in a periodic (``theta = 0``) or Bloch cell, PML on the propagation axis z.

    ``length`` is the z extent of the interior; the transverse faces come from
    ``fd.Bloch.from_source`` in the given background, so the cell is consistent by
    construction. Returns ``(scene, pw)``.
    """
    b = band() if src_band is None else src_band
    dx = float(b.wavelength) / cells_per_wavelength
    pw = fd.PlaneWave('pw', z=at, direction=direction, theta=theta, phi=phi,
                      polarization=polarization, band=b, amplitude=amplitude, phase=phase,
                      delay=delay)
    faces = fd.Bloch.from_source(pw, background=background)
    scene = fd.Scene(domain=fd.Domain(size=(transverse[0], transverse[1],
                                            length + 2 * pml * dx)),
                     boundaries=fd.Boundaries(**faces, z=fd.PML(cells=pml)),
                     resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps),
                     background=background, structures=tuple(structures),
                     sources=(pw,), monitors=tuple(monitors))
    return scene, pw


def tfsf_box(direction='-z', polarization='x', box=(0.5, 0.5, 0.5), center=(0.0, 0.0, 0.0),
             substrate=None, cells_per_wavelength=20, margin=0.5, pml=12, steps=500,
             monitors=(), src_band=None, amplitude=1.0, phase=0.0, delay=0.0,
             background=None):
    """A TFSF box in an open box (PML all round), in vacuum or on a substrate.

    ``margin`` is the scattered-field clearance on every side between the box and the PML's
    inner face. ``substrate=(eps, top)`` adds a slab across the propagation axis from ``top``
    down through the low face of that axis (a half-space running out through the PML, which a
    lossless ``fd.Material`` may do) -- the layered aux-grid background the TFSF gate's Airy
    case needs; a box that crosses ``top`` is what that gate asks for. Returns the scene.
    """
    b = band() if src_band is None else src_band
    dx = float(b.wavelength) / cells_per_wavelength
    size = tuple(s + 2 * margin + 2 * pml * dx for s in box)
    P = 'xyz'.index(direction[1])
    structures = ()
    if substrate is not None:
        eps, top = substrate
        low = center[P] - size[P] / 2 - dx
        structures = (fd.Structure(fd.Slab(**{direction[1]: (low, top)}),
                                   fd.Material(eps=eps)),)
    source = fd.TFSF('t', box=fd.Box(size=box, center=center), direction=direction,
                     polarization=polarization, band=b, amplitude=amplitude, phase=phase,
                     delay=delay)
    return fd.Scene(domain=fd.Domain(size=size, center=center),
                    boundaries=fd.Boundaries.open(fd.PML(cells=pml)),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps),
                    background=background, structures=structures, sources=(source,),
                    monitors=tuple(monitors))


def ring_monitors(points, prefix='p'):
    """Six ``fd.Record`` per point, named ``f'{prefix}{i}:{component}'``."""
    return tuple(fd.Record(f'{prefix}{i}:{c}', at=tuple(at), component=c)
                 for i, at in enumerate(points) for c in COMPONENTS)


def peak_of(res, prefix):
    """``max |value|`` over every recorded series whose name starts with ``prefix``.

    A non-finite series is refused rather than compared: Python's ``max`` keeps whichever of
    ``nan`` and a number it saw first, so a run that blew up could read as a small peak.
    """
    peaks = [float(np.max(np.abs(to_numpy(res[name].values))))
             for name in res if name.startswith(prefix)]
    assert peaks and np.all(np.isfinite(peaks)), f'non-finite or no series under {prefix!r}'
    return max(peaks)


def box_node_range(plan, source):
    """``(lo, hi)`` node indices per axis (x, y, z) a ``fd.TFSF`` box snapped to."""
    grid = plan.grid
    lo = tuple(int(round((source.box.low[a] - grid.low[a]) / grid.dx)) for a in range(3))
    hi = tuple(int(round((source.box.high[a] - grid.low[a]) / grid.dx)) for a in range(3))
    return lo, hi


def scattered_region_peak(res, source):
    """``max |F|`` over the final E and H outside the TFSF box's correction layer.

    The correction layer is one cell either side of the snapped box; everything beyond it is
    pure scattered field, which an empty box must leave at round-off. A non-finite field is
    refused rather than compared (``max(0.0, nan)`` is ``0.0``, so a run that blew up would
    otherwise read as a perfect box).
    """
    lo, hi = box_node_range(res.plan, source)
    peak = 0.0
    for F in (res.final.E, res.final.H):
        F = np.abs(to_numpy(F))
        assert np.all(np.isfinite(F)), 'the run produced a non-finite field'
        mask = np.ones(F.shape[1:], dtype=bool)
        mask[lo[2] - 1:hi[2] + 2, lo[1] - 1:hi[1] + 2, lo[0] - 1:hi[0] + 2] = False
        peak = max(peak, float(F[:, mask].max()))
    return peak
