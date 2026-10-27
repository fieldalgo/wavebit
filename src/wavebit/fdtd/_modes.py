"""Waveguide modes of the discrete Yee system: ``fdtd.solve_modes``.

A mode here is a solution of the **kernel's own** update equations, not of the continuum
Maxwell equations, so that a mode injected into a run or projected out of one is exact on the
grid. The derivation, the registered acceptance criteria and the measured results are in
``docs/validation/fdtd/fd-waveguide-modes.md``; the manual is ``docs/guide/fdtd/waveguide-modes.md``.

**The discrete eigenproblem.** With ``c = eps0 = mu0 = 1`` and ``H`` stored as ``Z0 H``
(``src/wavebit/fdtd/_kernel/step.py``), a time-harmonic run ``exp(-i omega t)`` satisfies, for phasors taken at
each component's own times (``src/wavebit/fdtd/_post.py``, eq. (*)),

    curl_d E = i Omega mu H,      curl_d H = -i Omega eps E,      Omega = (2/dt) sin(omega dt/2),

with ``curl_d`` the grid's differences divided by ``dx``. On a plane across axis ``w`` take the
right-handed transverse pair ``u = w + 1``, ``v = w + 2`` (mod 3); the Yee offsets are invariant
under that cyclic relabelling. For fields ``exp(i beta w)`` with every component referred to
its own Yee position, the forward and the backward difference along ``w`` both become ``i q``,

    q = (2/dx) sin(beta dx/2),

exactly. Eliminating ``E_w`` and ``H_w`` (each through its own diagonal coefficient and the
masks of the faces) leaves, for ``e = (E_u, E_v)`` and ``h = (H_v, -H_u)``,

    q h = P e,    q e = Q h,    q^2 e = Q P e,
    P = Omega diag(eps_u, eps_v) - (1/Omega) C^T diag(m_Hw / mu_w) C,       C = [-D_v, D_u],
    Q = Omega diag(mu_v, mu_u)   - (1/Omega) G diag(m_Ew / eps_w) G^T,      G = [D_u; D_v],

with ``D_a`` the forward difference along ``a`` (the backward one is ``-D_a^T`` for the zero
and periodic neighbour rules this module admits) and ``m`` the 0/1 wall masks. ``P`` and ``Q``
are real and symmetric, which gives the exact orthogonality of two modes with different
``q^2`` under the unconjugated cross product ``e_m . h_n`` (the derivation is in the
validation record). That orthogonality is what normalizes a mode and what a mode
decomposition projects with.

**Power.** :func:`wavebit.fdtd._post.plane_flux` pairs the node-plane ``E`` with the mean of the
two half planes of ``H``, which for a mode is ``cos(beta dx/2)`` times the ``H`` referred to the
node. A mode is normalized so that this flux -- the one a ``FluxMonitor`` measures -- is 1 W
(1 W/m with one ``InvariantAxis``, 1 W/m^2 with two).

**What is refused** (by name, never approximated): a conductive or dispersive medium in the
window, a medium that changes along ``w`` at the plane, coupled smoothed (constitutive) or
extended-stencil material anywhere in the scene, a ``Bloch`` or ``MirrorSymmetry`` transverse
face, an ``InvariantAxis`` with nonzero ``k``, traced or gradient-carrying inputs (the
eigensolve is a host computation; it does not differentiate), and a mode that is not confined
inside a window edge that truncates the scene (an ``Open`` face, or a finite plane span).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from wavebit._common.backend import eager_under_torch_compile
from wavebit._common.declarations import Declaration, spec, static_float
from wavebit._common.execution import frozen_mapping
from wavebit._common.pytrees import register_dataclass
from wavebit.errors import UnsupportedFeatureError, ValidationError
from wavebit.units import Z0

__all__ = ['solve_modes', 'WaveguideMode', 'WaveguideModes']

#: A truncating window edge (an ``Open`` face or a finite plane span) admits a mode only when
#: its largest transverse E on the outermost cells is at most this fraction of its peak.
CONFINEMENT = 1e-3

#: Relative separation under which two eigenvalues ``q^2`` are one degenerate cluster.
DEGENERATE = 1e-8

#: Largest accepted relative eigen-residual ``|A e - q^2 e| / |q^2 e|``.
RESIDUAL = 1e-8

#: Relative difference under which two layers' coefficients count as the same medium.
UNIFORM = 1e-12

#: Problems with at most this many unknowns are solved densely.
DENSE_LIMIT = 1600

_AXES = 'xyz'
_COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')


@spec
class WaveguideMode(Declaration):
    """One mode of the discrete Yee system on a plane, normalized to unit power along ``+w``.

    ``effective_index`` is ``beta c / omega`` at the physical frequency, with ``beta`` the
    propagation constant of the grid's own dispersion (``beta_per_meter`` in rad/m). ``E`` and
    ``H`` map component names (``'Ex'`` ... ``'Hz'``) to complex arrays in V/m and A/m on that
    component's own Yee points of the plane (see :class:`WaveguideModes` for the axes and the
    coordinates). The transverse components are real and the longitudinal ones imaginary:
    the phasors of a run ``exp(-i omega t)`` taken at each component's own times, with ``E``
    on the node plane and the ``H`` components that lie half a cell along the axis referred
    to their own half plane. ``polarization`` gives the share of ``sum |E_t|^2`` along each
    transverse axis; ``power_watts`` is 1 (per metre of each invariant axis).
    """
    index: int
    effective_index: float
    beta_per_meter: float
    E: object
    H: object
    polarization: object
    power_watts: float
    metadata: object = field(default_factory=dict)

    def __post_init__(self):
        register_dataclass(type(self))
        for name in ('E', 'H', 'polarization', 'metadata'):
            object.__setattr__(self, name, frozen_mapping(getattr(self, name)))


@spec
class WaveguideModes(Declaration):
    """The modes :func:`solve_modes` found, by decreasing effective index.

    ``modes`` is the tuple of :class:`WaveguideMode`; ``modes[0]`` (or ``result[0]``) is the
    largest index. ``axis`` is the plane's normal and ``position`` the node plane it snapped
    to, in ``Problem.units``. ``axes`` names the two array axes of every field, in the order
    the grid stores them (``('y', 'x')`` for a z plane); ``coordinates[component]`` gives
    their coordinates. ``wavelength`` is the vacuum wavelength in ``Problem.units``.
    """
    modes: tuple
    axis: str
    position: float
    wavelength: float
    axes: tuple
    coordinates: object
    metadata: object = field(default_factory=dict)

    def __post_init__(self):
        register_dataclass(type(self))
        object.__setattr__(self, 'modes', tuple(self.modes))
        for name in ('coordinates', 'metadata'):
            object.__setattr__(self, name, frozen_mapping(getattr(self, name)))

    def __getitem__(self, index):
        return self.modes[index]

    def __len__(self):
        return len(self.modes)

    def __iter__(self):
        return iter(self.modes)


# ---------------------------------------------------------------------------------------------
# The discrete operator
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Side:
    """A window edge's neighbour rule: ``'wall'`` (PEC, or a truncation), ``'pmc'``,
    ``'periodic'`` or ``'invariant'``; ``truncates`` marks a wall that is not physical."""

    kind: str
    truncates: bool = False


def _difference(n, kind_low, kind_high, dx):
    """The forward difference along one axis of ``n`` cells, divided by ``dx`` (sparse)."""
    import scipy.sparse as sp
    if kind_low == 'invariant':
        return sp.csr_matrix((1, 1))
    D = sp.diags([-np.ones(n), np.ones(n - 1)], [0, 1], shape=(n, n), format='lil')
    if kind_high == 'periodic':
        D[n - 1, 0] += 1.0
    return D.tocsr() / dx


def _masks(Nu, Nv, sides):
    """0/1 masks of the six local components ``(Eu, Ev, Ew, Hu, Hv, Hw)``, each ``(Nu, Nv)``.

    The kernel's face masks (``src/wavebit/fdtd/_kernel/faces.py``): a wall on a low face zeroes the tangential
    E on its node layer and the normal H there; a PMC on a high face zeroes the tangential H on
    its last half layer and the normal E there. Every other side has none.
    """
    m = {name: np.ones((Nu, Nv)) for name in ('Eu', 'Ev', 'Ew', 'Hu', 'Hv', 'Hw')}
    for a, (low, high) in enumerate(sides):
        index = (slice(0, 1), slice(None)) if a == 0 else (slice(None), slice(0, 1))
        last = (slice(Nu - 1, Nu), slice(None)) if a == 0 else (slice(None), slice(Nv - 1, Nv))
        axis_name = 'uv'[a]
        tangential = [c for c in 'uvw' if c != axis_name]
        if low.kind == 'wall':
            for c in tangential:
                m['E' + c][index] = 0.0
            m['H' + axis_name][index] = 0.0
        if high.kind == 'pmc':
            for c in tangential:
                m['H' + c][last] = 0.0
            m['E' + axis_name][last] = 0.0
    return m


def discrete_modes(eps, mu, sides, dx, omega_d, count):
    """The ``count`` largest-``q^2`` modes of the window, on the local ``(u, v)`` arrays.

    ``eps`` and ``mu`` are dicts ``{'u': (Nu, Nv), 'v': ..., 'w': ...}`` of each local
    component's diagonal coefficient at its own Yee point; ``sides`` is
    ``((u_low, u_high), (v_low, v_high))`` of :class:`Side`; ``omega_d`` is ``Omega``. Returns
    ``(q2, fields, residuals)`` with ``fields[i]`` a dict of the six local components (complex
    arrays, unnormalized but real-phased) and ``q2`` descending. Lengths are in the units of
    ``dx``; ``H`` is ``Z0 H``.
    """
    import scipy.sparse as sp
    Nu, Nv = eps['u'].shape
    (ul, uh), (vl, vh) = sides
    du = _difference(Nu, ul.kind, uh.kind, dx)
    dv = _difference(Nv, vl.kind, vh.kind, dx)
    Iu, Iv = sp.identity(Nu, format='csr'), sp.identity(Nv, format='csr')
    Du = sp.kron(du, Iv, format='csr')
    Dv = sp.kron(Iu, dv, format='csr')
    m = _masks(Nu, Nv, sides)
    flat = {k: v.reshape(-1) for k, v in m.items()}
    ev = lambda d: sp.diags(d.reshape(-1))        # noqa: E731
    C = sp.hstack([-Dv, Du], format='csr')
    G = sp.vstack([Du, Dv], format='csr')
    P = (omega_d * sp.block_diag([ev(eps['u']), ev(eps['v'])])
         - (1.0 / omega_d) * (C.T @ ev(flat['Hw'] / mu['w'].reshape(-1)) @ C))
    Q = (omega_d * sp.block_diag([ev(mu['v']), ev(mu['u'])])
         - (1.0 / omega_d) * (G @ ev(flat['Ew'] / eps['w'].reshape(-1)) @ G.T))
    keep_e = np.concatenate([flat['Eu'], flat['Ev']]) > 0
    keep_h = np.concatenate([flat['Hv'], flat['Hu']]) > 0
    Re = sp.identity(2 * Nu * Nv, format='csr')[np.flatnonzero(keep_e)]
    Rh = sp.identity(2 * Nu * Nv, format='csr')[np.flatnonzero(keep_h)]
    Pr = (Rh @ P @ Re.T).tocsr()
    Qr = (Re @ Q @ Rh.T).tocsr()
    A = (Qr @ Pr).tocsc()
    n = A.shape[0]
    top = omega_d ** 2 * max(float(np.max(eps[c] * mu[d]))
                             for c, d in (('u', 'v'), ('v', 'u'), ('w', 'w')))
    wanted = min(count + 6, n)
    if n <= DENSE_LIMIT:
        values, vectors = np.linalg.eig(A.toarray())
    else:
        from scipy.sparse.linalg import eigs
        sigma = top * (1.0 + 1e-3)
        # A fixed start vector makes the result reproducible to the bit; a generic one (not,
        # say, all ones, which a symmetric guide's odd modes are orthogonal to) reaches every
        # mode.
        start = np.random.default_rng(0).standard_normal(n)
        values, vectors = eigs(A, k=min(wanted, n - 2), sigma=sigma, which='LM', v0=start)
    order = np.argsort(-values.real)
    values, vectors = values[order], vectors[:, order]
    values, vectors = values[:wanted], vectors[:, :wanted]
    # Real-phase each degenerate cluster and diagonalize the cross product inside it.
    q2_out, e_out = [], []
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and abs(values[j] - values[i]) <= DEGENERATE * abs(values[i]):
            j += 1
        V = vectors[:, i:j]
        k = j - i
        basis = np.concatenate([V.real, V.imag], axis=1)
        U, s, _ = np.linalg.svd(basis, full_matrices=False)
        if s.size and s[min(k, s.size) - 1] <= 1e-8 * s[0]:
            raise ValidationError('a degenerate cluster of modes did not span a real subspace',
                                  code='mode.convergence', path='solve_modes')
        Eb = U[:, :k]
        # Propagating candidates (Re q^2 > 0) of a lossless window are real; a complex pair
        # can appear only among evanescent ones (Re q^2 < 0), which no mode is taken from.
        if np.any((values[i:j].real > 0)
                  & (np.abs(values[i:j].imag) > DEGENERATE * np.abs(values[i:j].real))):
            raise ValidationError(
                f'a propagating eigenvalue q^2 = {complex(values[i]):.6g} is complex; a lossless '
                f'cross-section has a real spectrum there, so the eigensolve did not resolve '
                f'this window', code='mode.convergence', path='solve_modes')
        q2 = float(np.mean(values[i:j].real))
        if q2 > 0:
            H = (Pr @ Eb) / math.sqrt(q2)
            B = Eb.T @ H
            B = 0.5 * (B + B.T)
            _, W = np.linalg.eigh(B)
            Eb = Eb @ W
        for c in range(k):
            q2_out.append(q2)
            e_out.append(Eb[:, c])
        i = j
    fields, residuals = [], []
    for q2, e in zip(q2_out, e_out):
        r = np.linalg.norm(A @ e - q2 * e) / max(abs(q2) * np.linalg.norm(e), 1e-300)
        residuals.append(float(r))
        if not q2 > 0:
            fields.append(None)
            continue
        q = math.sqrt(q2)
        h = (Pr @ e) / q
        E = np.zeros(2 * Nu * Nv)
        E[keep_e] = e
        Hf = np.zeros(2 * Nu * Nv)
        Hf[keep_h] = h
        Eu, Ev = E[:Nu * Nv], E[Nu * Nv:]
        Hv, Hu = Hf[:Nu * Nv], -Hf[Nu * Nv:]
        Ew = flat['Ew'] * (G.T @ Hf) / (1j * omega_d * eps['w'].reshape(-1))
        Hw = flat['Hw'] * (C @ E) / (1j * omega_d * mu['w'].reshape(-1))
        shape = (Nu, Nv)
        fields.append(dict(Eu=Eu.reshape(shape) + 0j, Ev=Ev.reshape(shape) + 0j,
                           Ew=Ew.reshape(shape), Hu=Hu.reshape(shape) + 0j,
                           Hv=Hv.reshape(shape) + 0j, Hw=Hw.reshape(shape)))
    return q2_out, fields, residuals


def cross_product(f, g, weights):
    """``sum (E_u^f H_v^g - E_v^f H_u^g) dA`` over the window: the unconjugated pairing."""
    return float(np.sum((f['Eu'] * g['Hv'] - f['Ev'] * g['Hu']) * weights).real)


# ---------------------------------------------------------------------------------------------
# The public door
# ---------------------------------------------------------------------------------------------

def _refuse(message, code, path='solve_modes', cls=UnsupportedFeatureError):
    raise cls(message, code=code, path=path)


def _traced(value):
    """Whether ``value`` is a JAX tracer or a Torch tensor carrying a gradient."""
    if getattr(value, 'requires_grad', False):
        return True
    module = type(value).__module__ or ''
    return module.startswith('jax') and 'Tracer' in type(value).__name__


def _leaves(value):
    if isinstance(value, (tuple, list)):
        for v in value:
            yield from _leaves(v)
    elif value is not None:
        yield value


def _check_static_inputs(problem, numerics):
    """Refuse a traced or gradient-carrying input: the eigensolve does not differentiate."""
    if numerics.geometry_ranges:
        _refuse('solve_modes is a host eigensolve and does not differentiate; a structure with '
                'Numerics(geometry_ranges=...) traces its geometry. Solve the modes from concrete '
                'geometry.', 'mode.traced', 'numerics.geometry_ranges')
    from wavebit.materials import Material, ConductiveMaterial, DispersiveMaterial
    materials = [('scene.background', problem.scene.background)] + [
        ('scene.structures.' + s.name + '.material', s.material) for s in problem.scene.structures]
    for path, material in materials:
        names = ()
        if isinstance(material, Material):
            names = ('epsilon', 'mu')
        elif isinstance(material, ConductiveMaterial):
            names = ('epsilon', 'mu', 'conductivity')
        elif isinstance(material, DispersiveMaterial):
            names = ('epsilon_inf', 'mu')
        for name in names:
            for leaf in _leaves(getattr(material, name, None)):
                if _traced(leaf):
                    _refuse(f'{path}.{name} is traced or carries a gradient. solve_modes is a '
                            f'host eigensolve and does not differentiate; pass concrete '
                            f'material values.', 'mode.traced', path + '.' + name)


def _side(face, truncated_span, path):
    """The neighbour rule a window edge reads, from the public face declaration."""
    from . import _boundaries as boundaries
    if truncated_span:
        return Side('wall', truncates=True)
    if isinstance(face, boundaries.Open):
        return Side('wall', truncates=True)
    if isinstance(face, boundaries.PEC):
        return Side('wall')
    if isinstance(face, boundaries.PMC):
        return Side('pmc')
    if isinstance(face, boundaries.Periodic):
        return Side('periodic')
    if isinstance(face, boundaries.InvariantAxis):
        k = face.k
        if _traced(k):
            _refuse('an InvariantAxis k is traced; solve_modes needs a concrete k = 0',
                    'mode.traced', path)
        if float(k) != 0.0:
            _refuse(f'an InvariantAxis with k = {float(k):g} makes the cross-section a Bloch '
                    f'problem with a complex operator; solve_modes admits k = 0 only.',
                    'mode.boundary', path)
        return Side('invariant')
    name = type(face).__name__
    _refuse(f'a {name} transverse face is not supported by solve_modes: its neighbour rule '
            f'breaks the symmetric discrete operator the solver and the mode orthogonality rely '
            f'on. PEC, PMC, Periodic, Open and InvariantAxis (k = 0) are.', 'mode.boundary', path)


def _component_slice(arr, c, w, k, ranges):
    """``arr[c]`` on node or half layer ``k`` along ``w`` over ``ranges = {axis: (lo, hi)}``,
    as a ``(N_u, N_v)`` array with ``u = w+1``, ``v = w+2``."""
    u, v = (w + 1) % 3, (w + 2) % 3
    full = np.asarray(arr)
    idx = [c, slice(None), slice(None), slice(None)]
    for a in range(3):
        dim = 3 - a
        n = full.shape[dim]
        if a == w:
            idx[dim] = slice(0, 1) if n == 1 else slice(k, k + 1)
        else:
            lo, hi = ranges[a]
            idx[dim] = slice(None) if n == 1 else slice(lo, hi)
    block = full[tuple(idx)]                         # (z, y, x), length 1 along w
    # Move to (u, v, w) and drop w.
    order = [2 - u, 2 - v, 2 - w]
    block = np.transpose(block, order)[:, :, 0]
    shape = tuple(ranges[a][1] - ranges[a][0] for a in (u, v))
    return np.broadcast_to(block, shape).astype(np.float64, copy=True)


@eager_under_torch_compile
def solve_modes(problem, *, numerics, plane, wavelength, count=1, backend='numpy'):
    """The ``count`` largest-index modes of the discrete Yee system on ``plane``.

    ``fdtd.solve_modes(problem, numerics=numerics, plane=fdtd.Plane(z=0.), wavelength=1.55,
    count=2)`` prepares ``problem`` as :func:`prepare` does, reads the diagonal permittivity and
    permeability the time stepper will use on the plane, and returns
    :class:`WaveguideModes`: each mode's effective index, its E and H on the Yee points, and a
    normalization to 1 W along the plane's positive normal. ``wavelength`` is the vacuum
    wavelength in ``Problem.units``. The plane spans the physical cross-section unless given
    transverse spans, which truncate it with a wall. Across an ``Open`` face or a span, a mode
    must be confined: its transverse E on the outermost cells at most 1e-3 of its peak.
    ``backend`` chooses the array type of the returned fields; the eigensolve itself runs once
    on the host with SciPy and does not differentiate (traced inputs are refused).
    """
    from ._api import _issue
    from ._spelling import public as _public_spelling
    try:
        with _public_spelling():
            return _solve(problem, numerics, plane, wavelength, count, backend)
    except (TypeError, ValueError, NotImplementedError) as exc:
        public = _issue(exc)
        if public is exc:
            raise
        raise public from exc


def _solve(problem, numerics, plane, wavelength, count, backend):
    from ._problem import Problem
    from ._numerics import Numerics
    from . import _geometry as geometry
    if not isinstance(problem, Problem) or not isinstance(numerics, Numerics):
        raise ValidationError('expected fdtd.Problem and fdtd.Numerics', code='problem.type',
                              path='problem')
    if not isinstance(plane, geometry.Plane):
        raise ValidationError('expected fdtd.Plane', code='mode.plane', path='plane')
    if backend not in ('numpy', 'jax', 'torch'):
        raise ValidationError("backend is 'numpy', 'jax' or 'torch'", code='mode.backend',
                              path='backend', actual=backend)
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValidationError('count is a positive integer', code='mode.count', path='count',
                              actual=count)
    count = int(count)
    if _traced(wavelength):
        _refuse('the wavelength is traced or carries a gradient; solve_modes is a host '
                'eigensolve and does not differentiate', 'mode.traced', 'wavelength')
    try:
        lam = static_float('solve_modes', 'wavelength', wavelength)
    except (TypeError, ValueError) as exc:
        raise ValidationError(str(exc), code='mode.wavelength', path='wavelength') from exc
    if not (math.isfinite(lam) and lam > 0):
        raise ValidationError('the wavelength is a positive length', code='mode.wavelength',
                              path='wavelength', actual=wavelength)
    _check_static_inputs(problem, numerics)

    from ._lower_public import lower_problem
    from ._execution import _planned
    from ._prepare.coefficients import build
    from ._runtime import runtime
    scene, metadata = lower_problem(problem, numerics)
    plan, _ = _planned(scene)
    grid = plan.grid
    if plan.constitutive:
        _refuse('the scene has coupled smoothed (constitutive) material, whose off-diagonal '
                'cross coupling solve_modes does not model; it admits cell-snapped bodies and '
                'smoothed bodies that lower to a diagonal medium only.', 'mode.material')
    for index, structure in enumerate(scene.structures):
        if getattr(structure, 'discretisation', 'voxel') == 'stencil':
            _refuse("the scene uses discretization='stencil', whose extended operator "
                    'solve_modes does not model.', 'mode.material', 'numerics.discretization')

    w = plane.axis
    u, v = (w + 1) % 3, (w + 2) % 3
    bounds = problem.scene.boundaries
    if type(bounds.face(_AXES[w] + '-')).__name__ == 'InvariantAxis':
        raise ValidationError(f'the plane is across the InvariantAxis {_AXES[w]!r}; a mode '
                              f'propagates along a resolved axis.', code='mode.plane',
                              path='plane')
    pad = metadata['padding_cells']
    low = metadata['physical_domain']['low']
    size = metadata['physical_domain']['size']
    phys = {}
    for a in range(3):
        n0 = int(pad[_AXES[a] + '-'])
        phys[a] = (n0, n0 + int(round(size[a] / grid.dx)))
    position = float(plane.position)
    if not (low[w] - 1e-9 * grid.dx <= position <= low[w] + size[w] + 1e-9 * grid.dx):
        raise ValidationError('the plane is outside the physical domain',
                              code='mode.outside_domain', path='plane',
                              expected=(low[w], low[w] + size[w]), actual=position)
    k = int(round((position - grid.low[w]) / grid.dx))
    ranges, sides = {}, []
    for a in (u, v):
        span = plane.extent(a)
        lo, hi = phys[a]
        truncated = (False, False)
        if span is not None:
            i0 = int(round((static_float('Plane', _AXES[a], span[0]) - grid.low[a]) / grid.dx))
            i1 = int(round((static_float('Plane', _AXES[a], span[1]) - grid.low[a]) / grid.dx))
            if i1 <= i0 or i0 < lo or i1 > hi:
                raise ValidationError(f'the plane span along {_AXES[a]} snaps to nodes '
                                      f'{i0}..{i1}, outside the physical cross-section '
                                      f'{lo}..{hi} or empty', code='mode.plane',
                                      path='plane.' + _AXES[a], actual=span)
            truncated = (i0 > lo, i1 < hi)
            lo, hi = i0, i1
        ranges[a] = (lo, hi)
        pair = [_side(bounds.face(_AXES[a] + sign), cut, 'scene.boundaries.' + _AXES[a] + sign)
                for sign, cut in zip('-+', truncated)]
        if any(truncated) and any(s.kind == 'periodic' for s in pair):
            # A span across part of a periodic axis is no longer periodic: both ends truncate.
            pair = [Side('wall', truncates=True), Side('wall', truncates=True)]
        sides.append(tuple(pair))
    ranges[w] = (k, k + 1)

    rt = runtime('numpy')
    coefficients = build(scene, plan, rt)
    shape4 = (3, plan.nz, plan.ny, plan.nx)
    arrays = {}
    for name in ('ae', 'be', 'ah', 'bh'):
        value = coefficients[name]
        if not isinstance(value, (np.ndarray, float, int, np.floating)):
            _refuse('a material coefficient is traced; solve_modes needs concrete values',
                    'mode.traced', 'scene')
        arrays[name] = np.broadcast_to(np.asarray(value, dtype=np.float64), shape4)
    ratio = plan.dt / plan.dx
    eps, mu = {}, {}
    window = {a: ranges[a] for a in (u, v)}
    layers = [kk for kk in (k - 1, k, k + 1) if 0 <= kk < grid.n[w]]
    for local, c in (('u', u), ('v', v), ('w', w)):
        for name, store in (('be', eps), ('bh', mu)):
            slices = [_component_slice(arrays[name], c, w, kk, {**window, w: (kk, kk + 1)})
                      for kk in layers]
            for other in slices[1:]:
                # Relative 1e-12: a smoothed body's occupancy differs by round-off from layer to
                # layer (measured 8e-17 on a Box along its own axis); a real change does not.
                if not np.allclose(other, slices[0], rtol=UNIFORM, atol=0.0):
                    _refuse(f'the medium changes along {_AXES[w]} at the plane '
                            f'({_AXES[w]} = {position:g}): a mode is a solution of a '
                            f'cross-section that is uniform along its axis. Move the plane '
                            f'into a uniform section.', 'mode.nonuniform', 'plane')
            store[local] = ratio / slices[layers.index(k)]
        for name in ('ae', 'ah'):
            for kk in layers:
                s = _component_slice(arrays[name], c, w, kk, {**window, w: (kk, kk + 1)})
                if not np.all(s == 1.0):
                    _refuse('the window holds a conductive or absorbing medium (a lossy '
                            'update coefficient); solve_modes admits lossless media only, '
                            'and a lossy or leaky mode is not a mode of this solver.',
                            'mode.material', 'scene')
    _refuse_dispersive(scene, grid, window, w, layers)

    omega = 2.0 * math.pi / lam
    omega_d = 2.0 * math.sin(omega * plan.dt / 2.0) / plan.dt
    q2s, fields, residuals = discrete_modes(eps, mu, tuple(sides), grid.dx, omega_d, count)

    invariant = [a for a, s in zip((u, v), sides) if s[0].kind == 'invariant']
    L = problem.units.length_unit_meters
    dA = 1.0
    for a, s in zip((u, v), sides):
        if s[0].kind != 'invariant':
            dA *= grid.dx
    weights = np.full(eps['u'].shape, dA)
    modes = []
    for index, (q2, f, residual) in enumerate(zip(q2s, fields, residuals)):
        if len(modes) == count:
            break
        if f is None or not (0 < q2 < (2.0 / grid.dx) ** 2):
            break
        if residual > RESIDUAL:
            _refuse(f'mode {index}: the eigen-residual {residual:.2g} exceeds {RESIDUAL:g}',
                    'mode.convergence')
        edge = _edge_ratio(f, sides)
        if edge > CONFINEMENT:
            break
        q = math.sqrt(q2)
        beta = (2.0 / grid.dx) * math.asin(q * grid.dx / 2.0)
        cos = math.cos(beta * grid.dx / 2.0)
        N = cross_product(f, f, weights)
        power = 0.5 * N * cos * L ** (2 - len(invariant)) / Z0
        if not power > 0:
            _refuse(f'mode {index} carries its power against its phase velocity; solve_modes '
                    f'admits forward modes only.', 'mode.convergence')
        scale = 1.0 / math.sqrt(power)
        # Phase: the largest transverse E entry positive.
        Et = np.concatenate([f['Eu'].real.ravel(), f['Ev'].real.ravel()])
        if Et[np.argmax(np.abs(Et))] < 0:
            scale = -scale
        f = {key: value * scale for key, value in f.items()}
        modes.append((q, beta, f, residual, edge))
    if len(modes) < count:
        _refuse(f'solve_modes found {len(modes)} confined mode(s) at wavelength {lam:g}, fewer '
                f'than count = {count}. A mode is admitted while its index stays above the '
                f'grid limit and, across an Open face or a plane span, its transverse E on the '
                f'outermost cells stays within {CONFINEMENT:g} of its peak; a mode beyond that '
                f'is leaky, near cutoff or a mode of the truncated box. Ask for fewer modes or '
                f'widen the cross-section.', 'mode.count', 'count')

    return _package(problem, plan, grid, w, u, v, k, ranges, lam, omega, omega_d, modes,
                    backend, sides)


def _edge_ratio(f, sides):
    """Largest |E_t| on the outermost cells of each truncating edge over the peak |E_t|."""
    Et = np.maximum(np.abs(f['Eu']), np.abs(f['Ev']))
    peak = float(np.max(Et))
    if peak == 0:
        return math.inf
    worst = 0.0
    for a, (low, high) in enumerate(sides):
        if low.truncates:
            worst = max(worst, float(np.max(Et[0] if a == 0 else Et[:, 0])))
        if high.truncates:
            worst = max(worst, float(np.max(Et[-1] if a == 0 else Et[:, -1])))
    return worst / peak


def _refuse_dispersive(scene, grid, window, w, layers):
    """A dispersive body (or background) reaching the window is refused.

    A pole's memory is a separate term (``src/wavebit/fdtd/_kernel/poles.py``) and leaves the update
    coefficients looking lossless, so it is found by material here; a conductivity is not a
    term and shows in ``ae``, which :func:`_solve` reads instead.
    """
    from wavebit.fdtd._legacy import Dispersive
    from wavebit.materials._time_domain import as_material
    background = as_material(getattr(scene, 'background', None), 'Scene(background=)')
    if isinstance(background, Dispersive):
        _refuse('the background is dispersive; solve_modes admits non-dispersive media only.',
                'mode.material', 'scene.background')
    box = dict(window)
    box[w] = (min(layers), max(layers) + 1)
    for index, structure in enumerate(scene.structures):
        material = as_material(structure.material, 'Structure(material=)')
        if not isinstance(material, Dispersive):
            continue
        support = structure.shape.support(grid)
        if all(support[a][0] < box[a][1] and support[a][1] > box[a][0] for a in range(3)):
            _refuse(f'structures[{index}] is dispersive and reaches the plane; solve_modes '
                    f'admits non-dispersive media only, and a dispersive mode is not a mode of '
                    f'this solver.',
                    'mode.material', 'scene.structures')


def _to_backend(array, backend):
    if backend == 'numpy':
        return array
    if backend == 'jax':
        import jax.numpy as jnp
        return jnp.asarray(array)
    import torch
    return torch.from_numpy(np.ascontiguousarray(array))


def _package(problem, plan, grid, w, u, v, k, ranges, lam, omega, omega_d, modes, backend, sides):
    from ._kernel.grid import OFFSETS
    L = problem.units.length_unit_meters
    names = {0: 'x', 1: 'y', 2: 'z'}
    local = {'u': u, 'v': v, 'w': w}
    # Array axes in storage order (z, y, x) with w dropped.
    stored = tuple(a for a in (2, 1, 0) if a != w)
    axes = tuple(names[a] for a in stored)
    coordinates = {}
    for field_name in ('E', 'H'):
        for c in range(3):
            off = OFFSETS[field_name][c]
            coordinates[field_name + names[c]] = {
                names[a]: grid.low[a] + (np.arange(*ranges[a]) + 0.5 * off[a]) * grid.dx
                for a in (u, v)}
    out = []
    for index, (q, beta, f, residual, edge) in enumerate(modes):
        E, H = {}, {}
        for name, c in local.items():
            for field_name, store, scale in (('E', E, 1.0), ('H', H, 1.0 / Z0)):
                arr = f[field_name + name]                     # (Nu, Nv)
                # (u, v) -> storage order of the two transverse axes.
                if stored == (u, v):
                    data = arr
                else:
                    data = arr.T
                store[field_name + names[c]] = _to_backend(np.ascontiguousarray(data * scale),
                                                           backend)
        Eu, Ev = np.abs(f['Eu']) ** 2, np.abs(f['Ev']) ** 2
        total = float(np.sum(Eu) + np.sum(Ev))
        polarization = {names[u]: float(np.sum(Eu)) / total, names[v]: float(np.sum(Ev)) / total}
        out.append(WaveguideMode(
            index=index, effective_index=beta / omega, beta_per_meter=beta / L,
            E=E, H=H, polarization=polarization, power_watts=1.0,
            metadata={'q': q, 'residual': residual, 'edge_ratio': edge}))
    invariant = tuple(names[a] for a, s in zip((u, v), sides) if s[0].kind == 'invariant')
    return WaveguideModes(
        modes=tuple(out), axis=names[w], position=grid.low[w] + k * grid.dx, wavelength=lam,
        axes=axes, coordinates=coordinates,
        metadata={'spacing': grid.dx, 'time_step': plan.dt, 'omega_discrete': omega_d,
                  'node_index': k, 'window': {names[a]: ranges[a] for a in (u, v)},
                  'invariant_axes': invariant,
                  'power_units': 'W' if not invariant else ('W/m' if len(invariant) == 1 else 'W/m^2'),
                  'sides': {names[a]: (s[0].kind, s[1].kind) for a, s in zip((u, v), sides)}})
