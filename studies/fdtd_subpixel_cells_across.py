"""How few cells across may a smoothed body be before its answer is a different particle?

Stage 4's study S2b, run before the cells-across door (C5). The plan first meant to set that
number from the fill rule's volume error, and that question has closed: every public body's fill
is its exact clip, and the lowered volume matches the analytic one to round-off at any
resolution. The electromagnetic question has not. A body a few cells across is a handful of
mixed voxels, and what the grid then scatters from is set by the rule's effective polarizability
of those voxels, not by the particle's -- plain volume averaging gives a sub-voxel sphere a
polarizability ``V (eps - 1)`` against the exact ``3 V (eps - 1)/(eps + 2)``, 4.67 times too
large at ``eps = 12``. Below some number of cells across, refining does not converge an answer;
the answer is a different particle, and it depends on where the grid happens to fall.

**Instrument 1, static.** The body's zero-frequency polarizability on the Yee node lattice, with
the production inverse-material operator: ``phi`` on nodes, ``E = -grad_h phi`` on edges,
``D = K^-1 E`` with ``K`` the lowered diagonal ``be`` plus the constitutive cross action,
``div_h D = 0`` at interior nodes, ``phi = -E0 x`` on the outer node planes of a PEC box, and
``p = sum (D - E) dx^3``. The box's image dipoles are removed with a fitted cubic coefficient
(``KAPPA``; it moved 0.444 -> 0.431 between 4 and 8 cells across, fitted over boxes 3-8
diameters wide, residual <= 8e-5). The sphere is judged against Clausius-Mossotti; the other
bodies, which have no closed form, against a Richardson extrapolation from 12 and 24 cells
across, checked at 16. No
field is stepped: only ``fd.plan`` and ``fd.lower`` run, with the constitutive energy
certificate held open (at ``eps = 12`` it refuses some placements and not others at every
resolution below 16 cells across; the operator it judges is positive -- this study measures
the geometry's answer, not the certificate).

**Instrument 2, dynamic.** A few public ``fd.run`` solves of a small ``eps = 12`` sphere against
exact Mie theory, to say whether the static number is the scattering number.

**Instrument 3, the certificate.** Where the energy certificate refuses, by contrast and cells
across, counted over the placements with ``fd.plan`` alone. That certificate was the Gershgorin
lower bound; the star lift (``wavebit/fd/_lower/stars.py``) replaced it, and ``validate``
now refuses only an uncertified lift, so a rerun of this instrument records every placement as
certified. The logged rows are the Gershgorin bound's, at the commit that wrote them.

Instruments 1 and 2 measured the default rule **before the lift**: ``lowered`` reads
``build``'s diagonal, which the lift patches only later, in ``params.lower``. The door's numbers
(:data:`wavebit.fd._lower.checks.CELLS_ACROSS_REFUSED`) are that operator's.

Run from the repository root: ``python studies/fdtd_subpixel_cells_across.py [static|bodies|
dynamic|certificate]``; each row goes to ``studies/results/fdtd_subpixel_cells_across.log`` as
JSON lines (appended; the default runs the first three).
Single-threaded BLAS; the static sweep takes minutes, the dynamic runs about half a minute each.
"""
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import numpy as np                                                              # noqa: E402
import scipy.fft as sfft                                                         # noqa: E402
import scipy.sparse as sp                                                        # noqa: E402
import scipy.sparse.linalg as spla                                               # noqa: E402
import wavebit.fd as fd                                                          # noqa: E402
import wavebit.fd._lower.constitutive as constitutive                            # noqa: E402
import wavebit.fd._lower.plan as plan_module                                     # noqa: E402
from wavebit.fd._lower.coefficients import build                                # noqa: E402
from wavebit.fd._lower.occupancy import occupancy                               # noqa: E402
from wavebit.fd._runtime import runtime                                         # noqa: E402

OUT = ROOT / 'studies' / 'results' / 'fdtd_subpixel_cells_across.log'
KAPPA = 0.43
PLACEMENTS = {'node': (0, 0, 0), 'cell': (0.5, 0.5, 0.5), 'mieA': (0.5, 1 / 3, 0.2),
              'mieB': (0.25, 2 / 3, 0.4), 'c': (0.137, 0.719, 0.371)}
DX = 0.05

_held_open = {}


def _validate(scene, p):
    """The energy certificate, recorded instead of obeyed (see the module docstring)."""
    try:
        _real_validate(scene, p)
        _held_open['certified'] = True
    except ValueError as error:
        _held_open['certified'] = False
        _held_open['refusal'] = str(error)[:120]


_real_validate = constitutive.validate
constitutive.validate = _validate
plan_module.constitutive = constitutive
# The cells-across door (C5) is what this study sets; it refuses the small bodies measured
# here, so it is held open too. Its verdict is the ``across`` of each row.
import wavebit.fd._lower.checks as _checks                                       # noqa: E402
_checks.check_cells_across = lambda scene, grid: None


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def closed_scene(structures, n, dx):
    return fd.Scene(domain=fd.Domain(size=(n * dx,) * 3),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(steps=4), structures=structures)


def _area(v):
    return 0.5 * sum(u0 * v1 - u1 * v0 for (u0, v0), (u1, v1) in zip(v, v[1:] + v[:1]))


def body(kind, across, dx, center):
    """A body whose smallest extent is ``across`` cells, and its analytic volume."""
    s = across * dx
    if kind == 'sphere':
        return fd.Sphere(radius=s / 2, center=center), math.pi * s ** 3 / 6
    if kind == 'box30':
        return (fd.Box(size=(1.6 * s, s, 1.3 * s), center=center, angle=30, axis='z'),
                1.6 * 1.3 * s ** 3)
    if kind == 'cylinder':
        return (fd.Cylinder(radius=s / 2, height=1.5 * s, axis='z', center=center),
                math.pi * s * s / 4 * 1.5 * s)
    if kind == 'hexprism':
        r = s / math.sqrt(3)                       # flat to flat: s
        v = tuple((r * math.cos(math.pi / 3 * i), r * math.sin(math.pi / 3 * i)) for i in range(6))
        return (fd.Prism(vertices=v, height=1.5 * s, axis='x', center=center, angle=10),
                _area(v) * 1.5 * s)
    raise KeyError(kind)


def lowered(shape, eps, n, dx):
    scene = closed_scene((fd.Structure(shape, fd.Material(eps=eps)),), n, dx)
    _held_open.clear()
    plan = quiet(fd.plan, scene)
    rt = runtime('numpy')
    coefficients = build(scene, plan, rt)
    cross = constitutive.lower(scene, plan, rt)
    ratio = plan.dt / plan.dx
    permittivity = ratio / np.broadcast_to(np.asarray(coefficients['be']), (3, n, n, n))
    terms = [(t, np.asarray(cross[t.name])) for t in plan.constitutive if t.name in cross]
    return permittivity, terms, dict(_held_open), scene, plan


def _edges(eps_c, n, c):
    """A component array ``[z, y, x]`` as edges, the n-th node plane padded with vacuum."""
    shape = [n + 1, n + 1, n + 1]
    shape[2 - c] = n
    out = np.ones(shape)
    out[:n, :n, :n] = eps_c
    return out


def _gradients(n):
    d = sp.diags([-np.ones(n), np.ones(n)], [0, 1], shape=(n, n + 1), format='csr')
    identity = sp.identity(n + 1, format='csr')
    return (sp.kron(identity, sp.kron(identity, d), format='csr'),
            sp.kron(identity, sp.kron(d, identity), format='csr'),
            sp.kron(d, sp.kron(identity, identity), format='csr'))


def _cross_matrix(cross, shape):
    """The kernel's ``cross_action`` as a sparse matrix (matches it to 5.6e-17)."""
    rz, ry, rx = shape

    def shift(axis):
        mats = [sp.identity(rz), sp.identity(ry), sp.identity(rx)]
        size = (rz, ry, rx)[axis]
        mats[axis] = sp.diags([np.ones(size - 1)], [-1], shape=(size, size))
        return sp.kron(mats[0], sp.kron(mats[1], mats[2]), format='csr')
    count = rz * ry * rx
    average = [0.5 * (sp.identity(count) + shift(2 - j)) for j in range(3)]
    blocks = [[None] * 3 for _ in range(3)]
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        weights = sp.diags(cross[k].ravel())
        blocks[i][j] = average[i].T @ weights @ average[j]
        blocks[j][i] = average[j].T @ weights @ average[i]
    for i in range(3):
        blocks[i][i] = sp.csr_matrix((count, count))
    return sp.bmat(blocks, format='csr')


def polarizability(n, dx, eps_edges, terms, field_axis=2, tol=1e-11):
    """``p`` for a unit applied field along ``field_axis`` (preconditioned CG on phi)."""
    gradients = _gradients(n)
    idx = np.arange(n + 1)
    Z, Y, X = np.meshgrid(idx, idx, idx, indexing='ij')
    coord = (X, Y, Z)[field_axis].astype(float) - n / 2
    interior = ((X > 0) & (X < n) & (Y > 0) & (Y < n) & (Z > 0) & (Z < n)).ravel()
    phi_b = np.where(interior, 0.0, -coord.ravel() * dx)
    flat = [e.ravel() for e in eps_edges]
    regions = []
    for term, cross in terms:
        (k0, k1), (j0, j1), (i0, i1) = term.box
        C = _cross_matrix(cross, (k1 - k0, j1 - j0, i1 - i0))
        select = []
        for c in range(3):
            kk, jj, ii = np.meshgrid(np.arange(k0, k1), np.arange(j0, j1), np.arange(i0, i1),
                                     indexing='ij')
            select.append(np.ravel_multi_index((kk.ravel(), jj.ravel(), ii.ravel()),
                                               eps_edges[c].shape))
        inverse = np.concatenate([1.0 / flat[c][select[c]] for c in range(3)])
        regions.append((select, (sp.diags(inverse) + C).tocsr(), inverse))

    def displacement(e):
        d = [flat[c] * e[c] for c in range(3)]
        for select, K, inverse in regions:
            rhs = np.concatenate([e[c][select[c]] for c in range(3)])
            sol, info = spla.cg(K, rhs, x0=rhs / inverse, rtol=1e-13, atol=0.0, maxiter=2000,
                                M=sp.diags(1.0 / inverse))
            if info != 0:
                raise RuntimeError(f'inner CG info={info}')
            m = len(select[0])
            for c in range(3):
                d[c][select[c]] = sol[c * m:(c + 1) * m]
        return d

    def apply(phi):
        return sum(G.T @ Dc for G, Dc in zip(gradients, displacement([G @ phi for G in gradients])))
    m = n - 1
    lam = 2 - 2 * np.cos(np.pi * np.arange(1, n) / n)
    laplace = lam[:, None, None] + lam[None, :, None] + lam[None, None, :]
    count = int(interior.sum())

    def operator(v):
        phi = np.zeros((n + 1) ** 3)
        phi[interior] = v
        return apply(phi)[interior]

    def preconditioner(r):
        return sfft.idstn(sfft.dstn(r.reshape(m, m, m), type=1) / laplace, type=1).ravel()
    v, info = spla.cg(spla.LinearOperator((count, count), operator), -apply(phi_b)[interior],
                      rtol=tol, atol=0.0, maxiter=3000,
                      M=spla.LinearOperator((count, count), preconditioner))
    if info != 0:
        raise RuntimeError(f'outer CG info={info}')
    phi = phi_b.copy()
    phi[interior] = v
    e = [-(G @ phi) / dx for G in gradients]
    d = displacement(e)
    return np.array([float(np.sum(d[c] - e[c])) * dx ** 3 for c in range(3)])


def alpha(kind, across, eps, placement, box_diameters, field_axis=2, arm='production'):
    """``alpha / V`` along ``field_axis``, the box image removed, and whether it certified."""
    n = int(math.ceil(box_diameters * across))
    n = max(n, int(math.ceil(across)) + 8)
    n += n % 2                                    # even, so the origin is a node (see the docs)
    shape, volume = body(kind, across, DX, tuple(o * DX for o in placement))
    eps_c, terms, held, scene, plan = lowered(shape, eps, n, DX)
    if arm == 'arithmetic':                        # plain voxel averaging, the design's control
        occ = occupancy(scene, plan, runtime('numpy'), fields=('E',))
        (i0, i1), (j0, j1), (k0, k1) = occ[0]['box']
        eps_c = np.ones((3, n, n, n))
        for c in range(3):
            eps_c[c, k0:k1, j0:j1, i0:i1] = 1 + np.asarray(occ[0]['E'][c][0]) * (eps - 1)
        terms = []
    elif arm == 'no cross terms':
        terms = []
    edges = [_edges(eps_c[c], n, c) for c in range(3)]
    raw = polarizability(n, DX, edges, terms, field_axis)[field_axis] / volume
    fraction = volume / (n * DX) ** 3
    return raw / (1 + KAPPA * raw * fraction), held.get('certified')


def static():
    """The sphere against Clausius-Mossotti: three contrasts, five placements, and two controls."""
    rows = []
    for eps in (2.25, 4.0, 12.0):
        exact = 3 * (eps - 1) / (eps + 2)
        arms = ('production', 'no cross terms', 'arithmetic') if eps == 12.0 else ('production',)
        for arm in arms:
            for across in (1, 2, 3, 4, 6, 8, 12, 24):
                errors, certified = {}, []
                for name, placement in PLACEMENTS.items():
                    a, ok = alpha('sphere', across, eps, placement, 6 if across <= 16 else 4,
                                  arm=arm)
                    errors[name] = a / exact - 1
                    certified.append(ok)
                v = np.array(list(errors.values()))
                rows.append(dict(study='static', body='sphere', eps=eps, arm=arm, across=across,
                                 errors=errors, lo=float(v.min()), hi=float(v.max()),
                                 certified=sum(bool(c) for c in certified), of=len(certified)))
                _emit(rows[-1])
    return rows


def bodies():
    """The turned box, the cylinder and the hexagonal prism at eps 12, against their own limit."""
    for kind in ('box30', 'cylinder', 'hexprism'):
        for axis in (0, 2):
            for name, placement in (('node', (0, 0, 0)), ('mieA', (0.5, 1 / 3, 0.2)),
                                    ('c', (0.137, 0.719, 0.371))):
                values = {}
                for across in (2, 3, 4, 6, 8, 12, 16, 24):
                    values[across] = alpha(kind, across, 12.0, placement, 5, field_axis=axis)[0]
                # Richardson on 12 and 24 at second order: the reference the small ones are
                # read against. The 16-cell value, read against it, is the check on the
                # extrapolation (``reference_spread`` in the log: the 16-cell error, not a
                # spread of the reference).
                reference = (values[24] * 24 ** 2 - values[12] * 12 ** 2) / (24 ** 2 - 12 ** 2)
                spread = abs(values[16] / reference - 1)
                _emit(dict(study='bodies', body=kind, eps=12.0, axis='xyz'[axis],
                           placement=name, reference=reference, reference_spread=spread,
                           errors={str(k): v / reference - 1 for k, v in values.items()}))


def certificate():
    """Where the energy certificate refuses: contrast by cells across, over the placements.

    Only ``fd.plan`` runs, in a PEC box eight cells wider than the body, and a row counts the
    placements the certificate passed. It is the evidence behind the certificate's refusal
    message, which advised refining the grid; this says where refining does not help.
    """
    for eps in (6.0, 12.0, 12.5, 13.0, 20.0):
        for kind in ('sphere', 'box30', 'cylinder', 'hexprism'):
            placements = PLACEMENTS if kind == 'sphere' else {
                k: PLACEMENTS[k] for k in ('node', 'mieA', 'c')}
            for across in (4, 8, 16, 24, 32):
                n = across * 2 + 8
                n += n % 2
                passed = 0
                for name, placement in placements.items():
                    shape, _ = body(kind, across, DX, tuple(o * DX for o in placement))
                    scene = closed_scene((fd.Structure(shape, fd.Material(eps=eps)),), n, DX)
                    _held_open.clear()
                    quiet(fd.plan, scene)
                    passed += bool(_held_open.get('certified'))
                _emit(dict(study='certificate', body=kind, eps=eps, across=across,
                           certified=passed, of=len(placements)))


def dynamic():
    """Public runs of a small eps-12 sphere against Mie: is the static number the scattering one?"""
    from references.mie import sphere_efficiencies
    from fd_helpers import to_numpy
    cpw, eps = 40, 12.0
    dx = 1 / cpw
    for radius, offset, tag in ((0.025, (0, 0, 0), '2 across, node'),
                                (0.05, (0, 0, 0), '4 across, node'),
                                (0.05, (0.5, 0.5, 0.5), '4 across, cell'),
                                (0.1, (0, 0, 0), '8 across, node'),
                                (0.1, (0.5, 0.5, 0.5), '8 across, cell')):
        center = tuple(o * dx for o in offset)
        size = 2 * (0.3 + 0.2 + 12 * dx)
        scene = fd.Scene(
            domain=fd.Domain(size=(size,) * 3),
            boundaries=fd.Boundaries.open(fd.PML(cells=12, reflection=1e-8)),
            resolution=fd.Resolution(dx=dx), run=fd.Run(periods=24.),
            structures=(fd.Structure(fd.Sphere(radius=radius, center=center), eps),),
            sources=(fd.TFSF('t', box=fd.Region(size=(0.5,) * 3), direction='+z',
                             polarization='x', band=fd.Band(wavelength=1., df=.1)),),
            monitors=(fd.Flux('abs', box=fd.Region(size=(.4,) * 3), wavelengths=(1.,)),
                      fd.Flux('sca', box=fd.Region(size=(.6,) * 3), wavelengths=(1.,))))
        start = time.perf_counter()
        row = dict(study='dynamic', body='sphere', eps=eps, cpw=cpw, radius=radius,
                   across=2 * radius / dx, placement=list(offset), tag=tag)
        try:
            res = quiet(fd.run, scene, 'numpy')
        except ValueError as error:
            row.update(refused=str(error)[:200])
            _emit(row)
            continue
        incident = to_numpy(res.sources['t'].intensity((1.,)))[0]
        area = math.pi * radius ** 2
        qsca = float(to_numpy(res['sca'].flux)[0] / incident / area)
        q = sphere_efficiencies(math.sqrt(eps), 2 * math.pi * radius)
        row.update(qsca=qsca, mie=float(q[1]), error=qsca / float(q[1]) - 1,
                   decay={k: float(v) for k, v in res.diagnostics['decay'].items()},
                   seconds=time.perf_counter() - start)
        _emit(row)


def _emit(row):
    line = json.dumps(row)
    print(line, flush=True)
    with open(OUT, 'a', encoding='utf-8') as out:
        out.write(line + '\n')


if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    for part in (sys.argv[1:] or ['static', 'bodies', 'dynamic']):
        {'static': static, 'bodies': bodies, 'dynamic': dynamic,
         'certificate': certificate}[part]()
