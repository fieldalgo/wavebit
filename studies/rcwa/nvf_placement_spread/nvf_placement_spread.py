"""Does the normal field of a multi-shape layer depend on where the shapes sit in the cell?

A rigid translation of every shape by ``d`` multiplies each Fourier coefficient by
``exp(-i G.d)`` and changes nothing else, so the un-phased projector coefficients
``exp(+i G.d) c[G]`` must be independent of ``d``. A single shape satisfies that to rounding on
every route. Until 2026-09-11 the closed form (``parametric.normal_projector_coeffs``) sampled a
multi-shape layer where it sat, and the coefficients spread with the placement at roughly
``1/N^1.4`` in the normal grid ``N``, the solve-level ``sum(T)`` spread following; since then it
builds the layer with its first shape at the cell centre and carries it by a phase, and the same
rows are at rounding. This study measures that spread, its exponent, and which routes are exact. The
2026-09-07 roadmap instruments it reconstructs were never committed; their numbers were
reproduced here to every printed digit before the fix, and ``DOCUMENTED`` keeps them so a rerun
says at once whether a route has moved (write-up: ``docs/validation/rcwa/nvf-contact-followup.md``, "Placement
dependence", and ``docs/validation/rcwa/normal-field-known-limits.md``).

Fixture: square cell 700 x 700, background eps 1, boxes eps 12.25, angle 0, truncation (8, 8)
so each projector grid is 33 x 33, NumPy float64.
  TWO_CLOSED   two DISJOINT 250 x 300 boxes at cx = -160 / +160    (production: closed form)
  TWO_PAINTED  two OVERLAPPING 300 x 300 boxes at cx = -100 / +100 (production: polygon route)
  ONE_BOX      one 300 x 300 box at the centre                      (single-shape control)
  ONE_PAINTED  a 200 box covered by a 300 box, one surviving loop   (single-loop control)
Coefficient level: 30 rigid shifts of k/16 px (k = 1..15) along x and along y, px = 700/N;
spread = max |unphased(c(shifted)) - c(base)| over the stacked (n_x^2, n_x n_y, n_y^2) grids.
A 29-shift x-only variant (k/30 px) is kept because the recorded numbers used it.
Solve level: ``sum(T)`` max - min over 8 x-shifts of k/8 px, wavelength 900, thickness 300,
top = bot = 1, theta 0.2, phi 0.3, psi 0.4, projector cache cleared before every solve.

Routes at the coefficient level:
  closed              parametric.normal_projector_coeffs on the shapes  (what production uses
                      for a layer whose shapes do not overlap: freeze_geometry reports no plan)
  polygon_raw         boundary.normal_projector_coeffs on the raw loops, no plan, no anchor: the
                      polygon analogue of the closed form's missing phase (a probe; production
                      never routes a disjoint layer here)
  polygon_planned     plan_layer loops + reduction transform only -- the 2026-09-07 painted
                      instrument as it was written
  polygon_production  the call boundary.conv_mats makes: segment data, periodic images, image
                      groups and the painted-union centroid as anchor
  raster_contour      pixel-centre raster on the torus, factorization.normal_vector_field
                      (method='contour'), integer-pixel shifts only -- a sub-pixel shift moves
                      the staircase itself and measures the raster, not the field

Usage, from the repo root::

    python studies/rcwa/nvf_placement_spread/nvf_placement_spread.py                 # all sections, N = 64 128 256
    python studies/rcwa/nvf_placement_spread/nvf_placement_spread.py --sections coeff --ns 64 128

Rows stream to ``studies/rcwa/nvf_placement_spread/records/nvf_placement_spread.log`` as JSON lines.
"""

from wavebit.compat import legacy as _legacy_wavebit
import argparse
import json
import os
import subprocess
import time
import traceback

import numpy as np
import array_api_compat.numpy as xnp

from wavebit.rcwa._engine import boundary as _b, factorization as F, parametric as P; from wavebit._common.geometry import clipper as _c
from wavebit._common.geometry.interfaces import replay_interfaces
from wavebit.lattice import as_lattice, reduction_transform, sample_grid

PER = 700.0
LAT = as_lattice(period=(PER, PER))
HU = HV = 8
_KY, _KX = _b._k_grids(xnp, HU, HV, xnp.float64)
KY, KX = np.asarray(_KY), np.asarray(_KX)
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'results', 'nvf_placement_spread.log')


def box(**g):
    return dict(dict(type='rectangle', cx=0.0, cy=0.0, angle=0.0, material=12.25), **g)


FIXTURES = dict(
    TWO_CLOSED=[box(lx=250., ly=300., cx=-160.), box(lx=250., ly=300., cx=160.)],
    TWO_PAINTED=[box(lx=300., ly=300., cx=-100.), box(lx=300., ly=300., cx=100.)],
    ONE_BOX=[box(lx=300., ly=300.)],
    ONE_PAINTED=[box(lx=200., ly=200.), box(lx=300., ly=300.)],
)

# The values the 2026-09-07 session recorded (pre-anchor for the closed rows, which read ~3e-17
# since 2026-09-11), so a rerun says at once whether anything moved.
DOCUMENTED = {
    ('closed', 'TWO_CLOSED', 'xy16'): (1.074e-2, 4.056e-3, 1.592e-3),
    ('closed', 'TWO_CLOSED', 'x30'): (1.074e-2, 4.056e-3, 1.592e-3),
    ('polygon_planned', 'TWO_PAINTED', 'xy16'): (8.096e-3, 3.854e-3, 1.926e-3),
    ('solve', 'TWO_CLOSED', 'x8'): (1.242e-3, 6.395e-4, 1.560e-4),
}


def shifted(shapes, dx, dy):
    return [dict(s, cx=s['cx'] + dx, cy=s['cy'] + dy) for s in shapes]


def unphase(c, dx, dy):
    """Coefficients of f(r - d) are exp(-i G.d) c[G]; undo that so shifted grids compare."""
    return c * np.exp(+2j * np.pi * (KX * dx / PER + KY * dy / PER))


def shifts_xy16(px):
    return [s for k in range(1, 16) for s in ((k / 16 * px, 0.0), (0.0, k / 16 * px))]


def shifts_x30(px):
    return [(k / 30 * px, 0.0) for k in range(1, 30)]


def shifts_integer(px):
    return [(j * px, 0.0) for j in (1, 2, 5, 11)] + [(0.0, j * px) for j in (1, 3, 7)]


SHIFTS = dict(xy16=shifts_xy16, x30=shifts_x30, integer=shifts_integer)


# ------------------------------------------------------------------------------ builders
def closed_coeffs(shapes, N):
    P._PROJECTOR_CACHE.clear()
    c = P.normal_projector_coeffs(xnp, shapes, (N, N), LAT, HU, HV, xnp.complex128)
    return np.stack([np.asarray(a) for a in c])


def _loops(shapes):
    return [_b.shape_vertices(xnp, _b._concrete_shape(s), LAT) for s in shapes]


def polygon_raw(shapes, N):
    c = _b.normal_projector_coeffs(xnp, _loops(shapes), (N, N), LAT, HU, HV, xnp.float64,
                                   xnp.complex128, transform=reduction_transform(LAT))
    return np.stack([np.asarray(a) for a in c])


def _planned(shapes):
    """The regions ``boundary.conv_mats`` receives: painting resolved, materials attached."""
    verts = _loops(shapes)
    recs, regions, transform, interfaces = _b.plan_layer(shapes, LAT)
    vals = _c.eval_table(xnp, recs, verts)
    regs = [(s['material'], [_c.loop_verts(xnp, lp, vals) for lp in region])
            for s, region in zip(shapes, regions)]
    return regs, transform, interfaces


def polygon_planned(shapes, N):
    regs, transform, _ = _planned(shapes)
    loops = [V for _, polys in regs for V in polys]
    c = _b.normal_projector_coeffs(xnp, loops, (N, N), LAT, HU, HV, xnp.float64, xnp.complex128,
                                   transform=transform)
    return np.stack([np.asarray(a) for a in c])


def polygon_production(shapes, N, background=1.0):
    """Mirrors the projector call inside ``boundary.conv_mats`` (anchor = painted centroid)."""
    regs, transform, interfaces = _planned(shapes)
    loops = [V for _, polys in regs for V in polys]
    segment_data = replay_interfaces(xnp, regs, background, LAT, transform, interfaces)
    anchor = (_b._paint_centroid(xnp, segment_data, interfaces)
              if segment_data is not None else None)
    c = _b.normal_projector_coeffs(xnp, loops, (N, N), LAT, HU, HV, xnp.float64, xnp.complex128,
                                   transform=transform, segment_data=segment_data,
                                   images=interfaces.images, image_groups=interfaces.groups,
                                   anchor=anchor)
    return np.stack([np.asarray(a) for a in c])


def rasterize(shapes, N):
    """Pixel-centre membership on the torus.

    The distance to a box centre is wrapped into [-P/2, P/2) so that a box shifted across the
    cell edge re-enters on the other side: only then is an integer-pixel shift a permutation
    of the pixels, which is what makes the integer rows a covariance check. A first draft that
    did not wrap read 4.5e-2 at N = 64 -- a raster defect, not a property of the field.
    """
    X, Y = sample_grid(xnp, (N, N), LAT, xnp.float64)
    X, Y = np.asarray(X), np.asarray(Y)
    eps = np.ones((N, N))
    mid = 0.5 * PER

    def wrapped(u, c):
        return (u - c + 0.5 * PER) % PER - 0.5 * PER

    for s in shapes:
        inside = ((np.abs(wrapped(X, s['cx'] + mid)) < 0.5 * s['lx'])
                  & (np.abs(wrapped(Y, s['cy'] + mid)) < 0.5 * s['ly']))
        eps[inside] = float(s['material'])
    return eps


def raster_contour(shapes, N):
    nx, ny = F.normal_vector_field(rasterize(shapes, N), None, lattice=LAT, method='contour')
    nx, ny = np.asarray(nx), np.asarray(ny)
    return np.stack([np.asarray(P._sampled_coeffs(xnp, xnp.asarray(f, dtype=xnp.complex128),
                                                  HU, HV))
                     for f in (nx * nx, nx * ny, ny * ny)])


BUILDERS = dict(closed=closed_coeffs, polygon_raw=polygon_raw, polygon_planned=polygon_planned,
                polygon_production=polygon_production, raster_contour=raster_contour)

# (route, fixture, shift protocol): the rows, in the order they print.
COEFF_ROWS = [
    ('closed', 'TWO_CLOSED', 'xy16'), ('closed', 'TWO_CLOSED', 'x30'),
    ('polygon_raw', 'TWO_CLOSED', 'xy16'),
    ('polygon_production', 'TWO_CLOSED', 'xy16'),
    ('raster_contour', 'TWO_CLOSED', 'integer'),
    ('polygon_planned', 'TWO_PAINTED', 'xy16'),
    ('polygon_production', 'TWO_PAINTED', 'xy16'),
    ('closed', 'ONE_BOX', 'xy16'), ('polygon_raw', 'ONE_BOX', 'xy16'),
    ('polygon_production', 'ONE_BOX', 'xy16'), ('raster_contour', 'ONE_BOX', 'integer'),
    ('polygon_planned', 'ONE_PAINTED', 'xy16'), ('polygon_production', 'ONE_PAINTED', 'xy16'),
]

CFG = dict(backend='numpy', period=[PER, PER], thickness=(300.0,), truncation_order=dict(u=8, v=8),
           top=1.0, bot=1.0, wavelength=900.0, theta=0.2, phi=0.3, psi=0.4)


def sum_T(shapes, N):
    P._PROJECTOR_CACHE.clear()
    solver = _legacy_wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)], shape_nvf_grid_size=N, **CFG)
    return float(np.sum(np.asarray(solver.solve().T)).real)


# --------------------------------------------------------------------------- measurement
class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def emit(self, row):
        with open(self.path, 'a') as sink:
            sink.write(json.dumps(row) + '\n')


def coeff_row(log, route, fixture, protocol, N):
    shapes, build = FIXTURES[fixture], BUILDERS[route]
    shift_set = SHIFTS[protocol](PER / N)
    row = dict(kind='coeff', route=route, fixture=fixture, protocol=protocol, N=N,
               nshifts=len(shift_set))
    t0 = time.time()
    try:
        base = build(shapes, N)
        devs = [float(np.abs(unphase(build(shifted(shapes, dx, dy), N), dx, dy) - base).max())
                for dx, dy in shift_set]
        row.update(spread=max(devs), median=float(np.median(devs)),
                   dc_nx2=float(abs(base[0, 2 * HV, 2 * HU])), ok=True)
        print(f'  {route:<19s} {fixture:<12s} {protocol:<8s} N={N:4d} spread={max(devs):.3e} '
              f'median={row["median"]:.3e} DC<n_x^2>={row["dc_nx2"]:.4f} '
              f'({time.time() - t0:.1f}s)', flush=True)
    except Exception:
        row.update(ok=False, traceback_tail=''.join(traceback.format_exc().splitlines(True)[-4:]))
        print(f'  !! {route} {fixture} {protocol} N={N} FAILED\n{row["traceback_tail"]}',
              flush=True)
    row['seconds'] = round(time.time() - t0, 2)
    log.emit(row)
    return row


def solve_row(log, fixture, N):
    shapes = FIXTURES[fixture]
    solver = _legacy_wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)], shape_nvf_grid_size=64, **CFG)
    painted = solver.freeze_geometry().boundary_plans[0] is not None
    px = PER / N
    t0 = time.time()
    vals = np.array([sum_T([dict(s, cx=s['cx'] + k / 8 * px) for s in shapes], N)
                     for k in range(8)])
    row = dict(kind='solve', route='solve', fixture=fixture, protocol='x8', N=N, painted=painted,
               sumT=vals.tolist(), spread=float(vals.max() - vals.min()), ok=True,
               seconds=round(time.time() - t0, 1))
    print(f'  solve {fixture:<12s} painted={painted!s:<5s} N={N:4d} T(0)={vals[0]:.12f} '
          f'spread={row["spread"]:.3e} ({row["seconds"]:.1f}s)', flush=True)
    log.emit(row)
    return row


def fit_exponents(log, rows, ns):
    """spread ~ N^-p by least squares over the N run, with the pairwise exponents."""
    print(f'== exponent fit over N = {list(ns)} (spread ~ N^-p)')
    groups = {}
    for r in rows:
        if r.get('ok'):
            groups.setdefault((r['route'], r['fixture'], r['protocol']), {})[r['N']] = r['spread']
    for key, by_n in groups.items():
        if sorted(by_n) != sorted(ns) or len(ns) < 2:
            continue
        n = np.array(sorted(by_n), dtype=float)
        v = np.array([by_n[int(k)] for k in n])
        fit = dict(kind='fit', route=key[0], fixture=key[1], protocol=key[2],
                   values={str(int(k)): float(by_n[int(k)]) for k in n})
        doc = DOCUMENTED.get(key)
        if doc is not None and len(doc) == len(v):
            fit['documented'] = list(doc)
            fit['ratio_to_documented'] = [float(a / b) for a, b in zip(v, doc)]
        label = f'{key[0]} {key[1]} {key[2]}'
        if v.max() < 1e-12:
            fit['exponent'] = None
            print(f'  {label:<42s} ' + ' '.join(f'{x:.3e}' for x in v) + '  at rounding')
        else:
            slope, _ = np.polyfit(np.log(n), np.log(v), 1)
            pair = [float(np.log(v[i] / v[i + 1]) / np.log(n[i + 1] / n[i]))
                    for i in range(len(v) - 1)]
            fit.update(exponent=float(-slope), pairwise=pair)
            print(f'  {label:<42s} ' + ' '.join(f'{x:.3e}' for x in v)
                  + f'  p={-slope:.3f} pairwise={[round(q, 3) for q in pair]}'
                  + (f'  documented {list(doc)}' if doc else ''))
        log.emit(fit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sections', nargs='+', choices=['coeff', 'solve'],
                        default=['coeff', 'solve'])
    parser.add_argument('--ns', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--out', default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True,
                          text=True, cwd=HERE).stdout.strip()
    log = Log(args.out)
    log.emit(dict(kind='header', head=head, numpy=np.__version__, ns=args.ns,
                  sections=args.sections, started=time.strftime('%Y-%m-%d %H:%M:%S')))
    print(f'HEAD={head} numpy {np.__version__}')
    rows = []
    if 'coeff' in args.sections:
        print('== coefficient level: max |unphased c(shifted) - c(base)|')
        for N in args.ns:
            for route, fixture, protocol in COEFF_ROWS:
                rows.append(coeff_row(log, route, fixture, protocol, N))
    if 'solve' in args.sections:
        print('== solve level: sum(T) max - min over 8 x-shifts of k/8 px')
        for fixture in ('TWO_CLOSED', 'ONE_BOX'):
            for N in args.ns:
                rows.append(solve_row(log, fixture, N))
    fit_exponents(log, rows, args.ns)
    print(f'rows appended to {os.path.relpath(args.out, HERE)}')


if __name__ == '__main__':
    main()
