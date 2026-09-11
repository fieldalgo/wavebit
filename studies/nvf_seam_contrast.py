"""Does a different-material seam still step at contact and at material equality, now that the
seam is put back by its contrast rather than by a sign on the pair record?

Until 2026-09-11 the piece field (``parametric._piece_field``) read a cross pair's two
material slots as an equality bit: equal cancelled, different co-oriented along the *primary's*
normal. That was right on the shared line and wrong everywhere near it -- two opposed faces a
positive gap apart carry no record and cancelled whatever their materials, so ``sum(T)``
stepped at contact (-2.7e-3 at order 4 on the K fixture), stepped again when a material swept
through equality (-5e-3), and depended on which shape was listed first (1.5e-3). The seam
contrast cancels every record and adds, after the sum, ``c * h * max(0, -n_a.n_b) * (n_a -
n_b)`` for every pair of pieces of different owners, with ``c = (|eps_a - eps_b| / (|eps_a| +
|eps_b|)) ** 4``, ``h`` the harmonic mean of the two weights and the difference oriented from
the lower-|eps| shape to the higher (docs/normal-field-known-limits.md, "The seam contrast").
This study reproduces the evidence that decided it, so a later change can say at once which
of these numbers it moved:

  ladder   the K fixture at orders 4..12: the contact step ``T(gap 1e-9) - T(0)`` at N = 512
           beside the same-material control, the distance to the explicit Li rule at contact,
           and the material-equality step ``T(9 + 1e-7 | 9) - T(9 | 9)`` at N = 128 beside
           Li's own
  ramp     ``T(9 + d | 9) - T(9 | 9)`` at order 4 for d from 1e-7 to 1, against Li: the seam
           has to open linearly in the contrast, which is what the fourth power buys (a lower
           power leaves a band of width ~ d**(p/4) and a bump of T against the material)
  listing  the same structure listed ``[A, B]`` and ``[B, A]`` at N = 504, where the 300 offset
           between the two centres is an integer number of pixels and the listing swap is a
           pure permutation of the samples (at N = 512 a sampling offset of the first-shape
           anchor remains, ~4e-5, which is a different thing and is measured too)
  three    three boxes ``4 | 9 | 12`` touching: opening either seam by 1e-9
  band     the K field against the material-aware exact-interface oracle over a 1.5-pixel
           band (``tests/gamma_instruments.py::band_error``), and the leak above the seam's
           end -- ``|n_x|`` on the seam line past the boxes' top face, exactly 0 before, ~c now
           (a reading that means something only where the seam's end is the two boxes' common
           corner: with unequal heights, a circle or a rotated pair the window holds a real
           exposed corner, whose wedge reads the diagonal, ~0.7, on both trees)
  matrix   eleven held-out cells at order 4 (one at 8), each read for its contact step, its
           equality ramp against Li at d = 1e-3, its band error and its distance to Li at
           contact: five contrasts (4|9, 2.25|12.25, 4|4.4, lossy 4|4+0.5j, metal
           12.25|-10+1j), unequal heights 300|280, a rotated pair (angle 0.3 in place, which
           overlaps and routes to the polygon route, so it is a control the fix must not move),
           a box against a circle, theta 0, 700 nm and order 8

Fixture (``tests/test_gamma_layers.py::K_SHORT``): square cell 700, two 300 x 300 boxes at
cx = -150 / +150 of eps 4 | 9 on background 1, thickness 300, wavelength 900, top 1, bot 2.25,
theta 0.2, phi = psi = 0, NumPy float64, projector cache cleared before every solve.

Usage, from the repo root::

    python studies/nvf_seam_contrast.py                    # every section
    python studies/nvf_seam_contrast.py --sections ladder ramp --tag production
    PYTHONPATH=<git archive of an older commit> python studies/nvf_seam_contrast.py --tag before

The second form is how the "before" column was taken: ``wavebit`` is imported from the
archived tree while the instrument and the oracle come from this checkout, and the header row
records the tag and where ``wavebit`` was imported from relative to this file. Rows stream to
``studies/results/nvf_seam_contrast.log`` as JSON lines; the log holds both runs.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import wavebit
from wavebit._core import overlap as O
from wavebit._core import parametric as P
from wavebit._core.lattice import as_lattice, sample_grid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tests'))
from gamma_instruments import band_error  # noqa: E402

DEFAULT_OUT = os.path.join(HERE, 'results', 'nvf_seam_contrast.log')
PERIOD = 700.0
SQUARE = np.array([[PERIOD, 0.0], [0.0, PERIOD]])
LAT = as_lattice(lattice=SQUARE)
SOLVE_KW = dict(backend=0, wavelength=900.0, top=1.0, bot=2.25, theta=0.2, phi=0.0, psi=0.0,
                thickness=[300.0])
GAP = 1e-9
N_SOLVE, N_EQUAL, N_LIST = 512, 128, 504
ORDERS = (4, 6, 8, 10, 12)
RAMP = (1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 1.0)

# What the 2026-09-11 session recorded on the K ladder, before (b92e85a) and after the seam
# contrast, so a rerun says at once whether either column is the one it reproduces. The after
# column is at rounding, so its sign is noise: this instrument reads it negative.
DOCUMENTED = dict(
    contact_before=(-2.7e-3, -8.7e-4, -2.1e-4, -7.2e-4, -3.4e-4),
    contact_after=(2.2e-12, 1.4e-12, 1.2e-12, 1.0e-12, 1.4e-12),
    equality_before=(-5.0e-3, -4.8e-4, -6.0e-4, -1.4e-4, -4.0e-5),
    equality_after=2.2e-9,
    listing_before=1.5e-3, listing_after=6e-15,
)


def _box(**geom):
    return dict(dict(type='rectangle', cx=0.0, cy=0.0, angle=0.0, material=4.0), **geom)


def pair(ea, eb, ly_b=300.0, angle=0.0, second='rectangle'):
    """Two 300-wide shapes touching at ``x = 0``; the K fixture at ``ea, eb = 4, 9``."""
    a = _box(cx=-150.0, lx=300.0, ly=300.0, material=ea, angle=angle)
    if second == 'circle':
        b = dict(type='circle', cx=150.0, cy=0.0, lx=300.0, material=eb)
    else:
        b = _box(cx=150.0, lx=300.0, ly=ly_b, material=eb, angle=angle)
    return [a, b]


def opened(shapes, gap=GAP):
    """The same shapes with the seam at ``x = 0`` opened by ``gap``, symmetrically."""
    return [dict(s, cx=s['cx'] - 0.5 * gap * np.sign(s['cx'])) for s in shapes]


K = pair(4.0, 9.0)
D = pair(4.0, 4.0)
THREE = [_box(cx=-200.0, lx=200.0, ly=300.0, material=4.0),
         _box(cx=0.0, lx=200.0, ly=300.0, material=9.0),
         _box(cx=200.0, lx=200.0, ly=300.0, material=12.0)]


# ------------------------------------------------------------------------------- solves
def sum_T(shapes, order=4, normal_grid=N_SOLVE, factorization=None, **incidence):
    P._PROJECTOR_CACHE.clear()
    kw = dict(SOLVE_KW, **incidence)
    if factorization is not None:
        # the explicit rule samples no normal field, and a grid it cannot read is refused
        kw['factorization'] = factorization
    else:
        kw['normal_grid'] = normal_grid
    res = wavebit.rcwa(lattice=SQUARE, structure=[dict(background=1.0, shapes=shapes)],
                       truncation_order=dict(u=order, v=order), **kw).solve()
    return float(np.sum(np.asarray(res.T)).real)


def li_T(shapes, order=4, **incidence):
    """The explicit Li rule on the same shapes -- axis-aligned rectangles only."""
    return sum_T(shapes, order=order, factorization='li', **incidence)


def li_applies(shapes):
    return all(s['type'] == 'rectangle' and s.get('angle', 0.0) == 0.0 for s in shapes)


def field(shapes, n=N_SOLVE):
    return P.normal_vector_field_analytical(np, (n, n), shapes, np.float64, LAT)


def oracle(shapes, groups):
    return O.interfaces_of(P._primitives(shapes, LAT), LAT, P.plan_gamma(shapes, LAT).ring,
                           groups=groups)


def band(shapes, n=N_SOLVE):
    """``(mean_deg, over_45, samples)`` against the material-aware oracle."""
    n_x, n_y = field(shapes, n)
    labels = tuple(complex(s['material']) for s in shapes)
    return band_error(n_x, n_y, (n, n), SQUARE, oracle(shapes, labels)), (n_x, n_y)


def leak(n_x, n=N_SOLVE, top=150.0, reach=100.0):
    """``max |n_x|`` on the column nearest the seam line, past the boxes' top face.

    The seam runs over ``|y| < 150``; above it the two boxes' ``+-u`` face lines continue into
    background, where before the seam contrast the two faces cancelled to exactly zero and the
    field read the top faces' ``(0, +-1)``. The contrast's co-orientation carries ``c`` past
    the run's end, so this reads about ``c`` now. The sample grid is in the internal frame
    (the cell spans ``[0, 700)`` and the shapes sit about its centre), rows within two pixels
    of the corner are skipped as the corner's own wedge, and the window stops ``reach`` above
    the face: further up the top faces meet the periodic images' bottom faces on a midline
    tie that reads the u axis, which is right and is not the leak.
    """
    X, Y = sample_grid(np, (n, n), LAT, np.float64)
    X, Y = np.asarray(X), np.asarray(Y)
    col = int(np.argmin(np.abs(X[0] - 0.5 * PERIOD)))
    y = Y[:, col] - 0.5 * PERIOD
    rows = (y > top + 2.0 * PERIOD / n) & (y < top + reach)
    return float(np.abs(np.asarray(n_x)[rows, col]).max())


# -------------------------------------------------------------------------------- output
class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def emit(self, row):
        with open(self.path, 'a') as sink:
            sink.write(json.dumps(_plain(row)) + '\n')


def _plain(v):
    """JSON-safe: complex as its repr string, numpy scalars as floats, nothing else touched."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, complex):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


def _failed(exc):
    """The class and the first line of the message -- never a traceback, which names files."""
    return f'{type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ""}'


def timed(log, row, work):
    t0 = time.time()
    try:
        row.update(work(), ok=True)
    except Exception as exc:  # noqa: BLE001 -- a refused cell is a row, not a crash
        row.update(ok=False, error=_failed(exc))
    row['seconds'] = round(time.time() - t0, 1)
    log.emit(row)
    return row


# ------------------------------------------------------------------------------ sections
def ladder(log):
    print('== ladder: the K fixture, orders 4..12')
    for i, order in enumerate(ORDERS):
        def work(order=order, i=i):
            t0, t_gap = sum_T(K, order), sum_T(opened(K), order)
            d0, d_gap = sum_T(D, order), sum_T(opened(D), order)
            li0, li_gap = li_T(K, order), li_T(opened(K), order)
            e0 = sum_T(pair(9.0, 9.0), order, N_EQUAL)
            e1 = sum_T(pair(9.0 + 1e-7, 9.0), order, N_EQUAL)
            l0 = li_T(pair(9.0, 9.0), order)
            l1 = li_T(pair(9.0 + 1e-7, 9.0), order)
            out = dict(T0=t0, contact_step=t_gap - t0, control_step=d_gap - d0,
                       li_contact_step=li_gap - li0, li_distance=t0 - li0,
                       equality_step=e1 - e0, li_equality_step=l1 - l0,
                       documented_contact=(DOCUMENTED['contact_before'][i],
                                           DOCUMENTED['contact_after'][i]),
                       documented_equality=(DOCUMENTED['equality_before'][i],
                                            DOCUMENTED['equality_after']))
            print(f'  order {order:2d}  contact {out["contact_step"]:+.2e}  control '
                  f'{out["control_step"]:+.2e}  Li contact {out["li_contact_step"]:+.2e}  '
                  f'Li distance {out["li_distance"]:+.2e}  equality {out["equality_step"]:+.2e}'
                  f'  Li equality {out["li_equality_step"]:+.2e}', flush=True)
            return out
        timed(log, dict(kind='ladder', order=order, N=N_SOLVE, N_equality=N_EQUAL), work)


def ramp(log):
    print('== ramp: T(9 + d | 9) - T(9 | 9) at order 4, N = 128, against Li')
    base_nvf, base_li = sum_T(pair(9.0, 9.0), 4, N_EQUAL), li_T(pair(9.0, 9.0), 4)
    for d in RAMP:
        def work(d=d):
            nvf = sum_T(pair(9.0 + d, 9.0), 4, N_EQUAL) - base_nvf
            li = li_T(pair(9.0 + d, 9.0), 4) - base_li
            out = dict(nvf=nvf, li=li, ratio=nvf / li if li else None)
            print(f'  d = {d:.0e}  nvf {nvf:+.3e}  li {li:+.3e}  ratio {out["ratio"]:.4f}',
                  flush=True)
            return out
        timed(log, dict(kind='ramp', d=d, order=4, N=N_EQUAL), work)


def listing(log):
    print('== listing: T([A, B]) - T([B, A]) at order 4')
    for n in (N_LIST, N_SOLVE):
        def work(n=n):
            ab, ba = sum_T(K, 4, n), sum_T(K[::-1], 4, n)
            print(f'  N = {n}  {ab - ba:+.2e}', flush=True)
            return dict(T_ab=ab, T_ba=ba, swap=ab - ba)
        timed(log, dict(kind='listing', order=4, N=n,
                        documented=(DOCUMENTED['listing_before'], DOCUMENTED['listing_after'])),
              work)


def three(log):
    print('== three: 4 | 9 | 12 touching, either seam opened by 1e-9, order 4')

    def work():
        t0 = sum_T(THREE, 4)
        left = [dict(THREE[0], cx=THREE[0]['cx'] - GAP)] + THREE[1:]
        right = THREE[:2] + [dict(THREE[2], cx=THREE[2]['cx'] + GAP)]
        out = dict(T0=t0, open_left=sum_T(left, 4) - t0, open_right=sum_T(right, 4) - t0)
        print(f'  left seam {out["open_left"]:+.2e}  right seam {out["open_right"]:+.2e}',
              flush=True)
        return out
    timed(log, dict(kind='three', order=4, N=N_SOLVE, gap=GAP), work)


def band_section(log):
    print('== band: the K field against the material-aware oracle, N = 512')

    def work():
        (mean, over, count), (n_x, _) = band(K)
        (c_mean, c_over, _), (c_x, _) = band(D)
        out = dict(mean_deg=mean, over_45=over, samples=count, leak=leak(n_x),
                   control_mean_deg=c_mean, control_over_45=c_over, control_leak=leak(c_x))
        print(f'  K {mean:.4f} deg, over-45 {over:.5f}, leak {out["leak"]:.4f}; same-material '
              f'control {c_mean:.4f} deg, over-45 {c_over:.5f}, leak {out["control_leak"]:.1e}',
              flush=True)
        return out
    timed(log, dict(kind='band', N=N_SOLVE, band_px=1.5), work)


# name -> (shapes, solve kwargs); the base material of the equality ramp is the second shape's.
CELLS = {
    '4|9': (pair(4.0, 9.0), {}),
    '2.25|12.25': (pair(2.25, 12.25), {}),
    '4|4.4': (pair(4.0, 4.4), {}),
    'lossy 4|4+0.5j': (pair(4.0, 4.0 + 0.5j), {}),
    'metal 12.25|-10+1j': (pair(12.25, -10.0 + 1.0j), {}),
    'heights 300|280': (pair(4.0, 9.0, ly_b=280.0), {}),
    'rotated 0.3 (overlaps: polygon route)': (pair(4.0, 9.0, angle=0.3), {}),
    'box|circle': (pair(4.0, 9.0, second='circle'), {}),
    'theta 0': (pair(4.0, 9.0), dict(theta=0.0)),
    '700 nm': (pair(4.0, 9.0), dict(wavelength=700.0)),
    'order 8': (pair(4.0, 9.0), dict(order=8)),
}


def matrix(log):
    print('== matrix: eleven held-out cells (order 4 unless named), N = 512, ramp d = 1e-3')
    for name, (shapes, kw) in CELLS.items():
        def work(name=name, shapes=shapes, kw=kw):
            eb = shapes[1]['material']
            kw = dict(kw)
            order = kw.pop('order', 4)
            equal = [dict(s, material=eb) for s in shapes]
            ramped = [dict(equal[0], material=eb + 1e-3), equal[1]]
            t0 = sum_T(shapes, order, **kw)
            out = dict(order=order, contact_step=sum_T(opened(shapes), order, **kw) - t0,
                       ramp=sum_T(ramped, order, **kw) - sum_T(equal, order, **kw))
            if li_applies(shapes):
                out['li_ramp'] = li_T(ramped, order, **kw) - li_T(equal, order, **kw)
                out['li_distance'] = t0 - li_T(shapes, order, **kw)
            try:
                (mean, over, count), (n_x, _) = band(shapes)
                out.update(band_mean_deg=mean, band_over_45=over, band_samples=count,
                           leak=leak(n_x))
            except Exception as exc:  # noqa: BLE001 -- the overlapping control is refused here
                out['band_error'] = _failed(exc)
            fmt = lambda k: f'{out[k]:+.2e}' if k in out else '   n/a   '  # noqa: E731
            print(f'  {name:<38s} contact {fmt("contact_step")}  ramp {fmt("ramp")} vs Li '
                  f'{fmt("li_ramp")}  band {out.get("band_mean_deg", float("nan")):.3f} deg '
                  f'/ {out.get("band_over_45", float("nan")):.4f}  Li distance '
                  f'{fmt("li_distance")}  leak {out.get("leak", float("nan")):.3f}', flush=True)
            return out
        timed(log, dict(kind='matrix', cell=name, N=N_SOLVE, ramp_d=1e-3,
                        materials=[s['material'] for s in shapes]), work)


SECTIONS = dict(ladder=ladder, ramp=ramp, listing=listing, three=three, band=band_section,
                matrix=matrix)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sections', nargs='+', choices=list(SECTIONS), default=list(SECTIONS))
    parser.add_argument('--tag', default='production',
                        help='names the run in the header; "before" for an archived tree')
    parser.add_argument('--out', default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True,
                          text=True, cwd=HERE).stdout.strip()
    source = os.path.relpath(os.path.dirname(wavebit.__file__), HERE)
    log = Log(args.out)
    log.emit(dict(kind='header', head=head, tag=args.tag, numpy=np.__version__,
                  wavebit_in_checkout=source == os.path.join('..', 'wavebit'),
                  sections=args.sections, started=time.strftime('%Y-%m-%d %H:%M:%S')))
    print(f'HEAD={head} tag={args.tag} numpy {np.__version__}')
    for name in args.sections:
        SECTIONS[name](log)
    print(f'rows appended to {os.path.relpath(args.out, HERE)}')


if __name__ == '__main__':
    main()
