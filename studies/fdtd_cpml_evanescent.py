"""What ``KAPPA_MAX`` and ``ALPHA_RELATIVE`` buy on a source's near field, and what they cost.

**The question.** Stage 1 measured the CPML with a travelling packet only
(``tests/test_fd_pml.py``) and found that both CFS constants of ``src/wavebit/fd/_lower/cpml.py``
*cost* accuracy there: with ``ALPHA_RELATIVE = 0`` the 12-cell normal-incidence number falls
127x and with ``KAPPA_MAX = 1`` 1.4x. What the two constants are classically for -- absorbing
the *evanescent* near field of a source close to the layer -- no probe could see until stage 2
gave the kernel a point dipole. This study measures it over ``KAPPA_MAX`` in (1, 2, 3, 5) x
``ALPHA_RELATIVE`` in (0, 0.5, 1.5, 3), and re-measures the propagating table at all 16
settings, so the trade is in one place.

**The near-field probe.** An ``fd.Dipole`` (``Band(wavelength=1, df=0.25)``, the default cubic
B-spline stamp) at ``d`` = 0.05 / 0.1 / 0.2 / 0.5 wavelengths from the inner face of a 12-cell
PML on ``z+``, on a 40-cells-per-wavelength grid, so ``d`` = 2 / 4 / 8 / 20 cells. At 20 cells
per wavelength the 0.05 row would put the stamp inside the layer, which ``fd`` refuses. The box
has 1 wavelength of interior across x and y, 0.5 wavelengths below the dipole, and a 12-cell
PML on every face. Two orientations: ``Ez`` (normal to the face) and ``Ex`` (parallel to it).
Each run records the dipole's own component on the axis at two points: ``p``, **halfway
between the dipole and the layer**, and ``f``, at the layer's inner face.

**The doubling method.** The reference is the same run on the same cells in a box whose ``z+``
face sits ``ext`` further out, with ``2 ext + 1.5 d >= 1.3 T`` for the window ``T``. So nothing
the reference's own ``z+`` layer returns reaches the probe inside the window. Only the ``z+``
face's profiles are swept; the other five faces keep the default constants in both runs. So
``test - reference`` at ``p`` is the swept layer's return and nothing else, and one reference
per (d, orientation) serves all 16 settings. Three numbers are reported, all at ``p``:

- ``R_face = |DFT(test - ref) at p| / |DFT(ref) at f|`` at the band centre ``f0``: the return
  over the field that reached the layer, the closest thing to a reflection coefficient.
- ``R_local``: the same return over the reference field at ``p`` itself, which near the dipole
  is the strong near field. It is what a structure sitting at ``p`` would see wrong.
- the worst in time, ``max |test - ref| / max |ref|`` at ``p``.

``R_face`` and ``R_local`` share the numerator, so for one (d, orientation) they rank the 16
settings identically; they differ only in how the eight cases weigh against each other. The
control in the log moves ``ext`` by 40 more cells: the reference probe series change by exactly
0.0 at both points, which is what licenses reading the difference as the layer's return.

**The propagating table** is ``tests/test_fd_pml.py``'s, measured by its own helper
(``fd_helpers.doubling_reflection``: 20 cells per wavelength, courant 0.5, the default
``reflection=1e-8``, the amplitude ratio at the carrier), all five depths x four angles. The
two constants are patched in ``wavebit.fd._lower.cpml`` exactly as that file's
``test_the_cfs_pole_and_kappa_are_not_settled_by_this_probe`` patches them. The 12-cell row is
measured again at 40 cells per wavelength, the near-field grid, where the same 12 cells are
half as thick in wavelengths. The default row of the 20-cell table reproduces
``tests/test_fd_pml.py::TABLE`` to its printed digits.

**The decision rule the stage-2 plan set.** The constants move only if one setting wins on the
near field without losing more than 2x on the propagating table. "Losing 2x" is read entry by
entry: a setting is *admissible* when every one of the 20 entries is at most 2x the default's
(``KAPPA_MAX = 2``, ``ALPHA_RELATIVE = 1.5``, measured in the same run). "Wins on the near field"
has more than one honest reading, and they disagree, so the log prints all four.

Measured 2026-09-19, CPU, NumPy, 5 workers x 2 threads, 841 s wall
(``studies/results/fdtd_cpml_evanescent.log``). ``*`` marks the default.

``R_face`` at ``f0``::

    setting        Ez@0.05  Ez@0.1   Ez@0.2   Ez@0.5   Ex@0.05  Ex@0.1   Ex@0.2   Ex@0.5
    k=1 a=0        3.9e-07  5.3e-07  3.6e-07  2.3e-07  1.1e-06  4.6e-07  2.0e-07  1.1e-07
    k=1 a=0.5      3.4e-07  4.2e-07  1.6e-07  9.6e-08  1.0e-06  4.9e-07  4.6e-08  9.1e-08
    k=1 a=1.5      2.1e-07  2.7e-07  2.1e-07  6.7e-07  5.5e-07  4.1e-07  2.5e-06  4.9e-06
    k=1 a=3        2.5e-08  5.2e-07  4.8e-06  2.9e-05  3.2e-06  1.9e-05  9.2e-05  1.8e-04
    k=2 a=0        3.9e-07  5.1e-07  3.0e-07  1.8e-07  1.1e-06  5.1e-07  1.5e-07  9.8e-08
    k=2 a=0.5      3.5e-07  4.3e-07  1.6e-07  1.0e-07  1.0e-06  5.1e-07  4.5e-08  8.3e-08
    k=2 a=1.5 *    2.3e-07  2.9e-07  2.0e-07  7.2e-07  6.2e-07  5.9e-07  2.7e-06  5.2e-06
    k=2 a=3        6.5e-08  5.4e-07  4.7e-06  2.9e-05  2.9e-06  1.8e-05  8.8e-05  1.7e-04
    k=3 a=0.5      3.6e-07  4.4e-07  1.6e-07  8.6e-08  1.1e-06  5.2e-07  4.7e-08  7.4e-08
    k=5 a=0.5      3.7e-07  4.7e-07  1.7e-07  6.7e-08  1.1e-06  5.5e-07  7.1e-08  3.7e-08
    (the other six rows are in the log)

The trade, every column a ratio to the default (< 1 is better), except the two ``worst``
columns, which are the largest absolute ``R`` over the eight near-field cases::

    setting      near max  near gmean  worst R_face  worst R_local  table max  row12  row12@40  adm.
    k=1 a=0         1.86      0.50       1.1e-06        3.3e-07        4.27     0.40    0.95   no
    k=1 a=0.5       1.62      0.31       1.0e-06        2.9e-07        0.89     0.43    0.58   yes
    k=1 a=1.5       1.06      0.91       4.9e-06        2.7e-06        1.37     0.78    0.95   yes
    k=2 a=0         1.81      0.46       1.1e-06        3.3e-07        4.16     0.69    0.95   no
    k=2 a=0.5       1.66      0.32       1.0e-06        3.0e-07        1.00     0.73    0.74   yes
    k=2 a=1.5 *     1.00      1.00       5.2e-06        2.8e-06        1.00     1.00    1.00   yes
    k=3 a=0.5       1.69      0.31       1.1e-06        3.1e-07        1.09     0.88    0.89   yes
    k=3 a=1.5       1.25      1.09       5.7e-06        3.1e-06        2.04     1.45    1.42   no
    k=5 a=0.5       1.78      0.30       1.1e-06        3.2e-07        1.53     1.03    1.05   yes
    k=5 a=1.5       1.81      1.36       7.8e-06        4.3e-06        9.42     3.34    1.90   no
    any a=3        39-55     8.9-15      1.7-2.4e-04    0.9-1.3e-04      40-286  21-57   29-41  no

and the propagating 12-cell row at 20 cells per wavelength (0 / 30 / 60 / 80 degrees)::

    k=2 a=1.5 *   9.2e-05  4.7e-05  1.1e-04  3.4e-02
    k=2 a=0.5     2.5e-06  1.9e-06  5.4e-05  2.5e-02
    k=1 a=0.5     1.7e-06  1.4e-06  1.9e-05  1.5e-02

**What it says.**

1. **The evanescent near field is not what limits a 12-cell layer at any admissible setting.**
   A dipole 0.05 wavelengths (2 cells) from the layer gets back 2e-7 to 1.1e-6 of the field at
   the face, and 1.4e-7 to 3.2e-7 of its own local field. The propagating 12-cell entry at
   normal incidence is 9.2e-5. The near-field number moves by less than 2x across every
   admissible setting at 0.05 and 0.1 wavelengths.
2. **The CFS pole does what the textbook says, but only at the closest distance, and only a
   little.** At ``d = 0.05`` a larger ``alpha`` lowers ``R``: 3.9e-7 / 3.5e-7 / 2.3e-7 / 6.5e-8
   for ``alpha`` = 0 / 0.5 / 1.5 / 3 on the ``Ez`` dipole at ``kappa = 2``. Further out the
   same pole *raises* it: ``alpha = 1.5`` against ``alpha = 0.5`` is 7x at ``d = 0.5`` on
   ``Ez`` and 60x at 0.2 and 0.5 on ``Ex`` (1.25x on ``Ez`` at 0.2), because by then the
   return is the propagating part of the dipole's field. That is the propagating table's own
   finding seen from a dipole.
3. **``alpha = 3`` is refused by both probes** (30-50x worse on the near field from 0.2
   wavelengths on, 40-286x on the table). **``alpha = 0`` is not admissible**: it loses 4x on
   the deep 30-degree entries (5.6e-7 against 1.4e-7 at 24 cells), and on the near field it
   is nowhere more than 1.1x better than ``alpha = 0.5``.
4. **``kappa`` matters little on the near field.** At ``alpha <= 0.5`` the four ``kappa``
   rows agree within 1.15x at ``d <= 0.1``; at ``alpha = 1.5``, ``kappa = 5`` is up to 2.7x
   worse than ``kappa = 1``. Above 2 it costs on the table, most at 80 degrees and in the thin
   rows. ``kappa = 1`` gives the lowest 12-cell row at every ``alpha``, and at
   ``alpha = 0.5`` also the lowest near-field max ratio of the four.

**Which setting.** Read as "no near-field case may get worse" (the smallest max ratio), the
default ``(2, 1.5)`` wins: nothing admissible beats it on all eight cases, because at 0.05
wavelengths (both orientations) and at 0.1 (``Ez``) ``alpha = 1.5`` is 1.4-1.7x better than
0.5. Read as "the worst absolute
near-field return", or as the geometric mean over the eight cases, ``alpha = 0.5`` wins at any
``kappa`` by 5x (worst ``R_face`` 1.0e-6 against 5.2e-6) or 3.2x respectively. Among those,
**``KAPPA_MAX = 1, ALPHA_RELATIVE = 0.5``** is the only one that also improves every entry of
the propagating table (worst ratio 0.89; the 12-cell row 0.43x, 9.2e-5 -> 1.7e-6 at normal
incidence, 3.4e-2 -> 1.5e-2 at 80 degrees). ``(2, 0.5)`` is the conservative variant: it
changes one constant, no table entry gets worse (worst ratio 1.00), and 12c0 goes to 2.5e-6.

**What moving to (1, 0.5) would cost.** The near-field return rises 1.4-1.6x at 0.05
wavelengths (``R_face`` 3.4e-7 against 2.3e-7 for ``Ez``, 1.0e-6 against 6.2e-7 for ``Ex``) and
at 0.1 on ``Ez``, and falls at 0.1 on ``Ex`` (0.82x). In absolute terms it stays at or below
1.0e-6. In the repository it would move every CPML-dependent
number: ``tests/test_fd_pml.py::TABLE`` and the thin-PML refusal message that quotes it, that
file's ``test_the_cfs_pole_and_kappa_are_not_settled_by_this_probe``, which pins ``(1.5, 2.0)``,
and the CPML arrays of ``fd_golden_stage1.npz``, which would need a re-freeze. The 12-cell
refusal line would also want re-reading: at ``(1, 0.5)`` the 8-cell row reads 3.4e-4 /
2.4e-4 / 9.5e-5 / 7.3e-3 against the default's 4.6e-3 / 3.3e-3 / 8.8e-4 / 2.6e-2.

**What this does not measure.** The window is 7.7 periods, so a late-time growth of an
``alpha = 0`` layer, or a slow low-frequency tail, is outside it. There is one band (f0 +- 0.93
f0 at the cutoff) and one probe line, the axis. The B-spline stamp spreads the dipole over +-2
cells, which cuts the evanescent spectrum off near ``k ~ 1/(2 dx)``, so a point source finer
than the grid is not represented. Only the ``z+`` face was swept; a corner, where two layers
meet a near field, was not isolated. The ``Absorber`` strength (``damping_profile``) is a
separate question and is not measured here.

Run from the repository root::

    python studies/fdtd_cpml_evanescent.py [--workers 5] > studies/results/fdtd_cpml_evanescent.log

NumPy, CPU, about 4100 s of worker time. Each worker is given two BLAS/OpenMP threads.
"""
import argparse
import contextlib
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '2')

import numpy as np                                      # noqa: E402

sys.path.insert(0, '.')
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')
import wavebit.fd as fd                                 # noqa: E402
import wavebit.fd._lower.cpml as cpml_module            # noqa: E402
from wavebit.fd._lower.params import _cpml              # noqa: E402
from fd_helpers import doubling_reflection, run_mutated, to_numpy   # noqa: E402

KAPPAS = (1.0, 2.0, 3.0, 5.0)
ALPHAS = (0.0, 0.5, 1.5, 3.0)
DEFAULT = (2.0, 1.5)
DISTANCES = (0.05, 0.1, 0.2, 0.5)          # wavelengths from the PML's inner face
ORIENTATIONS = ('Ez', 'Ex')
TABLE_CELLS = (6, 8, 12, 16, 24)
TABLE_ANGLES = (0, 30, 60, 80)

WAVELENGTH = 1.0
DF = 0.25
CPW_NEAR = 40
PML = 12
ACROSS = 40                                 # interior cells across x and y (1 wavelength)
BELOW = 20                                  # interior cells between the z- layer and the dipole
COURANT = 0.5
EXTRA_TAIL = 3.0                            # window beyond the pulse's 2 t0, in periods
EXT_CONTROL = 40                            # extra reference cells in the control


@contextlib.contextmanager
def constants(kappa, alpha):
    """``KAPPA_MAX`` and ``ALPHA_RELATIVE`` patched in the one module that reads them."""
    saved = cpml_module.KAPPA_MAX, cpml_module.ALPHA_RELATIVE
    cpml_module.KAPPA_MAX, cpml_module.ALPHA_RELATIVE = kappa, alpha
    try:
        yield
    finally:
        cpml_module.KAPPA_MAX, cpml_module.ALPHA_RELATIVE = saved


def all_faces(kappa, alpha):
    """``mutate_params`` rebuilding every face's profiles at (kappa, alpha)."""
    def mutate(params, scene, plan, rt):
        with constants(kappa, alpha):
            return {**params, 'cpml': _cpml(scene, plan, rt)}
    return mutate


def high_z_face_only(kappa, alpha):
    """``mutate_params`` rebuilding the ``z+`` face's profiles only; the rest stay default."""
    def mutate(params, scene, plan, rt):
        with constants(kappa, alpha):
            swept = _cpml(scene, plan, rt)
        cpml = dict(params['cpml'])
        cpml.update({k: v for k, v in swept.items() if k.startswith('z+:')})
        return {**params, 'cpml': cpml}
    return mutate


# ---------------------------------------------------------------------------------------------
# The near-field probe
# ---------------------------------------------------------------------------------------------

def band():
    return fd.Band(wavelength=WAVELENGTH, df=DF)


def window():
    """``(steps, T, dt, dx)`` of every near-field run."""
    dx = WAVELENGTH / CPW_NEAR
    dt = COURANT * dx / math.sqrt(3.0)
    b = band()
    t0 = b.sigmas / (2 * math.pi * DF)
    T = 2 * t0 + EXTRA_TAIL * WAVELENGTH
    steps = int(math.ceil(T / dt))
    return steps, steps * dt, dt, dx


def extension(d_cells):
    """Reference cells beyond the test box: ``2 ext + 1.5 d >= 1.3 T`` in cells."""
    _, T, _, dx = window()
    return int(math.ceil((1.3 * T / dx - 1.5 * d_cells) / 2))


def near_scene(distance, component, extra=0):
    """The dipole ``distance`` wavelengths below the ``z+`` layer; the box ``extra`` cells longer."""
    steps, _, _, dx = window()
    d_cells = int(round(distance * CPW_NEAR))
    nz = PML + BELOW + d_cells + PML + extra
    n = ACROSS + 2 * PML
    z_dipole = (PML + BELOW) * dx
    z_probe = z_dipole + (d_cells // 2) * dx
    source = fd.Dipole('d', center=(0.0, 0.0, z_dipole), component=component, band=band())
    z_face = (PML + BELOW + d_cells) * dx
    monitors = (fd.Record('p', at=(0.0, 0.0, z_probe), component=component),
                fd.Record('f', at=(0.0, 0.0, z_face - 0.25 * dx), component=component))
    return fd.Scene(domain=fd.Domain(size=(n * dx, n * dx, nz * dx),
                                     center=(0.0, 0.0, nz * dx / 2)),
                    boundaries=fd.Boundaries.open(fd.PML(cells=PML)),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps, courant=COURANT),
                    sources=(source,), monitors=monitors)


def probe_series(scene, mutate=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = run_mutated(scene, mutate_params=mutate)
    t = np.asarray(res['p'].t, dtype=float)
    return t, {k: np.asarray(to_numpy(res[k].values), dtype=float) for k in ('p', 'f')}


def reference_task(args):
    distance, component, more = args
    d_cells = int(round(distance * CPW_NEAR))
    start = time.time()
    t, v = probe_series(near_scene(distance, component, extension(d_cells) + more))
    return (distance, component, more), (t, v), time.time() - start


def near_task(args):
    kappa, alpha, distance, component, reference = args
    start = time.time()
    t, got = probe_series(near_scene(distance, component), high_z_face_only(kappa, alpha))
    t_ref, want = reference
    assert np.array_equal(t, t_ref)
    phase = np.exp(1j * 2 * math.pi / WAVELENGTH * t)
    diff = got['p'] - want['p']
    returned = abs(np.sum(diff * phase))
    local = returned / abs(np.sum(want['p'] * phase))
    face = returned / abs(np.sum(want['f'] * phase))
    worst = float(np.max(np.abs(diff)) / np.max(np.abs(want['p'])))
    return (('near', kappa, alpha, distance, component), (local, worst, face),
            time.time() - start)


def table_task(args):
    kappa, alpha, cells, angle, cpw = args
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        spectral, worst, _ = doubling_reflection(cells, float(angle),
                                                 mutate_params=all_faces(kappa, alpha),
                                                 cells_per_wavelength=cpw)
    return ('table', kappa, alpha, cells, angle, cpw), (spectral, worst), time.time() - start


# ---------------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------------

def settings():
    return [(k, a) for k in KAPPAS for a in ALPHAS]


def label(kappa, alpha):
    mark = ' *' if (kappa, alpha) == DEFAULT else '  '
    return f'k={kappa:g} a={alpha:<3g}{mark}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    steps, T, dt, dx = window()
    print(f'near field: {CPW_NEAR} cells per wavelength (dx = {dx:g}), Band(wavelength='
          f'{WAVELENGTH:g}, df={DF:g}), shortest wavelength {band().shortest_wavelength:.4f}; '
          f'{steps} steps, window T = {T:.3f}; PML {PML} cells on every face, only z+ swept')
    print(f'distances {DISTANCES} wavelengths = '
          f'{tuple(int(round(d * CPW_NEAR)) for d in DISTANCES)} cells; reference extension '
          f'{tuple(extension(int(round(d * CPW_NEAR))) for d in DISTANCES)} cells')
    started = time.time()
    results, seconds = {}, 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        refs = {}
        jobs = [(d, c, 0) for d in DISTANCES for c in ORIENTATIONS]
        jobs += [(DISTANCES[0], c, EXT_CONTROL) for c in ORIENTATIONS]
        for key, series, s in pool.map(reference_task, jobs):
            refs[key] = series
            seconds += s
        tasks = [(k, a, d, c, refs[(d, c, 0)]) for k, a in settings()
                 for d in DISTANCES for c in ORIENTATIONS]
        table = [(k, a, n, ang, 20) for k, a in settings()
                 for n in TABLE_CELLS for ang in TABLE_ANGLES]
        table += [(k, a, PML, ang, CPW_NEAR) for k, a in settings() for ang in TABLE_ANGLES]
        futures = ([pool.submit(table_task, t) for t in table]
                   + [pool.submit(near_task, t) for t in tasks])
        for f in futures:
            key, value, s = f.result()
            results[key] = value
            seconds += s

    # -- the control: a longer reference box changes nothing inside the window
    print('\ncontrol: the reference with 40 more cells beyond z+, against the reference used')
    for c in ORIENTATIONS:
        for probe in ('p', 'f'):
            a = refs[(DISTANCES[0], c, 0)][1][probe]
            b = refs[(DISTANCES[0], c, EXT_CONTROL)][1][probe]
            print(f'  {c} dipole, d = {DISTANCES[0]}, probe {probe}: max |ref - ref_longer| / '
                  f'max |ref| = {np.max(np.abs(a - b)) / np.max(np.abs(a)):.1e}')

    print('\nreference amplitude |DFT| at f0: at the midpoint probe (p), at the inner face (f)')
    for c in ORIENTATIONS:
        for d in DISTANCES:
            t, v = refs[(d, c, 0)]
            phase = np.exp(1j * 2 * math.pi / WAVELENGTH * t)
            p_, f_ = (abs(np.sum(v[k] * phase)) for k in ('p', 'f'))
            print(f'  {c} d = {d:<4g}: p {p_:9.3e}   f {f_:9.3e}   p/f {p_ / f_:7.2f}')

    def near(k, a, d, c, which=0):
        return results[('near', k, a, d, c)][which]

    def prop(k, a, n, ang, cpw=20, which=0):
        return results[('table', k, a, n, ang, cpw)][which]

    cases = [(c, d) for c in ORIENTATIONS for d in DISTANCES]
    head = ' '.join(f'{c}@{d:<4g}'.rjust(8) for c, d in cases)
    for which, title in ((2, 'R_face = |DFT(test - ref) at p| / |DFT(ref) at f|, at f0'),
                         (0, 'R_local = |DFT(test - ref) at p| / |DFT(ref) at p|, at f0'),
                         (1, 'worst in time, max|test - ref| / max|ref| at p')):
        print(f'\nnear-field reflection, z+ face swept: {title}')
        print(f'{"setting":16s} {head}')
        for k, a in settings():
            print(f'{label(k, a):16s} ' + ' '.join(f'{near(k, a, d, c, which):8.1e}'
                                                  for c, d in cases))

    kd, ad = DEFAULT
    print('\nthe same, as the ratio to the default setting (normalisation-free: < 1 is better)')
    print(f'{"setting":16s} {head}')
    for k, a in settings():
        print(f'{label(k, a):16s} ' + ' '.join(f'{near(k, a, d, c) / near(kd, ad, d, c):8.2f}'
                                              for c, d in cases))

    print('\npropagating table (tests/test_fd_pml.py), R at the carrier, 20 cells/wavelength')
    cols = [(n, ang) for n in TABLE_CELLS for ang in TABLE_ANGLES]
    print(f'{"setting":16s} ' + ' '.join(f'{n:>2d}c{ang:<2d}'.rjust(7) for n, ang in cols))
    for k, a in settings():
        print(f'{label(k, a):16s} ' + ' '.join(f'{prop(k, a, n, ang):7.1e}' for n, ang in cols))

    print(f'\n12-cell row again at {CPW_NEAR} cells/wavelength (the near-field grid)')
    print(f'{"setting":16s} ' + ' '.join(f'{ang:>7d}' for ang in TABLE_ANGLES))
    for k, a in settings():
        print(f'{label(k, a):16s} ' + ' '.join(f'{prop(k, a, PML, ang, CPW_NEAR):7.1e}'
                                              for ang in TABLE_ANGLES))

    # -- the trade, one row per setting
    print('\nthe trade, every column a ratio to the default setting (< 1 is better):\n'
          '  near max / near gmean: over the 8 near-field cases (4 distances x 2 orientations)\n'
          '  table max: worst over all 20 entries of the propagating table (20 cells/wl)\n'
          '  row12 max: worst over its 12-cell row; row12@40 max: the same row at 40 cells/wl')
    print('  worst R_face / worst R_local: the largest ABSOLUTE near-field number over the 8 '
          'cases\n  (not a ratio), under each normalisation')
    print(f'{"setting":16s} {"near max":>9s} {"near gmean":>10s} {"worst Rface":>11s} '
          f'{"worst Rlocal":>12s} {"table max":>9s} {"row12 max":>9s} {"row12@40 max":>12s}'
          f'  admissible')
    summary = {}
    for k, a in settings():
        ratios = [near(k, a, d, c) / near(kd, ad, d, c) for c, d in cases]
        face = max(near(k, a, d, c, 2) for c, d in cases)
        local = max(near(k, a, d, c, 0) for c, d in cases)
        tab = max(prop(k, a, n, ang) / prop(kd, ad, n, ang) for n, ang in cols)
        row12 = max(prop(k, a, PML, ang) / prop(kd, ad, PML, ang) for ang in TABLE_ANGLES)
        row40 = max(prop(k, a, PML, ang, CPW_NEAR) / prop(kd, ad, PML, ang, CPW_NEAR)
                    for ang in TABLE_ANGLES)
        ok = tab <= 2.0
        summary[(k, a)] = dict(max=max(ratios), gmean=math.exp(np.mean(np.log(ratios))),
                               face=face, local=local, tab=tab, row12=row12, ok=ok)
        print(f'{label(k, a):16s} {max(ratios):9.2f} {summary[(k, a)]["gmean"]:10.2f} '
              f'{face:11.2e} {local:12.2e} {tab:9.2f} {row12:9.2f} {row40:12.2f}  '
              f'{"yes" if ok else "no"}')
    admissible = [s for s in summary if summary[s]['ok']]
    print(f'\nadmissible (every table entry <= 2x default): '
          f'{", ".join(label(*s).strip() for s in admissible)}')
    print('the setting each reading of "minimises the near-field reflection" picks, among the '
          'admissible:')
    for key, name in (('face', 'smallest worst R_face'), ('local', 'smallest worst R_local'),
                      ('gmean', 'smallest gmean ratio to default'),
                      ('max', 'smallest max ratio to default')):
        s = min(admissible, key=lambda s: summary[s][key])
        v = summary[s]
        print(f'  {name:32s}: {label(*s).strip():14s} near max {v["max"]:.2f}, gmean '
              f'{v["gmean"]:.2f}, worst R_face {v["face"]:.2e}, worst R_local '
              f'{v["local"]:.2e}, table max {v["tab"]:.2f}, row12 max {v["row12"]:.2f}')
    print(f'\n{len(results)} measurements, {seconds:.0f} s of worker time, '
          f'{time.time() - started:.0f} s wall')


if __name__ == '__main__':
    main()
