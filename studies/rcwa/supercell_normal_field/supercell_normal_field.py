"""Why does a supercell of identical pillars answer differently from the primitive cell?

A hexagonal array of TiO2 pillars (n = 2.43, constant) on fused silica (n = 1.46), pitch 350 nm,
height 600 nm, diameter 150 nm, 532 nm, normal incidence from air, s. The primitive cell holds one
pillar on ``(a1, a2)``; the supercell ``(N*a1, a2)`` holds N copies of it at ``(k - (N-1)/2)*a1``.
The supercell keeps ``|m| <= N*M, |n| <= M``, which contains the primitive cell's ``|m|, |n| <= M``
exactly, so both cells hold the same structure and the same primitive orders. Under the default
settings their zero-order transmission still differs at the level of the truncation error.

The parts, in the order of the argument:

  rule        N = 4 against the primitive cell at M = 3 and 4, under the default NVF and under
              ``factorization='laurent'``. Laurent's rule reads no normal field.
  field       the default (``'per_shape'``) normal field of each cell, evaluated at the same
              physical points by the builder the solve uses, banded by the distance from the
              nearest pillar boundary. The inputs are the ones the solve hands that builder: the
              first shape at the cell centre and the others at their offsets from it.
  grid        the comparison at M = 3 for ``shape_nvf_grid_size`` 512, 1024 and 2048.
  convergence N = 2 against the primitive cell at M = 3 to 9, both against the primitive cell at
              M = 14.
  interfaces  the comparison at M = 3 and 4 with ``shape_normal_method='interfaces'``
              (experimental; about ten minutes a pair on four cores).

Run from the repository root:

    python studies/rcwa/supercell_normal_field/supercell_normal_field.py
    python studies/rcwa/supercell_normal_field/supercell_normal_field.py --part interfaces

The two printed logs, concatenated in that order, are
``studies/rcwa/supercell_normal_field/records/supercell_normal_field.log``. The write-up is §10 of
``docs/validation/rcwa/normal-field-known-limits.md``.
"""

import argparse
import os
import platform
import time
import warnings
from importlib import metadata

import numpy as np

from wavebit import Lattice, Material, Units, eps_from_refractive_index, rcwa
from wavebit.rcwa._engine import parametric

PITCH = 350.0
HEIGHT = 600.0
WAVELENGTH = 532.0
DIAMETER = 150.0
A1 = np.array([PITCH*np.sqrt(3)/2, -PITCH/2])
A2 = np.array([0.0, PITCH])
PILLAR = Material(epsilon=eps_from_refractive_index(2.43))
SUBSTRATE = Material(epsilon=eps_from_refractive_index(1.46))
FIELD_POINTS = 241           # per side, over one primitive cell around a pillar
BANDS = ((0, 1), (1, 5), (5, 20), (20, 60), (60, 200))   # nm from the nearest pillar boundary


def environment():
    print(f'python {platform.python_version()}  {platform.system()} {platform.machine()}  '
          f'logical CPUs {os.cpu_count()}')
    try:
        with open('/proc/cpuinfo', encoding='utf-8') as handle:
            model = next((line.split(':', 1)[1].strip() for line in handle
                          if line.startswith('model name')), None)
    except OSError:
        model = None
    if model:
        print(f'cpu {model}')
    print('  '.join(f'{name} {metadata.version(name)}' for name in ('wavebit', 'numpy', 'scipy')))
    threads = {key: os.environ[key] for key in
               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS') if key in os.environ}
    print(f'thread settings {threads or "library defaults"}')


def problem(count):
    regions = tuple(
        rcwa.Region(geometry=rcwa.Circle(radius=DIAMETER/2, center=tuple((k-(count-1)/2)*A1)),
                    material=PILLAR)
        for k in range(count))
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(lattice=Lattice.from_columns(np.column_stack([count*A1, A2])),
                         bottom=SUBSTRATE, layers=(rcwa.Layer(thickness=HEIGHT, regions=regions),)),
        excitation=rcwa.Excitation(wavelength=WAVELENGTH,
                                   sources=(rcwa.PlaneWave(polarization='s'),)),
    )


def solve(count, order, **numerics):
    result = rcwa.run(problem(count), numerics=rcwa.Numerics(orders=(count*order, order), **numerics))
    reflected, transmitted = float(result.R.sum()), float(result.T.sum())
    np.testing.assert_allclose(reflected+transmitted, 1.0, atol=1e-9, rtol=0)
    zero = complex(np.asarray(result.transmission.order(0, 0).amplitude_ratio_s).reshape(-1)[0])
    return transmitted, zero


def compare(count, order, **numerics):
    """Supercell minus primitive cell: transmittance, and the phase of the zero order."""
    started = time.perf_counter()
    t1, a1 = solve(1, order, **numerics)
    tn, an = solve(count, order, **numerics)
    return tn-t1, float(np.angle(an/a1)), time.perf_counter()-started


def part_rule():
    print('\nrule: N = 4 against the primitive cell, d = 150 nm')
    print('    factorization   M   T(N=4) - T(1)   phase(N=4) - phase(1) (rad)   time (s)')
    for label, numerics in (('default (nvf)', {}), ('laurent', {'factorization': 'laurent'})):
        for order in (3, 4):
            dt, dphase, seconds = compare(4, order, **numerics)
            print(f'    {label:14s}  {order}   {dt:+.3e}        {dphase:+.3e}                  {seconds:6.1f}',
                  flush=True)


def _field(shapes, lattice, x, y):
    plan = parametric.plan_gamma(shapes, lattice)
    slots = parametric._material_slots(np, shapes, np.float64)
    nx, ny = parametric._piece_field(np, x, y, shapes, plan, slots, lattice, np.float64)
    return np.asarray(nx), np.asarray(ny), len(plan.pieces)


def _angle(ax, ay, bx, by):
    """Angle between two normal directions in degrees; a normal and its opposite are one direction."""
    return np.degrees(np.arccos(np.clip(np.abs(ax*bx+ay*by), 0.0, 1.0)))


def part_field():
    count = 4
    circle = dict(type='circle', lx=DIAMETER, material=PILLAR)
    primitive = [dict(circle, cx=0.0, cy=0.0)]
    supercell = [dict(circle, cx=float(k*A1[0]), cy=float(k*A1[1])) for k in range(count)]
    lattice_1 = Lattice.from_columns(np.column_stack([A1, A2]))
    lattice_n = Lattice.from_columns(np.column_stack([count*A1, A2]))

    s = (np.arange(FIELD_POINTS)+0.5)/FIELD_POINTS-0.5
    grid_s, grid_t = np.meshgrid(s, s, indexing='ij')
    offset = grid_s[..., None]*A1+grid_t[..., None]*A2       # one primitive cell about a pillar
    radius = np.linalg.norm(offset, axis=-1)
    # Distances to the nearest and second-nearest pillar boundary over the lattice.
    distances = np.sort(np.stack([np.abs(np.linalg.norm(offset-(i*A1+j*A2), axis=-1)-DIAMETER/2)
                                  for i in range(-2, 3) for j in range(-2, 3)]), axis=0)
    nearest, second = distances[0], distances[1]

    # Internal frame: a shape's (cx, cy) is measured from the cell centre 0.5*(a1 + a2), so the
    # primitive pillar sits at 0.5*(a1 + a2) and supercell pillar k at 0.5*(N*a1 + a2) + k*a1.
    nx1, ny1, pieces_1 = _field(primitive, lattice_1, *(0.5*(A1+A2)+offset).transpose(2, 0, 1))
    angles = []
    for k in range(count):
        centre = 0.5*(count*A1+A2)+((k+count//2) % count-count//2)*A1
        nxk, nyk, pieces_n = _field(supercell, lattice_n, *(centre+offset).transpose(2, 0, 1))
        angles.append(_angle(nx1, ny1, nxk, nyk))

    # The primitive cell's one piece should read the radial normal of the nearest pillar image.
    images = [i*A1+j*A2 for i in range(-2, 3) for j in range(-2, 3)]
    relative = np.stack([offset-image for image in images])
    closest = np.linalg.norm(relative, axis=-1).argmin(axis=0)
    toward = np.take_along_axis(relative, closest[None, ..., None], axis=0)[0]
    length = np.linalg.norm(toward, axis=-1)
    radial = toward/np.where(length > 0, length, 1.0)[..., None]
    off_radial = _angle(nx1, ny1, radial[..., 0], radial[..., 1]) > 1.0
    tie = np.abs(second-nearest) < 1.0
    print(f'\nfield: the default normal field, {FIELD_POINTS} x {FIELD_POINTS} points over one '
          f'primitive cell about a pillar')
    print(f'    pieces in the blend: primitive cell {pieces_1}, supercell N = {count}: {pieces_n}')
    print(f'    primitive field against the radial normal of the nearest pillar image: {off_radial.sum()} '
          f'of {off_radial.size} points differ by more than 1 deg; {(off_radial & tie).sum()} of them '
          f'lie within 1 nm of equidistant from two pillars, {(off_radial & (length < 1)).sum()} '
          'at a centre')
    print("    angle between the two cells' fields (deg), by distance from the nearest pillar boundary")
    print('    pillar   ' + '   '.join(f'{lo:>3}-{hi:<3} nm max / mean' for lo, hi in BANDS))
    for k, angle in enumerate(angles):
        cells = []
        for lo, hi in BANDS:
            band = (nearest >= lo) & (nearest < hi)
            cells.append(f'{angle[band].max():9.3f} / {angle[band].mean():6.3f}')
        print(f'      {k}      ' + '   '.join(cells))
    # arccos near 1 turns a rounding of the dot product into ~1e-6 deg, so this bound is 1e-5.
    spread = max(float(np.abs(angle-angles[0]).max()) for angle in angles[1:])
    assert spread < 1e-5
    print(f'    (the four supercell pillars give the same angles to {spread:.1e} deg)')


def part_grid():
    print('\ngrid: N = 4 against the primitive cell at M = 3, shape_nvf_grid_size')
    print('    grid    T(N=4) - T(1)   phase(N=4) - phase(1) (rad)   time (s)')
    for grid in (512, 1024, 2048):
        dt, dphase, seconds = compare(4, 3, shape_nvf_grid_size=grid)
        print(f'    {grid:5d}   {dt:+.3e}      {dphase:+.3e}                  {seconds:6.1f}', flush=True)


def part_convergence():
    print('\nconvergence: N = 2 against the primitive cell; reference T(1) at M = 14')
    started = time.perf_counter()
    reference, _ = solve(1, 14)
    print(f'    reference T = {reference:.8f}  ({time.perf_counter()-started:.0f} s)')
    print('     M   orders (1 / N=2)   T(1) - ref    T(N=2) - ref   T(N=2) - T(1)   phase(N=2) - phase(1) (rad)')
    for order in range(3, 10):
        t1, a1 = solve(1, order)
        t2, a2 = solve(2, order)
        print(f'    {order:2d}     {(2*order+1)**2:4d} / {(4*order+1)*(2*order+1):4d}      '
              f'{t1-reference:+.2e}     {t2-reference:+.2e}      {t2-t1:+.2e}        '
              f'{np.angle(a2/a1):+.2e}', flush=True)


def part_interfaces():
    print('\ninterfaces: N = 4 against the primitive cell, shape_normal_method="interfaces" (experimental)')
    print('    M   T(N=4) - T(1)   phase(N=4) - phase(1) (rad)   time (s)')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rcwa.diagnostics.ExperimentalWarning)
        for order in (3, 4):
            dt, dphase, seconds = compare(4, order, shape_normal_method='interfaces')
            print(f'    {order}   {dt:+.3e}      {dphase:+.3e}                  {seconds:6.1f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--part', choices=('main', 'interfaces'), default='main')
    arguments = parser.parse_args()
    environment()
    if arguments.part == 'interfaces':
        part_interfaces()
    else:
        part_rule()
        part_field()
        part_grid()
        part_convergence()
