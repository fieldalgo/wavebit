"""Does averaging the painting route's medial-axis switches make its shape derivatives right?

The painting route builds its NVF normal projector from the nearest material interface
(``boundary.material_projector_fields``). The nearest interface switches across the medial axis,
between the intervals of one material pair and between pairs, and the direction turns there by up
to a right angle. A shape parameter moves the switch, so every normal-grid sample it crosses steps,
and autodiff of the sampled field sees none of the steps. Until 2026-10-15 the field took every
switch hard; since then it averages each one over half a sample step, averaging projectors rather
than vectors and weighting each candidate by its distance to the bisector with the nearest one.
This study measures what that bought and what it cost.

  1 windows    the derivative of total transmission integrated across four pixels of the 128 grid,
               against the change of total transmission across them (the test's instrument):
               the reflex vertex of a tilted L, a rotated rectangle written as its four corners,
               and a rotated rectangle painted under a second one, at shape_nvf_grid_size 64, 128
               and 256 (the L also at 512, and at orders (6, 6) and (9, 9)); and a second L, of
               permittivity 5.76 on glass and lit from it, the zeroth order across 4 nm.
  2 limits     total transmission against the normal grid, orders 4 and 8, on an axis-aligned L at
               normal incidence near a resonance (the scene the exact-pixel Li reference test
               uses), the tilted L, the painted pair, and two overlapping boxes of nearby
               materials; and an axis-aligned painted pair of unequal materials against Li's rule
               on the raster that draws it exactly, at the same orders.
  3 bands      what the averaged projector is: real, symmetric, of unit trace, positive
               semidefinite, rank two inside a switch and where unequal material pairs mix (so
               the rank-two fraction falls with the grid toward the mixing regions' share); on an
               interface the exact normal; and two overlapping boxes of one material against the
               box they make, which the averaged field tells apart where the outline passes
               from one box's region label to the other's (a split within one pair is no
               corner since 2026-10-16: ``boundary._continued_endpoints``).
  4 symmetric  a square sits on its own medial axis: central differences of total transmission
               in its size, h = 1e-3 to 1e-7.
  5 variants   the two averages rejected on the way, as instruments: signed vectors instead of
               projectors (the direction from an interface turns over across it), and the plain
               distance gap instead of the distance to the bisector (one-sided mixing beside a
               corner, a first-order bias in the grid spacing).
  6 cost       the field alone at the default grid 1024, best of two, and whole solves.
  7 before     with ``--before REV``: sections 1, 2, 4 and 6 on a ``git archive`` of REV.

Units: lengths in units of the period (1 x 1 cell), wavelength 1.3, except the L on glass (nm).
Every forward solve is NumPy complex128; every derivative is PyTorch autograd through
``rcwa.run``. Run from the repository root:

    python studies/rcwa/painted_switch_average/painted_switch_average.py --before v0.0.3 --only 1 2 3 4 5
    python studies/rcwa/painted_switch_average/painted_switch_average.py --before v0.0.3 --only 6

The two print the log committed as
``studies/rcwa/painted_switch_average/records/painted_switch_average.log``, in about an hour and
a quarter of an hour; the timing section runs alone so that both trees are timed close together.
The record archived ``3d20221b``, a commit that never left the machine it was made on, whose
``src/`` differs from release 0.0.3 only in its version string: ``v0.0.3`` reproduces it.
``--before`` extracts REV with ``git archive`` into a temporary directory and runs the comparable
sections there in a subprocess, by this same script; ``--only`` applies there too. The reading is
in ``docs/validation/rcwa/normal-field-known-limits.md`` section 1b.
"""

import argparse
import importlib
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

# Select the checkout before importing any wavebit module: the comparison in section 7 runs this
# script on an archived tree.
ROOT = str(Path(__file__).resolve().parents[3])
_root_parser = argparse.ArgumentParser(add_help=False)
_root_parser.add_argument('--root', default=ROOT)
_SELECTED_ROOT = Path(_root_parser.parse_known_args()[0].root).expanduser().resolve()
sys.path.insert(0, str(_SELECTED_ROOT / 'src'))
_selected = importlib.import_module('wavebit')
if Path(_selected.__file__).resolve() != (_SELECTED_ROOT / 'src' / 'wavebit' / '__init__.py').resolve():
    raise RuntimeError(f'wavebit was imported from {_selected.__file__}, not {_SELECTED_ROOT}; '
                       'run this study in a fresh Python process.')

import numpy as np                                                      # noqa: E402
from wavebit import Material, rcwa                                      # noqa: E402
import wavebit.rcwa._engine.boundary as B                               # noqa: E402

warnings.simplefilter('ignore')

L_SHAPE = ((-.35, -.35), (.3, -.35), (.3, -.1), (-.1, -.1), (-.1, .3), (-.35, .3))
_C, _S = np.cos(.2), np.sin(.2)
#: The L turned by 0.2 rad, so that no edge is axis-aligned (tests/rcwa/test_polygon_region.py).
L_TILTED = tuple(tuple(map(float, p)) for p in np.asarray(L_SHAPE) @ np.array([[_C, _S], [-_S, _C]]))


def _rectangle_corners(cx, cy, lx, ly, angle):
    c, s = np.cos(angle), np.sin(angle)
    return tuple((float(cx + c*u - s*v), float(cy + s*u + c*v))
                 for u, v in ((-lx/2, -ly/2), (lx/2, -ly/2), (lx/2, ly/2), (-lx/2, ly/2)))


RECTANGLE = _rectangle_corners(.07, -.04, .46, .31, .37)


def header(number, title):
    print(f'\n{number}. {title}', flush=True)


def region(geometry, epsilon=6.25):
    if not isinstance(geometry, (rcwa.Rectangle, rcwa.Polygon)):
        geometry = rcwa.Polygon(vertices=geometry)
    return rcwa.Region(geometry=geometry, material=Material(epsilon))


def oblique(*regions):
    """The operating point of tests/rcwa/test_polygon_region.py: s, theta 0.35, phi 0.6."""
    layer = rcwa.Layer(thickness=.4, background=Material(1.), regions=regions)
    return rcwa.Problem(stack=rcwa.Stack(period=(1., 1.), layers=(layer,), bottom=Material(2.25)),
                        excitation=rcwa.Excitation(wavelength=1.3, sources=(
                            rcwa.PlaneWave(theta=.35, phi=.6, polarization='s'),)))


def normal(*regions):
    """Normal incidence on the L near a resonance (test_the_default_solve_lands_on_...)."""
    layer = rcwa.Layer(thickness=.4, background=Material(1.), regions=regions)
    return rcwa.Problem(stack=rcwa.Stack(period=(1., 1.), layers=(layer,)),
                        excitation=rcwa.Excitation(wavelength=1.3, sources=(rcwa.PlaneWave(),)))


def pair(cx):
    """A rotated rectangle painted under a second one; ``cx`` moves the first."""
    return oblique(rcwa.Region(geometry=rcwa.Rectangle(size=(.5, .22), center=(cx, -.05), angle=.3),
                               material=Material(6.25)),
                   rcwa.Region(geometry=rcwa.Rectangle(size=(.3, .4), center=(.15, .12), angle=-.15),
                               material=Material(3.)))


SCENES = {
    'axis-aligned L, normal incidence': lambda: normal(region(L_SHAPE)),
    'tilted L': lambda: oblique(region(L_TILTED)),
    'painted pair': lambda: pair(-.08),
    'overlapping boxes 4 | 4.84': lambda: oblique(
        rcwa.Region(geometry=rcwa.Rectangle(size=(.4, .4), center=(-.1, 0.), angle=.2),
                    material=Material(4.)),
        rcwa.Region(geometry=rcwa.Rectangle(size=(.4, .4), center=(.15, .05), angle=.2),
                    material=Material(4.84))),
}


def numerics(order, grid):
    return rcwa.Numerics(orders=(order, order), shape_nvf_grid_size=grid)


def total(problem, order, grid):
    return float(np.sum(rcwa.run(problem, numerics=numerics(order, grid)).T))


# -------------------------------------------------------------------------- 1. windows

def _vertex_value_and_grad(vertices, vertex, numerics_):
    """``sum(T)`` and its derivative in one vertex's x, through the public lifecycle."""
    import torch

    def evaluate(x):
        v = torch.tensor(vertices, dtype=torch.float64)
        q = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        rows = [v[i] if i != vertex else torch.stack([q, v[i, 1]]) for i in range(len(vertices))]
        out = rcwa.run(oblique(region(torch.stack(rows))), numerics=numerics_, backend='torch').T.sum()
        out.backward()
        return float(out.detach()), float(q.grad)
    return evaluate


def _pair_value_and_grad(numerics_):
    import torch

    def evaluate(cx):
        q = torch.tensor(cx, dtype=torch.float64, requires_grad=True)
        out = rcwa.run(pair(q), numerics=numerics_, backend='torch').T.sum()
        out.backward()
        return float(out.detach()), float(q.grad)
    return evaluate


#: A second L, on glass and lit from it: 250 x 200 nm with 90 nm arms, turned by 0.3 rad, in a
#: 600 nm layer of permittivity 5.76 on a 400 nm square period, 633 nm, polarization (1, i),
#: the zeroth transmitted order. Its window is 4 nm of the reflex corner's x at 61 samples.
_TIO2_L = np.array([(0., 0.), (250., 0.), (250., 90.), (90., 90.), (90., 200.), (0., 200.)])
_TIO2_L = (_TIO2_L - (_TIO2_L.min(0) + _TIO2_L.max(0))/2) @ np.array([[np.cos(.3), np.sin(.3)],
                                                                      [-np.sin(.3), np.cos(.3)]])


def _glass_value_and_grad(numerics_):
    import torch

    def evaluate(x):
        v = torch.tensor(_TIO2_L, dtype=torch.float64)
        q = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        rows = [v[i] if i != 3 else torch.stack([q, v[i, 1]]) for i in range(len(_TIO2_L))]
        layer = rcwa.Layer(thickness=600., background=Material(1.), regions=(rcwa.Region(
            geometry=rcwa.Polygon(vertices=torch.stack(rows)), material=Material(5.76)),))
        problem = rcwa.Problem(
            stack=rcwa.Stack(period=(400., 400.), layers=(layer,), bottom=Material(2.25)),
            excitation=rcwa.Excitation(wavelength=633., sources=(
                rcwa.PlaneWave(side='bottom', polarization=(1, 1j)),)))
        order = numerics_.orders[0]
        out = rcwa.run(problem, numerics=numerics_, backend='torch').T[0, order, order]
        out.backward()
        return float(out.detach()), float(q.grad)
    return evaluate


WINDOWS = {
    'tilted L, reflex vertex x': (lambda n: _vertex_value_and_grad(L_TILTED, 3, n), L_TILTED[3][0]),
    'rectangle as corners, corner 3 x': (lambda n: _vertex_value_and_grad(RECTANGLE, 3, n),
                                         RECTANGLE[3][0]),
    'painted pair, cx': (_pair_value_and_grad, -.08),
    'L on glass, T00, reflex vertex x': (_glass_value_and_grad, float(_TIO2_L[3][0])),
}
#: Window shape per scene, (pixels, samples per pixel, pixel width): the L on glass is swept
#: across 4 nm in 61 samples, the others across four pixels of the 128 grid at eight a pixel.
WINDOW_SHAPE = {'L on glass, T00, reflex vertex x': (1, 60, 4.)}
#: (shape_nvf_grid_size, order) per window: every grid at orders (3, 3), and the L at higher
#: orders too, where the missing share was smaller (6.8% at (6, 6), 2.8% at (9, 9)).
WINDOW_RUNS = {'tilted L, reflex vertex x': ((64, 3), (128, 3), (256, 3), (512, 3), (128, 6), (128, 9)),
               'rectangle as corners, corner 3 x': ((64, 3), (128, 3), (256, 3)),
               'painted pair, cx': ((64, 3), (128, 3), (256, 3)),
               'L on glass, T00, reflex vertex x': ((256, 3), (256, 6))}


def window(value_and_grad, centre, pixels=4, per_pixel=8, pixel=1/128):
    """(integral of the derivative - change) / change across ``pixels`` pixels of ``pixel``,
    with the derivative taken on the loss's own abscissae, ``per_pixel`` to a pixel."""
    ps = centre + np.linspace(-pixels/2, pixels/2, int(pixels*per_pixel) + 1)*pixel
    values, grads = map(np.asarray, zip(*(value_and_grad(float(p)) for p in ps)))
    change = values[-1] - values[0]
    return (np.trapezoid(grads, ps) - change)/change, change


def windows():
    out = {}
    for name, (make, centre) in WINDOWS.items():
        for grid, order in WINDOW_RUNS[name]:
            pixels, per_pixel, pixel = WINDOW_SHAPE.get(name, (4, 8, 1/128))
            miss, change = window(make(numerics(order, grid)), centre, pixels, per_pixel, pixel)
            out[f'window/{name}/{grid}/{order}'] = np.asarray([miss, change])
    return out


def section_windows(before=None):
    header(1, 'windows -- (integral of the derivative - change) / change: sum(T) across four '
              'pixels of the 128 grid, or T00 across 4 nm for the L on glass')
    ours = windows()
    print(f'  {"":34s} {"grid":>5s} {"orders":>7s} {"this tree":>11s}'
          + (f' {"before":>11s}' if before else '') + f' {"change":>11s}')
    for key, (miss, change) in ours.items():
        _, name, grid, order = key.split('/')
        row = f'  {name:34s} {grid:>5s} {f"({order}, {order})":>7s} {miss:+11.2e}'
        if before:
            row += f' {before[key][0]:+11.2e}'
        print(row + f' {change:11.4e}')


# --------------------------------------------------------------------------- 2. limits

LADDER = (512, 1024, 2048)

#: An axis-aligned painted pair of unequal materials whose edges a 20 x 20 raster draws exactly:
#: A (6.25) painted first, B (3) over it. Li's rule on that raster is the reference.
PAIR_A = dict(x=(-.35, .25), y=(-.30, .05), eps=6.25)
PAIR_B = dict(x=(-.05, .35), y=(-.15, .30), eps=3.)
LI_ORDERS = (4, 6, 8, 10, 12)


def _rectangle(r):
    (x0, x1), (y0, y1) = r['x'], r['y']
    return rcwa.Region(geometry=rcwa.Rectangle(size=(x1 - x0, y1 - y0), center=((x0 + x1)/2, (y0 + y1)/2)),
                       material=Material(r['eps']))


def li_reference(order):
    c = (np.arange(20) + .5)/20 - .5
    X, Y = np.meshgrid(c, c)
    raster = np.ones((20, 20), complex)
    for r in (PAIR_A, PAIR_B):
        raster = np.where((X > r['x'][0]) & (X < r['x'][1]) & (Y > r['y'][0]) & (Y < r['y'][1]),
                          r['eps'], raster)
    layer = rcwa.Layer(thickness=.4, epsilon=raster)
    return float(np.sum(rcwa.run(normal_layer(layer), numerics=rcwa.Numerics(
        orders=(order, order), factorization='li', fourier_coefficients='integrated')).T))


def normal_layer(layer):
    return rcwa.Problem(stack=rcwa.Stack(period=(1., 1.), layers=(layer,)),
                        excitation=rcwa.Excitation(wavelength=1.3, sources=(rcwa.PlaneWave(),)))


def limits():
    out = {}
    for name, make in SCENES.items():
        grids = LADDER + ((4096,) if name.startswith('axis') else ())
        for order in (4, 8):
            for grid in grids:
                out[f'limit/{name}/{order}/{grid}'] = np.asarray(total(make(), order, grid))
    painted = normal_layer(rcwa.Layer(thickness=.4, background=Material(1.),
                                      regions=(_rectangle(PAIR_A), _rectangle(PAIR_B))))
    for order in LI_ORDERS:
        out[f'li/{order}'] = np.asarray(total(painted, order, 1024))
    return out


def section_limits(before=None):
    header(2, 'limits -- sum(T) against shape_nvf_grid_size; the default grid is 1024')
    ours = limits()
    print('  axis-aligned painted pair (6.25 under 3), normal incidence, grid 1024, against Li on '
          'the exact 20 x 20 raster at the same orders (Li at (16, 16): '
          f'{li_reference(16):.8f})')
    for order in LI_ORDERS:
        li = li_reference(order)
        mine = float(ours[f'li/{order}'])
        row = f'    orders ({order:2d}, {order:2d})  Li {li:.8f}  this tree - Li {mine - li:+.2e}'
        if before:
            theirs = float(before[f'li/{order}'])
            row += f'   before - Li {theirs - li:+.2e}   this tree - before {mine - theirs:+.2e}'
        print(row)
    for name in SCENES:
        for order in (4, 8):
            keys = [k for k in ours if k.startswith(f'limit/{name}/{order}/')]
            print(f'  {name}, orders ({order}, {order})')
            for key in keys:
                grid = key.rsplit('/', 1)[1]
                row = f'    grid {grid:>5s}  this tree {float(ours[key]):.10f}'
                if before:
                    row += (f'   before {float(before[key]):.10f}   difference '
                            f'{float(ours[key]) - float(before[key]):+.2e}')
                print(row)


# ---------------------------------------------------------------------------- 3. bands

def _painted_inputs(shapes, lattice):
    from wavebit._common.geometry.interfaces import replay_interfaces, interface_contrasts
    verts = [B.shape_vertices(np, s, lattice) for s in shapes]
    recs, regions, transform, plan = B.plan_layer(shapes, lattice)
    vals = B._clipper.eval_table(np, recs, verts)
    regs = [(s['material'], [B._clipper.loop_verts(np, loop, vals) for loop in reg])
            for s, reg in zip(shapes, regions)]
    return (replay_interfaces(np, regs, 1., lattice, transform, plan),
            interface_contrasts(np, regs, 1., plan, np.float64), plan, transform)


def section_bands():
    from wavebit.lattice import as_lattice
    header(3, 'bands -- the averaged projector of the painted pair (section 1)')
    lattice = as_lattice(period=(1., 1.))
    shapes = [dict(type='rectangle', cx=-.08, cy=-.05, lx=.5, ly=.22, angle=.3, material=6.25),
              dict(type='rectangle', cx=.15, cy=.12, lx=.3, ly=.4, angle=-.15, material=3.)]
    data, contrasts, plan, transform = _painted_inputs(shapes, lattice)
    previous = None
    for grid in (64, 128, 256, 512, 1024):
        xx, xy, yy = (np.asarray(f) for f in B.material_projector_fields(
            np, (grid, grid), lattice, np.float64, data, contrasts, plan, transform))
        det = xx*yy - xy*xy
        two = float(np.mean(det > 1e-9))
        print(f'  grid {grid:5d}: |trace - 1| {np.abs(xx + yy - 1).max():.1e}, min eigenvalue '
              f'{float(np.min((xx + yy)/2 - np.sqrt(((xx - yy)/2)**2 + xy*xy))):+.1e}, '
              f'rank two (det > 1e-9) on {two:.4f} of the samples'
              + (f', {previous/two:.2f}x fewer than at half the grid' if previous else ''))
        previous = two
    starts, edges, _, active = data
    midpoints = starts[active] + edges[active]/2
    normals = edges[active][:, ::-1]*[1., -1.]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    original = B.sample_grid
    B.sample_grid = lambda *args: (midpoints[:, 0][None, :], midpoints[:, 1][None, :])
    try:
        fields = B.material_projector_fields(np, (1, len(midpoints)), lattice, np.float64, data,
                                             contrasts, plan, transform)
    finally:
        B.sample_grid = original
    expected = np.stack([normals[:, 0]**2, normals[:, 0]*normals[:, 1], normals[:, 1]**2])
    print(f'  on the {len(midpoints)} interval midpoints: the exact normal dyad to '
          f'{np.abs(np.asarray(fields)[:, 0] - expected).max():.1e}')
    # Two overlapping boxes of one material are the box they make: the outline is the same,
    # under two region labels, so it passes from one box's material pair to the other's. (The
    # record's run, 1.3e-3 / 1.2e-4 / 8.1e-6, also counted every split *within* a pair as a
    # corner; since 2026-10-16 it reads 4.3e-4 / 4.3e-5 / 3.0e-6 -- records/, addendum.)
    cell = ((700., 0.), (0., 700.))
    left, right, union = (dict(type='rectangle', cx=cx, cy=0., lx=lx, ly=300., angle=0., material=4.)
                          for cx, lx in ((-100., 300.), (100., 300.), (0., 500.)))
    import array_api_compat.numpy as xnp
    for grid in (64, 256, 1024):
        mats = [B.to_conv_mat_boundary_layer(xnp, shapes, 1., 2, 2, cell, xnp.complex128,
                                             shape_nvf_grid_size=grid)
                for shapes in ([left, right], [union])]
        gap = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(*mats))
        print(f'  two overlapping boxes of one material against the box they make, orders (2, 2), '
              f'grid {grid:4d}: matrices {gap:.1e}')


# ------------------------------------------------------------------------ 4. symmetric

SYMMETRIC_STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


def symmetric():
    def value(a):
        return total(oblique(region(((-a, -a), (a, -a), (a, a), (-a, a)))), 4, 128)
    return {f'symmetric/{h:.0e}': np.asarray((value(.25 + h) - value(.25 - h))/(2*h))
            for h in SYMMETRIC_STEPS}


def section_symmetric(before=None):
    header(4, 'symmetric -- a 0.5 square at the cell centre, its diagonals through the samples of '
              'the 128 grid: (T(a+h) - T(a-h)) / 2h in its half-width a, orders (4, 4)')
    ours = symmetric()
    for key, value in ours.items():
        row = f'  h {key.split("/")[1]}: this tree {float(value):+.6e}'
        if before:
            row += f'   before {float(before[key]):+.6e}'
        print(row)


# ------------------------------------------------------------------------- 5. variants

def variant_fields(signed, gap):
    """``material_projector_fields`` with one of the two rejected averages, as an instrument.

    ``signed``: average the unit vectors (the direction from the interface toward the sample)
    and form one projector from the sum, as the field before 2026-10-15 did with its hard
    choices. ``gap``: weight a candidate by its distance gap to the nearest, not by its distance
    to their bisector. Everything else is the shipped construction.
    """
    from wavebit.rcwa._engine.parametric import _BLEND_EXPONENT

    def fields(xp, grid_shape, lattice, dtype, segment_data, contrasts, interface_plan,
               transform=None, chunk=None, candidate_fields=None):
        A, E, L2, active = segment_data
        ids = [i for group in interface_plan.groups for i in group]
        groups = {}
        for image_index, i in enumerate(ids):
            piece = interface_plan.pieces[i]
            key = tuple(sorted((piece.left, piece.right)))
            if key[0] != key[1]:
                groups.setdefault(key, []).append(image_index)
        A, E, L2 = B.periodic_segments(xp, A, E, L2, lattice, transform,
                                       interface_plan.images, interface_plan.groups)
        extended = xp.asarray(ids, dtype=xp.int64)
        active, contrasts = active[extended], contrasts[extended]
        candidates = [(A[xp.asarray(ix)], E[xp.asarray(ix)], L2[xp.asarray(ix)], contrasts[ix[0]],
                       xp.sum(E[xp.asarray(ix)]**2, axis=1) > 0.) for ix in groups.values()]
        pts = B._folded_sample_points(xp, grid_shape, lattice, dtype, transform)
        scale = B._cell_scale(xp, lattice, dtype)
        threshold2 = B._on_interface_threshold2(xp, lattice, dtype)
        unit2 = 2.**(2*np.round(np.log2(float(scale))))
        strengths = xp.stack([c for _, _, _, c, _ in candidates])**_BLEND_EXPONENT
        width = B._SWITCH_WIDTH*B._sample_spacing(xp, grid_shape, lattice, dtype)

        def root(d2):
            off = d2 > threshold2
            return xp.where(off, xp.sqrt(xp.where(off, d2, xp.ones_like(d2))), xp.zeros_like(d2))

        def distance(gap_, turn):
            if gap:
                return gap_
            floor = xp.asarray(B._SAME_DIRECTION**2, dtype=dtype)
            return gap_/xp.sqrt(xp.maximum(xp.sum(turn*turn, axis=-1), floor))

        def average(weight, u):
            if signed:
                return xp.sum(weight[..., None]*u, axis=-2)
            ux, uy = u[..., 0], u[..., 1]
            return xp.stack([xp.sum(weight*ux*ux, axis=-1), xp.sum(weight*ux*uy, axis=-1),
                             xp.sum(weight*uy*uy, axis=-1)], axis=-1)

        def evaluate(p):
            rows = xp.arange(p.shape[0])
            d2s, towards, means = [], [], []
            for a, e, l2, _, valid in candidates:
                dist2, foot = B._seg_dist(xp, p, a, e, l2)
                dist2 = xp.where(valid[None, :], dist2, (1e3*scale)**2)
                nearest = xp.argmin(dist2, axis=1)
                off = dist2 > threshold2
                length = root(dist2)
                n_edge = xp.stack([e[:, 1], -e[:, 0]], axis=-1)
                n_edge = n_edge/xp.sqrt(xp.sum(n_edge**2, axis=1))[:, None]
                u = xp.where(off[:, :, None],
                             (p[:, None, :] - foot)/xp.where(off, length, xp.ones_like(length))[:, :, None],
                             n_edge[None])
                toward = u[rows, nearest]
                weight = xp.where((dist2[rows, nearest] <= threshold2)[:, None], xp.astype(~off, dtype),
                                  xp.exp(-distance(length - length[rows, nearest][:, None],
                                                   u - toward[:, None, :])/width))
                weight = xp.where(valid[None, :], weight, xp.zeros_like(weight))
                d2s.append(dist2[rows, nearest])
                towards.append(toward)
                means.append(average(weight, u)/xp.sum(weight, axis=1)[:, None])
            d2 = xp.stack(d2s, axis=1)
            minimum = xp.min(d2, axis=1)
            on = minimum <= threshold2
            ratio = xp.where(on, xp.ones_like(minimum), minimum/unit2)[:, None]/xp.where(
                (d2 > 0.) & ~on[:, None], d2/unit2, xp.ones_like(d2))
            influence = strengths[None, :]*xp.where(on[:, None], xp.astype(d2 <= threshold2, dtype),
                                                    ratio**(_BLEND_EXPONENT/2))
            order = xp.argsort(-influence, axis=1, stable=True)
            ranked = influence[rows[:, None], order]
            reach = root(d2)[rows[:, None], order]
            toward = xp.stack(towards, axis=1)[rows[:, None], order]
            mean = xp.stack(means, axis=1)[rows[:, None], order]
            step = ranked - xp.concat([ranked[:, 1:], xp.zeros_like(ranked[:, :1])], axis=1)
            best, best_toward = reach[:, 0], toward[:, 0]
            mixed = xp.zeros_like(mean[:, 0])
            for j in range(len(candidates)):
                if j:
                    nearer = reach[:, j] < best
                    best = xp.where(nearer, reach[:, j], best)
                    best_toward = xp.where(nearer[:, None], toward[:, j], best_toward)
                share = xp.exp(-distance(reach[:, :j+1] - best[:, None],
                                         toward[:, :j+1] - best_toward[:, None, :])/width)
                mixed = mixed + step[:, j][:, None]*xp.sum(share[:, :, None]*mean[:, :j+1], axis=1) \
                    / xp.sum(share, axis=1)[:, None]
            if signed:
                sx, sy = mixed[:, 0], mixed[:, 1]
                norm2 = sx*sx + sy*sy
                return xp.stack([sx*sx/norm2, sx*sy/norm2, sy*sy/norm2], axis=-1)
            return mixed/ranked[:, :1]

        chunk_ = max(1, 1_500_000//max(len(A), 1))
        result = xp.concat([evaluate(pts[lo:lo + chunk_]) for lo in range(0, pts.shape[0], chunk_)])
        return tuple(xp.reshape(result[:, i], grid_shape) for i in range(3))
    return fields


def section_variants():
    header(5, 'variants -- the two averages rejected on the way, on the painted pair window '
              '(grid 128) and the axis-aligned L at orders (4, 4)')
    shipped = B.material_projector_fields
    name = 'axis-aligned L, normal incidence'
    for label, signed, gap in (('signed vectors, bisector distance', True, False),
                               ('projectors, plain distance gap', False, True),
                               ('projectors, bisector distance (shipped)', False, False)):
        B.material_projector_fields = variant_fields(signed, gap)
        try:
            miss, _ = window(_pair_value_and_grad(numerics(3, 128)), -.08)
            values = [total(SCENES[name](), 4, grid) for grid in (512, 1024, 2048)]
        finally:
            B.material_projector_fields = shipped
        print(f'  {label:40s} pair window {miss:+.2e};  L sum(T) at grids 512 / 1024 / 2048: '
              + ' / '.join(f'{v:.7f}' for v in values))


# ----------------------------------------------------------------------------- 6. cost

def cost():
    from wavebit.lattice import as_lattice
    lattice = as_lattice(period=(1., 1.))
    painted = {
        'tilted L': [dict(type='polygon', vertices=L_TILTED, material=6.25)],
        'painted pair': [dict(type='rectangle', cx=-.08, cy=-.05, lx=.5, ly=.22, angle=.3, material=6.25),
                         dict(type='rectangle', cx=.15, cy=.12, lx=.3, ly=.4, angle=-.15, material=3.)],
        'disc over a rectangle': [dict(type='rectangle', cx=-.05, cy=0., lx=.5, ly=.3, angle=.1, material=4.),
                                  dict(type='circle', cx=.15, cy=.05, lx=.34, ly=.34, material=9.)],
    }
    out = {}
    for name, shapes in painted.items():
        data, contrasts, plan, transform = _painted_inputs(shapes, lattice)
        times = []
        for _ in range(2):
            start = time.perf_counter()
            B.material_projector_fields(np, (1024, 1024), lattice, np.float64, data, contrasts,
                                        plan, transform)
            times.append(time.perf_counter() - start)
        out[f'cost/field/{name}'] = np.asarray([min(times), sum(len(g) for g in plan.groups)])
    for name, make in (('tilted L', SCENES['tilted L']), ('painted pair', SCENES['painted pair'])):
        for order in (4, 8, 12):
            times = []
            for _ in range(2):
                start = time.perf_counter()
                total(make(), order, 1024)
                times.append(time.perf_counter() - start)
            out[f'cost/solve/{name}/{order}'] = np.asarray(min(times))
    return out


def section_cost(before=None):
    header(6, 'cost -- seconds, NumPy, this machine (ratios carry; absolute times move between runs)')
    ours = cost()
    for key, value in ours.items():
        parts = key.split('/')
        label = (f'field at grid 1024, {parts[2]} ({int(value[1])} interval images)'
                 if parts[1] == 'field' else f'whole solve, {parts[2]}, orders ({parts[3]}, {parts[3]})')
        mine = float(np.ravel(value)[0])
        row = f'  {label:60s} this tree {mine:7.3f}'
        if before:
            theirs = float(np.ravel(before[key])[0])
            row += f'   before {theirs:7.3f}   {mine/theirs:5.2f}x'
        print(row)


# ---------------------------------------------------------------------------- driver

#: What section 7 runs on the archived tree, by section number.
COMPARABLE = {1: windows, 2: limits, 4: symmetric, 6: cost}


def harness(only=None):
    out = {}
    for number, measure in COMPARABLE.items():
        if not only or number in only:
            out.update(measure())
    return out


def before(rev, only=None):
    with tempfile.TemporaryDirectory() as tmp:
        tree = os.path.join(tmp, 'tree')
        os.mkdir(tree)
        archive = subprocess.run(['git', '-C', ROOT, 'archive', rev], check=True,
                                 capture_output=True).stdout
        subprocess.run(['tar', '-x', '-C', tree], input=archive, check=True)
        dump = os.path.join(tmp, 'before.npz')
        subprocess.run([sys.executable, os.path.abspath(__file__), '--dump', dump, '--root', tree]
                       + (['--only', *map(str, only)] if only else []),
                       check=True, env=dict(os.environ, PYTHONWARNINGS='ignore'))
        return dict(np.load(dump))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--before', metavar='REV', help='section 7: compare with this commit')
    parser.add_argument('--only', type=int, nargs='*', help='run these sections only')
    parser.add_argument('--dump', help=argparse.SUPPRESS)
    parser.add_argument('--root', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.dump:
        np.savez(args.dump, **harness(args.only))
        return
    theirs = before(args.before, args.only) if args.before else None
    import torch
    print('The painting route averages its switches -- '
          'studies/rcwa/painted_switch_average/painted_switch_average.py')
    print(f'numpy {np.__version__}, torch {torch.__version__}; forward solves NumPy complex128, '
          f'derivatives PyTorch autograd; before: '
          + (f'a git archive of {args.before}' if args.before else 'skipped (no --before)'))
    sections = {1: lambda: section_windows(theirs), 2: lambda: section_limits(theirs),
                3: section_bands, 4: lambda: section_symmetric(theirs), 5: section_variants,
                6: lambda: section_cost(theirs)}
    for number, run in sections.items():
        if not args.only or number in args.only:
            run()


if __name__ == '__main__':
    main()
