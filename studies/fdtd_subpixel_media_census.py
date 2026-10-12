"""How often does a voxel hold three media, or a creased interface, in the scenes stage 4 grades?

Stage 4's plan runs this (S7) before the extruded bodies ship (C4), because two different
populations were once conflated and the conflation nearly cost the feature: a voxel where
**three or more media** meet has no answer from a two-phase mixing rule and must be refused
(or clipped exactly, C10's polytope clip), while a voxel holding a **creased two-media**
interface -- a box corner, a prism edge, a cylinder rim, all inside one body -- is still two
phases, and the kernel answers its fill exactly. Refusing the second population would refuse
every prism edge and every rotated box corner. So they are counted separately, and a third
population beside them: voxels where **two structures' surfaces** meet but only two media are
present (a flush joint), which ``src/wavebit/fd/_lower/checks.py``'s door 5 refuses today.

Method: every E-component voxel within reach of any surface, at 20/40/80 cells per unit
length, over eight scenes -- a sphere, a cylinder, an axis-aligned and a 30-degree box, a
hexagonal prism, a cylinder pillar standing flush on a substrate of another material, and two
sphere dimers. A voxel is *cut* where a body's exact fill lies strictly between 0 and 1. It is
*creased* where one body's surface there is not one plane: for a polygonal body, where the exact
fill differs from its tangent-plane fill; for a cylinder, where the side surface cuts the
voxel's transverse square and a cap plane lies inside its axial interval (the rim). The media in
a voxel are the distinct materials whose fill is nonzero, plus the background where the fills do
not sum to one. The kernel's own ``occupancy``, ``surface``, ``plane`` and
``clipped_disc_area`` answer every question; nothing here is a second geometry.

Run from the repository root: ``python studies/fdtd_subpixel_media_census.py``; it prints the
table and writes ``studies/results/fdtd_subpixel_media_census.log``. About six seconds.
"""
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np                                                               # noqa: E402
from wavebit.fd._kernel.grid import OFFSETS                                      # noqa: E402
from wavebit.fd._shapes import (Box, Cylinder, Prism, Sphere, clipped_disc_area,  # noqa: E402
                                occupancy)

#: A fill this close to 0 or 1 is not a cut: the kernel's own round-off is below 1e-14.
TOL = 1e-9
#: A voxel's half diagonal, in cells: no surface farther than this can cut it.
REACH = math.sqrt(3) / 2
RESOLUTIONS = (20, 40, 80)


def points_for(lo, hi, dx, c):
    """E-component ``c``'s Yee points over ``[lo, hi]`` with two cells of margin."""
    axes = []
    for a in range(3):
        k0, k1 = math.floor(lo[a] / dx) - 2, math.ceil(hi[a] / dx) + 2
        axes.append((np.arange(k0, k1) + 0.5 * OFFSETS['E'][c][a]) * dx)
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def creased(body, points, dx, fill):
    """Voxels where one body's surface is not one plane (see the module docstring)."""
    if isinstance(body, Sphere):
        return np.zeros(len(points), bool)
    if isinstance(body, Cylinder):
        u, v, k = {'x': (1, 2, 0), 'y': (2, 0, 1), 'z': (0, 1, 2)}[body.axis]
        area = clipped_disc_area(body.center[u], body.center[v], body.radius,
                                 points[:, u] - dx / 2, points[:, u] + dx / 2,
                                 points[:, v] - dx / 2, points[:, v] + dx / 2, np)
        side = (area > TOL * dx * dx) & (area < (1 - TOL) * dx * dx)
        s = points[:, k]
        cap = np.zeros(len(points), bool)
        for plane in (body.center[k] - body.height / 2, body.center[k] + body.height / 2):
            cap |= (plane > s - dx / 2) & (plane < s + dx / 2)
        return side & cap
    return np.abs(np.asarray(fill) - np.asarray(body.plane(points, dx)[0])) > 1e-7


def census(structures, dx, lo, hi):
    """Counts over the E-component voxels near any surface of ``structures`` (body, medium)."""
    out = dict(interface=0, creased=0, three=0, joint=0)
    for c in range(3):
        points = points_for(lo, hi, dx, c)
        near = np.zeros(len(points), bool)
        for body, _ in structures:
            near |= np.abs(np.asarray(body.surface(points)[0])) <= REACH * dx
        points = points[near]
        fills, cuts = [], []
        for body, _ in structures:
            fill = np.asarray(occupancy(body, points, dx)[0])
            cut = (fill > TOL) & (fill < 1 - TOL)
            fills.append(fill)
            cuts.append(cut)
            out['creased'] += int((cut & creased(body, points, dx, fill)).sum())
        fills, cuts = np.array(fills), np.array(cuts)
        any_cut = cuts.any(axis=0)
        out['interface'] += int(any_cut.sum())
        kinds = (fills.sum(axis=0) < 1 - TOL).astype(int)            # the background
        present = {}
        for fill, (_, medium) in zip(fills, structures):
            present[medium] = present.get(medium, np.zeros(len(points), bool)) | (fill > TOL)
        for mask in present.values():
            kinds += mask
        out['three'] += int(((kinds >= 3) & any_cut).sum())
        out['joint'] += int(((cuts.sum(axis=0) >= 2) & (kinds == 2)).sum())
    return out


def hexagon(radius):
    return tuple((radius * math.cos(math.pi / 3 * i), radius * math.sin(math.pi / 3 * i))
                 for i in range(6))


CENTRE = (0.013, -0.007, 0.021)
NEAR = ((-.3, -.3, -.3), (.3, .3, .3))
SCENES = {
    'sphere r.25': ([(Sphere(0.25, CENTRE), 'A')], *NEAR),
    'cylinder r.25 h.5': ([(Cylinder(0.25, 0.5, 'z', CENTRE), 'A')], *NEAR),
    'box .4x.3x.5': ([(Box((0.4, 0.3, 0.5), CENTRE), 'A')], *NEAR),
    'box .4x.3x.5 turned 30 deg': ([(Box((0.4, 0.3, 0.5), CENTRE, angle=math.radians(30)), 'A')],
                                   *NEAR),
    'hexagonal prism R.25 h.5': ([(Prism(hexagon(0.25), 0.5, 'z', CENTRE), 'A')], *NEAR),
    'pillar flush on a substrate': ([(Box((40., 40., 20.), (0., 0., -10.)), 'S'),
                                     (Cylinder(0.25, 0.5, 'z', (0.013, -0.007, 0.25)), 'A')],
                                    (-.3, -.3, -.1), (.3, .3, .55)),
    'sphere dimer, gap .5 D': ([(Sphere(0.25, (-0.375, 0, 0.021)), 'A'),
                                (Sphere(0.25, (0.375, 0, 0.021)), 'A')],
                               (-.65, -.3, -.3), (.65, .3, .3)),
    'dimer, gap .05 D, two metals': ([(Sphere(0.25, (-0.2625, 0, 0.021)), 'A'),
                                      (Sphere(0.25, (0.2625, 0, 0.021)), 'B')],
                                     (-.55, -.3, -.3), (.55, .3, .3)),
}


def main():
    rows = [f'{"scene":30s} {"cells":>5s} {"cut":>7s} {"creased":>8s} {"% cut":>6s} '
            f'{">=3 media":>9s} {"% cut":>6s} {"joint":>6s}']
    start = time.perf_counter()
    for name, (structures, lo, hi) in SCENES.items():
        for cells in RESOLUTIONS:
            r = census(structures, 1.0 / cells, lo, hi)
            cut = max(r['interface'], 1)
            rows.append(f'{name:30s} {cells:5d} {r["interface"]:7d} {r["creased"]:8d} '
                        f'{100 * r["creased"] / cut:6.2f} {r["three"]:9d} '
                        f'{100 * r["three"] / cut:6.2f} {r["joint"]:6d}')
    rows.append(f'({time.perf_counter() - start:.1f} s; E-component voxels near a surface; '
                f'"cells" per unit length, the bodies are about 0.5 across)')
    text = '\n'.join(rows)
    print(text)
    out = ROOT / 'studies' / 'results' / 'fdtd_subpixel_media_census.log'
    out.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
