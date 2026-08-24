"""
scene_hash.py — a byte fingerprint of the solver, wide enough to arbitrate a refactor.

A refactor that must not move a number needs an instrument that would notice if it did, and
the instrument has to be wider than the change. This one sweeps

    4 cell shapes    square, rectangle, hexagon, shear
  x 4 modeling paths raster, shapes (summed), shapes (painted), block
  x 2 incidence      top, bottom
  x 3 backends       numpy, jax, torch
  = 96 scenes

and hashes, per scene, the per-order efficiencies **and** the reconstructed field. Both,
because `R` and `T` are sums over orders: a per-order wavevector wrong by a whole reciprocal
step passes them while conserving energy to 1e-15, which is the shape of `b7c4645`. The field
is where such a thing is a different *shape* rather than a different total.

**The backend axis is the point.** A sweep on numpy alone cannot arbitrate a change to the
per-backend layer, which is two thirds of what such a change touches. It is also the axis that
costs the most to add later: the jax and torch arms need their optional dependencies present,
so a run that silently skipped them would report a smaller sweep as a clean one. It says so
instead -- an absent backend is recorded by name in the digest rather than omitted from it.

**Refusals are hashed too, by exception name.** Twelve of the 96 are refused on purpose: the
block modeller on the two skewed cells, both incidence sides, all three backends. A refactor
that changes *which* refusal fires, or removes one, moves the digest exactly as a changed
number does. Hashing only the successes would make "it stopped refusing" a silent pass, which
is the failure mode `c5b061f` was.

A refusal that is *not* on purpose is the trap on the other side, and this sweep walked into it
before it was fixed: forcing `factorization='nvf'` onto the block path and handing the summed
rule two overlapping discs refused 48 of the 96 for reasons that were the scene's fault, not
the solver's. Half the instrument was measuring its own configuration. Any scene added here has
to be checked for that -- a high refusal count is a symptom, not a feature.

Usage::

    python studies/scene_hash.py             # one digest, plus a per-scene table
    python studies/scene_hash.py --quiet     # the digest alone, for a diff

To check that a change moved nothing, run it against an extraction of the commit before the
change rather than against a stash or a checkout::

    git archive <rev> | tar -x -C /tmp/before
    (cd /tmp/before && python studies/scene_hash.py --quiet)

`git archive` touches no shared git state, which matters: several sessions share this repo's
worktrees and refs (see `.claude/`), and a `git stash` or a `git checkout` in one of them is
visible to the others.

This is a *study*, not a test: it pins nothing and asserts nothing, because a digest committed
as a golden would have to be regenerated on every legitimate change and would then arbitrate
nothing. It is an instrument to be run twice -- before and after -- by whoever is making the
change.
"""
import argparse
import hashlib
import importlib.util
import os
import sys
import warnings

import numpy as np

# The repo root, not the caller's cwd: a script run from anywhere else imports whichever
# checkout is on the path first, and in a worktree that is the *main* one -- which would make
# a before/after comparison measure the wrong tree, quietly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wavebit                                                        # noqa: E402

P, WL, H, ROD = 400.0, 900.0, 150.0, 110.0

CELLS = {
    'square': np.diag([P, P]),
    'rect': np.diag([P, 1.6 * P]),
    'hex': np.column_stack(([P, 0.0], [P / 2, P * np.sqrt(3) / 2])),
    'shear': np.column_stack(([P, 0.0], [0.35 * P, 1.2 * P])),
}
BACKENDS = {'numpy': 0, 'jax': 1, 'torch': 2}
SIDES = ('top', 'bottom')


def _rod(basis, n=40, radius=ROD):
    """A disc on the cell's own parametric grid, so every cell carries the same object."""
    a1, a2 = np.asarray(basis)[:, 0], np.asarray(basis)[:, 1]
    t = (np.arange(n) + 0.5) / n
    u, v = np.meshgrid(t, t)
    pts = u[..., None] * a1 + v[..., None] * a2
    centre = 0.5 * (a1 + a2)
    inside = np.zeros(pts.shape[:2], dtype=bool)
    for i in (-1, 0, 1):                               # minimum image over the lattice
        for j in (-1, 0, 1):
            s = i * a1 + j * a2
            inside |= (((pts[..., 0] - centre[0] - s[0]) ** 2
                        + (pts[..., 1] - centre[1] - s[1]) ** 2) <= radius ** 2)
    return np.where(inside, 5.76, 1.0)[None]


def _shapes(basis, painted=False):
    """Two discs on the closed-form path -- disjoint when summed, overlapping when painted.

    The two rules take different scenes on purpose. `combine='add'` refuses an overlap by
    design, so the summed arm is given discs that do not touch; the painted arm is given ones
    that do, because a painting with nothing overlapping is the summed structure again and
    would exercise the resolver not at all.
    """
    a1, a2 = np.asarray(basis)[:, 0], np.asarray(basis)[:, 1]
    centre = 0.5 * (a1 + a2)
    if painted:
        offsets, radii = ((0.0, 0.0), (40.0, 0.0)), (ROD, ROD)
    else:
        offsets, radii = ((-0.22 * P, 0.0), (0.22 * P, 0.0)), (0.16 * P, 0.16 * P)
    layer = {'background': 1.0, 'shapes': [
        {'type': 'circle', 'cx': float(centre[0]) + dx, 'cy': float(centre[1]) + dy,
         'lx': 2 * r, 'material': eps}
        for (dx, dy), r, eps in zip(offsets, radii, (5.76, 9.0))]}
    if painted:
        layer['combine'] = 'paint'
    return [layer]


BLOCK = [[1.0, [['ellipse', P / 2, P / 2, 0.6 * P, 0.6 * P, 6.25, 0.0, 24, 24]]]]


def _structure(kind, basis):
    return {'raster': _rod(basis), 'add': _shapes(basis),
            'paint': _shapes(basis, True), 'block': BLOCK}[kind]


def _available(name):
    return name == 'numpy' or importlib.util.find_spec(name) is not None


def scenes():
    """Every (cell, path, side, backend) combination, in a fixed order."""
    for cell in CELLS:
        for kind in ('raster', 'add', 'paint', 'block'):
            for side in SIDES:
                for backend in BACKENDS:
                    yield cell, kind, side, backend


def digest_one(cell, kind, side, backend):
    """``(sha256 hex, note)`` for one scene. A refusal hashes its exception's name."""
    if not _available(backend):
        return hashlib.sha256(f'absent:{backend}'.encode()).hexdigest(), f'{backend} absent'
    basis = CELLS[cell]
    h = hashlib.sha256()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            solver = wavebit.rcwa(
                backend=BACKENDS[backend], structure=_structure(kind, basis), period=[P, P],
                lattice=np.asarray(basis).tolist(), truncation_order=[4, 4], thickness=[H],
                wavelength=WL, top=1.0, bot=2.25, theta=0.3, phi=0.4, psi=0.2,
                incidence_side=side, answers='field',
                # NVF everywhere it is available. The block path builds its coefficients its
                # own way and refuses the name, so it takes the default -- forcing one
                # factorization across all four paths would refuse a quarter of the sweep and
                # call the refusal a result.
                **({} if kind == 'block' else {'factorization': 'nvf'}))
            result = solver.solve()
            field = solver.reconstruct_field(sample_x=7, sample_y=5, sample_z=3)
        for value in (result.R, result.T, field):
            h.update(np.ascontiguousarray(np.asarray(value, dtype=np.complex128)).tobytes())
        return h.hexdigest(), 'ok'
    except Exception as exc:                           # noqa: BLE001 - the name is the datum
        h.update(type(exc).__name__.encode())
        return h.hexdigest(), type(exc).__name__


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--quiet', action='store_true', help='print the digest and nothing else')
    args = ap.parse_args()

    overall = hashlib.sha256()
    rows, counts = [], {}
    for cell, kind, side, backend in scenes():
        one, note = digest_one(cell, kind, side, backend)
        overall.update(one.encode())
        rows.append((f'{cell}/{kind}/{side}/{backend}', one[:12], note))
        counts[note] = counts.get(note, 0) + 1

    if not args.quiet:
        for name, short, note in rows:
            print(f'  {name:34s} {short}  {note}')
        print()
        for note, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f'  {note:36s} {n}')
        print()
    print(f'{len(rows)} scenes  {overall.hexdigest()}')


if __name__ == '__main__':
    main()
