"""Instruments for the normal field's interface set -- shared by the gamma test files.

Three things, each an oracle or a harness rather than a test:

* :func:`gamma_points` -- the **exact interface set** of a layer, from the outlines alone
  (:func:`wavebit._core.overlap.interfaces_of`): every image's outline sampled densely, a sample
  kept only if a step along its outward normal lands in no image. It shares no arithmetic with
  the pair sweep or the field, which is what lets it judge both.
* :func:`band_error` -- a field compared against that set over a band around it, as the angle
  ``acos|n . n_oracle|`` at every sample within ``band`` of an interface point. Sign is free
  (the factorization reads ``n n^T``), so the comparison is of directions.
* :func:`mutated` -- a source-mutation harness that says **DID-NOT-APPLY** when the text it was
  asked to replace is not there and **NO-OP** when the replacement changes nothing, and
  restores the file by md5 afterwards. A mutation that does not apply, or applies to nothing,
  reads exactly like a passing test unless the harness says so out loud -- which is how two of
  the eighteen mutants over the previous mechanism were first miscounted.

Validated before it was trusted: the 1-D grating (700 square, ``lx = 350``, ``ly = 700``) reads
exactly 0.000 deg against :func:`gamma_points` over its boundary band, on a field that is
provably ``(+-1, 0)`` everywhere, so the oracle carries no y-normal on the seam a bar with no
y-face could be measured against (:func:`test_gamma_plan.test_the_oracle_reads_the_grating_exactly`).
"""
import contextlib
import hashlib
import pathlib

import numpy as np

from wavebit._core import overlap as _overlap
from wavebit._core import parametric as _parametric
from wavebit._core.lattice import as_lattice, sample_grid


def gamma_points(shapes, lattice, ring=None, n=1024, step=None):
    """``(K, 4)`` rows ``(x, y, nx, ny)`` of the exact interface set, in the internal frame.

    ``ring`` defaults to the plan's own, so the oracle sees every image the sweep saw.
    """
    lattice = as_lattice(lattice=lattice)
    prims = _parametric._primitives(shapes, lattice)
    if ring is None:
        ring = _parametric.plan_gamma(shapes, lattice).ring
    return _overlap.interfaces_of(prims, lattice, ring, n=n, step=step)


def band_error(normal_x, normal_y, grid_shape, lattice, gamma, band_px=1.5):
    """Angle statistics of a sampled field against the oracle, over a band around Gamma.

    Returns ``(mean_deg, fraction_over_45_deg, n_samples)`` over the samples of the
    ``grid_shape`` grid that lie within ``band_px`` pixels of an interface point, each compared
    with the normal of its nearest interface point.
    """
    from scipy.spatial import cKDTree

    lattice = as_lattice(lattice=lattice)
    n_v, n_u = grid_shape
    X, Y = sample_grid(np, grid_shape, lattice, np.float64)
    pixel = min(float(np.hypot(lattice[0][0], lattice[1][0])) / n_u,
                float(np.hypot(lattice[0][1], lattice[1][1])) / n_v)
    dist, nearest = cKDTree(gamma[:, :2]).query(np.column_stack([X.ravel(), Y.ravel()]))
    keep = dist <= band_px * pixel
    ox, oy = gamma[nearest[keep], 2], gamma[nearest[keep], 3]
    fx, fy = np.asarray(normal_x).ravel()[keep], np.asarray(normal_y).ravel()[keep]
    cos = np.clip(np.abs(fx * ox + fy * oy), 0.0, 1.0)
    angle = np.degrees(np.arccos(cos))
    return float(angle.mean()), float((angle > 45.0).mean()), int(keep.sum())


@contextlib.contextmanager
def mutated(path, old, new):
    """Replace ``old`` by ``new`` in ``path`` for the block, then restore the file by md5.

    Raises ``AssertionError('DID-NOT-APPLY ...')`` when ``old`` is not in the file and
    ``AssertionError('NO-OP ...')`` when the replacement leaves the text unchanged, so a mutant
    that never ran is never counted as one that survived. The module under test has to be
    re-imported by the caller (``importlib.reload``) for the mutation to be live.
    """
    path = pathlib.Path(path)
    source = path.read_text()
    digest = hashlib.md5(source.encode()).hexdigest()
    if old not in source:
        raise AssertionError(f'DID-NOT-APPLY: {old!r} is not in {path.name}')
    changed = source.replace(old, new, 1)
    if changed == source:
        raise AssertionError(f'NO-OP: replacing {old!r} by {new!r} changes nothing')
    path.write_text(changed)
    try:
        yield
    finally:
        path.write_text(source)
        assert hashlib.md5(path.read_text().encode()).hexdigest() == digest, \
            f'{path.name} was not restored'
