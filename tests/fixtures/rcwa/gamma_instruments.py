"""Instruments for the normal field's interface set -- shared by the gamma test files.

Three things, each an oracle or a harness rather than a test:

* :func:`gamma_points` -- the **exact interface set** of a layer, from the outlines alone
  (:func:`wavebit._common.geometry.overlap.interfaces_of`): every image's outline sampled densely, a sample
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

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_root
import contextlib
import hashlib
import importlib.util
import pathlib

import numpy as np

# A maintained study may select an archived flat/src checkout before loading this
# same oracle. Resolve private imports there without changing the instrument math.
if importlib.util.find_spec('wavebit._core') is not None:
    from wavebit._core import overlap as _overlap, parametric as _parametric
    from wavebit._core.lattice import as_lattice, sample_grid
else:
    from wavebit._common.geometry import overlap as _overlap
    from wavebit.rcwa._engine import parametric as _parametric
    from wavebit.lattice import as_lattice, sample_grid


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

    **The number is scene-dependent, and a gate on it belongs to the scene.** Every box scene
    carries the fixed-normal wedge beyond a run's end, where the band's samples read the
    diagonal of two faces (C3's price): a lone 350 box reads 0.24 deg at N = 512, the K seam
    0.22, the A|B|C scene 0.27, and two same-material boxes 300 and 400 tall abutting inside
    the cell 0.32 -- identical to the last digit whether the taller one is written upright or
    turned a quarter turn, so it is the wedge and not a defect, and a flat 0.3 gate would refuse
    that correct field. Gate a new contact scene by comparison with the same geometry's
    one-shape spelling where one exists (``D_SHORT`` against ``ONE_SHORT``), and otherwise by
    its separation from the mutants that must fail it (the A|B|C gate is 0.35 against 3.7 and
    5.9 for the mutants of the sign window).
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

    The compiled bytecode is deleted on the way in and on the way out. A pyc records the
    source's size and whole-second mtime, and a same-size replacement written within the
    second of the last compile -- ``_BLEND_EXPONENT = 4`` to ``2`` -- is served the stale
    bytecode by ``reload`` and never runs: the third way a mutant can look like a survivor
    while having done nothing, found by exactly that mutant.
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
    _drop_bytecode(path)
    try:
        yield
    finally:
        path.write_text(source)
        _drop_bytecode(path)
        assert hashlib.md5(path.read_text().encode()).hexdigest() == digest, \
            f'{path.name} was not restored'


def _drop_bytecode(path):
    import importlib
    import importlib.util
    cached = pathlib.Path(importlib.util.cache_from_source(str(path)))
    if cached.exists():
        cached.unlink()
    importlib.invalidate_caches()


def run_isolated(code, timeout=600):
    """Run ``code`` in a fresh interpreter on the sources as they are on disk right now.

    For a mutant that cannot be reloaded in place: ``domain.py`` defines :class:`Material`,
    which ``as_material`` type-checks against and which ``engine`` and ``spec`` import by name,
    so a ``reload`` leaves two classes called ``Material`` in one process and every later test
    that builds one fails for a reason that has nothing to do with the mutant. A subprocess
    imports the mutated file once, cleanly, and its exit status is the verdict: ``True`` means
    the instrument passed (the mutant SURVIVED), ``False`` that it raised (killed). The stderr
    is returned beside it so a survivor can be read rather than guessed at.
    """
    import subprocess
    import sys
    root = repository_root()
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                          timeout=timeout, cwd=str(root),
                          env=dict(__import__('os').environ, PYTHONPATH=__import__('os').pathsep.join((str(root / 'src'), str(root))),
                                   PYTHONDONTWRITEBYTECODE='1'))
    return proc.returncode == 0, proc.stderr[-2000:]
