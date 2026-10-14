"""How far should the star lift go? Its floor ``THETA``, measured before it is chosen.

The star lift (``src/wavebit/fdtd/_prepare/stars.py``) moves the two edges of each axis at a node toward
the node's Kottke tensor until the node's star -- the six edges that touch it -- clears ``THETA``
times its physical scale. ``THETA`` in ``(0, 1)`` is its one free constant. The lift is positive
definite by construction at any ``THETA``; what ``THETA`` changes is how many stars move, how
far, and so how far the operator is from the default rule's.

**Instruments.**

* ``counts``: stars moved, the largest move, and the assembled operator's smallest eigenvalue
  times ``eps_max`` (dense ``eigvalsh`` on the term region's matrix, up to :data:`DENSE`
  unknowns) -- lift off and at each ``THETA``, by contrast, body, cells across and placement.
* ``bits``: which ``THETA`` moves no star of an ``eps <= 6`` body, so that those scenes lower to
  the bits they did before the lift.
* ``static``: the sphere's static polarizability against Clausius-Mossotti, lift off and at each
  ``THETA`` -- study S2b's instrument 1 (``studies/fdtd/subpixel_cells_across/fdtd_subpixel_cells_across.py``, imported: the
  node-lattice potential solve with the production operator, the box image removed with its
  fitted ``KAPPA``), with the lifted diagonal in place of ``build``'s. Where the unlifted operator
  is indefinite the solve does not converge and the row says so.
* ``order``: the same error as the grid is refined, 8 to 24 cells across at ``eps`` 12, 25, 40
  and 100.

**The first form, and why it is not this one.** The lift's first form, in the history just
before the star, used Werner, Bauer and Cary's eight 3x3 blocks per node. This study measured it
first, with the same instruments and rule; its rows (fdtd_subpixel_lift_blocks.log) are
archived outside the repository. The rule picked ``THETA = 0.02`` for it (worst static error over the rule's set 1.814%, against
9.217% unlifted), and its ``order`` rows showed what that rule could not: at ``eps = 25`` the
lifted error settled near -0.2% from 12 to 24 cells across (the unlifted rule went +0.96% ->
+0.36% and is positive definite there), because the eight blocks move 5-12% of a positive
definite sphere's blocks at every resolution and each move is of order one -- an error of first
order. At ``eps = 12`` the same bias was hidden by cancellation against the rule's own positive
error. The star's test is weaker (the harmonic mean of an axis's two edges, where a block takes
one of them); it moves no star of an ``eps = 12`` sphere at 8 to 24 cells across, and at
``eps = 25`` its error follows the unlifted rule's down (24 cells, the mean over three placements:
+0.30% against +0.35%).

**The star runs' result** (fdtd_subpixel_lift.log, archived outside the repository). No star of
any ``eps <= 6`` body moved at any ``THETA`` tried; the worst static errors over the rule's set were 5.189 / 5.196
/ 5.207 / 5.239 / 5.279% at ``THETA`` 0.02 / 0.05 / 0.1 / 0.2 / 0.3 (9.217% unlifted, which
cannot be solved at all at ``eps = 100`` and 16 cells across at three of five placements), so the
rule chose 0.02. Its fitted static orders, 8 to 24 cells across: 1.808 at ``eps = 12`` (the
unlifted rule's, to the bit), 1.730 at 25 (1.637 unlifted), 1.770 at 40 (1.506) and 1.767 at 100
(unlifted unsolvable).

**The rule, written before the star runs.** ``THETA`` is the value in :data:`THETAS` that
minimises the largest ``|static error|`` of the sphere over ``eps`` 12, 20, 25, 40 and 100, 8 and
16 cells across and the five placements, among the values that move no star of any body in
``bits``. A value within 0.05 percentage points of the best goes to the smaller ``THETA``, which
moves fewer stars. The ``order`` rows are reported beside it: if the chosen ``THETA``'s fitted
static order at ``eps`` 12 or 25 falls more than 0.2 below the unlifted rule's, the choice goes
to the owner instead. (The first form's rule read the order at ``eps = 12`` only, where the
cancellation hid the bias; ``eps = 25`` was added after that run.) The dynamic check,
independent of this instrument, is the Mie scattering of the small high-index sphere.

Run from the repository root: ``python studies/fdtd/subpixel_lift/fdtd_subpixel_lift.py [selftest|counts|bits|
static|order]``; rows go to ``studies/fdtd/subpixel_lift/records/fdtd_subpixel_lift.log`` as JSON
lines (appended; a rerun resumes the static and order rows), each with the SHA-256 of
``stars.py`` it measured. The recorded run's log is archived outside the repository, so a rerun
here starts empty. Single-threaded BLAS.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_root
import hashlib
import json
import math
import os
import sys
import time

ROOT = repository_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np                                                              # noqa: E402
import scipy.sparse as sp                                                        # noqa: E402
import wavebit.fdtd._legacy as fd                                                          # noqa: E402
from wavebit.fdtd._kernel.observe import InteriorEnergy                            # noqa: E402
from wavebit.fdtd._prepare import constitutive, stars                                # noqa: E402
from wavebit.fdtd._prepare.coefficients import build                                # noqa: E402
from wavebit.fdtd._runtime import runtime                                         # noqa: E402
# S2b's instrument. Importing it holds the cells-across door open (its small bodies are
# measured here too) and wraps constitutive.validate, which now refuses only an uncertified
# lift; that wrapper records it instead.
import studies.fdtd.subpixel_cells_across.fdtd_subpixel_cells_across as s2b                                 # noqa: E402

OUT = ROOT / 'studies' / 'results' / 'fdtd_subpixel_lift.log'
THETAS = (0.02, 0.05, 0.1, 0.2, 0.3)
SOURCE = hashlib.sha256((ROOT / 'src/wavebit' / 'fd' / '_lower' / 'stars.py').read_bytes()).hexdigest()
DX = s2b.DX


def _emit(row):
    row = dict(row, stars_sha256=SOURCE[:16])
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def lifted(scene, plan, theta):
    """The edge permittivity ``(3, n, n, n)`` with the lift at ``theta`` (``None``: the default
    rule before the lift), the E terms' cross coefficients, and each term's ``stars.Lifted``.

    The patch is ``constitutive.lift``'s own arithmetic, ``be + ratio * delta`` over the term's
    view, with ``theta`` passed through.
    """
    rt = runtime('numpy')
    coefficients = dict(build(scene, plan, rt))
    nodes = {}
    cross = constitutive.lower(scene, plan, rt, nodes=nodes)
    n = plan.grid.n
    be = np.array(np.broadcast_to(np.asarray(coefficients['be']), (3, n[2], n[1], n[0])))
    lifts = {}
    for term in plan.constitutive:
        if term.field != 'E' or theta is None:
            continue
        c = np.asarray(cross[term.name])
        diagonal = np.broadcast_to(InteriorEnergy._cut(be, slice(None), term.box) / term.ratio,
                                   c.shape)
        out = stars.lift('E', diagonal, c, *nodes[term.name], rt, theta=theta)
        lifts[term.name] = out
        if out.delta is not None:
            be[term.view] = be[term.view] + term.ratio * np.asarray(out.delta)
    permittivity = (plan.dt / plan.dx) / be
    terms = [(t, np.asarray(cross[t.name])) for t in plan.constitutive
             if t.field == 'E' and t.name in cross]
    return permittivity, terms, lifts


#: Regions up to this many unknowns get the exact smallest eigenvalue (dense ``eigvalsh``);
#: larger ones are counted only. ARPACK's smallest-algebraic mode did not converge in 7 runs of
#: the understand workflow's probes, and a wrong eigenvalue is worse than none.
DENSE = 7000


def smallest(permittivity, terms):
    """The smallest eigenvalue of each E term's region operator, ``diag(1/eps) + C``, or
    ``None`` for a region above :data:`DENSE` unknowns."""
    out = []
    for term, cross in terms:
        (k0, k1), (j0, j1), (i0, i1) = term.box
        diagonal = np.concatenate([1.0 / permittivity[c, k0:k1, j0:j1, i0:i1].ravel()
                                   for c in range(3)])
        if diagonal.size > DENSE:
            out.append(None)
            continue
        matrix = sp.diags(diagonal) + s2b._cross_matrix(cross, (k1 - k0, j1 - j0, i1 - i0))
        out.append(float(np.linalg.eigvalsh(matrix.toarray())[0]))
    return out


def scene_for(kind, across, eps, placement, box_diameters):
    """A closed PEC box around the body: ``box_diameters`` across it (S2b's ``alpha``), or, at
    ``box_diameters=None``, ``2 across + 8`` cells (S2b's certificate instrument), which clears
    the turned box's corners."""
    if box_diameters is None:
        n = 2 * int(math.ceil(across)) + 8
    else:
        n = int(math.ceil(box_diameters * across))
        n = max(n, int(math.ceil(across)) + 8)
    n += n % 2                                    # even, so the origin is a node (S2b)
    shape, volume = s2b.body(kind, across, DX, tuple(o * DX for o in placement))
    scene = s2b.closed_scene((fd.Structure(shape, fd.Material(eps=eps)),), n, DX)
    return scene, s2b.quiet(fd.plan, scene), volume, n


def alpha(kind, across, eps, placement, box_diameters, theta, field_axis=2):
    """``alpha / V`` along ``field_axis`` (S2b's ``alpha``, box image removed) and the lifts."""
    scene, plan, volume, n = scene_for(kind, across, eps, placement, box_diameters)
    permittivity, terms, lifts = lifted(scene, plan, theta)
    edges = [s2b._edges(permittivity[c], n, c) for c in range(3)]
    raw = s2b.polarizability(n, DX, edges, terms, field_axis)[field_axis] / volume
    fraction = volume / (n * DX) ** 3
    return raw / (1 + s2b.KAPPA * raw * fraction), lifts


def _done(study):
    """The ``(eps, across, theta)`` rows of ``study`` already in the log: a rerun resumes."""
    if not OUT.exists():
        return set()
    rows = [json.loads(line) for line in open(OUT, encoding='utf-8') if line.strip()]
    return {(r['eps'], r['across'], r['theta']) for r in rows if r.get('study') == study}


def _alpha_or_none(*args):
    """``alpha``, or ``(None, {})`` where S2b's inner solve on the region does not converge --
    which it cannot on an indefinite operator (the unlifted ``eps = 100`` sphere at 16 cells
    across: CG stopped at its 2000-iteration limit)."""
    try:
        return alpha(*args)
    except RuntimeError as error:
        if 'CG info' not in str(error):
            raise
        return None, {}


def _counts(lifts):
    return (sum(out.moved for out in lifts.values()), sum(out.stars for out in lifts.values()),
            max((out.largest for out in lifts.values()), default=0.0),
            all(out.certified for out in lifts.values()))


def selftest():
    """The unlifted arm is S2b's production arm, to the bit, and the lifted arm at the module's
    THETA is what ``fd.lower`` hands the kernel."""
    for eps in (2.25, 12.0):
        mine = alpha('sphere', 6, eps, s2b.PLACEMENTS['c'], 6, None)[0]
        theirs = s2b.alpha('sphere', 6, eps, s2b.PLACEMENTS['c'], 6)[0]
        assert mine == theirs, (eps, mine, theirs)
    scene, plan, _, n = scene_for('sphere', 8, 20.0, s2b.PLACEMENTS['c'], 3)
    permittivity, _, lifts = lifted(scene, plan, stars.THETA)
    be = np.broadcast_to(np.asarray(fd.lower(scene, plan, runtime('numpy'))['be']), (3, n, n, n))
    assert np.array_equal((plan.dt / plan.dx) / be, permittivity)
    assert _counts(lifts)[0] > 0
    _emit(dict(study='selftest', passed=True))


def run_counts():
    """Stars moved and the smallest eigenvalue, lift off and at each THETA."""
    cases = [('sphere', eps, across, name) for eps in (6.0, 9.0, 12.0, 20.0, 25.0, 40.0, 100.0)
             for across in (8, 16) for name in s2b.PLACEMENTS]
    cases += [(kind, eps, 8, name) for kind in ('box30', 'cylinder', 'hexprism')
              for eps in (12.0, 25.0) for name in ('node', 'mieA', 'c')]
    for kind, eps, across, name in cases:
        scene, plan, _, _ = scene_for(kind, across, eps, s2b.PLACEMENTS[name], None)
        for theta in (None,) + THETAS:
            started = time.perf_counter()
            permittivity, terms, lifts = lifted(scene, plan, theta)
            moved, total, largest, certified = _counts(lifts)
            _emit(dict(study='counts', body=kind, eps=eps, across=across, placement=name,
                       theta=theta, moved=moved, stars=total, largest=largest,
                       certified=certified,
                       lmin_eps=[None if v is None else v * eps
                                 for v in smallest(permittivity, terms)],
                       seconds=time.perf_counter() - started))


def run_bits():
    """Which THETA moves no star of a low-contrast body."""
    for eps in (2.25, 4.0, 6.0):
        for kind in ('sphere', 'box30', 'cylinder', 'hexprism'):
            placements = s2b.PLACEMENTS if kind == 'sphere' else {
                k: s2b.PLACEMENTS[k] for k in ('node', 'mieA', 'c')}
            for across in (4, 8, 16):
                for name, placement in placements.items():
                    scene, plan, _, _ = scene_for(kind, across, eps, placement, None)
                    for theta in THETAS:
                        _, _, lifts = lifted(scene, plan, theta)
                        moved, total, largest, _ = _counts(lifts)
                        _emit(dict(study='bits', body=kind, eps=eps, across=across,
                                   placement=name, theta=theta, moved=moved, stars=total,
                                   largest=largest))


def run_static():
    """The sphere's static error against Clausius-Mossotti, lift off and at each THETA."""
    done = _done('static')
    for eps in (12.0, 20.0, 25.0, 40.0, 100.0):
        exact = 3 * (eps - 1) / (eps + 2)
        for across in (4, 8, 16):
            for theta in (None,) + THETAS:
                if (eps, across, theta) in done:
                    continue
                errors, moved = {}, {}
                started = time.perf_counter()
                for name, placement in s2b.PLACEMENTS.items():
                    a, lifts = _alpha_or_none('sphere', across, eps, placement, 6, theta)
                    errors[name] = None if a is None else a / exact - 1
                    moved[name] = _counts(lifts)[0]
                v = np.array([e for e in errors.values() if e is not None])
                _emit(dict(study='static', body='sphere', eps=eps, across=across, theta=theta,
                           errors=errors, moved=moved,
                           lo=float(v.min()) if v.size else None,
                           hi=float(v.max()) if v.size else None,
                           worst=float(np.abs(v).max()) if v.size else None,
                           unsolved=sum(e is None for e in errors.values()),
                           seconds=time.perf_counter() - started))


def run_order():
    """The static error as the grid is refined, lift off and at four THETAs."""
    done = _done('order')
    for eps in (12.0, 25.0, 40.0, 100.0):
        exact = 3 * (eps - 1) / (eps + 2)
        for theta in (None, 0.02, 0.05, 0.1, 0.2):
            for across in (8, 12, 16, 24):
                if (eps, across, theta) in done:
                    continue
                errors = {}
                started = time.perf_counter()
                for name in ('node', 'mieA', 'c'):
                    a, _ = _alpha_or_none('sphere', across, eps, s2b.PLACEMENTS[name],
                                          6 if across <= 16 else 4, theta)
                    errors[name] = None if a is None else a / exact - 1
                _emit(dict(study='order', body='sphere', eps=eps, theta=theta, across=across,
                           errors=errors, seconds=time.perf_counter() - started))


if __name__ == '__main__':
    modes = {'selftest': selftest, 'counts': run_counts, 'bits': run_bits,
             'static': run_static, 'order': run_order}
    for mode in (sys.argv[1:] or ['selftest', 'bits', 'counts', 'static', 'order']):
        modes[mode]()
