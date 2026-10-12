"""What does a structure a Symmetry face cannot carry do to the answer, with nothing refusing it?

Stage 4's C5 audit, run before the Symmetry door (``checks.check_symmetry``). A ``Symmetry`` face
stands for a scene that is its own mirror image: the run carries the domain side and reflects
the field there, so what it models is the structures' domain part and their image. The plane is
the node plane at the wall on a ``-`` face and the half plane ``dx/2`` inside the wall on a ``+``
face. This study measures the three ways a declaration can disagree with that, each with the
door held open: a body declared only on the domain side and flush against the plane (the
voxels the plane cuts read the declared half and not the image's), a body crossing the plane
asymmetrically (the run models the mirror of the domain part, not the declaration), and a body
centred on a ``+`` face's wall instead of its plane.

**Instrument.** The fundamental frequency of a PEC cavity, ``0.8 x 0.3 x 0.3`` long in x (0.85
on the ``+`` side, so that its centre is a cell centre) at ``dx = 0.05``, holding a bar of
``eps = 4`` and ``0.1 x 0.1`` section along x -- a ``Symmetry(parity=1)`` half box against the
full box. An initial ``Ez = cos(pi x / 0.8) cos(pi y / 0.3)`` rings for 4000 steps at courant
0.5, and a matrix pencil on one probe's ``Ez`` gives ``omega``. The bar moves ``omega`` by about
-10.8% from the empty cavity, so a shift of 1e-3 is about 1% of the bar's own effect. Each row
compares the half box with the full box holding what the user meant: the symmetric bar for a
flush half and a wall-centred bar, and the declared bar itself for an asymmetric crossing. Each
row also reads the lowered ``eps`` of the component whose voxel the plane cuts, at the bar's
axis, in the half box and in the full one.

Run from the repository root: ``python studies/fdtd_symmetry_mirror.py``; rows go to
``studies/results/fdtd_symmetry_mirror.log`` as JSON lines. About a minute on one thread.
"""
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np                                                              # noqa: E402
import wavebit.fd as fd                                                          # noqa: E402
import wavebit.fd._lower.checks as checks                                        # noqa: E402

OUT = ROOT / 'studies' / 'results' / 'fdtd_symmetry_mirror.log'
DX, M, T, STEPS, LENGTH = 0.05, 8, 0.3, 4000, 0.8


def hold_the_door_open():
    """The measurement is of the scene the door refuses, so the door is a no-op here."""
    if hasattr(checks, 'check_symmetry'):
        checks.check_symmetry = lambda scene, grid: None


def mode(x, y, z):
    return np.cos(np.pi * x / LENGTH) * np.cos(np.pi * y / T) * np.ones_like(z)


def cavity(side, half, shapes, eps=4.0):
    """The cavity; a ``+`` side's full box has an odd cell count so its centre is a cell centre."""
    if side == '-':
        n_full = 2 * M
        low_full = -n_full * DX / 2
        low, n = (0.0, M) if half else (low_full, n_full)
    else:
        n_full = 2 * M + 1
        low_full = -n_full * DX / 2
        low, n = (low_full, M + 1) if half else (low_full, n_full)
    size = n * DX
    if half:
        mirror = fd.Symmetry(parity=1)
        x = (mirror, fd.PEC()) if side == '-' else (fd.PEC(), mirror)
    else:
        x = fd.PEC()
    probe = (0.1125, 0.0125, 0.0) if side == '-' else (-0.1125, 0.0125, 0.0)
    return fd.Scene(domain=fd.Domain(size=(size, T, T), center=(low + size / 2, 0, 0)),
                    boundaries=fd.Boundaries(x=x, y=fd.PEC(), z=fd.PEC()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=STEPS, courant=0.5),
                    initial=fd.InitialField(E=(None, None, mode)),
                    monitors=(fd.Record('Ez', at=probe, component='Ez'),),
                    structures=tuple(fd.Structure(s, eps) for s in shapes))


def omega(series, dt, order=8):
    """The dominant ringing frequency: a matrix pencil on the probe's series."""
    s = np.asarray(series, float)
    n = len(s)
    half = n // 2
    hankel = np.array([s[i:i + half + 1] for i in range(n - half)])
    _, _, vh = np.linalg.svd(hankel, full_matrices=False)
    v = vh[:order].conj().T
    z = np.linalg.eigvals(np.linalg.pinv(v[:-1]) @ v[1:])
    amplitude = np.linalg.lstsq(np.vander(z, n, increasing=True).T, s, rcond=None)[0]
    k = np.argmax(np.abs(amplitude) * (np.abs(np.angle(z)) > 1e-6))
    return abs(np.angle(z[k])) / dt, float(abs(amplitude[k]) / np.abs(amplitude).sum())


def run(scene):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = fd.run(scene)
        plan = fd.plan(scene)
    series = result['Ez'].values
    return omega(np.asarray(series), plan.dt)


def plane_eps(side, half_shape, full_shape):
    """The lowered ``eps`` on the plane at the bar's axis, half box and full box.

    The component whose voxel the plane cuts: ``Ey`` on the ``-`` side (its point is on the
    node plane), ``Ex`` on the ``+`` side (on the half plane, the centre of the last cell).
    """
    component, row = (1, 3) if side == '-' else (0, 3)
    values = []
    for half, shape in ((True, half_shape), (False, full_shape)):
        scene = cavity(side, half, [shape])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plan = fd.plan(scene)
            be = np.asarray(fd.lower(scene, plan, 'numpy')['be'])
        layer = 0 if (side == '-' and half) else M
        values.append(float((plan.dt / plan.dx) / be[component, 3, row, layer]))
    return values


BAR = (0.4, 0.1, 0.1)
CASES = (
    # side, label, shape in the half box, shape the full box compares against
    ('-', 'control: Box symmetric about the plane', fd.Box(size=BAR), fd.Box(size=BAR)),
    ('-', 'Box flush half [0, 0.2]', fd.Box(size=(0.2, 0.1, 0.1), center=(0.1, 0, 0)),
     fd.Box(size=BAR)),
    ('-', 'Box shifted one cell [-0.15, 0.25]', fd.Box(size=BAR, center=(0.05, 0, 0)),
     fd.Box(size=BAR)),
    ('-', 'Box declared [-0.1, 0.2]', fd.Box(size=(0.3, 0.1, 0.1), center=(0.05, 0, 0)),
     fd.Box(size=(0.3, 0.1, 0.1), center=(0.05, 0, 0))),
    ('-', 'control: Region symmetric about the plane', fd.Region(size=BAR), fd.Region(size=BAR)),
    ('-', 'Region declared [-0.1, 0.2]', fd.Region(size=(0.3, 0.1, 0.1), center=(0.05, 0, 0)),
     fd.Region(size=(0.3, 0.1, 0.1), center=(0.05, 0, 0))),
    ('+', 'control: Box symmetric about the plane', fd.Box(size=BAR), fd.Box(size=BAR)),
    ('+', 'Box centred on the wall, not the plane',
     fd.Box(size=BAR, center=(0.025, 0, 0)), fd.Box(size=BAR)),
    ('+', 'Box flush half [-0.2, 0]', fd.Box(size=(0.2, 0.1, 0.1), center=(-0.1, 0, 0)),
     fd.Box(size=BAR)),
)


def main():
    hold_the_door_open()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as log:
        for side in '-+':
            empty, _ = run(cavity(side, False, []))
            row = dict(case='empty full box', side=side, omega=empty)
            log.write(json.dumps(row) + '\n')
            print(row)
        for side, label, half_shape, full_shape in CASES:
            start = time.perf_counter()
            reference, weight = run(cavity(side, False, [full_shape]))
            value, _ = run(cavity(side, True, [half_shape]))
            eps_half, eps_full = plane_eps(side, half_shape, full_shape)
            row = dict(case=label, side=side, omega_half=value, omega_full=reference,
                       dominant_weight=weight, shift=(value - reference) / reference,
                       plane_eps_half=eps_half, plane_eps_full=eps_full,
                       seconds=time.perf_counter() - start)
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(row)


if __name__ == '__main__':
    main()
