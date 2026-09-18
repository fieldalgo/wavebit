"""The scenes, samples and digest that ``fd_golden_stage0.npz`` and its test share.

One module, imported by both ``interface/freeze_fd_stage0.py`` (which writes the golden) and
``tests/test_fd_backends.py`` (which reads it), so that the two cannot disagree about which
scene a key names or which cells a sample holds. The scenes are the gate 1-4 boxes of
``tests/fd_helpers.py``, built here under fixed names; nothing in this module solves
anything.

What is frozen per scene and per backend is deliberately small: the probe series the
gates read, the final E and H at eight cells per component (``sample_indices``: corners,
the centre, a low-face and a high-face layer, interior cells), and one whole-volume
scalar, the energy ``sum |E|^2 + |H|^2``, which is what catches drift in a cell no sample
sits on. Not the volumes: at 16^3 they would be 200 KB per field per backend per scene
and say nothing the energy and the samples do not.

The golden's integrity digest is a SHA-256 over every numeric array's name, dtype, shape
and bytes in name order (``content_digest``). It is recomputed by the test from the loaded
arrays and compared with the value the archive records, so a hand-edited array fails
before any number is compared. The freezer validates the digest before it lets it
arbitrate anything: two independent freezes must agree byte for byte before one is
written, which is what says the value is a function of the code and not of the process.
"""
import hashlib
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                   # tests/, for fd_helpers
from fd_helpers import cavity_box, plane_wave_box, symmetry_pair, to_numpy   # noqa: E402

#: The committed golden.
GOLDEN = HERE / 'fd_golden_stage0.npz'

#: Every backend the golden pins, in the order the freezer runs them.
BACKENDS = ('numpy', 'jax', 'torch')

#: Gate 5's budget: NumPy, JAX and Torch agree, and each agrees with its own golden, to this
#: fraction of the array's peak. Measured 2026-09-18 on the freezing machine: Torch is
#: bit-identical to NumPy on every real scene and 2e-15 from it on the complex one; JAX
#: sits 1e-15 to 4e-14 from both (XLA fuses the update into a different rounding). The
#: budget is 25x above the worst of that so a CPU that contracts ``a*F + b*cF`` into an FMA
#: differently still passes, and 1e3 below anything a wrong sign, pad or mask does.
TOL = 1e-12

#: The Bloch phases of the complex scene, the ones gate 4 runs.
BLOCH_PHASE = (0.7, -1.1)


def scenes():
    """``{name: (scene, omega)}`` -- the gate 1-4 scenes; ``omega`` is None off an eigenmode.

    The four eigenmode boxes carry the discrete frequency the gates check the recurrence
    at, so the frozen series can be held to the same closed form the live ones are. The
    four mirror scenes are the *half* boxes of the symmetry gate (random seeded initial
    data, PEC far walls): the Symmetry face rules are exercised nowhere else.
    """
    out = {}
    for kind in ('pec', 'pmc'):
        scene, _, omega = cavity_box(kind, N=16, steps=300)
        out[kind] = (scene, omega)
    scene, _, omega = plane_wave_box(None, N=16, steps=300)
    out['periodic'] = (scene, omega)
    scene, _, omega = plane_wave_box(BLOCH_PHASE, N=16, steps=300)
    out['bloch'] = (scene, omega)
    for label, side in (('node', '-'), ('half', '+')):
        for word, parity in (('even', +1), ('odd', -1)):
            half, _, _ = symmetry_pair(side, parity)
            out[f'mirror_{label}_{word}'] = (half, None)
    return out


def sample_indices(shape):
    """``(S, 4)`` cell indices ``(c, k, j, i)`` to freeze of a ``(3, nz, ny, nx)`` field.

    Eight cells per component, chosen for where a defect would show rather than at random:
    the two corners, the centre, a cell on the high-x layer (the E pad's) and one on the
    low-y layer (the H pad's and the PEC masks'), a cell on the last z layer, and two
    interior cells off every symmetry plane of the boxes.
    """
    _, nz, ny, nx = shape
    picks = ((0, 0, 0), (nz - 1, ny - 1, nx - 1), (nz // 2, ny // 2, nx // 2),
             (nz // 3, ny // 2, nx - 1), (nz // 2, 0, nx // 4), (nz - 1, ny // 3, 1),
             (1, 2, 3), (2, ny - 1, nx // 2))
    return np.array([(c, k, j, i) for c in range(3) for k, j, i in picks], dtype=np.int64)


def capture(res, samples):
    """The arrays frozen from one run: ``final__E``, ``final__H``, ``energy``, ``series__*``."""
    E, H = to_numpy(res.final.E), to_numpy(res.final.H)
    idx = tuple(samples.T)
    out = {
        'final__E': np.ascontiguousarray(E[idx]),
        'final__H': np.ascontiguousarray(H[idx]),
        'energy': np.array(float(np.sum(np.abs(E) ** 2) + np.sum(np.abs(H) ** 2))),
    }
    for name in res:
        out[f'series__{name}'] = np.ascontiguousarray(to_numpy(res[name].values))
    return out


def content_digest(arrays):
    """SHA-256 over the numeric arrays of a golden, by name; the ``metadata`` string is excluded."""
    h = hashlib.sha256()
    for key in sorted(arrays):
        if key == 'metadata':
            continue
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode())
        h.update(b'\0' + a.dtype.str.encode() + b'\0' + repr(a.shape).encode() + b'\0')
        h.update(a.tobytes())
    return h.hexdigest()


def relative_gap(got, want):
    """``max |got - want| / max |want|``; the scale is the reference's own peak."""
    got, want = np.asarray(got), np.asarray(want)
    scale = float(np.max(np.abs(want)))
    if scale == 0.0:
        return float(np.max(np.abs(got)))
    return float(np.max(np.abs(got - want)) / scale)
