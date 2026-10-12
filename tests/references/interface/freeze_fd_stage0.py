"""Freeze the stage-0 kernel's own answers on the gate 1-4 scenes, per backend, into
``tests/references/fd_golden_stage0.npz`` -- a drift detector, not an oracle.

    python tests/references/interface/freeze_fd_stage0.py               # reproduce and compare
    python tests/references/interface/freeze_fd_stage0.py --rebaseline  # write, printing what moved
    python tests/references/interface/freeze_fd_stage0.py --output NEW.npz   # a fresh archive elsewhere

Run it from the repository root with the project interpreter; it puts this checkout first on
``sys.path`` and refuses to freeze from any other ``wavebit`` (an installed copy pointing at
another checkout is the usual way a golden gets taken from code it does not describe).

**What this file can and cannot say.** Every number in the archive comes from
``wavebit.fd`` itself, so the archive cannot say the kernel is right -- gates 1 to 4 do
that, against closed forms (``tests/test_fd_pec.py``, ``test_fd_pmc.py``,
``test_fd_symmetry.py``, ``test_fd_bloch.py``). What it says is that the kernel has not
*moved*, on each backend separately, and that is a claim cross-backend agreement cannot
make: a defect that moves NumPy, JAX and Torch together -- a pad sign, a dropped mask, a
wall one cell off -- leaves all three agreeing to 1e-15. ``tests/test_fd_backends.py`` holds
both halves, agreement and the pin, and its mutant is a plan with its masks dropped, which
passes the first half and fails the second on every backend.

**Per backend, on purpose.** One shared golden would pin the three to a common number and
attribute a Torch-only regression to whichever backend the reader ran first. Each backend is
held to what it produced, so a red test names the backend that moved.

**The digest is validated before it is trusted.** The freeze runs twice, in one process,
and refuses to write unless every array and the metadata come back byte for byte -- a
freeze whose output depends on the process (an ``id()``, an address in a ``repr``, an
unseeded draw) would otherwise be committed as if it were a function of the code. The
archive then records a SHA-256 over its own numeric content, which the test recomputes.
Source digests are taken over LF-normalised bytes so a checkout with ``autocrlf`` freezes
the same value as one without.

**Regenerating is not how a failing test gets fixed.** Without ``--rebaseline`` this script
never writes over the committed file: it reproduces the archive and prints, array by array,
what moved and by how much against the test's budget. A change that moves these numbers on
purpose is re-frozen with ``--rebaseline`` and the reason goes in the commit message.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))                      # this checkout's wavebit first
sys.path.insert(0, str(HERE.parent))               # tests/references, for the case list

import wavebit                                               # noqa: E402
import wavebit.fd as fd                                      # noqa: E402
from _fd_stage0_cases import (BACKENDS, GOLDEN, TOL, capture, content_digest,   # noqa: E402
                              relative_gap, sample_indices, scenes)

CASES = HERE.parent / '_fd_stage0_cases.py'


def check_this_checkout_is_the_one_imported():
    """Refuse to freeze from a ``wavebit`` that is not the one beside this script."""
    got = pathlib.Path(wavebit.__file__).resolve()
    if ROOT not in got.parents:
        raise SystemExit(f'wavebit was imported from {got}, not from this checkout ({ROOT}); '
                         f'a golden must be frozen from the code it describes. Run from the '
                         f'repository root with this checkout first on sys.path.')


def _source_digest(path):
    """SHA-256 of a source file with CRLF folded to LF, so line endings do not move it."""
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def _versions():
    import array_api_compat
    import jax
    import torch
    return dict(python=platform.python_version(), wavebit=wavebit.__version__,
                numpy=np.__version__, array_api_compat=array_api_compat.__version__,
                jax=jax.__version__, torch=torch.__version__)


def _platform():
    import jax
    return dict(system=platform.system(), machine=platform.machine(),
                processor=platform.processor(), jax_device=jax.devices()[0].platform,
                torch_device='cpu')


def freeze():
    """``(arrays, metadata)`` from a fresh run of every scene on every backend."""
    arrays, meta_scenes = {}, {}
    for name, (scene, omega) in scenes().items():
        p = fd.plan(scene)
        samples = sample_indices(p.shape)
        arrays[f'{name}__samples'] = samples
        idx = tuple(samples.T)
        # The traced side of the scene, from the NumPy lowering: the initial field at the
        # sampled cells. The skeleton pins everything static about the scene; this pins the
        # excitation, so a moved helper amplitude reads as a moved scene and not as drift.
        params = fd.lower(scene, p, 'numpy')
        for field in ('E', 'H'):
            F0 = params[f'{field}0']
            if F0 is not None:
                arrays[f'{name}__initial__{field}'] = np.ascontiguousarray(np.asarray(F0)[idx])
        meta_scenes[name] = dict(
            skeleton=p.skeleton, cells=[p.nx, p.ny, p.nz], steps=p.steps, dx=p.dx, dt=p.dt,
            dtype=p.dtype, monitors=[m.name for m in scene.monitors], has_E0=p.has_E0,
            has_H0=p.has_H0, omega=omega)
        for backend in BACKENDS:
            res = fd.run(scene, backend=backend)
            for key, value in capture(res, samples).items():
                arrays[f'{name}__{backend}__{key}'] = value
    metadata = dict(
        archive=GOLDEN.name,
        what='the stage-0 kernel (wavebit.fd) on the gate 1-4 scenes, each backend pinned to '
             'its own answer; a drift detector, not an oracle',
        frozen=datetime.date.today().isoformat(),
        versions=_versions(), platform=_platform(),
        generator_sha256=_source_digest(pathlib.Path(__file__)),
        cases_sha256=_source_digest(CASES),
        backends=list(BACKENDS), tolerance=TOL, scenes=meta_scenes,
        content_sha256=content_digest(arrays))
    return arrays, metadata


def _payload(arrays, metadata):
    return dict(arrays, metadata=np.array(json.dumps(metadata, sort_keys=True, indent=1)))


def check_two_freezes_agree(first, second):
    """The digest's own validation: the same code, twice, must give the same bytes."""
    a, ma = first
    b, mb = second
    moved = sorted(k for k in set(a) | set(b)
                   if k not in a or k not in b or a[k].tobytes() != b[k].tobytes()
                   or a[k].dtype != b[k].dtype or a[k].shape != b[k].shape)
    if ma != mb:
        moved.append('metadata')
    if moved:
        raise SystemExit('two freezes of the same code disagree on: ' + ', '.join(moved)
                         + '\nA golden cannot be taken from a freeze that depends on the '
                           'process; find what varies before writing anything.')


def compare(arrays, metadata, committed):
    """Print how ``arrays`` sit against the committed archive; True if inside the budget."""
    old = json.loads(str(committed['metadata']))
    for name, meta in metadata['scenes'].items():
        was = old['scenes'].get(name, {}).get('skeleton')
        if was != meta['skeleton']:
            print(f'  {name}: the scene is not the frozen one (skeleton {meta["skeleton"][:12]} '
                  f'against {str(was)[:12]})')
    numeric = [k for k in arrays if k != 'metadata']
    exact, inside, outside, missing = [], [], [], []
    for key in sorted(numeric):
        if key not in committed:
            missing.append(key)
            continue
        if committed[key].tobytes() == arrays[key].tobytes():
            exact.append(key)
            continue
        gap = relative_gap(arrays[key], committed[key])
        (inside if gap <= TOL else outside).append((key, gap))
    for key in sorted(k for k in committed.files if k not in arrays and k != 'metadata'):
        print(f'  committed but no longer frozen: {key}')
    for key in missing:
        print(f'  frozen but not committed: {key}')
    print(f'  {len(exact)} of {len(numeric)} arrays reproduce bit for bit')
    for key, gap in inside:
        print(f'  {key}: moved {gap:.2e} of its peak -- inside the {TOL:g} budget; not a '
              f'reason to rebaseline')
    for key, gap in outside:
        print(f'  {key}: moved {gap:.2e} of its peak -- OUTSIDE the {TOL:g} budget')
    if old['versions'] != metadata['versions']:
        print(f'  versions moved: frozen {old["versions"]}, now {metadata["versions"]}')
    for key, what in (('generator_sha256', 'this freezer'), ('cases_sha256', 'the case list')):
        if old.get(key) != metadata[key]:
            print(f'  {what} has changed since the freeze (provenance only; the arrays above '
                  f'say whether the numbers moved)')
    return not outside and not missing


def main(argv=None):
    parser = argparse.ArgumentParser(description='freeze fd_golden_stage0.npz; see the module')
    parser.add_argument('--rebaseline', action='store_true',
                        help='write over the committed archive (the reason goes in the commit)')
    parser.add_argument('--output', type=pathlib.Path, default=None,
                        help='write a fresh archive here instead; refuses an existing file')
    args = parser.parse_args(argv)
    check_this_checkout_is_the_one_imported()

    first, second = freeze(), freeze()
    check_two_freezes_agree(first, second)
    arrays, metadata = first
    print(f'froze {len(arrays)} arrays from {len(metadata["scenes"])} scenes on '
          f'{len(BACKENDS)} backends, twice, byte-identical; content {metadata["content_sha256"][:12]}')

    if args.output is not None:
        if args.output.exists():
            raise SystemExit(f'{args.output} exists; --output writes a fresh archive to a new '
                             f'path and never over one.')
        with args.output.open('xb') as target:
            np.savez(target, **_payload(arrays, metadata))
        print(f'wrote {args.output}')
        return 0
    if GOLDEN.exists():
        with np.load(GOLDEN, allow_pickle=False) as committed:
            print(f'against the committed {GOLDEN.name}:')
            ok = compare(arrays, metadata, committed)
        if not args.rebaseline:
            if ok:
                print('the committed golden is reproduced; nothing written')
                return 0
            print('the committed golden is NOT reproduced. Regenerating is not how a failing '
                  'test gets fixed: find what moved the numbers, and if the move is the point, '
                  'pass --rebaseline and say why in the commit message.')
            return 1
    np.savez(GOLDEN, **_payload(arrays, metadata))
    print(f'wrote {GOLDEN}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
