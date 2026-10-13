"""Freeze the stage-1 kernel's own answers on the six material and absorber scenes, per
backend, into ``tests/references/fd_golden_stage1.npz`` -- a drift detector, not an oracle.

    python tests/references/interface/freeze_fd_stage1.py               # reproduce and compare
    python tests/references/interface/freeze_fd_stage1.py --rebaseline  # write, printing what moved
    python tests/references/interface/freeze_fd_stage1.py --output NEW.npz   # a fresh archive elsewhere

Run it from the repository root with the project interpreter; it puts this checkout first on
``sys.path`` and refuses to freeze from any other ``wavebit`` (an installed copy pointing at
another checkout is the usual way a golden gets taken from code it does not describe).

The stage-0 freezer beside this one (``freeze_fd_stage0.py``) says what a freezer of our own
numbers can and cannot claim, and every rule there holds here: two freezes in one process must
come back byte for byte before anything is written, each backend is pinned to *its own*
answer so a red test names the backend that moved, source digests are LF-normalised, and
``--rebaseline`` is the only way to overwrite the committed file.

**What is different in stage 1.** The scenes are the six of ``../_fd_stage1_cases.py`` -- a
six-faced PML box, an oblique complex PML line, a dielectric slab, a Drude + Lorentz
half-space running out through an ``Absorber``, a conductive slab and a bare ``Absorber`` box
-- and beside each backend's fields and series the archive carries, once per scene, the
numbers stage 1 added to the lowering: the four update coefficient arrays, every CPML
profile (``b``, ``a``, ``1/kappa``) and every pole's recurrence coefficients. Those are the
same on every backend by construction, so freezing them once separates a moved profile from a
moved kernel in the failure message: if only the ``cpml__*`` rows move, the layer was
regraded; if only the per-backend rows move, the step changed under an unchanged layer.

What the archive cannot say is that any of it is *right*. ``tests/fdtd/test_fd_pml.py`` says the
layer absorbs (the doubling table and its angular-law oracle),
``tests/fdtd/test_fd_materials_doors.py`` says the doors refuse what they claim to, and
``tests/fdtd/test_fd_stage1_grad.py`` says the derivatives are the derivatives. This file says the
numbers have not moved.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path
import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import sys
import warnings

import numpy as np

HERE = repository_path('tests/references/interface')
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))                      # this checkout's wavebit first
sys.path.insert(0, str(HERE.parent))               # tests/references, for the case list

import wavebit                                               # noqa: E402
import wavebit.fdtd._legacy as fd                                      # noqa: E402
from tests.references._fd_stage1_cases import BACKENDS, GOLDEN, TOL, capture, content_digest, lowered, relative_gap, sample_indices, scenes

CASES = HERE.parent / '_fd_stage1_cases.py'
SHARED_CASES = HERE.parent / '_fd_stage0_cases.py'


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
    for name, scene in scenes().items():
        with warnings.catch_warnings():
            # The dispersive half-space reaches its Absorber on purpose, and says so every
            # time it is planned; the gate asserts that warning separately.
            warnings.simplefilter('ignore', fd.AbsorberProximityWarning)
            p = fd.plan(scene)
            samples = sample_indices(p.shape)
            arrays[f'{name}__samples'] = samples
            for key, value in lowered(scene, p, samples).items():
                arrays[f'{name}__{key}'] = value
            meta_scenes[name] = dict(
                skeleton=p.skeleton, cells=[p.nx, p.ny, p.nz], steps=p.steps, dx=p.dx,
                dt=p.dt, dtype=p.dtype, monitors=[m.name for m in scene.monitors],
                has_E0=p.has_E0, has_H0=p.has_H0,
                absorbing=[list(f) for f in p.absorbing_faces],
                poles=sum(t.n_poles for t in p.terms if hasattr(t, 'n_poles')),
                spans=[list(s) for s in p.spans])
            for backend in BACKENDS:
                res = fd.run(scene, backend=backend)
                for key, value in capture(res, samples).items():
                    arrays[f'{name}__{backend}__{key}'] = value
    metadata = dict(
        archive=GOLDEN.name,
        what='the stage-1 kernel (wavebit.fdtd._legacy: PML, Absorber, materials, poles) on the six '
             'scenes of _fd_stage1_cases.py, each backend pinned to its own answer and the '
             'lowered coefficients pinned once; a drift detector, not an oracle',
        frozen=datetime.date.today().isoformat(),
        versions=_versions(), platform=_platform(),
        generator_sha256=_source_digest(pathlib.Path(__file__)),
        cases_sha256=_source_digest(CASES),
        shared_cases_sha256=_source_digest(SHARED_CASES),
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
    exact, inside, outside, missing, reshaped = [], [], [], [], []
    for key in sorted(numeric):
        if key not in committed:
            missing.append(key)
            continue
        if (committed[key].shape != arrays[key].shape
                or committed[key].dtype != arrays[key].dtype):
            # A changed shape is a changed *scene*, and comparing the two would raise rather
            # than report. The stage-0 freezer's compare has no such branch because its cases
            # have never changed shape; this one says so instead of tracing back.
            reshaped.append((key, committed[key].shape, arrays[key].shape))
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
    for key, was, now in reshaped:
        print(f'  {key}: frozen at shape {was}, now {now} -- the scene moved, not the numbers')
    for key, gap in inside:
        print(f'  {key}: moved {gap:.2e} of its peak -- inside the {TOL:g} budget; not a '
              f'reason to rebaseline')
    for key, gap in outside:
        print(f'  {key}: moved {gap:.2e} of its peak -- OUTSIDE the {TOL:g} budget')
    if old['versions'] != metadata['versions']:
        print(f'  versions moved: frozen {old["versions"]}, now {metadata["versions"]}')
    for key, what in (('generator_sha256', 'this freezer'), ('cases_sha256', 'the case list'),
                      ('shared_cases_sha256', 'the shared stage-0 case list')):
        if old.get(key) != metadata[key]:
            print(f'  {what} has changed since the freeze (provenance only; the arrays above '
                  f'say whether the numbers moved)')
    return not outside and not missing and not reshaped


def main(argv=None):
    parser = argparse.ArgumentParser(description='freeze fd_golden_stage1.npz; see the module')
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
          f'{len(BACKENDS)} backends, twice, byte-identical; content '
          f'{metadata["content_sha256"][:12]}')

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
