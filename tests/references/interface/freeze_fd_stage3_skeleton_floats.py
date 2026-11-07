"""Write the float leaves of the stage-3 scenes' frozen skeletons into
``tests/references/fd_golden_stage3_skeleton_floats.json``.

    python tests/references/interface/freeze_fd_stage3_skeleton_floats.py           # compare
    python tests/references/interface/freeze_fd_stage3_skeleton_floats.py --write   # write

``tests/fdtd/test_fd_stage3_backends.py`` holds each scene's plan to the skeleton recorded in
``fd_golden_stage3.npz`` with its floats inside the gate's tolerance
(``tests/references/_plan_skeleton_floats.py`` says why). The frozen floats it needs are not in
the archive, which records only the hash, so they are kept here beside it.

``--write`` refuses unless every scene's live skeleton *is* the archive's, bit for bit: the
floats are taken only where the archive's platform is reproduced (Linux for the committed
archive). The record is then checked by the archive itself, since only the right floats
rebuild the recorded hash. Re-freezing ``fd_golden_stage3.npz`` means running this afterwards
on the same machine.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path
import argparse
import json
import pathlib
import platform
import sys
import warnings

import numpy as np

HERE = repository_path('tests/references/interface')
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))                      # this checkout's wavebit first

import wavebit                                               # noqa: E402
import wavebit.fdtd._legacy as fd                            # noqa: E402
from tests.references._fd_stage3_cases import EXPECTED, GOLDEN, scenes          # noqa: E402
from tests.references._plan_skeleton_floats import encoded_static, float_leaves  # noqa: E402

RECORD = GOLDEN.with_name('fd_golden_stage3_skeleton_floats.json')


def check_this_checkout_is_the_one_imported():
    got = pathlib.Path(wavebit.__file__).resolve()
    if ROOT not in got.parents:
        raise SystemExit(f'wavebit was imported from {got}, not from this checkout ({ROOT}).')


def record():
    """The record of every scene: the archive's skeleton and the live float leaves."""
    with np.load(GOLDEN, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata']))
    frozen = metadata['scenes']
    out, moved = {}, []
    for name, scene in scenes().items():
        with warnings.catch_warnings():
            for category in EXPECTED:
                warnings.simplefilter('ignore', category)
            p = fd.plan(scene)
        if p.skeleton != frozen[name]['skeleton']:
            moved.append(name)
        out[name] = dict(skeleton=frozen[name]['skeleton'],
                         floats=float_leaves(encoded_static(p)))
    if moved:
        raise SystemExit('this platform does not reproduce the archive\'s skeleton for: '
                         + ', '.join(moved) + '. The floats must be taken where it does (the '
                         f'archive was frozen on {metadata["platform"]["system"]}); nothing '
                         'written.')
    return dict(archive=GOLDEN.name, platform=platform.system(),
                what='the float and complex leaves, in order, of each stage-3 scene\'s skeleton '
                     'encoding, taken where the archive\'s skeleton is reproduced bit for bit',
                scenes=out)


def main(argv=None):
    parser = argparse.ArgumentParser(description='record the stage-3 skeleton floats')
    parser.add_argument('--write', action='store_true', help='write the record')
    args = parser.parse_args(argv)
    check_this_checkout_is_the_one_imported()
    live = record()
    if not args.write:
        if RECORD.exists() and json.loads(RECORD.read_text(encoding='utf-8'))['scenes'] == live['scenes']:
            print(f'{RECORD.name} is reproduced; nothing written')
            return 0
        print(f'{RECORD.name} is not reproduced; pass --write to rewrite it')
        return 1
    RECORD.write_text(json.dumps(live, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(f'wrote {RECORD}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
