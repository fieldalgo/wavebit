"""Real production mutations for every C7 metasurface test; source bytes restored after each.

Each mutation edits one line of ``src/wavebit/fdtd``, runs one test, and must make that test fail
on an assertion (exit code 1, an ``AssertionError``, no collection error), not on a crash; the
original bytes are written back and checked before the next. Logs and ``witnesses.json`` go to
``results/fdtd_metasurface_controls``.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_root
import hashlib
import json
import subprocess
import sys

ROOT = repository_root()
LOGS = ROOT / 'studies/cross_solver/metasurface/records/fdtd_metasurface_controls'
TESTS = 'tests/fdtd/test_fd_metasurface.py::'
TRANSLATION = 'test_a_pillar_moved_by_a_cell_moves_each_order_by_its_lateral_phase'
ROWS = 'test_a_metasurface_cell_repeats_its_campaign_row_and_its_distance_from_rcwa'
CONICAL = 'test_the_conical_silicon_box_repeats_its_campaign_row_and_its_distance_from_rcwa'

#: (name, file under src/wavebit/fdtd, before, after, test node, extra pytest arguments)
MUTATIONS = [
    # the u projection reads -k: every order along x is mislabelled
    ('u_projection_conjugated', '_post.py',
     '    Wu = xp.exp(-1j * (pos_u[:, None] * ku[None, :])) / Nu\n',
     '    Wu = xp.exp(1j * (pos_u[:, None] * ku[None, :])) / Nu\n',
     TRANSLATION + '[x]', ()),
    # the v projection reads -k: the (m, n) and (m, -n) orders trade places
    ('v_projection_conjugated', '_post.py',
     '    Wv = xp.exp(-1j * (kv[:, None] * pos_v[None, :])) / Nv\n',
     '    Wv = xp.exp(1j * (kv[:, None] * pos_v[None, :])) / Nv\n',
     TRANSLATION + '[y]', ()),
    # the half-cell stagger is no longer removed: every order's phase moves
    ('stagger_kept_in_the_orders', '_post.py',
     "    pos_u = rt.array((np.arange(Nu) + 0.5 * offsets[0]) * grid.dx, 'float64')\n",
     "    pos_u = rt.array((np.arange(Nu) + 0.0 * offsets[0]) * grid.dx, 'float64')\n",
     ROWS + '[M1-20-p]', ()),
    # (attempt 1 kept the v stagger for M2 and survived: M2's orders are all n = 0, so no v
    # phase reaches it -- the y half is M1's; results/.../*_SURVIVED_attempt1.log)
    # the de-embedding walks each order the wrong way between the monitor and the reference
    ('de_embedding_reversed', '_assemble.py',
     '    shift = xp.exp(1j * (out_dir * kz) * (z_ref - z_m))\n',
     '    shift = xp.exp(-1j * (out_dir * kz) * (z_ref - z_m))\n',
     ROWS + '[M2-25-s]', ()),
    # (attempt 2 made 'staircase' every structure's default: the substrate then asked for a
    # junction it does not have and the plan was refused -- a crash, not an assertion;
    # results/.../*_CRASHED_attempt2.log)
    # a box standing on its substrate mixes its u edge by the v wall's weight
    ('junction_box_walls_swapped', '_prepare/junction.py',
     '        uu = 1/((1-alpha)*mix(xp, eps_g, eps_p, fu, puu) + alpha*eps_s)\n',
     '        uu = 1/((1-alpha)*mix(xp, eps_g, eps_p, fu, pvv) + alpha*eps_s)\n',
     CONICAL, ('--run-slow',)),
]


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    records = []
    for name, relative, before, after, test, extra in MUTATIONS:
        target = TESTS + test
        source = ROOT / 'src/wavebit/fdtd' / relative
        original = source.read_bytes()
        try:
            contents = original.decode('utf-8')
            newline = '\r\n' if '\r\n' in contents else '\n'
            old, new = before.replace('\n', newline), after.replace('\n', newline)
            assert contents.count(old) == 1, (name, contents.count(old))
            source.write_bytes(contents.replace(old, new).encode('utf-8'))
            result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p',
                                     'no:cacheprovider', *extra, target], cwd=ROOT,
                                    capture_output=True, text=True)
            output = result.stdout + '\n' + result.stderr
            (LOGS / (name + '.log')).write_text(output, encoding='utf-8')
        finally:
            source.write_bytes(original)
        assert source.read_bytes() == original
        records.append(dict(mutation=name, source=relative, test=target, extra=list(extra),
                            before=before, after=after, exit=result.returncode,
                            source_sha256=hashlib.sha256(original).hexdigest()))
        (LOGS / 'witnesses.json').write_text(json.dumps(records, indent=2) + '\n',
                                             encoding='utf-8')
        assert result.returncode == 1, output[-3000:]
        assert 'AssertionError' in output, output[-3000:]
        assert not any(s in output for s in ('INTERNALERROR', 'ERROR collecting', 'crashed'))
        print(name, 'killed; source bytes restored', flush=True)


if __name__ == '__main__':
    main()
