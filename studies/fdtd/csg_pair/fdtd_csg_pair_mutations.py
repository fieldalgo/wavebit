"""Real source mutations for the independent CSG pair feasibility regression."""

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
SOURCE = ROOT/'src/wavebit/fdtd/_shapes/_bodies.py'
LOGS = ROOT/'studies/fdtd/csg_pair/records/fdtd_csg_pair_controls'
TEST = ('tests/fdtd/test_fd_csg_pairs.py::'
        'test_pair_separation_matches_primal_volume_feasibility[float64-numpy]')
MUTATIONS = [
    ('omit_first_endpoint', 'worst = xp.maximum(end_b, end_a)', 'worst = end_b'),
    ('omit_third_knot', 'for k in range(3):\n        den = n_b',
     'for k in range(2):\n        den = n_b'),
    ('contact_holds_volume', 'return worst >= 0', 'return worst > 0'),
    ('wrong_complement', 'other = 1 - t', 'other = t'),
]


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    records = []
    for name, before, after in MUTATIONS:
        original = SOURCE.read_bytes()
        try:
            source = SOURCE.read_text(encoding='utf-8')
            assert source.count(before) == 1
            SOURCE.write_text(source.replace(before, after), encoding='utf-8')
            result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p',
                                     'no:cacheprovider', TEST], cwd=ROOT,
                                    capture_output=True, text=True)
            output = result.stdout+'\n'+result.stderr
            (LOGS/(name+'.log')).write_text(output, encoding='utf-8')
        finally:
            SOURCE.write_bytes(original)
        assert SOURCE.read_bytes() == original
        records.append(dict(mutation=name, test=TEST, exit=result.returncode,
                            source_sha256=hashlib.sha256(original).hexdigest()))
        (LOGS/'witnesses.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
        assert result.returncode == 1 and 'AssertionError' in output, output[-3000:]
        assert 'INTERNALERROR' not in output and 'ERROR collecting' not in output
        print(name, 'killed; source bytes restored', flush=True)


if __name__ == '__main__':
    main()
