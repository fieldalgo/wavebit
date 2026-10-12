"""Sequential full, slow and read-only freezer gates; no golden writes."""
import argparse
import sys
from fdtd_state_bytes_support import (ROOT, RESULTS, PREFIX, BASE, load, require,
    verify_guard, attempt_record, pytest_pair, run_command, write, finish)


def main(attempt):
    setup = load(RESULTS/(PREFIX+'prevalidation.json'))
    require(setup['base'] == BASE, 'Wrong prepared base.')
    verify_guard(setup['guard'])
    directory, manifest, record = attempt_record('validation', attempt)
    files = [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'tests').glob('test_fd_*.py'))]
    for name, extra, targets in [('full', [], files+['tests/test_docs_contract.py']),
                                 ('slow', ['--run-slow', '-m', 'slow'], files)]:
        pytest_pair(record, manifest, name, targets, directory, extra=extra, workers=5)
        verify_guard(setup['guard'])
    for stage, expected in enumerate((143, 288, 140, 570)):
        row, text = run_command(record, manifest, 'freezer'+str(stage),
            [sys.executable, '-B', f'tests/references/interface/freeze_fd_stage{stage}.py'], directory)
        require(row['exit'] == 0, 'Freezer command failed.')
        require(f'{expected} of {expected} arrays reproduce bit for bit' in text, 'Frozen array mismatch.')
        row['arrays_exact'] = expected
        write(manifest, record)
        verify_guard(setup['guard'])
    finish('validation', manifest, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=int)
    main(parser.parse_args().attempt)
