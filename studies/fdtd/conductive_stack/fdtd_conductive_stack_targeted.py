"""Current-source conductive-stack and compatibility baselines, with raw per-node reports."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
from studies.fdtd.conductive_stack.fdtd_conductive_stack_support import NEW_TEST, TARGETS, attempt_record, pytest_pair, finish


def main(attempt):
    directory, manifest, record = attempt_record('targeted', attempt)
    pytest_pair(record, manifest, 'new_stack', [NEW_TEST], directory)
    pytest_pair(record, manifest, 'targeted', list(TARGETS), directory)
    finish('targeted', manifest, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=int)
    main(parser.parse_args().attempt)
