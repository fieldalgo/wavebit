"""Current-source conductive-stack and compatibility baselines, with raw per-node reports."""
import argparse
from fdtd_conductive_stack_support import (NEW_TEST, TARGETS,
    attempt_record, pytest_pair, finish)


def main(attempt):
    directory, manifest, record = attempt_record('targeted', attempt)
    pytest_pair(record, manifest, 'new_stack', [NEW_TEST], directory)
    pytest_pair(record, manifest, 'targeted', list(TARGETS), directory)
    finish('targeted', manifest, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=int)
    main(parser.parse_args().attempt)
