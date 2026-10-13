"""Record actual allocation/report collections and all targeted outcomes."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
from studies.fdtd.state_bytes.fdtd_state_bytes_support import ROOT, TARGETS, NEW_TEST, snapshot, pytest_pair, attempt_record, write, finish, require


def main(attempt):
    directory, manifest, record = attempt_record('targeted', attempt)
    record['snapshots'] = [snapshot(ROOT/p, directory/(p.replace('/', '__')+'.txt.gz')) for p in TARGETS]
    write(manifest, record)
    new = pytest_pair(record, manifest, 'new_allocation', [NEW_TEST], directory)
    require(new['observed']['collected'] == 72, 'Expected 72 actual allocation parameter cases.')
    require(len({node.split('::')[1].split('[')[0] for node in new['observed']['node_status']}) == 1,
            'Allocation oracle function contract changed.')
    pytest_pair(record, manifest, 'targeted', list(TARGETS), directory, workers=5)
    finish('targeted', manifest, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=int)
    main(parser.parse_args().attempt)
