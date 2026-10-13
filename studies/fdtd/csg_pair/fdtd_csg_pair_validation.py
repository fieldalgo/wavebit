"""Run the inherited full/slow/read-only-freezer gates for the pair optimization."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path
import sys

import studies.fdtd.polygon_tie.fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = repository_path('studies/fdtd/csg_pair/records/fdtd_csg_pair_controls')
    # The generic runner accepts task metadata without changing old defaults.
    validation.main(base='cd5ea836', task='csg_pair')
