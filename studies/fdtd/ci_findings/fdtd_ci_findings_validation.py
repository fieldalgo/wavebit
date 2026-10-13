"""Sequential full/slow/read-only-freezer gates for the CI findings fixed before landing."""

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
    validation.LOGS = repository_path('studies/fdtd/ci_findings/records/fdtd_ci_findings_controls')
    validation.main(base='8827c9e8', task='ci_findings', maxfail=1)
