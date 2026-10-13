"""Sequential full/slow/read-only-freezer gates for the bounded finite mixed Debye/oscillator film."""

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
    validation.LOGS = repository_path('studies/fdtd/hybrid_film/records/fdtd_hybrid_film_controls')
    validation.main(base='ab02b9deeb27b4084291510c1f6b586fc0ce7d0a', task='hybrid_film', maxfail=1)
