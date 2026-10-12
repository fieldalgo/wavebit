"""Sequential full/slow/read-only-freezer gates for the CI findings fixed before landing."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_ci_findings_controls'
    validation.main(base='8827c9e8', task='ci_findings', maxfail=1)
