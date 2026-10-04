"""Sequential full/slow/read-only-freezer gates for the C7 ridges follow-up (a study and docs)."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_metasurface_ridges_controls'
    validation.main(base='10cbfc77', task='metasurface_ridges', maxfail=1)
