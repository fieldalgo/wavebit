"""Sequential full/slow/read-only-freezer gates for the C7 corner follow-up (a study and docs)."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_metasurface_corners_controls'
    validation.main(base='d77dfe21', task='metasurface_corners', maxfail=1)
