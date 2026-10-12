"""Sequential full/slow/read-only-freezer gates for the sphere centre-tie derivative fix."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_sphere_center_tie_controls'
    validation.main(base='a8eda4e0', task='sphere_center_tie', maxfail=1)
