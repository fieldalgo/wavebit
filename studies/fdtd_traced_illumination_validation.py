"""Sequential full/slow/read-only-freezer gates for geometry gradients under plane-wave illumination."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_traced_illumination_controls'
    validation.main(base='769ad48c', task='traced_illumination', maxfail=1)
