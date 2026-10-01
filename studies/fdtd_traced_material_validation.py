"""Sequential full/slow/read-only-freezer gates for traced material values with geometry."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_traced_material_controls'
    validation.main(base='c0f317f3', task='traced_material', maxfail=1)
