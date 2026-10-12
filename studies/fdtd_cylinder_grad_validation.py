"""Required full/slow/read-only-freezer checks for public cylinder geometry gradients."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_cylinder_grad_controls'
    validation.main(base='b9cdecf2', task='cylinder_grad', maxfail=1)
