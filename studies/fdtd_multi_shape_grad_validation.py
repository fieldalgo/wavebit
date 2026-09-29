"""Sequential full/slow/read-only-freezer gates for multiple ranged geometry gradients."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_multi_shape_grad_controls'
    validation.main(base='8872059a', task='multi_shape_grad', maxfail=1)
