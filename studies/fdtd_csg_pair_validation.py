"""Run the inherited full/slow/read-only-freezer gates for the pair optimization."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_csg_pair_controls'
    # The generic runner accepts task metadata without changing old defaults.
    validation.main(base='cd5ea836', task='csg_pair')
