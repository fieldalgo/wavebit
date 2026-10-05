"""Sequential regression and read-only freezer gates for the 640-cell ridge study."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_metasurface_fine_controls'
    validation.main(base='9f67bbf2', task='metasurface_fine', maxfail=1)
