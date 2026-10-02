"""Sequential full/slow/read-only-freezer gates for a traced substrate held to the grazing rule."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_traced_substrate_controls'
    validation.main(base='ea4d0b0d', task='traced_substrate', maxfail=1)
