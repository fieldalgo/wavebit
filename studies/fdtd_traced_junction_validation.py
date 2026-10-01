"""Sequential full/slow/read-only-freezer gates for a traced pillar standing on a substrate."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_traced_junction_controls'
    validation.main(base='68d92fe5', task='traced_junction', maxfail=1)
