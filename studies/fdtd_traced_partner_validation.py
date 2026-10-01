"""Sequential full/slow/read-only-freezer gates for traced bodies beside concrete partners."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_traced_partner_controls'
    validation.main(base='bd149464', task='traced_partner', maxfail=1)
