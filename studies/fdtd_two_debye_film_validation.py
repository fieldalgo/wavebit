"""Sequential full/slow/read-only-freezer gates for the bounded two-Debye film."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_two_debye_film_controls'
    validation.main(base='297d5abf', task='two_debye_film', maxfail=1)
