"""Sequential full/slow/read-only-freezer gates for the bounded finite mixed Debye/oscillator film."""
import sys
from pathlib import Path

import fdtd_polygon_tie_validation as validation


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend(['full', 'slow', 'freezers'])
    validation.LOGS = Path(__file__).resolve().parent/'results/fdtd_hybrid_film_controls'
    validation.main(base='ab02b9deeb27b4084291510c1f6b586fc0ce7d0a', task='hybrid_film', maxfail=1)
