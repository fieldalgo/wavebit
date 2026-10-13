"""Second reference-only prescan after height .15 showed a peak near .975.

The first scan and its code remain unchanged. Inspect heights .20 and .10 on
the same sampled band and both truncations before selecting a detuned control.
No FDTD values have been computed for these controls.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import studies.cross_solver.metasurface_fine.fdtd_metasurface_fine_scan as scan


if __name__ == '__main__':
    scan.OUT = scan.HERE / 'results/fdtd_metasurface_fine_scan_more.jsonl'
    scan.SPECS = {'H20': (12., .20), 'H10': (12., .10)}
    scan.main()
