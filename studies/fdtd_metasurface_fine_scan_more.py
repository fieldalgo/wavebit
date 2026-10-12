"""Second reference-only prescan after height .15 showed a peak near .975.

The first scan and its code remain unchanged. Inspect heights .20 and .10 on
the same sampled band and both truncations before selecting a detuned control.
No FDTD values have been computed for these controls.
"""
import fdtd_metasurface_fine_scan as scan


if __name__ == '__main__':
    scan.OUT = scan.HERE / 'results/fdtd_metasurface_fine_scan_more.jsonl'
    scan.SPECS = {'H20': (12., .20), 'H10': (12., .10)}
    scan.main()
