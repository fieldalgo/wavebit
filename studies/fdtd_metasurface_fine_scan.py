"""Reference-only band scan before the fine y-ridge campaign; no FDTD rows.

Keep Y12's width, period, angle and substrate. Compare epsilon 4 at the same
height and epsilon 12 at half the height. The height choice precedes this scan.
The scan can establish a smooth sampled band, not the absence of all resonances.
Run from the checkout root with the numerical thread counts set to one.
"""
import hashlib
import json
import time
from pathlib import Path

import fdtd_metasurface_ridges as ridge


HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/fdtd_metasurface_fine_scan.jsonl'
SPECS = {'Y12': (12., .30), 'Y4': (4., .30), 'D12': (12., .15)}
WAVELENGTHS = tuple(.90 + .0125 * i for i in range(17))


def main():
    if OUT.exists():
        raise SystemExit('The reference scan already exists; preserve it.')
    source = hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        for name, (eps, height) in SPECS.items():
            for n in (40, 80):
                for wl in WAVELENGTHS:
                    ridge.EPS_P, ridge.H = eps, height
                    start = time.perf_counter()
                    v = ridge.rcwa_point('Y', wl, n)
                    row = dict(scene=name, eps=eps, height=height, wavelength=wl,
                               N=n, source=source, wall=time.perf_counter()-start,
                               values={k: [x.real, x.imag] for k, x in v.items()})
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    print(name, n, f'{wl:.4f}',
                          ' '.join(f'{v[k].real:.8f}' for k in ('s|r|R', 'p|r|R')),
                          flush=True)


if __name__ == '__main__':
    main()
