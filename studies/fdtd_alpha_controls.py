"""Separate runtime, absorber depth and padding sensitivity at fixed spatial grids.

Run from the repository root:
    python studies/fdtd_alpha_controls.py --output studies/results/fdtd_alpha_controls.json

Uses the same two dielectric scenes and channel masks as fdtd_alpha_convergence.py.
Each control changes one parameter from the 240-cycle baseline. This is an internal
sensitivity measurement, not an independent oracle or a universal accuracy guarantee.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import warnings

from fdtd_alpha_convergence import ROOT, CASES, channels, eps_of, gaps, np, wavebit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = [Path(__file__), ROOT / 'studies/fdtd_alpha_convergence.py',
             ROOT / 'tests/references/_meep_crossed_cases.py',
             *sorted((ROOT / 'wavebit').rglob('*.py'))]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    report = dict(schema=1, source_sha256=digest.hexdigest(),
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                   text=True).strip(),
                  python=platform.python_version(), platform=platform.platform(),
                  versions={n: importlib.metadata.version(n)
                            for n in ('numpy', 'scipy', 'array-api-compat')},
                  scope='Two crossed lossless dielectric scenes; propagating channels only; '
                        'no extrapolation to other geometries or materials.', cases={})
    for name, case in CASES.items():
        common = dict(wavelength=case['wavelength'], period=list(case['period']),
                      thickness=list(case['thickness']), structure=eps_of(case['index']),
                      top=eps_of(case['n_top']), bot=eps_of(case['n_bot']),
                      theta=np.deg2rad(case['theta_deg']), phi=np.deg2rad(case['phi_deg']),
                      psi=np.deg2rad(30.))
        ref = channels(wavebit.rcwa(backend=0, truncation_order=dict(x=13, y=13),
                                   **common).solve(), case)
        entry = report['cases'][name] = dict(runs={}, comparisons={})
        solutions = {}
        controls = [('time120', 120, 12, 1.), ('time180', 180, 12, 1.),
                    ('time240', 240, 12, 1.), ('pml20', 240, 20, 1.),
                    ('pad1.5', 240, 12, 1.5)]
        for label, cycles, pml, pad in controls:
            settings = dict(resolution=25. if name == 'square_18deg' else 30.,
                            n_periods=cycles, n_pml=pml, pad_scale=pad,
                            courant=.99, subpixel=True, grading=None)
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result, diag = wavebit.fdtd(**common, **settings, return_diagnostics=True)
            solutions[label] = channels(result, case)
            grid = diag['grid']
            entry['runs'][label] = dict(settings=settings, seconds=time.perf_counter()-start,
                                       nx=grid.nx, ny=grid.ny, nz=grid.nz,
                                       dx=grid.dx, dy=grid.dy, dz_min=diag['dz'],
                                       dz_max=diag['dz_max'], dt=float(diag['dt']),
                                       n_steps=diag['n_steps'], pml_thickness=diag['pml_thickness'],
                                       residual=diag['residual'],
                                       warnings=[str(w.message) for w in caught],
                                       error_vs_rcwa13=gaps(solutions[label], ref))
            for a, b in [('time120', 'time180'), ('time180', 'time240'),
                         ('time240', 'pml20'), ('time240', 'pad1.5')]:
                if a in solutions and b in solutions:
                    entry['comparisons'][f'{a}_to_{b}'] = gaps(solutions[a], solutions[b])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(name, label, entry['runs'][label], entry['comparisons'], flush=True)
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
