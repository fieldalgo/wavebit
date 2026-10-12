"""Measure selected crossed dielectric scenes; run from the repository root.

    python studies/fdtd_alpha_convergence.py --output studies/results/fdtd_alpha.json

This is an internal convergence study, not an independent oracle. Case definitions
are shared with the frozen external-reference tests; no external solver is imported.
Results are written after each run so an interrupted sweep retains its measurements.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests' / 'references'))
from _meep_crossed_cases import CASES, eps_of, margin  # noqa: E402


from studies._fdtd_legacy import legacy_fdtd, legacy_rcwa  # noqa: E402


def channels(result, case):
    iu = [list(result.orders_u).index(m) for m in case['orders_u']]
    iv = [list(result.orders_v).index(m) for m in case['orders_v']]
    answer = {}
    for leg in ('s', 'p'):
        sub = getattr(result, f'{leg}_inc')
        for key in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
            n = case['n_top'] if key in ('r_s', 'r_p', 'R') else case['n_bot']
            mask = np.array([[margin(case, m, l, n) > .05 for m in case['orders_u']]
                             for l in case['orders_v']])
            answer[f'{leg}.{key}'] = np.asarray(getattr(sub, key))[np.ix_(iv, iu)][mask]
    return answer


def gaps(a, b):
    return {kind: max(float(np.max(np.abs(a[k] - b[k]))) for k in a
                      if (k.endswith('.R') or k.endswith('.T')) == (kind == 'power'))
            for kind in ('amplitude', 'power')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = [Path(__file__), ROOT / 'tests/references/_meep_crossed_cases.py',
             *sorted((ROOT / 'src/wavebit').rglob('*.py'))]
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
                  scope='Two lossless crossed dielectric rasters, two linear basis sources; '
                        'propagating channels with n^2-k_parallel^2 > 0.05 only.',
                  settings=dict(n_periods=120, n_pml=12, courant=.99, subpixel=True,
                                pad_scale=1., grading=None), cases={})

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    for name, case in CASES.items():
        common = dict(wavelength=case['wavelength'], period=list(case['period']),
                      thickness=list(case['thickness']), structure=eps_of(case['index']),
                      top=eps_of(case['n_top']), bot=eps_of(case['n_bot']),
                      theta=np.deg2rad(case['theta_deg']), phi=np.deg2rad(case['phi_deg']),
                      psi=np.deg2rad(30.))
        refs = {}
        entry = report['cases'][name] = dict(runs=[])
        for order in (9, 13):
            start = time.perf_counter()
            refs[order] = channels(legacy_rcwa(backend='numpy', truncation_order=dict(x=order,y=order),
                                              **common).solve(), case)
            print(name, 'RCWA', order, time.perf_counter()-start, flush=True)
        entry['reference_9_to_13'] = gaps(refs[9], refs[13])
        for resolution in ((50/3, 25., 100/3) if name == 'square_18deg' else (20., 30., 40.)):
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result, diag = legacy_fdtd(**common, **report['settings'], resolution=resolution,
                                           return_diagnostics=True)
            grid = diag['grid']
            entry['runs'].append(dict(resolution=resolution, seconds=time.perf_counter()-start,
                                     nx=grid.nx, ny=grid.ny, nz=grid.nz,
                                     dx=grid.dx, dy=grid.dy, dz_min=diag['dz'], dz_max=diag['dz_max'],
                                     dt=diag['dt'], n_steps=diag['n_steps'],
                                     pml_thickness=diag['pml_thickness'], residual=diag['residual'],
                                     warnings=[str(w.message) for w in caught],
                                     error_vs_rcwa13=gaps(channels(result, case), refs[13])))
            save()
            print(name, entry['runs'][-1], flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
