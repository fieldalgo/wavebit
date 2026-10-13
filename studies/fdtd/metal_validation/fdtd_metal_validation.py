"""Measure the negative-epsilon reference and separate Wavebit time/boundary controls.

Run in the development environment (uses the reference test comparison helpers):
    python studies/fdtd/metal_validation/fdtd_metal_validation.py --output studies/fdtd/metal_validation/records/fdtd_metal_validation.json

The external NPZ must already exist. No external solver is imported or regenerated.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_root
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

import numpy as np

ROOT = repository_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests/references'))
from tests.references._meep_crossed_metal_case import CASE  # noqa: E402
from tests.references._meep_crossed_material_cases import epsilon  # noqa: E402
from tests.references.test_meep_crossed_materials_golden import scene, extract, gap, frozen  # noqa: E402


from studies._support._fdtd_legacy import legacy_fdtd, legacy_rcwa  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = [Path(__file__), ROOT / 'tests/references/_meep_crossed_metal_case.py',
             ROOT / 'tests/references/_meep_crossed_material_cases.py',
             ROOT / 'tests/references/_meep_crossed_cases.py',
             ROOT / 'tests/references/test_meep_crossed_materials_golden.py',
             *sorted((ROOT / 'src/wavebit').rglob('*.py'))]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    report = dict(complete=False, source_sha256=digest.hexdigest(),
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                   text=True).strip(),
                  python=platform.python_version(), platform=platform.platform(),
                  versions={n: importlib.metadata.version(n) for n in ('numpy', 'scipy')},
                  runs={}, comparisons={})
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    kw = scene(CASE)
    kw['structure'] = epsilon(CASE, CASE['wavelength'])
    refs = {}
    for order in (7, 11):
        kw['truncation_order'] = dict(x=order, y=order)
        refs[order] = extract(legacy_rcwa(backend='numpy', **kw).solve(), CASE)
    report['rcwa7_to_11'] = gap(refs[7], refs[11], CASE)
    kw['structure'] = CASE['index']**2
    omitted = extract(legacy_rcwa(backend='numpy', **kw).solve(), CASE)
    solutions = {}
    for label, resolution, cycles, pml, pad in [
        ('coarse', 30., 160, 12, 1.), ('fine', 60., 160, 12, 1.),
        ('time240', 60., 240, 12, 1.), ('pml20', 60., 240, 20, 1.),
        ('pad1.5', 60., 240, 12, 1.5),
    ]:
        start = time.perf_counter()
        settings = dict(resolution=resolution, n_periods=cycles, n_pml=pml, pad_scale=pad,
                        courant=.99, subpixel=True, grading=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result, diag = legacy_fdtd(**scene(CASE), dispersion=CASE['dispersion'],
                                       poles=CASE['poles'], **settings, return_diagnostics=True)
        solutions[label] = extract(result, CASE)
        entry = report['runs'][label] = dict(settings=settings, seconds=time.perf_counter()-start,
                                             nx=diag['grid'].nx, ny=diag['grid'].ny, nz=diag['nz'],
                                             dx=diag['grid'].dx, dy=diag['grid'].dy,
                                             dz=diag['dz'], dt=float(diag['dt']),
                                             n_steps=diag['n_steps'], residual=diag['residual'],
                                             pml_thickness=diag['pml_thickness'],
                                             warnings=[str(w.message) for w in caught],
                                             error_vs_rcwa11=gap(solutions[label], refs[11], CASE),
                                             absorbed={leg: float(1-np.sum(solutions[label][f'{leg}.R'])
                                                                   -np.sum(solutions[label][f'{leg}.T']))
                                                       for leg in ('s', 'p')})
        for a, b in [('fine', 'time240'), ('time240', 'pml20'), ('time240', 'pad1.5')]:
            if a in solutions and b in solutions:
                report['comparisons'][f'{a}_to_{b}'] = gap(solutions[a], solutions[b], CASE)
        save()
        print(label, entry, report['comparisons'], flush=True)
    reference = ROOT / 'tests/references/golden_meep_crossed_metal.npz'
    report['reference_sha256'] = hashlib.sha256(reference.read_bytes()).hexdigest()
    with np.load(reference) as data:
        assert data['complete']
        gold = frozen(data, 'metal', 60)
        report['meep60_vs_rcwa11'] = gap(gold, refs[11], CASE)
        report['meep30_vs_rcwa11'] = gap(frozen(data, 'metal', 30), refs[11], CASE)
        report['omitted_vs_meep60'] = gap(omitted, gold, CASE)
        for label, values in solutions.items():
            report['runs'][label]['error_vs_meep60'] = gap(values, gold, CASE)
            report['runs'][label]['power_error_vs_meep60'] = max(
                float(np.max(np.abs(values[key] - gold[key]))) for key in values
                if key.endswith(('.R', '.T')))
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
