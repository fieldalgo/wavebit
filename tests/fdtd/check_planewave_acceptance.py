"""Run the fixed PlaneWave measurement qualification campaign without changing goldens.

Use ``PYTHONPATH=src:. python -m tests.fdtd.check_planewave_acceptance
--output NEW_DIRECTORY``. Partial campaigns cannot produce an overall PASS.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import warnings

import numpy as np
import wavebit
import wavebit.fdtd._legacy as fd
from wavebit.fdtd._assemble import _scene_period
from tests.fixtures.fdtd.fd_helpers import bloch_stack, conical_grating
from tests.fixtures.fdtd.conductive_contact import build as contact
from tests.fixtures.fdtd.acceptance import TIME_FACTORS, assess, maximum_gap, overall
from tests.fixtures.fdtd.acceptance_diagnostics import diagnostic_basis


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, complex):
        return dict(real=serial(value.real), imag=serial(value.imag))
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(serial(data), stream, indent=2, allow_nan=False)
        stream.write('\n')


def fixture(kind, n, factor=1., variant='base'):
    args = dict(cells_per_wavelength=n, polarization='p',
                periods=(20 if kind == 'planar' else 160)*factor,
                pml=24 if variant == 'thick' else 12)
    if kind == 'planar':
        scene, info = bloch_stack(**args, substrate=1.44**2, transverse=(2/n, 2/n))
    else:
        scene, info = conical_grating(**args)
    if variant == 'far':
        # Double the clearances from the structure to each inner PML face.
        # Source, monitors, reference planes and material interfaces stay fixed.
        size, center = scene.domain.size, scene.domain.center
        scene = replace(scene, domain=fd.Domain(size=(size[0], size[1], size[2]+2.2),
                                                center=(center[0], center[1], center[2]+.3)))
        if kind == 'planar':
            last = scene.structures[-1]
            scene = replace(scene, structures=scene.structures[:-1] +
                            (replace(last, shape=fd.Slab(z=(last.shape.z[0]-.8,
                                                           last.shape.z[1]))),))
    return scene, info


def run(scene):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fd.run(scene, backend='numpy')
    diagnostics = result.diagnostics
    period = _scene_period(scene, result.plan)
    basis = diagnostic_basis(diagnostics, result.plan.steps*result.plan.dt, period,
                             any(issubclass(w.category, fd.GrowingFieldWarning) for w in caught))
    report = dict(valid=basis is not None, diagnostic_basis=basis, diagnostic_period=period,
                  diagnostics=diagnostics,
                  warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                  steps=result.plan.steps, dt=result.plan.dt, shape=result.plan.shape,
                  scene=repr(scene))
    return result, report


def observations(result, scene, info, kind, n):
    values, expected, check, limits, labels = [], [], [], [], []
    theta = 25 if kind == 'planar' else 15
    truncations = (0, 1) if kind == 'planar' else (15, 30)
    amp = (.005 if n == 80 else .015) if kind == 'planar' else .0355
    power = .0012 if kind == 'planar' else .01
    for i, wavelength in enumerate(scene.monitors[0].wavelengths):
        angle = np.degrees(np.arcsin(wavelength*np.sin(np.radians(theta))))
        refs = [info['rcwa'](i, angle, truncation_order=t) for t in truncations]
        orders = [0] if kind == 'planar' else [-1, 0, 1]
        names = ('r_s', 'r_p', 't_s', 't_p', 'R', 'T')
        for m in orders:
            for name in names:
                side = 'r' if name[0] in 'rR' else 't'
                values.append(complex(np.asarray(getattr(result[side].orders.order(m, 0), name))[i]))
                expected.append(complex(getattr(refs[0].order(m, 0), name)))
                check.append(complex(getattr(refs[1].order(m, 0), name)))
                limits.append(power if name in ('R', 'T') else
                              1e-12 if kind == 'planar' and name.endswith('_s') else amp)
                labels.append(f'{wavelength}:{m}:{name}')
                if kind == 'planar' and name.endswith('_p'):
                    # Relative magnitude and phase have their own original budgets.
                    ratio = values[-1]/expected[-1]
                    ref_ratio = check[-1]/expected[-1]
                    values.extend((abs(ratio), np.angle(ratio)))
                    expected.extend((1., 0.))
                    check.extend((abs(ref_ratio), np.angle(ref_ratio)))
                    limits.extend((.02, .06))
                    labels.extend((f'{wavelength}:{m}:{name}:relative_magnitude',
                                   f'{wavelength}:{m}:{name}:phase'))
        values.append(float(np.sum(result['r'].orders.at(i).R).real +
                            np.sum(result['t'].orders.at(i).T).real))
        expected.append(1.)
        check.append(float(np.sum(refs[1].R).real + np.sum(refs[1].T).real))
        limits.append(1e-4 if kind == 'planar' else 1e-3)
        labels.append(f'{wavelength}:closure')
    return np.array(values), expected, check, limits, labels


def campaign(output, kind, n):
    rows = []
    controls = [(factor, 'base') for factor in TIME_FACTORS] + [(2., 'thick'), (2., 'far')]
    for index, (factor, variant) in enumerate(controls):
        scene, info = fixture(kind, n, factor, variant)
        result, row = run(scene)
        values, expected, check, limits, labels = observations(result, scene, info, kind, n)
        row.update(values=values, factor=factor, variant=variant)
        rows.append(row)
        write(output/f'{kind}-{n}-{index}.json', row)
        print(f'{kind} n={n} factor={factor} variant={variant} valid={row["valid"]}', flush=True)
    # Coarse planar accuracy is judged by the existing grid-convergence test.
    # The same fixed budget still bounds uncertainty at that resolution.
    verdict = assess(rows[:5], rows[5:], expected, check, limits,
                     qualification_only=(kind == 'planar' and n == 20))
    verdict.update(labels=labels, limits=limits, reference=expected, reference_check=check)
    write(output/f'{kind}-{n}-verdict.json', verdict)
    return verdict


def residual(output):
    rows = []
    for periods in (60, 90, 120):
        scene = contact(n=80, periods=periods)
        result, row = run(scene)
        row['values'] = np.array([getattr(result[side].orders.order(0, 0), side+'_p')
                                  for side in ('r', 't')])
        rows.append(row)
        write(output/f'contact-{periods}.json', row)
        print(f'contact periods={periods} valid={row["valid"]}', flush=True)
    finite = all(np.isfinite(row['values']).all() for row in rows)
    gap = float(maximum_gap([r['values'] for r in rows]).max()) if finite else float('inf')
    status = ('BLOCKED' if not finite or not all(r['valid'] for r in rows)
              else 'PASS' if gap <= 2e-7 else 'FAIL')
    return dict(status=status, max_pairwise_gap=gap, limit=2e-7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    inputs = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
              for name in tracked if name and (root/name).is_file() and
              (name.startswith(('src/', 'tests/fixtures/')) or name == 'pyproject.toml')}
    for name in ('tests/fixtures/fdtd/acceptance.py', 'tests/fixtures/fdtd/acceptance_diagnostics.py',
                 'tests/fdtd/check_planewave_acceptance.py'):
        inputs[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    write(args.output/'inputs.json', dict(python=platform.python_version(),
          source=wavebit.__file__, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
          versions={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'jax', 'torch')},
          sha256=inputs, time_factors=TIME_FACTORS, uncertainty_fractions=dict(time=.1, boundary=.1, reference=.05)))
    verdicts = {}
    for kind, resolutions in (('planar', (20, 40, 80)), ('conical', (20, 40))):
        for n in resolutions:
            verdicts[f'{kind}-{n}'] = campaign(args.output, kind, n)
    verdicts['contact'] = residual(args.output)
    status = overall(v['status'] for v in verdicts.values())
    write(args.output/'result.json', dict(status=status,
          scope='measurement qualification only; source, accuracy/convergence, backend, AD and regression gates are separate',
          verdicts=verdicts))
    raise SystemExit(0 if status == 'PASS' else 1)


if __name__ == '__main__':
    main()
