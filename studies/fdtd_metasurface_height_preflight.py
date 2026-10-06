"""Readiness evidence for a possible M3 height study; no new field solve.

This reads already published M3 rows, checks the public height-doubled scene's
layout and lowest-resolution plan, and estimates cost from previous timings.
The historical cellwise analysis is post hoc. Its intervals include only the
existing 80-cell run-length check, NOT an absorber check, and must not be used
as fully controlled classifications of a new experiment. No tall-box protocol
is registered and no new FDTD or RCWA response is measured by this script.
"""
from collections import Counter
import ctypes
import hashlib
import json
import math
from pathlib import Path
import subprocess

from scipy.optimize import brentq

import fdtd_metasurface_rcwa as c7
import fdtd_metasurface_lifted as lifted


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'studies/results/fdtd_metasurface_height_preflight.json'
RUNGS = (40, 80, 120)
SAME = 1e-10


def ratio_at(p):
    """Unequal-rung change ratio, including its removable p=0 limit."""
    x, y = math.log(2), math.log(1.5)
    return x/y if p == 0 else math.expm1(p*x)/(-math.expm1(-p*y))


def order_at(ratio):
    if not math.isfinite(ratio) or not ratio_at(-64) < ratio < ratio_at(64):
        return None
    return brentq(lambda p: ratio_at(p)-ratio, -64, 64, xtol=1e-12)


def interval(a, b, uncertainty):
    if min(a, b) <= max(SAME, uncertainty):
        return None
    ends = (order_at((a-uncertainty)/(b+uncertainty)),
            order_at((a+uncertainty)/(b-uncertainty)))
    return ends if all(p is not None for p in ends) else None


def classify(bounds):
    if bounds is None:
        return 'unresolved'
    if 1.8 <= bounds[0] and bounds[1] <= 2.3:
        return 'second'
    return 'low' if bounds[1] <= 1.75 else 'mixed'


def memory():
    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ('physical', 'available_physical', 'pagefile', 'available_pagefile',
             'virtual', 'available_virtual', 'available_extended')]
    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('GlobalMemoryStatusEx failed')
    return {k: getattr(status, k) for k in
            ('physical', 'available_physical', 'pagefile', 'available_pagefile')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def main():
    # These are mathematical instrument checks, not new production pytest cases.
    for p in (-2., -.5, 0., 1., 1.444, 2., 3.):
        assert math.isclose(order_at(ratio_at(p)), p, abs_tol=1e-11)
    for p in (-2., -.5, 1., 1.444, 2., 3.):
        values = [complex(.3, .7)*n**(-p) for n in RUNGS]
        a, b = (abs(x-y) for x, y in zip(values, values[1:]))
        bounds = interval(a, b, 0.)
        assert bounds is not None and all(math.isclose(x, p, abs_tol=1e-10) for x in bounds)
    assert interval(0., 0., 0.) is None
    assert interval(1., 1., 1.) is None
    c7.PROV = c7.provenance()
    registration_path = ROOT / 'studies/results/fdtd_metasurface_rcwa_registration.json'
    registration = json.loads(registration_path.read_text())
    assert registration['sha256'] == sha(c7.__file__)
    assert registration['digest'] == c7.PROV['digest']
    rows = c7.load(c7.LOG)
    result = dict(scope=__doc__, commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        sources={Path(p).relative_to(ROOT).as_posix(): sha(p) for p in
                 (Path(__file__), Path(c7.__file__), Path(lifted.__file__), c7.LOG,
                  c7.RCWA_LOG, registration_path)},
        historical_cells={}, layout=[], memory=memory(),
        no_new_field_solve=True, protocol_registered=False)
    runs_by_pol = {}
    for pol in c7.POLS:
        runs = {n: c7.rung_run(rows, 'M3', n, pol)[0] for n in RUNGS}
        runs_by_pol[pol] = runs
        long = rows[c7.key_of(c7.spec('M3', 80, pol, periods=180))]
        assert all(not c7.validity('M3', r) for r in [*runs.values(), long])
        values = [lifted.cells('M3', runs[n], n) for n in RUNGS]
        longer = lifted.cells('M3', long, 80)
        entries = []
        for key in values[0]:
            a = abs(values[0][key]-values[1][key])
            b = abs(values[1][key]-values[2][key])
            noise = abs(longer[key]-values[1][key])
            share = 2*noise/min(a, b) if min(a, b) > SAME else None
            bounds = interval(a, b, 2*noise) if share is not None and share <= .1 else None
            entries.append(dict(observable=key[0], wavelength=c7.WAVELENGTHS[key[1]],
                                changes=[a,b], time_proxy=noise, time_share=share,
                                self_order=order_at(a/b) if min(a,b)>SAME else None,
                                time_only_interval=bounds, time_only_class=classify(bounds)))
        assert len(entries) == 54
        result['historical_cells'][pol] = entries
    original_height = c7.SCENES['M3']['H']
    try:
        baseline_faces = [c7.faces('M3', n) for n in RUNGS]
        assert all(face == baseline_faces[0] for face in baseline_faces)
        for height in (.30, .60):
            c7.SCENES['M3']['H'] = height
            for n in RUNGS:
                scene, _ = c7.build('M3', n, 's')
                grid = [round(length*n) for length in scene.domain.size]
                assert c7.faces('M3', n) == baseline_faces[0]
                assert scene.structures[1].shape.size == (.30, .20, height)
                if n == 40:
                    plan = c7.fd.plan(scene)
                    assert list(plan.grid.n) == grid
                    assert plan.steps == runs_by_pol['s'][40]['steps']
                base = runs_by_pol['s'][n]
                result['layout'].append(dict(height=height, n=n, grid=grid,
                                             cells=math.prod(grid), faces=c7.faces('M3', n),
                                             seconds_scaled_from_old_s=base['wall']*math.prod(grid)
                                             / math.prod(base['grid'])))
    finally:
        c7.SCENES['M3']['H'] = original_height
    # The previous lifted-box resource study measured about 34 dense complex
    # matrices of size (2 * Fourier-order-count)^2; this is an estimate, not an
    # allocation guarantee. A 2-GiB reserve is a resource margin, not a physics gate.
    result['reference_memory_estimates'] = [
        dict(N=n, estimated_peak_bytes=34*16*(2*(2*n+1)**2)**2,
             fits_current_memory_with_2GiB_reserve=(34*16*(2*(2*n+1)**2)**2+2*2**30
                <= min(result['memory']['available_physical'], result['memory']['available_pagefile'])))
        for n in (16,20,24,26,28,30,32)]
    result['time_only_counts'] = {pol: dict(Counter(c['time_only_class'] for c in cells))
                                  for pol, cells in result['historical_cells'].items()}
    result['max_time_shares'] = {pol: max(c['time_share'] for c in cells
                                        if c['time_share'] is not None)
                                  for pol, cells in result['historical_cells'].items()}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('historical_cells','sources')},
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
