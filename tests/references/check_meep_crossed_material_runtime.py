"""Check the new Meep references at twice the post-source runtime, without replacing them.

    conda run -n meep python tests/references/check_meep_crossed_material_runtime.py --output /tmp/material-runtime.json

Only the public Meep API is called. No Wavebit solver is imported; traceback source
frames are suppressed by the shared generator. The original reference remains immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from _meep_crossed_material_cases import CASES
from _meep_crossed_cases import margin
from gen_goldens_meep_crossed_materials import material_factory
from gen_goldens_meep_crossed import _run, _amplitudes


def main():
    import meep as mp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Choose a new output path; do not overwrite reference evidence.')
    here = Path(__file__).resolve().parent
    reference = here / 'golden_meep_crossed_materials.npz'
    files = [Path(__file__), here / '_meep_crossed_material_cases.py',
             here / 'gen_goldens_meep_crossed.py', here / 'gen_goldens_meep_crossed_materials.py',
             here / '_meep_crossed_cases.py']
    report = dict(meep_version=mp.__version__,
                  reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  harness_sha256=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest(),
                  resolution=40, until_after_sources=240., cases={}, complete=False)
    empty = {}
    with np.load(reference) as frozen:
        for name, original in CASES.items():
            case = dict(original, meep_until=240.)
            factory = material_factory(case)
            for leg in ('s', 'p'):
                start = time.perf_counter()
                if leg not in empty:
                    empty[leg] = _run(case, leg, 40, empty=True)
                struct = _run(case, leg, 40, empty=False, material_at=factory)
                got, leak, purity = _amplitudes(case, leg, struct, empty[leg])
                errors = dict(amplitude=0., power=0.)
                for key, values in got.items():
                    n = case['n_top'] if key in ('r_s', 'r_p', 'R') else case['n_bot']
                    mask = np.array([[margin(case, m, l, n) > .05 for m in case['orders_u']]
                                     for l in case['orders_v']])
                    delta = float(np.max(np.abs(values[mask] - frozen[f'{name}__40__{leg}__{key}'][mask])))
                    kind = 'power' if key in ('R', 'T') else 'amplitude'
                    errors[kind] = max(errors[kind], delta)
                report['cases'][f'{name}.{leg}'] = dict(errors=errors, leak=leak, purity=purity,
                                                       seconds=time.perf_counter()-start)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                print('RUNTIME_RESULT', name, leg, errors, flush=True)
                if errors['amplitude'] > .001 or leak > 1e-9 or purity > .01:
                    raise SystemExit('Runtime or source control failed; investigate without replacing the reference.')
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
