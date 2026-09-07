"""Generate an independent negative-epsilon crossed reference with runtime controls.

    conda run -n meep python tests/references/gen_goldens_meep_crossed_metal.py --output /tmp/metal.npz

Calls only the public Meep API, never reads implementation source, and never imports
Wavebit. The original positive-epsilon material references are not changed.
"""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np

from _meep_crossed_metal_case import CASE
from _meep_crossed_cases import margin
from gen_goldens_meep_crossed_materials import material_factory
from gen_goldens_meep_crossed import _run, _amplitudes


def main():
    import meep as mp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Choose a new output path; frozen references are never overwritten.')
    here = Path(__file__).resolve().parent
    files = [Path(__file__), here / '_meep_crossed_metal_case.py',
             here / 'gen_goldens_meep_crossed_materials.py', here / 'gen_goldens_meep_crossed.py',
             here / '_meep_crossed_material_cases.py', here / '_meep_crossed_cases.py']
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    out = dict(meep_version=np.array(mp.__version__), generator_sha256=np.array(digest.hexdigest()),
               complete=np.array(False), resolutions=np.array([30, 60]),
               gap=np.array(.1), df_frac=np.array(.2))
    factory = material_factory(CASE)
    for resolution, until in ((30, 120.), (60, 120.), (60, 240.)):
        case = dict(CASE, meep_until=until)
        tag = str(resolution) if until == 120. else f'{resolution}_time240'
        for leg in ('s', 'p'):
            start = time.perf_counter()
            empty = _run(case, leg, resolution, empty=True)
            struct = _run(case, leg, resolution, empty=False, material_at=factory)
            data, leak, purity = _amplitudes(case, leg, struct, empty)
            if leak > 1e-9 or purity > .01:
                raise SystemExit(f'Invalid incident control: {leak=} {purity=}')
            absorbed = 1 - float(data['R'].sum() + data['T'].sum())
            if not 0. < absorbed < 1.:
                raise SystemExit(f'Unexpected passive absorption: {absorbed=}')
            prefix = f'metal__{tag}__{leg}__'
            out.update({prefix+k: v for k, v in data.items()})
            out.update({prefix+'leak': np.array(leak), prefix+'purity': np.array(purity),
                        prefix+'absorbed': np.array(absorbed),
                        prefix+'seconds': np.array(time.perf_counter()-start)})
            if until == 240.:
                largest = 0.
                for key in ('r_s', 'r_p', 't_s', 't_p'):
                    n = case['n_top'] if key.startswith('r') else case['n_bot']
                    mask = np.array([[margin(case, m, l, n) > .05 for m in case['orders_u']]
                                     for l in case['orders_v']])
                    largest = max(largest, float(np.max(np.abs(
                        data[key][mask] - out[f'metal__60__{leg}__{key}'][mask]))))
                out[prefix+'runtime_change'] = np.array(largest)
                if largest > .001:
                    raise SystemExit(f'Reference has not settled: {largest=}')
            np.savez(args.output, **out)
            print('METAL_RESULT', tag, leg, 'absorption', absorbed, flush=True)
    out['complete'] = np.array(True)
    np.savez(args.output, **out)


if __name__ == '__main__':
    main()
