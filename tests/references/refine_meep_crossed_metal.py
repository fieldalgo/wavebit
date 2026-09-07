"""Add a finer independent tier without changing the coarse reference arrays.

    conda run -n meep python tests/references/refine_meep_crossed_metal.py --output /tmp/metal-fine.npz

Only public Meep APIs are called. The physics harness digest includes all helpers used
by the exploratory single-polarization runs as well as by this combined reproduction.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from _meep_crossed_metal_case import CASE
from gen_goldens_meep_crossed_materials import material_factory
from gen_goldens_meep_crossed import _run, _amplitudes


def metadata():
    here = Path(__file__).resolve().parent
    files = [here / '_meep_crossed_metal_case.py', here / '_meep_crossed_material_cases.py',
             here / 'gen_goldens_meep_crossed_materials.py', here / 'gen_goldens_meep_crossed.py']
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return dict(physics_harness_sha256=np.array(digest.hexdigest()),
                coarse_reference_sha256=np.array(hashlib.sha256(
                    (here / 'golden_meep_crossed_metal.npz').read_bytes()).hexdigest()),
                resolution=np.array(120),
                until_after_sources=np.array(CASE['meep_until']), gap=np.array(.1),
                df_frac=np.array(.2))


def main():
    import meep as mp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Choose a new output path; do not replace frozen data.')
    out = dict(**metadata(), meep_version=np.array(mp.__version__), complete=np.array(False))
    for leg in ('s', 'p'):
        empty = _run(CASE, leg, 120, empty=True)
        struct = _run(CASE, leg, 120, empty=False, material_at=material_factory(CASE))
        values, leak, purity = _amplitudes(CASE, leg, struct, empty)
        if leak > 1e-9 or purity > .01:
            raise SystemExit(f'Invalid incident control: {leak=} {purity=}')
        out.update({f'metal__120__{leg}__{key}': value for key, value in values.items()})
        out.update({f'metal__120__{leg}__leak': np.array(leak),
                    f'metal__120__{leg}__purity': np.array(purity)})
        np.savez(args.output, **out)
    out['complete'] = np.array(True)
    np.savez(args.output, **out)


if __name__ == '__main__':
    main()
