"""The settled energy residual T + R - 1 across the substrate's permittivity.

``dT + dR`` is the derivative of the residual. If the residual drifts linearly with the
substrate's index at the 1e-6 level, its slope is what ``substrate_mode``'s settled
derivatives show, and it is the discrete solver's own energy defect, not a derivative error.
"""
import dataclasses
import json
from pathlib import Path
import warnings

import numpy as np

import wavebit.fd as fd
from studies import fdtd_traced_junction_probe as junction
from studies.fdtd_traced_material_probe import SUBSTRATE, substrate_scene

warnings.filterwarnings('ignore')
cells, steps, df = 24, 6000, .05


def rows(v):
    """``substrate_mode``'s settled run: the bottom PML matched to the seed's glass."""
    base = substrate_scene(v)
    pw = base.sources[0].replace(band=fd.Band(wavelength=1., df=df))
    grow = (cells - junction.CELLS) * junction.DX
    scene = dataclasses.replace(
        base, sources=(pw,), run=fd.Run(steps=steps, courant=.5),
        domain=fd.Domain(size=(.6, .6, 2.1 + 2 * grow), center=(0., 0., .05)),
        boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=(
            fd.PML(cells=cells, wavelength=1., background=junction.GLASS),
            fd.PML(cells=cells, wavelength=1.))))
    result = fd.run(scene, 'numpy')
    return (np.asarray(result['T'].orders.given.T[:, 0, 0]),
            np.asarray(result['R'].orders.given.R[:, 0, 0]))


out = []
for eps in (2.0, 2.05, 2.1, 2.15, 2.2):
    v = SUBSTRATE.copy()
    v[4] = eps
    T, R = rows(v)
    out.append(dict(eps=eps, residual=(T + R - 1).tolist()))
    print(out[-1], flush=True)
residual = np.array([r['residual'] for r in out])
eps = np.array([r['eps'] for r in out])
slopes = [float(np.polyfit(eps, residual[:, k], 1)[0]) for k in range(residual.shape[1])]
print('fitted slope per wavelength', slopes, flush=True)
path = Path(__file__).resolve().parent / 'substrate_residual_sweep.json'
path.write_text(json.dumps(dict(rows=out, slopes=slopes), indent=2) + '\n', encoding='utf-8')
