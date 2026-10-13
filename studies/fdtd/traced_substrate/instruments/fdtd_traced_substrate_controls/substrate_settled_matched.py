"""Where the substrate's settled dT + dR residual comes from: the bottom PML's match.

``substrate_mode``'s settled run keeps the bottom PML matched to the seed's glass (2.1) while
central differences move the substrate's permittivity, which is what a traced run does (the
stretch profile is static). Here the same differences are taken twice: once so (the seed's
match), once with the PML matched to each perturbed medium. If the residual is the PML's
mismatch, the second leaves only the run's own energy residual.
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
cells, steps, df, h = 24, 6000, .05, 3e-6


def make(v, matched):
    base = substrate_scene(v)
    pw = base.sources[0].replace(band=fd.Band(wavelength=1., df=df))
    grow = (cells - junction.CELLS) * junction.DX
    bottom = fd.Material(eps=v[4]) if matched else junction.GLASS
    return dataclasses.replace(
        base, sources=(pw,), run=fd.Run(steps=steps, courant=.5),
        domain=fd.Domain(size=(.6, .6, 2.1 + 2 * grow), center=(0., 0., .05)),
        boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=(
            fd.PML(cells=cells, wavelength=1., background=bottom),
            fd.PML(cells=cells, wavelength=1.))))


def rows(v, matched):
    result = fd.run(make(v, matched), 'numpy')
    return (np.asarray(result['T'].orders.given.T[:, 0, 0]),
            np.asarray(result['R'].orders.given.R[:, 0, 0]))


out = {}
d = np.eye(5)[4] * h
for matched in (False, True):
    plus, minus = rows(SUBSTRATE + d, matched), rows(SUBSTRATE - d, matched)
    dT, dR = (plus[0] - minus[0]) / (2 * h), (plus[1] - minus[1]) / (2 * h)
    out['matched' if matched else 'seed_match'] = dict(dT0=dT.tolist(), dR0=dR.tolist(),
                                                       dT_plus_dR=(dT + dR).tolist())
    print(matched, out['matched' if matched else 'seed_match'], flush=True)
path = Path(__file__).resolve().parent / 'substrate_settled_matched.json'
path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
