"""Freeze the legacy analytic (shape-based) Fourier builder as reference data.

Run while fieldwave/on_numpy/emsolver/fourier_analytic.py still exists.
"""
import pathlib

import numpy as np

from fieldwave.on_numpy.emsolver.fourier_analytic import to_conv_mat_analytic_layers

PERIOD = (1.0, 1.2)
CASES = {
    'rect':      [dict(n_base=1.0, shapes=[dict(type='rectangle', cx=0.5, cy=0.6, lx=0.4, ly=0.3, angle=0.0, n_index=2.5)])],
    'rect_rot':  [dict(n_base=1.0, shapes=[dict(type='rectangle', cx=0.5, cy=0.6, lx=0.4, ly=0.3, angle=0.4, n_index=2.5)])],
    'circle':    [dict(n_base=1.0, shapes=[dict(type='circle', cx=0.5, cy=0.6, lx=0.35, n_index=3.0)])],
    'ellipse':   [dict(n_base=1.2, shapes=[dict(type='ellipse', cx=0.4, cy=0.5, lx=0.5, ly=0.25, angle=0.3, n_index=2.2)])],
    'two_shape': [dict(n_base=1.0, shapes=[dict(type='rectangle', cx=0.25, cy=0.3, lx=0.2, ly=0.2, angle=0.0, n_index=2.5),
                                           dict(type='circle', cx=0.75, cy=0.8, lx=0.25, n_index=3.5)])],
    'two_layer': [dict(n_base=1.0, shapes=[dict(type='circle', cx=0.5, cy=0.6, lx=0.35, n_index=3.0)]),
                  dict(n_base=1.5, shapes=[dict(type='rectangle', cx=0.5, cy=0.6, lx=0.4, ly=0.3, angle=0.2, n_index=2.0)])],
}

out = {}
for name, layers in CASES.items():
    for grid in (128, 256):
        mats = to_conv_mat_analytic_layers(layers, 2, 2, PERIOD, normal_grid=grid)
        for slot, m in zip(('epx', 'epy', 'epxy', 'epyx', 'epz_i'), mats):
            out[f'{name}__g{grid}__{slot}'] = np.asarray(m)

np.savez_compressed(str(pathlib.Path(__file__).parent / 'golden_legacy_analytic.npz'), **out)
print(f'wrote {len(out)} arrays for {len(CASES)} cases')
