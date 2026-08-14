"""Freeze the legacy convolution-matrix builders as reference data.

Run while wavebit/on_numpy/emsolver/{convolution_matrix,fourier_analysis,nvf}.py still exist.
Once frozen, those modules can be deleted and the core still has an oracle to answer to.
"""
import pathlib

import numpy as np

from wavebit.on_numpy.emsolver.convolution_matrix import (
    to_conv_mat_raster_sampled, to_conv_mat_raster_integrated,
    to_conv_mat_raster_nvf, to_conv_mat_block, cell_compression,
)


def rand(shape, seed):
    return np.random.default_rng(seed).uniform(1.0, 4.0, size=shape)


SAMPLED = [('1d_1layer', rand((1, 1, 8), 0), 4, 0), ('2d_1layer', rand((1, 5, 5), 1), 2, 2),
       ('2layer_rect', rand((2, 4, 6), 2), 2, 1), ('3layer_1d', rand((3, 1, 10), 3), 5, 0)]
NVF = [('nvf_5x5', rand((1, 5, 5), 10), 2, 2), ('nvf_2layer', rand((2, 4, 6), 11), 2, 2),
       ('nvf_binary', np.where(rand((1, 8, 8), 12) > 2.5, 3.5, 1.0), 2, 2)]
INTEGRATED = [('cfs_1d', rand((1, 1, 8), 20), 4, 0), ('cfs_2d', rand((1, 5, 5), 21), 2, 2),
       ('cfs_binary', np.where(rand((1, 6, 6), 22) > 2.5, 3.5, 1.0), 2, 2),
       ('cfs_2layer', rand((2, 4, 6), 23), 2, 1)]

out = {}
for name, ucell, fx, fy in SAMPLED:
    for tag, enh in (('enhanced', True), ('plain', False)):
        for slot, m in zip(('epx', 'epy', 'epz_i'),
                           to_conv_mat_raster_sampled(ucell, fx, fy)):
            out[f'dfs__{name}__{tag}__{slot}'] = m
for name, ucell, fx, fy in NVF:
    for slot, m in zip(('epx', 'epy', 'epxy', 'epyx', 'epz_i'),
                       to_conv_mat_raster_nvf(ucell, fx, fy)):
        out[f'nvf__{name}__{slot}'] = m
for name, ucell, fx, fy in INTEGRATED:
    for slot, m in zip(('epx', 'epy', 'epz_i'),
                       to_conv_mat_raster_integrated(ucell, fx, fy)):
        out[f'cfs__{name}__{slot}'] = m

# block path: explicit segment edges, no compression step
info = [list(cell_compression(rand((5, 5), 24))), list(cell_compression(rand((4, 6), 25)))]
for i, (cell, x, y) in enumerate(info):
    out[f'vector__in{i}__cell'] = cell
    out[f'vector__in{i}__x'] = x
    out[f'vector__in{i}__y'] = y
for slot, m in zip(('epx', 'epy', 'epz_i'), to_conv_mat_block(info, 2, 2)):
    out[f'vector__{slot}'] = m

# cell_compression itself (the core splits it into indices + gather)
for seed in (26, 27):
    ucell = np.where(rand((6, 7), seed) > 2.5, 3.0, 1.0)
    cell, x, y = cell_compression(ucell)
    out[f'compress__{seed}__ucell'] = ucell
    out[f'compress__{seed}__cell'] = cell
    out[f'compress__{seed}__x'] = x
    out[f'compress__{seed}__y'] = y

np.savez_compressed(str(pathlib.Path(__file__).parent / 'golden_legacy_factorization.npz'), **out)
print(f'wrote {len(out)} arrays')
