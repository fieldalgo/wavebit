"""Freeze the legacy field-reconstruction output as reference data.

Run while fieldwave/on_numpy/emsolver/field_distribution.py still exists. Field reconstruction
had no test coverage at all, so this is the first oracle it has ever had.
"""
import pathlib

import numpy as np

import fieldwave

CASES = {
    # name: call_mee kwargs -> exercises grating_type 0 / 1 / 2
    'te_1d':    dict(pol=0, phi=None, ucell=[[[3., 3., 3., 1., 1., 1.]]], thickness=[0.15],
                     n_top=2., n_bot=1., theta=0.2, fto=[4, 0], period=[0.77], wavelength=0.777),
    'tm_1d':    dict(pol=1, phi=None, ucell=[[[3., 3., 3., 1., 1., 1.]]], thickness=[0.15],
                     n_top=2., n_bot=1., theta=0.2, fto=[4, 0], period=[0.77], wavelength=0.777),
    '1d_2layer': dict(pol=0, phi=None, ucell=[[[3., 3., 1., 1., 3., 1.]], [[2., 2., 2., 1., 1., 1.]]],
                      thickness=[0.15, 0.22], n_top=1., n_bot=1., theta=0.1, fto=[3, 0],
                      period=[0.7], wavelength=0.9),
    'conical':  dict(psi=0.4, phi=0.3, ucell=[[[3., 3., 3., 1., 1., 1.]]], thickness=[0.15],
                     n_top=2., n_bot=1., theta=0.2, fto=[4, 0], period=[0.77], wavelength=0.777),
    '2d':       dict(psi=0.9, phi=0.4, ucell=np.random.default_rng(0).uniform(1, 4, (1, 5, 5)),
                     thickness=[0.2], n_top=1., n_bot=1.5, theta=0.3, fto=[2, 2],
                     period=[1., 1.], wavelength=0.9),
    '2d_2layer': dict(psi=1.1, phi=0.5, ucell=np.random.default_rng(1).uniform(1, 4, (2, 4, 4)),
                      thickness=[0.15, 0.22], n_top=1.2, n_bot=1., theta=0.2, fto=[2, 2],
                      period=[0.9, 0.9], wavelength=0.7),
}

out = {}
for name, kw in CASES.items():
    kw = dict(kw)
    kw['ucell'] = np.asarray(kw['ucell'], dtype=float)
    mee = fieldwave.call_mee(backend=0, **kw)
    mee.conv_solve()
    for tag, sfi in (('default', (True, False, False)), ('all', (True, True, True))):
        field = mee.calculate_field(res_x=5, res_y=4, res_z=6, set_field_input=sfi)
        out[f'{name}__{tag}'] = np.asarray(field)
    out[f'{name}__grating_type'] = np.array(mee.grating_type_assigned)

np.savez_compressed(str(pathlib.Path(__file__).parent / 'golden_legacy_field.npz'), **out)
print(f'wrote {len(out)} arrays; shapes: ' +
      ', '.join(f'{k}={v.shape}' for k, v in list(out.items())[:3]))
