"""Freeze the legacy field-reconstruction output as reference data.

**This script can no longer do what it says, and refuses to run.** It had to run against the
legacy per-backend field reconstruction, which was deleted as soon as the rewrite reached that
subsystem. Re-running it now would capture *the current solver's* output under the name
``golden_legacy_field.npz`` — turning the only legacy-parity evidence in the repo into a
change detector, silently, with every test still green.

The refusal is unconditional rather than a check for the old module, because there is nothing
that could make this correct again: the reference it was written to capture no longer exists
anywhere, and a file reappearing under that name would be a different implementation.

It is kept, rather than deleted, because it is the record of exactly which cases were frozen
and with which arguments; ``tests/test_field_core.py`` reads that from here.

If the shipped output must change on purpose, transform the golden **at the comparison** and
say why, the way ``test_field_core.legacy_to_declared`` does for the 2026-08-14 sign fix. That
keeps the file meaning what it meant and puts the departure where a reader will find it.
"""
import pathlib
import sys

import numpy as np

import wavebit

sys.exit('refusing to run: the legacy implementation this was written to capture is gone, so '
         'this would re-freeze the current solver as its own reference. See the module '
         'docstring.')

CASES = {
    # name: rcwa kwargs -> exercises grating_type 0 / 1 / 2
    'te_1d':    dict(tetm='te', phi=None, ucell=[[[3., 3., 3., 1., 1., 1.]]], thickness=[0.15],
                     n_top=2., n_bot=1., theta=0.2, fto=[4, 0], period=[0.77], wavelength=0.777),
    'tm_1d':    dict(tetm='tm', phi=None, ucell=[[[3., 3., 3., 1., 1., 1.]]], thickness=[0.15],
                     n_top=2., n_bot=1., theta=0.2, fto=[4, 0], period=[0.77], wavelength=0.777),
    '1d_2layer': dict(tetm='te', phi=None, ucell=[[[3., 3., 1., 1., 3., 1.]], [[2., 2., 2., 1., 1., 1.]]],
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
    solver = wavebit.rcwa(backend=0, **kw)
    solver.conv_solve()
    for tag, sfi in (('default', (True, False, False)), ('all', (True, True, True))):
        field = solver.calculate_field(res_x=5, res_y=4, res_z=6, set_field_input=sfi)
        out[f'{name}__{tag}'] = np.asarray(field)
    out[f'{name}__grating_type'] = np.array(solver.grating_type_assigned)

np.savez_compressed(str(pathlib.Path(__file__).parent / 'golden_legacy_field.npz'), **out)
print(f'wrote {len(out)} arrays; shapes: ' +
      ', '.join(f'{k}={v.shape}' for k, v in list(out.items())[:3]))
