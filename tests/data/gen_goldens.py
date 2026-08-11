"""Capture reference results from the LEGACY solver, to pin the core against forever.

Run against a checkout where _BaseRCWA still uses the transfer_method chains.

The write is under ``__main__``: everything above it is import-safe, so another module can
read ``CASES`` without silently overwriting the golden file. It was not, and doing exactly
that cost a restore from git.
"""
import pathlib

import numpy as np

import fieldwave

FIELDS = ('R_s','R_p','T_s','T_p','R','R_s','R_p','T','T_s','T_p')

def rand(shape, seed):
    return np.random.default_rng(seed).uniform(1.0, 4.0, size=shape)

CASES = {
 'te_1layer':   dict(pol=0, phi=None, ucell=[[[3.,3.,3.,1.,1.,1.]]], thickness=[0.15], n_top=2., n_bot=1., theta=0., fto=[6,0], period=[0.77], wavelength=0.777),
 'tm_1layer':   dict(pol=1, phi=None, ucell=[[[3.,3.,3.,1.,1.,1.]]], thickness=[0.15], n_top=2., n_bot=1., theta=0., fto=[6,0], period=[0.77], wavelength=0.777),
 'te_oblique':  dict(pol=0, phi=None, ucell=[[[3.,3.,3.,1.,1.,1.]]], thickness=[0.2], n_top=1., n_bot=1.5, theta=0.3, fto=[6,0], period=[0.77], wavelength=0.777),
 'tm_oblique':  dict(pol=1, phi=None, ucell=[[[3.,3.,3.,1.,1.,1.]]], thickness=[0.2], n_top=1., n_bot=1.5, theta=0.3, fto=[6,0], period=[0.77], wavelength=0.777),
 'te_2layer':   dict(pol=0, phi=None, ucell=[[[3.,3.,1.,1.,3.,1.]],[[2.,2.,2.,1.,1.,1.]]], thickness=[0.15,0.22], n_top=1., n_bot=1., theta=0., fto=[5,0], period=[0.7], wavelength=0.9),
 'te_same_mat': dict(pol=0, phi=None, ucell=[[[3.,3.,1.,1.,3.,1.]],[[1.,1.,1.,1.,1.,1.]]], thickness=[0.2,0.1], n_top=1., n_bot=1., theta=0., fto=[5,0], period=[0.7], wavelength=0.9),
 'conical':     dict(psi=0.4, phi=0.3, ucell=[[[3.,3.,3.,1.,1.,1.]]], thickness=[0.15], n_top=2., n_bot=1., theta=0.2, fto=[6,0], period=[0.77], wavelength=0.777),
 'conical_2l':  dict(psi=1.1, phi=0.5, ucell=[[[3.,3.,1.,1.,3.,1.]],[[2.,2.,2.,1.,1.,1.]]], thickness=[0.15,0.22], n_top=1., n_bot=1.5, theta=0.3, fto=[5,0], period=[0.7], wavelength=0.9),
 '2d_normal':   dict(psi=0., phi=0., ucell=rand((1,5,5),0), thickness=[0.2], n_top=1., n_bot=1., theta=0., fto=[2,2], period=[1.,1.], wavelength=0.9),
 '2d_oblique':  dict(psi=0.9, phi=0.4, ucell=rand((1,6,6),1), thickness=[0.3], n_top=1., n_bot=1.5, theta=0.3, fto=[2,2], period=[1.,1.2], wavelength=0.8),
 '2d_2layer':   dict(psi=1.1, phi=0.5, ucell=rand((2,4,4),2), thickness=[0.15,0.22], n_top=1.2, n_bot=1., theta=0.2, fto=[2,2], period=[0.9,0.9], wavelength=0.7),
 '2d_nvf':      dict(psi=0.9, phi=0.4, ucell=rand((1,5,5),3), thickness=[0.2], n_top=1., n_bot=1., theta=0.2, fto=[2,2], period=[1.,1.], wavelength=0.9, factorization='nvf'),
 '2d_cfs':      dict(psi=0.7, phi=0.3, ucell=rand((1,5,5),4), thickness=[0.2], n_top=1.5, n_bot=1., theta=0.2, fto=[2,2], period=[1.,1.], wavelength=0.9, fourier_coefficients='integrated'),
}

out = {}
for name, kw in CASES.items():
    kw = dict(kw); kw['ucell'] = np.asarray(kw['ucell'], dtype=float)
    solver = fieldwave.rcwa(backend=0, **kw)
    r = solver.conv_solve()
    out[f'{name}__grating_type'] = np.array(solver.grating_type_assigned)
    for sub in ('res', 'res_te_inc', 'res_tm_inc'):
        s = getattr(r, sub)
        if s is None:
            continue
        for f in FIELDS:
            out[f'{name}__{sub}__{f}'] = np.asarray(getattr(s, f))
    out[f'{name}__T1'] = np.asarray(solver.T1)
    for i, li in enumerate(solver.layer_info_list):
        for j, nm in enumerate(('epz_i','W','V','q','d','A_i','B')):
            out[f'{name}__layer{i}__{nm}'] = np.asarray(li[j])

if __name__ == '__main__':
    np.savez_compressed(str(pathlib.Path(__file__).parent / 'golden_legacy.npz'), **out)
    print(f'wrote {len(out)} arrays for {len(CASES)} cases')
