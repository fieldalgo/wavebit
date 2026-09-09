"""Rejected material-defined NVF candidate: representation equality is insufficient.

Exact permittivity Fourier coefficients define a smoothed gradient covariance,
so equivalent primitive partitions supply the same field. This experiment tests
that gain against the independent thin-gap Li derivative. It is not a production
normal method, does not enforce exact boundary normals, and has no AD contract.

Run from the checkout:
    PYTHONPATH=$PWD python studies/nvf_material_projection.py --widths 4 8 16 32

The Gaussian width is stated in reciprocal orders of the 700-square fixture.
Changing it is an experiment, not a recommendation for numerical tuning.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from wavebit._core import parametric as P, factorization as F

# Use the existing independently checked fixture and Li instrument.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from resonant_nvf_probe import make_solver  # noqa: E402
from test_contact_jones_probe import solver,pair  # noqa: E402
from contact_instruments import rectangular_li  # noqa: E402


def build(xp,shapes,background,hu,hv,lattice,dtype,normal_grid=128,plan=None,*,width=8.,lift='jones',**kwargs):
    if xp.__name__ not in ('numpy', 'array_api_compat.numpy'):
        raise ValueError('this research instrument supports NumPy only')
    N=normal_grid
    F.check_normal_grid((N,N),hu,hv)
    if width<=0 or lift not in ('real','half','jones'):
        raise ValueError('positive spectral width and a known lift are required')
    cut=(N-1)//4
    e,_=P.assemble_eps_coeffs(xp,shapes,background,cut,cut,lattice,dtype)
    modes=np.arange(-2*cut,2*cut+1);KY,KX=np.meshgrid(modes,modes,indexing='ij')
    G=np.linalg.inv(np.asarray(lattice)).T@np.array([KX.ravel(),KY.ravel()])
    gxmode,gymode=G.reshape(2,*KX.shape)*700
    mask=np.exp(-.5*(gxmode**2+gymode**2)/width**2)
    def realspace(c):
        z=np.zeros((N,N),complex)
        z[KY%N,KX%N]=c*np.exp(1j*np.pi*(KX+KY)/N)
        return np.fft.ifft2(z)*N*N
    gx=realspace(e*mask*(1j*gxmode));gy=realspace(e*mask*(1j*gymode))
    freq=np.fft.fftfreq(N)*N;v,u=np.meshgrid(freq,freq,indexing='ij')
    blur=np.exp(-.5*(u*u+v*v)/width**2)
    smooth=lambda f:np.fft.ifft2(np.fft.fft2(f)*blur).real
    a=smooth(abs(gx)**2);b=smooth((gx*np.conj(gy)).real);c=smooth(abs(gy)**2)
    r=float((a+c).mean()/N**2)
    if r<=0:
        raise ValueError('this candidate requires a nonuniform material fixture')
    a=a+r;c=c+r
    if lift=='half':
        d=a+c
        det=np.maximum(0,a*c-b*b)
        fs=(a/d,(b-1j*np.sqrt(det))/d,c/d)
    elif lift=='real':
        d=a+c
        fs=(a/d,b/d,c/d)
    else:
        d=a*a+2*b*b+c*c
        fs=((a*a+b*b)/d,((a+c)*b+1j*(b*b-a*c))/d,(c*c+b*b)/d)
    cs=[P.conv_matrix_2d(xp,P._sampled_coeffs(xp,f,hu,hv),hu,hv) for f in fs]
    cyx=P.conv_matrix_2d(xp,P._sampled_coeffs(xp,np.conj(fs[1]),hu,hv),hu,hv)
    ec,ic=P.assemble_eps_coeffs(xp,shapes,background,hu,hv,lattice,dtype)
    return F._nvf_assemble(xp,P.conv_matrix_2d(xp,ec,hu,hv),P.conv_matrix_2d(xp,ic,hu,hv),*cs,cyx)

def gap_total(gap,order,grid):
    return float(solver([dict(type='rectangle',lx=350.,ly=700.-gap,material=12.25)],
                        order=order,grid=grid).solve().T.sum())


def run(width,lift,grid,gap_order,resonant_order):
    candidate=partial(build,width=width,lift=lift)
    P._PROJECTOR_CACHE.clear()
    try:
        with patch.object(P,'to_conv_mat_parametric_nvf',candidate):
            split=solver(pair(epsilon=4.),order=3,grid=grid).solve().T.sum()
            merged=solver([dict(type='rectangle',lx=600.,ly=300.,material=4.)],
                          order=3,grid=grid).solve().T.sum()
            zero=gap_total(0.,gap_order,grid)
            steps=(1e-4,1e-6)
            slopes=[(gap_total(step,gap_order,grid)-zero)/step for step in steps]
            result=make_solver(resonant_order,grid,'circular').solve()
        with patch.object(P,'to_conv_mat_parametric_nvf',rectangular_li):
            reference_zero=gap_total(0.,gap_order,grid)
            reference=[(gap_total(step,gap_order,grid)-reference_zero)/step for step in steps]
        return dict(width=width,lift=lift,grid=grid,split_error=float(split-merged),
                    gap_order=gap_order,gap_steps=steps,gap_slopes=slopes,
                    li_gap_slopes=reference,resonant_order=resonant_order,
                    resonant_T=float(result.T.sum()),
                    energy_error=float(abs(result.T.sum()+result.R.sum()-1)))
    finally:
        P._PROJECTOR_CACHE.clear()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--widths',nargs='+',type=float,default=[4.,8.,16.,32.])
    parser.add_argument('--lifts',nargs='+',choices=['real','half','jones'],
                        default=['real','half','jones'])
    parser.add_argument('--grid',type=int,default=256)
    parser.add_argument('--gap-order',type=int,default=7)
    parser.add_argument('--resonant-order',type=int,default=10)
    args=parser.parse_args()
    for width in args.widths:
        for lift in args.lifts:
            print(json.dumps(run(width,lift,args.grid,args.gap_order,args.resonant_order)),
                  flush=True)


if __name__=='__main__':
    main()
