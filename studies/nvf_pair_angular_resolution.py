"""Measure full pair contact optics while varying angular and Fourier spaces.

Each invocation computes one complete cell, applies its original physical
guards and checks Fourier reflection before reporting both polarizations.
Optical directional differences use the complete analytical constitutive
derivative. Their agreement is not a physical gap finite-difference oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from studies.nvf_pair_material_sector import GEOMETRY,solver
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--order',type=int,nargs='+',default=[4],metavar='N',
                        help='One shared Fourier order or separate x and y orders.')
    parser.add_argument('--angular-order',type=int,choices=range(5),default=4)
    parser.add_argument('--refinement',type=int,choices=range(4),default=0)
    parser.add_argument('--quadrature',type=int,default=64)
    parser.add_argument('--angular-quadrature',type=int,default=None,
                        help='Separate angular integration order; omitted uses the radial order.')
    parser.add_argument('--oscillatory-radial',action='store_true',
                        help='Add the fixed real logarithmic cosine/sine radial pair.')
    args=parser.parse_args()
    if len(args.order) not in (1,2) or any(n<0 for n in args.order):
        parser.error('order must be one or two nonnegative integers')
    orders=tuple(args.order*2 if len(args.order)==1 else args.order)
    root=Path(__file__).resolve().parents[1]
    sources=sorted((root/'wavebit/_core/conforming').glob('*.py'))
    sources.extend([Path(__file__).resolve(),root/'studies/nvf_pair_material_sector.py'])
    def emit(row):
        print(json.dumps(row,default=lambda a:np.asarray(a).tolist()),flush=True)
    configuration=vars(args)|{'order':orders}
    emit(dict(kind='header',numpy=np.__version__,**configuration,geometry=dict(
        period=700.,widths=[300.,300.],height=300.,gap=0.,offset=[0.,0.],
        epsilon=[[12.25,0.],[-10.,1.]],background=1.,wavelength=900.,
        thickness=300.,top=1.,bottom=2.25,theta=.2,phi=0.),
        exponents=[.6,.8],degree=3,mesh='symmetric',coercivity_angle=1.53,
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sources}))
    start=time.perf_counter()
    cell=CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0,args.refinement),
        orders,degree=3,quadrature=args.quadrature,junctions=True,
        angular_order=args.angular_order,coercivity_angle=1.53,mesh='symmetric',
        oscillatory_radial=args.oscillatory_radial,angular_quadrature=args.angular_quadrature)
    value,tangent,metadata=cell.closure(gap=0.,epsilon=(12.25,-10+1j),**GEOMETRY)
    value,tangent=np.asarray(value),np.asarray(tangent)
    nx,ny=orders
    modes=[(i,j) for j in range(-ny,ny+1) for i in range(-nx,nx+1)]
    lookup={mode:i for i,mode in enumerate(modes)}
    reflection=np.array([lookup[(i,-j)] for i,j in modes])
    defects=[]
    for array,tolerance in ((value,3e-10),(tangent,3e-9)):
        errors=[]
        for slot,sign in zip(array,(1,1,1,-1,-1)):
            reflected=sign*slot[:,reflection][:,:,reflection]
            np.testing.assert_allclose(slot,reflected,atol=tolerance,rtol=0)
            errors.append(float(np.max(abs(slot-reflected))))
        defects.append(errors)
    op=solver(orders,'p')
    def powers(slots):
        result=op.solve_conv(*slots);output=[]
        for sub in (result.p_inc,result.s_inc):
            t,r=float(np.sum(sub.T).real),float(np.sum(sub.R).real)
            assert min(t,r)>=-1e-9 and t+r<=1+1e-9
            output.append([t,r])
        return np.asarray(output)
    optical=powers(value);steps=(6.25e-6,3.125e-6);slopes=[]
    for h in steps:
        y=[powers(value+k*h*tangent)[:,0] for k in (-2,-1,1,2)]
        slopes.append((y[0]-8*y[1]+8*y[2]-y[3])/(12*h))
    emit(dict(kind='contact',polarizations=['p','s'],T=optical[:,0],R=optical[:,1],
        optical_direction_steps=steps,optical_direction_slopes=slopes,
        optical_direction_stencil_change=abs(slopes[1]-slopes[0]),
        reflection_defects=defects,diagnostics=metadata,seconds=time.perf_counter()-start))


if __name__=='__main__':main()
