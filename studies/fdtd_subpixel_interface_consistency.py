"""Measure the production map on exact static interface edge/face averages.

This is a consistency diagnostic, not an FDTD convergence or pass criterion.
An interface normal is fixed while its offset is .23 of a cell. E tangent
and D normal are continuous. E is integrated on its edge; D on its dual
face (a transverse line in this invariant-x reduction). See Werner et al.
2013, section 7, for these finite-integration degrees of freedom.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from wavebit.fd._kernel.constitutive import cross_action
from wavebit.fd._lower.coefficients import mix


def fraction(cut, half_span):
    if half_span == 0:
        return (cut >= 0).astype(float)
    # Exact clipped length of a segment, not a numerical repair clamp.
    return np.maximum(0.,np.minimum(1.,.5+cut/(2*half_span)))


def square_fraction(cut, ny, nz, dx):
    if ny == 0: return fraction(cut,nz*dx/2)
    if nz == 0: return fraction(cut,ny*dx/2)
    a,b=ny*dx,nz*dx
    t=cut+(a+b)/2
    # Inclusion/exclusion of triangles in the positive quadrant. Explicit
    # whole/empty branches avoid cancellation far from the interface.
    partial=(np.maximum(t,0)**2-np.maximum(t-a,0)**2
             -np.maximum(t-b,0)**2+np.maximum(t-a-b,0)**2)/(2*a*b)
    return np.where(t<=0,0.,np.where(t>=a+b,1.,partial))


def measure(cells, normal, eps, et, dn):
    dx=1/cells
    ny,nz=normal
    n=np.array([0.,ny,nz]);t=np.array([0.,-nz,ny])
    # Fixed physical sampling square plus two-cell halo for interpolation.
    q=np.arange(-cells-2,cells+3)*dx
    zz,yy=np.meshgrid(q,q,indexing='ij')
    point=np.stack([np.zeros_like(yy),yy,zz],axis=-1)[:,:,None,:]
    e=np.zeros((3,)+point.shape[:-1]);d=e.copy();diagonal=e.copy()
    offset=.23*dx  # Chosen subcell experiment position, not a fitted constant.
    for c in range(3):
        p=point.copy();p[...,c]+=dx/2
        cut=offset-np.sum(p*n,axis=-1)
        edge=fraction(cut,abs(n[c])*dx/2)
        # Invariant x makes each yz-normal dual face a single segment.
        face=fraction(cut,(nz if c==1 else ny)*dx/2)
        e[c]=et*t[c]+dn*n[c]*(edge/eps+1-edge)
        d[c]=et*t[c]*(face*eps+1-face)+dn*n[c]
        f=square_fraction(cut,ny,nz,dx)
        diagonal[c]=1/mix(np,np.ones_like(f),eps,f,n[c]**2)
    f=square_fraction(offset-np.sum(point*n,axis=-1),ny,nz,dx)
    cross=np.zeros_like(d)
    cross[2]=f*(1-f)*(eps-1)**2/(eps*(f*eps+1-f))*ny*nz
    predicted=diagonal*d+cross_action(d,cross,np,'E',bounded=True)
    error=(predicted-e)[:,2:-2,2:-2,:]
    interior=(f[2:-2,2:-2,:]>0)&(f[2:-2,2:-2,:]<1)
    def circulation(v):
        return ((np.roll(v[2],-1,axis=1)-v[2])
                -(np.roll(v[1],-1,axis=0)-v[1]))[2:-2,2:-2,:]
    divergence=(d[1]-np.roll(d[1],1,axis=1)
                +d[2]-np.roll(d[2],1,axis=0))[2:-2,2:-2,:]
    return dict(cells=cells,normal=normal,eps=eps,et=et,dn=dn,
                max_abs_error=float(np.max(abs(error))),
                mean_abs_error=float(np.mean(abs(error[1:]))),
                rms_error=float(np.sqrt(np.mean(error[1:]**2))),
                exact_E_circulation_over_dx=float(np.max(abs(circulation(e)))/dx),
                exact_D_divergence_over_dx=float(np.max(abs(divergence))/dx),
                predicted_E_circulation=float(np.max(abs(circulation(predicted)))),
                intersected_nodes=int(interior.sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    output=parser.parse_args().output
    if output.exists(): raise RuntimeError('Refusing to overwrite diagnostic')
    watched=[Path(__file__), ROOT/'src/wavebit/fd/_kernel/constitutive.py',
             ROOT/'src/wavebit/fd/_lower/coefficients.py']
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    rows=[measure(c,n,eps,et,dn) for n in ((1.,0.),(.6,.8))
          for eps in (1.,2.25,12.) for et,dn in ((1.,0.),(0.,1.)) for c in (16,32,64)]
    assert before=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    result=dict(status='completed_static_consistency_diagnostic_not_scattering_proof',
                source_sha256=before,rows=rows)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    for r in rows: print(json.dumps(r),flush=True)


if __name__=='__main__': main()
