"""Compensated free Fourier moments on the original contact triangles.

The same physical measures and their variations as the common weak matrices
supply every Fourier entry. Nearly coincident phases use complete exponential
moments; no quadrature order, field cutoff or material regularization is added.
"""
import numpy as np
from wavebit.rcwa._engine.conforming.pair_complex_precision import C
from wavebit.rcwa._engine.conforming.pair_corner_precision import D
from wavebit.rcwa._engine.conforming.pair_contact_quadrature import PI
from wavebit.rcwa._engine.conforming.pair_contact_quadrature import triangle_measures
from wavebit.rcwa._engine.conforming.triangle_exponential_precision import moment


def moments(obj,settings,base):
    info=base[-1];nx,ny=obj.order;modes=np.array([(i,j) for j in range(-ny,ny+1) for i in range(-nx,nx+1)])
    difference=modes[None,:,:]-modes[:,None,:];unique,lookup=np.unique(difference.reshape(-1,2),axis=0,return_inverse=True)
    length,_,zero,velocity=obj._chart_geometry(0.,{k:np.asarray(settings[k]) for k in ('basis','widths','height','offset')})
    period=D(np.diag(settings['basis'])/length);total=C(np.zeros(len(unique)));delta=C(np.zeros(len(unique)));gram=C(np.zeros(len(unique)))
    for tri,local in zip(obj.data['triangles'],info['local']):
        p=D(zero[tri[:3]]);dp=D(velocity[tri[:3]])*info['ds'];z=2*PI*(p/period@D(unique.T));dz=2*PI*(dp/period@D(unique.T))
        area,slope=triangle_measures(p,velocity[tri[:3]])
        value,change=moment([z[i] for i in range(3)],[dz[i] for i in range(3)])
        total=total+value*C(area)*C(local[12])
        delta=delta+change*C(area)*C(local[12])+value*(C(area)*C(local[13])+C(D(info['ds'])*slope)*C(local[12]))
        gram=gram+value*C(area)
    N=len(modes)
    return [tuple(v[lookup].reshape(N,N) for v in value.pair) for value in (total,delta,gram)]
