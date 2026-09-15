"""Compensated free Fourier moments on the original contact triangles.

The same physical measures and their variations as the common weak matrices
supply every Fourier entry. Nearly coincident phases use complete exponential
moments; no quadrature order, field cutoff or material regularization is added.
"""
import numpy as np
from .pair_complex_precision import C
from .pair_corner_precision import D
from .pair_contact_quadrature import PI,opening_measure
from .triangle_exponential_precision import moment


def moments(obj,settings,base):
    info=base[-1];nx,ny=obj.order;modes=np.array([(i,j) for j in range(-ny,ny+1) for i in range(-nx,nx+1)])
    difference=modes[None,:,:]-modes[:,None,:];unique,lookup=np.unique(difference.reshape(-1,2),axis=0,return_inverse=True)
    length,_,zero,velocity=obj._chart_geometry(0.,{k:np.asarray(settings[k]) for k in ('basis','widths','height','offset')})
    period=D(np.diag(settings['basis'])/length);total=C(np.zeros(len(unique)));delta=C(np.zeros(len(unique)));gram=C(np.zeros(len(unique)))
    for tri,local in zip(obj.data['triangles'],info['local']):
        p=D(zero[tri[:3]]);dp=D(velocity[tri[:3]])*info['ds'];z=2*PI*(p/period@D(unique.T));dz=2*PI*(dp/period@D(unique.T))
        value,change=moment([z[i] for i in range(3)],[dz[i] for i in range(3)])
        total=total+value*C(local[9])*C(local[12])
        delta=delta+change*C(local[9])*C(local[12])+value*(C(local[9])*C(local[13])+C(opening_measure(info['ds'],local))*C(local[12]))
        gram=gram+value*C(local[9])
    N=len(modes)
    return [tuple(v[lookup].reshape(N,N) for v in value.pair) for value in (total,delta,gram)]
