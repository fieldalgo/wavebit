"""Complete Fourier means of the unchanged disappearing normal potentials.

Periodic integration by parts gives <exp(-ikx), grad(V)> = ik <exp(-ikx), V>.
The distributional gradient includes the collapsed seam and horizontal jumps.
Integrating the bounded potential avoids cancellation between its large bulk
gradients and interface terms. Every original coordinate remains present.
"""
import numpy as np
from .pair_corner_precision import D
from .pair_extended_precision import Triple as T,ComplexTriple as C,exponential
from .pair_transmission_extended_precision import _coordinates,_stack,_bernstein
from .pair_contact_normal_precision import _chebyshev
from .pair_contact_quadrature import gauss,Exact,stack,triangle_measures,phase_values,PI


def _trace(trace,geometry,y):
    """The original stored trace polynomial, before its cancelling projections."""
    data=geometry.get('polynomial_frame')
    if data is None:
        assert trace.order==0
        return T(np.ones((len(y),1)))/T(geometry['transform'][0,0])
    phase=exponential(1j*C(T(data['frequency'])*y))
    cosine,sine=phase.real,phase.imag;u=sine/data['amplitude'];values=[]
    for index in range(2*trace.order+1):
        if index==0:coefficients=data['residual']
        elif index==1 and data['odd'] is not None:coefficients=data['odd']
        else:coefficients=np.eye(1,index+1,index).ravel()
        value,_=_chebyshev(coefficients,u)
        values.append(cosine*value if index%2 else value)
    return _stack(values,1)@T(data['frame'])


def _integrate(obj,settings,base,transform,chart,order):
    from .pair_contact_precision import pair,mat,cp,add_at
    info=base[-1];nz,N=info['nz'],info['size']
    common,projection,change=transform;extra=change.shape[1]
    modes=np.array([(i,j) for j in range(-obj.order[1],obj.order[1]+1)
                    for i in range(-obj.order[0],obj.order[0]+1)])
    zero,velocity=obj._chart_geometry(0.,{k:np.asarray(settings[k])
        for k in ('basis','widths','height','offset')})[2:]
    t,w=gauss(order);u=D.broadcast(t[:,None],(order,order));v=D.broadcast(t[None,:],(order,order))
    bary=Exact(stack([1-u,u*(1-v),u*v],-1).reshape(-1,3))
    weight=(u*w[:,None]*w[None,:]).reshape(-1)
    coordinates=_coordinates(bary);points=_stack(coordinates)
    bernstein=_bernstein(coordinates,obj.degree)
    result=pair(np.zeros((N,nz+extra),complex))
    for tri,tag,local in zip(obj.data['triangles'],obj.data['tags'],info['local']):
        if local[11]:continue
        ci,zi=local[1:3];p=zero[tri[:3]];area,_=triangle_measures(p,velocity[tri[:3]])
        nodal=obj.data['periodic'][tri]
        z=bernstein@T(obj.data['Zfull'][nodal][:,zi])
        old=T.zeros((len(bary),len(change)));xy=points@T(p)-T(chart[7])
        if tag:
            a,b=map(T,chart[5])
            profile=-(xy[:,0]+a)/(2*a) if tag==1 else (b-xy[:,0])/(2*b)
            old=profile[:,None]*_trace(chart[8],chart[9],xy[:,1])
        old=old-bernstein@(T(obj.data['Cfull'][nodal,1:][:,ci])@T(common[ci]))
        added=(old-z@T(projection[zi]))@T(change)
        values=D(np.concatenate([z.hi,added.hi],1),np.concatenate([z.lo,added.lo],1))
        indices=np.r_[zi,nz+np.arange(extra)];weighted=values*(weight*area)[:,None]
        for first in range(0,len(bary),512):
            sl=slice(first,first+512)
            phase,_=phase_values(bary[sl],p,np.zeros((3,2)),modes,chart[4])
            result=add_at(result,np.ix_(np.arange(N),indices),
                          mat(tuple(x.conj().T for x in phase),weighted[sl].pair))
    blocks=[]
    for axis in range(2):
        wave=2*PI*D(modes[:,axis])/D(chart[4][axis])
        blocks.append(cp(tuple(1j*x[:,None] for x in wave.pair),result))
    return tuple(np.vstack([block[word] for block in blocks]) for word in range(2))


def moments(obj,settings,base,transform,chart,order):
    """Resolve the smooth potential integral before any physical projection.

    Compare complete coefficients at increasing Gauss orders. The internal
    3e-28 criterion is the existing complete-field accuracy target; failure
    raises instead of accepting a rounded or symmetry-projected mean.
    """
    from .pair_contact_precision import pair,add,neg
    if not any(obj.order):
        return pair(np.zeros((2,base[-1]['nz']+transform[2].shape[1]),complex))
    first=max(24,obj.degree+2*sum(obj.order)+8)
    previous=_integrate(obj,settings,base,transform,chart,first)
    limit=max(order,first+8)
    for degree in range(first+8,limit+8,8):
        current=_integrate(obj,settings,base,transform,chart,min(degree,limit))
        delta=add(current,neg(previous))
        error=np.max(abs(delta[0]+delta[1]))
        scale=max(1.,float(np.max(abs(current[0]+current[1]))))
        if np.isfinite(error) and error<=3e-28*scale:return current
        previous=current
    raise ArithmeticError('complete normal Fourier means did not resolve at the available quadrature')
