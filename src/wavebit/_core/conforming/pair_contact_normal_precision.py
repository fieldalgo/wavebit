"""Complete physical polynomial normal fields and the stored seam chart.

Preserve the original basis coefficients, trace frame and geometric coordinate
changes. Both the contact equations and independent energy reconstruction
receive the same unrounded polynomial gradients and trace values.
"""
from math import factorial
import numpy as np
from .pair_corner_precision import D
from .pair_complex_precision import C
from .pair_contact_quadrature import coordinates,stack,triangle_measures
from .triangle_polynomials import multi_indices,gradient_coefficients

def polynomial_gradients(degree,bary,coefficients,inverse):
    """Differentiate the full nodal potential before its physical map."""
    points=coordinates(bary);values=[]
    for alpha in multi_indices(degree-1):
        value=D(np.ones(len(bary)))*factorial(degree-1)
        for point,power in zip(points,alpha):value=value*point**power/factorial(power)
        values.append(value)
    values=stack(values,1);reference=gradient_coefficients(degree)
    gradients=[values@(D(reference[:,:,axis])@D(coefficients)) for axis in range(2)]
    return stack([sum(gradients[axis]*D(inverse)[axis,component] for axis in range(2)) for component in range(2)],1)

def normal_map(points,velocity,thin):
    """Map the normal opening limit with adj(J0), including collapsed cells.

    A disappearing potential has a nonzero transverse reference gradient.
    Its finite scaled gradient uses adj(J0)/det'(J0). The common potential
    instead annihilates adj(J0) and uses the separate adj(J1) contact map.
    """
    p=D(points);jac=(p[1:]-p[0]).T;area,slope=triangle_measures(points,velocity)
    adj=stack([stack([jac[1,1],-jac[0,1]]),stack([-jac[1,0],jac[0,0]])])
    return adj/(slope if thin else area)

def _clenshaw(coefficients,u):
    """Evaluate the whole Chebyshev polynomial and its derivative."""
    one=two=done=dtwo=type(u).zeros_like(u)
    for k in range(len(coefficients)-1,0,-1):
        value=2*u*one-two+coefficients[k]
        derivative=2*(one+u*done)-dtwo
        two,one=one,value;dtwo,done=done,derivative
    return coefficients[0]+u*one-two,one+u*done-dtwo


def _chebyshev(coefficients,u):
    number=type(u);coefficients=number(coefficients)
    even=not np.any(coefficients.hi[1::2]) and not np.any(coefficients.lo[1::2])
    odd=not np.any(coefficients.hi[::2]) and not np.any(coefficients.lo[::2])
    if even:
        # T_(2j)(u) = T_j(2u^2-1); only identically zero coefficients vanish.
        value,first=_clenshaw(coefficients[::2],2*u*u-1)
        return value,4*u*first
    if odd:
        # T_(2j+1)(u)/u has Chebyshev coefficients (-1)^j at degree
        # zero and 2*(-1)^(j-k) at degrees 1..j. Keep the complete
        # alternating coefficient sums before evaluating that polynomial.
        tail=number(0.);reduced=number.zeros((len(coefficients)//2,))
        for j in range(len(reduced)-1,-1,-1):
            tail=coefficients[2*j+1]-tail;reduced[j]=(1 if j==0 else 2)*tail
        value,first=_clenshaw(reduced,2*u*u-1)
        return u*value,value+4*u*u*first
    return _clenshaw(coefficients,u)

def trace_values(trace,geometry,y):
    """Evaluate the same stored real polynomial trace chart in both words."""
    y=D(y);data=geometry.get('polynomial_frame')
    if data is None:
        assert trace.order==0
        return D(np.ones((len(y),1)))/D(geometry['transform'][0,0]),D.zeros((len(y),1))
    frequency=D(data['frequency']);angle=frequency*y
    phase=C(1j*angle.hi,1j*angle.lo).exp()
    cosine,sine=phase.real,phase.imag;u=sine/data['amplitude'];du=frequency*cosine/data['amplitude']
    values=[];first=[]
    for index in range(2*trace.order+1):
        if index==0:coefficients=data['residual']
        elif index==1 and data['odd'] is not None:coefficients=data['odd']
        else:coefficients=np.eye(1,index+1,index).ravel()
        value,derivative=_chebyshev(coefficients,u);derivative=derivative*du
        if index%2:value,derivative=cosine*value,-frequency*sine*value+cosine*derivative
        values.append(value);first.append(derivative)
    return stack(values,1)@D(data['frame']),stack(first,1)@D(data['frame'])

def contact_normals(obj,transform,chart,tri,tag,local,bary,zero,velocity):
    """Unchanged complete polynomial normal fields and projected seam lift."""
    common,projection,change=transform;data=obj.data;ci,zi=local[1:3]
    p=zero[tri[:3]];v=velocity[tri[:3]];thin=local[11]
    coefficients=data['Zfull'][data['periodic'][tri]][:,zi]
    gz=polynomial_gradients(obj.degree,bary,coefficients,normal_map(p,v,thin))
    points=stack(coordinates(bary),1)@D(p)-D(chart[7])
    old=D.zeros((len(bary),2,len(change)))
    if thin:
        vertices=chart[2][tri[:3]]
        if np.all(vertices[:,0]==0) and vertices[:,1].min()>=-chart[6]/2 and vertices[:,1].max()<=chart[6]/2:
            old[:,0]=trace_values(chart[8],chart[9],points[:,1])[0]
    else:
        if tag:
            value,first=trace_values(chart[8],chart[9],points[:,1])
            old[:,0]=-value/(2*D(chart[5][tag-1]))
            a,b=map(D,chart[5]);profile=(b-points[:,0])/(2*b)
            left=points.hi[:,0]+points.lo[:,0]<0
            profile[left]=-(points[left,0]+a)/(2*a)
            old[:,1]=profile[:,None]*first
        node_coefficients=D(data['Cfull'][data['periodic'][tri],1:][:,ci])@D(common[ci])
        old=old-polynomial_gradients(obj.degree,bary,node_coefficients,normal_map(p,v,False))
    # Perform coordinate combinations only after the polynomial gradients
    # and the original stored trace chart retain both words.
    added=(old.reshape(2*len(bary),len(change))-gz.reshape(2*len(bary),len(zi))@D(projection[zi]))@D(change)
    return gz,added.reshape(len(bary),2,change.shape[1])
