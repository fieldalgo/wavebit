"""Complete oscillatory fields in compensated radial/angular coordinates.

The prepared symmetric space is evaluated as four real radial kernels times
exact angular polynomials, minus its common polynomial interpolant. All
coordinate operations precede the final rounding. Physical gradients and
moving gradients belong to this same potential; no tensor is projected or
combined with a separately evaluated response.
"""
import numpy as np
from math import factorial
from math import prod
from wavebit.rcwa._engine.conforming.triangle_polynomials import multi_indices
from wavebit.rcwa._engine.conforming.pair_corner_precision import D
from wavebit.rcwa._engine.conforming.pair_corner_precision import logarithm
from wavebit.rcwa._engine.conforming.pair_corner_precision import simplex_coordinates
from wavebit.rcwa._engine.conforming.pair_corner_precision import product as geometry_product
from wavebit.rcwa._engine.conforming.pair_oscillatory_polynomials import prepare
import wavebit.rcwa._engine.conforming.pair_oscillatory_scalar as scalar

_COEFFICIENT_BYTES=64*1024*1024


def prepared_coefficients(fields,tri):
    key=tuple(tri);cache=fields._oscillatory_coordinate_cache
    with fields._oscillatory_coordinate_lock:
        if key in cache:
            value,size=cache.pop(key);cache[key]=(value,size)
            return value
    value=prepare(fields,tri)
    size=8*len(key)+sum(ids.nbytes+keep.nbytes+sv.hi.nbytes+sv.lo.nbytes+
             sum(a.hi.nbytes+a.lo.nbytes for a in (*matrices,ordinary))
             for _,ids,sv,keep,matrices,ordinary in value)
    if size>_COEFFICIENT_BYTES:return value
    with fields._oscillatory_coordinate_lock:
        previous=cache.pop(key,None)
        if previous is not None:fields._oscillatory_coordinate_bytes-=previous[1]
        while cache and fields._oscillatory_coordinate_bytes+size>_COEFFICIENT_BYTES:
            _,(_,removed)=cache.popitem(last=False)
            fields._oscillatory_coordinate_bytes-=removed
        cache[key]=(value,size);fields._oscillatory_coordinate_bytes+=size
    return value


class Jet:
    def __init__(self,value,static=False):
        self.static=static
        if isinstance(value,Jet):self.parts=value.parts;self.static=value.static
        elif isinstance(value,(tuple,list)) and len(value)==6:self.parts=tuple(map(D,value))
        else:
            value=D(value);zero=D.zeros_like(value);self.static=True
            self.parts=(value,zero,zero,zero,zero,zero)
    def __getitem__(self,index):return Jet(tuple(a[index] for a in self.parts),self.static)
    def __add__(self,other):
        other=Jet(other)
        return Jet(tuple(a+b for a,b in zip(self.parts,other.parts)),self.static and other.static)
    __radd__=__add__
    def __neg__(self):return Jet(tuple(-a for a in self.parts),self.static)
    def __sub__(self,other):return self+-Jet(other)
    def __rsub__(self,other):return Jet(other)+-self
    def __mul__(self,other):
        other=Jet(other);a,b=self.parts,other.parts
        if self.static and other.static:
            value=a[0]*b[0];zero=D.zeros_like(value)
            return Jet((value,zero,a[2]*b[0]+a[0]*b[2],a[3]*b[0]+a[0]*b[3],zero,zero),True)
        return Jet((a[0]*b[0],a[1]*b[0]+a[0]*b[1],a[2]*b[0]+a[0]*b[2],a[3]*b[0]+a[0]*b[3],
            a[4]*b[0]+a[2]*b[1]+a[1]*b[2]+a[0]*b[4],a[5]*b[0]+a[3]*b[1]+a[1]*b[3]+a[0]*b[5]))
    __rmul__=__mul__
    def __pow__(self,power):
        value=Jet(np.ones_like(self.parts[0].hi))
        for _ in range(power):value=value*self
        return value

def quotient(numerator,denominator):
    a,b=numerator.parts,denominator.parts
    value=a[0]/b[0]
    if numerator.static and denominator.static:
        zero=D.zeros_like(value)
        return Jet((value,zero,*((a[i]-value*b[i])/b[0] for i in (2,3)),zero,zero),True)
    tangent=(a[1]-value*b[1])/b[0]
    gradients=[(a[i]-value*b[i])/b[0] for i in (2,3)]
    changes=[(a[i]-tangent*b[j]-value*b[i]-gradient*b[1])/b[0]
             for i,j,gradient in zip((4,5),(2,3),gradients)]
    return Jet((value,tangent,*gradients,*changes))

def compose(s,value,first,second=None):
    _,dt,gx,gy,dx,dy=s.parts
    if s.static:
        zero=D.zeros_like(value)
        return Jet((value,zero,first*gx,first*gy,zero,zero),True)
    return Jet((value,first*dt,first*gx,first*gy,second*dt*gx+first*dx,second*dt*gy+first*dy))

def bernstein(points,degree,keep=None):
    rows=len(points[0].parts[0]);values=[point.parts[0] for point in points]
    powers=[[D(np.ones(rows))] for _ in range(3)]
    for i in range(3):
        for _ in range(degree):powers[i].append(powers[i][-1]*values[i])
    def basis(n):
        entries=[]
        for alpha in multi_indices(n):
            scale=factorial(n)//prod(factorial(int(a)) for a in alpha)
            entries.append(scale*powers[0][alpha[0]]*powers[1][alpha[1]]*powers[2][alpha[2]])
        return D(np.stack([a.hi for a in entries],-1),np.stack([a.lo for a in entries],-1))
    indices=[tuple(map(int,alpha)) for n,alpha in enumerate(multi_indices(degree)) if keep is None or keep[n]]
    value=basis(degree)
    if keep is not None:value=value[:,keep]
    first=basis(degree-1)
    def lower(base,n,axes):
        lookup={tuple(map(int,alpha)):i for i,alpha in enumerate(multi_indices(n))}
        result=D.zeros((rows,len(indices)))
        for column,alpha in enumerate(indices):
            beta=list(alpha)
            for axis in axes:beta[axis]-=1
            if min(beta)>=0:result[:,column]=base[:,lookup[tuple(beta)]]
        return result
    derivatives=[degree*lower(first,degree-1,(i,)) for i in range(3)]
    def direction(i,k):return D.broadcast(points[i].parts[k],(rows,))[:,None]
    if all(point.static for point in points):
        zero=D.zeros_like(value)
        return Jet((value,zero,*(sum(derivatives[i]*direction(i,k) for i in range(3)) for k in (2,3)),zero,zero),True)
    second=basis(degree-2)
    hessians=[[degree*(degree-1)*lower(second,degree-2,(i,j)) for j in range(3)] for i in range(3)]
    parts=[value]
    for k in (1,2,3):parts.append(sum(derivatives[i]*direction(i,k) for i in range(3)))
    for k,gradient in ((4,2),(5,3)):
        parts.append(sum(derivatives[i]*direction(i,k) for i in range(3))+
                     sum(hessians[i][j]*direction(i,1)*direction(j,gradient) for i in range(3) for j in range(3)))
    return Jet(tuple(parts))


def mapped(jet,matrix,static=False,gradients_only=False):
    rows=len(jet.parts[0]);columns=jet.parts[0].hi.shape[1]
    selection=(2,3) if static else (2,3,4,5)
    if not gradients_only:selection=(0,2,3) if static else tuple(range(6))
    values=geometry_product(D(np.concatenate([jet.parts[i].hi for i in selection],0),np.concatenate([jet.parts[i].lo for i in selection],0)),matrix)
    parts=[D.zeros((rows,matrix.hi.shape[1])) for _ in range(6)]
    for row,index in enumerate(selection):parts[index]=values[row*rows:(row+1)*rows]
    return Jet(tuple(parts),static)

def kernels(s):
    value=s.parts[0];log=logarithm(value)
    argument=tuple(a.astype(complex) for a in log.pair);out=[]
    for index,node in enumerate(scalar.NODES[:3]):
        alpha=scalar.exact(node)
        radial=scalar.exp_pair(scalar.cp(alpha,argument))
        derivative=scalar.cp(alpha,radial)
        second=None if s.static else scalar.cp(scalar.cp(alpha,scalar.add(alpha,scalar.real(-1.))),radial)
        parts=('real','imag') if index==2 else ('real',)
        for part in parts:
            r,dr=(D(*(getattr(v,part) for v in a)) for a in (radial,derivative))
            ddr=None if s.static else D(*(getattr(v,part) for v in second))/value/value
            out.append(compose(s,r,dr/value,ddr))
    return out

def sample_complete(fields,tri,bary,inverse,dinverse,bary_tangent=None,gradients_only=False):
    coefficients=prepared_coefficients(fields,tri)
    dbary=np.zeros_like(bary) if bary_tangent is None else bary_tangent
    static=not np.any(dbary)
    ref=D(np.array([[-1.,-1.],[1.,0.],[0.,1.]]))
    g=ref if static else ref@D(inverse)
    dg=D.zeros((3,2)) if static else ref@D(dinverse)
    coordinates,changes=simplex_coordinates(bary,dbary)
    points=[Jet((coordinates[i],changes[i],g[i,0],g[i,1],dg[i,0],dg[i,1]),static) for i in range(3)]
    ordinary=bernstein(points,fields.obj.degree)
    gradient=np.zeros((len(bary),2,fields.count));change=np.zeros_like(gradient)
    output=[gradient[:,0],gradient[:,1],change[:,0],change[:,1]]
    if not gradients_only:
        value=np.zeros((len(bary),fields.count));output.insert(0,value)
    for corner,ids,complement,keep,radial_coefficients,polynomial in coefficients:
        s=sum(points[i]*complement[i] for i in range(3));h=sum(points[i]*(1-complement[i]) for i in range(3))
        if np.any(s.parts[0].hi+s.parts[0].lo<=0):
            raise ValueError('corner requires a positive complement')
        angle=[quotient(points[i]*complement[i],s) for i in range(3)]
        basis=bernstein(angle,2+fields.angular_order,keep)
        potential=-mapped(ordinary,polynomial,static,gradients_only)
        for radial,coefficient in zip(kernels(s),radial_coefficients):
            potential=potential+(h*radial)[:,None]*mapped(basis,coefficient,static)
        if static:
            refg=potential.parts[2:4]
            components=(potential.parts[0],*(sum(refg[a]*D(inverse[a,b]) for a in range(2)) for b in range(2)),
                        *(sum(refg[a]*D(dinverse[a,b]) for a in range(2)) for b in range(2)))
        else:components=tuple(potential.parts[index] for index in (0,2,3,4,5))
        if gradients_only:components=components[1:]
        for target,component in zip(output,components):target[:,ids]=component.hi+component.lo
    return (gradient,change) if gradients_only else (value,gradient,change)

def sample(fields,tri,bary,inverse,dinverse,bary_tangent=None,gradients_only=False):
    shapes=((len(bary),2,fields.count),)*2 if gradients_only else ((len(bary),fields.count),(len(bary),2,fields.count),(len(bary),2,fields.count))
    result=tuple(np.empty(shape) for shape in shapes)
    for first in range(0,len(bary),256):
        part=slice(first,first+256)
        value=sample_complete(fields,tri,bary[part],inverse,dinverse,None if bary_tangent is None else bary_tangent[part],gradients_only)
        for target,block in zip(result,value):target[part]=block
    return result
