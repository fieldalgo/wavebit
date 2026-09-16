"""Complete nested transmission cutoffs in three-word common coordinates.

Keep every original mode and every requested polynomial cutoff. The complete
potential, its spatial gradient and its moving gradient use the same fixed
reference parameters. Only the final complete field is rounded to two words.
"""
from math import factorial,prod
import numpy as np
from .pair_corner_precision import D as Double
from .pair_complex_precision import C as ComplexDouble
from .pair_extended_precision import Triple as T,ComplexTriple as C,PI,logarithm,exponential
from .triangle_polynomials import multi_indices,gradient_coefficients


def _coordinates(bary):
    exact=getattr(bary,'exact',None)
    if exact is None:
        values=np.asarray(bary);out=[T(values[:,i].copy()) for i in range(3)]
    else:
        exact=T(exact);values=exact.hi
        out=[T(*(v[:,i].copy() for v in exact.words)) for i in range(3)]
    anchor=np.argmax(abs(values),axis=1)
    for i in range(3):
        rows=np.flatnonzero(anchor==i)
        if len(rows):out[i][rows]=1-sum(out[j][rows] for j in range(3) if j!=i)
    return out


def _stack(values,axis=1):
    return T(*(np.stack([v.words[k] for v in values],axis) for k in range(3)))


def _bernstein(values,degree):
    rows=len(values[0]);powers=[[T(np.ones(rows))] for _ in range(3)]
    for i in range(3):
        for _ in range(degree):powers[i].append(powers[i][-1]*values[i])
    return _stack([factorial(degree)//prod(factorial(int(a)) for a in alpha)
                   *powers[0][alpha[0]]*powers[1][alpha[1]]*powers[2][alpha[2]]
                   for alpha in multi_indices(degree)])


def _polynomial(degree,values,changes,coefficients,moving):
    basis=_bernstein(values,degree-1)
    matrices=[T(gradient_coefficients(degree)[:,:,axis])@coefficients for axis in range(2)]
    gradients=[basis@matrix for matrix in matrices]
    if not moving:return gradients,[T.zeros_like(v) for v in gradients]
    lower=_bernstein(values,degree-2)
    lookup={tuple(map(int,alpha)):i for i,alpha in enumerate(multi_indices(degree-2))}
    delta=T.zeros(basis.hi.shape)
    for column,alpha in enumerate(multi_indices(degree-1)):
        for axis,power in enumerate(alpha):
            if power:
                beta=list(map(int,alpha));beta[axis]-=1
                delta[:,column]=delta[:,column]+(degree-1)*changes[axis]*lower[:,lookup[tuple(beta)]]
    return gradients,[delta@matrix for matrix in matrices]


def _parameters(values):
    if isinstance(values[0],ComplexDouble):
        return C(T(np.array([v.hi.real for v in values]),np.array([v.lo.real for v in values])),
                 T(np.array([v.hi.imag for v in values]),np.array([v.lo.imag for v in values])))
    return np.array(values)


def _modes(x,y,exponent,start,sector,center_angle,u,flux,coefficient):
    radius=T(np.hypot(x.hi,y.hi))
    for _ in range(2):radius=radius+(x*x+y*y-radius*radius)/(2*radius)
    theta=np.arctan2(y.hi,x.hi);rotated=exponential(C(theta)*1j)
    delta=(y*rotated.real-x*rotated.imag)/(x*rotated.real+y*rotated.imag)
    exact_theta=T(theta)+delta-delta**3/3
    quarters=np.rint(np.asarray(start)/(np.pi/2));shift=np.round((center_angle-(theta-start))/(2*np.pi))
    angle=exact_theta+(2*shift-(quarters+sector)/2)*PI
    u,flux,coefficient,exponent=map(C,(u,flux,coefficient,exponent));q=flux/coefficient
    logarithmic=C(logarithm(radius))
    positive=(u-1j*q)/2*exponential(exponent*(logarithmic+1j*C(angle)))
    negative=(u+1j*q)/2*exponential(exponent*(logarithmic-1j*C(angle)))
    left=positive/(C(x)+1j*C(y));right=negative/(C(x)-1j*C(y))
    return positive+negative,exponent*(left+right),1j*exponent*(left-right)


class CommonCoordinates:
    def __init__(self,fields,mapping):
        self.fields=fields;self.mapping=T(mapping);self.cache={}
        nold=fields.obj.data['Cfull'].shape[1]-1
        self.raw=T.zeros((fields.count,self.mapping.hi.shape[1]))
        if not fields.count:return
        for ids,transform in zip(fields.groups,fields.transforms):
            if len(ids):self.raw[ids]=T(fields.column_scale[ids,None])*(T(transform)@self.mapping[nold+ids])

    def prepared(self,tri):
        key=tuple(tri)
        if key not in self.cache:
            data=self.fields.obj.data;nold=data['Cfull'].shape[1]-1
            ordinary=T(data['Cfull'][data['periodic'][tri],1:])@self.mapping[:nold]
            if self.fields.count:
                ordinary=ordinary-T(self.fields.interpolant[data['pmap'][data['periodic'][tri]]])@self.raw
            self.cache[key]=ordinary
        return self.cache[key]

    def sample(self,tri,bary,inverse,dinverse,bary_tangent=None):
        fields=self.fields;bb=_coordinates(bary)
        moving=bary_tangent is not None and np.any(bary_tangent)
        db=[T(0.)]*3
        if moving:
            anchor=np.argmax(abs(np.asarray(bary)),axis=1)
            db=[T(np.asarray(bary_tangent)[:,i].copy()) for i in range(3)]
            for i in range(3):
                rows=np.flatnonzero(anchor==i)
                if len(rows):db[i][rows]=-sum(db[j][rows] for j in range(3) if j!=i)
        gradients,changes=_polynomial(fields.obj.degree,bb,db,self.prepared(tri),moving)
        vertices=fields.zero[tri[:3]];hats=fields.hats[tri[:3]]
        first=0;entries=[];xs=[];ys=[];unique={};representatives=[];sample_columns=[]
        for (corner,center,mode,parts),power in zip(fields.modes,fields.cutoff_powers):
            cutoff=hats[:,corner]
            if not np.any(cutoff):first+=len(parts);continue
            offsets=vertices-center;offsets[cutoff==1]=0.
            jac=(T(offsets[1:])-T(offsets[0])).T
            x,y=(sum(bb[i]*offsets[i,axis] for i in range(3)) for axis in range(2))
            centroid=offsets.mean(axis=0);angle=(np.arctan2(centroid[1],centroid[0])-mode.start)%(2*np.pi)
            sector=min(np.searchsorted(np.cumsum(mode.angles),angle,side='right'),2)
            entries.append((mode,parts,first,cutoff,offsets,jac,sector,angle,power))
            xs.append(x);ys.append(y);first+=len(parts)
            # Every cutoff of this immutable mode shares exactly the same
            # Cartesian power. Evaluate it once and retain every multiplier.
            if id(mode) not in unique:
                unique[id(mode)]=len(representatives);representatives.append(len(entries)-1)
            sample_columns.append(unique[id(mode)])
        if entries:
            selected=[entries[index] for index in representatives]
            arguments=[_parameters([fn(mode,sector,angle) for mode,parts,first,cutoff,offsets,jac,sector,angle,power in selected]) for fn in (
                lambda m,s,a:m.complete_exponent,lambda m,s,a:m.start,lambda m,s,a:s,lambda m,s,a:a,
                lambda m,s,a:m.complete_states[s][0],lambda m,s,a:m.complete_states[s][1],lambda m,s,a:m.complete_coefficients[s])]
            sampled=_modes(_stack([xs[index] for index in representatives]),_stack([ys[index] for index in representatives]),*arguments)
        for column,(mode,parts,first,cutoff,offsets,jac,sector,angle,power) in enumerate(entries):
            x,y=xs[column],ys[column];f,gx,gy=(v[:,sample_columns[column]] for v in sampled)
            h=sum(bb[i]*cutoff[i] for i in range(3));s=sum(bb[i]*(1-cutoff[i]) for i in range(3))
            gh=np.array([cutoff[1]-cutoff[0],cutoff[2]-cutoff[0]])
            weight=h*s**power
            first_weight=s**power if power==0 else s**power-power*h*s**(power-1)
            second_weight=T(0.) if power==0 else (T(-2.) if power==1 else power*(power-1)*h*s**(power-2)-2*power*s**(power-1))
            raw=[C(weight)*(gx*jac[0,a]+gy*jac[1,a])+f*C(first_weight)*gh[a] for a in range(2)]
            if moving:
                dx,dy=(sum(db[i]*offsets[i,axis] for i in range(3)) for axis in range(2))
                scale=(C(mode.complete_exponent)-1)/C(x*x+y*y)
                fxx=scale*(gx*C(x)-gy*C(y));fxy=scale*(gx*C(y)+gy*C(x))
                dgx,dgy=fxx*C(dx)+fxy*C(dy),fxy*C(dx)-fxx*C(dy)
                df=gx*C(dx)+gy*C(dy);dh=sum(db[i]*cutoff[i] for i in range(3))
                draw=[C(first_weight*dh)*(gx*jac[0,a]+gy*jac[1,a])
                      +C(weight)*(dgx*jac[0,a]+dgy*jac[1,a])
                      +df*C(first_weight)*gh[a]+f*C(second_weight*dh)*gh[a] for a in range(2)]
            for part in parts:
                for a in range(2):
                    gradients[a]=gradients[a]+getattr(raw[a],part)[:,None]*self.raw[first]
                    if moving:changes[a]=changes[a]+getattr(draw[a],part)[:,None]*self.raw[first]
                first+=1
        result=[sum(gradients[a]*T(matrix[a,b]) for a in range(2)) for matrix in (inverse,dinverse) for b in range(2)]
        if moving:
            for b in range(2):result[2+b]=result[2+b]+sum(changes[a]*T(inverse[a,b]) for a in range(2))
        return [Double(*value.pair) for value in result]
