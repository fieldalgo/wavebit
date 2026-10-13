"""Complete polynomial/transmission fields in compensated common coordinates.

Fixed reference exponents and one-sided angular states define the same finite
field space as the ordinary sampler. Combine every polynomial and real/imaginary
transmission component before rounding the complete physical gradients.
"""
import numpy as np
from wavebit.rcwa._engine.conforming.pair_complex_precision import C
from wavebit.rcwa._engine.conforming.pair_corner_precision import D
from wavebit.rcwa._engine.conforming.pair_corner_precision import logarithm
from wavebit.rcwa._engine.conforming.pair_corner_precision import simplex_coordinates
from wavebit.rcwa._engine.conforming.pair_contact_quadrature import coordinates
from wavebit.rcwa._engine.conforming.pair_contact_quadrature import PI
import wavebit.rcwa._engine.conforming.pair_oscillatory_fields as jet
from wavebit.rcwa._engine.conforming.pair_complex_exponential import exponential


def evaluated_modes(x,y,exponent,start,sector,center_angle,u,flux,coefficient):
    """Evaluate every fixed one-sided sector power in one complete batch."""
    radius=D(np.hypot(x.hi,y.hi));radius=radius+(x*x+y*y-radius*radius)/(2*radius)
    theta=np.arctan2(y.hi,x.hi);rotated=exponential(C(theta)*1j)
    delta=(y*rotated.real-x*rotated.imag)/(x*rotated.real+y*rotated.imag)
    exact_theta=D(theta)+delta
    quarters=np.rint(np.asarray(start)/(np.pi/2));shift=np.round((center_angle-(theta-start))/(2*np.pi))
    angle=exact_theta+(2*shift-(quarters+sector)/2)*PI
    u,flux,coefficient,exponent=map(C,(u,flux,coefficient,exponent));q=flux/coefficient
    logarithmic=C(logarithm(radius));positive=(u-1j*q)/2*exponential(exponent*(logarithmic+1j*C(angle)))
    negative=(u+1j*q)/2*exponential(exponent*(logarithmic-1j*C(angle)))
    left=positive/(C(x)+1j*C(y));right=negative/(C(x)-1j*C(y))
    return positive+negative,exponent*(left+right),1j*exponent*(left-right)

def local_mode(mode,x,y,sector,center_angle):
    u,flux=mode.complete_states[sector]
    return evaluated_modes(x,y,mode.complete_exponent,mode.start,sector,center_angle,u,flux,mode.complete_coefficients[sector])

def packed_parameters(values):
    """Retain both words of every fixed mode parameter when batching fields."""
    if isinstance(values[0],C):
        return C(np.array([v.hi for v in values]),np.array([v.lo for v in values]))
    return np.array(values)

class CommonCoordinates:
    def __init__(self,fields,mapping):
        self.fields=fields;self.mapping=D(mapping);self.cache={}
        nold=fields.obj.data['Cfull'].shape[1]-1
        self.raw=D.zeros((fields.count,self.mapping.hi.shape[1]))
        if not fields.count:return
        for ids,transform in zip(fields.groups,fields.transforms):
            if len(ids):self.raw[ids]=D(fields.column_scale[ids,None])*(D(transform)@self.mapping[nold+ids])
    def prepared(self,tri):
        key=tuple(tri)
        if key not in self.cache:
            data=self.fields.obj.data;nold=data['Cfull'].shape[1]-1
            ordinary=D(data['Cfull'][data['periodic'][tri],1:])@self.mapping[:nold]
            if self.fields.count:
                ordinary=ordinary-D(self.fields.interpolant[data['pmap'][data['periodic'][tri]]])@self.raw
            self.cache[key]=ordinary
        return self.cache[key]
    def sample(self,tri,bary,inverse,dinverse,bary_tangent=None):
        fields=self.fields;bb=coordinates(bary);ref=np.array([[-1.,-1.],[1.,0.],[0.,1.]])
        moving=bary_tangent is not None and np.any(bary_tangent)
        db=simplex_coordinates(np.asarray(bary),bary_tangent)[1] if moving else [D(0.)]*3
        points=[jet.Jet((bb[i],db[i],D(ref[i,0]),D(ref[i,1]),D(0.),D(0.)),not moving) for i in range(3)]
        polynomial=jet.mapped(jet.bernstein(points,fields.obj.degree),self.prepared(tri),not moving,True)
        gradients=list(polynomial.parts[2:4]);changes=list(polynomial.parts[4:6])
        vertices=fields.zero[tri[:3]];hats=fields.hats[tri[:3]]
        # Batch every active mode, preserving all real/imaginary components.
        # Only exact zero cutoff support is absent from the local table.
        first=0;entries=[];xs=[];ys=[]
        for corner,center,mode,parts in fields.modes:
            cutoff=hats[:,corner]
            if not np.any(cutoff):first+=len(parts);continue
            offsets=vertices-center;offsets[cutoff==1]=0.
            jac=(D(offsets[1:])-D(offsets[0])).T
            x,y=(sum(bb[i]*offsets[i,axis] for i in range(3)) for axis in range(2))
            centroid=offsets.mean(axis=0);angle=(np.arctan2(centroid[1],centroid[0])-mode.start)%(2*np.pi)
            sector=min(np.searchsorted(np.cumsum(mode.angles),angle,side='right'),2)
            entries.append((mode,parts,first,cutoff,offsets,jac,sector,angle))
            xs.append(x);ys.append(y);first+=len(parts)
        def packed(values):return D(np.stack([v.hi for v in values],1),np.stack([v.lo for v in values],1))
        if entries:
            xx,yy=packed(xs),packed(ys)
            arguments=[packed_parameters([fn(mode,sector,angle) for mode,parts,first,cutoff,offsets,jac,sector,angle in entries]) for fn in (
                lambda m,s,a:m.complete_exponent,lambda m,s,a:m.start,lambda m,s,a:s,lambda m,s,a:a,
                lambda m,s,a:m.complete_states[s][0],lambda m,s,a:m.complete_states[s][1],lambda m,s,a:m.complete_coefficients[s])]
            sampled=evaluated_modes(xx,yy,*arguments)
        for column,(mode,parts,first,cutoff,offsets,jac,sector,angle) in enumerate(entries):
            x,y=xs[column],ys[column]
            f,gx,gy=(v[:,column] for v in sampled)
            h=sum(bb[i]*cutoff[i] for i in range(3));gh=np.array([cutoff[1]-cutoff[0],cutoff[2]-cutoff[0]])
            raw=[C(h)*(gx*jac[0,a]+gy*jac[1,a])+f*gh[a] for a in range(2)]
            if moving:
                dx,dy=(sum(db[i]*offsets[i,axis] for i in range(3)) for axis in range(2))
                # Each sector potential is harmonic and homogeneous. These
                # identities differentiate its Cartesian gradient without
                # subtracting nearby singular field samples.
                scale=(mode.complete_exponent-1)/C(x*x+y*y)
                fxx=scale*(gx*C(x)-gy*C(y))
                fxy=scale*(gx*C(y)+gy*C(x))
                dgx,dgy=fxx*C(dx)+fxy*C(dy),fxy*C(dx)-fxx*C(dy)
                df=gx*C(dx)+gy*C(dy)
                dh=sum(db[i]*cutoff[i] for i in range(3))
                draw=[C(dh)*(gx*jac[0,a]+gy*jac[1,a])
                      +C(h)*(dgx*jac[0,a]+dgy*jac[1,a])+df*gh[a] for a in range(2)]
            for part in parts:
                for a in range(2):
                    gradients[a]=gradients[a]+getattr(raw[a],part)[:,None]*self.raw[first]
                    if moving:changes[a]=changes[a]+getattr(draw[a],part)[:,None]*self.raw[first]
                first+=1
        result=[sum(gradients[a]*D(matrix[a,b]) for a in range(2)) for matrix in (inverse,dinverse) for b in range(2)]
        if moving:
            for b in range(2):result[2+b]=result[2+b]+sum(changes[a]*D(inverse[a,b]) for a in range(2))
        return result
