"""Direct physical contact energy for the explicit polynomial pair fields.

Evaluate the complete field before multiplying it, including vanishing-domain
energy and both sides of the shrinking taper. This NumPy verifier receives no
candidate tensor or tensor derivative and reassembles no projected weak matrix.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from .real_trace_coupling import _contract, _field_product


def verify_contact_energy(obj, settings, base, chart, transform,
                          field, field_change, potential, quadrature):
    """Stationary reciprocal variation at fixed auxiliary coefficients."""
    common,projection,change=transform
    _,_,zero,velocity,period,widths,height,center,trace,tg=chart
    info=base[-1]
    nc,nz,size,ds=(info[k] for k in ('nc','nz','size','ds'))
    reverse=info['reverse']
    extra=potential[nc+nz:]
    energy=np.zeros((2*size,2*size),complex)
    derivative=np.zeros_like(energy)
    t,w=leggauss(quadrature);t=(t+1)/2;w=w/2
    bary=np.stack([np.broadcast_to(1-t[:,None],(quadrature,quadrature)),
                   t[:,None]*(1-t),t[:,None]*t],axis=-1).reshape(-1,3)
    weight=(t[:,None]*w[:,None]*w).ravel()
    polynomial=obj._gradient_basis(bary,np.zeros_like(bary))[0]
    nx,ny=obj.order
    modes=np.array([(m,n) for n in range(-ny,ny+1) for m in range(-nx,nx+1)])
    wave=2*np.pi*modes/period
    rows=obj.data['contact_numerators'][obj.data['mapping']]
    boundary=[rows[:,inds].sum(axis=1)==obj.data['weight_denominator']
              for inds in ([4,5,6],[7,8,9])]
    a,b=widths
    def profile(x):
        return np.where(x<0,-(x+a)/(2*a),(b-x)/(2*b))
    def retained(points, speeds=None):
        phase=np.exp(1j*(points+center)@wave.T)
        value=np.zeros((len(points),2,2*size),complex)
        value[:,0,:size]=phase;value[:,1,size:]=phase
        if speeds is None:
            return value
        delta=1j*(speeds@wave.T)*phase
        tangent=np.zeros_like(value)
        tangent[:,0,:size]=delta;tangent[:,1,size:]=delta
        return value,tangent
    for triangle,tag,local in zip(obj.data['triangles'],obj.data['tags'],info['local']):
        _,ci,zi,_,_,cc,c1,zz,_,d0,d1,thin,material,dmaterial,*_=local
        p=zero[triangle[:3]];points=bary@p
        denominator=d1 if thin else d0
        gc=_contract('qv,vai->qai',polynomial,c1 if thin else cc)/denominator
        gz=_contract('qv,vai->qai',polynomial,zz)/denominator
        old=np.zeros((len(bary),2,change.shape[0]))
        phi,dphi=retained(points,ds*(bary@velocity[triangle[:3]]))
        electric=_field_product(phi,field)+_field_product(gc,potential[ci])
        if thin:
            if np.all(p[:,0]==0) and p[:,1].min()>=-height/2 and p[:,1].max()<=height/2:
                old[:,0]=trace._basis(points[:,1],tg)[0]
            added=(old-gz@projection[zi])@change
            electric+=_field_product(gz,potential[nc+zi])+_field_product(added,extra)
            measure=ds*d1*weight
            derivative+=material*_contract('q,qai,qaj->ij',measure,electric[:,:,reverse],electric)
            continue
        if tag:
            functions,slopes,_=trace._basis(points[:,1],tg)
            old[:,0]=-functions/(2*widths[tag-1])
            old[:,1]=profile(points[:,0])[:,None]*slopes
        added=(old-gc@common[ci]-gz@projection[zi])@change
        dgc=ds*(_contract('qv,vai->qai',polynomial,c1)/d0-gc*d1/d0)
        delectric=(_field_product(dphi,field)+_field_product(phi,field_change)
                   +_field_product(dgc,potential[ci])+ds*_field_product(gz,potential[nc+zi])
                   +ds*_field_product(added,extra))
        measure=d0*weight
        dmeasure=(dmaterial*d0+material*ds*d1)*weight
        energy+=material*_contract('q,qai,qaj->ij',measure,electric.conj(),electric)
        derivative+=(_contract('q,qai,qaj->ij',dmeasure,electric[:,:,reverse],electric)
                     +material*_contract('q,qai,qaj->ij',measure,delectric[:,:,reverse],electric)
                     +material*_contract('q,qai,qaj->ij',measure,electric[:,:,reverse],delectric))
        for mask,edge,sign in zip(boundary,(-height/2,height/2),(1.,-1.)):
            local_ids=np.flatnonzero(mask[triangle[:3]])
            if len(local_ids)!=2:
                continue
            ia,ib=local_ids
            if p[ia,0]>p[ib,0]:ia,ib=ib,ia
            left,right=p[ia,0],p[ib,0]
            if right<=left:
                continue
            bb=np.zeros((quadrature,3));bb[:,ia]=1-t;bb[:,ib]=t
            x=left+(right-left)*t
            poly=obj._gradient_basis(bb,np.zeros_like(bb))[0]
            gradient=_contract('qv,vai->qai',poly,cc)/d0
            phi_edge=retained(np.column_stack([x,np.full_like(x,edge)]))
            electric_edge=_field_product(phi_edge,field)+_field_product(gradient,potential[ci])
            jump=(trace._basis(np.array([edge]),tg)[0][0]@change)@extra
            measure=.5*ds*sign*w*(right-left)*profile(x)
            derivative+=material*(_contract('q,qi,j->ij',measure,electric_edge[:,1,reverse],jump)
                                  +_contract('q,i,qj->ij',measure,jump[reverse],electric_edge[:,1]))
    # Integrate the complete small physical field. Correcting only its free
    # Fourier part would magnify a roundoff-sized quadrature difference by
    # large, cancelling auxiliary coefficients. The independent exact free
    # transform is checked by the original retained/flux equations instead.
    return energy,derivative
