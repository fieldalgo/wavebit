"""Physical projected contact solve for the explicit polynomial pair family.

Real geometry-only QR coordinates span every closed correction field. Assemble
its material energy from those fields directly: subtracting integrated K, SQ
and Q.HB first loses digits when nodal and retained fields almost coincide.
The complete contact state and its differentiated original equations are
solved together. This NumPy-only kernel supplies no replacement tensor blocks.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from wavebit.rcwa._engine.conforming.real_trace_coupling import _contract
from wavebit.rcwa._engine.conforming.real_trace_coupling import _field_product


def _real_geometry(field):
    """Symmetric Fourier modes represent a real geometric projection.

    Pairing positive and negative frequencies is equivalent to a real
    cosine/sine basis. Reject a non-real field instead of discarding a
    material-dependent imaginary part; these inputs contain no material.
    """
    if np.linalg.norm(field.imag) > 1e-12*max(1., np.linalg.norm(field.real)):
        raise ValueError('contact geometry projection is not real')
    return field.real


def projected_contact_response(obj, blocks, chart, transform, free_energy, quadrature):
    """Return U, dUC, x, dx, H, dH and the physical closed-system condition.

    The invertible QR transform is frozen for the local variation. Its
    derivative is unnecessary: the same varying original field space is
    spanned in this fixed coordinate system. Gap and complex material
    directions enter every physical volume, field and interface term.
    """
    base,C,Z,Qbar,Rbar,EE=blocks
    k,dk,q,dq,r,dr,qt,dqt,source,dsource,e,de,info=base
    nc,nz,size,ds=(info[key] for key in ('nc','nz','size','ds'))
    rev=info['reverse'];common,projection,change=transform;ne=change.shape[1];m=nz+ne
    qc=q[:,:nc];dqc=dq[:,:nc];qbar=np.hstack([qt[nc:].conj().T,Qbar])
    _,_,zero,velocity,period,widths,height,center,trace,tg=chart
    nx,ny=obj.order;modes=np.array([(i,j) for j in range(-ny,ny+1) for i in range(-nx,nx+1)])
    wave=2*np.pi*modes/period
    t,w=leggauss(quadrature);t=(t+1)/2;w=w/2
    bary=np.stack([np.broadcast_to(1-t[:,None],(quadrature,quadrature)),t[:,None]*(1-t),t[:,None]*t],axis=-1).reshape(-1,3)
    weight=(t[:,None]*w[:,None]*w).ravel();poly=obj._gradient_basis(bary,np.zeros_like(bary))[0]
    def phi_at(points,speeds=None):
        phase=np.exp(1j*(points+center)@wave.T);phi=np.zeros((len(points),2,2*size),complex)
        phi[:,0,:size]=phase;phi[:,1,size:]=phase
        if speeds is None:return phi
        dphase=1j*(speeds@wave.T)*phase;dphi=np.zeros_like(phi);dphi[:,0,:size]=dphase;dphi[:,1,size:]=dphase
        return phi,dphi
    def samples():
        for tri,tag,local in zip(obj.data['triangles'],obj.data['tags'],info['local']):
            _,ci,zi,_,_,cc,c1,zz,_,d0,d1,thin,ep,dep,*_=local
            points=bary@zero[tri[:3]];den=d1 if thin else d0
            gc=_contract('qv,vai->qai',poly,c1 if thin else cc)/den
            gz=_contract('qv,vai->qai',poly,zz)/den
            phi,dphi=phi_at(points,ds*(bary@velocity[tri[:3]]))
            # Flatten only the sample/component axes. Keep the physical
            # subtraction before the QR transform: reassociating Q*T first
            # amplifies cancellation between nearly coincident fields.
            psi=-_field_product(phi,qc);psi[:,:,ci]+=gc;psi=_real_geometry(psi)
            if thin:dpsi=None
            else:
                dg=ds*(_contract('qv,vai->qai',poly,c1)/d0-gc*d1/d0)
                dpsi=-(_field_product(dphi,qc)+_field_product(phi,dqc));dpsi[:,:,ci]+=dg;dpsi=_real_geometry(dpsi)
            yield tri,tag,local,points,gc,gz,phi,dphi,psi,dpsi
    # QR acts on real high-mode fields before forming any energy product.
    # Every column survives; a singular basis is an error, not a rank cutoff.
    # Only one triangle and the current upper factor are stored at a time.
    upper=np.zeros((0,nc))
    for tri,tag,local,points,gc,gz,phi,dphi,psi,dpsi in samples():
        if local[11]:continue
        rows=(np.sqrt(local[9]*weight)[:,None,None]*psi).reshape(-1,nc)
        upper=np.linalg.qr(np.vstack([upper,rows]),mode='r')
    transform_c=np.linalg.solve(upper,np.eye(nc))
    mc=np.zeros((nc,nc),complex);dmc=np.zeros_like(mc)
    sc=np.zeros((nc,2*size),complex);dsc=np.zeros_like(sc)
    mz=np.zeros((m,nc),complex);sz=np.zeros((m,2*size),complex)
    rows=obj.data['contact_numerators'][obj.data['mapping']]
    boundary=[rows[:,inds].sum(axis=1)==obj.data['weight_denominator'] for inds in ([4,5,6],[7,8,9])]
    a,b=widths
    profile=lambda x:np.where(x<0,-(x+a)/(2*a),(b-x)/(2*b))
    for tri,tag,local,points,gc,gz,phi,dphi,psi,dpsi in samples():
        _,ci,zi,_,_,cc,c1,zz,_,d0,d1,thin,ep,dep,*_=local
        psi=_field_product(psi,transform_c)
        old=np.zeros((len(bary),2,ne));v=zero[tri[:3]]
        if thin:
            if np.all(v[:,0]==0) and v[:,1].min()>=-height/2 and v[:,1].max()<=height/2:
                old[:,0]=trace._basis(points[:,1],tg)[0]
            added=(old-_field_product(gz,projection[zi]))@change
        else:
            if tag:
                fun,fun_y,_=trace._basis(points[:,1],tg)
                old[:,0]=-fun/(2*widths[tag-1]);old[:,1]=profile(points[:,0])[:,None]*fun_y
            added=(old-_field_product(gc,common[ci])-_field_product(gz,projection[zi]))@change
        chi=np.zeros((len(bary),2,m));chi[:,:,zi]=gz;chi[:,:,nz:]=added
        if not thin:chi-=_real_geometry(_field_product(phi,qbar))
        measure=(d1 if thin else d0)*weight
        mz+=ep*_contract('q,qai,qaj->ij',measure,chi,psi)
        sz+=ep*_contract('q,qai,qaj->ij',measure,chi,phi)
        if thin:
            dmc+=ds*ep*_contract('q,qai,qaj->ij',measure,psi,psi)
            dsc+=ds*ep*_contract('q,qai,qaj->ij',measure,psi,phi)
            continue
        dpsi=_field_product(dpsi,transform_c)
        mc+=ep*_contract('q,qai,qaj->ij',measure,psi,psi)
        sc+=ep*_contract('q,qai,qaj->ij',measure,psi,phi)
        dm=(dep*d0+ep*ds*d1)*weight
        dmc+=(_contract('q,qai,qaj->ij',dm,psi,psi)+ep*_contract('q,qai,qaj->ij',measure,dpsi,psi)+ep*_contract('q,qai,qaj->ij',measure,psi,dpsi))
        dsc+=(_contract('q,qai,qaj->ij',dm,psi,phi)+ep*_contract('q,qai,qaj->ij',measure,dpsi,phi)+ep*_contract('q,qai,qaj->ij',measure,psi,dphi))
        for mask,edge,sign in zip(boundary,(-height/2,height/2),(1.,-1.)):
            ij=np.flatnonzero(mask[tri[:3]])
            if len(ij)!=2:continue
            ia,ib=ij
            if v[ia,0]>v[ib,0]:ia,ib=ib,ia
            left,right=v[ia,0],v[ib,0]
            if right<=left:continue
            bb=np.zeros((quadrature,3));bb[:,ia]=1-t;bb[:,ib]=t;x=left+(right-left)*t
            gp=_contract('qv,vai->qai',obj._gradient_basis(bb,np.zeros_like(bb))[0],cc)/d0
            ph=phi_at(np.column_stack([x,np.full_like(x,edge)]))
            ps=-_field_product(ph,qc);ps[:,:,ci]+=gp;ps=_field_product(_real_geometry(ps),transform_c)
            jump=trace._basis(np.array([edge]),tg)[0][0]@change
            ew=.5*sign*w*(right-left)*profile(x)
            mz[nz:]+=ep*_contract('q,i,qj->ij',ew,jump,ps[:,1])
            sz[nz:]+=ep*_contract('q,i,qj->ij',ew,jump,ph[:,1])
    mzz=np.block([[k[nc:,nc:],Z],[Z.T,EE]])
    uc=np.linalg.solve(mc,-sc);uz=np.linalg.solve(mzz,-sz-mz@uc)
    duc=np.linalg.solve(mc,-dsc-dmc@uc-ds*mz.T@uz)
    A,DA=free_energy
    B=sc[:,rev].T;dB=dsc[:,rev].T;BZ=sz[:,rev].T
    H=A+B@uc;DH=DA+dB@uc+B@duc+ds*BZ@uz
    u=np.vstack([transform_c@uc,uz]);du=transform_c@duc
    x=np.eye(2*size)-qc@u[:nc]
    dx=-dqc@u[:nc]-qc@du-ds*qbar@uz
    return u, du, x, dx, H, DH, float(np.linalg.cond(mc))
