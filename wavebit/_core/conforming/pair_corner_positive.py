"""Unsplit physical nodal blocks and split added-trace products."""
import numpy as np
from .real_trace_coupling import _field_product as mul,_contract
from .projected_pair_contact import _real_geometry
from .pair_corner_assembly import samples
from .cartesian_pair_galerkin import _relative


def response(obj,settings,base,order,quadrature,free,Q,dQ,original_blocks):
    info=base[-1];nc,nz,N,s,ds=(info[k] for k in ('nc','nz','size','s','ds'))
    total=Q.shape[1];nd=total-nc
    qc=Q[:,:nc];dqc=dQ[:,:nc]
    QT,dQT=original_blocks[4:6]
    qb=QT[nc:].conj().T;dqb=dQT[nc:].conj().T
    def make(row,thin):
        (ids,ci,zi,w,dw,ep,dep,phase,dphase,gc,dgc,gz,dgz,le,dle,gn,dgn,wb,dwb,sgz,dsgz,test,dtest)=row[:23]
        phi=np.zeros((len(w),2,2*N),complex);dphi=np.zeros_like(phi)
        phi[:,0,:N]=phase.conj();phi[:,1,N:]=phase.conj()
        dphi[:,0,:N]=dphase.conj();dphi[:,1,N:]=dphase.conj()
        pc=-mul(phi,qc);dpc=-(mul(dphi,qc)+mul(phi,dqc))
        pc[:,:,ci]+=gc;dpc[:,:,ci]+=dgc
        if thin:
            zd=-s*mul(phi,qb);dzd=-ds*mul(phi,qb)-s*(mul(dphi,qb)+mul(phi,dqb))
            zd[:,:,zi]+=sgz;dzd[:,:,zi]+=dsgz
            if le.shape[2]:zd[:,:,nz:]+=le;dzd[:,:,nz:]+=dle
            wz,dwz=wb,dwb;factor,dfactor=1.,0.
        else:
            zd=-mul(phi,qb);dzd=-(mul(dphi,qb)+mul(phi,dqb))
            zd[:,:,zi]+=gz;dzd[:,:,zi]+=dgz
            if test.shape[2]:zd[:,:,nz:]+=test;dzd[:,:,nz:]+=dtest
            wz,dwz=w,dw;factor,dfactor=s,ds
        return w,dw,wz,dwz,factor,dfactor,ep,dep,phi,dphi,_real_geometry(pc),_real_geometry(dpc),_real_geometry(zd),_real_geometry(dzd)
    def unsplit():
        for row,local,weight in samples(obj,settings,base,order):
            if local[11]:row=(*row[:17],weight*local[10],np.zeros_like(weight),*row[19:])
            yield make(row,local[11])
    def split():
        for row in quadrature():yield make(row,row[23])
    upperc=np.zeros((0,nc));upperz=np.zeros((0,nd))
    for w,dw,wz,dwz,f,df,ep,dep,phi,dphi,pc,dpc,pz,dpz in unsplit():
        upperc=np.linalg.qr(np.vstack([upperc,(np.sqrt(w)[:,None,None]*pc).reshape(-1,nc)]),mode='r')
    for w,dw,wz,dwz,f,df,ep,dep,phi,dphi,pc,dpc,pz,dpz in split():
        upperz=np.linalg.qr(np.vstack([upperz,(np.sqrt(f*wz)[:,None,None]*pz).reshape(-1,nd)]),mode='r')
    tc=np.linalg.solve(upperc,np.eye(nc));tz=np.linalg.solve(upperz,np.eye(nd))
    M=np.zeros((total,total),complex);DM=np.zeros_like(M)
    S=np.zeros((total,2*N),complex);DS=np.zeros_like(S)
    def terms(weight,change,ep,dep,a,da,b,db):
        value=ep*_contract('q,qai,qaj->ij',weight,a,b)
        tangent=(_contract('q,qai,qaj->ij',dep*weight+ep*change,a,b)
                 +ep*_contract('q,qai,qaj->ij',weight,da,b)+ep*_contract('q,qai,qaj->ij',weight,a,db))
        return value,tangent
    c=slice(0,nc);z=slice(nc,nc+nz);t=slice(nc+nz,total)
    for divided,generator in ((False,unsplit),(True,split)):
        for w,dw,wz,dwz,f,df,ep,dep,phi,dphi,pc,dpc,pd,dpd in generator():
            pc=mul(pc,tc);dpc=mul(dpc,tc)
            pz,dpz=pd[:,:,:nz],dpd[:,:,:nz];pt,dpt=pd[:,:,nz:],dpd[:,:,nz:]
            pairs=((c,c,w,dw,pc,dpc,pc,dpc),(z,c,wz,dwz,pz,dpz,pc,dpc),(z,z,f*wz,df*wz+f*dwz,pz,dpz,pz,dpz)) if not divided else (
                (t,c,wz,dwz,pt,dpt,pc,dpc),(t,z,f*wz,df*wz+f*dwz,pt,dpt,pz,dpz),(t,t,f*wz,df*wz+f*dwz,pt,dpt,pt,dpt))
            for rows,columns,weight,change,a,da,b,db in pairs:
                v,dv=terms(weight,change,ep,dep,a,da,b,db);M[rows,columns]+=v;DM[rows,columns]+=dv
            sources=((c,w,dw,pc,dpc),(z,wz,dwz,pz,dpz)) if not divided else ((t,wz,dwz,pt,dpt),)
            for rows,weight,change,a,da in sources:
                v,dv=terms(weight,change,ep,dep,a,da,phi,dphi);S[rows]+=v;DS[rows]+=dv
    M[z,t]=M[t,z].T;DM[z,t]=DM[t,z].T
    for matrix in (M,DM):
        matrix[nc:,:nc]=tz.T@matrix[nc:,:nc]
        matrix[nc:,nc:]=tz.T@matrix[nc:,nc:]@tz
    S[nc:]=tz.T@S[nc:];DS[nc:]=tz.T@DS[nc:]
    M[:nc,nc:]=s*M[nc:,:nc].T
    DM[:nc,nc:]=ds*M[nc:,:nc].T+s*DM[nc:,:nc].T
    # Balance every disappearing coordinate by sqrt(gap). The normalized
    # equations have off-diagonal blocks Mdc and gap*Mdc.T; their symmetric
    # coordinates avoid an ill-conditioned full derivative solve near contact.
    # Differentiate the coordinate map along with every matrix/source block.
    tau=np.sqrt(s);dtau=ds/(2*tau)
    balanced=M.copy();dbalanced=DM.copy()
    balanced[nc:,:nc]=tau*M[nc:,:nc]
    balanced[:nc,nc:]=balanced[nc:,:nc].T
    dbalanced[nc:,:nc]=dtau*M[nc:,:nc]+tau*DM[nc:,:nc]
    dbalanced[:nc,nc:]=dbalanced[nc:,:nc].T
    source=S.copy();dsource=DS.copy()
    source[nc:]*=tau
    dsource[nc:]=dtau*S[nc:]+tau*DS[nc:]
    v=np.linalg.solve(balanced,-source)
    dv=np.linalg.solve(balanced,-dsource-dbalanced@v)
    rev=info['reverse'];B=source[:,rev].T;DB=dsource[:,rev].T
    H=free[0]+B@v
    DH=free[1]+DB@v+B@dv
    u=v.copy();du=dv.copy()
    u[nc:]/=tau
    du[nc:]=(dv[nc:]-dtau*u[nc:])/tau
    U=np.vstack([tc@u[:nc],tz@u[nc:]])
    DU=np.vstack([tc@du[:nc],tz@du[nc:]])
    x=np.eye(2*N)-Q@U;dx=-dQ@U-Q@DU
    coordinate=np.zeros((total,total))
    coordinate[:nc,:nc]=tc;coordinate[nc:,nc:]=tz
    KK,dKK,R,dR,QT,dQT,SS,dSS,A,DA=original_blocks
    for _ in range(2):
        terms=(dKK@U,KK@DU,dSS@x,SS@dx,-dQT@H,-QT@DH)
        defect=sum(terms);retained=dx+dQ@U+Q@DU
        flux=DH-DA@x-A@dx-dR@U-R@DU
        retained_error=_relative(retained,dx,dQ@U,Q@DU)
        flux_error=_relative(flux,DH,DA@x,A@dx,dR@U,R@DU)
        if max(_relative(defect,*terms),retained_error,flux_error)<1e-11:break
        for group in (slice(0,nc),slice(nc,nc+nz),slice(nc+nz,total)):
            if _relative(defect[group],*(v[group] for v in terms))<1e-11:defect[group]=0.
        if retained_error<1e-11:retained[:]=0.
        if flux_error<1e-11:flux[:]=0.
        forcing=-defect+(SS-QT@A)@retained-QT@flux
        right=coordinate.T@forcing;right[nc:]*=tau
        local_correction=np.linalg.solve(balanced,right);local_correction[nc:]/=tau
        correction=coordinate@local_correction
        DU+=correction;dx-=retained+Q@correction
        DH+=-flux-A@retained+(R-A@Q)@correction
    return U,DU,x,dx,H,DH,float(np.linalg.cond(balanced))


def verify(obj,settings,base,order,quadrature,U,x,dxgeo):
    # The stationary variation receives neither the candidate H/DH nor DU.
    # Form the complete original E before adding actual trace cross terms.
    info=base[-1];nc,nz,N=(info[k] for k in ('nc','nz','size'));rev=info['reverse']
    energy=np.zeros((2*N,2*N),complex);delta=np.zeros_like(energy)
    def fields(row):
        ids,ci,zi,w,dw,ep,dep,phase,dphase,gc,dgc,gz,dgz,le,dle,gn,dgn,wb,dwb,sgz,dsgz,*_=row
        phi=np.zeros((len(w),2,2*N),complex);dphi=np.zeros_like(phi)
        phi[:,0,:N]=phase.conj();phi[:,1,N:]=phase.conj()
        dphi[:,0,:N]=dphase.conj();dphi[:,1,N:]=dphase.conj()
        field=mul(phi,x)+mul(gc,U[ci])+mul(sgz,U[nc+zi])
        change=mul(dphi,x)+mul(phi,dxgeo)+mul(dgc,U[ci])+mul(dsgz,U[nc+zi])
        added=mul(le,U[nc+nz:]) if le.shape[2] else np.zeros_like(field)
        dadded=mul(dle,U[nc+nz:]) if le.shape[2] else np.zeros_like(field)
        return w,dw,ep,dep,field,change,added,dadded
    def integrate(w,dw,ep,dep,left,dleft,right,dright):
        value=ep*_contract('q,qai,qaj->ij',w,left.conj(),right)
        tangent=(_contract('q,qai,qaj->ij',dep*w+ep*dw,left[:,:,rev],right)
                 +ep*_contract('q,qai,qaj->ij',w,dleft[:,:,rev],right)
                 +ep*_contract('q,qai,qaj->ij',w,left[:,:,rev],dright))
        return value,tangent
    for row,local,weight in samples(obj,settings,base,order):
        w,dw,ep,dep,e,de,a,da=fields(row)
        v,dv=integrate(w,dw,ep,dep,e,de,e,de);energy+=v;delta+=dv
    for row in quadrature():
        w,dw,ep,dep,e,de,a,da=fields(row)
        for left,dleft,right,dright in ((e,de,a,da),(a,da,e,de),(a,da,a,da)):
            v,dv=integrate(w,dw,ep,dep,left,dleft,right,dright);energy+=v;delta+=dv
    return energy,delta
