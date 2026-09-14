"""Complete contact tensor/JVP with common corner potentials and real seam traces."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from .pair_corner_assembly import augment,samples
from .cartesian_pair_seam_galerkin import _geometry,_contact_blocks,_trace_change,_changed_contact,_free_fourier_energy,_check_response
from .projected_pair_contact import _real_geometry
from .real_trace_coupling import _contract,_field_product as mul,coupled_contact


def edge_rows(obj,settings,base,chart,quadrature):
    info=base[-1];fields=info['corner_fields'];nold,nc=info['original_nc'],info['nc']
    length,s,zero,velocity=obj._chart_geometry(0.,{k:np.asarray(settings[k]) for k in ('basis','widths','height','offset')})
    center=chart[7];a,b=chart[5];height=chart[6]
    nums=obj.data['contact_numerators'][obj.data['mapping']]
    boundary=[nums[:,inds].sum(axis=1)==obj.data['weight_denominator'] for inds in ([4,5,6],[7,8,9])]
    t,w=leggauss(quadrature);t=(t+1)/2;w=w/2
    for tri,local in zip(obj.data['triangles'],info['local']):
        if local[11]:continue
        _,ci,zi,_,_,cc,c1,zz,_,d0,d1,thin,ep,*_=local
        p=zero[tri[:3]];jac=np.column_stack([p[1]-p[0],p[2]-p[0]]);inv=np.linalg.inv(jac)
        for mask,edge,sign in zip(boundary,(-height/2,height/2),(1.,-1.)):
            ij=np.flatnonzero(mask[tri[:3]])
            if len(ij)!=2:continue
            ia,ib=ij
            if p[ia,0]>p[ib,0]:ia,ib=ib,ia
            left,right=p[ia,0]-center[0],p[ib,0]-center[0]
            if right<=left:continue
            left_corner=np.any(fields.hats[tri[ia],[4,6,7,9]]==1)
            right_corner=np.any(fields.hats[tri[ib],[4,6,7,9]]==1)
            if left_corner:bl,br=1-t**5,t**5;ww=5*t**4*w
            elif right_corner:bl,br=t**5,1-t**5;ww=5*t**4*w
            else:bl,br=1-t,t;ww=w
            bb=np.zeros((quadrature,3));bb[:,ia]=bl;bb[:,ib]=br
            poly=obj._gradient_basis(bb,np.zeros_like(bb))[0]
            gc=_contract('qv,vai->qai',poly,cc)/d0
            _,added,_=fields.sample(tri,bb,inv,np.zeros_like(inv))
            gc=np.concatenate([gc,added],axis=2);indices=np.r_[ci,np.arange(nold,nc)]
            # The small complementary barycentric weight must never be
            # reconstructed by subtracting a rounded endpoint coordinate.
            fleft=0. if left_corner else (-(left+a)/(2*a) if (left+right)<0 else (b-left)/(2*b))
            fright=0. if right_corner else (-(right+a)/(2*a) if (left+right)<0 else (b-right)/(2*b))
            profile=bl*fleft+br*fright
            yield indices,ep,bb@p,gc,.5*sign*ww*(right-left)*profile,edge


def context(obj,settings,base,transform,quadrature):
    info=base[-1];nc,nz,N=(info[k] for k in ('nc','nz','size'))
    common,projection,change=transform
    chart=_geometry(obj,settings);center=chart[7];a,b=chart[5];height=chart[6];trace,tg=chart[8:]
    length,s,zero,velocity=obj._chart_geometry(0.,{k:np.asarray(settings[k]) for k in ('basis','widths','height','offset')})
    def records():
        # Coordinate values for the common sampler use the same barycentric
        # partition; recreate its material triangle order without guessing
        # coordinates from oscillatory phases.
        rows=iter(samples(obj,settings,base,quadrature))
        for tri,tag,local in zip(obj.data['triangles'],obj.data['tags'],info['local']):
            for bary,weight in info['corner_fields'].quadrature(tri,quadrature):
                row,_,_=next(rows)
                _,ci,zi,w,dw,ep,dep,phase,dphase,gc,dgc,gz,*_=row
                points=bary@zero[tri[:3]]-center
                old=np.zeros((len(w),2,change.shape[0]))
                if local[11]:
                    vertices=chart[2][tri[:3]]
                    if np.all(vertices[:,0]==0) and vertices[:,1].min()>=-height/2 and vertices[:,1].max()<=height/2:
                        old[:,0]=trace._basis(points[:,1],tg)[0]
                    added=(old-mul(gz,projection[zi]))@change
                else:
                    if tag:
                        f,df,_=trace._basis(points[:,1],tg)
                        old[:,0]=-f/(2*chart[5][tag-1])
                        profile=np.where(points[:,0]<0,-(points[:,0]+a)/(2*a),(b-points[:,0])/(2*b))
                        old[:,1]=profile[:,None]*df
                    added=(old-mul(gc,common[ci])-mul(gz,projection[zi]))@change
                phi=np.zeros((len(w),2,2*N),complex);dphi=np.zeros_like(phi)
                phi[:,0,:N]=phase.conj();phi[:,1,N:]=phase.conj()
                dphi[:,0,:N]=dphase.conj();dphi[:,1,N:]=dphase.conj()
                yield row,local,weight,phi,dphi,added
    return chart,records


def blocks(obj,settings,quadrature,exponents):
    original=_contact_blocks(obj,settings,quadrature)
    oldtransform=_trace_change(obj,settings,quadrature)
    changed=_changed_contact(obj,settings,original,oldtransform,quadrature)
    base=augment(obj,settings,exponents=exponents,quadrature=quadrature)
    nc=base[-1]['nc'];nold=base[-1]['original_nc']
    common=np.zeros((nc,oldtransform[0].shape[1]));common[:nold]=oldtransform[0]
    transform=(common,*oldtransform[1:]);chart,records=context(obj,settings,base,transform,quadrature)
    C=np.zeros((nc,changed[1].shape[1]),complex);C[:nold]=changed[1]
    for row,local,weight,phi,dphi,added in records():
        if local[11]:continue
        _,ci,zi,w,dw,ep,dep,phase,dphase,gc,*_=row
        C[nold:]+=ep*_contract('q,qai,qaj->ij',w,gc[:,:,len(ci)-(nc-nold):],added)
    trace,tg=chart[8:]
    for ci,ep,p,gc,weight,edge in edge_rows(obj,settings,base,chart,quadrature):
        jump=trace._basis(np.array([edge]),tg)[0][0]@transform[2]
        C[nold:]+=ep*_contract('q,qi,j->ij',weight,gc[:,1,len(ci)-(nc-nold):],jump)
    return (base,C,*changed[2:]),transform,chart,records


def solve(obj,settings,quadrature=48,exponents=(.6,.8)):
    data,transform,chart,records=blocks(obj,settings,quadrature,exponents)
    base,C,Z,Qbar,Rbar,EE=data
    k,dk,q,dq,r,dr,qt,dqt,S,dS,e,de,info=base
    nc,nz,N,ds=(info[key] for key in ('nc','nz','size','ds'))
    rev=info['reverse'];ne=Qbar.shape[1];total=nz+ne
    qc=q[:,:nc];dqc=dq[:,:nc];qb=np.hstack([qt[nc:].conj().T,Qbar])
    def fields():
        for row,local,weight,phi,dphi,added in records():
            _,ci,zi,w,dw,ep,dep,phase,dphase,gc,dgc,gz,*_=row
            psi=-mul(phi,qc);psi[:,:,ci]+=gc;psi=_real_geometry(psi)
            dpsi=-(mul(dphi,qc)+mul(phi,dqc));dpsi[:,:,ci]+=dgc;dpsi=_real_geometry(dpsi)
            yield row,local,weight,phi,dphi,added,psi,dpsi
    upper=np.zeros((0,nc))
    for row,local,weight,phi,dphi,added,psi,dpsi in fields():
        if local[11]:continue
        upper=np.linalg.qr(np.vstack([upper,(np.sqrt(row[3])[:,None,None]*psi).reshape(-1,nc)]),mode='r')
    transform_c=np.linalg.solve(upper,np.eye(nc))
    mc=np.zeros((nc,nc),complex);dmc=mc.copy();sc=np.zeros((nc,2*N),complex);dsc=sc.copy()
    mz=np.zeros((total,nc),complex);sz=np.zeros((total,2*N),complex)
    for row,local,weight,phi,dphi,added,psi,dpsi in fields():
        _,ci,zi,w,dw,ep,dep,phase,dphase,gc,dgc,gz,*_=row
        thin=local[11];measure=weight*local[10] if thin else w
        psi=mul(psi,transform_c)
        chi=np.zeros((len(w),2,total));chi[:,:,zi]=gz;chi[:,:,nz:]=added
        if not thin:chi-=_real_geometry(mul(phi,qb))
        mz+=ep*_contract('q,qai,qaj->ij',measure,chi,psi)
        sz+=ep*_contract('q,qai,qaj->ij',measure,chi,phi)
        if thin:
            dmc+=ds*ep*_contract('q,qai,qaj->ij',measure,psi,psi)
            dsc+=ds*ep*_contract('q,qai,qaj->ij',measure,psi,phi)
        else:
            dpsi=mul(dpsi,transform_c)
            mc+=ep*_contract('q,qai,qaj->ij',w,psi,psi);sc+=ep*_contract('q,qai,qaj->ij',w,psi,phi)
            dmc+=_contract('q,qai,qaj->ij',dep*w+ep*dw,psi,psi)+ep*_contract('q,qai,qaj->ij',w,dpsi,psi)+ep*_contract('q,qai,qaj->ij',w,psi,dpsi)
            dsc+=_contract('q,qai,qaj->ij',dep*w+ep*dw,psi,phi)+ep*_contract('q,qai,qaj->ij',w,dpsi,phi)+ep*_contract('q,qai,qaj->ij',w,psi,dphi)
    nx,ny=obj.order;modes=np.array([(m,n) for n in range(-ny,ny+1) for m in range(-nx,nx+1)])
    waves=2*np.pi*modes/chart[4];trace,tg=chart[8:]
    def phiat(p):
        phase=np.exp(1j*p@waves.T);phi=np.zeros((len(p),2,2*N),complex)
        phi[:,0,:N]=phase;phi[:,1,N:]=phase;return phi
    for ci,ep,p,gc,weight,edge in edge_rows(obj,settings,base,chart,quadrature):
        ph=phiat(p);psi=-mul(ph,qc);psi[:,:,ci]+=gc;psi=mul(_real_geometry(psi),transform_c)
        jump=trace._basis(np.array([edge]),tg)[0][0]@transform[2]
        mz[nz:]+=ep*_contract('q,i,qj->ij',weight,jump,psi[:,1]);sz[nz:]+=ep*_contract('q,i,qj->ij',weight,jump,ph[:,1])
    mzz=np.block([[k[nc:,nc:],Z],[Z.T,EE]])
    uc=np.linalg.solve(mc,-sc);uz=np.linalg.solve(mzz,-sz-mz@uc)
    duc=np.linalg.solve(mc,-dsc-dmc@uc-ds*mz.T@uz)
    B=sc[:,rev].T;dB=dsc[:,rev].T;BZ=sz[:,rev].T
    A,DA=np.kron(np.eye(2),e),np.kron(np.eye(2),de)
    H=A+B@uc;DH=DA+dB@uc+B@duc+ds*BZ@uz
    U=np.vstack([transform_c@uc,uz]);dUC=transform_c@duc
    x=np.eye(2*N)-qc@U[:nc];dx=-dqc@U[:nc]-qc@dUC-ds*qb@uz
    def energy(field,field_change,potential):
        value=np.zeros_like(H);delta=value.copy();extra=potential[nc+nz:]
        for row,local,weight,phi,dphi,added in records():
            _,ci,zi,w,dw,ep,dep,phase,dphase,gc,dgc,gz,*_=row
            E=mul(phi,field)+mul(gc,potential[ci])
            if local[11]:
                E+=mul(gz,potential[nc+zi])+mul(added,extra)
                delta+=ds*ep*_contract('q,qai,qaj->ij',weight*local[10],E[:,:,rev],E)
            else:
                DE=mul(dphi,field)+mul(phi,field_change)+mul(dgc,potential[ci])+ds*mul(gz,potential[nc+zi])+ds*mul(added,extra)
                value+=ep*_contract('q,qai,qaj->ij',w,E.conj(),E)
                delta+=_contract('q,qai,qaj->ij',dep*w+ep*dw,E[:,:,rev],E)+ep*_contract('q,qai,qaj->ij',w,DE[:,:,rev],E)+ep*_contract('q,qai,qaj->ij',w,E[:,:,rev],DE)
        for ci,ep,p,gc,weight,edge in edge_rows(obj,settings,base,chart,quadrature):
            E=mul(phiat(p),field)+mul(gc,potential[ci])
            jump=(trace._basis(np.array([edge]),tg)[0][0]@transform[2])@extra
            delta+=ds*ep*(_contract('q,qi,j->ij',weight,E[:,1,rev],jump)+_contract('q,i,qj->ij',weight,jump[rev],E[:,1]))
        return value,delta
    response=lambda:(U,dUC,x,dx,H,DH,float(np.linalg.cond(mc)))
    result=coupled_contact(obj,*data,energy_verifier=energy,response_solver=response)
    _check_response(*result)
    return result
