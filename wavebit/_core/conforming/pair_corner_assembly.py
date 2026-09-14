"""Original pair C/Z equations augmented by real corner potentials."""
import numpy as np
from .real_trace_coupling import _contract
from .pair_corner_fields import PairCornerFields


def augment(obj, settings, *, exponents=(.6,.8), quadrature=48, junctions=False):
    original=obj._assemble(**settings)
    k,dk,q,dq,r,dr,qt,dqt,source,dsource,e,de,info=original
    fields=PairCornerFields(obj,exponents,junctions=junctions)
    nold,nz,N=info['nc'],info['nz'],info['size'];nc=nold+fields.count
    indices=np.r_[np.arange(nold),np.arange(nc,nc+nz)];extra=np.arange(nold,nc)
    def square(a):
        result=np.zeros((nc+nz,nc+nz),complex);result[np.ix_(indices,indices)]=a;return result
    def columns(a):
        result=np.zeros((2*N,nc+nz),complex);result[:,indices]=a;return result
    K,DK=square(k),square(dk)
    Q,DQ,R,DR=(columns(a) for a in (q,dq,r,dr))
    QT,DQT,S,DS=(columns(a.T).T for a in (qt,dqt,source,dsource))
    length,s,zero,velocity=obj._chart_geometry(settings['gap'],{key:np.asarray(settings[key]) for key in ('basis','widths','height','offset')})
    ds=info['ds'];nx,ny=obj.order
    modes=np.array([(m,n) for n in range(-ny,ny+1) for m in range(-nx,nx+1)])
    wave=2*np.pi*modes@np.linalg.inv(np.asarray(settings['basis'])/length)
    for tri,local in zip(obj.data['triangles'],info['local']):
        _,ci,zi,normal,n1,c,c1,z,z1,d0,d1,thin,ep,dep,*_=local
        if thin and not fields.junctions:continue
        p=zero[tri[:3]]+s*velocity[tri[:3]];dp=ds*velocity[tri[:3]]
        jac=np.column_stack([p[1]-p[0],p[2]-p[0]])
        djac=np.column_stack([dp[1]-dp[0],dp[2]-dp[0]])
        det=d0+s*d1;ddet=ds*d1
        if thin:
            j1=np.column_stack([velocity[tri[1]]-velocity[tri[0]],velocity[tri[2]]-velocity[tri[0]]])
            inv=np.array([[j1[1,1],-j1[0,1]],[-j1[1,0],j1[0,0]]])/d1
            dinv=np.zeros_like(inv)
        else:
            inv=np.linalg.inv(jac);dinv=-inv@djac@inv
        for bary,weight in fields.quadrature(tri,quadrature):
            _,g,dg=fields.sample(tri,bary,inv,dinv)
            poly=obj._gradient_basis(bary,np.zeros_like(bary))[0]
            if thin:
                gc=_contract('qv,vai->qai',poly,c1)/d1;dgc=np.zeros_like(gc)
                gz=_contract('qv,vai->qai',poly,z)/d1;dgz=ds*_contract('qv,vai->qai',poly,z1)/d1
            else:
                gc=_contract('qv,vai->qai',poly,c)/det
                dgc=ds*_contract('qv,vai->qai',poly,c1)/det-gc*ddet/det
                gz=_contract('qv,vai->qai',poly,z)/det
                dgz=ds*_contract('qv,vai->qai',poly,z1)/det-gz*ddet/det
            w=weight*det;dw=weight*ddet
            for ids,other,dother,scale,dscale in ((ci,gc,dgc,1.,0.),(nc+zi,gz,dgz,s,ds),(extra,g,dg,1.,0.)):
                zcross=thin and other is gz
                measure=weight*d1 if zcross else w
                dmeasure=np.zeros_like(measure) if zcross else dw
                cross=ep*_contract('q,qai,qaj->ij',measure,g,other)
                dcross=(_contract('q,qai,qaj->ij',dep*measure+ep*dmeasure,g,other)+ep*_contract('q,qai,qaj->ij',measure,dg,other)+ep*_contract('q,qai,qaj->ij',measure,g,dother))
                K[np.ix_(extra,ids)]+=scale*cross
                DK[np.ix_(extra,ids)]+=dscale*cross+scale*dcross
                if ids is not extra:
                    K[np.ix_(ids,extra)]+=cross.T;DK[np.ix_(ids,extra)]+=dcross.T
            phase=np.exp(-1j*(bary@p)@wave.T);dphase=-1j*(bary@dp)@wave.T*phase
            moment=_contract('q,qm,qaj->amj',w,phase,g).reshape(2*N,-1)
            delta=(_contract('q,qm,qaj->amj',dw,phase,g)+_contract('q,qm,qaj->amj',w,dphase,g)+_contract('q,qm,qaj->amj',w,phase,dg)).reshape(2*N,-1)
            Q[:,extra]+=moment;DQ[:,extra]+=delta;R[:,extra]+=ep*moment;DR[:,extra]+=dep*moment+ep*delta
    QT[extra]=Q[:,extra].conj().T;DQT[extra]=DQ[:,extra].conj().T
    S[extra]=R[info['reverse']][:,extra].T;DS[extra]=DR[info['reverse']][:,extra].T
    return (K,DK,Q,DQ,R,DR,QT,DQT,S,DS,e,de,dict(info,nc=nc,original_nc=nold,corner_fields=fields,base_original=original))


def samples(obj,settings,base,quadrature):
    info=base[-1];nc,nz,N,nold=(info[k] for k in ('nc','nz','size','original_nc'))
    fields=info['corner_fields'];extra=np.arange(nold,nc)
    length,s,zero,velocity=obj._chart_geometry(settings['gap'],{key:np.asarray(settings[key]) for key in ('basis','widths','height','offset')})
    ds=info['ds'];nx,ny=obj.order
    modes=np.array([(m,n) for n in range(-ny,ny+1) for m in range(-nx,nx+1)])
    wave=2*np.pi*modes@np.linalg.inv(np.asarray(settings['basis'])/length)
    for tri,local in zip(obj.data['triangles'],info['local']):
        ids,ci,zi,normal,n1,c,c1,z,z1,d0,d1,thin,ep,dep,*_=local
        p=zero[tri[:3]]+s*velocity[tri[:3]];dp=ds*velocity[tri[:3]]
        det=d0+s*d1;ddet=ds*d1
        if not thin:
            jac=np.column_stack([p[1]-p[0],p[2]-p[0]])
            djac=np.column_stack([dp[1]-dp[0],dp[2]-dp[0]])
            inv=np.linalg.inv(jac);dinv=-inv@djac@inv
        for bary,weight in fields.quadrature(tri,quadrature):
            poly=obj._gradient_basis(bary,np.zeros_like(bary))[0]
            if thin:
                gc=_contract('qv,vai->qai',poly,c1)/d1;dgc=np.zeros_like(gc)
                sgz=_contract('qv,vai->qai',poly,z)/d1
                dsgz=ds*_contract('qv,vai->qai',poly,z1)/d1
                j1=np.column_stack([velocity[tri[1]]-velocity[tri[0]],velocity[tri[2]]-velocity[tri[0]]])
                inv=np.array([[j1[1,1],-j1[0,1]],[-j1[1,0],j1[0,0]]])/d1
                _,g,dg=fields.sample(tri,bary,inv,np.zeros_like(inv))
            else:
                gc=_contract('qv,vai->qai',poly,c)/det
                dgc=ds*_contract('qv,vai->qai',poly,c1)/det-gc*ddet/det
                sgz=s*_contract('qv,vai->qai',poly,z)/det
                dsgz=(ds*_contract('qv,vai->qai',poly,z)+s*ds*_contract('qv,vai->qai',poly,z1))/det-sgz*ddet/det
                _,g,dg=fields.sample(tri,bary,inv,dinv)
            common=np.concatenate([gc,g],axis=2);dcommon=np.concatenate([dgc,dg],axis=2)
            phase=np.exp(-1j*(bary@p)@wave.T);dphase=-1j*(bary@dp)@wave.T*phase
            w=weight*det;dw=weight*ddet;empty=np.zeros((len(w),2,0))
            gc,ci=common,np.r_[ci,extra]
            # Contact uses the normalized Z field; positive samples retain
            # the unscaled Z separately for the original physical verifier.
            if thin:gz=sgz if s==0 else sgz/s
            else:gz=_contract('qv,vai->qai',poly,z)/det
            dgz=np.zeros_like(gz) if s==0 else (dsgz-ds*gz)/s
            gn=_contract('qv,vai->qai',poly,normal.swapaxes(-1,-2))
            dgn=ds*_contract('qv,vai->qai',poly,n1.swapaxes(-1,-2))
            if not thin or s>0:
                dgn=dgn/det-gn*ddet/det**2;gn=gn/det
            row=(ids,ci,zi,w,dw,ep,dep,phase,dphase,gc,dcommon,gz,dgz,empty,empty,gn,dgn,w,dw,sgz,dsgz,empty,empty)
            yield row,local,weight
            ci=local[1]
