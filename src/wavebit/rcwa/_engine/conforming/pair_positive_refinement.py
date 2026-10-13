"""Refine all positive-gap variables from the original physical weak fields.

Large cancelling potential coefficients can amplify matrix-assembly roundoff
even when the matrix's backward residual is small. Pointwise weak residuals
retain those cancellations inside the complete physical field. The original
balanced matrix is only a preconditioner for this full-state correction.
Neither the independent energy integral nor its acceptance result enters it.
"""
import numpy as np
from wavebit.rcwa._engine.conforming.cartesian_pair_galerkin import _relative
from wavebit.rcwa._engine.conforming.pair_corner_assembly import samples
from wavebit.rcwa._engine.conforming.real_trace_coupling import _contract
from wavebit.rcwa._engine.conforming.real_trace_coupling import _field_product as mul
from wavebit.rcwa._engine.conforming.pair_residual_arithmetic import residual


def physical_constraints(obj,settings,base,order,quadrature,state):
    """Integrate weak equations and their full variations with finite row scales."""
    U,DU,x,dx,H,DH=state
    nc,nz,N=(base[-1][key] for key in ('nc','nz','size'))
    F=np.zeros_like(U);DF=np.zeros_like(U)
    unsplit=lambda:((row,local[11],weight*local[10])
                    for row,local,weight in samples(obj,settings,base,order))
    split=lambda:((row,None,None) for row in quadrature())
    for added,generator in ((False,unsplit),(True,split)):
        for row,thin,normalized_weight in generator():
            ci,zi,w,dw,ep,dep,phase,dphase,gc,dgc,le,dle,sgz,dsgz=(
                row[i] for i in (1,2,3,4,5,6,7,8,9,10,13,14,19,20))
            phi=np.zeros((len(w),2,2*N),complex);dphi=np.zeros_like(phi)
            phi[:,0,:N]=phase.conj();phi[:,1,N:]=phase.conj()
            dphi[:,0,:N]=dphase.conj();dphi[:,1,N:]=dphase.conj()
            field=mul(phi,x)+mul(gc,U[ci])+mul(sgz,U[nc+zi])
            change=(mul(dphi,x)+mul(phi,dx)+mul(dgc,U[ci])+mul(gc,DU[ci])
                    +mul(dsgz,U[nc+zi])+mul(sgz,DU[nc+zi]))
            T=ep*field-mul(phi,H)
            DT=dep*field+ep*change-mul(dphi,H)-mul(phi,DH)
            gz,dgz=row[11:13]
            if added:
                wb,dwb,_,_,test,dtest=row[17:23]
                # Thin triangles and clipped gap strips pair w/s with the
                # physical s*Z field. Elsewhere use w with the finite Z field.
                zw,zdw,zg,zdg=(wb,dwb,sgz,dsgz) if row[23] else (w,dw,gz,dgz)
                extra=mul(le,U[nc+nz:])
                dextra=mul(dle,U[nc+nz:])+mul(le,DU[nc+nz:])
                extraT=ep*extra;extraDT=dep*extra+ep*dextra
                terms=((ci,w,dw,gc,dgc,extraT,extraDT),
                       (nc+zi,zw,zdw,zg,zdg,extraT,extraDT),
                       (np.arange(nc+nz,len(U)),wb,dwb,test,dtest,T+extraT,DT+extraDT))
            else:
                zw,zdw,zg,zdg=(normalized_weight,np.zeros_like(w),sgz,dsgz) if thin else (w,dw,gz,dgz)
                terms=((ci,w,dw,gc,dgc,T,DT),(nc+zi,zw,zdw,zg,zdg,T,DT))
            for ids,weight,change,g,dg,t,dt in terms:
                F[ids]+=_contract('q,qai,qaj->ij',weight,g,t)
                DF[ids]+=(_contract('q,qai,qaj->ij',change,g,t)
                          +_contract('q,qai,qaj->ij',weight,dg,t)
                          +_contract('q,qai,qaj->ij',weight,g,dt))
    return F,DF


def _residuals(blocks,Q,dQ,state):
    U,DU,x,dx,H,DH=state
    K,dK,R,dR,QT,dQT,S,dS,A,dA=blocks
    return (K@U+S@x-QT@H,
            dK@U+K@DU+dS@x+S@dx-dQT@H-QT@DH,
            residual(x-np.eye(len(H)),(Q,U)),residual(dx,(dQ,U),(Q,DU)),
            residual(H,(-A,x),(-R,U)),residual(DH,(-dA,x),(-A,dx),(-dR,U),(-R,DU)))


def _forward_error(info,state,residuals):
    """Weight original equation defects by the complete physical coefficients."""
    U,DU,x,dx,H,DH=state
    F,DF,r,dr,flux,dflux=residuals
    nc,s,ds,rev=(info[key] for key in ('nc','s','ds','reverse'))
    weight=np.r_[np.ones(nc),np.full(len(U)-nc,s)]
    dweight=np.r_[np.zeros(nc),np.full(len(U)-nc,ds)]
    weak=weight[:,None]*F
    dweak=dweight[:,None]*F+weight[:,None]*DF
    xr=x[rev][:,rev].T;dxr=dx[rev][:,rev].T
    return max(_relative(U.conj().T@weak,H),_relative(x.conj().T@flux,H),
               _relative(r.conj().T@H,H),_relative(DU[:,rev].T@weak,DH),
               _relative(U[:,rev].T@dweak,DH),_relative(dxr@flux+xr@dflux,DH),
               _relative(dr[rev][:,rev].T@H+r[rev][:,rev].T@DH,DH))


def correct(blocks,Q,dQ,coordinate,metric,info,state,weak,variation):
    """Correct value and derivative weak/retained/flux equations together."""
    U,DU,x,dx,H,DH=state
    K,dK,R,dR,QT,dQT,S,dS,A,dA=blocks
    nc=info['nc'];tau=np.sqrt(info['s'])
    def solve(f,r,flux):
        forcing=-f+(S-QT@A)@r-QT@flux
        right=coordinate.T@forcing;right[nc:]*=tau
        local=np.linalg.solve(metric,right);local[nc:]/=tau
        return coordinate@local
    r=residual(x-np.eye(len(H)),(Q,U));flux=residual(H,(-A,x),(-R,U))
    c=solve(weak,r,flux)
    cx=-r-Q@c;cH=-flux-A@r+(R-A@Q)@c
    U=U+c;x=x+cx;H=H+cH
    # Changing the value also changes the differentiated weak equation.
    # Correct this forcing before solving for every derivative variable.
    variation=residual(variation,(dK,c),(dS,cx),(-dQT,cH))
    dr=residual(dx,(dQ,U),(Q,DU));dflux=residual(DH,(-dA,x),(-A,dx),(-dR,U),(-R,DU))
    dc=solve(variation,dr,dflux)
    DU=DU+dc;dx=dx-dr-Q@dc
    DH=DH-dflux-A@dr+(R-A@Q)@dc
    return U,DU,x,dx,H,DH


def refine(obj,settings,base,order,quadrature,blocks,Q,dQ,coordinate,metric,state):
    """Use at most two complete corrections when forward-weighted defects grow."""
    info=base[-1];info['pointwise_refinement_steps']=0
    if _forward_error(info,state,_residuals(blocks,Q,dQ,state))<1e-11:
        return state
    for _ in range(2):
        weak,variation=physical_constraints(obj,settings,base,order,quadrature,state)
        residuals=(weak,variation,*_residuals(blocks,Q,dQ,state)[2:])
        if _forward_error(info,state,residuals)<1e-11:break
        state=correct(blocks,Q,dQ,coordinate,metric,info,state,weak,variation)
        info['pointwise_refinement_steps']+=1
    return state
