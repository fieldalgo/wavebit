"""Direct physical projected solve for positive polynomial pair gaps.

The real geometry-only transform retains the full nodal/trace field space.
Weighted QR rows are streamed by moving subtriangle. All field, material and
partition variations are assembled before solving the same original retained
constraints; no derivative or tensor block is replaced after the solve.
"""
import numpy as np
from .real_trace_coupling import _contract
from .projected_pair_contact import _real_geometry
from .cartesian_pair_galerkin import _relative


def projected_positive_response(info, quadrature, free, Q, dQ, original_blocks):
    """Return the complete original U, dU, x, dx, H, dH and condition.

    The positive diagonal sqrt-gap scale and real QR transform are frozen
    local coordinates. Their derivatives are zero in this coordinate chart;
    every physical gap derivative is retained in the original fields.
    """
    nc,nz,N,s,ds=(info[k] for k in ('nc','nz','size','s','ds'))
    total=Q.shape[1];scale=np.r_[np.ones(nc),np.full(total-nc,1/np.sqrt(s))]
    q=Q*scale;dq=dQ*scale
    def fields():
        for ids,ci,zi,w,dw,ep,dep,phase,dphase,gc,dgc,gz,dgz,le,dle,gn,dgn,wb,dwb,sgz,dsgz,test,dtest in quadrature():
            phi=np.zeros((len(w),2,2*N),complex);dphi=np.zeros_like(phi)
            phi[:,0,:N]=phase.conj();phi[:,1,N:]=phase.conj()
            dphi[:,0,:N]=dphase.conj();dphi[:,1,N:]=dphase.conj()
            psi=-(phi@q);dpsi=-(dphi@q+phi@dq)
            psi[:,:,ci]+=gc;dpsi[:,:,ci]+=dgc
            psi[:,:,nc+zi]+=sgz*scale[nc+zi];dpsi[:,:,nc+zi]+=dsgz*scale[nc+zi]
            psi[:,:,nc+nz:]+=le*scale[nc+nz:];dpsi[:,:,nc+nz:]+=dle*scale[nc+nz:]
            yield w,dw,ep,dep,phi,dphi,_real_geometry(psi),_real_geometry(dpsi)
    upper=np.zeros((0,total))
    for w,dw,ep,dep,phi,dphi,psi,dpsi in fields():
        rows=(np.sqrt(w)[:,None,None]*psi).reshape(-1,total)
        upper=np.linalg.qr(np.vstack([upper,rows]),mode='r')
    transform=np.linalg.solve(upper,np.eye(total))
    mc=np.zeros((total,total),complex);dmc=np.zeros_like(mc)
    sc=np.zeros((total,2*N),complex);dsc=np.zeros_like(sc)
    for w,dw,ep,dep,phi,dphi,psi,dpsi in fields():
        psi=psi@transform;dpsi=dpsi@transform
        mc+=ep*_contract('q,qai,qaj->ij',w,psi,psi)
        sc+=ep*_contract('q,qai,qaj->ij',w,psi,phi)
        dmc+=(_contract('q,qai,qaj->ij',dep*w+ep*dw,psi,psi)+ep*_contract('q,qai,qaj->ij',w,dpsi,psi)+ep*_contract('q,qai,qaj->ij',w,psi,dpsi))
        dsc+=(_contract('q,qai,qaj->ij',dep*w+ep*dw,psi,phi)+ep*_contract('q,qai,qaj->ij',w,dpsi,phi)+ep*_contract('q,qai,qaj->ij',w,psi,dphi))
    u=np.linalg.solve(mc,-sc);du=np.linalg.solve(mc,-dsc-dmc@u)
    reverse=info['reverse'];B=sc[:,reverse].T;dB=dsc[:,reverse].T
    H=free[0]+B@u;DH=free[1]+dB@u+B@du
    original=(scale[:,None]*transform)@u;doriginal=(scale[:,None]*transform)@du
    field=np.eye(2*N)-Q@original;dfield=-dQ@original-Q@doriginal
    # Correct the complete differentiated state only while its original
    # equations remain unresolved. Do not amplify native matrix roundoff
    # by refining an already accurate physical solution. The target below
    # is ten times stricter than the caller's unchanged residual guard.
    KK,dKK,R,dR,QT,dQT,S,dS,A,DA=original_blocks
    coordinate=scale[:,None]*transform
    physical=np.r_[np.ones(nc),np.full(total-nc,s)]
    flux_correction=R-A@Q
    for _ in range(2):
        terms=(dKK@original,KK@doriginal,dS@field,S@dfield,-dQT@H,-QT@DH)
        defect=sum(terms)
        retained=dfield+dQ@original+Q@doriginal
        flux=DH-DA@field-A@dfield-dR@original-R@doriginal
        retained_error=_relative(retained,dfield,dQ@original,Q@doriginal)
        flux_error=_relative(flux,DH,DA@field,A@dfield,dR@original,R@doriginal)
        residual=max(_relative(defect,*terms),retained_error,flux_error)
        if residual < 1e-11:
            break
        # A trace row may need refinement while the original nodal equations
        # already satisfy this stricter target. Feeding their matrix roundoff
        # back through a nearly coincident closed basis corrupts an accurate
        # physical derivative. Correct only unresolved equation groups; the
        # coupled stiffness still updates the entire original state, and all
        # groups are verified again by the caller at the unchanged guard.
        for group in (slice(0,nc),slice(nc,nc+nz),slice(nc+nz,total)):
            if _relative(defect[group],*(term[group] for term in terms)) < 1e-11:
                defect[group]=0.
        if retained_error < 1e-11:
            retained[:]=0.
        if flux_error < 1e-11:
            flux[:]=0.
        forcing=-defect+(S-QT@A)@retained-QT@flux
        correction=coordinate@np.linalg.solve(mc,coordinate.T@(physical[:,None]*forcing))
        doriginal+=correction
        dfield-=retained+Q@correction
        DH+=-flux-A@retained+flux_correction@correction
    return original,doriginal,field,dfield,H,DH,float(np.linalg.cond(mc))
