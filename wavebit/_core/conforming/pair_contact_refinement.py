"""Refine the original complete contact variation in fixed real coordinates."""
import numpy as np
from .cartesian_pair_galerkin import _relative


def refine(base,C,Qbar,Rbar,coordinate,metric,U,dUC,x,dx,H,DH):
    """Correct weak, retained and flux defects together, preserving every field.

    The physical closed metric is only a preconditioner for the original
    differentiated equations. No independent energy result enters this solve.
    The caller still applies all of its original physical/energy checks.
    """
    k,dk,q,dq,r,dr,qt,dqt,source,dsource,e,de,info=base
    nc,ds=info['nc'],info['ds']
    qc,rc,sc,qtc=q[:,:nc],r[:,:nc],source[:nc],qt[:nc]
    A,DA=np.kron(np.eye(2),e),np.kron(np.eye(2),de)
    dQ=np.hstack([dq,ds*Qbar]);dR=np.hstack([dr,ds*Rbar])
    dK=np.hstack([dk[:nc],ds*C])
    for _ in range(2):
        terms=(dK@U,k[:nc,:nc]@dUC,dsource[:nc]@x,
               sc@dx,-dqt[:nc]@H,-qtc@DH)
        weak=sum(terms)
        retained=dx+dQ@U+qc@dUC
        flux=DH-DA@x-A@dx-dR@U-rc@dUC
        errors=(_relative(weak,*terms),_relative(retained,dx,dQ@U,qc@dUC),
                _relative(flux,DH,DA@x,A@dx,dR@U,rc@dUC))
        if max(errors)<1e-11:break
        forcing=-weak+(sc-qtc@A)@retained-qtc@flux
        correction=coordinate@np.linalg.solve(metric,coordinate.T@forcing)
        dUC+=correction
        dx-=retained+qc@correction
        DH+=-flux-A@retained+(rc-A@qc)@correction
    return dUC,dx,DH
