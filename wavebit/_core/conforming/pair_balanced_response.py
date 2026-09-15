"""Complete positive-gap response in balanced coordinates."""
import numpy as np


def solve(M,DM,S,DS,nc,s,ds,free,reverse):
    """Keep balancing fixed while solving the original physical variation.

    The original matrix has common/disappearing off-diagonal blocks s*C.T
    and C. Its row weight W=diag(I,s) makes K=W*M symmetric. The stationary
    derivative of the same Schur response uses d(W*M) and d(W*S); it does
    not multiply a small stationarity defect by a large solved variation.
    """
    tau=np.sqrt(s)
    balanced=M.copy()
    balanced[nc:,:nc]=tau*M[nc:,:nc]
    balanced[:nc,nc:]=balanced[nc:,:nc].T
    source=S.copy();source[nc:]*=tau
    v=np.linalg.solve(balanced,-source)
    u=v.copy();u[nc:]/=tau
    right=-DS-DM@u;right[nc:]*=tau
    du=np.linalg.solve(balanced,right);du[nc:]/=tau
    weight=np.r_[np.ones(nc),np.full(len(M)-nc,s)]
    dweight=np.r_[np.zeros(nc),np.full(len(M)-nc,ds)]
    dmetric=weight[:,None]*DM+dweight[:,None]*M
    dforce=weight[:,None]*DS+dweight[:,None]*S
    H=free[0]+source[:,reverse].T@v
    DH=(free[1]+dforce[:,reverse].T@u+u[:,reverse].T@dforce
        +u[:,reverse].T@dmetric@u)
    return u,du,H,DH,balanced
