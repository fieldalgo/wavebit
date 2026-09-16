"""Complete compensated contact equations for the real root-20 space.

All common, disappearing and seam-trace fields are retained. Unprojected
weak matrices and Fourier moments are assembled with high/low products
before forming the closed projection and its full physical variation.
The caller checks the result against the unchanged original constraints
and its independent pointwise energy integrals.
"""
import numpy as np
from .pair_corner_common_precision import CommonCoordinates
from .pair_corner_precision import D
from .pair_residual_arithmetic import _product,_sum_pairs
from .pair_oscillatory_scalar import cp as complex_cp
from .projected_pair_contact import _real_geometry
from .pair_contact_quadrature import quadrature,phases,triangle_measures
from .pair_contact_fourier import moments as free_moments


def pair(a):return np.asarray(a),np.zeros_like(a)
def add(*terms):return _sum_pairs(terms)
def neg(a):return -a[0],-a[1]
def mat(a,b):
    # Only the leading product needs its full error expansion. Cross products
    # already contain one low part, so their ordinary rounding is second order.
    # Retain every term, including low @ low, in the normalized high/low sum.
    return _sum_pairs([_product(a[0],b[0]),
                       (a[0]@b[1]+a[1]@b[0],a[1]@b[1])])
def retained_field(phase,coefficients):
    """Apply all Fourier coefficients using the exact Cartesian block support.

    Scalar phases have one column per Fourier mode. The first and second
    coefficient blocks supply x and y respectively; output rows interleave
    the two components. Only structural zeros of the lifted phase are absent.
    """
    rows,n=phase[0].shape
    if phase[1].shape!=(rows,n) or any(v.shape!=coefficients[0].shape for v in coefficients) or coefficients[0].shape[0]!=2*n:
        raise ValueError('retained fields require two complete Fourier coefficient blocks')
    if not all(np.all(np.isfinite(v)) for v in (*phase,*coefficients)):
        raise ArithmeticError('retained fields require finite coefficients')
    out=pair(np.zeros((2*rows,coefficients[0].shape[1]),complex))
    if not np.any(phase[0]) and not np.any(phase[1]):return out
    for component in range(2):
        block=slice(component*n,(component+1)*n)
        value=mat(phase,tuple(v[block] for v in coefficients))
        for target,part in zip(out,value):target[component::2]=part
    return out
def cp(a,b):
    if not any(np.iscomplexobj(v) for v in (*a,*b)):
        return (D(*a)*D(*b)).pair
    return complex_cp(a,b)
def transpose(a,conjugate=False):
    return tuple(v.conj().T if conjugate else v.T for v in a)


def solve(a,b):
    """Refine every column of the same linear system with full residuals."""
    x=pair(np.linalg.solve(a[0],b[0]+b[1]))
    for _ in range(4):
        r=add(b,neg(mat(a,x)))
        x=add(x,pair(np.linalg.solve(a[0],r[0]+r[1])))
    return x


def add_at(value,key,*terms):
    """Accumulate every supported entry before the global coordinate change."""
    updated=add(tuple(a[key] for a in value),*terms)
    for target,part in zip(value,updated):
        target[key]=part
    return value


def local_coordinates(fields,tri,nc):
    """Exact topological support; all global columns survive assembly."""
    data=fields.obj.data;nold=data['Cfull'].shape[1]-1
    nodal=np.flatnonzero(np.any(data['Cfull'][data['periodic'][tri],1:]!=0,axis=0))
    present=np.any(fields.hats[tri[:3]]!=0,axis=0)
    radial=[nold+ids for corner,ids in zip(fields.controls,getattr(fields,'groups',())) if present[corner]]
    indices=np.r_[nodal,*radial].astype(int)
    mapping=np.zeros((nc,len(indices)))
    mapping[indices,np.arange(len(indices))]=1.
    coordinate_type=getattr(fields,'common_coordinate_type',CommonCoordinates)
    return indices,coordinate_type(fields,mapping)


def physical_data(base,Qbar,K,DK,Q,DQ,R,DR,L,SZ,MZZ):
    """Keep the unprojected physical equations before solving any response.

    Undo only the Fourier subtraction in the normal test functions. These
    directly integrated blocks supply the original weak, retained and flux
    checks; their definitions and tolerances do not change. The pointwise
    energy verifier remains independent of this matrix assembly.
    """
    info=base[-1];nc,nz,ds=(info[k] for k in ('nc','nz','ds'));rev=info['reverse']
    qb=np.hstack([base[6][nc:].conj().T,Qbar])
    KC=add(L,mat(transpose(pair(qb),True),R))
    source=add(SZ,mat(transpose(pair(qb),True),pair(np.kron(np.eye(2),base[10]))))
    K,DK,Q,DQ,R,DR,KC,source,MZZ=(a+b for a,b in (K,DK,Q,DQ,R,DR,KC,source,MZZ))
    arrays=[a.copy() for a in base[:-1]]
    k,dk,q,dq,r,dr,qt,dqt,S,dS,e,de=arrays
    k[:nc,:nc]=K;k[nc:,:nc]=KC[:nz];k[nc:,nc:]=MZZ[:nz,:nz]
    dk[:nc,:nc]=DK;dk[:nc,nc:]=ds*KC[:nz].T
    q[:,:nc]=Q;dq[:,:nc]=DQ;dq[:,nc:]=ds*qb[:,:nz]
    r[:,:nc]=R;dr[:,:nc]=DR;dr[:,nc:]=ds*source[:nz,rev].T
    qt[:nc]=Q.conj().T;dqt[:nc]=DQ.conj().T
    S[:nc]=R[rev].T;dS[:nc]=DR[rev].T;S[nc:]=source[:nz]
    return ((*arrays,info),KC[nz:].T,MZZ[:nz,nz:],Qbar,
            source[nz:,rev].T,MZZ[nz:,nz:])


def response(obj,settings,base,transform,chart,order,T,Qbar,records,edge_rows,*,assembled=None,method='auto'):
    info=base[-1];nc,N,ds=info['nc'],info['size'],info['ds'];fields=info['corner_fields'];size=2*N
    if method not in ('auto','metric','retained'):
        raise ValueError('unknown complete contact elimination')
    retained=(method=='retained' or (method=='auto' and T is None and 2*size<nc))
    if method=='retained' and T is not None:
        raise ValueError('retained contact elimination does not consume a common frame')
    free,dfree,_=free_moments(obj,settings,base)
    # The free block is multiplied by large complete Fourier coefficients.
    # Keep its low part through projection, just as for K, Q and R. Original
    # physical checks consume the rounded directly integrated block.
    base=(*base[:10],free[0]+free[1],dfree[0]+dfree[1],info)
    K=pair(np.zeros((nc,nc),complex));Q=pair(np.zeros((size,nc),complex));R=pair(np.zeros((size,nc),complex));DQ=pair(np.zeros((size,nc),complex))
    DK=pair(np.zeros((nc,nc),complex));DR=pair(np.zeros((size,nc),complex))
    nz=info['nz'];total=nz+Qbar.shape[1];qb=np.hstack([base[6][nc:].conj().T,Qbar]);L=pair(np.zeros((total,nc),complex));SZ=pair(np.zeros((total,size),complex));MZZ=pair(np.zeros((total,total),complex))
    geometry=pair(np.zeros((nc,nc))) if T is None and not retained else None
    rows=iter(records())
    nx,ny=obj.order;modes=np.array([(i,j) for j in range(-ny,ny+1) for i in range(-nx,nx+1)])
    length,s,zero,velocity=obj._chart_geometry(0.,{k:np.asarray(settings[k]) for k in ('basis','widths','height','offset')})
    for tri,local in zip(obj.data['triangles'],info['local']):
        indices,common=local_coordinates(fields,tri,nc)
        local_count=len(indices);square=np.ix_(indices,indices);columns=(slice(None),indices)
        p=zero[tri[:3]];dp=ds*velocity[tri[:3]]
        area,slope=triangle_measures(p,velocity[tri[:3]])
        if local[11]:
            j1=np.column_stack([velocity[tri[1]]-velocity[tri[0]],velocity[tri[2]]-velocity[tri[0]]]);inv=np.array([[j1[1,1],-j1[0,1]],[-j1[1,0],j1[0,0]]])/local[10];dinv=np.zeros_like(inv)
        else:
            jac=np.column_stack([p[1]-p[0],p[2]-p[0]]);djac=np.column_stack([dp[1]-dp[0],dp[2]-dp[0]]);inv=np.linalg.inv(jac);dinv=-inv@djac@inv
        for bary,weight in quadrature(fields,tri,order):
            row,_,_,phi,dphi,added=next(rows);_,ci,zi,w,dw,ep,dep,phase,dphase,gc,dgc,gz,*_=row
            for first in range(0,len(bary),512):
                sl=slice(first,first+512);samples=common.sample(tri,bary[sl],inv,dinv)
                def grad(offset):return tuple(np.stack([getattr(samples[offset+a],part) for a in range(2)],axis=1).reshape(-1,local_count) for part in ('hi','lo'))
                g,dg=grad(0),grad(2);ph,dph=phases(bary[sl],p,dp,modes,chart[4])
                mw=weight.exact[sl]*area;mdw=weight.exact[sl]*D(ds)*slope
                ww=tuple(np.repeat(v,2)[:,None] for v in mw.pair)
                dww=tuple(np.repeat(v,2)[:,None] for v in mdw.pair)
                chi=np.zeros((len(bary[sl]),2,total));chi[:,:,zi]=gz[sl];chi[:,:,nz:]=added[sl]
                chi=pair(chi.reshape(-1,total))
                if not local[11]:
                    chi=add(chi,neg(mat(ph,pair(qb))));_real_geometry(chi[0]+chi[1]);chi=tuple(v.real for v in chi)
                mw=weight.exact[sl]*(slope if local[11] else area)
                measure=tuple(np.repeat(v,2)[:,None] for v in mw.pair)
                L=add_at(L,columns,cp(pair(ep),mat(transpose(chi),cp(measure,g))))
                SZ=add(SZ,cp(pair(ep),mat(transpose(chi),cp(measure,ph))))
                if local[11]:
                    MZZ=add(MZZ,cp(pair(ep),mat(transpose(chi),cp(measure,chi))))
                    DK=add_at(DK,square,cp(pair(ds),cp(pair(ep),mat(transpose(g),cp(measure,g)))))
                if not local[11]:
                    wg=cp(ww,g)
                    geometric=mat(transpose(g),wg)
                    if geometry is not None:geometry=add_at(geometry,square,geometric)
                    K=add_at(K,square,cp(pair(ep),geometric))
                    q=mat(transpose(ph,True),wg);Q=add_at(Q,columns,q);R=add_at(R,columns,cp(pair(ep),q))
                    partial=add(mat(transpose(dph,True),wg),mat(transpose(ph,True),cp(ww,dg)))
                    DQ=add_at(DQ,columns,partial);DR=add_at(DR,columns,cp(pair(ep),partial),cp(pair(dep),q))
                    DK=add_at(DK,square,cp(pair(ep),mat(transpose(dg),wg)),cp(pair(ep),mat(transpose(g),cp(ww,dg))),cp(pair(dep),mat(transpose(g),wg)),cp(pair(ep),mat(transpose(g),cp(dww,g))))
                moment=mat(transpose(ph,True),cp(dww,g));DQ=add_at(DQ,columns,moment);DR=add_at(DR,columns,cp(pair(ep),moment))
    nx,ny=obj.order;modes=np.array([(i,j) for j in range(-ny,ny+1) for i in range(-nx,nx+1)]);waves=2*np.pi*modes/chart[4];trace,tg=chart[8:]
    for ci,ep,p,gc,weight,edge,tri,bary,inverse in edge_rows(obj,settings,base,chart,order,coordinates=True):
        indices,common=local_coordinates(fields,tri,nc);columns=(slice(None),indices)
        phase=np.exp(1j*p@waves.T);ph=np.zeros((len(p),size),complex);ph[:,N:]=phase
        rawjump=trace._basis(np.array([edge]),tg)[0][0]@transform[2];jump=np.r_[np.zeros(nz),rawjump]
        samples=common.sample(tri,bary,inverse,np.zeros_like(inverse));gy=samples[1].pair
        left=cp(pair(weight[:,None]),pair(np.broadcast_to(jump,(len(p),total))))
        L=add_at(L,columns,cp(pair(ep),mat(transpose(left),gy)))
        SZ=add(SZ,cp(pair(ep),mat(transpose(left),pair(ph))))
    if assembled is not None:
        assembled['data']=physical_data(base,Qbar,K,DK,Q,DQ,R,DR,L,SZ,MZZ)
    if retained:
        from .pair_contact_retained import response as retained_response
        return retained_response(base,free,dfree,K,DK,Q,DQ,R,DR,L,SZ,MZZ,qb,assembled)
    if assembled is not None:
        assembled['elimination']='physical_projected'
        assembled['contact_solve_size']=nc
        assembled['common_geometric_frame']='complete_metric' if T is None else 'provided'
    # Integrate local support first, then apply the complete invertible
    # common transform. High/low matrices retain cancellations in both steps.
    if T is None:
        # The full closed geometric metric is an invertible coordinate map,
        # not a modification of the physical material or its retained space.
        from .pair_positive_precision import cholesky
        geometry=add(geometry,neg(mat(transpose(Q,True),Q)))
        _real_geometry(geometry[0]+geometry[1])
        lower=cholesky(D(*(v.real for v in geometry)))
        T=np.linalg.solve(lower.hi,np.eye(nc)).T
    tt=pair(T);K=mat(transpose(tt),mat(K,tt));DK=mat(transpose(tt),mat(DK,tt))
    Q,R,DQ,DR,L=(mat(a,tt) for a in (Q,R,DQ,DR,L))
    A=tuple(np.kron(np.eye(2),v) for v in free);DA=tuple(np.kron(np.eye(2),v) for v in dfree)
    QT=transpose(Q,True);DQT=transpose(DQ,True)
    S=tuple(v[info['reverse']].T for v in R);DS=tuple(v[info['reverse']].T for v in DR)
    M=add(K,neg(mat(S,Q)),neg(mat(QT,add(R,neg(mat(A,Q))))))
    DM=add(DK,neg(mat(DS,Q)),neg(mat(S,DQ)),neg(mat(DQT,R)),neg(mat(QT,DR)),mat(mat(DQT,A),Q),mat(mat(QT,DA),Q),mat(mat(QT,A),DQ))
    SC=add(S,neg(mat(QT,A)));DSC=add(DS,neg(mat(DQT,A)),neg(mat(QT,DA)))
    MZ=add(L,neg(mat(SZ,Q)))
    uc=solve(M,neg(SC));uz=solve(MZZ,neg(add(SZ,mat(MZ,uc))))
    duc=solve(M,neg(add(DSC,mat(DM,uc),cp(pair(ds),mat(transpose(MZ),uz)))))
    xx=add(pair(np.eye(size)),neg(mat(Q,uc)))
    dx=neg(add(mat(DQ,uc),mat(Q,duc),cp(pair(ds),mat(pair(qb),uz))))
    H=add(A,mat(tuple(v[:,info['reverse']].T for v in SC),uc))
    DH=add(DA,mat(tuple(v[:,info['reverse']].T for v in DSC),uc),mat(tuple(v[:,info['reverse']].T for v in SC),duc),cp(pair(ds),mat(tuple(v[:,info['reverse']].T for v in SZ),uz)))
    common=mat(pair(T),uc);dcommon=mat(pair(T),duc)
    U=np.vstack([common[0]+common[1],uz[0]+uz[1]])
    if assembled is not None:
        # The independent physical verifier receives the complete coefficients,
        # not the rounded diagnostics or the candidate tensor/JVP. Hold the
        # common potential fixed when differentiating its retained constraint.
        geometry_dx=neg(add(mat(DQ,uc),cp(pair(ds),mat(pair(qb),uz))))
        potential=tuple(np.vstack([c,z]) for c,z in zip(common,uz))
        assembled['energy_coordinates']=(xx,geometry_dx,potential)
    return (U,dcommon[0]+dcommon[1],xx[0]+xx[1],dx[0]+dx[1],H[0]+H[1],DH[0]+DH[1],float(np.linalg.cond(M[0]+M[1])))
