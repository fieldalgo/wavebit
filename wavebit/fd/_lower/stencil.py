"""The research symmetric extended-stencil material operator, in preparation but not routed.

This is the operator ``docs/fd-symmetric-stencil.md`` derives and measures, moved here from
``studies/fdtd_subpixel_symmetric_stencil.py`` unchanged. **Nothing public reaches it**: no
``plan()`` call, no ``params`` key, no kernel branch. The study still owns the scene scope
refusals, the surface clearance rule, the term and ``replace``; it imports these five
functions from here instead of defining them.

The move is the groundwork, and it is the whole of this commit: every number the study
produces is the number it produced before, byte for byte, which is what a move has to be able
to say for itself (memory ``a-pure-move-is-provable``).

**What it does.** At each target edge the interface normal is frozen and the exact constant
tangential-E / normal-D states are integrated over every nearby edge and dual face. ``fit``
solves the row constraints ``sum_j K_ij Pi_j(n_i) = Gamma_i(n_i)`` for a symmetric ``K``,
minimising a weighted Frobenius correction to the symmetric triplet baseline;
``triplet_baseline`` builds that baseline, ``geometry`` assembles the two- and
three-dimensional cases from a node lattice, ``energy_bounds`` certifies the assembled
operator, and ``padded`` compresses it to its active rows. Positive energy is not assumed
anywhere: the whole assembled operator is inspected before any field is advanced.

**The seam left open.** ``geometry`` and ``triplet_baseline`` still call into ``studies/`` for
the exact edge and face fraction helpers and for the triplet orientation maps, so those two
import it where they use it and fail outside a checkout that has ``studies`` importable. That
is deliberate and temporary: it keeps this commit a move and nothing else. Importing this
module costs nothing and pulls no study -- ``tests/test_fd_symmetric_stencil.py`` pins that --
and ``fit``, ``energy_bounds`` and ``padded`` run with no study at all.

**Host code, not backend code.** This assembles on SciPy, on the host, in float64. It is not
the traced, differentiable geometry path and cannot silently stand in for one: an input that
arrives as a JAX tracer or a Torch tensor is refused at the study's door, not discretised
some other way.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lsmr, eigsh

__all__ = ['energy_bounds', 'fit', 'geometry', 'padded', 'triplet_baseline']


def fit(points, axes, normals, signed, ell, area, eps, reach=2., periodic=None, baseline=None, lattice=None):
    # The one-cell-radius screen had minimum eigenvalue -5.84418e-4 at
    # radius 12 cells. Two cells gave +.02390 there and positive spectra
    # at radii 4/6/8. This chooses a research stencil, not a universal bound.
    dim=points.shape[1];count=len(points)
    base=np.where(signed<0,1/eps,1.)
    if baseline is None: baseline=diags(base).tocsr()
    # A dual face extends sqrt(dim-1)/2 cells from its midpoint. Include
    # every edge within one stencil radius of any potentially cut face.
    width=reach+np.sqrt(dim-1)/2
    # The hard radial support produced a 1.41418e-3 operator change under
    # a rigid world translation. Give a correction vanishing value AND
    # derivative at that artificial boundary. The lowest-degree even
    # polynomial with w(0)=1 and w(+-1)=w'(+-1)=0 is (1-q^2)^2.
    # This is a support regularizer; it is not fitted to any scattering data.
    weight=np.maximum(0.,1-(signed/width)**2)**2
    ids=np.flatnonzero(weight>0)
    pairs=[(i,i) for i in ids]
    # Stencil neighbours are lattice topology. World-coordinate subtraction
    # made an exactly two-cell separation round above/below the cutoff.
    # Geometry supplies exact integer/half-integer lattice positions here.
    lattice=points if lattice is None else lattice
    positions=lattice[ids] if periodic is None else lattice[ids]%periodic
    pairs.extend((ids[i],ids[j]) for i,j in sorted(cKDTree(positions,boxsize=periodic).query_pairs(reach)))
    rowids={idx:i for i,idx in enumerate(ids)}
    # Preallocated rather than appended. The entry counts are exact, not estimates: one
    # dim-block of constraint entries per diagonal pair and two per off-diagonal one, so
    # stored = 2*len(pairs) - len(ids) and entries = dim*stored. The 56-cell sphere reached
    # 10,840,896 of them in each of rr/cc/vv (1,826,928 pairs, 40,224 active rows), every one
    # of which was a boxed Python float or int. The arithmetic below is unchanged: each slice
    # assignment evaluates the same product in the same order as the scalar append it
    # replaces, and the order entries reach coo_matrix is the same, so the assembled matrix
    # is bit identical at a fixed BLAS thread count.
    stored=2*len(pairs)-len(ids)
    rhs=np.empty(dim*len(ids))
    rr=np.empty(dim*stored,dtype=np.int64);cc=np.empty_like(rr);vv=np.empty(dim*stored)
    span=np.arange(dim)
    for row,i in enumerate(ids):
        c=axes[i];n=normals[i];nn=np.outer(n,n);pt=np.eye(dim)-nn
        gamma=np.eye(dim)[c]+(1/eps-1)*ell[i]*nn[c]
        a,b=baseline.indptr[i:i+2];js=baseline.indices[a:b]
        pis=np.eye(dim)[axes[js]]+(eps-1)*area[js,None]*pt[axes[js]]
        rhs[dim*row:dim*row+dim]=gamma-baseline.data[a:b]@pis
    at=0
    for k,(i,j) in enumerate(pairs):
        for target,source in ((i,j),) if i==j else ((i,j),(j,i)):
            c=axes[source];n=normals[target];pt=np.eye(dim)-np.outer(n,n)
            pi=np.eye(dim)[c]+(eps-1)*area[source]*pt[c]
            rr[at:at+dim]=dim*rowids[target]+span;cc[at:at+dim]=k
            vv[at:at+dim]=pi*weight[i]*weight[j]/(1. if i==j else np.sqrt(2.))
            at+=dim
    if at!=dim*stored: raise ValueError(f'StencilAssemblyRefused: {at} of {dim*stored} entries')
    constraints=coo_matrix((vv,(rr,cc)),shape=(dim*len(ids),len(pairs))).tocsr()
    # Floating-point solver request; actual residual and stop code retained.
    # The initial static screen needed at most 563/340 = 1.656 iterations
    # per equation. Round that measured iteration cap upward to two.
    # Exact row scaling leaves feasible solutions and their least norm
    # unchanged. It balances rows whose support weight approaches zero.
    rownorm=np.sqrt(np.asarray(constraints.multiply(constraints).sum(axis=1)).ravel())
    scale=np.divide(1.,rownorm,out=np.ones_like(rownorm),where=rownorm>0)
    balanced=diags(scale)@constraints
    solved=lsmr(balanced,rhs*scale,atol=np.finfo(float).eps,btol=np.finfo(float).eps,
                conlim=1/np.finfo(float).eps,maxiter=2*len(rhs))
    err=constraints@solved[0]-rhs
    rows=np.empty(stored,dtype=np.int64);cols=np.empty_like(rows);values=np.empty(stored)
    at=0
    for k,(i,j) in enumerate(pairs):
        v=solved[0][k]*weight[i]*weight[j]/(1. if i==j else np.sqrt(2.))
        rows[at]=i;cols[at]=j;values[at]=v;at+=1
        if i!=j:
            rows[at]=j;cols[at]=i;values[at]=v;at+=1
    if at!=stored: raise ValueError(f'StencilAssemblyRefused: {at} of {stored} stored entries')
    matrix=(baseline+coo_matrix((values,(rows,cols)),shape=(count,count))).tocsr()
    low,vec=eigsh(matrix,k=1,which='SA')
    residual=float(np.linalg.norm(matrix@vec[:,0]-low[0]*vec[:,0]))
    # Symmetric row absolute-sum bound is an actual upper spectral bound.
    upper=float(np.max(np.asarray(abs(matrix).sum(axis=1))))
    report=dict(unknowns=len(pairs),equations=len(rhs),lsmr_stop=int(solved[1]),
        iterations=int(solved[2]),residual_max=float(abs(err).max(initial=0.)),
        residual_rms=float(np.sqrt(np.mean(err**2))),
        minimum_eigenvalue=float(low[0]),eigenpair_residual=residual,
        spectral_upper_bound=upper,reach=reach,active_edges=len(ids),
        weighted_frobenius_correction=float(np.linalg.norm(solved[0])),
        frobenius_correction=float(np.linalg.norm((matrix-baseline).data)))
    return matrix,report


def energy_bounds(matrix):
    """Whole symmetric operator, sparse LDL and comparison-inverse bound."""
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError('StencilFiniteRefused')
    difference=matrix-matrix.T
    difference.eliminate_zeros()
    if difference.nnz:
        raise ValueError('StencilSymmetryRefused')
    diagonal=matrix.diagonal()
    cross=matrix-diags(diagonal);cross.eliminate_zeros()
    coupled=np.flatnonzero(np.diff(cross.indptr)>0)
    untouched=np.setdiff1d(np.arange(len(diagonal)),coupled)
    lower_bound=float(np.min(diagonal[untouched],initial=np.inf))
    relative_residual=0.
    if len(coupled):
        from scipy.sparse import eye
        from scipy.sparse.linalg import splu,spsolve_triangular
        block=matrix[coupled][:,coupled].tocsc()
        try:
            factor=splu(block,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,
                        options={'SymmetricMode':True})
        except RuntimeError as error:
            raise ValueError('StencilEnergyRefused: sparse factorization failed') from error
        if not np.array_equal(factor.perm_r,factor.perm_c):
            raise ValueError('StencilEnergyRefused: factorization is not symmetrically ordered')
        order=np.argsort(factor.perm_r)
        target=block[order][:,order]
        pivots=factor.U.diagonal()
        if not np.all(np.isfinite(pivots)&(pivots>0)):
            raise ValueError('StencilEnergyRefused: nonpositive LDL pivot')
        # Reconstruct in sqrt(N)-row blocks to keep temporary storage below
        # a dense N-by-N array. This changes storage, not the residual norm.
        residual_norm=0.
        step=max(1,int(np.sqrt(len(coupled))))
        weighted=factor.L@diags(pivots)
        for start in range(0,len(coupled),step):
            stop=min(start+step,len(coupled))
            residual=target[start:stop]-weighted[start:stop]@factor.L.T
            residual_norm=max(residual_norm,float(np.max(np.asarray(abs(residual).sum(axis=1)))))
        block_norm=float(np.max(np.asarray(abs(block).sum(axis=1))))
        relative_residual=residual_norm/block_norm
        # For unit lower L, |L^-1| <= (I-|strict_lower(L)|)^-1 entrywise.
        # Positive triangular solves bound its row and column sums exactly
        # in real arithmetic, without constructing a dense inverse.
        unit=eye(len(coupled),format='csc')
        comparison=unit-abs(factor.L-unit)
        row=spsolve_triangular(comparison,np.ones(len(coupled)),lower=True,unit_diagonal=True)
        column=spsolve_triangular(comparison.T,np.ones(len(coupled)),lower=False,unit_diagonal=True)
        if not np.all(np.isfinite(row)&(row>=1)) or not np.all(np.isfinite(column)&(column>=1)):
            raise ValueError('StencilEnergyRefused: nonfinite comparison-inverse bound')
        lower_bound=min(lower_bound,float(min(pivots)/(max(row)*max(column))-residual_norm))
    if lower_bound<=0 or not np.isfinite(lower_bound):
        raise ValueError('StencilEnergyRefused: nonpositive residual-adjusted bound')
    return dict(energy_lower_bound=float(lower_bound),factor_relative_residual=relative_residual,
                spectral_upper_bound=float(np.max(np.asarray(abs(matrix).sum(axis=1)))))


def triplet_baseline(nodes,dx,center,radius,eps,components):
    from studies import fdtd_subpixel_triplet_cylinder as tc
    from studies import fdtd_subpixel_triplet_sphere as ts
    dim=len(components);study=tc if dim==2 else ts
    _,blocks=(study.circle_maps if dim==2 else study.sphere_maps)(nodes,dx,center,radius,eps)
    shape=nodes.shape[:-1];size=int(np.prod(shape));ids=np.arange(size).reshape(shape)
    # Preallocated at the exact upper bound and sliced to what was written: each of the
    # len(ORIENTATIONS)*dim*dim blocks covers at most the whole lattice, and covers less
    # exactly where a rolled component has no source cell. The 56-cell sphere reached
    # 10.8M appended entries here as well; nothing about the values or their order changes.
    rows=np.empty(len(study.ORIENTATIONS)*dim*dim*size,dtype=np.int64)
    cols=np.empty_like(rows);vals=np.empty(len(rows))
    at=0
    for k,orient in enumerate(study.ORIENTATIONS):
        index=[];valid=[]
        for i,sign in enumerate(orient):
            axis=2-components[i]
            src=np.roll(ids,1,axis=axis) if sign<0 else ids
            mask=np.ones(shape,dtype=bool)
            if sign<0:
                part=[slice(None)]*3;part[axis]=0;mask[tuple(part)]=False
            index.append(src);valid.append(mask)
        for i in range(dim):
            for j in range(dim):
                mask=valid[i]&valid[j];taken=int(mask.sum())
                rows[at:at+taken]=(i*size+index[i][mask]).ravel();cols[at:at+taken]=(j*size+index[j][mask]).ravel()
                vals[at:at+taken]=((blocks[k,...,i,j]-(1 if i==j else 0))/len(study.ORIENTATIONS))[mask].ravel()
                at+=taken
    return (diags(np.ones(dim*size))+coo_matrix((vals[:at],(rows[:at],cols[:at])),shape=(dim*size,dim*size))).tocsr()


def geometry(nodes, dx, center, radius, eps, components, baseline_kind='triplet'):
    # Deferred, not top level: the exact fraction helpers still live under studies/
    # (see the module docstring), and this module must import without them.
    from studies.fdtd_subpixel_triplet_cylinder import circle_fraction
    from studies.fdtd_subpixel_triplet_sphere import edge_fraction,face_fraction
    dim=len(components);shape=nodes.shape[:-1];count=int(np.prod(shape))
    pts=[];axes=[]
    lattice=[]
    integer=np.indices(shape)
    integer=np.stack(tuple(integer[2-c] for c in components),axis=-1).reshape(-1,dim)
    for c in range(dim):
        p=(nodes.reshape(-1,dim)-np.asarray(center))/dx
        p[:,c]+=.5;pts.append(p);axes.extend([c]*count)
        local=integer.astype(float);local[:,c]+=.5;lattice.append(local)
    points=np.concatenate(pts);axes=np.array(axes)
    norm=np.linalg.norm(points,axis=-1);signed=norm-radius/dx
    normals=points/np.where(norm>0,norm,1.)[:,None]
    normals[norm==0]=np.eye(dim)[0]
    ell=np.zeros(len(points));area=ell.copy()
    for c in range(dim):
        idx=axes==c
        if dim==2:
            ell[idx]=circle_fraction(points[idx],1.,c,radius/dx)
            area[idx]=circle_fraction(points[idx],1.,1-c,radius/dx)
        else:
            ell[idx]=edge_fraction(points[idx],1.,c,radius/dx)
            area[idx]=face_fraction(points[idx],1.,c,radius/dx)
    baseline=triplet_baseline(nodes,dx,center,radius,eps,components) if baseline_kind=='triplet' else None
    return fit(points,axes,normals,signed,ell,area,eps,baseline=baseline,lattice=np.concatenate(lattice))


def padded(matrix):
    """Pad only nonzero off-diagonal rows, with an exact zero output slot."""
    diagonal=matrix.diagonal();cross=matrix-diags(diagonal);cross.eliminate_zeros()
    active=np.flatnonzero(np.diff(cross.indptr))
    width=int(np.diff(cross.indptr).max(initial=0))
    index=np.zeros((len(active),width),dtype=np.int32);value=np.zeros_like(index,dtype=float)
    output=np.zeros(matrix.shape[0],dtype=np.int32)
    output[active]=np.arange(1,len(active)+1,dtype=np.int32)
    for row,i in enumerate(active):
        a,b=cross.indptr[i:i+2];index[row,:b-a]=cross.indices[a:b];value[row,:b-a]=cross.data[a:b]
    # Radius-10 premeasurement: 73,693,152 -> 26,385,540 bytes, 2.75x
    # isolated JAX product speed, and exactly unchanged product values.
    # Keep every stored nonzero; this is storage compression, not a cutoff.
    return diagonal,index,value,output
