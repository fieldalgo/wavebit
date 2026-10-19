"""The symmetric extended-stencil material operator, and the preparation a scene asks for.

This is the operator ``docs/design/fd-symmetric-stencil.md`` derives and measures. A structure asks
for it by name -- ``fd.Structure(shape, material, discretisation='stencil')`` -- and
:func:`prepare` assembles it **once, eagerly, on the host**, at plan time, for that structure
alone. A structure that does not ask, and every scene that holds none that do, is untouched:
the default is exactly the rule it was, to the bit, and ``tests/fdtd/test_fd_stencil.py`` proves
that against a ``git archive`` of the commit before this one.

**Preparation is plan time, never lowering.** ``fit`` is SciPy on the host in float64 -- a
sparse least-squares solve, a symmetric eigenpair, a sparse factorization -- and none of the
three has a JAX or a Torch spelling. Assembling in ``plan()`` is what keeps NumPy, JAX eager,
``jax.jit`` and Torch identical: the solve is upstream of every trace, the way the CPML
stretch profile already is (``src/wavebit/fdtd/_prepare/cpml.py``). :func:`lower` only reads the
prepared arrays, moves them to the backend and patches the coefficients out of place.

**What it does.** At each target edge the interface normal is frozen and the exact constant
tangential-E / normal-D states are integrated over every nearby edge and dual face. ``fit``
solves the row constraints ``sum_j K_ij Pi_j(n_i) = Gamma_i(n_i)`` for a symmetric ``K``,
minimising a weighted Frobenius correction to the symmetric triplet baseline;
``triplet_baseline`` builds that baseline, ``geometry`` assembles the two- and
three-dimensional cases from a node lattice, ``energy_bounds`` certifies the assembled
operator, and ``padded`` compresses it to its active rows. Positive energy is not assumed
anywhere: the whole assembled operator is inspected before any field is advanced.

**No study behind it (2026-09-26).** ``geometry`` takes its exact edge and face fractions
from ``src/wavebit/fdtd/_shapes``'s :class:`~wavebit.fdtd._shapes.Sphere`, and ``triplet_baseline``
its orientation table and local maps from :mod:`~wavebit.fdtd._prepare.triplet`. Both used to be
imported out of ``studies/`` at plan time, so ``discretisation='stencil'`` ran from a
repository checkout and not from an installed wheel. Both were moved into the package
unchanged: the fractions onto ``Sphere``, over which the sphere study's ``edge_fraction`` and
``face_fraction`` stay thin wrappers, and the maps into ``triplet``, whose objects the two
triplet studies now hand out under their old names.
``tests/fdtd/test_fd_symmetric_stencil.py`` pins that importing this module and assembling an
operator pull no study, and that the operator built with ``studies`` hidden is the one built
beside it, bit for bit.

**Host code, not backend code.** This assembles on SciPy, on the host, in float64. It is not
the traced, differentiable geometry path and cannot silently stand in for one. Of the leaves
the operator reads, two kinds reach it, and each has one door:

* **The ball's radius and centre are static fields of** ``fd.Sphere``, concretised at
  construction by ``_spec.static_float``, which refuses every traced value by name -- a JAX
  tracer, a ``requires_grad`` leaf, a forward-mode dual, a ``torch.func`` wrapper
  (``_spec.torch_carried``). They arrive here as Python floats, on either discretisation,
  and no door in this module asks about them again: an arm that could never fire was the
  third review's finding against ``b24a708``.
* **The material values are gradient leaves** everywhere else, so a structure that asks for
  the stencil takes **plain numbers only** in its ``eps`` and ``mu`` and the background's --
  Python ``int`` and ``float`` and NumPy real scalars (:func:`_plain`) -- and any array or
  tensor of any library, traced or not, is refused **by name** (``StencilNumberRefused``) at
  :func:`prepare`'s door and again at :func:`lower`'s, not discretised some other way.
  Deciding by type rather than by "is it traced" is what closed that class: a Torch
  forward-mode dual reads as a plain number through ``float()`` and passed a traced-or-not
  test at both doors, and ``fd.run`` returned a tangent of exactly 0.0 where the central
  difference is 5.64e-9.
"""
import hashlib

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse import diags
from scipy.sparse.linalg import lsmr
from scipy.sparse.linalg import eigsh

from wavebit.fdtd._shapes import Sphere
from wavebit.fdtd._prepare.triplet import PAIRS
from wavebit.fdtd._prepare.triplet import TRIPLETS
from wavebit.fdtd._prepare.triplet import circle_maps
from wavebit.fdtd._prepare.triplet import sphere_maps
from wavebit.fdtd._public_names import public_description, public_name

__all__ = ['DEFAULT', 'MEMORY_BOUND_BYTES', 'REACH', 'STENCIL', 'active_rows', 'arguments',
           'energy_bounds', 'fit', 'geometry', 'lower', 'operator_digest', 'padded',
           'predicted_bytes', 'prepare', 'prepared_digest', 'refuse_discretisation',
           'source_digest', 'surface_clearance', 'triplet_baseline']

#: The names :func:`fit` reads a complex value out of, and refuses. Each one reaches the
#: preallocated ``float64`` buffers below through ``gamma``, ``pis`` or ``pi``.
_REAL_INPUTS = ('eps', 'ell', 'area', 'normals', 'baseline')


def fit(points, axes, normals, signed, ell, area, eps, reach=2., periodic=None, baseline=None, lattice=None):
    # Real inputs only, refused by name once per preparation and never per cell. The buffers
    # below are preallocated float64, and a complex eps -- a lossy material -- would reach them
    # through gamma, pis and pi, where numpy casts it down with a ComplexWarning rather than an
    # error: the imaginary part would be dropped silently and the operator would look fine.
    # This is a refusal and not a promotion to complex because the whole routine is real
    # symmetric by construction -- lsmr solves a real least-squares problem, eigsh(which='SA')
    # asks for a smallest algebraic eigenvalue, and energy_bounds certifies an LDL with positive
    # pivots -- so a complex eps has no meaning here even in complex buffers. The public route
    # cannot deliver one today -- fd.Material refuses a complex permittivity and prepare()
    # refuses a Conductor or a Dispersive by name -- and this says so by name as well, rather
    # than by the absence of a caller.
    complex_inputs = [name for name, value in
                      zip(_REAL_INPUTS, (eps, ell, area, normals,
                                         None if baseline is None else baseline.data))
                      if value is not None and np.iscomplexobj(value)]
    if complex_inputs:
        raise ValueError(f'StencilComplexRefused: the extended stencil is assembled in real '
                         f'float64 and is real symmetric throughout; {", ".join(complex_inputs)} '
                         f'{"is" if len(complex_inputs) == 1 else "are"} complex. A lossy '
                         f'material has no operator here -- it is not that it is truncated.')
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
        from scipy.sparse.linalg import splu
        from scipy.sparse.linalg import spsolve_triangular
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
    dim=len(components)
    orientations,maps=(PAIRS,circle_maps) if dim==2 else (TRIPLETS,sphere_maps)
    _,blocks=maps(nodes,dx,center,radius,eps)
    shape=nodes.shape[:-1];size=int(np.prod(shape));ids=np.arange(size).reshape(shape)
    # Preallocated at the exact upper bound and sliced to what was written: each of the
    # len(orientations)*dim*dim blocks covers at most the whole lattice, and covers less
    # exactly where a rolled component has no source cell. The 56-cell sphere reached
    # 10.8M appended entries here as well; nothing about the values or their order changes.
    rows=np.empty(len(orientations)*dim*dim*size,dtype=np.int64)
    cols=np.empty_like(rows);vals=np.empty(len(rows))
    at=0
    for k,orient in enumerate(orientations):
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
                vals[at:at+taken]=((blocks[k,...,i,j]-(1 if i==j else 0))/len(orientations))[mask].ravel()
                at+=taken
    return (diags(np.ones(dim*size))+coo_matrix((vals[:at],(rows[:at],cols[:at])),shape=(dim*size,dim*size))).tocsr()


def geometry(nodes, dx, center, radius, eps, components, baseline_kind='triplet'):
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
    # The public closed forms, not the wrappers over them (triplet.circle_fraction, the sphere
    # study's edge_fraction and face_fraction): a ball's chord along a component's edge and its
    # cross-section in the component's dual face. Everything here is in cell units, so the body
    # is a sphere of radius/dx at the origin and the cell is one wide. In two dimensions the
    # circle is this sphere's central plane -- the lift to (0, y, z) makes the transverse sum
    # 0*0 + t*t, which is the t**2 the circle form wrote -- so the same edge fraction answers
    # both the length and the transverse area there.
    ball=Sphere(radius/dx)
    for c in range(dim):
        idx=axes==c
        if dim==2:
            plane=points[idx]
            lifted=np.stack((np.zeros_like(plane[...,0]),plane[...,0],plane[...,1]),axis=-1)
            ell[idx]=ball.exact_edge_fill(lifted,1.,c+1)
            area[idx]=ball.exact_edge_fill(lifted,1.,(1-c)+1)
        else:
            ell[idx]=ball.exact_edge_fill(points[idx],1.,c)
            area[idx]=ball.exact_face_fill(points[idx],1.,c)
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


def surface_clearance(scene,p,matrix,box,components):
    """Check actual material rows against snapped source/flux surfaces."""
    import wavebit.fdtd._legacy as fd
    from wavebit.fdtd._prepare.monitors import _box_nodes
    from wavebit.fdtd._prepare.monitors import _node
    delta=matrix-diags(np.ones(matrix.shape[0]));delta.eliminate_zeros()
    used=np.flatnonzero(np.diff(delta.indptr)>0)
    if not len(used):return []
    shape=tuple(hi-lo for lo,hi in box);size=int(np.prod(shape))
    z,y,x=np.unravel_index(used%size,shape)
    points=np.column_stack((x+box[2][0],y+box[1][0],z+box[0][0])).astype(float)
    points[np.arange(len(used)),np.asarray(components)[used//size]]+=.5
    low=points.min(axis=0);high=points.max(axis=0);reports=[]
    for monitor in scene.monitors:
        if not isinstance(monitor,(fd.Flux,fd.FarField)):continue
        if monitor.box is None:raise ValueError('StencilScopeRefused: research flux monitors need a closed box')
        lo,hi=_box_nodes(monitor.name,p.grid,monitor.box)
        inside=np.all((low>lo)&(high<hi))
        outside=np.any((high<lo)|(low>hi))
        if not (inside or outside):
            raise ValueError(f'StencilSurfaceRefused: {monitor.name} cuts nonlocal material support at nodes {low.tolist()}..{high.tolist()}')
        reports.append(dict(name=monitor.name,relation='encloses' if inside else 'excludes'))
    for source in scene.sources:
        if isinstance(source,fd.TFSF):
            lo,hi=_box_nodes(source.name,p.grid,source.box)
            # The existing TFSF face correction reads the node and adjacent
            # half plane. Reserve its one-cell injection band in vacuum.
            clear=np.all((low>np.asarray(lo)+1)&(high<np.asarray(hi)-1))
        elif isinstance(source,fd.PlaneWave):
            axis=source.axis;k=_node(p.grid,axis,source.position)
            clear=high[axis]<k-1 or low[axis]>k+1
        else:
            raise ValueError('StencilScopeRefused: only the research TFSF/PlaneWave sources are supported')
        if not clear:
            raise ValueError(f'StencilSourceRefused: {source.name} injection band meets nonlocal material support')
        reports.append(dict(name=source.name,relation='injection_band_clear'))
    return reports


# -- the public preparation ------------------------------------------------------------------

#: The two spellings of ``fd.Structure(discretisation=)``. ``DEFAULT`` is what every structure
#: has had until now and is *not* one rule: it is "the rule this structure's shape already
#: names" -- a whole-cell replace for a cell-snapped :class:`~wavebit.fdtd._legacy.Slab` or
#: :class:`~wavebit.fdtd._legacy.Region`, the diagonal Farjadpour/Kottke mix plus the constitutive cross
#: terms for a shape that carries a body. That is why the word is ``'voxel'`` and not the name
#: of a rule: both of those read one component's **own voxel** and nothing else.
DEFAULT = 'voxel'
STENCIL = 'stencil'

#: The stencil radius, in cells, that :func:`fit` screens neighbour pairs at -- its own default
#: and the value every research number was measured with. Its provenance is in ``fit``: a
#: one-cell radius read minimum eigenvalue -5.84418e-4 at radius 12 cells, two cells +0.02390
#: there and a positive spectrum at radii 4/6/8. It is a research stencil, not a universal
#: bound, and :func:`prepare` does not let a caller vary it.
REACH = 2.0

#: Pairs of coupled edges per active row -- ``fit``'s unknown count over its ``active_edges``.
#: It is a lattice-topology number, the count of lattice points within :data:`REACH` of a
#: point over the three component sublattices, so it depends on the shell's shape and not on
#: the radius. Measured 2026-09-24 on five ball preparations, 1830 to 7836 active rows:
#: **41.72, 42.98, 43.43, 44.33, 44.76**, rising with the shell's size and flattening; and
#: 2026-09-25 through the public ``fd.plan`` on the research epsilon-12 sphere: **45.35** at 40
#: cells (21,060 rows) and **45.42** at 56 (40,224 rows, the recorded research value).
#: Rounded **up** past all of them, because it multiplies a memory bound and a bound that is
#: occasionally low is not a bound.
PAIRS_PER_ACTIVE_ROW = 46.0

#: Bytes of peak host working set per unknown that the **whole** host preparation costs --
#: the solve, ``energy_bounds``' sparse LDL factorization, the padding and the halo check,
#: i.e. :func:`_assemble` -- over the process's working set just before it starts. Measured
#: 2026-09-25, each in its own process through ``fd.plan`` with OMP/OPENBLAS/MKL_NUM_THREADS=2,
#: from ``GetProcessMemoryInfo``'s ``PeakWorkingSetSize`` (the probe is recorded in
#: ``docs/design/fd-symmetric-stencil.md``):
#:
#:     scene                         placement      cells  unknowns   peak MB  bytes/unknown
#:     test fixture, ball .14, 2.25  origin            18    76,353        77        1,009.7
#:     ball .25, eps 2.25            origin            24   350,730       613        1,747.6
#:     ball .25, eps 2.25            origin            28   467,346       889        1,901.5
#:     ball .25, eps 2.25            origin            32   622,560     1,502        2,412.0
#:     research sphere, eps 12       (0, 0, 0)         40   954,978     2,423        2,537.2
#:     research sphere, eps 12       (0, 0, 0)         56 1,826,928     5,984        3,275.6
#:     research sphere, eps 12       (.5, 1/3, .2)     56 1,840,944     4,887        2,654.5
#:
#: (the last row measured by the review of ``199463e``, which also reproduced the 40-cell
#: origin row to 0.1%). Past the solve it is the factorization that sets the peak (the solve
#: alone read 605 and 556 bytes per unknown at 24 and 56 cells -- the 750 this constant held
#: before was measured over the solve alone and under-predicted the preparation by up to 4.4
#: times). The cost per unknown is **not** a function of the size alone: at 56 cells it read
#: 3,275.6 at one sub-cell placement of the same sphere and 2,654.5 at another, and the
#: placement with 0.8% *more* unknowns needed 18% *less* memory. The fill of a sparse factorization depends
#: on the ordering its heuristic finds for the pattern, which an unknown count does not fix.
#: So this is the largest of the seven measured values, rounded up: an upper bound over the
#: scenes measured, and a statement about no other -- past 56 cells nobody has measured
#: whether it is low or high.
BYTES_PER_UNKNOWN = 3300.0

#: The declared preparation bound: what one host preparation may be predicted to cost before
#: it is refused rather than started. A laptop that panics rather than slows is on record in
#: this repository (memory ``laptop-ram-panics-not-just-slows``), which is why the door is
#: here at all and why it is shut before the solve rather than after it.
#:
#: 6144 MiB. What it has to admit is the research sphere at every placement the convergence
#: orders were measured at -- ``(0, 0, 0)``, ``(1/2, 1/3, 1/5)`` and ``(1/4, 2/3, 2/5)`` voxels
#: (:data:`RESEARCH_PLACEMENTS`) at 40 and 56 cells (:data:`MEASURED_RESOLUTIONS`) -- because
#: owner decision (C)'s public spot check has to be possible at each. Their predictions, read
#: through the public selector with no solve (2026-09-25; the 40-cell placed spheres with the
#: research's own .45 absorption half-width, which the occupancy door requires):
#:
#:     cells  placement          active rows   predicted MiB
#:        40  (0, 0, 0)               21,060        3,048.81
#:        40  (1/2, 1/3, 1/5)         20,916        3,027.96
#:        40  (1/4, 2/3, 2/5)         20,906        3,026.51
#:        56  (0, 0, 0)               40,224        5,823.14
#:        56  (1/2, 1/3, 1/5)         40,481        5,860.34
#:        56  (1/4, 2/3, 2/5)         40,489        5,861.50
#:
#: The bound before this one was the first 56-cell row rounded up to the MiB, 5824, which
#: refused the other two 56-cell placements -- one of which needs 1.1 GB *less* than the one
#: it admitted. The prediction is the row count times two constants, and the row count moves
#: with sub-cell placement: over 1,006 placements at 56 cells (the three above, three
#: symmetric ones and 1,000 uniform random ones, seeded) it ran from 40,224 to 40,610 rows,
#: 5,879.02 MiB at the top. 6144 MiB is 4.5% above that largest one and 4.8% above the
#: largest research placement, i.e. room for 42,440 rows; at 40 cells the largest row count
#: over the same sweep was 21,060. The 80-cell preparation (about twice the 56-cell rows) is
#: not admitted: nothing has measured it. A structure raises the bound for itself with
#: ``fd.Structure(..., discretisation='stencil', memory_bound=<bytes>)``.
MEMORY_BOUND_BYTES = 6144 * 1024 ** 2

#: The unknown count of the largest preparation measured, the resolutions and the placements
#: of the research sphere the default bound has to admit -- what the refusal quotes, and what
#: ``tests/fdtd/test_fd_stencil.py`` asks the default bound to admit at every pair.
MEASURED_UNKNOWNS = 1840944
MEASURED_RESOLUTIONS = (40, 56)
RESEARCH_PLACEMENTS = ((0., 0., 0.), (.5, 1 / 3, .2), (.25, 2 / 3, .4))


def operator_digest(matrix):
    """``sha256`` over a CSR operator's shape and its three stored arrays.

    The bit-identity hook owner decision (C) rests on: the public preparation and the research
    driver call **one** :func:`geometry`, so showing they hand it the same arguments
    (:func:`arguments`) and that what comes back digests the same is the whole proof, and it
    costs one small assembly rather than the 40/56/80 sweep. Content only -- no ``id()``, no
    ``repr()``, no address (memory ``a-digest-you-did-not-validate``).
    """
    h = hashlib.sha256()
    m = matrix.tocsr()
    h.update(f'{m.shape}'.encode())
    for a in (m.data, m.indices, m.indptr):
        a = np.ascontiguousarray(a)
        h.update(f'{a.dtype.str}{a.shape}'.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def prepared_digest(index, value, output, diagonal):
    """``sha256`` over the four arrays a prepared operator reduces to.

    The other half of the bit-identity hook (:func:`operator_digest` digests the assembled
    CSR): once ``padded`` has compressed the operator, *this* is what a run actually reads, so
    two paths that produce the same four arrays produce the same run. It is what lets owner
    decision (C) rest on a small fixture -- the public preparation and the research driver's
    ``replace`` are compared here rather than by running the 40/56/80 sweep twice.

    ``diagonal`` is the diagonal **as it reaches** ``be``, already multiplied by the term's
    ratio, and not the diagonal ``padded`` returned. The two differ: recovering one from the
    other is a multiply followed by a divide, measured 1.11e-16 apart on the fixture, and a
    hook that has to insert a round trip to compare is not a bit-identity hook. What the run
    reads is ``be``, so that is what is digested.
    """
    h = hashlib.sha256()
    for a in (index, value, output, diagonal):
        a = np.ascontiguousarray(a)
        h.update(f'{a.dtype.str}{a.shape}'.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def source_digest(nodes, dx, center, radius, eps, components):
    """``sha256`` over the exact arguments :func:`geometry` is called with.

    The operator is a function of these and of nothing else -- the reach and the baseline are
    this module's constants, and both are in the hash -- so two preparations with the same
    source digest are the same solve. That is what lets ``fd.run(scene, plan=...)`` take the
    operator the plan already carries instead of solving it again (:func:`prepare`'s
    ``reuse``), and what lets :func:`lower` refuse a scene whose sphere or permittivity is not
    the one the operator was fitted for. Content only: the node lattice's bytes and the five
    numbers as float64 bytes, never a ``repr`` or an address.
    """
    h = hashlib.sha256()
    a = np.ascontiguousarray(nodes, dtype=np.float64)
    h.update(f'{a.dtype.str}{a.shape}'.encode())
    h.update(a.tobytes())
    numbers = np.asarray((dx,) + tuple(center) + (radius, eps, REACH), dtype=np.float64)
    h.update(numbers.tobytes())
    h.update(repr((tuple(int(c) for c in components), 'triplet')).encode())
    return h.hexdigest()


def active_rows(nodes, dx, center, radius, components, reach=REACH):
    """The number of target edges the operator will carry a correction on.

    The same screen :func:`fit` applies -- an edge is active exactly where its support weight
    is nonzero, i.e. where ``|signed| < reach + sqrt(dim-1)/2`` -- computed here with no
    neighbour search and no solve, so that the memory door can be shut **before** the
    expensive part rather than after it. It is a second spelling of four lines of
    :func:`geometry`, which is only honest because it is gated against the real thing:
    ``tests/fdtd/test_fd_stencil.py::test_the_predicted_active_rows_are_the_ones_the_fit_reports``
    compares it with ``report['active_edges']`` on a prepared fixture.
    """
    dim = len(components)
    width = reach + np.sqrt(dim - 1) / 2
    flat = (nodes.reshape(-1, dim) - np.asarray(center)) / dx
    count = 0
    for c in range(dim):
        p = flat.copy()
        p[:, c] += .5
        signed = np.linalg.norm(p, axis=-1) - radius / dx
        count += int(np.count_nonzero(np.abs(signed) < width))
    return count


def predicted_bytes(nodes, dx, center, radius, components, reach=REACH):
    """``(predicted peak bytes, predicted unknowns, active rows)`` of the host preparation.

    A prediction and not a measurement, and the refusal that reads it says so. Two measured
    constants multiply: :data:`PAIRS_PER_ACTIVE_ROW` and :data:`BYTES_PER_UNKNOWN`. The row
    count itself is not a prediction -- it is the same screen ``fit`` applies, and it read the
    fit's own ``active_edges`` exactly on every measured fixture.
    """
    rows = active_rows(nodes, dx, center, radius, components, reach)
    unknowns = int(round(rows * PAIRS_PER_ACTIVE_ROW))
    return int(unknowns * BYTES_PER_UNKNOWN), unknowns, rows


def arguments(scene, plan, index, box=None):
    """The exact argument tuple :func:`geometry` is called with for ``structures[index]``.

    One function, two callers: this is what the public preparation hands it, and the research
    driver's ``replace`` builds the same five things the same way. Separating it out is what
    makes the bit-identity hook a comparison of *inputs* rather than of two reimplementations
    (see :func:`operator_digest`).
    """
    from wavebit.fdtd._prepare.constitutive import common_points
    structure = tuple(getattr(scene, 'structures', ()) or ())[index]
    components = (0, 1, 2)
    if box is None:
        old, = plan.constitutive
        box = _stencil_box(old, components, plan)
    nodes = common_points(plan.grid, tuple(reversed(box)), 'E')[..., components]
    shape = structure.shape
    return (nodes, plan.dx, tuple(shape.center), float(shape.radius),
            float(structure.material.eps[0]), components)


def _stencil_box(old, components, plan):
    """The constitutive region grown by the operator's nonlocal reach, three nodes a side.

    Three and not two: a two-cell reach couples an edge to neighbours two cells away, and the
    zero extension the compact region uses needs one further vacuum node beyond the last
    coupled one for the halo check to be able to say the operator ends in the identity. The
    arithmetic is the research driver's, so that both callers grow the same box.

    This grown box is what the prepared term **owns**: its diagonal replaces ``be`` over every
    cell of it and its gather reads the curl over every cell of it. That is why nothing else
    may be in it (:func:`_refuse_scope`, :func:`_refuse_occupancy`).
    """
    box = tuple((lo - 3, hi + 3) if 2 - i in components else (lo, hi)
                for i, (lo, hi) in enumerate(old.box))
    for a, ((lo, hi), size) in enumerate(zip(box, (plan.nz, plan.ny, plan.nx))):
        if not 0 <= lo < hi <= size:
            raise ValueError(
                f'StencilRegionRefused: the extended stencil needs three nodes of vacuum '
                f'beyond the smoothed support on every axis, and on {"zyx"[a]} that reaches '
                f'{lo}..{hi} of a box {size} cells deep. The operator is nonlocal: an edge '
                f'couples to every edge within {REACH:g} cells, and a support that runs out '
                f'of the box has no such neighbourhood to be fitted on. Enlarge '
                f'fd.Domain(size=...) or move the structure inwards.')
    return box


def _materials(scene):
    from wavebit.fdtd._prepare.coefficients import materials_of
    return materials_of(scene)


def _plain(v):
    """Whether ``v`` is a number the host preparation can read with nothing riding on it.

    Admitted: Python ``int`` and ``float`` (not ``bool``, and not a subclass other than NumPy's
    own), and a NumPy real scalar of at most 64 bits -- ``np.float64`` (itself a ``float``
    subclass), ``np.float32``, the NumPy integers. A NumPy scalar is immutable host data that
    no transform can wrap: ``jax.grad``, ``jax.vmap`` and ``jax.jvp`` replace their
    *arguments* with tracers, and a closed-over NumPy scalar stays what it was; every Torch
    wrapper -- a ``requires_grad`` leaf, a forward-mode dual, a ``torch.func`` wrapper -- is a
    ``torch.Tensor``. It carries no tangent and no batch axis, and ``float()`` of it is exact
    at every width up to 64 bits (a wider ``np.longdouble`` would be rounded, so it is not
    admitted). It is admitted because a sweep over ``np.linspace`` hands one over, and there is
    nothing on it to drop.

    Refused, whether or not it is traced: every ``jax.Array``, every ``torch.Tensor`` and every
    NumPy ``ndarray`` -- a 0-d one too. A 0-d array is mutable, so the plan's operator could
    stop describing it without the object changing, and it is the type every array library
    converts through; ``float(x)`` is the spelling that works for a constant.
    """
    if type(v) in (int, float):
        return True
    return isinstance(v, (np.floating, np.integer)) and np.dtype(type(v)).itemsize <= 8


def _refuse_numbers(where, leaves, when):
    """``StencilNumberRefused`` for the first leaf of ``leaves`` that is not :func:`_plain`.

    ``leaves`` is ``((name, value), ...)``. The same sentence at ``fd.plan`` and at
    ``fd.lower``: the recipe that differentiates builds the plan from plain numbers and lowers a
    traced scene against it, so the lowering is the door every transform actually reaches.
    """
    for name, value in leaves:
        if _plain(value):
            continue
        kind = f'{type(value).__module__}.{type(value).__qualname__}'
        raise ValueError(
            f'StencilNumberRefused: {where} asked for discretization={STENCIL!r} and '
            f'{"fd.plan" if when == "plan" else "fd.lower"} was handed a {kind} as its {name}, '
            f'not a plain number. The operator is prepared once on the host in SciPy -- a sparse '
            f'least-squares fit, a symmetric eigenpair and a sparse factorization -- so it is a '
            f'constant of the plan with no derivative: it reads a leaf as a number and drops '
            f'whatever rides on it, a gradient, a forward-mode tangent or a batch axis, and its '
            f'diagonal would then overwrite the permittivity being differentiated, so the '
            f'derivative would come back exactly zero. A structure that asks for the stencil '
            f'therefore takes plain Python numbers (and NumPy scalars) only, and any array or '
            f'tensor -- JAX, Torch or NumPy, traced or not -- is refused, at fd.plan and again '
            f'at fd.lower. For the stencil at this value, write the value as a number '
            f'(e.g. eps=2.25). For a derivative with respect to this structure, use the voxel '
            f'rule in BOTH scenes: Numerics(discretization={DEFAULT!r}) (or leave it out), '
            f'p = fd.plan(scene) from plain numbers, then fd.lower(traced_scene, p, backend) -- '
            f'the recipe tests/fdtd/test_fd_constitutive.py uses. The commit that differentiates '
            f'through a prepared operator is what lifts this.')


def _refuse_structure(structure, index, when='plan'):
    """The doors a single structure has to pass, at plan time and again at lowering.

    These come first, and that ordering is the point of the function existing. A
    cell-snapped shape that asked for the stencil makes no smoothed region at all, so a
    scene-wide question in front of it would answer it with a sentence about the scene when
    what is wrong is the shape; and a magnetic structure makes two constitutive terms, so a
    count in front of the permeability door would tell it it has two regions when it has one.

    Two arms run **at lowering only**: the conductivity-or-poles arm and the anisotropy arm.
    At plan time ``checks.check_smoothed`` refuses both for a smoothed shape on either
    discretisation (its doors 1 and 4) before :func:`prepare` is ever called, so there they
    could not fire and had no possible witness -- the review of ``199463e`` found them
    unreachable from ``fd.plan``, and they are deleted from that path rather than kept as
    dead code. At lowering ``check_smoothed`` never runs, and both are live: the source digest
    reads ``eps[0]`` only, so an anisotropic scene lowered against an isotropic plan would pass
    it. ``tests/fdtd/test_fd_stencil.py::test_loss_and_anisotropy_meet_check_smoothed_at_plan_and_the_stencil_door_at_lowering``
    pins which door answers where.
    """
    from wavebit.fdtd._legacy_geometry import Sphere as _SphereShape
    from wavebit.materials.time_domain import Conductor
    from wavebit.materials.time_domain import Dispersive
    where = f'structures[{index}] ({public_description(structure.shape)})'
    if structure.shape.body() is None:
        raise ValueError(
            f'StencilShapeRefused: {where} asked for discretization={STENCIL!r}, but it is '
            f'cell-snapped -- its edge IS a cell boundary, so there is no interface inside a '
            f'voxel for the operator to constrain the edge and dual-face states across. The '
            f'extended stencil is for a shape with a fractional edge; leave a SnappedSlab or a '
            f'SnappedBox on Numerics(discretization={DEFAULT!r}).')
    if type(structure.shape) is not _SphereShape:
        raise ValueError(
            f'StencilShapeRefused: {where} asked for discretization={STENCIL!r}. The operator '
            f'is assembled from a ball\'s exact edge and dual-face fills, and fd.Sphere is '
            f'the one shape that answers both in closed form today '
            f'(src/wavebit/fdtd/_shapes/_bodies.py) -- fd.Sphere itself, not a subclass, whose '
            f'body may be its own. The tangent plane\'s forms beside them are not wired in as '
            f'a fallback: every convergence order this operator has measured was taken with '
            f'the exact ones, and the plane\'s mean gap to the exact sphere falls only like '
            f'h. Ask with fd.Sphere(radius=..., center=...) itself, or keep this shape -- an '
            f'fd.Box, fd.Cylinder or fd.Prism, or a subclass -- on '
            f'Numerics(discretization={DEFAULT!r}). '
            f'The commit that measures a body without them is what lifts this.')
    material = structure.material
    # The radius and centre are not asked about here: fd.Sphere's own static door
    # (_spec.static_float) refused any traced value at construction, so they are floats.
    _refuse_numbers(where, tuple((f'eps[{a}]', v) for a, v in enumerate(material.eps))
                    + tuple((f'mu[{a}]', v) for a, v in enumerate(material.mu_value)), when)
    if when != 'plan' and isinstance(material, (Conductor, Dispersive)):
        raise ValueError(
            f'StencilScopeRefused: {where} asked for discretization={STENCIL!r} with a '
            f'conductivity or poles. The interface states the operator constrains are the '
            f'lossless constant tangential-E and normal-D ones, and it is fitted from the real '
            f'eps alone, so a loss would be dropped from the region without a word. Neither '
            f'discretisation smooths a conductive or dispersive ball yet -- fd.plan refuses one '
            f'on either (checks.check_smoothed, door 1; C8 owns both) -- so the spelling that '
            f'lowers is the scene this plan was prepared from, with its lossless '
            f'fd.Material(eps=...).')
    if material.is_magnetic:
        unit = all(_plain(v) and v == 1 for v in material.mu_value)
        raise ValueError(
            f'StencilScopeRefused: {where} asked for discretization={STENCIL!r} and names a '
            f'permeability (mu={material.mu!r}'
            f'{", which these doors read as magnetic whatever its value" if unit else ""}). '
            f'The operator is assembled over the E edges and their dual faces alone, so a '
            f'permeability would keep the rule it has today on H while E moved to another one, '
            f'and the two would no longer be the same discretisation; the research driver '
            f'refused any named mu. Leave mu unnamed -- fd.Material(eps=...) -- or use '
            f'Numerics(discretization={DEFAULT!r}), which smooths a magnetic ball today. The '
            f'magnetic half is the commit that measures it.')
    eps = tuple(float(v) for v in material.eps)
    if when != 'plan' and len(set(eps)) != 1:
        raise ValueError(
            f'StencilScopeRefused: {where} asked for discretization={STENCIL!r} with the '
            f'anisotropic permittivity eps={eps}. The operator is fitted from one scalar '
            f'contrast -- the research driver refused anything else -- so an anisotropic '
            f'material would be prepared as isotropic eps={eps[0]!r} and its other two '
            f'components silently dropped. Neither discretisation smooths an anisotropic ball '
            f'yet -- fd.plan refuses one on either (checks.check_smoothed, door 4) -- so the '
            f'spelling that lowers is the scene this plan was prepared from, with its isotropic '
            f'fd.Material(eps=...). The commit that fits a tensor contrast is what lifts this.')


def _refuse_background(scene, where, when='plan'):
    """The scene around the body must be vacuum: the operator was fitted for nothing else."""
    from wavebit.materials.time_domain import Material
    background, _ = _materials(scene)
    if type(background) is Material:
        _refuse_numbers(where, tuple((f'Scene(background=) eps[{a}]', v)
                                     for a, v in enumerate(background.eps))
                        + tuple((f'Scene(background=) mu[{a}]', v)
                                for a, v in enumerate(background.mu_value)), when)
        vacuum = all(v == 1 for v in background.eps)
        if vacuum and background.is_magnetic and all(v == 1 for v in background.mu_value):
            raise ValueError(
                f'StencilBackgroundRefused: {where} asked for discretization={STENCIL!r} in a '
                f'scene whose background names a permeability, mu={background.mu!r}. Its '
                f'values are vacuum\'s, but a named mu is a magnetic medium to these doors '
                f'whatever its value -- as it was to the research driver every number of this '
                f'operator came through -- because it gives the run a permeability path the '
                f'operator was never measured beside. Leave mu unnamed: '
                f'Scene(background=fd.Material(eps=1.)), or leave Scene(background=) unset; or '
                f'use Numerics(discretization={DEFAULT!r}).')
        if vacuum and not background.is_magnetic:
            return
    raise ValueError(
        f'StencilBackgroundRefused: {where} asked for discretization={STENCIL!r} in a '
        f'scene whose background is {background!r}, and the operator is fitted with '
        f'VACUUM outside the body -- its baseline is 1/eps inside and 1 outside and its '
        f'constraint rows carry the contrast eps - 1 -- while its diagonal replaces be '
        f'over the whole grown region. So a non-vacuum background would come back as a '
        f'cube of vacuum wrapped round the sphere, and the run would complete. Fitting '
        f'against another background is new physics with no measurement behind it: the '
        f'commit that carries the background permittivity into the fit\'s baseline and '
        f'contrast, and measures it, is what lifts this. For this scene as written use '
        f'Numerics(discretization={DEFAULT!r}); for the stencil, leave Scene(background=) '
        f'unset.')


def _refuse_scope(scene, plan, structures, asked):
    """What a scene that asks for the stencil must be, once, before anything is assembled.

    Every condition of the research driver's ``validate_scope``
    (``studies/fdtd/subpixel_symmetric_stencil/fdtd_subpixel_symmetric_stencil.py``) is here, each by its own name, because
    every number the operator has is a number of a scene that passed it: a lossless isotropic
    non-magnetic ``fd.Sphere`` of plain numbers (:func:`_refuse_structure`), the **only** structure,
    in a **vacuum** background, in a real ``float64`` run with neither an initial E nor an
    initial H, whose region meets no wall. Its line-by-line map is in
    ``docs/design/fd-symmetric-stencil.md``.
    """
    for index in asked:
        _refuse_structure(structures[index], index)
    index = asked[0]
    structure = structures[index]
    where = f'structures[{index}] ({public_description(structure.shape)})'
    if len(structures) != 1:
        raise ValueError(
            f'StencilStructuresRefused: {where} asked for discretization={STENCIL!r} in a '
            f'scene of {len(structures)} structures. The prepared operator owns its whole '
            f'grown region -- its diagonal replaces be over the smoothed support plus three '
            f'nodes a side -- and was fitted with vacuum everywhere outside the one body, so '
            f'another structure inside that region would be erased from the run, and one '
            f'outside it composes with the operator in a way nothing has measured. The '
            f'research driver every number of this operator came through refused any second '
            f'structure. For this scene as written, use Numerics(discretization={DEFAULT!r}); '
            f'for the stencil, take the other structures out, which removes this '
            f'refusal (the ball alone still has to pass the stencil\'s other doors). The commit '
            f'that measures a prepared operator beside others, and refuses the ones that meet '
            f'its region, is what lifts this.')
    _refuse_background(scene, where)
    if plan.dtype != 'float64':
        raise ValueError(
            f'StencilDtypeRefused: {where} asked for discretization={STENCIL!r} in a '
            f'{plan.dtype} run (a Bloch face makes the fields complex). Every number this '
            f'operator has was measured in a real float64 run -- its research driver refused '
            f'any other -- and a Bloch cell\'s periodic images are ones the fit '
            f'(periodic=None) never saw. For this Bloch cell use '
            f'Numerics(discretization={DEFAULT!r}); '
            f'for the stencil, a real run -- fd.Periodic() in place of fd.Bloch, the zero '
            f'phase -- removes this refusal (the scene still has to pass the stencil\'s other '
            f'doors). The commit that measures it in a Bloch cell is what lifts this.')
    for field, present in (('E', plan.has_E0), ('H', plan.has_H0)):
        if present:
            raise ValueError(
                f'StencilScopeRefused: {where} asked for discretization={STENCIL!r} with '
                f'InitialField.{field}. The term stores D and not E, so an initial E means '
                f'inverting the prepared operator once, which is not implemented, and the '
                f'research driver refused an initial H beside it: no run of this operator '
                f'has started from a field. Excite the scene with a source instead.')
    old, = plan.constitutive
    if old.masks:
        raise ValueError(
            f'StencilScopeRefused: {where} asked for discretization={STENCIL!r} and its region '
            f'meets a PEC or PMC wall. The operator\'s rows reach across the wall, and a wall '
            f'inside the fitted neighbourhood is a boundary condition the fit was not given. '
            f'Move the structure clear of the wall.')
    from wavebit.fdtd._legacy_sources import PlaneWave
    from wavebit.fdtd._legacy_sources import TFSF
    for source in getattr(scene, 'sources', ()) or ():
        if not isinstance(source, (TFSF, PlaneWave)):
            raise ValueError(
                f'StencilSourceRefused: source {source.name!r} is an '
                f'fdtd.{public_name(source)}, and the '
                f'stencil\'s clearance rule reads a TFSF box or a PlaneWave plane. The '
                f'operator\'s rows are nonlocal, so a source that injects inside them writes '
                f'into a neighbourhood the fit assumed was vacuum; the check that says how '
                f'far away is far enough exists for those two and for no other source yet. '
                f'Excite the ball with an fd.TFSF box that holds its region or an '
                f'fd.PlaneWave plane clear of it, or keep this {public_name(source)} with '
                f'Numerics(discretization={DEFAULT!r}).')
    return index, structure


def _refuse_occupancy(scene, plan, box, index):
    """No source and no monitor surface may be in the grown region the operator owns.

    ``box`` is ``(z, y, x)`` cell ranges; along each axis the term's cells ``lo..hi-1`` carry
    E at nodes and half nodes up to ``hi - 1/2``, so the closed node span ``lo..hi`` is taken
    as occupied -- conservatively, by half a node. A TFSF face and a PlaneWave plane inject
    into one node either side of themselves (the band the research clearance rule reserves),
    and a TFSF box has to hold the region **inside** it, as every research scene did. A closed
    ``Flux`` or ``FarField`` surface has to enclose the region or exclude it, and a flux or
    diffraction plane has to miss it; a plane is judged as if it were infinite, which refuses
    a finite plane that would in fact pass beside the region. ``Field`` and ``Record`` read
    fields and assume no medium, so they are not surfaces here.

    This is stricter than the research ``surface_clearance``, which judged the rows the
    operator actually changes; the grown region contains those rows, so a scene this passes
    passes that too, and the public preparation no longer calls it.
    """
    from wavebit.fdtd._legacy_monitors import Diffraction
    from wavebit.fdtd._legacy_monitors import FarField
    from wavebit.fdtd._legacy_monitors import Flux
    from wavebit.fdtd._legacy_sources import PlaneWave
    from wavebit.fdtd._legacy_sources import TFSF
    from wavebit.fdtd._prepare.monitors import _box_nodes
    from wavebit.fdtd._prepare.monitors import _node
    span = tuple(box[2 - a] for a in range(3))           # x, y, z node spans

    def encloses(lo, hi, band):
        return all(lo[a] + band < span[a][0] and hi[a] - band > span[a][1] for a in range(3))

    def excludes(lo, hi):
        return any(hi[a] < span[a][0] or lo[a] > span[a][1] for a in range(3))

    def misses(a, k, band):
        return k + band < span[a][0] or k - band > span[a][1]

    region = ', '.join(f'{"xyz"[a]} {span[a][0]}..{span[a][1]}' for a in range(3))
    owned = (f'the operator for structures[{index}] owns nodes {region} -- its diagonal '
             f'replaces be over every cell there and its gather reads the curl there')
    for monitor in getattr(scene, 'monitors', ()) or ():
        if isinstance(monitor, (Flux, FarField)) and getattr(monitor, 'box', None) is not None:
            lo, hi = _box_nodes(monitor.name, plan.grid, monitor.box)
            if not (encloses(lo, hi, 0) or excludes(lo, hi)):
                raise ValueError(
                    f'StencilSurfaceRefused: {monitor.name}\'s surface at nodes {lo}..{hi} '
                    f'passes through the region {owned}. A closed surface has to enclose it '
                    f'or exclude it, because the surface integral assumes one medium on each '
                    f'of its sides. Move the surface or the structure.')
        elif isinstance(monitor, (Flux, Diffraction)):
            plane = monitor.plane if isinstance(monitor, Flux) else monitor
            a, k = plane.axis, _node(plan.grid, plane.axis, plane.position)
            if not misses(a, k, 0):
                raise ValueError(
                    f'StencilSurfaceRefused: {monitor.name}\'s plane at node {k} on '
                    f'{"xyz"[a]} crosses the region {owned}. Move the plane clear of it.')
    for source in getattr(scene, 'sources', ()) or ():
        if isinstance(source, TFSF):
            lo, hi = _box_nodes(source.name, plan.grid, source.box)
            if not encloses(lo, hi, 1):
                raise ValueError(
                    f'StencilSourceRefused: {source.name}\'s injection band (its faces at '
                    f'nodes {lo}..{hi}, one node either side) meets the region {owned}, or '
                    f'does not hold it inside. The fit assumed a neighbourhood with no source '
                    f'in it. Enlarge the TFSF box so that it clears the region by more than '
                    f'one node on every side.')
        elif isinstance(source, PlaneWave):
            a, k = source.axis, _node(plan.grid, source.axis, source.position)
            if not misses(a, k, 1):
                raise ValueError(
                    f'StencilSourceRefused: {source.name}\'s injection band (node {k} on '
                    f'{"xyz"[a]}, one node either side) meets the region {owned}. Move the '
                    f'source plane clear of it by more than one node.')


def _refuse_layers(plan, box, components, index):
    """The support against the absorbing layers, once, before anything is assembled."""
    layers = dict(plan.absorbing_faces)
    for a, ((lo, hi), size) in enumerate(zip(box, (plan.nz, plan.ny, plan.nx))):
        if 2 - a not in components:
            continue
        axis = 'zyx'[a]
        low, high = layers.get(axis + '-', 0), layers.get(axis + '+', 0)
        if lo < low or hi > size - high:
            raise ValueError(
                f'StencilLayerRefused: structures[{index}]\'s stencil support reaches '
                f'{lo}..{hi} on {axis}, and the absorbing layers hold 0..{low} and '
                f'{size - high}..{size}. The operator is fitted on constant interface states '
                f'in a lossless medium, and a graded absorber inside the fitted '
                f'neighbourhood is a material the fit never saw -- so the rows it returns '
                f'there are not the rows the layer needs. Leave the support clear of the '
                f'layer, or give the domain more room.')


def _refuse_memory(structure, index, args):
    """The predicted host preparation against the declared bound, before the solve starts."""
    nodes, dx, center, radius, eps, components = args
    bound = MEMORY_BOUND_BYTES if structure.memory_bound is None else structure.memory_bound
    predicted, unknowns, rows = predicted_bytes(nodes, dx, center, radius, components)
    if predicted > bound:
        raised = structure.memory_bound is not None
        raise ValueError(
            f'StencilMemoryRefused: preparing structures[{index}] is predicted to need about '
            f'{predicted / 1024 ** 2:.0f} MiB of host memory ({rows} active rows, about '
            f'{unknowns} unknowns at {BYTES_PER_UNKNOWN:g} bytes each), above the '
            f'{"bound this structure declared" if raised else "default bound"} of '
            f'{bound / 1024 ** 2:.0f} MiB. That count is a prediction from two measured '
            f'constants and not a measurement of this scene. The solve runs on the host '
            f'before the run starts, so exceeding the bound is a machine that stops rather '
            f'than a run that slows. Coarsen fd.Resolution(dx=...) or shrink the structure, or '
            f'raise the bound for this structure with fdtd.Numerics('
            f'discretization={STENCIL!r}, memory_bound=<bytes>). The default admits the '
            f'research sphere at its three placements at 40 and 56 cells with a 4.5% margin '
            f'(the largest preparation measured is {MEASURED_UNKNOWNS:,} unknowns, '
            f'docs/design/fd-symmetric-stencil.md). The bytes per unknown is the largest of seven '
            f'measured peaks, and at 56 cells two sub-cell placements of one sphere read 3,276 '
            f'and 2,655 of them, so it is not tight anywhere; past 56 cells nobody has measured '
            f'whether it is low or high.')


def prepare(scene, plan, reuse=None):
    """``plan`` with every structure that asked for the extended stencil prepared into it.

    Called from :func:`wavebit.fdtd._prepare.plan.plan`, once, eagerly, on the host, after the
    constitutive terms are in and before they are validated. A scene where nothing asks pays
    one generator expression and gets the plan it was handed, unchanged and identical --
    which is what keeps all 23 stage-0..3 reference skeletons byte-identical.

    Every door runs before the solve, and every one of them runs whether or not the solve is
    then needed: ``reuse`` is a plan that may already carry this scene's operator
    (``fd.run(scene, plan=...)`` passes the caller's), and when one of its terms was prepared
    from the same arguments (:func:`source_digest`) in the same region, that term is taken as
    it is -- the pattern, the diagonal and the coefficients the caller's plan holds -- and only
    the time-step door is asked again. The rest of what the solve certified (consistency,
    energy, the halo) is a function of those arguments alone.

    The refusals are here and not per cell, and none of them is a fallback: there is no branch
    in this function that answers a scene the operator cannot be prepared for with the other
    rule. That is deliberate. The two discretisations converge at different orders, and a
    scene that silently got the one it did not ask for would read as a physics result.
    """
    from wavebit.fdtd._kernel.masks import Region
    structures = tuple(getattr(scene, 'structures', ()) or ())
    asked = tuple(i for i, s in enumerate(structures)
                  if getattr(s, 'discretisation', DEFAULT) == STENCIL)
    if not asked:
        return plan
    index, structure = _refuse_scope(scene, plan, structures, asked)
    old, = plan.constitutive
    components = (0, 1, 2)
    box = _stencil_box(old, components, plan)
    _refuse_layers(plan, box, components, index)
    _refuse_occupancy(scene, plan, box, index)
    args = arguments(scene, plan, index, box)
    _refuse_memory(structure, index, args)
    region = Region(z=box[0], y=box[1], x=box[2])
    source = source_digest(*args)
    term = _reused(reuse, old, region, source)
    if term is None:
        term = _assemble(old, region, box, args, source, index)
    _refuse_cfl(scene, plan, term.spectral_upper_bound)
    return plan.with_static(terms=tuple(term if t is old else t for t in plan.terms))


def _reused(reuse, old, region, source):
    """The term of ``reuse`` that is this preparation already done, or ``None``."""
    if reuse is None:
        return None
    from wavebit.fdtd._kernel.stencil import StencilTerm
    for term in reuse.terms:
        if (isinstance(term, StencilTerm) and term.source == source and term.name == old.name
                and term.field == old.field and term.region == region
                and term.ratio == old.ratio and term.masks == old.masks):
            return term
    return None


def _assemble(old, region, box, args, source, index):
    """The solve, its certificates, and the term that carries what it produced."""
    from wavebit.fdtd._kernel.stencil import StencilTerm
    components = args[-1]
    matrix, report = geometry(*args)
    # The static width/size screen measured <=7.673e-14; 1e-11 is over
    # 100 times that rounding scale. It is not a scattering-accuracy gate.
    if report['lsmr_stop'] not in (0, 1, 2) or report['residual_max'] > 1e-11:
        raise ValueError(f'StencilConsistencyRefused: {report}')
    if report['minimum_eigenvalue'] <= 0:
        raise ValueError(f'StencilEnergyRefused: {report}')
    report.update(energy_bounds(matrix))
    diagonal, idx, value, output = padded(matrix)
    _refuse_halo(matrix, diagonal, box, components, region)
    diagonal = diagonal.reshape((len(components),) + region.shape)
    return StencilTerm(old.name, old.field, region, old.ratio, old.masks,
                       components=components, index=idx, output=output, diagonal=diagonal,
                       value=value, structure=index, source=source,
                       spectral_upper_bound=float(report['spectral_upper_bound']))


def _refuse_halo(matrix, diagonal, box, components, region):
    """The operator must end in the identity on the outermost node layer of its own box."""
    shape = region.shape
    grid = np.arange(len(components) * int(np.prod(shape))).reshape((len(components),) + shape)
    for a in range(3):
        if 2 - a not in components:
            continue
        for end in (0, -1):
            halo = np.take(grid, end, axis=a + 1).ravel()
            expected = coo_matrix((np.ones(len(halo)), (np.arange(len(halo)), halo)),
                                  shape=(len(halo), len(diagonal))).tocsr()
            if (matrix[halo] - expected).nnz:
                raise ValueError(
                    f'StencilHaloRefused: the prepared operator is not the identity on the '
                    f'{"zyx"[a]} {"low" if end == 0 else "high"} node layer of its own region '
                    f'{box}. The kernel extends the region by zero outside it, which is the '
                    f'exact adjoint only while the operator has already returned to vacuum '
                    f'there. Give the structure more room.')


def _refuse_cfl(scene, plan, spectral_upper_bound):
    """The prepared operator's own time-step bound, from its spectral upper bound.

    Every medium is non-magnetic here (:func:`_refuse_scope`), so the H side contributes one.
    """
    import math

    from wavebit.fdtd._prepare.checks import cfl_axis_sum
    rate = plan.dt ** 2 * cfl_axis_sum(scene, plan.dx) * spectral_upper_bound
    if not math.isfinite(rate) or not 0 < rate < 1:
        raise ValueError(
            f'StencilCFLRefused: the prepared operator\'s time step bound reads '
            f'dt^2 * rate = {rate:.17g}, which is not in (0, 1). The extended stencil\'s '
            f'spectral upper bound is {spectral_upper_bound:.17g}, above the '
            f'diagonal rule\'s, so a courant number the default accepts may not survive it. '
            f'The rate goes as the square of the courant number, so fd.Run(courant=...) below '
            f'{_courant_limit(scene.run.courant, rate)} passes this door. It is read from the '
            f'prepared operator itself and so comes after the preparation (minutes at 40 and '
            f'56 cells, docs/design/fd-symmetric-stencil.md), not before it.')


def _courant_limit(courant, rate):
    """The courant number, rounded down to four places, at which ``rate`` would reach one."""
    import math
    if not math.isfinite(rate) or rate <= 0:
        return 'the one that was asked for'
    return f'{math.floor(courant / math.sqrt(rate) * 1e4) / 1e4:.4f}'


def refuse_discretisation(scene, plan):
    """The scene's word and the plan's operator must agree -- in both directions -- at lowering.

    ``fd.lower(scene, plan, backend)`` takes the two separately, and the caller chooses the
    plan: that is the repository's gradient recipe. A plan carries an operator for a structure
    exactly when the scene it was built from asked for ``discretisation='stencil'`` there, so
    the two can disagree two ways, and each is a silent change of discretisation:

    * the scene asks and the plan carries none -- the plan came from the scene's voxel twin,
      or from a scene without the structure. Measured on ``199463e`` (review 1): ``fd.lower``
      accepted it and painted the ball with the voxel rule, and on the traced path returned
      the voxel gradient, -12.667 on ``jax.grad``, ``jax.jit(jax.grad)`` and Torch alike --
      reached through the very recipe the number refusal recommends;
    * the plan carries one and the scene does not ask, or no longer holds the structure --
      its diagonal would replace the voxel rule over its region.

    Called by :func:`wavebit.fdtd._prepare.params.lower` before anything is lowered, whether or not
    the plan has constitutive terms (a plan built from a scene with no structure has none). A
    scene where nothing asks and a plan that carries nothing pass after one loop over each,
    and nothing is read from any array.
    """
    from wavebit.fdtd._kernel.stencil import StencilTerm
    structures = tuple(getattr(scene, 'structures', ()) or ())
    held = {t.structure: t for t in plan.terms if isinstance(t, StencilTerm)}
    for index, structure in enumerate(structures):
        if getattr(structure, 'discretisation', DEFAULT) == STENCIL and index not in held:
            raise ValueError(
                f'StencilMismatchRefused: structures[{index}] '
                f'({public_description(structure.shape)}) asks for discretization={STENCIL!r}, and the '
                f'plan fd.lower was handed carries no operator prepared for it -- it was built '
                f'from a scene that asked for {DEFAULT!r} there, or that did not hold this '
                f'structure. Lowered anyway, the ball would get the voxel rule while the scene '
                f'says stencil, and a derivative through it would be the voxel rule\'s: '
                f'jax.grad and a finite difference of fd.run would compute two different '
                f'functions. Build the plan from this scene with fd.plan(scene), from plain '
                f'numbers; or, to differentiate, set Numerics(discretization={DEFAULT!r}) for '
                f'the scene you lower as well as for the one you plan.')
    for index, term in held.items():
        asks = (getattr(structures[index], 'discretisation', DEFAULT)
                if index < len(structures) else None)
        if asks != STENCIL:
            said = (f'has no structures[{index}]' if asks is None else
                    f'asks for discretization={asks!r} there')
            raise ValueError(
                f'StencilMismatchRefused: the plan carries {term.name}, an operator prepared '
                f'for structures[{index}] with discretization={STENCIL!r}, and the scene '
                f'fd.lower was handed {said}. Lowered anyway, the operator\'s diagonal would '
                f'replace the rule that scene asks for over the whole grown region. Build the '
                f'plan from this scene with fd.plan(scene), or lower the scene the plan was '
                f'prepared from.')

    if held:
        # arguments() reads the held plan's grid. Its source digest therefore
        # cannot notice a different grid requested by the scene at lowering.
        # Compare snapped grids before params.lower allocates any coefficients.
        from wavebit.fdtd._kernel.grid import Grid
        requested = Grid.from_domain(scene.domain.size, scene.domain.center, scene.resolution.dx)
        if requested != plan.grid:
            raise ValueError(
                'StencilMismatchRefused: the scene grid (spacing, snapped cell counts or '
                'low corner) differs from the grid the stencil operator was prepared for. '
                'Build the plan from this scene with fd.plan(scene).')


def _refuse_lowering(scene, plan, term):
    """The doors again, at lowering, where every traced path passes.

    ``fd.plan`` refuses an array or a tensor in a leaf, but the repository's gradient recipe
    builds the plan from plain numbers and lowers a *traced* scene against it -- and
    ``jax.grad``, ``jax.jvp``, ``jax.vmap``, Torch autograd, ``torch.func`` and Torch
    forward-mode duals all arrive here. Lowered, the prepared diagonal would overwrite the
    traced permittivity and the derivative would be exactly zero, a silently wrong answer. So
    the structure's doors run again, with the background's and the one-structure rule beside
    them, and then the operator's own source digest is compared: a plain scene whose sphere or
    permittivity is not the one the operator was fitted for is refused rather than run with
    the other one's operator. That the scene asks for the operator the plan carries at all is
    :func:`refuse_discretisation`'s, which runs first and in both directions.
    """
    structures = tuple(getattr(scene, 'structures', ()) or ())
    if len(structures) != 1 or term.structure != 0:
        raise ValueError(
            f'StencilMismatchRefused: the plan carries {term.name}, prepared for '
            f'structures[{term.structure}] of a scene of one structure, and fd.lower was '
            f'handed a scene of {len(structures)}. A prepared operator owns its grown region, '
            f'so it is lowered only against the scene it was prepared for. Lower that scene, '
            f'or build the plan from this one with fd.plan(scene).')
    structure = structures[0]
    where = f'structures[0] ({public_description(structure.shape)})'
    _refuse_structure(structure, 0, when='lower')
    _refuse_background(scene, where, when='lower')
    box = (term.region.z, term.region.y, term.region.x)
    if source_digest(*arguments(scene, plan, 0, box)) != term.source:
        raise ValueError(
            f'StencilMismatchRefused: {where} is not the structure the plan\'s operator '
            f'{term.name} was prepared for -- its centre, radius or permittivity differs, and '
            f'the operator is a function of exactly those. Build the plan from this scene '
            f'with fd.plan(scene).')


def lower(scene, plan, rt):
    """``(tables, patches)``: what the kernel reads, and what ``be`` is patched with.

    ``tables`` goes into ``params['stencil']`` and each patch is ``(box, diagonal * ratio)``
    for :func:`wavebit.fdtd._prepare.coefficients._paste` to write into ``be`` **out of place**.
    Nothing is solved here and nothing branches on a value -- which is what makes NumPy, JAX
    eager, ``jax.jit`` and Torch the same run. What *is* here is the doors
    (:func:`_refuse_lowering`), because lowering is the one place every traced path passes.
    """
    from wavebit.fdtd._kernel.stencil import StencilTerm
    tables, patches = {}, []
    for term in plan.terms:
        if not isinstance(term, StencilTerm):
            continue
        _refuse_lowering(scene, plan, term)
        tables[term.name] = {'index': rt.array(term.index),
                             'value': rt.array(term.value, 'float64'),
                             'output': rt.array(term.output)}
        patches.append(((term.region.x, term.region.y, term.region.z),
                        rt.array(term.diagonal, 'float64') * term.ratio))
    return tables, patches
