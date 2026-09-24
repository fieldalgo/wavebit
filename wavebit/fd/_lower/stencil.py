"""The symmetric extended-stencil material operator, and the preparation a scene asks for.

This is the operator ``docs/fd-symmetric-stencil.md`` derives and measures. A structure asks
for it by name -- ``fd.Structure(shape, material, discretisation='stencil')`` -- and
:func:`prepare` assembles it **once, eagerly, on the host**, at plan time, for that structure
alone. A structure that does not ask, and every scene that holds none that do, is untouched:
the default is exactly the rule it was, to the bit, and ``tests/test_fd_stencil.py`` proves
that against a ``git archive`` of the commit before this one.

**Preparation is plan time, never lowering.** ``fit`` is SciPy on the host in float64 -- a
sparse least-squares solve, a symmetric eigenpair, a sparse factorization -- and none of the
three has a JAX or a Torch spelling. Assembling in ``plan()`` is what keeps NumPy, JAX eager,
``jax.jit`` and Torch identical: the solve is upstream of every trace, the way the CPML
stretch profile already is (``wavebit/fd/_lower/cpml.py``). :func:`lower` only reads the
prepared arrays, moves them to the backend and patches the coefficients out of place.

**What it does.** At each target edge the interface normal is frozen and the exact constant
tangential-E / normal-D states are integrated over every nearby edge and dual face. ``fit``
solves the row constraints ``sum_j K_ij Pi_j(n_i) = Gamma_i(n_i)`` for a symmetric ``K``,
minimising a weighted Frobenius correction to the symmetric triplet baseline;
``triplet_baseline`` builds that baseline, ``geometry`` assembles the two- and
three-dimensional cases from a node lattice, ``energy_bounds`` certifies the assembled
operator, and ``padded`` compresses it to its active rows. Positive energy is not assumed
anywhere: the whole assembled operator is inspected before any field is advanced.

**The seam, half closed.** ``geometry`` took its exact edge and face fractions from
``studies/``; it takes them from ``wavebit/fd/_shapes``'s :class:`~wavebit.fd._shapes.Sphere`
now, which is where those closed forms live, and the study helpers that used to own them are
thin wrappers over the same methods. So ``fit``, ``energy_bounds``, ``padded`` and
``geometry(..., baseline_kind=None)`` reach no study at all and run from an installed wheel.

What is left is ``triplet_baseline``, which ``geometry`` calls for its default baseline: the
triplet orientation maps and their ``ORIENTATIONS`` table are research material with no public
equivalent, and moving them is not this commit. It imports the study where it uses it, through
:func:`_study_module`, which **refuses by name** where ``studies/`` is not importable rather
than failing deep inside on a module the caller never named, and :func:`prepare` names the
same seam for a scene (``StencilSeamRefused``) rather than picking another rule behind it.
Importing this module costs nothing and pulls no study -- ``tests/test_fd_symmetric_stencil.py``
pins that, and pins that the public route runs with ``studies`` blocked.

**Host code, not backend code.** This assembles on SciPy, on the host, in float64. It is not
the traced, differentiable geometry path and cannot silently stand in for one: an input that
arrives as a JAX tracer or a Torch tensor is refused **by name** at :func:`prepare`'s door
(``StencilTracedRefused``), not discretised some other way.
"""
import hashlib
import importlib

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lsmr, eigsh

from .._shapes import Sphere

__all__ = ['DEFAULT', 'MEMORY_BOUND_BYTES', 'REACH', 'STENCIL', 'active_rows', 'arguments',
           'energy_bounds', 'fit', 'geometry', 'lower', 'operator_digest', 'padded',
           'predicted_bytes', 'prepare', 'prepared_digest', 'surface_clearance',
           'triplet_baseline']

#: The names :func:`fit` reads a complex value out of, and refuses. Each one reaches the
#: preallocated ``float64`` buffers below through ``gamma``, ``pis`` or ``pi``.
_REAL_INPUTS = ('eps', 'ell', 'area', 'normals', 'baseline')


def _study_module(name):
    """A study module by name, refused by name where ``studies/`` is not importable.

    :func:`triplet_baseline` is the one function here that still needs one: the triplet
    orientation maps and their ``ORIENTATIONS`` table are research material with no public
    equivalent, unlike the exact edge and face fractions, which :func:`geometry` now takes from
    ``wavebit/fd/_shapes``. An installed wheel carries no ``studies/``, so the import fails
    there -- and without this it fails several frames down, naming a module the caller never
    asked for and saying nothing about the route that does work. Deferred to the call and not
    written at the top of the file, so that importing this module still pulls no study.
    """
    try:
        return importlib.import_module(f'studies.{name}')
    except ImportError as error:
        raise ImportError(
            f'StencilStudyRefused: triplet_baseline() builds the research triplet baseline out '
            f'of studies.{name}, which lives in the repository checkout and not in an installed '
            f'wavebit. Run it from a checkout, or ask geometry(..., baseline_kind=None) for the '
            f'diagonal baseline, which reaches no study.') from error


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
    dim=len(components)
    study=_study_module('fdtd_subpixel_triplet_cylinder' if dim==2
                        else 'fdtd_subpixel_triplet_sphere')
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
    # The public closed forms, not the study's wrappers over them: a ball's chord along a
    # component's edge and its cross-section in the component's dual face. Everything here is
    # in cell units, so the body is a sphere of radius/dx at the origin and the cell is one
    # wide. In two dimensions the study's circle is this sphere's central plane -- the lift to
    # (0, y, z) makes the transverse sum 0*0 + t*t, which is the t**2 the circle form wrote --
    # so the same edge fraction answers both the length and the transverse area there.
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
    import wavebit.fd as fd
    from wavebit.fd._lower.monitors import _box_nodes,_node
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
#: names" -- a whole-cell replace for a cell-snapped :class:`~wavebit.fd.Slab` or
#: :class:`~wavebit.fd.Box`, the diagonal Farjadpour/Kottke mix plus the constitutive cross
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
#: **41.72, 42.98, 43.43, 44.33, 44.76**, rising with the shell's size and flattening; the
#: recorded 56-cell research sphere reads 1,826,928 / 40,224 = **45.42**
#: (``docs/fd-symmetric-stencil.md``). Rounded **up** past all six, because it multiplies a
#: memory bound and a bound that is occasionally low is not a bound.
PAIRS_PER_ACTIVE_ROW = 46.0

#: Bytes of peak host working set per unknown that the preparation costs, over the process's
#: working set just before ``geometry`` is called. Measured 2026-09-24 in five separate
#: processes at OMP/OPENBLAS/MKL_NUM_THREADS=2, one preparation each, through
#: ``GetProcessMemoryInfo``'s ``PeakWorkingSetSize``:
#:
#:     unknowns    76,353   117,864   135,768   249,462   350,730
#:     bytes/unknown  749.2     728.2     710.8     658.3     620.9
#:
#: It **falls** with size -- the fixed cost of SciPy, NumPy and the imported package amortises
#: -- so the largest measured value is taken as an upper bound over the measured range and
#: nothing is extrapolated past it. :data:`MEMORY_BOUND_BYTES` is deliberately set inside that
#: range so that this number is never quoted at a size it was not measured at.
BYTES_PER_UNKNOWN = 750.0

#: The declared preparation bound: what one host preparation may be predicted to cost before
#: it is refused rather than started. A laptop that panics rather than slows is on record in
#: this repository (memory ``laptop-ram-panics-not-just-slows``), which is why the door is
#: here at all and why it is shut before the solve rather than after it.
#:
#: 256 MiB is 357,913 unknowns at :data:`BYTES_PER_UNKNOWN`, which is just past the largest
#: preparation the constant was measured on (350,730). That is the whole reason for the value:
#: a larger bound would be a prediction made with a constant nobody has measured there, and
#: the number it printed would be the confident part of a guess. Raising it is a deliberate
#: act and the refusal says so.
MEMORY_BOUND_BYTES = 256 * 1024 ** 2


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


def active_rows(nodes, dx, center, radius, components, reach=REACH):
    """The number of target edges the operator will carry a correction on.

    The same screen :func:`fit` applies -- an edge is active exactly where its support weight
    is nonzero, i.e. where ``|signed| < reach + sqrt(dim-1)/2`` -- computed here with no
    neighbour search and no solve, so that the memory door can be shut **before** the
    expensive part rather than after it. It is a second spelling of four lines of
    :func:`geometry`, which is only honest because it is gated against the real thing:
    ``tests/test_fd_stencil.py::test_the_predicted_active_rows_are_the_ones_the_fit_reports``
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
    fit's own ``active_edges`` exactly on all five measured fixtures.
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
    from .constitutive import common_points
    structure = tuple(getattr(scene, 'structures', ()) or ())[index]
    components = (0, 1, 2)
    old, = plan.constitutive
    box = _stencil_box(old, components, plan) if box is None else box
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
    from .coefficients import materials_of
    return materials_of(scene)


def _refuse_structure(structure, index):
    """The doors a single structure has to pass, whatever else the scene holds.

    These come first, and that ordering is the point of the function existing. A
    cell-snapped shape that asked for the stencil makes no smoothed region at all, so a
    scene-wide "one region please" count in front of it would answer it with a sentence
    about regions when what is wrong is the shape.
    """
    from ..geometry import Sphere as _SphereShape
    from ..materials import Conductor, Dispersive, concrete
    where = f'structures[{index}] ({structure.shape.describe()})'
    if structure.shape.body() is None:
        raise ValueError(
            f'StencilShapeRefused: {where} asked for discretisation={STENCIL!r}, but it is '
            f'cell-snapped -- its edge IS a cell boundary, so there is no interface inside a '
            f'voxel for the operator to constrain the edge and dual-face states across. The '
            f'extended stencil is for a shape with a fractional edge; leave a Slab or a Box '
            f'on discretisation={DEFAULT!r}.')
    if not isinstance(structure.shape, _SphereShape):
        raise ValueError(
            f'StencilShapeRefused: {where} asked for discretisation={STENCIL!r}. The operator '
            f'is assembled from the body\'s exact edge and dual-face fills, and fd.Sphere is '
            f'the one shape that answers both in closed form today '
            f'(wavebit/fd/_shapes/_bodies.py). The tangent plane\'s forms beside them are not '
            f'wired in as a fallback: every convergence order this operator has measured was '
            f'taken with the exact ones, and the plane\'s mean gap to the exact sphere falls '
            f'only like h. The commit that measures a body without them is what lifts this.')
    leaves = (structure.shape.radius,) + tuple(structure.shape.center) \
        + tuple(structure.material.eps) + tuple(structure.material.mu_value)
    if any(concrete(v) is None for v in leaves):
        raise ValueError(
            f'StencilTracedRefused: {where} asked for discretisation={STENCIL!r} with a traced '
            f'radius, centre or material value. The operator is solved once on the host in '
            f'SciPy -- a sparse least-squares fit, a symmetric eigenpair and a sparse '
            f'factorization -- so it is static data of the plan and cannot be a function of a '
            f'leaf. It is refused rather than quietly discretised the other way, which would '
            f'make jax.grad and a finite difference compute two different functions. A '
            f'gradient with respect to a smoothed structure runs on discretisation='
            f'{DEFAULT!r}; the commit that differentiates through a prepared operator is what '
            f'lifts this.')
    if isinstance(structure.material, (Conductor, Dispersive)):
        raise ValueError(
            f'StencilScopeRefused: {where} asked for discretisation={STENCIL!r} with a '
            f'conductivity or poles. The interface states the operator constrains are the '
            f'lossless constant tangential-E and normal-D ones; sigma and the pole '
            f'coefficients are not smoothed at all yet (checks.check_smoothed, door 1). C8 '
            f'owns both.')


def _refuse_scope(scene, plan, structures, asked):
    """What a scene that asks for the stencil must be, once, before anything is assembled."""
    for index in asked:
        _refuse_structure(structures[index], index)
    if len(asked) > 1 or len(plan.constitutive) != 1:
        raise ValueError(
            f'StencilScopeRefused: {len(asked)} structure(s) asked for '
            f'discretisation={STENCIL!r} and the plan carries {len(plan.constitutive)} '
            f'smoothed region(s). The prepared operator is one symmetric matrix over one '
            f'region, and two of them sharing an interpolation cell would each fit a '
            f'neighbourhood the other has already changed. One smoothed structure per scene '
            f'asks for the stencil today; the commit that measures two regions is what '
            f'lifts this.')
    index, = asked
    structure = structures[index]
    where = f'structures[{index}] ({structure.shape.describe()})'
    background, _ = _materials(scene)
    if background.is_magnetic or any(s.material.is_magnetic for s in structures):
        raise ValueError(
            f'StencilScopeRefused: {where} asked for discretisation={STENCIL!r} in a magnetic '
            f'scene. The operator is assembled over the E edges and their dual faces alone, '
            f'so a permeability would keep the rule it has today on H while E moved to '
            f'another one, and the two would no longer be the same discretisation. The '
            f'magnetic half is the commit that measures it.')
    old, = plan.constitutive
    if old.masks:
        raise ValueError(
            f'StencilScopeRefused: {where} asked for discretisation={STENCIL!r} and its region '
            f'meets a PEC or PMC wall. The operator\'s rows reach across the wall, and a wall '
            f'inside the fitted neighbourhood is a boundary condition the fit was not given. '
            f'Move the structure clear of the wall.')
    if plan.has_E0:
        raise ValueError(
            f'StencilScopeRefused: {where} asked for discretisation={STENCIL!r} with '
            f'InitialField.E. The term stores D and not E, so starting from a field means '
            f'inverting the prepared operator once, which is not implemented. Excite the '
            f'scene with a source instead; a diagonal division would initialise the wrong '
            f'energy (the sentence constitutive.validate already says).')
    for source in getattr(scene, 'sources', ()) or ():
        from ..sources import PlaneWave, TFSF
        if not isinstance(source, (TFSF, PlaneWave)):
            raise ValueError(
                f'StencilSourceRefused: {source.name} is an {type(source).__name__} and the '
                f'stencil\'s clearance rule reads a TFSF box or a PlaneWave plane. The '
                f'operator\'s rows are nonlocal, so a source that injects inside them writes '
                f'into a neighbourhood the fit assumed was vacuum; the check that says how '
                f'far away is far enough exists for those two and for no other source yet.')
    return index, structure


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


#: The off-diagonal coefficients of every operator prepared in this process, by the prepared
#: operator's own digest. They are the one part of it that is *not* a plan field: they are
#: floats the kernel multiplies by, so they belong in ``params`` (:func:`lower`), while the
#: plan carries the pattern, the diagonal and this key.
#:
#: The key is the digest and **not** the term's name. Two scenes in one process name their
#: terms the same thing -- ``constitutive[E0]`` is a position, not an identity -- so a name
#: would let the second preparation overwrite the first, and lowering the first plan
#: afterwards would silently run the second's coefficients under the first's pattern. A digest
#: over all four arrays cannot do that: two entries collide only when the arrays are equal, in
#: which case the value is the same one. A plan built in this process finds its entry here;
#: one that did not is refused by :func:`lower` rather than lowered without the coefficients
#: that make it the operator it says it is.
_prepared = {}


def prepare(scene, plan):
    """``plan`` with every structure that asked for the extended stencil prepared into it.

    Called from :func:`wavebit.fd._lower.plan.plan`, once, eagerly, on the host, after the
    constitutive terms are in and before they are validated. A scene where nothing asks pays
    one generator expression and gets the plan it was handed, unchanged and identical --
    which is what keeps all 23 stage-0..3 reference skeletons byte-identical.

    The refusals are here and not per cell, and none of them is a fallback: there is no branch
    in this function that answers a scene the operator cannot be prepared for with the other
    rule. That is deliberate. The two discretisations converge at different orders, and a
    scene that silently got the one it did not ask for would read as a physics result.
    """
    from .._kernel.masks import Region
    from .._kernel.stencil import StencilTerm
    structures = tuple(getattr(scene, 'structures', ()) or ())
    asked = tuple(i for i, s in enumerate(structures)
                  if getattr(s, 'discretisation', DEFAULT) == STENCIL)
    if not asked:
        return plan
    index, _ = _refuse_scope(scene, plan, structures, asked)
    old, = plan.constitutive
    components = (0, 1, 2)
    box = _stencil_box(old, components, plan)
    _refuse_layers(plan, box, components, index)
    args = arguments(scene, plan, index, box)
    nodes, dx, center, radius, eps, components = args
    predicted, unknowns, rows = predicted_bytes(nodes, dx, center, radius, components)
    if predicted > MEMORY_BOUND_BYTES:
        raise ValueError(
            f'StencilMemoryRefused: preparing structures[{index}] is predicted to need about '
            f'{predicted / 1024 ** 2:.0f} MiB of host memory ({rows} active rows, about '
            f'{unknowns} unknowns at {BYTES_PER_UNKNOWN:g} bytes each), above the declared '
            f'bound of {MEMORY_BOUND_BYTES / 1024 ** 2:.0f} MiB. That count is a prediction '
            f'from two measured constants and not a measurement of this scene. The solve runs '
            f'on the host before the run starts, so exceeding the bound is a machine that '
            f'stops rather than a run that slows. Coarsen fd.Resolution(dx=...) or shrink the '
            f'structure. Raising wavebit.fd._lower.stencil.MEMORY_BOUND_BYTES is a deliberate '
            f'act: the bound sits at the top of the range BYTES_PER_UNKNOWN was measured on '
            f'(350,730 unknowns), so past it the prediction is an extrapolation of a constant '
            f'that has not been measured there -- measure it at that size first.')
    try:
        matrix, report = geometry(*args)
    except ImportError as err:
        raise ValueError(
            f'StencilSeamRefused: the operator\'s triplet baseline still lives under studies/ '
            f'and this checkout cannot import it ({err}). That seam is named in this '
            f'module\'s docstring and closing it is its own commit; '
            f'until then discretisation={STENCIL!r} runs from a checkout and not from an '
            f'installed wheel. It is refused rather than prepared some other way.') from err
    report['surface_clearance'] = surface_clearance(scene, plan, matrix, box, components)
    # The static width/size screen measured <=7.673e-14; 1e-11 is over
    # 100 times that rounding scale. It is not a scattering-accuracy gate.
    if report['lsmr_stop'] not in (0, 1, 2) or report['residual_max'] > 1e-11:
        raise ValueError(f'StencilConsistencyRefused: {report}')
    if report['minimum_eigenvalue'] <= 0:
        raise ValueError(f'StencilEnergyRefused: {report}')
    report.update(energy_bounds(matrix))
    diagonal, idx, value, output = padded(matrix)
    region = Region(z=box[0], y=box[1], x=box[2])
    _refuse_halo(matrix, diagonal, box, components, region)
    _refuse_cfl(scene, plan, report, structures)
    diagonal = diagonal.reshape((len(components),) + region.shape)
    key = prepared_digest(idx, value, output, diagonal)
    term = StencilTerm(old.name, old.field, region, old.ratio, old.masks,
                       components=components, index=idx, output=output, diagonal=diagonal,
                       key=key)
    _prepared[key] = value
    return plan.with_static(terms=tuple(term if t is old else t for t in plan.terms))


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


def _refuse_cfl(scene, plan, report, structures):
    """The prepared operator's own time-step bound, from its spectral upper bound."""
    import math

    from .checks import cfl_axis_sum
    background, _ = _materials(scene)
    inverse_mu = max(1.0 / float(v)
                     for m in [background] + [s.material for s in structures]
                     for v in m.mu_value)
    rate = plan.dt ** 2 * cfl_axis_sum(scene, plan.dx) \
        * report['spectral_upper_bound'] * inverse_mu
    report['dt_squared_rate'] = rate
    if not math.isfinite(rate) or not 0 < rate < 1:
        raise ValueError(
            f'StencilCFLRefused: the prepared operator\'s time step bound reads '
            f'dt^2 * rate = {rate:.17g}, which is not in (0, 1). The extended stencil\'s '
            f'spectral upper bound is {report["spectral_upper_bound"]:.17g}, above the '
            f'diagonal rule\'s, so a courant number the default accepts may not survive it. '
            f'Lower fd.Run(courant=...).')


def lower(plan, rt):
    """``(tables, patches)``: what the kernel reads, and what ``be`` is patched with.

    ``tables`` goes into ``params['stencil']`` and each patch is ``(box, diagonal * ratio)``
    for :func:`wavebit.fd._lower.coefficients._paste` to write into ``be`` **out of place**.
    Nothing is solved here and nothing branches on a value -- which is what makes NumPy, JAX
    eager, ``jax.jit`` and Torch the same run.
    """
    from .._kernel.stencil import StencilTerm
    tables, patches = {}, []
    for term in plan.terms:
        if not isinstance(term, StencilTerm):
            continue
        value = _prepared.get(term.key)
        if value is None:
            raise ValueError(
                f'StencilUnpreparedRefused: the plan carries {term.name} with prepared '
                f'operator {term.key[:12]}, and this process never prepared it. The '
                f'operator\'s off-diagonal coefficients are assembled on the host by '
                f'fd.plan(scene) and held beside the plan, not inside it; a plan rebuilt or '
                f'unpickled elsewhere has the pattern, the diagonal and this key but not the '
                f'coefficients. Call fd.plan(scene) in this process.')
        tables[term.name] = {'index': rt.array(term.index),
                             'value': rt.array(value, 'float64'),
                             'output': rt.array(term.output)}
        patches.append(((term.region.x, term.region.y, term.region.z),
                        rt.array(term.diagonal, 'float64') * term.ratio))
    return tables, patches
