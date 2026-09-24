"""Research symmetric extended-stencil material map, not a public API.

At each target edge, freeze the interface normal and integrate the exact
constant tangential-E / normal-D states on every nearby edge and dual face.
Solve K Pi_i = Gamma_i row constraints with K symmetric, minimizing a
weighted Frobenius correction to the symmetric triplet map. Positive energy is NOT
assumed; inspect the entire assembled operator before advancing any field.
"""
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lsmr, eigsh

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from studies.fdtd_subpixel_triplet_cylinder import circle_fraction
from studies.fdtd_subpixel_triplet_sphere import edge_fraction,face_fraction
from wavebit.fd._kernel.constitutive import ConstitutiveTerm


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
    rhs=[];rr=[];cc=[];vv=[]
    for i in ids:
        c=axes[i];n=normals[i];nn=np.outer(n,n);pt=np.eye(dim)-nn
        gamma=np.eye(dim)[c]+(1/eps-1)*ell[i]*nn[c]
        a,b=baseline.indptr[i:i+2];js=baseline.indices[a:b]
        pis=np.eye(dim)[axes[js]]+(eps-1)*area[js,None]*pt[axes[js]]
        rhs.extend(gamma-baseline.data[a:b]@pis)
    for k,(i,j) in enumerate(pairs):
        for target,source in ((i,j),) if i==j else ((i,j),(j,i)):
            c=axes[source];n=normals[target];pt=np.eye(dim)-np.outer(n,n)
            pi=np.eye(dim)[c]+(eps-1)*area[source]*pt[c]
            for a in range(dim):
                rr.append(dim*rowids[target]+a);cc.append(k)
                vv.append(pi[a]*weight[i]*weight[j]/(1. if i==j else np.sqrt(2.)))
    constraints=coo_matrix((vv,(rr,cc)),shape=(dim*len(ids),len(pairs))).tocsr()
    rhs=np.array(rhs)
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
    rows=[];cols=[];values=[]
    for k,(i,j) in enumerate(pairs):
        v=solved[0][k]*weight[i]*weight[j]/(1. if i==j else np.sqrt(2.))
        rows.append(i);cols.append(j);values.append(v)
        if i!=j: rows.append(j);cols.append(i);values.append(v)
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
    rows=[];cols=[];vals=[]
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
                mask=valid[i]&valid[j]
                rows.extend((i*size+index[i][mask]).ravel());cols.extend((j*size+index[j][mask]).ravel())
                vals.extend(((blocks[k,...,i,j]-(1 if i==j else 0))/len(study.ORIENTATIONS))[mask].ravel())
    return (diags(np.ones(dim*size))+coo_matrix((vals,(rows,cols)),shape=(dim*size,dim*size))).tocsr()


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
    diagonal=matrix.diagonal();cross=matrix-diags(diagonal);cross.eliminate_zeros()
    width=int(np.diff(cross.indptr).max(initial=0))
    index=np.zeros((matrix.shape[0],width),dtype=np.int32);value=np.zeros_like(index,dtype=float)
    for i in range(matrix.shape[0]):
        a,b=cross.indptr[i:i+2];index[i,:b-a]=cross.indices[a:b];value[i,:b-a]=cross.data[a:b]
    return diagonal,index,value


@dataclasses.dataclass(frozen=True)
class StencilTerm(ConstitutiveTerm):
    components: tuple=()
    def prepare(self,curl,mem,params,rt):
        local=curl[self.view]
        compact=rt.xp.stack(tuple(local[c] for c in self.components)).reshape(-1)
        value=rt.xp.sum(compact[params['stencil_index']]*params['stencil_value'],axis=1)
        value=value.reshape((len(self.components),)+self.region.shape)
        delta=rt.xp.stack(tuple(value[self.components.index(c)] if c in self.components
            else rt.xp.zeros_like(local[c]) for c in range(3)))
        return mem+self.ratio*local,self.ratio*delta


def validate_scope(scene,p,center,radius,eps,components):
    """Refuse unsupported research inputs on the host, before assembly."""
    from numbers import Real
    import wavebit.fd as fd
    from studies.fdtd_subpixel_coupled_cylinder import CylinderSlice
    from studies.fdtd_subpixel_mie_cylinder import RADIUS
    def scalar(value):
        # Never detach a Torch/JAX array or a traced material parameter.
        return isinstance(value,Real) and not isinstance(value,(bool,np.bool_)) and np.isfinite(value)
    def medium(material,value):
        return (type(material) is fd.Material and material.mu is None
                and all(scalar(v) and v==value for v in material.eps))
    valid=(isinstance(center,(tuple,list)) and all(scalar(v) for v in center)
           and scalar(radius) and radius>0 and scalar(eps) and eps>0
           and components in ((1,2),(0,1,2)) and len(center)==len(components)
           and len(scene.structures)==1 and len(p.constitutive)==1
           and p.dtype=='float64' and not p.has_E0 and not p.has_H0
           and medium(scene.background,1.))
    if valid:
        structure=scene.structures[0];body=structure.shape;old=p.constitutive[0]
        valid=(medium(structure.material,eps) and old.field=='E' and not old.masks
               and ((components==(1,2) and type(body) is CylinderSlice and p.nx==1
                     and radius==RADIUS and eps==12.)
                    or (components==(0,1,2) and type(body) is fd.Sphere and body.radius==radius))
               and tuple(body.center)==tuple(center))
    if not valid:
        raise ValueError('StencilScopeRefused: a concrete float64 lossless sphere or historical TE cylinder in vacuum is required')


def replace(scene,p,params,center,radius,eps,components):
    validate_scope(scene,p,center,radius,eps,components)
    from wavebit.fd._lower.constitutive import common_points
    from wavebit.fd._kernel.masks import Region
    from wavebit.fd._lower.checks import cfl_axis_sum
    old,=p.constitutive
    # Two-cell reach extends the old compact region by three nodes, covering
    # every coupled edge and an exact vacuum halo for zero extension.
    box=tuple((lo-3,hi+3) if 2-i in components else (lo,hi) for i,(lo,hi) in enumerate(old.box))
    for (lo,hi),size in zip(box,(p.nz,p.ny,p.nx)):
        if not 0<=lo<hi<=size: raise ValueError('StencilRegionRefused')
    term=StencilTerm(old.name,old.field,Region(z=box[0],y=box[1],x=box[2]),old.ratio,old.masks,components)
    nodes=common_points(p.grid,tuple(reversed(box)),'E')[...,components]
    matrix,report=geometry(nodes,p.dx,center,radius,eps,components)
    # The static width/size screen measured <=7.673e-14; 1e-11 is over
    # 100 times that rounding scale. It is not a scattering-accuracy gate.
    if report['lsmr_stop'] not in (0,1,2) or report['residual_max']>1e-11:
        raise ValueError(f'StencilConsistencyRefused: {report}')
    diagonal,index,value=padded(matrix)
    if report['minimum_eigenvalue']<=0: raise ValueError(f'StencilEnergyRefused: {report}')
    report.update(energy_bounds(matrix))
    layers=dict(p.pml_cells)
    for a,((lo,hi),size) in enumerate(zip(box,(p.nz,p.ny,p.nx))):
        if 2-a not in components: continue
        if lo<layers.get('zyx'[a]+'-',0) or hi>size-layers.get('zyx'[a]+'+',0):
            raise ValueError('StencilPMLRefused')
        shape=term.region.shape
        grid=np.arange(len(components)*np.prod(shape)).reshape((len(components),)+shape)
        for end in (0,-1):
            halo=np.take(grid,end,axis=a+1).ravel()
            expected=coo_matrix((np.ones(len(halo)),(np.arange(len(halo)),halo)),shape=(len(halo),len(diagonal))).tocsr()
            if (matrix[halo]-expected).nnz: raise ValueError('StencilHaloRefused')
    rate=p.dt**2*cfl_axis_sum(scene,p.dx)*report['spectral_upper_bound']*float(np.max(params['bh']/old.ratio))
    report['dt_squared_rate']=rate
    if not np.isfinite(rate) or not 0<rate<1: raise ValueError(f'StencilCFLRefused: {rate}')
    params=dict(params,be=params['be'].copy(),stencil_index=index,stencil_value=value)
    d=diagonal.reshape((len(components),)+term.region.shape)
    for k,c in enumerate(components): params['be'][(c,)+term.region.slices]=d[k]*term.ratio
    params['stable']=1.
    p=p.with_static(terms=tuple(term if t is old else t for t in p.terms))
    return p,params,report


def cylinder(cpw=40,placement=(0.,0.)):
    import time,warnings
    from studies.fdtd_subpixel_coupled_cylinder import prepare,solve
    from studies.fdtd_subpixel_mie_cylinder import scene_for,SCHEDULE,CONTOUR,RADIUS,incident_intensity,scattered_flux,cylinder_cross_sections
    from wavebit.fd._lower import plan,lower
    from wavebit.fd._run import execute,_results
    from wavebit.fd._runtime import runtime
    from types import SimpleNamespace
    begin=time.perf_counter();empty=scene_for(cpw,'TE',*SCHEDULE[12.]);rt=runtime('jax');base=plan(empty)
    center=tuple(v/cpw for v in placement);scene,p=prepare(empty,base,12.,center)
    p,params,bounds=replace(scene,p,lower(scene,p,'numpy'),center,RADIUS,12.,(1,2))
    print(json.dumps(dict(preparation=bounds)),flush=True)
    ref=solve(empty,empty,base,rt)
    result=_results(empty,p,execute(p,params,rt),rt,params)
    incident=incident_intensity(ref,'TE',CONTOUR,base.grid)
    sigma=scattered_flux(result,ref,'TE',CONTOUR,p.grid)[0]/incident
    exact=float(cylinder_cross_sections(RADIUS,12.,1.,'TE')[1])
    zero={'box':SimpleNamespace(fields={c:0*v for c,v in result['box'].fields.items()},coords=result['box'].coords)}
    absorption=-scattered_flux(result,zero,'TE',CONTOUR,p.grid)[0]/incident
    times,energies=result.diagnostics['interior_energy']
    return dict(cpw=cpw,placement=placement,sigma=sigma,exact=exact,error=sigma/exact-1,
        absorption=absorption,decay=float(result.diagnostics['decay']['box']),
        # Same four-cell contour displacement as the historical comparison.
        contours=[scattered_flux(result,ref,'TE',CONTOUR+n/cpw,p.grid)[0]/incident for n in (-4,4)],
        preparation=bounds,seconds=time.perf_counter()-begin,
        interior_energy=dict(time=np.asarray(times).tolist(),value=np.asarray(energies).tolist()))


def sphere(cpw=40,eps=12.,periods=48.,radius=.1,placement=(0.,0.,0.)):
    import time
    from studies.fdtd_subpixel_mie_sphere import sphere_scene,read_result
    from wavebit.fd._lower import plan,lower
    from wavebit.fd._runtime import runtime
    from wavebit.fd._run import execute,_results
    begin=time.perf_counter()
    scene=sphere_scene(cpw,eps,periods,placement,radius=radius,propagation='yee')
    p=plan(scene)
    center=scene.structures[0].shape.center
    p,params,bounds=replace(scene,p,lower(scene,p,'numpy'),center,radius,eps,(0,1,2))
    print(json.dumps(dict(preparation=bounds)),flush=True)
    rt=runtime('jax');res=_results(scene,p,execute(p,params,rt),rt,params)
    row=read_result(res,eps,radius)
    times,energy=res.diagnostics['interior_energy']
    row.update(cpw=cpw,eps=eps,radius=radius,periods=periods,placement=placement,
        preparation=bounds,interior_energy=dict(time=np.asarray(times).tolist(),value=np.asarray(energy).tolist()),
        seconds=time.perf_counter()-begin)
    return row


if __name__=='__main__':
    import argparse,warnings
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--shape',choices=('cylinder','sphere'),default='cylinder')
    parser.add_argument('--cpw',type=int,default=40)
    parser.add_argument('--sweep-cylinder',action='store_true',help='All historical 40/56/80 by eight placement cases')
    args=parser.parse_args();out=args.output
    if out.exists(): raise ValueError('DuplicateMeasurementRefused')
    watched=list((ROOT/'wavebit/fd').rglob('*.py'))+list((ROOT/'studies').glob('fdtd_subpixel*.py'))+[Path(__file__)]
    def hashes():return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    before=hashes()
    if args.sweep_cylinder:
        if args.shape!='cylinder': parser.error('--sweep-cylinder requires --shape cylinder')
        from studies.fdtd_subpixel_mie_cylinder import PLACEMENTS,convergence
        result=dict(status='running_research_cylinder_sweep',rows=[],source_sha256=before)
        def save():out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        save()
        try:
            errors=[]
            for cpw in (40,56,80):
                group=[]
                for index,placement in enumerate(PLACEMENTS):
                    result['current']=dict(cpw=cpw,placement=index);save()
                    with warnings.catch_warnings(record=True) as notes:
                        row=cylinder(cpw,placement)
                    row.update(placement_index=index,warnings=[str(w.message) for w in notes])
                    result['rows'].append(row);group.append(row['error']);save()
                    print(json.dumps({k:row[k] for k in ('cpw','placement_index','error','decay','seconds')}),flush=True)
                    if hashes()!=before:raise ValueError('MeasurementSourceChangedRefused')
                    # Preserve the existing cylinder temporal control.
                    if not np.isfinite(row['error']) or row['decay']>=1e-7:
                        raise ValueError('StencilTemporalControlUnresolved')
                errors.append(group)
            result.update(status='completed_measurements_not_C3_pass',current=None,
                fit=convergence((40,56,80),errors),source_unchanged=hashes()==before)
            save()
        except Exception as error:
            result.update(status='stopped_for_review',failure=f'{type(error).__name__}: {error}',source_unchanged=hashes()==before)
            save();raise
        sys.exit(0)
    with warnings.catch_warnings(record=True) as notes:
        row=(cylinder if args.shape=='cylinder' else sphere)(args.cpw)
    assert before==hashes()
    out.write_text(json.dumps(dict(status='research_control_not_C3_pass',source_sha256=before,
        source_unchanged=True,warnings=[str(w.message) for w in notes],row=row),indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k!='interior_energy'}),flush=True)
