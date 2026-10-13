"""Research symmetric extended-stencil material map, not a public API.

At each target edge, freeze the interface normal and integrate the exact
constant tangential-E / normal-D states on every nearby edge and dual face.
Solve K Pi_i = Gamma_i row constraints with K symmetric, minimizing a
weighted Frobenius correction to the symmetric triplet map. Positive energy is NOT
assumed; inspect the entire assembled operator before advancing any field.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_root
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix

ROOT=repository_root()
sys.path.insert(0,str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
# The five assembly functions and the surface clearance rule live in
# src/wavebit/fdtd/_prepare/stencil.py and are re-exported here under their old names: replace()
# below calls geometry/padded/energy_bounds/surface_clearance, and
# tests/fdtd/test_fd_symmetric_stencil.py reaches fit and triplet_baseline through this module.
from wavebit.fdtd._prepare.stencil import energy_bounds, fit, geometry, padded, surface_clearance, triplet_baseline  # noqa: F401
from wavebit.fdtd._kernel.constitutive import ConstitutiveTerm


@dataclasses.dataclass(frozen=True)
class StencilTerm(ConstitutiveTerm):
    components: tuple=()
    def prepare(self,curl,mem,params,rt):
        local=curl[self.view]
        compact=rt.xp.stack(tuple(local[c] for c in self.components)).reshape(-1)
        value=rt.xp.sum(compact[params['stencil_index']]*params['stencil_value'],axis=1)
        value=rt.xp.concat((rt.xp.zeros_like(compact[:1]),value))[params['stencil_output']]
        value=value.reshape((len(self.components),)+self.region.shape)
        delta=rt.xp.stack(tuple(value[self.components.index(c)] if c in self.components
            else rt.xp.zeros_like(local[c]) for c in range(3)))
        return mem+self.ratio*local,self.ratio*delta


def validate_scope(scene,p,center,radius,eps,components):
    """Refuse unsupported research inputs on the host, before assembly."""
    from numbers import Real
    import wavebit.fdtd._legacy as fd
    from studies.fdtd.subpixel_coupled_cylinder.fdtd_subpixel_coupled_cylinder import CylinderSlice
    from studies.fdtd.subpixel_mie_cylinder.fdtd_subpixel_mie_cylinder import RADIUS
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
    from wavebit.fdtd._prepare.constitutive import common_points
    from wavebit.fdtd._kernel.masks import Region
    from wavebit.fdtd._prepare.checks import cfl_axis_sum
    old,=p.constitutive
    # Two-cell reach extends the old compact region by three nodes, covering
    # every coupled edge and an exact vacuum halo for zero extension.
    box=tuple((lo-3,hi+3) if 2-i in components else (lo,hi) for i,(lo,hi) in enumerate(old.box))
    for (lo,hi),size in zip(box,(p.nz,p.ny,p.nx)):
        if not 0<=lo<hi<=size: raise ValueError('StencilRegionRefused')
    term=StencilTerm(old.name,old.field,Region(z=box[0],y=box[1],x=box[2]),old.ratio,old.masks,components)
    nodes=common_points(p.grid,tuple(reversed(box)),'E')[...,components]
    matrix,report=geometry(nodes,p.dx,center,radius,eps,components)
    report['surface_clearance']=surface_clearance(scene,p,matrix,box,components)
    # The static width/size screen measured <=7.673e-14; 1e-11 is over
    # 100 times that rounding scale. It is not a scattering-accuracy gate.
    if report['lsmr_stop'] not in (0,1,2) or report['residual_max']>1e-11:
        raise ValueError(f'StencilConsistencyRefused: {report}')
    diagonal,index,value,output=padded(matrix)
    report['stencil_storage']=dict(rows=matrix.shape[0],active_rows=len(index),width=index.shape[1],
        full_padded_bytes=matrix.shape[0]*index.shape[1]*(index.itemsize+value.itemsize),
        active_padded_bytes=index.nbytes+value.nbytes+output.nbytes)
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
    params=dict(params,be=params['be'].copy(),stencil_index=index,stencil_value=value,stencil_output=output)
    d=diagonal.reshape((len(components),)+term.region.shape)
    for k,c in enumerate(components): params['be'][(c,)+term.region.slices]=d[k]*term.ratio
    params['stable']=1.
    p=p.with_static(terms=tuple(term if t is old else t for t in p.terms))
    return p,params,report


def cylinder(cpw=40,placement=(0.,0.)):
    import time
    from studies.fdtd.subpixel_coupled_cylinder.fdtd_subpixel_coupled_cylinder import prepare, solve
    from studies.fdtd.subpixel_mie_cylinder.fdtd_subpixel_mie_cylinder import scene_for, SCHEDULE, CONTOUR, RADIUS, incident_intensity, scattered_flux, cylinder_cross_sections
    from wavebit.fdtd._prepare import plan, lower
    from wavebit.fdtd._execution import execute, _results
    from wavebit.fdtd._runtime import runtime
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


def sphere(cpw=40,eps=12.,periods=48.,radius=.1,placement=(0.,0.,0.),absorption_half=.4):
    import time
    import wavebit.fdtd._legacy as fd
    from studies.fdtd.subpixel_mie_sphere.fdtd_subpixel_mie_sphere import sphere_scene, read_result
    from wavebit.fdtd._prepare import plan, lower
    from wavebit.fdtd._runtime import runtime
    from wavebit.fdtd._execution import execute, _results
    begin=time.perf_counter()
    scene=sphere_scene(cpw,eps,periods,placement,radius=radius,propagation='yee')
    # The shifted 20-cell radius-.25 audit found 22/45 nonvacuum rows on
    # the old .4-halfwidth surface; .45 enclosed all of them exactly.
    # This is an explicit experiment input, never an automatic relocation.
    if absorption_half!=.4:
        scene=scene.replace(monitors=tuple(dataclasses.replace(m,box=fd.Region(size=(2*absorption_half,)*3))
            if m.name=='abs' else m for m in scene.monitors))
    p=plan(scene)
    center=scene.structures[0].shape.center
    p,params,bounds=replace(scene,p,lower(scene,p,'numpy'),center,radius,eps,(0,1,2))
    print(json.dumps(dict(preparation=bounds)),flush=True)
    rt=runtime('jax');res=_results(scene,p,execute(p,params,rt),rt,params)
    row=read_result(res,eps,radius)
    times,energy=res.diagnostics['interior_energy']
    row.update(cpw=cpw,eps=eps,radius=radius,periods=periods,placement=placement,absorption_half=absorption_half,
        preparation=bounds,interior_energy=dict(time=np.asarray(times).tolist(),value=np.asarray(energy).tolist()),
        seconds=time.perf_counter()-begin)
    return row


if __name__=='__main__':
    import argparse, warnings
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--shape',choices=('cylinder','sphere'),default='cylinder')
    parser.add_argument('--cpw',type=int,default=40)
    parser.add_argument('--absorption-half',type=float,default=.4,help='Explicit sphere absorption surface half-width')
    parser.add_argument('--sweep-cylinder',action='store_true',help='All historical 40/56/80 by eight placement cases')
    args=parser.parse_args();out=args.output
    if args.shape!='sphere' and args.absorption_half!=.4:parser.error('--absorption-half requires --shape sphere')
    if out.exists(): raise ValueError('DuplicateMeasurementRefused')
    watched=list((ROOT/'src/wavebit/fdtd').rglob('*.py'))+list((ROOT/'studies').glob('fdtd_subpixel*.py'))+[Path(__file__)]
    def hashes():return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    before=hashes()
    if args.sweep_cylinder:
        if args.shape!='cylinder': parser.error('--sweep-cylinder requires --shape cylinder')
        from studies.fdtd.subpixel_mie_cylinder.fdtd_subpixel_mie_cylinder import PLACEMENTS, convergence
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
        row=cylinder(args.cpw) if args.shape=='cylinder' else sphere(args.cpw,absorption_half=args.absorption_half)
    assert before==hashes()
    out.write_text(json.dumps(dict(status='research_control_not_C3_pass',source_sha256=before,
        source_unchanged=True,warnings=[str(w.message) for w in notes],row=row),indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k!='interior_energy'}),flush=True)
