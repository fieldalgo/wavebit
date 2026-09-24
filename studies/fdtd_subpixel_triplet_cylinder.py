"""Research-only invariant-x version of Werner et al. (2013), sections 4/7.

No production source is replaced. The local edge/dual-face map is symmetrized
before use in time stepping. Its positive eigenvalues bound the assembled
operator, an average of four permuted block-diagonal operators. This is a
candidate comparator, not a claim of second-order accuracy or a public API.
"""
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from wavebit.fd._kernel.constitutive import ConstitutiveTerm
from wavebit.fd._shapes import Sphere
from studies.fdtd_subpixel_interface_consistency import fraction

# Four oriented pairs replace the eight 3D triplets when x is invariant.
ORIENTATIONS = ((1,1), (1,-1), (-1,1), (-1,-1))


def shift(v, direction, axis, xp=np):
    head, tail = [slice(None)]*3, [slice(None)]*3
    head[axis] = slice(0,1)
    zero = xp.zeros_like(v[tuple(head)])
    if direction < 0:
        tail[axis] = slice(1,None)
        return xp.concat((v[tuple(tail)],zero), axis=axis)
    tail[axis] = slice(None,-1)
    return xp.concat((zero,v[tuple(tail)]), axis=axis)


def gather(v, sign, component, xp=np):
    return v if sign > 0 else shift(v,1,2-component,xp)


def scatter(v, sign, component, xp=np):
    return v if sign > 0 else shift(v,-1,2-component,xp)


def local_map(normal, lengths, areas, eps):
    """Gamma Pi^-1 from E edge lengths and D dual-face areas, eps2=1.

    Gamma_p = I + (1/eps_p-1) nn^T and Pi_p = eps_p Gamma_p.
    Fractions left-multiply these matrices row by row (paper Eqs. 15-32).
    """
    nn = normal[..., :, None]*normal[..., None, :]
    eye = np.eye(2)
    gamma = eye + lengths[..., :, None]*(1/eps-1)*nn
    pi = eye + areas[..., :, None]*(eps-1)*(eye-nn)
    determinant = np.linalg.det(pi)
    if not np.all(np.isfinite(determinant) & (determinant > 0)):
        raise ValueError('TripletPiRefused: nonpositive or nonfinite determinant at preparation')
    accurate = np.linalg.solve(np.swapaxes(pi,-1,-2),np.swapaxes(gamma,-1,-2))
    accurate = np.swapaxes(accurate,-1,-2)
    return accurate, (accurate+np.swapaxes(accurate,-1,-2))/2


def planar_maps(points, dx, normal, offset, eps):
    normal = np.asarray(normal)
    accurate, symmetric = [], []
    for sy,sz in ORIENTATIONS:
        ell, area = [], []
        for i,sign in enumerate((sy,sz)):
            p = points.copy()
            p[...,i] += sign*dx/2
            cut = offset-np.sum(p*normal,axis=-1)
            ell.append(fraction(cut,abs(normal[i])*dx/2))
            area.append(fraction(cut,abs(normal[1-i])*dx/2))
        a,s = local_map(normal,np.stack(ell,axis=-1),np.stack(area,axis=-1),eps)
        accurate.append(a); symmetric.append(s)
    return np.stack(accurate),np.stack(symmetric)


def circle_fraction(point, dx, axis, radius, xp=np):
    """Exact length of the circle's intersection with a centered segment.

    The invariant-x plane of a sphere **is** a circle of the same radius, so
    this is `wavebit/fd/_shapes/_bodies.py::Sphere.exact_edge_fill` on the
    (y,z) point lifted to a triple with x exactly zero: the lifted transverse
    sum is `0*0 + t*t`, which is the `t**2` this used to write. The public form
    holds the two guards this docstring used to carry -- the segment-coordinate
    clip that keeps a whole segment exactly one (the world-coordinate spelling
    gave 1+8.88179e-16 in the radius-one, dx=.1 audit) and the guarded root.

    `xp` is what the lift is built with, and it is why this function has one at
    all: without it a JAX or Torch point is silently pulled through `np.stack`
    into NumPy, which loses the trace and the gradient with it. The 2D research
    path calls this with `axis` 0 for y and 1 for z, so the lifted component
    index is `axis+1`.
    """
    zero = xp.zeros_like(point[...,0])
    lifted = xp.stack((zero,point[...,0],point[...,1]),axis=-1)
    return Sphere(radius).exact_edge_fill(lifted,dx,axis+1)


def circle_maps(points, dx, center, radius, eps):
    relative = points-np.asarray(center)
    norm = np.linalg.norm(relative,axis=-1)
    normal = relative/np.where(norm>0,norm,1.)[...,None]
    # The center is uniform material; either normal gives the same inverse.
    normal = np.where((norm==0)[...,None],np.array((1.,0.)),normal)
    accurate,symmetric = [],[]
    for sy,sz in ORIENTATIONS:
        ell,area = [],[]
        for i,sign in enumerate((sy,sz)):
            p = relative.copy(); p[...,i] += sign*dx/2
            ell.append(circle_fraction(p,dx,i,radius))
            area.append(circle_fraction(p,dx,1-i,radius))
        a,s = local_map(normal,np.stack(ell,axis=-1),np.stack(area,axis=-1),eps)
        accurate.append(a); symmetric.append(s)
    return np.stack(accurate),np.stack(symmetric)


def action(vector, matrices, xp=np):
    """Assemble I + average P^T(local-I)P with a vacuum exterior.

    Subtracting I retains unit diagonal at a truncated vacuum halo.
    Interior production use requires the correction to vanish at that halo.
    """
    out = [vector[0], vector[1], vector[2]]
    for k,(sy,sz) in enumerate(ORIENTATIONS):
        y,z = gather(vector[1],sy,1,xp),gather(vector[2],sz,2,xp)
        m = matrices[k]
        dy = (m[...,0,0]-1)*y+m[...,0,1]*z
        dz = m[...,1,0]*y+(m[...,1,1]-1)*z
        out[1] = out[1]+scatter(dy,sy,1,xp)/4
        out[2] = out[2]+scatter(dz,sz,2,xp)/4
    return xp.stack(out)


def diagonal(matrices):
    out = [np.ones(matrices.shape[1:-2]) for _ in range(3)]
    for k,(sy,sz) in enumerate(ORIENTATIONS):
        out[1] += scatter(matrices[k,...,0,0]-1,sy,1)/4
        out[2] += scatter(matrices[k,...,1,1]-1,sz,2)/4
    return np.stack(out)


def cross_action(vector, matrices, xp=np):
    """Apply only off-diagonal terms; the diagonal stays in the usual slot."""
    out = [xp.zeros_like(vector[0]) for _ in range(3)]
    for k,(sy,sz) in enumerate(ORIENTATIONS):
        y,z = gather(vector[1],sy,1,xp),gather(vector[2],sz,2,xp)
        out[1] = out[1]+scatter(matrices[k,...,0,1]*z,sy,1,xp)/4
        out[2] = out[2]+scatter(matrices[k,...,1,0]*y,sz,2,xp)/4
    return xp.stack(out)


def certificate(scene, p, term, matrices, params):
    """Whole prepared research operator: local SPD, vacuum halo and CFL.

    Four permutations of local SPD blocks, each with weight 1/4, cover each
    yz edge exactly once. Outside the finite patch the blocks are I. Thus
    min/max local eigenvalues, together with the uncoupled x entries, bound
    the global spectrum. The existing curl norm bound supplies the CFL test.
    """
    from wavebit.fd._lower.checks import cfl_axis_sum
    if (term.field != 'E' or term.masks or p.nx != 1 or scene.structures[0].material.eps != (12.,)*3):
        raise ValueError('TripletScopeRefused: only this lossless invariant-x epsilon-12 fixture is implemented')
    for (lo,hi),size in zip(term.box,(p.nz,p.ny,p.nx)):
        if not 0 <= lo < hi <= size:
            raise ValueError('TripletRegionRefused: compact patch exceeds the grid')
    layers=dict(p.pml_cells)
    if term.region.z[0] < layers.get('z-',0) or term.region.z[1] > p.nz-layers.get('z+',0):
        raise ValueError('TripletLayerRefused: compact patch reaches a PML layer')
    if not np.all(np.isfinite(matrices)):
        raise ValueError('TripletFiniteRefused: nonfinite local coefficient')
    asymmetry = float(np.max(abs(matrices-np.swapaxes(matrices,-1,-2))))
    if asymmetry != 0:
        raise ValueError(f'TripletSymmetryRefused: max asymmetry={asymmetry}')
    correction = matrices-np.eye(2)
    # Array axes: orientation, z, y, x, row, column. Only y/z are coupled.
    for axis in (1,2):
        for end in (0,-1):
            if np.any(np.take(correction,end,axis=axis) != 0):
                raise ValueError('TripletHaloRefused: non-vacuum local block at compact patch end')
    eigenvalues = np.linalg.eigvalsh(matrices)
    lo = min(1.,float(eigenvalues.min()),float(np.min(params['be'][0]/term.ratio)))
    hi = max(1.,float(eigenvalues.max()),float(np.max(params['be'][0]/term.ratio)))
    h_upper = float(np.max(params['bh']/term.ratio))
    cfl = p.dt**2*cfl_axis_sum(scene,p.dx)*hi*h_upper
    if not (lo > 0 and np.isfinite(hi) and cfl < 1):
        raise ValueError(f'TripletEnergyCFLRefused: spectrum=[{lo},{hi}], dt^2 rate={cfl}')
    return dict(lower=lo,upper=hi,dt_squared_rate=cfl,local_asymmetry=asymmetry,
                vacuum_halo_exact=True,method='average_of_four_permuted_local_SPD_blocks')


def operator_audit(matrices):
    """Compare shift assembly with an independently indexed sparse matrix."""
    from scipy.sparse import coo_matrix
    shape = matrices.shape[1:-2]
    size = int(np.prod(shape))
    rows,cols,values=[],[],[]
    for k,(sy,sz) in enumerate(ORIENTATIONS):
        for z,y,x in np.ndindex(shape):
            cells=((z,y if sy>0 else y-1,x),(z if sz>0 else z-1,y,x))
            for i in range(2):
                for j in range(2):
                    if any(not(0<=v<n) for cell in (cells[i],cells[j]) for v,n in zip(cell,shape)):
                        continue
                    rows.append(i*size+np.ravel_multi_index(cells[i],shape))
                    cols.append(j*size+np.ravel_multi_index(cells[j],shape))
                    values.append((matrices[k,z,y,x,i,j]-(1. if i==j else 0.))/4)
    idx=np.arange(2*size)
    operator=coo_matrix((values,(rows,cols)),shape=(2*size,2*size)).tocsr()
    operator=operator+coo_matrix((np.ones(2*size),(idx,idx)),shape=operator.shape).tocsr()
    vector=np.sin(np.arange(3*size).reshape((3,)+shape))
    actual=action(vector,matrices)[1:].reshape(-1)
    expected=operator@vector[1:].reshape(-1)
    split=(diagonal(matrices)*vector+cross_action(vector,matrices))[1:].reshape(-1)
    difference=operator-operator.T
    return dict(indexed_action_gap=float(np.max(abs(actual-expected))),
                diagonal_cross_split_gap=float(np.max(abs(actual-split))),
                assembled_asymmetry=float(np.max(abs(difference.data),initial=0.)),
                assembled_min_eigenvalue=float(np.linalg.eigvalsh(operator.toarray())[0]))


def prepare_circle(empty, base, eps, center):
    from studies.fdtd_subpixel_coupled_cylinder import prepare, RADIUS
    from wavebit.fd._lower import lower
    from wavebit.fd._lower.constitutive import common_points
    from wavebit.fd._runtime import runtime
    scene,p = prepare(empty,base,eps,center)
    original = lower(scene,p,runtime('numpy'))
    old = p.constitutive[0]
    from wavebit.fd._kernel.masks import Region
    # One further node includes the end of each negatively oriented edge.
    # At 40 cells the old halo had a 7.10543e-15 grazing correction: retain
    # it in this wider patch, rather than zeroing it with a tolerance.
    z,y,x=old.box
    region=Region(z=(z[0]-1,z[1]+1),y=(y[0]-1,y[1]+1),x=x)
    new=TripletTerm(old.name,old.field,region,old.ratio,old.masks)
    points = common_points(p.grid,tuple(reversed(new.box)),'E')[...,1:]
    _,matrices=circle_maps(points,p.dx,center,RADIUS,eps)
    params=dict(original,be=original['be'].copy(),research_triplets=matrices)
    d=diagonal(matrices)
    params['be'][(slice(1,3),)+new.region.slices]=d[1:]*old.ratio
    bounds=certificate(scene,p,new,matrices,params)
    params['stable']=1. # The named whole-operator host certificate just succeeded.
    p=p.with_static(terms=tuple(new if t is old else t for t in p.terms))
    return scene,p,params,bounds,matrices


@dataclasses.dataclass(frozen=True)
class TripletTerm(ConstitutiveTerm):
    """Research TE term; prepared host-side SPD/CFL certificate is required."""
    def prepare(self,curl,mem,params,rt):
        local = curl[self.view]
        flux = mem+self.ratio*local
        delta = self.ratio*cross_action(local,params['research_triplets'],rt.xp)
        return flux,delta


def static_audit():
    rows=[]
    for normal in ((1.,0.),(.6,.8)):
        for eps in (1.,2.25,12.):
            for cells in (16,32,64):
                dx=1/cells
                q=np.arange(-cells-2,cells+3)*dx
                zz,yy=np.meshgrid(q,q,indexing='ij')
                points=np.stack((yy,zz),axis=-1)[:,:,None,:]
                accurate,symmetric=planar_maps(points,dx,normal,.23*dx,eps)
                n=np.array(normal);t=np.array((-n[1],n[0]))
                for et,dn in ((1.,0.),(0.,1.)):
                    e=np.zeros((3,)+points.shape[:-1]);d=e.copy()
                    for i in range(2):
                        p=points.copy();p[...,i]+=dx/2
                        cut=.23*dx-np.sum(p*n,axis=-1)
                        ell=fraction(cut,abs(n[i])*dx/2)
                        area=fraction(cut,abs(n[1-i])*dx/2)
                        e[i+1]=et*t[i]+dn*n[i]*(ell/eps+1-ell)
                        d[i+1]=et*t[i]*(area*eps+1-area)+dn*n[i]
                    row=dict(normal=normal,eps=eps,cells=cells,et=et,dn=dn)
                    for name,m in (('accurate_asymmetric',accurate),('symmetric',symmetric)):
                        v=action(d,m)
                        err=(v-e)[:,2:-2,2:-2]
                        curl=(np.roll(v[2],-1,axis=1)-v[2]-np.roll(v[1],-1,axis=0)+v[1])[2:-2,2:-2]
                        row[name]=dict(max_error=float(np.max(abs(err))),
                                       mean_error=float(np.mean(abs(err[1:]))),
                                       raw_circulation=float(np.max(abs(curl))))
                    row['min_local_eigenvalue']=float(np.min(np.linalg.eigvalsh(symmetric)))
                    rows.append(row)
    return rows


import argparse
import os
import time
from types import SimpleNamespace
import warnings
from studies.fdtd_subpixel_coupled_cylinder import prepare, solve
from studies.fdtd_subpixel_mie_cylinder import (
    scene_for, SCHEDULE, CONTOUR, RADIUS, PLACEMENTS, incident_intensity,
    scattered_flux, cylinder_cross_sections, convergence,
)
from wavebit.fd._lower import plan
from wavebit.fd._run import execute, _results
from wavebit.fd._runtime import runtime


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--mode',choices=('scatter','static','preflight'),default='scatter')
    parser.add_argument('--cells',default='40')
    parser.add_argument('--placements',default='0')
    parser.add_argument('--production-control',action='store_true')
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists(): raise ValueError('DuplicateMeasurementRefused')
    if args.mode=='static':
        output.write_text(json.dumps(dict(status='static_diagnostic_not_scattering_pass',rows=static_audit()),indent=2)+'\n',encoding='utf8')
        return
    cells=tuple(map(int,args.cells.split(',')))
    indices=tuple(map(int,args.placements.split(',')))
    watched=sorted((ROOT/'wavebit/fd').rglob('*.py'))+[
        ROOT/'studies/fdtd_subpixel_coupled_cylinder.py',
        ROOT/'studies/fdtd_subpixel_mie_cylinder.py',
        Path(__file__)]
    def hashes():
        return {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p.name):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    source=hashes()
    out=dict(status='running_research_comparator',pid=os.getpid(),source_sha256=source,
             public_scene_preparation=False,method='symmetric_edge_face_triplets',rows=[],references=[])
    def save(): output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    save()
    exact=float(cylinder_cross_sections(RADIUS,12.,1.,'TE')[1])
    if args.mode=='preflight':
        for cpw in cells:
            empty=scene_for(cpw,'TE',*SCHEDULE[12.])
            for index in indices:
                _,p,params,bounds,matrices=prepare_circle(empty,plan(empty),12.,tuple(v/cpw for v in PLACEMENTS[index]))
                out['rows'].append(dict(cpw=cpw,placement_index=index,certificate=bounds,operator_audit=operator_audit(matrices)))
        out.update(status='completed_preflight_not_scattering_pass',source_unchanged=hashes()==source)
        save()
        return
    rt=runtime('jax')
    with warnings.catch_warnings(record=True) as notes:
        for cpw in cells:
            empty=scene_for(cpw,'TE',*SCHEDULE[12.])
            base=plan(empty)
            begin=time.perf_counter()
            ref=solve(empty,empty,base,rt)
            incident=incident_intensity(ref,'TE',CONTOUR,base.grid)
            out['references'].append(dict(cpw=cpw,seconds=time.perf_counter()-begin,
                incident=incident,decay=float(ref.diagnostics['decay']['box'])))
            save();print(json.dumps(out['references'][-1]),flush=True)
            for index in indices:
                placement=PLACEMENTS[index];center=tuple(v/cpw for v in placement)
                variants=('production','triplet') if args.production_control else ('triplet',)
                for variant in variants:
                    if hashes()!=source: raise RuntimeError('MeasurementSourceChanged')
                    begin=time.perf_counter()
                    if variant=='production':
                        scene,p=prepare(empty,base,12.,center)
                        result=solve(empty,scene,p,rt)
                        bounds=None
                    else:
                        scene,p,params,bounds,_=prepare_circle(empty,base,12.,center)
                        result=_results(empty,p,execute(p,params,rt),rt,params)
                    sigma=scattered_flux(result,ref,'TE',CONTOUR,p.grid)[0]/incident
                    zero={'box':SimpleNamespace(fields={c:0*v for c,v in result['box'].fields.items()},
                                                coords=result['box'].coords)}
                    absorption=-scattered_flux(result,zero,'TE',CONTOUR,p.grid)[0]/incident
                    times,values=result.diagnostics['interior_energy'];values=np.asarray(values)
                    row=dict(cpw=cpw,placement_index=index,placement=placement,variant=variant,
                        sigma=sigma,exact=exact,error=sigma/exact-1,certificate=bounds,
                        decay=float(result.diagnostics['decay']['box']),
                        absorption_cross_section=absorption,
                        final_energy_over_peak=float(values[-1]/max(values)),
                        contours=[scattered_flux(result,ref,'TE',CONTOUR+n/cpw,p.grid)[0]/incident
                                  for n in (-4,4)],
                        seconds=time.perf_counter()-begin,
                        energy=dict(time=np.asarray(times).tolist(),value=values.tolist()))
                    out['rows'].append(row);save()
                    print(json.dumps({k:v for k,v in row.items() if k!='energy'}),flush=True)
                    if hashes()!=source: raise RuntimeError('MeasurementSourceChanged')
        if len(cells)>=3:
            out['fit']=convergence(cells,[[r['error'] for r in out['rows']
                    if r['cpw']==cpw and r['variant']=='triplet'] for cpw in cells])
        out.update(status='completed_research_comparator_not_C3_pass',source_unchanged=True,
                   warnings=list(dict.fromkeys(str(w.message) for w in notes)))
        save()


if __name__=='__main__':
    from pathlib import Path
    main()
