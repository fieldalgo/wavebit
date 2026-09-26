"""Separate tangent-normal error from the production coupling discretization.

Its logged rows predate the per-block lift (``wavebit/fd/_lower/blocks.py``): the production
operator it lowered then was the unlifted default rule, and ``guard``'s positivity certificate
was the Gershgorin lower bound, which this study now takes itself for the candidate operators it
builds (they have not been through the lift). Rerun, the ``tangent`` variant lowers today's
lifted operator and the ``exact_both`` variant replaces its diagonal with an unlifted one.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from wavebit.fd._lower import lower, plan
from wavebit.fd._lower.coefficients import mix
from wavebit.fd._lower.constitutive import common_points, guard, enclosure
from wavebit.fd._lower.occupancy import yee_points
from wavebit.fd._runtime import runtime
from wavebit.fd._run import execute, _results
from wavebit.fd._shapes._exact import clipped_disc_area
from studies.fdtd_subpixel_coupled_cylinder import prepare, solve
from studies.fdtd_subpixel_mie_cylinder import (
    RADIUS, CONTOUR, SCHEDULE, scene_for, incident_intensity, scattered_flux,
    cylinder_cross_sections,
)


def arc_moments(points, dx, center=(0.,0.)):
    """Integrate nn^T ds exactly over circle arcs clipped by each rectangle.

    The four rectangle lines intersect the circle at at most eight angles.
    A midpoint selects each resulting arc. With m its midpoint and d its
    angular width, integral cos(2 theta) dtheta = cos(2m)*sin(d).
    sinc evaluates sin(d)/d continuously; no fitted small-angle switch exists.
    """
    y,z=points[...,1]-center[0],points[...,2]-center[1]
    lo_y,hi_y,lo_z,hi_z=y-dx/2,y+dx/2,z-dx/2,z+dx/2
    breaks=[np.zeros_like(y),np.full_like(y,2*np.pi)]
    for edge in (lo_y,hi_y):
        a=np.arccos(np.clip(edge/RADIUS,-1,1))
        breaks.extend((a,2*np.pi-a))
    for edge in (lo_z,hi_z):
        a=np.arcsin(np.clip(edge/RADIUS,-1,1))
        breaks.extend((np.remainder(a,2*np.pi),np.pi-a))
    angles=np.sort(np.stack(breaks,axis=-1),axis=-1)
    width=np.diff(angles,axis=-1)
    middle=(angles[...,1:]+angles[...,:-1])/2
    ym,zm=RADIUS*np.cos(middle),RADIUS*np.sin(middle)
    live=((ym>=lo_y[...,None]) & (ym<=hi_y[...,None])
          & (zm>=lo_z[...,None]) & (zm<=hi_z[...,None]))
    weights=np.where(live,width,0)
    length=RADIUS*np.sum(weights,axis=-1)
    sinc=np.sinc(width/np.pi)
    yy=RADIUS*np.sum(weights*(.5+.5*np.cos(2*middle)*sinc),axis=-1)
    zz=RADIUS*np.sum(weights*(.5-.5*np.cos(2*middle)*sinc),axis=-1)
    yz=RADIUS*np.sum(weights*.5*np.sin(2*middle)*sinc,axis=-1)
    denominator=np.where(length>0,length,1)
    diagonal=np.stack([np.zeros_like(length),yy/denominator,zz/denominator],axis=-1)
    return length,diagonal,yz/denominator


def independent_check(points,dx):
    """Divergence identities use clipped area and straight face integrals."""
    y,z=points[...,1],points[...,2]
    lo_y,hi_y,lo_z,hi_z=y-dx/2,y+dx/2,z-dx/2,z+dx/2
    def face(edge,lo,hi):
        live=abs(edge)<RADIUS
        half=np.sqrt(np.where(live,RADIUS**2-edge**2,1))
        a,b=np.maximum(lo,-half),np.minimum(hi,half)
        length=np.where(live & (b>a),b-a,0)
        return length,length*(a+b)/2
    ly0,_=face(lo_y,lo_z,hi_z);ly1,_=face(hi_y,lo_z,hi_z)
    lz0,iy0=face(lo_z,lo_y,hi_y);lz1,iy1=face(hi_z,lo_y,hi_y)
    area=clipped_disc_area(0.,0.,RADIUS,lo_y,hi_y,lo_z,hi_z,np)
    diagonal=np.stack([np.zeros_like(area),(area-hi_y*ly1+lo_y*ly0)/RADIUS,
                      (area-hi_z*lz1+lo_z*lz0)/RADIUS],axis=-1)
    return diagonal,(iy0-iy1)/RADIUS


from types import SimpleNamespace
from wavebit.fd._kernel.observe import InteriorEnergy
from studies.fdtd_subpixel_mie_cylinder import PLACEMENTS
import os


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--cells',type=int,default=80)
    parser.add_argument('--placement',type=int,choices=range(8),default=4)
    args=parser.parse_args()
    output=args.output
    if output.exists():
        raise RuntimeError('DuplicateMeasurementRefused')
    cpw = args.cells
    # Default: the largest error in the existing eight-placement 80-cell sweep.
    placement = PLACEMENTS[args.placement]
    center = tuple(v/cpw for v in placement)
    def hashes():
        return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT/'wavebit/fd').rglob('*.py'))}
    source = hashes()
    out = dict(status='running', pid=os.getpid(), cpw=cpw, placement=placement,
               center=center, source_sha256=source, rows=[])
    def save():
        output.write_text(json.dumps(out, indent=2)+'\n', encoding='utf8')
    save()
    rt, host = runtime('jax'), runtime('numpy')
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as notes:
        empty = scene_for(cpw, 'TE', *SCHEDULE[12.])
        base = plan(empty)
        reference = solve(empty, empty, base, rt)
        incident = incident_intensity(reference, 'TE', CONTOUR, base.grid)
        out['reference'] = dict(seconds=time.perf_counter()-started, incident=incident,
                               decay=float(reference.diagnostics['decay']['box']))
        print(json.dumps(out['reference']), flush=True)
        save()
        scene, p = prepare(empty, base, 12., center)
        original = lower(scene, p, host)
        body = scene.structures[0].shape.body()
        box = scene.structures[0].shape.support(p.grid)
        view = tuple(slice(*a) for a in reversed(box))
        diag = original['be'].copy()
        gaps = []
        for c in range(3):
            points = yee_points(p.grid, box, 'E', c)
            length, w, cross = arc_moments(points, p.dx, center)
            oracle, oracle_cross = independent_check(points-np.array((0., *center)), p.dx)
            fill = body.exact_fill(points, p.dx)
            difference = np.maximum(np.max(abs(length[...,None]*w-oracle), axis=-1),
                                    abs(length*cross-oracle_cross))/p.dx
            gaps.append(dict(raw=float(np.max(difference)),
                             fill_weighted=float(np.max(fill*(1-fill)*difference))))
            diag[(c,)+view] = (p.dt/p.dx)/mix(np, np.ones_like(fill), 12., fill, w[...,c])
        term = p.constitutive[0]
        points = common_points(p.grid, tuple(reversed(term.box)), 'E')
        length, w, yz = arc_moments(points, p.dx, center)
        fill = body.exact_fill(points, p.dx)
        contrast = fill*(1-fill)*(12.-1.)**2/(12.*(fill*12.+1.-fill))
        cross = np.zeros_like(original['constitutive'][term.name])
        cross[2] = contrast*yz
        out.update(moment_divergence_gap_over_dx=gaps,
                   max_diagonal_inverse_change=float(np.max(abs(diag-original['be']))/(p.dt/p.dx)),
                   max_cross_inverse_change=float(np.max(abs(cross-original['constitutive'][term.name]))),
                   normal_tensor_min_eigenvalue=float(np.min(
                       (w[...,1]+w[...,2]-np.sqrt((w[...,1]-w[...,2])**2+4*yz**2))/2)))
        for variant in ('tangent', 'exact_both'):
            if hashes() != source:
                raise RuntimeError('MeasurementSourceChanged')
            started = time.perf_counter()
            params = dict(original)
            if variant == 'exact_both':
                params['be'] = diag
                params['constitutive'] = dict(original['constitutive'], **{term.name: cross})
            d = InteriorEnergy._cut(params['be'], slice(None), term.box)/term.ratio
            bounds = [float(v) for v in enclosure(d, params['constitutive'][term.name], np)]
            certificate = float(guard(scene, p, params, host, bounds[0] > 0))
            if not np.isfinite(certificate):
                raise RuntimeError('CandidateOperatorRefused')
            params['stable'] = certificate
            res = _results(empty, p, execute(p, params, rt), rt, params)
            sigma = scattered_flux(res, reference, 'TE', CONTOUR, p.grid)[0]/incident
            zero = {'box': SimpleNamespace(fields={c: 0*v for c,v in res['box'].fields.items()},
                                          coords=res['box'].coords)}
            absorption = -scattered_flux(res, zero, 'TE', CONTOUR, p.grid)[0]/incident
            _, energy = res.diagnostics['interior_energy']
            energy = np.asarray(energy)
            row = dict(variant=variant, sigma=sigma,
                       error=sigma/float(cylinder_cross_sections(RADIUS,12.,1.,'TE')[1])-1,
                       decay=float(res.diagnostics['decay']['box']),
                       absorption_cross_section=absorption,
                       contours=[scattered_flux(res,reference,'TE',CONTOUR+n/cpw,p.grid)[0]/incident
                                 for n in (-4,4)],
                       final_energy_over_peak=float(energy[-1]/max(energy)),
                       inverse_material_enclosure=bounds, seconds=time.perf_counter()-started)
            out['rows'].append(row)
            save()
            print(json.dumps(row), flush=True)
        if hashes() != source:
            raise RuntimeError('MeasurementSourceChanged')
        out.update(status='completed_research_comparator_not_a_production_fix',
                   source_unchanged=True, warnings=list(dict.fromkeys(str(w.message) for w in notes)))
        save()


if __name__ == '__main__':
    main()
