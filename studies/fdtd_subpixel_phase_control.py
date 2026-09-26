"""Measure source phase, Yee propagation and discrete extinction before changing C3.

The mixed surface flux follows by expanding the discrete Poynting product for
F_total = F_incident + F_scattered. It is a surface extinction diagnostic, not a
corrected continuum far-field amplitude. No fitted wave number is substituted.
See docs/fd-mie-validation.md for the derivation, measurements and limitations.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
import numpy as np
import wavebit.fd as fd
from studies.fdtd_subpixel_mie_sphere import sphere_scene,read_result
from studies.fdtd_subpixel_mie_transform import records
from wavebit.fd._post import box_flux,far_field
from wavebit.fd._runtime import runtime
from wavebit.fd._kernel.grid import OFFSETS

def host(v):return np.asarray(v.detach() if hasattr(v,'detach') else v)
def pair(v):return [float(np.real(v)),float(np.imag(v))]

def scene(cpw, empty=False, eps=1.1, periods=24., reflection=1e-8):
    # Existing C3 surfaces: .65/.85 leave two cells of source/layer clearance
    # at 20 cells/wavelength. The .9-long line stays inside the 1.1 source box;
    # its .1 transverse width samples multiple columns on that same grid.
    s=sphere_scene(cpw,eps,periods,reflection=reflection)
    extra=(fd.Field('line',box=fd.Region(size=(.1,.1,.9)),components=('Ex','Hy'),wavelengths=(1.,)),)
    extra+=tuple(fd.FarField(n,box=fd.Region(size=(2*h,)*3),wavelengths=(1.,)) for n,h in (('near',.65),('far',.85)))
    return s.replace(monitors=s.monitors+extra,structures=() if empty else s.structures)

def column_audit(res):
    """Independent raw-series transform for this +z, x-polarized vacuum fixture."""
    src=res.sources['t'];dt=res.plan.dt;dx=res.plan.dx
    omega=2*np.pi;Omega=2*np.sin(omega*dt/2)/dt
    kappa=2*np.arcsin(Omega*dx/2)/dx
    def transform(v,t):
        v,t=host(v),host(t)
        return dt*np.sum(v*np.exp(1j*omega*t)[:,None],axis=0)+1j/Omega*np.exp(1j*omega*(t[-1]+dt/2))*v[-1]
    E=transform(src.incident,src.t)[0];H=transform(src.incident_H,src.t_H)[1]
    delta=kappa*(src.plane_H-src.plane)
    forward=(H+np.exp(-1j*delta)*E)/(2*np.cos(delta))
    A0=forward*np.exp(-1j*kappa*src.plane)
    backward=(E-forward)*np.exp(1j*kappa*src.plane)/A0
    reported=complex(host(src.incident_amplitude((1.,),plane=0.))[0])
    return dict(A0=pair(A0),reported=pair(reported),backward_ratio=pair(backward),relative_gap=abs(A0/reported-1),
                kappa=kappa,dt=dt,dx=dx,intensity=float(host(src.intensity((1.,)))[0]),
                intensity_ratio=float(host(src.intensity((1.,)))[0])/(.5*abs(A0)**2),
                cosine=np.cos(kappa*dx/2))

def line_audit(res):
    src=column_audit(res);A=complex(*src['reported']);kappa=src['kappa']
    values=[];matrices=[];rows=[]
    for field in ('Ex','Hy'):
        F=host(res['line'].fields[field])[0]/A
        z=host(res['line'].coords[field][2])
        mean=F.mean(axis=(1,2));expected=np.exp(1j*kappa*z)
        backward=complex(*src['backward_ratio'])
        standing=expected+(1 if field=='Ex' else -1)*backward*expected.conj()
        rows.append(dict(field=field,z=z.tolist(),values=[pair(v) for v in mean],
            forward_only_gap=float(np.max(abs(F-expected[:,None,None]))),
            two_wave_gap=float(np.max(abs(F-standing[:,None,None]))),
            continuum_gap=float(np.max(abs(F-np.exp(2j*np.pi*z[:,None,None])))),
            fitted_k=float(np.polyfit(z,np.unwrap(np.angle(mean)),1)[0])))
        matrices.append(np.stack((expected,expected.conj()*(1 if field=='Ex' else -1)),axis=-1));values.append(mean)
    M=np.concatenate(matrices);v=np.concatenate(values)
    coeff=np.linalg.lstsq(M,v,rcond=None)[0]
    return dict(source=src,rows=rows,forward_fit=pair(coeff[0]),backward_fit=pair(coeff[1]),fit_residual=float(np.max(abs(M@coeff-v))))

def incident_records(grid,faces,kappa,backward=0j):
    """Unit forward and specified backward vacuum waves at actual Yee points.

    E_x = exp(i*kappa*z) + B*exp(-i*kappa*z),
    H_y = exp(i*kappa*z) - B*exp(-i*kappa*z), with Z0 = 1.
    kappa is obtained from the Yee update equations, not fitted to these data.
    """
    accs={}
    for name,_,layout in faces:
        acc={}
        for label,field,slab in layout.parts():
            axes=[grid.low[d]+(np.arange(*rr)+.5*OFFSETS[field][slab.c][d])*grid.dx for d,rr in enumerate((slab.x,slab.y,slab.z))]
            z,y,x=np.meshgrid(axes[2],axes[1],axes[0],indexing='ij')
            active=(field=='E' and slab.c==0) or (field=='H' and slab.c==1)
            phase=np.exp(1j*kappa*z)
            value=phase+(1 if field=='E' else -1)*backward*phase.conj()
            acc[label]=(value if active else np.zeros_like(z,dtype=complex))[None]
        accs[name]=acc
    return accs

def mixed_flux(scattered,incident,faces,dx):
    # Polarization identity of the bilinear Poynting flux, evaluated without
    # subtracting the much larger incident power from the total field.
    rt=runtime('numpy')
    one={f:{k:(s[k] if k.startswith('E') else incident[f][k]) for k in s} for f,s in scattered.items()}
    two={f:{k:(incident[f][k] if k.startswith('E') else s[k]) for k in s} for f,s in scattered.items()}
    return float(box_flux(rt,one,faces,dx)[0][0]+box_flux(rt,two,faces,dx)[0][0])

def surface_audit(res,empty,eps):
    """Compare physical fields and bilinear extinction on the three C3 surfaces.

    The E/H gains below are descriptive least-squares fits only. They are
    never applied to fields, radiation, power, or the pass/fail criteria.
    """
    source=column_audit(res);A=complex(*source['reported']);Ae=complex(host(empty.sources['t'].incident_amplitude((1.,),plane=0.))[0])
    rt=runtime('numpy');rows=[];arrays={}
    for name,half in (('near',.65),('ff',.75),('far',.85)):
        _,raw,faces,grid,_,_,_=res[name]._args
        emptyraw=empty[name]._args[1]
        data={face:{k:host(v)/A for k,v in part.items()} for face,part in raw.items()}
        blank={face:{k:host(v)/Ae for k,v in part.items()} for face,part in emptyraw.items()}
        sca={face:{k:v-blank[face][k] for k,v in part.items()} for face,part in data.items()}
        _,exactfaces,exact=records(round(1/grid.dx),eps,half)
        if faces!=exactfaces:
            raise ValueError('MieSurfaceLayoutMismatchRefused')
        for face,part in data.items():
            for key,v in part.items():
                arrays[name+'__'+face+'__'+key]=v
                arrays['empty__'+name+'__'+face+'__'+key]=blank[face][key]
        f=complex(far_field(rt,data,faces,grid,(1.,),1.,1.,0.,0.)[0][0])
        fp=complex(far_field(rt,sca,faces,grid,(1.,),1.,1.,0.,0.)[0][0])
        fe=complex(far_field(rt,exact,faces,grid,(1.,),1.,1.,0.,0.)[0][0])
        f0=complex(far_field(rt,blank,faces,grid,(1.,),1.,1.,0.,0.)[0][0])
        fit={}
        for field in ('E','H'):
            a=np.concatenate([sca[face][key].ravel() for face in sca for key in sca[face] if key.startswith(field)])
            b=np.concatenate([exact[face][key].ravel() for face in exact for key in exact[face] if key.startswith(field)])
            gain=np.vdot(b,a)/np.vdot(b,b)
            fit[field]=dict(gain=pair(gain),phase_deg=float(np.degrees(np.angle(gain))),relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),shape_residual=float(np.linalg.norm(a-gain*b)/np.linalg.norm(b)))
        incidence=incident_records(grid,faces,source['kappa'])
        power=float(box_flux(rt,sca,faces,grid.dx)[0][0])
        ext=-mixed_flux(sca,incidence,faces,grid.dx)
        standing=-mixed_flux(sca,incident_records(grid,faces,source['kappa'],complex(*source['backward_ratio'])),faces,grid.dx)
        cont=-mixed_flux(sca,incident_records(grid,faces,2*np.pi),faces,grid.dx)
        absorp=-float(host(res['abs'].flux)[0])/abs(A)**2
        rows.append(dict(name=name,half=half,forward=pair(f),paired_forward=pair(fp),exact_injected_forward=pair(fe),empty_forward=pair(f0),
            empty_field_max=max(float(np.max(abs(v))) for p in blank.values() for v in p.values()),
            fit=fit,power=power,absorption=absorp,grid_extinction=ext,two_wave_extinction=standing,continuum_extinction=cont,
            grid_closure=ext/(power+absorp)-1,two_wave_closure=standing/(power+absorp)-1,continuum_closure=cont/(power+absorp)-1))
    return rows,arrays


def main():
    import argparse
    import hashlib
    import json
    import time
    import warnings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cpw', type=int, default=20)
    parser.add_argument('--eps', type=float, default=1.1)
    # Existing settled weak-sphere duration and layer target, not new gates.
    parser.add_argument('--periods', type=float, default=24.)
    parser.add_argument('--reflection', type=float, default=1e-8)
    args = parser.parse_args()
    raw_path = args.output.with_suffix('.npz')
    if args.output.exists() or raw_path.exists():
        raise ValueError('DuplicateMeasurementRefused')
    watched = sorted((ROOT/'wavebit/fd').rglob('*.py'))+[
        Path(__file__), ROOT/'studies/fdtd_subpixel_mie_sphere.py',
        ROOT/'studies/fdtd_subpixel_mie_transform.py',
        ROOT/'tests/references/mie.py', ROOT/'tests/references/mie_fields.py']
    def hashes():
        return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    before = hashes()
    out = dict(status='running_phase_diagnostic', cpw=args.cpw, eps=args.eps,
               periods=args.periods, reflection=args.reflection, source_sha256=before)
    def save():
        args.output.write_text(json.dumps(out, indent=2)+'\n', encoding='utf8')
    save()
    try:
        with warnings.catch_warnings(record=True) as notes:
            start = time.perf_counter()
            empty = fd.run(scene(args.cpw, True, args.eps, args.periods, args.reflection), backend='jax')
            out.update(empty_seconds=time.perf_counter()-start, empty=line_audit(empty),
                empty_decay={k:float(v) for k,v in empty.diagnostics['decay'].items()})
            save()
            if hashes() != before:
                raise ValueError('MeasurementSourceChangedRefused')
            start = time.perf_counter()
            result = fd.run(scene(args.cpw, False, args.eps, args.periods, args.reflection), backend='jax')
            out.update(sphere_seconds=time.perf_counter()-start, sphere=read_result(result,args.eps),
                       sphere_source=column_audit(result))
            out['surfaces'], arrays = surface_audit(result, empty, args.eps)
        for label, run in (('empty', empty), ('sphere', result)):
            series = run.sources['t']
            for key in ('t','t_H','incident','incident_H'):
                arrays[label+'__source__'+key] = host(getattr(series,key))
            for key, val in run['line'].fields.items():
                arrays[label+'__line__'+key] = host(val)
        np.savez_compressed(raw_path, **arrays)
        if hashes() != before:
            raise ValueError('MeasurementSourceChangedRefused')
        out.update(status='completed_phase_diagnostic_not_C3_pass', source_unchanged=True,
            warnings=[str(n.message) for n in notes], raw_array_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
        save()
        print(json.dumps({k:out[k] for k in ('status','cpw','eps','empty_seconds','sphere_seconds','surfaces')},indent=2),flush=True)
    except Exception as error:
        out.update(status='stopped_for_review', failure=f'{type(error).__name__}: {error}', source_unchanged=hashes()==before)
        save()
        raise


if __name__ == '__main__':
    main()
