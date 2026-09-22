"""Public FDTDX 0.6.2 cylinder and source, paired empty-field subtraction.

This is the released PyPI package, whose native Cylinder uses a binary mask.
It is a staircase control, not a smoothed-shape reference. No Meep-derived
projection is invoked or inspected. Latest web documentation describes
APIs absent from this release, so the interface uses the installed public API.
Period 50 and gate 45 match the wavebit high-contrast cylinder fixture; controls
and residual contour dependence are recorded in docs/fd-mie-validation.md.
"""
import argparse
import json
import importlib.metadata
from pathlib import Path
import time
import numpy as np
# Exact SI definition, not a fitted simulation constant; the fixture wavelength
# is one micrometre, matching the wavebit and Meep dimensionless length unit.
C=299792458.
WL=1e-6


def surface_power(fields,h,inset=0):
    """Outward power per axial length on four collocated, trapezoidal faces.

    FDTDX's exact_interpolation aligns E and H in space and averages H in time
    before its phasor detector. These samples therefore use a common lattice;
    applying wavebit's staggered sampling formula again would interpolate twice.
    """
    v=fields[:,inset:fields.shape[1]-inset,inset:fields.shape[2]-inset]
    e,H=v[:3],v[3:]
    sy=.5*np.real(e[2]*np.conj(H[0])-e[0]*np.conj(H[2]))
    sz=.5*np.real(e[0]*np.conj(H[1])-e[1]*np.conj(H[0]))
    return float(h*(np.trapezoid(sy[-1,:]-sy[0,:])+np.trapezoid(sz[:,-1]-sz[:,0])))


def run_case(cpw,eps=12.,pol='TE',period=50.,gate=45.):
    import jax
    import jax.numpy as jnp
    import fdtdx as f
    jax.config.update('jax_enable_x64',True)
    h=WL/cpw
    nz=2*int(np.ceil(1.4*cpw))+24
    ny=int(round(period*cpw))
    # The released plane source squeezes away a one-cell transverse dimension.
    # Two identical periodic x slices represent the same invariant cylinder.
    shape=(2,ny,nz)
    config=f.SimulationConfig(time=gate*WL/C,resolution=h,backend='cpu',dtype=jnp.float64)
    volume=f.SimulationVolume(name='volume',partial_grid_shape=shape)
    bc=f.BoundaryConfig(boundary_type_minx='periodic',boundary_type_maxx='periodic',
                        boundary_type_miny='periodic',boundary_type_maxy='periodic',
                        thickness_grid_minz=12,thickness_grid_maxz=12)
    boundaries,constraints=f.boundary_objects_from_config(bc,volume)
    wave=f.WaveCharacter(wavelength=WL)
    pulse=f.GaussianPulseProfile(center_wave=wave,spectral_width=f.WaveCharacter(frequency=.25*C/WL))
    src=f.UniformPlaneSource(name='source',partial_grid_shape=(2,ny,1),wave_character=wave,
                             temporal_profile=pulse,direction='-',normalize_by_energy=False,
                             fixed_E_polarization_vector=(0.,1.,0.) if pol=='TE' else (1.,0.,0.))
    source_k=nz//2+round(.9*cpw)
    constraints += [src.set_grid_coordinates((0,1,2),('-',)*3,(0,0,source_k))]
    m=round(.65*cpw)
    detector=f.PhasorDetector(name='box',partial_grid_shape=(2,2*m+1,2*m+1),
                              wave_characters=(wave,),dtype=jnp.complex128,
                              exact_interpolation=True,scaling_mode='continuous')
    constraints += [detector.set_grid_coordinates((0,1,2),('-',)*3,(0,ny//2-m,nz//2-m))]
    body=f.Cylinder(name='cylinder',radius=.15*WL,axis=0,partial_grid_shape=(2,None,None),
                     materials={'body':f.Material(permittivity=eps)},material_name='body')
    common=[volume,*boundaries.values(),src,detector]
    results=[]
    for empty in (True,False):
        objects,arrays,params,conf,info=f.place_objects(common if empty else common+[body],config,
                constraints if empty else constraints+[body.place_at_center(volume)],jax.random.PRNGKey(0))
        arrays,objects,_=f.apply_params(arrays,objects,params,jax.random.PRNGKey(0))
        state=jax.jit(lambda a: f.run_fdtd(a,objects,conf,jax.random.PRNGKey(0),show_progress=False))(arrays)
        final=state[1]
        fields=np.asarray(final.detector_states['box']['phasor'])[0,0,:,0]
        results.append(fields)
        print('leg',cpw,empty,fields.shape,flush=True)
    ref=results[0]
    incident=float(abs(np.mean(.5*np.real(ref[0,:,-1]*np.conj(ref[4,:,-1])-
                                          ref[1,:,-1]*np.conj(ref[3,:,-1])))))
    scat=results[1]-ref
    sigmas=[surface_power(scat,h,n)/incident/WL for n in (0,2,4)]
    return dict(solver='fdtdx',version=importlib.metadata.version('fdtdx'),cpw=cpw,eps=eps,pol=pol,
                period=period,gate=gate,shape=shape,sigma=sigmas[0],contours=sigmas,
                abs_sigma=-surface_power(results[1],h)/incident/WL,incident=incident,
                geometry='public Cylinder, binary voxel mask, no subpixel projection')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('--cells',default='20,28,40,56')
    p.add_argument('--eps',type=float,default=12.)
    p.add_argument('--pol',choices=('TM','TE'),default='TE')
    p.add_argument('--period',type=float,default=50.)
    p.add_argument('--gate',type=float,default=45.)
    a=p.parse_args()
    rows=[]
    for cpw in map(int,a.cells.split(',')):
        start=time.perf_counter()
        row=run_case(cpw,a.eps,a.pol,a.period,a.gate)
        row['seconds']=time.perf_counter()-start
        rows.append(row)
        a.output.write_text(json.dumps(rows,indent=2))
        print('RESULT',json.dumps(row),flush=True)
