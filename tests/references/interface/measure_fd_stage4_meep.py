"""Black-box Meep comparison from its public documented Simulation/Flux API.

No Meep implementation is read. Outputs are new measurements, not replacements
for any existing golden. Micrometres are the chosen length unit. The procedure
uses the public Mie tutorial (documentation, not implementation source):
https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/

Geometry fixtures match the wavebit sphere and cylinder studies. Source pulse
definitions and flux interpolation differ between solvers; compare each to Mie,
retain placement spread, and measure time/PML/monitor controls before claiming
an accuracy advantage. See docs/validation/fdtd/fd-mie-validation.md for the measured limits.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np


def scattering_efficiency(fluxes, incident_flux, contour, radius, dimension):
    """Normalize signed face fluxes to cross-section (2-D) or efficiency (3-D).

    Faces are ordered x-/x+, y-/y+, then z-/z+ in three dimensions.
    The incident flux is the positive-direction empty-run entry-face power.
    """
    pairs = np.asarray(fluxes, dtype=float).reshape(dimension, 2)
    outward = np.sum(pairs[:, 1] - pairs[:, 0])
    intensity = incident_flux / (2 * contour)**(dimension - 1)
    area = math.pi * radius**2 if dimension == 3 else 1.
    return float(outward / intensity / area)


def run_case(cpw, eps, dimension=2, pol='TE', offset=(0.,0.,0.),
             averaging=True, after=24., pml_cells=12, gap=.2, source=-.85,
             contour=None):
    """Run an empty cell and scatterer through the documented Meep API.

    The 2-D labels refer to axial electric (TM) or magnetic (TE) fields. In 3-D,
    the incident electric field is along x and propagation is along +z.
    Runtime is measured after the Gaussian source ends; it is not wavebit's
    total-period parameter. Retained final times make that difference visible.
    """
    import meep as mp
    mp.verbosity(0)
    radius=.15 if dimension==2 else .25
    contour=(.55 if dimension==2 else .7) if contour is None else contour
    inner=.4
    half=.75+gap
    size=2*(half+pml_cells/cpw)
    axes=(0,1) if dimension==2 else (0,1,2)
    prop=axes[-1]
    component=mp.Ez if dimension==2 and pol=='TM' else mp.Ex
    src_center=[0.,0.,0.]
    src_center[prop]=source
    src_size=[size if a in axes and a!=prop else 0. for a in range(3)]
    dim_size=mp.Vector3(size,size,size if dimension==3 else 0.)
    source_spec=mp.Source(mp.GaussianSource(1.,fwidth=.25 if dimension==2 else .1,
                                           is_integrated=True),component=component,
                          center=mp.Vector3(*src_center),size=mp.Vector3(*src_size))
    material=mp.Medium(epsilon=eps)
    centre=mp.Vector3(*(x/cpw for x in offset))
    body=(mp.Cylinder(radius,material=material,center=centre) if dimension==2 else
          mp.Sphere(radius,material=material,center=centre))
    def faces(sim,width):
        out=[]
        for axis in axes:
            for sign in (-1,1):
                c=[0.,0.,0.]; c[axis]=sign*width
                s=[2*width if a in axes and a!=axis else 0. for a in range(3)]
                m=sim.add_flux(1.,0.,1,mp.FluxRegion(center=mp.Vector3(*c),size=mp.Vector3(*s)))
                out.append((axis,sign,m))
        return out
    records=[]
    saved=None
    for empty in (True,False):
        sim=mp.Simulation(cell_size=dim_size,resolution=cpw,
                          boundary_layers=[mp.PML(pml_cells/cpw)],
                          sources=[source_spec],geometry=[] if empty else [body],
                          dimensions=dimension,k_point=mp.Vector3(),eps_averaging=averaging,
                          Courant=.5)
        scattering=faces(sim,contour)
        total=faces(sim,inner)
        if not empty:
            for (_,_,m),data in zip(scattering,saved):
                sim.load_minus_flux_data(m,data)
        sim.run(until_after_sources=after)
        flux=[float(mp.get_fluxes(m)[0]) for _,_,m in scattering]
        absorbed=-sum(sign*float(mp.get_fluxes(m)[0]) for _,sign,m in total)
        if empty:
            saved=[sim.get_flux_data(m) for _,_,m in scattering]
            idx=next(i for i,(a,s,m) in enumerate(scattering) if a==prop and s==-1)
            intensity=flux[idx]/(2*contour)**(dimension-1)
        records.append(dict(flux=flux,absorbed=absorbed,time=float(sim.meep_time())))
        sim.reset_meep()
    area=math.pi*radius**2 if dimension==3 else 1.
    qsca=scattering_efficiency(records[1]['flux'], records[0]['flux'][idx],
                              contour, radius, dimension)
    sigma=qsca*area
    return dict(solver='meep',version=mp.__version__,cpw=cpw,eps=eps,dimension=dimension,
                pol=pol,offset=offset,radius=radius,averaging=averaging,after=after,
                pml_cells=pml_cells,gap=gap,source=source,contour=contour,
                requested_cell_size=size,sigma=sigma,qsca=qsca,
                qabs=records[1]['absorbed']/intensity/area,incident=intensity,records=records)


if __name__=='__main__':
    # External exceptions must not print implementation-source excerpts.
    import sys
    sys.tracebacklimit = 0
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('--cells',default='20,28,40,56')
    p.add_argument('--eps',type=float,default=12.)
    p.add_argument('--dimension',type=int,choices=(2,3),default=2)
    p.add_argument('--pol',choices=('TM','TE'),default='TE')
    p.add_argument('--offset',default='0,0,0')
    p.add_argument('--no-averaging',action='store_true')
    p.add_argument('--after',type=float,default=24.)
    p.add_argument('--pml-cells',type=int,default=12)
    p.add_argument('--gap',type=float,default=.2)
    p.add_argument('--source',type=float,default=-.85)
    p.add_argument('--contour',type=float)
    a=p.parse_args()
    rows=[]
    for cpw in map(int,a.cells.split(',')):
        start=time.perf_counter()
        row=run_case(cpw,a.eps,a.dimension,a.pol,tuple(map(float,a.offset.split(','))),
                     not a.no_averaging,a.after,a.pml_cells,a.gap,a.source,a.contour)
        row['seconds']=time.perf_counter()-start
        rows.append(row)
        a.output.write_text(json.dumps(rows,indent=2))
        print('RESULT',json.dumps(row),flush=True)
