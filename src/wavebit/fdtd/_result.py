"""Owned observations with SI field/time units and explicit Fourier normalization."""
from wavebit._common.backend import array_namespace, eager_under_torch_compile
from dataclasses import dataclass, field
import math
import numpy as np
from wavebit._common.declarations import Declaration, renamed_attributes, spec
from wavebit._common.execution import frozen_mapping
from wavebit._common.pytrees import register_dataclass
from wavebit.observations import (TimeSeries, FieldSamples, RawFieldSpectrum,
                                 DiffractionOrders, RawDiffractionSpectrum, Validity)
from wavebit.observations.fields import _component_units
from wavebit.units import C0, Z0
from . import _monitors as monitors, _boundaries as boundaries


#: SpectralFlux fields that were renamed (fieldalgo/wavebit#57), and their replacements.
_FLUX_RENAMED = {'frequencies': 'frequencies_hz', 'energy_density': 'spectral_energy'}


@spec(renamed=_FLUX_RENAMED)
class SpectralFlux(Declaration):
    """The two-sided pulse spectral energy through a flux monitor, per wavelength.

    ``spectral_energy`` is in ``units`` (J/Hz, or per metre of each invariant axis);
    ``frequencies_hz`` are the frequencies of ``wavelengths``, in Hz.
    """
    wavelengths: object
    frequencies_hz: object
    spectral_energy: object
    units: str
    faces: object = None
    metadata: object = field(default_factory=dict)

    def __post_init__(self):
        register_dataclass(type(self))
        object.__setattr__(self,'metadata',frozen_mapping(self.metadata))

    __getattr__ = renamed_attributes(_FLUX_RENAMED, prefix='fdtd.')


#: Result fields that were renamed, and their replacements.
_RESULT_RENAMED = {'sources': 'source_references'}


@spec(renamed=_RESULT_RENAMED)
class Result(Declaration):
    """The outputs of one FDTD run by monitor name, with validity, diagnostics and provenance.

    ``source_references`` maps each source name to its measured reference (the incident or
    drive time series the run injected), in SI units.
    """
    outputs: object
    validity: Validity
    diagnostics: object
    provenance: object
    source_references: object = field(default_factory=dict)
    final_fields: object = None

    def __post_init__(self):
        register_dataclass(type(self))
        for name in ('outputs','diagnostics','provenance','source_references'):
            object.__setattr__(self,name,frozen_mapping(getattr(self,name)))

    def __getitem__(self,name):
        return self.outputs[name]

    __getattr__ = renamed_attributes(_RESULT_RENAMED, prefix='fdtd.')


def _invariant_axes(problem):
    """The names of the InvariantAxis axes, in x, y, z order; ``()`` when there are none."""
    return tuple(a for a in 'xyz' if isinstance(problem.scene.boundaries.face(a+'-'),boundaries.InvariantAxis))


def _measure(problem, raw):
    invariant=len(_invariant_axes(problem))
    exponent=2-invariant
    return (problem.units.length_unit_meters**exponent / (Z0*raw.plan.dx**invariant),
            'J/Hz' if invariant==0 else ('J/(Hz*m)' if invariant==1 else 'J/(Hz*m^2)'), invariant)


def _finite(values, xp):
    """Whether every array in ``values`` is finite; traced-safe, as ``runtime_stability`` is.

    ``values`` holds ``(array, usable)`` pairs: ``usable`` (or ``None`` for everywhere) marks the
    entries that must be finite, since a DiffractionOrders output publishes NaN on purpose at
    wavelengths its source cannot normalize.
    """
    finite=True
    for value,usable in values:
        ns=array_namespace(value)
        ok=ns.isfinite(value)
        if usable is not None:
            ok=ok|~usable
        ok=ns.all(ok)
        # Host arrays beside the run's backend (NumPy beside JAX or Torch) are concrete.
        finite=finite&(ok if ns is xp else bool(ok))
    return finite


def _port(normal):
    """The face label ('z+', 'x-', ...) of a plane with this outward normal."""
    axis=next(i for i,v in enumerate(normal) if v)
    return 'xyz'[axis]+('+' if normal[axis]>0 else '-')


def _coords(raw,component):
    return raw.plan.grid.yee_points(component[0], 'xyz'.index(component[1]))


@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class FarFieldSpectrum(Declaration):
    """Two-sided pulse spectral energy per solid angle; angles are radians."""
    _raw: object = field(repr=False,compare=False)
    _factor: float
    wavelengths: tuple
    units: str

    def intensity(self,theta,phi=0.):
        return self._raw.intensity(theta*180/math.pi,phi*180/math.pi)*self._factor

    def fields(self,theta,phi=0.):
        # This is the radiation coefficient r E_tilde, with geometric r in metres.
        e,p=self._raw.fields(theta*180/math.pi,phi*180/math.pi)
        scale=math.sqrt(self._factor*Z0/2)
        return e*scale,p*scale


#: ``Result.provenance['coordinate_frame']``.
_COORDINATE_FRAME=('Cartesian x, y, z in Problem.units, as the scene declares them; '
                   'Yee components at their staggered sites')


def adapt_result(problem,numerics,execution,raw,metadata):
    units=problem.units;ts=units.time_unit_seconds
    measure,flux_units,invariant=_measure(problem,raw)
    invariant_axes=_invariant_axes(problem)
    # Every monitor output and the final field, for Validity.checks['numerical_finite'].
    measured=[(raw.final.E,None),(raw.final.H,None)]
    spectral_factor=2*ts**2*measure
    outputs={}
    source_usable={}
    transform=dict(transform='integral exp(+i 2 pi f t) dt',window='rectangular finite run',
                   sidedness='two-sided',sampling='raw staggered Yee samples',
                   duration_seconds=raw.plan.steps*raw.plan.dt*ts)
    requests={m.name:m for m in problem.outputs.monitors}
    for name,m in requests.items():
        value=raw[name]
        if isinstance(m,monitors.TimeSeriesMonitor):
            scale=1/Z0 if m.component.startswith('H') else 1.
            observer=next(o for o in raw.plan.observers if o.name==name)
            axes=raw.plan.grid.yee_points(m.component[0], 'xyz'.index(m.component[1]))
            coordinates={'requested':m.at,'sample':tuple(axes[a][index] for a,index in enumerate((observer.i,observer.j,observer.k)))}
            for key in ('index','indices','at'):
                if hasattr(observer,key):coordinates['sample_'+key]=getattr(observer,key)
            measured.append((value.values,None))
            outputs[name]=TimeSeries(times_seconds=value.t*ts,coordinates=coordinates,values=value.values*scale,
                components=(m.component,),units=_component_units((m.component,),units.length),sampling='raw Yee component',
                metadata={'component_time_level':m.component[0]})
        elif isinstance(m,monitors.FieldMonitor):
            values={c:v*(ts/Z0 if c.startswith('H') else ts) for c,v in value.fields.items()}
            measured.extend((v,None) for v in value.fields.values())
            outputs[name]=RawFieldSpectrum(coordinates=value.coords,values=values,
                components=tuple(value.fields),
                units=_component_units(tuple(value.fields),units.length,electric='V*s/m',magnetic='A*s/m'),
                sampling='raw Yee components',wavelengths=value.wavelengths,
                frequencies_hz=np.asarray([C0/(w*units.length_unit_meters) for w in value.wavelengths]),
                window=transform['window'],metadata={'duration_seconds':transform['duration_seconds']})
        elif isinstance(m,monitors.FluxMonitor):
            measured.append((value.flux,None))
            measured.extend((v,None) for v in (value.faces or {}).values())
            outputs[name]=SpectralFlux(wavelengths=value.wavelengths,
                frequencies_hz=np.asarray([C0/(w*units.length_unit_meters) for w in value.wavelengths]),
                spectral_energy=value.flux*spectral_factor,units=flux_units,
                faces=None if value.faces is None else {f:v*spectral_factor for f,v in value.faces.items()},
                metadata=dict(transform,orientation='positive plane normal or outward closed surface',invariant_axes=invariant_axes))
        elif isinstance(m,monitors.FarFieldMonitor):
            outputs[name]=FarFieldSpectrum(_raw=value,_factor=spectral_factor,wavelengths=value.wavelengths,
                                           units=flux_units+'/sr')
        elif isinstance(m,monitors.DiffractionMonitor):
            reflected=m.wave=='reflected';side='r' if reflected else 't';order=value.orders.incident
            a=getattr(order,side+'_s');b=getattr(order,side+'_p')
            efficiency=getattr(order,'R' if reflected else 'T')
            source=raw.sources[m.source]
            rt=source.context.rt;xp=rt.xp
            incident=source.incident_amplitude(value.wavelengths,plane=value.incident_plane)*ts
            declaration=next(s for s in problem.sources if s.name==m.source)
            frequency=rt.array([C0/(w*units.length_unit_meters) for w in value.wavelengths],'float64')
            nominal=xp.exp(-.5*((frequency-C0/(declaration.pulse.wavelength*units.length_unit_meters))/
                                declaration.pulse.frequency_sigma_hz)**2)
            # Both a nominal source-band condition and a measured signal floor are
            # required. A zero-amplitude pulse cannot define scattering ratios.
            floor=100*np.finfo(float).eps*xp.max(xp.abs(source.incident))*transform['duration_seconds']
            usable=(nominal>=declaration.pulse.cutoff)&xp.isfinite(incident)&(xp.abs(incident)>floor)
            source_usable[name]=usable
            reference=dict(port=_port(value.raw['outward_normal']),wave=m.wave,
                plane=value.reference_plane,incident_plane=value.incident_plane,length_units=units.length,
                lateral_origin=value.lateral_origin,propagation='yee',polarization_frame='per-order s/p',
                source=m.source,normalization='measured incident reference',
                amplitude_units='1',invariant_axes=invariant_axes,
                epsilon=value.raw['epsilon'],mu=value.raw['mu'],
                outward_normal=value.raw['outward_normal'],tangent_axes=value.raw['tangent_axes'],
                incident_amplitude=incident,incident_amplitude_units='V*s/m',
                # The periodic cell's measure: an area, or per unit length of each invariant axis.
                cell_measure=value.raw['cell_area']*units.length_unit_meters**(2-invariant)/raw.plan.dx**invariant,
                cell_measure_units=('m^2','m','1')[invariant],wavevector_units='rad/'+units.length)
            common=dict(wavelengths=value.wavelengths,orders_u=tuple(value.orders.orders_u),
                        orders_v=tuple(value.orders.orders_v),k_normal=value.kz*next(v for v in value.raw['outward_normal'] if v),k_parallel=value.raw['k_parallel'],
                        reference_wavelength_fields=('epsilon','mu','incident_amplitude'),
                        metadata={'theta_radians':value.theta_at*math.pi/180,'spectral_axis_retained':True,
                            'source_usable':usable,'source_band_cutoff':declaration.pulse.cutoff,
                            'measured_reference_floor':floor})
            if m.raw:
                measured.extend((value.raw[k],None) for k in ('raw_s','raw_p','raw_flux'))
                reference.update(normalization='none (raw pulse transform)',amplitude_units='V*s/m',
                                 spectral_energy_units=flux_units)
                outputs[name]=RawDiffractionSpectrum(**common,amplitude_s=value.raw['raw_s']*ts,
                    amplitude_p=value.raw['raw_p']*ts,
                    spectral_energy=value.raw['raw_flux']*value.raw['cell_area']*spectral_factor,
                    reference=reference,window=transform['window'],amplitude_units='V*s/m')
            else:
                valid=usable[:,None,None]
                measured.extend((v,valid) for v in (a,b,efficiency))
                measured.append((getattr(order,side+'_xyz'),valid[...,None]))
                outputs[name]=DiffractionOrders(**common,amplitude_s=xp.where(valid,a,complex('nan')),
                    amplitude_p=xp.where(valid,b,complex('nan')),efficiency=xp.where(valid,efficiency,float('nan')),
                    amplitude_xyz=xp.where(valid[...,None],getattr(order,side+'_xyz'),complex('nan')),
                    reference=reference)
    source_references={}
    for name,s in raw.sources.items():
        if s.kind.startswith('Dipole'):
            magnetic=s.kind=='Dipole(H)'
            drive_scale=units.length_unit_meters**2/(1. if magnetic else Z0)
            # ``field_units`` names the field the ``field`` series samples, E or H.
            source_references[name]=dict(times_seconds=s.t*ts,drive=s.drive*drive_scale,
                drive_units='V*m' if magnetic else 'A*m',field=s.field/(Z0 if magnetic else 1.),
                field_units={'H':'A/m'} if magnetic else {'E':'V/m'},field_times_seconds=s.t_field*ts,
                delivered_power=s.work*units.length_unit_meters**2/Z0,power_units='W',time_units='s')
        else:
            source_references[name]=dict(times_seconds=s.t*ts,incident_E=s.incident,
                magnetic_times_seconds=None if s.t_H is None else s.t_H*ts,
                incident_H=None if s.incident_H is None else s.incident_H/Z0,
                plane_E=s.plane,plane_H=s.plane_H,field_units={'E':'V/m','H':'A/m'},time_units='s',
                normalization='measured Yee incident reference',length_units=units.length)
    final=None
    if problem.outputs.final_fields:
        # Coordinates are reported per component because Yee offsets differ.
        coordinates={c:_coords(raw,c) for c in ('Ex','Ey','Ez','Hx','Hy','Hz')}
        final=FieldSamples(coordinates=coordinates,values={'E':raw.final.E,'H':raw.final.H/Z0},
            components=('Ex','Ey','Ez','Hx','Hy','Hz'),
            units=_component_units(('Ex','Ey','Ez','Hx','Hy','Hz'),units.length),sampling='raw Yee components',
            metadata={'E_time_seconds':raw.plan.steps*raw.plan.dt*ts,
                      'H_time_seconds':(raw.plan.steps-.5)*raw.plan.dt*ts,'allocated_domain':metadata['allocated_domain']})
    diagnostics=dict(raw.diagnostics)
    if 'stopping' in diagnostics:
        stopping = dict(diagnostics['stopping'])
        stopping['window_seconds'] = stopping.pop('window') * ts
        stopping['eligible_time_seconds'] = stopping.pop('eligible_time') * ts
        stopping['state_amplitude_units'] = 'internal normalized Yee state per group; ratios are dimensionless'
        diagnostics['stopping'] = stopping
    validity=Validity(checks={'admitted':True,'runtime_stability':diagnostics.get('stable'),
                              'numerical_finite':_finite(measured,array_namespace(raw.final.E)),
                              'source_reference':source_usable},
                      convergence={'continuum_convergence':'not assessed','decay':diagnostics.get('decay')})
    provenance=dict(solver='fdtd',execution=execution,numerics=numerics,units=units,
        coordinate_frame=_COORDINATE_FRAME,time_convention='exp(-i*omega*t)',
        actual_steps=raw.plan.steps,time_step_seconds=raw.plan.dt*ts,source_normalization={s.name:('current moment in A*m' if s.kind=='electric' else 'equivalent magnetic current moment in V*m')
            if hasattr(s,'current_moment') else 'electric field amplitude in V/m' for s in problem.sources},**metadata)
    return Result(outputs=outputs,validity=validity,diagnostics=diagnostics,provenance=provenance,
                  source_references=source_references,final_fields=final)
