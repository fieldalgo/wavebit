"""Translate physical declarations to the existing normalized Yee implementation."""
import dataclasses
import math

from wavebit.errors import ValidationError, UnsupportedFeatureError
from wavebit.units import Z0
from wavebit.materials._lower import lower_material
from wavebit._common.declarations import static_float
from . import _scene as old_scene
from . import _legacy_geometry as old_geometry
from . import _legacy_boundaries as old_boundaries
from . import _legacy_sources as old_sources
from . import _legacy_monitors as old_monitors
from . import geometry, boundaries, sources, monitors
from .problem import Problem
from .numerics import Numerics, UniformGrid, Duration, PML
from ._kernel.grid import cell_count


def _number(value, path):
    try:
        return static_float('fdtd', path, value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(str(exc), code='static_input', path=path) from exc


def _geometry(shape, reserved=None):
    if isinstance(shape, (geometry.Slab, geometry.Region, geometry.Plane)):
        return shape
    if not isinstance(shape, (geometry.Sphere, geometry.Box, geometry.Cylinder, geometry.Prism)):
        raise UnsupportedFeatureError('unsupported geometry', path='scene.structures.shape', actual=type(shape).__name__)
    fields = {f.name: getattr(shape, f.name) for f in dataclasses.fields(shape)}
    if 'angle' in fields:
        fields['angle'] = math.degrees(_number(fields['angle'], 'scene.structures.shape.angle'))
    fields['range'] = reserved
    return getattr(old_geometry, type(shape).__name__)(**fields)


def _pulse(value, units):
    if not isinstance(value, sources.GaussianPulse):
        raise ValidationError('expected GaussianPulse', path='sources.pulse')
    return old_sources.Band(wavelength=value.wavelength,
                           df=units.rate_to_internal(value.frequency_sigma_hz), cutoff=value.cutoff)


def _source(source, units):
    if not isinstance(source, (sources.PlaneWave, sources.Dipole, sources.TFSF)):
        raise UnsupportedFeatureError('unsupported source', path='sources', actual=type(source).__name__)
    kw = dict(name=source.name, band=_pulse(source.pulse, units),
              amplitude=source.current_moment if isinstance(source,sources.Dipole) else source.amplitude,
              phase=source.phase, delay=units.time_to_internal(source.delay))
    if isinstance(source, sources.PlaneWave):
        if not isinstance(source.plane, geometry.Plane):
            raise ValidationError('expected Plane', path='sources.'+source.name+'.plane')
        kw['xyz'[source.plane.axis]] = source.plane.position
        return old_sources.PlaneWave(**kw, direction=source.direction,
            theta=math.degrees(_number(source.theta, 'sources.'+source.name+'.theta')),
            phi=math.degrees(_number(source.phi, 'sources.'+source.name+'.phi')),
            polarization=source.polarization)
    if isinstance(source, sources.TFSF):
        return old_sources.TFSF(**kw, box=_geometry(source.box), direction=source.direction,
                               polarization=source.polarization)
    scale = (Z0 if source.kind == 'electric' else 1.) / units.length_metres ** 2
    kw['amplitude'] = source.current_moment * scale
    return old_sources.Dipole(**kw, center=source.center, component=source.component,
                             kind=source.kind, stamp=source.stamp)


def _scaled_initial(value, scale):
    if value is None:
        return None
    if callable(value):
        return lambda x, y, z: value(x, y, z) * scale
    if isinstance(value, (tuple, list)):
        return tuple(_scaled_initial(v, scale) for v in value)
    return value * scale


def _monitor(m, problem):
    if isinstance(m, monitors.Record):
        return m
    if isinstance(m, (monitors.Field, monitors.Flux)):
        kw = dict(name=m.name, wavelengths=m.wavelengths,
                  plane=None if m.plane is None else _geometry(m.plane),
                  box=None if m.box is None else _geometry(m.box))
        if isinstance(m, monitors.Field):
            kw['components'] = m.components
        return getattr(old_monitors, type(m).__name__)(**kw)
    if isinstance(m, monitors.FarField):
        return old_monitors.FarField(m.name, box=_geometry(m.box), wavelengths=m.wavelengths,
                                    propagation=m.propagation)
    if isinstance(m, monitors.Diffraction):
        choices = [s for s in problem.sources if s.name == m.source]
        if len(choices) != 1 or not isinstance(choices[0], sources.PlaneWave):
            raise ValidationError('diffraction source must name one PlaneWave', path='outputs.'+m.name+'.source')
        if len(problem.sources) != 1:
            raise UnsupportedFeatureError('normalized diffraction currently requires one PlaneWave', path='outputs.'+m.name)
        if not isinstance(m.plane, geometry.Plane):
            raise ValidationError('expected Plane', path='outputs.'+m.name+'.plane')
        return old_monitors.Diffraction(m.name, **{'xyz'[m.plane.axis]: m.plane.position},
            side=m.side, wavelengths=m.wavelengths, orders=m.orders,
            reference_plane=m.reference_plane, incident_plane=m.incident_plane)
    raise UnsupportedFeatureError('unsupported monitor', path='outputs.monitors', actual=type(m).__name__)


def lower_problem(problem, numerics):
    if not isinstance(problem, Problem) or not isinstance(numerics, Numerics):
        raise ValidationError('expected fdtd.Problem and fdtd.Numerics', path='problem')
    if not isinstance(numerics.grid, UniformGrid) or not isinstance(numerics.duration, Duration):
        raise ValidationError('expected UniformGrid and Duration', path='numerics')
    if not isinstance(numerics.absorber, PML):
        raise UnsupportedFeatureError('the supported absorber implementation is PML', path='numerics.absorber')
    scene, units = problem.scene, problem.units
    dx = _number(numerics.grid.spacing, 'numerics.grid.spacing')
    old_scene.Resolution(dx=dx)
    physical_n = tuple(cell_count(s, dx) for s in scene.domain.size)
    physical_size = tuple(n * dx for n in physical_n)
    physical_low = tuple(c-s/2 for c, s in zip(scene.domain.center, physical_size))
    physical_high = tuple(a+s for a, s in zip(physical_low, physical_size))
    pad = {}
    for axis in 'xyz':
        for sign in '-+':
            pad[axis+sign] = numerics.absorber.cells if isinstance(scene.boundaries.face(axis+sign), boundaries.Open) else 0
    # Legacy PML validates integral cells and explicit thin-layer opt-in.
    old_boundaries.PML(cells=numerics.absorber.cells, allow_thin=numerics.absorber.allow_thin)
    size = tuple(s+(pad[a+'-']+pad[a+'+'])*dx for a,s in zip('xyz', physical_size))
    center = tuple(c+(pad[a+'+']-pad[a+'-'])*dx/2 for a,c in zip('xyz', scene.domain.center))
    allocated_low = tuple(c-s/2 for c,s in zip(center,size))
    allocated_high = tuple(c+s/2 for c,s in zip(center,size))
    background = lower_material(scene.background, units=units)
    faces = {f: background for f, n in pad.items() if n}
    structures = []
    unknown_ranges = set(numerics.geometry_ranges)-{s.name for s in scene.structures}
    if unknown_ranges:
        raise ValidationError('range names an absent structure', path='numerics.geometry_ranges', actual=sorted(unknown_ranges))
    for item in scene.structures:
        material = lower_material(item.material, units=units)
        shape = _geometry(item.shape, numerics.geometry_ranges.get(item.name))
        if isinstance(shape, old_geometry.Slab):
            axis_name = shape.axis
            axis = 'xyz'.index(axis_name)
            bounds = list(getattr(shape, axis_name))
            for sign, index in (('-',0), ('+',1)):
                face = axis_name+sign
                edge = physical_low[axis] if index == 0 else physical_high[axis]
                if pad[face] and bounds[0] <= edge <= bounds[1]:
                    bounds[index] = allocated_low[axis] if index == 0 else allocated_high[axis]
                    faces[face] = material
            shape = shape.replace(**{axis_name:tuple(bounds)})
            # A slab cutting an orthogonal open face is heterogeneous along that face.
            if any(n for f,n in pad.items() if f[0] != axis_name):
                raise UnsupportedFeatureError('a slab crossing an orthogonal Open face has no homogeneous PML continuation', code='padding_continuation', path='scene.structures.'+item.name)
        elif any(pad.values()):
            # Supported bodies are fully contained in the physical region; their lowerer
            # subsequently applies the stricter Yee/interpolation halo clearance checks.
            if isinstance(shape, old_geometry.Region):
                lo, hi = shape.low, shape.high
            elif getattr(shape, 'range', None) is not None:
                lo, hi = zip(*shape.range)
            else:
                lo, hi = shape.low, shape.high
            for a, axis in enumerate('xyz'):
                if ((pad[axis+'-'] and _number(lo[a], 'scene.structures.'+item.name) <= physical_low[a]) or
                    (pad[axis+'+'] and _number(hi[a], 'scene.structures.'+item.name) >= physical_high[a])):
                    raise UnsupportedFeatureError('a finite body touches an Open face; its continuation into padding is unspecified', code='padding_continuation', path='scene.structures.'+item.name)
        structures.append(old_geometry.Structure(shape, material,
                          discretisation=numerics.discretisation, junction=numerics.junction))
    converted = {}
    for axis in 'xyz':
        pair=[]
        for sign in '-+':
            face=scene.boundaries.face(axis+sign)
            if isinstance(face,boundaries.Open):
                opts=dataclasses.asdict(numerics.absorber)
                pair.append(old_boundaries.PML(**opts, background=faces[axis+sign]))
            elif isinstance(face,boundaries.Bloch):
                pair.append(old_boundaries.Bloch(phase=face.phase,k=face.k))
            else:pair.append(face)
        # Share one wrapping declaration so traced phase equality never detaches a leaf.
        if (not isinstance(scene.boundaries.face(axis+'-'),boundaries.Open) and
                scene.boundaries.face(axis+'-') is scene.boundaries.face(axis+'+')):
            pair[1]=pair[0]
        converted[axis]=tuple(pair)
    duration=numerics.duration
    if duration.periods is not None:
        selected=[s for s in problem.sources if s.name==duration.source]
        if len(selected)!=1:
            raise ValidationError('duration source must name exactly one source',path='numerics.duration.source')
        wavelength=_number(selected[0].pulse.wavelength,'sources.'+duration.source+'.pulse.wavelength')
        run=old_scene.Run(time=_number(duration.periods,'numerics.duration.periods')*wavelength,courant=numerics.courant)
    elif duration.seconds is not None:
        run=old_scene.Run(time=units.time_to_internal(duration.seconds),courant=numerics.courant)
    else:run=old_scene.Run(steps=duration.steps,courant=numerics.courant)
    initial=None
    if problem.initial is not None:
        if not isinstance(problem.initial,sources.InitialField):
            raise ValidationError('expected InitialField',path='initial')
        initial=old_sources.InitialField(E=problem.initial.E,H=_scaled_initial(problem.initial.H,Z0),amplitude=problem.initial.amplitude)
    # Validate physical monitor positions before adding the implementation padding.
    outputs=tuple(_monitor(m,problem) for m in problem.outputs.monitors)
    for m in outputs:
        if isinstance(m,old_monitors.Record):
            if any(not(a<=p<=b) for a,p,b in zip(physical_low,m.at,physical_high)):
                raise ValidationError('monitor is outside the physical domain',path='outputs.'+m.name)
        else:old_scene._monitor_inside(m,physical_low,physical_high)
    for source in problem.sources:
        if isinstance(source,sources.PlaneWave):
            axis=source.plane.axis
            if not physical_low[axis]<=source.plane.position<=physical_high[axis]:
                raise ValidationError('source is outside the physical domain',path='sources.'+source.name+'.plane')
        elif isinstance(source,sources.Dipole):
            if any(not a<=p<=b for a,p,b in zip(physical_low,source.center,physical_high)):
                raise ValidationError('source is outside the physical domain',path='sources.'+source.name+'.center')
        elif isinstance(source,sources.TFSF):
            if any(a<low or b>high for a,b,low,high in zip(source.box.low,source.box.high,physical_low,physical_high)):
                raise ValidationError('source is outside the physical domain',path='sources.'+source.name+'.box')
    legacy=old_scene.Scene(domain=old_scene.Domain(size=size,center=center),
        boundaries=old_boundaries.Boundaries(**converted),resolution=old_scene.Resolution(dx=dx),run=run,
        background=background,structures=tuple(structures),sources=tuple(_source(s,units) for s in problem.sources),
        monitors=outputs,initial=initial)
    metadata=dict(requested_domain=dict(size=scene.domain.size,center=scene.domain.center),
                  physical_domain=dict(size=physical_size,center=scene.domain.center,low=physical_low),
                  allocated_domain=dict(size=size,center=center,low=allocated_low),padding_cells=pad,
                  continuation={f:'uniform boundary medium' for f in faces})
    return legacy,metadata
