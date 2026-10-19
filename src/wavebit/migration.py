"""Explicit conversion of versioned historical RCWA jobs to physical declarations."""
from wavebit._common.backend import eager_under_torch_compile
from wavebit.errors import UnsupportedFeatureError, ValidationError
from wavebit.units import require_units


@eager_under_torch_compile
def migrate_legacy_jobs(document, *, units):
    """Convert a JobSpec v10 mapping/JSON into a tuple of current ``Job`` objects.

    The required Units states what the original lengths meant. All sweep points
    are expanded in their original Cartesian-product order. Layer order, exact
    Fourier/numerical choices, complex polarization and table interpolation are
    preserved. Conversion computes no fields and does not modify input files.
    Run the returned jobs and compare observations before replacing old records.
    This is a migration boundary, not an old mutable solver namespace.
    """
    import numpy as np
    from wavebit import rcwa
    from wavebit._common._jobs_v1 import JobSpec
    from wavebit._common.execution import Execution
    from wavebit.jobs import Job
    from wavebit.materials import Material, TabulatedMaterial
    from wavebit.materials._tabulated import TabulatedMaterial as OldTable
    from wavebit.lattice import as_lattice
    require_units(units)
    spec=JobSpec.from_json(document) if isinstance(document,str) else JobSpec.from_dict(document)

    def material(value):
        if isinstance(value,OldTable):
            make=TabulatedMaterial.from_refractive_index if value._quantity=='index' else TabulatedMaterial
            return make(value.wavelengths,value.values,mu=value.mu,wavelength_unit=units.length)
        return value if isinstance(value,Material) else Material(value)

    def region(shape):
        kind=shape.get('type','rectangle')
        allowed=({'type','vertices','material'} if kind=='polygon' else
                 {'type','cx','cy','lx','ly','l','angle','material'})
        unknown=set(shape)-allowed
        if unknown:
            raise ValidationError(f'unsupported shape fields: {sorted(unknown)}',path='stack.layers.shapes')
        from wavebit._common.geometry.recipes import check_shape_spec
        check_shape_spec(shape)
        center=(shape.get('cx',0.),shape.get('cy',0.))
        angle=shape.get('angle',0.)
        if kind in ('rectangle','square'):
            lx=shape.get('lx',shape.get('l',0.))
            ly=lx if kind=='square' else shape.get('ly',shape.get('l',0.))
            geometry=rcwa.Rectangle(size=(lx,ly),center=center,angle=angle)
        elif kind in ('circle','disk'):
            geometry=rcwa.Circle(radius=shape.get('lx',shape.get('l',0.))/2,center=center)
        elif kind=='ellipse':
            geometry=rcwa.Ellipse(radii=(shape.get('lx',shape.get('l',0.))/2,shape.get('ly',shape.get('l',0.))/2),center=center,angle=angle)
        elif kind=='polygon':
            geometry=rcwa.Polygon(vertices=shape['vertices'])
        else:
            raise UnsupportedFeatureError(f'cannot migrate shape {kind!r}',path='stack.layers.shapes')
        return rcwa.Region(geometry=geometry,material=material(shape['material']))

    result=[]
    for point in spec.expand():
        if point.solver!='rcwa':
            raise UnsupportedFeatureError('JobSpec migration accepts RCWA v10 jobs',path='solver')
        layers=[]
        for layer in point.stack.layers:
            structure=layer.structure
            if isinstance(structure,dict):
                if layer.mu is not None:
                    raise ValidationError('parametric permeability belongs to each region material',path='stack.layers.mu')
                unknown=set(structure)-{'background','shapes'}
                if unknown:
                    raise ValidationError(f'unsupported layer fields: {sorted(unknown)}',path='stack.layers')
                layers.append(rcwa.Layer(thickness=layer.thickness,background=material(structure.get('background',1.)),
                    regions=tuple(region(shape) for shape in structure.get('shapes',()))))
            else:
                layers.append(rcwa.Layer(thickness=layer.thickness,epsilon=structure,mu=layer.mu))
        old=point.incidence
        if old.polarization is not None:
            polarization=old.polarization
        elif old.tetm is not None:
            polarization='s' if old.tetm.name.lower()=='te' else 'p'
        elif old.psi is None:
            polarization='s'
        elif old.chi is None:
            polarization=(np.sin(old.psi),np.cos(old.psi))
        else:
            polarization=(np.sin(old.psi)*np.cos(old.chi)+1j*np.cos(old.psi)*np.sin(old.chi),
                          np.cos(old.psi)*np.cos(old.chi)-1j*np.sin(old.psi)*np.sin(old.chi))
        cell=dict(lattice=as_lattice(lattice=point.stack.lattice) if point.stack.lattice is not None
                  else as_lattice(period=point.stack.period))
        settings=point.numerics
        problem=rcwa.Problem(units=units,stack=rcwa.Stack(layers=tuple(layers),
            top=material(point.stack.top),bottom=material(point.stack.bot),**cell),
            excitation=rcwa.Excitation(wavelength=old.wavelength,sources=(rcwa.PlaneWave(
                side=old.incidence_side,polarization=polarization,theta=old.theta,phi=old.phi,kx=old.kx,ky=old.ky),)),
            outputs=rcwa.Outputs(fields=settings.answers.value=='field'))
        enum_names=('fourier_coefficients','factorization','raster_normal_method','nvf_projector','li_nesting')
        options={name:None if getattr(settings,name) is None else getattr(settings,name).value for name in enum_names}
        numerics=rcwa.Numerics(orders=tuple(settings.truncation_order),algorithm=settings.layer_algorithm.lower(),
            shape_nvf_grid_size=settings.shape_nvf_grid_size,**options)
        result.append(Job(solver='rcwa',problem=problem,numerics=numerics,
            execution=Execution(backend=settings.backend.value,precision='double' if settings.complex_dtype==128 else 'single')))
    return tuple(result)


__all__=['migrate_legacy_jobs']
