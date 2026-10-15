"""Lower immutable public declarations to the existing numerical engine.

Every call constructs fresh execution state. Prepared objects retain only static
geometry records, never a solver, material matrix, or result from a previous run.
"""
import dataclasses
from collections.abc import Mapping

import numpy as np

from .._common.backend import as_array
from ..materials import Material, TabulatedMaterial
from ..errors import ValidationError, UnsupportedFeatureError, IncompatibleProblemError
from .problem import Rectangle, Circle, Ellipse, Polygon
from .sources import PlaneWave


def leaves(value):
    if dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            if not f.name.startswith('_'):
                yield from leaves(getattr(value, f.name))
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from leaves(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from leaves(v)
    elif value is not None and not isinstance(value, (str, bool)):
        yield value


def abstract(value):
    """Structural detection; never obtains a primal or detaches a tangent."""
    import sys
    jax = sys.modules.get('jax')
    if jax is not None and isinstance(value, jax.core.Tracer):
        return True
    return False


def has_tracer(value):
    return any(abstract(v) for v in leaves(value))


def host(value, path):
    if has_tracer(value):
        raise UnsupportedFeatureError('This value makes a host-only geometry decision. Prepare a concrete problem outside the transform.',
                                      code='rcwa.host_geometry', path=path)
    # Reading values is only for validation; the original leaves are passed to
    # the engine untouched and keep their derivative graph. A sequence is read
    # leaf by leaf, since np.asarray cannot read a grad-tracking tensor in it.
    if isinstance(value, (list, tuple)):
        return np.asarray([host(v, path) for v in value])
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def wavelengths(problem):
    value = problem.excitation.wavelength
    shape = getattr(value, 'shape', None)
    if shape is None:
        if isinstance(value, (tuple, list)):
            if not value:
                raise ValidationError('At least one wavelength is required.', code='wavelength.empty', path='excitation.wavelength')
            return tuple(value)
        return (value,)
    if len(shape) == 0:
        return (value,)
    if len(shape) != 1 or shape[0] == 0:
        raise ValidationError('wavelength must be a scalar or a nonempty one-dimensional spectrum.',
                              code='wavelength.shape', path='excitation.wavelength')
    return tuple(value[i] for i in range(shape[0]))


def physical_material(model, wavelength, units, execution=None):
    if not hasattr(model, 'at'):
        raise ValidationError('Use a common Material model, not an untyped material number.',
                              code='material.type', path='stack.material')
    response = model.at(wavelength, units=units)
    if execution is not None:
        xp = _namespace(execution)
        return Material(epsilon=_array(xp, response.epsilon, execution, True),
                        mu=None if response.mu is None else _array(xp, response.mu, execution, True))
    return Material(epsilon=response.epsilon, mu=response.mu)


def region_recipe(region, wavelength, units, execution):
    return dict(geometry_recipe(region.geometry),
                material=physical_material(region.material, wavelength, units, execution))


def geometry_recipe(g):
    """Numeric shape fields shared by the solver and its topology-only checks."""
    if isinstance(g, Rectangle):
        return dict(type='rectangle', cx=g.center[0], cy=g.center[1],
                    lx=g.size[0], ly=g.size[1], angle=g.angle)
    if isinstance(g, Circle):
        return dict(type='circle', cx=g.center[0], cy=g.center[1], lx=2*g.radius)
    if isinstance(g, Ellipse):
        return dict(type='ellipse', cx=g.center[0], cy=g.center[1],
                    lx=2*g.radii[0], ly=2*g.radii[1], angle=g.angle)
    if isinstance(g, Polygon):
        return dict(type='polygon', vertices=g.vertices)
    raise ValidationError('Unsupported typed region geometry.', code='shape.type', path='stack.layers.regions')


def _namespace(execution):
    if execution.backend == 'numpy':
        import array_api_compat.numpy as xp
    elif execution.backend == 'jax':
        import jax.numpy as xp
    else:
        import array_api_compat.torch as xp
    return xp


def _array(xp, value, execution, complex_value=False):
    dtype = getattr(xp, ('complex128' if execution.precision == 'double' else 'complex64')
                    if complex_value else ('float64' if execution.precision == 'double' else 'float32'))
    kwargs = {}
    if execution.backend == 'torch':
        kwargs['device'] = execution.device
    return as_array(xp, value, dtype=dtype, **kwargs)


def polarization(source, xp, execution):
    value = source.polarization
    if isinstance(value, str):
        value = (1., 0.) if value == 's' else (0., 1.)
    if isinstance(value, (list, tuple)):
        # torch.asarray copies a sequence of tensors into a fresh leaf, which
        # would sever a grad-tracking component; stack the components instead.
        pair = xp.stack([_array(xp, v, execution, complex_value=True) for v in value])
    else:
        pair = _array(xp, value, execution, complex_value=True)
    if pair.shape != (2,):
        raise ValidationError('Jones polarization must have exactly two components.',
                              code='source.polarization_shape', path='excitation.sources.polarization')
    norm = xp.sqrt(xp.sum(xp.abs(pair)**2))
    # Eager validation below rejects a zero norm; traced invalid values propagate
    # through the validity guard, rather than receiving a made-up polarization.
    return pair / norm


def single_plane(problem):
    return len(problem.excitation.sources) == 1 and type(problem.excitation.sources[0]) is PlaneWave


def build_solver(problem, numerics, execution, wavelength):
    from ._factory import rcwa as legacy_rcwa
    from ._engine import sources as legacy_sources
    xp = _namespace(execution)
    kwargs = numerics._engine_kwargs()
    device = execution.device
    if execution.backend == 'jax':
        import jax
        platform, _, index = device.partition(':')
        device = jax.devices(platform)[int(index or 0)]
    kwargs.update(backend=execution.backend, device=device,
                  complex_dtype=0 if execution.precision == 'double' else 1,
                  wavelength=wavelength,
                  top=physical_material(problem.stack.top, wavelength, problem.units, execution),
                  bot=physical_material(problem.stack.bottom, wavelength, problem.units, execution),
                  answers='field' if problem.outputs.fields else 'amplitude')
    kwargs['thickness'] = (xp.stack([_array(xp, layer.thickness, execution) for layer in problem.stack.layers])
                           if problem.stack.layers else _array(xp, [], execution))
    if problem.stack.lattice is not None:
        kwargs['lattice'] = problem.stack.lattice
    else:
        kwargs['period'] = problem.stack.period
    raster = [layer.epsilon is not None for layer in problem.stack.layers]
    if any(raster) and not all(raster):
        raise UnsupportedFeatureError('A stack cannot mix sampled layers and region layers. Represent homogeneous layers on the same raster, or use regions throughout.',
                                      code='rcwa.mixed_layer_encoding', path='stack.layers')
    if any(raster):
        kwargs['structure'] = xp.stack([_array(xp, l.epsilon, execution, True) for l in problem.stack.layers])
        if any(l.mu is not None for l in problem.stack.layers):
            kwargs['mu'] = xp.stack([xp.ones_like(_array(xp, l.epsilon, execution, True)) if l.mu is None
                                    else _array(xp, l.mu, execution, True) for l in problem.stack.layers])
    else:
        kwargs['structure'] = [dict(background=physical_material(l.background, wavelength, problem.units, execution),
                                    shapes=[region_recipe(r, wavelength, problem.units, execution) for r in l.regions])
                               for l in problem.stack.layers]
    if not problem.stack.layers:
        # A zero-width uniform layer lowers the interface to the existing engine
        # without adding a physical optical path or moving the z=0 reference.
        kwargs['structure'] = [dict(background=physical_material(problem.stack.top, wavelength, problem.units, execution), shapes=[])]
        kwargs['thickness'] = _array(xp, [0.], execution)
    if single_plane(problem):
        source = problem.excitation.sources[0]
        kwargs.update(incidence_side=source.side, polarization=polarization(source, xp, execution))
        kwargs.update({k: getattr(source, k) for k in ('theta', 'phi', 'kx', 'ky') if getattr(source, k) is not None})
    else:
        adapted = []
        for source in problem.excitation.sources:
            values = {f.name: getattr(source, f.name) for f in dataclasses.fields(source)}
            if 'polarization' in values:
                values['polarization'] = polarization(source, xp, execution)
            if 'orders' in values:
                values['orders'] = dict(values['orders'])
            adapted.append(getattr(legacy_sources, type(source).__name__)(**values))
        kwargs.update(excitation=tuple(adapted), power_region=problem.outputs.power_region,
                      length_unit=problem.units.length_metres)
    if single_plane(problem):
        formulation = kwargs.pop('formulation', 'auto')
        solver = legacy_rcwa(**kwargs)
        solver.formulation = formulation
        return solver
    return legacy_rcwa(**kwargs)


def structure(value):
    """Only types, declared absence, counts, and array shapes; never live values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, TabulatedMaterial):
        return ('tabulated', value.wavelength_unit, value._quantity, structure(value.wavelengths),
                structure(value.values), structure(value.mu))
    if dataclasses.is_dataclass(value):
        return (type(value).__module__, type(value).__name__,
                tuple((f.name, structure(getattr(value, f.name))) for f in dataclasses.fields(value)
                      if f.init and not f.name.startswith('_')))
    if isinstance(value, Mapping):
        return tuple((k, structure(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(structure(v) for v in value)
    if hasattr(value, 'shape'):
        return ('numeric', tuple(value.shape))
    if isinstance(value, (int, float, complex, np.number)):
        return ('numeric', ())
    raise ValidationError(f'Unsupported problem leaf {type(value).__name__}.',
                          code='problem.leaf', path='problem')


def structure_key(problem):
    # Period, shape values, and thickness can vary eagerly if the original
    # topology record still validates. Units and source grouping stay fixed.
    key = structure(problem)
    order_labels = tuple(tuple(k for k, _ in s.orders) if hasattr(s, 'orders') else None
                         for s in problem.excitation.sources)
    sampling = tuple(tuple(s.sampling) if hasattr(s, 'sampling') else None
                     for s in problem.excitation.sources)
    return key, order_labels, sampling


def geometry_values(problem):
    values = {'stack.cell': problem.stack.lattice if problem.stack.lattice is not None else problem.stack.period}
    for i, layer in enumerate(problem.stack.layers):
        for j, region in enumerate(layer.regions):
            for f in dataclasses.fields(region.geometry):
                values[f'stack.layers[{i}].regions[{j}].geometry.{f.name}'] = getattr(region.geometry, f.name)
    return values


def primal(value):
    """``value`` with every JAX tracer replaced by the number it carries, for a check only.

    An eager ``jax.grad`` or ``jax.jacfwd`` tracer holds a concrete primal, as a PyTorch
    tensor does, and ``stop_gradient`` hands it over without touching the caller's value or
    its tangent. A tracer under ``jax.jit`` or ``jax.vmap``, alone or beneath a grad, has no
    number yet and the conversion raises (measured on jax 0.11.0).
    """
    if isinstance(value, (tuple, list)):
        return tuple(primal(v) for v in value)
    if not abstract(value):
        return value
    import jax
    return np.asarray(jax.lax.stop_gradient(value))


def carries_primal(value):
    """Whether :func:`primal` can read every tracer in ``value``."""
    import jax
    try:
        primal(value)
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        return False
    return True


def primal_geometry(problem):
    """``problem`` with the cell and every region geometry read by :func:`primal`.

    Everything else is kept as given, so a topology check on it sees what a concrete
    problem at the same design would show, and nothing it does reaches the traced run.
    """
    stack = problem.stack
    cell = ({'lattice': primal(stack.lattice)} if stack.lattice is not None
            else {'period': primal(stack.period)})
    layers = tuple(dataclasses.replace(layer, regions=tuple(
        dataclasses.replace(region, geometry=dataclasses.replace(region.geometry, **{
            f.name: primal(getattr(region.geometry, f.name)) for f in dataclasses.fields(region.geometry)}))
        for region in layer.regions)) for layer in stack.layers)
    return dataclasses.replace(problem, stack=dataclasses.replace(stack, layers=layers, **cell))


def check_structure(problem, expected):
    if structure_key(problem) != expected:
        raise IncompatibleProblemError('The problem declaration layout changed. Prepare it again (layer/source/model counts, tensor shape, magnetic presence, outputs, and spectral shape are structural).',
                                       code='prepared.structure', path='problem')


def coherent_dependencies(problem):
    """Concrete dependencies of the engine's source grouping and order window.

    Until a backend grouping guard exists, coherent replay fixes these values.
    Amplitudes, Jones weights, dipole moments and finite-layer materials remain
    live. Recording the full dependency values is conservative, not a claim
    that only the resulting group count determines compatibility.
    """
    if single_plane(problem):
        return None
    def exact(value):
        if value is None or isinstance(value, (str, bool)):
            return value
        if dataclasses.is_dataclass(value):
            return tuple((f.name, exact(getattr(value, f.name))) for f in dataclasses.fields(value))
        if isinstance(value, (list, tuple)):
            return tuple(exact(v) for v in value)
        a = host(value, 'excitation.source_grouping')
        return a.shape, tuple(a.reshape(-1).tolist())
    fields = ('theta', 'phi', 'kx', 'ky', 'waist', 'sampling', 'spectral_extent', 'bloch_k')
    return (exact(problem.stack.lattice if problem.stack.lattice is not None else problem.stack.period),
            exact(problem.excitation.wavelength),
            tuple(tuple((name, exact(getattr(source, name))) for name in fields if hasattr(source, name))
                  for source in problem.excitation.sources),
            tuple((exact(physical_material(model, w, problem.units).epsilon),
                   exact(physical_material(model, w, problem.units).mu))
                  for w in wavelengths(problem) for model in (problem.stack.top, problem.stack.bottom)))


def validate_values(problem):
    if problem.stack.period is not None and not has_tracer(problem.stack.period):
        period = host(problem.stack.period, 'stack.period')
        if period.shape != (2,) or not np.isrealobj(period) or not np.all(np.isfinite(period) & (period > 0)):
            raise ValidationError('period must contain two positive finite real lengths.',
                                  code='stack.period', path='stack.period')
    for index, wavelength in enumerate(wavelengths(problem)):
        if not has_tracer(wavelength):
            a = host(wavelength, 'excitation.wavelength')
            if a.shape or not np.isrealobj(a) or not np.isfinite(a) or a <= 0:
                raise ValidationError('Vacuum wavelengths must be positive finite real scalars.',
                                      code='wavelength.value', path=f'excitation.wavelength[{index}]')
    for i, layer in enumerate(problem.stack.layers):
        if not has_tracer(layer.thickness):
            a = host(layer.thickness, 'stack.layers.thickness')
            if a.shape or not np.isrealobj(a) or not np.isfinite(a) or a < 0:
                raise ValidationError('Layer thickness must be finite and nonnegative.',
                                      code='layer.thickness', path=f'stack.layers[{i}].thickness')
    for i, source in enumerate(problem.excitation.sources):
        if hasattr(source, 'polarization') and not isinstance(source.polarization, str) and not has_tracer(source.polarization):
            pair = host(source.polarization, 'excitation.sources.polarization')
            if pair.shape != (2,) or not np.all(np.isfinite(pair)) or np.sum(np.abs(pair)**2) <= 0:
                raise ValidationError('Jones polarization must be a finite, nonzero pair.',
                                      code='source.polarization_value', path=f'excitation.sources[{i}].polarization')
        if hasattr(source, 'amplitude') and not has_tracer(source.amplitude):
            a = host(source.amplitude, 'excitation.sources.amplitude')
            if a.shape or not np.isfinite(a) or (single_plane(problem) and abs(a) == 0):
                raise ValidationError('Source amplitude must be finite; a single incident plane wave must be nonzero.',
                                      code='source.amplitude', path=f'excitation.sources[{i}].amplitude')
