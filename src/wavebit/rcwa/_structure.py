"""Concrete material sampling for the physical Problem API."""
import numpy as np

from .._common.backend import eager_under_torch_compile
from .._common.execution import Execution
from ..errors import ValidationError, UnsupportedFeatureError
from ..materials import Material
from . import _adapt
from .problem import Problem, Excitation, Outputs
from .numerics import Numerics
from .sources import PlaneWave


@eager_under_torch_compile
def sample_structure(problem, *, layer=0, samples=512, wavelength_index=0,
                     quantity='epsilon', component=None, **removed):
    """Sample one layer's relative permittivity or permeability without a solve.

    ``problem`` is an :class:`Problem`; ``layer`` is its nonnegative bottom-to-top
    layer index. ``wavelength_index`` selects a vacuum wavelength from its excitation,
    in ``problem.units.length``. Dispersive materials are evaluated with those units.
    ``quantity`` is ``'epsilon'`` or ``'mu'``; tensor values require an xyz component
    pair such as ``component='xy'``. Scalar values do not take a component.

    Returns an owned complex NumPy array with axes ``(y, x)``. Row zero is the
    negative-y edge. Region layers use ``samples`` cell-centered samples along each lattice
    vector; sampled layers retain their original resolution. Later regions paint
    over earlier ones. Sampling reuses the engine's geometry renderer, including
    its curved-painting approximation and refusal of painted skewed cells. It
    does not compute Fourier operators, scattering, fields or derivatives.

    This is an eager inspection operation on concrete values, outside autodiff
    transforms. It samples the selected layer independently; a picture does not
    establish that the entire problem is supported by ``check`` or has converged.
    No incident source or half-space value participates in the picture. Missing
    permeability means scalar one. Use a separate call for each spectral sample.
    """
    if removed:
        from .._common.declarations import renamed_keyword_message
        gone = [k for k in removed if k == 'n']
        if gone:
            raise TypeError(renamed_keyword_message('sample_structure', gone, {'n': 'samples'}))
        raise TypeError(f'sample_structure() does not take {sorted(removed)}. It takes problem, '
                        f'layer=, samples=, wavelength_index=, quantity= and component=.')
    if not isinstance(problem, Problem):
        raise ValidationError('Supply an rcwa.Problem.', code='problem.type', path='problem')
    if type(layer) is not int or not 0 <= layer < len(problem.stack.layers):
        raise ValidationError('layer must index a finite layer from zero.',
                              code='structure.layer', path='layer')
    if type(samples) is not int or samples < 1:
        raise ValidationError('samples must be a positive integer.', code='structure.samples',
                              path='samples')
    if quantity not in ('epsilon', 'mu'):
        raise ValidationError("quantity must be 'epsilon' or 'mu'.",
                              code='structure.quantity', path='quantity')
    if _adapt.has_tracer(problem):
        raise UnsupportedFeatureError('Sample a concrete problem outside autodiff transforms.',
                                      code='structure.concrete', path='problem')
    wavelengths = _adapt.wavelengths(problem)
    if type(wavelength_index) is not int or not 0 <= wavelength_index < len(wavelengths):
        raise ValidationError('wavelength_index must index the excitation spectrum from zero.',
                              code='structure.wavelength', path='wavelength_index')
    wavelength = wavelengths[wavelength_index]
    selected = problem.stack.layers[layer]

    def material(model):
        response = _adapt.physical_material(model, wavelength, problem.units)
        value = response.epsilon if quantity == 'epsilon' else response.mu_value
        return Material(epsilon=_adapt.host(value, 'stack.layers.material'))

    if selected.epsilon is None:
        selected = selected.replace(background=material(selected.background), regions=tuple(
            region.replace(material=material(region.material)) for region in selected.regions))
    else:
        epsilon = _adapt.host(selected.epsilon, 'stack.layers.epsilon')
        if quantity == 'mu':
            # Tensor raster axes follow (y, x, 3, 3); scalar mu is still one,
            # not an all-ones 3x3 tensor.
            epsilon = (np.ones(epsilon.shape[:2]) if selected.mu is None
                       else _adapt.host(selected.mu, 'stack.layers.mu'))
        selected = selected.replace(epsilon=epsilon, mu=None)
    sampled = problem.replace(
        stack=problem.stack.replace(layers=(selected,), top=Material(1.), bottom=Material(1.)),
        excitation=Excitation(wavelength=wavelength, sources=(PlaneWave(),)), outputs=Outputs())
    _adapt.validate_values(sampled)
    execution = Execution()
    with execution._scope():
        solver = _adapt.build_solver(sampled, Numerics(), execution, wavelength)
        from ._plotting import slice_structure
        values = np.array(slice_structure(solver, layer=0, n=samples, component=component), copy=True)
    if not np.isfinite(values).all():
        raise ValidationError('Sampled material values must be finite.',
                              code='structure.nonfinite', path='stack.layers')
    return values
