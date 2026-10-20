"""Explicit cell-centred Fourier operators for advanced RCWA research.

``assemble`` builds constitutive operators without solving scattering. ``solve``
consumes those operators with their recorded physical and numerical contract.
Replace operator arrays explicitly to study alternative factorizations. Inputs
are borrowed backend arrays; keep their buffers unchanged while in use.
"""
from dataclasses import field, fields, is_dataclass
from types import MappingProxyType

from .._common.declarations import spec
from .._common.backend import eager_under_torch_compile
from ..errors import ValidationError, UnsupportedFeatureError, IncompatibleProblemError, NumericalError
from ._engine.factorization import ConvMatrices
from ._engine.anisotropy import TensorConvMatrices
from ._engine.scattering import Port, ScatteringMatrix
from . import _adapt
from ._lifecycle import prepare, _normalize_errors, _runtime_checks


__all__ = ['Operators', 'ConvMatrices', 'TensorConvMatrices', 'Port', 'ScatteringMatrix',
           'assemble', 'solve']


def _concrete_key(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, _adapt.TabulatedMaterial):
        return (type(value).__qualname__, value.wavelength_unit, value._quantity,
                _concrete_key(value.wavelengths), _concrete_key(value.values),
                _concrete_key(value.mu))
    if is_dataclass(value):
        return (type(value).__qualname__, tuple((f.name, _concrete_key(getattr(value, f.name)))
                                              for f in fields(value)))
    if isinstance(value, (tuple, list)):
        return tuple(_concrete_key(v) for v in value)
    a = _adapt.host(value, 'operators.physical_contract')
    return (tuple(a.shape), tuple(a.reshape(-1).tolist()))


def _contract(problem):
    # Stack includes half-space responses, topology and double-negative mode
    # labels. Holding those inputs fixed avoids silently mixing a new physical
    # model with old constitutive matrices. Source amplitudes/polarization stay
    # live. Reassembly is required even for a thickness-only stack change.
    return (_concrete_key(problem.stack), _concrete_key(problem.excitation.wavelength),
            problem.units.length)


@spec
class Operators:
    """One epsilon/mu tuple per wavelength, using the centre-origin convention.

    Each tuple has five ConvMatrices or nine TensorConvMatrices slots. The
    first three are xx, yy, inverse zz; off-diagonal slots carry the engine's
    negative-sign convention. ``mu=None`` means the nonmagnetic arithmetic.
    This object stores no solved fields or scattering matrix.
    """
    epsilon: tuple
    mu: tuple
    wavelengths: object
    numerics: object
    execution: object
    metadata: object = field(default_factory=dict)
    _prepared: object = field(default=None, repr=False)
    _physical_contract: object = field(default=None, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'epsilon', tuple(self.epsilon))
        object.__setattr__(self, 'mu', tuple(self.mu))
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))


@eager_under_torch_compile
def assemble(problem, *, numerics, backend='numpy', device='cpu', precision='double'):
    """Build operators for a concrete physical problem, without solving modes.

    This admission/geometry operation is eager. Differentiable research can
    replace the returned operator arrays and pass them to ``solve``; a traced
    physical stack itself must use the ordinary Prepared.run lifecycle.
    """
    if not _adapt.single_plane(problem):
        raise UnsupportedFeatureError('Operator assembly requires one plane-wave declaration; coherent groups have different order windows.',
                                      code='operators.source_groups', path='excitation.sources')
    if numerics.detuning is not None:
        raise UnsupportedFeatureError('Preassembled operators target their exact wavelength. Omit detuning.',
                                      code='operators.detuning', path='numerics.detuning')
    prepared = prepare(problem, numerics=numerics, backend=backend, device=device, precision=precision)
    epsilon, mu = [], []
    with prepared.execution._scope(), _normalize_errors():
        solvers = prepared._solvers(problem)
        for solver, geometry in zip(solvers, prepared._geometry):
            epsilon.append(solver.eps_conv_matrices(geometry=geometry))
            mu.append(solver.mu_conv_matrices(geometry=geometry))
    return Operators(epsilon=tuple(epsilon), mu=tuple(mu),
                     wavelengths=tuple(_adapt.wavelengths(problem)),
                     numerics=numerics, execution=prepared.execution,
                     metadata={'fourier_origin': 'cell center (0,0)',
                               'slot_order': ('xx', 'yy', 'inverse_zz', 'xy', 'yx', 'xz', 'yz', 'zx', 'zy'),
                               'off_diagonal_sign': 'negative',
                               'length_units': problem.units.length,
                               'physical_contract': 'stack and vacuum wavelengths fixed; replace arrays explicitly for a different constitutive operator'},
                     _prepared=prepared, _physical_contract=_contract(problem))


@eager_under_torch_compile
def solve(problem, operators):
    """Solve explicit operators without rebuilding their constitutive matrices.

    Stack and wavelength declarations must match the assembly. Operator arrays
    may be replaced with differentiable arrays of the same shape and backend.
    Physical stack differentiation belongs to Prepared.run, which rebuilds all
    material dependencies. The result records the use of explicit operators.
    """
    from .results import pack
    if not isinstance(operators, Operators) or operators._prepared is None:
        raise ValidationError('Supply Operators returned by assemble (optionally with replaced arrays).',
                              code='operators.type', path='operators')
    if _adapt.has_tracer(problem.stack) or _adapt.has_tracer(problem.excitation.wavelength):
        raise UnsupportedFeatureError('The explicit-operator stack contract is concrete; differentiate operator arrays or use Prepared.run for live physical inputs.',
                                      code='operators.physical_transform', path='problem.stack')
    if _contract(problem) != operators._physical_contract:
        raise IncompatibleProblemError('The physical stack or wavelength differs from the assembled operator contract. Assemble it again.',
                                       code='operators.physical_contract', path='problem')
    prepared = operators._prepared
    if operators.numerics != prepared.numerics or operators.execution != prepared.execution:
        raise IncompatibleProblemError('The numerical/execution contract changed. Assemble operators again.',
                                       code='operators.execution', path='operators')
    with prepared.execution._scope(), _normalize_errors():
        solvers = prepared._solvers(problem)
        if len(operators.epsilon) != len(solvers) or len(operators.mu) != len(solvers):
            raise ValidationError('One operator tuple is required per wavelength.', code='operators.spectral_shape', path='operators')
        checks = _runtime_checks(problem, prepared.execution, prepared._partitions)
        solved = []
        for solver, epsilon, mu in zip(solvers, operators.epsilon, operators.mu):
            n = (2*solver.order_window[0][1]+1)*(2*solver.order_window[1][1]+1)
            expected = (len(solver.thickness), n, n)
            for name, slots in (('epsilon', epsilon), ('mu', mu)):
                if slots is None and name == 'mu':
                    continue
                if not isinstance(slots, (tuple, list)) or len(slots) not in (5, 9):
                    raise ValidationError('Operators need five or nine ordered slots.', code='operators.slots', path=f'operators.{name}')
                for i, slot in enumerate(slots):
                    if (slot is None and i < 3) or (slot is not None and tuple(slot.shape) != expected):
                        raise ValidationError('A convolution slot has an incompatible shape.', code='operators.shape',
                                              path=f'operators.{name}[{i}]', expected=expected,
                                              actual=None if slot is None else tuple(slot.shape))
            result = solver.solve_conv(*epsilon, mu=mu, eps_top=solver.top.epsilon,
                                       eps_bot=solver.bot.epsilon, mu_top=solver.top.mu_value,
                                       mu_bot=solver.bot.mu_value)
            solved.append((solver, result))
        result = pack(problem, operators.numerics, operators.execution, solved, checks, prepared.report)
        finite = result.validity.checks['numerical_finite']
        if not _adapt.has_tracer(finite) and not bool(_adapt.host(finite, 'result')):
            raise NumericalError('Explicit operators produced nonfinite observables.', code='rcwa.nonfinite', path='operators')
        return result.replace(provenance={**result.provenance, 'constitutive_operators': 'explicit cell-centered operators'})
