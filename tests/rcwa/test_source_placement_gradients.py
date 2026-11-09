"""Derivatives with respect to where a source sits and what it emits.

``ElectricDipole.position`` and ``.moment`` and ``GaussianBeam.focus`` are compared, component by
component, with central differences of the same scalar computed by the NumPy backend, in JAX and
in PyTorch. Validation of those fields was a NumPy conversion, so a grad tensor raised
``RuntimeError: Can't call numpy() on Tensor that requires grad`` and a JAX tracer could not be
converted at all. The scalar mixes outgoing power, a boundary field and interior fields, so
every output a placement moves is in it. Two steps estimate the finite-difference error, which is
asserted to sit far below the comparison tolerance.
"""

import numpy as np
import pytest

from wavebit._common.backend import _enable_x64
from wavebit.rcwa._engine.sources import ElectricDipole, GaussianBeam
from wavebit.rcwa._factory import rcwa as engine


BOX = ((-3., 3.), (-3., 3.))
#: The dipoles sit in layer 0, z in (0, 0.43); layer 1, z in (0.43, 0.78), is patterned.
PATTERNED = [dict(background=4., shapes=[]),
             dict(background=1.7, shapes=[dict(type='rectangle', cx=.2, cy=-.1, lx=1., ly=.8,
                                               material=6.)])]
#: The public API reports watts (about 1e-14 here); bring them to the fields' order.
WATTS = 1e13


def solve(backend, excitation, region):
    return engine(backend=backend, wavelength=1., period=(2.3, 2.7), top=1., bot=2.25,
                  structure=PATTERNED, thickness=[.43, .35], truncation_order=(1, 1),
                  answers='field', excitation=excitation, length_unit=1e-6).solve(
                      power_region=region)


def observed(result):
    """Outgoing power on both faces, a boundary field and interior fields in both layers."""
    face = result.boundary_fields('top', [.13], [-.07], part='outgoing')[0, 0]
    inner = result.reconstruct_field([.13, -.4], [-.07], [.05, .35, .6])
    # reconstruct_field indexes (z, y, x, component).
    return (face[0].real + face[1].imag + inner[0, 0, 0, 0].real + inner[1, 0, 1, 1].real
            + inner[2, 0, 0, 2].imag + inner[2, 0, 1, 0].real
            + result.power.outgoing_top + result.power.outgoing_bottom)


def central_difference(scalar, x0, step):
    """Per-component central difference and an error estimate from the doubled step."""
    fine, error = np.zeros(len(x0)), np.zeros(len(x0))
    for i in range(len(x0)):
        e = np.zeros(len(x0))
        e[i] = step
        f = (scalar(x0+e) - scalar(x0-e))/(2*step)
        c = (scalar(x0+2*e) - scalar(x0-2*e))/(4*step)
        fine[i], error[i] = f, abs(f-c)
    return fine, error


def gradient(backend, scalar, x0):
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        _enable_x64()
        return np.asarray(jax.grad(lambda x: scalar('jax', x))(jnp.asarray(x0)))
    torch = pytest.importorskip('torch')
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    value, = torch.autograd.grad(scalar('torch', x), x)
    return value.numpy()


def compare(backend, scalar, x0, step):
    reference, error = central_difference(lambda x: float(scalar('numpy', x)), np.asarray(x0), step)
    scale = np.max(np.abs(reference))
    assert np.all(error < 1e-7*scale), (reference, error)
    np.testing.assert_allclose(gradient(backend, scalar, x0), reference, rtol=1e-6, atol=1e-6*scale)


DIPOLES = {
    'periodic': dict(position=(.1, -.07, .2), moment=(1e-29, .2e-29j, .3e-29), region='cell'),
    'bloch_k': dict(position=(.1, -.07, .2), moment=(1e-29, .2e-29j, .3e-29), bloch_k=(.2, .1),
                    region='cell'),
    'isolated': dict(position=(-.2, .05, .3), moment=(.2e-29, 1e-29, .5e-29j),
                     repetition='isolated', sampling=(3, 3), region=BOX),
}


def dipole_at(kind):
    options = dict(DIPOLES[kind])
    region = options.pop('region')
    base = options.pop('position')

    def scalar(backend, shift):
        position = tuple(base[i] + shift[i] for i in range(3))
        return observed(solve(backend, ElectricDipole(position=position, **options), region))
    return scalar


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('kind', list(DIPOLES))
def test_dipole_position_derivative_matches_finite_differences(backend, kind):
    """x and y move the source phase exp(-i k.r); z moves its plane inside the host layer,
    which splits that layer into two segments whose thicknesses follow it."""
    compare(backend, dipole_at(kind), [0., 0., 0.], 1e-5)


def dipole_moment(backend, parts):
    """The moment as base + (re + i im), six real parameters."""
    options = dict(DIPOLES['bloch_k'])
    region = options.pop('region')
    base = options.pop('moment')
    moment = tuple(base[i] + 1e-29*(parts[i] + 1j*parts[3+i]) for i in range(3))
    return observed(solve(backend, ElectricDipole(moment=moment, **options), region))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_dipole_moment_derivative_matches_finite_differences(backend):
    """The scalar has a linear part (fields) and a quadratic one (power) in the moment."""
    compare(backend, dipole_moment, [0.]*6, 1e-4)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_complex_moment_gradient_follows_each_backend_convention(backend):
    """For a real scalar L of a complex leaf p = a + ib, PyTorch's gradient is dL/da + i dL/db
    and JAX's is its conjugate, dL/da - i dL/db; both are checked against the real parts."""
    parts, _ = central_difference(lambda x: float(dipole_moment('numpy', x)), np.zeros(6), 1e-4)
    expected = (parts[:3] + 1j*parts[3:])/1e-29
    options = dict(DIPOLES['bloch_k'])
    region = options.pop('region')
    base = np.asarray(options.pop('moment'))

    def scalar(backend, moment):
        return observed(solve(backend, ElectricDipole(moment=moment, **options), region))
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        _enable_x64()
        found = np.conj(np.asarray(jax.grad(lambda p: scalar('jax', p))(jnp.asarray(base))))
    else:
        torch = pytest.importorskip('torch')
        p = torch.tensor(base, dtype=torch.complex128, requires_grad=True)
        found, = torch.autograd.grad(scalar('torch', p), p)
        found = found.numpy()
    np.testing.assert_allclose(found, expected, rtol=1e-6, atol=1e-6*np.max(np.abs(expected)))


BEAMS = {
    'top': dict(side='top', waist=1.5, focus=(.1, -.2, .9), sampling=(3, 3), theta=.2, phi=.3,
                polarization=(.6, .8j)),
    'bottom': dict(side='bottom', waist=1.7, focus=(-.15, .1, -.3), sampling=(3, 3), theta=.1,
                   phi=-.4, polarization='p'),
}


def beam_at(side):
    options = dict(BEAMS[side])
    base = options.pop('focus')

    def scalar(backend, shift):
        focus = tuple(base[i] + shift[i] for i in range(3))
        return observed(solve(backend, GaussianBeam(focus=focus, **options), BOX))
    return scalar


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('side', list(BEAMS))
def test_gaussian_beam_focus_derivative_matches_finite_differences(backend, side):
    """The focus enters only each component's phase, k.(face - focus)."""
    compare(backend, beam_at(side), [0., 0., 0.], 1e-5)


@pytest.mark.parametrize('kind', ['dipole', 'beam'])
def test_forward_mode_jacobian_matches_finite_differences(kind):
    """jax.jacfwd carries tangents through the same live placements."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _enable_x64()
    scalar = dipole_at('bloch_k') if kind == 'dipole' else beam_at('bottom')
    reference, _ = central_difference(lambda x: float(scalar('numpy', x)), np.zeros(3), 1e-5)
    found = np.asarray(jax.jacfwd(lambda x: scalar('jax', x))(jnp.zeros(3)))
    np.testing.assert_allclose(found, reference, rtol=1e-6, atol=1e-6*np.max(np.abs(reference)))


def public(backend, source):
    from wavebit import Material, Units, rcwa
    problem = rcwa.Problem(units=Units(length='um'),
        stack=rcwa.Stack(period=(2.3, 2.7), bottom=Material(2.25), layers=(
            rcwa.Layer(thickness=.43, background=Material(4.)),
            rcwa.Layer(thickness=.35, background=Material(1.7), regions=(rcwa.Region(
                geometry=rcwa.Rectangle(size=(1., .8), center=(.2, -.1)),
                material=Material(6.)),)))),
        excitation=rcwa.Excitation(wavelength=1., sources=(source,)),
        outputs=rcwa.Outputs(diffraction=False, fields=True, power_region=BOX))
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(1, 1)), backend=backend)
    field = result.boundary_fields(side='top', x=[.13], y=[-.07], part='outgoing').values
    return (field[0, 0, 0, 0].real + field[0, 0, 0, 1].imag
            + WATTS*(result.outputs['top'].outgoing[0] + result.outputs['bottom'].outgoing[0]))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('kind', ['dipole', 'beam'])
def test_public_declarations_carry_placement_derivatives(backend, kind):
    """The public sources hand their tuples to the engine leaf by leaf."""
    from wavebit import rcwa

    def scalar(backend, shift):
        if kind == 'dipole':
            source = rcwa.ElectricDipole(position=(.1+shift[0], -.07+shift[1], .2+shift[2]),
                                         moment=(1e-29, .2e-29j, .3e-29), repetition='isolated',
                                         sampling=(3, 3))
        else:
            source = rcwa.GaussianBeam(waist=1.5, focus=(.1+shift[0], -.2+shift[1], .9+shift[2]),
                                       sampling=(3, 3), theta=.2, polarization=(.6, .8j))
        return public(backend, source)
    compare(backend, scalar, [0., 0., 0.], 1e-5)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_one_array_position_is_differentiated_as_a_tuple_is(backend):
    """An engine dipole also takes position and moment as single length-3 arrays."""
    def scalar(backend, x):
        if backend == 'numpy':
            x = np.asarray(x)
        return observed(solve(backend, ElectricDipole(position=x, moment=x*1e-28 + 1e-29), 'cell'))
    compare(backend, scalar, [.1, -.07, .2], 1e-5)


# -- validation and refusals ---------------------------------------------------------------

def traced(backend, value, run):
    """Call ``run`` with ``value`` turned into a JAX grad tracer or a grad-tracking tensor."""
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        _enable_x64()
        return jax.grad(lambda x: run(x).real if hasattr(run(x), 'real') else 0.)(
            jnp.asarray(value))
    torch = pytest.importorskip('torch')
    return run(torch.tensor(value, requires_grad=True))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('name,value,message', [
    ('position', [.1, float('nan'), .2], 'position must contain 3 finite real components'),
    ('position', [.1, .2], 'position must contain 3 finite real components'),
    ('focus', [.1, float('inf'), .2], 'focus must contain 3 finite real components'),
    ('moment', [1., float('nan'), 0.], 'moment must contain 3 finite components'),
])
def test_bad_values_carrying_a_derivative_are_refused_with_the_concrete_message(
        backend, name, value, message):
    """The value is read through the tensor's detached copy or the tracer's primal."""
    def build(x):
        if name == 'focus':
            GaussianBeam(focus=x)
        elif name == 'position':
            ElectricDipole(position=x, moment=(1., 0., 0.))
        else:
            ElectricDipole(position=(0., 0., .1), moment=x)
        return 0.
    with pytest.raises(ValueError, match=message):
        traced(backend, value, build)
    with pytest.raises(ValueError, match=message):   # the same words as a concrete value
        build(np.asarray(value))


def test_under_jit_a_placement_is_checked_by_shape_and_dtype_only():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _enable_x64()

    @jax.jit
    def build(x):
        ElectricDipole(position=(x[0], x[1], x[2]), moment=(x[0], 1j*x[1], x[2]))
        GaussianBeam(focus=x, waist=x[2]+1.)
        return x.sum()
    assert float(build(jnp.asarray([.1, .2, .3]))) == pytest.approx(.6)
    for bad in (lambda x: ElectricDipole(position=x[:2], moment=(1., 0., 0.)),
                lambda x: ElectricDipole(position=x*1j, moment=(1., 0., 0.)),
                lambda x: GaussianBeam(focus=(x[0], x[1]))):
        with pytest.raises(ValueError, match='must contain'):
            jax.jit(lambda x, bad=bad: bad(x) and x)(jnp.asarray([.1, .2, .3]))


@pytest.mark.parametrize('waist', ['torch', 'numpy'])
def test_a_zero_dimensional_waist_is_the_number_it_holds(waist):
    """``GaussianBeam(waist=torch.tensor(1.5))`` was refused as not a positive real number."""
    if waist == 'torch':
        torch = pytest.importorskip('torch')
        value = torch.tensor(1.5, dtype=torch.float64)
    else:
        value = np.array(1.5)
    options = dict(BEAMS['top']) | dict(waist=value)
    reference = observed(solve('numpy', GaussianBeam(**dict(BEAMS['top'])), BOX))
    assert observed(solve('numpy', GaussianBeam(**options), BOX)) == reference
    with pytest.raises(ValueError, match='waist must be a positive finite real number'):
        GaussianBeam(waist=value*-1)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_a_waist_derivative_is_refused_by_name(backend):
    """The waist scales the quadrature directions, which are host constants: a derivative
    with respect to it would be partial, as one with respect to the direction would."""
    from wavebit import Material, Units, rcwa
    from wavebit.errors import UnsupportedFeatureError

    def power(backend, waist):
        problem = rcwa.Problem(units=Units(length='um'),
            stack=rcwa.Stack(period=(2.3, 2.7), bottom=Material(2.25),
                             layers=(rcwa.Layer(thickness=.43, background=Material(4.)),)),
            excitation=rcwa.Excitation(wavelength=1., sources=(rcwa.GaussianBeam(
                waist=waist, sampling=(3, 3), theta=.2),)),
            outputs=rcwa.Outputs(diffraction=False, power_region=BOX))
        result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(1, 1)), backend=backend)
        return result.outputs['bottom'].outgoing[0]
    with pytest.raises(UnsupportedFeatureError, match='with respect to its waist is not') as caught:
        gradient(backend, power, 1.5)
    assert caught.value.code == 'rcwa.unsupported'


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_a_differentiated_dipole_on_an_interface_is_refused_as_a_concrete_one_is(backend):
    """The host layer is a discrete choice: a dipole on an interface has no layer, and no
    one-sided derivative is invented for it."""
    def run(backend, z):
        return observed(solve(backend, ElectricDipole(position=(.1, -.07, z),
                                                      moment=(1e-29, 0., 0.)), 'cell'))
    message = 'strictly inside a layer, away from its interfaces'
    with pytest.raises(ValueError, match=message):
        run('numpy', .43)
    with pytest.raises(ValueError, match=message):
        gradient(backend, lambda b, z: run(b, z), .43)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_two_dipoles_on_one_plane_refuse_a_derivative_of_either_height(backend):
    from wavebit.errors import UnsupportedFeatureError
    from wavebit import Material, Units, rcwa

    def run(backend, z):
        problem = rcwa.Problem(units=Units(length='um'),
            stack=rcwa.Stack(period=(2.3, 2.7), bottom=Material(2.25),
                             layers=(rcwa.Layer(thickness=.43, background=Material(4.)),)),
            excitation=rcwa.Excitation(wavelength=1., sources=(
                rcwa.ElectricDipole(position=(.1, -.07, z), moment=(1e-29, 0., 0.)),
                rcwa.ElectricDipole(position=(-.3, .2, .2), moment=(0., 1e-29, 0.)))),
            outputs=rcwa.Outputs(diffraction=False, power_region='cell'))
        result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(1, 1)), backend=backend)
        return result.outputs['top'].outgoing[0]
    with pytest.raises(UnsupportedFeatureError, match='share the source plane') as caught:
        gradient(backend, run, .2)
    assert caught.value.code == 'rcwa.unsupported'


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_a_bloch_k_derivative_is_refused_by_name(backend):
    from wavebit.errors import UnsupportedFeatureError
    from wavebit import Material, Units, rcwa

    def run(backend, k):
        problem = rcwa.Problem(units=Units(length='um'),
            stack=rcwa.Stack(period=(2.3, 2.7), bottom=Material(2.25),
                             layers=(rcwa.Layer(thickness=.43, background=Material(4.)),)),
            excitation=rcwa.Excitation(wavelength=1., sources=(rcwa.ElectricDipole(
                position=(.1, -.07, .2), moment=(1e-29, 0., 0.), bloch_k=(k, .1)),)),
            outputs=rcwa.Outputs(diffraction=False, power_region='cell'))
        result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(1, 1)), backend=backend)
        return result.outputs['top'].outgoing[0]
    with pytest.raises(UnsupportedFeatureError, match='with respect to bloch_k') as caught:
        gradient(backend, run, .2)
    assert caught.value.code == 'rcwa.unsupported'
