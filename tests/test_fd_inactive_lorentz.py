"""Analytic inactive-Lorentz identities and zero-primal differentiation.

These tests exercise public scalar material helpers only. No field solve, mesh,
new physical tolerance, or change to eps_inf differentiation is involved.
"""
import math

import numpy as np
import pytest

import wavebit.fd as fd


DOORS = ('term', 'pole', 'list', 'tuple', 'material', 'diagonal')
CASES = ('inactive-resonance', 'inactive-huge-omega0', 'inactive-huge-gamma',
         'active-off', 'active-near', 'active-damped-resonance', 'mixed-sum')
IDENTITIES = [(case, door, 'python') for case in CASES for door in DOORS]
IDENTITIES += [(case, 'term', carrier)
               for carrier in ('numpy-scalar', 'numpy-array', 'jax', 'torch',
                               'negative-zero', 'jax-captured-jit')
               for case in ('inactive-resonance', 'inactive-huge-omega0')]
IDENTITIES += [('small-positive', 'term', 'longdouble')]


def _number(value, carrier):
    if carrier == 'numpy-scalar':
        return np.float64(value)
    if carrier == 'numpy-array':
        return np.asarray(value, dtype=np.float64)
    if carrier in ('jax', 'jax-captured-jit'):
        jax = pytest.importorskip('jax')
        jnp = pytest.importorskip('jax.numpy')
        with jax.enable_x64(True):
            return jnp.asarray(value, dtype=jnp.float64)
    if carrier == 'torch':
        torch = pytest.importorskip('torch')
        return torch.tensor(value, dtype=torch.float64)
    return -0.0 if carrier == 'negative-zero' else value


def _read(door, poles, wavelength=1., omega=None, baseline=2.5):
    if omega is None:
        omega = 2*math.pi/wavelength
    if door == 'term':
        return sum(p.eps_term(omega) for p in poles)
    if door == 'pole':
        assert len(poles) == 1
        return fd.eps_at(poles[0], wavelength, baseline)
    if door == 'list':
        return fd.eps_at(list(poles), wavelength, baseline)
    if door == 'tuple':
        return fd.eps_at(tuple(poles), wavelength, baseline)
    background = (2., 3., 5.) if door == 'diagonal' else baseline
    return fd.eps_at(fd.Dispersive(eps_inf=background, poles=tuple(poles)), wavelength)


def _plain(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return complex(np.asarray(value))


@pytest.mark.parametrize('case,door,carrier', IDENTITIES,
                         ids=['-'.join(row) for row in IDENTITIES])
def test_inactive_lorentz_analytic_identity_and_active_controls(case, door, carrier):
    omega = 2*math.pi
    if case == 'small-positive':
        # Extended precision is platform-dependent. Keep a positive, representable
        # response on both branches without adding a platform skip.
        if np.finfo(np.longdouble).minexp < -1022:
            strength = np.ldexp(np.longdouble(1), -1075)
        else:
            strength = np.nextafter(np.float64(0), np.float64(1))
        pole = fd.Lorentz(omega_0=.75, gamma=0., strength=strength)
        try:
            value = pole.eps_term(.25)
        except ArithmeticError as error:
            value = None
            failure = repr(error)
        else:
            failure = None
        assert failure is None, failure
        assert value != 0, 'a positive representable susceptibility is not inactive'
        # Denominator is exactly 1/2. Doubling even the minimum subnormal
        # is representable, so this comparison needs no tolerance.
        assert value == 2*strength
        return

    strength, omega0, gamma = 0., omega, 0.
    if case == 'inactive-huge-omega0':
        omega0 = 1e200
    elif case == 'inactive-huge-gamma':
        gamma = 1e308
    elif case == 'active-off':
        strength, omega0, gamma = 2., 9., .7
    elif case == 'active-near':
        strength, omega0 = 2**-10, omega*(1+2**-8)
    elif case == 'active-damped-resonance':
        strength, gamma = 2., .7
    pole = fd.Lorentz(omega_0=omega0, gamma=gamma,
                      strength=_number(strength, carrier))
    # Independent rational formula is evaluated only for active finite controls.
    expected = (0j if strength == 0 else
                strength/complex(omega0*omega0-omega*omega, -gamma*omega))
    poles = (pole,)
    if case == 'mixed-sum':
        # The single-pole door still sees an inactive term; the sum doors also
        # retain independent active Lorentz and Debye contributions.
        if door != 'pole':
            poles += (fd.Lorentz(omega_0=9., gamma=.7, strength=2.),
                      fd.Debye(delta_eps=1.5, tau=.2))
            expected += 2/complex(81-omega*omega, -.7*omega)
            expected += 1.5/complex(1., -.2*omega)
    wanted = (expected if door == 'term' else
              tuple(x+expected for x in (2., 3., 5.)) if door == 'diagonal' else
              2.5+expected)
    try:
        if carrier in ('jax', 'jax-captured-jit'):
            jax = pytest.importorskip('jax')
            with jax.enable_x64(True):
                # Strength is a captured concrete JAX scalar. A Python Boolean
                # comparison staged inside jit must not be required for admission.
                actual = (jax.jit(lambda w: _read(door, poles, omega=w))(omega)
                          if carrier == 'jax-captured-jit' else _read(door, poles))
        else:
            actual = _read(door, poles)
    except ArithmeticError as error:
        actual, failure = None, repr(error)
    else:
        failure = None
    assert failure is None, failure
    if isinstance(wanted, tuple):
        assert isinstance(actual, tuple) and len(actual) == 3
        values = tuple(_plain(x) for x in actual)
        if expected == 0j:
            assert values == wanted
        else:
            assert values == pytest.approx(wanted, rel=3e-13, abs=3e-13)
    elif expected == 0j:
        assert _plain(actual) == wanted
    else:
        assert _plain(actual) == pytest.approx(wanted, rel=3e-13, abs=3e-13)


MODES = ('numpy-secant', 'jax-reverse', 'jax-jvp', 'torch-reverse',
         'torch-forward', 'torch-func-jvp')
PARAMETERS = ('strength', 'omega0', 'gamma', 'frequency')


@pytest.mark.parametrize('door', ('term', 'pole', 'list'))
@pytest.mark.parametrize('parameter', PARAMETERS)
@pytest.mark.parametrize('mode', MODES)
def test_zero_primal_lorentz_derivatives_are_preserved(mode, parameter, door):
    omega = 2*math.pi
    initial = {'strength': 0., 'omega0': omega, 'gamma': 0.,
               'frequency': omega if door == 'term' else 1.}[parameter]
    # A strength tangent is not zero at zero strength away from resonance.
    # Tangents in all other parameters are zero for a fixed inactive pole,
    # including at nominal undamped resonance.
    wanted = (1/complex(81-omega*omega, -.7*omega)
              if parameter == 'strength' else 0j)

    def response(value):
        strength = value if parameter == 'strength' else 0.
        omega0 = value if parameter == 'omega0' else (9. if parameter == 'strength' else omega)
        gamma = value if parameter == 'gamma' else (.7 if parameter == 'strength' else 0.)
        own = fd.Lorentz(omega_0=omega0, gamma=gamma, strength=strength)
        if parameter == 'frequency':
            return _read(door, (own,), omega=value, baseline=0.) if door == 'term' else _read(
                door, (own,), wavelength=value, baseline=0.)
        return _read(door, (own,), baseline=0.)

    if mode == 'numpy-secant':
        primal = response(np.float64(initial))
        # The numerator is exactly linear in strength. For inactive other
        # parameters, both sides have the same exact zero analytic response.
        tangent = response(np.float64(initial+1.))-primal
    elif mode.startswith('jax'):
        jax = pytest.importorskip('jax')
        jnp = pytest.importorskip('jax.numpy')
        with jax.enable_x64(True):
            value = jnp.asarray(initial, dtype=jnp.float64)
            primal = response(value)
            if mode == 'jax-reverse':
                tangent = (jax.grad(lambda x: jnp.real(response(x)))(value)
                           + 1j*jax.grad(lambda x: jnp.imag(response(x)))(value))
            else:
                primal, tangent = jax.jvp(response, (value,), (jnp.ones_like(value),))
    else:
        torch = pytest.importorskip('torch')
        value = torch.tensor(initial, dtype=torch.float64,
                             requires_grad=mode == 'torch-reverse')
        if mode == 'torch-reverse':
            primal = response(value)
            assert isinstance(primal, torch.Tensor) and primal.requires_grad
            tangent = (torch.autograd.grad(primal.real, value, retain_graph=True)[0]
                       + 1j*torch.autograd.grad(primal.imag, value)[0])
        elif mode == 'torch-forward':
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(value, torch.ones_like(value))
                result = response(dual)
                assert isinstance(result, torch.Tensor)
                primal, tangent = torch.autograd.forward_ad.unpack_dual(result)
                assert tangent is not None, 'a zero primal must not erase its tangent'
        else:
            primal, tangent = torch.func.jvp(response, (value,), (torch.ones_like(value),))
    assert _plain(primal) == 0j
    if wanted == 0j:
        assert _plain(tangent) == 0j
    else:
        assert _plain(tangent) == pytest.approx(wanted, rel=3e-13, abs=3e-13)
