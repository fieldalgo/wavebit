"""``wavebit.materials.Bruggeman``: the two-phase effective medium and its passive root.

Each assertion is against something the implementation does not compute: the mixing equation
itself, an independent root found by ``numpy.roots``, the exact limits, the symmetry of the
symmetric rule, and a dispersive constituent evaluated at its own wavelength.
"""
import numpy as np
import pytest

from wavebit import Units, ValidationError
from wavebit.materials import Bruggeman, DispersiveMaterial, Drude, Material

UNITS = Units(length='um')
CASES = [(15.2 + 0.16j, 1.0, 0.5), (2.25, 1.0, 0.3), (1.0, -20 + 1.5j, 0.2),
         (-20 + 1.5j, 2.25, 0.7), (12.25, 2.1025, 0.05), (4.0 + 2.0j, 1.0, 0.33)]


def _epsilon(host, inclusion, fraction, wavelength=0.6):
    model = Bruggeman(host=Material(epsilon=host), inclusion=Material(epsilon=inclusion), fraction=fraction)
    return complex(np.asarray(model.at(wavelength, units=UNITS).epsilon))


def _independent(host, inclusion, fraction):
    b = (2 - 3*fraction)*host + (3*fraction - 1)*inclusion
    roots = np.roots([2, -b, -host*inclusion])
    passive = [r for r in roots if r.imag >= -1e-12]
    return max(passive, key=lambda r: (round(r.imag, 12), r.real))


@pytest.mark.parametrize('host, inclusion, fraction', CASES)
def test_the_effective_permittivity_solves_the_mixing_equation(host, inclusion, fraction):
    e = _epsilon(host, inclusion, fraction)
    residual = (1 - fraction)*(host - e)/(host + 2*e) + fraction*(inclusion - e)/(inclusion + 2*e)
    assert abs(residual) < 1e-14
    assert abs(e - _independent(host, inclusion, fraction)) < 1e-13*abs(e)
    assert e.imag >= 0


@pytest.mark.parametrize('host, inclusion, fraction', CASES)
def test_the_rule_is_symmetric_in_its_two_phases(host, inclusion, fraction):
    assert abs(_epsilon(host, inclusion, fraction) - _epsilon(inclusion, host, 1 - fraction)) < 1e-13


def test_the_limits_are_the_constituents():
    assert _epsilon(15.2 + 0.16j, 1.0, 0.) == 15.2 + 0.16j
    assert abs(_epsilon(15.2 + 0.16j, 1.0, 1.) - 1.0) < 1e-15


def test_a_dispersive_constituent_is_evaluated_at_each_wavelength():
    metal = DispersiveMaterial(epsilon_inf=1.0, poles=(Drude(omega_p=1.37e16, gamma=1.0e14),))
    wavelengths = np.array([0.4, 0.6, 0.8])
    mixed = np.asarray(Bruggeman(host=metal, inclusion=Material(epsilon=1.0), fraction=0.4)
                       .at(wavelengths, units=UNITS).epsilon)
    host = np.asarray(metal.at(wavelengths, units=UNITS).epsilon)
    assert mixed.shape == (3,)
    for e, h in zip(mixed, host):
        assert abs(e - _independent(h, 1.0, 0.4)) < 1e-12*abs(e)


def test_mixtures_nest():
    inner = Bruggeman(host=Material(epsilon=12.25), inclusion=Material(epsilon=2.25), fraction=0.5)
    outer = Bruggeman(host=inner, inclusion=Material(epsilon=1.0), fraction=0.5)
    e = complex(np.asarray(outer.at(0.6, units=UNITS).epsilon))
    assert abs(e - _independent(_independent(12.25, 2.25, 0.5), 1.0, 0.5)) < 1e-13*abs(e)


@pytest.mark.parametrize('fraction', (-0.1, 1.2, float('nan')))
def test_a_fraction_outside_zero_to_one_is_refused(fraction):
    with pytest.raises(ValidationError) as caught:
        Bruggeman(host=Material(epsilon=2.25), inclusion=Material(epsilon=1.0), fraction=fraction)
    assert caught.value.issue.code in ('material.fraction', 'material.scalar')


def test_magnetic_and_tensor_constituents_are_refused():
    magnetic = Bruggeman(host=Material(epsilon=2.25, mu=2.0), inclusion=Material(epsilon=1.0), fraction=0.5)
    with pytest.raises(ValidationError) as caught:
        magnetic.at(0.6, units=UNITS)
    assert caught.value.issue.code == 'material.mixture'
    tensor = Bruggeman(host=Material(epsilon=(2.25, 2.25, 3.0)), inclusion=Material(epsilon=1.0), fraction=0.5)
    with pytest.raises(ValidationError) as caught:
        tensor.at(0.6, units=UNITS)
    assert caught.value.issue.code == 'material.mixture'
    with pytest.raises(ValidationError) as caught:
        Bruggeman(host=2.25, inclusion=Material(epsilon=1.0), fraction=0.5)
    assert caught.value.issue.code == 'material.type'


def test_the_fraction_differentiates_on_torch_and_jax():
    torch = pytest.importorskip('torch')
    host, inclusion, f0 = 15.2 + 0.16j, 1.0, 0.4
    step = 1e-6
    difference = (_epsilon(host, inclusion, f0 + step) - _epsilon(host, inclusion, f0 - step))/(2*step)
    fraction = torch.tensor(f0, dtype=torch.float64, requires_grad=True)
    e = Bruggeman(host=Material(epsilon=host), inclusion=Material(epsilon=inclusion), fraction=fraction).at(
        0.6, units=UNITS).epsilon
    (real,) = torch.autograd.grad(e.real, fraction, retain_graph=True)
    (imag,) = torch.autograd.grad(e.imag, fraction)
    assert abs(complex(float(real), float(imag)) - difference) < 1e-8*abs(difference)
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)

    def part(f, which):
        value = Bruggeman(host=Material(epsilon=host), inclusion=Material(epsilon=inclusion), fraction=f).at(
            0.6, units=UNITS).epsilon
        return getattr(jax.numpy, which)(value)
    derivative = complex(float(jax.grad(lambda f: part(f, 'real'))(f0)), float(jax.grad(lambda f: part(f, 'imag'))(f0)))
    assert abs(derivative - difference) < 1e-8*abs(difference)
