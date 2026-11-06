"""A refused public material reads in the caller's own names and SI values (issue #56).

``lower_material`` turns a public declaration into the engine's normalized one, whose own
checks used to raise first: a public caller who wrote ``ConductiveMaterial(conductivity=-5.)``
read ``Conductor(sigma=...)`` and ``-0.00188`` (the engine's conductivity unit), and a complex
permittivity was told to say ``ConductiveMaterial(eps=..., sigma=...)``, which raises
``TypeError``. The same rules are now asked of the public declaration first. They are read off
the converted values, so the set admitted is the engine's exactly; only the words change.
"""
import itertools
import re

import pytest

from wavebit.errors import UnsupportedFeatureError
from wavebit.materials import ConductiveMaterial, Debye, DispersiveMaterial, Drude, Lorentz, Material
from wavebit.materials import _time_domain as td
from wavebit.materials._lower import lower_material
from wavebit.units import Units

UM = Units(length='um')

#: Engine spellings no public caller wrote.
PRIVATE = re.compile(r'\beps\b|\beps=|\bsigma\b|\beps_inf\b|\bdelta_eps\b|\bConductor\(|'
                     r'\bDispersive\(|\bfd\.|eps_at')

CASES = [
    (Material(epsilon=4 + .1j), 'Material(epsilon=(4+0.1j)): a complex permittivity',
     'ConductiveMaterial(epsilon=..., conductivity=...)'),
    (ConductiveMaterial(epsilon=2., conductivity=-5.),
     'ConductiveMaterial(conductivity=...): conductivity is a finite non-negative number in S/m',
     'got -5.0.'),
    (ConductiveMaterial(epsilon=-1., conductivity=1.),
     'ConductiveMaterial(epsilon=...): epsilon is a finite positive number', 'got -1.0.'),
    (Material(epsilon=-2.), 'Material(epsilon=...): epsilon is a finite positive number',
     'got -2.0.'),
    (Material(epsilon=2., mu=-1.), 'Material(mu=...): mu is a finite positive number',
     'got -1.0.'),
    (ConductiveMaterial(epsilon=2., conductivity=1., mu=-1.),
     'ConductiveMaterial(mu=...): mu is a finite positive number', 'got -1.0.'),
    (Material(epsilon=(2., 2.)), 'Material(epsilon=(2.0, 2.0)): a diagonal is (x, y, z), three',
     'got 2.'),
    (DispersiveMaterial(epsilon_inf=(2., 2., -1.), poles=(Debye(tau=1e-15, delta_epsilon=1.),)),
     'DispersiveMaterial(epsilon_inf=...) (z entry): epsilon_inf is a finite positive number',
     'got -1.0.'),
    (DispersiveMaterial(epsilon_inf=2., poles=(Drude(omega_p=0., gamma=1e13),)),
     'Drude(omega_p=...): omega_p is a finite positive number in rad/s', 'got 0.0.'),
    (DispersiveMaterial(epsilon_inf=2., poles=(Drude(omega_p=1e-320, gamma=1e13),)),
     'Drude(omega_p=...): omega_p is a finite positive number in rad/s', 'got 1e-320.'),
    (DispersiveMaterial(epsilon_inf=2., poles=(Lorentz(omega_0=0., gamma=1e13, strength=1e30),)),
     'Lorentz(omega_0=...): omega_0 is a finite positive number in rad/s', 'got 0.0.'),
    (DispersiveMaterial(epsilon_inf=2., poles=(Lorentz(omega_0=1e15, gamma=1e13, strength=-1e30),)),
     'Lorentz(strength=...): strength is a finite non-negative number in rad^2/s^2',
     'got -1e+30.'),
    (DispersiveMaterial(epsilon_inf=2., poles=(Debye(tau=1e-15, delta_epsilon=-1.),)),
     'Debye(delta_epsilon=...): delta_epsilon is a finite non-negative number', 'got -1.0.'),
]


@pytest.mark.parametrize('model,starts,says', CASES, ids=[repr(c[0])[:40] for c in CASES])
def test_a_refused_material_names_the_public_field_and_its_si_value(model, starts, says):
    with pytest.raises(UnsupportedFeatureError) as caught:
        lower_material(model, units=UM)
    message = str(caught.value)
    assert caught.value.code == 'material.fdtd'
    assert message.startswith('material: ' + starts) and says in message, message
    assert not PRIVATE.search(message), message


def _engine(model, units):
    """The engine's own construction of the converted values, without the public checks."""
    scale = units.time_unit_seconds
    if isinstance(model, ConductiveMaterial):
        return td.Conductor(eps=model.epsilon, sigma=units._conductivity_to_internal(
            model.conductivity), mu=model.mu)
    if isinstance(model, Material):
        return td.Material(eps=model.epsilon, mu=model.mu)
    poles = []
    for p in model.poles:
        if isinstance(p, Drude):
            poles.append(td.Drude(omega_p=p.omega_p * scale, gamma=p.gamma * scale))
        elif isinstance(p, Lorentz):
            poles.append(td.Lorentz(omega_0=p.omega_0 * scale, gamma=p.gamma * scale,
                                    strength=p.strength * scale ** 2))
        else:
            poles.append(td.Debye(tau=p.tau / scale, delta_eps=p.delta_epsilon))
    return td.Dispersive(eps_inf=model.epsilon_inf, poles=tuple(poles), mu=model.mu)


def _admits(build, *args, **kwargs):
    try:
        build(*args, **kwargs)
        return True
    except (ValueError, TypeError):
        return False


def _grid():
    signs = (-1., -1e-300, 0., 1e-300, 1.)
    for e, mu in itertools.product((-2., 0., 1e-300, 2.), (None, -1., 0., 1.5)):
        yield Material(epsilon=e, mu=mu)
    for e, s in itertools.product((-1., 2.), (-5., -1e-300, 0., 1e-300, 5e6, 1e300)):
        yield ConductiveMaterial(epsilon=e, conductivity=s)
    for wp, g in itertools.product((0., 1e-320, 1e-300, 2e15), (0., 1e-300, 1e13)):
        yield DispersiveMaterial(epsilon_inf=2., poles=(Drude(omega_p=wp, gamma=g),))
    for w0, s in itertools.product((0., 1e-320, 3e15), (-1e30, 0., 1e30)):
        yield DispersiveMaterial(epsilon_inf=2., poles=(Lorentz(omega_0=w0, gamma=1e13, strength=s),))
    for tau, d in itertools.product((1e-320, 1e-300, 1e-15, 1e300), signs):
        yield DispersiveMaterial(epsilon_inf=2., poles=(Debye(tau=tau, delta_epsilon=d),))
    for e in ((2., 2., -1.), (2., 3., 4.), (2., 0., 4.)):
        yield DispersiveMaterial(epsilon_inf=e, poles=(Debye(tau=1e-15, delta_epsilon=1.),))


@pytest.mark.parametrize('length', ('nm', 'um', 'm'))
def test_the_public_checks_admit_exactly_what_the_engine_admits(length):
    """Across zeros, signs, underflowing rates (1e-320 rad/s is zero once in engine units) and
    overflowing ones, in three length units: refused publicly exactly where the engine refuses
    the converted values, so the change is in the words alone."""
    units = Units(length=length)
    seen = set()
    for model in _grid():
        engine = _admits(_engine, model, units)
        public = _admits(lower_material, model, units=units)
        seen.add(engine)
        assert public is engine, (model, length, engine)
    assert seen == {True, False}


def test_a_traced_value_is_left_to_the_runtime_guards():
    """A traced negative conductivity is not read on the host, by the engine or here."""
    import torch
    leaf = torch.tensor(-5., dtype=torch.float64, requires_grad=True)
    lowered = lower_material(ConductiveMaterial(epsilon=2., conductivity=leaf), units=UM)
    assert lowered.sigma[0].requires_grad
