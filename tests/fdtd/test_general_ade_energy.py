"""Scalar ADE energy must identify the pole family independently of its roots."""
import json
import math

import numpy as np
import pytest

from wavebit import Units, fdtd
from wavebit.materials import Debye, DispersiveMaterial, Drude, Lorentz


UNITS = Units(length='um')
DT = .03125
EPSILON = 2.25
STRENGTH = 4.
STEPS = 17
TOL = 2e-12


def problem(kind):
    scale = UNITS.time_unit_seconds
    gamma = (2/DT)/scale
    pole = (Drude(omega_p=math.sqrt(STRENGTH)/scale, gamma=gamma) if kind == 'drude'
            else Lorentz(omega_0=2/scale, strength=STRENGTH/scale**2, gamma=gamma))
    initial = np.zeros((3, 8, 8, 8))
    initial[1] = 1.
    return fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(1.,)*3),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Periodic()),
        structures=(fdtd.Structure('body', shape=fdtd.SnappedSlab(z=(-.5, .5)),
                                   material=DispersiveMaterial(epsilon_inf=EPSILON, poles=(pole,))),)),
        initial=fdtd.InitialField(E=initial), outputs=fdtd.Outputs(final_fields=True))


def numerics():
    # This Courant choice produces the binary-exact internal dt=1/32.
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.125),
                         duration=fdtd.Duration(steps=STEPS), courant=math.sqrt(3)/4)


def oscillator_reference(kind):
    """Uniform Ampere recurrence and discrete oscillator storage, without lowerers."""
    omega = 0. if kind == 'drude' else 2.
    gain = (2-omega**2*DT**2)/2
    drive = STRENGTH*DT**2/2
    electric, polarization, previous = 1., 0., 0.
    energies = []
    for _ in range(STEPS):
        electric -= (polarization-previous)/EPSILON
        new = gain*polarization+drive*electric
        previous, polarization = polarization, new
        increment = polarization-previous
        energies.append(8**3*(EPSILON*electric**2 + increment**2/(DT**2*STRENGTH)
                              + omega**2*polarization*previous/STRENGTH-electric*increment))
    return electric, np.asarray(energies)


@pytest.mark.parametrize('backend,device', [('numpy', 'cpu'), ('torch', 'cpu'),
                                           ('jax', 'cpu'), ('torch', 'cuda:0')],
                         ids=['cpu_numpy', 'cpu_torch', 'cpu_jax', 'gpu_torch'])
@pytest.mark.parametrize('kind', ['drude', 'lorentz'])
def test_zero_hold_oscillator_energy_is_not_debye(kind, backend, device, record_property):
    if backend == 'jax':
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        torch.set_num_threads(1)
        if device != 'cpu' and not torch.cuda.is_available():
            pytest.skip('actual CUDA required')
    declaration = problem(kind)
    prepared = fdtd.prepare(declaration, numerics=numerics(), backend=backend, device=device)
    assert prepared._plan.dt == DT
    # Confirm this is the exact ambiguous coefficient, not a nearby damping value.
    scene, _ = prepared._lower(declaration)
    assert scene.structures[0].material.poles[0].coefficients(DT)[1] == 0.
    result = prepared.run(declaration)
    electric, expected = oscillator_reference(kind)
    field = result.final_fields.values['E']
    if hasattr(field, 'detach'):
        field = field.detach().cpu().numpy()
    np.testing.assert_allclose(field[1], electric, rtol=TOL, atol=0.)
    times, energy = result.diagnostics['interior_energy']
    # The diagnostic samples integer E times; its values are normalized summed storage.
    np.testing.assert_allclose(times, np.array([1, 16])*DT, rtol=0., atol=0.)
    wanted = expected[np.array([1, 16])-1]
    record_property('energy_observation', json.dumps({
        'kind': kind, 'backend': backend, 'device': device,
        'actual': np.asarray(energy).tolist(), 'expected': wanted.tolist(),
        'runtime_stability': bool(result.validity.checks['runtime_stability'])}))
    assert np.isfinite(energy).all()
    np.testing.assert_allclose(energy, wanted, rtol=TOL, atol=0.)


@pytest.mark.parametrize('backend', ['numpy', 'torch', 'jax'])
@pytest.mark.parametrize('delta', [0., .5], ids=['zero_strength', 'debye'])
def test_cpu_debye_energy_keeps_its_declared_family(backend, delta):
    if backend != 'numpy':
        module = pytest.importorskip(backend)
        if backend == 'jax':
            module.config.update('jax_enable_x64', True)
    declaration = problem('drude')
    material = DispersiveMaterial(epsilon_inf=EPSILON, poles=(
        Debye(tau=DT/2*UNITS.time_unit_seconds, delta_epsilon=delta),))
    body = declaration.scene.structures[0].replace(material=material)
    declaration = declaration.replace(scene=declaration.scene.replace(structures=(body,)))
    result = fdtd.prepare(declaration, numerics=numerics(), backend=backend).run(declaration)
    # At tau=dt/2, g=0, c=delta/2, drive=delta/2. The carried polarization is
    # the split memory P~, so physical storage is |P~_old+c E|^2/delta.
    electric, polarization, previous = 1., 0., 0.
    energies = []
    for _ in range(STEPS):
        electric -= (polarization-previous)/(EPSILON+delta/2)
        previous, polarization = polarization, delta/2*electric
        stored = (previous+delta/2*electric)**2/delta if delta else 0.
        energies.append(8**3*(EPSILON*electric**2+stored))
    field = result.final_fields.values['E']
    if hasattr(field, 'detach'):
        field = field.detach().cpu().numpy()
    np.testing.assert_allclose(field[1], electric, rtol=TOL, atol=0.)
    np.testing.assert_allclose(result.diagnostics['interior_energy'][1],
                               np.array(energies)[[0, 15]], rtol=TOL, atol=0.)


@pytest.mark.parametrize('debye', [False, True])
def test_cpu_bare_coefficient_dictionary_retains_energy_test_hook(debye):
    from wavebit.fdtd._kernel.observe import InteriorEnergy
    from wavebit.fdtd._runtime import runtime

    rt = runtime('numpy')
    gain, hold, drive = (.25, 0., .125) if debye else (1.5, .75, .125)
    coefficients = {'step_gain': (np.array(gain),), 'hold': (np.array(hold),),
                    'drive': (np.array(drive),)}
    mem = {'P': (np.full((3, 1, 1, 1), .3),),
           'previous': (np.full((3, 1, 1, 1), .2),)}
    field = np.full((3, 1, 1, 1), .7)
    args = (1, ((0, 1),)*3, (0,)*3, DT, rt)
    legacy = InteriorEnergy._pole(mem, field, coefficients, *args)
    declared = InteriorEnergy._pole(mem, field, {
        **coefficients, 'energy_is_debye': (np.array(float(debye)),)}, *args)
    assert legacy == declared
    assert np.isfinite(declared)
