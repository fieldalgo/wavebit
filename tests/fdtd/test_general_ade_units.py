"""Same-SI homogeneous ADE on translated, upward-snapped rectangular domains.

The expected cell counts, Yee locations, discrete wave vector and state matrix
are written independently of product grid/lowering helpers. These cases vary
the full SnappedSlab axis, geometry units, and fixed scalar material constants.
"""
import math
from dataclasses import replace

import numpy as np
import pytest
from wavebit import Units, fdtd
from wavebit.materials import DispersiveMaterial, Drude, Lorentz
from wavebit.units import C0, Z0

from tests.fdtd._homogeneous_ade_reference import (
    FIELD_TOL, UNITS, fourier_mode, host, make_problem, matrix_history, mesh,
    numerics, pole_parameters, probe_at, run_problem, scaled_error, transition,
)


CELLS = (13, 10, 15)
DX_SI = .05e-6
SIZE_SI = np.array(CELLS)*DX_SI
DECLARED_SI = np.array([.61, .49, .71])*1e-6
CENTER_SI = np.array([.1, -.15, .2])*1e-6
STEPS = 23


def fixture(kind, slab_axis, unit_name):
    units = Units(length=unit_name)
    scale = units.length_unit_meters
    low = CENTER_SI-SIZE_SI/2
    k = 2*np.pi*np.array([1., 2., 1.])/SIZE_SI
    discrete_k = 2*np.sin(k*DX_SI/2)/DX_SI
    direction = discrete_k/np.linalg.norm(discrete_k)
    e = np.cross(direction, [.37, -.8, .53])
    e /= np.linalg.norm(e)
    initial = np.zeros(15, dtype=complex)
    initial[:3] = e
    initial[3:6] = .35*np.cross(direction, e)

    def axes(field, component):
        return tuple(low[a]+(np.arange(n)+.5*(a == component if field == 'E'
                                              else a != component))*DX_SI
                     for a, n in enumerate(CELLS))

    def phase(field, component, point=None):
        if point is None:
            x, y, z = axes(field, component)
            z, y, x = np.meshgrid(z, y, x, indexing='ij')
            point = (x, y, z)
        return np.exp(1j*sum(ki*r for ki, r in zip(k, point)))

    def probe(field, component):
        return tuple(a[i] for a, i in zip(axes(field, component), (2, 3, 4)))

    electric = np.stack([(initial[c]*phase('E', c)).real for c in range(3)])
    magnetic = np.stack([(initial[3+c]*phase('H', c)).real/Z0 for c in range(3)])
    time_scale = 1e-6/C0
    omega0, gamma, strength = (0. if kind == 'drude' else 4.2/time_scale,
                               .4/time_scale, 9./time_scale**2)
    epsilon, mu = 1.8, 1.1
    pole = (Drude(omega_p=math.sqrt(strength), gamma=gamma) if kind == 'drude' else
            Lorentz(omega_0=omega0, gamma=gamma, strength=strength))
    material = DispersiveMaterial(epsilon_inf=epsilon, mu=mu, poles=(pole,))
    axis = 'xyz'.index(slab_axis)
    # All declared dimensions round upward. Span the actual complete physical
    # grid, not just the smaller user declaration (.61,.49,.71) micrometres.
    span = (low[axis]/scale, (low[axis]+SIZE_SI[axis])/scale)
    problem = fdtd.Problem(units=units, scene=fdtd.Scene(
        domain=fdtd.Domain(size=tuple(DECLARED_SI/scale), center=tuple(CENTER_SI/scale)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic()),
        structures=(fdtd.Structure('uniform', shape=fdtd.SnappedSlab(**{slab_axis: span}), material=material),)),
        initial=fdtd.InitialField(E=electric, H=magnetic),
        outputs=fdtd.Outputs(final_fields=True, monitors=tuple(
            fdtd.TimeSeriesMonitor(field+'xyz'[c], component=field+'xyz'[c],
                        at=tuple(r/scale for r in probe(field, c)))
            for field in ('E', 'H') for c in range(3))))
    policy = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX_SI/scale),
                           duration=fdtd.Duration(steps=STEPS), courant=.5)
    dt = .5*DX_SI/(C0*math.sqrt(3))
    states = matrix_history(transition(dt, discrete_k, omega0, gamma, strength, epsilon, mu),
                            initial, STEPS)
    expected = {}
    for field, offset in (('E', 0), ('H', 3)):
        expected[field] = np.stack([(states[-1, offset+c]*phase(field, c)).real
                                    for c in range(3)])
        for c in range(3):
            expected[field+'xyz'[c]] = (states[1:, offset+c] *
                                       phase(field, c, probe(field, c))).real
    return problem, policy, expected, dt


@pytest.mark.parametrize('kind', ['lorentz', 'drude'])
@pytest.mark.parametrize('slab_axis', list('xyz'))
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'cuda'])
def test_translated_rectangular_full_slab_same_si_units(device, slab_axis, kind, record_property):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device != 'cpu' and not torch.cuda.is_available():
        pytest.skip('requires an actual CUDA device')
    previous = None
    errors = {}
    for unit_name in ('um', 'nm'):
        problem, policy, expected, dt = fixture(kind, slab_axis, unit_name)
        saved_e, saved_h = problem.initial.E.copy(), problem.initial.H.copy()
        result = run_problem(problem, policy, device=device)
        np.testing.assert_allclose(result.provenance['dt_seconds'], dt, rtol=2e-15, atol=0.)
        actual = {}
        for field in ('E', 'H'):
            values = result.final_fields.values[field]
            assert values.shape == (3, CELLS[2], CELLS[1], CELLS[0])
            actual[field] = host(values)*(Z0 if field == 'H' else 1.)
            for c in range(3):
                label = field+'xyz'[c]
                values = result[label].values
                assert values.dtype == torch.float64 and values.device == torch.device(device)
                actual[label] = host(values)*(Z0 if field == 'H' else 1.)
                times = (np.arange(STEPS)+(1. if field == 'E' else .5))*dt
                np.testing.assert_allclose(host(result[label].times_seconds), times, rtol=2e-14, atol=0.)
        for label, values in actual.items():
            assert np.max(abs(expected[label])) > .01
            errors[unit_name+'_'+label] = scaled_error(values, expected[label])
            assert errors[unit_name+'_'+label] < FIELD_TOL
            if previous is not None:
                assert scaled_error(values, previous[label]) < FIELD_TOL
        previous = {label: value.copy() for label, value in actual.items()}
        np.testing.assert_array_equal(problem.initial.E, saved_e)
        np.testing.assert_array_equal(problem.initial.H, saved_h)
    record_property('maximum_field_record_relative_error', max(errors.values()))
    record_property('resolved_physical_cells_xyz', repr(CELLS))
    record_property('slab_axis_units', repr((slab_axis, 'um', 'nm')))


@pytest.mark.parametrize('kind', ['lorentz', 'drude'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'cuda'])
def test_prepared_reuse_concrete_material_constants(device, kind, record_property):
    """Fixed topology reuses the plan while the concrete scalar coefficients change."""
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device != 'cpu' and not torch.cuda.is_available():
        pytest.skip('requires an actual CUDA device')
    cells, steps = 12, 23
    _, discrete_k, initial, electric, magnetic, spatial = fourier_mode(cells, 0)
    problem = make_problem(kind, True, electric, magnetic, cells)
    policy = numerics(cells, steps)
    prepared = fdtd.prepare(problem, numerics=policy, backend='torch', device=device)
    dt = .5/cells/math.sqrt(3)*UNITS.time_unit_seconds
    saved_electric, saved_magnetic = electric.copy(), magnetic.copy()
    with torch.no_grad():
        original = prepared.run(problem)
    original_values = tuple(original.final_fields.values.values()) + tuple(
        original[field+'xyz'[c]].values for field in ('E', 'H') for c in range(3))
    original_copies = tuple(value.clone() for value in original_values)
    _, _, strength = pole_parameters(kind, True)
    omega0 = 0. if kind == 'drude' else 4.2/UNITS.time_unit_seconds
    gamma = .4/UNITS.time_unit_seconds
    epsilon, mu = 1.8, 1.1
    pole = (Drude(omega_p=math.sqrt(strength), gamma=gamma) if kind == 'drude' else
            Lorentz(omega_0=omega0, gamma=gamma, strength=strength))
    structure = problem.scene.structures[0]
    material = replace(structure.material, epsilon_inf=epsilon, mu=mu, poles=(pole,))
    changed = replace(problem, scene=replace(problem.scene, structures=(replace(
        structure, material=material),)))
    assert prepared.validate(changed).ok
    with torch.no_grad():
        result = prepared.run(changed)
    fresh = run_problem(changed, policy, device=device)
    assert bool(result.validity.checks['runtime_stability'])
    assert prepared.contract['resolved_steps'] == steps
    np.testing.assert_allclose(result.provenance['dt_seconds'], dt, rtol=2e-15, atol=0.)
    states = matrix_history(transition(dt, discrete_k, omega0, gamma, strength, epsilon, mu),
                            initial, steps)
    errors = {}
    for field, offset in (('E', 0), ('H', 3)):
        scale = Z0 if field == 'H' else 1.
        raw = result.final_fields.values[field]
        assert raw.device == torch.device(device) and raw.dtype == torch.float64
        wanted = np.stack([(states[-1, offset+c]*spatial(field, c, mesh(cells, field, c))).real
                           for c in range(3)])
        values = host(raw)*scale
        errors[field] = scaled_error(values, wanted)
        assert errors[field] < FIELD_TOL
        assert scaled_error(values, host(fresh.final_fields.values[field])*scale) < FIELD_TOL
        for c in range(3):
            label = field+'xyz'[c]
            wanted = (states[1:, offset+c]*spatial(field, c, probe_at(cells, field, c))).real
            assert np.max(abs(wanted)) > .01
            raw = result[label].values
            assert raw.device == torch.device(device) and raw.dtype == torch.float64
            values = host(raw)*scale
            errors[label] = scaled_error(values, wanted)
            assert errors[label] < FIELD_TOL
            assert scaled_error(values, host(fresh[label].values)*scale) < FIELD_TOL
    assert np.max(abs(host(result.final_fields.values['E']) -
                     host(original.final_fields.values['E']))) > .01
    for value, saved in zip(original_values, original_copies):
        assert torch.equal(value, saved)
    np.testing.assert_array_equal(electric, saved_electric)
    np.testing.assert_array_equal(magnetic, saved_magnetic)
    record_property('changed_epsilon_mu_omega0_gamma', repr((epsilon, mu, omega0, gamma)))
    record_property('maximum_field_record_relative_error', max(errors.values()))
