"""Independent SI oscillator references and bounded public homogeneous fixtures."""
import math

import numpy as np
from wavebit import Units, fdtd
from wavebit.materials import DispersiveMaterial, Drude, Lorentz
from wavebit.units import C0

UNITS = Units(length='um')
EPSILON, MU = 2.25, 1.3
FIELD_TOL, RECURRENCE_TOL = 1e-11, 1e-12


def pole_parameters(kind, damped):
    scale = UNITS.time_unit_seconds
    return (0. if kind == 'drude' else 5./scale,
            (.7 if damped else 0.)/scale, 12./scale**2)


def coefficients(dt, omega0, gamma, strength):
    denominator = 1+gamma*dt/2
    return ((2-omega0**2*dt**2)/denominator,
            (1-gamma*dt/2)/denominator, strength*dt**2/denominator)


def uniform_history(steps, dt, omega0, gamma, strength, epsilon=EPSILON):
    n = np.arange(steps+1)
    w2 = omega0**2+strength/epsilon
    equilibrium = omega0**2/w2
    if gamma == 0:
        theta = 2*np.arcsin(dt*np.sqrt(w2)/2)
        return equilibrium+(1-equilibrium)*np.cos((n-.5)*theta)/np.cos(theta/2)
    g, h, d = coefficients(dt, omega0, gamma, strength)
    roots = np.roots([1., -(g-d/epsilon), h]).astype(complex)
    assert abs(roots[0]-roots[1]) > 1e-5
    first = (1-equilibrium)*(1-roots[1])/(roots[0]-roots[1])
    second = (1-equilibrium)-first
    result = equilibrium+first*roots[0]**n+second*roots[1]**n
    assert np.max(abs(result.imag)) < 1e-13
    return result.real


def continuum_history(times, omega0, gamma, strength, epsilon=EPSILON):
    w2 = omega0**2+strength/epsilon
    equilibrium = omega0**2/w2
    omega = np.sqrt(w2-gamma**2/4)
    return equilibrium+(1-equilibrium)*np.exp(-gamma*times/2)*(
        np.cos(omega*times)+gamma/(2*omega)*np.sin(omega*times))


def yee_axes(cells, field, component):
    return tuple(-.5+(np.arange(cells)+.5*(axis == component if field == 'E'
                                         else axis != component))/cells
                 for axis in range(3))


def mesh(cells, field, component):
    x, y, z = yee_axes(cells, field, component)
    z, y, x = np.meshgrid(z, y, x, indexing='ij')
    return x, y, z


def probe_at(cells, field, component):
    return tuple(values[index] for values, index in zip(
        yee_axes(cells, field, component), (cells//2+1,)*3))


def make_problem(kind, damped, electric, magnetic, cells):
    omega0, gamma, strength = pole_parameters(kind, damped)
    pole = (Drude(omega_p=math.sqrt(strength), gamma=gamma) if kind == 'drude'
            else Lorentz(omega_0=omega0, gamma=gamma, strength=strength))
    material = DispersiveMaterial(epsilon_inf=EPSILON, mu=MU, poles=(pole,))
    return fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(1.,)*3),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic()),
        structures=(fdtd.Structure('uniform', shape=fdtd.SnappedSlab(z=(-.5, .5)), material=material),)),
        initial=fdtd.InitialField(E=electric, H=magnetic),
        outputs=fdtd.Outputs(final_fields=True, monitors=tuple(
            fdtd.TimeSeriesMonitor(field+'xyz'[c], component=field+'xyz'[c],
                        at=probe_at(cells, field, c))
            for field in ('E', 'H') for c in range(3))))


def numerics(cells, steps, courant=.5):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=1/cells),
                         duration=fdtd.Duration(steps=steps), courant=courant)


def cross_matrix(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=complex)


def transition(dt, discrete_k_si, omega0, gamma, strength, epsilon=EPSILON, mu=MU):
    """Dense SI-based matrix; no product coefficient/step helper is imported."""
    c = 1j*cross_matrix(discrete_k_si)
    eye = np.eye(3)
    a, b = C0*dt/epsilon, C0*dt/mu
    g, h, d = coefficients(dt, omega0, gamma, strength)
    out = np.zeros((15, 15), dtype=complex)
    out[:3, :3] = eye-a*b*c@c
    out[:3, 3:6] = a*c
    out[:3, 12:15] = -eye/epsilon
    out[3:6, :3] = -b*c
    out[3:6, 3:6] = eye
    out[6:9] = d*out[:3]
    out[6:9, 6:9] += g*eye
    out[6:9, 9:12] -= h*eye
    out[9:12, 6:9] = eye
    out[12:15] = out[6:9]
    out[12:15, 6:9] -= eye
    return out


def fourier_mode(cells, rotation):
    k = np.roll(2*np.pi*np.array([1., 2., 1.]), rotation)
    discrete_k = 2*cells*np.sin(k/(2*cells))
    direction = discrete_k/np.linalg.norm(discrete_k)
    transverse = np.cross(direction, np.roll([.3, -.8, .5], rotation))
    transverse /= np.linalg.norm(transverse)
    electric_amplitude = transverse+.2*direction
    initial = np.zeros(15, dtype=complex)
    initial[:3] = electric_amplitude

    def spatial(field, component, at):
        return np.exp(1j*sum(ki*coordinate for ki, coordinate in zip(k, at)))

    electric = np.stack([(electric_amplitude[c]*spatial('E', c, mesh(cells, 'E', c))).real
                         for c in range(3)])
    magnetic = np.zeros_like(electric)
    return k, discrete_k/UNITS.length_unit_meters, initial, electric, magnetic, spatial


def matrix_history(matrix, initial, steps):
    return np.stack([np.linalg.matrix_power(matrix, n)@initial for n in range(steps+1)])


def host(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def scaled_error(actual, expected, zero_scale=1.):
    peak = float(np.max(abs(expected)))
    return float(np.max(abs(actual-expected))/(peak if peak else zero_scale))


def run_problem(problem, policy, device='cpu'):
    import pytest
    torch = pytest.importorskip('torch')
    with torch.no_grad():
        result = fdtd.run(problem, numerics=policy, backend='torch', device=device)
    assert bool(result.validity.checks['runtime_stability'])
    for values in result.final_fields.values.values():
        assert values.device == torch.device(device) and values.dtype == torch.float64
    return result
