"""Independent Gaussian pulse jets and forced SI Maxwell-Yee recurrence.

Coordinates are p=(l,b,phi,d): wavelength=2*l um, bandwidth=.08*b/T Hz,
phase=phi radians and delay=BASE_DELAY+T*d seconds. BASE_DELAY is fixed;
changing bandwidth moves the Gaussian peak. No product numerical helper is used.
"""
import math

import numpy as np
from wavebit import Units, fdtd
from wavebit.materials import ConductiveMaterial
from wavebit.units import C0, EPSILON0, Z0

T = 1e-6/C0
DX = .05e-6
DT = .5*DX/(C0*np.sqrt(3.))
CELLS = 8
EPSILON, MU, SIGMA = 2.25, 1.3, 600.
MOMENT = 1.7e-16
CUTOFF = 1e-3
KAPPA = np.sqrt(-2*np.log(CUTOFF))
BASE_DELAY = -KAPPA*T/(2*np.pi*.08)
# Jacobian to public (um, Hz, radians, seconds); SI wavelength factor is 2e-6.
SCALES = np.array([2., .08/T, 1., T])
POINTS = ((1., 1., .4, .12), (.97, 1.01, -.2, .16))
CENTER = np.array([.013, -.021, .007])*1e-6
PROBE = (5, 3, 2)
LABELS = tuple(f+c for f in ('E', 'H') for c in 'xyz')
FIELD_WEIGHTS = np.array([1., .7, 1.3, .6, 1.1, .9])
PAIRS = tuple((i, j) for i in range(4) for j in range(i, 4))
RTOL = 2e-12
# Absolute steps in normalized p. Bandwidth shifts the carrier phase strongly;
# smaller steps control the O(h^2) gradient-to-Hessian secant error.
SECANT_STEPS = (1e-5, 5e-6)
SECANT_TOL = 2e-6
NONZERO = 1e-6


def time_weights(steps):
    return 1+.2*np.arange(steps)/steps


def fixture(point=POINTS[0], stamp='bspline', steps=17, block=None, unit='um'):
    units = Units(length=unit)
    scale = units.length_unit_meters
    l, bandwidth, phase, delay = point
    records = []
    for field in ('E', 'H'):
        for c in range(3):
            offsets = np.array([.5*(a == c if field == 'E' else a != c) for a in range(3)])
            at = (-CELLS*DX/2+(np.asarray(PROBE)+offsets)*DX)/scale
            label = field+'xyz'[c]
            records.append(fdtd.TimeSeriesMonitor(label, component=label, at=tuple(at)))
    problem = fdtd.Problem(units=units, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(CELLS*DX/scale,)*3),
        background=ConductiveMaterial(epsilon=EPSILON, mu=MU, conductivity=SIGMA),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic())),
        sources=(fdtd.CurrentDipole('d', center=tuple(CENTER/scale), component='Ey', kind='electric',
            current_moment=MOMENT, stamp=stamp, phase=phase, delay_seconds=BASE_DELAY+T*delay,
            pulse=fdtd.GaussianPulse(wavelength=(2e-6/scale)*l,
                                     frequency_sigma_hz=(.08/T)*bandwidth, cutoff=CUTOFF)),),
        outputs=fdtd.Outputs(final_fields=True, monitors=tuple(records)))
    policy = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX/scale),
                          duration=fdtd.Duration(steps=steps), courant=.5, checkpoint_steps=block)
    return problem, policy


def pulse_jet(point, steps, mutant=None):
    """Current, four first and full 4x4 second derivatives in normalized p."""
    l, bandwidth, phase, delay = point
    times = (np.arange(steps)+(0. if mutant == 'wrong_clock' else .5))*DT
    frequency = .08*bandwidth/T
    width_rate = 2*np.pi*frequency
    rate_prime = 2*np.pi*.08/T
    omega = 2*np.pi*C0/(2e-6*l)
    physical_delay = BASE_DELAY+T*delay
    u = times-physical_delay-KAPPA/width_rate
    du = np.zeros((4, steps))
    du[1] = KAPPA*rate_prime/width_rate**2
    du[3] = -T if mutant != 'wrong_delay_units' else -1.
    ddu = np.zeros((4, 4, steps))
    ddu[1, 1] = -2*KAPPA*rate_prime**2/width_rate**3
    if mutant == 'frozen_peak':
        du[1] = 0.
        ddu[1, 1] = 0.
    rate_d = np.array([0., rate_prime, 0., 0.])
    omega_d = np.array([-omega/l, 0., 0., 0.])
    omega_dd = np.zeros((4, 4))
    omega_dd[0, 0] = 2*omega/l**2
    if mutant == 'omit_carrier':
        omega_d[:] = 0.
        omega_dd[:] = 0.
    v = width_rate*u
    dv = rate_d[:, None]*u+width_rate*du
    ddv = (rate_d[:, None, None]*du[None]+rate_d[None, :, None]*du[:, None]
           +width_rate*ddu)
    theta_d = omega_d[:, None]*u+omega*du
    theta_d[2] += 1.
    theta_dd = (omega_dd[:, :, None]*u+omega_d[:, None, None]*du[None]
                +omega_d[None, :, None]*du[:, None]+omega*ddu)
    q = -.5*v*v+1j*(omega*u+phase)
    q_d = -v*dv+1j*theta_d
    q_dd = -dv[:, None]*dv[None]-v*ddv+1j*theta_dd
    wave = np.exp(q)
    value = MOMENT*wave.imag
    first = MOMENT*(wave*q_d).imag
    second = MOMENT*(wave*(q_d[:, None]*q_d[None]+q_dd)).imag
    if mutant == 'omit_mixed':
        for i in range(4):
            for j in range(4):
                if i != j:
                    second[i, j] = 0.
    if mutant == 'omit_second':
        second[:] = 0.
    return value, first, second


def stamp_array(kind, mutant=None):
    """Cardinal cubic via truncated powers, or the documented width-two tent."""
    parts = []
    for axis in range(3):
        coordinates = -CELLS*DX/2+(np.arange(CELLS)+(.5 if axis == 1 else 0.))*DX
        u = (coordinates-CENTER[axis])/DX
        if kind == 'bspline':
            value = sum((-1.)**k*math.comb(4, k)*np.maximum(u+2-k, 0.)**3
                        for k in range(5))/6
            value = np.where(abs(u) < 2., value, 0.)
        elif kind == 'tent':
            value = np.maximum(2-abs(u), 0.)/4
        else:
            raise ValueError('unknown independent stamp')
        parts.append(value)
    result = np.einsum('k,j,i->kji', parts[2], parts[1], parts[0])
    if mutant == 'stamp_squared_normalization':
        result /= np.sum(result**2)
    return result


def difference(value, axis, forward):
    storage_axis = value.ndim-1-axis
    n = value.shape[storage_axis]
    adjacent = np.take(value, (np.arange(n)+(1 if forward else -1)) % n, axis=storage_axis)
    return (adjacent-value if forward else value-adjacent)/DX


def curl(value, forward):
    x, y, z = (value[:, c] for c in range(3))
    return np.stack((difference(z, 1, forward)-difference(y, 2, forward),
                     difference(x, 2, forward)-difference(z, 0, forward),
                     difference(y, 0, forward)-difference(x, 1, forward)), axis=1)


def unpack_lanes(lanes):
    value, first = lanes[0], lanes[1:5]
    second = np.empty((4, 4)+value.shape)
    for lane, (i, j) in enumerate(PAIRS, 5):
        second[i, j] = second[j, i] = lanes[lane]
    return value, first, second


def reference(point=POINTS[0], stamp='bspline', steps=17, mutant=None):
    drive, ddrive, dddrive = pulse_jet(point, steps, mutant)
    forcing = np.stack((drive, *ddrive, *(dddrive[i, j] for i, j in PAIRS)))
    spatial = np.zeros((3, CELLS, CELLS, CELLS))
    spatial[1] = stamp_array(stamp, mutant)
    r = SIGMA*DT/(2*EPSILON0)
    denominator = EPSILON+r
    q, g, b = (EPSILON-r)/denominator, C0*DT/denominator, C0*DT/MU
    source_scale = -DT/(EPSILON0*denominator*DX**3)
    electric = np.zeros((15, 3, CELLS, CELLS, CELLS))
    magnetic = np.zeros_like(electric)  # h=Z0*H, including all tangent lanes.
    records = np.zeros((15, 6, steps))
    ix, iy, iz = PROBE
    for n in range(steps):
        magnetic = magnetic-b*curl(electric, True)
        electric = q*electric+g*curl(magnetic, False)+source_scale*forcing[:, n, None, None, None, None]*spatial
        records[:, :3, n] = electric[:, :, iz, iy, ix]
        records[:, 3:, n] = magnetic[:, :, iz, iy, ix]/Z0
    values, first, second = {}, {}, {}
    for field, lanes in (('E', electric), ('H', magnetic/Z0)):
        values[field], first[field], second[field] = unpack_lanes(lanes)
    record_values, record_first, record_second = unpack_lanes(records)
    return dict(fields=values, first_fields=first, second_fields=second,
                records=dict(zip(LABELS, record_values)),
                first_records=dict(zip(LABELS, record_first.transpose(1, 0, 2))),
                second_records=dict(zip(LABELS, record_second.transpose(2, 0, 1, 3))),
                drive=drive, first_drive=ddrive, second_drive=dddrive,
                drive_times=(np.arange(steps)+.5)*DT,
                times={label: (np.arange(steps)+(1. if label.startswith('E') else .5))*DT
                       for label in LABELS})


def weighted_jet(value, first, second, weight):
    axes = tuple(range(1, first.ndim))
    second_axes = tuple(range(2, second.ndim))
    return (float(np.mean(weight*value**2)),
            np.mean(2*weight*value*first, axis=axes),
            np.mean(2*weight*(first[:, None]*first[None]+value*second), axis=second_axes))


def objectives(expected):
    """Drive mean-square, and 1/6 weighted field means +1/24 weighted TimeSeriesMonitor means."""
    steps = len(expected['drive'])
    drive = weighted_jet(expected['drive']/MOMENT, expected['first_drive']/MOMENT,
                         expected['second_drive']/MOMENT, 1.)
    values = np.concatenate((expected['fields']['E'], Z0*expected['fields']['H']))
    first = np.concatenate((expected['first_fields']['E'], Z0*expected['first_fields']['H']), axis=1)
    second = np.concatenate((expected['second_fields']['E'], Z0*expected['second_fields']['H']), axis=2)
    field = weighted_jet(values, first, second, FIELD_WEIGHTS[:, None, None, None])
    records = np.stack([expected['records'][key]*(Z0 if key.startswith('H') else 1.) for key in LABELS])
    dr = np.stack([expected['first_records'][key]*(Z0 if key.startswith('H') else 1.) for key in LABELS], axis=1)
    ddr = np.stack([expected['second_records'][key]*(Z0 if key.startswith('H') else 1.) for key in LABELS], axis=2)
    record = weighted_jet(records, dr, ddr, FIELD_WEIGHTS[:, None]*time_weights(steps))
    return {'drive': drive, 'fields_records': tuple(a+.25*b for a, b in zip(field, record))}
