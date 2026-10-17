"""Two exactly touching conductive films: a local DC plus Debye realization.

For fractions f, g=1-f and media a+u/s, b+v/s (s=-i*omega),
A=f*b+g*a and B=f*v+g*u give the harmonic response
    eps_inf + sigma_dc/s + delta/(1+s*tau),
where eps_inf=a*b/A, sigma_dc=u*v/B, tau=A/B and
    delta=f*g*(a*v-b*u)**2/(A*B**2).
All storage and dissipation weights are nonnegative. Tangential components
use arithmetic epsilon and conductivity. Endpoint and zero-rate limits are
handled without dividing by zero. Only two materials can occupy a Yee voxel:
each film is at least dx thick, as required by the singleton descriptor.
"""
import numpy as np

from wavebit.fdtd._prepare import conductive_film as single


def touching(stack):
    if len(stack.films) != 2:
        return False
    left, right = sorted(stack.films, key=lambda film: film.low)
    return left.high == right.low


def profiles(stack, grid, dt):
    """One memory over the union, with exactly one pole at each cut voxel."""
    left, right = sorted(stack.films, key=lambda film: film.low)
    arrays = [np.array(v, copy=True).reshape(3, -1)
              for v in single._profiles(left, grid, dt)]
    other = [v.reshape(3, -1) for v in single._profiles(right, grid, dt)]
    eps, sigma, gain, drive, shift, inverse, rates = arrays
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for component in range(3):
                f = single._fill(left, grid, component)
                h = single._fill(right, grid, component)
                for dst, src in zip(arrays, other):
                    dst[component, h > 0] = src[component, h > 0]
                mixed = (f > 0) & (h > 0)
                fl = f[mixed]
                gl = 1-fl  # exact contact partitions the entire voxel
                a, u, b, v = left.eps, left.sigma, right.eps, right.sigma
                for value in arrays[1:]:
                    value[component, mixed] = 0.
                if component != stack.axis:
                    eps[component, mixed] = fl*a+gl*b
                    sigma[component, mixed] = fl*u+gl*v
                    continue
                A, B = fl*b+gl*a, fl*v+gl*u
                eps[component, mixed] = a*b/A
                active = B > 0
                dc = np.zeros_like(B)
                dc[active] = u*(v/B[active])
                sigma[component, mixed] = dc
                cross = a*v-b*u
                live = active & (cross != 0)
                if not np.any(live):
                    continue
                positions = np.flatnonzero(mixed)[live]
                al, bl, fg = A[live], B[live], fl[live]*gl[live]
                D = 2*al+dt*bl
                weight = fg*cross**2
                gain[component, positions] = (2*al-dt*bl)/D
                shift[component, positions] = weight*dt/(al*bl*D)
                drive[component, positions] = 4*weight*dt/(bl*D*D)
                inverse[component, positions] = al*bl*bl/weight
                rates[component, positions] = weight/(al*al*a*b)
                eps[component, positions] += shift[component, positions]
    except (FloatingPointError, OverflowError) as error:
        raise ValueError('conductive contact coefficients exceed finite float64 range; '
                         'rescale units or use less extreme material values.') from error
    if (not all(np.isfinite(v).all() for v in arrays) or np.any(eps <= 0)
            or any(np.any(v < 0) for v in (sigma, drive, shift, inverse, rates))
            or np.any(np.abs(gain) > 1)):
        raise ValueError('conductive contact coefficients are not finite and passive.')
    return tuple(single._compact(v, stack.axis) for v in arrays)


def box(stack):
    return tuple((min(f.box[k][0] for f in stack.films),
                  max(f.box[k][1] for f in stack.films)) for k in range(3))


def pole_params(stack, plan, rt):
    _, _, gain, drive, shift, inverse, _ = profiles(stack, plan.grid, plan.dt)
    cut = [slice(None)]*4
    cut[3-stack.axis] = slice(*box(stack)[stack.axis])
    cut = tuple(cut)
    array = lambda value: rt.array(value[cut], 'float64')
    return {'conductive_stack[contact]': {
        'step_gain': (array(gain),), 'hold': (rt.array(0., 'float64'),),
        'drive': (array(drive),), 'dx_over_dt': rt.array(plan.dx/plan.dt, 'float64'),
        'energy_inverse_delta': (array(inverse),), 'energy_instantaneous': (array(shift),),
    }}
