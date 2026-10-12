"""Streaming Fourier field planes on the existing Yee grid."""
import numpy as np


_NAMES = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
_LAYOUT = {'ex': (False, True, False), 'ey': (False, False, True),
           'ez': (True, False, False), 'hx': (True, False, True),
           'hy': (True, True, False), 'hz': (False, True, True)}


def _back_half(values, axis, bloch):
    previous = np.roll(values, 1, axis=axis).copy()
    edge = [slice(None)]*values.ndim
    edge[axis] = 0
    previous[tuple(edge)] /= bloch
    return .5*(values+previous)


def plane_flux(fields, cell_area):
    """H is normalized as Z0 H; the phasor flux includes the cycle-average half."""
    electric = np.stack([fields[n] for n in _NAMES[:3]], axis=-1)
    magnetic = np.stack([fields[n] for n in _NAMES[3:]], axis=-1)
    density = .5*np.real(np.cross(electric, np.conj(magnetic)))
    return density, density[..., 2].mean(axis=(-2, -1))*cell_area


class PlaneDFT:
    def __init__(self, grid, planes, wavelengths):
        self.grid = grid
        self.wavelengths = np.asarray(wavelengths)
        self.planes = {}
        self.source = np.zeros(len(wavelengths), complex)
        boundary = .5*(grid.z_node[grid.k_tfsf-1]+grid.z_node[grid.k_tfsf])
        for label, z in planes.items():
            if not isinstance(label, str) or not label:
                raise ValueError('field_monitors needs nonempty string names.')
            if np.ndim(z) or np.iscomplexobj(z) or not np.isfinite(z):
                raise ValueError('Monitor z must be a finite real coordinate.')
            brackets = {}
            for name, (half, _, _) in _LAYOUT.items():
                axis = (grid.z_node[:-1]+grid.z_node[1:])/2 if half else grid.z_node
                if z < axis[0] or z > axis[-1]:
                    raise ValueError(f'Monitor {label} lies outside the common field domain.')
                hi = int(np.searchsorted(axis, z))
                lo = hi if axis[hi] == z else hi-1
                weight = 0. if lo == hi else (z-axis[lo])/(axis[hi]-axis[lo])
                if axis[lo] <= boundary <= axis[hi]:
                    raise ValueError(f'Monitor {label} interpolation touches the TFSF discontinuity.')
                brackets[name] = lo, hi, weight
            self.planes[label] = dict(z=float(z), brackets=brackets,
                                     region='total' if z > boundary else 'scattered',
                                     fields={n: np.zeros((len(wavelengths), grid.ny, grid.nx), complex)
                                             for n in _NAMES})

    def update(self, active, step, pulse):
        dt = self.grid.dt
        electric = np.exp(2j*np.pi*(step+1)*dt/self.wavelengths)*dt
        magnetic = np.exp(2j*np.pi*(step+.5)*dt/self.wavelengths)*dt
        self.source += electric*pulse
        for plane in self.planes.values():
            for name, field in active.items():
                lo, hi, weight = plane['brackets'][name]
                sample = (1-weight)*field[lo]+weight*field[hi]
                kernel = magnetic if name.startswith('h') else electric
                plane['fields'][name] += kernel[:, None, None]*sample

    def result(self):
        g = self.grid
        if (not np.isfinite(self.source).all() or not np.any(abs(self.source))
                or np.any(abs(self.source) < 1e-12*max(abs(self.source)))):
            raise ValueError('Field monitor source spectrum is unresolved at a requested wavelength.')
        result = {}
        area = g.period*(g.period_y if g.ny > 1 else 1.)
        for label, plane in self.planes.items():
            fields = {}
            for name, value in plane['fields'].items():
                _, xh, yh = _LAYOUT[name]
                if xh:
                    value = _back_half(value, 2, g.bloch)
                if yh and g.ny > 1:
                    value = _back_half(value, 1, g.bloch_y)
                fields[name] = value/self.source[:, None, None]
            density, power = plane_flux(fields, area)
            result[label] = dict(z=plane['z'], region=plane['region'], fields=fields,
                                 wavelengths=self.wavelengths, source_spectrum=self.source.copy(),
                                 x=np.arange(g.nx)*g.dx-.5*g.period,
                                 y=np.arange(g.ny)*g.dy-.5*g.period_y if g.ny > 1 else np.zeros(1),
                                 normalization='soft-source spectrum', cell_area=area,
                                 flux_density=density, power_z=power,
                                 per_unit_y=g.ny == 1)
        return result


def combine_monitors(left, right, a, b):
    result = {}
    for name, plane in left.items():
        fields = {n: a*v+b*right[name]['fields'][n] for n, v in plane['fields'].items()}
        density, power = plane_flux(fields, plane['cell_area'])
        result[name] = dict(plane, fields=fields, flux_density=density, power_z=power)
    return result
