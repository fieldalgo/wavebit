"""Point probes on native Yee records, with explicit space/time interpolation."""
import warnings

import numpy as np


_COMPONENTS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')


def _coordinate(value, name):
    raw = np.asarray(value)
    if raw.ndim or np.iscomplexobj(raw):
        raise ValueError(f'{name} must be a finite real scalar coordinate.')
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f'{name} must be a finite real scalar coordinate.')
    return result


def _bracket(axis, value, name):
    if value < axis[0] or value > axis[-1]:
        raise ValueError(f'{name}={value} is outside the recorded interval [{axis[0]}, {axis[-1]}].')
    upper = int(np.searchsorted(axis, value, side='left'))
    if axis[upper] == value:
        return upper, upper, 0.
    lower = upper-1
    return lower, upper, (value-axis[lower])/(axis[upper]-axis[lower])


def sample_fdtd_fields(diagnostics, *, z, x=0., y=0., t=None):
    """Sample all six transient fields at one physical point and common times.

    Requires ``record_components='all'`` and ``return_diagnostics=True``. Linear
    interpolation uses each component's own Yee x/y/z coordinates and timestamp,
    including graded z and the magnetic half-step. No extrapolation or periodic
    wrapping is performed. A singleton axis accepts only its recorded coordinate.

    Args:
        diagnostics: dictionary returned by :func:`wavebit.fdtd`.
        z, x, y: point in the scene's length units; x/y are cell-centred.
        t: scalar or increasing time vector, in c=1 length units. By default use
            Ex times within the intersection of all six recorded time intervals.

    Returns:
        dict: ``t``, point coordinates, complex ``E`` and normalized ``H=Z0 H_SI``
        arrays shaped (time, 3), with columns x/y/z. ``region`` identifies total
        or scattered fields when the diagnostics includes the source grid.

    Sampling across the TFSF discontinuity is refused. Linear interpolation cannot
    recover an undersampled pulse and warns if the record violates temporal Nyquist.
    At material interfaces interpolation accuracy requires spatial refinement.
    """
    point = {name: _coordinate(value, name) for name, value in (('x', x), ('y', y), ('z', z))}
    records = diagnostics.get('fields', {})
    if any(name not in records for name in _COMPONENTS):
        raise ValueError("Point probes require all six fields; solve with record_components='all'.")
    grid = diagnostics.get('grid')
    if (grid is not None and (getattr(grid, 'kx_inc', 0.) != 0. or getattr(grid, 'ky_inc', 0.) != 0.)
            and diagnostics.get('field_temporal_representation') != 'positive_frequency_carrier'):
        raise ValueError('Bloch probes require positive-frequency-carrier records; regenerate the time record.')
    source_z = diagnostics.get('source_z') if grid is None else .5*(grid.z_node[grid.k_tfsf-1]+grid.z_node[grid.k_tfsf])
    traces, times = [], []
    for name in _COMPONENTS:
        record = records[name]
        if any(np.iscomplexobj(record[a]) for a in ('t', 'z', 'y', 'x')):
            raise ValueError(f'{name} axes must be real.')
        axes = {a: np.asarray(record[a], dtype=float) for a in ('t', 'z', 'y', 'x')}
        for axis_name, axis in axes.items():
            if (axis.ndim != 1 or not len(axis) or not np.isfinite(axis).all()
                    or np.any(np.diff(axis) <= 0)):
                raise ValueError(f'{name}.{axis_name} must be a finite increasing nonempty axis.')
        values = np.asarray(record['values'])
        if values.shape != tuple(len(axes[a]) for a in ('t', 'z', 'y', 'x')):
            raise ValueError(f'{name} values do not match their t/z/y/x axes.')
        # Interpolate only the eight neighbouring spatial samples, never copy or
        # interpolate the full recorded volume onto another 4D grid.
        brackets = {a: _bracket(axes[a], point[a], f'{name}.{a}') for a in ('z', 'y', 'x')}
        lower, upper, _ = brackets['z']
        if source_z is not None and (point['z'] == source_z or
                axes['z'][lower] <= source_z <= axes['z'][upper]):
            boundary = 'source' if grid is None else 'TFSF'
            raise ValueError(f'The point interpolation crosses the {boundary} discontinuity; choose a point farther from it.')
        trace = np.zeros(len(axes['t']), dtype=complex)
        for iz, wz in ((brackets['z'][0], 1-brackets['z'][2]), (brackets['z'][1], brackets['z'][2])):
            for iy, wy in ((brackets['y'][0], 1-brackets['y'][2]), (brackets['y'][1], brackets['y'][2])):
                for ix, wx in ((brackets['x'][0], 1-brackets['x'][2]), (brackets['x'][1], brackets['x'][2])):
                    weight = wz*wy*wx
                    if weight:
                        trace += weight*values[:, iz, iy, ix]
        if not np.isfinite(trace).all():
            raise ValueError(f'{name} contains non-finite samples around the requested point.')
        traces.append(trace)
        times.append(axes['t'])
    start, stop = max(a[0] for a in times), min(a[-1] for a in times)
    if t is None:
        target = times[0][(times[0] >= start) & (times[0] <= stop)]
    else:
        raw = np.asarray(t)
        if np.iscomplexobj(raw):
            raise ValueError('t must contain real times.')
        target = np.atleast_1d(np.asarray(t, dtype=float))
    if (target.ndim != 1 or not len(target) or not np.isfinite(target).all()
            or np.any(np.diff(target) <= 0)):
        raise ValueError('t must be a finite increasing nonempty time vector within the common record interval.')
    if target[0] < start or target[-1] > stop:
        raise ValueError(f't is outside the common record interval [{start}, {stop}].')
    band = np.asarray(diagnostics.get('wavelengths', []))
    if band.size and any(len(a) > 1 and np.max(np.diff(a)) >= .5*np.min(band) for a in times):
        warnings.warn('Time record undersamples the optical period; interpolation cannot recover '
                      'missing oscillations. Reduce record_field.', UserWarning, stacklevel=2)
    sampled = np.stack([np.interp(target, axis, trace) for axis, trace in zip(times, traces)], axis=1)
    region = ('total' if grid is None and source_z is not None else
              None if source_z is None else ('total' if point['z'] > source_z else 'scattered'))
    return dict(t=target, **point, E=sampled[:, :3], H=sampled[:, 3:], region=region)


def fdtd_poynting(diagnostics, *, z, x=0., y=0., t=None):
    """Compute instantaneous energy flow at a point after Yee interpolation.

    Returns :func:`sample_fdtd_fields` output plus real ``S`` shaped (time, 3):
    ``cross(E.real, H.real)``. There is no conjugation or factor 1/2: this is an
    instantaneous physical field, not a harmonic time average. H is normalized
    as Z0 H, so S is in squared source-electric-field units (Z0 times SI flux
    if the electric source amplitude is specified in V/m), not absolute W/m².
    Positive Sz points down into the stack. Above TFSF this is scattered flux.
    """
    sampled = sample_fdtd_fields(diagnostics, z=z, x=x, y=y, t=t)
    sampled['S'] = np.cross(sampled['E'].real, sampled['H'].real)
    return sampled
