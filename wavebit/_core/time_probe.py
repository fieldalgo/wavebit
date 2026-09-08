"""Point probes and offline mode analysis of native Yee time records."""
import warnings

import numpy as np

from .conventions import time_pole_parameters

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


def fit_fdtd_modes(t, signal, *, modes=1, start=None, stop=None, pencil=None, rcond=1e-10):
    """Fit a point ringdown to a finite sum of damped complex exponentials.

    The model is ``sum(a * exp((-decay_rate-2j*pi*frequency)*(t-time_origin)))``.
    Frequencies are cycles per c=1 time unit; positive decay_rate means amplitude
    decay, and Q=pi*abs(frequency)/decay_rate. Growing modes are retained, with
    NaN Q for nonpositive decay. Amplitudes refer to the first selected sample,
    not the simulation's t=0. No positivity, conjugacy or decay constraint is
    imposed. A real cosine needs two modes (the conjugate-frequency pair).

    Args:
        t: real, finite, increasing, uniformly spaced one-dimensional times.
        signal: one real or complex trace of the same length. Select a point
            and component from native records or :func:`sample_fdtd_fields`.
        modes: positive integer number of exponentials; at least 2*modes+1
            selected samples are required. Increase it only with independent
            window/order checks; noise can produce spurious modes.
        start, stop: optional inclusive time bounds. Select a source-free
            interval; this function cannot check whether the source has stopped.
        pencil: Hankel column count, between modes and samples-modes. Default
            min(samples//2, max(128, 2*modes)) bounds the default matrix width.
        rcond: relative singular-value floor in (0,1). A requested mode count
            beyond the resolved Hankel rank is refused.

    Returns:
        dict: frequency-sorted ``frequency``, ``decay_rate``, ``quality_factor``,
        ``amplitude``, ``complex_frequency`` and ``discrete_pole`` arrays;
        ``time_origin``, ``dt``, selected ``t`` and ``fitted`` trace;
        ``relative_residual``, normalized ``singular_values``, ``basis_condition``,
        ``modes`` and ``pencil``. The residual measures the selected record only;
        it is not a mode-frequency uncertainty or a convergence certificate.

    This is NumPy CPU postprocessing, not a differentiable/JIT operation. Move
    backend records to host arrays explicitly (detach Torch records if needed).
    Storage is O(samples*pencil). No resampling or anti-alias filter is applied:
    frequencies are reported on the signed sampled Nyquist branch. Undersampled
    modes cannot be recovered. Weak damping and close modes require window,
    sampling, model-order and physical boundary/grid checks before interpreting Q.

    Independently implemented from the matrix-pencil construction of Hua and
    Sarkar, IEEE TASSP 38 (1990), 814-824, doi:10.1109/29.56027, section II.
    No external solver or harmonic-inversion implementation is used.
    """
    times, values = np.asarray(t), np.asarray(signal)
    if (times.ndim != 1 or times.dtype.kind not in 'fiu' or len(times) < 3
            or not np.isfinite(times).all() or np.any(np.diff(times.astype(float)) <= 0)):
        raise ValueError('t must contain at least three finite increasing real times.')
    if (values.ndim != 1 or values.shape != times.shape or values.dtype.kind not in 'fciu'
            or not np.isfinite(values).all()):
        raise ValueError('signal must be one finite numeric trace with the same shape as t.')
    if isinstance(modes, (bool, np.bool_)) or not isinstance(modes, (int, np.integer)) or modes < 1:
        raise ValueError('modes must be a positive integer.')
    floor = np.asarray(rcond)
    if floor.shape != () or floor.dtype.kind not in 'fiu':
        raise ValueError('rcond must be a finite real scalar strictly between zero and one.')
    floor = float(floor)
    if not np.isfinite(floor) or not 0 < floor < 1:
        raise ValueError('rcond must be a finite real scalar strictly between zero and one.')
    lower = times[0] if start is None else _coordinate(start, 'start')
    upper = times[-1] if stop is None else _coordinate(stop, 'stop')
    if lower > upper or lower < times[0] or upper > times[-1]:
        raise ValueError('start/stop must select an ordered interval inside the recorded times.')
    selected = (times >= lower) & (times <= upper)
    times, values = times[selected].astype(float), values[selected].astype(complex)
    count = len(times)
    if count < 2*modes+1:
        raise ValueError('The selected interval needs at least 2*modes+1 samples.')
    dt = float((times[-1]-times[0])/(count-1))
    if not np.allclose(np.diff(times), dt, rtol=1e-7, atol=1e-12*dt):
        raise ValueError('Selected t must be uniformly spaced; no resampling is performed.')
    if pencil is None:
        pencil = min(count//2, max(128, 2*modes))
    if (isinstance(pencil, (bool, np.bool_)) or not isinstance(pencil, (int, np.integer))
            or not modes <= pencil <= count-modes):
        raise ValueError('pencil must be an integer between modes and samples-modes.')
    scale = np.max(np.abs(values))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError('signal needs a finite nonzero amplitude scale.')
    normalized = values/scale
    windows = np.lib.stride_tricks.sliding_window_view(normalized, pencil+1)
    h0, h1 = windows[:, :-1], windows[:, 1:]
    u, singular, vh = np.linalg.svd(h0, full_matrices=False)
    if singular[modes-1] <= floor*singular[0]:
        raise ValueError('Requested modes exceed the resolved Hankel rank; reduce modes or inspect rcond and the signal.')
    # The nonzero eigenvalues of the truncated pseudoinverse H0^+ H1
    # can be found in this modes-by-modes projection (Hua/Sarkar eq. 2.17).
    reduced = (u[:, :modes].conj().T @ h1 @ vh[:modes].conj().T)/singular[:modes, None]
    poles = np.linalg.eigvals(reduced)
    if not np.isfinite(poles).all() or np.any(poles == 0):
        raise ValueError('The fitted discrete poles cannot represent finite continuous frequencies; reduce modes or change the interval.')
    exponents, frequency, decay = time_pole_parameters(np, poles, dt)
    order = np.argsort(frequency)
    poles, exponents, frequency, decay = (v[order] for v in (poles, exponents, frequency, decay))
    # Anchor a growing column at the last sample, keeping every exponential's
    # magnitude <=1. This avoids overflowing the least-squares basis.
    anchor = np.where(exponents.real > 0, count-1, 0)
    basis = np.exp((np.arange(count)[:, None]-anchor)*exponents)
    coefficients, _, rank, basis_singular = np.linalg.lstsq(basis, normalized, rcond=floor)
    if rank < modes:
        raise ValueError('The fitted modal amplitudes are unresolved; reduce modes or change the interval.')
    fitted = basis @ coefficients
    amplitude = coefficients*np.exp(-anchor*exponents)*scale
    quality = np.full(modes, np.nan)
    decaying = decay > 0
    quality[decaying] = np.pi*np.abs(frequency[decaying])/decay[decaying]
    return dict(frequency=frequency, decay_rate=decay, quality_factor=quality, amplitude=amplitude,
                complex_frequency=frequency-1j*decay/(2*np.pi), discrete_pole=poles,
                time_origin=float(times[0]), dt=dt, t=times, fitted=fitted*scale,
                relative_residual=float(np.linalg.norm(normalized-fitted)/np.linalg.norm(normalized)),
                singular_values=singular/singular[0], basis_condition=float(basis_singular[0]/basis_singular[-1]),
                modes=int(modes), pencil=int(pencil))
