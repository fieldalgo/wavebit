"""Pictures of recorded Yee fields, using each component's own coordinates.

Matplotlib is optional and imported only when a picture is requested. A time record is
raw source-driven E or normalized H, not an intensity or a frequency-domain reconstruction.
"""
import operator
import warnings

import numpy as np

from wavebit._common.backend import eager_under_torch_compile


def _record(diagnostics, component, y, part):
    if 'field_e' not in diagnostics:
        raise ValueError('No time record: solve with record_field=N and return_diagnostics=True.')
    primary = diagnostics['field_e_component']
    component = primary if component is None else component.lower()
    if component in diagnostics.get('fields', {}):
        record = diagnostics['fields'][component]
        field = np.asarray(record['values'])
        t, z, x, yy = (np.asarray(record[k]) for k in ('t', 'z', 'x', 'y'))
    else:
        if component == primary:
            key, xkey, ykey = 'field_e', 'field_x', 'field_y'
        elif component == 'ex' and 'field_ex' in diagnostics:
            key, xkey, ykey = 'field_ex', 'field_ex_x', 'field_ex_y'
        else:
            raise ValueError(f'Component {component!r} was not recorded; primary component is {primary!r}.')
        field = np.asarray(diagnostics[key])
        t, z, x, yy = (np.asarray(diagnostics[k])
                        for k in ('field_t', 'field_z', xkey, ykey))
    transforms = {'real': np.real, 'imag': np.imag, 'abs': np.abs}
    if part not in transforms:
        raise ValueError("part must be 'real', 'imag', or 'abs' (field amplitude, not intensity).")
    if any(a.ndim != 1 or not len(a) or not np.all(np.isfinite(a))
           or np.any(np.diff(a) <= 0) for a in (t, z, x, yy)):
        raise ValueError('Recorded axes must be finite, nonempty, increasing vectors.')
    if field.shape != (len(t), len(z), len(yy), len(x)):
        raise ValueError('Recorded field shape does not match its time and component axes.')
    if not np.isfinite(y):
        raise ValueError('y must be a finite coordinate.')
    iy = int(np.argmin(np.abs(yy - y)))
    values = transforms[part](field[:, :, iy, :])
    if not np.all(np.isfinite(values)):
        raise ValueError('Recorded field contains non-finite values; inspect the solve before plotting.')
    band = np.asarray(diagnostics['wavelengths'])
    if len(t) > 1 and np.max(np.diff(t)) >= 0.5 * np.min(band):
        warnings.warn('Time record undersamples the shortest requested optical period; '
                      'reduce record_field to avoid temporal aliasing. Playback speed '
                      'cannot restore missing samples.', UserWarning, stacklevel=3)
    return values, t, z, x, yy[iy], component


def _draw(diagnostics, data, frame, part, ax, vmax):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('FDTD field plots need matplotlib installed.') from exc
    values, t, z, x, y, component = data
    frame = operator.index(frame)
    if not -len(t) <= frame < len(t):
        raise IndexError(f'frame must index one of the {len(t)} recorded snapshots.')
    if vmax is None:
        vmax = float(np.max(np.abs(values))) or 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError('vmax must be finite and positive.')
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    fig = ax.figure
    # Nearest shading forms cell edges from these measured sample coordinates. It neither
    # assumes a uniform z mesh nor substitutes pixel centres for a staggered component.
    mesh = ax.pcolormesh(x, z, values[frame], shading='nearest',
                         cmap='magma' if part == 'abs' else 'RdBu_r',
                         vmin=0 if part == 'abs' else -vmax, vmax=vmax)
    name = component.capitalize()
    label = {'real': f'Re {name}', 'imag': f'Im {name}', 'abs': f'|{name}|'}[part]
    unit = 'Z0 H, source units' if component.startswith('h') else 'source units'
    fig.colorbar(mesh, ax=ax, label=f'{label} ({unit})')
    grid = diagnostics.get('grid')
    if grid is None or 'pml_bounds' in diagnostics:
        low, inner_low, inner_high, high = diagnostics['pml_bounds']
        source = diagnostics['source_z']
        source_label = ('Current sheet (total field on both sides)' if grid is None
                        else 'TFSF (total field toward the structure)')
        layer_planes = diagnostics['layer_planes']
    else:
        npml = diagnostics['n_pml']
        boundary_z = np.asarray(diagnostics['field_z'])
        low, inner_low = boundary_z[0], boundary_z[npml]
        inner_high, high = boundary_z[-1-npml], boundary_z[-1]
        kt = grid.k_tfsf
        source = .5*(boundary_z[kt-1]+boundary_z[kt])
        source_label = 'TFSF (scattered field above)'
        layer_planes = np.r_[0., np.cumsum(grid.thickness)]
    ax.axhspan(low, inner_low, color='gray', alpha=.2, label='PML')
    ax.axhspan(inner_high, high, color='gray', alpha=.2)
    ax.axhline(source, color='darkorange', linestyle='--', linewidth=1.2, label=source_label)
    for i, depth in enumerate(layer_planes):
        ax.axhline(depth, color='black', linewidth=0.7, linestyle=':',
                   label='Layer planes' if i == 0 else None)
    ax.set(xlabel='x (length units)', ylabel='z (length units)', ylim=((z[0], z[-1]) if 'coordinate_convention' in diagnostics else (z[-1], z[0])))
    ax.legend(loc='upper left', fontsize='small')
    title = ax.set_title(_title(component, part, t[frame], y))
    return fig, ax, mesh, title


def _title(component, part, t, y):
    return f'{component.capitalize()} ({part}) · t={t:.5g} (c=1) · y={y:.5g}'


@eager_under_torch_compile
def plot_fdtd_field(diagnostics, *, frame=0, component=None, y=0.0, part='real',
                    ax=None, vmax=None):
    """Plot one recorded x-z electric or magnetic field slice; return ``(figure, axes, mesh)``.

    Args:
        diagnostics: The dictionary from ``fdtd(record_field=N, return_diagnostics=True)``.
        frame: Snapshot index, including negative indices. Default is the first snapshot.
        component: ``'ey'`` or ``'ex'`` when recorded; default is ``field_e_component``.
        y: Physical y coordinate. Selects the nearest recorded row on this component's grid;
            the title reports the actual coordinate. An invariant-y record has one row.
        part: ``'real'`` (default), ``'imag'``, or ``'abs'``. These are E, not intensity.
        ax: Optional Matplotlib axes. Matplotlib is needed only when this function is called.
        vmax: Positive color limit. By default the whole selected slice's time history sets
            one fixed limit, so separate frames remain comparable.

    Uses the recorded staggered coordinates and nonuniform z mesh. PML and layer planes
    are marked; above TFSF only the scattered field is recorded. Time is in the solver's
    c=1 units (the same units as lengths). Warns if snapshots undersample the shortest
    requested optical period; a smaller ``record_field`` is needed to recover motion.
    """
    data = _record(diagnostics, component, y, part)
    fig, ax, mesh, _ = _draw(diagnostics, data, frame, part, ax, vmax)
    return fig, ax, mesh


@eager_under_torch_compile
def animate_fdtd_field(diagnostics, *, component=None, y=0.0, part='real',
                       interval=50, ax=None, vmax=None):
    """Animate a recorded x-z slice; return a Matplotlib ``FuncAnimation``.

    Selection, coordinates, units and color limits follow :func:`plot_fdtd_field`.
    ``interval`` is playback milliseconds per snapshot, not simulated time. Keep the
    returned object alive, then use ``plt.show()``, ``animation.to_jshtml()`` in a notebook,
    or ``animation.save(...)``. GIF saving can use Matplotlib's Pillow writer; MP4 needs
    an installed video writer. This function does not display or save anything itself.
    """
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError('interval must be finite and positive.')
    data = _record(diagnostics, component, y, part)
    _, ax, mesh, title = _draw(diagnostics, data, 0, part, ax, vmax)
    from matplotlib.animation import FuncAnimation
    values, t, _, _, actual_y, component = data

    def update(i):
        mesh.set_array(values[i])
        title.set_text(_title(component, part, t[i], actual_y))
        return mesh, title

    return FuncAnimation(ax.figure, update, frames=len(t), interval=interval,
                         blit=False, cache_frame_data=False)
