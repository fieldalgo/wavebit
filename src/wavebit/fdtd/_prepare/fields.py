"""Evaluate an ``InitialField`` on the grid: callables at the Yee points, arrays checked.

The result is one ``(3, nz, ny, nx)`` array of the run's dtype on the run's backend. A
callable is given that component's three coordinate grids, each ``(nz, ny, nx)``; whatever
it returns is broadcast to the component's shape, so a function of ``x`` alone is fine. A
value that is already a backend array is converted with the graph kept, so a traced
pattern can be differentiated; the cheap way to get a gradient with respect to the
excitation is still ``InitialField(amplitude=...)``. Public Torch CUDA execution
scopes a stricter rule: evaluated patterns must be static host values; their
amplitude remains live. CPU, JAX and internal execution keep the graph above.

A complex value on a real run is refused rather than truncated: NumPy would drop the
imaginary part with a warning and JAX and Torch would each do something else.

Along an ``fd.Axis1D`` axis (stage 3) there is no stagger: the grid holds one cell of ``F exp(i
k a)`` with one phase reference for every component, so a callable is given that axis's low
coordinate for every component (``flat=``), not the Yee offset a differenced axis would carry.
"""

import contextlib
import contextvars

__all__ = ['evaluate']

_STATIC_HOST_PATTERNS = contextvars.ContextVar('wavebit_fdtd_static_host_patterns', default=False)


@contextlib.contextmanager
def static_host_patterns(enabled):
    """Scope the public CUDA pattern contract without changing legacy execution."""
    token = _STATIC_HOST_PATTERNS.set(enabled)
    try:
        yield
    finally:
        _STATIC_HOST_PATTERNS.reset(token)


def _check_static_host_pattern(name, value):
    """Inspect evaluated values without calling a user function or reading a tensor."""
    if not _STATIC_HOST_PATTERNS.get():
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _check_static_host_pattern(name, item)
        return
    from wavebit._common.backend import carries_autodiff
    from wavebit._common.declarations import torch_carried
    from wavebit.errors import IncompatibleProblemError, UnsupportedFeatureError
    if torch_carried(value) or carries_autodiff(value):
        raise IncompatibleProblemError(
            'CUDA initial field patterns must be static host values; use '
            'InitialField.amplitude for a live excitation scale',
            code='static_input', path=f'initial.{name}')
    device = getattr(value, 'device', None)
    if device is not None:
        kind = getattr(device, 'type', getattr(device, 'platform', device))
        if str(kind).split(':')[0] != 'cpu':
            raise UnsupportedFeatureError(
                'CUDA initial field patterns must be host values, including callable results',
                code='cuda_scope', path=f'initial.{name}')


def _is_complex(v):
    if isinstance(v, complex):
        return True
    dtype = getattr(v, 'dtype', None)
    return dtype is not None and 'complex' in str(dtype)


def _component(name, c, entry, grid, field, rt, dtype, flat=(), concrete_only=False):
    shape = (grid.n[2], grid.n[1], grid.n[0])
    if entry is None:
        return rt.zeros(shape, dtype)
    if callable(entry):
        coords = list(grid.meshgrid(field, c))
        for a in flat:
            coords[a] = coords[a] * 0.0 + grid.low[a]
        # The policy constrains this callable's result, not work it invokes.
        # Nested public calls select their own policy; internal calls retain
        # their original behavior. Restore the outer policy before validation.
        with static_host_patterns(False):
            entry = entry(*coords)
    if concrete_only:
        from wavebit.fdtd._decay import reject_carried
        reject_carried(entry)
    if dtype == 'float64' and _is_complex(entry):
        raise ValueError(
            f'InitialField({name}=...): the {"xyz"[c]} component is complex but the run is '
            f'real (no Bloch face). A complex initial field needs a Bloch boundary, or take '
            f'its real part.')
    _check_static_host_pattern(name, entry)
    x = rt.array(entry, dtype)
    if tuple(x.shape) != shape:
        try:
            x = rt.xp.broadcast_to(x, shape)
        except Exception as err:
            raise ValueError(
                f'InitialField({name}=...): the {"xyz"[c]} component has shape '
                f'{tuple(x.shape)}, which does not broadcast to the grid (nz, ny, nx) = '
                f'{shape}.') from err
    return x


def evaluate(name, value, grid, field, rt, dtype, flat=(), concrete_only=False):
    """``value`` (an ``InitialField.E`` or ``.H`` entry) as a ``(3, nz, ny, nx)`` array, or None.

    ``flat`` are the ``Axis1D`` axes (0, 1, 2), evaluated at their low coordinate.
    """
    if value is None:
        return None
    if concrete_only:
        from wavebit.fdtd._decay import reject_carried
        reject_carried(value)
    shape = (3, grid.n[2], grid.n[1], grid.n[0])
    if hasattr(value, 'shape') and not callable(value):
        if tuple(value.shape) != shape:
            raise ValueError(f'InitialField({name}=array): the grid is (3, nz, ny, nx) = {shape}; '
                             f'the array is {tuple(value.shape)}.')
        if dtype == 'float64' and _is_complex(value):
            raise ValueError(f'InitialField({name}=array) is complex but the run is real (no '
                             f'Bloch face); take its real part or give a Bloch boundary.')
        _check_static_host_pattern(name, value)
        return rt.array(value, dtype)
    entries = (value, value, value) if callable(value) else tuple(value)
    parts = [_component(name, c, e, grid, field, rt, dtype, flat, concrete_only)
             for c, e in enumerate(entries)]
    return rt.xp.stack(parts, axis=0)
