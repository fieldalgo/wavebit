"""Evaluate an ``InitialField`` on the grid: callables at the Yee points, arrays checked.

The result is one ``(3, nz, ny, nx)`` array of the run's dtype on the run's backend. A
callable is given that component's three coordinate grids, each ``(nz, ny, nx)``; whatever
it returns is broadcast to the component's shape, so a function of ``x`` alone is fine. A
value that is already a backend array is converted with the graph kept, so a traced
pattern can be differentiated; the cheap way to get a gradient with respect to the
excitation is still ``InitialField(amplitude=...)``.

A complex value on a real run is refused rather than truncated: NumPy would drop the
imaginary part with a warning and JAX and Torch would each do something else.
"""

__all__ = ['evaluate']


def _is_complex(v):
    if isinstance(v, complex):
        return True
    dtype = getattr(v, 'dtype', None)
    return dtype is not None and 'complex' in str(dtype)


def _component(name, c, entry, grid, field, rt, dtype):
    shape = (grid.n[2], grid.n[1], grid.n[0])
    if entry is None:
        return rt.zeros(shape, dtype)
    if callable(entry):
        X, Y, Z = grid.meshgrid(field, c)
        entry = entry(X, Y, Z)
    if dtype == 'float64' and _is_complex(entry):
        raise ValueError(
            f'InitialField({name}=...): the {"xyz"[c]} component is complex but the run is '
            f'real (no Bloch face). A complex initial field needs a Bloch boundary, or take '
            f'its real part.')
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


def evaluate(name, value, grid, field, rt, dtype):
    """``value`` (an ``InitialField.E`` or ``.H`` entry) as a ``(3, nz, ny, nx)`` array, or None."""
    if value is None:
        return None
    shape = (3, grid.n[2], grid.n[1], grid.n[0])
    if hasattr(value, 'shape') and not callable(value):
        if tuple(value.shape) != shape:
            raise ValueError(f'InitialField({name}=array): the grid is (3, nz, ny, nx) = {shape}; '
                             f'the array is {tuple(value.shape)}.')
        if dtype == 'float64' and _is_complex(value):
            raise ValueError(f'InitialField({name}=array) is complex but the run is real (no '
                             f'Bloch face); take its real part or give a Bloch boundary.')
        return rt.array(value, dtype)
    entries = (value, value, value) if callable(value) else tuple(value)
    parts = [_component(name, c, e, grid, field, rt, dtype) for c, e in enumerate(entries)]
    return rt.xp.stack(parts, axis=0)
