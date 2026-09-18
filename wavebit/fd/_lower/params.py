"""``lower(scene, plan, rt) -> Params``: the traced side of a run, as a pytree.

``Params`` is a plain dict, because a dict is a pytree on JAX and needs no registration on
Torch or NumPy. Its leaves are the numbers a gradient may flow through:

    ae, be, ah, bh    the update coefficients -- 1, dt/dx, 1, dt/dx in vacuum
    bloch             {axis: exp(i*phi)} for each Bloch axis (the low face conjugates it)
    amplitude         the InitialField amplitude, cast to the run's dtype
    E0, H0            the initial fields as (3, nz, ny, nx) arrays, or None

Whether ``E0`` or ``H0`` is ``None`` is static and recorded on the plan (``has_E0``,
``has_H0``); their values are leaves. That is the pytree rule this repository keeps
everywhere: branch on None-versus-present, never on a value.
"""
import numpy as np

from wavebit._core.backend import check_input_precision

from ..boundaries import AXES, Bloch
from .fields import evaluate

__all__ = ['lower']


def lower(scene, plan, rt='numpy', device=None):
    """The traced parameters of ``scene`` under ``plan``, as arrays of runtime ``rt``.

    ``rt`` is a runtime object, or a backend name / ``wavebit.Backend`` to build one from.
    """
    if not hasattr(rt, 'slab_add'):
        from .._runtime import runtime
        rt = runtime(rt, device)
    grid = plan.grid
    real = 'float64'
    params = {
        'ae': rt.array(1.0, real), 'be': rt.array(plan.dt / plan.dx, real),
        'ah': rt.array(1.0, real), 'bh': rt.array(plan.dt / plan.dx, real),
        'bloch': {}, 'amplitude': None, 'E0': None, 'H0': None,
    }
    for a, axis in enumerate(AXES):
        face = getattr(scene.boundaries, axis)[0]
        if isinstance(face, Bloch):
            given = face.phase if face.k is None else face.k
            check_input_precision(f'Bloch on {axis}', given, np.float64)
            phase = given if face.k is None else face.k * grid.size[a]
            params['bloch'][axis] = rt.xp.exp(1j * rt.array(phase, real))
    initial = scene.initial
    if initial is None:
        params['amplitude'] = rt.array(1.0, plan.dtype)
        return params
    amp = initial.amplitude
    if plan.dtype == 'float64' and (isinstance(amp, complex)
                                    or 'complex' in str(getattr(amp, 'dtype', ''))):
        raise ValueError('InitialField(amplitude=...) is complex but the run is real (no Bloch '
                         'face).')
    # A leaf handed in below float64 -- a jnp scalar built before x64 was lifted is the
    # usual case -- is widened here without recovering anything; say so.
    check_input_precision('InitialField.amplitude', amp, np.float64)
    for name, value in (('InitialField.E', initial.E), ('InitialField.H', initial.H)):
        if hasattr(value, 'dtype'):
            check_input_precision(name, value, np.float64)
    params['amplitude'] = rt.array(amp, plan.dtype)
    params['E0'] = evaluate('E', initial.E, grid, 'E', rt, plan.dtype)
    params['H0'] = evaluate('H', initial.H, grid, 'H', rt, plan.dtype)
    return params
