"""Explicit replay boundary for FDTD studies recorded before the API integration.

These studies describe layers top-to-bottom, z increasing into the stack, and
complex amplitudes at the cell corner. Their serialized numerical evidence is
immutable. Private periodic Yee engines still implement that coordinate system;
calling them here preserves the measured experiment, including raw field probes
and diagnostic meshes. This is a study replay helper, not public API validation.
New examples and application code must use the public wavebit entry points.

RCWA references reverse only the layer axis at input, then restore the recorded
corner phase and the old pure-1D TE/TM frame at output. Half-space labels stay
physical. This helper exposes scattering results only, not public field queries.
"""
import numpy as np

import wavebit
from wavebit._core.domain import as_material
from wavebit._core.enums import backend_of
from wavebit._core.result import Result, ResultSub


def _options(stack, incidence, options):
    options = dict(options)
    for name, carrier in (('stack', stack), ('incidence', incidence)):
        if carrier is not None:
            values = carrier.as_kwargs()
            overlap = options.keys() & values.keys()
            if overlap:
                raise ValueError(f'{name} already sets {sorted(overlap)}.')
            options.update(values)
    for name in ('psi', 'chi', 'polarization', 'kx', 'ky'):
        if options.get(name) is None:
            options.pop(name, None)
    return options


def legacy_fdtd(stack=None, incidence=None, **options):
    """Replay an archived periodic experiment in its original coordinate frame."""
    options = _options(stack, incidence, options)
    backend = options.pop('backend', None)
    if backend is None:
        from wavebit._core.fdtd import solve_fdtd
        return solve_fdtd(**options)
    from wavebit._core.scene_fdtd import solve_scene_fdtd
    return solve_scene_fdtd(backend=backend_of(backend), **options)


def archive_order(order, quantity):
    """Read a general-FDTD order in the corner phase used by historical rows."""
    value = getattr(order, quantity)
    return -value if quantity in ('r_s', 'r_p', 't_s', 't_p') and (order.m+order.n) % 2 else value


class _LegacyReference:
    def __init__(self, solver, options):
        self._solver, self._options = solver, options

    def solve(self):
        result = self._solver.solve()
        m, n = np.asarray(result.orders_u), np.asarray(result.orders_v)
        sign = np.where((n[:, None]+m[None, :]) % 2, -1., 1.)
        options = self._options
        # Until 2026-09-27 pure 1D tetm used a fixed transverse frame. Coupled
        # psi/polarization calls already used each outgoing order's own frame.
        if (options.get('tetm') is not None and len(n) == 1 and n[0] == 0
                and options.get('phi', 0.) in (None, 0.)):
            medium = as_material(options['top' if options.get('incidence_side', 'top') == 'top' else 'bot'])
            index = np.sqrt(medium.epsilon*(1. if medium.mu is None else medium.mu))
            kx = 2*np.pi/options['wavelength']*index*np.sin(options.get('theta') or 0.)
            period = options['period']
            period = period['x'] if isinstance(period, dict) else np.atleast_1d(period)[0]
            frame = np.where(np.real(kx+2*np.pi*m/period) < 0, -1., 1.)
            sign = sign*frame*(-1. if np.real(kx) < 0 else 1.)

        def convert(sub):
            if sub is None:
                return None
            values = {name: getattr(sub, name) for name in ResultSub._FIELDS}
            for name in ('r_s', 'r_p', 't_s', 't_p'):
                if values[name] is not None:
                    values[name] = values[name]*sign
            # Archived studies predate Cartesian outputs and net-entering flux.
            # Avoid attaching new-frame vectors or complete-port metadata to a
            # result deliberately expressed in the historical amplitude frame.
            values.update(r_xyz=None, t_xyz=None, power_entering=None)
            return ResultSub(**values)

        return Result(incident=convert(result.incident), s_inc=convert(result.s_inc),
                      p_inc=convert(result.p_inc), order_window=result.order_window)


def legacy_rcwa(backend='numpy', *, stack=None, incidence=None, **options):
    """RCWA scattering reference for an archived top-to-bottom study scene."""
    options = _options(stack, incidence, options)
    current = dict(options)
    for name in ('structure', 'thickness', 'mu'):
        value = current.get(name)
        if value is not None:
            current[name] = value[::-1]
    return _LegacyReference(wavebit.rcwa(backend=backend, **current), options)
