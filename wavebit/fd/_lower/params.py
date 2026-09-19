"""``lower(scene, plan, rt) -> Params``: the traced side of a run, as a pytree.

``Params`` is a plain dict, because a dict is a pytree on JAX and needs no registration on
Torch or NumPy. Its leaves are the numbers a gradient may flow through:

    ae, be, ah, bh    the update coefficients -- 1, dt/dx, 1, dt/dx in vacuum, arrays once a
                      material or an Absorber makes them so (``coefficients.py``)
    minus_bh          ``-bh``, lowered once because negating an array in the step body is a
                      full-volume temporary per step (``coefficients.build`` says what that
                      measured)
    bloch             {axis: exp(i*phi)} for each Bloch axis (the low face conjugates it)
    amplitude         the InitialField amplitude, cast to the run's dtype
    E0, H0            the initial fields as (3, nz, ny, nx) arrays, or None
    cpml              {psi key: {'b', 'a', 'ik'}} -- the CPML profiles, 1-D along the
                      absorbing axis and reshaped to broadcast, never at slab shape
    poles             {term name: {'step_gain', 'hold', 'drive', 'dx_over_dt'}} per pole
    sources           {source name: {'amplitude', 'phase', 'delay', 'f0', 'tau', 't0', ...}}
                      per stage-2 source, with a Dipole's 'stamp', a column's own params
                      ('column', and 'driver' for a layered TFSF column's uniform drive) and
                      transverse Bloch 'pattern' (``wavebit/fd/_lower/sources.py``); the
                      waveform table built from them is added inside the traced function

Whether ``E0`` or ``H0`` is ``None`` is static and recorded on the plan (``has_E0``,
``has_H0``); their values are leaves. That is the pytree rule this repository keeps
everywhere: branch on None-versus-present, never on a value.

**The stability guard.** A material value may be traced, and a traced value cannot be
compared against the CFL bound at lowering time. When the bound cannot be read, ``be`` and
``bh`` are multiplied by a factor that is ``1`` if the run is stable and ``nan`` if it is
not, so an unstable run returns ``nan`` arrays rather than a number shaped like a field --
the stage-0 rule for a condition only run time can know. When every value *is* concrete the
bound was checked at the door (``checks.check_stability``) and no guard is built, so a vacuum
run's coefficients are the stage-0 ones to the bit.
"""
import math

import numpy as np

from wavebit._core.backend import check_input_precision

from ..boundaries import AXES, Bloch
from ..materials import as_material
from .._kernel.cpml import psi_key
from . import checks
from .coefficients import build as build_coefficients
from .cpml import face_eps_r, profiles
from .fields import evaluate
from .sources import lower_sources

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
    params = dict(build_coefficients(scene, plan, rt))
    params.update({'bloch': {}, 'amplitude': None, 'E0': None, 'H0': None,
                   'cpml': _cpml(scene, plan, rt), 'poles': _poles(scene, plan, rt),
                   'sources': lower_sources(scene, plan, rt)})
    guard = _stability_guard(scene, plan, rt)
    if guard is not None:
        params['be'] = params['be'] * guard
        params['bh'] = params['bh'] * guard
        # Rebuilt rather than scaled, so that ``minus_bh`` is the negation of the ``bh`` the
        # step will actually multiply by however the guard was applied.
        params['minus_bh'] = -params['bh']
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


#: Position of each spatial axis (0 = x, 1 = y, 2 = z) in a single-component ``(nz, ny, nx)``
#: array -- the shape a CPML profile has to broadcast against.
_COMPONENT_AXIS = {0: 2, 1: 1, 2: 0}


def _cpml(scene, plan, rt):
    """``{psi key: {'b', 'a', 'ik'}}``: the profiles of every PML face, reshaped to broadcast.

    A profile is 1-D along its absorbing axis and is reshaped to ``(cells, 1, 1)`` or
    ``(1, cells, 1)`` or ``(1, 1, cells)`` -- never materialised at slab shape, which for a
    128^3 box with six faces is the difference between 36 MB of psi and the same again in
    coefficients that are constant across the face.

    Each face carries two profiles half a cell apart: ``half`` for the forward difference in
    ``curl E`` (which lands on the H points) and ``node`` for the backward one in ``curl H``.
    The psi key says which, through the field it differenced.
    """
    if not plan.cpml:
        return {}
    background = as_material(getattr(scene, 'background', None), 'Scene(background=)')
    wavelengths = dict(plan.pml_wavelengths)
    out = {}
    for face in plan.cpml:
        boundary = scene.boundaries.face(face.name)
        matched = background if boundary.background is None else boundary.background
        eps_r = face_eps_r(matched, face.axis)
        if eps_r is None:
            raise ValueError(
                f'PML on {face.name!r} is matched to a background whose permittivity or '
                f'permeability is traced. The stretch profile is static -- it fixes the '
                f'params\' shapes and is built once per plan -- so the background it is '
                f'matched to is a static value, not a leaf. Give PML(background=...) a '
                f'concrete material, or differentiate a structure instead.')
        table = profiles(face.cells, plan.dx, plan.dt, wavelengths[face.name], eps_r,
                         face.side, boundary.reflection)
        shape = [1, 1, 1]
        shape[_COMPONENT_AXIS[face.axis]] = face.cells
        for field, which in (('E', 'half'), ('H', 'node')):
            b, a, ik = table[which]
            for c in range(3):
                if c == face.axis:
                    continue
                out[psi_key(face.name, field, c)] = {
                    'b': rt.array(b.reshape(shape), 'float64'),
                    'a': rt.array(a.reshape(shape), 'float64'),
                    'ik': rt.array(ik.reshape(shape), 'float64')}
    return out


def _poles(scene, plan, rt):
    """``{term name: {'step_gain', 'hold', 'drive', 'dx_over_dt'}}`` for every pole term.

    The coefficients are per pole and are scalars, because a stage-1 structure is one uniform
    material over its slab; they are leaves, so a gradient with respect to a pole strength
    flows through ``drive``. ``dx_over_dt`` is what turns the polarisation increment into the
    right addition to ``cH``, which carries no ``1/dx`` (``wavebit/fd/_kernel/poles.py``).
    """
    structures = [s for s in (getattr(scene, 'structures', ()) or ()) if s.material.poles]
    out = {}
    for term in plan.terms:
        if not hasattr(term, 'n_poles'):
            continue                                    # a test hook's term, not a pole set
        index = int(term.name[len('poles['):-1])
        poles = (getattr(scene, 'structures', ()) or ())[index].material.poles
        step_gain, hold, drive = [], [], []
        for pole in poles:
            g, h, d = pole.coefficients(plan.dt)
            step_gain.append(rt.array(g, 'float64'))
            hold.append(rt.array(h, 'float64'))
            drive.append(rt.array(d, 'float64'))
        out[term.name] = {'step_gain': tuple(step_gain), 'hold': tuple(hold),
                          'drive': tuple(drive),
                          'dx_over_dt': rt.array(plan.dx / plan.dt, 'float64')}
    if structures and len(out) != len(structures):
        # Not reachable through ``fd.plan``, which always builds one term per dispersive
        # structure; a test hook that replaces ``plan.terms`` can get here, and a bare assert
        # would land as a message-less AssertionError deep in the lowering.
        raise ValueError(
            f'the plan carries {len(out)} pole term(s) but the scene has {len(structures)} '
            f'dispersive structure(s). A plan drives the scene it was built from; a scene '
            f'whose poles are not in the plan would run with part of its material missing. '
            f'Use fd.plan(scene) for this scene, or take the structures out of it.')
    return out


def _stability_guard(scene, plan, rt):
    """``None`` when the CFL bound was read concretely, else ``1`` or ``nan`` as a leaf.

    The bound involves the materials, which may be traced. When every number is concrete the
    door already refused an unstable scene and nothing is built here -- which is what keeps a
    vacuum run's ``be`` the stage-0 float to the bit. When one is not, the comparison is done
    on the backend and its answer multiplies ``be`` and ``bh``, so the first step turns the
    whole state into ``nan`` and the result is unmistakably not a field.
    """
    bound, _ = checks.cfl_bound(scene, plan)
    if bound is not None:
        return None
    return _traced_guard(scene, plan, rt)


def _traced_guard(scene, plan, rt):
    """The same bound, computed on the backend, as ``where(dt < bound, 1, nan)``."""
    background = as_material(getattr(scene, 'background', None), 'Scene(background=)')
    materials = [background] + [s.material for s in (getattr(scene, 'structures', ()) or ())]
    n_min, resonance, coupling = None, None, None
    for material in materials:
        for c in range(3):
            n = rt.xp.sqrt(rt.array(material.eps[c], 'float64')
                           * rt.array(material.mu_value[c], 'float64'))
            n_min = n if n_min is None else rt.xp.minimum(n_min, n)
        total = None
        for pole in material.poles:
            w0_squared, strength = pole.rate()
            w0_squared = rt.array(w0_squared, 'float64')
            strength = rt.array(strength, 'float64')
            resonance = w0_squared if resonance is None else rt.xp.maximum(resonance,
                                                                          w0_squared)
            total = strength if total is None else total + strength
        if total is not None:
            eps_min = None
            for c in range(3):
                e = rt.array(material.eps[c], 'float64')
                eps_min = e if eps_min is None else rt.xp.minimum(eps_min, e)
            ratio = total / eps_min
            coupling = ratio if coupling is None else rt.xp.maximum(coupling, ratio)
    zero = rt.array(0.0, 'float64')
    rate = (resonance if resonance is not None else zero) \
        + (coupling if coupling is not None else zero)
    vacuum = n_min * (plan.dx / math.sqrt(3.0))
    bound = 1.0 / rt.xp.sqrt(1.0 / vacuum ** 2 + rate / 4.0)
    return rt.xp.where(rt.array(plan.dt, 'float64') < bound, rt.array(1.0, 'float64'),
                       rt.array(float('nan'), 'float64'))
