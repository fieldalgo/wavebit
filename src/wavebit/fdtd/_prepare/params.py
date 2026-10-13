"""``lower(scene, plan, rt) -> Params``: the traced side of a run, as a pytree.

``Params`` is a plain dict, because a dict is a pytree on JAX and needs no registration on
Torch or NumPy. Its leaves are the numbers a gradient may flow through:

    ae, be, ah, bh    the update coefficients -- 1, dt/dx, 1, dt/dx in vacuum, arrays once a
                      material or an Absorber makes them so (``coefficients.py``)
    minus_bh          ``-bh``, lowered once because negating an array in the step body is a
                      full-volume temporary per step (``coefficients.build`` says what that
                      measured)
    bloch             {axis: exp(i*phi)} for each Bloch axis (the low face conjugates it)
    axis1d            {axis: i k dx} for each Axis1D axis (stage 3): the whole derivative
                      along it, in the curl's units (``src/wavebit/fdtd/_kernel/curl.py``); ``{}``
                      without one
    stable            the stability guard (``1.0`` or ``nan``) when one was built, else None --
                      what ``res.diagnostics['stable']`` reads
    amplitude         the InitialField amplitude, cast to the run's dtype
    E0, H0            the initial fields as (3, nz, ny, nx) arrays, or None
    cpml              {psi key: {'b', 'a', 'ik'}} -- the CPML profiles, 1-D along the
                      absorbing axis and reshaped to broadcast, never at slab shape
    cpml_loss         {'<face>:<E|H>': {'keep', 'lift'}} -- the PML's stabilizing loss on
                      the normal component, shaped like a CPML profile; only faces with
                      ``stabilize > 0`` (``src/wavebit/fdtd/_kernel/cpml.py::damp``)
    poles             {term name: {'step_gain', 'hold', 'drive', 'dx_over_dt'}} per pole
    sources           {source name: {'amplitude', 'phase', 'delay', 'f0', 'tau', 't0', ...}}
                      per stage-2 source, with a Dipole's 'stamp', a column's own params
                      ('column', and 'driver' for a layered TFSF column's uniform drive) and
                      transverse Bloch 'pattern' (``src/wavebit/fdtd/_prepare/sources.py``); the
                      waveform table built from them is added inside the traced function
    dft               {region name: 2 pi f} per stage-3 ``DFTRegion`` observer; ``{}`` without
    stencil           {term name: {'index', 'value', 'output'}} per structure that asked for
                      ``discretisation='stencil'`` -- the off-diagonal coefficients of the
                      operator assembled on the host at plan time, and the integer gather
                      that reads them. Absent from a scene that asked for none. The
                      operator's *diagonal* is not here: it is patched into ``be``, because
                      that is what the step and the interior-energy diagnostic read
                      (:func:`_stencil`)

Whether ``E0`` or ``H0`` is ``None`` is static and recorded on the plan (``has_E0``,
``has_H0``); their values are leaves. That is the pytree rule this repository keeps
everywhere: branch on None-versus-present, never on a value.

**The stability guard.** A material value may be traced, and a traced value cannot be
compared against the CFL bound at lowering time. When the bound cannot be read, ``be`` and
``bh`` are multiplied by a factor that is ``1`` if the run is stable and ``nan`` if it is
not, so an unstable run returns ``nan`` arrays rather than a number shaped like a field --
the stage-0 rule for a condition only run time can know. When every value *is* concrete the
bound was checked at the door (``checks.check_stability``) and no guard is built, so a vacuum
run's coefficients are the stage-0 ones to the bit. **An Axis1D face with a traced ``k``** adds
one condition only run time can know, ``|k| <= k_max`` (the plan's CFL bound is made for
``k_max``), to the same guard.
"""
from wavebit._common.backend import eager_under_torch_compile as _eager_under_torch_compile
import math

import numpy as np

from wavebit._common.backend import check_input_precision

from wavebit.fdtd._legacy_boundaries import AXES
from wavebit.fdtd._legacy_boundaries import Axis1D
from wavebit.fdtd._legacy_boundaries import Bloch
from wavebit.materials.time_domain import as_material
from wavebit.fdtd._kernel.cpml import loss_key
from wavebit.fdtd._kernel.cpml import psi_key
import wavebit.fdtd._prepare.checks as checks
from wavebit.fdtd._prepare.coefficients import build as build_coefficients
from wavebit.fdtd._prepare.cpml import face_eps_r
from wavebit.fdtd._prepare.cpml import profiles
from wavebit.fdtd._prepare.cpml import stabilizing_factors
from wavebit.fdtd._prepare.fields import evaluate
from wavebit.fdtd._prepare.sources import lower_sources

__all__ = ['lower']


@_eager_under_torch_compile
def lower(scene, plan, rt='numpy', device=None):
    """The traced parameters of ``scene`` under ``plan``, as arrays of runtime ``rt``.

    ``rt`` is a runtime object, or a backend name / ``wavebit.Backend`` to build one from.

    The first thing it asks is whether the scene and the plan agree about which structures
    carry a prepared extended-stencil operator (``stencil.refuse_discretisation``): the plan
    is the caller's choice here, and a scene lowered against a plan built for its other
    discretisation would otherwise get that one silently. A scene and a plan with no such
    structure pass it after one loop over each.
    """
    from wavebit.fdtd._decay import validate_plan
    from wavebit.fdtd._decay import validate_runtime
    from wavebit.fdtd._decay import reject_carried
    validate_plan(scene, plan)
    stopping = plan.decay is not None
    from wavebit.fdtd._prepare.constitutive_absorber import refuse_mismatch as refuse_absorber_constitutive
    refuse_absorber_constitutive(scene, plan)
    from wavebit.fdtd._prepare.stencil import refuse_discretisation
    refuse_discretisation(scene, plan)
    from wavebit.fdtd._prepare.oscillator_stack import refuse_mismatch as refuse_oscillator_stack_mismatch
    refuse_oscillator_stack_mismatch(scene, plan)
    from wavebit.fdtd._prepare.debye_stack import refuse_mismatch as refuse_debye_stack_mismatch
    refuse_debye_stack_mismatch(scene, plan)
    from wavebit.fdtd._prepare.conductive_stack import refuse_mismatch as refuse_stack_mismatch
    refuse_stack_mismatch(scene, plan)
    from wavebit.fdtd._prepare.conductive_film import refuse_mismatch as refuse_film_mismatch
    refuse_film_mismatch(scene, plan)
    from wavebit.fdtd._prepare.single_pole_film import refuse_mismatch as refuse_dispersive_film_mismatch
    refuse_dispersive_film_mismatch(scene, plan)
    from wavebit.fdtd._prepare.debye_film import refuse_mismatch as refuse_debye_film_mismatch
    refuse_debye_film_mismatch(scene, plan)
    from wavebit.fdtd._prepare.two_debye_film import refuse_mismatch as refuse_two_debye_film_mismatch
    refuse_two_debye_film_mismatch(scene, plan)
    from wavebit.fdtd._prepare.debye_sum_film import refuse_mismatch as refuse_debye_sum_film_mismatch
    refuse_debye_sum_film_mismatch(scene, plan)
    from wavebit.fdtd._prepare.oscillator_sum_film import refuse_mismatch as refuse_oscillator_sum_film_mismatch
    refuse_oscillator_sum_film_mismatch(scene, plan)
    from wavebit.fdtd._prepare.hybrid_film import refuse_mismatch as refuse_hybrid_film_mismatch
    refuse_hybrid_film_mismatch(scene, plan)
    from wavebit.fdtd._prepare.oscillator_stack import validate as validate_oscillator_stack_resolution
    validate_oscillator_stack_resolution(scene, plan.grid)
    from wavebit.fdtd._prepare.debye_stack import validate as validate_debye_stack_resolution
    validate_debye_stack_resolution(scene, plan.grid)
    from wavebit.fdtd._prepare.conductive_stack import validate as validate_stack_resolution
    validate_stack_resolution(scene, plan.grid)
    from wavebit.fdtd._prepare.single_pole_resolution import validate as validate_film_resolution
    validate_film_resolution(scene, plan.grid)
    from wavebit.fdtd._prepare.debye_resolution import validate as validate_debye_resolution
    validate_debye_resolution(scene, plan.grid)
    from wavebit.fdtd._prepare.two_debye_resolution import validate as validate_two_debye_resolution
    validate_two_debye_resolution(scene, plan.grid)
    from wavebit.fdtd._prepare.debye_sum_resolution import validate as validate_debye_sum_resolution
    validate_debye_sum_resolution(scene, plan.grid)
    from wavebit.fdtd._prepare.oscillator_sum_resolution import validate as validate_oscillator_sum_resolution
    validate_oscillator_sum_resolution(scene, plan.grid)
    from wavebit.fdtd._prepare.hybrid_resolution import validate as validate_hybrid_resolution
    validate_hybrid_resolution(scene, plan.grid)
    from wavebit.fdtd._prepare.junction import refuse_mismatch
    refuse_mismatch(scene, plan)
    from wavebit.fdtd._prepare.shape_ranges import refuse_mismatch as refuse_range_mismatch
    refuse_range_mismatch(scene, plan)
    from wavebit.fdtd._prepare.shape_ranges import validate as validate_shapes
    from wavebit.fdtd._prepare.shape_ranges import guard as shape_guard
    from wavebit.fdtd._prepare.shape_ranges import resolution_guard
    from wavebit.fdtd._prepare.shape_ranges import grazing_guard
    validate_shapes(scene, plan)
    if not hasattr(rt, 'slab_add'):
        from wavebit.fdtd._runtime import runtime
        rt = runtime(rt, device)
    if stopping:
        validate_runtime(rt)
    grid = plan.grid
    real = 'float64'
    params = dict(build_coefficients(scene, plan, rt))
    certified = True
    if plan.constitutive:
        from wavebit.fdtd._prepare.constitutive import lift
        from wavebit.fdtd._prepare.constitutive import lower as lower_constitutive
        nodes = {}
        from wavebit.fdtd._prepare.constitutive_absorber import add_profiles
        from wavebit.fdtd._prepare.constitutive_absorber import validate_links
        add_profiles(scene, plan, params, rt)
        params['constitutive'] = lower_constitutive(scene, plan, rt, nodes=nodes)
        validate_links(plan, params, rt)
        _stencil(scene, plan, params, rt)
        try:
            certified = lift(plan, params, nodes, rt)
        except np.linalg.LinAlgError:
            # A traced eps of 0, nan or inf -- an optimizer's step -- reaches the lift's host
            # eigensolver before any guard, and NumPy's eigh raises on it where the traced
            # path's returns nan. Nothing is certified, so constitutive.guard makes the run
            # nan, as every other traced failure does; the concrete value is refused by name
            # before any of this (materials._positive), and a finite contrast past float64's
            # reach by fd.plan (constitutive.validate).
            certified = False
    params.update({'bloch': {}, 'axis1d': {}, 'amplitude': None, 'E0': None, 'H0': None,
                   'cpml': _cpml(scene, plan, rt), 'cpml_loss': _cpml_loss(scene, plan, rt),
                   'poles': _poles(scene, plan, rt),
                   'sources': lower_sources(scene, plan, rt), 'dft': _dft(plan, rt),
                   'stable': None})
    guard = _stability_guard(scene, plan, rt)
    guard = _axis1d_guard(scene, rt, guard)
    guard = _material_door_guard(scene, rt, guard)
    guard = resolution_guard(scene, plan, rt, guard)
    guard = grazing_guard(scene, plan, rt, guard)
    guard = shape_guard(scene, plan, rt, guard)
    if plan.constitutive:
        from wavebit.fdtd._prepare.constitutive import guard as constitutive_guard
        coupled_guard = constitutive_guard(scene, plan, params, rt, certified)
        guard = coupled_guard if guard is None else guard * coupled_guard
        params['constitutive'] = {k: v * guard for k, v in params['constitutive'].items()}
    params['stable'] = guard
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
        elif isinstance(face, Axis1D):
            check_input_precision(f'Axis1D on {axis}', face.k, np.float64)
            params['axis1d'][axis] = (1j * grid.dx) * rt.array(face.k, real)
    initial = scene.initial
    if initial is None:
        params['amplitude'] = rt.array(1.0, plan.dtype)
        if stopping:
            reject_carried(params)
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
    flat = tuple(a for a, axis in enumerate(AXES)
                 if isinstance(getattr(scene.boundaries, axis)[0], Axis1D))
    params['E0'] = evaluate('E', initial.E, grid, 'E', rt, plan.dtype, flat,
                            concrete_only=stopping)
    params['H0'] = evaluate('H', initial.H, grid, 'H', rt, plan.dtype, flat,
                            concrete_only=stopping)
    if stopping:
        reject_carried(params)
    return params


def _stencil(scene, plan, params, rt):
    """``params['stencil']`` and the patched ``be``, for every prepared extended-stencil term.

    Two halves of one operator go two places, and both of them are the same arrays the host
    assembled at plan time -- nothing is solved, fitted or branched on here.

    * the **off-diagonal** coefficients become ``params['stencil'][name]['value']``, beside
      ``params['constitutive']``, with the integer gather pattern that reads them;
    * the **diagonal** is written into ``be`` over the term's own region.

    The diagonal has to reach ``be`` and not a private key. ``be`` is what the step multiplies
    the curl by, and it is also what ``InteriorEnergy._weight`` recovers the medium from --
    so an effective permittivity that lived only in ``params['stencil']`` would be invisible to
    the one diagnostic whose job is to see it, and a run that changed the medium would report
    the energy of the medium it no longer has. The gate is
    ``tests/fdtd/test_fd_stencil.py::test_the_prepared_diagonal_reaches_be_and_the_energy_weight``.

    The write is :func:`~wavebit.fdtd._prepare.coefficients._paste` and not an indexed assignment.
    ``be`` may be the same object an earlier painter still holds, ``a[i] = v`` does not exist
    on JAX at all and mutates in place on Torch, and the concatenation is the one spelling
    that is out of place and identical on all three backends.
    """
    from wavebit.fdtd._prepare.coefficients import _paste
    from wavebit.fdtd._prepare.stencil import lower as lower_stencil
    tables, patches = lower_stencil(scene, plan, rt)
    if not tables:
        return
    params['stencil'] = tables
    be = params['be']
    full = (3, plan.nz, plan.ny, plan.nx)
    if getattr(be, 'shape', ()) != full:
        be = rt.xp.broadcast_to(be, full)
    for box, block in patches:
        be = _paste(rt.xp, be, box, block)
    params['be'] = be


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


def _cpml_loss(scene, plan, rt):
    """``{loss key: {'keep', 'lift'}}``: the stabilizing factors of every stabilized face.

    Only a face whose plan entry has ``stabilize > 0`` gets one, and the kernel skips the
    others on that static value (``_kernel/cpml.py::damp``). The profile is matched to the
    ``eps_r`` the stretch is (:func:`_cpml`, which has already refused a traced one). The
    normal ``E`` component sits half a cell along the axis and reads the ``half`` profile,
    the normal ``H`` one sits on the node; both are 1-D along the axis, reshaped to broadcast
    over the slab like a CPML profile.
    """
    background = as_material(getattr(scene, 'background', None), 'Scene(background=)')
    out = {}
    for face in plan.cpml:
        if not face.stabilize:
            continue
        boundary = scene.boundaries.face(face.name)
        matched = background if boundary.background is None else boundary.background
        table = stabilizing_factors(face.cells, plan.dx, plan.dt,
                                    face_eps_r(matched, face.axis), face.side,
                                    boundary.reflection, face.stabilize)
        shape = [1, 1, 1]
        shape[_COMPONENT_AXIS[face.axis]] = face.cells
        for field, which in (('E', 'half'), ('H', 'node')):
            keep, lift = table[which]
            out[loss_key(face.name, field)] = {
                'keep': rt.array(keep.reshape(shape), 'float64'),
                'lift': rt.array(lift.reshape(shape), 'float64')}
    return out


def _dft(plan, rt):
    """``{region name: 2 pi f (nf,)}`` for every ``DFTRegion``; empty for a scene with none."""
    return {o.name: rt.array(2.0 * math.pi * np.asarray(o.freqs, dtype=np.float64), 'float64')
            for o in plan.observers if hasattr(o, 'freqs')}


def _poles(scene, plan, rt):
    """``{term name: {'step_gain', 'hold', 'drive', 'dx_over_dt'}}`` for every pole term.

    The coefficients are per pole and are scalars, because a stage-1 structure is one uniform
    material over its slab; they are leaves, so a gradient with respect to a pole strength
    flows through ``drive``. ``dx_over_dt`` is what turns the polarisation increment into the
    right addition to ``cH``, which carries no ``1/dx`` (``src/wavebit/fdtd/_kernel/poles.py``).
    """
    from wavebit.fdtd._prepare.conductive_film import InterfacePoleTerm
    from wavebit.fdtd._prepare.conductive_film import pole_params as film_poles
    from wavebit.fdtd._prepare.single_pole_film import DispersiveFilmTerm
    from wavebit.fdtd._prepare.single_pole_film import descriptor
    from wavebit.fdtd._prepare.single_pole_film import pole_params as dispersive_film_poles
    from wavebit.fdtd._prepare.debye_film import DebyeFilmTerm
    from wavebit.fdtd._prepare.debye_film import descriptor as debye_descriptor
    from wavebit.fdtd._prepare.debye_film import pole_params as debye_film_poles
    from wavebit.fdtd._prepare.two_debye_film import TwoDebyeFilmTerm
    from wavebit.fdtd._prepare.two_debye_film import descriptor as two_debye_descriptor
    from wavebit.fdtd._prepare.two_debye_film import pole_params as two_debye_film_poles
    from wavebit.fdtd._prepare.debye_sum_film import DebyeSumFilmTerm
    from wavebit.fdtd._prepare.debye_sum_film import descriptor as debye_sum_descriptor
    from wavebit.fdtd._prepare.debye_sum_film import pole_params as debye_sum_film_poles
    from wavebit.fdtd._prepare.oscillator_stack import OscillatorStackTerm
    from wavebit.fdtd._prepare.oscillator_stack import descriptor as oscillator_stack_descriptor
    from wavebit.fdtd._prepare.oscillator_stack import pole_params as oscillator_stack_poles
    from wavebit.fdtd._prepare.debye_stack import DebyeStackTerm
    from wavebit.fdtd._prepare.debye_stack import descriptor as debye_stack_descriptor
    from wavebit.fdtd._prepare.debye_stack import pole_params as debye_stack_poles
    from wavebit.fdtd._prepare.oscillator_sum_film import OscillatorSumFilmTerm
    from wavebit.fdtd._prepare.oscillator_sum_film import descriptor as oscillator_sum_descriptor
    from wavebit.fdtd._prepare.oscillator_sum_film import pole_params as oscillator_sum_film_poles
    from wavebit.fdtd._prepare.hybrid_film import HybridFilmTerm
    from wavebit.fdtd._prepare.hybrid_film import descriptor as hybrid_descriptor
    from wavebit.fdtd._prepare.hybrid_film import pole_params as hybrid_film_poles
    structures = [s for s in (getattr(scene, 'structures', ()) or ()) if s.material.poles]
    if (descriptor(scene, plan.grid) is not None
            or debye_descriptor(scene, plan.grid) is not None
            or two_debye_descriptor(scene, plan.grid) is not None
            or debye_sum_descriptor(scene, plan.grid) is not None
            or debye_stack_descriptor(scene, plan.grid) is not None
            or oscillator_stack_descriptor(scene, plan.grid) is not None
            or oscillator_sum_descriptor(scene, plan.grid) is not None
            or hybrid_descriptor(scene, plan.grid) is not None):
        structures = []                     # represented by spatial film terms instead
    out = {}
    for term in plan.terms:
        if isinstance(term, (InterfacePoleTerm, DispersiveFilmTerm, DebyeFilmTerm,
                             TwoDebyeFilmTerm, DebyeSumFilmTerm, DebyeStackTerm,
                             OscillatorSumFilmTerm, HybridFilmTerm, OscillatorStackTerm)):
            continue
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
    from wavebit.fdtd._prepare.conductive_stack import pole_params as stack_poles
    out.update(stack_poles(scene, plan, rt))
    out.update(film_poles(scene, plan, rt))
    out.update(dispersive_film_poles(scene, plan, rt))
    out.update(debye_film_poles(scene, plan, rt))
    out.update(two_debye_film_poles(scene, plan, rt))
    out.update(debye_sum_film_poles(scene, plan, rt))
    out.update(debye_stack_poles(scene, plan, rt))
    out.update(oscillator_stack_poles(scene, plan, rt))
    out.update(oscillator_sum_film_poles(scene, plan, rt))
    out.update(hybrid_film_poles(scene, plan, rt))
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
    from wavebit.fdtd._prepare.shape_ranges import materials_traced
    if bound is not None and not materials_traced(scene):
        return None
    # A Torch forward-mode dual reads as its primal through ``materials.concrete``, so the
    # bound above can be readable for a value that carries a tangent; with material seeds the
    # plan was certified at the seed, not there, so such a value is guarded like any other.
    return _traced_guard(scene, plan, rt)


def _axis1d_guard(scene, rt, guard):
    """``guard`` times ``where(|k| <= k_max, 1, nan)`` for every Axis1D face with a traced ``k``.

    A concrete ``k`` past ``k_max`` was refused at ``fd.Axis1D``; a traced one is read here,
    on the backend, so the run turns into ``nan`` rather than returning a field computed past
    the plan's CFL bound.
    """
    from wavebit.materials.time_domain import concrete
    for axis in AXES:
        face = getattr(scene.boundaries, axis)[0]
        if not isinstance(face, Axis1D) or concrete(face.k) is not None:
            continue
        k = rt.array(face.k, 'float64')
        ok = rt.xp.where(rt.xp.abs(k) <= face.k_max * (1.0 + 1e-12), rt.array(1.0, 'float64'),
                         rt.array(float('nan'), 'float64'))
        guard = ok if guard is None else guard * ok
    return guard


def _material_door_guard(scene, rt, guard):
    """``guard`` times ``where(every traced material value passes its door, 1, nan)``.

    The concrete constructors refuse ``eps`` or ``mu`` not positive, ``sigma`` negative and
    every pole parameter out of range (``materials._positive``, ``_pole_scalar``); a traced
    value passes them unread. So each traced one is read here, on the backend, and a scene the
    concrete path would have refused runs to ``nan`` -- built only when some value is traced,
    so a concrete scene keeps ``stable = None`` and its ``be`` to the bit. Found in the C5
    review: a traced ``sigma = -200`` built no guard at all (the CFL bound does not read
    ``sigma``) and grew to 4.6e147 in 200 steps reading stable; a traced Lorentz
    ``strength = -4000`` read stable and grew to 3.4e64 in 60.
    """
    from wavebit.materials.time_domain import concrete
    zero = rt.array(0.0, 'float64')
    checks_ = []
    background = as_material(getattr(scene, 'background', None), 'Scene(background=)')
    for material in [background] + [s.material for s in (getattr(scene, 'structures', ())
                                                          or ())]:
        leaves = [(v, True) for v in material.eps] + [(v, True) for v in material.mu_value]
        leaves += [(v, False) for v in getattr(material, 'sigma', ()) or ()]
        for pole in material.poles:
            leaves += [(getattr(pole, name), strict)
                       for name, strict in _POLE_DOORS.get(type(pole).__name__, ())]
        for value, strict in leaves:
            if concrete(value) is not None:
                continue
            v = rt.array(value, 'float64')
            checks_.append(v > zero if strict else v >= zero)
    if not checks_:
        return guard
    ok = checks_[0]
    for c in checks_[1:]:
        ok = rt.xp.logical_and(ok, c)
    ok = rt.xp.where(ok, rt.array(1.0, 'float64'), rt.array(float('nan'), 'float64'))
    return ok if guard is None else guard * ok


#: The pole parameters each constructor checks, and whether strictly positive
#: (``materials._pole_scalar``): what :func:`_material_door_guard` reads when one is traced.
_POLE_DOORS = {'Drude': (('omega_p', True), ('gamma', False)),
               'Lorentz': (('omega_0', True), ('gamma', False), ('strength', False)),
               'Debye': (('tau', True), ('delta_eps', False))}


def _traced_guard(scene, plan, rt):
    """The same bound, computed on the backend, as ``where(dt < bound, 1, nan)``.

    ``n_min`` is ``sqrt(min eps * min mu)`` over every phase, as ``checks.cfl_bound``'s is (its
    docstring derives it); before stage 4's C5 this guard took each phase's own index, and a
    traced ``eps = 4, mu = 0.25`` pair at courant 0.9 read stable and ran to NaN. A phase whose
    traced ``eps`` or ``mu`` is not positive reads unstable too -- the concrete path refuses
    one at ``Material``, and a traced ``eps = -2, mu = -1`` pair had multiplied back to a
    positive index and read stable (the C5 audit measured the field at 1.5e290 in 20 steps on
    a smoothed ``fd.Box`` of it, whose mixed voxels take ``eps`` through zero). The pole terms
    start at zero and take the maximum, as ``cfl_bound``'s do, so that the twin reads the same
    bound; a traced negative strength could otherwise loosen it (the C5 review), but
    :func:`_material_door_guard` refuses that value first, so the clamp is not reached today. The pole loop names a phase's own permittivity ``own`` so that it
    cannot overwrite the scene-wide ``eps_min`` this pairing introduced; the mutant that reuses
    the name drops every earlier phase from the pairing.
    """
    background = as_material(getattr(scene, 'background', None), 'Scene(background=)')
    materials = [background] + [s.material for s in (getattr(scene, 'structures', ()) or ())]
    zero = rt.array(0.0, 'float64')
    eps_min, mu_min, resonance, coupling = None, None, zero, zero
    for material in materials:
        for c in range(3):
            e = rt.array(material.eps[c], 'float64')
            m = rt.array(material.mu_value[c], 'float64')
            eps_min = e if eps_min is None else rt.xp.minimum(eps_min, e)
            mu_min = m if mu_min is None else rt.xp.minimum(mu_min, m)
        total = None
        for pole in material.poles:
            w0_squared, strength = pole.rate()
            w0_squared = rt.array(w0_squared, 'float64')
            strength = rt.array(strength, 'float64')
            resonance = rt.xp.maximum(resonance, w0_squared)
            total = strength if total is None else total + strength
        if total is not None:
            own = None                                     # this phase's own eps, for its poles
            for c in range(3):
                e = rt.array(material.eps[c], 'float64')
                own = e if own is None else rt.xp.minimum(own, e)
            coupling = rt.xp.maximum(coupling, total / own)
    one = rt.array(1.0, 'float64')
    positive = rt.xp.logical_and(eps_min > zero, mu_min > zero)
    n_min = rt.xp.sqrt(rt.xp.where(positive, eps_min * mu_min, one))
    rate = resonance + coupling
    axes = checks.cfl_axis_sum(scene, plan.dx)
    if axes == 3.0 / plan.dx ** 2:
        vacuum = n_min * (plan.dx / math.sqrt(3.0))
    else:
        vacuum = n_min / math.sqrt(axes)                   # an Axis1D face (stage 3)
    bound = 1.0 / rt.xp.sqrt(1.0 / vacuum ** 2 + rate / 4.0)
    stable = rt.xp.logical_and(positive, rt.array(plan.dt, 'float64') < bound)
    return rt.xp.where(stable, one, rt.array(float('nan'), 'float64'))
