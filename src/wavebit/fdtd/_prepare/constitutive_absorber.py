"""Exact link-level admission for matched absorption of coupled dielectrics.

No numerical tolerance decides commutation. Each nonzero common-node cross
coefficient supplies four edge-to-edge links; each live link must connect
identical represented rates. The diagonal star lift cannot alter this fact.
The old absorber-clear route has no new coefficient key or arithmetic.
"""
import dataclasses
import math

import numpy as np

from wavebit.fdtd._legacy_boundaries import Absorber
from wavebit.fdtd._legacy_boundaries import Bloch
from wavebit.fdtd._legacy_boundaries import PEC
from wavebit.fdtd._legacy_boundaries import PMC
from wavebit.fdtd._legacy_boundaries import Periodic
from wavebit.materials.time_domain import Material
from wavebit.materials.time_domain import as_material
from wavebit.materials.time_domain import concrete
from wavebit.fdtd._legacy_geometry import Box
from wavebit.fdtd._legacy_geometry import Cylinder
from wavebit.fdtd._legacy_geometry import Prism
from wavebit.fdtd._legacy_geometry import Sphere
from wavebit.fdtd._legacy_geometry import Slab
from wavebit.fdtd._legacy_geometry import Region
from wavebit.fdtd._legacy_sources import Dipole
from wavebit.fdtd._kernel.constitutive_absorber import AbsorberConstitutiveTerm
from wavebit.fdtd._kernel.observe import InteriorEnergy
from wavebit.fdtd._prepare.coefficients import _damping
from wavebit.fdtd._prepare.coefficients import materials_of
from wavebit.fdtd._prepare.shape_ranges import _carries
from wavebit.fdtd._prepare.shape_ranges import materials_traced

PAIRS = ((0, 1), (0, 2), (1, 2))


def _reaches(term, plan, face, cells):
    axis = 'xyz'.index(face[0])
    low, high = tuple(reversed(term.box))[axis]
    return low < cells if face[1] == '-' else high > plan.grid.n[axis] - cells


def _record(value):
    """Only plain concrete specification data; no repr of an opaque object."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_record(v) for v in value)
    if dataclasses.is_dataclass(value):
        return (type(value).__module__, type(value).__qualname__,
                tuple((f.name, _record(getattr(value, f.name)))
                      for f in dataclasses.fields(value)))
    scalar = concrete(value)
    if scalar is None or not math.isfinite(scalar):
        raise ValueError('ConstitutiveAbsorberConcrete: finite concrete specification required.')
    return scalar


def _scope(scene, plan):
    """Validate and bind the first nonmagnetic, electric-Dipole-only contract."""
    if getattr(scene, 'initial', None) is not None:
        raise ValueError('ConstitutiveAbsorberScope: initial fields are not admitted.')
    if getattr(scene.run, 'decay', None) is not None:
        raise ValueError('ConstitutiveAbsorberScope: automatic stopping needs a separate '
                         'state mapping; use a fixed duration for this new term.')
    if _carries(scene) or materials_traced(scene):
        raise ValueError('ConstitutiveAbsorberConcrete: this first route needs concrete inputs.')
    background, structures = materials_of(scene)
    material_values = []
    for material in (background,) + tuple(s.material for s in structures):
        material = as_material(material)
        if type(material) is not Material:
            raise ValueError('ConstitutiveAbsorberScope: only plain lossless Material is admitted.')
        eps = tuple(concrete(v) for v in material.eps)
        mu = tuple(concrete(v) for v in material.mu_value)
        if (any(v is None or not math.isfinite(v) or v <= 0 for v in eps)
                or eps[0] != eps[1] or eps[0] != eps[2]
                or mu != (1.0, 1.0, 1.0)):
            raise ValueError('ConstitutiveAbsorberScope: positive isotropic eps and mu=1 required.')
        material_values.append(eps[0])
    for source in scene.sources:
        if type(source) is not Dipole or source.kind != 'electric':
            raise ValueError('ConstitutiveAbsorberScope: only electric CurrentDipoles are '
                             'admitted.')
    actual = []
    for axis in 'xyz':
        for side in '-+':
            face = scene.boundaries.face(axis + side)
            if type(face) is Absorber:
                actual.append((axis + side, face.cells))
            elif type(face) not in (PEC, PMC, Periodic, Bloch):
                raise ValueError('ConstitutiveAbsorberScope: CPML, mirrors and reduced axes '
                                 'are outside this first absorber route.')
    if (not actual or len({name[0] for name, _ in actual}) != 1
            or tuple(actual) != tuple(plan.absorber_cells) or plan.pml_cells):
        raise ValueError('ConstitutiveAbsorberScope: one matched absorbing axis and an '
                         'unchanged absorbing layout are required.')
    for structure in structures:
        if (structure.discretisation != 'voxel' or getattr(structure.shape, 'range', None)
                is not None or type(structure.shape) not in
                (Box, Cylinder, Prism, Sphere, Slab, Region)):
            raise ValueError('ConstitutiveAbsorberScope: ordinary concrete voxel shapes required.')
    return ('constitutive-absorber-v1', tuple(actual), tuple(plan.grid.n), plan.dx,
            tuple(material_values), tuple(_record(s) for s in structures),
            tuple(_record(scene.boundaries.face(a + s)) for a in 'xyz' for s in '-+'))


def promote(scene, plan, terms):
    """Replace only E terms whose finite support reaches a matched absorber."""
    touched = tuple(t for t in terms if any(_reaches(t, plan, name, cells)
                                            for name, cells in plan.absorber_cells))
    if not touched:
        return tuple(terms)
    signature = _scope(scene, plan)
    selected = {t.name for t in touched}
    out = []
    for term in terms:
        if term.name not in selected:
            out.append(term)
            continue
        if term.field != 'E':
            raise ValueError('ConstitutiveAbsorberScope: only electric coupling is admitted.')
        out.append(AbsorberConstitutiveTerm(term.name, term.field, term.region,
                                           term.ratio, term.masks, signature))
    return tuple(out)


def refuse_mismatch(scene, plan):
    """Direct lower callers cannot remove, replace or inherit an unrelated term."""
    held = tuple(t for t in plan.constitutive if isinstance(t, AbsorberConstitutiveTerm))
    if not held and not any(type(scene.boundaries.face(a + s)) is Absorber
                            for a in 'xyz' for s in '-+'):
        return
    from wavebit.fdtd._prepare.constitutive import terms
    all_expected = terms(scene, plan)
    expected = tuple(t for t in all_expected if isinstance(t, AbsorberConstitutiveTerm))
    if expected != held or (expected and tuple(plan.constitutive) != all_expected):
        raise ValueError('ConstitutiveAbsorberPlan: changed or missing damped flux memory '
                         'requires a new supported plan.')
    if held:
        from wavebit.fdtd._prepare.sources import source_terms
        if tuple(plan.terms) != all_expected + source_terms(scene, plan):
            raise ValueError('ConstitutiveAbsorberPlan: the complete current/source/memory '
                             'term tuple differs from this supported scene.')
        signature = _scope(scene, plan)
        if any(t.signature != signature for t in held):
            raise ValueError('ConstitutiveAbsorberPlan: the certified concrete scene changed.')


def validate_runtime(rt):
    """No implicit host copy may turn an unvalidated device into an admitted solve."""
    if rt.name not in ('numpy', 'jax', 'torch'):
        raise ValueError('ConstitutiveAbsorberRuntime: use a concrete CPU runtime.')
    if rt.name == 'torch' and rt.device.type != 'cpu':
        raise ValueError('ConstitutiveAbsorberRuntime: GPU execution is not admitted.')
    if rt.name == 'jax':
        platform = rt.jax.default_backend() if rt.device is None else rt.device.platform
        if platform != 'cpu':
            raise ValueError('ConstitutiveAbsorberRuntime: GPU execution is not admitted.')


def add_profiles(scene, plan, params, rt):
    """Add compact factors only when the plan owns a new damped term."""
    selected = tuple(t for t in plan.constitutive if isinstance(t, AbsorberConstitutiveTerm))
    if not selected:
        return
    _scope(scene, plan)
    validate_runtime(rt)
    axis = 'xyz'.index(plan.absorber_cells[0][0][0])
    e_rate = _damping(plan, {axis}, 'E')
    h_rate = _damping(plan, {axis}, 'H')
    half = (0.5 * plan.dt) * e_rate
    drive = 1.0 / (1.0 + half)
    keep = (1.0 - half) / (1.0 + half)
    if (not np.isfinite(e_rate).all() or not np.isfinite(h_rate).all()
            or not np.isfinite(half).all() or not np.isfinite(keep).all()
            or not np.isfinite(drive).all() or np.any(drive <= 0.0)):
        raise ValueError('ConstitutiveAbsorberRange: damping factors are outside float64 range.')
    tables = {}
    for term in selected:
        tables[term.name] = {key: rt.array(InteriorEnergy._cut(value, slice(None), term.box),
                                          'float64')
                             for key, value in (('keep', keep), ('drive', drive), ('rate', e_rate))}
    params['constitutive_absorber'] = tables
    params['constitutive_absorber_rate_h'] = rt.array(h_rate, 'float64')
    params['constitutive_absorber_drive_e'] = rt.array(drive, 'float64')


def physical_diagonal(term, params, rt):
    """Undo only the new route's drive before applying a physical K certificate."""
    key = 'be' if term.field == 'E' else 'bh'
    value = InteriorEnergy._cut(params[key], slice(None), term.box) / term.ratio
    if isinstance(term, AbsorberConstitutiveTerm):
        value = value / params['constitutive_absorber'][term.name]['drive']
    return rt.xp.broadcast_to(value, (3,) + term.region.shape)


def lifted_delta(term, params, delta):
    value = term.ratio * delta
    if isinstance(term, AbsorberConstitutiveTerm):
        value = params['constitutive_absorber'][term.name]['drive'] * value
    return value


def _neighbour(value, axis, field):
    """The other edge at this common node plus its finite-domain validity mask."""
    index = np.arange(value.shape[axis]) + (-1 if field == 'E' else 1)
    valid = (index >= 0) & (index < value.shape[axis])
    result = np.take(value, np.clip(index, 0, value.shape[axis] - 1), axis=axis)
    shape = [1, 1, 1]
    shape[axis] = value.shape[axis]
    return result, valid.reshape(shape)


def certify_links(cross, rates, field, name='constitutive'):
    """A sufficient exact finite-operator [K,R]=0 certificate, no tolerance."""
    cross = np.asarray(cross)
    rates = np.broadcast_to(np.asarray(rates), cross.shape)
    if (cross.ndim != 4 or cross.shape[0] != 3 or field not in ('E', 'H')
            or not np.isfinite(cross).all() or not np.isfinite(rates).all()
            or np.any(rates < 0)):
        raise ValueError('ConstitutiveAbsorberRange: invalid cross coefficients or rates.')
    for row, (i, j) in enumerate(PAIRS):
        active = cross[row] != 0.0
        other_i, valid_i = _neighbour(rates[i], 2 - i, field)
        other_j, valid_j = _neighbour(rates[j], 2 - j, field)
        bad = active & ((rates[i] != rates[j])
                        | (valid_i & (other_i != rates[j]))
                        | (valid_j & (rates[i] != other_j))
                        | (valid_i & valid_j & (other_i != other_j)))
        if np.any(bad):
            node = tuple(int(v) for v in np.argwhere(bad)[0])
            raise ValueError(f'ConstitutiveAbsorberCommutation: {name} {field} '
                             f'{"xyz"[i]}{"xyz"[j]} link at z/y/x={node} joins '
                             'unequal absorber rates. Keep this interpolation support '
                             'outside the layer, or use an aligned extrusion.')
    return True


def validate_links(plan, params, rt):
    """Concrete lowering rechecks links; an outer traced solve is not admitted."""
    for term in plan.constitutive:
        if not isinstance(term, AbsorberConstitutiveTerm):
            continue
        try:
            cross = np.asarray(rt.numpy(params['constitutive'][term.name]))
            rate = np.asarray(rt.numpy(params['constitutive_absorber'][term.name]['rate']))
        except Exception as err:
            raise ValueError('ConstitutiveAbsorberConcrete: link certification requires '
                             'concrete CPU coefficients, not an outer transformed solve.') from err
        diagonal = np.asarray(rt.numpy(physical_diagonal(term, params, rt)))
        if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
            raise ValueError('ConstitutiveAbsorberRange: no finite positive undamped diagonal.')
        certify_links(cross, rate, term.field, term.name)
