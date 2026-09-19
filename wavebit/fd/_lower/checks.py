"""The refusals and warnings stage 1 owns, between the scene and the plan.

Everything here reads *static* content -- the grid, the faces, the snapped structure spans --
and, where it can, the material numbers. A material number may be traced (a gradient with
respect to a permittivity is what stage 1's AD gate is for), and ``materials.concrete``
answers ``None`` for one; every check below asks first, and a check that cannot read its
number **says so in the report** rather than passing silently. The one place that is not
enough is the CFL bound, where an unreadable number cannot simply be waved through: there the
lowering carries a guard that turns every returned array into ``nan`` (``params.py``), which
is the stage-0 rule for a condition that can only be known at run time.

The table, and where each row lives:

    PML(cells=n) below 12                       boundaries.PML.__post_init__
    PML with no wavelength                      face_wavelength, here (stage 2: inferred
                                                  from the sources' band centre if they agree)
    a PML layer that does not fit its axis       check_faces, here
    a dispersive or conductive structure         check_absorbing_layers, here (Absorber)
      reaching a PML face
    a structure that ends inside an absorbing    check_absorbing_layers, here (PML and
      layer (PML or Absorber)                      Absorber alike)
    cells per wavelength in a material < 8       check_resolution, here (< 12 warns); with
                                                  sources, at band.shortest_wavelength, and in
                                                  vacuum too
    a structure within half a wavelength of      check_resolution, here (warns)
      an absorbing face
    complex eps                                 materials._value
    Slab naming two axes, any other shape       geometry.Slab / Structure
    pole CFL violated, concretely                check_stability, here
    two structures overlap, one dispersive      check_overlaps, here
    (stage 3) a Bloch phase on a one-cell axis    check_one_cell_bloch, here (names Axis1D;
      in a scene with a source                      a source-free kernel study keeps it)
    (stage 3) an Axis1D k past k_max              boundaries.Axis1D (concrete); the params'
                                                  guard (traced), which makes the run nan
"""
import math
import warnings

from ..diagnostics import AbsorberProximityWarning, CoarseMeshWarning, PMLStabilityWarning
from ..geometry import cell_box, describe, slab_axis
from ..materials import Conductor, Dispersive, concrete, eps_at

__all__ = ['face_wavelength', 'check_faces', 'check_overlaps', 'check_absorbing_layers',
           'check_stability', 'notes', 'cfl_bound', 'check_one_cell_bloch', 'cfl_axis_sum']


def face_wavelength(face, scene, name):
    """The wavelength a ``PML`` face sets ``alpha_max`` from, or a refusal naming the doors.

    In order: the face's own ``wavelength=``, the scene's ``Resolution(wavelength=)``, and
    (stage 2) the band centre of the scene's sources when every source has the same concrete
    one -- the frequency the layer will actually be asked to absorb.
    """
    if face.wavelength is not None:
        return face.wavelength
    if scene.resolution.wavelength is not None:
        return scene.resolution.wavelength
    from .sources import source_wavelength
    inferred = source_wavelength(scene)
    if inferred is not None:
        return inferred
    sources = tuple(getattr(scene, 'sources', ()) or ())
    why = ('the scene has no source to infer one from' if not sources else
           'its sources do not share one concrete band centre ('
           + ', '.join(repr(s.band.wavelength) for s in sources) + ')')
    raise ValueError(
        f'PML on {name!r} has no wavelength, and {why}. It sets alpha_max, the '
        f'complex-frequency shift, which is a frequency and so means nothing without one to '
        f'be relative to -- and a shift of zero is not the safe default, it is what stops the '
        f'layer absorbing the evanescent field a structure radiates into it. Say '
        f'fd.PML(cells=..., wavelength=...), or give the whole scene one with '
        f'fd.Resolution(cells_per_wavelength=..., wavelength=...).')


def check_faces(scene, grid, faces):
    """A PML or Absorber layer must fit inside its axis, with room between the two ends."""
    for a, axis in enumerate('xyz'):
        low, high = getattr(scene.boundaries, axis)
        # One object given for the axis still means two layers, one per face.
        total = getattr(low, 'cells', 0) + getattr(high, 'cells', 0)
        if total and total >= grid.n[a]:
            raise ValueError(
                f'the absorbing layers on {axis} are {total} cells deep in total but the axis '
                f'has only {grid.n[a]}: the two layers would meet, and a box that is all '
                f'absorber has no interior for a field to exist in. Give the axis more cells '
                f'({total + 1} at least, and in practice enough interior for the physics), or '
                f'fewer cells per layer.')


def check_one_cell_bloch(scene, grid):
    """A ``Bloch`` face on an axis one cell long, in a scene with a source: refused, naming
    ``Axis1D`` (stage 3).

    On one cell the Bloch difference ``(exp(i k dx) - 1)/dx`` is the only thing that axis
    contributes. Where the one cell stands for an axis the problem is invariant along -- the
    conical grating, a 2-D problem at an oblique ``k`` -- that is an artificial dispersion: the
    wave sees ``2 sin(k dx/2)/dx`` in place of ``k``, and ``fd.Axis1D(k=...)`` is the exact
    derivative. So a scene with a source (a PlaneWave, a Dipole, a TFSF -- the scenes a user
    asks a physical question of) is refused.

    **Where the door does not stand, and why.** The same one cell is also the exact stand-in
    for a *multi-cell* periodic grid's plane wave -- the difference is the one every cell of
    such a grid applies to ``exp(i k x)`` -- and that is what two things are built on: the
    stage-2 source columns (one cell across and Bloch by construction; they never reach this
    door, ``plan(_column=True)``) and the stage-1 kernel studies, whose ``InitialField`` packets
    run on a ``1 x 1 x nz`` line under a one-cell Bloch phase (the PML doubling table that
    ``PML(cells < 12)`` quotes, the Absorber and Airy gates, and the stage-1 golden's
    ``pml_oblique`` and ``conductor`` scenes). A source-free scene is therefore left to say what
    it means; the design's unconditional refusal would have removed the measurement the PML
    refusal is quoted from.
    """
    from ..boundaries import Bloch
    if not tuple(getattr(scene, 'sources', ()) or ()):
        return
    for a, axis in enumerate('xyz'):
        face = scene.boundaries.face(axis + '-')
        if isinstance(face, Bloch) and grid.n[a] == 1:
            given = 'k=...' if face.k is not None else 'phase=...'
            raise ValueError(
                f'Bloch({given}) on {axis}, an axis one cell long: the one-cell difference '
                f'(exp(i k dx) - 1)/dx is all that axis would contribute, and it is an artificial '
                f'dispersion -- the wave sees 2 sin(k dx/2)/dx instead of k. For the invariant axis '
                f'of a 2-D (conical) problem say fd.Axis1D(k=..., k_max=...), whose derivative is '
                f'i k exactly (fd.Bloch.from_source(pw, axis1d={axis!r}) writes it); for a real '
                f'period, give the axis more than one cell.')


def cfl_axis_sum(scene, dx):
    """``sum_a max K_a^2 / 4``: ``1/dx^2`` per differenced axis, ``k_max^2/4`` per ``Axis1D``.

    The leapfrog is stable while ``(dt / n)^2 sum_a max K_a^2 / 4 < 1``, with ``K_a`` the
    eigenvalue of the derivative along ``a``: ``2 sin(k dx/2)/dx`` for a difference (at most
    ``2/dx``), and ``k`` itself on an ``Axis1D`` axis -- a multiplication by ``i k``, bounded by
    nothing but the ``k_max`` the face declares. Three differenced axes give the stage-0
    ``3/dx^2``, i.e. ``dt < n dx / sqrt(3)``.
    """
    from ..boundaries import Axis1D
    total = 0.0
    for axis in 'xyz':
        face = scene.boundaries.face(axis + '-')
        total += (face.k_max ** 2 / 4.0) if isinstance(face, Axis1D) else 1.0 / dx ** 2
    return total


def check_overlaps(scene, grid):
    """Two structures may overlap, unless a :class:`~wavebit.fd.Dispersive` is party to it.

    "Structures are applied in the order they are given, so a later one overwrites an earlier
    one where they overlap" is the documented rule, and it is enforced in exactly one place:
    ``coefficients.owner_map``, which builds ``eps``, ``mu`` and ``sigma``. The **polarisation
    is not routed through that map**. ``plan._pole_terms`` gives every dispersive structure a
    term over its own snapped cells, so an overwritten dispersive medium keeps driving ``cH``
    from cells it no longer owns, and two overlapping dispersive structures add their
    polarisations twice. Measured on a 64-cell line: layering ``Material(eps=2.25)`` over a
    ``Dispersive(eps_inf=2.25, ...)`` on the same cells -- a scene the rule says must equal the
    glass alone -- gives a field 64% of the peak away from the glass scene and bit-identical to
    the *metal* scene, and ``(metal, metal)`` differs from ``(metal,)`` by 72% of its peak.

    Rather than route the term (a still-owned set is a box minus later boxes, which is not a
    box), stage 1 refuses the case at the door. Non-dispersive overlaps are untouched: they go
    through ``owner_map`` and the order rule holds for them exactly.

    Note that two slabs on *different* axes always intersect -- each is full-width on the
    other's axis -- so this refuses a dispersive slab crossed by any other structure, and names
    the two spellings that work.
    """
    structures = tuple(getattr(scene, 'structures', ()) or ())
    boxes = [cell_box(s.shape, grid) for s in structures]
    for i in range(len(structures)):
        for j in range(i + 1, len(structures)):
            if not _intersects(boxes[i], boxes[j]):
                continue
            kinds = [isinstance(structures[k].material, Dispersive) for k in (i, j)]
            if not any(kinds):
                continue
            which = f'structures[{i}]' if kinds[0] else f'structures[{j}]'
            raise ValueError(
                f'structures[{i}] ({describe(structures[i].shape)}) and '
                f'structures[{j}] ({describe(structures[j].shape)}) '
                f'share cells, and {which} is an fd.Dispersive. Where structures overlap a '
                f'later one overwrites an earlier one, but that rule is carried by the '
                f'permittivity map alone: a pole term runs over its own structure\'s whole '
                f'slab, so an overlapped dispersive medium would keep driving the field from '
                f'cells it no longer owns and two overlapping ones would count their '
                f'polarisation twice. Stage 1 refuses it rather than answer the wrong scene. '
                f'Say it as disjoint slabs (two slabs on the SAME axis with touching but '
                f'non-overlapping spans -- slabs on different axes always intersect), or as '
                f'one fd.Dispersive covering the whole of what it should fill.')


def _intersects(left, right):
    """Whether two cell boxes ``((i0, i1), (j0, j1), (k0, k1))`` share a cell.

    A Slab's box is full on its two other axes (``geometry.cell_box``), so two slabs on
    different axes always intersect, as stage 1 said; a Box is snapped on all three. A span
    that snapped to nothing was already refused where it was snapped.
    """
    return all(a0 < b1 and b0 < a1 for (a0, a1), (b0, b1) in zip(left, right))


def check_absorbing_layers(scene, plan):
    """What may sit inside an absorbing layer: no interface, and (in a PML) no loss or memory.

    Two different rules, on two different sets of faces:

    * **A structure that cuts the layer along its own axis**, ending strictly inside it, is
      refused on **every** absorbing face -- ``PML`` and ``Absorber`` alike. What the user
      wanted there was a termination and what the grid gives them is a mirror: the interface
      reflects and only the part of the layer in *front* of it is in the way of that
      reflection. Measured in ``tests/test_fd_absorber.py``, a lossless slab stopping half way
      into a twelve-cell layer reflects 1.9e-1 against 6.0e-4 running it through on a ``PML``
      face and 2.4e-1 against 1.2e-2 on an ``Absorber`` face -- 317x and 20x. (The
      ``Absorber`` half of this rule is not in the stage-1 design, which names the ``PML``
      case only; it is here because the measurement says the geometry is as bad on the face
      the ``PML`` refusal used to point at as an escape.)
    * **A material with loss or memory** -- a :class:`~wavebit.fd.Conductor` or a
      :class:`~wavebit.fd.Dispersive` -- is refused in a ``PML`` face only, naming
      ``Absorber``, which needs no match. See :func:`_refuse_lossy_in_layer` for what that
      refusal is and is not measured to protect against.

    A structure that runs *across* the layer -- a z slab inside an x PML, which is what a
    slab in an open box always is -- is uniform along the absorbing axis and is allowed. The
    stretch is still matched to the background, so the part of the layer that is not the
    background reflects more than the table says; that is the ``AbsorberProximityWarning``
    below, not a refusal. (The stage-1 design's table refuses this case outright. It is
    relaxed here because the canonical stage-1 scene -- a dielectric slab with
    ``Boundaries.open(...)`` -- is exactly it, and refusing the canonical scene would mean
    nothing could use the feature.)
    """
    grid = plan.grid
    pml = dict(plan.pml_cells)
    for name, cells in plan.absorbing_faces:
        kind = 'PML' if name in pml else 'Absorber'
        a, side = 'xyz'.index(name[0]), name[1]
        layer = (0, cells) if side == '-' else (grid.n[a] - cells, grid.n[a])
        for index, structure in enumerate(getattr(scene, 'structures', ()) or ()):
            box = cell_box(structure.shape, grid)
            k0, k1 = box[a]
            if k1 <= layer[0] or k0 >= layer[1]:
                continue                          # nowhere near the layer
            if k0 <= layer[0] and k1 >= layer[1]:
                # Runs across this layer and is uniform along it (a slab on another axis,
                # or a box that reaches through the face): no interface inside the layer,
                # but its material is in there.
                if kind == 'PML':
                    _refuse_lossy_in_layer(structure.material, name, f'structures[{index}]')
                continue
            raise ValueError(
                f'structures[{index}] ({describe(structure.shape)}, cells {k0}..{k1} along '
                f'{name[0]}) ends inside the {kind} on {name!r} (cells '
                f'{layer[0]}..{layer[1]}). An absorbing layer terminates the box; an '
                f'interface inside one is a mirror the layer cannot undo, because the '
                f'reflection leaves the way it came and only the cells in front of the '
                f'interface are in its way. (In a PML the stretched coordinate is complex '
                f'there as well, so nothing matches it at all.) Move the interface out of '
                f'the layer, or run the structure all the way through it.')
        if kind == 'PML':
            _refuse_lossy_in_layer(_background(scene), name, 'Scene(background=)')


def _background(scene):
    from ..materials import as_material
    return as_material(getattr(scene, 'background', None), 'Scene(background=)')


def _refuse_lossy_in_layer(material, name, where):
    """A ``Conductor`` or ``Dispersive`` reaching a PML face: refused, naming ``Absorber``.

    **What this door is measured to be worth, and what it is not.** A coordinate stretch is a
    change of variable, so it stays matched to a *uniform* medium whatever that medium's loss
    is; what a lossy medium costs the layer is the grading's calibration, which is normalised
    by ``sqrt(eps_r)`` of a background the medium is not. Measured with the door held open
    (``tests/test_fd_absorber.py``, the test named for what this refusal turns away), the same
    twelve-cell PML face reflects 9.9e-5 in vacuum, 5.2e-4 with
    ``Conductor(eps=2.25, sigma=0.2)`` running through it and 6.9e-4 with a Lorentz medium --
    a factor of five -- against 1.1e-2 for the ``fd.Absorber`` this message names. So the
    named remedy is **twenty-one times worse than the thing refused** at normal incidence, and
    the message says so rather than claiming the layer would not absorb.

    The door stands because of what that probe cannot see: for a *dispersive* medium the
    calibration is frequency dependent, and a CFS-PML inside a dispersive medium is the
    classic late-time instability. 20 000 steps at one probe, through a ninth of a wavelength
    of medium, left the tail at 1e-8 of the peak -- which is not a stability proof. Until
    something in this repository can be a stability proof, the refusal is the conservative
    answer and the numbers above are on record beside it.
    """
    if isinstance(material, (Conductor, Dispersive)):
        what = ('a conductivity' if isinstance(material, Conductor)
                else f'{len(material.poles)} pole(s)')
        raise ValueError(
            f'{where} is an fd.{type(material).__name__} ({what}) and reaches the PML on '
            f'{name!r}. A PML is graded against one real (eps, mu), and a lossy or dispersive '
            f'medium has no such pair at more than one frequency, so the layer is no longer '
            f'calibrated for what is in it -- measured, five times the reflection in vacuum '
            f'for a conductor -- and for a dispersive medium a CFS-PML in it is the classic '
            f'late-time instability, which nothing here can yet rule out. Use '
            f'fd.Absorber(cells=...) on that face -- a graded matched conductivity, which '
            f'needs no match, absorbs worse at every angle and hopelessly at grazing (measured '
            f'21x worse than the mismatched stretch this refuses, at normal incidence), and is '
            f'the fallback this case exists for.')


def cfl_bound(scene, plan):
    """``(bound, reason)``: the largest stable ``dt``, or ``(None, reason)`` if unreadable.

    The vacuum bound is ``n_min dx / sqrt(3)`` -- the *slowest* index, because a medium with
    ``eps < 1`` is faster than vacuum and the usual spelling that drops ``n_min`` lets a
    perfectly ordinary ``eps = 0.81`` diverge. The poles tighten it further
    (``wavebit/_core/grid_fdtd.py``, a kernel seed):

        dt < 1 / sqrt(1/dt_vac^2 + rate/4),   rate = max(w0^2) + max(sum strength/eps)

    ``rate`` is a rate squared: the fastest oscillation the coupled E-P system supports,
    which is the pole resonance plus the plasma-like coupling back into E. A Debye pole
    contributes ``delta_eps/tau^2`` for the strength term (``materials.Debye`` derives why
    that is the only combination with the right units, and says it is a bound by analogy).
    """
    background, structures = _background(scene), tuple(getattr(scene, 'structures', ()) or ())
    materials = [background] + [s.material for s in structures]
    n_min, resonance, coupling = None, 0.0, 0.0
    for material in materials:
        n = material.index_bound()
        if n is None:
            return None, 'a material value is traced, so the bound cannot be read here'
        n_min = n if n_min is None else min(n_min, n)
        strength_sum = 0.0
        for pole in material.poles:
            w0_squared, strength = pole.rate()
            w0_squared, strength = concrete(w0_squared), concrete(strength)
            if w0_squared is None or strength is None:
                return None, 'a pole value is traced, so the bound cannot be read here'
            resonance = max(resonance, w0_squared)
            strength_sum += strength
        eps_min = min(concrete(e) for e in material.eps)
        coupling = max(coupling, strength_sum / eps_min)
    axes = cfl_axis_sum(scene, plan.dx)
    if axes == 3.0 / plan.dx ** 2:
        vacuum = n_min * plan.dx / math.sqrt(3.0)          # the stage-0 float, to the bit
    else:
        vacuum = n_min / math.sqrt(axes)                   # an Axis1D face (stage 3)
    rate = resonance + coupling
    return 1.0 / math.sqrt(1.0 / vacuum ** 2 + rate / 4.0), None


def check_stability(scene, plan):
    """Refuse a concretely unstable ``dt``; return whether the bound could be read at all."""
    bound, reason = cfl_bound(scene, plan)
    if bound is None:
        return False
    if plan.dt < bound:
        return True
    poles = [(name, p) for name, p in _all_poles(scene)]
    worst = ''
    if poles:
        name, pole = max(poles, key=lambda np_: sum(abs(v) for v in _rate_values(np_[1])))
        worst = (f' The tightest pole is {name}: {type(pole).__name__}'
                 f'({", ".join(f"{f}={getattr(pole, f)!r}" for f in pole.__dataclass_fields__)}).')
    from ..boundaries import Axis1D
    axis1d = [f'{a} (k_max = {scene.boundaries.face(a + "-").k_max:g})' for a in 'xyz'
              if isinstance(scene.boundaries.face(a + '-'), Axis1D)]
    if axis1d:
        worst += (f' An fd.Axis1D axis adds k_max^2/4 to the sum under the bound in place of the 1/dx^2 a '
                  f'differenced axis adds (its derivative is i k, bounded by nothing but k_max): '
                  f'{", ".join(axis1d)}; a smaller k_max= widens the bound.')
    raise ValueError(
        f'dt = {plan.dt:g} is at or past the stability bound {bound:g} of this scene. The '
        f'vacuum condition dt < n_min dx / sqrt(3) is tightened by every pole to '
        f'dt < 1/sqrt(1/dt_vac^2 + rate/4) with rate = max(w0^2) + max(sum strength/eps), '
        f'because the coupled E-P system oscillates faster than the grid does.{worst} Lower '
        f'Run(courant=...) to at most {bound / (plan.dx / math.sqrt(3.0)):.3f}, or refine dx.')


def _all_poles(scene):
    background = _background(scene)
    for pole in background.poles:
        yield 'Scene(background=)', pole
    for index, structure in enumerate(getattr(scene, 'structures', ()) or ()):
        for pole in structure.material.poles:
            yield f'structures[{index}]', pole


def _rate_values(pole):
    return [concrete(v) or 0.0 for v in pole.rate()]


def notes(scene, plan):
    """``[(category, message), ...]``: the stage-1 warnings, for ``check`` and for ``plan``."""
    out = []
    out += _resolution_notes(scene, plan)
    out += _proximity_notes(scene, plan)
    out += _stabilize_notes(scene, plan)
    return out


def _stabilize_notes(scene, plan):
    """A stabilized PML face asking for less than the loss measured to suffice at its target."""
    from .cpml import MEASURED_STABILIZE, spell, sufficient_stabilize
    out = []
    for face in plan.cpml:
        if not face.stabilize:
            continue                    # the pure stretch: the decay warning speaks for it
        reflection = scene.boundaries.face(face.name).reflection
        need = sufficient_stabilize(reflection)
        if need is None:
            out.append((PMLStabilityWarning,
                        f'PML on {face.name!r}: stabilize={face.stabilize:g} at '
                        f'reflection={reflection:g}, a target weaker than any the stabilizing '
                        f'loss was measured at (the weakest is '
                        f'{spell(MEASURED_STABILIZE[-1][0])}, where {spell(MEASURED_STABILIZE[-1][1])} '
                        f'sufficed, and a weaker target needs more). A bound mode next to this '
                        f'face may still grow'))
        elif face.stabilize < need * (1.0 - 1e-12):
            out.append((PMLStabilityWarning,
                        f'PML on {face.name!r}: stabilize={face.stabilize:g} at '
                        f'reflection={reflection:g}; the measured value that stops a bound mode '
                        f'growing at this target is {spell(need)} (3e-3 holds only at the default '
                        f'1e-8 and stronger targets; wavebit/fd/_lower/cpml.py::'
                        f'MEASURED_STABILIZE). Below it, a lossless structure with a bound '
                        f'mode next to this face may grow instead of settling'))
    return out


def _wavelength(scene, plan):
    if getattr(scene, 'sources', ()):
        from .sources import shortest_wavelength
        return shortest_wavelength(scene)
    if scene.resolution.wavelength is not None:
        return scene.resolution.wavelength
    lengths = [w for _, w in plan.pml_wavelengths]
    return lengths[0] if lengths else None


def _index_at(material, wavelength):
    """``(n_real, skin_depth)`` of ``material`` at ``wavelength``, or ``(None, None)``."""
    values = []
    for c in range(3):
        eps = concrete(material.eps[c])
        mu = concrete(material.mu_value[c])
        if eps is None or mu is None:
            return None, None
        if material.poles:
            eps = eps_at(material.poles, wavelength, eps)
            if any(concrete(getattr(p, f)) is None
                   for p in material.poles for f in p.__dataclass_fields__):
                return None, None
        values.append(complex(eps) * mu)
    worst = max(values, key=lambda v: abs(v))
    n = complex(worst) ** 0.5
    if n.imag < 0:
        n = -n                                  # Im(n) > 0 is loss, in this convention
    skin = None
    if n.real < abs(n.imag):
        k0 = 2.0 * math.pi / wavelength
        skin = 1.0 / (k0 * n.imag) if n.imag > 0 else None
    return n.real, skin


def _resolution_notes(scene, plan):
    """Cells per wavelength inside every material; a metal by its skin depth instead."""
    materials = [('Scene(background=)', _background(scene))]
    materials += [(f'structures[{i}]', s.material)
                  for i, s in enumerate(getattr(scene, 'structures', ()) or ())]
    sources = tuple(getattr(scene, 'sources', ()) or ())
    if not sources and len(materials) == 1 and not materials[0][1].poles \
            and _index_at(materials[0][1], 1.0)[0] == 1.0:
        return []                     # vacuum: there is nothing for a mesh to be coarse for
    wavelength = _wavelength(scene, plan)
    if wavelength is None and sources:
        return [(CoarseMeshWarning,
                 'a source band is traced, so the cells-per-wavelength check at '
                 'band.shortest_wavelength did not run')]
    if wavelength is None:
        return [(CoarseMeshWarning,
                 'this scene has a material but no wavelength (no Resolution(wavelength=) and '
                 'no PML carrying one), so the cells-per-wavelength check did not run -- the '
                 'mesh may be too coarse for it and nothing here can tell')]
    out = []
    for where, material in materials:
        n, skin = _index_at(material, wavelength)
        if n is None:
            out.append((CoarseMeshWarning,
                        f'{where} has a traced value, so its cells per wavelength could not '
                        f'be checked'))
            continue
        if skin is not None:
            cells = skin / plan.dx
            if cells < 1.0:
                raise ValueError(
                    f'{where} has a skin depth of {skin:g} at wavelength {wavelength:g}, which '
                    f'is {cells:.2f} cells: the field decays to nothing inside one cell, so '
                    f'the grid cannot represent the surface at all and the reflectance it '
                    f'returns is the grid\'s, not the metal\'s. Refine dx below {skin:g}, or '
                    f'model the metal as fd.PEC-backed geometry. (This line is stated from the '
                    f'skin-depth argument, not measured on this kernel.)')
            if cells < 2.0:
                out.append((CoarseMeshWarning,
                            f'{where}: {cells:.2f} cells per skin depth at wavelength '
                            f'{wavelength:g}; two is the fewest that resolves the decay'))
            continue
        cells = wavelength / (n * plan.dx)
        if cells < 8.0:
            raise ValueError(
                f'{where} is sampled at {cells:.2f} cells per wavelength (index {n:.3f}, '
                f'wavelength {wavelength:g}{_which(sources)}, dx {plan.dx:g}). The measured convergence behind '
                f'the legacy engine\'s CELLS_PER_WAVELENGTH runs a worst per-order error of '
                f'1.2e-2 at ~11 cells and 2.8e-3 at 23, so below 8 not even the first digit '
                f'survives and the run would return a number shaped like an answer. Refine dx '
                f'to at most {wavelength / (8.0 * n):g}, or say a coarser wavelength.')
        if cells < 12.0:
            out.append((CoarseMeshWarning,
                        f'{where}: {cells:.2f} cells per wavelength (index {n:.3f}); 12 is '
                        f'where the second digit starts to hold and 20 is the usual rule'))
    return out


def _which(sources):
    return ", the sources' band.shortest_wavelength" if sources else ''


def _proximity_notes(scene, plan):
    """A structure within half a wavelength, in its own medium, of an absorbing face."""
    wavelength = _wavelength(scene, plan)
    if wavelength is None or not plan.absorbing_faces:
        return []
    out, grid = [], plan.grid
    for index, structure in enumerate(getattr(scene, 'structures', ()) or ()):
        n, _ = _index_at(structure.material, wavelength)
        if n is None or n <= 0:
            continue
        clearance = 0.5 * wavelength / n
        box = cell_box(structure.shape, grid)
        for name, cells in plan.absorbing_faces:
            fa, side = 'xyz'.index(name[0]), name[1]
            k0, k1 = box[fa]
            if (k0, k1) == (0, grid.n[fa]) and slab_axis(structure.shape) != fa:
                continue                # a transverse face: the structure runs through it
            lo = grid.low[fa] + k0 * grid.dx
            hi = grid.low[fa] + k1 * grid.dx
            inner = (grid.low[fa] + cells * grid.dx if side == '-'
                     else grid.high[fa] - cells * grid.dx)
            gap = (lo - inner) if side == '-' else (inner - hi)
            if gap < clearance:
                out.append((AbsorberProximityWarning,
                            f'structures[{index}] is {gap:g} from the inner face of the '
                            f'absorber on {name!r}; half a wavelength in its own medium is '
                            f'{clearance:g}. The measured optimum clearance is half a '
                            f'wavelength (legacy PAD_WAVELENGTHS: 6.57e-4 worst per-order '
                            f'error at 0.5 against 8.81e-4 at 0.125), and the layer is '
                            f'matched to the background, not to this structure\'s near field'))
    return out


def warn_all(scene, plan):
    """Emit every note as a real warning; ``plan`` calls this, ``report`` lists them instead."""
    for category, message in notes(scene, plan):
        warnings.warn(message, category, stacklevel=3)
