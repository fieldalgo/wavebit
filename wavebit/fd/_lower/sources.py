"""Stage 2's lowering: the source terms of a plan, their traced leaves, and the refusals they own.

The refusals, and where each lives:

    a source that writes a cell of a PEC/PMC/Symmetry wall   _check_indices, here
      layer, inside an absorbing layer, or across a
      periodic seam
    a TFSF box that touches an absorbing layer or crosses    _tfsf_term -> _check_indices, here
      a face; a TFSF whose propagation axis is not
      terminated by PML/Absorber faces
    a structure that crosses a TFSF face and is not part of  _tfsf_term, here
      the aux grid's 1-D background
    TFSF with an oblique direction                           sources.TFSF (the measurement)
    PlaneWave on non-Periodic/Bloch/Axis1D transverse        _plane_wave_term, here
      faces, or a nonzero transverse k on a Periodic face
    an Axis1D k that is not the PlaneWave's k_par (stage 3)  _plane_wave_term, here
    a source spanning cells along an Axis1D axis (a Dipole   _check_indices, here
      stamp, a TFSF box; stage 3)
    a Bloch phase inconsistent with the PlaneWave's k_par    _plane_wave_term, here
    a structure at a PlaneWave sheet, or any structure       _plane_wave_term, here
      that is not a slab across the propagation axis
    a layered TFSF whose incident side is not one uniform    _tfsf_driver, here
      medium (the driver column needs one), or is a
      Dispersive one
    a PlaneWave whose column needs a PML at a traced band    _column_wavelength, here
      centre (a non-absorbing face along its axis)
    a complex waveform on a real run                         lower_sources, here
    Dipole trilinear / centre outside range                  sources.Dipole
    Dipole support (its range) crossing a wall               _dipole_term -> _check_indices
    Run(periods=) with no source, disagreeing or traced      run_steps, here
      band centres
    a PML with no wavelength -- now inferred from the        checks.face_wavelength
      sources' band centre when they agree
    cells per wavelength at band.shortest_wavelength         checks._resolution_notes
"""
import math
import warnings

import numpy as np

from wavebit._core.backend import check_input_precision

from ..boundaries import AXES, PML_CELLS, Absorbing, Axis1D, Bloch, Periodic, PML
from ..geometry import cell_box, describe, slab_axis
from ..materials import Dispersive, as_material, concrete
from ..sources import Dipole, PlaneWave, TFSF, _is_complex
from .._kernel.grid import OFFSETS
from .._kernel.masks import Slab
from .._kernel.sources import ColumnTerm, Correction, DipoleTerm, box_corrections, stamp

__all__ = ['source_terms', 'lower_sources', 'source_wavelength', 'run_steps',
           'shortest_wavelength', 'column_scene', 'column_pads', 'dipole_indices']


def source_wavelength(scene):
    """The band centre every source agrees on, concretely; ``None`` otherwise (or no source)."""
    centres = [s.band.concrete_wavelength for s in getattr(scene, 'sources', ()) or ()]
    if not centres or any(c is None for c in centres):
        return None
    if any(abs(c - centres[0]) > 1e-12 * centres[0] for c in centres[1:]):
        return None
    return centres[0]


def shortest_wavelength(scene):
    """``min(band.shortest_wavelength)`` over the sources; ``None`` if none or any traced."""
    values = [s.band.shortest_wavelength for s in getattr(scene, 'sources', ()) or ()]
    if not values or any(v is None for v in values):
        return None
    return min(values)


def run_steps(scene, dt):
    """The step count: ``Run(steps=)``, ``ceil(time / dt)`` or ``ceil(periods * wavelength / dt)``."""
    if scene.run.steps is not None:
        return scene.run.steps
    if getattr(scene.run, 'time', None) is not None:
        return max(1, int(math.ceil(scene.run.time / dt - 1e-9)))
    sources = tuple(getattr(scene, 'sources', ()) or ())
    if not sources:
        raise ValueError('Run(periods=...) counts optical periods of the sources\' band centre, '
                         'and this scene has no source; say Run(steps=...).')
    centres = [s.band.concrete_wavelength for s in sources]
    if any(c is None for c in centres):
        raise ValueError('Run(periods=...) needs a concrete band centre, and a source\'s '
                         'Band(wavelength=...) is traced (a step count fixes the scan and '
                         'cannot be a leaf); say Run(steps=...).')
    if any(abs(c - centres[0]) > 1e-12 * centres[0] for c in centres[1:]):
        raise ValueError(f'Run(periods=...): the sources\' band centres disagree ({centres}), '
                         f'so "a period" names no one length; say Run(steps=...).')
    return int(math.ceil(scene.run.periods * centres[0] / dt - 1e-9))


# ---------------------------------------------------------------------------------------------
# Plan time: the terms
# ---------------------------------------------------------------------------------------------

def _check_indices(what, axis, lo, hi, scene, plan):
    """Refuse ``what`` writing indices ``lo..hi`` (inclusive) along ``axis`` past a face.

    A wrapping face refuses a support that crosses the periodic seam (nothing here writes
    across it); an absorbing face refuses one that reaches into its layer; a PEC, PMC or
    Symmetry face refuses one that touches the edge layer, where the wall's mask or mirror
    decides the field and a source there would drive the wall rather than the interior.
    """
    n = plan.grid.n[axis]
    absorbing = dict(plan.absorbing_faces)
    name = AXES[axis]
    for side in ('-', '+'):
        face_name = name + side
        face = scene.boundaries.face(face_name)
        if isinstance(face, Axis1D):
            ok = lo == 0 and hi == 0
            reason = (f'spans cells along an fd.Axis1D axis, which holds one cell of a field '
                      f'exp(i k {name}); a source there must be uniform along it (a PlaneWave '
                      f"is), and a Dipole's stamp or a TFSF box is not -- not in this build")
        elif isinstance(face, (Periodic, Bloch)):
            ok = lo >= 0 if side == '-' else hi <= n - 1
            reason = (f'crosses the periodic seam of the box on {face_name!r}; a source '
                      f'that wraps around the cell is not supported -- move it inside')
        elif face_name in absorbing:
            cells = absorbing[face_name]
            ok = lo >= cells if side == '-' else hi <= n - 1 - cells
            kind = type(face).__name__
            reason = (f'reaches into the {kind} on {face_name!r} ({cells} cells, so indices '
                      f'{cells}..{n - 1 - cells} are clear of both layers on {name}); a '
                      f'source inside an absorbing layer is absorbed as it is made and its '
                      f'reference series no longer describes what reached the interior')
        else:
            ok = lo >= 1 if side == '-' else hi <= n - 2
            reason = (f'touches the {type(face).__name__} wall on {face_name!r}, whose edge '
                      f'layer the wall\'s mask or mirror decides; keep one cell clear '
                      f'(indices 1..{n - 2} on {name})')
        if not ok:
            raise ValueError(f'{what} writes indices {lo}..{hi} along {name} (of {n}) and '
                             f'{reason}.')


def source_terms(scene, plan):
    """The source terms of ``scene`` under the stage-0/1 ``plan``, rows assigned in order."""
    terms, row = [], 0
    for src in getattr(scene, 'sources', ()) or ():
        if isinstance(src, Dipole):
            term = _dipole_term(scene, plan, src, row)
        elif isinstance(src, PlaneWave):
            term = _plane_wave_term(scene, plan, src, row)
        elif isinstance(src, TFSF):
            term = _tfsf_term(scene, plan, src, row)
        else:                                       # Scene already refused anything else
            raise TypeError(type(src).__name__)
        terms.append(term)
        row += len(term.rows)
    return tuple(terms)


def dipole_indices(src, grid):
    """``((i0, i1), ...)`` per axis: the first and last index (inclusive) the stamp writes.

    The stamp is zero from two cells out (``stamp_weights``), so these are the points of the
    dipole's component within two cells of its span, the moving range included. The monitors
    (``monitors.py`` beside this one) read the same range to keep a flux surface off the stamp.
    """
    field, c = src.field, src.index
    out = []
    for a in range(3):
        offset = 0.5 * OFFSETS[field][c][a]
        lo, hi = src.span[a]
        i0 = int(math.floor((lo - grid.low[a]) / grid.dx - offset - 2.0)) + 1
        i1 = int(math.ceil((hi - grid.low[a]) / grid.dx - offset + 2.0)) - 1
        out.append((i0, i1))
    return tuple(out)


def _dipole_term(scene, plan, src, row):
    grid = plan.grid
    field, c = src.field, src.index
    ranges = []
    what = f'Dipole({src.name!r})' + ('' if src.range is None else ' over its range')
    for a, (i0, i1) in enumerate(dipole_indices(src, grid)):
        _check_indices(what, a, i0, i1, scene, plan)
        ranges.append((i0, i1 + 1))
    slab = Slab(c=c, z=ranges[2], y=ranges[1], x=ranges[0])
    return DipoleTerm(name=src.name, field=field, slab=slab, row=row, dx=grid.dx,
                      stamp=src.stamp)


def _node_index(grid, axis, position):
    return int(round((position - grid.low[axis]) / grid.dx))


def _structure_cells(structure, grid):
    """``(slab axis or None, cell box)`` of a structure: its axis if a Slab, and its cells."""
    return slab_axis(structure.shape), cell_box(structure.shape, grid)


def _plane_wave_term(scene, plan, src, row):
    grid, P = plan.grid, src.axis
    transverse = [a for a in range(3) if a != P]
    faces = {a: scene.boundaries.face(AXES[a] + '-') for a in transverse}
    for a, face in faces.items():
        if not isinstance(face, (Periodic, Bloch, Axis1D)):
            raise ValueError(
                f'PlaneWave({src.name!r}) needs a periodic cell across its propagation axis, '
                f'but the {AXES[a]} faces are {type(face).__name__}. A plane wave is infinite '
                f'across; on a finite box use fd.TFSF, and in a periodic cell say '
                f'fd.Boundaries(**fd.Bloch.from_source(pw), {AXES[P]}=...), which puts '
                f'Periodic or the matching Bloch phase on both transverse axes.')
    k, e, _ = src.vectors()
    needs_k = any(abs(k[a]) >= 1e-12 for a in transverse) \
        or any(isinstance(f, (Bloch, Axis1D)) for f in faces.values())
    if needs_k:
        wavelength = src.band.concrete_wavelength
        index = as_material(scene.background, 'Scene(background=)').index_bound()
        if wavelength is None or index is None:
            raise ValueError(f'PlaneWave({src.name!r}) at theta = {src.theta:g}: the transverse '
                             f'wavevector the Bloch faces must carry is fixed at the band '
                             f'centre in the background, and one of the two is traced here.')
        k0 = index * 2.0 * math.pi / wavelength
        for a in transverse:
            face, expected = faces[a], k0 * k[a]
            if isinstance(face, Periodic):
                if abs(k[a]) >= 1e-12:
                    raise ValueError(
                        f'PlaneWave({src.name!r}, theta={src.theta:g}, phi={src.phi:g}) has a '
                        f'transverse wavevector {expected:.6g} along {AXES[a]}, but that axis '
                        f'is Periodic, which carries only k = 0 (mod 2 pi/L). Say '
                        f'fd.Boundaries(**fd.Bloch.from_source(pw, background=...), ...).')
                continue
            if isinstance(face, Axis1D):
                given = concrete(face.k)
                if given is not None and abs(given - expected) > 1e-9 * max(abs(expected),
                                                                              k0):
                    raise ValueError(
                        f'PlaneWave({src.name!r}): the Axis1D face on {AXES[a]} carries k = '
                        f"{given:.9g} but the wave's transverse wavevector at the band centre is "
                        f'{expected:.9g} (n = {index:g}, wavelength {wavelength:g}, theta '
                        f'{src.theta:g}, phi {src.phi:g}). Use fd.Bloch.from_source(pw, '
                        f'axis1d={AXES[a]!r}, background=...).')
                continue
            if abs(k[a]) < 1e-12:
                raise ValueError(
                    f'PlaneWave({src.name!r}) has no transverse wavevector along {AXES[a]}, '
                    f'but that axis is Bloch; the phase is inconsistent with the wave. Say '
                    f'Periodic() there -- fd.Bloch.from_source(pw) writes both axes.')
            L = grid.size[a]
            given = concrete(face.k) if face.k is not None else (
                None if concrete(face.phase) is None else concrete(face.phase) / L)
            if given is None:
                continue                    # a traced phase: nothing to compare against
            if abs(given - expected) > 1e-9 * max(abs(expected), 2.0 * math.pi / L):
                raise ValueError(
                    f'PlaneWave({src.name!r}): the Bloch face on {AXES[a]} carries k = '
                    f'{given:.9g} but the wave\'s transverse wavevector at the band centre is '
                    f'{expected:.9g} (n = {index:g}, wavelength {wavelength:g}, theta '
                    f'{src.theta:g}, phi {src.phi:g}). A sheet launched with one k into a cell '
                    f'that imposes another is not a plane wave of either. Use '
                    f'fd.Bloch.from_source(pw, background=...).')
    n = grid.n
    k0i = _node_index(grid, P, src.position)
    used = (k0i - 1, k0i) if src.sign > 0 else (k0i, k0i + 1)
    _check_indices(f'PlaneWave({src.name!r})', P, used[0], used[1], scene, plan)
    for index, structure in enumerate(scene.structures):
        s_axis, box = _structure_cells(structure, grid)
        c0, c1 = box[P]
        if s_axis is not None and s_axis != P:
            raise ValueError(
                f'PlaneWave({src.name!r}) along {AXES[P]} with structures[{index}], a slab '
                f'across {structure.shape.axis}: the structure is not uniform across the '
                f'sheet, so the sheet would not be launching into the medium its incident '
                f'field is computed in. A PlaneWave takes layers across its propagation axis, '
                f'and (stage 3) cell-snapped fd.Box structures -- a patterned layer of finite '
                f'height -- that stay clear of the sheet; other patterned shapes are stage 4.')
        if c0 <= used[1] and c1 > used[0]:
            raise ValueError(
                f'PlaneWave({src.name!r}) sits at {AXES[P]} = {src.position:g} (cells '
                f'{used[0]}..{used[1]}), inside structures[{index}] (cells {c0}..{c1 - 1}). '
                f'The sheet\'s incident field is computed in the uniform background, so the '
                f'sheet must sit in the background: move it at least one cell clear.')
    lo, hi = [0, 0, 0], list(n)
    if src.sign > 0:
        lo[P], side = k0i, '-'
    else:
        hi[P], side = k0i, '+'
    pattern = plan.dtype == 'complex128' and any(isinstance(f, Bloch) for f in faces.values())
    # Column indices are the main grid's plus the column's low padding (a PML the column
    # carries where the scene's own face along P would reflect).
    off = column_pads(scene, src)[0]
    corrections = tuple(
        Correction(slot=slot, slab=slab, src=s, p0=read + off, p1=read + off + 1, sign=sign,
                   pattern=pattern)
        for slot, slab, s, a, read, sign in box_corrections(
            lo, hi, transverse, transverse, n, faces=((P, side),)))
    drive = tuple((c, float(e[c])) for c in transverse if abs(e[c]) > 1e-15)
    ks = k0i - 1 if src.sign > 0 else k0i + 1
    entry_H = k0i if src.sign > 0 else k0i - 1
    plane = grid.low[P] + k0i * grid.dx
    sub = _column_plan(scene, src, plan)
    return ColumnTerm(name=src.name, kind='PlaneWave', row=row, axis=P, sub=sub, drive=drive,
                      ks=ks + off, corrections=corrections, entry_E=k0i + off,
                      entry_H=entry_H + off, plane_E=plane,
                      plane_H=plane + src.sign * 0.5 * grid.dx)


def _tfsf_term(scene, plan, src, row):
    grid, P, pol = plan.grid, src.axis, src.pol
    h = 3 - P - pol
    n = grid.n
    for side in ('-', '+'):
        face = scene.boundaries.face(AXES[P] + side)
        if not isinstance(face, Absorbing):
            raise ValueError(
                f'TFSF({src.name!r}) propagates along {AXES[P]}, whose {AXES[P] + side!r} face '
                f'is {type(face).__name__}. The incident field comes from a 1-D auxiliary '
                f'grid on the scene\'s own cells and faces along that axis, and it must leave '
                f'through them: make both {AXES[P]} faces fd.PML(...) or fd.Absorber(...).')
    lo, hi = [0, 0, 0], [0, 0, 0]
    for a in range(3):
        lo[a] = _node_index(grid, a, src.box.low[a])
        hi[a] = _node_index(grid, a, src.box.high[a])
        if hi[a] <= lo[a]:
            raise ValueError(f'TFSF({src.name!r}): the box snaps to no cell along {AXES[a]} '
                             f'(size {src.box.size[a]:g} at dx = {grid.dx:g}).')
        _check_indices(f'TFSF({src.name!r}) box with its one-cell correction layer', a,
                       lo[a] - 1, hi[a] + 1, scene, plan)
    for index, structure in enumerate(scene.structures):
        s_axis, box = _structure_cells(structure, grid)
        if s_axis == P:
            continue                            # part of the aux grid's layered background
        if s_axis is None:
            _check_scatterer(src, index, structure, box, lo, hi)
            continue
        c0, c1 = box[s_axis]
        if c0 <= hi[s_axis] and c1 > lo[s_axis] - 1:
            raise ValueError(
                f'structures[{index}] (a slab across {structure.shape.axis}, cells {c0}..'
                f'{c1 - 1}) crosses the faces of TFSF({src.name!r}) (cells {lo[s_axis] - 1}..'
                f'{hi[s_axis]} along {structure.shape.axis}). The incident field comes from a '
                f'1-D grid along {AXES[P]} whose background is the slabs across {AXES[P]} '
                f'only, so it is wrong wherever this structure changes the medium on the box '
                f'surface. A slab across {AXES[P]} is part of that background and may cross '
                f'the box; a finite scatterer is an fd.Box structure clear of its faces '
                f'(stage 3; every other shape is a stage-4 shape).')
    layered = any(slab_axis(s.shape) == P for s in scene.structures)
    driver, sheet = None, ()
    # The one read upstream of the entry plane: the tangential H half a cell outside it,
    # which the E face on the entry side reads. In a layered column that half cell is
    # upstream of the column's own sheet, so the incident field there is column + driver.
    outside_H = lo[P] - 1 if src.sign > 0 else hi[P]
    out = []
    for slot, slab, s, a, read, sign in box_corrections(lo, hi, (pol,), (h,), n):
        if a == P:
            p0, p1 = read, read + 1
        else:
            p0, p1 = {0: slab.x, 1: slab.y, 2: slab.z}[P]
        out.append(Correction(slot=slot, slab=slab, src=s, p0=p0, p1=p1, sign=sign,
                              driver=layered and slot == 'E' and a == P and read == outside_H))
    if src.sign > 0:
        ks, entry_E, entry_H = lo[P] - 1, lo[P], lo[P]
    else:
        ks, entry_E, entry_H = hi[P] + 1, hi[P], hi[P] - 1
    plane = grid.low[P] + entry_E * grid.dx
    sub = _column_plan(scene, src, plan)
    if layered:
        _tfsf_driver(scene, src, grid)                  # refuses a non-uniform incident side
        driver = _column_plan(scene, src, plan, role='driver')
        # The column's sheet: a one-sided TFSF plane on the entry plane, total field on the
        # box's side -- the PlaneWave construction on the column's 1 x 1 cross-section.
        clo, chi = [0, 0, 0], list(sub.grid.n)
        if src.sign > 0:
            clo[P], side = lo[P], '-'
        else:
            chi[P], side = hi[P], '+'
        sheet = tuple(
            Correction(slot=slot, slab=slab, src=s, p0=read, p1=read + 1, sign=sign)
            for slot, slab, s, a, read, sign in box_corrections(
                clo, chi, (pol,), (h,), sub.grid.n, faces=((P, side),)))
    return ColumnTerm(name=src.name, kind='TFSF', row=row, axis=P, sub=sub,
                      drive=((pol, 1.0),), ks=ks, corrections=tuple(out), entry_E=entry_E,
                      entry_H=entry_H, plane_E=plane,
                      plane_H=plane + src.sign * 0.5 * grid.dx, driver=driver, sheet=sheet)


def _check_scatterer(src, index, structure, box, lo, hi):
    """A cell-snapped Box near a TFSF box: strictly inside it, or clear of its surface.

    The corrections read the incident field on the node planes ``lo`` and ``hi`` and write
    the cells one outside them, so a scatterer must keep one cell clear of every face on the
    inside (cells ``lo+1 .. hi-2``) or stay entirely outside the shell ``lo-1 .. hi`` on some
    axis. Anything else changes the medium on the box's surface, where the column's 1-D
    background is what the incident field was computed in.
    """
    inside = all(lo[a] + 1 <= box[a][0] and box[a][1] <= hi[a] - 1 for a in range(3))
    outside = any(box[a][1] <= lo[a] - 1 or box[a][0] >= hi[a] + 1 for a in range(3))
    if inside or outside:
        return
    raise ValueError(
        f'structures[{index}] ({describe(structure.shape)}, cells '
        f'{tuple(tuple(r) for r in box)}) touches the surface of TFSF({src.name!r}) (nodes '
        f'{tuple(lo)}..{tuple(hi)}). The incident field on the box\'s faces comes from a 1-D '
        f'grid whose background is the slabs across the propagation axis, so a scatterer must '
        f'sit strictly inside the box with one cell to spare on every side, or outside it '
        f'clear of the one-cell correction layer.')


def _box_nodes(grid, box):
    return ([_node_index(grid, a, box.low[a]) for a in range(3)],
            [_node_index(grid, a, box.high[a]) for a in range(3)])


def _tfsf_driver(scene, src, grid):
    """The medium a layered TFSF's wave arrives from: its driver column's background.

    The driver is a uniform column whose downward wave the sheet hands to the layered column,
    and that hand-over is exact only if the layered column is that same medium everywhere
    upstream of the entry plane: every cell from the entry plane out through the incident
    side's face (cells ``0 .. lo-1`` for a wave in ``+P``, ``hi .. n-1`` for one in ``-P`` --
    the cells the scattered-region E and H points there draw their coefficients from). A
    medium that changes in there has no incident plane wave of one medium to hand over, so
    it is refused; a slab that covers all of it (the wave arriving from inside a substrate)
    is the driver's background.
    """
    P = src.axis
    lo, hi = _box_nodes(grid, src.box)
    cells = range(0, lo[P]) if src.sign > 0 else range(hi[P], grid.n[P])
    owners = set()
    spans = [(i, s, s.shape.cells(grid)) for i, s in enumerate(scene.structures)
             if slab_axis(s.shape) == P]
    for j in cells:
        owner = None
        for i, _, (c0, c1) in spans:                # a later structure overwrites
            if c0 <= j < c1:
                owner = i
        owners.add(owner)
    if len(owners) > 1:
        named = ', '.join('the background' if o is None else f'structures[{o}]'
                          for o in sorted(owners, key=lambda o: -1 if o is None else o))
        raise ValueError(
            f'TFSF({src.name!r}) propagates in {"+" if src.sign > 0 else "-"}{AXES[P]}, '
            f'and its incident side (the cells between the box\'s entry plane and the '
            f'{AXES[P]}{"-" if src.sign > 0 else "+"} face) is not one medium: it holds '
            f'{named}. The incident field is a plane wave arriving from a uniform half space, '
            f'handed to the layered column on the entry plane, and a layer on the incident '
            f'side has no such wave. Move the box so its entry plane is past every layer on '
            f'that side, or extend the layer out through the face.')
    owner = owners.pop() if owners else None
    material = scene.background if owner is None else scene.structures[owner].material
    material = as_material(material, 'Scene(background=)')
    if isinstance(material, Dispersive):
        raise ValueError(
            f'TFSF({src.name!r}): the wave arrives from structures[{owner}], a Dispersive '
            f'medium. The driver column that carries the incident wave is uniform in that '
            f'medium, and a dispersive background is not supported (it has no pole term '
            f'outside a structure). Put the box\'s entry plane in a non-dispersive layer.')
    return material


# ---------------------------------------------------------------------------------------------
# The column: a 1 x 1 scene along the propagation axis
# ---------------------------------------------------------------------------------------------

def column_pads(scene, src):
    """``(low, high)``: the cells a column carries past the scene's own along ``src.axis``.

    A ``PlaneWave`` column models the free incident wave in the uniform background, and it is
    driven by a hard source, which reflects anything coming back up it (``r = -1``). So its
    two ends must absorb: where the scene's face along the axis does not (a PEC, PMC,
    Symmetry, Periodic or Bloch face), the column puts a ``PML`` of :data:`PML_CELLS` cells
    on cells of its own past that face. Without it a PEC-backed cell turned the column into
    a lossless cavity, the mirror's reflection was counted as incident, and the sheet
    cancelled it -- no reflection ever appeared above the sheet. The main grid keeps its own
    face, so the mirror's reflection is scattered field and crosses the sheet. A ``TFSF``
    already refuses a non-absorbing face on its axis, so its column is never padded.
    """
    if not isinstance(src, PlaneWave):
        return 0, 0
    P = AXES[src.axis]
    return tuple(0 if isinstance(scene.boundaries.face(P + side), Absorbing) else PML_CELLS
                 for side in ('-', '+'))


def _column_wavelength(scene, src):
    """The wavelength a padded column's PML is matched to: the band centre, concretely."""
    wavelength = src.band.concrete_wavelength
    if wavelength is None:
        wavelength = scene.resolution.wavelength
    if wavelength is None:
        raise ValueError(
            f'PlaneWave({src.name!r}) has a non-absorbing face along {AXES[src.axis]}, so its '
            f'incident column carries a PML of its own past that face (the column must not '
            f'reflect), and that PML needs a concrete wavelength; the band centre is traced '
            f'here. Give the scene fd.Resolution(wavelength=...), or a concrete '
            f'Band(wavelength=...).')
    return wavelength


def column_scene(scene, src, plan, role='column'):
    """The column's own scene: the scene's cells along ``src.axis``, 1 x 1 across.

    Across, the column is ``Periodic`` (a TFSF, or a PlaneWave with no transverse k) or
    ``Bloch`` with the main cell's phase per cell (``phase / n``, or the same ``k``), so the
    one-cell difference is exactly the main grid's. Along, it keeps the scene's two faces,
    with a PML's inferred wavelength written in (the column has no source to infer it from),
    except that a PlaneWave column replaces a non-absorbing face by a PML on padded cells
    (:func:`column_pads`). Its structures are the slabs across the axis for a TFSF (the
    layered background) and none for a PlaneWave (the sheet launches into the uniform
    background). ``role='driver'`` is a layered TFSF's driver: no structure, and the medium
    the wave arrives from as its background (:func:`_tfsf_driver`). Built at plan time for
    the sub-plan and again at lowering time for its traced params -- a Bloch phase or a
    material value may be a leaf.
    """
    from .. import scene as scene_module
    from ..boundaries import Boundaries
    from .checks import face_wavelength
    grid = plan.grid
    P = src.axis
    pads = column_pads(scene, src)
    faces = {}
    for a, axis in enumerate(AXES):
        if a == P:
            pair = []
            for side, pad in zip(('-', '+'), pads):
                face = scene.boundaries.face(axis + side)
                if pad:
                    face = PML(cells=pad, wavelength=_column_wavelength(scene, src))
                elif isinstance(face, PML) and face.wavelength is None:
                    face = face.replace(wavelength=face_wavelength(face, scene, axis + side))
                pair.append(face)
            faces[axis] = tuple(pair)
            continue
        face = scene.boundaries.face(axis + '-')
        if isinstance(src, PlaneWave) and isinstance(face, Bloch):
            faces[axis] = (Bloch(k=face.k) if face.k is not None
                           else Bloch(phase=face.phase / grid.n[a]))
        elif isinstance(face, Axis1D):
            # The main grid's own face: one cell and the exact i k, in the column as well.
            faces[axis] = face
        else:
            faces[axis] = Periodic()
    size = [grid.dx, grid.dx, grid.dx]
    center = [grid.low[a] + grid.dx / 2 for a in range(3)]
    size[P] = (grid.n[P] + pads[0] + pads[1]) * grid.dx
    center[P] = scene.domain.center[P] + 0.5 * (pads[1] - pads[0]) * grid.dx
    structures, background = (), scene.background
    if role == 'driver':
        background = _tfsf_driver(scene, src, grid)
    elif isinstance(src, TFSF):
        structures = tuple(s for s in scene.structures if slab_axis(s.shape) == P)
    return scene_module.Scene(
        domain=scene_module.Domain(size=tuple(size), center=tuple(center)),
        boundaries=Boundaries(**faces), resolution=scene_module.Resolution(dx=grid.dx),
        run=scene_module.Run(steps=plan.steps, courant=scene.run.courant),
        background=background, structures=structures)


def _column_plan(scene, src, plan, role='column'):
    """The column's sub-plan, at the main plan's dtype.

    A column's own faces are Periodic across unless a PlaneWave's Bloch cell makes them
    Bloch, so on its own a TFSF column in a Bloch (complex) run would be real -- and the
    complex waveform row written into it would fail on NumPy and Torch and lose its imaginary
    part on JAX (an ``amplitude=1j`` source vanished with no error). So a complex run's column
    is complex too.
    """
    from .plan import plan as make_plan
    with warnings.catch_warnings():
        # The column repeats the scene's own warnings (its mesh, its layers); the scene
        # already emitted them once, where the user can see which scene they belong to.
        warnings.simplefilter('ignore')
        sub = make_plan(column_scene(scene, src, plan, role), _column=True)
    if plan.dtype == 'complex128' and sub.dtype != plan.dtype:
        sub = sub.with_static(dtype=plan.dtype)
    pads = column_pads(scene, src)
    if (sub.grid.n[src.axis] != plan.grid.n[src.axis] + sum(pads)
            or abs(sub.dt - plan.dt) > 0.0):
        raise AssertionError("the column does not share the scene's cells and dt")
    return sub


# ---------------------------------------------------------------------------------------------
# Lowering time: the traced leaves
# ---------------------------------------------------------------------------------------------

def lower_sources(scene, plan, rt):
    """``{name: {...}}``: every source term's leaves -- the waveform's, and its own."""
    from .params import lower
    by_name = {s.name: s for s in getattr(scene, 'sources', ()) or ()}
    out = {}
    for term in plan.terms:
        if not getattr(term, 'rows', ()):
            continue
        src = by_name.get(term.name)
        if src is None:
            raise ValueError(f'the plan carries a source term {term.name!r} that the scene does '
                             f'not declare; a plan drives the scene it was built from.')
        complex_amplitude = _is_complex(src.amplitude)
        if complex_amplitude and plan.dtype == 'float64':
            raise ValueError(
                f'{type(src).__name__}({src.name!r}, amplitude={src.amplitude!r}): a complex '
                f'waveform on a real run is refused. The run is real (no Bloch face), so the '
                f'field is real and a complex drive has no meaning in it -- NumPy would drop '
                f'the imaginary part and JAX or Torch would do something else. Say the phase '
                f'with phase=... (radians) and a real amplitude.')
        band = src.band
        # A leaf handed in below float64 -- a jnp scalar built before x64 was lifted is the
        # usual case, and it cost a TFSF gradient seven digits when this was written -- is
        # widened below without recovering anything; say so, as the other leaves' doors do.
        kind = type(src).__name__
        for label, value in (('amplitude', src.amplitude), ('phase', src.phase),
                             ('delay', src.delay), ('band.wavelength', band.wavelength),
                             ('band.df', band.df)) + tuple(
                (f'center[{a}]', c) for a, c in enumerate(getattr(src, 'center', None) or ())):
            check_input_precision(f'{kind}({src.name!r}).{label}', value,
                                  np.complex128 if label == 'amplitude' else np.float64)
        wavelength = rt.array(band.wavelength, 'float64')
        df = rt.array(band.df, 'float64')
        tau = 1.0 / ((2.0 * math.pi) * df)
        entry = {'amplitude': rt.array(src.amplitude,
                                       'complex128' if complex_amplitude else 'float64'),
                 'phase': rt.array(src.phase, 'float64'),
                 'delay': rt.array(src.delay, 'float64'),
                 'f0': 1.0 / wavelength, 'tau': tau, 't0': band.sigmas * tau}
        if isinstance(term, DipoleTerm):
            center = tuple(rt.array(c, 'float64') for c in src.center)
            entry['stamp'] = stamp(term, center, plan.grid, rt)
        else:
            entry['column'] = lower(column_scene(scene, src, plan), term.sub, rt)
            if term.driver is not None:
                entry['driver'] = lower(column_scene(scene, src, plan, 'driver'),
                                        term.driver, rt)
            if any(c.pattern for c in term.corrections):
                entry['pattern'] = _pattern(scene, plan, term.axis, rt)
        out[term.name] = entry
    return out


def _pattern(scene, plan, P, rt):
    """``exp(i k_par . r)`` per cell of a plane across ``P``, relative to cell ``(0, 0)``."""
    grid = plan.grid
    shape = [1, 1, 1]
    total = None
    array_axis = {0: 2, 1: 1, 2: 0}
    for a in range(3):
        if a == P:
            continue
        face = scene.boundaries.face(AXES[a] + '-')
        if not isinstance(face, Bloch):
            continue
        n = grid.n[a]
        per_cell = (rt.array(face.k, 'float64') * grid.dx if face.k is not None
                    else rt.array(face.phase, 'float64') / n)
        index = rt.array(np.arange(n, dtype=np.float64), 'float64')
        factor = rt.xp.exp(1j * (per_cell * index))
        s = list(shape)
        s[array_axis[a]] = n
        factor = rt.xp.reshape(factor, tuple(s))
        total = factor if total is None else total * factor
    return total
