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

from wavebit._common.backend import check_input_precision

from wavebit.fdtd._legacy_boundaries import AXES
from wavebit.fdtd._legacy_boundaries import PML_CELLS
from wavebit.fdtd._legacy_boundaries import Absorbing
from wavebit.fdtd._legacy_boundaries import Axis1D
from wavebit.fdtd._legacy_boundaries import Bloch
from wavebit.fdtd._legacy_boundaries import Periodic
from wavebit.fdtd._legacy_boundaries import PML
from wavebit.materials.time_domain import Dispersive
from wavebit.materials.time_domain import as_material
from wavebit.materials.time_domain import concrete
from wavebit.fdtd._legacy_sources import Dipole
from wavebit.fdtd._legacy_sources import PlaneWave
from wavebit.fdtd._legacy_sources import TFSF
from wavebit.fdtd._legacy_sources import _is_complex
from wavebit.fdtd._kernel.grid import OFFSETS
from wavebit.fdtd._kernel.masks import Slab
from wavebit.fdtd._kernel.sources import ColumnTerm
from wavebit.fdtd._kernel.sources import Correction
from wavebit.fdtd._kernel.sources import DipoleTerm
from wavebit.fdtd._kernel.sources import box_corrections
from wavebit.fdtd._kernel.sources import stamp
from wavebit.fdtd._public_names import public_description, public_name
from wavebit.fdtd._spelling import angle

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
            reason = (f'spans cells along an fdtd.InvariantAxis axis, which holds one cell of a '
                      f'field exp(i k {name}); a source there must be uniform along it (a '
                      f"PlaneWave is), and a CurrentDipole's stamp or a TFSF box is not -- not in "
                      f'this build')
        elif isinstance(face, (Periodic, Bloch)):
            ok = lo >= 0 if side == '-' else hi <= n - 1
            reason = (f'crosses the periodic seam of the box on {face_name!r}; a source '
                      f'that wraps around the cell is not supported -- move it inside')
        elif face_name in absorbing:
            cells = absorbing[face_name]
            ok = lo >= cells if side == '-' else hi <= n - 1 - cells
            kind = public_name(face)
            reason = (f'reaches into the {kind} on {face_name!r} ({cells} cells, so indices '
                      f'{cells}..{n - 1 - cells} are clear of both layers on {name}); a '
                      f'source inside an absorbing layer is absorbed as it is made and its '
                      f'reference series no longer describes what reached the interior')
        else:
            ok = lo >= 1 if side == '-' else hi <= n - 2
            reason = (f'touches the {public_name(face)} wall on {face_name!r}, whose edge '
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
            raise TypeError(public_name(src))
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
    what = f'CurrentDipole({src.name!r})' + ('' if src.range is None else ' over its range')
    for a, (i0, i1) in enumerate(dipole_indices(src, grid)):
        _check_indices(what, a, i0, i1, scene, plan)
        ranges.append((i0, i1 + 1))
    slab = Slab(c=c, z=ranges[2], y=ranges[1], x=ranges[0])
    return DipoleTerm(name=src.name, field=field, slab=slab, row=row, dx=grid.dx,
                      stamp=src.stamp)


def _node_index(grid, axis, position):
    return int(round((position - grid.low[axis]) / grid.dx))


def _structure_cells(structure, grid):
    """``(layer axis or None, support)``: the axis it is a layer across, and the cells it may touch.

    The support and not the filled cells, because both readers -- the slab-crosses-the-TFSF-
    faces refusal and :func:`_check_scatterer` -- are asking whether the structure can change
    the medium anywhere on the box's surface, where the incident field came from a 1-D column
    that knows only the background. A cell the shape merely clips is already wrong there.

    The layer's own span is ``support[layer_axis]``, which is why nothing here reads
    ``shape.axis`` or ``shape.cells(grid)``: neither is on :class:`~wavebit.fdtd._legacy.Shape`, and a
    shape that answered the whole protocol and declared itself a layer used to die on them
    with a bare ``AttributeError``.
    """
    return structure.shape.layer_axis(), structure.shape.support(grid)


def _plane_wave_term(scene, plan, src, row):
    grid, P = plan.grid, src.axis
    transverse = [a for a in range(3) if a != P]
    faces = {a: scene.boundaries.face(AXES[a] + '-') for a in transverse}
    for a, face in faces.items():
        if not isinstance(face, (Periodic, Bloch, Axis1D)):
            raise ValueError(
                f'PlaneWave({src.name!r}) needs a periodic cell across its propagation axis, '
                f'but the {AXES[a]} faces are {public_name(face)}. A plane wave is infinite '
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
            raise ValueError(f'PlaneWave({src.name!r}) at theta = {angle(src.theta)}: the '
                             f'transverse wavevector the Bloch faces must carry is fixed at the '
                             f'band centre in the background, and one of the two is traced here.')
        k0 = index * 2.0 * math.pi / wavelength
        for a in transverse:
            face, expected = faces[a], k0 * k[a]
            if isinstance(face, Periodic):
                if abs(k[a]) >= 1e-12:
                    raise ValueError(
                        f'PlaneWave({src.name!r}, theta={angle(src.theta)}, '
                        f'phi={angle(src.phi)}) has a '
                        f'transverse wavevector {expected:.6g} along {AXES[a]}, but that axis '
                        f'is Periodic, which carries only k = 0 (mod 2 pi/L). Say '
                        f'fd.Boundaries(**fd.Bloch.from_source(pw, background=...), ...).')
                continue
            if isinstance(face, Axis1D):
                given = concrete(face.k)
                if given is not None and abs(given - expected) > 1e-9 * max(abs(expected),
                                                                              k0):
                    raise ValueError(
                        f'PlaneWave({src.name!r}): the InvariantAxis face on {AXES[a]} carries k = '
                        f"{given:.9g} but the wave's transverse wavevector at the band centre is "
                        f'{expected:.9g} (n = {index:g}, wavelength {wavelength:g}, theta '
                        f'{angle(src.theta)}, phi {angle(src.phi)}). Use fd.Bloch.from_source(pw, '
                        f'invariant_axes={AXES[a]!r}, background=...).')
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
                    f'{angle(src.theta)}, phi {angle(src.phi)}). A sheet launched with one k '
                    f'into a cell that imposes another is not a plane wave of either. Use '
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
                f'across {AXES[s_axis]}: the structure is not uniform across the '
                f'sheet, so the sheet would not be launching into the medium its incident '
                f'field is computed in. A PlaneWave takes layers across its propagation axis, '
                f'and patterned structures of finite height -- the cell-snapped '
                f'fdtd.SnappedBox, or a smoothed fd.Sphere, fd.Box, fd.Cylinder or fd.Prism -- '
                f'that stay clear of the sheet.')
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
    drive = plane_wave_drive(scene, plan, src)
    ks = k0i - 1 if src.sign > 0 else k0i + 1
    entry_H = k0i if src.sign > 0 else k0i - 1
    plane = grid.low[P] + k0i * grid.dx
    sub = _column_plan(scene, src, plan)
    return ColumnTerm(name=src.name, kind='PlaneWave', row=row, axis=P, sub=sub, drive=drive,
                      ks=ks + off, corrections=corrections, entry_E=k0i + off,
                      entry_H=entry_H + off, plane_E=plane,
                      plane_H=plane + src.sign * 0.5 * grid.dx)


def plane_wave_drive(scene, plan, src):
    """``((component, weight), ...)``: the stored tangential E a PlaneWave's hard source writes.

    **The discrete plane wave in a Bloch cell.** A field ``A exp(i k . r)`` on the Yee grid is
    stored with each component at its own position (``__init__``'s offsets), so the value in
    cell ``i`` of component ``F_c`` is ``A_c exp(i k . r_c)``, ``r_c`` the component's position:
    relative to the cell's low corner, ``E_u`` carries ``exp(i k_u dx/2)`` (half a cell along
    ``u``), ``E_v`` ``exp(i k_v dx/2)``, ``E_P`` ``exp(i k_P dx/2)``, ``H_u`` ``exp(i (k_v + k_P)
    dx/2)`` and so on -- each component's stagger along its own offsets. With the stagger
    removed (``A`` referred to each component's own position), every one-cell difference of
    the grid is ``i K_a`` with ``K_a = 2 sin(k_a dx/2)/dx`` (``k_a`` itself on an ``Axis1D``
    axis, whose difference is the exact ``i k`` and which has no stagger), and the time
    difference is ``-i Omega``, ``Omega = 2 sin(w dt/2)/dt``, so the update equations become
    the continuum's with ``k -> K``, ``w -> Omega``: ``Z0 H = K x E / (Omega mu)``, ``K . E =
    0`` and ``|K|^2 = Omega^2 eps mu``. The two polarisations are therefore transverse to the
    **discrete** ``K``: ``s_hat = P_hat x K_hat_par`` and ``p_hat = s_hat x K_hat``, whose
    tangential part is ``(K_P / |K|) K_hat_par`` -- along ``K_hat_par`` for every frequency, only
    its length depending on ``Omega``. That is the frame ``_post.frame`` reads every order in.

    **The drive.** The hard source imposes the stored tangential E at ``ks``; the column
    radiates the superposition of the s and p modes whose tangential E matches it. Since the
    two modes' tangential E are ``s_hat_t`` and ``K_hat_par``, orthogonal at every frequency, a
    drive along ``s_hat_t`` launches pure s and one along ``K_hat_par`` pure p **exactly**,
    provided the drive is written in stagger-removed amplitudes. So the stored weights are

        s:  (-K_v, K_u) / |K_par|,                       times exp(i k_c dx/2) per component
        p:  sigma cos(theta) (K_u, K_v) / |K_par|,       times exp(i k_c dx/2) per component

    (``sigma`` the direction's sign; ``cos(theta)`` keeps the continuum's length, which the
    normalisation -- read off the column -- does not depend on). At ``K_par = 0`` the frame is
    the PlaneWave's own ``phi`` and every stagger phase is 1. The H partners need nothing
    here: the column is an exact discrete solution, so its H carries its own stagger and the
    discrete impedance, and the Huygens sheet reads both through the per-cell ``pattern``.

    Before this (stage 2) the weights were the continuum ``e``'s tangential part with no phase:
    the launched wave's tangential E was elliptical by ``~ (k_u - k_v) dx/2``, a first-order p
    part in an s wave off ``phi = 0, 90`` (``|t_p| = 9.1e-3`` in an empty conical cell at 20
    cells per wavelength; ``tests/fdtd/test_fd_axis1d.py``, ``tests/fdtd/test_fd_diffraction.py``).

    ``k_par`` is the wave's own at the band centre, which the transverse faces were checked to
    carry (``_plane_wave_term``); a traced Bloch phase is not read here, so the drive is static.
    """
    grid, P = plan.grid, src.axis
    u, v = (P + 1) % 3, (P + 2) % 3
    k, e, _ = src.vectors()
    kpar = {u: 0.0, v: 0.0}
    if abs(k[u]) >= 1e-12 or abs(k[v]) >= 1e-12:
        index = as_material(scene.background, 'Scene(background=)').index_bound()
        k0 = index * 2.0 * math.pi / src.band.concrete_wavelength
        kpar = {a: (k0 * k[a] if abs(k[a]) >= 1e-12 else 0.0) for a in (u, v)}
    dx = grid.dx
    axis1d = {a: isinstance(scene.boundaries.face(AXES[a] + '-'), Axis1D) for a in (u, v)}
    K = {a: kpar[a] if axis1d[a] else 2.0 * math.sin(kpar[a] * dx / 2.0) / dx for a in (u, v)}
    norm = math.hypot(K[u], K[v])
    if norm <= 1e-12 * (1.0 + norm):
        tangential = {u: e[u], v: e[v]}
    else:
        qu, qv = K[u] / norm, K[v] / norm
        if src.polarization == 's':
            tangential = {u: -qv, v: qu}
        else:
            length = src.sign * math.cos(math.radians(src.theta))
            tangential = {u: length * qu, v: length * qv}
    drive = []
    for c in (u, v):
        weight = tangential[c]
        if abs(weight) <= 1e-15:
            continue
        stagger = 0.0 if axis1d[c] else kpar[c] * dx / 2.0
        if stagger != 0.0:
            if plan.dtype != 'complex128':
                raise AssertionError('a transverse k on a real run')
            weight = complex(weight * math.cos(stagger), weight * math.sin(stagger))
        else:
            weight = float(weight)
        drive.append((c, weight))
    return tuple(sorted(drive))


def _tfsf_term(scene, plan, src, row):
    grid, P, pol = plan.grid, src.axis, src.pol
    h = 3 - P - pol
    n = grid.n
    for side in ('-', '+'):
        face = scene.boundaries.face(AXES[P] + side)
        if not isinstance(face, Absorbing):
            raise ValueError(
                f'TFSF({src.name!r}) propagates along {AXES[P]}, whose {AXES[P] + side!r} face '
                f'is {public_name(face)}. The incident field comes from a 1-D auxiliary '
                f'grid on the scene\'s own cells and faces along that axis, and it must leave '
                f'through them: make both {AXES[P]} faces fdtd.Open(), with fd.PML(...) or '
                f'fdtd.ConductiveAbsorber(...) as Numerics(absorber=...).')
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
                f'structures[{index}] (a slab across {AXES[s_axis]}, cells {c0}..'
                f'{c1 - 1}) crosses the faces of TFSF({src.name!r}) (cells {lo[s_axis] - 1}..'
                f'{hi[s_axis]} along {AXES[s_axis]}). The incident field comes from a '
                f'1-D grid along {AXES[P]} whose background is the slabs across {AXES[P]} '
                f'only, so it is wrong wherever this structure changes the medium on the box '
                f'surface. A slab across {AXES[P]} is part of that background and may cross '
                f'the box; a finite scatterer -- a cell-snapped fdtd.SnappedBox or a smoothed '
                f'body -- is a structure clear of its faces.')
    layered = any(s.shape.layer_axis() == P for s in scene.structures)
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
    """A cell-snapped Region near a TFSF box: strictly inside it, or clear of its surface.

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
        f'structures[{index}] ({public_description(structure.shape)}, cells '
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
    layers = [(i, s) for i, s in enumerate(scene.structures) if s.shape.layer_axis() == P]
    for j in cells:
        owner = None
        for i, s in layers:                         # a later structure overwrites
            # "Owns cell layer j" is the shape's own ``uniform_along``, not a span read off a
            # Slab attribute: on a cell-snapped layer it is the identical inequality
            # (``k0 <= j and k1 >= j + 1``), and on a layer whose face falls inside cell j it
            # is False, which leaves the cell to whatever was under it and makes ``owners``
            # hold two -- the refusal below, which is the right answer for a half-filled cell.
            if s.shape.uniform_along(P, (j, j + 1), grid):
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
    import wavebit.fdtd._scene as scene_module
    from wavebit.fdtd._legacy_boundaries import Boundaries
    from wavebit.fdtd._prepare.checks import face_wavelength
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
        structures = tuple(s for s in scene.structures if s.shape.layer_axis() == P)
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
    from wavebit.fdtd._prepare.plan import plan as make_plan
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
    from wavebit.fdtd._prepare.params import lower
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
            # A dipole's amplitude is held in the engine's units, not as the current_moment
            # its public caller gave, so it is not quoted.
            said = '' if isinstance(src, Dipole) else f', amplitude={src.amplitude!r}'
            raise ValueError(
                f'{public_name(src)}({src.name!r}{said}): a complex '
                f'waveform on a real run is refused. The run is real (no Bloch face), so the '
                f'field is real and a complex drive has no meaning in it -- NumPy would drop '
                f'the imaginary part and JAX or Torch would do something else. Say the phase '
                f'with phase=... (radians) and a real amplitude.')
        band = src.band
        # A leaf handed in below float64 -- a jnp scalar built before x64 was lifted is the
        # usual case, and it cost a TFSF gradient seven digits when this was written -- is
        # widened below without recovering anything; say so, as the other leaves' doors do.
        kind = public_name(src)
        for label, value in (('amplitude', src.amplitude), ('phase', src.phase),
                             ('delay_seconds', src.delay), ('band.wavelength', band.wavelength),
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
