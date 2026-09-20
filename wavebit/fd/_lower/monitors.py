"""Stage 3's lowering: every frequency-domain monitor as one or more ``DFTRegion`` observers.

A monitor is lowered here to static regions -- which Yee points of which component, per
frequency -- and nothing else; what the accumulators mean is ``wavebit/fd/_post.py``'s. The
layouts are computed by the same functions at plan time (to build the regions) and after the
run (to read them back), so the two cannot disagree about which point is which.

**A flux plane** across axis ``P`` at node plane ``k`` (``u, v`` the transverse pair in cyclic
order) reads the tangential E on the node plane and the tangential H on the **two** half
planes ``k -+ 1/2`` straddling it -- the "2-cell face" -- because the Poynting product needs E
and H at one point and on the Yee grid the tangential H is never on a node plane. Across the
plane, ``E_u`` and ``H_v`` share their Yee position (half along ``u``, on the ``v`` node), and
so do ``E_v`` and ``H_u``; nothing is interpolated transversally. ``_post.py`` derives why
the average of the two H planes is the right pairing (the flux then closes exactly).

**The refusals, and where each lives** (all at plan time, before anything runs):

    a flux / diffraction plane on the first or last node of its axis     _plane_index, here
      (one of the two H half planes would not exist)
    a flux box whose faces snap to no cell or touch the domain's edge    _box_nodes, here
    a finite plane extent that snaps to no cell or past the last node    plane_layout, here
    a monitor that reaches into an absorbing layer, or a full-axis       _clear_layout, here
      plane across an axis whose faces absorb
    a Field extent holding no point of a component                       field_points, here
    Diffraction with no PlaneWave or more than one, across another       diffraction_layout, here
      axis than the PlaneWave's, or on non-periodic transverse faces
    Diffraction orders the plane's cells cannot resolve, or a default    diffraction_window, here
      window over a traced medium or Bloch phase
    Diffraction (or its PlaneWave's sheet) in a medium that is not one   _post.medium_at, called
      plain isotropic fd.Material across the plane                         here
    Diffraction orders along an Axis1D axis (it has the single order 0)  diffraction_window, here
    FarField in a periodic/Bloch/Axis1D cell (names Diffraction), with a farfield_checks, here
      non-absorbing face, in a background that is not a plain isotropic
      Material, with a structure not strictly inside its box, or cut by
      a TFSF surface
    a Flux / Diffraction plane or a Flux / FarField box that cuts a      _clear_of_sources, here
      Dipole's stamp, lies on (or a box around) a PlaneWave's sheet, or
      (Flux, Diffraction) cuts or lies on a TFSF surface

**Why a flux surface must keep off every source.** The 2-cell face pairs E on the node plane
with the mean of the two H half planes either side, so it reads one cell of the grid on each
side of the surface. A source whose support reaches that cell is counted in part: a Dipole's
stamp (two cells of B-spline or tent either side of its centre) cut by a box face leaves the
box short of the dipole's power (-0.916 of it for a 0.05 box inside the stamp, -1.6e-3 for a
0.3 box whose face cuts the stamp's tail, at 13 cells per wavelength); a plane on a PlaneWave
sheet pairs the total field below it with the scattered field above (a Diffraction plane there
read R = 0.08 / 0.15 / 0.08 for a stack whose R is 0.001 / 0.037 / 0.093, and a Flux plane
-0.48 of the incident flux where one cell off reads -0.999 or 0.001); a surface on a TFSF face
mixes the two fields the same way. So a closed box keeps the stamp or the TFSF box strictly
inside with one cell to spare on every side, or clear of it by one cell on some axis; a plane
stays one cell clear of both; nothing touches a sheet's node plane, and a box does not enclose
part of a sheet (the flux of a sheet's slice is the power it injects, which
``res.sources[name].intensity()`` already is).

**A far-field box** lowers to the same six face regions a ``Flux`` box does (``box_faces``), so
the two read exactly the same points and ``res[name].flux`` of a FarField is the Flux of its
box; ``_post.far_field`` turns them into the radiation vectors.
"""
import dataclasses
import math

import numpy as np

from ..boundaries import AXES, Axis1D, Bloch, Periodic
from ..geometry import Box, Plane
from ..monitors import Diffraction, FarField, Field, Flux
from .._kernel.grid import OFFSETS
from .._kernel.masks import Slab
from .._kernel.observe import DFTRegion

__all__ = ['monitor_regions', 'PlaneLayout', 'plane_layout', 'box_faces', 'field_points',
           'diffraction_layout', 'diffraction_window', 'farfield_checks', 'FACE_SEP']

#: A box monitor's faces are regions named ``monitor + FACE_SEP + face``.
FACE_SEP = '::'


def _node(grid, a, position):
    return int(round((position - grid.low[a]) / grid.dx))


def _slab(c, ranges):
    """``Slab`` of component ``c`` over ``ranges = {axis: (lo, hi)}`` (x, y, z indices)."""
    return Slab(c=c, z=tuple(ranges[2]), y=tuple(ranges[1]), x=tuple(ranges[0]))


@dataclasses.dataclass(frozen=True)
class PlaneLayout:
    """Where a flux or diffraction plane reads: node plane ``k`` across axis ``P``.

    ``half[a]`` is the half-open range of half-cell indices along transverse axis ``a`` (the
    positions ``i + 1/2``), ``nodes[a]`` the half-open range of node indices, and
    ``weights[a]`` the quadrature weight of each node (``None`` for all ones). ``full[a]``
    says the plane runs the whole axis.
    """

    P: int
    k: int
    half: tuple
    nodes: tuple
    weights: tuple
    full: tuple

    @property
    def u(self):
        return (self.P + 1) % 3

    @property
    def v(self):
        return (self.P + 2) % 3

    def parts(self):
        """``((label, field, Slab), ...)`` for E_u, E_v, H_u, H_v -- the four a flux needs."""
        P, u, v = self.P, self.u, self.v
        out = []
        for label, field, c in (('Eu', 'E', u), ('Ev', 'E', v), ('Hu', 'H', u), ('Hv', 'H', v)):
            ranges = {P: (self.k, self.k + 1) if field == 'E' else (self.k - 1, self.k + 1)}
            # E_u / H_v sit half along u and on the v nodes; E_v / H_u the other way round.
            along_u = (label in ('Eu', 'Hv'))
            ranges[u] = self.half[u] if along_u else self.nodes[u]
            ranges[v] = self.nodes[v] if along_u else self.half[v]
            out.append((label, field, _slab(c, ranges)))
        return tuple(out)


def _plane_index(where, grid, P, position):
    k = _node(grid, P, position)
    n = grid.n[P]
    if not 1 <= k <= n - 1:
        raise ValueError(
            f'{where} at {AXES[P]} = {position:g} snaps to node {k} of {n}: a flux or '
            f'diffraction plane reads the tangential H on the two half planes either side of '
            f'its node plane, so it needs one cell on each side (nodes 1..{n - 1}). Move it '
            f'inside.')
    return k


def _trapezoid(count):
    """Weights ``1/2, 1, ..., 1, 1/2`` over ``count`` nodes spanning an interval."""
    w = np.ones(count)
    if count >= 2:
        w[0] = w[-1] = 0.5
    return tuple(float(x) for x in w)


def plane_layout(where, grid, P, position, extents=(None, None, None), k=None):
    """The :class:`PlaneLayout` of a plane across ``P``; ``extents[a]`` a ``(lo, hi)`` or None.

    A full axis reads every node and half cell with weight one -- on a periodic axis that is
    the exact sum over the period. A finite extent snaps its ends to node planes ``i0 .. i1``
    and integrates by the trapezoid over the nodes (weight one half at the two end nodes) and
    by the midpoint over the half cells ``i0 + 1/2 .. i1 - 1/2`` between them: each point
    carries the length of the dual segment it stands for, so the face is exactly
    ``(i1 - i0) dx`` long in both readings. ``_post.py`` measures the closure this gives.
    """
    if k is None:
        k = _plane_index(where, grid, P, position)
    half, nodes, weights, full = [None] * 3, [None] * 3, [None] * 3, [False] * 3
    for a in range(3):
        if a == P:
            continue
        ext = extents[a]
        n = grid.n[a]
        if ext is None:
            half[a], nodes[a], full[a] = (0, n), (0, n), True
            continue
        i0, i1 = _node(grid, a, ext[0]), _node(grid, a, ext[1])
        if i1 <= i0:
            raise ValueError(f'{where}: the extent {ext} along {AXES[a]} snaps to no cell on '
                             f'dx = {grid.dx:g}.')
        if i0 < 0 or i1 > n - 1:
            raise ValueError(
                f'{where}: the extent {ext} along {AXES[a]} snaps to nodes {i0}..{i1}, past the '
                f'last stored node {n - 1} of that axis (node {n} is the high face itself). '
                f'Keep a finite plane inside the box.')
        half[a], nodes[a] = (i0, i1), (i0, i1 + 1)
        weights[a] = _trapezoid(i1 - i0 + 1)
    return PlaneLayout(P=P, k=k, half=tuple(half), nodes=tuple(nodes), weights=tuple(weights),
                       full=tuple(full))


def _box_nodes(where, grid, box):
    lo = tuple(_node(grid, a, box.low[a]) for a in range(3))
    hi = tuple(_node(grid, a, box.high[a]) for a in range(3))
    for a in range(3):
        if hi[a] <= lo[a]:
            raise ValueError(f'{where}: the box snaps to no cell along {AXES[a]} (size '
                             f'{box.size[a]:g} at dx = {grid.dx:g}).')
        if lo[a] < 1 or hi[a] > grid.n[a] - 1:
            raise ValueError(
                f'{where}: the box snaps to nodes {lo[a]}..{hi[a]} along {AXES[a]} of '
                f'{grid.n[a]}; each face reads the H half planes either side of it, so the box '
                f'needs one cell of clearance from the domain on every side (nodes 1..'
                f'{grid.n[a] - 1}).')
    return lo, hi


def box_faces(where, grid, box):
    """``((face name, outward sign, PlaneLayout), ...)`` of a closed flux box, six faces."""
    lo, hi = _box_nodes(where, grid, box)
    out = []
    for P in range(3):
        extents = [None] * 3
        for a in range(3):
            if a != P:
                extents[a] = (grid.low[a] + lo[a] * grid.dx, grid.low[a] + hi[a] * grid.dx)
        for side, k, sign in (('-', lo[P], -1.0), ('+', hi[P], 1.0)):
            layout = plane_layout(where, grid, P, None, extents, k=k)
            out.append((AXES[P] + side, sign, layout))
    return tuple(out)


def _plane_extents(plane):
    return tuple(None if a == plane.axis else plane.extent(a) for a in range(3))


# ---------------------------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------------------------

def field_points(monitor, grid):
    """``{component: (Slab, (x, y, z) coordinate arrays)}`` of a Field monitor, per component.

    Every component on its own Yee points: on a plane, the plane of that component's points
    nearest the position; across it (or in a box), every point of the component inside the
    extent, ends included. Nothing is interpolated.
    """
    out = {}
    for comp in monitor.components:
        field, c = comp[0], 'xyz'.index(comp[1])
        ranges, coords = {}, {}
        for a in range(3):
            o = 0.5 * OFFSETS[field][c][a]
            n = grid.n[a]
            if monitor.plane is not None and a == monitor.plane.axis:
                i = int(round((monitor.plane.position - grid.low[a]) / grid.dx - o))
                i = min(max(i, 0), n - 1)
                lo, hi = i, i + 1
            else:
                if monitor.box is not None:
                    ext = (monitor.box.low[a], monitor.box.high[a])
                else:
                    ext = monitor.plane.extent(a)
                if ext is None:
                    lo, hi = 0, n
                else:
                    lo = int(math.ceil((ext[0] - grid.low[a]) / grid.dx - o - 1e-9))
                    hi = int(math.floor((ext[1] - grid.low[a]) / grid.dx - o + 1e-9)) + 1
                    lo, hi = max(lo, 0), min(hi, n)
                    if hi <= lo:
                        raise ValueError(
                            f'Field({monitor.name!r}): no {comp} point lies inside the extent '
                            f'{ext} along {AXES[a]} on dx = {grid.dx:g}.')
            ranges[a] = (lo, hi)
            coords[a] = grid.low[a] + (np.arange(lo, hi) + o) * grid.dx
        out[comp] = (_slab(c, ranges), (coords[0], coords[1], coords[2]))
    return out


# ---------------------------------------------------------------------------------------------
# Diffraction
# ---------------------------------------------------------------------------------------------

def plane_waves(scene):
    from ..sources import PlaneWave
    return [s for s in getattr(scene, 'sources', ()) or () if isinstance(s, PlaneWave)]


def diffraction_layout(monitor, scene, grid):
    """The :class:`PlaneLayout` of a Diffraction monitor, after its scene-level refusals."""
    where = f'Diffraction({monitor.name!r})'
    waves = plane_waves(scene)
    if len(waves) != 1:
        raise ValueError(
            f'{where}: the orders are normalised by the incident amplitude of the scene\'s '
            f'fd.PlaneWave, read from its own column, and this scene has {len(waves)} '
            f'PlaneWave sources. A Diffraction monitor needs exactly one.')
    pw = waves[0]
    P = monitor.axis
    if P != pw.axis:
        raise ValueError(
            f'{where} is a plane across {AXES[P]}, but PlaneWave({pw.name!r}) travels along '
            f'{AXES[pw.axis]}. The orders live on the plane across the propagation axis: say '
            f'Diffraction({monitor.name!r}, {AXES[pw.axis]}=...).')
    for a in range(3):
        if a == P:
            continue
        face = scene.boundaries.face(AXES[a] + '-')
        if not isinstance(face, (Periodic, Bloch, Axis1D)):
            raise ValueError(
                f'{where}: the {AXES[a]} faces are {type(face).__name__}; diffraction orders '
                f'exist in a periodic, Bloch or Axis1D cell only (on an open scene the far '
                f'field is fd.FarField, the transform of a closed box).')
    layout = plane_layout(where, grid, P, monitor.position)
    return layout, pw


def diffraction_window(monitor, scene, grid, eps, mu):
    """``(M_u, M_v)``: the monitor's ``orders``, or every order propagating in the band.

    Counted from the concrete medium at the plane and the concrete Bloch ``k``; either being
    traced needs ``orders=`` given. A window the plane's cells cannot resolve -- ``2M + 1`` orders
    on fewer cells -- would alias one order onto another, and is refused.
    """
    from .. import _post
    from ..boundaries import Bloch
    from ..materials import concrete
    where = f'Diffraction({monitor.name!r})'
    P = monitor.axis
    u, v = (P + 1) % 3, (P + 2) % 3
    if monitor.orders is not None:
        Mu, Mv = monitor.orders
        for M, a in ((Mu, u), (Mv, v)):
            if M and isinstance(scene.boundaries.face(AXES[a] + '-'), Axis1D):
                spelled = f'({Mu}, 0)' if a == v else f'(0, {Mv})'
                raise ValueError(
                    f'{where}(orders={monitor.orders}): {AXES[a]} is an fd.Axis1D axis, one '
                    f'cell of a field exp(i k {AXES[a]}), which has the single order 0; say '
                    f'orders={spelled}.')
    else:
        kpar = []
        for a in (u, v):
            face = scene.boundaries.face(AXES[a] + '-')
            if not isinstance(face, Bloch):
                kpar.append(0.0)
            elif face.k is not None:
                kpar.append(concrete(face.k))
            else:
                c = concrete(face.phase)
                kpar.append(None if c is None else c / grid.size[a])
        n_c, mu_c = concrete(eps), concrete(mu)
        if n_c is None or mu_c is None or any(k is None for k in kpar):
            raise ValueError(f'{where}: the medium at the plane or the Bloch phase is traced, so '
                             f'the propagating orders cannot be counted when the plan is built; '
                             f'give orders=(M_u, M_v).')
        Mu, Mv = _post.default_orders(monitor, scene, grid, math.sqrt(n_c * mu_c), kpar)
    for M, a in ((Mu, u), (Mv, v)):
        if 2 * M + 1 > grid.n[a]:
            raise ValueError(
                f'{where}: {2 * M + 1} orders along {AXES[a]} need at least that many cells '
                f'across the period, which has {grid.n[a]}; an order the plane cannot resolve '
                f'would alias onto another. Refine dx, or ask for orders=... no wider than '
                f'{(grid.n[a] - 1) // 2}.')
    return Mu, Mv


# ---------------------------------------------------------------------------------------------
# FarField
# ---------------------------------------------------------------------------------------------

def farfield_checks(monitor, scene, grid):
    """The scene-level refusals of a FarField monitor; returns the background ``(eps, mu)``.

    The transform integrates the box's surface currents against the free-space Green's function
    of one uniform medium, so everything outside the box must be that medium, reaching to
    infinity: every face absorbing (a periodic cell's far field is its orders -- Diffraction; a
    PEC, PMC or mirror face would need image currents, which are not in this build), no
    structure outside the box or touching its surface, and no TFSF surface crossing it (a box
    around a TFSF box sees the scattered field alone; one cutting it sees a discontinuity).
    """
    from ..boundaries import Absorbing, Axis1D, Bloch, Periodic
    from ..sources import TFSF
    from .. import _post
    where = f'FarField({monitor.name!r})'
    for a, axis in enumerate(AXES):
        for side in '-+':
            face = scene.boundaries.face(axis + side)
            if isinstance(face, (Periodic, Bloch, Axis1D)):
                raise ValueError(
                    f'{where}: the {axis} faces are {type(face).__name__} -- a periodic cell, '
                    f'whose far field is a set of plane waves, the diffraction orders. Use '
                    f'fd.Diffraction(...) there; a FarField is the transform of a closed box in '
                    f'an open scene.')
            if not isinstance(face, Absorbing):
                raise ValueError(
                    f'{where}: the face {axis + side!r} is {type(face).__name__}. The transform '
                    f'is the free-space one, so every face must absorb (fd.PML or fd.Absorber); '
                    f'a wall or a mirror would need image currents, which are not in this build.')
    lo, hi = _box_nodes(where, grid, monitor.box)
    for index, s in enumerate(scene.structures):
        box = s.shape.support(grid)            # everything it may touch must be inside
        if not all(lo[a] + 1 <= box[a][0] and box[a][1] <= hi[a] - 1 for a in range(3)):
            raise ValueError(
                f'{where}: structures[{index}] (cells {tuple(tuple(r) for r in box)}) is not '
                f'strictly inside the box (nodes {tuple(lo)}..{tuple(hi)}, one cell to spare). '
                f'The transform assumes one uniform medium outside the box out to infinity -- a '
                f'substrate or a neighbour outside it would need its own Green\'s function, '
                f'which is not in this build. Enlarge the box around every structure.')
    for src in scene.sources:
        if not isinstance(src, TFSF):
            continue
        tlo = [_node(grid, a, src.box.low[a]) for a in range(3)]
        thi = [_node(grid, a, src.box.high[a]) for a in range(3)]
        inside = all(lo[a] <= tlo[a] - 2 and thi[a] + 2 <= hi[a] for a in range(3))
        outside = any(hi[a] <= tlo[a] - 2 or lo[a] >= thi[a] + 2 for a in range(3))
        if not (inside or outside):
            raise ValueError(
                f'{where}: the box (nodes {tuple(lo)}..{tuple(hi)}) cuts the surface of '
                f'TFSF({src.name!r}) (nodes {tuple(tlo)}..{tuple(thi)}), so part of it reads the '
                f'total field and part the scattered one. Enclose the TFSF box with at least two '
                f'cells to spare (the far field of the scattered wave), or keep clear of it.')
    background = _post.medium_at(scene, grid, 0, (lo[0] - 1, lo[0]), where)
    return background


# ---------------------------------------------------------------------------------------------
# A flux surface keeps off every source
# ---------------------------------------------------------------------------------------------

def _plane_nodes(grid, layout):
    """``(lo, hi)`` node ranges of a plane: ``k`` across it, its extent (or the axis) along."""
    lo, hi = [0, 0, 0], [0, 0, 0]
    for a in range(3):
        if a == layout.P:
            lo[a] = hi[a] = layout.k
        elif layout.full[a]:
            lo[a], hi[a] = 0, grid.n[a]
        else:
            lo[a], hi[a] = layout.nodes[a][0], layout.nodes[a][1] - 1
    return tuple(lo), tuple(hi)


def _nodes_text(lo, hi):
    return ', '.join(f'{AXES[a]} {lo[a]:g}..{hi[a]:g}' for a in range(3))


def _clear_of_sources(where, scene, grid, lo, hi, closed, tfsf=True):
    """Refuse a flux surface (node ranges ``lo..hi`` per axis) that reads a source's cells.

    ``closed`` is a box (six faces; it may enclose a source) against a plane (one face; it must
    stay clear). Positions are in cells from the grid's low corner: a node is an integer, a
    half cell ``i + 1/2``. The module docstring gives the measurements behind each refusal.
    """
    from ..sources import Dipole, PlaneWave, TFSF
    from .sources import dipole_indices
    shape = 'box' if closed else 'plane'
    for src in scene.sources:
        if isinstance(src, Dipole):
            offsets = [0.5 * OFFSETS[src.field][src.index][a] for a in range(3)]
            first = [i0 + o for (i0, _), o in zip(dipole_indices(src, grid), offsets)]
            last = [i1 + o for (_, i1), o in zip(dipole_indices(src, grid), offsets)]
            inside = closed and all(lo[a] + 1 <= first[a] and last[a] <= hi[a] - 1
                                    for a in range(3))
            outside = any(last[a] <= lo[a] - 1 or first[a] >= hi[a] + 1 for a in range(3))
            if not (inside or outside):
                stamp = ', '.join(f'{AXES[a]} {first[a]:g}..{last[a]:g}' for a in range(3))
                keep = ('enclose it with one cell to spare on every side (a larger box) or '
                        'keep clear of it by one cell' if closed else
                        'keep the plane one cell clear of it')
                raise ValueError(
                    f'{where}: the {shape} (nodes {_nodes_text(lo, hi)}) cuts the stamp of '
                    f'Dipole({src.name!r}), which writes its {src.field}{"xyz"[src.index]} at '
                    f'cells {stamp} (two cells either side of its centre). The 2-cell face '
                    f'reads one cell each side of the surface, so it would count part of the '
                    f'source and miss part of its power; {keep}.')
        elif isinstance(src, PlaneWave):
            P = src.axis
            k = _node(grid, P, src.position)
            if lo[P] <= k <= hi[P]:
                position = grid.low[P] + k * grid.dx
                low_side = 'total' if src.sign < 0 else 'scattered'
                high_side = 'scattered' if src.sign < 0 else 'total'
                what = ('lies on' if lo[P] == hi[P] or k in (lo[P], hi[P]) else 'encloses part of')
                raise ValueError(
                    f'{where}: the {shape} (nodes {_nodes_text(lo, hi)}) {what} the sheet of '
                    f'PlaneWave({src.name!r}) at {AXES[P]} = {position:g} (node {k}), where the '
                    f'field jumps from the total field on one side to the scattered field on '
                    f'the other; a surface there pairs the two. Move it at least one cell off '
                    f'the sheet: at {AXES[P]} = {position - grid.dx:g} or below it reads the '
                    f'{low_side} field, at {AXES[P]} = {position + grid.dx:g} or above the '
                    f'{high_side} field. The power the sheet injects is '
                    f'res.sources[{src.name!r}].intensity().')
        elif isinstance(src, TFSF) and tfsf:
            tlo = [_node(grid, a, src.box.low[a]) for a in range(3)]
            thi = [_node(grid, a, src.box.high[a]) for a in range(3)]
            inside = all(tlo[a] + 1 <= lo[a] and hi[a] <= thi[a] - 1 for a in range(3))
            around = closed and all(lo[a] <= tlo[a] - 1 and thi[a] + 1 <= hi[a]
                                    for a in range(3))
            outside = any(hi[a] <= tlo[a] - 1 or lo[a] >= thi[a] + 1 for a in range(3))
            if not (inside or around or outside):
                raise ValueError(
                    f'{where}: the {shape} (nodes {_nodes_text(lo, hi)}) cuts or lies on the '
                    f'surface of TFSF({src.name!r}) (nodes {_nodes_text(tlo, thi)}), so part of '
                    f'it reads the total field and part the scattered one. Keep it strictly '
                    f'inside the TFSF box with one cell to spare (the total field), '
                    + ('around it with one cell to spare (the scattered field), '
                       if closed else '')
                    + 'or one cell clear of it.')


# ---------------------------------------------------------------------------------------------
# The regions
# ---------------------------------------------------------------------------------------------

def _clear_of_absorbers(where, scene, grid, P, k_lo, k_hi):
    """Refuse a monitor whose indices ``k_lo..k_hi`` along ``P`` reach an absorbing layer."""
    from ..boundaries import Absorbing
    n = grid.n[P]
    for side in ('-', '+'):
        face = scene.boundaries.face(AXES[P] + side)
        if not isinstance(face, Absorbing):
            continue
        cells = face.cells
        ok = k_lo >= cells if side == '-' else k_hi <= n - 1 - cells
        if not ok:
            raise ValueError(
                f'{where} reads indices {k_lo}..{k_hi} along {AXES[P]} and reaches the '
                f'{type(face).__name__} on {AXES[P] + side!r} ({cells} cells). Inside an '
                f'absorbing layer the stretched field is not the physical one and its flux '
                f'means nothing; keep the monitor at indices {cells}..{n - 1 - cells}.')


def monitor_regions(scene, grid, dt):
    """The ``DFTRegion`` observers of every frequency-domain monitor in ``scene``, in order."""
    out = []
    for m in scene.monitors:
        if isinstance(m, FarField):
            freqs = tuple(1.0 / w for w in m.wavelengths)
            where = f'FarField({m.name!r})'
            farfield_checks(m, scene, grid)
            lo, hi = _box_nodes(where, grid, m.box)
            _clear_of_sources(where, scene, grid, lo, hi, closed=True, tfsf=False)
            for face, _, layout in box_faces(where, grid, m.box):
                _clear_layout(where, scene, grid, layout)
                out.append(DFTRegion(name=m.name + FACE_SEP + face, monitor=m.name,
                                     freqs=freqs, parts=layout.parts(), dt=dt))
        elif isinstance(m, Flux):
            freqs = tuple(1.0 / w for w in m.wavelengths)
            where = f'Flux({m.name!r})'
            if m.box is not None:
                lo, hi = _box_nodes(where, grid, m.box)
                _clear_of_sources(where, scene, grid, lo, hi, closed=True)
                for face, _, layout in box_faces(where, grid, m.box):
                    _clear_layout(where, scene, grid, layout)
                    out.append(DFTRegion(name=m.name + FACE_SEP + face, monitor=m.name,
                                         freqs=freqs, parts=layout.parts(), dt=dt))
            else:
                layout = plane_layout(where, grid, m.plane.axis, m.plane.position,
                                      _plane_extents(m.plane))
                _clear_layout(where, scene, grid, layout)
                _clear_of_sources(where, scene, grid, *_plane_nodes(grid, layout), closed=False)
                out.append(DFTRegion(name=m.name, monitor=m.name, freqs=freqs,
                                     parts=layout.parts(), dt=dt))
        elif isinstance(m, Field):
            freqs = tuple(1.0 / w for w in m.wavelengths)
            points = field_points(m, grid)
            parts = tuple((comp, comp[0], slab) for comp, (slab, _) in points.items())
            out.append(DFTRegion(name=m.name, monitor=m.name, freqs=freqs, parts=parts, dt=dt))
        elif isinstance(m, Diffraction):
            freqs = tuple(1.0 / w for w in m.wavelengths)
            layout, pw = diffraction_layout(m, scene, grid)
            where = f'Diffraction({m.name!r})'
            _clear_layout(where, scene, grid, layout)
            _clear_of_sources(where, scene, grid, *_plane_nodes(grid, layout), closed=False)
            # The plane-wave algebra needs one plain medium at the plane and at the sheet;
            # asked here so a scene that cannot be answered is refused before it runs.
            from .. import _post
            eps, mu = _post.medium_at(scene, grid, layout.P, (layout.k - 1, layout.k), where)
            k_sheet = _node(grid, pw.axis, pw.position)
            _post.medium_at(scene, grid, pw.axis, (k_sheet - 1, k_sheet),
                            f'PlaneWave({pw.name!r}) normalisation for {where}')
            diffraction_window(m, scene, grid, eps, mu)
            out.append(DFTRegion(name=m.name, monitor=m.name, freqs=freqs,
                                 parts=layout.parts(), dt=dt))
    return tuple(out)


def _clear_layout(where, scene, grid, layout):
    _clear_of_absorbers(where, scene, grid, layout.P, layout.k - 1, layout.k)
    from ..boundaries import Absorbing
    for a in (layout.u, layout.v):
        if layout.full[a]:
            faces = [scene.boundaries.face(AXES[a] + s) for s in '-+']
            if any(isinstance(f, Absorbing) for f in faces):
                raise ValueError(
                    f'{where} runs the whole {AXES[a]} axis, whose faces absorb: it would add up '
                    f'the stretched field inside the layers, which is not a flux of anything. '
                    f'Give the plane an extent clear of them, fd.Plane(..., {AXES[a]}=(lo, hi)), '
                    f'or use a closed fd.Box.')
            continue                              # a periodic (or walled) axis, read whole
        lo, hi = layout.nodes[a]
        _clear_of_absorbers(where, scene, grid, a, lo, hi - 1)


def is_box(monitor):
    return isinstance(getattr(monitor, 'box', None), Box)


def is_plane(monitor):
    return isinstance(getattr(monitor, 'plane', None), Plane)
