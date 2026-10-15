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
                                                  vacuum too -- but with sources the refusal is
                                                  check_wavelength_resolution's, which runs
                                                  before the rules that read the band
    a structure within half a wavelength of      check_resolution, here (warns); not a
      an absorbing face                            layer running into a PML matched to its
                                                   own material
    a wavelength a rule will read at < 8 cells   check_wavelength_resolution, here (stage 3):
      per wavelength in a material                 a Diffraction monitor's, and the band
    complex eps                                 materials._value
    Slab naming two axes, any other shape       geometry.Slab / Structure
    pole CFL violated, concretely                check_stability, here
    two structures overlap, one dispersive      check_overlaps, here
    (stage 4 C2) a conductive or dispersive     check_smoothed, here: sigma and the poles
      material where a smoothed shape reaches     are not smoothed in C2
    (stage 4 C2) a smoothed shape at a          check_smoothed, here: the periodic image
      Periodic, Bloch or Axis1D face              ring is not written; a body uniform
                                                  along the axis is carried (C4)
    (stage 4 C2) a smoothed shape the grid      check_smoothed, here: two arms, split by
      cannot carry whole at a face of the box     CAUSE -- a body that LEAVES the domain,
                                                  and one whose last half cell the plan
                                                  TRIMS -- and two messages
    (stage 4 C2) an anisotropic eps or mu on    check_smoothed, here: the mixing rule is
      a smoothed shape                            derived for two isotropic media
    (stage 4 C2) two smoothed shapes whose      check_smoothed, here: the two-phase rule
      surfaces pass through one voxel             takes one interface, not two
    (stage 4 C4) a smoothed surface in a        check_smoothed, here: the snapped cells
      voxel a cell-snapped face cuts              are all or nothing where the geometry
                                                  splits -- three media, and a rule of two
    (stage 4 C5) a smoothed body under 4 cells  check_cells_across, here (study S2b); 4 to 8
      across its smallest extent, not a layer     warns (CoarseMeshWarning)
    (stage 4 C5) a cell-snapped shape across    check_seams, here: clipped, not wrapped --
      one periodic seam, not spanning it          its periodic image is missing
    (stage 4 C5) a structure past a PEC or      _clipping_notes, here (warns, with the share
      PMC face                                    lost); absorbing faces and spans do not
    (stage 4 C5) a structure crossing a         check_symmetry, here: the run models the
      Symmetry plane that is not its own          mirror of the domain part; a smoothed body
      mirror image, or a smoothed body within     flush against a plane whose parity leaves
      half a cell of one                          the plane's components free reads half
    (stage 3) a Bloch phase on a one-cell axis    check_one_cell_bloch, here (names Axis1D;
      in a scene with a source                      a source-free kernel study keeps it)
    (stage 3) an Axis1D k past k_max              boundaries.Axis1D (concrete); the params'
                                                  guard (traced), which makes the run nan
"""
import collections
import math
import warnings

import numpy as np

from wavebit.fdtd.diagnostics import AbsorberProximityWarning
from wavebit.fdtd.diagnostics import CoarseMeshWarning
from wavebit.fdtd.diagnostics import PMLStabilityWarning
from wavebit.materials.time_domain import Conductor
from wavebit.materials.time_domain import Dispersive
from wavebit.materials.time_domain import Material
from wavebit.materials.time_domain import concrete
from wavebit.materials.time_domain import eps_at

__all__ = ['face_wavelength', 'check_faces', 'check_overlaps', 'check_absorbing_layers',
           'check_stability', 'notes', 'cfl_bound', 'check_one_cell_bloch', 'cfl_axis_sum',
           'check_wavelength_resolution']


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
    from wavebit.fdtd._prepare.sources import source_wavelength
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
    from wavebit.fdtd._legacy_boundaries import Bloch
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
    from wavebit.fdtd._legacy_boundaries import Axis1D
    total = 0.0
    for axis in 'xyz':
        face = scene.boundaries.face(axis + '-')
        total += (face.k_max ** 2 / 4.0) if isinstance(face, Axis1D) else 1.0 / dx ** 2
    return total


def check_overlaps(scene, grid):
    """Two structures may overlap, unless a :class:`~wavebit.fdtd._legacy.Dispersive` is party to it.

    "Structures are applied in the order they are given, so a later one overwrites an earlier
    one where they overlap" is the documented rule, and it is enforced in exactly one place:
    ``coefficients.painters``, the one painter per structure that builds ``eps``, ``mu`` and
    ``sigma``. The **polarisation is not routed through those painters**. ``plan._pole_terms``
    gives every dispersive structure a term over its own snapped cells, so an overwritten
    dispersive medium keeps driving ``cH`` from cells it no longer owns, and two overlapping
    dispersive structures add their polarisations twice. Measured on a 64-cell line: layering ``Material(eps=2.25)`` over a
    ``Dispersive(eps_inf=2.25, ...)`` on the same cells -- a scene the rule says must equal the
    glass alone -- gives a field 64% of the peak away from the glass scene and bit-identical to
    the *metal* scene, and ``(metal, metal)`` differs from ``(metal,)`` by 72% of its peak.

    Rather than route the term (a still-owned set is a box minus later boxes, which is not a
    box), stage 1 refuses the case at the door.

    **Does the argument still hold now that a painter may mix rather than replace?** For the
    overlaps this function decides, yes, and the reason is that the two discretisations do not
    meet here. A *cell-snapped* painter still replaces the cell outright, so a non-dispersive
    overlap of ``Slab``s and ``Region``s goes through ``coefficients._blend`` exactly as it
    went through the owner index, bit for bit, and the order rule holds for it exactly. A
    *smoothed* painter mixes, and "later overwrites earlier" is then only true where the
    arriving shape fills the voxel whole -- so the case where it is not, **two smoothed
    surfaces in one voxel**, is refused by name in :func:`check_smoothed` (73% of ``eps``
    measured there, on two half spaces of the same material), and so is a smoothed surface
    in a voxel a cell-snapped face also cuts (door 6 there: the painter reads the snapped
    shape as all there or all absent in a voxel the geometry splits). A smoothed shape wholly
    over a snapped one away from its faces composes exactly, measured 2.0e-16 against the
    hand two-phase mix by
    ``tests/fdtd/test_fd_occupancy.py::test_a_smoothed_shape_clear_of_a_snapped_face_composes_exactly``.
    What is left for this function is unchanged, and it is **cell-snapped pairs only** since
    C5: a pair with a smoothed shape in it is door 1's of :func:`check_smoothed`, which refuses a
    dispersive medium wherever a smoothed shape reaches and says why, where this message's
    advice -- spell it as disjoint slabs -- is no use for a sphere.

    Note that two slabs on *different* axes always intersect -- each is full-width on the
    other's axis -- so this refuses a dispersive slab crossed by any other structure, and names
    the two spellings that work.
    """
    structures = tuple(getattr(scene, 'structures', ()) or ())
    boxes = [s.shape.support(grid) for s in structures]
    smoothed = [s.shape.body() is not None for s in structures]
    for i in range(len(structures)):
        for j in range(i + 1, len(structures)):
            if not _intersects(boxes[i], boxes[j]):
                continue
            if smoothed[i] or smoothed[j]:
                # check_smoothed's door 1 refuses a dispersive material wherever a smoothed
                # shape reaches, for the reason that is true there; this door's support-box
                # test and its advice to spell slabs refused a Drude dimer on the diagonal
                # with a gap of half a diameter (C5 audit). Lifting door 1 is C8's.
                continue
            kinds = [isinstance(structures[k].material, Dispersive) for k in (i, j)]
            if not any(kinds):
                continue
            which = f'structures[{i}]' if kinds[0] else f'structures[{j}]'
            raise ValueError(
                f'structures[{i}] ({structures[i].shape.describe()}) and '
                f'structures[{j}] ({structures[j].shape.describe()}) '
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

    The boxes are supports (``Shape.support``): the cells each shape may touch, which is the
    right question here because an overlap the refusal misses is the silent outcome. A Slab's
    support is full on its two other axes, so two slabs on different axes always intersect, as
    stage 1 said; a Region is snapped on all three. A span that snapped to nothing was already
    refused where it was snapped.
    """
    return all(a0 < b1 and b0 < a1 for (a0, a1), (b0, b1) in zip(left, right))


def check_smoothed(scene, grid):
    """What a smoothed shape may meet in C2: the five doors subpixel averaging opens.

    Each is a case where the rule C2 writes is exact and the neighbouring case is not, and the
    principle stage 4 is built on is that the second is refused **by name**, with the spelling
    that works, rather than answered approximately. Four of the five carry the number the
    scene reads with the door held open, which is what "silently wrong" means here: the run
    completes and the answer has the right shape.

    **1. No conductivity and no poles where a smoothed shape reaches.** C2 smooths ``eps`` and
    ``mu`` and nothing else. ``sigma`` and the pole coefficients keep the cell-snapped rule
    they have today -- the structure's material over the cells its shape *fills* -- and a
    shape with a fractional edge has no such set of cells, so there is no rule to keep. The
    two answers that exist are both measured and both belong to a later commit:

    * ``sigma`` mixed arithmetically on every component, exact for the two tangential ones and
      first order on the normal one. The analogous legacy inconsistency is priced in
      ``docs/validation/fdtd/fdtd-oblique-pulse.md``:84-88 at TM 0.0595 against TE 0.0082 at resolution 40.
      The exact normal form is the same machinery (harmonically mixing ``eps_r + i sigma/w``
      with a real medium is exactly ``eps_inf' + one Debye pole``), and it turns a declared
      non-dispersive structure into one carrying a plan term, which is why it is priced rather
      than assumed.
    * a one-pole dispersive medium mixed exactly, as **two** Lorentz poles with the same
      ``gamma`` and closed-form per-cell roots. Arithmetic-only pole smoothing is
      qualitatively wrong for a metal: at 0.532 um and ``f = 0.5``, gold reads
      ``eps = -4.70 + 0.13i`` arithmetically against ``+2.21`` harmonically.

    Neither is guessed at here. The door covers the smoothed structure's **own** material, the
    background (which fills every cell, so it always meets one) and any structure whose
    support shares a cell with it -- because in all three cases the unsmoothed quantity would
    be read over a voxel the smoothed shape only partly fills, and the number that comes back
    is plausible. Measured with the door held open: a lossless ``Sphere(radius=0.25)`` in a
    ``Conductor(eps=1, sigma=0.2)`` background lowers ``sigma = 0.2001`` on all 402 voxels the
    ball fills **outright**, where the truth is 0 -- not a smoothing error at the surface, the
    whole interior of the ball conducting.

    **2. No smoothed shape at a wrapping face.** A body that straddles a periodic face is only
    whole if the geometry is summed over the 3x3x3 ring of its images, which
    ``_shapes.occupancy`` can do but only when the body lies *in* the cell -- and a body
    carries no position to fold with, so the fold belongs to the caller that owns the cell. C2
    does not build it. Measured with this door held open: ``Sphere(radius=0.2)`` centred on the
    ``x-`` face of a 1.2-wide periodic cell lowers **59.3%** of its volume against 1.0000 for
    the same ball at the centre, and nothing sees the missing 40.7% -- ``sum_c w_c`` still
    reads 1 to 4.4e-16 on every voxel with any surface in it. The support box is tested with
    its own one-cell margin, so this refuses a ball that comes within a cell of the face as
    well as one that crosses it; that is deliberately conservative and the message says how far
    to move it.

    The wrapping faces are the three ``boundaries._WRAPPING`` names, ``Periodic``, ``Bloch``
    **and** ``Axis1D``, and the third gets its own sentence rather than the image-ring one. An
    ``Axis1D`` axis is one cell by its own contract and stands for an axis the field is
    *invariant* along, so the missing images are not the point: the only structure that axis
    can carry is one that is uniform along it, whatever the cell count. Measured with the door
    held open, ``Sphere(radius=0.2)`` on ``Boundaries(x=Periodic, y=Axis1D, z=PEC)`` over a
    ``(1.2, 0.05, 1.2)`` domain lowers to ``(3, 24, 1, 24)`` holding **18.4%** of the ball's
    volume -- the slab cross-section, which is neither the ball nor the cylinder a reader would
    take it for -- while the identical scene with ``y=Periodic()`` is refused by name.

    **3. No smoothed shape the grid cannot carry whole at a face of the box.** A cell-snapped
    ``fd.Region`` is clipped to the domain exactly, because its material is a range of cell
    indices. A smoothed body is not, and the six components do not agree about where the box
    ends -- three of them have their Yee point on the node plane and three half a cell off it.
    **Two different things go wrong because of that, and the door's two arms are split by which
    one it is, not by which end of the axis it happened at.**

    *A body that LEAVES the domain* has material in voxels that lie wholly outside it, and
    which three components read that material -- and whether they gain it or lose it with the
    rest -- flips between the two ends of the axis. Measured on a 1.2 box at ``dx = 0.05``,
    ``Sphere(radius=0.2, center=(0, 0.59, 0))``, sticking **3.8 cells** past the ``y+`` wall:
    ``Ey``, ``Hx`` and ``Hz`` lower **0.01801075** of volume, which is exactly the part inside
    the wall, and ``Ex``, ``Ez`` and ``Hy`` lower **0.01487374**, **17.4% less**, because they
    also drop the last half cell. Mirror it to ``y-`` and the first three still read 0.01801075
    while the other three read **0.02110849**, **17.2% more**. Centred *on* the ``x-`` face,
    ``Ex`` reads half the ball (**0.01675516**, ``2 pi R^3/3``) and ``Ey`` and ``Ez``
    **0.01988039**, **18.7% more**. None of those is the ball, the hemisphere or the cap.

    *A body the plan TRIMS* is entirely inside the domain and still loses its last half cell at
    a ``+`` face: the three components whose point is on the node plane have their last point
    on the last node, so their last voxel stops at ``high - dx/2`` and nothing of theirs holds
    what lies between that and the wall. Measured on the same ball at ``dx = 0.05``: with its
    surface **0.48 cells clear** of the ``x+`` wall it falls short of ``4 pi R^3/3`` by
    **1.872e-05 of that volume** (6.27e-07 absolute) on ``Ey`` and ``Ez`` and by nothing on
    ``Ex``, rising to **1.114e-02 of it** (3.73e-04) at tangency, while the mirror image of
    each of those at the ``x-`` wall loses nothing at all (1.3e-15). A ball **inscribed** in the
    domain is therefore refused at its ``+`` faces and accepted at its ``-`` ones, which is not
    a symmetry the grid has to offer. The remedy that works here is to keep the surface half a
    cell clear of the wall or to enlarge the domain; "move it inside" and "say the truncated
    body with the cell-snapped ``fd.Region``" are the other arm's and mean nothing here, because
    this body is already inside and nothing sticks out to truncate.

    **What separates them is the probe voxel's own geometry**, not the side. At a ``-`` face
    all six probe voxels lie wholly outside the domain; at a ``+`` face the three whose Yee
    offset along the axis is half a cell are wholly outside and the other three straddle the
    wall. Material in a wholly outside one is the first arm, material in the straddling one
    alone is the second, and the first wins wherever both are true. Splitting by side instead
    put ``Sphere(radius=0.2, center=(0, 0.59, 0))`` -- the 3.8-cell case above, with three
    probe voxels full of material outside the domain -- on the second arm's message, which told
    it that "nothing outside the box is being counted here" and that "there is nothing sticking
    out to truncate".

    The test is on the body and not on the support box, because the support carries a cell of
    margin: :func:`_reaches_outside` asks the body for its fill over the cell layer just beyond
    each face the support reaches, at all six components' points, and refuses when any of them
    is above :func:`_clip_tolerance` -- what separates the exact clip's own round-off from a
    leak worth refusing, measured and stated at :data:`_CLIP_ROUNDOFF`. A shape that is
    **uniform across** the face -- a substrate, a half space -- answers ``Shape.uniform_along``
    ``True`` in that cell layer and is exempt from both arms, because a body that does not
    change across the wall neither gains nor loses anything there. A ``Symmetry`` face is not
    a wall for this door: what a body may be at a mirror plane is :func:`check_symmetry`'s
    question (C5).

    **4. No anisotropic eps or mu where a smoothed shape reaches.** The rule
    ``coefficients.mix`` applies is the diagonal of the Farjadpour/Kottke tensor's inverse
    **for two isotropic phases**, and all three of its citations derive it for that case. The
    Kottke construction for an anisotropic phase is the tau-transform, which is not a blend of
    each component with that component's own ``eps`` -- and on a ball the surface normal is
    almost never along a principal axis, so the case the scalar rule is wrong for is the
    generic one and not a corner. Held open, ``Sphere(radius=0.2)`` with
    ``Material(eps=(2.0, 5.0, 9.0))`` lowers with per-component maxima ``[2.0, 5.0, 9.0]`` and
    no refusal at all. The door covers the same three media as door 1 and reads the triples the
    trace-safe way (:func:`_isotropic`): a scalar ``eps=`` puts the *same object* in all three
    slots, which is the reading ``materials._positive`` already uses.

    **5. No two smoothed shapes whose surfaces pass through one voxel.** A painter mixes its
    material into whatever the cell already holds, and that is exactly the two-phase interface
    rule -- *two* phases, and *one* surface. It composes exactly where the arriving shape fills
    the voxel whole (so a core/shell ball pair is right and is not refused), and it does not
    compose where two smoothed surfaces are in the same voxel: the second mix treats the first mixture as a medium with a second interface
    across it, and the background survives with weight ``(1-f1)(1-f2)`` in a voxel that holds
    none of it. Measured against a closed form, two half spaces of the **same** material whose
    faces sit in one cell (tops at 12.5 and 12.5000001 cells, ``eps`` 12 in 1) read
    ``eps_z = 3.20000`` where the truth is ``1.84615`` -- **73% high**; at 12.45/12.55 cells,
    3.22364 against 2.01681 (60%); at 12.2/12.8, 4.34783 against 3.75000 (16%); a whole cell
    apart, 5.2e-13 and the defect is gone. Through the public door, the same
    ``fd.Sphere(radius=0.2, center=(0.013, -0.007, 0.021))`` with the same ``Material(eps=12)``
    declared **twice** instead of once moves ``eps`` by up to **122%** (8.144451 against
    3.666166) on 648 voxels, and 796 voxels move by more than 1%.

    The test is drawn on voxels and not on bounding boxes, which is the design's own stated
    risk (a bounding-box test falsely refuses a dimer at any gap below 0.414 diameters, because
    two equal spheres on a 45 degree diagonal have overlapping axis-aligned boxes out to a
    centre distance of ``2 sqrt(2) R`` -- the canonical plasmonic scene).
    :func:`_shared_surface_voxels` walks the intersection of the two supports at all six
    components' points, marks where each body's signed distance is within half a voxel
    diagonal (a surface inside a voxel is within that of the voxel's own point, and a composed
    CSG distance under-estimates, so the mark cannot be missed), and only then pays for the
    exact fill at the marked points, refusing where both are fractional **to within
    :func:`_clip_tolerance` of the two ends** -- because the exact clip returns neither end
    exactly and comparing against 0 and 1 themselves refused 740 voxels of a legal scene. It is the
    narrow case C2's one smoothed shape can reach of the media-counting refusal C5 owns; a
    creased interface *within* one body (a prism edge, a block corner) is one painter and is
    handled exactly, not refused. A flush Cylinder or unturned Box on an unturned Box face
    is the narrow exception: ``junction.classify`` certifies the whole shared set and
    returns static write masks for a separate split or staircase lowering. Every other
    shared pair retains this refusal.

    **6. No smoothed surface in a voxel a cell-snapped face cuts** (C4). A snapped shape gives
    whole cells, so the three components whose Yee point lies on one of its faces have a voxel
    half on each side of it, painted all one side or all the other; on its own that is the
    snapped shape's documented half-cell rule. A smoothed body cutting the same voxel mixes
    into what the painter left, so the voxel's other two media are read as one: three media
    where the rule knows two. Measured by hand from the geometry, a pillar ``fd.Box`` of
    ``eps`` 12 standing flush on an ``fd.Slab`` substrate of ``eps`` 2.25 at ``dx = 0.05``
    reads tangential ``eps`` 6.5 at the joint where the voxel holds 7.125 (8.8%), and 3.2
    against 3.825 (16%) with its base 0.3 of a cell above the face. Sinking the base half a
    cell into the substrate puts the pillar's material over the whole joint voxel under it,
    but not at its rim: where its sides meet the substrate's face, 24 voxels of that pillar
    hold the substrate, the pillar and the background at once, and they are refused too (S7
    counts that rim). A body a voxel clear of every snapped face is right and is not refused.
    It is the scene a metasurface pillar on its substrate is, which C4's bodies made the
    ordinary one, and before C4 a ball on a substrate did it silently. :func:`_snapped_joint` finds the voxels: the snapped cells'
    exact share of each voxel (three 1-D overlaps; a face on the domain's own wall is no
    face), the body's fill only where that share is fractional and the body's surface within
    reach, both against :data:`_CLIP_ROUNDOFF`.
    """
    from wavebit.fdtd._legacy_boundaries import Axis1D
    from wavebit.fdtd._legacy_boundaries import Bloch
    from wavebit.fdtd._legacy_boundaries import Periodic
    from wavebit.fdtd._legacy_boundaries import Symmetry
    from wavebit.fdtd._prepare.occupancy import smoothed_structures
    structures = tuple(getattr(scene, 'structures', ()) or ())
    smoothed = smoothed_structures(scene)
    from wavebit.fdtd._prepare.conductive_film import descriptor as conductive_film
    film = conductive_film(scene, grid)
    from wavebit.fdtd._prepare.conductive_stack import descriptor as conductive_stack
    stack = conductive_stack(scene, grid)
    from wavebit.fdtd._prepare.debye_stack import descriptor as debye_stack
    relaxation_stack = debye_stack(scene, grid)
    from wavebit.fdtd._prepare.oscillator_stack import descriptor as oscillator_stack
    resonant_stack = oscillator_stack(scene, grid)
    from wavebit.fdtd._prepare.single_pole_film import descriptor as single_pole_film
    dispersive_film = single_pole_film(scene, grid)
    if dispersive_film is None:
        from wavebit.fdtd._prepare.oscillator_sum_film import descriptor as oscillator_sum_film
        dispersive_film = oscillator_sum_film(scene, grid)
    if dispersive_film is None:
        from wavebit.fdtd._prepare.hybrid_film import descriptor as hybrid_film
        dispersive_film = hybrid_film(scene, grid)
    from wavebit.fdtd._prepare.debye_film import descriptor as debye_film
    relaxation_film = debye_film(scene, grid)
    if relaxation_film is None:
        from wavebit.fdtd._prepare.two_debye_film import descriptor as two_debye_film
        relaxation_film = two_debye_film(scene, grid)
    if relaxation_film is None:
        from wavebit.fdtd._prepare.debye_sum_film import descriptor as debye_sum_film
        relaxation_film = debye_sum_film(scene, grid)
    for index, structure, body in smoothed:
        box = structure.shape.support(grid)
        where = f'structures[{index}] ({structure.shape.describe()})'
        for a, axis in enumerate('xyz'):
            k0, k1 = box[a]
            if k1 <= k0:
                raise ValueError(
                    f'{where} reaches no cell along {axis}: it lies outside the box '
                    f'entirely. A smoothed shape is not clipped to the domain the way a '
                    f'cell-snapped fd.Region is, so it would lower to nothing at all here while '
                    f'the scene still reported it. (One that reaches the box and still '
                    f'leaves it, or whose last half cell the grid has no point for, is refused '
                    f'too, a few lines below, and each on its own message.) '
                    f'Move it inside Domain(size=..., center=...), or enlarge the domain.')
            face = scene.boundaries.face(axis + '-')
            # A body uniform along the axis over the whole domain is the same cross-section
            # in every layer, wrapped or not: it needs no image ring and is carried.
            uniform = structure.shape.uniform_along(a, (0, grid.n[a]), grid)
            if isinstance(face, Axis1D) and not uniform:
                raise ValueError(
                    f'{where} is smoothed and {axis} is an fd.Axis1D face. Axis1D stands for '
                    f'an axis the field is *invariant* along -- its domain is one cell and the '
                    f'derivative along it is i k exactly -- so the only structure it can carry '
                    f'is one that is uniform along it, whatever the cell count. A body that '
                    f'curves is not: measured on a Sphere(radius=0.2) over a domain one '
                    f'cell thick, the scene lowers to an invariant slab cross-section holding '
                    f'18.4% of the ball\'s volume -- neither the ball asked for nor the '
                    f'cylinder a reader would take it for -- and no warning fires. Say the '
                    f'invariant cross-section with a body extruded along {axis} that reaches '
                    f'a whole cell beyond each {axis} face (its extent at least the domain plus '
                    f'2 dx = {2 * grid.dx:g}) -- fd.Cylinder(axis={axis!r}), '
                    f'fd.Prism(axis={axis!r}), or an unturned fd.Box spanning {axis} -- or with '
                    f'the cell-snapped fd.Region, or give {axis} real cells and a face that is '
                    f'not invariant.')
            if (isinstance(face, (Periodic, Bloch)) and not uniform
                    and not (k0 > 0 and k1 < grid.n[a])):
                raise ValueError(
                    f'{where} reaches the {type(face).__name__} '
                    f'{axis} face (cells {k0}..{k1} of {grid.n[a]}). A smoothed shape that '
                    f'crosses a periodic face is only whole if its geometry is summed over the '
                    f'ring of its own periodic images, and C2 does not do that: the part '
                    f'outside the cell is simply lost, measured at 40.7% of the volume for a '
                    f'ball centred on the face, with every invariant the lowering checks still '
                    f'satisfied. Keep it at least one cell clear of both {axis} faces; or, if '
                    f'it is meant to run the whole period, say so with a body uniform along '
                    f'{axis} over the whole domain -- extruded along {axis}, or an unturned '
                    f'fd.Box spanning it, a cell beyond each face -- which is carried because '
                    f'every layer holds the same section; or say the periodic geometry with '
                    f'the cell-snapped fd.Region (which is clipped, and whose clipping is how a '
                    f'lamellar grating spells "runs the whole period").')
        mirrors = tuple((a, side) for a, axis in enumerate('xyz') for side in '-+'
                        if isinstance(scene.boundaries.face(axis + side), Symmetry))
        outside = _reaches_outside(structure.shape, body, grid, box, mirrors)
        if outside is not None:
            cause, a, side, fill = outside
            axis = 'xyz'[a]
            if cause is _LEAVES:
                raise ValueError(
                    f'{where} leaves the domain through the {axis}{side} face: its material '
                    f'fills {fill:.4f} of a voxel that lies wholly outside the domain there. A '
                    f'smoothed shape is not clipped to the domain the way a cell-snapped '
                    f'fd.Region is -- a Region\'s material is a range of cell indices and is '
                    f'cut to the grid exactly, a body\'s is a fraction of each voxel and out '
                    f'there is no voxel to cut. What the lowering does with the part outside is neither '
                    f'"keep" nor "drop": the six components disagree, because three of them '
                    f'have their Yee point on the node plane along {axis} and three half a '
                    f'cell off it, and WHICH three are wrong flips between the two ends of the '
                    f'axis. Measured on a 1.2 box at dx = 0.05, Sphere(radius=0.2) sticking '
                    f'3.8 cells past the y+ wall lowers 0.01801075 of volume on Ey, Hx and Hz '
                    f'-- exactly the part inside the wall -- and 0.01487374 on Ex, Ez and Hy, '
                    f'17.4% LESS; mirrored to y- the same three read 0.01801075 and the other '
                    f'three 0.02110849, 17.2% MORE. A ball centred on the x- face reads half '
                    f'the ball (0.01675516) on Ex and 0.01988039 on Ey and Ez, 18.7% more. So '
                    f'a body that leaves the box is none of the shapes a reader would name, '
                    f'and it is refused rather than silently trimmed. Move it inside '
                    f'Domain(size=..., center=...), enlarge the domain, or say the truncated '
                    f'body with the cell-snapped fd.Region, which really is clipped. Inside is '
                    f'not quite enough at a + face: a surface nearer than half a cell '
                    f'({0.5 * grid.dx:g}) to a + wall loses that last half cell, and the OTHER '
                    f'arm of this door says so in its own words. A shape that is uniform '
                    f'ACROSS the face -- a substrate, a half space -- is not refused at all: '
                    f'it answers uniform_along() True in that cell layer and the clip takes '
                    f'nothing from it.')
            raise ValueError(
                f'{where} is inside the domain, and its surface still comes within half a '
                f'cell of the {axis}+ face: {fill:.4f} of a voxel straddling that wall holds '
                f'its material, and the three components whose Yee point sits on the node '
                f'plane along {axis} have no point beyond wall - dx/2 to carry it, so the '
                f'domain\'s own last half cell there is simply LOST. Nothing is outside the '
                f'box -- the probe voxel that read this is itself half INSIDE, and the three '
                f'whose probe voxels really are outside read exactly zero, which is how this '
                f'case is told from a body that leaves the domain. The body may be entirely '
                f'inside and this still happens, while its mirror image at the {axis}- wall '
                f'loses nothing at all (1.3e-15) -- not a symmetry the grid has to offer. '
                f'Measured, Sphere(radius=0.2) at dx = 0.05 whose surface is 0.48 cells clear '
                f'of the x+ wall falls short of 4 pi R^3/3 by 1.872e-05 OF that volume '
                f'(6.27e-07 of volume) on Ey and Ez and by nothing on Ex, and by 1.114e-02 of '
                f'it (3.73e-04) at tangency. Keep the surface at least half a cell '
                f'({0.5 * grid.dx:g}) clear of the {axis}+ wall, or enlarge the domain with '
                f'Domain(size=..., center=...). "Move it inside the domain" and "say the '
                f'truncated body with the cell-snapped fd.Region" are the OTHER arm\'s remedies, '
                f'for a body that really does leave the box, and neither means anything here: '
                f'this one is already inside and there is nothing sticking out to truncate. A '
                f'shape that is uniform ACROSS the face -- a substrate, a half space -- is not '
                f'refused: it answers uniform_along() True in that cell layer and a body that '
                f'does not change across the wall loses nothing there.')
        media = [(structure.material, where),
                 (_background(scene), 'Scene(background=)')]
        for other, s in enumerate(structures):
            if other != index and _intersects(box, s.shape.support(grid)):
                media.append((s.material, f'structures[{other}] ({s.shape.describe()})'))
        for material, name in media:
            if not isinstance(material, (Conductor, Dispersive)):
                continue
            if (film is not None or stack is not None or dispersive_film is not None
                    or relaxation_film is not None or relaxation_stack is not None
                    or resonant_stack is not None) and name == where:
                continue  # These bounded C8 film scopes carry their exact memory.
            what = ('a conductivity' if isinstance(material, Conductor)
                    else f'{len(material.poles)} pole(s)')
            # Door 4 below reads the self case the same way. A structure does not "share cells
            # with" itself, and saying so was the first door a user met.
            whose = (f'is smoothed and its own material is an fd.{type(material).__name__} '
                     f'({what})' if name == where else
                     f'is an fd.{type(material).__name__} ({what}) and shares cells with '
                     f'{where}, which is smoothed')
            raise ValueError(
                f'{name} {whose}. Stage 4 C2 smooths eps and mu at each '
                f'component\'s own Yee point and nothing else: sigma and the pole '
                f'coefficients keep the cell-snapped rule, and a shape whose edge is not a '
                f'cell boundary has no cells to apply it over -- so a partly filled voxel '
                f'would carry the full conductivity or the full polarisation, which is a '
                f'plausible number and a wrong one. Use the cell-snapped fd.Region for the lossy '
                f'or dispersive part (first order in dx, and the whole of stages 1-3 rests on '
                f'it), or keep it out of the smoothed shape\'s cells. C8 currently admits '
                f'one concrete axis-aligned conductive, Drude/Lorentz-sum or Debye-sum '
                f'film, or a same-axis Conductor-only, Debye-only or single-oscillator '
                f'stack with disjoint padded supports, '
                f'in a plain dielectric, '
                f'clear of absorbing layers, with no initial field; this scene is outside '
                f'that measured scope.')
        for material, name in media:
            for what, triple in (('eps', material.eps), ('mu', material.mu_value)):
                if _isotropic(triple):
                    continue
                whose = ('is smoothed and declares' if name == where else
                         f'declares an anisotropic {what}={tuple(triple)!r} and shares cells '
                         f'with {where}, which is smoothed. It')
                raise ValueError(
                    f'{name} {whose} an anisotropic {what}={tuple(triple)!r}. '
                    f'The mixing rule C2 applies at a '
                    f'fractional edge is the diagonal of the Farjadpour/Kottke tensor\'s '
                    f'inverse for two ISOTROPIC media -- all three of its citations derive it '
                    f'for that case -- and blending each component separately with that '
                    f'component\'s own {what} is not the tau-transform the anisotropic '
                    f'construction calls for. On a ball the surface normal is almost never '
                    f'along a principal axis, so that is the generic case and not a corner. '
                    f'Held open, a Sphere with eps=(2.0, 5.0, 9.0) lowers with per-component '
                    f'maxima [2.0, 5.0, 9.0] and no sign that the rule was never derived for '
                    f'it. Give the smoothed shape and everything sharing its cells a scalar '
                    f'{what}=, or say the anisotropic medium with the cell-snapped fd.Region, '
                    f'whose material is not mixed at all. The tensor form is stage 5\'s, and '
                    f'study S1 found no robust order it buys (docs/validation/fdtd/fd-subpixel-occupancy.md), '
                    f'so nothing schedules it.')
    import wavebit.fdtd._prepare.junction as junction
    junctions = []
    for i in range(len(smoothed)):
        for j in range(i + 1, len(smoothed)):
            (ia, sa, ba), (ib, sb, bb) = smoothed[i], smoothed[j]
            boxa, boxb = sa.shape.support(grid), sb.shape.support(grid)
            if not _intersects(boxa, boxb):
                continue
            shared = tuple((max(u[0], v[0]), min(u[1], v[1])) for u, v in zip(boxa, boxb))
            count, worst = _shared_surface_voxels(ba, bb, grid, shared, include_common=True)
            if not count:
                continue
            record = junction.classify(scene, grid, ia, ib, count)
            if isinstance(record, junction.Junction):
                junctions.append(record)
                continue
            _, field, c, point, fa, fb = worst
            raise ValueError(
                f'structures[{ia}] ({sa.shape.describe()}) and structures[{ib}] '
                f'({sb.shape.describe()}) are both smoothed and their surfaces pass through '
                f'the same voxel: {count} of them, worst at the '
                f'{field}{"xyz"[c] if c < 3 else " common"} point '
                f'{point}, where they fill {fa:.4f} and {fb:.4f} of it -- the voxel the two '
                f'surfaces cut most evenly of all of them. A smoothed shape mixes '
                f'its material into whatever the cell already holds, which is the two-phase '
                f'interface rule -- two phases and ONE surface. Two surfaces in one voxel do '
                f'not compose: the second mix treats the first mixture as a medium with a '
                f'second interface across it, and the background survives with weight '
                f'(1-f1)(1-f2) in a voxel that holds none of it. Measured against a closed '
                f'form, two half spaces of the same material whose faces sit in one cell read '
                f'eps 3.20000 where the truth is 1.84615, 73% high, and the same fd.Sphere '
                f'declared twice instead of once moves eps by up to 122% on 648 voxels. Move '
                f'them until their SURFACES are a voxel apart (a dimer with a gap is fine and '
                f'is not refused -- this is drawn on voxels, not on bounding boxes), or say '
                f'the two as ONE structure so that there is one body, one interface and one '
                f'exact answer: today that is a single fd.Sphere, fd.Box, fd.Cylinder or '
                f'fd.Prism, or the cell-snapped fd.Region for the part that overlaps. The CSG '
                f'spellings that say it properly -- fd.Union, fd.Intersection and '
                f'fd.Difference, which resolve the two into one surface before any fill is '
                f'taken -- are not on the public door. Study S7 counted what this door '
                f'leaves: no single body and no canonical dimer reaches three media in a '
                f'voxel; the polytope clip that would answer them is C10, conditional. '
                f'No junction rule covers this pair: {record}. The junction rules '
                f"(Structure(junction=), default 'split') cover a Cylinder or an unturned "
                f'Box standing flush on a smoothed, unturned Box face.')

    snapped = [(j, s) for j, s in enumerate(structures) if s.shape.body() is None]
    for index, structure, body in smoothed:
        for j, other in snapped:
            joint = _snapped_joint(body, other.shape, grid, structure.shape.support(grid))
            if joint is None:
                continue
            count, field, c, point, fill, share = joint
            raise ValueError(
                f'structures[{index}] ({structure.shape.describe()}) is smoothed and its '
                f'surface cuts {count} voxel(s) that a face of the cell-snapped '
                f'structures[{j}] ({other.shape.describe()}) cuts too -- worst at the '
                f'{field}{"xyz"[c]} point {point}, where the body fills {fill:.4f} of the voxel '
                f'and the snapped structure\'s cells {share:.4f}. A cell-snapped structure '
                f'owns whole cells, so to the painter such a voxel is all it or all not, while '
                f'the geometry is split; the smoothed body then mixes its material into '
                f'whatever the painter left, and the rest of the voxel is read as one medium '
                f'where it holds two. Measured, a pillar of eps 12 standing flush on a Slab '
                f'substrate of eps 2.25 at dx = 0.05: the tangential eps at the joint reads '
                f'6.5 where the voxel holds 7.125 (8.8%), and 3.2 against 3.825 (16%) with '
                f'the base 0.3 of a cell above the face -- with every invariant the lowering '
                f'checks still satisfied. Sinking the body into the snapped structure does '
                f'not cure it: where the body\'s side meets the snapped face, a voxel holds '
                f'three media (the snapped material, the body\'s and the background) and the '
                f'rule knows two. Keep the body\'s surface a voxel clear of the snapped '
                f'structure\'s faces. For a flush pillar, spell the substrate as a smoothed, '
                f"unturned fd.Box: Structure(junction=) uses 'split' by default. A "
                f'cell-snapped substrate stays refused under a smoothed pillar.')
    return junction.finish(scene, junctions)


#: The exact clip's own round-off: the floor of :func:`_clip_tolerance`, which doors 3 and 5 read
#: a fill against, and still the whole tolerance at the other sites that read one (below).
#:
#: A fill is a number in ``[0, 1]`` built from coordinates of size ``|x|``, so its round-off is
#: of order ``|x|/dx`` ulps: it grows with the grid and with how far the body's own description
#: reaches, not with the body's size. Measured over the two bodies C2 can reach -- the ball, and
#: the box far wider than the domain that the co-location oracle spells a half space with -- at
#: ``dx`` from 0.2 down to 0.0125 and at four sub-cell placements each:
#:
#:     end                                        worst deviation
#:     a voxel the body fills WHOLE  (f = 1)          4.0e-13
#:     a voxel the body MISSES       (f = 0)          1.2e-13
#:
#: and ``coefficients._Smoothed`` already carries -2.55e-12 for the second end at twenty cells
#: per radius. **Neither end is ever reached exactly**: at ``dx = 0.05`` a voxel the half space
#: fills whole reads 0.99999999999991 -- 8.8e-14 short -- so a condition written ``f < 1`` is
#: true for it and a door written that way refuses the scene it was meant to let through (740
#: voxels of two half spaces of one material; the two painting orders agree to 3.8e-14 there).
#:
#: The smallest thing worth refusing at either end -- a leak into a voxel outside the domain, a
#: second surface inside one -- is a fraction of a voxel. This constant sits three orders above
#: the measured round-off and eight below that. It bounds the geometric fill distinction
#: made by this door, not the resulting material-coefficient error: the mixing rule's
#: sensitivity also depends on the positive material values and their ratio.
#:
#: **It is a floor and not the whole tolerance**, because the sweep above holds ``|x|/dx`` at
#: about 24 while the paragraph's own first sentence says the round-off grows with that ratio.
#: It does, cleanly. Measured on ``fd.Sphere(radius=0.2)`` at ``dx = 0.05`` with the whole
#: scene -- domain and ball together -- moved away from the coordinate origin, reading the
#: voxels the ball fills WHOLE (``d <= -dx sqrt(3)/2``):
#:
#:     |x|/dx        |f - 1| where the ball fills whole       as a multiple of (|x|/dx) eps
#:     1.2e+01                   6.7e-16                                 0.25
#:     2.0e+03                   6.8e-13                                 1.53
#:     2.0e+05                   4.4e-11                                 0.98
#:     2.0e+06                   7.0e-10                                 1.57
#:     2.0e+07                   2.8e-09                                 0.63
#:     2.0e+08                   4.5e-08                                 1.01
#:     2.0e+09                   7.2e-07                                 1.61
#:
#: eleven decades on one law, and the half space (top 0.013 above the centre) reads the same
#: numbers from ``2.0e3`` on; at ``1.2e1`` it reads 1.0e-13, the ratio-free round-off the sweep
#: above was set by. The law is a staircase and not a line: the error steps with the binade of
#: ``|x|``, so the coefficient swings within each step -- 0.62 to 2.31 over forty ratios from
#: 4.0e5 to 8.0e6, all six components -- and the absolute ``1e-9`` is first *crossed* at
#: ``|x|/dx`` of about **5.2e6**, where ``|x|`` passes ``2^18`` at ``dx = 0.05`` and the error
#: steps from 7.0e-10 to 2.8e-9. From there a door written against it alone reads a voxel the
#: body fills whole as fractional, which is exactly what the 740-voxel refusal did at
#: ``dx = 0.05``. The end the body **misses** never drifts at all: 0.000e+00 at every one of
#: those ratios, for both bodies, which is why door 3, which reads only that end, was never in
#: danger and door 5, which reads both, is. :func:`_clip_tolerance` is what those two call, and
#: ``tests/fdtd/test_fd_occupancy.py::test_the_clip_tolerance_follows_the_grid_and_not_a_number``
#: re-measures the table. Every other reader still reads this constant bare -- here, door 6's
#: :func:`_snapped_joint` (both ends, as door 5), :func:`check_symmetry`, the report's cut
#: counts and the fill-contract guard (which scales its own); outside this module the
#: constitutive plan, the junction, the shape ranges and ``_post`` -- and
#: whether they move to the grid-scaled tolerance is a decision not taken here.
_CLIP_ROUNDOFF = 1e-9

#: The margin :func:`_clip_tolerance` carries over that ``(|x|/dx) eps`` law -- more than two
#: orders over the worst coefficient measured (2.31), about what :data:`_CLIP_ROUNDOFF`
#: carries over the round-off at the grids this suite runs (1e-9 against 4.0e-13), and for the
#: same reason: reading a whole voxel as fractional refuses a legal scene outright. This
#: geometric tolerance is not a material-error budget. The two arms cross at ``|x|/dx`` of
#: about ``4.5e3``; the historical origin-centred fixtures remain below it, while the new
#: translated-grid gates exercise the relative arm.
#:
#: The tolerance passes 1% of a voxel at ``|x|/dx`` of about ``4.5e10``; admission below
#: the range limit is not a geometry-accuracy guarantee. At ``tol >= 0.5`` the two endpoint
#: neighbourhoods meet and no fill can satisfy ``tol < f < 1 - tol``. Continuing would disable
#: the shared-surface door; ``tol >= 1`` would also hide every outside fill. The helper refuses
#: that range and nonfinite tolerances explicitly, without a silent cap. Recentring the scene
#: or increasing ``dx`` reduces the coordinate-to-cell ratio.
_CLIP_ULPS = 1e3

#: ``float64``'s own step, the unit the law above is written in.
_ULP = float(np.finfo(np.float64).eps)


def _clip_tolerance(grid):
    """How far from 0 and 1 a fill may land on **this** grid and still be read as an end.

    :data:`_CLIP_ROUNDOFF` floors it and the measured ``(|x|/dx) eps`` law lifts it once the
    scene's own coordinates are large, where ``|x|`` is the furthest corner of the domain from
    the coordinate origin -- the largest number a voxel centre here can be built from, and so
    the one whose ulp the clip's cancellation is in.
    """
    reach = max(max(abs(grid.low[a]), abs(grid.low[a] + grid.n[a] * grid.dx)) for a in range(3))
    tol = max(_CLIP_ROUNDOFF, _CLIP_ULPS * (reach / grid.dx) * _ULP)
    if not math.isfinite(tol) or tol >= 0.5:
        raise ValueError(
            f'ClipPrecisionRefused: the coordinate-to-cell ratio requires fill tolerance '
            f'{tol!r}; it must be finite and below 0.5. The endpoint neighbourhoods '
            f'overlap at 0.5, so no shared-surface fill can be distinguished there. '
            f'Recenter the domain and all geometry near the origin, or coarsen the grid '
            f'(increase Resolution(dx=...)); the tolerance is not silently capped.')
    return tol

#: Half a voxel's diagonal, in cells. A surface inside the voxel centred on a point is within
#: this of that point, so ``|d| > _VOXEL_REACH * dx`` proves the surface is not in the voxel --
#: and a composed CSG distance under-estimates, so the proof holds for those too.
_VOXEL_REACH = math.sqrt(3.0) / 2.0


def _isotropic(triple):
    """Whether a material triple is one number, read the way a traced value allows.

    ``materials._diagonal`` puts the *same object* in all three slots for a scalar ``eps=``,
    which is the reading ``materials._positive`` already uses to decide whether to name an axis
    in its refusal. A traced triple of three distinct tracers cannot be shown equal and is
    therefore not isotropic here -- which is the refusing direction.
    """
    if triple[0] is triple[1] and triple[1] is triple[2]:
        return True
    values = [concrete(v) for v in triple]
    return None not in values and values[0] == values[1] == values[2]


#: The body's material is **outside the domain**: it was read in a probe voxel that lies wholly
#: outside, so the run would gain material the scene does not have there -- or, on the three
#: components that stop at the wall, lose it. Either way the six components disagree.
_LEAVES = 'leaves'

#: The body is **inside** the domain and the grid still cannot carry all of it: its material
#: reaches into the last half cell before a ``+`` wall, which the three components whose Yee
#: point sits on the node plane have no point of their own in. Nothing is outside the box.
_TRIMMED = 'trimmed'


def _reaches_outside(shape, body, grid, box, mirrors=()):
    """``(cause, axis, '-'/'+', fill)`` where the body is not lowered whole, or ``None``.

    A face in ``mirrors`` (``(axis index, side)`` pairs) is a ``Symmetry`` plane, and what a
    body may be there is :func:`check_symmetry`'s question, not this one: a body symmetric
    about the plane is the right way to cross it, and neither of this door's two messages is
    true of it.

    Asked of the **body** and not of the support box, because the support carries a cell of
    margin. For each face the support reaches, the cell layer one cell beyond it is built at
    all six components' points (widened by a cell on the two transverse axes, so a body
    leaving through a corner is seen on whichever axis it leaves by) and the body's own fill
    is taken there. An axis the shape is **uniform along** in that boundary cell layer is
    exempt: nothing is gained or lost across the wall by a body that does not change across
    it, which is what lets a substrate or a half space run out of the box.

    **The split is by CAUSE and not by side, and it is the probe voxel's own geometry that
    decides which.** At a ``-`` face the probe layer is cell ``-1`` and all six components'
    voxels there lie wholly outside the domain. At a ``+`` face the probe layer is cell ``n``
    and the six split in two: the three whose Yee offset along that axis is half a cell sit at
    ``wall + dx/2``, so their probe voxel is ``[wall, wall + dx]`` and is wholly outside too,
    while the other three sit exactly ON the wall and theirs is ``[wall - dx/2, wall + dx/2]``,
    half **inside**. So

    * a fill in a **wholly outside** probe voxel -- either face -- is material outside the
      domain, and that is :data:`_LEAVES`;
    * a fill in the ``+`` face's **half inside** probe voxel, with nothing in the wholly
      outside ones, is material in the domain's own last half cell that those same three
      components have no point to carry, and that is :data:`_TRIMMED`.

    Measured on the 1.2 box at ``dx = 0.05``: the ball inscribed in it reads ``Ey`` 0.482619,
    ``Ez`` 0.482619, ``Hx`` 0.472177 in the ``x+`` probe -- all of it at ``x <= 0.6``, i.e.
    inside -- while ``Ex``, ``Hy`` and ``Hz``, whose probe voxels really are outside, read
    0.000000, so that scene is ``_TRIMMED``. ``Sphere(radius=0.2, center=(0, 0.59, 0))``, which
    sticks **3.8 cells** past the ``y+`` wall, reads 1.000000 in every one of the six and has
    37, 30 and 30 whole voxels in the three wholly outside ones, so it is ``_LEAVES``. Splitting
    by side alone put the second of those on the first's message, which told a body with three
    probe voxels full of material outside the domain that "nothing outside the box is being
    counted here" and that "there is nothing sticking out to truncate".

    ``_LEAVES`` wins wherever both are true and is returned as soon as it is found, because it
    is the stronger statement and its remedy is the one to act on first; a ``_TRIMMED`` face
    found earlier in the axis order is held and returned only if no face leaves.

    Two stages, as in :func:`_shared_surface_voxels` and for the same reason -- the exact fill
    is 97% of what a smoothed lowering costs, and the answer here is almost always "nothing
    out there". The first is the signed distance, and it is ``d <= reach`` and **not**
    ``|d| <= reach``: a voxel the body fills *whole* is deep inside it, where ``d`` is large
    and negative, so the absolute value would step straight over a half space crossing the
    wall. Any material in the voxel puts some point ``q`` of the body within it, so
    ``d(p) <= |p - q| <= reach`` by the Lipschitz bound, and no marked voxel can be missed.
    Measured on the worst scene in the suite -- a ball whose support clips to the whole 24^3
    grid, so all six faces are tested over the full transverse extent -- the first stage takes
    ``plan()`` from **357 ms to 3.4 ms**, and the scene that is refused still pays the fill only
    where the body really is.
    """
    from wavebit.fdtd._kernel.grid import OFFSETS
    from wavebit.fdtd._shapes import occupancy as body_occupancy
    from wavebit.fdtd._prepare.occupancy import yee_points
    reach, tol = _VOXEL_REACH * grid.dx, _clip_tolerance(grid)
    trimmed = None
    for a in range(3):
        for side, reached, layer in (('-', box[a][0] == 0, (0, 1)),
                                     ('+', box[a][1] == grid.n[a], (grid.n[a] - 1, grid.n[a]))):
            if not reached or (a, side) in mirrors or shape.uniform_along(a, layer, grid):
                continue
            beyond = [(box[b][0] - 1, box[b][1] + 1) for b in range(3)]
            beyond[a] = (-1, 0) if side == '-' else (grid.n[a], grid.n[a] + 1)
            away = straddling = 0.0
            for field in ('E', 'H'):
                for c in range(3):
                    points = np.asarray(yee_points(grid, tuple(beyond), field, c))
                    near = np.asarray(body.surface(points)[0]) <= reach
                    if not near.any():
                        continue
                    fill, _ = body_occupancy(body, points[near], grid.dx)
                    worst = float(np.max(np.asarray(fill)))
                    # A probe voxel is wholly outside unless it is the one straddling a `+`
                    # wall, which only the three node-plane components have there.
                    if side == '-' or OFFSETS[field][c][a]:
                        away = max(away, worst)
                    else:
                        straddling = max(straddling, worst)
            if away > tol:
                return _LEAVES, a, side, away
            if straddling > tol and trimmed is None:
                trimmed = (_TRIMMED, a, side, straddling)
    return trimmed


def _snapped_joint(body, shape, grid, box):
    """Where a smoothed body's surface cuts a voxel that a snapped shape's face also cuts.

    ``(count, field, c, point, fill, share)`` for the most evenly cut such voxel, or ``None``.
    A cell-snapped shape gives whole cells: its material is on the Yee points of the cells
    ``filled_cells(grid)``, all six components alike. So a voxel that straddles one of its
    faces -- the three components whose Yee point lies on the face have their voxel half on
    each side -- is painted as all one side or all the other, while the geometry is split. On
    its own that is the snapped shape's documented half-cell rule. What goes wrong is a
    smoothed body cutting the *same* voxel: its painter mixes the body's material with
    whatever the voxel was painted, so one of the voxel's two other media is lost, and three
    media meet where the rule knows two.

    ``share`` is the snapped cells' exact part of each voxel, a product of three 1-D overlaps;
    a face on the domain's own wall is no face (the material runs to the wall and a clipped
    shape is not cut there), so a side that reaches the wall is open. The body's fill is
    asked only where a voxel straddles and the body's surface is within reach, as in
    :func:`_reaches_outside`.
    """
    from wavebit.fdtd._shapes import occupancy as body_occupancy
    from wavebit.fdtd._prepare.occupancy import yee_points
    cells = shape.filled_cells(grid)
    lo = [-np.inf if cells[a][0] == 0 else grid.low[a] + cells[a][0] * grid.dx for a in range(3)]
    hi = [np.inf if cells[a][1] == grid.n[a] else grid.low[a] + cells[a][1] * grid.dx
          for a in range(3)]
    reach = _VOXEL_REACH * grid.dx
    half = grid.dx / 2
    count, worst = 0, None
    for field in ('E', 'H'):
        for c in range(3):
            points = np.asarray(yee_points(grid, box, field, c)).reshape(-1, 3)
            share = np.ones(len(points))
            for a in range(3):
                share = share * np.clip(np.minimum(points[:, a] + half, hi[a])
                                        - np.maximum(points[:, a] - half, lo[a]),
                                        0.0, None) / grid.dx
            straddle = (share > _CLIP_ROUNDOFF) & (share < 1.0 - _CLIP_ROUNDOFF)
            if not straddle.any():
                continue
            near = straddle & (np.abs(np.asarray(body.surface(points)[0])) <= reach)
            if not near.any():
                continue
            fill = np.asarray(body_occupancy(body, points[near], grid.dx)[0])
            cut = (fill > _CLIP_ROUNDOFF) & (fill < 1.0 - _CLIP_ROUNDOFF)
            if not cut.any():
                continue
            count += int(cut.sum())
            evenness = np.where(cut, np.minimum(fill, 1.0 - fill), -1.0)
            k = int(np.argmax(evenness))
            if worst is None or evenness[k] > worst[0]:
                worst = (float(evenness[k]), field, c,
                         tuple(round(float(v), 6) for v in points[near][k]),
                         float(fill[k]), float(share[near][k]))
    if worst is None:
        return None
    return (count,) + worst[1:]


def _shared_surface_voxels(body_a, body_b, grid, box, include_common=False, masks=False):
    """``(count, worst)`` voxels of ``box`` whose fill is fractional for **both** bodies.

    Two stages, because the exact fill is 97% of what a smoothed lowering costs and the answer
    is almost always "none". First the two signed distances, which are cheap: a body whose
    surface is inside a voxel is within half a voxel diagonal of that voxel's own point, and a
    CSG-composed distance under-estimates the true one, so ``|d| > _VOXEL_REACH * dx`` proves
    the surface is elsewhere and no marked voxel can be missed. Then, at the marked points
    only, the exact fill: fractional for **both**, read against :func:`_clip_tolerance` at
    both ends.

    **The tolerance is the point, not a slackening of the rule.** The two ends carry meaning --
    ``f = 1`` means the arriving shape fills the voxel whole, which composes exactly, and
    ``f = 0`` means it is not there at all -- and an earlier writing of this function took that
    as licence to compare against 0 and 1 themselves. The exact clip never returns either: at
    ``dx = 0.05`` a voxel a half space fills whole reads 0.99999999999991, so ``f < 1`` was true
    for it and the door refused two half spaces of the SAME material, faces at 13.0 and 12.5
    cells, on **740** voxels -- not one of which either body cut, and whose two painting orders
    agree to 3.8e-14 against a one-painter lowering of the same geometry to 9.3e-14. It also
    inflated what the refusals it was right to make counted: 736 of the 2464 voxels of the
    gate's own 12.5/12.5000001 scene, and 68 of the 1883 of the public sphere-twice scene, are
    voxels one body fills whole or misses.

    ``worst`` is ``(share, field, c, point, fill_a, fill_b)`` for the voxel both surfaces cut
    most evenly -- the largest ``min(f, 1-f)`` over the two -- which is the one a reader should
    look at first, because it is where the two-phase rule is furthest from having a whole
    medium on either side of it. Without the tolerance that pick was meaningless too: on the
    scene above it named a voxel filled 1.0000 and 0.5000 "the voxel the two surfaces cut most
    evenly of all of them".
    """
    from wavebit.fdtd._shapes import occupancy as body_occupancy
    from wavebit.fdtd._prepare.occupancy import yee_points
    from wavebit.fdtd._prepare.constitutive import common_points
    reach, tol = _VOXEL_REACH * grid.dx, _clip_tolerance(grid)
    count, worst = 0, None
    markings = {}
    for field in ('E', 'H'):
        for c in range(4 if include_common else 3):
            points = np.asarray(yee_points(grid, box, field, c) if c < 3
                                else common_points(grid, box, field))
            if masks:
                markings[field, c] = np.zeros(points.shape[:-1], dtype=bool)
            near = np.abs(np.asarray(body_a.surface(points)[0])) <= reach
            if not near.any():
                continue
            near &= np.abs(np.asarray(body_b.surface(points)[0])) <= reach
            if not near.any():
                continue
            marked = points[near]
            fa = np.asarray(body_occupancy(body_a, marked, grid.dx)[0])
            fb = np.asarray(body_occupancy(body_b, marked, grid.dx)[0])
            both = ((fa > tol) & (fa < 1.0 - tol)
                    & (fb > tol) & (fb < 1.0 - tol))
            if masks:
                markings[field, c][near] = both
            if not both.any():
                continue
            count += int(both.sum())
            share = np.minimum(np.minimum(fa[both], 1.0 - fa[both]),
                               np.minimum(fb[both], 1.0 - fb[both]))
            k = int(np.argmax(share))
            if worst is None or float(share[k]) > worst[0]:
                worst = (float(share[k]), field, c,
                         tuple(round(float(v), 6) for v in marked[both][k]),
                         float(fa[both][k]), float(fb[both][k]))
    return (count, worst, markings) if masks else (count, worst)


def shown(length, dx):
    """``length`` as a message prints it: rounded to a billionth of a cell, and no ``-0``.

    A grid's low corner is ``center - n dx / 2`` in floating point, so a plane at 0 can come
    back ``-5.55e-17`` and a refusal would print that (C5).
    """
    return round(length / dx, 9) * dx + 0.0


def mirror_plane(grid, a, side):
    """Where a ``Symmetry`` face's plane is along axis ``a``: the node plane on the wall at a
    ``-`` face, the half plane ``dx/2`` inside the wall at a ``+`` face (``_kernel.faces``)."""
    if side == '-':
        return grid.low[a]
    return grid.low[a] + (grid.n[a] - 0.5) * grid.dx


def check_symmetry(scene, grid):
    """What a structure may be where it meets a ``Symmetry`` face (stage 4, C5).

    A ``Symmetry`` face stands for a scene that is its own mirror image. The run carries the
    domain side and reflects the field across the plane, so what it models is every
    structure's domain part **and that part's image**: whatever was declared beyond the plane
    is never read, and in a voxel the plane cuts, the declared body's fill is read where the
    scene holds the body and its image. The plane is the node plane on the wall at a ``-``
    face and the half plane ``dx/2`` **inside** the wall at a ``+`` face (:func:`mirror_plane`).

    A structure wholly on the domain side is a legitimate reading -- it and its image are the
    scene, the dimer a mirror exists to halve -- and it is not refused. Two readings are
    silently wrong, and each is refused by name, with the plane's position:

    * **A structure that crosses the plane and is not its own mirror image there.** The run
      models the mirror of its domain part, a different body. A smoothed body is judged by
      its exact fill at all six components' points in a band as deep as it reaches past the
      plane, against the fill at the mirrored points, to :data:`_CLIP_ROUNDOFF`; the audit
      measured a symmetric body at most 6.6e-13 there, a ball 1e-6 of a length off the plane
      4.0e-5, and a ball centred on a ``+`` face's wall rather than its plane 0.97. The test
      reads voxel fills, so a body symmetric voxel by voxel is accepted -- the unturned
      ``fd.Box``, the one body that may cross a mirror in stage 4, is separable, and for it
      that is symmetry. A cell-snapped shape crosses only where its snapped cells do, and the
      edge the mirror makes must lie within half a cell of the declared one -- the contract
      the snap keeps for every edge anyway (a symmetric declaration with an even cell count on
      a ``+`` face, whose plane is a cell centre, cannot do better than half a cell). Parity
      plays no part here: the body is different whatever the field does.
    * **A smoothed body within half a cell of the plane that does not cross it, where the
      mirror leaves the components on the plane free.** Their voxels are cut in half by the
      plane; the declared body fills the domain half and its image the other, and the fill
      read is the declared body's alone. Where the parity zeroes those components -- the
      tangential E and normal H on a ``-`` face at ``parity=-1``, the normal E and tangential
      H on a ``+`` face at ``parity=+1`` (``_kernel.faces``) -- their coefficient is never
      used and a flush half is exact, so it is not refused. Only E is read: the parity frees
      E and H on the plane together, and their voxels there tile the same slab half a cell
      either side of it, so a body in one is in the other. A cell-snapped shape is whole
      cells: flush against a ``-`` plane (the wall) it is exactly its own half. A ``+`` plane
      is the centre of the last cell, so a snapped face declared ON it rounds up or down by
      the cell count's parity and by round-off -- the plane's cell full (joined to the image)
      or empty (a cell apart from it); the C5 review found it empty at 15 of the counts 8 to
      40. That face is refused, naming the two spellings that are not ambiguous: to the wall,
      or half a cell short of the plane.

    Measured with this door held open (``studies/fdtd/symmetry_mirror/fdtd_symmetry_mirror.py``: a PEC cavity with
    an ``eps = 4`` bar along the mirror's axis, the ``Symmetry(parity=1)`` half box against
    the full box holding what was meant; the bar itself moves the resonance by about -10.8%):
    a symmetric ``fd.Box`` agrees to round-off; the box declared flush against the plane as
    ``[0, 0.2]`` lowers ``eps`` 2.5 on the plane at the bar's axis where the scene
    holds 4, and moves the resonance by +5.50e-03; the box declared
    ``[-0.1, 0.2]`` across the plane moves it by -1.55e-02 against the same box in the full
    cavity, and the same ``fd.Region`` by -9.09e-03; a box centred on the ``+`` face's
    wall instead of its plane, by +6.91e-03. Two controls say what those are not. At the
    parity that zeroes the plane's components the flush half is exact -- on the ``+`` face at
    ``parity=+1`` it agrees to -3.43e-15 although its plane voxel lowers ``eps``
    1.6 -- which is why the second rule reads the parity. And a symmetric
    ``fd.Region`` already differs from its full box by +4.08e-03, the snap's own half-cell
    rule (node components are painted at a Region's low face and not its high one), so the
    Region's asymmetric crossing costs about twice what snapping does; it is refused for its
    geometry, a body a third longer than the one declared, not for that number.

    A curved body -- ``fd.Sphere``, ``fd.Cylinder``, ``fd.Prism``, a turned ``fd.Box`` --
    that passes this door and still reaches the plane is refused by the constitutive plan,
    whose cross terms there would need a mirror-weighted adjoint that is not written.
    """
    from wavebit.fdtd._legacy_boundaries import Symmetry
    structures = tuple(getattr(scene, 'structures', ()) or ())
    mirrors = []
    for a, axis in enumerate('xyz'):
        for side in '-+':
            face = scene.boundaries.face(axis + side)
            if isinstance(face, Symmetry):
                mirrors.append((a, side, face.parity))
    if not mirrors or not structures:
        return
    for index, structure in enumerate(structures):
        shape = structure.shape
        body = shape.body()
        where = f'structures[{index}] ({shape.describe()})'
        for a, side, parity in mirrors:
            axis = 'xyz'[a]
            plane = shown(mirror_plane(grid, a, side), grid.dx)
            wall = shown(grid.low[a] + (grid.n[a] if side == '+' else 0) * grid.dx, grid.dx)
            plus = (f' On a + face the plane is half a cell INSIDE the wall: {axis} = '
                    f'{plane:g}, not the wall at {axis} = {wall:g}.' if side == '+' else '')
            if body is None:
                found = _snapped_mirror(shape, grid, a, side)
                if found is None:
                    continue
                if found[1] is None:
                    lo, hi = found[0]
                    raise ValueError(
                        f'{where} has its {axis} face declared on the {axis}+ Symmetry plane at '
                        f'{axis} = {plane:g} (declared {axis} = ({lo:g}, {hi:g})). On a + face '
                        f'the plane is the centre of the last cell, so a cell-snapped face on it '
                        f'rounds up or down by the cell count\'s parity and by round-off: the '
                        f'plane\'s cell full, the body joined to its image, or empty, the two a '
                        f'cell apart -- and the scene says neither. Declare it to the wall '
                        f'({axis} = {wall:g}) to join its image, or at least half a cell '
                        f'({0.5 * grid.dx:g}) short of the plane to keep them apart.')
                (lo, hi), (r0, r1), gap = found
                raise ValueError(
                    f'{where} crosses the {axis}{side} Symmetry plane at {axis} = {plane:g}: '
                    f'declared {axis} = ({lo:g}, {hi:g}), it runs as ({r0:g}, {r1:g}) -- its '
                    f'snapped domain part and that part\'s mirror image, which is what a '
                    f'Symmetry face models -- and the edge the mirror makes is {gap:.2f} cells '
                    f'from the declared one, where the snap alone moves an edge by at most '
                    f'half a cell. Measured with this door held open '
                    f'(studies/fdtd/symmetry_mirror/fdtd_symmetry_mirror.py), an fd.Region declared (-0.1, 0.2) '
                    f'across a mirror at 0 runs as (-0.2, 0.2) and moves a cavity\'s resonance '
                    f'by -9.09e-03 against the same Region in the full cavity. Centre it '
                    f'on the plane, or keep it on the domain side: flush against a - plane it '
                    f'is exact, the mirror joining it to its image; on a + face declare it to '
                    f'the wall for that.{plus}')
            found = _smoothed_mirror(body, shape, grid, a, side, parity)
            if found is None:
                continue
            kind, component, point, value, mirrored = found
            at = '(' + ', '.join(f'{v:.6g}' for v in point) + ')'
            if kind == 'crosses':
                raise ValueError(
                    f'{where} crosses the {axis}{side} Symmetry plane at {axis} = {plane:g} and '
                    f'is not its own mirror image there: the {component} voxel at {at} holds '
                    f'{value:.4f} of it and the mirrored voxel {mirrored:.4f}. A Symmetry face '
                    f'stands for a scene that is its own mirror image; the run carries the '
                    f'domain side and reflects it, so the body it models is this one\'s domain '
                    f'part and that part\'s image -- not the one declared. Measured with this '
                    f'door held open (studies/fdtd/symmetry_mirror/fdtd_symmetry_mirror.py), an fd.Box declared '
                    f'(-0.1, 0.2) across a mirror at 0 moves a cavity\'s resonance by '
                    f'-1.55e-02 against the same box in the full cavity, and a box centred on '
                    f'a + face\'s wall instead of its plane by +6.91e-03. Centre it on the '
                    f'plane ({axis} = {plane:g}), or keep it wholly on the domain side and let '
                    f'the mirror add its image. Only the unturned fd.Box may cross a mirror in '
                    f'stage 4: a body with constitutive cross terms is refused at the plane '
                    f'anyway, because its mirror-weighted adjoint is not written.{plus}')
            raise ValueError(
                f'{where} reaches within half a cell of the {axis}{side} Symmetry plane at '
                f'{axis} = {plane:g} without crossing it. The plane cuts the {component} voxel '
                f'at {at} in half: the body fills {value:.4f} of it on the domain side and its '
                f'image as much on the other, but the fill read there is the body\'s alone, '
                f'and at parity={parity:+d} the mirror leaves that component free. Measured '
                f'with this door held open (studies/fdtd/symmetry_mirror/fdtd_symmetry_mirror.py), an fd.Box '
                f'declared flush against a parity=+1 mirror as (0, 0.2) lowers eps '
                f'2.5 on the plane where the scene holds 4, and '
                f'moves a cavity\'s resonance by +5.50e-03. Declare the whole body, '
                f'symmetric about the plane -- the unturned fd.Box may cross a mirror -- or '
                f'keep its surface at least half a cell ({0.5 * grid.dx:g}) clear of the plane '
                f'(a whole cell for a body with constitutive cross terms). At '
                f'parity={-parity:+d} the same body is exact here, because the mirror zeroes '
                f'that component on the plane.{plus}')


def _snapped_mirror(shape, grid, a, side):
    """``((lo, hi), (r0, r1), gap)`` where a snapped shape crosses a mirror unmirrored, or None.

    ``(lo, hi)`` is the span declared along ``a``, ``(r0, r1)`` the one the run models, and
    ``gap`` the distance in cells between the declared and the modelled edge beyond the plane.
    ``((lo, hi), None, None)`` for a face declared on a ``+`` plane, which is ambiguous
    (:func:`check_symmetry`).
    Only ``fd.Region`` and an ``fd.Slab`` across ``a`` have an edge there; a Slab along
    another axis runs through the mirror unchanged.
    """
    from wavebit.fdtd._legacy_geometry import Region
    from wavebit.fdtd._legacy_geometry import Slab
    if isinstance(shape, Region):
        lo, hi = shape.low[a], shape.high[a]
    elif isinstance(shape, Slab) and shape.axis == 'xyz'[a]:
        lo, hi = shape.span
    else:
        return None
    n, dx, low = grid.n[a], grid.dx, grid.low[a]
    alpha, beta = (lo - low) / dx, (hi - low) / dx
    k0, k1 = int(round(alpha)), int(round(beta))
    slack = 0.5 * (1.0 + 1e-9)
    if side == '-':
        if k0 >= 0:
            return None
        made = -min(k1, n)                  # the mirror of its domain part's far edge
        gap = max(alpha - made, 0.0) if k1 >= n else abs(alpha - made)
        modelled = (low + made * dx, low + min(k1, n) * dx)
    else:
        if abs(beta - (n - 0.5)) <= 1e-6 and k0 < n:
            return (lo, hi), None, None     # a face ON the plane: rounds either way
        if k1 <= n:
            return None
        made = 2 * n - 1 - max(k0, 0)       # the plane is the centre of cell n - 1
        gap = max(made - beta, 0.0) if k0 <= 0 else abs(beta - made)
        modelled = (low + max(k0, 0) * dx, low + made * dx)
    if gap <= slack:
        return None
    return (lo, hi), modelled, gap


def _exact_fill(body, points, grid):
    """The body's exact fill at ``points`` ``(k, 3)``, paying for the clip only near its surface.

    A voxel whose point is more than :data:`_VOXEL_REACH` cells from the surface holds none of
    it or all of it, by the signed distance's sign (negative inside; a composed distance
    under-estimates, so the proof holds for those too).
    """
    from wavebit.fdtd._shapes import occupancy as body_occupancy
    d = np.asarray(body.surface(points)[0], dtype=float)
    out = np.where(d < 0, 1.0, 0.0)
    near = np.abs(d) <= _VOXEL_REACH * grid.dx
    if near.any():
        values, _ = body_occupancy(body, points[near], grid.dx)
        out[near] = np.asarray(values, dtype=float)
    return out


def _smoothed_mirror(body, shape, grid, a, side, parity):
    """``(kind, component, point, fill, mirrored)`` where a smoothed body is not what a mirror
    carries, or ``None``; ``kind`` is ``'crosses'`` or ``'flush'`` (see :func:`check_symmetry`).
    """
    from wavebit.fdtd._kernel.grid import OFFSETS
    from wavebit.fdtd._prepare.occupancy import yee_points
    n, dx = grid.n[a], grid.dx
    plane = mirror_plane(grid, a, side)
    box = shape.support(grid)
    wide = [(box[b][0] - 1, box[b][1] + 1) for b in range(3)]

    def points(layers, field, c):
        cells = list(wide)
        cells[a] = layers
        return np.asarray(yee_points(grid, tuple(cells), field, c)).reshape(-1, 3)

    # Material past the plane: the layer beyond it, whose voxels abut the plane on the far
    # side for one of the two offsets (the half-offset E/H at a - face, the node ones at +).
    beyond = (-1, 0) if side == '-' else (n, n + 1)
    crosses = max(float(np.max(_exact_fill(body, points(beyond, f, c), grid)))
                  for f in 'EH' for c in range(3))
    if crosses > _CLIP_ROUNDOFF:
        if hasattr(shape, 'low') and hasattr(shape, 'high'):
            depth = plane - shape.low[a] if side == '-' else shape.high[a] - plane
            layers = min(n, int(math.ceil(depth / dx)) + 2)
        else:
            layers = n
        band = (0, layers) if side == '-' else (n - layers, n)
        worst = (0.0, None)
        for field in 'EH':
            for c in range(3):
                q = points(band, field, c)
                m = q.copy()
                m[:, a] = 2.0 * plane - q[:, a]
                fq, fm = _exact_fill(body, q, grid), _exact_fill(body, m, grid)
                k = int(np.argmax(np.abs(fq - fm)))
                if abs(fq[k] - fm[k]) > worst[0]:
                    worst = (abs(fq[k] - fm[k]),
                             ('crosses', field + 'xyz'[c], tuple(q[k]), float(fq[k]),
                              float(fm[k])))
        return worst[1] if worst[0] > _CLIP_ROUNDOFF else None
    if parity != (1 if side == '-' else -1):
        return None                         # the mirror zeroes what sits on the plane
    on_plane = 0 if side == '-' else 1
    layer = (0, 1) if side == '-' else (n - 1, n)
    worst = (0.0, None)
    for c in range(3):
        if OFFSETS['E'][c][a] != on_plane:
            continue
        q = points(layer, 'E', c)
        f = _exact_fill(body, q, grid)
        k = int(np.argmax(f))
        if f[k] > worst[0]:
            worst = (float(f[k]), ('flush', 'E' + 'xyz'[c], tuple(q[k]), float(f[k]), None))
    return worst[1] if worst[0] > _CLIP_ROUNDOFF else None


def _declared_span(shape, a):
    """``(lo, hi)`` a shape declares along axis ``a``, or ``None`` where it runs the whole box.

    ``fd.Region`` and the smoothed bodies declare all three; an ``fd.Slab`` only its own axis.
    A shape without ``low``/``high`` (a CSG node, a protocol test's shape) declares none here.
    """
    from wavebit.fdtd._legacy_geometry import Region
    from wavebit.fdtd._legacy_geometry import Slab
    if isinstance(shape, Slab):
        return shape.span if shape.axis == 'xyz'[a] else None
    if isinstance(shape, Region) or shape.body() is not None:
        low, high = getattr(shape, 'low', None), getattr(shape, 'high', None)
        if low is not None and high is not None:
            return low[a], high[a]
    return None


def check_seams(scene, grid):
    """A cell-snapped shape across a periodic seam must span the period (stage 4, C5).

    ``fd.Region`` and ``fd.Slab`` are clipped to the cell, not wrapped: a shape that crosses
    one seam of a ``Periodic`` or ``Bloch`` axis without spanning the period loses the part past
    it, and its periodic image never appears at the other end of the cell. Measured in the C5
    audit, an ``fd.Region`` half across the seam ran with 50% of its cells and no warning. The
    one clipping that is right there is a shape that spans the period, which is how a lamellar
    grating spells "runs the whole period" and is not refused; a smoothed body at a seam is
    ``check_smoothed``'s door 2. Judged on the snapped cells, unclipped: crossing is
    ``k0 < 0`` or ``k1 > n``, spanning is ``k0 <= 0`` and ``k1 >= n``.
    """
    from wavebit.fdtd._legacy_boundaries import Bloch
    from wavebit.fdtd._legacy_boundaries import Periodic
    structures = tuple(getattr(scene, 'structures', ()) or ())
    for index, structure in enumerate(structures):
        shape = structure.shape
        if shape.body() is not None:
            continue
        for a, axis in enumerate('xyz'):
            face = scene.boundaries.face(axis + '-')
            if not isinstance(face, (Periodic, Bloch)):
                continue
            span = _declared_span(shape, a)
            if span is None:
                continue
            n = grid.n[a]
            k0 = int(round((span[0] - grid.low[a]) / grid.dx))
            k1 = int(round((span[1] - grid.low[a]) / grid.dx))
            if (k0 < 0 or k1 > n) and not (k0 <= 0 and k1 >= n):
                lost = (max(0, -k0) + max(0, k1 - n)) / (k1 - k0)
                raise ValueError(
                    f'structures[{index}] ({shape.describe()}) crosses the '
                    f'{type(face).__name__} {axis} seam without spanning the period: its '
                    f'snapped cells are {k0}..{k1} along {axis}, of {n}. A cell-snapped shape is '
                    f'clipped to the cell, not wrapped, so the part past the seam -- '
                    f'{100 * lost:.1f}% of its cells -- is not in the run, and its periodic '
                    f'image does not appear at the other end of the cell. Say the two pieces as '
                    f'two structures, one at each end of the cell; or move Domain(center=...) '
                    f'by whole cells so the shape lies inside the cell; or, if it is meant to '
                    f'run the whole period, let it span the cell -- the lamellar spelling, which '
                    f'is clipped exactly. (A smoothed body at a seam is refused the same way, by '
                    f'check_smoothed.)')


def _modelled_ends(scene, grid, a, lo, hi):
    """``(side, past, face)`` for each end of the span the run models along ``a``.

    The box, except across a Symmetry face, where the scene is the domain and its image: the
    end there is the mirror image of the opposite face, and a structure past it is cut by that
    face as surely as by the face itself. Measured before this (the C5 review): a box running
    past a PEC wall across a mirror warned 20.0% where 40% of it is not in the run.
    """
    from wavebit.fdtd._legacy_boundaries import Symmetry
    axis = 'xyz'[a]
    out = []
    for side in '-+':
        other = '+' if side == '-' else '-'
        if isinstance(scene.boundaries.face(axis + side), Symmetry):
            plane = mirror_plane(grid, a, side)
            wall = grid.high[a] if side == '-' else grid.low[a]
            end = 2.0 * plane - wall
            past = (end - lo) if side == '-' else (hi - end)
            out.append((side, past, other))
        else:
            past = (grid.low[a] - lo) if side == '-' else (hi - grid.high[a])
            out.append((side, past, side))
    return out


def _clipping_notes(scene, plan):
    """A ``GeometryClippedWarning`` per structure losing declared volume to a PEC or PMC face.

    The share lost is the length past the wall over the declared extent, multiplied over the
    axes -- exact, because what reaches a wall and is planned has a constant section across
    it: a cell-snapped box or layer, or a smoothed body uniform across the face
    (``check_smoothed``'s door 3 refuses any other). Only PEC and PMC faces are walls here: a
    structure through an absorbing face is how a substrate runs on out of the box, one that
    spans a periodic axis is a lamellar grating, and a Symmetry plane is ``check_symmetry``'s.
    Measured in the C5 audit, all silent before this note: an ``fd.Region`` of
    ``(0.4, 0.4, 0.2)`` at ``x = 0.4`` in a box 1.0 wide lost 25.0%, an ``fd.Slab``
    ``x=(0.2, 0.9)`` 57.1%, a smoothed ``fd.Box`` at the Region's place and a cylinder along
    x out through the face 25% each.
    """
    from wavebit.fdtd._legacy_boundaries import PEC
    from wavebit.fdtd._legacy_boundaries import PMC
    from wavebit.fdtd.diagnostics import GeometryClippedWarning
    grid = plan.grid
    out = []
    for index, structure in enumerate(tuple(getattr(scene, 'structures', ()) or ())):
        kept, faces = 1.0, []
        for a, axis in enumerate('xyz'):
            span = _declared_span(structure.shape, a)
            if span is None:
                continue
            lo, hi = span
            cut = 0.0
            for side, past, end in _modelled_ends(scene, grid, a, lo, hi):
                face = scene.boundaries.face(axis + end)
                if isinstance(face, (PEC, PMC)) and past > 1e-9 * grid.dx:
                    cut += past
                    faces.append(f"'{axis}{end}' ({type(face).__name__})"
                                 + ('' if end == side else f' mirrored in {axis}{side}'))
            kept *= max(hi - lo - cut, 0.0) / (hi - lo)
        if faces:
            out.append((GeometryClippedWarning,
                        f'structures[{index}] ({structure.shape.describe()}) loses '
                        f'{100 * (1 - kept):.1f}% of its declared volume to the '
                        f'{" and ".join(faces)} face{"s" if len(faces) > 1 else ""}: the part '
                        f'past a PEC or PMC wall is not in the run. A structure meant to run on '
                        f'out of the box needs fd.PML or fd.Absorber on that face, and one meant '
                        f'to fill a periodic axis spans it; neither warns.'))
    return out


#: Cells across a smoothed body's smallest bounded extent below which it is refused, and below
#: which it warns. Set by study S2b (``studies/fdtd/subpixel_cells_across/fdtd_subpixel_cells_across.py``,
#: ``docs/validation/fdtd/fd-subpixel-occupancy.md``): at 2 cells an eps-12 sphere's polarizability is 9.6% to
#: 43% high over five placements and it scatters 123% too much; at 4 every body tried at eps 12
#: is within 9.2%, and the sphere at eps 4 within 2.90%; at 8, every body within 2.63%.
CELLS_ACROSS_REFUSED, CELLS_ACROSS_WARNED = 4, 8

#: Relative slack on those thresholds. A body declared exactly 4 cells across can read
#: 3.9999999999999996 through ``center +- size/2`` or a hull width (the C5 review refused a
#: 0.2 box at some centres and not others, and S2b's own 4-cell hexagonal prism), and it is 4.
_ACROSS_SLACK = 1e-9


def _straight(shape, a):
    """Whether ``shape``'s section is the same all along axis ``a``: an unturned ``fd.Box``
    on any axis, an extruded body (``fd.Box``, ``fd.Cylinder``, ``fd.Prism``) along its own."""
    from wavebit.fdtd._legacy_geometry import Box
    from wavebit.fdtd._legacy_geometry import _Extruded
    if isinstance(shape, Box) and not shape.angle:
        return True
    return isinstance(shape, _Extruded) and shape.axis == 'xyz'[a]


def _covers(shape, a, grid, mirrors):
    """Whether the body runs the whole domain along axis ``a``, as one of a layer's two axes.

    ``uniform_along`` over the whole axis, or -- for a body straight along ``a`` -- a declared
    span reaching both ends, where an end is a wall or a Symmetry plane (the image continues
    the body past it). ``uniform_along`` asks a cell of margin past each face, which the
    nearest-surface normal needs, so a film flush against a PEC wall or a mirror read as
    bounded and was refused at its thickness (the C5 review).
    """
    if shape.uniform_along(a, (0, grid.n[a]), grid):
        return True
    span = _declared_span(shape, a)
    if span is None or not _straight(shape, a):
        return False
    ends = [grid.low[a], grid.high[a]]
    for axis, side in mirrors:
        if axis == a:
            ends[0 if side == '-' else 1] = mirror_plane(grid, a, side)
    tol = 1e-9 * grid.dx
    return span[0] <= ends[0] + tol and span[1] >= ends[1] - tol


def _extended(grid, mirrors):
    """The box grown past each Symmetry face to the image of the domain.

    Past a ``-`` plane (on the wall) the image is ``n`` cells; past a ``+`` plane, which sits
    ``dx/2`` inside the wall, it reaches ``n - 1`` cells beyond the wall -- a whole ``n`` made a
    film symmetric about a ``+`` plane reach a cell further than its image before it read as
    spanning (the C5 review).
    """
    from wavebit.fdtd._kernel.grid import Grid
    for a, side in mirrors:
        n, low = list(grid.n), list(grid.low)
        if side == '-':
            low[a] -= grid.n[a] * grid.dx
            n[a] += grid.n[a]
        else:
            n[a] += grid.n[a] - 1
        grid = Grid(n=tuple(n), dx=grid.dx, low=tuple(low))
    return grid


def _whole_with_image(shape, grid, mirrors, ext, extent):
    """An unturned ``fd.Box`` flush against a Symmetry plane, measured with its image.

    At the parity that zeroes the plane's components ``check_symmetry`` accepts a box declared
    only on the domain side, flush against the plane, as exact: the scene is it and its image,
    joined. Measured as declared it was half that, and was refused as half-size (the C5
    review). The box is the one smoothed body this reaches -- any other flush body plans cross
    terms at the plane, which ``constitutive.validate`` refuses.
    """
    from wavebit.fdtd._legacy_geometry import Box
    if not mirrors or not (isinstance(shape, Box) and not shape.angle):
        return extent
    tol = 1e-9 * grid.dx
    lengths = []
    for b in range(3):
        if _covers(shape, b, grid, mirrors):
            continue
        lo, hi = shape.low[b], shape.high[b]
        length = min(hi, ext.high[b]) - max(lo, ext.low[b])
        for axis, side in mirrors:
            plane = mirror_plane(grid, b, side) if axis == b else None
            if plane is not None and side == '-' and abs(lo - plane) <= tol:
                length = 2.0 * (min(hi, grid.high[b]) - plane)
            elif plane is not None and side == '+' and abs(hi - plane) <= tol:
                length = 2.0 * (plane - max(lo, grid.low[b]))
        lengths.append(length)
    return min(lengths) if lengths else extent


def _cells_across(shape, grid, mirrors=()):
    """``(cells, extent)`` across a smoothed body's smallest bounded extent, or ``None``.

    ``None`` for a **layer** -- a body running the whole domain along two axes, a film or a half
    space (:func:`_covers`). Its interfaces are planes, and for a planar interface the rule's
    two means -- arithmetic along it, harmonic across it -- are the layer's exact effective
    medium at any thickness; G5 (``tests/fdtd/test_fd_airy.py``) measured the lowering against the
    scheme's own recurrence on slabs 4 to 10 cells thick, to 3.2e-6, and no thinner film has
    been run. What S2b measured is a particle's polarizability, which a body bounded along two
    or three axes has; a rod through the domain is a two-dimensional particle, judged by its
    section.

    ``mirrors`` are the ``(axis index, side)`` of the scene's ``Symmetry`` faces. The body is
    measured on the box grown to the image of the domain (:func:`_extended`), because a body
    crossing a mirror is the body declared, not the half the box holds: measured on the box, a
    bar 7.4 cells long symmetric about the plane read 3.7 and was refused. An unturned box
    flush against the plane is measured with its image (:func:`_whole_with_image`); another
    body declared only on the domain side is measured as declared, half the scene's body, and
    its refusal says to declare it whole.
    """
    bounded = [a for a in range(3) if not _covers(shape, a, grid, mirrors)]
    if len(bounded) < 2:
        return None
    ext = _extended(grid, mirrors)
    extent = shape.min_extent(ext)
    if extent is None:
        return None
    extent = _whole_with_image(shape, grid, mirrors, ext, extent)
    return extent / grid.dx, extent


def check_cells_across(scene, grid):
    """A smoothed body under four cells across its smallest extent is refused (stage 4, C5).

    A body a few cells across is a handful of mixed voxels, and the grid scatters from the
    effective polarizability of those voxels, which depends on where the grid falls -- not
    from the body's. Study S2b measured it (the numbers are :data:`CELLS_ACROSS_REFUSED`'s),
    and a public ``fd.run`` of an eps-12 sphere 4 cells across scatters 13.2% too much centred
    on a node and 1.3% centred in a cell. The fill cannot see this: every public body's fill is
    its exact clip at any size. The door reads ``Shape.min_extent`` over the axes the body is
    bounded along and exempts a layer (:func:`_cells_across`); between 4 and 8 cells it warns
    (:func:`_cells_across_notes`). It is about accuracy and holds at any contrast: the
    star lift that runs after it (``stars.lift``) would make such a body's operator
    positive definite, and says nothing about whether its answer is right.
    """
    from wavebit.fdtd._prepare.occupancy import smoothed_structures
    mirrors = _mirror_faces(scene)
    for index, structure, _ in smoothed_structures(scene):
        found = _cells_across(structure.shape, grid, mirrors)
        if found is None or found[0] >= CELLS_ACROSS_REFUSED * (1.0 - _ACROSS_SLACK):
            continue
        cells, extent = found
        mirror = (' At a Symmetry face a body declared only on the domain side is measured as '
                  'declared; declared whole, symmetric about the plane, it is measured whole.'
                  if mirrors else '')
        raise ValueError(
            f'structures[{index}] ({structure.shape.describe()}) is {cells:.2f} cells across '
            f'its smallest extent ({extent:g} at dx = {grid.dx:g}), under the '
            f'{CELLS_ACROSS_REFUSED} a smoothed body needs. Across so few cells what the grid '
            f'scatters from is set by where the mixed voxels fall, not by the body: measured '
            f'in study S2b (studies/fdtd/subpixel_cells_across/fdtd_subpixel_cells_across.py) on compact bodies, an '
            f'eps-12 sphere 2 cells across has a polarizability 9.6% to 43% high over five '
            f'placements and scatters 123% too much against Mie theory, and at eps 2.25 it is '
            f'still up to 9% off; at 4 cells across every body tried at eps 12 is within 9.2%, '
            f'and the sphere at eps 4 within 3.0%. Refine to dx <= '
            f'{extent / CELLS_ACROSS_REFUSED:g} for 4 cells across, or dx <= '
            f'{extent / CELLS_ACROSS_WARNED:g} for 8, where every body tried is within 2.7%. '
            f'S2b\'s bodies were compact (aspect at most 1.6): a plate, thin along one axis and '
            f'wide along two, was not measured, and is judged by its thickness on these '
            f'numbers. A layer -- a body running the whole domain along two axes, like a film '
            f'-- is exempt: for a plane the rule is the exact effective medium.{mirror}')


def _mirror_faces(scene):
    """``(axis index, side)`` of every ``Symmetry`` face of ``scene``."""
    from wavebit.fdtd._legacy_boundaries import Symmetry
    return tuple((a, side) for a, axis in enumerate('xyz') for side in '-+'
                 if isinstance(scene.boundaries.face(axis + side), Symmetry))


def _cells_across_notes(scene, plan):
    """A ``CoarseMeshWarning`` per smoothed body 4 to 8 cells across (:func:`check_cells_across`)."""
    from wavebit.fdtd._prepare.occupancy import smoothed_structures
    out = []
    mirrors = _mirror_faces(scene)
    for index, structure, _ in smoothed_structures(scene):
        found = _cells_across(structure.shape, plan.grid, mirrors)
        if found is None or not found[0] < CELLS_ACROSS_WARNED * (1.0 - _ACROSS_SLACK):
            continue
        cells, extent = found
        out.append((CoarseMeshWarning,
                    f'structures[{index}] ({structure.shape.describe()}) is {cells:.1f} cells '
                    f'across its smallest extent ({extent:g}); between {CELLS_ACROSS_REFUSED} '
                    f'and {CELLS_ACROSS_WARNED}, study S2b measured an eps-12 body\'s '
                    f'polarizability within 9.2% -- a sphere\'s between +1.2% and +6.3% by '
                    f'where the grid falls -- against 2.7% for every body tried at '
                    f'{CELLS_ACROSS_WARNED}. dx <= {extent / CELLS_ACROSS_WARNED:g} reaches '
                    f'{CELLS_ACROSS_WARNED}.'))
    return out


#: One smoothed structure as ``fd.check`` prints it (stage 4, C5). Lengths in the scene's unit.
#: ``across`` is ``(cells, extent)`` or ``None`` for a layer; ``asked`` the analytic volume or
#: ``None`` for a body that does not answer ``volume()``; ``inside`` the part of it the box
#: holds; ``outside`` the ends of the modelled scene its declared extent passes (a wall, or a
#: wall's image across a mirror -- a mirror itself is no end) and ``share`` the part past them;
#: ``held``, ``cover`` and ``cut`` per lowered component (``'Ex'`` ...): the fill summed times
#: ``dx^3``, the body's volume inside that component's own voxels (a node component's reach
#: half a cell past a minus face and stop half a cell short of a plus one), and the voxels it
#: cuts; ``terms`` the constitutive terms its cross coefficients live in and ``shared`` the
#: other structures in each; ``stencil`` the prepared stencil terms it is in; ``normals``
#: ``'exact'`` (the sphere's moment), ``'none'`` (no cross term) or the measured proxy gap;
#: ``mirrors`` the ``(face, plane)`` pairs of the Symmetry faces.
BodyRow = collections.namedtuple(
    'BodyRow', 'index describe support across asked inside outside share held cover cut '
               'terms shared stencil normals mirrors')


def _body_rows(scene, plan, occupied, proxy):
    """The :data:`BodyRow` of every smoothed structure, from fills ``_smoothed_notes`` has.

    No geometry is evaluated here: the fills are the ones the notes already took, which is
    what keeps the block from costing a fifth occupancy pass (the C5 audit counted four).
    """
    from wavebit.fdtd._legacy_boundaries import Symmetry
    from wavebit.fdtd._kernel.grid import OFFSETS
    from wavebit.fdtd._kernel.stencil import StencilTerm
    from wavebit.fdtd._shapes import Sphere as KernelSphere
    from wavebit.fdtd._prepare.constitutive import cross_free
    from wavebit.fdtd._prepare.occupancy import smoothed_structures
    grid = plan.grid
    tol = 1e-9 * grid.dx
    mirrors = tuple((axis + side, mirror_plane(grid, a, side))
                    for a, axis in enumerate('xyz') for side in '-+'
                    if isinstance(scene.boundaries.face(axis + side), Symmetry))
    smoothed = smoothed_structures(scene)
    crossed = {i: s.shape.support(grid) for i, s, b in smoothed if not cross_free(b)}
    rows = []
    for index, structure, body in smoothed:
        shape = structure.shape
        support = shape.support(grid)
        volume = getattr(shape, 'volume', None)
        asked = volume() if callable(volume) else None
        spans = {a: _declared_span(shape, a) for a in range(3)}
        spans = {a: s for a, s in spans.items() if s is not None}
        in_box, kept, outside = 1.0, 1.0, []
        for a, (lo, hi) in spans.items():
            axis = 'xyz'[a]
            in_box *= max(min(hi, grid.high[a]) - max(lo, grid.low[a]), 0.0) / (hi - lo)
            cut_off = 0.0
            for side, past, end in _modelled_ends(scene, grid, a, lo, hi):
                if past > tol:
                    cut_off += past
                    outside.append(axis + end + ('' if end == side else
                                                 f' (mirrored in {axis}{side})'))
            kept *= max(hi - lo - cut_off, 0.0) / (hi - lo)
        held, cover, cut = {}, {}, {}
        for field in ('E', 'H'):
            for c, (fill, _) in enumerate(occupied.get(index, {}).get(field, ())):
                fill = np.asarray(fill, dtype=float)
                name = field + 'xyz'[c]
                held[name] = float(fill.sum()) * grid.dx ** 3
                cut[name] = int(np.count_nonzero((fill > _CLIP_ROUNDOFF)
                                                 & (fill < 1.0 - _CLIP_ROUNDOFF)))
                if asked is not None:
                    share = 1.0
                    for a, (lo, hi) in spans.items():
                        shift = (0.5 * OFFSETS[field][c][a] - 0.5) * grid.dx
                        reach = (grid.low[a] + shift, grid.high[a] + shift)
                        share *= max(min(hi, reach[1]) - max(lo, reach[0]), 0.0) / (hi - lo)
                    cover[name] = asked * share
        stencil = tuple(t.name for t in plan.constitutive if isinstance(t, StencilTerm)
                        and _intersects(tuple(reversed(t.box)), support))
        if cross_free(body):
            terms, normals = (), 'none'
        else:
            terms = tuple(t.name for t in plan.constitutive if not isinstance(t, StencilTerm)
                          and _intersects(tuple(reversed(t.box)), support))
            normals = 'exact' if isinstance(body, KernelSphere) else proxy.get(index, 0.0)
        shared = {name: tuple(j for j, box in crossed.items() if j != index and any(
            t.name == name and _intersects(tuple(reversed(t.box)), box)
            for t in plan.constitutive)) for name in terms}
        rows.append(BodyRow(index=index, describe=shape.describe(), support=support,
                            across=_cells_across(shape, grid, _mirror_faces(scene)),
                            asked=asked, inside=None if asked is None else asked * in_box,
                            outside=tuple(outside),
                            share=None if asked is None else 1.0 - kept,
                            held=held, cover=cover, cut=cut, terms=terms, shared=shared,
                            stencil=stencil, normals=normals, mirrors=mirrors))
    return rows


def check_absorbing_layers(scene, plan):
    """What may sit inside an absorbing layer: no interface, and (in a PML) no loss or memory.

    Two different rules, on two different sets of faces:

    * **A structure that cuts the layer along its own axis**, ending strictly inside it, is
      refused on **every** absorbing face -- ``PML`` and ``Absorber`` alike. What the user
      wanted there was a termination and what the grid gives them is a mirror: the interface
      reflects and only the part of the layer in *front* of it is in the way of that
      reflection. Measured in ``tests/fdtd/test_fd_absorber.py``, a lossless slab stopping half way
      into a twelve-cell layer reflects 1.9e-1 against 6.0e-4 running it through on a ``PML``
      face and 2.4e-1 against 1.2e-2 on an ``Absorber`` face -- 317x and 20x. (The
      ``Absorber`` half of this rule is not in the stage-1 design, which names the ``PML``
      case only; it is here because the measurement says the geometry is as bad on the face
      the ``PML`` refusal used to point at as an escape.)
    * **A material with loss or memory** -- a :class:`~wavebit.fdtd._legacy.Conductor` or a
      :class:`~wavebit.fdtd._legacy.Dispersive` -- is refused in a ``PML`` face only, naming
      ``Absorber``, which needs no match. See :func:`_refuse_lossy_in_layer` for what that
      refusal is and is not measured to protect against.

    A structure that runs *across* the layer -- a z slab inside an x PML, which is what a
    slab in an open box always is -- is uniform along the absorbing axis and is allowed. The
    stretch is still matched to the face's one medium, so the part of the layer that is not
    that medium returns more than the table says: by the exact layer return at ``dx = 0.05``
    on 12 cells, 2.6x for glass in a vacuum-matched face at normal incidence (the ``PML``
    docstring's numbers). It is accepted and not warned about: ``_proximity_notes`` skips a
    structure that runs through a transverse face, which leaves it no clearance to keep.
    (The stage-1 design's table refuses this case outright. It is relaxed here because the
    canonical stage-1 scene -- a dielectric slab with ``Boundaries.open(...)`` -- is exactly
    it, and refusing the canonical scene would mean nothing could use the feature.)
    """
    grid = plan.grid
    pml = dict(plan.pml_cells)
    for name, cells in plan.absorbing_faces:
        kind = 'PML' if name in pml else 'Absorber'
        a, side = 'xyz'.index(name[0]), name[1]
        layer = (0, cells) if side == '-' else (grid.n[a] - cells, grid.n[a])
        for index, structure in enumerate(getattr(scene, 'structures', ()) or ()):
            k0, k1 = structure.shape.support(grid)[a]
            if k1 <= layer[0] or k0 >= layer[1]:
                continue                          # its support cannot reach the layer
            if structure.shape.uniform_along(a, layer, grid):
                # Runs across this layer and is uniform along it (a slab on another axis,
                # or a box that reaches through the face): no interface inside the layer,
                # but its material is in there.  The question is the shape's, not its
                # bounding box's: a box spanning the layer says nothing about a shape whose
                # surface curves through it, and the measured cost of that confusion is the
                # 317x in this function's own docstring.
                if kind == 'PML':
                    _refuse_lossy_in_layer(structure.material, name, f'structures[{index}]')
                continue
            raise ValueError(
                f'structures[{index}] ({structure.shape.describe()}, cells {k0}..{k1} along '
                f'{name[0]}) ends inside the {kind} on {name!r} (cells '
                f'{layer[0]}..{layer[1]}). An absorbing layer terminates the box; an '
                f'interface inside one is a mirror the layer cannot undo, because the '
                f'reflection leaves the way it came and only the cells in front of the '
                f'interface are in its way. (In a PML the stretched coordinate is complex '
                f'there as well, so nothing matches it at all.) Move the interface out of '
                f'the layer, or run the structure all the way through it -- which today '
                f'the cell-snapped fd.Region or fd.Slab and an unturned fd.Box can, reaching '
                f'past the face. An fd.Cylinder or fd.Prism along {name[0]!r} is refused in a '
                f'layer by its cross terms until C9; a sphere cannot run through one.')
        if kind == 'PML':
            _refuse_lossy_in_layer(_background(scene), name, 'Scene(background=)')


def _background(scene):
    from wavebit.materials.time_domain import as_material
    return as_material(getattr(scene, 'background', None), 'Scene(background=)')


def _refuse_lossy_in_layer(material, name, where):
    """A ``Conductor`` or ``Dispersive`` reaching a PML face: refused, naming ``Absorber``.

    **What this door is measured to be worth, and what it is not.** A coordinate stretch is a
    change of variable, so it stays matched to a *uniform* medium whatever that medium's loss
    is; what a lossy medium costs the layer is the grading's calibration, which is normalised
    by ``sqrt(eps_r)`` of a background the medium is not. Measured with the door held open
    (``tests/fdtd/test_fd_absorber.py``, the test named for what this refusal turns away), the same
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
    """``(bound, reason)``: a conservative ``dt`` bound, or ``(None, reason)`` if unreadable.

    The vacuum bound is ``n_min dx / sqrt(3)`` -- the *smallest* index, because a medium with
    ``eps < 1`` is faster than vacuum and the usual spelling that drops ``n_min`` lets a
    perfectly ordinary ``eps = 0.81`` diverge. The poles tighten it further
    (``src/wavebit/fdtd/_periodic/grid.py``, a kernel seed):

        dt < 1 / sqrt(1/dt_vac^2 + rate/4),   rate = max(w0^2) + max(sum strength/eps)

    ``rate`` is a rate squared: the fastest oscillation the coupled E-P system supports,
    which is the pole resonance plus the plasma-like coupling back into E. A Debye pole
    contributes ``delta_eps/tau^2`` for the strength term (``materials.Debye`` derives why
    that is the only combination with the right units, and says it is a bound by analogy).

    **``n_min`` is ``sqrt(min eps * min mu)`` over every value the scene holds**, and not the
    smallest phase's own index. E and H sit at different points of the Yee cell, so wherever
    two media meet -- snapped or smoothed -- an E point can hold one phase's ``eps`` while the
    H points around it hold another's ``mu``, and the step is bounded by the norm of
    ``eps^-1/2 C^T mu^-1 C eps^-1/2``, which is at most ``max(1/eps) max(1/mu)`` times the curl's
    -- the bound below, rigorous for any arrangement of **diagonal** coefficients (a smoothed
    mix stays between its phases). A smoothed body's constitutive cross coefficients can push
    the inverse-material operator past ``max(1/eps)`` -- the C5 review measured 1.054 against
    1.0 for an ``eps = 12`` sphere -- and that case is ``constitutive.validate``'s and
    ``constitutive.guard``'s, whose Gershgorin enclosure tightens this bound further.
    The phase-by-phase index is not: ``Material(eps=4, mu=0.25)`` inside
    ``Material(eps=0.25, mu=4)`` has every phase at ``n = 1``, and as a snapped ``fd.Region``
    in that background at ``dx = 0.05`` it was measured, found in stage 4's C5 audit, to plan at
    any courant below 1 and to reach NaN within 3000 steps at courant 0.60 to 0.99, while 0.50
    and 0.55 stayed bounded over 20000. It is conservative, and measured so: that pair smoothed
    goes unstable between courant 0.84 and 0.86 against this bound's 0.25, and a box of it in
    vacuum is refused at the default 0.5 against a real limit near 1.0. No cheaper bound found
    is both safe and tighter -- pairing each E point with its own H neighbours gives ``n = 1``
    there, which is unsafe, and the extrema of the lowered arrays give 0.25 again, because
    pure cells of each phase exist. Where one component of one phase holds both the smallest
    ``eps`` and the smallest ``mu`` -- every scene with ``mu = 1`` throughout, and every scene
    with a vacuum phase and ``eps, mu >= 1`` elsewhere -- it is the phase-by-phase index, to
    the bit; elsewhere it is below it (``eps = 1, mu = 2`` beside ``eps = 2, mu = 1`` reads 1,
    not ``sqrt(2)``, and a background of ``eps = (1, 4, 4), mu = (4, 1, 1)`` alone reads 1, where
    its own index is 2 -- E along x sees ``eps = 1`` and the H beside it ``mu = 1``).
    """
    background, structures = _background(scene), tuple(getattr(scene, 'structures', ()) or ())
    materials = [background] + [s.material for s in structures]
    n_min, resonance, coupling = None, 0.0, 0.0
    from wavebit.fdtd._prepare.oscillator_sum_film import policy_rate as oscillator_policy_rate
    oscillator_rate = oscillator_policy_rate(scene, plan)
    from wavebit.fdtd._prepare.oscillator_stack import policy_rate as oscillator_stack_policy_rate
    oscillator_stack_rate = oscillator_stack_policy_rate(scene, plan)
    from wavebit.fdtd._prepare.hybrid_film import policy_rate as hybrid_policy_rate
    hybrid_rate = hybrid_policy_rate(scene, plan)
    from wavebit.fdtd._prepare.debye_film import policy_rate as debye_policy_rate
    relaxation_rate = debye_policy_rate(scene, plan)
    if relaxation_rate is None:
        from wavebit.fdtd._prepare.two_debye_film import policy_rate as two_debye_policy_rate
        relaxation_rate = two_debye_policy_rate(scene, plan)
    if relaxation_rate is None:
        from wavebit.fdtd._prepare.debye_sum_film import policy_rate as debye_sum_policy_rate
        relaxation_rate = debye_sum_policy_rate(scene, plan)
    from wavebit.fdtd._prepare.debye_stack import policy_rate as debye_stack_policy_rate
    debye_stack_rate = debye_stack_policy_rate(scene, plan)
    if relaxation_rate is None:
        relaxation_rate = debye_stack_rate
    for material in materials:
        n = material.index_bound()
        if n is None:
            return None, 'a material value is traced, so the bound cannot be read here'
        n_min = n if n_min is None else min(n_min, n)
        strength_sum = 0.0
        for pole in material.poles:
            if (relaxation_rate is not None or oscillator_rate is not None
                    or oscillator_stack_rate is not None or hybrid_rate is not None):
                # These scopes have one film or a separated stack in a plain host. Their
                # replacement rate includes every member without counting a memory twice.
                # Each scope owns its declared inactive-frequency policy.
                continue
            w0_squared, strength = pole.rate()
            w0_squared, strength = concrete(w0_squared), concrete(strength)
            if w0_squared is None or strength is None:
                return None, 'a pole value is traced, so the bound cannot be read here'
            resonance = max(resonance, w0_squared)
            strength_sum += strength
        eps_min = min(concrete(e) for e in material.eps)
        coupling = max(coupling, strength_sum / eps_min)
    # E and H sit at different points: the smallest eps and the smallest mu need not come
    # from one phase, snapped or smoothed (the docstring's operator-norm bound). Static, no
    # tolerance, and dt is never recomputed.
    eps_min = min(concrete(e) for material in materials for e in material.eps)
    mu_min = min(concrete(m) for material in materials for m in material.mu_value)
    n_min = min(n_min, math.sqrt(eps_min * mu_min))
    axes = cfl_axis_sum(scene, plan.dx)
    if axes == 3.0 / plan.dx ** 2:
        vacuum = n_min * plan.dx / math.sqrt(3.0)          # the stage-0 float, to the bit
    else:
        vacuum = n_min / math.sqrt(axes)                   # an Axis1D face (stage 3)
    from wavebit.fdtd._prepare.conductive_film import extra_rate as conductive_rate
    rate = resonance + coupling
    from wavebit.fdtd._prepare.conductive_stack import extra_rate as stack_rate
    film_rate = max(conductive_rate(scene, plan), stack_rate(scene, plan))
    if film_rate:
        rate += film_rate
    if relaxation_rate is not None:
        rate = max(rate, relaxation_rate)
    if oscillator_rate is not None:
        rate = max(rate, oscillator_rate)
    if oscillator_stack_rate is not None:
        rate = max(rate, oscillator_stack_rate)
    if hybrid_rate is not None:
        rate = max(rate, hybrid_rate)
    return 1.0 / math.sqrt(1.0 / vacuum ** 2 + rate / 4.0), None


def check_stability(scene, plan):
    """Refuse a concretely unstable ``dt``; return whether the bound could be read at all."""
    bound, reason = cfl_bound(scene, plan)
    if bound is None:
        return False
    if plan.dt < bound:
        return True
    from wavebit.fdtd._prepare.debye_film import policy_rate as debye_policy_rate
    relaxation_rate = debye_policy_rate(scene, plan)
    if relaxation_rate is None:
        from wavebit.fdtd._prepare.two_debye_film import policy_rate as two_debye_policy_rate
        relaxation_rate = two_debye_policy_rate(scene, plan)
    if relaxation_rate is None:
        from wavebit.fdtd._prepare.debye_sum_film import policy_rate as debye_sum_policy_rate
        relaxation_rate = debye_sum_policy_rate(scene, plan)
    from wavebit.fdtd._prepare.debye_stack import policy_rate as debye_stack_policy_rate
    debye_stack_rate = debye_stack_policy_rate(scene, plan)
    if relaxation_rate is None:
        relaxation_rate = debye_stack_rate
    from wavebit.fdtd._prepare.oscillator_sum_film import policy_rate as oscillator_policy_rate
    oscillator_rate = oscillator_policy_rate(scene, plan)
    from wavebit.fdtd._prepare.oscillator_stack import policy_rate as oscillator_stack_policy_rate
    oscillator_stack_rate = oscillator_stack_policy_rate(scene, plan)
    from wavebit.fdtd._prepare.hybrid_film import policy_rate as hybrid_policy_rate
    hybrid_rate = hybrid_policy_rate(scene, plan)
    poles = [(name, p) for name, p in _all_poles(scene)]
    worst = ''
    if oscillator_stack_rate is not None:
        worst = (' The bounded oscillator stack uses the maximum member resonance squared '
                 'plus its own strength divided by high-frequency permittivity, keeping '
                 'each sum on its member before taking the spatial maximum. The '
                 'singleton inactive Lorentz frequency contribution is retained.')
    elif debye_stack_rate is not None:
        worst = (' The bounded Debye stack uses the maximum of every member\'s bulk '
                 'and actual local summed relaxation rates; separated members are '
                 'not added across space. The curl bound uses physical high-frequency '
                 'permittivity, not the affine update shift.')
    elif hybrid_rate is not None:
        worst = (' The bounded hybrid film includes every active oscillator and a '
                 'conservative Debye penalty. Its curl bound uses the physical '
                 'high-frequency permittivity, not the affine update shift.')
    elif oscillator_rate is not None:
        worst = (' The bounded oscillator-sum film includes the maximum active bulk '
                 'resonance plus every active strength divided by the film permittivity; '
                 'this also bounds its coupled normal response.')
    elif poles and relaxation_rate is not None and len(poles) > 2:
        worst = (' The bounded Debye-sum film includes the combined bulk and '
                 'transformed local relaxation rates of every active pole.')
    elif poles and relaxation_rate is not None and len(poles) > 1:
        worst = (' The bounded two-Debye film includes the combined bulk and '
                 'transformed local relaxation rates of both poles.')
    elif poles:
        # The supported single-Debye film has one pole. Its scoped rate also avoids
        # evaluating an inactive extreme relaxation time in the diagnostic.
        name, pole = (poles[0] if relaxation_rate is not None else
                      max(poles, key=lambda np_: sum(abs(v) for v in _rate_values(np_[1]))))
        worst = (f' The tightest pole is {name}: {type(pole).__name__}'
                 f'({", ".join(f"{f}={getattr(pole, f)!r}" for f in pole.__dataclass_fields__)}).')
    from wavebit.fdtd._prepare.conductive_film import extra_rate as conductive_rate
    from wavebit.fdtd._prepare.conductive_stack import extra_rate as stack_rate
    if stack_rate(scene, plan):
        worst += (' The conductive stack includes the maximum induced interface '
                  'Debye rate across all members under the existing conservative policy.')
    if conductive_rate(scene, plan):
        worst += (' The conductive film carries an induced interface Debye pole; its '
                  'conservative rate is included in this bound.')
    if relaxation_rate is not None and debye_stack_rate is None:
        worst += (' The Debye film includes the maximum bulk and actual local '
                  'relaxation rate under the existing conservative Debye policy.')
    from wavebit.fdtd._legacy_boundaries import Axis1D
    axis1d = [f'{a} (k_max = {scene.boundaries.face(a + "-").k_max:g})' for a in 'xyz'
              if isinstance(scene.boundaries.face(a + '-'), Axis1D)]
    if axis1d:
        worst += (f' An fd.Axis1D axis adds k_max^2/4 to the sum under the bound in place of the 1/dx^2 a '
                  f'differenced axis adds (its derivative is i k, bounded by nothing but k_max): '
                  f'{", ".join(axis1d)}; a smaller k_max= widens the bound.')
    background = _background(scene)
    phases = [background] + [s.material for s in (getattr(scene, 'structures', ()) or ())]
    own = min(m.index_bound() for m in phases)
    eps_min = min(concrete(e) for m in phases for e in m.eps)
    mu_min = min(concrete(m) for p in phases for m in p.mu_value)
    if math.sqrt(eps_min * mu_min) < own * (1 - 1e-12):
        worst += (f' E and H sit at different points of the cell, so an E component is '
                  f'updated with the mu of the H components beside it, which can belong to '
                  f'another phase or to another axis of the same one: the bound pairs the '
                  f'smallest eps with the smallest mu, sqrt(min(eps) * min(mu)) = '
                  f'{math.sqrt(eps_min * mu_min):g}, below each phase\'s own index '
                  f'({own:g}). It is rigorous for the diagonal coefficients and conservative, '
                  f'and it does not change dt.')
    raise ValueError(
        f'dt = {plan.dt:g} is at or past the stability bound {bound:g} of this scene. The '
        f'vacuum condition dt < n_min dx / sqrt(3) is tightened by every pole to '
        f'dt < 1/sqrt(1/dt_vac^2 + rate/4) with rate = max(w0^2) + max(sum strength/eps), '
        f'because the coupled E-P system oscillates faster than the grid does.{worst} Lower '
        f'Run(courant=...) to {_courant_below(bound, plan.dx)} or less, or refine dx.')


def _courant_below(bound, dx):
    """A courant under ``bound`` to three places, which itself plans (the C5 review).

    Rounded to nearest, the limit printed as "at most 0.500" was the refused value itself when
    the bound fell on it, as it does for every snapped magnetic body with
    ``sqrt(min eps * min mu) = 0.5`` at the default courant.
    """
    limit = bound / (dx / math.sqrt(3.0))
    return f'{math.floor(limit * (1.0 - 1e-9) * 1000.0) / 1000.0:.3f}'


def _all_poles(scene):
    background = _background(scene)
    for pole in background.poles:
        yield 'Scene(background=)', pole
    for index, structure in enumerate(getattr(scene, 'structures', ()) or ()):
        for pole in structure.material.poles:
            yield f'structures[{index}]', pole


def _rate_values(pole):
    return [concrete(v) or 0.0 for v in pole.rate()]


def notes(scene, plan, rows=None, lifted=None):
    """``[(category, message), ...]``: the stage-1 warnings, for ``check`` and for ``plan``.

    ``rows``, when a list, receives one :data:`BodyRow` per smoothed structure (``check``'s
    block), from the fills the notes take anyway. ``lifted`` is the star lift by term
    name: ``plan`` passes ``constitutive.validate``'s; an empty dict is filled here, which is
    how ``check`` reads it (:func:`_lift_notes`).
    """
    out = []
    out += _resolution_notes(scene, plan)
    out += _proximity_notes(scene, plan)
    out += _stabilize_notes(scene, plan)
    from wavebit.fdtd._prepare.shape_ranges import traced
    from wavebit.fdtd._prepare.shape_ranges import name
    from wavebit.fdtd._prepare.shape_ranges import materials_traced
    from wavebit.fdtd._prepare.shape_ranges import seeded
    material = materials_traced(scene) and seeded(scene, plan)
    if traced(scene):
        from wavebit.fdtd.diagnostics import SubpixelAccuracyWarning
        out.append((SubpixelAccuracyWarning,
                    f'{name(scene)} geometry is traced: body volume, cut-cell counts, cells across '
                    'and star-lift measurements are unavailable in this report. Run the '
                    'concrete geometry to inspect them. The runtime checks the declared '
                    'range, four-cell minimum extent and constitutive stability; these '
                    'checks do not establish physical convergence.'))
        out += _traced_material_notes(material)
        return out
    if material:
        return out + _traced_material_notes(material)
    out += _smoothed_notes(scene, plan, rows)
    from wavebit.fdtd._prepare.junction import notes as junction_notes
    out += junction_notes(scene, plan)
    out += _lift_notes(scene, plan, lifted)
    out += _clipping_notes(scene, plan)
    out += _cells_across_notes(scene, plan)
    out += _radiation_notes(scene)
    return out


def _traced_material_notes(material):
    """The note for a traced material value beside smoothed coupling, or nothing.

    The smoothed notes lower the cross coefficients on NumPy, which a traced value cannot
    enter; the host planned the plan's own values (``shape_ranges._seeded_materials``), so
    nothing measured there is this run's.
    """
    if not material:
        return []
    from wavebit.fdtd.diagnostics import SubpixelAccuracyWarning
    return [(SubpixelAccuracyWarning,
             'A material value is traced: the cross coefficients, star-lift measurements and '
             'the cells per wavelength of the traced medium are unavailable in this report, '
             'and the plan was certified at the values it was built with. Run the concrete '
             'values to inspect them. The runtime checks positivity, the time-step bound, '
             'the lifted operator\'s stability, 8 cells per wavelength in a traced plain '
             'Material (at the band\'s shortest and every Diffraction wavelength) and the '
             'grazing rule in a PML layer at the traced values; these checks do not '
             'establish physical convergence.')]


def _lift_notes(scene, plan, lifted=None):
    """Say where the star lift changed the default rule's operator (``stars.lift``).

    A scene the Gershgorin certificate used to refuse now plans, with a modified operator; this
    is the note that it was modified, and how much. ``lifted`` as :func:`notes` takes it.
    """
    from wavebit.fdtd._kernel.stencil import StencilTerm
    from wavebit.fdtd.diagnostics import SubpixelAccuracyWarning
    if not any(not isinstance(t, StencilTerm) for t in plan.constitutive):
        return []
    if lifted is None:
        lifted = {}
    if not lifted:
        # The plan keeps validate's lift for the scene it was planned from (plan.plan), so a
        # report on that scene -- fd.run's, after every run -- does not lower it again.
        planned = getattr(plan, '_lifted', None)
        if planned is not None and planned[0] is scene:
            lifted.update(planned[1])
        else:
            from wavebit.fdtd._prepare.constitutive import lifted_on_host
            lifted.update(lifted_on_host(scene, plan)[1])
    moved = {name: out for name, out in lifted.items() if out.moved}
    if not moved:
        return []
    each = ', '.join(f'{name} {out.moved} of {out.stars}' for name, out in moved.items())
    largest = max(out.largest for out in moved.values())
    return [(SubpixelAccuracyWarning,
             f'The star lift moved the inverse-material diagonal at some nodes toward the '
             f'node\'s Kottke tensor ({each} stars; the largest move is {largest:.3g} of the '
             f'way) so that the smoothed operator is positive definite: at this contrast the '
             f'default rule falls below the floor there. The run is stable by construction '
             f'(src/wavebit/fdtd/_prepare/stars.py); the effect on accuracy is measured in '
             f'docs/validation/fdtd/fd-subpixel-occupancy.md, "The star lift".')]


def _radiation_notes(scene):
    """One scene-level note for the explicitly retained continuum approximation."""
    from wavebit.fdtd._legacy_monitors import FarField
    from wavebit.fdtd.diagnostics import RadiationApproximationWarning
    names = [m.name for m in scene.monitors if isinstance(m, FarField) and m.propagation == 'continuum']
    if not names:
        return []
    # Same unchanged epsilon=1.1 sphere records, R/lambda=.25, three grids.
    # These measured residuals explain the approximation; they are not a
    # tolerance or a predicted error for another geometry or polarization.
    return [(RadiationApproximationWarning,
        f'FarField {names} uses continuum propagation on discrete grid fields. '
        'A weak sphere (epsilon=1.1, radius/wavelength=0.25) measured optical-theorem '
        'residuals 52.05%/26.38%/12.91% at 20/28/40 cells per wavelength. '
        'These are examples, not an error bound for this scene. Use propagation="yee" '
        'for the validated lattice radiation map within its supported frequency band; '
        'keep continuum only when its approximation is intended. See docs/fd-lattice-radiation.md.')]


def _smoothed_notes(scene, plan, rows=None):
    """Read approximation bounds once for the static scene, outside numerical lowering.

    Use the same components the coefficients spend. This also serves ``fd.check`` when
    params already exist: a report must not depend on whether a byte count required lowering.
    The collector holds component-voxel counts, summed over structures and CSG nodes.
    """
    from wavebit.fdtd._runtime import runtime
    from wavebit.fdtd._shapes import Approximations
    from wavebit.fdtd.diagnostics import BoundedVoxelWarning
    from wavebit.fdtd.diagnostics import SubpixelAccuracyWarning
    from wavebit.fdtd._prepare.occupancy import bound_note
    from wavebit.fdtd._prepare.occupancy import occupancy
    from wavebit.fdtd._prepare.occupancy import smoothed_structures
    from wavebit.fdtd._prepare.stencil import STENCIL
    smoothed = smoothed_structures(scene)
    if not smoothed:
        return []
    asked = [index for index, structure, _ in smoothed if structure.discretisation == STENCIL]
    out = []
    from wavebit.fdtd._prepare.conductive_film import descriptor
    film = descriptor(scene, plan.grid)
    from wavebit.fdtd._prepare.conductive_stack import descriptor as stack_descriptor
    stack = stack_descriptor(scene, plan.grid)
    if stack is not None:
        out.append((SubpixelAccuracyWarning,
                    f'The conductive stack has {len(stack.films)} films normal to '
                    f'{"xyz"[stack.axis]}, each with independent induced Debye memory '
                    'on disjoint padded support. This bounded scope admits only concrete '
                    'parallel Conductor films in a plain lossless dielectric; touching '
                    'films, other pole families and junctions are outside it. Validate '
                    'observables under refinement and check duration and absorbers '
                    'separately. See docs/fd-conductive-stack.md.'))
    if film is not None:
        out.append((SubpixelAccuracyWarning,
                    'The conductive film uses an induced Debye memory at normal mixed '
                    'cells and arithmetic conductivity along its faces. This C8 scope is '
                    'one concrete axis-aligned film in a lossless dielectric; curved '
                    'conductors, declared dispersive poles and junctions are not supported. '
                    'An exact local material response alone does not establish spatial '
                    'convergence. Validate the observables under refinement and check run '
                    'duration separately. See docs/fd-conductive-film.md.'))
    from wavebit.fdtd._prepare.debye_stack import descriptor as debye_stack_descriptor
    relaxation_stack = debye_stack_descriptor(scene, plan.grid)
    if relaxation_stack is not None:
        out.append((SubpixelAccuracyWarning,
                    f'The Debye stack has {len(relaxation_stack.films)} films normal to '
                    f'{"xyz"[relaxation_stack.axis]}, with separate declared Debye memories '
                    'on disjoint padded supports. Each member retains its single-film '
                    'harmonic normal and arithmetic tangential response in one plain '
                    'lossless dielectric. Touching films, mixed pole families, initial '
                    'fields, tracing and Decay are outside this bounded scope. Validate '
                    'observables under refinement and check duration and absorbers '
                    'separately. See docs/fd-debye-stack.md.'))
    from wavebit.fdtd._prepare.oscillator_stack import descriptor as oscillator_stack_descriptor
    resonant_stack = oscillator_stack_descriptor(scene, plan.grid)
    if resonant_stack is not None:
        out.append((SubpixelAccuracyWarning,
                    f'The oscillator stack has {len(resonant_stack.films)} films normal to '
                    f'{"xyz"[resonant_stack.axis]}, each retaining one Drude or Lorentz '
                    'memory on disjoint padded supports. Different member pole kinds are '
                    'admitted; the harmonic normal and arithmetic tangential singleton '
                    'response is unchanged. This bounded scope excludes within-film '
                    'sums, Debye poles, touching films, initial fields, tracing and Decay. '
                    'Validate observables under refinement and check duration and '
                    'absorbers separately. See docs/fd-oscillator-stack.md.'))
    from wavebit.fdtd._prepare.single_pole_film import descriptor as single_pole_descriptor
    dispersive_film = single_pole_descriptor(scene, plan.grid)
    if dispersive_film is not None:
        out.append((SubpixelAccuracyWarning,
                    'The single-pole film uses a spatial Drude/Lorentz oscillator for '
                    'the harmonic normal and arithmetic tangential response. This C8 '
                    'scope is one concrete axis-aligned film in a lossless dielectric. '
                    'The source-band mesh bound certifies bulk material scales only; '
                    'validate observables under refinement and check run duration '
                    'separately. See docs/fd-single-pole-film.md.'))
    from wavebit.fdtd._prepare.debye_film import descriptor as debye_descriptor
    relaxation_film = debye_descriptor(scene, plan.grid)
    if relaxation_film is not None:
        out.append((SubpixelAccuracyWarning,
                    'The Debye film uses spatial relaxation memory for the exact '
                    'harmonic normal and arithmetic tangential response. This C8 '
                    'scope is one concrete axis-aligned film in a plain dielectric. '
                    'The source-band mesh bound certifies bulk scales only; validate '
                    'observables under refinement and check run duration and absorbers '
                    'separately. See docs/fd-debye-film.md.'))
    from wavebit.fdtd._prepare.two_debye_film import descriptor as two_debye_descriptor
    two_relaxation_film = two_debye_descriptor(scene, plan.grid)
    if two_relaxation_film is not None:
        out.append((SubpixelAccuracyWarning,
                    'The two-Debye film uses two spatial relaxation memories for '
                    'the harmonic normal and arithmetic tangential response. This '
                    'bounded C8 scope is one concrete axis-aligned film in a plain '
                    'dielectric. Validate observables under refinement and check '
                    'duration and absorbers separately. The spectral guard covers '
                    'bulk scales only. See docs/fd-two-debye-film.md.'))
    from wavebit.fdtd._prepare.debye_sum_film import descriptor as debye_sum_descriptor
    sum_relaxation_film = debye_sum_descriptor(scene, plan.grid)
    if sum_relaxation_film is not None:
        out.append((SubpixelAccuracyWarning,
                    'The Debye-sum film uses coupled normal relaxation modes and '
                    'arithmetic tangential response for a finite passive Debye sum. '
                    'This bounded C8 scope is one concrete axis-aligned film in a '
                    'plain dielectric. Validate observables under refinement and '
                    'check duration and absorbers separately. The spectral guard '
                    'covers bulk scales only. See docs/fd-debye-sum-film.md.'))
    from wavebit.fdtd._prepare.oscillator_sum_film import descriptor as oscillator_sum_descriptor
    sum_oscillator_film = oscillator_sum_descriptor(scene, plan.grid)
    if sum_oscillator_film is not None:
        out.append((SubpixelAccuracyWarning,
                    'The oscillator-sum film uses coupled physical polarizations for '
                    'the harmonic normal and arithmetic tangential response, with '
                    'each Drude/Lorentz pole retaining its own damping. This bounded '
                    'C8 scope is one concrete axis-aligned film in a plain dielectric. '
                    'Validate observables under refinement and check duration and '
                    'absorbers separately. The spectral guard covers bulk scales only. '
                    'See docs/fd-oscillator-sum-film.md.'))
    from wavebit.fdtd._prepare.hybrid_film import descriptor as hybrid_descriptor
    hybrid = hybrid_descriptor(scene, plan.grid)
    if hybrid is not None:
        out.append((SubpixelAccuracyWarning,
                    'The hybrid film couples Debye and Drude/Lorentz polarizations '
                    'while retaining their distinct discrete response symbols. '
                    'This bounded C8 scope is one concrete axis-aligned film in a '
                    'plain dielectric. Validate observables under refinement and '
                    'check duration and absorbers separately. The spectral guard '
                    'covers bulk scales only. See docs/fd-hybrid-film.md.'))
    if (len(asked) < len(smoothed) and film is None and stack is None and dispersive_film is None
            and relaxation_stack is None and resonant_stack is None and relaxation_film is None
            and two_relaxation_film is None
            and sum_relaxation_film is None and sum_oscillator_film is None
            and hybrid is None):
        # The controlled C3 tensor study measured p=2.02267353134 at eps=2.25,
        # R/lambda=.25, 20/28/40 cells per wavelength and three grid placements.
        # It does not certify the production engine across the remaining contrasts.
        out.append((SubpixelAccuracyWarning,
                    'Smoothed structures use adjoint inverse-material cross coupling. '
                    'The controlled glass-sphere study measured convergence order 2.023 '
                    '(epsilon=2.25, radius/wavelength=0.25, 20/28/40 vacuum cells per '
                    'wavelength, three placements). Production validation across contrasts is '
                    'incomplete. This benchmark is not an error bound for this scene. Validate '
                    'its observables under refinement; exact voxel volume alone does not '
                    'guarantee electromagnetic accuracy. See docs/validation/fdtd/fd-mie-validation.md; this '
                    'rule has not passed the stage-4 Mie gate, which '
                    "discretisation='stencil' passed (C3)."))
    if asked:
        # The voxel rule's sentence above is not this structure's: its E is the prepared
        # operator's. The orders are the research sweeps' (docs/design/fd-symmetric-stencil.md,
        # "Glass and weak sphere refinement": 2.063617 at eps=2.25, 2.057240 at eps=1.1; "C3:
        # the high-index sphere, and the verdict": 2.2442819 at eps=12, -4.082356% at 80).
        where = ', '.join(f'structures[{index}]' for index in asked)
        out.append((SubpixelAccuracyWarning,
                    f'{where} use{"s" if len(asked) == 1 else ""} '
                    f'discretisation={STENCIL!r}: the symmetric extended-stencil operator on E. '
                    'It passes the stage-4 C3 Mie gate: convergence order 2.244 on the '
                    'epsilon=12 sphere (radius/wavelength=0.25, 40/56/80 vacuum cells per '
                    'wavelength, three placements; -4.08% at 80), 2.064 on the glass sphere '
                    '(epsilon=2.25) and 2.057 at epsilon=1.1 (20/28/40). On glass and weak '
                    'spheres at those grids its error is larger than with the default rule. These '
                    'benchmarks are not an error bound for this scene. Validate its observables '
                    'under refinement. See docs/fd-symmetric-stencil.md.'))
    report = Approximations()
    proxy_stats, proxy_by_index = {}, {}
    magnetic = _background(scene).is_magnetic or any(
        s.material.is_magnetic for s in scene.structures)
    try:
        occupied = occupancy(scene, plan, runtime('numpy'),
                             fields=('E', 'H') if magnetic else ('E',), report=report)
        if plan.constitutive:
            from wavebit.fdtd._prepare.constitutive import lower as cross_coefficients
            cross_coefficients(scene, plan, runtime('numpy'), report=report,
                               proxy_stats=proxy_stats, proxy_by_index=proxy_by_index)
    except (TypeError, ValueError, RuntimeError) as err:
        raise ValueError('fd.plan requires concrete geometry to measure smoothed voxel '
                         'bounds before numerical lowering; a geometry value could not be '
                         f'evaluated on NumPy ({type(err).__name__}: {err}). '
                         'A traced public shape with range= is stage 4 C6.') from err
    _check_fill_contract(scene, occupied, plan.grid)
    if rows is not None:
        rows.extend(_body_rows(scene, plan, occupied, proxy_by_index))
    note = bound_note(report)
    if proxy_stats:
        details = ', '.join(f'{name}: {value:.6g}' for name, value in sorted(proxy_stats.items()))
        category, message = out[0]           # the voxel rule's: a custom body cannot ask
        out[0] = (category, message + ' The cross terms of a body other than the sphere '
                  '(fd.Box, fd.Cylinder, fd.Prism, or a custom body) use a nearest-surface '
                  'tangent normal. At their common sampling points its squared components '
                  'differ from the supplied diagonal normal moment by maxima {' + details +
                  '}. These are measured geometry-proxy discrepancies, not bounds on '
                  'electromagnetic error. Beyond the sphere, only the unturned fd.Box has '
                  'a convergence gate of its own -- its planar faces, against the exact '
                  'Airy slab (G5); turned faces, curved sides, creases and CSG cross moments '
                  'are unvalidated.')
    if note is not None:
        out.append((BoundedVoxelWarning, note))
    return out


def _check_fill_contract(scene, occupied, grid=None):
    """Refuse a smoothed body whose fill or normal moment leaves ``[0, 1]`` (stage 4, C5).

    The mixing rule stays between its two phases -- so every ``eps`` and ``mu`` it builds is
    positive -- exactly when ``f`` and ``w_c`` lie in ``[0, 1]``, and every public body keeps
    that to round-off: the C5 audit measured fills within ``[-2.6e-12, 1 + 3e-14]`` and the
    lowered ``eps`` inside its phases to 3.2e-13, over the sphere, the turned and unturned box,
    the cylinder and the prism at contrasts 2.25, 12 and 80. So this is the ``eps <= 0`` guard
    ``coefficients._stack`` defers to, written as the body contract it rests on and read
    against :data:`_CLIP_ROUNDOFF`: it fires on a body that breaks the contract -- an external
    ``fd.Shape`` or a kernel bug -- and not on round-off. With a fill of 1.5 against
    ``eps = 12`` in vacuum, the component the normal crosses reads the harmonic mean
    ``1/(1.5/12 - 0.5/1) = -2.67``: a negative permittivity, and a field that grows.

    Three corrections from the C5 review. The fill's round-off grows with ``|x|/dx`` (the
    clip works in absolute coordinates), and a domain 3e5 cells from the origin read fills up
    to ``1 + 2.8e-9`` on a correct sphere, so the tolerance is scaled by the grid's largest
    coordinate over ``dx`` when ``grid`` is given. A NaN passed every comparison; it is
    refused by name. And the guard runs from ``coefficients.build`` as well as from the plan's
    notes, so that it speaks before the star lift reads a fill: the Gershgorin
    certificate the lift replaced read a contract-breaking turned box's real negative
    ``1/eps`` as an inconclusive bound.
    """
    tol = _CLIP_ROUNDOFF
    if grid is not None:
        reach = max(abs(v) for v in tuple(grid.low) + tuple(grid.high)) / grid.dx
        tol = max(tol, 64.0 * np.finfo(float).eps * reach)
    for index, entry in occupied.items():
        for field in ('E', 'H'):
            for c, pair in enumerate(entry.get(field, ())):
                for name, values in zip(('fill', 'w'), pair):
                    values = np.asarray(values, dtype=float)
                    structure = scene.structures[index]
                    where = (f'structures[{index}] ({structure.shape.describe()}) answers a '
                             f'{name} on {field}{"xyz"[c]}')
                    if not np.all(np.isfinite(values)):
                        raise ValueError(f'{where} that is not finite. A fill is the fraction '
                                         f'of the voxel inside and w a squared normal '
                                         f'component; this body breaks the fd.Shape contract.')
                    low, high = float(values.min()), float(values.max())
                    if low < -tol or high > 1.0 + tol:
                        off = (f'{-low:.3g} below 0' if low < -tol else
                               f'{high - 1.0:.3g} above 1')
                        raise ValueError(
                            f'{where} reaching {off}, outside [0, 1] (tolerance {tol:.2g}). The '
                            f'mixing rule keeps every eps and mu between its two phases only '
                            f'for a fill and a normal moment in [0, 1]; outside it the lowered '
                            f'eps can be negative and the run grows without bound. Every public '
                            f'body keeps [0, 1] to round-off, so this is a body that breaks the '
                            f'fd.Shape contract (a fill is the fraction of the voxel inside; w '
                            f'is a squared normal component).')


def _stabilize_notes(scene, plan):
    """A stabilized PML face asking for less than the loss measured to suffice at its target."""
    from wavebit.fdtd._prepare.cpml import MEASURED_STABILIZE
    from wavebit.fdtd._prepare.cpml import spell
    from wavebit.fdtd._prepare.cpml import sufficient_stabilize
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
                        f'1e-8 and stronger targets; src/wavebit/fdtd/_prepare/cpml.py::'
                        f'MEASURED_STABILIZE). Below it, a lossless structure with a bound '
                        f'mode next to this face may grow instead of settling'))
    return out


def _wavelength(scene, plan):
    if getattr(scene, 'sources', ()):
        from wavebit.fdtd._prepare.sources import shortest_wavelength
        return shortest_wavelength(scene)
    if scene.resolution.wavelength is not None:
        return scene.resolution.wavelength
    lengths = [w for _, w in plan.pml_wavelengths]
    return lengths[0] if lengths else None


def _index_at(material, wavelength, *, conductive=False):
    """``(n_real, skin_depth)`` of ``material`` at ``wavelength``, or ``(None, None)``."""
    values = []
    for c in range(3):
        eps = concrete(material.eps[c])
        mu = concrete(material.mu_value[c])
        if eps is None or mu is None:
            return None, None
        if material.poles:
            from wavebit.materials.time_domain import Lorentz
            from wavebit.fdtd._prepare.shape_ranges import _carries
            # An exactly inactive concrete oscillator is zero at its nominal resonance
            # too. This reader also serves post-run source round-trip diagnostics.
            poles = tuple(p for p in material.poles if not (
                type(p) is Lorentz and concrete(p.strength) == 0 and not _carries(p.strength)))
            eps = eps_at(poles, wavelength, eps)
            if any(concrete(getattr(p, f)) is None
                   for p in material.poles for f in p.__dataclass_fields__):
                return None, None
        if conductive and isinstance(material, Conductor):
            sigma = concrete(material.sigma[c])
            if sigma is None:
                return None, None
            eps = complex(eps) + 1j * sigma * wavelength / (2.0 * math.pi)
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
    from wavebit.fdtd._prepare.oscillator_stack import notes as oscillator_stack_resolution_notes
    oscillator_stack_notes = oscillator_stack_resolution_notes(scene, plan, _wavelength(scene, plan))
    if oscillator_stack_notes is not None:
        return oscillator_stack_notes
    from wavebit.fdtd._prepare.debye_stack import notes as debye_stack_resolution_notes
    debye_stack_notes = debye_stack_resolution_notes(scene, plan, _wavelength(scene, plan))
    if debye_stack_notes is not None:
        return debye_stack_notes
    from wavebit.fdtd._prepare.conductive_stack import notes as stack_resolution_notes
    stack_notes = stack_resolution_notes(scene, plan, _wavelength(scene, plan))
    if stack_notes is not None:
        return stack_notes
    from wavebit.fdtd._prepare.single_pole_resolution import notes as film_resolution_notes
    film_notes = film_resolution_notes(scene, plan, _wavelength(scene, plan))
    if film_notes is not None:
        return film_notes
    from wavebit.fdtd._prepare.debye_resolution import notes as debye_resolution_notes
    debye_notes = debye_resolution_notes(scene, plan, _wavelength(scene, plan))
    if debye_notes is not None:
        return debye_notes
    from wavebit.fdtd._prepare.two_debye_resolution import notes as two_debye_resolution_notes
    two_debye_notes = two_debye_resolution_notes(scene, plan, _wavelength(scene, plan))
    if two_debye_notes is not None:
        return two_debye_notes
    from wavebit.fdtd._prepare.debye_sum_resolution import notes as debye_sum_resolution_notes
    debye_sum_notes = debye_sum_resolution_notes(scene, plan, _wavelength(scene, plan))
    if debye_sum_notes is not None:
        return debye_sum_notes
    from wavebit.fdtd._prepare.oscillator_sum_resolution import notes as oscillator_sum_resolution_notes
    oscillator_sum_notes = oscillator_sum_resolution_notes(scene, plan, _wavelength(scene, plan))
    if oscillator_sum_notes is not None:
        return oscillator_sum_notes
    from wavebit.fdtd._prepare.hybrid_resolution import notes as hybrid_resolution_notes
    hybrid_notes = hybrid_resolution_notes(scene, plan, _wavelength(scene, plan))
    if hybrid_notes is not None:
        return hybrid_notes
    from wavebit.fdtd._prepare.conductive_film import descriptor
    conductive = descriptor(scene, plan.grid) is not None
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
        n, skin = _index_at(material, wavelength, conductive=conductive)
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
                    f'returns is the grid\'s, not the metal\'s. Refine dx below {skin:g}. (A '
                    f'perfectly conducting body is not in this build: fd.PEC is a face of the '
                    f'box, not a material. This line is stated from the skin-depth argument, '
                    f'not measured on this kernel.)')
            if cells < 2.0:
                out.append((CoarseMeshWarning,
                            f'{where}: {cells:.2f} cells per skin depth at wavelength '
                            f'{wavelength:g}; two is the fewest that resolves the decay'))
            continue
        cells = wavelength / (n * plan.dx)
        if cells < 8.0:
            raise _coarse_refusal(where, n, wavelength, plan.dx, sources)
        if cells < 12.0:
            out.append((CoarseMeshWarning,
                        f'{where}: {cells:.2f} cells per wavelength (index {n:.3f}); 12 is '
                        f'where the second digit starts to hold and 20 is the usual rule'))
    return out


def _which(sources):
    return ", the sources' band.shortest_wavelength" if sources else ''


def _coarse_refusal(where, n, wavelength, dx, sources=()):
    """The refusal a material sampled below 8 cells per wavelength raises, from either door.

    One text, two call sites: :func:`_resolution_notes`, which reads the sources' band at the
    end of ``plan`` (through ``warn_all``), and :func:`check_wavelength_resolution`, which reads
    the same band and each ``Diffraction`` wavelength at the start of it, before a rule does.
    """
    cells = wavelength / (n * dx)
    return ValueError(
        f'{where} is sampled at {cells:.2f} cells per wavelength (index {n:.3f}, '
        f'wavelength {wavelength:g}{_which(sources)}, dx {dx:g}). The measured convergence behind '
        f'the legacy engine\'s CELLS_PER_WAVELENGTH runs a worst per-order error of '
        f'1.2e-2 at ~11 cells and 2.8e-3 at 23, so below 8 not even the first digit '
        f'survives and the run would return a number shaped like an answer. Refine dx '
        f'to at most {wavelength / (8.0 * n):g}, or say a coarser wavelength.')


def check_wavelength_resolution(scene, grid):
    """Refuse a wavelength a rule will read at which a material is sampled below 8 cells per
    wavelength: a ``Diffraction`` monitor's, and the sources' band.

    :func:`_resolution_notes` holds the sources' band to that floor, but it runs from
    ``warn_all``, at the end of ``plan`` -- after the grazing rule -- and it reads
    ``band.shortest_wavelength`` only, so nothing held a monitor's own wavelengths to it at
    all. Both wavelengths reach the grazing rule, whose model rests on the grid carrying every
    order it judges (8 cells per wavelength keeps ``dx^2 (n^2 W^2 - K^2)`` inside ``[0, 4]``:
    ``layer_return.py``), and both met an order past the grid's Nyquist along the axis: the
    review's ``dx = 1/13`` in an ``eps 1.2`` tooth at 0.22 / 0.2 / 0.18 (4.4 .. 3.6 cells per
    wavelength in the tooth) was a bare 'math domain error' on the incidence row and a silent
    NaN verdict on the orders, and a stabilized column at 3.0 cells per wavelength -- with no
    ``Diffraction`` monitor at all, the incidence judged at ``band.concrete_wavelength`` --
    raised ``layer_return``'s own refusal by name. Below 8 not even the first digit of the
    orders survives (the table :func:`_resolution_notes` quotes), so the wavelength is refused
    here, before any rule reads it: the monitors first, shortest first, so the refusal names
    the worst one, then the band in :func:`_resolution_notes`'s own words (that refusal is the
    same text, from :func:`_coarse_refusal`; what stays in ``warn_all`` is its warnings, and
    the scenes with no sources, whose wavelength the plan infers).
    """
    from wavebit.fdtd._prepare.oscillator_stack import validate as validate_oscillator_stack_resolution
    if validate_oscillator_stack_resolution(scene, grid):
        return
    from wavebit.fdtd._prepare.debye_stack import validate as validate_debye_stack_resolution
    if validate_debye_stack_resolution(scene, grid):
        return
    from wavebit.fdtd._prepare.conductive_stack import validate as validate_stack_resolution
    if validate_stack_resolution(scene, grid):
        return
    from wavebit.fdtd._prepare.single_pole_resolution import validate as validate_film_resolution
    if validate_film_resolution(scene, grid):
        return
    from wavebit.fdtd._prepare.debye_resolution import validate as validate_debye_resolution
    if validate_debye_resolution(scene, grid):
        return
    from wavebit.fdtd._prepare.two_debye_resolution import validate as validate_two_debye_resolution
    if validate_two_debye_resolution(scene, grid):
        return
    from wavebit.fdtd._prepare.debye_sum_resolution import validate as validate_debye_sum_resolution
    if validate_debye_sum_resolution(scene, grid):
        return
    from wavebit.fdtd._prepare.oscillator_sum_resolution import validate as validate_oscillator_sum_resolution
    if validate_oscillator_sum_resolution(scene, grid):
        return
    from wavebit.fdtd._prepare.hybrid_resolution import validate as validate_hybrid_resolution
    if validate_hybrid_resolution(scene, grid):
        return
    from wavebit.fdtd._legacy_monitors import Diffraction
    from wavebit.fdtd._prepare.conductive_film import descriptor
    conductive = descriptor(scene, grid) is not None
    materials = [('Scene(background=)', _background(scene))]
    materials += [(f'structures[{i}]', s.material)
                  for i, s in enumerate(getattr(scene, 'structures', ()) or ())]
    asked = {}
    for m in getattr(scene, 'monitors', ()) or ():
        if isinstance(m, Diffraction):
            for w in m.wavelengths:
                asked.setdefault(w, m.name)
    for w in sorted(asked):
        for where, material in materials:
            n, skin = _index_at(material, w, conductive=conductive)
            if n is None or skin is not None:
                continue                    # traced (not countable), or a metal: its skin
            cells = w / (n * grid.dx)
            if cells < 8.0:
                raise ValueError(
                    f'Diffraction({asked[w]!r}) asks for wavelength {w:g}, where {where} is '
                    f'sampled at {cells:.2f} cells per wavelength (index {n:.3f}, dx '
                    f'{grid.dx:g}): below the 8 the plan holds the sources\' band to, where '
                    f'not even the first digit of an order survives -- the run would return '
                    f'numbers shaped like efficiencies. Drop that wavelength (the sources\' '
                    f'band sets which ones carry power), or refine dx to at most '
                    f'{w / (8.0 * n):g}.')
    sources = tuple(getattr(scene, 'sources', ()) or ())
    if not sources:
        return                              # no band to read, and no incidence row to judge
    from wavebit.fdtd._prepare.sources import shortest_wavelength
    band = shortest_wavelength(scene)
    if band is None:
        return                              # traced: _resolution_notes says so, in a warning
    for where, material in materials:
        n, skin = _index_at(material, band, conductive=conductive)
        if n is not None and skin is None and band / (n * grid.dx) < 8.0:
            raise _coarse_refusal(where, n, band, grid.dx, sources)


def _proximity_notes(scene, plan):
    """A structure within half a wavelength, in its own medium, of an absorbing face.

    Not a layer across the face's own axis that runs into a ``PML`` matched to its own
    material: a substrate ``wavebit.fdtd`` continues through the padding, whose face it
    matches to that substrate, or a compat ``fd.PML(background=)`` naming the layer's
    material. There the stretch *is* matched to this structure and there is no interface
    in front of the layer to keep clear. The same layer under a PML matched to anything
    else, or under an ``Absorber``, keeps its note.
    """
    wavelength = _wavelength(scene, plan)
    if wavelength is None or not plan.absorbing_faces:
        return []
    from wavebit.fdtd._prepare.conductive_film import descriptor
    conductive = descriptor(scene, plan.grid) is not None
    from wavebit.fdtd._prepare.single_pole_film import descriptor as single_pole_descriptor
    from wavebit.fdtd._prepare.single_pole_resolution import index_at as film_index_at
    film = single_pole_descriptor(scene, plan.grid)
    if film is None:
        from wavebit.fdtd._prepare.debye_film import descriptor as debye_descriptor
        from wavebit.fdtd._prepare.debye_resolution import index_at as film_index_at
        film = debye_descriptor(scene, plan.grid)
    if film is None:
        from wavebit.fdtd._prepare.two_debye_film import descriptor as two_debye_descriptor
        from wavebit.fdtd._prepare.two_debye_resolution import index_at as film_index_at
        film = two_debye_descriptor(scene, plan.grid)
    if film is None:
        from wavebit.fdtd._prepare.debye_sum_film import descriptor as debye_sum_descriptor
        from wavebit.fdtd._prepare.debye_sum_resolution import index_at as film_index_at
        film = debye_sum_descriptor(scene, plan.grid)
    if film is None:
        from wavebit.fdtd._prepare.oscillator_sum_film import descriptor as oscillator_sum_descriptor
        from wavebit.fdtd._prepare.oscillator_sum_resolution import index_at as film_index_at
        film = oscillator_sum_descriptor(scene, plan.grid)
    if film is None:
        from wavebit.fdtd._prepare.hybrid_film import descriptor as hybrid_descriptor
        from wavebit.fdtd._prepare.hybrid_resolution import index_at as film_index_at
        film = hybrid_descriptor(scene, plan.grid)
    from wavebit.fdtd._prepare.conductive_stack import descriptor as stack_descriptor
    from wavebit.fdtd._prepare.conductive_stack import index_at as stack_index_at
    stack = stack_descriptor(scene, plan.grid)
    from wavebit.fdtd._prepare.debye_stack import descriptor as debye_stack_descriptor
    from wavebit.fdtd._prepare.debye_stack import index_at as debye_stack_index_at
    relaxation_stack = debye_stack_descriptor(scene, plan.grid)
    from wavebit.fdtd._prepare.oscillator_stack import descriptor as oscillator_stack_descriptor
    from wavebit.fdtd._prepare.oscillator_stack import index_at as oscillator_stack_index_at
    resonant_stack = oscillator_stack_descriptor(scene, plan.grid)
    out, grid, pml = [], plan.grid, dict(plan.pml_cells)
    for index, structure in enumerate(getattr(scene, 'structures', ()) or ()):
        if resonant_stack is not None:
            n, _ = oscillator_stack_index_at(resonant_stack.films[index], wavelength)
        elif relaxation_stack is not None:
            n, _ = debye_stack_index_at(relaxation_stack.films[index], wavelength)
        elif stack is not None:
            n, _ = stack_index_at(stack.films[index], wavelength)
        elif film is not None:
            n, _ = film_index_at(film, wavelength)
        else:
            n, _ = _index_at(structure.material, wavelength, conductive=conductive)
        if n is None or n <= 0:
            continue
        clearance = 0.5 * wavelength / n
        box = structure.shape.support(grid)
        for name, cells in plan.absorbing_faces:
            fa, side = 'xyz'.index(name[0]), name[1]
            k0, k1 = box[fa]
            if (structure.shape.uniform_along(fa, (0, grid.n[fa]), grid)
                    and structure.shape.layer_axis() != fa):
                # A transverse face: the structure runs through it with no face of its own in
                # the way, so the clearance it would be measured against is its own thickness.
                # The ``layer_axis`` half keeps a layer *across this very axis* out of the
                # skip even where its span happens to fill the box -- a scene that meant
                # ``Scene(background=)``; stage 1 printed the note for it and C0 moves no
                # number, so it still does -- unless the PML is matched to it, below.
                continue
            if (name in pml and structure.shape.layer_axis() == fa
                    and (k0 < cells if side == '-' else k1 > grid.n[fa] - cells)
                    and _same_medium(structure.material, _matched_medium(scene, name))):
                # A layer across this axis running into a PML whose stretch is matched to
                # its own material: the half-space the public module continues through the
                # padding. The note's premise -- a layer matched to something other than
                # this structure -- is false there, and its gap would be negative.
                continue
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
                            f'error at 0.5 against 8.81e-4 at 0.125), and '
                            f'{_matched_phrase(scene, plan, name)}, not to this '
                            f'structure\'s near field'))
    return out


def _matched_medium(scene, name):
    """The medium the ``PML`` on face ``name`` is matched to (``params._cpml`` reads the same)."""
    background = scene.boundaries.face(name).background
    return _background(scene) if background is None else background


def _matched_phrase(scene, plan, name):
    """What the absorbing face ``name`` is matched to, as the proximity note says it.

    A ``PML`` is matched to one medium, named by the index its stretch is graded for
    (:func:`~wavebit.fdtd._prepare.cpml.face_eps_r`) and marked when that medium is the
    scene's background. An ``Absorber`` has no medium to name: its equal electric and
    magnetic rates match it by construction, at normal incidence only.
    """
    if name not in dict(plan.pml_cells):
        return 'an Absorber there is matched at normal incidence only'
    from wavebit.fdtd._prepare.cpml import face_eps_r
    matched = _matched_medium(scene, name)
    eps_r = face_eps_r(matched, 'xyz'.index(name[0]))
    if eps_r is None:
        return 'the PML there is matched to a traced medium'
    where = ' (the background)' if _same_medium(matched, _background(scene)) else ''
    return f'the PML there is matched to index {math.sqrt(eps_r):g}{where}'


def _same_medium(a, b):
    """Whether ``a`` and ``b`` are one plain ``Material``, ``eps`` and ``mu`` entry by entry.

    Read from concrete numbers only: a traced entry is not known to agree, so it answers
    ``False`` and the note stays. ``mu=None`` and ``mu=1`` agree, as ``mu_value`` reads them.
    """
    if type(a) is not Material or type(b) is not Material:
        return False
    for x, y in zip(a.eps + a.mu_value, b.eps + b.mu_value):
        cx, cy = concrete(x), concrete(y)
        if cx is None or cy is None or cx != cy:
            return False
    return True


def warn_all(scene, plan, lifted=None):
    """Emit every note as a real warning; ``plan`` calls this, ``report`` lists them instead."""
    from wavebit.fdtd._spelling import spoken
    for category, message in notes(scene, plan, lifted=lifted):
        warnings.warn(spoken(message), category, stacklevel=3)
