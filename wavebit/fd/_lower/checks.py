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
    a structure within half a wavelength of      check_resolution, here (warns)
      an absorbing face
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
    (stage 4 C2) a smoothed shape whose         check_smoothed, here: three of the six
      material reaches a face of the box          components gain material outside a minus
                                                  face, and lose the last half cell at a
                                                  plus one -- two arms, two messages
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
import math
import warnings

import numpy as np

from ..diagnostics import AbsorberProximityWarning, CoarseMeshWarning, PMLStabilityWarning
from ..materials import Conductor, Dispersive, concrete, eps_at

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
    ``tests/test_fd_occupancy.py::test_a_smoothed_shape_clear_of_a_snapped_face_composes_exactly``.
    What is left for this function is unchanged.

    Note that two slabs on *different* axes always intersect -- each is full-width on the
    other's axis -- so this refuses a dispersive slab crossed by any other structure, and names
    the two spellings that work.
    """
    structures = tuple(getattr(scene, 'structures', ()) or ())
    boxes = [s.shape.support(grid) for s in structures]
    for i in range(len(structures)):
        for j in range(i + 1, len(structures)):
            if not _intersects(boxes[i], boxes[j]):
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
      ``docs/fdtd-oblique-pulse.md``:84-88 at TM 0.0595 against TE 0.0082 at resolution 40.
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

    **3. No smoothed shape whose material reaches a face of the box.** A cell-snapped
    ``fd.Region`` is clipped to the domain exactly, because its material is a range of cell
    indices. A smoothed body is not, and the six components do not agree about where the box
    ends -- three of them have their Yee point on the node plane and three half a cell off it.
    **That disagreement points opposite ways at the two ends of an axis, and the two arms of
    this door say so separately.**

    *At a minus face, three components GAIN material that is not in the box.* Their first
    voxel is centred on the wall and so reaches half a cell outside it. Measured,
    ``Sphere(radius=0.2)`` centred on the ``x-`` PEC face of a 1.2 box at ``dx = 0.05``: ``Ex``
    lowers **0.01675516** of volume, exactly half the ball (``2 pi R^3/3 = 0.01675516``), and
    ``Ey`` and ``Ez`` lower **0.01988039**, **18.7% more** -- the half cell beyond the wall
    counted in. So a ball half outside the box is neither the ball nor the hemisphere.

    *At a plus face the same three LOSE the last half cell.* Their last point is the last node,
    so their last voxel stops at ``high - dx/2`` and no voxel of theirs holds anything between
    that and the wall. Nothing outside the box is counted there -- the probe voxel this door
    reads at a ``+`` face is itself half inside the domain for those three -- so a body
    entirely inside the box is refused, for the material it is about to lose rather than for
    any it would gain. Measured on the same ball at ``dx = 0.05``: with its surface **0.48
    cells clear** of the ``x+`` wall it lowers 1.87e-5 short of ``4 pi R^3/3`` on ``Ey`` and
    ``Ez`` and exactly on ``Ex``, rising to **1.11e-2** at tangency, while the mirror image of
    each of those at the ``x-`` wall loses nothing at all (8e-16). A ball **inscribed** in the
    domain is therefore refused at its ``+`` faces and accepted at its ``-`` ones, which is not
    a symmetry the grid has to offer. The spelling that works at a ``+`` face is to keep the
    surface half a cell clear of the wall, or to enlarge the domain; "say the truncated body
    with the cell-snapped ``fd.Region``" is the ``-`` arm's remedy and means nothing here, because
    there is nothing sticking out to truncate.

    The test is on the body and not on the support box, because the support carries a cell of
    margin: :func:`_reaches_outside` asks the body for its fill over the cell layer just beyond
    each face the support reaches, at all six components' points, and refuses when any of them
    is above :data:`_CLIP_ROUNDOFF` -- the tolerance that separates the exact clip's own
    round-off from a leak worth refusing, measured and stated there. A shape that is **uniform
    across** the face -- a substrate, a half space -- answers ``Shape.uniform_along`` ``True``
    in that cell layer and is exempt from both arms, because a body that does not change across
    the wall neither gains nor loses anything there. A ``Symmetry`` face is not a wall for this
    door: what a body may be at a mirror plane is :func:`check_symmetry`'s question (C5).

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
    :data:`_CLIP_ROUNDOFF` of the two ends** -- because the exact clip returns neither end
    exactly and comparing against 0 and 1 themselves refused 740 voxels of a legal scene. It is the
    narrow case C2's one smoothed shape can reach of the media-counting refusal C5 owns; a
    creased interface *within* one body (a prism edge, a block corner) is one painter and is
    handled exactly, not refused.

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
    from ..boundaries import Axis1D, Bloch, Periodic, Symmetry
    from .occupancy import smoothed_structures
    structures = tuple(getattr(scene, 'structures', ()) or ())
    smoothed = smoothed_structures(scene)
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
                    f'the scene still reported it. (One that reaches the box but crosses a '
                    f'face is refused too, a few lines below, and for the sharper reason: '
                    f'three of the six components would count the material beyond the wall.) '
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
            a, side, fill = outside
            axis = 'xyz'[a]
            if side == '-':
                raise ValueError(
                    f'{where} reaches past the {axis}- face of the box: its material '
                    f'fills {fill:.4f} of a voxel that lies outside the domain there. A '
                    f'smoothed shape is not clipped to the domain the way a cell-snapped '
                    f'fd.Region is, and the six components do not agree about where the box ends '
                    f'-- at a MINUS face three of them have their point ON the wall, so their '
                    f'first voxel reaches half a cell outside it and counts the material '
                    f'there. Measured, a Sphere(radius=0.2) centred on a PEC x- face at '
                    f'dx = 0.05 lowers 0.01675516 of volume on Ex, exactly half the ball, and '
                    f'0.01988039 on Ey and Ez -- 18.7% more. So a ball half outside the box is '
                    f'neither the ball nor the hemisphere, and it is refused rather than '
                    f'silently trimmed. Move it inside Domain(size=..., center=...), enlarge '
                    f'the domain, or say the truncated body with the cell-snapped fd.Region, '
                    f'which really is clipped. A shape that is uniform ACROSS the face -- a '
                    f'substrate, a half space -- is not refused: it answers uniform_along() '
                    f'True in that cell layer and the clip takes nothing from it.')
            raise ValueError(
                f'{where} reaches the {axis}+ face of the box: {fill:.4f} of a voxel there '
                f'holds material the grid has no point to carry. A smoothed shape is not '
                f'clipped to the domain the way a cell-snapped fd.Region is, and the six '
                f'components do not agree about where the box ends -- but a PLUS face is the '
                f'opposite case to a minus one, and nothing outside the box is being counted '
                f'here. At this face the three components whose Yee offset along {axis} is '
                f'zero have their last point on the last node, so their last voxel stops half '
                f'a cell short of the wall and whatever the body puts between there and the '
                f'wall is simply LOST. The body may be entirely inside the domain and this '
                f'still happens. Measured, a Sphere(radius=0.2) at dx = 0.05 whose surface is '
                f'0.48 cells clear of the x+ wall lowers 1.87e-5 short of 4 pi R^3/3 on Ey and '
                f'Ez and exactly on Ex, and 1.11e-2 short at tangency, while the same ball '
                f'mirrored to the x- wall loses nothing (8e-16). Keep the surface at least '
                f'half a cell ({0.5 * grid.dx:g}) clear of the {axis}+ wall, or enlarge the '
                f'domain with Domain(size=..., center=...) -- a truncated fd.Region is the minus '
                f'face\'s remedy and is no use here, because there is nothing sticking out to '
                f'truncate. A shape that is uniform ACROSS the face -- a substrate, a half '
                f'space -- is not refused: it answers uniform_along() True in that cell layer '
                f'and a body that does not change across the wall loses nothing there.')
        media = [(structure.material, where),
                 (_background(scene), 'Scene(background=)')]
        for other, s in enumerate(structures):
            if other != index and _intersects(box, s.shape.support(grid)):
                media.append((s.material, f'structures[{other}] ({s.shape.describe()})'))
        for material, name in media:
            if not isinstance(material, (Conductor, Dispersive)):
                continue
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
                f'it), or keep it out of the smoothed shape\'s cells. Smoothing them is C8, '
                f'which measures the rule before it writes it.')
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
                    f'whose material is not mixed at all. The tensor form is stage 5 and only '
                    f'if study S1 says it buys an order.')
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
                f'taken -- reach the public door later in stage 4, and C5 owns the general '
                f'refusal this is the reachable case of.')

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
                f'structure\'s faces. A pillar standing on a substrate cannot be said exactly '
                f'yet, with either shape snapped or smoothed; how a body joins a substrate is '
                f'an open stage-4 decision.')


#: The exact clip's own round-off, and the tolerance **every** door here reads a fill against.
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
#: the measured round-off and eight below that, and what reading a fill within it as 0 or 1
#: costs is bounded by itself: the mixing rule is Lipschitz in ``f``, so the mixed value moves by
#: order ``tol`` of the two media's contrast and no more.
_CLIP_ROUNDOFF = 1e-9

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


def _reaches_outside(shape, body, grid, box, mirrors=()):
    """``(axis, '-'/'+', fill)`` where the body's material reaches a face, or ``None``.

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

    **The two sides are not the same question, and the caller's two messages say so.** At a
    ``-`` face the probe layer is cell ``-1``, and every one of the six components' voxels
    there lies wholly outside the domain, so a fill is material the run would gain. At a ``+``
    face the probe layer is cell ``n``, and for the three components whose Yee offset along
    that axis is zero the point sits exactly ON the wall, so their probe voxel spans
    ``[wall - dx/2, wall + dx/2]`` and is half **inside** the domain: what it reads is the
    material those same three components are about to lose, because their last real voxel
    stops at ``wall - dx/2``. Measured on the ball inscribed in the 1.2 box at ``dx = 0.05``,
    the ``+`` probe reads ``Ey`` 0.482619, ``Ez`` 0.482619, ``Hx`` 0.472177 -- all of it at
    ``x <= 0.6``, i.e. inside -- and ``Ex``, ``Hy``, ``Hz``, whose probe voxels really are
    outside, read 0.000000. Refusing either way is right; naming the ``-`` cause at a ``+``
    face is not, which is what a previous writing of the message did.

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
    from .._shapes import occupancy as body_occupancy
    from .occupancy import yee_points
    reach = _VOXEL_REACH * grid.dx
    for a in range(3):
        for side, reached, layer in (('-', box[a][0] == 0, (0, 1)),
                                     ('+', box[a][1] == grid.n[a], (grid.n[a] - 1, grid.n[a]))):
            if not reached or (a, side) in mirrors or shape.uniform_along(a, layer, grid):
                continue
            beyond = [(box[b][0] - 1, box[b][1] + 1) for b in range(3)]
            beyond[a] = (-1, 0) if side == '-' else (grid.n[a], grid.n[a] + 1)
            worst = 0.0
            for field in ('E', 'H'):
                for c in range(3):
                    points = np.asarray(yee_points(grid, tuple(beyond), field, c))
                    near = np.asarray(body.surface(points)[0]) <= reach
                    if not near.any():
                        continue
                    fill, _ = body_occupancy(body, points[near], grid.dx)
                    worst = max(worst, float(np.max(np.asarray(fill))))
            if worst > _CLIP_ROUNDOFF:
                return a, side, worst
    return None


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
    from .._shapes import occupancy as body_occupancy
    from .occupancy import yee_points
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


def _shared_surface_voxels(body_a, body_b, grid, box, include_common=False):
    """``(count, worst)`` voxels of ``box`` whose fill is fractional for **both** bodies.

    Two stages, because the exact fill is 97% of what a smoothed lowering costs and the answer
    is almost always "none". First the two signed distances, which are cheap: a body whose
    surface is inside a voxel is within half a voxel diagonal of that voxel's own point, and a
    CSG-composed distance under-estimates the true one, so ``|d| > _VOXEL_REACH * dx`` proves
    the surface is elsewhere and no marked voxel can be missed. Then, at the marked points
    only, the exact fill: fractional for **both**, read against :data:`_CLIP_ROUNDOFF` at both
    ends.

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
    from .._shapes import occupancy as body_occupancy
    from .occupancy import yee_points
    from .constitutive import common_points
    reach = _VOXEL_REACH * grid.dx
    count, worst = 0, None
    for field in ('E', 'H'):
        for c in range(4 if include_common else 3):
            points = np.asarray(yee_points(grid, box, field, c) if c < 3
                                else common_points(grid, box, field))
            near = np.abs(np.asarray(body_a.surface(points)[0])) <= reach
            if not near.any():
                continue
            near &= np.abs(np.asarray(body_b.surface(points)[0])) <= reach
            if not near.any():
                continue
            marked = points[near]
            fa = np.asarray(body_occupancy(body_a, marked, grid.dx)[0])
            fb = np.asarray(body_occupancy(body_b, marked, grid.dx)[0])
            tol = _CLIP_ROUNDOFF
            both = ((fa > tol) & (fa < 1.0 - tol)
                    & (fb > tol) & (fb < 1.0 - tol))
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
    return count, worst


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
      cells, and a flush one is exactly its own half.

    Measured with this door held open (``studies/fdtd_symmetry_mirror.py``: a PEC cavity with
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
    from ..boundaries import Symmetry
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
            plane = mirror_plane(grid, a, side)
            wall = grid.low[a] + (grid.n[a] if side == '+' else 0) * grid.dx
            plus = (f' On a + face the plane is half a cell INSIDE the wall: {axis} = '
                    f'{plane:g}, not the wall at {axis} = {wall:g}.' if side == '+' else '')
            if body is None:
                found = _snapped_mirror(shape, grid, a, side)
                if found is None:
                    continue
                (lo, hi), (r0, r1), gap = found
                raise ValueError(
                    f'{where} crosses the {axis}{side} Symmetry plane at {axis} = {plane:g}: '
                    f'declared {axis} = ({lo:g}, {hi:g}), it runs as ({r0:g}, {r1:g}) -- its '
                    f'snapped domain part and that part\'s mirror image, which is what a '
                    f'Symmetry face models -- and the edge the mirror makes is {gap:.2f} cells '
                    f'from the declared one, where the snap alone moves an edge by at most '
                    f'half a cell. Measured with this door held open '
                    f'(studies/fdtd_symmetry_mirror.py), an fd.Region declared (-0.1, 0.2) '
                    f'across a mirror at 0 runs as (-0.2, 0.2) and moves a cavity\'s resonance '
                    f'by -9.09e-03 against the same Region in the full cavity. Centre it '
                    f'on the plane, or keep it on the domain side: flush against the plane is '
                    f'exact for a cell-snapped shape, which the mirror joins to its image.'
                    f'{plus}')
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
                    f'door held open (studies/fdtd_symmetry_mirror.py), an fd.Box declared '
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
                f'with this door held open (studies/fdtd_symmetry_mirror.py), an fd.Box '
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
    Only ``fd.Region`` and an ``fd.Slab`` across ``a`` have an edge there; a Slab along
    another axis runs through the mirror unchanged.
    """
    from ..geometry import Region, Slab
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
    from .._shapes import occupancy as body_occupancy
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
    from .._kernel.grid import OFFSETS
    from .occupancy import yee_points
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
    from ..geometry import Region, Slab
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
    from ..boundaries import Bloch, Periodic
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
    from ..boundaries import PEC, PMC
    from ..diagnostics import GeometryClippedWarning
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
            for side, past in (('-', grid.low[a] - lo), ('+', hi - grid.high[a])):
                face = scene.boundaries.face(axis + side)
                if isinstance(face, (PEC, PMC)) and past > 1e-9 * grid.dx:
                    cut += past
                    faces.append(f"'{axis}{side}' ({type(face).__name__})")
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
#: which it warns. Set by study S2b (``studies/fdtd_subpixel_cells_across.py``,
#: ``docs/fd-subpixel-occupancy.md``): at 2 cells an eps-12 sphere's polarizability is 9.6% to
#: 43% high over five placements and it scatters 123% too much; at 4 every body tried is within
#: 9.2% at eps 12 and 2.9% at eps <= 4; at 8, within 2.6%.
CELLS_ACROSS_REFUSED, CELLS_ACROSS_WARNED = 4, 8


def _cells_across(shape, grid, mirrors=()):
    """``(cells, extent)`` across a smoothed body's smallest bounded extent, or ``None``.

    ``None`` for a **layer** -- a body uniform along two axes over the whole domain, a film or
    a half space -- because the interface rule is exact for a plane at any thickness: G5, the
    Airy slab spelled as a smoothed ``fd.Box``, meets the discrete gate to 3.2e-6 relative
    whatever its cells (``tests/test_fd_airy.py``). What S2b measured is a particle's
    polarizability, which a body bounded along two or three axes has; a rod uniform along one
    axis is a two-dimensional particle, judged by its section.

    ``mirrors`` are the ``(axis index, side)`` of the scene's ``Symmetry`` faces. The box is
    extended past each by its own length before the body is measured, because a body crossing
    a mirror is the body declared, not the half the box holds: measured on the box, a bar 7.4
    cells long symmetric about the plane read 3.7 and was refused. A body declared only on the
    domain side of a mirror (its flush half) is measured as declared, which is half the scene's
    body -- conservative, and its message's remedy, declaring it whole, is measured whole.
    """
    from .._kernel.grid import Grid
    for a, side in mirrors:
        n, low = list(grid.n), list(grid.low)
        if side == '-':
            low[a] -= grid.n[a] * grid.dx
        n[a] += grid.n[a]
        grid = Grid(n=tuple(n), dx=grid.dx, low=tuple(low))
    bounded = [a for a in range(3) if not shape.uniform_along(a, (0, grid.n[a]), grid)]
    if len(bounded) < 2:
        return None
    extent = shape.min_extent(grid)
    if extent is None:
        return None
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
    (:func:`_cells_across_notes`). It runs in ``plan`` ahead of the constitutive energy
    certificate, which refuses many small high-contrast bodies too, for a reason that is its
    own slack and with advice that is wrong for them.
    """
    from .occupancy import smoothed_structures
    mirrors = _mirror_faces(scene)
    for index, structure, _ in smoothed_structures(scene):
        found = _cells_across(structure.shape, grid, mirrors)
        if found is None or found[0] >= CELLS_ACROSS_REFUSED:
            continue
        cells, extent = found
        raise ValueError(
            f'structures[{index}] ({structure.shape.describe()}) is {cells:.2f} cells across '
            f'its smallest extent ({extent:g} at dx = {grid.dx:g}), under the '
            f'{CELLS_ACROSS_REFUSED} a smoothed body needs. A body that small is a handful of '
            f'mixed voxels, and what the grid scatters from is set by where they fall, not by '
            f'the body: measured in study S2b (studies/fdtd_subpixel_cells_across.py), an '
            f'eps-12 sphere 2 cells across has a polarizability 9.6% to 43% high over five '
            f'placements and scatters 123% too much against Mie theory, and at eps 2.25 it is '
            f'still up to 9% off; at 4 cells across every body tried is within 9.2% at eps 12 '
            f'and 2.9% at eps 4. Refine to dx <= {extent / CELLS_ACROSS_REFUSED:g} for 4 cells '
            f'across, or dx <= {extent / CELLS_ACROSS_WARNED:g} for 8, where every body tried '
            f'is within 2.6%. A layer -- a body uniform along two axes, like a film -- is '
            f'exempt: the rule is exact for a plane at any thickness.')


def _mirror_faces(scene):
    """``(axis index, side)`` of every ``Symmetry`` face of ``scene``."""
    from ..boundaries import Symmetry
    return tuple((a, side) for a, axis in enumerate('xyz') for side in '-+'
                 if isinstance(scene.boundaries.face(axis + side), Symmetry))


def _cells_across_notes(scene, plan):
    """A ``CoarseMeshWarning`` per smoothed body 4 to 8 cells across (:func:`check_cells_across`)."""
    from .occupancy import smoothed_structures
    out = []
    mirrors = _mirror_faces(scene)
    for index, structure, _ in smoothed_structures(scene):
        found = _cells_across(structure.shape, plan.grid, mirrors)
        if found is None or not found[0] < CELLS_ACROSS_WARNED:
            continue
        cells, extent = found
        out.append((CoarseMeshWarning,
                    f'structures[{index}] ({structure.shape.describe()}) is {cells:.1f} cells '
                    f'across its smallest extent ({extent:g}); between {CELLS_ACROSS_REFUSED} '
                    f'and {CELLS_ACROSS_WARNED}, study S2b measured an eps-12 body\'s '
                    f'polarizability within 9.2% -- a sphere\'s between +1.2% and +6.3% by '
                    f'where the grid falls -- against 2.6% for every body tried at '
                    f'{CELLS_ACROSS_WARNED}. dx <= {extent / CELLS_ACROSS_WARNED:g} reaches '
                    f'{CELLS_ACROSS_WARNED}.'))
    return out


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
    """``(bound, reason)``: a conservative ``dt`` bound, or ``(None, reason)`` if unreadable.

    The vacuum bound is ``n_min dx / sqrt(3)`` -- the *smallest* index, because a medium with
    ``eps < 1`` is faster than vacuum and the usual spelling that drops ``n_min`` lets a
    perfectly ordinary ``eps = 0.81`` diverge. The poles tighten it further
    (``wavebit/_core/grid_fdtd.py``, a kernel seed):

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
    -- the bound below, rigorous for any arrangement (a smoothed mix stays between its phases).
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
    background = _background(scene)
    phases = [background] + [s.material for s in (getattr(scene, 'structures', ()) or ())]
    own = min(m.index_bound() for m in phases)
    eps_min = min(concrete(e) for m in phases for e in m.eps)
    mu_min = min(concrete(m) for p in phases for m in p.mu_value)
    if math.sqrt(eps_min * mu_min) < own * (1 - 1e-12):
        worst += (f' E and H sit at different points of the cell, so where two media meet an '
                  f'E point can hold one phase\'s eps and its H neighbours another\'s mu: the '
                  f'bound uses sqrt(min(eps) * min(mu)) = {math.sqrt(eps_min * mu_min):g} over '
                  f'every phase, below each phase\'s own index ({own:g}). It is rigorous and '
                  f'conservative, and it does not change dt.')
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
    out += _smoothed_notes(scene, plan)
    out += _clipping_notes(scene, plan)
    out += _cells_across_notes(scene, plan)
    out += _radiation_notes(scene)
    return out


def _radiation_notes(scene):
    """One scene-level note for the explicitly retained continuum approximation."""
    from ..monitors import FarField
    from ..diagnostics import RadiationApproximationWarning
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


def _smoothed_notes(scene, plan):
    """Read approximation bounds once for the static scene, outside numerical lowering.

    Use the same components the coefficients spend. This also serves ``fd.check`` when
    params already exist: a report must not depend on whether a byte count required lowering.
    The collector holds component-voxel counts, summed over structures and CSG nodes.
    """
    from .._runtime import runtime
    from .._shapes import Approximations
    from ..diagnostics import BoundedVoxelWarning, SubpixelAccuracyWarning
    from .occupancy import bound_note, occupancy, smoothed_structures
    from .stencil import STENCIL
    smoothed = smoothed_structures(scene)
    if not smoothed:
        return []
    asked = [index for index, structure, _ in smoothed if structure.discretisation == STENCIL]
    out = []
    if len(asked) < len(smoothed):
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
                    'guarantee electromagnetic accuracy. See docs/fd-mie-validation.md; the '
                    'stage-4 Mie gate is incomplete.'))
    if asked:
        # The voxel rule's sentence above is not this structure's: its E is the prepared
        # operator's. The orders are the research sweeps' (docs/fd-symmetric-stencil.md,
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
    proxy_stats = {}
    magnetic = _background(scene).is_magnetic or any(
        s.material.is_magnetic for s in scene.structures)
    try:
        occupancy(scene, plan, runtime('numpy'),
                  fields=('E', 'H') if magnetic else ('E',), report=report)
        if plan.constitutive:
            from .constitutive import lower as cross_coefficients
            cross_coefficients(scene, plan, runtime('numpy'), report=report,
                               proxy_stats=proxy_stats)
    except (TypeError, ValueError, RuntimeError) as err:
        raise ValueError('fd.plan requires concrete geometry to measure smoothed voxel '
                         'bounds before numerical lowering; a geometry value could not be '
                         f'evaluated on NumPy ({type(err).__name__}: {err}). '
                         'A traced public shape with range= is stage 4 C6.') from err
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
    from ..monitors import Diffraction
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
            n, skin = _index_at(material, w)
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
    from .sources import shortest_wavelength
    band = shortest_wavelength(scene)
    if band is None:
        return                              # traced: _resolution_notes says so, in a warning
    for where, material in materials:
        n, skin = _index_at(material, band)
        if n is not None and skin is None and band / (n * grid.dx) < 8.0:
            raise _coarse_refusal(where, n, band, grid.dx, sources)


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
                # number, so it still does.
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
                            f'error at 0.5 against 8.81e-4 at 0.125), and the layer is '
                            f'matched to the background, not to this structure\'s near field'))
    return out


def warn_all(scene, plan):
    """Emit every note as a real warning; ``plan`` calls this, ``report`` lists them instead."""
    for category, message in notes(scene, plan):
        warnings.warn(message, category, stacklevel=3)
