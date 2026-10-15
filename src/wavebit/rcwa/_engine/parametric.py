"""Single-source parametric (staircase-free) Fourier modeling.

A layer is described by a background material plus a list of parametric shapes — rectangles,
circles, ellipses — and its permittivity is Fourier-transformed *analytically* rather than
sampled onto a raster. That removes staircase error entirely, so the same accuracy needs far
fewer harmonics on curved or tilted boundaries.

A layer is ``dict(background=..., shapes=[...])`` and a shape is
``dict(type='rectangle'|'circle'|'ellipse', cx=, cy=, lx=, ly=, angle=, material=...)``. A
``dict(type='polygon', vertices=..., material=...)`` is a shape too, but it is never transformed
here: a layer holding one always takes the boundary route (:func:`layer_needs_resolving`).

Both material slots take a :class:`~wavebit._common.periodic.Material` -- relative permittivity
and the permeability that goes with it -- or a bare number, which is the scalar shorthand for
``Material(epsilon=x)``. They were ``n_base`` and ``n_index`` and held a refractive index; both
spellings are refused by name now, because the *quantity* changed with them and a rename that
kept the number would build a different, legal, energy-conserving structure
(``docs/reference/common/conventions.md``).

**Shapes superpose in this module; they do not paint over each other.** Each one contributes
``(eps_shape - eps_base) * FT(indicator)``, so where two overlap the layer would get *both*
contributions and the permittivity there is neither shape's. Two 120x120 rods of permittivity
4 and 9 in a background of 1, overlapping by 7200 of a 90000 cell, come out at a mean
permittivity of 2.76 that way, where painting the second over the first gives 2.52 -- and the
efficiency follows, 0.9856 against the 0.9792 a raster of the same recipe agrees on.

So the sum is not what an overlapping layer means, and :func:`assemble_eps_coeffs` refuses one
outright. Such a layer is dispatched to :mod:`.boundary`, which resolves the overlap into
disjoint polygons -- later shape over earlier -- and transforms those, edges still exact. Where
nothing overlaps the two rules describe the same structure, so the layer stays here: exact, and
cheaper. Nothing is asked of the recipe either way; the overlap makes the choice.

A ``combine=`` key used to make it explicit, ``'add'`` for the sum and ``'paint'`` for the
painting, and each shape path refused the rule it could not build. It was removed on 2026-08-27
together with the sum: what ``'add'`` builds in an overlap is ``eps_a + eps_b - eps_bg``, which
is not a material, and being the rule nobody expects it had to be asked for -- which taxed every
layer, including the non-overlapping ones this path exists for.

``'paint'`` used to be refused here too, on the grounds that it needs the transform of a shape
*difference*. It still does — :mod:`.boundary` computes that difference as a polygon rather
than in closed form, which is what made it buildable.

The overlap test costs a pairwise geometry test per layer (see :mod:`wavebit._common.geometry.overlap`)
and needs concrete numbers, so a layer whose parameters are tracers is passed through unchecked;
:meth:`freeze_geometry` is the eager call that still sees them.

**Shape parameters are absolute lengths, in whatever unit ``period`` is given in** -- not
fractions of the period. A rod filling half a 0.77-long period is ``lx=0.385``. Passing
fractions builds a rod ``1/period`` times too large and says nothing about it, so
``test_parametric_shape_parameters_are_absolute_lengths`` pins the distinction.

**A centre is measured from the centre of the cell**, not from its corner, so that same rod
sits at ``cx=0`` -- which is also what an omitted ``cx`` means. See :func:`shape_geometry`.

Everything here is differentiable with respect to the shape parameters (``cx``, ``cy``, ``lx``,
``ly``, ``angle``, ``material``) — those are what an inverse-design user actually optimizes. A
material is a design variable like any other, which is why :class:`Material` is a registered
JAX pytree rather than an opaque leaf. The
one primitive that made this hard, Bessel ``J1``, lives in :mod:`wavebit._common.backend`: SciPy
has it, PyTorch's has no autograd backward, and JAX's returns NaN near zero, so each backend
reaches it a different way behind one name.

The convolution matrices are built with the NVF (Li / Schuster et al.) normal-vector
factorization: ``eps`` and ``1/eps`` come from the exact shape transform, while the normal
projectors come from the shapes' *analytic* unit normal field sampled on a grid.

``cx`` and ``cy`` are the exception to "sampled": a lone shape's normal field is built at the
centre of the cell and translated by a phase, which keeps its position derivative exact. See
:func:`normal_projector_coeffs`.

**Several shapes used to be the broken case and are not any more** (2026-08-17). The field was
a normalized *sum of unit vectors*, so a shape ten cells away pushed exactly as hard as the one
underfoot: with two shapes the normal at a boundary was already 40 deg from the true surface
normal, and with three, 43. It also sampled ``X - cx``, the offset to the shape *as named*, so
naming a shape one period over slid a different part of an infinite wedge into the cell and
moved the answer by 1.9e-2. Two changes fix both, and neither carries a width to tune: the
displacement is folded to the nearest image, and the pieces are combined by distance-ratio
weights (:func:`_piece_field`), which collapse to one piece's own normal exactly at its
surface. Boundaries now read 0.000 deg for any number of shapes and naming a period over moves
nothing (5.7e-13).

**A box is four face pieces, each carried over the ring of its periodic images** (2026-09-06,
:class:`GammaPlan` and :func:`_piece_field`). A face is a segment: its image under a lattice
vector lands on a parallel line, slid along it, and where an opposite face's image lands on
the same line the material runs on and there is no interface -- a bar spanning the cell, a box
on a sheared cell, a tilted box whose faces meet their own images part way. The previous
mechanism folded each face distance by the repeat along the shape's own axis, which reaches the
zero-slide case exactly and refused every other by name (3e-2 in ``sum(T)``, not decaying);
now every face is evaluated at every instance that can be its nearest, a coincident pair is
told so by the plan and cancels bitwise, and the sheared and tilted scenes build at 0.19 deg
against an exact interface oracle. The same rewrite removed ``sign()`` from the rectangle
branch, which the position gradient had been blocked on.

**Several shapes are one blend over all of their pieces, and a seam between two of them is
an interface iff their materials differ** (2026-09-06, the second half). Two boxes that abut
-- inside the cell, or one against the other's periodic image -- share a face line over a run,
and the plan records that pair too. Over the run the two faces' weights are equal by the same
bitwise construction a lone box uses across its own images, and the pair cancels (two touching
same-material bars solve as the one bar they are, to 1e-14 on the invariants). A seam between
two *different* materials is then put back co-oriented, in proportion to their contrast and at
any gap (the seam contrast, 2026-09-11), so its normal survives -- the ``eps 4 | eps 9`` seam
used to be a null band, then a binary sign that stepped at contact and at material equality.
There is no per-shape stage in between: a
two-stage blend switches extension at the moment of contact, measured as a 1.1e-2 step in
``sum(T)``, and a single sum over every piece has nothing to switch.

**What it costs, which the commit that made the change did not measure.** The blend evaluates a
signed distance per shape and a product over the others, and a rectangle adds four blended faces
on top, so building the field is far dearer than the sum of level-set gradients it replaced:
31.9 ms to **378.6 ms** on a tilted-rectangle pair and 47.4 ms to **161.6 ms** on a circle pair,
at the ``shape_nvf_grid_size`` of 1024 the solver defaults to. That is 12x and 3.4x, and it is the price
of the boundary exactness and the position gradient above rather than an accident -- but it was
paid before it was counted, and it moves what is worth optimising: the field now costs 4-5x the
three FFTs that transform it (0.36 s against 0.08 s at N = 1024), so the transform is not the
place to look. Whole-solve share falls away with ``truncation_order`` as the eigendecomposition takes
over, from 8.5% at truncation_order 4 to 0.13% at truncation_order 16, which is the range where it might
matter to an optimizer looping at low truncation. Side effect worth knowing when reading
``docs/guide/rcwa/boundary-modeling.md``: the cost ratio it quotes for swapping in the boundary route's
field fell from 12x/486x to 2x/182x because this side got slower, not because that one got
faster.

**And what the pieces cost on top of that** (2026-09-06, measured after the fact again, by a
verifier rather than by the commit). Every box layer is 2 to 4.5x the blend it replaced, since
each face is evaluated at every instance of a size-only ring (13 per face on a lone 350 box, 21
on the grating), and round shapes are unchanged. NumPy field build, best of 3, ``shape_nvf_grid_size``
1024, against the base commit 8bfc6aa on the same machine: the lone 350 box 101.5 -> **285.8
ms** (2.8x), the 350 x 700 grating 102.0 -> **463.2** (4.5x), the L box 97.2 -> 424.6, the B
box on the sheared cell 200.0 -> 513.2 (its previous field was the refused one), two boxes
apart 194.2 -> 428.5, the stacked D pair 365.5 -> 748.5, the K seam 210.8 -> 622.5, two
circles 73.9 -> 70.3 and a lone circle 35.5 -> 31.5. At 512: the lone box 24.4 -> 69.7, the
grating 26.0 -> 116.8. Under ``jax.jit`` at 1024 the lone box is 10.5 -> 41.3 ms and the K
seam 22.4 -> 110.6. Through a whole solve at order 6 the lone box goes 0.31 -> 0.44 s and the
K seam 0.42 -> 0.83, and M small boxes at 512 cost a flat 2x (204 / 455 / 1580 / 2971 ms at
M = 4 / 9 / 25 / 49 against 108 / 247 / 699 / 1520), the plan itself 0.9 / 4.1 / 30 / 121 ms.
Re-measured on the tree that closed the verifiers' findings: 247 / 404 / 370 / 458 / 568 ms
for the lone box, grating, L, B and K at 1024 (61 / 100 / 91 / 118 / 145 at 512), 24.5 for a
lone circle and 63 for two. One round scene is new and dear: an ellipse whose support leaves
its Wigner-Seitz cell is carried over its ring with an exact distance per instance,
**1842 ms** at 1024 (454 at 512) against the 5 ms of one that stays inside
(:func:`_arc_offsets`). The driver everywhere is the size-only instance list, and the cost-only
options are known: for a lone shape the field is built at the cell centre, so a position-free
Euclidean test can prune each face's list there with no staleness exposure, and the
per-instance ``K . (A e)`` products can be hoisted per offset.

**Every enumeration of periodic images is on the Lagrange-reduced basis** (2026-09-06,
:func:`~wavebit.lattice.reduced_basis`). The fold rounds fractional coordinates and
sweeps one cell a side, and that finds the Euclidean nearest image only when the basis is
reduced -- ``as_lattice`` accepts ``a2 = 3 a1 + (50, 700)`` as it is, and a lone 693 circle
on it read its own level set 8.89 deg off in the mean on its boundary (69.9 max; 43.8 / 90.0
at ``10 a1``), silently, on every backend, since the nearest centre lay two cells past the
3x3. The plan records the integer transform (:class:`GammaPlan`), the replay reduces the
traced basis with it, and every offset in a plan counts cells of the reduced spelling: the
same structure written on any spelling plans identically and builds the same field. The
Fourier side never sees it -- the sample grid, the reciprocal vectors, the truncation window
and the origin a centre is measured from are the basis as given, since a window on a
different spelling is a different set of retained orders.
"""

from wavebit._common.geometry.recipes import _cell_floats
from wavebit._common.geometry.recipes import check_layer_spec
from wavebit._common.geometry.recipes import check_shape_spec
from wavebit._common.geometry.recipes import shape_geometry
from wavebit._common.geometry.recipes import holds_polygon
from wavebit._common.geometry.recipes import is_polygon
from wavebit._common.geometry.polygon import outline_values
from wavebit._common.geometry.polygon import polygon_defect
import math
from collections import namedtuple


import wavebit._common.geometry.overlap as _overlap
from wavebit._common.backend import array_namespace
from wavebit._common.backend import as_array
from wavebit._common.backend import checked_float
from wavebit._common.backend import j1
from wavebit._common.backend import jinc
from wavebit._common.backend import sinc
from wavebit._common.backend import torch_autodiff
from wavebit.materials.response import as_material
from wavebit.rcwa._engine.factorization import ConvMatrices
from wavebit.rcwa._engine.factorization import check_normal_grid
from wavebit.rcwa._engine.factorization import _nvf_assemble
from wavebit.lattice import periods_of
from wavebit.lattice import is_axis_aligned
from wavebit.lattice import reduced_basis
from wavebit.lattice import reduction_transform
from wavebit.lattice import sample_grid
from wavebit._common.geometry.overlap import OVERLAP_TOL_REL
from wavebit.rcwa._engine.geometry import StaleGeometryError


def _grid(xp, half_width_u, half_width_v, dev, dtype):
    """Integer reciprocal grid ``(KY, KX)`` spanning the doubled range a convolution matrix needs.

    Doubled because the matrix is indexed by *differences*: ``r - c`` and ``s - d`` each reach
    ``+-2*truncation_order``, so a grid of ``2*truncation_order+1`` per axis cannot fill its corners. See
    ``docs/theory/rcwa/factorization-terms.md`` for the layout and why the result is BTTB, not Toeplitz.
    """
    ky = xp.asarray([float(m) for m in range(-2 * half_width_v, 2 * half_width_v + 1)], dtype=dtype, device=dev)
    kx = xp.asarray([float(m) for m in range(-2 * half_width_u, 2 * half_width_u + 1)], dtype=dtype, device=dev)
    return ky[:, None] + 0.0 * kx[None, :], 0.0 * ky[:, None] + kx[None, :]






def _primitives(shapes, lattice):
    """The layer's shapes as convex primitives, or ``None`` if the parameters are not concrete.

    The overlap test compares numbers, which a JAX tracer is not — and PyTorch tensors, which
    do convert, keep working. Any backend's refusal to hand over a value is that backend
    saying "this is traced", so it means *skip the check*, never *fail the solve*: a traced
    call already had its eager pass through :meth:`freeze_geometry`, and turning a
    differentiable sweep into an exception would be a worse bargain than the one this check
    exists to fix.

    A polygon is not a primitive and comes back ``None`` in its slot, the way an unknown type
    does from :func:`~wavebit._common.geometry.overlap.primitive` -- decided by its *type*, ahead
    of the read, because :func:`shape_geometry` refuses a polygon and the ``except`` below would
    otherwise read that refusal as "traced" and skip every check on the layer. Its outline is
    :func:`_polygon_outlines`'s to read.
    """
    try:
        return [None if is_polygon(shape) else _overlap.primitive(*shape_geometry(shape, lattice))
                for shape in shapes]
    except Exception:                                  # noqa: BLE001 - "not a concrete value"
        return None


def _polygon_outlines(shapes, layer_index):
    """``{index: (N, 2) float array}`` -- each polygon's declared vertices, concrete -- or ``None``.

    ``None`` means a vertex had no value to read (a ``jax.jit`` tracer), which every check here
    answers by stepping aside, as :func:`_primitives` does for a primitive's parameters. The
    vertices stay relative to the cell centre; :func:`_layer_outlines` places them.
    """
    out = {}
    for i, shape in enumerate(shapes):
        if not is_polygon(shape):
            continue
        try:
            out[i] = outline_values(shape['vertices'])
        except TypeError:
            return None
        except ValueError as error:
            raise ValueError(f'Analytic layer {layer_index}: shape {i} (polygon): '
                             f'{error}.') from None
    return out


def _curve_slack_rel():
    """How far outside the curve a polygonised ellipse's corners sit, per unit semi-axis.

    :func:`~wavebit._common.geometry.polygon.ellipse_verts` scales its semi-axes by
    ``sqrt(2 pi / (n sin(2 pi / n)))`` to match the area, so its corners stand out by that factor
    minus one: 5.0e-5 at the boundary route's 256 vertices.
    """
    from wavebit.rcwa._engine.boundary import POLYGON_VERTICES as n
    return math.sqrt(2.0 * math.pi / (n * math.sin(2.0 * math.pi / n))) - 1.0


def _layer_outlines(shapes, lattice, prims):
    """Every shape of a layer as the concrete outline the boundary route builds, with its slack.

    Read through :func:`~wavebit.rcwa._engine.boundary.shape_vertices`, the route's own reader, so
    the outlines swept are the ones the clipper will cut: a polygon is its vertices moved to the
    cell centre, a box its four exact corners, and a curved primitive the area-matched polygon
    the route integrates -- with a slack of its corners' stand-off, so that a shape touching the
    true curve is not refused for a sliver of its chords
    (:func:`~wavebit._common.geometry.overlap.overlapping_outlines`).
    """
    import numpy as np
    from wavebit.rcwa._engine import boundary as _boundary
    basis = _boundary._concrete_basis(lattice)
    polys = [np.asarray(_boundary.shape_vertices(np, _boundary._concrete_shape(shape), basis,
                                                 _boundary.POLYGON_VERTICES), float)
             for shape in shapes]
    slack = [_curve_slack_rel() * max(prim[3], prim[4])
             if prim is not None and prim[0] == _overlap.ELLIPSE else 0.0 for prim in prims]
    return polys, slack


def _shape_name(shapes, i):
    return f"{i} ({shapes[i].get('type', 'rectangle')})"


def _length_keys_of(shape):
    """What to *call* a shape's two half-widths when refusing one of them.

    It mirrors :func:`shape_geometry`'s own fallback, so the message quotes back the key that
    actually supplied the value: a dict saying ``l=-150`` is told about ``l``, not about an
    ``ly`` it never wrote, and ``lx=120, l=-150`` is told about ``l`` for the second one.

    A round type needs no case of its own. :func:`shape_geometry` sets its second length from
    its first, so the two half-widths are equal and the first is always what raises -- an
    ``ly`` a ``circle`` does not take (and that :func:`check_shape_spec` refuses as a *key*)
    can never be named. A branch for it was written first and deleted: nothing could reach it,
    which a mutation showed by surviving its removal.
    """
    first = 'lx' if 'lx' in shape else ('l' if 'l' in shape else 'lx')
    second = 'ly' if 'ly' in shape else ('l' if 'l' in shape else 'ly')
    return (first, second)


def _across(ox, oy):
    """Which period boundary a periodic image sits over, as a phrase or ``''`` for none."""
    axes = [name for name, off in (('x', ox), ('y', oy)) if off]
    return f" across the {' and '.join(axes)} period boundary" if axes else ''


#: Keys every shape takes, whatever its type.

#: The length keys each type takes, on top of :data:`_COMMON_KEYS`. ``l`` is the both-axes
#: spelling. The round types have no ``ly`` **because they have no second length** — that is
#: the whole content of this table, and the reason it is a table rather than a comment.




#: Everything a layer dict may say. Keep this in step with what the builders below read.




def check_layer_shapes(layer, lattice, layer_index=0):
    """Check every shape's keys, and refuse a layer that needs a shape's periodic image.

    This function is the funnel the parametric path reaches the solver through, which is why the
    per-shape check is repeated here and — unlike :func:`shape_geometry`, where it would sit
    closer to the mistake — not inside :func:`_primitives`, whose ``except Exception`` would
    catch the refusal and answer it by silently building without that shape.

    A **non-positive length** is refused first, and separately, because it is not a smaller
    shape -- it is a subtracted one. :func:`shape_ft` carries the sign straight through to the
    ``G = 0`` coefficient, which is the fraction of the cell the shape claims: ``lx = +120`` on
    a 300 period gives ``+0.200000`` and ``lx = -120`` gives ``-0.200000``, so the shape takes
    its contrast *out* of the background instead of putting it in. A 120 x 150 rod of eps 4 on
    eps 1 comes back as eps 0.4 rather than 4, ``sum(T)`` 0.444002054 against 0.980652726, and
    at ``lx = -340`` the average permittivity is **negative** -- a metal, out of a spec that
    named two dielectrics. Every backend agrees on it, and nothing downstream can notice.

    It also silently disabled the two refusals below, which is why it is checked before them:
    :func:`~wavebit._common.geometry.overlap.primitive` stores ``lx / 2`` as a half-width, so a negative
    length is a negative projection radius and the separating-axis test finds every image
    clear -- ``overlapping_pairs`` returns ``[]`` for a rod of ``lx = -340`` on a 300 period,
    exactly as it does for one that fits. ``lx = +340`` is refused; ``lx = -340`` was built,
    and answered 0.047049717.

    An optimiser reaches this by the most ordinary move it has -- shrinking a feature. There is
    no cliff at zero to stop it: ``T`` is even to second order there (0.999998510 at +0.5
    against 0.999998461 at -0.5, the derivative merely changing sign), so a descent slides
    through and starts growing anti-material on the far side. Zero itself goes with it: a
    rectangle of ``lx = 0`` is a shape that is not there, and an ``ellipse`` of ``lx = 0``
    raises ``LinAlgError`` from inside the solve with nothing said about the geometry. An
    omitted length is the same case, since :func:`shape_geometry` defaults it to ``0.0``.

    **Neither transform here reads a shape's periodic images.** The closed form sums each
    shape's own transform; :func:`~wavebit.rcwa._engine.boundary.resolve_painting` clips loops in
    unwrapped Cartesian coordinates and enumerates no images either. (The *normal field* does
    read them, since 2026-09-06 -- :class:`GammaPlan` -- which is why a box abutting its own
    image along part of a face is built now rather than refused.) So two more things are
    refused, and they are one fact wearing two faces:

    * a shape **larger than the period**, which runs into its own image. Its transform is a
      single sinc, which is not the transform of any periodic indicator. Measured on the
      boundary builder: a 1.2x1.4-period rod gives ``<1/eps>`` = 0.186 where the geometry is
      0.250, and a pair with one rod 1.4 periods tall that does not tile the cell, 0.618
      against 0.695.
    * two shapes that **meet only through a cell edge**. An overlap inside the cell resolves
      into disjoint polygons and is built; one that happens through an edge is clipped away,
      and what comes back does not converge — ``sum(T)`` of 0.309 / 0.632 / 0.913 at truncation
      order 5 / 9 / 13 where a wrapped raster holds at 0.997.

    Both would otherwise solve and conserve energy. The second was refused by name until
    2026-08-27, when ``combine`` was removed: the refusal was reached only by a layer that had
    not named its rule, and ``combine='paint'`` walked past it into the answer above. Removing
    the keyword made that branch the default, so the refusal moved here — where it does not
    depend on what the layer says.

    The sweep is ``image_only``: quadratic in the shapes, like the routing question next door,
    and asked of the images alone so that a pair overlapping both inside the cell and through
    an edge cannot be reported as the harmless one. It needs concrete numbers, so a layer whose
    parameters are tracers passes through unchecked; :meth:`freeze_geometry` is the eager call
    that still sees them.

    This was ``check_layer_combination`` until 2026-08-27, when it also refused an overlapping
    layer that had not said whether its shapes add or paint. Nothing is asked of the layer any
    more: ``combine`` went with the summing rule, and the geometry picks the builder.
    """
    shapes = layer.get('shapes', [])
    for i, shape in enumerate(shapes):
        check_shape_spec(shape, i, layer_index)
    if not shapes:
        return

    try:
        Px, Py = (checked_float(v) for v in periods_of(lattice))
        prims = _primitives(shapes, lattice)
    except Exception:                                  # noqa: BLE001 - traced parameters
        return
    if prims is None:
        return
    outlines = _polygon_outlines(shapes, layer_index)
    if outlines is None:
        return

    # A polygon's outline is checked before anything is swept against it, for the reason the
    # lengths below are: the sweep measures overlap as an intersection's area, which on a bow
    # tie is the difference of two lobes and on a degenerate outline is nothing at all.
    for i, V in outlines.items():
        defect = polygon_defect(V)
        if defect is not None:
            raise ValueError(
                f'Analytic layer {layer_index}: shape {i} (polygon) is not a simple outline: '
                f'{defect}. The transform integrates an outline with the winding number of '
                f'each point it encloses, so a crossing outline is not a shape -- its lobes '
                f'would cancel where they should add -- and a degenerate one has no inside to '
                f'fill. Give the vertices in order around the boundary, each once.')

    # Before the sweep, not after -- not because the message would differ (a negative
    # half-width makes the sweep find nothing, so the length is reported either way; moving
    # this loop below changes no test), but because the sweep cannot answer over one. It reads
    # half-widths as projection radii, so `overlapping_pairs` on a rod of `lx = -340` on a 300
    # period returns `[]`, the same as for a rod that fits. Asking first means it is never
    # asked a question it has no way to answer.
    for i, prim in enumerate(prims):
        if prim is None:
            continue                                   # a polygon: checked above
        for half_width, key in zip(prim[3:5], _length_keys_of(shapes[i])):
            if half_width > 0.0:
                continue
            raise ValueError(
                f'Analytic layer {layer_index}: shape {_shape_name(shapes, i)} has '
                f'{key} = {2.0 * half_width:g}. A length here is a size, and the transform '
                f'carries its sign: the shape\'s coefficient at G = 0 is the fraction of the '
                f'cell it claims, so a negative length *subtracts* the shape\'s contrast from '
                f'the background instead of adding it. A 120 x 150 rod of eps 4 on eps 1 comes '
                f'back as eps 0.4, and a long enough one as a negative permittivity -- which '
                f'solves, conserves energy, and is not a structure anything asked for. Zero is '
                f'refused with it: it is a shape that is not there, and on a round type it '
                f'reaches the solve as a division by zero.\n'
                f'  Give it a positive size. If an optimiser walked it through zero, bound the '
                f'parameter or optimise a quantity that cannot go negative -- there is no '
                f'cliff at zero to stop a descent on its own.')

    # `OVERLAP_TOL_REL * max(Px, Py)` is a tolerance *scale*, not geometry: it says how much
    # interpenetration counts, in units of the cell's own size. It keeps *touching* out --
    # a rod exactly filling the period comes out at a gap of zero and is a legal structure,
    # and so are two rods abutting across an edge.
    tol = OVERLAP_TOL_REL * max(Px, Py)

    hits = _overlap.overlapping_pairs(prims, tol, lattice, image_only=True)
    if outlines:
        # Every pair a polygon is in, on the outlines the boundary route will clip -- the
        # primitive sweep above cannot read a polygon, and its pairs are its own.
        polys, slack = _layer_outlines(shapes, lattice, prims)
        pairs = [(i, j) for i in range(len(shapes)) for j in range(i, len(shapes))
                 if i in outlines or j in outlines]
        hits = hits + _overlap.overlapping_outlines(polys, slack, tol, lattice, pairs)
    if not hits:
        return

    i, j, ox, oy, _ = hits[0]
    more = f' (and {len(hits) - 1} more such pair(s))' if len(hits) > 1 else ''
    if i == j and i in outlines:
        what = (f'shape {_shape_name(shapes, i)} runs into its own periodic image'
                f'{_across(ox, oy)}')
        fix = ('Keep it clear of its own images -- an outline may reach past the cell, but '
               'not into a copy of itself')
    elif i == j:
        what = (f'shape {_shape_name(shapes, i)} is larger than the period and runs into its '
                f'own image{_across(ox, oy)}')
        fix = 'Shrink it below the period'
    else:
        what = (f'shapes {_shape_name(shapes, i)} and {_shape_name(shapes, j)} overlap '
                f'only{_across(ox, oy)}')
        fix = ('Move them so the overlap falls inside the cell -- an overlap that does is '
               'resolved into disjoint polygons and built exactly')
    raise ValueError(
        f'Analytic layer {layer_index}: {what}{more}. Neither shape builder here reads a '
        f"shape's periodic images: the closed form sums each shape's own transform, and the "
        f'boundary route clips its polygons in unwrapped coordinates. So the layer would be '
        f'built as a different structure -- one that solves, conserves energy, and does not '
        f'converge.\n'
        f'  {fix}, or state the layer as a raster structure, which carries the wrap.')


def layer_needs_resolving(layer, lattice):
    """Does this layer actually need the boundary route — i.e. is it painted *and* overlapping?

    While nothing intersects, painting and summing describe the same structure, and the closed
    form here is both exact and cheaper than resolving a polygon that has nothing to resolve.
    Routing every shape layer through the boundary route would also move the answer on a layer
    that has no overlap at all -- measured 0.8% apart on disjoint rods.

    Raises when the parameters are not concrete and no plan was frozen. That case used to
    return ``False`` on the reasoning that a traced call has already had its eager pass through
    :meth:`freeze_geometry` — but nothing enforced it, and the cost of the assumption being
    wrong is not a slower path or a warning. It is a **different structure**: the layer is
    summed instead of painted, silently. Measured under ``jax.grad`` on two overlapping rods,
    ``T`` came out 0.8743 where painting gives 0.8539, and the gradient was bit-identical to
    the one the summing builder returns on its own — which is what identified it.

    PyTorch never hit this: its tensors convert to floats, so the check ran. JAX tracers do
    not, so it silently did not.

    **A layer holding a polygon always resolves**, and is told so by the shape's type alone, so
    the answer needs no value and is the same eager, under ``grad`` and under ``jit``. Nothing
    here can sum a polygon: the closed form has no normal field for an outline that is a list
    of vertices -- its field is built per primitive, face by face and arc by arc -- while the
    boundary route builds its field from whatever loops it is handed. A lone polygon therefore
    takes the route even with nothing to resolve, and its plan is the trivial one-loop painting.
    """
    shapes = layer.get('shapes', [])
    if holds_polygon(layer):
        return True
    if len(shapes) < 2:
        return False
    try:
        Px, Py = (checked_float(v) for v in periods_of(lattice))
        prims = _primitives(shapes, lattice)
    except Exception:                                  # noqa: BLE001 - a traced basis
        prims = None
    if prims is None:
        raise ValueError(_NEEDS_A_FREEZE.format(
            what='whether the shapes overlap, which decides how the painting resolves into '
                 'disjoint polygons'))
    return bool(_overlap.overlapping_pairs(prims, OVERLAP_TOL_REL * max(Px, Py),
                                           lattice))


#: The message for the one question a trace cannot answer.
#:
#: It briefly guarded two. `can_plan_eagerly` stood beside it for one revision, on the theory
#: that "can I read the geometry?" and "can I cut the polygons?" had come apart -- and they had
#: not: unifying the read (`boundary._concrete_shape` now uses the same `checked_float`) made
#: planning work wherever reading does. Mutation-tested as dead before it was deleted: under
#: `jit` `layer_needs_resolving` raises here first and the guard was never reached at all
#: (0 calls), while under eager and `grad` it always answered yes.
_NEEDS_A_FREEZE = (
    "a traced shape layer needs a frozen geometry: {what} cannot be asked of a traced "
    "value. Guessing it would solve a different structure than the one asked for, without "
    "saying so.\n"
    "  Freeze once, outside the trace, and replay:\n"
    "    geometry = solver.freeze_geometry()\n"
    "    result = solver.solve(geometry=geometry)\n"
    "  Re-take the freeze whenever the shapes may have started or stopped overlapping; the "
    "plan compares by content, so an unchanged one is free.")


def shape_ft(xp, shape, ky, kx, lattice):
    """Continuous Fourier transform of one shape's indicator, divided by the cell area.

    Returns ``c(ky, kx) = 1/area * FT{1_shape}(Gx, Gy)`` on the integer reciprocal grid — the
    analytic analogue of one Fourier coefficient the ``fourier2d_*`` builders read off a raster.

    ``lattice`` is the cell basis (``None`` = the axis-aligned ``diag(period)``). It reaches
    only the two lines below, and that is the whole of what a skewed cell costs this function:
    ``(Gx, Gy)`` is the reciprocal vector ``G = m*b1 + n*b2`` in **Cartesian** components, and
    everything after it — the rotation, the phase, the sinc product, the Bessel forms — is
    already written in Cartesian frequency and so does not care how ``G`` was reached.

    Nothing about the shape deforms either, because shape lengths are absolute Cartesian
    quantities (see :func:`shape_geometry`), not fractions of the cell. A rectangle stays a
    rectangle under a skew; only the grid of frequencies its transform is *sampled at* stops
    being rectangular. That is why this path needed far less than the raster one, where the
    normal field genuinely is index-based and needs the metric.
    """
    stype, cx, cy, lx, ly, angle = shape_geometry(shape, lattice)

    pi = 3.141592653589793
    # The `lattice is None` arm that stood here computed the rectangle's answer from
    # `Px`, `Py`. The solver stopped reaching it once `_geometry_lattice_for` became
    # total (`cc016fc`): driving only the public API, the one function that ever sees a
    # `None` cell is `as_lattice`, where the conversion happens. Deleted rather than
    # re-keyed on `is_axis_aligned` -- production takes *this* branch on a rectangle
    # today, and the two differ in the last place.
    # (Gx, Gy) = 2*pi * A^-T @ (m, n), written out for the same reason as in
    # wavebit.lattice.order_kxy: one expression per component, no matrix inverse.
    a1x, a2x = lattice[0][0], lattice[0][1]
    a1y, a2y = lattice[1][0], lattice[1][1]
    det = a1x * a2y - a2x * a1y
    Gx = 2.0 * pi * (a2y * kx - a1y * ky) / det
    Gy = 2.0 * pi * (-a2x * kx + a1x * ky) / det
    area = abs(det)

    # Rotated frequency coordinates: kappa = Rot(angle)^T @ k
    ct = xp.cos(as_array(xp, angle, dtype=Gx.dtype))
    st = xp.sin(as_array(xp, angle, dtype=Gx.dtype))
    kxr = Gx * ct + Gy * st
    kyr = -Gx * st + Gy * ct

    phase = xp.exp(-1j * (Gx * cx + Gy * cy))

    if stype in ('rectangle', 'square'):
        s = lx * sinc(kxr * lx / 2.0) * ly * sinc(kyr * ly / 2.0)
    elif stype in ('circle', 'disk'):
        R = lx / 2.0
        # Same DC guard as the ellipse below, and for the same reason: ``rho`` is zero there,
        # ``d(sqrt)/d`` is infinite, and ``jinc``'s own ``where`` turns that into NaN. Here the
        # radicand depends on ``period``, so it is ``d(R)/d(period)`` that used to come back
        # NaN on a circle while the guarded ellipse was fine.
        r2 = Gx * Gx + Gy * Gy
        nz = r2 != 0
        rho = xp.where(nz, xp.sqrt(xp.where(nz, r2, xp.ones_like(r2))), xp.zeros_like(r2))
        s = (pi * R * R) * jinc(R * rho)          # 2 pi R J1(R rho)/rho
    elif stype == 'ellipse':
        a = lx / 2.0
        b = ly / 2.0
        # Guard the DC point so d/d(shape) of sqrt(0) is not NaN.
        kap2 = (a * kxr) ** 2 + (b * kyr) ** 2
        nz = kap2 != 0
        kap = xp.sqrt(xp.where(nz, kap2, xp.ones_like(kap2)))
        j = xp.where(nz, 2.0 * j1(kap) / xp.where(nz, kap, xp.ones_like(kap)),
                     xp.ones_like(kap))
        s = (pi * a * b) * j
    else:
        raise ValueError(f'Unknown parametric shape type: {stype!r}')

    return phase * s / area


def _refuse_overlap(shapes, lattice):
    """Raise if these shapes overlap -- the case the superposition below cannot represent."""
    # One shape is enough to overlap: a shape wider than the period runs into its own image,
    # and summing it is the same plausible wrong answer. Every public route refuses that in
    # `check_layer_shapes` first; this keeps a direct call from summing it silently.
    if not shapes:
        return
    try:
        Px, Py = (checked_float(v) for v in periods_of(lattice))
        prims = _primitives(shapes, lattice)
    except Exception:                                  # noqa: BLE001 - traced parameters
        return          # the concrete pass already answered this; see `layer_needs_resolving`
    if prims is None:
        return
    hits = _overlap.overlapping_pairs(prims, OVERLAP_TOL_REL * max(Px, Py), lattice)
    if not hits:
        return          # nothing to resolve: painting and adding are the same structure here

    # `_across` is not decoration: the sweep tests the eight periodic images too, so the pair
    # a user is told about can be two shapes that plainly miss each other inside the cell.
    # Without the phrase the message reads as a false positive and the layer looks unbuildable.
    i, j, ox, oy = hits[0][0], hits[0][1], hits[0][2], hits[0][3]
    where = _across(ox, oy)
    what = (f'shape {_shape_name(shapes, i)} runs into its own periodic image{where}'
            if i == j else
            f'shapes {_shape_name(shapes, i)} and {_shape_name(shapes, j)} overlap{where}')
    raise ValueError(
        f'an overlapping layer cannot be built here: {what}, and this builder sums every '
        f"shape's "
        f'(eps_shape - eps_base) * FT(indicator). Summing an overlap is a different structure '
        f'from painting it -- measured 1.99 against 1.90 in <eps> for two 300x300 rods of '
        f'permittivity 4 and 9 -- so returning the sum under the name of the paint would '
        f'be a plausible wrong answer rather than a refusal.\n'
        f'  For the painted layer, go through wavebit.rcwa._engine.boundary.to_conv_mat_boundary_layer '
        f'(or parametric.to_conv_mat_parametric_layers, which routes to it), where the overlap is '
        f'resolved into disjoint polygons before anything is transformed.')


def assemble_eps_coeffs(xp, shapes, background, half_width_u, half_width_v, lattice, dtype):
    """Analytic Fourier coefficients of ``eps`` and ``1/eps`` for one layer.

    Each grid has shape ``(4*half_width_v+1, 4*half_width_u+1)``, indexed ``[ky+2*half_width_v,
    kx+2*half_width_u]`` — the doubled range a ``(2*truncation_order+1)**2`` convolution matrix needs
    (``docs/theory/rcwa/factorization-terms.md`` §2 for why it doubles).

    ``background`` and each shape's ``material`` are read through
    :func:`~wavebit._common.periodic.as_material`, so a bare number means ``Material(epsilon=x)``
    and the permittivity is taken straight out -- **there is no squaring here any more**, which
    is one of the sites the convention flip is made of. Wavebit works in ``e^{-iwt}``
    throughout (``Im(eps) > 0`` = loss), with no conjugation on the way in.

    ``mu`` is not read here and must not be: this function is called a second time by
    ``CoreFactorizationMixin._mu_conv_matrices``, on a layer whose materials have had their ``mu``
    put where their ``epsilon`` was, and that second pass is what builds the permeability
    slots. Reading both here would build one of them twice and the other never.

    **This sums, and an overlapping layer is not a sum.** Where two shapes overlap, summing and
    painting describe two different structures, so reconstructing a painted layer from this
    function is wrong -- silently: two 300x300 rods of permittivity 4 and 9 in a 1000x1000 cell
    come back at ``<eps> = 1.99`` where the solver integrates ``1.90``, a 4.7% error that looks
    entirely plausible. The solver does not make that mistake -- :func:`to_conv_mat_parametric_layers`
    routes an overlapping layer to :mod:`.boundary` and only reaches this builder where the
    shapes are disjoint, and where they are disjoint the two rules coincide exactly.

    So an overlap is refused here unconditionally. It was conditional on a ``combine=`` argument
    until 2026-08-27, when ``combine='add'`` was removed: a layer could then ask for the sum,
    and the caller had to say so for the check to run. There is no such ask now, so there is
    nothing for the caller to state and no way to turn the check off.
    """
    _refuse_overlap(shapes, lattice)

    eps_base = as_material(background, 'layer background').epsilon
    dev = getattr(eps_base, 'device', None) if hasattr(eps_base, 'device') else None
    real_dtype = xp.float64 if dtype == xp.complex128 else xp.float32
    KKY, KKX = _grid(xp, half_width_u, half_width_v, dev, real_dtype)

    eps_bg = as_array(xp, eps_base, dtype=dtype)
    inv_bg = 1.0 / eps_bg

    # DC term only: the background is constant over the whole cell.
    is_dc = (KKY == 0) & (KKX == 0)
    zero = xp.zeros(KKY.shape, dtype=dtype)
    eps_c = xp.where(is_dc, eps_bg, zero)
    inv_eps_c = xp.where(is_dc, inv_bg, zero)

    for shape in shapes:
        eps_s = as_array(xp, as_material(shape['material'], 'shape material').epsilon,
                         dtype=dtype)
        ft = xp.astype(shape_ft(xp, shape, KKY, KKX, lattice), dtype)
        eps_c = eps_c + (eps_s - eps_bg) * ft
        inv_eps_c = inv_eps_c + (1.0 / eps_s - inv_bg) * ft

    return eps_c, inv_eps_c


def conv_matrix_2d(xp, coeff2d, half_width_u, half_width_v):
    """Convolution matrix from a 2D coefficient grid.

    ``M[(r,s),(c,d)] = coeff(ky=r-c, kx=s-d)`` — the ordering the ``fourier2d_*`` builders
    produce. The
    sign is the order-index convention (see ``_circulant_index``); the two builders have to
    agree with each other and with ``kx``, which ``test_conv_matrix_matches_grid_convention``
    pins.

    **Rule-neutral, which is what the name is for.** This was ``laurent_conv_2d`` until
    2026-08-12, and ``Laurent`` is the *direct* factorization rule — a rule word on a builder
    that implements neither. :func:`to_conv_mat_parametric_nvf` calls it on ``eps`` and on
    ``1/eps`` two lines apart, and on the three normal-field products; the inverse rule's
    inversion happens in the caller.

    The result is **BTTB** (block-Toeplitz with Toeplitz blocks), not Toeplitz: with the row
    and column pairs flattened, ``i - j`` does not determine ``(r-c, s-d)``, so the diagonals
    are not constant. ``docs/theory/rcwa/factorization-terms.md`` has the check and the vocabulary.
    """
    n_orders_u = 2 * half_width_u + 1
    n_orders_v = 2 * half_width_v + 1
    dev = _device(coeff2d)

    iy = xp.asarray([[[[r - c + 2 * half_width_v]]] for r in range(n_orders_v) for c in range(n_orders_v)],
                    device=dev)
    iy = xp.reshape(iy, (n_orders_v, n_orders_v))
    ix = xp.asarray([[s - d + 2 * half_width_u for d in range(n_orders_u)] for s in range(n_orders_u)], device=dev)

    m4 = coeff2d[iy[:, None, :, None], ix[None, :, None, :]]      # [r, s, c, d]
    return xp.reshape(m4, (n_orders_v * n_orders_u, n_orders_v * n_orders_u))


def _device(x):
    return x.device if hasattr(x, 'device') else None


# FROZEN 2026-08-17 from the sweep in the blend below. The exponent has a *plateau*, not a
# peak, which is what makes it a measured constant rather than the tuning the mollifier route
# was rejected for: p = 3..8 all give 0.000 deg at every surface and hold the two-circle
# position gradient under the 1e-3 the suite asks (2.5e-4 at p = 4, its best). Below it the
# blend is too soft -- p = 1 leaves 0.59 deg at the surface band and misses the gradient at
# 3.3e-3; above it the blend approaches a hard nearest-shape pick and the jump it exists to
# avoid comes back, 1.0e-3 by p = 12. Four is the interior of that window.
#
# A third axis was measured afterwards and does not move the choice: the sampling error of the
# projector coefficients, |f(N) - f(2N)|, falls monotonically as p RISES (1.9e-3, 1.5e-3,
# 1.3e-3, 1.1e-3, 7.3e-4 at p = 2, 3, 4, 6, 8 on two circles at N = 256). The reasoning that
# predicted the opposite is worth recording as wrong: a sharper crossover looks like it should
# integrate worse and integrates better, because a high p leaves the field equal to one shape's
# own smooth field over most of the cell and mixed only in a narrow band, while a low p mixes
# everywhere. Trading p = 4 for p = 8 buys 1.3x there and gives back 1.7x on the position
# gradient (2.5e-4 -> 4.3e-4), so the middle stays.
_BLEND_EXPONENT = 4


def normal_vector_field_analytical(xp, grid_shape, shapes, dtype, lattice, plan=None):
    """Unit normal field of the shapes, sampled on ``grid_shape`` — analytic, not from a raster.

    The sample points come from :func:`wavebit.lattice.sample_grid`, which is where the
    sample grid's ``(Px, Py)`` unpack lives and why it lives in exactly one place — this body
    used to carry its own copy, and ``boundary.distance_gradient_normal_field`` was written from
    that copy.

    ``period`` is still read here, though, and not only for the grid: :func:`shape_geometry`
    takes the pair as well and is reached whether or not a basis was given, so a shape that
    leaves ``cx`` / ``cy`` to default is sensitive to the order even when the sample grid is
    built from a lattice. That is what makes this path less contained than the boundary one, and
    it had to be *measured* rather than read off the shared line: reversing the pair on a 2:1
    cell moves this field by 1.96 with an explicit centre and no lattice, by 1.7 with a defaulted
    centre and a lattice, and by exactly 0 with both.

    **The field is built from pieces** (:func:`_piece_field`): a box is its four faces, each a
    segment carried over the ring of its periodic images, and a round shape is one piece. The
    combinatorics -- which faces share a line with which image, over what, and whether anything
    is left -- is the :class:`GammaPlan`, planned once from concrete numbers and replayed by
    array ops on the traced ones. Left ``None`` it is planned here; under a trace with no primal
    to read (``jax.jit`` with no frozen record) :func:`_blind_plan` stands in, and that costs
    the exact cancellation of an abutting pair -- see there. A traced *basis* with no record is
    refused there rather than guessed at, the ring of images being bounded by the cell, and so
    is a traced *ellipse*, whose instance list is bounded by its size.

    Written without in-place mutation so it stays differentiable in the shape parameters. A
    position is not among them for a lone shape: :func:`normal_projector_coeffs` builds the
    field with the shape at the centre of the cell and translates the *coefficients*.
    """
    Ny, Nx = grid_shape
    X, Y = sample_grid(xp, grid_shape, lattice, dtype)

    if plan is None:
        plan = plan_gamma(shapes, lattice)
        if plan is None:
            plan = _blind_plan(shapes, lattice)
    if plan.empty:
        # Every face is covered by the shape's own images: the shape tiles the plane, the
        # layer is a uniform slab written as a shape, and there is no interface to point at.
        # The same nothing-to-point-at answer an empty shape list gets, and for the same
        # reason: every conv matrix is `eps * I` whatever this returns, so it returns the
        # constant Li's rules read as "x is the normal".
        return xp.ones((Ny, Nx), dtype=dtype), xp.zeros((Ny, Nx), dtype=dtype)

    slots = _material_slots(xp, shapes, dtype)
    # An arc over its ring is one piece evaluated at every instance; for the tiling question
    # the instances are the working set, so they are what is counted (`_tiles_pay`).
    n_pieces = sum(len(pc.imgs) if pc.face == 'arc' else 1 for pc in plan.pieces)
    if _tiles_pay(X, Ny, Nx, n_pieces):
        return _field_tiled(xp, X, Y, shapes, plan, slots, lattice, dtype, (Ny, Nx))
    return _piece_field(xp, X, Y, shapes, plan, slots, lattice, dtype)


#: Side of the square block :func:`_field_tiled` works in. 64 doubles is 32 KB per array, so
#: the ten-odd live arrays of one piece's evaluation stay inside L2 and the whole pipeline runs
#: without touching memory again. Measured against 128 / 192 / 256 on two and four shapes at
#: N = 512 / 1024 / 2048: 64 wins or ties everywhere, by up to 1.3x over the next size.
_TILE = 64

#: Below this many pixels the full-grid arrays already fit in cache and tiling only adds Python
#: loop overhead — measured 1.00x at N = 256 and 1.09-1.24x at N = 512.
_TILE_MIN_PIXELS = 512 * 512


def _tiles_pay(sample, n_v, n_u, n_pieces):
    """Whether to evaluate the field block by block instead of over the whole grid.

    **NumPy only, and that is the point rather than a limitation.** The win here is cache
    residency, which is a property of eager elementwise evaluation: each of the ~50 array
    operations per piece reads and writes the whole grid, so at N = 1024 a two-shape layer
    moves roughly 800 MB through memory to do a few tens of megaflops. JAX and Torch under a
    trace do not have that problem — XLA and the compiler fuse the same chain into one pass —
    and a Python loop over 256 tiles would instead unroll into the graph and wreck compile
    time. So the tiling is taken only where it helps and skipped where it would hurt.

    Decided on the **array**, not on the namespace's name. The obvious spelling,
    ``xp.__name__.endswith('numpy')``, is wrong and quietly so: ``jax.numpy`` ends with
    ``numpy`` too, and it would have turned the tiling on for exactly the backend the paragraph
    above says to keep it off. ``test_only_numpy_takes_the_tiled_path`` is what caught that.

    ``n_pieces`` counts the plan's pieces, not the layer's shapes: a lone box is four of them,
    each carried over a ring of images, and that is already the working set the tiles exist
    for. A lone round shape inside its Wigner-Seitz cell is one piece and nothing to blend;
    one that leaves it is still one piece, evaluated at every instance of its ring, and it
    counts its instances here because that evaluation is the working set the question is
    about.
    """
    import numpy as np
    return (n_pieces > 1 and n_v * n_u >= _TILE_MIN_PIXELS
            and type(sample) is np.ndarray)


def _field_tiled(xp, X, Y, shapes, plan, slots, lattice, dtype, grid_shape):
    """:func:`_piece_field`, evaluated one cache-sized block at a time.

    Bit-identical to the whole-grid path by construction: same operations on the same values in
    the same order, only the traversal differs. Pinned by
    ``test_the_tiled_blend_is_bit_identical_to_the_whole_grid_one``.
    """
    import numpy as np
    n_v, n_u = grid_shape
    # A multi-shape seam carries the complex phase of the material difference.
    # Allocate from the first block so tiling preserves its dtype as well as its values.
    out_x = out_y = None
    for i0 in range(0, n_v, _TILE):
        for j0 in range(0, n_u, _TILE):
            block = (slice(i0, min(i0 + _TILE, n_v)), slice(j0, min(j0 + _TILE, n_u)))
            bx = np.ascontiguousarray(X[block])
            by = np.ascontiguousarray(Y[block])
            nx, ny = _piece_field(xp, bx, by, shapes, plan, slots, lattice, dtype)
            if out_x is None:
                out_x = np.empty(grid_shape, dtype=nx.dtype)
                out_y = np.empty(grid_shape, dtype=ny.dtype)
            out_x[block], out_y[block] = nx, ny
    return out_x, out_y


# ------------------------------------------------------------------------------ the plan

class GammaPlan(namedtuple('GammaPlan', 'ring pieces pairs empty reduction')):
    """The combinatorics of one layer's interface set, planned once and replayed by array ops.

    Two halves, the way :mod:`.boundary` already splits painting: a **plan** -- pure Python,
    content-hashable, taken eagerly on concrete floats -- and a **replay** on the traced
    parameters (:func:`_piece_field`). The plan holds **no coordinates and no material facts**.
    Everything a length or an angle decides is re-derived from the traced value at replay, and
    a material never enters at all, so a sweep over sizes, angles, positions or materials never
    moves the record -- a sweep over what *touches* what does, and that is what the record is.

    ``ring``
        One integer ``r``: every image ``(m, n)`` with ``|m|, |n| <= r`` was swept for
        contacts. ``1 + ceil(max_ij (R_i + R_j) / h_min)`` with ``h_min = |det| / max(|a1|,
        |a2|)`` the shortest lattice height -- a real length ``rho`` moves a fractional
        coordinate by at most ``rho / h_min``, and centre differences are within one cell. It
        is 1 to 3 on an ordinary cell: the L box, 689 x 200 on a 700 square, needs 3.
    ``pieces``
        One :class:`Piece` per face of every box (``+u``, ``-u``, ``+v``, ``-v`` -- whole
        faces, never split) and one ``'arc'`` per round shape, in shape order. Each carries
        ``imgs``, the offsets its replay evaluates it at, and ``dead``, whether the shape's own
        images cover the whole of it (a self run union at full length). ``dead`` is written
        by the planner and read by ``empty`` and by the staleness check alone: the replay
        never reads it, cancelling a covered face through its pair records run by run,
        whether or not the union of the runs is the whole face. An arc's ``imgs`` is
        ``((0, 0),)`` while the shape stays inside the Wigner-Seitz cell of its centre, where
        the nearest centre's level set is the only one whose interface enters the cell (not,
        for an ellipse, the nearest boundary off it: see :func:`_arc_offsets`), and the size
        ring once it leaves; the replay then reads one piece whose distance and
        direction are the nearest instance's, by a hard minimum (:func:`_round_instances`).
    ``pairs``
        One :class:`Pair` per coincidence: piece ``q`` at lattice offset ``(m, n)`` -- relative
        to ``p``'s own instance -- lies on ``p``'s line, opposed, over a run of positive
        length. The run's bounds are not stored: they are the max / min of projections of the
        traced geometry, which the replay computes where it needs them. A self pair (same
        shape) is recorded once, from its ``+`` face; a cross pair from the lower shape index.
    ``empty``
        Every piece is a dead face: the shape tiles the plane, the layer is uniform, and the
        field is the constant ``(1, 0)`` -- Li's checkerboard reaches that with a 2.5 x 1.25
        bar on ``a1 = (1.25, 1.25), a2 = (1.25, -1.25)``.
    ``reduction``
        The integer unimodular ``((t00, t01), (t10, t11))`` that takes the basis as given to
        its Lagrange-reduced spelling (:func:`~wavebit.lattice.reduction_transform`);
        the identity on every reduced basis. **Every offset above counts cells of the reduced
        basis**, the ring included, and the replay reduces the traced basis with these
        integers rather than reading it -- so a record taken on ``a2 = 10 a1 + (50, 700)``
        is the record of ``a2 = (50, 700)``, pair at ``(0, 1)``, and replays on either.

    ``imgs`` is bounded by **size alone**: every offset ``s`` with ``|s . A| <= diam + 2 R``,
    taken *around the shape's own fold* -- the sample point is first reduced to the shape's
    nearest lattice cell, which leaves a residual of at most half the cell's diameter, and the
    nearest instance of any piece within ``R`` of the shape's centre then sits within
    ``2 * (diam / 2) + 2 R`` of that. A superset at every position, so a translated shape never
    lacks its nearest instance and no staleness check reads a position. The list is cost-only:
    a longer one changes no value.
    """
    __slots__ = ()


class Piece(namedtuple('Piece', 'shape face imgs dead')):
    """One piece of :class:`GammaPlan`: shape index, face name (or ``'arc'``), offsets, dead.

    ``dead`` -- the face's own images cover it end to end, the union of its self runs being
    its full length -- is the planner's word, written for ``empty``'s sake and read by it and
    by the staleness check. :func:`_piece_field` never reads it: a covered face cancels
    through the pair records, run by run, so a partly covered face (``dead = False``) and a
    fully covered one are built by the same arithmetic. An arc is never dead.
    """
    __slots__ = ()


class Pair(namedtuple('Pair', 'p q m n flush')):
    """One coincidence in :class:`GammaPlan`: piece ``q`` at offset ``(m, n)`` shares ``p``'s line.

    ``flush`` says the two are the *same segment* -- no slide, equal half-lengths, both at
    rounding scale -- and not merely the same line over a run: a bar spanning the cell against
    its own image, at any angle. Its replay then takes the partner's tangential expression
    from the primary as well as its normal one, so the two distances agree bitwise everywhere
    rather than over the run alone, and the cancellation is exact at every sample -- which is
    what a quarter turn of the grating needs, where ``cos(pi/2)`` puts a 4e-14 slide on a
    segment that has none.
    """
    __slots__ = ()




def _cell_diameter(a1x, a1y, a2x, a2y):
    """The longer diagonal of the cell -- how far a point of it can sit from its centre, doubled."""
    return max(math.hypot(a1x + a2x, a1y + a2y), math.hypot(a1x - a2x, a1y - a2y))


def _ring_offsets(a1x, a1y, a2x, a2y, reach):
    """Every ``(m, n)`` with ``|m a1 + n a2| <= reach`` -- ``<=`` at the boundary, on purpose.

    ``|m a1 + n a2|`` is at least ``|m|`` times the height of ``a1`` over ``a2`` (and the same
    for ``n``), which bounds the search box; the test inside it is the Euclidean one.
    """
    det = abs(a1x * a2y - a2x * a1y)
    m_max = int(math.ceil(reach * math.hypot(a2x, a2y) / det))
    n_max = int(math.ceil(reach * math.hypot(a1x, a1y) / det))
    limit = reach * (1.0 + 1e-12)
    return tuple((m, n) for m in range(-m_max, m_max + 1) for n in range(-n_max, n_max + 1)
                 if math.hypot(m * a1x + n * a2x, m * a1y + n * a2y) <= limit)


def _arc_offsets(prim, a1x, a1y, a2x, a2y, diam, tol):
    """A round shape's instance list: its own alone inside its Wigner-Seitz cell, the ring outside.

    The ``'arc'`` piece reads the level set of the nearest *centre*
    (:func:`_round_field`), and that is the only level set whose interface enters the cell
    wherever the shape stays inside the Wigner-Seitz cell of its centre: then it and every
    image lie on opposite sides of the cell's bounding planes, so an image's boundary never
    enters the cell, and on every interface point in the cell the fold reads that interface's
    own instance. It does not make the nearest centre the nearest *boundary* off the
    interface. For a circle it is (see :func:`_round_field`); for an ellipse a neighbour's tip
    just across a bounding plane is often nearer a sample than the shape's own boundary --
    127 of 129 random single-image ellipses on a 600 nm square read a farther boundary, by
    up to 280 nm. This piece reads its own instance's level set there by design, a choice of
    extension; the tensor curve candidates, which carry the nearest boundary, search the 3x3
    around the fold instead (:func:`~wavebit.rcwa._engine.tensor_shapes._curve_candidate`). Beside
    other pieces the piece's distance is also its weight in the blend: the single image's is
    the algebraic one at the nearest centre (:func:`_round_field`), the ring's past the
    threshold below the exact one, so ``sum(T)`` steps where the support crosses it, beside
    another shape as alone. ``docs/validation/rcwa/normal-field-known-limits.md`` (case L) records both
    steps, and why the exact distance over the 3x3, which removes the blended one, is not
    taken. The condition is the support function against every lattice vector, ``h(t) <= |t| / 2`` with
    ``h(t) = sqrt((a t.e1)^2 + (b t.e2)^2)`` -- which only a vector shorter than ``2 R`` can
    fail, so that is the ring swept. A shape that leaves the cell (an ellipse 720 x 200 at 10
    deg on a 700 square: ``h = 354.9 > 350`` along ``a1``) has part of a neighbour's boundary
    *inside* the cell, and the fold read the wrong level set there -- 7.0 deg mean on Gamma,
    89.9 max, 3.09 deg / 3.73% over the band -- so it is carried over a ring instead, with the
    exact distance to each
    instance (:func:`_round_instances`). The ring is every offset ``s`` with
    ``|A s| <= diam + min(a, b) + h(s)``: the boundary of the shape's own instance is within
    ``diam / 2 + min(a, b)`` of any sample of the folded cell, the instance at ``s`` reaches
    ``h(s)`` back towards it, and the fold adds ``diam / 2`` again -- so an instance farther
    out can never be the nearest, and the list is cost-only (13 offsets on the 720 x 200
    scene by size alone, 9 by support). Each instance costs an exact distance, so the
    difference is paid on every build.

    A circle can never leave: ``h = R`` in every direction, and ``R > |t| / 2`` is the overlap
    ``check_layer_shapes`` refuses. That is a statement about the geometry, true on any
    spelling of the basis; the fold *finding* the nearest centre is a statement about the
    arithmetic, true on the reduced one, which is the one it is handed since 2026-09-06
    (:func:`_fold_to_nearest_image`). The threshold steps like every combinatorial fact of the
    plan -- a frozen record holds it, and a shape grown across it is refused by name -- and
    the tolerance is the collinearity one, rounding scale, so a disc exactly filling its cell
    keeps its single instance on either side of the rounding.
    """
    _, _, _, a, b, _, _ = prim

    def support(m, n):
        tx, ty = m * a1x + n * a2x, m * a1y + n * a2y
        length = math.hypot(tx, ty)
        return length, _overlap._support_radius(prim, tx / length, ty / length)

    leaves = any(h > 0.5 * length + tol
                 for length, h in (support(m, n)
                                   for m, n in _ring_offsets(a1x, a1y, a2x, a2y, 2.0 * max(a, b))
                                   if (m, n) != (0, 0)))
    if not leaves:
        return ((0, 0),)
    ring = []
    for m, n in _ring_offsets(a1x, a1y, a2x, a2y, diam + a + b):
        if (m, n) == (0, 0):
            ring.append((m, n))
            continue
        length, h = support(m, n)
        if length <= (diam + min(a, b) + h) * (1.0 + 1e-12):
            ring.append((m, n))
    return tuple(ring)


def plan_gamma(shapes, lattice):
    """The :class:`GammaPlan` of one layer, or ``None`` when the parameters are not concrete.

    Eager NumPy on ``checked_float`` primals, like every geometry check on this path: a
    ``jax.grad`` tracer carries a value and is planned per call, a ``jax.jit`` tracer does not
    and gets ``None``, which the caller answers with :func:`_blind_plan` until the frozen
    record reaches here.

    The sweep is :func:`~wavebit._common.geometry.overlap.face_pairs` over the ring; coverage is the
    **union** of a face's self runs (:func:`~wavebit._common.geometry.overlap.covered_length`); the
    tolerances are the rounding-scale ones that module states. Nothing about a material is
    read, and nothing about a position survives: the same structure named one period over
    plans identically. Neither does the spelling of the basis: the ring, the offsets and the
    pairs are counted on the Lagrange-reduced basis (:func:`~wavebit.lattice.reduced_basis`)
    and the transform to it is part of the plan, so ``a2 = 10 a1 + (50, 700)`` plans as
    ``(50, 700)`` does -- ring 3, the pair at ``(0, 1)``, 21 offsets per face -- where it
    used to plan ring 14, the pair at ``(-10, 1)`` and 479 offsets, all of them the same
    images under longer names.

    A layer holding a polygon has no plan here and is refused rather than planned without it:
    the pieces are a primitive's faces and arcs, and a polygon's slot would be silently empty.
    It is built on the boundary route, whose interface plan is part of the painting's record.
    """
    if any(is_polygon(shape) for shape in shapes):
        raise ValueError('a layer holding a polygon has no closed-form interface plan: it is '
                         'built on the boundary route, whose plan is the painting record')
    prims = _primitives(shapes, lattice)
    if prims is None:
        return None
    try:
        transform = reduction_transform(lattice)
        reduced = reduced_basis(lattice, transform)
        a1x, a1y, a2x, a2y = _cell_floats(reduced)
    except Exception:                                  # noqa: BLE001 - a traced basis
        return None
    scale = max(math.hypot(a1x, a1y), math.hypot(a2x, a2y))
    det = abs(a1x * a2y - a2x * a1y)
    h_min = det / scale
    diam = _cell_diameter(a1x, a1y, a2x, a2y)
    tol = _overlap.COLLINEAR_TOL_REL * scale

    radii = [0.0 if p is None else _overlap._bounding_radius(p) for p in prims]
    ring = 1 + int(math.ceil(2.0 * max(radii, default=0.0) / h_min))
    records = _overlap.face_pairs(prims, reduced, ring, tol)

    pieces, index = [], {}
    for i, prim in enumerate(prims):
        if prim is None:
            continue
        imgs = _ring_offsets(a1x, a1y, a2x, a2y, diam + 2.0 * radii[i])
        if prim[0] != _overlap.BOX:
            index[(i, 0)] = len(pieces)
            pieces.append(Piece(i, 'arc', _arc_offsets(prim, a1x, a1y, a2x, a2y, diam, tol),
                                False))
            continue
        halves = [face[6] for face in _overlap.box_faces(prim)]
        for f, name in enumerate(_overlap.FACES):
            runs = [(c.lo, c.hi) for c in records if c.i == i and c.f == f and c.j == i]
            dead = _overlap.covered_length(runs, tol) >= 2.0 * halves[f] - tol
            index[(i, f)] = len(pieces)
            pieces.append(Piece(i, name, imgs, dead))

    pairs = []
    for c in records:
        # One record per coincidence. Each is swept from both of its faces, and the mirror
        # of (i, f) against (j, g) at (m, n) is (j, g) against (i, f) at (-m, -n): keep the
        # orientation from the lower shape, and within a shape from the `+` face.
        if c.j < c.i or (c.j == c.i and c.g < c.f):
            continue
        flush = abs(c.slide) <= tol and abs(c.h_p - c.h_q) <= tol
        pairs.append(Pair(index[(c.i, c.f)], index[(c.j, c.g)], c.m, c.n, flush))

    empty = all(pc.face != 'arc' and pc.dead for pc in pieces)
    return GammaPlan(ring, tuple(pieces), tuple(pairs), empty, transform)


def _blind_plan(shapes, lattice):
    """A plan taken without reading a shape: every piece, a ring from the basis, and **no pairs**.

    What a jitted solve without a frozen record gets. The pieces need only the shape *types*;
    the ring needs the basis, and it is read as numbers -- a traced basis is **refused** with
    :data:`_NEEDS_A_FREEZE`, exactly as :func:`layer_needs_resolving` refuses a traced
    multi-shape layer. It used to fall back to a fixed 7x7 ring, with no argument that the
    ring held every instance the planned list holds, and on an accepted unreduced basis it
    did not: ``a2 = 10 a1 + (50, 700)`` is the same cell as ``a2 = (50, 700)``, which
    ``as_lattice`` accepts without reducing, and the planned list per piece there is 479
    offsets out to ring 14, 460 of them outside the 7x7. The two fields disagreed on 43662 of
    65536 samples at N = 256 (max 90 deg) and the jitted solve sat 6e-3 in ``sum(T)`` from the
    eager one, with nothing said. A traced basis with no record now raises, and the message
    names the freeze; with a record the planned list replays under the trace as it always did.

    What the readable basis gets is every offset within ``2 * diam`` of the fold, ``diam``
    the longer diagonal of the **reduced** cell (:func:`~wavebit.lattice.reduced_basis`;
    the transform rides in the plan, as it does in a planned one). The planned list is bounded
    by ``diam + 2 R`` (:class:`GammaPlan`), so this is a superset of it whenever ``R <= diam
    / 2``: every box or ellipse that fits *inside* its cell, since the diagonal of such a shape
    is at most the cell's. It is not a theorem about every accepted shape -- a thin bar turned
    to thread between its images can be far longer than the cell's diagonal without meeting
    one, and its ``R`` exceeds ``diam / 2`` -- and nothing here can read the size that would
    say so. Such a shape under ``jax.jit`` with no record is built on the short ring; freezing
    first is what gives it the planned one.

    What the ring costs is set by that diagonal, and reducing the basis is what keeps it a
    property of the cell rather than of its spelling: on ``a2 = 10 a1 + (50, 700)`` the ring
    was 1559 offsets per face before the reduction and a jitted blind build of a lone box did
    not finish compiling in four minutes; reduced it is the 25 of ``a2 = (50, 700)`` and
    compiles in a second. A cell that is *genuinely* elongated pays for it: on ``a1 = (700,
    0)``, ``a2 = (0, 7000)`` -- reduced already, since the two are orthogonal -- the diagonal
    is 7035, the reach 14070 and the blind ring 121 offsets per face, four faces per box, where
    the planned list of a 350 box on it is 35 by size (45 for a 500 x 700 one). That is the
    price of not reading the size, and a record is the way not to pay it.

    What no blind plan can have is the pair table, which needs the numbers. So an abutting
    pair -- a bar spanning the cell, a box against its own image -- is not told that it is
    one, and its two faces cancel only to rounding: the same values the planned field carries,
    minus the exact zero. That is a field within ulps of the planned one everywhere, with no
    tie rule at the points where the planned field would have needed one. A scene that leans
    on the tie -- Li's checkerboard on a jitted trace -- should freeze its geometry first.

    **An ellipse is refused here, a circle is not.** Whether a round shape leaves its
    Wigner-Seitz cell (:func:`_arc_offsets`) is a question of its *size*, and the size is
    what a trace cannot read; only the type is. A circle never leaves -- one of diameter
    ``<= P`` sits inside the cell of its centre on any basis, and a larger one is refused
    eagerly as an overlap -- so its single instance is the planned list and it builds. That
    eager refusal is the whole of what bounds the circle: under ``jax.jit`` with a traced
    ``lx`` the overlap check cannot fire, so a jitted sweep that carries a circle past ``P``
    builds an overlapping structure on the single instance with nothing said (a 750 circle on
    the 700 square built ``sum(T)`` 0.928891 where the eager solve refuses) -- a sweep must
    not carry it past ``P``, and re-taking the freeze on a schedule, as the README says
    (every step), is what catches it. An ellipse can leave, and until 2026-09-06 one that did was built here on its single
    instance, which is the fold's field the ring replaced: a 720 x 200 ellipse at 10 deg on a
    700 square under ``jax.jit`` with no record solved -8.2e-3 in ``sum(T)`` from the eager
    one at order 4 (-7.1e-3 at order 6), bit-identical to a NumPy solve with the plan forced
    to ``None``, and nothing said so. Now it raises with :data:`_NEEDS_A_FREEZE`, the same
    answer a traced basis gets one paragraph up: what cannot be read is refused, not guessed.
    With a record the planned list -- one instance or the ring -- replays under the trace.
    """
    try:
        transform = reduction_transform(lattice)
        a1x, a1y, a2x, a2y = _cell_floats(reduced_basis(lattice, transform))
    except Exception:                                  # noqa: BLE001 - a traced basis
        raise ValueError(_NEEDS_A_FREEZE.format(
            what='the ring of periodic images each face is evaluated over, which the cell '
                 'bounds,')) from None
    imgs = _ring_offsets(a1x, a1y, a2x, a2y, 2.0 * _cell_diameter(a1x, a1y, a2x, a2y))
    pieces = []
    for i, shape in enumerate(shapes):
        stype = shape.get('type', 'rectangle')
        if stype in _overlap._BOX_TYPES:
            pieces.extend(Piece(i, name, imgs, False) for name in _overlap.FACES)
        elif stype == 'ellipse':
            raise ValueError(_NEEDS_A_FREEZE.format(
                what=f'whether shape {_shape_name(shapes, i)} leaves the Wigner-Seitz cell '
                     f'of its centre, which decides whether its level set is read at one '
                     f'instance or over a ring of images and is a question of its size,'))
        elif stype in _overlap._ELLIPSE_TYPES:
            pieces.append(Piece(i, 'arc', ((0, 0),), False))
    return GammaPlan(3, tuple(pieces), (), False, transform)


# ---------------------------------------------------------------------------- the replay

def _material_slots(xp, shapes, dtype):
    """Each shape's material slot as a 0-d array -- what a seam's contrast reads.

    The slot's ``epsilon``, in both passes. In the permittivity pass that is the permittivity;
    in the permeability pass it is the permeability, because ``_mu_structure`` hands this
    builder a *view* whose material slots already hold ``mu_value`` -- ``1.0`` for a material
    that named none, by the type's own contract -- so one reader serves both quantities and the
    interface set is decided per quantity, as the raster path's twin records already are.
    Preserve whether the input is real or complex, a dtype fact available under a trace.
    Real inputs keep the real field and three projector components; complex inputs retain
    the material difference's phase and need four Hermitian components. No material value
    selects a branch. The background never enters, since a seam between two shapes is an
    interface iff *they* differ.
    """
    cdtype = xp.complex64 if dtype == xp.float32 else xp.complex128
    values = [as_material(shape.get('material', 1.0), 'shape material').epsilon
              for shape in shapes]
    # Infer only the kind with the temporary conversion; cast the original value to avoid
    # first rounding Python floats to a backend's default precision (notably Torch float32).
    return [as_array(xp, value, dtype=cdtype if xp.isdtype(as_array(xp, value).dtype,
                                                         'complex floating') else dtype)
            for value in values]


class _Frame:
    """One box's traced geometry against the sample points, in the box's own frame.

    ``U``, ``V`` are the samples' coordinates along the box's axes measured from its centre,
    **not folded**; ``km``, ``kn`` are the integer fold -- which lattice cell the sample sits in
    relative to the centre, by rounding the fractional coordinates. An instance of the box is
    named by its *absolute* image ``K = fold + offset``, and :meth:`at` gives the sample's
    coordinates relative to that instance as ``U - K . (A e_u)``: one expression for every
    reader. Two pieces that name the same absolute image -- a face, and the face of another
    box that lies on its line -- therefore compute bitwise the same numbers whatever fold each
    started from, which is what lets a cross pair cancel exactly without the two shapes
    sharing a fold. (The partner offset between two folds *is* array-valued, as the previous
    commit noted; written this way it cancels out of the arithmetic rather than needing a
    ``where``.) The fold still bounds the instance list: the residual after rounding sits
    within half the cell's diameter of the centre, so the size-only ring of
    :class:`GammaPlan` is a superset of the nearest instance at every position -- on the
    reduced basis, which is what makes the rounded fold's residual at most half of a diameter
    that bounds the list (see :func:`_fold_to_nearest_image` for the bound the reduction buys).
    """
    __slots__ = ('xp', 'cos', 'sin', 'a', 'b', 'cx', 'cy', 'U', 'V', 'km', 'kn',
                 't1u', 't1v', 't2u', 't2v')

    def __init__(self, xp, X, Y, shape, lattice, reduced, dtype):
        # `lattice` fixes the origin a centre is measured from (as given); `reduced` is the
        # basis the fold and the instance offsets count cells of.
        _, cx, cy, lx, ly, angle = shape_geometry(shape, lattice)
        self.xp = xp
        self.cos = xp.cos(as_array(xp, angle, dtype=dtype))
        self.sin = xp.sin(as_array(xp, angle, dtype=dtype))
        self.a, self.b = lx / 2.0, ly / 2.0
        self.cx, self.cy = cx, cy
        a1x, a2x = reduced[0][0], reduced[0][1]
        a1y, a2y = reduced[1][0], reduced[1][1]
        det = a1x * a2y - a2x * a1y
        dx, dy = X - cx, Y - cy
        self.km = xp.round((a2y * dx - a2x * dy) / det)
        self.kn = xp.round((-a1y * dx + a1x * dy) / det)
        self.U = dx * self.cos + dy * self.sin
        self.V = -dx * self.sin + dy * self.cos
        self.t1u, self.t1v = a1x * self.cos + a1y * self.sin, -a1x * self.sin + a1y * self.cos
        self.t2u, self.t2v = a2x * self.cos + a2y * self.sin, -a2x * self.sin + a2y * self.cos

    def at(self, Km, Kn):
        """``(du, dv)``: the samples relative to the instance at absolute image ``(Km, Kn)``."""
        return (self.U - Km * self.t1u - Kn * self.t2u,
                self.V - Km * self.t1v - Kn * self.t2v)

    def parts(self, face, du, dv, zero):
        """``(squared tangential excess, squared normal coordinate, tangential coordinate)``.

        Of one face, at the instance whose relative coordinates are ``(du, dv)`` -- the ONE
        spelling of each, so a primary and its dependent compute bitwise the same numbers. The
        u faces run along v with half-length ``b``; the tangential coordinate is measured from
        the face's midpoint, which is where a run's bounds are stated.
        """
        xp = self.xp
        if face[1] == 'u':
            sigma = dv
            excess = xp.maximum(xp.abs(dv) - self.b, zero)
        else:
            sigma = du
            excess = xp.maximum(xp.abs(du) - self.a, zero)
        t = {'+u': du - self.a, '-u': du + self.a, '+v': dv - self.b, '-v': dv + self.b}[face]
        return excess * excess, t * t, sigma

    def midpoint(self, face):
        """The face's midpoint in the cell frame -- the centre plus the half-length along the normal."""
        ux, uy = self.cos, self.sin
        vx, vy = -self.sin, self.cos
        return {'+u': (self.cx + self.a * ux, self.cy + self.a * uy),
                '-u': (self.cx - self.a * ux, self.cy - self.a * uy),
                '+v': (self.cx + self.b * vx, self.cy + self.b * vy),
                '-v': (self.cx - self.b * vx, self.cy - self.b * vy)}[face]


def _run_window(xp, pframe, p_face, qframe, q_face, m, n, reduced, dtype):
    """``(lo, hi)``: the run a cross pair shares, along ``p``'s tangent from ``p``'s midpoint.

    The plan records the pair by *name*; the bounds are re-derived here from the traced
    geometry as the max / min of projections, the same arithmetic
    :func:`~wavebit._common.geometry.overlap.face_pairs` used on the concrete numbers. ``q`` sits at
    lattice offset ``(m, n)`` relative to ``p``'s own instance, counted on the reduced basis
    the plan was taken on.
    """
    a1x, a2x = reduced[0][0], reduced[0][1]
    a1y, a2y = reduced[1][0], reduced[1][1]
    pmx, pmy = pframe.midpoint(p_face)
    qmx, qmy = qframe.midpoint(q_face)
    dx = qmx + (m * a1x + n * a2x) - pmx
    dy = qmy + (m * a1y + n * a2y) - pmy
    if p_face[1] == 'u':
        tx, ty, h_p = -pframe.sin, pframe.cos, pframe.b        # the u faces run along v
    else:
        tx, ty, h_p = pframe.cos, pframe.sin, pframe.a
    h_q = qframe.b if q_face[1] == 'u' else qframe.a
    slid = tx * dx + ty * dy
    lo = xp.maximum(as_array(xp, -h_p, dtype=dtype), as_array(xp, slid - h_q, dtype=dtype))
    hi = xp.minimum(as_array(xp, h_p, dtype=dtype), as_array(xp, slid + h_q, dtype=dtype))
    return lo, hi


def _piece_field(xp, X, Y, shapes, plan, slots, lattice, dtype, dyadic=False):
    """The normal field at ``(X, Y)`` from the plan's pieces -- ``(normal_x, normal_y)``.

    ``dyadic=True`` selects the complex shape projector and returns
    ``(P_xx, P_xy, P_yy, P_yx)`` instead. It combines positive dyads before
    normalization and ignores signed-pair substitutions. The description below
    concerns the default linear field.

    **One blend over every piece of every shape.** A box is its four faces and a round shape
    one piece; each piece has a distance (:class:`_Frame` for a face over its instances,
    :func:`_round_field` for an arc) and a direction, and they are combined by the ratio
    weights ``w = where(d == d_min, 1, (d_min / d) ** 4)`` in one sum. There is no per-shape
    stage: a two-stage blend switches extension where two shapes come into contact, which was
    measured as a 1.1e-2 step in ``sum(T)`` at order 6, and the one-stage sum has nothing to
    switch. A lone round shape inside its Wigner-Seitz cell is one piece and returns its own
    field, as it always did; one that leaves it is still **one** piece, whose distance and
    direction are the nearest instance's of its ring by a hard minimum
    (:func:`_round_instances`: the exact distance to every instance, and the level-set normal
    of the one that wins). The switch between two instances sits on the Voronoi boundary
    between their boundaries, where the exact nearest-interface field switches too, so it is
    not a seam of the field's making: on the 720 x 200 ellipse's own boundary and its
    neighbours' inside the cell the piece reads 2.6e-7 deg mean and 1.7e-6 max over 4001
    points, with no null. A face takes the nearest of its instances the same way.

    **A piece's distance is the segment distance to the nearest of its instances.** For a face
    with tangential coordinate ``sigma`` and normal coordinate ``nu`` at instance ``K``,
    ``d_K = hypot(max(0, |sigma| - h), nu)`` and ``d = min_K d_K`` over the plan's image list:
    a superset of candidates that contains the minimiser is the exact periodic distance.

    **A piece's direction is its outward normal, everywhere** -- not the foot direction past
    the extent, not the corner-radial rule the fold's box branch used. Measured: the foot rule
    breaks the 1-D grating (``max|n_y|`` 0.76 where it must be 0, because a covered face and
    its partner share endpoints and their foot directions reinforce instead of cancelling) and
    puts a 4e-4 non-decaying step at every merge; fixed normals give the grating exactly, a
    zero step at every merge, and the sheared and tilted scenes at 0.19 deg against the exact
    oracle. The price is ~4e-4 in ``sum(T)`` off the interface on a lone box at order 6.

    **A pair cancels or reinforces by arithmetic, and the materials decide which.** The plan
    says which face's instance lies on which face's line, opposed. The partner's normal
    coordinate is computed from the *primary's* expression at the coincident absolute image --
    bitwise the same value -- so where both are the nearest instance of their piece the two
    weights are equal. A self pair (the shape against its own image) is then an exact zero
    through the box's own axis sum. The substitution is made per *record*, inside that
    record's own run window (:func:`_run_window` on the traced geometry, as for a cross
    pair): one face can be covered from both ends by two different images -- a box on
    ``a1 = (700, 0)``, ``a2 = (400, 700)``, or a grating written as a box on a hex cell -- and
    each covered run has to read the primary at *its* coincident instance. A flush record is the
    same segment and substitutes everywhere, tangential excess included; that is what a
    quarter-turned grating needs. A cross pair (two shapes) cancels over the run it shares,
    whatever the materials: over the run the partner's normal coordinate is the primary's,
    bitwise, so the two weights are one number and the opposed normals sum to an exact zero.
    Until 2026-09-12 a transfer also moved the partner's weight to the primary's axis, one on
    the run and ``n2/(n2+e**2)`` off the line beyond its ends (the 2026-09-07 fix of a value
    step along the ray extending each run endpoint, carrying the material sign); once every
    record cancelled it took the same amount from both members, opposed on one line, and
    changed the sum by nothing but rounding, so it was removed -- the run's end is continuous
    without it, because nothing in the sum reads the run beyond the substitution.

    **A seam between different materials is put back by its contrast and material phase.**
    After the signed sum, every opposed pair of pieces of different owners adds
    ``c * h * max(0, -n_a.n_b) * q * (n_a - n_b)``, where
    ``c = (|eps_b - eps_a| / (|eps_a| + |eps_b|)) ** _BLEND_EXPONENT``,
    ``h = 2 w_a w_b / (w_a + w_b)`` and ``q = (eps_b - eps_a) / |eps_b - eps_a|``.
    At equality q is zero; c*q tends continuously to zero as
    ``(eps_b-eps_a) * |eps_b-eps_a|**(p-1) / (|eps_a|+|eps_b|)**p``. The fourth power
    keeps the seam's shrinking band linear in the material difference, as measured by the
    equality ramp against the explicit Li rule. Both zero slots give zero contrast.

    The material phase replaces the 2026-09-11 ranking by |eps| (2026-09-13). That ranking
    flipped a finite correction when unequal materials crossed equal magnitudes: at order 2,
    ``eps_a=4``, ``eps_b=(2+sqrt(12)*i)*(1+-h)``, h=1e-8, the transmission jumped 4.74e-4;
    at ``4 | -4`` it jumped 0.0358. A real scalar orientation cannot distinguish every complex
    material difference continuously without a zero or switch. The complex phase is
    antisymmetric under exchanging owners, as is their normal difference, so their product
    is listing-invariant. Real inputs retain real arithmetic, and positive real materials
    recover the old orientation; complex inputs retain both parts of the vector.

    On a live seam both weights are one and ``on_gamma`` zeroes all other pieces. The vector
    is then a nonzero complex multiple of the real interface normal. Hermitian normalization
    and the projector ``n n^H`` remove its phase: the exact interface normal projector remains.
    :func:`normal_projector_coeffs` carries both xy and yx components for a complex field,
    rather than dropping its imaginary part or assuming a symmetric projector. The tiled
    path preserves that dtype. At a positive gap the same pointwise formula is used, with no
    material rank or pair-record switch; equal materials add exactly zero.

    The corrected positive-gap study reads K's contact change at about 1e-12 over orders
    4..12, and its material-equality ramp follows Li within 4% from 1e-7 to 1e-2. Its original
    opening helper moved the boxes inward; the old ladder and matrix logs are labelled as
    overlap-side measurements. Current evidence and finite-order limitations are in
    ``docs/validation/rcwa/normal-field-known-limits.md``, section 8e, and ``studies/rcwa/nvf_seam_contrast/nvf_seam_contrast.py``.

    **The tie.** Where the sum is exactly zero -- a cancelled pair's line (a run its partner
    covers), the midline of a surviving pair -- the direction is the axis of the pair that did
    *not* cancel at that sample, read off the records by array equality of the members'
    distances: for each box in turn, the axis whose records did not cancel where the other
    axis's did, and ``(1, 0)`` when both cancelled or neither did, for every box. Sign is
    free because the factorization reads ``n n^T``. Li's checkerboard samples that line once
    per row however fine the grid, and the answer there is ``v``, the axis that has an
    interface.

    **A lone box's centre is a tie in exact arithmetic only.** The tie fires when the four
    face distances are bitwise equal, which the exact centre gives and a sample one ulp off
    it does not: there the sum of four weights within ulps of one reads whichever direction
    the rounding leaves. Measured on the 350 box at the sample ``(-5.7e-14, -5.7e-14)`` from
    the centre, the grid point of a verifier's 175-point sweep: ``(-0.707, -0.707)`` on
    ``a2 = (50, 700)`` and on the square, ``(0, -1)`` on ``a2 = (7050, 700)`` -- that cell's
    centre is 3875, whose ulp swallows the x offset, so only ``dy`` survives -- and ``(1, 0)``,
    the documented answer, at the exact centre on every one of them. One sample of measure
    zero, recorded as an observation and not fixed: the fix would be a tolerance on the tie,
    a width in a rule that has none.

    ``slots`` is the layer's material slot values (:func:`_material_slots`), computed from the
    shapes when ``None``; self pairs never read one.
    """
    if dyadic:
        check_dyadic_shapes(shapes)
    if slots is None:
        slots = _material_slots(xp, shapes, dtype)
    zero = xp.zeros_like(X)
    one = xp.ones_like(X)
    index = {(pc.shape, pc.face): k for k, pc in enumerate(plan.pieces)}
    # The basis every image is counted on: the plan's integers applied to the basis as given,
    # traced or not, so a jitted replay on a traced cell reduces without reading it.
    reduced = reduced_basis(lattice, plan.reduction)

    if not dyadic and len(plan.pieces) == 1 and plan.pieces[0].face == 'arc':
        shape, imgs = shapes[plan.pieces[0].shape], plan.pieces[0].imgs
        if len(imgs) == 1:
            _, gx, gy = _round_field(xp, X, Y, shape, lattice, reduced, dtype)
        else:
            _, gx, gy = _round_instances(xp, X, Y, shape, lattice, reduced, dtype, imgs)
        return gx, gy

    frames = {}
    for pc in plan.pieces:
        if pc.face != 'arc' and pc.shape not in frames:
            frames[pc.shape] = _Frame(xp, X, Y, shapes[pc.shape], lattice, reduced, dtype)

    # The pair records by role. A self pair substitutes within the box's own instance table,
    # over its own run (a flush one over the whole line); a cross pair carries a run window
    # from the traced values, and its materials' equality for the tie below -- the sum reads
    # no sign from a record since 2026-09-11 (the seam contrast, 3b).
    self_records, cross_by_q, windows, equalities = {}, {}, {}, {}
    for r, pr in enumerate(() if dyadic else plan.pairs):
        p, q = plan.pieces[pr.p], plan.pieces[pr.q]
        if p.shape == q.shape:
            window = None if pr.flush else _run_window(
                xp, frames[p.shape], p.face, frames[q.shape], q.face, pr.m, pr.n, reduced, dtype)
            self_records.setdefault(p.shape, []).append((p.face, q.face, pr.m, pr.n, window))
            continue
        cross_by_q.setdefault(pr.q, []).append(r)
        windows[r] = _run_window(xp, frames[p.shape], p.face, frames[q.shape], q.face,
                                 pr.m, pr.n, reduced, dtype)
        equalities[r] = xp.all(slots[p.shape] == slots[q.shape])

    # 1. Every piece's distance at the nearest instance.
    d = [None] * len(plan.pieces)
    round_dir = {}
    for i, frame in frames.items():
        faces = {face: plan.pieces[index[(i, face)]] for face in _overlap.FACES}
        offsets = sorted({mn for pc in faces.values() for mn in pc.imgs})
        own = {(m, n): frame.at(frame.km + m, frame.kn + n) for m, n in offsets}
        records = self_records.get(i, ())
        for face, piece in faces.items():
            k = index[(i, face)]
            cross = cross_by_q.get(k, ())
            best = None
            for m, n in piece.imgs:
                du, dv = own[(m, n)]
                e2, n2, _ = frame.parts(face, du, dv, zero)
                # A self pair member's instance reads the primary's expression at the partner
                # offset: q's instance s coincides with p's instance s - (m, n), both on this
                # box's fold. Per RECORD, and inside that record's own run: one face can be
                # covered by two images from its two ends (a 600 x 700 box on
                # a1 = (700, 0), a2 = (400, 700) has its +v face under the (0, 1) image over
                # [100, 300] and under the (-1, 1) image over [-300, 0]), and the two primaries
                # lie on one line mathematically but their `at()` values differ by the lattice
                # vector's own-axis component, 3e-14 once the scene is turned. Keyed by
                # instance alone the second record overwrote the first, the covered run of the
                # first read the primary at the wrong instance, `on_gamma` handed the whole
                # weight to the member that happened to read 0.0, and the run carried a full
                # normal. A flush record is the same segment and substitutes everywhere.
                for p_face, q_face, mp, np_, window in records:
                    offset = (m - mp, n - np_)
                    if q_face != face or offset not in own:
                        continue
                    pe2, pn2, psigma = frame.parts(p_face, *own[offset], zero)
                    if window is None:
                        e2, n2 = pe2, pn2                   # flush: the same segment, bitwise
                    else:
                        lo, hi = window
                        n2 = xp.where((psigma >= lo) & (psigma <= hi), pn2, n2)
                for r in cross:
                    pr = plan.pairs[r]
                    p = plan.pieces[pr.p]
                    pframe = frames[p.shape]
                    # The primary's expression at the coincident absolute image: this box's
                    # fold plus this instance, less the pair's offset -- the same integers
                    # the primary's own evaluation names when this instance is the nearest.
                    # Over the run (a flush record everywhere) the partner's normal coordinate
                    # is the primary's, bitwise, which is what lets two opposed faces cancel
                    # to an exact zero. The 2026-09-07 transfer that also moved a partner's
                    # weight to the primary's axis here -- one on the run, ``n2/(n2+e**2)``
                    # off the line beyond its ends -- went on 2026-09-12: once every record
                    # cancels (3b) it took the same amount from both members, opposed on one
                    # line, and changed nothing but rounding.
                    pdu, pdv = pframe.at(frame.km + (m - pr.m), frame.kn + (n - pr.n))
                    pe2, pn2, sigma = pframe.parts(p.face, pdu, pdv, zero)
                    lo, hi = windows[r]
                    on_segment = (sigma >= lo) & (sigma <= hi)
                    if pr.flush:
                        e2, n2 = pe2, pn2
                    else:
                        n2 = xp.where(on_segment, pn2, n2)
                val = e2 + n2
                if best is None:
                    best = val
                else:
                    best = xp.where(val < best, val, best)
            # The root's radicand is guarded, not its result: a sample exactly on the face has
            # `best == 0`, where `d(sqrt)` is infinite and the cotangent `where` hands the
            # discarded branch, zero, makes `0 * inf` -- the `_hypot_safe` lesson. Taking the
            # root of 1 there and writing the 0 back leaves every value bitwise where it was.
            on_face = best == 0.0
            d[k] = xp.where(on_face, zero, xp.sqrt(xp.where(on_face, one, best)))
    round_dyads = {}
    for k, pc in enumerate(plan.pieces):
        if pc.face != 'arc':
            continue
        if dyadic:
            d[k], aa, bb, cc = _round_dyad(
                xp, X, Y, shapes[pc.shape], lattice, reduced, dtype, plan.ring)
            round_dyads[k] = (aa, bb, cc)
            continue
        if len(pc.imgs) == 1:
            d[k], gx, gy = _round_field(xp, X, Y, shapes[pc.shape], lattice, reduced, dtype)
        else:
            d[k], gx, gy = _round_instances(xp, X, Y, shapes[pc.shape], lattice, reduced, dtype,
                                            pc.imgs)
        round_dir[k] = (gx, gy)

    # 2. The ratio weights, `(d_min / d) ** 4`. The winner's ratio is `d / d`, exactly 1.0 in
    #    IEEE arithmetic, so no `where` picks it out -- and that is a derivative, not a
    #    nicety. The previous spelling, `where(d == d_min, 1, ratio ** 4)`, gave every piece
    #    *tied* with the winner a weight of 1 with a zero derivative, where the ratio's
    #    derivative is `4 (d_min' - d') / d` and only vanishes for the winner itself. Such
    #    ties are not rare: a u face and a v face are equidistant on lines the grid samples
    #    (`5 j - 7 i = 97` on the sheared cell's grid), and freezing their weights put the
    #    lone sheared box's `d(sum T)/dlx` 0.5% from every central difference down to
    #    `h = 1e-6`, constant in `h` -- a wrong derivative, not a sawtooth. With the honest
    #    ratio the two tied weights' derivatives are antisymmetric, which is the central
    #    difference (and `minimum`'s balanced tie rule). The one guard left is the sample on
    #    an interface, `d_min == 0`, where the winner is 1 and the rest 0 by the limit; its
    #    divisor is 1 so the discarded branch never sees a zero.
    d_min = d[0]
    for dk in d[1:]:
        d_min = xp.minimum(d_min, dk)
    on_gamma = d_min == 0.0
    safe_min = xp.where(on_gamma, one, d_min)
    w = []
    for dk in d:
        on_face = dk == 0.0
        ratio = safe_min / xp.where(on_face, one, dk)
        w.append(xp.where(on_gamma, xp.where(on_face, one, zero), ratio ** _BLEND_EXPONENT))

    if dyadic:
        # Lift positive normal dyads into a complex basis. Opposite face normals
        # have the same dyad: no signed-vector cancellation or zero-norm tie.
        a, b, c = zero, zero, zero
        for i, frame in frames.items():
            u = w[index[(i, '+u')]] + w[index[(i, '-u')]]
            v = w[index[(i, '+v')]] + w[index[(i, '-v')]]
            a = a + u * frame.cos**2 + v * frame.sin**2
            b = b + (u-v) * frame.cos * frame.sin
            c = c + u * frame.sin**2 + v * frame.cos**2
        for k, (aa, bb, cc) in round_dyads.items():
            a, b, c = a + w[k]*aa, b + w[k]*bb, c + w[k]*cc
        if round_dyads:
            # An isotropic extension makes a curved primitive's zero gradient
            # (its centre) regular. It vanishes on Gamma, where the pole's
            # normal dyad alone survives. This is part of the field definition,
            # not a numerical tolerance or a perturbation of material geometry.
            scale = _reduced_cell_height(xp, reduced, dtype)
            isotropic = (d_min / scale)**_BLEND_EXPONENT
            a, c = a + isotropic, c + isotropic
            # P is invariant under a positive scalar on M. Scale before
            # squaring, so small inclusions in float32 do not underflow norm2.
            peak = xp.maximum(a, c)
            underflow = peak == 0.
            divisor = xp.where(underflow, one, peak)
            a = xp.where(underflow, one, a / divisor)
            b = b / divisor
            c = xp.where(underflow, one, c / divisor)
        norm2 = a*a + 2*b*b + c*c
        # M is positive semidefinite with positive trace: a face contributes,
        # or a curved pole contributes on Gamma and its isotropic extension
        # contributes off Gamma. Thus norm2 >= trace(M)**2/2 > 0.
        xx = (a*a+b*b)/norm2
        yy = (c*c+b*b)/norm2
        xy = ((a+c)*b + 1j*(b*b-a*c))/norm2
        return xx, xy, yy, xp.conj(xy)

    # 3. The sum. Every face contributes along its own outward normal, and each box's two
    #    opposed faces are differenced *before* the rotation so that a self pair's cancellation
    #    is an exact zero on the box's own axis; a cross pair's two faces carry one weight over
    #    their run (step 1) and cancel across the boxes the same way.
    sx, sy = zero, zero
    for i, frame in frames.items():
        wu = w[index[(i, '+u')]] - w[index[(i, '-u')]]
        wv = w[index[(i, '+v')]] - w[index[(i, '-v')]]
        sx = sx + (wu * frame.cos - wv * frame.sin)
        sy = sy + (wu * frame.sin + wv * frame.cos)
    for k, (gx, gy) in round_dir.items():
        sx = sx + w[k] * gx
        sy = sy + w[k] * gy

    # 3b. The seam contrast. Every opposed pair cancelled above, at any gap and over a shared
    #     run through its record; a seam between two DIFFERENT materials is put back here,
    #     co-oriented, in proportion to the contrast -- see the docstring. Pointwise in the
    #     weights, so the tiled path is bit-identical to the whole-grid one; it reads no
    #     record, so a positive gap and a contact are one formula. Equal materials add exactly
    #     zero and leave the field bit for bit.
    if slots is not None and not dyadic and len(plan.pieces) > 1:
        dirs = {}
        for i, frame in frames.items():
            dirs[index[(i, '+u')]] = (frame.cos * one, frame.sin * one)
            dirs[index[(i, '-u')]] = (-frame.cos * one, -frame.sin * one)
            dirs[index[(i, '+v')]] = (-frame.sin * one, frame.cos * one)
            dirs[index[(i, '-v')]] = (frame.sin * one, -frame.cos * one)
        for k, g in round_dir.items():
            dirs[k] = g
        pieces = plan.pieces
        for a in range(len(pieces)):
            for b in range(a + 1, len(pieces)):
                if pieces[a].shape == pieces[b].shape or a not in dirs or b not in dirs:
                    continue
                ea, eb = slots[pieces[a].shape], slots[pieces[b].shape]
                difference = xp.abs(eb - ea)
                scale = xp.abs(ea) + xp.abs(eb)
                contrast = difference / xp.where(scale > 0.0, scale, xp.ones_like(scale))
                contrast = xp.astype(contrast, dtype) ** _BLEND_EXPONENT
                nax, nay = dirs[a]
                nbx, nby = dirs[b]
                dot = nax * nbx + nay * nby
                opposed = xp.where(dot < 0.0, -dot, zero)
                total = w[a] + w[b]
                mean = 2.0 * w[a] * w[b] / xp.where(total > 0.0, total, one)
                # The material difference supplies an antisymmetric complex phase. Ranking
                # |eps| flips a finite correction when distinct materials have equal moduli.
                # Here contrast*phase tends continuously to zero at material equality, and
                # exchanging the owners reverses both phase and normal difference. On Gamma
                # the surviving vector is a complex scalar times the real interface normal;
                # its Hermitian projector removes that phase exactly.
                phase = (eb - ea) / xp.where(difference > 0.0, difference,
                                              xp.ones_like(difference))
                corr = contrast * mean * opposed * phase
                sx = sx + corr * (nax - nbx)
                sy = sy + corr * (nay - nby)

    # 4. The tie.
    tie = (sx == 0.0) & (sy == 0.0)
    cancelled = {}
    for r, pr in enumerate(plan.pairs):
        p, q = plan.pieces[pr.p], plan.pieces[pr.q]
        same = d[pr.p] == d[pr.q]
        if r in windows:
            same = same & equalities[r]
        for key in ((p.shape, p.face[1]), (q.shape, q.face[1])):
            cancelled[key] = same if key not in cancelled else (cancelled[key] | same)
    none = tie & ~tie
    axis_x, axis_y = one, zero
    for i in reversed(list(frames)):                 # the first box with an answer wins
        frame = frames[i]
        c_u = cancelled.get((i, 'u'), none)
        c_v = cancelled.get((i, 'v'), none)
        pick_u = c_v & ~c_u
        pick_v = c_u & ~c_v
        axis_x = xp.where(pick_u, frame.cos * one, xp.where(pick_v, -frame.sin * one, axis_x))
        axis_y = xp.where(pick_u, frame.sin * one, xp.where(pick_v, frame.cos * one, axis_y))
    sx = xp.where(tie, axis_x, sx)
    sy = xp.where(tie, axis_y, sy)
    norm = (_hypot_safe(xp, xp.abs(sx), xp.abs(sy))
            if xp.isdtype(sx.dtype, 'complex floating') else _hypot_safe(xp, sx, sy))
    return sx / norm, sy / norm


def check_dyadic_shapes(shapes):
    """Current complex shape projector support, shared by solve and freeze."""
    supported = _overlap._BOX_TYPES + _overlap._ELLIPSE_TYPES
    if any(is_polygon(s) for s in shapes):
        raise NotImplementedError(
            "nvf_projector='circular' requires rectangles, circles or ellipses: it corrects the "
            "primitives' own analytic extension near a curve, and a polygon has neither -- its "
            "edges are straight and its field is the boundary route's nearest-interface one. "
            "Leave nvf_projector unset (the linear projector) for a layer holding a polygon.")
    if any(s.get('type', 'rectangle') not in supported for s in shapes):
        raise NotImplementedError("nvf_projector='circular' requires rectangles, circles or ellipses")


def _reduced_cell_height(xp, reduced, dtype):
    a, b = [as_array(xp, v, dtype=dtype) for v in reduced[0]]
    c, d = [as_array(xp, v, dtype=dtype) for v in reduced[1]]
    return xp.abs(a*d-b*c) / xp.sqrt(xp.maximum(a*a+c*c, b*b+d*d))


def _round_dyad(xp, X, Y, shape, lattice, reduced, dtype, ring):
    """Periodic positive dyads of a curved level set, with compact image support.

    For f=(u/a)^2+(v/b)^2-1, use g=min(a,b)*grad(f), a pole distance
    min(a,b)*abs(f), and the positive tensor g g^T. Unlike a unit normal,
    g is regular at the centre. The caller adds an isotropic extension off Gamma.

    Each image is tapered by (1-r^2/cutoff^2)_+^4, with cutoff=max(a,b)+h/2
    and h the smaller reduced-cell height. Every real boundary lies strictly
    inside this support. Images beyond the plan's ring have zero weight: folded
    fractional coordinates are in [-1/2,1/2], and a supported centre has each
    index at most 1+max(a,b)/h. The ring is 1+ceil(2*max(a,b)/h).
    Consequently image switching adds only identically zero terms, rather than
    changing the chosen normal. No nearest-image direction or centre tie is read.
    """
    _, cx, cy, lx, ly, angle = shape_geometry(shape, lattice)
    a, b = as_array(xp, lx/2., dtype=dtype), as_array(xp, ly/2., dtype=dtype)
    size = xp.minimum(a,b)
    cutoff = xp.maximum(a,b) + _reduced_cell_height(xp, reduced, dtype)/2.
    co = xp.cos(as_array(xp, angle, dtype=dtype))
    si = xp.sin(as_array(xp, angle, dtype=dtype))
    ax, bx = reduced[0]
    ay, by = reduced[1]
    det = ax*by-bx*ay
    dx, dy = X-cx, Y-cy
    km = xp.round((by*dx-bx*dy)/det)
    kn = xp.round((ax*dy-ay*dx)/det)
    zero, one = xp.zeros_like(X), xp.ones_like(X)
    terms = []
    # Include the caller's isotropic term in the common ratio scale. This
    # bounds its weight by one without changing the normalized projector.
    minimum = _reduced_cell_height(xp, reduced, dtype)
    for m in range(-ring,ring+1):
        for n in range(-ring,ring+1):
            x, y = dx-(km+m)*ax-(kn+n)*bx, dy-(km+m)*ay-(kn+n)*by
            u, v = x*co+y*si, -x*si+y*co
            distance = size*xp.abs((u/a)**2+(v/b)**2-1.)
            gu, gv = 2.*size*u/(a*a), 2.*size*v/(b*b)
            gx, gy = gu*co-gv*si, gu*si+gv*co
            taper = xp.maximum(1.-(x*x+y*y)/(cutoff*cutoff),zero)**4
            terms.append((distance,taper*gx*gx,taper*gx*gy,taper*gy*gy))
            minimum = xp.minimum(minimum,distance)
    on_gamma = minimum == 0.
    safe_min = xp.where(on_gamma,one,minimum)
    aa, bb, cc = zero, zero, zero
    for distance,a,b,c in terms:
        on_face = distance == 0.
        ratio = safe_min/xp.where(on_face,one,distance)
        weight = xp.where(on_gamma,xp.where(on_face,one,zero),ratio**_BLEND_EXPONENT)
        aa, bb, cc = aa+weight*a, bb+weight*b, cc+weight*c
    return minimum, aa, bb, cc


def dyadic_projector_fields(xp, grid_shape, shapes, dtype, lattice, plan=None):
    """Complex normal projector M(1,i), for shape NVF.

    M is a positive sum of normal dyads. This removes the cancellation of
    opposite signed normals. Primitive faces remain extension geometry even
    when materials match; finite-order split/merged results need not coincide.
    Disjoint curves use periodic level-set dyads and an isotropic extension.
    """
    check_dyadic_shapes(shapes)
    X, Y = sample_grid(xp, grid_shape, lattice, dtype)
    if plan is None:
        plan = plan_gamma(shapes, lattice)
        if plan is None:
            plan = _blind_plan(shapes, lattice)
    if not plan.pieces:
        one, zero = xp.ones_like(X), xp.zeros_like(X)
        return one, zero, zero, zero
    slots = _material_slots(xp, shapes, dtype)
    if _tiles_pay(X, *grid_shape, len(plan.pieces)):
        import numpy as np
        cdtype = np.complex128 if dtype == xp.float64 else np.complex64
        out = [np.empty(grid_shape, dtype=cdtype) for _ in range(4)]
        for i in range(0, grid_shape[0], _TILE):
            for j in range(0, grid_shape[1], _TILE):
                block = (slice(i, i+_TILE), slice(j, j+_TILE))
                values = _piece_field(xp, np.ascontiguousarray(X[block]),
                                      np.ascontiguousarray(Y[block]), shapes, plan,
                                      slots, lattice, dtype, dyadic=True)
                for target, value in zip(out, values):
                    target[block] = value
        return tuple(out)
    return _piece_field(xp, X, Y, shapes, plan, slots, lattice, dtype, dyadic=True)


def _fold_to_nearest_image(xp, dx, dy, reduced):
    """``(dx, dy)`` reduced to the nearest periodic image, on the cell's own lattice.

    ``d % P - P/2`` answers "nearest image" only when the images sit at ``(m*Px, n*Py)``. They
    sit at ``m*a1 + n*a2``, so on a skewed cell the rectangular fold reduces towards a point
    where no image is: on a 60-degree cell it looks straight up while the neighbour is up and
    to the right. The field is then wrong wherever a neighbour's surface is the nearest one,
    which is why it only shows once a shape fills enough of the cell.

    Rounding the *fractional* coordinates lands in the fundamental parallelogram — the nearest
    image in the lattice metric, not the Euclidean one. The 3x3 sweep after it is what makes
    the pick Euclidean, **and it is enough only on a Lagrange-reduced basis**, which is what
    ``reduced`` has to be (:func:`~wavebit.lattice.reduced_basis`; the caller reduces
    once, with the plan's integers). The argument: with ``|a1 . a2| <= |a_i|^2 / 2`` for both
    vectors, a point of the Wigner-Seitz cell written as ``s1 a1 + s2 a2`` has ``|s1 + s2 mu1|
    <= 1/2`` and ``|s2 + s1 mu2| <= 1/2`` with ``|mu_i| <= 1/2``, hence ``|s_i| <= 1``; the
    rounded residual sits within ``|s_i| <= 1/2``, so the image it is nearest is at most
    ``3/2`` cells away in each index and the integer offset is in ``{-1, 0, 1}``. On an
    unreduced basis the Wigner-Seitz cell reaches ``|s| ~ (1 + k) / 2`` for ``a2 = k a1 +
    ...``, outside the sweep from ``k = 3``: a lone 693 circle on ``a2 = 3 a1 + (50, 700)`` read
    its own level set 8.89 deg off in the mean on its boundary, 43.8 at ``k = 10``, and this
    docstring said one cell a side was enough for any basis the solver accepts. It is now,
    because the basis handed here is never the one the solver accepted but its reduction.
    """
    a1x, a2x = reduced[0][0], reduced[0][1]
    a1y, a2y = reduced[1][0], reduced[1][1]
    det = a1x * a2y - a2x * a1y
    su = xp.round((a2y * dx - a2x * dy) / det)
    sv = xp.round((-a1y * dx + a1x * dy) / det)

    best_x = best_y = best_d2 = None
    for du in (-1.0, 0.0, 1.0):
        for dv in (-1.0, 0.0, 1.0):
            mu, mv = su + du, sv + dv
            ex = dx - mu * a1x - mv * a2x
            ey = dy - mu * a1y - mv * a2y
            d2 = ex * ex + ey * ey
            if best_d2 is None:
                best_x, best_y, best_d2 = ex, ey, d2
            else:
                take = d2 < best_d2
                best_x = xp.where(take, ex, best_x)
                best_y = xp.where(take, ey, best_y)
                best_d2 = xp.where(take, d2, best_d2)
    return best_x, best_y


def _nearest_centre_displacement(xp, X, Y, cx, cy, reduced):
    """``(X - cx, Y - cy)`` carried to the nearest periodic image of the centre ``(cx, cy)``.

    :func:`_round_field`'s fold, shared with the tensor curve candidates
    (:func:`~wavebit.rcwa._engine.tensor_shapes._curve_candidate`) so the two cannot drift apart: the
    rectangular fold on ``diag(Lx, Ly)``, where it and the lattice fold are the same fold, and
    :func:`_fold_to_nearest_image` on every other basis. ``reduced`` must be the
    Lagrange-reduced basis (see there for why).
    """
    if is_axis_aligned(reduced):
        Px, Py = periods_of(reduced)
        dx = (X - cx + Px / 2.0) % Px - Px / 2.0
        dy = (Y - cy + Py / 2.0) % Py - Py / 2.0
        return dx, dy
    return _fold_to_nearest_image(xp, X - cx, Y - cy, reduced)


def _round_field(xp, X, Y, shape, lattice, reduced, dtype):
    """``(distance, normal_x, normal_y)`` of one round shape -- the ``'arc'`` piece.

    The displacement is folded to the nearest image and the level set's gradient is the
    normal -- for a circle, the radial direction exactly. The distance is the algebraic
    ``|f| / |grad f|``, first order at the interface and nowhere else: ``(R**2 - r**2) / (2 r)``
    for a circle at radius ``r``, three halves of the true distance at ``r = R / 2`` and
    unbounded at the centre. A lone piece never reads it; beside other pieces it is this
    piece's weight in the blend, for a circle and a single-image ellipse alike, and it stays
    so because every result since 2026-08-17 rests on it. For an ellipse it is why the blend
    steps where the support leaves its cell and the piece takes the exact distance over its
    ring (:func:`_arc_offsets`). The nearest *centre* is the nearest instance
    for any circle that does not overlap its own images (a boundary point nearer another
    instance would put two centres closer than two radii). For any shape that stays inside
    the Wigner-Seitz cell of its centre it is the instance whose interface the cell holds
    (:func:`_arc_offsets`), which is why the ``'arc'`` piece has a single instance there and
    this arithmetic is the one the round branch always had -- bit-identical to every result
    since 2026-08-17 on a reduced basis. For an ellipse that is not always the nearest
    *boundary* off the interface -- a neighbour's tip can be nearer
    (:func:`~wavebit.rcwa._engine.tensor_shapes._curve_candidate`) -- and this field reads its own
    instance's level set there by design. The tensor curve candidates take the same
    level-set direction from the nearest boundary's instance instead, so a lone ellipse and
    the same ellipse in a disjoint layer agree wherever the two instances are one.
    *Finding* that nearest centre is the fold's job, and it does it on ``reduced``, the
    Lagrange-reduced basis (:func:`_fold_to_nearest_image` for why one cell a side is enough
    there and was not on ``a2 = 3 a1 + (50, 700)``); ``lattice`` is the basis as given, which
    fixes only the origin a centre is measured from. A ring of instances is not taken *with
    this distance* on purpose: the algebraic ``|f| / |grad f|`` is unbounded at a shape's own
    centre, so a minimum over instances would hand the centre of every circle to its
    neighbour. A shape that leaves its cell takes the ring with the exact distance instead
    (:func:`_round_instances`).

    ``diag(Lx, Ly)`` is the one case where the two folds ARE the same fold, so it keeps the
    cheaper spelling -- and stays bit-identical to every result predating the lattice branch.
    Read off the reduced basis: a shear of ``k a1`` on a rectangle reduces *to* the
    rectangle, and the rectangular fold is then the right one.
    """
    _stype, cx, cy, lx, ly, angle = shape_geometry(shape, lattice)
    cos_a = xp.cos(as_array(xp, angle, dtype=dtype))
    sin_a = xp.sin(as_array(xp, angle, dtype=dtype))

    dx, dy = _nearest_centre_displacement(xp, X, Y, cx, cy, reduced)
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a
    a, b = lx / 2.0, ly / 2.0

    gxr = 2.0 * xr / (a ** 2)
    gyr = 2.0 * yr / (b ** 2)
    # The algebraic distance |f| / |grad f| -- first order at the level set, for a circle as
    # for an ellipse; the exact one is _ellipse_distance's (see the docstring).
    f = (xr / a) ** 2 + (yr / b) ** 2 - 1.0
    dist = xp.abs(f) / _hypot_safe(xp, gxr, gyr)

    gx = gxr * cos_a - gyr * sin_a
    gy = gxr * sin_a + gyr * cos_a
    norm = _hypot_safe(xp, gx, gy)
    return dist, gx / norm, gy / norm


#: Halvings of the angle bracket in :func:`_ellipse_distance` (2e-5 rad), and the Newton steps
#: that take the root from there to rounding. Measured against a brute-force search over 1200
#: points on three ellipses (the axes, the evolute and a point 1e-7 off the major axis
#: included): 24 + 3 reads 4.3e-13 at worst, 16 + 3 the same 4.8e-13 at 42 ms per instance on
#: a 512 x 512 grid against 56, 12 + 4 1.3e-12, and 8 + 5 breaks down at 1.7e-8.
_ARC_BISECTIONS = 16
_ARC_NEWTON = 3


def _ellipse_distance(xp, u, v, a, b):
    """The Euclidean distance from ``(u, v)`` to the ellipse ``(u/a)^2 + (v/b)^2 = 1``, exactly.

    In the shape's own frame. The nearest boundary point is found by its angle ``t``: for a
    point folded into the closed first quadrant, ``G(t) = (a^2 - b^2) cos t sin t - a u sin t
    + b v cos t`` -- minus half the derivative of the squared distance along the boundary --
    is ``b v >= 0`` at ``t = 0``, ``-a u <= 0`` at ``pi / 2``, and changes sign exactly once
    in between, at the nearest point: the distance's other critical points lie in the other
    quadrants. So the root is bisected on the sign, :data:`_ARC_BISECTIONS` halvings with
    the bracket's ends carried as ``(cos, sin)`` and the midpoint by the half-angle rule (no
    trigonometry in the loop), then polished by :data:`_ARC_NEWTON` Newton steps on ``G``,
    clamped to the bracket. Measured: 6e-13 at worst against a brute-force search over 800
    random points on four ellipses, 7e-15 in the median; 1e-13 on boundary points; the centre
    reads ``min(a, b)``; and ``d/da``, ``d/db``, ``d/du`` agree with central differences to
    2e-8 at nine points including the axes, the centre and a point 1e-7 off the major axis
    inside the evolute. The derivative is the implicit one because the distance is stationary
    in ``t`` at the nearest point, so ``dt/d(theta)`` -- which a bisection gets wrong -- is
    multiplied by ``dD/dt = 0``; the polish is what makes that zero exact.

    Why the angle and not the Lagrange multiplier. The textbook form (Eberly's bracketed
    bisection on the multiplier ``s``) has a pole at ``s = -1`` that every interior point near
    the major axis approaches -- ``s + 1 ~ 1e-9`` for a point 1e-7 off it -- and a float64
    ``s`` there keeps seven digits of ``s + 1``, which came back as 4e-6 in the distance. The
    angle has no pole, and the bracket ``[0, pi / 2]`` is the same width for every point.

    Why a hard bracket rather than Newton from a guess: ``G`` is not monotone, so a Newton
    step from the wrong side leaves the quadrant, and the bisection is what makes the answer
    array-only with a fixed step count under ``jit``. A circle is not special-cased: ``a == b``
    makes ``G = b (v cos t - u sin t)``, whose root is the ray through the point, and the
    distance the radial one. Every discarded ``where`` branch is finite, so autograd never
    meets ``0 * inf`` (the :func:`_hypot_safe` lesson).

    The arithmetic is :func:`_ellipse_root`'s, which :func:`_ellipse_nearest_point` reads
    too: one copy, so the distance and the normal the tensor curve candidates take from the
    same nearest point -- around a circle, and at an ellipse's centre -- cannot drift apart.
    """
    return _ellipse_root(xp, u, v, a, b)[0]


def _ellipse_root(xp, u, v, a, b):
    """``(distance, t, t_lo, t_hi)``: :func:`_ellipse_distance`, the root ``t`` it found in
    the first quadrant and the bisection's final bracket around it."""
    one, zero = xp.ones_like(u), xp.zeros_like(u)
    y0, y1 = xp.abs(u), xp.abs(v)
    k = a * a - b * b
    au, bv = a * y0, b * y1
    c_lo, s_lo = one, zero                                # t = 0
    c_hi, s_hi = zero, one                                # t = pi / 2
    for _ in range(_ARC_BISECTIONS):
        c_sum = c_lo * c_hi - s_lo * s_hi                 # cos(lo + hi)
        c_m = xp.sqrt(xp.maximum(0.5 * (1.0 + c_sum), zero))
        s_m = xp.sqrt(xp.maximum(0.5 * (1.0 - c_sum), zero))
        pos = (k * c_m * s_m - au * s_m + bv * c_m) > 0.0
        c_lo, s_lo = xp.where(pos, c_m, c_lo), xp.where(pos, s_m, s_lo)
        c_hi, s_hi = xp.where(pos, c_hi, c_m), xp.where(pos, s_hi, s_m)
    t_lo, t_hi = xp.atan2(s_lo, c_lo), xp.atan2(s_hi, c_hi)
    t = 0.5 * (t_lo + t_hi)
    for _ in range(_ARC_NEWTON):
        c, s = xp.cos(t), xp.sin(t)
        g = k * c * s - au * s + bv * c
        gp = k * (c * c - s * s) - au * c - bv * s
        gp = xp.where(gp < 0.0, gp, -one)                 # G' < 0 at a strict minimum
        t = xp.minimum(xp.maximum(t - g / gp, t_lo), t_hi)
    c, s = xp.cos(t), xp.sin(t)
    distance2 = (a * c - y0) ** 2 + (b * s - y1) ** 2
    on_boundary = distance2 == 0.0
    # The unsigned distance has a cusp here, but the normal-field blend can be
    # smooth. Guard the radicand so its zero cotangent never meets sqrt'(0).
    return xp.where(on_boundary, zero,
                    xp.sqrt(xp.where(on_boundary, one, distance2))), t, t_lo, t_hi


def _ellipse_nearest_point(xp, u, v, a, b):
    """``(distance, c, s)``: :func:`_ellipse_distance`, and where its nearest point is.

    The nearest point of ``(u/a)^2 + (v/b)^2 = 1`` to ``(u, v)`` is ``(a c su, b s sv)``,
    with ``c = cos t`` and ``s = sin t`` for the root ``t`` in ``[0, pi / 2]`` of the folded
    problem (:func:`_ellipse_distance`) and ``su``, ``sv`` the signs of ``u`` and ``v``, a
    zero counting as positive. The outward normal there is along ``(b c su, a s sv)``, never
    shorter than ``min(a, b)``. Where the nearest point is not unique -- the medial axis, the
    segment of the major axis between the evolute's cusps, and so the centre -- this is the
    one the fold picks, ``v = 0`` taking the ``+v`` side. ``t`` is exact to rounding wherever
    the Newton polish converges quadratically; at the cusps, where ``G`` has a multiple root,
    it converges linearly and ``t`` stays inside the bisection's 2e-5 rad bracket.

    The fold is the distance's and not the point's: the distance is even in ``u`` and in
    ``v``, the nearest point odd, and ``abs()`` differentiates to zero at zero. So on an axis
    the root :func:`_ellipse_root` returns carries ``dt/du = 0`` (``dt/dv``), which the
    distance never reads -- its slope across an axis is zero, or a ridge on the medial axis
    -- and a direction does: ``s`` moves off ``0`` linearly in ``v`` across the ``u`` axis
    beyond the medial segment, which for a circle, whose candidate carries this normal at
    every sample, is the whole axis but its centre. Autodiff read the normal's ``v``
    component as constant there. One more Newton step on the signed fold ``su u``, ``sv v``
    -- the same values, whose derivative is ``+-1`` rather than ``0`` -- leaves ``t`` where
    it is to rounding, since ``G(t) = 0``, and gives it the implicit derivative
    ``-dG / G'`` everywhere the polish converged, the axes included. A step that would leave
    the bracket (a double root) is not taken, but one that leaves it by rounding is: where the
    bisection meets the root exactly -- its first midpoint is the root on a circle's diagonals
    -- the root is the bracket's end, the step can land an ulp past it, and refusing that step
    dropped the normal's derivative to zero (at a sample on a circle's diagonal in the odd-grid
    test of ``tests/rcwa/test_tensor_curve_normals.py``, where the dyad's ``xx`` slope is ``1/240``
    per nm both ways).
    """
    distance, t, t_lo, t_hi = _ellipse_root(xp, u, v, a, b)
    one = xp.ones_like(u)
    k = a * a - b * b
    au = a * (xp.where(u < 0.0, -one, one) * u)
    bv = b * (xp.where(v < 0.0, -one, one) * v)
    c, s = xp.cos(t), xp.sin(t)
    g = k * c * s - au * s + bv * c
    gp = k * (c * c - s * s) - au * c - bv * s
    step = t - g / xp.where(gp < 0.0, gp, -one)
    slack = 4.0 * xp.finfo(t.dtype).eps
    t = xp.where((step >= t_lo - slack) & (step <= t_hi + slack), step, t)
    return distance, xp.cos(t), xp.sin(t)


def _instance_displacements(xp, X, Y, cx, cy, reduced, imgs):
    """``(X, Y)`` relative to each instance ``imgs`` of the centre ``(cx, cy)``, one pair at a time.

    The offsets count cells of ``reduced`` from the centre's image nearest the sample *in the
    lattice metric*: rounding the fractional coordinates, a parallelogram fold, which is what
    the ring of :func:`_arc_offsets` is bounded around. :func:`_round_instances`' arithmetic,
    shared with the tensor curve candidates
    (:func:`~wavebit.rcwa._engine.tensor_shapes._curve_candidate`). A generator, so a long ring holds
    one instance's displacement at a time.
    """
    a1x, a2x = reduced[0][0], reduced[0][1]
    a1y, a2y = reduced[1][0], reduced[1][1]
    det = a1x * a2y - a2x * a1y
    dx0, dy0 = X - cx, Y - cy
    km = xp.round((a2y * dx0 - a2x * dy0) / det)
    kn = xp.round((-a1y * dx0 + a1x * dy0) / det)
    for m, n in imgs:
        yield dx0 - (km + m) * a1x - (kn + n) * a2x, dy0 - (km + m) * a1y - (kn + n) * a2y


def _round_instances(xp, X, Y, shape, lattice, reduced, dtype, imgs):
    """``(distance, normal_x, normal_y)`` of one round shape over the instances ``imgs``.

    The ``'arc'`` piece of a shape that leaves its Wigner-Seitz cell (:func:`_arc_offsets`):
    :func:`_round_field` with the nearest instance chosen by its *boundary* rather than by
    its centre. The sample is folded to the shape's nearest lattice cell as :class:`_Frame`
    folds it, the offsets are taken from there, each instance carries the exact Euclidean
    distance to its boundary (:func:`_ellipse_distance`) -- bounded, ``min(a, b)`` at the
    centre, so the minimum over instances is the nearest interface and nothing hands a
    shape's centre to its neighbour -- and the piece's direction is the level-set normal of
    the nearest instance at the sample, the same direction :func:`_round_field` reads. The
    two fields are not the same away from the cell's edge: :func:`_round_field` weighs a
    piece by the algebraic distance ``|f| / |grad f|``, not this exact one, and for an
    ellipse its nearest centre is often not the instance whose boundary is nearest, well
    inside the cell too (:func:`_arc_offsets`). A hard minimum, as a face takes over its instances: the fold
    already switches instance across the cell's edge, and this moves that switch to the
    Voronoi boundary between the two boundaries, which is where the geometry has it. The
    fold and the offsets count cells of ``reduced``, as everywhere; ``lattice`` is the origin.
    """
    _stype, cx, cy, lx, ly, angle = shape_geometry(shape, lattice)
    cos_a = xp.cos(as_array(xp, angle, dtype=dtype))
    sin_a = xp.sin(as_array(xp, angle, dtype=dtype))
    a, b = lx / 2.0, ly / 2.0
    best = gx_best = gy_best = None
    for dx, dy in _instance_displacements(xp, X, Y, cx, cy, reduced, imgs):
        xr = dx * cos_a + dy * sin_a
        yr = -dx * sin_a + dy * cos_a
        gxr, gyr = 2.0 * xr / (a ** 2), 2.0 * yr / (b ** 2)
        gx = gxr * cos_a - gyr * sin_a
        gy = gxr * sin_a + gyr * cos_a
        norm = _hypot_safe(xp, gx, gy)
        gx, gy = gx / norm, gy / norm
        dist = _ellipse_distance(xp, xr, yr, a, b)
        if best is None:
            best, gx_best, gy_best = dist, gx, gy
        else:
            take = dist < best
            best = xp.where(take, dist, best)
            gx_best = xp.where(take, gx, gx_best)
            gy_best = xp.where(take, gy, gy_best)
    return best, gx_best, gy_best


def _hypot_safe(xp, a, b):
    """``hypot`` with zeros replaced by 1, so the division that follows is well defined.

    The guard is on the *input* to the root, not on its output. Patching the output
    (``where(n == 0, 1, n)`` on ``n = (a*a + b*b)**0.5``) fixes the forward value and nothing
    else: autograd still differentiates the branch ``where`` discarded, ``d/da`` of that root
    at ``a == b == 0`` is infinite, and ``where``'s ``0 * inf`` is NaN. Taking the root of the
    replaced radicand instead means it is only ever evaluated at 1, where the derivative is
    finite — and ``sqrt(1) == 1`` is the same replacement value, so the forward result is
    unchanged.

    Both callers reach zero on ordinary geometry: a shape's own centre gives ``g == 0``, and
    two shapes placed symmetrically make their unit normals exactly opposed on the segment
    between them, which the grid lands on exactly.
    """
    n2 = a * a + b * b
    return xp.where(n2 == 0, xp.ones_like(n2), n2) ** 0.5


def _sampled_coeffs(xp, field, half_width_u, half_width_v):
    """Fourier coefficients of a sampled field on the doubled reciprocal grid.

    ``c(ky, kx) = 1/(Ny*Nx) * sum f[iy, ix] exp(-2i*pi*(ky*iy/Ny + kx*ix/Nx))``, laid out the
    way :func:`assemble_eps_coeffs` lays its grids out. Feeding it to :func:`conv_matrix_2d`
    reproduces ``fourier2d_pointwise(field, ...)`` — that equivalence is what
    ``test_conv_matrix_matches_grid_convention`` states, and it is what lets a translation be
    applied here as a phase instead of by moving the samples.
    """
    Ny, Nx = field.shape
    dev = _device(field)
    if half_width_u == half_width_v == 0:
        # The sole coefficient is exactly the spatial mean. Avoid an unnecessary FFT and
        # its reverse transform, which can fail XLA's CPU layout check in a zero-order solve.
        value = xp.astype(xp.mean(field), xp.result_type(field.dtype, xp.complex64))
        return xp.reshape(value, (1, 1))
    F = xp.fft.fft2(field) / (Ny * Nx)
    iy = xp.asarray([k % Ny for k in range(-2 * half_width_v, 2 * half_width_v + 1)], device=dev)
    ix = xp.asarray([k % Nx for k in range(-2 * half_width_u, 2 * half_width_u + 1)], device=dev)
    coeff = F[iy[:, None], ix[None, :]]
    # The cell-centre half-pixel shift, the same one
    # :func:`~wavebit.rcwa._engine.factorization._half_pixel_phase` applies on the raster side. The
    # samples come from `sample_grid`, which places index `i` at `(i + 0.5)/N` -- so a bare DFT,
    # which places it at `i/N`, describes a field shifted half a pixel against the permittivity
    # these coefficients are multiplied by. Applied here as well as there because these two are
    # the *same transform written twice*, and `test_conv_matrix_matches_grid_convention` is what
    # says so; correcting one alone silently splits the parametric path from the raster one.
    ky = xp.asarray([float(k) for k in range(-2 * half_width_v, 2 * half_width_v + 1)], device=dev)
    kx = xp.asarray([float(k) for k in range(-2 * half_width_u, 2 * half_width_u + 1)], device=dev)
    pi = 3.141592653589793
    phase = (xp.exp(-1j * pi * xp.astype(ky, coeff.dtype) / Ny)[:, None]
             * xp.exp(-1j * pi * xp.astype(kx, coeff.dtype) / Nx)[None, :])
    return coeff * phase


_PROJECTOR_CACHE = {}
_PROJECTOR_CACHE_MAX = 8


def _coeffs_hold_tracer(coeffs):
    """Whether any coefficient grid is a JAX tracer -- built inside a trace, not to be kept.

    Structural (``isinstance``), never a read; ``False`` whenever JAX is not in the process,
    so the other backends never pay for the question. The engine's ``_holds_tracer`` walks
    containers for the same reason; this one only has arrays to look at.
    """
    import sys
    jax = sys.modules.get('jax')
    if jax is None:
        return False
    return any(isinstance(c, jax.core.Tracer) for c in coeffs)


def _static_float(v):
    """``v`` as a plain float, refusing anything a cache must not key on.

    Two refusals, and they are different. A JAX tracer cannot be converted at all, so ``float``
    raises on its own. A PyTorch tensor *can* be converted -- which is exactly the trap
    :func:`layer_needs_resolving` documents -- so one that carries ``requires_grad`` has to be
    refused explicitly, or a sweep over it would be served a result built at the first value --
    under either PyTorch mode, ``requires_grad`` or a ``forward_ad`` dual, which does not set it
    (:func:`~wavebit._common.backend.forward_tangent`), and under ``torch.func.vmap`` too, where
    both sit under a ``BatchedTensor`` that reads neither
    (:func:`~wavebit._common.backend.torch_autodiff`). Not
    :func:`~wavebit._common.backend.carries_autodiff`, which counts every JAX tracer: a jitted
    value refuses ``float`` on its own, and an eager one keeps its cache.
    """
    if torch_autodiff(v):
        raise TypeError('a value under torch autograd is not a cache key')
    detach = getattr(v, 'detach', None)
    if detach is not None:
        v = detach()
    return float(v)


def _static_complex(v):
    """``v`` as a plain complex, with :func:`_static_float`'s two refusals -- for a material."""
    if torch_autodiff(v):
        raise TypeError('a value under torch autograd is not a cache key')
    detach = getattr(v, 'detach', None)
    if detach is not None:
        v = detach()
    return complex(v)


def _material_key(shapes):
    """Every shape's material slot, concretely, for a layer of several shapes -- else nothing.

    What the field reads of the materials (:func:`_piece_field`, the seam contrast): for a
    multi-shape layer the contrast of every pair of owners, a continuous function of their
    values, so the key holds each value and a material sweep rebuilds per value -- ``4 | 9``
    and ``4 | 12`` are two entries. A lone shape's field reads no material, so its key holds
    none and its sweeps hit as they always did. Until 2026-09-11 this held only the equality
    bit of each cross pair, which was every material fact the field then read; a key that
    stopped there now would serve ``4 | 9``'s coefficients to ``4 | 12``, and a torch material
    with ``requires_grad`` on a pair-free layer would be built into stored coefficients. A
    traced or autograd value raises here, and the key is ``None``: rebuild.
    """
    if len(shapes) < 2:
        return ()
    return tuple(_static_complex(as_material(shape.get('material', 1.0), 'shape material').epsilon)
                 for shape in shapes)


def _projector_cache_key(xp, built, grid_shape, lattice, half_width_u, half_width_v, dtype,
                         plan=None):
    """A content key for the coefficients built below, or ``None`` when caching would be wrong.

    Everything :func:`normal_vector_field_analytical` reads goes in and nothing else does. It
    reads each shape through :func:`shape_geometry`, the plan by content, and of the materials
    what the seam contrast reads -- every shape's slot, for a layer of several shapes, and
    nothing for a lone one (:func:`_material_key`) -- which is what makes the cache worth
    having for the lone shape: its material sweeps hit, and a multi-shape sweep rebuilds per
    value since 2026-09-11. ``None`` means at least one value is traced or under autograd, so the
    build has to happen again -- the gradient runs through it. ``plan`` is ``None`` for a
    traced layer, and then the shapes are too, so the key is.
    """
    try:
        shapes_key = tuple(tuple(v if isinstance(v, str) else _static_float(v)
                                 for v in shape_geometry(shape, lattice))
                           for shape in built)
        lat = tuple(_static_float(v) for row in lattice for v in row)
        bits = _material_key(built)
        # Equal numerical values can request different representations: real 4.0 uses
        # three symmetric components, complex 4.0+0j uses four Hermitian components.
        complex_material = len(built) > 1 and any(xp.isdtype(as_array(
            xp, as_material(s.get('material', 1.0), 'shape material').epsilon).dtype,
            'complex floating') for s in built)
    except (TypeError, ValueError):
        return None

    # A lone round shape's key used to drop its size, keeping only the aspect: the level-set
    # gradient of `x^2/a^2 + y^2/b^2` is `(2x/a^2, 2y/b^2)`, whose direction does not move
    # when `a` and `b` scale together, so a radius sweep rebuilt a bit-identical field five
    # times over. That was true of a field that never looked past the shape's own nearest
    # image, and it stops being true the moment a round piece is evaluated over a ring of
    # instances -- two ellipses of one aspect and different sizes that both reach their
    # neighbours are different fields, and a shared key would serve the second the first's
    # coefficients silently. The normalisation went on 2026-09-06, ahead of that ring rather
    # than after it, so that nothing can inherit the trap; a radius sweep now rebuilds.
    # The namespace is redundant with ``str(dtype)``, which already spells it out
    # (``numpy.complex128`` against ``torch.complex128``) -- but that is an accident of how the
    # dtypes are named, not a guarantee, and a cross-backend collision hands one namespace's
    # array to another.
    #
    # **The device is read too, just not from an argument.** The build starts from Python
    # floats, so it lands wherever a fresh array lands right now -- under JAX, the solver's
    # device scope -- and the coefficients come back committed there. Left out of the key, a
    # solver on one device was handed another device's arrays: measured 2026-09-11, a `cpu:1`
    # solve followed by a `cpu:0` one raised `Received incompatible devices for jitted
    # computation` on a 49x49 matmul, and a CPU solve followed by a GPU one did the same on a
    # CUDA box. It could not happen before then only because every JAX array landed on JAX's
    # default device whatever `device=` said. NumPy and Torch put a constant here.
    return (getattr(xp, '__name__', ''), _device(xp.zeros(())), shapes_key, tuple(grid_shape),
            int(half_width_u), int(half_width_v), str(dtype), lat, plan, bits, complex_material)


def normal_projector_coeffs(xp, shapes, grid_shape, lattice, half_width_u, half_width_v,
                            dtype, plan=None, nvf_projector=None):
    """Coefficient grids of the normal projector, in ``(xx, xy, yy[, yx])`` order.

    A real piece field supplies three symmetric components. Complex material inputs can
    produce a complex piece field: its unit Hermitian projector supplies four components,
    with ``xy = normal_x * conj(normal_y)`` and ``yx = conj(xy)`` before Fourier sampling.
    Conjugate in real space, not on the coefficients after applying a translation phase.

    With ``nvf_projector='circular'``, rectangular dyads supply four components in
    ``(xx, xy, yy, yx)`` order. Their cache entries are separate from linear
    fields; the same translation phase applies to every component.

    ``plan`` is the layer's :class:`GammaPlan` -- the frozen record's entry when there is one,
    planned here from the concrete values when there is not (``None`` again under a trace
    with no record, which :func:`normal_vector_field_analytical` answers with the blind plan).
    A plan holds no coordinate, so the one taken over ``built`` -- a lone shape moved to the
    cell's centre, below -- is the layer's own.

    **One shape is built at the centre of the cell and carried to ``(cx, cy)`` by a phase**,
    the same way :func:`assemble_eps_coeffs` carries the permittivity. Two things follow, and
    both are the point.

    *The position becomes differentiable.* Sampling the field at the shape's actual place puts
    ``cx`` and ``cy`` through ``sign()``, whose derivative is zero, so the projectors used to
    contribute *nothing* to ``d/dcy`` — while the permittivity term contributed its exact
    share. Translation invariance holds by those two cancelling, so what was left was the
    permittivity term standing alone: -1.7e-1 on ``d(sum R)/dcy`` for a rectangle, against
    a true value of zero. Through the phase the two carry the same factor and cancel to
    machine precision.

    *And the model becomes translation-covariant, which it was not.* The normal field is
    defined over the whole plane — a rectangle's wedges and a circle's rays do not stop at the
    shape — so sampling it over a fixed cell clips a different part of it depending on where
    the shape sits. That is not a discretization: ``<normal_x**2>`` for a circle moves 2.4e-2 between
    0.30 and 0.50 of the period and does not budge when the grid goes from 256 to 4096. It
    reached ``sum(R)`` as 1.6e-3, so the same structure translated gave a different answer.
    Built here it reproduces the centred one to 1e-14, at every ``truncation_order``.

    **Several shapes are built with their first shape at the centre and carried by the same
    phase** (2026-09-11). The other shapes keep their offsets from the first, so a rigid
    translation of the layer samples the same field at the same points and is restored exactly
    as ``exp(-i G.d)``. Before this the layer was sampled where it sat, and the un-phased
    coefficients of two disjoint boxes spread 1.074e-2 / 4.056e-3 / 1.592e-3 over sub-pixel
    translations at ``shape_nvf_grid_size`` 64 / 128 / 256, ``sum(T)`` by 1.242e-3 / 6.395e-4 / 1.560e-4
    (``studies/rcwa/nvf_placement_spread/nvf_placement_spread.py``); both are at rounding now, as a lone shape's always
    were. The choice of the first shape is the caller's own spelling, like painting order, and
    it is the choice that moves no other invariant: naming any shape a period over shifts the
    built layer by a lattice vector or not at all, which the fold and the periodic grid cannot
    see, so a multi-shape layer stays periodic across the cell edge to 1e-9 in the total
    (``test_a_multishape_layer_is_periodic_across_the_cell_edge``), and every position stays
    continuous. The area centroid of the shapes -- the polygon route's anchor for a painted
    union -- was built and measured first, and rejected on exactly that invariant: naming one
    shape a period over moves the unfolded centroid by a fraction of the period, so the same
    torus structure was sampled at a different offset and the case-G total moved 3.2e-5 where
    1e-9 is promised; folding the centres instead puts a step of the same size wherever a shape
    crosses the fold line. No continuous, translation-equivariant, period-invariant centre of
    several points on a torus exists (two shapes half a period apart have no canonical
    midpoint), so one shape's centre is the anchor.

    What the anchor does not do is make the *relative* motion of one shape exact: the second
    shape's staircase moves against the grid when it moves, as it always did, and the
    coefficients keep a sampling ripple across a pixel. Measured against a 1024 reference on
    the same two boxes, the second one moved across one pixel: the error varies by 8.6e-3 at
    ``shape_nvf_grid_size`` 64 and 1.2e-3 at 256, where the legitimate coefficient change over the same
    move is 2.9e-4 / 2.6e-4. That ripple, and the sampling error itself, are what ``shape_nvf_grid_size``
    is for, since the field is still read off a grid.

    The unsampled alternative was measured and is not taken. Exact coefficients exist for a
    field that is piecewise constant on polygons, which the blend is not; the field that is --
    the nearest-interface (segment Voronoi) extension the polygon route samples -- carries a
    value step at every same-material merge. As two 250 x 300 boxes close from a gap of 20 to
    0.01, the blend's coefficients approach the merged box's (largest deviation 0.081 -> 0.027,
    reaching 0.024 at contact, which is case D's split-spelling residual) while the
    nearest-point field's deviation plateaus at 0.086 at ``shape_nvf_grid_size`` 256 and again at 512:
    the gap face's Voronoi cell is a triangle of a quarter of the box that reads ``n = +-x``
    until the face dies. The pair records of :func:`_piece_field` exist to remove exactly that
    step, so the extension stays and only its sampling origin moved.

    The centroid anchor was rejected once, against the *sum of unit vectors* this field used to
    be: two shapes opposed about the cell centre cancelled along the whole segment between them
    and the common-translation gradient came back NaN through ``0 * inf`` in
    :func:`_hypot_safe`. The ratio blend does not cancel there -- zero grid points reach
    ``|n| == 0`` at N = 256, 512 and 1024 on two identical circles, against 85, 144 and 303
    under the old rule -- and the rejection's other number, 1.5-3.4x more relative jitter, was
    measured on that old field too; on this one the jitter falls, as above. Two other routes
    stay rejected: interpolating each shape's field onto the common grid (forward 7e-5,
    derivative 2e-1), and mollifying ``sign()`` (a width that is a tuning constant, and it broke
    ``test_analytic_angle_is_smooth_through_normal``). A fourth, the boundary route's
    gradient-of-distance field for disjoint layers, integrates its position gradient to the
    wrong sign across a window (``studies/rcwa/benchmark_boundary_normal_field/benchmark_boundary_normal_field.py``,
    ``docs/guide/rcwa/boundary-modeling.md``) and costs 2x / 182x on a rectangle / circle pair.

    A multi-shape layer is periodic across the cell edge -- ``cx`` and ``cx + Px`` are one
    structure to 1e-9 in the total and 1e-12 in the projectors
    (``test_a_multishape_layer_is_periodic_across_the_cell_edge``) -- since every distance in
    the field folds to the nearest image, and a fold is the same fold about any origin.

    ``grid_shape`` is checked against the ``4*truncation_order+1`` floor
    (:func:`~wavebit.rcwa._engine.factorization.check_normal_grid`). The boundary route has done that
    since it was written and this one did not, which was not a decision: at
    ``truncation_order=10`` the two disagreed about ``shape_nvf_grid_size=8``, one refusing and the
    other building it 4.9e-4 adrift with energy conserved to twelve digits. The floor is a
    property of reading a field off a grid, not of which builder put it there.
    """
    check_normal_grid(grid_shape, half_width_u, half_width_v)
    real_dtype = xp.float64 if dtype == xp.complex128 else xp.float32

    if shapes:
        _, cx, cy, _, _, _ = shape_geometry(shapes[0], lattice)
        # The `lattice is None` arm that stood here read `Px / 2` as the centre and
        # `1 / Px` as the fraction. The solver stopped reaching it once
        # `_geometry_lattice_for` became total (`cc016fc`), and it differs from this one in
        # the last place, so it is deleted rather than re-keyed on `is_axis_aligned`.
        a1x, a2x = lattice[0][0], lattice[0][1]
        a1y, a2y = lattice[1][0], lattice[1][1]
        mid_x, mid_y = 0.5 * (a1x + a2x), 0.5 * (a1y + a2y)
        # Same quantity, stated properly: the displacement in *fractional lattice*
        # coordinates, A^-1 @ (c - mid). The phase is exp(-2i*pi*(m*u_d + n*v_d)), which is
        # exp(-i G.d) already -- it was lattice-general all along, it was only ever handed
        # its argument as a fraction of Px / Py.
        det = a1x * a2y - a2x * a1y
        dx, dy = cx - mid_x, cy - mid_y
        shift_u = (a2y * dx - a2x * dy) / det
        shift_v = (-a1y * dx + a1x * dy) / det
        # `0.0`, not `mid`: a centre is measured from `0.5*(a1 + a2)` since 2026-08-31, so the
        # centre of the cell is what an explicit zero names. Writing `mid` here would add the
        # origin twice and build the field for a shape one cell-centre away from where the
        # phase then puts it. The other shapes keep their offsets from the first, in the same
        # centre-relative spelling, so the layer's relative geometry is what gets sampled and
        # its position is what the phase carries -- see the docstring for why the first shape
        # and not a centroid.
        ax, ay = shapes[0].get('cx', 0.0), shapes[0].get('cy', 0.0)
        built = [dict(shapes[0], cx=0.0, cy=0.0)]
        built += [dict(s, cx=s.get('cx', 0.0) - ax, cy=s.get('cy', 0.0) - ay) for s in shapes[1:]]
    else:
        built, shift_v, shift_u = shapes, None, None

    # The build above is the expensive half and the phase below is the cheap one, so what is
    # cached is the split: for a single shape ``built`` sits at the centre of the cell whatever
    # ``(cx, cy)`` are, which makes a position sweep -- and a position *gradient*, which runs
    # through the phase -- reuse one field instead of rebuilding it every step. (Several shapes
    # are keyed on their offsets from the first, which a rigid slide changes by an ulp, so a
    # multi-shape slide rebuilds.) A lone shape's material sweep reuses it for the other reason:
    # its key holds no material because its build reads none; a multi-shape layer's key holds
    # every material since 2026-09-11 (the seam contrast reads them), so its sweeps rebuild.
    # Measured on a five-point sweep at ``shape_nvf_grid_size = 1024``, ``truncation_order = 4``: the call
    # 83% of the wall clock for one shape and 93% for six.
    if plan is None:
        plan = plan_gamma(built, lattice)
    key = _projector_cache_key(xp, built, grid_shape, lattice, half_width_u, half_width_v, dtype,
                               plan)
    if nvf_projector == 'circular' and key is not None:
        key = ('dyadic-projector', key)
    coeffs = _PROJECTOR_CACHE.pop(key, None) if key is not None else None
    if coeffs is None:
        if nvf_projector == 'circular':
            fields = dyadic_projector_fields(xp, grid_shape, built, real_dtype, lattice, plan=plan)
        else:
            normal_x, normal_y = normal_vector_field_analytical(xp, grid_shape, built,
                                                                real_dtype, lattice, plan=plan)
            if xp.isdtype(normal_x.dtype, 'complex floating'):
                xy = normal_x * xp.conj(normal_y)
                fields = (xp.real(normal_x * xp.conj(normal_x)), xy,
                          xp.real(normal_y * xp.conj(normal_y)), xp.conj(xy))
            else:
                fields = (normal_x * normal_x, normal_x * normal_y, normal_y * normal_y)
        coeffs = tuple(_sampled_coeffs(xp, xp.astype(f, dtype), half_width_u, half_width_v)
                       for f in fields)
    here = _device(xp.zeros(())) if key is not None else None
    if (key is not None and not _coeffs_hold_tracer(coeffs)
            and all(_device(c) == here for c in coeffs)):
        # Least-recently-used: the read above is a `pop`, so a hit takes the entry out and this
        # puts it back at the end, and a full cache evicts only on a genuine miss. The case that
        # needs it is a stack holding both a moving layer and a still one -- the moving layer
        # misses every step, and without the refresh the still one ages out from under it.
        # Measured on a two-layer stack over 30 steps: the still layer is built once and the
        # cache stays pinned at its bound, against 60 builds with no cache at all.
        #
        # Never stored: coefficients built INSIDE a jit trace. The key is concrete -- every
        # input was -- but under omnistaging the arrays the build returns are tracers, and a
        # tracer put under a concrete key is served to the next eager solve on that key, on
        # any solver in the process, which raises `UnexpectedTracerError` at `c * phase` until
        # someone clears the cache by hand. Measured 2026-09-06: build, freeze, jit over the
        # wavelength as the FIRST solve on the key, then `solve()` -- every eager call raised.
        # An eager solve before the jit hid it by warming the key, which is why the recipe
        # tests passed. A miss inside a trace is computed and discarded, as a traced key is.
        #
        # Never stored either: coefficients that are not where the key says. The key reads the
        # device a fresh array lands on -- the scope's -- but the build follows any committed
        # operand it reads, so a shape value committed to another device builds the field
        # there. A live solver moved with `device=` does exactly that, because it keeps its
        # stored arrays where they were. Measured 2026-09-11 on two host devices, and between
        # the CPU and the GPU on a CUDA box: the solve that filed such an entry raised, and the
        # entry -- keyed on one device, holding the other's arrays -- made every later solve of
        # that geometry on the key's device raise `Received incompatible devices`, on any
        # solver in the process, until the cache was cleared by hand. The mismatched build
        # still serves its own call; it is only not kept.
        if len(_PROJECTOR_CACHE) >= _PROJECTOR_CACHE_MAX:
            _PROJECTOR_CACHE.pop(next(iter(_PROJECTOR_CACHE)))
        _PROJECTOR_CACHE[key] = coeffs

    if shift_v is None:
        phase = 1.0
    else:
        KKY, KKX = _grid(xp, half_width_u, half_width_v, _device(coeffs[0]), real_dtype)
        arg = KKY * shift_v + KKX * shift_u
        phase = xp.exp(xp.astype(arg, dtype) * (-2j * 3.141592653589793))

    return tuple(c * phase for c in coeffs)


def to_conv_mat_parametric_nvf(xp, shapes, background, half_width_u, half_width_v, lattice, dtype,
                           shape_nvf_grid_size=256, plan=None, nvf_projector=None):
    """The five 2D-solve convolution matrices for one parametric layer, via NVF.

    ``eps`` / ``1/eps`` come from the exact shape transform (no staircase); the normal
    projectors come from the shapes' analytic normal field. Same Hermitian symmetrization as
    the raster NVF path — see :func:`wavebit.rcwa._engine.factorization.nvf_conv_matrices`.

    Every one of the five is ``D X D^-1`` for the shape's translation phase ``D`` and a
    position-free ``X`` (see :func:`normal_projector_coeffs`), so sliding a single-shape layer
    conjugates the whole solve by a diagonal and leaves each order's efficiency alone.

    The coefficients come from :func:`assemble_eps_coeffs`, which sums -- see there for why
    a layer whose shapes overlap must not arrive at this function. ``plan`` is the layer's
    frozen :class:`GammaPlan`, or ``None`` to plan from the values.
    """
    eps_c, inv_eps_c = assemble_eps_coeffs(xp, shapes, background, half_width_u,
                                           half_width_v, lattice, dtype)
    coeffs = normal_projector_coeffs(
        xp, shapes, (shape_nvf_grid_size, shape_nvf_grid_size), lattice, half_width_u, half_width_v, dtype,
        plan=plan, **({} if nvf_projector is None else {'nvf_projector': nvf_projector}))
    return nvf_from_coeffs(xp, eps_c, inv_eps_c, *coeffs[:3], half_width_u, half_width_v,
                           c_yx=coeffs[3] if len(coeffs) == 4 else None,
                           eps_bound=largest_eps(xp, [background] + [s['material'] for s in shapes],
                                                 dtype))


def largest_eps(xp, materials, dtype):
    """``max |eps|`` over ``materials`` (numbers or :class:`Material`), as an ``xp`` scalar.

    The bound :func:`~wavebit.rcwa._engine.factorization._nvf_assemble` keeps the inverse rule under.
    Read through :func:`~wavebit._common.periodic.as_material` like every other material here, so
    a bare number and a ``Material`` agree, and through :func:`as_array` so a traced or
    grad-tracking permittivity keeps its graph.
    """
    mags = [xp.abs(as_array(xp, as_material(m, 'layer material').epsilon, dtype=dtype))
            for m in materials]
    out = mags[0]
    for m in mags[1:]:
        out = xp.maximum(out, m)
    return out


def nvf_from_coeffs(xp, eps_c, inv_eps_c, c_xx, c_xy, c_yy, half_width_u, half_width_v,
                    c_yx=None, eps_bound=None):
    """Assemble the five convolution matrices from coefficient grids — the NVF tensor itself.

    The last step of every non-raster NVF path, and the same step for all of them: what differs
    between the parametric, boundary and level-set routes is *how* the five coefficient grids were
    obtained, never what is done with them. The closing itself is
    :func:`wavebit.rcwa._engine.factorization._nvf_assemble`, shared with the raster path
    (:func:`wavebit.rcwa._engine.factorization.nvf_conv_matrices`), so a change to the tensor cannot
    reach one path and miss another -- until the inverse-rule floor this function carried its own
    copy of the formula, and the two had to be changed in step.

    ``eps_c`` / ``inv_eps_c`` are the ``eps`` and ``1/eps`` coefficients on the doubled
    reciprocal grid; ``c_xx`` / ``c_xy`` / ``c_yy`` those of ``normal_x**2``, ``normal_x*normal_y``, ``normal_y**2``.
    ``eps_bound`` is the layer's largest ``|eps|`` (see :func:`largest_eps`); ``None`` leaves the
    inverse rule unfloored, which is the floored closing bit for bit only on a layer of passive
    materials with every ``Re eps > 0`` (see ``INVERSE_RULE_BOUND``).
    Returns ``(eps_xx, eps_yy, epz_i, eps_xy, eps_yx)`` — ``solve_conv``'s parameter order,
    as every builder does; see "Slot order" in :mod:`wavebit.rcwa._engine.factorization`.
    """
    ep_dir = conv_matrix_2d(xp, eps_c, half_width_u, half_width_v)
    ep_inv = conv_matrix_2d(xp, inv_eps_c, half_width_u, half_width_v)

    C_xx = conv_matrix_2d(xp, c_xx, half_width_u, half_width_v)
    C_xy = conv_matrix_2d(xp, c_xy, half_width_u, half_width_v)
    C_yy = conv_matrix_2d(xp, c_yy, half_width_u, half_width_v)

    C_yx = None if c_yx is None else conv_matrix_2d(xp, c_yx, half_width_u, half_width_v)
    return _nvf_assemble(xp, ep_dir, ep_inv, C_xx, C_xy, C_yy, C_yx, eps_bound=eps_bound)


def laurent_from_coeffs(xp, eps_c, half_width_u, half_width_v):
    """Laurent's rule from a coefficient grid: ``([[eps]], [[eps]], [[eps]]^-1, None, None)``.

    :func:`nvf_from_coeffs`'s twin for the direct rule, and the last step of both shape routes
    under ``factorization='laurent'`` -- the closed form and the painted polygons differ only in
    how ``eps_c`` was obtained. ``ep_dir`` there is exactly this matrix; what Laurent drops is
    everything NVF builds on top of it, the normal field included. The slots are those of
    :func:`wavebit.rcwa._engine.factorization.laurent_conv_matrices`, which states the rule and its
    sources (Moharam & Gaylord, JOSA 71, 811, 1981; L. Li, JOSA A 13, 1870, 1996).
    """
    direct = conv_matrix_2d(xp, eps_c, half_width_u, half_width_v)
    return ConvMatrices(direct, direct, xp.linalg.inv(direct), None, None)


def check_no_plan_still_fits(layer, lattice, index):
    """Refuse a frozen *"this layer does not overlap"* that the layer has since outgrown.

    A frozen record answers the overlap question for every layer, and a ``None`` plan is one of
    its two answers -- "looked at while the numbers were concrete, and nothing overlapped" --
    rather than an absence. That distinction is what
    ``test_a_frozen_geometry_answers_the_overlap_question_for_every_layer`` settled; this is its
    other half, because an answer that is recorded can also stop being true. Move a shape until
    it does overlap and the answer is exactly as stale as a plan whose loops have been re-cut.

    It used to go stale in a worse place. The dispatch below summed on the record's word, and
    :func:`_refuse_overlap` stopped it there with a ``ValueError`` about *this* builder --
    naming neither the frozen record nor the fix, and pointing at the boundary route the caller
    was already trying to reach. Reproduced by freezing a disjoint rectangle-and-circle layer and
    sliding the rectangle onto the circle; the same call without ``geometry=`` returns
    0.933226529 quite happily.

    Eagerly the question can simply be re-asked, which is the same test the unfrozen call makes
    and costs one primitive overlap check. Under a trace there is nothing to re-ask -- the limit
    every other check on this record has.
    """
    try:
        needs = layer_needs_resolving(layer, lattice)
    except ValueError:
        return                      # traced: no concrete values to re-ask with
    if not needs:
        return
    raise StaleGeometryError(
        f'the frozen geometry no longer fits this structure: it recorded layer {index} as '
        f'not overlapping, and its shapes overlap now. That record is what routes '
        f'the layer, so replaying it would sum a layer that has to be painted -- a different '
        f'structure. Call freeze_geometry() again, or drop geometry= to route by comparison. '
        f'(Moving a shape never trips this on its own; moving it into another shape does.)')


def check_gamma_plan_still_fits(layer, lattice, plan, index, dyadic=False):
    """Refuse a frozen interface set that the layer in front of it has outgrown.

    With ``dyadic=True`` only the shape/image/reduction decisions are read, so
    contact and dead-face changes do not make the circular field's plan stale.
    The signed-pair restrictions below apply to the default linear field.

    A :class:`GammaPlan` records *combinatorics* -- which face lies on which image's face,
    over a run, and which faces that leaves covered end to end (``dead``) -- and deliberately
    holds no coordinate and no material, which is what lets a replay carry a gradient to a
    shape that has moved and a
    material that has swept. What makes it stale is the combinatorics changing: a face leaving
    its partner's line (the recorded pair would then cancel two faces that are interfaces), a
    run appearing (two interfaces would be kept where the material runs on), a shape growing
    across a ring boundary of its image list (the list is bounded by size alone, so a grown
    shape can lack its nearest instance), a basis whose reduction changed (every offset then
    names another image). Nothing about the answer says so -- it comes back unit-length,
    energy-conserving and wrong.

    Re-planning is the whole cost of the comparison, microseconds for a handful of shapes.
    Under a trace it is skipped -- a tracer has no primal to re-plan from, the same reading of
    "not concrete" :func:`_primitives` takes -- and the concreteness is probed *before* the
    planner is called, so a jitted replay never plans at all. ``None`` recorded for a layer
    means nothing was decided for it here, and nothing is checked: its shapes were traced at
    the freeze, or it holds a polygon and is built on the boundary route, whose own record
    (:func:`~wavebit.rcwa._engine.boundary.check_plan_still_fits`) is what goes stale for it.
    """
    if plan is None:
        return
    shapes = layer.get('shapes', [])
    try:
        _cell_floats(lattice)
        concrete = _primitives(shapes, lattice) is not None
    except Exception:                                  # noqa: BLE001 - a traced basis
        return
    if not concrete:
        return
    fresh = plan_gamma(shapes, lattice)
    if fresh is None or fresh == plan:
        return
    if dyadic:
        # Circular projection reads image lists, but never the signed pair
        # substitutions or dead-face flags. Contact can change without changing
        # this build's topology; larger image bounds still require refreezing.
        if (fresh.reduction == plan.reduction and len(fresh.pieces) == len(plan.pieces)
                and (not any(p.face == 'arc' for p in fresh.pieces) or fresh.ring <= plan.ring)
                and all(a.shape == b.shape and a.face == b.face
                        and (a.face == 'arc' or set(a.imgs).issubset(b.imgs))
                        for a, b in zip(fresh.pieces, plan.pieces))):
            return
        raise StaleGeometryError(
            f'circular NVF image plan changed for layer {index}; freeze_geometry() again')
    raise StaleGeometryError(
        f'the frozen geometry no longer fits this structure: {_gamma_moved(plan, fresh, shapes)}. '
        f'The recorded interface set of layer {index} describes faces this layer no longer '
        f'has where it has them, so replaying it would cancel an interface or keep one that '
        f'the material runs through. Call freeze_geometry() again, or drop geometry= to plan '
        f'eagerly. (Moving a shape never trips this on its own, and neither does a material '
        f'sweep; a face leaving its partner\'s line, or a shape growing into a wider ring of '
        f'images, does.)')


def _gamma_moved(held, fresh, shapes):
    """The one clause of the refusal that names what the re-plan found, in the layer's terms."""
    def name(plan, pr):
        p, q = plan.pieces[pr.p], plan.pieces[pr.q]
        return (f"shape {_shape_name(shapes, p.shape)}'s {p.face} face against shape "
                f"{_shape_name(shapes, q.shape)}'s {q.face} face at image ({pr.m}, {pr.n})")

    held_pairs = {(held.pieces[pr.p].shape, held.pieces[pr.p].face, held.pieces[pr.q].shape,
                   held.pieces[pr.q].face, pr.m, pr.n): pr for pr in held.pairs}
    fresh_pairs = {(fresh.pieces[pr.p].shape, fresh.pieces[pr.p].face, fresh.pieces[pr.q].shape,
                    fresh.pieces[pr.q].face, pr.m, pr.n): pr for pr in fresh.pairs}
    clauses = []
    for key in sorted(set(fresh_pairs) - set(held_pairs)):
        clauses.append(f'a pair appeared, {name(fresh, fresh_pairs[key])}')
    for key in sorted(set(held_pairs) - set(fresh_pairs)):
        clauses.append(f'a pair vanished, {name(held, held_pairs[key])}')
    for old, new in zip(held.pieces, fresh.pieces):
        if old.dead != new.dead:
            clauses.append(f"shape {_shape_name(shapes, new.shape)}'s {new.face} face is "
                           f"{'covered by its own images now' if new.dead else 'an interface again'}")
        if old.imgs != new.imgs and old.face in ('arc', '+u'):
            why = ('its support crossed the edge of its Wigner-Seitz cell' if old.face == 'arc'
                   else 'its size crossed a ring boundary')
            clauses.append(f"shape {_shape_name(shapes, new.shape)}'s image list changed "
                           f"({len(old.imgs)} -> {len(new.imgs)} instances: {why})")
    if held.ring != fresh.ring:
        clauses.append(f'the ring of images swept went from {held.ring} to {fresh.ring}')
    if held.reduction != fresh.reduction:
        clauses.append(f'the basis reduces differently now ({held.reduction} -> '
                       f'{fresh.reduction}), so every recorded offset names another image')
    if held.empty != fresh.empty:
        clauses.append('the layer ' + ('tiles the plane now' if fresh.empty else
                                       'no longer tiles the plane'))
    return '; '.join(clauses) or 'the recorded plan differs from a fresh one'


def to_conv_mat_parametric_layers(layer_shapes, half_width_u, half_width_v, lattice,
                              shape_nvf_grid_size=256, complex_dtype=None, xp=None,
                              boundary_plans=None, gamma_plans=None, nvf_projector=None,
                              factorization=None):
    """Stack the parametric convolution matrices over layers, under NVF or Laurent's rule.

    ``layer_shapes`` is one ``dict(background=..., shapes=[...])`` per layer, with the shapes'
    ``cx`` / ``cy`` / ``lx`` / ``ly`` as absolute lengths in the units of ``period`` (see the
    module docstring). Returns ``(epx, epy, epz_i, epxy, epyx)``, each
    ``(n_layers, n_orders, n_orders)``.

    Every layer goes through :func:`check_layer_spec` and :func:`check_layer_shapes` first —
    this is the funnel the parametric path reaches the solver through, so it is the one place that
    can guarantee a misspelled recipe is never answered with a plausible structure.

    The array namespace is taken from the first differentiable shape parameter found, so
    passing JAX or PyTorch scalars for e.g. ``lx`` makes the whole build run — and
    differentiate — on that backend. Pass ``xp`` to force it.

    ``boundary_plans`` and ``gamma_plans`` are the frozen record's two per-layer entries
    (:class:`~wavebit.rcwa._engine.geometry.StaticGeometry`): the painting's resolution for a layer
    that overlaps, and the interface set the closed form's normal field replays. Both are
    re-asked eagerly and replayed under a trace.

    ``factorization`` is ``'nvf'`` (``None`` means it) or ``'laurent'``. Laurent's rule takes
    the same ``[[eps]]`` from the same geometry -- the closed-form sum of shape transforms, or
    the painted polygons where shapes overlap -- and drops the normal field, so it reads the
    painting's plan and never ``gamma_plans``, ``shape_nvf_grid_size`` or ``nvf_projector``
    (:func:`laurent_from_coeffs`). Li's rules on shapes are :mod:`.rectilinear`'s, not this
    function's, and are refused here by name.
    """
    from wavebit.rcwa._engine.enums import Factorization
    from wavebit.rcwa._engine.enums import factorization_of
    rule = Factorization.NVF if factorization is None else factorization_of(factorization)
    if rule == Factorization.LI:
        raise ValueError("factorization='li' on shapes is built by wavebit.rcwa._engine.rectilinear "
                         "(axis-aligned rectangles only); this builder takes 'nvf' or 'laurent'")
    laurent = rule == Factorization.LAURENT
    if laurent and (nvf_projector is not None or gamma_plans is not None):
        raise ValueError("factorization='laurent' builds no normal field, so nvf_projector and "
                         "gamma_plans have no reader here; pass neither")
    if xp is None:
        xp = _infer_namespace(layer_shapes)
    dtype = xp.complex128 if complex_dtype is None else complex_dtype

    out = [[], [], [], [], []]
    for index, layer in enumerate(layer_shapes):
        check_layer_spec(layer, index)
        # A frozen geometry carries the decision for *every* layer, so when one is present it
        # is the whole answer: `plan is None` there means "looked at while the numbers were
        # concrete, and it does not overlap". Re-asking would put the question to traced
        # values that cannot answer it.
        frozen = boundary_plans is not None
        plan = boundary_plans[index] if frozen else None
        gamma = gamma_plans[index] if gamma_plans is not None else None
        if frozen and plan is None:
            check_no_plan_still_fits(layer, lattice, index)
        check_layer_shapes(layer, lattice, index)
        background = layer.get('background', 1.0)
        if plan is not None or (not frozen
                                and layer_needs_resolving(layer, lattice)):
            # Shapes that overlap paint -- later over earlier -- and this module can only
            # sum, which is a different structure. Resolve the painting into disjoint polygons
            # and transform those. Where nothing overlaps the two are the same structure, so
            # the layer stays here -- exact, and cheaper.
            from wavebit.rcwa._engine.boundary import to_conv_mat_boundary_layer
            options = {}
            if laurent:
                options['laurent'] = True
            elif nvf_projector == 'circular':
                if gamma_plans is not None:
                    check_gamma_plan_still_fits(layer, lattice, gamma, index,
                                               dyadic=nvf_projector == 'circular')
                if any(s.get('type', 'rectangle') in _overlap._ELLIPSE_TYPES
                       for s in layer.get('shapes', [])):
                    options.update(nvf_projector='circular', projector_plan=gamma)
                else:
                    options['projector_coeffs'] = normal_projector_coeffs(
                        xp, layer.get('shapes', []), (shape_nvf_grid_size, shape_nvf_grid_size), lattice,
                        half_width_u, half_width_v, dtype, plan=gamma, nvf_projector=nvf_projector)
            mats = to_conv_mat_boundary_layer(
                xp, layer.get('shapes', []), background, half_width_u, half_width_v, lattice,
                dtype, shape_nvf_grid_size=shape_nvf_grid_size, plan=plan, **options)
        else:
            # Reaching this branch means the shapes are disjoint, so the summing this
            # builder does is the painted structure. `assemble_eps_coeffs` re-checks that --
            # a second reading of the dispatch just above, on the one branch where getting it
            # wrong returns a plausible number instead of an error.
            # That failure has happened once already (see `layer_needs_resolving`). It held
            # only for the unfrozen route until `check_no_plan_still_fits` was added: a record
            # frozen while the shapes were disjoint routes by its own answer, not by the two
            # rules, and once the shapes moved together the check below is what fired -- with
            # a message about this builder rather than about the stale record.
            if laurent:
                eps_c, _ = assemble_eps_coeffs(xp, layer.get('shapes', []), background,
                                               half_width_u, half_width_v, lattice, dtype)
                mats = laurent_from_coeffs(xp, eps_c, half_width_u, half_width_v)
            else:
                if gamma_plans is not None:
                    check_gamma_plan_still_fits(layer, lattice, gamma, index,
                                                dyadic=nvf_projector == 'circular')
                mats = to_conv_mat_parametric_nvf(
                    xp, layer.get('shapes', []), background, half_width_u, half_width_v,
                    lattice, dtype, shape_nvf_grid_size=shape_nvf_grid_size, plan=gamma,
                    **({} if nvf_projector is None else {'nvf_projector': nvf_projector}))
        for slot, m in zip(out, mats):
            slot.append(m)

    # Laurent leaves the off-diagonal pair empty on every layer, as Li's rules do.
    return ConvMatrices(*(None if slot[0] is None else xp.stack(slot) for slot in out))


def _material_value(value):
    """A material slot's ``epsilon``, or the raw value, or ``None`` -- for namespace sniffing.

    The point is to look *inside* a :class:`~wavebit._common.periodic.Material`. A material is a
    legitimate design variable, so its ``epsilon`` may be the only backend array in a layer
    written entirely with Python floats; missing it would build the whole layer in NumPy and
    drop the gradient at the very parameter being optimised.
    """
    if value is None:
        return None
    return as_material(value).epsilon


def _infer_namespace(layer_shapes):
    """Namespace of the first array-like shape parameter; NumPy when everything is a scalar."""
    for layer in layer_shapes:
        candidates = [_material_value(layer.get('background'))]
        for shape in layer.get('shapes', []):
            candidates.extend(shape.get(k) for k in
                              ('cx', 'cy', 'lx', 'ly', 'l', 'angle'))
            candidates.append(_material_value(shape.get('material')))
        for c in candidates:
            if c is not None and hasattr(c, 'dtype') and hasattr(c, 'shape'):
                return array_namespace(c)
    import numpy as np
    return array_namespace(np.zeros(1))
