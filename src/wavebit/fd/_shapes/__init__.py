"""The geometry kernel: what fraction of a voxel a body fills, and which way its surface faces.

Stage 4's one idea is that a shape stops answering "which cells do you own" and answers, per
component's **own Yee point**, a pair::

    occupancy(body, points, dx) -> (fill, w)      # fill in [0, 1]; w_c = <n_c^2>, sum_c w_c == 1

``fill`` is the fraction of the voxel centred on that point that the body holds. ``w_c`` is the
mean of ``n_c^2`` over the body's surface inside that voxel -- the diagonal of the interface
normal's **second** moment ``<n n^T>``. The mixing rule, CSG, the differentiable edge and the
absorbing-layer question all fall out of that pair, and nothing else in stage 4 asks a body
anything.

**Why the second moment and not the first.** A first moment ``N``, normalised as
``N N^T/|N|^2``, collapses to rank one at a symmetric 90-degree corner --
``[[1/4, 1/4], [1/4, 1/4]]``, a spurious diagonal normal, where ``n n^T`` reads
``diag(1/2, 1/2)`` -- and cancels to zero between the antiparallel faces of a feature thinner
than a cell, which is exactly the laminate for which ``eps = <eps> I - Delta n n^T`` is exact.
The defect is on record in this repository at ``src/wavebit/_core/scene_fdtd.py``:844-846, where the
paragraph ends "it is not a factor of two waiting to be simplified away". It is a named gate
mutant here (``tests/test_fd_shapes.py::test_the_exact_face_moments_are_the_surfaces_own``,
which also records where the two rules do **not** differ: at a symmetric 90-degree corner both
read a diagonal of ``(1/2, 1/2)``, so the case the design's wording names would not have caught
it -- what does is a corner of unequal faces, and a feature thinner than a cell, where the first
moment cancels to exactly zero and has no direction left to normalise).

**The rule this pair feeds**, which is C2's and not this package's, is the diagonal of the
Farjadpour/Kottke effective tensor's inverse::

    1/eps_c = w_c (f/eps_a + (1-f)/eps_b) + (1 - w_c) / (f eps_a + (1-f) eps_b)

Farjadpour et al., Opt. Lett. **31**, 2972 (2006); Kottke, Farjadpour & Johnson, Phys. Rev. E
**77**, 036611 (2008); Oskooi et al., Comput. Phys. Commun. **181**, 687 (2010). The tensor form
is quoted from Meep's **documentation**; no Meep source is read at any point, and the
Meep-derived FDTDX projection file this repository has already excluded
(``docs/fdtd-layer-smoothing.md``:74, ``docs/fdtd-layer-gradients.md``:72) stays excluded.

**What is here**

``_cut.py``
    ``cut(d, n, dx) -> (fill, w)``: a cube cut by a half space, the C1-continuous closed form,
    written as a divided difference taken in a stable order so that the 2-D, 1-D and 0-D limits
    are exact rather than tolerance-branched. The tangent plane's measured ``+0.250 (h/R)^2``
    over-fill of a sphere lives in that docstring, beside the table it is fitted from. The same
    cut one and two dimensions down -- ``edge_cut(d, n, dx, axis)`` on a component's own edge
    and ``face_cut`` on its dual face, which is the pair the Werner/Bauer/Cary material operator
    constrains and which a CSG node can answer where an exact clip cannot exist. They are a
    **preparation**, reached from no solver file: nothing dispatches to them, and the one body
    that has closed forms for the same two quantities (``Sphere``) carries them itself.
``_bodies.py``
    ``Sphere``, ``Cylinder``, ``Prism``, ``Box`` as signed distance plus unit gradient, and
    ``Union`` / ``Intersection`` / ``Difference`` composed on the distance, resolved once,
    before any fill is taken -- except where two children's surfaces **touch**, which the
    composed distance gets wrong by half a voxel and which a per-pair inclusion-exclusion
    answers exactly. ``_Combination`` carries that rule, what it costs where it does not
    reach, and the measured price of the exact multi-plane cut that would replace it.
``_exact.py``
    The closed forms this repository already owns, **ported** with their guards and their
    sources named: the 1-D overlap, the disc-box clip, the polygon-box clip and the polygon's
    edge moments. Nothing under ``src/wavebit/fd/`` imports the old engines
    (``src/wavebit/fd/_lower/cpml.py`` states that rule), and two of the four needed generalising
    from four vertices to ``n`` anyway, which the originals refuse by name. The ball-box clip is
    the one form here that is *not* a port, because there was nothing to port.
``_occupancy.py``
    The 3x3x3 periodic image ring, and the one call that puts it all together.

**This package was reachable from no solver file in the commit that added it.** C2 wires it
into ``src/wavebit/fd/_lower/occupancy.py``, a new lowering stage named after this package's one
entry point, and through it into the coefficients; ``src/wavebit/fd/geometry.py`` builds a body
from a shape and imports nothing else here. It was built and gated first, on its own, because
every number stage 4 claims stands on it and because a geometry bug found through a scattering
cross-section costs a 400-second run to see.

**Conventions.** Points and normals are typed x-first triples ``(..., 3)``; the ``[component,
z, y, x]`` array layout is C2's business and nothing here has an opinion about it (memory: two
axis-order rules that meet -- arrays v-first, typed pairs x-first). Lengths are absolute scene
coordinates, never fractions of a period. ``dx`` is one number: a non-uniform grid is not in
v1. Everything runs on NumPy, JAX and Torch from one body of code, and every selection is an
array ``where`` and never a Python ``if`` on a value -- so a traced radius and a concrete one
take the same path, which is what makes ``jax.grad`` and a finite difference compute the same
function.
"""
from ._bodies import (AXES, Approximation, Approximations, Body, Box, Cylinder, Difference,
                      Intersection, Prism, Sphere, Union)
from ._cut import cut, cut_parts, edge_cut, face_cut
from ._exact import (clip_capacity, clipped_ball_volume, clipped_disc_area, clipped_polygon_area,
                     overlap, polygon_edge_moments)
from ._occupancy import image_shifts, occupancy

__all__ = ['AXES', 'Approximation', 'Approximations', 'Body', 'Box', 'Cylinder', 'Difference', 'Intersection', 'Prism', 'Sphere',
           'Union', 'cut', 'cut_parts', 'edge_cut', 'face_cut', 'clip_capacity', 'clipped_ball_volume',
           'clipped_disc_area', 'clipped_polygon_area', 'overlap', 'polygon_edge_moments',
           'image_shifts', 'occupancy']
