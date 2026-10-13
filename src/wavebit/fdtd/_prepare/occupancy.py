"""The one lowering stage between the geometry and the coefficients.

``src/wavebit/fd/_shapes`` answers for a *point*: given a body and an array of points, what
fraction of the voxel around each point the body fills and which way its surface faces there.
This module is the other end of that wire. It knows what a scene is, what a grid is and where
each of the six field components sits, and it turns "structures[2] is a ball of radius 0.2"
into six arrays of ``(fill, w_c)`` -- one per component -- over the cells that ball can reach,
on the run's backend, differentiable in the body's own leaves.

Nothing else in the package knows what a sphere is. ``coefficients.py`` is handed pairs of
numbers and a rule for mixing two media; ``checks.py`` is handed the doors; the kernel sees
neither. That is the whole reason this file exists as a stage rather than as three lines
inside :func:`wavebit.fdtd._prepare.coefficients.build`.

**Each component is asked at its own Yee point, and that is not a refinement.** ``Ez`` sits
half a cell from ``Ex`` and ``Ey`` along z; ``Hx`` sits half a cell along both y and z. This
repository has already paid once for sampling a material at the wrong one of those: the legacy
engine read ``Ey``'s permittivity at the half position and ``Ey``'s pole state at the integer
one, and a published crossed-metal absorption moved 0.400 -> 0.433 against an RCWA 0.431
(memory ``legacy-ey-pole-fill-half-cell``, fixed in ``a6cb803``). So ``eps`` is built from the
``E`` points and ``mu`` from the ``H`` points, each component separately, and
``tests/fdtd/test_fd_occupancy.py::test_each_component_reads_the_interface_at_its_own_yee_point``
is the gate with the cell-centre mutant beside it.

**What is *not* asked here.** ``sigma`` and the pole coefficients are not smoothed in C2. They
keep the cell-snapped rule they have today, and a fractional edge has no cell-snapped rule, so
``checks.check_smoothed`` refuses a conductive or dispersive material anywhere a smoothed
shape can reach rather than letting one through unsmoothed. The design prices the conductive
half -- arithmetic ``sigma`` on every component, exact for the two tangential ones and first
order on the normal one, with ``docs/validation/fdtd/fdtd-oblique-pulse.md``:84-88's TM 0.0595 against TE
0.0082 at resolution 40 as the analogous legacy inconsistency -- and the dispersive half is
the exact two-pole Kottke realization. Both wait for the commit that measures them (C8), and
neither is guessed at here.

**The periodic image ring is not used in C2, and the door says so.** ``_shapes.occupancy``
takes a ``period=`` and sums the body over a 3x3x3 ring of images, which is what a body
straddling a ``Periodic`` face needs. It is exact only while the body lies *in* the cell, and
a body carries no position to fold with (a CSG node has none), so the fold belongs to the
caller that owns the cell -- which is this one. Rather than half-build it,
``checks.check_smoothed`` refuses a smoothed shape that reaches a ``Periodic`` or ``Bloch``
face at all -- unless it is **uniform along** that axis over the whole domain, which an
extruded body along its own axis or an unturned ``fd.Box`` spanning it is (C4): every layer then
holds the same section, the wrapped part is the part already there, and no fold is needed. The
fold itself, for a body that crosses a periodic face and changes across it, is still not
written. The third wrapping
face, ``Axis1D``, is refused there too but for a different reason and with its own sentence:
its axis stands for one the field is *invariant* along, so no fold would make a body that
curves right on it. Measured, with the door held open:
``Sphere(radius=0.2)`` centred on the ``x-`` face of a 1.2-wide periodic cell lowers **59.3%**
of its volume (the same ball at the centre of the cell reads 1.0000), and the missing 40.7% --
the part that should have wrapped to the far face -- is silent: ``sum_c w_c`` still reads 1 to
4.4e-16 on every voxel that has any surface in it, so no invariant sees it.

**Approximation reports belong to the static scene.** ``checks.notes`` evaluates the
geometry on NumPy while planning, with one ``Approximations`` collector for the whole scene.
It warns once with a count, width and operation-specific direction; ``fd.check`` lists the
same note even when given prebuilt params. A Union or Difference takes an upper volume bound,
whereas an Intersection takes a lower one. A blanket claim that all bounds add material is
false. Numerical lowering does not inspect Python counters or emit warnings: a JIT trace and
its cached execution must use the same already-checked geometry.
"""
import numpy as np

from wavebit.fdtd._kernel.grid import OFFSETS
from wavebit.fdtd._shapes import occupancy as body_occupancy

__all__ = ['occupancy', 'smoothed_structures', 'component_occupancy', 'yee_points',
           'bound_note']


def smoothed_structures(scene):
    """``((index, structure, body), ...)`` for every structure whose shape carries a body.

    The one dispatch on "is this shape smoothed", written once, asked as a question of the
    shape (``Shape.body()``) and never as an ``isinstance``. An empty answer is the common
    case and is what keeps every stage-0..3 scene on exactly the code it had.
    """
    out = []
    for index, structure in enumerate(getattr(scene, 'structures', ()) or ()):
        body = structure.shape.body()
        if body is not None:
            out.append((index, structure, body))
    return tuple(out)


def yee_points(grid, box, field, c):
    """``(nz, ny, nx, 3)`` of the points component ``c`` of ``field`` occupies inside ``box``.

    ``box`` is the shape's support, ``((i0, i1), (j0, j1), (k0, k1))`` in x, y, z. The array
    axes are ``[z, y, x]`` and each point is a typed x-first triple ``(x, y, z)``: the two
    axis-order rules this repository keeps, meeting in one function (memory
    ``two-axis-order-rules-that-meet``). Static NumPy -- a grid coordinate is never a leaf --
    and cast to the run's backend by the caller.
    """
    axes = []
    for a in range(3):
        lo, hi = box[a]
        axes.append(grid.low[a] + (np.arange(lo, hi) + 0.5 * OFFSETS[field][c][a]) * grid.dx)
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
    return np.stack([x, y, z], axis=-1)


def bound_note(report):
    """One scene note from the geometry kernel's actual report interface.

    A traced report is unknown, not empty. Planning must read a concrete geometry before
    compiled lowering begins; otherwise the warning would disappear on cached JIT calls.
    """
    if report.voxels is None or report.width is None:
        raise ValueError('smoothed geometry cannot be checked while its approximation report '
                         'is traced; build fd.plan from concrete geometry before lowering. '
                         'Traced geometry with a declared range is stage 4 C6.')
    return report.message()


def component_occupancy(body, grid, box, field, c, rt, report=None):
    """``(fill, w_c)`` over ``box`` at component ``c`` of ``field``'s own points.

    Both are ``(nz, ny, nx)`` arrays of the runtime's kind. ``w_c`` is the ``c``-th diagonal
    entry of the interface normal's second moment ``<n n^T>`` -- the component's *own*
    direction, because the mixing rule asks how much of the field at this point runs across
    the interface rather than along it.

    The points are built in NumPy and handed to the backend once. That matters for tracing:
    ``_shapes`` picks its namespace from the points it is given, so a body with a traced
    radius and NumPy points would try to run JAX tracers through ``numpy.where``. It is also
    the cheaper order -- one host-to-backend transfer per component instead of three
    ``arange``s and a ``meshgrid`` inside the graph.
    """
    points = rt.array(yee_points(grid, box, field, c), 'float64')
    fill, w = body_occupancy(body, points, grid.dx, report=report)
    return fill, w[..., c]


def occupancy(scene, plan, rt, fields=('E', 'H'), report=None):
    """``{index: {'box': box, 'E': ((f, w) x3), 'H': ((f, w) x3)}}`` per smoothed structure.

    ``fields`` is what the caller will actually spend. ``coefficients.build`` asks for ``H``
    only when something in the scene is magnetic, because ``bh`` is the scalar ``dt/dx``
    otherwise and the three ``H``-point evaluations would be thrown away. Measured for a ball
    at 12 cells per radius in a 64^3 box, E only against E and H: **2.04x / 1.97x / 2.13x** on
    numpy / jax / torch. That is the whole of the saving, and it is why the argument exists
    rather than this always being ``('E', 'H')``.

    **What the stage costs, and where.** The same scene lowers in 0.4211 s on numpy, 0.6892 s
    on jax and 0.3112 s on torch, against 0.0066 / 0.0074 / 0.0050 s for a cell-snapped
    ``fd.Region`` of the same bounding size -- 62 to 93 times. Per component array of 17 576
    voxels on numpy: building the points 0.0003 s, the body's ``exact_fill`` **0.1181 s**, the
    tangent plane that carries the moment 0.0032 s. So **97% of it is the exact clipped-ball
    volume**, at 6.7 us a voxel, and neither the moment nor the mixing rule is worth looking at
    until that is. It grows as ``(R/h)^3``: at 40 cells per radius the same ball is 512 000
    voxels per component and the stage is about twenty seconds. The peak memory adds nothing
    at all -- the fills are support-box sized (0.14 MB a component here) and the RSS the
    lowering takes is the coefficient arrays', 6.3 MB either way.

    **``report``**: an :class:`~wavebit.fdtd._shapes.Approximations` the caller may hand in to be
    told about voxels a body answered with a bound rather than exactly. One accumulator spans
    every structure and every component, because that is the granularity the user is warned at
    -- planning passes one and turns it into a scene note. Numerical lowering leaves it out,
    so neither reduction values nor warning side effects are read inside a compiled call.
    """
    grid = plan.grid
    out = {}
    for index, structure, body in smoothed_structures(scene):
        box = structure.shape.support(grid)
        entry = {'box': box}
        for field in fields:
            entry[field] = tuple(component_occupancy(body, grid, box, field, c, rt, report)
                                 for c in range(3))
        out[index] = entry
    return out
