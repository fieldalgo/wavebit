"""Vector geometry modeler (fieldwave._core.modeler) against the frozen legacy output.

The modeler decomposes parametric shapes — rectangles and ellipses, optionally rotated — into
the piecewise-constant blocks the vector factorization consumes. Its reference values were
captured from the legacy per-backend implementation before that was deleted
(gen_goldens_modeler.py -> golden_legacy_modeler.npz).

The ellipse is the exception — see ``LEGACY_PARITY`` below and test_modeler_ellipse.py.
"""
import pathlib

import importlib

import numpy as np
import pytest

import fieldwave
from fieldwave._core.modeler import Modeling

GOLDEN = np.load(pathlib.Path(__file__).parent / 'golden_legacy_modeler.npz')

PERIOD = [300.0, 300.0]
SI, SIO2, SI3N4 = 3.638751670074983, 1.4518, 2.0056

# Must stay identical to gen_goldens_modeler.py.
INSTRUCTIONS = {
    'rect_1d': [[SI3N4, [['rectangle', 50, 150, 31, 300, SI, 0, 0, 0]]], [SI, []]],
    'rect_2d': [[SI3N4, [['rectangle', 80, 120, 60, 90, SI, 0, 0, 0],
                         ['rectangle', 200, 180, 50, 70, SIO2, 0, 0, 0]]], [SI, []]],
    'rect_rot': [[SI3N4, [['rectangle', 150, 150, 80, 60, SI, 0.4, 4, 4]]], [SIO2, []]],
    'ellipse': [[SI3N4, [['ellipse', 150, 150, 90, 60, SI, 0.0, 4, 4]]], [SIO2, []]],
    'ellipse_rot': [[SI3N4, [['ellipse', 140, 160, 100, 50, SI, 0.5, 4, 4]]], [SIO2, []]],
}
SOLVE_KW = dict(period=PERIOD, thickness=[205.0, 100.0], fto=[3, 3],
                n_top=1.0, n_bot=1.0, wavelength=900.0, pol=0.5)

# The ellipse is deliberately not held to its legacy golden. The decomposition those numbers
# were captured from did not converge — refining the contour changed the structure instead of
# resolving it, so the efficiency swung between 0.24 and 0.68 as ``n_split`` grew, and one
# family of splits deleted the shape outright. Reproducing that is not a property worth
# keeping, so the ellipse is checked against geometry in test_modeler_ellipse.py instead. The
# rectangle, which does converge, keeps its parity here.
LEGACY_PARITY = [name for name in INSTRUCTIONS if not name.startswith('ellipse')]

# ``rect_rot`` is held to the golden through the solve and through the geometry, but not
# array-for-array: it is the only case here whose blocks touch, and blocks that touch now share
# a grid line instead of getting a nudged one each. Its grid is shorter (and sorted) as a
# result. See the dedicated test below, which pins what did not change — the material anywhere
# in the layer — and what did.
BITWISE_PARITY = [name for name in LEGACY_PARITY if name != 'rect_rot']


#: Offsets the sample points off any round coordinate. Landing exactly on a grid line makes
#: the lookup depend on which side of that line the sampler rounds to, which is precisely what
#: differs between a grid with a zero-width band at a boundary and one without — at 601 evenly
#: spaced samples one row sat exactly on ``cy`` and reported a difference that was not there.
_OFF_LATTICE = 0.3183098861837907


def material_at(cell, x, y, n=600, period=300.0):
    """The layer's refractive index sampled on an ``n`` x ``n`` grid of points.

    Comparing *this* rather than the raw arrays is what survives a change in how grid lines are
    ordered or how many of them a boundary is spelled with: it asks what the solver would see
    at a position, which is the thing that must not move.
    """
    cell = np.asarray(cell)
    x_edges = np.real(np.asarray(x)).reshape(-1)
    y_edges = np.real(np.asarray(y)).reshape(-1)

    # Sorted here rather than assumed, so a legacy array with rows out of order can be sampled
    # on the same footing as a fixed one.
    order_y = np.argsort(y_edges, kind='stable')
    order_x = np.argsort(x_edges, kind='stable')
    y_edges, x_edges = y_edges[order_y], x_edges[order_x]
    cell = cell[order_y][:, order_x]

    points = (np.arange(n) + _OFF_LATTICE) / n * period
    iy = np.clip(np.searchsorted(y_edges, points, 'left'), 0, cell.shape[0] - 1)
    ix = np.clip(np.searchsorted(x_edges, points, 'left'), 0, cell.shape[1] - 1)
    return cell[np.ix_(iy, ix)]


@pytest.mark.parametrize('name', BITWISE_PARITY)
def test_vector_blocks_match_legacy_golden(name):
    """The decomposed blocks and their edges, layer by layer."""
    info = Modeling(period=PERIOD).modeling_vector_instruction(INSTRUCTIONS[name])
    assert len(info) == int(GOLDEN[f'{name}__n_layers'])
    for i, (cell, x, y) in enumerate(info):
        for tag, got in (('cell', cell), ('x', x), ('y', y)):
            want = GOLDEN[f'{name}__L{i}__{tag}']
            got = np.asarray(got)
            assert got.shape == want.shape, (i, tag, want.shape, got.shape)
            assert np.array_equal(want, got), (i, tag, np.max(np.abs(want - got)))


def test_rotated_rectangle_grid_keeps_the_legacy_layer_with_fewer_lines():
    """The grid drops its duplicate lines and comes back sorted, without moving any material.

    The stored golden has one grid line per block edge, including a near-duplicate wherever two
    blocks meet — the builder used to nudge the second edge off the first by a relative 1e-14
    rather than let them share a line, and inserted it at the slot found *before* the nudge, so
    the list was not even sorted (``rect_rot__L0__y`` has two inversions, and a negative
    segment width is what ``_cfs`` makes of one).

    The claim here is narrow and checkable: the layer is the same object, described with fewer
    lines. The material at 360000 points across both layers is identical, the edges that remain
    are a subset of the legacy ones to within the old nudge, and
    ``test_vector_solve_matches_legacy_golden[rect_rot]`` still holds the efficiencies to 1e-12.
    """
    info = Modeling(period=PERIOD).modeling_vector_instruction(INSTRUCTIONS['rect_rot'])

    for i, (cell, x, y) in enumerate(info):
        want_y = np.asarray(GOLDEN[f'rect_rot__L{i}__y']).reshape(-1).real
        got_y = np.asarray(y).reshape(-1).real
        got_x = np.asarray(x).reshape(-1).real

        assert np.all(np.diff(got_y) > 0), i
        assert np.all(np.diff(got_x) > 0), i
        assert len(got_y) <= len(want_y), i

        # Every surviving line was a legacy line (up to the nudge the legacy applied to it).
        nearest = np.min(np.abs(got_y[:, None] - np.sort(want_y)[None, :]), axis=1)
        assert np.max(nearest) < 1e-9, (i, np.max(nearest))

        assert np.array_equal(material_at(cell, x, y),
                              material_at(GOLDEN[f'rect_rot__L{i}__cell'],
                                          GOLDEN[f'rect_rot__L{i}__x'],
                                          GOLDEN[f'rect_rot__L{i}__y'])), i

    # The golden really is the longer, unsorted one — otherwise this test proves nothing.
    assert len(np.asarray(GOLDEN['rect_rot__L0__y'])) > len(np.asarray(info[0][2]))
    assert not np.all(np.diff(np.asarray(GOLDEN['rect_rot__L0__y']).reshape(-1).real) > 0)


@pytest.mark.parametrize('angle,swapped', [(0.0, False), (np.pi / 2, True), (np.pi, False),
                                           (3 * np.pi / 2, True), (2 * np.pi, False)])
@pytest.mark.parametrize('n_split', [0, 1])
def test_axis_aligned_rectangle_is_the_exact_single_block(angle, swapped, n_split):
    """An aligned rectangle needs no decomposition and does not get one.

    It used to reach that through an ``angle_margin`` special case. Now it falls out: the two
    triangles have zero height at an aligned angle, so only the band is left, and one slab
    across the band *is* the rectangle. Held to ``rectangle_no_approximation`` bit-for-bit,
    including the side swap past a quarter turn.
    """
    modeler = Modeling(period=PERIOD)
    got = modeler.rectangle(150.0, 150.0, 80.0, 60.0, 2.0, angle, n_split, n_split)
    lx, ly = (60.0, 80.0) if swapped else (80.0, 60.0)
    want = modeler.rectangle_no_approximation(150.0, 150.0, lx, ly, 2.0)

    assert len(got) == 1
    for got_corner, want_corner in zip(got[0][:2], want[0][:2]):
        for g, w in zip(got_corner, want_corner):
            assert float(np.real(np.asarray(g).reshape(-1)[0])) == float(np.real(w))


@pytest.mark.parametrize('angle', [0.0, 1e-9, 0.05, -0.05, 0.4, -0.4, 1.0, 2.5, 4.0,
                                   np.pi / 2, -np.pi / 2])
def test_rectangle_blocks_sum_to_the_true_area(angle):
    """The blocks account for ``lx*ly`` exactly, at every angle.

    The chord is linear in height between the two inner corners, and those corners are slab
    boundaries, so reading the chord at the middle of a slab gives that slab its exact area.
    """
    blocks = Modeling(period=PERIOD).rectangle(150.0, 150.0, 80.0, 60.0, 2.0, angle, 4, 4)
    f = lambda v: float(np.real(np.asarray(v).reshape(-1)[0]))
    area = sum((f(b[1][0]) - f(b[0][0])) * (f(b[1][1]) - f(b[0][1])) for b in blocks)

    assert area == pytest.approx(80.0 * 60.0, rel=1e-12)


def test_rectangle_has_no_step_where_the_angle_margin_used_to_end():
    """Crossing 1e-5 rad no longer moves the answer on its own.

    The margin returned the exact block below 1e-5 and the decomposition above it, so the
    efficiency sat on its ``angle = 0`` value across the window and then dropped 6.75e-5 in one
    step at the edge — a step that did not shrink from ``n_split`` 4 to 1024. Here the slope
    across that crossing has to match the slope just outside it.
    """
    cfg = dict(period=PERIOD, thickness=[205.0], fto=[3, 3], n_top=1.0, n_bot=1.0,
               wavelength=900.0, pol=0.5)

    def de_ti(angle):
        ucell = [[3.5, [['rectangle', 150.0, 150.0, 80.0, 60.0, 2.0, angle, 8, 8]]]]
        return float(np.sum(np.asarray(
            fieldwave.rcwa(backend=0, ucell=ucell, **cfg).conv_solve().de_ti)))

    across = (de_ti(1.02e-5) - de_ti(0.98e-5)) / (0.04e-5)
    outside = (de_ti(1.10e-5) - de_ti(1.06e-5)) / (0.04e-5)

    assert across == pytest.approx(outside, rel=1e-3), (across, outside)


def test_rectangle_angle_carries_gradient_near_an_aligned_angle():
    """``backward()`` used to raise here, not return zero.

    Inside the margin the exact branch never touched ``angle``, so the graph was cut and torch
    reported ``element 0 of tensors does not require grad``. 1e-5 rad is inside that window.

    At an exactly aligned angle the derivative is still not the one-sided slope: the two
    triangles appear only once the angle leaves zero, so the structure differs on the two sides
    and autograd returns a subgradient of the kink. That is one point rather than a window, and
    it is checked separately below.
    """
    torch = pytest.importorskip('torch')
    tensor = lambda v: torch.tensor(v, dtype=torch.float64)
    cfg = dict(period=PERIOD, thickness=[205.0], fto=[3, 3], n_top=1.0, n_bot=1.0,
               wavelength=900.0, pol=0.5)

    def de_ti(angle):
        ucell = [[3.5, [['rectangle', tensor(150.0), tensor(150.0), tensor(80.0),
                         tensor(60.0), 2.0, angle, 8, 8]]]]
        return torch.sum(fieldwave.rcwa(backend=2, ucell=ucell, **cfg).conv_solve().de_ti)

    p = torch.tensor(1e-5, dtype=torch.float64, requires_grad=True)
    out = de_ti(p)
    assert out.requires_grad, 'angle left the graph — the margin is back'
    out.backward()
    auto = float(p.grad)

    step = 5e-6
    with torch.no_grad():
        fd = float((de_ti(tensor(1e-5 + step)) - de_ti(tensor(1e-5 - step))) / (2 * step))

    assert abs(fd) > 1.0, fd
    assert auto == pytest.approx(fd, rel=1e-3), (auto, fd)


@pytest.mark.parametrize('shape,args', [
    ('ellipse', (350., 350., 400., 250., 3.0, 0.0, 24, 24)),     # symmetric: edges coincide
    ('ellipse', (350., 350., 400., 250., 3.0, 0.0, 25, 25)),     # the other parity
    ('ellipse', (350., 350., 400., 250., 3.0, 0.4, 80, 80)),
    ('rectangle', (150., 150., 80., 60., 2.0, 0.4, 16, 16)),
    ('rectangle', (150., 150., 80., 60., 2.0, 0.4, 64, 64)),
])
def test_grid_lines_come_back_strictly_increasing(shape, args):
    """``x_list`` / ``y_list`` are segment edges; out of order they are a negative width.

    Blocks that meet at a coordinate share one line, so a stack of ``n`` slabs spends ``n + 1``
    rows and not ``2n + 1``. The unrotated ellipse is the sharp case in both respects: ``sin``
    is symmetric about pi/2, so slab ``i`` and slab ``n - 1 - i`` also produce bit-identical
    *side* edges. It used to spend 49 lines each way at ``n_split`` 24, ten of the columns
    inverted.
    """
    modeler = Modeling(period=[700.0, 700.0])
    blocks = getattr(modeler, shape)(*args)
    _, x, y = modeler.draw([[1.0, blocks]])[0]

    for tag, edges in (('x', x), ('y', y)):
        values = np.real(np.asarray(edges)).reshape(-1)
        assert np.all(np.diff(values) > 0), (tag, np.min(np.diff(values)))

    # One row per slab, plus the closing period line and the one below the shape — no twins.
    assert np.asarray(y).shape[0] <= len(blocks) + 2, np.asarray(y).shape


def test_two_objects_sharing_an_edge_keep_their_gradient():
    """Two blocks meeting at a coordinate share one grid line — the shared line still moves.

    This is the case the old nudge was there for, so it is the one that has to be measured
    rather than argued. Both rectangles' facing edges are ``100 + p/2``, reached by different
    arithmetic (``cx + lx/2`` against ``(cx + p) - lx/2``), so they are two nodes in the graph
    that the grid builder collapses into one line. The derivative that survives has to be the
    real one.
    """
    torch = pytest.importorskip('torch')
    tensor = lambda v: torch.tensor(v, dtype=torch.float64)

    def total(p):
        ucell = [[1.0, [['rectangle', tensor(100.), tensor(150.), p, tensor(60.), 2.0,
                         tensor(0.)],
                        ['rectangle', tensor(100.) + p, tensor(150.), p, tensor(60.), 3.0,
                         tensor(0.)]]]]
        cfg = dict(period=PERIOD, thickness=[205.0], fto=[3, 3], n_top=1.0, n_bot=1.0,
                   wavelength=900.0, pol=0.5)
        return torch.sum(fieldwave.rcwa(backend=2, ucell=ucell, **cfg).conv_solve().de_ti)

    p = torch.tensor(80.0, dtype=torch.float64, requires_grad=True)
    total(p).backward()
    auto = float(p.grad)

    step = 1e-3
    with torch.no_grad():
        fd = float((total(tensor(80.0 + step)) - total(tensor(80.0 - step))) / (2 * step))

    assert abs(fd) > 1e-6, fd
    assert auto == pytest.approx(fd, rel=1e-4), (auto, fd)


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('name', LEGACY_PARITY)
def test_vector_solve_matches_legacy_golden(name, backend):
    """End to end: modeled geometry -> factorization -> efficiencies, on every backend."""
    if backend:
        pytest.importorskip('jax' if backend == 1 else 'torch')
    result = fieldwave.rcwa(backend=backend, ucell=INSTRUCTIONS[name], **SOLVE_KW).conv_solve()
    for field in ('de_ri', 'de_ti'):
        want = GOLDEN[f'{name}__{field}']
        got = np.asarray(getattr(result.res, field))
        tol = 1e-12   # not bit-exact: the vector path's inverses do not commute
                      # with the 2026-08 order-index flip to the last digit
        assert np.allclose(want, got, atol=tol, rtol=0), (field, np.max(np.abs(want - got)))


def test_rectangle_no_approximation_and_draw():
    """The exact-rectangle helper and ``draw`` (used directly, not via instructions)."""
    modeler = Modeling(period=[1000.0, 1000.0])
    element = modeler.rectangle_no_approximation(cx=500, cy=500, lx=200, ly=200, base=2.0)
    for i, (cell, x, y) in enumerate(modeler.draw([[1.0, element]])):
        for tag, got in (('cell', cell), ('x', x), ('y', y)):
            want = GOLDEN[f'draw__L{i}__{tag}']
            assert np.array_equal(want, np.asarray(got)), (i, tag)


def test_torch_device_code_is_not_a_device_index():
    """``device=0`` is fieldwave's code for CPU, not torch device index 0.

    Passing it through raw selects MPS on a Mac, which rejects float64 outright.
    """
    torch = pytest.importorskip('torch')
    modeler = Modeling(period=PERIOD, type_complex=torch.complex128, device=0,
                       xp=__import__('array_api_compat').array_namespace(torch.zeros(1)))
    info = modeler.modeling_vector_instruction(INSTRUCTIONS['rect_rot'])

    # Compared against the NumPy backend rather than the stored golden: the golden's rect_rot
    # rows are in the pre-fix order (see the reordering test above), and what this test is
    # actually about is that torch produced the same layer at all.
    want = Modeling(period=PERIOD).modeling_vector_instruction(INSTRUCTIONS['rect_rot'])
    assert np.array_equal(np.asarray(want[0][0]), np.asarray(info[0][0]))


def test_jax_device_setter_round_trips():
    """``solver.device = solver.device`` must be accepted on the jax backend.

    The getter returns resolved ``jax.Device`` objects, and several paths re-assign the
    property, so rejecting them breaks ``conv_solve_field``. The old check compared the
    repr of the implementation class, which stopped matching when jaxlib moved ``Device``
    from ``xla_extension`` to ``_jax``.
    """
    jax = pytest.importorskip('jax')

    solver = fieldwave.rcwa(
        backend=1, pol=0, n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, fto=[5],
        wavelength=900.0, period=[1000.0], thickness=[500.0],
        ucell=np.array([[[1.0, 3.0, 3.0, 1.0]]]), device=0)

    solver.device = solver.device                      # the round-trip that used to raise
    solver.device = jax.devices('cpu')[0]              # a bare device, not a list
    assert all(isinstance(d, jax.Device) for d in solver.device)

    with pytest.raises(ValueError):
        solver.device = 'not-a-device'

    # The caller-visible symptom was here: @jax_device_set re-assigns the property, so the
    # rejected round-trip surfaced as a bare ValueError out of conv_solve_field. The jitted
    # variant is still blocked by a separate same-material/tracer problem, so pin the
    # no-jit one, which goes through the same decorator.
    solver.conv_solve_field_no_jit(res_x=10, res_y=1, res_z=10)


@pytest.mark.parametrize('backend,name', [(1, 'jax'), (2, 'torch')])
def test_vector_path_differentiates_wrt_edge_positions(backend, name):
    """An OCD fit differentiates the *edges* — CD, sidewall angle, pitch — not the pixels.

    Those live in the segment edge list the vector modeler produces, and ``_cfs`` used to lift
    it with ``asarray(..., dtype=)``, which detaches under PyTorch. The solve still returned a
    tensor, just one with no graph behind it, so this failed only when someone called
    ``backward()``.

    Compared against a central difference rather than a stored number: the point is that the
    derivative is the real one, not that it is any particular value.
    """
    pytest.importorskip(name)
    if backend == 1:
        importlib.import_module('fieldwave.on_jax')

    cfg = dict(period=[300.0, 300.0], thickness=[205.0], fto=[3, 3],
               n_top=1.0, n_bot=1.0, wavelength=900.0, pol=0.5)

    def total(cd, xp_scalar):
        inst = [[3.5, [['rectangle', xp_scalar(150.0), xp_scalar(150.0), cd,
                        xp_scalar(60.0), 2.0, xp_scalar(0.0)]]]]
        res = fieldwave.rcwa(backend=backend, ucell=inst, **cfg).conv_solve()
        return res.de_ti

    if backend == 2:
        import torch
        T = lambda v: torch.tensor(v, dtype=torch.float64)
        cd = torch.tensor(80.0, dtype=torch.float64, requires_grad=True)
        torch.sum(total(cd, T)).backward()
        auto = float(cd.grad)
        with torch.no_grad():
            h = 1e-3
            fd = float((torch.sum(total(T(80.0 + h), T)) - torch.sum(total(T(80.0 - h), T))) / (2 * h))
    else:
        import jax, jax.numpy as jnp
        J = lambda v: jnp.asarray(v, dtype=jnp.float64)
        f = lambda cd: jnp.sum(total(cd, J)).real
        auto = float(jax.grad(f)(80.0))
        h = 1e-3
        fd = float((f(80.0 + h) - f(80.0 - h)) / (2 * h))

    assert auto != 0.0
    assert abs(auto - fd) / abs(fd) < 1e-4, (auto, fd)
