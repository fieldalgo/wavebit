"""Block geometry modeler (wavebit._core.modeler) against the frozen legacy output.

The modeler decomposes parametric shapes — rectangles and ellipses, optionally rotated — into
the piecewise-constant blocks the vector factorization consumes. Its reference values were
captured from the legacy per-backend implementation before that was deleted
(gen_goldens_modeler.py -> golden_legacy_modeler.npz).

The ellipse is the exception — see ``LEGACY_PARITY`` below and test_modeler_ellipse.py.
"""
import pathlib


import numpy as np
import pytest

import wavebit

#: The linear angle each pure declaration is the special case of.
_PSI_OF = {'te': np.pi / 2, 'tm': 0.0}
from wavebit._core.modeler import Modeling

GOLDEN = np.load(pathlib.Path(__file__).parent / 'data' / 'golden_legacy_modeler.npz')

PERIOD = [300.0, 300.0]
#: The three media as refractive INDEX -- the quantity the generator wrote and the quantity
#: the frozen ``*__cell`` arrays still hold. Only the names below carry it; everything handed
#: to the modeler uses the squares.
SI, SIO2, SI3N4 = 3.638751670074983, 1.4518, 2.0056
EPS_SI, EPS_SIO2, EPS_SI3N4 = SI ** 2, SIO2 ** 2, SI3N4 ** 2


def golden_cell(key):
    """One frozen ``cell`` array, as PERMITTIVITY.

    ``golden_legacy_modeler.npz`` stores eleven ``*__cell`` arrays that literally hold a
    refractive index, because the modeler used to emit one; it emits permittivity now. The
    square happens here, at the comparison, rather than in a regenerated file: three of the
    four generators in ``tests/data`` ``sys.exit()`` unconditionally and cannot be re-run, and
    the whole point of a legacy golden is that it is not re-derived from the code under test.

    Exact, not approximate: ``EPS_SI`` above is ``SI ** 2`` by the same float operation, so
    the comparisons stay ``array_equal`` rather than becoming tolerances.
    """
    return np.asarray(GOLDEN[key]) ** 2


# Must stay identical to gen_goldens_modeler.py -- as a *scene*. The generator names each
# material by its index; here every one of them is squared, which is the conversion the
# solver used to apply between the two.
INSTRUCTIONS = {
    'rect_1d': [[EPS_SI3N4, [['rectangle', 50, 150, 31, 300, EPS_SI, 0, 0, 0]]], [EPS_SI, []]],
    'rect_2d': [[EPS_SI3N4, [['rectangle', 80, 120, 60, 90, EPS_SI, 0, 0, 0],
                             ['rectangle', 200, 180, 50, 70, EPS_SIO2, 0, 0, 0]]],
                [EPS_SI, []]],
    'rect_rot': [[EPS_SI3N4, [['rectangle', 150, 150, 80, 60, EPS_SI, 0.4, 4, 4]]],
                 [EPS_SIO2, []]],
    'ellipse': [[EPS_SI3N4, [['ellipse', 150, 150, 90, 60, EPS_SI, 0.0, 4, 4]]],
                [EPS_SIO2, []]],
    'ellipse_rot': [[EPS_SI3N4, [['ellipse', 140, 160, 100, 50, EPS_SI, 0.5, 4, 4]]],
                    [EPS_SIO2, []]],
}
SOLVE_KW = dict(period=PERIOD, thickness=[205.0, 100.0], truncation_order=[3, 3],
                top=1.0, bot=1.0, wavelength=900.0, psi=np.pi / 4)

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
    """The layer's permittivity sampled on an ``n`` x ``n`` grid of points.

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
    info = Modeling(period=PERIOD).modeling_block_instruction(INSTRUCTIONS[name])
    assert len(info) == int(GOLDEN[f'{name}__n_layers'])
    for i, (cell, x, y) in enumerate(info):
        for tag, got in (('cell', cell), ('x', x), ('y', y)):
            key = f'{name}__L{i}__{tag}'
            want = golden_cell(key) if tag == 'cell' else GOLDEN[key]
            got = np.asarray(got)
            assert got.shape == want.shape, (i, tag, want.shape, got.shape)
            assert np.array_equal(want, got), (i, tag, np.max(np.abs(want - got)))


def test_rotated_rectangle_grid_keeps_the_legacy_layer_with_fewer_lines():
    """The grid drops its duplicate lines and comes back sorted, without moving any material.

    The stored golden has one grid line per block edge, including a near-duplicate wherever two
    blocks meet — the builder used to nudge the second edge off the first by a relative 1e-14
    rather than let them share a line, and inserted it at the slot found *before* the nudge, so
    the list was not even sorted (``rect_rot__L0__y`` has two inversions, and a negative
    segment width is what ``_fourier1d_integrated`` makes of one).

    The claim here is narrow and checkable: the layer is the same object, described with fewer
    lines. The material at 360000 points across both layers is identical, the edges that remain
    are a subset of the legacy ones to within the old nudge, and
    ``test_vector_solve_matches_legacy_golden[rect_rot]`` still holds the efficiencies to 1e-12.
    """
    info = Modeling(period=PERIOD).modeling_block_instruction(INSTRUCTIONS['rect_rot'])

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
                              material_at(golden_cell(f'rect_rot__L{i}__cell'),
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
    got = modeler.rectangle(150.0, 150.0, 80.0, 60.0, 4.0, angle, n_split, n_split)
    lx, ly = (60.0, 80.0) if swapped else (80.0, 60.0)
    want = modeler.rectangle_no_approximation(150.0, 150.0, lx, ly, 4.0)

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
    blocks = Modeling(period=PERIOD).rectangle(150.0, 150.0, 80.0, 60.0, 4.0, angle, 4, 4)
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
    cfg = dict(period=PERIOD, thickness=[205.0], truncation_order=[3, 3], top=1.0, bot=1.0,
               wavelength=900.0, psi=np.pi / 4)

    def T(angle):
        structure = [[12.25, [['rectangle', 150.0, 150.0, 80.0, 60.0, 4.0, angle, 8, 8]]]]
        return float(np.sum(np.asarray(
            wavebit.rcwa(backend=0, structure=structure, **cfg).solve().T)))

    across = (T(1.02e-5) - T(0.98e-5)) / (0.04e-5)
    outside = (T(1.10e-5) - T(1.06e-5)) / (0.04e-5)

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
    cfg = dict(period=PERIOD, thickness=[205.0], truncation_order=[3, 3], top=1.0, bot=1.0,
               wavelength=900.0, psi=np.pi / 4)

    def T(angle):
        structure = [[12.25, [['rectangle', tensor(150.0), tensor(150.0), tensor(80.0),
                           tensor(60.0), 4.0, angle, 8, 8]]]]
        return torch.sum(wavebit.rcwa(backend=2, structure=structure, **cfg).solve().T)

    p = torch.tensor(1e-5, dtype=torch.float64, requires_grad=True)
    out = T(p)
    assert out.requires_grad, 'angle left the graph — the margin is back'
    out.backward()
    auto = float(p.grad)

    step = 5e-6
    with torch.no_grad():
        fd = float((T(tensor(1e-5 + step)) - T(tensor(1e-5 - step))) / (2 * step))

    assert abs(fd) > 1.0, fd
    assert auto == pytest.approx(fd, rel=1e-3), (auto, fd)


@pytest.mark.parametrize('shape,args', [
    ('ellipse', (350., 350., 400., 250., 3.0, 0.0, 24, 24)),     # symmetric: edges coincide
    ('ellipse', (350., 350., 400., 250., 3.0, 0.0, 25, 25)),     # the other parity
    ('ellipse', (350., 350., 400., 250., 9.0, 0.4, 80, 80)),
    ('rectangle', (150., 150., 80., 60., 4.0, 0.4, 16, 16)),
    ('rectangle', (150., 150., 80., 60., 4.0, 0.4, 64, 64)),
])
def test_grid_lines_come_back_strictly_increasing(shape, args):
    """``edges_u`` / ``edges_v`` are segment edges; out of order they are a negative width.

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
        structure = [[1.0, [['rectangle', tensor(100.), tensor(150.), p, tensor(60.), 4.0,
                         tensor(0.)],
                        ['rectangle', tensor(100.) + p, tensor(150.), p, tensor(60.), 9.0,
                         tensor(0.)]]]]
        cfg = dict(period=PERIOD, thickness=[205.0], truncation_order=[3, 3], top=1.0, bot=1.0,
                   wavelength=900.0, psi=np.pi / 4)
        return torch.sum(wavebit.rcwa(backend=2, structure=structure, **cfg).solve().T)

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
    result = wavebit.rcwa(backend=backend, structure=INSTRUCTIONS[name], **SOLVE_KW).solve()
    # Golden keys keep the frozen legacy names (de_ri/de_ti); the attributes were renamed.
    for key, field in (('de_ri', 'R'), ('de_ti', 'T')):
        want = GOLDEN[f'{name}__{key}']
        got = np.asarray(getattr(result.given, field))
        tol = 1e-12   # not bit-exact: the block path's inverses do not commute
                      # with the 2026-08 order-index flip to the last digit
        assert np.allclose(want, got, atol=tol, rtol=0), (field, np.max(np.abs(want - got)))


def test_rectangle_no_approximation_and_draw():
    """The exact-rectangle helper and ``draw`` (used directly, not via instructions)."""
    modeler = Modeling(period=[1000.0, 1000.0])
    element = modeler.rectangle_no_approximation(cx=500, cy=500, lx=200, ly=200, material=4.0)
    for i, (cell, x, y) in enumerate(modeler.draw([[1.0, element]])):
        for tag, got in (('cell', cell), ('x', x), ('y', y)):
            key = f'draw__L{i}__{tag}'
            want = golden_cell(key) if tag == 'cell' else GOLDEN[key]
            assert np.array_equal(want, np.asarray(got)), (i, tag)


def test_torch_device_code_is_not_a_device_index():
    """``device=0`` is wavebit's code for CPU, not torch device index 0.

    Passing it through raw selects MPS on a Mac, which rejects float64 outright.
    """
    torch = pytest.importorskip('torch')
    modeler = Modeling(period=PERIOD, complex_dtype=torch.complex128, device=0,
                       xp=__import__('array_api_compat').array_namespace(torch.zeros(1)))
    info = modeler.modeling_block_instruction(INSTRUCTIONS['rect_rot'])

    # Compared against the NumPy backend rather than the stored golden: the golden's rect_rot
    # rows are in the pre-fix order (see the reordering test above), and what this test is
    # actually about is that torch produced the same layer at all.
    want = Modeling(period=PERIOD).modeling_block_instruction(INSTRUCTIONS['rect_rot'])
    assert np.array_equal(np.asarray(want[0][0]), np.asarray(info[0][0]))


def test_jax_device_setter_round_trips():
    """``solver.device = solver.device`` must be accepted on the jax backend.

    The getter returns resolved ``jax.Device`` objects, and several paths re-assign the
    property, so rejecting them breaks ``solve_field``. The old check compared the
    repr of the implementation class, which stopped matching when jaxlib moved ``Device``
    from ``xla_extension`` to ``_jax``.
    """
    jax = pytest.importorskip('jax')

    solver = wavebit.rcwa(
        backend=1, psi=np.pi / 2, top=1.0, bot=1.0, theta=0.0, phi=0.0, truncation_order=[5],
        wavelength=900.0, period=[1000.0], thickness=[500.0],
        structure=np.array([[[1.0, 3.0, 3.0, 1.0]]]), device=0, answers='field')

    solver.device = solver.device                      # the round-trip that used to raise
    solver.device = jax.devices('cpu')[0]              # a bare device, not a list
    assert all(isinstance(d, jax.Device) for d in solver.device)

    with pytest.raises(ValueError):
        solver.device = 'not-a-device'

    # The caller-visible symptom was here: @jax_device_set re-assigns the property, so the
    # rejected round-trip surfaced as a bare ValueError out of solve_field. The jitted
    # variant requires a frozen geometry passed up front by contract, so pin the no-jit one,
    # which goes through the same decorator.
    solver.solve_field_eager(sample_x=10, sample_y=1, sample_z=10)


@pytest.mark.parametrize('backend,name', [(1, 'jax'), (2, 'torch')])
def test_vector_path_differentiates_wrt_edge_positions(backend, name):
    """An OCD fit differentiates the *edges* — CD, sidewall angle, pitch — not the pixels.

    Those live in the segment edge list the block modeler produces, and ``_fourier1d_integrated`` used to lift
    it with ``asarray(..., dtype=)``, which through torch 2.11.x detached -- see
    :func:`wavebit._core.backend.as_array` for the version boundary. The solve still returned a
    tensor, just one with no graph behind it, so this failed only when someone called
    ``backward()``. torch 2.12.0 fixed that spelling, so on the current floor this no longer
    discriminates the original defect; it stands as a gradient check on the vector path, which
    the edge-position derivative is worth having on its own.

    Compared against a central difference rather than a stored number: the point is that the
    derivative is the real one, not that it is any particular value.
    """
    pytest.importorskip(name)
    if backend == 1:
        wavebit.enable_x64()

    cfg = dict(period=[300.0, 300.0], thickness=[205.0], truncation_order=[3, 3],
               top=1.0, bot=1.0, wavelength=900.0, psi=np.pi / 4)

    def total(cd, xp_scalar):
        inst = [[12.25, [['rectangle', xp_scalar(150.0), xp_scalar(150.0), cd,
                          xp_scalar(60.0), 4.0, xp_scalar(0.0)]]]]
        res = wavebit.rcwa(backend=backend, structure=inst, **cfg).solve()
        return res.T

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


@pytest.mark.parametrize('tetm,name', [('te', 'TE'), ('tm', 'TM')])
def test_the_staircase_costs_a_kink_at_an_aligned_angle(tetm, name):
    """What the rewrite did *not* fix, pinned so nobody reads the missing margin as a fix.

    A staircase and a smoothly tilted boundary do not share Fourier coefficients at a finite
    harmonic count, and the gap goes like ``|angle|``: the efficiency falls off its aligned
    value at the same rate on both sides, so the two one-sided slopes are equal and opposite
    instead of equal. Adding blocks does not help — the coefficient is flat in ``n_split`` —
    which is the whole reason an angle fit near normal belongs on the smooth path.

    Checked in pure polarizations. At ``psi=np.pi / 4`` the TE and TM amplitudes interfere and the
    cross term is genuinely odd in the angle, which hides this under a real first-order term;
    measuring there is what made it look absent.
    """
    cfg = dict(period=PERIOD, thickness=[205.0], truncation_order=[3, 3], top=1.0, bot=1.0,
               wavelength=900.0, psi=_PSI_OF[tetm], theta=0.0, phi=0.0)

    def T(angle, n_split):
        structure = [[12.25, [['rectangle', 150.0, 150.0, 80.0, 60.0, 4.0, angle,
                           n_split, n_split]]]]
        return float(np.sum(np.asarray(
            wavebit.rcwa(backend=0, structure=structure, **cfg).solve().T)))

    step = 1e-4
    at_zero = T(0.0, 64)
    right = (T(step, 64) - at_zero) / step
    left = (at_zero - T(-step, 64)) / step

    # Even in the angle, to numerical noise: the two sides are mirror images of one structure.
    assert T(step, 64) == pytest.approx(T(-step, 64), abs=1e-11), name
    # ... which is exactly why equal-and-opposite one-sided slopes mean a kink, not a slope.
    assert right == pytest.approx(-left, rel=1e-6), (left, right)
    assert abs(right) > 0.1, right

    # Flat in n_split: more blocks resolve the shape, not the factorization.
    coarse = (T(0.0, 16) - T(step, 16)) / step
    assert coarse == pytest.approx(-right, rel=0.02), (coarse, right)


def _rectangle_chord(y, cx, cy, lx, ly, angle):
    """[left, right] of the rotated rectangle at height ``y``, walked round its corners."""
    ca, sa = np.cos(angle), np.sin(angle)
    corners = [(ca * sx - sa * sy + cx, sa * sx + ca * sy + cy)
               for sx, sy in ((lx / 2, ly / 2), (lx / 2, -ly / 2),
                              (-lx / 2, -ly / 2), (-lx / 2, ly / 2))]
    xs = []
    for i in range(4):
        (x1, y1), (x2, y2) = corners[i], corners[(i + 1) % 4]
        if (y1 - y) * (y2 - y) <= 0 and y1 != y2:
            xs.append(x1 + (x2 - x1) * (y - y1) / (y2 - y1))
    return (min(xs), max(xs)) if len(xs) >= 2 else None


def test_random_rectangles_tile_exactly_and_sit_on_the_true_edges():
    """The fixed cases above, swept: 40 rectangles drawn from anywhere in the parameter space.

    Checked against the rotated rectangle's own corners, so nothing depends on how the
    decomposition is written, and the angles run past a full turn in both directions because
    the fold onto a quarter turn is the kind of thing a fixed list misses.
    """
    rng = np.random.default_rng(20260805)
    f = lambda v: float(np.real(np.asarray(v).reshape(-1)[0]))

    for _ in range(40):
        cx, cy = rng.uniform(80, 220, 2)
        lx, ly = rng.uniform(15, 110, 2)
        angle = float(rng.uniform(-2 * np.pi, 2 * np.pi))
        n = int(rng.integers(1, 20))
        case = (cx, cy, lx, ly, angle, n)

        raw = Modeling(period=PERIOD).rectangle(cx, cy, lx, ly, 2.0, angle, n, n)
        bs = [(f(b[0][0]), f(b[0][1]), f(b[1][0]), f(b[1][1])) for b in raw]

        for ylo, xlo, yhi, xhi in bs:
            assert yhi > ylo and xhi > xlo, case
        for (ylo, _, _, _), (_, _, yhi_next, _) in zip(bs[:-1], bs[1:]):
            assert ylo == pytest.approx(yhi_next, abs=1e-9 * max(1.0, abs(ylo))), case

        area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in bs)
        assert area == pytest.approx(lx * ly, rel=1e-12), case

        # Each block's sides are the true chord at some height inside its own slab.
        for ylo, xlo, yhi, xhi in bs:
            heights = np.linspace(ylo, yhi, 41)[1:-1]
            chords = [_rectangle_chord(y, cx, cy, lx, ly, angle) for y in heights]
            gap = min(abs(xlo - c[0]) + abs(xhi - c[1]) for c in chords if c is not None)
            assert gap < 1e-6, (case, ylo, xlo, yhi, xhi, gap)
