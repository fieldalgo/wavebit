"""The block modeler's ellipse decomposition, checked for *convergence* rather than parity.

An ellipse has no exact block decomposition — it is approximated by a stack of axis-aligned
blocks — so the only meaningful contract is that refining ``n_split`` makes the approximation
better, and better in a way that settles on the real ellipse. A stored number cannot say that:
the decomposition this replaced reproduced its own stored numbers perfectly while its answer
swung between 0.24 and 0.68 as the contour was refined, because refining changed the structure
instead of resolving it.

So everything here is measured against something the modeler does not compute: the ellipse's
implicit equation, its analytic area ``pi*lx*ly/4``, a raster of the same ellipse, and the
same ellipse solved through the raster Fourier path.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.modeler import Modeling

PERIOD = [300.0, 300.0]
CX, CY = 150.0, 150.0

# lx, ly, angle. 0.0 and pi/2 are the alignments a sampled contour lands its points exactly on;
# the rest are angles nothing lines up with.
SHAPES = [
    (90.0, 60.0, 0.0),
    (90.0, 60.0, 0.4),
    (100.0, 50.0, 1.234),
    (100.0, 50.0, np.pi / 2),
    (70.0, 70.0, 0.7),          # a circle: lx == ly
    (90.0, 60.0, -0.6),
]


def _f(v):
    """One block coordinate as a Python float (they are length-1 arrays at the complex type)."""
    return float(np.real(np.asarray(v).reshape(-1)[0]))


def blocks(lx, ly, angle, n_split, cx=CX, cy=CY):
    """``[(y_lo, x_lo, y_hi, x_hi), ...]``, top block first."""
    raw = Modeling(period=PERIOD).ellipse(cx, cy, lx, ly, 4.0, angle, n_split, n_split)
    return [(_f(b[0][0]), _f(b[0][1]), _f(b[1][0]), _f(b[1][1])) for b in raw]


def implicit(x, y, lx, ly, angle, cx=CX, cy=CY):
    """The rotated ellipse's implicit form: ``== 1`` exactly on the boundary."""
    dx, dy = x - cx, y - cy
    ca, sa = np.cos(angle), np.sin(angle)
    u, v = ca * dx + sa * dy, -sa * dx + ca * dy
    return (u / (lx / 2)) ** 2 + (v / (ly / 2)) ** 2


def area(bs):
    return sum((yhi - ylo) * (xhi - xlo) for ylo, xlo, yhi, xhi in bs)


def heights_on_boundary(x, lx, ly, angle, cx=CX, cy=CY):
    """The heights at which the ellipse's boundary passes through the vertical line ``x``.

    Solves ``implicit(x, y) == 1`` for ``y`` — a quadratic — so it is derived from the implicit
    form alone and knows nothing about how the decomposition is parameterized.
    """
    dx = x - cx
    ca, sa = np.cos(angle), np.sin(angle)
    a, b = lx / 2, ly / 2
    p = (sa / a) ** 2 + (ca / b) ** 2
    q = 2 * dx * ca * sa * (1 / a ** 2 - 1 / b ** 2)
    r = (ca * dx / a) ** 2 + (sa * dx / b) ** 2 - 1
    disc = q * q - 4 * p * r
    if disc < 0:
        return []
    root = np.sqrt(disc)
    return [cy + (-q - root) / (2 * p), cy + (-q + root) / (2 * p)]


# ------------------------------------------------------------------------------------ geometry

@pytest.mark.parametrize('lx,ly,angle', SHAPES)
@pytest.mark.parametrize('n_split', [2, 3, 5, 8, 17, 40])
def test_each_block_is_a_true_chord_of_the_ellipse(lx, ly, angle, n_split):
    """A block's left and right edges are the two ends of one horizontal chord.

    Found by asking, from the implicit equation alone, at what height the boundary crosses the
    block's left edge — and then requiring the right edge to be on the boundary at that same
    height, somewhere inside the block. That shares nothing with the parametric form the
    decomposition is built from, so a sign or axis slip in one is not echoed by the other. It
    is also what fails for an edge-sampled decomposition, where the two sides of a block come
    from different heights.
    """
    bs = blocks(lx, ly, angle, n_split)
    assert len(bs) == n_split

    for ylo, xlo, yhi, xhi in bs:
        heights = [y for y in heights_on_boundary(xlo, lx, ly, angle) if ylo <= y <= yhi]
        assert heights, (ylo, xlo, yhi, xhi)
        assert any(implicit(xhi, y, lx, ly, angle) == pytest.approx(1.0, abs=1e-12)
                   for y in heights), (ylo, xlo, yhi, xhi, heights)


@pytest.mark.parametrize('lx,ly,angle', SHAPES)
@pytest.mark.parametrize('n_split', [2, 3, 5, 8, 17, 40])
def test_blocks_tile_the_bounding_box_without_gaps_or_overlap(lx, ly, angle, n_split):
    """Blocks stack downwards, each one starting exactly where the last ended."""
    bs = blocks(lx, ly, angle, n_split)

    for (ylo, _, _, _), (_, _, yhi_next, _) in zip(bs[:-1], bs[1:]):
        assert ylo == yhi_next                       # exact: they are the same expression

    for ylo, xlo, yhi, xhi in bs:
        assert yhi > ylo
        assert xhi > xlo

    # The stack spans the rotated ellipse's exact y extent.
    half_height = np.hypot(ly / 2 * np.cos(angle), lx / 2 * np.sin(angle))
    assert bs[0][2] == pytest.approx(CY + half_height, abs=1e-10)
    assert bs[-1][0] == pytest.approx(CY - half_height, abs=1e-10)


@pytest.mark.parametrize('lx,ly,angle', SHAPES)
def test_area_converges_to_the_analytic_area(lx, ly, angle):
    """``pi*lx*ly/4``, approached monotonically and at second order.

    Second order is the point of taking the chord at the middle of each slab rather than at one
    of its edges; an edge-sampled decomposition is only first order, and the one this replaced
    was not even that — its error changed sign instead of shrinking.
    """
    exact = np.pi * lx * ly / 4
    errors = [abs(area(blocks(lx, ly, angle, n)) - exact) for n in (8, 16, 32, 64, 128)]

    assert errors == sorted(errors, reverse=True), errors
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert coarse / fine > 3.5, errors          # 4x is the second-order rate


@pytest.mark.parametrize('lx,ly,angle', SHAPES)
def test_symmetric_difference_with_a_raster_of_the_same_ellipse_shrinks(lx, ly, angle):
    """Right area, right place: the mismatch against a fine raster falls off monotonically.

    Area alone would pass a decomposition that put the right amount of material in the wrong
    place, which is exactly the failure mode being guarded against.
    """
    n_pix = 900
    axis = (np.arange(n_pix) + 0.5) / n_pix * PERIOD[0]
    X, Y = np.meshgrid(axis, axis)
    inside = implicit(X, Y, lx, ly, angle) <= 1.0
    pixel = (PERIOD[0] / n_pix) ** 2

    mismatch = []
    for n in (8, 16, 32, 64):
        covered = np.zeros_like(inside)
        for ylo, xlo, yhi, xhi in blocks(lx, ly, angle, n):
            covered |= (Y >= ylo) & (Y < yhi) & (X >= xlo) & (X < xhi)
        mismatch.append(float((covered ^ inside).sum()) * pixel)

    assert mismatch == sorted(mismatch, reverse=True), mismatch
    assert mismatch[-1] / (np.pi * lx * ly / 4) < 0.01


@pytest.mark.parametrize('n_split', [5, 9, 13, 17, 21])
def test_an_unrotated_ellipse_survives_a_sampling_that_hits_its_vertex(n_split):
    """These splits used to delete the ellipse outright.

    The decomposition this replaced walked a sampled contour down both sides from the topmost
    point and decided which side led by comparing that point's two neighbours. An unrotated
    ellipse sampled with ``n_split % 4 == 1`` puts a sample exactly on the top vertex, its two
    neighbours come out level, neither side leads, and the interleave produced an empty list —
    every block collapsed to a zero-size box at the origin and the shape silently vanished
    from the layer.
    """
    bs = blocks(90.0, 60.0, 0.0, n_split)

    assert area(bs) == pytest.approx(np.pi * 90.0 * 60.0 / 4, rel=0.25)
    for ylo, xlo, yhi, xhi in bs:
        assert (yhi - ylo) > 0 and (xhi - xlo) > 0
        assert abs((ylo + yhi) / 2 - CY) <= 30.0     # inside the layer, not at the origin


def test_split_counts_below_two_are_rejected():
    with pytest.raises(ValueError, match='at least 2 slabs'):
        Modeling(period=PERIOD).ellipse(CX, CY, 90.0, 60.0, 4.0, 0.4, 1, 1)


# ------------------------------------------------------------------------------- through solve

SOLVE_KW = dict(period=[700.0, 700.0], thickness=[400.0], fto=[4, 4],
                top=1.0, bot=1.0, wavelength=500.0, psi=np.pi / 4)
#: One ellipse, described once and reached through four paths. ``material`` / ``background``
#: are permittivities: 9.0 is the index 3.0 this rod used to be written with.
ELLIPSE = dict(cx=350.0, cy=350.0, lx=400.0, ly=250.0, angle=0.5, background=1.0,
               material=9.0)


def solve_vector(n_split, backend=0):
    e = ELLIPSE
    structure = [[e['background'], [['ellipse', e['cx'], e['cy'], e['lx'], e['ly'], e['material'],
                             e['angle'], n_split, n_split]]]]
    T = wavebit.rcwa(backend=backend, structure=structure, **SOLVE_KW).solve().T
    return float(np.sum(np.asarray(T)))


def test_efficiency_converges_monotonically_in_n_split():
    """The regression the stored-number tests could not express.

    Asserted as a *shape* of the sequence — successive changes shrinking at the second-order
    rate — not as a value, so it keeps meaning if the reference number ever legitimately moves.
    Before the fix this sequence read 0.30, 0.24, 0.24, 0.68, 0.32, 0.55 on a similar case.
    """
    values = [solve_vector(n) for n in (20, 40, 80, 160, 320)]
    steps = [abs(b - a) for a, b in zip(values[:-1], values[1:])]

    assert steps == sorted(steps, reverse=True), values
    for coarse, fine in zip(steps[:-1], steps[1:]):
        assert coarse / fine > 3.0, values
    # Monotone approach, not a sequence that crosses the limit and comes back.
    assert len(set(np.sign(np.diff(values)))) == 1, values


def test_converged_efficiency_matches_a_raster_of_the_same_ellipse():
    """The block path's limit is the same structure the raster Fourier path sees."""
    n_pix = 512
    axis = (np.arange(n_pix) + 0.5) / n_pix * SOLVE_KW['period'][0]
    X, Y = np.meshgrid(axis, axis)
    inside = implicit(X, Y, ELLIPSE['lx'], ELLIPSE['ly'], ELLIPSE['angle'],
                      ELLIPSE['cx'], ELLIPSE['cy']) <= 1.0
    structure = np.where(inside, ELLIPSE['material'], ELLIPSE['background'])[None]

    raster = float(np.sum(wavebit.rcwa(backend=0, structure=structure,
                                         fourier_coefficients='sampled', factorization='li',
                                         **SOLVE_KW).solve().T))

    assert solve_vector(320) == pytest.approx(raster, abs=2e-4)


def test_converged_efficiency_matches_the_analytic_shape_path():
    """The same ellipse built as a parametric shape instead of as blocks.

    A looser check than the raster one, and deliberately so: the smooth path factorizes with
    NVF while the block path uses Li's rules on the block edges, so the two agree only once
    the harmonic count is high enough for both — 0.3% at ``fto`` 8, 0.1% at 10. What it does
    pin tightly is the *rotation convention*: feeding the smooth path the opposite sign of
    ``angle`` misses by an order of magnitude more, so a flipped rotation cannot hide inside
    the factorization tolerance.
    """
    kw = dict(SOLVE_KW, fto=[8, 8])
    e = ELLIPSE

    def analytic(angle):
        structure = [dict(background=e['background'],
                      shapes=[dict(type='ellipse', cx=e['cx'], cy=e['cy'], lx=e['lx'],
                                   ly=e['ly'], angle=angle, material=e['material'])])]
        return float(np.sum(wavebit.rcwa(backend=0, structure=structure, **kw).solve().T))

    structure = [[e['background'], [['ellipse', e['cx'], e['cy'], e['lx'], e['ly'], e['material'],
                             e['angle'], 240, 240]]]]
    vector = float(np.sum(wavebit.rcwa(backend=0, structure=structure, **kw).solve().T))

    assert vector == pytest.approx(analytic(e['angle']), rel=0.01)
    assert abs(vector - analytic(-e['angle'])) > 10 * abs(vector - analytic(e['angle']))


@pytest.mark.parametrize('backend', [1, 2], ids=['jax', 'torch'])
def test_the_other_backends_decompose_the_ellipse_the_same_way(backend):
    """No stored ellipse golden survives the fix, so the backends are pinned to each other."""
    pytest.importorskip('jax' if backend == 1 else 'torch')
    assert solve_vector(80, backend=backend) == pytest.approx(solve_vector(80), abs=1e-9)


def _grad_and_difference(param, value, step, angle, n_split):
    """Autograd derivative of ``sum(T)`` and the central difference beside it (torch)."""
    import torch

    def total(overrides):
        args = dict(cx=350.0, cy=350.0, lx=400.0, ly=250.0, angle=angle)
        args.update(overrides)
        args = {k: v if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float64)
                for k, v in args.items()}
        structure = [[ELLIPSE['background'],
                  [['ellipse', args['cx'], args['cy'], args['lx'], args['ly'],
                    ELLIPSE['material'], args['angle'], n_split, n_split]]]]
        kw = dict(SOLVE_KW, fto=[3, 3])
        return torch.sum(wavebit.rcwa(backend=2, structure=structure, **kw).solve().T)

    p = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    total({param: p}).backward()

    with torch.no_grad():
        fd = float((total({param: value + step}) - total({param: value - step})) / (2 * step))
    return float(p.grad), fd


@pytest.mark.parametrize('n_split', [24, 25])
@pytest.mark.parametrize('param,value,step', [('lx', 400.0, 1e-2), ('ly', 250.0, 1e-2)])
def test_gradient_survives_the_coincident_edges_of_an_unrotated_ellipse(param, value, step,
                                                                       n_split):
    """An unrotated ellipse puts two slabs' side edges at exactly the same x, by symmetry.

    ``x_half`` is ``a*sin(s_mid)`` and ``sin`` is symmetric about pi/2, so slab ``i`` and slab
    ``n - 1 - i`` produce bit-identical left and right edges — 10 coincident pairs at
    ``n_split`` 24, which ``block_per_layer_numeric`` collapses onto one grid line each. The
    two carry the same derivative, so the collapse is free; this asserts the outcome that
    matters, autograd still agreeing with a difference quotient, at both parities of
    ``n_split``, since the odd one also has a slab centred on pi/2 with no twin.

    ``test_two_objects_sharing_an_edge_keep_their_gradient`` in test_modeler_core.py is the
    same question asked where the two edges belong to different objects and reach the same
    coordinate by different arithmetic.
    """
    pytest.importorskip('torch')
    auto, fd = _grad_and_difference(param, value, step, angle=0.0, n_split=n_split)

    assert abs(fd) > 1e-6, fd
    assert auto == pytest.approx(fd, rel=1e-4), (auto, fd)


@pytest.mark.parametrize('param,value,step', [('lx', 400.0, 1e-2),
                                              ('ly', 250.0, 1e-2),
                                              ('angle', 0.5, 1e-5)])
def test_the_blocks_still_carry_gradient_to_the_shape_parameters(param, value, step):
    """An inverse-design fit differentiates the ellipse's axes and its rotation.

    ``angle`` is included because it is the one the rectangle cannot offer: that path picks a
    rotation octant by branching, and flattens the derivative inside ``angle_margin``. The
    chord form has no branch, so the ellipse's angle derivative is real at every value.
    """
    pytest.importorskip('torch')
    auto, fd = _grad_and_difference(param, value, step, angle=0.5, n_split=24)

    assert abs(fd) > 1e-6, fd                        # the case has to be sensitive to be a test
    assert auto == pytest.approx(fd, rel=1e-4), (auto, fd)


# ------------------------------------------------------------------------------- property sweep

def test_random_ellipses_are_chords_that_tile_and_converge():
    """The fixed cases above, swept: 40 ellipses drawn from anywhere in the parameter space.

    Every invariant is checked against the ellipse's own equation, so nothing here depends on
    how the decomposition is written. The angles run past a full turn in both directions
    because the fold onto a quarter turn is exactly the kind of thing a fixed list misses.
    """
    rng = np.random.default_rng(20260805)

    for _ in range(40):
        cx, cy = rng.uniform(80, 220, 2)
        lx, ly = rng.uniform(15, 110, 2)
        angle = float(rng.uniform(-2 * np.pi, 2 * np.pi))
        n = int(rng.integers(2, 40))
        case = (cx, cy, lx, ly, angle, n)

        bs = blocks(lx, ly, angle, n, cx=cx, cy=cy)
        assert len(bs) == n, case

        for ylo, xlo, yhi, xhi in bs:
            assert yhi > ylo and xhi > xlo, case
            inside = [h for h in heights_on_boundary(xlo, lx, ly, angle, cx, cy)
                      if ylo - 1e-9 <= h <= yhi + 1e-9]
            assert inside, case
            assert any(implicit(xhi, h, lx, ly, angle, cx, cy) == pytest.approx(1.0, abs=1e-9)
                       for h in inside), case

        for (ylo, _, _, _), (_, _, yhi_next, _) in zip(bs[:-1], bs[1:]):
            assert ylo == pytest.approx(yhi_next, abs=1e-9 * max(1.0, abs(ylo))), case

        exact = np.pi * lx * ly / 4
        errs = [abs(sum((b[2] - b[0]) * (b[3] - b[1])
                        for b in blocks(lx, ly, angle, m, cx=cx, cy=cy)) - exact)
                for m in (16, 32, 64)]
        assert errs[0] > errs[1] > errs[2], (case, errs)
        for coarse, fine in zip(errs[:-1], errs[1:]):
            assert 3.5 < coarse / fine < 4.5, (case, errs)
