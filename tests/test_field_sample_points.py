"""Sampling the field where the caller asks, rather than only on a uniform grid.

``reconstruct_field`` used to take three *counts* and lay out ``linspace`` grids from them. It now
also takes the coordinates themselves. The claim this file defends is not that some new
approximation is accurate — it is that **there is no approximation**: the reconstruction was
already pointwise in x, y and z, so naming the points changes which ones come back and nothing
else.

That is worth testing rather than asserting, because "pointwise" is exactly the kind of property
that holds until someone adds an FFT for speed. The strongest statement available is
*bit-identity*, and it is what most of these tests use: a point asked for explicitly must equal
the same point asked for as part of a grid, to the bit, not to a tolerance.

``sample_x`` / ``sample_y`` are measured from the **centre of the cell**, the same origin an
parametric shape's ``cx`` uses since 2026-08-31, so a count spans ``[-period/2, +period/2]``.
:func:`test_the_shape_origin_and_the_sample_origin_are_one_origin` is the one that says the two
halves of that sentence are the same origin, against a raster, which has no origin to move.

The one place the two spellings legitimately differ is a layer interface, where ``E_z`` is
two-valued. A count returns both limits (the interface appears twice, as one layer's top and
the next one's bottom); explicit coordinates can only return one, and the documented rule picks the
layer above. :func:`test_an_interface_returns_the_upper_layers_limit` pins that, including the
nudge that recovers the other side.
"""
import numpy as np
import pytest

import wavebit

WL = 620.0
PERIOD = (400.0, 380.0)
NX = 20

BACKENDS = [0, 1, 2]
BACKEND_IDS = ['numpy', 'jax', 'torch']

# Which reconstruction each name exercises. All five paths, because the sampling helpers are
# shared and a helper used by four of five is a helper that will break the fifth.
PATHS = ('te', 'tm', 'conical', '2d', '4ff')


def _bar(eps_bg, eps_hi):
    row = np.full(NX, eps_bg, dtype=complex)
    row[NX // 4:NX // 2] = eps_hi
    return row


# Permittivities: the squares of the indices 1.0 / 2.0 and 1.3 / 1.6 these bars used to hold.
TWO_LAYER = np.stack([_bar(1.0, 4.0),
                      _bar(1.6900000000000002, 2.5600000000000005)])[:, None, :]
THICK = [150.0, 260.0]
TOTAL = float(sum(THICK))


def _solver(path, backend=0):
    kw = dict(backend=backend, top=1.0, bot=1.0, wavelength=WL, thickness=THICK,
              theta=np.deg2rad(24.0), fourier_coefficients='integrated', answers='field')
    if path in ('te', 'tm'):
        kw.update(tetm=path, phi=None, truncation_order=6, period=[PERIOD[0]], structure=TWO_LAYER)
    elif path == 'conical':
        kw.update(psi=np.pi / 2, phi=np.deg2rad(15.0), truncation_order=6, period=[PERIOD[0]],
                  structure=TWO_LAYER)
    else:
        kw.update(psi=np.pi / 2, phi=np.deg2rad(15.0), truncation_order=(3, 3), period=PERIOD,
                  structure=np.repeat(TWO_LAYER, 4, axis=1))
    s = wavebit.rcwa(**kw)
    if path == '4ff':
        s.formulation = '4ff'
    s.solve()
    return s


def _field(path, backend=0, **kw):
    return np.asarray(_solver(path, backend).reconstruct_field(**kw))


def _zaxis(cell):
    """The z axis, which the coupled paths push one place along behind the incidence axis."""
    return 0 if cell.ndim == 4 else 1


@pytest.mark.parametrize('path', PATHS)
def test_explicit_points_reproduce_the_grid_they_were_taken_from(path):
    """Ask for the grid's own coordinates and get the grid back, to the bit.

    This is the whole claim in one assertion. ``sample_z`` is left as a count here so that the
    comparison is against an unchanged z axis and only x is under test; z gets its own tests
    below, where the interface rule has to be accounted for.
    """
    grid = _field(path, sample_x=11, sample_y=5, sample_z=4)
    named = _field(path, sample_x=np.linspace(-PERIOD[0] / 2, PERIOD[0] / 2, 11),
                   sample_y=(np.linspace(-PERIOD[1] / 2, PERIOD[1] / 2, 5)
                             if path in ('2d', '4ff') else 5),
                   sample_z=4)
    assert named.shape == grid.shape
    assert np.array_equal(named, grid), np.max(np.abs(named - grid))


def test_the_shape_origin_and_the_sample_origin_are_one_origin():
    """``cx`` and ``sample_x`` are measured from the same point, and a raster says which.

    Both moved to the centre of the cell on 2026-08-31, and they had to move together: a shape
    stated in one frame and sampled in the other returns the field half a period from the
    structure it belongs to. Nothing inside either path could notice -- a translation of the
    whole layer is a diagonal phase on every convolution matrix, so ``R`` and ``T`` come back
    identical either way and the cell that arrives is complete, smooth and wrong.

    **The raster is the referee**, because it did not move: an array *is* the geometry, so the
    same grating written as pixels fixes both origins from outside the shape path. The rod's
    edges land on pixel boundaries, so the two structures are one permittivity function and
    what is left between them is the factorization -- measured 2.8e-3 of the field's own scale.
    The control is the half-period alternative at 8.6e-1, a factor of 300, which is what says
    the tolerance is not the thing carrying the argument.
    """
    P, N, EPS = 400.0, 40, 12.25
    x = (np.arange(N) + 0.5) / N * P
    raster = np.where((x >= 100.0) & (x < 200.0), EPS, 1.0)[None, None, :]
    # the same rod: absolute x in [100, 200), which is 50 left of the centre of the cell
    shapes = [dict(background=1.0, shapes=[dict(type='rectangle', cx=-50.0, cy=0.0,
                                                lx=100.0, ly=P, material=EPS)])]
    kw = dict(backend=0, top=1.0, bot=1.0, wavelength=900.0, thickness=[220.0],
              theta=np.deg2rad(18.0), phi=0.0, psi=np.pi / 2, truncation_order=[8, 0],
              period=[P], answers='field')

    def field(structure, sample_x, **over):
        s = wavebit.rcwa(structure=structure, **dict(kw, **over))
        s.solve()
        return np.asarray(s.reconstruct_field(sample_x=sample_x, sample_y=1, sample_z=6))

    xs = np.linspace(-P / 2, P / 2, 25, endpoint=False)
    ref = field(raster, xs)
    scale = np.abs(ref).max()
    same = np.abs(ref - field(shapes, xs, factorization='nvf')).max() / scale
    half = np.abs(ref - field(shapes, xs + P / 2, factorization='nvf')).max() / scale
    assert same < 1e-2, same
    assert half > 30 * same, (same, half)


@pytest.mark.parametrize('path', PATHS)
def test_a_subset_costs_nothing_in_accuracy(path):
    """Three scattered x, asked for alone, equal those three inside a full grid — bitwise.

    The point of the feature: sample a line cut or a monitor plane without paying for a grid,
    and without the answer depending on what else was requested alongside it.
    """
    fine = np.linspace(-PERIOD[0] / 2, PERIOD[0] / 2, 41)
    pick = [3, 17, 38]
    full = _field(path, sample_x=fine, sample_y=1, sample_z=3)
    few = _field(path, sample_x=fine[pick], sample_y=1, sample_z=3)
    assert np.array_equal(few, np.take(full, pick, axis=-2))


@pytest.mark.parametrize('path', PATHS)
def test_the_x_points_need_not_be_sorted_or_uniform_or_inside_the_cell(path):
    """No grid structure is assumed anywhere, so none of these are special cases.

    A coordinate outside the cell -- which is ``[-period/2, +period/2)``, x being measured
    from the cell's centre -- is a periodic continuation rather than an error: the kernel is
    ``exp(i k.r)`` and the Bloch phase carries it. Checked against the in-cell point it
    repeats, which is a statement about the physics rather than about the indexing.
    """
    xs = np.array([110.0, 0.0, -102.7, -159.0])
    got = _field(path, sample_x=xs, sample_y=1, sample_z=3)
    shifted = _field(path, sample_x=xs + PERIOD[0], sample_y=1, sample_z=3)

    # unsorted / non-uniform: each entry equals the same coordinate asked for on its own
    for j, x in enumerate(xs):
        one = _field(path, sample_x=np.array([x]), sample_y=1, sample_z=3)
        assert np.array_equal(np.take(got, [j], axis=-2), one)

    # one period over: the Bloch phase, not equality
    kx0 = 1.0 * np.sin(np.deg2rad(24.0)) * (2 * np.pi / WL)
    if path in ('2d', '4ff', 'conical'):
        kx0 *= np.cos(np.deg2rad(15.0))
    assert np.allclose(shifted, got * np.exp(1j * kx0 * PERIOD[0]), rtol=0, atol=1e-9)


@pytest.mark.parametrize('path', PATHS)
def test_explicit_z_reproduces_the_grid_away_from_interfaces(path):
    """Every z the count spelling visits, except the doubled interface, comes back identical.

    A count puts ``sample_z`` samples in each layer inclusive of both faces, so the internal
    interface appears twice. Those two rows are the only ones explicit coordinates cannot
    reproduce together, and they are excluded here and tested on their own next.
    """
    nz = 5
    grid = _field(path, sample_x=7, sample_y=1, sample_z=nz)
    per = [np.linspace(0, d, nz) for d in THICK]
    z_abs = np.concatenate([per[0], per[1] + THICK[0]])

    named = _field(path, sample_x=7, sample_y=1, sample_z=z_abs)
    assert named.shape == grid.shape

    ax = _zaxis(grid)
    keep = [i for i in range(len(z_abs)) if i != nz - 1]      # drop layer 0's top face
    assert np.array_equal(np.take(named, keep, axis=ax), np.take(grid, keep, axis=ax))


@pytest.mark.parametrize('path', PATHS)
def test_an_interface_returns_the_upper_layers_limit(path):
    """The documented rule, and the nudge that gets the other side.

    Half-open ``[bottom, top)`` means the interface belongs to the layer above; ``z - eps``
    recovers the lower layer's limit.

    **Whether the two limits differ at all depends on the path, and that is physics rather
    than bookkeeping.** Only ``E_z`` is discontinuous across a *horizontal* interface. TE's
    layout is ``(Ey, Hx, Hz)`` — three tangential components, all continuous — so for it the
    rule is unobservable and both spellings must agree exactly. Every other path carries an
    ``E_z`` and must show the jump. Asserting the split rather than skipping TE keeps this
    honest: if TE ever *did* differ at an interface, something is wrong with it.
    """
    nz = 5
    grid = _field(path, sample_x=7, sample_y=1, sample_z=nz)
    ax = _zaxis(grid)
    lower = np.take(grid, [nz - 1], axis=ax)      # layer 0's top face
    upper = np.take(grid, [nz], axis=ax)          # layer 1's bottom face

    at = _field(path, sample_x=7, sample_y=1, sample_z=np.array([THICK[0]]))
    assert np.array_equal(at, upper)

    just_below = _field(path, sample_x=7, sample_y=1, sample_z=np.array([THICK[0] - 1e-9]))
    assert np.allclose(just_below, lower, rtol=0, atol=1e-6)

    jump = np.max(np.abs(upper - lower)) / np.max(np.abs(grid))
    if path == 'te':
        assert jump < 1e-14, f'TE has no discontinuous component, yet the interface jumps {jump:.2e}'
    else:
        # or the rule under test would be unobservable and this would pass on a bug
        assert jump > 1e-3, f'no E_z jump to detect ({jump:.2e}) — the case is not exercising the rule'


@pytest.mark.parametrize('path', PATHS)
def test_the_caller_s_z_order_is_preserved(path):
    """Points come back in the order asked for, not grouped by the layer they landed in.

    The reconstruction has to bucket by layer to propagate, so the natural output order is
    layer-major. Anything a caller can hand to a plotting routine must be in *their* order.
    """
    z = np.array([0.0, 300.0, 75.0, TOTAL, 150.0, 20.0])
    perm = np.array([4, 0, 5, 2, 3, 1])
    a = _field(path, sample_x=5, sample_y=1, sample_z=z)
    b = _field(path, sample_x=5, sample_y=1, sample_z=z[perm])
    ax = _zaxis(a)
    assert np.array_equal(np.take(a, perm, axis=ax), b)


@pytest.mark.parametrize('path', PATHS)
def test_a_repeated_z_is_returned_twice(path):
    """No implicit de-duplication — a request of length n returns n samples."""
    z = np.array([80.0, 80.0, 300.0])
    got = _field(path, sample_x=5, sample_y=1, sample_z=z)
    ax = _zaxis(got)
    assert got.shape[ax] == 3
    assert np.array_equal(np.take(got, [0], axis=ax), np.take(got, [1], axis=ax))


@pytest.mark.parametrize('path', PATHS)
def test_the_stack_s_own_faces_are_reachable(path):
    """z = 0 and z = total, the closed ends, must not fall off either rule."""
    got = _field(path, sample_x=5, sample_y=1, sample_z=np.array([0.0, TOTAL]))
    assert np.all(np.isfinite(np.asarray(got)))
    grid = _field(path, sample_x=5, sample_y=1, sample_z=3)
    ax = _zaxis(grid)
    assert np.array_equal(np.take(got, [0], axis=ax), np.take(grid, [0], axis=ax))
    assert np.array_equal(np.take(got, [1], axis=ax), np.take(grid, [-1], axis=ax))


@pytest.mark.parametrize('z', [-1e-3, 410.001, 1e4])
def test_z_outside_the_stack_is_refused(z):
    """Not extrapolated: the half-spaces carry mode sets this state does not hold."""
    with pytest.raises(ValueError, match='inside the stack'):
        _field('tm', sample_x=5, sample_y=1, sample_z=np.array([z]))


def test_a_float_count_is_not_silently_a_coordinate():
    """``sample_z=5.0`` is a mistake, and it should look like one rather than mean ``[5.0]``."""
    got = _field('tm', sample_x=5, sample_y=1, sample_z=np.array([5.0]))
    ax = _zaxis(got)
    assert got.shape[ax] == 1                      # an array of one z is one sample

    # a bare float is neither, and reading it as either would be a silent wrong answer.
    # `sample_y` is checked on a 2-D path: the pure 1-D solves overwrite it with 1, because the
    # structure is y-invariant and one row is the whole answer, so nothing there can see it.
    for path, kw in (('tm', dict(sample_z=5.0)), ('tm', dict(sample_x=5.0)),
                     ('2d', dict(sample_z=5.0)), ('2d', dict(sample_x=5.0)),
                     ('2d', dict(sample_y=5.0))):
        call = dict(sample_x=5, sample_y=2, sample_z=3)
        call.update(kw)
        with pytest.raises(TypeError, match='integer count or a 1-D array'):
            _field(path, **call)


@pytest.mark.parametrize('backend,bid', list(zip(BACKENDS, BACKEND_IDS)), ids=BACKEND_IDS)
def test_every_backend_samples_the_same_points(backend, bid):
    """The helpers are in ``_core`` and shared, so all three must agree — and with the grid."""
    z = np.array([0.0, 60.0, 150.0, 275.0, TOTAL])
    x = np.array([0.0, 133.3, 251.7, 400.0])
    ref = _field('tm', backend=0, sample_x=x, sample_y=1, sample_z=z)
    got = _field('tm', backend=backend, sample_x=x, sample_y=1, sample_z=z)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=0, atol=1e-10 * np.max(np.abs(ref)))


@pytest.mark.parametrize('path', ['tm', '2d'])
@pytest.mark.parametrize('backend,bid', list(zip(BACKENDS, BACKEND_IDS))[1:], ids=BACKEND_IDS[1:])
def test_a_python_list_of_points_keeps_double_precision(backend, bid, path):
    """Points written as a plain list are converted at the solve's precision, not the backend's.

    ``torch.asarray`` makes a list of Python floats ``float32`` (Torch's default dtype), and the
    helpers cast only afterwards: ``x = -130.7`` landed 3.1e-6 off, ``z = 275.1`` 6.1e-6, and the
    solver-level field at explicit points sat 1.5e-8 (1D TM) and 2.1e-8 (2D) of the cell's
    largest entry from NumPy's, while every count-sampled cell agreed to 1e-15. The test above
    hands NumPy arrays, which carry their own float64, so it could not see it. Held here, both
    in-plane axes on the 2D path, at 1e-11 --
    above the 1.4e-13 the JAX 2D solve differs from NumPy's by anyway, three orders below the
    defect.
    """
    x, z = [-130.7, 0.0, 91.3], [60.3, 150.0, 275.1]
    y = [0.0] if path == 'tm' else [-47.9, 120.1]
    ref = _field(path, backend=0, sample_x=x, sample_y=y, sample_z=z)
    got = _field(path, backend=backend, sample_x=x, sample_y=y, sample_z=z)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-11 * np.abs(ref).max(), np.abs(got - ref).max()


def test_traced_thickness_with_explicit_z_says_why_it_cannot():
    """Bucketing a z into a layer is data-dependent, so it refuses instead of guessing.

    Counts stay traceable — that path never asks which layer a point is in — so the refusal is
    specific to explicit ``sample_z``, and the message says what to do instead.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def run(thick, sample_z):
        s = wavebit.rcwa(backend=1, top=1.0, bot=1.0, wavelength=WL,
                         thickness=thick, theta=np.deg2rad(24.0), tetm='tm', phi=None,
                         truncation_order=4, period=[PERIOD[0]], structure=jnp.asarray(TWO_LAYER),
                         fourier_coefficients='integrated', answers='field')
        s.solve()
        return jnp.sum(jnp.abs(s.reconstruct_field(sample_x=5, sample_y=1, sample_z=sample_z)) ** 2)

    # a count traces fine, gradient and all
    g = jax.grad(lambda t: run(t, 3))(jnp.asarray(THICK))
    assert np.all(np.isfinite(np.asarray(g)))

    with pytest.raises(TypeError, match='traced'):
        jax.grad(lambda t: run(t, np.array([50.0, 200.0])))(jnp.asarray(THICK))
