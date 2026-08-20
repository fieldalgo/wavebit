"""Sampling the field where the caller asks, rather than only on a uniform grid.

``calculate_field`` used to take three *counts* and lay out ``linspace`` grids from them. It now
also takes the coordinates themselves. The claim this file defends is not that some new
approximation is accurate — it is that **there is no approximation**: the reconstruction was
already pointwise in x, y and z, so naming the points changes which ones come back and nothing
else.

That is worth testing rather than asserting, because "pointwise" is exactly the kind of property
that holds until someone adds an FFT for speed. The strongest statement available is
*bit-identity*, and it is what most of these tests use: a point asked for explicitly must equal
the same point asked for as part of a grid, to the bit, not to a tolerance.

The one place the two spellings legitimately differ is a layer interface, where ``E_z`` is
two-valued. A count returns both limits (the interface appears twice, as one layer's bottom and
the next one's top); explicit coordinates can only return one, and the documented rule picks the
layer below. :func:`test_an_interface_returns_the_lower_layers_limit` pins that, including the
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
        kw.update(tetm=path, phi=None, fto=6, period=[PERIOD[0]], structure=TWO_LAYER)
    elif path == 'conical':
        kw.update(psi=np.pi / 2, phi=np.deg2rad(15.0), fto=6, period=[PERIOD[0]],
                  structure=TWO_LAYER)
    else:
        kw.update(psi=np.pi / 2, phi=np.deg2rad(15.0), fto=(3, 3), period=PERIOD,
                  structure=np.repeat(TWO_LAYER, 4, axis=1))
    s = wavebit.rcwa(**kw)
    if path == '4ff':
        s.formulation = '4ff'
    s.conv_solve()
    return s


def _field(path, backend=0, **kw):
    return np.asarray(_solver(path, backend).calculate_field(**kw))


def _zaxis(cell):
    """The z axis, which the coupled paths push one place along behind the incidence axis."""
    return 0 if cell.ndim == 4 else 1


@pytest.mark.parametrize('path', PATHS)
def test_explicit_points_reproduce_the_grid_they_were_taken_from(path):
    """Ask for the grid's own coordinates and get the grid back, to the bit.

    This is the whole claim in one assertion. ``res_z`` is left as a count here so that the
    comparison is against an unchanged z axis and only x is under test; z gets its own tests
    below, where the interface rule has to be accounted for.
    """
    grid = _field(path, res_x=11, res_y=5, res_z=4)
    named = _field(path, res_x=np.linspace(0, PERIOD[0], 11),
                   res_y=np.linspace(0, PERIOD[1], 5) if path in ('2d', '4ff') else 5,
                   res_z=4)
    assert named.shape == grid.shape
    assert np.array_equal(named, grid), np.max(np.abs(named - grid))


@pytest.mark.parametrize('path', PATHS)
def test_a_subset_costs_nothing_in_accuracy(path):
    """Three scattered x, asked for alone, equal those three inside a full grid — bitwise.

    The point of the feature: sample a line cut or a monitor plane without paying for a grid,
    and without the answer depending on what else was requested alongside it.
    """
    fine = np.linspace(0, PERIOD[0], 41)
    pick = [3, 17, 38]
    full = _field(path, res_x=fine, res_y=1, res_z=3)
    few = _field(path, res_x=fine[pick], res_y=1, res_z=3)
    assert np.array_equal(few, np.take(full, pick, axis=-2))


@pytest.mark.parametrize('path', PATHS)
def test_the_x_points_need_not_be_sorted_or_uniform_or_inside_the_cell(path):
    """No grid structure is assumed anywhere, so none of these are special cases.

    A coordinate outside ``[0, period)`` is a periodic continuation rather than an error: the
    kernel is ``exp(i k.r)`` and the Bloch phase carries it. Checked against the in-cell point
    it repeats, which is a statement about the physics rather than about the indexing.
    """
    xs = np.array([310.0, 0.0, 97.3, 41.0])
    got = _field(path, res_x=xs, res_y=1, res_z=3)
    shifted = _field(path, res_x=xs + PERIOD[0], res_y=1, res_z=3)

    # unsorted / non-uniform: each entry equals the same coordinate asked for on its own
    for j, x in enumerate(xs):
        one = _field(path, res_x=np.array([x]), res_y=1, res_z=3)
        assert np.array_equal(np.take(got, [j], axis=-2), one)

    # one period over: the Bloch phase, not equality
    kx0 = 1.0 * np.sin(np.deg2rad(24.0)) * (2 * np.pi / WL)
    if path in ('2d', '4ff', 'conical'):
        kx0 *= np.cos(np.deg2rad(15.0))
    assert np.allclose(shifted, got * np.exp(1j * kx0 * PERIOD[0]), rtol=0, atol=1e-9)


@pytest.mark.parametrize('path', PATHS)
def test_explicit_z_reproduces_the_grid_away_from_interfaces(path):
    """Every z the count spelling visits, except the doubled interface, comes back identical.

    A count puts ``res_z`` samples in each layer inclusive of both faces, so the internal
    interface appears twice. Those two rows are the only ones explicit coordinates cannot
    reproduce together, and they are excluded here and tested on their own next.
    """
    nz = 5
    grid = _field(path, res_x=7, res_y=1, res_z=nz)
    per = [np.linspace(0, d, nz) for d in THICK]
    z_abs = np.concatenate([per[0], per[1] + THICK[0]])

    named = _field(path, res_x=7, res_y=1, res_z=z_abs)
    assert named.shape == grid.shape

    ax = _zaxis(grid)
    keep = [i for i in range(len(z_abs)) if i != nz - 1]      # drop layer 0's bottom face
    assert np.array_equal(np.take(named, keep, axis=ax), np.take(grid, keep, axis=ax))


@pytest.mark.parametrize('path', PATHS)
def test_an_interface_returns_the_lower_layers_limit(path):
    """The documented rule, and the nudge that gets the other side.

    Half-open ``[top, bottom)`` means the interface belongs to the layer below; ``z - eps``
    recovers the upper layer's limit.

    **Whether the two limits differ at all depends on the path, and that is physics rather
    than bookkeeping.** Only ``E_z`` is discontinuous across a *horizontal* interface. TE's
    layout is ``(Ey, Hx, Hz)`` — three tangential components, all continuous — so for it the
    rule is unobservable and both spellings must agree exactly. Every other path carries an
    ``E_z`` and must show the jump. Asserting the split rather than skipping TE keeps this
    honest: if TE ever *did* differ at an interface, something is wrong with it.
    """
    nz = 5
    grid = _field(path, res_x=7, res_y=1, res_z=nz)
    ax = _zaxis(grid)
    upper = np.take(grid, [nz - 1], axis=ax)      # layer 0's bottom face
    lower = np.take(grid, [nz], axis=ax)          # layer 1's top face

    at = _field(path, res_x=7, res_y=1, res_z=np.array([THICK[0]]))
    assert np.array_equal(at, lower)

    just_above = _field(path, res_x=7, res_y=1, res_z=np.array([THICK[0] - 1e-9]))
    assert np.allclose(just_above, upper, rtol=0, atol=1e-6)

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
    a = _field(path, res_x=5, res_y=1, res_z=z)
    b = _field(path, res_x=5, res_y=1, res_z=z[perm])
    ax = _zaxis(a)
    assert np.array_equal(np.take(a, perm, axis=ax), b)


@pytest.mark.parametrize('path', PATHS)
def test_a_repeated_z_is_returned_twice(path):
    """No implicit de-duplication — a request of length n returns n samples."""
    z = np.array([80.0, 80.0, 300.0])
    got = _field(path, res_x=5, res_y=1, res_z=z)
    ax = _zaxis(got)
    assert got.shape[ax] == 3
    assert np.array_equal(np.take(got, [0], axis=ax), np.take(got, [1], axis=ax))


@pytest.mark.parametrize('path', PATHS)
def test_the_stack_s_own_faces_are_reachable(path):
    """z = 0 and z = total, the closed ends, must not fall off either rule."""
    got = _field(path, res_x=5, res_y=1, res_z=np.array([0.0, TOTAL]))
    assert np.all(np.isfinite(np.asarray(got)))
    grid = _field(path, res_x=5, res_y=1, res_z=3)
    ax = _zaxis(grid)
    assert np.array_equal(np.take(got, [0], axis=ax), np.take(grid, [0], axis=ax))
    assert np.array_equal(np.take(got, [1], axis=ax), np.take(grid, [-1], axis=ax))


@pytest.mark.parametrize('z', [-1e-3, 410.001, 1e4])
def test_z_outside_the_stack_is_refused(z):
    """Not extrapolated: the half-spaces carry mode sets this state does not hold."""
    with pytest.raises(ValueError, match='inside the stack'):
        _field('tm', res_x=5, res_y=1, res_z=np.array([z]))


def test_a_float_count_is_not_silently_a_coordinate():
    """``res_z=5.0`` is a mistake, and it should look like one rather than mean ``[5.0]``."""
    got = _field('tm', res_x=5, res_y=1, res_z=np.array([5.0]))
    ax = _zaxis(got)
    assert got.shape[ax] == 1                      # an array of one z is one sample

    # a bare float is neither, and reading it as either would be a silent wrong answer.
    # `res_y` is checked on a 2-D path: the pure 1-D solves overwrite it with 1, because the
    # structure is y-invariant and one row is the whole answer, so nothing there can see it.
    for path, kw in (('tm', dict(res_z=5.0)), ('tm', dict(res_x=5.0)),
                     ('2d', dict(res_z=5.0)), ('2d', dict(res_x=5.0)),
                     ('2d', dict(res_y=5.0))):
        call = dict(res_x=5, res_y=2, res_z=3)
        call.update(kw)
        with pytest.raises(TypeError, match='integer count or a 1-D array'):
            _field(path, **call)


@pytest.mark.parametrize('backend,bid', list(zip(BACKENDS, BACKEND_IDS)), ids=BACKEND_IDS)
def test_every_backend_samples_the_same_points(backend, bid):
    """The helpers are in ``_core`` and shared, so all three must agree — and with the grid."""
    z = np.array([0.0, 60.0, 150.0, 275.0, TOTAL])
    x = np.array([0.0, 133.3, 251.7, 400.0])
    ref = _field('tm', backend=0, res_x=x, res_y=1, res_z=z)
    got = _field('tm', backend=backend, res_x=x, res_y=1, res_z=z)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=0, atol=1e-10 * np.max(np.abs(ref)))


def test_traced_thickness_with_explicit_z_says_why_it_cannot():
    """Bucketing a z into a layer is data-dependent, so it refuses instead of guessing.

    Counts stay traceable — that path never asks which layer a point is in — so the refusal is
    specific to explicit ``res_z``, and the message says what to do instead.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def run(thick, res_z):
        s = wavebit.rcwa(backend=1, top=1.0, bot=1.0, wavelength=WL,
                         thickness=thick, theta=np.deg2rad(24.0), tetm='tm', phi=None,
                         fto=4, period=[PERIOD[0]], structure=jnp.asarray(TWO_LAYER),
                         fourier_coefficients='integrated', answers='field')
        s.conv_solve()
        return jnp.sum(jnp.abs(s.calculate_field(res_x=5, res_y=1, res_z=res_z)) ** 2)

    # a count traces fine, gradient and all
    g = jax.grad(lambda t: run(t, 3))(jnp.asarray(THICK))
    assert np.all(np.isfinite(np.asarray(g)))

    with pytest.raises(TypeError, match='traced'):
        jax.grad(lambda t: run(t, np.array([50.0, 200.0])))(jnp.asarray(THICK))
