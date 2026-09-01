"""`projector='circular'` — the direction-free limit of the NVF projector.

The NVF projector has always been `P = n n^T`, built from the normal field's *direction*
alone. On the **medial axis** — the ridge where every boundary point is equidistant — there is
no direction: the tie is exact, both answers are arbitrary, and which one is reported is an
accident of where the samples fell. No better direction fixes that, because none exists.

`projector='circular'` stops reading one. It builds `P = J J^H` from a Jones vector that
carries the field's magnitude as well (Antos 2009; Liu & Fan 2012, Eq. (52)), so that

    |t| = 1   ->  phi = 0     ->  P = n n^T, exactly
    |t| -> 0  ->  phi = pi/4  ->  Pxx = Pyy = 1/2, Pxy = -i/2, Pyx = +i/2

and the second row does not mention the direction at all. It is reached as a **limit**: no
threshold, no "is this pixel on the medial axis?" test, nothing to tune.

**A separate axis, not a fifth `normal_method`.** `normal_method` chooses how the field is
*built*; this chooses what is done *with* it. They compose, so a fifth value would have made
one list of eight where there are two lists of four and two.

**What it is measured with.** The translation identity, `max |E(rolled) - D E D^-1|` with
`D = diag(exp(-i G.d))`: translating a periodic structure multiplies every Fourier coefficient
by a phase, so every convolution matrix transforms as a *similarity*. It holds at any
truncation, needs no solve, and `epz_i` — which reads no normal field at all — is the built-in
control that pins the convention whatever the field does. See
`test_nvf_lattice_translation._translation_defect`, whose harness this file reuses.

**Two honest limits on what is claimed here.**

* On today's `develop` the covariance defect this was prototyped against is **already gone**:
  `_CONTOUR_TIE_SHEAR` (2026-09-01) settles the medial-axis tie the same way at every
  placement, which is all covariance asks for. So the projector is measured here with the
  shear switched off, which is the only configuration in which the instrument can still see
  the arbitrariness. The two fixes are answers to different halves of one defect — the shear
  makes the arbitrary answer *stable*, this makes it *not be read*.
* **The saddle is not fixed**, and it is pinned here saying so. Its two nearest boundary
  points sit on *different* discs, so the tie is a discrete jump between two separated
  directions rather than a direction fading out — `|t|` does not go to zero there, so the
  limit is never reached. Roughly one order, and no more.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core import factorization as _fac
from wavebit._core.factorization import (jones_projector_fields, normal_vector_field,
                                         to_conv_mat_raster_nvf)
from wavebit._core.geometry import StaleGeometryError

N = 64
_AX = np.arange(N)
_U, _V = np.meshgrid(_AX, _AX)
_R2 = (_U - 32.) ** 2 + (_V - 32.) ** 2

#: A hole leaves a medial *point* at its centre and a medial *ring* through the wall; the
#: cleanest raster scene that reaches the null without also reaching a saddle.
ANNULUS = np.where((_R2 <= 30 ** 2) & (_R2 > 14 ** 2), 12.25, 1.0)
DISC = np.where(_R2 <= 30 ** 2, 12.25, 1.0)
#: The saddle: two nearest boundary points on *different* discs. Kept precisely because the
#: limit does not fix it — a pin that only covers what works proves the wrong thing.
SADDLE = np.where((((_U - 16.) ** 2 + (_V - 32.) ** 2) <= 14 ** 2) |
                  (((_U - 48.) ** 2 + (_V - 32.) ** 2) <= 14 ** 2), 12.25, 1.0)

SOLVE_KW = dict(backend=0, thickness=[0.3], truncation_order=(6, 6), top=1.0, bot=1.0,
                wavelength=0.94, psi=np.pi / 2, theta=0.1, phi=0.2, period=[1.0, 1.0],
                factorization='nvf')


# ------------------------------------------------------------------ the projector itself

@pytest.mark.parametrize('magnitude,name', [(1.0, 'full'), (0.0, 'null')])
def test_the_two_limits_are_exact(magnitude, name):
    """Both ends of Eq. (52) are exact rather than approached, and that is the whole design.

    At `|t| = 1` the projector must be *bit-identical* to `n n^T`, because that is what makes
    `'circular'` a deformation of the shipped scheme rather than a different one: everywhere
    the field is at full strength — which is everywhere but the medial axis — the two agree to
    the last bit and the option costs nothing.

    At `|t| = 0` every entry must lose the direction. That is checked by *varying* the
    direction over a full turn and asserting the answer does not move, which is a stronger
    statement than comparing against the four constants and the one the medial axis needs.
    """
    theta = np.linspace(0.0, 2 * np.pi, 17)
    n_x, n_y = np.cos(theta), np.sin(theta)
    p_xx, p_xy, p_yx, p_yy = jones_projector_fields(n_x, n_y, np.full_like(theta, magnitude))

    if name == 'full':
        assert np.array_equal(p_xx, n_x ** 2), 'Pxx must be n_x^2 to the bit at |t| = 1'
        assert np.array_equal(p_yy, n_y ** 2)
        assert np.abs(p_xy - n_x * n_y).max() == 0.0, 'Pxy must be n_x n_y to the bit'
        assert np.abs(p_yx - n_x * n_y).max() == 0.0
        return

    for slot, want in ((p_xx, 0.5), (p_yy, 0.5), (p_xy, -0.5j), (p_yx, +0.5j)):
        assert np.abs(slot - want).max() < 1e-15, f'{slot[:3]} != {want} at |t| = 0'
        assert np.ptp(slot.real) < 1e-15 and np.ptp(slot.imag) < 1e-15, (
            'the null projector still moved with the direction, which is the one thing it '
            'exists not to do')


@pytest.mark.parametrize('magnitude', [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_it_is_a_hermitian_rank_one_projector_at_every_magnitude(magnitude):
    """`P @ P == P`, `P = P^H`, `trace(P) = 1` — all the way along, not only at the ends.

    The trace is the one the rest of the module depends on: `_nvf_assemble` reaches Li's
    rules exactly when `C_xx + C_yy = I`, which holds only because `P_xx + P_yy = 1`
    pointwise. A deformation that lost it would conserve energy and quietly stop being a
    factorization of anything.
    """
    theta = np.linspace(0.0, 2 * np.pi, 9)
    n_x, n_y = np.cos(theta), np.sin(theta)
    p_xx, p_xy, p_yx, p_yy = jones_projector_fields(n_x, n_y, np.full_like(theta, magnitude))
    P = np.array([[p_xx, p_xy], [p_yx, p_yy]]).transpose(2, 0, 1)

    assert np.abs(p_xx + p_yy - 1.0).max() < 1e-15, 'trace(P) != 1'
    assert np.abs(P - np.conj(P.transpose(0, 2, 1))).max() < 1e-15, 'P is not Hermitian'
    assert np.abs(P @ P - P).max() < 1e-15, 'P is not idempotent'
    assert np.abs(np.linalg.eigvalsh(P) - np.array([0.0, 1.0])).max() < 1e-14, 'not rank one'


def test_a_pixel_with_no_direction_still_gets_a_projector():
    """`normal_vector_field` answers `(0, 0)` on the skeleton, and `(0, 0)` is not a unit
    vector — so the algebra has to put *some* direction there.

    Any one is exact, because `|t|` is zero at precisely those pixels and the direction has
    cancelled out of every entry. This asserts the outcome rather than the substitution: the
    projector at a direction-less pixel is the null projector, not something degenerate.
    """
    n_x = np.array([0.0, 0.0, 1.0])
    n_y = np.array([0.0, 0.0, 0.0])
    p_xx, p_xy, p_yx, p_yy = jones_projector_fields(n_x, n_y, np.array([0.0, 0.0, 1.0]))
    assert np.allclose([p_xx[0], p_yy[0]], [0.5, 0.5])
    assert np.allclose([p_xy[0], p_yx[0]], [-0.5j, 0.5j])
    assert np.abs(p_xx + p_yy - 1.0).max() < 1e-15


# ------------------------------------------------------------------ the field API

@pytest.mark.parametrize('method', ['distance', 'contour', 'spectral', 'smooth'])
def test_the_magnitude_is_the_same_field_for_every_construction(method):
    """`|grad d|` is a property of the *interface set*, not of a construction.

    The four constructions disagree about the direction — that is what `normal_method` is —
    and agree about where the medial axis is, because all four are looking at the same
    interfaces. So the magnitude comes from a distance transform run beside the construction
    rather than out of it; a unit field has no magnitude of its own to read.

    Also pins the two properties the formula's domain needs: max exactly 1, and a genuine
    null on the medial axis (which is what there is to find).
    """
    pytest.importorskip('contourpy') if method == 'contour' else None
    _, _, mag = normal_vector_field(ANNULUS, method=method, with_magnitude=True)
    reference = normal_vector_field(ANNULUS, method='distance', with_magnitude=True)[2]

    assert np.array_equal(mag, reference), (
        f"{method!r} reported a different magnitude from 'distance'; it is the same "
        f'distance transform and must be the same array')
    assert mag.max() == 1.0, 'the scaling has to put the maximum exactly at 1'
    assert mag.min() < 1e-12, 'this scene was supposed to have a medial axis'


def test_a_one_cell_axis_reports_no_null():
    """On a genuine one-row raster the normal is the constant along-axis one, and exact.

    The shortcut in `normal_vector_field` never builds a distance map there, so it reports
    magnitude 1 everywhere and `'circular'` is a no-op — which is right, because NVF with a
    constant normal reduces to Li's rules *exactly* and there is nothing to improve.

    That is a fact about the one-row array and **not** about 1D gratings in general; the next
    test is the one that says what happens to those.
    """
    row = np.where(np.abs(np.arange(48) - 24) < 11, 12.25, 1.0)[None, :]
    _, _, mag = normal_vector_field(row, with_magnitude=True)
    assert np.array_equal(mag, np.ones_like(mag))
    assert normal_vector_field(row, with_magnitude=True)[0].shape == row.shape


def test_a_1d_grating_reached_through_the_2d_path_costs_accuracy():
    """The cost is largest exactly where the truth is known, and that is worth pinning.

    A y-invariant raster with more than one row goes through the 2D builder, and
    `_normal_field_floor` block-repeats it to `4*truncation_order+1` rows — so the one-cell
    shortcut does not apply and the stripe centres become real medial *columns*. Li's rules
    are **exact** for such a structure, which makes this the one scene with a reference that
    is not itself an approximation.

    Total `T`, 8 identical rows of a 64-wide 50%% grating, against `factorization='li'`:

    ====  ==========  ============
    to    nvf linear  nvf circular
    ====  ==========  ============
    4     9.2e-05     3.5e-01
    6     2.1e-06     2.0e-02
    8     3.7e-05     1.6e-02
    12    8.4e-05     8.9e-03
    ====  ==========  ============

    **Why the null degrades so differently under the two.** `'distance'` answers `(0, 0)` on
    a medial column, so the linear projector is **zero** there and the correction term simply
    switches off — the scheme degrades to the direct rule, which is a graceful fallback. The
    circular projector is a *unit-trace* projector at every pixel including that one, so it
    applies a half-and-half correction with a definite handedness instead. "No opinion" and
    "an arbitrary opinion" are not equally safe, and this is the measurement that says so.
    """
    row = np.repeat(np.where(np.abs(np.arange(64) - 32) < 15, 12.25, 1.0)[None, :],
                    8, axis=0)[None]
    kw = dict(structure=row, **{k: v for k, v in SOLVE_KW.items()
                                if k not in ('factorization', 'truncation_order')},
              truncation_order=(8, 8))
    exact = float(np.sum(np.asarray(wavebit.rcwa(**kw, factorization='li').solve().T)))
    linear = float(np.sum(np.asarray(
        wavebit.rcwa(**kw, factorization='nvf').solve().T)))
    circular = float(np.sum(np.asarray(
        wavebit.rcwa(**kw, factorization='nvf', projector='circular').solve().T)))

    assert abs(linear - exact) < 1e-3, (
        f'the linear projector should be near Li on a 1D grating: {linear:.10f} against '
        f'{exact:.10f}')
    assert 1e-3 < abs(circular - exact) < 1e-1, (
        f'the circular cost on a 1D grating moved out of its measured band: '
        f'{circular:.10f} against {exact:.10f}. It is pinned from *both* sides — shrinking '
        f'is good news that this test should be updated for, not silently absorbed.')


def test_a_uniform_layer_stays_exactly_linear():
    """No interface, no distance map, no medial axis — and the direct rule is already exact.

    The guard is in `_field_magnitude`: an all-zero gradient would divide by its own zero
    maximum. It answers 1 everywhere instead, which keeps this layer on `n n^T`.
    """
    flat = np.full((16, 16), 4.0)
    lin = to_conv_mat_raster_nvf(flat[None], 3, 3, projector='linear')
    cir = to_conv_mat_raster_nvf(flat[None], 3, 3, projector='circular')
    for a, b in zip(lin, cir):
        assert np.array_equal(np.asarray(a), np.asarray(b))


# ------------------------------------------------------------------ the default is untouched

@pytest.mark.parametrize('method', ['distance', 'contour', 'spectral', 'smooth'])
def test_the_default_is_bit_identical_to_naming_it(method):
    """An untouched keyword and an explicit `'linear'` must be the same computation.

    Not "close": the same. Everything measured in `docs/` was measured on this path, and an
    option that moved it by a last bit would invalidate all of it silently.
    """
    pytest.importorskip('contourpy') if method == 'contour' else None
    unset = to_conv_mat_raster_nvf(ANNULUS[None], 4, 4, normal_method=method)
    named = to_conv_mat_raster_nvf(ANNULUS[None], 4, 4, normal_method=method,
                                   projector='linear')
    for a, b in zip(unset, named):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_the_linear_path_fills_both_off_diagonal_slots_with_one_object():
    """`P = n n^T` is symmetric, so `eps_xy` and `eps_yx` are not merely equal — they are one
    matrix, built once.

    That is what makes the default path provably free of the new arithmetic rather than
    coincidentally equal to it, and it is the property `_nvf_assemble`'s `C_yx=None` states.
    """
    n_x, n_y = normal_vector_field(ANNULUS)
    eps = np.asarray(ANNULUS, dtype=np.complex128)
    conv = _fac.nvf_conv_matrices(np, eps, n_x, n_y, 4, 4)
    assert conv.epxy is conv.epyx, 'the linear path built the symmetric slot twice'

    # ...and it survives to the builder as an equality, which is what a caller sees. (Not as
    # identity: `to_conv_mat_raster_nvf` stacks the per-layer slots, and two stacks of the
    # same matrix are two arrays.)
    stacked = to_conv_mat_raster_nvf(ANNULUS[None], 4, 4)
    assert np.array_equal(np.asarray(stacked.epxy), np.asarray(stacked.epyx))


def test_the_circular_path_makes_the_two_off_diagonal_slots_differ():
    """The other half of the same fact, and the first time this path produces `eps_xy != eps_yx`.

    A complex `P` is Hermitian rather than symmetric, so the two slots stop coinciding. If
    they still did, the imaginary half of the projector would have gone missing somewhere
    between the field and the assembly — and every real-valued check in the suite would still
    pass.
    """
    conv = to_conv_mat_raster_nvf(ANNULUS[None], 4, 4, projector='circular')
    xy, yx = np.asarray(conv.epxy), np.asarray(conv.epyx)
    assert np.abs(xy - yx).max() > 1e-3, 'the two off-diagonal slots did not actually differ'
    # And they differ the way a Hermitian projector says they must: `P_yx = conj(P_xy)`
    # pointwise, so their convolution matrices are conjugate transposes of one another.
    # Checked because "different" alone would also be satisfied by the imaginary half landing
    # in one slot and not the other.
    assert np.abs(xy - np.conj(np.swapaxes(yx, -1, -2))).max() < 1e-12


def test_the_projector_slots_still_sum_to_the_identity():
    """`C_xx + C_yy = I`, which is what makes NVF a superset of Li rather than a rival to it.

    It survives the deformation because `trace(P) = 1` pointwise at every magnitude. Checked
    through `_nvf_projector` rather than through the pointwise fields, because it is the
    *convolution matrices* the assembly adds.
    """
    _, _, mag = normal_vector_field(ANNULUS, with_magnitude=True)
    n_x, n_y = normal_vector_field(ANNULUS)
    C = np.complex128
    for magnitude in (None, mag):
        slots = _fac._nvf_projector(np, n_x, n_y, 4, 4, C, magnitude=magnitude)
        c_xx, c_yy = np.asarray(slots[0]), np.asarray(slots[2])
        eye = np.eye(c_xx.shape[0], dtype=C)
        # `'distance'` answers (0, 0) on the skeleton, so the *linear* projector genuinely
        # falls short of the identity there and the tolerance is the skeleton's own weight.
        # The circular one has no such hole: it is a unit-trace projector at every pixel.
        tol = 1e-2 if magnitude is None else 1e-13
        assert np.abs(c_xx + c_yy - eye).max() < tol


# ------------------------------------------------------------------ the solve

@pytest.mark.parametrize('scene,name', [(ANNULUS, 'annulus'), (DISC, 'disc')])
def test_a_circular_solve_conserves_energy(scene, name):
    """A lossless stack, so `R + T = 1` — the floor every factorization has to clear.

    It clears nothing else, and that is worth saying beside the assertion: NVF's assembly
    stays Hermitian under an arbitrary projector, so a *wrong* one conserves energy just as
    exactly. This is a guard against a broken build, not evidence the projector is right.
    """
    res = wavebit.rcwa(structure=scene[None], **SOLVE_KW, projector='circular').solve()
    total = float(np.sum(np.asarray(res.R)) + np.sum(np.asarray(res.T)))
    assert abs(total - 1.0) < 1e-11, f'{name}: R + T = {total:.12f}'


def test_both_projectors_converge_to_the_same_physics():
    """The correctness question, separate from the accuracy one: do they agree in the limit?

    They have to. As the truncation rises `(1/eps)^ ^-1 -> eps^`, so `Delta = eps^ - A -> 0`
    and the projector's entire contribution vanishes — any unit-trace projector gives the same
    answer eventually, and one that did not would not be a factorization of this structure at
    all. This is what rules out the failure that *looks* like a working option: a projector
    landed in the wrong slot returns a plausible, energy-conserving, entirely different number
    (the slot-order trap in `_core/factorization`'s module docstring, measured at `R + T =
    1.61`).

    Against the analytic reference at `truncation_order=14` — the exact shape with the exact
    normal, a different modeller and a different series — on the circle of the doc's own A/B
    table (r=0.30, eps 5.76, N=256):

    ====  ==========  ============
    to    nvf linear  nvf circular
    ====  ==========  ============
    6     5.15e-03    6.69e-03
    8     3.70e-04    1.99e-03
    10    7.61e-06    1.78e-03
    14    3.83e-04    1.36e-03
    ====  ==========  ============

    Both converge on the reference; `'circular'` is 3-5x further out at every order. That
    ratio is the accuracy half of the trade, and the doc's table is where it is recorded in
    full. N=128 here rather than 256 to keep the test cheap; the numbers differ, the
    conclusion does not.
    """
    n = 128
    ax = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(ax, ax)
    raster = np.where((X - .5) ** 2 + (Y - .5) ** 2 <= 0.3 ** 2, 5.76, 1.0)[None]
    shapes = [{'background': 1.0, 'shapes': [{'type': 'circle', 'cx': 0.0, 'cy': 0.0,
                                              'lx': 0.6, 'material': 5.76}]}]
    kw = dict(backend=0, period=[1., 1.], wavelength=.55, thickness=[.3],
              theta=np.deg2rad(12), phi=np.deg2rad(30), psi=np.deg2rad(40), top=1., bot=1.)

    def total(structure, to, **over):
        return float(np.sum(np.asarray(wavebit.rcwa(
            structure=structure, truncation_order=[to, to], **kw, **over).solve().T)))

    ref = total(shapes, 12)
    nvf = dict(factorization='nvf', normal_method='contour')
    linear = abs(total(raster, 10, **nvf) - ref)
    circular = abs(total(raster, 10, **nvf, projector='circular') - ref)
    assert linear < 5e-3, f'the linear projector missed its own reference by {linear:.2e}'
    assert circular < 1e-2, (
        f'the circular projector is {circular:.2e} from the analytic reference — far enough '
        f'that it is answering a different problem rather than converging more slowly')


def test_the_basis_is_chiral_and_that_is_where_the_cost_is():
    """The finding that matters most about this option, pinned so it cannot be forgotten.

    `J = n cos(phi) + i n_perp sin(phi)` picks a rotation *sense*: `n_perp = (-n_y, n_x)` is a
    fixed handedness, and `-i/2` in the null projector is left- rather than right-circular. So
    the construction is **chiral**, and a structure and its mirror image stop agreeing.

    Measured at the solve (off-centre scenes, N=48, `truncation_order=5`, `max|dR|` between a
    structure and its x-mirror):

    =========  ==========  ==========  ==========  ==========
    scene      distance    contour     spectral    smooth
    =========  ==========  ==========  ==========  ==========
    circle     2.1e-15     7.7e-05     1.6e-04     2.4e-15
    circle*    3.96e-04    3.79e-04    5.22e-04    3.50e-04
    annulus    3.0e-15     2.4e-04     7.2e-04     2.9e-15
    annulus*   7.40e-03    8.25e-03    7.46e-03    8.04e-03
    =========  ==========  ==========  ==========  ==========

    (starred rows are `projector='circular'`.) `'distance'` and `'smooth'` mirror **exactly**
    under the linear projector — `test_nvf_normal_methods.test_mirror_asymmetry_of_the_solve_is_bounded`
    pins them at 1e-12 — and the circular projector costs them that outright, by four to
    twelve orders.

    **It is the method, not this implementation, and the check below is what says so.**
    Conjugating the projector is the *other* circular state and an equally valid rank-one
    projector. Mirroring the structure and conjugating the projector agree to **3.4e-15**, so
    the mirror asymmetry is exactly the basis's handedness and nothing else. The two
    handednesses differ from each other by 7.4e-3, so the choice is real rather than a gauge;
    the sign implemented is the published one (Liu & Fan Eq. (52) with `+i sin phi`).

    A workflow that needs exact mirror symmetry of the output must stay on `'linear'`.
    """
    eps = np.zeros((48, 48))
    ax = (np.arange(48) + 0.5) / 48
    X, Y = np.meshgrid(ax, ax)
    rr = np.hypot(X - 0.4, Y - 0.55)
    eps = np.where((rr <= 0.3) & (rr > 0.14), 5.76, 1.0)
    cfg = dict(backend=0, period=[1., 1.], wavelength=.55, thickness=[.3], theta=0.,
               phi=0., psi=0., top=1., bot=1., truncation_order=[5, 5],
               factorization='nvf', normal_method='distance')

    def mirror_gap(**over):
        a = wavebit.rcwa(structure=eps[None], **cfg, **over).solve()
        b = wavebit.rcwa(structure=eps[:, ::-1].copy()[None], **cfg, **over).solve()
        return float(np.abs(np.asarray(a.R) - np.asarray(b.R)[:, ::-1]).max())

    assert mirror_gap() < 1e-12, "'distance' must mirror exactly under the linear projector"
    circular = mirror_gap(projector='circular')
    assert circular > 1e-4, (
        f'the circular projector mirrored to {circular:.2e}, where the chirality of the Jones '
        f'basis should show. If this has genuinely become symmetric the construction has '
        f'changed and the docs need re-measuring — it is not a free improvement.')

    # And the diagnosis: mirroring the structure IS conjugating the projector.
    original = _fac.jones_projector_fields

    def conjugated(n_x, n_y, magnitude):
        p_xx, p_xy, p_yx, p_yy = original(n_x, n_y, magnitude)
        return p_xx, np.conj(p_xy), np.conj(p_yx), p_yy

    left = wavebit.rcwa(structure=eps[None], **cfg, projector='circular').solve()
    _fac.jones_projector_fields = conjugated
    try:
        right = wavebit.rcwa(structure=eps[:, ::-1].copy()[None], **cfg,
                             projector='circular').solve()
    finally:
        _fac.jones_projector_fields = original
    gap = float(np.abs(np.asarray(left.R) - np.asarray(right.R)[:, ::-1]).max())
    assert gap < 1e-12, (
        f'mirroring the structure and flipping the handedness disagreed by {gap:.2e}. They '
        f'are the same operation, so this says the asymmetry has a second source and the '
        f'docstring above is wrong about where the cost comes from.')


# ------------------------------------------------------------------ the refusals

def test_it_is_refused_where_nothing_would_read_it():
    """An unread knob is the defect this library refuses everywhere, and this is its door.

    Two places have no raster normal field to form a projector from: Li builds none at all,
    and the smooth shape path builds one from the geometry, which carries no magnitude for the
    limit to read. Both run to completion with the request ignored, which is exactly the
    silent non-choice `_reject_unread_normal_method` exists to stop one axis over.
    """
    with pytest.raises(NotImplementedError, match="factorization='li' builds no normal field"):
        wavebit.rcwa(structure=DISC[None], **{**SOLVE_KW, 'factorization': 'li'},
                     projector='circular').solve()

    shapes = [dict(background=1.0,
                   shapes=[dict(type='circle', cx=0.0, cy=0.0, l=0.6, material=12.25)])]
    with pytest.raises(NotImplementedError, match='carries no magnitude'):
        wavebit.rcwa(structure=shapes, **{k: v for k, v in SOLVE_KW.items()
                                          if k != 'factorization'},
                     projector='circular').solve()


def test_an_unknown_value_names_the_ones_that_exist():
    with pytest.raises(ValueError, match=r"projector='bogus'.*'linear', 'circular'"):
        wavebit.rcwa(structure=DISC[None], **SOLVE_KW, projector='bogus')


def test_the_axis_is_not_a_normal_method_value():
    """The API decision, pinned so it cannot be undone by accident.

    `'circular'` describes what is done *with* a field and `normal_method` describes how one
    is *built*, so folding it in would make one keyword of eight values where the two are
    genuinely independent — and every `normal_method` would then need a circular twin.
    """
    from wavebit import NormalMethod, Projector
    assert {m.value for m in NormalMethod}.isdisjoint({m.value for m in Projector})
    with pytest.raises(ValueError, match='normal_method'):
        wavebit.rcwa(structure=DISC[None], **SOLVE_KW, normal_method='circular')


# ------------------------------------------------------------------ frozen geometry

def test_a_frozen_field_carries_the_magnitude_it_was_frozen_for():
    """The record is the field, and under `'circular'` the field includes a magnitude."""
    solver = wavebit.rcwa(structure=ANNULUS[None], **SOLVE_KW, projector='circular')
    record = solver.freeze_geometry()
    assert len(record.normal_field[0]) == 3, 'the frozen field carried no magnitude'
    replayed = float(np.sum(np.asarray(solver.solve(geometry=record).T)))
    direct = float(np.sum(np.asarray(solver.solve().T)))
    assert abs(replayed - direct) < 1e-12, 'the replay did not reproduce the rebuild'


def test_a_linear_record_is_refused_under_the_circular_projector():
    """It describes a field this configuration cannot use, so it is refused rather than
    silently served with the other projector.

    The arity is the tell and it is exact: a two-entry record has no magnitude, full stop.
    Serving it under `'circular'` would answer with `n n^T` while the solver reported
    `projector='circular'` — the unread knob again, one level down.
    """
    solver = wavebit.rcwa(structure=ANNULUS[None], **SOLVE_KW)
    record = solver.freeze_geometry()
    solver.projector = 'circular'
    with pytest.raises(StaleGeometryError, match="frozen under projector='linear'"):
        solver.solve(geometry=record)


def test_a_circular_record_replays_under_the_linear_projector():
    """The other direction is not an error: the extra entry is simply not read.

    Worth pinning rather than leaving to chance — the natural way to write the arity check is
    an equality, which would refuse this and force a re-freeze for nothing.
    """
    solver = wavebit.rcwa(structure=ANNULUS[None], **SOLVE_KW, projector='circular')
    record = solver.freeze_geometry()
    solver.projector = 'linear'
    replayed = float(np.sum(np.asarray(solver.solve(geometry=record).T)))
    direct = float(np.sum(np.asarray(wavebit.rcwa(structure=ANNULUS[None],
                                                  **SOLVE_KW).solve().T)))
    assert abs(replayed - direct) < 1e-12


def test_it_crosses_a_jit_boundary_and_carries_a_gradient():
    """The frozen record grew a third entry, and a jitted solve is where that would show.

    `solve_field` flattens the solver on every call, so a record whose shape the trace cannot
    handle takes out the whole jitted path rather than one test — which is how `excitation`
    was once dropped on every jitted solve while a test named for it passed. Built **outside**
    the trace for that reason: a solver constructed inside one is never flattened, so a jit
    test that builds it there proves nothing.

    The gradient matters as much as the value. The magnitude is frozen geometry like the
    direction, so it must be a constant to the trace and not a hole in it.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    solver = wavebit.rcwa(structure=ANNULUS[None], **{**SOLVE_KW, 'backend': 1},
                          projector='circular')
    frozen = solver.freeze_geometry()
    eager = float(np.sum(np.asarray(solver.solve(geometry=frozen).T)))

    jitted = jax.jit(lambda e: jnp.sum(solver.solve(structure=e, geometry=frozen).T))
    assert abs(float(jitted(jnp.asarray(ANNULUS[None]))) - eager) < 1e-11

    grad = jax.grad(lambda e: jnp.sum(solver.solve(structure=e, geometry=frozen).T).real)(
        jnp.asarray(ANNULUS[None]))
    grad = np.asarray(grad)
    assert np.isfinite(grad).all(), 'the circular projector put a NaN in the gradient'
    assert np.abs(grad).max() > 0.0, 'the gradient to the raster vanished entirely'


@pytest.mark.parametrize('backend,name', [(1, 'jax'), (2, 'torch')])
def test_every_backend_builds_the_same_circular_projector(backend, name):
    """The complex projector is the first thing on this path with a genuinely complex
    pointwise field, so it is the first place a backend could differ on the imaginary half.
    """
    pytest.importorskip({1: 'jax', 2: 'torch'}[backend])
    def total(which):
        return float(np.sum(np.asarray(wavebit.rcwa(
            structure=ANNULUS[None], **{**SOLVE_KW, 'backend': which},
            projector='circular').solve().T)))

    reference, got = total(0), total(backend)
    assert abs(got - reference) < 1e-12, f'{name} {got:.12f} against numpy {reference:.12f}'


# ------------------------------------------------------------------ the translation identity

def _phase_diagonal(to, shift_v, shift_u, n_v, n_u):
    """``D = diag(exp(-i G.d))`` for an integer pixel shift, in ``conv_matrix_2d``'s order."""
    m = np.arange(-to, to + 1)
    V, U = np.meshgrid(m, m, indexing='ij')
    return np.exp(-2j * np.pi * (V.ravel() * shift_v / n_v + U.ravel() * shift_u / n_u))


def _translation_defect(cell, shift, method, projector, to=3):
    """``max |E(rolled) - D E D^-1|`` over the four slots the normal field reaches.

    `test_nvf_lattice_translation._translation_defect`'s harness, with `epyx` added — under
    `'circular'` it is no longer `epxy`, so leaving it out would skip half of what the new
    projector produces. Every slot is divided by the *largest* of them for the reason that
    file records: a slot that is identically zero reports round-off over round-off when
    divided by itself.
    """
    n_v, n_u = cell.shape
    rolled = np.roll(np.roll(cell, shift[0], axis=0), shift[1], axis=1)
    kw = dict(normal_method=method, projector=projector)
    A = to_conv_mat_raster_nvf(cell[None], to, to, **kw)
    B = to_conv_mat_raster_nvf(rolled[None], to, to, **kw)
    d = _phase_diagonal(to, shift[0], shift[1], n_v, n_u)
    slots = [(np.asarray(A[k])[0], np.asarray(B[k])[0]) for k in (0, 1, 3, 4)]
    scale = max(max(float(np.abs(a).max()) for a, _ in slots), 1e-30)
    return max(float(np.abs(b - (d[:, None] * a) * np.conj(d)[None, :]).max()) / scale
               for a, b in slots)


#: Rolls chosen from the **failing** set of the shear-off sweep, for the reason
#: `test_nvf_lattice_translation` gives: a pin that never reaches the defect proves nothing.
#: The full sweep is in `docs/nvf-normal-methods.md`; this is its pin.
BROKEN_ROLLS = ((1, 1), (5, 5), (13, 29), (17, 3), (23, 5), (31, 17), (40, 40))


@pytest.mark.parametrize('shift', BROKEN_ROLLS)
def test_the_circular_limit_removes_the_medial_axis_defect(monkeypatch, shift):
    """With the tie-shear off, `'linear'` is non-covariant on the annulus and `'circular'` is not.

    **Why the shear has to be off.** `_CONTOUR_TIE_SHEAR` (2026-09-01) already settles the
    medial-axis tie the same way at every placement, so with it on both projectors sit on the
    floor and this instrument sees nothing. That is not the projector being unnecessary — it
    is two answers to different halves of one defect. The shear makes the arbitrary answer
    *stable*; this makes it *not be read*. Turning the shear off restores the configuration
    the limit was prototyped against, and is the only way to measure the second claim
    independently of the first.

    Swept over all 4095 rolls with the shear off, `normal_method='contour'`, N=64,
    `truncation_order=3`:

    ===================  ==================  ==================
    scene                linear              circular
    ===================  ==================  ==================
    annulus              9.87e-05, 3172      **1.53e-15, 0**
    disc                 6.52e-05, 2215      **1.93e-15, 0**
    two discs (saddle)   3.01e-04, 3858      2.83e-04, 3858
    ===================  ==================  ==================

    (worst defect, and how many of the 4095 rolls exceed 1e-12. Swept rather than sampled:
    the rolls that break are not the ones a hand-picked set finds — of the eight first
    written here three were clean, which is the sampling error this file's sibling records
    costing it an order of magnitude.)

    The saddle row is the honest one and has its own test below.
    """
    monkeypatch.setattr('wavebit._core.normal_methods._CONTOUR_TIE_SHEAR', 0.0)
    pytest.importorskip('contourpy')

    linear = _translation_defect(ANNULUS, shift, 'contour', 'linear')
    circular = _translation_defect(ANNULUS, shift, 'contour', 'circular')
    assert linear > 1e-8, (
        f'roll {shift} was chosen because the linear projector breaks there and it did not '
        f'({linear:.2e}) — the pin has stopped reaching the defect')
    assert circular < 1e-12, f'roll {shift}: circular moved E by {circular:.2e}'


def test_the_saddle_is_not_fixed_by_the_limit():
    """Stated as a pin rather than left out, because leaving it out would read as fixed.

    A saddle's two nearest boundary points sit on **different** discs. The tie there is a
    discrete jump between two separated directions, not a direction fading out — `|t|` stays
    O(1) through it, `phi` stays near 0, and the limit is never reached. The prototype's
    "~10x" was measured before `_translation_defect` stopped dividing each slot by its own
    norm; on the shared denominator this harness uses, it is a few percent.

    The medial axis and the saddle are two different geometric facts that happen to break the
    same instrument, and only one of them is a vanishing direction.
    """
    import wavebit._core.normal_methods as nm
    old, nm._CONTOUR_TIE_SHEAR = nm._CONTOUR_TIE_SHEAR, 0.0
    try:
        linear = _translation_defect(SADDLE, (32, 0), 'contour', 'linear')
        circular = _translation_defect(SADDLE, (32, 0), 'contour', 'circular')
    finally:
        nm._CONTOUR_TIE_SHEAR = old
    assert linear > 1e-8, 'the saddle roll stopped reaching its defect'
    assert circular > 1e-8, (
        f'the saddle now holds at {circular:.2e} under the circular projector. That would be '
        f'good news and this test is wrong — but check first that the *shear* is not simply '
        f'back on, which fixes it for an unrelated reason.')


@pytest.mark.parametrize('method', ['distance', 'contour', 'spectral', 'smooth'])
def test_the_shipped_configuration_is_covariant_under_both_projectors(method):
    """Nothing regressed: with the shear on, all four constructions hold at the floor.

    `epz_i` is the control the harness leaves out on purpose — it is `inv(<eps>)`, reads no
    normal field at all, and pins the convention at ~5e-15 whatever the projector does. A
    defect that showed up there would be in this harness, not in the field.
    """
    pytest.importorskip('contourpy') if method == 'contour' else None
    for shift in ((13, 29), (2, 2), (31, 17)):
        for projector in ('linear', 'circular'):
            got = _translation_defect(ANNULUS, shift, method, projector)
            assert got < 1e-12, f'{method}/{projector} roll {shift}: E moved by {got:.2e}'
