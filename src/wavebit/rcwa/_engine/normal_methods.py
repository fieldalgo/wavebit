"""The raster normal-field constructions behind ``raster_normal_method`` — all but the original.

:func:`wavebit.rcwa._engine.factorization.normal_vector_field` dispatches here for every method
except ``'distance'``, which stays where it always was. Each construction takes the same
inputs — a concrete permittivity raster, its interface mask, the cell basis, and the shape
the raster had before the ``4*truncation_order+1`` floor lift — and returns ``(normal_x, normal_y)`` float arrays
with ``|n| = 1`` everywhere it emits a direction at all.

Why these exist: the distance-transform gradient quantizes. Central differences of a
distance-to-a-digital-set over a two-cell span land on small integers, so the boundary band
emits directions from the short list ``arctan(p/q)`` — in practice {0, 45} degrees — with a
mean error of ~14 degrees that does not shrink as the raster is refined, because a finer
staircase is the same staircase. All three constructions here break the quantization the
same way the raster-native literature does (Gotz 2008, fmmax, subpixel-smoothed
projection): smooth the indicator *before* differentiating, and keep the boundary's
location from the raw raster. ``docs/theory/rcwa/nvf-normal-methods.md`` records the measurements — the
angular tables, the solver-level A/B against the analytic reference, and the sweeps that
froze each construction's one policy.

Shared conventions, stated once:

- **Everything is eager NumPy/SciPy**, whatever the backend, because the normal field is
  frozen geometry (see :func:`~wavebit.rcwa._engine.factorization.normal_vector_field`): a
  constant handed to the solve, outside the AD graph. Reading the raster's *values* is what
  these constructions add over the mask-only ``'distance'`` path, so a traced layer cannot
  take them — the dispatcher turns that into an error naming ``freeze_geometry()``.
- **Sign is free.** The factorization consumes ``n n^T`` only, so ``n`` and ``-n`` are the
  same field; no construction here spends effort orienting.
- **The base grid sets the smoothing scale.** A layer lifted to the ``4*truncation_order+1`` floor is
  the same staircase spelled with more pixels (that is the whole finding of
  ``docs/validation/rcwa/nvf-upsampling.md``), so every pixel-unit policy below reads ``base_shape`` and
  scales by the lift factor — a 10x10 layer lifted 4x smooths over 4x as many lifted
  pixels, not over a quarter of its base features.
- **Any basis, skewed included.** Square-pixel cells reduce bit-identically to no metric
  at all (the same normalization :func:`normal_vector_field` uses); every other cell works
  in physical coordinates, through the single matrix :func:`_metric` returns. *How* each
  construction carries the metric is worth knowing, because it is not the same story three
  times: ``'contour'`` and ``'spectral'`` never take a difference in index space at all —
  one measures between Cartesian points, the other multiplies by a Cartesian wavevector —
  so a skewed cell costs them nothing and neither has a pullback to apply. ``'smooth'``
  does differentiate a grid, and both of the scipy routines it leans on are per-axis by
  construction, so it changes tools when the metric stops being diagonal
  (:func:`_separable` is the switch). Until 2026-08-21 all three refused a skewed basis and
  ``'distance'`` — the least accurate of the four — was the only field an oblique cell
  could have.
- **Only values-through-geometry are read.** The indicators mark the smaller phase, found
  from the corner (:func:`_two_phase_indicator`), so relabeling the two phases moves
  nothing and neither does translating the raster, and a
  raster with 3+ distinct values — including the continuous/grayscale densities of a
  mid-run topology optimization — delegates to ``'distance'``: a value-built source
  field would break the geometry-only contract the frozen-record staleness tell relies
  on. The record additionally pins the phase *count*, the one value-derived fact the
  mask cannot see. The grayscale case is a stated limit, not an accident: an SSP-style
  filtered-density indicator (arXiv:2503.20189) is the known follow-up.
- **Covariance under a mirror is exact for one of the three.** ``'smooth'`` mirrors
  bit-exactly in both directions, like ``'distance'``. ``'contour'`` and ``'spectral'``
  carry tie-breaks — the nearest polyline sample, the nearest valid band donor — and a
  mirror can break those the other way, worst along x and concentrated on the periodic
  medial ridge where the direction is genuinely ambiguous. ``'contour'``'s is
  chirality-sensitive on purpose (:data:`_CONTOUR_TIE_SHEAR` is a shear, and
  :data:`_CONTOUR_SAMPLE_OFFSET` is measured along the traversal), which is exactly what
  settles the ties a mirror-symmetric rule leaves exact. Measured on an off-center circle
  at N=48, a structure and its x-mirror differ at the solve by ``max|dR|`` of 7.7e-5
  (contour, from 1.4e-4 before 2026-09-01) and 1.6e-4 (spectral), against ~2e-15 for
  ``'distance'``/``'smooth'`` — an order below these methods' accuracy gains at the same
  settings, and the bound a test pins. A workflow that needs exact mirror symmetry
  of the *output* should use ``'smooth'`` or ``'distance'``.
"""
import numpy as np

from wavebit._common.backend import constant_array
from wavebit.lattice import is_axis_aligned
from wavebit.lattice import jacobian


# ------------------------------------------------------------------- shared helpers

def _metric(lattice, shape):
    """The cell's metric as one ``(2, 2)`` matrix: ``(x, y) = M @ (i, j)``.

    ``M`` is :func:`~wavebit.lattice.jacobian` — columns ``a1/n_u`` and ``a2/n_v``, the
    Cartesian displacement of one index step along each axis — divided through by the length
    of its first column. That normalisation is the one
    :func:`~wavebit.rcwa._engine.factorization.normal_vector_field` applies and is there for the same
    reason: a uniform scale cannot survive the unit normalisation at the end, so pinning the
    x-step to 1 costs nothing and makes a square-pixel cell land on exactly the identity,
    bit-identical to taking no metric at all.

    ``M`` is diagonal exactly when the cell is axis-aligned, and every construction below reads
    it as a full matrix — so a skewed cell is the same expression rather than a second path,
    which is the whole reason the three of them port. Where an expression *cannot* be shared it
    is because scipy is on the other end of it and scipy's tools are per-axis; :func:`_separable`
    is that switch, and the two places it sends elsewhere say so. One diagonal approximation
    survives, in :func:`spectral_normal_field`'s bulk copy, and it is documented where it is
    made rather than avoided: it decides which band pixel the *homogeneous interior* copies its
    normal from, and every factorization rule agrees there whatever the normal says.
    """
    if lattice is None:
        return np.eye(2)
    n_v, n_u = shape
    try:
        M = constant_array(jacobian(lattice, n_v, n_u))
    except Exception:                           # noqa: BLE001 - a cell under autodiff
        raise TypeError(_TRACED_CELL) from None
    return M / (float(np.hypot(M[0, 0], M[1, 0])) or 1.0)


#: What a cell under autodiff meets in a construction that reads its metric in NumPy.
_TRACED_CELL = (
    "raster_normal_method='contour', 'spectral' and 'smooth' build the normal field in NumPy "
    "from the cell's metric, so a derivative with respect to the cell (period or lattice) "
    "cannot pass through them. raster_normal_method='distance' carries it, on JAX and PyTorch "
    "alike.")


def _steps(M):
    """``(step_v, step_u)``: the physical length of one index step along each axis.

    The column norms of :func:`_metric`. Two quite different things are read off them and it is
    worth keeping them apart. ``n_u * step_u`` and ``n_v * step_v`` are the cell's two edge
    lengths, which is exact on any basis. The pair *as a metric* is exact only on a diagonal
    ``M``, where it is the whole of it; a caller that hands it to a per-axis consumer on a
    skewed cell is approximating, and says so at the line.
    """
    return float(np.hypot(M[0, 1], M[1, 1])), float(np.hypot(M[0, 0], M[1, 0]))


def _pixel_size(M):
    """Geometric-mean physical pixel size — the square root of the pixel *area*, ``|det M|``.

    ``sqrt(step_v * step_u)`` exactly when the cell is axis-aligned. On a skewed one the
    column-norm product overstates the area by ``1/sin(gamma)`` — a factor of 2 at a
    30-degree cell — and it is the area that a smoothing scale is stated against.
    """
    return float(np.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])))


def _reciprocal(M):
    """``M^-T``: index-space frequencies and index-space gradients, mapped to Cartesian ones.

    Both directions of the same identity. A raster Fourier mode is
    ``exp(2i*pi*(fi*i + fj*j))``; substituting ``(i, j) = M^-1 r`` makes it ``exp(i G.r)`` with
    ``G = 2*pi * M^-T @ (fi, fj)``, and by the same substitution a gradient pulls back as
    ``grad_xy f = M^-T @ (df/di, df/dj)`` — the inverse transpose, not the matrix: index steps
    push forward, gradients pull back. It is the correction
    :func:`~wavebit.rcwa._engine.factorization.normal_vector_field` applies on the ``'distance'``
    path, spelled once here for the three that also need it.
    """
    return np.linalg.inv(M).T


def _separable(M):
    """Whether the metric is diagonal — the case scipy's per-axis tools express exactly.

    A diagonal ``M`` is a per-axis rescaling, so a physically isotropic Gaussian stays the
    product of two 1-D Gaussians and a physical distance stays a per-axis-weighted index
    distance: ``gaussian_filter``'s ``sigma`` and ``distance_transform_edt``'s ``sampling``
    can each be handed the whole metric. Off the diagonal neither can — scipy has no way to
    be told the two axes are not perpendicular — and the two constructions that use them take
    their other branch.

    This is :func:`~wavebit.lattice.is_axis_aligned` asked of the metric rather than of
    the basis, which is the same question one column scaling later. A tolerance rather than an
    exact zero, deliberately: a square cell spelled ``a2 = P * (cos(pi/2), sin(pi/2))`` carries
    a 1e-14 off-diagonal nobody meant, and sending it down the skewed branch would move an
    answer that has to be bit-identical.
    """
    return is_axis_aligned(M)


def _cartesian_frequencies(M, shape):
    """``(gx, gy)``: the Cartesian spatial frequency of every FFT bin, cycles per unit length.

    ``G/(2*pi) = M^-T @ (fi, fj)`` (:func:`_reciprocal`), evaluated on the transform's own
    index-space frequency grid. On an axis-aligned cell it is ``(fi, fj/step_v)``.
    """
    n_v, n_u = shape
    B = _reciprocal(M)
    fi = np.fft.fftfreq(n_u)[None, :]
    fj = np.fft.fftfreq(n_v)[:, None]
    return B[0, 0] * fi + B[0, 1] * fj, B[1, 0] * fi + B[1, 1] * fj


def _isotropic_blur(field, sigma, M):
    """Periodic Gaussian blur of ``field`` by a **physically** isotropic ``sigma``.

    Isotropy is the point. The smoothing scale these constructions freeze is a length in the
    cell, not a count of pixels, and a kernel that is round in index space is an ellipse in
    the cell whenever the pixels are not square. On an axis-aligned cell the round physical
    kernel is still separable — one sigma per axis, in that axis's own step — which is what
    ``scipy.ndimage.gaussian_filter`` can express and what this has always done.

    On a **skewed** cell it is not separable at all: a circle pulled back through ``M^-1`` is
    a tilted ellipse, and a separable filter has no way to tilt. So that branch does the same
    convolution in the frequency domain, where isotropy is one expression —
    ``exp(-sigma^2 |G|^2 / 2)`` with ``G`` the mode's *Cartesian* wavevector
    (:func:`_reciprocal`) — and where periodicity is exact rather than a boundary mode.

    The two branches are the same kernel and are not bit-identical: ``gaussian_filter``
    truncates its kernel at 4 sigma and the transform does not. The axis-aligned branch is
    therefore kept for the axis-aligned case rather than folded into the general one, so that
    every measurement in ``docs/theory/rcwa/nvf-normal-methods.md`` — including the sweep that froze
    ``_CONTOUR_SIGMA_COEF`` — still describes the code that runs there.
    """
    if _separable(M):
        from scipy.ndimage import gaussian_filter
        step_v, step_u = _steps(M)
        return gaussian_filter(field, sigma=(sigma / step_v, sigma / step_u), mode='wrap')
    gx, gy = _cartesian_frequencies(M, field.shape)
    H = np.exp(-0.5 * (2.0 * np.pi * sigma) ** 2 * (gx ** 2 + gy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(field) * H))


#: A blurred gradient this far below the largest one in the cell is round-off, not a
#: direction -- see the snap at the end of :func:`_isotropic_blur_gradient`. Twelve orders
#: sits far above the ~1e-16 the transform actually leaves and far below any gradient a
#: geometry produces, so no shape has to be consulted to choose it.
_GRADIENT_ROUNDOFF = 1e-12


def _isotropic_blur_gradient(field, sigma, M):
    """Cartesian gradient of ``field`` blurred by a physically isotropic ``sigma``.

    The derivative-of-Gaussian :func:`smoothed_distance_normal_field` reads, with the same two
    branches and the same reason as :func:`_isotropic_blur`. The separable branch
    differentiates per index axis and pulls the pair back through ``M^-T``; the
    frequency-domain branch multiplies by ``i*G`` and is Cartesian already, with no pullback
    to apply.
    """
    if _separable(M):
        from scipy.ndimage import gaussian_filter
        step_v, step_u = _steps(M)
        sig = (sigma / step_v, sigma / step_u)
        g_i = gaussian_filter(field, sig, order=(0, 1), mode='wrap')
        g_j = gaussian_filter(field, sig, order=(1, 0), mode='wrap')
        B = _reciprocal(M)
        return B[0, 0] * g_i + B[0, 1] * g_j, B[1, 0] * g_i + B[1, 1] * g_j
    gxf, gyf = _cartesian_frequencies(M, field.shape)
    C = np.fft.fft2(field) * np.exp(-0.5 * (2.0 * np.pi * sigma) ** 2 * (gxf ** 2 + gyf ** 2))
    gx = np.real(np.fft.ifft2(2j * np.pi * gxf * C))
    gy = np.real(np.fft.ifft2(2j * np.pi * gyf * C))
    # Snap the round-off on the skeleton, where the separable branch returns an *exact* zero.
    # It gets one because a symmetric kernel over a locally symmetric field cancels term by
    # term in the same floating-point order; a transform has no such pairwise structure and
    # leaves ~1e-16 there instead. That matters out of all proportion to its size: the caller
    # normalises, so 1e-16 becomes a full unit vector pointing wherever the round-off fell, and
    # a lattice translation of the raster settles it differently -- the field stops being
    # covariant at a handful of pixels (4 of 256 on a rod in a hex cell, 90 degrees apart)
    # where the direction is not merely uncertain but undefined, the two nearest interfaces
    # being equidistant. An exact zero is what the factorization reads as "no normal here,
    # collapse to the direct rule", which is the right answer on a skeleton.
    mag = np.hypot(gx, gy)
    roundoff = mag <= _GRADIENT_ROUNDOFF * float(mag.max())
    return np.where(roundoff, 0.0, gx), np.where(roundoff, 0.0, gy)


def _periodic_signed_distance(inside, M):
    """Signed physical distance to the nearest opposite-phase pixel, on the torus.

    Positive outside the ``inside`` set and negative within it, measured to pixel *centres* —
    the quantity ``distance_transform_edt`` computes, in the cell's own metric.

    Two branches, one quantity. A diagonal metric is a per-axis rescaling, which is exactly
    what scipy's ``sampling`` expresses, so there the transform answers it directly on a 3x3
    tiling — the tiling being what makes it periodic, since a pixel's nearest interface may
    sit in the next copy. A skewed metric is not expressible that way at all, so that branch
    asks a kd-tree for the same nearest-neighbour distance in physical coordinates. It costs
    about 4x the transform (0.73 s against 0.16 s at N=512, both trees and both queries) and
    is exact rather than close.

    Handing scipy the column norms and calling it done is the tempting shortcut, and it is
    measurably wrong: paired with the separable Gaussian it leaves this construction between
    2.1 and 11.8 degrees on a skewed disc where the exact pair sits at 0.07-0.13, worst on the
    *small* features, where the kernel is widest relative to the shape
    (``docs/theory/rcwa/nvf-normal-methods.md``).
    """
    n_v, n_u = inside.shape
    if _separable(M):
        from scipy.ndimage import distance_transform_edt
        step_v, step_u = _steps(M)
        sampling = None if step_v == step_u == 1.0 else (step_v, step_u)
        t = np.tile(inside, (3, 3))
        sd = (distance_transform_edt(~t, sampling=sampling)
              - distance_transform_edt(t, sampling=sampling))
        return sd[n_v:2 * n_v, n_u:2 * n_u]

    from scipy.spatial import cKDTree
    Ii, Jj = np.meshgrid(np.arange(n_u, dtype=float), np.arange(n_v, dtype=float))
    P = np.column_stack([Ii.ravel(), Jj.ravel()]) @ M.T

    def nearest(phase):
        jj, ii = np.nonzero(phase)
        tiled = np.vstack([np.column_stack([ii + dx * n_u, jj + dy * n_v]).astype(float)
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        return cKDTree(tiled @ M.T).query(P, workers=-1)[0].reshape(n_v, n_u)

    # Each phase measures to the other one; a pixel's distance to its OWN phase is zero and is
    # what the sign selects away, exactly as the pair of transforms above does.
    return np.where(inside, -nearest(~inside), nearest(inside))


def _repeats(shape, base_shape):
    """Lift factors ``(repeat_v, repeat_u)`` of the floor-lifted grid over its base."""
    if base_shape is None:
        return 1.0, 1.0
    return shape[0] / base_shape[0], shape[1] / base_shape[1]


def _constant_along_x(shape):
    return np.ones(shape, dtype=float), np.zeros(shape, dtype=float)


def _two_phase_indicator(eps):
    """Minority-phase indicator, canonical up to complementation at half fill.

    Which phase is *marked* is read from the corner pixel and then oriented by area, and both
    halves are load-bearing.

    **The corner, not** ``np.unique(eps)[0]``: swapping the two material values exchanges which
    value ``unique`` sorts first and inverts a value-sorted indicator, but it leaves the
    corner's *region* where it was — so this indicator is invariant under relabeling the
    phases, which is what keeps the whole construction a function of the geometry alone (the
    contract :func:`~wavebit.rcwa._engine.factorization.normal_vector_field` states, and the
    frozen-record staleness tell relies on).

    **Then the smaller area**, since 2026-08-29, because the corner alone is not invariant
    under a *translation*. A raster rolled is the same structure written from a different
    corner, and a roll can put that corner inside the feature — at which point the corner rule
    inverts the indicator wholesale. Three of the four constructions never noticed: an
    inversion flips their field globally and NVF consumes only ``n n^T``. ``'contour'`` did.
    The 0.5 level set of a field and of its complement is the same curve, but contourpy
    traverses it the other way, and the sampling has a direction in it — samples sit
    :data:`_CONTOUR_SAMPLE_OFFSET` of the way *along* each sub-interval — so the sampled
    field is not the same field. Measured on the pinned disc (radius 30 in a 64 cell), ten
    rolls came back as *exactly two* values — 0.328909134249 for the six whose corner lands outside and
    0.329294462911 for the four whose corner lands inside, each class tight to 5e-13, nothing
    in between. Orienting by area collapses all ten to 4.157e-13.

    It is not a large-feature problem, which is how it was first described: a disc of radius
    12 in the same cell breaks on exactly the one roll of ten that moves its centre within 12
    pixels of the corner. Any feature does, given the placement.

    The two readings converge, so this picks between them rather than correcting one: the same
    disc at 64 / 128 / 256 samples a side has the branches 3.85e-4, 2.51e-5 and 6.73e-7 apart.
    What moves is ``'contour'`` on a raster whose feature covers more than half the cell, which
    is where the orientation now differs from the corner's; the other three constructions are
    bit-identical (0.00e+00 on both discs tried), as is ``'contour'`` below half.

    At exactly half fill the indicator itself still follows the corner, so a
    translation may complement it. The consuming fields must therefore be
    invariant to that complement. The contour path now canonicalizes traversal
    geometrically before offset resampling: closed loops by signed area and
    winding curves by winding direction. This removes the former ~2e-4
    transmission step on an asymmetric 2048-of-4096-pixel blob without reading
    material values or weakening the geometry-only frozen-record contract.
    See ``test_half_fill_contour_translation_is_invariant``. The other methods
    already ignore a global indicator inversion in their normal projector.
    """
    indicator = (eps != eps.flat[0]).astype(float)
    return 1.0 - indicator if indicator.mean() > 0.5 else indicator


# ==================================================================== 'contour'

# FROZEN 2026-08-16 from the angular sweep (docs/theory/rcwa/nvf-normal-methods.md): sigma follows the
# sqrt(N) family because every fixed-in-px sigma leaves the error FLAT in N (a fixed
# kernel keeps a fixed fraction of the staircase wiggle) while every sqrt(N) policy
# converges; the coefficient is the smallest of the top group, tie-broken by the measured
# thin-feature erasure threshold (~1.4 sigma px), not tuned per case. D_SWITCH and
# RESAMPLE_H were measured inert (< 0.3 deg across 4x changes).
_CONTOUR_SIGMA_COEF = 0.25    # sigma = 0.25 * sqrt(N) px, N the geometric-mean base size
_CONTOUR_D_SWITCH = 0.5       # below this distance (px) use the local segment normal
_CONTOUR_RESAMPLE_H = 0.25    # polyline resampling step (px)

#: Where in each sub-interval the sample sits, and **it is the open interval (0, 1/2): 0 is
#: measurably wrong and 1/2 is measurably worse than the inside.** 14 scenes on square and
#: skewed cells, 63 lattice translations each, worst over the translation identity, against a
#: floor of 2.9e-15:
#:
#: * ``0.0`` puts a sample on every polyline *vertex*, so the sample set inherits the level
#:   set's own symmetries wholesale and the ties `_CONTOUR_TIE_SHEAR` settles stop being
#:   separable. **Seven of the fourteen break**: a small disc 8.8e-05 on 63 of 63 rolls, the
#:   saddle 5.4e-05 on 56, an annulus 5.2e-05 on 63, a cross 3.1e-06 on 63. It also lands a
#:   sample exactly ``_CONTOUR_D_SWITCH`` from a pixel centre wherever the interface runs
#:   along an axis -- the blurred indicator is antisymmetric about a straight interface, so
#:   its 0.5 crossing sits exactly on the half-integer and the pixel beside it is exactly half
#:   a pixel away -- and that pixel flips between the two branches of the direction rule on
#:   the last bit (5.1e-08 on 16 of 63, the smallest defect in this file and still one).
#: * ``0.5`` is the sub-interval midpoint, and the midpoint set is carried to itself by
#:   *reversing* the traversal, so a curve and its mirror image get the same samples. Nothing
#:   breaks -- but the percolating scene on a 45-degree cell sits at **4.1e-13**, 200x the
#:   floor every other scene holds at and within 2.5x of what the pins assert. That is a
#:   margin, not a failure, and it is why this is stated as a measurement rather than a proof.
#:
#: A quarter of the way in is neither. 0.05, 0.1, 0.2, 0.25, 0.3 and 0.4 all land on the
#: floor, so the value inside the interval is free.
_CONTOUR_SAMPLE_OFFSET = 0.25

#: A deterministic infinitesimal perturbation of the metric the nearest-*sample* query is
#: answered in -- the standard way to make a geometric predicate with exact ties single
#: valued (Edelsbrunner & Muecke, "Simulation of Simplicity", ACM TOG 9, 66 (1990)).
#:
#: **What it is for.** On the medial axis two points of the level set are exactly equidistant
#: from a pixel and the direction is genuinely undefined there. That is not the problem. The
#: problem is that *which* of them the kd-tree returns is then settled by the last bit, and a
#: raster roll settles it the other way, so the field stops being covariant at those pixels.
#: The ties are exact rather than near, because a level set drawn on a raster inherits the
#: raster's symmetries. Without this, over 63 translations each: a cross 1.8e-03 on 63 rolls,
#: a percolating post-and-arm 1.2e-03 on 56, a small disc 4.2e-04 on 57, a saddle between two
#: discs 2.9e-04 on 8, an annulus 9.9e-05 on 48 -- seven of fourteen scenes, and the annulus
#: and the saddle are the two that ``tests/rcwa/test_nvf_lattice_translation.py`` carried as strict
#: xfails for the medial axis from 2026-08-29 to 2026-09-01.
#:
#: It does **not** make the direction on the medial axis defined -- nothing here can, and the
#: circular-polarization limit that would is still the open item (see
#: ``tests/rcwa/test_nvf_lattice_translation.py``). It makes the arbitrary choice the *same*
#: arbitrary choice at every placement, which is the whole of what covariance asks.
#:
#: **It chooses which sample, never where it is**: every distance, foot and direction the
#: construction returns is computed in the true metric afterwards, so the only thing it can
#: change is a tie, or an ordering closer together than ``e`` relative, where either answer
#: is the same answer to that many digits. The working window is enormous, which is what says
#: this is not a tuned knob: every value from 1e-13 to 1e-3 gives an identical clean sweep,
#: six orders either side of this one, against the ~1e-15 the tie sits inside.
#:
#: The shear carries an off-diagonal term *and* an unequal diagonal, which separate a pair
#: mirrored in an axis and a pair mirrored in a diagonal respectively. Measured, either half
#: alone already cleans all fourteen scenes, so the second is insurance against a mirror
#: class this scene set does not contain rather than a demonstrated need.
_CONTOUR_TIE_SHEAR = 1e-9


#: The two copies of a seam carry bit-identical samples, so a level set cut there comes back
#: with its two cut ends at the same interpolated coordinate to the last bit. The tolerance is
#: for a contourpy that walks the two edges in a different order, not for bridging a gap.
_SEAM_TOL = 1e-9


def _contour_polylines(smoothed):
    """0.5 level set of one period, closed **on the torus**.

    Returns ``(points, winding)`` pairs in index coordinates (x = column, y = row). Every
    polyline repeats its first vertex at the end, displaced by ``winding`` whole cells:
    ``points[-1] == points[0] + (wu * n_u, wv * n_v)``. A ``winding`` of ``(0, 0)`` is an
    ordinary closed loop; anything else is a level set that **percolates** — it leaves the
    cell and comes back, so it closes only on the torus, and its tiled copies chain end to
    end into one curve of infinite extent. ``winding is None`` is the degenerate fallback
    below, and is the only case that carries genuine curve ends.

    **The window is one period with the seam duplicated, not a 3x3 tile.** contourpy has no
    periodic domain: it is handed a finite rectangle and cuts whatever leaves it. Padding
    3x3 with ``mode='wrap'`` moves the cut a period away, which is enough for a *contractible*
    loop — some copy of it lands whole inside the window — and does nothing at all for a
    percolating one, which is cut wherever the window ends however wide it is. Padding
    instead by a single duplicated row and column covers each torus cell exactly once, and
    then the cut has an exact partner: the level set leaving through column ``n_u`` re-enters
    at column ``0`` at a coordinate interpolated from the same two samples, bit for bit.

    **So the cut ends are stitched back together.** :func:`_stitch_across_the_seam` matches
    each fragment's tail to the head sitting at the same point of the torus and walks the
    cycle, accumulating the whole-cell offsets it crosses; that sum is the winding. A
    contractible loop straddling the seam comes back as one loop with winding ``(0, 0)`` —
    the fragments reassemble — and a percolating one as a single arc whose ends are one
    lattice vector apart. Neither is a fragment any more, so ``_assemble`` never sees an
    endpoint that is a cut mark rather than a curve end.

    Until 2026-09-01 open curves were returned exactly as contourpy cut them, and only when
    *no* closed loop had survived — one disc beside a percolating stripe deleted the stripe's
    boundary from the field outright, and the fragment set moved with the placement (5 / 6 /
    9 / 12 fragments across rolls of one scene), which is what made ``'contour'`` the one
    construction that is not covariant under a lattice translation on a percolating level
    set. That regime is ordinary rather than exotic: on a 20-degree cell the lattice rows sit
    0.34 of a period apart, so any feature wider than that percolates, and the parametric path
    refuses such a shape entirely.

    Each polyline is then folded by whole periods so its centroid lands in the cell. The
    tiles below are laid around the representative, so a curve left sitting a period out
    would put the cell at the edge of the tiled region instead of its middle — and a
    reassembled loop can sit there, because the chain starts wherever the seam cut it. On a
    loop already inside the window the fold is the identity.

    **The fallback.** A fragment whose partner is missing or ambiguous — a level set touching
    a corner of the seam, where the wrap is two-dimensional and the match is not unique —
    is returned as a genuinely open polyline rather than dropped. That is the old behaviour
    for the one case that still needs it, minus the discard.
    """
    try:
        import contourpy
    except ImportError:
        raise ImportError(
            "raster_normal_method='contour' extracts subpixel level sets with contourpy, which "
            "is not installed. `pip install contourpy` (it ships with matplotlib), or use "
            "raster_normal_method='spectral', which needs nothing beyond NumPy/SciPy.") from None
    n_v, n_u = smoothed.shape
    padded = np.pad(smoothed, ((0, 1), (0, 1)), mode='wrap')
    gen = contourpy.contour_generator(z=padded, line_type=contourpy.LineType.Separate)
    loops, cuts = [], []
    for pts in gen.lines(0.5):
        pts = np.asarray(pts, dtype=float)
        if pts.shape[0] < 2:
            continue
        if np.allclose(pts[0], pts[-1]):
            if pts.shape[0] < 4:            # fewer than three distinct vertices
                continue
            # Close it on its own first vertex rather than on contourpy's repeat of it, so
            # that `points[-1] == points[0] + winding` holds exactly for every polyline here
            # and `_resample` needs no second spelling for the loop case.
            loops.append((np.vstack([pts[:-1], pts[:1]]), (0, 0)))
        else:
            cuts.append(pts)
    out = []
    for pts, winding in loops + _stitch_across_the_seam(cuts, n_v, n_u):
        centre = pts[:-1].mean(0) if winding is not None else pts.mean(0)
        # `floor(c/n)` and not `round(c/n)`, for the tie rather than the general case:
        # `round` is half-to-even, so a centre at exactly +-n/2 stays where it is and lands
        # the representative a full period outside the cell it describes. What reaches that
        # tie is a shape written about the *pixel index* n//2 -- `(U - N//2)**2 +
        # (V - N//2)**2 < r**2`, whose centroid is exactly n/2 -- and not a shape centred in
        # the cell, whose centroid is (n-1)/2 and misses it. (An earlier spelling here was
        # `floor((c - n/2)/n + 0.5)`, which is *identically* `floor(c/n)` -- checked over
        # 20001 samples per n for n in 2, 3, 32, 64, 1024, 4097, zero differences. It was
        # written as if the `+0.5` were doing the work. It is not.)
        out.append((pts - np.array([np.floor(centre[0] / n_u) * n_u,
                                    np.floor(centre[1] / n_v) * n_v]), winding))
    return out


def _stitch_across_the_seam(cuts, n_v, n_u):
    """Close the fragments a seam cut, on the torus, and read off each cycle's winding.

    Every fragment starts and ends on the window border, and the border is the wrap: a tail
    at column ``n_u`` is the same point of the torus as a head at column ``0``. Matching tails
    to heads therefore gives each fragment exactly one successor, and following successors
    walks a cycle that is one whole component of the level set. Summing the whole-cell jumps
    taken along the way gives the winding — ``(0, 0)`` when the component is a loop that
    merely straddles the seam, and non-zero when it genuinely percolates.

    Matching is by position and not by "the opposite edge", which is what makes it work on a
    skewed cell: the raster's periodicity is an index shift of a whole cell whatever the
    lattice does with those indices, so the partner of a border point is a whole number of
    columns or rows away in *index* space, and the metric never enters. The jump is snapped
    to that whole number, so the winding is an exact integer pair and the reassembled
    polyline sits exactly on the lattice.

    Fragments left unmatched — an ambiguous match at a seam corner, or an odd component
    contourpy split some other way — come back with ``winding is None``, meaning a polyline
    with real ends. They are returned rather than discarded.
    """
    if not cuts:
        return []
    period = np.array([float(n_u), float(n_v)])
    heads = np.array([c[0] for c in cuts])
    tails = np.array([c[-1] for c in cuts])
    def on_torus(p):
        # Fold the border coordinate n -> 0, so the two copies of a seam share one key.
        return np.where(np.abs(p - period) <= _SEAM_TOL, 0.0, p)

    head_keys, tail_keys = on_torus(heads), on_torus(tails)

    successor = np.full(len(cuts), -1, dtype=int)
    for i, t in enumerate(tail_keys):
        hit = np.nonzero(np.abs(head_keys - t).max(axis=1) <= _SEAM_TOL)[0]
        if hit.size == 1:
            successor[i] = int(hit[0])

    out, consumed = [], set()
    for i0 in range(len(cuts)):
        if i0 in consumed or successor[i0] < 0:
            continue
        walk, seen, i = [], set(), i0
        while i >= 0 and i not in seen and i not in consumed:
            seen.add(i)
            walk.append(i)
            i = int(successor[i])
        if i != i0:                     # ran into another chain, or into a dead end
            continue
        consumed.update(walk)
        pieces, off = [cuts[i0]], np.zeros(2)
        for a, b in zip(walk, walk[1:] + [i0]):
            # Where fragment b has to sit for its head to land on fragment a's tail. The two
            # differ by a whole number of cells; snapping to it keeps the joint on the lattice.
            off = np.round(((cuts[a][-1] + off) - cuts[b][0]) / period) * period
            if b != i0:
                pieces.append(cuts[b][1:] + off)     # its head repeats the previous tail
        # The last fragment's tail *is* the first fragment's head one winding on, so the
        # chain already ends where it has to; `off` is what that winding came to.
        out.append((np.vstack(pieces), tuple(int(w) for w in np.round(off / period))))

    out.extend((cuts[i], None) for i in range(len(cuts)) if i not in consumed)
    return out


#: `ceil` sits exactly on its own jump for the commonest segment there is, so the count needs
#: a round-off guard. A level set running straight along an axis crosses the grid lines a whole
#: pixel apart, so its segments are *exactly* 1.0 long and `1.0 / 0.25` is *exactly* 4 -- while
#: contourpy's vertices are covariant under a raster roll only to about 1e-15, because it
#: interpolates `j + t` from a different integer `j`. Without the guard that last bit decides
#: between 4 sub-intervals and 5, one segment's samples move wholesale, and the axis-aligned
#: scenes stop being covariant: the percolating post-and-arm 6.4e-04 on 56 of 63 rolls and
#: 1.1e-04 on 44 at 45 degrees, a cross 5.3e-04 on 56. The ratio never exceeds ~6 on a
#: marching-squares segment, so an absolute slack this size can only ever absorb round-off.
_SUBDIVISION_SLACK = 1e-9


def _resample(pts, h, periodic):
    """Subdivide every segment into equal parts no longer than ``h``; ``None`` if degenerate.

    **Per segment, and never along the whole curve, because the phase is the defect.** The
    obvious spelling walks arc length from the first vertex and drops a sample every ``h`` —
    and then *where the curve starts* sets where every sample lands. A closed loop has no
    intrinsic start; contourpy's is the first crossing its raster scan meets, which is a
    geometric point of the loop and so moves with a lattice translation exactly as the loop
    does, and that is the only reason arc-length resampling was ever covariant here. It stops
    being true the moment a curve has to be reassembled from pieces: the reassembly starts
    wherever the seam cut it, the cut slides with the placement, and the whole sample set
    slides a fraction of ``h`` with it. Measured by putting the arc-length spelling back into
    the finished construction: **8 of 14 scenes**, worst 7.0e-03 on a circle overlapping its
    own images, 1.2e-03 on a disc merely large enough to straddle the seam — three orders past
    anything the construction's own accuracy justifies.

    Subdividing each segment into ``ceil(len/h)`` equal parts has no phase to get wrong. Every
    sample is a fixed fraction of one segment, the segments come from marching squares on the
    raster, and a whole-pixel roll translates those exactly — so the sample set translates
    exactly, whatever the curve was cut into or which end of it was walked first. It costs
    about one extra sample per vertex (753 -> 900 on a radius-30 disc in a 64 cell, whose
    level set has 236 of them), keeps the same guarantee the step ``h`` is there for — no gap
    wider than ``h``, so a pixel's nearest sample always sits on a segment adjacent to its
    nearest point — and it is what lets ``periodic`` mean only "the last vertex is the first
    one a winding away, so do not emit it", rather than naming a sampling rule.

    Where in the sub-interval the sample sits is :data:`_CONTOUR_SAMPLE_OFFSET`, and that it
    is neither end of it is measured there.
    """
    seg = np.diff(pts, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    live = seglen > 0.0
    if not live.any():
        return None
    seg, seglen, starts = seg[live], seglen[live], pts[:-1][live]
    counts = np.maximum(np.ceil(seglen / h - _SUBDIVISION_SLACK).astype(int), 1)
    owner = np.repeat(np.arange(counts.size), counts)
    step = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    out = (starts[owner]
           + ((step + _CONTOUR_SAMPLE_OFFSET) / counts[owner])[:, None] * seg[owner])
    return out if periodic else np.vstack([out, pts[-1:]])


def _orient_half_fill_contours(polylines):
    """Choose traversal from the curves, independently of which phase was marked.

    Complementing a half-filled indicator reverses contourpy's traversal. An
    off-center resampling phase then selects different points on the same curve.
    Orient contractible loops by signed area and winding curves by their first
    nonzero winding component. Both rules survive a translation and material
    relabeling. Reverse the winding along with the vertices so tiled adjacency
    still follows the same geometric curve.
    """
    oriented = []
    for pts, winding in polylines:
        if winding == (0, 0):
            local = pts - pts[0]
            area2 = np.sum(local[:-1, 0] * local[1:, 1]
                           - local[:-1, 1] * local[1:, 0])
            reverse = area2 < 0
        else:
            direction = pts[-1] - pts[0] if winding is None else winding
            first = next((v for v in direction if v != 0), 0)
            reverse = first < 0
        if reverse:
            pts = pts[::-1]
            winding = None if winding is None else tuple(-v for v in winding)
        oriented.append((pts, winding))
    return oriented


def _assemble(polylines, h):
    """Concatenate resampled polylines into ``(points, prev, nxt, links)`` adjacency arrays.

    Adjacency is cyclic within a closed loop and sequential otherwise. ``links`` carries the
    joins a single tile cannot express: ``(first, last, wu, wv)`` for each percolating curve,
    saying that its last sample is followed by its first sample **in the tile ``(wu, wv)``
    cells away**. :func:`contour_normal_field` patches those in once the tiles exist; without
    them a percolating curve would have a spurious end at every tile boundary.

    Open ends — the degenerate fallback only — point at themselves, so projection degenerates
    to the endpoint, which is correct at a real curve end.
    """
    all_pts, all_prev, all_next, links = [], [], [], []
    base = 0
    for pts, winding in polylines:
        rs = _resample(pts, h, winding is not None)
        if rs is None:
            continue
        n = rs.shape[0]
        idx = np.arange(n)
        if winding == (0, 0):
            prev, nxt = base + (idx - 1) % n, base + (idx + 1) % n
        else:
            prev, nxt = base + np.maximum(idx - 1, 0), base + np.minimum(idx + 1, n - 1)
            if winding is not None:
                links.append((base, base + n - 1, winding[0], winding[1]))
        all_pts.append(rs)
        all_prev.append(prev)
        all_next.append(nxt)
        base += n
    if not all_pts:
        return None
    return np.vstack(all_pts), np.concatenate(all_prev), np.concatenate(all_next), links


def _project(P, A, B):
    """Foot of ``P`` on segment ``A -> B`` (clamped), squared distance, segment vector."""
    AB = B - A
    denom = (AB * AB).sum(1)
    safe = np.where(denom > 0.0, denom, 1.0)
    t = np.clip(((P - A) * AB).sum(1) / safe, 0.0, 1.0)
    t = np.where(denom > 0.0, t, 0.0)
    Q = A + t[:, None] * AB
    return Q, ((P - Q) ** 2).sum(1), AB


def _tile_rings(M, points, n_v, n_u):
    """How many rings of periodic images the nearest-sample query has to be given.

    **It has never once been measured to be needed, and saying so is the point.** With the
    fold in :func:`_contour_polylines` landing the representative on the cell rather than on
    the origin, a single ring misses nothing: 0 of 4096 pixels at 90 / 60 / 45 / 20 degrees, on
    a disc and on a percolating stripe, at five placements each
    (``test_one_ring_reaches_the_nearest_point_once_the_fold_is_right``). The covering-radius
    argument this function computes was standing in for that fold bug.

    **It used to be kept for a second reason and that reason is gone.** With a single ring the
    percolating stripe was non-periodic at 45 and 20 degrees -- 61 and 62 of 63 rolls -- and
    *not* because a nearest point was missed: open curves got no fold and no dedup, so the
    padding window cut them into a placement-dependent set of fragments, and a wider ring
    masked it. :func:`_contour_polylines` stitches them on the torus since 2026-09-01, and
    forcing this to 1 now leaves all fourteen scenes of the translation sweep clean.

    So what is left is the covering-radius guarantee alone, and this stays because that
    guarantee is about a cell this sweep does not contain -- the shortest lattice vector runs
    away from the covering radius as the cell flattens, whatever the level set does. The cost
    is quadratic and worth stating: 9 tiles at 1, 25 at 2, 81 at 4, 729 at the 13 it asks for
    at 5 degrees.

    The two terms:

    * the query needs the covering radius, not the cell -- 0.707 on a square cell,
      0.866 at 60 degrees, 1.21 at 45, 2.84 at 20.
    * how far the samples stray from the cell. Closed loops are folded onto it now,
      so they stray 0.23 to 0.38 cells; open curves are not folded at all.
    """

    cell = M @ np.diag([float(n_u), float(n_v)])          # fractional cell coords -> Cartesian
    a1, a2 = cell[:, 0], cell[:, 1]
    shortest = min(np.hypot(*a1), np.hypot(*a2),
                   np.hypot(*(a1 - a2)), np.hypot(*(a1 + a2)))
    covering = 0.5 * max(np.hypot(*(a1 + a2)), np.hypot(*(a1 - a2)))
    frac = np.linalg.solve(cell, points.T).T              # samples in fractional cell coords
    stray = max(0.0, float(np.max(frac) - 1.0), float(-np.min(frac)))
    return int(np.ceil(stray)) + int(np.ceil(covering / shortest))


def contour_normal_field(eps, interface_mask, lattice=None, base_shape=None):
    """Subpixel level-set contour, then the exact direction of grad(distance-to-curve).

    The construction: smooth the two-phase indicator with a periodic Gaussian, extract its
    0.5 level set as subpixel polylines closed on the torus (contourpy, then
    :func:`_stitch_across_the_seam`), resample them finely, and give every pixel the
    direction to its exact nearest point on the curve — which *is* the gradient of the
    distance-to-curve field, i.e. the exact version of the extension the ``'distance'``
    construction approximates on the integer grid. Below half a pixel the point-difference
    direction is noisy at polyline vertices, so the local segment normal stands in (the two
    coincide in segment interiors).

    Two rules in it exist only to keep the answer the same under a lattice translation of the
    raster, and both are stated where they are set: samples sit
    :data:`_CONTOUR_SAMPLE_OFFSET` of the way along each sub-interval rather than at its end,
    and the nearest-sample query is answered in a metric sheared by
    :data:`_CONTOUR_TIE_SHEAR` so that the exact ties a raster's symmetries produce on the
    medial axis are settled the same way at every placement.

    This is the measured winner: it won every resolved cell in the solver A/B on both
    scenes and lost none, and it converges with N where ``'distance'`` is flat
    (docs/theory/rcwa/nvf-normal-methods.md).

    Emits no zeros — on the medial axis the nearest point is non-unique but the direction
    is still unit, merely discontinuous, exactly like a distance gradient one pixel off
    the ridge.

    A raster with 3+ distinct values delegates to ``'distance'``: the single indicator
    merges the non-background phases, so an interface *between* two inclusions would draw
    no contour at all. Thin features are the other stated limit: a strip narrower than
    ~1.4 sigma px loses its 0.5 crossing and vanishes from the *field* (the permittivity
    itself is untouched); such pixels get their nearest surviving contour's direction.

    **A skewed cell costs this construction nothing**, which is what makes it the one to
    reach for there as well. Everything metric in it happens between Cartesian points — the
    level set's own coordinates and the pixel centres, both carried through :func:`_metric`
    — and the direction to the nearest point of a curve is a Cartesian normal however the
    sample grid was sheared. There is no index-space difference anywhere in it, so there is
    no pullback to get wrong. The basis reaches one other thing, the Gaussian that locates
    the level set, and that stays physically isotropic (:func:`_isotropic_blur`).
    """
    shape = eps.shape
    values = np.unique(eps)
    if values.size > 2:
        from wavebit.rcwa._engine.factorization import normal_vector_field
        return normal_vector_field(eps, interface_mask, lattice)
    if values.size < 2 or not interface_mask.any():
        return _constant_along_x(shape)

    M = _metric(lattice, shape)
    repeat_v, repeat_u = _repeats(shape, base_shape)
    base_shape = base_shape or shape

    # The frozen sigma, stated on the base grid and carried to the lifted one: physically
    # isotropic, 0.25 * sqrt(N_base) base pixels wide with N_base the geometric-mean base
    # size. In physical units that is 0.25 / sqrt(N_base) of the period, -> 0 with real
    # resolution, which is what makes the construction converge under refinement.
    sigma_base = _CONTOUR_SIGMA_COEF * float(np.sqrt(np.sqrt(
        base_shape[0] * base_shape[1])))
    px = _pixel_size(M)
    sigma_phys = sigma_base * float(np.sqrt(repeat_v * repeat_u)) * px
    # The inert switch length stays in geometric-mean physical pixels; the resample step
    # is an index-space quantity and needs no conversion.
    d_switch = _CONTOUR_D_SWITCH * px

    from scipy.spatial import cKDTree

    indicator = _two_phase_indicator(eps)
    smoothed = _isotropic_blur(indicator, sigma_phys, M)
    polylines = _contour_polylines(smoothed)
    if np.count_nonzero(indicator) * 2 == indicator.size:
        polylines = _orient_half_fill_contours(polylines)
    asm = _assemble(polylines, _CONTOUR_RESAMPLE_H)
    if asm is None:
        # No 0.5 crossing survived the smoothing (a feature thinner than the kernel).
        return _constant_along_x(shape)
    points, prev, nxt, links = asm

    n_v, n_u = shape
    # Everything metric happens in physical coordinates from here on, and that is the whole
    # of the skew correction: `M` maps a point's index coordinates to Cartesian ones, and
    # the direction to the nearest point of the curve is then a Cartesian normal by
    # construction — no pullback to apply, because no gradient was ever taken in index space.
    points = points @ M.T
    tiles_pts, tiles_prev, tiles_next = [], [], []
    n_samples = points.shape[0]
    rings = _tile_rings(M, points, n_v, n_u)
    tile_at = {}
    for dy in range(-rings, rings + 1):
        for dx in range(-rings, rings + 1):
            # The tile offset is a lattice vector, `dx*a1 + dy*a2`, which is what an index
            # offset of a whole cell maps to.
            k = len(tiles_pts)
            tile_at[(dx, dy)] = k
            tiles_pts.append(points + M @ np.array([dx * n_u, dy * n_v], dtype=float))
            tiles_prev.append(prev + k * n_samples)
            tiles_next.append(nxt + k * n_samples)
    tpts = np.vstack(tiles_pts)
    tprev = np.concatenate(tiles_prev)
    tnext = np.concatenate(tiles_next)
    # A percolating curve does not close inside one tile: its last sample is followed by its
    # first sample a *winding* away, in another tile. Joining the copies here is what makes
    # the tiled level set one continuous curve rather than a stack of arcs with an artificial
    # end apiece, and it is the half of the seam stitch that `_contour_polylines` cannot do
    # on its own. A join that would leave the ring is left pointing at itself; that is the old
    # open-end behaviour, on the outermost tiles only, and the translation sweep is clean with
    # it (14 scenes, 1023 rolls) rather than proved unreachable.
    for first, last, wu, wv in links:
        for (dx, dy), k in tile_at.items():
            k2 = tile_at.get((dx + wu, dy + wv))
            if k2 is None:
                continue
            tnext[k * n_samples + last] = k2 * n_samples + first
            tprev[k2 * n_samples + first] = k * n_samples + last

    # Per-sample tangent normal (central difference along the polyline) — the fallback
    # when the chosen segment is degenerate.
    tvec = tpts[tnext] - tpts[tprev]
    tnorm = np.hypot(tvec[:, 0], tvec[:, 1])
    ok = tnorm > 0.0
    samp_n = np.empty_like(tvec)
    samp_n[:, 0] = np.where(ok, -tvec[:, 1] / np.where(ok, tnorm, 1.0), 1.0)
    samp_n[:, 1] = np.where(ok, tvec[:, 0] / np.where(ok, tnorm, 1.0), 0.0)

    # The nearest-sample query is answered in a metric sheared by `_CONTOUR_TIE_SHEAR`, so
    # that two samples exactly equidistant from a pixel -- the medial axis, which a raster's
    # own symmetries make an exact tie rather than a near one -- are ordered the same way at
    # every placement instead of by the last bit. It picks the sample; everything below is
    # measured in the true metric.
    e = _CONTOUR_TIE_SHEAR
    shear = np.array([[1.0, e], [0.0, 1.0 + 2.0 * e]])
    tree = cKDTree(tpts @ shear.T)
    Ii, Jj = np.meshgrid(np.arange(n_u, dtype=float), np.arange(n_v, dtype=float))
    P = np.column_stack([Ii.ravel(), Jj.ravel()]) @ M.T

    _, idx = tree.query(P @ shear.T, workers=-1)
    Q1, d21, AB1 = _project(P, tpts[tprev[idx]], tpts[idx])
    Q2, d22, AB2 = _project(P, tpts[idx], tpts[tnext[idx]])
    use2 = d22 < d21
    Q = np.where(use2[:, None], Q2, Q1)
    AB = np.where(use2[:, None], AB2, AB1)
    d = np.sqrt(np.where(use2, d22, d21))

    seg_n = np.column_stack([-AB[:, 1], AB[:, 0]])
    seg_len = np.hypot(seg_n[:, 0], seg_n[:, 1])
    seg_ok = seg_len > 0.0
    seg_n = np.where(seg_ok[:, None],
                     seg_n / np.where(seg_ok, seg_len, 1.0)[:, None],
                     samp_n[idx])

    v = P - Q
    vnorm = np.hypot(v[:, 0], v[:, 1])
    use_seg = (d < d_switch) | (vnorm < 1e-12)
    n_vec = np.where(use_seg[:, None], seg_n,
                     v / np.where(vnorm > 0.0, vnorm, 1.0)[:, None])
    nn = np.hypot(n_vec[:, 0], n_vec[:, 1])
    nn[nn == 0.0] = 1.0
    return ((n_vec[:, 0] / nn).reshape(n_v, n_u),
            (n_vec[:, 1] / nn).reshape(n_v, n_u))


# ==================================================================== 'spectral'

# FROZEN 2026-08-16 from the angular sweep: Gaussian over hard cutoff, because every
# Gaussian policy converges monotonically with N while hard cutoffs floor out on their own
# ringing; a FIXED cutoff over one tied to N, because every N-tied variant plateaus
# exactly like 'distance' (tying the band to the raster re-admits the staircase
# harmonics). The cutoff VALUE is the one thing a circle harness cannot pick (smaller is
# always better on a shape that stays radially symmetric under blur), so it is pinned at
# the low end of the band the factorization itself reads — orders to 2*truncation_order, truncation_order >= 4
# in practice.
_SPECTRAL_CUTOFF = 8.0   # Fourier orders per period, fixed — never tied to N
_SPECTRAL_REL_MAG_TOL = 1e-3   # band pixels below this fraction of the band max carry
                               # no usable direction and are dropped from the band


def spectral_normal_field(eps, interface_mask, lattice=None, base_shape=None):
    """Band-limited spectral gradient on the interface band, copied into the bulk.

    The construction: FFT the layer's scalar source field (the two-phase indicator when
    there are exactly two values — exact and contrast-independent — else ``Re(eps)``),
    multiply by an isotropic Gaussian low-pass at a *fixed* number of orders per period,
    and read the spectral gradient ``i*k`` per axis. The rationale is the factorization's
    own: it only ever reads orders up to ``2*truncation_order`` of tensors built from ``n n^T``, so a
    band-limited boundary is the boundary the solve can see anyway. Interface-band pixels
    take ``n = g/|g|``; every other pixel copies its nearest band pixel's normal
    (``distance_transform_edt(return_indices=True)``), which keeps ``|n| = 1`` everywhere
    with no zeros.

    Periodic by construction, in both stages — the spectral gradient trivially, and the
    bulk copy through a 3x3 tiling of the band, so a wrapped pixel copies its
    *periodically* nearest band pixel. ``np.gradient`` is one-sided at the array edge, so
    a shape crossing the cell wrap gets structurally wrong differences from ``'distance'``
    and correct ones here.

    A raster with 3+ distinct values delegates to ``'distance'``, exactly like the other
    two constructions: a value-built source field (``Re(eps)``) would make the emitted
    normals depend on the phase *values*, which breaks the geometry-only contract the
    frozen-record staleness tell relies on, and quietly drops interfaces whose phases
    differ only in ``Im(eps)``. The distance mask sees every interface.

    A skewed cell costs this construction nothing either, for a related reason: both
    frequency-domain quantities — the low-pass and the gradient — are stated on the mode's
    *Cartesian* wavevector ``G = 2*pi * M^-T @ (fi, fj)`` (:func:`_cartesian_frequencies`),
    so again there is no index-space difference to pull back. The bulk copy's
    nearest-donor search is the one place a diagonal metric is assumed, and it reaches only
    the homogeneous interior, where every factorization rule agrees whatever the normal says.

    ``base_shape`` is accepted for the shared signature and deliberately unread: the
    cutoff is stated in physical orders per period, which a block-repeat of the raster
    does not change.
    """
    del base_shape
    shape = eps.shape
    values = np.unique(eps)
    if values.size > 2:
        from wavebit.rcwa._engine.factorization import normal_vector_field
        return normal_vector_field(eps, interface_mask, lattice)
    if values.size < 2 or not interface_mask.any():
        return _constant_along_x(shape)

    M = _metric(lattice, shape)
    step_v, step_u = _steps(M)
    B = _reciprocal(M)
    n_v, n_u = shape

    chi = _two_phase_indicator(eps)

    # The raster's Fourier modes are indexed by the grid; what the two quantities below need
    # is the *Cartesian* wavevector each one stands for, which is
    # `G = 2*pi * M^-T @ (fi, fj)` (:func:`_reciprocal`). Both the cutoff and the gradient
    # are stated on `G`, so both are exact on a skewed cell rather than approximated -- this
    # construction never takes a difference in index space, which is what it has instead of
    # a pullback.
    fi = np.fft.fftfreq(n_u)[None, :]            # cycles per index step, axis 1
    fj = np.fft.fftfreq(n_v)[:, None]            # cycles per index step, axis 0
    gxf = B[0, 0] * fi + B[0, 1] * fj           # cycles per unit length, Cartesian
    gyf = B[1, 0] * fi + B[1, 1] * fj
    L_ref = max(n_u * step_u, n_v * step_v)       # longer cell edge, in x-steps
    rr = np.hypot(gxf, gyf) * L_ref             # orders across that edge
    H = np.exp(-0.5 * (rr / _SPECTRAL_CUTOFF) ** 2)

    # The derivative multipliers are built per *index* axis and mixed to Cartesian only
    # afterwards, because the Nyquist mode that carries no odd derivative is an index-space
    # fact -- one column and one row of the transform, which the mixing spreads over both
    # Cartesian components.
    di = 2j * np.pi * np.broadcast_to(fi, shape).astype(complex).copy()
    dj = 2j * np.pi * np.broadcast_to(fj, shape).astype(complex).copy()
    if n_u % 2 == 0:
        di[:, n_u // 2] = 0.0                    # Nyquist: no odd derivative
    if n_v % 2 == 0:
        dj[n_v // 2, :] = 0.0
    kx = B[0, 0] * di + B[0, 1] * dj
    ky = B[1, 0] * di + B[1, 1] * dj

    C = np.fft.fft2(chi) * H
    gx = np.real(np.fft.ifft2(kx * C))
    gy = np.real(np.fft.ifft2(ky * C))
    mag = np.hypot(gx, gy)

    mask = np.asarray(interface_mask, dtype=bool)
    band_max = float(mag[mask].max())
    valid = mask & (mag > _SPECTRAL_REL_MAG_TOL * band_max)
    if not valid.any():
        # No usable spectral contrast (e.g. phases differing only in Im(eps) inside a
        # 3+-phase layer): the 'distance' construction reads the mask alone and can.
        from wavebit.rcwa._engine.factorization import normal_vector_field
        return normal_vector_field(eps, mask, lattice)

    safe = np.where(mag == 0.0, 1.0, mag)
    nx_band = np.where(valid, gx / safe, 0.0)
    ny_band = np.where(valid, gy / safe, 0.0)

    # Bulk copy on a 3x3 tiling, cropped to the centre cell, indices folded back: each
    # pixel copies its PERIODICALLY nearest band pixel. The unwrapped transform would
    # hand wrapped-edge pixels a farther in-cell donor instead (measured: up to 72 deg
    # off on a wrap-crossing disc), while for shapes clear of the wrap the nearest donor
    # is in-cell either way and the tiling changes nothing.
    from scipy.ndimage import distance_transform_edt
    # The one diagonal approximation in this construction, and it reaches nothing that
    # matters: scipy's transform cannot express a non-diagonal metric, so on a skewed cell
    # the per-axis steps are the most it can be told, and a bulk pixel may copy a donor that
    # is not quite the physically nearest one. The *band* is untouched, and the bulk is where
    # the layer is homogeneous -- every factorization rule agrees there whatever the normal
    # says, which is why the field is extended into it at all rather than left at zero.
    sampling = None if step_v == step_u == 1.0 else (step_v, step_u)
    idx = distance_transform_edt(np.tile(~valid, (3, 3)), return_indices=True,
                                 sampling=sampling)[1]
    iy = idx[0][n_v:2 * n_v, n_u:2 * n_u] % n_v
    ix = idx[1][n_v:2 * n_v, n_u:2 * n_u] % n_u
    return nx_band[iy, ix], ny_band[iy, ix]


# ==================================================================== 'smooth'

# FROZEN 2026-08-16 from the angular sweep: the SIGNED distance, because the unsigned
# distance-to-band has a crease AT the interface (it rises on both sides), and smoothing
# across the crease cancels gradients in exactly the band that matters; sigma = N/16 px,
# because the N-scaled sigma is what converges (fixed-in-px stays flat) and N/16 is the
# largest scaling that still does (N/8 reaches the skeleton and degrades).
#
# The sweep ran on square cells, where "N/16 pixels" and "sqrt(cell area)/16 of length"
# are the same number, so it froze the coefficient and not which of the two it meant. The
# length is what the code says now (2026-08-21) — see the docstring.
_SMOOTH_SIGMA_DIVISOR = 16.0
_SMOOTH_SIGMA_FLOOR = 1.0


def smoothed_distance_normal_field(eps, interface_mask, lattice=None, base_shape=None):
    """Derivative-of-Gaussian of the periodic *signed* distance field.

    The nearest construction to ``'distance'`` that removes the quantization: label
    inside/outside (two-phase only — 3+ values delegate to ``'distance'``, whose mask
    sees every interface), build the periodic signed distance
    (:func:`_periodic_signed_distance`), and differentiate it with a derivative-of-Gaussian
    at ``sigma = sqrt(cell area) / 16``. Exact-zero gradients stay ``(0, 0)`` — the skeleton
    handling the factorization already collapses to the direct rule.

    **Kept with a warning in its name**: this construction had the best angular numbers
    on the circle and *lost every resolved solver cell on the rotated rectangle* (5 of 5,
    the sixth unresolved) —
    smoothing at period/16 rounds corners. It is the measured demonstration that the best
    angular field is not the best solver field; prefer ``'contour'`` unless the layer is
    genuinely smooth everywhere (docs/theory/rcwa/nvf-normal-methods.md has both tables).

    **The sigma is a length, and one number rather than one per axis** (changed 2026-08-21).
    It used to be ``base_N/16`` *pixels* along each axis, which on a square cell is the same
    quantity and on any other cell is an ellipse elongated toward whichever axis spans more
    of the cell — a shape nobody chose, since the sweep that froze the 1/16 saw square cells
    only. It also was not a property of the *lattice*: ``a2' = a2 + a1`` generates the
    identical point lattice and would have moved the kernel. ``sqrt(cell area)/16`` is the
    reading that is isotropic and basis-invariant at once, and it reduces to ``N/16`` px
    bit-identically on a square cell. On the 2:1 cell the non-square test draws its disc in,
    it takes the mean angular error from 2.22 to 0.39 degrees.

    **A skewed cell needs both of this construction's index-space assumptions repaired.** It
    is the only one here that has any: scipy's distance transform and scipy's Gaussian are
    per-axis by construction and a skewed metric is not diagonal, so :func:`_separable`
    routes both to a kd-tree and a frequency-domain kernel instead. Left as they were, the
    construction measures 2.1 to 11.8 degrees on a skewed disc — worst on the *small*
    features, where the kernel is widest relative to the shape — against 0.07 to 0.13 with
    both repaired, and neither repair alone gets there
    (``docs/theory/rcwa/nvf-normal-methods.md`` has the 2x2).
    """
    shape = eps.shape
    values = np.unique(eps)
    if values.size > 2:
        from wavebit.rcwa._engine.factorization import normal_vector_field
        return normal_vector_field(eps, interface_mask, lattice)
    if values.size < 2 or not interface_mask.any():
        return _constant_along_x(shape)

    M = _metric(lattice, shape)
    repeat_v, repeat_u = _repeats(shape, base_shape)
    base_shape = base_shape or shape

    sd = _periodic_signed_distance(_two_phase_indicator(eps).astype(bool), M)

    # Sigma on the BASE grid, carried to the lifted one: 1/16 of the cell's linear size,
    # floored at one base pixel. A *length*, and one isotropic number rather than one per
    # axis -- see the docstring for why that is the reading, and what the per-axis one cost.
    sigma_base = max(_SMOOTH_SIGMA_FLOOR,
                     float(np.sqrt(base_shape[0] * base_shape[1])) / _SMOOTH_SIGMA_DIVISOR)
    sigma = sigma_base * float(np.sqrt(repeat_v * repeat_u)) * _pixel_size(M)
    gx, gy = _isotropic_blur_gradient(sd, sigma, M)

    norm = np.hypot(gx, gy)
    norm = np.where(norm == 0, 1.0, norm)   # exact zeros stay (0, 0): skeleton handling
    return gx / norm, gy / norm


DISPATCH = {
    'contour': contour_normal_field,
    'spectral': spectral_normal_field,
    'smooth': smoothed_distance_normal_field,
}
