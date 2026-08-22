"""The raster normal-field constructions behind ``normal_method`` — all but the original.

:func:`wavebit._core.factorization.normal_vector_field` dispatches here for every method
except ``'distance'``, which stays where it always was. Each construction takes the same
inputs — a concrete permittivity raster, its interface mask, the cell basis, and the shape
the raster had before the ``4*truncation_order+1`` floor lift — and returns ``(nx, ny)`` float arrays
with ``|n| = 1`` everywhere it emits a direction at all.

Why these exist: the distance-transform gradient quantizes. Central differences of a
distance-to-a-digital-set over a two-cell span land on small integers, so the boundary band
emits directions from the short list ``arctan(p/q)`` — in practice {0, 45} degrees — with a
mean error of ~14 degrees that does not shrink as the raster is refined, because a finer
staircase is the same staircase. All three constructions here break the quantization the
same way the raster-native literature does (Gotz 2008, fmmax, subpixel-smoothed
projection): smooth the indicator *before* differentiating, and keep the boundary's
location from the raw raster. ``docs/nvf-normal-methods.md`` records the measurements — the
angular tables, the solver-level A/B against the analytic reference, and the sweeps that
froze each construction's one policy.

Shared conventions, stated once:

- **Everything is eager NumPy/SciPy**, whatever the backend, because the normal field is
  frozen geometry (see :func:`~wavebit._core.factorization.normal_vector_field`): a
  constant handed to the solve, outside the AD graph. Reading the raster's *values* is what
  these constructions add over the mask-only ``'distance'`` path, so a traced layer cannot
  take them — the dispatcher turns that into an error naming ``freeze_geometry()``.
- **Sign is free.** The factorization consumes ``n n^T`` only, so ``n`` and ``-n`` are the
  same field; no construction here spends effort orienting.
- **The base grid sets the smoothing scale.** A layer lifted to the ``4*truncation_order+1`` floor is
  the same staircase spelled with more pixels (that is the whole finding of
  ``docs/nvf-upsampling.md``), so every pixel-unit policy below reads ``base_shape`` and
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
- **Only values-through-geometry are read.** The indicators are corner-canonical
  (:func:`_two_phase_indicator`), so relabeling the two phases moves nothing, and a
  raster with 3+ distinct values — including the continuous/grayscale densities of a
  mid-run topology optimization — delegates to ``'distance'``: a value-built source
  field would break the geometry-only contract the frozen-record staleness tell relies
  on. The record additionally pins the phase *count*, the one value-derived fact the
  mask cannot see. The grayscale case is a stated limit, not an accident: an SSP-style
  filtered-density indicator (arXiv:2503.20189) is the known follow-up.
- **Covariance under a mirror is exact for one of the three.** ``'smooth'`` mirrors
  bit-exactly in both directions, like ``'distance'``. ``'contour'`` and ``'spectral'``
  carry tie-breaks (the nearest polyline sample, the nearest valid band donor) and a
  resampling phase, and a mirror can break those the other way — worst along x, and
  concentrated on the periodic medial ridge where the direction is genuinely ambiguous.
  Measured on an off-center circle at N=48: pointwise p99 up to ~3 deg (contour) /
  ~34 deg (spectral, ridge donors) with a few percent of pixels past 1 deg; at the
  solve, a structure and its x-mirror differ by ``max|dR| ~ 1.6e-4`` against ~1e-15
  for ``'distance'``/``'smooth'`` — an order below these methods' accuracy gains at the
  same settings, and the bound a test pins. A workflow that needs exact mirror symmetry
  of the *output* should use ``'smooth'`` or ``'distance'``.
"""
import numpy as np

from .lattice import is_axis_aligned, jacobian


# ------------------------------------------------------------------- shared helpers

def _metric(lattice, shape):
    """The cell's metric as one ``(2, 2)`` matrix: ``(x, y) = M @ (i, j)``.

    ``M`` is :func:`~wavebit._core.lattice.jacobian` — columns ``a1/nx`` and ``a2/ny``, the
    Cartesian displacement of one index step along each axis — divided through by the length
    of its first column. That normalisation is the one
    :func:`~wavebit._core.factorization.normal_vector_field` applies and is there for the same
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
    ny, nx = shape
    M = np.asarray(jacobian(lattice, ny, nx), dtype=float)
    return M / (float(np.hypot(M[0, 0], M[1, 0])) or 1.0)


def _steps(M):
    """``(step_v, step_u)``: the physical length of one index step along each axis.

    The column norms of :func:`_metric`. Two quite different things are read off them and it is
    worth keeping them apart. ``nx * step_u`` and ``ny * step_v`` are the cell's two edge
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
    :func:`~wavebit._core.factorization.normal_vector_field` applies on the ``'distance'``
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

    This is :func:`~wavebit._core.lattice.is_axis_aligned` asked of the metric rather than of
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
    ny, nx = shape
    B = _reciprocal(M)
    fi = np.fft.fftfreq(nx)[None, :]
    fj = np.fft.fftfreq(ny)[:, None]
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
    every measurement in ``docs/nvf-normal-methods.md`` — including the sweep that froze
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
    (``docs/nvf-normal-methods.md``).
    """
    ny, nx = inside.shape
    if _separable(M):
        from scipy.ndimage import distance_transform_edt
        step_v, step_u = _steps(M)
        sampling = None if step_v == step_u == 1.0 else (step_v, step_u)
        t = np.tile(inside, (3, 3))
        sd = (distance_transform_edt(~t, sampling=sampling)
              - distance_transform_edt(t, sampling=sampling))
        return sd[ny:2 * ny, nx:2 * nx]

    from scipy.spatial import cKDTree
    Ii, Jj = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    P = np.column_stack([Ii.ravel(), Jj.ravel()]) @ M.T

    def nearest(phase):
        jj, ii = np.nonzero(phase)
        tiled = np.vstack([np.column_stack([ii + dx * nx, jj + dy * ny]).astype(float)
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        return cKDTree(tiled @ M.T).query(P, workers=-1)[0].reshape(ny, nx)

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
    """Not-the-corner-phase indicator of a two-phase raster — canonical in the REGION.

    The background is whatever phase occupies the corner pixel, not ``np.unique(eps)[0]``:
    swapping the two material values exchanges which value ``unique`` sorts first and
    inverts a value-sorted indicator, but it leaves the corner's *region* where it was —
    so this indicator is invariant under relabeling the phases, which is what keeps the
    whole construction a function of the geometry alone (the contract
    :func:`~wavebit._core.factorization.normal_vector_field` states, and the frozen-record
    staleness tell relies on).
    """
    return (eps != eps.flat[0]).astype(float)


# ==================================================================== 'contour'

# FROZEN 2026-08-16 from the angular sweep (docs/nvf-normal-methods.md): sigma follows the
# sqrt(N) family because every fixed-in-px sigma leaves the error FLAT in N (a fixed
# kernel keeps a fixed fraction of the staircase wiggle) while every sqrt(N) policy
# converges; the coefficient is the smallest of the top group, tie-broken by the measured
# thin-feature erasure threshold (~1.4 sigma px), not tuned per case. D_SWITCH and
# RESAMPLE_H were measured inert (< 0.3 deg across 4x changes).
_CONTOUR_SIGMA_COEF = 0.25    # sigma = 0.25 * sqrt(N) px, N the geometric-mean base size
_CONTOUR_D_SWITCH = 0.5       # below this distance (px) use the local segment normal
_CONTOUR_RESAMPLE_H = 0.25    # polyline resampling step (px)


def _contour_polylines(smoothed):
    """0.5 level set of the wrap-padded smoothed indicator — **one whole copy of each loop**.

    Returns ``(points, closed)`` pairs in unpadded index coordinates (x = column, y = row).
    Two things here are load-bearing, and the construction was not covariant under a lattice
    translation of the raster until both were (see ``docs/nvf-normal-methods.md``).

    **Pad by a full cell, not by the kernel's support.** The pad used to be ``4*sigma+3``,
    which on a 16 px cell is about five pixels — far narrower than a period. The level set is
    then extracted through a window that cuts the periodic copies at a *fixed* place while a
    roll slides the shape past it, so which copies come back whole depends on where the
    structure happens to sit. A full cell is the smallest pad that guarantees at least one
    complete copy of every loop, whatever its extent.

    **Keep one whole copy of each loop, not every fragment.** Even at a full cell, a shape
    straddling the wrap still has copies clipped by the outer boundary: measured on a rod
    rolled across it, nine whole loops become six whole and six fragments. Fragments are not
    new geometry — the 3x3 tiling in :func:`contour_normal_field` regenerates every periodic
    copy downstream — but they are actively harmful twice over: a fragment's endpoint is not a
    curve end, so ``_assemble`` points it at itself and the segment normal there is one-sided;
    and :func:`_resample` samples an open curve by ``linspace`` and a closed one by
    ``arange``, so fragments inject sample positions the whole loops do not have (67 distinct
    points became 129 on that rod). A pixel's nearest sample then depends on the placement.

    So each closed loop is folded back to the centre cell by whole periods and deduplicated
    there, leaving exactly one representative per loop. Open curves are kept only when *no*
    closed loop survived: those are percolating level sets — a stripe spanning the cell has no
    closed 0.5 crossing at all — and there is no whole copy of them to prefer.

    Cost is neutral to favourable despite the wider window, because the dedup shrinks what the
    kd-tree downstream indexes: measured 1.06x at N=64 and 0.79x at N=1024.
    """
    try:
        import contourpy
    except ImportError:
        raise ImportError(
            "normal_method='contour' extracts subpixel level sets with contourpy, which "
            "is not installed. `pip install contourpy` (it ships with matplotlib), or use "
            "normal_method='spectral', which needs nothing beyond NumPy/SciPy.") from None
    ny, nx = smoothed.shape
    pad = max(ny, nx)
    padded = np.pad(smoothed, pad, mode='wrap')
    gen = contourpy.contour_generator(z=padded, line_type=contourpy.LineType.Separate)
    closed_loops, open_curves = [], []
    for pts in gen.lines(0.5):
        pts = np.asarray(pts, dtype=float)
        if pts.shape[0] < 2:
            continue
        pts = pts - pad
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
            if pts.shape[0] < 3:
                continue
            closed_loops.append(pts)
        else:
            open_curves.append(pts)

    out, seen = [], set()
    for pts in closed_loops:
        centre = pts.mean(0)
        folded = pts - np.array([np.round(centre[0] / nx) * nx,
                                 np.round(centre[1] / ny) * ny])
        centre = folded.mean(0)
        key = (round(float(centre[0]), 3), round(float(centre[1]), 3), folded.shape[0])
        if key in seen:
            continue
        seen.add(key)
        out.append((folded, True))
    if out:
        return out
    return [(pts, False) for pts in open_curves]


def _resample(pts, closed, h):
    """Uniform arc-length resampling with step <= ``h``; ``None`` for degenerate input."""
    if closed:
        seg = np.diff(np.vstack([pts, pts[:1]]), axis=0)
        knots = np.vstack([pts, pts[:1]])
    else:
        seg = np.diff(pts, axis=0)
        knots = pts
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    L = float(seglen.sum())
    if L <= 0.0:
        return None
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    if closed:
        n = max(int(np.ceil(L / h)), 8)
        t = np.arange(n) * (L / n)
    else:
        n = max(int(np.ceil(L / h)), 2)
        t = np.linspace(0.0, L, n + 1)
    return np.column_stack([np.interp(t, s, knots[:, 0]), np.interp(t, s, knots[:, 1])])


def _assemble(polylines, h):
    """Concatenate resampled polylines into ``(points, prev, nxt)`` adjacency arrays.
    Open ends point at themselves, so projection degenerates to the endpoint — correctly.
    """
    all_pts, all_prev, all_next = [], [], []
    base = 0
    for pts, closed in polylines:
        rs = _resample(pts, closed, h)
        if rs is None:
            continue
        n = rs.shape[0]
        idx = np.arange(n)
        if closed:
            prev, nxt = base + (idx - 1) % n, base + (idx + 1) % n
        else:
            prev, nxt = base + np.maximum(idx - 1, 0), base + np.minimum(idx + 1, n - 1)
        all_pts.append(rs)
        all_prev.append(prev)
        all_next.append(nxt)
        base += n
    if not all_pts:
        return None
    return np.vstack(all_pts), np.concatenate(all_prev), np.concatenate(all_next)


def _project(P, A, B):
    """Foot of ``P`` on segment ``A -> B`` (clamped), squared distance, segment vector."""
    AB = B - A
    denom = (AB * AB).sum(1)
    safe = np.where(denom > 0.0, denom, 1.0)
    t = np.clip(((P - A) * AB).sum(1) / safe, 0.0, 1.0)
    t = np.where(denom > 0.0, t, 0.0)
    Q = A + t[:, None] * AB
    return Q, ((P - Q) ** 2).sum(1), AB


def contour_normal_field(eps, interface_mask, lattice=None, base_shape=None):
    """Subpixel level-set contour, then the exact direction of grad(distance-to-curve).

    The construction: smooth the two-phase indicator with a periodic Gaussian, extract its
    0.5 level set as subpixel polylines (contourpy), resample them finely, and give every
    pixel the direction to its exact nearest point on the curve — which *is* the gradient
    of the distance-to-curve field, i.e. the exact version of the extension the
    ``'distance'`` construction approximates on the integer grid. Below half a pixel the
    point-difference direction is noisy at polyline vertices, so the local segment normal
    stands in (the two coincide in segment interiors).

    This is the measured winner: it won every resolved cell in the solver A/B on both
    scenes and lost none, and it converges with N where ``'distance'`` is flat
    (docs/nvf-normal-methods.md).

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
        from .factorization import normal_vector_field
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
    asm = _assemble(_contour_polylines(smoothed), _CONTOUR_RESAMPLE_H)
    if asm is None:
        # No 0.5 crossing survived the smoothing (a feature thinner than the kernel).
        return _constant_along_x(shape)
    points, prev, nxt = asm

    ny, nx = shape
    # Everything metric happens in physical coordinates from here on, and that is the whole
    # of the skew correction: `M` maps a point's index coordinates to Cartesian ones, and
    # the direction to the nearest point of the curve is then a Cartesian normal by
    # construction — no pullback to apply, because no gradient was ever taken in index space.
    points = points @ M.T
    tiles_pts, tiles_prev, tiles_next = [], [], []
    n_samples = points.shape[0]
    k = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            # The tile offset is a lattice vector, `dx*a1 + dy*a2`, which is what an index
            # offset of a whole cell maps to.
            tiles_pts.append(points + M @ np.array([dx * nx, dy * ny], dtype=float))
            tiles_prev.append(prev + k * n_samples)
            tiles_next.append(nxt + k * n_samples)
            k += 1
    tpts = np.vstack(tiles_pts)
    tprev = np.concatenate(tiles_prev)
    tnext = np.concatenate(tiles_next)

    # Per-sample tangent normal (central difference along the polyline) — the fallback
    # when the chosen segment is degenerate.
    tvec = tpts[tnext] - tpts[tprev]
    tnorm = np.hypot(tvec[:, 0], tvec[:, 1])
    ok = tnorm > 0.0
    samp_n = np.empty_like(tvec)
    samp_n[:, 0] = np.where(ok, -tvec[:, 1] / np.where(ok, tnorm, 1.0), 1.0)
    samp_n[:, 1] = np.where(ok, tvec[:, 0] / np.where(ok, tnorm, 1.0), 0.0)

    tree = cKDTree(tpts)
    Ii, Jj = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    P = np.column_stack([Ii.ravel(), Jj.ravel()]) @ M.T

    _, idx = tree.query(P, workers=-1)
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
    return ((n_vec[:, 0] / nn).reshape(ny, nx),
            (n_vec[:, 1] / nn).reshape(ny, nx))


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
        from .factorization import normal_vector_field
        return normal_vector_field(eps, interface_mask, lattice)
    if values.size < 2 or not interface_mask.any():
        return _constant_along_x(shape)

    M = _metric(lattice, shape)
    step_v, step_u = _steps(M)
    B = _reciprocal(M)
    ny, nx = shape

    chi = _two_phase_indicator(eps)

    # The raster's Fourier modes are indexed by the grid; what the two quantities below need
    # is the *Cartesian* wavevector each one stands for, which is
    # `G = 2*pi * M^-T @ (fi, fj)` (:func:`_reciprocal`). Both the cutoff and the gradient
    # are stated on `G`, so both are exact on a skewed cell rather than approximated -- this
    # construction never takes a difference in index space, which is what it has instead of
    # a pullback.
    fi = np.fft.fftfreq(nx)[None, :]            # cycles per index step, axis 1
    fj = np.fft.fftfreq(ny)[:, None]            # cycles per index step, axis 0
    gxf = B[0, 0] * fi + B[0, 1] * fj           # cycles per unit length, Cartesian
    gyf = B[1, 0] * fi + B[1, 1] * fj
    L_ref = max(nx * step_u, ny * step_v)       # longer cell edge, in x-steps
    rr = np.hypot(gxf, gyf) * L_ref             # orders across that edge
    H = np.exp(-0.5 * (rr / _SPECTRAL_CUTOFF) ** 2)

    # The derivative multipliers are built per *index* axis and mixed to Cartesian only
    # afterwards, because the Nyquist mode that carries no odd derivative is an index-space
    # fact -- one column and one row of the transform, which the mixing spreads over both
    # Cartesian components.
    di = 2j * np.pi * np.broadcast_to(fi, shape).astype(complex).copy()
    dj = 2j * np.pi * np.broadcast_to(fj, shape).astype(complex).copy()
    if nx % 2 == 0:
        di[:, nx // 2] = 0.0                    # Nyquist: no odd derivative
    if ny % 2 == 0:
        dj[ny // 2, :] = 0.0
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
        from .factorization import normal_vector_field
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
    iy = idx[0][ny:2 * ny, nx:2 * nx] % ny
    ix = idx[1][ny:2 * ny, nx:2 * nx] % nx
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
    genuinely smooth everywhere (docs/nvf-normal-methods.md has both tables).

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
    (``docs/nvf-normal-methods.md`` has the 2x2).
    """
    shape = eps.shape
    values = np.unique(eps)
    if values.size > 2:
        from .factorization import normal_vector_field
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
