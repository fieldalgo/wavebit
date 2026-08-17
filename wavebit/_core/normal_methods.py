"""The raster normal-field constructions behind ``normal_method`` — all but the original.

:func:`wavebit._core.factorization.normal_vector_field` dispatches here for every method
except ``'distance'``, which stays where it always was. Each construction takes the same
inputs — a concrete permittivity raster, its interface mask, the cell basis, and the shape
the raster had before the ``4*fto+1`` floor lift — and returns ``(nx, ny)`` float arrays
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
- **The base grid sets the smoothing scale.** A layer lifted to the ``4*fto+1`` floor is
  the same staircase spelled with more pixels (that is the whole finding of
  ``docs/nvf-upsampling.md``), so every pixel-unit policy below reads ``base_shape`` and
  scales by the lift factor — a 10x10 layer lifted 4x smooths over 4x as many lifted
  pixels, not over a quarter of its base features.
- **The basis must be axis-aligned.** Square-pixel cells reduce bit-identically to no
  metric at all (the same normalization :func:`normal_vector_field` uses); non-square
  cells work in physical coordinates. A skewed basis raises: the raster NVF path *does*
  run on an oblique lattice, but only the ``'distance'`` construction carries the
  Jacobian pullback that makes an index-space gradient a Cartesian normal there — these
  three are not ported to a skewed cell, and the engine refuses the combination up front
  (``_reject_unsupported_lattice``); the raise here guards direct callers.
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


# ------------------------------------------------------------------- shared helpers

def _steps(lattice, shape):
    """Physical length of one index step per axis, ``(step_y, step_x)``, normalised so a
    square-pixel cell lands on exactly ``(1.0, 1.0)`` — the same normalisation
    :func:`~wavebit._core.factorization.normal_vector_field` applies, for the same reason:
    a uniform scale cannot survive the unit normalisation at the end, and pinning the
    x-step to 1 makes the common case bit-identical to taking no metric at all.

    Raises for a skewed basis. The oblique-lattice raster path exists, but only
    ``'distance'`` is ported to it (it carries the ``J^-T`` pullback); the engine refuses
    the combination up front (``_reject_unsupported_lattice``), so this raise only guards
    direct callers.
    """
    if lattice is None:
        return 1.0, 1.0
    from .lattice import is_axis_aligned, jacobian
    if not is_axis_aligned(lattice):
        raise NotImplementedError(
            "normal_method='contour'/'spectral'/'smooth' support an axis-aligned cell "
            "basis only; on a skewed cell only normal_method='distance' is implemented "
            "(it carries the Jacobian pullback).")
    ny, nx = shape
    J = np.asarray(jacobian(lattice, ny, nx), dtype=float)
    J = J / (float(np.hypot(J[0, 0], J[1, 0])) or 1.0)
    return float(np.hypot(J[0, 1], J[1, 1])), float(np.hypot(J[0, 0], J[1, 0]))


def _repeats(shape, base_shape):
    """Lift factors ``(repeat_y, repeat_x)`` of the floor-lifted grid over its base."""
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


def _contour_polylines(smoothed, sigma_yx):
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
    del sigma_yx                      # the pad is set by the period now, not by the kernel
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
    """
    shape = eps.shape
    values = np.unique(eps)
    if values.size > 2:
        from .factorization import normal_vector_field
        return normal_vector_field(eps, interface_mask, lattice)
    if values.size < 2 or not interface_mask.any():
        return _constant_along_x(shape)

    step_y, step_x = _steps(lattice, shape)
    repeat_y, repeat_x = _repeats(shape, base_shape)
    base_shape = base_shape or shape

    # The frozen sigma, stated on the base grid and carried to the lifted one: physically
    # isotropic, 0.25 * sqrt(N_base) base pixels wide with N_base the geometric-mean base
    # size. In physical units that is 0.25 / sqrt(N_base) of the period, -> 0 with real
    # resolution, which is what makes the construction converge under refinement.
    sigma_base = _CONTOUR_SIGMA_COEF * float(np.sqrt(np.sqrt(
        base_shape[0] * base_shape[1])))
    sigma_phys = sigma_base * float(np.sqrt((repeat_y * step_y) * (repeat_x * step_x)))
    sigma_yx = (sigma_phys / step_y, sigma_phys / step_x)
    # The inert switch length stays in geometric-mean physical pixels; the resample step
    # is an index-space quantity and needs no conversion.
    px = float(np.sqrt(step_y * step_x))
    d_switch = _CONTOUR_D_SWITCH * px

    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    indicator = _two_phase_indicator(eps)
    smoothed = gaussian_filter(indicator, sigma=sigma_yx, mode='wrap')
    asm = _assemble(_contour_polylines(smoothed, sigma_yx), _CONTOUR_RESAMPLE_H)
    if asm is None:
        # No 0.5 crossing survived the smoothing (a feature thinner than the kernel).
        return _constant_along_x(shape)
    points, prev, nxt = asm

    ny, nx = shape
    # Everything metric happens in physical coordinates from here on.
    points = points * np.array([step_x, step_y])
    tiles_pts, tiles_prev, tiles_next = [], [], []
    M = points.shape[0]
    k = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            tiles_pts.append(points + np.array([dx * nx * step_x, dy * ny * step_y]))
            tiles_prev.append(prev + k * M)
            tiles_next.append(nxt + k * M)
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
    X, Y = np.meshgrid(np.arange(nx, dtype=float) * step_x,
                       np.arange(ny, dtype=float) * step_y)
    P = np.column_stack([X.ravel(), Y.ravel()])

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
# the low end of the band the factorization itself reads — orders up to 2*fto, fto >= 4
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
    own: it only ever reads orders up to ``2*fto`` of tensors built from ``n n^T``, so a
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

    step_y, step_x = _steps(lattice, shape)
    aspect = step_y / step_x
    ny, nx = shape

    chi = _two_phase_indicator(eps)

    fy = np.fft.fftfreq(ny)[:, None] / aspect   # cycles per x-step, axis 0
    fx = np.fft.fftfreq(nx)[None, :]            # cycles per x-step, axis 1
    L_ref = max(nx, ny * aspect)                # longer cell extent, in x-steps
    rr = np.hypot(fx, fy) * L_ref               # orders across that extent
    H = np.exp(-0.5 * (rr / _SPECTRAL_CUTOFF) ** 2)

    kx = 2j * np.pi * fx.astype(complex)
    ky = 2j * np.pi * fy.astype(complex)
    if nx % 2 == 0:
        kx[0, nx // 2] = 0.0                    # Nyquist: no odd derivative
    if ny % 2 == 0:
        ky[ny // 2, 0] = 0.0

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
    sampling = None if aspect == 1.0 else (aspect, 1.0)
    idx = distance_transform_edt(np.tile(~valid, (3, 3)), return_indices=True,
                                 sampling=sampling)[1]
    iy = idx[0][ny:2 * ny, nx:2 * nx] % ny
    ix = idx[1][ny:2 * ny, nx:2 * nx] % nx
    return nx_band[iy, ix], ny_band[iy, ix]


# ==================================================================== 'smooth'

# FROZEN 2026-08-16 from the angular sweep: the SIGNED distance, because the unsigned
# distance-to-band has a crease AT the interface (it rises on both sides), and smoothing
# across the crease cancels gradients in exactly the band that matters; sigma = N/16 px
# per axis, because the N-scaled sigma is what converges (fixed-in-px stays flat) and
# N/16 is the largest scaling that still does (N/8 reaches the skeleton and degrades).
_SMOOTH_SIGMA_DIVISOR = 16.0
_SMOOTH_SIGMA_FLOOR = 1.0


def smoothed_distance_normal_field(eps, interface_mask, lattice=None, base_shape=None):
    """Derivative-of-Gaussian of the periodic *signed* distance field.

    The nearest construction to ``'distance'`` that removes the quantization: label
    inside/outside (two-phase only — 3+ values delegate to ``'distance'``, whose mask
    sees every interface), build the periodic signed Euclidean distance on a 3x3 tiling,
    and differentiate with a derivative-of-Gaussian at ``sigma = base_N/16`` pixels per
    axis. Exact-zero gradients stay ``(0, 0)`` — the skeleton handling the factorization
    already collapses to the direct rule.

    **Kept with a warning in its name**: this construction had the best angular numbers
    on the circle and *lost every resolved solver cell on the rotated rectangle* (5 of 5,
    the sixth unresolved) —
    smoothing at period/16 rounds corners. It is the measured demonstration that the best
    angular field is not the best solver field; prefer ``'contour'`` unless the layer is
    genuinely smooth everywhere (docs/nvf-normal-methods.md has both tables).
    """
    shape = eps.shape
    values = np.unique(eps)
    if values.size > 2:
        from .factorization import normal_vector_field
        return normal_vector_field(eps, interface_mask, lattice)
    if values.size < 2 or not interface_mask.any():
        return _constant_along_x(shape)

    step_y, step_x = _steps(lattice, shape)
    repeat_y, repeat_x = _repeats(shape, base_shape)
    base_shape = base_shape or shape

    from scipy.ndimage import distance_transform_edt, gaussian_filter

    inside = _two_phase_indicator(eps).astype(bool)
    ny, nx = shape
    t = np.tile(inside, (3, 3))
    sampling = None if step_y == step_x == 1.0 else (step_y, step_x)
    sd = (distance_transform_edt(~t, sampling=sampling)
          - distance_transform_edt(t, sampling=sampling))
    sd = sd[ny:2 * ny, nx:2 * nx]

    # Sigma per axis on the BASE grid, carried to the lifted one: 1/16 of the period
    # along each axis, floored at one base pixel.
    sig_y = max(_SMOOTH_SIGMA_FLOOR, base_shape[0] / _SMOOTH_SIGMA_DIVISOR) * repeat_y
    sig_x = max(_SMOOTH_SIGMA_FLOOR, base_shape[1] / _SMOOTH_SIGMA_DIVISOR) * repeat_x
    gx = gaussian_filter(sd, (sig_y, sig_x), order=(0, 1), mode='wrap') / step_x
    gy = gaussian_filter(sd, (sig_y, sig_x), order=(1, 0), mode='wrap') / step_y

    norm = np.hypot(gx, gy)
    norm = np.where(norm == 0, 1.0, norm)   # exact zeros stay (0, 0): skeleton handling
    return gx / norm, gy / norm


DISPATCH = {
    'contour': contour_normal_field,
    'spectral': spectral_normal_field,
    'smooth': smoothed_distance_normal_field,
}
