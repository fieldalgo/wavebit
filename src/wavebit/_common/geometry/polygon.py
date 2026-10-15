# -*- coding: utf-8 -*-
"""Closed-form Fourier transform of a polygon, and the vertices of the primitives.

The divergence theorem turns the area integral of ``exp(-i k.r)`` over a polygon into a sum
over its edges, which is **exact** for any straight edge at any angle -- there is no staircase
and no quadrature. ``k = 0`` is the one point where that form is singular; it is an isolated
lattice point (the smallest non-zero ``|k|`` is ``2*pi/max(Px, Py)``), so a single ``where``
handles it and there is no small-denominator regime to approach.

The result is not new here. S.-W. Lee and R. Mittra, "Fourier transform of a polygonal shape
function and its application in electromagnetics", *IEEE Trans. Antennas Propag.* **31**(1),
99-103 (1983), is the standard electromagnetics reference for it; the derivation is repeated
above because the sign and factor-of-two conventions have to match the raster path's
:func:`wavebit.rcwa._engine.factorization._pixel_correction` exactly, and getting ``sinc((k.e)/2)`` wrong as
``sinc(k.e)`` is invisible on axis-aligned rectangles.

:func:`polygon_area` is the shoelace formula (Gauss's area formula).

A hole needs no special case. The transform is a signed contour integral, so a loop wound the
other way subtracts by itself, and :mod:`.clipper` returns holes exactly that way.

The second half of that is a **guarantee the clipper deliberately provides**, not an incidental
property to be re-derived here. It stopped being true once, when the clipper normalised every
loop to counter-clockwise and moved the orientation into a separate field; the sentence was
left standing and :func:`.boundary.conv_mats` kept believing it, which silently solved the
wrong structure for any shape painted entirely inside another. See
:func:`.clipper.subtract_loop` for where the convention is established.
"""
from wavebit._common.backend import in_namespace


def _sinc(xp, x):
    """sin(x)/x with the removable singularity filled, safe for autodiff."""
    small = xp.abs(x) < 1e-8
    xs = xp.where(small, xp.ones_like(x), x)
    return xp.where(small, xp.ones_like(x) - x * x / 6.0, xp.sin(xs) / xs)


def polygon_area(xp, verts):
    """Signed area (shoelace); positive for counter-clockwise."""
    x, y = verts[:, 0], verts[:, 1]
    xn, yn = xp.roll(x, -1), xp.roll(y, -1)
    return 0.5 * xp.sum(x * yn - xn * y)


def polygon_ft(xp, verts, ky, kx, lattice):
    """1/area * FT of the polygon indicator, on the integer reciprocal grid (ky, kx).

    ``verts`` is (N, 2) in (x, y), counter-clockwise. Returns an array shaped like ``kx``.

    ``lattice`` is the cell basis (``None`` = axis-aligned ``diag(period)``), and as in
    :func:`wavebit.rcwa._engine.parametric.shape_ft` it reaches only the two lines that turn the order
    indices into ``(Gx, Gy)``. Everything after them is the contour integral in **Cartesian**
    frequency, so a skewed cell changes where the transform is sampled and nothing about the
    polygon. Vertices are absolute Cartesian positions, and the shape needs no wrapping at the
    cell edge either: ``exp(-i G.r)`` carries the lattice periodicity, so integrating over the
    polygon wherever it sits gives the right coefficient.
    """
    pi = 3.141592653589793
    # The `lattice is None` arm that used to sit here computed the rectangle's answer from
    # `Px`, `Py`. The solver stopped reaching it once `_geometry_lattice_for` became total
    # (`cc016fc`) -- driving only the public API, the one function that ever sees a `None`
    # cell is `as_lattice` itself, where the conversion happens. Deleted rather than
    # re-keyed on `is_axis_aligned`: production takes *this* branch on a rectangle today,
    # and the two differ in the last place (1.3e-16, measured).
    a1x, a2x = lattice[0][0], lattice[0][1]
    a1y, a2y = lattice[1][0], lattice[1][1]
    det = a1x * a2y - a2x * a1y
    Gx = 2.0 * pi * (a2y * kx - a1y * ky) / det
    Gy = 2.0 * pi * (-a2x * kx + a1x * ky) / det
    cell_area = abs(det)

    x, y = verts[:, 0], verts[:, 1]
    xn, yn = xp.roll(x, -1), xp.roll(y, -1)
    ex, ey = xn - x, yn - y                       # edge vectors
    mx, my = 0.5 * (x + xn), 0.5 * (y + yn)       # edge midpoints

    # Broadcast: (..., 1) frequency against (N,) edges.
    GX, GY = Gx[..., None], Gy[..., None]
    k_dot_e = GX * ex + GY * ey
    k_cross_n = GX * ey - GY * ex                   # (k . n_hat) * |e|
    terms = k_cross_n * xp.exp(-1j * (GX * mx + GY * my)) * _sinc(xp, 0.5 * k_dot_e)
    contour = xp.sum(terms, axis=-1)

    k2 = Gx * Gx + Gy * Gy
    nz = k2 != 0
    k2_safe = xp.where(nz, k2, xp.ones_like(k2))
    area = polygon_area(xp, verts)
    F = xp.where(nz, 1j * contour / k2_safe, (area + 0j) * xp.ones_like(contour))
    return F / cell_area


# ---------------------------------------------------------------- primitives

def _speaks(xp, dt):
    """Can ``xp`` be *asked* for ``dt``? PyTorch cannot be asked for one of NumPy's.

    A shape's centre carries whatever the **cell basis** is made of, which on an ordinary torch
    solve is a NumPy float64 -- ``as_lattice`` holds the basis in the namespace it was built
    from, not the one the solve runs in. ``torch.asarray(..., dtype=numpy.dtype('float64'))``
    raises; the arithmetic that follows is fine either way, so it is only the *request* that
    has to be spelled in the namespace's own vocabulary.

    This was reachable before the centre moved (2026-08-31) and had no test: an omitted centre
    already resolved to a NumPy scalar, so a **painted** layer on torch whose shapes did not
    state their centres raised ``TypeError: asarray(): argument 'dtype' must be torch.dtype``
    out of ``rect_verts``. Every torch case in ``test_boundary.py`` spelled its centres out, so
    nothing reached it. Now that every shape carries the origin the branch is on the main road,
    which is what turned a latent failure into five red tests.
    """
    try:
        xp.asarray(0.0, dtype=dt)
    except Exception:                          # noqa: BLE001 - "not this namespace's dtype"
        return False
    return True


def _vertex_dtype(xp, *args):
    """The real dtype these vertices should be built in.

    Taken from the arguments when any of them is an array **of this namespace**, so a traced
    parameter carries its own precision through; ``float64`` otherwise.

    The fallback is the point. ``xp.asarray([1.0, ...])`` takes the *namespace default*, and
    torch's is ``float32`` no matter what ``complex_dtype`` the solver was given. A layer whose
    shape parameters are plain Python floats -- the ordinary case -- therefore had its geometry
    built in single precision on PyTorch alone, silently.
    """
    for a in args:
        dt = getattr(a, 'dtype', None)
        if dt is not None and _speaks(xp, dt):
            return dt
    return xp.float64


def _scalar(xp, v, dtype):
    """A shape parameter as a 0-d array of ``dtype``, whether it arrived as one or as a float.

    ``torch.cos`` refuses a Python float, and ``numpy.cos`` accepts one -- so a function that
    passes scalars straight to a ufunc works on NumPy and JAX and raises on PyTorch, for every
    input including ``angle = 0.0``. Promoting here is what makes the three namespaces
    interchangeable.

    An array of ``xp`` itself passes through untouched, since it may carry a graph (a JAX
    tracer, a Torch tensor that requires grad) and nothing here reads its value; everything
    else is a constant. The test is the namespace (:func:`~wavebit._common.backend.in_namespace`),
    not the presence of a ``dtype``: a NumPy scalar has one, and passing it through on that
    reading is what made every Torch build of an ellipse whose ``angle`` was
    ``numpy.float64(0.3)``, a 0-d array or an element read back out of a NumPy array raise
    ``TypeError`` from ``torch.cos`` -- a tensor shape layer, or a scalar one painted through
    the polygons, while NumPy and JAX solved it. A foreign value with a floating dtype keeps
    it, as NumPy itself keeps it: a ``numpy.float32`` angle takes its cosine in single
    precision on every backend, as it did on NumPy and JAX before, rather than being widened
    on two of the three. Any other foreign value -- a Python number, or a NumPy integer -- is
    built in ``dtype``. An integer dtype is not one to keep: Torch takes the cosine of an
    int64 tensor in float32, so a painted ellipse whose ``angle`` was ``numpy.int64(1)`` sat
    3.49e-10 from ``angle=1`` on Torch alone, where NumPy and JAX agreed to the bit.
    """
    if in_namespace(xp, v):
        return v
    if getattr(v, 'dtype', None) is not None:
        out = xp.asarray(v)
        if xp.isdtype(out.dtype, 'real floating'):
            return out
    return xp.asarray(v, dtype=dtype)


def rect_verts(xp, cx, cy, lx, ly, angle):
    """CCW vertices of a rotated rectangle. Exact -- a polygon *is* a rectangle."""
    dt = _vertex_dtype(xp, cx, cy, lx, ly, angle)
    one = xp.asarray([1.0, 1.0, 1.0, 1.0], dtype=dt)
    sx = xp.asarray([-0.5, 0.5, 0.5, -0.5], dtype=dt)
    sy = xp.asarray([-0.5, -0.5, 0.5, 0.5], dtype=dt)
    lx_, ly_ = sx * lx, sy * ly
    c, s = xp.cos(_scalar(xp, angle, dt) * one), xp.sin(_scalar(xp, angle, dt) * one)
    return xp.stack([cx + lx_ * c - ly_ * s, cy + lx_ * s + ly_ * c], axis=-1)


def polygon_verts(xp, vertices, cx, cy):
    """A declared polygon's vertices as an ``(N, 2)`` array of ``xp``, moved by ``(cx, cy)``.

    ``vertices`` is what a ``type='polygon'`` shape carries: an ``(N, 2)`` array of any
    namespace, or a sequence of ``N`` coordinate pairs, measured from the centre of the cell,
    which ``(cx, cy)`` is (:func:`~wavebit._common.geometry.recipes.cell_centre`). The result
    is in absolute Cartesian coordinates, as :func:`rect_verts` and :func:`ellipse_verts` are.

    **The declared order is kept, clockwise included.** The winding is read once, on concrete
    values, where the painting is planned (:func:`.clipper.resolve_painting`), and what a plan
    records is an index order -- so a traced outline is replayed in exactly the order it was
    planned on, and nothing here has to branch on the sign of an area.

    A coordinate of ``xp`` itself passes untouched and keeps its graph (a tracer, a tensor that
    requires grad); everything else is built in the dtype :func:`_vertex_dtype` picks -- float64
    for Python numbers, on Torch too, as a box's corners are.
    """
    if getattr(vertices, 'ndim', None) == 2:
        if in_namespace(xp, vertices):
            V = vertices if xp.isdtype(vertices.dtype, 'real floating') else \
                xp.astype(vertices, xp.float64)
        else:
            V = _scalar(xp, vertices, xp.float64)
            if not xp.isdtype(V.dtype, 'real floating'):
                V = xp.astype(V, xp.float64)
        dt = V.dtype
    else:
        coords = [c for pair in vertices for c in pair]
        dt = _vertex_dtype(xp, *coords, cx, cy)
        values = [_scalar(xp, c, dt) for c in coords]
        V = xp.reshape(xp.stack([v if v.dtype == dt else xp.astype(v, dt) for v in values]),
                       (len(vertices), 2))
    return V + xp.stack([_scalar(xp, cx, dt), _scalar(xp, cy, dt)])


def outline_values(vertices):
    """A declared outline as a plain float ``(N, 2)`` NumPy array -- what a check or a plan reads.

    Read the way :func:`~wavebit._common.backend.checked_float` reads a scalar, and drawing the
    same line: a Torch tensor is detached, a ``jax.grad`` tracer gives up its primal, and a
    ``jax.jit`` tracer has no number at all, which raises ``TypeError`` -- the signal every
    geometry check on the shape routes reads as *traced, skip* and every planner as *freeze
    first*. A layout that is not ``N`` coordinate pairs, or a value that is not real, raises
    ``ValueError``: that is about the outline, not about the trace.
    """
    import numpy as np

    def number(v):
        detach = getattr(v, 'detach', None)
        if detach is not None:
            v = detach()
            v = v.cpu() if hasattr(v, 'cpu') else v
        try:
            return np.asarray(v)
        except Exception:                      # noqa: BLE001 - a tracer of some kind
            pass
        if type(v).__module__.split('.')[0] == 'jax':
            import jax                         # local: jax is optional
            try:
                return np.asarray(jax.lax.stop_gradient(v))
            except Exception:                  # noqa: BLE001 - a jit tracer: no primal
                pass
        raise TypeError(f'{type(v).__name__} carries no value a check can read')

    if getattr(vertices, 'ndim', None) is not None:
        a = number(vertices)
    elif isinstance(vertices, (tuple, list)):
        rows = []
        for pair in vertices:
            if getattr(pair, 'ndim', None) is not None:
                rows.append(number(pair))
            elif isinstance(pair, (tuple, list)):
                rows.append(np.asarray([number(c) for c in pair]))
            else:
                raise ValueError(f'a vertex is a coordinate pair, got {type(pair).__name__}')
        if any(r.shape != (2,) for r in rows):
            raise ValueError('every vertex is an (x, y) pair')
        a = np.stack(rows) if rows else np.zeros((0, 2))
    else:
        raise ValueError(f'vertices are a sequence of (x, y) pairs or an (N, 2) array, got '
                         f'{type(vertices).__name__}')
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f'vertices are an (N, 2) array of (x, y) pairs, got shape {a.shape}')
    if a.dtype.kind == 'c':
        if np.any(a.imag != 0):
            raise ValueError('vertex coordinates are lengths, so they must be real')
        a = a.real
    if a.dtype.kind not in 'fiu':
        raise ValueError(f'vertex coordinates must be real numbers, got dtype {a.dtype}')
    return a.astype(float)


def polygon_defect(V):
    """Why the outline ``V`` is not a simple polygon with area -- or ``None`` when it is one.

    ``V`` is a concrete ``(N, 2)`` float array (:func:`outline_values`). Simple means what the
    transform and the clipper both need: at least three vertices, every one finite, no
    zero-length edge, no edge doubling back along its neighbour, no two other edges meeting
    anywhere -- crossing, touching or running along each other -- and a non-zero area. A bow tie
    is the case that matters: :func:`polygon_ft` would integrate it with winding numbers of
    both signs, so its lobes would cancel instead of adding, and the result is a legal-looking
    structure nobody drew.

    A vertex in the middle of a straight run (two collinear edges in the same direction) is
    allowed: the outline is still simple, and the transform of a straight edge is exact however
    it is split.

    The tolerance is rounding scale, ``1e-12`` of the outline's own span: the outline is checked
    as a figure, before any cell is known.
    """
    import numpy as np
    n = len(V)
    if n < 3:
        return f'it has {n} vertices; an outline needs at least three'
    bad = np.flatnonzero(~np.all(np.isfinite(V), axis=1))
    if bad.size:
        return f'vertex {bad[0]} is {tuple(V[bad[0]])}, not a finite point'
    span = float(max(np.ptp(V[:, 0]), np.ptp(V[:, 1])))
    if span == 0.0:
        return 'every vertex is the same point, so it encloses no area'
    eps = 1e-12 * span
    A, B = V, np.roll(V, -1, axis=0)
    E = B - A
    L = np.hypot(E[:, 0], E[:, 1])
    short = np.flatnonzero(L <= eps)
    if short.size:
        i = int(short[0])
        return f'vertices {i} and {(i + 1) % n} coincide, which leaves a zero-length edge'
    En, Ln = np.roll(E, -1, axis=0), np.roll(L, -1)
    cross = E[:, 0] * En[:, 1] - E[:, 1] * En[:, 0]
    back = np.flatnonzero((np.abs(cross) <= 1e-12 * L * Ln) & (np.sum(E * En, axis=1) < 0))
    if back.size:
        i = int(back[0])
        return f'edge {(i + 1) % n} doubles back along edge {i} at vertex {(i + 1) % n}'
    for i in range(n - 2):
        js = np.arange(i + 2, n if i else n - 1)
        if not js.size:
            continue
        d = _segment_distances(A[i], B[i], A[js], B[js])
        hit = np.flatnonzero(d <= eps)
        if hit.size:
            return (f'edges {i} and {int(js[hit[0]])} cross or touch, so the outline is not '
                    f'simple')
    if abs(0.5 * float(np.sum(A[:, 0] * B[:, 1] - B[:, 0] * A[:, 1]))) <= eps * span:
        return 'its vertices are collinear, so it encloses no area'
    return None


def _segment_distances(p, q, a, b):
    """Distance from segment ``pq`` to each segment ``a[k] b[k]``: zero where they cross."""
    import numpy as np

    def orient(o, s, t):
        return (s[..., 0] - o[..., 0]) * (t[..., 1] - o[..., 1]) - \
               (s[..., 1] - o[..., 1]) * (t[..., 0] - o[..., 0])

    def to_segment(x, s, t):
        d = t - s
        dd = np.sum(d * d, axis=-1)
        u = np.clip(np.sum((x - s) * d, axis=-1) / np.where(dd > 0, dd, 1.0), 0.0, 1.0)
        f = s + u[..., None] * d
        return np.hypot(*(x - f).T)

    p, q = np.broadcast_to(p, a.shape), np.broadcast_to(q, a.shape)
    crossing = (orient(p, q, a) * orient(p, q, b) < 0) & (orient(a, b, p) * orient(a, b, q) < 0)
    d = np.minimum(np.minimum(to_segment(p, a, b), to_segment(q, a, b)),
                   np.minimum(to_segment(a, p, q), to_segment(b, p, q)))
    return np.where(crossing, 0.0, d)


def ellipse_verts(xp, cx, cy, lx, ly, angle, n):
    """CCW polygonisation of an ellipse with ``n`` vertices, matched in **area**.

    Polygonised in the shape's own frame, *then* rotated -- so the approximation error is
    exactly rotation-covariant and cannot introduce angle-dependent error.

    **The vertices sit slightly outside the curve, on purpose.** Putting them *on* it inscribes
    the polygon, which is short of the true area by ``(2*pi**2/3)/n**2``. An area deficit is a
    deficit in the DC coefficient, so it does not stay in the high orders where a shape
    approximation belongs -- it lands on the *low* ones, exactly where the answer is decided.
    Scaling the semi-axes by ``sqrt(2*pi/(n*sin(2*pi/n)))`` makes the areas equal, and the same
    factor serves an ellipse of any aspect: the ``a*b`` cancels out of the ratio.

    Measured against the closed-form transform at ``truncation_order`` 11, a circle at ``n = 256``:
    inscribed **2.84e-05**, area-matched **8.43e-10**. Checked at aspect 1:1, 3:1 and 9:1.

    RETICOLO's ellipse texture (``retu.m``) likewise makes the inclusion's total area the
    ellipse's.
    """
    two_pi = 6.283185307179586
    dt = _vertex_dtype(xp, cx, cy, lx, ly, angle)
    t = xp.asarray([i * two_pi / n for i in range(n)], dtype=dt)
    import math
    scale = math.sqrt(two_pi / (n * math.sin(two_pi / n)))
    a, b = lx / 2.0 * scale, ly / 2.0 * scale
    px, py = a * xp.cos(t), b * xp.sin(t)
    ang = _scalar(xp, angle, dt)
    c, s = xp.cos(ang), xp.sin(ang)
    return xp.stack([cx + px * c - py * s, cy + px * s + py * c], axis=-1)
