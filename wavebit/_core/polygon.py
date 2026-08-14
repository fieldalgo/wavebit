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
above because the sign and factor-of-two conventions have to match
:func:`wavebit._core.factorization._fourier2d_grid` exactly, and getting ``sinc((k.e)/2)`` wrong as
``sinc(k.e)`` is invisible on axis-aligned rectangles.

:func:`polygon_area` is the shoelace formula (Gauss's area formula).

A hole needs no special case. The transform is a signed contour integral, so a loop wound the
other way subtracts by itself, and :mod:`.clipper` returns holes exactly that way.
"""


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


def polygon_ft(xp, verts, ky, kx, period, lattice=None):
    """1/area * FT of the polygon indicator, on the integer reciprocal grid (ky, kx).

    ``verts`` is (N, 2) in (x, y), counter-clockwise. Returns an array shaped like ``kx``.

    ``lattice`` is the cell basis (``None`` = axis-aligned ``diag(period)``), and as in
    :func:`wavebit._core.analytic.shape_ft` it reaches only the two lines that turn the order
    indices into ``(u, v)``. Everything after them is the contour integral in **Cartesian**
    frequency, so a skewed cell changes where the transform is sampled and nothing about the
    polygon. Vertices are absolute Cartesian positions, and the shape needs no wrapping at the
    cell edge either: ``exp(-i G.r)`` carries the lattice periodicity, so integrating over the
    polygon wherever it sits gives the right coefficient.
    """
    Px, Py = period[0], period[1]
    pi = 3.141592653589793
    if lattice is None:
        u = 2.0 * pi * kx / Px
        v = 2.0 * pi * ky / Py
        cell_area = Px * Py
    else:
        a1x, a2x = float(lattice[0][0]), float(lattice[0][1])
        a1y, a2y = float(lattice[1][0]), float(lattice[1][1])
        det = a1x * a2y - a2x * a1y
        u = 2.0 * pi * (a2y * kx - a1y * ky) / det
        v = 2.0 * pi * (-a2x * kx + a1x * ky) / det
        cell_area = abs(det)

    x, y = verts[:, 0], verts[:, 1]
    xn, yn = xp.roll(x, -1), xp.roll(y, -1)
    ex, ey = xn - x, yn - y                       # edge vectors
    mx, my = 0.5 * (x + xn), 0.5 * (y + yn)       # edge midpoints

    # Broadcast: (..., 1) frequency against (N,) edges.
    U, V = u[..., None], v[..., None]
    k_dot_e = U * ex + V * ey
    k_cross_n = U * ey - V * ex                   # (k . n_hat) * |e|
    terms = k_cross_n * xp.exp(-1j * (U * mx + V * my)) * _sinc(xp, 0.5 * k_dot_e)
    contour = xp.sum(terms, axis=-1)

    k2 = u * u + v * v
    nz = k2 != 0
    k2_safe = xp.where(nz, k2, xp.ones_like(k2))
    area = polygon_area(xp, verts)
    F = xp.where(nz, 1j * contour / k2_safe, (area + 0j) * xp.ones_like(contour))
    return F / cell_area


# ---------------------------------------------------------------- primitives

def _vertex_dtype(xp, *args):
    """The real dtype these vertices should be built in.

    Taken from the arguments when any of them is an array, so a traced parameter carries its
    own precision through; ``float64`` otherwise.

    The fallback is the point. ``xp.asarray([1.0, ...])`` takes the *namespace default*, and
    torch's is ``float32`` no matter what ``type_complex`` the solver was given. A layer whose
    shape parameters are plain Python floats -- the ordinary case -- therefore had its geometry
    built in single precision on PyTorch alone, silently.
    """
    for a in args:
        dt = getattr(a, 'dtype', None)
        if dt is not None:
            return dt
    return xp.float64


def _scalar(xp, v, dtype):
    """A shape parameter as a 0-d array of ``dtype``, whether it arrived as one or as a float.

    ``torch.cos`` refuses a Python float, and ``numpy.cos`` accepts one -- so a function that
    passes scalars straight to a ufunc works on NumPy and JAX and raises on PyTorch, for every
    input including ``angle = 0.0``. Promoting here is what makes the three namespaces
    interchangeable.
    """
    if getattr(v, 'dtype', None) is not None:
        return v
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

    Measured against the closed-form transform at ``fto`` 11, a circle at ``n = 256``:
    inscribed **2.84e-05**, area-matched **8.43e-10**. Checked at aspect 1:1, 3:1 and 9:1.

    This is what RETICOLO's ellipse texture has always done -- ``retu.m``, "la surface totale de
    l'inclusion etant celle de l'ellipse".
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
