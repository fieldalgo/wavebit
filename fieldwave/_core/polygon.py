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
:func:`fieldwave._core.factorization.dfs2d` exactly, and getting ``sinc((k.e)/2)`` wrong as
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


def _frequencies(kx, ky, period, lattice):
    """``(u, v, cell_area)`` — the Cartesian frequency each order index stands for."""
    Px, Py = period[0], period[1]
    pi = 3.141592653589793
    if lattice is None:
        return 2.0 * pi * kx / Px, 2.0 * pi * ky / Py, Px * Py
    a1x, a2x = float(lattice[0][0]), float(lattice[0][1])
    a1y, a2y = float(lattice[1][0]), float(lattice[1][1])
    det = a1x * a2y - a2x * a1y
    u = 2.0 * pi * (a2y * kx - a1y * ky) / det
    v = 2.0 * pi * (-a2x * kx + a1x * ky) / det
    return u, v, abs(det)


def edge_ft(xp, A, E, area, ky, kx, period, lattice=None, chunk=None):
    """1/area * FT of a region's indicator, from its **directed boundary segments**.

    ``A`` is (M, 2) segment starts and ``E`` = (M, 2) segment vectors, directed so the region
    lies on the *left* — equivalently, so the outward normal is ``(ey, -ex)``. ``area`` is the
    region's signed area, which supplies the one order the contour integral cannot: at ``k = 0``
    the divergence-theorem form is singular.

    **Segments, not a vertex ring, is the general statement.** The theorem needs ``dOmega``
    closed, not a polygon closed *in the plane*: a region whose boundary wraps the period closes
    on the torus, and ``exp(-i G.r)`` is a function on that torus for every reciprocal ``G``. So
    the sum below is over the boundary as a set of segments, and which loop a segment belongs to
    — or whether the loops even close without leaving the cell — never enters. That is what lets
    :mod:`.levelset` transform a connected, cell-spanning design with no polygon assembly at all.
    ``k = 0`` is the one place the loop structure would have mattered, and ``area`` carries it.

    ``lattice`` is the cell basis (``None`` = axis-aligned ``diag(period)``), and as in
    :func:`fieldwave._core.analytic.shape_ft` it reaches only the two lines that turn the order
    indices into ``(u, v)``. Everything after them is the contour integral in **Cartesian**
    frequency, so a skewed cell changes where the transform is sampled and nothing about the
    boundary.

    ``chunk`` bounds the ``orders x segments`` temporary. The sum is over segments and each term
    is independent, so a chunked run differs from a flat one only in float summation order.
    """
    u, v, cell_area = _frequencies(kx, ky, period, lattice)

    # Broadcast: (..., 1) frequency against (M,) segments.
    U, V = u[..., None], v[..., None]
    M = A.shape[0]
    n_orders = 1
    for s in u.shape:
        n_orders *= s
    if chunk is None:
        chunk = max(1, min(M, 4_000_000 // max(n_orders, 1)))

    contour = xp.zeros(u.shape, dtype=xp.complex128 if u.dtype == xp.float64
                       else xp.complex64)
    for lo in range(0, M, chunk):
        a, e = A[lo:lo + chunk], E[lo:lo + chunk]
        ex, ey = e[:, 0], e[:, 1]
        mx, my = a[:, 0] + 0.5 * ex, a[:, 1] + 0.5 * ey      # segment midpoints
        k_dot_e = U * ex + V * ey
        k_cross_n = U * ey - V * ex               # (k . n_hat) * |e|
        terms = k_cross_n * xp.exp(-1j * (U * mx + V * my)) * _sinc(xp, 0.5 * k_dot_e)
        contour = contour + xp.sum(terms, axis=-1)

    k2 = u * u + v * v
    nz = k2 != 0
    k2_safe = xp.where(nz, k2, xp.ones_like(k2))
    F = xp.where(nz, 1j * contour / k2_safe, (area + 0j) * xp.ones_like(contour))
    return F / cell_area


def polygon_ft(xp, verts, ky, kx, period, lattice=None):
    """1/area * FT of the polygon indicator, on the integer reciprocal grid (ky, kx).

    ``verts`` is (N, 2) in (x, y), counter-clockwise. Returns an array shaped like ``kx``.

    Vertices are absolute Cartesian positions, and the shape needs no wrapping at the cell edge:
    ``exp(-i G.r)`` carries the lattice periodicity, so integrating over the polygon wherever it
    sits gives the right coefficient.

    A closed vertex ring is the special case of :func:`edge_ft` where the last vertex joins the
    first *in the plane*, which is what ``roll`` says here.
    """
    x, y = verts[:, 0], verts[:, 1]
    xn, yn = xp.roll(x, -1), xp.roll(y, -1)
    E = xp.stack([xn - x, yn - y], axis=-1)
    return edge_ft(xp, verts, E, polygon_area(xp, verts), ky, kx, period, lattice)


# ---------------------------------------------------------------- primitives

def rect_verts(xp, cx, cy, lx, ly, angle):
    """CCW vertices of a rotated rectangle. Exact -- a polygon *is* a rectangle."""
    one = xp.asarray([1.0, 1.0, 1.0, 1.0])
    sx = xp.asarray([-0.5, 0.5, 0.5, -0.5])
    sy = xp.asarray([-0.5, -0.5, 0.5, 0.5])
    lx_, ly_ = sx * lx, sy * ly
    c, s = xp.cos(angle * one), xp.sin(angle * one)
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
    t = xp.asarray([i * two_pi / n for i in range(n)])
    import math
    scale = math.sqrt(two_pi / (n * math.sin(two_pi / n)))
    a, b = lx / 2.0 * scale, ly / 2.0 * scale
    px, py = a * xp.cos(t), b * xp.sin(t)
    c, s = xp.cos(angle), xp.sin(angle)
    return xp.stack([cx + px * c - py * s, cy + px * s + py * c], axis=-1)
