"""The unit cell's lattice basis — the one place a non-orthogonal cell enters the solve.

A lattice is a 2x2 real matrix ``A`` whose **columns** are the in-plane basis vectors
``a1`` and ``a2``::

    A = [[a1x, a2x],
         [a1y, a2y]]

``period=(Lx, Ly)`` is the axis-aligned spelling of ``A = diag(Lx, Ly)`` and stays the
default; everything below reduces to the pre-existing expressions for that case, exactly,
which is what keeps the orthogonal path bit-identical.

**What the basis does and does not reach.** Bloch's theorem expands the field over
``G_mn = m*b1 + n*b2``, so the retained orders carry Cartesian components

    (kx, ky)_mn = k_inc + lambda * A^-T @ (m, n)

and *that is the whole of it*. The eigenproblem
(:func:`wavebit._core.solve_2d.layer_modes_2d`) is written in terms of diagonal ``Kx`` /
``Ky`` and the convolution matrices, so it never learns the lattice was skewed. Neither does
the Fourier machinery: substituting ``r = u*a1 + v*a2`` turns the cell integral into

    eps_hat[m, n] = \\int_0^1 \\int_0^1 eps(u, v) exp(-2i*pi*(m*u + n*v)) du dv

an ordinary DFT on the parametric ``(u, v)`` grid for *any* basis, still indexed by integer
differences and therefore still block-Toeplitz. That is why the ``fourier2d_*`` builders and
the convolution matrix builds in :mod:`wavebit._core.factorization` are untouched by
this module.

(``block-Toeplitz`` is the usual shorthand and is what matters here — the indexing by
differences survives the skew. The exact structure is BTTB; ``docs/factorization-terms.md`` §3
has the distinction and the check.)

The raster follows that substitution: ``structure[layer][j, i]`` is the permittivity at
``r = (i + 0.5)/Nx * a1 + (j + 0.5)/Ny * a2``. Axis 1 runs along ``a1`` (and pairs with
``truncation_order[0]`` / ``m``), axis 0 along ``a2`` (with ``truncation_order[1]`` / ``n``) — the same pairing
axis-aligned case has always used, read in lattice coordinates rather than Cartesian ones.

What the skew *does* reach is any quantity that is Cartesian-geometric rather than
index-based: see :func:`jacobian`, which is what the NVF normal field needs.
"""
import numpy as np

from .backend import as_array

__all__ = ['as_lattice', 'is_axis_aligned', 'jacobian', 'order_kxy', 'periods_of',
           'sample_grid']


def as_lattice(period=None, lattice=None):
    """The ``(2, 2)`` basis for a ``period`` / ``lattice`` pair, as a NumPy array.

    Exactly one of the two is expected; ``lattice`` wins if both are given, and ``period`` is
    then only a derived quantity (see :func:`periods_of`). Held eagerly in NumPy because it is
    a handful of numbers describing the cell, not a design variable — the k-vectors built from
    it in :func:`order_kxy` are what carry any gradient.
    """
    if lattice is not None:
        A = np.asarray(lattice, dtype=float)
        if A.shape != (2, 2):
            raise ValueError(
                f'lattice: expected a 2x2 matrix whose columns are the basis vectors a1, a2; '
                f'got shape {A.shape}')
        if not np.all(np.isfinite(A)):
            raise ValueError('lattice: entries must be finite')
        if abs(float(np.linalg.det(A))) < 1e-12 * float(np.linalg.norm(A)) ** 2:
            raise ValueError(
                'lattice: a1 and a2 are parallel (zero cell area), so the cell has no '
                'reciprocal basis')
        return A

    if period is None:
        raise ValueError('as_lattice: needs one of period or lattice')
    p = np.atleast_1d(np.asarray(period, dtype=float))
    if p.size == 1:
        p = np.array([p[0], p[0]])
    if p.size != 2:
        raise ValueError(f'period: expected 1 or 2 lengths, got {p.size}')
    return np.diag(p)


def is_axis_aligned(A, tol=1e-12):
    """Whether ``A`` is ``diag(Lx, Ly)`` — the case the tensor-product k-layout can express.

    Orthogonality alone is *not* enough. A basis rotated 45 degrees still has ``a1 . a2 == 0``
    but gives every order a ``kx`` that depends on both ``m`` and ``n``, which is precisely the
    layout that cannot be written as two 1-D vectors. Only a diagonal basis keeps them
    separable, so that is what this tests.
    """
    A = np.asarray(A, dtype=float)
    scale = float(np.max(np.abs(A)))
    if scale == 0:
        return True
    return bool(abs(A[0, 1]) <= tol * scale and abs(A[1, 0]) <= tol * scale)


def periods_of(A):
    """``(|a1|, |a2|)`` — the basis-vector lengths, which is what ``period`` reports.

    For an axis-aligned cell these are ``(Lx, Ly)`` and nothing has changed. For a skewed one
    they are the edge lengths of the parallelogram, which is the right thing for callers that
    want a physical extent per raster axis and the wrong thing for anyone wanting a bounding
    box — those need the basis itself.
    """
    A = np.asarray(A, dtype=float)
    return float(np.hypot(A[0, 0], A[1, 0])), float(np.hypot(A[0, 1], A[1, 1]))


def jacobian(A, ny, nx):
    """Map from raster index steps to Cartesian displacement: ``(2, 2)``, columns ``(di, dj)``.

    One step along axis 1 (``i``, the ``a1`` direction) moves ``a1 / nx``; one step along axis
    0 (``j``, ``a2``) moves ``a2 / ny``. So::

        J = [[a1x/nx, a2x/ny],
             [a1y/nx, a2y/ny]]

    A gradient carries the *inverse transpose* of this, not the Jacobian itself: for a scalar
    ``f`` sampled on the grid, ``grad_xy f = J^-T @ (df/di, df/dj)``. That distinction is the
    whole correction the NVF normal field needs on a skewed cell — see
    :func:`wavebit._core.factorization.normal_vector_field`.
    """
    A = np.asarray(A, dtype=float)
    return np.column_stack((A[:, 0] / nx, A[:, 1] / ny))


def sample_grid(xp, grid_shape, period, dtype, lattice=None):
    """Cartesian coordinates of the cell's pixel centres: two ``(Ny, Nx)`` arrays ``(X, Y)``.

    ``r = (i + 0.5)/Nx * a1 + (j + 0.5)/Ny * a2`` — the raster placement stated at the top of
    this module, evaluated as *positions* rather than read as indices. That is what the two
    non-raster normal-field builders need
    (:func:`wavebit._core.analytic.normal_vector_field_analytical`,
    :func:`wavebit._core.boundary.distance_gradient_normal_field`): both measure something
    Cartesian at each sample — analytic shape gradients on one side, distances to polygon edges
    on the other — so a skewed basis moves the sample points and nothing else. No metric follows.
    :func:`jacobian`, which converts index steps, is the raster path's correction and not theirs.

    Here rather than once per builder because ``period`` is a positional pair of two same-typed
    floats: unpacking it as ``(Py, Px)`` is a mistake nothing reacts to — not the types, not the
    runtime, and not a test on a square cell, where both names bind to the same number. It
    happened. The reversed unpack travelled from the analytic builder to the boundary one with a
    copy of this block, while the caller that reversed it back did not, so the boundary path
    sampled its field over a transposed cell on any non-square period (fixed 2026-08-11). One
    reader is one opportunity to get it wrong.

    ``period`` is read only when ``lattice`` is None, which is the rarer branch: ``engine``
    hands the shape paths ``as_lattice(period=...)`` whenever the period is *readable*
    (:meth:`~wavebit._core.engine.CoreFactorizationMixin._geometry_lattice_for`). Counting the
    calls says what that leaves — direct callers, and one solver case: a ``grad`` with respect to
    ``period`` itself, where the pair arrives as a tracer with no primal to read. Which is also
    why the pair stays a pair here instead of being folded into the basis: this branch has to
    keep tracing, and ``as_lattice`` converts to float. The ``lattice`` branch does too, so it
    holds a *constant* cell — fine, because a skewed cell is not a design variable, and the
    axis-aligned one is the one that can be.

    A check written on ``lattice=None`` alone therefore pins the branch the solver mostly does
    not take, and it takes two to hold the convention.
    """
    Ny, Nx = grid_shape
    u = (xp.asarray([float(i) for i in range(Nx)], dtype=dtype) + 0.5) / Nx
    v = (xp.asarray([float(i) for i in range(Ny)], dtype=dtype) + 0.5) / Ny
    if lattice is None:
        Px, Py = period[0], period[1]
        return (u * Px)[None, :] + 0.0 * v[:, None], (v * Py)[:, None] + 0.0 * u[None, :]
    a1x, a2x = float(lattice[0][0]), float(lattice[0][1])
    a1y, a2y = float(lattice[1][0]), float(lattice[1][1])
    return u[None, :] * a1x + v[:, None] * a2x, u[None, :] * a1y + v[:, None] * a2y


def order_kxy(xp, A, max_order_x, max_order_y, wavelength, k_inc_x, k_inc_y,
              dtype, device=None, centre=(0, 0)):
    """Per-order in-plane wavevectors on the ``(n, m)`` index grid: two
    ``(n_orders_y, n_orders_x)`` arrays.

    Both are normalized by ``k0`` — i.e. these are ``kx/k0``, matching what
    ``order_kxy`` returns — and laid out so that ``reshape(-1)`` gives the
    flat order list the convolution matrices are indexed by (``m`` fastest, matching
    ``_fourier2d_grid``'s final ``(n_orders_y, n_orders_x)`` reshape).

    The closed form, written out rather than taken from a matrix inverse so it stays one
    expression per component and differentiates cleanly::

        det = a1x*a2y - a2x*a1y
        kx  = k_inc_x + (lambda/det) * ( a2y*m - a1y*n)
        ky  = k_inc_y + (lambda/det) * (-a2x*m + a1x*n)

    For ``A = diag(Lx, Ly)`` that is ``k_inc_x + m*lambda/Lx`` and ``k_inc_y + n*lambda/Ly``,
    the existing expressions unchanged.
    """
    a1x, a2x = float(A[0, 0]), float(A[0, 1])
    a1y, a2y = float(A[1, 0]), float(A[1, 1])
    det = a1x * a2y - a2x * a1y

    cx, cy = int(centre[0]), int(centre[1])
    m = xp.asarray(np.arange(cx - max_order_x, cx + max_order_x + 1, dtype=float)[None, :],
                   dtype=dtype, device=device)
    n = xp.asarray(np.arange(cy - max_order_y, cy + max_order_y + 1, dtype=float)[:, None],
                   dtype=dtype, device=device)

    lam = as_array(xp, wavelength, dtype=dtype, device=device) / det
    kx = as_array(xp, k_inc_x, dtype=dtype, device=device) + lam * (a2y * m - a1y * n)
    ky = as_array(xp, k_inc_y, dtype=dtype, device=device) + lam * (-a2x * m + a1x * n)

    n_orders_x, n_orders_y = 2 * max_order_x + 1, 2 * max_order_y + 1
    return xp.broadcast_to(kx, (n_orders_y, n_orders_x)), xp.broadcast_to(ky, (n_orders_y, n_orders_x))
