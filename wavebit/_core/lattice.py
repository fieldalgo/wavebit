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

The raster follows that substitution. ``structure[layer][j, i]`` sits at the parametric point
``u = (i + 0.5)/Nx``, ``v = (j + 0.5)/Ny``, and **the cell is centred at the origin**, so its
Cartesian position is

    r = (u - 0.5) * a1 + (v - 0.5) * a2

Axis 1 runs along ``a1`` (and pairs with ``truncation_order[0]`` / ``m``), axis 0 along ``a2``
(with ``truncation_order[1]`` / ``n``) — the same pairing the axis-aligned case has always
used, read in lattice coordinates rather than Cartesian ones.

The ``- 0.5`` is a relabeling and nothing else: the integral above runs over ``u, v`` in
``[0, 1]`` either way and the array is indexed the same, so **no coefficient moves**. What
changed on 2026-08-31 is where the origin *is*, and it changed for every coordinate this
library takes or hands back — a shape's ``cx`` / ``cy``, ``reconstruct_field``'s ``sample_x`` /
``sample_y``, and :func:`grid`. One origin, so that a centred feature is ``0`` everywhere and
nobody writes ``0.5*(a1 + a2)`` by hand.

What the skew *does* reach is any quantity that is Cartesian-geometric rather than
index-based: see :func:`jacobian`, which is what the NVF normal field needs.
"""
import dataclasses
import sys
from typing import Any

import numpy as np

from .backend import array_namespace, as_array

__all__ = ['Lattice', 'as_lattice', 'grid', 'hexagonal', 'is_axis_aligned', 'jacobian',
           'oblique', 'order_kxy', 'periods_of', 'rectangular', 'sample_grid']

_PYTREE_REGISTERED = False


@dataclasses.dataclass(frozen=True)
class Lattice:
    """The cell basis, with the one fact about it that must not be traced.

    ``u`` and ``v`` are the two primitive vectors, each ``(2,)`` — the same names and the same
    shape fmmax's ``LatticeVectors`` uses, so a reader crossing between them is reading one
    thing. ``columns`` derives the ``(2, 2)`` whose columns they are, which is what most of this
    package still wants; indexing delegates to it, so ``A[0][1]`` reads ``a2x`` as it always did.

    Named rather than positional because these are the two directions everything else in the
    package is named after: ``half_width_u`` counts orders along ``u``, ``n_u`` counts raster
    samples along it, and ``a1`` was only ever a synonym. ``A.u`` says which one it is;
    ``A[:, 0]`` needs the reader to know that columns are the vectors and rows are x and y.

    ``axis_aligned`` says whether the pair is ``diag(Lx, Ly)`` -- which is the case
    ``period=(Lx, Ly)`` spells. **``period`` is syntactic sugar for this class**, not a second
    way of describing a cell: it is the short form for the cell that has no angle to state, and
    :attr:`~wavebit._core.engine.CoreSolveMixin.lattice` materialises it. Reading ``period``
    back off a skewed cell gives ``periods_of`` -- the edge lengths, with the angle gone -- so
    the sugar only round-trips where it was the spelling to begin with.

    **Why those two are not the same kind of thing.** The numbers are a legitimate design
    variable -- optimising a hex angle or a shear is a real thing to want, and until this class
    existed ``lattice`` was eager NumPy so ``jax.grad`` could not reach it at all. But
    ``axis_aligned`` decides *which branch runs* (:func:`~wavebit._core.analytic.
    _shape_surface_normal` picks the rectangular fold from it, and it is what keeps the
    orthogonal path bit-identical), and a traced value cannot be branched on. Carried as a
    child it would kill every jitted solve while staying invisible eagerly -- the same trap
    ``Material`` documents from the other direction, where a value that belonged in a child was
    frozen into aux_data.

    So: numbers are the pytree **child**, ``axis_aligned`` is **aux_data**, and it is settled
    once at ingest where the answer is still readable. When it is not readable -- a basis handed
    in already traced -- it is ``False``, which is the *general* path and therefore always
    correct, merely not bit-identical to the rectangular shortcut. A ``period`` is diagonal by
    construction, so that spelling stays ``True`` even under a tracer.

    Indexing delegates to ``columns``, so ``A[0][1]`` reads the same as it did when this was a
    bare array.
    """

    u: Any
    v: Any
    axis_aligned: bool

    def __post_init__(self):
        _register_pytree_if_jax_loaded()

    @property
    def columns(self):
        """``(2, 2)`` with ``u`` and ``v`` as its columns — the spelling the solve reads."""
        xp = np if isinstance(self.u, np.ndarray) else array_namespace(self.u)
        return xp.stack([self.u, self.v], axis=-1)

    def __getitem__(self, key):
        return self.columns[key]

    def __repr__(self):
        u, v = (np.asarray(x, dtype=float) for x in (self.u, self.v))
        if self.axis_aligned:                       # what `period=(Lx, Ly)` spells
            return f'Lattice(diag({u[0]:g}, {v[1]:g}))'
        return (f'Lattice(u=({u[0]:g}, {u[1]:g}), v=({v[0]:g}, {v[1]:g}), '
                f'|u|={float(np.hypot(*u)):g}, |v|={float(np.hypot(*v)):g}, '
                f'angle={float(np.degrees(np.arccos(np.clip(u @ v / (np.hypot(*u) * np.hypot(*v)), -1, 1)))):.4g} deg)')

    def __array__(self, dtype=None, copy=None):
        a = np.asarray(self.columns, dtype=float)
        return a if dtype is None else a.astype(dtype)

    # pytree protocol -- registered lazily; see _register_pytree_if_jax_loaded
    def tree_flatten(self):
        return (self.u, self.v), (self.axis_aligned,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], children[1], aux[0])


def _register_pytree_if_jax_loaded():
    """Register :class:`Lattice` as a JAX pytree, if JAX is in the process already.

    Lazy for the reason ``backends/jax_backend.py`` is imported only on use: ``import wavebit``
    must not import JAX
    for a caller who will never use it, and importing it here flips ``jax_enable_x64``
    process-wide. Reads ``sys.modules`` and never imports -- the same pattern as
    :func:`wavebit._core.domain._register_pytree_if_jax_loaded`, which the docstring there
    explains at length.
    """
    global _PYTREE_REGISTERED
    if _PYTREE_REGISTERED:
        return
    jax = sys.modules.get('jax')
    if jax is None:
        return
    tree_util = getattr(jax, 'tree_util', None)
    if tree_util is None:
        return                      # jax is mid-import; the next Lattice will try again
    try:
        tree_util.register_pytree_node(Lattice, Lattice.tree_flatten, Lattice.tree_unflatten)
    except ValueError:
        pass                        # already registered; idempotent by design
    _PYTREE_REGISTERED = True


def as_lattice(period=None, lattice=None):
    """The ``(2, 2)`` basis for a ``period`` / ``lattice`` pair, as a NumPy array.

    Exactly one of the two is expected; ``lattice`` wins if both are given, and ``period`` is
    then only a derived quantity (see :func:`periods_of`). Held eagerly in NumPy because it is
    a handful of numbers describing the cell, not a design variable — the k-vectors built from
    it in :func:`order_kxy` are what carry any gradient.
    """
    if lattice is not None:
        if isinstance(lattice, Lattice):
            return lattice
        try:
            A = np.asarray(lattice, dtype=float)
        except Exception:                       # noqa: BLE001 - a tracer, not a mistake
            # A basis handed in already traced. Its numbers are the point -- that is the whole
            # reason this is a pytree -- but `axis_aligned` cannot be read off a tracer, so it
            # takes the general path. Always correct, merely not the rectangular shortcut.
            cols = lattice
            return Lattice(cols[:, 0], cols[:, 1], False)
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
        return Lattice(A[:, 0], A[:, 1], is_axis_aligned(A))

    if period is None:
        raise ValueError('as_lattice: needs one of period or lattice')
    try:
        p = np.atleast_1d(np.asarray(period, dtype=float))
    except Exception:                       # noqa: BLE001 - a traced period, not a mistake
        # `period` is the variable under differentiation. Its *shape* is still a diagonal, so
        # `axis_aligned` is known without reading a number -- which is the whole reason that
        # flag is aux_data and not a child.
        # A tuple or list of tracers is a natural spelling -- `period=(P, P)` under
        # `jax.grad` -- and `array_namespace` cannot read one, so ask its elements.
        xp = (array_namespace(*period) if isinstance(period, (tuple, list))
              else array_namespace(period))
        q = period if getattr(period, 'shape', (1,))[-1] == 2 else xp.stack([period[0], period[0]])
        zero = 0.0 * q[0]
        return Lattice(xp.stack([q[0], zero]), xp.stack([zero, q[1]]), True)
    if p.size == 1:
        p = np.array([p[0], p[0]])
    if p.size != 2:
        raise ValueError(f'period: expected 1 or 2 lengths, got {p.size}')
    D = np.diag(p)
    return Lattice(D[:, 0], D[:, 1], True)   # a period spells a diagonal; nothing to test


def is_axis_aligned(A, tol=1e-12):
    """Whether ``A`` is ``diag(Lx, Ly)`` — the case the tensor-product k-layout can express.

    Orthogonality alone is *not* enough. A basis rotated 45 degrees still has ``a1 . a2 == 0``
    but gives every order a ``kx`` that depends on both ``m`` and ``n``, which is precisely the
    layout that cannot be written as two 1-D vectors. Only a diagonal basis keeps them
    separable, so that is what this tests.
    """
    if isinstance(A, Lattice):
        return A.axis_aligned
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
    cols = A.columns if isinstance(A, Lattice) else A
    try:
        M = np.asarray(cols, dtype=float)
    except Exception:                       # noqa: BLE001 - a tracer, not a mistake
        xp = array_namespace(cols)
        return (xp.hypot(cols[0, 0], cols[1, 0]), xp.hypot(cols[0, 1], cols[1, 1]))
    return float(np.hypot(M[0, 0], M[1, 0])), float(np.hypot(M[0, 1], M[1, 1]))


def jacobian(A, n_v, n_u):
    """Map from raster index steps to Cartesian displacement: ``(2, 2)``, columns ``(di, dj)``.

    One step along axis 1 (``i``, the ``a1`` direction) moves ``a1 / n_u``; one step along axis
    0 (``j``, ``a2``) moves ``a2 / n_v``. So::

        J = [[a1x/n_u, a2x/n_v],
             [a1y/n_u, a2y/n_v]]

    A gradient carries the *inverse transpose* of this, not the Jacobian itself: for a scalar
    ``f`` sampled on the grid, ``grad_xy f = J^-T @ (df/di, df/dj)``. That distinction is the
    whole correction the NVF normal field needs on a skewed cell — see
    :func:`wavebit._core.factorization.normal_vector_field`.
    """
    cols = A.columns if isinstance(A, Lattice) else A
    try:
        M = np.asarray(cols, dtype=float)
    except Exception:                       # noqa: BLE001 - a tracer, not a mistake
        xp = array_namespace(cols)
        return xp.stack([cols[:, 0] / n_u, cols[:, 1] / n_v], axis=-1)
    return np.column_stack((M[:, 0] / n_u, M[:, 1] / n_v))


def sample_grid(xp, grid_shape, lattice, dtype):
    """Pixel-centre coordinates in the **internal** frame: two ``(Ny, Nx)`` arrays ``(X, Y)``.

    ``r = (i + 0.5)/Nx * a1 + (j + 0.5)/Ny * a2`` -- the cell's *corner* at the origin, which
    is the frame the Fourier machinery works in and the one
    :func:`~wavebit._core.analytic.shape_geometry` resolves a shape's centre into. **The public
    door is :func:`grid`**, which re-bases to the cell centre; the two differ by
    ``0.5*(a1 + a2)``, and reaching for this one from user-facing code misplaces a structure by
    half a period with nothing to say so.

    Evaluated as *positions* rather than read as indices. That is what the two
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
    a1x, a2x = lattice[0][0], lattice[0][1]
    a1y, a2y = lattice[1][0], lattice[1][1]
    return u[None, :] * a1x + v[:, None] * a2x, u[None, :] * a1y + v[:, None] * a2y


def order_kxy(xp, A, half_width_u, half_width_v, wavelength, k_inc_x, k_inc_y,
              dtype, device=None, centre=(0, 0)):
    """Per-order in-plane wavevectors on the ``(n, m)`` index grid: two
    ``(n_orders_v, n_orders_u)`` arrays.

    Both are normalized by ``k0`` — i.e. these are ``kx/k0``, matching what
    ``order_kxy`` returns — and laid out so that ``reshape(-1)`` gives the
    flat order list the convolution matrices are indexed by (``m`` fastest, matching
    ``_fourier2d_grid``'s final ``(n_orders_v, n_orders_u)`` reshape).

    The closed form, written out rather than taken from a matrix inverse so it stays one
    expression per component and differentiates cleanly::

        det = a1x*a2y - a2x*a1y
        kx  = k_inc_x + (lambda/det) * ( a2y*m - a1y*n)
        ky  = k_inc_y + (lambda/det) * (-a2x*m + a1x*n)

    For ``A = diag(Lx, Ly)`` that is ``k_inc_x + m*lambda/Lx`` and ``k_inc_y + n*lambda/Ly``,
    the existing expressions unchanged.
    """
    # Read, not converted: `float()` here is what kept a basis from ever being a design
    # variable. A NumPy scalar arithmetics identically, so the orthogonal path is unmoved.
    a1x, a2x = A[0, 0], A[0, 1]
    a1y, a2y = A[1, 0], A[1, 1]
    det = a1x * a2y - a2x * a1y

    cx, cy = int(centre[0]), int(centre[1])
    m = xp.asarray(np.arange(cx - half_width_u, cx + half_width_u + 1, dtype=float)[None, :],
                   dtype=dtype, device=device)
    n = xp.asarray(np.arange(cy - half_width_v, cy + half_width_v + 1, dtype=float)[:, None],
                   dtype=dtype, device=device)

    lam = as_array(xp, wavelength, dtype=dtype, device=device) / det
    kx = as_array(xp, k_inc_x, dtype=dtype, device=device) + lam * (a2y * m - a1y * n)
    ky = as_array(xp, k_inc_y, dtype=dtype, device=device) + lam * (-a2x * m + a1x * n)

    # Closed with a cast, as `_separable_kxy` is. The basis is read rather than converted --
    # that is what lets it be a design variable -- so `det` arrives at whatever width the caller
    # built the cell in, in practice float64. Under NEP 50 a `complex64 / float64` is a
    # `complex128`, so `lam` widened and carried the whole return with it: a skewed cell
    # answered in double while `complex_dtype=1` was asked for, silently, and the float32
    # branch-resolution guard that fires on the rectangular path did not fire here.
    n_orders_u, n_orders_v = 2 * half_width_u + 1, 2 * half_width_v + 1
    return (xp.astype(xp.broadcast_to(kx, (n_orders_v, n_orders_u)), dtype),
            xp.astype(xp.broadcast_to(ky, (n_orders_v, n_orders_u)), dtype))


def grid(shape, period=None, lattice=None, backend=0, dtype=None):
    """Cartesian coordinates of a raster's pixel centres: ``(X, Y)``, both shaped ``shape``.

    The public way to build a structure without deciding an axis order. Write the layer in terms
    of the coordinates and the layout follows on its own::

        X, Y = wavebit.grid((64, 96), period=[700., 1100.])
        layer = np.where(X ** 2 + Y ** 2 < 200. ** 2, 12.25, 1.0)

    ``np.where`` is elementwise, so ``layer`` simply inherits whatever layout ``X`` and ``Y``
    already had. **That is the whole point**: an expression written in *coordinates* cannot get
    the axis order wrong, while one written in *indices* (``layer[:2] = ...``) can, and does so
    silently on a non-square raster. Hand-rolling the grid is what pushes callers towards
    indices; this exists so they need not.

    **The cell is centred at the origin**, the same origin an analytic shape's ``cx`` / ``cy``
    and ``reconstruct_field``'s ``sample_x`` / ``sample_y`` use, so the rod above is centred by
    writing nothing. It read ``(X - 350.)**2 + (Y - 550.)**2`` until 2026-08-31 -- the caller
    computed the centre of the cell, which on a skewed cell is ``0.5*(a1 + a2)``, two
    components neither of which is a period. That line appeared in the README, in
    ``docs/oblique-lattice.md``, in a tutorial and in eight helpers; one origin deletes all of
    them. :func:`sample_grid` is the internal corner-based twin and is not this function.

    A skewed cell comes free. ``lattice=`` places the samples on the parallelogram, so the same
    coordinate expression describes the same physical shape whatever the cell -- where writing
    ``meshgrid`` by hand means writing ``u*a1 + v*a2`` by hand too, once per caller. The
    equivalent elsewhere in this field is ``numpy.meshgrid(..., indexing='ij')``: fmmax's
    examples spell that out eleven times because nothing hands it to them, and forgetting it
    transposes the structure with nothing to say so.

    ``shape`` is ``(rows, columns)``, read as numpy reads it. Rows run along ``a2`` and columns
    along ``a1`` -- y and x on a rectangular cell, and neither of those on a skewed one, which
    is why the return is coordinates rather than axis names.

    Exactly one of ``period`` / ``lattice``; ``backend`` matches :func:`wavebit.rcwa`
    (0 NumPy, 1 JAX, 2 PyTorch). ``dtype`` defaults to the backend's float64.
    """
    A = as_lattice(period=period, lattice=lattice)
    shape = tuple(int(v) for v in shape)
    if len(shape) != 2:
        raise ValueError(f'grid: shape must be (rows, columns); got {shape!r}')
    if min(shape) < 1:
        raise ValueError(f'grid: shape must be positive; got {shape!r}')

    if backend == 0:
        import array_api_compat.numpy as xp
    elif backend == 1:
        import jax.numpy as xp
    elif backend == 2:
        import array_api_compat.torch as xp
    else:
        raise ValueError(f'grid: backend must be 0 (NumPy), 1 (JAX) or 2 (PyTorch); got {backend!r}')

    X, Y = sample_grid(xp, shape, A, dtype if dtype is not None else xp.float64)
    # The one place the public frame is put on. `sample_grid` builds the cell from its corner
    # because that is what the transform integrates over; a caller sees it from the centre.
    return X - 0.5 * (A[0][0] + A[0][1]), Y - 0.5 * (A[1][0] + A[1][1])


def rectangular(lx, ly=None):
    """``diag(lx, ly)`` — the axis-aligned cell, spelled as a basis. ``ly`` defaults to ``lx``."""
    return as_lattice(period=[lx] if ly is None else [lx, ly])


def oblique(lx, ly, angle):
    """A cell given by two edge lengths and the **angle between them**, in radians.

    ``a1`` lies along x with length ``lx``; ``a2`` has length ``ly`` and makes ``angle`` with
    it, so::

        A = [[lx, ly*cos(angle)],
             [ 0, ly*sin(angle)]]

    Radians, because every other angle in this API is (``theta``, ``phi``, a shape's ``angle``).
    ``np.radians(60)`` for the usual ones, or :func:`hexagonal` for the usual-est.

    This is a *spelling*, not a new kind of cell -- it returns the same :class:`Lattice` that
    ``lattice=`` does, and there is no ``angle`` stored anywhere afterwards. Which matters for
    differentiation: build it inside the function you differentiate and the angle is a design
    variable, because the basis it produces is one. Build it outside and you have handed the
    solver a constant, exactly as with any other expression.

    ``angle`` near 0 or pi is refused rather than returning a basis with no cell area -- the
    same refusal :func:`as_lattice` makes for parallel columns, reached earlier and by name.

    A right angle comes back *as* a right angle -- ``axis_aligned`` is read off the basis, not
    assumed -- but to 1e-16 and not to the bit, since ``cos(pi/2)`` is 6.1e-17. Use
    :func:`rectangular` where an exact diagonal is wanted, which is every case that does not
    have an angle to vary.
    """
    xp = array_namespace(angle) if not isinstance(angle, (int, float)) else np
    a = as_array(xp, angle) if xp is not np else float(angle)
    if xp is np and not (1e-9 < abs(float(np.sin(a))) ):
        raise ValueError(
            f'oblique: angle={angle!r} rad puts a1 and a2 on the same line, so the cell has no '
            f'area. Use a value strictly between 0 and pi (np.radians(60) for a hex rhombus).')
    cos_a, sin_a = xp.cos(a), xp.sin(a)
    zero = 0.0 * cos_a
    cols = xp.stack([xp.stack([lx + zero, ly * cos_a]),
                     xp.stack([zero, ly * sin_a])])
    if xp is np:
        M = np.asarray(cols, dtype=float)
        return Lattice(M[:, 0], M[:, 1], is_axis_aligned(M))    # a right angle says so
    return Lattice(cols[:, 0], cols[:, 1], False)   # traced: the general path, always correct


def hexagonal(a, angle=2.0 * np.pi / 3.0):
    """A primitive cell of a hexagonal array: the rhombus of side ``a``, **120 degrees**.

    120 is the crystallographic standard -- the hexagonal system is defined with
    ``a = b, alpha = beta = 90, gamma = 120`` -- and the reason is not taste. 60 spans the same
    lattice, but *the three-fold rotation axis is not visible in the primitive vectors there*:
    the 120 pair maps into itself under a 120 rotation, and the 60 pair does not. That is why
    the standard settings pick it, and it is why this is the default.

    60 is a legitimate alternative basis, the one that makes the primitive cell two equilateral
    triangles, and it is what ``docs/oblique-lattice.md`` and the S4 cross-check in it are
    written on. ``hexagonal(a, np.radians(60))``.

    They generate the **same lattice** -- ``a2' = a2 + a1`` carries one to the other -- but they
    are not the same calculation:

    * the *fundamental cell* differs, so an omitted shape centre lands elsewhere: ``(100, 173)``
      against ``(300, 173)`` at ``a = 400``. See :func:`~wavebit._core.analytic.shape_geometry`.
    * ``truncation_order`` cuts a parallelogram in ``(m, n)``, and a different basis maps that
      onto a different set of ``G``. Measured at ``truncation_order=4``: 81 orders either way,
      **61 of them shared**. The two agree in the converged limit and not before it.

    So this is a choice with consequences at finite truncation, which is why it is an argument
    rather than a constant folded into the name.
    """
    return oblique(a, a, angle)
