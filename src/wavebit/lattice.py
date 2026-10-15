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
(:func:`wavebit.rcwa._engine.solve_2d.layer_modes_2d`) is written in terms of diagonal ``Kx`` /
``Ky`` and the convolution matrices, so it never learns the lattice was skewed. Neither does
the Fourier machinery: substituting ``r = u*a1 + v*a2`` turns the cell integral into

    eps_hat[m, n] = \\int_0^1 \\int_0^1 eps(u, v) exp(-2i*pi*(m*u + n*v)) du dv

an ordinary DFT on the lattice ``(u, v)`` grid for *any* basis, still indexed by integer
differences and therefore still block-Toeplitz. That is why the ``fourier2d_*`` builders and
the convolution matrix builds in :mod:`wavebit.rcwa._engine.factorization` are untouched by
this module.

(``block-Toeplitz`` is the usual shorthand and is what matters here — the indexing by
differences survives the skew. The exact structure is BTTB; ``docs/theory/rcwa/factorization-terms.md`` §3
has the distinction and the check.)

The raster follows that substitution. ``structure[layer][j, i]`` sits at the lattice point
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
``sample_y``, and :func:`meshgrid_xy`. One origin, so that a centred feature is ``0`` everywhere and
nobody writes ``0.5*(a1 + a2)`` by hand.

What the skew *does* reach is any quantity that is Cartesian-geometric rather than
index-based: see :func:`jacobian`, which is what the NVF normal field needs.
"""
import dataclasses
import sys
from typing import Any

import numpy as np

from wavebit._common.backend import array_namespace
from wavebit._common.backend import as_array
from wavebit._common.backend import carries_autodiff
from wavebit._common.backend import checked_float
from wavebit._common.backend import constant_array
from wavebit._common.backend import lift_jax_x64
from wavebit._common.enums import Backend
from wavebit._common.enums import backend_of
from wavebit._common.backend import eager_under_torch_compile

__all__ = ['DrawnOn', 'Lattice', 'as_lattice', 'meshgrid_xy', 'is_axis_aligned', 'jacobian',
           'order_kxy', 'periods_of', 'reduced_basis', 'reduction_transform', 'sample_grid']

_PYTREE_REGISTERED = False


@eager_under_torch_compile
@dataclasses.dataclass(frozen=True)
class DrawnOn:
    """A raster together with the cell it was drawn on -- what :meth:`Lattice.drawn` returns.

    **Not an ndarray subclass, deliberately.** A subclass was tried first and it only survives
    on NumPy: ``jnp.asarray`` and ``torch.as_tensor`` copy the numbers into their own type and
    the Python attribute is gone, so the check would have been live on one backend and silently
    absent on the other two. A wrapper does not have that problem because **it never travels**:
    :attr:`~wavebit.rcwa._engine.engine.CoreSolveMixin.structure` takes it apart at ingest, keeps the
    array exactly as it always did, and keeps the basis in a slot beside it. Nothing downstream
    -- no backend, no builder, no document -- ever sees this type.

    ``values`` is whatever the caller handed over, layer axis included; the basis is not read
    from it and no shape is assumed.
    """

    values: Any
    lattice: Any


@eager_under_torch_compile
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
    :attr:`~wavebit.rcwa._engine.engine.CoreSolveMixin.lattice` materialises it. Reading ``period``
    back off a skewed cell gives ``periods_of`` -- the edge lengths, with the angle gone -- so
    the sugar only round-trips where it was the spelling to begin with.

    **Why those two are not the same kind of thing.** The numbers are a legitimate design
    variable -- optimising a hex angle or a shear is a real thing to want, and until this class
    existed ``lattice`` was eager NumPy so ``jax.grad`` could not reach it at all. But
    ``axis_aligned`` decides *which branch runs* (:func:`~wavebit.rcwa._engine.parametric.
    _round_field` picks the rectangular fold from it, and it is what keeps the
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

    def meshgrid_xy(self, grid_shape, backend='numpy', dtype=None):
        """The cell's sample points as Cartesian ``(X, Y)`` -- ``numpy.meshgrid`` on this cell.

        Identical to ``wavebit.meshgrid_xy(grid_shape, lattice=A)`` in every respect, and it exists
        for the
        naming and nothing else: the lattice appears **twice** in a raster workflow, once here
        and once at ``lattice=`` on the solver, and the only way that goes wrong is the two
        disagreeing. Measured on a 64-px raster of a radius-100 circle in a 640-unit cell,
        drawn at one cell and solved at another::

            grid's cell   solve's cell   what is actually solved
            -----------   ------------   -----------------------
            90 deg        90 deg         a circle,   1.00:1
            90 deg        20 deg         an ellipse, 5.67:1
            20 deg        90 deg         an ellipse, 5.60:1
            20 deg        20 deg         a circle,   1.01:1

        Only the diagonal is a circle. The *angle* is irrelevant -- **sameness** is the whole
        rule -- and the off-diagonal cases are the dangerous ones, because the top-right one is
        exactly what a caller who swaps ``lattice=`` and reuses the array gets: ``sum(T)`` moves
        0.157 -> 0.997 with no error, no warning, and energy conserved throughout. Pinned as
        ``tests/common/test_lattice_pytree.py::test_the_two_uses_of_a_lattice_have_to_be_the_same_one``.

        **A nudge, not a guarantee, and it cannot be made one.** ``wavebit.meshgrid_xy(
        grid_shape, lattice=B)`` still exists and still answers; this only makes the agreeing spelling the
        shorter one. Nothing downstream can catch the disagreement either, because what comes
        back is a pair of ordinary coordinate arrays and the first ``np.where`` consumes them --
        a raster is numbers on indices, and which cell it was drawn in is not among them. So the
        rule the naming is meant to carry has to be stated too: **change the cell and the array
        has to be drawn again.**
        """
        return meshgrid_xy(grid_shape, lattice=self, backend=backend, dtype=dtype)

    def meshgrid_uv(self, grid_shape, backend='numpy', dtype=None):
        """The same sample points as **lattice** ``(U, V)`` -- fractions of the cell, in [-0.5, 0.5].

        The other half of the pair, and the reason both are spelled with a suffix: one mesh, two
        frames to express it in. The points are the same points either way -- both lie in the
        cell -- and what differs is the units: ``meshgrid_xy`` gives them as **Cartesian
        lengths**, which move when the basis does, and this gives them as **fractions of the
        cell**, which do not.

        **It does not read the basis, and that is the whole point.** A pattern written on ``U``
        and ``V`` is a set of *fractions* -- a duty cycle, a fill fraction, a design scaled as a
        whole -- so the same array stays correct on every cell, and reusing it across a change
        of ``lattice=`` is right rather than the mistake it would be in the Cartesian frame.
        Scale a design and the answer must not move: measured, the held array gives
        ``sum(T) = 0.124231941`` at 0.5x, 1x, 2x and 4x, identical to nine digits.

        Written out by hand this is ``(np.arange(n) + 0.5)/n - 0.5`` per axis, then a
        ``meshgrid``; the tests in this repository spell that out in eight places, which is why
        it is here. ``docs/theory/rcwa/oblique-lattice.md``, "The two frames a raster can mean", is the
        write-up.
        """
        A = as_lattice(period=[1.0, 1.0])          # the unit square: u, v are the basis-free half
        return A.meshgrid_xy(grid_shape, backend=backend, dtype=dtype)

    def drawn(self, values):
        """Declare that ``values`` was drawn on **this** cell, so solving it on another refuses.

        The Cartesian frame is the half of "The two frames a raster can mean" that has a
        checkable invariant: a pattern written on :meth:`meshgrid_xy` is a set of absolute
        lengths, so it belongs to the cell it was written on and has to be rebuilt when that
        cell moves. The lattice frame has nothing to check -- an array of *fractions* is
        supposed to be portable -- so it stays what it has always been: a bare array, accepted
        anywhere, and this is not needed for it::

            X, Y = lattice.meshgrid_xy((256, 256))
            layer = np.where(X ** 2 + Y ** 2 < 100. ** 2, 12.25, 1.0)
            wavebit.rcwa(lattice=lattice, structure=lattice.drawn(layer[None]), ...)

        Hand it the finished structure, layer axis and all, because it wraps rather than
        forwards: there is no indexing left to do afterwards.

        **What it buys, measured.** A radius-100 circle drawn on a 640 square and solved on
        ``hexagonal(640.)`` returns ``sum(T) = 0.252202`` where redrawing gives 0.352836 -- no
        error, no warning, energy conserved, and it reads as the lattice effect the caller was
        asking for. That is now a refusal.

        **It is opt-in and cannot be otherwise.** A bare array has to keep working, so the
        guard exists only for callers who ask for it; and the tag cannot be attached to the
        coordinates instead, because ``np.where`` -- which every raster expression in these
        docs ends with -- drops it. Applying it after the expression is the only placement that
        works.

        **A parametric structure is refused rather than wrapped.** That path has no frame to
        declare: a shape's ``lx`` and ``cx`` are absolute Cartesian lengths, so a circle stays
        the same circle on every cell and moving the basis is a legitimate thing to do with the
        same structure. Wrapping one and solving it elsewhere would refuse an answer that is
        *correct* -- measured, two shapes on a 640 square give 0.999233 and the identical
        structure on ``hexagonal(640.)`` gives 0.999851, both right -- and a guard that blocks
        correct work is worse than no guard at all.
        """
        if isinstance(values, list):
            raise ValueError(
                'drawn: this is the parametric spelling (a list of layer dicts), and it has no '
                'frame to declare -- a shape\'s `lx` and `cx` are absolute Cartesian lengths, '
                'so it describes the same geometry on every cell and may be solved on any of '
                'them. Pass it to `structure=` directly.\n'
                '`drawn` is for a *raster*, whose pixels are fractions of the cell and so '
                'belong to the one they were drawn on.')
        return DrawnOn(values, self.columns)

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

    def tolist(self):
        """The ``(2, 2)`` as nested lists -- the array form, in plain Python.

        Beside :meth:`__array__` because it is the same view of the same numbers, and it earns
        its place at the *document* boundary: :func:`wavebit.compat._jobs_v1._jsonable` recognises a
        value it can write by ``hasattr(value, 'tolist')``, so without this a cell built the
        documented way -- ``Stack.from_structure(..., lattice=wavebit.Lattice.hexagonal(400.))`` --
        was accepted by ``Stack.validate`` and then refused by ``to_json`` as "value of type
        Lattice is not JSON-representable". Accepting what cannot be written is the failure
        mode; a plain 2x2 always worked.

        ``axis_aligned`` is not carried and does not need to be: it is read back off the basis
        at ingest, which is where it is settled in the first place. So the round trip returns a
        nested list rather than a ``Lattice``, exactly as it already does for an ndarray.
        """
        return np.asarray(self.columns, dtype=float).tolist()

    @classmethod
    def from_columns(cls, matrix):
        """The cell a ``(2, 2)`` spells, its columns being ``a1`` and ``a2``.

        The way into this type for a basis written out by hand, which the docs bless and the
        solver has always accepted at ``lattice=``. Without it a hand-written cell was a bare
        array, so it had no :meth:`meshgrid_xy` and the "name the basis once" spelling was unavailable
        to exactly the callers writing the most explicit code.

        Identical to what ``lattice=<2x2>`` does at ingest -- ``axis_aligned`` is read off the
        matrix, so a diagonal reports itself as one.
        """
        return as_lattice(lattice=matrix)

    @classmethod
    def rectangular(cls, lx, ly=None):
        """``diag(lx, ly)`` — the axis-aligned cell, spelled as a basis. ``ly`` defaults to ``lx``."""
        return as_lattice(period=[lx] if ly is None else [lx, ly])


    @classmethod
    def oblique(cls, lx, ly, angle):
        """A cell given by two edge lengths and the **angle between them**, in radians.

        ``a1`` lies along x with length ``lx``; ``a2`` has length ``ly`` and makes ``angle`` with
        it, so::

            A = [[lx, ly*cos(angle)],
                 [ 0, ly*sin(angle)]]

        Radians, because every other angle in this API is (``theta``, ``phi``, a shape's ``angle``).
        ``np.radians(60)`` for the usual ones, or :meth:`hexagonal` for the usual-est.

        This is a *spelling*, not a new kind of cell -- it returns the same :class:`Lattice` that
        ``lattice=`` does, and there is no ``angle`` stored anywhere afterwards. Which matters for
        differentiation: build it inside the function you differentiate and the angle is a design
        variable, because the basis it produces is one. Build it outside and you have handed the
        solver a constant, exactly as with any other expression.

        ``angle`` near 0 or pi is refused rather than returning a basis with no cell area -- the
        same refusal :func:`as_lattice` makes for parallel columns, reached earlier and by name.

        A right angle comes back *as* a right angle -- ``axis_aligned`` is read off the basis, not
        assumed -- but to 1e-16 and not to the bit, since ``cos(pi/2)`` is 6.1e-17. Use
        :meth:`rectangular` where an exact diagonal is wanted, which is every case that does not
        have an angle to vary.

        Any of the three can be the design variable, so the namespace is read off whichever of
        the three carries a derivative (:func:`carries_autodiff`); a concrete array of another
        framework is a number here, as it was. The namespace was the angle's alone, and a
        traced length beside a plain angle took the NumPy branch:
        ``np.stack`` refused a tensor under autograd and a JAX tracer, and converted a PyTorch
        forward-mode dual to its primal, so ``hexagonal(a)`` of a dual ``a`` came back a constant
        basis and its derivative zero, with no error.
        """
        traced = [v for v in (lx, ly, angle) if carries_autodiff(v)]
        if traced:
            xp = array_namespace(*traced)
        else:
            xp = array_namespace(angle) if not isinstance(angle, (int, float)) else np
        if xp is np:
            a = float(angle)
        elif traced and not any(v is angle for v in traced):
            # A plain angle beside a traced length, in the length's precision: PyTorch would
            # otherwise make it its default float32, and the cell single precision.
            a = as_array(xp, angle, dtype=traced[0].dtype)
        else:
            a = as_array(xp, angle)
        if isinstance(angle, (int, float)) and not (1e-9 < abs(float(np.sin(float(angle))))):
            raise ValueError(
                f'Lattice.oblique: angle={angle!r} rad puts a1 and a2 on the same line, so the cell has no '
                f'area. Use a value strictly between 0 and pi (np.radians(60) for a hex rhombus).')
        cos_a, sin_a = xp.cos(a), xp.sin(a)
        zero = 0.0 * cos_a
        cols = xp.stack([xp.stack([lx + zero, ly * cos_a]),
                         xp.stack([zero, ly * sin_a])])
        if xp is np:
            M = np.asarray(cols, dtype=float)
            return cls(M[:, 0], M[:, 1], is_axis_aligned(M))    # a right angle says so
        return cls(cols[:, 0], cols[:, 1], False)   # traced: the general path, always correct


    @classmethod
    def hexagonal(cls, a, angle=2.0 * np.pi / 3.0):
        """A primitive cell of a hexagonal array: the rhombus of side ``a``, **120 degrees**.

        120 is the crystallographic standard -- the hexagonal system is defined with
        ``a = b, alpha = beta = 90, gamma = 120`` -- and the reason is not taste. 60 spans the same
        lattice, but *the three-fold rotation axis is not visible in the primitive vectors there*:
        the 120 pair maps into itself under a 120 rotation, and the 60 pair does not. That is why
        the standard settings pick it, and it is why this is the default.

        60 is a legitimate alternative basis, the one that makes the primitive cell two equilateral
        triangles, and it is what ``docs/theory/rcwa/oblique-lattice.md`` and the S4 cross-check in it are
        written on. ``Lattice.hexagonal(a, np.radians(60))``.

        They generate the **same lattice** -- ``a2' = a2 + a1`` carries one to the other -- but they
        are not the same calculation:

        * the *fundamental cell* differs, so an omitted shape centre lands elsewhere: ``(100, 173)``
          against ``(300, 173)`` at ``a = 400``. See :func:`~wavebit.rcwa._engine.parametric.shape_geometry`.
        * ``truncation_order`` cuts a parallelogram in ``(m, n)``, and a different basis maps that
          onto a different set of ``G``. Measured at ``truncation_order=4``: 81 orders either way,
          **61 of them shared**. The two agree in the converged limit and not before it.

        So this is a choice with consequences at finite truncation, which is why it is an argument
        rather than a constant folded into the name.
        """
        return cls.oblique(a, a, angle)

    # pytree protocol -- registered lazily; see _register_pytree_if_jax_loaded
    def tree_flatten(self):
        return (self.u, self.v), (self.axis_aligned,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], children[1], aux[0])


def _register_pytree_if_jax_loaded():
    """Register :class:`Lattice` as a JAX pytree, if JAX is in the process already.

    Lazy for the reason ``src/wavebit/rcwa/_backends/jax_backend.py`` is imported only on use: ``import wavebit``
    must not import JAX
    for a caller who will never use it, and importing it here flips ``jax_enable_x64``
    process-wide. Reads ``sys.modules`` and never imports -- the same pattern as
    :func:`wavebit._common.periodic._register_pytree_if_jax_loaded`, which the docstring there
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


def _stack_geometry_entries(values):
    """Stack entries without detaching derivatives or rounding literals before promotion."""
    # NumPy constants may sit beside the native arrays carrying the derivative.
    native = [value for value in values
              if hasattr(value, 'dtype') and not isinstance(value, (np.ndarray, np.generic))]
    xp = array_namespace(*(native or values))
    device = getattr(native[0], 'device', None) if native else None
    arrays = [as_array(xp, value, device=device)
              for value in values if hasattr(value, 'dtype')]
    literals = [value for value in values if not hasattr(value, 'dtype')]
    dtype = xp.result_type(*arrays, *literals)
    return xp.stack([as_array(xp, value, dtype=dtype, device=device) for value in values])


def as_lattice(period=None, lattice=None):
    """The ``(2, 2)`` basis for a ``period`` / ``lattice`` pair, as a :class:`Lattice`.

    Exactly one of the two is expected; ``lattice`` wins if both are given, and ``period`` is
    then only a derived quantity (see :func:`periods_of`). Concrete inputs are validated
    eagerly. Traced arrays and sequences retain their derivatives, and their shapes are
    checked without reading their values.
    """
    if lattice is not None:
        if isinstance(lattice, Lattice):
            return lattice
        try:
            A = constant_array(lattice)
        except Exception:                       # noqa: BLE001 - a tracer, not a mistake
            # A basis handed in already traced. Its numbers are the point -- that is the whole
            # reason this is a pytree -- but `axis_aligned` cannot be read off a tracer, so it
            # takes the general path. Always correct, merely not the rectangular shortcut.
            cols = lattice
            if isinstance(cols, (tuple, list)):
                if len(cols) != 2:
                    raise ValueError('lattice: expected a 2x2 matrix of basis columns')
                for row in cols:
                    shape = getattr(row, 'shape', None)
                    if shape is None and isinstance(row, (tuple, list)):
                        shape = (len(row),)
                    if shape != (2,):
                        raise ValueError('lattice: expected a 2x2 matrix of basis columns')
                values = [value for row in cols for value in row]
                if any(isinstance(value, (tuple, list)) or getattr(value, 'ndim', 0) != 0
                       for value in values):
                    raise ValueError('lattice: expected a 2x2 matrix of basis columns')
                entries = _stack_geometry_entries(values)
                cols = array_namespace(entries).reshape(entries, (2, 2))
            if getattr(cols, 'shape', None) != (2, 2):
                raise ValueError('lattice: expected a 2x2 matrix of basis columns')
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
        p = np.atleast_1d(constant_array(period))
    except Exception:                       # noqa: BLE001 - a traced period, not a mistake
        # `period` is the variable under differentiation. Its *shape* is still a diagonal, so
        # `axis_aligned` is known without reading a number -- which is the whole reason that
        # flag is aux_data and not a child.
        # A tuple/list of tracers and a scalar still carry known shapes. Normalize these
        # spellings before stacking, rather than duplicating the first entry of a pair.
        if isinstance(period, (tuple, list)):
            values = period
        else:
            shape = period.shape
            if len(shape) > 1:
                raise ValueError('period: expected one length or a one-dimensional pair of lengths')
            values = [period] if not shape else [period[i] for i in range(shape[0])]
        if len(values) not in (1, 2):
            raise ValueError(f'period: expected 1 or 2 lengths, got {len(values)}')
        if any(isinstance(value, (tuple, list)) or getattr(value, 'ndim', 0) != 0
               for value in values):
            raise ValueError('period: expected one length or a one-dimensional pair of lengths')
        if len(values) == 1:
            values = [values[0], values[0]]
        q = _stack_geometry_entries(values)
        xp = array_namespace(q)
        zero = 0.0 * q[0]
        return Lattice(xp.stack([q[0], zero]), xp.stack([zero, q[1]]), True)
    if p.ndim != 1:
        raise ValueError('period: expected one length or a one-dimensional pair of lengths')
    if p.size == 1:
        p = np.array([p[0], p[0]])
    if p.size != 2:
        raise ValueError(f'period: expected 1 or 2 lengths, got {p.size}')
    D = np.diag(p)
    return Lattice(D[:, 0], D[:, 1], True)   # a period spells a diagonal; nothing to test


#: How far past the Lagrange condition a basis may sit and still be its own reduction. The
#: condition is ``|a1 . a2| <= min(|a1|^2, |a2|^2) / 2``, and a hexagonal cell sits *on* it:
#: ``a2 = (cos 60, sin 60) |a1|`` gives exactly one half, and the 120-degree spelling minus one
#: half, but ``Lattice.hexagonal`` reaches them through ``cos(2 pi / 3)``, which is one half
#: to 1e-16 and not to the bit. A basis that is reduced up to rounding must come back as
#: itself -- the same object, so the field built on it is bit-identical to the one built
#: before the reduction existed -- and this band, 1e-9 relative, is what "up to rounding"
#: means here. A basis past it is reduced; the 3x3 sweep the fold makes after it needs only
#: the condition to hold within a quantity of this order, which it does with room to spare.
REDUCTION_TOL = 1e-9

_IDENTITY = ((1, 0), (0, 1))


def _nearest_multiple(mu):
    """The integer nearest ``mu``, a tie resolved toward zero, within :data:`REDUCTION_TOL`.

    ``round`` alone rounds half to even, which sends ``1.5`` to ``2`` and ``0.5`` to ``0`` --
    the second is what a reduced basis on the boundary needs, the first is merely a
    different reduced basis, and depending on which half a value falls on is not a rule.
    Toward zero, always: a projection of exactly one half (the hexagonal cell) is already
    reduced, and the band keeps it so through the rounding of its own construction.
    """
    k = int(mu)                                  # toward zero
    frac = mu - k
    if frac > 0.5 + REDUCTION_TOL:
        k += 1
    elif frac < -0.5 - REDUCTION_TOL:
        k -= 1
    return k


def reduction_transform(A):
    """The integer unimodular ``T`` with ``A @ T`` Lagrange-reduced: ``((t00, t01), (t10, t11))``.

    Gauss / Lagrange reduction of the two column vectors: while the longer's projection on
    the shorter exceeds half of it, subtract the nearest multiple of the shorter from the
    longer. It ends with ``|a1' . a2'| <= min(|a1'|^2, |a2'|^2) / 2`` (within
    :data:`REDUCTION_TOL`), which is the condition under which the Wigner-Seitz cell of the
    origin lies inside one cell a side of the fundamental parallelogram -- the fact the
    normal field's 3x3 fold relies on (:func:`~wavebit.rcwa._engine.parametric._fold_to_nearest_image`)
    and had assumed of every accepted basis. It was not true of ``a2 = 3 a1 + (50, 700)``,
    which :func:`as_lattice` accepts as it is: a lone 693 circle on that cell read its level
    set 8.89 deg off in the mean on its own boundary (69.9 max) and 43.8 / 90.0 at ``k =
    10``, because the nearest centre was two cells over from the rounded fractional
    coordinates. Reduced, both read 2.5e-7, the same as ``k = 0``.

    ``T`` is the identity on a basis that is already reduced -- the square, the rectangle
    either way round, both hexagonal spellings, the ``(200, 700)`` shear, Li's checkerboard
    -- and an integer matrix of determinant ``+-1`` otherwise, so ``A @ T`` spans the same
    lattice and enumerates the same images under different names. It never swaps the two
    vectors: the reduced basis is asked to be the *input* wherever the input qualifies, and
    the Fourier side, the sample grid and the shape origin keep the basis as given in every
    case. Read through :func:`~wavebit._common.backend.checked_float`, so a ``jax.grad`` tracer
    reduces from its primal and a ``jax.jit`` tracer raises ``TypeError`` -- the caller that
    cannot read the basis cannot plan on it either.
    """
    a1x, a1y = checked_float(A[0][0]), checked_float(A[1][0])
    a2x, a2y = checked_float(A[0][1]), checked_float(A[1][1])
    t00, t01, t10, t11 = 1, 0, 0, 1            # a1' = t00 a1 + t10 a2, a2' = t01 a1 + t11 a2
    while True:
        n1 = a1x * a1x + a1y * a1y
        n2 = a2x * a2x + a2y * a2y
        dot = a1x * a2x + a1y * a2y
        if n1 == 0.0 or n2 == 0.0:
            break                                # degenerate: `as_lattice` refuses it anyway
        if n1 <= n2:
            k = _nearest_multiple(dot / n1)
            if k == 0:
                break
            a2x, a2y = a2x - k * a1x, a2y - k * a1y
            t01, t11 = t01 - k * t00, t11 - k * t10
        else:
            k = _nearest_multiple(dot / n2)
            if k == 0:
                break
            a1x, a1y = a1x - k * a2x, a1y - k * a2y
            t00, t10 = t00 - k * t01, t10 - k * t11
    return ((t00, t01), (t10, t11))


def reduced_basis(A, transform=None):
    """``A`` on its Lagrange-reduced basis -- the same lattice, spelled so that its images are near.

    A :class:`Lattice`; the very object handed in when the reduction is the identity, so the
    field built on a basis that was reduced to begin with is the field built before this
    function existed, to the bit. Otherwise the two new vectors are ``A @ T`` computed on
    the basis's own entries -- a traced basis stays traced, and ``dR / dA = T`` exactly, since
    ``T`` is piecewise constant in ``A`` -- and ``axis_aligned`` is read off the numbers where
    they can be read (a shear of ``10 a1`` on a square reduces *to* the square) and ``False``
    where they cannot.

    ``transform`` is the integer ``T`` when the caller already holds it -- the frozen
    :class:`~wavebit.rcwa._engine.parametric.GammaPlan` records it, so a jitted replay on a traced
    basis reduces with the recorded integers rather than asking the tracer for a number.

    What reads this and what does not is the whole design (2026-09-06): every enumeration of
    periodic *images* -- the fold, the instance lists, the pair sweep, the overlap check, the
    oracle -- goes through it, and nothing on the Fourier side does. ``order_kxy``, the
    truncation window, ``periods_of``, the sample grid, the origin a shape's ``cx`` is measured
    from: all of those are the basis as the caller spelled it, because they *are* a spelling
    -- which ``G`` is order ``(m, n)`` is the user's naming of the reciprocal set, and a window
    on a reduced spelling is a different set of retained orders.
    """
    A = as_lattice(lattice=A)
    T = reduction_transform(A) if transform is None else tuple(map(tuple, transform))
    if T == _IDENTITY:
        return A
    (t00, t01), (t10, t11) = T
    u = t00 * A.u + t10 * A.v
    v = t01 * A.u + t11 * A.v
    try:
        cols = np.array([[checked_float(u[0]), checked_float(v[0])],
                         [checked_float(u[1]), checked_float(v[1])]])
        aligned = is_axis_aligned(cols)
    except Exception:                       # noqa: BLE001 - a traced basis: the general path
        aligned = False
    return Lattice(u, v, aligned)


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
        M = constant_array(cols)
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
    :func:`wavebit.rcwa._engine.factorization.normal_vector_field`.
    """
    cols = A.columns if isinstance(A, Lattice) else A
    try:
        M = constant_array(cols)
    except Exception:                       # noqa: BLE001 - a tracer, not a mistake
        xp = array_namespace(cols)
        return xp.stack([cols[:, 0] / n_u, cols[:, 1] / n_v], axis=-1)
    return np.column_stack((M[:, 0] / n_u, M[:, 1] / n_v))


def sample_grid(xp, grid_shape, lattice, dtype):
    """Pixel-centre coordinates in the **internal** frame: two ``(Ny, Nx)`` arrays ``(X, Y)``.

    ``r = (i + 0.5)/Nx * a1 + (j + 0.5)/Ny * a2`` -- the cell's *corner* at the origin, which
    is the frame the Fourier machinery works in and the one
    :func:`~wavebit.rcwa._engine.parametric.shape_geometry` resolves a shape's centre into. **The public
    door is :func:`meshgrid_xy`**, which re-bases to the cell centre; the two differ by
    ``0.5*(a1 + a2)``, and reaching for this one from user-facing code misplaces a structure by
    half a period with nothing to say so.

    Evaluated as *positions* rather than read as indices. That is what the two
    non-raster normal-field builders need
    (:func:`wavebit.rcwa._engine.parametric.normal_vector_field_analytical`,
    :func:`wavebit.rcwa._engine.boundary.distance_gradient_normal_field`): both measure something
    Cartesian at each sample — parametric shape gradients on one side, distances to polygon edges
    on the other — so a skewed basis moves the sample points and nothing else. No metric follows.
    :func:`jacobian`, which converts index steps, is the raster path's correction and not theirs.

    Here rather than once per builder because ``period`` is a positional pair of two same-typed
    floats: unpacking it as ``(Py, Px)`` is a mistake nothing reacts to — not the types, not the
    runtime, and not a test on a square cell, where both names bind to the same number. It
    happened. The reversed unpack travelled from the parametric builder to the boundary one with a
    copy of this block, while the caller that reversed it back did not, so the boundary route
    sampled its field over a transposed cell on any non-square period (fixed 2026-08-11). One
    reader is one opportunity to get it wrong.

    ``period`` is read only when ``lattice`` is None, which is the rarer branch: ``engine``
    hands the shape paths ``as_lattice(period=...)`` whenever the period is *readable*
    (:meth:`~wavebit.rcwa._engine.engine.CoreFactorizationMixin._geometry_lattice_for`). Counting the
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
    flat order list the convolution matrices are indexed by (``m`` fastest, matching the
    ``i = n*n_orders_u + m`` layout every builder in ``wavebit.rcwa._engine.factorization``
    returns).

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


@eager_under_torch_compile
def meshgrid_xy(grid_shape, period=None, lattice=None, backend='numpy', dtype=None):
    """Cartesian coordinates of a raster's pixel centres: ``(X, Y)``, both that shape.

    ``numpy.meshgrid`` for a unit cell. On a skewed cell the two cannot be separated into 1-D
    axes at all -- ``X`` depends on the row as well as the column, because a pixel's own edges
    are ``a1/Nx`` and ``a2/Ny`` and the pixel is therefore a parallelogram rather than a square.
    The values are the **centres** of those parallelograms; ``(i + 0.5)`` is what puts them
    there, and there are ``Nx`` of them per row rather than ``Nx + 1``.

    The public way to build a structure without deciding an axis order. Write the layer in terms
    of the coordinates and the layout follows on its own::

        X, Y = wavebit.meshgrid_xy((64, 96), period=[700., 1100.])
        layer = np.where(X ** 2 + Y ** 2 < 200. ** 2, 12.25, 1.0)

    ``np.where`` is elementwise, so ``layer`` simply inherits whatever layout ``X`` and ``Y``
    already had. **That is the whole point**: an expression written in *coordinates* cannot get
    the axis order wrong, while one written in *indices* (``layer[:2] = ...``) can, and does so
    silently on a non-square raster. Hand-rolling the grid is what pushes callers towards
    indices; this exists so they need not.

    **The cell is centred at the origin**, the same origin a parametric shape's ``cx`` / ``cy``
    and ``reconstruct_field``'s ``sample_x`` / ``sample_y`` use, so the rod above is centred by
    writing nothing. It read ``(X - 350.)**2 + (Y - 550.)**2`` until 2026-08-31 -- the caller
    computed the centre of the cell, which on a skewed cell is ``0.5*(a1 + a2)``, two
    components neither of which is a period. That line appeared in the README, in
    ``docs/theory/rcwa/oblique-lattice.md``, in a tutorial and in eight helpers; one origin deletes all of
    them. :func:`sample_grid` is the internal corner-based twin and is not this function.

    A skewed cell comes free. ``lattice=`` places the samples on the parallelogram, so the same
    coordinate expression describes the same physical shape whatever the cell -- where writing
    ``meshgrid`` by hand means writing ``u*a1 + v*a2`` by hand too, once per caller. The
    equivalent elsewhere in this field is ``numpy.meshgrid(..., indexing='ij')``: fmmax's
    examples spell that out eleven times because nothing hands it to them, and forgetting it
    transposes the structure with nothing to say so.

    ``grid_shape`` is ``(rows, columns)``, read as numpy reads an array's shape. Rows run along
    ``a2`` and columns along ``a1`` -- y and x on a rectangular cell, and neither of those on a
    skewed one, which is why the return is coordinates rather than axis names.

    **Neither half of the name is spare.** Plain ``shape`` collides with geometry: this package
    spells a piece of geometry a "shape" (``structure=[{'shapes': [...]}]``,
    :func:`~wavebit.rcwa._engine.parametric.shape_ft`, ``shape_geometry``), so ``A.meshgrid_xy(shape)`` read as
    *"put a shape on this lattice"*, which is the one thing it does not do. But ``shape`` is
    also what carries the **axis order**: an array's shape is rows-first, and that is what this
    pair is. A name that drops the word loses that -- ``samples`` would collide with
    :meth:`~wavebit.rcwa._engine.engine.CoreSolveMixin.reconstruct_field`'s ``sample_x`` / ``sample_y``,
    which are counts per *named* axis and therefore x-first, and would invite ``(nx, ny)``. So
    the word stays and the prefix disambiguates it, which is also what
    :func:`sample_grid` and the three builders in :mod:`wavebit.rcwa._engine.boundary` already call it.

    **Prefer :meth:`Lattice.meshgrid_xy` where the cell is already named.** That method is this
    function with ``lattice=A``, and it spells the basis once instead of twice -- which matters
    because the *other* use of it is ``lattice=A`` on the solver, and a raster drawn on one cell
    and solved on another is a different structure, silently. That method's docstring has the
    measurement and the limits of the nudge.

    Exactly one of ``period`` / ``lattice``; ``backend`` matches :func:`wavebit.rcwa`
    (``'numpy'``, ``'jax'``, ``'torch'``). ``dtype`` defaults to the backend's float64.
    """
    A = as_lattice(period=period, lattice=lattice)
    shape = tuple(int(v) for v in grid_shape)
    if len(shape) != 2:
        raise ValueError(f'meshgrid_xy: grid_shape must be (rows, columns); got {grid_shape!r}')
    if min(shape) < 1:
        raise ValueError(f'meshgrid_xy: grid_shape must be positive; got {grid_shape!r}')

    backend = backend_of(backend)
    if backend is Backend.NUMPY:
        import array_api_compat.numpy as xp
    elif backend is Backend.JAX:
        import jax.numpy as xp
        lift_jax_x64(required=True)
    else:
        import array_api_compat.torch as xp

    X, Y = sample_grid(xp, shape, A, dtype if dtype is not None else xp.float64)
    # The one place the public frame is put on. `sample_grid` builds the cell from its corner
    # because that is what the transform integrates over; a caller sees it from the centre.
    return X - 0.5 * (A[0][0] + A[0][1]), Y - 0.5 * (A[1][0] + A[1][1])
