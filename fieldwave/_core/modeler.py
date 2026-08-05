"""Single-source vector (shape-based) geometry modeler.

Replaces the three near-identical copies in ``fieldwave/on_{numpy,jax,torch}/modeler/modeling.py``
with one Array-API implementation, in the same style as :mod:`fieldwave._core.factorization` and
:mod:`fieldwave._core.fields`.

What this module produces is the *vector* (staircase-free) description of a layer: a list of
``(cell, x_list, y_list)`` triples, where ``cell`` holds the refractive index of each
piecewise-constant block and ``x_list`` / ``y_list`` the block edges. That is exactly the input
:func:`fieldwave._core.factorization.to_conv_mat_vector` consumes, so a shape drawn here is
Fourier-transformed analytically rather than off a raster.

Two conventions carry over from the already-ported modules:

* **No in-place assignment.** The legacy builders wrote into preallocated arrays
  (``rotate[0, 0] = ...``, ``ucell_layer[r0:r1, c0:c1] = pmty``, ``arr[:, ::2] = ...``). Each is
  rewritten as a build-then-``stack``/``concat``, or as a ``where`` over a statically computed
  index mask, so the values stay in the differentiable chain on every backend.
* **Geometry decisions are eager, material values flow through.** Sorting the block edges and
  picking the rectangle's rotation octant branch on values and are narrowed to Python with
  ``bool()`` / ``int()``. The coordinates and refractive indices themselves are never narrowed,
  so gradients still reach ``cx``, ``cy``, ``lx``, ``ly``, ``angle`` and ``n_index``. The
  ellipse needs no such decision at all: its blocks are one closed-form expression of the
  shape parameters, valid at every angle.

The one genuinely backend-specific corner is the material table: it is read off disk
(``np.loadtxt`` / ``scipy.io.loadmat``) and interpolated with NumPy, because there is no
Array-API ``interp`` and the table is static data rather than a design variable.
"""
from os import walk
from typing import NamedTuple
from pathlib import Path

import numpy as np

from .backend import array_namespace
from .geometry import StaleGeometryError      # raised here too; one home for the contract

_PI = 3.141592653589793


# ------------------------------------------------------------------ namespace / dtype helpers

def _is_array(v):
    """True if ``v`` belongs to a supported array namespace (NumPy / JAX / PyTorch)."""
    try:
        array_namespace(v)
    except Exception:
        return False
    return True


def _namespace_of(*vals):
    """Namespace of the first array among ``vals``, or ``None`` if there is none."""
    for v in vals:
        if _is_array(v):
            return array_namespace(v)
    return None


def _dev(x):
    """Device of ``x``, so arrays the modeler creates land beside the ones it was given."""
    return x.device if hasattr(x, 'device') else None


def _resolve_device(xp, device):
    """Map fieldwave's device code onto a real device object for ``xp``.

    ``0``/``'cpu'`` and ``1``/``'gpu'``/``'cuda'`` are fieldwave's own codes, not backend device
    indices — handing ``0`` straight to torch selects device *index* 0, which on a Mac is MPS
    and rejects float64. NumPy and JAX place arrays themselves, so they get ``None``.
    """
    if device is None or 'torch' not in xp.__name__:
        return None
    import torch
    if device in (0, 'cpu'):
        return torch.device('cpu')
    if device in (1, 'gpu', 'cuda'):
        return torch.device('cuda')
    if isinstance(device, torch.device):
        return device
    raise ValueError(f'unsupported device: {device!r}')


def _device_of(*vals):
    for v in vals:
        if _is_array(v):
            return _dev(v)
    return None


def _to_ns(xp, v, device=None):
    """``v`` as an array of namespace ``xp`` (no-op when it already is one)."""
    if _is_array(v) and array_namespace(v) is xp:
        return v
    return xp.asarray(v, device=device)


def _cast(xp, v, dtype, device=None):
    """``v`` as an ``xp`` array of ``dtype``.

    Scalars are *created* at ``dtype`` rather than created-then-cast: torch's default is
    float32, so casting afterwards would already have thrown the precision away.
    """
    if _is_array(v):
        v = _to_ns(xp, v, device)
        return v if v.dtype == dtype else xp.astype(v, dtype)
    return xp.asarray(v, dtype=dtype, device=device)


def _as1(xp, v, dtype=None, device=None):
    """``v`` reshaped to a length-1 array of ``xp`` (the legacy ``np.array(v).reshape(1)``)."""
    if _is_array(v):
        a = xp.reshape(_to_ns(xp, v, device), (-1,))
        if dtype is not None and a.dtype != dtype:
            a = xp.astype(a, dtype)
        return a
    return xp.asarray([v], dtype=dtype, device=device)


def _shape_param(xp, v, dtype, device=None):
    """A shape parameter (``cx`` / ``lx`` / ``angle`` ...) as a length-1 array at ``dtype``.

    Python scalars are created *at* ``dtype`` rather than created-then-cast: torch's default is
    float32, so a plain ``asarray(0.4)`` would already have rounded the angle away (this cost
    the rotated cases ~1e-6 in the golden comparison before it was fixed). Arrays that are
    already complex are left alone, as the legacy did.
    """
    if not _is_array(v):
        return xp.asarray([v], dtype=dtype, device=device)
    a = xp.reshape(_to_ns(xp, v, device), (-1,))
    if xp.isdtype(a.dtype, 'complex floating') and not xp.isdtype(dtype, 'complex floating'):
        return a
    return a if a.dtype == dtype else xp.astype(a, dtype)


def _flat(xp, v, type_float, type_complex, device=None):
    """``v`` flattened to 1-D, creating scalars at the modeler's working precision."""
    if _is_array(v):
        return xp.reshape(_to_ns(xp, v, device), (-1,))
    dtype = type_complex if isinstance(v, complex) else type_float
    return xp.asarray([v], dtype=dtype, device=device)


def _result_dtype(xp, dtypes):
    """Promoted dtype of ``dtypes`` (NumPy's ``concatenate`` promotion, done explicitly).

    ``concat`` promotes on all three backends, but doing it here keeps the edge lists in one
    dtype before they meet, which is what the legacy NumPy path produced implicitly.
    """
    out = dtypes[0]
    for d in dtypes[1:]:
        out = xp.result_type(out, d)
    return out


def _precision_is_double(type_complex):
    """Map any spelling of the complex dtype onto the double / single precision flag.

    Accepts the legacy shorthands (``0`` = complex128, ``1`` = complex64) as well as any
    backend's dtype object — matching on the bit width keeps this free of ``if torch:``.
    """
    if type_complex is None:
        return True
    if isinstance(type_complex, int) and not isinstance(type_complex, bool):
        if type_complex == 0:
            return True
        if type_complex == 1:
            return False
        raise ValueError(f'unsupported type_complex: {type_complex}')
    s = str(type_complex).lower()
    if '128' in s:
        return True
    if '64' in s or '32' in s:
        return False
    raise ValueError(f'unsupported type_complex: {type_complex}')


def _edge_list(xp, parts, type_float, type_complex, device):
    """The ``(n, 1)`` grid line column both the eager build and the replay hand back."""
    flat = [_flat(xp, v, type_float, type_complex, device) for v in parts]
    dtype = _result_dtype(xp, [p.dtype for p in flat])
    return xp.reshape(xp.concat([xp.astype(p, dtype) for p in flat]), (-1, 1))


class VectorGridPlan(NamedTuple):
    """Where each grid line came from, and which rows and columns each block owns.

    Everything here is a Python int or ``None`` — no arrays — so a plan is hashable and can be
    passed as a static argument to a jitted call, and so replaying it involves no comparison of
    any traced value.

    ``row_sources`` / ``col_sources`` name, per grid line and in order, the block edge that
    became it: ``(obj_index, 0 or 1, 0 or 1)`` for the near or far edge of that object's row or
    column, or ``None`` for the closing period line. Two blocks that met at a coordinate appear
    once, under whichever of them got there first — which is the sharing, recorded.
    """
    row_sources: tuple
    col_sources: tuple
    ranges: tuple


def _check_plan_still_fits(xp, row_list, col_list):
    """Reject a replayed grid whose lines have fallen out of order.

    A recorded plan says which block edge becomes which grid line, in what order. Move the
    shape parameters far enough and the edges themselves reorder, at which point the plan is
    describing a layer that is no longer there — and the result is wrong without being
    obviously wrong, which is the failure this exists to stop. Measured: an ellipse frozen at
    ``lx`` 85 and replayed at 120 lands 1.5e-4 off; two rectangles frozen apart and replayed
    overlapping land 4% off.

    The tell is free. Grid lines are sorted by construction, so a replayed list that is not
    sorted is a plan that no longer fits — every stale case measured shows it, and every valid
    one is strictly increasing with room to spare (``min diff`` +1.13 at ``lx`` 100 against
    -7.80 at 120).

    Only reachable eagerly. Under a trace the comparison below is itself the thing that cannot
    be done — which is what the plan exists to avoid — so a jitted call carries no check, and
    a long fit should re-freeze on a schedule rather than rely on being told.
    """
    for name, values in (('y', row_list), ('x', col_list)):
        try:
            edges = [float(xp.real(v)[0]) for v in values]
        except Exception:                       # a tracer: nothing to compare, nothing to do
            return
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            if hi <= lo:
                raise StaleGeometryError(
                    f'the frozen geometry no longer fits this structure: replaying it puts '
                    f'{name} grid line {i + 1} at {hi:.6g}, at or below its predecessor '
                    f'{lo:.6g}. The shape parameters have moved far enough to reorder the '
                    f'block edges, so the recorded plan describes a different layer. Call '
                    f'freeze_geometry() again, or drop geometry= to rebuild by comparison.')


def _bisect_left(seq, value):
    """``bisect.bisect_left`` comparing real parts only.

    The legacy code fed complex edges straight to ``bisect_left``, which works on NumPy (it
    orders complex lexicographically) but raises on PyTorch. Every edge here has zero imaginary
    part, so comparing real parts is the same ordering — and works everywhere.
    """
    lo, hi = 0, len(seq)
    key = value.real
    while lo < hi:
        mid = (lo + hi) // 2
        if bool(seq[mid].real < key):
            lo = mid + 1
        else:
            hi = mid
    return lo


class Modeling:
    """Vector modeler: draws rectangles / ellipses and reduces them to block-edge lists.

    The namespace is inferred from whatever arrays it is handed — the shape parameters of
    :meth:`rectangle` / :meth:`ellipse`, or ``period`` — and falls back to NumPy, so a modeler
    built from Python scalars still works and one fed ``torch`` tensors stays on torch.
    """

    def __init__(self, period=(1., 1.), type_complex=None, xp=None, device=None,
                 *args, **kwargs):
        self._ucell = None
        self.ucell_vector = None
        self.grid_plans = None
        self._last_plan = None
        self.shape_choices = None
        self._shape_replay = self._shape_record = None
        self.x_list = None
        self.y_list = None
        self.mat_table = None
        self.ucell_info_list = None
        self.shapes_list = []
        self.film_layer = None

        self._xp = xp if xp is not None else array_namespace(np.zeros(1))
        self.device = _resolve_device(self._xp, device)

        self._double = True
        self._period = None
        self.type_complex = type_complex
        self.period = period

    # ---------------------------------------------------------------- namespace & precision

    @property
    def xp(self):
        """Default array namespace (arrays passed in may override it per call)."""
        return self._xp

    def _ns(self, *vals):
        return _namespace_of(*vals) or self._xp

    def _dev(self, *vals):
        dev = _device_of(*vals)
        return self.device if dev is None else dev

    def _complex(self, xp):
        return xp.complex128 if self._double else xp.complex64

    def _float(self, xp):
        return xp.float64 if self._double else xp.float32

    def _int(self, xp):
        return xp.int64 if self._double else xp.int32

    @property
    def type_complex(self):
        return self._complex(self._xp)

    @type_complex.setter
    def type_complex(self, type_complex):
        self._double = _precision_is_double(type_complex)
        if getattr(self, '_period', None) is not None:
            self._period = _cast(self._xp, self._period, self._float(self._xp), self.device)

    @property
    def type_float(self):
        return self._float(self._xp)

    @property
    def type_int(self):
        return self._int(self._xp)

    # ---------------------------------------------------------------------------- geometry

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, period):
        if period is None:
            self._period = None
            return

        if isinstance(period, (int, float)):
            vals = [period, period]
        elif _is_array(period) and getattr(period, 'ndim', 1) > 0:
            self._xp = array_namespace(period)
            flat = self._xp.reshape(period, (-1,))
            if flat.shape[0] == 1:
                flat = self._xp.concat([flat, flat])
            self._period = _cast(self._xp, flat, self._float(self._xp), _dev(period))
            return
        elif isinstance(period, (list, tuple)) or _is_array(period):
            vals = list(period)
            if len(vals) == 1:
                vals = [vals[0], vals[0]]
        else:
            raise ValueError('period')

        xp = self._ns(*vals)
        self._xp = xp
        dev = self._dev(*vals)
        F = self._float(xp)
        if any(_is_array(v) for v in vals):
            self._period = xp.concat([_as1(xp, v, F, dev) for v in vals])
        else:
            self._period = xp.asarray(vals, dtype=F, device=dev)

    @property
    def ucell(self):
        return self._ucell

    @ucell.setter
    def ucell(self, ucell):
        self._ucell = ucell

    # ------------------------------------------------------------------------------ shapes

    @staticmethod
    def rectangle_no_approximation(cx, cy, lx, ly, base):
        """Axis-aligned rectangle: one block, given as ``[top_left, bottom_right, index]``."""
        a = [cy - ly / 2, cx - lx / 2]  # row, col
        b = [cy + ly / 2, cx + lx / 2]  # row, col

        return [[a, b, base]]  # top_left, bottom_right

    def rectangle(self, cx, cy, lx, ly, n_index, angle=0, n_split_triangle=2,
                  n_split_parallelogram=2, angle_margin=1E-5):
        """Rectangle, rotated by ``angle`` (radians), decomposed into axis-aligned blocks.

        The shape is cut into horizontal slabs and each slab becomes one block whose left and
        right edges are the rectangle's *exact* horizontal chord at the slab's mid-height.

        The two inner corners are the only heights where that chord changes slope, so they are
        made slab boundaries: an upper triangle, a parallelogram band and a lower triangle, held
        by ``n_split_triangle`` and ``n_split_parallelogram`` — which is what those names have
        always said. Inside a region the chord is linear, and a linear chord read at the middle
        of a slab gives the slab its exact area, so the blocks sum to ``lx*ly`` at every angle.

        This draws the *same region* as the ramp-and-mean construction it replaces — that one
        was already the mid-height chord, written differently — in two fewer blocks: measured on
        an 80x60 rectangle at 0.05, 0.4 and 1.0 rad, the union area and the symmetric difference
        against the true rotated rectangle agree to every digit, and ``rect_rot``'s efficiencies
        match the frozen golden to 3e-14. What is gained is the margin below, the four-way
        octant branch, and two redundant blocks per call.

        The triangles are thin and still deserve as many slabs as the band. What a region costs
        is not its height but how far the boundary slides sideways across it: on an 80x60
        rectangle at 0.08 rad the upper triangle is 7.2 tall and the band 52.6, yet the left
        edge travels 89.7 across the triangle and only 4.2 across the band. Equal counts are
        close to the split that equalises the two.

        The angle is resolved on ``[0, pi/2)`` — a quarter turn is the same rectangle with its
        sides swapped — which is what keeps ``cos(phi)`` away from zero, and each chord edge is
        the narrower of two strips chosen by a comparison with the divisions cleared, so the
        strip whose coefficient vanishes at an aligned angle is never the one divided by. There
        is no division left to guard, and ``angle_margin`` is accepted for call compatibility
        and ignored.

        **This does not make the path fit a sidewall angle near normal.** A staircase and a
        smoothly tilted boundary do not have the same Fourier coefficients at a finite harmonic
        count, and the difference goes like ``|angle|``: on an 80x60 rectangle in TE at
        ``fto = [3, 3]``, ``sum(de_ti)`` falls off its aligned value at 0.59 per radian on
        *both* sides. That coefficient is flat in ``n_split`` (0.546 at 4, 0.595 at 256) and
        only comes down with harmonics (0.59 at ``fto`` 3, 0.16 at 12), so adding blocks cannot
        remove it — it is the staircase's cost, not its resolution. Only harmonics do, and
        slowly: 0.594 / 0.217 / 0.147 / 0.111 at ``fto`` 3 / 5 / 11 / 17.

        The analytic path builds the rotated shape's coefficients in closed form, and its kink
        is not small but *absent* — measured at 4e-4 or below at every one of those counts. It
        is also where the vector path is heading: at 0.4 rad the analytic answer settles by
        ``fto`` 8 and moves by 4e-4 after, while the vector one is still coming down at 17
        (0.3746 / 0.3606 / 0.3484 against 0.3081 / 0.3363 / 0.3367). Use it for angle fits, as
        :mod:`fieldwave._core.analytic` says.

        That margin used to return the exact single block within ``1e-5`` of a multiple of
        ``pi/2``. Its cost was a plateau and a cliff: the efficiency was pinned to its
        ``angle = 0`` value across the window, then dropped 6.75e-5 in one step at the edge (a
        step that did not shrink from ``n_split`` 4 to 1024), and ``angle`` was absent from the
        graph inside it, so ``backward()`` raised rather than returning zero. All three are
        gone. An aligned rectangle is still exact — with ``n_split`` 0 or 1 it is bit-for-bit
        the single block :meth:`rectangle_no_approximation` returns, sides swapped past a
        quarter turn — it just no longer needs a special case to get there.
        """
        xp = self._ns(cx, cy, lx, ly, angle, self._period)
        dev = self._dev(cx, cy, lx, ly, angle)
        C, F = self._complex(xp), self._float(xp)

        cx = _shape_param(xp, cx, C, dev)
        cy = _shape_param(xp, cy, C, dev)
        lx = _shape_param(xp, lx, C, dev)
        ly = _shape_param(xp, ly, C, dev)
        angle = _shape_param(xp, angle, F, dev)

        # Fold onto [0, pi/2): the shape repeats every pi, and a quarter turn swaps the sides.
        # The choices below are discrete — which quarter turn, which regions exist, which
        # strip bounds each slab — so they are recorded on the way through and replayed from
        # the record when one is supplied, for the same reason the grid build is (see
        # :meth:`vector_per_layer_numeric`). Only the arithmetic is left for a tracer.
        replay = self._take_shape_choice()
        choice = {}

        phi = angle % _PI
        swap = replay['swap'] if replay else bool(phi >= _PI / 2)
        choice['swap'] = swap
        if swap:
            phi = phi - _PI / 2
            A, B = ly / 2, lx / 2
        else:
            A, B = lx / 2, ly / 2

        cos_p = xp.astype(xp.cos(phi), C)
        sin_p = xp.astype(xp.sin(phi), C)

        half_height = A * sin_p + B * cos_p          # top and bottom corners
        inner = A * sin_p - B * cos_p                # the other two, at +-|inner|
        negate = replay['negate'] if replay else bool((xp.real(inner) < 0)[0])
        choice['negate'] = negate
        if negate:
            inner = -inner

        # Slab boundaries, region by region. A region of zero height contributes none: at an
        # aligned angle the triangles vanish and the band is the whole shape.
        n_tri = max(1, int(n_split_triangle))
        n_par = max(1, int(n_split_parallelogram))
        has_triangles = (replay['has_triangles'] if replay
                         else bool((xp.real(half_height - inner) > 0)[0]))
        has_band = replay['has_band'] if replay else bool((xp.real(inner) > 0)[0])
        choice['has_triangles'], choice['has_band'] = has_triangles, has_band

        regions = []
        if has_triangles:
            regions.append((half_height, inner, n_tri))
        if has_band:
            regions.append((inner, -inner, n_par))
        if has_triangles:
            regions.append((-inner, -half_height, n_tri))

        edges = [half_height]
        for top, bottom, count in regions:
            for k in range(1, count + 1):
                edges.append(top + (bottom - top) * (k / count))

        res = []
        strips = []
        for slab, (top, bottom) in enumerate(zip(edges[:-1], edges[1:])):
            dy = (top + bottom) / 2

            # |cos*dx + sin*dy| <= A and |-sin*dx + cos*dy| <= B each bound dx; the chord is the
            # tighter one. Both comparisons are multiplied through by cos*sin >= 0 so that a
            # vanishing sin picks the cos strip instead of dividing by zero.
            right_on_cos = (replay['strips'][slab][0] if replay else bool(
                (xp.real((A - sin_p * dy) * sin_p - (cos_p * dy + B) * cos_p) <= 0)[0]))
            right = ((A - sin_p * dy) / cos_p if right_on_cos
                     else (cos_p * dy + B) / sin_p)

            left_on_cos = (replay['strips'][slab][1] if replay else bool(
                (xp.real((-A - sin_p * dy) * sin_p - (cos_p * dy - B) * cos_p) >= 0)[0]))
            left = ((-A - sin_p * dy) / cos_p if left_on_cos
                    else (cos_p * dy - B) / sin_p)
            strips.append((right_on_cos, left_on_cos))

            res.append([[cy + bottom, cx + left], [cy + top, cx + right], n_index])

        choice['strips'] = tuple(strips)
        self._record_shape_choice(choice)
        return res

    def ellipse(self, cx, cy, lx, ly, n_index, angle=0, n_split_w=2, n_split_h=2,
                angle_margin=1E-5, debug=False):
        """Ellipse (axes ``lx`` / ``ly``, rotated by ``angle``) as a stack of blocks.

        The shape is cut into horizontal slabs and each slab becomes one axis-aligned block
        whose centre and width are the ellipse's *exact* horizontal chord at the slab's
        mid-height. Taking the chord at the middle of the slab rather than at its edge is what
        makes the error cancel to first order, so the decomposition converges like
        ``n_split**-2`` and does so monotonically.

        The slab edges are spaced uniformly in the contour angle measured from the top vertex,
        not uniformly in ``y``, which keeps the slabs short where the boundary turns fastest —
        the caps — and is the same sampling the contour walk this replaced used.

        The decomposition is a stack in ``y``, so it has one resolution and not two;
        ``n_split_w`` and ``n_split_h`` were never independent either (the contour walk stacked
        the two samplings, so unequal counts raised), and the larger of the two is the slab
        count. ``angle_margin`` is accepted for call compatibility with :meth:`rectangle` and
        ignored: the chord is a closed form that stays exact and smooth through every angle, so
        there is no aligned special case to fall back to and nothing to wrap ``angle`` into.
        That also means ``angle`` carries a usable gradient here at every value, which is not
        true of :meth:`rectangle` (see its docstring). ``debug`` returns the slab edges and the
        chord centres and half-widths alongside the blocks.

        The previous decomposition sampled the contour and paired consecutive points into
        blocks. Each block took its left edge from one sample and its right edge from another
        at a *different* height, so the width error did not shrink with ``n_split``, it only
        changed sign — measured against the true chord it alternated by several percent of the
        radius at ``n_split`` 20 and again at 80, which is the oscillation this replaces. It
        also had a silent failure: when the sampling put a point exactly on the top vertex
        (an unrotated ellipse with ``n_split % 4 == 1``) the two sides came out level, the
        interleave produced nothing, and every block collapsed to the origin — the ellipse
        vanished from the layer with no error raised.
        """
        xp = self._ns(cx, cy, lx, ly, angle, self._period)
        dev = self._dev(cx, cy, lx, ly, angle)
        C, F = self._complex(xp), self._float(xp)

        # The ellipse (unlike the rectangle) carries its axes as complex from the start.
        cx = _shape_param(xp, cx, C, dev)
        cy = _shape_param(xp, cy, C, dev)
        lx = _shape_param(xp, lx, C, dev)
        ly = _shape_param(xp, ly, C, dev)
        angle = _shape_param(xp, angle, F, dev)

        n_slab = max(int(n_split_w), int(n_split_h))
        if n_slab < 2:
            raise ValueError(
                f'an ellipse needs at least 2 slabs, got {n_slab}: raise n_split_w / n_split_h')

        a, b = lx / 2, ly / 2
        cos_a = xp.astype(xp.cos(angle), C)
        sin_a = xp.astype(xp.sin(angle), C)

        # Half-height of the rotated ellipse's bounding box. Every quantity below is written in
        # terms of it because the chord then needs no square root of a difference: at the angle
        # ``s`` from the top vertex the boundary is y = cy + R*cos(s), and the chord there is
        # exactly ``2*a*b*sin(s)/R`` wide, centred at ``cx + (a**2 - b**2)*cos*sin*cos(s)/R``.
        # (Substituting y into the rotated conic gives a discriminant of ``R**2 - (y - cy)**2``
        # over ``(a*b)**2``, which the substitution reduces to ``sin(s)**2`` outright.)
        R = xp.sqrt((b * cos_a) ** 2 + (a * sin_a) ** 2)

        s = xp.linspace(0., _PI, n_slab + 1, dtype=F, device=dev)
        s_edge = xp.astype(xp.reshape(s, (-1, 1)), C)
        s_mid = (s_edge[:-1] + s_edge[1:]) / 2

        y_edge = cy + R * xp.cos(s_edge)                 # descending: top vertex first
        x_center = cx + (a ** 2 - b ** 2) * cos_a * sin_a * xp.cos(s_mid) / R
        x_half = a * b * xp.sin(s_mid) / R

        res = [[[y_edge[i + 1], x_center[i] - x_half[i]],
                [y_edge[i], x_center[i] + x_half[i]], n_index]
               for i in range(n_slab)]

        if debug:
            return res, (y_edge, x_center, x_half)
        return res

    # ------------------------------------------------------------------------------ gridding

    def _take_shape_choice(self):
        """The next recorded shape choice to replay, or ``None`` while recording."""
        queue = getattr(self, '_shape_replay', None)
        return queue.pop(0) if queue else None

    def _record_shape_choice(self, choice):
        record = getattr(self, '_shape_record', None)
        if record is not None:
            record.append(choice)

    def _replay_grid(self, xp, dev, datatype, F, C, pmtvy_base, obj_list, plan):
        """Build the layer from a recorded :class:`VectorGridPlan`, comparing nothing.

        The edge *values* are still the traced ones — they are gathered from the blocks the
        plan names — and only their order, their sharing and each block's index range come from
        the record. That is the whole difference: the eager build reaches those three by
        comparing coordinates, which is exactly what a tracer cannot do.
        """
        if self.period is None:
            raise ValueError('period is not set')
        period = _cast(xp, self.period, F, dev)

        def edge_of(source, is_row):
            if source is None:                       # the closing period line
                return _cast(xp, xp.reshape(period[1 if is_row else 0], (1,)), datatype, dev)
            obj_index, corner, axis = source
            return obj_list[obj_index][corner][axis]

        row_list = [edge_of(src, True) for src in plan.row_sources]
        col_list = [edge_of(src, False) for src in plan.col_sources]

        _check_plan_still_fits(xp, row_list, col_list)

        n_row, n_col = len(row_list), len(col_list)
        ucell_layer = xp.ones((n_row, n_col), dtype=datatype, device=dev) * pmtvy_base
        rows = xp.arange(n_row, device=dev)
        cols = xp.arange(n_col, device=dev)

        for obj, (row_begin, row_end, col_begin, col_end) in zip(obj_list, plan.ranges):
            mask = (xp.reshape((rows >= row_begin) & (rows < row_end), (-1, 1))
                    & xp.reshape((cols >= col_begin) & (cols < col_end), (1, -1)))
            ucell_layer = xp.where(mask, _cast(xp, obj[2], datatype, dev), ucell_layer)

        return ucell_layer, _edge_list(xp, col_list, F, C, dev), _edge_list(xp, row_list, F, C, dev)

    def vector_per_layer_numeric(self, layer_info, x64=True, plan=None):
        """Reduce one layer's block list to ``(cell, x_list, y_list)``.

        Every distinct block edge becomes a grid line, and blocks that meet at a coordinate
        share the one line rather than getting one each, so ``x_list`` / ``y_list`` come back
        strictly increasing. See :func:`insert_edge` for why sharing is the right answer.

        Which edge becomes which line, and which two share one, are *discrete* decisions: they
        come out of comparing coordinates, and comparing a traced coordinate is what stops this
        function under ``jax.jit``. So they are separable from the arithmetic. Called without a
        ``plan`` this records them into :class:`VectorGridPlan` and hangs it on ``self.grid_plans``;
        called with one it replays them and never compares anything, which leaves only gathers
        and a ``where`` — all traceable. The recorded plan is read back through
        :meth:`~fieldwave._core.engine.CoreSolveMixin.freeze_geometry`, the same route the
        other frozen decisions take. What it costs: move the parameters far enough and the block
        edges reorder, at which point the plan describes a layer you no longer have. "Far
        enough" is not far — an ellipse frozen at ``lx`` 85 is already wrong at 110. Replaying
        eagerly raises :class:`StaleGeometryError` when that happens, since a plan that no
        longer fits produces an unsorted grid; a *traced* replay cannot check, so a long fit
        should re-freeze on a schedule rather than wait to be told.
        """
        pmtvy_base, obj_list = layer_info

        xp = self._ns(*[c for obj in obj_list for corner in obj[:2] for c in corner],
                      self._period)
        dev = self._dev(*[c for obj in obj_list for corner in obj[:2] for c in corner])

        # TODO: activate and apply 'x64' option thru this function and connect to meent class.
        datatype = xp.complex128 if x64 else xp.complex64

        F, C = self._float(xp), self._complex(xp)

        if plan is not None:
            return self._replay_grid(xp, dev, datatype, F, C, pmtvy_base, obj_list, plan)

        # Griding
        row_list = []
        col_list = []
        row_src = []
        col_src = []

        def insert_edge(seq, src, value, source):
            """Add one block edge to its sorted grid line list, reusing a line already there.

            A block's rows and columns are located below with ``list.index``, which returns the
            *first* match, so two blocks whose edges coincide have to agree on which line they
            mean. They get the same one: the block below a shared boundary ends on it and the
            block above begins on it, which is what abutting blocks are.

            This used to nudge the second edge off the first by a growing relative perturbation
            instead, leaving two lines a few times 1e-14 apart with a sliver of background
            between them. Every stack of slabs hit it — a decomposed ellipse has a shared
            boundary between every pair — so the grid came out about twice as long as the
            geometry: 49 lines each way at ``n_split`` 24, against 26 rows and 39 columns of
            real content. The nudge was also what made the edge lists non-monotonic, since it
            moved an edge past the twin it was then inserted in front of.

            Sharing is safe for autograd, which was the other reason given for the nudge.
            Where two edges coincide *identically* — abutting slabs, two objects built to touch
            — they carry the same derivative and dropping one changes nothing;
            ``test_two_objects_sharing_an_edge_keep_their_gradient`` measures it. Where they
            merely cross at this parameter value, the layer has a genuine kink there and
            autograd returns the one-sided derivative, exactly as it did with the nudge.
            """
            index = _bisect_left(seq, value)
            if len(seq) > index and bool(value == seq[index]):
                return seq[index]
            seq.insert(index, value)
            src.insert(index, source)
            return value

        for obj_index, obj in enumerate(obj_list):
            top_left, bottom_right, _ = obj

            top_left[0] = insert_edge(row_list, row_src, top_left[0], (obj_index, 0, 0))
            bottom_right[0] = insert_edge(row_list, row_src, bottom_right[0], (obj_index, 1, 0))
            top_left[1] = insert_edge(col_list, col_src, top_left[1], (obj_index, 0, 1))
            bottom_right[1] = insert_edge(col_list, col_src, bottom_right[1], (obj_index, 1, 1))

        if self.period is None:
            raise ValueError('period is not set')

        period = _cast(xp, self.period, self._float(xp), dev)

        if not row_list or bool(row_list[-1] != period[1]):
            row_list.append(_cast(xp, xp.reshape(period[1], (1,)), datatype, dev))
            row_src.append(None)
        if not col_list or bool(col_list[-1] != period[0]):
            col_list.append(_cast(xp, xp.reshape(period[0], (1,)), datatype, dev))
            col_src.append(None)

        if row_list and bool(row_list[0] == 0):
            row_list, row_src = row_list[1:], row_src[1:]
        if col_list and bool(col_list[0] == 0):
            col_list, col_src = col_list[1:], col_src[1:]

        n_row, n_col = len(row_list), len(col_list)
        ucell_layer = xp.ones((n_row, n_col), dtype=datatype, device=dev) * pmtvy_base

        rows = xp.arange(n_row, device=dev)
        cols = xp.arange(n_col, device=dev)

        ranges = []
        for obj in obj_list:
            top_left, bottom_right, pmty = obj

            row_begin = 0 if bool(top_left[0] == 0) else row_list.index(top_left[0]) + 1
            row_end = row_list.index(bottom_right[0]) + 1

            col_begin = 0 if bool(top_left[1] == 0) else col_list.index(top_left[1]) + 1
            col_end = col_list.index(bottom_right[1]) + 1
            ranges.append((row_begin, row_end, col_begin, col_end))

            # The block bounds are integer indices, so the mask is static; the value written
            # through it is not, and stays differentiable.
            mask = (xp.reshape((rows >= row_begin) & (rows < row_end), (-1, 1))
                    & xp.reshape((cols >= col_begin) & (cols < col_end), (1, -1)))
            ucell_layer = xp.where(mask, _cast(xp, pmty, datatype, dev), ucell_layer)

        self._last_plan = VectorGridPlan(row_sources=tuple(row_src), col_sources=tuple(col_src),
                                        ranges=tuple(ranges))

        return (ucell_layer, _edge_list(xp, col_list, F, C, dev),
                _edge_list(xp, row_list, F, C, dev))

    def draw(self, layer_info_list, plans=None):
        """Grid every layer. Returns a list of ``[cell, x_list, y_list]``, bottom layer first.

        ``plans`` is a per-layer sequence of :class:`VectorGridPlan` to replay; omit it and the
        grid decisions are taken by comparison, which needs concrete coordinates. Either way
        the plans used end up on ``self.grid_plans``, which is what ``freeze_geometry`` reads.
        """
        ucell_info_list = []
        film_flags = []
        used_plans = []

        for i, layer_info in enumerate(layer_info_list):
            plan = None if plans is None else plans[i]
            ucell_layer, x_list, y_list = self.vector_per_layer_numeric(layer_info, plan=plan)
            used_plans.append(plan if plan is not None else self._last_plan)
            ucell_info_list.append([ucell_layer, x_list, y_list])
            # TODO: eig for homo layer
            film_flags.append(1.0 if x_list.shape[0] == y_list.shape[0] == 1 else 0.0)

        xp = self._ns(*[info[0] for info in ucell_info_list])
        dev = self._dev(*[info[0] for info in ucell_info_list])
        self.film_layer = xp.asarray(film_flags, dtype=self._float(xp), device=dev)
        self.ucell_info_list = ucell_info_list
        self.grid_plans = used_plans

        return ucell_info_list

    # ------------------------------------------------------------------------------ material

    def put_refractive_index_in_ucell(self, ucell, mat_list, wl, type_complex=None, device=None):
        """Replace the material *indices* of a raster ``ucell`` with refractive index values."""
        xp = self._ns(ucell)
        dev = device if device is not None else self._dev(ucell)
        C = self._complex(xp) if type_complex is None else (
            xp.complex128 if _precision_is_double(type_complex) else xp.complex64)

        ucell_mask = _cast(xp, ucell, C, dev)
        res = xp.zeros(ucell_mask.shape, dtype=C, device=dev)

        for i_mat, material in enumerate(mat_list):
            if isinstance(material, str):
                if not self.mat_table:
                    self.mat_table = read_material_table()
                assign_value = find_nk_index(material, self.mat_table, wl)
            else:
                assign_value = material

            res = xp.where(ucell_mask == i_mat, _cast(xp, assign_value, C, dev), res)

        return res

    def modeling_vector_instruction(self, instructions, plans=None, shape_choices=None):
        """Run a full instruction set: ``[[base_index, [[shape, *params], ...]], ...]``."""
        self._shape_replay = list(shape_choices) if shape_choices is not None else None
        self._shape_record = None if shape_choices is not None else []
        layer_info_list = []
        for layer in instructions:
            obj_list_per_layer = []
            base_refractive_index = layer[0]
            for vector_object in layer[1]:
                func = getattr(self, vector_object[0])
                obj_list_per_layer += func(*vector_object[1:])

            layer_info_list.append([base_refractive_index, obj_list_per_layer])

        self.shape_choices = (list(shape_choices) if shape_choices is not None
                              else self._shape_record)
        self._shape_replay = self._shape_record = None
        return self.draw(layer_info_list, plans=plans)


# ------------------------------------------------------------------------------ material table
#
# The nk tables are static data read off disk, not design variables, so they are held and
# interpolated in NumPy: there is no Array-API ``interp``, and nothing downstream
# differentiates through the table lookup.

def find_nk_index(material, mat_table, wl):
    """Complex refractive index of ``material`` at ``wl``, interpolated from the nk table.

    A ``__real`` suffix on the name asks for ``n`` only (no absorption).
    """
    if material[-6:] == '__real':
        material = material[:-6]
        n_only = True
    else:
        n_only = False

    mat_data = mat_table[material.upper()]
    n_index = np.interp(wl, mat_data[:, 0], mat_data[:, 1])

    if n_only:
        return n_index

    k_index = np.interp(wl, mat_data[:, 0], mat_data[:, 2])

    return n_index + 1j * k_index


def read_material_table(nk_path=None, type_complex=None):
    """Load every nk table under ``nk_path`` (defaults to the bundled ``nk_data``)."""
    dtype = np.float64 if _precision_is_double(type_complex) else np.float32

    mat_table = {}

    if nk_path is None:
        nk_path = str(Path(__file__).resolve().parent.parent) + '/nk_data'

    full_path_list, name_list = [], []
    for (dirpath, _dirnames, filenames) in walk(nk_path):
        full_path_list.extend([f'{dirpath}/{filename}' for filename in filenames])
        name_list.extend(filenames)

    for path, name in zip(full_path_list, name_list):
        if name[-3:] == 'txt':
            data = np.loadtxt(path, skiprows=1)
            mat_table[name[:-4].upper()] = data.astype(dtype)

        elif name[-3:] == 'mat':
            from scipy.io import loadmat
            data = loadmat(path)
            data = np.array([data['WL'], data['n'], data['k']], dtype=dtype)[:, :, 0].T
            mat_table[name[:-4].upper()] = data

    return mat_table
