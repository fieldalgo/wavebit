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
from pathlib import Path

import numpy as np

from .backend import array_namespace

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

        Within ``angle_margin`` of a multiple of pi/2 the exact single-block form is returned;
        otherwise the shape is split into an upper triangle, a mid parallelogram and a lower
        triangle, each sampled into ``n_split_*`` blocks.

        **``angle`` carries no gradient inside the margin**, and that is the lesser of two
        evils rather than an oversight. The block decomposition of a nearly-aligned rectangle
        misses the true shape by a staircase whose area grows like ``|angle|``, and refining
        ``n_split`` does not remove it: measured against the analytic path, the decomposition
        adds a spurious ``0.21*|angle|`` to the diffraction efficiency (stable from
        ``n_split`` 4 through 64) on top of a true derivative of ``-0.011``. A one-sided
        difference through the rotated branch therefore reports about ``+-0.2`` — twenty times
        the real slope, with the sign set by the direction of approach — while the exact
        branch reports zero. A central difference still recovers the derivative, because the
        ``|angle|`` term cancels; that is why this is invisible away from the margin.

        So: to fit a sidewall angle near normal, use the analytic path
        (``ucell=[dict(n_base=..., shapes=[...])]``), which builds the rotated shape's Fourier
        coefficients in closed form and is smooth through zero.
        """
        xp = self._ns(cx, cy, lx, ly, angle, self._period)
        dev = self._dev(cx, cy, lx, ly, angle)
        C, F = self._complex(xp), self._float(xp)

        # Lengths stay real (the legacy cast them to the float, not the complex, type).
        lx = _shape_param(xp, lx, F, dev)
        ly = _shape_param(xp, ly, F, dev)
        angle = _shape_param(xp, angle, F, dev)

        angle = angle % (2 * _PI)

        # No rotation (within the margin): exact, no splitting needed. The range runs to 4,
        # not 3: ``angle`` has already been wrapped into [0, 2*pi), so a small *negative*
        # angle arrives as ``2*pi - eps`` and only a target at 2*pi catches it. Without that
        # last target, -1e-6 took the rotated branch while +1e-6 took the exact one, and the
        # two disagreed by 2e-7 in efficiency — a discontinuity sitting exactly on the
        # unrotated case that any angle fit passes through.
        abs_angle = xp.abs(angle) % (2 * _PI)
        for k in range(5):
            target = k * _PI / 2
            if bool(target - angle_margin <= abs_angle) and \
                    bool(abs_angle <= target + angle_margin):
                if k % 2:  # a quarter turn swaps the side lengths
                    return self.rectangle_no_approximation(cx, cy, ly, lx, n_index)
                return self.rectangle_no_approximation(cx, cy, lx, ly, n_index)

        # Yes rotation.
        cos_a = xp.astype(xp.cos(angle), C)
        sin_a = xp.astype(xp.sin(angle), C)
        rotate = xp.stack([xp.concat([cos_a, -sin_a]), xp.concat([sin_a, cos_a])])

        def _corner(sx, sy):
            # Cast before the matmul: torch (unlike NumPy) will not promote mixed operands.
            return rotate @ xp.astype(xp.stack([sx, sy]), C)

        UR = _corner(lx / 2, ly / 2)
        RD = _corner(lx / 2, -ly / 2)
        DL = _corner(-lx / 2, -ly / 2)
        LU = _corner(-lx / 2, ly / 2)

        center = xp.stack([_as1(xp, cx, C, dev), _as1(xp, cy, C, dev)])
        UR, RD, DL, LU = UR + center, RD + center, DL + center, LU + center

        if bool(angle < _PI / 2):
            angle_inside = (_PI / 2) - angle
            top1, top4 = UR, DL                       # trail = L + U
            if bool(LU[1].real > RD[1].real):
                top2, top3 = LU, RD
                length_top12 = lx  # length_top24 = ly (unused)
                top2_left = True
            else:
                top2, top3 = RD, LU
                length_top12 = ly  # length_top24 = lx (unused)
                top2_left = False
        elif bool(angle < _PI):
            angle_inside = _PI - angle
            top1, top4 = RD, LU                       # trail = U + R
            if bool(UR[1].real > DL[1].real):
                top2, top3 = UR, DL
                length_top12 = ly  # length_top24 = lx (unused)
                top2_left = True
            else:
                top2, top3 = DL, UR
                length_top12 = lx  # length_top24 = ly (unused)
                top2_left = False
        elif bool(angle < _PI / 2 * 3):
            angle_inside = (_PI * 3 / 2) - angle
            top1, top4 = DL, UR                       # trail = R + D
            if bool(RD[1].real > LU[1].real):
                top2, top3 = RD, LU
                length_top12 = lx  # length_top24 = ly (unused)
                top2_left = True
            else:
                top2, top3 = LU, RD
                length_top12 = ly  # length_top24 = lx (unused)
                top2_left = False
        elif bool(angle < _PI * 2):
            angle_inside = (_PI * 2) - angle
            top1, top4 = LU, RD                       # trail = D + L
            if bool(DL[1].real > UR[1].real):
                top2, top3 = DL, UR
                length_top12 = ly  # length_top24 = lx (unused)
                top2_left = True
            else:
                top2, top3 = UR, DL
                length_top12 = lx  # length_top24 = ly (unused)
                top2_left = False
        else:
            raise ValueError('angle')

        # Ramps over the split counts. Small integer counts are exact in every dtype, so
        # creating them at the working complex type costs nothing.
        def _ramp(n, reverse=False):
            r = xp.arange(n, -1, -1, dtype=C, device=dev) if reverse \
                else xp.arange(n + 1, dtype=C, device=dev)
            return xp.reshape(r, (-1, 1))

        def _ones(n):
            return xp.ones((n + 1, 1), dtype=C, device=dev)

        ramp_t = _ramp(n_split_triangle)
        ramp_p = _ramp(n_split_parallelogram)
        ramp_t_rev = _ramp(n_split_triangle, reverse=True)

        # point in region 1 (top1~top2), 2 (top2~top3) and 3 (top3~top4)
        if top2_left:
            length = length_top12 / xp.sin(angle_inside)
            top3_cp = [top3[0] - length, top3[1]]

            # 1: Upper triangle
            xxx1 = top1[0] - (top1[0] - top2[0]) / n_split_triangle * ramp_t
            yyy1 = top1[1] - (top1[1] - top2[1]) / n_split_parallelogram * ramp_t
            xxx_cp1 = xxx1 + length / n_split_triangle * ramp_t
            yyy_cp1 = yyy1 * _ones(n_split_triangle)

            # 2: Mid parallelogram
            xxx2 = top2[0] + (top3_cp[0] - top2[0]) / n_split_triangle * ramp_p
            yyy2 = top2[1] - (top2[1] - top3_cp[1]) / n_split_parallelogram * ramp_p
            xxx_cp2 = (xxx2 + length) * _ones(n_split_parallelogram)
            yyy_cp2 = yyy2 * _ones(n_split_parallelogram)

            # 3: Lower triangle
            xxx3 = top3_cp[0] + (top4[0] - top3_cp[0]) / n_split_triangle * ramp_t
            yyy3 = top3_cp[1] - (top3_cp[1] - top4[1]) / n_split_parallelogram * ramp_t
            xxx_cp3 = xxx3 + length / n_split_triangle * ramp_t_rev
            yyy_cp3 = yyy3 * _ones(n_split_triangle)
        else:
            length = length_top12 / xp.cos(angle_inside)
            top3_cp = [top3[0] + length, top3[1]]

            # 1: Top triangle
            xxx1 = top1[0] + (top2[0] - top1[0]) / n_split_triangle * ramp_t
            yyy1 = top1[1] - (top1[1] - top2[1]) / n_split_parallelogram * ramp_t
            xxx_cp1 = xxx1 - length / n_split_triangle * ramp_t
            yyy_cp1 = yyy1 * _ones(n_split_triangle)

            # 2: Mid parallelogram
            xxx2 = top2[0] - (top2[0] - top3_cp[0]) / n_split_triangle * ramp_p
            yyy2 = top2[1] - (top2[1] - top3_cp[1]) / n_split_parallelogram * ramp_p
            xxx_cp2 = xxx2 - length * _ones(n_split_parallelogram)
            yyy_cp2 = yyy2 * _ones(n_split_parallelogram)

            # 3: Lower triangle
            xxx3 = top3_cp[0] - (top3_cp[0] - top4[0]) / n_split_triangle * ramp_t
            yyy3 = top3_cp[1] - (top3_cp[1] - top4[1]) / n_split_parallelogram * ramp_t
            xxx_cp3 = xxx3 - length / n_split_triangle * ramp_t_rev
            yyy_cp3 = yyy3 * _ones(n_split_triangle)

        xxx = xp.concat([xxx1, xxx2, xxx3])
        yyy = xp.concat([yyy1, yyy2, yyy3])
        xxx_cp = xp.concat([xxx_cp1, xxx_cp2, xxx_cp3])
        yyy_cp = xp.concat([yyy_cp1, yyy_cp2, yyy_cp3])

        x_mean_arr = (xxx + xp.roll(xxx, -1, axis=0)) / 2
        x_cp_mean_arr = (xxx_cp + xp.roll(xxx_cp, -1, axis=0)) / 2
        y_cp_next_arr = xp.roll(yyy_cp, -1, axis=0)

        if top2_left:
            return [[[y_cp_next_arr[i], x_mean_arr[i]], [yyy[i], x_cp_mean_arr[i]], n_index]
                    for i in range(xxx.shape[0] - 1)]
        return [[[y_cp_next_arr[i], x_cp_mean_arr[i]], [yyy[i], x_mean_arr[i]], n_index]
                for i in range(xxx.shape[0] - 1)]

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

    def vector_per_layer_numeric(self, layer_info, x64=True):
        """Reduce one layer's block list to ``(cell, x_list, y_list)``.

        Every distinct block edge becomes a grid line; coincident edges are nudged apart by a
        growing perturbation so the sort stays strict, and the returned ``x_list`` / ``y_list``
        are strictly increasing. See :func:`insert_edge` below for what the nudge is for.
        """
        pmtvy_base, obj_list = layer_info

        xp = self._ns(*[c for obj in obj_list for corner in obj[:2] for c in corner],
                      self._period)
        dev = self._dev(*[c for obj in obj_list for corner in obj[:2] for c in corner])

        # TODO: activate and apply 'x64' option thru this function and connect to meent class.
        if x64:
            datatype = xp.complex128
            perturbation = 0
            perturbation_unit = 1E-14
        else:
            datatype = xp.complex64
            perturbation = 0
            perturbation_unit = 1E-6

        # Griding
        row_list = []
        col_list = []

        def insert_edge(seq, value, downward):
            """Add one block edge to its sorted grid line list, nudged off any edge already there.

            The nudge is what the grid builder below depends on: it locates each block's rows
            and columns with ``list.index``, which returns the *first* match, so two edges
            sharing a coordinate would put two different blocks on one grid line. Nudging the
            later one keeps them distinct. ``downward`` moves a block's far edge down and its
            near edge up, so a block never grows into its neighbour.

            The insertion slot is taken *after* the nudge. Taking it before — which is what this
            did — puts an upward-nudged edge in front of the twin it has just moved past, and
            the edge list comes back out of order. It was easy to miss because the disorder is
            the size of the nudge: an unrotated ellipse at ``n_split`` 24 had 10 of its 49
            column edges inverted, by 8e-11 each, which reaches ``_cfs`` as a negative segment
            width. Harmless at ``perturbation_unit`` 1e-14; the ``x64=False`` branch below uses
            1e-6, where a few hundred collisions would put the disorder at ~0.1 length units.

            A nudge landing on a *third* edge is not chased; nothing here places distinct edges
            within a nudge of each other.
            """
            nonlocal perturbation
            index = _bisect_left(seq, value)
            if len(seq) > index and bool(value == seq[index]):
                # TODO: change; save how many perturbations were applied in a variable
                perturbation += perturbation_unit
                if downward:
                    value = value - (value * perturbation)
                elif bool(value == 0):
                    value = value + perturbation
                else:
                    value = value + (value * perturbation)
                index = _bisect_left(seq, value)
            seq.insert(index, value)
            return value

        # overlap check and apply perturbation
        for obj in obj_list:
            top_left, bottom_right, _ = obj

            top_left[0] = insert_edge(row_list, top_left[0], downward=False)
            bottom_right[0] = insert_edge(row_list, bottom_right[0], downward=True)
            top_left[1] = insert_edge(col_list, top_left[1], downward=False)
            bottom_right[1] = insert_edge(col_list, bottom_right[1], downward=True)

        if self.period is None:
            raise ValueError('period is not set')

        period = _cast(xp, self.period, self._float(xp), dev)

        if not row_list or bool(row_list[-1] != period[1]):
            row_list.append(_cast(xp, xp.reshape(period[1], (1,)), datatype, dev))
        if not col_list or bool(col_list[-1] != period[0]):
            col_list.append(_cast(xp, xp.reshape(period[0], (1,)), datatype, dev))

        if row_list and bool(row_list[0] == 0):
            row_list = row_list[1:]
        if col_list and bool(col_list[0] == 0):
            col_list = col_list[1:]

        n_row, n_col = len(row_list), len(col_list)
        ucell_layer = xp.ones((n_row, n_col), dtype=datatype, device=dev) * pmtvy_base

        rows = xp.arange(n_row, device=dev)
        cols = xp.arange(n_col, device=dev)

        for obj in obj_list:
            top_left, bottom_right, pmty = obj

            row_begin = 0 if bool(top_left[0] == 0) else row_list.index(top_left[0]) + 1
            row_end = row_list.index(bottom_right[0]) + 1

            col_begin = 0 if bool(top_left[1] == 0) else col_list.index(top_left[1]) + 1
            col_end = col_list.index(bottom_right[1]) + 1

            # The block bounds are integer indices, so the mask is static; the value written
            # through it is not, and stays differentiable.
            mask = (xp.reshape((rows >= row_begin) & (rows < row_end), (-1, 1))
                    & xp.reshape((cols >= col_begin) & (cols < col_end), (1, -1)))
            ucell_layer = xp.where(mask, _cast(xp, pmty, datatype, dev), ucell_layer)

        F, C = self._float(xp), self._complex(xp)
        x_parts = [_flat(xp, x, F, C, dev) for x in col_list]
        y_parts = [_flat(xp, y, F, C, dev) for y in row_list]

        x_dtype = _result_dtype(xp, [p.dtype for p in x_parts])
        y_dtype = _result_dtype(xp, [p.dtype for p in y_parts])

        x_list = xp.reshape(xp.concat([xp.astype(p, x_dtype) for p in x_parts]), (-1, 1))
        y_list = xp.reshape(xp.concat([xp.astype(p, y_dtype) for p in y_parts]), (-1, 1))

        return ucell_layer, x_list, y_list

    def draw(self, layer_info_list):
        """Grid every layer. Returns a list of ``[cell, x_list, y_list]``, bottom layer first."""
        ucell_info_list = []
        film_flags = []

        for layer_info in layer_info_list:
            ucell_layer, x_list, y_list = self.vector_per_layer_numeric(layer_info)
            ucell_info_list.append([ucell_layer, x_list, y_list])
            # TODO: eig for homo layer
            film_flags.append(1.0 if x_list.shape[0] == y_list.shape[0] == 1 else 0.0)

        xp = self._ns(*[info[0] for info in ucell_info_list])
        dev = self._dev(*[info[0] for info in ucell_info_list])
        self.film_layer = xp.asarray(film_flags, dtype=self._float(xp), device=dev)
        self.ucell_info_list = ucell_info_list

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

    def modeling_vector_instruction(self, instructions):
        """Run a full instruction set: ``[[base_index, [[shape, *params], ...]], ...]``."""
        layer_info_list = []
        for layer in instructions:
            obj_list_per_layer = []
            base_refractive_index = layer[0]
            for vector_object in layer[1]:
                func = getattr(self, vector_object[0])
                obj_list_per_layer += func(*vector_object[1:])

            layer_info_list.append([base_refractive_index, obj_list_per_layer])

        return self.draw(layer_info_list)


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
