"""Coordinates that crowd their lines at material walls (adaptive spatial resolution).

A layer whose walls run along the cell's axes is solved in coordinates ``(u, v)`` with
``x = F(u)`` and ``y = G(v)``, each a monotone function that advances slowly at the walls, so
the retained Fourier orders resolve the field there more finely than their count says. With
``f = F'`` and ``g = G'``, and the components ``E~_x = f E_x``, ``E~_y = g E_y``,
``E~_z = E_z`` (``H`` alike), Maxwell's equations keep their form in ``(u, v, z)`` and only the
materials change::

    eps~ = eps diag(g/f, f/g, f g),        mu~ = mu diag(g/f, f/g, f g)

A wall stays on a coordinate line, so Li's rules nest as they do without the stretch.

**The function.** With an axis's walls at ``x_1 < ... < x_K`` on the period and at
``u_1 < ... < u_K`` in the computational cell, on the piece from wall ``l`` to wall ``l + 1``
(the last piece runs to the first wall's periodic image)::

    x(u) = x_l + (dx/du)(u - u_l) + (G du - dx)/(2 pi) sin(2 pi (u - u_l)/du),
    dx = x_{l+1} - x_l,   du = u_{l+1} - u_l

so ``x'(u) = G`` at every wall, ``x`` is C^1 and ``x(u + P) = x(u) + P``. This is the
linear-plus-sine function of Granet's adaptive spatial resolution in the form of Vallius and
Honkanen (Opt. Express 10, 24 (2002)). The ``u_l`` are equidistant, ``du = P/K``: every piece
gets the same share of the coordinate lines whatever its physical width. ``G`` is the caller's
``wall_resolution``.

Everything here is geometry on concrete numbers (NumPy, double precision): the walls come from
the rectangles' edges, which the stretched solve reads as plain floats. What depends on a
material or on the incident wave is assembled by the callers on their own backend from the
tables built here.
"""
import math

import numpy as np

#: Gauss-Legendre nodes on each smooth piece of an integrand. The integrands are a piece of the
#: sine profile (or its reciprocal, which peaks at the piece's ends, where the nodes crowd)
#: times a harmonic of at most ``4 * half_width`` periods over the cell; the count below is
#: taken per piece and grows with the harmonic.
_BASE_NODES = 96


def _nodes(count):
    x, w = np.polynomial.legendre.leggauss(count)
    return 0.5 * (x + 1.0), 0.5 * w                      # on [0, 1]


class Axis:
    """One axis's coordinate function ``x = F(u)`` on the cell ``[0, period)``, with ``F(0) = 0``.

    ``walls`` are the wall positions on the period (any order, any whole-period image, repeats
    allowed); an axis without walls is the identity. ``slope`` is ``F'`` at every wall.
    """

    def __init__(self, walls, period, slope):
        period = float(period)
        slope = float(slope)
        if not (period > 0 and math.isfinite(period)):
            raise ValueError('a stretched axis needs a positive finite period')
        if not 0 < slope <= 1:
            raise ValueError('the slope at the walls must be in (0, 1]')
        folded = sorted({round((float(w) / period) % 1.0, 14) % 1.0 for w in walls})
        self.period, self.slope = period, slope
        self.walls = np.asarray(folded, dtype=float) * period
        count = len(folded)
        self.identity = count == 0
        if self.identity:
            self.wall_u = np.zeros((0,))
            return
        # Pieces before the origin is fixed: wall l at l P / K. `_x0`, `_dx` and the common
        # `_du` describe F_0, and F(u) = F_0(u + shift) with F_0(shift) = 0.
        self._du = period / count
        self._x0 = self.walls
        self._dx = np.diff(np.append(self.walls, self.walls[0] + period))
        if np.any(2.0 * self._dx / self._du <= slope):
            raise ValueError(
                'two walls are closer than the stretch can hold: the coordinate would not rise '
                f'between walls {np.min(self._dx):g} apart on a piece {self._du:g} wide at slope '
                f'{slope:g}')
        self._shift = 0.0
        self._shift = self._solve(0.0, guess=None)
        self.wall_u = np.sort((np.arange(count) * self._du - self._shift) % period)

    def _raw(self, t):
        """``(F_0, F_0')`` at ``t``, any period."""
        t = np.asarray(t, dtype=float)
        turns = np.floor(t / self.period)
        local = t - turns * self.period
        index = np.minimum((local / self._du).astype(int), len(self._x0) - 1)
        inside = local - index * self._du
        dx = self._dx[index]
        swing = self.slope * self._du - dx
        phase = 2.0 * np.pi * inside / self._du
        value = (self._x0[index] + dx / self._du * inside + swing / (2.0 * np.pi) * np.sin(phase)
                 + turns * self.period)
        return value, dx / self._du + swing / self._du * np.cos(phase)

    def _solve(self, x, guess):
        """``t`` with ``F_0(t + shift) = x``, by bisection on the monotone function and Newton."""
        x = np.asarray(x, dtype=float)
        if guess is None:
            # F_0(t) - t is bounded by the period, so the root is within one period of x.
            low, high = x - 2.0 * self.period, x + 2.0 * self.period
            for _ in range(60):
                middle = 0.5 * (low + high)
                below = self._raw(middle + self._shift)[0] < x
                low, high = np.where(below, middle, low), np.where(below, high, middle)
            guess = 0.5 * (low + high)
        for _ in range(3):
            value, slope = self._raw(guess + self._shift)
            guess = guess - (value - x) / slope
        return guess

    def F(self, u):
        """Physical coordinate of computational ``u``."""
        if self.identity:
            return np.asarray(u, dtype=float)
        return self._raw(np.asarray(u, dtype=float) + self._shift)[0]

    def dF(self, u):
        """``F'(u)``: the slope, ``slope`` at a wall."""
        if self.identity:
            return np.ones_like(np.asarray(u, dtype=float))
        return self._raw(np.asarray(u, dtype=float) + self._shift)[1]

    def inverse(self, x):
        """Computational coordinate of physical ``x`` (any period)."""
        if self.identity:
            return np.asarray(x, dtype=float)
        return self._solve(x, guess=None)

    def _breaks(self, low, high):
        """``[low, ..., high]`` cut at every wall strictly inside, in computational units."""
        if self.identity:
            return [low, high]
        first = math.floor((low - self.wall_u[0]) / self.period) - 1
        cuts = [w + turn * self.period for turn in (first, first + 1, first + 2, first + 3)
                for w in self.wall_u]
        span = self.period * 1e-13
        return [low] + sorted(c for c in cuts if low + span < c < high - span) + [high]

    def _integrals(self, low, high, orders, weight, harmonic):
        """``(1/P) int_low^high weight(u) exp(-2 pi i k u / P) du`` for each ``k`` of ``orders``.

        Gauss-Legendre on each piece between walls, where the integrand is analytic. ``harmonic``
        bounds ``|k|`` and sets the node count.
        """
        out = np.zeros(len(orders), dtype=complex)
        k = 2.0 * np.pi * np.asarray(orders, dtype=float) / self.period
        breaks = self._breaks(low, high)
        for a, b in zip(breaks[:-1], breaks[1:]):
            width = b - a
            if width <= 0:
                continue
            node, w = _nodes(_BASE_NODES + 2 * int(math.ceil(harmonic * width / self.period)))
            u = a + width * node
            out += (w * weight(u) * width / self.period) @ np.exp(-1j * np.outer(u, k))
        return out

    def _weight(self, name):
        return {'slope': self.dF, 'one': lambda u: np.ones_like(u),
                'reciprocal': lambda u: 1.0 / self.dF(u)}[name]

    def segment_table(self, edges, half_width, weight='slope'):
        """Weighted Fourier coefficients of the segments a layer's edges cut the period into.

        ``edges`` are right-hand physical edges on the unit cell, increasing, the last one ``1``
        (what :func:`~wavebit.rcwa._engine.rectilinear.layer_partition` gives). Returns
        ``(4 * half_width + 1, n_segments)``: entry ``[k, j]`` is the coefficient of order
        ``k - 2 * half_width`` of the function that is ``weight`` on segment ``j`` of the
        *computational* cell and zero elsewhere, so that a profile holding ``a_j`` on segment
        ``j`` has the coefficients ``table @ a``. With the identity axis this is the closed
        form of the unstretched segment integral.
        """
        orders = list(range(-2 * half_width, 2 * half_width + 1))
        right = np.asarray(edges, dtype=float).reshape(-1) * self.period
        left = np.concatenate([[0.0], right[:-1]])
        table = np.zeros((len(orders), len(right)), dtype=complex)
        function = self._weight(weight)
        for j, (a, b) in enumerate(zip(left, right)):
            if b <= a:
                continue
            ua, ub = float(self.inverse(a)), float(self.inverse(b))
            if not self.identity:
                # An edge that is a wall sits exactly on its piece's end.
                ua, ub = self._snap(ua), self._snap(ub)
                if j == len(right) - 1:
                    ub = self.period
            table[:, j] = self._integrals(ua, ub, orders, function, 2 * half_width)
        return table

    def _snap(self, u):
        near = self.wall_u[np.argmin(np.abs(self.wall_u - u))] if len(self.wall_u) else u
        return float(near) if abs(near - u) < 1e-9 * self.period else u

    def period_table(self, half_width, weight):
        """Coefficients ``-2 hw .. 2 hw`` of ``weight`` over the whole period: ``(4 hw + 1,)``."""
        orders = list(range(-2 * half_width, 2 * half_width + 1))
        return self._integrals(0.0, self.period, orders, self._weight(weight), 2 * half_width)

    def wave_nodes(self, half_width):
        """Quadrature nodes and weights over the period for :func:`plane_wave_matrix`.

        ``(u, F(u), F'(u), weights)``, the weights summing to one. Piecewise Gauss-Legendre, so
        that the caller's backend only multiplies and sums.
        """
        if self.identity:
            count = 8 * half_width + 16
            u = (np.arange(count) + 0.5) / count * self.period
            return u, u, np.ones_like(u), np.full(count, 1.0 / count)
        us, ws = [], []
        breaks = self._breaks(0.0, self.period)
        for a, b in zip(breaks[:-1], breaks[1:]):
            node, w = _nodes(_BASE_NODES + 4 * int(math.ceil(
                (2 * half_width + 2) * (b - a) / self.period)))
            us.append(a + (b - a) * node)
            ws.append(w * (b - a) / self.period)
        u = np.concatenate(us)
        return u, self.F(u), self.dF(u), np.concatenate(ws)


def plane_wave_matrix(xp, axis_nodes, period, k_incident, half_width, weighted, dtype):
    """One axis's factor of the physical plane waves written in the computational basis.

    The wave of physical order ``q``, ``exp(i k_q x)`` at ``x = F(u)`` with
    ``k_q = k_incident + 2 pi q / P``, times ``F'`` when ``weighted`` (the field component along
    this axis carries the slope), has on the computational harmonic ``p`` the coefficient

        K[p, q] = (1/P) int_0^P w(u) exp(i (k_q F(u) - k_p u)) du,    k_p = k_incident + 2 pi p / P

    Returns ``K`` for ``p, q = -half_width .. half_width``. ``k_incident`` is in the reciprocal
    of ``period``'s unit and may be a backend scalar; the nodes are concrete
    (:meth:`Axis.wave_nodes`). The identity axis gives the identity matrix to rounding.
    """
    u, F, f, w = axis_nodes
    real = xp.float32 if dtype == xp.complex64 else xp.float64
    orders = xp.asarray([float(o) for o in range(-half_width, half_width + 1)], dtype=real)
    u, F, f, w = (xp.asarray(a, dtype=real) for a in (u, F, f, w))
    two_pi = 2.0 * math.pi / period
    bloch = xp.exp(1j * xp.asarray(k_incident, dtype=dtype) * xp.astype(F - u, dtype))
    weight = xp.astype(w * f if weighted else w, dtype) * bloch
    rows = xp.exp(-1j * xp.astype(two_pi * u[:, None] * orders[None, :], dtype))     # [i, p]
    columns = xp.exp(1j * xp.astype(two_pi * F[:, None] * orders[None, :], dtype))   # [i, q]
    return xp.matrix_transpose(rows * weight[:, None]) @ columns


class Stretch:
    """The stretch of a stack: one :class:`Axis` per direction, shared by every layer.

    ``walls_x`` / ``walls_y`` gather the wall positions of every patterned layer (cell
    coordinates, the cell's corner at 0). A wall of one layer is a harmless coordinate line in
    another.
    """

    def __init__(self, walls_x, walls_y, period_x, period_y, slope):
        self.slope = float(slope)
        self.x = Axis(walls_x, period_x, slope)
        self.y = Axis(walls_y, period_y, slope)

    @property
    def identity(self):
        return self.x.identity and self.y.identity

    def describe(self):
        """What was resolved, for a report: the slope and each axis's walls and their places."""
        return {'slope': self.slope,
                'walls_x': tuple(float(v) for v in self.x.walls),
                'walls_y': tuple(float(v) for v in self.y.walls),
                'computational_x': tuple(float(v) for v in self.x.wall_u),
                'computational_y': tuple(float(v) for v in self.y.wall_u)}


#: Pixels per period, at least, of the computational raster a matched layer's tensor is
#: sampled on (:meth:`Matched.raster`). The prototype gave the same R to five digits and the
#: same field error at 512, 1024 and 2048 with the matched lines on pixel edges.
_MATCHED_PIXELS = 512

#: Shape types :class:`Matched` takes. ``disk`` is ``circle``'s other name.
MATCHED_TYPES = ('circle', 'disk', 'ellipse')


class Matched:
    """Coordinates matched to one circle or axis-aligned ellipse, behind the stretch of :class:`Axis`.

    Two maps, composed. First each axis's stretch, ``(u, v) -> (s, t) = (F(u), G(v))``, with its
    walls at the lines ``s = cx +- c_x`` and ``t = cy +- c_y``, ``c = r / sqrt(2)``: so those
    lines sit half a period apart in ``u`` and ``v`` and the coordinates advance at the slope
    there. Then the mesh of Weiss et al. (Opt. Express 17, 8051 (2009)), as restated by Weiss
    (thesis, 2011, Eqs. 3.12-3.13) and by Kuechenmeister, Zebrowski and Busch (Opt. Express 20,
    17319 (2012)), which carries the rectangle
    ``|p| <= c_x, |q| <= c_y`` (``p = s - cx`` and ``q = t - cy`` wrapped onto the period; the
    rectangle's corners are the ellipse's 45 degree points) onto the ellipse::

        f_x(q) = r_x sqrt(1 - q^2 / r_y^2)   for |q| <= c_y,   c_x  otherwise
        |p| <= c_x:   X = (p / c_x) f_x(q)
        |p| >  c_x:   X = sign(p) [(h_x - |p|) f_x(q) + (|p| - c_x) h_x] / (h_x - c_x),   h_x = P_x / 2

    and ``Y`` the same with the axes exchanged; ``x = cx + X``, ``y = cy + Y``. The rectangle's
    four sides become the ellipse's four arcs, the four side patches stretch only across the
    wall, the four corner patches are the identity, and so is the cell's edge: the map is
    periodic. It is continuous and its Jacobian jumps on the four lines, across the whole cell.
    That is what Li's rules need: across a coordinate line of a continuous map the tangential
    ``E~`` and the normal ``D~`` stay continuous whatever the Jacobian does. At the rectangle's
    four corners the Jacobian is singular (a corner is flattened onto a smooth curve); the
    material tensor is sampled at pixel centres, never there.

    With ``J = d(x, y)/d(u, v)`` and the components ``E~ = J^T E`` (``E~_z = E_z``; ``H`` alike)
    Maxwell's equations keep their form and the materials become full in-plane tensors::

        eps~ = eps det(J) (J^T J)^-1  (in plane),   eps~_zz = eps det(J);   mu~ the same with mu

    Everything here is geometry on concrete numbers (NumPy, double precision), as in
    :class:`Axis`. ``centre`` is in cell coordinates, the corner at 0.
    """

    def __init__(self, centre, radii, periods, slope):
        cx, cy = (float(v) for v in centre)
        rx, ry = (float(v) for v in radii)
        px, py = (float(v) for v in periods)
        if not (0 < rx < 0.5 * px and 0 < ry < 0.5 * py):
            raise ValueError(
                f'an ellipse of semi-axes {rx:g} and {ry:g} does not fit inside a cell of '
                f'{px:g} by {py:g}: the matched coordinates need room between its wall and the '
                'cell\'s edge on every side')
        self.slope = float(slope)
        self.centre, self.radii, self.period = (cx, cy), (rx, ry), (px, py)
        self.inner = (rx / math.sqrt(2.0), ry / math.sqrt(2.0))
        self.x = Axis([cx - self.inner[0], cx + self.inner[0]], px, slope)
        self.y = Axis([cy - self.inner[1], cy + self.inner[1]], py, slope)

    identity = False

    def describe(self):
        """What was resolved, for a report."""
        return {'slope': self.slope, 'matched': 'ellipse',
                'centre': tuple(float(v) for v in self.centre),
                'radii': tuple(float(v) for v in self.radii),
                'walls_x': tuple(float(v) for v in self.x.walls),
                'walls_y': tuple(float(v) for v in self.y.walls),
                'computational_x': tuple(float(v) for v in self.x.wall_u),
                'computational_y': tuple(float(v) for v in self.y.wall_u)}

    @staticmethod
    def _wrap(t, period):
        """``t`` on ``[-period / 2, period / 2)``."""
        return t - period * np.floor(t / period + 0.5)

    @staticmethod
    def _one(p, q, rp, rq, half):
        """One coordinate of the mesh, ``X(p, q)``, with ``dX/dp`` and ``dX/dq``."""
        cp, cq = rp / math.sqrt(2.0), rq / math.sqrt(2.0)
        aq = np.abs(q)
        along = aq <= cq
        root = np.sqrt(1.0 - (np.minimum(aq, cq) / rq) ** 2)        # >= 1/sqrt(2)
        f = np.where(along, rp * root, cp)
        df = np.where(along, -rp * q / (rq * rq * root), 0.0)
        ap, sign = np.abs(p), np.sign(p)
        inner = ap <= cp
        X = np.where(inner, p / cp * f, sign * ((half - ap) * f + (ap - cp) * half) / (half - cp))
        Xp = np.where(inner, f / cp, (half - f) / (half - cp))
        Xq = np.where(inner, p / cp * df, sign * (half - ap) * df / (half - cp))
        return X, Xp, Xq

    def at(self, u, v):
        """``(x, y, x_u, x_v, y_u, y_v, inside)`` at computational points (broadcast).

        ``inside`` is whether the point is in the ellipse. ``x - u`` and ``y - v`` are periodic.
        """
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        (cx, cy), (rx, ry), (px, py) = self.centre, self.radii, self.period
        s, f = self.x.F(u), self.x.dF(u)
        t, g = self.y.F(v), self.y.dF(v)
        p, q = self._wrap(s - cx, px), self._wrap(t - cy, py)
        X, Xp, Xq = self._one(p, q, rx, ry, 0.5 * px)
        Y, Yq, Yp = self._one(q, p, ry, rx, 0.5 * py)
        inside = (np.abs(p) < self.inner[0]) & (np.abs(q) < self.inner[1])
        return s + (X - p), t + (Y - q), Xp * f, Xq * g, Yp * f, Yq * g, inside

    def _mesh_inverse(self, X, Y):
        """``(p, q)`` of the mesh's point ``(X, Y)``, both wrapped about the centre. Closed form."""
        (rx, ry), (cpx, cpy) = self.radii, self.inner
        hx, hy = 0.5 * self.period[0], 0.5 * self.period[1]
        ax, ay = np.abs(X), np.abs(Y)
        # Inside: X / rx = sqrt2 a sqrt(1 - b^2), Y / ry = sqrt2 b sqrt(1 - a^2) with a = p / rx,
        # b = q / ry; with S, T the left sides over sqrt2, a^2 - b^2 = S^2 - T^2 = d, and a^2 and
        # b^2 are the smaller roots of A^2 - (1 + d) A + S^2 = 0 and B^2 - (1 - d) B + T^2 = 0,
        # each taken in the form that does not cancel (B = A - d would, where T is small).
        inside = (X / rx) ** 2 + (Y / ry) ** 2 <= 1.0
        S = np.where(inside, X, 0.0) / (math.sqrt(2.0) * rx)
        T = np.where(inside, Y, 0.0) / (math.sqrt(2.0) * ry)
        d = S * S - T * T
        A = 2.0 * S * S / ((1.0 + d) + np.sqrt(np.maximum((1.0 + d) ** 2 - 4.0 * S * S, 0.0)))
        B = 2.0 * T * T / ((1.0 - d) + np.sqrt(np.maximum((1.0 - d) ** 2 - 4.0 * T * T, 0.0)))
        p_in, q_in = np.sign(S) * np.sqrt(A) * rx, np.sign(T) * np.sqrt(B) * ry
        # A side patch keeps the coordinate along the wall and is linear across it.
        fx = rx * np.sqrt(1.0 - (np.minimum(ay, cpy) / ry) ** 2)
        fy = ry * np.sqrt(1.0 - (np.minimum(ax, cpx) / rx) ** 2)
        p_side = np.sign(X) * (ax * (hx - cpx) - hx * fx + cpx * hx) / (hx - fx)
        q_side = np.sign(Y) * (ay * (hy - cpy) - hy * fy + cpy * hy) / (hy - fy)
        side_x = ~inside & (ay <= cpy)
        side_y = ~inside & ~side_x & (ax <= cpx)
        p = np.where(inside, p_in, np.where(side_x, p_side, X))
        q = np.where(inside, q_in, np.where(side_y, q_side, Y))
        return p, q

    def inverse(self, x, y):
        """Computational ``(u, v)`` of physical points ``(x, y)`` (broadcast, any period)."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        (cx, cy), (px, py) = self.centre, self.period
        X, Y = self._wrap(x - cx, px), self._wrap(y - cy, py)
        p, q = self._mesh_inverse(X, Y)
        # x = s + (X - p) with s = F(u), and the same whole periods as x.
        return self.x.inverse(x + (p - X)), self.y.inverse(y + (q - Y))

    def _raster_axis(self, axis, half_width):
        """``(right edges, centres)`` of one axis's pixels, the matched lines on pixel edges.

        ``M`` equal pixels, ``M`` even so that both lines (half a period apart) are edges; the
        first pixel edge is a line, so the pixel across ``u = 0`` is cut in two segments, which
        hold the same value.
        """
        period = axis.period
        count = _MATCHED_PIXELS
        while count < 16 * (2 * half_width + 1):
            count *= 2
        step = period / count
        start = float(axis.wall_u[0]) % step
        bounds = start + np.arange(count) * step
        if start < 1e-12 * period:
            return np.append(bounds[1:], period), bounds + 0.5 * step
        centres = np.concatenate([[bounds[-1] + 0.5 * step - period], bounds + 0.5 * step])
        return np.append(bounds, period), centres

    def raster(self, half_u, half_v):
        """The unit tensor ``det(J) (J^T J)^-1`` and ``det(J)`` on the computational raster.

        Returns ``(edges_u, edges_v, unit, inside)``: right-hand segment edges per axis (for
        :func:`~wavebit.rcwa._engine.tensor_li.layer_matrices`), the ``(rows, columns, 3, 3)``
        tensor of a medium of permittivity 1 (rows along ``v``), and the ellipse's pixels.
        """
        edges_u, centres_u = self._raster_axis(self.x, half_u)
        edges_v, centres_v = self._raster_axis(self.y, half_v)
        U, V = np.meshgrid(centres_u, centres_v)
        _, _, xu, xv, yu, yv, inside = self.at(U, V)
        det = xu * yv - xv * yu
        g11, g12, g22 = xu * xu + yu * yu, xu * xv + yu * yv, xv * xv + yv * yv
        unit = np.zeros(U.shape + (3, 3))
        unit[..., 0, 0], unit[..., 1, 1] = g22 / det, g11 / det
        unit[..., 0, 1] = unit[..., 1, 0] = -g12 / det
        unit[..., 2, 2] = det
        return edges_u, edges_v, unit, inside

    def wave_columns(self, xp, k_u, k_v, half_u, half_v, orders, dtype):
        """Physical plane waves of the given orders written in the computational basis.

        For order ``(m, n)``, ``exp(i (k_x x + k_y y))`` with ``k_x = k_u + 2 pi m / P_x`` and
        ``k_y = k_v + 2 pi n / P_y``, the computational components of a field along ``x`` are
        ``x_u`` and ``x_v`` times that wave, of one along ``y`` ``y_u`` and ``y_v`` times it
        (``E~ = J^T E``). Their coefficients on the computational harmonics
        ``exp(i (k_u + 2 pi p / P_x) u + i (k_v + 2 pi r / P_y) v)`` are the integrals::

            (1 / (P_x P_y)) int int w(u, v) exp(i (k_x x + k_y y - k_u u - k_v v)) exp(-2 pi i (p u / P_x + r v / P_y)) du dv

        on Gauss-Legendre nodes per patch, where the integrand is smooth. Returns
        ``(T_ux, T_uy, T_vx, T_vy)``, each ``((2 half_v + 1) (2 half_u + 1), len(orders))``
        with rows ``r * (2 half_u + 1) + p`` as the orders run elsewhere: ``T_ux`` is ``x_u``'s,
        ``T_uy`` ``y_u``'s, ``T_vx`` ``x_v``'s, ``T_vy`` ``y_v``'s. ``k_u`` and ``k_v`` may be
        backend scalars.
        """
        u, _, _, wu = self.x.wave_nodes(half_u)
        v, _, _, wv = self.y.wave_nodes(half_v)
        V, U = np.meshgrid(v, u, indexing='ij')
        x, y, xu, xv, yu, yv, _ = self.at(U, V)
        real = xp.float32 if dtype == xp.complex64 else xp.float64
        px, py = self.period
        weight = xp.asarray(wv[:, None] * wu[None, :], dtype=real)
        dx, dy = xp.asarray(x - U, dtype=real), xp.asarray(y - V, dtype=real)
        X, Y = xp.asarray(x, dtype=real), xp.asarray(y, dtype=real)
        rows_u = xp.exp(-1j * xp.astype(xp.asarray(
            2.0 * math.pi / px * np.outer(np.arange(-half_u, half_u + 1), u), dtype=real), dtype))
        rows_v = xp.exp(-1j * xp.astype(xp.asarray(
            2.0 * math.pi / py * np.outer(np.arange(-half_v, half_v + 1), v), dtype=real), dtype))
        k_u, k_v = xp.asarray(k_u, dtype=dtype), xp.asarray(k_v, dtype=dtype)
        bloch = xp.exp(1j * (k_u * xp.astype(dx, dtype) + k_v * xp.astype(dy, dtype)))
        slopes = [xp.astype(xp.asarray(a, dtype=real) * weight, dtype) * bloch
                  for a in (xu, yu, xv, yv)]
        out = [[] for _ in slopes]
        for m, n in orders:
            wave = xp.exp(1j * xp.astype(2.0 * math.pi * (m / px * X + n / py * Y), dtype))
            for column, slope in zip(out, slopes):
                coefficients = rows_v @ (slope * wave) @ xp.matrix_transpose(rows_u)
                column.append(xp.reshape(coefficients, (-1,)))
        return tuple(xp.stack(column, axis=1) for column in out)


def matched_layer_matrices(xp, matched, inside, outside, half_u, half_v, dtype, raster=None):
    """The five Li slots of a layer in :class:`Matched` coordinates.

    ``inside`` and ``outside`` are the permittivities of the ellipse and around it (equal for a
    uniform medium; ``1`` and ``1`` give the permeability of vacuum). The layer is the tensor
    ``scalar * unit`` of :meth:`Matched.raster`, which jumps on the matched lines only, and it
    goes through the tensor Li of shapes with those pixels' edges as segment edges. ``raster``
    reuses an earlier :meth:`Matched.raster` of the same orders.
    """
    from wavebit.rcwa._engine import tensor_li
    edges_u, edges_v, unit, mask = matched.raster(half_u, half_v) if raster is None else raster
    real = xp.float32 if dtype == xp.complex64 else xp.float64
    inside = xp.asarray(inside, dtype=dtype)
    outside = xp.asarray(outside, dtype=dtype)
    scalar = xp.where(xp.asarray(mask), inside, outside)
    tensor = scalar[..., None, None] * xp.astype(xp.asarray(unit, dtype=real), dtype)
    edges = (xp.asarray(edges_u[:, None], dtype=real), xp.asarray(edges_v[:, None], dtype=real))
    # The map is in-plane: `unit` has no xz, yz, zx or zy entry, which the values cannot show
    # under a trace.
    in_plane = not np.any(np.asarray(unit)[..., :2, 2]) and not np.any(np.asarray(unit)[..., 2, :2])
    return tuple(tensor_li.layer_matrices(xp, tensor, half_u, half_v, edges, in_plane=in_plane))
