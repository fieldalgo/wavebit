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
