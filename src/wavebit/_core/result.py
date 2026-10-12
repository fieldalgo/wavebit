"""The solve result — one backend-agnostic container, and the naming contract.

Naming follows the optics convention the rest of the ecosystem reads by sight:
**uppercase is power, lowercase is amplitude.**

``R``, ``T``
    Per-order diffraction efficiency, shape ``(n_orders_v, n_orders_u)`` with the incident order
    at the centre (``[half_width_v, half_width_u]``); index ``+1`` along the last axis is order
    ``m = +1``. ``R`` is the fraction of incident power returned into the incidence medium,
    ``T`` the fraction crossing the opposite interface. For real transverse wavevectors,
    each is normalized by the incident wave alone at the illuminated interface, including
    absorbing incident media. In that case ``1 - sum(R) - sum(T)`` need not be the finite
    stack's absorption: the net entering flux also includes incident/reflected interference.
    General complex-transverse-wavevector power integrals are not supported by this definition.

``R_s``, ``R_p``, ``T_s``, ``T_p``
    The same efficiencies resolved by outgoing polarization (``R = R_s + R_p``).

``power_entering``, ``absorption``
    Scalar power fractions summed over all orders. ``power_entering`` is the net
    flux into the finite stack, including incident/reflected interference, divided
    by incident-only power. ``absorption = power_entering - sum(T)`` covers all
    finite layers and excludes both half-spaces. Both are available at every
    answer level and are NaN when the transverse wavevector is complex.

``r_s``, ``r_p``, ``t_s``, ``t_p``
    Complex **Fresnel amplitudes** — the E-field ratio against a unit incident E-field, in
    ``e^{-iωt}``, directly comparable to the textbook Fresnel/Airy coefficients. Each order is
    resolved in **its own** s/p frame, the standard ``s_hat = zhat x khat_par``, on every path
    -- the pure 1D ``tetm`` path included since 2026-09-27; on the ``ky == 0`` row that frame is
    ``sign(Re kx) * yhat``, so an order's amplitude changes sign where its own ``kx`` crosses
    zero (the frame's singular point; ``R`` does not move). See ``docs/amplitude-sp-frame.md``,
    which is also where the 2026-08-16 fix to that frame on the ``Re(kx) < 0`` half-plane is
    recorded — it moved these four and nothing else, since a reversed frame cannot survive
    ``|r|**2`` — and the 2026-09-27 move of the pure 1D path, which moved these four and, at
    ``theta < 0``, the sign of that path's incident field. The p sign
    convention is the one where a single interface at normal incidence gives
    ``r_p = (n2 - n1)/(n2 + n1) = -r_s``. The solves construct these from their boundary
    variables at pack time (the p channel's boundary variable is a magnetic amplitude; the
    conversion differs between the pure-1D and the coupled paths and is measured to 6e-16
    against the Airy forms in ``tests/test_lossy_media.py``), so no caller ever needs to
    know a boundary variable existed. Phase reference: ``r`` at the illuminated interface, ``t`` at
    the opposite one; laterally, the cell centre ``x = y = 0`` -- the origin every public
    coordinate uses -- so each amplitude is a Fourier coefficient of the public field on its
    face, and the incidence is unit there. Until 2026-09-27 it was the cell corner (the
    kernels' Fourier origin): on a single incident wave every order with ``m + n`` odd changed
    sign that day, and ``R`` and ``T`` did not move.

``r_xyz``, ``t_xyz``
    The same outgoing waves as Cartesian E vectors, ``(Ex, Ey, Ez)`` on a last axis of length
    3: ``r_s * s_hat + r_p * p_hat`` in the fixed public frame (+z up). No per-order frame, so
    continuous where an order's ``kx`` crosses zero on the ``ky == 0`` row, where ``r_s`` turns
    over with ``s_hat``. Same normalisation and phase reference as ``r_s``: the outgoing E on the
    lit face is ``sum_m r_xyz[m] exp(1j k_m . x)``, ``x`` measured from the cell centre.

Until 2026-08 the efficiencies were named ``de_ri`` / ``de_ti`` (meent lineage) and
``R_s``/``R_p`` named the raw *boundary amplitudes* — the ones needing a ``/(-i n_top)``
footnote before they matched any textbook. Renamed outright before the first release;
nothing aliases the old names.
"""

import numpy as np

from .field_result import ResultFields
from .backend import eager_under_torch_compile


@eager_under_torch_compile
class ResultSub:
    """One incidence's per-order efficiencies and Fresnel amplitudes."""

    #: Positional: ``tree_unflatten`` rebuilds from this order, so a new field goes last.
    _FIELDS = ('r_s', 'r_p', 't_s', 't_p',
               'R', 'R_s', 'R_p', 'T', 'T_s', 'T_p', 'power_entering', 'r_xyz', 't_xyz')

    def __init__(self, r_s, r_p, t_s, t_p, R, R_s, R_p, T, T_s, T_p,
                 power_entering=None, r_xyz=None, t_xyz=None):
        self.r_s = r_s
        self.r_p = r_p
        self.t_s = t_s
        self.t_p = t_p
        self.R = R
        self.R_s = R_s
        self.R_p = R_p
        self.T = T
        self.T_s = T_s
        self.T_p = T_p
        self.power_entering = power_entering
        self.r_xyz = r_xyz
        self.t_xyz = t_xyz

    @property
    def absorption(self):
        """Net power absorbed in the finite layers, divided by incident power.

        Excludes absorption in either half-space. Negative values describe
        net generation or numerical error; they are not clipped to zero.
        NaN for complex transverse wavevectors, whose cell-power integral
        is not supported. None if entering power or transmitted power is absent.
        """
        if self.power_entering is None or self.T is None:
            return None
        return self.power_entering - self.T.sum()

    # pytree protocol (used by the JAX backend's registered subclass; harmless elsewhere)
    def tree_flatten(self):
        return tuple(getattr(self, name) for name in self._FIELDS), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@eager_under_torch_compile
class OrderView:
    """Everything the solve knows about one diffraction order, as scalars."""

    __slots__ = ('m', 'n', 'R', 'T', 'r_s', 'r_p', 't_s', 't_p', 'r_xyz', 't_xyz')

    def __init__(self, m, n, R, T, r_s, r_p, t_s, t_p, r_xyz=None, t_xyz=None):
        self.m, self.n = m, n
        self.R, self.T = R, T
        self.r_s, self.r_p, self.t_s, self.t_p = r_s, r_p, t_s, t_p
        #: The order's outgoing E as ``(Ex, Ey, Ez)``, length-3 vectors.
        self.r_xyz, self.t_xyz = r_xyz, t_xyz

    def __repr__(self):
        return (f'OrderView(m={self.m}, n={self.n}, R={self.R!r}, T={self.T!r}, '
                f'r_s={self.r_s!r}, r_p={self.r_p!r}, t_s={self.t_s!r}, t_p={self.t_p!r}, '
                f'r_xyz={self.r_xyz!r}, t_xyz={self.t_xyz!r})')


@eager_under_torch_compile
class Result(ResultFields):
    """Solve result: the actual incident state and the pure-s/p variants.

    ``incident`` holds the response to the solver's incident state, including the
    default pure-s state when no polarization was specified. ``s_inc`` / ``p_inc``
    hold the pure-s and pure-p responses when the solve computes those variants.
    The top-level ``R``, ``T``, ``r_s``, ``r_p``, ``t_s``, and ``t_p`` read the
    corresponding arrays from ``incident`` without copying them.

    A manually constructed result with ``incident=None`` averages the available
    pure-s/p power responses. It has no incident-state amplitude: the amplitude
    shortcuts raise rather than averaging complex coefficients.
    """

    def __init__(self, incident=None, s_inc=None, p_inc=None, order_window=None, field_state=None):
        self._field_state = field_state
        self.incident = incident
        self.s_inc = s_inc
        self.p_inc = p_inc
        #: The orders this result is on: ``((centre_u, half_u), (centre_v, half_v))``, the same
        #: shape :attr:`wavebit._core.engine.CoreSolveMixin.order_window` has. Static metadata
        #: for :meth:`order`; not a pytree child.
        #:
        #: The **retained** window rather than the declared ``truncation_order``, and the two
        #: part company as soon as an ``excitation`` widens one: :meth:`order` reads the array,
        #: so it has to describe the array. A result carrying the declaration instead indexed
        #: one order out and returned a plausible number.
        self.order_window = (None if order_window is None else
                             tuple((int(c), int(h)) for c, h in order_window))

    @property
    def orders_u(self):
        """The order index each **column** of ``R`` is, as a 1-D int array along ``a1``.

        ``R.shape == (len(orders_v), len(orders_u))``, and that is the point rather than a
        remark. Every array axis in this package runs ``v`` first, while every hand-written
        pair runs ``x`` first -- two rules that are each consistent and meet here, and the
        meeting is where readers go wrong. ``truncation_order=[4, 1]`` gives ``R.shape ==
        (3, 9)``; a reader who takes the leading ``4`` to mean the leading axis indexes the
        wrong entry and gets a plausible number back.

        Zipping against these labels turns that into a length mismatch on the spot. It is the
        same service the surveyed solvers provide -- ``basis_coefficients`` in fmmax,
        ``GetBasisSet()`` in S4, ``.order`` in RETICOLO, ``obj.G`` in grcwa -- adapted to a
        separable grid, where the labels factor into two vectors instead of an ``(N, 2)`` table.

        ``u`` and ``v`` rather than ``x`` and ``y`` because the order axes follow the cell:
        ``u`` runs along ``a1``, ``v`` along ``a2``, and they *are* x and y on a rectangular
        cell. Read from the retained window, so an ``excitation`` that widened or moved it is
        already accounted for -- see :attr:`order_window`.
        """
        return self._labels(0)

    @property
    def orders_v(self):
        """The order index each **row** of ``R`` is, as a 1-D int array along ``a2``."""
        return self._labels(1)

    def _labels(self, axis):
        if self.order_window is None:
            raise ValueError(
                'this result carries no order_window metadata, so its orders cannot be named')
        centre, half = self.order_window[axis]
        return np.arange(centre - half, centre + half + 1, dtype=int)

    # pytree protocol (see ResultSub)
    def tree_flatten(self):
        return (self.incident, self.s_inc, self.p_inc, self._field_state), self.order_window

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children[:3], order_window=aux_data, field_state=children[3])

    def _delegate(self, name):
        if self.incident is not None:
            return getattr(self.incident, name)
        te = getattr(self.s_inc, name) if self.s_inc is not None else None
        tm = getattr(self.p_inc, name) if self.p_inc is not None else None
        if te is not None and tm is not None:
            return 0.5 * (te + tm)
        return te if te is not None else tm

    def _incident_amplitude(self, name):
        if self.incident is None:
            raise ValueError(
                'complex amplitudes require an incident result; a power average has '
                'no single complex amplitude. Read s_inc or p_inc for a pure incidence.')
        return getattr(self.incident, name)

    @property
    def r_s(self):
        """Reflected s amplitudes for the incident state; None at efficiency-only level."""
        return self._incident_amplitude('r_s')

    @property
    def r_p(self):
        """Reflected p amplitudes for the incident state; None at efficiency-only level."""
        return self._incident_amplitude('r_p')

    @property
    def t_s(self):
        """Transmitted s amplitudes for the incident state; None at efficiency-only level."""
        return self._incident_amplitude('t_s')

    @property
    def t_p(self):
        """Transmitted p amplitudes for the incident state; None at efficiency-only level."""
        return self._incident_amplitude('t_p')

    @property
    def r_xyz(self):
        """Each reflected order's E as ``(Ex, Ey, Ez)``, shape ``r_s.shape + (3,)``.

        ``r_s * s_hat + r_p * p_hat`` in the fixed public frame: no per-order frame, so it is
        continuous where an order's ``kx`` crosses zero. Same normalisation and phase
        reference as ``r_s`` -- laterally the cell centre, ``x = y = 0``. None at
        efficiency-only level.
        """
        return self._incident_amplitude('r_xyz')

    @property
    def t_xyz(self):
        """Each transmitted order's E as ``(Ex, Ey, Ez)``; see :attr:`r_xyz`."""
        return self._incident_amplitude('t_xyz')

    @property
    def R(self):
        """Reflection efficiency per order; averages variants only if incident is absent."""
        return self._delegate('R')

    @property
    def T(self):
        """Transmission efficiency per order; averages variants only if incident is absent."""
        return self._delegate('T')

    @property
    def power_entering(self):
        """Net power crossing the illuminated interface into the finite stack.

        Includes incident/reflected interference, summed over orders and
        divided by incident-only power at that interface. Positive into the
        structure for either incidence side. NaN for complex transverse
        wavevectors. Available at every ``answers`` level.
        """
        return self._delegate('power_entering')

    @property
    def absorption(self):
        """Finite-stack absorption: ``power_entering - T.sum()``.

        Average of the available variants when no incident result exists. The absorbing
        half-spaces are excluded. NaN for complex transverse wavevectors.
        """
        return self._delegate('absorption')

    def order(self, m, n=0):
        """Scalars for diffraction order ``(m, n)`` of the solved (``incident``) incidence.

        Spares the caller the array layout: efficiencies and amplitudes are stored as
        ``(n_orders_v, n_orders_u)`` grids — note ``y`` first, the transpose of the
        ``truncation_order=[x, y]`` input order — centred on the specular order, or on
        :attr:`centre` when an ``excitation`` moved the retained window. This indexes them by
        the order indices themselves, which is the whole point: the array layout is not the
        caller's problem, and it stopped being derivable from the declared ``truncation_order`` the
        moment a window could widen.
        """
        sub = self.incident if self.incident is not None else (
            self.s_inc if self.s_inc is not None else self.p_inc)
        if sub is None:
            raise ValueError('no solved incidence to index into')
        if self.order_window is None:
            raise ValueError(
                'this result carries no order_window metadata; index the arrays directly')
        (cx, fx), (cy, fy) = self.order_window

        def at(flat):
            # One cell rule for both pickers: `flat`'s leading axis is the rows (`v`).
            row = flat[n - cy + fy if flat.shape[0] > 1 else 0]
            return row[m - cx + fx]

        def pick(a):
            return None if a is None else at(a.reshape(-1, a.shape[-1]))

        def pick_vector(a):
            # The component axis is last, so the order grid is the two before it: `pick`'s
            # `a.shape[-1]` would read the 3 as the u axis and index a component.
            return None if a is None else at(a.reshape(-1, a.shape[-2], a.shape[-1]))

        if not (cx - fx <= m <= cx + fx):
            raise IndexError(f'order m={m} outside the retained set [{cx - fx}, {cx + fx}]')
        n_v = None
        for a in (sub.R, sub.T):
            if a is not None:
                n_v = a.reshape(-1, a.shape[-1]).shape[0]
                break
        if n_v is not None and n_v > 1 and not (cy - fy <= n <= cy + fy):
            raise IndexError(f'order n={n} outside the retained set [{cy - fy}, {cy + fy}]')
        if n_v is not None and n_v == 1 and n != 0:
            raise IndexError('this solve retained a single row of orders; only n=0 exists')

        return OrderView(m, n, pick(sub.R), pick(sub.T),
                         pick(sub.r_s), pick(sub.r_p), pick(sub.t_s), pick(sub.t_p),
                         pick_vector(sub.r_xyz), pick_vector(sub.t_xyz))
