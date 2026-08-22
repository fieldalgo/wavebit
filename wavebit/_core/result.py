"""The solve result — one backend-agnostic container, and the naming contract.

Naming follows the optics convention the rest of the ecosystem reads by sight:
**uppercase is power, lowercase is amplitude.**

``R``, ``T``
    Per-order diffraction efficiency, shape ``(n_orders_2, n_orders_1)`` with the incident order
    at the centre (``[half_width_2, half_width_1]``); index ``+1`` along the last axis is order
    ``m = +1``. ``R`` is the fraction of incident power returned into the incidence medium,
    ``T`` the fraction leaving through the far side. For an absorbing superstrate the
    incident-power normalization is convention-dependent and these are not comparable to ``1`` —
    use the amplitudes there.

``R_s``, ``R_p``, ``T_s``, ``T_p``
    The same efficiencies resolved by outgoing polarization (``R = R_s + R_p``).

``r_s``, ``r_p``, ``t_s``, ``t_p``
    Complex **Fresnel amplitudes** — the E-field ratio against a unit incident E-field, in
    ``e^{-iωt}``, directly comparable to the textbook Fresnel/Airy coefficients. Each order is
    resolved in **its own** s/p frame, the standard ``s_hat = zhat x khat_par``; see
    ``docs/amplitude-sp-frame.md``, which is also where the 2026-08-16 fix to that frame on the
    ``Re(kx) < 0`` half-plane is recorded — it moved these four and nothing else, since a
    reversed frame cannot survive ``|r|**2``. The p sign
    convention is the one where a single interface at normal incidence gives
    ``r_p = (n2 - n1)/(n2 + n1) = -r_s``. The solves construct these from their boundary
    variables at pack time (the p channel's boundary variable is a magnetic amplitude; the
    conversion differs between the pure-1D and the coupled paths and is measured to 6e-16
    against the Airy forms in ``tests/test_lossy_media.py``), so no caller ever needs to
    know a boundary variable existed. Phase reference: ``r`` at the top interface, ``t`` at
    the bottom one.

Until 2026-08 the efficiencies were named ``de_ri`` / ``de_ti`` (meent lineage) and
``R_s``/``R_p`` named the raw *boundary amplitudes* — the ones needing a ``/(-i n_top)``
footnote before they matched any textbook. Renamed outright before the first release;
nothing aliases the old names.
"""


class ResultSub:
    """One incidence's per-order efficiencies and Fresnel amplitudes."""

    _FIELDS = ('r_s', 'r_p', 't_s', 't_p',
               'R', 'R_s', 'R_p', 'T', 'T_s', 'T_p')

    def __init__(self, r_s, r_p, t_s, t_p, R, R_s, R_p, T, T_s, T_p):
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

    # pytree protocol (used by the JAX backend's registered subclass; harmless elsewhere)
    def tree_flatten(self):
        return tuple(getattr(self, name) for name in self._FIELDS), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class OrderView:
    """Everything the solve knows about one diffraction order, as scalars."""

    __slots__ = ('m', 'n', 'R', 'T', 'r_s', 'r_p', 't_s', 't_p')

    def __init__(self, m, n, R, T, r_s, r_p, t_s, t_p):
        self.m, self.n = m, n
        self.R, self.T = R, T
        self.r_s, self.r_p, self.t_s, self.t_p = r_s, r_p, t_s, t_p

    def __repr__(self):
        return (f'OrderView(m={self.m}, n={self.n}, R={self.R!r}, T={self.T!r}, '
                f'r_s={self.r_s!r}, r_p={self.r_p!r}, t_s={self.t_s!r}, t_p={self.t_p!r})')


class Result:
    """Solve result: the specified incidence plus the pure-s/p variants.

    ``given`` holds the incidence actually asked for — by ``psi``, or by the complex
    amplitude pair ``polarization=(a_s, a_p)``; ``s_inc`` / ``p_inc`` the pure-s and pure-p
    variants the coupled solves produce for free. The top-level ``R`` / ``T`` delegate to
    ``given`` when it exists and otherwise average the two pure incidences — unpolarized
    illumination.

    All three were spelled ``res`` / ``res_te_inc`` / ``res_tm_inc`` until 2026-08-20:
    ``res`` was a result inside a Result and also the universal local name for the whole
    object, and the ``te``/``tm`` pair contradicted the prose beside it, which has always
    called these the pure-s and pure-p variants. ``.s_inc``/``.p_inc`` rather than ``.s``/
    ``.p`` because ``R_s`` and ``r_s`` on the sub-result are about the *outgoing*
    polarization; the suffix says this one names the incidence.
    """

    def __init__(self, given=None, s_inc=None, p_inc=None, order_window=None):
        self.given = given
        self.s_inc = s_inc
        self.p_inc = p_inc
        #: The orders this result is on: ``((centre_1, half_1), (centre_2, half_2))``, the same
        #: shape :attr:`wavebit._core.engine.CoreSolveMixin.order_window` has. Static metadata
        #: for :meth:`order`; not a pytree child.
        #:
        #: The **retained** window rather than the declared ``truncation_order``, and the two
        #: part company as soon as an ``excitation`` widens one: :meth:`order` reads the array,
        #: so it has to describe the array. A result carrying the declaration instead indexed
        #: one order out and returned a plausible number.
        self.order_window = (None if order_window is None else
                             tuple((int(c), int(h)) for c, h in order_window))

    # pytree protocol (see ResultSub)
    def tree_flatten(self):
        return (self.given, self.s_inc, self.p_inc), self.order_window

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, order_window=aux_data)

    def _delegate(self, name):
        if self.given is not None:
            return getattr(self.given, name)
        te = getattr(self.s_inc, name) if self.s_inc is not None else None
        tm = getattr(self.p_inc, name) if self.p_inc is not None else None
        if te is not None and tm is not None:
            return 0.5 * (te + tm)
        return te if te is not None else tm

    @property
    def R(self):
        """Reflection efficiency per order (unpolarized average if no ``psi`` was given)."""
        return self._delegate('R')

    @property
    def T(self):
        """Transmission efficiency per order (unpolarized average if no ``psi`` was given)."""
        return self._delegate('T')

    def order(self, m, n=0):
        """Scalars for diffraction order ``(m, n)`` of the solved (``given``) incidence.

        Spares the caller the array layout: efficiencies and amplitudes are stored as
        ``(n_orders_2, n_orders_1)`` grids — note ``y`` first, the transpose of the
        ``truncation_order=[x, y]`` input order — centred on the specular order, or on
        :attr:`centre` when an ``excitation`` moved the retained window. This indexes them by
        the order indices themselves, which is the whole point: the array layout is not the
        caller's problem, and it stopped being derivable from the declared ``truncation_order`` the
        moment a window could widen.
        """
        sub = self.given if self.given is not None else (
            self.s_inc if self.s_inc is not None else self.p_inc)
        if sub is None:
            raise ValueError('no solved incidence to index into')
        if self.order_window is None:
            raise ValueError(
                'this result carries no order_window metadata; index the arrays directly')
        (cx, fx), (cy, fy) = self.order_window

        def pick(a):
            if a is None:
                return None
            flat = a.reshape(-1, a.shape[-1])
            row = flat[n - cy + fy if flat.shape[0] > 1 else 0]
            return row[m - cx + fx]

        if not (cx - fx <= m <= cx + fx):
            raise IndexError(f'order m={m} outside the retained set [{cx - fx}, {cx + fx}]')
        ny = None
        for a in (sub.R, sub.T):
            if a is not None:
                ny = a.reshape(-1, a.shape[-1]).shape[0]
                break
        if ny is not None and ny > 1 and not (cy - fy <= n <= cy + fy):
            raise IndexError(f'order n={n} outside the retained set [{cy - fy}, {cy + fy}]')
        if ny is not None and ny == 1 and n != 0:
            raise IndexError('this solve retained a single row of orders; only n=0 exists')

        return OrderView(m, n, pick(sub.R), pick(sub.T),
                         pick(sub.r_s), pick(sub.r_p), pick(sub.t_s), pick(sub.t_p))
