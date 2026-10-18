"""What fills a cell: real diagonal eps and mu, a conductivity, and dispersive poles.

Units, as everywhere in :mod:`wavebit.fdtd._legacy`: ``c = 1``, ``eps0 = mu0 = 1``, lengths in the
user's unit. So a frequency is ``1/wavelength``, an angular frequency is ``2*pi/wavelength``,
a conductivity ``sigma`` is an inverse length (the rate ``sigma/eps`` is one too, and
``exp(-sigma t / (2 eps))`` is the amplitude decay of a uniform lossy medium), and a pole's
``omega_p``, ``omega_0`` and ``gamma`` are angular frequencies in the same unit.

**Three materials, and why there are three rather than one complex number.**

``Material(eps=2.25)``
    A real, lossless, possibly anisotropic medium. ``eps`` is one number or a diagonal
    ``(ex, ey, ez)``; ``mu`` is the same, and ``mu=None`` (the default) means non-magnetic.
``Conductor(eps=2.25, sigma=0.01)``
    The same with an electric conductivity, which is the one loss a time-domain step can
    carry without memory: it folds into the update coefficients (``src/wavebit/fdtd/_prepare/coefficients.py``)
    and costs nothing per step.
``Dispersive(eps_inf=1.0, poles=(...))``
    Loss and dispersion with memory: each pole carries its own polarisation ``P`` through the
    run and feeds ``-dP/dt`` back into the E update (``src/wavebit/fdtd/_kernel/poles.py``).

A single **complex** permittivity is refused (``Material(4 + 0.1j)``). It is right at one
frequency and silently wrong at every other one, and a time-domain run is broadband by
construction -- there is no "the" frequency for it to be right at. The refusal names
:class:`Conductor` (a loss that is flat in ``sigma``) and :class:`Dispersive` (a loss with
the frequency dependence a real material has), which are the two ways to say what was meant.
:func:`eps_at` is how a caller goes the other way, from a pole list to the one complex number
another solver (``wavebit.rcwa``) wants at one wavelength.

**Relation to the shared physical material API.** These are the normalized internal
constitutive declarations retained by the private FDTD engines. New calculations use
``wavebit.materials.Material``, ``ConductiveMaterial`` and ``DispersiveMaterial`` for both
solvers. ``src/wavebit/materials/_lower.py`` converts their SI rates and conductivity to the
scene's normalized units and checks the supported time-domain constitutive scope. RCWA
evaluates the same physical model at a wavelength. The internal types here keep their
real diagonal, conductivity and pole contracts; they are not the shared public model and
must not be interpreted as carrying SI rates.

**Pole types.** One second-order recurrence covers Drude and Lorentz (the ``_PoleSet`` seed
in ``src/wavebit/fdtd/_periodic/legacy.py``); Debye is first order and is written in the same three
coefficients with ``hold = 0`` -- see :class:`Debye`, which derives it, and note that it is
*not* the ``omega_0 = 0`` case of the second-order form (that case is Drude). Debye is also
the one pole whose response is not entirely in the recurrence: a trapezoidal rule on a
first-order ODE is implicit in ``E``, so part of it is folded into the permittivity the E
update divides by (:meth:`Pole.instantaneous`, :meth:`MaterialBase.eps_for_update`). The
explicit alternative is first order and gives the material more loss than it asked for; the
derivation and the measurement are in :class:`Debye`.

In this package's ``e^{-i w t}`` convention a lossy material has ``Im(eps) > 0``
(``docs/reference/common/conventions.md``), and every closed form below is written in it.
"""
from wavebit._common.backend import eager_under_torch_compile as _eager_under_torch_compile
import math

from wavebit._common.declarations import is_jax_typed
from wavebit._common.declarations import spec
from wavebit._common.declarations import torch_carried

__all__ = ['Material', 'Conductor', 'Dispersive', 'Switched', 'Drude', 'Lorentz', 'Debye',
           'MaterialBase', 'Pole', 'eps_at', 'concrete', 'as_material']


def _complex_refusal(where, name, value):
    return ValueError(
        f'{where}({name}={value!r}): a complex permittivity is refused. One complex constant '
        f'is right at one frequency and silently wrong at every other, and a time-domain run '
        f'is broadband by construction. Say fd.Conductor(eps=..., sigma=...) for a loss that '
        f'is flat in sigma, or fd.Dispersive(eps_inf=..., poles=(fd.Drude(...), '
        f'fd.Lorentz(...), fd.Debye(...))) for one with a frequency dependence; '
        f'fd.eps_at(poles, wavelength, eps_inf) turns the latter back into the single complex '
        f'number a frequency-domain solver wants.')


def _is_complex(v):
    if isinstance(v, complex):
        return True
    dtype = getattr(v, 'dtype', None)
    return dtype is not None and 'complex' in str(dtype)


def concrete(v):
    """``float(v)`` when the value can be read now; ``None`` when it is a traced scalar.

    Material values are leaves -- a gradient with respect to a permittivity, a conductivity
    or a pole strength flows through them -- so a JAX tracer or a Torch tensor that asked for
    a gradient must pass every door here. It is also the *only* thing that passes unread:
    every check in ``src/wavebit/fdtd/_prepare/plan.py`` that needs the number (the CFL bound, the cells per
    wavelength, whether a structure matches the PML background) asks this first and says in
    the report when the answer was ``None``.
    """
    if _is_complex(v):
        return None
    if getattr(v, 'requires_grad', False):
        return None
    detach = getattr(v, 'detach', None)
    if detach is not None:
        v = detach()
    try:
        return float(v)
    except Exception:
        return None


def _value(where, name, v):
    """One material number: refused if complex or not a scalar, ``v`` itself otherwise."""
    if _is_complex(v):
        raise _complex_refusal(where, name, v)
    if isinstance(v, (bool, str)) or v is None:
        raise ValueError(f'{where}({name}={v!r}): a real number is needed, not a '
                         f'{type(v).__name__}.')
    shape = getattr(v, 'shape', None)
    if shape is not None and tuple(shape) != ():
        raise ValueError(f'{where}({name}=...): one scalar per component; got shape '
                         f'{tuple(shape)}. A value per cell is not a material: say the '
                         f'geometry with shapes -- fd.Slab, fd.Region, fd.Sphere, fd.Box, '
                         f'fd.Cylinder, fd.Prism -- each with one material; a diagonal is the '
                         f'triple {name}=(x, y, z).')
    if concrete(v) is None and not (is_jax_typed(v) or getattr(v, 'requires_grad', False)):
        raise ValueError(f'{where}({name}={v!r}): a real number is needed (or a traced scalar '
                         f'of jax or torch).')
    return v


def _diagonal(where, name, value):
    """``value`` as a triple of per-axis numbers; a scalar means the same number three times."""
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(f'{where}({name}={value!r}): a diagonal is (x, y, z), three '
                             f'entries; got {len(value)}. Off-diagonal anisotropy is not in '
                             f'stage 1.')
        return tuple(_value(where, f'{name}[{i}]', v) for i, v in enumerate(value))
    v = _value(where, name, value)
    return (v, v, v)


def _positive(where, name, triple, *, strict=True):
    for a, v in zip('xyz', triple):
        c = concrete(v)
        if c is None:
            continue
        if not math.isfinite(c) or (c <= 0 if strict else c < 0):
            what = 'positive' if strict else 'non-negative'
            axis = '' if triple[0] is triple[1] is triple[2] else f' ({a} entry)'
            raise ValueError(f'{where}({name}=...){axis}: {name} is a finite {what} number; '
                             f'got {c!r}.')


@_eager_under_torch_compile
class MaterialBase:
    """Base of the three material types; carries nothing, exists so ``isinstance`` has a name.

    Every subclass exposes the same four readings, which is the whole interface the lowering
    uses: ``eps`` and ``mu_value`` as ``(x, y, z)`` triples, ``sigma`` as a triple, and
    ``poles`` as a tuple. A subclass that has none of a thing returns the value that thing
    stands for -- zeros for ``sigma``, ones for ``mu_value``, ``()`` for ``poles`` -- so no
    reader branches on the type.
    """

    __slots__ = ()

    @property
    def mu_value(self):
        """``mu`` as numbers: what was named, or ``(1, 1, 1)`` when nothing was."""
        return (1.0, 1.0, 1.0) if self.mu is None else self.mu

    @property
    def sigma(self):
        return (0.0, 0.0, 0.0)

    @property
    def poles(self):
        return ()

    @property
    def is_magnetic(self):
        """Whether a permeability was *named*; a structural question, answerable under a trace."""
        return self.mu is not None

    def eps_for_update(self, dt):
        """``eps`` as the E update must see it: ``eps_inf`` plus every pole's *instantaneous* part.

        Most poles have none and this is ``self.eps`` unchanged, object for object, which is
        what keeps a vacuum or a Drude/Lorentz scene's coefficients bit-identical. A
        :class:`Debye` pole has one, because its recurrence is written for the auxiliary
        ``P~ = P - c E`` rather than for ``P`` itself -- see :class:`Debye`, which derives both
        ``c`` and why the alternative is a first-order scheme. Moving ``c dE/dt`` to the left
        of Ampere's law is exactly adding ``c`` to the permittivity the update divides by.

        ``dt`` is in it because ``c`` is: it is a property of the discretisation and not of the
        material, and it vanishes as ``dt -> 0``. Nothing that reports physics reads this --
        the CFL bound, the cells per wavelength and :func:`eps_at` all read ``eps``/``eps_inf``.
        """
        shift = None
        for pole in self.poles:
            part = pole.instantaneous(dt)
            if part is None:
                continue
            shift = part if shift is None else shift + part
        if shift is None:
            return self.eps
        return tuple(e + shift for e in self.eps)

    def index_bound(self):
        """``min(sqrt(eps*mu))`` over the diagonal, or ``None`` if any entry is traced.

        The smallest index, i.e. the fastest phase velocity, that one component supports on its
        own. The CFL bound pairs values across components and phases as well
        (``_lower.checks.cfl_bound``), because E and H sit at different points.
        For a :class:`Dispersive` material it is built from ``eps_inf``: the poles only add
        permittivity at and below their resonances, so ``eps_inf`` is the smallest the
        permittivity gets and therefore the fastest the medium is.
        """
        out = []
        for e, m in zip(self.eps, self.mu_value):
            ce, cm = concrete(e), concrete(m)
            if ce is None or cm is None:
                return None
            out.append(math.sqrt(ce * cm))
        return min(out)


@spec
class Material(MaterialBase):
    """A real, lossless medium: ``eps`` and optionally ``mu``, each scalar or ``(x, y, z)``.

    ``Material(eps=2.25)`` is glass; ``Material(eps=(2.0, 2.0, 2.4))`` is a uniaxial crystal
    with its axis along z. ``mu=None`` (the default) is how "non-magnetic" is spelled -- the
    same structural absence ``wavebit.Material`` uses, and for the same reason: "was a
    permeability named?" is answerable inside a trace where "is this permeability one?" is
    not. A run whose every material leaves ``mu`` unnamed keeps the stage-0 scalar ``bh``.
    """

    eps: object = 1.0
    mu: object = None

    def __post_init__(self):
        object.__setattr__(self, 'eps', _diagonal(type(self).__name__, 'eps', self.eps))
        _positive(type(self).__name__, 'eps', self.eps)
        if self.mu is not None:
            object.__setattr__(self, 'mu', _diagonal(type(self).__name__, 'mu', self.mu))
            _positive(type(self).__name__, 'mu', self.mu)


@spec
class Conductor(MaterialBase):
    """A real medium with an electric conductivity: ``eps``, ``sigma >= 0``, optionally ``mu``.

    ``sigma`` is in the scene's units (``c = 1``, ``eps0 = 1``), so it is an inverse length
    and the amplitude of a uniform field in it decays as ``exp(-sigma t / (2 eps))``. It
    enters the E update only, through ``ae`` and ``be``; a *magnetic* conductivity is not a
    material here and exists only inside :class:`~wavebit.fdtd._legacy.Absorber`, where it is the
    matched half of the graded damping.

    ``Conductor`` is the right spelling for a metal only well below its plasma frequency.
    Above it, use :class:`Dispersive` with a :class:`Drude` pole: a flat ``sigma`` gets the
    skin depth right at one frequency and the reflectance wrong everywhere else.
    """

    eps: object = 1.0
    sigma: object = 0.0
    mu: object = None

    def __post_init__(self):
        object.__setattr__(self, 'eps', _diagonal('Conductor', 'eps', self.eps))
        object.__setattr__(self, 'sigma', _diagonal('Conductor', 'sigma', self.sigma))
        _positive('Conductor', 'eps', self.eps)
        _positive('Conductor', 'sigma', self.sigma, strict=False)
        if self.mu is not None:
            object.__setattr__(self, 'mu', _diagonal('Conductor', 'mu', self.mu))
            _positive('Conductor', 'mu', self.mu)


@_eager_under_torch_compile
class Pole:
    """Base of the three pole types; see :class:`Drude`, :class:`Lorentz` and :class:`Debye`.

    Every pole answers three questions and nothing else:

    ``eps_term(omega)``
        its contribution to ``eps(w)``, in the ``e^{-i w t}`` convention where loss is
        ``Im(eps) > 0``. The closed form :func:`eps_at` sums, and the one a Fresnel gate
        compares against.
    ``coefficients(dt)``
        ``(step_gain, hold, drive)`` of the one recurrence the kernel runs,
        ``P^{n+1} = step_gain P^n - hold P^{n-1} + drive E^n``.
    ``instantaneous(dt)``
        the part of this pole's response that follows ``E`` with no delay and so cannot live
        in that recurrence at all -- ``None`` for a pole that has none, which is Drude and
        Lorentz. See :meth:`MaterialBase.eps_for_update`.
    ``rate()``
        what it contributes to the pole-aware CFL bound: ``(resonance, strength)``, the two
        terms of ``rate = max(w0^2) + max(sum strength/eps)``.
    """

    __slots__ = ()

    def instantaneous(self, dt):
        """``None``: this pole's discrete response is entirely in the recurrence.

        A second-order pole's ``P^{n+1}`` depends on ``E^n`` and no later, so nothing of it is
        simultaneous with the field being solved for. ``None`` rather than ``0.0`` because the
        caller must branch on presence and never on a value that may be traced -- the rule
        this repository keeps everywhere a pytree is involved.
        """
        return None


def _pole_scalar(where, name, v, *, strict=True):
    v = _value(where, name, v)
    c = concrete(v)
    if c is not None and (not math.isfinite(c) or (c <= 0 if strict else c < 0)):
        what = 'positive' if strict else 'non-negative'
        raise ValueError(f'{where}({name}={c!r}): {name} is a finite {what} number.')
    return v


@spec
class Drude(Pole):
    """A free-electron pole: ``eps += -omega_p**2 / (w**2 + i gamma w)``.

    The second-order recurrence with ``omega_0 = 0``, ``strength = omega_p**2``. ``gamma``
    is the collision rate and may be zero (a lossless plasma). Both are angular frequencies
    in the scene's units, so ``omega_p = 2*pi/lambda_p``.
    """

    omega_p: object
    gamma: object = 0.0

    def __post_init__(self):
        object.__setattr__(self, 'omega_p', _pole_scalar('Drude', 'omega_p', self.omega_p))
        object.__setattr__(self, 'gamma', _pole_scalar('Drude', 'gamma', self.gamma,
                                                       strict=False))

    def eps_term(self, omega):
        return _drude_response(omega, self.omega_p, self.gamma)

    def coefficients(self, dt):
        return _second_order(dt, 0.0, self.gamma, self.omega_p ** 2)

    def rate(self):
        return 0.0, self.omega_p ** 2


@spec
class Lorentz(Pole):
    """A bound-oscillator pole: ``eps += strength / (omega_0**2 - w**2 - i gamma w)``.

    ``strength`` is ``delta_eps * omega_0**2`` if you think in oscillator strengths: the
    pole adds ``strength / omega_0**2`` to the static permittivity.
    """

    omega_0: object
    gamma: object = 0.0
    strength: object = 1.0

    def __post_init__(self):
        object.__setattr__(self, 'omega_0', _pole_scalar('Lorentz', 'omega_0', self.omega_0))
        object.__setattr__(self, 'gamma', _pole_scalar('Lorentz', 'gamma', self.gamma,
                                                       strict=False))
        object.__setattr__(self, 'strength', _pole_scalar('Lorentz', 'strength', self.strength,
                                                          strict=False))

    def eps_term(self, omega):
        # A concrete inactive oscillator is zero even at its removable resonance.
        # A zero primal with a carried strength must keep d(chi)/d(strength).
        # Check host equality too: float() can underflow an extended-precision value.
        # A captured JAX constant instead avoids staging equality into a Python bool.
        if (not torch_carried(self.strength) and concrete(self.strength) == 0
                and (is_jax_typed(self.strength) or self.strength == 0)):
            # Zero each finite operand separately: no squares/products may overflow,
            # and traced frequency, resonance and damping retain their zero tangents.
            return 0j * omega + 0j * self.omega_0 + 0j * self.gamma + 0j * self.strength
        return _damping_eval(self.gamma, _response_damping_limit(omega),
            lambda g: self.strength / (self.omega_0 ** 2 - omega ** 2 - 1j * g * omega),
            lambda g: (_large_divide(self.strength, g) / (_large_divide(self.omega_0, g) * self.omega_0
                       - _large_divide(omega, g) * omega - 1j * omega)))

    def coefficients(self, dt):
        return _second_order(dt, self.omega_0, self.gamma, self.strength)

    def rate(self):
        return self.omega_0 ** 2, self.strength


@spec
class Debye(Pole):
    """A relaxation pole: ``eps += delta_eps / (1 - i w tau)``.

    **It is first order, and it is not the ``omega_0 = 0`` case of the other two** -- that
    case is :class:`Drude`. The derivation, because the difference is easy to assert and
    wrong: the second-order pole answers ``strength / (w0^2 - w^2 - i gamma w)``, which at
    ``w0 = 0`` is ``strength / (-i w (gamma - i w))``; matching that to
    ``delta_eps / (1 - i w tau) = (delta_eps/tau) / (1/tau - i w)`` with ``gamma = 1/tau``
    leaves ``strength / (-i w)`` where a constant is needed, so the strength would have to
    depend on ``w``. There is no constant-numerator second-order pole equal to a Debye term.

    What *is* shared is the recurrence's shape. Debye's ODE is ``tau dP/dt + P = delta_eps E``,
    and the trapezoidal rule on it -- both sides centred at ``n + 1/2`` -- gives

        P^{n+1} (tau/dt + 1/2) = P^n (tau/dt - 1/2) + delta_eps (E^{n+1} + E^n) / 2

    that is ``P^{n+1} = step_gain P^n + c (E^{n+1} + E^n)`` with

        step_gain = (2 tau - dt) / (2 tau + dt),   c = delta_eps dt / (2 tau + dt)

    **The ``E^{n+1}`` on the right is not optional, and dropping it is what makes a scheme
    first order.** The kernel's recurrence reads the field one step behind and cannot be
    handed ``E^{n+1}``, so writing ``+ 2 c E^n`` there instead (the explicit left rectangle)
    is tempting and wrong: measured on the transfer function against :meth:`eps_term`, it
    converges at first order where Drude and Lorentz converge at second, and the error is
    almost pure *loss* -- ``Im(eps)`` comes out high by ``w dt / 2`` relative to
    ``|1 - i w tau|``, which at 20 cells per wavelength is 2.44x the true absorption of a
    nearly lossless relaxation (``tau = 0.005``) and +14% at ``tau = 0.05``. A material given
    twice the loss it asked for is not a discretisation error a user can see in the shape of
    an answer.

    **The split that keeps the recurrence explicit and second order.** Write ``P = P~ + c E``.
    Substituting, the ``c E^{n+1}`` cancels and ``P~`` obeys a recurrence in ``E^n`` alone:

        P~^{n+1} = step_gain P~^n + c (1 + step_gain) E^n

    so ``hold = 0`` and ``drive = c (1 + step_gain) = 4 delta_eps tau dt / (2 tau + dt)**2``,
    and the kernel still runs one recurrence for all three pole types with Debye never reading
    ``P^{n-1}``. What is left of ``P`` is the instantaneous ``c E``, and ``c dE/dt`` moved to
    the left of Ampere's law is simply ``c`` added to the permittivity -- which is what
    :meth:`instantaneous` reports and :meth:`MaterialBase.eps_for_update` folds in. The two
    together reproduce the trapezoidal transfer function exactly:
    ``c(1+g)/(z-g) + c = c(1+z)/(z-g)``.

    Its contribution to the CFL rate is reported as ``delta_eps / tau**2``, the only
    combination with the units of ``strength``; that is a bound by analogy and not a measured
    one, and a Debye pole on its own is unconditionally stable under the rule above.
    """

    tau: object
    delta_eps: object = 1.0

    def __post_init__(self):
        object.__setattr__(self, 'tau', _pole_scalar('Debye', 'tau', self.tau))
        object.__setattr__(self, 'delta_eps', _pole_scalar('Debye', 'delta_eps',
                                                           self.delta_eps, strict=False))

    def eps_term(self, omega):
        return self.delta_eps / (1.0 - 1j * omega * self.tau)

    def instantaneous(self, dt):
        """``c = delta_eps dt / (2 tau + dt)``: the part of ``P`` that follows ``E`` with no delay."""
        return self.delta_eps * dt / (2.0 * self.tau + dt)

    def coefficients(self, dt):
        den = 2.0 * self.tau + dt
        step_gain = (2.0 * self.tau - dt) / den
        return (step_gain, 0.0 * self.tau,
                self.delta_eps * dt / den * (1.0 + step_gain))

    def rate(self):
        return 0.0, self.delta_eps / self.tau ** 2


def _stage_numerator(value):
    # Keep XLA from folding a representable division into a subnormal constant
    # reciprocal before multiplying a live numerator. Identity, including AD.
    if is_jax_typed(value):
        from jax import lax
        return lax.optimization_barrier(value)
    return value


def _large_divide(numerator, denominator):
    # Rescale by an exact power of two: the reciprocal of denominator/8 is
    # normal for every finite float64 denominator. Stage the numerator first.
    return _stage_numerator(numerator * .125) / (denominator * .125)


def _drude_response(omega, omega_p, gamma):
    # A finite complex quotient can still lose its frequency tangent: AD divides
    # by the denominator again before multiplying by gamma. Rescale before the
    # reciprocal square becomes subnormal, including a finite primal. The exact
    # power-of-two margin keeps both sides of the switch in the normal range.
    import sys
    limit = _response_damping_limit(omega, maximum=.5 * math.sqrt(1. / sys.float_info.min))
    return _damping_eval(gamma, limit,
        lambda g: omega_p ** 2 / (-omega ** 2 - 1j * g * omega),
        lambda g: (_large_divide(omega_p, g) * omega_p
                   / (-_large_divide(omega, g) * omega - 1j * omega)))


def _response_damping_limit(omega, *, maximum=None):
    """Machine arithmetic threshold for gamma*omega; not a physical cutoff."""
    import sys
    maximum = sys.float_info.max if maximum is None else maximum
    c = concrete(omega)
    if c is not None:
        return maximum / max(1., abs(c))
    from wavebit._common.backend import array_namespace
    xp = array_namespace(omega)
    return maximum / xp.maximum(xp.abs(omega), xp.ones_like(omega))


def _damping_eval(gamma, limit, ordinary, rescaled):
    """Preserve ordinary arithmetic; mask both branches before backend selection.

    The alternate expression is algebraically identical. Safe inactive operands
    prevent an overflow/NaN in an unselected branch from contaminating gradients.
    """
    g, bound = concrete(gamma), concrete(limit)
    if g is not None and bound is not None:
        return rescaled(gamma) if abs(g) > bound else ordinary(gamma)
    from wavebit._common.backend import array_namespace, as_array
    arrays = [v for v in (gamma, limit) if hasattr(v, 'dtype')]
    xp = array_namespace(*arrays)
    g = as_array(xp, gamma, dtype=xp.float64, device=getattr(arrays[0], 'device', None))
    large = xp.abs(g) > limit
    # Keep the inactive optical denominator damped at a Lorentz resonance.
    a = ordinary(xp.where(large, 1., g))
    b = rescaled(xp.where(large, g, 1.))
    if isinstance(a, tuple):
        return tuple(xp.where(large, y, x) for x, y in zip(a, b))
    return xp.where(large, b, a)


def _second_order(dt, omega0, gamma, strength):
    """``(step_gain, hold, drive)`` of ``P'' + gamma P' + omega0^2 P = strength E``.

    Central differences on both derivatives, exactly as ``src/wavebit/fdtd/_periodic/legacy.py::_PoleSet``
    spells it (the kernel seed): ``P^{n+1}`` reads ``E^n`` and never ``E^{n+1}``, so nothing
    has to be solved simultaneously.
    """
    import sys
    def ordinary(g):
        scale = 1.0 / dt ** 2 + 0.5 * g / dt
        return ((2.0 / dt ** 2 - omega0 ** 2) / scale,
                (1.0 / dt ** 2 - 0.5 * g / dt) / scale, strength / scale)
    def rescaled(g):
        # Multiply numerator and denominator by dt before gamma/dt overflows.
        scale = .25 / dt + .125 * g
        return ((.5 / dt - .25 * omega0 ** 2 * dt) / scale,
                (.25 / dt - .125 * g) / scale, _stage_numerator(strength * (.25 * dt)) / scale)
    result = _damping_eval(gamma, sys.float_info.max * min(dt, 1.) / 4., ordinary, rescaled)
    # CPU JAX flushes subnormal ADE drives to zero. Do not silently remove an
    # active oscillator on one backend. Compare scaled operands before dividing.
    tiny = sys.float_info.min
    g, s = concrete(gamma), concrete(strength)
    if g is not None and s is not None and all(concrete(v) is not None for v in result):
        valid = (s == 0 or abs(s * dt) >= tiny * abs(1.0 / dt + .5 * g))
        valid = valid and all(math.isfinite(concrete(v)) for v in result)
        return result if valid else (float('nan'),) * 3
    from wavebit._common.backend import array_namespace, as_array
    arrays = [v for v in (*result, gamma, strength) if hasattr(v, 'dtype')]
    xp = array_namespace(*arrays)
    device = getattr(arrays[0], 'device', None)
    st = as_array(xp, strength, dtype=xp.float64, device=device)
    ga = as_array(xp, gamma, dtype=xp.float64, device=device)
    zero = st == 0
    if is_jax_typed(st):
        # Arithmetic equality on CPU can flush a nonzero subnormal to zero.
        # Test the declaration's magnitude bits, preserving +/-0 as inactive.
        from jax import lax
        import jax.numpy as jnp
        bits = lax.bitcast_convert_type(st, jnp.uint64) & jnp.uint64(0x7fffffffffffffff)
        # Fold the magnitude's words before testing zero. This preserves the
        # nonzero classification in the observed ARM CPU compiled regression
        # when the zero test is combined with the drive bound.
        folded = (bits >> jnp.uint64(32)) | (bits & jnp.uint64(0xffffffff))
        zero = lax.optimization_barrier(folded) == 0
    valid = zero | (xp.abs(st * dt) >= tiny * xp.abs(1.0 / dt + .5 * ga))
    for v in result:
        valid = valid & xp.isfinite(as_array(xp, v, dtype=xp.float64, device=device))
    return tuple(xp.where(valid, as_array(xp, v, dtype=xp.float64, device=device),
                          as_array(xp, float('nan'), dtype=xp.float64, device=device))
                 for v in result)


@spec
class Dispersive(MaterialBase):
    """A medium with memory: ``eps_inf`` and a tuple of poles, optionally ``mu``.

    ``eps(w) = eps_inf + sum(pole.eps_term(w))`` -- :func:`eps_at` is that sum, written once
    so that nothing has to compare against a second implementation of it. ``eps_inf`` is the
    high-frequency limit and is what the CFL bound and the cells-per-wavelength check read,
    because the poles only ever add permittivity below their resonances.

    ``eps_inf`` may be a diagonal; the poles may not (a per-axis pole set is anisotropic
    dispersion, and stage 1 does not have it). A single pole list drives all three
    components.
    """

    eps_inf: object = 1.0
    poles: tuple = ()
    mu: object = None

    def __post_init__(self):
        object.__setattr__(self, 'eps_inf', _diagonal('Dispersive', 'eps_inf', self.eps_inf))
        _positive('Dispersive', 'eps_inf', self.eps_inf)
        poles = (self.poles,) if isinstance(self.poles, Pole) else tuple(self.poles)
        if not poles:
            raise ValueError('Dispersive(poles=()): a dispersive material with no pole is '
                             'fd.Material(eps=eps_inf); say that instead.')
        for p in poles:
            if not isinstance(p, Pole):
                raise TypeError(f'Dispersive(poles=...): every entry is an fd.Drude, '
                                f'fd.Lorentz or fd.Debye; got {type(p).__name__}.')
        object.__setattr__(self, 'poles', poles)
        if self.mu is not None:
            object.__setattr__(self, 'mu', _diagonal('Dispersive', 'mu', self.mu))
            _positive('Dispersive', 'mu', self.mu)

    @property
    def eps(self):
        """The high-frequency permittivity; what the update coefficients are built from."""
        return self.eps_inf


@spec
class Switched(MaterialBase):
    """A lossless medium whose permittivity jumps once: ``eps`` before ``time``, ``after`` from it.

    A temporal boundary. The electric displacement ``D = eps(t) E`` and the magnetic field are
    continuous across the jump, so a wave keeps its wavenumber and changes its frequency by
    ``n_before / n_after``, splitting into a forward and a backward wave (Morgenthaler, IRE
    Trans. Microwave Theory Tech. 6, 167 (1958); Xiao, Maywar and Agrawal, Opt. Lett. 39, 574
    (2014)). The update is written for ``D``, not as ``eps dE/dt``: the latter is the right
    equation for the wrong field and is off by a factor of two at the jump
    (``src/wavebit/fdtd/_kernel/switched.py``).

    ``time`` is in the scene's time unit (``c = 1``: a length). The coefficients are built from
    ``eps`` (the value before the jump); :class:`~wavebit.fdtd._kernel.switched.SwitchedTerm`
    corrects the update over the structure's cells. ``index_bound`` reads both values, so the
    CFL bound holds on either side of the jump.
    """

    eps: object = 1.0
    after: object = 1.0
    time: object = 0.0

    @property
    def mu(self):
        return None

    def __post_init__(self):
        object.__setattr__(self, 'eps', _diagonal('Switched', 'eps', self.eps))
        object.__setattr__(self, 'after', _diagonal('Switched', 'after', self.after))
        _positive('Switched', 'eps', self.eps)
        _positive('Switched', 'after', self.after)
        if concrete(self.time) is None:
            raise TypeError('Switched(time=...): the switching time decides which step the '
                            'jump falls on, a static choice; it cannot be traced.')
        if not math.isfinite(float(concrete(self.time))):
            raise ValueError(f'Switched(time={self.time!r}): the switching time must be finite.')

    def index_bound(self):
        """The smaller of the two media's bounds: the faster side sets the CFL limit."""
        out = []
        for e in self.eps + self.after:
            ce = concrete(e)
            if ce is None:
                return None
            out.append(math.sqrt(ce))
        return min(out)


@_eager_under_torch_compile
def eps_at(poles, wavelength, eps_inf=1.0):
    """``eps(w)`` of a pole list at ``wavelength``, in the ``e^{-i w t}`` convention.

    The closed form the time stepping discretises, written once. ``poles`` is a tuple of
    :class:`Drude` / :class:`Lorentz` / :class:`Debye`, or a :class:`Dispersive` material, in
    which case ``eps_inf`` comes from it. Loss is ``Im(eps) > 0``; a pole that returns a
    negative imaginary part is gain, and the Fresnel gate's mutant is exactly that sign
    flipped.

    Ported from ``src/wavebit/fdtd/_periodic/legacy.py::permittivity_at``, which takes ``(omega0, gamma,
    strength)`` triples; the pole objects here answer the same sum term by term and add the
    Debye form, which that function has no spelling for.
    """
    if isinstance(poles, Dispersive):
        eps_inf, poles = poles.eps_inf, poles.poles
    if isinstance(poles, Pole):
        poles = (poles,)
    if isinstance(eps_inf, (tuple, list)):
        # A diagonal answers per axis -- except when the three agree, which is every
        # isotropic material, where a caller wants the one number and not a triple of it.
        values = tuple(eps_at(poles, wavelength, e) for e in eps_inf)
        return values[0] if values[0] == values[1] == values[2] else values
    omega = 2.0 * math.pi / wavelength
    eps = complex(eps_inf)
    for p in poles:
        eps = eps + p.eps_term(omega)
    return eps


def as_material(value, where='material'):
    """A material from what a caller wrote: a material, a bare number, or ``None`` for vacuum.

    The single coercion point, so the scalar shorthand (``background=2.25``) is defined once.
    ``None`` is vacuum, ``fd.Material(eps=1.0)`` -- permittivity one and no permeability
    named, so nothing magnetic is built for a scene that never mentions one.
    """
    if value is None:
        return Material(eps=1.0)
    if isinstance(value, MaterialBase):
        return value
    if isinstance(value, (tuple, list)) or _is_complex(value) or concrete(value) is not None \
            or is_jax_typed(value) or getattr(value, 'requires_grad', False):
        return Material(eps=value)
    raise TypeError(f'{where}={value!r}: a material is fd.Material, fd.Conductor, '
                    f'fd.Dispersive, a bare permittivity, or None for vacuum; got '
                    f'{type(value).__name__}.')
