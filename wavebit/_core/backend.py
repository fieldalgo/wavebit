"""Backend adapter — the only code that dispatches on the array framework.

The single-source core (migration Phase 3+) calls these; the current per-backend solvers do
not yet. Keeping the surface tiny is the point: everything divergent between NumPy / JAX /
PyTorch lives here, so it can be tested in one place.

``eig`` is the one primitive that genuinely differs, and the one whose *gradient* this module
supplies rather than delegates: both frameworks can differentiate a non-symmetric complex
``eig`` (JAX via ``enable_eigvec_derivs``, PyTorch natively) but both assume distinct
eigenvalues and return NaN when that fails, which an ordinary structure can cause. See the
note above :func:`eig`. NumPy needs no gradient.
"""
import array_api_compat as ac
import numpy as _np


def array_namespace(*xs):
    """Return the Array-API namespace for the given array(s) (numpy / jax.numpy / torch)."""
    return ac.array_namespace(*xs)


def carries_autodiff(value):
    """Whether selecting equal entries could discard independent input derivatives.

    Every JAX tracer counts, including a plain JIT input: that compiled function can later
    be differentiated with grad(jit(f)). Deciding only when a JVP tracer arrives would make
    the derivative depend on transformation order. Eager arrays need no special treatment.
    """
    if getattr(value, 'requires_grad', False):
        return True
    import sys
    jax = sys.modules.get('jax')
    return jax is not None and isinstance(value, jax.core.Tracer)


def _readable(v, xp):
    """``v`` as something ``xp`` can reduce, without disturbing an autograd graph.

    Deliberately not :func:`as_array`, and deliberately not a bare ``xp.asarray``: through
    torch 2.11.x the latter handed back *the same tensor* with ``requires_grad`` flipped off
    in place (see :func:`as_array` for the version history), which made a read-only guard a
    way to silently sever a caller's gradient. torch 2.12.0 fixed that, so on the supported
    floor the pass-through is no longer what keeps the graph alive; it is kept because a
    guard that only reads has no business rebuilding its input. An array already belonging to
    ``xp`` is therefore passed through untouched; a Python scalar or a foreign array —
    ``n_top`` can arrive as either — has no graph to lose and is converted, which is also what
    gives ``xp.isdtype`` below a dtype it recognizes.
    """
    if hasattr(v, 'dtype') and hasattr(v, 'shape'):
        try:
            if ac.array_namespace(v) is xp:
                return v
        except Exception:
            pass
    return xp.asarray(v)


def checked_float(v):
    """``v`` as a plain float **for a check**, raising only when there is no value to read.

    The numeric twin of :func:`_concrete`, and it draws the same line in the same place: a
    ``jax.grad`` tracer carries a concrete primal and is on the **checked** side, a ``jax.jit``
    tracer has no number and is skipped. `_concrete` gets there by reducing to a boolean, which
    narrows on its own; a geometry check needs the number itself, and the spelling decides
    whether it can have it. Measured:

        read                      eager   jax.grad   jax.jit   torch(requires_grad)
        float(x)                  ok      **fails**  fails     ok, with a UserWarning
        float(stop_gradient(x))   ok      ok         fails     TypeError
        float(x.detach())         ok      --         --        ok, silent

    So no single spelling covers both live backends, and this is the sequence that does:
    ``detach`` where it exists, then ``float``, then ``stop_gradient`` for what is left.

    **Why it matters that this is not `float`.** :func:`wavebit._core.overlap.primitive` spelled
    it ``float(cx)``, and :func:`wavebit._core.parametric._primitives` reads the failure as "this
    is traced -- skip the check". A grad tracer refuses ``float()`` while answering
    ``stop_gradient`` and ``bool(x > 300)`` perfectly, so it was saying "this spelling carries a
    tangent", not "there is no value here" -- and the geometry refusals were skipped under
    exactly the gradient loop they exist to protect. A rod grown by plain gradient ascent from
    ``lx = 280`` on a period of 300 ran to completion at **lx = 911.71**, three times the
    period, with ``R + T = 1.000000000000`` at every step; the same run on torch died at step 1,
    because torch tensors satisfy ``float()``.

    ``detach`` first also silences the ``UserWarning`` torch emits for a ``requires_grad``
    scalar conversion, which the checks were printing on every eager torch shape solve.

    Deliberately **not** :func:`wavebit._core.parametric._static_float`, which refuses an autograd
    value on purpose: that one keys a cache, where reading a differentiated value would serve a
    stale build. Reading one to *check* geometry is the opposite -- it is the case that most
    needs checking.
    """
    detach = getattr(v, 'detach', None)
    if detach is not None:                      # torch, including requires_grad, and silent
        v = detach()
    try:
        return float(v)
    except Exception:                           # noqa: BLE001 - a tracer of some kind
        pass
    if type(v).__module__.split('.')[0] == 'jax':
        import jax                              # local: jax is optional
        return float(jax.lax.stop_gradient(v))  # a jit tracer still raises, as it must
    raise TypeError(f'{type(v).__name__} carries no value a check can read')


def _concrete(flag):
    """``bool(flag)`` for a 0-d reduction, or ``None`` when the value cannot be read.

    ``None`` means the value is *abstract* — a ``jax.jit`` tracer, a PyTorch ``FakeTensor``
    under ``compile`` — where there is genuinely no number to look at. That, and not the
    namespace, is what the guards below are keyed on. Keying on the namespace was the bug:
    an ordinary eager ``backend=1`` / ``backend=2`` call has its values right there, and was
    skipped anyway.

    ``jax.grad`` sits on the *checked* side of the line, which is the whole point. An
    un-jitted grad tracer carries a concrete primal, so the reduction narrows to a real
    boolean and the guard runs — and an un-jitted gradient loop is precisely where a
    silently wrong solve does the most damage. ``jax.jit``, and ``jit(grad)``, refuse the
    read and are skipped. Measured rather than assumed; the same asymmetry is tabulated in
    ``docs/backend-choice.md`` §3 and already relied on by
    :func:`wavebit._core.solve_2d_4ff._forward_count` and
    :func:`wavebit._core.conventions._check_resolvable`.

    The reduction itself runs in the caller's own namespace, so what crosses the boundary is
    one boolean rather than a copy of the array — on an accelerator that is a single sync,
    not a device-to-host transfer.

    Catching broadly rather than naming the exception keeps this backend-agnostic, the same
    trade :func:`wavebit._core.engine._concrete_period` makes.
    """
    try:
        return bool(flag)
    except Exception:
        return None


def check_finite(xp, **named):
    """Raise if any named result carries a NaN or an infinity.

    The solve reaches the edges of double precision on purpose — a strongly evanescent mode's
    ``exp(-k0*q*d)`` lands in the subnormal range — so a result that came back non-finite is
    worth stopping on rather than returning. This is the check that says so.

    Runs on every backend whose values are concrete, which includes eager JAX and eager
    PyTorch and an un-jitted ``jax.grad``; see :func:`_concrete` for where the line falls and
    why it is drawn at concreteness rather than at the namespace. Under ``jax.jit`` nothing is
    readable and the check is skipped — a traced solve that goes non-finite still returns
    quietly, which is a real remaining gap and not one this formulation can close.

    **The remedy in the message is measured, and the two suggestions it used to make are
    not remedies**, which is why they are gone. On the case this fires for — an order landing
    exactly on a light line, ``kx_m == n`` — the degeneracy is fixed by ``kx`` against an
    index: neither the truncation nor the layer thickness touches it, so changing ``truncation_order`` or
    re-slicing only re-rolls the round-off. Measured before the conical path was reformulated
    (see :func:`~wavebit._core.solve_1d.layer_modes_1d_conical`), those re-rolls returned
    quiet wrong answers scattered from 0.18 to 626 against a true 0.215, re-slicing among
    them (1/2/4/8/20 slices: 11.9, 3.6, 1.8, 27.7, 110.2); on the 2D path they still do
    (see :func:`~wavebit._core.solve_2d.layer_modes_2d`). Detuning moves ``kx`` itself, which
    is the only thing that does anything, and one ULP of wavelength, period or angle is
    already enough — measured to agree with the exact-point answer of the reformulated paths
    to eight digits.
    """
    bad = []
    for name, v in named.items():
        ok = _concrete(xp.all(xp.isfinite(_readable(v, xp))))
        if ok is None:
            return                              # traced — no value to inspect
        if not ok:
            bad.append(name)
    if bad:
        raise FloatingPointError(
            f"non-finite value in {', '.join(sorted(bad))} — the solve did not produce a "
            'physical result. The usual cause is a degenerate structure: an order landing '
            'exactly on a light line (kx_m == n for some medium, so its kz is 0), a layer '
            'identical to its neighbour that was not detected as such, or a zero or '
            'near-zero thickness.\n'
            'If it is a grazing order, move off the condition — one ULP of wavelength, '
            'period or angle is enough, and the answer either side of it agrees to eight '
            'digits; solve(auto_detune=True), the default, does that retry for you, so '
            'seeing this means it was disabled or the retry failed too. Do NOT reach for '
            'truncation_order or thinner slices instead: neither knob touches the degeneracy (it is '
            'fixed by kx against an index), so at best they re-roll the round-off — '
            'measured to return quiet wrong answers on the paths that have not been '
            'reformulated against this. See layer_modes_1d_conical.')


class UnphysicalResultWarning(UserWarning):
    """A solve returned finite but physically impossible efficiencies.

    Its own category so it can be escalated (``warnings.simplefilter('error',
    UnphysicalResultWarning)`` gives a traceback with the values still in scope) or silenced
    independently of everything else the library might warn about.
    """


class UnresolvedBranchWarning(UserWarning):
    """A ``sqrt`` branch was chosen from a value that carries no information about it.

    Selecting the physical root asks whether ``Re(v)`` is zero, and that is answerable only
    while ``Re(v)`` stands clear of the working precision's round-off. At complex64 it often
    does not: round-off is ~1e-7, and both a lossless layer at a complex incidence angle and a
    weakly absorbing superstrate put real parts at exactly that scale — one of them noise, the
    other signal, and nothing in the value distinguishes them. The branch is then taken from
    the sign of round-off, which is why a single-precision solve can return ``|R| ~ 4`` while
    the same solve at double precision is exact to 1e-14.

    Its own category so it can be escalated or silenced on its own; see ``docs/branch-cuts.md``.
    """


class PrecisionLossWarning(UserWarning):
    """A user-supplied value arrived narrower than the precision the solve was asked for.

    Widening it is not the same as having had it: the digits are already gone. Its own
    category so it can be silenced by a caller who is passing single-precision data on purpose.
    """


class RayleighPointWarning(UserWarning):
    """A layer eigenmode sits numerically on its light line, on a path that loses it there.

    Fired by :func:`check_mode_cutoff` from the 2D mode builder — since the coupled-cluster
    rebuild (see :func:`wavebit._core.solve_2d._rebuild_grazing_cluster`) only on its
    fallback paths: a grazing cluster whose E- and H-side sizes disagree, or eigenvalues
    that are concrete but not materializable (a tensor carrying grad, an un-jitted
    ``jax.grad`` tracer). There the old product construction is what ships, and the result
    it annotates is finite, in range, and possibly wrong with no other sign — see the
    caller for the mechanism and :func:`check_finite`'s message for the remedy (a one-ULP
    detune, which ``solve``'s default ladder applies on seeing this warning). Its own
    category so a sweep that knowingly steps across a Rayleigh anomaly can silence or
    escalate it on its own.
    """


class AxisNameWarning(UserWarning):
    """An ``x``/``y`` pair was read as ``u``/``v`` on a cell where they are not the same.

    The order axes run along the cell's reciprocal vectors: ``u`` along ``a1``, ``v`` along
    ``a2``. On a rectangular cell those *are* x and y, and on a skewed one they are neither --
    measured, a 60-degree hex sends ``m += 1`` to ``(dkx, dky) = (+0.600, -0.346)``.

    Read rather than refused, which is the opposite of what ``inverse_rule`` and explicit
    ``sample_x`` coordinates do on the same cells, and the difference is worth stating.
    Those two have **no** correct reading: Li's rule on a skewed cell answers the rectangular
    problem, and an ``(x, y)`` does not determine the ``(u, v)`` a sampler needs. Here there is
    one -- ``truncation_order`` has only ever meant "this many orders along each cell
    direction", so a caller who wrote ``x`` out of rectangular habit meant ``u``, and the solve
    that follows is correct. Refusing would block a run that was going to be right; the warning
    corrects the caller's *vocabulary*, not their answer.

    Its own category so a sweep over skewed cells can silence it without touching
    :class:`RayleighPointWarning` or the rest.
    """


class ComplexIncidenceWarning(UserWarning):
    """The incident wave's transverse wavevector is complex, so R and T are formal ratios.

    Only a complex ``theta`` / ``phi`` reaches one -- a direct ``kx`` / ``ky`` must be real --
    through the formal ``kx/k0 = n*sin(theta)*cos(phi)`` parameterization. Such a wave's
    intensity varies along the surface, and in the exit medium no root of ``kz`` both carries
    power away and decays away, so which one a solver takes is a convention: measured on one
    interface (top 4+3i, bot 2.25), fmmax's ``Im(kz) > 0`` gave ``r = 1.727-1.966i`` where
    this solver and RETICOLO's Maystre cut (``Re(kz) + Im(kz) > 0``) gave ``0.252+0.287i``;
    see ``docs/time-convention-audit.md`` section 9. The per-order ``R`` and ``T``
    are then amplitude-derived ratios rather than fractions of an incident power -- their sum
    exceeds 1 on a lossless stack -- and ``power_entering`` and ``absorption`` are NaN.

    Evanescent incidence (``theta = pi/2 + 1j*alpha``) keeps ``kx`` real and does not fire.
    For an absorbing incident medium, a real ``theta`` is the physical input: the phase angle
    keeps ``kx`` real and puts the loss in ``kz``. Its own category so a study that means the
    formal parameterization can silence it without silencing anything else.
    """


class AutoDetuneWarning(UserWarning):
    """An ``auto_detune`` retry moved the solve off the requested point — this is the record.

    Fired once per ``solve(auto_detune=...)`` that actually retried; the message carries
    the relative step and the wavelength the returned result was evaluated at, and the solver
    keeps that wavelength so everything downstream (field reconstruction included) describes
    one consistent problem. Its own category so a sweep can silence the notices while keeping
    :class:`RayleighPointWarning` escalated, or the reverse. This is the loud replacement for
    the silent ``perturbation`` knob that was removed — what changed is always said.
    """


# Where the 2D mode-cutoff warning fires: below this ratio of min|eigenvalue|/max|eigenvalue|
# the measured efficiency error exceeds ~1e-5, and it keeps growing as ~eps/ratio down to
# order-1 at the exact point. Measured on four exact-graze 2D stacks over a detune ladder
# (2026-08-16): ratio 1e-17 -> err 2e-3..8e0 (some with no other warning), 1e-15 -> ~1e-3,
# 1e-14 -> ~1e-4, 1e-13 -> ~1e-5, >=1e-12 -> <=2e-6. On the same stacks the ratio crosses
# this line ~1000 wavelength-ULPs from the condition, so the flagged band is ~1e-13 wide in
# relative wavelength; 60 random non-grazing 2D solves gave 0 false positives, and the full
# suite runs green with the check in place.
_CUTOFF_REL = 1e-13


def check_mode_cutoff(xp, eigenvalues):
    """Warn when a layer eigenvalue is at round-off scale relative to the spectrum.

    In this regime the 2D product construction ``V = (Omega_R @ W) / q`` no longer
    represents any nearby problem — the grazing cluster's product eigvecs are eps-unstable
    against the factor split, and the answer can be quietly wrong (measured: 5 of 48
    exact-graze 2D solves returned R+T of 0.96-0.99 on a lossless stack with nothing else
    fired). Since the coupled-cluster rebuild
    (:func:`wavebit._core.solve_2d._rebuild_grazing_cluster`) the 2D builder handles the
    regime itself — exactly, or with a loud refusal — and reaches this check only on its
    fallback paths, where the old construction is what ships; the warning is then what
    drives ``solve``'s default detune ladder. The 1D paths do not need this check: the
    pure path never had the disease and the conical path was reformulated (see
    ``layer_modes_1d_conical``).

    A warning rather than an error, deliberately: the error scales as ``~eps/ratio`` (see
    ``_CUTOFF_REL``), so most of the flagged band is wrong at the 1e-5 level rather than
    garbage, and a refusal would take working-precision answers away from a sweep that merely
    steps near the condition. Runs where the eigenvalues are concrete; traced solves skip it,
    like every other guard here.
    """
    aw = xp.abs(_readable(eigenvalues, xp))
    flag = _concrete(xp.min(aw) < _CUTOFF_REL * xp.max(aw))
    if flag:
        import warnings
        warnings.warn(
            'a layer eigenmode sits within round-off of its cutoff (a Rayleigh/grazing '
            'condition) on the 2D path, which loses that mode to cancellation — the result '
            'is finite and may be wrong at the 1e-5 level near the condition, growing to '
            'order 1 exactly on it, with no other sign. Move off the condition: one ULP of '
            'wavelength, period or angle is enough, and solve(auto_detune=True), the '
            'default, does that retry for you. (The 1D paths solve this point exactly; a '
            'y-invariant cell routed through the 1D solves does not need the detune.)',
            RayleighPointWarning, stacklevel=3)


def _real_itemsize(dtype):
    """Bytes per real component, or None for anything that is not a float/complex dtype.

    Has to swallow three spellings: a NumPy scalar type (``np.complex128``), a dtype object,
    and a PyTorch dtype, whose ``str`` is ``'torch.complex64'`` and which ``np.dtype`` refuses.
    """
    if dtype is None:
        return None
    import numpy as np
    try:
        d = np.dtype(dtype)
    except Exception:
        try:
            d = np.dtype(str(dtype).rsplit('.', 1)[-1])
        except Exception:
            return None
    if d.kind == 'f':
        return d.itemsize
    if d.kind == 'c':
        return d.itemsize // 2
    return None


def check_input_precision(name, value, target_dtype):
    """Warn when ``value`` is already narrower than ``target_dtype`` asks for.

    The case this exists for: JAX's 32-bit cap is lifted by :func:`wavebit.enable_x64`, and
    otherwise only when the JAX backend is *first used* -- the setting is process-wide and
    belongs to the caller, so nothing flips it on plain ``import wavebit`` unless JAX was
    already loaded. An array built before that moment is silently single precision::

        import jax.numpy as jnp, wavebit
        theta = jnp.asarray(0.4 + 0.3j)          # complex64 -- x64 is not on yet
        wavebit.rcwa(backend=1, theta=theta, complex_dtype=0, ...)   # asks for double

    ``wavebit.enable_x64()`` before the second line is the fix.

    ``jnp.array(theta, dtype=complex128)`` in the setter then promotes it without recovering
    anything, and the solve runs with ~8 significant digits missing from its incidence angle.
    That produced a ``|R| = 3.94`` against a true 0.67 while every dtype in sight said
    complex128 — and it was read as a solver bug before it was read as this.

    Not JAX-specific: any array handed in below the working precision loses the same way.
    Silent widening is the failure mode, so this is the one place that says it out loud.
    """
    target = _real_itemsize(target_dtype)
    have = _real_itemsize(getattr(value, 'dtype', None))
    if target is None or have is None or have >= target:
        return
    import warnings
    warnings.warn(
        f'{name} was given at {8 * have}-bit precision but this solve runs at '
        f'{8 * target}-bit, and widening it cannot recover the digits already lost. If this '
        'is JAX, the usual cause is building the array before JAX\'s 32-bit cap was lifted: '
        'call `wavebit.enable_x64()` once, before building it. `import wavebit` lifts the cap '
        'only if JAX was already loaded, and the backend lifts it at the first solve, which '
        'is too late for an array made before that.',
        PrecisionLossWarning, stacklevel=3)


def check_physical(xp, de_ri, de_ti, n_top, atol=1e-3, *, power_entering=None):
    """Warn if efficiencies are finite but cannot describe a physical result.

    :func:`check_finite` catches contamination that announces itself. This catches the tier
    below it — a solve that stayed finite and is still wrong, which nothing else would notice.
    Two signals, both properties of the result rather than of the arithmetic that produced it:

    * a negative efficiency, which is negative power;
    * transmitted power greater than the net power entering a passive stack.

    ``power_entering`` includes incident/reflected interference, so this check
    also applies to an absorbing incident medium with real transverse k.
    NaN marks an unsupported cell-power integral and skips the check. When
    entering power is not supplied, only a real-index incident medium permits
    the fallback ``power_entering = 1 - sum(R)``.

    Known false positive: a gain medium (``Im(n) < 0`` in this codebase's convention) legitimately
    exceeds 1. Filter :class:`UnphysicalResultWarning` if that is what you are modelling.

    The message carries no numbers, so Python's default filter collapses a sweep's worth of
    occurrences into one report instead of re-emitting per wavelength; escalate the category to
    an error when you want to catch a specific one.

    Runs wherever the values are concrete — eager NumPy, eager JAX, eager PyTorch, and an
    un-jitted ``jax.grad``. See :func:`_concrete`.
    """
    if power_entering is None:
        n = _readable(n_top, xp)
        if xp.isdtype(n.dtype, 'complex floating'):
            lossy = _concrete(xp.any(xp.imag(n) != 0))
            if lossy is None or lossy:
                return
    else:
        power_entering = _readable(power_entering, xp)
        if _concrete(xp.all(xp.isfinite(power_entering))) is not True:
            return

    r, t = xp.real(_readable(de_ri, xp)), xp.real(_readable(de_ti, xp))
    negative = _concrete(xp.minimum(xp.min(r), xp.min(t)) < -atol)
    excess = _concrete(xp.sum(r) + xp.sum(t) > 1 + atol if power_entering is None
                       else xp.sum(t) > power_entering + atol)
    if negative is None or excess is None:
        return                                  # traced — no value to inspect

    problems = []
    if negative:
        problems.append('a negative efficiency (negative power)')
    if excess:
        problems.append('R+T above 1 for a passive stack (gain out of nothing)'
                        if power_entering is None else
                        'transmitted power above net entering power for a passive stack')
    if problems:
        import warnings
        warnings.warn(
            f"solve returned {' and '.join(problems)}. The result is finite but not physical, "
            'so treat it as a failed solve: check for a degenerate structure, and try adjusting '
            'truncation_order or re-slicing the stack into thinner layers.',
            UnphysicalResultWarning, stacklevel=2)


def forward_strided(v):
    """``v`` with no negative stride, copying only a NumPy view that has one.

    ``np.rot90``, ``np.flip`` and ``a[::-1]`` all return views with a negative stride, and
    PyTorch refuses those outright: ``ValueError: At least one stride in the given numpy array
    is negative``. NumPy and JAX take them without complaint, so the natural spelling of "solve
    the rotated cell" is a crash on ``backend=2`` and a result on the other two — a backend
    difference in what the *input contract* accepts, which is exactly what this module exists
    to absorb.

    Normalising is safe precisely where it applies. Only NumPy arrays are copied, and a NumPy
    array never carries an autograd graph; a torch tensor cannot have a negative stride in the
    first place, and NumPy-copying one that tracks gradient is not even possible
    (``np.ascontiguousarray`` raises ``RuntimeError: Can't call numpy() on Tensor that requires
    grad``). Anything else is returned untouched.
    """
    if isinstance(v, _np.ndarray) and any(s < 0 for s in v.strides):
        return _np.ascontiguousarray(v)
    return v


def as_array(xp, v, dtype=None, device=None, copy=False):
    """``v`` as an ``xp`` array, keeping any autograd graph intact.

    Values that are already arrays of this namespace are cast with ``astype``, which is
    differentiable; everything else (Python scalars, NumPy arrays, lists) goes through
    ``asarray``, where there is no graph to lose.

    Use this anywhere a *user-supplied* quantity enters the algebra — an index, an angle, a
    period, a wavelength, an edge position. `xp.asarray` is still right for values the core
    builds itself, such as index arrays and constants.

    A foreign array crossing into ``xp`` also gets its strides normalised
    (:func:`forward_strided`), because PyTorch is the one namespace that refuses a reversed
    NumPy view. That half is current: on torch 2.13.0 ``asarray(np.flip(a))`` still raises
    ``ValueError: At least one stride in the given numpy array is negative``.

    The *gradient* hazard this function was written for is fixed upstream and is no longer
    reachable from the supported floor. Through **torch 2.11.x**, ``torch.asarray`` defaulted
    ``requires_grad`` to ``False`` and then applied it to the tensor it was about to return
    (``torch/csrc/utils/tensor_new.cpp``: ``detach()`` for a non-leaf, ``set_requires_grad()``
    for a leaf). So a leaf whose dtype already matched came back as *the same tensor* with
    ``requires_grad`` flipped off in place — a caller's optimizer parameter stopped carrying
    gradient and the very next ``backward()`` raised — while every other path returned a
    detached copy. Reproduced here on torch 2.0.0, and that branch is still verbatim in the
    v2.11.0 source. pytorch#170897 (landed 2026-03-31), first released in **torch 2.12.0**
    (2026-05-11), changed the default to ``obj.requires_grad`` and warns once about the old
    one; measured on torch 2.13.0 — this project's floor, see ``docs/torch-compile.md`` —
    every ``asarray`` path now preserves both ``requires_grad`` and the graph, so on gradients
    the two spellings agree. ``array_api_compat`` never entered into it: its torch ``asarray``
    is a passthrough that only fixes device propagation (changelog 1.12.0, pytorch#150199).

    So the ``astype`` branch is no longer the graph guard it was. What justifies it now is
    ``device``, described below: the branch *moves* a value already in ``xp``, where a rebuild
    through ``asarray`` would have to be told to. The strides normalisation is not a reason to
    keep this branch and never was -- ``forward_strided`` runs past the early return, only for
    a foreign array -- it is why ``as_array`` beats ``xp.asarray`` on that path instead.

    ``device`` is honoured on **both** paths. It used to be honoured only on the second one:
    a value already in ``xp`` was cast and returned wherever it happened to live, so a
    ``device='cuda'`` solver handed a CPU tensor kept the CPU tensor. Nothing on a CPU-only
    machine can see that -- there is one device and every placement is correct -- which is why
    it survived five call sites. ``to_device`` is the array-API spelling for a move and is
    identity when the device already matches, so the ordinary call still copies nothing; under
    Torch it is ``.to()``, which is differentiable, so a caller's optimizer leaf survives the
    move.

    ``copy=True`` says the result may not share memory with a NumPy buffer the caller still
    holds. That is what a *stored* attribute needs -- a solver's ``period`` must not change
    because the caller reused the array it was built from -- and NumPy input is the whole of
    it: a list or a scalar is never shared, and a native array is shared **on purpose**,
    because that is what lets a Torch optimizer's in-place step reach the solver it was handed
    to. One sentence, the same on all three backends: a NumPy array you pass in is copied, a
    native array you pass in is kept.

    Not the default, because the callers that pass a wavelength or a wavevector *into* the
    algebra share it harmlessly -- the next operation allocates anyway.
    """
    if copy and isinstance(v, _np.ndarray):
        v = v.copy()                          # a NumPy array carries no graph to lose
    if hasattr(v, 'dtype') and hasattr(v, 'shape'):
        try:
            same_ns = ac.array_namespace(v) is xp
        except Exception:
            same_ns = False
        if same_ns:
            out = v if dtype is None or v.dtype == dtype else xp.astype(v, dtype)
            return out if device is None else ac.to_device(out, device)
    v = forward_strided(v)
    if dtype is None:
        return xp.asarray(v, device=device)
    return xp.asarray(v, dtype=dtype, device=device)


# --- eig, and the one place eigenvalue degeneracy is handled ---------------------------
#
# The eigenvector derivative carries a ``1 / (w_i - w_j)``, so it is undefined where two
# eigenvalues coincide. That is not an exotic case here: a *uniform* layer at normal
# incidence has ``q(+m) == q(-m)`` exactly, for every retained order. Both frameworks'
# native rules return NaN there — the whole gradient, not just the degenerate directions —
# and a uniform buffer or spacer layer at normal incidence is an ordinary thing to put in a
# stack, so an inverse-design run walks into it without doing anything unusual.
#
# Lorentzian broadening, ``1/d -> conj(d)/(|d|**2 + eps)``, keeps that reciprocal
# finite. A repeated eigenspace has gauge freedom, but setting its reciprocal to zero
# alone is not a general derivative of a matrix function at degeneracy. The uniform
# optical controls below do agree with their approach limits; that is a tested case,
# not a universal guarantee for arbitrary repeated or defective eigenvalues.
#
# `_eig_grad` below is a PORT of `_eig_bwd` from fmmax (MIT, Copyright (c) 2023 Meta
# Research, facebookresearch/fmmax, in its `utils.py`; the maintained fork invrs-io/fmmax
# carries it in `eig.py`). The broadening and every term of `rhs` are theirs; our
# changes generalize it to the array API and PyTorch's conjugate convention, and
# replace its two epsilon constants with a dtype/dimension-relative guard. See LICENSE.fmmax — the
# MIT terms require that notice to travel with this code.
#
# The former relative denominator 1e-12 suppressed distinct pairs separated by about
# 1e-6 of the spectral range. A painted-layer derivative read -0.71919175 against
# the finite-difference limit -0.71978335. Normalize the gaps and use (n*machine_eps)^2
# instead; the same response reads -0.71978335. A direct A@A derivative independently
# checks nearby distinct modes over sixteen decades of operator scale.
#
# The gauge term (the third one in `rhs`) is theirs too; grcwa carries it, meent and
# torcwa drop it. It is kept here on their authority rather than on evidence: for a
# gauge-invariant loss — which every physical loss is — it provably contributes nothing, and
# no measurement made here separates it from its absence. It costs one matmul.
#
# Equation 4.77 of Boeddeker et al., arXiv:1701.00392. Cross-checked against both
# frameworks' native rules where those are defined (see tests/test_eig_grad).

def _eig_grad(xp, w, V, gw, gV):
    """Gradient of ``A`` given the eigendecomposition's, in PyTorch's complex convention.

    ``gw`` / ``gV`` are cotangents with respect to the *conjugates* of the outputs, which is
    what PyTorch hands a custom Function and what it wants back. JAX's convention is the
    other one, so :func:`_jax_eig` conjugates on the way in and out; that asymmetry is the
    only framework-specific part of this, and taking it on faith is how the PyTorch rule
    first came out wrong by a term (native and finite differences both said so).
    """
    n = w.shape[-1]
    I = xp.eye(n, device=_device_of(w))
    eye = xp.astype(I, xp.bool)

    delta = w[..., None, :] - w[..., :, None]
    # Normalize before squaring: the guard is on a relative eigenvalue gap,
    # independent of operator units and without underflow in tiny matrices.
    # Squaring a fixed 1e-6 relative gap (the former 1e-12 denominator term)
    # damped well-resolved modes and biased physical shape derivatives. Use
    # the dtype's rounding scale accumulated over n entries instead. This is
    # a numerical guard, not an exact derivative rule at repeated eigenvalues
    # or a condition estimate for defective eigenvectors.
    scale = xp.max(xp.abs(delta), axis=(-2, -1), keepdims=True)
    scale = xp.where(scale > 0., scale, xp.ones_like(scale))
    relative = delta / scale
    eps = (n * xp.finfo(xp.real(w).dtype).eps) ** 2
    f = (xp.conj(relative) / (xp.abs(relative)**2 + eps)) / scale
    f = xp.where(eye, xp.zeros_like(f), f)            # no broadening on the diagonal

    V_H = xp.matrix_transpose(xp.conj(V))
    VHgV = V_H @ gV

    rhs = (xp.astype(I, gw.dtype) * gw[..., None, :]
           + xp.conj(f) * VHgV
           - xp.conj(f) * (V_H @ V) @ xp.where(eye, xp.astype(xp.real(VHgV), VHgV.dtype),
                                               xp.zeros_like(VHgV))) @ V_H
    return xp.linalg.solve(V_H, rhs)


def _device_of(x):
    """``device=`` argument for building an array alongside ``x`` (None on NumPy/JAX)."""
    return x.device if hasattr(x, 'device') else None


def _eig_raw(A):
    """The framework's own eigendecomposition, with no gradient rule attached."""
    xp = ac.array_namespace(A)
    name = xp.__name__
    if 'torch' in name:
        import torch
        return torch.linalg.eig(A)
    if 'jax' in name:
        import jax.lax.linalg as jll
        out = jll.eig(A, compute_left_eigenvectors=False, compute_right_eigenvectors=True,
                      enable_eigvec_derivs=False)
        return out[0], out[-1]
    import numpy as np
    return np.linalg.eig(A)


def eig(A):
    """Eigendecomposition of a (possibly non-Hermitian) complex matrix.

    Returns ``(eigenvalues, right_eigenvectors)`` and is differentiable on JAX and PyTorch,
    including where eigenvalues are degenerate — see the note above this function.
    """
    xp = ac.array_namespace(A)
    name = xp.__name__
    if 'torch' in name:
        return _torch_eig(A)
    if 'jax' in name:
        return _jax_eig(A)
    return _eig_raw(A)


_JAX_EIG = None


def _jax_eig(A):
    """JAX ``eig`` with the broadened rule, built on first use so importing stays cheap."""
    global _JAX_EIG
    if _JAX_EIG is None:
        import jax

        @jax.custom_vjp
        def _eig_jax(A):
            return _eig_raw(A)

        def _fwd(A):
            w, V = _eig_raw(A)
            return (w, V), (w, V)

        def _bwd(res, g):
            import jax.numpy as jnp
            w, V = res
            grad_w, grad_V = g
            # JAX cotangents are with respect to the parameters; _eig_grad wants them with
            # respect to the conjugates, and returns one in the same convention.
            return (jnp.conj(_eig_grad(jnp, w, V, jnp.conj(grad_w), jnp.conj(grad_V))),)

        _eig_jax.defvjp(_fwd, _bwd)
        _JAX_EIG = _eig_jax
    return _JAX_EIG(A)


_TORCH_EIG = None


def _torch_eig(A):
    """PyTorch ``eig`` with the broadened rule (same construction as the JAX one)."""
    global _TORCH_EIG
    if _TORCH_EIG is None:
        import torch

        class _Eig(torch.autograd.Function):
            @staticmethod
            def forward(ctx, A):
                w, V = torch.linalg.eig(A)
                ctx.save_for_backward(w, V)
                return w, V

            @staticmethod
            def backward(ctx, grad_w, grad_V):
                w, V = ctx.saved_tensors
                return _eig_grad(ac.array_namespace(V), w, V,
                                 grad_w.resolve_conj(), grad_V.resolve_conj())

        _TORCH_EIG = _Eig
    return _TORCH_EIG.apply(A)


# --- Bessel J1: the second primitive that genuinely differs per framework ---------------
#
# NumPy has SciPy's j1. PyTorch has an accurate forward (`torch.special.bessel_j1`) but no
# autograd backward, so it needs the analytic derivative attached. JAX's
# `jax.scipy.special.bessel_jn` returns NaN for small arguments, so it needs its own
# implementation entirely. Same value, three routes — exactly what this adapter is for.

# The two pieces are textbook, not invented here: the ascending series is DLMF 10.2.2
# (Abramowitz & Stegun 9.1.10) and the large-argument form is Hankel's expansion, DLMF 10.17.3
# (A&S 9.2.1-9.2.6), whose P/Q coefficients are the a_k(nu) recursion built just below.
# Series below the threshold, Hankel asymptotic expansion above. Both constants were chosen
# by measurement against mpmath at 40 digits, 6000 points on (0, 95]: for each candidate
# (threshold, term count) the max error of the *combined* function was evaluated, and this is
# the minimum. Moving either one away from these values makes the worst case worse.
#
# The two branches fail in different ways, which is why the optimum is a compromise rather than
# "more terms is better":
#
#   * the series is limited by **cancellation**, not truncation. Its largest term at x is
#     ~(x/2)^x / (m!(m+1)!), which dwarfs the result, so past x ~ 12 it loses digits no term
#     count can restore -- 30 terms and 80 terms give the same 4.8e-13 at x <= 12.
#   * the asymptotic expansion is limited by truncation and is *divergent*: past its optimal
#     term count the error grows again. At x >= 12.5 it improves to 12 terms and worsens after.
#
# **The crossover depends on the precision, which is why there are two of them.** The series'
# loss is cancellation, so it costs a fixed number of *digits* -- which is a much larger share
# of float32's budget than of float64's. Optimising the two separately against mpmath:
# 12.5 in double (7.8e-13) and 7.0 in single (2.1e-06). Using the double-precision crossover
# in single costs 200x, and that was measured after it had already been written that way.
_J1_THR = 12.5
_J1_THR_F32 = 7.0
_J1_SERIES_TERMS = 30
_J1_ASYM_TERMS = 12
_J1_A = [1.0]
for _k in range(1, 2 * _J1_ASYM_TERMS + 2):
    _J1_A.append(_J1_A[-1] * (4.0 - (2 * _k - 1) ** 2) / (_k * 8.0))


def _j1_series(xp, x):
    """``(x/2) * sum_m (-x^2/4)^m / (m! (m+1)!)`` — accurate and stable for small ``x``."""
    term = x / 2.0
    s = term
    z = -(x * x) / 4.0
    for m in range(1, _J1_SERIES_TERMS):
        term = term * z / (m * (m + 1))
        s = s + term
    return s


def _j1_asymptotic(xp, x):
    """Hankel asymptotic expansion — takes over where the series loses precision."""
    # Positive powers of large x overflow in float32, including in derivatives
    # of a quotient whose value is finite. Horner evaluation in 1/x**2 keeps
    # every intermediate bounded on this branch's x >= crossover domain.
    inverse = 1.0 / x
    inverse2 = inverse * inverse
    P = xp.zeros_like(x)
    Q = xp.zeros_like(x)
    for k in reversed(range(_J1_ASYM_TERMS)):
        P = ((-1) ** k) * _J1_A[2 * k] + inverse2 * P
        Q = ((-1) ** k) * _J1_A[2 * k + 1] + inverse2 * Q
    Q = inverse * Q
    omega = x - 3.0 * 3.141592653589793 / 4.0
    return xp.sqrt((2.0 / 3.141592653589793) * inverse) * (P * xp.cos(omega) - Q * xp.sin(omega))


_J1_CHOICES = ('auto', 'hybrid', 'native')
_j1_impl = 'auto'


def set_j1_implementation(name):
    """Choose which ``J_1`` the traced backends use. Returns the previous setting.

    Only PyTorch has a real choice here; see ``docs/bessel-j1.md`` for the measurements.

    ``'auto'`` (default)
        The more accurate one for the precision in hand: the hybrid in float64 (1.1e-12
        against ``torch.special.bessel_j1``'s 4.1e-07) and the framework's own in float32
        (4.7e-07 against the hybrid's 2.1e-06). Neither is better at both, so the default
        does not pretend one is.

    ``'hybrid'``
        The series/asymptotic hybrid always. Pick this when you want the two traced backends
        to compute bit-comparable values regardless of dtype, or when profiling something
        else and you want one fewer variable.

    ``'native'``
        ``torch.special.bessel_j1``, wrapped for the autograd backward it does not ship.
        Faster as a function -- 29-170x -- though that is 0.01-0.3% of a solve, so this is
        rarely the reason to choose it. **Costs four orders of accuracy in float64.**

    JAX ignores the setting and always takes the hybrid: ``jax.scipy.special.bessel_jn``
    returns NaN for small arguments, so there is nothing to switch to. Asking for
    ``'native'`` does not raise for that reason -- a global switch should not break one
    backend because another lacks an option -- but it does not change JAX's behaviour.
    """
    global _j1_impl
    if name not in _J1_CHOICES:
        raise ValueError(f"j1 implementation {name!r} is not one of {list(_J1_CHOICES)}")
    previous, _j1_impl = _j1_impl, name
    return previous


def get_j1_implementation():
    """The current :func:`set_j1_implementation` setting."""
    return _j1_impl


def j1(x):
    """Bessel function of the first kind, order 1 — differentiable on every backend.

    Defined for ``x >= 0`` (the parametric path's shape transforms only ever evaluate it there).
    Which implementation runs is :func:`set_j1_implementation`'s to decide.
    """
    xp = ac.array_namespace(x)
    name = xp.__name__
    if 'torch' in name:
        single = xp.finfo(x.dtype).bits <= 32
        # 'auto' takes the accurate one for this precision, which is the framework's below
        # 64-bit and the hybrid at 64-bit. See set_j1_implementation.
        if _j1_impl == 'native' or (_j1_impl == 'auto' and single):
            return _TorchBesselJ1.apply(x)
    if 'torch' in name or 'jax' in name:
        # No `abs` here, deliberately. It looks harmless because the domain is x >= 0, but it
        # is even where J1 is odd, and -- worse -- its derivative at 0 is 0 in both frameworks,
        # so it silently zeroes dJ1/dx at the origin. The series below is odd by construction
        # and gives the correct 1/2 there.
        #
        # As an array of x's own dtype: torch.minimum refuses a Python float, and building it
        # this way keeps the clamp in the input's precision.
        single = xp.finfo(x.dtype).bits <= 32
        thr = xp.full_like(x, _J1_THR_F32 if single else _J1_THR)
        # Clamp each branch's input so the *unselected* branch stays finite: a NaN there
        # would still poison the gradient through `where`.
        small = x < thr
        # A min/max clamp halves the selected derivative at exact equality.
        # Select safe inputs with the same predicate as the output instead.
        return xp.where(small,
                        _j1_series(xp, xp.where(small, x, thr)),
                        _j1_asymptotic(xp, xp.where(small, thr, x)))
    from scipy.special import j1 as _scipy_j1
    import numpy as np
    return _scipy_j1(np.asarray(x))


class _TorchBesselJ1Meta(type):
    """Builds the torch autograd Function lazily so importing this module never needs torch."""
    _impl = None

    def apply(cls, x):
        if cls._impl is None:
            import torch

            class _Impl(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return torch.special.bessel_j1(x)

                @staticmethod
                def backward(ctx, grad_out):
                    # dJ1/dx = J0(x) - J1(x)/x, and 1/2 at x = 0.
                    (x,) = ctx.saved_tensors
                    safe = torch.where(x == 0, torch.ones_like(x), x)
                    dj1 = torch.where(
                        x == 0, torch.full_like(x, 0.5),
                        torch.special.bessel_j0(x) - torch.special.bessel_j1(x) / safe)
                    return grad_out * dj1

            cls._impl = _Impl
        return cls._impl.apply(x)


class _TorchBesselJ1(metaclass=_TorchBesselJ1Meta):
    """``torch.special.bessel_j1`` with the analytic backward it is missing."""


def jinc(x):
    """``2 J1(x) / x``, with the removable singularity filled at 0 (value 1)."""
    xp = ac.array_namespace(x)
    safe = xp.where(x == 0, xp.ones_like(x), x)
    return xp.where(x == 0, xp.ones_like(x), 2.0 * j1(safe) / safe)


def sinc(z):
    """Unnormalized ``sin(z)/z``, with the removable singularity filled at 0."""
    xp = ac.array_namespace(z)
    safe = xp.where(z == 0, xp.ones_like(z), z)
    return xp.where(z == 0, xp.ones_like(z), xp.sin(safe) / safe)
