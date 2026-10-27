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
import contextvars
import functools
import sys

import array_api_compat as ac
import numpy as _np


#: What Dynamo prints when it meets a wavebit door: ``torch.compile(fullgraph=True)`` and a
#: strict ``torch.export`` raise ``Unsupported`` carrying it.
EAGER_REASON = ('wavebit runs eagerly under torch.compile: compiling it buys no speed, and '
                'Inductor has silently miscompiled its complex arithmetic before (torch 2.13, '
                'fixed in 2.14) -- see docs/development/torch-compile.md')

#: Each door's ``torch.compiler.disable`` wrapper, keyed by the function it runs.
_EAGER_DOORS = {}

#: Why a graph capture that runs Python natively is refused (non-strict ``torch.export``).
EXPORT_REFUSAL = (
    'wavebit cannot be exported: torch.export records one branch of each value-dependent decision '
    'wavebit takes, and from an example input that needs no gradient none of its modal derivative '
    '-- measured, the exported module returns eager values and no gradient. Call wavebit eagerly '
    '(torch.compile of a caller is exact: wavebit runs eagerly inside it).')

#: Why ``torch.jit.trace`` is refused (by every door, :func:`_eager_function`).
JIT_TRACE_REFUSAL = (
    'wavebit cannot be traced by torch.jit.trace: the trace records each value-dependent decision '
    'wavebit takes as a constant and drops the gradient wavebit contributes -- measured, a loss '
    'mixing a traced solve with another term read a gradient of 0.0 where eager torch reads '
    '0.272, with no error. Call wavebit eagerly, or from a torch.compile\'d function, inside '
    'which wavebit runs eagerly.')


def _refusing_export(fn):
    """``fn``, refused while ``torch.export`` records it.

    A strict export goes through Dynamo and stops at the door's ``torch.compiler.disable`` with
    :data:`EAGER_REASON`; a non-strict export runs the door natively under a fake-tensor proxy
    mode, where ``disable`` does not apply (it only annotates), so it is refused here. It needs
    no Dynamo-free twin: a non-strict export imports ``torch._dynamo`` (its ``make_fx`` does), so
    it always takes the door's Dynamo route. The flag is PyTorch's public
    ``torch.compiler.is_exporting()`` (in ``torch.compiler.__all__``;
    docs.pytorch.org/docs/2.13/generated/torch.compiler.is_exporting.html), which "would only be
    set to True when torch.export is used".
    """
    @functools.wraps(fn)
    def eager(*args, **kwargs):
        if sys.modules['torch'].compiler.is_exporting():
            raise NotImplementedError(EXPORT_REFUSAL)
        return fn(*args, **kwargs)
    return eager


def _eager_function(fn):
    """``fn`` as a door ``torch.compile`` never traces into; the arithmetic is ``fn``'s own.

    Inductor (torch 2.13, CPU, complex128 with an input that requires grad) returned wrong values
    and gradients for this solver with no error: it emitted the in-place diagonal write of
    ``array_api_compat``'s ``eye`` after the kernels that read it, so the modal exponential's
    identity read as zero. Dynamo alone and AOTAutograd alone were exact, so the cure was to keep
    wavebit's frames out of the compiled graph, not to change its arithmetic. PyTorch fixed the
    defect in 2.14.0 (pytorch#182012), which is the floor; the doors stay, because compiling
    wavebit's part buys no speed and they keep it out of the next Inductor defect too.

    Two shortcuts that look equivalent are not:

    * **A refusal cannot be raised instead.** Dynamo turns an exception raised while tracing
      into a graph break and reruns the frame natively, where
      ``torch.compiler.is_compiling()`` is False, so the refusal is swallowed.
    * **``is_compiling()`` cannot choose the route.** Dynamo stops tracing a wrapper whose
      frame compiled to nothing and runs it natively from then on, with the door's own frames
      still intercepted and compiled.

    So whenever Dynamo is loaded every call goes through the door's ``torch.compiler.disable``
    wrapper -- PyTorch's documented way to keep a function and everything it calls out of
    ``torch.compile`` (docs.pytorch.org/docs/2.13/generated/torch.compiler.disable.html) -- which
    clears Dynamo's frame hook for the call and restores it after; nested doors pay it again,
    because a caller's compiled callable can set the hook back inside an outer door. The wrapper
    is built on the door's first call with Dynamo loaded, the idea of PyTorch's own
    ``torch._compile._disable_dynamo`` (torch 2.13), which wraps lazily to keep ``import torch``
    from importing Dynamo. Unlike it, a door never imports Dynamo: when ``torch._dynamo`` is not
    in ``sys.modules`` nothing can be compiling (``torch.compile`` imports it), and the door
    calls ``fn`` directly.

    **Every door refuses ``torch.jit.trace``, by name** (:data:`JIT_TRACE_REFUSAL`), on both
    routes. Dynamo is not involved and the values are concrete, so nothing else notices it, and
    it is silently wrong: it records each value-dependent decision as a constant and returns an
    output with no ``grad_fn``. The door is the one place every public call passes. A check
    where a solve reads a value (``_concrete``) missed what never reaches it:
    ``reconstruct_field`` of a solved result at a traced height replayed one layer's formula
    across an interface (d/dz -0.00406 where eager reads 0.00344) and answered outside the
    stack, where eager refuses, and a ``Lattice`` built from a traced length became a NumPy
    constant (``meshgrid_xy`` replayed the example's value, with no gradient). The check reads
    PyTorch's public ``torch.jit.is_tracing()``
    (docs.pytorch.org/docs/2.13/generated/torch.jit.is_tracing.html) from ``sys.modules``, so a
    NumPy or JAX caller never imports torch for it; Dynamo folds it to ``False`` while it traces
    (``torch._dynamo.variables.torch.tracing_state_functions``, torch 2.13). It adds about
    100 ns to a door call (``docs/development/torch-compile.md`` §6).
    """
    if hasattr(fn, '_wavebit_eager_door'):
        return fn

    @functools.wraps(fn)
    def door(*args, **kwargs):
        modules = sys.modules
        torch = modules.get('torch')
        if torch is not None and torch.jit.is_tracing():
            raise NotImplementedError(JIT_TRACE_REFUSAL)
        if 'torch._dynamo' not in modules:
            return fn(*args, **kwargs)
        run = _EAGER_DOORS.get(fn)
        if run is None:
            run = _EAGER_DOORS[fn] = modules['torch'].compiler.disable(
                _refusing_export(fn), reason=EAGER_REASON)
        return run(*args, **kwargs)

    door._wavebit_eager_door = fn
    return door


#: Class members a door would not protect: hooks Python calls while *building* a class.
_NOT_DOORS = frozenset({'__init_subclass__', '__class_getitem__', '__set_name__'})


def _eager_member(name, member):
    if name in _NOT_DOORS or (name.startswith('_') and not
                              (name.startswith('__') and name.endswith('__'))):
        return member                         # private: reached only from inside a door
    if isinstance(member, property):
        return property(*(None if f is None else _eager_function(f)
                          for f in (member.fget, member.fset, member.fdel)), member.__doc__)
    if isinstance(member, (classmethod, staticmethod)):
        return type(member)(_eager_function(member.__func__))
    if callable(member) and hasattr(member, '__code__'):
        return _eager_function(member)
    return member


def eager_under_torch_compile(target):
    """Keep a public function, or every public member of a class, out of ``torch.compile``.

    On a class: each public method, property accessor, ``classmethod`` / ``staticmethod`` and
    dunder in the class's *own* namespace becomes a door (see :func:`_eager_function`);
    single-underscore members are reached only from inside a door and stay as they are. An
    inherited member is a door when the class that defines it carries this decorator, which
    is why every wavebit class in a public class's MRO carries it.

    The rule, which ``tests/api/test_torch_compile.py`` walks on every installed backend: each
    class in ``wavebit.__all__``, each type a public call hands back (an attribute, a pytree
    node, ``Lattice.drawn``'s return) and every wavebit base of either. A class only ever built
    and used inside a door needs none, since everything a door calls is already kept out.
    """
    if isinstance(target, type):
        for name, member in list(vars(target).items()):
            door = _eager_member(name, member)
            if door is not member:
                setattr(target, name, door)
        target._wavebit_eager_class = target
        return target
    return _eager_function(target)


def array_namespace(*xs):
    """Return the Array-API namespace for the given array(s) (numpy / jax.numpy / torch)."""
    return ac.array_namespace(*xs)



from contextvars import ContextVar
ALLOW_LEGACY_PRECISION_LIFT = ContextVar('wavebit_allow_legacy_precision_lift', default=True)

def lift_jax_x64(required=False):
    """Lift JAX's 32-bit cap, if JAX is loaded, and say whether it is lifted. **The one body.**

    It lifts a *cap*, not a default: with ``jax_enable_x64`` off, ``jnp.asarray(x,
    jnp.float64)`` still yields **float32**, so double precision cannot be opted into at all.
    That is why the ordering is a correctness condition rather than a preference — wavebit
    asks for float64 correctly, JAX hands back float32, every reported dtype still reads
    float64 because the result is widened on the way out, and nothing warns. Widening
    afterwards cannot recover digits that were never computed: a solve reporting
    ``complex_dtype=0`` everywhere once returned ``|R| = 3.94`` against a true 0.67. See
    :class:`PrecisionLossWarning` and ``docs/theory/rcwa/branch-cuts.md``.

    Historical numerical engine entry points call this after JAX is loaded and
    before building their first array: the private JAX RCWA backend, the periodic
    JaxTimeLoop, lattice sampling and the explicit _enable_x64 instrument helper.
    The current physical lifecycle uses Execution.scope to disable those automatic
    changes and validates the user's explicit precision configuration instead.
    Importing wavebit never lifts the cap.

    This function does not import JAX. With required=False, an absent or partially
    imported JAX is a no-op and the return value reports the current cap. Internal
    callers that have imported JAX and require double precision pass required=True,
    so a failed lift raises before arrays are constructed. Under Execution.scope,
    the helper only reports the setting and never changes it.
    """
    import sys
    jax = sys.modules.get('jax')
    if not ALLOW_LEGACY_PRECISION_LIFT.get():
        return bool(getattr(getattr(jax, 'config', None), 'jax_enable_x64', False))
    failure = None
    try:
        jax.config.update('jax_enable_x64', True)
    except Exception as err:          # absent, or mid-import and not yet usable
        failure = err
    # Read back rather than trusted: what the run gets is the config, not the call's return.
    lifted = bool(getattr(getattr(jax, 'config', None), 'jax_enable_x64', False))
    if required and not lifted:
        raise RuntimeError(
            "could not lift JAX's 32-bit cap (jax_enable_x64), so a JAX run would silently be "
            'single precision') from failure
    return lifted



def carries_autodiff(value):
    """Whether selecting equal entries could discard independent input derivatives.

    Every JAX tracer counts, including a plain JIT input: that compiled function can later
    be differentiated with grad(jit(f)). Deciding only when a JVP tracer arrives would make
    the derivative depend on transformation order. On PyTorch both modes count: a tensor
    under reverse mode (``requires_grad``) and a forward-mode dual (:func:`forward_tangent`),
    which has ``requires_grad`` False -- asking only the first sent every dual down the
    constant branch of each dispatch that reads this, and the tangent came back wrong with no
    error. Eager arrays need no special treatment.

    Inside a ``torch.func`` transform the PyTorch answer is :func:`torch_autodiff`'s, which
    asks every wrapper level under the value as well as the value itself. ``torch.func.vmap``
    hands the solve ``BatchedTensor`` wrappers, and a wrapper reads ``requires_grad`` False
    even when the tensor it batches requires grad, so until 2026-10-11 every modal gate took
    its constant branch under ``vmap`` and ``backward()`` through it returned the derivative
    with the eigen-derivative missing, with no error: ``dR/d(wavelength)`` on a conical
    grating read -0.003033 where eager autograd and a central difference read -0.004225, and
    ``create_graph=True`` through ``vmap`` returned a partial Hessian where eager refuses.
    """
    if torch_autodiff(value):
        return True
    import sys
    jax = sys.modules.get('jax')
    return jax is not None and isinstance(value, jax.core.Tracer)


def torch_autodiff(value):
    """Whether ``value`` is a PyTorch tensor that some PyTorch autodiff level differentiates.

    That is ``requires_grad`` or a forward-mode tangent (:func:`forward_tangent`) at the tensor
    itself or, inside a ``torch.func`` transform, at any level under its wrappers. A
    ``torch.func.vmap`` ``BatchedTensor`` reads ``requires_grad`` False even when the tensor it
    batches requires grad -- a captured parameter, or the input of ``vmap(f)(x).sum().backward()``
    -- so asking the outermost wrapper alone sent every modal gate down its constant branch under
    ``vmap``, silently: ``dR/d(wavelength)`` of a conical grating read -0.003033 where eager
    autograd and a central difference read -0.004225, and a census of 34 scenes under ten
    transforms found 60 of its 340 cells wrong with no error, by up to a factor of 20.

    Each level's *own* flag is asked, never the wrapper's type. A ``torch.func.grad`` wrapper
    detached inside the transform is still a ``GradTrackingTensor``, and counting it as live
    would send an operator the solve stopped on purpose into the first-order rules. A level's
    tangent is read at that level (:func:`_own_tangent`): read from above, a jvp level beneath
    another ``torch.func`` grad or jvp level, or a ``forward_ad`` dual beneath one, reads no
    tangent, and a mixed partial such as ``torch.func.jacfwd`` over a permittivity of
    ``torch.func.grad`` over a thickness went down the constant branch and came back 0.15
    relative off with no error (806661e4 and before) -- it is refused at the rule's door now.
    The door itself is :func:`functorch_differentiates`.

    The walk is ``torch.func.debug_unwrap``'s, in PyTorch's ``torch._functorch.eager_transforms``:
    ``is_functorch_wrapped_tensor`` and ``get_unwrapped`` from ``torch._C._functorch``, which is
    private (:func:`_functorch` refuses by name a PyTorch without them). It runs only while a
    transform is active (:func:`_functorch_active`); wrappers exist only inside one, so eager
    PyTorch never calls ``torch._C._functorch`` while the transforms-active probe exists, and
    takes exactly the path it took before; the probe itself is read on every call. ``False``
    for anything that is not a PyTorch tensor, without importing PyTorch -- which is why the
    raw readers of a geometry or a cache key call this rather than :func:`carries_autodiff`:
    that one also counts every JAX tracer.
    """
    if getattr(value, 'requires_grad', False):
        return True
    import sys
    torch = sys.modules.get('torch')
    if torch is None or not isinstance(value, torch.Tensor):
        return False
    active = _functorch_active(torch)
    if _forward_tangent(torch, value, active):
        return True
    if not active:
        return False
    functorch = _functorch(torch)
    # The tangents below were all asked above; what is left is each level's own `requires_grad`.
    while functorch.is_functorch_wrapped_tensor(value):
        value = functorch.get_unwrapped(value)
        if value.requires_grad:
            return True
    return False


#: The private ``torch._C._functorch`` calls the wrapper walks need: the two the public
#: ``torch.func.debug_unwrap`` is written with (in ``torch._functorch.eager_transforms``), the
#: two that name a level's kind, and the four that read a tangent at its own level
#: (:func:`_own_tangent`): a tensor's level, the top interpreter's, and popping and pushing the
#: interpreter stack, as ``torch._functorch.pyfunctorch.temporarily_pop_interpreter_stack`` and
#: ``temporarily_clear_interpreter_stack`` do (the latter's pop loop and reverse push).
_FUNCTORCH_CALLS = ('is_functorch_wrapped_tensor', 'get_unwrapped', 'is_batchedtensor',
                    'is_gradtrackingtensor', 'maybe_get_level', 'maybe_current_level',
                    'pop_dynamic_layer_stack', 'push_dynamic_layer_stack')


def _functorch_active(torch):
    """Whether a ``torch.func`` transform is active: the private
    ``torch._C._are_functorch_transforms_active``.

    Wrappers exist only inside a transform, so outside one there is nothing to walk and the
    eager answers are the plain flags. Where this PyTorch lacks the probe, a transform is
    assumed active, so the walk runs -- correct, at the walk's cost -- and :func:`_functorch`
    names any call it is missing. The probe is private: if it read False inside a transform,
    the walks would be skipped and the silent ``vmap`` defect :func:`torch_autodiff` describes
    would return, which ``tests/rcwa/test_torch_vmap_autograd.py`` trips on.
    """
    active = getattr(torch._C, '_are_functorch_transforms_active', None)
    return True if active is None else bool(active())


#: What :func:`linear_solve` reads beside them: which transform owns each level, and the grad
#: mode a grad level was entered with (``torch._functorch.pyfunctorch`` reads the stack the same
#: way, and ``GradInterpreter.prev_grad_mode`` through ``CGradInterpreterPtr``). Added to
#: :data:`_FUNCTORCH_CALLS` when :func:`_functorch` is called, not bound here, so the list is
#: read at call time.
_LEVEL_EXTRA = ('get_interpreter_stack', 'TransformType', 'CGradInterpreterPtr')


def _functorch(torch, extra=()):
    """``torch._C._functorch``, refusing by name a PyTorch without a call the walks need.

    Those calls are private (:data:`_FUNCTORCH_CALLS`), and a PyTorch that renamed one would
    otherwise raise ``AttributeError`` deep in a solve -- or, worse, be taken for a tensor with
    no derivative. Called only inside a transform -- or on every call, eager included, on a
    PyTorch without the transforms-active probe (:func:`_functorch_active`), where eager solves
    would then raise this too if a call were missing as well.
    """
    functorch = getattr(torch._C, '_functorch', None)
    missing = [name for name in _FUNCTORCH_CALLS + tuple(extra) if not hasattr(functorch, name)]
    if missing:
        raise RuntimeError(
            f'wavebit reads autodiff through torch.func transforms with torch._C._functorch.'
            f'{missing[0]}, which PyTorch {torch.__version__} does not have, so a derivative '
            f'through this transform cannot be taken correctly; use torch.autograd outside '
            f'torch.func')
    return functorch


def functorch_differentiates(value):
    """Whether a ``torch.func`` grad or jvp level may differentiate ``value``: the door of the
    first-order rules (``modal_gradient._variation`` and the NVF floor), which refuse it.

    ``torch.func.grad``, ``vjp``, ``jvp``, ``jacrev``, ``jacfwd`` and ``hessian`` all record
    with ``GradTrackingTensor`` wrappers, and so do their compositions with ``vmap`` in either
    order; ``vmap`` alone wraps in ``BatchedTensor`` only. Every tensor computed inside a grad or
    jvp transform carries that level's wrapper, live or not: a value built from a parameter
    captured with ``requires_grad`` -- an ``nn.Parameter``-style permittivity -- is lifted to a
    ``GradTrackingTensor`` whose own ``requires_grad`` is False while the level under it
    requires grad. So the first such level in the chain, the transform's own, counts only when
    it is live there: its ``requires_grad``, or a tangent read at that level
    (:func:`_own_tangent`). ``torch.func.grad`` or ``jvp`` over a thickness stays exact with
    a captured ``requires_grad`` permittivity in the operator (-0.0005768798063578665 on the 1D
    TM SMM scene, eager's, as at 806661e4), and a later ``backward()`` to that permittivity is
    the eager mixed derivative. Counted by its type, that level refused both: a value computed
    inside the transform carries its wrapper whether or not it depends on the input.

    Any further grad or jvp level, beneath that one, counts by its *type*, live or not -- a
    nested composition, ``hessian``, ``grad(grad)``, ``jvp(jvp)`` or a mixed partial, where the
    operator carries any derivative: a refusal must not miss a level that could differentiate
    the rules, and two levels that do would differentiate the first-order rule twice, a partial
    Hessian, should a level's tangent ever read wrong. A solve under nested transforms reaches
    this door only where the operator carries a derivative at all (:func:`torch_autodiff`),
    so over a thickness with a constant permittivity it stays exact. ``False`` outside a
    transform and for anything that is not a PyTorch tensor, without importing PyTorch.
    """
    import sys
    torch = sys.modules.get('torch')
    if torch is None or not isinstance(value, torch.Tensor) or not _functorch_active(torch):
        return False
    functorch = _functorch(torch)
    from torch.autograd import forward_ad
    nested = False
    while functorch.is_functorch_wrapped_tensor(value):
        if functorch.is_gradtrackingtensor(value):
            if nested or value.requires_grad or _own_tangent(functorch, forward_ad, value):
                return True
            nested = True
        value = functorch.get_unwrapped(value)
    return False


def forward_tangent(x):
    """Whether ``x`` is a PyTorch forward-mode dual: ``torch.autograd.forward_ad``'s, which
    ``torch.func.jvp`` and ``jacfwd`` make too. A dual has ``requires_grad`` False, so a test
    for reverse mode alone does not see it, and the constant branch hands the tangent a zero:
    on the gold disks at 81 harmonics a directional derivative of a read-out of
    ``eps_conv_matrices()`` read -0.538854 through the inverse-rule floor where reverse mode
    and a central difference read -27.626431, and ``d(R + T)/d(diameter)`` of dielectric disks
    at 81 harmonics -5.48e-6 through the modal operator where they read -7.35e-5, with no
    error. ``False`` for anything that is not a PyTorch tensor, without importing PyTorch.

    A tensor, not anything of PyTorch's: inside a dual level ``unpack_dual`` raises
    ``TypeError`` for a ``torch.dtype``, ``device`` or ``Size``, which the solver holds and
    ``engine._holds_autodiff`` walks on an ``auto_detune`` retry -- a plain solve at a Rayleigh
    point raised there whenever a dual level was open.

    Inside a ``torch.func`` transform the levels under ``x``'s wrappers are asked too, as
    :func:`torch_autodiff` asks them: ``forward_ad`` through ``torch.func.vmap`` puts the dual
    *under* a ``BatchedTensor``. A batched level is skipped rather than asked -- ``unpack_dual``
    has no batching rule, and inside a dual level it raises "Batching rule not implemented for
    aten::_unpack_dual" on one -- and the level under it is asked instead, at its own level
    (:func:`_own_tangent`). Outside a transform this is exactly the plain ``unpack_dual`` test
    it always was."""
    import sys
    torch = sys.modules.get('torch')
    if torch is None or not isinstance(x, torch.Tensor):
        return False
    return _forward_tangent(torch, x, _functorch_active(torch))


def _forward_tangent(torch, x, active):
    """:func:`forward_tangent` of a tensor ``x``, with :func:`_functorch_active` already asked."""
    from torch.autograd import forward_ad
    if not active:
        return forward_ad.unpack_dual(x).tangent is not None
    functorch = _functorch(torch)
    while True:
        if not functorch.is_batchedtensor(x) and _own_tangent(functorch, forward_ad, x):
            return True
        if not functorch.is_functorch_wrapped_tensor(x):
            return False
        x = functorch.get_unwrapped(x)


# --- linear solves, and PyTorch's derivatives of them ----------------------------------
#
# Two PyTorch 2.13 defects make a derivative through `torch.linalg.solve` wrong with no error,
# and both live in the rules PyTorch attaches to it, not in the solve. Their status per torch
# version, and when this routing can be narrowed or removed: docs/development/torch-upstream-defects.md.
#
# 1. Its forward-mode rule (`linalg_solve_jvp`, torch/csrc/autograd/FunctionsManual.cpp) is
#    `dX = lu_solve(LU, dB - dA X)` on the LU factors of the primal, which
#    tools/autograd/derivatives.yaml marks non-differentiable, and its backward
#    (`linalg_solve_backward`) reuses them the same way whenever grad mode is off. A derivative
#    of either rule holds A^-1 fixed and drops `A^-1 A' X'`: `jvp(jvp)` equals the exact second
#    derivative plus that term to 8e-16, and a thickness second derivative of a 2D ETM raster
#    by `jvp(jvp)` read 760.7 where it is 812.6 (pytorch#196695). PyTorch's fix
#    (pytorch#192667, commit 7c2af32, in the 2.15 release candidates, not in 2.14.1) re-solves
#    the forward rule; the backward's reuse with grad mode off stays (pytorch#200600, open; still
#    wrong on the 2.16 nightly).
# 2. `lu_solve` under two `torch.func.vmap` levels, its factors batched by the outer one and its
#    right side by the inner one, returns wrong lanes -- the value itself. The forward rule
#    above calls it so, so `vmap(jacfwd(f))`, a first derivative, read 197.0 for 0.01147 on a
#    1D TM grating's thickness (lane 0 right, the others wrong or NaN), and `vmap(hessian(f))`
#    likewise. The cause is the CPU kernel's batch stride: `apply_lu_solve` reads the distance
#    between LU matrices from `LU.stride(-3)`, which a size-1 axis inserted there does not carry
#    -- it reproduces without `vmap` on `LU[:, None]` (pytorch#200599, open; still wrong on the
#    2.16 nightly).
#
# Where neither can happen this is `xp.linalg.solve` itself -- every eager solve, every first
# derivative but (2)'s and, under `torch.no_grad()`, `vmap(grad)` and `vmap(vjp)`, and the
# second-derivative routes PyTorch gets right (`hessian`, `jvp` of `grad`, `create_graph=True`,
# whose backward runs with grad mode on and re-solves) -- so their arithmetic is unchanged byte for
# byte, and their cost. Grad mode off is not only `create_graph` left False: `torch.func.vjp`,
# `jacrev` and `hessian` run their backward with `create_graph` equal to the grad mode they are
# called in, and PyTorch documents `torch.func`'s grad transforms as ignoring an outer
# `torch.no_grad()`, so under one their backward reuses the factors: `hessian` of the 2D ETM raster
# read the 760.7 above, and `vmap(jacrev)`, a first derivative, NaN on lanes 1 and 2, its backward's
# `lu_solve` running under the outer `vmap` and `jacrev`'s own. Where one can, the solve is spelled
# in operations whose rules PyTorch differentiates correctly to every order: `lu_factor` +
# `lu_solve` outside `vmap`, which are the LAPACK calls `linalg.solve` makes itself, so the value is
# the same bytes; under `vmap`, `lu_factor` + `lu_unpack` + two `solve_triangular`, which avoid
# `lu_solve`'s batching and agree with it to rounding. Both are the textbook solve by Gaussian
# elimination with partial pivoting, PA = LU, then a unit lower and an upper triangular solve (G. H.
# Golub and C. F. Van Loan, Matrix Computations, 4th ed., Johns Hopkins, 2013, sections 3.2-3.4).
# The decision borrows the test PyTorch's own `linalg_solve_backward` (FunctionsManual.cpp) makes
# before reusing the factors -- whether grad mode is on, that is, whether its result will be
# differentiated again -- and asks it of the transform levels. A level's rule holds the factors
# fixed if the level is forward mode, or a `torch.func` grad level entered with grad mode off (read
# as `torch._functorch.pyfunctorch`'s `GradInterpreter.prev_grad_mode` reads it). The rule is
# rewritten only where such a level sits inside another level; where a `forward_ad` dual is made
# inside a grad level, whose tangent sits on that level's wrapper and whose forward rule the level
# then differentiates (0.57 relative off on the closed form); or under enough `vmap` levels for (2).
# A backward whose grad mode differs from the one its level was entered with -- a `vjp_fn` called
# under `torch.no_grad()` or with `create_graph=False` -- has not run when the solve is taken; where
# it would be wrong it is refused when it runs (`_guard_the_backward`). Held to a closed form over
# every nest of up to three `torch.func` transforms under each order of plain autograd and
# `forward_ad` (`tests/common/test_linear_solve_derivatives.py`): of 784 nests of first to third order,
# `torch.linalg.solve` gets 360 wrong, this none, and PyTorch itself refuses 168 ("Nested forward
# mode AD is not supported"); and on the routes above under `torch.no_grad()`, with a dual made
# inside `torch.func.grad`, `vjp` or `jacrev`, and with a `vjp_fn` called in another grad mode.

def linear_solve(xp, A, B):
    """``xp.linalg.solve(A, B)``, whose derivatives are exact on PyTorch at every order.

    Every solve whose operands can carry a derivative goes through here (see the note above).
    It *is* ``xp.linalg.solve(A, B)`` on NumPy and JAX, and on PyTorch except where a level
    whose rule holds the factors fixed would have that rule differentiated again: a forward
    level inside another (``jvp(jvp)``, ``jacfwd(jacfwd)``, ``grad(jvp)``, ``jacrev(jacfwd)``, a
    backward pass over a ``forward_ad`` tangent, a tangent through a backward pass, any higher
    nest with a forward level inside another), a grad level entered with grad mode off inside
    another (``hessian``, ``jvp(jacrev)`` or ``jvp(vjp)`` under ``torch.no_grad()``), or a
    ``forward_ad`` dual made inside a grad level (``torch.func.grad``, ``vjp`` or ``jacrev`` of
    a tangent). Or where such a rule would run under two ``vmap`` levels (``vmap(jacfwd)``,
    ``vmap(hessian)``, and ``vmap(jacrev)`` under ``torch.no_grad()``, and with it ``vmap(grad)``
    and ``vmap(vjp)``, which cannot be told from it at the solve).

    A grad level is read by the grad mode it was entered with, the mode ``torch.func.vjp``'s
    ``vjp_fn`` runs its backward in when ``jacrev`` or ``hessian`` calls it. A ``vjp_fn`` called
    in another mode, or with ``create_graph=False``, cannot be seen here -- its backward has not
    run yet -- so where that would be wrong it is refused from inside the backward pass
    (:func:`_guard_the_backward`).
    """
    matrix = _derivative_levels(A)
    # Both cases need a level on the matrix or a torch.func transform; an eager solve stops here.
    if matrix or _transform_active(A):
        levels = matrix | _derivative_levels(B)
        if levels:
            modes, vmaps, grad_off = _transform_modes()
            # Outside a transform `modes` is empty: an eager solve reads nothing private.
            within = _duals_within_grad((A, B), modes) if modes and (matrix or vmaps) else set()
            if ((matrix and _forward_inside(levels, modes, grad_off, within))
                    or _forward_under_two_vmaps(levels, modes, vmaps, grad_off, within)):
                return _differentiable_solve(A, B, batched=bool(vmaps))
            if modes:
                solved = xp.linalg.solve(A, B)
                _guard_the_backward(solved, matrix, levels, modes, vmaps)
                return solved
    return xp.linalg.solve(A, B)


def _transform_active(value):
    """Whether ``value`` is a PyTorch tensor while a ``torch.func`` transform is active."""
    import sys
    torch = sys.modules.get('torch')
    return torch is not None and isinstance(value, torch.Tensor) and _functorch_active(torch)


def _differentiable_solve(A, B, batched):
    """``torch.linalg.solve(A, B)`` from operations differentiable at every order, reading ``B``
    as ``array_api_compat``'s ``solve`` (1.15) does: promoted to a common dtype, a 1-D ``B`` a
    vector, a ``B`` of ``A``'s shape less its last axis one matrix. ``batched``: a ``vmap`` level
    is active, so ``lu_solve`` is not used (defect 2 in the note above)."""
    import torch
    dtype = torch.promote_types(A.dtype, B.dtype)
    A, B = A.to(dtype), B.to(dtype)
    if B.ndim != 1 and A.ndim - 1 == B.ndim and A.shape[:-1] == B.shape:
        B = B[None]
    vector = B.ndim == 1
    B = B[..., None] if vector else B
    LU, pivots = torch.linalg.lu_factor(A)
    if batched:
        P, L, U = torch.lu_unpack(LU, pivots)
        Y = torch.linalg.solve_triangular(L, P.mT @ B, upper=False, unitriangular=True)
        X = torch.linalg.solve_triangular(U, Y, upper=True)
    else:
        X = torch.linalg.lu_solve(LU, pivots, B)
    return X[..., 0] if vector else X


def _transform_modes():
    """``({level: TransformType}, [vmap level, ...], {grad level entered with grad mode off,
    ...})`` for the ``torch.func`` interpreters on the stack -- empty unless a transform is
    active, so an eager solve reads nothing private.

    The mode is read off each interpreter's ``key()``, as ``torch._functorch.pyfunctorch``'s
    ``coerce_cinterpreter`` reads it, and compared with ``TransformType.Jvp``, ``Vmap`` and
    ``Grad``. A key that is not a ``TransformType`` member -- a PyTorch that renumbered the
    stack's keys, or returned a plain int -- would compare unequal to all three, so every level
    would read as no forward level and every solve would take PyTorch's own rule: the silent
    second derivative this routing exists for. It is refused by name instead, as a missing call
    is.

    A grad level's grad mode at entry is ``CGradInterpreterPtr(interpreter).prevGradMode()``, the
    call ``torch._functorch.pyfunctorch.GradInterpreter.prev_grad_mode`` makes. ``torch.func``
    turns grad mode on inside every grad level, so the mode inside says nothing; the mode it was
    entered with is the one ``torch.func.vjp``, ``jacrev`` and ``hessian`` run their backward in
    (``create_graph`` defaults to it), and with it off that backward reuses the factors."""
    import sys
    torch = sys.modules['torch']
    if not _functorch_active(torch):
        return {}, [], set()
    functorch = _functorch(torch, _LEVEL_EXTRA)
    kinds = functorch.TransformType
    members = getattr(kinds, '__members__', {})
    modes, grad_off = {}, set()
    for interpreter in functorch.get_interpreter_stack() or ():
        key = interpreter.key()
        if not ('Jvp' in members and 'Vmap' in members and 'Grad' in members
                and key in members.values()):
            raise RuntimeError(
                f'wavebit reads which torch.func transform owns each level with '
                f'torch._C._functorch.get_interpreter_stack, whose interpreter key() PyTorch '
                f'{torch.__version__} returns as {key!r}, not a member of '
                f'torch._C._functorch.TransformType with Grad, Jvp and Vmap, so a derivative '
                f'through this transform cannot be taken correctly; use torch.autograd outside '
                f'torch.func')
        modes[interpreter.level()] = key
        if key == kinds.Grad and not functorch.CGradInterpreterPtr(interpreter).prevGradMode():
            grad_off.add(interpreter.level())
    vmap = kinds.Vmap
    return modes, sorted(level for level, key in modes.items() if key == vmap), grad_off


def _forward_inside(levels, modes, grad_off, within):
    """Whether a level whose rule holds the factors fixed differentiates the solve with another
    level outside it, or a ``forward_ad`` dual made inside a grad level (``within``, from
    :func:`_duals_within_grad`) has that level differentiate its forward rule.

    PyTorch's rule is exact wherever every level but the outermost is reverse mode run with
    grad mode on: each re-solves, and the outermost level's rule is never differentiated. So
    the question is whether any level *above the outermost* is forward mode, or a grad level
    entered with grad mode off (``grad_off``; its backward reuses the factors, see
    :func:`_transform_modes`). Plain autograd and a plain ``forward_ad`` dual sit outside every
    ``torch.func`` level; their order between themselves cannot be told here -- a backward pass
    over a tangent, or a tangent through a backward pass that may run with grad mode off -- so
    the two together count as yes. A ``torch.func`` level's mode is its interpreter's
    (``TransformType.Jvp`` for ``jvp`` and ``jacfwd``); a level whose transform has exited
    differentiates nothing and counts as no. ``torch.func.grad`` itself always re-solves, but
    its interpreter cannot be told from ``vjp``'s, so under ``torch.no_grad()`` it is rewritten
    too: exact either way.
    """
    if within:
        return True
    if len(levels) < 2:
        return False
    if {'requires_grad', 'dual'} <= levels:
        return True
    numbered = sorted(level for level in levels if isinstance(level, int))
    inside = numbered if len(numbered) < len(levels) else numbered[1:]
    return any(_is_forward(modes.get(level)) or level in grad_off for level in inside)


def _forward_under_two_vmaps(levels, modes, vmaps, grad_off, within):
    """Whether a rule that calls ``lu_solve`` on the factors runs it beneath two ``vmap`` levels,
    where it can see factors batched by the outer one and a right side by the inner one (defect
    2 in the note above): a ``torch.func`` forward level with two ``vmap`` levels numbered below
    it, or a grad level that reuses the factors (``grad_off``, or one holding a dual made inside
    it, ``within``) with one -- its backward adds the second, ``jacrev``'s own over the basis.
    ``vmap(grad)`` and ``vmap(vjp)`` under ``torch.no_grad()`` are rewritten too, exact either
    way: their levels cannot be told from ``jacrev``'s here. ``jacfwd(vmap(f))`` has one ``vmap``
    beneath its ``jvp`` and is exact as PyTorch takes it."""
    for level in levels:
        if not isinstance(level, int):
            continue
        below = sum(v < level for v in vmaps)
        if below >= 2 and _is_forward(modes.get(level)):
            return True
        if below >= 1 and (level in grad_off or level in within):
            return True
    return False


def _guard_the_backward(solved, matrix, levels, modes, vmaps):
    """Refuse, from inside the backward pass, a grad level's backward through ``solved`` that
    runs with grad mode off where :func:`linear_solve` read the level on, and would be wrong.

    :func:`linear_solve` reads a grad level by the grad mode it was entered with. A
    ``vjp_fn`` that ``torch.func.vjp`` returns may yet be called in another -- under
    ``torch.no_grad()``, or with ``create_graph=False`` -- and PyTorch's backward then reuses
    the factors (defect 1 in the note above): a ``jvp`` of it read 0.57 relative off on the
    closed form, with no error. That backward has not run when the solve is taken, so it cannot
    be routed; it is refused when it runs. It is wrong only where something differentiates it
    again -- a forward level outside the grad level that moves the matrix (``jvp``,
    ``jacfwd``, a plain ``forward_ad`` dual) -- or where it runs under a ``vmap`` added after
    the level, beside one under it, which batches the factors (defect 2). So a hook goes on
    ``solved``'s wrapper at each such grad level, and asks when that level's backward reaches
    it. It returns nothing: every value, and every backward run with grad mode on, is unchanged.
    """
    import sys
    torch = sys.modules['torch']
    functorch = _functorch(torch, _LEVEL_EXTRA)
    grad = functorch.TransformType.Grad
    forward = [level for level in matrix if isinstance(level, int) and _is_forward(modes.get(level))]
    for level in levels:
        if not isinstance(level, int) or modes.get(level) != grad:
            continue
        # A level entered with grad mode off never gets here with either reason: inside another
        # level, or above a vmap level, :func:`linear_solve` has rewritten the solve already.
        outside = 'dual' in matrix or any(f < level for f in forward)
        below = sum(v < level for v in vmaps)
        if not (outside or below):
            continue
        wrapper = solved
        while functorch.is_functorch_wrapped_tensor(wrapper) and functorch.maybe_get_level(wrapper) > level:
            wrapper = functorch.get_unwrapped(wrapper)
        if functorch.maybe_get_level(wrapper) == level and wrapper.requires_grad:
            wrapper.register_hook(_grad_mode_check(torch, functorch, outside, below))


def _grad_mode_check(torch, functorch, outside, below):
    """The hook :func:`_guard_the_backward` registers: raise if the backward runs with grad mode
    off and is differentiated again (``outside``) or batched by a ``vmap`` the level did not see
    (more ``vmap`` levels on the stack than the ``below`` under it)."""
    def check(gradient):
        if torch.is_grad_enabled():
            return None
        vmap = functorch.TransformType.Vmap
        if outside:
            reason = ('a forward-mode level outside it (torch.func.jvp, jacfwd or a forward_ad dual) '
                      'differentiates that backward again')
        elif sum(i.key() == vmap for i in functorch.get_interpreter_stack() or ()) > below:
            reason = ('it runs under a torch.func.vmap added after the grad level, beside one under '
                      'it, where lu_solve batches the factors wrongly')
        else:
            return None
        raise RuntimeError(
            f'wavebit: the backward pass of a torch.func grad level entered with grad mode on '
            f'(torch.func.vjp) reaches a linear solve with grad mode off -- a vjp_fn called under '
            f'torch.no_grad() or with create_graph=False -- and {reason}; PyTorch '
            f'{torch.__version__} then reuses LU factors it does not differentiate, so the '
            f'derivative would be wrong with no error. Call the vjp_fn in the grad mode '
            f'torch.func.vjp was entered with, or enter torch.func.vjp under torch.no_grad() too')
    return check


def _is_forward(mode):
    import sys
    return mode is not None and mode == sys.modules['torch']._C._functorch.TransformType.Jvp


def _derivative_levels(value):
    """The PyTorch autodiff levels that differentiate ``value``: each ``torch.func`` grad or jvp
    level, by its wrapper's level number (an outer ``jvp`` level's tangent cannot be read from
    inside an inner one, so per-level flags would count one where there are two; its wrapper is
    there), plus ``'requires_grad'`` and ``'dual'`` for plain autograd and a plain ``forward_ad``
    dual under them, the dual read at its own level (:func:`_own_tangent`). Empty for anything
    that is not a PyTorch tensor, without importing it."""
    import sys
    torch = sys.modules.get('torch')
    if torch is None or not isinstance(value, torch.Tensor):
        return set()
    levels = set()
    from torch.autograd import forward_ad
    if _functorch_active(torch):
        functorch = _functorch(torch, _LEVEL_EXTRA)
        while functorch.is_functorch_wrapped_tensor(value):
            if functorch.is_gradtrackingtensor(value):
                levels.add(functorch.maybe_get_level(value))
            value = functorch.get_unwrapped(value)
        # Beneath a grad or jvp interpreter a plain read lifts the base and finds no tangent, so
        # it is read at its own level: read plainly, a dual and `requires_grad` outside
        # `torch.func.grad` counted one plain level where there are two, and a third derivative
        # -- a backward pass over a `forward_ad` tangent of it, or the reverse -- came back 0.34
        # to 0.52 relative off (48 of the census's 784 nests, per dtype).
        dual = _own_tangent(functorch, forward_ad, value)
    else:
        dual = forward_ad.unpack_dual(value).tangent is not None
    if value.requires_grad:
        levels.add('requires_grad')
    if dual:
        levels.add('dual')
    return levels


def _duals_within_grad(values, modes):
    """The ``torch.func`` grad levels at which ``values`` carry a ``forward_ad`` tangent of their
    own: a dual made inside ``torch.func.grad``, ``vjp`` or ``jacrev``.

    Such a tangent sits on that level's wrapper (read at its own level, :func:`_own_tangent`),
    not on the base :func:`_derivative_levels` reads, where a dual made outside the transform
    sits. The forward rule runs inside the level, which then differentiates it -- forward inside
    reverse, within one level -- and read as the one grad level it looks like, ``torch.func.grad``
    of a tangent came back 0.57 relative off on the closed form, and 760.746214 for 812.595272 on
    a 2D ETM raster's thickness, silently. A jvp level's wrapper holds that level's own tangent,
    which is no dual of this kind, so only grad levels are read; read by type, a single ``jvp``
    would be rewritten, and its bytes would change."""
    import sys
    torch = sys.modules['torch']
    functorch = _functorch(torch, _LEVEL_EXTRA)
    grad = {level for level, key in modes.items() if key == functorch.TransformType.Grad}
    found = set()
    if not grad:
        return found
    from torch.autograd import forward_ad
    for value in values:
        if not isinstance(value, torch.Tensor):
            continue
        while functorch.is_functorch_wrapped_tensor(value):
            if functorch.is_gradtrackingtensor(value):
                level = functorch.maybe_get_level(value)
                if level in grad and level not in found and _own_tangent(functorch, forward_ad, value):
                    found.add(level)
            value = functorch.get_unwrapped(value)
    return found


def _own_tangent(functorch, forward_ad, x):
    """Whether ``x``, one level of a wrapper chain inside a ``torch.func`` transform, carries a
    forward-mode tangent at its own level.

    ``unpack_dual`` is an operator, dispatched at the top of the interpreter stack, and a grad or
    jvp interpreter there lifts a tensor of a level below it into a fresh wrapper of its own,
    which has no tangent. Read plainly, a jvp level beneath a ``torch.func.grad`` or another
    jvp reads None, and so does a ``forward_ad`` dual made outside ``torch.func.grad``: measured
    on toy tensors, and on mixed partials -- ``jacfwd`` or ``jvp`` over a permittivity of
    ``torch.func.grad`` or ``jvp`` over a thickness, or a ``forward_ad`` dual permittivity
    outside ``torch.func.grad`` -- that took the constant branch and returned 0.15 relative off
    on the 1D TM SMM scene, silently (806661e4 and before). So the interpreters above ``x``'s
    level are popped for the read and pushed back after it, as PyTorch's
    ``torch._functorch.pyfunctorch.temporarily_pop_interpreter_stack`` and
    ``temporarily_clear_interpreter_stack`` do (the latter's pop loop and reverse push, stopped
    at ``x``'s level); a level the stack
    tops already is read with nothing popped. The calls are private (:data:`_FUNCTORCH_CALLS`):
    should popping stop moving the dispatch, the reads would go back to None -- the silent
    mixed partials -- which ``tests/rcwa/test_torch_vmap_autograd.py`` trips on.
    """
    level = functorch.maybe_get_level(x)
    popped = []
    try:
        while True:
            top = functorch.maybe_current_level()
            if top is None or top <= level:
                break
            popped.append(functorch.pop_dynamic_layer_stack())
        return forward_ad.unpack_dual(x).tangent is not None
    finally:
        while popped:
            functorch.push_dynamic_layer_stack(popped.pop())


def constant_array(value, dtype=float):
    """``np.asarray(value, dtype=dtype)``, refusing a value that carries a derivative.

    The test this package spells as ``try: np.asarray(v) ... except: <the traced path>`` relies
    on the conversion refusing a value under autodiff. A JAX tracer and a PyTorch tensor with
    ``requires_grad`` both refuse; a ``torch.autograd.forward_ad`` dual does not -- it converts
    to its primal, the tangent dropped -- so that test sent a dual down the constant path with
    no error. A ``period`` given as a dual became a NumPy lattice, and the field's sample grid
    on it lost its derivative: 0.55% off reverse mode in a weighted read-out of the field of a
    2D grating, 20% in one of its samples. This raises ``TypeError`` for a dual, in a list or
    tuple as well, so every such test sends it where reverse mode goes.
    """
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        elif forward_tangent(item):
            raise TypeError('a forward-mode dual is not a constant')
    return _np.asarray(value, dtype=dtype)


def principal_sqrt(xp, z):
    """``xp.sqrt(z)`` of a complex array, on the lip a signed zero names -- on every backend.

    The principal root's cut is the negative real axis, and a value on it sits on the lip the
    sign of its zero imaginary part names (W. Kahan, "Branch cuts for complex elementary
    functions, or Much ado about nothing's sign bit", in *The State of the Art in Numerical
    Analysis*, eds. A. Iserles and M. J. D. Powell, Clarendon Press (1987); ISO/IEC 9899:1999
    section 7.3.3 and Annex G): ``sqrt(complex(-4, -0.0))`` is ``-2j``,
    ``sqrt(complex(-4, +0.0))`` is ``+2j``. NumPy,
    PyTorch and :mod:`cmath` do; ``jax.numpy.sqrt`` (measured on jax and jaxlib 0.11.0,
    complex64 and complex128, eager and under ``jit``) returns ``+2j`` for both, so a lossless
    value written with ``-0.0`` got the other root on JAX alone. Where it does, the root is
    negated -- as ``0 - w``, which keeps the real zero ``+0.0`` and so NumPy's bits, and not as
    ``conj(w)``, so that its derivative is the chosen root's own, ``1/(2*(-w))`` -- and every
    other value is ``jnp.sqrt``'s to the bit. NumPy and PyTorch are handed back ``xp.sqrt``
    itself, untouched.
    """
    root = xp.sqrt(z)
    if 'jax' not in xp.__name__:
        return root
    im = xp.imag(z)
    other_lip = (xp.real(z) < 0) & (im == 0) & xp.signbit(im) & (xp.imag(root) > 0)
    return xp.where(other_lip, 0.0 - root, root)


def in_namespace(xp, v):
    """Whether ``v`` is an array of ``xp`` itself -- an array, a tensor or a tracer of it.

    The one test for "may carry a graph, pass it through" against "a constant from elsewhere,
    convert it", read by :func:`as_array`, :func:`_readable` and the vertex builders'
    ``polygon._scalar``. Having a ``dtype`` is not the test: a NumPy scalar has one, and on
    JAX or Torch it is as much a constant as a Python float -- ``torch.cos`` refuses both.
    """
    if not (hasattr(v, 'dtype') and hasattr(v, 'shape')):
        return False
    try:
        return ac.array_namespace(v) is xp
    except Exception:                          # noqa: BLE001 - no namespace recognises it
        return False


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
    if in_namespace(xp, v):
        return v
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

    **Why it matters that this is not `float`.** :func:`wavebit._common.geometry.overlap.primitive` spelled
    it ``float(cx)``, and :func:`wavebit.rcwa._engine.parametric._primitives` reads the failure as "this
    is traced -- skip the check". A grad tracer refuses ``float()`` while answering
    ``stop_gradient`` and ``bool(x > 300)`` perfectly, so it was saying "this spelling carries a
    tangent", not "there is no value here" -- and the geometry refusals were skipped under
    exactly the gradient loop they exist to protect. A rod grown by plain gradient ascent from
    ``lx = 280`` on a period of 300 ran to completion at **lx = 911.71**, three times the
    period, with ``R + T = 1.000000000000`` at every step; the same run on torch died at step 1,
    because torch tensors satisfy ``float()``.

    ``detach`` first also silences the ``UserWarning`` torch emits for a ``requires_grad``
    scalar conversion, which the checks were printing on every eager torch shape solve.

    Deliberately **not** :func:`wavebit.rcwa._engine.parametric._static_float`, which refuses an autograd
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

    ``None`` means the value is *abstract* — a ``jax.jit`` tracer — where there is genuinely no
    number to look at. That, and not the namespace, is what the guards below are keyed on.
    Keying on the namespace was the bug: an ordinary eager ``backend='jax'`` /
    ``backend='torch'`` call has its values right there, and was skipped anyway.

    PyTorch does not hand an abstract value over under ``torch.compile``: wavebit runs eagerly
    inside a compiled caller (:func:`eager_under_torch_compile`), so the value here is concrete.
    A non-strict ``torch.export`` and ``torch.jit.trace`` are refused at the door before they
    get here: a ``torch.jit.trace`` value is concrete too, and every branch this read decides
    would be recorded as a constant.

    ``jax.grad`` sits on the *checked* side of the line, which is the whole point. An
    un-jitted grad tracer carries a concrete primal, so the reduction narrows to a real
    boolean and the guard runs — and an un-jitted gradient loop is precisely where a
    silently wrong solve does the most damage. ``jax.jit``, and ``jit(grad)``, refuse the
    read and are skipped. Measured rather than assumed; the same asymmetry is tabulated in
    ``docs/validation/rcwa/backend-choice.md`` §3 and already relied on by
    :func:`wavebit.rcwa._engine.solve_2d_first_order._forward_count` and
    :func:`wavebit._common.conventions._check_resolvable`.

    The reduction itself runs in the caller's own namespace, so what crosses the boundary is
    one boolean rather than a copy of the array — on an accelerator that is a single sync,
    not a device-to-host transfer.

    Catching broadly rather than naming the exception keeps this backend-agnostic, the same
    trade :func:`wavebit.rcwa._engine.engine._concrete_period` makes.
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
    (see :func:`~wavebit.rcwa._engine.solve_1d.layer_modes_1d_conical`), those re-rolls returned
    quiet wrong answers scattered from 0.18 to 626 against a true 0.215, re-slicing among
    them (1/2/4/8/20 slices: 11.9, 3.6, 1.8, 27.7, 110.2); on the 2D path they still do
    (see :func:`~wavebit.rcwa._engine.solve_2d.layer_modes_2d`). Detuning moves ``kx`` itself, which
    is the only thing that does anything, and one ULP of wavelength, period or angle is
    already enough — measured to agree with the exact-point answer of the reformulated paths
    to eight digits.

    That agreement is a ``ky == 0`` statement. On the 1D conical path with ``ky != 0`` the same
    ``kx_m == n`` is an x-cutoff (``kz = |ky|``), where a nearby point is *not* a remedy: the
    family construction is wrong over a band, 2e-2 one ULP off and still 5e-5 at a relative
    1e-8. That layer is now rebuilt in a regular basis that is exact at the point
    (:func:`~wavebit.rcwa._engine.solve_1d._regular_conical_modes`), so an eager solve does not reach
    this check there -- unless the rebuild's gate refuses the cluster (a root under 1e-9, which
    takes ``|ky|`` under about 1.4e-9, or a pair with opposite labels), where an exact zero still
    lands here as before. Under an abstract trace (``jax.jit``, ``jax.vmap``, a ``jax.lax.map``
    or ``scan`` body, ``jax.checkpoint`` -- under ``jax.grad`` too -- and ``torch.func.vmap``)
    the layer is rebuilt as well, its masks carried as data (``docs/theory/rcwa/rayleigh-points.md``
    section 7); only a cluster the gate refuses keeps the old construction's arithmetic there,
    NaN at an exact zero, which this check cannot see under a trace.
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
            'digits (a 1D conical order with ky != 0 at kx_m == n is not a graze, and its '
            'layer is normally rebuilt exactly -- it lands here only where that rebuild '
            'declines, as at |ky| below about 1.4e-9); '
            'solve(auto_detune=True), the default, does that '
            'retry for you, so seeing this means it was disabled or the retry failed too. '
            'Do NOT reach for '
            'truncation_order or thinner slices instead: neither knob touches the degeneracy (it is '
            'fixed by kx against an index), so at best they re-roll the round-off — '
            'measured to return quiet wrong answers on the paths that have not been '
            'reformulated against this. See layer_modes_1d_conical.')


class UnresolvedBranchWarning(UserWarning):
    """A ``sqrt`` branch was chosen from a value that carries no information about it.

    Choosing a root reads the sign of a value: ``Re(kz) + Im(kz)`` for a half-space and
    ``Re(q) - Im(q)`` for a layer mode under ETM. A sign is answerable only while the value
    stands clear of the working precision's round-off. Both are at least ``|kz|`` (``|q|``) at
    every real ``k_par`` in a passive medium, so this fires only where ``kz**2`` or ``q**2`` sits
    within round-off of its cut -- far more often at complex64 than at complex128, where it has
    not been seen outside a constructed case.

    Until 2026-10-02 both decisions read a real part alone, which at complex64 *is* round-off for
    an exactly propagating or evanescent wave: a single-precision solve returned ``|R| ~ 4`` from
    the half-space rule, and ETM lost 5e-2 to its layer labels. SMM's layer labels still read
    ``Re(q)`` alone, but SMM does not depend on them, so they are not checked. Its own category
    so it can be escalated or silenced on its own; see ``docs/theory/rcwa/branch-cuts.md``.
    """


class PrecisionLossWarning(UserWarning):
    """A user-supplied value arrived narrower than the precision the solve was asked for.

    Widening it is not the same as having had it: the digits are already gone. Its own
    category so it can be silenced by a caller who is passing single-precision data on purpose.
    """


class ExceptionalPointWarning(UserWarning):
    """A 2D or first-order layer retains an unresolved defective mode cluster.

    Public path: ``wavebit.rcwa.diagnostics.ExceptionalPointWarning``.

    Isolated exceptional-point pairs are rebuilt in a regular basis. An eager screen of each
    layer's final mode basis warns when it still contains a nearly dependent cluster with a
    non-normal restriction of the layer operator. Ordinary diagonalizable degeneracies do not
    trigger it.

    The result may be inaccurate. Detuning, ``rcwa.Numerics(detuning=...)``, does not act on
    this warning: its Rayleigh detuning ladder is too small to leave a coincident
    exceptional-point band. The screen cannot run under an abstract trace (``jax.jit`` or
    ``jax.vmap``), where the unresolved cluster remains silent; see
    ``docs/theory/rcwa/rayleigh-points.md`` for the measured limits.
    """


class RayleighPointWarning(UserWarning):
    """A layer eigenmode sits numerically on its light line, on a path that loses it there.

    Public path: ``wavebit.rcwa.diagnostics.RayleighPointWarning``.

    Fired by the 2D mode builder, and since the coupled-cluster rebuild of grazing modes only
    on its fallback paths: a grazing cluster whose E- and H-side sizes disagree, or
    eigenvalues that are concrete but not materializable (a tensor carrying grad, an
    un-jitted ``jax.grad`` tracer). There the earlier product construction is what runs, and
    the result it annotates is finite, in range, and possibly wrong with no other sign. The
    remedy is to move off the condition: one ULP of wavelength, period or angle is enough.
    ``rcwa.Numerics(detuning=...)`` makes that retry and reports it with
    :class:`AutoDetuneWarning`; it is off by default (``detuning=None``). Its own category so
    a sweep that knowingly steps across a Rayleigh anomaly can silence or escalate it on its
    own. See ``docs/theory/rcwa/rayleigh-points.md``.
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

    Public path: ``wavebit.rcwa.diagnostics.ComplexIncidenceWarning``.

    Only a complex ``theta`` / ``phi`` reaches one -- a direct ``kx`` / ``ky`` must be real --
    through the formal ``kx/k0 = n*sin(theta)*cos(phi)`` parameterization. Such a wave's
    intensity varies along the surface, and in the exit medium no root of ``kz`` both carries
    power away and decays away, so which one a solver takes is a convention: measured on one
    interface (top 4+3i, bot 2.25), fmmax's ``Im(kz) > 0`` gave ``r = 1.727-1.966i``, where
    this solver gave ``0.252+0.287i``. So did RETICOLO, whose Maystre cut
    (``Re(kz) + Im(kz) > 0``) this solver has taken since 2026-10-02;
    see ``docs/validation/rcwa/time-convention-audit.md`` section 9. The per-order ``R`` and ``T``
    are then amplitude-derived ratios rather than fractions of an incident power. Their sum
    can exceed 1 on a lossless stack. An order whose root decays with ``Re(kz) < 0`` gives a
    negative ratio. ``power_entering`` and ``absorption`` are NaN.

    Evanescent incidence (``theta = pi/2 + 1j*alpha``) keeps ``kx`` real and does not fire.
    From a lossless medium its incident wave carries no normal power, so there is no ratio,
    formal or not, to report: ``R``, ``T`` and ``power_entering`` are NaN instead
    (:func:`~wavebit.observations.power.carries_normal_power`, #84). No warning is added for
    it. This one exists because R and T stay finite and read like efficiencies; a NaN marks
    itself, and is marked under ``jax.jit`` too, where no warning can run.
    For an absorbing incident medium, a real ``theta`` is the physical input: a real angle
    keeps ``kx`` real and puts the loss in ``kz``. Its own category so a study that means the
    formal parameterization can silence it without silencing anything else.
    """


class GainCutoffWarning(UserWarning):
    """An order of a gain half-space sits near grazing, where no plane-wave value is the answer.

    Public path: ``wavebit.rcwa.diagnostics.GainCutoffWarning``.

    A half-space with gain (``Im(eps) < 0`` or ``Im(mu) < 0``) takes Maystre's root, which is
    the root of switching the gain on from zero at this frequency
    (:func:`~wavebit._common.conventions.boundary_kz`). Away from grazing a smooth (Gaussian)
    beam a few gain lengths wide converges to that value, whatever the gain's dispersion, as
    long as the medium's branch point crosses the real in-plane wavevector axis once as
    ``Im(w)`` falls to zero (every stable Lorentz line studied does). Near
    grazing it does not: light skimming the interface is amplified along it, so a beam's
    response depends on its width, and close enough to grazing it grows with the width without
    bound. A sharp-edged beam seeds such a growing wave at its edges at any angle, which no
    per-order band can flag. The band
    is ``|Re(kz**2)| < 2*|Im(kz**2)|`` per order, which for weak gain is an in-plane
    wavevector within about ``2*k0*|Im(n)|`` of grazing. It holds the window where a Gaussian
    beam's response grows with its width (a ratio below about 1) and the zone beyond it where a
    beam a few gain lengths wide has not yet converged; ``docs/theory/rcwa/branch-cuts.md``
    ("Gain near grazing") gives the measurements behind it.

    A warning rather than a refusal: the value is the exact answer to a stated problem, the
    band has no provable edge, and a refusal would stop a sweep and NaN a traced solve, which
    is the history of the Rayleigh-band refusals (``docs/theory/rcwa/rayleigh-points.md``). Constant text,
    so the default filter shows it once per sweep. Not a ``rcwa.Numerics(detuning=...)``
    trigger: a one-ULP nudge cannot leave a band this wide. A traced run cannot check and
    stays silent. ``rcwa.operators.ScatteringMatrix.cascade`` does not re-check a joined
    stack: each piece was checked by its own run, against its own half-spaces. A
    coherent-source run applies the same test on its own ports for the sectors a
    ``PlaneWave``, ``GaussianBeam`` or ``IncidentOrders`` lights, once per side per run. It
    skips dipole sources: a point source is sharp-edged, the case the band does not describe.
    """


class AutoDetuneWarning(UserWarning):
    """A detuning retry moved the run off the requested point — this is the record.

    Public path: ``wavebit.rcwa.diagnostics.AutoDetuneWarning``.

    Fired once per run with ``rcwa.Numerics(detuning=...)`` that actually retried; detuning
    is off by default (``detuning=None``). The message carries the relative step and the
    wavelength the returned result was evaluated at, and the result keeps that wavelength
    (``Result.diagnostics['effective_wavelengths']``) so everything downstream, field
    reconstruction included, describes one consistent problem. Its own category so a sweep can
    silence the notices while keeping :class:`RayleighPointWarning` escalated, or the reverse.
    What changed is always said; nothing is moved silently.
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

_RAYLEIGH_MESSAGE = (
    'a layer eigenmode sits within round-off of its cutoff (a Rayleigh/grazing '
    'condition) on the 2D path, which loses that mode to cancellation — the result '
    'is finite and may be wrong at the 1e-5 level near the condition, growing to '
    'order 1 exactly on it, with no other sign. Move off the condition: one ULP of '
    'wavelength, period or angle is enough, and solve(auto_detune=True), the '
    'default, does that retry for you. (The 1D paths solve this point exactly -- with '
    'ky != 0, where kx_m == n is an x-cutoff rather than a graze, by rebuilding that '
    'layer in a regular basis -- so a y-invariant cell routed through the 1D solves '
    'does not need the detune.)')

#: Set by ``solve``'s auto_detune ladder around each solve it runs; :func:`check_mode_cutoff`
#: reports a hit here instead of warning, and the ladder decides what the caller sees.
#:
#: A direct channel rather than the warnings machinery, because the ladder used to catch every
#: warning of every solve (``catch_warnings(record=True)``) to look for this one and then replay
#: the rest. ``catch_warnings`` bumps the filters version on entry and exit, which empties every
#: module's ``__warningregistry__``, so a warning with constant text -- written that way so the
#: default filter shows it once per sweep -- came back once per sweep point on the default
#: ``solve()``: measured, 5 reports for 5 points where ``auto_detune=False`` gave 1. Passing a
#: persistent registry to ``warn_explicit`` does not help (the version bump clears it too). With
#: this channel the ladder catches nothing, and every other warning reaches the caller's
#: filters as the solve emitted it.
_RAYLEIGH_LISTENER = contextvars.ContextVar('_RAYLEIGH_LISTENER', default=None)


def division_at_a_cutoff():
    """Silence NumPy's divide/invalid flags where a layer's mode eigenvalue divides.

    That divisor is exactly zero where an order meets the layer's light line (a Rayleigh point;
    in conical incidence also where ``kx**2`` alone meets it, an x-cutoff), and what the zero
    makes is handled downstream: the 2D coupled-cluster rebuild replaces the columns
    (:func:`wavebit.rcwa._engine.solve_2d._rebuild_grazing_cluster`), or :func:`check_finite` refuses
    the NaN with a structural hint. A conical x-cutoff (``ky != 0``) is rebuilt before these
    blocks run (:func:`wavebit.rcwa._engine.solve_1d._regular_conical_modes`), whose divisors are
    guarded instead -- under an abstract trace (``jax.jit``, ``jax.vmap``, a ``jax.lax.map`` or
    ``scan`` body, ``jax.checkpoint``, ``torch.func.vmap``) too, where every conical layer is
    rebuilt; it reaches these blocks only where its gate refuses the cluster. The block covered
    includes the products that carry an ``inf`` quotient into NaN. NumPy's own
    ``RuntimeWarning`` adds nothing to either outcome.
    It did harm: from an attempt the ``auto_detune`` ladder discarded it reached the caller,
    and under a caller's ``simplefilter('error', RuntimeWarning)`` it raised before the ladder
    could retry (``tests/rcwa/test_rayleigh_guard.py::test_auto_detune_retries_under_an_error_filter``).

    Only ``divide`` and ``invalid``, and only around those blocks; ``over`` and ``under``
    keep the caller's policy. A caller's ``np.seterr(divide='raise')`` is overridden there too,
    which trades a ``FloatingPointError`` at the division for the refusal or the rebuild.
    NumPy only: JAX and PyTorch raise no floating-point warnings.
    """
    return _np.errstate(divide='ignore', invalid='ignore')


def check_mode_cutoff(xp, eigenvalues):
    """Warn when a layer eigenvalue is at round-off scale relative to the spectrum.

    In this regime the 2D product construction ``V = (Omega_R @ W) / q`` no longer
    represents any nearby problem — the grazing cluster's product eigvecs are eps-unstable
    against the factor split, and the answer can be quietly wrong (measured: 5 of 48
    exact-graze 2D solves returned R+T of 0.96-0.99 on a lossless stack with nothing else
    fired). Since the coupled-cluster rebuild
    (:func:`wavebit.rcwa._engine.solve_2d._rebuild_grazing_cluster`) the 2D builder handles the
    regime itself — exactly, or with a loud refusal — and reaches this check only on its
    fallback paths, where the old construction is what ships; the warning is then what
    drives ``solve``'s default detune ladder. The 1D paths do not need this check: the
    pure path never had the disease and the conical path was reformulated (see
    ``layer_modes_1d_conical``) -- at ``ky == 0`` by building its blocks from the
    eigensystem, and at ``ky != 0``, where ``kx_m == n`` is an x-cutoff and the family basis
    itself collapses, by rebuilding that layer in a regular basis
    (``solve_1d._regular_conical_modes``).

    A warning rather than an error, deliberately: the error scales as ``~eps/ratio`` (see
    ``_CUTOFF_REL``), so most of the flagged band is wrong at the 1e-5 level rather than
    garbage, and a refusal would take working-precision answers away from a sweep that merely
    steps near the condition. Runs where the eigenvalues are concrete; traced solves skip it,
    like every other guard here.
    """
    aw = xp.abs(_readable(eigenvalues, xp))
    flag = _concrete(xp.min(aw) < _CUTOFF_REL * xp.max(aw))
    if flag:
        listener = _RAYLEIGH_LISTENER.get()
        if listener is not None:
            listener.append(_RAYLEIGH_MESSAGE)      # the ladder retries; it says what stands
            return
        import warnings
        warnings.warn(_RAYLEIGH_MESSAGE, RayleighPointWarning, stacklevel=3)


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
        wavebit.rcwa(backend='jax', theta=theta, complex_dtype=0, ...)   # asks for double

    ``jax.config.update("jax_enable_x64", True)`` before the second line is the fix.

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
        'call `jax.config.update("jax_enable_x64", True)` before building it. '
        '`import wavebit` leaves that setting unchanged; a later precision change cannot '
        'restore an array made before it.',
        PrecisionLossWarning, stacklevel=3)


def forward_strided(v):
    """``v`` with no negative stride, copying only a NumPy view that has one.

    ``np.rot90``, ``np.flip`` and ``a[::-1]`` all return views with a negative stride, and
    PyTorch refuses those outright: ``ValueError: At least one stride in the given numpy array
    is negative``. NumPy and JAX take them without complaint, so the natural spelling of "solve
    the rotated cell" is a crash on ``backend='torch'`` and a result on the other two — a backend
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
    NumPy view. That half is current: on torch 2.14.1 ``asarray(np.flip(a))`` still raises
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
    one; measured on torch 2.13.0 and again on 2.14.1 (the floor is 2.14, see
    ``docs/development/torch-compile.md``), every ``asarray`` path now preserves both ``requires_grad`` and the graph, so on gradients
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
    if in_namespace(xp, v):
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
# The gauge term (the third one in `rhs`) is theirs too. It is kept here on their
# authority rather than on evidence: for a
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
        # The Function exists only to carry a derivative. Every solve hands this a detached
        # operator -- the modal rule (`modal_gradient`) carries the layer's derivative instead --
        # so where there is none to carry it is not entered, with the same values. That keeps a
        # `torch.func` transform, which refuses any `autograd.Function` without a functorch
        # rule, from dying here with PyTorch's `setup_context` error before it reaches the rule
        # that names its limit.
        #
        # Liveness is asked of `A` as handed in -- its own `requires_grad`, and a forward-mode
        # tangent, which `forward_tangent` reads under a batched level since `unpack_dual` cannot
        # read one -- and not through `carries_autodiff`, which also asks the `requires_grad` of
        # every level under a `torch.func` wrapper. That is the answer `carries_autodiff` gave
        # here for every eager value. Under `torch.func.vmap` an operator whose batched tensor
        # requires grad keeps going to the native `torch.linalg.eig`, as it did, and one that
        # carries a `forward_ad` tangent reaches `_Eig`, which stays an old-style Function with
        # no functorch rule, so PyTorch refuses it by its `setup_context` error. No solve hands
        # this either: each one passes the operator through `modal_gradient._stop`.
        live = getattr(A, 'requires_grad', False) or forward_tangent(A)
        return _torch_eig(A) if live else _eig_raw(A)
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


@eager_under_torch_compile
def set_j1_implementation(name):
    """Choose which ``J_1`` the traced backends use. Returns the previous setting.

    Only PyTorch has a real choice here; see ``docs/validation/rcwa/bessel-j1.md`` for the measurements.

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
        ``torch.special.bessel_j1``, wrapped for the derivative it does not ship -- a ``backward``
        and a ``jvp``, and a functorch rule for ``torch.func``.
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


@eager_under_torch_compile
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
                # `forward` without `ctx` and a `setup_context`, so `torch.func` transforms take
                # the rule too; elementwise, so their `vmap` rule is generated.
                generate_vmap_rule = True

                @staticmethod
                def forward(x):
                    return torch.special.bessel_j1(x)

                @staticmethod
                def setup_context(ctx, inputs, output):
                    (x,) = inputs
                    ctx.save_for_backward(x)
                    ctx.save_for_forward(x)

                @staticmethod
                def backward(ctx, grad_out):
                    (x,) = ctx.saved_tensors
                    return grad_out * _torch_j1_derivative(x)

                @staticmethod
                def jvp(ctx, tangent):
                    # Forward mode: `torch.autograd.forward_ad` and `torch.func.jvp`. Without it
                    # a dual raised PyTorch's own "jvp is not implemented" error.
                    (x,) = ctx.saved_tensors
                    return tangent * _torch_j1_derivative(x)

            cls._impl = _Impl
        return cls._impl.apply(x)


def _torch_j1_derivative(x):
    """``dJ1/dx = J0(x) - J1(x)/x``, and 1/2 at ``x = 0``."""
    import torch
    safe = torch.where(x == 0, torch.ones_like(x), x)
    return torch.where(x == 0, torch.full_like(x, 0.5),
                       torch.special.bessel_j0(x) - torch.special.bessel_j1(x) / safe)


class _TorchBesselJ1(metaclass=_TorchBesselJ1Meta):
    """``torch.special.bessel_j1`` with the analytic derivative it is missing, in both modes."""


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


@eager_under_torch_compile
def _enable_x64():
    """Explicit precision setup for private numerical regression instruments.

    Public callers configure JAX before creating arrays; importing Wavebit never
    changes their precision. This helper keeps historical engine tests explicit.
    """
    import importlib
    importlib.import_module('jax')
    lift_jax_x64(required=True)
    from wavebit.rcwa._engine.scattering import _register_pytree_if_jax_loaded
    _register_pytree_if_jax_loaded()
