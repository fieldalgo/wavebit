"""Array/device and time-loop policies for the single-source FDTD core.

Use the same Array API namespaces and graph-preserving conversion as RCWA.
Only execution/checkpointing differs here; Maxwell/ADE/CPML equations stay in
wavebit.fdtd._periodic.grid and must not be duplicated by a backend.
"""
import dataclasses
import functools
import threading
import types
import warnings

import numpy as np

from wavebit._common.backend import as_array
from wavebit._common.backend import lift_jax_x64

# Compiled eager scan programs, keyed by structure and scan length; least recently used
# first. See _eager_scan.
_SCAN_PROGRAMS = {}
_SCAN_PROGRAMS_MAX = 64
# Whichever thread solves reads and writes the cache; a `pop` racing an eviction could otherwise
# raise `KeyError`, or change the dict's size under the eviction loop.
_SCAN_LOCK = threading.Lock()
_API_WARNED = []
_FUNCTION_TYPES = (types.FunctionType, types.MethodType, types.BuiltinFunctionType,
                   functools.partial)


@functools.lru_cache(maxsize=None)
def _jax_api():
    """The JAX names the program key reads, or ``None`` when this JAX no longer has one of them.

    The jaxpr types are read off two tiny public traces (``jax.make_jaxpr``) rather than
    imported from ``jax.extend``: from jax 0.12.0, importing ``jax.extend`` adds entries to
    JAX's tracing context, so importing it partway through a session made every compiled
    program miss its cache once -- a user's ``jax.jit`` around the first solve traced twice,
    and the next solve's eager operations all compiled again (measured 2026-10-19). Reading
    the types off traces changes nothing (checked on jax 0.11.0 and 0.12.0: the same classes
    ``jax.extend.core`` names, and an unchanged context). Not all are public promises, so a
    JAX that moves one costs the sharing, never the solve: :func:`_eager_scan` warns once and
    runs the plain ``lax.scan``.
    """
    try:
        import jax
        closed = jax.make_jaxpr(lambda x: x * 2.)(1.)
        literal = closed.jaxpr.eqns[0].invars[1]
        if not hasattr(literal, 'val'):   # a constant that is no longer a literal atom
            return None
        api = types.SimpleNamespace(
            Tracer=jax.core.Tracer, typeof=jax.typeof, eval_jaxpr=jax.core.eval_jaxpr,
            ClosedJaxpr=type(closed), Jaxpr=type(closed.jaxpr), Literal=type(literal))
    except (ImportError, AttributeError, IndexError):
        return None
    api.functions = _FUNCTION_TYPES + _wrapped_function_types(jax)
    return api


def _wrapped_function_types(jax):
    """JAX's wrapped-function class, read off a ``custom_jvp`` trace, or ``()`` if none shows.

    A ``custom_jvp`` equation carries its rule as a ``WrappedFun`` parameter (jax 0.11.0 and
    0.12.0). Missing it costs only sharing: :func:`_param_key` would hash such a rule instead
    of refusing it, which keys every call apart.
    """
    try:
        @jax.custom_jvp
        def probe(x):
            return x
        probe.defjvp(lambda primals, tangents: (primals[0], tangents[0]))
        params = jax.make_jaxpr(probe)(1.).jaxpr.eqns[0].params
    except Exception:
        return ()
    return tuple({type(v) for v in params.values() if type(v).__name__ == 'WrappedFun'})


def _eager_scan(f, init, xs):
    """``jax.lax.scan(f, init, xs)``, compiled once per program instead of once per closure.

    An eager ``lax.scan`` dispatches through ``xla_primitive_callable`` (jax 0.11.0, in JAX's
    dispatch module), whose cache key holds the traced body jaxpr, and a jaxpr
    hashes by identity. ``fdtd_grid`` builds a fresh ``step`` closure on every call, so every
    eager scan traced to a new jaxpr and ran XLA again: 8 compilations on every scene solve,
    which contains 2 distinct programs (4 when it records a field). Here the body is traced
    once, and its closed-over arrays become explicit arguments of one jitted program (closure
    conversion; ``jax.closure_convert`` is JAX's own form of it). The program is keyed on the
    jaxpr's structure (``_program_key``) and on the scan's length: an equal key means the same
    equations, literals, shapes and trip count, so the program runs on this call's arrays
    exactly as a fresh compile would. Python scalars (``dt``, a record offset) are literals in
    the jaxpr, so a changed one is a different key. The length is in the key so that a sweep of
    tail lengths (``checkpoint_steps``) is one entry per length under the LRU bound, rather than
    one entry whose jit grows a compiled copy per length where the bound cannot see it. A cached
    program holds none of the arrays it was first traced with (:func:`_without_consts`).

    Falls back to the plain call whenever sharing could not be exact: inside a trace (nothing
    is dispatched here then), when a closed-over value is a tracer (eager AD), when a carry is
    weakly typed (``lax.scan`` would retrace it promoted), when the key cannot see every value
    the program depends on (``_Unshareable``), and when this JAX lacks a name the key reads
    (warned once).
    """
    import jax
    leaves = jax.tree.leaves((init, xs))
    # Inside a trace nothing is dispatched, so the plain scan is exact: return before the key's
    # names are resolved. Resolving them is a session-wide side effect (it used to import
    # ``jax.extend``, which from jax 0.12.0 made a user's ``jax.jit`` around the first solve
    # trace twice; see :func:`_jax_api`), and a trace is the wrong place for one.
    if any(isinstance(v, jax.core.Tracer) for v in leaves):
        return jax.lax.scan(f, init, xs)
    api = _jax_api()
    if api is None:
        if not _API_WARNED:
            _API_WARNED.append(True)
            warnings.warn(
                'wavebit: this JAX moved a name the eager-scan program cache reads, so every '
                'eager FDTD scan compiles again on each call, as it did before the cache. The '
                'answers are unaffected.', RuntimeWarning, stacklevel=2)
        return jax.lax.scan(f, init, xs)
    lengths = tuple(np.shape(x)[:1] for x in jax.tree.leaves(xs))
    if (not lengths or () in lengths
            or any(isinstance(v, api.Tracer) for v in leaves)):
        return jax.lax.scan(f, init, xs)
    row = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), xs)
    closed, shape = jax.make_jaxpr(f, return_shape=True)(init, row)
    if (any(isinstance(c, api.Tracer) for c in closed.consts)
            or any(api.typeof(v).weak_type for v in leaves)):
        return jax.lax.scan(f, init, xs)
    out_tree = jax.tree.structure(shape)
    try:
        key = (_program_key(closed.jaxpr), jax.tree.structure((init, xs)), out_tree, lengths)
    except _Unshareable:
        return jax.lax.scan(f, init, xs)
    with _SCAN_LOCK:
        program = _SCAN_PROGRAMS.pop(key, None)
        if program is None:
            program = _scan_program(_without_consts(closed.jaxpr), out_tree)
        _SCAN_PROGRAMS[key] = program
        while len(_SCAN_PROGRAMS) > _SCAN_PROGRAMS_MAX:
            del _SCAN_PROGRAMS[next(iter(_SCAN_PROGRAMS))]
    return program(closed.consts, init, xs)


def _without_consts(jaxpr):
    """``jaxpr`` holding no constants of its own; the cached program is handed them per call.

    jax 0.11 made ``ClosedJaxpr`` and ``Jaxpr`` one class, so ``closed.jaxpr`` is ``closed``
    itself, the first call's closed-over arrays included. A cached program that closed over it
    kept those arrays alive on the device -- about 120 KB per program at resolution 20, up to 64
    programs -- although it never reads them: ``eval_jaxpr`` takes the constants it is handed.
    """
    if getattr(jaxpr, 'consts', None):
        return jaxpr.replace(consts=[])
    return jaxpr


def _scan_program(jaxpr, out_tree):
    import jax

    def scan_program(consts, init, xs):
        def body(carry, x):
            flat = jax.core.eval_jaxpr(jaxpr, consts, *jax.tree.leaves((carry, x)))
            return jax.tree.unflatten(out_tree, flat)
        return jax.lax.scan(body, init, xs)
    return jax.jit(scan_program)


class _Unshareable(Exception):
    """A traced body whose program the key cannot see whole; it runs unshared."""


def _param_key(v):
    """One equation parameter as part of a key, or ``_Unshareable`` when the key cannot see it."""
    api = _jax_api()
    if isinstance(v, (api.Jaxpr, api.ClosedJaxpr)):
        # jax 0.11 made the two one class, so a nested *Jaxpr* can carry constants -- an inner
        # `jax.jit` closing over an array puts that array here. The branch that used to check
        # sat below the Jaxpr branch, reachable only on a JAX where they were two classes, and
        # the key then handed a second call the first call's constants: [4, 8, 12] where
        # [40, 80, 120] was asked, silently (measured 2026-09-12).
        if getattr(v, 'consts', None):
            raise _Unshareable
        return ('jaxpr', _program_key(v if isinstance(v, api.Jaxpr) else v.jaxpr))
    if isinstance(v, (tuple, list)):
        return (type(v), tuple(_param_key(w) for w in v))
    if isinstance(v, np.ndarray):
        return (np.ndarray, v.dtype.str, v.shape, v.tobytes())
    if isinstance(v, (float, complex, np.inexact)):
        return (type(v), np.asarray(v).tobytes())
    if isinstance(v, api.functions):
        # A rule the primitive calls -- a custom_jvp's jvp, a custom_vjp's bwd, a callback -- is
        # rebuilt by every trace. It hashes, so it would not raise below; it would key every
        # call apart instead: a compile per call, and a cache filling with programs that nothing
        # asks for again, evicting the ones that are asked for.
        raise _Unshareable
    if dataclasses.is_dataclass(v) and not isinstance(v, type) and any(
            isinstance(getattr(v, field.name), api.functions) for field in dataclasses.fields(v)):
        # A callback primitive (pure_callback, io_callback, debug.callback) carries the caller's
        # function inside a frozen dataclass. That hashes, and compares the function by identity,
        # so it gets past both checks here and would key every call apart just the same.
        raise _Unshareable
    try:
        hash(v)
    except TypeError:
        raise _Unshareable from None
    return (type(v), v)


def _program_key(jaxpr):
    """Structural identity of a jaxpr: equal keys are one program.

    Built like the key JAX's eager dispatch caches executables on (primitive and parameters,
    ``xla_primitive_callable``), except that a nested jaxpr enters by structure instead of by
    identity: variables by position, literals by bit pattern (so ``-0.0`` is not ``0.0``),
    avals and effects by value. A nested jaxpr carrying constants, a function-valued parameter,
    or one with no hash raises ``_Unshareable`` (:func:`_param_key`).

    Any other parameter enters by its own equality. An enum member, a dtype, the empty device
    mesh a ``jit`` equation names and JAX's forwarding tuples all compare equal across traces --
    the only parameter kinds the FDTD body holds besides plain values, checked 2026-09-12 -- but
    an object built afresh by each trace and compared by identity would key every call apart:
    no wrong answer, only the compile per call this cache exists to remove. Equation contexts
    are compared by their own equality too, which jax 0.11 backs by interning them; a JAX whose
    contexts stopped comparing equal would cost the sharing, never the answer.
    """
    api = _jax_api()
    names = {}

    def bind(variables):
        avals = []
        for v in variables:
            names[v] = len(names)
            avals.append(v.aval)
        return tuple(avals)

    def atom(v):
        if isinstance(v, api.Literal):
            value = np.asarray(v.val)
            return ('literal', value.dtype.str, value.shape, value.tobytes(), v.aval)
        return names[v]

    head = (bind(jaxpr.constvars), bind(jaxpr.invars), frozenset(jaxpr.effects))
    equations = []
    for eqn in jaxpr.eqns:
        inputs = tuple(atom(v) for v in eqn.invars)
        params = tuple((name, _param_key(v)) for name, v in sorted(eqn.params.items()))
        equations.append((eqn.primitive, inputs, params, eqn.ctx, frozenset(eqn.effects),
                          bind(eqn.outvars)))
    return head, tuple(equations), tuple(atom(v) for v in jaxpr.outvars)


class NumpyTimeLoop:
    def __init__(self, device=None, reference=None):
        import array_api_compat.numpy as xp
        if device not in (None, 'cpu', 0):
            raise ValueError('NumPy execution requires a CPU device.')
        self.xp = xp
        self.device = None

    def array(self, value, dtype=None):
        return as_array(self.xp, value, dtype=dtype, device=self.device)

    def zeros(self, shape, dtype):
        return self.xp.zeros(shape, dtype=dtype, device=self.device)

    def conditional(self, predicate, yes, no, operand):
        return yes(operand) if predicate else no(operand)

    def advance(self, step, state, steps, checkpoint_steps):
        for n in range(steps):
            state, _ = step(state, n)
        return state

    def advance_recorded(self, step, state, steps, checkpoint_steps, interval, capture):
        """Keep requested snapshots, without retaining unsampled forward fields."""
        state, _ = step(state, 0)
        samples = [capture(state)]
        count = (steps-1)//interval
        for index in range(count):
            start = 1+index*interval
            def shifted(current, n, start=start):
                return step(current, n+start)
            state = self.advance(shifted, state, interval, checkpoint_steps)
            samples.append(capture(state))
        start = 1+count*interval
        if start < steps:
            state = self.advance(lambda current, n: step(current, n+start),
                                 state, steps-start, checkpoint_steps)
        return state, self.xp.stack(samples)


class JaxTimeLoop(NumpyTimeLoop):
    def __init__(self, device=None, reference=None):
        import jax
        # The FDTD door onto JAX, and it must lift the 32-bit cap for the same reason
        # `src/wavebit/rcwa/_backends/jax_backend.py` does at its own top: with x64 off, `jnp.asarray(x,
        # float64)` still yields float32, so the time loop runs at complex64 while every
        # reported dtype reads complex128 and nothing warns. This constructor is the
        # earliest point on the path -- `scene_fdtd` and `grid_fdtd` both take `runtime.xp`
        # and `runtime.array` from it before building anything -- so it goes here: after
        # `import jax` (the shared body only acts on a loaded JAX) and before any array is
        # built. Not at module level: `import wavebit` already imports this module, and four
        # committed assertions pin that it pulls in no JAX.
        lift_jax_x64(required=True)
        import jax.numpy as xp
        if device in (0, 1):
            device = 'cpu' if device == 0 else 'gpu'
        if isinstance(device, str):
            device = jax.devices(device)[0]
        self.xp = xp
        self.device = device

    def conditional(self, predicate, yes, no, operand):
        import jax
        return jax.lax.cond(predicate, yes, no, operand)

    def advance(self, step, state, steps, checkpoint_steps):
        import jax
        if self.device is not None:
            state = jax.tree.map(lambda v: jax.device_put(v, self.device), state)
        chunks, tail = divmod(steps, checkpoint_steps)
        def block(current, ns):
            return jax.lax.scan(step, current, ns)[0], None
        if chunks:
            ns = self.array(np.arange(chunks*checkpoint_steps).reshape(chunks, checkpoint_steps))
            state = _eager_scan(jax.checkpoint(block), state, ns)[0]
        if tail:
            state = _eager_scan(step, state, self.array(np.arange(chunks*checkpoint_steps, steps)))[0]
        return state

    def advance_recorded(self, step, state, steps, checkpoint_steps, interval, capture):
        import jax
        if self.device is not None:
            state = jax.tree.map(lambda v: jax.device_put(v, self.device), state)
        state, _ = step(state, 0)
        first = capture(state)[None]
        count = (steps-1)//interval
        def block(current, index):
            start = 1+index*interval
            current = self.advance(lambda value, n: step(value, n+start),
                                   current, interval, checkpoint_steps)
            return current, capture(current)
        if count:
            state, samples = _eager_scan(jax.checkpoint(block), state, self.array(np.arange(count)))
            first = self.xp.concat((first, samples), axis=0)
        start = 1+count*interval
        if start < steps:
            state = self.advance(lambda current, n: step(current, n+start),
                                 state, steps-start, checkpoint_steps)
        return state, first


class TorchTimeLoop(NumpyTimeLoop):
    def __init__(self, device=None, reference=None):
        import torch
        import array_api_compat.torch as xp
        if device is None:
            device = reference.device if isinstance(reference, torch.Tensor) else 'cpu'
        elif device in (0, 'cpu'):
            device = 'cpu'
        elif device in (1, 'gpu'):
            device = 'cuda'
        self.xp = xp
        self.device = torch.device(device)
        # Python constants should not lose precision through Torch's global
        # float32 default before meeting a double-precision material array.
        dtype = getattr(reference, 'dtype', None)
        single = str(dtype) in ('float32', 'torch.float32')
        self.real_dtype = torch.float32 if single else torch.float64
        self.complex_dtype = torch.complex64 if single else torch.complex128

    def array(self, value, dtype=None):
        import torch
        def contains_tensor(v):
            return isinstance(v, torch.Tensor) or (isinstance(v, (tuple, list))
                                                   and any(contains_tensor(w) for w in v))
        if isinstance(value, (tuple, list)) and contains_tensor(value):
            return self.xp.stack([self.array(v, dtype=dtype) for v in value])
        if dtype is None and isinstance(value, (float, complex, tuple, list)):
            # Only constant containers belong here. Native tensors go directly
            # through the common graph-preserving RCWA conversion function.
            kind = np.asarray(value).dtype.kind
            if kind in ('f', 'c'):
                dtype = self.real_dtype if kind == 'f' else self.complex_dtype
        return as_array(self.xp, value, dtype=dtype, device=self.device)

    def advance(self, step, state, steps, checkpoint_steps):
        import torch
        from torch.utils.checkpoint import checkpoint
        for start in range(0, steps, checkpoint_steps):
            # Bind indices now: checkpoint backward may replay after this loop
            # has advanced. Non-reentrant checkpoint supports nested state and
            # differentiable coefficients captured by the shared step function.
            def block(current, start=start, stop=min(start+checkpoint_steps, steps)):
                for n in range(start, stop):
                    current, _ = step(current, n)
                return current
            state = (checkpoint(block, state, use_reentrant=False) if torch.is_grad_enabled()
                     else block(state))
        return state


def time_loop_backend(backend, device=None, reference=None):
    """RCWA-compatible IDs 0/1/2, with descriptive aliases for existing callers."""
    choices = {0: NumpyTimeLoop, 1: JaxTimeLoop, 2: TorchTimeLoop,
               'numpy': NumpyTimeLoop, 'jax': JaxTimeLoop, 'torch': TorchTimeLoop}
    try:
        policy = choices[backend]
    except (KeyError, TypeError):
        raise ValueError('backend must be 0/numpy, 1/jax, or 2/torch.') from None
    return policy(device=device, reference=reference)
