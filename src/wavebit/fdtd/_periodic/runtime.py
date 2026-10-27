"""Array/device and time-loop policies for the single-source FDTD core.

Use the same Array API namespaces and graph-preserving conversion as RCWA.
Only execution/checkpointing differs here; Maxwell/ADE/CPML equations stay in
wavebit.fdtd._periodic.grid and must not be duplicated by a backend.
"""
import threading

import numpy as np

from wavebit._common.backend import as_array
from wavebit._common.backend import lift_jax_x64

# Compiled JAX scan programs, least recently used first. See JaxTimeLoop._run.
_SCAN_PROGRAMS = {}
_SCAN_PROGRAMS_MAX = 64
# Whichever thread solves reads and writes the cache; a `pop` racing an eviction could otherwise
# raise `KeyError`, or change the dict's size under the eviction loop.
_SCAN_LOCK = threading.Lock()


class Step:
    """One time step held as data: ``build(runtime, settings, arrays)`` returns ``step(state, n)``.

    ``build`` is a module-level function, ``settings`` a hashable value of plain Python numbers,
    tuples and slices, and ``arrays`` a pytree of the step's arrays. ``fdtd_grid`` used to hand
    the time loop a fresh closure on every call; an eager ``lax.scan`` caches its executable on
    the traced body, which hashes by identity, so every solve compiled again. Held as data, the
    same program shape has the same builder and equal settings on every call, and the JAX loop
    runs it through one plain ``jax.jit`` that receives the arrays as arguments
    (:meth:`JaxTimeLoop._run`). NumPy and Torch simply call it.

    A leaf of ``arrays`` that is a Python or NumPy scalar rather than an array (an absorber-free
    ``ah`` is the float ``1.0``) is treated as part of the program, exactly as a closure would
    have baked it in, and not as an argument.
    """

    def __init__(self, build, runtime, settings, arrays, offset=0):
        self.build, self.runtime, self.settings, self.arrays = build, runtime, settings, arrays
        self.offset = offset
        self._bound = None

    def shifted(self, start):
        """This step with every time index advanced by the static integer ``start``."""
        return Step(self.build, self.runtime, self.settings, self.arrays, self.offset+start)

    def __call__(self, state, n):
        if self._bound is None:
            self._bound = _bind(self.build, self.runtime, self.settings, self.arrays, self.offset)
        return self._bound(state, n)


def _bind(build, runtime, settings, arrays, offset):
    step = build(runtime, settings, arrays)
    if offset:
        return lambda state, n: step(state, n+offset)
    return step


def _static_key(v):
    """``v`` as a cache key, by type and value; ``TypeError`` when it cannot be one.

    Floats enter by bit pattern, so ``-0.0`` is not ``0.0``; the type enters too, so ``1``,
    ``1.0`` and ``True`` -- equal and equally hashed in Python -- are three keys, as they are
    three programs.
    """
    if isinstance(v, (tuple, list)):
        return (type(v), tuple(_static_key(w) for w in v))
    if isinstance(v, (float, complex, np.inexact)):
        return (type(v), np.asarray(v).tobytes())
    if isinstance(v, np.ndarray):
        raise TypeError('an array is not a static value')
    hash(v)
    return (type(v), v)


def _signature(tree):
    """Tree structure, shapes and dtypes: what a jit compiles one executable for."""
    import jax
    leaves, structure = jax.tree.flatten(tree)
    return structure, tuple((np.shape(v), str(getattr(v, 'dtype', type(v))),
                             getattr(v, 'weak_type', None)) for v in leaves)


def _chunk_scan(step, state, ns):
    def block(current, row):
        import jax
        return jax.lax.scan(step, current, row)[0], None
    import jax
    return jax.lax.scan(jax.checkpoint(block), state, ns)


def _step_scan(step, state, ns):
    import jax
    return jax.lax.scan(step, state, ns)


def _record_scan(runtime, step, state, indices, interval, checkpoint_steps, capture):
    import jax

    def block(current, index):
        start = 1+index*interval
        current = runtime.advance(lambda value, n: step(value, n+start),
                                  current, interval, checkpoint_steps)
        return current, capture(current)
    return jax.lax.scan(jax.checkpoint(block), state, indices)


def _scan_program(kind, build, runtime, settings, structure, constants, offset, extra):
    """``jax.jit`` of one scan over a :class:`Step`, taking every array as an argument.

    Nothing reaches the compiled program but its arguments and the static values in its cache
    key: the builder, the settings, the scalar leaves, the offset and the scan's own static
    options. So two calls with equal keys and equal argument shapes and dtypes are one program.
    """
    import jax

    def scan_program(dynamic, state, xs):
        values = iter(dynamic)
        leaves = [next(values) if c is _ARRAY else c for c in constants]
        step = _bind(build, runtime, settings, jax.tree.unflatten(structure, leaves), offset)
        if kind == 'chunks':
            return _chunk_scan(step, state, xs)
        if kind == 'steps':
            return _step_scan(step, state, xs)
        return _record_scan(runtime, step, state, xs, *extra)
    return jax.jit(scan_program)


_ARRAY = object()


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


def _traced(jax, xp, reference):
    """Whether a JAX transform is tracing this call: a traced input, or a jit trace staging
    every new array (``jnp.zeros`` is a tracer only there; checked on jax 0.11.0 and 0.12.0)."""
    return (any(isinstance(v, jax.core.Tracer) for v in jax.tree.leaves(reference))
            or isinstance(xp.zeros(()), jax.core.Tracer))


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
        import jax.numpy as xp
        if not jax.config.jax_enable_x64 and _traced(jax, xp, reference):
            # Too late to lift: the caller's trace already holds its inputs at 32 bits, so the
            # solve would run on rounded values, and a jitted caller's next call fails on a
            # 32-/64-bit input mismatch. Same refusal as the public API's.
            raise RuntimeError(
                'JAX double precision requires jax.config.update("jax_enable_x64", True) '
                'before creating arrays; this FDTD solve was reached inside a JAX transform '
                '(jit, grad or vmap) while it was off, so its inputs are already 32-bit.')
        lift_jax_x64(required=True)
        if device in (0, 1):
            device = 'cpu' if device == 0 else 'gpu'
        if isinstance(device, str):
            device = jax.devices(device)[0]
        self.xp = xp
        self.device = device

    def conditional(self, predicate, yes, no, operand):
        """``lax.cond``, or the chosen branch directly when ``predicate`` is a Python bool.

        A time step given a Python index (the recorded loop's eager first step) has a known
        predicate; an eager ``lax.cond`` on it would compile a fresh program on every solve,
        because its branches are new closures each time.
        """
        import jax
        if isinstance(predicate, (bool, np.bool_)):
            return yes(operand) if predicate else no(operand)
        return jax.lax.cond(predicate, yes, no, operand)

    def advance(self, step, state, steps, checkpoint_steps):
        import jax
        if self.device is not None:
            state = jax.tree.map(lambda v: jax.device_put(v, self.device), state)
        chunks, tail = divmod(steps, checkpoint_steps)
        if chunks:
            ns = self.array(np.arange(chunks*checkpoint_steps).reshape(chunks, checkpoint_steps))
            state = self._run('chunks', step, state, ns)[0]
        if tail:
            state = self._run('steps', step, state, self.array(np.arange(chunks*checkpoint_steps, steps)))[0]
        return state

    def advance_recorded(self, step, state, steps, checkpoint_steps, interval, capture):
        import jax
        if self.device is not None:
            state = jax.tree.map(lambda v: jax.device_put(v, self.device), state)
        state, _ = step(state, 0)
        first = capture(state)[None]
        count = (steps-1)//interval
        if count:
            state, samples = self._run('record', step, state, self.array(np.arange(count)),
                                       (interval, checkpoint_steps, capture))
            first = self.xp.concat((first, samples), axis=0)
        start = 1+count*interval
        if start < steps:
            shifted = (step.shifted(start) if isinstance(step, Step)
                       else lambda current, n: step(current, n+start))
            state = self.advance(shifted, state, steps-start, checkpoint_steps)
        return state, first

    def _run(self, kind, step, state, xs, extra=()):
        """One scan of ``step``: a cached ``jax.jit`` program for a :class:`Step`, else plain.

        ``kind`` is ``'chunks'`` (checkpointed rows of steps), ``'steps'`` or ``'record'``. The
        program is keyed on static values alone -- the builder, its settings and scalar leaves,
        the offset, the scan's options, this loop's device -- and on the shapes and dtypes of
        its arguments, so a repeated solve of one program shape compiles nothing, and a cached
        program holds none of the arrays it ran on. The shapes are in the key so that a sweep
        of scan lengths or grid sizes is one entry per program under the LRU bound, rather than
        one entry whose jit grows a compiled copy per shape where the bound cannot see it.

        Anything else runs the plain ``lax.scan``, which is exact and only compiles again on
        each call: a plain function, a builder that closes over values, settings that cannot be
        a key, and any call holding a tracer. Inside a user's ``jax.jit`` nothing is dispatched here, and a nested jit would
        change that user's compiled program (measured on jax 0.12.0: the same bytes out, from a
        larger program with more temporary memory); eager AD keeps the path it had before.
        """
        import jax
        leaves, structure = jax.tree.flatten(getattr(step, 'arrays', ()))
        # A builder that closes over a value would bake it into the cached program unseen by the
        # key, and a later call would get the first call's value back: only a builder whose
        # inputs are all its arguments is shared.
        if (not isinstance(step, Step) or getattr(step.build, '__closure__', None)
                or any(isinstance(v, jax.core.Tracer) for v in
                       leaves + jax.tree.leaves((state, xs)))):
            return self._plain(kind, step, state, xs, extra)
        is_array = [isinstance(v, (jax.Array, np.ndarray)) for v in leaves]
        dynamic = [v for v, a in zip(leaves, is_array) if a]
        constants = tuple(_ARRAY if a else v for v, a in zip(leaves, is_array))
        try:
            key = (kind, step.build, type(self), self.device, _static_key(step.settings),
                   structure, _static_key(tuple(v for v in constants if v is not _ARRAY)),
                   tuple(is_array), step.offset, _static_key(extra[:2]), extra[2:],
                   _signature((dynamic, state, xs)))
            hash(key)
        except TypeError:
            return self._plain(kind, step, state, xs, extra)
        with _SCAN_LOCK:
            program = _SCAN_PROGRAMS.pop(key, None)
            if program is None:
                program = _scan_program(kind, step.build, self, step.settings, structure,
                                        constants, step.offset, extra)
            _SCAN_PROGRAMS[key] = program
            while len(_SCAN_PROGRAMS) > _SCAN_PROGRAMS_MAX:
                del _SCAN_PROGRAMS[next(iter(_SCAN_PROGRAMS))]
        return program(dynamic, state, xs)

    def _plain(self, kind, step, state, xs, extra):
        if kind == 'record':
            return _record_scan(self, step, state, xs, *extra)
        return (_chunk_scan if kind == 'chunks' else _step_scan)(step, state, xs)


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
