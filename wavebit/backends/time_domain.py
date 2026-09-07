"""Array/device and time-loop policies for the single-source FDTD core.

Use the same Array API namespaces and graph-preserving conversion as RCWA.
Only execution/checkpointing differs here; Maxwell/ADE/CPML equations stay in
_core.grid_fdtd and must not be duplicated by a backend.
"""
import numpy as np

from .._core.backend import as_array


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

    def advance(self, step, state, steps, checkpoint_steps):
        for n in range(steps):
            state, _ = step(state, n)
        return state


class JaxTimeLoop(NumpyTimeLoop):
    def __init__(self, device=None, reference=None):
        import jax
        import jax.numpy as xp
        if device in (0, 1):
            device = 'cpu' if device == 0 else 'gpu'
        if isinstance(device, str):
            device = jax.devices(device)[0]
        self.xp = xp
        self.device = device

    def advance(self, step, state, steps, checkpoint_steps):
        import jax
        if self.device is not None:
            state = jax.tree.map(lambda v: jax.device_put(v, self.device), state)
        chunks, tail = divmod(steps, checkpoint_steps)
        def block(current, ns):
            return jax.lax.scan(step, current, ns)[0], None
        if chunks:
            ns = self.array(np.arange(chunks*checkpoint_steps).reshape(chunks, checkpoint_steps))
            state = jax.lax.scan(jax.checkpoint(block), state, ns)[0]
        if tail:
            state = jax.lax.scan(step, state, self.array(np.arange(chunks*checkpoint_steps, steps)))[0]
        return state


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
