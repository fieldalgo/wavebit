"""The runtime primitives: the only backend-specific code in the package.

The kernel is one body of code written against an ``rt`` object; this module supplies three
of them. What each primitive is on each backend was decided by measurement
(``studies/fdtd_slab_add_cost.py``, 2026-09-18, CPU), and the table is the contract:

    primitive        NumPy                         JAX                              Torch
    scan             for loop                      nested lax.scan under            Python loop; torch.utils.checkpoint
                                                   jax.checkpoint per block         (non-reentrant) per block
    slab_add         np.add(view, v, out=view)     4-D index scatter-add            autograd.Function: add_ in place,
                                                   (never a slice add)              mark_dirty; backward slices g
    slab_mul         in place                      4-D index scatter-multiply       same Function pattern with mul_
    diff / update    out= into preallocated        functional                       functional (out of place)
                     buffers
    array            np.asarray                    jnp.asarray, x64 lifted first    torch.as_tensor(..., device=)
    jit_for          the function itself           jax.jit, once per plan, kept     the function itself
                                                   on the instance

**Why the JAX slab add is a scatter and not ``a.at[slices].add``.** The slice form lowers to
a dynamic-update-slice that costs 1.4-2.1 full-volume passes per term unless the slab's
origin is a multiple of its width; the index form ``a.at[(c, z, y, x)].add(v,
indices_are_sorted=True, unique_indices=True, mode='promise_in_bounds')`` costs +0.04 ms at
N = 96 whatever the alignment. The four index arrays are built once per slab and closed
over; the carry is never reshaped inside the scan body (that is one full copy per step).

**Why Torch's in-place slab write is an ``autograd.Function``.** A bare ``add_`` on a slice
under autograd costs one full-volume cotangent copy per term per step in backward; the
Function whose forward marks the tensor dirty and whose backward returns ``(g, g[slab])``
is O(slab) there. The contract that comes with it: a slab write goes only into a tensor no
operation has consumed yet. ``slab_mul`` cannot be O(slab) in backward -- the cotangent of
the multiplied slab changes, so its backward clones ``g`` once; in stage 0 the only
multiplies are the wall masks, two thin layers per face.

**Why NumPy works in preallocated buffers.** A stand-in kernel that allocated a temporary
per operation ran 40-50x slower than JAX's; the real kernel keeps H, E and the two curl
scratch arrays live across steps and writes every full-volume result with ``out=``. The
pads (one layer thick) are allocated per call: a layer is ``1/N`` of the volume.

**x64 on JAX.** ``wavebit._core.backend.lift_jax_x64(required=True)`` is called right after
JAX is imported and before any array is built; it is the one door. JAX is imported lazily
here, behind the backend choice, so ``import wavebit`` never pulls it in.
"""
import difflib
import enum

import numpy as np

__all__ = ['runtime', 'BACKENDS']

BACKENDS = ('numpy', 'jax', 'torch')

_DTYPES = {'float64': np.float64, 'complex128': np.complex128}


def _flatten(tree):
    """Leaves of a nested dict/tuple/list, and a function rebuilding the same structure."""
    if isinstance(tree, dict):
        keys = list(tree)
        subs = [_flatten(tree[k]) for k in keys]
        leaves = [leaf for s, _ in subs for leaf in s]
        counts = [len(s) for s, _ in subs]

        def rebuild(flat):
            out, pos = {}, 0
            for k, (_, rb), c in zip(keys, subs, counts):
                out[k] = rb(flat[pos:pos + c])
                pos += c
            return out
        return leaves, rebuild
    if isinstance(tree, (tuple, list)):
        subs = [_flatten(t) for t in tree]
        leaves = [leaf for s, _ in subs for leaf in s]
        counts = [len(s) for s, _ in subs]
        kind = type(tree)

        def rebuild(flat):
            out, pos = [], 0
            for (_, rb), c in zip(subs, counts):
                out.append(rb(flat[pos:pos + c]))
                pos += c
            return kind(out)
        return leaves, rebuild
    if tree is None:
        return [], lambda flat: None
    return [tree], lambda flat: flat[0]


def _slab_indices(slab):
    """The four sorted, unique index arrays of ``slab`` for a 4-D scatter, built once."""
    z, y, x = np.meshgrid(np.arange(*slab.z), np.arange(*slab.y), np.arange(*slab.x),
                          indexing='ij')
    n = z.size
    return (np.full(n, slab.c, dtype=np.int32), z.reshape(-1).astype(np.int32),
            y.reshape(-1).astype(np.int32), x.reshape(-1).astype(np.int32))


def _slice_along(a, ax, lo, hi):
    idx = [slice(None)] * a.ndim
    idx[ax] = slice(lo, hi)
    return a[tuple(idx)]


class _Functional:
    """What JAX and Torch share: out-of-place differences and updates through ``xp``."""

    def scratch(self, key, shape, dtype):
        return None

    def fresh(self, x):
        return x

    def zero_layer(self, a, ax):
        shape = list(a.shape)
        shape[ax] = 1
        return self.xp.zeros(tuple(shape), dtype=a.dtype, **self._place)

    def take_layer(self, a, ax, index):
        return _slice_along(a, ax, index, index + 1)

    def conj(self, x):
        return self.xp.conj(x)

    def diff(self, a, ax, side, layer, out=None):
        n = a.shape[ax]
        if side == '+':
            return self.xp.concat([_slice_along(a, ax, 1, n), layer], axis=ax) - a
        return a - self.xp.concat([layer, _slice_along(a, ax, 0, n - 1)], axis=ax)

    def sub(self, a, b, out=None):
        return a - b

    def assemble(self, parts, out):
        return self.xp.stack(parts, axis=0)

    def update(self, F, cF, alpha, beta):
        return alpha * F + beta * cF

    def jit_for(self, key, make_fn):
        return self.jit(make_fn())


class NumpyRuntime:
    """NumPy: preallocated buffers, ``out=`` everywhere at full volume, a plain loop."""

    name = 'numpy'
    xp = np

    def __init__(self, device=None):
        if device not in (None, 'cpu'):
            raise ValueError(f"backend='numpy' has one device, the CPU; device={device!r} "
                             f"names another.")
        self._buffers = {}

    def dtype(self, name):
        return _DTYPES[name]

    def array(self, v, dtype=None):
        return np.asarray(v, dtype=None if dtype is None else _DTYPES[dtype])

    def zeros(self, shape, dtype):
        return np.zeros(shape, dtype=_DTYPES[dtype])

    def fresh(self, x):
        return np.array(x, copy=True)

    def scratch(self, key, shape, dtype):
        buf = self._buffers.get(key)
        if buf is None or buf.shape != tuple(shape) or buf.dtype != dtype:
            buf = np.empty(shape, dtype=dtype)
            self._buffers[key] = buf
        return buf

    def zero_layer(self, a, ax):
        shape = list(a.shape)
        shape[ax] = 1
        return np.zeros(tuple(shape), dtype=a.dtype)

    def take_layer(self, a, ax, index):
        return _slice_along(a, ax, index, index + 1)

    def conj(self, x):
        return np.conj(x)

    def diff(self, a, ax, side, layer, out=None):
        n = a.shape[ax]
        if out is None:
            out = np.empty_like(a)
        if side == '+':
            np.subtract(_slice_along(a, ax, 1, n), _slice_along(a, ax, 0, n - 1),
                        out=_slice_along(out, ax, 0, n - 1))
            np.subtract(layer, _slice_along(a, ax, n - 1, n), out=_slice_along(out, ax, n - 1, n))
        else:
            np.subtract(_slice_along(a, ax, 1, n), _slice_along(a, ax, 0, n - 1),
                        out=_slice_along(out, ax, 1, n))
            np.subtract(_slice_along(a, ax, 0, 1), layer, out=_slice_along(out, ax, 0, 1))
        return out

    def sub(self, a, b, out=None):
        return np.subtract(a, b, out=out)

    def assemble(self, parts, out):
        return out                      # every part was written into out[c] already

    def update(self, F, cF, alpha, beta):
        # alpha*F + beta*cF, in place in F; cF is scratch and may be scaled in place.
        if float(alpha) != 1.0:
            np.multiply(F, alpha, out=F)
        np.multiply(cF, beta, out=cF)
        np.add(F, cF, out=F)
        return F

    def slab_add(self, a, slab, v):
        view = a[slab.slices]
        np.add(view, v, out=view)
        return a

    def slab_mul(self, a, slab, m):
        view = a[slab.slices]
        np.multiply(view, m, out=view)
        return a

    def record(self, acc, n, value):
        acc[n] = value
        return acc

    def scan(self, body, carry, steps, schedule=None):
        for n in range(steps):
            carry = body(carry, n)
        return carry

    def jit(self, fn):
        return fn

    def jit_for(self, key, make_fn):
        return make_fn()

    def to_device(self, tree):
        return tree

    def numpy(self, x):
        return np.asarray(x)


class JaxRuntime(_Functional):
    """JAX: x64 lifted at construction, ``lax.scan`` with optional checkpointed blocks.

    The executor is compiled once per plan (``jit_for``): a fresh ``jax.jit`` per call would
    retrace and recompile the whole scan every ``fd.run``, which on a 16^3, 20-step box is
    250 ms against 5 ms for the NumPy run of the same scene, and a sweep or a ``jax.grad``
    loop pays it every iteration. The plan is hashable by value and equal plans mean equal
    programs, so it is the key; ``runtime()`` hands out one instance per backend and device
    so the cache is reached across calls.
    """

    name = 'jax'

    def __init__(self, device=None):
        import jax
        from wavebit._core.backend import lift_jax_x64
        lift_jax_x64(required=True)
        import jax.numpy as jnp
        self.jax, self.xp = jax, jnp
        self.device = None
        if device is not None:
            self.device = jax.devices(device)[0] if isinstance(device, str) else device
        self._place = {}
        self._indices = {}
        self._compiled = {}

    def jit_for(self, key, make_fn):
        """``jit(make_fn())`` for ``key`` (a plan), built the first time and kept."""
        fn = self._compiled.get(key)
        if fn is None:
            fn = self._compiled[key] = self.jit(make_fn())
        return fn

    def dtype(self, name):
        return self.xp.dtype(name)

    def array(self, v, dtype=None):
        x = self.xp.asarray(v, dtype=None if dtype is None else self.dtype(dtype))
        return x if self.device is None else self.jax.device_put(x, self.device)

    def zeros(self, shape, dtype):
        return self.xp.zeros(shape, dtype=self.dtype(dtype))

    def _idx(self, slab):
        idx = self._indices.get(slab)
        if idx is None:
            idx = self._indices[slab] = _slab_indices(slab)
        return idx

    def _scatter(self, a, slab):
        return a.at[self._idx(slab)]

    def slab_add(self, a, slab, v):
        if getattr(v, 'ndim', 0):
            v = v.reshape(-1)
        return self._scatter(a, slab).add(v, indices_are_sorted=True, unique_indices=True,
                                          mode='promise_in_bounds')

    def slab_mul(self, a, slab, m):
        if getattr(m, 'ndim', 0):
            m = m.reshape(-1)
        return self._scatter(a, slab).multiply(m, indices_are_sorted=True, unique_indices=True,
                                               mode='promise_in_bounds')

    def record(self, acc, n, value):
        return acc.at[n].set(value)

    def scan(self, body, carry, steps, schedule=None):
        lax, jnp = self.jax.lax, self.xp

        def one(c, n):
            return body(c, n), None

        if schedule is None:
            carry, _ = lax.scan(one, carry, jnp.arange(steps))
            return carry
        K = int(schedule)
        blocks, rem = divmod(steps, K)

        @self.jax.checkpoint
        def block(c, b):
            c, _ = lax.scan(lambda c2, i: (body(c2, b * K + i), None), c, jnp.arange(K))
            return c, None

        if blocks:
            carry, _ = lax.scan(block, carry, jnp.arange(blocks))
        if rem:
            carry, _ = lax.scan(one, carry, blocks * K + jnp.arange(rem))
        return carry

    def jit(self, fn):
        return self.jax.jit(fn)

    def to_device(self, tree):
        return tree if self.device is None else self.jax.device_put(tree, self.device)

    def numpy(self, x):
        return np.asarray(x)


def _torch_functions(torch):
    """The two in-place slab Functions; built once per process, after torch is imported."""

    def sum_to(g, shape):
        while g.ndim > len(shape):
            g = g.sum(0)
        for d, s in enumerate(shape):
            if s == 1 and g.shape[d] != 1:
                g = g.sum(d, keepdim=True)
        return g

    class SlabAdd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, v, slices):
            ctx.slices = slices
            ctx.vshape = tuple(getattr(v, 'shape', ()))
            a[slices].add_(v)
            ctx.mark_dirty(a)
            return a

        @staticmethod
        def backward(ctx, g):
            # A gradient is returned for the value only when it is a tensor that asked for
            # one: Torch refuses a gradient "at position 2" for a Python-float value, which
            # is what a static term (the kernel's own SlabTerm hook) passes.
            if not ctx.needs_input_grad[1]:
                return g, None, None
            gv = g[ctx.slices]
            if ctx.vshape != tuple(gv.shape):
                gv = sum_to(gv, ctx.vshape) if ctx.vshape else gv.sum()
            return g, gv, None

    class SlabMul(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, m, slices):
            ctx.slices = slices
            ctx.m = m
            a[slices].mul_(m)
            ctx.mark_dirty(a)
            return a

        @staticmethod
        def backward(ctx, g):
            ga = g.clone()
            ga[ctx.slices] *= ctx.m
            return ga, None, None

    return SlabAdd, SlabMul


class TorchRuntime(_Functional):
    """Torch: float64 everywhere, a Python loop, in-place slab writes through a Function."""

    name = 'torch'

    def __init__(self, device=None):
        import torch
        import array_api_compat.torch as xp
        self.torch, self.xp = torch, xp
        self.device = torch.device('cpu' if device is None else device)
        self._place = {'device': self.device}
        self.SlabAdd, self.SlabMul = _torch_functions(torch)

    def dtype(self, name):
        return {'float64': self.torch.float64, 'complex128': self.torch.complex128}[name]

    def array(self, v, dtype=None):
        return self.torch.as_tensor(v, dtype=None if dtype is None else self.dtype(dtype),
                                    device=self.device)

    def zeros(self, shape, dtype):
        return self.torch.zeros(shape, dtype=self.dtype(dtype), device=self.device)

    def _live(self, *xs):
        return self.torch.is_grad_enabled() and any(getattr(x, 'requires_grad', False) for x in xs)

    def slab_add(self, a, slab, v):
        if self._live(a, v):
            return self.SlabAdd.apply(a, v, slab.slices)
        a[slab.slices].add_(v)
        return a

    def slab_mul(self, a, slab, m):
        if self._live(a, m):
            return self.SlabMul.apply(a, m, slab.slices)
        a[slab.slices].mul_(m)
        return a

    def record(self, acc, n, value):
        acc[n] = value
        return acc

    def scan(self, body, carry, steps, schedule=None):
        if schedule is None:
            for n in range(steps):
                carry = body(carry, n)
            return carry
        from torch.utils.checkpoint import checkpoint
        K = int(schedule)
        blocks, rem = divmod(steps, K)
        leaves, rebuild = _flatten(carry)

        def block(b, *flat):
            c = rebuild(list(flat))
            for i in range(K):
                c = body(c, b * K + i)
            return tuple(_flatten(c)[0])

        for b in range(blocks):
            leaves = list(checkpoint(block, b, *leaves, use_reentrant=False))
        carry = rebuild(leaves)
        for n in range(blocks * K, steps):
            carry = body(carry, n)
        return carry

    def jit(self, fn):
        return fn

    def to_device(self, tree):
        return tree

    def numpy(self, x):
        return x.detach().cpu().numpy()


_RUNTIMES = {'numpy': NumpyRuntime, 'jax': JaxRuntime, 'torch': TorchRuntime}

#: One runtime per (class, device): the JAX compile cache and the slab index cache live on
#: the instance, so ``fd.run`` must get the same instance back to reach them. Keyed by the
#: class as looked up at call time, so a test that swaps ``_RUNTIMES['torch']`` for a
#: subclass gets its own instance and never a stale one.
_INSTANCES = {}


def runtime(backend, device=None):
    """The runtime for ``backend``: ``'numpy'`` | ``'jax'`` | ``'torch'`` or ``wavebit.Backend``.

    The same object comes back for the same backend and device (see ``_INSTANCES``); a
    device that cannot be a dictionary key gets a fresh runtime each time.
    """
    name = backend
    if isinstance(backend, enum.Enum):
        name = backend.name.lower()
    elif isinstance(backend, int) and not isinstance(backend, bool):
        from wavebit._core.enums import Backend
        try:
            name = Backend(backend).name.lower()
        except ValueError:
            name = repr(backend)
    if not isinstance(name, str) or name not in _RUNTIMES:
        near = difflib.get_close_matches(str(name).lower(), BACKENDS, n=1, cutoff=0.5)
        hint = f' Did you mean {near[0]!r}?' if near else ''
        raise ValueError(f'backend={backend!r} is not one of {BACKENDS} (or wavebit.Backend).{hint}')
    cls = _RUNTIMES[name]
    try:
        key = (cls, device)
        hash(key)
    except TypeError:
        return cls(device)
    rt = _INSTANCES.get(key)
    if rt is None:
        rt = _INSTANCES[key] = cls(device)
    return rt
