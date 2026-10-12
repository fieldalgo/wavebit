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
    cpml_stretch     four fused in-place ops on    d.at[slices].add -- a DUS,       the same four in place; no
                     the slab view, one            never the scatter                scratch at all
                     slab-shaped scratch
    diff / update    out= into preallocated        functional                       functional (out of place)
                     buffers
    dft_add          multiply into a region        acc + phase * view               acc + phase * view
                     scratch, add in place
    weighted_energy  einsum over a stride-0        stop_gradient(sum(w |v|^2))      no_grad (w |v|^2).sum()
                     broadcast of w
    real_inner       einsum, real and imag parts   stop_gradient(sum w Re(a* b))    no_grad (w (a* b).real).sum()
    energy_weight    r (1 + a)/(2 b) times t,      stop_gradient(r (1 + a) t/(2 b)) no_grad r (1 + a) t/(2 b)
                     into one scratch
    constant         the value itself              lax.stop_gradient                detach
    when             Python if                     lax.cond                         Python if
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

**Why the CPML write is not the slab add, on any backend.** ``studies/fdtd_cpml_slab_cost.py``
(2026-09-18, CPU) settled it: the psi of a CPML face lives in the slab everywhere -- 36 MB
against 192 MB at 128^3 with six faces, and faster on all three backends -- and the *writing*
is a different primitive from ``slab_add``. On JAX it is ``d.at[slices].add(...)``, the
dynamic-update-slice the stage-0 study rejected for a small arbitrary cube, because a CPML
slab is not one: it has full extent on two axes, is contiguous on the third and is anchored
at 0 or ``N - cells``, so the loop bounds are exact and LLVM has nothing to unswitch. It beat
the scatter at every N and on every axis (+6585..+8161 us against +9503..+12910 at N = 128)
and needs no index arrays, which a scatter pays 1.5x the psi's own bytes for. On NumPy and
Torch the four operations of the recurrence run in place on the slab *view* of the
difference; NumPy's fused form is 1.11x against 1.24x for a gather-correct-add-back at
128^3 / 6 faces. Under reverse mode on JAX the transposed scan body carries exactly one
full-volume cotangent copy for the DUS form (against five for full-volume psi and none for
the scatters, which trade them for gather fusions) -- but the times do not follow the counts
(+21..+27 ms/step at N = 128 for every layout), so read that as the reason the slab layout
does not get *worse* under reverse mode, not as a saving. **Under Torch autograd neither
in-place writing applies unchanged**: a psi is read as well as written, so its
``autograd.Function``'s backward is not the slab-add one, and that case is unmeasured. Until
it is, a differentiated Torch run takes the out-of-place writing below and pays for it.

**Why NumPy works in preallocated buffers.** A stand-in kernel that allocated a temporary
per operation ran 40-50x slower than JAX's; the real kernel keeps H, E and the two curl
scratch arrays live across steps and writes every full-volume result with ``out=``. The
pads (one layer thick) are allocated per call: a layer is ``1/N`` of the volume.

**The DFT accumulation (stage 3) is two primitives, ``dft_add`` and ``energy``.** Every
frequency-domain monitor is a running sum ``acc += phase[f] * F[region]`` per step, over a region
that is a plane, a box face or a small volume -- never the whole grid unless a caller asks
for it. On NumPy it writes through one region-sized scratch per component, ``np.multiply(
phase, view, out=scratch)`` then ``np.add(acc, scratch, out=acc)``, so a step allocates
nothing at full volume and nothing at region size after the first; the view is a view. On
JAX and Torch it is the functional ``acc + phase * view``. ``energy`` is ``sum |F|^2`` over
the region for the decay diagnostic, with the gradient stopped on every backend (it is a
diagnostic; tracking it would put the region into the backward pass for nothing).

**Concrete stopping amplitudes.** ``state_amplitude`` and ``positive_maximum`` are
used only by the optional Decay executor. NumPy/Torch use component abs/max; JAX uses
sign-cleared float64 integer keys for both reductions so a positive subnormal is not
flushed to zero. Nonfinite values become sticky +Inf. These primitives do not claim
a derivative and do not change the existing fixed-length solver arithmetic.

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


def _native_state_amplitude(rt, array):
    """Component amplitude for the native CPU NumPy/Torch stopping reductions."""
    if any(n == 0 for n in array.shape):
        return rt.array(0.0, 'float64')
    xp = rt.xp
    value = xp.max(xp.abs(xp.real(array)))
    if 'complex' in str(array.dtype):
        value = xp.maximum(value, xp.max(xp.abs(xp.imag(array))))
    return xp.where(xp.isfinite(value), value, xp.full_like(value, np.inf))


def _native_positive_maximum(xp, a, b):
    """Nonnegative maxima, with any nonfinite input permanently represented by +Inf."""
    value = xp.maximum(a, b)
    return xp.where(xp.isfinite(value), value, xp.full_like(value, np.inf))


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

    #: An update returns a new array and leaves the old one as it was (the step reads the
    #: interior's energy after its H update from both, ``src/wavebit/fd/_kernel/step.py``).
    in_place = False

    def state_amplitude(self, array):
        """Concrete stopping norm; no square, complex modulus or mixed-unit sum."""
        return _native_state_amplitude(self, array)

    def positive_maximum(self, a, b):
        """Stopping-window maximum of nonnegative arrays; nonfinite is sticky."""
        return _native_positive_maximum(self.xp, a, b)

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

    #: An update overwrites its field and scales its curl in place (the step reads the
    #: interior's energy before its H update, ``src/wavebit/fd/_kernel/step.py``).
    in_place = True

    name = 'numpy'
    xp = np

    def __init__(self, device=None):
        if device not in (None, 'cpu'):
            raise ValueError(f"backend='numpy' has one device, the CPU; device={device!r} "
                             f"names another.")
        self._buffers = {}

    def state_amplitude(self, array):
        """Concrete stopping norm; no square, complex modulus or mixed-unit sum."""
        return _native_state_amplitude(self, array)

    def positive_maximum(self, a, b):
        """Stopping-window maximum of nonnegative arrays; nonfinite is sticky."""
        return _native_positive_maximum(self.xp, a, b)

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
        # ``alpha`` is a scalar in vacuum and a (3, nz, ny, nx) array once a material or an
        # Absorber makes it one; the skip is for the scalar 1.0 only, so a vacuum run keeps
        # the stage-0 operation count exactly (and its measured transient allocation).
        if getattr(alpha, 'ndim', 0) or float(alpha) != 1.0:
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

    def slab_scale(self, a, slab, m):
        """``a[slab] *= m`` with ``m`` broadcasting against the slab (a 1-D profile, reshaped).

        The stabilizing loss's primitive (``_kernel/cpml.py::damp``). On NumPy and Torch it is
        ``slab_mul``; on JAX it is a different writing, see there.
        """
        return self.slab_mul(a, slab, m)

    def cpml_stretch(self, d, region, psi, b, a, ik, key):
        """The fused CPML recurrence, four in-place operations on the slab view of ``d``.

        ``psi <- b psi + a d`` then ``d <- d/kappa + psi``, in that order, with one
        slab-shaped scratch for the ``a * d`` product. Fused rather than gathered: 1.11x
        against 1.24x over the no-absorber baseline at 128^3 with six faces, and it
        reproduces the full-volume-psi field exactly (0e+00), where the equivalent
        add-back form ``d += d(1/kappa - 1) + psi`` differs by one rounding per cell per
        step.
        """
        view = d[region.slices]
        tmp = self.scratch('cpml_' + key, view.shape, view.dtype)
        np.multiply(psi, b, out=psi)
        np.add(psi, np.multiply(a, view, out=tmp), out=psi)
        np.multiply(view, ik, out=view)
        np.add(view, psi, out=view)
        return d, psi

    def record(self, acc, n, value):
        acc[n] = value
        return acc

    def dft_add(self, acc, view, phase, key):
        """``acc += phase[:, None, None, None] * view`` in place, through one scratch."""
        tmp = self.scratch(key, acc.shape, acc.dtype)
        np.multiply(phase.reshape((phase.shape[0],) + (1,) * view.ndim), view, out=tmp)
        np.add(acc, tmp, out=acc)
        return acc

    def energy(self, view):
        """``sum |view|^2`` without a region-sized temporary (einsum over the strided view)."""
        if np.iscomplexobj(view):
            return (np.einsum('ijk,ijk->', view.real, view.real)
                    + np.einsum('ijk,ijk->', view.imag, view.imag))
        return np.einsum('ijk,ijk->', view, view)

    def weighted_energy(self, view, w):
        """``sum w |view|^2`` over a ``(3, nz, ny, nx)`` view, ``w`` a scalar or broadcasting
        against it: no view-sized temporary either way (a stride-0 ``broadcast_to`` of an
        array weight into one einsum)."""
        parts = (view.real, view.imag) if np.iscomplexobj(view) else (view,)
        if np.ndim(w) == 0:
            return float(w) * sum(float(np.einsum('cijk,cijk->', p, p)) for p in parts)
        w = np.broadcast_to(w, view.shape)
        return sum(float(np.einsum('cijk,cijk,cijk->', w, p, p)) for p in parts)

    def real_inner(self, a, b, w=1.0):
        """``Re sum w conj(a) b`` over two ``(3, nz, ny, nx)`` views, ``w`` a scalar or
        broadcasting against them: no temporary either way (einsum, a stride-0 broadcast of an
        array weight)."""
        pairs = [(a.real, b.real)]
        if np.iscomplexobj(a) and np.iscomplexobj(b):
            pairs.append((a.imag, b.imag))
        if np.ndim(w) == 0:
            return float(w) * sum(float(np.einsum('cijk,cijk->', p, q)) for p, q in pairs)
        w = np.broadcast_to(w, a.shape)
        return sum(float(np.einsum('cijk,cijk,cijk->', w, p, q)) for p, q in pairs)

    def energy_weight(self, ratio, a, b, key, times=1.0):
        """``ratio (1 + a) / (2 b)``, the ``eps`` or ``mu`` an update's ``(a, b)`` pair was built
        from (``src/wavebit/fd/_kernel/observe.py::InteriorEnergy``), times ``times``: a float for
        scalars, else written into one scratch of the coefficients' own (broadcast-compressed)
        shape, so a sample allocates nothing."""
        if np.ndim(a) == 0 and np.ndim(b) == 0 and np.ndim(times) == 0:
            return ratio * (1.0 + float(a)) / (2.0 * float(b)) * float(times)
        out = self.scratch(key, np.broadcast_shapes(np.shape(a), np.shape(b), np.shape(times)),
                           np.float64)
        np.add(a, 1.0, out=out)
        np.multiply(out, 0.5 * ratio, out=out)
        np.divide(out, b, out=out)
        if np.ndim(times) or float(times) != 1.0:
            np.multiply(out, times, out=out)
        return out

    def when(self, pred, fn, operand):
        """``fn(operand)`` if ``pred`` else ``operand``: ``pred`` is a Python bool here."""
        return fn(operand) if pred else operand

    def constant(self, x):
        """The value itself: NumPy carries no gradient to stop."""
        return x

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

    def _amplitude_keys(self, value):
        """Nonnegative IEEE-754 keys, preserving every float64 subnormal bit.

        Structural stopping gate attempt 1 measured abs/max on JAX CPU mapping 1e-323
        to zero. Float comparison in the later window merge can flush it again, so both
        reductions stay in integer space. Clearing the sign makes positive finite
        float64 values monotonically ordered int64 keys. Clamp every NaN/Inf encoding
        to +Inf, which remains a permanent failure of the concrete stopping criterion.
        This does not change field arithmetic or runtime/XLA configuration.
        """
        if value.dtype != self.xp.dtype('float64'):
            raise ValueError('DecayMetricDtype: the stopping metric requires float64 components.')
        bits = self.jax.lax.bitcast_convert_type(value, self.xp.int64)
        positive = self.xp.bitwise_and(bits, self.xp.int64(0x7fffffffffffffff))
        return self.xp.minimum(positive, self.xp.int64(0x7ff0000000000000))

    def state_amplitude(self, array):
        """Exact component maximum of the represented float64 values, including subnormals."""
        if any(n == 0 for n in array.shape):
            return self.array(0.0, 'float64')
        key = self.xp.max(self._amplitude_keys(self.xp.real(array)))
        if 'complex' in str(array.dtype):
            key = self.xp.maximum(key, self.xp.max(self._amplitude_keys(self.xp.imag(array))))
        return self.jax.lax.bitcast_convert_type(key, self.xp.float64)

    def positive_maximum(self, a, b):
        """Merge stopping-window metrics in integer space; return the selected float bits."""
        keys = self.xp.maximum(self._amplitude_keys(a), self._amplitude_keys(b))
        return self.jax.lax.bitcast_convert_type(keys, self.xp.float64)

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

    def slab_scale(self, a, slab, m):
        """``a[slab] *= m`` as a full-volume multiply by a broadcast factor that is 1 elsewhere.

        ``slab`` is a CPML slab -- full extent on two axes, contiguous on the third -- so the
        factor is ``m`` padded with ones along that axis and placed on component ``slab.c``,
        a ``(3, n, 1, 1)``-like array, and the multiply is elementwise. XLA fuses it into the
        update around it, so it costs no pass over memory. Measured in the compiled body
        (``tests/test_fd_pml.py``): ``a.at[slices].multiply(m)`` lowered to a ``scatter-mul``,
        and a dynamic-update-slice of the scaled slab -- the ``cpml_stretch`` writing -- forced
        a full-volume copy of the curl or of the masked field per call, four per step.
        """
        jnp = self.xp
        spans = (slab.z, slab.y, slab.x)
        axis = next((k for k in range(3) if m.shape[k] != 1),
                    next((k for k, n in enumerate(a.shape[1:]) if spans[k] != (0, n)), 0))
        n = a.shape[1 + axis]
        lo, hi = spans[axis]
        width = [(0, 0)] * 3
        width[axis] = (lo, n - hi)
        factor = jnp.pad(m, width, constant_values=1.0)
        factor = jnp.where(jnp.arange(3).reshape((3, 1, 1, 1)) == slab.c, factor[None], 1.0)
        return a * factor

    def cpml_stretch(self, d, region, psi, b, a, ik, key):
        """The CPML recurrence, functionally, writing ``d`` with a dynamic-update-slice.

        ``d.at[slices].add(...)`` and never the index scatter ``slab_add`` uses: see the
        module docstring for why the two are different primitives and must stay so. The
        quantity added is ``d (1/kappa - 1) + psi``, which is ``d/kappa + psi`` written as an
        increment; it differs from the fused in-place form by one rounding per cell per step
        (measured 7e-16..1e-15), which is inside every cross-backend tolerance in the suite.
        """
        view = d[region.slices]
        psi = b * psi + a * view
        return d.at[region.slices].add(view * (ik - 1.0) + psi), psi

    def record(self, acc, n, value):
        return acc.at[n].set(value)

    def dft_add(self, acc, view, phase, key):
        return acc + phase.reshape((phase.shape[0],) + (1,) * view.ndim) * view[None]

    def energy(self, view):
        return self.jax.lax.stop_gradient(self.xp.sum(self.xp.abs(view) ** 2))

    def weighted_energy(self, view, w):
        return self.jax.lax.stop_gradient(self.xp.sum(w * self.xp.abs(view) ** 2))

    def real_inner(self, a, b, w=1.0):
        return self.jax.lax.stop_gradient(self.xp.sum(w * self.xp.real(self.xp.conj(a) * b)))

    def energy_weight(self, ratio, a, b, key, times=1.0):
        return self.jax.lax.stop_gradient(ratio * (1.0 + a) / (2.0 * b) * times)

    def constant(self, x):
        return self.jax.lax.stop_gradient(x)

    def when(self, pred, fn, operand):
        """``lax.cond``: ``pred`` is traced inside the scan (it is read off the step index).

        The branch is skipped on the steps it is false, but the scan's body carries it at
        every step, so what it costs is a property of the body and not of how often it fires:
        the interior-energy reading costs the same at a stride of 16 and at a stride that
        samples twice in 320 steps (``src/wavebit/fd/_kernel/observe.py::ENERGY_STRIDE``).
        """
        return self.jax.lax.cond(pred, fn, lambda a: a, operand)

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

    def slab_scale(self, a, slab, m):
        """``slab_mul`` with a broadcasting ``m``: in place, or through ``SlabMul`` under grad."""
        return self.slab_mul(a, slab, m)

    def cpml_stretch(self, d, region, psi, b, a, ik, key):
        """The CPML recurrence in place -- four operations, no scratch -- or functionally.

        In place it is exactly the measured writing: ``psi.mul_(b).addcmul_(a, view)`` then
        ``view.mul_(ik).add_(psi)``, 1.13x against 1.18x for the gathered form at 128^3 with
        six faces.

        **Under autograd it is out of place instead, and that is a stated gap, not a
        preference.** The stage-0 study settled that a Torch in-place *slab add* needs a
        ``mark_dirty`` autograd Function whose backward slices the cotangent; a CPML psi is
        read as well as written, so that Function's backward is a different one and it was
        not measured. Writing in place without it would be wrong (autograd would see a
        version-counter violation or, worse, differentiate the wrong graph); writing with an
        unmeasured one would be a claim this package has no number for. So a differentiated
        Torch run takes the functional form and pays a full-volume copy per stretch per step,
        and measuring the Function is the first thing stage 1 owes stage 6.
        """
        view = d[region.slices]
        if self._live(d, psi, b, a, ik):
            psi = b * psi + a * view
            out = d.clone()                      # the full-volume copy the gap costs
            out[region.slices] = view * ik + psi
            return out, psi
        psi.mul_(b).addcmul_(a, view)
        view.mul_(ik).add_(psi)
        return d, psi

    def record(self, acc, n, value):
        acc[n] = value
        return acc

    def dft_add(self, acc, view, phase, key):
        return acc + phase.reshape((phase.shape[0],) + (1,) * view.ndim) * view[None]

    def energy(self, view):
        with self.torch.no_grad():
            return (view.abs() ** 2).sum()

    def weighted_energy(self, view, w):
        with self.torch.no_grad():
            return (w * view.abs() ** 2).sum()

    def real_inner(self, a, b, w=1.0):
        with self.torch.no_grad():
            return (w * ((a.conj() * b).real if a.is_complex() else a * b.real)).sum()

    def energy_weight(self, ratio, a, b, key, times=1.0):
        with self.torch.no_grad():
            return ratio * (1.0 + a) / (2.0 * b) * times

    def constant(self, x):
        return x.detach() if isinstance(x, self.torch.Tensor) else x

    def when(self, pred, fn, operand):
        """``fn(operand)`` if ``pred`` else ``operand``: ``pred`` is a Python bool here."""
        return fn(operand) if pred else operand

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
            # A block works on copies of the state it is handed. The checkpoint keeps those
            # inputs to recompute the block in backward, and the step writes in place
            # wherever nothing is differentiated yet -- a CPML psi before the field is live,
            # a series accumulator -- so without the copy the first block's own writes (or
            # the next block's) moved an input the recompute reads, and backward raised a
            # version-counter error on every source run and on a stage-0 ``fd.Record``. One
            # copy of the state per block, not per step.
            c = rebuild([x.clone() if hasattr(x, 'clone') else x for x in flat])
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
