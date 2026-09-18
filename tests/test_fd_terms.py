"""Gates 6 and 7 -- the term slots: Torch's in-place order contract, the permutation, the lowering.

Stage 0 has no term of its own, so the two slots the kernel step exposes (``before_H`` adds
into ``curl E``, ``before_E`` into ``curl H``) are driven here by terms that exist only for
tests: the ``SlabTerm`` hook in ``wavebit.fd._testing`` and a few defined below. Three
things are held:

* **Gate 6, the Torch order contract.** On Torch a slab add is an in-place ``add_`` under a
  ``mark_dirty`` autograd Function (``wavebit/fd/_runtime.py``), which is only sound when the
  tensor written has not yet been consumed by an operation that saved it for backward. A
  reader term that reads a view of ``curl H`` into the loss, followed by a writer term that
  slab-adds into the same array, must therefore raise ``RuntimeError`` when the run is
  differentiated with respect to the initial amplitude -- the mutant. With the writer first
  the run differentiates, and its gradient equals an out-of-place reference (the same value
  added as a full-volume pattern) and JAX's functional gradient of the same plan, to 1e-12.
  The reader-first order is not an error on NumPy or JAX, and not on Torch without autograd:
  it is a different computation (the reader sees the curl before the add), which is why the
  contract is stated for a differentiated run and why a reader and a writer of one slab are
  not a commuting pair.

* **Gate 7, the permutation.** Two terms of one slot must commute: a pair with disjoint
  slabs, and a pair split across the two slots (the term list never reorders the slots),
  give bit-identical fields and series in either order on every backend. A pair whose slabs
  overlap additively commutes to round-off only -- ``(x + a) + b`` and ``(x + b) + a`` differ by
  an ulp -- and is held to 1e-13 relative (measured 1.8e-15 over 64 steps, worst over the
  fields and the probes). The mutant is a term that overwrites its slab instead of adding
  (a zero then an add): against an additive term on overlapping cells the two orders differ
  by 0.63, so the tolerance is doing work.

* **The JAX lowering.** ``slab_add`` on JAX must lower to a scatter on the 4-D carry inside
  the scan body and never to a dynamic-update-slice, which costs one to two full-volume
  passes per term unless the slab's origin happens to be aligned (``studies/fdtd_slab_add_cost.py``
  measured both). The compiled HLO of the executor is read the way the study reads it: the
  ``while`` body and every computation it calls, counted for ``scatter`` and
  ``dynamic-update-slice`` instructions on the full-volume shape. A periodic box carries no
  mask, so its baseline body has neither; a slab term adds exactly one scatter, sorted and
  unique, and no full-volume copy. The mutant is the slice form ``a.at[slices].add``, which
  the same inspection sees as a dynamic-update-slice. A ``Record`` is the one legitimate
  dynamic-update-slice: its ``(steps,)`` accumulator, never the volume.
"""
import dataclasses
import re

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd import _runtime, _testing
from wavebit.fd._lower import lower
from wavebit.fd._run import execute
from wavebit.fd._runtime import runtime
from fd_helpers import backends, cavity_box, plane_wave_box, to_numpy

Slab, SlabTerm = _testing.Slab, _testing.SlabTerm

GRAD_TOL = 1e-12         # in-place gradient against the out-of-place reference, relative
ROUNDOFF = 1e-13         # overlapping additive terms in either order, relative (measured 1.8e-15)
MUTANT_FLOOR = 1e-3      # the non-additive pair in either order, relative (measured 0.63)

#: The slab gate 6 reads and writes: interior cells of Ey, off every wall of an 8^3 box.
SLAB = Slab(c=1, z=(2, 5), y=(2, 5), x=(2, 5))
#: Gate 7's slabs: ``A`` and ``B_DISJOINT`` share no cell; ``B_OVERLAP`` shares 8 with ``A``.
A = Slab(c=1, z=(1, 4), y=(1, 4), x=(1, 4))
B_DISJOINT = Slab(c=2, z=(3, 6), y=(3, 6), x=(3, 6))
B_OVERLAP = Slab(c=1, z=(2, 6), y=(2, 6), x=(2, 6))


# ---------------------------------------------------------------------------------------------
# Test-only terms. Each is a frozen dataclass of static values, as the Term protocol asks,
# and lives in the plan (and its skeleton) like the kernel's own terms will.
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Reader:
    """Reads a view of ``curl H`` over ``slab`` and folds its energy into the term's memory.

    The memory is a one-element list holding the running sum, so that ``before_E`` -- which
    returns only ``cH`` -- can update it; the list is the term's own state entry and the run
    carries it like any memory. The read is a multiply, which on Torch saves its operand
    (the view) for backward: that saved view is what an in-place write after it trips over.
    """

    name: str
    slab: Slab

    def init_mem(self, plan, rt):
        return [rt.zeros((), plan.dtype)]

    def before_H(self, cE, E, mem, n, params, rt):
        return cE

    def before_E(self, cH, H, mem, n, params, rt):
        v = cH[self.slab.slices]
        mem[0] = mem[0] + (v * v).sum()
        return cH

    def advance(self, mem, E, H, n, params, rt):
        return mem


@dataclasses.dataclass(frozen=True)
class ArrayWriter:
    """``slab_add`` of ``value`` carried as an array of the runtime, as a source's amplitude is.

    ``SlabTerm`` passes its value as a Python float; this term hands the primitive a leaf of
    the backend instead (built once, in ``init_mem``), which is the common case for a term
    whose value is traced.
    """

    name: str
    slab: Slab
    value: float

    def init_mem(self, plan, rt):
        return rt.array(self.value, 'float64')

    def before_H(self, cE, E, mem, n, params, rt):
        return cE

    def before_E(self, cH, H, mem, n, params, rt):
        return rt.slab_add(cH, self.slab, mem)

    def advance(self, mem, E, H, n, params, rt):
        return mem


@dataclasses.dataclass(frozen=True)
class DenseWriter:
    """The out-of-place reference: ``cH + pattern``, ``value`` on the slab and zero elsewhere."""

    name: str
    slab: Slab
    value: float

    def init_mem(self, plan, rt):
        pattern = np.zeros(plan.shape)
        pattern[self.slab.slices] = self.value
        return rt.array(pattern, 'float64')

    def before_H(self, cE, E, mem, n, params, rt):
        return cE

    def before_E(self, cH, H, mem, n, params, rt):
        return cH + mem

    def advance(self, mem, E, H, n, params, rt):
        return mem


@dataclasses.dataclass(frozen=True)
class SetTerm:
    """Gate 7's mutant: overwrites its slab of ``curl H`` with ``value`` (a zero, then an add)."""

    name: str
    slab: Slab
    value: float

    def init_mem(self, plan, rt):
        return None

    def before_H(self, cE, E, mem, n, params, rt):
        return cE

    def before_E(self, cH, H, mem, n, params, rt):
        return rt.slab_add(rt.slab_mul(cH, self.slab, 0.0), self.slab, self.value)

    def advance(self, mem, E, H, n, params, rt):
        return mem


@dataclasses.dataclass(frozen=True)
class SliceAddTerm:
    """The lowering mutant: the slice form the design forbids on JAX, ``a.at[slices].add``."""

    name: str
    slab: Slab
    value: float

    def init_mem(self, plan, rt):
        return None

    def before_H(self, cE, E, mem, n, params, rt):
        return cE

    def before_E(self, cH, H, mem, n, params, rt):
        return cH.at[self.slab.slices].add(self.value)

    def advance(self, mem, E, H, n, params, rt):
        return mem


# ---------------------------------------------------------------------------------------------
# Driving the kernel with terms
# ---------------------------------------------------------------------------------------------

def _final_state(scene, terms, backend, amplitude=None):
    """The final kernel state ``(E, H, mem, obs)`` of ``scene`` with ``terms`` injected.

    ``run_with_terms`` returns ``Results``, which drop the term memory; the reader's sum lives
    there, so this goes one level down. ``amplitude`` replaces the initial field's, so a
    backend scalar that requires a gradient can be threaded through ``lower``.
    """
    if amplitude is not None:
        scene = scene.replace(initial=scene.initial.replace(amplitude=amplitude))
    plan = _testing.with_terms(fd.plan(scene), terms)
    rt = runtime(backend)
    return execute(plan, lower(scene, plan, rt), rt)


def _loss(state):
    """The final E energy plus what the reader accumulated: both reach every term's write."""
    return (state['E'] ** 2).sum() + state['mem']['reader'][0]


def _torch_gradient(scene, terms):
    """``d loss / d amplitude`` by ``torch.autograd``, and the final state it came from."""
    torch = pytest.importorskip('torch')
    amp = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    state = _final_state(scene, terms, 'torch', amp)
    return torch.autograd.grad(_loss(state), amp)[0].item(), state


def _jax_gradient(scene, terms):
    """The same derivative by ``jax.grad``, through the jitted executor."""
    jax = pytest.importorskip('jax')
    plan = _testing.with_terms(fd.plan(scene), terms)
    rt = runtime('jax')

    def loss(a):
        s = scene.replace(initial=scene.initial.replace(amplitude=a))
        return _loss(execute(plan, lower(s, plan, rt), rt))
    return float(jax.grad(loss)(1.0))


def _both_orders(scene, pair, backend):
    """The results of ``pair`` and of ``pair`` reversed, on ``backend``."""
    return (_testing.run_with_terms(scene, list(pair), backend=backend),
            _testing.run_with_terms(scene, list(pair[::-1]), backend=backend))


def _compared(r1, r2):
    """``(final E, final H, every series)`` of two results, as NumPy pairs."""
    pairs = [(r1.final.E, r2.final.E), (r1.final.H, r2.final.H)]
    pairs += [(r1[name].values, r2[name].values) for name in r1]
    return [(to_numpy(a), to_numpy(b)) for a, b in pairs]


def _identical(r1, r2):
    return all(np.array_equal(a, b) for a, b in _compared(r1, r2))


def _worst_relative(r1, r2):
    return max(float(np.max(np.abs(a - b)) / np.max(np.abs(a))) for a, b in _compared(r1, r2))


@pytest.fixture(scope='module')
def box():
    """Gate 6's box: the PEC eigenmode, small and short -- the gate is about order, not physics."""
    return cavity_box('pec', N=8, steps=20)[0]


@pytest.fixture(scope='module')
def box7():
    """Gate 7's box: the same, run long enough for an order difference to show in the probes."""
    return cavity_box('pec', N=8, steps=64)[0]


# ---------------------------------------------------------------------------------------------
# Gate 6: the Torch order contract
# ---------------------------------------------------------------------------------------------

def test_gate6_mutant_a_view_read_before_an_in_place_slab_add_raises_under_autograd(box):
    """Reader first, writer second: the write lands on a tensor a saved view belongs to."""
    reader, writer = Reader('reader', SLAB), ArrayWriter('writer', SLAB, 0.3)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        _torch_gradient(box, [reader, writer])


def test_gate6_the_write_before_the_read_gives_the_out_of_place_gradient(box):
    """Writer first: the fresh curl is written before anything reads it; the gradient is right."""
    reader = Reader('reader', SLAB)
    g_in_place, in_place = _torch_gradient(box, [ArrayWriter('writer', SLAB, 0.3), reader])
    g_reference, reference = _torch_gradient(box, [DenseWriter('writer', SLAB, 0.3), reader])
    assert abs(g_in_place - g_reference) <= GRAD_TOL * abs(g_reference)
    assert abs(g_reference) > 1.0, 'the loss does not depend on the amplitude'
    # and the forward pass agrees too: the same fields, the same reading
    E1, E2 = to_numpy(in_place['E']), to_numpy(reference['E'])
    assert np.max(np.abs(E1 - E2)) <= GRAD_TOL * np.max(np.abs(E2))
    m1, m2 = to_numpy(in_place['mem']['reader'][0]), to_numpy(reference['mem']['reader'][0])
    assert abs(m1 - m2) <= GRAD_TOL * abs(m2)


def test_gate6_the_in_place_gradient_equals_the_functional_one_on_jax(box):
    """The same plan on JAX is functional throughout; both backends run one kernel body."""
    terms = [ArrayWriter('writer', SLAB, 0.3), Reader('reader', SLAB)]
    g_torch, _ = _torch_gradient(box, terms)
    g_jax = _jax_gradient(box, terms)
    assert abs(g_jax - g_torch) <= GRAD_TOL * abs(g_torch)


@pytest.mark.parametrize('backend', backends())
def test_gate6_a_reader_and_a_writer_of_one_slab_do_not_commute(box, backend):
    """Reader first reads the curl before the add, on every backend alike; only autograd objects."""
    reader, writer = Reader('reader', SLAB), ArrayWriter('writer', SLAB, 0.3)
    before = float(to_numpy(_final_state(box, [reader, writer], backend)['mem']['reader'][0]))
    after = float(to_numpy(_final_state(box, [writer, reader], backend)['mem']['reader'][0]))
    assert abs(before - after) > MUTANT_FLOOR * abs(after)
    # the three backends agree on both readings: the order is a computation, not an accident
    before_np = float(_final_state(box, [reader, writer], 'numpy')['mem']['reader'][0])
    after_np = float(_final_state(box, [writer, reader], 'numpy')['mem']['reader'][0])
    assert abs(before - before_np) <= GRAD_TOL * abs(before_np)
    assert abs(after - after_np) <= GRAD_TOL * abs(after_np)


def test_gate6_a_static_float_slab_value_is_differentiable_through(box):
    """The kernel's own hook term, correct order: its gradient must equal the dense reference.

    ``SlabTerm`` passes a Python float, so ``SlabAdd.backward`` must return ``None`` for the
    value's gradient (``ctx.needs_input_grad[1]`` is False); returning one made Torch raise
    "returned a gradient different than None at position 2" on every static term (fixed
    2026-09-18).
    """
    reader = Reader('reader', SLAB)
    g_static, _ = _torch_gradient(box, [SlabTerm('writer', SLAB, 0.3, phase='before_E'), reader])
    g_reference, _ = _torch_gradient(box, [DenseWriter('writer', SLAB, 0.3), reader])
    assert abs(g_static - g_reference) <= GRAD_TOL * abs(g_reference)


def test_gate6_mutant_with_the_kernels_own_hook_term_raises_the_in_place_error(box):
    reader, writer = Reader('reader', SLAB), SlabTerm('writer', SLAB, 0.3, phase='before_E')
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        _torch_gradient(box, [reader, writer])


def test_gate6_the_torch_slab_writes_are_the_mark_dirty_functions_under_autograd():
    """The contract itself, pinned: under autograd a slab write is the ``autograd.Function``
    whose forward marks the tensor dirty (``grad_fn`` is its backward, the storage is the
    input's); without autograd it is the bare in-place op on the same object.

    A bare ``add_`` under autograd would pass every other gate-6 test -- it raises the same
    version-counter error on the reader-first order and gives the same gradient -- at one
    full-volume cotangent copy per term per step in backward (measured in
    ``studies/fdtd_slab_add_cost.py``), which is what this contract exists to forbid.
    """
    torch = pytest.importorskip('torch')
    rt = runtime('torch')
    a = torch.zeros((3, 4, 4, 4), dtype=torch.float64, requires_grad=True)
    v = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    for primitive, backward in (('slab_add', 'SlabAddBackward'), ('slab_mul', 'SlabMulBackward')):
        b = a * 1.0
        out = getattr(rt, primitive)(b, SLAB, v)
        assert type(out.grad_fn).__name__ == backward
        assert out.data_ptr() == b.data_ptr()
        with torch.no_grad():
            c = a.detach().clone()
            assert getattr(rt, primitive)(c, SLAB, 0.3) is c


# ---------------------------------------------------------------------------------------------
# Gate 7: the permutation
# ---------------------------------------------------------------------------------------------

TERM_A = SlabTerm('A', A, 0.3, phase='before_E')


@pytest.mark.parametrize('backend', backends())
def test_gate7_disjoint_slab_terms_commute_bit_for_bit(box7, backend):
    pair = (TERM_A, SlabTerm('B', B_DISJOINT, -0.7, phase='before_E'))
    r1, r2 = _both_orders(box7, pair, backend)
    assert np.max(np.abs(to_numpy(r1.final.E))) > 0.0
    assert _identical(r1, r2)


@pytest.mark.parametrize('backend', backends())
def test_gate7_terms_in_different_slots_commute_bit_for_bit(box7, backend):
    """Overlapping cells, but one term adds into ``curl E`` and the other into ``curl H``."""
    pair = (TERM_A, SlabTerm('B', B_OVERLAP, -0.7, phase='before_H'))
    r1, r2 = _both_orders(box7, pair, backend)
    assert _identical(r1, r2)


@pytest.mark.parametrize('backend', backends())
def test_gate7_overlapping_additive_terms_commute_to_roundoff(box7, backend):
    """``(x + a) + b`` against ``(x + b) + a``: an ulp per shared cell per step, never more."""
    pair = (TERM_A, SlabTerm('B', B_OVERLAP, -0.7, phase='before_E'))
    r1, r2 = _both_orders(box7, pair, backend)
    assert _worst_relative(r1, r2) < ROUNDOFF


@pytest.mark.parametrize('backend', backends())
def test_gate7_mutant_a_non_additive_overlap_is_detected(box7, backend):
    """An overwrite of cells another term adds into: the two orders are two different runs."""
    pair = (TERM_A, SetTerm('S', B_OVERLAP, -0.7))
    r1, r2 = _both_orders(box7, pair, backend)
    assert _worst_relative(r1, r2) > MUTANT_FLOOR


# ---------------------------------------------------------------------------------------------
# The JAX lowering: a scatter in the scan body, never a dynamic-update-slice
# ---------------------------------------------------------------------------------------------

def _compiled(scene, terms):
    """The optimized HLO of the executor JAX compiles for ``scene`` with ``terms``, and the plan.

    The runtime's ``jit`` is wrapped on a private instance (``runtime('jax')`` hands out the
    shared one, which must not carry the wrapper) so that the program the executor really
    runs is the one lowered -- the same arguments, the same closure over the plan.
    """
    jax = pytest.importorskip('jax')
    rt = _runtime.JaxRuntime()
    captured = {}

    def jit(fn):
        compiled = jax.jit(fn)

        def wrapped(*args):
            captured['lowered'] = compiled.lower(*args)
            return compiled(*args)
        return wrapped
    rt.jit = jit
    plan = _testing.with_terms(fd.plan(scene), terms)
    execute(plan, lower(scene, plan, rt), rt)
    return captured['lowered'].compile().as_text(), plan


def _scan_body(text):
    """The scan's ``while`` body and every computation it calls, recursively, as one text."""
    blocks = dict(re.findall(r'^(?:ENTRY )?(\S+) \([^\n]*\{\n(.*?)\n\}', text, re.M | re.S))
    todo, seen, parts = re.findall(r'body=(\S+?)[,\s]', text), set(), []
    assert todo, 'no while loop: the scan did not lower to one'
    while todo:
        name = todo.pop()
        if name in seen or name not in blocks:
            continue
        seen.add(name)
        parts.append(blocks[name])
        todo.extend(re.findall(r'(?:calls|to_apply)=(\S+?)[,\s)]', blocks[name]))
    return '\n'.join(parts)


def _counts(body, plan):
    """Instructions in the body: every scatter and DUS, and those on the full-volume shape."""
    full = r'\w+\[3,%d,%d,%d\]\{[^}]*\} ' % (plan.nz, plan.ny, plan.nx)
    return dict(scatter=body.count(' scatter('), dus=body.count(' dynamic-update-slice('),
                full_scatter=len(re.findall(full + r'scatter\(', body)),
                full_dus=len(re.findall(full + r'dynamic-update-slice\(', body)),
                full_copy=len(re.findall(full + r'copy\(', body)))


@pytest.fixture(scope='module')
def periodic():
    """A periodic box with no monitor: no mask, no accumulator, so nothing scatters but a term."""
    return plane_wave_box(N=8, steps=10)[0].replace(monitors=())


def test_the_jax_slab_add_lowers_to_a_scatter_in_the_scan_body(periodic):
    text, plan = _compiled(periodic, [])
    base = _counts(_scan_body(text), plan)
    assert base['scatter'] == 0 and base['dus'] == 0
    terms = [SlabTerm('e', SLAB, 0.3, phase='before_E'), SlabTerm('h', A, -0.7, phase='before_H')]
    text, plan = _compiled(periodic, terms)
    body = _scan_body(text)
    got = _counts(body, plan)
    assert got['full_scatter'] == 2, got            # one scatter per term, on the 4-D carry
    assert got['scatter'] == 2, got
    assert got['dus'] == 0, got                     # no dynamic-update-slice anywhere in the body
    assert got['full_copy'] == base['full_copy'], got   # and no full-volume copy the baseline lacks
    assert body.count('indices_are_sorted=true, unique_indices=true') == 2


def test_mutant_a_slice_add_lowers_to_a_dynamic_update_slice(periodic):
    """The forbidden form, seen by the same inspection: the check can fail."""
    text, plan = _compiled(periodic, [SliceAddTerm('t', SLAB, 0.3)])
    got = _counts(_scan_body(text), plan)
    assert got['full_dus'] == 1 and got['scatter'] == 0, got


def test_a_records_accumulator_is_the_only_dynamic_update_slice(periodic):
    """With monitors on, the body's dynamic-update-slices are the ``(steps,)`` accumulators."""
    scene = periodic.replace(monitors=(fd.Record('Ey', at=(0.1, 0.0, -0.05), component='Ey'),
                                       fd.Record('Hx', at=(0.1, 0.0, -0.05), component='Hx')))
    text, plan = _compiled(scene, [SlabTerm('e', SLAB, 0.3, phase='before_E')])
    body = _scan_body(text)
    got = _counts(body, plan)
    assert got['full_scatter'] == 1 and got['full_dus'] == 0, got
    assert got['dus'] == 2, got
    assert len(re.findall(r'\w+\[%d\]\{0\} dynamic-update-slice\(' % plan.steps, body)) == 2


def test_the_wall_masks_lower_to_scatters_too(box):
    """A PEC box's masks are ``slab_mul`` calls: the same index form, the same lowering."""
    text, plan = _compiled(box.replace(monitors=()), [])
    got = _counts(_scan_body(text), plan)
    assert plan.masks, 'a PEC box has masks'
    assert got['full_scatter'] == len(plan.masks) and got['dus'] == 0, (got, len(plan.masks))
