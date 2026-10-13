"""Test hooks: run a scene with extra terms injected into its plan, without touching the kernel.

Stage 0 has no term of its own, so the slots the step exposes are exercised by terms that
exist only for tests: ``SlabTerm`` adds a static pattern times an amplitude into one slab
of the curl each step, in the slot it names. Two of them commute, which is what the
permutation gate checks; one whose ``before_E`` reads a view of ``cH`` and is followed by an
in-place add is the Torch order mutant.

``run_with_terms(scene, terms, backend=...)`` is ``fd.run`` with the terms appended to the
plan. It bypasses ``fd.run``'s skeleton check on purpose: a plan with extra terms has a
different skeleton from ``plan(scene)``, and that refusal is right for users and wrong for
a test that is deliberately driving the kernel with something the scene does not declare.
"""
import dataclasses

from wavebit.fdtd._kernel.masks import Slab
from wavebit.fdtd._prepare import lower
import wavebit.fdtd._prepare.plan as _plan
from wavebit.fdtd._execution import _results
from wavebit.fdtd._execution import execute
from wavebit.fdtd._runtime import runtime

__all__ = ['SlabTerm', 'with_terms', 'run_with_terms', 'Slab']


@dataclasses.dataclass(frozen=True)
class SlabTerm:
    """Add ``amplitude`` (a static float) into ``slab`` of the curl in slot ``phase`` each step.

    ``phase`` is ``'before_H'`` (adds into ``curl E``) or ``'before_E'`` (into ``curl H``).
    ``value`` is a static scalar; the term carries no memory. It is a frozen dataclass of
    primitives, so it lives in the plan and its skeleton.
    """

    name: str
    slab: Slab
    value: float
    phase: str = 'before_E'

    def init_mem(self, plan, rt):
        return None

    def before_H(self, cE, E, mem, n, params, rt):
        if self.phase == 'before_H':
            return rt.slab_add(cE, self.slab, self.value)
        return cE

    def before_E(self, cH, H, mem, n, params, rt):
        if self.phase == 'before_E':
            return rt.slab_add(cH, self.slab, self.value)
        return cH

    def advance(self, mem, E, H, n, params, rt):
        return mem


def with_terms(plan, terms):
    """``plan`` with ``terms`` appended and its skeleton recomputed."""
    names = [t.name for t in tuple(plan.terms) + tuple(terms)]
    if len(set(names)) != len(names):
        raise ValueError(f'term names must be unique: {names}')
    return plan.with_static(terms=tuple(plan.terms) + tuple(terms))


def run_with_terms(scene, terms, backend='numpy', device=None, plan=None):
    """``fd.run`` with ``terms`` injected into the plan (no skeleton check; see the module)."""
    p = with_terms(_plan(scene) if plan is None else plan, terms)
    rt = runtime(backend, device)
    params = lower(scene, p, rt)
    return _results(scene, p, execute(p, params, rt), rt)
