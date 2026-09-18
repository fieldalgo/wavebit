"""``run(scene, backend=, plan=None, device=None) -> Results``, and the executor under it.

``run`` is three calls in a row -- ``plan``, ``lower``, ``execute`` -- and one refusal: a
``plan=`` whose skeleton is not the scene's. The executor builds the initial state from the
params, runs the runtime's ``scan`` over the kernel step and returns the final state. On
JAX the whole executor is jitted with the plan closed over and the params as the argument,
so a ``jax.grad`` around ``run`` reaches every leaf of the params; it is compiled once per
plan and kept on the runtime, which ``runtime()`` hands out once per backend and device, so
a second ``run`` of the same scene -- a sweep, a gradient loop -- pays no compile. On Torch
the executor is eager and autograd follows the tensors; on NumPy it is a loop over
preallocated buffers.
"""
from ._check import report as _report
from ._kernel.step import initial_state, step
from ._lower import lower, plan as _plan
from ._runtime import runtime
from .results import Final, Results, Series

__all__ = ['run', 'execute']


def execute(plan, params, rt):
    """The final state of ``plan`` driven by ``params`` on ``rt``; jitted once per plan on JAX."""
    def make():
        def go(traced):
            def body(state, n):
                return step(plan, traced, state, n, rt)
            state = initial_state(plan, traced, rt)
            return rt.scan(body, state, plan.steps, plan.schedule)
        return go

    return rt.jit_for(plan, make)(rt.to_device(params))


def _results(scene, plan, state, rt):
    series = {}
    for o in plan.observers:
        series[o.name] = Series(name=o.name, component=f'{o.field}{"xyz"[o.c]}',
                                t=o.times(plan), values=state['obs'][o.name])
    return Results(series, Final(E=state['E'], H=state['H']), plan, _report(scene, plan), rt.name)


def run(scene, backend='numpy', plan=None, device=None):
    """Run ``scene`` on ``backend`` (``'numpy'`` | ``'jax'`` | ``'torch'`` or ``wavebit.Backend``)."""
    p = _plan(scene)
    if plan is not None:
        if plan.skeleton != p.skeleton:
            raise ValueError(
                f'run(scene, plan=...): the plan was built for a different scene -- its '
                f'skeleton {plan.skeleton[:12]} is not the scene\'s {p.skeleton[:12]}. Build it '
                f'with fd.plan(scene), or pass plan=None.')
        p = plan
    rt = runtime(backend, device)
    params = lower(scene, p, rt)
    state = execute(p, params, rt)
    return _results(scene, p, state, rt)
