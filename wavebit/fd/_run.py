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

**Stage 3** turns the finished state into results after the scan (``_assemble.py``): each
frequency-domain monitor's result, each source's spectral normalisation (methods on its
``SourceSeries``), and the always-on decay and growth diagnostics. A
``Diffraction(polarizations='both')`` monitor makes ``run`` solve the scene a second time with
the PlaneWave's other polarisation; ``res.diagnostics['runs']`` says how many solves the
result is.
"""
import dataclasses

import numpy as np

from ._check import report as _report
from ._kernel.sources import has_sources, waveform_table
from ._kernel.step import initial_state, step
from ._lower import lower, plan as _plan
from ._runtime import runtime
from .monitors import Diffraction, Record
from .results import Final, Results, Series

__all__ = ['run', 'execute']


def _program(plan):
    """``plan`` as the step reads it: every digested array field of every term set to ``None``.

    A field declared ``metadata={'digest': True}`` is static data the skeleton records by
    content -- a prepared extended-stencil operator's ``index``, ``value``, ``output`` and
    ``diagonal`` -- and the step never reads it: what the kernel multiplies by reaches it
    through the params (``params['stencil']``, and the diagonal patched into ``be``). But the
    executor is compiled once per plan on JAX and kept, by a runtime ``runtime()`` keeps for the
    life of the process, and both the cache key and the compiled closure held the plan -- so
    after one ``fd.run(scene, 'jax')`` its operator was never released. Measured on
    ``199463e`` (review 1): a three-radius JAX sweep with the caller keeping nothing left 1, 2
    and 3 live operators (2.4, 4.9 and 7.5 MB on a 20-cell fixture; about 77 MB each at 56
    cells), where NumPy and Torch left none.

    Stripping the arrays here, and keying and closing over what is left, keeps the program
    and releases the data. Equality is unchanged: the skeleton is kept, and it holds the
    arrays' digests, so two stripped plans are equal exactly when the plans were -- which is
    what "equal plans mean equal programs" needs. Holding the plan weakly instead cannot work:
    the compiled closure has to reference the plan it steps, so a weak-keyed entry keeps its
    own key alive. A plan whose terms carry no digested field -- every default scene -- is
    returned as the same object, so its key and its closure are what they were.
    """
    terms = tuple(_stripped(t) for t in plan.terms)
    junctions = tuple(_stripped(r) for r in plan.junctions)
    if (all(a is b for a, b in zip(terms, plan.terms))
            and all(a is b for a, b in zip(junctions, plan.junctions))):
        return plan
    return dataclasses.replace(plan, terms=terms, junctions=junctions)


def _stripped(term):
    if not dataclasses.is_dataclass(term):
        return term
    marked = {f.name: None for f in dataclasses.fields(term)
              if f.metadata.get('digest') and getattr(term, f.name) is not None}
    return dataclasses.replace(term, **marked) if marked else term


def execute(plan, params, rt):
    """The final state of ``plan`` driven by ``params`` on ``rt``; jitted once per plan on JAX.

    The program is :func:`_program` of the plan -- the plan without its digested arrays -- and
    that is also the compile cache's key, so a prepared operator lives exactly as long as the
    caller's plan does, on JAX too.
    """
    plan = _program(plan)

    def make():
        def go(traced):
            if has_sources(plan):
                # Built here, inside the traced function, from traced leaves: one table per
                # run, so the amplitude, phase, delay and band differentiate.
                traced = dict(traced, waveform=waveform_table(plan, traced, rt))

            def body(state, n):
                return step(plan, traced, state, n, rt)
            state = initial_state(plan, traced, rt)
            return rt.scan(body, state, plan.steps, plan.schedule)
        return go

    return rt.jit_for(plan, make)(rt.to_device(params))


@dataclasses.dataclass(frozen=True)
class _Context:
    """What a source's spectral methods need from the run: the scene, plan, term, runtime."""

    scene: object
    plan: object
    term: object
    rt: object
    wavelengths: tuple = None


def _shared_wavelengths(scene):
    lists = {m.wavelengths for m in getattr(scene, 'monitors', ()) if not isinstance(m, Record)}
    return lists.pop() if len(lists) == 1 else None


def _results(scene, plan, state, rt, params=None, grazing=None):
    series = {}
    for o in plan.observers:
        if not hasattr(o, 'field'):
            continue                                    # a DFTRegion: assembled below
        series[o.name] = Series(name=o.name, component=f'{o.field}{"xyz"[o.c]}',
                                t=o.times(plan), values=state['obs'][o.name])
    wavelengths = _shared_wavelengths(scene)
    sources = {}
    for t in plan.terms:
        if not hasattr(t, 'series'):
            continue
        s = t.series(plan, state['mem'][t.name], rt)
        sources[t.name] = dataclasses.replace(
            s, context=_Context(scene=scene, plan=plan, term=t, rt=rt, wavelengths=wavelengths))
    from ._assemble import decay_report, monitor_results
    if any(not isinstance(m, Record) for m in getattr(scene, 'monitors', ())):
        series.update(monitor_results(scene, plan, state, rt, sources, params))
        # In scene order, so ``for name in res`` reads the monitors as they were declared.
        series = {m.name: series[m.name] for m in scene.monitors}
    # Always on: every run reports its decay and its growth; only a frequency-domain monitor
    # warns unsettled, and a growth warns for every monitor that sees it (a Record too, where
    # its values are concrete) or, when none does, for the interior.
    stable = _stable(params, rt)
    diagnostics = decay_report(scene, plan, state, rt, sources,
                               stable=stable if isinstance(stable, bool) else None)
    diagnostics['runs'] = 1
    diagnostics['stable'] = stable
    # The survey the plan's refusal judged (``run`` hands it over), else surveyed here.
    if grazing is None:
        from ._lower.grazing import survey
        grazing = survey(scene, plan)
    diagnostics['grazing'] = grazing
    return Results(series, Final(E=state['E'], H=state['H']), plan,
                   _report(scene, plan, params), rt.name, sources=sources,
                   diagnostics=diagnostics)


def _stable(params, rt):
    """Whether the run was inside its stability bound: ``True``, ``False``, or ``None``.

    ``True`` when the bound was checked concretely at the door (no guard was built); otherwise
    the guard the lowering built (``params['stable']``, ``1`` or ``nan``) read back -- ``False``
    for a traced material past the CFL bound or a traced ``Axis1D`` ``k`` past its ``k_max``, in
    which case every returned array is ``nan``. Under a JAX trace (``jax.jit``, ``jax.grad``) the
    guard has no value yet, and this is the traced boolean itself, ``isfinite(guard)`` -- a
    jitted function can return it.
    """
    guard = None if params is None else params.get('stable')
    if guard is None:
        return True
    try:
        value = float(np.asarray(rt.numpy(guard)))
    except Exception:
        return rt.xp.isfinite(guard)
    return bool(value == value)


def _solve(scene, p, rt, grazing=None):
    params = lower(scene, p, rt)
    state = execute(p, params, rt)
    return _results(scene, p, state, rt, params, grazing)


def _planned(scene, reuse=None):
    """``(plan(scene), its grazing survey or None)``: the rows its refusal judged, once.

    ``reuse`` is a plan the caller already paid for: a prepared extended-stencil operator it
    carries is taken as it is when this scene would prepare the same one
    (``wavebit/fd/_lower/stencil.py::prepare``), instead of being solved again on the host --
    which on the test fixture was 3 of a 4 second run, and 493 s on the 56-cell research
    sphere. Every door still runs, and the skeleton comparison below still decides.
    """
    rows = []
    p = _plan(scene, _grazing=rows, _reuse=reuse)
    return p, (rows[0] if rows else None)


def _other_polarization(scene):
    """``scene`` with its PlaneWave's other polarisation (a Diffraction 'both' second run)."""
    from .sources import PlaneWave
    sources = tuple(s.replace(polarization='p' if s.polarization == 's' else 's')
                    if isinstance(s, PlaneWave) else s for s in scene.sources)
    return scene.replace(sources=sources)


def run(scene, backend='numpy', plan=None, device=None):
    """Run ``scene`` on ``backend`` (``'numpy'`` | ``'jax'`` | ``'torch'`` or ``wavebit.Backend``)."""
    # The survey is always this scene's: from its own plan, even when the caller passes one
    # (a plan with the same skeleton may be another scene's, whose materials it does not hold).
    p, rows = _planned(scene, reuse=plan)
    if plan is not None:
        if plan.skeleton != p.skeleton:
            raise ValueError(
                f'run(scene, plan=...): the plan was built for a different scene -- its '
                f'skeleton {plan.skeleton[:12]} is not the scene\'s {p.skeleton[:12]}. Build it '
                f'with fd.plan(scene), or pass plan=None.')
        p = plan
    rt = runtime(backend, device)
    res = _solve(scene, p, rt, rows)
    both = [m for m in scene.monitors
            if isinstance(m, Diffraction) and m.polarizations == 'both']
    if both:
        # The second solve: the same scene with the PlaneWave's other polarisation. Its own
        # plan (the column's drive weights are static), its own decay check.
        other = _other_polarization(scene)
        p2, rows2 = _planned(other, reuse=p)
        res2 = _solve(other, p2, rt, rows2)
        for m in both:
            res._series[m.name] = res[m.name].with_both(res2[m.name].orders.given)
        res.diagnostics['runs'] = 2
        res.diagnostics['second_run'] = res2.diagnostics
    return res
