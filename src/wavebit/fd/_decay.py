"""Concrete CPU stopping on separate, positive state-amplitude tails.

This is an explicit user-requested heuristic, not an energy, spectral-error or future-echo
bound. Each evolving array owns its own all-time peak. Complete non-overlapping windows
read every state after a step; source histories and accumulating observers never enter the
criterion. A Gaussian is not truncated: eligibility only bounds its remaining envelope.
The fixed-length executor is deliberately separate and unchanged.
"""
import contextlib
import dataclasses
import math
import sys
from fractions import Fraction

import numpy as np

from ._spec import torch_carried

__all__ = ['state_groups', 'amplitude', 'source_gate', 'window_steps', 'Criterion',
           'validate_scene', 'validate_plan', 'validate_runtime', 'reject_carried', 'execute']


def reject_carried(value):
    """Refuse a derivative before any backend conversion could read only its primal."""
    if isinstance(value, dict):
        for x in value.values():
            reject_carried(x)
    elif isinstance(value, (tuple, list)):
        for x in value:
            reject_carried(x)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            reject_carried(getattr(value, f.name))
    else:
        jax = sys.modules.get('jax')
        if torch_carried(value) or (jax is not None and isinstance(value, jax.core.Tracer)):
            raise ValueError('DecayConcreteOnly: stopping requires concrete forward values; '
                             'AD, batching and an enclosing JAX trace are not supported.')


def validate_scene(scene):
    """Check the opt-in scene before a planning seed or lowering can detach a leaf."""
    if scene.run.decay is None:
        return
    from .sources import Dipole
    reject_carried(scene)
    if any(type(source) is not Dipole for source in scene.sources):
        raise ValueError('DecaySourceRefused: only InitialField and Dipole excitation are '
                         'supported; PlaneWave/TFSF source columns can carry later echoes.')
    source_gate(scene, scene.run.decay.threshold)


def _term_types():
    from ._kernel.constitutive import ConstitutiveTerm
    from ._kernel.poles import PoleTerm
    from ._kernel.sources import DipoleTerm
    from ._kernel.stencil import StencilTerm
    from ._lower.conductive_film import InterfacePoleTerm
    from ._lower.conductive_stack import ConductiveStackTerm
    from ._lower.debye_film import DebyeFilmTerm
    from ._lower.debye_sum_film import DebyeSumFilmTerm
    from ._lower.hybrid_film import HybridFilmTerm
    from ._lower.oscillator_sum_film import OscillatorSumFilmTerm
    from ._lower.single_pole_film import DispersiveFilmTerm
    from ._lower.two_debye_film import TwoDebyeFilmTerm
    poles = (PoleTerm, InterfacePoleTerm, ConductiveStackTerm, DebyeFilmTerm,
             DebyeSumFilmTerm, HybridFilmTerm, OscillatorSumFilmTerm,
             DispersiveFilmTerm, TwoDebyeFilmTerm)
    return poles, (ConstitutiveTerm, StencilTerm), DipoleTerm


def validate_plan(scene, plan):
    """The optional configuration, hard cap and enumerated memory schema must agree."""
    if scene.run.decay is None and plan.decay is None:
        return
    validate_scene(scene)
    from ._lower.plan import run_steps
    if scene.run.decay != plan.decay or run_steps(scene, plan.dt) != plan.steps:
        raise ValueError('DecayPlanMismatch: build a plan for this hard cap and Decay '
                         'configuration; an early-result plan describes its actual prefix.')
    from ._kernel.observe import DFTRegion, PointRecord
    poles, flux, dipole = _term_types()
    if any(type(t) not in poles + flux + (dipole,) for t in plan.terms):
        raise ValueError('DecayMemoryRefused: this term has no certified evolving-state '
                         'group schema for stopping.')
    if any(type(o) not in (PointRecord, DFTRegion) for o in plan.observers):
        raise ValueError('DecayObserverRefused: only the built-in Record and DFT observers '
                         'have an execution-prefix result contract.')
    if len({t.name for t in plan.terms}) != len(plan.terms):
        raise ValueError('DecayMemoryRefused: term memory names must be unique.')
    window_steps(plan.decay.window, plan.dt)


def validate_runtime(rt):
    """The first stopping scope is concrete NumPy/JAX/Torch on a CPU only."""
    from ._runtime import JaxRuntime, NumpyRuntime, TorchRuntime
    if type(rt) not in (NumpyRuntime, JaxRuntime, TorchRuntime):
        raise ValueError('DecayBackendRefused: use the NumPy, JAX or Torch CPU runtime.')
    if rt.name == 'torch' and rt.device.type != 'cpu':
        raise ValueError('DecayDeviceRefused: stopping is supported on CPU only.')
    if rt.name == 'jax':
        platform = rt.jax.default_backend() if rt.device is None else rt.device.platform
        if platform != 'cpu':
            raise ValueError('DecayDeviceRefused: stopping is supported on CPU only.')


def source_gate(scene, threshold):
    """Latest trailing Gaussian-envelope eligibility time, including delayed dipoles.

    Row times are (n+1/2)dt for electric current and n dt for magnetic current. Requiring
    the WINDOW START n*dt past this bound is conservative for both. The waveform remains
    its original infinite Gaussian. The last nextafter rounds the computed time upward;
    this is a floating-point eligibility rule, not a rigorously rounded exp certificate.
    """
    latest = 0.0
    for source in scene.sources:
        try:
            band = source.band
            tau = 1.0 / (2.0 * math.pi * float(band.df))
            t0 = float(band.sigmas) * tau
            tail = math.sqrt(-2.0 * math.log(min(float(band.cutoff), threshold))) * tau
            time = math.fsum((float(source.delay), t0, tail))
            if not (math.isfinite(tau) and tau > 0 and math.isfinite(t0)
                    and t0 > 0 and math.isfinite(tail) and tail > 0
                    and math.isfinite(time)):
                raise ValueError
            time = math.nextafter(time, math.inf)
            if not math.isfinite(time):
                raise ValueError
            latest = max(latest, time)
        except (ArithmeticError, ValueError) as err:
            raise ValueError('DecaySourceRange: the source eligibility time must be '
                             'representable as a finite float64 value.') from err
    return latest


def window_steps(window, dt):
    """Exact ceiling for the represented positive floats, without quotient overflow."""
    if not (math.isfinite(window) and window > 0 and math.isfinite(dt) and dt > 0):
        raise ValueError('DecayWindowRange: window and dt must be finite and positive.')
    ratio = Fraction(window) / Fraction(dt)
    return (ratio.numerator + ratio.denominator - 1) // ratio.denominator


def state_groups(plan, state):
    """Ordered array groups. Unknown or changed memory layouts are refused, not omitted."""
    poles, flux, dipole = _term_types()
    out = {'E': state['E'], 'H': state['H']}
    for name in sorted(state['psi']):
        out['psi:' + name] = state['psi'][name]
    expected_psi = {key for face in plan.cpml for key in face.keys()}
    if set(state['psi']) != expected_psi or set(state['mem']) != {t.name for t in plan.terms}:
        raise ValueError('DecayMemoryRefused: the state does not match the plan memory.')
    for term in plan.terms:
        mem = state['mem'][term.name]
        if type(term) in poles:
            keys = ('P', 'previous', 'delta')
            if getattr(term, 'memory_arrays_per_pole', 3) == 4:
                keys += ('aux',)
            if set(mem) != set(keys) or any(len(mem[k]) != term.n_poles for k in keys):
                raise ValueError('DecayMemoryRefused: unsupported pole memory layout.')
            for key in keys:
                for i, value in enumerate(mem[key]):
                    out[f'mem:{term.name}:{key}[{i}]'] = value
        elif type(term) in flux:
            out[f'mem:{term.name}:flux'] = mem
        elif type(term) is not dipole:
            raise ValueError('DecayMemoryRefused: unsupported evolving term.')
    return out


def amplitude(array, rt):
    """max(|real|, |imag|), with no squared norm, complex modulus or mixed-unit sum."""
    return rt.state_amplitude(array)


class Criterion:
    """Host-side decision at window boundaries; device reductions supply their maxima."""

    def __init__(self, plan, initial_metrics, eligible_time):
        self.plan = plan
        self.groups = tuple(initial_metrics)
        self.eligible_time = eligible_time
        self.window_steps = window_steps(plan.decay.window, plan.dt)
        self.peaks = {key: 0.0 for key in self.groups}
        self.window_max = {key: 0.0 for key in self.groups}
        self.nonfinite = False
        self.complete_windows = 0
        self._incorporate(initial_metrics)

    def _incorporate(self, values):
        if tuple(values) != self.groups:
            raise ValueError('DecayMemoryRefused: state group order changed during execution.')
        for key, value in values.items():
            if not math.isfinite(value) or value < 0:
                self.nonfinite = True
            else:
                self.peaks[key] = max(self.peaks[key], value)

    def observe(self, window_metrics, start_step, count):
        self.window_max = dict(window_metrics)
        self._incorporate(window_metrics)
        complete = count == self.window_steps
        if complete:
            self.complete_windows += 1
        # Compare exact products of the represented floats; n*dt cannot overflow or round
        # an early window start past the declared source gate.
        eligible = Fraction(start_step) * Fraction(self.plan.dt) >= Fraction(self.eligible_time)
        return bool(complete and eligible and not self.nonfinite and self.field_seen
                    and all(Fraction(window_metrics[k]) <= Fraction(self.plan.decay.threshold)
                            * Fraction(self.peaks[k]) for k in self.groups))

    @property
    def field_seen(self):
        return self.peaks['E'] > 0 or self.peaks['H'] > 0

    def report(self, actual_steps, reached):
        return dict(reason='decay' if reached else 'cap', steps=actual_steps,
                    cap_steps=self.plan.steps, cap_skeleton=self.plan.skeleton,
                    reached=bool(reached), threshold=self.plan.decay.threshold,
                    window=self.plan.decay.window, window_steps=self.window_steps,
                    eligible_time=self.eligible_time, groups=self.groups,
                    peaks=dict(self.peaks), window_max=dict(self.window_max),
                    # Ratios are display-only rounded floats. The decision compares exact
                    # products of the represented finite amplitudes and threshold.
                    ratios={k: (self.window_max[k] / self.peaks[k] if self.peaks[k] else 0.0)
                            for k in self.groups}, nonfinite=self.nonfinite,
                    field_seen=self.field_seen, complete_windows=self.complete_windows)


def _prefix(plan, state, steps, rt):
    """Own shortened time histories; do not retain the discarded cap through a view."""
    if steps == plan.steps:
        return state

    def own(value):
        return rt.xp.asarray(value, copy=True)

    from ._kernel.observe import PointRecord
    from ._kernel.sources import DipoleTerm
    out = dict(state)
    out['obs'] = dict(state['obs'])
    for observer in plan.observers:
        acc = state['obs'][observer.name]
        out['obs'][observer.name] = (own(acc[:steps]) if type(observer) is PointRecord
                                    else dict(acc, energy=own(acc['energy'][:steps])))
    out['mem'] = dict(state['mem'])
    for term in plan.terms:
        if type(term) is DipoleTerm:
            mem = dict(state['mem'][term.name])
            for key in ('drive', 'field', 'work'):
                mem[key] = own(mem[key][:steps])
            out['mem'][term.name] = mem
    # With stride one the completed rows are T^1 .. T^(N-1). Functional backends may
    # already hold the pending electric-only T^N; it is not a completed energy sample.
    out['energy'] = own(state['energy'][:max(steps - 1, 0)])
    return out


def execute(scene, plan, params, rt):
    """Return (actual plan, prefix state, stopping report), checking once per full window."""
    validate_plan(scene, plan)
    validate_runtime(rt)
    reject_carried(params)
    from ._kernel.sources import has_sources, waveform_table
    from ._kernel.step import initial_state, step
    from ._run import _program
    program = _program(plan)
    params = rt.to_device(params)
    placement = (rt.jax.default_device(rt.device or rt.jax.devices('cpu')[0])
                 if rt.name == 'jax' else contextlib.nullcontext())
    with placement:
        if has_sources(program):
            params = dict(params, waveform=waveform_table(program, params, rt))
        reject_carried(params)
        state = initial_state(program, params, rt)
        reject_carried(state)
        groups = state_groups(program, state)
        norms = rt.xp.stack([amplitude(a, rt) for a in groups.values()])
        group_names = tuple(groups)
        group_count = len(group_names)
        initial = dict(zip(group_names, map(float, np.asarray(rt.numpy(norms)))))
        criterion = Criterion(plan, initial, source_gate(scene, plan.decay.threshold))
        reached, done = False, 0
        while done < plan.steps:
            count = min(criterion.window_steps, plan.steps - done)

            def make(count=count):
                def block(parameters, carry, offset):
                    def body(pair, local):
                        current, maxima = pair
                        current = step(program, parameters, current, offset + local, rt)
                        values = rt.xp.stack([amplitude(a, rt) for a in
                                              state_groups(program, current).values()])
                        # The runtime preserves subnormal bits through this merge too;
                        # native floating max can flush them on JAX CPU. NaN/Inf is sticky.
                        return current, rt.positive_maximum(maxima, values)
                    zeros = rt.zeros((group_count,), 'float64')
                    return rt.scan(body, (carry, zeros), count, program.schedule)
                return block

            key = ('decay-window', program, count)
            state, maxima = rt.jit_for(key, make)(params, state, done)
            observed = dict(zip(group_names, map(float, np.asarray(rt.numpy(maxima)))))
            reached = criterion.observe(observed, done, count)
            done += count
            if reached:
                break
    actual = plan if done == plan.steps else plan.with_static(steps=done)
    return actual, _prefix(plan, state, done, rt), criterion.report(done, reached)
