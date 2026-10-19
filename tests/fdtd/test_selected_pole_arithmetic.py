"""Check the recurrence selected by the plan, including concrete inactive slots."""
import dataclasses
import math

import numpy as np
import pytest

from wavebit import Units, fdtd as public
from wavebit.materials import Debye, DispersiveMaterial, Lorentz
from wavebit.fdtd import _legacy as fd
from wavebit.fdtd._lower_public import lower_problem
from wavebit.fdtd._prepare import checks
from wavebit.fdtd._runtime import runtime
from tests.fdtd.test_fd_hybrid_film import film_scene
from tests.fdtd.test_material_resolution_guard import quiet, scene


def hybrid_problem(active):
    units = Units(length='m')
    scale = units.time_seconds
    poles = [Debye(tau=1e-250*scale, delta_epsilon=0.),
             Lorentz(omega_0=1e200/scale, gamma=1e200/scale, strength=0.)]
    if active == 'debye':
        poles.append(Debye(tau=.15*scale, delta_epsilon=3.))
    elif active == 'lorentz':
        poles.append(Lorentz(omega_0=8./scale, gamma=1./scale, strength=10./scale**2))
    problem = public.Problem(units=units, scene=public.Scene(
        domain=public.Domain(size=(.4, .4, 2.)),
        boundaries=public.Boundaries(x=public.Periodic(), y=public.Periodic(),
                                     z=public.Periodic()),
        structures=(public.Structure('film', shape=public.Box(size=(.8, .8, .54),
                                                        center=(0., 0., .03)),
                                     material=DispersiveMaterial(1., tuple(poles))),)))
    numerics = public.Numerics(grid=public.UniformGrid(spacing=.1),
                              duration=public.Duration(steps=2), courant=.2)
    return problem, numerics


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('active', ['none', 'debye', 'lorentz'])
def test_public_hybrid_inactive_extremes_keep_the_selected_recurrence(backend, active):
    runtime(backend)
    problem, numerics = hybrid_problem(active)
    # A regression must fail this assertion, not escape as an arithmetic/import error.
    error = None
    try:
        report = quiet(public.check, problem, numerics=numerics)
    except ArithmeticError as exc:
        error = exc
    assert error is None, repr(error)
    assert report.ok, report.issues
    quiet(public.prepare, problem, numerics=numerics, backend=backend)
    lowered, _ = lower_problem(problem, numerics)
    plan = quiet(fd.plan, lowered)
    params = quiet(fd.lower, lowered, plan, backend)
    co = params['poles']['hybrid_film[0]']
    for key, values in co.items():
        for value in values if isinstance(values, tuple) else (values,):
            if hasattr(value, 'detach'):
                value = value.detach().numpy()
            assert np.isfinite(np.asarray(value)).all(), key
    assert len(co['drive']) == (2 if active == 'none' else 3)
    for key in ('drive', 'coupling', 'debye_c', 'debye_drive', 'debye_feedback',
                'energy_inverse_delta', 'energy_inverse_strength', 'energy_active'):
        for slot in co[key][:2]:
            assert np.count_nonzero(np.asarray(slot)) == 0
    if active == 'none':
        assert np.count_nonzero(np.asarray(co['energy_hybrid_shift'])) == 0
    bound, reason = checks.cfl_bound(lowered, plan)
    assert reason is None
    vacuum = plan.dx/math.sqrt(3.)
    if active == 'none':
        assert bound == vacuum
    else:
        assert bound < vacuum
    with pytest.raises(ValueError, match='stability|bound|courant'):
        checks.check_stability(lowered, dataclasses.replace(plan, dt=2*bound))


@pytest.mark.parametrize('scope', ['bulk-lorentz', 'hybrid-lorentz', 'hybrid-debye'])
@pytest.mark.parametrize('strength', [1e-300, 1.])
def test_active_extreme_poles_still_receive_explicit_refusals(scope, strength):
    if scope == 'bulk-lorentz':
        seed = scene(fd.Dispersive(poles=(fd.Lorentz(omega_0=1., strength=1.),)))
        plan = quiet(fd.plan, seed)
        bad = scene(fd.Dispersive(poles=(fd.Lorentz(omega_0=1e200, strength=strength),)))
        operation = lambda: checks.check_stability(bad, plan)
        message = 'PoleArithmeticRefused'
    else:
        # Keep the other family's extreme declaration exactly inactive.
        specs = (('debye', 1e-250, 0., strength if scope == 'hybrid-debye' else 0.),
                 ('lorentz', 1e200, 1e200, strength if scope == 'hybrid-lorentz' else 0.))
        operation = lambda: quiet(fd.plan, film_scene(eps=1., pole_specs=specs))
        message = ('stability bound' if scope == 'hybrid-debye' and strength == 1e-300
                   else 'not representable')
    error = None
    try:
        operation()
    except (ValueError, ArithmeticError) as exc:
        error = exc
    assert isinstance(error, ValueError), repr(error)
    assert message in str(error)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_bulk_signed_zero_strength_keeps_the_live_drive_tangent(backend):
    runtime(backend)
    def make(strength):
        return scene(fd.Dispersive(poles=(fd.Lorentz(
            omega_0=2., gamma=.5, strength=strength),)))
    plan = quiet(fd.plan, make(0.))
    expected = plan.dt**2/(1 + .5*.5*plan.dt)
    def drive(strength):
        return quiet(fd.lower, make(strength), plan, backend)['poles']['poles[0]']['drive'][0]
    for strength in (0., -0., 1e-4):
        if backend == 'jax':
            import jax
            value, tangent = jax.jit(jax.value_and_grad(drive))(strength)
            value, tangent = float(value), float(tangent)
        else:
            import torch
            x = torch.tensor(strength, dtype=torch.float64, requires_grad=True)
            result = drive(x)
            value = float(result.detach())
            tangent = float(torch.autograd.grad(result, x)[0])
        assert value == pytest.approx(strength*expected, rel=1e-12, abs=0.)
        assert expected > 0
        assert tangent == pytest.approx(expected, rel=1e-12, abs=0.)
    # A one-sided difference at zero stays inside the nonnegative material domain.
    def eager(s):
        return quiet(fd.lower, make(s), plan)['poles']['poles[0]']['drive'][0]
    assert (eager(1e-4)-eager(0.))/1e-4 == pytest.approx(expected, rel=1e-12, abs=0.)
