"""Public prepared material reuse: physical media, static PML references and survey scope."""
import math

import numpy as np
import pytest

from wavebit import Material, fdtd
from wavebit.fdtd import _legacy as old
from wavebit.fdtd._lower_public import lower_problem
from wavebit.fdtd._prepare.grazing import accepted_cos, exponent
from wavebit.fdtd._prepare.shape_ranges import _coupled
from wavebit.fdtd._runtime import runtime
from tests.fdtd.test_substrate_continuation import pillar, numerics
from tests.fdtd.test_grazing_material_ad import slab, quiet


def problem(value=2.1, *, kind='slab', coefficient='epsilon', monitor=True):
    p = pillar()
    medium = Material(value) if coefficient == 'epsilon' else Material(1., mu=value)
    glass = p.scene.structures[0].replace(material=medium)
    if kind == 'slab':
        glass = glass.replace(shape=fdtd.Slab(z=(-.6, -.2375)), continuation=())
    scene = (p.scene.replace(structures=(), background=medium) if kind == 'background'
             else p.scene.replace(structures=(glass,)))
    monitors = (p.outputs.monitors[0].replace(wavelengths=(.62,)),) if monitor else ()
    return p.replace(scene=scene, outputs=p.outputs.replace(monitors=monitors))


def settings(kind='slab', *, stabilize=0., reference=True, steps=160):
    n = numerics(steps=steps).replace(geometry_ranges={})
    refs = {'z-': Material(2.1), 'z+': Material(2.1 if kind == 'background' else 1.)}
    return n.replace(absorber=n.absorber.replace(stabilize=stabilize,
                     reference_media=refs if reference else {}))


def numpy(value):
    return value.detach().numpy() if hasattr(value, 'detach') else np.asarray(value)


@pytest.mark.parametrize('backend', ['torch', 'jax'])
@pytest.mark.parametrize('kind', ['slab', 'continuation', 'background'])
@pytest.mark.parametrize('coefficient', ['epsilon', 'mu'])
def test_prepared_allowed_values_and_derivatives_and_grazing_boundary(backend, kind, coefficient):
    runtime(backend)
    n = settings(kind)
    make = lambda v=2.1: problem(v, kind=kind, coefficient=coefficient)
    prepared = quiet(fdtd.prepare, make(), numerics=n, backend=backend)
    assert not _coupled(prepared._plan)
    if kind in ('slab', 'background'):
        assert all(s.shape.body() is None for s in lower_problem(make(), n)[0].structures)
    def objective(v):
        r = quiet(prepared.run, make(v))
        return (sum((e*e).sum() for e in r.final_fields.values['E']),
                (r.final_fields.values['E'], r.final_fields.values['H'],
                 r.validity.checks['runtime_stability']))
    if backend == 'jax':
        import jax
        evaluate = jax.jit(jax.value_and_grad(objective, has_aux=True))
    else:
        import torch
        def evaluate(v):
            v = torch.tensor(v, dtype=torch.float64, requires_grad=True)
            value, aux = objective(v)
            derivative, = torch.autograd.grad(value, v)
            return (value.detach(), aux), derivative.detach()
    value = 2.05
    (got, (electric, magnetic, stable)), derivative = evaluate(value)
    direct = quiet(fdtd.run, make(value), numerics=n)
    assert bool(stable)
    for actual, expected in ((electric, direct.final_fields.values['E']),
                             (magnetic, direct.final_fields.values['H'])):
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(numpy(a), e, rtol=3e-12, atol=1e-14)
    def fresh(v):
        return sum((e*e).sum() for e in quiet(fdtd.run, make(v), numerics=n).final_fields.values['E'])
    np.testing.assert_allclose(numpy(got), fresh(value), rtol=3e-12)
    h = 1e-5
    finite = (fresh(value+h)-fresh(value-h))/(2*h)
    assert abs(finite) > 1e-8
    np.testing.assert_allclose(numpy(derivative), finite, rtol=2e-5, atol=1e-9)
    # The physical material changes while the numerical stretch reference stays put.
    lowered, _ = lower_problem(make(value), n)
    reference = lowered.boundaries.face('z-').background
    assert reference.eps == (2.1,)*3 and reference.mu is None
    medium = lowered.background if kind == 'background' else lowered.structures[0].material
    assert medium.index_bound() == math.sqrt(value)
    low = (.62/.6)**2
    c = accepted_cos(1e-8, 8, exponent(.62), prepared._plan.steps*prepared._plan.dt)
    high = low/(1-c*c)
    for v, allowed in ((high*(1-1e-7), False), (high*(1+1e-7), True)):
        if allowed:
            quiet(fdtd.prepare, make(v), numerics=n)
        else:
            with pytest.raises(ValueError, match='Grazing order refused'):
                quiet(fdtd.prepare, make(v), numerics=n)
        (out, (_, _, stable)), grad = evaluate(v)
        assert bool(stable) == allowed
        assert bool(np.isfinite(numpy(out))) == allowed
        assert bool(np.isfinite(numpy(grad))) == allowed


@pytest.mark.parametrize('backend', ['torch', 'jax'])
@pytest.mark.parametrize('kind', ['slab', 'continuation', 'background'])
def test_prepared_traced_stretch_reference_is_explicitly_refused(backend, kind):
    runtime(backend)
    n = settings(kind, reference=False, steps=2)
    p = quiet(fdtd.prepare, problem(kind=kind), numerics=n, backend=backend)
    def run(e):
        return quiet(p.run, problem(e, kind=kind)).final_fields.values['E'][1]
    with pytest.raises(ValueError, match='stretch profile is static'):
        if backend == 'jax':
            import jax
            jax.jit(run)(2.1)
        else:
            import torch
            run(torch.tensor(2.1, dtype=torch.float64, requires_grad=True))


@pytest.mark.parametrize('side', ['-', '+'])
@pytest.mark.parametrize('location', ['substrate', 'background'])
def test_stabilized_refusal_tracks_the_actual_layer_owner_on_either_face(side, location):
    import torch
    runtime('torch')
    def make(v, rho, *, monitored=True, driven=True):
        s = slab(v if location == 'substrate' else 2.1, wavelength=.95)
        if location == 'background':
            s = s.replace(structures=(), background=old.Material(eps=v))
        elif side == '+':
            s = s.replace(structures=(s.structures[0].replace(shape=old.Slab(z=(.6, 2.))),),
                          sources=(s.sources[0].replace(z=.25),))
        faces = tuple(f.replace(stabilize=rho if sign == side else 0.,
                              background=old.Material(eps=2.1 if location == 'background' else 1.))
                      for sign, f in zip(('-', '+'), s.boundaries.z))
        return s.replace(boundaries=s.boundaries.replace(z=faces),
                         monitors=s.monitors if monitored else (), sources=s.sources if driven else ())
    e = torch.tensor(2.1, dtype=torch.float64, requires_grad=True)
    concrete = make(2.1, .0001)
    p = quiet(old.plan, concrete)
    with pytest.raises(ValueError, match='traced material value and fills a stabilized PML layer'):
        quiet(old.lower, make(e, .0001), p, 'torch')
    # Without a wave or without Diffraction, this physical layer has no judged
    # outgoing order. Keep that existing supported material derivative path.
    for monitored, driven in ((False, True), (True, False)):
        # A Diffraction monitor itself requires the PlaneWave, so the second
        # branch is tested without the monitor as well.
        monitored = monitored and driven
        s = make(2.1, .0001, monitored=monitored, driven=driven)
        q = quiet(old.plan, s)
        out = quiet(old.lower, make(e, .0001, monitored=monitored, driven=driven), q, 'torch')
        assert torch.isfinite(out['be']).all()
