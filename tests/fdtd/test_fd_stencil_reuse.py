"""Stencil reuse must bind both geometry and the snapped scene grid.

A prepared operator's node digest is not a scene-grid check: building those
nodes from the held plan hides a changed resolution/domain at direct lowering.
On d89627d4 the spacing/size/centre cases and their allocation-order witnesses
failed (six cases). Disabling the product grid guard reproduces those six failures;
removing the product memory check fails all four memory-order witnesses.
Use small physical spheres and spies, never a deliberately oversized allocation.
"""
from functools import lru_cache
import numpy as np
import pytest
from wavebit.fdtd import _legacy as fd
from wavebit.fdtd._prepare import plan as build_plan, lower
from wavebit.fdtd._prepare import stencil
from tests.fdtd.test_fd_stencil import scene, quiet, with_structure, stencil_term


@lru_cache(maxsize=1)
def prepared():
    s = scene().replace(monitors=())
    return s, quiet(build_plan, s)


def changed(s, kind):
    if kind == 'radius':
        return with_structure(s, fd.Sphere(radius=.145))
    if kind == 'center':
        return with_structure(s, fd.Sphere(radius=.14,center=(.005,0.,0.)))
    if kind == 'material':
        return with_structure(s, material=2.3)
    if kind == 'spacing':
        return s.replace(resolution=fd.Resolution(dx=1/19))
    if kind == 'domain_size':
        return s.replace(domain=s.domain.replace(size=tuple(v+2/18 for v in s.domain.size)))
    if kind == 'domain_center':
        return s.replace(domain=s.domain.replace(center=(.01,0.,0.)))
    raise AssertionError(kind)


@pytest.mark.parametrize('kind', ['radius','center','material','spacing','domain_size','domain_center'])
def test_direct_lowering_refuses_changed_operator_inputs(kind):
    s,p = prepared()
    with pytest.raises(ValueError,match='StencilMismatchRefused'):
        quiet(lower,changed(s,kind),p,'numpy')


@pytest.mark.parametrize('kind', ['spacing','domain_size','domain_center'])
def test_changed_grid_is_refused_before_coefficient_allocation(monkeypatch,kind):
    s,p = prepared()
    from wavebit.fdtd._prepare import params
    def forbidden(*a,**kw):
        pytest.fail('stale grid reached coefficient allocation')
    monkeypatch.setattr(params,'build_coefficients',forbidden)
    with pytest.raises(ValueError,match='StencilMismatchRefused'):
        quiet(lower,changed(s,kind),p,'numpy')


@pytest.mark.parametrize('kind', ['radius','center','material','spacing','domain_size','domain_center'])
def test_replanning_never_reuses_a_changed_operator(monkeypatch,kind):
    s,p = prepared()
    class Assemble(Exception):
        pass
    def stop(*a,**kw):
        raise Assemble()
    monkeypatch.setattr(stencil,'_assemble',stop)
    with pytest.raises(Assemble):
        quiet(build_plan,changed(s,kind),_reuse=p)


def test_identical_reuse_preserves_operator_and_nonzero_fields(monkeypatch):
    s,p = prepared()
    first = quiet(fd.run,s,plan=p)
    def forbidden(*a,**kw):
        pytest.fail('identical preparation refitted the operator')
    monkeypatch.setattr(stencil,'_assemble',forbidden)
    reused = quiet(build_plan,s,_reuse=p)
    assert stencil_term(reused) is stencil_term(p)
    second = quiet(fd.run,s,plan=p)
    np.testing.assert_array_equal(first.final.E,second.final.E)
    np.testing.assert_array_equal(first.final.H,second.final.H)
    region = stencil_term(p).region
    box = (slice(None),slice(*region.z),slice(*region.y),slice(*region.x))
    assert np.any(first.final.E[box] != 0), 'operator region must be excited'


@pytest.mark.parametrize('reuse', [False,True])
@pytest.mark.parametrize('bound', ['structure','default'])
def test_tight_memory_bound_refuses_before_sparse_work(monkeypatch,reuse,bound):
    s,p = prepared()
    limit = 128*1024**2  # Existing fixture prediction is about 265 MiB.
    limited = with_structure(s,memory_bound=limit) if bound == 'structure' else s
    if bound == 'default':
        monkeypatch.setattr(stencil,'MEMORY_BOUND_BYTES',limit)
    def forbidden(*a,**kw):
        pytest.fail('memory refusal came after sparse preparation')
    for name in ('_assemble','geometry','fit','energy_bounds','padded','_reused'):
        monkeypatch.setattr(stencil,name,forbidden)
    with pytest.raises(ValueError,match='StencilMemoryRefused'):
        quiet(build_plan,limited,_reuse=p if reuse else None)


def test_equivalent_snapped_grid_keeps_the_prepared_operator():
    s,p = prepared()
    equivalent = s.replace(domain=s.domain.replace(size=(1.99,)*3))
    assert equivalent.domain.size != s.domain.size
    first,second = quiet(lower,s,p,'numpy'),quiet(lower,equivalent,p,'numpy')
    np.testing.assert_array_equal(first['be'],second['be'])
    for name in first['stencil']:
        for key in first['stencil'][name]:
            np.testing.assert_array_equal(first['stencil'][name][key],second['stencil'][name][key])


@pytest.mark.parametrize('backend', ['numpy','jax','torch'])
def test_grid_guard_preserves_stencil_no_ad_contract(backend):
    s,p = prepared()
    def loss(value):
        other = with_structure(s,material=fd.Material(eps=value))
        return quiet(lower,other,p,backend)['be'].sum()
    with pytest.raises(ValueError,match='StencilNumberRefused'):
        if backend == 'numpy':
            loss(np.asarray(2.25))
        elif backend == 'jax':
            jax = pytest.importorskip('jax')
            jax.grad(loss)(2.25)
        else:
            torch = pytest.importorskip('torch')
            loss(torch.tensor(2.25,dtype=torch.float64,requires_grad=True))
