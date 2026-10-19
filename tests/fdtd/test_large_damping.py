"""Large finite damping must not overflow an otherwise representable ADE."""
from decimal import Decimal, localcontext
import numpy as np
import pytest

from wavebit import fdtd as public, Units
from wavebit.materials import DispersiveMaterial, Drude
from wavebit.fdtd import _legacy as fd
from wavebit.fdtd._runtime import runtime
from wavebit.fdtd._lower_public import lower_problem
from wavebit.materials.time_domain import _second_order
from tests.fdtd.test_material_resolution_guard import backend_call, quiet, scene
from wavebit.fdtd._prepare.shape_ranges import resolution_guard
from types import SimpleNamespace


def initial(x,y,z):
    return 1+0*x


def physical_problem(gamma=1e308, omega_p=1e20):
    return public.Problem(units=Units(length='m'), scene=public.Scene(
        domain=public.Domain(size=(1e-11,)*3), boundaries=public.Boundaries(
            x=public.Periodic(), y=public.Periodic(), z=public.Periodic()),
        structures=(public.Structure('film', shape=public.SnappedBox(size=(4e-12,)*3),
            material=DispersiveMaterial(1., (Drude(omega_p=omega_p, gamma=gamma),))),)),
        initial=public.InitialField(E=(None, initial, None)),
        outputs=public.Outputs(monitors=(public.TimeSeriesMonitor('p', at=(0.,)*3, component='Ey'),)))


def numerics():
    return public.Numerics(grid=public.UniformGrid(spacing=1e-12),
                           duration=public.Duration(steps=3))


def reference(dt, w0, gamma, strength):
    with localcontext() as ctx:
        ctx.prec = 90
        t,w,g,s = map(Decimal.from_float, (dt,w0,gamma,strength))
        den = 1/t**2+g/(2*t)
        return np.array([float((2/t**2-w**2)/den),
                         float((1/t**2-g/(2*t))/den), float(s/den)])


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('strength', [0., 1e100, 3e100])
def test_recurrence_near_overflow_boundary_matches_decimal(backend, strength):
    rt = runtime(backend)
    dt = .01
    cutoff = np.finfo(float).max*dt/4
    evaluate = backend_call(backend, lambda g: rt.xp.stack(tuple(
        rt.array(x, 'float64') for x in _second_order(dt, 1., g, strength))))
    for gamma in (float(np.nextafter(cutoff, 0.)), float(cutoff),
                  float(np.nextafter(cutoff, np.inf)), 1e308):
        got = evaluate(gamma)
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, reference(dt,1.,gamma,strength),rtol=1e-12,atol=0.)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_large_damping_resolution_uses_finite_optical_response(backend):
    rt = runtime(backend)
    plan=SimpleNamespace(dx=.05, grid=None, pml_wavelengths=())
    for kind in ('drude', 'lorentz'):
        def make(g):
            pole = (fd.Drude(omega_p=1., gamma=g) if kind=='drude' else
                    fd.Lorentz(omega_0=1., gamma=g, strength=1.))
            return fd.Dispersive(eps_inf=1., poles=(pole,))
        evaluate=backend_call(backend,lambda g: resolution_guard(scene(make(g)),plan,rt,None))
        assert float(evaluate(1e308)) == 1.


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_public_check_prepare_and_live_run_agree_at_extreme_finite_damping(backend):
    runtime(backend)
    p,n=physical_problem(),numerics()
    assert quiet(public.check,p,numerics=n).ok
    prepared=quiet(public.prepare,p,numerics=n,backend=backend)
    s,_=lower_problem(p,n)
    q=quiet(fd.lower,s,quiet(fd.plan,s))
    assert all(np.isfinite(v) for row in q['poles'].values() for k in ('step_gain','hold','drive') for v in row[k])
    def fn(g):
        r=quiet(prepared.run,physical_problem(g))
        return r['p'].values
    got=backend_call(backend,fn)(1e308)
    want=np.asarray(quiet(public.run,p,numerics=n)['p'].values)
    assert np.isfinite(got).all() and np.any(got != 0)
    np.testing.assert_allclose(got,want,rtol=1e-12,atol=0.)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_subnormal_active_drive_is_explicitly_refused_on_both_paths(backend):
    from wavebit.errors import ValidationError
    runtime(backend)
    p,n=physical_problem(omega_p=1e8),numerics()
    report=quiet(public.check,p,numerics=n)
    assert not report.ok
    assert any('PoleArithmeticRefused' in str(i.message) for i in report.issues)
    with pytest.raises((ValueError,ValidationError),match='PoleArithmeticRefused'):
        quiet(public.prepare,p,numerics=n)
    seed=quiet(public.prepare,physical_problem(),numerics=n,backend=backend)
    def run(w):
        r=quiet(seed.run,physical_problem(omega_p=w))
        valid=r.validity.checks['runtime_stability']
        return r['p'].values, valid
    if backend=='jax':
        import jax
        values,valid=jax.jit(run)(1e8)
        values=np.asarray(values)
    else:
        import torch
        values,valid=run(torch.tensor(1e8,dtype=torch.float64,requires_grad=True))
        values=values.detach().numpy()
    assert not bool(valid) and np.isnan(values).all()


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_normal_drive_boundary_and_zero_strength(backend):
    rt=runtime(backend)
    evaluate=backend_call(backend,lambda s: rt.xp.stack(tuple(
        rt.array(v,'float64') for v in _second_order(.01,1.,1e308,s))))
    for s,accepted in ((0.,True),(100.,False),(120.,True)):
        got=evaluate(s)
        assert bool(np.isfinite(got).all()) == accepted
        if accepted:
            np.testing.assert_allclose(got,reference(.01,1.,1e308,s),rtol=1e-12,atol=0.)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_representable_strength_gradient_and_zero_strength_tangent(backend):
    runtime(backend)
    def objective(s):
        return _second_order(.01,1.,1e308,s)[2]*1e300
    expected=reference(.01,1.,1e308,1.)[2]*1e300
    for s in (0.,1e100):
        if backend=='jax':
            import jax
            grad=float(jax.jit(jax.grad(objective))(s))
        else:
            import torch
            x=torch.tensor(s,dtype=torch.float64,requires_grad=True)
            out=objective(x)
            grad=float(torch.autograd.grad(out,x)[0])
        assert expected != 0
        assert grad == pytest.approx(expected,rel=1e-12)
    h=1e94
    d=(reference(.01,1.,1e308,1e100+h)[2]-reference(.01,1.,1e308,1e100-h)[2])*1e300/(2*h)
    assert d == pytest.approx(expected,rel=2e-6)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('family', ['drude','lorentz'])
def test_large_finite_susceptibility_matches_decimal_without_complex_overflow(backend,family):
    rt=runtime(backend)
    omega=6.25
    def fn(g):
        p=(fd.Drude(omega_p=1e50,gamma=g) if family=='drude' else
           fd.Lorentz(omega_0=1.,gamma=g,strength=1e100))
        value=p.eps_term(omega)
        return rt.xp.stack((rt.xp.real(value),rt.xp.imag(value)))
    with localcontext() as ctx:
        ctx.prec=90
        w=Decimal.from_float(omega);g=Decimal.from_float(1e308)
        strength=Decimal.from_float(1e50)**2 if family=='drude' else Decimal.from_float(1e100)
        a=-w*w if family=='drude' else 1-w*w
        b=g*w
        expected=np.array([float(strength*a/(a*a+b*b)),float(strength*b/(a*a+b*b))])
    assert expected[1] != 0
    np.testing.assert_allclose(backend_call(backend,fn)(1e308),expected,rtol=1e-12,atol=0.)


@pytest.mark.parametrize('backend',['jax','torch'])
def test_traced_damping_cannot_flush_an_active_drive_to_zero(backend):
    def make(g):
        return scene(fd.Dispersive(eps_inf=1.,poles=(fd.Drude(omega_p=1.,gamma=g),)))
    seed=quiet(fd.plan,make(1.))
    with pytest.raises(ValueError,match='PoleArithmeticRefused'):
        quiet(fd.plan,make(1e308))
    evaluate=backend_call(backend,lambda g: quiet(fd.lower,make(g),seed,backend)['stable'])
    assert np.isnan(evaluate(1e308))


@pytest.mark.parametrize('backend',['jax','torch'])
@pytest.mark.parametrize('strength', [np.nextafter(0., 1.), 1e-320,
                                      np.finfo(float).tiny/2])
def test_nonzero_subnormal_strength_is_not_an_inactive_zero(backend, strength):
    rt=runtime(backend)
    evaluate=backend_call(backend,lambda s: rt.xp.stack(tuple(
        rt.array(v,'float64') for v in _second_order(.01,1.,1.,s))))
    assert np.isfinite(evaluate(0.)).all()
    assert np.isfinite(evaluate(-0.)).all()
    assert np.isnan(evaluate(strength)).all()


@pytest.mark.parametrize('backend',['jax','torch'])
def test_traced_eps_does_not_hide_a_concrete_unrepresentable_pole(backend):
    def make(eps,gamma):
        return scene(fd.Dispersive(eps_inf=eps,poles=(fd.Drude(omega_p=1.,gamma=gamma),)))
    seed=quiet(fd.plan,make(1.,1.))
    evaluate=backend_call(backend,lambda e: quiet(fd.lower,make(e,1e308),seed,backend)['stable'])
    assert np.isnan(evaluate(1.))


@pytest.mark.parametrize('backend',['jax','torch'])
def test_public_zero_strength_lorentz_stays_inactive_with_huge_finite_damping(backend):
    from wavebit.materials import Lorentz
    runtime(backend)
    def make(g):
        p=physical_problem()
        body=p.scene.structures[0].replace(material=DispersiveMaterial(1.,(
            Lorentz(omega_0=1e8,gamma=g,strength=0.),)))
        return p.replace(scene=p.scene.replace(structures=(body,)))
    p,n=make(1e308),numerics()
    assert quiet(public.check,p,numerics=n).ok
    prepared=quiet(public.prepare,p,numerics=n,backend=backend)
    def fn(g):
        return quiet(prepared.run,make(g))['p'].values
    got=backend_call(backend,fn)(1e308)
    assert np.isfinite(got).all()
    np.testing.assert_array_equal(got,np.ones_like(got))
