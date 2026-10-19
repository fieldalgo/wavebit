"""Bounded high-cos probes: valid public inputs, eager admission, exact float endpoints."""
from contextlib import contextmanager
import math
import signal
import warnings
import numpy as np
import pytest
from wavebit import Material, Units, fdtd as fd
from wavebit.units import C0
from wavebit.fdtd._lower_public import lower_problem
from wavebit.fdtd._prepare import grazing


@contextmanager
def bounded(seconds=20):
    if not hasattr(signal,'setitimer'):
        yield
        return
    def expired(*args):
        raise TimeoutError('grazing endpoint search exceeded its time budget')
    old=signal.signal(signal.SIGALRM,expired)
    previous=signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,*previous)
        signal.signal(signal.SIGALRM,old)


def public_case(wavelength=10., reflection=1e-8, sine=.3):
    units=Units(length='um')
    source=fd.PlaneWave('pw',plane=fd.Plane(z=.2),theta=math.asin(sine),
        pulse=fd.GaussianPulse(wavelength=1.,frequency_sigma_hz=.1*C0/1e-6))
    faces=fd.Bloch.from_source(source,units=units,invariant_axes=('x','y'))
    def problem(epsilon=1.):
        return fd.Problem(units=units,scene=fd.Scene(domain=fd.Domain(size=(.01,.01,1.)),
            boundaries=fd.Boundaries(**faces,z=fd.Open()),
            structures=(fd.Structure('slab',shape=fd.SnappedSlab(z=(-.5,-.1)),material=Material(epsilon)),)),
            sources=(source,),outputs=fd.Outputs(final_fields=True,monitors=(fd.DiffractionMonitor('T',
                plane=fd.Plane(z=-.3),wave='transmitted',source='pw',wavelengths=(wavelength,),orders=0),)))
    numerics=fd.Numerics(grid=fd.UniformGrid(spacing=.01),duration=fd.Duration(steps=2),
        absorber=fd.PML(cells=12,wavelength=1.,reflection=reflection,reference_media={'z-':Material(1.)}))
    return problem,numerics


def ratio(value,ku,kv,wavelength):
    k=math.sqrt(value)*2*math.pi/wavelength
    return (ku*ku+kv*kv)/(k*k)


def accepted(value,c,ku=2.,kv=3.,wavelength=1.):
    s2=ratio(value,ku,kv,wavelength)
    return s2<1. and math.sqrt(1.-s2)>=c


@pytest.mark.parametrize('c',[1.,1.01,2.,math.inf])
def test_high_threshold_has_bounded_eager_exact_interval(c):
    with bounded(2):
        low,high=grazing._rounded_band(2.,3.,1.,c)
    assert ratio(low,2.,3.,1.)>=1.
    assert ratio(math.nextafter(low,math.inf),2.,3.,1.)<1.
    if c>1.:
        assert high==math.inf
        assert not accepted(float.fromhex('0x1.fffffffffffffp+1023'),c)
    else:
        assert math.isfinite(high) and accepted(high,c)
        assert not accepted(math.nextafter(high,-math.inf),c)


@pytest.mark.parametrize('backend',['torch','jax'])
@pytest.mark.parametrize('wavelength',[3.,5.,10.])
def test_public_off_design_traced_admission_matches_eager(backend,wavelength):
    reflection=(math.exp(math.log(grazing.GRAZING_TOLERANCE)/grazing.exponent(wavelength))
                if wavelength in (3.,5.) else 1e-8)
    c=grazing.accepted_cos(reflection,12,grazing.exponent(wavelength),None)
    assert (c<1. if wavelength==3. else c==1. if wavelength==5. else c>1.)
    sine=.4 if wavelength==3. else .3
    problem,numerics=public_case(wavelength,reflection,sine=sine)
    with warnings.catch_warnings(),bounded():
        warnings.simplefilter('ignore')
        if backend=='jax':
            jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
        prepared=fd.prepare(problem(),numerics=numerics,backend=backend)
        assert prepared._plan.nx==prepared._plan.ny==1
        if backend=='torch':
            torch=pytest.importorskip('torch')
            def run(e):
                r=prepared.run(problem(torch.tensor(e,dtype=torch.float64,requires_grad=True)))
                return bool(r.validity.checks['runtime_stability']),np.asarray(r.final_fields.values['E'].detach())
        else:
            import jax.numpy as jnp
            @jax.jit
            def evaluate(e):
                r=prepared.run(problem(e))
                return r.validity.checks['runtime_stability'],r.final_fields.values['E']
            def run(e):
                valid,field=evaluate(jnp.asarray(e))
                return bool(valid),np.asarray(field)
        cutoff=(sine*wavelength)**2
        for epsilon in (1.,.8*cutoff,1.2*cutoff):
            eager=fd.check(problem(epsilon),numerics=numerics)
            assert eager.ok==(epsilon<cutoff)
            live,field=run(epsilon)
            assert live==eager.ok
            assert np.isfinite(field).all() if live else np.isnan(field).all()


def test_zero_order_above_one_is_not_silently_omitted():
    with warnings.catch_warnings(),bounded():
        warnings.simplefilter('ignore')
        seed,n=public_case(wavelength=1.,sine=0.)
        prepared=fd.prepare(seed(),numerics=n)
        impossible,n=public_case(wavelength=10.,sine=0.)
        assert not fd.check(impossible(),numerics=n).ok
        scene,_=lower_problem(impossible(),n)
        bands=grazing.refused_bands(scene,prepared._plan,4.)
        assert bands
        assert all(held==((0.,math.inf),) for held in bands.values())


@pytest.mark.parametrize('c',[.2,.5,math.nextafter(1.,0.)])
def test_subunit_endpoints_keep_the_existing_eager_rounding(c):
    with bounded(2):
        low,high=grazing._rounded_band(2.,3.,1.,c)
    assert ratio(low,2.,3.,1.)>=1.
    assert ratio(math.nextafter(low,math.inf),2.,3.,1.)<1.
    assert accepted(high,c)
    assert not accepted(math.nextafter(high,-math.inf),c)
