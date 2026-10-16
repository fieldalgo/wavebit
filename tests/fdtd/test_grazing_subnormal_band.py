"""A valid explicit public Axis1D wavevector can put the band below binary64's floor."""
import math
import warnings
import numpy as np
import pytest
from wavebit import fdtd as fd
from wavebit.fdtd._prepare import grazing
from tests.fdtd.test_grazing_cos_boundary import public_case,bounded

KU=math.ldexp(math.sqrt(2.),-535)


@pytest.mark.parametrize('c',[.5,1.,2.])
def test_zero_lower_sentinel_keeps_eager_positive_float_endpoints(c):
    with bounded(2):
        low,high=grazing._rounded_band(KU,0.,1.,c)
    assert low==0.  # No positive float lies below the first propagating one.
    first=math.nextafter(0.,math.inf)
    values=[first,math.nextafter(high,-math.inf)]
    if math.isfinite(high):values.extend([high,math.nextafter(high,math.inf)])
    for value in values:
        orders=grazing._orders(((KU,None,True),(0.,None,True)),math.sqrt(value),1.)
        assert orders
        assert (low<value<high)==(orders[0][1]<c)


@pytest.mark.parametrize('backend',['torch','jax'])
def test_explicit_tiny_axis1d_keeps_admitted_public_traced_run(backend):
    seed,numerics=public_case(1.,1e-8,sine=0.)
    def problem(epsilon=1.):
        p=seed(epsilon)
        return p.replace(scene=p.scene.replace(boundaries=fd.Boundaries(
            x=fd.Axis1D(k=KU),y=fd.Axis1D(k=0.),z=fd.Open())))
    with warnings.catch_warnings(),bounded(20):
        warnings.simplefilter('ignore')
        assert fd.check(problem(),numerics=numerics).ok
        if backend=='torch':
            torch=pytest.importorskip('torch')
            prepared=fd.prepare(problem(),numerics=numerics,backend=backend)
            result=prepared.run(problem(torch.tensor(1.,dtype=torch.float64,requires_grad=True)))
            valid=bool(result.validity.checks['runtime_stability'])
            fields=[value.detach().cpu().numpy() for value in result.final_fields.values.values()]
        else:
            jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
            import jax.numpy as jnp
            prepared=fd.prepare(problem(),numerics=numerics,backend=backend)
            @jax.jit
            def run(e):
                r=prepared.run(problem(e))
                return r.validity.checks['runtime_stability'],tuple(r.final_fields.values.values())
            valid,fields=run(jnp.asarray(1.))
        assert bool(valid)
        assert all(np.isfinite(np.asarray(field)).all() for field in fields)
