"""Nonlinear material parameterizations must not bypass the first-order contract."""
import numpy as np
import pytest
from wavebit._common.backend import array_namespace, eig
from wavebit.rcwa._engine.modal_gradient import _stop, generator, exponential


@pytest.mark.parametrize('backend',['jax','jit','torch'])
@pytest.mark.parametrize('parameterization',['squared','sigmoid'])
@pytest.mark.parametrize('operation',['root','exponential'])
def test_nonlinear_material_hessian_is_refused(backend,parameterization,operation):
    def loss(t):
        xp=array_namespace(t)
        material=t*t if parameterization=='squared' else 1/(1+xp.exp(-t))
        a=xp.reshape(4.+material,(1,1))
        w,W=eig(_stop(a))
        root=operation=='root'
        q=generator(xp,a,w,W,xp.sqrt(w) if root else w,square_root=root)
        return xp.real(xp.sum(q if root else exponential(xp,q,.31)))
    sigmoid=1/(1+np.exp(-1.))
    a=5. if parameterization=='squared' else 4.+sigmoid
    da=2. if parameterization=='squared' else sigmoid*(1-sigmoid)
    expected=da/(2*np.sqrt(a)) if operation=='root' else .31*da*np.exp(.31*a)
    if backend in ('jax','jit'):
        jax=pytest.importorskip('jax')
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64',True)
        derivative=jax.grad(loss)
        hessian=jax.grad(derivative)
        if backend=='jit':
            derivative,hessian=jax.jit(derivative),jax.jit(hessian)
        assert float(derivative(jnp.asarray(1.)))==pytest.approx(expected,rel=2e-12)
        with pytest.raises(NotImplementedError,match='first order'):
            hessian(jnp.asarray(1.))
    else:
        torch=pytest.importorskip('torch')
        t=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
        derivative,=torch.autograd.grad(loss(t),t)
        assert float(derivative)==pytest.approx(expected,rel=2e-12)
        with pytest.raises(RuntimeError,match='first order'):
            gradient,=torch.autograd.grad(loss(t),t,create_graph=True)
            torch.autograd.grad(gradient,t)
