"""Temporary recipe edits include NumPy leaves and arrays inside Materials."""
import numpy as np
import pytest
import wavebit

jax=pytest.importorskip('jax')


def make(backend='jax',width=100.,epsilon=(4.,3.,2.),thickness=170.):
    recipe=[dict(background=1.,shapes=[dict(type='rectangle',lx=np.array(width),ly=100.,
                                           material=wavebit.Material(np.array(epsilon)))])]
    return wavebit.rcwa(backend=backend,structure=recipe,period=(600.,750.),thickness=[thickness],
                       wavelength=900.,theta=.13,phi=.21,truncation_order=(0,0),factorization='li')


@pytest.mark.parametrize('helper',['grad','fit'])
@pytest.mark.parametrize('slot',['width','material'])
@pytest.mark.parametrize('phase',['forward','loss'])
@pytest.mark.parametrize('failure',[False,True])
def test_numpy_recipe_edits_are_isolated_in_success_and_failure(helper,slot,phase,failure):
    s=make();original=s.structure
    width=original[0]['shapes'][0]['lx'];epsilon=original[0]['shapes'][0]['material'].epsilon
    def mutate():
        shape=s.structure[0]['shapes'][0]
        if slot=='width':shape['lx'][...]=120.
        else:shape['material'].epsilon[0]=5.
        if failure:raise RuntimeError('intentional mutable recipe failure')
    def forward(thickness,geometry=None):
        if phase=='forward':mutate()
        return s.solve(thickness=thickness,geometry=geometry)
    def loss(result):
        value=result.T.sum().real
        if phase=='loss':mutate()
        return value
    def run():
        if helper=='grad':return s.grad(['thickness'],forward,loss)['thickness'][0]
        optax=pytest.importorskip('optax')
        return s.fit(['thickness'],forward,loss,optax.sgd(.1),iteration=1,
                     prepare_geometry=s.freeze_geometry)['thickness'][0]
    with jax.checking_leaks():
        if failure:
            with pytest.raises(RuntimeError,match='intentional mutable recipe failure'):run()
        else:got=float(run())
    assert s.structure is original
    assert float(width)==100.
    np.testing.assert_array_equal(epsilon,[4.,3.,2.])
    assert s.T1 is None and s.layer_info_list==[]
    if not failure:
        def independent(t):
            cfg=({'width':120.} if slot=='width' else {'epsilon':(5.,3.,2.)}) if phase=='forward' else {}
            return float(make('numpy',thickness=t,**cfg).solve().T.sum().real)
        h=1e-3;gradient=(independent(170.+h)-independent(170.-h))/(2*h)
        expected=gradient if helper=='grad' else 170.-.1*gradient
        assert got==pytest.approx(expected,abs=2e-9,rel=2e-5)
    final_t=float(np.asarray(s.thickness)[0])
    np.testing.assert_allclose(s.solve().T,make('numpy',thickness=final_t).solve().T,atol=2e-12,rtol=2e-12)
