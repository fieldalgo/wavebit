"""Temporary cell and ambient edits preserve caller-owned NumPy conditions."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest
import wavebit


@pytest.mark.parametrize('helper',['grad','fit'])
@pytest.mark.parametrize('phase',['forward','loss'])
@pytest.mark.parametrize('failure',[False,True])
@pytest.mark.parametrize('slot',['lattice','drawn-cell','top','bot'])
def test_numpy_lattice_is_preserved_by_temporary_helper_state(helper,phase,failure,slot):
    jax=pytest.importorskip('jax')
    _legacy_wavebit.enable_x64()
    cell=np.array([[600.,0.],[0.,750.]])
    before=cell.copy()
    eps_top=np.array(1.);eps_bot=np.array(1.2)
    raw=np.array([[[4.+.07j]]])
    structure=wavebit.Lattice(cell[:,0],cell[:,1],True).drawn(raw) if slot=='drawn-cell' else raw
    s=_legacy_wavebit.rcwa(backend='jax',lattice=cell,structure=structure,thickness=[170.],
                   top=wavebit.Material(eps_top),bot=wavebit.Material(eps_bot),
                   wavelength=900.,truncation_order=(0,0),theta=.17,phi=.29,factorization='li')
    original=s.lattice
    original_drawn=s.structure_drawn_on
    original_top=s.top;original_bot=s.bot
    def mutate():
        if slot in ('lattice','drawn-cell'):
            s.lattice.u[0]=610.
            if slot=='drawn-cell':s.structure_drawn_on[0,0]=610.
        else:getattr(s,slot).epsilon[...]=1.4
        if failure:raise RuntimeError('intentional lattice callback failure')
    def forward(thickness,geometry=None):
        if phase=='forward':mutate()
        return s.solve(thickness=thickness,geometry=geometry)
    def loss(result):
        value=result.T.sum().real
        if phase=='loss':mutate()
        return value
    def run():
        if helper=='grad':return s.grad(['thickness'],forward,loss)
        optax=pytest.importorskip('optax')
        return s.fit(['thickness'],forward,loss,optax.sgd(.1),iteration=1,prepare_geometry=s.freeze_geometry)
    with jax.checking_leaks():
        if failure:
            with pytest.raises(RuntimeError,match='intentional lattice callback failure'):run()
        else:run()
    assert s.lattice is original
    np.testing.assert_array_equal(cell,before)
    np.testing.assert_array_equal(s.lattice.columns,before)
    assert s.top is original_top and s.bot is original_bot
    assert s.structure_drawn_on is original_drawn
    assert float(eps_top)==1. and float(eps_bot)==1.2
    if original_drawn is not None:np.testing.assert_array_equal(original_drawn,before)


@pytest.mark.parametrize('changed',[False,True])
def test_jitted_grad_retains_validated_drawn_cell_identity(changed):
    jax=pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit.enable_x64()
    cell=wavebit.Lattice.rectangular(600.,750.)
    raw=np.array([[[4.+.07j]]])
    s=_legacy_wavebit.rcwa(backend='jax',lattice=cell,structure=cell.drawn(raw),thickness=[170.],
                   wavelength=900.,truncation_order=(0,0),theta=.17,phi=.29,factorization='li')
    def calculation(solver):
        def forward(thickness):
            if changed:solver.lattice=wavebit.Lattice.rectangular(jnp.asarray(610.),jnp.asarray(750.))
            return solver.solve(thickness=thickness)
        return solver.grad(['thickness'],forward,lambda result:result.T.sum().real)['thickness'][0]
    with jax.checking_leaks():
        if changed:
            with pytest.raises(ValueError,match='no value|cannot be compared'):
                jax.jit(calculation)(s)
        else:
            got=float(jax.jit(calculation)(s))
            def reference(t):
                return float(_legacy_wavebit.rcwa(backend='numpy',lattice=cell,structure=cell.drawn(raw),
                    thickness=[t],wavelength=900.,truncation_order=(0,0),theta=.17,phi=.29,
                    factorization='li').solve().T.sum().real)
            fd=(reference(170.001)-reference(169.999))/.002
            assert got==pytest.approx(fd,abs=2e-10,rel=2e-6)
