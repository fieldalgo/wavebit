"""Bottom incidence: physical Fresnel frames, original-coordinate fields and graphs."""
import numpy as np
import pytest
import wavebit


def settings(**changes):
    return dict(dict(thickness=[.13,.21],top=1.,bot=2.25,period=.4,wavelength=1.,
                     resolution=30,n_periods=35,truncation_order=0,tetm='te'),**changes)


@pytest.mark.parametrize('polarization', ['te','tm','jones'])
def test_bottom_complex_magnetic_stack_against_rcwa(polarization):
    kw=settings(top=wavebit.Material(1.2,1.1),bot=wavebit.Material(2.,1.3),
                theta=.27,resolution=40,incidence_side='bottom')
    if polarization=='jones':
        kw.update(tetm=None,phi=.41,polarization=(1.,.6j))
    else:
        kw['tetm']=polarization
    structure=np.array([[[2.1]],[[3.2]]]);mu=np.array([[[1.1]],[[1.3]]])
    actual=wavebit.fdtd(structure=structure,mu=mu,backend=0,**kw)
    reference=wavebit.rcwa(structure=structure,mu=mu,backend=0,
                           **{k:v for k,v in kw.items() if k not in ('resolution','n_periods')}).solve()
    for key in ('r_s','r_p','t_s','t_p'):
        np.testing.assert_allclose(getattr(actual.given,key),getattr(reference.given,key),atol=.015)
    np.testing.assert_allclose(actual.R+actual.T,1,atol=.01)


def test_bottom_crossed_dispersive_scene_is_exact_mirror_without_mutating_input():
    structure=np.array([[[1.4,2.],[2.1,1.6]],[[2.2,1.8],[1.2,1.9]]])
    labels=np.array([[[1,0],[0,1]],[[0,0],[0,0]]])
    original=structure.copy()
    kw=settings(period=[.4,.4],resolution=10,tetm=None,theta=.18,phi=.53,polarization=(1.,1j),
                dispersion=labels,poles={1:[(8.,.4,2.)]},return_diagnostics=True,
                record_field=15,field_monitors={'exit':-.2})
    actual,d=wavebit.fdtd(structure=structure,backend=0,incidence_side='bottom',**kw)
    depth=sum(kw['thickness'])
    mirror=dict(kw,thickness=kw['thickness'][::-1],top=kw['bot'],bot=kw['top'],
                dispersion=labels[::-1],field_monitors={'exit':depth+.2})
    expected,e=wavebit.fdtd(structure=structure[::-1],backend=0,**mirror)
    np.testing.assert_array_equal(structure,original)
    for key in ('r_s','r_p','t_s','t_p','R','T'):
        np.testing.assert_array_equal(getattr(actual.given,key),getattr(expected.given,key))
    signs=dict(ex=1,ey=1,ez=-1,hx=-1,hy=-1,hz=1)
    for name,sign in signs.items():
        np.testing.assert_array_equal(d['fields'][name]['values'],sign*e['fields'][name]['values'][:,::-1])
        np.testing.assert_allclose(d['fields'][name]['z'],depth-e['fields'][name]['z'][::-1])
        assert np.all(np.diff(d['fields'][name]['z'])>0)
    np.testing.assert_array_equal(d['monitors']['exit']['power_z'],-e['monitors']['exit']['power_z'])
    assert d['monitors']['exit']['power_z'].item()<0
    assert d['source_z']>depth
    np.testing.assert_allclose(d['layer_planes'],[0.,.13,.34])
    assert d['reflection_z']==depth and d['transmission_z']==0


@pytest.mark.parametrize('backend',[1,2])
def test_typed_bottom_material_gradient_matches_numpy(backend):
    from wavebit import Stack,Incidence
    def run(value,which):
        structure=value*ones if which else value*np.array([[[1.]],[[1.3]]])
        stack=Stack.from_structure(structure,[.13,.21],top=1.,bot=2.25,period=.4)
        result=wavebit.fdtd(stack,Incidence(wavelength=1.,tetm='tm',incidence_side='bottom'),
                            backend=which,resolution=10,n_periods=30,truncation_order=0)
        return result.T.sum()
    if backend==1:
        jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
        import jax.numpy as jnp
        ones=jnp.array([[[1.]],[[1.3]]])
        value,derivative=jax.jit(jax.value_and_grad(lambda v:run(v,1)))(2.)
    else:
        torch=pytest.importorskip('torch')
        ones=torch.tensor([[[1.]],[[1.3]]],dtype=torch.float64)
        parameter=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
        value=run(parameter,2)
        derivative=torch.autograd.grad(value,parameter)[0].detach().numpy()
        value=value.detach().numpy()
    np.testing.assert_allclose(value,run(2.,0),atol=1e-11)
    delta=1e-4
    finite=(run(2.+delta,0)-run(2.-delta,0))/(2*delta)
    np.testing.assert_allclose(derivative,finite,rtol=2e-5,atol=1e-7)
    assert abs(finite)>.01


def test_bottom_vacuum_records_have_upward_poynting_and_original_plot_coordinates():
    _,d=wavebit.fdtd(structure=np.ones((2,1,1)),backend=0,incidence_side='bottom',
                     **settings(bot=1.,resolution=10,record_field=2,return_diagnostics=True))
    sample=wavebit.fdtd_poynting(d,z=.1)
    assert sample['S'][:,2].sum()<0
    assert sample['region']=='total'
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    fig,ax,_=wavebit.plot_fdtd_field(d,component='hy',frame=-1)
    source=[line for line in ax.lines if 'Current sheet' in line.get_label()][0]
    np.testing.assert_allclose(source.get_ydata(),d['source_z'])
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_lossy_stack_transmission_reciprocity_at_matched_transverse_wavevector():
    structure=np.array([[[2.]],[[3.]]]);labels=np.array([[[1]],[[0]]])
    gaps=[]
    for resolution in (40,80):
        kw=settings(period=.1,resolution=resolution,dispersion=labels,poles={1:[(8.,.8,6.)]},theta=.25)
        top=wavebit.fdtd(structure=structure,backend=0,**kw)
        bottom=wavebit.fdtd(structure=structure,backend=0,incidence_side='bottom',
                            **dict(kw,theta=np.arcsin(np.sin(.25)/1.5)))
        gaps.append(abs(top.T.item()-bottom.T.item()))
        assert abs(top.R.item()-bottom.R.item())>.001
    assert gaps[1]<.004
    assert gaps[1]<gaps[0]*.6


@pytest.mark.parametrize('changes,match',[
    ({'structure':np.ones((2,2))},'shape'),
    ({'thickness':[.2]},'thickness'),
    ({'mu':1.},'raster shape'),
    ({'incidence_side':'below'},'top.*bottom'),
])
def test_bottom_input_validation(changes,match):
    kw=dict(settings(),structure=np.ones((2,1,1)),incidence_side='bottom',backend=0)
    kw.update(changes)
    with pytest.raises(ValueError,match=match):
        wavebit.fdtd(**kw)
