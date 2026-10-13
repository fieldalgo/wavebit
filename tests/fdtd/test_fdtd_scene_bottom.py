"""Bottom incidence: physical Fresnel frames, original-coordinate fields and graphs."""

from wavebit.compat import legacy as _legacy_wavebit
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
    actual=_legacy_wavebit.fdtd(structure=structure,mu=mu,backend='numpy',**kw)
    reference=_legacy_wavebit.rcwa(structure=structure,mu=mu,backend='numpy',
                           **{k:v for k,v in kw.items() if k not in ('resolution','n_periods')}).solve()
    for key in ('r_s','r_p','t_s','t_p'):
        np.testing.assert_allclose(getattr(actual.incident,key),getattr(reference.incident,key),atol=.015)
    np.testing.assert_allclose(actual.R+actual.T,1,atol=.01)


def test_bottom_crossed_dispersive_scene_is_exact_mirror_without_mutating_input():
    structure=np.array([[[1.4,2.],[2.1,1.6]],[[2.2,1.8],[1.2,1.9]]])
    labels=np.array([[[1,0],[0,1]],[[0,0],[0,0]]])
    original=structure.copy()
    kw=settings(period=[.4,.4],resolution=10,tetm=None,theta=.18,phi=.53,polarization=(1.,1j),
                dispersion=labels,poles={1:[(8.,.4,2.)]},return_diagnostics=True,
                record_field=15,field_monitors={'exit':.54})
    actual,d=_legacy_wavebit.fdtd(structure=structure,backend='numpy',incidence_side='bottom',**kw)
    depth=sum(kw['thickness'])
    mirror=dict(kw,thickness=kw['thickness'][::-1],top=kw['bot'],bot=kw['top'],
                dispersion=labels[::-1],field_monitors={'exit':-.2})
    expected,e=_legacy_wavebit.fdtd(structure=structure[::-1],backend='numpy',**mirror)
    np.testing.assert_array_equal(structure,original)
    for key in ('r_s','r_p','t_s','t_p','R','T'):
        np.testing.assert_array_equal(getattr(actual.incident,key),getattr(expected.incident,key))
    signs=dict(ex=1,ey=1,ez=-1,hx=-1,hy=-1,hz=1)
    for name,sign in signs.items():
        np.testing.assert_array_equal(d['fields'][name]['values'],sign*e['fields'][name]['values'][:,::-1])
        np.testing.assert_allclose(d['fields'][name]['z'],depth-e['fields'][name]['z'][::-1])
        assert np.all(np.diff(d['fields'][name]['z'])>0)
    # Reflected monitor coordinates differ by a final floating-point subtraction.
    np.testing.assert_allclose(d['monitors']['exit']['power_z'],
                               -e['monitors']['exit']['power_z'], atol=2e-15, rtol=0)
    assert d['monitors']['exit']['power_z'].item()>0
    assert d['source_z']<0
    np.testing.assert_allclose(d['layer_planes'],[0.,.13,.34])
    assert d['reflection_z']==0 and d['transmission_z']==depth


@pytest.mark.parametrize('backend',['jax','torch'])
def test_typed_bottom_material_gradient_matches_numpy(backend):
    from wavebit.compat.legacy import Stack, Incidence
    def run(value,which):
        structure=value*ones if which != 'numpy' else value*np.array([[[1.3]],[[1.]]])
        stack=Stack.from_structure(structure,[.21,.13],top=1.,bot=2.25,period=.4)
        result=_legacy_wavebit.fdtd(stack,Incidence(wavelength=1.,tetm='tm',incidence_side='bottom'),
                            backend=which,resolution=10,n_periods=30,truncation_order=0)
        return result.T.sum()
    if backend=='jax':
        jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
        import jax.numpy as jnp
        ones=jnp.array([[[1.3]],[[1.]]])
        value,derivative=jax.jit(jax.value_and_grad(lambda v:run(v,'jax')))(2.)
    else:
        torch=pytest.importorskip('torch')
        ones=torch.tensor([[[1.3]],[[1.]]],dtype=torch.float64)
        parameter=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
        value=run(parameter,'torch')
        derivative=torch.autograd.grad(value,parameter)[0].detach().numpy()
        value=value.detach().numpy()
    np.testing.assert_allclose(value,run(2.,'numpy'),atol=1e-11)
    delta=1e-4
    finite=(run(2.+delta,'numpy')-run(2.-delta,'numpy'))/(2*delta)
    np.testing.assert_allclose(derivative,finite,rtol=2e-5,atol=1e-7)
    assert abs(finite)>.01


def test_bottom_vacuum_records_have_upward_poynting_and_original_plot_coordinates():
    _,d=_legacy_wavebit.fdtd(structure=np.ones((2,1,1)),backend='numpy',incidence_side='bottom',
                     **settings(bot=1.,resolution=10,record_field=2,return_diagnostics=True))
    sample=_legacy_wavebit.fdtd_poynting(d,z=.1)
    assert sample['S'][:,2].sum()>0
    assert sample['region']=='total'
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    fig,ax,_=_legacy_wavebit.plot_fdtd_field(d,component='hy',frame=-1)
    source=[line for line in ax.lines if 'Current sheet' in line.get_label()][0]
    np.testing.assert_allclose(source.get_ydata(),d['source_z'])
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_lossy_stack_transmission_reciprocity_at_matched_transverse_wavevector():
    structure=np.array([[[2.]],[[3.]]]);labels=np.array([[[1]],[[0]]])
    gaps=[]
    for resolution in (40,80):
        kw=settings(period=.1,resolution=resolution,dispersion=labels,poles={1:[(8.,.8,6.)]},theta=.25)
        top=_legacy_wavebit.fdtd(structure=structure,backend='numpy',**kw)
        bottom=_legacy_wavebit.fdtd(structure=structure,backend='numpy',incidence_side='bottom',
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
    kw=dict(settings(),structure=np.ones((2,1,1)),incidence_side='bottom',backend='numpy')
    kw.update(changes)
    with pytest.raises(ValueError,match=match):
        _legacy_wavebit.fdtd(**kw)
