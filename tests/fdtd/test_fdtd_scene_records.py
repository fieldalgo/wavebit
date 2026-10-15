"""Shared-scene monitors and causal carrier records, including autograd."""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd

from wavebit.fdtd._periodic.plotting import plot_fdtd_field as _legacy_wavebit_plot_fdtd_field

from wavebit.fdtd._periodic.probes import sample_fdtd_fields as _legacy_wavebit_sample_fdtd_fields
import numpy as np
import pytest


def settings(**kw):
    return dict(dict(thickness=[.2],top=1.,bot=1.,period=.4,wavelength=1.,resolution=10,
                     n_periods=25,truncation_order=0,tetm='te',return_diagnostics=True),**kw)


def test_record_dft_equals_normalized_monitor_and_preserves_diffraction():
    material=np.full((1,1,1),2.)
    result,d=_legacy_wavebit_fdtd(structure=material,backend='numpy',record_field=1,
                         field_monitors={'out':-.45},**settings())
    bare,_=_legacy_wavebit_fdtd(structure=material,backend='numpy',**settings())
    np.testing.assert_array_equal(result.T,bare.T)
    field=d['fields']['ey']
    time=field['t']; z=field['z']; dz=d['spacing'][0]
    hi=np.searchsorted(z,-.45);lo=hi-1; fraction=(-.45-z[lo])/dz
    plane=(1-fraction)*field['values'][:,lo]+fraction*field['values'][:,hi]
    drive_time=np.arange(d['steps'])*d['dt']
    pulse=np.exp(-((drive_time-5.)**2))*np.exp(-2j*np.pi*drive_time)
    source=np.sum(pulse*np.exp(2j*np.pi*(drive_time+d['dt'])))*d['dt']
    offline=np.sum(plane*np.exp(2j*np.pi*time)[:,None,None],axis=0)*d['dt']/source
    np.testing.assert_allclose(offline,d['monitors']['out']['fields']['ey'][0],atol=1e-12)
    assert d['monitors']['out']['region']=='total'
    assert d['fields']['hx']['t'][0] == d['dt']/2
    sample=_legacy_wavebit_sample_fdtd_fields(d,z=-.4)
    assert sample['region']=='total'
    assert np.isfinite(sample['E']).all()
    with pytest.raises(ValueError,match='source discontinuity'):
        _legacy_wavebit_sample_fdtd_fields(d,z=d['source_z'])


@pytest.mark.parametrize('backend',['jax','torch'])
def test_monitor_and_history_material_gradients_and_numpy_parity(backend):
    kw=settings(record_field=13,record_components=['ey','hx'],field_monitors={'out':-.45})
    def run(value,which):
        _,d=_legacy_wavebit_fdtd(structure=value*ones if which != 'numpy' else value*np.ones((1,1,1)),backend=which,**kw)
        return d['fields']['ey']['values'],d['monitors']['out']['power_z']
    if backend=='jax':
        jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
        import jax.numpy as jnp
        ones=jnp.ones((1,1,1))
        execute=jax.jit(lambda v:run(v,'jax'))
        values,power=execute(2.)
        derivative=jax.grad(lambda v:abs(execute(v)[0]).sum()+execute(v)[1].sum())(2.)
        host=np.asarray
    else:
        torch=pytest.importorskip('torch')
        ones=torch.ones((1,1,1),dtype=torch.float64)
        value=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
        values,power=run(value,'torch')
        derivative=torch.autograd.grad(abs(values).sum()+power.sum(),value)[0].detach().numpy()
        host=lambda x:x.detach().numpy()
    expected,expected_power=run(2.,'numpy')
    np.testing.assert_allclose(host(values),expected,atol=1e-11)
    np.testing.assert_allclose(host(power),expected_power,atol=1e-11)
    def loss(v):
        a,b=run(v,'numpy')
        return abs(a).sum()+b.sum()
    delta=1e-4
    finite=(loss(2.+delta)-loss(2.-delta))/(2*delta)
    np.testing.assert_allclose(derivative,finite,rtol=2e-4,atol=1e-5)


def test_circular_carrier_preserves_quadrature_and_plots_current_sheet():
    _,d=_legacy_wavebit_fdtd(structure=np.ones((1,1,1)),backend='numpy',
                     **settings(record_field=3,tetm=None,polarization=(1.,1j)))
    ex=d['fields']['ex']['values'];ey=d['fields']['ey']['values']
    np.testing.assert_allclose(ex,-1j*ey,atol=1e-12)
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    fig,ax,_=_legacy_wavebit_plot_fdtd_field(d,component='hx',frame=-1)
    assert any('Current sheet' in t for t in ax.get_legend_handles_labels()[1])
    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.parametrize('extra,match',[
    ({'record_field':True},'positive integer'),
    ({'record_components':'all'},'requires record_field'),
    ({'field_monitors':{}},'nonempty mapping'),
    ({'field_monitors':{'':0.}},'nonempty strings'),
    ({'field_monitors':{'outside':-100.}},'outside the PML'),
])
def test_invalid_record_and_monitor_requests(extra,match):
    with pytest.raises(ValueError,match=match):
        _legacy_wavebit_fdtd(structure=np.ones((1,1,1)),backend='numpy',**settings(**extra))


def test_time_calibration_does_not_add_grazing_diffraction_orders_to_band():
    # Carrier=1 is exactly grazing for m=+-1, but neither requested wavelength is.
    result,diag=_legacy_wavebit_fdtd(structure=np.ones((1,1,1)),backend='numpy',
                             **settings(period=1.,wavelength=[.9,1.1],truncation_order=1,
                                        record_field=20))
    assert len(result)==2
    assert np.isfinite(diag['fields']['ey']['values']).all()
    np.testing.assert_allclose([r.T.sum() for r in result],1,atol=.002)
