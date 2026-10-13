"""Streaming phasors must equal recorded DFTs without retaining a time history."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

from wavebit.fdtd._periodic.legacy import gaussian_pulse, SOURCE_CYCLES, SOURCE_DELAY


def test_streaming_field_is_the_offline_dft_and_does_not_change_diffraction():
    kw=dict(structure=np.ones((1,1,2)), thickness=[.1],top=1.,bot=1.,period=[.2],
            wavelength=.9,tetm='te',resolution=20.,n_periods=80,return_diagnostics=True)
    result,d=_legacy_wavebit.fdtd(**kw,field_monitors={'inside':.05},record_field=1,record_components='all')
    plain,p=_legacy_wavebit.fdtd(**kw,field_monitors={'inside':.05})
    np.testing.assert_array_equal(result.incident.t_s,plain.incident.t_s)
    assert 'field_e' not in p
    m=d['monitors']['inside']
    f=d['fields']['ey'];z=f['z'];hi=np.searchsorted(z,.05);lo=hi-1
    trace=f['values'][:,lo]*(z[hi]-.05)/(z[hi]-z[lo])+f['values'][:,hi]*(.05-z[lo])/(z[hi]-z[lo])
    kernel=np.exp(2j*np.pi*f['t']/.9)*d['dt']
    source=np.sum(kernel*gaussian_pulse(f['t'],1/.9,SOURCE_CYCLES*.9,SOURCE_DELAY*SOURCE_CYCLES*.9))
    expected=np.sum(trace*kernel[:,None,None],axis=0)/source
    np.testing.assert_allclose(m['fields']['ey'][0],expected,atol=1e-12)
    np.testing.assert_array_equal(m['fields']['ey'],p['monitors']['inside']['fields']['ey'])
    assert m['power_z'][0]<0


@pytest.mark.parametrize('crossed',[False,True])
def test_elliptical_planes_coherently_mix_all_six_components(crossed):
    kw=dict(structure=np.ones((1,2 if crossed else 1,2)),thickness=[.1],top=1.,bot=1.,
            period=[.2,.3] if crossed else [.2],wavelength=[.85,.9,.95],theta=.3,phi=.4,
            resolution=20.,n_periods=80,return_diagnostics=True,field_monitors={'probe':.05})
    _,s=_legacy_wavebit.fdtd(**kw,polarization=(1,0))
    _,p=_legacy_wavebit.fdtd(**kw,polarization=(0,1))
    _,mix=_legacy_wavebit.fdtd(**kw,polarization=(1,1j))
    for name in ('ex','ey','ez','hx','hy','hz'):
        np.testing.assert_allclose(mix['monitors']['probe']['fields'][name],
                                   (s['monitors']['probe']['fields'][name]+1j*p['monitors']['probe']['fields'][name])/np.sqrt(2),atol=1e-12)
    assert np.all(mix['monitors']['probe']['power_z']<0)


def test_plane_flux_ratios_recover_slab_reflection_and_transmission():
    common=dict(thickness=[.1],top=1.,bot=1.,period=[.2],wavelength=.9,tetm='te',
                resolution=30.,n_periods=100,return_diagnostics=True,
                field_monitors={'reflection':.75,'transmission':-.2})
    _,empty=_legacy_wavebit.fdtd(structure=np.ones((1,1,2)),**common)
    result,film=_legacy_wavebit.fdtd(structure=np.full((1,1,2),2.25),**common)
    incident=empty['monitors']['transmission']['power_z'][0]
    reflection=-film['monitors']['reflection']['power_z'][0]/incident
    transmission=film['monitors']['transmission']['power_z'][0]/incident
    np.testing.assert_allclose(reflection,result.R.sum(),atol=.01,rtol=0)
    np.testing.assert_allclose(transmission,result.T.sum(),atol=.01,rtol=0)
    assert film['monitors']['reflection']['region']=='scattered'
    assert film['monitors']['transmission']['region']=='total'


@pytest.mark.parametrize('planes', [{}, {'':.05}, {'bad':100.}, {'bad':float('nan')}])
def test_invalid_monitor_specifications_are_refused(planes):
    with pytest.raises(ValueError):
        _legacy_wavebit.fdtd(structure=np.ones((1,1,2)),thickness=[.1],top=1.,bot=1.,
                     period=[.2],wavelength=.9,tetm='te',resolution=20.,n_periods=20,
                     return_diagnostics=True,field_monitors=planes)
