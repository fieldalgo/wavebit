"""Measure near fields and spectral plane power without a time history."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np


def main():
    common=dict(thickness=[.1],top=1.,bot=1.,period=[.2],wavelength=[.85,.9,.95],
                tetm='te',resolution=30.,n_periods=100,return_diagnostics=True,
                field_monitors={'reflection':.75,'transmission':-.2})
    _,reference=_legacy_wavebit.fdtd(structure=np.ones((1,1,2)),**common)
    result,diag=_legacy_wavebit.fdtd(structure=np.full((1,1,2),2.25),**common)
    # +z points up: top illumination carries negative transmitted plane power.
    incident=reference['monitors']['transmission']['power_z']
    reflected=-diag['monitors']['reflection']['power_z']/incident
    transmitted=diag['monitors']['transmission']['power_z']/incident
    print('Plane reflection:',reflected)
    print('Plane transmission:',transmitted)
    print('Near-field Ey shape:',diag['monitors']['transmission']['fields']['ey'].shape)
    np.testing.assert_allclose(reflected,[r.R.sum() for r in result],atol=.01)
    np.testing.assert_allclose(transmitted,[r.T.sum() for r in result],atol=.01)
    assert 'field_e' not in diag


if __name__=='__main__':
    main()
