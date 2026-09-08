"""Public-API Meep source physically below an asymmetric stack; no Meep import."""
import json
from pathlib import Path

import numpy as np
import wavebit


def test_bottom_scene_complex_amplitudes_against_meep_in_original_coordinates():
    ref=json.loads((Path(__file__).parent/'golden_scene_bottom_meep.json').read_text())
    results=wavebit.fdtd(structure=np.asarray(ref['epsilon']).reshape(-1,1,1),
                         thickness=ref['thickness'],top=1.,bot=1.,period=.1,
                         wavelength=ref['wavelengths'],backend=0,incidence_side='bottom',
                         resolution=ref['resolution'],n_periods=35,truncation_order=0,tetm='te')
    for key in ('r','t'):
        expected=np.array(ref[key+'_real'])+1j*np.array(ref[key+'_imag'])
        actual=np.array([getattr(r.given,key+'_s').item() for r in results])
        # Different Yee time steps and interface sampling: measured <6e-4.
        np.testing.assert_allclose(actual,expected,atol=.0015,rtol=0)
