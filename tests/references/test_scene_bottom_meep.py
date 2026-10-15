"""Public-API Meep source physically below an asymmetric stack; no Meep import."""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import json

import numpy as np


def test_bottom_scene_complex_amplitudes_against_meep_in_original_coordinates():
    ref=json.loads((repository_path('tests/references/golden_scene_bottom_meep.json')).read_text())
    results=_legacy_wavebit_fdtd(structure=np.asarray(ref['epsilon'])[::-1].reshape(-1,1,1),
                         thickness=ref['thickness'][::-1],top=1.,bot=1.,period=.1,
                         wavelength=ref['wavelengths'],backend='numpy',incidence_side='bottom',
                         resolution=ref['resolution'],n_periods=35,truncation_order=0,tetm='te')
    for key in ('r','t'):
        expected=np.array(ref[key+'_real'])+1j*np.array(ref[key+'_imag'])
        actual=np.array([getattr(r.incident,key+'_s').item() for r in results])
        # Different Yee time steps and interface sampling: measured <6e-4.
        np.testing.assert_allclose(actual,expected,atol=.0015,rtol=0)
