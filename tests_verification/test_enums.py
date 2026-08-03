"""Typed enums (fieldwave.Pol / FourierType / Backend) are a non-breaking, IntEnum upgrade.

Passing the enum must give byte-identical results to passing the raw int code.
"""
import numpy as np

import fieldwave
from fieldwave import Pol, FourierType, Backend


def test_intenum_equals_int_codes():
    assert Pol.TE == 0 and Pol.TM == 1
    assert FourierType.DFS == 0 and FourierType.CONTINUOUS == 1 and FourierType.NVF == 2
    assert Backend.NUMPY == 0 and Backend.JAX == 1 and Backend.TORCH == 2
    # the solver's membership check for NVF still works with the enum
    assert FourierType.NVF in (2, 'nvf', 'NVF')


def _solve(pol, fourier_type, backend):
    ucell = np.array([[[3.0, 3.0, 3.0, 1.0, 1.0, 1.0]]])
    mee = fieldwave.call_mee(backend=backend, n_top=2.0, n_bot=1.0, theta=0.0, phi=None,
                          pol=pol, fto=[6, 0], period=[0.77], wavelength=0.777,
                          thickness=[0.15], fourier_type=fourier_type, ucell=ucell)
    r = mee.conv_solve()
    return float(np.array(r.de_ri).sum()), float(np.array(r.de_ti).sum())


def test_enum_matches_int_api():
    for pol_e, pol_i in [(Pol.TE, 0), (Pol.TM, 1)]:
        a = _solve(pol_e, FourierType.DFS, Backend.NUMPY)
        b = _solve(pol_i, 0, 0)
        assert np.allclose(a, b, atol=0, rtol=0), (pol_e, a, b)
