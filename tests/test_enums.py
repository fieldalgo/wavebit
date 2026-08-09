"""Typed enums are a non-breaking upgrade over the raw values the solver stores.

``Pol`` / ``Backend`` are ``IntEnum``s equal to their int codes; ``FourierCoefficients`` /
``Factorization`` are ``str`` enums equal to their lowercase strings. Passing the enum must
give byte-identical results to passing the raw value.
"""
import numpy as np

import fieldwave
from fieldwave import Backend, Factorization, FourierCoefficients, FourierType, Pol


def test_intenum_equals_int_codes():
    assert Pol.TE == 0 and Pol.TM == 1
    assert Backend.NUMPY == 0 and Backend.JAX == 1 and Backend.TORCH == 2


def test_str_enums_equal_their_strings():
    assert FourierCoefficients.SAMPLED == 'sampled'
    assert FourierCoefficients.INTEGRATED == 'integrated'
    assert Factorization.LI == 'li' and Factorization.NVF == 'nvf'


def test_deprecated_fourier_type_still_equals_its_int_codes():
    """``FourierType`` is deprecated but must keep comparing as it always did."""
    assert FourierType.DFS == 0 and FourierType.CONTINUOUS == 1 and FourierType.NVF == 2
    assert FourierType.NVF in (2, 'nvf', 'NVF')


def _solve(pol, backend, **strategy):
    ucell = np.array([[[3.0, 3.0, 3.0, 1.0, 1.0, 1.0]]])
    solver = fieldwave.rcwa(backend=backend, n_top=2.0, n_bot=1.0, theta=0.0, phi=None,
                          pol=pol, fto=[6, 0], period=[0.77], wavelength=0.777,
                          thickness=[0.15], ucell=ucell, **strategy)
    r = solver.conv_solve()
    return float(np.array(r.de_ri).sum()), float(np.array(r.de_ti).sum())


def test_enum_matches_raw_api():
    for pol_e, pol_i in [(Pol.TE, 0), (Pol.TM, 1)]:
        a = _solve(pol_e, Backend.NUMPY, fourier_coefficients=FourierCoefficients.SAMPLED,
                   factorization=Factorization.LI)
        b = _solve(pol_i, 0, fourier_coefficients='sampled', factorization='li')
        assert np.allclose(a, b, atol=0, rtol=0), (pol_e, a, b)
