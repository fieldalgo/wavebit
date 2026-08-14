"""Typed enums are a non-breaking upgrade over the raw values the solver stores.

``Pol`` / ``Backend`` are ``IntEnum``s equal to their int codes; ``FourierCoefficients`` /
``Factorization`` are ``str`` enums equal to their lowercase strings. Passing the enum must
give byte-identical results to passing the raw value.
"""
import numpy as np
import pytest

import wavebit
from wavebit import Backend, Factorization, FourierCoefficients, Pol


def test_intenum_equals_int_codes():
    assert Pol.TE == 0 and Pol.TM == 1
    assert Backend.NUMPY == 0 and Backend.JAX == 1 and Backend.TORCH == 2


def test_str_enums_equal_their_strings():
    assert FourierCoefficients.SAMPLED == 'sampled'
    assert FourierCoefficients.INTEGRATED == 'integrated'
    assert Factorization.LI == 'li' and Factorization.NVF == 'nvf'


def _solve(tetm, backend, **strategy):
    ucell = np.array([[[3.0, 3.0, 3.0, 1.0, 1.0, 1.0]]])
    solver = wavebit.rcwa(backend=backend, n_top=2.0, n_bot=1.0, theta=0.0, phi=None,
                          tetm=tetm, fto=[6, 0], period=[0.77], wavelength=0.777,
                          thickness=[0.15], ucell=ucell, **strategy)
    r = solver.conv_solve()
    return float(np.array(r.R).sum()), float(np.array(r.T).sum())


def test_enum_matches_raw_api():
    """The enum and the word are one input; the retired integer spelling is not.

    ``tetm`` takes ``Pol.TE`` or ``'te'`` and nothing else. A raw ``0`` is refused on
    purpose -- a number is what let the old ``pol`` mean a fraction, which was an angle in
    disguise rather than a declaration.
    """
    for enum_spelling, word in [(Pol.TE, 'te'), (Pol.TM, 'tm')]:
        a = _solve(enum_spelling, Backend.NUMPY,
                   fourier_coefficients=FourierCoefficients.SAMPLED,
                   factorization=Factorization.LI)
        b = _solve(word, 0, fourier_coefficients='sampled', factorization='li')
        assert np.allclose(a, b, atol=0, rtol=0), (enum_spelling, a, b)

    with pytest.raises(ValueError, match='tetm'):
        _solve(0, 0)
