"""Typed enums are a non-breaking upgrade over the raw values the solver stores.

``Tetm`` is an ``IntEnum`` equal to its int code; ``Backend``, ``FourierCoefficients`` and
``Factorization`` are ``str`` enums equal to their lowercase strings. Passing the enum must
give byte-identical results to passing the raw value.
"""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest

from wavebit.rcwa._engine.enums import Factorization

from wavebit.rcwa._engine.enums import FourierCoefficients

from wavebit._common.enums import Tetm
from wavebit import Backend


def test_intenum_equals_int_codes():
    assert Tetm.TE == 0 and Tetm.TM == 1


def test_str_enums_equal_their_strings():
    assert Backend.NUMPY == 'numpy' and Backend.JAX == 'jax' and Backend.TORCH == 'torch'
    assert FourierCoefficients.SAMPLED == 'sampled'
    assert FourierCoefficients.INTEGRATED == 'integrated'
    assert Factorization.LI == 'li' and Factorization.NVF == 'nvf'


def _solve(tetm, backend, **strategy):
    structure = np.array([[[9.0, 9.0, 9.0, 1.0, 1.0, 1.0]]])
    solver = _legacy_wavebit_rcwa(backend=backend, top=4.0, bot=1.0, theta=0.0, phi=None,
                          tetm=tetm, truncation_order=[6, 0], period=[0.77], wavelength=0.777,
                          thickness=[0.15], structure=structure, **strategy)
    r = solver.solve()
    return float(np.array(r.R).sum()), float(np.array(r.T).sum())


def test_enum_matches_raw_api():
    """The enum and the word are one input; the retired integer spelling is not.

    ``tetm`` takes ``Tetm.TE`` or ``'te'`` and nothing else. A raw ``0`` is refused on
    purpose -- a number is what let the old ``pol`` mean a fraction, which was an angle in
    disguise rather than a declaration.
    """
    for enum_spelling, word in [(Tetm.TE, 'te'), (Tetm.TM, 'tm')]:
        a = _solve(enum_spelling, Backend.NUMPY,
                   fourier_coefficients=FourierCoefficients.SAMPLED,
                   factorization=Factorization.LI)
        b = _solve(word, 'numpy', fourier_coefficients='sampled', factorization='li')
        assert np.allclose(a, b, atol=0, rtol=0), (enum_spelling, a, b)

    with pytest.raises(ValueError, match='tetm'):
        _solve(0, 'numpy')
