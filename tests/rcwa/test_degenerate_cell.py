"""A cell that is not finite or spans no area is refused at check, whatever spelled it.

The matrix spelling already refused a NaN or a parallel pair, but a basis of zeros passed its
relative area test (``0 < 1e-12 * 0``), and a ``Lattice`` built directly -- or from a zero or
NaN ``period`` -- skipped the test altogether: `check` came back clean and the solve died with
a ``ZeroDivisionError`` or the runtime guard's ``prepared.finite_inputs``.
"""
import numpy as np
import pytest

from wavebit import Lattice, Material, Units, rcwa
from wavebit.errors import ValidationError

P = 400.
NUMERICS = rcwa.Numerics(orders=(1, 1))
NOT_FINITE = 'entries must be finite'
NO_AREA = 'zero cell area'


def _problem(lattice):
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(lattice=lattice, layers=(rcwa.Layer(thickness=100., background=Material(epsilon=2.)),)),
        excitation=rcwa.Excitation(wavelength=620., sources=(rcwa.PlaneWave(theta=.2),)))


@pytest.mark.parametrize('lattice,expected', [
    (np.zeros((2, 2)), NO_AREA),
    (np.array([[P, 2*P], [0., 0.]]), NO_AREA),
    (np.array([[P, 0.], [0., np.nan]]), NOT_FINITE),
    (Lattice(np.array([P, 0.]), np.array([0., 0.]), True), NO_AREA),
    (Lattice(np.array([P, 0.]), np.array([2*P, 0.]), False), NO_AREA),
    (Lattice(np.array([P, 0.]), np.array([0., np.nan]), True), NOT_FINITE),
], ids=['zeros', 'parallel', 'nan', 'lattice-zero', 'lattice-parallel', 'lattice-nan'])
def test_check_refuses_a_cell_with_no_area_or_no_number(lattice, expected):
    issues = rcwa.check(_problem(lattice), numerics=NUMERICS).issues
    assert len(issues) == 1 and issues[0].code == 'rcwa.invalid' and expected in issues[0].message
    with pytest.raises(ValidationError, match=expected):
        rcwa.run(_problem(lattice), numerics=NUMERICS)


@pytest.mark.parametrize('lengths,expected', [((0., P), 'no area'), ((np.nan,), 'must be finite')])
def test_a_rectangular_cell_needs_finite_nonzero_sides(lengths, expected):
    with pytest.raises(ValueError, match=expected):
        Lattice.rectangular(*lengths)


def test_cells_with_area_still_solve():
    """A positive, a negative-sided and a sheared cell keep solving, lossless and balanced."""
    for lattice in (Lattice.rectangular(P), Lattice.rectangular(-P, P),
                    Lattice.from_columns(np.array([[P, .3*P], [0., P]]))):
        result = rcwa.run(_problem(lattice), numerics=NUMERICS)
        assert not rcwa.check(_problem(lattice), numerics=NUMERICS).issues
        np.testing.assert_allclose(float(result.R.sum() + result.T.sum()), 1., rtol=0, atol=1e-12)
