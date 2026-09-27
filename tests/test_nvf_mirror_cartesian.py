"""Compare parity-first interval assembly with the unreduced Cartesian path."""
import numpy as np
import pytest
from scipy.linalg import block_diag

from studies.nvf_gap_convergence import profile_li, power_record
from studies.nvf_mirror_cartesian import cartesian_sector, cartesian_powers
from test_nvf_mirror_convergence import dense_parity

pytestmark = pytest.mark.research


@pytest.mark.parametrize('order,gap', [((0, 3), 17.), ((3, 0), 17.), ((2, 4), 17.), (3, 0.), (3, 400.)])
def test_cartesian_mirror_matrices_and_powers_match_unreduced_intervals(order, gap):
    xx, yy, zi, xy, yx = profile_li(order, gap, 350., [-31., 73.], [4., 9., 6.25])
    full = np.block([[xx, -xy], [-yx, yy]])
    for pol, parity in (('p', 1), ('s', -1)):
        actual = cartesian_sector(order, gap, 350., [-31., 73.], [4., 9., 6.25], polarization=pol)
        a, b = dense_parity(order, parity), dense_parity(order, -parity)
        u = block_diag(a, b)
        np.testing.assert_allclose(actual['effective'], u.T@full@u, atol=5e-12, rtol=0)
        np.testing.assert_allclose(actual['z_inverse'], a.T@zi@a, atol=3e-13, rtol=0)
    actual = cartesian_powers(order, gap, 350., [-31., 73.], [4., 9., 6.25])
    expected = power_record(order, (xx, yy, zi, xy, yx))
    for key in ('p_T', 's_T', 'p_R', 's_R'):
        assert actual[key] == pytest.approx(expected[key], abs=3e-12, rel=0)


def test_redundant_material_partition_and_invalid_profiles_keep_existing_semantics():
    base = cartesian_powers(2, 17., 350., [-31., 73.], [4., 9., 6.25])
    split = cartesian_powers(2, 17., 350., [-100., -31., 0., 73., 100.], [4., 4., 9., 9., 6.25, 6.25])
    for key in ('p_T', 's_T', 'p_R', 's_R'):
        assert split[key] == pytest.approx(base[key], abs=3e-12, rel=0)
    for gap, width, seams, values in ((-1., 350., [], [4.]), (17., 700., [], [4.]),
                                     (17., 350., [180.], [4., 9.]), (17., 350., [], [0.])):
        with pytest.raises(ValueError, match='profile'):
            cartesian_powers(2, gap, width, seams, values)
