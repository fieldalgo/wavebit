"""The core reproduces the *entire* legacy result surface, not just (de_ri, de_ti).

Wiring ``call_mee`` into the core needs more than efficiencies: the public ``Result`` carries
reflected/transmitted amplitudes, their s/p split, and — for the conical and 2D paths — the
pure-TE and pure-TM incidence variants. Every one of those fields must be byte-identical
before the legacy solvers can be retired.
"""
import numpy as np
import pytest

import fieldwave
from fieldwave.on_numpy.emsolver.convolution_matrix import to_conv_mat_raster_discrete
from fieldwave._core.solve_1d import solve_1d, solve_1d_conical
from fieldwave._core.solve_2d import solve_2d

FIELDS = ('R_s', 'R_p', 'T_s', 'T_p',
          'de_ri', 'de_ri_s', 'de_ri_p', 'de_ti', 'de_ti_s', 'de_ti_p')


def _setup(**kw):
    mee = fieldwave.call_mee(backend=0, **kw)
    legacy = mee.conv_solve()
    epx, epy, epz = to_conv_mat_raster_discrete(
        mee.ucell, mee.fto[0], mee.fto[1], type_complex=mee.type_complex,
        enhanced_dfs=mee.enhanced_dfs, use_pinv=mee.use_pinv)
    kx, ky = mee.get_kx_ky_vector(kw['wavelength'])
    return mee, legacy, (epx, epy, epz), kx, ky


def _assert_all_fields(legacy_sub, got):
    for f in FIELDS:
        want = np.array(getattr(legacy_sub, f))
        assert np.array_equal(want, np.array(got[f])), (
            f, np.max(np.abs(want - np.array(got[f]))))


@pytest.mark.parametrize('pol', [0, 1], ids=['te', 'tm'])
def test_solve_1d_full_result_matches_legacy(pol):
    kw = dict(n_top=2.0, n_bot=1.0, theta=0.3, phi=None, pol=pol, fto=[6, 0],
              period=[0.77], wavelength=0.777, thickness=[0.15],
              ucell=np.array([[[3., 3., 3., 1., 1., 1.]]]))
    mee, legacy, conv, kx, _ = _setup(**kw)
    got = solve_1d(pol, kw['wavelength'], kx, kw['n_top'], kw['n_bot'], mee.theta,
                   mee.thickness, *conv, full=True)
    _assert_all_fields(legacy.res, got['res'])


CONICAL_2D = [
    ('conical', np.array([[[3., 3., 3., 1., 1., 1.]]]), [0.15], [6, 0], [0.77]),
    ('2d', np.random.default_rng(0).uniform(1, 4, (1, 5, 5)), [0.2], [2, 2], [1.0, 1.0]),
    ('2d_2layer', np.random.default_rng(1).uniform(1, 4, (2, 4, 4)), [0.15, 0.22], [2, 2],
     [1.0, 1.0]),
]


@pytest.mark.parametrize('case', CONICAL_2D, ids=[c[0] for c in CONICAL_2D])
def test_conical_and_2d_full_result_matches_legacy(case):
    name, ucell, thickness, fto, period = case
    kw = dict(n_top=1.5, n_bot=1.0, theta=0.2, phi=0.3, psi=0.7, fto=fto, period=period,
              wavelength=0.9, thickness=thickness, ucell=ucell)
    mee, legacy, conv, kx, ky = _setup(**kw)

    solver = solve_1d_conical if name == 'conical' else solve_2d
    args = ([kw['wavelength'], kx, ky, kw['n_top'], kw['n_bot'], mee.theta, mee.psi,
             mee.thickness] + list(conv))
    got = solver(*args, full=True)

    # all three incidence variants, every field
    _assert_all_fields(legacy.res, got['res'])
    _assert_all_fields(legacy.res_te_inc, got['res_te_inc'])
    _assert_all_fields(legacy.res_tm_inc, got['res_tm_inc'])


def test_full_flag_is_consistent_with_tuple_form():
    """``full=True`` must not perturb the numbers the default form returns."""
    kw = dict(n_top=1.5, n_bot=1.0, theta=0.2, phi=0.3, psi=0.7, fto=[2, 2],
              period=[1.0, 1.0], wavelength=0.9, thickness=[0.2],
              ucell=np.random.default_rng(2).uniform(1, 4, (1, 5, 5)))
    mee, _, conv, kx, ky = _setup(**kw)
    args = [kw['wavelength'], kx, ky, kw['n_top'], kw['n_bot'], mee.theta, mee.psi,
            mee.thickness] + list(conv)
    de_ri, de_ti = solve_2d(*args)
    full = solve_2d(*args, full=True)
    assert np.array_equal(np.array(de_ri), np.array(full['res']['de_ri']))
    assert np.array_equal(np.array(de_ti), np.array(full['res']['de_ti']))
