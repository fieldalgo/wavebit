"""The core produces the *entire* public result surface, not just (R, T).

``Result`` carries reflected/transmitted amplitudes, their s/p split, and — for the conical and
2D paths — the pure-TE and pure-TM incidence variants. These check that the ``'amplitude'``
rung of the ``answers`` declaration delivers all of it and that the wiring routes it through
unchanged; the values themselves are pinned against the frozen legacy reference in
test_golden_legacy_parity.py.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.factorization import to_conv_mat_raster_sampled
from wavebit._core.solve_1d import solve_1d, solve_1d_conical
from wavebit._core.solve_2d import solve_2d

FIELDS = ('r_s', 'r_p', 't_s', 't_p',
          'R', 'R_s', 'R_p', 'T', 'T_s', 'T_p')


def _setup(**kw):
    solver = wavebit.rcwa(backend=0, **kw)
    legacy = solver.solve()
    epx, epy, epz, *_ = to_conv_mat_raster_sampled(
        solver.structure, solver.truncation_order[0], solver.truncation_order[1])
    kx, ky = solver.order_kxy()
    return solver, legacy, (epx, epy, epz), kx, ky


def _assert_all_fields(legacy_sub, got):
    for f in FIELDS:
        want = np.array(getattr(legacy_sub, f))
        assert np.array_equal(want, np.array(got[f])), (
            f, np.max(np.abs(want - np.array(got[f]))))


@pytest.mark.parametrize('tetm', ['te', 'tm'], ids=['te', 'tm'])
def test_solve_1d_full_result_matches_legacy(tetm):
    # Permittivities at the public boundary: top eps 4 is the index 2, and the raster's 9 is
    # the index 3. The core solves below still take the index, which `solver.n_top` derives.
    kw = dict(top=4.0, bot=1.0, theta=0.3, phi=None, tetm=tetm, truncation_order=[6, 0],
              period=[0.77], wavelength=0.777, thickness=[0.15],
              structure=np.array([[[9., 9., 9., 1., 1., 1.]]]))
    solver, legacy, conv, kx, _ = _setup(**kw)
    got = solve_1d(solver.tetm_index, kw['wavelength'], kx, solver.n_top, solver.n_bot, solver.theta,
                   solver.thickness, *conv, answers='amplitude')
    _assert_all_fields(legacy.given, got['given'])


CONICAL_2D = [
    ('conical', np.array([[[9., 9., 9., 1., 1., 1.]]]), [0.15], [6, 0], [0.77]),
    ('2d', np.random.default_rng(0).uniform(1, 4, (1, 5, 5)) ** 2, [0.2], [2, 2], [1.0, 1.0]),
    ('2d_2layer', np.random.default_rng(1).uniform(1, 4, (2, 4, 4)) ** 2, [0.15, 0.22],
     [2, 2], [1.0, 1.0]),
]


@pytest.mark.parametrize('case', CONICAL_2D, ids=[c[0] for c in CONICAL_2D])
def test_conical_and_2d_full_result_matches_legacy(case):
    name, structure, thickness, truncation_order, period = case
    kw = dict(top=2.25, bot=1.0, theta=0.2, phi=0.3, psi=0.7, truncation_order=truncation_order, period=period,
              wavelength=0.9, thickness=thickness, structure=structure)
    solver, legacy, conv, kx, ky = _setup(**kw)

    solve_fn = solve_1d_conical if name == 'conical' else solve_2d
    args = ([kw['wavelength'], kx, ky, solver.n_top, solver.n_bot, solver.theta, solver._incidence_pair(),
             solver.thickness] + list(conv))
    got = solve_fn(*args, phi=solver.phi, answers='amplitude')

    # all three incidence variants, every field
    _assert_all_fields(legacy.given, got['given'])
    _assert_all_fields(legacy.s_inc, got['s_inc'])
    _assert_all_fields(legacy.p_inc, got['p_inc'])


def test_amplitude_rung_is_consistent_with_efficiency_rung():
    """Climbing the ladder must not perturb the numbers the rung below returns."""
    kw = dict(top=2.25, bot=1.0, theta=0.2, phi=0.3, psi=0.7, truncation_order=[2, 2],
              period=[1.0, 1.0], wavelength=0.9, thickness=[0.2],
              structure=np.random.default_rng(2).uniform(1, 4, (1, 5, 5)) ** 2)
    solver, _, conv, kx, ky = _setup(**kw)
    args = [kw['wavelength'], kx, ky, solver.n_top, solver.n_bot, solver.theta,
            solver._incidence_pair(),
            solver.thickness] + list(conv)
    R, T = solve_2d(*args, phi=solver.phi)
    for rung in ('amplitude', 'field'):
        up = solve_2d(*args, phi=solver.phi, answers=rung)
        assert np.array_equal(np.array(R), np.array(up['given']['R'])), rung
        assert np.array_equal(np.array(T), np.array(up['given']['T'])), rung
