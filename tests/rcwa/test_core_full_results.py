"""The core produces the *entire* public result surface, not just (R, T).

``Result`` carries reflected/transmitted amplitudes, their s/p split, and — for the conical and
2D paths — the pure-TE and pure-TM incidence variants. These check that the ``'amplitude'``
rung of the ``answers`` declaration delivers all of it and that the wiring routes it through
unchanged but for the centre sign; the values themselves are pinned against the frozen legacy
reference in test_golden_legacy_parity.py.

The one conversion the wiring makes: the kernels' amplitudes are referenced at the cell corner,
their Fourier origin, and the public ones at the centre (2026-09-27), which on a single incident
wave negates every order with ``m + n`` odd -- an exact sign bit, so the comparison stays by
bytes.
"""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

from wavebit.rcwa._engine.factorization import to_conv_mat_raster_sampled
from wavebit.rcwa._engine.solve_1d import solve_1d, solve_1d_conical
from wavebit.rcwa._engine.solve_2d import solve_2d

AMPLITUDES = ('r_s', 'r_p', 't_s', 't_p')
FIELDS = AMPLITUDES + ('R', 'R_s', 'R_p', 'T', 'T_s', 'T_p')


def _setup(**kw):
    solver = _legacy_wavebit.rcwa(backend='numpy', **kw)
    legacy = solver.solve()
    epx, epy, epz, *_ = to_conv_mat_raster_sampled(
        solver.structure, solver.truncation_order[0], solver.truncation_order[1])
    kx, ky = solver.order_kxy()
    return solver, legacy, (epx, epy, epz), kx, ky


def _assert_all_fields(result, legacy_sub, got):
    """Every field by its bytes: efficiencies as the kernel returned them, amplitudes negated
    on the orders with ``m + n`` odd, read off the result's own labels."""
    m, n = np.asarray(result.orders_u), np.asarray(result.orders_v)
    odd = (n[:, None] + m[None, :]) % 2 == 1
    for f in FIELDS:
        public = np.array(getattr(legacy_sub, f))
        kernel = np.array(got[f])
        want = np.where(np.reshape(odd, kernel.shape), -kernel, kernel) if f in AMPLITUDES else kernel
        assert public.dtype == want.dtype and public.tobytes() == want.tobytes(), (
            f, np.max(np.abs(want - public)))


@pytest.mark.parametrize('tetm', ['te', 'tm'], ids=['te', 'tm'])
def test_solve_1d_full_result_matches_legacy(tetm):
    # Permittivities at the public boundary: top eps 4 is the index 2, and the raster's 9 is
    # the index 3. The core solves below still take the index, which `solver.n_top` derives.
    kw = dict(top=4.0, bot=1.0, theta=0.3, phi=None, tetm=tetm, truncation_order=[6, 0],
              period=[0.77], wavelength=0.777, thickness=[0.15],
              structure=np.array([[[9., 9., 9., 1., 1., 1.]]]))
    solver, legacy, conv, kx, _ = _setup(**kw)
    got = solve_1d(solver.tetm_index, kw['wavelength'], kx, solver.n_bot, solver.n_top, solver.theta,
                   solver.thickness, *conv, answers='amplitude',
                   incidence_side=solver._kernel_incidence_side,
                   eps_top=kw['bot'], eps_bot=kw['top'])
    _assert_all_fields(legacy, legacy.incident, got['incident'])


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
    args = ([kw['wavelength'], kx, ky, solver.n_bot, solver.n_top, solver.theta, solver._incidence_pair(),
             solver.thickness] + list(conv))
    got = solve_fn(*args, phi=solver.phi, answers='amplitude',
                   incidence_side=solver._kernel_incidence_side,
                   eps_top=kw['bot'], eps_bot=kw['top'])

    # all three incidence variants, every field
    _assert_all_fields(legacy, legacy.incident, got['incident'])
    _assert_all_fields(legacy, legacy.s_inc, got['s_inc'])
    _assert_all_fields(legacy, legacy.p_inc, got['p_inc'])


def test_amplitude_rung_is_consistent_with_efficiency_rung():
    """Climbing the ladder must not perturb the numbers the rung below returns."""
    kw = dict(top=2.25, bot=1.0, theta=0.2, phi=0.3, psi=0.7, truncation_order=[2, 2],
              period=[1.0, 1.0], wavelength=0.9, thickness=[0.2],
              structure=np.random.default_rng(2).uniform(1, 4, (1, 5, 5)) ** 2)
    solver, _, conv, kx, ky = _setup(**kw)
    args = [kw['wavelength'], kx, ky, solver.n_bot, solver.n_top, solver.theta,
            solver._incidence_pair(),
            solver.thickness] + list(conv)
    core_kw = dict(phi=solver.phi, incidence_side=solver._kernel_incidence_side,
                   eps_top=kw['bot'], eps_bot=kw['top'])
    R, T, _ = solve_2d(*args, **core_kw)
    for rung in ('amplitude', 'field'):
        up = solve_2d(*args, answers=rung, **core_kw)
        assert np.array_equal(np.array(R), np.array(up['incident']['R'])), rung
        assert np.array_equal(np.array(T), np.array(up['incident']['T'])), rung
