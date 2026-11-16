"""``exceptional_points='off'``: the exceptional-point machinery skipped on every path.

The default, ``'rebuild'``, finds modes at and near an exceptional point, rebuilds them in a
regular basis and screens what it cannot rebuild. ``'off'`` is an advanced choice for speed:
no gate, no rebuild, no screen -- on the second-order and first-order formulations of a 2D
grating, on the 1D conical x-cutoff and in the coherent-source cascade, eagerly and under
``jax.jit``. These tests count calls into the gates (a value test cannot tell a gate that ran
and found nothing from one that never ran), compare the two settings away from an exceptional
point, and show the documented price at one.
"""
import warnings

import numpy as np
import pytest

import wavebit
from wavebit.errors import ValidationError
from wavebit.rcwa._engine import solve_1d, solve_2d, solve_2d_first_order
from wavebit.rcwa._engine.sources import PlaneWave as _PlaneWave
from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa

from tests.rcwa.test_conical_cutoff import PATTERNED, _solver as _conical_solver
from tests.rcwa.test_exceptional_point_clusters import _c4v_solver

#: A square pillar at normal incidence (``_c4v_solver``) where the two E-irrep blocks each hold
#: a Jordan chain of length three: six modes at one eigenvalue, found by Newton on the
#: centred power sums of each block's three nearest eigenvalues.
C4V_EP = (745.9149148229458, 4.9546179826310786)

#: The gates every path calls on every layer while the machinery is on (each eagerly returns
#: at once on a layer without an exceptional point, so a count is the only way to see them).
GATES = ((solve_2d, '_defective_pairs'), (solve_2d, '_defective_clusters'),
         (solve_2d, '_unresolved_defective_cluster'), (solve_2d_first_order, '_pair_gate'),
         (solve_1d, '_cutoff_cluster'))


@pytest.fixture
def gate_calls(monkeypatch):
    """The number of calls into the exceptional-point gates since the last reset."""
    calls = []
    for module, name in GATES:
        original = getattr(module, name)

        def counted(*args, _original=original, **kwargs):
            calls.append(1)
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, name, counted)
    return calls


def _numbers(result):
    def host(a):
        return a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)
    return np.concatenate([host(result.r_xyz).ravel(), host(result.t_xyz).ravel()])


def _solve(solver, mode):
    solver.exceptional_points = mode
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = solver.solve(auto_detune=False)
    loud = [w for w in caught if issubclass(w.category, wavebit.rcwa.diagnostics.ExceptionalPointWarning)]
    return _numbers(result), loud


def _oblique(algo, first_order=False):
    solver = _c4v_solver(700.0, 4.5, algo=algo, first_order=first_order)
    solver.theta, solver.phi = 0.2, 0.3
    return solver


PATHS = {
    'second order': lambda algo: _oblique(algo),
    'first order': lambda algo: _oblique(algo, first_order=True),
    '1D conical': lambda algo: _conical_solver(PATTERNED['PA616'], algo=algo,
                                               wavelength=600.0),
}


# ---------------------------------------------------------------- the two words, every door

def test_numerics_takes_the_two_words_and_refuses_the_rest():
    assert wavebit.rcwa.Numerics().exceptional_points == 'rebuild'
    assert 'exceptional_points' not in wavebit.rcwa.Numerics()._engine_kwargs()
    assert wavebit.rcwa.Numerics(exceptional_points='off')._engine_kwargs()['exceptional_points'] == 'off'
    for bad in ('Off', 'on', 'none', None, True, False):
        with pytest.raises(ValidationError) as info:
            wavebit.rcwa.Numerics(exceptional_points=bad)
        assert info.value.code == 'numerics.choice'


def test_the_solver_refuses_a_misspelt_value_at_both_doors():
    with pytest.raises(ValueError, match='exceptional_points'):
        _c4v_solver(700.0, 4.5).exceptional_points = 'of'
    st = np.ones((1, 4, 4))
    for bad in ('of', None, 0):
        with pytest.raises(ValueError, match='exceptional_points'):
            _legacy_wavebit_rcwa(wavelength=700., period=[1000., 1000.], truncation_order=[1, 1],
                                 structure=st, thickness=[100.], exceptional_points=bad)
    with pytest.raises(Exception, match='exceptional_point'):
        _legacy_wavebit_rcwa(wavelength=700., period=[1000., 1000.], truncation_order=[1, 1],
                             structure=st, thickness=[100.], exceptional_point='off')


def test_a_job_that_leaves_it_alone_keeps_its_key():
    """Joined ``rcwa.Numerics`` after the job schema was fixed: written only off its default."""
    from wavebit import Job
    layer = wavebit.rcwa.Layer(thickness=.2, background=wavebit.Material(2.25))
    problem = wavebit.rcwa.Problem(
        units=wavebit.Units(length='um'),
        stack=wavebit.rcwa.Stack(period=(.8, .9), layers=(layer,)),
        excitation=wavebit.rcwa.Excitation(wavelength=np.array([.6]),
                                           sources=(wavebit.rcwa.PlaneWave(),)))
    job = Job(solver='rcwa', problem=problem, numerics=wavebit.rcwa.Numerics())
    assert 'exceptional_points' not in job.to_json()
    off = Job(solver='rcwa', problem=problem, numerics=wavebit.rcwa.Numerics(exceptional_points='off'))
    assert off.key != job.key
    assert Job.from_json(off.to_json()).numerics.exceptional_points == 'off'


def test_the_public_run_reports_it_and_reaches_the_kernels(gate_calls):
    layer = wavebit.rcwa.Layer(thickness=.2, background=wavebit.Material(2.25),
                               regions=(wavebit.rcwa.Region(geometry=wavebit.rcwa.Rectangle(size=(.3, .4)),
                                                            material=wavebit.Material(6.25)),))
    problem = wavebit.rcwa.Problem(
        units=wavebit.Units(length='um'),
        stack=wavebit.rcwa.Stack(period=(.8, .9), layers=(layer,)),
        excitation=wavebit.rcwa.Excitation(wavelength=np.array([.6]),
                                           sources=(wavebit.rcwa.PlaneWave(theta=.2, phi=.3),)))
    runs = {}
    for mode in ('rebuild', 'off'):
        numerics = wavebit.rcwa.Numerics(orders=(2, 2), exceptional_points=mode)
        assert wavebit.rcwa.check(problem, numerics=numerics).resolved['exceptional_points'] == mode
        del gate_calls[:]
        result = wavebit.rcwa.run(problem, numerics=numerics)
        runs[mode] = (len(gate_calls), np.asarray(result.reflection.power))
    assert runs['rebuild'][0] > 0 and runs['off'][0] == 0, runs
    np.testing.assert_array_equal(runs['rebuild'][1], runs['off'][1])


# ---------------------------------------------------------------- every path, eagerly

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('path', list(PATHS))
def test_off_skips_every_gate_and_keeps_the_bits_away_from_a_point(path, algo, gate_calls):
    """Away from an exceptional point (oblique, no degenerate pair) the answer is the same to the
    bit; with the machinery on the gates were called, with it off never."""
    del gate_calls[:]
    on, loud_on = _solve(PATHS[path](algo), 'rebuild')
    assert gate_calls, f'{path} {algo}: no gate ran with the machinery on'
    del gate_calls[:]
    off, loud_off = _solve(PATHS[path](algo), 'off')
    assert not gate_calls, f'{path} {algo}: a gate ran with the machinery off'
    assert not loud_on and not loud_off
    assert np.array_equal(on, off), f'{path} {algo}: {np.max(np.abs(on - off)):.1e}'


def test_the_coherent_cascade_takes_it_too(gate_calls):
    options = dict(wavelength=1.3, period=(2., 2.2), top=1., bot=2.25,
                   structure=np.array([[[4., 4., 1., 1.]]]), thickness=[.12],
                   truncation_order=(2, 1), answers='field')
    beams = {'top': _PlaneWave(kx=.2, ky=.3, polarization=(.6, .8j)),
             'bottom': _PlaneWave(kx=.2, ky=.3, polarization=(.8, .6), amplitude=.5j)}
    out = {}
    for mode in ('rebuild', 'off'):
        del gate_calls[:]
        result = _legacy_wavebit_rcwa(**options, excitation=beams, exceptional_points=mode).solve()
        out[mode] = (len(gate_calls), np.asarray(result.absorption))
    assert out['rebuild'][0] > 0 and out['off'][0] == 0, out
    np.testing.assert_array_equal(out['rebuild'][1], out['off'][1])


@pytest.mark.parametrize('first_order', [False, True], ids=['second order', 'first order'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_at_a_point_off_is_quiet_and_moves_the_answer(algo, first_order):
    """The documented price: at the C4v point (two coincident Jordan chains of length three) the
    default answer is exact and quiet (``test_exceptional_point_clusters``); off, the eigenvector
    basis is kept and the answer moves by far more than rounding -- with no warning."""
    on, loud_on = _solve(_c4v_solver(*C4V_EP, algo=algo, first_order=first_order), 'rebuild')
    off, loud_off = _solve(_c4v_solver(*C4V_EP, algo=algo, first_order=first_order), 'off')
    assert not loud_on and not loud_off
    assert np.max(np.abs(on - off)) > 1e-10


# ---------------------------------------------------------------- under jax.jit

def test_jit_keeps_the_setting_and_skips_the_gates(gate_calls):
    jax = pytest.importorskip('jax')
    from wavebit._common.backend import _enable_x64
    _enable_x64()
    import jax.numpy as jnp
    solver = _c4v_solver(*C4V_EP, backend='jax')
    solver.exceptional_points = 'off'
    leaves, tree = jax.tree_util.tree_flatten(solver)
    assert jax.tree_util.tree_unflatten(tree, leaves).exceptional_points == 'off'

    def amplitudes(wavelength):
        result = solver.solve(wavelength=wavelength, auto_detune=False)
        return jnp.concatenate([result.r_xyz.ravel(), result.t_xyz.ravel()])
    del gate_calls[:]
    traced = np.asarray(jax.jit(amplitudes)(jnp.asarray(C4V_EP[0])))
    assert not gate_calls
    # The default answers NaN here under a trace (two chains overflow one lane); off does not.
    assert np.all(np.isfinite(traced))
    eager, _ = _solve(_c4v_solver(*C4V_EP), 'off')
    assert np.max(np.abs(traced - eager)) < 1e-6
