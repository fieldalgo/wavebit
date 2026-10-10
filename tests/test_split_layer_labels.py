"""A layer cut in two keeps one label: the ETM cap decides on the run's thickness, not the slice's.

ETM's labels are capped by thickness (:func:`~wavebit._core.conventions.etm_mode_q`): a mode keeps
the half-space's root until that root grows by ``eps**-0.25`` across the layer, and takes the
decaying root past it. The cap read each layer's own ``k0*d``. Two adjacent layers of the same
material are one layer cut in two, and a thin slice next to a thick one then kept a root its
neighbour had turned. The plane between them reflects nothing, so it had to turn one root into the
other with a reflection of exactly zero -- a pivot on that zero. At ``theta = 0.66+0.47j`` a 1 um
slice beside 8 um of the same uniform layer, under a grating, put ETM's amplitudes 3.0e-1 off the
uncut stack (TE), and 1.7e-8..1.9e-7 at worst on the TM, conical and 2D paths, while SMM's
cascade, which joins layers through its gap medium, stayed at 1e-14, and so does its field
state, which walks the gap amplitude and reads no labels.

:func:`~wavebit._core.conventions.run_k0_d` hands every slice of a run of identical layers --
equal to ``sqrt(eps)``, :func:`~wavebit._core.conventions.layer_runs` -- the run's ``k0*d``, and
the ETM steps across a run's planes exactly (:func:`~wavebit._core.solve_1d._run_ab`), so a cut
layer is the uncut layer. The controls below give each slice its own decision again, or take the
exact step away, and must break each case.
At a real ``k_par`` in a passive layer no cap fires, whatever thickness it is handed -- with one
exception, the spurious root a truncated metal operator can hold above the real axis, where a
split metal grating's ETM field was 1.5e-11 off the uncut one and now agrees. The last tests pin
both.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core import solve_1d
from wavebit._core.conventions import layer_runs, run_k0_d

pytestmark = pytest.mark.filterwarnings('ignore::wavebit.ComplexIncidenceWarning')

#: The class-B scene: a uniform eps = 12.75 layer under a grating of the same material, vacuum on
#: both sides, lit from below so the thin slice is the one the light meets first.
X = 12.75
UNIFORM = [[X] * 8]
GRATING = [[1, X, 1, X, X, X, X, X]]
THETA = 0.66 + 0.47j


def _no_runs(xp, slots, rules=None):
    """The control: no two layers join, so every slice decides and steps alone, as before."""
    return [False] * (len(slots) - 1)


@pytest.fixture
def per_layer(monkeypatch):
    def install():
        monkeypatch.setattr(solve_1d, 'layer_runs', _no_runs)
    return install


def _solver(layers, thickness, algorithm, path, theta=THETA, answers='amplitude'):
    kw = dict(tetm=path) if path in ('te', 'tm') else dict(phi=0.4, psi=0.3)
    return wavebit.rcwa(backend='numpy', top=1.0, bot=1.0, theta=theta, wavelength=1000.0,
                        period=[2000.0], truncation_order=[10],
                        structure=np.array(layers, dtype=complex), thickness=thickness,
                        layer_algorithm=algorithm, complex_dtype=0, incidence_side='bottom',
                        fourier_coefficients='integrated', factorization='li',
                        answers=answers, **kw)


def _amplitudes(layers, thickness, algorithm, path, theta=THETA):
    res = _solver(layers, thickness, algorithm, path, theta).solve(auto_detune=False)
    subs = ('incident',) if path in ('te', 'tm') else ('s_inc', 'p_inc')
    return np.concatenate([np.ravel(np.asarray(getattr(getattr(res, n), f)))
                           for n in subs for f in ('r_s', 'r_p', 't_s', 't_p')])


def _relative(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


#: (path, thick slice in nm, what the per-layer control left the split ETM at)
SPLITS = [('te', 8e3, 3.0e-1), ('te', 6e3, 1.2e-3), ('tm', 4.3e3, 1.9e-7),
          ('conical', 4.3e3, 1.7e-8)]


@pytest.mark.parametrize('path,thick,before', SPLITS,
                         ids=[f'{p}-1+{t / 1e3:g}um' for p, t, _ in SPLITS])
def test_a_split_layer_answers_as_the_uncut_one(per_layer, path, thick, before):
    """Split ETM against the uncut stack's SMM: within the uncut stack's own ETM level, 1e-12.

    The control (each slice on its own thickness) must leave it at least a decade above, at
    what it was before -- 3.0e-1 at 1 um + 8 um TE, 1.7e-8 conical.
    """
    merged = ([UNIFORM, GRATING], [1000.0 + thick, 4800.0])
    split = ([UNIFORM, UNIFORM, GRATING], [1000.0, thick, 4800.0])
    reference = _amplitudes(*merged, 'SMM', path)
    assert _relative(_amplitudes(*split, 'ETM', path), reference) < 1e-12
    assert _relative(_amplitudes(*split, 'SMM', path), reference) < 1e-12

    per_layer()
    broken = _relative(_amplitudes(*split, 'ETM', path), reference)
    assert broken > before / 10, broken


def _amplitudes_2d(layers, thickness, algorithm):
    """A 2D rod of eps 5 in vacuum, over a uniform eps 5 layer cut in two, lit from above."""
    s = wavebit.rcwa(backend='numpy', top=1.0, bot=1.0, theta=0.45 + 0.3j, phi=0.4, psi=0.3,
                     wavelength=1000.0, period=(1300.0, 1100.0), truncation_order=(3, 3),
                     structure=np.array(layers), thickness=thickness, layer_algorithm=algorithm,
                     complex_dtype=0, incidence_side='top')
    res = s.solve(auto_detune=False)
    return np.concatenate([np.ravel(np.asarray(getattr(getattr(res, n), f)))
                           for n in ('s_inc', 'p_inc') for f in ('r_s', 'r_p', 't_s', 't_p')])


def test_the_2d_path_splits_the_same(per_layer):
    """``solve_2d``'s loop reads the same runs: 4.8e-8 off before, 6e-13 now."""
    rod = np.full((6, 6), 1.0 + 0j)
    rod[1:4, 1:5] = 5.0
    uniform = np.full((6, 6), 5.0 + 0j)
    reference = _amplitudes_2d([rod, uniform], [600.0, 4500.0], 'SMM')
    split = ([rod, uniform, uniform], [600.0, 4000.0, 500.0])
    assert _relative(_amplitudes_2d(*split, 'ETM'), reference) < 5e-12

    per_layer()
    assert _relative(_amplitudes_2d(*split, 'ETM'), reference) > 1e-9


@pytest.mark.parametrize('path', ['te', 'tm', 'conical'])
def test_the_field_of_a_split_layer_is_the_uncut_field(per_layer, path):
    """The field inside a split stack is the uncut stack's field, on both algorithms.

    The ETM's field state walks each layer's ``+`` amplitude on the ETM's labels, so it needs the
    run: with each slice deciding alone its field went 6.3e-1 (TE), 9.7e-8 (TM) and 4.6e-9
    (conical) off. SMM's walks the gap amplitude between layers and reads no labels
    (:func:`~wavebit._core.scattering._field_state`), so it is the uncut field either way --
    which it was not while it was rebuilt on the ETM's labels (429f4ac): 1.0 off, conical.
    """
    thick = 8e3 if path != 'tm' else 4.3e3
    fractions = np.array([0.02, 0.08, 0.15, 0.4, 0.6, 0.8, 0.97])

    def field(layers, thickness, algorithm):
        s = _solver(layers, thickness, algorithm, path, answers='field')
        s.solve(auto_detune=False)
        kw = {} if path in ('te', 'tm') else dict(incidences=('s', 'p'))
        return np.asarray(s.reconstruct_field(sample_x=[300., 1100.], sample_y=[0.],
                                              sample_z=fractions * sum(thickness), **kw))

    merged = ([UNIFORM, GRATING], [1000.0 + thick, 4800.0])
    split = ([UNIFORM, UNIFORM, GRATING], [1000.0, thick, 4800.0])
    reference = field(*merged, 'SMM')
    level = _relative(field(*merged, 'ETM'), reference)       # the uncut stack's own agreement
    bound = max(20 * level, 1e-9)
    for algorithm in ('ETM', 'SMM'):
        assert _relative(field(*split, algorithm), reference) < bound

    per_layer()
    assert _relative(field(*split, 'ETM'), reference) > bound
    assert _relative(field(*split, 'SMM'), reference) < bound


# ------------------------------------------------------------------------- the rule itself

def _k0_ds(xp, k0, thickness, slots, rules=None):
    return run_k0_d(xp, k0, thickness, layer_runs(xp, slots, rules))


def test_without_a_run_every_layer_reads_its_own_k0_d():
    """No two layers alike: each reads ``k0 * thickness[i]``, the expression passed before."""
    rng = np.random.default_rng(1)
    slots = [(rng.standard_normal((3, 3)), None) for _ in range(4)]
    k0, thickness = 2 * np.pi / 1000.0, [10.0, 2e3, 3.3, 7e4]
    out = _k0_ds(np, k0, thickness, slots)
    assert all(o == k0 * d for o, d in zip(out, thickness))


def test_a_run_reads_its_total_and_stops_at_its_ends():
    """Runs are maximal: ``A A B A`` is three runs, and only the first two layers share a sum."""
    a, b = np.eye(2), 2 * np.eye(2)
    k0, thickness = 0.5, [1.0, 2.0, 4.0, 8.0]
    out = _k0_ds(np, k0, thickness, [(a,), (a.copy(),), (b,), (a,)])
    assert out == [k0 * 3.0, k0 * 3.0, k0 * 4.0, k0 * 8.0]
    # A run's total is summed once: every slice reads the same bits.
    three = _k0_ds(np, 1.0, [100.1, 200.2, 300.3], [(a,)] * 3)
    assert three[0] == three[1] == three[2]
    # -0.0 equals 0.0, a different shape or an absent slot does not equal a present one
    z = np.zeros((2, 2))
    assert _k0_ds(np, 1.0, [1.0, 1.0], [(z,), (-z,)]) == [2.0, 2.0]
    assert _k0_ds(np, 1.0, [1.0, 1.0], [(z,), (np.zeros((3, 3)),)]) == [1.0, 1.0]
    assert _k0_ds(np, 1.0, [1.0, 1.0], [(z, None), (z, z)]) == [1.0, 1.0]


def test_a_layer_written_twice_to_rounding_is_one_layer():
    """Slots that agree to ``sqrt(eps)`` of their largest entry join; farther apart they do not.

    The same material written two ways -- a tensor built as ``R D R^T`` beside ``D`` rotated by
    ``R`` -- differs by a few ulp, and the two slices must decide together: apart, first-order SMM
    went 7.7e4 off at the plane between them.
    """
    a = np.diag([2.4, 2.4, 3.2]).astype(complex)
    tol = np.sqrt(np.finfo(float).eps)
    assert layer_runs(np, [(a,), (a * (1 + 1e-15),)]) == [True]
    assert layer_runs(np, [(a,), (a * (1 + 0.5 * tol),)]) == [True]
    assert layer_runs(np, [(a,), (a * (1 + 4 * tol),)]) == [False]
    assert layer_runs(np, [(np.zeros((2, 2)),), (np.zeros((2, 2)),)]) == [True]


def test_layers_under_different_label_rules_are_no_run():
    """Equal slots under two different double-negative verdicts keep their own thicknesses."""
    from wavebit._core.conventions import DoubleNegativeLayer
    a = np.eye(2)
    out = _k0_ds(np, 1.0, [1.0, 2.0], [(a,), (a,)],
                 [DoubleNegativeLayer(True, True), DoubleNegativeLayer(False, None)])
    assert out == [1.0, 2.0]
    same = _k0_ds(np, 1.0, [1.0, 2.0], [(a,), (a,)],
                  [DoubleNegativeLayer(True, True), DoubleNegativeLayer(True, True)])
    assert same == [3.0, 3.0]


def test_a_traced_verdict_selects_the_same_sums():
    """Under ``jax.jit`` over the slots, and over the double-negative verdicts, the verdicts
    are traced and the sums are the eager ones."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from wavebit._core.conventions import DoubleNegativeLayer
    thickness = [1.0, 2.0, 4.0]

    def sums(a, b, c, holds):
        rules = [DoubleNegativeLayer(holds[i], None) for i in range(3)]
        return jnp.stack(_k0_ds(jnp, 0.5, thickness, [(a,), (b,), (c,)], rules))

    a, b = jnp.eye(2), 2 * jnp.eye(2)
    for args in ((a, a, b), (a, b, b), (a, a, a), (a, b, a)):
        for holds in ((False, False, False), (True, True, False), (True, False, True)):
            rules = [DoubleNegativeLayer(h, None) for h in holds]
            eager = [float(v) for v in _k0_ds(np, 0.5, thickness,
                                              [(np.asarray(x),) for x in args], rules)]
            traced = jax.jit(sums)(*args, jnp.asarray(holds))
            assert np.array_equal(np.asarray(traced), eager), (args, holds)


def test_a_split_stack_traced_under_jit_matches_numpy():
    """The class-B solve with the raster traced, every verdict a tracer, takes the runs too.

    Laurent's rule, which reads nothing off the raster's values, so the whole build traces. Held
    to the uncut stack's SMM, which a per-slice decision misses by 0.17 on this rule.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    def r(backend, structure, thickness, algorithm='ETM'):
        s = wavebit.rcwa(backend=backend, top=1.0, bot=1.0, theta=THETA, wavelength=1000.0,
                         period=[2000.0], truncation_order=[10], structure=structure,
                         thickness=thickness, layer_algorithm=algorithm, complex_dtype=0,
                         incidence_side='bottom', factorization='laurent', tetm='te')
        return s.solve(auto_detune=False).incident.r_s

    merged = np.asarray(r('numpy', np.array([UNIFORM, GRATING], dtype=complex), [9000.0, 4800.0],
                          'SMM'))
    split = np.array([UNIFORM, UNIFORM, GRATING], dtype=complex)
    traced = jax.jit(lambda st: r('jax', st, [1000.0, 8e3, 4800.0]))(jnp.asarray(split))
    assert _relative(np.asarray(traced), merged) < 1e-12


def test_a_metal_grating_split_at_a_real_angle_keeps_one_label(per_layer):
    """The real-angle exception: a truncated metal operator's spurious root, which grows.

    A 2D ``eps = -27.25+0.22j`` grating at normal incidence holds a root with ``Im(q**2) > 0``
    that grows by 7.0 nepers across 295 nm, under the cap. Cut into two 295 nm slices, each slice
    kept it growing and the plane between them pivoted on zero: the split ETM's field came out
    1.5e-11 off the uncut one -- a rounding residue, 5.6e-12 under some perturbations, so the
    bounds sit at 1e-13 between it and the good case. As one run of 590 nm it is capped, as the
    uncut layer is, and the two algorithms match the uncut layer's field to 1.8e-15 (ETM) and
    5.6e-15 (SMM). (The 1.9e-3 between the ETM and SMM fields
    first recorded here was SMM's, while its field state was rebuilt on the ETM's labels; it now
    reads none, and is the uncut field whatever the slices decide.)
    """
    metal = np.full((8, 8), 2.25 + 0j)
    metal[3:7, 2:5] = -27.25 + 0.22j
    rod = np.full((8, 8), 2.25 + 0j)
    rod[3:7, 3:6] = 12.0

    def field(layers, thickness, algorithm):
        s = wavebit.rcwa(backend='numpy', top=1.0, bot=1.0, theta=0.0, phi=0.6, psi=0.3,
                         wavelength=1000.0, period=(575.0, 477.0), truncation_order=(3, 2),
                         structure=np.array(layers), thickness=thickness,
                         layer_algorithm=algorithm, incidence_side='bottom', factorization='li',
                         answers='field')
        s.solve(auto_detune=False)
        return np.asarray(s.reconstruct_field(sample_x=[100., 300.], sample_y=[100., 250.],
                                              sample_z=[50., 200., 400., 600., 650.],
                                              incidences=('s', 'p')))

    uncut = field([metal, rod], [590.0, 116.0], 'ETM')
    split = ([metal, metal, rod], [295.0, 295.0, 116.0])
    for algorithm in ('ETM', 'SMM'):
        assert _relative(field(*split, algorithm), uncut) < 1e-13

    per_layer()
    assert _relative(field(*split, 'ETM'), uncut) > 1e-13
    assert _relative(field(*split, 'SMM'), uncut) < 1e-13


@pytest.mark.parametrize('path', ['te', 'tm', 'conical'])
def test_a_real_angle_stack_without_runs_is_untouched_to_the_bit(per_layer, path):
    """No two adjacent layers alike: the runs change nothing, at a real angle or a complex one."""
    stack = ([UNIFORM, GRATING, UNIFORM], [1000.0, 4800.0, 3000.0])
    runs = {t: _amplitudes(*stack, 'ETM', path, theta=t) for t in (0.66, THETA)}
    per_layer()
    for t in (0.66, THETA):
        assert np.array_equal(runs[t], _amplitudes(*stack, 'ETM', path, theta=t))


@pytest.mark.parametrize('path', ['te', 'tm', 'conical'])
def test_a_real_angle_split_layer_steps_as_the_uncut_one(per_layer, path):
    """At a real angle a split layer is the uncut one to rounding, stepped either way.

    Inside a run the recursion now takes ``a = I`` and ``b = Y`` exactly instead of reaching
    them through ``W^-1 W``, so the bits move; the answer does not (8e-15 to 9e-14 from the
    uncut stack on both forms).
    """
    uncut = _amplitudes([UNIFORM, GRATING], [9000.0, 4800.0], 'ETM', path, theta=0.66)
    split = ([UNIFORM, UNIFORM, GRATING], [1000.0, 8e3, 4800.0])
    assert _relative(_amplitudes(*split, 'ETM', path, theta=0.66), uncut) < 2e-13
    per_layer()
    assert _relative(_amplitudes(*split, 'ETM', path, theta=0.66), uncut) < 2e-13


def test_a_run_on_its_own_half_space_steps_as_the_uncut_layer(monkeypatch):
    """A run of the far half-space's own material, its thin slice touching the half-space.

    The run's capped label pivots on 1.7e-18 against the half-space, which only the whole run's
    ``exp(-k0 q d)`` balances: the thin slice's top carries a ``Y`` of 4e17. Through ``W^-1 W``
    the next slice's ``a`` came out with an exact zero on its diagonal and raised, where the
    uncut layer is right to 2.4e-13. The exact step (``a = I``, ``b = Y``) is the control's
    absence: with it taken away the split must break.
    """
    eps = 5.792813684844445

    def amplitudes(thickness):
        s = wavebit.rcwa(backend='numpy', top=1.0, bot=eps, theta=0.43 + 0.067j, wavelength=1000.0,
                         period=[2000.0], truncation_order=[10],
                         structure=np.full((len(thickness), 1, 8), eps + 0j),
                         thickness=thickness, layer_algorithm='ETM', complex_dtype=0, tetm='te')
        res = s.solve(auto_detune=False).incident
        return np.concatenate([np.ravel(np.asarray(res.r_s)), np.ravel(np.asarray(res.t_s))])

    uncut = amplitudes([60500.0])
    assert _relative(amplitudes([500.0, 60000.0]), uncut) < 1e-12

    monkeypatch.setattr(solve_1d, '_run_join', lambda links, index, below: False)
    try:
        broken = _relative(amplitudes([500.0, 60000.0]), uncut)
    except np.linalg.LinAlgError:
        return
    assert broken > 1e-6, broken

