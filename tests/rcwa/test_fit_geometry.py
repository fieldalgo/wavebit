"""A JIT optimizer must solve the current design, including after a plan changes.

The shape trajectory is checked against fresh NumPy values and central differences at
every iterate. It crosses an overlap transition, so retaining the initial plan is observably
wrong. Raster tests independently rebuild the geometry for every reference gradient. Runtime
callbacks observe the actual compiled values/gradients, not just the final parameter.
"""

from wavebit.rcwa._engine.geometry import StaleGeometryError as _legacy_wavebit_StaleGeometryError

from wavebit.materials._tabulated import TabulatedMaterial as _legacy_wavebit_TabulatedMaterial

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
from functools import partial

import numpy as np
import pytest

# CI may run each test of this file in a different shard (tools/run_verification.py --shard).
pytestmark = pytest.mark.shard_unit(per='test')


jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
optax = pytest.importorskip('optax')

SHAPE_KW = dict(period=(300., 300.), thickness=(205.,), truncation_order=(1, 1),
                top=1., bot=1., theta=.15, phi=.35, psi=.7, wavelength=900.,
                shape_nvf_grid_size=16, answers='field')


def _design(width):
    return dict(structure=[dict(background=12.25, shapes=[
        dict(type='rectangle', cx=-90., cy=0., lx=60., ly=100., angle=0., material=4.),
        dict(type='rectangle', cx=0., cy=0., lx=100.*width, ly=100., angle=0., material=9.),
        dict(type='rectangle', cx=90., cy=0., lx=60., ly=100., angle=0., material=6.25)])])


def _shape_value(width):
    return float(_legacy_wavebit_rcwa(backend='numpy', **SHAPE_KW, **_design(width)).solve().T.sum().real)


def _shape_gradient(width):
    # Past the overlap the painted layer's transmission at orders (1, 1) is steep and curved in
    # the width (dT/dw = 3.07 at w = 1.2266 since the painting route's normal field averages its
    # switches, 2026-10-15): at h = 1e-5 the central difference carried a truncation error of
    # 4.5e-7 there, and 5e-9 on the closed-form side -- which the replicated trajectories below
    # multiply by that slope. At 1e-6 both are below 1e-9, and what is left is round-off:
    # 5e-10 on macOS (Accelerate), 1e-9 on Linux (OpenBLAS).
    h = 1e-6
    return (_shape_value(width+h) - _shape_value(width-h)) / (2*h)


#: The gradient clip in front of every SGD below. The first step from width 1.18 (gradient
#: -0.058, under the clip) crosses the overlap at 1.2 as it always did; the second starts on
#: the painted layer, whose transmission at orders (1, 1) and normal grid 16 moves 3 per unit
#: width once the normal field averages its switches (0.37 before 2026-10-15), and a plain
#: step of .8 is no design at all (width -1.2: a negative length, which freeze_geometry
#: refuses). Clipped, every iterate is a legal design and the first still crosses, and the
#: optimizer's replay on the fresh-NumPy gradients below applies the same clip.
_CLIP = .1


def _sgd(learning_rate, **kwargs):
    return optax.chain(optax.clip(_CLIP), optax.sgd(learning_rate, **kwargs))


def _observed_optimizer(optimizer, observations):
    def update(grads, state, params=None):
        def remember(p, g):
            observations.append((jax.tree.map(lambda v: np.asarray(v).copy(), p),
                                 jax.tree.map(lambda v: np.asarray(v).copy(), g)))
        jax.debug.callback(remember, params, grads, ordered=True)
        return optimizer.update(grads, state, params)
    return optax.GradientTransformation(optimizer.init, update)


def _loss(values, traces, field=False):
    def loss(result):
        traces.append(None)  # Python runs once per compilation, the callback once per step.
        if field:
            result, _ = result
        value = result.T.sum().real
        jax.debug.callback(lambda v: values.append(float(v)), value, ordered=True)
        return value
    return loss


@pytest.mark.parametrize('door', ['solve', 'solve_field', 'solve_field_eager'])
def test_fit_crosses_an_overlap_with_fresh_numpy_values_and_gradients(door, monkeypatch):
    _legacy_wavebit_enable_x64()
    params = dict(width=jnp.asarray(1.18, dtype=jnp.float64))
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(**params))
    initial_geometry = solver.freeze_geometry()
    # This is also the old tutorial's spelling: a freeze captured in a partial. fit must
    # replace it, including on solve_field where there is a second compiled boundary.
    forward = partial(getattr(solver, door), geometry=initial_geometry)
    if door != 'solve':
        forward = partial(forward, 2, 2, 2)
    records = []
    freeze = solver.freeze_geometry

    def record(**conditions):
        geometry = freeze(**conditions)
        records.append(geometry)
        return geometry

    monkeypatch.setattr(solver, 'freeze_geometry', record)
    values, observations, traces = [], [], []
    optimizer = _sgd(.8, momentum=.4)
    final = solver.fit(params, forward, _loss(values, traces, field=door != 'solve'),
                       _observed_optimizer(optimizer, observations), iteration=3,
                       parameterize=_design)
    jax.effects_barrier()
    assert len(records) == len(values) == len(observations) == 3
    assert records[0] != records[1], 'the fit must actually cross the overlap transition'
    assert float(observations[0][0]['width']) < 1.2 < float(observations[1][0]['width'])

    expected, state = params, optimizer.init(params)
    for value, (point, gradient) in zip(values, observations):
        width = float(point['width'])
        np.testing.assert_allclose(width, expected['width'], atol=2e-7, rtol=0)
        assert value == pytest.approx(_shape_value(width), abs=2e-10)
        truth = _shape_gradient(width)
        assert float(gradient['width']) == pytest.approx(truth, abs=2e-7, rel=2e-5)
        updates, state = optimizer.update(dict(width=jnp.asarray(truth)), state, expected)
        expected = optax.apply_updates(expected, updates)
    np.testing.assert_allclose(final['width'], expected['width'], atol=1e-6, rtol=0)
    assert solver.structure[0]['shapes'][1]['lx'] == pytest.approx(100*float(final['width']))
    with pytest.raises(RuntimeError, match=r'fit\(\.\.\.\)'):
        solver.reconstruct_field(2, 2, 2)
    assert np.isfinite(float(solver.solve().T.sum().real))


@pytest.mark.parametrize('strategy', [dict(factorization='nvf'),
                                      dict(fourier_coefficients='integrated'),
                                      dict(factorization='nvf', fourier_coefficients='integrated')],
                         ids=['nvf', 'integrated', 'both'])
@pytest.mark.parametrize('quantity', ['structure', 'mu'])
def test_fit_replans_a_raster_when_equal_pixels_separate(strategy, quantity, monkeypatch):
    _legacy_wavebit_enable_x64()
    cfg = dict(period=(700., 700.), thickness=(200.,), wavelength=900.,
               truncation_order=(1, 1), theta=.13, phi=.21, psi=.37, **strategy)
    # Equal neighbours initially share a block/interface mask. Their unequal derivatives
    # split it after the first update, making the original record stale.
    raster = np.ones((1, 6, 6))
    raster[0, 1:4, 2:4] = 4.
    params = {quantity: jnp.asarray(raster)}
    if quantity == 'mu':
        cfg['structure'] = np.where(raster == 4., 2.25, 1.)
    solver = _legacy_wavebit_rcwa(backend='jax', **cfg, **params)
    initial = solver.freeze_geometry()
    values, observations, traces, records = [], [], [], []
    freeze = solver.freeze_geometry

    def record(**conditions):
        geometry = freeze(**conditions)
        records.append(geometry)
        return geometry

    monkeypatch.setattr(solver, 'freeze_geometry', record)
    optimizer = optax.adam(.01)
    final = solver.fit([quantity], partial(solver.solve, geometry=initial),
                       _loss(values, traces), _observed_optimizer(optimizer, observations),
                       iteration=3)
    jax.effects_barrier()
    assert len(records) == len(values) == len(observations) == 3
    assert records[0] != records[1], 'the test must invalidate the starting raster record'
    with pytest.raises(_legacy_wavebit_StaleGeometryError):
        _legacy_wavebit_rcwa(backend='jax', **cfg, **observations[1][0]).solve(geometry=initial)
    expected, state = params, optimizer.init(params)
    for value, (point, gradient) in zip(values, observations):
        np.testing.assert_allclose(point[quantity], expected[quantity], atol=2e-10, rtol=0)
        fresh = _legacy_wavebit_rcwa(backend='jax', **cfg, **point)
        geometry = fresh.freeze_geometry()
        def loss(material):
            return fresh.solve(**{quantity: material}, geometry=geometry).T.sum().real
        want, grad = jax.value_and_grad(loss)(jnp.asarray(point[quantity]))
        assert value == pytest.approx(float(want), abs=2e-10)
        np.testing.assert_allclose(gradient[quantity], np.conj(grad), atol=2e-9, rtol=2e-7)
        updates, state = optimizer.update({quantity: jnp.conj(grad)}, state, expected)
        expected = optax.apply_updates(expected, updates)
    np.testing.assert_allclose(final[quantity], expected[quantity], atol=2e-9, rtol=0)
    np.testing.assert_array_equal(getattr(solver, quantity), final[quantity])


def test_fit_reuses_a_compilation_when_the_shape_plan_stays_equal(monkeypatch):
    params = dict(width=jnp.asarray(1., dtype=jnp.float64))
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(**params))
    records, values, traces = [], [], []
    freeze = solver.freeze_geometry

    def record(**conditions):
        geometry = freeze(**conditions)
        records.append(geometry)
        return geometry

    monkeypatch.setattr(solver, 'freeze_geometry', record)
    final = solver.fit(params, solver.solve, _loss(values, traces), optax.sgd(.1),
                       iteration=3, parameterize=_design)
    jax.effects_barrier()
    assert float(final['width']) != float(params['width'])
    assert len(records) == len(values) == 3
    assert all(g == records[0] for g in records)
    assert len(traces) == 1


@pytest.mark.parametrize('mapped', [False, True])
def test_custom_forward_requires_and_consumes_fresh_preparation(mapped):
    params = dict(width=jnp.asarray(1.18, dtype=jnp.float64))
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(**params))
    prepared, used, traces = [], [], []

    def prepare(**conditions):
        scene = conditions if mapped else _design(**conditions)
        geometry = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **scene).freeze_geometry()
        prepared.append(geometry)
        return geometry

    def forward(geometry, **conditions):
        used.append(geometry)
        return solver.solve(**(conditions if mapped else _design(**conditions)), geometry=geometry)

    with pytest.raises(TypeError, match='prepare_geometry'):
        solver.fit(params, forward, lambda r: r.T.sum().real, optax.sgd(.8), iteration=3)
    assert prepared == used == []
    values, observations = [], []
    final = solver.fit(params, forward, _loss(values, traces),
                       _observed_optimizer(_sgd(.8), observations), iteration=3,
                       prepare_geometry=prepare, parameterize=_design if mapped else None)
    jax.effects_barrier()
    assert len(prepared) == len(values) == len(observations) == 3
    assert prepared[0] != prepared[1]
    assert all(any(record == g for g in used) for record in prepared)
    for value, (point, grad) in zip(values, observations):
        assert value == pytest.approx(_shape_value(float(point['width'])), abs=2e-10)
        assert float(grad['width']) == pytest.approx(_shape_gradient(float(point['width'])), abs=2e-7)
    assert np.isfinite(float(final['width']))
    assert not hasattr(solver, 'width'), 'custom design variables are not solver attributes'


def test_fixed_partial_conditions_and_dynamic_overrides_match_a_fresh_solver():
    raster = np.ones((1, 6, 6))
    raster[0, 1:4, 2:4] = 4.
    cfg = dict(period=(700., 700.), wavelength=900., truncation_order=(1, 1),
               theta=.13, phi=.21, psi=.37)
    solver = _legacy_wavebit_rcwa(backend='jax', **cfg, thickness=(200.,), structure=np.ones_like(raster))
    # Preparation must use the partial's pattern/rules, while the dynamic thickness must
    # override the partial's 10, just as calling the partial directly would.
    fixed = dict(structure=raster, factorization='nvf', fourier_coefficients='integrated')
    forward = partial(solver.solve, thickness=(10.,), **fixed)
    values, observations, traces = [], [], []
    final = solver.fit(['thickness'], forward, _loss(values, traces),
                       _observed_optimizer(optax.sgd(100.), observations), iteration=3)
    jax.effects_barrier()

    def fresh(t):
        return float(_legacy_wavebit_rcwa(backend='numpy', **cfg, **fixed, thickness=(t,)).solve().T.sum().real)

    assert len(values) == len(observations) == 3
    for value, (point, gradient) in zip(values, observations):
        t = float(point['thickness'][0])
        assert value == pytest.approx(fresh(t), abs=2e-10)
        fd = (fresh(t+1e-3) - fresh(t-1e-3)) / 2e-3
        assert float(gradient['thickness'][0]) == pytest.approx(fd, abs=2e-10, rel=2e-5)
    np.testing.assert_array_equal(solver.structure, raster)
    np.testing.assert_array_equal(solver.thickness, final['thickness'])
    assert len(traces) == 1


def test_nested_design_variables_retain_their_gradients_and_final_mapping():
    params = dict(design=dict(width=jnp.asarray(1., dtype=jnp.float64)))
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(1.))
    final = solver.fit(params, solver.solve, lambda r: r.T.sum().real, optax.sgd(.1),
                       iteration=1, parameterize=lambda design: _design(design['width']))
    want = 1. - .1 * _shape_gradient(1.)
    assert float(final['design']['width']) == pytest.approx(want, abs=2e-8)
    assert solver.structure[0]['shapes'][1]['lx'] == pytest.approx(100*want, abs=2e-6)
    assert not hasattr(solver, 'design')


def test_failed_traced_loss_leaves_concrete_conditions_and_no_reusable_solve_state():
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(1.))
    expected = float(solver.solve().T.sum().real)

    def fail(result):
        raise RuntimeError('intentional objective failure')

    with pytest.raises(RuntimeError, match='intentional objective failure'):
        solver.fit(['thickness'], solver.solve, fail, optax.sgd(.1), iteration=2)
    from wavebit.rcwa._engine.engine import _holds_tracer
    assert not any(_holds_tracer(v) for v in vars(solver).values())
    with pytest.raises(RuntimeError, match=r'fit\(\.\.\.\)'):
        solver.reconstruct_field(2, 2, 2)
    assert float(solver.solve().T.sum().real) == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize('mapping, error, message', [
    (lambda width: [], TypeError, 'mapping'),
    (lambda width: dict(geometry=None), ValueError, 'not a design condition'),
    (lambda width: dict(misspelled_structure=[]), TypeError, 'misspelled_structure'),
])
def test_bad_parameterization_is_refused_before_compiling(mapping, error, message):
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(1.))
    values, traces = [], []
    with pytest.raises(error, match=message):
        solver.fit(dict(width=jnp.asarray(1.)), solver.solve, _loss(values, traces),
                   optax.sgd(.1), parameterize=mapping)
    assert values == traces == []
    assert float(solver.solve().T.sum().real) == pytest.approx(_shape_value(1.), abs=2e-12)


def test_another_solvers_forward_is_not_mistaken_for_this_solvers_scene():
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(1.))
    other = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(1.3))
    with pytest.raises(TypeError, match='prepare_geometry'):
        solver.fit(['thickness'], other.solve, lambda r: r.T.sum().real, optax.sgd(.1))


def test_zero_iterations_preserve_the_design_and_eager_solve_state(monkeypatch):
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(1.))
    solver.solve()
    field = solver.reconstruct_field(2, 2, 2)
    thickness = np.asarray(solver.thickness).copy()

    def unexpected(**kwargs):
        pytest.fail('zero iterations must not prepare a geometry')

    monkeypatch.setattr(solver, 'freeze_geometry', unexpected)
    final = solver.fit(['thickness'], solver.solve, lambda r: r.T.sum().real,
                       optax.sgd(.1), iteration=0)
    np.testing.assert_array_equal(final['thickness'], thickness)
    np.testing.assert_array_equal(solver.reconstruct_field(2, 2, 2), field)


def test_preparation_failure_stops_before_another_step_and_clears_retained_tracers():
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(1.))
    prepared = []
    values, observations, traces = [], [], []

    def prepare(**conditions):
        prepared.append(conditions)
        if len(prepared) == 2:
            raise RuntimeError('intentional preparation failure')
        return solver.freeze_geometry(**conditions)

    def forward(geometry, **conditions):
        return solver.solve(geometry=geometry, **conditions)

    with pytest.raises(RuntimeError, match='intentional preparation failure'):
        solver.fit(['thickness'], forward, _loss(values, traces),
                   _observed_optimizer(optax.sgd(100.), observations), iteration=3,
                   prepare_geometry=prepare)
    jax.effects_barrier()
    assert len(prepared) == 2
    assert len(values) == len(observations) == 1
    from wavebit.rcwa._engine.engine import _holds_tracer
    assert not any(_holds_tracer(v) for v in vars(solver).values())
    with pytest.raises(RuntimeError, match=r'fit\(\.\.\.\)'):
        solver.reconstruct_field(2, 2, 2)
    assert np.isfinite(float(solver.solve().T.sum().real))


def test_geometry_preparation_cannot_be_hidden_inside_an_outer_jit():
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(1.))

    def run(width):
        return solver.fit(dict(width=width), solver.solve, lambda r: r.T.sum().real,
                          optax.sgd(.1), parameterize=_design)['width']

    with pytest.raises(ValueError, match='freeze_geometry'):
        jax.jit(run)(jnp.asarray(1.))
    from wavebit.rcwa._engine.engine import _holds_tracer
    assert not any(_holds_tracer(v) for v in vars(solver).values())


def test_fit_differentiates_a_field_intensity_objective_against_fresh_numpy_fields():
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(1.))
    values, observations = [], []

    def loss(pair):
        _, field = pair
        value = jnp.mean(jnp.abs(field)**2)
        jax.debug.callback(lambda v: values.append(float(v)), value, ordered=True)
        return value

    def fresh(width):
        reference = _legacy_wavebit_rcwa(backend='numpy', **SHAPE_KW, **_design(width))
        _, field = reference.solve_field(2, 2, 2)
        return float(np.mean(np.abs(field)**2))

    final = solver.fit(dict(width=jnp.asarray(1., dtype=jnp.float64)),
                       partial(solver.solve_field, 2, 2, 2), loss,
                       _observed_optimizer(optax.sgd(1e-3), observations), iteration=2,
                       parameterize=_design)
    jax.effects_barrier()
    assert len(values) == len(observations) == 2
    expected = 1.
    for value, (point, gradient) in zip(values, observations):
        width = float(point['width'])
        assert width == pytest.approx(expected, abs=2e-9)
        assert value == pytest.approx(fresh(width), abs=2e-10)
        truth = (fresh(width+1e-5) - fresh(width-1e-5)) / 2e-5
        assert float(gradient['width']) == pytest.approx(truth, abs=2e-7, rel=2e-5)
        expected -= 1e-3 * truth
    assert float(final['width']) == pytest.approx(expected, abs=2e-9)


def _airy_reflection(w):
    epsilon = 4. + (w-800.)*.025
    n = np.sqrt(complex(epsilon))
    r01, r12 = (1-n)/(1+n), (n-1)/(n+1)
    phase = np.exp(2j*np.pi*n*120/w)
    r = (r01 + r12*phase**2)/(1 + r01*r12*phase**2)
    return abs(r)**2


@pytest.mark.parametrize('door', ['solve', 'solve_field'])
def test_dispersive_wavelength_fit_uses_live_materials(door):
    material = _legacy_wavebit_TabulatedMaterial([800., 1000.], [4., 9.])
    solver = _legacy_wavebit_rcwa(backend='jax', wavelength=900., period=(600., 600.),
                         structure=[dict(background=material, shapes=[])], thickness=[120.],
                         truncation_order=(0, 0), polarization=(1., 0.),
                         factorization='li', answers='field')
    forward = getattr(solver, door)
    if door == 'solve_field':
        forward = partial(forward, 2, 2, 2)
    observations, values = [], []
    def loss(result):
        if door == 'solve_field':
            result, _ = result
        value = result.R.sum().real
        jax.debug.callback(lambda v: values.append(float(v)), value, ordered=True)
        return value
    final = solver.fit(['wavelength'], forward, loss,
                       _observed_optimizer(optax.sgd(1000.), observations), iteration=3)
    jax.effects_barrier()
    expected = 900.
    for value, (params, gradient) in zip(values, observations):
        w = float(params['wavelength'])
        assert w == pytest.approx(expected, abs=1e-8)
        assert value == pytest.approx(_airy_reflection(w), abs=1e-11)
        g = (_airy_reflection(w+.01)-_airy_reflection(w-.01))/.02
        assert float(gradient['wavelength']) == pytest.approx(g, abs=1e-10)
        expected -= 1000.*g
    assert len(values) == 3
    assert float(final['wavelength']) == pytest.approx(expected, abs=1e-7)
    assert complex(solver.structure[0]['background'].epsilon) == pytest.approx(
        4.+(float(final['wavelength'])-800.)*.025)


@pytest.mark.parametrize('door', ['solve', 'solve_field'])
def test_joint_shape_and_cell_optimization_matches_fresh_numpy(door):
    def parameterize(width, scale):
        return dict(**_design(width), period=(300.*scale, 300.))
    cfg = {k: v for k, v in SHAPE_KW.items() if k != 'period'}
    params = dict(width=jnp.asarray(1.1, dtype=jnp.float64),
                  scale=jnp.asarray(1.03, dtype=jnp.float64))
    solver = _legacy_wavebit_rcwa(backend='jax', **cfg, **parameterize(**params))
    observations, values = [], []
    forward = getattr(solver, door)
    if door == 'solve_field':
        forward = partial(forward, 2, 2, 2)
    def loss(result):
        if door == 'solve_field':
            result, _ = result
        value = result.T.sum().real
        jax.debug.callback(lambda v: values.append(float(v)), value, ordered=True)
        return value
    final = solver.fit(params, forward, loss, _observed_optimizer(optax.sgd(.01), observations),
                       iteration=2, parameterize=parameterize)
    jax.effects_barrier()
    def fresh(values):
        return float(_legacy_wavebit_rcwa(backend='numpy', **cfg, **parameterize(**values)).solve().T.sum().real)
    for value, (point, gradient) in zip(values, observations):
        assert value == pytest.approx(fresh(point), abs=1e-10)
        for key in point:
            plus, minus = dict(point), dict(point)
            plus[key], minus[key] = plus[key]+1e-5, minus[key]-1e-5
            g = (fresh(plus)-fresh(minus))/2e-5
            assert float(gradient[key]) == pytest.approx(g, abs=2e-7, rel=2e-5)
    assert len(observations) == 2
    assert np.isfinite(float(final['scale']))


@pytest.mark.parametrize('door, wrapped', [('solve_field', 'solve'),
                                         ('solve', 'solve_field'),
                                         ('solve', 'solve_field_eager')])
def test_fit_method_detection_tolerates_other_instrumented_entry_points(door, wrapped, monkeypatch):
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(1.))
    original = getattr(solver, wrapped)

    def instrumented(*args, **conditions):
        return original(*args, **conditions)

    monkeypatch.setattr(solver, wrapped, instrumented)
    forward = getattr(solver, door)
    if door == 'solve_field':
        forward = partial(forward, 2, 2, 2)

    def loss(result):
        return (result[0] if door == 'solve_field' else result).T.sum().real

    final = solver.fit(['thickness'], forward, loss, optax.sgd(.1), iteration=1)
    assert np.isfinite(np.asarray(final['thickness'])).all()


def test_a_wrapped_forward_uses_the_custom_preparation_contract(monkeypatch):
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(1.))
    original = solver.solve

    def instrumented(**conditions):
        return original(**conditions)

    monkeypatch.setattr(solver, 'solve', instrumented)
    with pytest.raises(TypeError, match='prepare_geometry'):
        solver.fit(['thickness'], solver.solve, lambda r: r.T.sum().real, optax.sgd(.1))
    final = solver.fit(['thickness'], solver.solve, lambda r: r.T.sum().real,
                       optax.sgd(.1), prepare_geometry=solver.freeze_geometry)
    assert np.isfinite(np.asarray(final['thickness'])).all()


@pytest.mark.parametrize('door', ['solve', 'solve_field', 'solve_field_eager', 'wrapper'])
def test_fit_replaces_a_geometry_captured_as_a_positional_argument(door):
    _legacy_wavebit_enable_x64()
    params = dict(width=jnp.asarray(1.18, dtype=jnp.float64))
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(**params))
    initial = solver.freeze_geometry()
    if door == 'wrapper':
        def method(geometry, **conditions):
            return solver.solve(geometry=geometry, **conditions)
    else:
        method = getattr(solver, door)
    # The solve's option after geometry must survive removal of that positional slot.
    args = ((initial, False) if door == 'solve' else (initial,) if door == 'wrapper'
            else (2, 2, 2, ('incident',), initial))
    values, observations, traces = [], [], []
    final = solver.fit(params, partial(method, *args),
                       _loss(values, traces, field=door in ('solve_field', 'solve_field_eager')),
                       _observed_optimizer(_sgd(.8), observations), iteration=2,
                       parameterize=_design,
                       prepare_geometry=solver.freeze_geometry if door == 'wrapper' else None)
    jax.effects_barrier()
    assert len(values) == len(observations) == 2
    assert float(observations[0][0]['width']) < 1.2 < float(observations[1][0]['width'])
    expected = 1.18
    for value, (point, gradient) in zip(values, observations):
        assert float(point['width']) == pytest.approx(expected, abs=2e-7)
        # The value at the width the fit actually solved, as the two tests above read it. Read
        # at the replicated width it carried the painted branch's slope (3.07) times the first
        # gradient's replication error -- 2.5e-9 on Linux, where the central difference's
        # round-off is 1e-9, against 6e-10 on macOS -- which is the line above's business.
        assert value == pytest.approx(_shape_value(float(point['width'])), abs=2e-9)
        want = _shape_gradient(expected)
        assert float(gradient['width']) == pytest.approx(want, abs=2e-7)
        expected -= .8 * max(-_CLIP, min(_CLIP, want))
    assert float(final['width']) == pytest.approx(expected, abs=4e-7)


@pytest.mark.parametrize('mode', ['builtin', 'wrapper', 'detached_preparation'])
@pytest.mark.parametrize('factorization', ['li', 'nvf'])
def test_explicit_preparation_and_forward_use_the_same_partial_conditions(mode, factorization):
    raster = np.ones((1, 6, 6))
    raster[0, 1:4, 2:4] = 4.
    cfg = dict(period=(700., 700.), wavelength=900., truncation_order=(1, 1),
               theta=.13, phi=.21, psi=.37, fourier_coefficients='integrated')
    solver = _legacy_wavebit_rcwa(backend='jax', **cfg, structure=np.ones_like(raster), thickness=(200.,))
    initial = solver.freeze_geometry()
    fixed = dict(structure=raster, factorization=factorization)
    records = []

    def prepare(**conditions):
        records.append(conditions)
        if mode == 'detached_preparation':
            # A callback need not mutate the fitted solver. Final conditions must still be
            # assigned on that solver, including the fixed part of the forward's scene.
            return _legacy_wavebit_rcwa(backend='jax', **(cfg | conditions)).freeze_geometry()
        return solver.freeze_geometry(**conditions)

    def wrapper(geometry, **conditions):
        return solver.solve(geometry=geometry, **conditions)

    method = wrapper if mode == 'wrapper' else solver.solve
    options = {} if mode == 'wrapper' else dict(auto_detune=False)
    forward = partial(partial(method, thickness=(10.,), geometry=initial), **fixed, **options)
    values, observations, traces = [], [], []
    optimizer = optax.sgd(500., momentum=.2)
    final = solver.fit(['thickness'], forward, _loss(values, traces),
                       _observed_optimizer(optimizer, observations), iteration=2,
                       prepare_geometry=prepare)
    jax.effects_barrier()
    assert len(records) == len(values) == len(observations) == 2

    def fresh(t):
        return float(_legacy_wavebit_rcwa(backend='numpy', **cfg, **fixed, thickness=(t,)).solve().T.sum().real)

    expected, momentum = 200., 0.
    for prepared, value, (point, gradient) in zip(records, values, observations):
        assert set(prepared) == {'structure', 'factorization', 'thickness'}
        np.testing.assert_array_equal(prepared['structure'], raster)
        t = float(point['thickness'][0])
        assert float(prepared['thickness'][0]) == pytest.approx(t, abs=1e-12)
        assert t == pytest.approx(expected, abs=1e-8)
        assert value == pytest.approx(fresh(t), abs=2e-10)
        want = (fresh(t+1e-3)-fresh(t-1e-3))/2e-3
        assert float(gradient['thickness'][0]) == pytest.approx(want, abs=2e-10)
        momentum = .2 * momentum + want
        expected -= 500. * momentum
    assert float(final['thickness'][0]) == pytest.approx(expected, abs=2e-7)
    if mode != 'wrapper':
        np.testing.assert_array_equal(solver.structure, raster)
        np.testing.assert_array_equal(solver.thickness, final['thickness'])
        assert solver.factorization == factorization


def test_custom_preparation_receives_custom_partial_keywords_with_dynamic_overrides():
    solver = _legacy_wavebit_rcwa(backend='jax', **SHAPE_KW, **_design(1.))
    prepared, values, traces, observations = [], [], [], []

    def prepare(width, scale):
        prepared.append((float(width), scale))
        return solver.freeze_geometry(**_design(width*scale))

    def forward(width, scale, geometry):
        return solver.solve(**_design(width*scale), geometry=geometry)

    params = dict(width=jnp.asarray(1.5, dtype=jnp.float64))
    final = solver.fit(params, partial(forward, width=99., scale=.75), _loss(values, traces),
                       _observed_optimizer(optax.sgd(.1), observations), iteration=2,
                       prepare_geometry=prepare)
    jax.effects_barrier()
    assert len(prepared) == len(values) == len(observations) == 2
    for (width, scale), value, (point, gradient) in zip(prepared, values, observations):
        assert width == pytest.approx(float(point['width']), abs=1e-12)
        assert scale == .75
        assert value == pytest.approx(_shape_value(width*scale), abs=2e-10)
        assert float(gradient['width']) == pytest.approx(scale*_shape_gradient(width*scale), abs=2e-7)
    assert float(final['width']) != 1.5
