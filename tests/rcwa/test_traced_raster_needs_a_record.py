"""A raster that ``jax.jit`` or ``jax.vmap`` traces is refused before planning, with the remedy.

Some raster builds take a geometry decision from the raster's values: the NVF normal field
reads which pixels border another phase, the ``integrated`` coefficients read where runs of
equal pixels end, and the tensor NVF build labels which pixels hold the same tensor. A raster
that a ``jax.jit`` or ``jax.vmap`` trace holds has no value to read, so without a frozen record
to replay there is nothing to plan from, and the solve is refused. That refusal used to be
whatever JAX raised from deep inside the planning -- ``TracerBoolConversionError`` from
``_as_bools``, or "freeze_geometry() before tracing a tensor raster" on the tensor path -- which
says nothing about what to do. It is now one ``ValueError``, raised before planning reads a
value, that names the traced raster, the decision it cannot take, and the two ways out: freeze
the geometry eagerly and pass ``geometry=``, or ``rcwa.prepare`` outside the transform.

The refusal must not reach further than the decision it guards. A traced raster *with* its
record still solves; the pairings that plan nothing still trace without one; an eager
``jax.grad`` tracer carries its value and still plans; and a raster that is a constant of the
trace is planned as it is eagerly (``tests/rcwa/test_raster_constant_under_jit.py``).

The public lifecycle never reaches the engine with such a raster: ``rcwa.prepare`` and
``rcwa.run`` refuse a traced problem first, and ``Prepared.run`` replays the record ``prepare``
took. The last tests pin that.
"""

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa

import numpy as np
import pytest

pytestmark = pytest.mark.shard_unit(per='test')

TRUNCATION = (2, 2)
BASE = dict(wavelength=620., period=(1000., 900.), thickness=(300.,), top=1.0, bot=2.1025,
            theta=0.2, phi=0.3, psi=0.4, truncation_order=TRUNCATION)
TENSOR = np.array([[2.9 + .02j, .21 - .05j, .13 + .04j],
                   [.21 - .05j, 3.4 + .01j, -.19 - .09j],
                   [.13 + .04j, -.19 - .09j, 2.8]])
TENSOR = (TENSOR + TENSOR.conj().T) / 2


def _rod(value=12.25):
    cell = np.ones((1, 8, 8))
    cell[0, 1:5, 2:7] = value
    return cell


def _tensor_rod():
    cell = np.broadcast_to(np.eye(3, dtype=complex), (1, 8, 8, 3, 3)).copy()
    cell[0, 1:5, 2:7] = TENSOR
    return cell


def _mu_rod():
    return _rod(1.7)


#: Every raster build that plans from the raster's values when it replays no record, and the
#: raster it is solved on.
PLANNING = {
    'nvf': (dict(factorization='nvf'), _rod),
    'nvf-integrated': (dict(factorization='nvf', fourier_coefficients='integrated'), _rod),
    'nvf-contour': (dict(factorization='nvf', raster_normal_method='contour'), _rod),
    'nvf-circular': (dict(factorization='nvf', nvf_projector='circular'), _rod),
    'li-integrated': (dict(factorization='li', fourier_coefficients='integrated'), _rod),
    'laurent-integrated': (dict(factorization='laurent', fourier_coefficients='integrated'), _rod),
    'nvf-tensor': (dict(factorization='nvf'), _tensor_rod),
}

#: The pairings that plan nothing, so a traced raster needs no record.
PLANS_NOTHING = {
    'li-sampled': (dict(factorization='li'), _rod),
    'laurent-sampled': (dict(factorization='laurent'), _rod),
    'li-tensor': (dict(factorization='li'), _tensor_rod),
}

#: The magnetic builds that plan from a traced ``mu`` raster beside a constant permittivity.
PLANNING_MU = ['nvf', 'nvf-integrated', 'li-integrated']

TRANSFORMS = ['jit', 'vmap', 'jit-vmap', 'jit-grad']


@pytest.fixture(autouse=True)
def _jax():
    pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()


def _solver(options, raster, **extra):
    return _legacy_wavebit_rcwa(backend='jax', structure=raster(), **BASE, **options, **extra)


def _traced(transform, f, x):
    """``f`` of the raster ``x``, with the raster traced the way ``transform`` traces it."""
    import jax
    import jax.numpy as jnp
    batch = jnp.stack([jnp.asarray(x), jnp.asarray(x)])
    if transform == 'jit':
        return jax.jit(f)(jnp.asarray(x))
    if transform == 'vmap':
        return jax.vmap(f)(batch)
    if transform == 'jit-vmap':
        return jax.jit(jax.vmap(f))(batch)
    assert transform == 'jit-grad'
    return jax.jit(jax.grad(lambda c: jnp.sum(f(c)).real))(jnp.asarray(x))


def _assert_told_to_freeze(error, quantity='permittivity', keyword='structure'):
    """The refusal names what is traced and both remedies, and is not JAX's own error."""
    message = str(error)
    assert f'{quantity} raster (`{keyword}`) is traced' in message, message
    assert 'freeze_geometry()' in message and f'solve({keyword}=' in message, message
    assert 'geometry=geometry' in message, message
    assert 'rcwa.prepare' in message and 'Prepared.run' in message, message
    assert 'docs/guide/common/autodiff.md' in message, message


# ----------------------------------------------------------------- refused, with the remedy

@pytest.mark.parametrize('transform', TRANSFORMS)
@pytest.mark.parametrize('path', list(PLANNING))
def test_a_traced_raster_without_a_record_is_told_to_freeze(path, transform):
    """Every planning build, under every transform that leaves the raster without a value."""
    options, raster = PLANNING[path]
    solver = _solver(options, raster)
    with pytest.raises(ValueError) as caught:
        _traced(transform, lambda c: solver.solve(structure=c, auto_detune=False).R, raster())
    _assert_told_to_freeze(caught.value)


@pytest.mark.parametrize('path', ['nvf', 'nvf-integrated', 'li-integrated', 'nvf-tensor'])
def test_the_refusal_names_the_decision_the_build_could_not_take(path):
    options, raster = PLANNING[path]
    solver = _solver(options, raster)
    with pytest.raises(ValueError) as caught:
        _traced('jit', lambda c: solver.solve(structure=c, auto_detune=False).R, raster())
    message = str(caught.value)
    named = {'nvf': ['NVF normal field'],
             'nvf-integrated': ['NVF normal field', 'integrated block compression'],
             'li-integrated': ['integrated block compression'],
             'nvf-tensor': ['tensor NVF phase labels']}[path]
    for decision in named:
        assert decision in message, message
    assert 'No `geometry=` was given' in message, message


@pytest.mark.parametrize('transform', ['jit', 'jit-vmap'])
@pytest.mark.parametrize('path', PLANNING_MU)
def test_a_traced_permeability_raster_is_told_to_freeze(path, transform):
    """A constant permittivity beside a traced ``mu``: the permeability pass plans on its own."""
    options, raster = PLANNING[path]
    solver = _solver(options, raster, mu=_mu_rod())
    with pytest.raises(ValueError) as caught:
        _traced(transform, lambda m: solver.solve(mu=m, auto_detune=False).R, _mu_rod())
    _assert_told_to_freeze(caught.value, quantity='permeability', keyword='mu')


def test_the_convolution_matrix_doors_are_refused_alike():
    """``eps_conv_matrices`` and ``mu_conv_matrices`` build the same slots outside a solve."""
    import jax
    import jax.numpy as jnp
    solver = _solver(dict(factorization='nvf'), _rod, mu=_mu_rod())
    with pytest.raises(ValueError) as caught:
        jax.jit(lambda c: solver.eps_conv_matrices(structure=c)[0])(jnp.asarray(_rod()))
    _assert_told_to_freeze(caught.value)
    with pytest.raises(ValueError) as caught:
        jax.jit(lambda m: solver.mu_conv_matrices(mu=m)[0])(jnp.asarray(_mu_rod()))
    _assert_told_to_freeze(caught.value, quantity='permeability', keyword='mu')


def test_a_record_that_lacks_the_decision_is_refused_as_no_record():
    """A record the build finds nothing in for this raster sends it back to planning.

    The factorization stamp lets an NVF record taken with sampled coefficients reach an
    integrated NVF build, which finds no compression in it; a record taken before a ``mu``
    raster was set holds nothing for the permeability. Eagerly both plan what is missing.
    Under a trace they were the same raw tracer error as no record at all.
    """
    import jax
    import jax.numpy as jnp
    sampled = _solver(dict(factorization='nvf'), _rod).freeze_geometry()
    integrated = _solver(PLANNING['nvf-integrated'][0], _rod)
    integrated.solve(geometry=sampled, auto_detune=False)            # eagerly: planned, answered
    with pytest.raises(ValueError) as caught:
        jax.jit(lambda c: integrated.solve(structure=c, geometry=sampled, auto_detune=False).T)(
            jnp.asarray(_rod()))
    _assert_told_to_freeze(caught.value)
    assert 'integrated block compression' in str(caught.value)
    assert 'The `geometry=` given holds no record of' in str(caught.value)

    magnetic = _solver(dict(factorization='nvf'), _rod, mu=_mu_rod())
    with pytest.raises(ValueError) as caught:
        jax.jit(lambda m: magnetic.solve(mu=m, geometry=sampled, auto_detune=False).T)(
            jnp.asarray(_mu_rod()))
    _assert_told_to_freeze(caught.value, quantity='permeability', keyword='mu')
    assert 'The `geometry=` given holds no record of' in str(caught.value)


# --------------------------------------------------------- and nothing past what it guards

@pytest.mark.parametrize('transform', ['jit', 'jit-vmap'])
@pytest.mark.parametrize('path', list(PLANNING))
def test_a_traced_raster_with_its_record_solves_as_it_does_eagerly(path, transform):
    """The remedy the message prints works: freeze on the concrete raster, replay traced."""
    options, raster = PLANNING[path]
    solver = _solver(options, raster)
    geometry = solver.freeze_geometry()
    want = np.asarray(solver.solve(geometry=geometry, auto_detune=False).R)
    got = _traced(transform, lambda c: solver.solve(structure=c, geometry=geometry,
                                                    auto_detune=False).R, raster())
    got = np.asarray(got if transform == 'jit' else got[1])
    assert np.any(want > 1e-3), want                     # a reflection there to compare
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-13)


@pytest.mark.parametrize('path', PLANNING_MU)
def test_a_traced_permeability_raster_with_its_record_solves_as_it_does_eagerly(path):
    import jax
    options, raster = PLANNING[path]
    solver = _solver(options, raster, mu=_mu_rod())
    geometry = solver.freeze_geometry()
    want = np.asarray(solver.solve(geometry=geometry, auto_detune=False).T)
    got = jax.jit(lambda m: solver.solve(mu=m, geometry=geometry, auto_detune=False).T)(
        _mu_rod())
    np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=1e-13)


@pytest.mark.parametrize('transform', ['jit', 'jit-vmap'])
@pytest.mark.parametrize('path', list(PLANS_NOTHING))
def test_the_pairings_that_plan_nothing_still_trace_without_a_record(path, transform):
    """Nothing to plan, so nothing to refuse: the guard asks only where a build would plan."""
    options, raster = PLANS_NOTHING[path]
    solver = _solver(options, raster)
    got = _traced(transform, lambda c: solver.solve(structure=c, auto_detune=False).R, raster())
    assert np.all(np.isfinite(np.asarray(got)))


@pytest.mark.parametrize('path', ['nvf', 'li-integrated'])
def test_an_eager_grad_tracer_still_plans_on_its_value(path):
    """A ``jax.grad`` tracer without ``jit`` carries its primal, so the build plans on it.

    It is the derivative the record gives under ``jax.jit``, to round-off: the decisions are
    the same, taken once from the primal and once from the frozen copy of it.
    """
    import jax
    import jax.numpy as jnp
    options, raster = PLANNING[path]
    solver = _solver(options, raster)
    geometry = solver.freeze_geometry()

    def loss(c, geometry=None):
        return jnp.sum(solver.solve(structure=c, geometry=geometry, auto_detune=False).T)

    eager = np.asarray(jax.grad(loss)(jnp.asarray(raster())))
    traced = np.asarray(jax.jit(jax.grad(lambda c: loss(c, geometry)))(jnp.asarray(raster())))
    assert np.any(np.abs(eager) > 1e-6), eager
    np.testing.assert_allclose(traced, eager, rtol=1e-10, atol=1e-14)


# --------------------------------------------------------------- the public lifecycle

def _problem(epsilon, mu=None, wavelength=620.):
    from wavebit import Material
    from wavebit import Units
    from wavebit import rcwa
    layer = rcwa.Layer(thickness=300., epsilon=epsilon, **({} if mu is None else {'mu': mu}))
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=(1000., 900.), bottom=Material(epsilon=2.1025), layers=(layer,)),
        excitation=rcwa.Excitation(wavelength=wavelength, sources=(rcwa.PlaneWave(),)))


#: Public problems whose layer is a planning raster: (numerics, epsilon, mu, which is traced).
PUBLIC = {
    'nvf': (dict(factorization='nvf'), _rod()[0], None, 'epsilon'),
    'li-integrated': (dict(factorization='li', fourier_coefficients='integrated'), _rod()[0],
                      None, 'epsilon'),
    'nvf-tensor': (dict(factorization='nvf'), _tensor_rod()[0], None, 'epsilon'),
    'nvf-mu': (dict(factorization='nvf'), _rod()[0], _mu_rod()[0], 'mu'),
}


def _public_problem(epsilon, mu, traced):
    """The public problem as a function of its traced raster, and that raster's value."""
    def problem(x):
        return _problem(x, mu) if traced == 'epsilon' else _problem(epsilon, x)
    return problem, (epsilon if traced == 'epsilon' else mu)


@pytest.mark.parametrize('transform', ['jit', 'vmap', 'jit-vmap'])
@pytest.mark.parametrize('door', ['run', 'prepare'])
@pytest.mark.parametrize('case', list(PUBLIC))
def test_the_public_lifecycle_refuses_a_traced_problem_before_any_planning(case, door, transform):
    """``rcwa.prepare`` and ``rcwa.run`` refuse a traced problem with ``prepare.concrete``."""
    from wavebit import UnsupportedFeatureError
    from wavebit import rcwa
    numerics, epsilon, mu, traced = PUBLIC[case]
    problem, raster = _public_problem(epsilon, mu, traced)
    numerics = rcwa.Numerics(orders=TRUNCATION, **numerics)

    def f(x):
        if door == 'prepare':
            rcwa.prepare(problem(x), numerics=numerics, backend='jax')
            return x.real.sum()
        result = rcwa.run(problem(x), numerics=numerics, backend='jax')
        return result.transmission.order(0, 0).efficiency[0]

    with pytest.raises(UnsupportedFeatureError) as caught:
        _traced(transform, f, raster)
    assert caught.value.code == 'prepare.concrete'
    assert 'Prepare outside jax.jit or jax.vmap' in str(caught.value)
    assert 'Prepared.run' in str(caught.value)


@pytest.mark.parametrize('transform', ['jit', 'jit-vmap'])
@pytest.mark.parametrize('case', list(PUBLIC))
def test_a_held_preparation_replays_its_record_on_a_traced_raster(case, transform):
    """``Prepared.run`` carries the record ``prepare`` took, so the traced run plans nothing."""
    from wavebit import rcwa
    numerics, epsilon, mu, traced = PUBLIC[case]
    problem, raster = _public_problem(epsilon, mu, traced)
    prepared = rcwa.prepare(problem(raster), numerics=rcwa.Numerics(orders=TRUNCATION, **numerics),
                            backend='jax')

    def f(x):
        return prepared.run(problem(x)).transmission.order(0, 0).efficiency[0]

    want = float(f(raster))
    got = np.asarray(_traced(transform, f, raster))
    np.testing.assert_allclose(got if transform == 'jit' else got[1], want, rtol=0, atol=1e-13)


# ------------------------------------------------------------ the freeze, inside the trace

def _freeze_inside(transform, solver, keyword, x, kept=None):
    """Assign a traced raster to the solver inside the trace and freeze there.

    The one way a solver holds a raster tracer at the freeze: ``solve()`` puts a traced
    structure back after its call, so only a direct assignment inside the trace leaves one.
    The concrete raster is put back afterwards by hand, as the assignment is not undone.
    """
    import jax.numpy as jnp

    def f(c):
        setattr(solver, keyword, c)
        record = solver.freeze_geometry()
        if kept is not None:
            kept.append(record)
        return jnp.sum(jnp.abs(c)) * 0.0
    try:
        return _traced(transform, f, x)
    finally:
        setattr(solver, keyword, x)


def _assert_freeze_refused(error, reads, quantity='permittivity', keyword='structure'):
    message = str(error)
    assert message.startswith('freeze_geometry:'), message
    assert f"this solver's {quantity} raster (`{keyword}`)" in message, message
    assert reads in message and 'concrete solver' in message, message
    assert f'solve({keyword}=x, geometry=geometry)' in message, message


#: What each planning path's freeze reads off the raster, as the refusal names it.
FREEZE_READS = {'nvf': 'NVF normal field', 'nvf-integrated': 'NVF normal field',
                'nvf-contour': 'NVF normal field', 'nvf-circular': 'NVF normal field',
                'li-integrated': 'integrated block compression',
                'laurent-integrated': 'integrated block compression',
                'nvf-tensor': 'tensor NVF phase labels'}


@pytest.mark.parametrize('transform', ['jit', 'vmap'])
@pytest.mark.parametrize('path', list(PLANNING))
def test_a_freeze_on_a_traced_raster_is_refused_by_name(path, transform):
    """The freeze plans from the raster's values; a raster tracer has none, so it is refused.

    It died inside the planning with JAX's own error: ``TracerArrayConversionError`` from
    the normal field and the tensor labels, ``TracerBoolConversionError`` from the
    compression -- while the freeze's own refusal skipped rasters, on the claim that they
    said so on their own.
    """
    options, raster = PLANNING[path]
    solver = _solver(options, raster)
    with pytest.raises(ValueError) as caught:
        _freeze_inside(transform, solver, 'structure', raster())
    _assert_freeze_refused(caught.value, FREEZE_READS[path])
    assert solver.freeze_geometry() is not None              # concrete again: a record


@pytest.mark.parametrize('path', PLANNING_MU)
def test_a_freeze_on_a_traced_permeability_raster_is_refused_by_name(path):
    options, raster = PLANNING[path]
    solver = _solver(options, raster, mu=_mu_rod())
    with pytest.raises(ValueError) as caught:
        _freeze_inside('jit', solver, 'mu', _mu_rod())
    _assert_freeze_refused(caught.value, FREEZE_READS[path], quantity='permeability', keyword='mu')


@pytest.mark.parametrize('path', list(PLANS_NOTHING))
def test_a_freeze_that_reads_no_value_still_records_a_traced_raster(path):
    """Sampled Li and Laurent, and tensor Li, decide nothing from values: an honest record.

    It is the record the eager freeze takes, so the refusal must not reach these builds.
    """
    options, raster = PLANS_NOTHING[path]
    solver = _solver(options, raster)
    eager, kept = solver.freeze_geometry(), []
    _freeze_inside('jit', solver, 'structure', raster(), kept=kept)
    (traced,) = kept
    assert traced.tensor_geometry == eager.tensor_geometry
    for name in ('normal_field', 'compression', 'mu_normal_field', 'mu_compression'):
        assert getattr(traced, name) is None and getattr(eager, name) is None, name


def _assert_same_record(got, want, path='record'):
    """Two frozen records equal field by field, every array to the byte and of the same dtype.

    A record holds decisions only, so the one a freeze takes inside a transform has to be the
    one the eager freeze takes on the same values -- with no tracer left in it, which
    ``np.asarray`` would refuse.
    """
    import dataclasses
    if dataclasses.is_dataclass(got):
        assert type(got) is type(want), path
        for field in dataclasses.fields(got):
            _assert_same_record(getattr(got, field.name), getattr(want, field.name),
                                f'{path}.{field.name}')
    elif isinstance(got, dict):
        assert isinstance(want, dict) and sorted(got, key=str) == sorted(want, key=str), path
        for key in got:
            _assert_same_record(got[key], want[key], f'{path}[{key!r}]')
    elif isinstance(got, (list, tuple)):
        assert isinstance(want, (list, tuple)) and len(got) == len(want), path
        for i, (a, b) in enumerate(zip(got, want)):
            _assert_same_record(a, b, f'{path}[{i}]')
    elif hasattr(got, 'shape') or isinstance(got, np.generic):
        a, b = np.asarray(got), np.asarray(want)
        assert (a.dtype, a.shape) == (b.dtype, b.shape) and a.tobytes() == b.tobytes(), path
    else:
        assert got == want, path


#: Every raster freeze that reads values, with the solver beside its raster: mu rasters too.
FREEZING = dict({name: (options, raster, {}) for name, (options, raster) in PLANNING.items()},
                **{'nvf-mu': (dict(factorization='nvf'), _rod, dict(mu=_mu_rod())),
                   'li-integrated-mu': (PLANNING['li-integrated'][0], _rod, dict(mu=_mu_rod()))})


@pytest.mark.parametrize('path', list(FREEZING))
def test_a_freeze_inside_jit_plans_a_constant_raster_as_it_does_eagerly(path):
    """Only the wavelength traced: the raster is a constant of the trace and has its values.

    The solve plans such a raster as eagerly since its layers are cut under ``concrete_scope``;
    the freeze cut them in the trace, so ``structure[i]`` was a tracer even of a constant and
    the integrated compression and the tensor labels raised JAX's own errors. The record has
    to be the eager one.
    """
    import jax
    options, raster, extra = FREEZING[path]
    solver = _solver(options, raster, **extra)
    eager, kept = solver.freeze_geometry(), []

    def traced(w):
        kept.append(solver.freeze_geometry(wavelength=w))
        return w
    jax.jit(traced)(620.)
    (inside,) = kept
    _assert_same_record(inside, eager)

