"""A frozen NVF record carries a derivative with respect to the cell.

The raster normal field is a function of the cell, so its record carries the basis it was taken
on (scaled to ``|a1| == 1``, which leaves the cell's shape), and a replay compares that with the
cell in front of it. The comparison read the basis with ``np.asarray``, which a ``jax.grad``
tracer and a tensor under autograd both refuse; it reads their values now, so a record from
another cell shape is still refused and the one it was taken on replays.

What the replay differentiates is the question the rest of this file asks. It used to hold the
recorded field, so the field's own change with the cell dropped out of every cell derivative a
record was replayed under -- ``jax.jit`` always, and every ``rcwa.prepare`` / ``Prepared.run``.
On the rod here that was 106% of ``d/du_x`` under ``'distance'``. The record now keeps the
choices the field was built from (``normal_plans``), and a replay rebuilds the rest in the live
cell: its derivative is the **rebuilt** field's, and the oracle is a central difference of the
NumPy solve **without** a record, where every step rebuilds the field. The instrument is checked
against the held field's own difference first, so a replay that held the field would not pass.

``'contour'`` cannot carry it -- its curve moves with the cell through the blur that locates it
-- and is refused where the derivative is taken; a forward evaluation over a traced cell still
replays.
"""

import dataclasses
import functools

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._engine.geometry import StaleGeometryError
from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa

import numpy as np
import pytest

pytestmark = pytest.mark.shard_unit(per='test')

#: An 8 x 8 rod at truncation (2, 2), as in ``test_raster_constant_under_jit.py``.
TRUNCATION = (2, 2)
#: Columns ``u = (1000, 0)`` and ``v = (40, 900)``: a skewed cell, the case NVF serves. The
#: derivative is taken in its first row, ``(u_x, v_x)`` -- a stretch and a shear.
SKEW = np.array([[1000., 40.], [0., 900.]])
#: The same cell sheared further: another shape, which a record frozen on ``SKEW`` does not fit.
MOVED = np.array([[1000., 140.], [0., 900.]])
#: An axis-aligned cell whose pixels are not square, for the ``period`` derivative -- and the
#: one that sends ``'smooth'`` down its separable branch.
PERIOD = (1000., 600.)
KW = dict(wavelength=620., thickness=(300.,), top=1.0, bot=2.1025, theta=0.2, phi=0.3, psi=0.4,
          truncation_order=TRUNCATION, factorization='nvf')
#: Fixed weights, so the derivative is not of a conserved total.
_WEIGHTS = np.random.default_rng(7).uniform(0.5, 1.5, (2 * TRUNCATION[1] + 1,
                                                      2 * TRUNCATION[0] + 1))
#: The three constructions whose recorded field carries the derivative.
CARRIED = ('distance', 'spectral', 'smooth')
#: The step of every central difference here, and what it costs: 3e-7 of the derivative at
#: most across this file's cases, against a held field that is 5% to 106% away.
STEP = 1e-3
RTOL = 2e-6


def _rod():
    cell = np.ones((1, 8, 8))
    cell[0, 1:5, 2:7] = 12.25
    return cell


def _backend(name):
    pytest.importorskip(name)
    if name == 'jax':
        _legacy_wavebit_enable_x64()


def _solver(backend, method='distance', projector='linear', **cell):
    cell = cell or dict(lattice=SKEW)
    return _legacy_wavebit_rcwa(backend=backend, structure=_rod(), raster_normal_method=method,
                                nvf_projector=projector, **cell, **KW)


def _jax_loss(solver, geometry):
    import jax.numpy as jnp

    def loss(row):
        cell = jnp.stack([jnp.stack([row[0], row[1]]), jnp.asarray(SKEW[1])])
        return jnp.sum(_WEIGHTS * solver.solve(lattice=cell, geometry=geometry,
                                               auto_detune=False).T)
    return loss


def _derivative(name, solver, geometry, row):
    """``d loss / d (A[0, 0], A[0, 1])`` at the cell whose first row is ``row``, eagerly."""
    if name == 'jax':
        import jax
        import jax.numpy as jnp
        return np.asarray(jax.grad(_jax_loss(solver, geometry))(jnp.asarray(row)))
    import torch
    variable = torch.tensor(row, dtype=torch.float64, requires_grad=True)
    cell = torch.stack([variable, torch.as_tensor(SKEW[1])])
    result = solver.solve(lattice=cell, geometry=geometry, auto_detune=False)
    torch.sum(torch.as_tensor(_WEIGHTS) * result.T).backward()
    return variable.grad.numpy()


def _central_difference(solver, geometry):
    """Central differences of the NumPy solve in ``(A[0, 0], A[0, 1])`` about ``SKEW``."""
    def loss(row):
        cell = np.stack([row, SKEW[1]])
        return float(np.sum(_WEIGHTS * np.asarray(
            solver.solve(lattice=cell, geometry=geometry, auto_detune=False).T)))
    return np.array([(loss(SKEW[0] + STEP * e) - loss(SKEW[0] - STEP * e)) / (2 * STEP)
                     for e in np.eye(2)])


def _differs(held, rebuilt, least):
    """The instrument's own check: holding the field has to move the derivative."""
    share = np.max(np.abs(held - rebuilt) / np.abs(rebuilt))
    assert share > least, (held, rebuilt)


@functools.lru_cache(maxsize=None)
def _rebuilt_difference(method, projector):
    """The rebuilt field's central difference, checked against the held field's.

    The held difference runs through a record whose source was dropped, which replays unchecked
    and with a constant cell -- the field it was frozen with. Each construction's held share is
    5% or more of this derivative here (6-106% ``'distance'``, 8-17% ``'spectral'``, 17-87%
    ``'smooth'``, 22-96% under ``'circular'``).
    """
    solver = _solver('numpy', method, projector)
    rebuilt = _central_difference(solver, None)
    held = _central_difference(
        solver, dataclasses.replace(solver.freeze_geometry(), normal_field_source=None))
    _differs(held, rebuilt, 0.05)
    assert np.all(np.abs(rebuilt) > 1e-5), rebuilt       # a derivative that is there to compare
    return rebuilt


@pytest.mark.parametrize('projector', ['linear', 'circular'])
@pytest.mark.parametrize('method', CARRIED)
@pytest.mark.parametrize('name', ['jax', 'torch'])
def test_a_replayed_cell_derivative_is_the_rebuilt_field_s(name, method, projector):
    """The checked replay differentiates to the rebuilt field's central difference."""
    _backend(name)
    solver = _solver(name, method, projector)
    got = _derivative(name, solver, solver.freeze_geometry(), SKEW[0])
    np.testing.assert_allclose(got, _rebuilt_difference(method, projector), rtol=RTOL, atol=0)


@pytest.mark.parametrize('method', CARRIED)
def test_jit_vmap_and_a_grad_of_jit_take_the_eager_derivative(method):
    """``jax.jit`` and ``jax.vmap`` hold a basis with no value and replay without a check; the
    rebuild needs no value -- its choices are in the record -- so they carry the same derivative.
    ``jax.grad(jax.jit(f))`` differentiates a trace that saw no tangent, and gets it too."""
    _backend('jax')
    import jax
    import jax.numpy as jnp
    solver = _solver('jax', method)
    loss = _jax_loss(solver, solver.freeze_geometry())
    row = jnp.asarray(SKEW[0])
    eager = np.asarray(jax.grad(loss)(row))
    jitted = np.asarray(jax.jit(jax.grad(loss))(row))
    outside = np.asarray(jax.grad(jax.jit(loss))(row))
    mapped = np.asarray(jax.vmap(jax.grad(loss))(jnp.stack([row, row])))
    for other in (jitted, outside, mapped[0], mapped[1]):
        np.testing.assert_allclose(other, eager, rtol=1e-8, atol=0)


@pytest.mark.parametrize('name', ['jax', 'torch'])
def test_a_record_from_another_cell_shape_is_refused_under_the_derivative(name):
    """Reading the primal is a comparison, not a pass: the moved cell is refused as eagerly."""
    _backend(name)
    solver = _solver(name)
    frozen = solver.freeze_geometry()
    with pytest.raises(StaleGeometryError, match='cell basis is now'):
        solver.solve(lattice=MOVED, geometry=frozen, auto_detune=False)
    with pytest.raises(StaleGeometryError, match='cell basis is now'):
        _derivative(name, solver, frozen, MOVED[0])


def test_the_replayed_value_is_the_record_s():
    """Only the tangent is rebuilt: the value under a cell derivative is the replay's own.

    ``value_and_grad`` evaluates the replay with the rebuilt field's tangent riding on the
    recorded field; the recorded field itself is what the projector reads, so the value is the
    eager replay's to the last bits a different program can move.
    """
    _backend('jax')
    import jax
    import jax.numpy as jnp
    for method in CARRIED:
        solver = _solver('jax', method)
        geometry = solver.freeze_geometry()
        value, _ = jax.value_and_grad(_jax_loss(solver, geometry))(jnp.asarray(SKEW[0]))
        eager = np.sum(_WEIGHTS * np.asarray(solver.solve(lattice=SKEW, geometry=geometry,
                                                          auto_detune=False).T))
        assert abs(float(value) - float(eager)) <= 1e-14 * abs(float(eager)), method


# ------------------------------------------------------------------------ what cannot carry it


def test_contour_refuses_a_differentiated_cell_and_replays_a_forward_one():
    """``'contour'`` is refused where the derivative is taken, by every route to one.

    Under ``jax.grad(jax.jit(f))`` the body is traced with no tangent in sight, which is why the
    refusal sits in a differentiation rule rather than at a door. A forward evaluation over a
    traced cell takes no derivative, and replays the record as it always has.
    """
    _backend('jax')
    import jax
    import jax.numpy as jnp
    solver = _solver('jax', 'contour')
    loss = _jax_loss(solver, solver.freeze_geometry())
    row = jnp.asarray(SKEW[0])
    for route in (jax.grad(loss), jax.jit(jax.grad(loss)), jax.grad(jax.jit(loss))):
        with pytest.raises(TypeError, match="'contour' cannot differentiate the cell"):
            route(row)
    eager = np.sum(_WEIGHTS * np.asarray(solver.solve(geometry=solver.freeze_geometry(),
                                                      auto_detune=False).T))
    np.testing.assert_allclose(float(jax.jit(loss)(row)), eager, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.vmap(loss)(jnp.stack([row, row]))), [eager, eager],
                               rtol=1e-12)


def test_contour_refuses_a_cell_under_pytorch_autograd():
    _backend('torch')
    solver = _solver('torch', 'contour')
    with pytest.raises(TypeError, match="'contour' cannot differentiate the cell"):
        _derivative('torch', solver, solver.freeze_geometry(), SKEW[0])


@pytest.mark.parametrize('name', ['jax', 'torch'])
def test_a_record_without_its_choices_refuses_a_cell_derivative(name):
    """A record assembled without the plans cannot rebuild its field, so it refuses rather than
    hold it; a cell with no derivative still replays it."""
    _backend(name)
    solver = _solver(name)
    bare = dataclasses.replace(solver.freeze_geometry(), normal_field_plan=None)
    solver.solve(geometry=bare, auto_detune=False)
    with pytest.raises(TypeError, match='not the choices it was built from'):
        _derivative(name, solver, bare, SKEW[0])


# ------------------------------------------------------------------------- the period, and others


def _period_loss(name, solver, geometry):
    if name == 'jax':
        import jax.numpy as jnp
        return lambda p: jnp.sum(_WEIGHTS * solver.solve(period=[p[0], p[1]], geometry=geometry,
                                                         auto_detune=False).T)
    import torch
    return lambda p: torch.sum(torch.as_tensor(_WEIGHTS) * solver.solve(
        period=[p[0], p[1]], geometry=geometry, auto_detune=False).T)


def _period_derivative(name, solver, geometry):
    if name == 'jax':
        import jax
        import jax.numpy as jnp
        return np.asarray(jax.grad(_period_loss(name, solver, geometry))(jnp.asarray(PERIOD)))
    import torch
    p = torch.tensor(PERIOD, dtype=torch.float64, requires_grad=True)
    _period_loss(name, solver, geometry)(p).backward()
    return p.grad.numpy()


def _period_difference(solver, geometry):
    def loss(p):
        return float(np.sum(_WEIGHTS * np.asarray(solver.solve(period=list(p), geometry=geometry,
                                                               auto_detune=False).T)))
    return np.array([(loss(np.add(PERIOD, STEP * e)) - loss(np.subtract(PERIOD, STEP * e)))
                     / (2 * STEP) for e in np.eye(2)])


@functools.lru_cache(maxsize=None)
def _rebuilt_period_difference(method, coefficients='sampled'):
    """As :func:`_rebuilt_difference`, in ``period``. The held share is 5% (``'distance'``,
    ``'spectral'``) to 27% (``'smooth'``) of ``d/dLy`` on this cell."""
    solver = _solver('numpy', method, period=list(PERIOD), fourier_coefficients=coefficients)
    rebuilt = _period_difference(solver, None)
    held = _period_difference(
        solver, dataclasses.replace(solver.freeze_geometry(), normal_field_source=None))
    _differs(held, rebuilt, 0.02)
    return rebuilt


@pytest.mark.parametrize('method', CARRIED)
@pytest.mark.parametrize('name', ['jax', 'torch'])
def test_a_period_derivative_through_a_record_carries_the_field(name, method):
    """A ``period`` under differentiation reached the build as the recorded cell -- a constant
    -- so the field was held there too. It reaches it live now, on both backends."""
    _backend(name)
    solver = _solver(name, method, period=list(PERIOD))
    got = _period_derivative(name, solver, solver.freeze_geometry())
    np.testing.assert_allclose(got, _rebuilt_period_difference(method), rtol=RTOL, atol=0)


def test_a_moved_pattern_is_refused_under_a_jitted_period_derivative():
    """Under ``jax.jit`` the cell has no value, and only the cell is taken on trust.

    A traced ``period`` used to reach the record check as the record's own cell, so the pattern
    was compared there. The live cell it gets now has no value to compare, and the check still
    compares everything else: a raster whose pattern moved is refused, not replayed stale.
    """
    _backend('jax')
    import jax
    import jax.numpy as jnp
    solver = _solver('jax', period=list(PERIOD))
    geometry = solver.freeze_geometry()
    moved = _rod()
    moved[0, 5:7, 2:7] = 12.25

    def loss(p):
        return jnp.sum(_WEIGHTS * solver.solve(period=[p[0], p[1]], geometry=geometry,
                                               auto_detune=False).T)
    p = jnp.asarray(PERIOD)
    jax.jit(jax.grad(loss))(p)
    # Set outside the trace, so the raster is a constant of it and its layers have values.
    solver.structure = jnp.asarray(moved)
    with pytest.raises(StaleGeometryError, match='interfaces have moved'):
        jax.jit(jax.grad(loss))(p)


def test_the_integrated_coefficients_replay_carries_the_period_derivative():
    """``('integrated', 'nvf')`` replays the same record through its own builder."""
    _backend('jax')
    solver = _solver('jax', period=list(PERIOD), fourier_coefficients='integrated')
    got = _period_derivative('jax', solver, solver.freeze_geometry())
    np.testing.assert_allclose(got, _rebuilt_period_difference('distance', 'integrated'),
                               rtol=RTOL, atol=0)


def test_a_magnetic_raster_carries_the_cell_derivative_through_its_own_record():
    """The permeability pass replays its own record, with its own plans."""
    _backend('jax')
    mu = np.ones((1, 8, 8))
    mu[0, 4:7, 1:4] = 2.25

    def magnetic(backend):
        s = _solver(backend, 'spectral')
        s.mu = mu
        return s
    solver = magnetic('numpy')
    rebuilt = _central_difference(solver, None)
    frozen = solver.freeze_geometry()
    assert frozen.mu_normal_field_plan is not None
    held = _central_difference(solver, dataclasses.replace(
        frozen, normal_field_source=None, mu_normal_field_source=None))
    _differs(held, rebuilt, 0.05)
    solver = magnetic('jax')
    got = _derivative('jax', solver, solver.freeze_geometry(), SKEW[0])
    np.testing.assert_allclose(got, rebuilt, rtol=RTOL, atol=0)


# ----------------------------------------------------------------------------- without a record


@pytest.mark.parametrize('method', ['spectral', 'smooth'])
def test_spectral_and_smooth_carry_a_cell_derivative_without_a_record(method):
    """Without a record they used to refuse a traced cell; where the cell has a value they take
    their choices on it, as a freeze would, and carry the same derivative a replay does."""
    _backend('jax')
    solver = _solver('jax', method)
    got = _derivative('jax', solver, None, SKEW[0])
    np.testing.assert_allclose(got, _rebuilt_difference(method, 'linear'), rtol=RTOL, atol=0)


@pytest.mark.parametrize('method', ['spectral', 'smooth'])
def test_without_a_record_a_cell_with_no_value_is_sent_to_freeze_geometry(method):
    _backend('jax')
    import jax
    import jax.numpy as jnp
    solver = _solver('jax', method)
    with pytest.raises(TypeError, match='through a record taken with freeze_geometry'):
        jax.jit(jax.grad(_jax_loss(solver, None)))(jnp.asarray(SKEW[0]))


def test_the_circular_projector_differentiates_the_cell_without_a_record():
    """``|grad d|`` took a square root at its zeros, and ``max`` spread the NaN to every pixel:
    the derivative read ``[nan, nan]``."""
    _backend('jax')
    solver = _solver('jax', 'distance', 'circular')
    got = _derivative('jax', solver, None, SKEW[0])
    np.testing.assert_allclose(got, _rebuilt_difference('distance', 'circular'), rtol=RTOL,
                               atol=0)


# ------------------------------------------------------------------------- through rcwa.prepare


def _problem(cell):
    from wavebit import Material
    from wavebit import Units
    from wavebit import rcwa
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(lattice=cell, bottom=Material(epsilon=2.1025),
                         layers=(rcwa.Layer(thickness=300., epsilon=_rod()[0]),)),
        excitation=rcwa.Excitation(wavelength=620., sources=(rcwa.PlaneWave(),)))


def _efficiency(result):
    return result.transmission.order(0, 0).efficiency[0]


def _prepared_derivative(name, prepared, row):
    def efficiency(cell):
        return _efficiency(prepared.run(_problem(cell)))
    if name == 'jax':
        import jax
        import jax.numpy as jnp
        return np.asarray(jax.grad(
            lambda r: efficiency(jnp.stack([jnp.stack([r[0], r[1]]), jnp.asarray(SKEW[1])])))(
                jnp.asarray(row)))
    import torch
    variable = torch.tensor(row, dtype=torch.float64, requires_grad=True)
    efficiency(torch.stack([variable, torch.as_tensor(SKEW[1])])).backward()
    return variable.grad.numpy()


def test_a_prepared_run_differentiates_the_basis_it_was_prepared_on():
    """``Prepared.run`` replays its record, so the same derivative reaches the same rebuild.

    The oracle is ``rcwa.run``, which prepares afresh at every cell -- the public spelling of
    a solve with no record. A moved cell is still refused with the code the eager run gives it.
    """
    from wavebit import IncompatibleProblemError
    from wavebit import rcwa
    _backend('jax')
    _backend('torch')
    numerics = rcwa.Numerics(orders=TRUNCATION, factorization='nvf')

    def fresh(row):
        return float(_efficiency(rcwa.run(_problem(np.stack([row, SKEW[1]])), numerics=numerics)))
    want = np.array([(fresh(SKEW[0] + STEP * e) - fresh(SKEW[0] - STEP * e)) / (2 * STEP)
                     for e in np.eye(2)])
    assert np.all(np.abs(want) > 1e-6), want
    for name in ('jax', 'torch'):
        prepared = rcwa.prepare(_problem(SKEW), numerics=numerics, backend=name)
        got = _prepared_derivative(name, prepared, SKEW[0])
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=0, err_msg=name)
        with pytest.raises(IncompatibleProblemError) as eager:
            prepared.run(_problem(MOVED))
        with pytest.raises(IncompatibleProblemError) as differentiated:
            _prepared_derivative(name, prepared, MOVED[0])
        assert eager.value.code == differentiated.value.code == 'prepared.topology', name


@pytest.mark.parametrize('name', ['jax', 'torch'])
def test_a_prepared_contour_run_refuses_a_cell_derivative_by_its_public_code(name):
    from wavebit import ValidationError
    from wavebit import rcwa
    _backend(name)
    numerics = rcwa.Numerics(orders=TRUNCATION, factorization='nvf', raster_normal_method='contour')
    prepared = rcwa.prepare(_problem(SKEW), numerics=numerics, backend=name)
    prepared.run(_problem(SKEW))
    with pytest.raises(ValidationError, match="'contour' cannot differentiate the cell") as refused:
        _prepared_derivative(name, prepared, SKEW[0])
    assert refused.value.code == 'rcwa.invalid'
