"""A frozen NVF record replays under an eager derivative in the cell basis.

The normal field is a function of the cell, so its record carries the basis it was taken on
(scaled to ``|a1| == 1``, which leaves the cell's shape), and a replay compares that with the
cell in front of it. The comparison read the basis with ``np.asarray``, which a ``jax.grad``
tracer and a tensor under autograd both refuse. So an eager derivative in a skewed basis that
replayed a record raised ``TracerArrayConversionError`` on JAX and "Can't call numpy() on Tensor
that requires grad" on PyTorch -- and so did the same derivative through ``rcwa.prepare`` and
``Prepared.run``, which replay the record they prepared. Both values carry a number: the grad
tracer its primal, the tensor once detached. The check now reads that number and compares it, so
a record from another cell shape is still refused.

What the replay differentiates is the solve with the recorded field *held*: the field is read
from the record, not rebuilt, so its own dependence on the cell is not part of the derivative.
That is the derivative a ``jax.jit`` trace has always taken here -- a basis there has no value,
so the record is replayed on trust -- and the oracle below takes it by central differences on
NumPy, through a record whose source was dropped (a record without one replays unchecked, on any
cell). On this rod the field-held derivative in ``u_x`` is about twice the rebuilt one, and the
oracle asserts that it is, so a replay that rebuilt the field instead would not pass.
"""

import dataclasses

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
KW = dict(wavelength=620., thickness=(300.,), top=1.0, bot=2.1025, theta=0.2, phi=0.3, psi=0.4,
          truncation_order=TRUNCATION, factorization='nvf')
#: Fixed weights, so the derivative is not of a conserved total.
_WEIGHTS = np.random.default_rng(7).uniform(0.5, 1.5, (2 * TRUNCATION[1] + 1,
                                                      2 * TRUNCATION[0] + 1))


def _rod():
    cell = np.ones((1, 8, 8))
    cell[0, 1:5, 2:7] = 12.25
    return cell


def _backend(name):
    pytest.importorskip(name)
    if name == 'jax':
        _legacy_wavebit_enable_x64()


def _solver(backend):
    return _legacy_wavebit_rcwa(backend=backend, lattice=SKEW, structure=_rod(), **KW)


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


def _central_difference(solver, geometry, h=1e-2):
    """Central differences of the NumPy solve in ``(A[0, 0], A[0, 1])`` about ``SKEW``."""
    def loss(row):
        cell = np.stack([row, SKEW[1]])
        return float(np.sum(_WEIGHTS * np.asarray(
            solver.solve(lattice=cell, geometry=geometry, auto_detune=False).T)))
    return np.array([(loss(SKEW[0] + h * e) - loss(SKEW[0] - h * e)) / (2 * h)
                     for e in np.eye(2)])


def _field_held_difference():
    """The central difference with the normal field held at the one frozen on ``SKEW``.

    The instrument is checked before it is used. It runs through the same builder as the
    derivative it judges, so a replay that rebuilt the field would move both alike and still
    agree; it is told apart by the rebuilt field's own difference, which is half of it in
    ``u_x`` (1.43e-4 against 2.96e-4).
    """
    solver = _solver('numpy')
    held = _central_difference(
        solver, dataclasses.replace(solver.freeze_geometry(), normal_field_source=None))
    rebuilt = _central_difference(solver, None)
    assert abs(held[0] - rebuilt[0]) > 0.3 * abs(held[0]), (held, rebuilt)
    return held


@pytest.mark.parametrize('name', ['jax', 'torch'])
def test_an_eager_derivative_in_the_basis_replays_its_record(name):
    """The checked replay differentiates to the field-held central difference.

    The step's truncation error is 1.6e-7 of the derivative at ``h = 1e-2`` (2e-8 at 1e-3).
    """
    _backend(name)
    solver = _solver(name)
    got = _derivative(name, solver, solver.freeze_geometry(), SKEW[0])
    want = _field_held_difference()
    assert np.all(np.abs(want) > 1e-5), want            # a derivative that is there to compare
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=0)


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


def test_the_eager_derivative_is_the_one_a_basis_with_no_value_replays_on_trust():
    """``jax.jit`` and ``jax.vmap`` hold a basis with no value and still replay without a check.

    The eager derivative now runs the check that those two cannot, and differentiates the same
    computation: the record's field on the live cell.
    """
    _backend('jax')
    import jax
    import jax.numpy as jnp
    solver = _solver('jax')
    loss = _jax_loss(solver, solver.freeze_geometry())
    row = jnp.asarray(SKEW[0])
    eager = np.asarray(jax.grad(loss)(row))
    jitted = np.asarray(jax.jit(jax.grad(loss))(row))
    mapped = np.asarray(jax.vmap(jax.grad(loss))(jnp.stack([row, row])))
    np.testing.assert_allclose(jitted, eager, rtol=1e-10, atol=0)
    np.testing.assert_allclose(mapped, np.stack([eager, eager]), rtol=1e-10, atol=0)


# ------------------------------------------------------------------- through rcwa.prepare


def _problem(cell):
    from wavebit import Material
    from wavebit import Units
    from wavebit import rcwa
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(lattice=cell, bottom=Material(epsilon=2.1025),
                         layers=(rcwa.Layer(thickness=300., epsilon=_rod()[0]),)),
        excitation=rcwa.Excitation(wavelength=620., sources=(rcwa.PlaneWave(),)))


def _prepared_derivative(name, prepared, row):
    def efficiency(cell):
        return prepared.run(_problem(cell)).transmission.order(0, 0).efficiency[0]
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
    """``Prepared.run`` replays its record, so the same derivative reached the same check.

    JAX raised ``rcwa.invalid`` from inside the solve and PyTorch ``prepared.topology`` from the
    run's own pre-check; both now differentiate, to one answer, and a moved cell is still refused
    with the code the eager run gives it.
    """
    from wavebit import IncompatibleProblemError
    from wavebit import rcwa
    _backend('jax')
    _backend('torch')
    numerics = rcwa.Numerics(orders=TRUNCATION, factorization='nvf')
    derivatives = {}
    for name in ('jax', 'torch'):
        prepared = rcwa.prepare(_problem(SKEW), numerics=numerics, backend=name)
        derivatives[name] = _prepared_derivative(name, prepared, SKEW[0])
        with pytest.raises(IncompatibleProblemError) as eager:
            prepared.run(_problem(MOVED))
        with pytest.raises(IncompatibleProblemError) as differentiated:
            _prepared_derivative(name, prepared, MOVED[0])
        assert eager.value.code == differentiated.value.code == 'prepared.topology', name
    assert np.all(np.abs(derivatives['jax']) > 1e-5), derivatives
    np.testing.assert_allclose(derivatives['torch'], derivatives['jax'], rtol=1e-10, atol=0)
