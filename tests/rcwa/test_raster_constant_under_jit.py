"""A raster that is a constant of a ``jax.jit`` trace is solved as it is eagerly.

With only the wavelength, an angle or a thickness traced, the raster reaches the Fourier
builders as a concrete array, and every geometry decision they take from it can be read: the
NVF interface mask and normal field, a raster's block compression, the tensor path's phase
labels, and the passive-dielectric check that spares the inverse rule its singular values.
Inside the trace ``structure[i]`` was staged all the same -- JAX stages every operation there,
an index into a constant included -- so the layer it returned was a tracer with no value.
``factorization='nvf'`` then raised ``TracerBoolConversionError`` from the interface mask, the
``integrated`` coefficients raised it from the compression, and a tensor raster was refused
with "freeze_geometry() before tracing a tensor raster", although nothing about the geometry
was traced. ``jax.vmap`` was never affected: it stages only the values it batches.

The builders now cut their layers under :func:`wavebit._common.backend.concrete_scope` (the
tensor path under the solver's own ``_eager``), so the traced solve takes the eager solve's
branches and has to reproduce its numbers. A raster the caller traces still needs a frozen
geometry; ``tests/rcwa/test_frozen_geometry.py`` pins that side.
"""

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._engine.geometry import StaleGeometryError
from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa

import numpy as np
import pytest

pytestmark = pytest.mark.shard_unit(per='test')

#: An 8 x 8 raster at truncation (2, 2): below the 4N+1 = 9 floor the NVF grid needs, so the
#: lift the normal field takes is inside the cut too. Off-centre and on a non-square cell, so no
#: symmetry zeroes a derivative.
TRUNCATION = (2, 2)
BASE = dict(wavelength=620., period=(1000., 900.), thickness=(300.,), top=1.0, bot=2.1025,
            theta=0.2, phi=0.3, psi=0.4, truncation_order=TRUNCATION)
#: The point every test solves at: (wavelength, theta, thickness).
POINT = (620., 0.2, 300.)
TENSOR = np.array([[2.9 + .02j, .21 - .05j, .13 + .04j],
                   [.21 - .05j, 3.4 + .01j, -.19 - .09j],
                   [.13 + .04j, -.19 - .09j, 2.8]])
TENSOR = (TENSOR + TENSOR.conj().T) / 2


def _rod(value=12.25):
    cell = np.ones((1, 8, 8), dtype=np.asarray(value).dtype)
    cell[0, 1:5, 2:7] = value
    return cell


def _tensor_rod():
    cell = np.broadcast_to(np.eye(3, dtype=complex), (1, 8, 8, 3, 3)).copy()
    cell[0, 1:5, 2:7] = TENSOR
    return cell


def _mu_rod():
    mu = np.ones((1, 8, 8))
    mu[0, 1:5, 2:7] = 1.7
    return mu


#: Every raster path that reads geometry off the layer. ('sampled', 'li') and
#: ('sampled', 'laurent') read none and traced before the fix as well.
PATHS = {
    'nvf': dict(structure=_rod(), factorization='nvf'),
    'nvf-integrated': dict(structure=_rod(), factorization='nvf',
                           fourier_coefficients='integrated'),
    'nvf-contour': dict(structure=_rod(), factorization='nvf', raster_normal_method='contour'),
    'nvf-circular': dict(structure=_rod(), factorization='nvf', nvf_projector='circular'),
    'li-integrated': dict(structure=_rod(), factorization='li',
                          fourier_coefficients='integrated'),
    'laurent-integrated': dict(structure=_rod(), factorization='laurent',
                               fourier_coefficients='integrated'),
    'nvf-mu': dict(structure=_rod(), mu=_mu_rod(), factorization='nvf'),
    'nvf-metal': dict(structure=_rod(-18.3 + 1.1j), factorization='nvf'),
    'nvf-tensor': dict(structure=_tensor_rod(), factorization='nvf'),
}

#: Fixed weights, so the derivative is not of a conserved total: on a lossless cell
#: sum(R) + sum(T) = 1 and its gradient is zero whatever the solve does.
_WEIGHTS = np.random.default_rng(7).uniform(0.5, 1.5, (2 * TRUNCATION[1] + 1,
                                                      2 * TRUNCATION[0] + 1))


@pytest.fixture(autouse=True)
def _jax():
    pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()


def _solver(path):
    return _legacy_wavebit_rcwa(backend='jax', **BASE, **PATHS[path])


def _observables(solver, wavelength, theta, thickness):
    import jax.numpy as jnp
    result = solver.solve(wavelength=wavelength, theta=theta,
                          thickness=jnp.stack([jnp.asarray(thickness)]), auto_detune=False)
    return result.R, result.T, result.r_xyz, result.t_xyz


def _loss(solver):
    import jax.numpy as jnp

    def loss(wavelength, theta, thickness):
        R, T, _, _ = _observables(solver, wavelength, theta, thickness)
        return jnp.sum(_WEIGHTS * T) + 0.5 * jnp.sum(_WEIGHTS * R)
    return loss


@pytest.mark.parametrize('path', list(PATHS))
def test_a_constant_raster_jits_to_the_eager_answer(path):
    """R, T and the outgoing Cartesian amplitudes, traced in wavelength, theta and thickness."""
    import jax
    solver = _solver(path)
    eager = [np.asarray(v) for v in _observables(solver, *POINT)]
    traced = jax.jit(lambda *a: _observables(solver, *a))(*POINT)
    for name, want, got in zip(('R', 'T', 'r_xyz', 't_xyz'), eager, traced):
        np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=1e-13, err_msg=name)


@pytest.mark.parametrize('path', list(PATHS))
def test_a_constant_raster_differentiates_under_jit_as_it_does_eagerly(path):
    """``jit(grad)`` in all three traced inputs against eager ``jax.grad`` at the same point."""
    import jax
    loss = _loss(_solver(path))
    eager = np.asarray(jax.grad(loss, argnums=(0, 1, 2))(*POINT))
    traced = np.asarray(jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(*POINT))
    assert np.all(np.abs(eager) > 1e-6), eager      # a derivative that is there to compare
    np.testing.assert_allclose(traced, eager, rtol=1e-10, atol=0)


@pytest.mark.parametrize('path', ['nvf', 'nvf-integrated', 'nvf-tensor'])
def test_a_constant_raster_vmaps_inside_jit(path):
    """``jit(vmap)`` over a wavelength batch: the trace is jit's, and the batch rides inside it."""
    import jax
    import jax.numpy as jnp
    solver = _solver(path)
    wavelengths = (600., 620., 655.)
    eager = np.stack([np.asarray(_observables(solver, w, *POINT[1:])[1]) for w in wavelengths])
    traced = jax.jit(jax.vmap(lambda w: _observables(solver, w, *POINT[1:])[1]))(
        jnp.asarray(wavelengths))
    np.testing.assert_allclose(np.asarray(traced), eager, rtol=0, atol=1e-13)


def test_the_traced_solve_takes_the_eager_branch_of_the_inverse_rule_floor(monkeypatch):
    """A passive dielectric layer skips the floor's SVD eagerly, and now under ``jax.jit`` too.

    The check reads the layer's values on the host. A staged layer has none, so even a
    dielectric kept the floor and paid a full singular-value decomposition per layer -- the
    branch for a value that cannot be read, taken for one that could.
    """
    import jax
    from wavebit.rcwa._engine import factorization as F
    calls = []
    floor = F._floor_singular_values
    monkeypatch.setattr(F, '_floor_singular_values',
                        lambda *a, **k: calls.append(1) or floor(*a, **k))
    for path in ('nvf', 'nvf-integrated'):
        solver = _solver(path)
        _observables(solver, *POINT)
        jax.jit(lambda *a: _observables(solver, *a))(*POINT)
    assert calls == []


@pytest.mark.parametrize('path', ['nvf', 'nvf-integrated', 'li-integrated'])
def test_a_stale_record_is_refused_under_jit_when_the_raster_is_a_constant(path):
    """The record checks read the layer, so a constant raster is checked under ``jax.jit`` too.

    A frozen record replayed on a raster whose pattern moved is refused eagerly. Under a trace
    it was replayed on trust whatever the raster was, and answered for the old pattern; only a
    raster the trace carries has nothing to compare (``test_a_traced_replay_carries_no_check``).
    """
    import jax
    frozen = _solver(path).freeze_geometry()
    moved = np.ones((1, 8, 8))
    moved[0, 3:7, 0:4] = 12.25
    solver = _legacy_wavebit_rcwa(backend='jax', **BASE, **dict(PATHS[path], structure=moved))
    with pytest.raises(StaleGeometryError, match='no longer fits'):
        solver.solve(geometry=frozen, auto_detune=False)
    with pytest.raises(StaleGeometryError, match='no longer fits'):
        jax.jit(lambda w: solver.solve(wavelength=w, geometry=frozen, auto_detune=False).T)(620.)


# ----------------------------------------------------------------- a traced cell beside it

#: A skewed cell, for the derivative in the basis itself; NVF is the rule that serves one.
SKEW = np.array([[1000., 40.], [0., 900.]])


def _skewed_basis(p):
    import jax.numpy as jnp
    return jnp.stack([jnp.stack([p[0], p[1]]), jnp.asarray([0., 900.])])


def _cell_loss(solver, cell, geometry=None):
    import jax.numpy as jnp

    def loss(p):
        keyword = (dict(period=jnp.stack([p, jnp.asarray(900.)])) if cell == 'period'
                   else dict(lattice=_skewed_basis(p)))
        result = solver.solve(**keyword, geometry=geometry, auto_detune=False)
        return jnp.sum(_WEIGHTS * result.T)
    return loss


def _cell_solver(cell):
    kw = dict(BASE, **PATHS['nvf'])
    if cell == 'lattice':
        kw.pop('period')
        kw['lattice'] = SKEW
    return _legacy_wavebit_rcwa(backend='jax', **kw)


@pytest.mark.parametrize('cell', ['period', 'lattice'])
def test_a_traced_cell_beside_a_constant_raster_differentiates_as_it_does_eagerly(cell):
    """No record: the normal field is built in the traced cell and carries its derivative.

    Before the fix this raised from the interface mask like every other jitted NVF solve; the
    eager derivative it now reproduces is the one the cell metric gives the field
    (``_geometry_lattice_for``), not the frozen-record one.
    """
    import jax
    import jax.numpy as jnp
    loss = _cell_loss(_cell_solver(cell), cell)
    point = jnp.asarray(1000.) if cell == 'period' else jnp.asarray(SKEW[0])
    eager = np.asarray(jax.grad(loss)(point))
    traced = np.asarray(jax.jit(jax.grad(loss))(point))
    assert np.all(np.abs(eager) > 1e-7), eager
    np.testing.assert_allclose(traced, eager, rtol=1e-10, atol=0)


def test_a_record_on_a_traced_skewed_cell_is_replayed_without_reading_the_cell():
    """The record checks now run under ``jax.jit``, and a basis the trace holds has no value.

    The cell is an input of the normal field, so the check compares it with the record's --
    which cannot be done on a ``jax.jit`` tracer. The solver's own pre-check skips the
    comparison there, and so does the builder's: until the constant raster's layers were cut
    concrete the builder never got that far under a trace, and this jitted derivative ran.
    Reading the cell instead raised ``TracerArrayConversionError``.
    """
    import jax
    solver = _cell_solver('lattice')
    loss = _cell_loss(solver, 'lattice', geometry=solver.freeze_geometry())
    traced = np.asarray(jax.jit(jax.grad(loss))(np.asarray(SKEW[0])))
    assert np.all(np.isfinite(traced)) and np.all(np.abs(traced) > 1e-7), traced


def test_a_held_preparation_refuses_a_moved_constant_raster_under_jit_as_it_does_eagerly():
    """``rcwa.prepare`` / ``Prepared.run``: the builder's check reaches the public code.

    Eagerly the run refuses a raster whose pattern moved with ``prepared.topology``. Under
    ``jax.jit`` with the pattern a constant of the trace, the sampled NVF build used to replay
    the record and return NaN; it now refuses with the eager run's error. A pattern the trace
    carries still has nothing to compare and is answered NaN with ``raster_partition`` false.
    """
    import jax
    import jax.numpy as jnp
    from wavebit import IncompatibleProblemError
    from wavebit import Material
    from wavebit import Units
    from wavebit import rcwa

    def problem(cell, wavelength=620.):
        return rcwa.Problem(
            units=Units(length='nm'),
            stack=rcwa.Stack(period=(1000., 900.), bottom=Material(epsilon=2.1025),
                             layers=(rcwa.Layer(thickness=300., epsilon=cell),)),
            excitation=rcwa.Excitation(wavelength=wavelength, sources=(rcwa.PlaneWave(),)))

    held, moved = _rod()[0], np.ones((8, 8))
    moved[3:7, 0:5] = 12.25
    prepared = rcwa.prepare(problem(held), numerics=rcwa.Numerics(orders=TRUNCATION,
                                                                  factorization='nvf'),
                            backend='jax')

    def run(cell, wavelength):
        return prepared.run(problem(cell, wavelength)).transmission.order(0, 0).efficiency[0]

    with pytest.raises(IncompatibleProblemError) as eager:
        run(moved, 620.)
    with pytest.raises(IncompatibleProblemError) as traced:
        jax.jit(lambda w: run(moved, w))(620.)
    assert eager.value.code == traced.value.code == 'prepared.topology'
    assert np.isnan(float(jax.jit(lambda c: run(c, 620.))(jnp.asarray(moved))))
