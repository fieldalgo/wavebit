"""``wavebit.ellipsometry``: the pure functions, on hand-made Jones matrices, on every backend.

The solver-level checks are in ``tests/rcwa/test_ellipsometry.py``. Here the inputs are
random complex ``2 x 2`` matrices, so each identity is asserted on its own: the Mueller matrix
against the written-out ``A (J (x) J*) A^-1``, its independence of the time convention, the
depolarization index and Cloude spectrum of a pure matrix, the standard isotropic form, the
``[0, 2 pi)`` range of ``delta``, broadcasting over leading axes and differentiability.
"""
from contextlib import nullcontext

import numpy as np
import pytest

from wavebit.ellipsometry import cloude_eigenvalues, depolarization_index, mueller_from_jones, psi_delta

RNG = np.random.default_rng(7)
JONES = RNG.normal(size=(5, 3, 2, 2)) + 1j * RNG.normal(size=(5, 3, 2, 2))

A = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, -1j, 1j, 0]])
PAULI = [np.eye(2), np.diag([1., -1.]), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])]


def _namespace(backend):
    if backend == 'numpy':
        return np, nullcontext()
    module = pytest.importorskip(backend)
    if backend == 'jax':
        return module.numpy, module.enable_x64(True)
    return module, nullcontext()


def reference_mueller(jones_sp):
    """Azzam and Bashara's ``A (J (x) J*) A^-1`` on one matrix, ``J`` reordered to ``(p, s)``."""
    j = jones_sp[::-1, ::-1]
    return (A @ np.kron(j, j.conj()) @ np.linalg.inv(A)).real


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_mueller_from_jones_is_the_written_out_transform_on_every_backend(backend):
    xp, scope = _namespace(backend)
    with scope:
        mueller = np.asarray(mueller_from_jones(xp.asarray(JONES), normalize=False))
    assert mueller.shape == (5, 3, 4, 4)
    for index in np.ndindex(5, 3):
        np.testing.assert_allclose(mueller[index], reference_mueller(JONES[index]), rtol=0, atol=1e-13)


def test_the_mueller_matrix_does_not_depend_on_the_time_convention():
    """An ``exp(+i omega t)`` code holds ``conj(J)`` and reads ``S3`` with the other sign: same ``M``."""
    other_sign = A.copy()
    other_sign[3] *= -1
    for index in np.ndindex(5, 3):
        j = np.conj(JONES[index])[::-1, ::-1]
        theirs = (other_sign @ np.kron(j, j.conj()) @ np.linalg.inv(other_sign)).real
        np.testing.assert_allclose(mueller_from_jones(JONES[index], normalize=False), theirs, rtol=0, atol=1e-13)


def test_normalization_divides_by_m11_which_is_the_unpolarized_intensity():
    raw, normalized = mueller_from_jones(JONES, normalize=False), mueller_from_jones(JONES)
    np.testing.assert_allclose(normalized[..., 0, 0], 1.)
    np.testing.assert_allclose(raw[..., 0, 0], np.sum(np.abs(JONES) ** 2, axis=(-2, -1)) / 2, rtol=0, atol=1e-13)
    np.testing.assert_allclose(normalized * raw[..., :1, :1], raw, rtol=0, atol=1e-13)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_a_jones_derived_matrix_is_pure(backend):
    xp, scope = _namespace(backend)
    with scope:
        mueller = mueller_from_jones(xp.asarray(JONES))
        index = np.asarray(depolarization_index(mueller))
        eigenvalues = np.asarray(cloude_eigenvalues(mueller))
    np.testing.assert_allclose(index, 1., rtol=0, atol=1e-12)
    assert eigenvalues.shape == (5, 3, 4)
    np.testing.assert_allclose(eigenvalues[..., :3], 0., atol=1e-12)
    np.testing.assert_allclose(eigenvalues[..., 3], 1., rtol=0, atol=1e-12)


def test_an_intensity_average_of_pure_matrices_depolarizes_and_stays_physical():
    """Averaging two different pure matrices lowers the index below 1 but keeps every Cloude
    eigenvalue nonnegative; a made-up matrix outside the physical set has a negative one."""
    averaged = (mueller_from_jones(JONES[0, 0]) + mueller_from_jones(JONES[0, 1])) / 2
    assert 0 < depolarization_index(averaged) < 1
    assert cloude_eigenvalues(averaged).min() > -1e-14
    unphysical = np.diag([1., 1., 1., -1.])
    assert cloude_eigenvalues(unphysical).min() < -.1


def test_cloude_coherency_is_the_pauli_expansion():
    mueller = mueller_from_jones(JONES[1, 2])
    coherency = sum(mueller[i, j] * np.kron(PAULI[i], PAULI[j].conj()) for i in range(4) for j in range(4)) / 4
    np.testing.assert_allclose(cloude_eigenvalues(mueller), np.linalg.eigvalsh(coherency), rtol=0, atol=1e-13)


def test_a_diagonal_jones_matrix_gives_the_standard_form_with_psi_delta():
    r_s, r_p = -.3 - .4j, .1 + .25j
    jones = np.diag([r_s, r_p])
    angles = psi_delta(jones)
    rho = r_p / r_s
    np.testing.assert_allclose(angles.psi, np.arctan(abs(rho)))
    np.testing.assert_allclose(angles.delta, (-np.angle(rho)) % (2 * np.pi))
    np.testing.assert_allclose(angles.rho, rho)
    n, c, s = np.cos(2 * angles.psi), np.sin(2 * angles.psi) * np.cos(angles.delta), np.sin(2 * angles.psi) * np.sin(angles.delta)
    np.testing.assert_allclose(mueller_from_jones(jones), [[1, -n, 0, 0], [-n, 1, 0, 0], [0, 0, c, s], [0, 0, -s, c]], rtol=0, atol=1e-14)
    assert angles.psi_ps == 0 and angles.psi_sp == 0


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_psi_delta_reads_the_out_in_entries_and_wraps_delta_into_a_turn(backend):
    xp, scope = _namespace(backend)
    with scope:
        angles = psi_delta(xp.asarray(JONES))
        psi, delta = np.asarray(angles.psi), np.asarray(angles.delta)
        generalized = {name: np.asarray(getattr(angles, name)) for name in ('rho_ps', 'rho_sp', 'delta_ps', 'delta_sp', 'psi_ps', 'psi_sp')}
    j_ss, j_sp, j_ps, j_pp = JONES[..., 0, 0], JONES[..., 0, 1], JONES[..., 1, 0], JONES[..., 1, 1]
    np.testing.assert_allclose(psi, np.arctan2(np.abs(j_pp), np.abs(j_ss)))
    np.testing.assert_allclose(np.exp(1j * delta), np.conj(j_pp / j_ss) / np.abs(j_pp / j_ss), rtol=0, atol=1e-14)
    assert np.all(delta >= 0) and np.all(delta < 2 * np.pi)
    np.testing.assert_allclose(generalized['rho_ps'], j_sp / j_pp)
    np.testing.assert_allclose(generalized['rho_sp'], j_ps / j_ss)
    np.testing.assert_allclose(generalized['psi_sp'], np.arctan2(np.abs(j_ps), np.abs(j_ss)))
    np.testing.assert_allclose(np.exp(1j * generalized['delta_ps']), np.conj(j_sp / j_pp) / np.abs(j_sp / j_pp), rtol=0, atol=1e-14)


def test_delta_at_the_wrap_point_is_zero_not_two_pi():
    np.testing.assert_array_equal(psi_delta(np.diag([1., 1.])).delta, 0.)
    np.testing.assert_allclose(psi_delta(np.diag([1., -1.])).delta, np.pi)


def test_jax_differentiates_through_the_angles_and_the_mueller_matrix():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    with jax.enable_x64(True):
        def psi_of(x):
            return psi_delta(jnp.asarray(JONES[0, 0]) * jnp.array([[1., 1.], [1., x]])).psi
        def m34_of(x):
            return mueller_from_jones(jnp.asarray(JONES[0, 0]) * jnp.array([[1., 1.], [1., x]]))[2, 3]
        for f in (psi_of, m34_of):
            step = 1e-6
            finite = (float(f(1. + step)) - float(f(1. - step))) / (2 * step)
            np.testing.assert_allclose(float(jax.grad(f)(1.)), finite, rtol=1e-6)


def test_torch_differentiates_through_the_angles_and_the_mueller_matrix():
    torch = pytest.importorskip('torch')
    x = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    jones = torch.as_tensor(JONES[0, 0]) * torch.stack([torch.stack([x * 0 + 1, x * 0 + 1]), torch.stack([x * 0 + 1, x])])
    psi = psi_delta(jones).psi
    m34 = mueller_from_jones(jones)[2, 3]
    for value in (psi, m34):
        (derivative,) = torch.autograd.grad(value, x, retain_graph=True)
        def f(v):
            j = JONES[0, 0] * np.array([[1., 1.], [1., v]])
            return float(psi_delta(j).psi) if value is psi else float(mueller_from_jones(j)[2, 3])
        step = 1e-6
        np.testing.assert_allclose(float(derivative), (f(1. + step) - f(1. - step)) / (2 * step), rtol=1e-6)
