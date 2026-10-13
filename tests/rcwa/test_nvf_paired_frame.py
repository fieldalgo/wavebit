"""Physical conjugate-pair energy, exact limits and finite paired derivatives."""
from functools import lru_cache

import numpy as np
import pytest

from tests.rcwa.test_nvf_pair_normal_frame import GEOMETRY, MATERIALS, make_cell, reference, rectangle_transforms, gather

pytestmark = pytest.mark.research


def paired_reference(epsilon, gap, *, quadrature=16):
    """Integrate both normalized physical fields; never transpose a closure."""
    points, weights, material, field = reference(epsilon, gap, quadrature=quadrature, return_fields=True)
    modes = np.array([(m, n) for n in range(-2, 3) for m in range(-1, 2)])
    plane = np.exp(2j*np.pi*(points/np.diag(GEOMETRY['basis']))@modes.T)
    basis = np.concatenate([field[:, :, 0, None]*plane[:, None, :],
                            field[:, :, 1, None]*plane[:, None, :]], axis=2)
    compressed = np.vstack([plane.conj().T@(weights[:, None]*basis[:, component, :])
                            for component in range(2)])
    normalized = basis@np.linalg.inv(compressed)
    count = len(modes)
    reverse = np.array([np.where(np.all(modes == -k, axis=1))[0][0] for k in modes])
    partner = normalized[:, :, np.r_[reverse, count+reverse]].conj()
    def energy(f):
        return np.einsum('rci,r,rcj->ij', f.conj(), weights*material, f)
    # Both fields retain exactly the same physical electric coordinates.
    for f in (normalized, partner):
        low = np.vstack([plane.conj().T@(weights[:, None]*f[:, component, :]) for component in range(2)])
        np.testing.assert_allclose(low, np.eye(2*count), rtol=0, atol=3e-12)
    tensor = (energy(normalized)+energy(partner))/2
    direct = plane.conj().T@((weights*material)[:, None]*plane)
    return np.array([tensor[:count, :count], tensor[count:, count:], np.linalg.inv(direct),
                     -tensor[:count, count:], -tensor[count:, :count]])[:, None]


@pytest.mark.parametrize('name', MATERIALS)
@pytest.mark.parametrize('gap', [0., 7., 1e-9])
def test_paired_closure_against_two_independent_physical_field_energies(name, gap):
    cell = make_cell()
    e = MATERIALS[name]
    slots, meta = cell.paired_closure(gap=gap, epsilon=e)
    np.testing.assert_allclose(slots, paired_reference(e, gap), rtol=0, atol=3e-10)
    _, _, raw = cell.evaluate(gap=gap, epsilon=e)
    assert meta['raw']['reciprocity_error'] == raw['reciprocity_error']
    assert meta['raw']['inverse_residual'] == raw['inverse_residual']
    assert meta['reciprocity_error'] < 1e-12
    assert meta['minimum_rotated_storage'] >= raw['minimum_rotated_storage']-1e-12
    assert meta['minimum_dissipation'] >= raw['minimum_dissipation']-1e-12


@pytest.mark.parametrize('e', [4., 4.+.2j, -10.+1j])
def test_uniform_paired_medium_matches_exact_scalar_material(e):
    slots, meta = make_cell().paired_closure(gap=0., epsilon=(e, e), background=e)
    identity = np.eye(15)
    expected = np.array([e*identity, e*identity, identity/e, 0*identity, 0*identity])[:, None]
    np.testing.assert_allclose(slots, expected, rtol=0, atol=3e-13)
    assert meta['maximum_pairing_change'] < 3e-13


@pytest.mark.parametrize('epsilon', [(4., 9.), (12.25, -10.+1j)])
@pytest.mark.parametrize('gap', [0., 7.])
def test_full_height_pair_retains_analytic_inverse_and_direct_rules(epsilon, gap):
    order = (1, 1)
    cell = make_cell(height=560., order=order, quadrature=24)
    slots, _ = cell.paired_closure(gap=gap, epsilon=epsilon)
    shape, _ = rectangle_transforms(gap, order=order, height=560.)
    def direct(e):
        coeff = np.einsum('s,svu->vu', np.asarray(e)-1, shape)
        coeff[2, 2] += 1
        return gather(coeff, order)
    eps, rho = direct(epsilon), direct(1/np.asarray(epsilon))
    expected = np.array([np.linalg.inv(rho), eps, np.linalg.inv(eps), 0*eps, 0*eps])[:, None]
    np.testing.assert_allclose(slots, expected, rtol=0, atol=3e-10)


@pytest.fixture(scope='module')
def runtime():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    old = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    @lru_cache(None)
    def program(name):
        cell = make_cell(order=(1, 1))
        def value(gap):
            return jnp.stack(cell.paired_closure(gap=gap, epsilon=MATERIALS[name], xp=jnp)[0])
        return cell, value, jax.jit(lambda g: jax.jvp(value, (g,), (jnp.asarray(1.),)))
    yield jax, jnp, program
    jax.config.update('jax_enable_x64', old)


@pytest.mark.parametrize('name', MATERIALS)
@pytest.mark.parametrize('gap', [0., 7.])
def test_paired_gap_jvp_matches_two_resolved_windows(name, gap, runtime):
    _, jnp, program = runtime
    cell, _, derivative = program(name)
    actual, tangent = derivative(jnp.asarray(gap))
    def value(g):
        return np.array(cell.paired_closure(gap=g, epsilon=MATERIALS[name])[0])
    base = value(gap)
    np.testing.assert_allclose(actual, base, rtol=0, atol=3e-10)
    slopes = []
    for h in ([2e-5, 1e-5] if gap == 0 else [.002, .001]):
        slopes.append((-3*base+4*value(h)-value(2*h))/(2*h) if gap == 0
                      else (value(gap+h)-value(gap-h))/(2*h))
    np.testing.assert_allclose(slopes[0], slopes[1], rtol=3e-5, atol=2e-8)
    np.testing.assert_allclose(tangent, slopes[1], rtol=3e-5, atol=2e-8)


@pytest.mark.parametrize('gap', [0., 7.])
def test_complex_material_jvp_and_reverse_gap_derivative(gap, runtime):
    jax, jnp, program = runtime
    cell, value, derivative = program('metal')
    cotangent = np.arange(405).reshape(5, 1, 9, 9)/405
    reverse = jax.jit(jax.grad(lambda g: jnp.real(jnp.sum(value(g)*cotangent))))(jnp.asarray(gap))
    _, forward = derivative(jnp.asarray(gap))
    np.testing.assert_allclose(reverse, np.real(np.sum(forward*cotangent)), rtol=1e-10, atol=1e-10)
    base = np.array([1.+.2j, 4.+.8j, 9.+.3j])
    direction = np.array([.05+.02j, .1-.03j, -.2+.01j])
    def material_value(t, xp):
        e = xp.asarray(base)+t*xp.asarray(direction)
        return xp.stack(cell.paired_closure(gap=gap, epsilon=e[1:], background=e[0], xp=xp)[0])
    _, tangent = jax.jit(lambda t: jax.jvp(lambda x: material_value(x, jnp), (t,), (jnp.asarray(1.),)))(jnp.asarray(0.))
    slopes = [(material_value(h, np)-material_value(-h, np))/(2*h) for h in [2e-5, 1e-5]]
    np.testing.assert_allclose(slopes[0], slopes[1], rtol=2e-5, atol=2e-8)
    np.testing.assert_allclose(tangent, slopes[1], rtol=2e-5, atol=2e-8)


@pytest.mark.parametrize('epsilon', [(4., 9.), (12.25, -10.+1j)])
def test_laminated_contact_jvp_matches_analytic_moving_material_integrals(epsilon, runtime):
    jax, jnp, _ = runtime
    cell = make_cell(height=560., order=(1, 1), quadrature=24)
    def value(gap):
        return jnp.stack(cell.paired_closure(gap=gap, epsilon=epsilon, xp=jnp)[0])
    _, tangent = jax.jit(lambda g: jax.jvp(value, (g,), (jnp.asarray(1.),)))(jnp.asarray(0.))
    shape, m = rectangle_transforms(0., height=560.)
    dshape = shape*np.array([1, -1])[:, None, None]*(1j*np.pi*m/700.)
    def matrices(e):
        coeff = np.einsum('s,svu->vu', np.asarray(e)-1, shape)
        coeff[2, 2] += 1
        diff = np.einsum('s,svu->vu', np.asarray(e)-1, dshape)
        return gather(coeff), gather(diff)
    eps, de = matrices(epsilon)
    rho, dr = matrices(1/np.asarray(epsilon))
    ei, ri = np.linalg.inv(eps), np.linalg.inv(rho)
    expected = np.array([-ri@dr@ri, de, -ei@de@ei, 0*de, 0*de])[:, None]
    np.testing.assert_allclose(tangent, expected, rtol=0, atol=3e-10)


def test_a_failed_raw_frame_is_not_repaired_by_pairing(monkeypatch):
    cell = make_cell()
    coefficients = cell.coefficients
    def corrupted(**kwargs):
        value, meta = coefficients(**kwargs)
        value[4, 4, 2] -= 100j
        return value, meta
    monkeypatch.setattr(cell, 'coefficients', corrupted)
    with pytest.raises(ValueError, match='physical/inverse'):
        cell.paired_closure(gap=0., epsilon=(4., 4.))


def test_invalid_jit_paired_replay_poisons_all_slots(runtime):
    jax, jnp, _ = runtime
    cell = make_cell()
    def value(gap):
        slots, meta = cell.paired_closure(gap=gap, epsilon=MATERIALS['metal'], xp=jnp)
        return jnp.stack(slots), meta['valid'], meta['raw']['valid']
    slots, valid, raw_valid = jax.jit(value)(jnp.asarray(-1.))
    assert not valid and not raw_valid
    assert np.isnan(slots).all()
