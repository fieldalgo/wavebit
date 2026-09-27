"""Metal contact cells against original nodal/Laurent and Duffy equations.

The half-plane certificate does not enter the physical equations. These tests
qualify the finite family and its gap tangent, not its optical convergence.
"""
from functools import lru_cache

import numpy as np
import pytest

from wavebit._core.conforming.cartesian_pair_opening import (
    CartesianPairOpeningCell, CartesianPairOpeningPlan,
)
from wavebit._core.conforming.cartesian_pair_gap import CartesianPairGapCell
from wavebit._core.conforming.cartesian_pair_gap_autodiff import CartesianPairGapEvaluator
from wavebit._core.conforming.material_sector import passive_pair_materials, phase_from_angle
from test_nvf_cartesian_pair_gap import (
    CASES, CRITERIA, settings_for, _reference,
    _oracle_differentiated_native_solve_reference,
)

pytestmark = pytest.mark.research


def scene(name="main_j0_r0"):
    case = CASES[name]
    settings = settings_for(CRITERIA, case)
    settings.update(epsilon=np.array([12.25, -10.+1j]), background=1.)
    plan = CartesianPairOpeningPlan.from_geometry(
        **{key: value for key, value in settings.items() if key not in ("epsilon", "background")},
        shift=case["shift"], refinement=case["refinement"],
    )
    return plan, tuple(case["order"]), settings


@lru_cache(None)
def independent(name, fraction):
    plan, order, settings = scene(name)
    cell = CartesianPairGapCell(plan, order, coercivity_angle=1.53)
    length = np.sqrt(np.linalg.det(settings["basis"]))
    gap = fraction * length
    q16, dq16, r16 = _reference(cell, settings, gap, 16)
    q24, dq24, r24 = _reference(cell, settings, gap, 24)
    np.testing.assert_allclose(q16, q24, rtol=0, atol=3e-10)
    np.testing.assert_allclose(length*dq16, length*dq24, rtol=0, atol=3e-9)
    assert max(*r16["residuals"], *r24["residuals"]) < 1e-10
    slots, dslots = _oracle_differentiated_native_solve_reference(q24, dq24, order)
    return cell, settings, length, gap, q24, dq24, slots, dslots


@pytest.mark.parametrize("name", ["main_j0_r0", "main_j0_r1", "main_j1_r1", "different_r1"])
@pytest.mark.parametrize("fraction", [0., 1/64, 1/16])
def test_metal_values_and_gap_derivatives_against_original_equations(name, fraction):
    cell, settings, length, gap, v, dv, slots, dslots = independent(name, fraction)
    actual, derivative, meta, _ = cell.coefficients(gap=gap, **settings)
    actual_slots, actual_dslots, cm, _ = cell.closure(gap=gap, **settings)
    np.testing.assert_allclose(actual, v, rtol=0, atol=3e-10)
    np.testing.assert_allclose(length*derivative, length*dv, rtol=0, atol=3e-9)
    np.testing.assert_allclose(actual_slots, slots, rtol=0, atol=3e-10)
    np.testing.assert_allclose(length*np.stack(actual_dslots), length*dslots, rtol=0, atol=3e-9)
    assert meta["contact_minimum_rotated_eigenvalue"] > 0
    assert meta["difference_minimum_rotated_eigenvalue"] > 0
    assert cm["minimum_rotated_storage"] > 0
    assert cm["minimum_dissipation"] > -1e-9
    value, evaluated, em, _ = CartesianPairGapEvaluator(
        cell.plan, cell.order, coercivity_angle=1.53
    ).evaluate(gap=gap, **settings)
    np.testing.assert_allclose(value, v, rtol=0, atol=3e-10)
    np.testing.assert_allclose(evaluated, slots, rtol=0, atol=3e-10)
    assert em["valid"]
    if fraction == 0:
        endpoint = CartesianPairOpeningCell(cell.plan, cell.order, coercivity_angle=1.53)
        ev, edv, _, _ = endpoint.coefficients(**settings)
        es, eds, _, _ = endpoint.closure(**settings)
        np.testing.assert_allclose(ev, v, rtol=0, atol=3e-10)
        np.testing.assert_allclose(length*edv, length*dv, rtol=0, atol=3e-9)
        np.testing.assert_allclose(es, slots, rtol=0, atol=3e-10)
        np.testing.assert_allclose(length*np.stack(eds), length*dslots, rtol=0, atol=3e-9)


@pytest.mark.parametrize("gap", [0., 7.])
@pytest.mark.parametrize("materials", [(12.25, -10.+1j), (4., 9.)])
def test_certificate_angle_cannot_change_the_physical_result(gap, materials):
    plan, order, settings = scene()
    settings["epsilon"] = materials
    before = None
    angles = (0., 1.5, 1.53, 1.56) if materials == (4., 9.) else (1.5, 1.53, 1.56)
    for angle in angles:
        cell = CartesianPairGapCell(plan, order, coercivity_angle=angle)
        value, tangent, _, detail = cell.coefficients(gap=gap, **settings)
        slots, dslots, _, _ = cell.closure(gap=gap, **settings)
        result = [value, tangent, np.stack(slots), np.stack(dslots), detail["M"], detail["f"]]
        if before is not None:
            for got, want in zip(result, before):
                np.testing.assert_array_equal(got, want)
        before = result


@pytest.mark.parametrize("gap", [0., 7.])
@pytest.mark.parametrize("epsilon", [-10.+1j, -4.])
def test_uniform_negative_real_material_has_exact_identity_field(gap, epsilon):
    plan, order, settings = scene()
    settings.update(epsilon=(epsilon, epsilon), background=epsilon)
    cell = CartesianPairGapCell(plan, order, coercivity_angle=np.pi)
    value, derivative, _, _ = cell.coefficients(gap=gap, **settings)
    expected = np.zeros_like(value)
    x, y = order
    expected[:, 2*y, 2*x] = [1, 0, 0, 1, epsilon, 0, epsilon, epsilon, 0]
    np.testing.assert_allclose(value, expected, rtol=0, atol=3e-12)
    np.testing.assert_allclose(derivative, 0, rtol=0, atol=3e-14)
    slots, dslots, metadata, _ = cell.closure(gap=gap, **settings)
    eye = np.eye((2*x+1)*(2*y+1))[None]
    for got, want in zip(slots, [epsilon*eye, epsilon*eye, eye/epsilon, 0*eye, 0*eye]):
        np.testing.assert_allclose(got, want, rtol=0, atol=3e-11)
    np.testing.assert_allclose(dslots, 0, rtol=0, atol=3e-13)
    assert metadata["minimum_storage"] < 0
    assert metadata["minimum_rotated_storage"] > 0


@pytest.mark.parametrize("gap", [0., 7.])
def test_energy_congruence_preserves_scalar_lower_bounds(gap):
    plan, order, settings = scene("main_j0_r1")
    settings.update(epsilon=(12.25+.2j, -10.+1j), background=1.+.01j)
    materials = np.r_[settings["background"], settings["epsilon"]]
    cell = CartesianPairGapCell(plan, order, coercivity_angle=1.53)
    _, _, metadata, _ = cell.closure(gap=gap, **settings)
    # Jensen/orthogonal projection: integral |B u|^2 >= |Fourier(B) u|^2.
    # After changing variables this yields H_theta >= min Re(e^-i theta eps) I.
    minimum = np.min(np.real(np.exp(-1.53j)*materials))
    assert metadata["minimum_rotated_storage"] >= minimum-1e-11
    assert metadata["minimum_dissipation"] >= np.min(materials.imag)-1e-11


@pytest.mark.parametrize("angle", [-.1, np.pi+.1, True, 1j, [1.53], np.nan, np.inf])
def test_bad_certificate_angles_are_rejected(angle):
    plan, order, _ = scene()
    for cls in (CartesianPairOpeningCell, CartesianPairGapCell, CartesianPairGapEvaluator):
        with pytest.raises(ValueError, match="coercivity_angle"):
            cls(plan, order, coercivity_angle=angle)


@pytest.mark.parametrize("angle,epsilon,background", [
    (0., (12.25, -10.+1j), 1.),
    (1., (12.25, -10.+1j), 1.),
    (np.pi/2, (1., -1.), 1.),
    (1.53, (12.25, -10.-1j), 1.),
    (1.53, (12.25, 0.), 1.),
    (1.53, (12.25, np.nan), 1.),
    (1.53, (12.25, -10.+1j), -1.),
])
def test_gain_zero_and_uncertified_materials_are_rejected(angle, epsilon, background):
    with pytest.raises(ValueError):
        passive_pair_materials(epsilon, background, phase_from_angle(angle))


def test_default_admission_preserves_tiny_positive_real_components():
    values, valid = passive_pair_materials((1e-100+1e300j, 4.), 1.)
    assert valid
    assert values[1] == 1e-100+1e300j


@pytest.fixture(scope="module")
def jax_runtime():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield jax, jnp
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("fraction", [0., 1/64])
def test_metal_jit_gap_tangent_uses_unrotated_original_reference(fraction, jax_runtime):
    jax, jnp = jax_runtime
    cell, settings, length, gap, v, dv, slots, dslots = independent("main_j0_r1", fraction)
    evaluator = CartesianPairGapEvaluator(cell.plan, cell.order, coercivity_angle=1.53)

    def values(g):
        value, matrices, _, _ = evaluator.evaluate(gap=g, **settings, xp=jnp)
        return value, jnp.stack(matrices)

    actual, derivative = jax.jit(lambda g: jax.jvp(values, (g,), (jnp.asarray(1.),)))(jnp.asarray(gap))
    for got, want in zip(actual, (v, slots)):
        np.testing.assert_allclose(got, want, rtol=0, atol=3e-10)
    for got, want in zip(derivative, (dv, dslots)):
        np.testing.assert_allclose(length*np.asarray(got), length*want, rtol=0, atol=3e-9)


def test_invalid_metal_certificate_is_nan_under_jit(jax_runtime):
    jax, jnp = jax_runtime
    plan, order, settings = scene()
    settings.pop("epsilon")
    cell = CartesianPairGapEvaluator(plan, order, coercivity_angle=1.53)

    @jax.jit
    def values(epsilon):
        value, slots, meta, _ = cell.evaluate(gap=0., epsilon=epsilon, **settings, xp=jnp)
        return value, jnp.stack(slots), meta["valid"]

    for epsilon in ((12.25, -10.-1j), (12.25, -10.+0j), (12.25, np.nan)):
        value, slots, valid = values(jnp.asarray(epsilon, dtype=jnp.complex128))
        assert not valid
        assert np.isnan(np.asarray(value)).all()
        assert np.isnan(np.asarray(slots)).all()
