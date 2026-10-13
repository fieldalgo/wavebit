"""Refined finite pair cells against original equations and simplex integrals.

The old absolute guards are retained. These tests validate finite assembly,
topology and gap tangents, not optical convergence of the enlarged spaces.
"""
from functools import lru_cache

import numpy as np
import pytest

from wavebit.rcwa._engine.conforming.cartesian_pair import CartesianPairCell, CartesianPairPlan
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningCell, CartesianPairOpeningPlan
from wavebit.rcwa._engine.conforming.cartesian_pair_gap import CartesianPairGapCell
from wavebit.rcwa._engine.conforming.cartesian_pair_gap_autodiff import CartesianPairGapEvaluator
from tests.rcwa.test_nvf_cartesian_pair_gap import _reference, _oracle_differentiated_native_solve_reference
from tests.rcwa.test_nvf_cartesian_pair import oracle as native_oracle, settings as native_settings


CASES = {
    "square-r2": (2, "square", False, False),
    "skew-metal-r2": (2, "skew", True, False),
    "signed-metal-r2": (2, "skew", True, True),
    "square-metal-r3": (3, "square", True, False),
    "skew-r3": (3, "skew", False, False),
    "signed-metal-r3": (3, "skew", True, True),
}


def scene(name):
    refinement, geometry, metal, signed = CASES[name]
    if geometry == "square":
        settings = dict(basis=np.diag([700., 700.]), widths=(300., 300.),
                        height=300., offset=(0., 0.))
    else:
        settings = dict(basis=np.array([[700., 315.], [0., 490.]]), widths=(140., 193.),
                        height=210., offset=(49., -21.))
    if signed:
        settings["basis"] = settings["basis"] @ np.array([[0., -1.], [1., 0.]])
    settings.update(epsilon=(12.25, -10.+1j) if metal else (4.+.8j, 9.+.3j),
                    background=1. if metal else 1.+.1j)
    order = (2, 1) if signed else (1, 2)
    geometry = {key: value for key, value in settings.items() if key not in ("epsilon", "background")}
    plan = CartesianPairOpeningPlan.from_geometry(**geometry, refinement=refinement)
    return plan, order, settings, 1.53 if metal else 0.


@lru_cache(None)
def independently_qualified(name, gap):
    plan, order, settings, angle = scene(name)
    cell = CartesianPairGapCell(plan, order, coercivity_angle=angle)
    length = np.sqrt(np.linalg.det(settings["basis"]))
    v16, d16, m16 = _reference(cell, settings, gap, 16)
    v24, d24, m24 = _reference(cell, settings, gap, 24)
    np.testing.assert_allclose(v16, v24, rtol=0, atol=3e-10)
    np.testing.assert_allclose(length*d16, length*d24, rtol=0, atol=3e-9)
    assert max(*m16["residuals"], *m24["residuals"]) < 1e-10
    slots, dslots = _oracle_differentiated_native_solve_reference(v24, d24, order)
    return cell, settings, length, v24, d24, slots, dslots


@pytest.mark.research
@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("gap", [0., 7.])
def test_refined_cells_against_original_nodal_and_laurent_equations(name, gap):
    cell, settings, length, value, derivative, slots, dslots = independently_qualified(name, gap)
    actual, tangent, meta, _ = cell.coefficients(gap=gap, **settings)
    actual_slots, actual_dslots, cm, _ = cell.closure(gap=gap, **settings)
    np.testing.assert_allclose(actual, value, rtol=0, atol=3e-10)
    np.testing.assert_allclose(length*tangent, length*derivative, rtol=0, atol=3e-9)
    np.testing.assert_allclose(actual_slots, slots, rtol=0, atol=3e-10)
    np.testing.assert_allclose(length*np.stack(actual_dslots), length*dslots, rtol=0, atol=3e-9)
    assert meta["original_flux_residual"] < 1e-10
    assert meta["original_flux_derivative_residual"] < 1e-10
    assert cm["minimum_dissipation"] > -1e-9
    evaluator = CartesianPairGapEvaluator(cell.plan, cell.order, coercivity_angle=cell.coercivity_angle)
    actual, actual_slots, em, _ = evaluator.evaluate(gap=gap, **settings)
    np.testing.assert_allclose(actual, value, rtol=0, atol=3e-10)
    np.testing.assert_allclose(actual_slots, slots, rtol=0, atol=3e-10)
    assert em["valid"]
    if gap == 0:
        endpoint = CartesianPairOpeningCell(cell.plan, cell.order, coercivity_angle=cell.coercivity_angle)
        ev, ed, _, _ = endpoint.coefficients(**settings)
        np.testing.assert_allclose(ev, value, rtol=0, atol=3e-10)
        np.testing.assert_allclose(length*ed, length*derivative, rtol=0, atol=3e-9)
    if cell.coercivity_angle == 0:
        # The ordinary finite adapter shares the enlarged plan admission too.
        native = CartesianPairCell(CartesianPairPlan(gap == 0, cell.plan.shift, cell.plan.refinement), cell.order)
        nv, _ = native.coefficients(gap=gap, **settings)
        np.testing.assert_allclose(nv, value, rtol=0, atol=3e-10)


@pytest.mark.research
@pytest.mark.parametrize("refinement", [2, 3])
@pytest.mark.parametrize("shift", range(4))
def test_refined_topology_has_exact_collapse_and_expected_counts(refinement, shift):
    cell = CartesianPairGapCell(CartesianPairOpeningPlan(shift, refinement))
    data = cell.data
    assert len(data["triangles"]) == 18*4**refinement
    assert len(data["contact_triangles"]) == 14*4**refinement
    assert data["C"].shape[1] == 7*4**refinement-1
    assert data["Z"].shape[1] == 2*4**refinement
    assert data["thin"].sum() == 4*4**refinement
    assert len(data["pmap"])-1 == 9*4**refinement-1
    # Signed area, tag preservation, connected pole graph and exact dyadic
    # mapping are checked by production topology before returning this data.
    for contact in (False, True):
        weights, triangles, tags, periodic = CartesianPairPlan(contact, shift, refinement).topology()
        np.testing.assert_array_equal(weights.sum(axis=1), np.ones(len(weights)))
        np.testing.assert_array_equal(weights*2**refinement, np.round(weights*2**refinement))
        assert all(not value.flags.writeable for value in (weights, triangles, tags, periodic))


@pytest.mark.parametrize("refinement", [-1, 4, 1.5, True])
def test_refinement_admission_remains_bounded(refinement):
    with pytest.raises(ValueError):
        CartesianPairOpeningPlan(0, refinement)


@pytest.fixture(scope="module")
def jax_runtime():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)

    @lru_cache(None)
    def program(name):
        plan, order, settings, angle = scene(name)
        evaluator = CartesianPairGapEvaluator(plan, order, coercivity_angle=angle)
        def values(gap):
            value, slots, _, _ = evaluator.evaluate(gap=gap, **settings, xp=jnp)
            return value, jnp.stack(slots)
        def tangent(gap):
            return jax.jvp(values, (gap,), (jnp.asarray(1.),))
        return evaluator, tangent, jax.jit(tangent)

    yield jax, jnp, program
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.research
@pytest.mark.parametrize("name", ["skew-metal-r2", "square-metal-r3"])
@pytest.mark.parametrize("gap", [0., 7.])
def test_refined_jit_gap_tangents_against_original_equations(name, gap, jax_runtime):
    _, jnp, program = jax_runtime
    cell, settings, length, value, derivative, slots, dslots = independently_qualified(name, gap)
    actual, tangent = program(name)[2](jnp.asarray(gap))
    for got, want in zip(actual, (value, slots)):
        np.testing.assert_allclose(got, want, rtol=0, atol=3e-10)
    for got, want in zip(tangent, (derivative, dslots)):
        np.testing.assert_allclose(length*np.asarray(got), length*want, rtol=0, atol=3e-9)


@pytest.mark.research
def test_gap_jvp_never_materializes_element_by_unknown_squared_arrays(jax_runtime):
    jax, jnp, program = jax_runtime
    evaluator, tangent, _ = program("square-metal-r3")
    traced = jax.make_jaxpr(tangent)(jnp.asarray(0.))
    e = len(evaluator.data["triangles"])
    n = len(evaluator.data["pmap"])
    # This is a resource contract for the actual value+tangent graph, including
    # nested primitives. The largest permissible tensor is the nodal restriction.
    limit = max(3*e*n, n*n)
    def visit(graph):
        graph = getattr(graph, "jaxpr", graph)
        for eqn in graph.eqns:
            for variable in eqn.outvars:
                shape = getattr(getattr(variable, "aval", None), "shape", ())
                assert np.prod(shape, dtype=int) <= limit, shape
            for parameter in eqn.params.values():
                for child in parameter if isinstance(parameter, (tuple, list)) else (parameter,):
                    if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                        visit(child)
    visit(traced)


@pytest.mark.research
@pytest.mark.parametrize("name,gap", [("square-r2", 7.), ("skew-r3", 0.)])
def test_native_refined_replay_and_contact_preserving_tangent(name, gap, jax_runtime):
    jax, jnp, _ = jax_runtime
    plan, order, settings, _ = scene(name)
    cell = CartesianPairCell(CartesianPairPlan(gap == 0, plan.shift, plan.refinement), order)
    a, w, h, offset = (np.asarray(settings[key]) for key in ("basis", "widths", "height", "offset"))
    ea, eb = settings["epsilon"]
    bg = settings["background"]
    p = np.r_[a.ravel()/700, w/700, h/700, gap/700, offset/700,
              ea.real, ea.imag, eb.real, eb.imag, bg.real, bg.imag]
    # One mixed geometry/material tangent. The exact-contact gap is fixed.
    dp = np.array([.01, -.02, .01, .015, .02, -.01, .01, 0., .01, -.02,
                   .1, .02, -.1, .01, .03, .01])
    v16, d16, m16 = native_oracle(p, dp, cell, 16)
    v24, d24, m24 = native_oracle(p, dp, cell, 24)
    np.testing.assert_allclose(v16, v24, rtol=0, atol=3e-10)
    np.testing.assert_allclose(d16, d24, rtol=0, atol=3e-9)
    for meta in (m16, m24):
        assert meta["original_residual"] < 1e-10
        assert meta["derivative_residual"] < 1e-10
    def value(parameters):
        return cell.coefficients(**native_settings(parameters, jnp), xp=jnp)[0]
    actual, tangent = jax.jit(lambda x, dx: jax.jvp(value, (x,), (dx,)))(jnp.asarray(p), jnp.asarray(dp))
    np.testing.assert_allclose(actual, v24, rtol=0, atol=3e-10)
    np.testing.assert_allclose(tangent, d24, rtol=0, atol=3e-9)
