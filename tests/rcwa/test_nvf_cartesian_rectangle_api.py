"""Explicit public rectangle admission and independent energy-closure derivatives."""

from wavebit.rcwa._engine.conforming.affine_profile import AffineProfileNVF as _legacy_wavebit_AffineProfileNVF

from wavebit.rcwa._engine.conforming.cartesian_rectangle import CartesianRectangleNVF as _legacy_wavebit_CartesianRectangleNVF

from wavebit.rcwa._engine.conforming import ConformingNVF as _legacy_wavebit_ConformingNVF

from wavebit.rcwa._engine.factorization import ConvMatrices as _legacy_wavebit_ConvMatrices

from wavebit.rcwa._engine.geometry import StaleGeometryError as _legacy_wavebit_StaleGeometryError

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from tests.fixtures.common._conv_frame import about_centre
import pytest

import wavebit
from wavebit.rcwa._engine.conforming.cartesian_rectangle import CartesianRectangleCell, CartesianRectanglePlan
from wavebit.rcwa._engine.parametric import assemble_eps_coeffs, conv_matrix_2d
from tests.rcwa.test_nvf_cartesian_cell import LENGTH, PARAMETERS, matrix_blocks, oracle, settings

PARAMETERS = PARAMETERS.copy()
PARAMETERS[6] = np.pi+.31


def scene(p=PARAMETERS, xp=np, *, wrapped=False):
    p = xp.asarray(p)
    material = lambda value: wavebit.Material(value) if wrapped else value
    rectangle = dict(type='rectangle', lx=LENGTH*p[4], ly=LENGTH*p[5], angle=p[6],
                     cx=LENGTH*p[7], cy=LENGTH*p[8], material=material(p[9]+1j*p[10]))
    return [dict(background=material(p[11]+1j*p[12]), shapes=[rectangle])], LENGTH*p[:4].reshape(2, 2)


def closure_variation(value, derivative):
    """Differentiate the defining closure using independent explicit inverses."""
    q, g, e = matrix_blocks(value, (1, 1), (2, 2))
    dq, dg, de = matrix_blocks(derivative, (1, 1), (2, 2))
    qi, ei = np.linalg.inv(q), np.linalg.inv(e)
    effective = qi.conj().T@g@qi
    variation = qi.conj().T@dg@qi-qi.conj().T@dq.conj().T@effective-effective@dq@qi
    z_variation = -ei@de@ei
    count = len(e)
    def slots(tensor, z):
        return np.stack([tensor[:count, :count], tensor[count:, count:], z,
                         -tensor[:count, count:], -tensor[count:, :count]])[:, None]
    return slots(effective, ei), slots(variation, z_variation)


@pytest.mark.research
@pytest.mark.parametrize('order', [(0, 0), (0, 1), (2, 0), (1, 2), (2, 1)])
def test_exact_physical_mapping_labels_and_independent_material_transform(order):
    builder = _legacy_wavebit_CartesianRectangleNVF(order, refinement=1)
    structure, basis = scene(wrapped=True)
    structure = [dict(background=wavebit.Material(2.3+.4j)), *structure, {}]
    plan = builder.plan(structure, basis)
    actual = builder.conv_matrices(structure, basis, plan=plan)
    assert isinstance(actual, _legacy_wavebit_ConvMatrices)
    assert plan.layers[0] is plan.layers[2] is None
    assert isinstance(plan.layers[1], CartesianRectanglePlan)
    cell = CartesianRectangleCell(order, refinement=1, plan=plan.layers[1])
    expected = cell.conv_matrices(**settings(PARAMETERS))
    count = (2*order[0]+1)*(2*order[1]+1)
    for value, direct in zip(actual, expected):
        assert value.shape == (3, count, count)
        assert value.dtype == np.complex128
        np.testing.assert_array_equal(value[1], about_centre(direct[0], order))
    for layer, epsilon in ((0, 2.3+.4j), (2, 1.)):
        for value, factor in zip(actual, (epsilon, epsilon, 1/epsilon, 0., 0.)):
            np.testing.assert_allclose(value[layer], factor*np.eye(count), rtol=0, atol=1e-15)
    coefficients, _ = assemble_eps_coeffs(np, structure[1]['shapes'], structure[1]['background'],
                                           *order, basis, np.complex128)
    expected_material = about_centre(conv_matrix_2d(np, coefficients, *order), order)
    np.testing.assert_allclose(np.linalg.inv(actual.epz_i[1]), expected_material, rtol=0, atol=2e-12)


@pytest.mark.parametrize('lengths,expected', [
    (dict(l=200.), (200., 200.)), (dict(l=200., lx=250.), (250., 200.)),
    (dict(l=200., ly=150.), (200., 150.)),
    (dict(l=190., lx=250., ly=150.), (250., 150.)), (dict(lx=250., ly=150.), (250., 150.))])
def test_ordinary_rectangle_alias_precedence_and_defaults(lengths, expected):
    basis = np.array([[700., 315.], [0., 490.]])
    builder = _legacy_wavebit_CartesianRectangleNVF(np.int64(0))
    structure = [dict(shapes=[dict(lengths, material=wavebit.Material(4.+.2j))])]
    plan = builder.plan(structure, basis)
    actual = builder.conv_matrices(structure, basis, plan=plan)
    cell = CartesianRectangleCell(0, plan=plan.layers[0])
    direct = cell.conv_matrices(basis=basis, width=expected[0], height=expected[1], epsilon=4.+.2j)
    for value, reference in zip(actual, direct):
        np.testing.assert_array_equal(value, reference)
    lattice = wavebit.Lattice(basis[:, 0], basis[:, 1], axis_aligned=False)
    for value, reference in zip(builder.conv_matrices(structure, lattice, plan=plan, xp=_legacy_wavebit_rcwa().xp), actual):
        np.testing.assert_array_equal(value, reference)


def test_concrete_planning_is_geometry_only_and_replay_does_not_replan(monkeypatch):
    structure, basis = scene()
    builder = _legacy_wavebit_CartesianRectangleNVF(0)
    with monkeypatch.context() as patch:
        patch.setattr(CartesianRectangleCell, 'coefficients', lambda *a, **k: pytest.fail('planning solved a cell'))
        plan = builder.plan(structure, basis)
    with pytest.raises(FrozenInstanceError):
        plan.layers = ()
    with pytest.raises(FrozenInstanceError):
        plan.layers[0].permutation = (0, 1, 2, 3)
    monkeypatch.setattr(CartesianRectanglePlan, 'from_geometry', lambda *a, **k: pytest.fail('replay replanned'))
    assert np.isfinite(builder.conv_matrices(structure, basis, plan=plan).epx).all()


@pytest.mark.research
def test_phase_joint_rotation_scale_and_equivalent_charts():
    builder = _legacy_wavebit_CartesianRectangleNVF((1, 1), refinement=1)
    structure, basis = scene()
    plan = builder.plan(structure, basis)
    original = builder.conv_matrices(structure, basis, plan=plan)
    shifted = PARAMETERS.copy()
    shifted[7:9] += [1.3, -.7]
    moved_scene, moved_basis = scene(shifted)
    moved = builder.conv_matrices(moved_scene, moved_basis, plan=plan)
    n, m = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing='ij')
    phase_shift = np.linalg.solve(PARAMETERS[:4].reshape(2, 2), shifted[7:9]-PARAMETERS[7:9])
    phase = np.exp(-2j*np.pi*(m.ravel()*phase_shift[0]+n.ravel()*phase_shift[1]))
    for value, reference in zip(moved, original):
        np.testing.assert_allclose(value[0], phase[:, None]*reference[0]*phase.conj()[None], rtol=0, atol=2e-10)
    scaled = PARAMETERS.copy()
    scaled[:6] *= .003
    scaled[7:9] *= .003
    for value, reference in zip(builder.conv_matrices(*scene(scaled), plan=plan), original):
        np.testing.assert_allclose(value, reference, rtol=0, atol=2e-10)
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = PARAMETERS.copy()
    rotated[:4] = (rotation@PARAMETERS[:4].reshape(2, 2)).ravel()
    rotated[6] += angle
    rotated[7:9] = rotation@PARAMETERS[7:9]
    actual = builder.conv_matrices(*scene(rotated), plan=plan)
    tensor = np.block([[original.epx[0], -original.epxy[0]], [-original.epyx[0], original.epy[0]]])
    transformation = np.kron(rotation, np.eye(9))
    tensor = transformation@tensor@transformation.T
    references = [tensor[:9, :9], tensor[9:, 9:], original.epz_i[0], -tensor[:9, 9:], -tensor[9:, :9]]
    for value, reference in zip(actual, references):
        np.testing.assert_allclose(value[0], reference, rtol=0, atol=2e-10)
    for swap in (False, True):
        equivalent = PARAMETERS.copy()
        equivalent[6] += np.pi/2 if swap else np.pi
        if swap:
            equivalent[4:6] = equivalent[4:6][::-1]
        structure, basis = scene(equivalent)
        equivalent_plan = builder.plan(structure, basis)
        for value, reference in zip(builder.conv_matrices(structure, basis, plan=equivalent_plan), original):
            np.testing.assert_allclose(value, reference, rtol=0, atol=2e-10)


def test_structural_and_material_admission_and_mandatory_plan():
    builder = _legacy_wavebit_CartesianRectangleNVF(0)
    structure, basis = scene()
    plan = builder.plan(structure, basis)
    with pytest.raises(TypeError, match='plan'):
        builder.conv_matrices(structure, basis)
    for bad_plan in (None, plan.layers, replace(plan, layers=list(plan.layers)),
                     replace(plan, layers=(True,)), replace(plan, layers=())):
        with pytest.raises(_legacy_wavebit_StaleGeometryError):
            builder.conv_matrices(structure, basis, plan=bad_plan)
    with pytest.raises(_legacy_wavebit_StaleGeometryError, match='shape count'):
        builder.conv_matrices([{}], basis, plan=plan)
    with pytest.raises(_legacy_wavebit_StaleGeometryError, match='shape count'):
        builder.conv_matrices(structure, basis, plan=builder.plan([{}], basis))
    for invalid in ([], {}, [None], [dict(shapes=None)], [dict(shapes=[None])],
                    [dict(profile={})], [dict(shpaes=[])], [dict(shapes=[dict(type='circle', lx=30.)])],
                    [dict(shapes=[{}, {}])], [dict(shapes=[dict(lx=30., ly=30., angel=.2)])]):
        with pytest.raises((ValueError, TypeError, NotImplementedError)):
            builder.plan(invalid, basis)
    for material in (-1., 1.-.1j, np.inf, np.eye(2), wavebit.Material(3., mu=1.), 'silicon'):
        for where in ('background', 'shape'):
            invalid, _ = scene()
            if where == 'background':
                invalid[0]['background'] = material
            else:
                invalid[0]['shapes'][0]['material'] = material
            with pytest.raises((ValueError, TypeError, NotImplementedError)):
                builder.plan(invalid, basis)
            with pytest.raises((ValueError, TypeError, NotImplementedError)):
                builder.conv_matrices(invalid, basis, plan=plan)
    for change in (dict(lx=0.), dict(ly=-1.), dict(lx=True), dict(lx=1400.), dict(cx=np.nan), dict(angle=1j)):
        invalid, _ = scene()
        invalid[0]['shapes'][0].update(change)
        with pytest.raises(ValueError):
            builder.plan(invalid, basis)
        with pytest.raises(ValueError):
            builder.conv_matrices(invalid, basis, plan=plan)
    for invalid_basis in ([[700., 315.], [0., -490.]], [[1., 1.], [1., 1.]],
                          [[np.inf, 0.], [0., 1.]], [[700., 1j], [0., 490.]], [700., 490.]):
        with pytest.raises(ValueError, match='lattice'):
            builder.plan([{}], invalid_basis)
        with pytest.raises(ValueError, match='lattice'):
            builder.conv_matrices([{}], invalid_basis, plan=builder.plan([{}], basis))
    with pytest.raises(NotImplementedError, match='NumPy and JAX'):
        builder.conv_matrices(structure, basis, plan=plan, xp=object())
    for order in (True, -1, .5, (1, -1)):
        with pytest.raises(ValueError):
            _legacy_wavebit_CartesianRectangleNVF(order)
    for refinement in (True, -1, .5, 3):
        with pytest.raises(ValueError):
            _legacy_wavebit_CartesianRectangleNVF(0, refinement=refinement)


def test_existing_public_admission_is_unchanged():
    structure, basis = scene()
    with pytest.raises(ValueError, match='orthogonal'):
        _legacy_wavebit_ConformingNVF(0).plan(structure, basis)
    with pytest.raises(ValueError, match='unknown keys'):
        _legacy_wavebit_AffineProfileNVF(0).conv_matrices(structure, basis)


def test_installed_package_surface_does_not_depend_on_studies(tmp_path):
    shutil.copytree(Path(wavebit.__file__).parent, tmp_path/'wavebit',
                    ignore=shutil.ignore_patterns('__pycache__'))
    code = """
import sys
import numpy as np
from wavebit.rcwa import experimental
from wavebit import Material
assert 'CartesianRectangleNVF' in experimental.__all__
builder = experimental.CartesianRectangleNVF((0, 0))
basis = np.array([[700.,315.],[0.,490.]])
structure = [dict(shapes=[dict(l=200., angle=3.45, material=Material(4.))])]
plan = builder.plan(structure,basis)
conv = builder.conv_matrices(structure,basis,plan=plan)
assert np.isfinite(conv.epx).all()
assert not any(n == 'studies' or n.startswith('studies.') for n in sys.modules)
assert 'jax' not in sys.modules
"""
    environment = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('order', [(0, 0), (1, 0), (0, 1)])
def test_engine_uniform_airy_and_patterned_slot_layout_only(order):
    """A tiny integration/energy control, not a patterned optical reference."""
    wavelength, thickness, theta = 900., 170., .23
    top, film, bottom = 1., 4., 2.25
    _, basis = scene()
    builder = _legacy_wavebit_CartesianRectangleNVF(order)
    def solve(structure):
        plan = builder.plan(structure, basis)
        conv = builder.conv_matrices(structure, basis, plan=plan)
        solver = _legacy_wavebit_rcwa(backend='numpy', lattice=basis, wavelength=wavelength,
                              thickness=[thickness], truncation_order=order,
                              theta=theta, phi=.47, top=top, bot=bottom)
        return solver.solve_conv(*conv)
    uniform = solve([dict(background=film)])
    q = np.sqrt(np.array([top, film, bottom])-top*np.sin(theta)**2)
    phase = np.exp(1j*2*np.pi/wavelength*q[1]*thickness)
    for polarization, admittance in ((uniform.p_inc, np.array([top, film, bottom])/q),
                                     (uniform.s_inc, q)):
        r01 = (admittance[0]-admittance[1])/(admittance[0]+admittance[1])
        r12 = (admittance[1]-admittance[2])/(admittance[1]+admittance[2])
        denominator = 1+r01*r12*phase**2
        reflected = abs((r01+r12*phase**2)/denominator)**2
        transmitted = admittance[2]/admittance[0]*abs((1+r01)*(1+r12)*phase/denominator)**2
        for actual, exact in ((polarization.T, transmitted), (polarization.R, reflected)):
            expected = np.zeros((2*order[1]+1, 2*order[0]+1))
            expected[order[1], order[0]] = exact
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-10)
    parameters = PARAMETERS.copy()
    parameters[[10, 12]] = 0.
    structure, _ = scene(parameters)
    patterned = solve(structure)
    for polarization in (patterned.p_inc, patterned.s_inc):
        assert np.isfinite(polarization.T).all() and np.isfinite(polarization.R).all()
        np.testing.assert_allclose(np.sum(polarization.T)+np.sum(polarization.R), 1., rtol=0, atol=2e-10)


@pytest.mark.research
@pytest.mark.parametrize('wrapped', [False, True])
def test_jax_all_thirteen_parameters_and_independent_closure_variation(wrapped):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = _legacy_wavebit_CartesianRectangleNVF((1, 1), refinement=1)
    plan = builder.plan(*scene(wrapped=wrapped))
    cell = CartesianRectangleCell((1, 1), refinement=1, plan=plan.layers[0])
    evaluate = jax.jit(lambda p, d: jax.jvp(
        lambda p: jnp.stack(builder.conv_matrices(*scene(p, jnp, wrapped=wrapped), plan=plan, xp=jnp)), (p,), (d,)))
    generator = np.array([[0., -1.], [1., 0.]])
    rotation = np.zeros(13)
    rotation[:4], rotation[6] = (generator@PARAMETERS[:4].reshape(2, 2)).ravel(), 1.
    rotation[7:9] = generator@PARAMETERS[7:9]
    scale = np.zeros(13)
    scale[:6], scale[7:9] = PARAMETERS[:6], PARAMETERS[7:9]
    shift = np.zeros(13)
    shift[7:9] = [.2, -.1]
    directions = [*np.eye(13), rotation, scale, shift,
                  np.array([.05, -.09, .04, .03, .06, -.05, .11, .07, -.08, .2, -.1, .09, .03])]
    for direction in directions:
        low, dlow, _ = oracle(PARAMETERS, direction, cell, 16)
        high, dhigh, _ = oracle(PARAMETERS, direction, cell, 24)
        np.testing.assert_allclose(high, low, rtol=0, atol=3e-10)
        np.testing.assert_allclose(dhigh, dlow, rtol=0, atol=3e-10)
        expected, derivative = (about_centre(a, (1, 1)) for a in closure_variation(high, dhigh))
        actual, tangent = evaluate(jnp.asarray(PARAMETERS), jnp.asarray(direction))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-10)
        np.testing.assert_allclose(tangent, derivative, rtol=0, atol=3e-9)


def test_jax_invalid_replay_poisoning_uniform_lattice_and_precision():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = _legacy_wavebit_CartesianRectangleNVF((1, 0))
    def stack(p, xp):
        patterned, basis = scene(p, xp, wrapped=True)
        return [dict(background=2.3), *patterned, {}], basis
    plan = builder.plan(*stack(PARAMETERS, np))
    evaluate = jax.jit(lambda p: jnp.stack(builder.conv_matrices(*stack(p, jnp), plan=plan, xp=jnp)))
    assert np.isfinite(np.asarray(evaluate(jnp.asarray(PARAMETERS)))).all()
    for index, invalid in ((4, 2.), (6, PARAMETERS[6]+np.pi), (7, np.nan), (9, 0.), (10, -.1), (12, -.1)):
        p = PARAMETERS.copy()
        p[index] = invalid
        value = np.asarray(evaluate(jnp.asarray(p)))
        assert np.isnan(value[:, 1]).all()
        assert np.isfinite(value[:, (0, 2)]).all()
    for index, invalid in ((3, -.7), (0, np.inf)):
        p = PARAMETERS.copy()
        p[index] = invalid
        assert np.isnan(np.asarray(evaluate(jnp.asarray(p)))).all()
    uniform_plan = builder.plan([{}, dict(background=2.3)], np.eye(2))
    uniform = jax.jit(lambda basis: jnp.stack(builder.conv_matrices(
        [{}, dict(background=2.3)], basis, plan=uniform_plan, xp=jnp)))
    assert np.isnan(np.asarray(uniform(jnp.zeros((2, 2))))).all()
    with pytest.raises(ValueError):
        jax.jit(lambda p: builder.plan(*scene(p, jnp)).layers[0].permutation[0])(jnp.asarray(PARAMETERS))
    jax.config.update('jax_enable_x64', False)
    try:
        with pytest.raises(ValueError, match='64-bit'):
            builder.conv_matrices(*scene(), plan=builder.plan(*scene()), xp=jnp)
    finally:
        jax.config.update('jax_enable_x64', True)
