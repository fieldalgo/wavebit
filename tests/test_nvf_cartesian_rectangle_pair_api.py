"""Explicit pair admission and independent physical/closure derivative controls."""
from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from _conv_frame import about_centre
import pytest

import wavebit
from wavebit._core.conforming.cartesian_pair import CartesianPairCell, CartesianPairPlan
from wavebit._core.parametric import assemble_eps_coeffs, conv_matrix_2d
from test_nvf_cartesian_pair import CASES, LENGTH, case_cell, closure_oracle, independent_case


def native_parameters(p):
    """Native physical inputs: A4, widths2, height, centers2, common y, materials6."""
    wleft, wright, height, gap = LENGTH*p[4:8]
    offset_x, offset_y = LENGTH*p[8:10]
    left = offset_x-(wright+gap)/2
    right = offset_x+(wleft+gap)/2
    physical = np.r_[LENGTH*p[:4], wleft, wright, height, left, right, offset_y]
    # These fixtures prescribe integer/half-integer nm inputs. Specify those
    # native primal decimals directly: this is not production contact snapping.
    exact_fixture = np.round(physical, decimals=12)
    np.testing.assert_allclose(exact_fixture, physical, rtol=0, atol=3e-13)
    return np.r_[exact_fixture, p[10:]]


def native_tangent(direction):
    result = np.r_[LENGTH*direction[:7], np.zeros(3), direction[10:]]
    result[7] = LENGTH*(direction[8]-(direction[5]+direction[7])/2)
    result[8] = LENGTH*(direction[8]+(direction[4]+direction[7])/2)
    result[9] = LENGTH*direction[9]
    return result


def scene(native, xp=np, *, reverse=False, wrapped=False):
    native = xp.asarray(native)
    material = lambda value: wavebit.Material(value) if wrapped else value
    left = dict(lx=native[4], ly=native[6], cx=native[7], cy=native[9],
                material=material(native[10]+1j*native[11]))
    right = dict(lx=native[5], ly=native[6], cx=native[8], cy=native[9],
                 material=material(native[12]+1j*native[13]))
    shapes = [right, left] if reverse else [left, right]
    return [dict(background=material(native[14]+1j*native[15]), shapes=shapes)], native[:4].reshape(2, 2)


def direct_settings(native):
    width1, width2, height, cx1, cx2, cy = native[4:10]
    return dict(basis=native[:4].reshape(2, 2), widths=(width1, width2), height=height,
                gap=(cx2-cx1)-(width1+width2)/2,
                offset=((cx1+cx2)/2+(width2-width1)/4, cy),
                epsilon=(native[10]+1j*native[11], native[12]+1j*native[13]),
                background=native[14]+1j*native[15])


@pytest.mark.research
@pytest.mark.parametrize('order', [(0, 0), (0, 1), (2, 0), (1, 2), (2, 1)])
@pytest.mark.parametrize('reverse', [False, True])
def test_physical_mapping_input_order_and_exact_material_transform(order, reverse):
    _, p = case_cell('gap-r1')
    native = native_parameters(p)
    structure, basis = scene(native, reverse=reverse, wrapped=True)
    builder = wavebit.CartesianRectanglePairNVF(order, refinement=1)
    shapes_before = tuple(id(shape) for shape in structure[0]['shapes'])
    structure = [dict(background=wavebit.Material(2.3+.4j)), *structure, {}]
    plan = builder.plan(structure, basis)
    assert tuple(id(shape) for shape in structure[1]['shapes']) == shapes_before
    assert plan.layers[0] is plan.layers[2] is None
    assert plan.layers[1].order == ((1, 0) if reverse else (0, 1))
    actual = builder.conv_matrices(structure, basis, plan=plan)
    direct, _ = CartesianPairCell(plan.layers[1].cell, order).conv_matrices(**direct_settings(native))
    count = (2*order[0]+1)*(2*order[1]+1)
    for value, reference in zip(actual, direct):
        assert value.shape == (3, count, count)
        assert value.dtype == np.complex128
        np.testing.assert_array_equal(value[1], about_centre(reference[0], order))
    for layer, epsilon in ((0, 2.3+.4j), (2, 1.)):
        for value, factor in zip(actual, (epsilon, epsilon, 1/epsilon, 0., 0.)):
            np.testing.assert_allclose(value[layer], factor*np.eye(count), rtol=0, atol=1e-15)
    material_coefficients, _ = assemble_eps_coeffs(np, structure[1]['shapes'], structure[1]['background'],
                                                   *order, basis, np.complex128)
    np.testing.assert_allclose(np.linalg.inv(actual.epz_i[1]), about_centre(conv_matrix_2d(np, material_coefficients, *order), order),
                               rtol=0, atol=3e-12)


@pytest.mark.parametrize('lengths,width', [(dict(l=140.), 140.), (dict(l=140., lx=150.), 150.),
                                          (dict(l=150., ly=140.), 150.), (dict(l=90., lx=150., ly=140.), 150.)])
def test_ordinary_alias_precedence_and_numpy_namespace(lengths, width):
    basis = np.array([[700., 315.], [0., 490.]])
    left = dict(lengths, cx=-100., material=4.+.2j)
    right = dict(lx=160., ly=140., cx=100., material=9.+.1j)
    structure = [dict(shapes=[left, right])]
    builder = wavebit.CartesianRectanglePairNVF(np.int64(0))
    plan = builder.plan(structure, basis)
    actual = builder.conv_matrices(structure, basis, plan=plan)
    direct, _ = CartesianPairCell(plan.layers[0].cell, 0).conv_matrices(
        basis=basis, widths=(width, 160.), height=140., gap=200.-(width+160.)/2,
        offset=((160.-width)/4, 0.), epsilon=(4.+.2j, 9.+.1j))
    for value, reference in zip(actual, direct):
        np.testing.assert_array_equal(value, reference)
    lattice = wavebit.Lattice(basis[:, 0], basis[:, 1], axis_aligned=False)
    for value, reference in zip(builder.conv_matrices(structure, lattice, plan=plan, xp=wavebit.rcwa().xp), actual):
        np.testing.assert_array_equal(value, reference)


def test_immutable_order_contact_and_geometry_only_planning(monkeypatch):
    _, p = case_cell('contact-r0')
    native = native_parameters(p)
    structure, basis = scene(native)
    builder = wavebit.CartesianRectanglePairNVF(0)
    with monkeypatch.context() as patch:
        patch.setattr(CartesianPairCell, 'coefficients', lambda *a, **k: pytest.fail('planning solved a cell'))
        plan = builder.plan(structure, basis)
    assert plan.layers[0].cell.contact
    with pytest.raises(FrozenInstanceError):
        plan.layers = ()
    with pytest.raises(FrozenInstanceError):
        plan.layers[0].order = (1, 0)
    with pytest.raises(FrozenInstanceError):
        plan.layers[0].cell.contact = False
    monkeypatch.setattr(CartesianPairPlan, 'from_geometry', lambda *a, **k: pytest.fail('replay replanned'))
    original = builder.conv_matrices(structure, basis, plan=plan)
    assert np.isfinite(original.epx).all()
    with pytest.raises(ValueError, match='order'):
        builder.conv_matrices(*scene(native, reverse=True), plan=plan)
    moved = native.copy()
    moved[8] += 1.
    with pytest.raises(ValueError, match='contact'):
        builder.conv_matrices(*scene(moved), plan=plan)
    # No fuzzy contact policy: a representable positive opening invalidates a contact plan.
    moved[8] = native[8]+4*np.spacing(native[8])
    assert direct_settings(moved)['gap'] > 0
    with pytest.raises(ValueError, match='contact'):
        builder.conv_matrices(*scene(moved), plan=plan)


def test_mandatory_plan_schema_material_and_horizontal_admission():
    _, p = case_cell('gap-r0')
    native = native_parameters(p)
    structure, basis = scene(native)
    builder = wavebit.CartesianRectanglePairNVF(0)
    plan = builder.plan(structure, basis)
    with pytest.raises(TypeError, match='plan'):
        builder.conv_matrices(structure, basis)
    for invalid in (None, plan.layers, replace(plan, layers=list(plan.layers)), replace(plan, layers=(True,)),
                    replace(plan, layers=()), wavebit.CartesianRectanglePairNVF(0, refinement=1).plan(structure, basis)):
        with pytest.raises(wavebit.StaleGeometryError):
            builder.conv_matrices(structure, basis, plan=invalid)
    for invalid in ([], {}, [None], [dict(shapes=None)], [dict(profile={})], [dict(shapes=[{}])],
                    [dict(shapes=[{}, {}, {}])], [dict(shapes=[None, {}])], [dict(shpaes=[])]):
        with pytest.raises((ValueError, TypeError, NotImplementedError)):
            builder.plan(invalid, basis)
    with pytest.raises(wavebit.StaleGeometryError):
        builder.conv_matrices([{}], basis, plan=plan)
    for change in (dict(ly=211.), dict(cy=1.), dict(angle=.01), dict(angle=np.pi), dict(angle=2*np.pi),
                   dict(lx=0.), dict(lx=True), dict(cx=1j), dict(ly=210.+0j), dict(type='square'), dict(angel=0.)):
        invalid, _ = scene(native)
        invalid[0]['shapes'][0].update(change)
        with pytest.raises((ValueError, TypeError, NotImplementedError)):
            builder.plan(invalid, basis)
        with pytest.raises((ValueError, TypeError, NotImplementedError)):
            builder.conv_matrices(invalid, basis, plan=plan)
    for material in (-1., 4.-.01j, np.inf, np.eye(2), wavebit.Material(3., mu=1.), 'silicon'):
        for where in ('background', 'left', 'right'):
            invalid, _ = scene(native)
            if where == 'background':
                invalid[0]['background'] = material
            else:
                invalid[0]['shapes'][0 if where == 'left' else 1]['material'] = material
            with pytest.raises((ValueError, TypeError, NotImplementedError)):
                builder.plan(invalid, basis)
            with pytest.raises((ValueError, TypeError, NotImplementedError)):
                builder.conv_matrices(invalid, basis, plan=plan)
    for invalid in (basis.astype(complex), [[700., 315.], [0., -490.]], [[1., 1.], [1., 1.]],
                    [[np.inf, 0.], [0., 1.]], [700., 490.]):
        with pytest.raises(ValueError):
            builder.plan([{}], invalid)
        with pytest.raises(ValueError):
            builder.conv_matrices([{}], invalid, plan=builder.plan([{}], basis))
    for order in (True, -1, .5, (1, -1)):
        with pytest.raises(ValueError):
            wavebit.CartesianRectanglePairNVF(order)
    for refinement in (True, -1, .5, 4):
        with pytest.raises(ValueError):
            wavebit.CartesianRectanglePairNVF(0, refinement=refinement)
    assert wavebit.CartesianRectanglePairNVF(0, refinement=3).refinement == 3
    with pytest.raises(NotImplementedError):
        builder.conv_matrices(structure, basis, plan=plan, xp=object())


@pytest.mark.research
def test_translation_scale_and_reversed_concrete_plan():
    _, p = case_cell('gap-r1')
    native = native_parameters(p)
    builder = wavebit.CartesianRectanglePairNVF((1, 1), refinement=1)
    plan = builder.plan(*scene(native))
    original = builder.conv_matrices(*scene(native), plan=plan)
    reversed_plan = builder.plan(*scene(native, reverse=True))
    for actual, expected in zip(builder.conv_matrices(*scene(native, reverse=True), plan=reversed_plan), original):
        np.testing.assert_array_equal(actual, expected)
    scaled = native.copy()
    scaled[:10] *= .25
    for actual, expected in zip(builder.conv_matrices(*scene(scaled), plan=plan), original):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-10)
    moved = native.copy()
    moved[7:9] += 37.
    moved[9] -= 23.
    result = builder.conv_matrices(*scene(moved), plan=plan)
    n, m = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing='ij')
    phase = np.exp(-2j*np.pi*(np.stack([m.ravel(), n.ravel()], axis=-1)
                             @np.linalg.solve(native[:4].reshape(2, 2), [37., -23.])))
    for actual, expected in zip(result, original):
        np.testing.assert_allclose(actual[0], phase[:, None]*expected[0]*phase.conj()[None], rtol=0, atol=2e-10)


def test_package_only_import_and_unchanged_other_adapters(tmp_path):
    _, p = case_cell('gap-r0')
    structure, basis = scene(native_parameters(p))
    with pytest.raises(NotImplementedError):
        wavebit.CartesianRectangleNVF(0).plan(structure, basis)
    with pytest.raises(ValueError, match='orthogonal'):
        wavebit.ConformingNVF(0).plan(structure, basis)
    with pytest.raises(ValueError):
        wavebit.AffineProfileNVF(0).conv_matrices(structure, basis)
    shutil.copytree(Path(wavebit.__file__).parent, tmp_path/'wavebit', ignore=shutil.ignore_patterns('__pycache__'))
    code = """
import sys
import numpy as np
import wavebit
assert 'CartesianRectanglePairNVF' in wavebit.__all__
b = wavebit.CartesianRectanglePairNVF(0)
a = np.array([[700.,315.],[0.,490.]])
s = [dict(shapes=[dict(lx=140.,ly=210.,cx=-47.5,cy=-21.,material=4.),
                  dict(lx=193.,ly=210.,cx=119.,cy=-21.,material=9.)])]
p = b.plan(s,a)
assert p.layers[0].cell.contact
assert np.isfinite(b.conv_matrices(s,a,plan=p).epx).all()
assert not any(n == 'studies' or n.startswith('studies.') for n in sys.modules)
assert 'jax' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                            env=dict(os.environ, PYTHONPATH=str(tmp_path)), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('order', [(0, 0), (1, 0), (0, 1)])
def test_engine_uniform_airy_and_patterned_slot_layout_only(order):
    """A tiny integration/energy control, not a patterned optical reference."""
    wavelength, thickness, theta = 900., 170., .23
    top, film, bottom = 1., 4., 2.25
    _, p = case_cell('gap-r0')
    native = native_parameters(p)
    _, basis = scene(native)
    builder = wavebit.CartesianRectanglePairNVF(order)
    def solve(structure):
        plan = builder.plan(structure, basis)
        conv = builder.conv_matrices(structure, basis, plan=plan)
        solver = wavebit.rcwa(backend='numpy', lattice=basis, wavelength=wavelength,
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
    native[[11, 13, 15]] = 0.
    structure, _ = scene(native)
    patterned = solve(structure)
    for polarization in (patterned.p_inc, patterned.s_inc):
        assert np.isfinite(polarization.T).all() and np.isfinite(polarization.R).all()
        np.testing.assert_allclose(np.sum(polarization.T)+np.sum(polarization.R), 1., rtol=0, atol=2e-10)


@pytest.mark.research
@pytest.mark.parametrize('case', tuple(CASES))
def test_jax_native_parameters_against_independent_pair_closure(case):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    cell, p = case_cell(case)
    native = native_parameters(p)
    reverse = case.startswith('shift1')
    wrapped = 'r1' in case
    builder = wavebit.CartesianRectanglePairNVF(cell.order, refinement=cell.plan.refinement)
    plan = builder.plan(*scene(native, reverse=reverse, wrapped=wrapped))
    assert plan.layers[0].cell.contact == cell.plan.contact
    assert plan.layers[0].cell.shift == cell.plan.shift
    expected, derivative, tangents = independent_case(case)
    def function(parameters):
        return jnp.stack(builder.conv_matrices(*scene(parameters, jnp, reverse=reverse, wrapped=wrapped), plan=plan, xp=jnp))
    evaluate = jax.jit(lambda parameters, tangent: jax.jvp(function, (parameters,), (tangent,)))
    for value, dvalue, tangent in zip(expected, derivative, tangents):
        reference, dslots = (about_centre(a, cell.order)
                             for a in closure_oracle(value, dvalue, cell.order))
        actual, dactual = evaluate(jnp.asarray(native), jnp.asarray(native_tangent(tangent)))
        np.testing.assert_allclose(actual, reference, rtol=0, atol=3e-10)
        np.testing.assert_allclose(dactual, dslots, rtol=0, atol=3e-9)


@pytest.mark.parametrize('case', ('gap-r0', 'contact-r0'))
def test_jax_invalid_pair_is_layer_local_and_invalid_lattice_is_global(case):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    _, p = case_cell(case)
    native = native_parameters(p)
    builder = wavebit.CartesianRectanglePairNVF(0)
    patterned, basis = scene(native)
    plan = builder.plan([{}, *patterned], basis)
    def function(parameters, height_delta, y_delta, angle):
        patterned, basis = scene(parameters, jnp)
        patterned[0]['shapes'][1]['ly'] += height_delta
        patterned[0]['shapes'][1]['cy'] += y_delta
        patterned[0]['shapes'][1]['angle'] = angle
        return jnp.stack(builder.conv_matrices([{}, *patterned], basis, plan=plan, xp=jnp))
    evaluate = jax.jit(function)
    for dh, dy, angle in ((1., 0., 0.), (0., 1., 0.), (0., 0., .01)):
        result = np.asarray(evaluate(jnp.asarray(native), dh, dy, angle))
        assert np.isfinite(result[:, 0]).all()
        assert np.isnan(result[:, 1]).all()
    invalid = native.copy()
    invalid[11] = -.1
    result = np.asarray(evaluate(jnp.asarray(invalid), 0., 0., 0.))
    assert np.isfinite(result[:, 0]).all() and np.isnan(result[:, 1]).all()
    invalid = native.copy()
    invalid[:4] = [700., 0., 0., 0.]
    assert np.isnan(evaluate(jnp.asarray(invalid), 0., 0., 0.)).all()
    invalid = native.copy()
    invalid[8] = invalid[7]+(invalid[4]+invalid[5])/2 if not plan.layers[1].cell.contact else invalid[8]+1.
    result = np.asarray(evaluate(jnp.asarray(invalid), 0., 0., 0.))
    assert np.isfinite(result[:, 0]).all() and np.isnan(result[:, 1]).all()
    with pytest.raises(ValueError):
        jax.jit(lambda parameters: builder.plan(*scene(parameters, jnp)))(jnp.asarray(native))
    jax.config.update('jax_enable_x64', False)
    try:
        with pytest.raises(ValueError, match='64-bit'):
            builder.conv_matrices([{}, *patterned], basis, plan=plan, xp=jnp)
    finally:
        jax.config.update('jax_enable_x64', True)
