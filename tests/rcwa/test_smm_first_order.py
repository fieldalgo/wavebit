"""Public first-order SMM: ports, fields, stack composition and stability."""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest

import wavebit
from tests.rcwa.test_full_tensor_patterns import host, tensors


def make_solver(algorithm='SMM', backend='numpy', side='top', **overrides):
    e, m = tensors(.04)
    options = dict(
        backend=backend, wavelength=730., period=(600., 570.),
        thickness=[73., 51.], truncation_order=(1, 1),
        kx=.001, ky=.0003, polarization=(1., .4j),
        top=wavebit.Material(1.6 + .05j, 1.2),
        bot=wavebit.Material(2.2 + .1j, 1.4),
        structure=[
            dict(background=1., shapes=[dict(type='rectangle', lx=237., ly=310.,
                                             material=wavebit.Material(e, m))]),
            dict(background=wavebit.Material([2.3, 3.1, 2.9]).rotated(.27, 'y'), shapes=[]),
        ], factorization='li', incidence_side=side,
        layer_algorithm=algorithm, answers='field')
    options.update(overrides)
    return _legacy_wavebit_rcwa(**options)


def compare_results(actual, expected, atol=3e-11):
    for channel in ('incident', 's_inc', 'p_inc'):
        for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T', 'power_entering', 'absorption'):
            np.testing.assert_allclose(host(getattr(getattr(actual, channel), name)),
                                       host(getattr(getattr(expected, channel), name)),
                                       atol=atol, rtol=0, err_msg=f'{channel}.{name}')


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_tilted_magnetic_pattern_fields_match_etm(side, backend):
    smm = make_solver(backend=backend, side=side)
    etm = make_solver('ETM', side=side)
    compare_results(smm.solve(auto_detune=False), etm.solve(auto_detune=False))
    assert smm.formulation_assigned == 'first_order'
    # Sample either side of the internal interface and request all incident channels.
    samples = dict(sample_x=3, sample_y=2, sample_z=[0., 36., 73.-1e-7, 73.+1e-7, 124.],
                   incidences=('incident', 's', 'p'))
    np.testing.assert_allclose(host(smm.reconstruct_field(**samples)),
                               host(etm.reconstruct_field(**samples)), atol=2e-10, rtol=0)
    assert all(np.isfinite(host(block)).all() for block in smm.scattering.blocks)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_second_order_port_basis_and_coherent_orders_agree(side):
    material = wavebit.Material([2.3, 3.1, 2.9], mu=[1.3, 1.2, 1.1]).rotated(.23)
    options = dict(thickness=[71.],
                   structure=[dict(background=1., shapes=[dict(type='rectangle', lx=241., ly=304.,
                                                               material=material)])],
                   period=(1800., 1700.), excitation={(0, 0): 1., (1, 0): .3j})
    solvers = [make_solver(side=side, **options) for _ in range(2)]
    solvers[0].formulation = 'second_order'
    solvers[1].formulation = 'first_order'
    compare_results(solvers[1].solve(auto_detune=False), solvers[0].solve(auto_detune=False))
    for actual, expected in zip(solvers[1].scattering.blocks, solvers[0].scattering.blocks):
        np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=0)
    samples = dict(sample_x=3, sample_y=2, sample_z=3)
    np.testing.assert_allclose(solvers[1].reconstruct_field(**samples),
                               solvers[0].reconstruct_field(**samples), atol=3e-11, rtol=0)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_scattering_composes_with_a_second_order_stack(side):
    tilted = wavebit.Material([2.4, 3.2, 4.1], mu=[1.1, 1.3, 1.4]).rotated(.31, 'y')
    flat = wavebit.Material([2.2, 3.5, 3.1]).rotated(.21)
    lower, upper = dict(background=tilted, shapes=[]), dict(background=flat, shapes=[])
    options = dict(truncation_order=(0, 0), answers='amplitude')
    whole = make_solver(side=side, structure=[lower, upper], thickness=[73., 51.], **options)
    bottom = make_solver(side=side, structure=[lower], thickness=[73.], top=1.69, **options)
    top = make_solver(side=side, structure=[upper], thickness=[51.], bot=1.69, **options)
    for s in (whole, bottom, top):
        s.solve(auto_detune=False)
    assert bottom.formulation_assigned == 'first_order'
    assert top.formulation_assigned == 'second_order'
    first, second = (top, bottom) if side == 'top' else (bottom, top)
    joined = first.scattering.cascade(second.scattering)
    for actual, expected in zip(joined.blocks, whole.scattering.blocks):
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=0)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_thick_absorber_has_fresnel_reflection_and_finite_fields(side):
    # Optical depth exceeds floating-point exponential range. No growing
    # exponential may appear in the cascade or the field recovery.
    n = .2 + 4j
    s = make_solver(side=side, top=1., bot=1., kx=0., ky=0.,
                    structure=[dict(background=n*n, shapes=[])],
                    thickness=[60000.], truncation_order=(0, 0))
    s.formulation = 'first_order'
    result = s.solve(auto_detune=False)
    assert float(result.R.sum()) == pytest.approx(abs((1-n)/(1+n))**2, abs=2e-12)
    assert float(result.T.sum()) == pytest.approx(0., abs=1e-300)
    fields = s.reconstruct_field(sample_x=1, sample_y=1, sample_z=[0., 100., 30000., 59900., 60000.])
    assert np.isfinite(fields).all()
    assert np.max(abs(fields[:, 2])) < 1e-100


@pytest.mark.parametrize('answers', ['efficiency', 'amplitude', 'field'])
def test_answer_levels_preserve_powers_and_only_retain_requested_state(answers):
    solver = make_solver(answers=answers)
    reference = make_solver('ETM').solve(auto_detune=False)
    actual = solver.solve(auto_detune=False)
    for name in ('R', 'T', 'power_entering', 'absorption'):
        np.testing.assert_allclose(getattr(actual, name), getattr(reference, name), atol=3e-11, rtol=0)
    assert bool(solver.layer_info_list) == (answers == 'field')
    assert (solver.T1 is not None) == (answers == 'field')
    assert (solver._scattering is not None) == (answers != 'efficiency')
