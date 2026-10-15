"""Independent transmission laws, spatial differences and whole-cell guards."""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
from functools import lru_cache
import numpy as np
import pytest

from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan

from wavebit.rcwa._engine.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell
from wavebit.rcwa._engine.conforming.pair_transmission_modes import TransmissionMode
from wavebit.rcwa._engine.conforming.pair_transmission_fields import PairTransmissionFields


REFERENCE = (12.25, -10+1j, 1.)
GEOMETRY = dict(basis=np.diag([1.2, .9]), widths=(.3, .4), height=.45,
                offset=(-.6, -.45))


@pytest.mark.research
def test_common_complex_reference_factor_preserves_the_real_trial_bank():
    cell = CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0), 0,
        reference=(4., 9., 1.), branches=2)
    fields = [PairTransmissionFields(cell._base, GEOMETRY, np.array([4., 9., 1.])*scale, 2, 24)
              for scale in (1., 1j)]
    assert fields[0].count == fields[1].count == 12
    bary = np.array([[.23, .31, .46], [.52, .17, .31]])
    for tri in cell._base.data['triangles']:
        samples = [bank.sample(tri, bary, np.eye(2), np.zeros((2, 2))) for bank in fields]
        for left, right in zip(*samples):
            np.testing.assert_allclose(left, right, atol=3e-10, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('coefficients', ((12.25, 1., 1.), (-10+1j, 1., 1.), REFERENCE))
@pytest.mark.parametrize('branch', range(4))
def test_transmission_laws_and_independent_spatial_derivatives(coefficients, branch):
    mode = TransmissionMode(coefficients, branch=branch)
    boundaries = np.r_[0., np.cumsum(mode.angles)]
    for right in range(3):
        theta = boundaries[right]
        point = np.array([[np.cos(theta), np.sin(theta)]])*.7
        normal = np.array([-np.sin(theta), np.cos(theta)])
        values, fluxes = [], []
        for side in ((right-1) % 3, right):
            angle = (boundaries[side]+boundaries[side+1])/2
            value, gradient = mode.sample(point, sector=side, center_angle=angle)
            values.append(value)
            fluxes.append(coefficients[side]*(gradient@normal))
        np.testing.assert_allclose(values[0], values[1], atol=3e-11, rtol=0)
        np.testing.assert_allclose(fluxes[0], fluxes[1], atol=3e-11, rtol=0)
    for sector in range(3):
        angle = (boundaries[sector]+boundaries[sector+1])/2
        theta = np.array([angle-.1, angle+.1])
        points = np.array([.2, .7])[:, None]*np.stack([np.cos(theta), np.sin(theta)], axis=1)
        _, gradient, hessian = mode.sample(points, sector=sector, center_angle=angle, hessian=True)
        for axis in range(2):
            step = 3e-5*np.eye(2)[axis]
            samples = [mode.sample(points+k*step, sector=sector, center_angle=angle)
                       for k in (-2, -1, 1, 2)]
            for slot, expected in ((0, gradient[:, axis]), (1, hessian[:, :, axis])):
                finite = (samples[0][slot]-8*samples[1][slot]+8*samples[2][slot]-samples[3][slot])/(12*3e-5)
                np.testing.assert_allclose(finite, expected, atol=3e-9, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('degree', (3, 4, 5, 6))
def test_complete_refined_field_transport_matches_spatial_differences(degree):
    cell = CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0, 1), 0,
        reference=REFERENCE, branches=2, degree=degree, angular_quadrature=32)
    fields = PairTransmissionFields(cell._base, GEOMETRY, REFERENCE, 2, 32)
    assert fields.count == 20
    _, _, points, velocity = cell._base._chart_geometry(0., GEOMETRY)
    bary = np.array([[.23, .31, .46], [.52, .17, .31]])
    db = np.array([[.07, -.02, -.05], [-.03, .01, .02]])
    for tri, thin in zip(cell._base.data['triangles'], cell._base.data['thin']):
        jacobian = np.column_stack([points[tri[1]]-points[tri[0]], points[tri[2]]-points[tri[0]]])
        if thin:
            adjugate = np.array([[jacobian[1, 1], -jacobian[0, 1]],
                                 [-jacobian[1, 0], jacobian[0, 0]]])
            gradient = fields.gradients(tri, bary, adjugate, np.zeros((2, 2)))[0]
            np.testing.assert_allclose(gradient, 0., atol=3e-10, rtol=0)
            continue
        change = np.column_stack([velocity[tri[1]]-velocity[tri[0]], velocity[tri[2]]-velocity[tri[0]]])
        inverse = np.linalg.inv(jacobian)
        actual = fields.gradients(tri, bary, inverse, -inverse@change@inverse, db)[1]
        # Sixth order permits a larger interval without truncation error,
        # resolving differences after the complete quartic coordinate map.
        h = 1e-3
        samples = [fields.gradients(tri, bary+k*h*db, np.linalg.inv(jacobian+k*h*change),
                                   np.zeros((2, 2)))[0] for k in (-3, -2, -1, 1, 2, 3)]
        finite = (-samples[0]+9*samples[1]-45*samples[2]+45*samples[3]-9*samples[4]+samples[5])/(60*h)
        np.testing.assert_allclose(actual, finite, atol=3e-9, rtol=0)


@lru_cache(None)
def closure(angular=64, scale=1., epsilon=(12.25, -10+1j), background=1.,
            gap_tangent=.7, epsilon_tangent=(.2+.1j, -.4+.2j), background_tangent=.1-.03j):
    cell = CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0), (1, 1),
        reference=REFERENCE, branches=2, angular_quadrature=angular, coercivity_angle=1.53)
    geometry = {key: np.asarray(value)*scale for key, value in GEOMETRY.items()}
    result = cell.closure(gap=0., **geometry, epsilon=epsilon, background=background,
                         gap_tangent=gap_tangent, epsilon_tangent=epsilon_tangent,
                         background_tangent=background_tangent)
    assert result[2]['transmission_directions'] == 20
    assert result[2]['transmission_reference'] == REFERENCE
    return tuple(np.asarray(a) for a in result[:2])


@pytest.mark.research
def test_full_joint_contact_reflection_and_quadrature():
    value, tangent = closure()
    modes = [(i, j) for j in range(-1, 2) for i in range(-1, 2)]
    reflected = [modes.index((i, -j)) for i, j in modes]
    for response, tolerance in ((value, 3e-10), (tangent, 3e-9)):
        for slot, sign in zip(response, (1, 1, 1, -1, -1)):
            np.testing.assert_allclose(slot, sign*slot[:, reflected][:, :, reflected], atol=tolerance, rtol=0)
    refined = closure(80)
    np.testing.assert_allclose(value, refined[0], atol=3e-10, rtol=0)
    np.testing.assert_allclose(tangent, refined[1], atol=3e-9, rtol=0)


@pytest.mark.research
def test_fixed_reference_preserves_physical_length_units():
    value, tangent = closure()
    scaled = closure(scale=700., gap_tangent=490.)
    np.testing.assert_allclose(value, scaled[0], atol=3e-10, rtol=0)
    np.testing.assert_allclose(tangent, scaled[1], atol=3e-9, rtol=0)


@pytest.mark.research
def test_flat_reference_adds_no_noninteger_direction():
    cell = CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0), 0,
        reference=(2., 2., 2.), angular_quadrature=12)
    value, tangent, metadata = cell.closure(gap=0., **GEOMETRY, epsilon=(3., 3.), background=3.)
    assert metadata['transmission_directions'] == 0
    expected = np.array([3., 3., 1/3, 0., 0.])[:, None, None, None]
    np.testing.assert_allclose(value, expected, atol=3e-10, rtol=0)
    np.testing.assert_allclose(tangent, 0., atol=3e-9, rtol=0)

    # The explicit public construction feeds the existing optical engine.
    settings = dict(lattice=GEOMETRY['basis'], truncation_order=0,
                    wavelength=2., thickness=(.3,), top=1., bot=2.25, theta=.2)
    auxiliary = _legacy_wavebit_rcwa(**settings).solve_conv(*value)
    uniform = _legacy_wavebit_rcwa(**settings, structure=np.full((1, 1, 1), 3.)).solve()
    for actual, reference in zip((auxiliary.p_inc, auxiliary.s_inc), (uniform.p_inc, uniform.s_inc)):
        np.testing.assert_allclose(actual.T, reference.T, atol=3e-10, rtol=0)
        np.testing.assert_allclose(actual.R, reference.R, atol=3e-10, rtol=0)


@pytest.mark.parametrize('method', ('li', 'nvf'))
def test_expert_example_leaves_auxiliary_off_and_refuses_unread_options(method, monkeypatch):
    from wavebit.rcwa import experimental
    from examples.rcwa.nvf_method_selection import solve_pair
    def forbidden(*args, **kwargs):
        raise AssertionError('ordinary selection must not construct the auxiliary cell')
    monkeypatch.setattr(experimental, 'CartesianPairTransmissionGalerkinGapCell', forbidden)
    answer, tangent, metadata, _ = solve_pair(method, order=1)
    assert tangent is None and metadata['factorization'] == method
    for result in answer.values():
        assert result.T.size == 9
        assert 0 <= result.T.sum() <= 1 and 0 <= result.R.sum() <= 1
        assert result.T.sum()+result.R.sum() <= 1+1e-9
    with pytest.raises(ValueError, match='auxiliary_options'):
        solve_pair(method, auxiliary_options={'degree': 5})


def test_expert_example_propagates_auxiliary_failure_without_fallback(monkeypatch):
    from wavebit.rcwa import experimental
    from examples.rcwa.nvf_method_selection import solve_pair
    def rejected(*args, **kwargs):
        raise ArithmeticError('auxiliary qualification rejected this case')
    monkeypatch.setattr(experimental, 'CartesianPairTransmissionGalerkinGapCell', rejected)
    with pytest.raises(ArithmeticError, match='qualification rejected'):
        solve_pair('auxiliary')
    with pytest.raises(ValueError, match='method must'):
        solve_pair('auto')


@pytest.mark.research
def test_positive_gap_and_material_jvp_matches_independent_values():
    cell = CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0), 0,
        reference=REFERENCE, branches=2, angular_quadrature=64, coercivity_angle=1.53)
    epsilon = np.array([4.+.1j, 9.+.2j])
    direction = np.array([.1+.02j, .2+.03j])
    background, dbg = 1.+.02j, .05+.01j
    def solve(step):
        return cell.closure(gap=.02+.7*step, **GEOMETRY, epsilon=epsilon+step*direction,
                            background=background+step*dbg, gap_tangent=.7,
                            epsilon_tangent=direction, background_tangent=dbg)
    value, tangent, metadata = solve(0.)
    assert metadata['transmission_reference'] == REFERENCE
    h = 3e-5
    samples = [np.asarray(solve(k*h)[0]) for k in (-2, -1, 1, 2)]
    finite = (samples[0]-8*samples[1]+8*samples[2]-samples[3])/(12*h)
    np.testing.assert_allclose(tangent, finite, atol=3e-9, rtol=0)


@pytest.mark.research
def test_metal_positive_gap_approaches_the_same_contact_derivative():
    cell = CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0), 0,
        reference=REFERENCE, branches=1, angular_quadrature=64, coercivity_angle=1.53)
    settings = dict(basis=np.eye(2)*700., widths=(300., 300.), height=300.,
                    epsilon=(12.25, -10+1j))
    value, tangent, _ = cell.closure(gap=0., **settings)
    value, tangent = np.asarray(value), np.asarray(tangent)
    errors = []
    for gap in (.01, .0001, .000001):
        opened, direction, _ = cell.closure(gap=gap, **settings)
        assert np.linalg.norm(np.asarray(opened)-value) < 2*gap*np.linalg.norm(tangent)
        errors.append(np.linalg.norm(np.asarray(direction)-tangent))
    assert errors[2] < errors[1] < errors[0]


@pytest.mark.research
@pytest.mark.parametrize('degree', (3,5))
def test_finite_bulk_mode_with_nonintegrable_opening_trace_is_refused(degree):
    reference=(2.,-.2+.1j,1.)
    mode=TransmissionMode(reference)
    assert 0<mode.exponent.real<.5
    # The mode has finite two-dimensional energy, but its junction trace
    # contributes integral(r**(2*Re(lambda)-2), dr) on each opening triangle.
    assert 2*mode.exponent.real-2 < -1
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        reference=reference,branches=4,degree=degree)
    with pytest.raises(ValueError,match='finite opening energy'):
        PairTransmissionFields(cell._base,GEOMETRY,reference,4,24)


@pytest.mark.research
def test_integrable_outer_mode_below_half_power_is_still_retained():
    reference=(12.25,-4+.1j,1.)
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        reference=reference,branches=1)
    fields=PairTransmissionFields(cell._base,GEOMETRY,reference,1,24)
    assert fields.count==10
    low=[corner for corner,_,mode,_ in fields.modes if mode.exponent.real<=.5]
    assert low and all(corner not in (5,8) for corner in low)
    for tri,thin in zip(cell._base.data['triangles'],cell._base.data['thin']):
        if thin:
            np.testing.assert_array_equal(fields.hats[tri[:3]][:,low],0.)


@pytest.mark.research
def test_exported_value_and_tangent_are_the_private_operators_about_the_centre():
    from wavebit.rcwa._engine.conforming.cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell
    from tests.fixtures.common._conv_frame import slots_about_centre
    cell = CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0), (1, 1),
        reference=(4., 9., 1.), branches=1, angular_quadrature=16)
    settings = dict(gap=0., basis=np.diag([1.2, .9]), widths=(.3, .4), height=.45,
                    epsilon=(4., 9.), offset=(.03, -.02), epsilon_tangent=(.1, -.2))
    exported = cell.closure(**settings)
    private = CartesianPairCornerGalerkinGapCell.closure(cell, **settings)
    for actual, original in zip(exported[:2], private[:2]):
        reference = slots_about_centre(original, (1, 1))
        assert any(a.tobytes() != b.tobytes() for a, b in zip(actual, original))
        for a, b in zip(actual, reference):
            assert a.tobytes() == b.tobytes()
