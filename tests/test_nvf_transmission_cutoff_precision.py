"""Independent complete moving potentials for every nested cutoff direction."""
import numpy as np
import pytest
from test_nvf_transmission_moving_precision import check_moving_potential
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell


@pytest.mark.parametrize('degree,refinement',((5,0),(6,0),(5,1)))
@pytest.mark.parametrize('cutoff_order',(1,2))
@pytest.mark.parametrize('exact_points',(False,True))
def test_nested_cutoff_moving_gradients_match_decimal_potential(degree,refinement,cutoff_order,exact_points):
    check_moving_potential(degree,refinement,cutoff_order,exact_points)


@pytest.mark.parametrize('cutoff_order',(True,False,-1,3,1.,np.nan))
def test_cutoff_order_rejects_invalid_values(cutoff_order):
    with pytest.raises(ValueError,match='cutoff_order'):
        CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),0,
            reference=(12.25,-10+1j,1.),degree=5,cutoff_order=cutoff_order)


@pytest.mark.parametrize('degree',(3,4))
def test_cutoff_order_requires_complete_high_degree_coordinates(degree):
    with pytest.raises(ValueError,match='degree 5 through 8'):
        CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),0,
            reference=(12.25,-10+1j,1.),degree=degree,cutoff_order=1)


@pytest.mark.parametrize('degree,refinement',((7,0),(8,0),(7,1),(8,1)))
@pytest.mark.parametrize('cutoff_order',(0,1,2))
def test_higher_degree_moving_potentials_retain_all_complete_directions(degree,refinement,cutoff_order):
    check_moving_potential(degree,refinement,cutoff_order,exact_points=True)


def test_flat_reference_has_no_spurious_nested_directions():
    from wavebit._core.conforming.pair_transmission_fields import PairTransmissionFields
    from wavebit._core.conforming.pair_transmission_common_precision import CommonCoordinates
    from test_nvf_pair_transmission import GEOMETRY
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        reference=(2.,2.,2.),degree=5,cutoff_order=2)
    fields=PairTransmissionFields(cell._base,GEOMETRY,(2.,2.,2.),2,12,2)
    assert fields.count==0
    assert fields.cutoff_powers==[]
    assert fields.common_coordinate_type is CommonCoordinates


@pytest.mark.parametrize('cutoff_order',(1,2))
def test_nested_space_contains_every_original_potential(cutoff_order):
    from wavebit._core.conforming.pair_transmission_fields import PairTransmissionFields
    from test_nvf_pair_transmission import GEOMETRY,REFERENCE
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        reference=REFERENCE,degree=5,cutoff_order=cutoff_order)
    original=PairTransmissionFields(cell._base,GEOMETRY,REFERENCE,2,32)
    extended=PairTransmissionFields(cell._base,GEOMETRY,REFERENCE,2,32,cutoff_order)
    assert extended.count==(cutoff_order+1)*original.count
    original_columns=[];first=0
    for (_,_,_,parts),power in zip(extended.modes,extended.cutoff_powers):
        if power==0:original_columns.extend(range(first,first+len(parts)))
        first+=len(parts)
    assert len(original_columns)==original.count
    bary=np.array([[.23,.31,.46],[.52,.17,.31]])
    delta=np.array([[.04,-.01,-.03],[-.03,.02,.01]])
    inverse=np.array([[1.25,.25],[.125,.875]])
    dinverse=np.array([[.03125,.0625],[-.015625,.03125]])
    for tri in cell._base.data['triangles'][[0,7,16,-1]]:
        expected=original._sample(tri,bary,inverse,dinverse,delta)
        actual=extended._sample(tri,bary,inverse,dinverse,delta)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a[...,original_columns],b)


def test_nested_cutoff_contact_preserves_original_physical_and_reflection_guards():
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),(1,0),
        reference=(12.25,-10+1j,1.),branches=2,degree=5,cutoff_order=1,
        quadrature=100,angular_quadrature=32,coercivity_angle=1.53)
    value,tangent,metadata=cell.closure(gap=0.,basis=np.eye(2)*700,
        widths=(300.,300.),height=300.,offset=(-350.,-350.),
        epsilon=(12.25,-10+1j),background=1.,gap_tangent=.7,
        epsilon_tangent=(.2+.1j,-.4+.2j),background_tangent=.1-.03j)
    assert metadata['transmission_directions']==40
    assert metadata['transmission_cutoff_order']==1
    assert metadata['retained_verification']=='physical_field_integral'
    for key in ('energy_residual','energy_derivative_residual','retained_residual'):
        assert metadata[key]<1e-10
    for response,tolerance in ((value,3e-10),(tangent,3e-9)):
        for slot in response[3:]:np.testing.assert_allclose(slot,0.,atol=tolerance,rtol=0)


@pytest.mark.parametrize('degree,order',((7,(0,1)),(8,(1,0))))
def test_higher_degree_contact_preserves_complete_joint_physical_response(degree,order):
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),order,
        reference=(12.25,-10+1j,1.),branches=2,degree=degree,
        quadrature=20*degree,angular_quadrature=48,coercivity_angle=1.53)
    value,tangent,metadata=cell.closure(gap=0.,basis=np.eye(2)*700,
        widths=(300.,300.),height=300.,offset=(-350.,-350.),
        epsilon=(12.25,-10+1j),background=1.,gap_tangent=.7,
        epsilon_tangent=(.2+.1j,-.4+.2j),background_tangent=.1-.03j)
    assert metadata['transmission_directions']==20
    assert metadata['retained_verification']=='physical_field_integral'
    for key in ('energy_residual','energy_derivative_residual','retained_residual',
                'physical_retained_derivative_residual'):
        assert metadata[key]<1e-10
    reflection=np.array([2,1,0] if order[1] else [0,1,2])
    for response,tolerance in ((value,3e-10),(tangent,3e-9)):
        for slot,sign in zip(response,(1,1,1,-1,-1)):
            np.testing.assert_allclose(slot,sign*slot[:,reflection][:,:,reflection],atol=tolerance,rtol=0)


# The full positive-gap space retains its 140-point radial rule and all
# physical fields; allow the same bounded deadline as the nested control.
@pytest.mark.timeout(3600)
def test_degree_seven_positive_gap_retains_joint_energy_and_fourier_mean():
    from test_nvf_pair_transmission import GEOMETRY,REFERENCE
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        reference=REFERENCE,branches=2,degree=7,quadrature=140,
        angular_quadrature=32,coercivity_angle=1.53)
    value,tangent,metadata=cell.closure(gap=.02,**GEOMETRY,
        epsilon=(4.+.1j,9.+.2j),background=1.+.02j,gap_tangent=.7,
        epsilon_tangent=(.1+.02j,.2+.03j),background_tangent=.05+.01j)
    assert metadata['transmission_directions']==20
    assert metadata['retained_verification']=='physical_field_integral'
    for key in ('energy_residual','energy_derivative_residual','retained_residual'):
        assert metadata[key]<1e-10
    for response,tolerance in ((value,3e-10),(tangent,3e-9)):
        for slot in response[3:]:np.testing.assert_allclose(slot,0.,atol=tolerance,rtol=0)


# The full independent three-word positive-gap control took 1,699 seconds
# before shared-mode sampling. Keep a bounded deadline for slower CI hosts.
@pytest.mark.timeout(3600)
def test_nested_cutoff_positive_gap_keeps_joint_physical_energy_and_mean():
    from test_nvf_pair_transmission import GEOMETRY,REFERENCE
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        reference=REFERENCE,branches=2,degree=5,cutoff_order=1,
        quadrature=100,angular_quadrature=32,coercivity_angle=1.53)
    value,tangent,metadata=cell.closure(gap=.02,**GEOMETRY,
        epsilon=(4.+.1j,9.+.2j),background=1.+.02j,gap_tangent=.7,
        epsilon_tangent=(.1+.02j,.2+.03j),background_tangent=.05+.01j)
    assert metadata['transmission_directions']==40
    assert metadata['transmission_cutoff_order']==1
    assert metadata['retained_verification']=='physical_field_integral'
    for key in ('energy_residual','energy_derivative_residual','retained_residual'):
        assert metadata[key]<1e-10
    for response,tolerance in ((value,3e-10),(tangent,3e-9)):
        for slot in response[3:]:np.testing.assert_allclose(slot,0.,atol=tolerance,rtol=0)
