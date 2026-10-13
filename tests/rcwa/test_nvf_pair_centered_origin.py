"""Independent physical charts and full Fourier covariance of the same cell."""
import numpy as np
import pytest
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit.rcwa._engine.conforming.cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell

pytestmark = pytest.mark.research


@pytest.mark.parametrize('gap,degree,order,exponents,oscillatory,epsilon',[
    (0.,3,(2,1),(.6,.8),False,(12.25,-10+1j)),
    (.017,4,(1,2),(),False,(4.,9.)),
    (0.,3,(1,1),(.55,.6,.8,.95),False,(4.,9.)),
    # Full positive-gap oscillatory translations are also checked separately;
    # their two physical solves exceed the per-test CI time budget.
    (0.,3,(1,0),(.6,.8),True,(4.,9.)),
])
def test_centered_response_matches_direct_translated_physical_equations(
        gap,degree,order,exponents,oscillatory,epsilon):
    cell=CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),order,
        degree=degree,quadrature=48,exponents=exponents,junctions=True,
        coercivity_angle=1.53,mesh='symmetric',angular_quadrature=12,
        oscillatory_radial=oscillatory)
    settings=dict(gap=gap,basis=np.diag([1.2,.9]),widths=(.3,.4),height=.45,
        offset=(.17,-.08),epsilon=epsilon,background=1.,gap_tangent=.7,
        epsilon_tangent=(.2+.1j,-.4+.2j),background_tangent=.1-.03j)
    value,tangent,metadata=cell.closure(**settings)
    # Assemble separate physical vertices, Fourier phases, weak matrices and
    # independent energy integrals at the caller's origin. This reference
    # never applies the Fourier translation used by the public response.
    direct=cell._closure_at_offset(settings)
    for actual,expected,tolerance in zip((value,tangent),direct[:2],(3e-10,3e-9)):
        np.testing.assert_allclose(actual,expected,atol=tolerance,rtol=0)
    assert metadata['coordinate_origin']=='cell_center'
    assert type(value) is type(direct[0])


def test_quartic_positive_metal_preserves_complete_joint_reflection():
    # The uncentered physical calculation is not an accurate derivative
    # reference for this resonant fixture (3.19e-9 reflection defect).
    # Use the independent physical symmetry, at its unchanged 3e-9 limit.
    cell=CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),(1,2),
        degree=4,quadrature=48,exponents=(),junctions=True,
        coercivity_angle=1.53,mesh='symmetric',angular_quadrature=12)
    value,tangent,_=cell.closure(gap=.017,basis=np.diag([1.2,.9]),widths=(.3,.4),
        height=.45,offset=(.17,-.08),epsilon=(12.25,-10+1j),gap_tangent=.7,
        epsilon_tangent=(.2+.1j,-.4+.2j),background_tangent=.1-.03j)
    modes=np.array([(i,j) for j in range(-2,3) for i in range(-1,2)])
    lookup={tuple(m):i for i,m in enumerate(modes)}
    reflected=np.array([lookup[i,-j] for i,j in modes])
    phase=np.exp(4j*np.pi*(modes[:,1][None,:]-modes[:,1][:,None])*(-.08/.9))
    for array,tolerance in ((value,3e-10),(tangent,3e-9)):
        for slot,sign in zip(array,(1,1,1,-1,-1)):
            np.testing.assert_allclose(slot,sign*phase*slot[:,reflected][:,:,reflected],
                                       atol=tolerance,rtol=0)


@pytest.mark.parametrize('change,match',[
    ({'offset':(0.,np.nan)},'offset'),
    ({'offset':(0.,)},'offset'),
    ({'basis':np.array([[1.,.1],[0.,1.]])},'positive diagonal'),
    ({'widths':(.6,.6)},'strictly contained'),
    ({'gap':-.01},'nonnegative'),
])
def test_centering_preserves_original_geometry_rejection(change,match):
    cell=CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        mesh='symmetric',exponents=())
    settings=dict(gap=0.,basis=np.eye(2),widths=(.3,.4),height=.45,epsilon=(4.,9.))
    with pytest.raises(ValueError,match=match):cell.closure(**(settings|change))
