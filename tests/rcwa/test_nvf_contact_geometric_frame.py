"""Complete retained elimination, geometric coordinates and physical QR."""
import numpy as np
import pytest

from wavebit.rcwa._engine.conforming import pair_contact_precision as precision
from wavebit.rcwa._engine.conforming.projected_pair_contact import _real_geometry
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit.rcwa._engine.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell

pytestmark = pytest.mark.research


@pytest.mark.parametrize('degree,order', ((5, (1, 0)), (6, (0, 0))))
def test_complete_contact_eliminations_preserve_physical_qr_response(monkeypatch, degree, order):
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),order,
        reference=(12.25,-10+1j,1.),degree=degree,quadrature=20*degree,
        angular_quadrature=24,coercivity_angle=1.53)
    settings=dict(gap=0.,basis=np.eye(2)*700,widths=(300.,300.),height=300.,
        offset=(-350.,-350.),epsilon=(12.25,-10+1j),background=1.,gap_tangent=.7,
        epsilon_tangent=(.2+.1j,-.4+.2j),background_tangent=.1-.03j)
    actual=cell.closure(**settings)
    assert actual[2]['closed_elimination']=='physical_retained'
    assert actual[2]['transmission_directions']==20
    original=precision.response
    monkeypatch.setattr(precision,'response',
        lambda *args,**kwargs:original(*args,**kwargs,method='metric'))
    metric=cell.closure(**settings)
    assert metric[2]['common_geometric_frame']=='complete_metric'

    def sampled_qr(obj,settings,base,transform,chart,quadrature,T,Qbar,records,edge_rows,**kwargs):
        assert T is None
        nc=base[-1]['nc'];q=base[2][:,:nc]
        upper=np.zeros((0,nc))
        for row,local,weight,phi,dphi,added in records():
            if local[11]:continue
            ci,w,g=row[1],row[3],row[9]
            psi=-(phi@q);psi[:,:,ci]+=g
            sample=(np.sqrt(w)[:,None,None]*_real_geometry(psi)).reshape(-1,nc)
            upper=np.linalg.qr(np.vstack([upper,sample]),mode='r')
        T=np.linalg.solve(upper,np.eye(nc))
        return original(obj,settings,base,transform,chart,quadrature,T,Qbar,records,edge_rows,**kwargs)

    monkeypatch.setattr(precision,'response',sampled_qr)
    expected=cell.closure(**settings)
    for result in (actual,metric,expected):
        assert result[2]['energy_residual']<1e-10
        assert result[2]['energy_derivative_residual']<1e-10
        np.testing.assert_allclose(result[0][3:],0.,atol=3e-10,rtol=0)
        np.testing.assert_allclose(result[1][3:],0.,atol=3e-9,rtol=0)
    for result in (actual,metric):
        np.testing.assert_allclose(result[0],expected[0],atol=3e-12,rtol=0)
        np.testing.assert_allclose(result[1],expected[1],atol=3e-11,rtol=0)
