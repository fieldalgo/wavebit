"""The physical contact verifier must reject independent response corruption."""
import numpy as np
import pytest
from wavebit._core.conforming import pair_contact_precision as precision
from wavebit._core.conforming import pair_corner_contact as contact
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell

pytestmark = pytest.mark.research


@pytest.mark.parametrize('target,metric',(
    ('value','energy_residual'),
    ('tangent','energy_derivative_residual'),
    ('potential','energy_residual'),
    ('retained_field','retained_residual'),
))
def test_complete_pointwise_contact_energy_rejects_corruption(monkeypatch,target,metric):
    response=precision.response
    check=contact._check_response
    observed=[]
    def corrupt(*args,**kwargs):
        result=list(response(*args,**kwargs))
        if target in ('potential','retained_field'):
            # Alter only the physical verifier's potential; its candidate
            # response and original-equation coefficients remain untouched.
            field,change,potential=kwargs['assembled']['energy_coordinates']
            if target=='potential':
                potential=tuple(v.copy() for v in potential)
                potential[0][0,0]+=.05
            else:
                field=tuple(v.copy() for v in field)
                field[0][:]+=.05*np.eye(len(field[0]))
            kwargs['assembled']['energy_coordinates']=(field,change,potential)
        else:
            index=4 if target=='value' else 5
            result[index]=result[index]+.01*np.eye(len(result[index]))
        return tuple(result)
    def record(value,tangent,metadata):
        observed.append(metadata)
        return check(value,tangent,metadata)
    monkeypatch.setattr(precision,'response',corrupt)
    monkeypatch.setattr(contact,'_check_response',record)
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        reference=(2.,2.,2.),degree=5,quadrature=48,angular_quadrature=12)
    with pytest.raises(ValueError,match='original/energy/physical guards'):
        cell.closure(gap=0.,basis=np.eye(2),widths=(.3,.3),height=.4,
                     offset=(0.,0.),epsilon=(4.,9.),background=1.)
    assert observed[0][metric]>1e-10
    if target in ('potential','retained_field'):
        assert observed[0]['weak_residual']<1e-10
        assert observed[0]['rounded_retained_defect']<1e-10
        assert observed[0]['flux_residual']<1e-10


def test_contact_mean_variation_rejects_energy_invisible_skew(monkeypatch):
    """A skew field variation cancels in uniform energy, but not in its mean."""
    response=precision.response;check=contact._check_response;observed=[]
    def corrupt(*args,**kwargs):
        result=response(*args,**kwargs)
        field,change,potential=kwargs['assembled']['energy_coordinates']
        change=tuple(v.copy() for v in change)
        change[0][:]+=.01*np.array([[0.,1.],[-1.,0.]])
        kwargs['assembled']['energy_coordinates']=(field,change,potential)
        return result
    def record(value,tangent,metadata):
        observed.append(metadata)
        return check(value,tangent,metadata)
    monkeypatch.setattr(precision,'response',corrupt)
    monkeypatch.setattr(contact,'_check_response',record)
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        reference=(2.,2.,2.),degree=5,quadrature=48,angular_quadrature=12)
    with pytest.raises(ValueError,match='physical_retained_derivative_residual'):
        cell.closure(gap=0.,basis=np.eye(2),widths=(.3,.3),height=.4,
                     offset=(0.,0.),epsilon=(2.,2.),background=2.,gap_tangent=.7)
    for key in ('energy_residual','energy_derivative_residual','weak_residual',
                'retained_residual','retained_derivative_residual','flux_residual'):
        assert observed[0][key]<1e-10
    assert observed[0]['physical_retained_derivative_residual']>1e-3
