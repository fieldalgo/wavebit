"""Original mixed equations and pointwise weak tests of complete refinement."""
import numpy as np
import pytest
from wavebit._core.conforming.pair_positive_refinement import correct,physical_constraints,refine
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
import wavebit._core.conforming.cartesian_pair_corner_galerkin as corner


def _mixed_problem(s,ds):
    rng=np.random.default_rng(137)
    draw=lambda shape:.1*(rng.normal(size=shape)+.2j*rng.normal(size=shape))
    K=np.array([[3+.2j,.2,.1*s],[.2,2+.3j,.2*s],[.1,.2,1+.05j]])
    dK=draw((3,3));dK[:2,2]=ds*K[2,:2]+s*dK[2,:2]
    rawq,rawr=draw((2,3)),draw((2,3))
    drawq,drawr=draw((2,3)),draw((2,3))
    weight=np.array([1.,1.,s]);dweight=np.array([0.,0.,ds])
    Q=rawq*weight;dQ=drawq*weight+rawq*dweight
    R=rawr*weight;dR=drawr*weight+rawr*dweight
    QT=rawq.conj().T;dQT=drawq.conj().T
    S=rawr[::-1].T;dS=drawr[::-1].T
    A=np.diag([1.6+.1j,2.1+.2j]);dA=draw((2,2))
    # Independent mixed solve: potentials, retained field and physical flux
    # are all unknowns. No projection or balanced elimination is used here.
    identity=np.eye(2);zero=np.zeros((2,2))
    mixed=np.block([[K,S,-QT],[Q,identity,zero],[-R,-A,identity]])
    dmixed=np.block([[dK,dS,-dQT],[dQ,zero,zero],[-dR,-dA,zero]])
    value=np.linalg.solve(mixed,np.vstack([np.zeros((3,2)),identity,zero]))
    tangent=np.linalg.solve(mixed,-dmixed@value)
    U,x,H=np.split(value,[3,5]);DU,dx,DH=np.split(tangent,[3,5])
    exact=(U,DU,x,dx,H,DH)
    initial=tuple(v+draw(v.shape)*.03 for v in exact)
    coordinate=np.array([[1.,.2,0.],[-.3,1.,0.],[0.,0.,1.4]])
    scale=np.array([1.,1.,np.sqrt(s)])
    metric=coordinate.T@(K-S@Q-QT@(R-A@Q))@coordinate
    metric=scale[:,None]*metric/scale[None,:]
    blocks=(K,dK,R,dR,QT,dQT,S,dS,A,dA)
    info=dict(nc=2,s=s,ds=ds,reverse=np.array([1,0]))
    return blocks,Q,dQ,coordinate,metric,info,initial,exact


@pytest.mark.parametrize('gap',[.003,1e-7,1e-12])
@pytest.mark.parametrize('direction',[0.,1/700,1.])
def test_all_value_and_derivative_variables_match_independent_mixed_system(gap,direction):
    blocks,Q,dQ,coordinate,metric,info,state,expected=_mixed_problem(gap,direction)
    K,dK,R,dR,QT,dQT,S,dS,A,dA=blocks
    U,DU,x,dx,H,DH=state
    weak=K@U+S@x-QT@H
    variation=dK@U+K@DU+dS@x+S@dx-dQT@H-QT@DH
    actual=correct(blocks,Q,dQ,coordinate,metric,info,state,weak,variation)
    for value,reference in zip(actual,expected):
        np.testing.assert_allclose(value,reference,atol=3e-14,rtol=0)


def test_resolved_original_equations_do_not_request_physical_reintegration():
    blocks,Q,dQ,coordinate,metric,info,_,state=_mixed_problem(.003,1.)
    def unexpected():raise AssertionError('resolved state must not request new quadrature')
    result=refine(None,None,(info,),None,unexpected,blocks,Q,dQ,coordinate,metric,state)
    assert result is state
    assert info['pointwise_refinement_steps']==0


@pytest.mark.parametrize('gap',[.003,1e-9])
def test_pointwise_normalized_constraints_match_independently_assembled_blocks(monkeypatch,gap):
    captured={}
    class Captured(Exception):pass
    def capture(obj,settings,base,order,quadrature,free,Q,dQ,blocks):
        captured.update(obj=obj,settings=settings,base=base,order=order,
                        quadrature=quadrature,Q=Q,dQ=dQ,blocks=blocks)
        raise Captured
    monkeypatch.setattr(corner,'solve_positive',capture)
    cell=corner.CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        degree=3,quadrature=32,junctions=True)
    with pytest.raises(Captured):
        cell.closure(gap=gap,basis=np.eye(2),widths=(.3,.4),height=.45,
                     offset=(.01,-.03),epsilon=(4+.2j,9+.3j),gap_tangent=.7,
                     epsilon_tangent=(.1+.2j,-.2+.1j),background_tangent=.03+.02j)
    K,dK,R,dR,QT,dQT,S,dS,A,dA=captured['blocks']
    rng=np.random.default_rng(38)
    draw=lambda shape:.001*(rng.normal(size=shape)+1j*rng.normal(size=shape))
    U,DU=draw((len(K),2)),draw((len(K),2))
    x,dx,H,DH=np.eye(2)+draw((2,2)),draw((2,2)),np.eye(2)*5+draw((2,2)),draw((2,2))
    state=(U,DU,x,dx,H,DH)
    weak,variation=physical_constraints(*(captured[k] for k in
        ('obj','settings','base','order','quadrature')),state)
    expected=K@U+S@x-QT@H
    dexpected=dK@U+K@DU+dS@x+S@dx-dQT@H-QT@DH
    for actual,reference in ((weak,expected),(variation,dexpected)):
        assert np.linalg.norm(actual-reference)/max(1.,np.linalg.norm(reference))<3e-11


def test_high_order_metal_value_and_gap_jvp_pass_all_original_physical_guards():
    # Keep the complete N4/refinement-1/angular-2 potential space at the
    # default quadrature. Its separate q64 control passes all original
    # guards but took 20 minutes, beyond the per-test CI budget.
    cell=corner.CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0,1),4,
        degree=3,quadrature=48,junctions=True,angular_order=2,coercivity_angle=1.53)
    value,tangent,metadata=cell.closure(gap=.003,basis=np.eye(2),widths=(.3,.4),
        height=.45,offset=(.01,-.03),epsilon=(12.25,-10+1j))
    assert metadata['pointwise_refinement_steps']>0
    assert metadata['energy_residual']<1e-10
    assert metadata['energy_derivative_residual']<1e-10
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(tangent))
