"""Higher polynomial fields against independent cardinal functions and quadrature."""
from math import factorial, prod
import numpy as np
import pytest

from wavebit._core.conforming.cartesian_pair_polynomial_galerkin import CartesianPairPolynomialGalerkinGapCell
from wavebit._core.conforming.cartesian_pair_galerkin import CartesianPairGalerkinGapCell
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.triangle_polynomials import phase_average
from wavebit._core.conforming.galerkin_error import estimate_closed_error, refine_closed_error, _closed_state, _dual_fields
from test_nvf_cartesian_pair_opening import _oracle_geometry
from test_nvf_pair_galerkin import nodal_reference, tensor
from test_nvf_pair_quadratic import duffy

pytestmark = pytest.mark.research

GEOMETRY = dict(basis=np.array([[700., 35.], [0., 560.]]), widths=(170., 230.),
                height=280., offset=(21., -14.))


def indices(p):
    corners = [(p,0,0), (0,p,0), (0,0,p)]
    return corners+[(i,j,p-i-j) for i in range(p+1) for j in range(p-i+1)
                    if (i,j,p-i-j) not in corners]


def cardinal_gradient(bary, degree):
    """Differentiate cardinal polynomials on the equispaced simplex nodes.

    Each product has a zero at every other lattice node and value one at
    its own node. This is a different basis from the candidate's Bernstein
    control net and uses neither its derivative matrix nor product moments.
    """
    result = []
    for alpha in indices(degree):
        values, derivatives = [], []
        for i,a in enumerate(alpha):
            factors = [(degree*bary[:,i]-j)/(a-j) for j in range(a)]
            values.append(np.prod(factors, axis=0) if a else np.ones(len(bary)))
            derivatives.append(sum((degree/(a-k)*np.prod([factors[j] for j in range(a) if j != k], axis=0)
                                    for k in range(a)), np.zeros(len(bary))))
        dl = [derivatives[i]*values[(i+1)%3]*values[(i+2)%3] for i in range(3)]
        result.append(np.column_stack([dl[1]-dl[0], dl[2]-dl[0]]))
    return np.stack(result, axis=1)


def bernstein(bary, degree):
    terms = indices(degree) if degree else [(0,0,0)]
    return np.array([factorial(degree)/prod(factorial(i) for i in a)*np.prod(bary**a,axis=1)
                     for a in terms]).T


@pytest.mark.parametrize('degree', range(6))
@pytest.mark.parametrize('gap', (0., 1e-12, .37))
def test_moving_polynomial_means_against_independent_duffy_integrals(degree, gap):
    points = np.array([[.2,.1],[.2+gap,.1],[.4,.9]])
    velocity = np.array([[-.5,0.],[.5,0.],[.5,0.]])
    waves = np.vstack([np.zeros((1,2)), [[0.,1.],[1e-13,1e-13],[2.,0.]],
                       np.random.default_rng(82).normal(size=(12,2))*25])
    actual, change = phase_average(points, velocity, waves, degree)
    bary, weight = duffy(64)
    functions = bernstein(bary, degree)
    phase = np.exp(-1j*(bary@points)@waves.T)
    expected = 2*functions.T@(weight[:,None]*phase)
    derivative = 2*functions.T@(weight[:,None]*phase*(-1j*(bary@velocity)@waves.T))
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=0)
    np.testing.assert_allclose(change, derivative, atol=3e-12, rtol=0)


@pytest.mark.parametrize('degree', (2,3,4))
@pytest.mark.parametrize('gap', (0.,7.))
@pytest.mark.parametrize('epsilon', ((4.,9.), (12.25,-10+1j)))
def test_complete_original_nodal_equations_and_contact_variation(degree, gap, epsilon):
    obj = CartesianPairPolynomialGalerkinGapCell(CartesianPairOpeningPlan(0), (1,0),
                                                degree=degree, coercivity_angle=1.53)
    length = np.sqrt(np.linalg.det(GEOMETRY['basis']))
    normalized = {k:np.asarray(v)/length for k,v in GEOMETRY.items()}
    graph = _oracle_geometry(obj.data, normalized)
    answers = [nodal_reference(graph, epsilon, gap, obj.order, quadrature, length,
        normalized['basis'], degree=degree, gradient_evaluator=lambda b:cardinal_gradient(b,degree))
        for quadrature in (24,32)]
    value, tangent, metadata = obj.closure(gap=gap, epsilon=epsilon, **GEOMETRY)
    np.testing.assert_allclose(answers[0][0], answers[1][0], atol=3e-10, rtol=0)
    np.testing.assert_allclose(tensor(value), answers[1][0], atol=3e-10, rtol=0)
    if gap == 0:
        np.testing.assert_allclose(length*answers[0][1], length*answers[1][1], atol=3e-9, rtol=0)
        np.testing.assert_allclose(length*tensor(tangent), length*answers[1][1], atol=3e-9, rtol=0)
    assert max(v for k,v in metadata.items() if k.endswith('residual')) < 1e-10


def test_bernstein_quadratic_space_matches_the_original_lagrange_space():
    plan = CartesianPairOpeningPlan(0,1)
    old = CartesianPairGalerkinGapCell(plan, (1,2), degree=2, coercivity_angle=1.53)
    new = CartesianPairPolynomialGalerkinGapCell(plan, (1,2), degree=2, coercivity_angle=1.53)
    for gap in (0.,7.,1e-9):
        a = old.closure(gap=gap, epsilon=(12.25,-10+1j), **GEOMETRY)
        b = new.closure(gap=gap, epsilon=(12.25,-10+1j), **GEOMETRY)
        np.testing.assert_allclose(a[0], b[0], atol=3e-10, rtol=0)
        np.testing.assert_allclose(a[1], b[1], atol=3e-9, rtol=0)


@pytest.mark.parametrize('gap', (0.,7.))
def test_complex_material_variations_remain_holomorphic_and_additive(gap):
    obj = CartesianPairPolynomialGalerkinGapCell(CartesianPairOpeningPlan(0), (1,1),
                                                degree=4, coercivity_angle=1.53)
    settings = dict(GEOMETRY,gap=gap,epsilon=np.array([12.25+.2j,-10+1j]),background=1+.1j)
    de,db = np.array([.3+.2j,-.4+.1j]),.2+.3j
    _,material,_ = obj.closure(**settings,gap_tangent=0.,epsilon_tangent=de,background_tangent=db)
    _,imaginary,_ = obj.closure(**settings,gap_tangent=0.,epsilon_tangent=1j*de,background_tangent=1j*db)
    _,opening,_ = obj.closure(**settings)
    _,combined,_ = obj.closure(**settings,gap_tangent=.7,epsilon_tangent=de,background_tangent=db)
    np.testing.assert_allclose(imaginary,1j*np.asarray(material),atol=3e-9,rtol=0)
    np.testing.assert_allclose(combined,.7*np.asarray(opening)+material,atol=3e-9,rtol=0)
    h=.001
    values = [np.asarray(obj.closure(**dict(settings,epsilon=settings['epsilon']+i*h*de,
                                          background=settings['background']+i*h*db))[0]) for i in (-2,-1,1,2)]
    expected=(values[0]-8*values[1]+8*values[2]-values[3])/(12*h)
    np.testing.assert_allclose(material,expected,atol=3e-9,rtol=0)


def test_uniform_medium_and_physical_unit_changes_at_subpixel_gap():
    obj = CartesianPairPolynomialGalerkinGapCell(CartesianPairOpeningPlan(0), (0,2),
                                                degree=4, coercivity_angle=1.53)
    eps=-10+1j
    value,change,_ = obj.closure(gap=1e-9,epsilon=(eps,eps),background=eps,**GEOMETRY)
    np.testing.assert_allclose(tensor(value),eps*np.eye(10),atol=3e-10,rtol=0)
    np.testing.assert_allclose(change,0.,atol=3e-10,rtol=0)
    a=obj.closure(gap=1e-9,epsilon=(12.25,eps),**GEOMETRY)
    scaled={k:np.asarray(v)*.001 for k,v in GEOMETRY.items()}
    b=obj.closure(gap=1e-12,epsilon=(12.25,eps),**scaled)
    np.testing.assert_allclose(a[0],b[0],atol=3e-10,rtol=0)
    np.testing.assert_allclose(a[1],.001*np.asarray(b[1]),atol=3e-9,rtol=0)


def test_nested_real_spaces_and_the_independent_energy_identity():
    estimates=[]
    for p in (2,3,4):
        obj=CartesianPairPolynomialGalerkinGapCell(CartesianPairOpeningPlan(0),(1,1),degree=p)
        result=estimate_closed_error(obj,epsilon=(4.,9.),**GEOMETRY)
        np.testing.assert_allclose(result.residual_gram,result.tensor-result.dual_tensor,atol=3e-10,rtol=0)
        estimates.append(result)
    for coarse,fine in zip(estimates,estimates[1:]):
        assert np.linalg.eigvalsh(coarse.tensor-fine.tensor)[0] >= -3e-10
        assert np.linalg.eigvalsh(fine.dual_tensor-coarse.dual_tensor)[0] >= -3e-10


@pytest.mark.parametrize('epsilon', ((4.,9.),(12.25,-10+1j)))
def test_error_gram_against_cardinal_physical_fields(epsilon):
    degree=4
    obj=CartesianPairPolynomialGalerkinGapCell(CartesianPairOpeningPlan(0),(1,0),degree=degree,coercivity_angle=1.53)
    args=dict(GEOMETRY,epsilon=np.asarray(epsilon),background=1.)
    primal=_closed_state(obj,args)
    surrogate=_closed_state(obj,dict(args,epsilon=1/np.conj(epsilon)))
    _,xd,ud,_=_dual_fields(primal,surrogate)
    length=np.sqrt(np.linalg.det(GEOMETRY['basis']))
    graph=_oracle_geometry(obj.data,{k:np.asarray(v)/length for k,v in GEOMETRY.items()})
    pts=np.asarray(graph['vertices0'],float)
    nodes=np.asarray(indices(degree))/degree
    transform=bernstein(nodes,degree)
    collapse=np.eye(max(graph['pmap'])+1)[graph['pmap']][:,1:]
    wave=2*np.pi*np.array([[-1,0],[0,0],[1,0]])@np.linalg.inv(GEOMETRY['basis']/length)
    results=[]
    for quadrature in (24,32):
        bary,weights=duffy(quadrature);gram=np.zeros((6,6),complex)
        for tri,tag,thin in zip(graph['triangles'],graph['tags'],graph['thin']):
            if thin:continue
            v=pts[tri[:3]];jac=np.column_stack([v[1]-v[0],v[2]-v[0]])
            gradient=cardinal_gradient(bary,degree)@np.linalg.inv(jac)
            c=transform@collapse[graph['periodic'][tri]]
            up=np.einsum('qnd,ni->qdi',gradient,c@primal['potential'][:c.shape[1]])
            dp=np.einsum('qnd,ni->qdi',gradient,c@ud[:c.shape[1]])
            phase=np.exp(1j*(bary@v)@wave.T)
            phi=np.zeros((len(bary),2,6),complex);phi[:,0,:3]=phase;phi[:,1,3:]=phase
            eps=[1.,*epsilon][tag]
            residual=eps*(phi@primal['field']+up)-(phi@xd+np.stack([-dp[:,1],dp[:,0]],axis=1))
            gram+=np.einsum('q,qdi,qdj->ij',weights*np.linalg.det(jac)/np.real(obj._phase*eps),
                            residual.conj(),residual,optimize=True)
        results.append(gram)
    result=estimate_closed_error(obj,**args);scale=max(1.,np.linalg.norm(results[-1]))
    np.testing.assert_allclose(results[0]/scale,results[1]/scale,atol=3e-11,rtol=0)
    np.testing.assert_allclose(result.residual_gram/scale,results[1]/scale,atol=3e-11,rtol=0)


def test_polynomial_refinement_uses_its_declared_resource_range():
    obj=CartesianPairPolynomialGalerkinGapCell(CartesianPairOpeningPlan(0),(0,0),degree=4)
    # Uniform medium accepts immediately; no expensive default-cap run needed.
    result=refine_closed_error(obj,absolute_tolerance=1e-8,epsilon=(4.,4.),background=4.,**GEOMETRY)
    assert result.converged and result.refinement == 0
    with pytest.raises(ValueError,match='supported range'):
        refine_closed_error(obj,absolute_tolerance=1e-8,max_refinement=3,epsilon=(4.,4.),**GEOMETRY)


@pytest.mark.parametrize('degree,refinement', ((1,0),(5,0),(6,0),(True,0),(3.5,0),(4,3)))
def test_invalid_or_oversized_polynomial_requests_are_refused(degree,refinement):
    with pytest.raises(ValueError):
        CartesianPairPolynomialGalerkinGapCell(CartesianPairOpeningPlan(0,refinement),degree=degree)
