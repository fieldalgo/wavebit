"""Exact mesh reflection and complete constrained pair-response controls."""
from collections import Counter
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import pytest

from wavebit._core.conforming.cartesian_pair import _physical_geometry
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan, canonical
from wavebit._core.conforming.cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell
from wavebit._core.conforming.pair_symmetric_mesh import refined, topology
from wavebit._core.conforming.pair_corner_fields import PairCornerFields
from wavebit._core.conforming.pair_corner_prefactors import selection
from wavebit._core.conforming.triangle_polynomials import values_and_tangent,gradient_coefficients
from wavebit._core.conforming.pair_contact_refinement import refine


@pytest.mark.parametrize('refinement',[0,1,2])
@pytest.mark.parametrize('contact',[False,True])
def test_exact_reflections_partition_and_original_contact_graph(refinement,contact):
    weights,triangles,tags,periodic=refined(contact,0,refinement)
    nx=3 if contact else 4
    lookup={tuple(row):i for i,row in enumerate(weights)}
    expected=Counter(canonical(row,tag) for row,tag in zip(triangles,tags))
    y=np.r_[[3,2,1,0],np.arange(4,4+2*nx)[::-1]]
    x=np.r_[[1,0,3,2],np.arange(4,4+nx)[::-1],np.arange(4+nx,4+2*nx)[::-1]]
    for controls,materials in ((y,tags),(x,np.array([0,2,1])[tags])):
        permutation=np.array([lookup[tuple(row[controls])] for row in weights])
        actual=Counter(canonical(row[[0,2,1]],tag)
                       for row,tag in zip(permutation[triangles],materials))
        assert actual==expected
    edges=Counter(tuple(sorted((int(a),int(b)))) for cell in triangles
                  for a,b in zip(cell,np.roll(cell,-1)))
    assert max(edges.values())==2
    assert all(not np.any(weights[[a,b],4:]) for (a,b),count in edges.items() if count==1)
    assert len(np.unique(triangles))==len(weights)
    control,_=_physical_geometry(basis=np.eye(2),widths=(.3,.4),height=.45,
                                 gap=0. if contact else .017,offset=(.01,-.03))
    points=weights@control
    area=np.linalg.det(np.stack([points[triangles[:,1]]-points[triangles[:,0]],
                                 points[triangles[:,2]]-points[triangles[:,0]]],axis=-1))/2
    assert np.all(area>0)
    np.testing.assert_allclose([sum(area),sum(area[tags==1]),sum(area[tags==2])],
                               [1.,.3*.45,.4*.45],rtol=0,atol=3e-14)
    data=topology(CartesianPairOpeningPlan(0,refinement))
    assert sum(data['thin'])==12*4**refinement
    assert len(data['triangles'])==60*4**refinement


@pytest.mark.parametrize('mesh',[None,True,1,[],{},'mirror'])
def test_mesh_selector_rejects_invalid_values(mesh):
    with pytest.raises(ValueError,match='corner mesh'):
        CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),mesh=mesh)


def test_symmetric_mesh_requires_an_explicit_shift_zero_plan():
    with pytest.raises(ValueError,match='explicit CartesianPairOpeningPlan'):
        CartesianPairCornerGalerkinGapCell(None,mesh='symmetric')
    with pytest.raises(ValueError,match='shift zero'):
        CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(1),mesh='symmetric')


@pytest.mark.parametrize('gap',[0.,.003])
def test_diagonal_normalization_preserves_exact_caller_unit_charts(gap):
    cell=_cell(exponents=())
    charts=[]
    for scale in (1.,700.):
        settings=_settings(gap*scale,scale)
        geometry={key:np.asarray(settings[key]) for key in ('basis','widths','height','offset')}
        length,*chart=cell._base._chart_geometry(settings['gap'],geometry)
        assert length==scale
        charts.append(chart)
    for a,b in zip(*charts):np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='positive diagonal'):
        cell._base._normalization_length(np.array([[1.,.1],[0.,1.]]))


@pytest.mark.parametrize('degree',[3,4])
@pytest.mark.parametrize('angular',[0,1,2,3,4])
def test_exact_prefactors_preserve_original_columns_and_remove_only_redundancy(degree,angular):
    original=CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        degree=degree,junctions=True,angular_order=angular)
    fields=PairCornerFields(original._base,junctions=True,angular_order=angular)
    assert selection(fields)[1]==[]
    symmetric=CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        degree=degree,junctions=True,angular_order=angular,mesh='symmetric')
    # Recreate all candidates on this mesh, before the exact quotient.
    raw=PairCornerFields(SimpleNamespace(data=symmetric._base.data,degree=degree),
                         junctions=True,angular_order=angular)
    reduced=PairCornerFields(symmetric._base,junctions=True,angular_order=angular)
    kept,removed=selection(raw)
    assert reduced.count==2*len(kept)
    assert reduced.count<raw.count
    assert selection(reduced)[1]==[]
    # This relation follows directly from equal outer hats on the support
    # of the lower junction. Check actual potentials and both gradients.
    a=raw.powers.index((5,0,.6));b=raw.powers.index((5,1,.6))
    bary=np.array([[.17,.31,.52],[.3,.2,.5]])
    dbary=np.array([[.02,-.03,.01],[-.01,.04,-.03]])
    for tri in symmetric._base.data['triangles']:
        for sampled in raw.sample(tri,bary,np.array([[1.2,.1],[-.2,.8]]),
                                  np.array([[.03,.01],[-.04,.02]]),dbary):
            np.testing.assert_allclose(sampled[...,a],sampled[...,b],rtol=0,atol=1e-14)
    assert (5,(1,),0) in removed
    # Reconstruction for the real trace projection must retain the mesh.
    reconstructed=type(symmetric._base)(symmetric._base.plan,0,degree=degree)
    np.testing.assert_array_equal(reconstructed.data['open_numerators'],symmetric._base.data['open_numerators'])


def _cell(*,degree=3,quadrature=48,exponents=(.6,.8),angular_order=0,order=0):
    return CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),order,
        degree=degree,quadrature=quadrature,exponents=exponents,junctions=True,
        angular_order=angular_order,coercivity_angle=1.53,mesh='symmetric')


def _settings(gap=0.,scale=700.):
    return dict(gap=gap,basis=np.eye(2)*scale,widths=np.array([.3,.4])*scale,
                height=.45*scale,offset=np.array([.01,-.03])*scale,
                epsilon=(12.25,-10+1j))


@lru_cache(None)
def _closure(gap=0.,scale=700.,degree=3,quadrature=48,exponents=(.6,.8),
             angular_order=0,order=0,gap_tangent=1.,material_direction=0.):
    settings=_settings(gap,scale)
    cell=_cell(degree=degree,quadrature=quadrature,exponents=exponents,
               angular_order=angular_order,order=order)
    return cell.closure(**dict(settings,gap_tangent=gap_tangent,
        epsilon_tangent=material_direction*np.asarray(settings['epsilon']),
        background_tangent=material_direction))


@pytest.mark.parametrize('gap',[0.,2.1])
@pytest.mark.parametrize('angular_order',[0,4])
def test_complete_metal_response_and_gap_jvp_preserve_reflection(gap,angular_order):
    value,tangent,metadata=_closure(gap,angular_order=angular_order)
    assert metadata['mesh']=='symmetric'
    np.testing.assert_allclose([value.epxy,value.epyx],0.,atol=3e-10,rtol=0)
    np.testing.assert_allclose([tangent.epxy,tangent.epyx],0.,atol=3e-9,rtol=0)


def test_high_order_positive_metal_retains_independent_physical_energy():
    # Raw fifth-root coordinates failed the original energy guards in the
    # separate q64 control of this full N4/angular-2 space. Keep every
    # potential at the default quadrature within the per-test CI time limit.
    value,tangent,metadata=_closure(gap=.003,scale=1.,angular_order=2,order=4)
    assert metadata['energy_residual']<1e-10
    assert metadata['energy_derivative_residual']<1e-10
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(tangent))


def test_positive_physical_gap_variation_matches_two_fourth_order_differences():
    value,tangent,metadata=_closure(2.1)
    for step in (.01,.005):
        samples=[np.asarray(_closure(2.1+k*step)[0])
                 for k in (-2,-1,1,2)]
        difference=(samples[0]-8*samples[1]+8*samples[2]-samples[3])/(12*step)
        np.testing.assert_allclose(difference,tangent,atol=3e-9,rtol=0)


@pytest.mark.parametrize('degree,angular,order',[(3,2,1),(4,0,0),(3,3,1),(3,4,1)])
def test_contact_reflection_with_angular_and_nonzero_fourier_fields(degree,angular,order):
    value,tangent,metadata=_closure(degree=degree,angular_order=angular,order=order)
    modes=np.array([(m,n) for n in range(-order,order+1) for m in range(-order,order+1)])
    lookup={tuple(mode):i for i,mode in enumerate(modes)}
    reflected=np.array([lookup[(m,-n)] for m,n in modes])
    phase=np.exp(4j*np.pi*(modes[:,1][None,:]-modes[:,1][:,None])*(-.03))
    for response,tolerance in ((value,3e-10),(tangent,3e-9)):
        for slot,sign in zip(response,(1,1,1,-1,-1)):
            reflected_slot=slot[:,reflected][:,:,reflected]
            np.testing.assert_allclose(slot,sign*phase*reflected_slot,atol=tolerance,rtol=0)


@pytest.mark.parametrize('degree,angular,exponents',
    [(p,a,e) for p in (3,4) for a in (0,1,2) for e in ((.6,),(.6,.8))]
    +[(3,0,(.55,.6,.8,.95))]
    +[(p,a,(.6,.8)) for p in (3,4) for a in (3,4)])
def test_stable_coordinates_recover_every_original_potential_and_variation(degree,angular,exponents):
    _assert_coordinate_recovery(degree,angular,exponents,np.linalg.solve)


def _assert_coordinate_recovery(degree,angular,exponents,inverse_solve):
    cell=_cell(degree=degree,exponents=exponents,angular_order=angular)
    fields=PairCornerFields(cell._base,exponents,junctions=True,angular_order=angular)
    bary=np.array([[.23,.31,.46],[.41,.17,.42]])
    db=np.array([[.03,-.01,-.02],[-.02,.04,-.02]])
    inverse=np.array([[1.2,.1],[-.2,.8]])
    dinverse=np.array([[.03,.01],[-.04,.02]])
    data=cell._base.data
    for tri in data['triangles']:
        raw=fields._sample(tri,bary,inverse,dinverse,db)
        sampled=fields.sample(tri,bary,inverse,dinverse,db)
        for actual,expected in zip(fields.gradients(tri,bary,inverse,dinverse,db),sampled[1:]):
            np.testing.assert_array_equal(actual,expected)
        restored=[array.copy() for array in sampled]
        for array in restored:
            for indices,transform in zip(fields.groups,fields.transforms):
                block=array[...,indices]
                array[...,indices]=inverse_solve(transform.T,block.reshape(-1,len(indices)).T).T.reshape(block.shape)
            array/=fields.column_scale
        # Add the globally represented Pp function, using the independent
        # Bernstein derivative coefficients rather than sampled differences.
        local=fields.interpolant[data['pmap'][data['periodic'][tri]]]
        p,_=values_and_tangent(bary,db,cell._base.degree)
        low,dlow=values_and_tangent(bary,db,cell._base.degree-1)
        reference=gradient_coefficients(cell._base.degree)
        r=np.einsum('qv,vna,ni->qai',low,reference,local)
        dr=np.einsum('qv,vna,ni->qai',dlow,reference,local)
        restored[0]+=p@local
        restored[1]+=np.einsum('ab,qai->qbi',inverse,r)
        restored[2]+=np.einsum('ab,qai->qbi',dinverse,r)+np.einsum('ab,qai->qbi',inverse,dr)
        for actual,expected in zip(restored,raw):
            np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=2e-9)


def test_expanded_angular_coordinates_recover_with_independent_decimal_inverse():
    # This inverse is only a reference instrument. A float64 inverse solve
    # amplifies its own rounding in these full, poorly scaled raw coordinates.
    # Restore the exact-input triangular map independently at 80 digits;
    # neither the sampled fields nor the original error limits change.
    from decimal import Decimal as D,localcontext
    @lru_cache(None)
    def rows(data,size):
        matrix=np.frombuffer(data,dtype=float).reshape(size,size)
        assert not np.any(np.triu(matrix,1))
        return tuple((D.from_float(float(matrix[i,i])),
                      tuple((j,D.from_float(float(matrix[i,j])))
                            for j in range(i) if matrix[i,j]!=0)) for i in range(size))
    def solve(matrix,right):
        lower=rows(np.asarray(matrix,float).tobytes(),len(matrix))
        result=np.zeros_like(right)
        with localcontext() as context:
            context.prec=80
            for column in range(right.shape[1]):
                if not np.any(right[:,column]):continue
                x=[]
                for i,(diagonal,entries) in enumerate(lower):
                    rhs=D.from_float(float(right[i,column]))
                    x.append((rhs-sum((value*x[j] for j,value in entries),D(0)))/diagonal)
                result[:,column]=[float(value) for value in x]
        return result
    _assert_coordinate_recovery(4,4,(.55,.6,.8,.95),solve)


def test_angular_extension_contact_preserves_material_directions_and_units():
    direction=.3+.2j;reference=None
    for scale in (1.,700.):
        value,tangent,_=_closure(scale=scale,angular_order=4,gap_tangent=0.,
                                 material_direction=direction)
        expected=np.asarray(value)*direction;expected[2]*=-1
        np.testing.assert_allclose(tangent,expected,atol=3e-9,rtol=0)
        if reference is None:reference=value
        else:np.testing.assert_allclose(value,reference,atol=3e-10,rtol=0)


def test_prepared_fields_keep_gap_materials_units_and_directions_live(monkeypatch):
    import wavebit._core.conforming.pair_corner_assembly as assembly
    cell=_cell();reference={}
    for gap in (0.,.003):
        value,tangent,_=cell.closure(**_settings(gap,1.))
        reference[gap]=np.asarray(value),np.asarray(tangent)
    # A material/geometry sweep must use its configured coordinates without
    # another reference QR or an ever-growing cache of physical settings.
    def forbidden(*args,**kwargs):
        raise AssertionError('a configured cell must reuse its prepared corner fields')
    monkeypatch.setattr(assembly,'PairCornerFields',forbidden)
    factor=1+.02j;direction=.3+.2j
    slot_scale=np.array([factor,factor,1/factor,factor,factor])[:,None,None,None]
    sign=np.array([1.,1.,-1.,1.,1.])[:,None,None,None]
    for gap in (.003,0.):
        settings=_settings(700*gap,700.)
        settings['epsilon']=factor*np.asarray(settings['epsilon'])
        settings.update(background=factor,gap_tangent=700.,
                        epsilon_tangent=direction*settings['epsilon'],
                        background_tangent=direction*factor)
        value,tangent,_=cell.closure(**settings)
        original,change=reference[gap]
        np.testing.assert_allclose(value,slot_scale*original,atol=3e-10,rtol=0)
        np.testing.assert_allclose(tangent,slot_scale*(change+direction*sign*original),atol=3e-9,rtol=0)


def test_expanded_contact_preserves_physical_units_and_complex_material_directions():
    # Multiplying every passive material and its background by the same
    # complex scalar multiplies the transverse tensor by that scalar and
    # divides the longitudinal inverse by it. This exact direction probes
    # the complete material derivative while the mesh remains fixed.
    direction=.3+.2j
    reference=None
    for scale in (1.,700.):
        value,tangent,_=_closure(scale=scale,exponents=(.55,.6,.8,.95),
                                 gap_tangent=0.,material_direction=direction)
        expected=np.asarray(value)*direction
        expected[2]*=-1
        np.testing.assert_allclose(tangent,expected,atol=3e-9,rtol=0)
        np.testing.assert_allclose([value.epxy,value.epyx],0.,atol=3e-10,rtol=0)
        if reference is None:reference=value
        else:np.testing.assert_allclose(value,reference,atol=3e-10,rtol=0)


@pytest.mark.parametrize('change',['quadrature','exponent_order','length_unit'])
def test_expanded_contact_preserves_complete_response_under_representations(change):
    exponents=(.55,.6,.8,.95)
    reference=_closure(exponents=exponents)
    options=dict(exponents=exponents)
    if change=='quadrature':options['quadrature']=96
    elif change=='exponent_order':options['exponents']=exponents[::-1]
    else:options['scale']=1.
    value,tangent,_=_closure(**options)
    factor=700. if change=='length_unit' else 1.
    np.testing.assert_allclose(value,reference[0],atol=3e-10,rtol=0)
    np.testing.assert_allclose(tangent,np.asarray(reference[1])*factor,atol=3e-9,rtol=0)


def test_expanded_positive_subpixel_gap_preserves_physical_reflection():
    # Exercise the expanded root-20 small-gap path with the most singular
    # allowed exponent within the per-test CI time limit. The complete
    # four-node space is tested at contact above; its separate positive-gap
    # qualification is recorded in nvf-pair-symmetric-potentials.md.
    value,tangent,metadata=_closure(gap=1e-7,exponents=(.55,))
    assert metadata['radial_root_order']==20
    assert metadata['integration_quadrature']==80
    np.testing.assert_allclose([value.epxy,value.epyx],0.,atol=3e-10,rtol=0)
    np.testing.assert_allclose([tangent.epxy,tangent.epyx],0.,atol=3e-9,rtol=0)


@pytest.mark.parametrize('ds',[0.,1/700,1.])
def test_contact_refinement_solves_all_original_linear_constraints_together(ds):
    rng=np.random.default_rng(83)
    draw=lambda shape:(rng.normal(size=shape)+.2j*rng.normal(size=shape))*.1
    nc,nz=2,1
    q=draw((2,3));q[:,nc:]=0
    r=draw((2,3));r[:,nc:]=0
    qt=q.conj().T;source=r.T.copy()
    k=np.diag([3.,2.,1.]).astype(complex)
    k[0,1]=k[1,0]=.2
    dk,dq,dr,dqt,dsource=(draw(shape) for shape in ((3,3),(2,3),(2,3),(3,2),(3,2)))
    e=np.array([[1.6+.05j]]);de=np.array([[.2+.03j]])
    C,Qbar,Rbar=draw((2,1)),draw((2,1)),draw((2,1))
    U=draw((4,2));x=np.eye(2)-q@U[:3]
    A,DA=np.kron(np.eye(2),e),np.kron(np.eye(2),de)
    H=A@x+r@U[:3]
    dK=np.hstack([dk[:nc],ds*C]);dQ=np.hstack([dq,ds*Qbar]);dR=np.hstack([dr,ds*Rbar])
    qc,rc,sc,qtc=q[:,:nc],r[:,:nc],source[:nc],qt[:nc]
    identity=np.eye(2);zero=np.zeros((2,2))
    # Independently solve the original weak + retained + flux equations,
    # without the projected metric or its preconditioner.
    system=np.block([[k[:nc,:nc],sc,-qtc],[qc,identity,zero],[-rc,-A,identity]])
    rhs=np.vstack([-dK@U-dsource[:nc]@x+dqt[:nc]@H,-dQ@U,DA@x+dR@U])
    expected=np.split(np.linalg.solve(system,rhs),3)
    coordinate=np.array([[1.,.3],[-.2,.8]])
    metric=coordinate.T@(k[:nc,:nc]-sc@qc-qtc@(rc-A@qc))@coordinate
    base=(k,dk,q,dq,r,dr,qt,dqt,source,dsource,e,de,dict(nc=nc,nz=nz,ds=ds))
    initial=[a+draw(a.shape)*.01 for a in expected]
    actual=refine(base,C,Qbar,Rbar,coordinate,metric,U,initial[0],x,initial[1],H,initial[2])
    np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=0)
    stable=refine(base,C,Qbar,Rbar,coordinate,metric,U,expected[0].copy(),x,
                  expected[1].copy(),H,expected[2].copy())
    for a,b in zip(stable,expected):np.testing.assert_array_equal(a,b)
