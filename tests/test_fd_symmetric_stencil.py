"""Independent controls of the experimental extended material stencil."""
from functools import lru_cache
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from studies import fdtd_subpixel_symmetric_stencil as study
from wavebit.fd._runtime import runtime
from wavebit.fd._kernel.masks import Region


def planar_fixture(size, normal, eps, offset=.23):
    """Exact periodic piecewise-constant E_t/D_n states on parallel slabs."""
    z,y=np.meshgrid(np.arange(size),np.arange(size),indexing='ij')
    p=np.stack((y,z),axis=-1).reshape(-1,2)
    points=np.concatenate((p+[.5,0.],p+[0.,.5]))
    axes=np.repeat([0,1],len(p));n=np.array(normal)
    integer=np.array([1.,0.]) if normal[0]==1 else np.array([3.,4.])
    phase=(points@integer-offset)%size
    signed=np.where(phase<size/2,-np.minimum(phase,size/2-phase),
                    np.minimum(phase-size/2,size-phase))/np.linalg.norm(integer)
    def average(axis):
        half=abs(integer[axis])/2
        if half==0:return (phase<size/2).astype(float)
        lo,hi=phase-half,phase+half
        return sum(np.maximum(0,np.minimum(hi,k*size+size/2)-np.maximum(lo,k*size))
                   for k in (-1,0,1))/(2*half)
    ell=np.choose(axes,[average(0),average(1)])
    area=np.choose(axes,[average(1),average(0)])
    matrix,report=study.fit(points,axes,np.broadcast_to(n,points.shape),signed,ell,area,eps,periodic=size)
    tangent=np.array([-n[1],n[0]])
    electric=np.stack((tangent[axes],(ell/eps+1-ell)*n[axes]),axis=-1)
    displacement=np.stack(((area*eps+1-area)*tangent[axes],n[axes]),axis=-1)
    return matrix,report,electric,displacement


@lru_cache(maxsize=2)
def small_geometry(dim):
    q=np.arange(-4,5)*.1
    if dim==2:
        z,y=np.meshgrid(q,q,indexing='ij')
        points=np.stack((y,z),axis=-1)[:,:,None,:]
        components=(1,2);center=(.017,-.023)
    else:
        z,y,x=np.meshgrid(q,q,q,indexing='ij')
        points=np.stack((x,y,z),axis=-1)
        components=(0,1,2);center=(.017,-.023,.009)
    matrix,report=study.geometry(points,.1,center,.16,12.,components)
    return points,components,center,matrix,report


def increment_gap(dim,backend):
    points,components,_,matrix,_=small_geometry(dim)
    shape=points.shape[:-1]
    diagonal,index,value=study.padded(matrix)
    rt=runtime('jax' if backend=='jax_jit' else backend)
    region=Region(z=(0,shape[0]),y=(0,shape[1]),x=(0,shape[2]))
    # A reproducible nonzero increment and memory are experiment inputs.
    term=study.StencilTerm('s','E',region,.037,(),components)
    local=np.sin(np.arange(3*np.prod(shape))).reshape((3,)+shape)
    memory=np.cos(np.arange(3*np.prod(shape))).reshape((3,)+shape)
    def step(curl,mem,idx,val):
        return term.prepare(curl,mem,dict(stencil_index=idx,stencil_value=val),rt)
    if backend=='jax_jit':
        import jax
        step=jax.jit(step)
    args=[rt.array(local,'float64'),rt.array(memory,'float64'),rt.array(index),rt.array(value,'float64')]
    flux,delta=step(*args);flux=rt.numpy(flux);delta=rt.numpy(delta)
    selected=np.stack(tuple(local[c] for c in components)).ravel()
    actual=diagonal*np.stack(tuple(flux[c]-memory[c] for c in components)).ravel()
    actual+=np.stack(tuple(delta[c] for c in components)).ravel()
    return float(np.max(abs(actual-.037*(matrix@selected))))


@pytest.mark.parametrize('normal',((1.,0.),(.6,.8)))
def test_exact_static_planar_states(normal):
    for eps in (1.,2.25,12.):
        for size in (16,24,32):
            matrix,report,electric,displacement=planar_fixture(size,normal,eps)
            # Eighteen preliminary cases measured at most 6.85563e-15.
            np.testing.assert_allclose(matrix@displacement,electric,rtol=0,atol=1e-13)
            assert report['minimum_eigenvalue']>0
            assert (matrix-matrix.T).nnz==0


def test_thirty_planar_interface_offsets():
    # Thirty distinct subcell phases; .37 avoids the aligned endpoint.
    for i in range(30):
        matrix,report,electric,displacement=planar_fixture(16,(.6,.8),12.,(i+.37)/30)
        # The preceding contrast/size audit measured 6.85563e-15.
        np.testing.assert_allclose(matrix@displacement,electric,rtol=0,atol=1e-13)
        assert report['minimum_eigenvalue']>0


@pytest.mark.parametrize('dim',(2,3))
@pytest.mark.parametrize('backend',('numpy','jax','jax_jit','torch'))
def test_increment_matches_independent_sparse_product(dim,backend):
    # Eight premeasurements, nonzero flux memory: maximum 1.08421e-16.
    assert increment_gap(dim,backend)<2e-15


def test_rigid_translation_keeps_the_material_operator():
    q=np.arange(-10,11)/40
    z,y=np.meshgrid(q,q,indexing='ij')
    points=np.stack((y,z),axis=-1)[:,:,None,:]
    matrix,_=study.geometry(points,.025,(0.,0.),.15,12.,(1,2))
    for shift in ((.3,.7),(1.1,-.7)):
        other,_=study.geometry(points+shift,.025,shift,.15,12.,(1,2))
        # Grazing cuts measured 4.04286e-9; twice that rounded upward.
        # The rejected hard support measured 1.41418e-3 on these inputs.
        assert abs(matrix-other).max()<1e-8


def test_energy_bounds_enclose_independent_eigenvalues():
    matrix=csr_matrix([[2.,.25],[.25,.5]])
    bounds=study.energy_bounds(matrix)
    low,high=np.linalg.eigvalsh(matrix.toarray())
    assert 0<bounds['energy_lower_bound']<=low
    assert bounds['spectral_upper_bound']>=high
    # Measured Cholesky reconstruction <=7.82e-16 on the full sphere.
    assert bounds['factor_relative_residual']<1e-14


def test_sparse_certificate_encloses_nonlocal_positive_matrix():
    # A known SPD Gram matrix with nonlocal couplings and mixed signs.
    rng=np.random.default_rng(10913)
    basis=np.eye(17)+np.tril(rng.uniform(-.4,.4,size=(17,17)),-1)
    matrix=basis@basis.T
    eigenvalues=np.linalg.eigvalsh(matrix)
    failure=None
    try:
        bounds=study.energy_bounds(csr_matrix(matrix))
    except ValueError as error:
        failure=str(error)
    assert failure is None, failure
    assert 0<bounds['energy_lower_bound']<=eigenvalues[0]
    assert bounds['spectral_upper_bound']>=eigenvalues[-1]
    # Radius-4/10 sparse screens measured reconstruction <=1.337e-15.
    assert bounds['factor_relative_residual']<1e-14


@pytest.mark.parametrize('broken,reason',(
    ([[-1.,0.],[0.,1.]],'StencilEnergyRefused'),
    ([[2.,1.],[0.,2.]],'StencilSymmetryRefused'),
    ([[float('nan'),0.],[0.,1.]],'StencilFiniteRefused'),
))
def test_invalid_energy_is_refused_before_stepping(broken,reason):
    with pytest.raises(ValueError,match=reason):
        study.energy_bounds(csr_matrix(broken))


@pytest.mark.parametrize('factor',(2.,float('nan'),0.))
def test_invalid_time_step_is_refused_before_stepping(factor):
    import dataclasses
    from studies.fdtd_subpixel_coupled_cylinder import prepare
    from studies.fdtd_subpixel_mie_cylinder import scene_for,SCHEDULE
    from wavebit.fd._lower import plan,lower
    empty=scene_for(40,'TE',*SCHEDULE[12.])
    scene,p=prepare(empty,plan(empty),12.,(0.,0.))
    # The original dt^2 rate is .282553; doubling dt makes it >1.
    bad=dataclasses.replace(p,dt=factor*p.dt)
    with pytest.raises(ValueError,match='StencilCFLRefused'):
        study.replace(scene,bad,lower(scene,p,'numpy'),(0.,0.),.15,12.,(1,2))


@pytest.mark.slow
@pytest.mark.parametrize('shape',('cylinder','sphere'))
def test_same_operator_physical_controls(shape):
    row=(study.cylinder if shape=='cylinder' else study.sphere)()
    # Preserve the previous independent physical-instrument limits:
    # cylinder 5%, sphere 3%. This candidate measured 3.51206%/2.46813%.
    assert abs(row['error'])<(.05 if shape=='cylinder' else .03)
    decay=row['decay'] if shape=='cylinder' else max(row['decay'].values())
    # Existing cylinder settling control, measured here <=3.729e-9.
    assert decay<1e-7
    assert row['preparation']['energy_lower_bound']>0
    assert row['preparation']['dt_squared_rate']<1



@lru_cache(maxsize=1)
def scope_fixture():
    from studies.fdtd_subpixel_coupled_cylinder import prepare
    from studies.fdtd_subpixel_mie_cylinder import scene_for,SCHEDULE
    from wavebit.fd._lower import plan
    empty=scene_for(40,'TE',*SCHEDULE[12.])
    return prepare(empty,plan(empty),12.,(0.,0.))


@pytest.mark.parametrize('fault',(
    'empty_scene','two_structures','empty_terms','magnetic_term','masked_term',
    'background','magnetic_background','conductivity','anisotropy','magnetic_material',
    'material_mismatch','center_mismatch','radius_mismatch','components','invariant_axis',
    'dtype','initial_field','nan_radius','nan_center','jax_parameter','torch_parameter',
))
def test_unsupported_research_scope_is_refused(fault):
    import dataclasses
    import wavebit.fd as fd
    from wavebit.fd._kernel.masks import Slab
    scene,p=scope_fixture();old=p.constitutive[0]
    args=dict(center=(0.,0.),radius=.15,eps=12.,components=(1,2))
    if fault=='empty_scene':scene=scene.replace(structures=())
    elif fault=='two_structures':scene=scene.replace(structures=scene.structures*2)
    elif fault=='empty_terms':p=p.with_static(terms=tuple(t for t in p.terms if t is not old))
    elif fault in ('magnetic_term','masked_term'):
        bad=dataclasses.replace(old,**({'field':'H'} if fault=='magnetic_term'
            else {'masks':(Slab(1,old.region.z,old.region.y,old.region.x),)}))
        p=p.with_static(terms=tuple(bad if t is old else t for t in p.terms))
    elif fault=='background':scene=scene.replace(background=2.)
    elif fault=='magnetic_background':scene=scene.replace(background=fd.Material(eps=1.,mu=1.))
    elif fault in ('conductivity','anisotropy','magnetic_material','material_mismatch'):
        material={'conductivity':fd.Conductor(eps=12.,sigma=.1),
                  'anisotropy':fd.Material(eps=(12.,12.,13.)),
                  'magnetic_material':fd.Material(eps=12.,mu=1.),
                  'material_mismatch':fd.Material(eps=2.25)}[fault]
        scene=scene.replace(structures=(fd.Structure(scene.structures[0].shape,material),))
    elif fault=='center_mismatch':args['center']=(.01,0.)
    elif fault=='radius_mismatch':args['radius']=.16
    elif fault=='components':args['components']=(0,2)
    elif fault=='invariant_axis':p=dataclasses.replace(p,nx=2)
    elif fault=='dtype':p=dataclasses.replace(p,dtype='float32')
    elif fault=='initial_field':p=dataclasses.replace(p,has_E0=True)
    elif fault=='nan_radius':args['radius']=float('nan')
    elif fault=='nan_center':args['center']=(float('nan'),0.)
    elif fault=='jax_parameter':
        import jax.numpy as jnp
        args['eps']=jnp.asarray(12.)
    elif fault=='torch_parameter':
        import torch
        args['eps']=torch.tensor(12.,requires_grad=True)
    with pytest.raises(ValueError,match='StencilScopeRefused'):
        study.validate_scope(scene,p,**args)


def test_replacement_refuses_scope_before_assembly(monkeypatch):
    scene,p=scope_fixture()
    def forbidden(*args,**kwargs):
        assert False, 'Unsupported material reached numerical assembly'
    monkeypatch.setattr(study,'geometry',forbidden)
    with pytest.raises(ValueError,match='StencilScopeRefused'):
        study.replace(scene.replace(background=2.),p,{},(0.,0.),.15,12.,(1,2))
