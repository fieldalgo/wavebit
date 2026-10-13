"""Independent consistency and stability checks for the C3 research comparator.

These certify the instrument's algebra, not scattering convergence. Production
still uses its existing constitutive map. Actual source mutants are recorded in
the companion study log; each assertion group has a demonstrated witness.
"""
from functools import lru_cache

import numpy as np
import pytest

from studies.fdtd.subpixel_triplet_cylinder.fdtd_subpixel_triplet_cylinder import action, certificate, circle_fraction, circle_maps, diagonal, operator_audit, prepare_circle, static_audit
from studies.fdtd.subpixel_mie_cylinder.fdtd_subpixel_mie_cylinder import scene_for, SCHEDULE
from wavebit.fdtd._prepare import plan
from wavebit.fdtd._runtime import runtime


@lru_cache(maxsize=1)
def fixture():
    empty=scene_for(40,'TE',*SCHEDULE[12.])
    return prepare_circle(empty,plan(empty),12.,(0.,0.))


def test_static_edge_face_solution():
    rows=static_audit()
    # 36 measurements, axis/tilted normals, three contrasts and spacings:
    # max exact-map field residual 1.11023e-15; use 16x that measured scale.
    assert max(r['accurate_asymmetric']['max_error'] for r in rows) < 2e-14
    # Independent circulation audit measured 7.96586e-15 across those rows.
    assert max(r['accurate_asymmetric']['raw_circulation'] for r in rows) < 2e-13


def test_circle_segment_geometry():
    # Unit circle at (3/5,4/5): either centered radial chord ends at its
    # midpoint. The point (1,0) is a y-segment half cut and a zero z chord.
    p=np.array([[0.,0.],[.6,.8],[1.,0.]])
    for dx in (.1,.2,.4):
        # Direct chord audit measured at most 6.66134e-16 absolute error.
        np.testing.assert_allclose(circle_fraction(p,dx,0,1.),[1.,.5,.5],rtol=0,atol=1e-14)
        np.testing.assert_allclose(circle_fraction(p,dx,1,1.),[1.,.5,0.],rtol=0,atol=1e-14)
    # A fully enclosed segment must remain exactly full even away from zero.
    assert circle_fraction(np.array([[.8,.3]]),.1,0,1.)[0] == 1.


def test_exact_arc_quadrants():
    from studies.fdtd.subpixel_circle_moments.fdtd_subpixel_circle_moments import arc_moments, RADIUS
    # Each voxel is one quadrant of the disc. Direct angular integration
    # gives length=pi*r/2, diagonal=1/2 and yz=sign(y*z)/pi.
    for sy,sz in ((1,1),(-1,1),(1,-1),(-1,-1)):
        p=np.array([[0.,sy*RADIUS/2,sz*RADIUS/2]])
        length,w,yz=arc_moments(p,RADIUS)
        # The 80-cell independent area/face identity measured 1.45717e-14
        # after normalization by dx; 1e-13 leaves margin for this oracle.
        np.testing.assert_allclose(length/RADIUS,np.pi/2,rtol=0,atol=1e-13)
        np.testing.assert_allclose(w,[[0.,.5,.5]],rtol=0,atol=1e-13)
        np.testing.assert_allclose(yz,[sy*sz/np.pi],rtol=0,atol=1e-13)


def test_uniform_inverse_and_indexed_assembly():
    shape=(7,7,1)
    matrices=np.broadcast_to(np.eye(2)/12.,(4,)+shape+(2,2)).copy()
    vector=np.sin(np.arange(3*np.prod(shape)).reshape((3,)+shape))
    actual=action(vector,matrices)
    # Uniform medium must give E=D/eps away from the vacuum patch ends.
    # Measured assembly residual 2.22045e-16; 1e-14 leaves roundoff margin.
    np.testing.assert_allclose(actual[1:,1:-1,1:-1],vector[1:,1:-1,1:-1]/12.,rtol=0,atol=1e-14)
    *_,matrices=fixture()
    audit=operator_audit(matrices)
    assert audit['indexed_action_gap'] < 1e-14
    assert audit['diagonal_cross_split_gap'] < 1e-14
    assert audit['assembled_asymmetry'] == 0.
    assert audit['assembled_min_eigenvalue'] > 0.


def test_local_map_is_symmetric_positive():
    # A curved interface spans all normal directions; these are geometry
    # samples, not acceptance constants. The actual certificate uses >0.
    q=np.arange(-9,10)/40
    zz,yy=np.meshgrid(q,q,indexing='ij')
    _,matrices=circle_maps(np.stack((yy,zz),axis=-1)[:,:,None],1/40,(0.,0.),.15,12.)
    np.testing.assert_array_equal(matrices,np.swapaxes(matrices,-1,-2))
    assert np.linalg.eigvalsh(matrices).min() > 0.


def test_circle_maps_give_each_slot_its_own_fraction():
    """Each pair's edge slot holds the chord along its edge, its face slot the chord across it.

    Measured 2026-09-26 while pinning the sphere's slots: with ``circle_maps`` reading the edge
    fraction across the edge axis -- so both slots hold the transverse chord -- none of the fast
    cases of the four files that gate the operator went red, and with the face read along the
    edge axis only ``test_rigid_translation_keeps_the_material_operator`` did. The maps do not
    hand their fractions out, so the oracle rebuilds every pair map from its own: the chord of
    the circle written out by hand for each slot, and ``Gamma Pi^-1`` by a dense inverse where
    ``pair_map`` solves. Points on the circle, where the two chords differ; the fixture is
    asserted to separate the slots, and the map built from swapped slots to be told apart.

    Mutants, each alone, over the 140 fast cases of the four files that gate the operator
    (measured 2026-09-26): the edge read across its axis -- this case alone; the face read
    along the edge axis -- this case and the rigid-translation control.
    """
    from studies.fdtd.subpixel_triplet_cylinder.fdtd_subpixel_triplet_cylinder import ORIENTATIONS
    rng=np.random.default_rng(926)
    angle=rng.uniform(0,2*np.pi,40)
    dx,radius,eps=1/40,.15,12.
    points=radius*np.stack((np.cos(angle),np.sin(angle)),axis=-1)+rng.uniform(-.5,.5,(40,2))*dx
    _,maps=circle_maps(points,dx,(0.,0.),radius,eps)

    def chord(p,axis):
        half=np.sqrt(max(0.,radius**2-p[1-axis]**2))
        return max(0.,min(p[axis]+dx/2,half)-max(p[axis]-dx/2,-half))/dx

    def pair(normal,ell,area):
        nn=np.outer(normal,normal)
        accurate=(np.eye(2)+ell[:,None]*(1/eps-1)*nn)@np.linalg.inv(np.eye(2)+area[:,None]*(eps-1)*(np.eye(2)-nn))
        return (accurate+accurate.T)/2
    separated=[0.,0.]
    for k,signs in enumerate(ORIENTATIONS):
        for n,point in enumerate(points):
            normal=point/np.linalg.norm(point)
            shifted=[point+sign*dx/2*np.eye(2)[i] for i,sign in enumerate(signs)]
            ell=np.array([chord(shifted[i],i) for i in range(2)])
            area=np.array([chord(shifted[i],1-i) for i in range(2)])
            want=pair(normal,ell,area)
            # Two spellings of one inverse and one chord: rounding only.
            np.testing.assert_allclose(maps[k,n],want,rtol=0,atol=1e-12,err_msg=f'{signs} {n}')
            separated=[max(separated[0],np.abs(pair(normal,area,area)-want).max()),
                       max(separated[1],np.abs(pair(normal,ell,ell)-want).max())]
    # The control that either slot read as the other would be seen at all.
    assert min(separated) > 1e-2, separated


def test_circle_local_rotation_covariance():
    from studies.fdtd.subpixel_triplet_cylinder.fdtd_subpixel_triplet_cylinder import ORIENTATIONS
    # A shifted sampling square cuts every quadrant of the historical
    # radius-.15 cylinder. Coordinates are experiment inputs, not fit knobs.
    q=np.linspace(-.18,.18,13)+.003
    z,y=np.meshgrid(q,q,indexing='ij')
    points=np.stack((y,z),axis=-1)[:,:,None]
    rotation=np.array([[0.,-1.],[1.,0.]])
    _,original=circle_maps(points,.025,(0.,0.),.15,12.)
    _,rotated=circle_maps(points@rotation.T,.025,(0.,0.),.15,12.)
    for k,(sy,sz) in enumerate(ORIENTATIONS):
        other=ORIENTATIONS.index((-sz,sy))
        # Measured 90-degree covariance discrepancy: at most 3.33067e-16.
        np.testing.assert_allclose(rotated[other],rotation@original[k]@rotation.T,
                                   rtol=0,atol=1e-14)


@pytest.mark.parametrize('backend',['numpy','jax','jax_jit','torch'])
def test_research_increment_uses_certified_operator(backend):
    scene,p,params,bounds,matrices=fixture()
    assert np.array_equal(matrices,np.swapaxes(matrices,-1,-2))
    assert bounds['lower'] > 0 and bounds['dt_squared_rate'] < 1
    term=p.constitutive[0]
    rt=runtime('jax' if backend=='jax_jit' else backend)
    size=int(np.prod(term.region.shape))
    local=np.sin(np.arange(3*size).reshape((3,)+term.region.shape))
    full=np.zeros(p.shape);full[term.view]=local
    mem=np.zeros_like(local)
    def compute(curl,memory,blocks):
        return term.prepare(curl,memory,{'research_triplets':blocks},rt)
    args=tuple(rt.array(v,'float64') for v in (full,mem,matrices))
    if backend=='jax_jit':
        import jax
        flux,delta=jax.jit(compute)(*args)
    else:
        flux,delta=compute(*args)
    flux,delta=np.asarray(rt.numpy(flux)),np.asarray(rt.numpy(delta))
    expected=term.ratio*action(local,matrices)
    actual=diagonal(matrices)*flux+delta
    # NumPy independent assembly measured 2.22045e-16 before this test.
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14)


def test_invalid_energy_certificate_is_refused_at_preparation():
    scene,p,params,_,matrices=fixture()
    term=p.constitutive[0]
    broken=matrices.copy()
    # One local block with a negative eigenvalue invalidates this certificate.
    broken[0,broken.shape[1]//2,broken.shape[2]//2,0]=-np.eye(2)
    with pytest.raises(ValueError,match='TripletEnergyCFLRefused'):
        certificate(scene,p,term,broken,params)
    # A time step twice the certified one exceeds its measured .268675 rate.
    import dataclasses
    with pytest.raises(ValueError,match='TripletEnergyCFLRefused'):
        certificate(scene,dataclasses.replace(p,dt=2*p.dt),term,matrices,params)


@pytest.mark.slow
def test_research_cylinder_physical_control():
    from studies.fdtd.subpixel_coupled_cylinder.fdtd_subpixel_coupled_cylinder import solve
    from studies.fdtd.subpixel_mie_cylinder.fdtd_subpixel_mie_cylinder import CONTOUR, RADIUS, incident_intensity, scattered_flux, cylinder_cross_sections
    from wavebit.fdtd._execution import execute, _results
    empty=scene_for(40,'TE',*SCHEDULE[12.])
    rt=runtime('jax');base=plan(empty)
    reference=solve(empty,empty,base,rt)
    scene,p,params,_,_=prepare_circle(empty,base,12.,(0.,0.))
    result=_results(empty,p,execute(p,params,rt),rt,params)
    incident=incident_intensity(reference,'TE',CONTOUR,p.grid)
    sigma=scattered_flux(result,reference,'TE',CONTOUR,p.grid)[0]/incident
    exact=float(cylinder_cross_sections(RADIUS,12.,1.,'TE')[1])
    # Fresh 24-case study: this origin's error was .0409268323. Five percent
    # leaves 22% relative margin. This single-case bound guards the instrument;
    # only the separate three-resolution/eight-placement fit grades its order.
    assert abs(sigma/exact-1) < .05
    # Largest late-monitor fraction in the 24 measured cases: 6.01283e-9.
    assert float(result.diagnostics['decay']['box']) < 1e-7


def test_circle_fraction_carries_the_namespace_it_was_given():
    """`circle_fraction` lifts a 2D point to a triple, and the lift is where a backend is lost.

    The chord is now `fd.Sphere.exact_edge_fill` on `(0, y, z)`, so the function has to build
    that zero component with the caller's own namespace. With `np.stack` instead -- the
    spelling this had before it took an `xp` -- a JAX array is pulled through `__array__` into
    NumPy and a Torch tensor is detached into one: the forward numbers still look right, the
    gradient is gone, and under `jax.jit` it raises a TracerArrayConversionError instead.

    Measured 2026-09-24 over 400 placements and both axes: JAX eager and `jax.jit` return the
    NumPy answer bit for bit, and Torch is 1.11e-15 away from it -- four or five ulps of a
    fraction, from `torch.sqrt`, and the same spread the public `Sphere` overrides carry in
    `tests/fdtd/test_fd_shapes.py`. So the gate is 1e-14 for the value and bit equality only where
    it was measured. The radius gradient is checked on both autodiff backends against a NumPy
    central difference at 1e-6.
    """
    rng=np.random.default_rng(6151)
    points=rng.normal(size=(400,2))
    points=points/np.linalg.norm(points,axis=-1)[:,None]*.25+rng.uniform(-.6,.6,(400,2))*.025
    reference=[np.asarray(circle_fraction(points,.025,axis,.25)) for axis in (0,1)]
    jax=pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    torch=pytest.importorskip('torch')
    for name,cast,xp,wrap,tol in (('jax',jnp.asarray,jnp,lambda f:f,0.),
                                  ('jit',jnp.asarray,jnp,jax.jit,0.),
                                  ('torch',torch.tensor,torch,lambda f:f,1e-14)):
        for axis in (0,1):
            got=wrap(lambda p,axis=axis,xp=xp: circle_fraction(p,.025,axis,.25,xp))(cast(points))
            assert type(got) is not np.ndarray, name
            np.testing.assert_allclose(np.asarray(got),reference[axis],rtol=0,atol=tol)
    def loss(radius,p,xp):
        return sum(xp.sum(circle_fraction(p,.025,axis,radius,xp)**2) for axis in (0,1))
    step=1e-6
    difference=(loss(.25+step,points,np)-loss(.25-step,points,np))/(2*step)
    assert abs(difference)>1.
    derivative=float(jax.grad(lambda r: loss(r,jnp.asarray(points),jnp))(jnp.asarray(.25)))
    assert abs(derivative-difference)<=1e-7*abs(difference)+1e-10*abs(loss(.25,points,np))
    arg=torch.tensor(.25,dtype=torch.float64,requires_grad=True)
    loss(arg,torch.tensor(points),torch).backward()
    assert abs(float(arg.grad)-difference)<=1e-7*abs(difference)+1e-10*abs(loss(.25,points,np))
