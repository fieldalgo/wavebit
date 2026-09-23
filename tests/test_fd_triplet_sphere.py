"""Independent geometry, algebra, energy and physical controls of 3D research."""
import dataclasses
from functools import lru_cache
import numpy as np
import pytest
from scipy.integrate import quad
from studies import fdtd_subpixel_triplet_sphere as study
from studies.fdtd_subpixel_mie_sphere import sphere_scene, read_result
from wavebit.fd._runtime import runtime


def samples():
    # Reproducible placement inputs; these are not acceptance constants.
    rng = np.random.default_rng(23923)
    shifts = rng.uniform(-.5, .5, size=(30, 3))*.025
    nodes = rng.normal(size=(60, 3))
    nodes /= np.linalg.norm(nodes, axis=-1)[:, None]
    return shifts, nodes*.25


def independent_face(point, dx, axis, radius):
    """Adaptive integration of clipped chords; no corner primitive is used."""
    a, b = (j for j in range(3) if j != axis)
    r2 = radius**2-point[axis]**2
    if r2 <= 0:
        return 0.
    r = np.sqrt(r2)
    lo, hi = max(point[a]-dx/2, -r), min(point[a]+dx/2, r)
    if hi <= lo:
        return 0.
    low, high = point[b]-dx/2, point[b]+dx/2
    cuts = [lo, hi]
    for y in (low, high):
        if abs(y) < r:
            x = np.sqrt(r2-y*y)
            cuts.extend(t for t in (-x, x) if lo < t < hi)
    cuts.sort()
    def chord(x):
        h = np.sqrt(max(0., r2-x*x))
        return max(0., min(high, h)-max(low, -h))
    # The 5,400-sample audit used these integration requests; its largest
    # independent discrepancy was 9.41425e-12 in dimensionless area.
    return sum(quad(chord, a, b, epsabs=1e-15, epsrel=1e-12)[0]
               for a, b in zip(cuts[:-1], cuts[1:]))/dx**2


def test_exact_sphere_edge_and_face_fractions():
    points = np.array([[0., 0., 0.], [.6, .8, 0.], [1., 0., 0.]])
    # Unit sphere chord endpoints (3/5,4/5,0); the axial pole is a half
    # x edge and zero-length y/z chords. The 2D audit was within 6.67e-16.
    np.testing.assert_allclose(study.edge_fraction(points, .1, 0, 1.), [1., .5, .5], atol=1e-14, rtol=0)
    np.testing.assert_allclose(study.edge_fraction(points, .1, 1, 1.), [1., .5, 0.], atol=1e-14, rtol=0)
    shifts, nodes = samples()
    for shift in shifts:
        for axis in range(3):
            points = nodes-shift
            actual = study.face_fraction(points, .025, axis, .25)
            reference = [independent_face(p, .025, axis, .25) for p in points]
            assert np.all((actual >= 0) & (actual <= 1))
            # Twice the 9.41425e-12 measured maximum, rounded upward.
            np.testing.assert_allclose(actual, reference, atol=2e-11, rtol=0)


def test_exact_static_edge_face_map():
    rng = np.random.default_rng(223)
    normal = rng.normal(size=(100, 3))
    normal /= np.linalg.norm(normal, axis=-1)[:, None]
    lengths, areas = rng.uniform(size=(2, 100, 3))
    accurate, symmetric = study.local_map(normal, lengths, areas, 12.)
    nn = normal[..., :, None]*normal[..., None, :]
    gamma = np.eye(3)+lengths[..., :, None]*(1/12.-1)*nn
    pi = np.eye(3)+areas[..., :, None]*11*(np.eye(3)-nn)
    # Direct dense inversion measured 1.77636e-15; allow >10x roundoff.
    np.testing.assert_allclose(accurate, gamma@np.linalg.inv(pi), atol=2e-14, rtol=0)
    np.testing.assert_array_equal(symmetric, symmetric.swapaxes(-1, -2))
    normal = np.array([2., 3., 6.])/7
    tangent = np.cross(normal, [1., 0., 0.])
    tangent /= np.linalg.norm(tangent)
    electric = tangent+(lengths/12+1-lengths)*normal
    displacement = (areas*12+1-areas)*tangent+normal
    accurate, _ = study.local_map(np.broadcast_to(normal, lengths.shape), lengths, areas, 12.)
    # Independently integrated constant tangential E and constant normal D:
    # preliminary maximum component residual was 1.22125e-15.
    np.testing.assert_allclose((accurate@displacement[..., None])[..., 0], electric, atol=2e-14, rtol=0)


def test_sphere_rotation_covariance():
    shifts, nodes = samples()
    points = nodes-shifts[3]
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    _, original = study.sphere_maps(points, .025, (0., 0., 0.), .25, 12.)
    _, rotated = study.sphere_maps(points@rotation.T, .025, (0., 0., 0.), .25, 12.)
    for k, (sx, sy, sz) in enumerate(study.ORIENTATIONS):
        other = study.ORIENTATIONS.index((-sy, sx, sz))
        # Geometry backend audit measured <=5.56e-15 matrix differences;
        # the face-integral control above has a 2e-11 dimensionless limit.
        np.testing.assert_allclose(rotated[other], rotation@original[k]@rotation.T, atol=2e-11, rtol=0)


def test_independent_indexed_sphere_operator():
    q = np.arange(-3, 4)*.1
    z, y, x = np.meshgrid(q, q, q, indexing='ij')
    points = np.stack((x, y, z), axis=-1)
    _, matrices = study.sphere_maps(points, .1, (.012, -.023, .008), .16, 12.)
    audit = study.operator_audit(matrices)
    # Indexed action/split discrepancies measured 2.49801e-16/2.84495e-16.
    assert audit['indexed_action_gap'] < 1e-14
    assert audit['split_gap'] < 1e-14
    assert audit['assembled_asymmetry'] == 0
    assert audit['minimum_eigenvalue'] > 0
    matrices = np.broadcast_to(np.eye(3)/12, (8, 7, 7, 7, 3, 3)).copy()
    vector = np.sin(np.arange(3*7**3)).reshape(3, 7, 7, 7)
    np.testing.assert_allclose(study.action(vector, matrices)[:, 1:-1, 1:-1, 1:-1],
        vector[:, 1:-1, 1:-1, 1:-1]/12, atol=1e-14, rtol=0)


@lru_cache(maxsize=1)
def fixture():
    scene = sphere_scene(20)
    p, params, bounds = study.prepare_sphere(scene)
    return scene, p, params, bounds


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'jax_jit', 'torch'))
def test_sphere_geometry_and_increment_backends(backend):
    shifts, nodes = samples()
    points = nodes[None, ...]-shifts[:, None, :]
    expected = study.sphere_maps(points, .025, (0., 0., 0.), .25, 12.)[1]
    rt = runtime('jax' if backend == 'jax_jit' else backend)
    def geometry(p):
        return study.sphere_maps(p, .025, (0., 0., 0.), .25, 12., rt.xp)[1]
    if backend == 'jax_jit':
        import jax
        geometry = jax.jit(geometry)
    actual = rt.numpy(geometry(rt.array(points, 'float64')))
    # The independent face oracle (not cross-backend self agreement) fixes
    # this 2e-11 scale; the earlier small-grid backend gap was <=5.56e-15.
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)
    _, p, params, _ = fixture()
    term, = p.constitutive
    local = np.sin(np.arange(3*np.prod(term.region.shape))).reshape((3,)+term.region.shape)
    full = np.zeros(p.shape); full[term.view] = local
    def step(curl, blocks):
        return term.prepare(curl, rt.xp.zeros_like(curl[term.view]), {'sphere_triplets': blocks}, rt)
    if backend == 'jax_jit':
        step = jax.jit(step)
    flux, delta = step(rt.array(full, 'float64'), rt.array(params['sphere_triplets'], 'float64'))
    actual = study.diagonal(params['sphere_triplets'])*rt.numpy(flux)+rt.numpy(delta)
    expected = term.ratio*study.action(local, params['sphere_triplets'])
    # Independent indexed assembly above measured 2.85e-16 roundoff.
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)


def test_sphere_preparation_certifies_energy_and_scope():
    scene, p, params, bounds = fixture()
    term, = p.constitutive
    assert bounds['lower'] > 0 and bounds['dt_squared_rate'] < 1
    blocks = params['sphere_triplets']
    fractions = np.zeros(blocks.shape[:-1])
    def check(b=blocks, pp=p, f=fractions, t=term):
        return study.certificate(scene, pp, t, b, f, f, params)
    broken = blocks.copy()
    middle = tuple(n//2 for n in term.region.shape)
    broken[(0,)+middle] = -np.eye(3)
    with pytest.raises(ValueError, match='SphereTripletEnergyCFLRefused'):
        check(b=broken)
    with pytest.raises(ValueError, match='SphereTripletEnergyCFLRefused'):
        check(pp=dataclasses.replace(p, dt=2*p.dt))
    broken = blocks.copy(); broken[:, 0, ..., 0, 0] = .9
    with pytest.raises(ValueError, match='SphereTripletHaloRefused'):
        check(b=broken)
    with pytest.raises(ValueError, match='SphereTripletGeometryRefused'):
        check(f=fractions-1)
    import wavebit.fd as fd
    with pytest.raises(ValueError, match='SphereTripletScopeRefused'):
        study.prepare_sphere(scene.replace(background=fd.Material(eps=2.)))


@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_sphere_radius_derivative(backend):
    _, nodes = samples()
    points = nodes[:20]+np.array([.002, -.003, .005])
    weights = np.sin(np.arange(8*20*9)).reshape(8, 20, 3, 3)
    def loss(radius, rt):
        matrices = study.sphere_maps(rt.array(points, 'float64'), .025,
            (.003, -.007, .011), radius, 12., rt.xp)[1]
        return rt.xp.sum(matrices*rt.array(weights, 'float64'))
    radius = .2537
    # Step sweep 1e-4..3e-7: the first three steps cross geometry events;
    # the final step gives 311.14950242 vs AD 311.14950624 (1.23e-8 relative).
    h = 3e-7
    nr = runtime('numpy')
    difference = (loss(radius+h, nr)-loss(radius-h, nr))/(2*h)
    rt = runtime(backend)
    if backend == 'jax':
        import jax
        derivative = float(jax.jit(jax.grad(lambda r: loss(r, rt)))(radius))
    else:
        arg = rt.array(radius, 'float64'); arg.requires_grad_(True)
        loss(arg, rt).backward()
        derivative = float(arg.grad)
    # Measured nonzero slope ~311; the predeclared C6 comparison budget is
    # 1e-7 relative plus 1e-10 times the observable. This is a local-map
    # radius check, not a physical adjoint or all-topology gradient claim.
    assert abs(derivative) > 1 and abs(difference) > 1
    assert abs(derivative-difference) <= 1e-7*abs(difference)+1e-10*abs(loss(radius, nr))


@pytest.mark.slow
def test_research_sphere_physical_control():
    from wavebit.fd._run import execute, _results
    scene, p, params, _ = fixture()
    rt = runtime('jax')
    row = read_result(_results(scene, p, execute(p, params, rt), rt, params), 2.25)
    # Fresh 20-cell origin: Mie error .0277385025 and Qabs -1.15897e-6.
    # Round up to 4% and 2e-6. The separate nine-case fit grades the order.
    assert abs(row['error']) < .04
    assert abs(row['qabs']) < 2e-6
    # All nine glass runs measured late-monitor fractions <=1.562e-11.
    assert max(row['decay'].values()) < 1e-9
