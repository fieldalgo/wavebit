"""Independent constitutive identities, conserved energy, and planning doors.

The Fourier symbol and self-adjoint identity do not read the implementation's
interpolation. Closed cavities test the full step, including D/B time levels.
Tolerance provenance and actual source-mutant results: docs/fd-mie-validation.md.
"""
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._kernel.constitutive import cross_action
from wavebit.fd._kernel.step import initial_state, step
from wavebit.fd._lower.constitutive import enclosure
from wavebit.fd._runtime import runtime
from fd_helpers import to_numpy


BACKENDS = ('numpy', 'jax', 'jit', 'torch')


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


def cavity(eps=2.25, mu=1.5, center=(.012, .007, -.009), steps=80):
    return fd.Scene(domain=fd.Domain(size=(.8,)*3), resolution=fd.Resolution(dx=.05),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    run=fd.Run(steps=steps), structures=(fd.Structure(
                        fd.Sphere(radius=.15, center=center), fd.Material(eps=eps, mu=mu)),))


def seeded_state(plan, params, rt):
    """Consistent arbitrary fluxes, not E divided by a tensor diagonal.

    This internal initial-state fixture bypasses no material or step rule.
    The public E/H initial-field inverse solve is explicitly refused for now.
    """
    state = initial_state(plan, params, rt)
    rng = np.random.default_rng(224)
    for field, key in (('E', 'be'), ('H', 'bh')):
        flux = rt.array(rng.normal(size=plan.shape), 'float64')
        for term in plan.constitutive:
            if term.field == field:
                local = rt.fresh(flux[term.view]*1.)
                for mask in term.masks:
                    local = rt.slab_mul(local, mask, 0.)
                state['mem'][term.name] = local
        value = params[key]/(plan.dt/plan.dx)*flux
        for term in plan.constitutive:
            if term.field == field:
                value = term.add(value, cross_action(state['mem'][term.name],
                    params['constitutive'][term.name], rt.xp, field, bounded=True), rt)
        for mask in plan.masks:
            value = getattr(mask, 'after_' + field)(value, rt)
        state[field] = value
    return state


def solve_cavity(backend, scene):
    rt = runtime('jax' if backend == 'jit' else backend)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, rt)
    state = seeded_state(plan, params, rt)
    def run(state):
        if backend == 'jax':
            for n in range(plan.steps):
                state = step(plan, params, state, n, rt)
            return state
        return rt.scan(lambda s, n: step(plan, params, s, n, rt), state, plan.steps)
    if backend == 'jit':
        import jax
        run = jax.jit(run)
    result = run(state)
    return {k: to_numpy(result[k]) for k in ('E', 'H', 'energy')}


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('field', ('E', 'H'))
def test_common_point_action_has_the_staggered_fourier_symbol(backend, field):
    rt = runtime('jax' if backend == 'jit' else backend)
    n = 7
    z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    wave = 2*np.pi*np.array([1, 2, 3])/n
    phase = wave[0]*x + wave[1]*y + wave[2]*z
    offsets = np.eye(3)/2 if field == 'E' else (1-np.eye(3))/2
    phases = [phase + np.dot(wave, offsets[i]) for i in range(3)]
    amplitude = np.array([.7, -.3, 1.2])
    vector = np.stack([amplitude[i]*np.cos(phases[i]) for i in range(3)])
    matrix = np.array([[0., .2, -.13], [.2, 0., .07], [-.13, .07, 0.]])
    cross = np.array([.2, -.13, .07])[:, None, None, None]*np.ones_like(vector)
    c = np.cos(wave/2)
    expected = np.stack([np.cos(phases[i])*c[i]*np.dot(matrix[i], c*amplitude)
                         for i in range(3)])
    action = lambda v, k: cross_action(v, k, rt.xp, field)
    if backend == 'jit':
        import jax
        action = jax.jit(action)
    actual = to_numpy(action(rt.array(vector), rt.array(cross)))
    # E-symbol construction gap measured 3.44e-16 in the independent study;
    # 3e-14 separates roundoff from the O(1) wrong-adjoint/half-position mutants.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-14)


@pytest.mark.parametrize('field', ('E', 'H'))
def test_finite_box_operator_is_adjoint_and_encloses_its_exact_eigenvalues(field):
    # Explicit dense matrix assembled by independent neighbour weights. The
    # fixture is tiny so eigvalsh can judge BOTH ends of the claimed enclosure.
    shape = (3, 2, 2, 3)
    rng = np.random.default_rng(923)
    cross = rng.uniform(-.15, .15, size=shape)
    diagonal = rng.uniform(.8, 1.2, size=shape)
    count = int(np.prod(shape))
    matrix = np.diag(diagonal.ravel())
    sign = -1 if field == 'E' else 1
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        for node in np.ndindex(shape[1:]):
            for di in (0, sign):
                for dj in (0, sign):
                    dest, src = list(node), list(node)
                    dest[2-i] += di
                    src[2-j] += dj
                    if any(v < 0 or v >= n for v, n in zip(dest, shape[1:])):
                        continue
                    if any(v < 0 or v >= n for v, n in zip(src, shape[1:])):
                        continue
                    a = np.ravel_multi_index((i, *dest), shape)
                    b = np.ravel_multi_index((j, *src), shape)
                    matrix[a, b] += cross[(k, *node)]/4
                    matrix[b, a] += cross[(k, *node)]/4
    vector = rng.normal(size=shape)
    actual = diagonal*vector + cross_action(vector, cross, np, field, bounded=True)
    # Exact four weights of 1/4; measured Fourier/adjoint gap <4e-16.
    np.testing.assert_allclose(actual.ravel(), matrix@vector.ravel(), rtol=0, atol=3e-14)
    values = np.linalg.eigvalsh(matrix)
    lo, hi = enclosure(diagonal, cross, np, field)
    assert lo <= values[0] and hi >= values[-1]
    assert lo > 0
    # A body-diagonal normal makes the lower radius indispensable (an
    # uncoupled third row otherwise hides an omitted-radius mutant).
    uniform = np.full(shape, .7)
    cross = np.full(shape, .05)
    lo, hi = enclosure(uniform, cross, np, field)
    assert lo < .7 and hi > .7


@pytest.mark.parametrize('backend', BACKENDS)
def test_closed_electric_and_magnetic_coupling_conserves_energy(backend):
    scene = cavity()
    result = solve_cavity(backend, scene)
    reference = solve_cavity('numpy', scene)
    energy = result['energy']
    # 80-step 16^3 electric+magnetic cavity: measured relative energy span
    # <=2.86e-15 and max backend field difference <=3.31e-14, seed 224.
    assert energy.min() > 0
    assert np.ptp(energy)/np.mean(energy) < 3e-13
    for key in result:
        np.testing.assert_allclose(result[key], reference[key], rtol=3e-13, atol=3e-12)


@pytest.mark.parametrize('field', ('E', 'H'))
def test_coupled_initial_field_is_refused_before_lowering(field):
    scene = cavity().replace(initial=fd.InitialField(**{field: np.zeros((3,16,16,16))}))
    with pytest.raises(ValueError, match='InitialField.' + field + '.*inverse material solve'):
        quiet(fd.plan, scene)
    assert quiet(fd.plan, cavity().replace(structures=())).constitutive == ()


def test_check_counts_the_compact_flux_memory():
    scene = cavity()
    plan = quiet(fd.plan, scene)
    result = quiet(fd.check, scene)
    expected = (2*np.prod(plan.shape) + sum(3*np.prod(t.region.shape)
                                           for t in plan.constitutive))*8
    assert result.bytes_per_state == expected
    assert all(np.prod(t.region.shape) < plan.cells for t in plan.constitutive)


@pytest.mark.parametrize('center', ((-.25, 0., 0.), (.20, 0., 0.)))
def test_wall_projection_and_finite_interpolation_preserve_energy(center):
    result = solve_cavity('numpy', cavity(center=center))
    # Measured spans 2.86e-15 / 3.70e-15 for a tangent low face and a
    # surface one cell from the high face, including magnetic coupling.
    assert np.ptp(result['energy'])/np.mean(result['energy']) < 3e-13


@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_coupled_material_gradient_matches_central_differences(backend):
    scene = cavity(steps=8)
    plan = quiet(fd.plan, scene)
    rt = runtime(backend)
    def loss(eps):
        material = fd.Material(eps=eps, mu=1.5)
        s = scene.replace(structures=(fd.Structure(scene.structures[0].shape, material),))
        params = fd.lower(s, plan, rt)
        state = seeded_state(plan, params, rt)
        for n in range(plan.steps):
            state = step(plan, params, state, n, rt)
        return rt.xp.sum(state['E']**2 + .3*state['H']**2)
    if backend == 'jax':
        import jax
        gradient = float(jax.jit(jax.grad(loss))(rt.array(2.25)))
    else:
        value = rt.array(2.25).requires_grad_(True)
        loss(value).backward()
        gradient = float(value.grad)
    # Step sweep 1e-2/1e-3/1e-4/1e-5: relative errors
    # 3.09e-5/3.09e-7/3.03e-9/1.18e-9. Use 1e-4 before subtraction noise.
    h = 1e-4
    finite = float(to_numpy((loss(2.25+h)-loss(2.25-h))/(2*h)))
    assert abs(finite/gradient-1) < 1e-7


@pytest.mark.parametrize('backend', BACKENDS)
def test_lowering_places_cross_terms_at_e_nodes_and_h_centres(backend):
    from wavebit.fd._shapes import Sphere
    scene = cavity()
    plan = quiet(fd.plan, scene)
    rt = runtime('jax' if backend == 'jit' else backend)
    function = lambda: fd.lower(scene, plan, rt)['constitutive']
    if backend == 'jit':
        import jax
        function = jax.jit(function)
    result = function()
    for term in plan.constitutive:
        z, y, x = np.meshgrid(*(np.arange(a, b) for a, b in term.box), indexing='ij')
        points = np.stack([x, y, z], axis=-1).astype(float)
        points = plan.low + plan.dx*(points + (0 if term.field == 'E' else .5))
        body = Sphere(.15, (.012, .007, -.009))
        fill = body.exact_fill(points, plan.dx)
        normal = points-np.array(body.center)
        normal /= np.linalg.norm(normal, axis=-1)[..., None]
        value = 2.25 if term.field == 'E' else 1.5
        # Independent continuity conditions: inverse epsilon is <1/eps>
        # along n and 1/<eps> along any tangent, so their difference is cross K.
        difference = fill/value + 1-fill - 1/(fill*value+1-fill)
        expected = np.stack([difference*normal[..., i]*normal[..., j]
                             for i, j in ((0,1), (0,2), (1,2))])
        # Existing exact-fill cross-backend sweep measured <=6.16e-14
        # relative in coefficients; this absolute allowance is 3e-12.
        np.testing.assert_allclose(to_numpy(result[term.name]), expected, rtol=0, atol=3e-12)


@pytest.mark.parametrize('field', ('E', 'H'))
@pytest.mark.parametrize('backend', BACKENDS)
def test_flux_memory_integrates_the_source_current(field, backend):
    from wavebit.fd._kernel.sources import DipoleTerm
    source = fd.Dipole('d', center=(.10, .08, .04), component=field+'x',
                       kind='electric' if field == 'E' else 'magnetic',
                       band=fd.Band(wavelength=2., df=.1))
    scene = cavity(steps=1).replace(sources=(source,))
    plan = quiet(fd.plan, scene)
    rt = runtime('jax' if backend == 'jit' else backend)
    params = fd.lower(scene, plan, rt)
    # A unit current sample isolates the step slot from pulse timing.
    params['waveform'] = rt.array(np.ones((1, 1)))
    state = initial_state(plan, params, rt)
    function = lambda s: step(plan, params, s, 0, rt)
    if backend == 'jit':
        import jax
        function = jax.jit(function)
    result = function(state)
    dipole = next(t for t in plan.terms if isinstance(t, DipoleTerm))
    term = next(t for t in plan.constitutive if t.field == field)
    expected = np.zeros(plan.shape)
    # Both D and B integrate minus the corresponding current: -dt J/dx^3.
    expected[dipole.slab.slices] = (-plan.dt/plan.dx**3
                                   * to_numpy(params['sources']['d']['stamp']))
    # A source-only first step has no incoming curl in its driven field.
    np.testing.assert_allclose(to_numpy(result['mem'][term.name]), expected[term.view],
                               rtol=3e-14, atol=3e-13)
    key = 'be' if field == 'E' else 'bh'
    response = to_numpy(params[key])/term.ratio * expected
    response[term.view] += cross_action(expected[term.view],
        to_numpy(params['constitutive'][term.name]), np, field, bounded=True)
    np.testing.assert_allclose(to_numpy(result[field]), response, rtol=3e-14, atol=3e-13)


def test_the_lift_plans_eps_100_and_a_traced_failure_is_still_visible():
    # eps=100 at this 16^3 placement was refused by the Gershgorin lower bound; its
    # unlifted operator really is indefinite (test_an_unlifted_eps_100_sphere_grows).
    # The star lift plans it. eps=2.25 at courant=.999 still fails the coupled
    # curl bound, which the lift does not touch.
    quiet(fd.plan, cavity(eps=100.))
    with pytest.raises(ValueError, match='Constitutive CFL bound'):
        quiet(fd.plan, cavity(mu=None).replace(run=fd.Run(steps=4, courant=.999)))
    plan = quiet(fd.plan, cavity())
    rt = runtime('jax')
    import jax
    @jax.jit
    def lower(eps):
        return fd.lower(cavity(eps=eps), plan, rt)
    good, lifted, invalid = lower(rt.array(2.25)), lower(rt.array(100.)), lower(rt.array(.1))
    assert float(good['stable']) == float(lifted['stable']) == 1.
    np.testing.assert_allclose(to_numpy(lifted['be']),
                               fd.lower(cavity(eps=100.), plan, runtime('numpy'))['be'],
                               rtol=1e-12, atol=0)
    # 1/eps = 10 against mu = 1.5 at courant .5: dt^2 (3/dx^2) (10/1.5) = 1.67 > 1.
    assert np.isnan(float(invalid['stable']))
    assert np.isnan(to_numpy(invalid['be'])).all()


@pytest.mark.parametrize('lifted', (True, False))
def test_an_unlifted_eps_100_sphere_grows(lifted, monkeypatch):
    # The control the lift is judged by: the same seeded cavity with the lift turned off
    # (the Gershgorin lower bound refused it before, and nothing else would stop it). Judge
    # the fields, not the energy: leapfrog conserves the discrete energy exactly, so an
    # indefinite one stays flat while its positive and negative parts grow together.
    # Measured (2026-09-27), this test's max(|E|, |H|) over 300 steps: x0.90 lifted (at every
    # THETA from 0.02 to 0.3), x1.9e7 unlifted. The unlifted energy read 21498.87 to about
    # step 160 while the fields grew, then drifted with the round-off of cancelling ~1e9 norms
    # (25600 at 300) and read -3.2e47 at 1000 (the review of the star corrected the first note,
    # which read the energy at 300 as flat and the lifted growth off |E| alone).
    from wavebit.fd._lower import stars
    if not lifted:
        monkeypatch.setattr(stars, 'lift', lambda *a, **k: stars.Lifted(None, True, 0, 0, 0.))
    size = lambda r: max(np.linalg.norm(r['E']), np.linalg.norm(r['H']))
    first = solve_cavity('numpy', cavity(eps=100., mu=None, steps=1))
    last = solve_cavity('numpy', cavity(eps=100., mu=None, steps=300))
    growth = size(last)/size(first)
    if lifted:
        assert growth < 10 and np.ptp(last['energy'])/np.mean(last['energy']) < 3e-13
    else:
        assert growth > 1e6


def test_disjoint_and_merged_supports_own_flux_once():
    # Overlapping support boxes are legal when the surfaces do not share a
    # voxel. The union must own ONE memory, independent of painting order.
    for distance, count in ((.6, 2), (.36, 1)):
        pair = tuple(fd.Structure(fd.Sphere(radius=.15, center=(x, 0., 0.)), 2.25)
                     for x in (-distance/2, distance/2))
        scene = cavity(mu=None).replace(domain=fd.Domain(size=(1.4,)*3), structures=pair)
        from wavebit.fd._lower.constitutive import terms
        base = quiet(fd.plan, scene.replace(structures=()))
        assert len(terms(scene, base)) == count
        plan = quiet(fd.plan, scene)
        assert len(plan.constitutive) == count
        reverse = quiet(fd.plan, scene.replace(structures=pair[::-1]))
        a, b = fd.lower(scene, plan), fd.lower(scene.replace(structures=pair[::-1]), reverse)
        for term in plan.constitutive:
            np.testing.assert_allclose(a['constitutive'][term.name],
                                       b['constitutive'][term.name], rtol=0, atol=3e-12)


def test_plan_checks_surfaces_at_common_nodes_too():
    from wavebit.fd._lower.checks import _shared_surface_voxels
    # A sweep of diagonal dimer gaps .9..1.7 cells found this last case:
    # zero shared Yee voxels, one common-node voxel with each fill 3.55e-6.
    half = (.4 + 1.7*.05)/(2*np.sqrt(3))
    shapes = tuple(fd.Sphere(radius=.2, center=(sign*half,)*3) for sign in (-1, 1))
    scene = cavity().replace(domain=fd.Domain(size=(1.2,)*3),
                            structures=tuple(fd.Structure(s, 2.25) for s in shapes))
    grid = quiet(fd.plan, scene.replace(structures=())).grid
    a, b = [s.support(grid) for s in shapes]
    box = tuple((max(x[0], y[0]), min(x[1], y[1])) for x, y in zip(a, b))
    assert _shared_surface_voxels(*(s.body() for s in shapes), grid, box)[0] == 0
    with pytest.raises(ValueError, match='same voxel: 1.*E common'):
        quiet(fd.plan, scene)


def test_loss_and_mirror_doors_cover_the_interpolation_halo():
    scene = cavity(mu=None)
    pml = fd.Boundaries.open(fd.PML(cells=3, wavelength=2., allow_thin=True))
    with pytest.raises(ValueError, match='interpolation support meets.*absorbing layer'):
        quiet(fd.plan, scene.replace(boundaries=pml))
    loss = fd.Structure(fd.Region(size=(.05,.1,.1), center=(.275,0.,0.)),
                        fd.Conductor(eps=1., sigma=.2))
    with pytest.raises(ValueError, match='interpolation support meets.*conductivity or poles'):
        quiet(fd.plan, scene.replace(structures=scene.structures+(loss,)))
    mirror = fd.Boundaries(x=(fd.Symmetry(parity=1), fd.PEC()), y=fd.PEC(), z=fd.PEC())
    # Tangent to the plane, the ball meets checks.check_symmetry first. Measured on
    # 2026-09-28 at dx=.05: 0.6 cells clear (x=-.22), the H plane-layer fill reaches
    # .285815030367076 and |cross| .004563667272660, so the adjoint door must remain.
    # At 1.0 cells clear (x=-.20), E fills are zero and H fills only -1.45219e-15;
    # the old |cross|=3.32200e-17 refusal was round-off, not a reaching surface.
    with pytest.raises(ValueError, match='within half a cell of the x- Symmetry plane'):
        quiet(fd.plan, cavity(center=(-.25,0.,0.)).replace(boundaries=mirror))
    for x in (-.225, -.22):
        with pytest.raises(ValueError, match='mirror-weighted adjoint'):
            quiet(fd.plan, cavity(center=(x,0.,0.)).replace(boundaries=mirror))
    for x in (-.20, -.19):
        quiet(fd.plan, cavity(center=(x,0.,0.)).replace(boundaries=mirror))


def test_equal_phases_have_no_cross_coupling_and_box_faces_have_none():
    from test_fd_occupancy import _Ball
    from wavebit.fd._shapes import Box
    class FlatFaces(_Ball):
        def body(self):
            return Box(size=(.25,.30,.35))
    sphere = cavity(mu=None).replace(background=2.25)
    box = cavity(mu=None).replace(structures=(fd.Structure(FlatFaces(.3), 2.25),))
    plan = quiet(fd.plan, sphere)
    for cross in fd.lower(sphere, plan)['constitutive'].values():
        assert np.array_equal(cross, np.zeros_like(cross))
    # An unturned box's cross coefficients are zero everywhere, so since C4 it is planned no
    # constitutive term at all (constitutive.cross_free) rather than one full of zeros.
    from wavebit.fd._kernel.constitutive import ConstitutiveTerm
    plan = quiet(fd.plan, box)
    assert not [t for t in plan.terms if isinstance(t, ConstitutiveTerm)]
    assert not fd.lower(box, plan).get('constitutive')


def test_a_round_off_fill_leaves_the_node_to_the_earlier_body():
    """A cylinder raised half a cell above a smoothed substrate whose face is a node plane:
    its base is the top face of the plane nodes' voxels, where coordinate round-off leaves it
    a fill of about 4e-15 (the junction study's harness grid) instead of zero. The doors read
    that as empty and admit the scene; the node rule once tested ``fill == 0`` and let the
    cylinder overwrite the substrate's node tensor there -- Xi_zz 1/<eps> = 0.615 where the
    substrate's <1/eps> is 0.722. The base is set 1e-12 of a cell into those voxels so the
    round-off does not depend on how this grid's coordinates happen to round."""
    from wavebit.fd._lower import constitutive
    from wavebit.fd._lower.constitutive import common_points
    from wavebit.fd._shapes import occupancy
    dx, eps_s = .05, 2.25
    glass = fd.Material(eps=eps_s)
    z_lo, z_hi = -1.2, 1.5                        # node planes at multiples of dx: z = 0 is one
    deep = z_lo - 2*dx                            # the substrate runs through the z- PML
    base = dx/2 - 1e-12*dx
    pillar = fd.Cylinder(radius=.15, height=.61, axis='z', center=(0., 0., base + .305))
    scene = fd.Scene(
        domain=fd.Domain(size=(.5, .5, z_hi - z_lo), center=(0., 0., (z_hi + z_lo)/2)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                 z=(fd.PML(cells=12, wavelength=1., background=glass),
                                    fd.PML(cells=12, wavelength=1.))),
        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=2),
        structures=(fd.Structure(fd.Box(size=(10., 10., -deep), center=(0., 0., deep/2)), glass),
                    fd.Structure(pillar, fd.Material(eps=4.))))
    plan = quiet(fd.plan, scene)
    nodes = {}
    quiet(constitutive.lower, scene, plan, runtime('numpy'), nodes=nodes)
    term = [t for t in plan.constitutive if t.field == 'E'][0]
    points = common_points(plan.grid, tuple(reversed(term.box)), 'E')
    fill = np.asarray(occupancy(pillar.body(), points, dx)[0])
    alpha = np.asarray(occupancy(scene.structures[0].shape.body(), points, dx)[0])
    touched = (np.abs(fill) > 0) & (np.abs(fill) < 1e-9) & (alpha > 1e-9) & (alpha < 1 - 1e-9)
    assert touched.sum() >= 20                    # the plane nodes under the pillar's base
    tensor, inverse = (np.asarray(v) for v in nodes[term.name])
    a = alpha[touched]
    np.testing.assert_allclose(tensor[2][touched], a/eps_s + (1 - a), rtol=0, atol=1e-12)
    np.testing.assert_allclose(tensor[0][touched], 1/(a*eps_s + (1 - a)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(inverse[touched], 1/(a*eps_s + (1 - a)), rtol=0, atol=1e-12)
