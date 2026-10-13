"""Traced material values beside traced geometry: a pillar's index with its shape.

The scenes are ``studies/fdtd/traced_material/fdtd_traced_material_probe.py``'s (results in
``studies/results/fdtd_traced_material_*.json``). Host planning reads a traced ``eps`` or ``mu``
at the plan's own value (``src/wavebit/fdtd/_prepare/shape_ranges.py::_seeded_materials``), lowering
reads it live, and the runtime guards judge the live value
(``shape_ranges.resolution_guard`` beside the existing ones).
"""
import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._prepare.shape_ranges import planning_scene
from wavebit.fdtd._runtime import runtime
from studies.fdtd.traced_material.fdtd_traced_material_probe import MULTI, PARTNERS, POINTS, SPHERE, SUBSTRATE, closed_scene, dimer, junction_scene, multi_objective, multi_scene, objectives, open_box, partners_loss, pml_box, scene, sphere_objective, sphere_scene, substrate_objectives, substrate_scene
from tests.fdtd.test_fd_sphere_range import quiet

TOLERANCE = 128 * np.finfo(float).eps
#: ``grazing.GRAZING_EXPONENT``, the grading's ``g`` at its design wavelength.
GRAZING_G = .919


def _traced(values, keep=()):
    """Torch leaves for every value, except the indices in ``keep``."""
    import torch
    return [v if i in keep else torch.tensor(v, dtype=torch.float64, requires_grad=True)
            for i, v in enumerate(values)]


def _ran(function, *args, **kwargs):
    try:
        return quiet(function, *args, **kwargs)
    except Exception as error:
        raise AssertionError(f'a certified traced scene failed: {error!r}') from error


def _close(got, want):
    np.testing.assert_allclose(got, want, rtol=0, atol=TOLERANCE * np.max(abs(want)))


@pytest.mark.parametrize('ranged', (True, False))
@pytest.mark.parametrize('kind', ('cylinder', 'box'))
def test_a_traced_index_lowers_and_runs_as_the_concrete_scene(kind, ranged):
    """Four (geometry, eps) points from the seed's plan -- with the geometry traced too, or
    concrete and unranged with eps alone traced -- against the concrete NumPy scene planned at
    each point. The cylinder plans a diagonal-rule term, so its material-only run is planned at
    the seed; the Box plans none (a junction alone), so its traced value is planned live, as
    before material seeds, with or without plan=. Measured: coefficients within 9.2e-15 of
    their peak, fields within 5.4e-16."""
    rtt = runtime('torch')
    seed = POINTS[kind][0]
    p = quiet(fd.plan, scene(seed, kind=kind, ranged=ranged, steps=40))
    points = POINTS[kind] if ranged else tuple(seed[:3] + (e,) for e in (2.25, 1.7, 2.5, 1.3))
    for v in points:
        traced = scene(_traced(v, keep=() if ranged else (0, 1, 2)), kind=kind, ranged=ranged,
                       steps=40)
        concrete = scene(v, kind=kind, ranged=ranged, steps=40)
        q = quiet(fd.plan, concrete)
        got, want = _ran(fd.lower, traced, p, 'torch'), quiet(fd.lower, concrete, q)
        assert got['be'].requires_grad
        _close(rtt.numpy(got['be'].detach()), want['be'])
        for name in want.get('constitutive', {}):
            _close(rtt.numpy(got['constitutive'][name].detach()), want['constitutive'][name])
        ran, direct = _ran(fd.run, traced, 'torch', plan=p), quiet(fd.run, concrete, plan=q)
        for f in ('E', 'H'):
            _close(rtt.numpy(getattr(ran.final, f).detach()), getattr(direct.final, f))
        assert any('material value is traced' in m for m in ran.report.warnings) == (
            ranged or kind == 'cylinder')
    if not ranged and kind == 'box':
        free = _ran(fd.run, scene(_traced(seed, keep=(0, 1, 2)), kind=kind, ranged=False,
                                  steps=40), 'torch')
        assert np.isfinite(rtt.numpy(free.final.E.detach())).all()


#: (kind, backend). Measured largest relative Jacobian error of NumPy central differences
#: (each geometry planned on its own) against reverse mode, over (radius, height, centre x,
#: eps) -- (x size, y size, height, eps) for the Box -- at steps 1e-4, 1e-5, 3e-6, 1e-6:
#: cylinder 0.27, 1.3e-6, 9.2e-8, 6.4e-8 (the eps column alone 1.6e-9, 9.2e-10, 3.8e-9,
#: 3.1e-10; the 1e-4 row crosses the cylinder's own wall/cap switch); box 3.9e-7, 3.9e-9,
#: 2.5e-9, 1.3e-8. Step 3e-6, the existing 2e-6 budget. JAX and Torch agree to 3.1e-15.
GRADIENT_CASES = (('cylinder', 'jax'), ('cylinder', 'torch'), ('box', 'torch'))


@pytest.mark.parametrize('kind,backend', GRADIENT_CASES)
def test_the_index_and_the_shape_differentiate_together(kind, backend):
    """The transmitted and reflected zero orders in the pillar's shape and permittivity,
    against the concrete solver as a user runs it."""
    point = np.array(POINTS[kind][0])
    p = quiet(fd.plan, scene(point, kind=kind))
    h = 3e-6

    def values(v):
        return np.array([float(x) for x in quiet(objectives, v, 'numpy', None, kind=kind)])
    expected = np.array([(values(point + d) - values(point - d)) / (2 * h)
                         for d in np.eye(4) * h]).T
    runtime(backend)
    if backend == 'jax':
        import jax
        import jax.numpy as jnp

        def fn(v):
            return jnp.stack(_ran(objectives, v, 'jax', p, kind=kind))
        got = np.asarray(jax.jacrev(fn)(jnp.asarray(point)))
    else:
        import torch
        v = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        out = _ran(objectives, v, 'torch', p, kind=kind)
        got = np.array([torch.autograd.grad(o, v, retain_graph=True)[0].numpy() for o in out])
    assert np.all(expected != 0)
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=0.)


def test_four_magnetic_bodies_differentiate_eps_mu_and_size():
    """``fdtd_multi_shape_grad_probe``'s four separated bodies, each one's size, eps and mu
    traced together (12 leaves), against central differences at the seed's plan. Measured
    at steps 1e-4 .. 1e-6: every material leaf within 3.8e-7 at every step (the largest is
    the sphere's mu, whose derivative is -0.0144); the cylinder's radius crosses a switch of
    its own rule within 1e-5 (2.2e-4 at 1e-5, 7.4e-4 at 3e-6, as in the multi-shape record)
    and agrees to 1.2e-9 at 1e-6. Step 1e-6, the existing 2e-6 budget."""
    import torch
    p = quiet(fd.plan, multi_scene())
    h = 1e-6
    expected = np.array([(multi_objective(MULTI + d, 'numpy', p)
                          - multi_objective(MULTI - d, 'numpy', p)) / (2 * h)
                         for d in np.eye(12) * h])
    v = torch.tensor(MULTI, dtype=torch.float64, requires_grad=True)
    _ran(multi_objective, v, 'torch', p).backward()
    np.testing.assert_allclose(v.grad.numpy(), expected, rtol=2e-6, atol=0.)


def test_a_smoothed_body_differentiates_its_material_through_run():
    """An unranged sphere's eps and mu alone through ``fd.run``: before material seeds the plan
    rebuilt there refused a traced value (``constitutive.validate``). Measured against central
    differences: 9.9e-9, 2.4e-9, 5.7e-9, 5.7e-9 at steps 1e-4 .. 1e-6; JAX and Torch 1.7e-14.
    Step 1e-5, the existing 2e-6 budget."""
    import torch
    p = quiet(fd.plan, sphere_scene())
    h = 1e-5
    expected = np.array([(sphere_objective(SPHERE + d, 'numpy', p)
                          - sphere_objective(SPHERE - d, 'numpy', p)) / (2 * h)
                         for d in np.eye(2) * h])
    v = torch.tensor(SPHERE, dtype=torch.float64, requires_grad=True)
    _ran(sphere_objective, v, 'torch', p).backward()
    np.testing.assert_allclose(v.grad.numpy(), expected, rtol=2e-6, atol=0.)


#: (eps, stable). The resolution floor at dx .05 and the band's shortest wavelength .642 is
#: eps 2.576; the time-step bound at courant .5 needs eps above .25; positivity; and the
#: three values the lift's host eigensolver raised on before any guard ran.
GUARD_CASES = ((2.25, True), (2.7, False), (.2, False), (-1., False), (0., False),
               (float('nan'), False), (float('inf'), False))


@pytest.mark.parametrize('eps,stable', GUARD_CASES)
def test_a_traced_index_past_a_door_runs_to_nan(eps, stable):
    """The run returns NaN fields and ``stable`` False where the concrete path refuses the same
    value by name (``fdtd_traced_material_guards.json``)."""
    rtt = runtime('torch')
    p = quiet(fd.plan, scene(POINTS['cylinder'][0], steps=20))
    v = _traced(POINTS['cylinder'][0][:3] + (eps,))
    result = _ran(fd.run, scene(v, steps=20), 'torch', plan=p)
    assert bool(result.diagnostics['stable']) is stable
    assert bool(np.isfinite(rtt.numpy(result.final.E.detach())).all()) is stable
    if not stable and np.isfinite(eps) and eps != 0:
        with pytest.raises(ValueError, match='cells per wavelength|stability bound|positive'):
            quiet(fd.plan, scene(POINTS['cylinder'][0][:3] + (eps,), steps=20))


def _cell(steps=1):
    """The probe's cylinder cell without sources or monitors, for planning-only refusals."""
    return scene(POINTS['cylinder'][0], steps=steps).replace(sources=(), monitors=(),
                                                             run=fd.Run(steps=steps))


def _t(x):
    import torch
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)


def stabilized(s, rho=.05):
    """``s`` with its bottom PML stabilized."""
    z = s.boundaries.z
    return s.replace(boundaries=s.boundaries.replace(z=(z[0].replace(stabilize=rho), z[1])))


def _refused(message, function, *args, **kwargs):
    """``function`` refuses by name: a ValueError matching ``message``. Any other outcome --
    no refusal, another reason, another exception -- fails by assertion."""
    import re
    try:
        quiet(function, *args, **kwargs)
    except ValueError as error:
        assert re.search(message, str(error)), f'refused for another reason: {error}'
        return
    except Exception as error:
        raise AssertionError(f'expected a refusal by name, got {error!r}') from error
    raise AssertionError(f'not refused: {message}')


def test_what_no_runtime_guard_covers_is_refused_by_name(monkeypatch):
    """Each refusal of ``_seeded_materials``, on ``fd.run``'s path (``planning_scene``) and on a
    direct ``fd.lower`` of concrete geometry. The admitted neighbours of each door are
    ``test_the_neighbours_of_each_door_run_as_their_concrete_scenes``."""
    p = quiet(fd.plan, _cell())
    full = quiet(fd.plan, scene(POINTS['cylinder'][0]))
    traced = scene(_traced(POINTS['cylinder'][0][:3]) + [2.25])
    sub, pillar = traced.structures
    e = _t(2.25)
    cases = (
        (traced.replace(structures=(sub, pillar.replace(material=fd.Material(
            eps=(e, e * 1., e))))), p, r'traced Cylinder requires isotropic lossless Material'),
        # The substrate fills the bottom PML along the PlaneWave's axis, where the T plane's
        # orders are judged (grazing._survey): on a stabilized face, on its exact return too.
        (stabilized(traced).replace(structures=(sub.replace(material=fd.Material(
            eps=_t(2.1))), pillar)), quiet(fd.plan, stabilized(scene(POINTS['cylinder'][0]))),
         r'structures\[0\] has a traced material value and fills a stabilized PML layer'),
        (traced.replace(structures=(sub.replace(material=fd.Conductor(eps=2.1, sigma=_t(.1))),
                                    pillar)), p,
         r'structures\[0\] has a traced value in an fd.Conductor'),
        (traced.replace(structures=(sub, pillar.replace(material=fd.Material(
            eps=_t(2.25), mu=1.1)))), p,
         r"structures\[1\]'s material carries a traced value.*naming mu"),
        (scene(POINTS['cylinder'][0]).replace(background=fd.Material(eps=_t(1.))), full,
         r'Scene\(background=\) has a traced value'))
    for current, plan, message in cases:
        _refused(message, planning_scene, current, plan)
    # Without the Diffraction monitors the survey judges no orders, so nothing on the host
    # reads the traced substrate's value, PlaneWave or not, stabilized face or not.
    for sources in ((), traced.sources):
        q = quiet(fd.plan, stabilized(scene(POINTS['cylinder'][0]).replace(sources=sources,
                                                                            monitors=())))
        _ran(planning_scene, stabilized(traced).replace(structures=(sub.replace(
            material=fd.Material(eps=_t(2.1))), pillar), sources=sources, monitors=()), q)
    # The background's three readers, each alone: the PlaneWave with both PMLs matched to
    # media of their own; a PML matched to it in a Dipole scene; a lattice FarField in an
    # Absorber box (results refuses a traced one only after the run).
    def own_pmls(x):
        s = scene(POINTS['cylinder'][0]).replace(background=x)
        z = s.boundaries.z
        return s.replace(boundaries=s.boundaries.replace(z=(z[0], z[1].replace(
            background=fd.Material(eps=1.)))))
    for make in (own_pmls, lambda x: pml_box(background=x, own=False), open_box):
        q = quiet(fd.plan, make(1.))
        _refused(r'Scene\(background=\) has a traced value', planning_scene,
                 make(fd.Material(eps=_t(1.2))), q)
    # A Diffraction plane in a traced medium needs orders=: a film in the air above the
    # PlaneWave's sheet (z .45, cells 29..30), z .6 to .8 (support from cell 31), filling the
    # R plane's two cell layers, with the domain raised so that the top PML starts at z 1.1.
    film = fd.Structure(fd.Box(size=(10., 10., .2), center=(0., 0., .7)),
                        fd.Material(eps=1.4))
    open_ = scene(POINTS['cylinder'][0], ranged=False).replace(
        domain=fd.Domain(size=(.6, .6, 2.5), center=(0., 0., .25)))
    reflected = lambda **orders: (fd.Diffraction('R', z=.7, side='reflect',
                                                 wavelengths=(1.,), **orders),)
    open_ = open_.replace(structures=open_.structures + (film,), monitors=reflected())
    q = quiet(fd.plan, open_)
    moved = open_.replace(structures=open_.structures[:2] + (film.replace(
        material=fd.Material(eps=_t(1.4))),))
    _refused(r"Diffraction\('R'\).*give orders=", planning_scene, moved, q)
    _refused(r"Diffraction\('R'\).*give orders=", fd.lower, moved, q, 'torch')
    _ran(planning_scene, moved.replace(monitors=reflected(orders=0)), q)
    # A sphere centred on a Symmetry face is planned only where its contrast is zero there; a
    # traced value would then couple across the mirror unchecked.
    _refused('reaches a Symmetry plane', fd.plan, closed_scene(mirror=True))
    mirrored = quiet(fd.plan, closed_scene(mirror=True, eps=1.))
    for run in (lambda s: planning_scene(s, mirrored), lambda s: fd.lower(s, mirrored, 'torch')):
        _refused('reaches the Symmetry plane.*traced', run, closed_scene(mirror=True,
                                                                         eps=_t(2.25)))
    # The direct lower of concrete geometry hears the per-axis refusal too.
    closed = quiet(fd.plan, closed_scene())
    _refused('traced eps given per axis', fd.lower, closed_scene(eps=(e, e * 1., e)), closed,
             'torch')
    # A seeded rebuild emits no warnings, and the skin-depth refusal lives among them
    # (checks._resolution_notes); a concrete Drude partner beside the traced sphere is held to
    # it before the run (_run._planned), not only by the report after it. Its skin depth at
    # the band's wavelength 3 is 0.22 at omega_p 5 and 0.033 at 30, against dx .1.
    def metal(omega_p, eps=2.25):
        return closed_scene(eps=eps, extra=(fd.Structure(
            fd.Region(size=(.1, .1, .1), center=(.7, -.7, -.7)),
            fd.Dispersive(eps_inf=1., poles=(fd.Drude(omega_p=omega_p, gamma=.5),))),))
    with pytest.raises(ValueError, match='skin depth'):
        quiet(fd.plan, metal(30.))
    lossy = quiet(fd.plan, metal(5.))
    _ran(fd.run, metal(5., eps=_t(2.25)), 'torch', plan=lossy)
    from wavebit.fdtd import _execution as _run

    def solved(*args, **kwargs):
        raise AssertionError('the skin-depth refusal came only after the run')
    monkeypatch.setattr(_run, '_solve', solved)
    with pytest.raises(ValueError, match='skin depth'):
        quiet(fd.run, metal(30., eps=_t(2.25)), 'torch', plan=lossy)


def test_the_neighbours_of_each_door_run_as_their_concrete_scenes():
    """What each door leaves admitted, against the concrete scene at the same values
    (``fdtd_traced_material_extras.json``, every gap within 8.9e-15 of the peak): a traced
    background in a closed cell, and beside PMLs that carry their own; the sphere a cell clear
    of its Symmetry face (the term box touches it, no fill there is fractional); a traced film
    in a Dipole scene's PML; a traced diagonal on a snapped block clear of the sphere; and a
    traced conductivity lowered directly, which ``fd.run`` refuses but ``fd.lower`` lowers as
    it always has."""
    rtt = runtime('torch')
    block = lambda material: fd.Structure(fd.Region(size=(.1, .1, .1), center=(.7, -.7, -.7)),
                                          material)
    runs = (
        (lambda x: closed_scene(background=x), fd.Material(eps=_t(1.2)), 1.2, 1.),
        (lambda x: pml_box(background=x), fd.Material(eps=_t(1.2)), 1.2, 1.),
        (lambda x: dimer(eps=x), _t(2.25), 2.25, 2.25),
        (lambda x: pml_box(film=x), _t(2.3), 2.3, 2.),
        (lambda x: closed_scene(extra=(block(fd.Material(eps=x)),)),
         (_t(2.), _t(2.3), _t(2.6)), (2., 2.3, 2.6), (2., 2., 2.)))
    for make, traced, value, seed in runs:
        ran = _ran(fd.run, make(traced), 'torch', plan=quiet(fd.plan, make(seed)))
        _close(rtt.numpy(ran.final.E.detach()), quiet(fd.run, make(value)).final.E)
    conductor = lambda sigma: closed_scene(extra=(block(fd.Conductor(eps=1., sigma=sigma)),))
    got = _ran(fd.lower, conductor(_t(.3)), quiet(fd.plan, closed_scene()), 'torch')
    assert got['be'].requires_grad
    _close(rtt.numpy(got['be'].detach()), quiet(fd.lower, conductor(.3),
                                                quiet(fd.plan, conductor(.3)))['be'])
    # A Drude block whose eps_inf alone (9) exceeds the plain floor (7.90): at the band's
    # wavelengths it is a metal (Re eps about -3.8 at omega_p 10, 1.8 cells per skin depth),
    # which the plan holds to its skin depth, not to 8 cells per wavelength.
    metal = lambda omega_p: closed_scene(extra=(block(fd.Dispersive(eps_inf=9., poles=(
        fd.Drude(omega_p=omega_p, gamma=.5),))),))
    plan = quiet(fd.plan, metal(10.))
    got = _ran(fd.lower, metal(_t(10.)), plan, 'torch')
    assert float(got['stable']) == 1.


def test_partner_materials_lower_live_beside_a_traced_body():
    """``tests/fdtd/test_fd_traced_partner_grad.py``'s junction scene with the ball's radius, the
    substrate's eps (the face of a concrete junction) and the pillar's eps traced together:
    the lowering equals the concrete scene planned at that point (3.9e-15 in ``be``, 5.5e-15 in
    the cross coefficients), and the derivative of every coefficient they move equals central
    differences of the concrete lowering, each point planned on its own: 3.4e-7, 3.5e-9,
    3.1e-8, 4.8e-10 at steps 1e-4 .. 1e-6. Step 1e-5, the existing 2e-6 budget."""
    import torch
    rtt = runtime('torch')
    p = quiet(fd.plan, junction_scene())
    point = (.22, 2.4, 3.5)
    got = _ran(fd.lower, junction_scene(*_traced(point)), p, 'torch')
    concrete = junction_scene(*point)
    want = quiet(fd.lower, concrete, quiet(fd.plan, concrete))
    _close(rtt.numpy(got['be'].detach()), want['be'])
    for name in want['constitutive']:
        _close(rtt.numpy(got['constitutive'][name].detach()), want['constitutive'][name])
    h = 1e-5
    expected = np.array([(float(quiet(partners_loss, PARTNERS + d, 'numpy', None))
                          - float(quiet(partners_loss, PARTNERS - d, 'numpy', None))) / (2 * h)
                         for d in np.eye(3) * h])
    v = torch.tensor(PARTNERS, dtype=torch.float64, requires_grad=True)
    _ran(partners_loss, v, 'torch', p).backward()
    np.testing.assert_allclose(v.grad.numpy(), expected, rtol=2e-6, atol=0.)


def test_the_resolution_floor_reads_mu_and_every_diffraction_wavelength():
    """The floor is on ``eps * mu`` at the shortest of the band's and the Diffraction
    wavelengths. In the magnetic multi-body scene (dx .1, band's shortest 2.248, floor
    7.90) eps 2.6 and mu 3.2 each pass alone and fail together, where 2.5 and 3.0 pass; in the
    pillar cell with its T plane at .62, below the band's .642 (floor 2.4025 against 2.576), a
    pillar eps of 2.5 fails and 2.3 passes. The concrete scene at each failing value is
    refused by name."""
    rtt = runtime('torch')
    pm = quiet(fd.plan, multi_scene())
    for eps, mu, stable in ((2.5, 3., True), (2.6, 3.2, False)):
        v = list(MULTI)
        v[4], v[8] = _t(eps), _t(mu)
        result = _ran(fd.run, multi_scene(v), 'torch', plan=pm)
        assert bool(result.diagnostics['stable']) is stable
        assert bool(np.isfinite(rtt.numpy(result.final.E.detach())).all()) is stable
    concrete = np.array(MULTI)
    concrete[4], concrete[8] = 2.6, 3.2
    with pytest.raises(ValueError, match='cells per wavelength'):
        quiet(fd.plan, multi_scene(concrete))
    short = (fd.Diffraction('T', z=-.55, side='transmit', wavelengths=(.62,), orders=0),)
    point = POINTS['cylinder'][0]
    p = quiet(fd.plan, scene(point, steps=20).replace(monitors=short))
    for eps, stable in ((2.3, True), (2.5, False)):
        result = _ran(fd.run, scene(_traced(point[:3] + (eps,)), steps=20).replace(
            monitors=short), 'torch', plan=p)
        assert bool(result.diagnostics['stable']) is stable
    with pytest.raises(ValueError, match=r"Diffraction\('T'\) asks for wavelength 0.62"):
        quiet(fd.plan, scene(point[:3] + (2.5,), steps=20).replace(monitors=short))


def test_a_forward_mode_index_is_held_to_the_time_step_too():
    """A Torch forward-mode dual reads as its primal through ``materials.concrete``, so the
    time-step bound was readable, and with the plan certified at the seed nothing compared the
    dual's own value to it. The Box pillar plans no diagonal-rule term, so the coupled guard
    does not stand in for it: eps .2 (past the bound at courant .5) runs to NaN, 2.25 runs."""
    import torch
    import torch.autograd.forward_ad as fwAD
    point = POINTS['box'][0]
    p = quiet(fd.plan, scene(point, kind='box', steps=20))
    for eps, stable in ((2.25, True), (.2, False)):
        with fwAD.dual_level():
            v = [fwAD.make_dual(torch.tensor(x, dtype=torch.float64),
                                torch.tensor(1., dtype=torch.float64))
                 for x in point[:3] + (eps,)]
            result = _ran(fd.run, scene(v, kind='box', steps=20), 'torch', plan=p)
            assert bool(result.diagnostics['stable']) is stable
            E = fwAD.unpack_dual(result.final.E).primal
            assert bool(torch.isfinite(E).all()) is stable


def test_a_plan_keeps_its_material_seeds():
    """``fd.plan`` records every plain Material's values as floats (``-1`` the background,
    ``mu`` None where it was not named), ``with_static`` keeps them, and a traced value at
    plan time records no seed for that material."""
    import torch
    p = quiet(fd.plan, multi_scene())
    seeds = dict((row[0], row[1:]) for row in p._material_seeds)
    assert seeds.get(-1) == ((1., 1., 1.), None)
    for i in range(4):
        assert seeds[i] == ((MULTI[4 + i],) * 3, (MULTI[8 + i],) * 3)
        assert all(type(x) is float for x in seeds[i][0] + seeds[i][1])
    assert getattr(p.with_static(schedule=8), '_material_seeds', None) == p._material_seeds
    box = scene(POINTS['box'][0], kind='box', ranged=False, steps=1)
    traced = box.replace(structures=(box.structures[0], box.structures[1].replace(
        material=fd.Material(eps=torch.tensor(2.25, dtype=torch.float64,
                                              requires_grad=True)))))
    assert 1 not in dict((row[0], row[1:]) for row in quiet(fd.plan, traced)._material_seeds)


def test_a_jitted_run_reuses_one_program_across_indices():
    """``jax.jit`` of a whole ``fd.run`` (40 steps, so the wave crosses the pillar) at two
    indices: each answer is the concrete scene's (6.3e-16 of the peak, measured), the two differ
    (by 5.1e-2), and one program serves both. The planning scene is built from the tracer
    (``_carries`` sees a JAX tracer, which ``float()`` refuses)."""
    runtime('jax')
    import jax
    import jax.numpy as jnp
    point = POINTS['cylinder'][0]
    p = quiet(fd.plan, scene(point, steps=40))

    def final(eps):
        return _ran(fd.run, scene(point[:3] + (eps,), steps=40), 'jax', plan=p).final.E
    compiled = jax.jit(final)
    fields = []
    for eps in (2.25, 1.7):
        fields.append(np.asarray(compiled(jnp.asarray(eps))))
        _close(fields[-1], quiet(fd.run, scene(point[:3] + (eps,), steps=40)).final.E)
    assert np.max(abs(fields[1] - fields[0])) > 1e-2 * np.max(abs(fields[0]))
    assert compiled._cache_size() == 1


def test_a_traced_substrate_runs_and_differentiates_as_the_concrete_scene():
    """The substrate's permittivity traced with the pillar's shape and index: it fills the
    bottom PML, whose grazing rule is held at run time. A 40-step run equals the concrete scene
    at three points (8.9e-17 to 1.8e-16 of the peak), and the five-leaf Jacobian of T and R
    equals central differences: the substrate column within 5.0e-9, 1.5e-10, 1.1e-9, 3.2e-9 at
    steps 1e-4 .. 1e-6, the whole within 9.2e-8 at 3e-6 (``fdtd_traced_material_substrate.json``).
    Step 3e-6, the existing 2e-6 budget."""
    import torch
    rtt = runtime('torch')
    p = quiet(fd.plan, substrate_scene(SUBSTRATE, steps=40))
    for v in (tuple(SUBSTRATE), (.127, .21, 0., 2., 2.3), (.1, .25, -.013, 2.4, 1.8)):
        ran = _ran(fd.run, substrate_scene(_traced(v), steps=40), 'torch', plan=p)
        _close(rtt.numpy(ran.final.E.detach()),
               quiet(fd.run, substrate_scene(v, steps=40)).final.E)
    p = quiet(fd.plan, substrate_scene(SUBSTRATE))
    h = 3e-6

    def values(v):
        return np.array([float(x) for x in quiet(substrate_objectives, v, 'numpy', None)])
    expected = np.array([(values(SUBSTRATE + d) - values(SUBSTRATE - d)) / (2 * h)
                         for d in np.eye(5) * h]).T
    v = torch.tensor(SUBSTRATE, dtype=torch.float64, requires_grad=True)
    out = _ran(substrate_objectives, v, 'torch', p)
    got = np.array([torch.autograd.grad(o, v, retain_graph=True)[0].numpy() for o in out])
    assert np.all(expected != 0)
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=0.)


def test_the_grazing_rule_holds_a_traced_substrate_at_run_time():
    """``grazing.accepted_cos`` is ``grazing._pure_verdict``'s acceptance on a grid of every
    argument it reads; and a traced substrate runs to NaN exactly where the plan of the
    concrete scene refuses a grazing order. In the cell (period .6, 8-cell PMLs) the first
    orders start to propagate in the glass at wavelength .95 for eps (.95/.6)^2 = 2.50694, at
    exit cos 0 -- refused up to the resolution floor 2.576. With the T plane asking for
    wavelength .62 instead, the band of refused indices closes again below the floor: an
    order that starts at eps 1.068 leaves steeply enough from about 1.28 on."""
    from wavebit.fdtd._prepare import grazing
    for R in (1e-16, 1e-12, 1e-8, 1e-6):
        for cells in (8, 24):
            for periods in (None, 100., 160., 400.):
                for g in (.89, GRAZING_G, .93):
                    c = grazing.accepted_cos(R, cells, g, periods)
                    for cos in np.linspace(0., 1., 2001):
                        assert (grazing._pure_verdict(cos, R, cells, g, periods) == 'ok') == (
                            cos >= c), (R, cells, periods, g, cos)

    def judged(make, eps):
        point = tuple(SUBSTRATE[:4]) + (eps,)
        p = quiet(fd.plan, make(SUBSTRATE))
        stable = bool(_ran(fd.run, make(_traced(point)), 'torch', plan=p)
                      .diagnostics['stable'])
        try:
            quiet(fd.plan, make(point))
            planned = True
        except ValueError as error:
            assert 'Grazing order refused' in str(error), error
            planned = False
        return stable, planned
    default = lambda v: substrate_scene(v, steps=20)
    short = lambda v: default(v).replace(monitors=(fd.Diffraction(
        'T', z=-.55, side='transmit', wavelengths=(.62,), orders=0),))
    for make, eps, accepted in ((default, 2.5069, True), (default, 2.5070, False),
                                (default, 2.55, False), (short, 1.05, True),
                                (short, 1.2, False), (short, 1.5, True)):
        assert judged(make, eps) == (accepted, accepted), (eps, accepted)
