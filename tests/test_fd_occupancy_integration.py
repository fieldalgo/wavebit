"""C2 integration: actual CSG reports, compiled coefficients, and preparation-time doors.

Mutants are applied as source edits and restored; docs/fd-subpixel-occupancy.md records
their exit codes. The existing mixing, component-location and gradient gates are in
test_fd_occupancy.py; this file covers what the isolated branch could not see.
"""
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._check import report as describe
from wavebit.fd._lower.checks import cfl_bound
from wavebit.fd._runtime import runtime
from wavebit.fd._shapes import Difference, Intersection, Sphere, Union

from test_fd_occupancy import _Ball, _grid_scene, _quiet


# Measured on this complete lowering at eps=3.7, all three CSG operations and four
# execution paths: max relative be discrepancy 6.16e-14. 1e-12 leaves a 16x margin.
COEFFICIENT_RTOL = 1e-12


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'jax.jit', 'torch'))
def test_unvalidated_curved_accuracy_is_reported_once_before_lowering(backend):
    name = 'jax' if backend == 'jax.jit' else backend
    pytest.importorskip(name)
    rt = runtime(name)
    shape = fd.Sphere(radius=.2)
    def scene(eps):
        return _grid_scene((fd.Structure(shape, eps),))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        plan = fd.plan(scene(2.25))
    notes = [str(w.message) for w in caught
             if issubclass(w.category, fd.SubpixelAccuracyWarning)]
    assert len(notes) == 1
    assert '2.023' in notes[0] and 'glass-sphere' in notes[0]
    assert 'not an error bound for this scene' in notes[0]
    assert 'stage-4 Mie gate is incomplete' in notes[0]
    params = fd.lower(scene(2.25), plan, 'numpy')
    assert describe(scene(2.25), plan, params).warnings.count(notes[0]) == 1
    run = lambda eps: fd.lower(scene(eps), plan, name)['be']
    if backend == 'jax.jit':
        import jax
        run = jax.jit(run)
    with warnings.catch_warnings(record=True) as lowered:
        warnings.simplefilter('always')
        for _ in range(2):
            got = rt.numpy(run(rt.array(2.25, 'float64')))
            assert np.isfinite(got).all()
    assert not [w for w in lowered if issubclass(w.category, fd.SubpixelAccuracyWarning)]


def test_cell_snapped_scenes_do_not_inherit_the_curved_accuracy_warning():
    scene = _grid_scene((fd.Structure(fd.Region(size=(.3,)*3), 2.25),))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fd.plan(scene)
    assert not [w for w in caught if issubclass(w.category, fd.SubpixelAccuracyWarning)]


class _Composed(_Ball):
    """An external Shape using the public protocol, with a static enclosing support."""

    def __init__(self, operation):
        # The union extends to 0.27; 0.35 encloses it plus the grid's voxel margin.
        super().__init__(0.35, bound=0.35)
        self.node = operation(Sphere(0.2, center=(-0.07, 0, 0)),
                              Sphere(0.2, center=(0.07, 0, 0)))

    def body(self):
        return self.node


@pytest.mark.parametrize('operation', (Union, Intersection, Difference))
@pytest.mark.parametrize('backend', ('numpy', 'jax', 'jax.jit', 'torch'))
def test_csg_bounds_are_planned_before_compiled_lowering(operation, backend):
    name = 'jax' if backend == 'jax.jit' else backend
    pytest.importorskip(name)
    rt = runtime(name)
    shape = _Composed(operation)

    def scene(eps):
        return _grid_scene((fd.Structure(shape, fd.Material(eps=eps)),))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        plan = fd.plan(scene(3.7))
    notes = [str(w.message) for w in caught if issubclass(w.category, fd.BoundedVoxelWarning)]
    # C11 also reads the common E nodes: measured 348 total bounded voxels,
    # widest 0.06124, versus 284 / 0.05377 on the three E components alone.
    assert len(notes) == 1 and '348 voxel(s)' in notes[0]
    assert '0.06124 of a voxel' in notes[0]
    assert f'{operation.__name__}(Sphere, Sphere)' in notes[0]
    assert 'cannot remove material a Union or a Difference has' in notes[0]
    assert 'or add material an Intersection has not' in notes[0]

    accuracy = [str(w.message) for w in caught
                if issubclass(w.category, fd.SubpixelAccuracyWarning)]
    # Measured common-point squared-normal discrepancies for this fixture:
    # Union .628253, Intersection/Difference .821168 (six significant digits).
    gap = '0.628253' if operation is Union else '0.821168'
    assert len(accuracy) == 1
    assert f'{operation.__name__}: {gap}' in accuracy[0]
    assert 'not bounds on electromagnetic error' in accuracy[0]

    reference = fd.lower(scene(3.7), plan, 'numpy')
    assert describe(scene(3.7), plan, reference).warnings.count(notes[0]) == 1
    assert describe(scene(3.7), plan, reference).warnings.count(accuracy[0]) == 1
    run = lambda eps: fd.lower(scene(eps), plan, name)['be']
    if backend == 'jax.jit':
        import jax
        run = jax.jit(run)
    # Both the first trace and a cached call see the already-issued plan note; no Python
    # warning or concrete-value inspection may hide in the numerical lowering.
    with warnings.catch_warnings(record=True) as lower_notes:
        warnings.simplefilter('always')
        for _ in range(2):
            got = np.asarray(rt.numpy(run(rt.array(3.7, 'float64'))))
            assert got.dtype == np.float64 and np.isfinite(got).all()
            np.testing.assert_allclose(got, reference['be'], rtol=COEFFICIENT_RTOL, atol=0)
    assert not [w for w in lower_notes if issubclass(w.category, fd.BoundedVoxelWarning)]


@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_plan_names_geometry_that_cannot_be_checked_concretely(backend):
    pytest.importorskip(backend)

    def make_plan(radius):
        scene = _grid_scene((fd.Structure(_Ball(radius, bound=0.3), 2.25),))
        return _quiet(fd.plan, scene).dt

    with pytest.raises(ValueError, match='fd.plan requires concrete geometry'):
        if backend == 'jax':
            import jax
            import jax.numpy as jnp
            jax.config.update('jax_enable_x64', True)
            jax.jit(make_plan)(jnp.asarray(0.2, dtype=jnp.float64))
        else:
            import torch
            make_plan(torch.tensor(0.2, dtype=torch.float64, requires_grad=True))


def test_smoothed_magnetic_phases_use_independent_stability_extrema():
    """Each phase has n=1, but E and H can sample different fractions of those phases.

    Measured before the door: a radius-0.25 ball at dx=0.05 reported bound/dt=2 at
    courant=0.5, while min sqrt(eps*mu) over component arrays was 0.562900 and the
    independent-extrema bound was 0.5. This gate checks the conservative mathematical
    bound, not a claim that 0.5 is the exact instability threshold of that finite sphere.
    """
    scene = _grid_scene((fd.Structure(fd.Sphere(radius=0.25), fd.Material(eps=4, mu=0.25)),))
    with pytest.raises(ValueError, match=r'sqrt\(min\(eps\) \* min\(mu\)\)') as caught:
        _quiet(fd.plan, scene.replace(run=fd.Run(steps=4, courant=0.5)))
    assert 'Run(courant=' in str(caught.value)
    safe = scene.replace(run=fd.Run(steps=4, courant=0.4))
    plan = _quiet(fd.plan, safe)
    bound, reason = cfl_bound(safe, plan)
    assert reason is None and bound == 0.5 * plan.dx / np.sqrt(3)
    assert plan.dt == safe.run.courant * plan.dx / np.sqrt(3), 'planning must not change dt'


def test_a_snapped_magnetic_pair_is_bounded_by_every_phase_pairing():
    """E and H sit at different points in any scene, so the bound pairs min eps with min mu.

    Found in stage 4's C5 audit: ``Material(eps=4, mu=0.25)`` as an ``fd.Region`` in a
    background of ``Material(eps=0.25, mu=4)`` -- every phase at ``n = 1`` -- planned at any
    courant below 1 while the bound read each phase's own index outside smoothed scenes, and
    ran to NaN within 3000 steps from courant 0.60 (0.50 and 0.55 stayed bounded over 20000).
    The bound is ``sqrt(min eps * min mu)`` in every scene now, snapped or smoothed: 0.25 of
    the vacuum courant here, conservative, and ``dt`` is never recomputed. A scene of
    ``eps, mu >= 1`` is untouched, to the bit.

    Mutant, measured 2026-09-26: the pairing applied only to smoothed scenes (as before) fails
    this case.
    """
    def scene(structures, courant, background=None):
        return _grid_scene(structures).replace(run=fd.Run(steps=4, courant=courant),
                                               background=background)
    pair = (fd.Structure(fd.Region(size=(0.3, 0.3, 0.3)), fd.Material(eps=4.0, mu=0.25)),)
    reciprocal = fd.Material(eps=0.25, mu=4.0)
    with pytest.raises(ValueError, match=r'sqrt\(min\(eps\) \* min\(mu\)\)') as caught:
        _quiet(fd.plan, scene(pair, 0.3, reciprocal))
    assert 'Run(courant=' in str(caught.value) and 'rigorous' in str(caught.value)
    safe = scene(pair, 0.2, reciprocal)
    plan = _quiet(fd.plan, safe)
    assert cfl_bound(safe, plan)[0] == 0.25 * plan.dx / np.sqrt(3)
    assert plan.dt == 0.2 * plan.dx / np.sqrt(3), 'planning must not change dt'
    ordinary = scene((fd.Structure(fd.Region(size=(0.3, 0.3, 0.3)), 2.25),
                      fd.Structure(fd.Region(size=(0.2, 0.2, 0.2), center=(0.2, 0.2, 0.2)),
                                   fd.Material(eps=1.0, mu=2.0))), 0.5)
    plan = _quiet(fd.plan, ordinary)
    assert cfl_bound(ordinary, plan)[0] == plan.dx / np.sqrt(3)


def test_the_traced_guard_bounds_every_pairing_and_a_nonpositive_phase():
    """The traced twin of the bound: ``stable`` is NaN wherever the concrete path would refuse.

    With a traced material the bound cannot be read at plan time, so ``fd.lower`` computes it
    on the backend (``params._traced_guard``) and a violated one poisons ``stable``. It took
    each phase's own index, and the reciprocal pair traced at courant 0.9 read stable and ran
    to NaN after 2000 steps; and a traced ``eps = -2, mu = -1`` phase multiplied back to a
    positive index and read stable while the field reached 1.5e290 in 20 steps. Now the guard
    pairs ``min eps`` with ``min mu`` and reads a non-positive phase as unstable.

    The same audit found the guard's pole loop reusing the name ``eps_min`` for a dispersive
    phase's own permittivity, so every phase listed before the last dispersive one dropped out
    of the pairing: the reciprocal pair with a weak Lorentz pole on the traced phase read
    stable at courant 0.9.

    Mutants, measured 2026-09-26: the guard taking each phase's own index, the guard without
    its positivity condition, and the pole loop writing into ``eps_min`` again, each fail this
    case.
    """
    torch = pytest.importorskip('torch')

    def traced(value):
        return torch.tensor(value, dtype=torch.float64, requires_grad=True)

    def stable(material, courant, background=None):
        scene = _grid_scene((fd.Structure(fd.Region(size=(0.3, 0.3, 0.3)), material),)).replace(
            run=fd.Run(steps=4, courant=courant), background=background)
        plan = _quiet(fd.plan, scene)
        return float(_quiet(fd.lower, scene, plan, 'torch')['stable'])
    reciprocal = fd.Material(eps=0.25, mu=4.0)
    pair = fd.Material(eps=traced(4.0), mu=traced(0.25))
    assert stable(pair, 0.2, reciprocal) == 1.0
    assert np.isnan(stable(pair, 0.9, reciprocal))
    assert stable(fd.Material(eps=traced(2.25), mu=traced(1.0)), 0.5) == 1.0
    assert np.isnan(stable(fd.Material(eps=traced(-2.0), mu=traced(-1.0)), 0.1))
    weak = fd.Lorentz(omega_0=1.0, strength=1e-6)
    dispersive = fd.Dispersive(eps_inf=traced(4.0), mu=0.25, poles=(weak,))
    assert stable(dispersive, 0.2, reciprocal) == 1.0
    assert np.isnan(stable(dispersive, 0.9, reciprocal))


@pytest.mark.parametrize('value', (float('inf'), float('-inf'), float('nan')))
@pytest.mark.parametrize('parameter', ('radius', 'center'))
def test_sphere_rejects_nonfinite_geometry_by_name(parameter, value):
    kwargs = {'radius': value} if parameter == 'radius' else {
        'radius': 0.2, 'center': (0., value, 0.)}
    with pytest.raises(ValueError, match=rf'Sphere\({parameter}=.*finite'):
        fd.Sphere(**kwargs)


def test_sphere_answers_the_current_extent_protocol():
    shape = fd.Sphere(radius=0.2)
    plan = _quiet(fd.plan, _grid_scene((fd.Structure(shape, 2.25),)))
    assert shape.min_extent(plan.grid) == 0.4
