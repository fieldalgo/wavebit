"""A cell-snapped waveguide core continued through an absorbing layer.

Criteria D1-D3 of ``docs/validation/fdtd/fd-waveguide-through-pml.md``, registered before the
door was first run; every limit here is quoted from there.
"""
import warnings

import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.errors import WavebitError


def _quiet():
    warnings.simplefilter('ignore')


def _run(problem, numerics):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fdtd.run(problem, numerics=numerics)
    return result, [type(w.message).__name__ for w in caught]


def _reflection(result, name='m'):
    m = result[name]
    return float(np.asarray(m.backward_energy)[0, 0] / np.asarray(m.forward_energy)[0, 0])


# ---------------------------------------------------------------------------------------------
# D1: slab waveguide
# ---------------------------------------------------------------------------------------------

def _slab(cells, periods=400, initial=True):
    """D1, launched by an initial field (a CurrentDipole is refused on an InvariantAxis):
    the TE0 mode at 1.55 um times a Gaussian-windowed carrier along z, centred at z = -2."""
    problem = fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(
            domain=fdtd.Domain(size=(9., .05, 8.)), background=Material(epsilon=2.085136),
            boundaries=fdtd.Boundaries(x=fdtd.Open(), y=fdtd.InvariantAxis(), z=fdtd.Open()),
            structures=(fdtd.Structure('core', shape=fdtd.SnappedSlab(x=(-.75, .75)),
                                       material=Material(epsilon=4.), continuation=('z-', 'z+')),)),
        outputs=fdtd.Outputs(monitors=(
            fdtd.ModeMonitor('m', plane=fdtd.Plane(z=2.), wavelengths=(1.55,), modes=1),)))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
                             duration=fdtd.Duration(seconds=periods * 1.55e-6 / 299792458.0),
                             absorber=fdtd.PML(cells=cells, wavelength=1.55))
    if initial:
        modes = fdtd.solve_modes(problem, numerics=numerics, plane=fdtd.Plane(z=2.),
                                 wavelength=1.55, count=1)
        beta = modes[0].beta_per_meter * 1e-6
        profile = np.asarray(modes[0].E['Ey'])[0].real            # (nx,) on the window's Ey nodes
        xs = np.asarray(modes.coordinates['Ey']['x'])

        def ey(x, y, z):
            shape = np.interp(x, xs, profile, left=0.0, right=0.0)
            return shape * np.exp(-(z + 2.) ** 2 / (2 * .6 ** 2)) * np.cos(beta * (z + 2.))
        problem = problem.replace(initial=fdtd.InitialField(E=(None, ey, None)))
    return problem, numerics


@pytest.mark.slow
def test_d1_slab_core_through_the_pml_reflects_little_and_stays_stable():
    _quiet()
    r12, caught = _run(*_slab(12))
    assert 'GrowingFieldWarning' not in caught, caught
    r24, _ = _run(*_slab(24))
    reflection12, reflection24 = _reflection(r12), _reflection(r24)
    assert reflection12 <= 1e-5, reflection12
    assert reflection24 < reflection12, (reflection24, reflection12)


# ---------------------------------------------------------------------------------------------
# D2: rectangular core
# ---------------------------------------------------------------------------------------------

def _rectangle(cells, periods=60):
    problem = fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(
            domain=fdtd.Domain(size=(3., 2.5, 5.)), boundaries=fdtd.Boundaries.open(),
            structures=(fdtd.Structure('core', shape=fdtd.SnappedBox(size=(1., .5, 10.)),
                                       material=Material(epsilon=4.), continuation=('z-', 'z+')),)),
        sources=(fdtd.CurrentDipole('d', center=(0., 0., -1.), component='Ex',
                                    pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=1.5e13)),),
        outputs=fdtd.Outputs(monitors=(
            fdtd.ModeMonitor('m', plane=fdtd.Plane(z=1.), wavelengths=(1.,), modes=1),)))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
                             duration=fdtd.Duration(periods=periods, source='d'),
                             absorber=fdtd.PML(cells=cells))
    return problem, numerics


@pytest.mark.slow
def test_d2_rectangular_core_through_the_pml_reflects_little():
    _quiet()
    r12, _ = _run(*_rectangle(12))
    r24, _ = _run(*_rectangle(24))
    reflection12, reflection24 = _reflection(r12), _reflection(r24)
    assert reflection12 <= 1e-5, reflection12
    assert reflection24 < reflection12, (reflection24, reflection12)


# ---------------------------------------------------------------------------------------------
# D3: refusals
# ---------------------------------------------------------------------------------------------

def _refused(structure, boundaries=None, match=None):
    _quiet()
    problem, numerics = _rectangle(12, periods=1)
    scene = problem.scene.replace(structures=(structure,))
    if boundaries is not None:
        scene = scene.replace(boundaries=boundaries)
    with pytest.raises(WavebitError) as info:
        fdtd.run(problem.replace(scene=scene), numerics=numerics)
    assert info.value.code == 'fdtd.padding_continuation', (info.value.code, str(info.value))
    assert match is None or match in str(info.value), str(info.value)


def test_d3_a_smoothed_core_cannot_continue():
    _refused(fdtd.Structure('core', shape=fdtd.Box(size=(1., .5, 10.)), material=Material(epsilon=4.),
                            continuation=('z-', 'z+')))


def test_d3_a_continued_core_touching_another_open_face_is_refused():
    _refused(fdtd.Structure('core', shape=fdtd.SnappedBox(size=(1., 4., 10.)), material=Material(epsilon=4.),
                            continuation=('z-', 'z+')))


def test_d3_continuation_through_a_face_without_padding_is_refused():
    boundaries = fdtd.Boundaries(x=fdtd.Open(), y=fdtd.Open(), z=(fdtd.PEC(), fdtd.Open()))
    _refused(fdtd.Structure('core', shape=fdtd.SnappedBox(size=(1., .5, 10.)), material=Material(epsilon=4.),
                            continuation=('z-', 'z+')), boundaries)


@pytest.mark.parametrize('center', [.5, -.75, .75])
def test_d3_a_continued_core_must_reach_its_face(center):
    # Reaching neither face, then only z- (it ends at z = 2.25 of 2.5), then only z+.
    size = 3. if center == .5 else 4.5
    _refused(fdtd.Structure('core', shape=fdtd.SnappedBox(size=(1., .5, size), center=(0., 0., center)),
                            material=Material(epsilon=4.), continuation=('z-', 'z+')))


def test_d3_a_continued_core_runs_along_one_axis():
    _refused(fdtd.Structure('core', shape=fdtd.SnappedBox(size=(1., .5, 10.)), material=Material(epsilon=4.),
                            continuation=('z-', 'x+')), match='one axis only')


def test_d3_a_continued_core_runs_through_the_padding():
    # The lowered core keeps its cross-section and spans the allocated domain along z.
    from wavebit.fdtd._lower_public import lower_problem
    _quiet()
    scene, _ = lower_problem(*_rectangle(12, periods=1))
    core = scene.structures[0].shape
    assert core.size[:2] == (1., .5) and core.center[2] == 0., core
    assert core.size[2] > scene.domain.size[2], (core, scene.domain)


def test_d3_the_registered_scenes_are_admitted():
    # The scenes of D1 and D2 pass check and prepare (the engine's absorbing-layer rule
    # included), so a refusal above is the rule it names and not a broken scene.
    _quiet()
    for problem, numerics in (_slab(12, periods=1, initial=False), _rectangle(12, periods=1)):
        assert fdtd.check(problem, numerics=numerics).ok
        fdtd.prepare(problem, numerics=numerics)


def test_d3_a_slab_does_not_continue_through_its_own_axis():
    boundaries = fdtd.Boundaries(x=fdtd.Open(), y=fdtd.InvariantAxis(), z=fdtd.Open())
    problem, numerics = _slab(12, periods=1)
    scene = problem.scene.replace(structures=(fdtd.Structure(
        'core', shape=fdtd.SnappedSlab(x=(-.75, .75)), material=Material(epsilon=4.),
        continuation=('x-', 'z+')),), boundaries=boundaries)
    _quiet()
    with pytest.raises(WavebitError) as info:
        fdtd.run(problem.replace(scene=scene), numerics=numerics)
    assert info.value.code == 'fdtd.padding_continuation'
    assert 'its own axis' in str(info.value), str(info.value)
