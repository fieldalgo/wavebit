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

def _slab(cells, periods=400):
    problem = fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(
            domain=fdtd.Domain(size=(9., .05, 8.)), background=Material(epsilon=2.085136),
            boundaries=fdtd.Boundaries(x=fdtd.Open(), y=fdtd.InvariantAxis(), z=fdtd.Open()),
            structures=(fdtd.Structure('core', shape=fdtd.SnappedSlab(x=(-.75, .75)),
                                       material=Material(epsilon=4.), continuation=('z-', 'z+')),)),
        sources=(fdtd.CurrentDipole('d', center=(0., 0., -2.), component='Ey',
                                    pulse=fdtd.GaussianPulse(wavelength=1.55, frequency_sigma_hz=2e13)),),
        outputs=fdtd.Outputs(monitors=(
            fdtd.ModeMonitor('m', plane=fdtd.Plane(z=2.), wavelengths=(1.55,), modes=1),)))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
                             duration=fdtd.Duration(periods=periods, source='d'),
                             absorber=fdtd.PML(cells=cells))
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
                                    pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=3e13)),),
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

def _refused(structure, boundaries=None):
    _quiet()
    problem, numerics = _rectangle(12, periods=1)
    scene = problem.scene.replace(structures=(structure,))
    if boundaries is not None:
        scene = scene.replace(boundaries=boundaries)
    with pytest.raises(WavebitError) as info:
        fdtd.run(problem.replace(scene=scene), numerics=numerics)
    assert info.value.code == 'fdtd.padding_continuation', (info.value.code, str(info.value))


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


def test_d3_a_continued_core_must_reach_its_face():
    _refused(fdtd.Structure('core', shape=fdtd.SnappedBox(size=(1., .5, 3.), center=(0., 0., .5)),
                            material=Material(epsilon=4.), continuation=('z-', 'z+')))


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
