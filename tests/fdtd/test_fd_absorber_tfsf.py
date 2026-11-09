"""A TFSF box behind ConductiveAbsorber faces: the source re-injected the absorber's echo.

A TFSF box's incident field comes from a 1-D column along its axis (and, over a layered
background, from a uniform driver column feeding it), hard-driven one cell upstream of the
entry plane. The columns used to end on the scene's own faces, so behind a
``ConductiveAbsorber`` -- PEC-backed, round-trip amplitude ~exp(-3) = 5e-2 in vacuum -- the
column's echo came back to the hard drive (``r = -1``) and went down again as a second
incident wave, and the box injected it. An empty box does not show it: the box cancels
whatever its column holds, echo included, so the scattered field outside it is exactly zero
either way. The incident spectrum and every scatterer do. The columns now end in a PML of
their own there (``_prepare.sources.column_pads``), a layer reaching the face continues
through it, and what is left is the main grid's own absorber return.

Measured against the same scene under ``PML(cells=12)`` (vacuum, 12-cell faces):

==============================================  ==================  ==================
quantity                                        before              after
==============================================  ==================  ==================
3-D box, incident spectrum ``|I|``              -2.2 % .. +2.0 %    <= 2e-5
3-D box, glass cube's scattered flux            -4.2 % .. +11.4 %   -3.4 % .. +2.2 %
layered column (glass through ``z-``), ``|t|``  -3.6 % .. +5.1 %    <= 4.2e-4
layered column, reflected ``|r|``               -16.6 % .. +14.2 %  <= 2.3e-3
==============================================  ==================  ==================

The cube's remaining few percent is the main grid's absorber returning the scattered field,
0.15 um outside the flux box here; in a 1.6 um domain with 0.3 um of clearance the same
cube read -8.9 % .. +16.7 % before and -0.85 % .. +0.83 % after.

Independent references: the PML run of the same scene, and for the layered column the
closed-form normal-incidence Fresnel amplitudes of the vacuum/glass interface
(``t = 2 / (1 + n)``, ``r = (n - 1) / (n + 1)`` in magnitude), which the PML run meets to
the grid's dispersion.
"""
import math
import warnings

import numpy as np
import pytest

from wavebit import ConductiveMaterial, Material, Units, fdtd
from wavebit.units import C0, EPSILON0

# ---------------------------------------------------------------------------------------------
# A 3-D box, empty and around a glass cube
# ---------------------------------------------------------------------------------------------

WAVELENGTHS = tuple(float(w) for w in np.round(np.linspace(.7, .9, 9), 4))
STEPS = 2400


def _box(cube):
    structures = ()
    if cube:
        structures = (fdtd.Structure('cube', shape=fdtd.SnappedBox(size=(.24, .24, .24)),
                                     material=Material(epsilon=2.25)),)
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(1.2, 1.2, 1.2)), structures=structures,
        background=Material(epsilon=1.), boundaries=fdtd.Boundaries.open()),
        sources=(fdtd.TFSF('t', box=fdtd.SnappedBox(size=(.6, .6, .6)), direction='-z',
                           polarization='x',
                           pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=4e13)),),
        outputs=fdtd.Outputs(monitors=(
            fdtd.FluxMonitor('out', box=fdtd.SnappedBox(size=(.9, .9, .9)),
                             wavelengths=WAVELENGTHS),)))


def _numerics(absorber, steps=STEPS, spacing=.03):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=spacing),
                         duration=fdtd.Duration(steps=steps), absorber=absorber)


ABSORBERS = (('pml', fdtd.PML(cells=12)), ('graded', fdtd.ConductiveAbsorber(cells=12)))


def _spectrum(times, field, wavelengths):
    frequencies = C0 / (np.asarray(wavelengths) * 1e-6)
    return np.array([np.trapezoid(field * np.exp(-2j * math.pi * f * times), times)
                     for f in frequencies])


def _quiet(call, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return call(*args, **kwargs)


@pytest.fixture(scope='module')
def boxes():
    out = {}
    # The incident column does not see the cube, so the empty box runs only where it leaks.
    for cube, absorbers in ((False, ABSORBERS[1:]), (True, ABSORBERS)):
        for label, absorber in absorbers:
            result = _quiet(fdtd.run, _box(cube), numerics=_numerics(absorber))
            reference = result.source_references['t']
            incident = np.asarray(reference['incident_E'])[:, 0]          # Ex, the polarization
            out[cube, label] = dict(
                flux=np.asarray(result['out'].spectral_energy),
                incident=_spectrum(np.asarray(reference['times_seconds']), incident,
                                   WAVELENGTHS))
    return out


def test_an_empty_box_leaks_nothing_behind_the_absorber(boxes):
    # The box is consistent with whatever its column holds, so this never saw the echo.
    assert np.all(boxes[False, 'graded']['flux'] == 0.0)
    np.testing.assert_array_equal(boxes[False, 'graded']['incident'],
                                  boxes[True, 'graded']['incident'])


def test_the_incident_spectrum_is_the_pml_scenes(boxes):
    # The column no longer echoes (it was off by -2.2 % .. +2.0 %).
    graded, pml = boxes[True, 'graded']['incident'], boxes[True, 'pml']['incident']
    np.testing.assert_allclose(np.abs(graded), np.abs(pml), rtol=1e-3)


def test_a_scatterers_flux_is_off_by_the_absorbers_own_return_only(boxes):
    # What is left is the main grid's return of the scattered field, -3.4 % .. +2.2 % (it was
    # -4.2 % .. +11.4 % with the column's echo illuminating the cube a second time).
    ratio = boxes[True, 'graded']['flux'] / boxes[True, 'pml']['flux']
    assert np.max(np.abs(ratio - 1)) < 4e-2, ratio


# ---------------------------------------------------------------------------------------------
# A layered column: a glass substrate running out through the absorber
# ---------------------------------------------------------------------------------------------

LAYER_WAVELENGTHS = (.9, .95, 1., 1.05, 1.1)
INDEX = 1.5


def _layered(substrate=True, material=None):
    pulse = fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=.05 * C0 / 1e-6)
    structures = ()
    if substrate:
        structures = (fdtd.Structure('substrate', shape=fdtd.SnappedSlab(z=(-2., -.3)),
                                     material=material or Material(epsilon=INDEX ** 2)),)
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.3, .3, 2.)), structures=structures,
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Open())),
        sources=(fdtd.TFSF('s', box=fdtd.SnappedBox(size=(.1, .1, 1.2)), pulse=pulse),),
        outputs=fdtd.Outputs(monitors=tuple(
            fdtd.FieldMonitor(name, plane=fdtd.Plane(z=z), wavelengths=LAYER_WAVELENGTHS,
                              components=('Ex',)) for name, z in (('t', -.5), ('r', .5)))))


@pytest.fixture(scope='module')
def layers():
    def run(problem, absorber):
        result = _quiet(fdtd.run, problem, numerics=_numerics(absorber, 4000, .05))
        return {m: np.asarray(result[m].values['Ex']).reshape(len(LAYER_WAVELENGTHS), -1)
                for m in ('t', 'r')}
    vacuum = run(_layered(False), fdtd.PML(cells=12))
    cell = int(np.argmax(np.abs(vacuum['t'][2])))          # inside the 2 x 2-cell box
    incident = {m: vacuum[m][:, cell] for m in ('t', 'r')}
    out = {}
    for label, absorber in ABSORBERS:
        fields = run(_layered(), absorber)
        out[label] = dict(t=fields['t'][:, cell] / incident['t'],
                          r=(fields['r'][:, cell] - incident['r']) / incident['r'])
    return out


def test_the_pml_layered_column_is_the_fresnel_interface(layers):
    np.testing.assert_allclose(np.abs(layers['pml']['t']), 2 / (1 + INDEX), atol=1e-2)
    np.testing.assert_allclose(np.abs(layers['pml']['r']), (INDEX - 1) / (INDEX + 1), atol=1e-2)


def test_a_substrate_through_the_absorber_is_the_pml_scenes(layers):
    # The column's substrate runs on through its own PML, matched to glass; the driver ends
    # in a vacuum one. Before, |t| moved by -3.6 % .. +5.1 % and |r| by -16.6 % .. +14.2 %.
    graded, pml = layers['graded'], layers['pml']
    np.testing.assert_allclose(np.abs(graded['t']), np.abs(pml['t']), rtol=2e-3)
    np.testing.assert_allclose(np.abs(graded['r']), np.abs(pml['r']), rtol=1e-2)


# ---------------------------------------------------------------------------------------------
# The column's ends, the warning and the refusal
# ---------------------------------------------------------------------------------------------

def test_the_column_is_padded_behind_the_absorber_only():
    from wavebit.fdtd._lower_public import lower_problem
    from wavebit.fdtd._prepare.sources import column_pads
    for label, absorber in ABSORBERS:
        scene, _ = lower_problem(_layered(), _numerics(absorber, 2, .05))
        expected = (12, 12) if label == 'graded' else (0, 0)
        assert column_pads(scene, scene.sources[0]) == expected, label


def test_check_warns_of_the_absorbers_return_with_a_tfsf():
    report = fdtd.check(_box(True), numerics=_numerics(fdtd.ConductiveAbsorber(cells=12), 2))
    assert report.ok
    (issue,) = [i for i in report.issues if i.code == 'numerics.absorber_reflection']
    assert issue.severity == 'warning' and issue.path == 'numerics.absorber'
    assert issue.actual == ('z-', 'z+') and 'TFSF' in issue.message and 'exp(-3)' in issue.message
    report = fdtd.check(_box(True), numerics=_numerics(fdtd.PML(cells=12), 2))
    assert not [i for i in report.issues if i.code == 'numerics.absorber_reflection']


@pytest.mark.parametrize('where', ['substrate', 'background'])
def test_a_lossy_medium_at_a_padded_face_is_refused(where):
    # The column's PML cannot be matched to a conductor; under PML the scene's own face
    # refuses it in its own words, and under the absorber the column end does.
    lossy = ConductiveMaterial(epsilon=2.25, conductivity=.05 * EPSILON0 * C0 / 1e-6)
    problem = _layered(material=lossy)
    if where == 'background':
        problem = problem.replace(scene=problem.scene.replace(structures=(), background=lossy))
    report = _quiet(fdtd.check, problem, numerics=_numerics(fdtd.PML(cells=12), 2, .05))
    assert [i.code for i in report.issues if i.severity == 'error'] == ['fdtd.invalid']
    report = _quiet(fdtd.check, problem,
                    numerics=_numerics(fdtd.ConductiveAbsorber(cells=12), 2, .05))
    (issue,) = [i for i in report.issues if i.severity == 'error']
    assert issue.code == 'fdtd.unsupported'
    assert "TFSF('s')" in issue.message and 'second incident wave' in issue.message
    assert 'ConductiveMaterial' in issue.message
    assert ('structures[0]' if where == 'substrate' else 'the background') in issue.message
