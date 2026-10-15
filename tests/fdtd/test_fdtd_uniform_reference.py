"""The scene solver's uniform reference, stepped on a column, is the full-grid reference.

``src/wavebit/fdtd/_periodic/scene.py`` calibrates the incident wave against a run of the same grid filled with
the top half-space material. That medium is uniform, so the run produces the lattice plane wave
and every transverse cell is the first one times ``exp(i(kx0 x + ky0 y))``. The solver therefore
steps it on a two-cell transverse column, keeping ``dz``, ``dy`` and ``dx`` so the discrete
transverse operator is unchanged, and continues the plane from it by that phase
(:func:`_uniform_reference`). Half of the route's four time loops were that run.

The saving is only sound while the continuation really is the full-grid answer, and nothing else
would notice if it stopped being: the reference is subtracted from the actual run, so an error in
it lands in every amplitude as a small bias rather than as a failure. This file is the check.

It runs one scene through the public door, captures what the solver actually handed the kernel,
rebuilds the full-grid reference from those same settings, and compares. The scene is oblique on
purpose -- at normal incidence the continuation is exactly 1 and the comparison could not fail --
and the last assertion pins that resolving power by requiring the transverse phase to vary.
"""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd
import numpy as np
import pytest

from wavebit.fdtd._periodic import scene as scene_fdtd

GRATING = np.where(np.arange(8) < 4, 2.25, 1.)[None, None, :]
SCENE = dict(structure=GRATING, thickness=[.3], top=1., bot=1.44, period=[.8], wavelength=.9,
             tetm='te', theta=np.deg2rad(25.), backend='numpy', resolution=10., n_periods=30)

#: Measured 2026-09-16 over the monitor spectra of the scenes in the battery: 0 at normal
#: incidence, on a graded mesh and from below; 3.9e-15 oblique TE/TM; 6.4e-15 conical crossed.
#: The bound here is three orders above the worst of those.
TOLERANCE = 1e-12


def _runs(monkeypatch):
    """What the solver did: the reference it built (its own continued arrays, not a rebuild of
    them) and the full-size call it made for the structure."""
    original_grid, original_reference = scene_fdtd.fdtd_grid, scene_fdtd._uniform_reference
    actual, reference = [], []

    def grid(epsilon, **keywords):
        result = original_grid(epsilon, **keywords)
        values = np.asarray(epsilon)
        if not np.all(values == values.flat[0]):
            actual.append((values, keywords))
        return result

    def uniform_reference(epsilon, *args, **keywords):
        result = original_reference(epsilon, *args, **keywords)
        reference.append((np.asarray(epsilon), result))
        return result

    monkeypatch.setattr(scene_fdtd, 'fdtd_grid', grid)
    monkeypatch.setattr(scene_fdtd, '_uniform_reference', uniform_reference)
    _legacy_wavebit_fdtd(**SCENE)
    monkeypatch.undo()
    assert reference and actual, 'the solver made no reference run'
    return reference[0], actual[0]


def test_the_column_reference_is_the_full_grid_reference(monkeypatch):
    (reference_epsilon, reference_result), (actual_epsilon, actual_keywords) = _runs(monkeypatch)
    ny, nx = actual_epsilon.shape[-2:]
    assert (ny, nx) != (2, 2), 'this scene must be wider than the column, or nothing is tested'

    full = scene_fdtd.fdtd_grid(
        np.ones_like(actual_epsilon)*reference_epsilon.flat[0], mu=1.,
        pml_background=((reference_epsilon.flat[0], 1.), (reference_epsilon.flat[0], 1.)),
        **{k: v for k, v in actual_keywords.items()
           if k not in ('mu', 'pml_background', 'poles', 'record_field', 'record_components',
                        'record_region')})

    for key in ('electric_spectrum', 'magnetic_spectrum'):
        if key not in full or key not in reference_result:
            continue
        expected = np.asarray(full[key])
        # What the solver itself handed the extraction, not a rebuild of it.
        continued = np.asarray(reference_result[key])
        assert continued.shape == expected.shape, key
        scale = np.max(np.abs(expected))
        assert scale > 0, key
        assert np.max(np.abs(continued - expected))/scale < TOLERANCE, (
            f'{key}: the reference the solver used is not the full-grid reference')

    # The test's own resolving power: on this scene the transverse phase really does vary, so the
    # comparison above is of a continuation and not of a constant repeated across the plane.
    plane = np.asarray(full['electric_spectrum'])[0, 0]
    assert np.max(np.abs(plane - plane[..., :1])) > .1*np.max(np.abs(plane))


def test_a_grid_already_two_cells_wide_is_stepped_as_it_is(monkeypatch):
    """The column is a saving, not a rule: a scene that is already two cells across keeps its run."""
    original = scene_fdtd.fdtd_grid
    widths = []

    def wrapper(epsilon, **keywords):
        widths.append(np.asarray(epsilon).shape[-2:])
        return original(epsilon, **keywords)

    monkeypatch.setattr(scene_fdtd, 'fdtd_grid', wrapper)
    _legacy_wavebit_fdtd(**dict(SCENE, structure=np.full((1, 1, 1), 2.25), period=[.12], theta=0.))
    monkeypatch.undo()
    assert widths and set(widths) == {(2, 2)}, widths


@pytest.mark.parametrize('theta_degrees', [0., 25.])
def test_the_answer_survives_the_column(theta_degrees):
    """An independent check on the physics the reference calibrates: a vacuum scene reflects nothing
    and transmits the free-space phase, whatever the incidence."""
    vacuum = _legacy_wavebit_fdtd(structure=np.ones((1, 1, 1)), thickness=[.3], top=1., bot=1.,
                          period=[.5], wavelength=.9, tetm='te',
                          theta=np.deg2rad(theta_degrees), backend='numpy', resolution=16., n_periods=20)
    assert float(np.sum(vacuum.R)) < 1e-6
    assert abs(float(np.sum(vacuum.T)) - 1.) < 2e-3
