"""Public junction solves against the pre-existing study and the exact film."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import json

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from tests.fdtd.test_fd_junction import quiet, scene_of


def study():
    from studies.fdtd.subpixel_junction import fdtd_subpixel_junction
    return fdtd_subpixel_junction


def measured_rows():
    path = repository_path('tests/references/fd_junction_study_rows.json')
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.mark.parametrize('index', range(8))
def test_public_solver_reproduces_the_measured_study(index):
    row = measured_rows()[index]
    h = study()
    scene, _ = h.make(row['spec'])
    if row['spec']['arm'] == '1a':
        sub, pillar = scene.structures
        scene = scene.replace(structures=(sub, pillar.replace(junction='staircase')))
    result = quiet(fd.run, scene, backend='numpy')
    got = h.amplitudes(result, row['spec']['pol'])
    assert row['prov'] == 'cc2ccf8761b414ba'
    # The study's cross-backend gate is 1e-10; these rows include the complete
    # phase as well as amplitude and use its pre-existing, recorded tolerance.
    for key in ('r', 't'):
        np.testing.assert_allclose(np.asarray(got[key]), np.asarray(row[key]), rtol=0, atol=1e-10)


@pytest.mark.parametrize('theta,pol,pins', (
    (0., 's', (.10269850862654427, .02425574264474361, .005976759327073628)),
    # Actual anchor|1790518977, cases[2].e2: the draft copied obsolete
    # oblique pins; its values differ by up to 3.9441e-7 after PML amendment 1.
    (45., 'p', (.08316041684012954, .02005072445163558, .004970824005669929)),
))
def test_flat_film_retains_the_measured_second_order(theta, pol, pins):
    h = study()
    reference = [h.rcwa_point('A', 'film', 0, wl, theta=theta)[pol] for wl in h.WAVELENGTHS]
    errors = []
    differences = []
    for n in (20, 40, 80):
        scene, _ = h.build('A', n, .25, film=True, theta=theta, pol=pol)
        split = h.amplitudes(quiet(fd.run, scene, backend='numpy'), pol)
        errors.append(max(abs(complex(*split[m][i])-complex(*reference[i][m]))
                          for i in range(3) for m in ('r', 't')))
        sub, pillar = scene.structures
        stepped = scene.replace(structures=(sub, pillar.replace(junction='staircase')))
        staircase = h.amplitudes(quiet(fd.run, stepped, backend='numpy'), pol)
        differences.append(max(abs(complex(*split[m][i])-complex(*staircase[m][i]))
                               for i in range(3) for m in ('r', 't')))
    # Existing study gates and recorded row anchor|1790518977, not fitted here.
    np.testing.assert_allclose(errors, pins, rtol=1e-8, atol=0)
    assert -np.polyfit(np.log([20,40,80]), np.log(errors), 1)[0] >= 1.8
    assert -np.polyfit(np.log([20,40,80]), np.log(differences), 1)[0] <= 1.2


def test_a_formerly_refused_scene_agrees_with_rcwa():
    h = study()
    scene, _ = h.build('A', 20, .25)
    sub, pillar = scene.structures
    with pytest.raises(ValueError, match="asked for junction='refuse'"):
        quiet(fd.plan, scene.replace(structures=(sub,pillar.replace(junction='refuse'))))
    result = h.amplitudes(quiet(fd.run, scene, backend='numpy'), 's')
    reference = h.rcwa_point('A', 'flush', 9, 1.)['s']
    # The study measured 2.87e-2; its design registers .035 as the sanity gate.
    assert abs(complex(*result['t'][1])-complex(*reference['t'])) <= .035


@pytest.mark.parametrize('kind', ('cylinder', 'box'))
@pytest.mark.parametrize('tie', (0., .5))
def test_height_continuity_includes_the_exact_ties(kind, tie):
    from wavebit.fdtd._prepare import coefficients
    from wavebit.fdtd._runtime import runtime
    samples = []
    for offset in np.linspace(-.02, .02, 41):
        scene = scene_of(kind=kind, zeta=tie+offset)
        plan = quiet(fd.plan, scene)
        be = coefficients.build(scene, plan, runtime('numpy'))['be']
        k = round(-plan.low[2]/plan.dx)
        samples.append(be[:,k-2:k+3].copy())
    jumps = np.max(np.abs(np.diff(np.asarray(samples), axis=0)), axis=(1,2,3,4))
    assert np.isfinite(jumps).all()
    # V5's registered factor 50; exact ties were omitted by the study sweep.
    assert jumps.max() <= 50*np.median(jumps)+1e-12, jumps
