"""The public FDTD rename changes names only: no plan, no number, and no message keeps an old name.

The renamed declarations (``docs/migration/public-names.md``) lower to the engine's own classes,
which keep their names because a static plan records them. The skeletons below were measured
on ``main`` at ``d7bf1ab``, before the rename, for the same scenes written in the old names; a
change that moves one is not a rename. The messages a public caller reads are checked too: a
removed name or keyword must not reach them.
"""
import math
import re
import warnings

import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.materials import Debye, DispersiveMaterial, Lorentz
from wavebit.units import C0

UM = Units(length='um')


def snapped():
    return fdtd.Problem(units=UM, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.4, .4, .4)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.PEC()),
        structures=(fdtd.Structure('film', shape=fdtd.SnappedSlab(z=(-.1, .05)),
                                   material=Material(epsilon=2.25)),
                    fdtd.Structure('rod', shape=fdtd.SnappedBox(size=(.1, .1, .1), center=(0., 0., .1)),
                                   material=Material(epsilon=4.)))),
        initial=fdtd.InitialField(E=(None, lambda x, y, z: np.cos(np.pi * z / .4), None)),
        outputs=fdtd.Outputs(final_fields=True,
                             monitors=(fdtd.TimeSeriesMonitor('e', at=(0, 0, 0), component='Ey'),)))


def flush():
    glass = Material(epsilon=2.25)
    return fdtd.Problem(units=UM, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.8, .8, 1.2)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.PEC()),
        structures=(
            fdtd.Structure('substrate', shape=fdtd.Box(size=(2., 2., 1.2125), center=(0., 0., -.59375)),
                           material=glass),
            fdtd.Structure('pillar', shape=fdtd.Cylinder(radius=.15, height=.4, center=(.017, -.013, .2125)),
                           material=Material(epsilon=4.)),
            fdtd.Structure('film', shape=fdtd.SnappedSlab(z=(.5, .55)), material=glass))),
        sources=(fdtd.CurrentDipole('d', center=(.3, .3, .3), delay_seconds=1e-15,
                                    pulse=fdtd.GaussianPulse(wavelength=1.6, frequency_sigma_hz=2e13)),),
        outputs=fdtd.Outputs(final_fields=True,
                             monitors=(fdtd.TimeSeriesMonitor('e', at=(0, 0, 0), component='Ey'),)))


def boxes():
    pulse = fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=4e13)
    return fdtd.Problem(units=UM, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(1.2, 1.2, 1.2)), boundaries=fdtd.Boundaries.open()),
        sources=(fdtd.TFSF('t', box=fdtd.SnappedBox(size=(.3, .3, .3)), pulse=pulse,
                           delay_seconds=2e-15),),
        outputs=fdtd.Outputs(final_fields=True, monitors=(
            fdtd.FluxMonitor('closed', box=fdtd.SnappedBox(size=(.6, .6, .6)), wavelengths=(1.,)),
            fdtd.FluxMonitor('plane', plane=fdtd.Plane(z=.4, x=(-.2, .2), y=(-.2, .2)),
                             wavelengths=(1., .9)),
            fdtd.FarFieldMonitor('far', box=fdtd.SnappedBox(size=(.7, .7, .7)), wavelengths=(1.,)),
            fdtd.FieldMonitor('f', plane=fdtd.Plane(z=-.3), wavelengths=(1.,),
                              components=('Ex', 'Hy')))))


def invariant():
    source = fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.2), theta=math.asin(.3), delay_seconds=1e-16,
                            pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=.1 * C0 / 1e-6))
    faces = fdtd.Bloch.from_source(source, units=UM, invariant_axes=('x', 'y'))
    return fdtd.Problem(units=UM, scene=fdtd.Scene(domain=fdtd.Domain(size=(.01, .01, 1.)),
        boundaries=fdtd.Boundaries(**faces, z=fdtd.Open()),
        structures=(fdtd.Structure('slab', shape=fdtd.SnappedSlab(z=(-.5, -.1)),
                                   material=Material(epsilon=2.)),)),
        sources=(source,), outputs=fdtd.Outputs(final_fields=True, monitors=(
            fdtd.DiffractionMonitor('T', plane=fdtd.Plane(z=-.3), wavelengths=(1.,), source='pw',
                                    wave='transmitted', orders=0),
            fdtd.DiffractionMonitor('R', plane=fdtd.Plane(z=.35), wavelengths=(1.,), source='pw',
                                    wave='reflected', orders=0))))


def mirror():
    return fdtd.Problem(units=UM, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.4, .4, .4)),
        boundaries=fdtd.Boundaries(x=(fdtd.MirrorSymmetry(parity=1), fdtd.PEC()), y=fdtd.PEC(),
                                   z=fdtd.PEC())),
        initial=fdtd.InitialField(E=(None, lambda x, y, z: np.cos(np.pi * z / .4) * np.cos(np.pi * x / .4),
                                     None)),
        outputs=fdtd.Outputs(final_fields=True,
                             monitors=(fdtd.TimeSeriesMonitor('e', at=(0, 0, 0), component='Ey'),)))


def hybrid():
    units = Units(length='m')
    scale = units.time_seconds
    poles = (Debye(tau=1e-250 * scale, delta_epsilon=0.),
             Lorentz(omega_0=1e200 / scale, gamma=1e200 / scale, strength=0.),
             Debye(tau=.15 * scale, delta_epsilon=3.))
    return fdtd.Problem(units=units, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.4, .4, 2.)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic()),
        structures=(fdtd.Structure('film', shape=fdtd.Box(size=(.8, .8, .54), center=(0., 0., .03)),
                                   material=DispersiveMaterial(1., poles)),)))


def numerics(grid, steps=None, seconds=None, courant=None, pml=None, absorber=None, stop=False):
    kw = dict(grid=fdtd.UniformGrid(spacing=grid),
              duration=fdtd.Duration(seconds=seconds) if seconds else fdtd.Duration(steps=steps))
    if courant:
        kw['courant'] = courant
    if pml:
        kw['absorber'] = fdtd.PML(cells=pml, wavelength=1.)
    if absorber:
        kw['absorber'] = fdtd.ConductiveAbsorber(cells=absorber)
    if stop:
        kw['stopping'] = fdtd.DecayStop(threshold=1e-3, window_seconds=5e-16)
    return fdtd.Numerics(**kw)


#: (scene, numerics, skeleton measured at d7bf1ab before the rename, in the old names).
PINNED = {
    'snapped': (snapped, dict(grid=.05, steps=3),
                '0ed7027b12155b3eac0e6215794f0cd09747330816c3ccf2280dbc973e6e560e'),
    'junction': (flush, dict(grid=.05, steps=3),
                 '0f31d017f5eb2cea6f821085798b22f5ac78ec404b9358632c447f946a83869a'),
    'monitor_and_tfsf_boxes': (boxes, dict(grid=.05, steps=3, pml=12),
                               '8bdbacc2f5a19ba7d37b8066473a058d08d236a8ead193824ac06e404eaea360'),
    'invariant_axes': (invariant, dict(grid=.01, steps=3, pml=12),
                       '585c0f639d1391b806b3def375f4fe1f25751e631d114f42b99e5c41d9cf888d'),
    'mirror': (mirror, dict(grid=.05, steps=3),
               '4aabecfaf3413b46de91e93d009f217db5723032d3b5f516e8f06a492d33f8e8'),
    'pole_film': (hybrid, dict(grid=.1, steps=2, courant=.2),
                  '5e80326123bad1a809bb7e68b25be6c8333305ed23e3b5067f7ba2374fdb3816'),
    'decay_stop': (flush, dict(grid=.05, seconds=2e-15, stop=True),
                   '65950a1b4504c2c0f73c8982c83294ce1eb13088908dc256e4febb65de3321ff'),
    'conductive_absorber': (boxes, dict(grid=.05, steps=3, absorber=4),
                            '6dd31b8ca1331e86185d5a9a7d348a4db6d9814c317bc3c1fe769a1cc3bc098d'),
}


@pytest.mark.parametrize('case', sorted(PINNED))
def test_a_renamed_declaration_prepares_the_plan_its_old_name_did(case):
    make, spec, skeleton = PINNED[case]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prepared = fdtd.prepare(make(), numerics=numerics(**spec))
    assert prepared._plan.skeleton == skeleton


#: A removed public name or keyword, as it could appear in a message. The engine's own refusal
#: codes (``DecaySourceRefused``) and source kinds (``Dipole(E)``) are identifiers, not names a
#: caller writes, and are allowed.
REMOVED = re.compile(
    r'(?<![\w.])(?:fd\.|fdtd\.)?(Record|Field|Flux|Diffraction|FarField|Region|Slab|Dipole|Axis1D'
    r'|Symmetry|Absorber|Decay)\b(?!\((?:E|H)\))'
    r'|discretisation|axis1d|\bstabilize=|\bdecay=|\bdelay=|\bside=')


def _pulse():
    return fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=4e13)


def _open(**changes):
    problem = fdtd.Problem(units=UM, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(1., 1., 1.)), boundaries=fdtd.Boundaries.open()),
        sources=(fdtd.CurrentDipole('d', center=(0., 0., 0.), pulse=_pulse()),))
    return problem.replace(**changes)


def _num(**changes):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=2),
                         absorber=fdtd.PML(cells=12, wavelength=1.)).replace(**changes)


def _check_messages(problem, numerics):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        report = fdtd.check(problem, numerics=numerics)
    assert not report.ok
    return [issue.message for issue in report.issues] + [str(issue.path) for issue in report.issues]


def _outputs(*monitors):
    return dict(outputs=fdtd.Outputs(monitors=monitors))


#: Public misuses of every renamed declaration and keyword, refused at construction or by check.
CONSTRUCTION = {
    'time_series_point': lambda: fdtd.TimeSeriesMonitor('p', at=(0., 0.)),
    'time_series_component': lambda: fdtd.TimeSeriesMonitor('p', at=(0., 0., 0.), component='Qx'),
    'snapped_slab_axes': lambda: fdtd.SnappedSlab(x=(0., .1), z=(0., .1)),
    'snapped_box_size': lambda: fdtd.SnappedBox(size=(1., 1.)),
    'invariant_axis_k_max': lambda: fdtd.InvariantAxis(k=2., k_max=1.),
    'mirror_parity': lambda: fdtd.MirrorSymmetry(parity=2),
    'diffraction_wave': lambda: fdtd.DiffractionMonitor('r', plane=fdtd.Plane(z=0.), wavelengths=(1.,),
                                                        source='d', wave='up'),
}

CHECKED = {
    'flux_wavelengths': lambda: (_open(**_outputs(fdtd.FluxMonitor('f', wavelengths=(),
                                                                   plane=fdtd.Plane(z=.2)))), _num()),
    'flux_region': lambda: (_open(**_outputs(fdtd.FluxMonitor('f', wavelengths=(1.,)))), _num()),
    'field_components': lambda: (_open(**_outputs(fdtd.FieldMonitor(
        'f', wavelengths=(1.,), plane=fdtd.Plane(z=.2), components=('Qx',)))), _num()),
    'far_field_box_body': lambda: (_open(**_outputs(fdtd.FarFieldMonitor(
        'ff', wavelengths=(1.,), box=fdtd.Box(size=(.4, .4, .4))))), _num()),
    'far_field_periodic': lambda: (_open(scene=fdtd.Scene(domain=fdtd.Domain(size=(1., 1., 1.)),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Open())), **_outputs(fdtd.FarFieldMonitor(
            'ff', wavelengths=(1.,), box=fdtd.SnappedBox(size=(.4, .4, .4))))), _num()),
    'diffraction_without_plane_wave': lambda: (_open(**_outputs(fdtd.DiffractionMonitor(
        'r', plane=fdtd.Plane(z=.3), wavelengths=(1.,), source='d', wave='reflected'))), _num()),
    'mirror_both_faces': lambda: (_open(scene=fdtd.Scene(domain=fdtd.Domain(size=(1., 1., 1.)),
        boundaries=fdtd.Boundaries(x=fdtd.MirrorSymmetry(parity=1), y=fdtd.PEC(), z=fdtd.PEC()))),
        _num()),
    'dipole_kind': lambda: (_open(sources=(fdtd.CurrentDipole('d', center=(0., 0., 0.), pulse=_pulse(),
                                                              kind='magnetic', component='Ey'),)), _num()),
    'conductive_absorber_cells': lambda: (_open(), _num(absorber=fdtd.ConductiveAbsorber(cells=0))),
    'decay_stop_threshold': lambda: (_open(), _num(stopping=fdtd.DecayStop(threshold=2., window_seconds=1e-15))),
    'discretization_value': lambda: (_open(), _num(discretization='smooth')),
    'stabilizing_loss_value': lambda: (_open(), _num(absorber=fdtd.PML(cells=12, wavelength=1.,
                                                                       stabilizing_loss=-1.))),
    'tfsf_box_body': lambda: (_open(sources=(fdtd.TFSF('t', box=fdtd.Box(size=(.3, .3, .3)),
                                                       pulse=_pulse()),)), _num()),
    'snapped_slab_open_faces': lambda: (_open(scene=fdtd.Scene(domain=fdtd.Domain(size=(1., 1., 1.)),
        boundaries=fdtd.Boundaries.open(), structures=(fdtd.Structure(
            'film', shape=fdtd.SnappedSlab(z=(-.1, .1)), material=Material(epsilon=2.)),))), _num()),
}


@pytest.mark.parametrize('case', sorted(CONSTRUCTION))
def test_a_construction_refusal_names_no_removed_public_name(case):
    with pytest.raises((TypeError, ValueError)) as caught:
        CONSTRUCTION[case]()
    assert not REMOVED.search(str(caught.value)), str(caught.value)


@pytest.mark.parametrize('case', sorted(CHECKED))
def test_a_checked_refusal_names_no_removed_public_name(case):
    for text in _check_messages(*CHECKED[case]()):
        assert not REMOVED.search(text), text


def test_delay_seconds_stays_a_live_source_leaf():
    """The prepared signature excludes live fields by name; ``delay_seconds`` must be one of them."""
    problem = _open(outputs=fdtd.Outputs(final_fields=True))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prepared = fdtd.prepare(problem, numerics=_num())
        later = problem.replace(sources=(problem.sources[0].replace(delay_seconds=3e-16),))
        assert prepared.validate(later).ok
        prepared.run(later)
