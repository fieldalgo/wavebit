"""Stage 3's doors: every refusal the monitors, ``Run(time=)`` and the ``Region`` structure add.

Each refusal is paired with the accepted spelling beside it, so a door that refuses too much
fails here as surely as one that refuses too little (``tests/test_fd_sources_doors.py`` is the
stage-2 pattern this follows). Nothing here checks a number the physics produces -- the flux
closure, the orders against RCWA, the normalisations and the decay line are the gates' -- except
where a door's whole point is a number: ``Run(time=)``'s step count, the decay warning that must
fire on a cavity and stay quiet on an open box, and the ``Region`` structure's snapped cells.

The table, and where each door lives:

    monitor names: empty, or the reserved ``::``            monitors._monitor_name
    wavelengths: missing, empty, non-positive, repeated     monitors._wavelengths
    plane= and box= both or neither, or the wrong type      monitors._region
    Field components not in the six, or repeated            monitors.Field
    Diffraction position, side, orders, polarizations       monitors.Diffraction
    ModeMonitor and Decay                                   monitors (stage 7)
    Plane: no scalar normal, two of them, a reversed span   geometry.Plane
    Run: two lengths, a non-positive time, explicit decay and unsupported aliases   scene.Run
    a monitor type the Scene does not know                  scene.Scene
    a plane or a box outside the domain                     scene._monitor_inside
    a plane on the edge node, a box face on the edge, an    _lower/monitors.py
      extent past the last node or into a layer, a full
      plane across absorbing faces, an empty Field extent
    Diffraction without one PlaneWave, across another       _lower/monitors.py
      axis, on non-periodic faces, with unresolvable
      orders, in a cut or lossy medium
    Structure(shape) other than Slab and Region (stage 4)   geometry.Structure
    Region structure snapping to nothing on an axis         geometry.Region.cells
    Region ending inside an absorbing layer; lossy Region   _lower/checks.py
      in a PML; a Region overlapping a Dispersive
    Region at a PlaneWave sheet; Region touching a TFSF     _lower/sources.py
      face
    a Flux / Diffraction plane or a Flux / FarField box     _lower/monitors._clear_of_sources
      cutting a Dipole's stamp, on (or a box around) a
      PlaneWave sheet, or on a TFSF surface
    source spectra on the wrong source, or with no          results.SourceSeries
      wavelengths to default to

Build B of stage 3 adds three more families, at the bottom of this file:

    FarField: a box that is not a Region; a periodic cell  monitors.FarField,
      (names Diffraction); a wall or mirror face; a          _lower/monitors.farfield_checks
      structure not strictly inside the box; a box
      cutting a TFSF surface; a box reaching a PML
    Axis1D: algebraic=False; k missing; a traced k with    boundaries.Axis1D
      no k_max; |k| past k_max
    a user Bloch phase on a one-cell axis (names Axis1D)   _lower/checks.check_one_cell_bloch
    Axis1D on an axis of more than one cell; paired with   _kernel/faces.lower_face,
      another face; its CFL (k_max^2/4); a source that        boundaries._pair,
      spans cells along it; a Diffraction order along it;    _lower/checks.cfl_bound,
      a PlaneWave whose k it does not carry                  _lower/sources, _lower/monitors
    the grazing rule: an order or the PlaneWave's own      _lower/grazing.enforce
      incidence below cos_min of its face, below 0.41
      without 24 cells and 160 periods, or below 0.204
"""
import math
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from fd_helpers import (WAVELENGTHS, band, bloch_plane_wave, box_grating, conical_grating,
                        dipole_box, farfield_dipole, flux_box_dipole, pec_cavity_dipole,
                        tfsf_box)

WS = WAVELENGTHS


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def _plan(scene):
    return quiet(fd.plan, scene)


def _pw_scene(monitors=(), **kwargs):
    scene, _ = bloch_plane_wave(theta=20.0, at=0.6, steps=40, **kwargs)
    return scene.replace(monitors=tuple(monitors))


# ---------------------------------------------------------------------------------------------
# The monitor specs
# ---------------------------------------------------------------------------------------------

def test_a_monitor_name_is_a_non_empty_string_without_the_face_separator():
    plane = fd.Plane(z=0.0)
    with pytest.raises(ValueError, match='non-empty string'):
        fd.Flux('', plane=plane, wavelengths=WS)
    with pytest.raises(ValueError, match='reserved'):
        fd.Flux('a::b', plane=plane, wavelengths=WS)
    assert fd.Flux('a:b', plane=plane, wavelengths=WS).name == 'a:b'


@pytest.mark.parametrize('cls', [fd.Flux, fd.Field])
def test_wavelengths_are_a_static_list_of_distinct_positive_lengths(cls):
    plane = fd.Plane(z=0.0)
    with pytest.raises(ValueError, match='needs the wavelengths'):
        cls('m', plane=plane)
    with pytest.raises(ValueError, match='at least one'):
        cls('m', plane=plane, wavelengths=())
    with pytest.raises(ValueError, match='positive length'):
        cls('m', plane=plane, wavelengths=(1.0, -0.5))
    with pytest.raises(ValueError, match='repeated'):
        cls('m', plane=plane, wavelengths=(1.0, 1.0))
    assert cls('m', plane=plane, wavelengths=1.0).wavelengths == (1.0,)
    assert cls('m', plane=plane, wavelengths=np.array([0.9, 1.1])).wavelengths == (0.9, 1.1)


def test_a_traced_wavelength_is_refused_as_static():
    torch = pytest.importorskip('torch')
    w = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match='is static'):
        fd.Flux('m', plane=fd.Plane(z=0.0), wavelengths=(w,))


@pytest.mark.parametrize('cls', [fd.Flux, fd.Field])
def test_a_region_is_one_plane_or_one_box(cls):
    plane, box = fd.Plane(z=0.0), fd.Region(size=(0.2, 0.2, 0.2))
    with pytest.raises(ValueError, match='exactly one'):
        cls('m', wavelengths=WS)
    with pytest.raises(ValueError, match='exactly one'):
        cls('m', plane=plane, box=box, wavelengths=WS)
    with pytest.raises(TypeError, match='fd.Plane'):
        cls('m', plane=0.0, wavelengths=WS)
    with pytest.raises(TypeError, match='fd.Region'):
        cls('m', box=(0.2, 0.2, 0.2), wavelengths=WS)
    cls('m', plane=plane, wavelengths=WS)
    cls('m', box=box, wavelengths=WS)


def test_field_components_are_named_from_the_six():
    plane = fd.Plane(z=0.0)
    with pytest.raises(ValueError, match='drawn from'):
        fd.Field('f', plane=plane, components=('Ex', 'Dx'), wavelengths=WS)
    with pytest.raises(ValueError, match='repeated'):
        fd.Field('f', plane=plane, components=('Ex', 'Ex'), wavelengths=WS)
    assert fd.Field('f', plane=plane, components='Hz', wavelengths=WS).components == ('Hz',)


def test_a_diffraction_monitor_names_its_plane_side_and_window():
    with pytest.raises(ValueError, match='one position'):
        fd.Diffraction('r', wavelengths=WS)
    with pytest.raises(ValueError, match='one position'):
        fd.Diffraction('r', z=1.0, x=0.0, wavelengths=WS)
    with pytest.raises(ValueError, match="'reflect'"):
        fd.Diffraction('r', z=1.0, side='up', wavelengths=WS)
    with pytest.raises(ValueError, match='>= 0'):
        fd.Diffraction('r', z=1.0, orders=-1, wavelengths=WS)
    with pytest.raises(ValueError, match='pair'):
        fd.Diffraction('r', z=1.0, orders=(1, 2, 3), wavelengths=WS)
    with pytest.raises(ValueError, match="'both'"):
        fd.Diffraction('r', z=1.0, polarizations='s', wavelengths=WS)
    d = fd.Diffraction('r', z=1.0, side='transmit', orders=2, wavelengths=WS,
                       reference_plane=0.0, polarizations='both')
    assert d.orders == (2, 2) and d.axis == 2 and d.position == 1.0
    assert fd.Diffraction('r', z=1.0, orders=(3, 0), wavelengths=WS).orders == (3, 0)


def test_mode_monitors_and_decay_stopping_are_stage_7():
    with pytest.raises(NotImplementedError, match='stage 7'):
        fd.ModeMonitor('m')
    # Decay is now explicit, but both the ratio and physical window are required.
    with pytest.raises(TypeError, match='window'):
        fd.Decay(1e-5)


def test_a_plane_names_one_normal_and_optional_spans():
    with pytest.raises(ValueError, match='ONE scalar'):
        fd.Plane()
    with pytest.raises(ValueError, match='ONE scalar'):
        fd.Plane(z=0.0, x=0.1)
    with pytest.raises(ValueError, match='lo < hi'):
        fd.Plane(z=0.0, x=(0.2, -0.2))
    with pytest.raises(ValueError, match='pair'):
        fd.Plane(z=0.0, x=(0.1, 0.2, 0.3))
    p = fd.Plane(z=0.2, x=(-0.3, 0.3))
    assert p.axis == 2 and p.position == 0.2 and p.extent(0) == (-0.3, 0.3)
    assert p.extent(1) is None


# ---------------------------------------------------------------------------------------------
# Run(time=)
# ---------------------------------------------------------------------------------------------

def test_run_takes_one_length_and_time_counts_steps_to_the_end_time():
    with pytest.raises(ValueError, match='one of steps, periods and time'):
        fd.Run(steps=10, time=1.0)
    with pytest.raises(ValueError, match='one of steps, periods and time'):
        fd.Run(periods=10, time=1.0)
    with pytest.raises(ValueError, match='positive end time'):
        fd.Run(time=0.0)
    with pytest.raises(ValueError, match='length of the run'):
        fd.Run()
    scene = dipole_box(size=0.6, steps=5)
    for time in (0.5, 1.0, 1.2345):
        p = _plan(scene.replace(run=fd.Run(time=time)))
        assert p.steps == math.ceil(time / p.dt - 1e-9)
        assert (p.steps - 1) * p.dt < time <= p.steps * p.dt + 1e-12
    # A time that is a whole number of steps is that many steps, not one more.
    p0 = _plan(scene)
    assert _plan(scene.replace(run=fd.Run(time=7 * p0.dt))).steps == 7


@pytest.mark.parametrize('keyword', ['decay', 'until', 'shutoff'])
def test_stopping_on_decay_is_refused_naming_stage_7(keyword):
    # A bare ratio lacks a window; alternate stopping spellings remain refused.
    exception = TypeError if keyword == 'decay' else NotImplementedError
    with pytest.raises(exception, match='Decay') as e:
        fd.Run(steps=10, **{keyword: 1e-5})
    assert 'window' in str(e.value)


# ---------------------------------------------------------------------------------------------
# The scene and the plan
# ---------------------------------------------------------------------------------------------

def test_the_scene_takes_the_five_monitor_types_and_nothing_else():
    scene = dipole_box(size=0.6, steps=5)
    with pytest.raises(TypeError, match='fd.Record, fd.Flux, fd.Field, fd.FarField or '
                                        'fd.Diffraction'):
        scene.replace(monitors=(fd.Region(size=(0.1, 0.1, 0.1)),))
    one = fd.Flux('f', box=fd.Region(size=(0.3, 0.3, 0.3)), wavelengths=WS)
    assert scene.replace(monitors=one).monitors == (one,)
    with pytest.raises(ValueError, match="repeated: \\['f'\\]"):
        scene.replace(monitors=(one, fd.Record('f', at=(0, 0, 0))))


def test_a_monitor_region_must_lie_in_the_domain():
    scene = dipole_box(size=0.6, steps=5)
    with pytest.raises(ValueError, match='outside the domain'):
        scene.replace(monitors=(fd.Flux('f', plane=fd.Plane(z=5.0, x=(-0.1, 0.1),
                                                            y=(-0.1, 0.1)), wavelengths=WS),))
    with pytest.raises(ValueError, match='outside the domain'):
        scene.replace(monitors=(fd.Flux('f', box=fd.Region(size=(0.2, 0.2, 0.2),
                                                        center=(2.0, 0, 0)),
                                        wavelengths=WS),))
    with pytest.raises(ValueError, match='outside the domain'):
        _pw_scene().replace(monitors=(fd.Diffraction('r', z=9.0, wavelengths=WS),))


def test_a_flux_plane_needs_a_cell_either_side_and_a_finite_extent_in_an_open_box():
    scene = dipole_box(size=0.6, steps=5)
    edge = scene.replace(monitors=(fd.Flux('f', plane=fd.Plane(z=-scene.domain.size[2] / 2,
                                                               x=(-0.1, 0.1), y=(-0.1, 0.1)),
                                           wavelengths=WS),))
    with pytest.raises(ValueError, match='one cell on each side'):
        _plan(edge)
    whole = scene.replace(monitors=(fd.Flux('f', plane=fd.Plane(z=0.0), wavelengths=WS),))
    with pytest.raises(ValueError, match='whole x axis, whose faces absorb'):
        _plan(whole)
    into = scene.replace(monitors=(fd.Flux('f', plane=fd.Plane(z=0.0, x=(-0.5, 0.1),
                                                               y=(-0.1, 0.1)),
                                           wavelengths=WS),))
    with pytest.raises(ValueError, match='reaches the PML'):
        _plan(into)
    thin = scene.replace(monitors=(fd.Flux('f', plane=fd.Plane(z=0.0, x=(0.0, 0.01),
                                                               y=(-0.1, 0.1)),
                                           wavelengths=WS),))
    with pytest.raises(ValueError, match='snaps to no cell'):
        _plan(thin)
    # z = 0.15, three cells above the dipole: clear of its stamp (next test).
    ok = scene.replace(monitors=(fd.Flux('f', plane=fd.Plane(z=0.15, x=(-0.1, 0.1),
                                                             y=(-0.1, 0.1)),
                                         wavelengths=WS),))
    assert any(getattr(o, 'monitor', None) == 'f' for o in _plan(ok).observers)
    # In a periodic cell the plane runs the whole period, which is the exact sum.
    _plan(_pw_scene(monitors=(fd.Flux('f', plane=fd.Plane(z=0.0), wavelengths=WS),)))


def test_a_flux_box_stays_clear_of_the_absorbers_and_the_domain_edge():
    scene = dipole_box(size=0.6, steps=5)
    big = scene.replace(monitors=(fd.Flux('f', box=fd.Region(size=(0.9, 0.9, 0.9)),
                                          wavelengths=WS),))
    with pytest.raises(ValueError, match='reaches the PML'):
        _plan(big)
    cell = scene.replace(boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                         monitors=(fd.Flux('f', box=fd.Region(size=scene.domain.size),
                                           wavelengths=WS),))
    with pytest.raises(ValueError, match='clearance from the domain'):
        _plan(cell)
    ok = scene.replace(monitors=(fd.Flux('f', box=fd.Region(size=(0.4, 0.4, 0.4)),
                                         wavelengths=WS),))
    names = [o.name for o in _plan(ok).observers]
    assert names == [f'f::{face}' for face in ('x-', 'x+', 'y-', 'y+', 'z-', 'z+')]


def test_a_flux_surface_keeps_off_a_dipoles_stamp():
    """The stamp writes two cells either side of the centre and the 2-cell face reads one cell
    either side of the surface: a 0.05 box inside the stamp missed 0.916 of the power, a 0.3 box
    cutting its tail 1.6e-3. Enclosing it with a cell to spare, or staying a cell clear, is
    accepted -- for Flux boxes and planes and for FarField boxes alike."""
    scene = dipole_box(size=0.6, steps=5)            # the dipole on a node at the origin

    def flux(**region):
        return scene.replace(monitors=(fd.Flux('f', wavelengths=WS, **region),))

    for size in (0.1, 0.2):                          # inside the stamp; through its tail
        with pytest.raises(ValueError, match=r"cuts the stamp of Dipole\('d'\)") as e:
            _plan(flux(box=fd.Region(size=(size,) * 3)))
        assert 'enclose it with one cell to spare' in str(e.value)
        with pytest.raises(ValueError, match='cuts the stamp'):
            _plan(scene.replace(monitors=(fd.FarField('ff', box=fd.Region(size=(size,) * 3),
                                                      wavelengths=WS),)))
    _plan(flux(box=fd.Region(size=(0.3,) * 3)))         # Ez reaches z -1.5..1.5 cells: 3 clear
    _plan(scene.replace(monitors=(fd.FarField('ff', box=fd.Region(size=(0.3,) * 3),
                                              wavelengths=WS),)))
    _plan(flux(box=fd.Region(size=(0.1,) * 3, center=(0.0, 0.0, 0.2))))   # beside it, clear
    with pytest.raises(ValueError, match='keep the plane one cell clear'):
        _plan(flux(plane=fd.Plane(z=0.05, x=(-0.1, 0.1), y=(-0.1, 0.1))))
    _plan(flux(plane=fd.Plane(z=0.15, x=(-0.1, 0.1), y=(-0.1, 0.1))))
    _plan(flux(plane=fd.Plane(z=0.05, x=(0.15, 0.25), y=(-0.1, 0.1))))  # off to the side


def test_a_flux_or_diffraction_plane_keeps_off_a_plane_wave_sheet():
    """On the sheet's node plane the 2-cell face pairs the total field below with the scattered
    field above (a Diffraction there read R = 0.08 for a stack whose R is 0.001); one cell off
    reads either field whole. A box around part of the sheet is refused too."""
    sheet = 0.6                                       # _pw_scene: direction -z, sheet at 0.6
    for monitor in (fd.Diffraction('r', z=sheet, wavelengths=WS),
                    fd.Flux('f', plane=fd.Plane(z=sheet), wavelengths=WS)):
        with pytest.raises(ValueError, match=r"lies on the sheet of PlaneWave\('pw'\)") as e:
            _plan(_pw_scene(monitors=(monitor,)))
        message = str(e.value)
        assert 'z = 0.55 or below it reads the total field' in message
        assert 'z = 0.65 or above the scattered field' in message
    for z in (sheet - 0.05, sheet + 0.05):
        _plan(_pw_scene(monitors=(fd.Diffraction('r', z=z, wavelengths=WS),
                                  fd.Flux('f', plane=fd.Plane(z=z), wavelengths=WS))))
    around = fd.Flux('f', box=fd.Region(size=(0.1, 0.1, 0.2), center=(0.0, 0.0, sheet)),
                     wavelengths=WS)
    with pytest.raises(ValueError, match='encloses part of the sheet'):
        _plan(_pw_scene(monitors=(around,)))
    _plan(_pw_scene(monitors=(fd.Flux('f', box=fd.Region(size=(0.1, 0.1, 0.2),
                                                      center=(0.0, 0.0, sheet - 0.2)),
                                      wavelengths=WS),)))


def test_a_flux_surface_keeps_off_a_tfsf_surface():
    """The TFSF box (nodes 22..32 on every axis) splits total from scattered field; a flux box
    strictly inside it or around it with one cell to spare reads one of them, and one that cuts
    it -- or a plane on its face -- reads both."""
    def with_flux(**region):
        return tfsf_box(steps=5, monitors=(fd.Flux('f', wavelengths=WS, **region),))

    with pytest.raises(ValueError, match="cuts or lies on the surface of TFSF"):
        _plan(with_flux(box=fd.Region(size=(0.5,) * 3)))
    with pytest.raises(ValueError, match="cuts or lies on the surface of TFSF"):
        _plan(with_flux(box=fd.Region(size=(0.3,) * 3, center=(0.25, 0.0, 0.0))))
    with pytest.raises(ValueError, match="cuts or lies on the surface of TFSF"):
        _plan(with_flux(plane=fd.Plane(z=0.25, x=(-0.1, 0.1), y=(-0.1, 0.1))))
    _plan(with_flux(box=fd.Region(size=(0.4,) * 3)))                   # inside, a cell to spare
    _plan(with_flux(box=fd.Region(size=(0.6,) * 3)))                   # around, a cell to spare
    _plan(with_flux(plane=fd.Plane(z=0.3, x=(-0.1, 0.1), y=(-0.1, 0.1))))   # a cell clear
    _plan(with_flux(plane=fd.Plane(z=0.2, x=(-0.1, 0.1), y=(-0.1, 0.1))))   # a cell inside


def test_a_field_extent_must_hold_a_point_of_each_component():
    scene = dipole_box(size=0.6, steps=5)
    empty = scene.replace(monitors=(fd.Field('f', plane=fd.Plane(z=0.0, x=(0.001, 0.002)),
                                             components=('Ex',), wavelengths=WS),))
    with pytest.raises(ValueError, match='no Ex point'):
        _plan(empty)
    ok = scene.replace(monitors=(fd.Field('f', plane=fd.Plane(z=0.0, x=(-0.1, 0.1)),
                                          components=('Ex', 'Hz'), wavelengths=WS),))
    _plan(ok)


def test_diffraction_needs_one_plane_wave_on_its_axis_in_a_periodic_cell():
    ok = fd.Diffraction('r', z=1.0, wavelengths=WS)
    with pytest.raises(ValueError, match='has 0 PlaneWave'):
        _plan(dipole_box(size=0.6, steps=5).replace(
            boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                     z=fd.PML(cells=12, wavelength=1.0)),
            monitors=(fd.Diffraction('r', z=0.0, wavelengths=WS),)))
    with pytest.raises(ValueError, match='travels along z'):
        _plan(_pw_scene(monitors=(fd.Diffraction('r', x=0.0, wavelengths=WS),)))
    _plan(_pw_scene(monitors=(ok,)))


def test_diffraction_refuses_a_window_its_cells_cannot_resolve():
    scene = _pw_scene(monitors=(fd.Diffraction('r', z=1.0, orders=(3, 0), wavelengths=WS),))
    with pytest.raises(ValueError, match='alias'):
        _plan(scene)                                   # 5 cells across x: at most M = 2
    _plan(scene.replace(monitors=(fd.Diffraction('r', z=1.0, orders=(2, 1), wavelengths=WS),)))


def test_diffraction_refuses_a_plane_in_a_cut_or_lossy_medium():
    scene, info = box_grating('lamellar', periods=1)
    r, t = scene.monitors
    inside = scene.replace(monitors=(r, t.replace(z=-0.1)))          # through the tooth
    with pytest.raises(ValueError, match='not one medium'):
        _plan(inside)
    lossy = fd.Structure(fd.Slab(z=(-5.0, -0.2)), fd.Conductor(eps=2.25, sigma=0.1))
    lossy_scene = scene.replace(structures=scene.structures + (lossy,),
                                boundaries=fd.Boundaries(**{a: scene.boundaries.face(a + '-')
                                                            for a in 'xy'},
                                                         z=(fd.Absorber(cells=12),
                                                            fd.PML(cells=12))))
    with pytest.raises(ValueError, match='fd.Conductor'):
        _plan(lossy_scene)
    _plan(scene)


def test_the_source_spectra_are_read_off_the_right_source():
    scene = flux_box_dipole(size=1.2, periods=1, empty_center=(0.3, 0.3, 0.25), box=0.3,
                            empty_size=0.2)
    res = quiet(fd.run, scene)
    with pytest.raises(TypeError, match='TFSF or PlaneWave'):
        res.sources['d'].intensity()
    assert np.shape(res.sources['d'].power()) == (len(WS),)
    assert np.shape(res.sources['d'].power((1.0,))) == (1,)
    lone = quiet(fd.run, dipole_box(size=0.6, steps=5))
    with pytest.raises(ValueError, match='say which wavelengths'):
        lone.sources['d'].power()
    with pytest.raises(TypeError, match='Dipole'):
        quiet(fd.run, _pw_scene()).sources['pw'].power((1.0,))


# ---------------------------------------------------------------------------------------------
# The decay diagnostic
# ---------------------------------------------------------------------------------------------

def test_an_unsettled_frequency_domain_monitor_warns_by_name_and_a_settled_one_does_not():
    with pytest.warns(fd.UnsettledFieldWarning, match="monitor 'in'"):
        res = fd.run(pec_cavity_dipole(periods=3))
    assert res.diagnostics['decay']['in'] > 1e-2
    settled = flux_box_dipole(size=1.2, periods=10, empty_center=(0.3, 0.3, 0.25), box=0.3,
                              empty_size=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        warnings.simplefilter('error', fd.UnsettledFieldWarning)
        res = fd.run(settled)
    assert all(v < 1e-5 for v in res.diagnostics['decay'].values())


def test_a_run_with_no_frequency_domain_monitor_reports_and_never_warns():
    with warnings.catch_warnings():
        warnings.simplefilter('error', fd.UnsettledFieldWarning)
        res = fd.run(dipole_box(size=0.6, steps=20,
                                monitors=(fd.Record('p', at=(0.1, 0, 0), component='Ez'),)))
    assert res.diagnostics['decay'] == {} and 'p' in res.diagnostics['records']
    assert res.diagnostics['runs'] == 1


# ---------------------------------------------------------------------------------------------
# Region as a structure
# ---------------------------------------------------------------------------------------------

def test_a_box_is_a_structure_and_other_shapes_are_stage_4():
    glass = fd.Material(eps=2.25)
    fd.Structure(fd.Region(size=(0.2, 0.2, 0.2)), glass)
    with pytest.raises(TypeError, match='stage 4'):
        fd.Structure(fd.Plane(z=0.0), glass)
    with pytest.raises(NotImplementedError, match='fd.Region') as e:
        fd.Slab(z=(0, 1), x=(0, 1))
    assert 'stage 4' in str(e.value)
    with pytest.raises(NotImplementedError, match='stage 4') as e:
        fd.Slab(size=1.0)
    assert 'fd.Region' in str(e.value)


def test_a_box_snaps_to_cells_on_every_axis_and_the_report_says_so():
    scene = dipole_box(size=0.6, steps=5)
    box = fd.Region(size=(0.23, 0.2, 0.11), center=(0.013, 0.0, -0.02))
    p = _plan(scene.replace(structures=(fd.Structure(box, 2.25),)))
    rows = [row for row in p.spans if row[0].startswith('box.')]
    assert [row[0] for row in rows] == ['box.x', 'box.y', 'box.z']
    grid = p.grid
    for a, (label, lo, hi, slo, shi) in enumerate(rows):
        assert (lo, hi) == (box.low[a], box.high[a])
        for s in (slo, shi):
            k = (s - grid.low[a]) / grid.dx
            assert abs(k - round(k)) < 1e-9                  # a cell boundary
        assert abs(slo - lo) <= grid.dx / 2 + 1e-12 and abs(shi - hi) <= grid.dx / 2 + 1e-12
    report = str(quiet(fd.check, scene.replace(structures=(fd.Structure(box, 2.25),))))
    assert 'box.x=' in report and 'snapped' in report
    with pytest.raises(ValueError, match='snaps to the empty span'):
        _plan(scene.replace(structures=(fd.Structure(fd.Region(size=(0.2, 0.2, 0.01)), 2.25),)))


def test_a_box_scene_without_monitors_keeps_the_slab_skeleton_rule():
    """A Slab scene's plan (and so every stage-0..2 golden's skeleton) is untouched by stage 3."""
    scene = dipole_box(size=0.6, steps=5)
    slab = scene.replace(structures=(fd.Structure(fd.Slab(z=(-0.1, 0.1)), 2.25),))
    rows = _plan(slab).spans
    assert len(rows) == 1 and rows[0][0] == 'z'


def test_a_box_obeys_the_absorbing_layer_rules():
    scene = dipole_box(size=0.6, steps=5)
    n = scene.domain.size[0]                  # 0.6 of interior and a 0.6-deep layer each side
    ends_inside = fd.Structure(fd.Region(size=(0.7, 0.2, 0.2), center=(0.25, 0.0, 0.0)), 2.25)
    with pytest.raises(ValueError, match='ends inside the PML'):
        _plan(scene.replace(structures=(ends_inside,)))
    through = fd.Structure(fd.Region(size=(2 * n, 0.2, 0.2)), fd.Conductor(eps=2.25, sigma=0.1))
    with pytest.raises(ValueError, match='fd.Absorber'):
        _plan(scene.replace(structures=(through,)))
    _plan(scene.replace(structures=(fd.Structure(fd.Region(size=(2 * n, 0.2, 0.2)), 2.25),)))
    inside = fd.Structure(fd.Region(size=(0.2, 0.2, 0.2)), fd.Conductor(eps=2.25, sigma=0.1))
    _plan(scene.replace(structures=(inside,)))


def test_a_box_overlapping_a_dispersive_structure_is_refused():
    scene = dipole_box(size=0.6, steps=5)
    metal = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=5.0, gamma=0.5),))
    box = fd.Structure(fd.Region(size=(0.2, 0.2, 0.2)), metal)
    with pytest.raises(ValueError, match='share cells'):
        _plan(scene.replace(structures=(box, fd.Structure(fd.Slab(z=(-0.05, 0.05)), 2.25))))
    apart = fd.Structure(fd.Region(size=(0.1, 0.1, 0.1), center=(0.15, 0.15, 0.15)), 2.25)
    quiet(_plan, scene.replace(structures=(box, apart)))


def test_a_box_must_stay_clear_of_a_plane_wave_sheet():
    scene = _pw_scene()
    at_sheet = fd.Structure(fd.Region(size=(0.1, 0.1, 0.2), center=(0.0, 0.0, 0.6)), 2.25)
    with pytest.raises(ValueError, match='inside structures\\[0\\]'):
        _plan(scene.replace(structures=(at_sheet,)))
    below = fd.Structure(fd.Region(size=(0.1, 0.1, 0.2), center=(0.0, 0.0, -0.3)), 2.25)
    _plan(scene.replace(structures=(below,)))


def test_a_box_scatterer_sits_inside_a_tfsf_box_or_clear_of_it():
    scene = tfsf_box(box=(0.5, 0.5, 0.5), steps=5)
    inside = fd.Structure(fd.Region(size=(0.2, 0.2, 0.2)), 2.25)
    _plan(scene.replace(structures=(inside,)))
    outside = fd.Structure(fd.Region(size=(0.1, 0.1, 0.1), center=(0.0, 0.0, 0.45)), 2.25)
    _plan(scene.replace(structures=(outside,)))
    across = fd.Structure(fd.Region(size=(0.2, 0.2, 0.4), center=(0.0, 0.0, 0.25)), 2.25)
    with pytest.raises(ValueError, match='touches the surface of TFSF'):
        _plan(scene.replace(structures=(across,)))
    flush = fd.Structure(fd.Region(size=(0.2, 0.2, 0.25), center=(0.0, 0.0, 0.125)), 2.25)
    with pytest.raises(ValueError, match='one cell to spare'):
        _plan(scene.replace(structures=(flush,)))


# ---------------------------------------------------------------------------------------------
# What check() and the result say
# ---------------------------------------------------------------------------------------------

def test_check_prints_the_dft_bytes_and_a_second_solve():
    scene = _pw_scene(monitors=(fd.Diffraction('r', z=1.0, wavelengths=WS,
                                               polarizations='both'),
                                fd.Flux('f', plane=fd.Plane(z=0.0), wavelengths=WS)))
    report = quiet(fd.check, scene)
    p = _plan(scene)
    regions = {o.monitor: o for o in p.observers if hasattr(o, 'monitor')}
    assert dict((n, b) for n, _, b in report.monitors) == {
        name: o.nbytes(p) for name, o in regions.items()}
    assert report.solves == 2 and 'solves     2' in str(report)
    plain = quiet(fd.check, _pw_scene())
    assert plain.monitors == () and 'monitor ' not in str(plain) and plain.solves == 1


def test_polarizations_both_runs_twice_and_says_so():
    scene = _pw_scene(monitors=(fd.Diffraction('r', z=1.0, wavelengths=(1.0,),
                                               polarizations='both'),))
    res = quiet(fd.run, scene)
    r = res['r']
    assert r.runs == 2 and res.diagnostics['runs'] == 2
    assert r.orders.s_inc is r.orders.incident and r.orders.p_inc is not None
    once = quiet(fd.run, _pw_scene(monitors=(fd.Diffraction('r', z=1.0, wavelengths=(1.0,)),)))
    assert once['r'].runs == 1 and once['r'].orders.s_inc is None
    assert once['r'].kz_convention == 'yee'


def test_a_bare_band_helper_still_builds():
    assert band().concrete_wavelength == 1.0


# ---------------------------------------------------------------------------------------------
# FarField (stage 3, Build B)
# ---------------------------------------------------------------------------------------------

def _ff(box=0.4, **kwargs):
    return fd.FarField('ff', box=fd.Region(size=(box,) * 3, **kwargs), wavelengths=WS)


def test_a_far_field_is_a_closed_box():
    with pytest.raises(TypeError, match='closed surface'):
        fd.FarField('ff', box=fd.Plane(z=0.0), wavelengths=WS)
    with pytest.raises(TypeError, match='closed surface'):
        fd.FarField('ff', wavelengths=WS)
    with pytest.raises(ValueError, match='needs the wavelengths'):
        fd.FarField('ff', box=fd.Region(size=(0.4,) * 3))
    with pytest.raises(ValueError, match='reserved'):
        fd.FarField('f::f', box=fd.Region(size=(0.4,) * 3), wavelengths=WS)
    assert _ff().wavelengths == WS
    names = [o.name for o in _plan(dipole_box(size=0.6, steps=5, monitors=(_ff(),))).observers]
    assert names == [f'ff::{face}' for face in ('x-', 'x+', 'y-', 'y+', 'z-', 'z+')]


def test_a_far_field_in_a_periodic_cell_is_refused_naming_diffraction():
    with pytest.raises(ValueError, match='fd.Diffraction'):
        _plan(_pw_scene(monitors=(fd.FarField('ff', box=fd.Region(size=(0.1, 0.1, 0.4)),
                                              wavelengths=WS),)))
    _plan(dipole_box(size=0.6, steps=5, monitors=(_ff(),)))


def test_a_far_field_needs_every_face_absorbing():
    scene = dipole_box(size=0.6, steps=5, monitors=(_ff(),))
    walled = scene.replace(boundaries=fd.Boundaries.open(fd.PML(cells=12),
                                                         z=(fd.PEC(), fd.PML(cells=12))))
    with pytest.raises(ValueError, match="face 'z-' is PEC"):
        _plan(walled)
    _plan(scene.replace(boundaries=fd.Boundaries.open(fd.Absorber(cells=12))))


def test_a_far_field_box_holds_every_structure_strictly_inside():
    scene = dipole_box(size=0.6, steps=5, monitors=(_ff(),))
    slab = fd.Structure(fd.Slab(z=(-0.1, 0.1)), 2.25)
    with pytest.raises(ValueError, match='strictly inside the box'):
        _plan(scene.replace(structures=(slab,)))
    flush = fd.Structure(fd.Region(size=(0.4, 0.1, 0.1)), 2.25)            # on the x faces
    with pytest.raises(ValueError, match='strictly inside the box'):
        _plan(scene.replace(structures=(flush,)))
    _plan(scene.replace(structures=(fd.Structure(fd.Region(size=(0.2, 0.1, 0.1)), 2.25),)))


def test_a_far_field_box_encloses_a_tfsf_box_or_keeps_clear():
    scene = tfsf_box(box=(0.3, 0.3, 0.3), steps=5)
    with pytest.raises(ValueError, match='cuts the surface of TFSF'):
        _plan(scene.replace(monitors=(_ff(box=0.3),)))
    _plan(scene.replace(monitors=(_ff(box=0.6),)))


def test_a_far_field_box_stays_clear_of_the_layers():
    scene = dipole_box(size=0.6, steps=5)
    with pytest.raises(ValueError, match='reaches the PML'):
        _plan(scene.replace(monitors=(_ff(box=0.9),)))
    _plan(scene.replace(monitors=(_ff(box=0.5),)))


def test_a_far_field_result_is_evaluated_at_the_angles_asked():
    res = quiet(fd.run, farfield_dipole(size=0.6, box=0.4, periods=1))
    ff = res['ff']
    theta = np.linspace(0.0, 180.0, 7)
    assert np.shape(ff.intensity(theta, 0.0)) == (len(WS), 7)
    assert np.shape(ff.intensity(theta[:, None], np.array([0.0, 90.0])[None, :])) == (
        len(WS), 7, 2)
    E_t, E_p = ff.fields(90.0, 30.0)
    assert np.shape(E_t) == (len(WS),) and np.shape(E_p) == (len(WS),)
    assert np.shape(ff.flux) == (len(WS),)
    assert set(ff.faces) == {'x-', 'x+', 'y-', 'y+', 'z-', 'z+'}
    assert 'ff' in res.diagnostics['decay']


# ---------------------------------------------------------------------------------------------
# Axis1D (stage 3, Build B)
# ---------------------------------------------------------------------------------------------

def _axis1d_scene(y=None, ny=1, steps=5, **kwargs):
    dx = 1.0 / 16
    face = fd.Axis1D(k=2.0) if y is None else y
    return fd.Scene(domain=fd.Domain(size=(1.0, ny * dx, 1.0)),
                    boundaries=fd.Boundaries(x=fd.Periodic(), y=face, z=fd.Periodic()),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps), **kwargs)


def test_axis1d_takes_a_real_k_and_the_k_max_it_is_planned_for():
    with pytest.raises(ValueError, match='algebraic i k'):
        fd.Axis1D(k=1.0, algebraic=False)
    with pytest.raises(ValueError, match='k=None'):
        fd.Axis1D(k=None)
    with pytest.raises(ValueError, match='above k_max'):
        fd.Axis1D(k=2.0, k_max=1.0)
    with pytest.raises(ValueError, match='a real scalar'):
        fd.Axis1D(k=1j)
    with pytest.raises(ValueError, match='non-negative'):
        fd.Axis1D(k=0.0, k_max=-1.0)
    assert fd.Axis1D(k=-2.0).k_max == 2.0
    assert fd.Axis1D(k=0.0).k_max == 0.0
    assert fd.Axis1D(k=1.0, k_max=3.0).k_max == 3.0


def test_a_traced_axis1d_k_needs_k_max():
    jax = pytest.importorskip('jax')
    seen = []

    def f(k):
        with pytest.raises(ValueError, match='needs k_max'):
            fd.Axis1D(k=k)
        seen.append(fd.Axis1D(k=k, k_max=4.0).k_max)
        return k
    jax.make_jaxpr(f)(1.0)
    assert seen == [4.0]


def test_a_bloch_phase_on_a_one_cell_axis_is_refused_naming_axis1d():
    scene, _ = conical_grating(periods=1)
    pw = scene.sources[0]
    one_cell = scene.replace(boundaries=fd.Boundaries(**fd.Bloch.from_source(pw),
                                                      z=scene.boundaries.face('z-')))
    with pytest.raises(ValueError, match='fd.Axis1D') as e:
        _plan(one_cell)                                             # y: Bloch(k) on one cell
    assert 'artificial' in str(e.value) and "axis1d='y'" in str(e.value)
    _plan(scene)                                                    # y: Axis1D(k)
    dx = scene.resolution.dx
    two = one_cell.replace(domain=one_cell.domain.replace(
        size=(one_cell.domain.size[0], 2 * dx, one_cell.domain.size[2])))
    _plan(two)                                                      # two cells: a real period
    assert _plan(_axis1d_scene()).dtype == 'complex128'
    assert _plan(_axis1d_scene(y=fd.Axis1D(k=0.0))).dtype == 'complex128'   # always complex
    _plan(_axis1d_scene(y=fd.Periodic()))                           # k = 0 exactly: no dispersion


def test_a_source_free_kernel_study_keeps_its_one_cell_bloch_line():
    """The stage-1 packet studies (and their golden) run a 1 x 1 x nz line under a one-cell
    Bloch phase, the exact stand-in for a multi-cell periodic grid's plane wave."""
    _plan(_axis1d_scene(y=fd.Bloch(k=2.0)))
    _plan(_axis1d_scene(y=fd.Bloch(phase=0.1)))


def test_the_stage_2_one_cell_columns_are_not_user_faces():
    """A PlaneWave's column is one cell across and Bloch; the one-cell door never sees it."""
    p = _plan(_pw_scene())
    columns = [t.sub for t in p.terms if hasattr(t, 'sub')]
    assert columns and all(sub.nx == 1 and sub.ny == 1 for sub in columns)
    assert all(dict(sub.face_rules)['x-'].kind == 'bloch' for sub in columns)


def test_axis1d_is_one_cell_and_wraps_both_faces():
    with pytest.raises(ValueError, match='one cell'):
        _plan(_axis1d_scene(ny=4))
    with pytest.raises(ValueError, match='Axis1D wrap the axis'):
        fd.Boundaries(x=fd.Periodic(), y=(fd.Axis1D(k=1.0), fd.PEC()), z=fd.Periodic())
    one = fd.Axis1D(k=1.0)
    fd.Boundaries(x=fd.Periodic(), y=(one, one), z=fd.Periodic())


def test_the_cfl_bound_carries_k_max_squared_over_four():
    """dt < n / sqrt(2/dx^2 + k_max^2/4): at courant 0.5 the line is k_max dx = 2 sqrt(10)."""
    dx = 1.0 / 16
    with pytest.raises(ValueError, match='k_max\\^2/4') as e:
        _plan(_axis1d_scene(y=fd.Axis1D(k=1.0, k_max=6.5 / dx)))
    assert 'stability bound' in str(e.value)
    _plan(_axis1d_scene(y=fd.Axis1D(k=1.0, k_max=6.0 / dx)))
    _plan(_axis1d_scene(y=fd.Axis1D(k=1.0, k_max=6.5 / dx)).replace(
        run=fd.Run(steps=5, courant=0.45)))


def test_a_traced_k_past_k_max_makes_the_run_nan_and_says_unstable():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def f(k):
        scene = _axis1d_scene(y=fd.Axis1D(k=k, k_max=3.0), steps=4,
                              initial=fd.InitialField(E=(None, None, lambda x, y, z:
                                                         np.exp(2j * np.pi * x))),
                              monitors=(fd.Record('p', at=(0.1, 0.0, 0.1), component='Ez'),))
        res = fd.run(scene, backend='jax')
        return jnp.sum(jnp.abs(res['p'].values)), res.diagnostics['stable']
    g = jax.jit(f)
    value, stable = g(jnp.asarray(2.0))
    assert np.isfinite(float(value)) and bool(stable)
    value, stable = g(jnp.asarray(3.5))
    assert np.isnan(float(value)) and not bool(stable)
    assert quiet(fd.run, _axis1d_scene()).diagnostics['stable'] is True


def test_a_source_spanning_cells_along_axis1d_is_refused():
    dip = fd.Dipole('d', center=(0.0, 0.0, 0.0), component='Ez', band=band())
    with pytest.raises(ValueError, match='fd.Axis1D axis'):
        _plan(_axis1d_scene(sources=(dip,)))
    scene, _ = conical_grating(periods=1)
    _plan(scene)                                        # a PlaneWave is uniform along it


def test_a_diffraction_order_along_axis1d_is_refused():
    scene, _ = conical_grating(periods=1)
    r, t = scene.monitors
    with pytest.raises(ValueError, match='single order 0') as e:
        _plan(scene.replace(monitors=(r.replace(orders=(1, 1)), t)))
    assert 'orders=(1, 0)' in str(e.value)
    assert _plan(scene.replace(monitors=(r.replace(orders=(1, 0)), t))).ny == 1


def test_a_plane_wave_must_carry_the_axis1d_k():
    scene, info = conical_grating(periods=1)
    wrong = scene.replace(boundaries=fd.Boundaries(x=scene.boundaries.face('x-'),
                                                   z=scene.boundaries.face('z-'),
                                                   y=fd.Axis1D(k=info['ky'] * 1.1)))
    with pytest.raises(ValueError, match='the Axis1D face on y carries'):
        _plan(wrong)
    with pytest.raises(ValueError, match='transverse axes'):
        fd.Bloch.from_source(scene.sources[0], axis1d='z')
    face = fd.Bloch.from_source(scene.sources[0], axis1d='y')['y']
    assert isinstance(face, fd.Axis1D) and face.k == info['ky'] and face.k_max == abs(face.k)


# ---------------------------------------------------------------------------------------------
# The grazing-order rule (stage 3, Build B)
# ---------------------------------------------------------------------------------------------

def _steep(theta=70.0, cells=12, reflection=1e-8, periods=None, **kwargs):
    """A PlaneWave at ``theta`` in a Bloch cell, the z faces ``PML(cells, reflection)``."""
    scene, _ = bloch_plane_wave(theta=theta, at=0.6, steps=40, pml=cells, **kwargs)
    scene = scene.replace(boundaries=fd.Boundaries(
        x=scene.boundaries.face('x-'), y=scene.boundaries.face('y-'),
        z=fd.PML(cells=cells, reflection=reflection)))
    if periods is not None:
        scene = scene.replace(run=fd.Run(periods=periods))
    return scene


def test_a_grazing_incidence_needs_24_cells_a_low_target_and_160_periods():
    """cos(70 deg) = 0.342: under the default face's reach 0.41, over 1e-12's 0.27."""
    with pytest.raises(ValueError, match='Grazing order refused') as e:
        _plan(_steep())
    message = str(e.value)
    for quoted in ('exit cos theta = 0.342', 'cos_min = 0.408', 'R_c = ',
                   'PML(cells>=24', 'Run(periods>=160)', '20 cpw 24 cells 1e-16',
                   'continuum R_c', 'fdtd_grazing_return'):
        assert quoted in message, quoted
    with pytest.raises(ValueError, match='Run\\(periods>=160\\)'):
        _plan(_steep(cells=24, reflection=1e-12, periods=40))
    with pytest.raises(ValueError, match='PML\\(cells>=24'):
        _plan(_steep(cells=12, reflection=1e-12, periods=160))
    with pytest.raises(ValueError, match='reflection<='):
        _plan(_steep(cells=24, reflection=1e-8, periods=160))       # 1e-8 reaches only 0.41
    ok = _steep(cells=24, reflection=1e-12, periods=160)
    from wavebit.fd._lower.grazing import survey
    rows = survey(ok, _plan(ok))
    assert rows and all(r['verdict'] == 'ok' for r in rows)
    assert all(abs(r['cos'] - math.cos(math.radians(70.0))) < 1e-12 for r in rows)


def test_below_the_measured_floor_an_order_is_refused_outright():
    with pytest.raises(ValueError, match='refused outright'):
        _plan(_steep(theta=80.0, cells=24, reflection=1e-16, periods=160))
    _plan(_steep(theta=75.0, cells=24, reflection=1e-16, periods=160))   # cos 0.259


def test_a_grazing_diffraction_order_is_refused_through_the_face_it_leaves_by():
    """A 1.05-wide cell at normal incidence: the (+-1, 0) orders leave at cos 0.305 at 1.0."""
    monitors = (fd.Diffraction('r', z=1.0, wavelengths=(1.0,)),)
    with pytest.raises(ValueError, match="order \\(-1, 0\\) leaving through 'z") as e:
        _plan(_steep(theta=0.0, transverse=(1.05, 0.2), monitors=monitors))
    assert 'exit cos theta = 0.305' in str(e.value)
    _plan(_steep(theta=0.0, transverse=(1.05, 0.2), monitors=monitors, cells=24,
                 reflection=1e-12, periods=160))
    # Without a Diffraction monitor the faces are asked nothing about orders ...
    _plan(_steep(theta=0.0, transverse=(1.05, 0.2)))
    # ... and an order the monitor's wavelengths leave evanescent is not asked about either.
    _plan(_steep(theta=0.0, transverse=(1.05, 0.2),
                 monitors=(fd.Diffraction('r', z=1.0, wavelengths=(1.1,)),)))


@pytest.mark.parametrize('wavelength, cos, refused', [(1.1, 0.411, True), (0.9, 0.405, False)])
def test_each_wavelength_is_judged_with_its_own_cfs_shift(wavelength, cos, refused):
    """The CFS shift is fixed at the face's design wavelength (the band centre, 1.0), so the
    law's ``g`` is 0.906 at 1.1 (``cos_min(1e-8)`` 0.414) and 0.931 at 0.9 (0.403). An order at
    cos 0.411 is refused at 1.1 although the design wavelength's 0.408 would pass it (``R_c`` =
    1.07e-3 > tau), and one at cos 0.405 is accepted at 0.9 although 0.408 would refuse it."""
    from wavebit.fd._lower import grazing
    period = wavelength / math.sqrt(1.0 - cos ** 2)      # the (+-1, 0) orders leave at ``cos``
    monitors = (fd.Diffraction('r', z=1.0, wavelengths=(wavelength,)),)
    scene = _steep(theta=0.0, transverse=(period, 0.2), monitors=monitors,
                   cells_per_wavelength=24 / period)
    g = grazing.exponent(wavelength)
    assert abs(g - (0.906 if refused else 0.931)) < 5e-4
    assert (grazing.continuum_return(1e-8, cos, g=g) > grazing.GRAZING_TOLERANCE) == refused
    if refused:
        with pytest.raises(ValueError, match='Grazing order refused') as e:
            _plan(scene)
        assert 'exit cos theta = 0.411' in str(e.value) and 'g = 0.906' in str(e.value)
        assert 'cos_min = 0.414' in str(e.value)
    else:
        rows = grazing.survey(scene, _plan(scene))
        orders = [r for r in rows if r['what'] == 'orders']
        assert orders and all(r['verdict'] == 'ok' for r in orders)
        assert all(abs(r['cos'] - cos) < 1e-9 and abs(r['g'] - g) < 1e-12 for r in orders)
        assert grazing.cos_min(1e-8) > cos             # the design wavelength's law refuses it


def test_the_grazing_law_is_the_gradings_own():
    from wavebit.fd._lower import grazing
    assert abs(grazing.GRAZING_EXPONENT - 0.919) < 5e-4
    assert round(grazing.cos_min(1e-8), 2) == 0.41
    assert round(grazing.cos_min(1e-12), 2) == 0.27
    assert round(grazing.cos_min(1e-16), 3) == 0.204
    assert abs(grazing.continuum_return(1e-8, 0.26) - 1.2e-2) < 1e-3     # the table's R_c
