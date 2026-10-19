"""Gate 5 of stage 2 -- the doors: every refusal the sources own, each beside the accepted spelling.

Not a physics gate. Each test pairs a refusal with the nearest scene that is accepted, so a
refusal is shown to be a decision and not a broken constructor, and asserts on the message's
*content* -- what it names (the rule, the fix, the measurement it cites) -- rather than on its
type alone: a refusal nobody can act on is half a door.

The rows, in the order of the stage-2 design's table (``src/wavebit/fdtd/_prepare/sources.py`` lists
where each lives):

1. a source whose support touches a PEC/PMC wall node, lies inside an absorbing layer, or
   crosses a periodic seam;
2. a TFSF box that touches an absorbing layer or crosses a face; a TFSF whose propagation
   axis is not terminated by PML/Absorber;
3. a structure that crosses a TFSF face and is not part of the aux grid's 1-D background;
4. TFSF with an oblique direction -- refused by name, citing the measurement;
5. PlaneWave on a cell whose transverse faces are not Periodic/Bloch, or theta != 0 with
   Periodic -- naming ``Bloch.from_source``;
6. a Bloch phase inconsistent with the PlaneWave's k_par at the band centre;
7. a complex waveform on a real run;
8. Dipole: the 2x2x2 trilinear stamp, a centre outside its ``range``, a range crossing a wall,
   a traced centre without a range;
9. a PML with no wavelength: inferred from the sources' band centre when they agree,
   refused when they do not;
10. cells per wavelength, checked at ``band.shortest_wavelength``;

and the doors around them: ``Run(periods=)``, ``Region`` as a structure, the stage-7 sources, the
spelling of ``Band``, ``Dipole``, ``PlaneWave`` and ``TFSF`` and the scene's ``sources=``.
"""
import math
import warnings

import pytest

import wavebit.fdtd._legacy as fd
from tests.fixtures.fdtd.fd_helpers import band, bloch_plane_wave, dipole_box, tfsf_box

DX = 0.05


def _plans(scene):
    """``fd.plan`` with the (expected, separately tested) mesh and proximity warnings silenced.

    A substrate running out through the PML is the proximity warning's own case, and it is
    what the TFSF gate's layered scene is; neither warning is what these doors test.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        warnings.simplefilter('ignore', fd.AbsorberProximityWarning)
        return fd.plan(scene)


def _pec_box(sources, size=1.0, **changes):
    return fd.Scene(domain=fd.Domain(size=(size, size, size)),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=10),
                    sources=tuple(sources), **changes)


# ---------------------------------------------------------------------------------------------
# 1. Support against walls, absorbers and seams
# ---------------------------------------------------------------------------------------------

def test_a_dipole_touching_a_pec_wall_node_is_refused_and_one_cell_in_is_accepted():
    """The stamp is 2 cells either side; at 1.5 cells from the wall it writes the wall layer."""
    low = -0.5
    touching = fd.Dipole('d', center=(low + 1.5 * DX, 0.0, 0.0), component='Ey', band=band())
    with pytest.raises(ValueError) as e:
        _plans(_pec_box((touching,)))
    message = str(e.value)
    assert "CurrentDipole('d')" in message and 'PEC wall' in message and "'x-'" in message
    assert 'one cell clear' in message
    clear = touching.replace(center=(low + 3.0 * DX, 0.0, 0.0))
    assert _plans(_pec_box((clear,))).terms[-1].slab.x[0] >= 1


def test_a_dipole_touching_a_pmc_wall_is_refused_too():
    scene = _pec_box((fd.Dipole('d', center=(0.0, 0.0, 0.5 - 1.2 * DX), component='Ex',
                                band=band()),)).replace(
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PMC()))
    with pytest.raises(ValueError, match="PMC wall on 'z\\+'"):
        _plans(scene)
    _plans(scene.replace(sources=(scene.sources[0].replace(center=(0.0, 0.0, 0.3)),)))


def test_a_dipole_inside_an_absorbing_layer_is_refused_and_clear_of_it_is_accepted():
    scene = dipole_box(center=(0.0, 0.0, 0.0), size=0.8)
    plan = _plans(scene)
    inner = plan.grid.low[2] + 12 * plan.dx                  # the z- PML's inner face
    inside = scene.replace(sources=(scene.sources[0].replace(center=(0.0, 0.0, inner)),))
    with pytest.raises(ValueError) as e:
        _plans(inside)
    message = str(e.value)
    assert "reaches into the PML on 'z-'" in message and '12 cells' in message
    assert 'absorbed as it is made' in message
    clear = scene.replace(sources=(scene.sources[0].replace(center=(0.0, 0.0, inner + 0.15)),))
    _plans(clear)


def test_a_dipole_across_a_periodic_seam_is_refused():
    base = bloch_plane_wave()[0]
    seam = base.replace(sources=(fd.Dipole('d', center=(base.domain.size[0] / 2 - 0.01, 0.0, 0.3),
                                           component='Ez', band=band()),))
    with pytest.raises(ValueError, match='periodic seam'):
        _plans(seam)
    wide = seam.replace(domain=fd.Domain(size=(0.6, 0.6, seam.domain.size[2])),
                        sources=(seam.sources[0].replace(center=(0.0, 0.0, 0.3)),))
    _plans(wide)


# ---------------------------------------------------------------------------------------------
# 2-3. TFSF: the box, its faces, its background
# ---------------------------------------------------------------------------------------------

def test_a_tfsf_box_touching_an_absorbing_layer_is_refused_and_a_smaller_one_accepted():
    with pytest.raises(ValueError) as e:
        _plans(tfsf_box(box=(0.5, 0.5, 0.5), margin=0.0))
    message = str(e.value)
    assert "TFSF('t') box with its one-cell correction layer" in message
    assert 'reaches into the PML' in message
    _plans(tfsf_box(box=(0.5, 0.5, 0.5), margin=0.1))


def test_a_tfsf_box_crossing_a_face_of_the_domain_is_refused():
    scene = tfsf_box()
    big = scene.sources[0].replace(box=fd.Region(size=(5.0, 0.5, 0.5)))
    with pytest.raises(ValueError, match='reaches into the PML'):
        _plans(scene.replace(sources=(big,)))
    periodic = scene.replace(boundaries=fd.Boundaries.open(fd.PML(cells=12), x=fd.Periodic()),
                             sources=(big,))
    with pytest.raises(ValueError, match='periodic seam'):
        _plans(periodic)
    _plans(scene.replace(boundaries=periodic.boundaries))       # the box inside: accepted


def test_a_tfsf_whose_propagation_faces_are_not_absorbing_is_refused():
    scene = tfsf_box(direction='-z')
    walls = scene.replace(boundaries=fd.Boundaries.open(fd.PML(cells=12), z=fd.PEC()))
    with pytest.raises(ValueError) as e:
        _plans(walls)
    message = str(e.value)
    assert "TFSF('t') propagates along z" in message and 'PEC' in message
    assert 'fd.PML(...) or fdtd.ConductiveAbsorber(...)' in message
    # the transverse faces may be anything; the propagation faces absorbing
    _plans(scene.replace(boundaries=fd.Boundaries.open(fd.PML(cells=12), x=fd.PEC())))
    _plans(scene.replace(boundaries=fd.Boundaries.open(fd.PML(cells=12),
                                                       z=fd.Absorber(cells=12))))


def test_a_structure_crossing_the_tfsf_surface_outside_the_aux_background_is_refused():
    scene = tfsf_box(direction='-z')
    rod = fd.Structure(fd.Slab(x=(-0.1, 0.1)), fd.Material(eps=2.25))
    with pytest.raises(ValueError) as e:
        _plans(scene.replace(structures=(rod,)))
    message = str(e.value)
    assert 'structures[0]' in message and 'crosses the faces' in message
    # It named "a stage-4 shape" until C5; it names the spellings that exist now.
    assert 'slabs across z' in message and 'a smoothed body' in message
    # a slab across the propagation axis is part of the aux background, box-crossing or not
    _plans(tfsf_box(direction='-z', substrate=(2.25, 0.0)))
    # a slab across x that stays clear of the box and its correction layer is accepted
    far = fd.Structure(fd.Slab(x=(0.5, 0.7)), fd.Material(eps=2.25))
    _plans(scene.replace(structures=(far,)))


# ---------------------------------------------------------------------------------------------
# 4. TFSF oblique: refused by name, with the measurement
# ---------------------------------------------------------------------------------------------

def test_an_oblique_tfsf_is_refused_citing_the_measurement():
    b = band()
    box = fd.Region(size=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError) as e:
        fd.TFSF('t', box=box, direction='-z', polarization='x', band=b, theta=30)
    message = str(e.value)
    assert 'oblique TFSF box is refused in v1' in message
    assert 'studies/fdtd/tfsf_oblique_leak/fdtd_tfsf_oblique_leak.py' in message
    assert '2.2e-3 / 4.2e-4 / 1.0e-4 / 2.6e-5' in message and '3-8e-2' in message
    assert 'fd.PlaneWave' in message and 'fd.Bloch.from_source' in message
    with pytest.raises(ValueError, match='oblique TFSF box is refused'):
        fd.TFSF('t', box=box, direction=(0.5, 0.0, -0.866), polarization='x', band=b)
    with pytest.raises(ValueError, match="'\\+z' or '-z'"):
        fd.TFSF('t', box=box, direction='z', polarization='x', band=b)
    assert fd.TFSF('t', box=box, direction='-z', polarization='x', band=b, theta=0).theta == 0.0


def test_a_tfsf_polarisation_along_its_direction_is_refused():
    with pytest.raises(ValueError, match="'x' or 'y'"):
        fd.TFSF('t', box=fd.Region(size=(0.5, 0.5, 0.5)), direction='-z', polarization='z',
                band=band())


# ---------------------------------------------------------------------------------------------
# 5-6. PlaneWave: the cell must carry the wave
# ---------------------------------------------------------------------------------------------

def test_a_plane_wave_on_walls_is_refused_naming_bloch_from_source():
    scene, pw = bloch_plane_wave()
    walls = scene.replace(boundaries=fd.Boundaries(x=fd.PEC(), y=fd.Periodic(),
                                                   z=fd.PML(cells=12)))
    with pytest.raises(ValueError) as e:
        _plans(walls)
    message = str(e.value)
    assert 'the x faces are PEC' in message and 'fd.Bloch.from_source(pw)' in message
    assert 'fd.TFSF' in message
    _plans(scene)


def test_an_oblique_plane_wave_on_a_periodic_cell_is_refused_naming_bloch_from_source():
    scene, pw = bloch_plane_wave(theta=20)
    periodic = scene.replace(boundaries=fd.Boundaries.periodic(z=fd.PML(cells=12)))
    with pytest.raises(ValueError) as e:
        _plans(periodic)
    message = str(e.value)
    assert 'theta=20' in message and 'Periodic' in message
    assert 'fd.Bloch.from_source' in message
    faces = fd.Bloch.from_source(pw)
    assert isinstance(faces['x'], fd.Bloch) and isinstance(faces['y'], fd.Periodic)
    assert math.isclose(faces['x'].k, 2 * math.pi * math.sin(math.radians(20)), rel_tol=1e-15)
    _plans(scene)


def test_bloch_from_source_at_normal_incidence_is_periodic():
    _, pw = bloch_plane_wave(theta=0.0)
    faces = fd.Bloch.from_source(pw)
    assert all(isinstance(f, fd.Periodic) for f in faces.values()) and set(faces) == {'x', 'y'}
    with pytest.raises(TypeError, match='fd.PlaneWave'):
        fd.Bloch.from_source(fd.Dipole('d', center=(0, 0, 0), band=band()))


def test_an_inconsistent_bloch_phase_is_refused():
    scene, pw = bloch_plane_wave(theta=20)
    k = fd.Bloch.from_source(pw)['x'].k
    wrong = scene.replace(boundaries=fd.Boundaries(x=fd.Bloch(k=1.1 * k), y=fd.Periodic(),
                                                   z=fd.PML(cells=12)))
    with pytest.raises(ValueError) as e:
        _plans(wrong)
    message = str(e.value)
    assert 'carries k =' in message and 'transverse wavevector at the band centre' in message
    assert 'fd.Bloch.from_source(pw, background=...)' in message
    # the same k given as a phase across the snapped period is the same cell: accepted
    L = _plans(scene).grid.size[0]
    _plans(scene.replace(boundaries=fd.Boundaries(x=fd.Bloch(phase=k * L), y=fd.Periodic(),
                                                  z=fd.PML(cells=12))))
    # a Bloch face along an axis the wave has no k on is inconsistent too
    extra = scene.replace(boundaries=fd.Boundaries(x=fd.Bloch(k=k), y=fd.Bloch(phase=0.3),
                                                   z=fd.PML(cells=12)))
    with pytest.raises(ValueError, match='no transverse wavevector along y'):
        _plans(extra)


def test_the_bloch_phase_is_taken_in_the_background():
    """In glass the transverse k is n times the vacuum one; the vacuum phase is refused."""
    glass = fd.Material(eps=2.25)
    scene, pw = bloch_plane_wave(theta=20, background=glass, cells_per_wavelength=30)
    assert math.isclose(fd.Bloch.from_source(pw, background=glass)['x'].k,
                        1.5 * fd.Bloch.from_source(pw)['x'].k, rel_tol=1e-15)
    _plans(scene)
    vacuum_phase = scene.replace(boundaries=fd.Boundaries(**fd.Bloch.from_source(pw),
                                                          z=fd.PML(cells=12)))
    with pytest.raises(ValueError, match='n = 1.5'):
        _plans(vacuum_phase)


def test_a_plane_wave_in_a_structure_or_with_a_patterned_layer_is_refused():
    glass = fd.Material(eps=2.25)
    scene, _ = bloch_plane_wave(cells_per_wavelength=30,
                                structures=(fd.Structure(fd.Slab(z=(-0.3, 0.3)), glass),))
    with pytest.raises(ValueError, match='inside structures\\[0\\]'):
        _plans(scene)
    _plans(scene.replace(structures=(fd.Structure(fd.Slab(z=(-0.6, -0.1)), glass),)))
    rod = scene.replace(structures=(fd.Structure(fd.Slab(x=(-0.05, 0.05)), glass),))
    with pytest.raises(ValueError, match='smoothed fd.Sphere, fd.Box'):  # "stage 4" before C5
        _plans(rod)


# ---------------------------------------------------------------------------------------------
# 7. A complex waveform on a real run
# ---------------------------------------------------------------------------------------------

def test_a_complex_waveform_on_a_real_run_is_refused_and_on_a_bloch_run_accepted():
    scene = dipole_box(size=0.6, steps=5, amplitude=1.0 + 0.5j)
    with pytest.raises(ValueError) as e:
        fd.run(scene)
    message = str(e.value)
    assert 'complex waveform on a real run is refused' in message and 'phase=' in message
    fd.run(dipole_box(size=0.6, steps=5, amplitude=1.0, phase=0.46))    # the real spelling
    bloch, _ = bloch_plane_wave(theta=20, steps=5, amplitude=1.0 + 0.5j)
    fd.run(bloch)                                                    # a complex run takes it
    with pytest.raises(ValueError, match='only the amplitude may be complex'):
        dipole_box(size=0.6, steps=5, phase=0.1 + 0.2j)


# ---------------------------------------------------------------------------------------------
# 8. Dipole: stamp, range, traced centre, spelling
# ---------------------------------------------------------------------------------------------

def test_the_trilinear_stamp_is_refused_and_the_two_smooth_ones_accepted():
    with pytest.raises(ValueError) as e:
        fd.Dipole('d', center=(0, 0, 0), band=band(), stamp='trilinear')
    message = str(e.value)
    assert '2x2x2 cloud-in-cell' in message and "stamp='bspline'" in message
    for stamp in ('bspline', 'tent'):
        assert fd.Dipole('d', center=(0, 0, 0), band=band(), stamp=stamp).stamp == stamp
    assert fd.Dipole('d', center=(0, 0, 0), band=band()).stamp == 'bspline'
    with pytest.raises(ValueError, match="one of \\('bspline', 'tent'\\)"):
        fd.Dipole('d', center=(0, 0, 0), band=band(), stamp='gauss')


def test_a_centre_outside_its_range_is_refused():
    rng = ((-0.1, 0.1), (0.0, 0.0), (0.0, 0.0))
    with pytest.raises(ValueError) as e:
        fd.Dipole('d', center=(0.2, 0.0, 0.0), band=band(), range=rng)
    message = str(e.value)
    assert 'x = 0.2 lies outside the declared range (-0.1, 0.1)' in message
    fd.Dipole('d', center=(0.05, 0.0, 0.0), band=band(), range=rng)
    with pytest.raises(ValueError, match='lo > hi'):
        fd.Dipole('d', center=(0, 0, 0), band=band(), range=((0.1, -0.1), (0, 0), (0, 0)))


def test_a_range_crossing_a_wall_is_refused():
    scene = dipole_box(size=0.8)
    plan = _plans(scene)
    inner = plan.grid.high[0] - 12 * plan.dx
    reaching = scene.sources[0].replace(range=((-0.1, inner), (0.0, 0.0), (0.0, 0.0)))
    with pytest.raises(ValueError) as e:
        _plans(scene.replace(sources=(reaching,)))
    message = str(e.value)
    assert "CurrentDipole('d') over its range" in message and "reaches into the PML on 'x+'" in message
    fine = scene.sources[0].replace(range=((-0.1, 0.1), (0.0, 0.0), (0.0, 0.0)))
    _plans(scene.replace(sources=(fine,)))


def test_a_traced_centre_needs_a_range():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def build(x, rng):
        return fd.Dipole('d', center=(x, 0.0, 0.0), band=band(), range=rng)

    with pytest.raises(ValueError) as e:
        jax.grad(lambda x: build(x, None).center[0])(jnp.asarray(0.0))
    assert 'range=((x0, x1), (y0, y1), (z0, z1))' in str(e.value)
    rng = ((-0.1, 0.1), (0.0, 0.0), (0.0, 0.0))
    assert float(jax.grad(lambda x: build(x, rng).center[0] * 2.0)(jnp.asarray(0.0))) == 2.0


def test_a_dipoles_kind_and_component_must_agree():
    with pytest.raises(ValueError) as e:
        fd.Dipole('d', center=(0, 0, 0), component='Hy', band=band())
    message = str(e.value)
    assert "component='Ey'" in message and "kind='magnetic'" in message
    fd.Dipole('d', center=(0, 0, 0), component='Hy', kind='magnetic', band=band())
    with pytest.raises(ValueError, match="'electric'"):
        fd.Dipole('d', center=(0, 0, 0), band=band(), kind='current')


# ---------------------------------------------------------------------------------------------
# 9. The PML wavelength, inferred from the band centre
# ---------------------------------------------------------------------------------------------

def test_a_pml_without_a_wavelength_takes_the_sources_band_centre():
    scene = dipole_box(size=0.6, src_band=band(wavelength=0.8), steps=5)
    assert all(w == 0.8 for _, w in _plans(scene).pml_wavelengths)
    # a PML's own wavelength still wins
    own = scene.replace(boundaries=fd.Boundaries.open(fd.PML(cells=12, wavelength=0.9)))
    assert all(w == 0.9 for _, w in _plans(own).pml_wavelengths)


def test_a_pml_without_a_wavelength_and_disagreeing_bands_is_refused():
    scene = dipole_box(size=0.6, steps=5)
    other = fd.Dipole('e', center=(0.1, 0.1, 0.1), component='Ex',
                      band=band(wavelength=0.9))
    with pytest.raises(ValueError) as e:
        _plans(scene.replace(sources=scene.sources + (other,)))
    message = str(e.value)
    assert 'do not share one concrete band centre (1.0, 0.9)' in message
    assert 'fd.PML(cells=..., wavelength=...)' in message
    agreeing = other.replace(band=band())
    _plans(scene.replace(sources=scene.sources + (agreeing,)))
    no_source = scene.replace(sources=())
    with pytest.raises(ValueError, match='no source to infer one from'):
        _plans(no_source)


# ---------------------------------------------------------------------------------------------
# 10. Cells per wavelength at the band's short edge
# ---------------------------------------------------------------------------------------------

def test_cells_per_wavelength_are_checked_at_the_shortest_wavelength():
    """20 cells at the centre is 12.8 at the short edge (fine); 12 is 7.7 (refused)."""
    wide = band()                                      # shortest 0.642
    assert math.isclose(wide.shortest_wavelength, 0.6420, rel_tol=1e-3)
    with pytest.raises(ValueError) as e:
        fd.plan(dipole_box(size=1.0, cells_per_wavelength=12, steps=5))
    message = str(e.value)
    assert "the sources' band.shortest_wavelength" in message and '7.70 cells' in message
    with warnings.catch_warnings():
        warnings.simplefilter('error', fd.CoarseMeshWarning)
        fd.plan(dipole_box(size=1.0, cells_per_wavelength=20, steps=5))
    with pytest.warns(fd.CoarseMeshWarning, match='cells per wavelength'):
        fd.plan(dipole_box(size=1.0, cells_per_wavelength=16, steps=5))   # 10.3 at the edge
    # a narrower band at the same mesh moves the edge out and passes
    narrow = band(df=0.05)                             # shortest 0.843: 10.1 cells, warns
    with pytest.warns(fd.CoarseMeshWarning, match='10.12 cells'):
        fd.plan(dipole_box(size=1.0, cells_per_wavelength=12, steps=5, src_band=narrow))


# ---------------------------------------------------------------------------------------------
# The doors around them
# ---------------------------------------------------------------------------------------------

def test_run_periods_doors():
    assert fd.Run(periods=12).steps is None
    with pytest.raises(ValueError, match='one or the other'):
        fd.Run(steps=10, periods=12)
    with pytest.raises(ValueError, match='positive number of optical periods'):
        fd.Run(periods=-1)
    with pytest.raises(ValueError, match="Run\\(steps=...\\) or Run\\(periods=...\\)"):
        fd.Run()
    scene = dipole_box(size=0.6, periods=4)
    with pytest.raises(ValueError, match='no source'):
        _plans(scene.replace(sources=(), boundaries=fd.Boundaries.open(
            fd.PML(cells=12, wavelength=1.0))))
    other = fd.Dipole('e', center=(0.1, 0.1, 0.1), band=band(wavelength=0.9))
    with pytest.raises(ValueError, match='band centres disagree'):
        _plans(scene.replace(sources=scene.sources + (other,),
                             boundaries=fd.Boundaries.open(fd.PML(cells=12, wavelength=1.0))))


def test_a_box_is_a_tfsf_region_and_since_stage_3_a_structure():
    # Stage 2 refused Structure(fd.Region(...)) naming stage 4; stage 3 made the cell-snapped Region
    # a structure (tests/fdtd/test_fd_monitors_doors.py holds its doors), and the shapes after it
    # are still stage 4.
    box = fd.Region(size=(0.2, 0.2, 0.2))
    assert fd.Structure(box, fd.Material(eps=2.25)).shape is box
    with pytest.raises(TypeError) as e:
        fd.Structure(fd.Plane(z=0.0), fd.Material(eps=2.25))
    assert 'stage 4' in str(e.value)
    with pytest.raises(ValueError, match='positive length'):
        fd.Region(size=(0.2, 0.0, 0.2))
    with pytest.raises(TypeError, match="fdtd.SnappedBox"):
        fd.TFSF('t', box=(0.2, 0.2, 0.2), direction='-z', band=band())


def test_the_stage_7_sources_are_refused_by_name():
    for name in ('GaussianBeam', 'ModeSource'):
        with pytest.raises(NotImplementedError, match='stage 7'):
            getattr(fd, name)(wavelength=1.0)


def test_a_band_is_required_and_spelled_in_frequency():
    with pytest.raises(TypeError, match='fd.Band\\(wavelength=..., df=...\\) is required'):
        fd.Dipole('d', center=(0, 0, 0))
    with pytest.raises(ValueError, match='reaches zero frequency'):
        fd.Band(wavelength=1.0, df=0.3)                # 3.72 * 0.3 > 1
    with pytest.raises(ValueError, match='a fraction in \\(0, 1\\)'):
        fd.Band(wavelength=1.0, df=0.1, cutoff=0.0)
    with pytest.raises(ValueError, match='positive'):
        fd.Band(wavelength=-1.0, df=0.1)
    fd.Band(wavelength=1.0, df=0.2)


def test_the_scene_takes_a_tuple_of_named_sources():
    b = band()
    one = fd.Dipole('d', center=(0, 0, 0), band=b)
    scene = dipole_box(size=0.6, steps=5)
    assert scene.replace(sources=one).sources == (one,)
    with pytest.raises(TypeError, match='fdtd.CurrentDipole, fd.PlaneWave or fd.TFSF'):
        scene.replace(sources=(fd.Record('r', at=(0, 0, 0)),))
    with pytest.raises(ValueError, match="repeated: \\['d'\\]"):
        scene.replace(sources=(one, one.replace(center=(0.1, 0, 0))))
    with pytest.raises(NotImplementedError, match='sources are a tuple'):
        fd.Scene(source=one, domain=scene.domain, boundaries=scene.boundaries,
                 resolution=scene.resolution, run=scene.run)


def test_a_plane_waves_position_is_named_on_its_own_axis():
    b = band()
    with pytest.raises(ValueError, match='give its position as z=...'):
        fd.PlaneWave('pw', x=0.0, direction='-z', band=b)
    with pytest.raises(ValueError, match="'s' \\(E out of"):
        fd.PlaneWave('pw', z=0.0, direction='-z', band=b, polarization='te')
    with pytest.raises(ValueError, match='\\[0, 90\\)'):
        fd.PlaneWave('pw', z=0.0, direction='-z', band=b, theta=95)
    pw = fd.PlaneWave('pw', z=0.0, direction='-z', band=b, theta=20, polarization='p')
    k, e, h = pw.vectors()
    assert abs(sum(a * c for a, c in zip(k, e))) < 1e-15            # transverse
    cross = (e[1] * h[2] - e[2] * h[1], e[2] * h[0] - e[0] * h[2], e[0] * h[1] - e[1] * h[0])
    assert all(abs(a - c) < 1e-15 for a, c in zip(cross, k))        # E x H along k
