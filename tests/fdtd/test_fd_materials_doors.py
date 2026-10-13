"""Stage 1's doors: every refusal it owns, each paired with the form that is accepted instead.

A refusal is only half a statement. "``Material(4+0.1j)`` raises" says nothing about whether
the door is *usable*, and a door that refuses everything passes that half of every test --
so each case below is a pair: the spelling that is refused, and beside it the spelling that
means what the refused one was reaching for, asserted to work. The pairs are the rows of the
stage-1 design's refusal table, plus the ones the implementation added.

    the door                                       refused                accepted
    PML depth                                      cells < 12             allow_thin=True, or 12
    PML frequency                                  no wavelength          wavelength=, Resolution(wavelength=)
    a lossy medium in a PML layer                  Conductor/Dispersive   fd.Absorber on that face
    a structure cutting an absorbing layer         an interface inside    running through, or clear of it
    mesh vs the material                           < 8 cells/wavelength   >= 12 (8..12 warns)
    a structure near an absorber                   (warns)                half a wavelength of clearance
    a complex permittivity                         Material(4+0.1j)       Conductor, Dispersive, eps_at
    a shape that is not a layer                    two axes, radius=...   Slab with one axis
    the pole CFL                                   dt past the bound      a lower courant
    a dispersive structure overlapping another     shared cells           disjoint slabs, or one of them
    a keyword nobody reads                         Absorber(background=)  PML(background=)

Every message is asserted on its *content*, not only its type: a refusal that does not name
the accepted spelling is a refusal the reader has to guess their way out of, and this file is
where that is enforced.
"""
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from tests.fixtures.fdtd.fd_helpers import PROBE


def scene(**changes):
    """A small open box with a wavelength: the scene every door below is opened on."""
    base = dict(
        domain=fd.Domain(size=(1.0, 1.0, 2.0)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                 z=fd.PML(cells=12, wavelength=1.0)),
        resolution=fd.Resolution(cells_per_wavelength=20, wavelength=1.0),
        run=fd.Run(steps=4),
        initial=fd.InitialField(E=np.zeros((3, 40, 20, 20))),
        monitors=(fd.Record('p', at=PROBE, component='Ey'),))
    base.update(changes)
    if 'initial' not in changes:
        cells = fd.plan(fd.Scene(**{**base, 'initial': None, 'structures': ()})).shape
        base['initial'] = fd.InitialField(E=np.zeros(cells))
    return fd.Scene(**base)


def quiet(fn, *args, **kwargs):
    """Call ``fn`` with stage-1 warnings silenced; the warning doors test them on purpose."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        warnings.simplefilter('ignore', fd.AbsorberProximityWarning)
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------------------------
# The PML's own two doors
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('cells', [1, 4, 6, 11])
def test_a_pml_thinner_than_twelve_cells_is_refused_and_allow_thin_takes_it(cells):
    with pytest.raises(ValueError, match='allow_thin=True') as e:
        fd.PML(cells=cells, wavelength=1.0)
    assert 'tests/fdtd/test_fd_pml.py' in str(e.value), 'the refusal cites the study behind it'
    assert fd.PML(cells=cells, wavelength=1.0, allow_thin=True).cells == cells
    assert fd.PML(cells=12, wavelength=1.0).cells == 12


def test_a_pml_with_no_wavelength_is_refused_naming_both_doors_that_supply_one():
    bare = fd.PML(cells=12)
    assert bare.wavelength is None, 'the refusal is at the scene, where Resolution is visible'
    with pytest.raises(ValueError, match='alpha_max') as e:
        quiet(fd.plan, scene(boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=bare),
                             resolution=fd.Resolution(dx=0.05)))
    assert 'fd.PML(cells=..., wavelength=...)' in str(e.value)
    assert 'fd.Resolution(' in str(e.value) and 'wavelength=' in str(e.value)
    # accepted, both ways
    assert quiet(fd.plan, scene()).pml_wavelengths == (('z-', 1.0), ('z+', 1.0))
    from_face = scene(resolution=fd.Resolution(dx=0.05),
                      boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                               z=fd.PML(cells=12, wavelength=0.8)))
    assert quiet(fd.plan, from_face).pml_wavelengths == (('z-', 0.8), ('z+', 0.8))


def test_a_pml_reflection_target_is_an_amplitude_ratio():
    for bad in (0.0, 1.0, 2.0, -1e-8):
        with pytest.raises(ValueError, match='amplitude ratio'):
            fd.PML(cells=12, wavelength=1.0, reflection=bad)
    assert fd.PML(cells=12, wavelength=1.0, reflection=1e-12).reflection == 1e-12


def test_absorbing_layers_that_do_not_fit_their_axis_are_refused():
    thin = fd.Domain(size=(1.0, 1.0, 1.0))      # 20 cells along z, two 12-cell layers
    with pytest.raises(ValueError, match='the two layers would meet') as e:
        quiet(fd.plan, scene(domain=thin, initial=None))
    assert 'give the axis more cells' in str(e.value).lower()
    assert quiet(fd.plan, scene()).cells == 40 * 20 * 20


# ---------------------------------------------------------------------------------------------
# What may sit inside a PML layer
# ---------------------------------------------------------------------------------------------

LOSSY = [pytest.param(fd.Conductor(eps=2.25, sigma=0.01), id='Conductor'),
         pytest.param(fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=4.0, gamma=1.0),)),
                      id='Dispersive')]


@pytest.mark.parametrize('material', LOSSY)
def test_a_lossy_structure_reaching_a_pml_face_is_refused_naming_absorber(material):
    """And the same structure on an ``Absorber`` face runs: that is what the fallback is for."""
    through = fd.Structure(fd.Slab(z=(-1.0, 0.0)), material)       # reaches the z- layer
    with pytest.raises(ValueError, match='fd.Absorber') as e:
        quiet(fd.plan, scene(structures=(through,)))
    assert 'matched' in str(e.value) and type(material).__name__ in str(e.value)
    absorbing = fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Absorber(cells=12))
    plan = quiet(fd.plan, scene(structures=(through,), boundaries=absorbing))
    assert plan.absorber_cells == (('z-', 12), ('z+', 12)) and plan.pml_cells == ()
    # or keep the lossy material clear of the layer
    inside = fd.Structure(fd.Slab(z=(-0.2, 0.2)), material)
    assert quiet(fd.plan, scene(structures=(inside,))).pml_cells == (('z-', 12), ('z+', 12))


@pytest.mark.parametrize('face, kind', [
    (fd.PML(cells=12, wavelength=1.0), 'PML'),
    (fd.Absorber(cells=12), 'Absorber'),
])
def test_a_structure_that_ends_inside_an_absorbing_layer_is_refused(face, kind):
    """Both absorbing faces refuse it, and for the reason they share.

    An interface inside a layer is a mirror the layer cannot undo: the reflection leaves the
    way it came, so only the cells in *front* of the interface are in its way. (Inside a PML
    the stretched coordinate is complex there as well, so nothing matches it at all -- but
    that is the second reason and not the first.) Measured in ``tests/fdtd/test_fd_absorber.py``
    with the door held open: 1.9e-1 against 6.0e-4 on a PML face and 2.4e-1 against 1.2e-2 on
    an Absorber face, 317x and 20x. The ``Absorber`` half is not in the stage-1 design; it is
    here because the measurement says the geometry is as bad there.
    """
    glass = fd.Material(eps=2.25)
    cutting = fd.Structure(fd.Slab(z=(-1.0, -0.8)), glass)          # ends 4 cells into z-
    boundaries = fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=face)
    with pytest.raises(ValueError, match=f'ends inside the {kind}') as e:
        quiet(fd.plan, scene(structures=(cutting,), boundaries=boundaries))
    assert 'run the structure all the way through' in str(e.value)
    assert 'a mirror the layer cannot undo' in str(e.value)
    # accepted: all the way through, or clear of the layer
    for span in ((-1.0, 0.0), (-0.4, 0.4)):
        assert quiet(fd.plan, scene(structures=(fd.Structure(fd.Slab(z=span), glass),),
                                    boundaries=boundaries)).spans


def test_a_slab_across_the_absorbing_axis_runs_through_a_transverse_pml_and_is_allowed():
    """The canonical stage-1 scene: a dielectric layer in an open box.

    The stage-1 design's table refuses any structure reaching a PML face that is not the
    background. Taken literally that refuses this -- a z slab reaches the x and y layers of
    every open box -- so the rule as implemented is about the *absorbing axis*: uniform along
    it, which a transverse slab is. The layer is still matched to the background, so the part
    of it that is not the background reflects more than the table says, which is what
    ``AbsorberProximityWarning`` is for.
    """
    everywhere = fd.Boundaries.open(fd.PML(cells=12, wavelength=1.0))
    glass = fd.Structure(fd.Slab(z=(-0.1, 0.1)), fd.Material(eps=2.25))
    plan = quiet(fd.plan, scene(boundaries=everywhere, structures=(glass,),
                                domain=fd.Domain(size=(2.0, 2.0, 2.0)), initial=None))
    assert len(plan.pml_cells) == 6 and plan.spans[0][0] == 'z'


# ---------------------------------------------------------------------------------------------
# The mesh against the material it has to resolve
# ---------------------------------------------------------------------------------------------

def test_too_few_cells_per_wavelength_in_a_material_is_refused_and_says_the_dx_that_is_not():
    coarse = fd.Structure(fd.Slab(z=(-0.2, 0.2)), fd.Material(eps=16.0))   # 20/4 = 5 per wl
    with pytest.raises(ValueError, match='cells per wavelength') as e:
        quiet(fd.plan, scene(structures=(coarse,)))
    message = str(e.value)
    assert 'Refine dx to at most' in message and 'index 4.000' in message
    finer = scene(structures=(coarse,),
                  resolution=fd.Resolution(cells_per_wavelength=80, wavelength=1.0))
    assert quiet(fd.plan, finer).dx == 1.0 / 80


def test_between_eight_and_twelve_cells_per_wavelength_warns_and_above_is_quiet():
    warned = fd.Structure(fd.Slab(z=(-0.2, 0.2)), fd.Material(eps=4.0))    # 20/2 = 10 per wl
    with pytest.warns(fd.CoarseMeshWarning, match='10.00 cells per wavelength'):
        fd.plan(scene(structures=(warned,)))
    fine = scene(structures=(warned,),
                 resolution=fd.Resolution(cells_per_wavelength=40, wavelength=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter('error', fd.CoarseMeshWarning)
        assert fd.plan(fine).dx == 1.0 / 40


def test_a_metal_is_checked_by_its_skin_depth_and_not_by_a_wavelength():
    """``Re(eps) < 0`` has no wavelength in it; the field decays instead of oscillating.

    A Drude metal well below its plasma frequency has ``lambda/n`` imaginary, so the
    cells-per-wavelength rule is meaningless there and would wave any mesh through. The
    check becomes cells per *skin depth*, refused below one -- at which point the grid
    cannot represent the surface at all and what it returns is the grid's reflectance.
    """
    metal = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=30.0, gamma=1.0),))
    assert fd.eps_at(metal.poles, 1.0, 1.0).real < 0, 'the premise: this is below the plasma'
    thick = fd.Structure(fd.Slab(z=(-0.2, 0.2)), metal)
    with pytest.raises(ValueError, match='skin depth') as e:
        quiet(fd.plan, scene(structures=(thick,)))
    assert 'cells' in str(e.value) and 'Refine dx below' in str(e.value)
    # It advised "fd.PEC-backed geometry" until C5; there is no PEC body to back.
    assert 'PEC-backed' not in str(e.value) and 'not a material' in str(e.value)
    fine = scene(structures=(thick,), initial=None,
                 resolution=fd.Resolution(cells_per_wavelength=40, wavelength=1.0))
    with pytest.warns(fd.CoarseMeshWarning, match='cells per skin depth'):
        fd.plan(fine)


def test_a_structure_close_to_an_absorbing_face_warns_and_half_a_wavelength_is_quiet():
    glass = fd.Material(eps=2.25)                       # half a wavelength in it is 1/3
    near = fd.Structure(fd.Slab(z=(-0.39, -0.29)), glass)     # 0.01 from the z- layer's face
    with pytest.warns(fd.AbsorberProximityWarning, match='half a wavelength in its own medium'):
        fd.plan(scene(structures=(near,)))
    far = fd.Structure(fd.Slab(z=(-0.05, 0.05)), glass)
    with warnings.catch_warnings():
        warnings.simplefilter('error', fd.AbsorberProximityWarning)
        assert fd.plan(scene(structures=(far,))).spans


# ---------------------------------------------------------------------------------------------
# The material doors
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('door, kwargs', [
    (fd.Material, dict(eps=4 + 0.1j)),
    (fd.Material, dict(eps=(2.0, 2.0, 1 + 1j))),
    (fd.Material, dict(eps=2.0, mu=1 - 0.5j)),
    (fd.Conductor, dict(eps=4 + 0.1j, sigma=0.1)),
    (fd.Dispersive, dict(eps_inf=4 + 0.1j, poles=(fd.Drude(omega_p=1.0),))),
])
def test_a_complex_permittivity_is_refused_naming_conductor_and_dispersive(door, kwargs):
    with pytest.raises(ValueError, match='complex permittivity is refused') as e:
        door(**kwargs)
    message = str(e.value)
    assert 'fd.Conductor(' in message and 'fd.Dispersive(' in message
    assert 'fd.eps_at(' in message, 'and the way back to one complex number'


def test_the_accepted_forms_reproduce_the_complex_number_that_was_refused():
    """``eps_at`` is the bridge: a pole set at one wavelength IS one complex permittivity."""
    wavelength = 0.55
    omega = 2 * math.pi / wavelength
    # a conductor: eps + i sigma/w, the flat-loss spelling
    conductor = fd.Conductor(eps=2.25, sigma=0.3)
    assert complex(conductor.eps[0]) + 1j * conductor.sigma[0] / omega == pytest.approx(
        2.25 + 1j * 0.3 / omega)
    # a pole set: the closed form, and loss is Im(eps) > 0 in this package's convention
    for pole in (fd.Drude(omega_p=9.0, gamma=0.5), fd.Lorentz(omega_0=12.0, gamma=1.0),
                 fd.Debye(tau=0.2, delta_eps=1.5)):
        eps = fd.eps_at((pole,), wavelength, 1.0)
        assert eps.imag > 0, (type(pole).__name__, eps)
    # the sum is the sum, and a Dispersive material answers for itself
    metal = fd.Dispersive(eps_inf=2.0, poles=(fd.Drude(omega_p=9.0, gamma=0.5),
                                              fd.Lorentz(omega_0=12.0, gamma=1.0, strength=5.0)))
    by_hand = 2.0 + sum(p.eps_term(omega) for p in metal.poles)
    assert fd.eps_at(metal, wavelength) == pytest.approx(by_hand)
    assert fd.eps_at(metal.poles, wavelength, 2.0) == pytest.approx(by_hand)


def discrete_eps_term(pole, omega, dt):
    """``eps_term(omega)`` as the kernel's recurrence actually delivers it, at time step ``dt``.

    ``P^{m+1} = step_gain P^m - hold P^{m-1} + drive E^m`` with ``E^m = E z^m``,
    ``z = exp(-i w dt)``, gives ``P/E = drive / (z - step_gain + hold/z)``; on top of that sits
    whatever the pole reports as :meth:`~wavebit.materials.time_domain.Pole.instantaneous`, which is the
    part of its response that is not in the recurrence at all (Debye's, and nothing else's).
    So this is the whole of what the discretisation answers, and the closed form it is
    compared against is :meth:`eps_term`.
    """
    step_gain, hold, drive = pole.coefficients(dt)
    z = complex(math.cos(omega * dt), -math.sin(omega * dt))
    instantaneous = pole.instantaneous(dt)
    return drive / (z - step_gain + hold / z) + (0.0 if instantaneous is None
                                                 else instantaneous)


@pytest.mark.parametrize('pole', [
    pytest.param(fd.Lorentz(omega_0=12.0, gamma=1.0, strength=60.0), id='Lorentz'),
    pytest.param(fd.Drude(omega_p=4.0, gamma=1.0), id='Drude'),
    pytest.param(fd.Debye(tau=0.05, delta_eps=1.0), id='Debye'),
    pytest.param(fd.Debye(tau=0.005, delta_eps=1.0), id='Debye, nearly lossless'),
])
def test_every_pole_discretises_at_second_order(pole):
    """Halve ``dt`` and the error in ``eps(w)`` falls by four: all three poles, not two.

    **This is the assertion a tolerance used to stand in for.** Debye's recurrence comes from
    the trapezoidal rule on ``tau P' + P = delta_eps E``, whose right-hand side is
    ``delta_eps (E^{n+1} + E^n)/2``; writing ``delta_eps E^n`` there instead -- the explicit
    left rectangle -- is first order, and its error is almost pure **loss**. Measured at the
    resolution this package recommends (20 cells per wavelength, courant 0.5) the explicit form
    gives ``Im(eps) = 0.3265`` against a true 0.2860 for ``tau = 0.05``, and 0.0765 against
    0.0314 for ``tau = 0.005`` -- 2.44x the absorption a nearly lossless relaxation asked for.
    A convergence assertion sees that where a fixed ``rel=`` at one ``dt`` does not, which is
    why the rate and not the value is what is pinned.

    Measured errors at 20 / 40 / 80 / 160 cells per wavelength with the shipped coefficients:
    Lorentz 1.55e-4 -> 2.43e-6, Drude 2.84e-4 -> 4.44e-6, Debye 9.81e-5 -> 1.53e-6, every ratio
    4.00. With the explicit drive, Debye runs 2.17e-2 -> 2.71e-3 at ratio 2.00.
    """
    omega = 2 * math.pi
    exact = pole.eps_term(omega)
    errors = []
    for cells_per_wavelength in (20, 40, 80, 160):
        dt = 0.5 / (cells_per_wavelength * math.sqrt(3.0))
        errors.append(abs(discrete_eps_term(pole, omega, dt) - exact))
    for coarse, fine in zip(errors, errors[1:]):
        assert 3.6 < coarse / fine < 4.4, (pole, errors)


def test_a_debye_pole_is_first_order_and_not_the_zero_frequency_lorentz():
    """The design says Debye is the ``omega_0 = 0`` second-order recurrence. It is not.

    ``omega_0 = 0`` is Drude, whose response diverges as ``1/w`` at low frequency where
    Debye's goes to ``delta_eps``. The class derives the first-order ODE's own form instead,
    carried in the same three coefficients with ``hold = 0`` plus an instantaneous part; this
    pins that the two are different materials and that Debye really does use the degenerate
    recurrence.
    """
    dt = 0.002
    debye = fd.Debye(tau=0.2, delta_eps=1.0)
    drude = fd.Drude(omega_p=math.sqrt(1.0), gamma=1.0 / 0.2)
    low = 2 * math.pi / 1e5
    assert debye.eps_term(low) == pytest.approx(1.0, abs=1e-4)
    assert abs(drude.eps_term(low)) > 1e3, 'the omega_0 = 0 second-order pole diverges'
    step_gain, hold, drive = debye.coefficients(dt)
    assert hold == 0.0
    assert step_gain == pytest.approx((2 * 0.2 - dt) / (2 * 0.2 + dt))
    # The auxiliary variable's drive, c (1 + step_gain), and the instantaneous c beside it --
    # the split that keeps the recurrence explicit in E and still second order. Together they
    # are the trapezoidal rule: c(1+g)/(z-g) + c = c(1+z)/(z-g).
    c = 1.0 * dt / (2 * 0.2 + dt)
    assert drive == pytest.approx(c * (1 + step_gain))
    assert debye.instantaneous(dt) == pytest.approx(c)
    assert drude.instantaneous(dt) is None and fd.Lorentz(omega_0=1.0).instantaneous(dt) is None
    # The residual is second order in ``w dt`` and nothing else: 1e-5 of |eps| at w dt = 0.013.
    # The *rate* is what test_every_pole_discretises_at_second_order pins.
    omega = 2 * math.pi
    assert discrete_eps_term(debye, omega, dt) == pytest.approx(debye.eps_term(omega), rel=1e-4)


@pytest.mark.parametrize('kwargs, names', [
    (dict(x=(0.0, 1.0), z=(0.0, 1.0)), 'stage 4'),
    (dict(x=(0.0, 1.0), y=(0.0, 1.0), z=(0.0, 1.0)), 'stage 4'),
])
def test_a_slab_naming_more_than_one_axis_is_refused_naming_stage_4(kwargs, names):
    with pytest.raises(NotImplementedError, match=names) as e:
        fd.Slab(**kwargs)
    assert 'exactly ONE axis' in str(e.value)
    assert fd.Slab(z=(0.0, 1.0)).axis == 'z'


@pytest.mark.parametrize('keyword', ['radius', 'center', 'size', 'vertices', 'raster', 'angle'])
def test_every_other_shape_is_refused_by_name(keyword):
    with pytest.raises(NotImplementedError, match='stage 4') as e:
        fd.Slab(**{keyword: 1.0})
    assert 'Slab(x=(lo, hi))' in str(e.value)


def test_a_slab_that_snaps_to_nothing_is_refused_rather_than_run_as_vacuum():
    grid = fd.plan(quiet(scene)).grid                       # dx = 0.05
    thin = fd.Slab(z=(0.0, 0.01))
    with pytest.raises(ValueError, match='snaps to the empty span') as e:
        thin.cells(grid)
    assert 'would run as vacuum' in str(e.value)
    assert fd.Slab(z=(0.0, 0.05)).cells(grid) == (20, 21)


def test_a_slab_is_cell_snapped_and_the_report_prints_both_spans():
    glass = fd.Structure(fd.Slab(z=(-0.223, 0.187)), fd.Material(eps=2.25))
    report = quiet(fd.check, scene(structures=(glass,)))
    axis, lo, hi, snapped_lo, snapped_hi = report.spans[0]
    assert (axis, lo, hi) == ('z', -0.223, 0.187)
    # the box spans [-1, 1] at dx = 0.05, so the cell boundaries are multiples of 0.05
    # from -1: -0.223 snaps to -0.2 (cell 16) and 0.187 to 0.2 (cell 24)
    assert (snapped_lo, snapped_hi) == pytest.approx((-0.2, 0.2))
    assert '<- snapped' in str(report)
    assert 'structure  z=(-0.223, 0.187)' in str(report)


def test_the_report_prints_the_params_bytes_beside_the_state_bytes():
    plain = quiet(fd.check, scene())
    structured = quiet(fd.check, scene(structures=(
        fd.Structure(fd.Slab(z=(-0.2, 0.2)), fd.Conductor(eps=2.25, sigma=0.1)),)))
    assert plain.bytes_per_params < structured.bytes_per_params
    # the coefficients vary along z only, so they cost 3 * nz * 8 bytes and not 3 * cells * 8
    assert structured.bytes_per_params < 0.05 * structured.bytes_per_state
    assert 'params' in str(plain) and 'MB (coefficients, profiles)' in str(plain)


# ---------------------------------------------------------------------------------------------
# The pole CFL
# ---------------------------------------------------------------------------------------------

def test_a_pole_past_the_cfl_bound_is_refused_with_the_bound_and_the_pole():
    fast = fd.Dispersive(eps_inf=1.0, poles=(fd.Lorentz(omega_0=4000.0, gamma=1.0,
                                                        strength=1.0),))
    structure = fd.Structure(fd.Slab(z=(-0.2, 0.2)), fast)
    with pytest.raises(ValueError, match='stability bound') as e:
        quiet(fd.plan, scene(structures=(structure,)))
    message = str(e.value)
    assert 'max(w0^2)' in message and 'Lorentz(' in message and 'omega_0=4000.0' in message
    assert 'Run(courant=' in message


def test_the_bound_without_the_strength_term_would_let_a_drude_pole_through():
    """The mutant the plan names: a bound that drops ``sum strength/eps`` stops refusing.

    A Drude pole has ``omega_0 = 0``, so a bound built from the resonances alone is the
    vacuum bound and admits any plasma frequency at all. This pins that the strength term is
    load-bearing by finding a pole the real bound refuses and the resonance-only bound does
    not.
    """
    from wavebit.fdtd._prepare.checks import cfl_bound
    plasma = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=3000.0, gamma=1.0),))
    unstable = scene(structures=(fd.Structure(fd.Slab(z=(-0.2, 0.2)), plasma),))
    with pytest.raises(ValueError, match='stability bound'):
        quiet(fd.plan, unstable)
    plan = quiet(fd.plan, scene())                       # the same grid, vacuum
    bound, reason = cfl_bound(unstable, plan)
    assert reason is None and bound < plan.dt, (bound, plan.dt)
    resonance_only = plan.dx / math.sqrt(3.0)            # what dropping the strength gives
    assert resonance_only > plan.dt, 'the resonance-only bound would have admitted it'


def test_a_slower_courant_takes_the_same_pole():
    # A Lorentz pole far above the operating frequency: eps is ~1 there, so the mesh checks
    # have nothing to say and the CFL is the only door in the way.
    fast = fd.Dispersive(eps_inf=1.0, poles=(fd.Lorentz(omega_0=4000.0, gamma=1.0,
                                                        strength=1.0),))
    structures = (fd.Structure(fd.Slab(z=(-0.2, 0.2)), fast),)
    with pytest.raises(ValueError, match='stability bound') as e:
        quiet(fd.plan, scene(structures=structures))
    # The refusal names a courant that itself plans (C5: it said "at most" the refused value).
    asked = float(str(e.value).split('Run(courant=...) to ')[1].split(' or less')[0])
    quiet(fd.plan, scene(structures=structures, run=fd.Run(steps=4, courant=asked)))
    ok = quiet(fd.plan, scene(structures=structures, run=fd.Run(steps=4, courant=asked * 0.5)))
    assert ok.terms and ok.terms[0].n_poles == 1


def test_a_medium_faster_than_vacuum_tightens_the_bound_too():
    """``eps < 1`` is an ordinary material and is faster than light in vacuum.

    The usual spelling of the CFL drops ``n_min`` and lets one diverge; the bound here is
    ``n_min dx / sqrt(3)`` before the poles touch it.
    """
    from wavebit.fdtd._prepare.checks import cfl_bound
    plan = quiet(fd.plan, scene())
    slow = scene(structures=(fd.Structure(fd.Slab(z=(-0.2, 0.2)), fd.Material(eps=4.0)),))
    fast = scene(structures=(fd.Structure(fd.Slab(z=(-0.2, 0.2)), fd.Material(eps=0.25)),))
    # n_min is over every material in the scene, the vacuum background included, so glass
    # does not loosen the bound -- and a medium faster than vacuum tightens it.
    assert cfl_bound(slow, plan)[0] == pytest.approx(1.0 * plan.dx / math.sqrt(3.0))
    assert cfl_bound(fast, plan)[0] == pytest.approx(0.5 * plan.dx / math.sqrt(3.0))
    with pytest.raises(ValueError, match='stability bound'):
        quiet(fd.plan, fast.replace(run=fd.Run(steps=4, courant=0.9)))


# ---------------------------------------------------------------------------------------------
# The scene's own doors
# ---------------------------------------------------------------------------------------------

def test_the_old_keywords_point_at_the_stage_1_spellings():
    base = quiet(scene)
    kw = dict(domain=base.domain, boundaries=base.boundaries, resolution=base.resolution,
              run=base.run)
    with pytest.raises(NotImplementedError, match='background') as e:
        fd.Scene(materials=(), **kw)
    assert 'structures=' in str(e.value)
    with pytest.raises(NotImplementedError, match='Boundaries') as e:
        fd.Scene(pml=0.5, **kw)
    assert 'fd.Boundaries.open(' in str(e.value)


def test_structures_takes_structures_and_the_scalar_shorthand_is_a_material():
    one = fd.Structure(fd.Slab(z=(-0.2, 0.2)), 2.25)
    assert one.material == fd.Material(eps=2.25)
    assert quiet(scene, structures=one).structures == (one,)
    with pytest.raises(TypeError, match='fd.Structure'):
        quiet(scene, structures=(fd.Slab(z=(-0.2, 0.2)),))
    with pytest.raises(TypeError, match='stage 4'):
        fd.Structure(shape='a sphere', material=2.25)
    assert quiet(scene).background == fd.Material(eps=1.0)
    assert quiet(scene, background=2.25).background == fd.Material(eps=2.25)


def test_a_dispersive_background_is_refused_naming_absorber_and_the_gate_it_lacks():
    metal = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=1.0, gamma=1.0),))
    with pytest.raises(NotImplementedError, match='fd.Absorber') as e:
        quiet(fd.plan, scene(background=metal))
    assert 'as a Structure' in str(e.value)


@pytest.mark.parametrize('bad, match', [
    (dict(eps=0.0), 'positive'),
    (dict(eps=-1.0), 'positive'),
    (dict(eps=(1.0, 2.0)), 'three entries'),
    (dict(eps=float('inf')), 'positive'),
])
def test_a_permittivity_that_is_not_a_permittivity_is_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        fd.Material(**bad)
    assert fd.Material(eps=(1.0, 2.0, 3.0)).eps == (1.0, 2.0, 3.0)


def test_a_negative_conductivity_is_refused_and_zero_is_not():
    with pytest.raises(ValueError, match='non-negative'):
        fd.Conductor(eps=1.0, sigma=-1e-9)
    assert fd.Conductor(eps=1.0, sigma=0.0).sigma == (0.0, 0.0, 0.0)


def test_a_dispersive_material_with_no_pole_is_refused_naming_material():
    with pytest.raises(ValueError, match='fd.Material'):
        fd.Dispersive(eps_inf=2.0, poles=())
    with pytest.raises(TypeError, match='fd.Drude'):
        fd.Dispersive(eps_inf=2.0, poles=(fd.Material(eps=2.0),))
    assert len(fd.Dispersive(poles=fd.Drude(omega_p=1.0)).poles) == 1


def test_boundaries_open_puts_one_absorbing_face_everywhere_and_refuses_the_rest():
    pml = fd.PML(cells=12, wavelength=1.0)
    assert fd.Boundaries.open(pml).face('y+') is pml
    assert fd.Boundaries.open(fd.Absorber(cells=8)).face('x-').cells == 8
    with pytest.raises(TypeError, match='absorbing face'):
        fd.Boundaries.open(fd.PEC())
    with pytest.raises(TypeError, match='Boundaries.open'):
        fd.Boundaries.open(pml, pmls=pml)
    assert fd.Absorber(cells=2).cells == 2
    with pytest.raises(ValueError, match='two is the minimum'):
        fd.Absorber(cells=1)


def test_absorber_takes_no_background_and_pml_does():
    """A keyword nobody reads is worse than a missing one; the ``@spec`` door refuses it.

    ``Absorber(background=...)`` was accepted, coerced and stored, and then read by nothing:
    only ``PML.background`` is consumed (``src/wavebit/fdtd/_prepare/params.py::_cpml``, which walks the PML
    faces), and an ``Absorber``'s rate is ``6 u**3 / thickness`` whatever the medium is. That
    is exactly the failure ``fd``'s unknown-keyword refusal exists to prevent -- the run
    answers the scene you had rather than the one you asked for -- so the field is gone and
    the door refuses the spelling by name.

    It is *not* made to read the value instead: the grading is matched by construction,
    because the electric and magnetic rates are equal whatever the medium is, so there is
    nothing for a background to change. That is the whole reason the keyword was dead.
    """
    with pytest.raises(TypeError, match='background') as e:
        fd.Absorber(cells=12, background=fd.Material(eps=2.25))
    assert 'Absorber' in str(e.value)
    assert not hasattr(fd.Absorber(cells=12), 'background')
    matched = fd.PML(cells=12, wavelength=1.0, background=fd.Material(eps=2.25))
    assert matched.background == fd.Material(eps=2.25)
    assert fd.PML(cells=12, wavelength=1.0).background is None


# ---------------------------------------------------------------------------------------------
# Overlapping structures, and the one combination the pole term cannot answer
# ---------------------------------------------------------------------------------------------

METAL = fd.Dispersive(eps_inf=2.25, poles=(fd.Lorentz(omega_0=12.0, gamma=1.0, strength=4.0),))
GLASS = fd.Material(eps=2.25)


@pytest.mark.parametrize('first, second', [
    pytest.param(METAL, GLASS, id='a later lossless slab over a dispersive one'),
    pytest.param(GLASS, METAL, id='a later dispersive slab over a lossless one'),
    pytest.param(METAL, METAL, id='two dispersive slabs on the same cells'),
])
def test_two_structures_sharing_cells_are_refused_when_either_is_dispersive(first, second):
    """The overlap rule is carried by the permittivity map, and the pole term is not in it.

    "A later structure overwrites an earlier one where they overlap" holds for ``eps``, ``mu``
    and ``sigma``, which go through ``coefficients.painters``. It does **not** hold for the
    polarisation: ``plan._pole_terms`` gives each dispersive structure a term over its own
    whole slab. Measured on a 64-cell line before this door existed, ``(metal, glass)`` on the
    same cells -- which the rule says must equal ``(glass,)`` -- was instead bit-identical to
    ``(metal,)`` and 64% of the peak away from the glass run, and ``(metal, metal)`` differed
    from ``(metal,)`` by 72% of its peak. Stage 1 refuses rather than answer the wrong scene.
    """
    span = fd.Slab(z=(-0.2, 0.2))
    both = (fd.Structure(span, first), fd.Structure(span, second))
    with pytest.raises(ValueError, match='share cells') as e:
        quiet(fd.plan, scene(structures=both))
    message = str(e.value)
    assert 'fd.Dispersive' in message and 'polarisation' in message
    assert 'disjoint slabs' in message, 'and the spelling that is accepted instead'


def test_slabs_on_different_axes_always_share_cells_and_the_message_says_so():
    """A z slab and an x slab are each full-width on the other's axis, so they always meet."""
    crossing = (fd.Structure(fd.Slab(z=(-0.2, 0.2)), METAL),
                fd.Structure(fd.Slab(x=(-0.2, 0.2)), GLASS))
    with pytest.raises(ValueError, match='share cells') as e:
        quiet(fd.plan, scene(structures=crossing))
    assert 'slabs on different axes always intersect' in str(e.value)


def test_the_accepted_spellings_run_and_a_lossless_overlap_still_follows_the_order_rule():
    """Disjoint slabs, one covering the union -- and non-dispersive overlaps are untouched."""
    disjoint = (fd.Structure(fd.Slab(z=(-0.4, -0.2)), METAL),
                fd.Structure(fd.Slab(z=(-0.2, 0.0)), GLASS))
    assert len(quiet(fd.plan, scene(structures=disjoint)).terms) == 1
    union = (fd.Structure(fd.Slab(z=(-0.4, 0.0)), METAL),)
    assert len(quiet(fd.plan, scene(structures=union)).terms) == 1
    # Two lossless structures may overlap, and the later one wins where they do: the pole term
    # is the only thing outside the painters, so nothing else needed a door.
    layered = scene(structures=(fd.Structure(fd.Slab(z=(-0.4, 0.4)), fd.Material(eps=4.0)),
                                fd.Structure(fd.Slab(z=(-0.2, 0.2)), GLASS)))
    single = scene(structures=(fd.Structure(fd.Slab(z=(-0.2, 0.2)), GLASS),))
    be_layered = np.asarray(fd.lower(layered, quiet(fd.plan, layered), 'numpy')['be'])
    be_single = np.asarray(fd.lower(single, quiet(fd.plan, single), 'numpy')['be'])
    # The box spans [-1, 1] at dx = 0.05, so the outer slab is cells 12..28 and the inner one
    # 16..24: cell 20 is shared and cell 13 belongs to the outer slab alone.
    assert be_layered[:, 20] == pytest.approx(be_single[:, 20]), 'the later one wins where both'
    assert be_layered[0, 13] == pytest.approx(be_single[0, 20] * 2.25 / 4.0), 'eps 4 there'
    assert be_layered[0, 13] != pytest.approx(be_single[0, 13]), 'the earlier one elsewhere'


def test_a_debye_structures_instantaneous_part_lands_in_the_update_permittivity():
    """``eps_for_update`` is what the coefficients divide by, and nothing else reads it.

    The trapezoidal Debye recurrence is implicit in ``E``; the part of the polarisation that
    is simultaneous with the field being solved for is a constant ``c`` added to the
    permittivity, and the recurrence carries the rest. This pins that ``be`` really is built
    from ``eps_inf + c`` while the CFL bound and ``eps_at`` still read ``eps_inf``, and that a
    pole with no instantaneous part -- every Drude and every Lorentz -- moves nothing.
    """
    relaxing = fd.Dispersive(eps_inf=2.0, poles=(fd.Debye(tau=0.2, delta_eps=1.0),))
    lorentzian = fd.Dispersive(eps_inf=2.0, poles=(fd.Lorentz(omega_0=12.0, gamma=1.0),))
    built = scene(structures=(fd.Structure(fd.Slab(z=(-0.2, 0.2)), relaxing),))
    plan = quiet(fd.plan, built)
    c = 1.0 * plan.dt / (2 * 0.2 + plan.dt)
    assert relaxing.eps_for_update(plan.dt) == pytest.approx((2.0 + c,) * 3)
    assert lorentzian.eps_for_update(plan.dt) is lorentzian.eps, 'the same object, untouched'
    assert fd.Material(eps=2.0).eps_for_update(plan.dt) == (2.0, 2.0, 2.0)
    be = np.asarray(fd.lower(built, plan, 'numpy')['be'])
    inside = be.shape[1] // 2
    assert be[0, inside] == pytest.approx(plan.dt / plan.dx / (2.0 + c))
    # and the physics readings are the high-frequency permittivity, not that
    assert relaxing.index_bound() == pytest.approx(math.sqrt(2.0))
    assert fd.eps_at(relaxing, 1.0).real == pytest.approx(2.0 + relaxing.poles[0].eps_term(
        2 * math.pi / 1.0).real)


def test_a_scene_run_refuses_is_refused_by_check_with_the_same_message():
    """``fd.check`` promises "what the run would be"; a report of a scene that cannot run is not.

    ``report`` lowers on NumPy to count the params' bytes, and that lowering used to be wrapped
    in a bare ``except Exception`` whose warning named only the exception class. So
    ``fd.check`` printed a complete, healthy-looking report -- cells, dt, dtype, bytes, the
    absorber lines, the skeleton -- for a scene ``fd.run`` then refused outright, and the
    refusal's own text, which is the thing that names the fix, never reached the reader. The
    catch is now narrowed to what it was written for: ``np.asarray`` on a traced leaf, which
    raises ``TypeError``/``RuntimeError`` and is not a refusal.
    """
    torch = pytest.importorskip('torch')
    traced = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
    refused = scene(boundaries=fd.Boundaries(
        x=fd.Periodic(), y=fd.Periodic(),
        z=fd.PML(cells=12, wavelength=1.0, background=fd.Material(eps=traced))))
    with pytest.raises(ValueError, match='traced') as by_check:
        quiet(fd.check, refused)
    with pytest.raises(ValueError, match='traced') as by_run:
        quiet(fd.run, refused, backend='torch')
    assert str(by_check.value) == str(by_run.value)
    assert 'PML(background=...)' in str(by_check.value), 'and it names the fix'
