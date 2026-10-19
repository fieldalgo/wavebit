"""Stage 1, gate 6 -- the ``Absorber``: what it costs, where it is the only door, and its limits.

An :class:`~wavebit.fdtd._legacy.Absorber` face is a graded matched conductivity instead of a stretched
coordinate: no memory, no kernel code, four coefficient arrays. It exists for the one thing a
``PML`` cannot take -- a conductive or dispersive medium that crosses the face -- and it is
never the default, because in vacuum it is *hundreds to millions* of times worse. This gate
measures both halves of that sentence with the doubling method gate 1 uses
(``tests/fdtd/test_fd_pml.py``), so that the two absorbers are compared by one measurement and not
by two.

**The method, and the one thing added to it.** Two boxes whose cells coincide one for one over
the whole of the shorter, differing only in how far the ``z+`` layer is; the difference at a
probe three cells in front of the layer is that layer's reflection and nothing else. To that,
this file adds ``2 * cells * dx / cos(theta)`` of window: ``fd_helpers.packet_layout`` measures
its window to the layer's *inner face*, which is enough for a PML (whose echo is six decades
down) and is not enough for an absorber, whose echo is 5e-2 and arrives after a full round trip
*through* the layer. Measured without the extra window, a 24-cell absorber reads 2.8e-2 instead
of 4.8e-2 -- the window ended before the echo did.

**The table**, measured 2026-09-18 at ``dx = wavelength/20``, ``courant = 0.5``, amplitude ratio
at the carrier frequency, with the same scenes' ``PML`` numbers beside it:

    cells      0 deg     30 deg     60 deg  |      PML at the same depth and angle
        8    5.0e-02    1.0e-01    3.8e-01  |   4.7e-03    3.5e-03    9.7e-04
       12    4.9e-02    9.5e-02    3.4e-01  |   9.9e-05    5.6e-05    1.1e-04
       16    4.9e-02    7.1e-02    2.9e-01  |   1.3e-06    9.9e-07    1.3e-04
       24    4.8e-02    3.8e-02    1.6e-01  |   2.3e-08    1.7e-07    1.7e-04
       32    4.8e-02    3.3e-02    8.2e-02  |   7.8e-09    2.1e-07    1.9e-04

**The number that is why ``PML`` is the default: 495x at twelve cells and normal incidence, and
it grows with depth to 2.1e6 at twenty-four**, because the PML's reflection falls a decade every
two cells and the absorber's does not fall at all. At grazing (80 degrees, the slow row) the
absorber reflects 7.1e-1 against the PML's 3.4e-2: it is matched at normal incidence and nowhere
else.

**Why the 0-degree column is flat, and the gate's oracle.** With equal electric and magnetic
rates the medium is impedance-matched, so at normal incidence nothing reflects off the grading
and the whole number is what survived the trip to the PEC behind the layer and back: for a
matched rate ``r(z)`` the amplitude attenuates as ``exp(-integral r dz)`` each way, and
``6 u^3`` integrates to ``3/2`` over the layer, so the round trip is ``exp(-3) = 4.98e-2``
**whatever the depth is** -- the profile is normalised by its own thickness, so more cells
spread the same total. :func:`test_the_normal_incidence_number_is_the_profiles_own_round_trip`
measures that law directly by scaling the profile: x1 / x2 / x3 give 4.9e-2 / 2.6e-3 / 1.6e-4
against ``exp(-3)`` / ``exp(-6)`` / ``exp(-9)`` = 5.0e-2 / 2.5e-3 / 1.2e-4. That is the oracle,
and it is what makes the flat column a statement about the absorber rather than a self-captured
table.

**Four findings this file measured. Three of them changed ``wavebit/`` and the fourth did not.**

1. *(fixed)* ``Absorber``'s docstring, and ``damping_profile``'s in
   ``src/wavebit/fdtd/_prepare/cpml.py``, said the amplitude decays by ``exp(-3)`` *across the layer*,
   "once through the electric rate and once through the equal magnetic one". ``exp(-3)`` is the **round trip**: two matched rates give the
   attenuation constant ``r``, not ``2 r``, so one pass is ``exp(-1.5)``. A reader who believed
   the docstring budgeted 2.5e-3 where the layer delivers 5.0e-2, twenty times more optimistic
   than the thing it described. Both now say the round trip;
   :func:`test_the_docstrings_say_the_round_trip_this_file_measures` keeps them saying it.
2. *(reported, not fixed)* The profile's strength is a ported constant with no measurement
   behind it in this kernel, and it is far from its own optimum: scaling ``6 u^3`` by 5 takes
   the normal-incidence number from 4.9e-2 to 1.3e-5, 3700x, while 30 and 60 degrees do not
   move (8.2e-2 and 3.7e-1). :func:`test_the_profile_strength_is_not_settled_by_this_probe`
   pins that, in the spirit of gate 1's
   ``test_the_cfs_pole_and_kappa_are_not_settled_by_this_probe``: a surprise recorded, not an
   optimum proposed, because what the constant is *for* is the evanescent near field of a
   structure sitting in the layer and nothing here can see one. The constant stays where it is
   and ``damping_profile``'s docstring now says so.
3. *(message fixed, door kept)* The refusal that sends a lossy medium from ``PML`` to
   ``Absorber`` costs 21x at normal incidence: with the door held open the PML reflects 5.2e-4
   against the Absorber's 1.1e-2, and 20 000 steps show no late-time growth. Its stated reason
   -- "the stretch would not be matched and the layer would reflect instead of absorb" --
   overstated the conductor case; a stretch is a change of variable and stays matched to a
   uniform medium whatever its loss, and what it loses is the grading's ``sqrt(eps_r)``
   calibration. The message now says that and carries both numbers; the door stands, because
   what it guards against is the dispersive case's late-time instability, which 20 000 steps at
   one probe cannot rule out.
   :func:`test_what_the_pml_the_refusal_turns_away_would_have_done_with_that_medium`.
4. *(fixed)* An ``Absorber`` face did **not** refuse the geometry a ``PML`` face refuses -- a
   structure that stops inside the layer -- because ``checks.check_pml_background`` (renamed
   ``check_absorbing_layers``, since it is no longer only about a PML's background) walked
   ``plan.pml_cells`` only. It costs 20x there too, so the refusal now walks
   ``plan.absorbing_faces`` and the PML message no longer offers "make that face an Absorber"
   as a way out of that geometry.
   :func:`test_a_structure_that_ends_inside_an_absorbing_layer_is_refused` is the door and
   :func:`test_what_the_interface_inside_the_layer_would_have_cost` is the measurement, now run
   on both faces.

**The mutants**, and why one of them only bites at strength. The grading exists so that the rate
rises from zero smoothly at the inner face; replacing it with a step of the *same integral*
changes nothing at the shipped strength (4.6e-2 against 4.9e-2 -- the transmission floor hides
it, which is finding 2 stated the other way round), and at x10, where that floor is gone, it is
2.9e-1 against 2.5e-5: **11 500x**. Unmatching the two rates bites at both: 2.7e-1 at the shipped
strength (5.6x, the only mutant that bites there) and 10 200x at x10.

**Cost.** 0 and 30 degrees are a tenth of a second a run; 60 degrees is half a second and 80 is
seven, so those two columns are ``slow``.
"""
import contextlib
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
import wavebit.fdtd._prepare.checks as checks_module
import wavebit.fdtd._prepare.coefficients as coefficients_module
from tests.fixtures.fdtd.fd_helpers import Packet, amplitude_at, backends, packet_layout, run_mutated, to_numpy
from wavebit.fdtd._legacy_results import Series

#: What this file measured for ``Absorber``: rows are cells, columns the nominal angle, values
#: the amplitude ratio at the carrier frequency.
TABLE = {
    8: {0: 5.04e-02, 30: 1.03e-01, 60: 3.77e-01},
    12: {0: 4.91e-02, 30: 9.49e-02, 60: 3.42e-01},
    16: {0: 4.86e-02, 30: 7.13e-02, 60: 2.85e-01},
    24: {0: 4.83e-02, 30: 3.80e-02, 60: 1.64e-01},
    32: {0: 4.81e-02, 30: 3.33e-02, 60: 8.15e-02},
}
#: The same scenes with a ``PML`` on the face instead, measured in this file's harness -- which
#: is gate 1's plus the round trip through the layer, hence 9.9e-5 where gate 1 reads 9.2e-5.
PML_TABLE = {
    8: {0: 4.68e-03, 30: 3.49e-03, 60: 9.71e-04},
    12: {0: 9.93e-05, 30: 5.61e-05, 60: 1.14e-04},
    16: {0: 1.32e-06, 30: 9.93e-07, 60: 1.28e-04},
    24: {0: 2.32e-08, 30: 1.74e-07, 60: 1.65e-04},
    32: {0: 7.78e-09, 30: 2.14e-07, 60: 1.92e-04},
}
#: How far a re-measurement may drift before this gate calls it a change. Tighter than gate 1's
#: factor of three, because the 0-degree column is pinned to ``exp(-3)`` by the physics and the
#: smallest mutant here moves a number by 5.6x.
DRIFT = 2.0
#: The integral of ``6 u**3`` over a unit-width layer: what each matched rate contributes to one
#: pass through the absorber. The round trip is ``exp(-2 * PROFILE_INTEGRAL)``.
PROFILE_INTEGRAL = 1.5

#: The medium the crossing gates drive: 13.3 cells per wavelength at this resolution, so the
#: mesh check passes, and lossy enough that the layer is not the only thing absorbing.
CONDUCTOR = fd.Conductor(eps=2.25, sigma=0.2)
#: A dispersive one: a Lorentz pole well above the carrier, so ``fd.eps_at`` reads
#: 2.573+0.034j at wavelength 1 -- index 1.60, 12.5 cells per wavelength, and lossy with
#: memory rather than lossy in a coefficient, which is the other half of what a ``PML``
#: cannot match.
DISPERSIVE = fd.Dispersive(eps_inf=2.0,
                           poles=(fd.Lorentz(omega_0=12.0, gamma=1.0, strength=60.0),))
#: A lossless slab, for the geometry refusals: the material is not what they are about.
GLASS = fd.Material(eps=2.25)


def angles(fast=(0, 30)):
    """``0``, ``30``, ``60``; the ones outside ``fast`` marked slow."""
    return [pytest.param(a, marks=() if a in fast else pytest.mark.slow) for a in (0, 30, 60)]


def table_cases():
    return [pytest.param(n, a, marks=() if a in (0, 30) else pytest.mark.slow)
            for a in (0, 30, 60) for n in sorted(TABLE)]


# ---------------------------------------------------------------------------------------------
# The harness: gate 1's doubling pair, with a face type, a medium, and the layer's own round trip
# ---------------------------------------------------------------------------------------------

def absorbing(cells):
    """The ``z`` faces of an :class:`~wavebit.fdtd._legacy.Absorber` scene."""
    return fd.Absorber(cells=cells)


def stretched(cells):
    """The ``z`` faces of the ``PML`` scene the same measurement is compared against."""
    return fd.PML(cells=cells, wavelength=1.0, allow_thin=True)


def doubling_pair(face_of, cells, angle, medium=None, stop=0, medium_gap=6, wavelength=1.0,
                  cells_per_wavelength=20.0, **kwargs):
    """``(test, reference, layout)``: gate 1's pair, with the face type and a medium chosen.

    ``face_of(cells)`` is the face type on both ``z`` faces. ``medium``, if given, fills from
    ``medium_gap`` cells in front of the probe to ``stop`` cells short of the box's end -- so
    ``stop=0`` runs the medium through the layer (what an ``Absorber`` face is for) and
    ``stop=6`` stops it half way inside a twelve-cell layer (what a ``PML`` face refuses). The
    medium's front is at the same cell in both boxes, so the doubling difference is still the
    far layer's reflection alone.

    This is not in ``fd_helpers`` because of the window: ``packet_layout`` measures to the
    layer's inner face, and an absorber's echo comes back *through* the layer, so
    ``2 * cells * dx / cos(theta)`` is added here. ``cells_per_wavelength`` and ``wavelength``
    are passed to both ``packet_layout`` and ``Resolution`` rather than a ``dx`` being handed
    to the latter, so that the two agree to the bit and the box still snaps to whole cells.
    """
    layout = dict(packet_layout(cells, angle, wavelength=wavelength,
                                cells_per_wavelength=cells_per_wavelength, **kwargs))
    dx, dt, nz = layout['dx'], layout['dt'], layout['nz']
    layout['steps'] += int(math.ceil(2 * cells * dx / (dt * math.cos(math.radians(angle)))))
    z_lo = 0.0
    packet = Packet(wavelength, angle, dx, dt, z0=z_lo + layout['z0_index'] * dx,
                    real=(angle == 0.0))
    phi = packet.kx * dx
    x_face = fd.Periodic() if phi == 0.0 else fd.Bloch(phase=phi)
    probe_z = z_lo + layout['probe_index'] * dx
    monitors = tuple(fd.Record(f'E{a}', at=(0.0, 0.0, probe_z), component=f'E{a}') for a in 'xyz')
    front = z_lo + (layout['probe_index'] - medium_gap) * dx

    def scene(n):
        structures = ()
        if medium is not None:
            structures = (fd.Structure(fd.Slab(z=(front, z_lo + (n - stop) * dx)), medium),)
        return fd.Scene(
            domain=fd.Domain(size=(dx, dx, n * dx), center=(0.0, 0.0, z_lo + n * dx / 2)),
            boundaries=fd.Boundaries(x=x_face, y=fd.Periodic(), z=face_of(cells)),
            resolution=fd.Resolution(cells_per_wavelength=cells_per_wavelength,
                                     wavelength=wavelength),
            run=fd.Run(steps=layout['steps'], courant=layout['courant']),
            initial=packet.initial(), monitors=monitors, structures=structures)
    return scene(nz), scene(nz + layout['extra']), layout


def reflection(face_of, cells, angle, backend='numpy', mutate_params=None, **kwargs):
    """``(R at the carrier, worst in time, layout)`` -- gate 1's two numbers, this file's scene."""
    test, reference, layout = doubling_pair(face_of, cells, angle, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.AbsorberProximityWarning)
        got, want = [run_mutated(s, backend=backend, mutate_params=mutate_params)
                     for s in (test, reference)]
    omega, spectral, worst = layout['omega'], 0.0, 0.0
    for name in got:
        a, b = got[name], want[name]
        peak = float(np.max(np.abs(to_numpy(b.values))))
        if peak == 0.0:
            continue
        difference = Series(name=name, component=b.component, t=b.t,
                            values=to_numpy(a.values) - to_numpy(b.values))
        reference_amplitude = abs(amplitude_at(b, omega))
        if reference_amplitude > 0:
            spectral = max(spectral, abs(amplitude_at(difference, omega)) / reference_amplitude)
        worst = max(worst, float(np.max(np.abs(to_numpy(a.values) - to_numpy(b.values)))) / peak)
    return spectral, worst, layout


def rebuilt_profile(factor=1.0, shape='graded', magnetic=1.0):
    """A ``mutate_params`` hook that rebuilds ``ae``/``be``/``ah``/``bh`` with a changed profile.

    ``factor`` scales the whole grading, ``shape='step'`` replaces it with a flat rate of the
    **same integral** (so only the shape changes, not how much is absorbed), and ``magnetic``
    scales the H rate alone, which is the one thing that unmatches the layer. The coefficients
    are the only place an ``Absorber`` exists, so rebuilding them is the whole mutant -- there
    is no kernel code and no memory to break instead.
    """
    def mutate(params, scene, plan, rt):
        profile, damping = coefficients_module.damping_profile, coefficients_module._damping

        def mutant_profile(depth, thickness):
            u = np.clip(np.asarray(depth) / thickness, 0.0, 1.0)
            if shape == 'graded':
                return factor * 6.0 * u ** 3 / thickness
            return np.where(u > 0.0, factor * PROFILE_INTEGRAL / thickness, 0.0)

        def mutant_damping(p, axes, field):
            out = damping(p, axes, field)
            return out * magnetic if field == 'H' else out
        coefficients_module.damping_profile = mutant_profile
        coefficients_module._damping = mutant_damping
        try:
            return {**params, **coefficients_module.build(scene, plan, rt)}
        finally:
            coefficients_module.damping_profile = profile
            coefficients_module._damping = damping
    return mutate


@contextlib.contextmanager
def pml_background_check_off():
    """Run a scene ``checks.check_absorbing_layers`` refuses, to measure what it would cost.

    That one function carries both absorbing-face refusals -- the lossy medium in a ``PML``
    layer and the interface inside any layer -- so this switches off both.

    A door is only justified by what is on the other side, and that is not visible from inside
    the door. Every use of this is a measurement of a refused configuration and says so in its
    docstring; nothing here weakens the door itself, and the refusal is asserted separately.
    """
    real = checks_module.check_absorbing_layers
    checks_module.check_absorbing_layers = lambda *a, **k: None
    try:
        yield
    finally:
        checks_module.check_absorbing_layers = real


# ---------------------------------------------------------------------------------------------
# The table, and the law under its flat column
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('cells, angle', table_cases())
def test_the_absorber_doubling_table_is_what_this_kernel_measures(cells, angle):
    """Every entry of the module docstring's table, re-measured, within a factor of two."""
    measured, worst, _ = reflection(absorbing, cells, float(angle))
    want = TABLE[cells][angle]
    assert want / DRIFT < measured < want * DRIFT, (cells, angle, measured, want)
    assert measured / 2 < worst < 20 * measured + 1e-9, (measured, worst)


def test_the_normal_incidence_number_is_the_profiles_own_round_trip():
    """``R = exp(-2 * integral(rate))`` at normal incidence: the gate's oracle, by scaling it.

    The grading reflects nothing at normal incidence because the two rates are equal, so what
    comes back is what survived the trip to the PEC and back. Scaling the whole profile by ``f``
    scales the exponent by ``f``, and the measurement follows for three decades -- until the
    grading's own discretization takes over, which is where a *deeper* layer would start to buy
    something and is the only place in this gate where one does.
    """
    for factor in (1.0, 2.0, 3.0):
        measured, _, _ = reflection(absorbing, 12, 0.0,
                                    mutate_params=rebuilt_profile(factor=factor))
        law = math.exp(-2.0 * PROFILE_INTEGRAL * factor)
        assert law / 1.5 < measured < law * 1.5, (factor, measured, law)


@pytest.mark.parametrize('cells', [8, 32])
def test_more_cells_buy_the_absorber_nothing_at_normal_incidence(cells):
    """Four times the depth, the same reflection: the profile is normalised by its thickness.

    This is the shape of the whole table and the reason ``cells`` is not the absorber's knob.
    Against it, the ``PML`` at the same two depths falls from 4.7e-3 to 7.8e-9.
    """
    measured, _, _ = reflection(absorbing, cells, 0.0)
    assert abs(measured - math.exp(-3.0)) < 0.08 * math.exp(-3.0), (cells, measured)


@pytest.mark.parametrize('angle', angles())
def test_the_absorber_costs_hundreds_of_times_the_pml_it_falls_back_from(angle):
    """495x at twelve cells and 0 degrees, 1700x at 30, 3000x at 60 -- why PML is the default."""
    absorber, _, _ = reflection(absorbing, 12, float(angle))
    pml, _, _ = reflection(stretched, 12, float(angle))
    assert absorber > 100 * pml, (angle, absorber, pml)
    want = TABLE[12][angle] / PML_TABLE[12][angle]
    assert want / 4 < absorber / pml < want * 4, (angle, absorber / pml, want)


@pytest.mark.slow
def test_at_grazing_incidence_the_absorber_is_hopeless_and_the_pml_is_merely_bad():
    """80 degrees: 7.1e-1 against 3.4e-2. A matched conductivity is matched at 0 degrees only."""
    absorber, _, _ = reflection(absorbing, 12, 80.0)
    pml, _, _ = reflection(stretched, 12, 80.0)
    assert 0.35 < absorber < 1.4, absorber
    assert absorber > 10 * pml, (absorber, pml)


@pytest.mark.parametrize('backend', backends())
def test_the_three_backends_absorb_alike(backend):
    """An ``Absorber`` is four coefficient arrays, built once per runtime; one number, to 1e-9."""
    want, _, _ = reflection(absorbing, 12, 0.0, backend='numpy')
    got, _, _ = reflection(absorbing, 12, 0.0, backend=backend)
    assert abs(got - want) < 1e-9 + 1e-6 * want, (backend, got, want)


# ---------------------------------------------------------------------------------------------
# What the Absorber is FOR: a lossy or dispersive medium that crosses the face
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('medium, name', [(CONDUCTOR, 'Conductor'), (DISPERSIVE, 'Dispersive')])
def test_a_lossy_medium_crossing_the_face_is_refused_on_a_pml_and_runs_on_an_absorber(medium,
                                                                                      name):
    """The gate's own sentence: the PML refuses naming ``Absorber``, the ``Absorber`` absorbs.

    Measured residual at the probe, six cells inside the medium, by the same doubling method:
    1.1e-2 for the conductor and 1.5e-2 for the Lorentz medium, against 4.9e-2 for the same
    layer in vacuum -- the medium's own loss is doing part of the work, which is why this
    configuration is worth having and not merely permitted.
    """
    with pytest.raises(ValueError) as e:
        fd.plan(doubling_pair(stretched, 12, 0.0, medium=medium)[0])
    message = str(e.value)
    assert 'fdtd.ConductiveAbsorber(cells=' in message and f'fd.{name}' in message
    assert "'z+'" in message

    residual, worst, _ = reflection(absorbing, 12, 0.0, medium=medium)
    vacuum, _, _ = reflection(absorbing, 12, 0.0)
    assert 3e-3 < residual < 3e-2, (name, residual)
    assert residual < vacuum, (name, residual, vacuum)
    assert worst < 3e-2, (name, worst)


def test_what_the_pml_the_refusal_turns_away_would_have_done_with_that_medium():
    """The mismatch is real and the fallback is worse: 9.9e-5 / 5.2e-4 / 1.1e-2, and no growth.

    Measured with the door held open, at twelve cells and normal incidence: the same PML face
    reflects 9.9e-5 in vacuum and 5.2e-4 with the conductor running through it -- a factor of
    five, which is the grading being normalised by ``sqrt(eps_r)`` of a background the medium
    is not -- against 1.1e-2 for the ``Absorber`` the refusal names. **So the remedy is
    twenty-one times worse here than the thing refused.** The message used to give as its
    reason that "the layer would reflect instead of absorb", which overstates what this
    configuration does: a coordinate stretch is a change of variable and stays matched to a
    *uniform* medium whatever its loss, and what it loses is the grading's calibration. It now
    says that instead, and carries both numbers.

    What the refusal does protect against is not visible in this number and is not measured
    anywhere yet: a *dispersive* medium's ``eps(w)`` makes the layer's calibration frequency
    dependent, and a CFS-PML in a dispersive medium is the classic late-time instability. This
    run does not see one -- 20 000 steps leave the tail at 1e-8 of the peak -- but 20 000 steps
    at a probe is not a stability proof, and the scenes here are a ninth of a wavelength of
    conductor. Reported, not fixed; the door stands and the number beside it is now on record.
    """
    absorber, _, _ = reflection(absorbing, 12, 0.0, medium=CONDUCTOR)
    vacuum, _, _ = reflection(stretched, 12, 0.0)
    with pml_background_check_off():
        pml, _, _ = reflection(stretched, 12, 0.0, medium=CONDUCTOR)
        long_box = doubling_pair(stretched, 12, 0.0, medium=CONDUCTOR)[0]
        long_box = long_box.replace(run=fd.Run(steps=20000, courant=0.5))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', fd.AbsorberProximityWarning)
            series = run_mutated(long_box)['Ey']
    assert pml > 3 * vacuum, (pml, vacuum)              # the mismatch is real: 5x
    assert pml < absorber / 10, (pml, absorber)         # and the named fallback is 21x worse
    values = np.abs(to_numpy(series.values))
    assert values[len(values) * 4 // 5:].max() < 1e-5 * values.max(), 'late-time growth'


def test_the_crossing_is_what_the_refusal_is_about_and_not_the_material():
    """Clear of the layer, the same conductor is a plain structure and the ``PML`` takes it.

    Pairing the two is what keeps the refusal readable as a rule about geometry rather than as
    a ban on ``fd.Conductor``: ``stop=12`` stops the medium exactly at the layer's inner face.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.AbsorberProximityWarning)
        plan = fd.plan(doubling_pair(stretched, 12, 0.0, medium=CONDUCTOR, stop=12)[0])
    assert plan.pml_cells == (('z-', 12), ('z+', 12))


@pytest.mark.parametrize('face_of, kind', [(stretched, 'PML'), (absorbing, 'ConductiveAbsorber')])
@pytest.mark.parametrize('stop', [6, 11])
def test_a_structure_that_ends_inside_an_absorbing_layer_is_refused(face_of, kind, stop):
    """An interface inside the layer, on **both** absorbing faces, and the two accepted spellings.

    ``stop`` is how many of the twelve cells of the layer the medium stops short of, so both
    values put its far face strictly inside the layer. Accepted: through the layer
    (``stop=0``) or clear of it (``stop=12``).

    The ``Absorber`` half of this refusal is not in the stage-1 design, and it is here because
    of what :func:`test_what_the_interface_inside_the_layer_would_have_cost` measures on each
    face: 317x on a ``PML`` face and 20x on an ``Absorber`` one. The old ``PML`` message
    offered "or make that face an ``fd.Absorber``" as a third way out of exactly this geometry,
    which the measurement says is not a way out at all.
    """
    with pytest.raises(ValueError) as e:
        fd.plan(doubling_pair(face_of, 12, 0.0, medium=GLASS, stop=stop)[0])
    message = str(e.value)
    assert f'ends inside the {kind}' in message and 'a mirror the layer cannot undo' in message
    assert "'z+'" in message
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.AbsorberProximityWarning)
        fd.plan(doubling_pair(face_of, 12, 0.0, medium=GLASS, stop=0)[0])
        fd.plan(doubling_pair(face_of, 12, 0.0, medium=GLASS, stop=12)[0])


@pytest.mark.parametrize('face_of, least, name', [
    pytest.param(stretched, 100.0, 'PML', id='PML: 317x'),
    pytest.param(absorbing, 10.0, 'Absorber', id='Absorber: 20x'),
])
def test_what_the_interface_inside_the_layer_would_have_cost(face_of, least, name):
    """317x and 20x, measured with the door held open -- and on a PML it is a Fresnel number.

    A lossless slab whose far face stops half way into the layer reflects 1.9e-1 against 6.0e-4
    for the same slab run through a ``PML``, and 2.4e-1 against 1.2e-2 on an ``Absorber``. On
    the PML face 1.9e-1 is very nearly the bare Fresnel coefficient of that interface,
    ``|(1 - 1.5)/(1 + 1.5)| = 0.2``: the layer does not absorb an interface's reflection just
    because the interface is inside it, because that reflection leaves the way it came and only
    the part of the layer in front of it is in the way. That is the whole content of the
    refusal -- a user who puts a slab face inside an absorbing layer is expecting a termination
    and is getting a mirror.

    **Running both faces is what justifies the Absorber half of the refusal.** The failure mode
    is the geometry's and not the stretch's, so it is as bad on the face the old PML message
    offered as an escape from it; measuring one face and refusing on both would be an
    assertion, and this is the measurement.
    """
    through, _, _ = reflection(face_of, 12, 0.0, medium=GLASS, stop=0)
    with pml_background_check_off():
        inside, _, _ = reflection(face_of, 12, 0.0, medium=GLASS, stop=6)
    assert inside > least * through, (name, inside, through)
    if name == 'PML':
        fresnel = abs((1.0 - 1.5) / (1.0 + 1.5))
        assert fresnel / 1.5 < inside < fresnel * 1.5, (inside, fresnel)


# ---------------------------------------------------------------------------------------------
# The proximity warning: it fires, and it says how far
# ---------------------------------------------------------------------------------------------

def test_a_structure_close_to_an_absorbing_face_warns_and_names_the_distance():
    """The warning carries both numbers a reader needs: the gap it found and the one it wants.

    The clearance is half a wavelength **in the structure's own medium** -- ``0.5/1.5 = 0.333``
    here, not 0.5 -- which is why the message prints the index it used. The structure below
    crosses the face, so the gap is negative, and that is the case the number has to survive
    being printed for.
    """
    scene = doubling_pair(absorbing, 12, 0.0, medium=CONDUCTOR)[0]
    with pytest.warns(fd.AbsorberProximityWarning) as record:
        plan = fd.plan(scene)
    said = [str(w.message) for w in record
            if issubclass(w.category, fd.AbsorberProximityWarning)]
    grid = plan.grid
    hi = scene.structures[0].shape.snapped(grid)[1]
    gap = (grid.high[2] - 12 * grid.dx) - hi
    clearance = 0.5 * 1.0 / 1.5
    assert gap < 0.0, gap                     # the medium crosses the face
    message = next(m for m in said if 'structures[0]' in m)
    assert f'{gap:g}' in message and f'{clearance:g}' in message, message
    assert "'z+'" in message and 'half a wavelength in its own medium' in message
    # An Absorber has no medium to name; it used to say "matched to the background".
    assert message.endswith(', and a ConductiveAbsorber there is matched at normal incidence only, not '
                            "to this structure's near field"), message


def test_the_same_structure_half_a_wavelength_clear_of_the_face_is_quiet():
    """The paired accepted spelling: the warning is about the distance, not about the face."""
    far = fd.Structure(fd.Slab(z=(1.0, 2.0)), CONDUCTOR)
    scene = doubling_pair(absorbing, 12, 0.0)[0].replace(structures=(far,))
    with warnings.catch_warnings():
        warnings.simplefilter('error', fd.AbsorberProximityWarning)
        fd.plan(scene)


# ---------------------------------------------------------------------------------------------
# The mutants, and the constant this probe can see but must not settle
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('factor, window', [
    pytest.param(1.0, (0.5, 2.0), id='shipped: the shared transmission floor hides the shape'),
    pytest.param(10.0, (1000.0, math.inf), id='x10: the entrance is the whole number'),
])
def test_mutant_the_grading_replaced_by_a_step_of_the_same_integral(factor, window):
    """The step absorbs exactly as much and enters abruptly; measured, that matters only at x10.

    At the shipped strength the real layer and the step are the same number (4.9e-2 and 4.6e-2),
    because both are pinned to ``exp(-3)`` by the integral they share and the step's entrance
    reflection is an order below it. At x10 the shared floor is ``exp(-30)`` and what is left is
    the entrance: 2.9e-1 against 2.5e-5. So this mutant is a statement about the strength as
    much as about the shape, and that is finding 2 of the module docstring from the other side.
    """
    real, _, _ = reflection(absorbing, 12, 0.0, mutate_params=rebuilt_profile(factor=factor))
    step, _, _ = reflection(absorbing, 12, 0.0,
                            mutate_params=rebuilt_profile(factor=factor, shape='step'))
    low, high = window
    assert low * real < step < high * real, (factor, step, real)


@pytest.mark.parametrize('factor, least', [
    pytest.param(1.0, 3.0, id='shipped: 5.6x, the only mutant that bites here'),
    pytest.param(10.0, 1000.0, id='x10: 10 200x'),
])
def test_mutant_the_electric_and_magnetic_rates_unmatched(factor, least):
    """Drop the magnetic rate and the layer is a conductor, not an absorber: it reflects.

    "Matched" is the whole of what makes a conductivity absorb rather than mirror -- equal rates
    leave the impedance ``sqrt(mu/eps)`` untouched at every depth, which is why a *graded*
    conductivity can be entered at all. With the magnetic half gone the impedance falls through
    the grading and every cell reflects: 2.7e-1 at the shipped strength (5.6x the real layer)
    and 2.6e-1 at x10 (10 200x). Halving it rather than dropping it gives 1.0e-1 and 1.1e-1, so
    this is a graded failure and not a switch, which is what a matching argument predicts.
    """
    real, _, _ = reflection(absorbing, 12, 0.0, mutate_params=rebuilt_profile(factor=factor))
    gone, _, _ = reflection(absorbing, 12, 0.0,
                            mutate_params=rebuilt_profile(factor=factor, magnetic=0.0))
    half, _, _ = reflection(absorbing, 12, 0.0,
                            mutate_params=rebuilt_profile(factor=factor, magnetic=0.5))
    assert gone > least * real, (factor, gone, real)
    assert real < half < gone, (factor, real, half, gone)
    assert gone > 0.1, (factor, gone)


def test_the_profile_strength_is_not_settled_by_this_probe():
    """``6 u^3`` is 3700x from its own optimum at normal incidence, and the angles do not care.

    Measured at twelve cells, the normal-incidence number runs 4.9e-2 / 2.6e-3 / 1.6e-4 / 1.3e-5
    / 2.5e-5 as the profile is scaled x1 / x2 / x3 / x5 / x10, so the shipped constant sits
    three and a half decades above the best this layer can do -- while 30 degrees moves only
    from 9.5e-2 to 8.2e-2 and 60 degrees not at all, because at an angle the number is the
    grading's own mismatch and not the transmission. Like gate 1's ``kappa`` and ``alpha``, the
    constant is ported from a legacy engine with its own probe; this pins the surprise so that
    nobody reads it as an optimum established here, and leaves it where it is, because what a
    stronger profile costs in the evanescent near field of a structure sitting in the layer is
    not measurable anywhere in this repository yet.
    """
    shipped, _, _ = reflection(absorbing, 12, 0.0)
    stronger, _, _ = reflection(absorbing, 12, 0.0, mutate_params=rebuilt_profile(factor=5.0))
    assert stronger < shipped / 1000.0, (shipped, stronger)
    shipped30, _, _ = reflection(absorbing, 12, 30.0)
    stronger30, _, _ = reflection(absorbing, 12, 30.0, mutate_params=rebuilt_profile(factor=5.0))
    assert 0.5 < stronger30 / shipped30 < 1.5, (shipped30, stronger30)


def test_the_docstrings_say_the_round_trip_this_file_measures():
    """The prose a user budgets from, checked against the number the layer delivers.

    ``Absorber`` and ``damping_profile`` used to say the amplitude decays by ``exp(-3)``
    *across* the layer, "once through the electric rate and once through the equal magnetic
    one" -- which predicts a round trip of ``exp(-6) = 2.5e-3`` where the layer delivers
    ``exp(-3) = 5.0e-2``, twenty times more optimistic. Two matched rates give the attenuation
    constant ``r`` and not ``2 r``, so one pass is ``exp(-1.5)`` and ``exp(-3)`` is the round
    trip; both docstrings now say that, and this is the check that keeps them saying it.

    A test of the *prose* and not of the physics: the physics is
    :func:`test_the_normal_incidence_number_is_the_profiles_own_round_trip`, which settles the
    same fact positively by scaling the profile.
    """
    from wavebit.fdtd._prepare.cpml import damping_profile
    for doc in (fd.Absorber.__doc__, damping_profile.__doc__):
        text = ' '.join(doc.split())            # a claim may be split across a wrapped line
        assert 'exp(-1.5)' in text, 'one pass'
        assert 'exp(-3)' in text and 'round trip' in text
        assert '4.98e-2' in text, 'and the number a reader should budget for'
    measured, _, _ = reflection(absorbing, 12, 0.0)
    assert abs(measured - math.exp(-3.0)) < 0.1 * math.exp(-3.0), (measured, math.exp(-3.0))
