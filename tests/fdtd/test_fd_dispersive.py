"""Gates 3, 4 and 5: dispersive media, a conductive medium, and the pole-aware CFL.

Three questions, each answered against a closed form this package already carries, and each
paired with the mutant that says the answer was not free.

**Gate 3 -- a dispersive half-space reflects what ``fd.eps_at`` says it should.** A Gaussian
packet is launched at a Drude half-space and at a Drude+Lorentz one, and the reflection
``R(omega)`` is compared, *amplitude and phase*, with the Fresnel coefficient
``r = (1 - n)/(1 + n)``, ``n = sqrt(eps(omega))`` from ``fd.eps_at`` -- the package's own
closed form, so the gate compares the time stepping with the formula it discretises and not
with a second implementation of that formula.

    The reflected field is taken as the **difference of two runs**: the same box with the
    structure and without it. That is the doubling method of ``test_fd_pml.py`` pointed at a
    structure instead of a layer, and it is what makes a probe half a wavelength in front of
    the interface legitimate -- the incident and reflected pulses overlap there completely,
    and the difference is the reflected field alone, exactly, at every sample. Windowing them
    apart would need a probe five wavelengths out and a box to match. Measured: with
    ``fd.Material(eps=1.0)`` as the "structure" the difference is 0 to 1e-16 of the peak, so
    the method has no floor of its own above round-off.

    **The far face is out of causal reach, not absorbed.** The medium runs to the far wall,
    which is a plain ``PEC``: the window ends before ``2 * body / c``, so nothing the wall
    does can reach the probe inside it. That is an argument from the light cone and holds for
    any material, which is what the alternative -- an ``fd.Absorber`` matched to nothing --
    does not: measured, an absorber terminating the medium scattered the numbers below by
    0.03 to 0.05 in absolute reflection, which at first reading looked like the engine.
    Moving the far wall three wavelengths further moves the answer by 5e-8.

    **The reference plane is half a cell in front of the snapped face**, and that is the whole
    of the cell-snapping error at normal incidence. A slab claims whole cells, so the
    tangential ``Ey`` node that sits *on* the front face already carries the medium's
    permittivity: the medium effectively begins ``dx/2`` earlier. Uncorrected, the measured
    ``R`` differs from the closed form by a pure phase, first order in ``dx``
    (0.383 / 0.186 / 0.093 at 20 / 40 / 80 cells per wavelength), and the offset fitted from
    that phase is 0.4910 to 0.5026 cells over every material, frequency and resolution here.
    Referred to ``front - dx/2`` the error is second order, and it is amplitude and phase
    both -- measured 2026-09-18, worst over the three probe wavelengths:

        cells per wavelength          20        40        80
        Material(eps=2.25)        5.4e-2    1.3e-2    3.2e-3
        Drude(4, 1)               3.0e-2    7.1e-3    2.1e-3
        Drude(3, .5)+Lorentz      5.9e-2    1.4e-2    3.8e-3

    Mutants: the pole's sign flipped, which is a gain medium, in the two ways that means.
    Flipping the *drive* -- the pole's whole contribution to ``eps`` -- diverges: ``|R|``
    reaches 5e2 to 3e6 where passivity caps a half-space at 1. Flipping only ``gamma``
    (``Im(eps) < 0``, ``Re(eps)`` untouched) stays bounded and is a frequency-by-frequency
    thing, since it moves ``r`` by about ``2*Im(r)``: complex error over ``|r|`` of
    0.045 / 0.267 / 0.622 at 0.85 / 1.0 / 1.2 on the Drude half-space and
    0.362 / 0.195 / 0.128 on the two-pole one. The first of those is inside the tolerance
    this gate allows -- one frequency would not have caught it -- and at two of the two-pole
    frequencies the *modulus* is inside it while the complex number is not, which is what
    makes comparing the phase load-bearing rather than decorative.

    **Debye is measured somewhere else, and that is deliberate.** It is the one pole whose
    discretisation is not a plain three-term recurrence -- its ODE is first order, so the
    trapezoidal rule on it is implicit in ``E`` and the kernel splits off an instantaneous part
    into the permittivity. Getting that split wrong is a *first-order* scheme whose error is
    almost pure loss, and the half-space rows cannot see it: the cell-snapped interface
    contributes 6.0e-2 at 20 cells per wavelength, larger than either mutant, and both pass
    here. So the Debye row below is coverage, and the oracle is
    ``test_a_uniform_debye_medium_relaxes_at_its_continuum_rate_to_second_order`` -- a fully
    periodic box with no curl anywhere, where the closed form is a single exponential with no
    spatial discretisation in it and the convergence separates 4.00 from 2.1.

**Gate 4 -- a conductive medium decays by the exact factor its own update coefficients set.**
Derived from the update and not from the continuum. With ``h = dt*sigma/(2*eps)`` the
coefficient is ``ae = (1 - h)/(1 + h)``, and for a field with no curl -- a spatially uniform
E in a periodic box, which is the pure ``sigma`` problem with the wave part removed -- the
update *is* ``E^{n+1} = ae E^n``, so ``E^n = ae^n E^0`` exactly. Measured: 7e-16 relative
over 100 steps, identically on NumPy, JAX and Torch.

    The continuum answer to that problem is ``exp(-sigma t / eps)`` (no factor 2: the 2 in
    the wave case is the half of the energy that is magnetic). It differs, because ``ae`` is
    the bilinear-transform image of ``exp(-2h)``: ``ln(ae) = -2h - 2h^3/3 - 2h^5/5 - ...``.
    At ``h = 0.0505`` the gap is 8.6083e-5 per step against a predicted 8.6083e-5 and 0.85%
    over 100 steps, so the gate asserts the series -- being *under* the continuum by exactly
    ``2(h^3/3 + h^5/5)`` per step -- rather than a tolerance that would pass for any small h.

    With the curl present the same algebra gives ``E^{n+1} - B E^n + ae E^{n-1} = 0`` with
    ``B = 1 + ae - be*bh*(Omega*dx)^2``, whose roots have modulus ``sqrt(ae)`` exactly: the
    wave decays at half the rate of the static field, the leapfrog's own statement that half
    the energy is in H. Both are checked to round-off.

    Mutant: ``sigma`` applied to the H update instead of the E update. On the uniform field
    it is caught flat -- the field never decays at all. On the *wave* it is invisible, and
    that is not a weakness of the probe but algebra: an equal magnetic rate is the
    impedance-matched medium, ``be*bh`` and ``ae``-versus-``ah`` are unchanged, and the E
    series obeys the same recurrence with the same envelope to 3e-16. The two differ only in
    their transient (0.8% of the peak) and in the wave impedance. Both facts are pinned
    below, because a gate that only ran the wave probe would have passed the mutant.

**Gate 5 -- the pole-aware CFL bound is real, and the strength term in it is load-bearing.**
``dt < 1/sqrt(1/dt_vac^2 + rate/4)`` with ``rate = max(w0^2) + max(sum strength/eps)``. A
concrete pole set past it is refused at ``fd.plan`` naming the bound, the pole and the
courant that would fit; a traced one cannot be read at the door, so ``check_stability``
answers ``False`` and the lowering multiplies ``be`` and ``bh`` by ``nan``, and the run
returns ``nan`` rather than a number shaped like a field. A bound built *without* the
strength term accepts ``Drude(omega_p=20)`` at ``dt = 0.144`` against a real bound of 0.094 --
and with the door's refusal removed, that run reaches 2.4e248 in 300 steps. That is the test
that the term is load-bearing: not that the formula has it, but that a formula without it
admits a run that explodes.
"""
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._prepare import checks, lower
from wavebit.fdtd._legacy_results import Series
from tests.fixtures.fdtd.fd_helpers import Packet, amplitude_at, backends, recurrence_residual, run_mutated, to_numpy

# ---------------------------------------------------------------------------------------------
# Gate 3: the half-space, the two-run difference, and the closed form
# ---------------------------------------------------------------------------------------------

#: The packet's half-length, in 1/e envelope widths. At 3.5 the envelope is 4.8e-6 of the
#: peak, so the packet is clear of the launch face and of the interface at t = 0 and its
#: truncation is far below the accuracy this gate claims.
CLEAR = 3.5

#: Where the probe sits, in wavelengths in front of the interface. It may be this close
#: because the incident field is subtracted rather than windowed away.
PROBE_AT = 0.5

#: The probe wavelengths. Each is a frequency the packet carries with weight > 0.6, and the
#: three sample a factor 1.4 in frequency, over which both materials below disperse visibly.
LAMS = (0.85, 1.0, 1.2)

GLASS = fd.Material(eps=2.25)
DRUDE = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=4.0, gamma=1.0),))
TWO_POLE = fd.Dispersive(eps_inf=1.5, poles=(fd.Drude(omega_p=3.0, gamma=0.5),
                                             fd.Lorentz(omega_0=9.0, gamma=1.0, strength=40.0)))

DISPERSIVE = [pytest.param(DRUDE, id='drude'), pytest.param(TWO_POLE, id='drude+lorentz')]


def half_space(material, cpw=20, wavelength=1.0, courant=0.5, cells=12, body=6.0,
               with_structure=True, margin=0.6):
    """A packet in +z, a half-space of ``material`` filling the far half of the box.

    ``(scene, geometry)``. The lead is ``2 * CLEAR`` packet widths plus the probe's offset,
    so the packet starts clear of the ``z-`` PML *and* of the interface; the run is long
    enough for the whole reflected packet to pass the probe (``2 * CLEAR`` widths plus the
    two probe transits plus ``margin``) and short enough that ``2 * body`` of light travel
    has not elapsed, which is what keeps the far wall out of the measurement.
    """
    dx = wavelength / cpw
    dt = courant * dx / math.sqrt(3.0)
    width = 8.0 / (2 * math.pi / wavelength)              # the packet's 1/e half width
    lead = int(round((2 * CLEAR * width + PROBE_AT * wavelength) / dx))
    body_cells = int(round(body * wavelength / dx))
    nz = cells + lead + body_cells
    front = (cells + lead) * dx
    packet = Packet(wavelength, 0.0, dx, dt, z0=front - (CLEAR * width + PROBE_AT * wavelength),
                    real=True)
    window = 2 * CLEAR * width + 2 * PROBE_AT * wavelength + margin
    structures = ()
    if with_structure:
        structures = (fd.Structure(fd.Slab(z=(front, nz * dx)), material),)
    scene = fd.Scene(
        domain=fd.Domain(size=(dx, dx, nz * dx), center=(0.0, 0.0, nz * dx / 2)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                 z=(fd.PML(cells=cells, wavelength=wavelength), fd.PEC())),
        resolution=fd.Resolution(dx=dx),
        run=fd.Run(steps=int(round(window / dt)), courant=courant),
        initial=packet.initial(), structures=structures,
        monitors=(fd.Record('probe', at=(0.0, 0.0, front - PROBE_AT * wavelength),
                            component='Ey'),))
    return scene, dict(dx=dx, dt=dt, nz=nz, front=front, window=window, body=body_cells * dx,
                       steps=scene.run.steps, wavelength=wavelength)


def vacuum_kz(omega, dx, dt):
    """The grid's own ``kz`` at ``omega`` along z: the leapfrog dispersion, inverted.

    ``sin(kz dx/2)/dx = sin(w dt/2)/dt`` for a wave travelling along z on a uniform Yee
    grid. The continuum ``kz = omega`` would leave a phase error of ``(k dx)^2/24`` per
    wavelength of travel in the reference plane below -- 0.4% at 20 cells, which is a tenth
    of what this gate claims and so worth not spending.
    """
    return 2.0 / dx * math.asin(dx / dt * math.sin(omega * dt / 2.0))


_MEASURED = {}


def reflection(material, cpw=20, body=6.0, mutant=None, offset=0.5):
    """``{wavelength: R}``: the amplitude reflection at each of :data:`LAMS`, cached.

    Two runs -- with the structure and without -- differenced at the probe, transformed at
    each frequency, divided by the incident transform, and referred from the probe to the
    plane ``front - offset*dx`` (``offset=0.5`` is the effective interface; ``offset=0`` is
    the snapped one, which is what the half-cell test measures).
    """
    key = (material, cpw, body, None if mutant is None else mutant.__name__, offset)
    if key in _MEASURED:
        return _MEASURED[key]
    test, geometry = half_space(material, cpw=cpw, body=body, with_structure=True)
    reference, _ = half_space(material, cpw=cpw, body=body, with_structure=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        warnings.simplefilter('ignore', fd.AbsorberProximityWarning)
        got = run_mutated(test, backend='numpy', mutate_params=mutant)
        want = run_mutated(reference, backend='numpy')
    incident = want['probe']
    reflected = Series(name='reflected', component='Ey', t=incident.t,
                       values=to_numpy(got['probe'].values) - to_numpy(incident.values))
    out = {}
    for lam in LAMS:
        omega = 2.0 * math.pi / lam
        k = vacuum_kz(omega, geometry['dx'], geometry['dt'])
        distance = PROBE_AT * geometry['wavelength'] - offset * geometry['dx']
        out[lam] = (amplitude_at(reflected, omega) / amplitude_at(incident, omega)
                    * np.exp(-2j * k * distance))
    out['geometry'] = geometry
    out['peak'] = float(np.max(np.abs(to_numpy(incident.values))))
    out['reflected_peak'] = float(np.max(np.abs(reflected.values)))
    _MEASURED[key] = out
    return out


def fresnel(material, wavelength):
    """``(r, n)`` at normal incidence from the package's own ``fd.eps_at``.

    ``r = (1 - n)/(1 + n)`` is the tangential-E coefficient (``1 + r = t`` at the interface),
    and the branch of the square root is the one this project's ``e^{-i w t}`` convention
    fixes: ``Im(n) > 0`` is loss.
    """
    if getattr(material, 'poles', ()):
        eps = complex(fd.eps_at(material.poles, wavelength, material.eps_inf))
    else:
        eps = complex(material.eps[0])
    n = eps ** 0.5
    if n.imag < 0:
        n = -n
    return (1 - n) / (1 + n), n


#: What the table above says, as a tolerance per resolution: the worst measured error times
#: a factor of about 1.4, so the gate fails on a change and not on the last digit.
FRESNEL_TOLERANCE = {20: 0.08, 40: 0.02, 80: 0.006}


@pytest.mark.parametrize('material', DISPERSIVE + [pytest.param(GLASS, id='glass')])
@pytest.mark.parametrize('cpw', [20, 40])
def test_a_half_space_reflects_the_closed_form_of_eps_at(material, cpw):
    """Amplitude and phase, at three frequencies, against ``fd.eps_at``'s own Fresnel.

    ``fd.Material(eps=2.25)`` runs here as the lossless control: it has no pole, so it
    isolates everything the two dispersive rows share with it -- the packet, the difference
    method, the reference plane -- from what only a pole set can get wrong.
    """
    got = reflection(material, cpw=cpw)
    for lam in LAMS:
        r, n = fresnel(material, lam)
        R = got[lam]
        error = abs(R - r) / abs(r)
        assert error < FRESNEL_TOLERANCE[cpw], (
            f'{lam}: measured {R:.4f}, closed form {r:.4f} (n = {n:.4f})')
        # Amplitude and phase, stated apart as well, so a failure says which half moved.
        assert abs(abs(R) / abs(r) - 1) < FRESNEL_TOLERANCE[cpw]
        assert abs(np.angle(R / r)) < FRESNEL_TOLERANCE[cpw]


@pytest.mark.parametrize('material', DISPERSIVE + [pytest.param(GLASS, id='glass')])
def test_the_fresnel_error_is_second_order_in_dx_about_the_effective_interface(material):
    """Halving ``dx`` divides the error by about four -- measured 3.4 to 4.3, asserted 3.

    The cell-snapped interface is a first-order error and the design says so; what is second
    order is what is left of it once the half-cell offset is taken out, which is the claim
    that the offset is the *whole* of it.
    """
    worst = []
    for cpw in (20, 40):
        got = reflection(material, cpw=cpw)
        worst.append(max(abs(got[lam] - fresnel(material, lam)[0]) / abs(fresnel(material, lam)[0])
                         for lam in LAMS))
    assert worst[0] / worst[1] > 3.0, f'errors {worst}'


@pytest.mark.slow
@pytest.mark.parametrize('material', DISPERSIVE + [pytest.param(GLASS, id='glass')])
def test_the_second_order_convergence_holds_for_a_third_point(material):
    """80 cells per wavelength: the third row of the table, and it costs a 1245-cell box."""
    worst = []
    for cpw in (20, 40, 80):
        got = reflection(material, cpw=cpw)
        worst.append(max(abs(got[lam] - fresnel(material, lam)[0]) / abs(fresnel(material, lam)[0])
                         for lam in LAMS))
    assert worst[1] / worst[2] > 3.0, f'errors {worst}'
    assert worst[2] < FRESNEL_TOLERANCE[80], f'errors {worst}'


@pytest.mark.parametrize('material', DISPERSIVE + [pytest.param(GLASS, id='glass')])
def test_the_effective_interface_is_half_a_cell_in_front_of_the_snapped_face(material):
    """Referred to the snapped face the error is a pure phase, and it fits one half cell.

    The offset is read out of the measurement rather than assumed: it is
    ``-arg(R/r) / (2 k dx)``, and it comes out 0.49 to 0.51 for every material, frequency
    and resolution here. The same numbers say the uncorrected error is a *phase* error and
    not an amplitude one: the modulus is already within what the corrected complex number
    claims (it *is* that number, to the fourth digit -- taking out the phase leaves exactly
    it), while the phase is 6 to 85 times larger.
    """
    for cpw in (20, 40):
        raw = reflection(material, cpw=cpw, offset=0.0)
        for lam in LAMS:
            r, _ = fresnel(material, lam)
            cells = -np.angle(raw[lam] / r) / (2 * (2 * math.pi / lam) * raw['geometry']['dx'])
            assert abs(cells - 0.5) < 0.02, f'{material} at {cpw}, {lam}: offset {cells:.4f}'
            modulus = abs(abs(raw[lam]) / abs(r) - 1)
            assert modulus < FRESNEL_TOLERANCE[cpw]
            assert abs(np.angle(raw[lam] / r)) > 5 * modulus, 'the error is the phase'


#: A relaxation whose pole sits on the probe band: ``omega tau = 0.94`` at wavelength 1, so
#: ``eps`` runs 2.68+0.78j / 2.79+0.75j / 2.95+0.68j over :data:`LAMS` -- dispersive in both
#: parts across the band, which is what makes the rows below a test of the *pole* and not of
#: ``eps_inf``. Index 1.70, so 11.8 cells per wavelength at ``cpw=20`` (a CoarseMeshWarning,
#: which ``reflection`` silences) and 23.5 at 40; the CFL bound is 0.0288 against dt 0.0144.
DEBYE = fd.Dispersive(eps_inf=2.0, poles=(fd.Debye(tau=0.15, delta_eps=1.5),))


#: The continuum relaxation time of a uniform Debye medium left to itself, derived below.
DEBYE_TIME = 0.15 * 2.0 / (1.5 + 2.0)


def uniform_relaxation_rate(courant, cells=8, dx=0.05, steps=60):
    """The per-unit-time decay rate of a spatially uniform ``Ey`` in a Debye medium.

    A fully periodic box filled with :data:`DEBYE` and started with a uniform ``Ey`` has no
    curl at any step, so the wave part of the update is identically zero and what is left is
    the E-P pair alone -- the pure polarisation problem, with no interface, no cell snapping
    and no grid dispersion to hide behind. The rate is read as a ratio of successive
    increments, ``(E^{n+2} - E^{n+1}) / (E^{n+1} - E^n)``, which is exact for ``A + B r^n`` and
    needs neither the asymptote nor the amplitude -- so the one-step startup offset the pole
    memory carries (it begins at rest, which drops the first step's drive) scales ``B`` and
    does not touch what is measured.
    """
    material = fd.Dispersive(eps_inf=2.0, poles=(fd.Debye(tau=0.15, delta_eps=1.5),))
    length = cells * dx
    scene = fd.Scene(
        domain=fd.Domain(size=(dx, dx, length), center=(0.0, 0.0, length / 2)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic()),
        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps, courant=courant),
        initial=fd.InitialField(E=(None, lambda x, y, z: np.ones_like(z), None)),
        structures=(fd.Structure(fd.Slab(z=(0.0, length)), material),),
        monitors=(fd.Record('Ey', at=(0.0, 0.0, length / 2), component='Ey'),))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        values = to_numpy(fd.run(scene)['Ey'].values).astype(float)
        dt = fd.plan(scene).dt
    increments = np.diff(values)
    ratio = float(np.median(increments[21:41] / increments[20:40]))
    return -math.log(ratio) / dt


def test_a_uniform_debye_medium_relaxes_at_its_continuum_rate_to_second_order():
    """The kernel's own field against a closed form, for the one pole that is not a recurrence.

    **The closed form.** With no curl, Ampere's law is ``eps_inf E' + P' = 0``, so
    ``eps_inf E + P`` is conserved and ``P = eps_inf (E0 - E)``. Substituting into
    ``tau P' + P = delta_eps E`` leaves ``E' = -(E - A)/T`` with
    ``T = tau eps_inf / (delta_eps + eps_inf)`` -- a single exponential, exactly, with no
    spatial discretisation in it at all.

    **Why this gate exists and the half-space one does not settle it.** Debye's ODE is first
    order, so the trapezoidal rule on it is implicit in ``E``; the recurrence cannot be driven
    by ``E^n`` alone without dropping to first order. The kernel runs it for the auxiliary
    ``P~ = P - c E`` and folds ``c`` into the permittivity the E update divides by
    (``materials.Debye``, ``MaterialBase.eps_for_update``, ``src/wavebit/fdtd/_prepare/coefficients.py``).
    Measured here, the rate error is 2.4e-3 / 5.9e-4 / 1.5e-4 as the courant is halved --
    ratio 4.00, second order. Both ways of getting the split wrong are first order and land at
    ratio 2.1: the explicit drive ``2 c E^n`` (4.0e-2 at courant 0.5, 17x) and the split drive
    with ``c`` left out of ``eps`` (1.8e-2, 7.5x).

    :func:`test_a_debye_half_space_reflects_the_closed_form_of_eps_at` runs the same material
    in the standard geometry and **passes for both of those mutants**, because there the
    cell-snapped interface contributes 6e-2 and swamps them. That is why the oracle for the
    scheme is here, where there is no interface.
    """
    errors = [abs(uniform_relaxation_rate(courant) * DEBYE_TIME - 1.0)
              for courant in (0.5, 0.25, 0.125)]
    assert errors[0] < 5e-3, errors
    for coarse, fine in zip(errors, errors[1:]):
        assert 3.5 < coarse / fine < 4.5, errors


@pytest.mark.parametrize('cpw', [20, 40])
def test_a_debye_half_space_reflects_the_closed_form_of_eps_at(cpw):
    """Debye in the standard geometry, held to the tolerance the other materials are.

    Coverage and not an oracle for the scheme, and the difference is worth stating: the
    cell-snapped interface contributes 6.0e-2 at 20 cells per wavelength and 1.4e-2 at 40 --
    the same size as the glass row -- which is more than either Debye discretisation error, so
    this row passes for both mutants named in
    :func:`test_a_uniform_debye_medium_relaxes_at_its_continuum_rate_to_second_order`. What it
    does pin is that the pole runs at all in a scene with a boundary and a structure: dropping
    the term entirely leaves ``eps_inf = 2.0`` where the closed form wants 2.8, which is far
    outside the tolerance.
    """
    got = reflection(DEBYE, cpw=cpw)
    for lam in LAMS:
        r, n = fresnel(DEBYE, lam)
        error = abs(got[lam] - r) / abs(r)
        assert error < FRESNEL_TOLERANCE[cpw], (
            f'{lam}: measured {got[lam]:.4f}, closed form {r:.4f} (n = {n:.4f})')
        assert abs(abs(got[lam]) / abs(r) - 1) < FRESNEL_TOLERANCE[cpw]
        assert abs(np.angle(got[lam] / r)) < FRESNEL_TOLERANCE[cpw]


def test_the_two_run_difference_has_no_floor_of_its_own():
    """A "structure" of vacuum reflects nothing, and the method says so to round-off.

    The scene with ``fd.Material(eps=1.0)`` builds the coefficient *arrays* a structure
    builds -- it takes the same branch of ``src/wavebit/fdtd/_prepare/coefficients.py`` -- and the run is
    nonetheless the reference run's, value for value.
    """
    got = reflection(fd.Material(eps=1.0), cpw=20)
    assert got['peak'] > 1.0
    assert got['reflected_peak'] / got['peak'] < 1e-15
    for lam in LAMS:
        assert abs(got[lam]) < 1e-14


def test_the_far_wall_is_out_of_causal_reach_of_the_probe():
    """Moving the PEC wall three wavelengths further changes nothing that matters.

    The window is 9.9 length units and the wall is 6 away, so 2*body/c = 12 > 9.9: no signal
    from it can have arrived. Measured 5e-8 rather than round-off, which is the packet's own
    Gaussian tail touching the wall at t = 0 and coming back; six orders below what the gate
    claims either way.
    """
    near = reflection(DRUDE, cpw=20, body=6.0)
    far = reflection(DRUDE, cpw=20, body=9.0)
    assert near['geometry']['window'] < 2 * near['geometry']['body']
    for lam in LAMS:
        assert abs(near[lam] - far[lam]) < 1e-5


def flipped_drive(params, scene, plan, rt):
    """Mutant: every pole's contribution to ``eps`` changes sign, which is gain."""
    return {**params,
            'poles': {name: {**c, 'drive': tuple(-d for d in c['drive'])}
                      for name, c in params['poles'].items()}}


def flipped_gamma(params, scene, plan, rt):
    """Mutant: every pole's damping runs backwards -- ``Im(eps) < 0``, ``Re(eps)`` untouched.

    Written out here rather than taken from ``Pole.coefficients``, because the door refuses a
    negative ``gamma`` (rightly: it is gain). These are the same central differences with
    ``gamma -> -gamma``.
    """
    dt, poles = plan.dt, {}
    for index, structure in enumerate(scene.structures):
        name = f'poles[{index}]'
        if name not in params['poles']:
            continue
        gain, hold, drive = [], [], []
        for pole in structure.material.poles:
            omega_0 = float(getattr(pole, 'omega_0', 0.0))
            gamma = float(pole.gamma)
            strength = float(pole.omega_p ** 2 if isinstance(pole, fd.Drude) else pole.strength)
            scale = 1.0 / dt ** 2 - 0.5 * gamma / dt
            gain.append(np.array((2.0 / dt ** 2 - omega_0 ** 2) / scale))
            hold.append(np.array((1.0 / dt ** 2 + 0.5 * gamma / dt) / scale))
            drive.append(np.array(strength / scale))
        poles[name] = {**params['poles'][name], 'step_gain': tuple(gain), 'hold': tuple(hold),
                       'drive': tuple(drive)}
    return {**params, 'poles': poles}


@pytest.mark.parametrize('material', DISPERSIVE)
def test_a_pole_whose_contribution_is_negated_is_a_gain_medium_and_leaves_passivity(material):
    """The drive sign flipped: ``|R|`` reaches 5e2 to 3e6 where a passive half-space caps at 1."""
    real = reflection(material, cpw=20)
    mutant = reflection(material, cpw=20, mutant=flipped_drive)
    for lam in LAMS:
        assert abs(real[lam]) < 1.0, 'the real half-space is passive'
        assert abs(mutant[lam]) > 100.0, f'{lam}: mutant |R| = {abs(mutant[lam]):.3e}'


def flipped_errors(material):
    """``[(complex error, modulus error), ...]`` of the damping-flip mutant, per wavelength."""
    mutant = reflection(material, cpw=20, mutant=flipped_gamma)
    out = []
    for lam in LAMS:
        r, _ = fresnel(material, lam)
        out.append((abs(mutant[lam] - r) / abs(r), abs(abs(mutant[lam]) / abs(r) - 1)))
    return out


@pytest.mark.parametrize('material', DISPERSIVE)
def test_flipping_only_the_damping_sign_is_caught_where_the_loss_is_worth_catching(material):
    """Gain at the same ``Re(eps)``, and it is a frequency-by-frequency thing.

    ``Im(eps)`` flips and ``Re(eps)`` does not, so the mutant moves ``r`` by about
    ``2*Im(r)`` -- which is nothing where the medium is nearly lossless and everything where
    it is not. Measured at 20 cells per wavelength, complex error over ``|r|``: the Drude
    half-space gives 0.045 / 0.267 / 0.622 at 0.85 / 1.0 / 1.2 and the two-pole one gives
    0.362 / 0.195 / 0.128. The first of those is *inside* this gate's own tolerance: a
    one-frequency version of this gate would have passed the mutant, which is why there are
    three and why the assertion is on the worst.
    """
    errors = flipped_errors(material)
    assert max(c for c, _ in errors) > 3 * FRESNEL_TOLERANCE[20], errors
    assert min(c for c, _ in errors) < 4 * FRESNEL_TOLERANCE[20], (
        'the point of this test is that some frequency nearly hides it')


def test_the_phase_half_of_the_fresnel_gate_is_load_bearing():
    """Two of the two-pole medium's three frequencies are caught by the phase alone.

    At 1.0 and 1.2 the damping-flip mutant's *modulus* is 0.075 and 0.046 from the closed
    form -- inside the tolerance the gate allows at 20 cells per wavelength -- while the
    complex numbers are 0.195 and 0.128 away, because ``Im(r)`` changed sign. A gate that
    compared ``|R|`` would have passed the mutant at those two frequencies.
    """
    caught_only_by_phase = [(c, m) for c, m in flipped_errors(TWO_POLE)
                            if m < FRESNEL_TOLERANCE[20] < c]
    assert len(caught_only_by_phase) >= 2, flipped_errors(TWO_POLE)


# ---------------------------------------------------------------------------------------------
# Gate 4: a conductive medium, and the factor its own coefficients set
# ---------------------------------------------------------------------------------------------

CONDUCTOR_CPW, CONDUCTOR_LAMBDA = 20, 1.0
CONDUCTOR_DX = CONDUCTOR_LAMBDA / CONDUCTOR_CPW
CONDUCTOR_DT = 0.5 * CONDUCTOR_DX / math.sqrt(3.0)

#: ``sigma`` is chosen so that ``h = dt*sigma/(2*eps)`` is 0.0505: large enough that the
#: bilinear transform's ``h^3`` term is 0.85% over the run and can be asserted as a number,
#: small enough that 100 steps of decay stay eleven orders above round-off.
UNIFORM_EPS, UNIFORM_SIGMA, UNIFORM_STEPS = 1.0, 7.0, 100


def conductive_box(eps, sigma, initial, n=6, steps=100, monitors=None):
    """A periodic box of one uniform :class:`fd.Conductor`; no interface, no absorber."""
    L = n * CONDUCTOR_DX
    if monitors is None:
        monitors = (fd.Record('Ey', at=(0.0, 0.0, 0.0), component='Ey'),)
    return fd.Scene(
        domain=fd.Domain(size=(L, L, L)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic()),
        resolution=fd.Resolution(cells_per_wavelength=CONDUCTOR_CPW,
                                 wavelength=CONDUCTOR_LAMBDA),
        run=fd.Run(steps=steps, courant=0.5), initial=initial,
        background=fd.Conductor(eps=eps, sigma=sigma), monitors=monitors)


def uniform_scene(eps=UNIFORM_EPS, sigma=UNIFORM_SIGMA, n=6, steps=UNIFORM_STEPS):
    """A spatially uniform ``Ey``: ``curl E`` and ``curl H`` vanish, so only ``ae`` acts."""
    E = np.zeros((3, n, n, n))
    E[1] = 1.0
    return conductive_box(eps, sigma, fd.InitialField(E=E), n=n, steps=steps)


def one_value(array):
    """The single number a coefficient array holds, refusing one that is not uniform.

    A uniform conductive background lowers ``ae`` and ``be`` at shape ``(3, 1, 1, 1)`` -- the
    three components of one material -- and the gates below are written against the scalar
    they all carry. Reading it through here rather than through ``[0]`` means a change that
    made the three differ would fail loudly instead of being silently read off component x.
    """
    flat = np.asarray(to_numpy(array)).reshape(-1)
    assert np.all(flat == flat[0]), f'not one value: {np.unique(flat)}'
    return float(flat[0])


def half_rate(scene, plan):
    """``h = dt*sigma/(2*eps)``, read off the scene the way the coefficients are built."""
    background = scene.background
    return plan.dt * float(background.sigma[1]) / (2.0 * float(background.eps[1]))


@pytest.mark.parametrize('backend', backends())
def test_a_uniform_field_in_a_conductor_decays_by_the_exact_update_coefficient(backend):
    """``E^n = ae^n E^0`` to 7e-16, with ``ae`` taken from the lowered params themselves."""
    scene = uniform_scene()
    plan = fd.plan(scene)
    ae = one_value(lower(scene, plan, 'numpy')['ae'])
    h = half_rate(scene, plan)
    assert abs(ae - (1 - h) / (1 + h)) < 1e-15, 'the coefficient is the bilinear one'
    values = to_numpy(fd.run(scene, backend=backend)['Ey'].values)
    exact = values[0] * ae ** np.arange(len(values))
    assert abs(values[0]) > 0.9 and abs(values[-1] / values[0]) < 1e-4, 'it really decayed'
    assert np.max(np.abs(values / exact - 1)) < 1e-14


def test_the_gap_to_the_continuum_is_the_bilinear_transform_s_own_cubic_term():
    """``ln(ae) = -2h - 2h^3/3 - 2h^5/5 - ...``: asserted as an equality, not as smallness.

    The continuum answer for a uniform field is ``exp(-sigma t / eps) = exp(-2h)^n`` -- with
    no factor of two, which the wave case has because half of its energy is magnetic. The
    run is under it by 0.85% over 100 steps, and the per-step log gap matches the series to
    3e-6 relative. A tolerance would have passed for any ``h``; this does not.
    """
    scene = uniform_scene()
    plan = fd.plan(scene)
    h = half_rate(scene, plan)
    values = to_numpy(fd.run(scene)['Ey'].values)
    steps = len(values) - 1
    continuum = values[0] * math.exp(-2 * h * steps)
    measured = math.log(values[-1] / continuum) / steps
    predicted = -2 * (h ** 3 / 3 + h ** 5 / 5)
    assert measured < 0, 'the discrete factor decays faster than the continuum'
    assert abs(measured / predicted - 1) < 1e-4, f'{measured:g} against {predicted:g}'
    assert 0.005 < abs(values[-1] / continuum - 1) < 0.02, 'and the gap is worth asserting'


def _plane_wave(eps, sigma, n=8, steps=400, probe=(0.013, -0.007, 0.021)):
    """One discrete plane wave of the periodic conductive box, and its ``(scene, K)``."""
    L = n * CONDUCTOR_DX
    k = tuple(2 * math.pi / L for _ in range(3))
    K = tuple(2 * math.sin(ki * CONDUCTOR_DX / 2) / CONDUCTOR_DX for ki in k)
    norm = math.hypot(K[0], K[1])
    A = (K[1] / norm, -K[0] / norm, 0.0)
    curl = tuple(K[(c + 1) % 3] * A[(c + 2) % 3] - K[(c + 2) % 3] * A[(c + 1) % 3]
                 for c in range(3))

    def phase(x, y, z):
        return k[0] * x + k[1] * y + k[2] * z

    E = tuple((lambda c: (lambda x, y, z: A[c] * np.cos(phase(x, y, z))))(c) for c in range(3))
    H = tuple((lambda c: (lambda x, y, z: (CONDUCTOR_DT / 2) * (-curl[c])
                          * np.sin(phase(x, y, z))))(c) for c in range(3))
    scene = conductive_box(eps, sigma, fd.InitialField(E=E, H=H), n=n, steps=steps,
                           monitors=(fd.Record('Ey', at=probe, component='Ey'),))
    return scene, K


def wave_recurrence(scene, plan, K):
    """``(ae, B, theta)`` of ``E^{n+1} - B E^n + ae E^{n-1} = 0`` for the box's own plane wave.

    ``B = 1 + ae - be*bh*(Omega*dx)^2``: subtract the two E updates a step apart, substitute
    the H update, and use ``curl curl`` on a discrete eigenmode, whose eigenvalue is
    ``-(Omega*dx)^2 = -sum (2 sin(k_i dx/2))^2`` for the *bare* difference the curl carries.
    The roots are ``sqrt(ae) exp(+-i theta)``, so ``sqrt(ae)`` is the per-step amplitude --
    the product of the roots is ``ae``, and they are conjugates -- and ``theta/dt`` is the
    damped frequency. Every coefficient here is read from the lowered params, so the gate
    compares the run against its own update and not against a second copy of it.
    """
    params = lower(scene, plan, 'numpy')
    ae, be, bh = (one_value(params[name]) for name in ('ae', 'be', 'bh'))
    B = 1 + ae - be * bh * sum((Ki * plan.dx) ** 2 for Ki in K)
    return ae, B, math.acos(B / (2 * math.sqrt(ae)))


def rescaled_envelope(values, ae, theta):
    """The amplitude of ``values / sqrt(ae)**n``, from each adjacent pair; constant if exact."""
    u = np.asarray(values) / math.sqrt(ae) ** np.arange(len(values))
    squared = (u[:-1] ** 2 + u[1:] ** 2 - 2 * math.cos(theta) * u[:-1] * u[1:])
    return np.sqrt(squared) / math.sin(theta)


def test_a_wave_in_a_conductor_decays_at_the_square_root_of_the_same_factor():
    """The damped recurrence to 3e-16, and the rescaled envelope constant to 9e-15."""
    scene, K = _plane_wave(2.0, 0.3)
    plan = fd.plan(scene)
    ae, B, theta = wave_recurrence(scene, plan, K)
    values = to_numpy(fd.run(scene)['Ey'].values)
    residual = np.max(np.abs(values[2:] - B * values[1:-1] + ae * values[:-2]))
    assert residual / np.max(np.abs(values)) < 1e-14
    # The same statement twice over: rescaled by the predicted decay, the series is the
    # undamped recurrence of the stage-0 gates, and its amplitude does not move.
    rescaled = np.asarray(values) / math.sqrt(ae) ** np.arange(len(values))
    assert recurrence_residual(rescaled, theta / plan.dt, plan.dt) < 1e-14
    envelope = rescaled_envelope(values, ae, theta)
    assert np.ptp(envelope) / np.mean(envelope) < 1e-12
    decay = math.sqrt(ae) ** (len(values) - 1)
    assert 0.6 < decay < 0.7, 'and it lost a third of its amplitude over the run'


def sigma_on_the_h_update(params, scene, plan, rt):
    """Mutant: the same rate moved from ``ae``/``be`` to ``ah``/``bh``.

    A magnetic conductivity of the matched rate, which is what the ``Absorber`` grades in and
    what a reader might think ``sigma`` should do. The E update is left lossless.
    """
    eps = float(scene.background.eps[1])
    h = plan.dt * float(scene.background.sigma[1]) / (2.0 * eps)
    bh = np.array(plan.dt / plan.dx / (1 + h))
    # ``minus_bh`` is a lowered leaf, not something the step derives, so a mutant that moves
    # ``bh`` has to move it too or the H update keeps the real coefficient.
    return {**params,
            'ae': np.array(1.0), 'be': np.array(plan.dt / plan.dx / eps),
            'ah': np.array((1 - h) / (1 + h)),
            'bh': bh, 'minus_bh': -bh}


def test_sigma_moved_to_the_h_update_leaves_the_uniform_field_undamped():
    """The mutant, caught flat: with no curl there is nothing for a magnetic loss to act on."""
    scene = uniform_scene()
    values = to_numpy(run_mutated(scene, mutate_params=sigma_on_the_h_update)['Ey'].values)
    assert np.max(np.abs(values - values[0])) == 0.0, 'the mutant does not decay at all'
    real = to_numpy(fd.run(scene)['Ey'].values)
    assert abs(real[-1] / real[0]) < 1e-4


def test_sigma_moved_to_the_h_update_is_invisible_to_the_wave_probe():
    """And why, pinned: a matched magnetic rate is the same recurrence with the same roots.

    ``be*bh`` is unchanged by the move and ``ah`` takes the value ``ae`` had, so ``B`` and
    the product of the roots are identical and the E series decays at exactly the same
    ``sqrt(ae)``. The two runs differ only in the transient the initial H sets up -- 0.8% of
    the peak -- and in the wave impedance, which this probe does not read. A gate built on
    the wave alone would have passed this mutant, which is what the uniform probe is for.
    """
    scene, K = _plane_wave(2.0, 0.3)
    plan = fd.plan(scene)
    ae, B, theta = wave_recurrence(scene, plan, K)
    real = to_numpy(fd.run(scene)['Ey'].values)
    mutant = to_numpy(run_mutated(scene, mutate_params=sigma_on_the_h_update)['Ey'].values)
    residual = np.max(np.abs(mutant[2:] - B * mutant[1:-1] + ae * mutant[:-2]))
    assert residual / np.max(np.abs(mutant)) < 1e-14, 'the same recurrence, to round-off'
    envelope = rescaled_envelope(mutant, ae, theta)
    assert np.ptp(envelope) / np.mean(envelope) < 1e-12, 'the same decay, to round-off'
    difference = np.max(np.abs(mutant - real)) / np.max(np.abs(real))
    assert 1e-3 < difference < 5e-2, f'they differ only in the transient ({difference:.2e})'


# ---------------------------------------------------------------------------------------------
# Gate 5: the pole-aware CFL
# ---------------------------------------------------------------------------------------------

#: A coarse box with **no wavelength anywhere**, so the cells-per-wavelength and skin-depth
#: checks do not run (they warn that they did not) and the only door left is the CFL one.
#: Coarse on purpose: ``dt`` is large, so a modest ``omega_p`` breaks the bound and the pole
#: stays far from the metallic regime a finer grid would need.
CFL_DX = 0.5


def pole_scene(material, courant=0.5, n=4, steps=300, seed=0):
    rng = np.random.default_rng(seed)
    return fd.Scene(
        domain=fd.Domain(size=(n * CFL_DX,) * 3),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic()),
        resolution=fd.Resolution(dx=CFL_DX), run=fd.Run(steps=steps, courant=courant),
        initial=fd.InitialField(E=rng.standard_normal((3, n, n, n)) * 1e-3),
        structures=(fd.Structure(fd.Slab(z=(-n * CFL_DX / 2, 0.0)), material),),
        monitors=(fd.Record('p', at=(0.0, 0.0, -CFL_DX / 2), component='Ey'),))


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        warnings.simplefilter('ignore', fd.AbsorberProximityWarning)
        return fn(*args, **kwargs)


#: ``rate = omega_p^2 = 400``, so the bound is ``1/sqrt(1/0.2887^2 + 100) = 0.0945`` against
#: ``dt = 0.1443`` at courant 0.5. Without the strength term the bound would be the vacuum
#: one, 0.2887, and this scene would be accepted.
FAST_POLE = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=20.0, gamma=0.0),))


def test_a_concrete_pole_past_the_bound_is_refused_naming_the_bound_and_the_pole():
    scene = pole_scene(FAST_POLE)
    with pytest.raises(ValueError, match='stability bound') as raised:
        quiet(fd.plan, scene)
    message = str(raised.value)
    dt = scene.run.courant * CFL_DX / math.sqrt(3.0)
    bound = 1.0 / math.sqrt(3.0 / CFL_DX ** 2 + 400.0 / 4.0)
    assert f'{dt:g}' in message and f'{bound:g}' in message
    assert 'Drude(omega_p=20.0, gamma=0.0)' in message, 'the offending pole, by value'
    assert 'rate = max(w0^2) + max(sum strength/eps)' in message, 'and the formula'
    assert f'{bound / (CFL_DX / math.sqrt(3.0)):.3f}' in message, 'and the courant that fits'


def test_the_courant_the_refusal_names_is_accepted_and_runs_bounded():
    """The paired accepted spelling: the refusal's own number is a scene that plans and runs."""
    bound = 1.0 / math.sqrt(3.0 / CFL_DX ** 2 + 400.0 / 4.0)
    courant = float(f'{bound / (CFL_DX / math.sqrt(3.0)):.3f}')
    scene = pole_scene(FAST_POLE, courant=courant)
    plan = quiet(fd.plan, scene)
    assert plan.dt < bound
    assert checks.check_stability(scene, plan) is True
    values = to_numpy(quiet(fd.run, scene)['p'].values)
    assert np.all(np.isfinite(values)) and np.max(np.abs(values)) < 1e-2


def bound_without_the_strength_term(scene, plan):
    """The mutant of ``checks.cfl_bound``: ``rate = max(w0^2)`` and nothing else.

    Written out rather than patched, so the two bounds can be compared on the same scene in
    the same test. Everything else -- ``n_min`` over every material including the background,
    the ``1/sqrt(1/dt_vac^2 + rate/4)`` shape -- is the real one's.
    """
    materials = [scene.background] + [s.material for s in scene.structures]
    resonance = max([0.0] + [float(pole.rate()[0]) for m in materials for pole in m.poles])
    vacuum = min(m.index_bound() for m in materials) * plan.dx / math.sqrt(3.0)
    return 1.0 / math.sqrt(1.0 / vacuum ** 2 + resonance / 4.0)


def test_a_bound_without_the_strength_term_would_admit_a_run_that_explodes(monkeypatch):
    """The strength term is load-bearing: removing it accepts a scene that reaches 1e248.

    ``max(sum strength/eps)`` dropped leaves, for a Drude pole (``w0 = 0``), the vacuum bound
    untouched -- so the mutant door would not refuse this scene at all, and nothing about the
    shape of the formula would look wrong. What is behind the real refusal is measured by
    taking the door away: with ``check_stability`` stubbed out, the same scene at the same
    ``dt`` grows without bound, while the courant the real bound allows stays at 1e-3.
    """
    scene = pole_scene(FAST_POLE)
    monkeypatch.setattr(checks, 'check_stability', lambda scene, plan: True)
    plan = quiet(fd.plan, scene)
    real_bound, why = checks.cfl_bound(scene, plan)
    mutant_bound = bound_without_the_strength_term(scene, plan)
    assert why is None and abs(real_bound - 0.0944911) < 1e-6, 'the bound, by hand'
    assert abs(mutant_bound - CFL_DX / math.sqrt(3.0)) < 1e-15, 'the mutant is the vacuum bound'
    assert real_bound < plan.dt < mutant_bound, (
        'the real bound refuses this dt and the mutant bound accepts it')
    with np.errstate(over='ignore', invalid='ignore'):     # a divergence is the point here
        values = to_numpy(quiet(fd.run, scene)['p'].values)
    assert np.max(np.abs(values)) > 1e6, f'peak {np.max(np.abs(values)):.3e}'
    monkeypatch.undo()
    stable = quiet(fd.run, pole_scene(FAST_POLE, courant=0.3))
    assert np.max(np.abs(to_numpy(stable['p'].values))) < 1e-2


def test_a_traced_pole_cannot_be_read_at_the_door_so_the_run_returns_nan():
    """``check_stability`` answers ``False`` and the guard turns the whole run into ``nan``.

    A Torch tensor with ``requires_grad`` is the traced leaf here (``materials.concrete``
    refuses to read one, which is the same answer a JAX tracer gives). The scene is the one
    the concrete door refuses; the difference is that nothing here can know that, so the
    refusal becomes a ``nan`` the caller cannot mistake for a field.
    """
    torch = pytest.importorskip('torch')
    strength = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    traced = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=strength, gamma=0.0),))
    scene = pole_scene(traced)
    plan = quiet(fd.plan, scene)
    assert checks.cfl_bound(scene, plan) == (None, 'a pole value is traced, so the bound '
                                                   'cannot be read here')
    assert checks.check_stability(scene, plan) is False
    result = quiet(fd.run, scene, backend='torch')
    assert np.all(np.isnan(to_numpy(result['p'].values)))
    assert np.all(np.isnan(to_numpy(result.final.E)))


def test_a_traced_pole_inside_the_bound_runs_and_stays_finite():
    """The paired accepted spelling: the guard is ``1`` when the same comparison passes."""
    torch = pytest.importorskip('torch')
    strength = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    traced = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=strength, gamma=0.0),))
    scene = pole_scene(traced)
    assert checks.check_stability(scene, quiet(fd.plan, scene)) is False, 'still unreadable'
    values = to_numpy(quiet(fd.run, scene, backend='torch')['p'].values)
    assert np.all(np.isfinite(values)) and np.max(np.abs(values)) < 1e-2
