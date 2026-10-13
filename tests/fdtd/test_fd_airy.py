"""Stage 1, gate 2 -- a dielectric slab: ``r(w)`` and ``t(w)`` against the Airy closed form.

**The method, and the reference run that makes it a measurement.** A broadband Gaussian packet
(``fd_helpers.Packet``, a superposition of exact discrete eigenmodes) travels in ``+z`` at a
slab of ``fd.Material(eps=...)`` in a box with ``PML`` on ``z`` and a one-cell transverse period.
Two ``fd.Record`` probes, half a wavelength in front of the slab and half a wavelength behind
it, record ``Ey`` once per step -- the engine only records; the DFT is in this file, because DFT
monitors are stage 3. The same scene **without the structure** is run as well, and that
reference run is what supplies the incident spectrum:

    incident(w)     = DFT of the reference run at the front probe
    reflected(w)    = DFT of (slab run - reference run) at the front probe
    transmitted(w)  = DFT of the slab run at the back probe

The subtraction is exact rather than approximate: the two runs differ only in the slab's
coefficients, they start from the same initial field on the same cells, and the front probe is
in front of the slab, so ``slab - reference`` there is the reflected wave and nothing else. A
window would not do instead -- the packet is 2.5 widths long and the probe is half a wavelength
out, so the incident and the reflected halves overlap in time by construction.
:func:`test_a_slab_of_vacuum_reflects_exactly_nothing` is the control that says the subtraction
measures the slab and not the difference between two runs.

**The closed forms, and which index closes which gap.** Two are compared against, and the
distance between them is the whole content of this gate.

*The continuum Airy formula*, ``r = (r1 + r2 e^{2ib}) / (1 + r1 r2 e^{2ib})`` and
``t = t1 t2 e^{ib} / (1 + r1 r2 e^{2ib})`` with the admittances ``Y = n/mu`` -- textbook, and
what a reader means by "the Airy formula". Compared against it, the measurement needs **two
corrections, and neither is optional**:

1. ``b = k_medium * d`` and every external propagation phase use the **discrete dispersion
   relation's** wavenumber, ``sin(k dx/2) = n (dx/dt) sin(w dt/2)``, not ``k = n w``. At 20
   cells per wavelength that is a 2.9% error in ``k`` and it is not subtle:
   :func:`test_the_reference_run_propagates_at_the_discrete_wavenumber_and_not_omega` measures
   the reference run's own phase between the two probes and finds the discrete relation to
   4e-7 and the continuum one to 5.6e-2, a factor of 1.3e5.
2. The **reference planes sit half a cell low**. ``eps`` lives on E nodes (``Ey`` is at the
   z node of its cell), the slab's cells ``k0 .. k1-1`` give their ``eps`` to the nodes
   ``k0 .. k1-1``, and node ``k0``'s share of Ampere's law is the dual cell
   ``[z_k0 - dx/2, z_k0 + dx/2]``. Giving that whole dual cell the slab's ``eps`` -- which is
   what cell-snapping without subpixel averaging *means* -- puts the effective interface at
   ``z_k0 - dx/2``. The thickness is unmoved; the slab is.

With both, the continuum formula agrees to **6.2% in |r| and 0.005 rad in arg r, 0.6% in
|t| and 0.005 rad in arg t** at 20 cells per wavelength over the band, and the error falls
**second order in dx** (6.2e-2 / 1.5e-2 / 3.6e-3 at 20 / 40 / 80 cells per wavelength, the
relative error of the complex ``r``). That is the tolerance this gate achieves against the
textbook
formula and :func:`test_the_continuum_airy_error_falls_second_order_in_dx` is why it is a
tolerance and not a fudge: a defect does not converge.

*The exact discrete Airy formula* is the same problem solved on the grid the engine runs.
Eliminating H from the two updates leaves one three-term recurrence per node,

    E[j+1] - 2E[j] + E[j-1] + (Omega^2 eps[j] - Kx^2) dx^2 E[j] = 0,  Omega = 2 sin(w dt/2)/dt

which is the semi-discrete Helmholtz equation (``Kx = 0`` at normal incidence, ``mu = 1``
throughout; the two mutants are measured against the unmutated slab's answer and need no form of
their own). ``eps[j]`` steps at ``j = 0`` and ``j = N``.
Matching an incoming-plus-reflected form on ``j <= 0``, a two-mode form on ``-1 <= j <= N`` and
an outgoing form on ``j >= N-1`` is four equations in ``(r, C, D, T)`` -- :func:`airy_discrete`,
which solves them and nothing else. It is a closed form, not a second stepper: it never
advances anything in time. Against it the measurement agrees to **5.6e-8 to 4.4e-7 absolute in
``r`` and ``t``, amplitude and phase** (3.3e-7 to 2.8e-6 read as a fraction, the largest of
those on a slab whose ``|r|`` dips to 0.028 at the band edge), for every slab tried, and the
floor is the PML rather than anything about the slab. Its reference planes are the nodes
``k0`` and ``k1`` exactly, with no half-cell correction -- the correction above is the
first-order reading of the phase this formula carries, which is why the two agree on where the
interface is to 1% of a cell while disagreeing by 6% on how much it reflects.

So: the continuum formula is the physics and converges to it; the discrete one is the same
physics on this grid and pins it to a part in a million. A gate with only the first could not
tell a 6% defect from the staircase; a gate with only the second would never notice that the
staircase converges to Maxwell at all.

**One oblique case, at 30 degrees**, where ``Ey`` is the s polarisation alone and the angle is
a Bloch phase on a one-cell x period. It is the only scene in this gate that puts a material and
a ``Bloch`` face together, and it is what makes the wavenumber a genuine ``kz``: the transverse
``Kx^2`` is a constant of the recurrence, so the same four equations answer, but ``kz`` is no
longer ``n`` times anything and a formula that had quietly stayed ``n omega`` would part company
there. Its band is three entries rather than five, because tilting the packet keeps ``kz``
Gaussian while ``omega = |k|``, which squeezes the spectrum toward the carrier.

**The mutants, and why both of them are caught by the reflection alone.**

``the slab shifted one cell``
    ``r`` acquires ``exp(2 i k dx)``: 0.48 to 0.77 in relative error against 3e-7 for the real
    slab, over a million times the agreement.
``the slab's eps applied to mu instead``
    ``n = sqrt(eps mu)`` is unchanged, so the slab is optically the same thickness; the
    admittance ``Y = n/mu`` is inverted, so every reflection coefficient changes sign. Relative
    error in ``r`` about 2 -- the two answers differ by a half turn.

**The transmission is blind to both** -- it moves by 2e-8, which is round-off between two runs
whose coefficient arrays differ -- and that is not an accident to be worked around: ``t`` is
invariant under a translation of the slab, and ``t1 t2`` and ``r1 r2`` are both invariant under
``Y -> 1/Y``. :func:`test_the_transmitted_wave_is_blind_to_both_mutants` states it, because it
is the reason this gate has to compare **phase** and not only ``|R|`` and ``|T|`` -- a gate that
checked the two power fractions would pass both mutants and pass a slab in the wrong place.

**Cost.** The default measurement is a 1 x 1 x 251 box for 1305 steps, twice (279 x 1729 at
30 degrees, and 443 x 2610 / 828 x 5219 for the two refinements); the whole file runs in about
12 s on NumPy and nothing is marked slow. The box is long because the packet has to be
clear of both the slab and the ``z-`` layer at ``t = 0`` (the envelope is 1e-7 at four widths)
and the layers are 24 cells because the band reaches 1.6 wavelengths, where 12 cells is only
0.4 of one.
"""
import functools
import math

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from tests.fixtures.fdtd.fd_helpers import Packet, backends, to_numpy

#: The carrier the packet is built around, and the unit every length below is quoted in.
WAVELENGTH = 1.0

#: Where ``r`` and ``t`` are compared. Vacuum wavelengths, all inside the packet's band: the
#: incident spectrum is at least 0.34 of its peak at every one of them, which
#: :func:`test_the_incident_spectrum_is_well_above_noise_across_the_band` measures rather than
#: assumes. The band is deliberately wider on the short side, where the grid dispersion is
#: worst and the continuum formula's error is largest.
BAND = (0.80, 0.90, 1.00, 1.15, 1.30)

#: The band of the one oblique case. Narrower than :data:`BAND` because tilting the packet
#: spreads its spectrum in ``omega`` -- only ``kz`` is Gaussian and ``omega`` is ``|k|`` -- so
#: 0.8 and 1.3 fall to 0.17 and 0.21 of the peak there while these three stay above 0.63.
BAND_OBLIQUE = (0.90, 1.00, 1.15)

#: The default resolution of this gate, in cells per vacuum wavelength at the carrier.
CPW = 20

#: How closely the measurement follows :func:`airy_discrete`, as an **absolute** error on the
#: complex ``r`` and ``t``. The floor is the PML (24 cells, 2e-8 to 1e-6 across this band), the
#: packet's truncated tails and the window, and it is an absolute floor rather than a relative
#: one: measured 5.6e-8 to 5.8e-7 over every case below, whatever ``|r|`` happens to be there.
DISCRETE_TOL = 2e-6

#: The same agreement read as a fraction of the answer. Looser than :data:`DISCRETE_TOL`
#: because one case (``eps = 4``, six cells at 30 cells per wavelength, the short end of the
#: band) sits on a reflection null where ``|r| = 0.028``, and dividing the same 1.5e-7 by that
#: is 2.8e-6. Both are
#: asserted, so neither a small ``|r|`` nor a large one can hide a defect.
DISCRETE_REL = 1e-5

#: How closely it follows the continuum Airy formula at :data:`CPW`, once the discrete
#: wavenumber and the half-cell interface offset are applied. This is the staircase error the
#: design says is there, and it is second order in ``dx`` -- see the convergence test.
CONTINUUM_TOL = dict(abs_r=0.07, arg_r=0.01, abs_t=0.01, arg_t=0.01)


# ---------------------------------------------------------------------------------------------
# The two closed forms
# ---------------------------------------------------------------------------------------------

def discrete_wavenumber(omega, n, dx, dt, kx=0.0):
    """``kz`` of a wave of angular frequency ``omega`` in a medium of index ``n``, on the grid.

    The Yee dispersion relation, ``eps mu sin^2(w dt/2)/dt^2 = sum_a sin^2(k_a dx/2)/dx^2``,
    solved for the one component along z::

        sin(kz dx/2) = (dx/2) sqrt((n Omega)^2 - Kx^2),  Omega = 2 sin(w dt/2)/dt,
                                                         Kx    = 2 sin(kx dx/2)/dx

    ``Kx`` is the grid's transverse wavenumber, which a one-cell x period with a Bloch phase
    ``phi = kx dx`` realises exactly (``2 sin(phi/2)/dx``), so the oblique case below costs a
    1 x 1 x N box and approximates nothing. At ``kx = 0`` this is
    ``sin(kz dx/2) = n (dx/dt) sin(w dt/2)``.

    This is the *only* place a wavenumber is made in this file: ``k = n omega`` is the continuum
    limit of it and appears nowhere except in the test that measures what dropping it costs.
    """
    transverse = 2 * math.sin(kx * dx / 2) / dx
    big_omega = 2 * np.sin(np.asarray(omega) * dt / 2) / dt
    return (2.0 / dx) * np.arcsin((dx / 2) * np.sqrt((n * big_omega) ** 2 - transverse ** 2))


def airy_continuum(n, mu, beta):
    """``(r, t)`` of a slab of admittance ``n/mu`` and one-way phase ``beta``, at its own faces.

    The textbook formulas, in the admittance form that keeps ``mu`` visible (``Y = n/mu``, so
    ``eps`` and ``mu`` swapped inverts every coefficient's sign without moving ``n``)::

        r = (r1 + r2 e^{2ib}) / (1 + r1 r2 e^{2ib})      r1 = (Y0 - Y)/(Y0 + Y),  r2 = -r1
        t = t1 t2 e^{ib} / (1 + r1 r2 e^{2ib})           t1 = 2Y0/(Y0 + Y), t2 = 2Y/(Y0 + Y)

    ``beta`` is the caller's: pass ``k_medium * d`` with ``k_medium`` from
    :func:`discrete_wavenumber` and this becomes the grid's own answer to second order.
    """
    Y0, Y = 1.0, n / mu
    r1 = (Y0 - Y) / (Y0 + Y)
    r2 = -r1
    t1, t2 = 2 * Y0 / (Y0 + Y), 2 * Y / (Y0 + Y)
    phase = np.exp(2j * np.asarray(beta))
    denominator = 1 + r1 * r2 * phase
    return (r1 + r2 * phase) / denominator, t1 * t2 * np.exp(1j * np.asarray(beta)) / denominator


def airy_discrete(omega, n, cells, dx, dt, kx=0.0):
    """``(r, t)`` of a ``cells``-cell slab, exactly, on the semi-discrete Yee grid.

    Eliminating H between the two updates leaves ``E[j+1] - 2E[j] + E[j-1] + (Omega^2 eps[j]
    - Kx^2) dx^2 E[j] = 0`` with ``Omega = 2 sin(w dt/2)/dt``. A slab occupying the cells
    ``0 .. cells-1`` gives its ``eps`` to the E nodes ``0 .. cells-1``, so the coefficient steps
    at ``j = 0`` and at ``j = cells``. Writing ``u = e^{i k1 dx}`` and ``v = e^{i k2 dx}`` with
    both wavenumbers from :func:`discrete_wavenumber`, the solution is

        j <= 0:            u^j + r u^-j
        -1 <= j <= cells:  C v^j + D v^-j
        j >= cells-1:      T u^(j - cells)

    each form extending one node past its region because the recurrence at the edge node reads
    across it. Agreement of the three forms at ``j = -1, 0`` and at ``j = cells-1, cells`` is
    four linear equations in ``(r, C, D, T)``, solved here. ``r`` is referenced to node 0 and
    ``t`` to node ``cells`` -- the requested faces, with no half-cell offset.

    ``kx`` enters only through the two wavenumbers, because ``Kx^2`` is a constant of the
    recurrence and not of the medium: the oblique s-polarised slab is the same four equations
    with ``kz`` in place of ``k``, which is why 30 degrees below costs one more pair of runs and
    no second formula.

    Nothing is stepped: this is a closed form of the scheme, and it is what lets the gate ask
    for a part in a million instead of the staircase's 6%.
    """
    u = np.exp(1j * discrete_wavenumber(omega, 1.0, dx, dt, kx) * dx)
    v = np.exp(1j * discrete_wavenumber(omega, n, dx, dt, kx) * dx)
    r, t = [], []
    for ui, vi in zip(np.atleast_1d(u), np.atleast_1d(v)):
        #                r        C                  D                     T
        matrix = np.array([[ui, -1 / vi, -vi, 0.0],
                           [1.0, -1.0, -1.0, 0.0],
                           [0.0, vi ** (cells - 1), vi ** -(cells - 1), -1 / ui],
                           [0.0, vi ** cells, vi ** -cells, -1.0]], dtype=complex)
        rhs = np.array([-1 / ui, -1.0, 0.0, 0.0], dtype=complex)
        solution = np.linalg.solve(matrix, rhs)
        r.append(solution[0])
        t.append(solution[3])
    return np.array(r), np.array(t)


# ---------------------------------------------------------------------------------------------
# The scene, the reference run, and the extraction
# ---------------------------------------------------------------------------------------------

def slab_pair(cells=4, eps=2.25, magnetic=False, shift=0, cpw=CPW, angle=0.0, pml=24,
              courant=0.5, probe_offset=0.5, lead_widths=4.0, back_widths=3.0, body=None):
    """``(slab_scene, reference_scene, geometry)``: the same packet, with and without the slab.

    ``cells`` is the slab's thickness in cells, so nothing is ever snapped and the gate measures
    the physics rather than the snapping (``tests/fdtd/test_fd_materials_doors.py`` measures the
    snapping). ``magnetic=True`` puts ``eps`` on ``mu`` instead -- the second mutant -- and
    ``shift`` moves the slab by that many cells without moving the reference planes, which is
    the first.

    ``body`` (a number of cells, 0.0 for faces on node planes) spells the same slab as the
    smoothed ``fd.Box`` -- stage 4's C4, gate G5 -- with its faces that far past the node planes
    the ``Slab`` would take; the box runs ten cells across the one-cell transverse period, so it
    is uniform along it. ``front`` and ``back`` are then the box's faces.

    ``angle`` tilts the packet by putting a Bloch phase ``kx dx`` on a **one-cell** x period,
    which is exact rather than cheap (see :func:`discrete_wavenumber`) and makes the run
    complex. ``Ey`` is then the s polarisation alone: ``kx`` lies along x, so the packet's p
    half has no y component, the probes read one polarisation and the closed form is the
    s-polarised one.

    The layout, in units of the packet's 1/e width ``W = 8/(k cos theta)`` (1.27 wavelengths at
    the carrier, 1.47 at 30 degrees): ``pml`` cells of layer, ``back_widths * W`` of run-up, the
    packet's centre, ``lead_widths * W``, the slab, the back probe, ``pml`` cells of layer. Four
    widths puts the packet's head at 1e-7 of its peak where the slab begins, which is what makes
    "nothing is inside the slab at t = 0" true to the accuracy this gate asks for; three widths
    behind puts 1e-4 of it inside the ``z-`` layer, which that layer absorbs.
    """
    dx = WAVELENGTH / cpw
    dt = courant * dx / math.sqrt(3.0)
    theta = math.radians(angle)
    width = 8.0 / (2 * math.pi / WAVELENGTH * math.cos(theta))
    run_up = int(math.ceil(back_widths * width / dx))
    lead = int(math.ceil(lead_widths * width / dx))
    first = pml + run_up + lead                                  # the slab's first cell
    tail = int(math.ceil(probe_offset * WAVELENGTH / dx)) + 10
    nz = first + cells + tail + pml
    z0 = (pml + run_up) * dx
    front, back = first * dx, (first + cells) * dx
    packet = Packet(WAVELENGTH, angle, dx, dt, z0=z0, real=(angle == 0.0))
    # the packet closes on the slab at cos(theta), so a tilted run is longer in steps as well
    steps = int(math.ceil(((front - (z0 - 4.0 * width)) + 2 * probe_offset * WAVELENGTH
                           + 6.0 * width) / (dt * math.cos(theta))))
    material = fd.Material(eps=1.0, mu=eps) if magnetic else fd.Material(eps=eps)
    low = (first + shift) * dx
    phase = packet.kx * dx
    across = fd.Periodic() if phase == 0.0 else fd.Bloch(phase=phase)
    monitors = (fd.Record('front', at=(0.0, 0.0, front - probe_offset * WAVELENGTH),
                          component='Ey'),
                fd.Record('back', at=(0.0, 0.0, back + probe_offset * WAVELENGTH),
                          component='Ey'))

    def scene(structures):
        return fd.Scene(domain=fd.Domain(size=(dx, dx, nz * dx),
                                         center=(0.0, 0.0, nz * dx / 2)),
                        boundaries=fd.Boundaries(x=across, y=fd.Periodic(),
                                                 z=fd.PML(cells=pml, wavelength=WAVELENGTH)),
                        resolution=fd.Resolution(dx=dx),
                        run=fd.Run(steps=steps, courant=courant),
                        initial=packet.initial(), structures=structures, monitors=monitors)

    slab = fd.Structure(fd.Slab(z=(low, low + cells * dx)), material)
    if body is not None:
        front, back = front + body * dx, back + body * dx
        slab = fd.Structure(fd.Box(size=(10 * dx, 10 * dx, cells * dx),
                                   center=(0.0, 0.0, (front + back) / 2)), material)
    geometry = dict(dx=dx, dt=dt, width=width, nz=nz, steps=steps, front=front, back=back,
                    cells=cells, eps=eps, index=math.sqrt(eps), kx=packet.kx, angle=angle,
                    first=first)
    return scene((slab,)), scene(()), geometry


def spectrum(series, omega):
    """``sum_n f(t_n) exp(+i omega t_n)`` -- the ``e^{-i w t}`` convention's forward amplitude.

    In the test, not in the engine: stage 1 records a time series and nothing more.
    """
    values = np.asarray(to_numpy(series.values))
    t = np.asarray(series.t)
    return np.array([complex(np.sum(values * np.exp(1j * w * t))) for w in np.atleast_1d(omega)])


def probe_z(results, name):
    """The z coordinate of the Yee point a ``Record`` actually landed on."""
    observer = next(o for o in results.plan.observers if o.name == name)
    return float(results.plan.grid.yee_points(observer.field, observer.c)[2][observer.k])


def extract(slab, reference, geometry, omega):
    """``(r, t)`` at each ``omega``, referenced to the slab's requested faces.

    With the incident wave ``A e^{i k (z - z_front)}`` and the reference run carrying exactly
    it, the reflected wave at the front probe is ``r A e^{i k (z_front - z_probe)}`` while the
    reference there is ``A e^{i k (z_probe - z_front)}``, so their ratio is ``r`` times
    ``e^{2 i k (z_front - z_probe)}``; behind the slab the transmitted wave is
    ``t A e^{i k (z - z_back)}`` against a reference of ``A e^{i k (z - z_front)}``, whose ratio
    is ``t e^{-i k d}``. Dividing by the reference run at the *same* probe is what cancels the
    packet's own spectrum and every propagation phase but those two.
    """
    k = discrete_wavenumber(omega, 1.0, geometry['dx'], geometry['dt'], geometry['kx'])
    thickness = geometry['back'] - geometry['front']
    front, front_reference = spectrum(slab['front'], omega), spectrum(reference['front'], omega)
    back, back_reference = spectrum(slab['back'], omega), spectrum(reference['back'], omega)
    r = ((front - front_reference) / front_reference
         * np.exp(2j * k * (probe_z(slab, 'front') - geometry['front'])))
    t = back / back_reference * np.exp(1j * k * thickness)
    return r, t


def band_of(angle):
    """:data:`BAND` at normal incidence, :data:`BAND_OBLIQUE` off it."""
    return np.array(BAND if angle == 0.0 else BAND_OBLIQUE)


@functools.lru_cache(maxsize=None)
def measured(cells=4, eps=2.25, magnetic=False, shift=0, cpw=CPW, angle=0.0, backend='numpy',
             body=None):
    """``(r, t, geometry)`` over the band for one slab -- both runs, cached by argument."""
    slab, reference, geometry = slab_pair(cells=cells, eps=eps, magnetic=magnetic, shift=shift,
                                          cpw=cpw, angle=angle, body=body)
    got = fd.run(slab, backend=backend)
    want = fd.run(reference, backend=backend)
    omega = 2 * math.pi / band_of(angle)
    r, t = extract(got, want, geometry, omega)
    return r, t, geometry


def airy_profile(omega, eps_nodes, first, dx, dt, front, back):
    """``(r, t)`` for ``eps`` given node by node, exactly, on the semi-discrete Yee grid.

    The same recurrence as :func:`airy_discrete`, ``E[j+1] - 2E[j] + E[j-1] + Omega^2 eps[j]
    dx^2 E[j] = 0`` at normal incidence, with ``eps[j]`` any profile on the nodes ``first ..
    first + len(eps_nodes) - 1`` and vacuum elsewhere -- which is what a smoothed interface
    lowers to, one mixed node at each face. It is marched rather than solved: from the
    transmitted side, where the field is ``u^((j - back)/dx)`` with ``T = 1``, down through the
    profile to two vacuum nodes in front, where ``A u^(j - front) + B u^-(j - front)`` fits them;
    ``r = B/A`` and ``t = 1/A``, referenced to ``front`` and ``back`` (lengths on the z axis).
    On the snapped profile it is :func:`airy_discrete` to 7e-16, and it knows nothing of the
    engine's painters: the profile is the test's own.
    """
    out_r, out_t = [], []
    last = first + len(eps_nodes) - 1
    for w in np.atleast_1d(omega):
        big = 2 * math.sin(w * dt / 2) / dt
        u = np.exp(1j * float(discrete_wavenumber(w, 1.0, dx, dt)) * dx)
        field = {j: u ** (j - back / dx) for j in (last + 3, last + 4)}
        for j in range(last + 3, first - 3, -1):
            e = eps_nodes[j - first] if first <= j <= last else 1.0
            field[j - 1] = 2 * field[j] - field[j + 1] - big ** 2 * e * dx ** 2 * field[j]
        a, b = first - 3, first - 2
        f = front / dx
        coefficients = np.linalg.solve(np.array([[u ** (a - f), u ** (f - a)],
                                                 [u ** (b - f), u ** (f - b)]]),
                                       np.array([field[a], field[b]]))
        out_r.append(coefficients[1] / coefficients[0])
        out_t.append(1 / coefficients[0])
    return np.array(out_r), np.array(out_t)


def box_profile(geometry):
    """``(eps_nodes, first)``: what the smoothed box's faces give ``Ey`` on its z nodes, by hand.

    ``Ey`` is tangential to both faces, so its normal moment is zero and the Farjadpour/Kottke
    rule is the arithmetic mean over its dual cell ``[z_j - dx/2, z_j + dx/2]`` -- the overlap of
    that interval with ``[front, back]``, written out here.
    """
    dx, eps = geometry['dx'], geometry['eps']
    front, back = geometry['front'] / dx, geometry['back'] / dx
    first, last = math.floor(front - 0.5), math.ceil(back + 0.5)
    fill = [max(0.0, min(back, j + 0.5) - max(front, j - 0.5)) for j in range(first, last + 1)]
    return [f * eps + (1 - f) for f in fill], first


def reference_of(cells=4, eps=2.25, cpw=CPW, angle=0.0):
    """:func:`airy_discrete` for that slab over the band."""
    geometry = slab_pair(cells=cells, eps=eps, cpw=cpw, angle=angle)[2]
    return airy_discrete(2 * math.pi / band_of(angle), geometry['index'], cells,
                         geometry['dx'], geometry['dt'], geometry['kx'])


def relative(got, want):
    """Largest ``|got - want| / |want|`` over the band."""
    return float(np.max(np.abs(np.asarray(got) - np.asarray(want)) / np.abs(np.asarray(want))))


def absolute(got, want):
    """Largest ``|got - want|`` over the band."""
    return float(np.max(np.abs(np.asarray(got) - np.asarray(want))))


#: ``(cells, eps, cpw)`` of the slabs this gate measures. Four cells at eps = 2.25 is the
#: default; seven puts the band across the half-wave null (``|r|`` drops to 0.08, where the
#: Airy denominator does the work and a wrong ``beta`` cannot hide); ten is thicker than the
#: shortest wavelength in the band; 1.44 is a weak contrast and 4.0 a strong one, the latter at
#: 30 cells per wavelength so that the medium still has 15 and no CoarseMeshWarning is due.
SLABS = [(4, 2.25, CPW), (7, 2.25, CPW), (10, 2.25, CPW), (6, 1.44, CPW), (6, 4.0, 30)]


# ---------------------------------------------------------------------------------------------
# The reference run
# ---------------------------------------------------------------------------------------------

def test_the_reference_run_propagates_at_the_discrete_wavenumber_and_not_omega():
    """The incident spectrum's phase between the two probes is the grid's dispersion relation.

    This is the measurement behind correction (1) in the module docstring, made on the run that
    supplies the incident spectrum and before any slab is involved: the ratio of the reference
    run's two probes is ``e^{i k L}`` over the distance between them, and only the discrete
    ``k`` fits. It also says the reference run carries a pure forward wave -- a backward
    component would show as an error here, not as a phase offset.
    """
    slab, reference, geometry = slab_pair()
    want = fd.run(reference, backend='numpy')
    omega = 2 * math.pi / np.array(BAND)
    distance = probe_z(want, 'back') - probe_z(want, 'front')
    ratio = spectrum(want['back'], omega) / spectrum(want['front'], omega)
    discrete = np.abs(ratio * np.exp(-1j * discrete_wavenumber(omega, 1.0, geometry['dx'],
                                                               geometry['dt']) * distance) - 1)
    continuum = np.abs(ratio * np.exp(-1j * omega * distance) - 1)
    assert float(np.max(discrete)) < 1e-5, discrete
    assert float(np.min(continuum)) > 1e-2, continuum
    # and the gap between them is what makes the choice a measurement and not a preference
    assert float(np.min(continuum)) / float(np.max(discrete)) > 1e3


def test_the_incident_spectrum_is_well_above_noise_across_the_band():
    """Every wavelength in :data:`BAND` carries at least a third of the packet's peak amplitude.

    The band is not a free choice: outside the packet's spectrum ``r`` and ``t`` are a ratio of
    two small numbers and the gate would be measuring its own round-off. This states where the
    edge is rather than trusting the packet's docstring for it.
    """
    slab, reference, geometry = slab_pair()
    want = fd.run(reference, backend='numpy')
    sweep = 2 * math.pi / np.linspace(0.6, 2.0, 200)
    peak = float(np.max(np.abs(spectrum(want['front'], sweep))))
    band = np.abs(spectrum(want['front'], 2 * math.pi / np.array(BAND)))
    assert float(np.min(band)) / peak > 0.3, band / peak


def test_a_slab_of_vacuum_reflects_exactly_nothing():
    """The control on the subtraction: a structure of ``eps = 1`` scatters zero, to the bit.

    If ``slab - reference`` at the front probe carried anything but the slab's reflection -- a
    difference in the two runs' arithmetic, a boundary that saw the structure list rather than
    the structure -- this would not be zero. It is the answer to "how would this check look if
    the thing it measures were broken", and it is exact rather than small because a permittivity
    of one lowers to the same coefficient a scene with no structure does.
    """
    slab, reference, geometry = slab_pair(eps=1.0)
    got, want = fd.run(slab, backend='numpy'), fd.run(reference, backend='numpy')
    omega = 2 * math.pi / np.array(BAND)
    r, t = extract(got, want, geometry, omega)
    assert float(np.max(np.abs(r))) == 0.0
    # t is referenced to the two faces, so a vacuum slab transmits the propagation phase across
    # its own thickness and not 1 -- which is exactly what airy_discrete says for n = 1.
    thickness = geometry['back'] - geometry['front']
    k = discrete_wavenumber(omega, 1.0, geometry['dx'], geometry['dt'])
    assert relative(t, np.exp(1j * k * thickness)) < 1e-12
    assert relative(t, airy_discrete(omega, 1.0, geometry['cells'], geometry['dx'],
                                     geometry['dt'])[1]) < 1e-12


# ---------------------------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('cells, eps, cpw', SLABS)
def test_the_slab_matches_the_exact_discrete_airy_formula(cells, eps, cpw):
    """``r`` and ``t``, amplitude and phase, against :func:`airy_discrete` to a part in a million.

    The tight half of the gate. Its reference planes are the requested faces with no offset, its
    wavenumbers are the grid's, and it is a closed form -- so what is left over is the PML, the
    packet's tails and the window, which is where :data:`DISCRETE_TOL` comes from.
    """
    r, t, geometry = measured(cells=cells, eps=eps, cpw=cpw)
    r_want, t_want = reference_of(cells=cells, eps=eps, cpw=cpw)
    assert absolute(r, r_want) < DISCRETE_TOL
    assert absolute(t, t_want) < DISCRETE_TOL
    assert relative(r, r_want) < DISCRETE_REL
    assert relative(t, t_want) < DISCRETE_REL
    # stated separately, because a complex tolerance can be met by an amplitude that is wrong
    # in the direction the phase is wrong
    assert float(np.max(np.abs(np.abs(r) - np.abs(r_want)))) < DISCRETE_TOL
    assert float(np.max(np.abs(np.angle(r / r_want)))) < DISCRETE_REL
    assert float(np.max(np.abs(np.abs(t) - np.abs(t_want)))) < DISCRETE_TOL
    assert float(np.max(np.abs(np.angle(t / t_want)))) < DISCRETE_REL


def test_the_slab_matches_the_s_polarised_airy_formula_at_thirty_degrees():
    """The same slab at 30 degrees, against both closed forms, with ``Ey`` as the s wave.

    Worth one more pair of runs because it is the only place in this gate where a material and a
    ``Bloch`` face are in the same scene, and because it makes the wavenumber the gate uses a
    genuine ``kz``: the transverse ``Kx`` enters the recurrence as a constant, so the same four
    equations answer, but ``kz`` is no longer ``n`` times anything and a formula that had
    quietly been ``n omega`` all along would part company here.

    The s-polarised continuum coefficients are the same :func:`airy_continuum` with the
    admittance ratio written as ``kz_medium / kz_vacuum`` -- that is what ``Y = n/mu`` becomes
    off normal -- and the same half-cell offset. Measured: 5.8e-7 absolute against the discrete
    form, and against the continuum one 2.5% in ``|r|`` with 0.002 rad of phase. The same
    staircase as at normal incidence, and a smaller fraction of a reflection that is half again
    as large at this angle.
    """
    angle = 30.0
    r, t, geometry = measured(angle=angle)
    r_want, t_want = reference_of(angle=angle)
    assert absolute(r, r_want) < DISCRETE_TOL
    assert absolute(t, t_want) < DISCRETE_TOL
    assert float(np.max(np.abs(np.angle(r / r_want)))) < DISCRETE_REL
    assert float(np.max(np.abs(np.angle(t / t_want)))) < DISCRETE_REL
    omega = 2 * math.pi / band_of(angle)
    dx, dt, kx = geometry['dx'], geometry['dt'], geometry['kx']
    thickness = geometry['back'] - geometry['front']
    kz = discrete_wavenumber(omega, 1.0, dx, dt, kx)
    kz_medium = discrete_wavenumber(omega, geometry['index'], dx, dt, kx)
    continuum = airy_continuum(kz_medium / kz, 1.0, kz_medium * thickness)
    r_continuum = continuum[0] * np.exp(-2j * kz * dx / 2)
    assert float(np.max(np.abs(np.abs(r) - np.abs(r_continuum)) / np.abs(r_continuum))) \
        < CONTINUUM_TOL['abs_r']
    assert float(np.max(np.abs(np.angle(r / r_continuum)))) < CONTINUUM_TOL['arg_r']
    assert float(np.max(np.abs(np.angle(t / continuum[1])))) < CONTINUUM_TOL['arg_t']


def test_the_oblique_band_is_well_above_noise_and_the_normal_one_would_not_be():
    """Why :data:`BAND_OBLIQUE` is three entries and not five.

    Tilting the packet keeps ``kz`` Gaussian but ``omega`` is ``sqrt(kx^2 + kz^2)``, so the
    spectrum in ``omega`` is squeezed toward the carrier: at 30 degrees the outer two entries of
    :data:`BAND` carry 0.17 and 0.21 of the peak where the three kept entries carry 0.63 and up.
    Stated rather than assumed, because a band chosen out of habit is how a gate ends up
    measuring its own round-off.
    """
    slab, reference, geometry = slab_pair(angle=30.0)
    want = fd.run(reference, backend='numpy')
    sweep = 2 * math.pi / np.linspace(0.6, 2.2, 240)
    peak = float(np.max(np.abs(spectrum(want['front'], sweep))))
    kept = np.abs(spectrum(want['front'], 2 * math.pi / np.array(BAND_OBLIQUE))) / peak
    dropped = np.abs(spectrum(want['front'], 2 * math.pi / np.array([BAND[0], BAND[-1]]))) / peak
    assert float(np.min(kept)) > 0.6, kept
    assert float(np.max(dropped)) < 0.3, dropped


def test_the_slab_matches_the_continuum_airy_formula_within_the_staircase_error():
    """The textbook formula, with the discrete wavenumber and the half-cell interface offset.

    Amplitude and phase separately, because they fail differently: the phase agrees to 0.005 rad
    (the half-cell offset is right to about 1% of a cell) while ``|r|`` is 6% high -- a
    cell-snapped interface reflects more than the interface it stands for, and no reference
    plane can fix an amplitude. What dropping either correction costs is measured next door, in
    ``test_the_half_cell_interface_offset_is_what_carries_the_reflected_phase``.
    """
    r, t, geometry = measured()
    omega = 2 * math.pi / np.array(BAND)
    dx, dt = geometry['dx'], geometry['dt']
    thickness = geometry['back'] - geometry['front']
    beta = discrete_wavenumber(omega, geometry['index'], dx, dt) * thickness
    k = discrete_wavenumber(omega, 1.0, dx, dt)
    r_want, t_want = airy_continuum(geometry['index'], 1.0, beta)
    r_want = r_want * np.exp(-2j * k * dx / 2)          # both faces sit half a cell low
    assert float(np.max(np.abs(np.abs(r) - np.abs(r_want)) / np.abs(r_want))) \
        < CONTINUUM_TOL['abs_r']
    assert float(np.max(np.abs(np.angle(r / r_want)))) < CONTINUUM_TOL['arg_r']
    assert float(np.max(np.abs(np.abs(t) - np.abs(t_want)) / np.abs(t_want))) \
        < CONTINUUM_TOL['abs_t']
    assert float(np.max(np.abs(np.angle(t / t_want)))) < CONTINUUM_TOL['arg_t']


def test_the_half_cell_interface_offset_is_what_carries_the_reflected_phase():
    """Dropping it costs 0.40 rad, 79 times the 0.005 that is left with it.

    The wavenumber's correction is measured here too, and it is honestly the smaller of the two
    *for this quantity*: with the offset applied, using ``k = n w`` instead of the grid's ``k``
    costs 0.027 rad, five times the residual and not eighty. The reason is cancellation -- the
    slab is 0.2 wavelengths thick, so the continuum error in ``beta`` and the continuum error in
    the half-cell offset are comparable and of opposite sign, and a thicker slab or a longer
    external path separates them again. The place the wavenumber choice is unmissable is
    :func:`test_the_reference_run_propagates_at_the_discrete_wavenumber_and_not_omega`, where
    the two differ by 1.3e5 over a path of a wavelength and a bit.

    Worth stating, because it is a trap: dropping the wavenumber correction makes ``|r|`` look
    *better* (3.3% against 6.2%), so an amplitude-only reading of this gate would conclude that
    the continuum wavenumber is the right one.
    """
    r, t, geometry = measured()
    omega = 2 * math.pi / np.array(BAND)
    dx, dt = geometry['dx'], geometry['dt']
    thickness = geometry['back'] - geometry['front']
    k = discrete_wavenumber(omega, 1.0, dx, dt)
    corrected = airy_continuum(geometry['index'],
                               1.0,
                               discrete_wavenumber(omega, geometry['index'], dx, dt) * thickness
                               )[0] * np.exp(-2j * k * dx / 2)
    unshifted = corrected * np.exp(+2j * k * dx / 2)
    continuum_k = airy_continuum(geometry['index'], 1.0,
                                 geometry['index'] * omega * thickness)[0] \
        * np.exp(-1j * omega * dx)
    good = float(np.max(np.abs(np.angle(r / corrected))))
    assert good < CONTINUUM_TOL['arg_r']
    assert float(np.max(np.abs(np.angle(r / unshifted)))) > 50 * good
    assert float(np.max(np.abs(np.angle(r / continuum_k)))) > 4 * good
    assert float(np.max(np.abs(np.abs(r) - np.abs(continuum_k)) / np.abs(continuum_k))) \
        < float(np.max(np.abs(np.abs(r) - np.abs(corrected)) / np.abs(corrected)))


def test_the_continuum_airy_error_falls_second_order_in_dx():
    """6.2e-2 / 1.5e-2 / 3.6e-3 at 20 / 40 / 80 cells per wavelength: the staircase converges.

    The slab is the same 0.2 wavelengths at every resolution (4, 8 and 16 cells), so this is a
    refinement in ``dx`` and not a change of problem. Second order is more than the design
    promises -- ``geometry.py`` calls a snapped interface first order in ``dx`` -- because here
    the interface lands exactly on a cell boundary and only its half-cell dual-cell share is
    wrong, and that error is symmetric. It is the reason the 6% above is a tolerance rather than
    a defect: a defect does not fall by four every time the cell halves.
    """
    errors = []
    for cpw in (20, 40, 80):
        cells = 4 * cpw // CPW
        r, t, geometry = measured(cells=cells, cpw=cpw)
        omega = 2 * math.pi / np.array(BAND)
        dx, dt = geometry['dx'], geometry['dt']
        thickness = geometry['back'] - geometry['front']
        beta = discrete_wavenumber(omega, geometry['index'], dx, dt) * thickness
        k = discrete_wavenumber(omega, 1.0, dx, dt)
        want = airy_continuum(geometry['index'], 1.0, beta)[0] * np.exp(-2j * k * dx / 2)
        errors.append(relative(r, want))
    assert errors[0] > 3e-2, errors
    for coarse, fine in zip(errors, errors[1:]):
        assert 3.2 < coarse / fine < 5.0, errors


@pytest.mark.parametrize('backend', backends())
def test_the_three_backends_measure_the_same_slab(backend):
    """NumPy, JAX and Torch agree on ``r`` and ``t`` to round-off, and all three meet the gate."""
    r, t, geometry = measured(backend=backend)
    r_numpy, t_numpy, _ = measured(backend='numpy')
    assert relative(r, r_numpy) < 1e-12
    assert relative(t, t_numpy) < 1e-12
    r_want, t_want = reference_of()
    assert absolute(r, r_want) < DISCRETE_TOL
    assert absolute(t, t_want) < DISCRETE_TOL


# ---------------------------------------------------------------------------------------------
# The mutants
# ---------------------------------------------------------------------------------------------

def test_the_slab_shifted_one_cell_is_caught_by_the_reflected_phase():
    """A one-cell move leaves every amplitude alone and turns ``r`` by ``2 k dx``.

    0.48 to 0.77 in relative error across the band against 3e-7 for the slab in its place. The
    amplitudes ``|r|`` and ``|t|`` do not move at all, which is why this mutant would survive a
    gate that compared only the two power fractions.
    """
    r, t, geometry = measured(shift=1)
    r_want, t_want = reference_of()
    assert relative(r, r_want) > 0.4
    assert float(np.max(np.abs(np.abs(r) - np.abs(r_want)))) < DISCRETE_TOL
    assert float(np.min(np.abs(np.angle(r / r_want)))) > 0.4


def test_the_permittivity_applied_to_the_permeability_is_caught_by_the_reflected_sign():
    """``eps`` on ``mu`` keeps ``n = sqrt(eps mu)`` and inverts ``Y = n/mu``: every sign flips.

    Relative error about 2 -- the two answers differ by close to a half turn -- against 3e-7 for
    the real slab. As with the shift, ``|r|`` barely moves: the mutant is a phase.
    """
    r, t, geometry = measured(magnetic=True)
    r_want, t_want = reference_of()
    assert relative(r, r_want) > 1.5
    assert float(np.min(np.abs(np.angle(r / r_want)))) > 2.5


def test_the_transmitted_wave_is_blind_to_both_mutants():
    """``t`` moves by 2e-8 under a translation of the slab and under ``eps`` <-> ``mu``.

    That is round-off between two runs whose coefficient arrays differ, not a response: both
    mutants still pass :data:`DISCRETE_TOL` against the *unmutated* slab's closed form, a
    hundred times inside the gate. Not a weakness to route around but the reason this gate
    compares phase. A translation multiplies the incident and the transmitted waves by the same
    phase, and ``t1 t2`` and ``r1 r2`` are both invariant under ``Y -> 1/Y``; so a gate that
    asked only for ``|T|``, or for ``|R|`` and ``|T|``, would pass a slab in the wrong place
    *and* a slab whose permittivity had been written into the permeability.
    """
    r_want, t_want = reference_of()
    for mutant in (dict(shift=1), dict(magnetic=True)):
        r, t, _ = measured(**mutant)
        assert relative(t, measured()[1]) < 1e-6, mutant
        assert absolute(t, t_want) < DISCRETE_TOL, mutant
        # while the reflection separates every one of them by more than five decades
        assert relative(r, r_want) > 1e5 * relative(measured()[0], r_want), mutant


# ---------------------------------------------------------------------------------------------
# G5 (stage 4, C4): the same slab spelled as the smoothed fd.Box
# ---------------------------------------------------------------------------------------------

def _continuum(geometry, omega):
    dx, dt = geometry['dx'], geometry['dt']
    beta = discrete_wavenumber(omega, geometry['index'], dx, dt) * (geometry['back']
                                                                    - geometry['front'])
    return airy_continuum(geometry['index'], 1.0, beta)


@pytest.mark.parametrize('cells, eps, cpw', SLABS)
def test_the_smoothed_box_matches_the_exact_discrete_profile(cells, eps, cpw):
    """G5: a smoothed box lowers its faces to one mixed node each, and runs that profile exactly.

    Faces on node planes, so the node under each face holds half the slab and half vacuum:
    ``Ey`` is tangential, the rule is the arithmetic mean, and the node reads ``(eps + 1)/2``
    -- :func:`box_profile`, by hand. :func:`airy_profile` solves the scheme on that profile,
    and the run agrees with it to the discrete gate's own tolerance on every slab of
    :data:`SLABS` (measured worst 4.3e-7 absolute, 3.2e-6 relative): the mixed node is where
    the lowering put it, with the value the rule gives.

    Mutants, each alone, measured 2026-09-26 over this file: the rule mixing a tangential
    component harmonically fails every body case but the half-cell control (whose fills are all
    0 or 1, where both rules agree); the public box handed to the kernel 0.04 of a cell high
    fails every body case; an unturned box planned a cross term fails every body case (its
    initial field is then refused).
    """
    r, t, geometry = measured(cells=cells, eps=eps, cpw=cpw, body=0.0)
    profile, first = box_profile(geometry)
    omega = 2 * math.pi / np.array(BAND)
    r_want, t_want = airy_profile(omega, profile, first, geometry['dx'], geometry['dt'],
                                  geometry['front'], geometry['back'])
    assert absolute(r, r_want) < DISCRETE_TOL, absolute(r, r_want)
    assert absolute(t, t_want) < DISCRETE_TOL, absolute(t, t_want)
    assert relative(r, r_want) < DISCRETE_REL, relative(r, r_want)
    assert relative(t, t_want) < DISCRETE_REL, relative(t, t_want)


def test_the_smoothed_box_needs_no_reference_plane_correction():
    """G5: the box's faces are where they were asked; the slab's sit half a cell low.

    Against the continuum Airy formula with the discrete wavenumber, the snapped slab needs its
    reference planes moved half a cell (:func:`test_the_half_cell_interface_offset_is_what_carries_the_reflected_phase`:
    0.005 rad with it, 0.40 without). The same slab spelled as the smoothed box, faces on the
    same node planes, needs **no** correction: within the same :data:`CONTINUUM_TOL` with none,
    and the slab's correction applied to it is the wrong answer by far more than 50 times --
    the gate's mutant, run here. Predicted beforehand from :func:`airy_profile` on
    :func:`box_profile`: 6.19% in ``|r|``, 0.0048 rad in ``arg r``, 0.57% in ``|t|``, and
    0.3999 rad with the correction (83x). Measured: 6.1931%, 0.00480 rad, 0.5723%, and
    0.39985 rad -- the slab, corrected, reads 6.1616%, 0.00508, 0.5970%.

    Mutants: the three of :func:`test_the_smoothed_box_matches_the_exact_discrete_profile`
    each fail this case too.
    """
    r, t, geometry = measured(body=0.0)
    omega = 2 * math.pi / np.array(BAND)
    k = discrete_wavenumber(omega, 1.0, geometry['dx'], geometry['dt'])
    r_want, t_want = _continuum(geometry, omega)
    assert float(np.max(np.abs(np.abs(r) - np.abs(r_want)) / np.abs(r_want))) \
        < CONTINUUM_TOL['abs_r']
    good = float(np.max(np.abs(np.angle(r / r_want))))
    assert good < CONTINUUM_TOL['arg_r']
    assert float(np.max(np.abs(np.abs(t) - np.abs(t_want)) / np.abs(t_want))) \
        < CONTINUUM_TOL['abs_t']
    assert float(np.max(np.abs(np.angle(t / t_want)))) < CONTINUUM_TOL['arg_t']
    corrected = r_want * np.exp(-2j * k * geometry['dx'] / 2)
    assert float(np.max(np.abs(np.angle(r / corrected)))) > 50 * good


def test_a_box_with_its_faces_half_a_cell_low_is_the_slab():
    """The control: faces at ``z_k0 - dx/2`` fill every node whole or not at all -- the slab.

    Every ``Ey`` dual cell is then wholly inside the box or wholly out of it, so the box lowers
    ``Ey`` as the snapped slab does, node for node; ``Ey`` is the only component the packet
    drives, so the two runs are the same run to round-off, and both probes record the same
    series. It is the half-cell reference plane of the snapped slab, pinned from the other
    side. (The two ``r`` differ by the phase of that half cell only because each is referenced
    to its own asked-for face.)

    Not bit for bit, and measured rather than assumed: on the four nodes whose dual cell ends
    exactly on a face the exact clip reads the fill as one to within its own round-off, not one
    itself (the reason ``checks._CLIP_ROUNDOFF`` exists), so the box's ``Ey`` coefficient there
    is 1.6e-15 off the slab's, and the two series differ by at most 7.0e-15 of their peak in
    front and 2.5e-15 behind. The gate is 1e-13 of the peak; a face half a cell out of place
    moves them by order one.
    """
    slab, _, _ = slab_pair()
    box, _, geometry = slab_pair(body=-0.5)
    profile, _ = box_profile(geometry)
    assert set(profile) <= {1.0, geometry['eps']}
    got, want = fd.run(box, backend='numpy'), fd.run(slab, backend='numpy')
    for probe in ('front', 'back'):
        a = np.asarray(to_numpy(got[probe].values))
        b = np.asarray(to_numpy(want[probe].values))
        assert np.max(np.abs(a - b)) <= 1e-13 * np.max(np.abs(b)), probe


@pytest.mark.parametrize('offset', (0.1, 0.25, 0.3, 0.7))
def test_a_smoothed_box_off_the_grid_matches_its_profile(offset):
    """G5 off the node planes: a face anywhere in a cell is one fractionally filled node.

    The continuum formula is not the oracle here -- a face between nodes costs up to 0.019 rad
    in ``arg r`` at 20 cells per wavelength against it, over the 0.01 its tolerance allows --
    but the exact discrete profile is, and it is met to the discrete tolerance at every offset.
    """
    r, t, geometry = measured(body=offset)
    profile, first = box_profile(geometry)
    assert any(0.0 < (value - 1.0) / (geometry['eps'] - 1.0) < 1.0 for value in profile)
    omega = 2 * math.pi / np.array(BAND)
    r_want, t_want = airy_profile(omega, profile, first, geometry['dx'], geometry['dt'],
                                  geometry['front'], geometry['back'])
    assert absolute(r, r_want) < DISCRETE_TOL, (offset, absolute(r, r_want))
    assert relative(t, t_want) < DISCRETE_REL, (offset, relative(t, t_want))


def test_the_smoothed_box_converges_to_the_continuum_at_second_order():
    """The box's complex ``r`` against the continuum formula, with no correction, at 20/40/80.

    The same 0.2-wavelength slab at every resolution, faces on node planes: the error falls by
    about four each time the cell halves, as the snapped slab's does with its correction
    (:func:`test_the_continuum_airy_error_falls_second_order_in_dx`) -- and without one.
    """
    errors = []
    for cpw in (20, 40, 80):
        cells = 4 * cpw // CPW
        r, _, geometry = measured(cells=cells, cpw=cpw, body=0.0)
        omega = 2 * math.pi / np.array(BAND)
        errors.append(relative(r, _continuum(geometry, omega)[0]))
    assert errors[0] > 3e-2, errors
    for coarse, fine in zip(errors, errors[1:]):
        assert 3.2 < coarse / fine < 5.0, errors
