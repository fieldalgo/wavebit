"""A transmission-phase library for a pillar metasurface, with the checks a library needs.

TiO2 pillars (n = 2.43) on fused silica (n = 1.46), hexagonal pitch 350 nm, height 600 nm, at
532 nm with light arriving from the air side: the geometry family of Khorasaninejad et al.,
Science 352, 1190 (2016), and the worked case in ``docs/theory/rcwa/oblique-lattice.md``.

Each diameter is solved as an infinite periodic array, and the zero-order transmitted amplitude
is read at the substrate face. Its phase against diameter is the table a metalens design looks
up. This script builds that table and stops there: placing pillars from it (the local periodic
approximation), propagating to a focus and writing a layout are not part of Wavebit.

The checks below hold for this scene; they are not an accuracy bound for other structures.

* Lossless power balance at every diameter.
* Only the zero order propagates, in air and in the substrate, so the table is one complex
  number per pillar. The pitch condition is evaluated and the solved orders confirm it.
* The same pillars as an ``a x a*sqrt(3)`` rectangular supercell, a different Fourier
  truncation of one structure, give the same amplitude to truncation error.
* s and p transmit alike at normal incidence, as the lattice's sixfold symmetry requires.
* The usable entries cover a full 2*pi, their phase is monotonic in diameter, and each of
  eight phase levels has an entry nearer than half a level.
"""
import time

import numpy as np

from wavebit import Lattice, Material, Units, eps_from_refractive_index, rcwa

PITCH = 350.0            # nm, hexagonal
HEIGHT = 600.0           # nm
WAVELENGTH = 532.0       # nm
N_PILLAR = 2.43          # TiO2
N_SUBSTRATE = 1.46       # fused silica
DIAMETERS = np.arange(80.0, 261.0, 10.0)   # nm
ORDERS = (8, 8)          # 17 x 17 = 289 retained orders
SUPERCELL_ORDERS = (8, 14)                 # 17 x 29 = 493 retained orders
MIN_TRANSMISSION = 0.9   # entries below this sit on a resonance and are left out
LEVELS = 8


def _materials():
    # Datasheet indices are converted explicitly to relative permittivity.
    return (Material(epsilon=eps_from_refractive_index(N_PILLAR)),
            Material(epsilon=eps_from_refractive_index(N_SUBSTRATE)))


def _excitation(polarization):
    return rcwa.Excitation(wavelength=WAVELENGTH,
                           sources=(rcwa.PlaneWave(polarization=polarization),))


def hexagonal_problem(diameter, polarization="s"):
    """One pillar at the lattice point of the primitive hexagonal cell."""
    pillar, substrate = _materials()
    return rcwa.Problem(
        units=Units(length="nm"),
        stack=rcwa.Stack(lattice=Lattice.hexagonal(PITCH), bottom=substrate, layers=(
            rcwa.Layer(thickness=HEIGHT, regions=(
                rcwa.Region(geometry=rcwa.Circle(radius=diameter/2), material=pillar),)),)),
        excitation=_excitation(polarization),
    )


def supercell_problem(diameter, polarization="s"):
    """The same array as two pillars in an ``a x a*sqrt(3)`` rectangular cell."""
    pillar, substrate = _materials()
    cx, cy = PITCH/4, PITCH*np.sqrt(3)/4   # the two sites differ by (a/2, a*sqrt(3)/2)
    return rcwa.Problem(
        units=Units(length="nm"),
        stack=rcwa.Stack(period=(PITCH, PITCH*np.sqrt(3)), bottom=substrate, layers=(
            rcwa.Layer(thickness=HEIGHT, regions=(
                rcwa.Region(geometry=rcwa.Circle(radius=diameter/2, center=(cx, cy)),
                            material=pillar),
                rcwa.Region(geometry=rcwa.Circle(radius=diameter/2, center=(-cx, -cy)),
                            material=pillar),)),)),
        excitation=_excitation(polarization),
    )


def zero_order(result, polarization="s"):
    """Co-polarized zero-order transmitted amplitude, relative to the incident field."""
    order = result.transmission.order(0, 0)
    ratio = order.amplitude_ratio_s if polarization == "s" else order.amplitude_ratio_p
    return complex(np.asarray(ratio).reshape(-1)[0])


def nearest_entries(relative_phase, levels):
    """For each of ``levels`` equally spaced phases, the index of the nearest entry and its error."""
    targets = 2*np.pi*np.arange(levels)/levels
    wrapped = np.angle(np.exp(1j*(relative_phase[None, :] - targets[:, None])))
    index = np.abs(wrapped).argmin(axis=1)
    return index, wrapped[np.arange(levels), index]


def run():
    # Pitch condition: the shortest reciprocal vector of a hexagonal lattice is 4*pi/(sqrt(3)*a).
    # It must exceed the wavenumber in the denser half-space for every other order to be evanescent.
    g_min = 4*np.pi/(np.sqrt(3)*PITCH)
    k_substrate = 2*np.pi*N_SUBSTRATE/WAVELENGTH
    print(f"|G|_min = {g_min:.5f} /nm, k0*n_substrate = {k_substrate:.5f} /nm")
    assert g_min > k_substrate

    numerics = rcwa.Numerics(orders=ORDERS)
    amplitude = np.empty(DIAMETERS.size, dtype=complex)
    transmission = np.empty(DIAMETERS.size)
    started = time.perf_counter()
    for i, diameter in enumerate(DIAMETERS):
        result = rcwa.run(hexagonal_problem(float(diameter)), numerics=numerics)
        r, t = float(result.R.sum()), float(result.T.sum())
        np.testing.assert_allclose(r+t, 1., atol=1e-10, rtol=0)
        zero = result.transmission.order(0, 0)
        # Every other transmitted order is evanescent, so the zero order carries all of T.
        np.testing.assert_allclose(float(np.asarray(zero.efficiency).reshape(-1)[0]), t,
                                   atol=1e-12, rtol=0)
        amplitude[i], transmission[i] = zero_order(result), t
    elapsed = time.perf_counter()-started

    usable = transmission >= MIN_TRANSMISSION
    # e^{-i*omega*t}: a longer optical path adds phase, so the unwrapped phase rises with diameter.
    phase = np.unwrap(np.angle(amplitude[usable]))
    relative = phase-phase[0]
    print("diameter (nm)   T        relative phase (rad)")
    row = iter(relative)
    for diameter, t, keep in zip(DIAMETERS, transmission, usable):
        print(f"{diameter:8.0f}     {t:.4f}    " + (f"{next(row):.3f}" if keep else "left out (resonance)"))
    print(f"{usable.sum()} of {DIAMETERS.size} entries usable, worst T {transmission[usable].min():.3f}, "
          f"phase span {relative[-1]:.3f} rad = {relative[-1]/(2*np.pi):.2f} x 2*pi")
    print(f"sweep: {elapsed:.1f} s for {DIAMETERS.size} diameters at {(2*ORDERS[0]+1)*(2*ORDERS[1]+1)} orders "
          "(this machine, NumPy backend; not a benchmark)")
    assert np.all(np.diff(relative) > 0)
    assert relative[-1] >= 2*np.pi

    index, error = nearest_entries(relative, LEVELS)
    print(f"{LEVELS} phase levels -> diameters (nm):", DIAMETERS[usable][index].astype(int).tolist())
    print("level errors (rad):", np.round(error, 3).tolist())
    # An entry farther than half a level from its target would leave that level unrepresented.
    assert np.abs(error).max() < np.pi/LEVELS

    # Two independent statements about one diameter, away from the resonances.
    diameter = 150.0
    reference = amplitude[DIAMETERS == diameter][0]

    # Sixfold symmetry makes the normal-incidence response polarization independent. The
    # parallelogram of retained orders does not share that symmetry, so s and p differ at a
    # finite truncation: 3.0e-3 in amplitude at these 289 orders, 4.8e-4 at 841.
    p_amplitude = zero_order(rcwa.run(hexagonal_problem(diameter, "p"), numerics=numerics), "p")
    print(f"s against p at {diameter:.0f} nm: |difference| = {abs(reference-p_amplitude):.1e}")
    assert abs(reference-p_amplitude) < 6e-3

    # The rectangular supercell holds the same pillars under a different truncation
    # (1.1e-3 in amplitude here). It needs more orders for the same resolution, which is the
    # cost the primitive cell avoids.
    supercell = rcwa.run(supercell_problem(diameter), numerics=rcwa.Numerics(orders=SUPERCELL_ORDERS))
    np.testing.assert_allclose(float(supercell.R.sum()+supercell.T.sum()), 1., atol=1e-10, rtol=0)
    supercell_amplitude = zero_order(supercell)
    print(f"primitive cell against supercell at {diameter:.0f} nm: "
          f"|difference| = {abs(reference-supercell_amplitude):.1e}")
    assert abs(reference-supercell_amplitude) < 2e-3
    return DIAMETERS[usable], relative, transmission[usable]


if __name__ == "__main__":
    run()
