"""How far a diameter error moves a pillar's phase: the derivative of a phase-library entry.

The scene is the TiO2 pillar array of ``metalens_phase_library.py`` (hexagonal pitch 350 nm,
height 600 nm, 532 nm). A fabricated pillar is never exactly the drawn one, and the number a
tolerance budget needs is d(phase)/d(diameter) at each library entry. Wavebit differentiates
the solve with respect to the circle's radius, so that number comes from one reverse pass
rather than from a second library.

Check: the automatic derivative on JAX against a central finite difference of the NumPy
backend's own solves, at two diameters. The two share no differentiation code, and they agree
to the finite difference's truncation error.

Requires ``wavebit[jax]``. The derivative is taken in eager execution; a circle under
``jax.jit`` needs ``Numerics(topology_check='host')``, described in
``docs/guide/rcwa/guarded-shape-jit.md``.
"""
import numpy as np

from wavebit import Lattice, Material, Units, eps_from_refractive_index, rcwa

PITCH = 350.0            # nm, hexagonal
HEIGHT = 600.0           # nm
WAVELENGTH = 532.0       # nm
N_PILLAR = 2.43          # TiO2
N_SUBSTRATE = 1.46       # fused silica
ORDERS = (8, 8)
DIAMETERS = (150.0, 240.0)   # nm, both away from the library's resonances
STEP = 0.05                  # nm, central finite-difference half step


def problem(diameter):
    # Datasheet indices are converted explicitly to relative permittivity.
    pillar = Material(epsilon=eps_from_refractive_index(N_PILLAR))
    substrate = Material(epsilon=eps_from_refractive_index(N_SUBSTRATE))
    return rcwa.Problem(
        units=Units(length="nm"),
        stack=rcwa.Stack(lattice=Lattice.hexagonal(PITCH), bottom=substrate, layers=(
            rcwa.Layer(thickness=HEIGHT, regions=(
                rcwa.Region(geometry=rcwa.Circle(radius=diameter/2), material=pillar),)),)),
        excitation=rcwa.Excitation(wavelength=WAVELENGTH, sources=(rcwa.PlaneWave(polarization="s"),)),
    )


def run():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    numerics = rcwa.Numerics(orders=ORDERS)

    def phase(diameter, backend):
        result = rcwa.run(problem(diameter), numerics=numerics, backend=backend)
        ratio = result.transmission.order(0, 0).amplitude_ratio_s
        xp = jnp if backend == "jax" else np
        return xp.angle(xp.reshape(xp.asarray(ratio), (-1,))[0])

    slopes = {}
    print("diameter (nm)   d(phase)/d(diameter) (rad/nm)   finite difference   relative gap")
    for diameter in DIAMETERS:
        automatic = float(jax.grad(lambda d: phase(d, "jax"))(jnp.asarray(diameter)))
        above, below = (float(phase(diameter+sign*STEP, "numpy")) for sign in (1., -1.))
        difference = float(np.angle(np.exp(1j*(above-below))))/(2*STEP)
        gap = abs(automatic-difference)/abs(difference)
        print(f"{diameter:8.0f}        {automatic:.6f}                       {difference:.6f}            {gap:.1e}")
        # Measured 1e-7 at 150 nm; the bound leaves room for the difference's own truncation.
        assert gap < 1e-5
        slopes[diameter] = automatic
    for diameter, slope in slopes.items():
        print(f"a 5 nm diameter error at {diameter:.0f} nm moves the phase by {5*slope:.2f} rad")
    return slopes


if __name__ == "__main__":
    run()
