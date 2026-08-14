"""Cases shared by ``gen_goldens_fmmax_patterned.py`` and ``test_fmmax_patterned_golden.py``.

Same split as ``_fmmax_cases.py`` and for the same reason: the generator runs in ``fwref``
(which has fmmax and no wavebit) and the test in the project env (which has the reverse), so
the one thing they share has to import numpy and nothing else. A copy that must stay identical
is a copy that will not.

These are the **patterned** anisotropic cases — a gyrotropic pillar in vacuum, which is what
``pn_conv_matrices`` was written for and what nothing in the repo could arbitrate before.
``gen_goldens_fmmax.py`` anticipated exactly this file: "None of the cases below are patterned
yet, because nothing in the solver can express one, but the wiring will be here when that
changes."

**The raster axis convention is the thing to get right here, and it is not symmetric.** This
project stores a layer as ``A[y, x]``; fmmax's array axis 0 runs along its first lattice vector,
which in this geometry is X. The same physical layer is therefore ``A.T`` to fmmax — a
relabelling of which axis means x, not a mirror, so the tensor *components* are untouched (both
solvers work in lab-frame x/y). The generator transposes; this module states the layer in this
project's convention.

That was measured, not assumed: with the transpose missing, both this project's factorizations
— the new one and the shipped isotropic NVF that predates it — disagreed with fmmax with the
s/p split *reversed*, and transposing moved fmmax onto them to within its own formulation
spread in all three contrasts. Which is also why the pillar below is deliberately **not
square**: on a square pillar the transpose is invisible and the check would pass while wrong.
"""
import numpy as np

WL = 620.0
PERIOD = 340.0
THICK = 185.0
GRID = 96
THETA = 8.0          # degrees; must be oblique, see the module docstring on the transpose

# name -> (n_in, gyrotropy g). The tensor is eps = n_in^2 I + i g (e_z x) inside the pillar and
# vacuum outside, so it is Hermitian and the layer is lossless: R + T = 1 is a real constraint.
CASES = {
    'gyro_pillar_n115_g010': (1.15, 0.10j),
    'gyro_pillar_n130_g020': (1.30, 0.20j),
    'gyro_pillar_n160_g030': (1.60, 0.30j),
}

# Fourier orders the golden is frozen at. High, because the oracle is a *converged* comparison:
# fmmax's own two formulations (FFT, the plain direct rule, and JONES_DIRECT, a vector one) are
# still 1.7e-4 apart at fto=14 on the strongest case, and no claim tighter than that spread
# means anything.
FTO = 14


def pillar(value, background, grid=GRID):
    """The layer in **this project's** ``[y, x]`` convention. The generator transposes it.

    Not square: wider in x than in y. A square pillar would hide exactly the axis mix-up this
    pair of files exists to pin down.
    """
    a = np.full((grid, grid), background, dtype=complex)
    a[grid // 3:2 * grid // 3, grid // 4:2 * grid // 3] = value
    return a


def tensor_rasters(n_in, g, grid=GRID):
    """``(eps, gyro)`` rasters: the scalar permittivity and the ``eps_xy`` field.

    The physical tensor is ``eps_xy = +gyro``, ``eps_yx = -gyro``. Slot negation happens at the
    caller — see ``docs/off-diagonal-slot-convention.md``.
    """
    return pillar(n_in ** 2, 1.0, grid), pillar(g, 0.0, grid)
