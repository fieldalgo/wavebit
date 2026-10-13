"""Cases for the RETICOLO **conical field** golden, shared by the generator and the test.

The companion to ``_reticolo_field_cases.py``, one grating type along. That one covers the pure
TE/TM route (``grating_type == 0``, ``field_1d``); this one covers ``field_1d_conical``
(``grating_type == 1``) — a 1D grating at a **non-zero azimuth**, where s and p no longer
decouple and the reconstruction returns all six components from a single coupled mode set.

**Why it needed its own oracle.** ``test_s4_field_golden.py`` runs the conical path, but only on
*uniform* stacks. ``test_fmmax_field_golden.py`` reaches a patterned layer through the **2D**
route. So a patterned layer on the conical path — the case with an azimuth *and* a
discontinuity, where the s/p mixing and the factorization have to be right at the same time —
was checked by nothing outside this repository.

What carries over from the pure 1D golden, unchanged and re-verified here rather than assumed:
the ``.m`` samples the same ``linspace(0, period, 11)``, ``res3`` lays a cell out the same way
(``RES_Z`` rows per profile entry, ambients included — :func:`layer_rows` is imported rather
than restated), RETICOLO's ``+z`` still points out of the stack -- the public frame's direction
too since 2026-09-21 -- and the reversal into the top-to-bottom depth frame of ``STACKS`` still
carries the declared frame's z-parity. What is new is the *shape*: ``res3``'s conical branch
returns ``(z, x, y, 6)`` — **x before y**, the transpose of this project's ``(z, y, x, 6)`` —
and six components instead of three, so the parity has six entries.

**Two slots are identically zero on a uniform slab and must be treated as such.** For pure s
incidence ``E_z`` vanishes (E is perpendicular to the plane of incidence, which has no z
component); for pure p, ``H_z`` does. A relative gap against zero is meaningless, so the test
requires ours to vanish too rather than dividing.
"""
import numpy as np

from tests.references._reticolo_field_cases import RES_X, RES_Z, layer_rows  # noqa: F401  (re-exported)

WL = 620.0
PERIOD = 400.0
MAX_ORDER = 8                    # 17 retained orders
NX = 40                    # raster columns; 10 nm cells
_CELL = PERIOD / NX

NAMES = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')

# Under z -> -z: `E_z`, `H_x`, `H_y` change sign, `E_t` and `H_z` do not. The same rule the
# pure 1D layouts carry, written out over the full six slots.
Z_PARITY = np.array([1.0, 1.0, -1.0, -1.0, -1.0, 1.0])

# `E_x` is the component normal to the grating's vertical interfaces, so it is the discontinuous
# one and the two codes resolve it differently at any finite truncation. Slot 0.
SLOW_SLOT = 0

# Identically zero on an unpatterned slab: slot 2 (`E_z`) for s, slot 5 (`H_z`) for p. On a
# patterned layer both are populated, so this only applies to the `uniform` stack.
ZERO_SLOT_UNIFORM = {'s': 2, 'p': 5}


def _bar(lo_nm, hi_nm, eps_bg, eps_hi):
    """One binary bar per period, given by its edges in nm. Edges avoid the x samples."""
    row = np.full(NX, eps_bg, dtype=complex)
    row[int(round(lo_nm / _CELL)):int(round(hi_nm / _CELL))] = eps_hi
    return row


# name -> (thicknesses, structure), of relative **permittivity** shaped (layer, 1, NX) as this
# project rasters a 1D cell, layers listed top to bottom as in the pure 1D list (the tests
# reverse them into the public bottom-to-top order). The same four shapes the pure 1D golden
# uses, so a disagreement that appears only here is about the conical coupling rather than
# about the geometry -- and ``test_at_zero_azimuth_the_two_reticolo_goldens_agree`` asserts the
# two lists still hold the same arrays, which is what keeps "the same shapes" true rather than
# merely intended.
#
# Permittivity since 2026-08-20, and identical to the pure 1D list value for value: each is the
# square of the index frozen into the golden, so the scene has not moved. The untidy entries are
# `1.3 ** 2` and `1.6 ** 2` as doubles, not roundings of them.
STACKS = {
    'uniform': ([260.0], np.full((1, 1, NX), 2.25, dtype=complex)),
    'grating': ([260.0], _bar(50.0, 230.0, 1.0, 4.0).reshape(1, 1, NX)),
    'bilayer': ([150.0, 260.0], np.stack([_bar(50.0, 230.0, 1.0, 4.0),
                                          _bar(130.0, 330.0, 1.6900000000000002,
                                               2.5600000000000005)])[:, None, :]),
    'lossy': ([260.0], _bar(50.0, 230.0, 1.0, 3.9975 + 0.2j).reshape(1, 1, NX)),
}

# (theta_deg, phi_deg). `phi = 0` is the control: the conical solve must reproduce the decoupled
# TE/TM answer there, so it is the one angle whose result is independently known. The other two
# are where the s/p mixing is real and a rotation convention has somewhere to hide.
ANGLES = [(24.0, 0.0), (24.0, 15.0), (41.0, 40.0)]

# `s` is `psi = pi/2`, `p` is `psi = 0` — and RETICOLO's own `einc`, which the `.m` sets to
# `[0, 1]` and `[1, 0]` respectively.
CASES = [(s, t, p, pol)
         for s in STACKS
         for t, p in ANGLES
         for pol in ('s', 'p')]


def key(stack, theta, phi, pol):
    return f'{stack}__{theta:g}__{phi:g}__{pol}'


def n_layers(stack):
    return len(STACKS[stack][0])


def x_samples():
    return np.linspace(0, PERIOD, RES_X)
