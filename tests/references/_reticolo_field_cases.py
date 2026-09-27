"""Cases for the RETICOLO **1D field** golden, shared by the generator and the test.

Imports numpy and nothing else: the generator needs Octave and a RETICOLO tree, the test needs
neither, and a case list that lives in only one of them is a case list that drifts.

This is the oracle for the *pure* TE/TM reconstruction (``grating_type == 0``,
``_core/fields.py::field_1d``), which the S4 and fmmax field goldens never reach — they both
route through the coupled 6-component paths. It has its own component layout (TE returns
``(Ey, Hx, Hz)``, TM returns ``(Hy, Ex, Ez)``) and its own sign handling, and RETICOLO returns
exactly those 3 components for a 1D grating, so the two line up without padding.

**The three grids, all measured rather than assumed** (see ``gen_goldens_reticolo_field.py``
for how, and ``README.md`` for what each was worth):

* **x** already coincides. ``interface/reticolo_res3.m`` hardcodes ``linspace(0, period, 11)``,
  which is this project's own inclusive convention, so ``RES_X`` is fixed at 11 and is not a
  free parameter here.
* **z** does not. RETICOLO emits ``npts`` rows for *every* profile entry including the
  zero-thickness ambients, so a stack of ``L`` layers comes back as ``(L + 2) * npts`` rows of
  which many share ``z == 0``. Rows are located by entry index, not by ``z`` -- see
  ``layer_rows``.
* **z runs the other way from** ``STACKS``. RETICOLO's ``+z`` points back out of the stack, from
  ``z = 0`` at its bottom, so against the top-to-bottom depth frame ``STACKS`` is written in the
  whole cell reverses, layer order included. That is a frame difference, and it carries the
  matching component parity: ``Z_PARITY``. It was also the frame difference from this
  library's own cell until 2026-09-21; since then the public frame is RETICOLO's -- ``+z`` up,
  bottom layer first -- so the tests reverse *both* cells into the depth frame, and the two
  transforms cancel.

**The stacks are stated in permittivity**, as everywhere in wavebit since 2026-08-20; RETICOLO
wants a refractive index and ``interface/Reticolo.py`` converts on the way out, in one place.
Both sides of this comparison therefore read the same numbers, which is the reason the case
list is shared at all.

**The geometry deliberately keeps every x sample off a permittivity jump.** In TM, ``E_x`` is
discontinuous across a vertical interface, so a sample sitting exactly on one has two correct
values and the two codes need not pick the same. That is a harness artefact rather than a
disagreement, and it is worth ~4e-1 -- large enough to drown the real story, which is that
``E_x`` converges only slowly. The high-index regions below start and end on odd multiples of
10 nm while the samples land on multiples of 40 nm.
"""
import numpy as np

WL = 620.0
PERIOD = 400.0
MAX_ORDER = 8                    # 17 retained orders
RES_X = 11                 # fixed by reticolo_res3.m's `x = linspace(0, period(1), 11)`
RES_Z = 7                  # per layer, inclusive of both faces, on both sides

NX = 40                    # raster columns; 10 nm cells
_CELL = PERIOD / NX

# TE is (Ey, Hx, Hz) and TM is (Hy, Ex, Ez), so the same three slots hold different components
# and the parity is not the same list twice. Under z -> -z, `E_z`, `H_x` and `H_y` change sign
# while `E_t` and `H_z` do not -- the rule `_core/fields.py` states for its own frame.
Z_PARITY = {'te': np.array([1.0, -1.0, 1.0]),
            'tm': np.array([-1.0, 1.0, -1.0])}
NAMES = {'te': ('Ey', 'Hx', 'Hz'), 'tm': ('Hy', 'Ex', 'Ez')}

# The one component neither code can be held to: TM's `E_x` is the discontinuous one, and the
# two resolve it differently at any finite truncation. Slot 1 of the TM layout.
SLOW_SLOT = {'te': None, 'tm': 1}


def _bar(lo_nm, hi_nm, eps_bg, eps_hi):
    """One binary bar per period, given by its edges in nm. Edges avoid the x samples."""
    row = np.full(NX, eps_bg, dtype=complex)
    row[int(round(lo_nm / _CELL)):int(round(hi_nm / _CELL))] = eps_hi
    return row


# name -> (thicknesses, structure). `structure` is (layer, 1, NX) of relative **permittivity**,
# this project's raster layout, with the layers listed **top to bottom** -- the order the bridge
# hands RETICOLO and the golden was frozen with. The public API lists them bottom to top since
# 2026-09-21, so the tests reverse both lists before solving.
#
# **The scene is exactly the one the golden was frozen against.** Every value here is the square
# of the refractive index this file carried before 2026-08-20, so the committed RETICOLO numbers
# still describe it: 2.25 was 1.5, 4 was 2, and 3.9975+0.2j was 2+0.05j. The untidy ones --
# `1.6900000000000002`, `2.5600000000000005` -- are `1.3 ** 2` and `1.6 ** 2` as doubles, and
# the tidy decimals beside them are different numbers; the bridge's `index_from_eps` carries
# that difference straight back into the texture Octave is handed.
STACKS = {
    # No truncation and no factorization question: this is where the frame and the amplitude
    # normalisation are decided, and it must be exact on all three components in both polarisations.
    'uniform': ([260.0], np.full((1, 1, NX), 2.25, dtype=complex)),
    # Asymmetric about the cell centre on purpose: a mirror-symmetric bar makes the +m and -m
    # orders equal, which is exactly the case a wrong Fourier order-index sign survives.
    'grating': ([260.0], _bar(50.0, 230.0, 1.0, 4.0).reshape(1, 1, NX)),
    # Two *different* patterned layers: the interface between them is under test, and so is
    # the layer walk -- these lists run top to bottom and RETICOLO's cell (like the public
    # reconstruction since 2026-09-21) bottom to top, and one layer cannot distinguish that
    # whole-cell reversal from a per-layer z reversal.
    'bilayer': ([150.0, 260.0], np.stack([_bar(50.0, 230.0, 1.0, 4.0),
                                          _bar(130.0, 330.0, 1.6900000000000002,
                                               2.5600000000000005)])[:, None, :]),
    # Absorbing: `Im(eps) > 0` is loss in this project's `e^{-iwt}` convention, and it survives
    # the conversion the bridge makes -- the principal root of `3.9975+0.2j` is `2+0.05j`, so
    # RETICOLO is handed a lossy index and its own index map comes back saying so. A field cell
    # is where a wrong sqrt branch shows as growth into the stack rather than decay.
    'lossy': ([260.0], _bar(50.0, 230.0, 1.0, 3.9975 + 0.2j).reshape(1, 1, NX)),
}

# (stack, theta_deg, pol). Normal incidence has no k_parallel to get wrong and is the control;
# the two oblique angles are where a sign or a branch has somewhere to hide.
CASES = [(s, t, p)
         for s in STACKS
         for t in (0.0, 24.0, 41.0)
         for p in ('te', 'tm')]


def key(stack, theta, pol):
    return f'{stack}__{theta:g}__{pol}'


def n_layers(stack):
    return len(STACKS[stack][0])


def layer_rows(n):
    """Which rows of RETICOLO's cell are the ``n`` layers, in *its* z direction: bottom to top.

    (``bilayer``'s kept rows run ``z = 0`` to ``410`` through its 260 nm layer first -- the one
    ``STACKS`` lists last.)

    ``res3`` emits ``RES_Z`` rows per profile entry, and the profile this bridge builds is
    ``[0, *thickness, 0]`` -- a zero-thickness ambient at each end. So entry ``k`` owns rows
    ``k*RES_Z : (k+1)*RES_Z`` and the layers are entries ``1 .. n``. Established by reading
    ``res3``'s index map, not by inspecting ``z``: eight of a single-slab cell's 21 rows carry
    ``z == 0`` and only one of them is the slab's own top face.

    Takes a count rather than a stack name because the conical golden beside this one
    (``_reticolo_conical_field_cases.py``) has its own stacks and the same rule -- it is a
    property of how ``res3`` lays a cell out, not of any particular case list.
    """
    return np.arange(RES_Z, (n + 1) * RES_Z)


def x_samples():
    return np.linspace(0, PERIOD, RES_X)
