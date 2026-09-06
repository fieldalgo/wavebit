"""Freeze Meep's answer on the 1D benchmark grating into ``golden_meep_1d.npz``.

Run once, by hand, in an environment that has Meep; ``test_meep_1d_golden.py`` then runs
anywhere. Rerun it only to add cases, never to make a failing test pass -- the whole value
of this oracle is that its numbers did not come from this codebase.

    conda run -n meep python tests/references/gen_goldens_meep_1d.py

**LICENCE FOOTING, and it is why this file exists rather than a dependency.** Meep is
GPL-2.0-or-later and wavebit is Apache-2.0. No file under the meep package is opened,
read, grepped or copied here: Meep is used only by *calling* its public Python API and
reading the numbers that come back, so every statement this file makes about Meep is a
black-box measurement. Nothing in the shipped package imports it, exactly as the RETICOLO
generator needs Octave and the fmmax ones need the ``fwref`` env. ``sys.tracebacklimit``
is set to 0 below so that an uncaught exception cannot print lines of another package's
source into the terminal.

**Why this file exists when ``gen_goldens_meep_crossed.py`` already does.** The crossed
generator answers a question this one cannot -- per-order amplitudes on a 2D order grid --
but it does not answer the question ``docs/fdtd.md`` §2 and ``README.md`` actually put to
a reader. Those two documents print a three-way table on *this* grating and lean on Meep's
column as the FDTD path's headline validation -- until 2026-09-06 in the words "our worst
per-order gap 7.4e-4 (TE) / 6.2e-4 (TM) against Meep's 7.4e-4 / 6.9e-4 -- the same
number". **Those Meep numbers were computed in a sibling lab repository on 2026-08-25 and
were quoted, not checked.** Nothing in this repository could reproduce them, which is a
strange place for a project's strongest published evidence to live. This generator puts
the table's Meep column in the repository, at two resolutions, with the harness that
produced it; both documents now print what it wrote and cite the golden rather than the
lab, and §2 carries a subsection on what re-deriving the table moved.

**Everything about the bridge is inherited from the crossed generator, deliberately.** The
axis map, the time convention, the s/p frame, the three leg signs, the origin factor, the
monitor quadrature, the source's Bloch phase, the z pixel parity and the integer-thickness
refusal are all *that* file's findings, re-used here rather than re-derived. Read its
docstring for the derivations and for the five harness faults, each of which returned a
plausible energy-conserving wrong number first. Two of them narrow to constants on this
scene and are worth restating:

* the **origin factor** is ``(-1)**m`` here, the 1D half of the crossed file's
  ``(-1)**m (-1)**l``. wavebit places a raster from the cell centre and references each
  order's amplitude phase from the cell corner; normalised by the incident order the two
  differ by ``exp(-i (k_m - k_inc) P / 2)``, which is that sign. It is applied, not fitted;
* the **integer-pixel refusals** bite on this scene at both tiers by construction: period
  0.70 is 140 and 70 pixels, thickness 0.30 is 60 and 30, and the two-column raster's
  0.35 boundary is 70 and 35. Nothing is rounded, so both codes see the same staircase.

**What is new here is the calibration, and it runs first.** ``slab_uniform`` is the same
0.30 layer with the tooth's patterning removed. A planar stack has an exact closed form,
and ``_meep_1d_cases.py::airy_rt`` writes it out from the boundary conditions with no
solver of any kind involved -- not Meep's, not wavebit's. ``_calibrate`` puts that scene
through the *same* ``_run`` and ``_amplitudes`` the grating uses and refuses to write a
golden unless the amplitudes come back within a bound the grid can explain. That ordering
is this repository's recorded rule that a reference must be shown able to answer a
question before its answer is believed, and it is what establishes the normalisation:
a flux convention wrong by ``cos(theta)``, an ``n`` or a factor of 2 reproduces every
efficiency ratio in the table below and fails the closed form at the first digit.

The calibration reaches three things no energy check can:

* the **absolute** normalisation, since ``|r|`` and ``|t|`` are pinned to a known number
  rather than to each other;
* the **phase**, which is where a misplaced reference plane lives -- the fault that cost
  the crossed golden 4.6e-2 while ``R + T`` stayed at 0.9954;
* the **p leg's sign**. At normal incidence ``r_p = -r_s`` and ``t_p = +t_s`` exactly, and
  no efficiency anywhere in this file can tell that from ``r_p = +r_s``.
"""
import pathlib
import sys
import time

sys.tracebacklimit = 0                     # never print frames from another package
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / 'golden_meep_1d.npz'

#: The coarse control runs at half the resolution. It is not a second golden: it is what
#: lets the committed file state its own error bar, and it is what a reader of
#: ``docs/fdtd.md`` §2 needs in order to see the convergence rather than take one number on
#: faith. At order q the two tiers differ by ``(2**q - 1)`` times the fine grid's error.
#: **Measured on this scene q is 1.73 to 2.17** -- 1.93 on the planar case in both legs,
#: 2.17 (TE) and 1.73 (TM) on the grating, against a converged RCWA answer. A 1D grating with
#: one vertical interface per period keeps second order where the crossed cells' corners cost
#: it (0.69 to 1.63 there), so the second-order reading is the honest one here; both are
#: printed anyway, because which end to read is a property of the scene rather than of this
#: file.
COARSE_RATIO = 2

#: How close the calibration has to come to the closed form before a golden is written,
#: as ``COEFF / resolution**2`` -- a *grid* bound rather than a fitted one. What is left
#: after the conventions are right is the slab's own discretisation error, which falls at
#: second order, and a convention that is wrong misses by 1e-1 to 2 instead. Stated as a
#: coefficient so that it follows a case added at another resolution rather than having to
#: be re-tabulated. Measured on this planar scene at resolutions 20 / 40 / 80: worst
#: departure 2.14e-1, 4.38e-2, 1.08e-2, which is order 2.29 then 2.02 and a coefficient of
#: 86, 70, 69. The bound below is ~1.7x that, and the separation it has to make is 60x:
#: with a convention wrong the departure is 1e-1 to 2 at every resolution, because it does
#: not depend on the grid at all.
CALIBRATION_COEFF = 1.2e2


def calibration_bound(resolution):
    """The bound the planar scene has to meet at ``resolution``."""
    return CALIBRATION_COEFF / float(resolution) ** 2


# ---------------------------------------------------------------------------------------
# the meep half -- the only part that needs the meep environment
# ---------------------------------------------------------------------------------------

def _z_layout(d, resolution, dpml, pad, gap, src_frac):
    """Cell extent and the three z planes, all snapped to the grid.

    The monitors sit ``gap`` from the structure and no further. Below a periodic structure
    the field is exactly a sum of Rayleigh orders at any depth and the spatial transform
    separates them exactly, so an evanescent order on the monitor plane cannot contaminate
    a propagating one -- the same finding ``docs/fdtd.md`` §2 records for wavebit's own
    ``pad_scale``. What the distance *does* buy is de-referencing error: the phase is
    undone with the analytic ``kz`` while meep propagates on the numerical one.

    The even pixel count is the crossed generator's fault 4: the z pixel COUNT decides
    where meep anchors its grid, and an odd count puts a grid point exactly *on* the
    structure's interfaces instead of between them. **It is inert on this scene** --
    ``2.9 * resolution`` is even at every legal resolution here (116 / 174 / 232 / 290 /
    348 / 580), measured EXACTLY zero -- so it protects other scenes and must not be read
    as the thing that controls where the structure lands in the grid. Something in that
    family is still live: see ``_meep_1d_cases.py::MEEP_UNEXPLAINED``.

    ``src_frac`` is ``meep_src_frac`` in the case dict, and it moved there on 2026-09-07
    from a bare ``0.30`` written into this line. It is not a determinant by itself
    (2.1e-7 at half its value) but it is what couples ``pad`` to the source plane -- so a
    ``pad`` sweep is never a clean clearance sweep -- and a number that lives only in a
    body is a number a regeneration can move in silence.
    """
    snap = lambda v: round(v * resolution) / resolution
    dpml, pad, gap, d = snap(dpml), snap(pad), snap(gap), snap(d)
    z_min, z_max = -(dpml + pad), d + pad + dpml
    if round((z_max - z_min) * resolution) % 2:      # fault 4
        z_max += 1.0 / resolution
    return dict(z_min=z_min, z_max=z_max, d=d,
                z_src=z_min + dpml + snap(src_frac * pad), z_ref=-gap, z_trn=d + gap)


def _runs(row):
    """``(i0, i1, index)`` per maximal run of identical adjacent columns.

    Not an optimisation. A block face meep can see is a face it smooths across, whether or
    not the two sides hold the same material -- see ``_layer_blocks`` -- so a run of equal
    columns has to reach it as one block rather than as several abutting ones.
    """
    out, i = [], 0
    while i < row.size:
        j = i
        while j + 1 < row.size and row[j + 1] == row[i]:
            j += 1
        out.append((i, j + 1, float(row[i])))
        i = j + 1
    return out


def _layer_blocks(mp, row, px, t, z0):
    """One layer's meep objects: a full-period background, then the runs that differ.

    **A block face meep can see is a face it smooths across, whether or not the material
    is the same on both sides of it.** That is the sixth harness fault in this bridge (the
    other five are ``gen_goldens_meep_crossed.py``'s), it appeared twice in two costumes,
    and both were caught by a symmetry the answer must have and the harness cannot arrange:

    | scene | what a block per raster column gave | what it must be |
    | --- | --- | --- |
    | planar slab, ``T(+-1)`` -- a uniform layer has no grating harmonic | 4.4e-5 each, and both orders propagate in the substrate | 0 |
    | 50 % binary grating at normal incidence, ``T(+1) - T(-1)`` -- the profile is mirror-symmetric | 6.2e-3 at resolution 40, **1.7e-3 at the frozen 200**, which is larger than the 1.3e-3 gap this golden exists to measure | 0 |

    With the construction below both go to round-off: 1.7e-16 off-specular on the slab, and a
    TE split of 8.3e-17 at both tiers. TM's split does not vanish -- 1.13e-4 fine, 1.92e-4
    coarse -- and that is the grid rather than a broken symmetry, because TM's ``Ex`` is the
    component normal to the tooth's vertical faces and therefore the discontinuous one.

    The grating's split is the instructive one. It is **not** the extraction: the same
    monitor samples through a plain uniform-weight quadrature over exactly one period give
    the same split to six digits, and the monitor *field* is itself asymmetric by 1.9e-2.
    It is not the periodic image either -- adding image copies of every block changes
    nothing. It is the **air block**: its right face lands on the cell edge with
    ``default_material`` (the same air) on the other side, and meep smooths it anyway.
    Dropping that one block takes the split to **exactly zero** and the answer becomes
    translation-invariant, as a periodic scene's must be:

        tooth block + air block   T(-1) = 0.198521, T(+1) = 0.204668
        tooth block alone         T(-1) = T(+1) = 0.203965, and the same for a tooth
                                  centred anywhere else in the cell

    So a block is emitted only where it is a **material interface**. The construction is a
    full-period background, then the other runs laid on top of it -- and "on top" is meep's
    own precedence, measured rather than assumed: ``[background, tooth]`` reproduces the
    tooth-alone answer to every digit while ``[tooth, background]`` returns a uniform
    layer, so the **later** object wins.

    **Which run is the background is not free, and it is the LAST one.** Meep does not wrap
    the geometry when it smooths at the low cell edge: it reads what the geometry literally
    says beyond ``-P/2``, which is the background and nothing else. So the background has
    to be the material that periodicity puts there -- the material of the *last* column,
    since ``x`` just below ``-P/2`` is ``x`` just below ``+P/2``. Measured on this grating
    at resolution 40, where mirror symmetry says the split must be 0:

        background = last run (air), tooth overlaid       split 0, exactly
        background = first run (tooth), air overlaid      split 1.3e-2
        a block per run, no background                    split 6.2e-3

    A run whose index equals the background is skipped, because the background already
    covers it and a block there would be exactly the fictitious face above.
    """
    runs = _runs(row)
    dx = px / row.size
    bg = runs[-1][2]
    out = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, t), center=mp.Vector3(0, 0, z0 + 0.5 * t),
                    material=mp.Medium(index=bg))]
    for i0, i1, n in runs[:-1]:
        if n == bg:
            continue
        # The raster is placed from the CELL CENTRE, which is where `wavebit.meshgrid_xy`
        # puts column i as well.
        out.append(mp.Block(size=mp.Vector3((i1 - i0) * dx, mp.inf, t),
                            center=mp.Vector3(0.5 * (i0 + i1) * dx - 0.5 * px, 0,
                                              z0 + 0.5 * t),
                            material=mp.Medium(index=n)))
    return out


def _run(case, leg, resolution, empty):
    """One meep run. Returns the two monitor lines and the metadata they need.

    **Every setting comes from the case dict.** ``gap`` and ``df_frac`` were function
    defaults here until 2026-09-07; their values are unchanged and the regenerated golden
    is identical, but a determinant that lives in a signature is one a regeneration can
    move without anybody writing it down -- which is the failure ``docs/fdtd.md`` §2 was
    published with. ``gap`` is worth 3.4e-3 at zero and nothing at all over 0.02-0.40;
    ``df_frac`` is worth 1e-7, because the DFT is at a single frequency.

    The cell has **three real axes with the y one a single pixel**: ``cell_size``'s y
    entry is 0, the raster is invariant along y, and ``k_point``'s y component is 0, so
    the run is a 1D grating solved in a cell that still has an x, a y and a z. That is the
    identity axis map the crossed generator uses, kept on purpose -- a genuinely
    two-dimensional cell would have to carry wavebit's ``z`` on some other letter, and the
    permutation is exactly the kind of thing that reproduces the right efficiencies with
    the polarisations exchanged.
    """
    import meep as mp

    dpml, pad = case['meep_dpml'], case['meep_pad']
    gap, df_frac, src_frac = case['meep_gap'], case['meep_df_frac'], case['meep_src_frac']
    wl = case['wavelength']
    px = case['period'][0]
    fcen = 1.0 / wl
    arr = np.asarray(case['index'], dtype=float)
    if arr.shape[1] != 1:
        raise SystemExit('this generator is the 1D one; a raster with Ny > 1 belongs to '
                         'gen_goldens_meep_crossed.py')
    # A period, a layer thickness or a raster column boundary that is not an integer number
    # of pixels is REFUSED rather than rounded, because rounding one is a *phase* error
    # that every energy check passes -- the crossed generator measured a 0.24 um layer
    # snapped from 14.4 pixels to 14 putting the specular amplitude 4.6e-2 from the
    # converged answer while `R + T` stayed at 0.9954 and every per-order efficiency stayed
    # inside 1.6e-2. It survived a two-tier convergence study as well, since both tiers
    # rounded to the same value.
    edges = [px, px / arr.shape[2]] + list(case['thickness'])
    for v in edges:
        if abs(v * resolution - round(v * resolution)) > 1e-9:
            raise SystemExit(
                f'{v} is {v * resolution:g} pixels at resolution {resolution}. Give a '
                f'length the grid can hold exactly, or the geometry meep sees is not the '
                f'one the case declares and the transmitted phase is referenced to a '
                f'plane the structure does not end on.')
    lay = _z_layout(float(sum(case['thickness'])), resolution, dpml, pad, gap, src_frac)
    th, ph = np.deg2rad(case['theta_deg']), np.deg2rad(case['phi_deg'])
    n_top, n_bot = case['n_top'], case['n_bot']

    kpt = mp.Vector3(fcen * n_top * np.sin(th) * np.cos(ph), 0, 0)
    kx = 2 * np.pi * kpt.x
    amp = None if kx == 0.0 else (lambda p, _a=kx: np.exp(1j * _a * p.x))

    # An ELECTRIC sheet along s_hat drives only the s wave -- the p wave's in-plane E lies
    # along (cos varphi, sin varphi), which is orthogonal to it. A MAGNETIC sheet along
    # s_hat drives only the p wave, whose H is -+ s_hat. At this scene's azimuth s_hat is
    # y_hat exactly, so TE is driven by Ey and TM by Hy: no polarisation is ever *named*,
    # only a component. `purity` below is what checks that this worked.
    sx, sy = -np.sin(ph), np.cos(ph)
    comps = (mp.Ex, mp.Ey) if leg == 'te' else (mp.Hx, mp.Hy)
    # `is_integrated=True`: a current sheet whose time profile has a nonzero integral
    # leaves a static field behind, and with a Bloch phase that field cannot radiate away.
    src = [mp.Source(mp.GaussianSource(fcen, fwidth=df_frac * fcen, is_integrated=True),
                     component=c, center=mp.Vector3(0, 0, lay['z_src']),
                     size=mp.Vector3(px, 0, 0), amp_func=amp, amplitude=a)
           for c, a in zip(comps, (sx, sy)) if a != 0.0]

    geom = []
    if not empty:
        z0 = 0.0
        for layer, t in zip(arr, case['thickness']):
            geom.extend(_layer_blocks(mp, layer[0], px, t, z0))
            z0 += t
        # Sized to the +z cell edge and NO further -- the crossed generator's fault 1:
        # meep replicates geometry periodically along the PML axis too, so a substrate
        # block overhanging the far edge wraps into the superstrate. Measured there at
        # `T = 0.412` against a true 0.684, with `R + T` exactly 1.
        geom.append(mp.Block(size=mp.Vector3(mp.inf, mp.inf, lay['z_max'] - z0),
                             center=mp.Vector3(0, 0, 0.5 * (z0 + lay['z_max'])),
                             material=mp.Medium(index=n_bot)))

    sim = mp.Simulation(cell_size=mp.Vector3(px, 0, lay['z_max'] - lay['z_min']),
                        resolution=resolution,
                        boundary_layers=[mp.PML(dpml, direction=mp.Z)],
                        geometry=geom, sources=src, k_point=kpt,
                        default_material=mp.Medium(index=n_top),
                        geometry_center=mp.Vector3(0, 0, 0.5 * (lay['z_min'] + lay['z_max'])),
                        force_complex_fields=True)
    sim.init_sim()
    mons = {k: sim.add_dft_fields([mp.Ex, mp.Ey], [fcen],
                                  center=mp.Vector3(0, 0, lay[f'z_{k}']),
                                  size=mp.Vector3(px, 0, 0))
            for k in ('ref', 'trn')}
    # A FIXED run length, not `stop_when_fields_decayed`: the cell holds ONE transverse
    # wavevector, so content below the light line is trapped and never reaches a PML. The
    # running DFT suppresses it because it sits far from the extraction frequency; the
    # check that says so is that doubling the length does not move the answer. Measured on
    # this grating (TE, resolution 40) at until = 60 / 120 / 240 / 480: `R + T`, `T0` and
    # `T(+1)` are identical to all eight printed digits and the worst amplitude moves
    # 3.5e-9, 7.1e-10, 3.2e-10. The cases run at 120, which is a 40x margin over where it
    # settles and still a third cheaper than the crossed generator's 200.
    sim.run(until_after_sources=case['meep_until'])

    out = {}
    for k, obj in mons.items():
        md = sim.get_array_metadata(dft_cell=obj)
        out[f'{k}__x'] = np.asarray(md[0], dtype=float).ravel()
        # The DFT monitor's samples are WEIGHTED, not uniform -- the crossed generator's
        # fault 2. A line spanning one period comes back overhanging it by half a cell at
        # each end, with the edge samples down-weighted so the weights sum to the period.
        # A plain mean put 9.9e-3 of a plane wave into orders a plane wave does not have.
        out[f'{k}__w'] = np.asarray(md[3], dtype=float).ravel()
        for comp, name in ((mp.Ex, 'Ex'), (mp.Ey, 'Ey')):
            out[f'{k}__{name}'] = np.asarray(sim.get_dft_array(obj, comp, 0)).ravel()
    out.update(lay)
    out['meep_version'] = str(mp.__version__)
    out['meep_processors'] = int(mp.count_processors())
    return out


# ---------------------------------------------------------------------------------------
# the numpy half -- the conventions, all of them derived in the crossed generator
# ---------------------------------------------------------------------------------------

def _kz(n, kx):
    """``kz / k0`` with ``Re >= 0``: a downgoing propagating wave, or decay toward +z."""
    v = np.sqrt(np.asarray(complex(n) ** 2 - np.asarray(kx) ** 2, dtype=complex))
    return np.where(np.real(v) < 0, -v, v)


def _frame(kx, phi):
    """``(cos varphi, sin varphi)`` -- the DIRECTION of ``(kx, 0)``, not a line.

    At the origin, where the in-plane wavevector carries no direction at all, the answer
    is ``phi``: a wave going straight down still needs a reference direction in the xy
    plane for its polarisation, and ``phi`` is by convention what supplies it. On this
    scene ``phi = 0``, so the specular order's frame is ``x_hat`` and every ``m < 0``
    order's is ``-x_hat`` -- the per-order half-turn, which is live here even though the
    mount is normal.
    """
    kpar = np.abs(np.real(kx))
    safe = np.where(kpar > 0, kpar, 1.0)
    return (np.where(kpar > 0, np.real(kx) / safe, np.cos(phi)),
            np.where(kpar > 0, 0.0 * np.real(kx), np.sin(phi)))


def _coeff(field, x, w, px, kx, k0):
    """One order's coefficient on one monitor line, on meep's own abscissae and weights."""
    return np.sum(np.asarray(w) * np.asarray(field)
                  * np.exp(-1j * k0 * kx * np.asarray(x, dtype=float))) / px


def _amplitudes(case, leg, struct, empty):
    """Per-order ``(r_s, r_p, t_s, t_p)`` and ``(R, T)`` on the ``(1, m)`` grid.

    The arrays carry a leading length-1 axis so that this golden has the same rank as
    ``golden_meep_crossed.npz``: a y-invariant grating is the crossed case with one row of
    orders, and giving it a different shape would make the two tests' helpers diverge for
    no reason.
    """
    wl, px = case['wavelength'], case['period'][0]
    n_top, n_bot = case['n_top'], case['n_bot']
    th, ph = np.deg2rad(case['theta_deg']), np.deg2rad(case['phi_deg'])
    k0 = 2.0 * np.pi / wl
    m = np.asarray(case['orders_u'], dtype=int)
    kxi = n_top * np.sin(th) * np.cos(ph)
    KX = kxi + m * wl / px
    c_m, s_m = _frame(KX, ph)
    c_i, s_i = _frame(np.asarray(kxi), ph)
    kz_top, kz_bot = _kz(n_top, KX), _kz(n_bot, KX)
    kz_inc = _kz(n_top, kxi)
    z_ref, z_trn, d = struct['z_ref'], struct['z_trn'], struct['d']

    def take(run, tag, comp, sub=None):
        f = np.asarray(run[f'{tag}__{comp}'])
        if sub is not None:
            f = f - np.asarray(sub[f'{tag}__{comp}'])
        return np.array([_coeff(f, run[f'{tag}__x'], run[f'{tag}__w'], px, kx, k0)
                         for kx in KX])

    inc = take(empty, 'ref', 'Ex'), take(empty, 'ref', 'Ey')
    i0 = int(np.flatnonzero(m == 0)[0])
    ex_i = inc[0][i0] * np.exp(-1j * k0 * kz_inc * z_ref)
    ey_i = inc[1][i0] * np.exp(-1j * k0 * kz_inc * z_ref)
    a_s_inc = -s_i * ex_i + c_i * ey_i
    a_p_inc = -(n_top / kz_inc) * (c_i * ex_i + s_i * ey_i)
    a_inc = a_s_inc if leg == 'te' else a_p_inc

    # THE INSTRUMENT. A plane wave has one order; anything the empty run leaves in the
    # others is the harness talking. This is what caught the unweighted quadrature and the
    # constant-amplitude oblique sheet.
    others = np.abs(inc[0]) + np.abs(inc[1])
    others[i0] = 0.0
    leak = float(others.max() / (abs(inc[0][i0]) + abs(inc[1][i0])))
    purity = float(abs(a_p_inc / a_s_inc) if leg == 'te' else abs(a_s_inc / a_p_inc))

    ref_x = take(struct, 'ref', 'Ex', empty) * np.exp(1j * k0 * kz_top * z_ref)
    ref_y = take(struct, 'ref', 'Ey', empty) * np.exp(1j * k0 * kz_top * z_ref)
    trn_x = take(struct, 'trn', 'Ex') * np.exp(-1j * k0 * kz_bot * (z_trn - d))
    trn_y = take(struct, 'trn', 'Ey') * np.exp(-1j * k0 * kz_bot * (z_trn - d))

    origin = (-1.0) ** m
    sig_r, sig_t = 1.0, -1.0                       # the canonical frame; see the header
    r_s = origin * (-s_m * ref_x + c_m * ref_y) / a_inc
    r_p = origin * sig_r * (n_top / kz_top) * (c_m * ref_x + s_m * ref_y) / a_inc
    t_s = origin * (-s_m * trn_x + c_m * trn_y) / a_inc
    t_p = origin * sig_t * (n_bot / kz_bot) * (c_m * trn_x + s_m * trn_y) / a_inc

    prop_r = np.abs(np.imag(kz_top)) < 1e-9
    prop_t = np.abs(np.imag(kz_bot)) < 1e-9
    R = (np.abs(r_s) ** 2 + np.abs(r_p) ** 2) * np.real(kz_top) / np.real(kz_inc) * prop_r
    T = (np.abs(t_s) ** 2 + np.abs(t_p) ** 2) * np.real(kz_bot) / np.real(kz_inc) * prop_t
    out = dict(r_s=r_s, r_p=r_p, t_s=t_s, t_p=t_p, R=R, T=T)
    return {k: np.asarray(v)[None, :] for k, v in out.items()}, leak, purity


_EMPTY_CACHE = {}


def _empty_run(case, leg, resolution):
    """The incident-only run, memoised on everything it depends on.

    Both scenes here share a wavelength, a cell, a mount and a superstrate, so their empty
    runs are the same computation -- and it is half the wall time. The key is the full set
    of parameters rather than the case name, so a case added with any of them different
    gets its own run instead of quietly borrowing this one's.
    """
    key = (leg, resolution, case['wavelength'], case['period'][0],
           tuple(case['thickness']), case['n_top'], case['theta_deg'], case['phi_deg'],
           case['meep_dpml'], case['meep_pad'], case['meep_until'],
           case['meep_gap'], case['meep_df_frac'], case['meep_src_frac'])
    if key not in _EMPTY_CACHE:
        _EMPTY_CACHE[key] = _run(case, leg, resolution, empty=True)
    return _EMPTY_CACHE[key]


def solve(case, leg, resolution):
    """One scene, one incident polarisation: the structure run and the empty control."""
    empty = _empty_run(case, leg, resolution)
    struct = _run(case, leg, resolution, empty=False)
    return _amplitudes(case, leg, struct, empty) + (struct['meep_version'],)


# ---------------------------------------------------------------------------------------
# the calibration -- run before anything is believed, let alone frozen
# ---------------------------------------------------------------------------------------

#: ``r_s`` is the TE leg's own amplitude and ``r_p`` the TM leg's; the cross terms are the
#: ones the calibration drives to zero.
LEG_TAG = {'te': 's', 'tm': 'p'}


def _closed_form(case):
    """``(r, t)`` for the case's stack, in the s convention, from the case file."""
    from _meep_1d_cases import airy_rt

    n_list = [float(np.asarray(layer).ravel()[0]) for layer in np.asarray(case['index'])]
    return airy_rt(case['wavelength'], case['n_top'], n_list, list(case['thickness']),
                   case['n_bot'])


def _calibrate(case, grids, resolution):
    """Refuse to freeze anything unless the planar scene reproduces its closed form.

    Four claims at once, and only the first survives an efficiency check:

    * the **normalisation** -- ``|r|`` and ``|t|`` against known numbers rather than
      against each other, so a stray ``cos(theta)``, ``n`` or 2 has nowhere to hide;
    * the **phase** of both, which is where a misplaced reference plane lives;
    * the **p leg's sign**: ``r_p = -r_s`` and ``t_p = +t_s`` at normal incidence, exactly,
      because the two waves' tangential fields obey identical equations and only the
      frame's ``p_hat`` reverses between the incident and reflected legs;
    * that a **uniform layer radiates nothing off-specular**, which the ``m = +-1`` orders
      -- propagating in the substrate on this scene -- are there to receive.

    Returns the worst departure, so the caller can print it and the golden can carry it.
    """
    r_exact, t_exact = _closed_form(case)
    want = {'te': dict(r_s=r_exact, t_s=t_exact, r_p=0.0, t_p=0.0),
            'tm': dict(r_p=-r_exact, t_p=t_exact, r_s=0.0, t_s=0.0)}
    i0 = int(np.flatnonzero(np.asarray(case['orders_u']) == 0)[0])
    worst, where = 0.0, ''
    for leg, g in grids.items():
        for key, value in want[leg].items():
            gap = abs(complex(g[key][0, i0]) - complex(value))
            if gap > worst:
                worst, where = gap, f'{leg}.{key}(0)'
        # Off-specular: a planar stack puts nothing there, and both orders propagate.
        for i, m in enumerate(case['orders_u']):
            if m == 0:
                continue
            for key in ('r_s', 'r_p', 't_s', 't_p'):
                gap = abs(complex(g[key][0, i]))
                if gap > worst:
                    worst, where = gap, f'{leg}.{key}({m})'
    bound = calibration_bound(resolution)
    print(f'  calibration at resolution {resolution}: worst departure from the closed '
          f'form {worst:.2e} at {where}  (bound {bound:g})')
    print(f'    exact  r = {r_exact.real:+.6f}{r_exact.imag:+.6f}j   '
          f't = {t_exact.real:+.6f}{t_exact.imag:+.6f}j')
    for leg, g in grids.items():
        rr, tt = complex(g[f'r_{LEG_TAG[leg]}'][0, i0]), complex(g[f't_{LEG_TAG[leg]}'][0, i0])
        print(f'    meep {leg}: r = {rr.real:+.6f}{rr.imag:+.6f}j   '
              f't = {tt.real:+.6f}{tt.imag:+.6f}j')
    if not worst < bound:
        raise SystemExit(
            f'the planar calibration is {worst:.2e} from the closed form at {where}, '
            f'against a {bound:g} bound that is what this grid can explain. The bridge '
            f'is not measuring what it claims; do not freeze a grating answer through it.')
    return worst


def _check_build():
    """Refuse to freeze anything on a build the record does not name.

    Eight settings that move this column are never passed by this file --
    ``eps_averaging``, ``ensure_periodicity``, ``Courant``, the PML profile and
    ``R_asymptotic``, the ``GaussianSource`` cutoff, the decimation factor and the
    subpixel tolerances -- four of them worth 5e-4 to 1.5e-2, which is the size of the
    agreement ``docs/fdtd.md`` §2 claims. **The version string is the only thing pinning
    them.** So a golden regenerated on another build is a different column wearing the
    same name, and this refuses rather than lets that happen quietly. Updating meep means
    updating ``_meep_1d_cases.py::MEEP_BUILD`` *and* re-deriving the table.
    """
    import meep as mp
    from _meep_1d_cases import MEEP_BUILD

    got, want = str(mp.__version__), MEEP_BUILD['version']
    if got != want:
        raise SystemExit(
            f'meep {got} is installed and the record names {want}. Eight determinants of '
            f'this column are meep defaults this file never passes, so the version IS '
            f'the record for them: update MEEP_BUILD (version and conda build string) '
            f'and re-derive docs/fdtd.md section 2 rather than overwriting the golden.')
    procs = int(mp.count_processors())
    if bool(MEEP_BUILD['parallel']) != (procs > 1):
        raise SystemExit(f'the record says parallel={MEEP_BUILD["parallel"]} and meep '
                         f'reports {procs} processors.')


def _record_arrays(cases):
    """Every determinant, into the ``.npz`` beside the numbers it produced.

    The frozen values and the conditions that produced them travel together or they come
    apart, which is exactly what happened to §2's table: the numbers were in a document
    and the conditions were in a phrase about defaults.
    """
    from _meep_1d_cases import (MEEP_BUILD, MEEP_CONSTRUCTION, MEEP_DERIVED,
                                MEEP_READING_FLOOR, MEEP_UNEXPLAINED, UNITS)

    out = {'record__units': np.array(UNITS)}
    for key, value in MEEP_BUILD.items():
        out[f'record__build__{key}'] = np.array(str(value))
    for key, value in MEEP_CONSTRUCTION.items():
        out[f'record__construction__{key}'] = np.array(str(value))
    for key, value in MEEP_READING_FLOOR.items():
        out[f'record__reading_floor__{key}'] = np.array(float(value))
    out['record__unexplained'] = np.array(str(MEEP_UNEXPLAINED))
    for res, lay in MEEP_DERIVED.items():
        for key, value in lay.items():
            out[f'record__derived__{res}__{key}'] = np.array(float(value))
    for name, case in cases.items():
        for key, value in case.items():
            if key.startswith('meep_') or key in ('wavelength', 'theta_deg', 'phi_deg',
                                                  'n_top', 'n_bot'):
                out[f'record__case__{name}__{key}'] = np.array(float(value))
        out[f'record__case__{name}__period'] = np.asarray(case['period'], dtype=float)
        out[f'record__case__{name}__thickness'] = np.asarray(case['thickness'], dtype=float)
        out[f'record__case__{name}__index'] = np.asarray(case['index'], dtype=float)
    return out


def main():
    from _meep_1d_cases import CASES

    _check_build()
    first = CASES[next(iter(CASES))]
    if np.asarray(first['index']).shape[2] != 1:
        raise SystemExit(
            'the first case must be planar. `_calibrate` is what says this bridge can '
            'answer at all, and it has to run before any patterned answer is frozen '
            'through it -- a reference is shown able to answer before it is believed.')

    out, t_start = {}, time.time()
    for name, case in CASES.items():
        fine = int(case['meep_resolution'])
        coarse = fine // COARSE_RATIO
        out[f'{name}__orders_u'] = np.asarray(case['orders_u'], dtype=int)
        out[f'{name}__orders_v'] = np.zeros(1, dtype=int)
        print(f'\n{name}: {case["period"][0]} um period, resolutions {fine} and {coarse}')
        for tier, res in (('fine', fine), ('coarse', coarse)):
            grids = {}
            out[f'{name}__{tier}__resolution'] = np.array(res)
            for leg in ('te', 'tm'):
                t0 = time.time()
                g, leak, purity, version = solve(case, leg, res)
                wall = time.time() - t0
                grids[leg] = g
                for key, val in g.items():
                    out[f'{name}__{leg}__{tier}__{key}'] = val
                out[f'{name}__{leg}__{tier}__leak'] = np.array(leak)
                out[f'{name}__{leg}__{tier}__purity'] = np.array(purity)
                out['meep_version'] = np.array(version)
                total = float(g['R'].sum() + g['T'].sum())
                # Nothing here absorbs, so R + T = 1 up to the Yee grid's O(dz**2). A gross
                # failure of it means the bridge is wired wrong, and freezing that would be
                # worse than having no oracle: refuse instead. It is a weak check on
                # purpose -- three of the crossed generator's five faults left `R + T` at 1
                # while the answer was wrong.
                if not 0.9 < total < 1.1:
                    raise SystemExit(
                        f'{name} {leg} {tier}: R + T = {total:.6f} on a lossless scene. '
                        f'The bridge is wired wrong; do not freeze this.')
                if leak > 1e-9:
                    raise SystemExit(
                        f'{name} {leg} {tier}: the incident-only run puts {leak:.2e} of '
                        f'the incident amplitude into orders a plane wave does not have.')
                print(f'  {tier:<6} {leg}  R+T = {total:.6f}   incident leak {leak:.1e}   '
                      f'cross-pol purity {purity:.1e}   wall {wall:.0f} s')
            if np.asarray(case['index']).shape[2] == 1:
                out[f'{name}__{tier}__calibration'] = np.array(
                    _calibrate(case, grids, res))
        # The committed file states its own error bar, printed as a RANGE because the
        # observed order decides which end to read.
        for leg in ('te', 'tm'):
            worst = 0.0
            for key in ('r_s', 'r_p', 't_s', 't_p'):
                a = out[f'{name}__{leg}__fine__{key}']
                b = out[f'{name}__{leg}__coarse__{key}']
                fin = np.isfinite(a) & np.isfinite(b)
                if fin.any():
                    worst = max(worst, float(np.abs(a[fin] - b[fin]).max()))
            print(f'  two tiers {leg}: worst |fine - coarse| = {worst:.2e}, so the fine '
                  f'grid carries {worst / (COARSE_RATIO ** 2 - 1):.2e} (if second order) '
                  f'to {worst / (COARSE_RATIO - 1):.2e} (if first)')

    out.update(_record_arrays(CASES))
    np.savez(OUT, **out)
    print(f'\nwrote {OUT} ({len(out)} arrays) in {time.time() - t_start:.0f} s')


if __name__ == '__main__':
    main()
