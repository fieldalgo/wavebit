"""Does C7's epsilon-12 box converge below second order because of its y faces? M3's yz section.

**The question.** C7 (``studies/fdtd_metasurface_rcwa.py``) found its epsilon-12 box under conical
incidence (M3) converging below second order: least-squares slopes 1.51 to 1.75 (s 1.64 / 1.51,
p 1.75 / 1.62, amplitudes / efficiencies). On C7's rows, the lifted study's reference-free
self-convergence order reads 1.63 / 1.47 for s and 1.90 / 1.64 for p. Two follow-ups narrowed the
cause:

* ``studies/fdtd_metasurface_corners.py``: M3's xz cross-section, as a ridge uniform along y,
  converged at second order from 40 to 320 cells per wavelength.
* ``studies/fdtd_metasurface_lifted.py``: lifting the box off its glass left the order where it
  was, so the low order does not need the junction's rim.

The xz ridge had no y faces and no edges along x. This study runs M3's yz cross-section, a ridge
uniform along x, and asks whether it converges at second order.

**Why a 2D scene cannot test more.** On an axis along which the structure does not vary, the
field is one harmonic ``exp(i k a)``. A Bloch-differenced axis therefore equals the ``Axis1D``
problem with ``k`` replaced by ``k_eff = (2/dx) sin(k dx/2)``: a smooth shift of order ``dx^2``,
which cannot lower an order. A first draft of this study ran Bloch-differenced twins of the
ridges to test "differencing along an edge"; a review before registration showed this identity
on the dry run's rows, and those scenes were dropped. In M3 the x and y edges are finite, end at
vertices and carry every diffraction order along their length, and the vertical edges run along
z. None of that has a 2D twin.

**The scenes** (:data:`SCENES`). Each ridge is epsilon ``EPS_P`` (12), 0.30 high, on epsilon-2.25
glass. Each is under M3's wave: theta 25, at the three wavelengths 0.95 / 1.0 / 1.05 of the
fixed-k band. The glass face sits 0.25 of a cell above a node plane.

* ``Y12`` -- M3's yz cross-section: a ridge 0.20 wide in y, uniform along x (one ``Axis1D``
  cell), cell 0.50 in y, its axis 0.15 cells past ``floor(cells/2)`` (M3's y placement), phi 30.
* ``X12`` -- ``Y12`` turned a quarter turn about z: the ridge 0.20 wide in x, uniform along y
  (``Axis1D``), cell 0.50 in x, axis 0.15 cells past ``floor(cells/2)``, phi -60. The turn
  ``x' = y``, ``y' = -x`` maps the Yee grid onto itself. A solver that treats x and y alike
  therefore gives ``Y12``'s numbers; a difference is a defect of its own.
* ``F12`` -- the corner study's epsilon-12 film, built by that study's ``build``: the control.

Rungs: 40, 80, 160 and 320 cells per vacuum wavelength, with s and p incidence. At these rungs
every face keeps its fraction of a cell. Every scene propagates only the zeroth order. Each
scene's rung 80 is also run at 180 periods (the run-length check). The corner study's xz ridge
``R12`` is not run again: its rows are read (item 2).

**The resonance.** An RCWA scan before registration
(``studies/results/fdtd_metasurface_ridges_scan/scan.log``) shows a p resonance of the epsilon-12
yz ridge at the band's long end: R_p 0.61, 0.41 and 0.10 at 1.025, 1.05 and 1.075. A grid error
that shifts the resonance answers nonlinearly at coarse rungs. The step 40 -> 80 is therefore
printed but not judged, and the class reads the steps 80 -> 160 and 160 -> 320. M3's own order
was low across 40 to 120, so a cause in the y faces would also show at 80 -> 160. The dry runs'
epsilon-6 ridges have no resonance in the band (R_p 0.019 to 0.031), so they could not show it.

**The reference** is ``wavebit.rcwa`` with the ridge centred, on its shape path, where the
normal-vector-field rule is Li's. It is solved at the run's own angle over the truncation
ladder ``YX_LADDER``, N = 100, 200, 300, 400, 600, 800 along the periodic axis: ``(0, N)`` for
``Y12``, ``(N, 0)`` for ``X12``. The film's truncation is 0, which is exact. ``X12``'s reference
is the same problem turned, solved separately; the report prints the gap between the two.

**The statistic.** ``A(n)`` and ``F(n)`` are the worst amplitude and efficiency errors against
the ladder's top. The worst is taken over the zeroth order's outgoing polarisations, both
incident polarisations and the three wavelengths. ``u`` is the largest distance of the three
truncations below the top from it. The local order of a step ``a -> b`` is
``log(X(a)/X(b))/log(b/a)``. The reference-free self-convergence order of each three-rung run
(as in the lifted study) is reported beside it, not judged.

**The registered rule** (written before any FDTD row of these scenes, with the numbers listed
under "Known before registration" in view).

1. *Validity.* Every run is valid by C7's per-run criterion 1:
   * it settles below the decay line, or its one 180-period repeat does;
   * no growth or unsettled warning, and not marked unstable;
   * ``theta_at`` on the fixed-k angle;
   * every cell present and finite.

   A run the plan refuses is invalid. Two further checks apply:
   * Each scene's rung 80 at 180 periods differs from its 120-period run by at most
     ``FLOOR_SHARE`` (0.1) of that scene's largest change between rungs 160 and 320.
   * Each ridge's reference judges: ``u_A`` and ``u_F`` are at most ``REFERENCE_SHARE`` (1/50)
     of the smallest ``A`` and ``F`` among the judged rungs (80, 160, 320). At an error ratio
     between 2.8 and 4 per step, that moves a judged local order by at most about 0.04, less
     than the 0.05 between ``LOW_LINE`` and ``ORDER_LINE``.
2. *Controls.*
   * ``F12``'s local orders of ``A`` and ``F`` are at least ``ORDER_LINE`` (1.8) at every step,
     and ``F12`` reproduces the corner study's ``F12`` rows to ``SAME_LINE`` (1e-10) in every
     cell at every rung: same code, same machine. Either failing makes the study *invalid*.
   * *The xz premise.* The *second* reading below says that what holds M3 down needs more than
     a cross-section. That needs M3's xz cross-section to be *second* under this rule. The
     corner study's ``R12`` rows are therefore scored as ``Y12`` is: zeroth order, both
     incident polarisations, against that study's own ladder (top 600), under criterion 1's
     reference gate. They must class *second*. If they do not, or cannot be scored (a missing
     row, an unsettled run, a failed gate or a non-finite order), a *second* ``Y12`` is not
     read; a *low* or *mixed* ``Y12`` does not need the premise, and the study is not made
     *invalid*.
   * *The first-order control*, in two signs. ``Y12``'s own rows are scored against a reference
     whose ridge is narrower, and against one whose ridge is wider, by ``CONTROL_SHIFT`` (1/32)
     of a cell at each rung, solved at truncation 600. Each reference carries an error first
     order in the cell. The scan's sizing (``scan.log``) puts it, worst over the cells, at A
     1.15e-2 and F 1.23e-2 at rung 160, and A 5.7e-3 and F 6.1e-3 at rung 320, almost all of it
     at the wavelength 1.05. Under this study's zeroth-order statistic that is about eight and
     sixteen times ``R12``'s errors at those rungs, for amplitudes and efficiencies alike (``R12``
     A 1.46e-3 / 3.65e-4, F 1.62e-3 / 4.02e-4). Against ``Y12``'s own error it is not large: a
     review's eps-shift proxy, calibrated on the corner study's FDTD rows, put ``Y12``'s A at
     0.8 to 1.3e-2 at rung 160 and 2.1 to 3.2e-3 at 320, about the shift at 160 and half of it
     at 320. That is why both signs are run and *second* has an upper line. If either sign
     classes *second*, the rule cannot see a first-order term of that size, and a *second*
     ``Y12`` is not read (*rule inert*). A *low* or *mixed* ``Y12`` is read either way.
3. *Symmetry.* ``X12`` reproduces ``Y12`` to ``TWIN_LINE`` (1e-10) in every cell at every rung.
   If it does not, the solver treats x and y differently. That is reported as the finding, and
   the class is printed but not read.
4. *The class* reads four local orders: ``A`` and ``F``, steps 80 -> 160 and 160 -> 320. A scene
   is:
   * *second* when all four lie in [``ORDER_LINE``, ``UPPER_LINE``] = [1.8, 2.3];
   * *low* when all four are at most ``LOW_LINE`` (1.75);
   * *mixed* otherwise.

   The upper line is there because an order above 2 comes from cancellation. A first-order
   part opposing the second-order one cancels it between rungs, so the error heads for a zero
   crossing. Without the line, a first-order share of up to about 65% at rung 160 would read
   *second*. With it, the rule tolerates about the same share either way.
5. *The reading* of ``Y12``'s class.
   * *second*: M3's yz cross-section, with its y faces and its edges along x, converges at
     second order as a 2D ridge, and so does its xz cross-section. What holds M3 down is
     three-dimensional: its vertical edges, its vertices, or how the field varies along finite
     edges. No 2D scene separates these. This reading needs the xz premise and a first-order
     control that does not read *second*; otherwise the verdict says which failed.
   * *low*: M3's yz cross-section reproduces a low order as a 2D ridge, so its y faces or its
     edges along x are enough to hold the order down.
   * *mixed*: inconclusive. The report says when an order is above the upper line.

The verdict is checked in this order:

1. *withheld* when any of these differs from the registration: this file, its registered copy,
   the two study files it imports, the code digests, or the digest of the corner study's rows
   it reads;
2. *incomplete* when a row is missing;
3. *invalid* when a check in item 1 fails, either ``F12`` control fails, or the first-order
   control's orders cannot be computed;
4. the symmetry finding (a twin gap that cannot be computed is printed as such, not as an
   asymmetry);
5. the reading, with its premise and its first-order control; when a *second* ``Y12`` is not
   read, the verdict names every reason.

While a row is missing, the report prints only what is missing and what is invalid so far, and
reads *incomplete* (or *withheld*). Once no row is missing, it prints every number it computes
before the verdict. An exception inside the report is printed and gives *report failed*, which is
not a registered outcome.

**Known before registration.**

* The corner study's ``R12`` read local orders, with its all-orders statistic, 2.10 / 2.05
  (40 -> 80), 2.02 / 2.01 (80 -> 160) and 2.00 / 2.01 (160 -> 320). Its ``F12`` read 2.00 / 2.00
  at 160 -> 320. A review scored the same ``R12`` rows with this study's zeroth-order statistic:
  1.99 / 2.05, 2.00 / 2.01 and 2.00 / 2.01, with ``u`` 6.7e-6 / 6.6e-6. That passes this rule's
  gate with margins of only 1.09 (A) and 1.21 (F); its zeroth-order A(320) is 3.65e-4.
* The yz ridge's own ladder, measured before registration
  (``studies/results/fdtd_metasurface_ridges_scan/ladder.log``). At top 600, ``u_A`` is 7.5e-6
  and ``u_F`` 1.4e-6, which would bar any ridge whose A(320) is under 3.8e-4; ``R12``'s 3.65e-4
  would be barred. At top 800, ``u_A`` is 3.8e-6 and ``u_F`` 6.8e-7, a floor of 1.9e-4 and
  3.4e-5. The 600 rung sits 6.1e-7 from the 800 rung. The eps-shift proxy of item 2 puts
  ``Y12``'s A(320) at 2.1 to 3.2e-3.
* Split by wavelength, C7's M3 s rows read self-convergence orders 1.97 / 1.70 (0.95), 1.63 /
  2.14 (1.0) and 1.66 / 1.47 (1.05). The low order is not confined to one wavelength.
* No FDTD number of these ridges was seen before the rule was written. Three dry runs then
  tried the harness on epsilon-6 ridges against a toy ladder (N 20 to 60), each with module
  constants overridden from outside. ``studies/results/fdtd_metasurface_ridges_dryrun/README.md``
  lists every override, pass and digest; the numbers they printed were seen.
  * *Dry run 1* used a first draft (SHA-256 ``53828629...``) at rungs 30 / 60 / 120. The draft's
    ``Y12`` and ``X12`` were Bloch-differenced over 4 cells; its ``Axis1D`` ``Y12a`` is today's
    ``Y12``. Both read local orders 2.00 / 2.08 (30 -> 60) and 1.96 / 2.03 (60 -> 120). The
    Bloch twin matched the Bloch ``Y12`` to 2.8e-15. The Bloch twin of the corner ridge read
    1.52 / 1.38, then 1.99 / 2.00. The film reproduced the corner study's rows to 0.0.
  * *Dry run 2* used a second draft (SHA-256 ``babf95b1...``) at rungs 30 / 60 / 120 / 240, with
    today's ``Axis1D`` ``Y12`` and ``X12``. They read 2.00 / 2.08, 1.96 / 2.03 and 1.93 / 2.02.
    The ``Axis1D`` twin matched to 1.8e-15 over 144 numbers. The narrower-ridge control read
    1.54 / 1.17 (60 -> 120) and 1.27 / 1.08 (120 -> 240): *low*. The film's 30-cell rung was
    refused by the plan, and the report then failed on the refused row (fixed here).
  * *Dry run 3* used a third draft (SHA-256 ``3a8ea222...``) at rungs 30 / 60 / 120 / 240. The
    registered file differs from it by the third review's corrections: the 800 rung, the
    finite guard on local orders, the labels of an uncomputable twin gap and premise, and
    wording. It ran end to end. The
    refused film rung made the verdict *invalid*, as the rule says. ``Y12`` and ``X12`` read as
    in dry run 2 and matched to 1.8e-15. The narrower-ridge control read 1.54 / 1.17 and
    1.27 / 1.08; the wider one read 1.66 / 0.97 and 0.67 / 0.89. Both are *low*. The xz premise
    printed nothing when the corner study's ladder was missing; that line was added.
* ``TWIN_LINE`` is 1e-10, C7's line for "the same numbers". That is tens of thousands of times
  both measured gaps, and far below the discretization error a real asymmetry would show.
* ``UPPER_LINE`` comes from a review's RCWA model. Pure second-order proxies of ``Y12``'s error
  read 1.92 to 2.14 at the judged steps. A first-order part opposing a dispersion-like grid
  error read 2.2 to 4.2, classed *second* without the line. The ``R12`` rows read at most 2.10.
* Three reviews before registration changed the rule:
  * the Bloch scenes were dropped (the identity above);
  * the judged steps moved to 80 -> 160 and 160 -> 320. That was prompted by the RCWA scan and
    by a review's eps-shift proxy calibrated on the corner study's FDTD rows, not by an FDTD
    row of these ridges. For a purely second-order error the proxy put ``Y12``'s F order at
    1.61 to 1.74 at 40 -> 80 (worst cell 1.05, p) and at least 1.92 at 80 -> 160;
  * the rungs went to 320 for every scene;
  * the reference gate became a fiftieth of the finest judged error, and the ridges' ladder
    gained a rung at 800 so that its spread does not bar a ridge as accurate as ``R12``;
  * the run-length line moved to the 160 -> 320 change;
  * the first-order control was added, in two signs;
  * the upper line and the xz premise were added;
  * the dependency hashes and the registered copy were added, and the verdict's order was
    written down. Two outcomes changed with it: a run the plan refuses is now *invalid* (the
    first draft counted it as missing, so dry run 1's first pass read *incomplete*); and the
    verdict is now also *withheld* when the code digests, the registered copy, an imported
    study file or the digest of the corner study's rows changes.

Run: ``python studies/fdtd_metasurface_ridges.py [register rcwa fd report]``.
"""
import hashlib
import inspect
import json
import math
import sys
import time
import traceback
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fdtd_metasurface_corners as cs                                            # noqa: E402
import fdtd_metasurface_rcwa as c7                                               # noqa: E402
import numpy as np                                                                # noqa: E402
import wavebit                                                                    # noqa: E402
import wavebit.fd as fd                                                           # noqa: E402
from scipy.optimize import brentq                                                 # noqa: E402
from wavebit._core.result import Result                                          # noqa: E402

RESULTS = HERE / 'results'
LOG = RESULTS / 'fdtd_metasurface_ridges.jsonl'
REPORT = RESULTS / 'fdtd_metasurface_ridges_report.log'
REGISTRATION = RESULTS / 'fdtd_metasurface_ridges_registration.json'
#: the study files this one imports; their hashes are part of the rule
DEPENDENCIES = (HERE / 'fdtd_metasurface_corners.py', HERE / 'fdtd_metasurface_rcwa.py')

WAVELENGTHS = cs.WAVELENGTHS
BAND = cs.BAND
H, EPS_S, THETA, ZETA = cs.H, cs.EPS_S, cs.THETA, cs.ZETA
PERIODS, LONG_PERIODS = cs.PERIODS, 180
JAX_FROM = cs.JAX_FROM
LADDER = cs.LADDER
#: the ridges' own ladder: the corner study's and a rung at 800 (module docstring)
YX_LADDER = cs.LADDER + (800,)
POLS = cs.POLS
EPS_P = 12.0
RUNGS = (40, 80, 160, 320)
JUDGED = ((80, 160), (160, 320))
RUN_LENGTH_RUNG = 80

#: The rule's lines (module docstring).
DECAY_LINE, THETA_GAP = 1e-5, 1e-9
REFERENCE_SHARE, FLOOR_SHARE = 1.0 / 50.0, 0.1
ORDER_LINE, UPPER_LINE, LOW_LINE = 1.8, 2.3, 1.75
SAME_LINE = 1e-10
#: set from the dry runs' 2.8e-15 and 1.8e-15 before registration (module docstring)
TWIN_LINE = 1e-10
#: the first-order control's change of the ridge's width, in cells of each rung, and its signs
CONTROL_SHIFT = 1.0 / 32.0
CONTROL_SIGNS = (-1, 1)

#: geometry: the periodic axis, its cell, the ridge's width and axis offset (cells), phi
GEOM = {
    'Y': dict(periodic='y', P=0.50, width=0.20, offset=0.15, phi=30.0),
    'X': dict(periodic='x', P=0.50, width=0.20, offset=0.15, phi=-60.0),
}
#: the uniform axis's period in RCWA (one Fourier order, so any length)
RCWA_UNIFORM = 1.0
SCENES = {'Y12': 'Y', 'X12': 'X', 'F12': None}


# -- geometry and orders -----------------------------------------------------------------------

def centre(geom, n):
    """The ridge's axis along its periodic axis at rung ``n``: ``offset`` cells past
    ``floor(cells/2)`` from the cell's low edge (the cell is centred on the origin)."""
    g = GEOM[geom]
    cells = round(g['P'] * n)
    if abs(g['P'] * n - cells) > 1e-9:
        raise AssertionError(f'{geom}: the cell {g["P"]} is not whole cells at {n}')
    return -g['P'] / 2 + (cells // 2 + g['offset']) / n


def theta_at(wl):
    return math.degrees(math.asin(math.sin(math.radians(THETA)) * wl))


def propagating(geom):
    """The orders propagating on either side at any wavelength; the study reads the zeroth
    alone (module docstring), and refuses a geometry with more."""
    if geom is None:
        return [(0, 0)]
    g = GEOM[geom]
    s = math.sin(math.radians(THETA))
    kx, ky = s * math.cos(math.radians(g['phi'])), s * math.sin(math.radians(g['phi']))
    out = []
    for k in range(-6, 7):
        m, nn = (k, 0) if g['periodic'] == 'x' else (0, k)
        qx = kx + (m / g['P'] if g['periodic'] == 'x' else 0.0)
        qy = ky + (nn / g['P'] if g['periodic'] == 'y' else 0.0)
        if any(wl * math.hypot(qx, qy) < math.sqrt(EPS_S) for wl in WAVELENGTHS):
            out.append((m, nn))
    if out != [(0, 0)]:
        raise AssertionError(f'{geom}: orders {out} propagate; the study reads the zeroth only')
    return out


def cell_keys():
    """Every cell's key: incident polarisation, side, quantity (the zeroth order)."""
    return [f'{pol}|{side}|{q}' for pol in POLS for side, power in (('r', 'R'), ('t', 'T'))
            for q in (f'{side}_s', f'{side}_p', power)]


def build(name, n, pol, periods=PERIODS):
    """The public scene of ``name`` at ``n`` cells per wavelength."""
    geom = SCENES[name]
    if geom is None:
        return cs.build('F12', n, pol, periods)
    g = GEOM[geom]
    propagating(geom)
    dx = 1.0 / n
    sheet = H + cs.SHEET
    r_plane, t_plane = sheet + cs.GAP_R, -cs.GAP_T
    _, z_lo, z_hi = cs.z_grid(n, ZETA, t_plane - cs.CLEAR, r_plane + cs.CLEAR, cs.PML_CELLS)
    glass = fd.Material(eps=EPS_S)
    deep = z_lo - 2 * dx
    sub = fd.Structure(fd.Box(size=(10.0, 10.0, abs(deep)), center=(0.0, 0.0, deep / 2)), glass)
    c = centre(geom, n)
    if g['periodic'] == 'x':
        body = fd.Box(size=(g['width'], 10.0, H), center=(c, 0.0, H / 2))
        size, uniform = (g['P'], dx), 'y'
    else:
        body = fd.Box(size=(10.0, g['width'], H), center=(0.0, c, H / 2))
        size, uniform = (dx, g['P']), 'x'
    ridge = fd.Structure(body, fd.Material(eps=EPS_P))
    pw = fd.PlaneWave('pw', z=sheet, direction='-z', theta=THETA, phi=g['phi'],
                      polarization=pol, band=fd.Band(**BAND))
    monitors = (fd.Diffraction('r', z=r_plane, side='reflect', wavelengths=WAVELENGTHS,
                               reference_plane=H, incident_plane=H),
                fd.Diffraction('t', z=t_plane, side='transmit', wavelengths=WAVELENGTHS,
                               reference_plane=0.0, incident_plane=H))
    layer = dict(cells=cs.PML_CELLS, reflection=cs.PML_REFLECTION)
    return fd.Scene(domain=fd.Domain(size=(size[0], size[1], z_hi - z_lo),
                                     center=(0.0, 0.0, 0.5 * (z_hi + z_lo))),
                    boundaries=fd.Boundaries(**fd.Bloch.from_source(pw, axis1d=uniform),
                                             z=(fd.PML(background=glass, **layer),
                                                fd.PML(**layer))),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(periods=periods),
                    structures=(sub, ridge), sources=(pw,), monitors=monitors)


def rcwa_point(geom, wl, N, width=None):
    """``{cell key: complex}`` of the centred ridge (``geom``, optionally ``width`` wide) or of
    the film (``None``) at ``wl``."""
    th = math.radians(theta_at(wl))
    if geom is None:
        structure = [dict(background=cs.SCENES['F12']['eps_p'], shapes=[])]
        trunc, period, phi = [0, 0], [1.0, 1.0], cs.PHI
    else:
        g = GEOM[geom]
        w = g['width'] if width is None else width
        if g['periodic'] == 'x':
            lx, ly, trunc, period = w, RCWA_UNIFORM, [N, 0], [g['P'], RCWA_UNIFORM]
        else:
            lx, ly, trunc, period = RCWA_UNIFORM, w, [0, N], [RCWA_UNIFORM, g['P']]
        structure = [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0, lx=lx,
                                                       ly=ly, material=EPS_P)])]
        phi = g['phi']
    res = wavebit.rcwa(backend=0, top=1.0, bot=EPS_S, theta=th, phi=math.radians(phi),
                       psi=math.pi / 2, truncation_order=trunc, period=period, wavelength=wl,
                       structure=structure, thickness=[H]).solve()
    out = {}
    for pol, sub in (('s', res.s_inc), ('p', res.p_inc)):
        o = Result(given=sub, order_window=res.order_window).order(0, 0)
        for side, power in (('r', 'R'), ('t', 'T')):
            for q in (f'{side}_s', f'{side}_p', power):
                out[f'{pol}|{side}|{q}'] = complex(np.asarray(getattr(o, q)))
    return out


# -- provenance and rows -----------------------------------------------------------------------

def sha_lf(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def control_job(sign, i, n):
    """``(key, (geom, wavelength, N, width))`` of one first-order control reference."""
    width = GEOM['Y']['width'] + sign * CONTROL_SHIFT / n
    N = LADDER[-1]
    return f'ctrl|{sign:+d}|{i}|{n}|{N}|{width!r}', ('Y', WAVELENGTHS[i], N, width)


def provenance():
    """``fd`` keys the FDTD rows, ``rcwa`` the ladder rows, ``ctrl`` the control rows: the
    wavebit trees, the functions and every constant the rows read (this file's and the corner
    study's), and the versions."""
    c7.PROV = c7.provenance()
    versions = c7.PROV['versions']
    shared = dict(WAVELENGTHS=WAVELENGTHS, BAND=BAND, H=H, EPS_S=EPS_S, THETA=THETA, ZETA=ZETA,
                  EPS_P=EPS_P, GEOM=GEOM, cs=[cs.SHEET, cs.GAP_R, cs.GAP_T, cs.CLEAR,
                                              cs.PML_CELLS, cs.PML_REFLECTION, cs.PHI,
                                              cs.SCENES['F12']])
    code = [inspect.getsource(f) for f in (centre, theta_at, propagating, build, fd_row,
                                           cs.build, cs.z_grid)]
    fd_digest = hashlib.sha256(json.dumps(
        [c7.PROV['fd_tree'], c7.PROV['wavebit_rest'], code, shared, JAX_FROM, versions],
        sort_keys=True, default=repr).encode()).hexdigest()[:16]
    rcode = [inspect.getsource(f) for f in (rcwa_point, theta_at, propagating)]
    rcwa_digest = hashlib.sha256(json.dumps(
        [c7.PROV['wavebit_rest'], rcode, shared, RCWA_UNIFORM, YX_LADDER, versions],
        sort_keys=True, default=repr).encode()).hexdigest()[:16]
    ctrl_digest = hashlib.sha256(json.dumps(
        [rcwa_digest, inspect.getsource(control_job), CONTROL_SHIFT, CONTROL_SIGNS, LADDER[-1],
         RUNGS], sort_keys=True, default=repr).encode()).hexdigest()[:16]
    return dict(fd=fd_digest, rcwa=rcwa_digest, ctrl=ctrl_digest, commit=c7.PROV['commit'],
                dirty=c7.PROV['dirty'], env=c7.PROV['env'])


PROV = None


def emit(row):
    c7.emit(LOG, row)


def load():
    """``{key: row}``, the last row per key under the current provenance; a cut or NUL-padded
    line is skipped with a note."""
    out = {}
    if LOG.exists():
        for k, line in enumerate(open(LOG, encoding='utf-8')):
            line = line.replace('\0', '').strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f'note: {LOG.name} line {k + 1} is not a whole row; skipped', flush=True)
                continue
            if row.get('prov') == PROV.get(row.get('kind')):
                out[row['key']] = row
    return out


def fd_key(name, n, pol, periods):
    return f'fd|{name}|{n}|{pol}|{periods}'


def fd_row(name, n, pol, periods):
    """One public solve as a JSON row. A plan refused by name is ``'refused'`` (invalid); any
    other exception is ``'error'`` and is solved again next time."""
    backend = 'jax' if n >= JAX_FROM else 'numpy'
    base = dict(kind='fd', key=fd_key(name, n, pol, periods), prov=PROV['fd'], scene=name, n=n,
                pol=pol, periods=periods, backend=backend, commit=PROV['commit'],
                dirty=PROV['dirty'], env=PROV['env'])
    t0 = time.perf_counter()
    stage = 'plan'
    try:
        scene = build(name, n, pol, periods)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            plan = fd.plan(scene)
            stage = 'run'
            res = fd.run(scene, plan=plan, backend=backend)
        stage = 'read'
        values = {}
        for side, power in (('r', 'R'), ('t', 'T')):
            o = res[side].orders.order(0, 0)
            for q in (f'{side}_s', f'{side}_p', power):
                values[f'{side}|{q}'] = [c7._c(v) for v in np.asarray(getattr(o, q))]
        names = sorted({f'{w.category.__name__}: {str(w.message)[:120]}' for w in caught})
        decay = {k: (None if v is None else float(v))
                 for k, v in (res.diagnostics.get('decay') or {}).items()}
        stable = res.diagnostics.get('stable')
        row = dict(base, status='ok', values=values, decay=decay, warnings=names,
                   theta_at=[float(v) for v in np.asarray(res['r'].theta_at)],
                   stable=stable if isinstance(stable, bool) else None, steps=int(plan.steps),
                   grid=list(plan.grid.n), wall=time.perf_counter() - t0)
        json.dumps(row, default=c7._json)
        return row
    except Exception as err:                                               # noqa: BLE001
        refused = stage == 'plan' and isinstance(err, ValueError)
        return dict(base, status='refused' if refused else 'error', stage=stage,
                    error=f'{type(err).__name__}: {err}'[:600],
                    trace=traceback.format_exc()[-2000:], wall=time.perf_counter() - t0)


def corner_rows():
    """The corner study's rows this study reads -- its ``F12`` and ``R12`` FDTD rows and its
    ``R12`` RCWA ladder -- as ``{key: row}`` (that log carries no provenance field; the
    registration records their digest), and that digest."""
    out = {}
    for line in open(cs.LOG, encoding='utf-8'):
        line = line.replace('\0', '').strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('key', '').startswith(('fd|F12|', 'fd|R12|', 'rcwa|R12|')):
            out[r['key']] = r
    digest = hashlib.sha256(json.dumps(
        [[k, out[k].get('status'), out[k].get('values')] for k in sorted(out)],
        sort_keys=True).encode()).hexdigest()[:16]
    return out, digest


def dependencies():
    return {p.name: sha_lf(p) for p in DEPENDENCIES}


def registered_copy(sha):
    return RESULTS / f'fdtd_metasurface_ridges_registered_{sha[:16]}.py.txt'


# -- the commands ------------------------------------------------------------------------------

def part_register():
    """Record the rule before any FDTD row of these scenes, with a copy of this file named by
    its hash (written once; a superseded registration is moved aside by hand and kept)."""
    if REGISTRATION.exists():
        print(f'{REGISTRATION.name} exists; the rule was registered before', flush=True)
        return
    if LOG.exists() and any('"kind": "fd"' in line for line in open(LOG, encoding='utf-8')):
        raise SystemExit(f'{LOG.name} already holds fd rows: a registration now would postdate '
                         f'data')
    sha = sha_lf(Path(__file__))
    copy = registered_copy(sha)
    if copy.exists() and sha_lf(copy) != sha:
        raise SystemExit(f'{copy.name} holds other text')
    RESULTS.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(Path(__file__).read_bytes())
    record = dict(file='studies/fdtd_metasurface_ridges.py', sha256=sha,
                  registered=time.strftime('%Y-%m-%dT%H:%M:%S%z'), commit=PROV['commit'],
                  dirty=PROV['dirty'], digests={k: PROV[k] for k in ('fd', 'rcwa', 'ctrl')},
                  dependencies=dependencies(), corner_rows=corner_rows()[1],
                  lines=dict(TWIN_LINE=TWIN_LINE, UPPER_LINE=UPPER_LINE, ORDER_LINE=ORDER_LINE,
                             LOW_LINE=LOW_LINE, REFERENCE_SHARE=REFERENCE_SHARE,
                             CONTROL_SHIFT=CONTROL_SHIFT), copy=copy.name)
    REGISTRATION.write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(record), flush=True)


def ladder_key(geom, i, N):
    return f'rcwa|{geom}|{i}|{N}'


def part_rcwa():
    done = load()
    jobs = [(ladder_key(g, i, N), 'rcwa', (g, wl, N, None))
            for g in ('Y', 'X', None) for N in (YX_LADDER if g else (0,))
            for i, wl in enumerate(WAVELENGTHS)]
    jobs += [(*control_job(sign, i, n)[:1], 'ctrl', control_job(sign, i, n)[1])
             for sign in CONTROL_SIGNS for n in RUNGS for i in range(len(WAVELENGTHS))]
    for key, kind, (g, wl, N, width) in jobs:
        if key in done:
            continue
        t0 = time.perf_counter()
        vals = rcwa_point(g, wl, N, width)
        row = dict(kind=kind, key=key, prov=PROV[kind], geom=g, wavelength=wl, N=N, width=width,
                   values={k: c7._c(v) for k, v in vals.items()}, wall=time.perf_counter() - t0)
        emit(row)
        done[key] = row


def settled(row):
    return (row is not None and row.get('status') == 'ok' and len(row.get('decay') or {}) == 2
            and all(v is not None and v < DECAY_LINE for v in row['decay'].values()))


def part_fd():
    """Every rung of every scene at 120 periods, rung 80 at 180 (the run-length check), and the
    one 180-period repeat of a run over the decay line."""
    done = load()
    for name in SCENES:
        wanted = [(n, pol, PERIODS) for n in RUNGS for pol in POLS]
        wanted += [(RUN_LENGTH_RUNG, pol, LONG_PERIODS) for pol in POLS]
        for n, pol, periods in wanted:
            key = fd_key(name, n, pol, periods)
            row = done.get(key)
            if row is None or row.get('status') == 'error':
                row = fd_row(name, n, pol, periods)
                emit(row)
                done[key] = row
            if periods == PERIODS and row.get('status') == 'ok' and not settled(row):
                again = fd_key(name, n, pol, LONG_PERIODS)
                if done.get(again) is None or done[again].get('status') == 'error':
                    done[again] = fd_row(name, n, pol, LONG_PERIODS)
                    emit(done[again])


# -- the rule ----------------------------------------------------------------------------------

def missing(row):
    """A run not made yet, or one that raised and will be made again: a gap, not a failure."""
    return row is None or row.get('status') == 'error'


def rung_run(rows, name, n, pol):
    first = rows.get(fd_key(name, n, pol, PERIODS))
    if first is not None and first.get('status') == 'ok' and not settled(first):
        return rows.get(fd_key(name, n, pol, LONG_PERIODS))
    return first


def finite_values(values, want, length):
    """Whether ``values`` holds exactly the keys ``want``, each ``length`` finite pairs."""
    return (set(values) == set(want) and all(
        len(values[k]) == length and all(math.isfinite(x) for pair in values[k] for x in pair)
        for k in want))


def validity(row, settle=True):
    if row is None:
        return ['never run']
    if row.get('status') != 'ok':
        return [f'{row.get("status")}: {row.get("error", "")[:200]}']
    bad = []
    if settle and not settled(row):
        bad.append(f'decay {row["decay"]}')
    if any('GrowingField' in w for w in row['warnings']):
        bad.append('growth warning')
    if settle and any('UnsettledField' in w for w in row['warnings']):
        bad.append('unsettled warning')
    if row.get('stable') is False:
        bad.append('stable False')
    gap = c7.worst(abs(t - theta_at(wl)) for t, wl in zip(row['theta_at'], WAVELENGTHS))
    if not gap <= THETA_GAP:
        bad.append(f'theta_at off by {gap}')
    if not finite_values(row['values'], {k.split('|', 1)[1] for k in cell_keys()},
                         len(WAVELENGTHS)):
        bad.append('cells missing, extra, short or not finite')
    return bad


def values_of(runs, n, key_of=lambda side, q: f'{side}|{q}'):
    """``{(cell key, i): complex}`` at rung ``n`` over both incident polarisations (the zeroth
    order needs no lateral phase). A run that is not ``'ok'`` gives NaN cells."""
    out = {}
    for key in cell_keys():
        pol, side, q = key.split('|')
        row = runs.get((n, pol)) or {}
        got = (row.get('values') if row.get('status') == 'ok' else None) or {}
        got = got.get(key_of(side, q))
        for i in range(len(WAVELENGTHS)):
            out[(key, i)] = complex(*got[i]) if got and i < len(got) else complex('nan')
    return out


def is_eff(cell):
    return cell[0].split('|')[-1] in ('R', 'T')


def ladder(rows, keys, Ns, pick=lambda k: k):
    """``{N: {cell key: [complex per wavelength]}}`` from ladder rows ``keys(i, N)``; ``None`` if
    a row is missing. ``pick`` maps a row's key to the study's cell key (or ``None``)."""
    out = {}
    for N in Ns:
        vals = {}
        for i in range(len(WAVELENGTHS)):
            row = rows.get(keys(i, N))
            if row is None:
                return None
            for k, v in row['values'].items():
                kk = pick(k)
                if kk is not None:
                    vals.setdefault(kk, [complex('nan')] * len(WAVELENGTHS))[i] = complex(*v)
        out[N] = vals
    return out


def errors(runs, refs, key_of=lambda side, q: f'{side}|{q}'):
    """``(A, F)`` over ``RUNGS``; ``refs[n]`` is the reference for rung ``n``. A cell the
    reference lacks is NaN."""
    A, F = [], []
    for n in RUNGS:
        v = values_of(runs, n, key_of)
        ref = refs[n]
        d = {c: abs(x - (ref.get(c[0]) or [complex('nan')] * len(WAVELENGTHS))[c[1]])
             for c, x in v.items()}
        A.append(c7.worst(e for c, e in d.items() if not is_eff(c)))
        F.append(c7.worst(e for c, e in d.items() if is_eff(c)))
    return A, F


def spread(lad):
    """``(u_A, u_F)``: the largest distance of the three truncations below the top from it."""
    top = lad[max(lad)]
    lower = [lad[N] for N in sorted(lad)[-4:-1]] or [top]
    return tuple(c7.worst(abs(lo[k][i] - top[k][i]) for lo in lower for k in top
                          for i in range(len(WAVELENGTHS)) if (k.split('|')[-1] in ('R', 'T'))
                          == eff) for eff in (False, True))


def gate(A, F, uA, uF):
    """Criterion 1's reference gate: ``u`` within ``REFERENCE_SHARE`` of the smallest judged
    error."""
    judged = sorted({n for step in JUDGED for n in step})
    gA = c7.worst([-A[RUNGS.index(n)] for n in judged])
    gF = c7.worst([-F[RUNGS.index(n)] for n in judged])
    return uA <= REFERENCE_SHARE * -gA and uF <= REFERENCE_SHARE * -gF, -gA, -gF


def local(X, a, b):
    i, j = RUNGS.index(a), RUNGS.index(b)
    ok = all(math.isfinite(x) and x > 0 for x in (X[i], X[j]))
    return math.log(X[i] / X[j]) / math.log(b / a) if ok else float('nan')


def self_order(runs, triple):
    """The reference-free order of a three-rung run (lifted study), amplitudes and efficiencies."""
    v = [values_of(runs, n) for n in triple]
    out = []
    for eff in (False, True):
        d1 = c7.worst(abs(v[0][c] - v[1][c]) for c in v[0] if is_eff(c) == eff)
        d2 = c7.worst(abs(v[1][c] - v[2][c]) for c in v[0] if is_eff(c) == eff)
        a, b, c = triple

        def f(p):
            return (a ** -p - b ** -p) / (b ** -p - c ** -p) - d1 / d2
        try:
            out.append(float(brentq(f, 0.05, 8.0)) if f(0.05) < 0 < f(8.0) else float('nan'))
        except (ValueError, ZeroDivisionError):
            out.append(float('nan'))
    return tuple(out)


def classify(A, F):
    four = [local(X, a, b) for a, b in JUDGED for X in (A, F)]
    if all(ORDER_LINE <= o <= UPPER_LINE for o in four):
        return 'second', four
    if all(o <= LOW_LINE for o in four):
        return 'low', four
    return 'mixed', four


def part_report():
    lines = []

    def say(text):
        lines.append(text)
        print(text, flush=True)
    verdict = 'report failed (not a registered outcome)'
    try:
        verdict = _report(say)
    except Exception as err:                                               # noqa: BLE001
        say(f'report failed: {type(err).__name__}: {err}')
        say(traceback.format_exc()[-1500:])
    finally:
        say(f'verdict: {verdict}')
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(REPORT, 'a', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(lines) + '\n\n')
    return verdict


def _fmt(X):
    return ', '.join(f'{x:.3e}' for x in X)


def _orders(four):
    return ', '.join(f'{o:.3f}' for o in four)


def _report(say):
    reg = json.loads(REGISTRATION.read_text(encoding='utf-8')) if REGISTRATION.exists() else {}
    crows, cdigest = corner_rows()
    sha = sha_lf(Path(__file__))
    changed = []
    if reg.get('sha256') != sha:
        changed.append('this file')
    copy = RESULTS / reg.get('copy', registered_copy(sha).name)
    if not copy.exists() or sha_lf(copy) != reg.get('sha256'):
        changed.append('the registered copy')
    if reg.get('dependencies') != dependencies():
        changed.append('an imported study file')
    if reg.get('digests') != {k: PROV[k] for k in ('fd', 'rcwa', 'ctrl')}:
        changed.append('the code digests')
    if reg.get('corner_rows') != cdigest:
        changed.append("the corner study's rows")
    say(f'ridges report {time.strftime("%Y-%m-%dT%H:%M:%S%z")}: file {sha[:16]}, registered '
        f'{reg.get("sha256", "never")[:16]} at {reg.get("registered")}; digests fd {PROV["fd"]} '
        f'rcwa {PROV["rcwa"]} ctrl {PROV["ctrl"]}; changed since registration: '
        f'{changed or "nothing"}')
    rows = load()
    gaps, bad = [], []
    runs = {}
    for name in SCENES:
        r = {}
        for n in RUNGS:
            for pol in POLS:
                row = rung_run(rows, name, n, pol)
                r[(n, pol)] = row
                if missing(row):
                    gaps.append((name, n, pol))
                else:
                    bad += [f'{name} {n} {pol}: {w}' for w in validity(row)]
        for pol in POLS:
            for periods in (PERIODS, LONG_PERIODS):
                if missing(rows.get(fd_key(name, RUN_LENGTH_RUNG, pol, periods))):
                    gaps.append((name, RUN_LENGTH_RUNG, pol, periods))
        runs[name] = r
    refs = {g: ladder(rows, lambda i, N, g=g: ladder_key(g, i, N), YX_LADDER if g else (0,))
            for g in ('Y', 'X', None)}
    gaps += [('reference', g) for g, v in refs.items() if v is None]
    ctrl = {}
    for sign in CONTROL_SIGNS:
        ctrl[sign] = {}
        for n in RUNGS:
            vals = {}
            for i in range(len(WAVELENGTHS)):
                key, (_, _, N, width) = control_job(sign, i, n)
                row = rows.get(key)
                if row is None or row.get('N') != N or row.get('width') != width:
                    gaps.append(('control reference', sign, n, i))
                    continue
                if set(row['values']) != set(cell_keys()) or not all(
                        len(v) == 2 and all(math.isfinite(x) for x in v)
                        for v in row['values'].values()):
                    bad.append(f'control reference {sign} {n} {i}: values missing or not finite')
                for k, v in row['values'].items():
                    vals.setdefault(k, [complex('nan')] * len(WAVELENGTHS))[i] = complex(*v)
            ctrl[sign][n] = vals
    if gaps:
        say(f'missing: {gaps}')
        if bad:
            say(f'not valid so far: {bad}')
        return f'withheld ({", ".join(changed)} changed)' if changed else 'incomplete'
    ty, tx = refs['Y'][YX_LADDER[-1]], refs['X'][YX_LADDER[-1]]
    say(f'reference Y against its turn X at N {YX_LADDER[-1]}: '
        f'{c7.worst(abs(ty[k][i] - tx[k][i]) for k in ty for i in range(len(WAVELENGTHS))):.1e}')
    out = {}
    for name, geom in SCENES.items():
        lad = refs[geom]
        A, F = errors(runs[name], {n: lad[max(lad)] for n in RUNGS})
        uA, uF = spread(lad)
        ok, gA, gF = gate(A, F, uA, uF)
        if geom is not None and not ok:
            bad.append(f'{name}: the reference cannot judge (u {uA:.1e}/{uF:.1e} against '
                       f'{REFERENCE_SHARE * gA:.1e}/{REFERENCE_SHARE * gF:.1e})')
        loc = {(a, b): (local(A, a, b), local(F, a, b)) for a, b in zip(RUNGS, RUNGS[1:])}
        selfo = [self_order(runs[name], RUNGS[k:k + 3]) for k in range(len(RUNGS) - 2)]
        moves = {False: [], True: []}
        for pol in POLS:
            a = rows.get(fd_key(name, RUN_LENGTH_RUNG, pol, PERIODS))
            b = rows.get(fd_key(name, RUN_LENGTH_RUNG, pol, LONG_PERIODS))
            why = validity(a, settle=False) + validity(b)
            if why:
                bad.append(f'{name} run-length pair {pol}: {why}')
                continue
            for key in a['values']:
                eff = key.split('|')[-1] in ('R', 'T')
                moves[eff] += [abs(complex(*x) - complex(*y))
                               for x, y in zip(a['values'][key], b['values'][key])]
        dA, dF = c7.worst(moves[False]), c7.worst(moves[True])
        lo_n, hi_n = JUDGED[-1]
        vf, vc = values_of(runs[name], hi_n), values_of(runs[name], lo_n)
        DA = c7.worst(abs(vf[c] - vc[c]) for c in vf if not is_eff(c))
        DF = c7.worst(abs(vf[c] - vc[c]) for c in vf if is_eff(c))
        if not (dA <= FLOOR_SHARE * DA and dF <= FLOOR_SHARE * DF):
            bad.append(f'{name}: run length moves {dA:.1e}/{dF:.1e} against '
                       f'{FLOOR_SHARE * DA:.1e}/{FLOOR_SHARE * DF:.1e}')
        out[name] = dict(A=A, F=F, loc=loc)
        say(f'{name}: A {_fmt(A)}; F {_fmt(F)}; local orders '
            + '; '.join(f'{a}->{b} {la:.3f}/{lf:.3f}' for (a, b), (la, lf) in loc.items())
            + f' (judged {list(JUDGED)}); self-convergence '
            + ', '.join(f'{RUNGS[k]}-{RUNGS[k + 2]} {p[0]:.3f}/{p[1]:.3f}'
                        for k, p in enumerate(selfo))
            + f'; u {uA:.1e}/{uF:.1e} against {REFERENCE_SHARE * gA:.1e}/'
            f'{REFERENCE_SHARE * gF:.1e}; run length {dA:.1e}/{dF:.1e} against '
            f'{FLOOR_SHARE * DA:.1e}/{FLOOR_SHARE * DF:.1e}')
    # the F12 controls
    f12 = out['F12']
    ok_f = all(x >= ORDER_LINE for pair in f12['loc'].values() for x in pair)
    say(f'control F12: local orders all at least {ORDER_LINE}: {ok_f}')
    if not ok_f:
        bad.append('control F12 below the line')
    diffs = []
    want = {f'{side}|0|{q}' for side, power in (('r', 'R'), ('t', 'T'))
            for q in (f'{side}_s', f'{side}_p', power)}
    for n in RUNGS:
        for pol in POLS:
            old, new = crows.get(f'fd|F12|{n}|{pol}'), runs['F12'][(n, pol)]
            if (old is None or old.get('status') != 'ok' or new is None
                    or new.get('status') != 'ok'
                    or not finite_values(old['values'], want, len(WAVELENGTHS))):
                diffs.append(float('nan'))
                continue
            for key, vals in new['values'].items():
                side, q = key.split('|')
                diffs += [abs(complex(*x) - complex(*y))
                          for x, y in zip(vals, old['values'][f'{side}|0|{q}'])]
    same = c7.worst(diffs)
    say(f"control F12 against the corner study's rows: {same:.1e} (line {SAME_LINE})")
    if not same <= SAME_LINE:
        bad.append("control F12 does not reproduce the corner study's rows")
    # the xz premise: the corner study's R12 rows under this rule
    r12 = {(n, pol): crows.get(f'fd|R12|{n}|{pol}') for n in RUNGS for pol in POLS}
    premise_bad = [f'R12 {n} {pol}' for (n, pol), r in r12.items()
                   if not r or r.get('status') != 'ok' or not settled(r)]
    rlad = ladder(crows, lambda i, N: f'rcwa|R12|{i}|{N}', LADDER,
                  pick=lambda k: (lambda p: f'{p[0]}|{p[1]}|{p[3]}' if p[2] == '0' else None)(
                      k.split('|')))
    if rlad is None:
        premise_bad.append("the corner study's R12 ladder")
        premise = 'not computable'
        say(f'xz premise (corner study R12 under this rule): not computable ({premise_bad})')
    else:
        rA, rF = errors(r12, {n: rlad[max(rlad)] for n in RUNGS},
                        key_of=lambda side, q: f'{side}|0|{q}')
        ruA, ruF = spread(rlad)
        rok, rgA, rgF = gate(rA, rF, ruA, ruF)
        r_class, r_four = classify(rA, rF)
        if not all(math.isfinite(o) for o in r_four):
            premise = f'not computable (orders {_orders(r_four)}, {premise_bad})'
        elif rok and not premise_bad:
            premise = r_class
        else:
            premise = f'{r_class} (gate {rok}, {premise_bad})'
        say(f'xz premise (corner study R12 under this rule): A {_fmt(rA)}; F {_fmt(rF)}; judged '
            f'orders {_orders(r_four)}; u {ruA:.1e}/{ruF:.1e} against '
            f'{REFERENCE_SHARE * rgA:.1e}/{REFERENCE_SHARE * rgF:.1e}; class {premise}')
    # the first-order control, both signs
    inert, control = False, []
    for sign in CONTROL_SIGNS:
        cA, cF = errors(runs['Y12'], ctrl[sign])
        c_class, c_four = classify(cA, cF)
        if not all(math.isfinite(o) for o in c_four):
            bad.append(f'first-order control {sign:+d}: an order is not finite')
        inert = inert or c_class == 'second'
        control.append(f'{sign:+d}: {c_class}')
        say(f'first-order control (ridge {"wider" if sign > 0 else "narrower"} by '
            f'{CONTROL_SHIFT:.4f} cells): A {_fmt(cA)}; F {_fmt(cF)}; judged orders '
            f'{_orders(c_four)}; class {c_class} (must not be second)')
    # symmetry
    twin = []
    for n in RUNGS:
        vy, vx = values_of(runs['Y12'], n), values_of(runs['X12'], n)
        twin += [abs(vy[c] - vx[c]) for c in vy]
    gap = c7.worst(twin)
    twin_ok = math.isfinite(gap)
    symmetric = twin_ok and gap <= TWIN_LINE
    say(f'X12 against Y12: {gap:.1e} (line {TWIN_LINE}): '
        + ('symmetric' if symmetric else 'NOT symmetric' if twin_ok
           else 'not computable (a Y12 or X12 run is not valid)'))
    y_class, y_four = classify(out['Y12']['A'], out['Y12']['F'])
    above = any(o > UPPER_LINE for o in y_four)
    say(f'Y12 class: {y_class} (judged orders {_orders(y_four)})'
        + ('; an order is above the upper line: cancellation is possible' if above else ''))
    if bad:
        say(f'not valid: {bad}')
    if changed:
        return f'withheld ({", ".join(changed)} changed since registration)'
    if bad:
        return 'invalid' + ('; also, the solver treats x and y differently'
                            if twin_ok and not symmetric else '')
    if not symmetric:
        return 'the solver treats x and y differently (the class is not read)'
    if y_class == 'second':
        reasons = ([f'the xz premise is {premise}'] if premise != 'second' else []) + (
            [f'rule inert (first-order control {control})'] if inert else [])
        if reasons:
            return 'second, not read: ' + '; '.join(reasons)
        return ("second: M3's yz cross-section converges at second order as a 2D ridge, as its xz "
                'cross-section does; what holds M3 down is three-dimensional (its vertical edges, '
                'its vertices, or how the field varies along finite edges)')
    if y_class == 'low':
        return ("low: M3's yz cross-section reproduces a low order as a 2D ridge; its y faces "
                'or its edges along x are enough to hold the order down')
    return 'mixed: inconclusive' + (' (an order above the upper line)' if above else '')


def main(argv):
    global PROV
    PROV = provenance()
    print(json.dumps({k: PROV[k] for k in ('fd', 'rcwa', 'ctrl', 'commit', 'dirty')}), flush=True)
    parts = argv or ['register', 'rcwa', 'fd', 'report']
    actions = dict(register=part_register, rcwa=part_rcwa, fd=part_fd, report=part_report)
    for part in parts:
        if part not in actions:
            raise SystemExit(f'unknown command {part!r}: one of {sorted(actions)}')
        if part in ('rcwa', 'fd') and not REGISTRATION.exists():
            raise SystemExit('register the rule before any row')
        actions[part]()


if __name__ == '__main__':
    main(sys.argv[1:])
