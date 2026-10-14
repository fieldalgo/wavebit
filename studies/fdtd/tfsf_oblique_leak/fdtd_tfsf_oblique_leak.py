"""How much an oblique TFSF box leaks when its incident field comes from a 1-D auxiliary grid.

Stage 2's TFSF box takes its incident field from a 1-D auxiliary grid. Along an axis the aux
grid IS the 3-D grid (same dx, same dt, same difference), so the incident field is an exact
discrete solution and the box cannot leak. Off an axis the 3-D points are projected onto the
propagation direction and the aux field is interpolated there (the incident-field-array method,
Taflove & Hagness ch. 5), and the incident field is no longer an exact solution of the 3-D
update: the scattered-field region sees the mismatch. The design's rule: if the leakage at
45 deg exceeds 1e-4 with the best variant, v1 supports axis-aligned TFSF only.

Nothing here imports ``wavebit.fdtd._legacy`` or either old engine. The stand-in is a vacuum Yee grid in
plain NumPy with ``wavebit.fdtd._legacy``'s conventions -- ``(3, nz, ny, nx)`` arrays, the Yee offsets of
``src/wavebit/fdtd/__init__.py``, H stored as ``Z0 H``, c = 1, ``H -= dt curl E`` with forward
differences then ``E += dt curl H`` with backward ones, ``dt = courant dx / sqrt(3)`` with the
default courant 0.5 -- periodic on all six faces. Periodic faces reflect nothing: the incident
field exists only inside the box, so the only thing that can reach the outside is leakage, and
the time window (the pulse through the whole box) ends before a leaked wave could matter. The
TFSF corrections are derived from that update and nothing else: with the stored field
``F = F_scat + chi F_inc`` (``chi`` the per-component total-field indicator of the closed box),
the curl at every node whose stencil crosses the surface is corrected by ``+/- sign * F_inc`` at
the crossing neighbour -- the ``before_H`` correction reads the aux E at step n, the ``before_E``
one reads the aux H at n + 1/2, the kernel's slot timing. Because the correction is built from
the stencil rather than from a textbook table of face terms, it is exact for any box the moment
the incident field is an exact discrete solution; that is what the 0 deg row checks. (In the
kernel the aux H must therefore be at n + 1/2 when ``before_E`` runs, while ``advance`` runs
once, after E: the term has to step its aux H inside the step, not in ``advance``.)

The pulse is a sine carrier under a Gaussian envelope of spectral sigma 0.15 f0 (the plan's
``Band(wavelength=0.55, df=0.3)`` order), started and ended at exp(-18), fed as a soft source
into the aux grid 4 cells before the box's first projected point; the aux grid is long enough
that its PEC ends return nothing within the window. Leak = the largest |E| or |H| component at
any scattered-field node over the run (read at >= 20 samples per period) over the largest aux
|e| across the box. Box: a cube of one wavelength of the band centre, a quarter-wavelength (>= 4
cells) margin. Every angle is the polar angle in the xz plane from +z (phi = 0); s has E along y,
p has E in the xz plane.

Aux-grid variants (the first two are the question; the last two are what it took to find the
floor):

* ``plain`` -- the aux grid with the 3-D dx and dt, linear interpolation at the projected
  distance, the continuum polarisation vectors.
* ``matched`` -- the standard matched-dispersion fix: the aux cell ``dxa`` chosen so that the
  1-D dispersion ``sin(k dxa/2)/dxa = sin(w dt/2)/dt`` at the band centre gives the 3-D grid's
  own k along the direction (solved from ``sum_a sin^2(k k_a/2) = (dx sin(w dt/2)/dt)^2``);
  same dt, linear interpolation.
* ``matched+cubic`` -- the same, four-point Lagrange interpolation.
* ``+K`` (``matched+cubic+K``) -- the same, plus the grid's own polarisation vectors at the
  centre: both discrete curls act on exp(i k.r) as ``i K x`` with ``K_a = 2 sin(k_a dx/2)/dx``,
  so the 3-D plane wave has ``K.E = 0`` and ``H = K x E / Omega``: E and H are transverse to K,
  not to k, and K is parallel to k only on an axis and on a lattice diagonal.

Numbers are CPU-only, on this machine, 2026-09-19 (NumPy 2.5.1, SciPy 1.18.0, 11 worker
processes on 12 logical CPUs, one thread each; 250 runs, 23731 s of worker time, most of it the
sixteen 80-cells-per-wavelength runs). Run from the repo root; an optional argument names the
JSON output (default ``artifacts/studies/fdtd_tfsf_oblique_leak.json``, not checked in);
``--quick`` runs only the 0/45 deg rows at 10 cells per wavelength with the mutant (seconds);
the printed output of the recorded run, fdtd_tfsf_oblique_leak.log, is archived outside the
repository. Measured (each cell: plain / matched / matched+cubic / +K):

| s-pol leak = peak scattered / peak incident | 0 deg | 15 deg | 30 deg | 45 deg | 60 deg |
| --- | --- | --- | --- | --- | --- |
| 10 cells/wavelength | 0 / 0 / 0 / 0 | 4.9e-02 / 2.6e-02 / 7.0e-03 / 2.3e-03 | 6.1e-02 / 1.6e-02 / 6.4e-03 / 1.9e-03 | 7.7e-02 / 3.4e-14 / 2.1e-14 / 2.1e-14 | 6.1e-02 / 1.6e-02 / 6.4e-03 / 1.9e-03 |
| 20 cells/wavelength | 0 / 0 / 0 / 0 | 1.8e-02 / 1.3e-02 / 2.1e-03 / 4.3e-04 | 1.7e-02 / 1.0e-02 / 2.0e-03 / 4.0e-04 | 2.4e-02 / 4.5e-14 / 2.2e-14 / 2.2e-14 | 1.7e-02 / 1.0e-02 / 2.0e-03 / 4.0e-04 |
| 40 cells/wavelength | 0 / 0 / 0 / 0 | 8.2e-03 / 6.8e-03 / 5.2e-04 / 1.0e-04 | 8.1e-03 / 5.1e-03 / 5.1e-04 / 1.0e-04 | 9.8e-03 / 2.2e-13 / 4.8e-14 / 4.8e-14 | 8.1e-03 / 5.1e-03 / 5.1e-04 / 1.0e-04 |
| 80 cells/wavelength | - | 4.0e-03 / 3.4e-03 / 1.3e-04 / 2.5e-05 | 3.9e-03 / 2.5e-03 / 1.3e-04 / 2.6e-05 | - | - |

| p-pol leak = peak scattered / peak incident | 0 deg | 15 deg | 30 deg | 45 deg | 60 deg |
| --- | --- | --- | --- | --- | --- |
| 10 cells/wavelength | 0 / 0 / 0 / 0 | 3.3e-02 / 2.3e-02 / 6.8e-03 / 2.2e-03 | 5.9e-02 / 1.9e-02 / 7.9e-03 / 2.2e-03 | 6.9e-02 / 6.5e-14 / 2.1e-14 / 2.1e-14 | 5.9e-02 / 1.9e-02 / 7.9e-03 / 2.2e-03 |
| 20 cells/wavelength | 0 / 0 / 0 / 0 | 8.2e-03 / 7.1e-03 / 2.0e-03 / 4.2e-04 | 1.3e-02 / 4.7e-03 / 2.2e-03 / 4.3e-04 | 1.8e-02 / 8.5e-14 / 2.2e-14 / 2.2e-14 | 1.3e-02 / 4.7e-03 / 2.2e-03 / 4.3e-04 |
| 40 cells/wavelength | 0 / 0 / 0 / 0 | 3.3e-03 / 3.0e-03 / 5.1e-04 / 1.0e-04 | 5.9e-03 / 2.3e-03 / 5.3e-04 / 1.0e-04 | 7.8e-03 / 4.6e-13 / 3.7e-14 / 3.7e-14 | 5.9e-03 / 2.3e-03 / 5.3e-04 / 1.0e-04 |
| 80 cells/wavelength | - | 1.6e-03 / 1.3e-03 / 1.3e-04 / 2.5e-05 | 2.7e-03 / 1.0e-03 / 1.3e-04 / 2.6e-05 | - | - |

| matched aux cell dxa/dx | 0 deg | 15 deg | 30 deg | 45 deg | 60 deg |
| --- | --- | --- | --- | --- | --- |
| 10 cells/wavelength | 1.000000 | 0.935072 | 0.790170 | 0.707107 | 0.790170 |
| 20 cells/wavelength | 1.000000 | 0.935331 | 0.790471 | 0.707107 | 0.790471 |
| 40 cells/wavelength | 1.000000 | 0.935394 | 0.790545 | 0.707107 | 0.790545 |

| 30 deg sensitivity, leak (same four variants) | main (sigma 0.15 f0, courant 0.5, box 1 wavelength) | narrow band (sigma 0.05 f0) | courant 0.9 | box 2 wavelengths |
| --- | --- | --- | --- | --- |
| s, 10 cells/wavelength | 6.1e-02 / 1.6e-02 / 6.4e-03 / 1.9e-03 | 6.2e-02 / 1.7e-02 / 8.3e-03 / 9.3e-04 | 6.0e-02 / 1.6e-02 / 6.4e-03 / 1.9e-03 | 1.4e-01 / 2.4e-02 / 1.1e-02 / 2.3e-03 |
| s, 20 cells/wavelength | 1.7e-02 / 1.0e-02 / 2.0e-03 / 4.0e-04 | 2.0e-02 / 9.6e-03 / 2.5e-03 / 2.9e-04 | 1.8e-02 / 1.1e-02 / 2.0e-03 / 4.0e-04 | 3.1e-02 / 1.0e-02 / 2.6e-03 / 6.6e-04 |
| s, 40 cells/wavelength | 8.1e-03 / 5.1e-03 / 5.1e-04 / 1.0e-04 | 8.1e-03 / 5.1e-03 / 6.4e-04 / 6.7e-05 | 8.1e-03 / 5.2e-03 / 5.1e-04 / 1.0e-04 | - |
| p, 10 cells/wavelength | 5.9e-02 / 1.9e-02 / 7.9e-03 / 2.2e-03 | 5.4e-02 / 2.0e-02 / 8.5e-03 / 9.2e-04 | 5.8e-02 / 1.9e-02 / 7.8e-03 / 2.1e-03 | 1.4e-01 / 2.1e-02 / 1.1e-02 / 2.4e-03 |
| p, 20 cells/wavelength | 1.3e-02 / 4.7e-03 / 2.2e-03 / 4.3e-04 | 1.8e-02 / 5.2e-03 / 2.8e-03 / 2.9e-04 | 1.3e-02 / 4.7e-03 / 2.2e-03 / 4.3e-04 | 2.8e-02 / 5.6e-03 / 2.6e-03 / 6.4e-04 |
| p, 40 cells/wavelength | 5.9e-03 / 2.3e-03 / 5.3e-04 / 1.0e-04 | 5.8e-03 / 2.3e-03 / 6.6e-04 / 6.9e-05 | 5.9e-03 / 2.3e-03 / 5.3e-04 / 1.0e-04 | - |

| off the xz plane (phi = 45 deg), leak (same four variants) | s, 10 | s, 20 | p, 10 | p, 20 | matched dxa/dx |
| --- | --- | --- | --- | --- | --- |
| theta 30 | 5.3e-02 / 1.7e-02 / 7.2e-03 / 1.8e-03 | 2.0e-02 / 8.8e-03 / 2.4e-03 / 4.6e-04 | 4.5e-02 / 1.1e-02 / 7.2e-03 / 1.8e-03 | 1.7e-02 / 4.9e-03 / 2.6e-03 / 5.1e-04 | 0.769912 |
| theta 45 | 5.9e-02 / 8.2e-03 / 3.0e-03 / 7.7e-04 | 2.8e-02 / 4.7e-03 / 1.2e-03 / 2.5e-04 | 6.7e-02 / 5.9e-03 / 3.5e-03 / 9.5e-04 | 2.8e-02 / 3.4e-03 / 1.4e-03 / 2.7e-04 | 0.612203 |
| theta 54.74 (the body diagonal) | 7.0e-02 / 1.5e-14 / 1.5e-14 / 1.5e-14 | 3.2e-02 / 2.8e-14 / 2.7e-14 / 2.7e-14 | 6.8e-02 / 1.4e-14 / 1.4e-14 / 1.4e-14 | 3.3e-02 / 2.9e-14 / 2.6e-14 / 2.6e-14 | 0.577350 |

Controls. At 0 deg every variant at every resolution and both polarisations leaks exactly 0.0,
and the box-centre total field equals the aux incident field exactly (0.0), with the box-centre
peak at 0.9953 / 0.9989 / 0.9997 of the incident peak at 10 / 20 / 40 cells per wavelength (the
node samples the carrier at discrete times) -- so the incident field is really inside the box.
The mutant, the low-z face's corrections with the wrong sign, leaks 2.6 at 0 deg for both
polarisations, against an exact 0.0: the measurement sees a broken surface. The box-centre
error (total minus the aux incident field inside the box) tracks the leak at 1-4x below it (the
log carries it per cell).

Reading. **Off the lattice's symmetry directions, no variant reaches 1e-4 below 40 cells per
wavelength, and the best one converges only at second order.**

1. The plain aux grid -- the one the plan names, same dx and dt -- leaks 3-8e-2 at 10 cells per
   wavelength and 3-10e-3 at 40, and the leak grows with the box: 1.4e-1 with a two-wavelength
   box at 30 deg against 6e-2 with one. That is the aux grid's phase error accumulating across
   the box against the 3-D grid's, which is smaller off an axis.
2. The matched-dispersion fix alone buys 1.1-4x and no more, because linear interpolation of
   the aux field at the projected distance costs ~ (k dxa)^2 / 8 on its own. Cubic interpolation
   takes another 3-26x. The remaining 3-5x comes from the polarisation vectors: the 3-D grid's
   plane wave is transverse to K, not k, and the continuum e and h are wrong at second order.
3. With all three (``+K``) the leak is 2.2e-3 / 4.2e-4 / 1.0e-4 / 2.6e-5 at 10 / 20 / 40 / 80
   cells per wavelength, the same at 15, 30 and 60 deg and no worse off the plane: second
   order, crossing 1e-4 at 40 cells per wavelength. A narrow band (sigma 0.05 f0) halves it at
   10 cells and takes a third off at 40 -- matching at the centre frequency is part of the
   floor, not all of it -- and Courant 0.9 moves nothing.
4. **45 deg in a coordinate plane and the body diagonal are exact.** There the matched cell is
   ``dx/sqrt(2)`` (``dx/sqrt(3)``) at every frequency -- ``2 sin^2(k dx/(2 sqrt 2))`` IS the
   1-D relation with that cell -- every projected node lands on an aux node or a midpoint, and
   K is parallel to k: every matched variant leaks at round-off (1e-14 to 5e-13), for both
   polarisations and any band. The plain aux grid still leaks 7-8e-2 there at 10 cells.

Decision. The rule as written passes: at 45 deg the best variant leaks 2e-14. It passes because
the rule samples the one in-plane angle where the method is exact, not because oblique TFSF
works: at 15, 30 and 60 deg, and at (45, 45), the best variant leaks 2e-3 at 10 and 4e-4 at 20
cells per wavelength, and the plain aux grid the plan names leaks 3-8e-2. **v1 supports
axis-aligned TFSF only and refuses an oblique direction by name.** Supporting the exact
diagonals would take a second aux-grid cell (``dx/sqrt(2)``, ``dx/sqrt(3)``) and an angle
whitelist, for directions a user rarely wants; an arbitrary-angle TFSF needs a different
incident source (a multi-point aux propagator on the 3-D lattice, or the 3-D grid's own
plane-wave solution per frequency), which is out of stage 2. A user who needs oblique incidence
in v1 has the periodic ``PlaneWave`` with ``Bloch``, for periodic scenes only.
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

ANGLES = (0.0, 15.0, 30.0, 45.0, 60.0)
CPWS = (10, 20, 40)
POLS = ('s', 'p')
VARIANTS = ('plain', 'matched', 'matched+cubic', 'matched+cubic+K')
COURANT = 0.5            # wavebit.fdtd._scene.COURANT_DEFAULT
REL_BW = 0.15            # spectral sigma of the Gaussian envelope over the centre frequency
BOX = 1.0                # TFSF box side, in wavelengths of the band centre
MARGIN = 0.25            # scattered-field margin on every side, in wavelengths (>= 4 cells)
ENVELOPE_SIGMAS = 6.0    # the envelope starts and ends at exp(-18)

# Yee offsets in half cells, (x, y, z), per component 0 = x, 1 = y, 2 = z (wavebit.fdtd._legacy).
E_OFF = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
H_OFF = ((0, 1, 1), (1, 0, 1), (1, 1, 0))
# curl component c = sum over (source component, axis, sign) of sign * d_axis F_source;
# axes are array axes of a (z, y, x) block: 0 = z, 1 = y, 2 = x.
CURL = {0: ((2, 1, +1), (1, 0, -1)),     # (curl F)_x = dy Fz - dz Fy
        1: ((0, 0, +1), (2, 2, -1)),     # (curl F)_y = dz Fx - dx Fz
        2: ((1, 2, +1), (0, 1, -1))}     # (curl F)_z = dx Fy - dy Fx


# ---------------------------------------------------------------------------------------
# the stand-in: a vacuum Yee grid, periodic on all six faces, wavebit.fdtd._legacy's update
# ---------------------------------------------------------------------------------------

def _sl(ax, a, b):
    idx = [slice(None)] * 3
    idx[ax] = slice(a, b)
    return tuple(idx)


def _diff(f, ax, forward, out):
    """Periodic one-sided difference of a (z, y, x) block along ``ax``, into ``out``."""
    if forward:     # out[i] = f[i+1] - f[i]
        np.subtract(f[_sl(ax, 1, None)], f[_sl(ax, None, -1)], out=out[_sl(ax, None, -1)])
        np.subtract(f[_sl(ax, 0, 1)], f[_sl(ax, -1, None)], out=out[_sl(ax, -1, None)])
    else:           # out[i] = f[i] - f[i-1]
        np.subtract(f[_sl(ax, 1, None)], f[_sl(ax, None, -1)], out=out[_sl(ax, 1, None)])
        np.subtract(f[_sl(ax, 0, 1)], f[_sl(ax, -1, None)], out=out[_sl(ax, 0, 1)])
    return out


def curl(F, forward, out, tmp):
    """curl F: forward differences for curl E (at the H nodes), backward for curl H."""
    for c, ((s1, a1, _), (s2, a2, _)) in CURL.items():
        _diff(F[s1], a1, forward, out[c])
        out[c] -= _diff(F[s2], a2, forward, tmp)
    return out


def node_coords(n, off):
    """Positions (z, y, x), in cells, of one component's nodes on an n^3 grid."""
    i = np.arange(n, dtype=float)
    z = i[:, None, None] + 0.5 * off[2]
    y = i[None, :, None] + 0.5 * off[1]
    x = i[None, None, :] + 0.5 * off[0]
    return np.broadcast_arrays(z, y, x)


def inside(n, off, lo, hi):
    """Total-field indicator: the node lies in the closed box [lo, hi]^3 (in cells)."""
    z, y, x = node_coords(n, off)
    return ((z >= lo) & (z <= hi) & (y >= lo) & (y <= hi) & (x >= lo) & (x <= hi))


def tfsf_pairs(n, chi_target, chi_source, backward):
    """Every (target node, source node, sign) where the curl crosses the TFSF surface.

    With the stored field ``F = F_scat + chi F_inc``, the curl at a target node must read
    the source field in the target's own region: a total-field target adds
    ``sign * F_inc`` for every scattered-field source in its stencil, a scattered-field
    target subtracts it for every total-field source. Derived from the update itself, so it
    holds for any box whose incident field is an exact discrete solution.
    """
    out = []
    for c, terms in CURL.items():
        for src, ax, sgn in terms:
            shifts = ((+1, +1), (0, -1)) if not backward else ((0, +1), (-1, -1))
            for shift, s2 in shifts:
                chi_s = np.roll(chi_source[src], -shift, axis=ax)   # chi at p + shift
                t_in = chi_target[c]
                plus = t_in & ~chi_s
                minus = ~t_in & chi_s
                for mask, pm in ((plus, +1.0), (minus, -1.0)):
                    idx = np.nonzero(mask)
                    if idx[0].size == 0:
                        continue
                    src_idx = list(idx)
                    src_idx[ax] = (src_idx[ax] + shift) % n
                    out.append((c, idx, src, tuple(src_idx), sgn * s2 * pm))
    return out


def discrete_polarisation(kvec, k, pol):
    """e, h of the 3-D grid's own plane wave at wavenumber k along kvec (dx = 1).

    Both discrete curls act on exp(i k.r) as ``i K x`` with ``K_a = 2 sin(k_a / 2)``, and the
    leapfrog in time as ``-i Omega``, so the grid's plane wave has ``K.E = 0`` and
    ``H = K x E / Omega`` with ``|K| = Omega``: the field vectors are transverse to K, not to
    k. K is parallel to k only on an axis and on the diagonals.
    """
    K = 2 * np.sin(k * kvec / 2)
    K /= np.linalg.norm(K)
    z = np.array([0.0, 0.0, 1.0])
    s = np.cross(z, K)
    s = s / np.linalg.norm(s) if np.linalg.norm(s) > 1e-12 else np.array([0.0, 1.0, 0.0])
    e = s if pol == 's' else np.cross(s, K)
    for v in (e,):
        v[np.abs(v) < 1e-15] = 0.0
    h = np.cross(K, e)
    h[np.abs(h) < 1e-15] = 0.0
    return e, h


def directions(theta_deg, phi_deg, pol):
    """k, e, h unit vectors (x, y, z); h = k x e (H stored as Z0 H, so |h| = |e|).

    theta from +z, phi from +x. s: e = (-sin phi, cos phi, 0), the y axis in the xz plane;
    p: e = s x k, which is (cos theta, 0, -sin theta) in the xz plane.
    """
    t, f = math.radians(theta_deg), math.radians(phi_deg)
    k = np.array([math.sin(t) * math.cos(f), math.sin(t) * math.sin(f), math.cos(t)])
    s = np.array([-math.sin(f), math.cos(f), 0.0])
    e = s if pol == 's' else np.cross(s, k)
    k[np.abs(k) < 1e-15] = 0.0
    e[np.abs(e) < 1e-15] = 0.0
    h = np.cross(k, e)
    h[np.abs(h) < 1e-15] = 0.0
    return k, e, h


# ---------------------------------------------------------------------------------------
# dispersion: the 3-D grid along the direction, and the aux cell that matches it
# ---------------------------------------------------------------------------------------

def bisect(f, a, b, iters=200):
    fa = f(a)
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = f(m)
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b = m
        if b - a <= 1e-15 * max(1.0, abs(m)):
            break
    return 0.5 * (a + b)


def k_3d(omega, dt, kdir):
    """Numerical wavenumber of the 3-D Yee grid (dx = 1) along ``kdir`` at ``omega``."""
    rhs = (math.sin(omega * dt / 2) / dt) ** 2
    return bisect(lambda k: sum(math.sin(k * a / 2) ** 2 for a in kdir) - rhs,
                  1e-9, math.pi / max(abs(a) for a in kdir) * 0.999)


def matched_dxa(omega, dt, kdir):
    """Aux cell size whose 1-D dispersion at ``omega`` gives the 3-D grid's k along kdir."""
    k = k_3d(omega, dt, kdir)
    rhs = math.sin(omega * dt / 2) / dt
    if abs(math.sin(k / 2) - rhs) < 1e-14:
        return 1.0, k                         # on an axis the aux IS the 3-D grid
    dxa = bisect(lambda d: math.sin(k * d / 2) / d - rhs, 0.3, 3.0)
    if abs(math.sin(k * dxa / 2) / dxa - rhs) > 1e-12 * rhs or dt / dxa >= 1.0:
        raise RuntimeError(f'no stable matched aux cell: dxa={dxa}, dt={dt}')
    return dxa, k


# ---------------------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------------------

def interp_matrix(u, n_aux, cubic):
    """Sparse rows interpolating an aux array at fractional indices ``u``."""
    rows, cols, vals = [], [], []
    base = np.floor(u).astype(np.int64)
    f = u - base
    if cubic:
        stencil = (-1, 0, 1, 2)
        w = (-f * (f - 1) * (f - 2) / 6, (f + 1) * (f - 1) * (f - 2) / 2,
             -(f + 1) * f * (f - 2) / 2, (f + 1) * f * (f - 1) / 6)
    else:
        stencil, w = (0, 1), (1 - f, f)
    r = np.arange(u.size)
    for s, ws in zip(stencil, w):
        rows.append(r)
        cols.append(base + s)
        vals.append(ws)
    cols = np.concatenate(cols)
    if cols.min() < 0 or cols.max() >= n_aux:
        raise RuntimeError('aux grid too short for the box')
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), cols)),
                         shape=(u.size, n_aux))


def run_one(cfg):
    cpw, theta, pol, variant = cfg['cpw'], cfg['theta'], cfg['pol'], cfg['variant']
    courant, rel_bw, mutant = cfg.get('courant', COURANT), cfg.get('rel_bw', REL_BW), \
        cfg.get('mutant', False)
    t_start = time.perf_counter()
    lam = float(cpw)                     # dx = 1
    f0 = 1.0 / lam
    omega = 2 * math.pi * f0
    dt = courant / math.sqrt(3.0)
    B = int(round(cfg.get('box', BOX) * lam))
    M = max(4, int(round(MARGIN * lam)))
    n = B + 2 * M
    lo, hi = float(M), float(M + B)
    kvec, evec, hvec = directions(theta, cfg.get('phi', 0.0), pol)
    kdir = (kvec[0], kvec[1], kvec[2])

    if variant == 'plain':
        dxa = 1.0
    else:
        dxa, k3 = matched_dxa(omega, dt, kdir)
    cubic = 'cubic' in variant
    if variant.endswith('+K'):
        evec, hvec = discrete_polarisation(kvec, k3, pol)

    # pulse, window
    tau = 1.0 / (2 * math.pi * rel_bw * f0)
    t0 = ENVELOPE_SIGMAS * tau
    corners = np.array([[x, y, z] for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)])
    xi_box = corners @ kvec
    xi_min, xi_max = xi_box.min() - 1.0, xi_box.max() + 1.0
    xi_src = xi_min - 3.0
    t_end = t0 + ENVELOPE_SIGMAS * tau + 1.15 * (xi_max - xi_src) + 2 * lam
    steps = int(math.ceil(t_end / dt))
    # aux grid: long enough that its PEC ends never return anything to the box in the window
    reach = steps * dt
    m_up = int(math.ceil(reach / dxa)) + 20
    xi0 = xi_src - m_up * dxa
    n_aux = m_up + int(math.ceil((xi_max - xi_src + reach) / dxa)) + 20
    m_src = m_up
    e_a = np.zeros(n_aux)
    h_a = np.zeros(n_aux - 1)
    ca = dt / dxa

    # masks and the correction operators
    chiE = np.stack([inside(n, E_OFF[c], lo, hi) for c in range(3)])
    chiH = np.stack([inside(n, H_OFF[c], lo, hi) for c in range(3)])

    def build(pairs, src_off, vec, half):
        tgt, rows_u, coef = [], [], []
        for c, idx, src, sidx, s in pairs:
            if vec[src] == 0.0:
                continue
            z, y, x = node_coords(n, src_off[src])
            pos_z, pos_y, pos_x = z[sidx], y[sidx], x[sidx]
            xi = pos_x * kvec[0] + pos_y * kvec[1] + pos_z * kvec[2]
            u = (xi - xi0) / dxa - (0.5 if half else 0.0)
            flat = np.ravel_multi_index((np.full(idx[0].size, c),) + tuple(idx), (3, n, n, n))
            sgn = np.full(idx[0].size, s * vec[src])
            if mutant:   # one face's correction with the wrong sign: the low-z face
                zt = node_coords(n, (H_OFF if not half else E_OFF)[c])[0][idx]
                sgn = np.where(zt < lo + 0.75, -sgn, sgn)
            tgt.append(flat)
            rows_u.append(u)
            coef.append(sgn)
        tgt = np.concatenate(tgt)
        u = np.concatenate(rows_u)
        coef = np.concatenate(coef)
        P = sp.diags(coef) @ interp_matrix(u, n_aux if not half else n_aux - 1, cubic)
        # merge duplicate targets
        uniq, inv = np.unique(tgt, return_inverse=True)
        S = sp.csr_matrix((np.ones(tgt.size), (inv, np.arange(tgt.size))),
                          shape=(uniq.size, tgt.size))
        return uniq, (S @ P).tocsr()

    # before_H: curl E at H nodes reads E_inc (aux e, time n)
    tH, AH = build(tfsf_pairs(n, chiH, chiE, backward=False), E_OFF, evec, half=False)
    # before_E: curl H at E nodes reads H_inc (aux h, time n + 1/2)
    tE, AE = build(tfsf_pairs(n, chiE, chiH, backward=True), H_OFF, hvec, half=True)

    # a probe at the box centre, E along the polarisation's largest component
    comp = int(np.argmax(np.abs(evec)))
    zc = yc = xc = int(M + B // 2)
    zp, yp, xp = (a[zc, yc, xc] for a in node_coords(n, E_OFF[comp]))
    u_p = np.array([(xp * kvec[0] + yp * kvec[1] + zp * kvec[2] - xi0) / dxa])
    probe_row = interp_matrix(u_p, n_aux, cubic)

    outE, outH = ~chiE, ~chiH
    # aux indices spanning the box, for the incident peak
    ia = int(math.floor((xi_min - xi0) / dxa)) - 2
    ib = int(math.ceil((xi_max - xi0) / dxa)) + 3

    E = np.zeros((3, n, n, n))
    H = np.zeros((3, n, n, n))
    cF = np.empty((3, n, n, n))
    tmp = np.empty((n, n, n))
    # the leak is read every `stride` steps: >= 20 samples per period of the band centre
    stride = max(1, int(lam / dt / 20))
    b = dt  # dt / dx
    leak = inc_peak = probe_err = box_peak = 0.0
    leak_t = 0
    for step in range(steps):
        cE = curl(E, True, cF, tmp)
        cE.ravel()[tH] += AH @ e_a
        cE *= b
        H -= cE
        h_a -= ca * (e_a[1:] - e_a[:-1])                       # aux H to n + 1/2
        cH = curl(H, False, cF, tmp)
        cH.ravel()[tE] += AE @ h_a
        cH *= b
        E += cH
        e_a[1:-1] -= ca * (h_a[1:] - h_a[:-1])                 # aux E to n + 1
        t = (step + 1) * dt
        e_a[m_src] += math.exp(-0.5 * ((t - t0) / tau) ** 2) * math.sin(omega * (t - t0))
        if step % stride == 0 or step == steps - 1:
            lk = max(np.abs(E[outE]).max(), np.abs(H[outH]).max())
            if lk > leak:
                leak, leak_t = lk, step + 1
        inc_peak = max(inc_peak, np.abs(e_a[ia:ib]).max())
        e_probe = float((probe_row @ e_a)[0]) * evec[comp]
        probe_err = max(probe_err, abs(E[comp, zc, yc, xc] - e_probe))
        box_peak = max(box_peak, abs(E[comp, zc, yc, xc]))
    return dict(cfg, n=n, box=B, margin=M, steps=steps, dt=dt, dxa=dxa,
                leak=leak / inc_peak, leak_step=leak_t, probe_err=probe_err / inc_peak,
                box_peak=box_peak / inc_peak / abs(evec[comp]), inc_peak=inc_peak,
                seconds=time.perf_counter() - t_start)


# ---------------------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------------------

SENS_ANGLE = 30.0
OFF_PLANE = ((30.0, 45.0), (45.0, 45.0), (54.7356103172453, 45.0))   # (theta, phi)


def configs():
    out = []

    def add(arm, cpws, thetas, **kw):
        for cpw in cpws:
            for theta in thetas:
                for pol in POLS:
                    for variant in VARIANTS:
                        out.append(dict(cpw=cpw, theta=theta, pol=pol, variant=variant,
                                        arm=arm, **kw))

    add('main', CPWS, ANGLES)
    for pol in POLS:     # the control that proves the leak can be seen: one face's sign flipped
        out.append(dict(cpw=10, theta=0.0, pol=pol, variant='plain', mutant=True,
                        arm='mutant'))
    # sensitivity at 30 deg (45 deg turns out to be a special angle, see the reading)
    add('narrow', CPWS, (SENS_ANGLE,), rel_bw=0.05)
    add('courant', CPWS, (SENS_ANGLE,), courant=0.9)
    add('box2', (10, 20), (SENS_ANGLE,), box=2.0)
    add('fine80', (80,), (15.0, SENS_ANGLE))
    for theta, phi in OFF_PLANE:
        add('offplane', (10, 20), (theta,), phi=phi)
    # the costly ones first, so the pool drains evenly
    return sorted(out, key=lambda c: -c['cpw'] ** 4 / c.get('rel_bw', REL_BW))


def fmt(x):
    return f'{x:.1e}'


def tables(rows):
    lines = []

    def get(arm, **kw):
        for r in rows:
            if r['arm'] == arm and all(r.get(k, None) == v for k, v in kw.items()):
                return r
        raise KeyError((arm, kw))

    def trio(arm, key='leak', **kw):  # one cell: every variant, in VARIANTS order
        return ' / '.join(fmt(get(arm, variant=v, **kw)[key]) for v in VARIANTS)

    head = ' | '.join(f'{a:g} deg' for a in ANGLES)
    for pol in POLS:
        lines.append(f'| {pol}-pol leak = peak scattered / peak incident; plain / matched / '
                     f'matched+cubic / +K | {head} |')
        lines.append('| --- |' + ' --- |' * len(ANGLES))
        for cpw in CPWS:
            lines.append(f'| {cpw} cells/wavelength | ' + ' | '.join(
                trio('main', cpw=cpw, theta=a, pol=pol) for a in ANGLES) + ' |')
        lines.append('')
    lines.append(f'| matched aux cell dxa/dx | {head} |')
    lines.append('| --- |' + ' --- |' * len(ANGLES))
    for cpw in CPWS:
        lines.append(f'| {cpw} cells/wavelength | ' + ' | '.join(
            f"{get('main', cpw=cpw, theta=a, pol='s', variant='matched')['dxa']:.6f}"
            for a in ANGLES) + ' |')
    lines.append('')
    for pol in POLS:
        lines.append(f'| {pol}-pol box-centre abs(E_total - E_incident) / peak incident; '
                     f'plain / matched / matched+cubic / +K | {head} |')
        lines.append('| --- |' + ' --- |' * len(ANGLES))
        for cpw in CPWS:
            lines.append(f'| {cpw} cells/wavelength | ' + ' | '.join(
                trio('main', key='probe_err', cpw=cpw, theta=a, pol=pol) for a in ANGLES)
                + ' |')
        lines.append('')
    sa = SENS_ANGLE
    lines.append(f'| {sa:g} deg sensitivity, leak plain / matched / matched+cubic / +K | main '
                 f'(band sigma 0.15 f0, courant 0.5, box 1 wavelength) | narrow band '
                 f'(sigma 0.05 f0) | courant 0.9 | box 2 wavelengths |')
    lines.append('| --- | --- | --- | --- | --- |')
    for pol in POLS:
        for cpw in CPWS + (80,):
            cells = []
            for arm in ('main', 'narrow', 'courant', 'box2'):
                src = 'fine80' if (cpw == 80 and arm == 'main') else arm
                try:
                    cells.append(trio(src, cpw=cpw, theta=sa, pol=pol))
                except KeyError:
                    cells.append('-')
            lines.append(f'| {pol}, {cpw} cells/wavelength | ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('| 80 cells/wavelength, leak plain / matched / matched+cubic / +K | s | p |')
    lines.append('| --- | --- | --- |')
    for a in (15.0, sa):
        lines.append(f'| {a:g} deg | ' + ' | '.join(
            trio('fine80', cpw=80, theta=a, pol=pol) for pol in POLS) + ' |')
    lines.append('')
    lines.append('| off the xz plane (phi = 45 deg), leak plain / matched / matched+cubic / +K '
                 '| s, 10 | s, 20 | p, 10 | p, 20 | matched dxa/dx at 10 |')
    lines.append('| --- | --- | --- | --- | --- | --- |')
    for theta, phi in OFF_PLANE:
        cells = [trio('offplane', cpw=c, theta=theta, pol=pol)
                 for pol in POLS for c in (10, 20)]
        dxa = get('offplane', cpw=10, theta=theta, pol='s', variant='matched')['dxa']
        lines.append(f'| theta {theta:.2f}, phi {phi:g} | ' + ' | '.join(cells)
                     + f' | {dxa:.6f} |')
    lines.append('')
    lines.append('| control | leak | box-centre error | box-centre peak / incident |')
    lines.append('| --- | --- | --- | --- |')
    for r in rows:
        if r['arm'] == 'main' and r['theta'] == 0.0:
            lines.append(f"| 0 deg, {r['cpw']} c/wl, {r['pol']}, {r['variant']} | "
                         f"{fmt(r['leak'])} | {fmt(r['probe_err'])} | {r['box_peak']:.4f} |")
    for r in rows:
        if r['arm'] == 'mutant':
            lines.append(f"| MUTANT: low-z face sign flipped, 0 deg, 10 c/wl, {r['pol']} | "
                         f"{fmt(r['leak'])} | {fmt(r['probe_err'])} | {r['box_peak']:.4f} |")
    lines.append('')
    sizes = sorted({(r['cpw'], r['n'], r['box'], r['margin'], r['steps']) for r in rows
                    if r['arm'] in ('main', 'fine80') and r['theta'] == 0.0 or
                    (r['arm'] == 'fine80' and r['theta'] == 15.0)})
    lines.append('Grids (main arm, 0 deg): ' + '; '.join(
        f'{c} c/wl: {nn}^3 cells, box {b}, margin {m}, {st} steps'
        for c, nn, b, m, st in sizes) + '.')
    lines.append(f"Worker time: {sum(r['seconds'] for r in rows):.0f} s over "
                 f"{len(rows)} runs.")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default='artifacts/studies/fdtd_tfsf_oblique_leak.json')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--quick', action='store_true', help='the 0 and 45 deg rows at 10 c/wl')
    args = ap.parse_args()
    cfgs = configs()
    if args.quick:
        cfgs = [c for c in cfgs if c['cpw'] == 10 and c['theta'] in (0.0, 45.0)
                and c['arm'] in ('main', 'mutant')]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(run_one, cfgs):
            rows.append(r)
            print(f"{r['arm']:8s} cpw={r['cpw']:3d} theta={r['theta']:4.0f} pol={r['pol']} "
                  f"{r['variant']:14s} bw={r.get('rel_bw', REL_BW):.2f} "
                  f"C={r.get('courant', COURANT):.2f} leak={r['leak']:.3e} "
                  f"probe_err={r['probe_err']:.3e} box_peak={r['box_peak']:.4f} "
                  f"dxa={r['dxa']:.6f} n={r['n']} steps={r['steps']} {r['seconds']:.1f}s",
                  flush=True)
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            with open(args.out, 'w') as fh:
                json.dump(rows, fh, indent=1, default=float)
    if not args.quick:
        print()
        print(tables(rows))


if __name__ == '__main__':
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(var, '1')
    sys.exit(main())
