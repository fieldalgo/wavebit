"""Freeze Meep's crossed-grating answer into ``golden_meep_crossed.npz``.

Run once, by hand, in an environment that has Meep; ``test_meep_crossed_golden.py`` then
runs anywhere. Rerun it only to add cases, never to make a failing test pass — the whole
value of this oracle is that its numbers did not come from this codebase.

    conda run -n meep python tests/references/gen_goldens_meep_crossed.py

**LICENCE FOOTING, and it is why this file exists rather than a dependency.** Meep is
GPL-2.0-or-later and wavebit is Apache-2.0. No file under the meep package is opened,
read, grepped or copied here: Meep is used only by *calling* its public Python API and
reading the numbers that come back, so every statement this file makes about Meep is a
black-box measurement. Nothing in the shipped package imports it, exactly as the
RETICOLO generator needs Octave and the fmmax ones need the ``fwref`` env. ``sys.tracebacklimit``
is set to 0 below so that an uncaught exception cannot print lines of another package's
source into the terminal.

**Why a fourth oracle, and why this one.** RETICOLO, S4 and fmmax are all RCWA: they
share a formulation with this project's frequency-domain path and cannot arbitrate the
*time-domain* one, whose errors are numerical dispersion, a staircased boundary and an
imperfect absorber rather than a truncated Fourier series. Meep is an FDTD. It is also
the only oracle here that reaches a **crossed grating solved in the time domain**, which
until now had no external reference of any kind: ``_extract_crossed`` is a line-for-line
copy of ``_extract_conical`` whose per-order signs, s/p frame and ``n / kz_m`` secant are
all invisible to ``R``, ``T`` and ``R + T``.

**Every convention below is derived, none is fitted.** The chain, in the order it was
measured (the working is in ``docs/fdtd.md`` §2 and ``docs/amplitude-sp-frame.md``):

* the **axis map is the identity** — a 3D meep cell has three real axes, so unlike the
  2D bridge nothing is permuted; wavebit's ``+z`` (down, into the stack) is meep's
  ``+z``, and wavebit's ``z = 0`` sits at meep's ``z = 0``;
* the **time convention needs no conjugation**: meep's DFT phasor is ``exp(+i k z)`` for
  a wave travelling ``+z``, which is wavebit's ``e^{-iwt}``;
* the **s/p frame** is ``s_hat = z_hat x k_hat_par`` and ``p_hat = k_hat x s_hat``, and
  ``E . k_hat = 0`` collapses the p projection to ``-+ (n / kz)(c Ex + s Ey)`` with the
  upper sign for a downgoing wave. So the three leg signs are geometry, not convention;
* the **origin** carries a ``(-1)**m (-1)**l``. wavebit places the raster from the cell
  centre and demodulates from the cell corner (its kernel runs over cells ``0 .. n-1``),
  while this harness does both from the centre. Normalised by the incident order the two
  differ by ``exp(-i (k_m - k_inc) P / 2)`` on each axis, which is exactly that sign.

Five harness faults, each of which returned a plausible energy-conserving wrong number
first. They are recorded because each one is still live for anyone extending this file:

1. **meep replicates geometry periodically along the PML axis too.** A substrate block
   overhanging the ``+z`` cell edge wraps into the superstrate. Measured in 2D at
   ``T = 0.412`` against a true 0.684, with ``R + T`` exactly 1. The block below is sized
   to the cell edge and no further.
2. **DFT monitor samples are weighted, not uniform.** A plane spanning one period comes
   back overhanging it by half a cell at each end, with the edge samples down-weighted so
   the weights sum to the period. A plain mean put 9.9e-3 of a plane wave into orders a
   plane wave does not have.
3. **An oblique source sheet must carry the Bloch phase itself** (``amp_func``), or the
   incident field is not one plane wave: measured leak 2.99 without it.
4. **The z pixel COUNT decides where meep anchors its grid**, and an odd count puts a
   grid point exactly *on* the structure's interfaces instead of between them. Measured
   on a uniform slab: resolutions 40 and 80 (122 and 244 pixels) give 4.5e-3 and 1.2e-3
   on ``r``, and 60 (183 pixels) gives **4.3e-2** — an O(dz) shift of the effective
   interface that shows in ``arg(r)`` while ``|r|`` and ``R + T`` stay on the
   second-order curve. ``_z_layout`` keeps the count even.
5. **A layer thickness that is not an integer number of pixels** gets rounded, and that
   moves the plane the transmitted phase is referenced to. It is refused by name in
   ``_run`` below, with the measurement beside the refusal. This one survived a
   *convergence study* as well, because both tiers rounded to the same value — so it is
   also the reason the committed golden is checked against a reference rather than only
   against itself.

**The instrument that caught two of those five is the empty run**: decompose the
incident-only field into orders and demand that every order a plane wave does not have
comes back at round-off. It is the cheapest check here and it is asserted, not printed.
"""
import pathlib
import sys
import time

sys.tracebacklimit = 0                     # never print frames from another package
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / 'golden_meep_crossed.npz'

#: The coarse control runs at half the resolution. It is not a second golden: it is what
#: lets the committed file state its own error bar. At order q the two tiers differ by
#: (2**q - 1) times the fine grid's error -- so 3x at the second order this would have
#: liked, and 1x at first. **Measured on these scenes q is 0.69 to 1.63**, not 2, so the
#: estimate below is printed both ways and the test reads the honest end. A crossed
#: raster is all corners, and neither this solver nor wavebit's own reaches second order
#: on one (``docs/fdtd.md`` sec.8g measures 1.83 and 1.50 for wavebit, with a control
#: that stalled at 0.67).
COARSE_RATIO = 2


# ---------------------------------------------------------------------------------------
# the meep half -- the only part that needs the meep environment
# ---------------------------------------------------------------------------------------

def _z_layout(d, resolution, dpml, pad, gap):
    """Cell extent and the three z planes, all snapped to the grid.

    The monitors sit ``gap`` from the structure and no further. Below a periodic
    structure the field is exactly a sum of Rayleigh orders at any depth and the spatial
    transform separates them exactly, so an evanescent order on the monitor plane cannot
    contaminate a propagating one — the same finding ``docs/fdtd.md`` §2 records for
    wavebit's own ``pad_scale``. What the distance *does* buy is de-referencing error:
    the phase is undone with the analytic ``kz`` while meep propagates on the numerical
    one. Measured on a uniform slab at resolution 60, moving the transmission monitor
    from 0.43 um to 0.10 um took ``|dt|`` from 1.5e-2 to 1.0e-2.
    """
    snap = lambda v: round(v * resolution) / resolution
    dpml, pad, gap, d = snap(dpml), snap(pad), snap(gap), snap(d)
    z_min, z_max = -(dpml + pad), d + pad + dpml
    if round((z_max - z_min) * resolution) % 2:      # fault 4
        z_max += 1.0 / resolution
    return dict(z_min=z_min, z_max=z_max, d=d,
                z_src=z_min + dpml + snap(0.30 * pad), z_ref=-gap, z_trn=d + gap)


def _run(case, leg, resolution, empty, gap=0.10, df_frac=0.2, material_at=None):
    """One meep run. Returns the two monitor planes and the metadata they need."""
    import meep as mp

    dpml, pad = case['meep_dpml'], case['meep_pad']
    wl = case['wavelength']
    px, py = case['period']
    fcen = 1.0 / wl
    for p in (px, py):
        if abs(p * resolution - round(p * resolution)) > 1e-9:
            raise SystemExit(f'period {p} is not an integer number of pixels at '
                             f'resolution {resolution}; the cell would be rounded and '
                             f'the lattice would not be the one the case declares')
    # A layer thickness that is not an integer number of pixels is REFUSED, and it is
    # refused rather than rounded because rounding it is a *phase* error that every energy
    # check passes. Measured: a 0.24 um layer at resolution 60 is 14.4 pixels; snapping it
    # to 14 moved the transmission reference plane by 6.7 nm and put the specular
    # amplitude 4.6e-2 from the converged RCWA answer, while R + T stayed at 0.9954 and
    # the per-order efficiencies stayed inside 1.6e-2. It was invisible to the two-tier
    # error bar as well, since resolution 30 snaps 7.2 pixels to the same 14/60.
    for t in case['thickness']:
        if abs(t * resolution - round(t * resolution)) > 1e-9:
            raise SystemExit(
                f'layer thickness {t} is {t * resolution:g} pixels at resolution '
                f'{resolution}. Give a thickness the grid can hold exactly, or the '
                f'transmitted phase is referenced to a plane the structure does not end on.')
    lay = _z_layout(float(sum(case['thickness'])), resolution, dpml, pad, gap)
    th, ph = np.deg2rad(case['theta_deg']), np.deg2rad(case['phi_deg'])
    n_top, n_bot = case['n_top'], case['n_bot']

    kpt = mp.Vector3(fcen * n_top * np.sin(th) * np.cos(ph),
                     fcen * n_top * np.sin(th) * np.sin(ph), 0)
    kx, ky = 2 * np.pi * kpt.x, 2 * np.pi * kpt.y
    amp = (None if kx == 0.0 and ky == 0.0
           else (lambda p, _a=kx, _b=ky: np.exp(1j * (_a * p.x + _b * p.y))))

    # An ELECTRIC sheet along s_hat drives only the s wave -- the p wave's in-plane E
    # lies along (cos varphi, sin varphi), which is orthogonal to it. A MAGNETIC sheet
    # along s_hat drives only the p wave, whose H is -+ s_hat. Both currents are purely
    # transverse, so neither needs a z-directed sheet, and the empty run's own s/p
    # decomposition is what checks that this worked (`purity` below).
    sx, sy = -np.sin(ph), np.cos(ph)
    comps = (mp.Ex, mp.Ey) if leg == 's' else (mp.Hx, mp.Hy)
    # `is_integrated=True`: a current sheet whose time profile has a nonzero integral
    # leaves a static field behind, and with a Bloch phase that field cannot radiate away.
    src = [mp.Source(mp.GaussianSource(fcen, fwidth=df_frac * fcen, is_integrated=True),
                     component=c, center=mp.Vector3(0, 0, lay['z_src']),
                     size=mp.Vector3(px, py, 0), amp_func=amp, amplitude=a)
           for c, a in zip(comps, (sx, sy)) if a != 0.0]

    geom = []
    if not empty:
        z0 = 0.0
        for layer, (arr, t) in enumerate(zip(np.asarray(case['index'], dtype=float),
                                             case['thickness'])):
            ny, nx = arr.shape
            dx, dy = px / nx, py / ny
            for j in range(ny):
                for i in range(nx):
                    geom.append(mp.Block(
                        size=mp.Vector3(dx, dy, t),
                        # The raster is placed from the CELL CENTRE, which is where
                        # `wavebit.meshgrid_xy` puts column i as well.
                        center=mp.Vector3((i + 0.5) * dx - 0.5 * px,
                                          (j + 0.5) * dy - 0.5 * py, z0 + 0.5 * t),
                        material=(mp.Medium(index=float(arr[j, i])) if material_at is None
                                  else material_at(layer, j, i))))
            z0 += t
        # Sized to the +z cell edge and NO further -- fault 1.
        geom.append(mp.Block(size=mp.Vector3(mp.inf, mp.inf, lay['z_max'] - z0),
                             center=mp.Vector3(0, 0, 0.5 * (z0 + lay['z_max'])),
                             material=mp.Medium(index=n_bot)))

    sim = mp.Simulation(cell_size=mp.Vector3(px, py, lay['z_max'] - lay['z_min']),
                        resolution=resolution,
                        boundary_layers=[mp.PML(dpml, direction=mp.Z)],
                        geometry=geom, sources=src, k_point=kpt,
                        default_material=mp.Medium(index=n_top),
                        geometry_center=mp.Vector3(0, 0, 0.5 * (lay['z_min'] + lay['z_max'])),
                        force_complex_fields=True)
    sim.init_sim()
    mons = {k: sim.add_dft_fields([mp.Ex, mp.Ey], [fcen], center=mp.Vector3(0, 0, lay[f'z_{k}']),
                                  size=mp.Vector3(px, py, 0))
            for k in ('ref', 'trn')}
    # A FIXED run length, not `stop_when_fields_decayed`: the cell holds ONE transverse
    # wavevector, so content below the light line is trapped and never reaches a PML. The
    # running DFT suppresses it because it sits far from the extraction frequency; the
    # check that says so is that doubling the length does not move the answer (measured
    # identical to every digit printed at until = 200 / 400 / 800).
    sim.run(until_after_sources=case['meep_until'])

    out = {}
    for k, obj in mons.items():
        md = sim.get_array_metadata(dft_cell=obj)
        out[f'{k}__x'] = np.asarray(md[0], dtype=float).ravel()
        out[f'{k}__y'] = np.asarray(md[1], dtype=float).ravel()
        out[f'{k}__w'] = np.asarray(md[3], dtype=float)          # fault 2
        out[f'{k}__Ex'] = np.asarray(sim.get_dft_array(obj, mp.Ex, 0))
        out[f'{k}__Ey'] = np.asarray(sim.get_dft_array(obj, mp.Ey, 0))
    out.update(lay)
    out['meep_version'] = str(mp.__version__)
    return out


# ---------------------------------------------------------------------------------------
# the numpy half -- the conventions, all of them derived above
# ---------------------------------------------------------------------------------------

def _kz(n, kx, ky):
    """``kz / k0`` with ``Re >= 0``: a downgoing propagating wave, or decay toward +z."""
    v = np.sqrt(np.asarray(complex(n) ** 2 - kx ** 2 - ky ** 2, dtype=complex))
    return np.where(np.real(v) < 0, -v, v)


def _frame(kx, ky, phi):
    """``(cos varphi, sin varphi)`` -- the DIRECTION of ``(kx, ky)``, not a line.

    At the origin, where the in-plane wavevector carries no direction at all, the answer
    is ``phi``: a wave going straight down still needs a reference direction in the xy
    plane for its polarisation, and ``phi`` is by convention what supplies it.
    """
    kpar = np.hypot(np.real(kx), np.real(ky))
    safe = np.where(kpar > 0, kpar, 1.0)
    return (np.where(kpar > 0, np.real(kx) / safe, np.cos(phi)),
            np.where(kpar > 0, np.real(ky) / safe, np.sin(phi)))


def _coeff(field, x, y, w, px, py, kx, ky, k0):
    """One order's coefficient on one monitor plane, on meep's own abscissae and weights."""
    fx = np.exp(-1j * k0 * kx * np.asarray(x, dtype=float))
    fy = np.exp(-1j * k0 * ky * np.asarray(y, dtype=float))
    return np.sum(np.asarray(w) * np.asarray(field) * fx[:, None] * fy[None, :]) / (px * py)


def _amplitudes(case, leg, struct, empty):
    """Per-order ``(r_s, r_p, t_s, t_p)`` and ``(R, T)`` on the ``(l, m)`` grid."""
    wl, (px, py) = case['wavelength'], case['period']
    n_top, n_bot = case['n_top'], case['n_bot']
    th, ph = np.deg2rad(case['theta_deg']), np.deg2rad(case['phi_deg'])
    k0 = 2.0 * np.pi / wl
    m = np.asarray(case['orders_u'], dtype=int)
    l = np.asarray(case['orders_v'], dtype=int)
    kxi = n_top * np.sin(th) * np.cos(ph)
    kyi = n_top * np.sin(th) * np.sin(ph)
    KX, KY = np.broadcast_arrays(kxi + m[None, :] * wl / px, kyi + l[:, None] * wl / py)
    c_m, s_m = _frame(KX, KY, ph)
    c_i, s_i = _frame(kxi, kyi, ph)
    kz_top, kz_bot = _kz(n_top, KX, KY), _kz(n_bot, KX, KY)
    kz_inc = _kz(n_top, kxi, kyi)
    z_ref, z_trn, d = struct['z_ref'], struct['z_trn'], struct['d']

    def take(run, tag, comp, sub=None):
        f = np.asarray(run[f'{tag}__{comp}'])
        if sub is not None:
            f = f - np.asarray(sub[f'{tag}__{comp}'])
        return np.array([[_coeff(f, run[f'{tag}__x'], run[f'{tag}__y'], run[f'{tag}__w'],
                                 px, py, KX[j, i], KY[j, i], k0)
                          for i in range(KX.shape[1])] for j in range(KX.shape[0])])

    inc = take(empty, 'ref', 'Ex'), take(empty, 'ref', 'Ey')
    j0 = int(np.flatnonzero(l == 0)[0])
    i0 = int(np.flatnonzero(m == 0)[0])
    ex_i = inc[0][j0, i0] * np.exp(-1j * k0 * kz_inc * z_ref)
    ey_i = inc[1][j0, i0] * np.exp(-1j * k0 * kz_inc * z_ref)
    a_s_inc = -s_i * ex_i + c_i * ey_i
    a_p_inc = -(n_top / kz_inc) * (c_i * ex_i + s_i * ey_i)
    a_inc = a_s_inc if leg == 's' else a_p_inc

    # THE INSTRUMENT. A plane wave has one order; anything the empty run leaves in the
    # others is the harness talking. This is what caught the unweighted quadrature and
    # the constant-amplitude oblique sheet.
    others = np.abs(inc[0]) + np.abs(inc[1])
    others[j0, i0] = 0.0
    leak = float(others.max() / (abs(inc[0][j0, i0]) + abs(inc[1][j0, i0])))
    purity = float(abs(a_p_inc / a_s_inc) if leg == 's' else abs(a_s_inc / a_p_inc))

    ref_x = take(struct, 'ref', 'Ex', empty) * np.exp(1j * k0 * kz_top * z_ref)
    ref_y = take(struct, 'ref', 'Ey', empty) * np.exp(1j * k0 * kz_top * z_ref)
    trn_x = take(struct, 'trn', 'Ex') * np.exp(-1j * k0 * kz_bot * (z_trn - d))
    trn_y = take(struct, 'trn', 'Ey') * np.exp(-1j * k0 * kz_bot * (z_trn - d))

    origin = ((-1.0) ** m)[None, :] * ((-1.0) ** l)[:, None]
    sig_r, sig_t = 1.0, -1.0                       # the canonical frame; see the header
    r_s = origin * (-s_m * ref_x + c_m * ref_y) / a_inc
    r_p = origin * sig_r * (n_top / kz_top) * (c_m * ref_x + s_m * ref_y) / a_inc
    t_s = origin * (-s_m * trn_x + c_m * trn_y) / a_inc
    t_p = origin * sig_t * (n_bot / kz_bot) * (c_m * trn_x + s_m * trn_y) / a_inc

    prop_r = np.abs(np.imag(kz_top)) < 1e-9
    prop_t = np.abs(np.imag(kz_bot)) < 1e-9
    R = (np.abs(r_s) ** 2 + np.abs(r_p) ** 2) * np.real(kz_top) / np.real(kz_inc) * prop_r
    T = (np.abs(t_s) ** 2 + np.abs(t_p) ** 2) * np.real(kz_bot) / np.real(kz_inc) * prop_t
    return dict(r_s=r_s, r_p=r_p, t_s=t_s, t_p=t_p, R=R, T=T), leak, purity


def solve(case, leg, resolution):
    """One scene, one incident polarisation: the structure run and the empty control."""
    empty = _run(case, leg, resolution, empty=True)
    struct = _run(case, leg, resolution, empty=False)
    return _amplitudes(case, leg, struct, empty)


def main():
    from _meep_crossed_cases import CASES

    out, t_start = {}, time.time()
    for name, case in CASES.items():
        fine = int(case['meep_resolution'])
        coarse = fine // COARSE_RATIO
        out[f'{name}__orders_u'] = np.asarray(case['orders_u'], dtype=int)
        out[f'{name}__orders_v'] = np.asarray(case['orders_v'], dtype=int)
        print(f'\n{name}: {case["period"]} um cell, resolutions {fine} and {coarse}')
        for tier, res in (('fine', fine), ('coarse', coarse)):
            for leg in ('s', 'p'):
                t0 = time.time()
                g, leak, purity = solve(case, leg, res)
                wall = time.time() - t0
                for key, val in g.items():
                    out[f'{name}__{leg}__{tier}__{key}'] = val
                out[f'{name}__{leg}__{tier}__leak'] = np.array(leak)
                out[f'{name}__{leg}__{tier}__purity'] = np.array(purity)
                total = float(g['R'].sum() + g['T'].sum())
                # Nothing here absorbs, so R + T = 1 up to the Yee grid's O(dz**2). A
                # gross failure of it means the bridge is wired wrong, and freezing that
                # would be worse than having no oracle: refuse instead.
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
        # The committed file states its own error bar. Printed as a RANGE, because the
        # observed order is between 1 and 2 rather than at 2 -- the second-order reading
        # is the optimistic end and the first-order one is the honest end.
        for leg in ('s', 'p'):
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

    np.savez(OUT, **out)
    print(f'\nwrote {OUT} ({len(out)} arrays) in {time.time() - t_start:.0f} s')


if __name__ == '__main__':
    main()
