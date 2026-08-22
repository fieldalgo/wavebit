"""
benchmark_truncation_shape.py -- rectangular vs circular Fourier-order truncation.


Wavebit keeps a full rectangle of Fourier orders, |m| <= half_width_1 and |n| <= half_width_2, so the
harmonic count is always (2*half_width_1+1)*(2*half_width_2+1).  S4 instead keeps a disc in reciprocal
space: the orders with the smallest |G|, with the last shell closed so lattice symmetry is
preserved.  At the same maximum order the disc holds about pi/4 of the harmonics of the box,
and RCWA cost scales as N^3, so the disc is ~2x cheaper -- if the discarded corner orders were
not doing any work.

This script measures that.  For each structure it solves the same problem with both truncation
sets and reports the error against a converged reference as a function of the harmonic count.
Both shapes go through the same permittivity raster, the same normal field and the same solver
primitives, so the only difference is which orders are kept.

Factorization modes:

  'nvf'      the production normal vector field path.  Every input to ``_nvf_assemble`` --
             ``<eps>``, ``<1/eps>``, ``<nx^2>``, ``<nx ny>``, ``<ny^2>`` -- is a plain direct
             rule matrix, i.e. a G-difference lookup, so it can be built *natively* on any
             order set.  The circular case here is a real circular NVF, not a restriction of a
             rectangular one.  This is the mode the verdict rests on.

  'laurent'  plain Laurent rule (no inverse rule), also an exact G-difference build.  A cheap
             control: at low index contrast it is essentially the exact factorization, so its
             error is essentially pure truncation error.

  'li'       the nested Li inverse rule.  Rectangular sets only -- the nested 1D-then-2D
             inversion needs a complete rectangle, and restricting the matrices built on the
             enclosing box leaks corner-order information a true circular Li could not have.
             Kept for the rectangular convergence curve, not run circularly.

The result: at equal accuracy the disc costs 1.20x on a round pillar, 0.78x on an axis-aligned
square and 1.01x on one rotated 30 deg -- structure-dependent and within +/-20%, so the ~2x the
harmonic count promises never arrives.  Li, JOSA A 14(10), 2758 (1997) introduced circular
truncation and concluded the parallelogramic one is superior; S4 and GRCWA default to circular
regardless.  Numbers, method and the S4/GRCWA order-selection analysis are in
``docs/fourier-truncation-shape.md``.
"""

import time

import numpy as np

from wavebit import eps_from_index
from wavebit._core.conventions import boundary_kz, order_azimuth
from wavebit._core.factorization import (_nvf_assemble, fourier2d_piecewise_constant,
                                           fourier2d_pointwise, normal_vector_field,
                                           to_conv_mat_raster_nvf)
from wavebit._core.solve_1d import _block, _diag, final_matching
from wavebit._core.solve_2d import layer_modes_2d, sp_modes_2d

C128 = np.complex128


# ---------------------------------------------------------------- order sets

def order_grid(half_width_1, half_width_2):
    """Flat (m, n) order lists in the solver's layout: flat index is iy * n_orders_1 + ix."""
    m = np.arange(-half_width_1, half_width_1 + 1)
    n = np.arange(-half_width_2, half_width_2 + 1)
    mm, nn = np.meshgrid(m, n)  # (n_orders_2, n_orders_1)

    return mm.flatten(), nn.flatten()


def circular_select(mm, nn, period, radius):
    """S4-style cut: keep |G| <= radius, ties included so the shell stays closed.

    radius is in units of 1/period[0], i.e. radius == max order along x.  The mask is
    centrosymmetric and always keeps (0, 0), so the zero order stays at flat index N // 2 --
    which is what lets :func:`final_matching` place the incident delta without being told.
    """
    g2 = mm ** 2 + (nn * period[0] / period[1]) ** 2

    return g2 <= radius ** 2 + 1e-9


# ------------------------------------------------------- convolution matrices

def _half_pixel_phase(k, n):
    """Cell-centre offset alone, matching ``factorization._half_pixel_phase``.

    *Both* kinds of grid carry this. Index ``j`` names the pixel spanning ``[j/n, (j+1)/n)``
    and the value attached to it sits at the centre ``(j+0.5)/n``, while a bare DFT places
    sample ``j`` at the left edge; the difference is ``exp(-i*pi*k/n)`` whether or not the
    value is held across the pixel.  Only a staircase additionally carries the ``sinc``.

    This script omitted it on the pointwise branch, which is the one the normal field goes
    through -- so the field sat half a pixel off the permittivity it multiplies.  The
    production path was fixed in 2026-08 and this copy was not, which is what the first
    selftest below now catches.
    """
    return np.exp(-1j * np.pi * k / n)


def _pixel_correction(k, n):
    """Exact-for-constant-pixels factor, matching ``factorization._pixel_correction``."""
    t = k / n

    return np.sinc(t) * _half_pixel_phase(k, n)


def toeplitz_conv(f, mm, nn, piecewise_constant):
    """Direct-rule convolution matrix on an arbitrary order set: ``E[i, j] = f_hat[G_i - G_j]``.

    The order-index sign follows ``factorization._circulant_index`` (``r - c``); the selftest
    below pins it against the ``fourier2d_*`` builders.
    """
    ny, nx = f.shape
    coef = np.fft.fft2(np.asarray(f, dtype=C128) / f.size)

    dn = nn[:, None] - nn[None, :]
    dm = mm[:, None] - mm[None, :]
    E = coef[dn % ny, dm % nx]

    if piecewise_constant:
        E = E * _pixel_correction(dn, ny) * _pixel_correction(dm, nx)
    else:
        E = E * _half_pixel_phase(dn, ny) * _half_pixel_phase(dm, nx)

    return E


def conv_mats(eps, normal_field, mm, nn, keep, max_order_box, mode):
    """``(epx, epy, epz_i, epxy, epyx)`` for the selected order set."""
    m, n = mm[keep], nn[keep]

    if mode == 'nvf':
        nx_f, ny_f = normal_field

        return _nvf_assemble(
            np,
            toeplitz_conv(eps, m, n, True),
            toeplitz_conv(1.0 / eps, m, n, True),
            toeplitz_conv(nx_f ** 2, m, n, False),
            toeplitz_conv(nx_f * ny_f, m, n, False),
            toeplitz_conv(ny_f ** 2, m, n, False),
        )

    if mode == 'laurent':
        E = toeplitz_conv(eps, m, n, True)
        O = np.zeros_like(E)

        return E, E, np.linalg.inv(E), O, O

    if mode == 'li':
        if not keep.all():
            raise ValueError('the nested Li rule has no circular form; see the module docstring')
        half_width_1, half_width_2 = max_order_box
        epz = fourier2d_piecewise_constant(np, eps, half_width_1, half_width_2, inverse_rule=None)
        epy = fourier2d_piecewise_constant(np, eps, half_width_1, half_width_2, inverse_rule='y')
        epx = fourier2d_piecewise_constant(np, eps, half_width_1, half_width_2, inverse_rule='x')
        O = np.zeros_like(epz)

        return epx, epy, np.linalg.inv(epz), O, O

    raise ValueError(mode)


# ------------------------------------------------------------------- solver

def amplitude_pair(psi):
    """The incident ``(a_s, a_p)`` at unit flux for a linear polarization angle ``psi``.

    Both ``final_matching`` and ``solve_2d`` take the resolved pair, not the angle -- the
    engine resolves it, and this script calls them underneath the engine.  ``psi = v`` is
    ``(sin v, cos v)``, so ``psi = 0`` is TM and ``pi/2`` is TE.
    """
    return (np.sin(psi), np.cos(psi))


def solve_flat(mm, nn, epx, epy, epz_i, epxy, epyx, wavelength, period, thickness,
               n_top, n_bot, theta, phi, psi):
    """2D RCWA (TMM) for an arbitrary flat set of Fourier orders.

    Same chain as ``_core.solve_2d.solve_2d`` -- and reusing its ``layer_modes_2d`` /
    ``sp_modes_2d`` / ``final_matching`` verbatim -- with the (n_orders_2, n_orders_1) tensor-product
    layout replaced by an explicit order list so non-rectangular sets work.  Single layer.

    ``n_top`` / ``n_bot`` are refractive indices, and stay so: this reimplements the solver
    *core*, which is the half of the codebase the permittivity flip deliberately did not
    touch.  ``top`` / ``bot`` take a material at the public front door and the setter derives
    the index once; below that line every reader still wants ``n``, because at ~16 of the
    sites the quantity genuinely is an amplitude ratio.  The one conversion this file does
    make is the raster's, in :func:`make_cell`.
    """
    xp = np
    n_orders = len(mm)
    k0 = 2 * np.pi / wavelength

    kx = np.asarray(n_top * np.sin(theta) * np.cos(phi)
                    + mm * (wavelength / period[0]), dtype=C128)
    ky = np.asarray(n_top * np.sin(theta) * np.sin(phi)
                    + nn * (wavelength / period[1]), dtype=C128)

    k_par_sq = kx ** 2 + ky ** 2
    kz_top = boundary_kz(xp, n_top ** 2, k_par_sq)
    kz_bot = boundary_kz(xp, n_bot ** 2, k_par_sq)
    # `order_azimuth` rather than a bare `atan(ky/kx)`: a ratio has already discarded
    # `sign(kx)`, so the frame of every `Re(kx) < 0` order is reversed.  That cancels out of R
    # and T -- it is a diagonal +-1 similarity -- but it is NaN at `kx == 0`, and this script
    # exists to compare order sets that differ in exactly which orders they contain.
    varphi = order_azimuth(xp, ky, kx, phi)

    I = np.astype(np.eye(n_orders), C128)
    O = np.astype(np.zeros((n_orders, n_orders)), C128)

    Kz_bot = _diag(xp, kz_bot)
    big_F = _block(xp, [[I, O], [O, -1j * Kz_bot / n_bot ** 2]])
    big_G = _block(xp, [[-1j * Kz_bot, O], [O, I]])
    big_T = np.astype(np.eye(2 * n_orders), C128)
    big_I = np.astype(np.eye(2 * n_orders), C128)

    Kx = _diag(xp, kx)
    Ky = _diag(xp, ky)
    F_c = _diag(xp, np.cos(varphi))
    F_s = _diag(xp, np.sin(varphi))

    W, V, q = layer_modes_2d(xp, Kx, Ky, I, epx, epy, epz_i, epxy, epyx, C128)

    X_1 = _diag(xp, np.exp(-k0 * q[:n_orders] * thickness))
    X_2 = _diag(xp, np.exp(-k0 * q[n_orders:] * thickness))
    big_X = _block(xp, [[X_1, O], [O, X_2]])

    big_W, big_V = sp_modes_2d(xp, W, V, F_c, F_s, n_orders)
    big_W_i = np.linalg.inv(big_W)
    big_V_i = np.linalg.inv(big_V)
    big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)
    big_A_i = np.linalg.inv(0.5 * (big_W_i @ big_F + big_V_i @ big_G))

    Y = big_X @ big_B @ big_A_i @ big_X
    big_F = big_W @ (big_I + Y)
    big_G = big_V @ (big_I - Y)
    big_T = big_T @ big_A_i @ big_X

    # n_orders_1 = n_orders, n_orders_2 = 1: the flat set *is* the layout. The incident delta lands at
    # n_orders // 2, which is the zero order for both shapes (see circular_select).
    R, T = final_matching(xp, C128, big_F, big_G, big_T, kz_top, kz_bot,
                          amplitude_pair(psi), theta,
                          n_top, n_bot, n_orders, 1, answers='efficiency')

    return R.flatten(), T.flatten(), kz_top, kz_bot


# ---------------------------------------------------------------- structures

def make_cell(kind, ngrid=512, n_ridge=3.48, n_bg=1.0, fill=0.4, rotation=0.):
    """Square-lattice pillar with a fixed fill factor, as a **permittivity** raster.

    The two materials are named by refractive index because that is how the structures are
    labelled in the tables below and in the write-up, and :func:`wavebit.eps_from_index` is
    the conversion.  Written out rather than folded into the literals so the quantity is
    visible at the point the datasheet number arrives: a raster is what
    ``to_conv_mat_raster_nvf`` and ``normal_vector_field`` take, and both read permittivity
    now -- the ``n ** 2`` that used to sit inside them is gone, so an index left here is a
    legal, energy-conserving, different pillar.
    """
    t = (np.arange(ngrid) + 0.5) / ngrid
    yy, xx = np.meshgrid(t, t, indexing='ij')

    dx, dy = xx - 0.5, yy - 0.5
    if rotation:
        c, s = np.cos(rotation), np.sin(rotation)
        dx, dy = c * dx + s * dy, -s * dx + c * dy

    if kind == 'circle':
        r = np.sqrt(fill / np.pi)
        mask = dx ** 2 + dy ** 2 <= r ** 2
    elif kind == 'square':
        a = np.sqrt(fill) / 2
        mask = (np.abs(dx) <= a) & (np.abs(dy) <= a)
    else:
        raise ValueError(kind)

    return np.where(mask, eps_from_index(n_ridge), eps_from_index(n_bg))


# ------------------------------------------------------------------ driver

# `n_top` / `n_bot` are refractive indices: they are the arguments of the solver-core entry
# points this script calls directly (`solve_flat` below, and `_core.solve_2d.solve_2d`), which
# the permittivity flip left alone.  The *materials* -- the raster -- are permittivity.
CASE = dict(
    wavelength=900.,
    period=(1200., 1200.),
    thickness=300.,
    n_top=1.,
    n_bot=1.,
    theta=20 / 180 * np.pi,
    phi=30 / 180 * np.pi,
    psi=0.,  # TM
)


def run_one(eps, normal_field, mm, nn, keep, max_order_box, mode):
    mats = conv_mats(eps, normal_field, mm, nn, keep, max_order_box, mode)

    t0 = time.perf_counter()
    R, T, kz_top, kz_bot = solve_flat(
        mm[keep], nn[keep], *mats,
        CASE['wavelength'], CASE['period'], CASE['thickness'],
        CASE['n_top'], CASE['n_bot'], CASE['theta'], CASE['phi'], CASE['psi'])
    dt = time.perf_counter() - t0

    return dict(m=mm[keep], n=nn[keep], R=R, T=T,
                kz_top=kz_top, kz_bot=kz_bot, n_harmonics=int(keep.sum()), time=dt)


def compare(res, ref):
    """Error of `res` against reference `ref`, on the orders they share."""
    ref_key = {(m, n): i for i, (m, n) in enumerate(zip(ref['m'], ref['n']))}
    idx_res, idx_ref = [], []
    for i, (m, n) in enumerate(zip(res['m'], res['n'])):
        j = ref_key.get((m, n))
        if j is not None:
            idx_res.append(i)
            idx_ref.append(j)
    idx_res = np.array(idx_res)
    idx_ref = np.array(idx_ref)

    prop = (ref['kz_top'][idx_ref].real > 1e-10) | (ref['kz_bot'][idx_ref].real > 1e-10)

    d_ri = np.abs(res['R'][idx_res] - ref['R'][idx_ref])
    d_ti = np.abs(res['T'][idx_res] - ref['T'][idx_ref])

    return dict(
        err_R_total=abs(res['R'].sum() - ref['R'].sum()),
        err_T_total=abs(res['T'].sum() - ref['T'].sum()),
        err_order_max=max(d_ri[prop].max(), d_ti[prop].max()) if prop.any() else np.nan,
    )


def equivalent_rect_N(rect_rows, err):
    """Harmonic count a rectangular set needs to reach `err`, log-interpolated."""
    pts = sorted((row[2]['n_harmonics'], row[3]['err_order_max']) for row in rect_rows)
    for (n0, e0), (n1, e1) in zip(pts, pts[1:]):
        if e0 >= err >= e1 and e0 > 0 and e1 > 0:
            w = np.log(e0 / err) / np.log(e0 / e1)
            return n0 + w * (n1 - n0)

    return np.nan


# ------------------------------------------------------------------ selftest

def selftest():
    """Pin the G-difference build and the flat solver against the production code."""
    eps = make_cell('circle', ngrid=64).astype(C128)
    truncation_order = 3
    mm, nn = order_grid(truncation_order, truncation_order)

    for flag in (False, True):
        direct = toeplitz_conv(eps, mm, nn, flag)
        nested = (fourier2d_piecewise_constant(np, eps, truncation_order, truncation_order, inverse_rule=None) if flag
                  else fourier2d_pointwise(np, eps, truncation_order, truncation_order))
        err = np.abs(direct - nested).max()
        print(f'selftest conv (piecewise_constant={flag}): {err:.3e}')
        assert err < 1e-12, err

    # flat solver vs the production NVF path, on a rectangular set
    structure = eps[None, :, :]
    epx, epy, epz_i, epxy, epyx = to_conv_mat_raster_nvf(structure, truncation_order, truncation_order)
    de_ri_ref, de_ti_ref = _core_solve_2d(truncation_order, epx, epy, epz_i, epxy, epyx)

    nf = normal_vector_field(eps)
    r = run_one(eps, nf, mm, nn, np.ones(mm.shape, bool), (truncation_order, truncation_order), 'nvf')
    err = max(np.abs(r['R'] - de_ri_ref.flatten()).max(),
              np.abs(r['T'] - de_ti_ref.flatten()).max())
    print(f'selftest solver (flat vs _core.solve_2d): {err:.3e}')
    assert err < 1e-12, err


def _core_solve_2d(truncation_order, epx, epy, epz_i, epxy, epyx):
    from wavebit._core.solve_2d import solve_2d

    C = CASE
    rng = np.arange(-truncation_order, truncation_order + 1)
    kx = C['n_top'] * np.sin(C['theta']) * np.cos(C['phi']) + rng * (C['wavelength'] / C['period'][0])
    ky = C['n_top'] * np.sin(C['theta']) * np.sin(C['phi']) + rng * (C['wavelength'] / C['period'][1])

    return solve_2d(C['wavelength'], kx, ky, C['n_top'], C['n_bot'], C['theta'],
                    amplitude_pair(C['psi']), [C['thickness']],
                    epx, epy, epz_i, epxy, epyx, phi=C['phi'])


# -------------------------------------------------------------------- sweep

def sweep(eps, normal_field, mode, max_order_ref, max_orders, radii, label):
    print(f'\n===== {label}  factorization={mode} '
          f'  (reference: rectangular truncation_order={max_order_ref}) =====')

    mm_r, nn_r = order_grid(max_order_ref, max_order_ref)
    ref = run_one(eps, normal_field, mm_r, nn_r, np.ones(mm_r.shape, dtype=bool),
                  (max_order_ref, max_order_ref), mode)
    print(f'reference: N={ref["n_harmonics"]}  '
          f'R={ref["R"].sum():.6f}  T={ref["T"].sum():.6f}  {ref["time"]:.1f}s')

    for back in (2, 4):
        mm_c, nn_c = order_grid(max_order_ref - back, max_order_ref - back)
        chk = run_one(eps, normal_field, mm_c, nn_c, np.ones(mm_c.shape, dtype=bool),
                      (max_order_ref - back, max_order_ref - back), mode)
        print(f'reference drift (truncation_order={max_order_ref - back} vs {max_order_ref}): '
              f'{compare(chk, ref)["err_order_max"]:.2e}')

    rect_rows, circ_rows = [], []
    for truncation_order in max_orders:
        mm, nn = order_grid(truncation_order, truncation_order)
        r = run_one(eps, normal_field, mm, nn, np.ones(mm.shape, bool), (truncation_order, truncation_order), mode)
        rect_rows.append(('rect', truncation_order, r, compare(r, ref)))

    if mode != 'li':
        for radius in radii:
            box = int(np.ceil(radius))
            mm, nn = order_grid(box, box)
            keep = circular_select(mm, nn, CASE['period'], radius)
            r = run_one(eps, normal_field, mm, nn, keep, (box, box), mode)
            circ_rows.append(('circ', radius, r, compare(r, ref)))

    print(f'{"shape":>6} {"maxord":>7} {"N":>6} {"time[s]":>8} '
          f'{"err_order":>11} {"err_R":>11} {"err_T":>11}')
    for shape, order, r, e in sorted(rect_rows + circ_rows,
                                     key=lambda row: row[2]['n_harmonics']):
        print(f'{shape:>6} {order:>7} {r["n_harmonics"]:>6} {r["time"]:>8.2f} '
              f'{e["err_order_max"]:>11.3e} {e["err_R_total"]:>11.3e} '
              f'{e["err_T_total"]:>11.3e}')

    if circ_rows:
        print(f'\n{"circ maxord":>12} {"N_circ":>7} {"err":>11} {"N_rect same err":>16} '
              f'{"N ratio":>9} {"cost ratio":>11}')
        for _, radius, r, e in circ_rows:
            n_eq = equivalent_rect_N(rect_rows, e['err_order_max'])
            ratio = r['n_harmonics'] / n_eq if n_eq == n_eq else np.nan
            print(f'{radius:>12} {r["n_harmonics"]:>7} {e["err_order_max"]:>11.3e} '
                  f'{n_eq:>16.0f} {ratio:>9.3f} {ratio ** 3:>11.3f}')

    return dict(mode=mode, label=label, rect=rect_rows, circ=circ_rows)


def main(max_order_ref=20, max_orders=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
         radii=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
         structures=(('circle pillar, n=3.48', dict(kind='circle')),
                     ('square pillar, n=3.48', dict(kind='square')),
                     ('square pillar rot 30deg, n=3.48',
                      dict(kind='square', rotation=np.pi / 6))),
         modes=('nvf', 'laurent'), plot=None):
    """Sweep both truncation shapes over `structures`; pass `plot=<path>` for the error curves."""

    selftest()

    blocks = []
    for label, kwargs in structures:
        eps = make_cell(**kwargs).astype(C128)
        normal_field = normal_vector_field(eps)
        for mode in modes:
            blocks.append(sweep(eps, normal_field, mode, max_order_ref, max_orders, radii, label))

    if plot:
        make_plot(blocks, plot)


def make_plot(blocks, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ncol = len({b['mode'] for b in blocks})
    nrow = int(np.ceil(len(blocks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow), squeeze=False)

    for ax, b in zip(axes.flatten(), blocks):
        for key, marker, name in (('rect', 'o', 'rectangular'), ('circ', 's', 'circular')):
            if not b[key]:
                continue
            n = [row[2]['n_harmonics'] for row in b[key]]
            e = [row[3]['err_order_max'] for row in b[key]]
            ax.loglog(n, e, marker=marker, label=name)
        ax.set_title(f'{b["label"]}\n{b["mode"]}', fontsize=9)
        ax.set_xlabel('harmonics N')
        ax.set_ylabel('max per-order DE error')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f'\nsaved {path}')


if __name__ == '__main__':
    main()
