"""Measure tilted-interface smoothing, retaining complex modes and the actual spectrum.

Run from the repository root. The JSON records every grid, signed error, eigenpair
residual and spectral endpoint; the printed table includes order and fit scatter.
Homogeneous controls measure dispersion in a different medium: they are comparisons,
not rigorous error floors, and are never subtracted from the interface errors.
See docs/validation/fdtd/fd-subpixel-occupancy.md for the measurements and their limits.

The operator and transfer-matrix construction follow the existing S1 study. This
version removes its fixed image window, uses a general eigensolver without discarding
imaginary parts, and measures the algebraic spectral minimum instead of modes near zero.
No Meep source is used.
"""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Dimensionless scene inputs: a half-filled unit-period stack; kt*p=0.6 lies below
# pi/sqrt(17), the narrowest Bloch zone among the four rational tilts.
PERIOD, DUTY, KT = 1.0, 0.5, 0.6
TILTS = ((1, 0), (4, 1), (2, 1), (1, 1))
CONTRASTS = (1.2, 2.25, 12.0)
CELLS = (16, 32, 48)
SPOTS = {'ex': (1, 0), 'ey': (0, 1), 'ez': (0, 0), 'corner': (0, 0)}


def square_half_plane(t, normal, h):
    """Exact square fraction below a plane, with no subtraction of distant squares."""
    a, b = sorted((abs(normal[0]) * h, abs(normal[1]) * h), reverse=True)
    if b == 0:
        return np.minimum(np.maximum((t + a / 2) / a, 0), 1)
    q = np.maximum((a + b) / 2 - np.abs(t), 0)
    small = np.where(q < b, q * q / (2 * a * b), (q - b / 2) / a)
    return np.where(t < 0, small, 1 - small)


def lamellar_fill(xs, ys, h, normal):
    """Periodic layer fractions, folded before clipping; no fixed image window or clamp."""
    d = np.remainder(normal[0] * xs + normal[1] * ys, PERIOD)
    reach = h * (abs(normal[0]) + abs(normal[1])) / 2
    first = math.floor(float(np.min(d - reach)) / PERIOD)
    last = math.floor(float(np.max(d + reach)) / PERIOD)
    out = np.zeros_like(d)
    for k in range(first, last + 1):
        lo = k * PERIOD - d
        out += (square_half_plane(lo + DUTY * PERIOD, normal, h)
                - square_half_plane(lo, normal, h))
    return out


def transfer(omega, kt, eps, thickness, polarization):
    """The 2x2 transfer matrix of one layer, for ``(E_t, H_z)`` (p) or ``(E_z, H_t)`` (s).

    From Maxwell with ``exp(-i omega t)``, ``c = mu = 1`` and a field ``~ exp(i k_t t)``:
    ``d/dn (u, v) = i [[0, a], [b, 0]] (u, v)``, with ``a = omega - k_t^2/(omega eps)`` and
    ``b = omega eps`` for p polarisation (H along z, E in the plane -- the case that has a
    field component ACROSS the layers), and ``a = omega``, ``b = omega eps - k_t^2/omega`` for
    s (E along z). Both give ``k_n^2 = a b = omega^2 eps - k_t^2``.
    """
    if polarization == 'p':
        a, b = omega - kt * kt / (omega * eps), omega * eps
    else:
        a, b = omega, omega * eps - kt * kt / omega
    kn = np.sqrt(complex(a * b))
    c = np.cos(kn * thickness)
    s = thickness * np.sinc(kn * thickness / np.pi)
    return np.array([[c, 1j * a * s], [1j * b * s, c]], complex)


def band_residual(omega, kt, eps_hi, eps_lo, polarization, kn_phase=0.0):
    m = (transfer(omega, kt, eps_lo, PERIOD * (1.0 - DUTY), polarization)
         @ transfer(omega, kt, eps_hi, PERIOD * DUTY, polarization))
    return float(np.real(np.trace(m)) / 2.0 - math.cos(kn_phase))


@lru_cache(maxsize=None)
def exact_omega(kt, eps_hi, eps_lo, polarization, steps=200000):
    """The lowest ``omega`` with ``tr(M)/2 = 1`` (Bloch phase zero across the period).

    The continuum answer every grid arm is judged against: the wave runs ALONG the layers, so
    ``k_n = 0``, and for p polarisation E has a component across them -- which is the branch the
    subpixel rule is about. Bracketed inside the physical window ``k_t / n_max < omega < k_t``
    (the mode's index lies between the two media's) and bisected.
    """
    lo, hi = kt / (math.sqrt(eps_hi) * 1.001), kt * 1.001

    def f(w):
        return band_residual(w, kt, eps_hi, eps_lo, polarization)

    prev, w0 = f(lo), lo
    for i in range(1, steps + 1):
        w = lo + (hi - lo) * i / steps
        cur = f(w)
        if prev * cur <= 0.0:
            a, b = w0, w
            for _ in range(200):
                mid = 0.5 * (a + b)
                if f(a) * f(mid) <= 0.0:
                    b = mid
                else:
                    a = mid
            return 0.5 * (a + b)
        prev, w0 = cur, w
    raise RuntimeError(f'no root in [{lo}, {hi}] for eps {eps_hi} {polarization}')


def kottke(f, normal, eps_hi, eps_lo):
    """``(K_xx, K_yy, K_xy)`` of the Farjadpour/Kottke inverse-permittivity tensor.

    ``eps~^-1 = P <eps^-1> + (I - P) <eps>^-1`` with ``P = <n n^T>``, which for a FLAT interface
    is exactly ``n n^T`` with no averaging at all -- which is why this study isolates the rule
    and never the moment. Its diagonal is the rule stage 4 ships.

    Farjadpour et al., Opt. Lett. **31**, 2972 (2006); Kottke, Farjadpour & Johnson, Phys. Rev.
    E **77**, 036611 (2008); Oskooi et al., Comput. Phys. Commun. **181**, 687 (2010). The
    tensor form is quoted from Meep's documentation; no Meep source is read here.
    """
    harmonic = f / eps_hi + (1.0 - f) / eps_lo
    inv_arith = 1.0 / (f * eps_hi + (1.0 - f) * eps_lo)
    nx, ny = normal
    return (nx * nx * harmonic + (1.0 - nx * nx) * inv_arith,
            ny * ny * harmonic + (1.0 - ny * ny) * inv_arith,
            nx * ny * (harmonic - inv_arith))


def _shifted(m, phase, axis, forward):
    """``u -> u`` shifted one cell, with the Bloch phase on the wrapped edge."""
    n = m * m
    idx = np.arange(n).reshape((m, m))
    src = np.roll(idx, -1 if forward else +1, axis=axis)
    edge = np.zeros((m, m), dtype=bool)
    if axis == 0:
        edge[-1 if forward else 0, :] = True
    else:
        edge[:, -1 if forward else 0] = True
    data = np.where(edge.ravel(), phase if forward else np.conj(phase), 1.0 + 0.0j)
    return sp.coo_matrix((data, (np.arange(n), src.ravel())), shape=(n, n)).tocsr()


def back_diff(m, phase, axis, h):
    """``(u[i] - u[i-1]) / h``: the backward difference, centres to nodes along ``axis``."""
    return (sp.identity(m * m, format='csr', dtype=complex)
            - _shifted(m, phase, axis, forward=False)) / h


def back_avg(m, phase, axis):
    """``(u[i] + u[i-1]) / 2``, the half-cell average that lands on the same node."""
    return 0.5 * (sp.identity(m * m, format='csr', dtype=complex)
                  + _shifted(m, phase, axis, forward=False))


def fwd_avg(m, phase, axis):
    return 0.5 * (sp.identity(m * m, format='csr', dtype=complex)
                  + _shifted(m, phase, axis, forward=True))


def corner_smoothed(kxy):
    """``K_xy`` at each corner replaced by the mean of the four CELL values around it.

    The design's second remedy for the corner branch -- "average the corner over its
    neighbours" -- written as the one average this file already has. A cell value is the
    four-corner average (the ``[1,2,1]/4`` half-cell average of ``back_avg``/``fwd_avg``, per
    axis); averaging those four again is the same stencil applied twice, i.e.
    ``[1,2,1]/4`` along each axis. It is symmetric and its weights sum to 1, so for a smooth
    ``K`` it moves the value by ``O(h^2)`` and cannot cost an order -- the homogeneous control
    is run on it to check exactly that. ``K_xy`` is periodic on the square Bloch cell by
    construction (``lamellar_fill`` sums the periodic images), so a plain roll is the right
    wrap and no Bloch phase enters.
    """
    out = kxy
    for axis in (0, 1):
        out = (0.25 * np.roll(out, 1, axis) + 0.5 * out + 0.25 * np.roll(out, -1, axis))
    return out


def te_operator(m, h, phases, kxx, kyy, kxy, *, cross='self-adjoint'):
    """The 2-D TE operator (H out of plane): ``A H = omega^2 H`` on the Yee grid.

    ``omega^2 H = -d_x(K_yy d_x H) - d_y(K_xx d_y H) + d_x(K_xy d_y H) + d_y(K_xy d_x H)``,
    i.e. ``A = G^H M G`` with ``G = (D_x, D_y)`` and ``M = R K R^T``, the tensor turned a
    quarter turn -- which is why ``D_x`` meets ``K_yy`` and not ``K_xx``. ``D_x H`` lands on the
    E_y points and ``D_y H`` on the E_x points, so each diagonal term reads the tensor entry at
    the point of the component it multiplies: the per-component sampling stage 4 pays for.

    The cross term has no such point: the two gradients live on different half planes.

    * ``cross='self-adjoint'`` averages both to the shared cell CORNER and writes
      ``-(P_x^H K_xy P_y + P_y^H K_xy P_x)`` with one ``K_xy``, read at that corner. Hermitian
      by construction, for any ``K``.
    * ``cross='naive'`` is the writing the stage-4 design measured at 2.3% non-Hermitian and
      asked to retry: each half applies the ``K_xy`` of ITS OWN component's point to the
      interpolated neighbour. The two interpolations ARE adjoint to each other (the natural
      four-point average and its transpose), so the only thing that breaks Hermiticity is that
      the two ``K_xy`` are read at two different points -- which is exactly what per-component
      sampling does.
    * ``cross='corner-averaged'`` is the self-adjoint writing with ``K_xy`` passed through
      ``corner_smoothed`` first -- the design's own second remedy for the 45-degree corner
      branch ("average the corner over its neighbours"), implemented so that the remedy can be
      MEASURED rather than recommended. Still Hermitian by construction: it is one ``K_xy`` per
      corner, only a different one.
    * ``cross='none'`` is the diagonal rule: the off-diagonal simply dropped.
    """
    px, py = phases
    dx_ = back_diff(m, px, 1, h)                 # centres -> E_y points (x node, y half)
    dy_ = back_diff(m, py, 0, h)                 # centres -> E_x points (x half, y node)
    a = dx_.conj().T @ sp.diags(kyy['ey'].ravel()) @ dx_
    a = a + dy_.conj().T @ sp.diags(kxx['ex'].ravel()) @ dy_
    if cross in ('self-adjoint', 'corner-averaged'):
        px_ = back_avg(m, py, 0) @ dx_           # E_y points -> corner
        py_ = back_avg(m, px, 1) @ dy_           # E_x points -> corner
        k = kxy['corner'] if cross == 'self-adjoint' else corner_smoothed(kxy['corner'])
        g = sp.diags(k.ravel())
        a = a - (px_.conj().T @ g @ py_ + py_.conj().T @ g @ px_)
    elif cross == 'naive':
        to_ey = fwd_avg(m, py, 0) @ back_avg(m, px, 1)      # E_x points -> E_y points
        a = a - dx_.conj().T @ sp.diags(kxy['ey'].ravel()) @ to_ey @ dy_
        a = a - dy_.conj().T @ sp.diags(kxy['ex'].ravel()) @ to_ey.conj().T @ dx_
    elif cross != 'none':
        raise ValueError(cross)
    return a.tocsc()


def tm_operator(m, h, phases, eps_z):
    """The 2-D TM pair (E out of plane): ``A E = omega^2 B E`` with ``B = diag(eps_z)``."""
    px, py = phases
    dx_ = back_diff(m, px, 1, h)
    dy_ = back_diff(m, py, 0, h)
    return ((dx_.conj().T @ dx_ + dy_.conj().T @ dy_).tocsc(),
            sp.diags(eps_z.ravel()).tocsc())


def hermiticity(a):
    n = spla.norm(a)
    return float(spla.norm(a - a.conj().T) / n) if n else 0.0


def build(rule, tilt, contrast, cells, kt):
    mm, nn = tilt
    s = math.hypot(mm, nn)
    normal = (mm / s, nn / s)
    side = PERIOD * s
    m = int(round(cells * s))
    h = side / m
    kx = -kt / PERIOD * normal[1]
    ky = kt / PERIOD * normal[0]
    phases = (np.exp(1j * kx * side), np.exp(1j * ky * side))
    node = np.arange(m) * h - side / 2.0
    half = node + h / 2.0

    kxx, kyy, kxy, epsz = {}, {}, {}, {}
    for name, (ox, oy) in SPOTS.items():
        xs = np.broadcast_to((half if ox else node)[None, :], (m, m))
        ys = np.broadcast_to((half if oy else node)[:, None], (m, m))
        if rule == 'staircase':
            f = (((normal[0] * xs + normal[1] * ys) % PERIOD) < DUTY * PERIOD).astype(float)
        else:
            f = lamellar_fill(xs, ys, h, normal)
        arith = f * contrast + (1.0 - f)
        if rule in ('staircase', 'arithmetic'):
            kxx[name] = kyy[name] = 1.0 / arith
            kxy[name] = np.zeros_like(f)
        else:
            a, b, c = kottke(f, normal, contrast, 1.0)
            kxx[name], kyy[name] = a, b
            kxy[name] = c if rule == 'tensor' else np.zeros_like(f)
        epsz[name] = arith
    return dict(m=m, h=h, phases=phases, kxx=kxx, kyy=kyy, kxy=kxy, epsz=epsz,
                cells_actual=PERIOD / h)


def eigenpair(a, b, guess):
    """The mode nearest guess, retaining complex eigenvalues and its equation residual.

    Three modes provide the selected value and two neighbours for a gap check. Use the
    general solver even for the Hermitian cases so the diagnostic cannot erase imaginary
    parts from the deliberately non-Hermitian control. The fixed initial vector makes
    repeated measurements comparable; no acceptance tolerance is imposed by this study.
    """
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(a.shape[0]) + 1j * rng.standard_normal(a.shape[0])
    vals, vecs = spla.eigs(a, k=3, M=b, sigma=guess, which='LM', v0=v0)
    i = int(np.argmin(np.abs(vals - guess)))
    value, v = vals[i], vecs[:, i]
    av, bv = a @ v, b @ v
    residual = float(np.linalg.norm(av - value * bv)
                     / (np.linalg.norm(av) + abs(value) * np.linalg.norm(bv)))
    gap = float(min(abs(value - w) for j, w in enumerate(vals) if j != i))
    return value, gap, residual


def spectral_bounds(a):
    """Algebraic endpoints of a Hermitian matrix, not the values nearest zero.

    The previous sigma=0/LM query missed large negative modes; its LM denominator
    could also be negative. For diag(-100, 1, ..., 100), that routine reported -0.01
    instead of -1 for the minimum/maximum ratio. SA and LA ask the actual questions.
    """
    v0 = np.random.default_rng(0).standard_normal(a.shape[0])
    lo = float(spla.eigsh(a, k=1, which='SA', v0=v0, return_eigenvectors=False)[0])
    hi = float(spla.eigsh(a, k=1, which='LA', v0=v0, return_eigenvectors=False)[0])
    return lo, hi


def measured_case(rule, tilt, contrast, cells, polarization='TE', cross='self-adjoint',
                  homogeneous=False, kt=KT):
    case = build(rule, tilt, contrast, cells, kt)
    m, h = case['m'], case['h']
    if homogeneous:
        normal = np.asarray(tilt) / math.hypot(*tilt)
        kxx, kyy, kxy = kottke(DUTY, normal, contrast, 1.)
        for key, value in (('kxx', kxx), ('kyy', kyy), ('kxy', kxy)):
            case[key] = {spot: np.full((m, m), value) for spot in SPOTS}
        kx, ky = -kt * normal[1] / PERIOD, kt * normal[0] / PERIOD
        exact = math.sqrt(kyy * kx * kx + kxx * ky * ky - 2 * kxy * kx * ky)
    else:
        exact = exact_omega(kt / PERIOD, contrast, 1., 'p' if polarization == 'TE' else 's')
    if polarization == 'TE':
        a = te_operator(m, h, case['phases'], case['kxx'], case['kyy'], case['kxy'],
                        cross=cross)
        b = sp.identity(m * m, format='csc', dtype=complex)
    else:
        a, b = tm_operator(m, h, case['phases'], case['epsz']['ez'])
    value, gap, residual = eigenpair(a, b, exact ** 2)
    omega = np.sqrt(complex(value))
    out = dict(cells=cells, cells_actual=PERIOD / h, grid=m, exact=exact,
               error=float(omega.real / exact - 1), imag_relative=float(omega.imag / exact),
               eigen_residual=residual, gap_relative=gap / abs(value),
               hermiticity=hermiticity(a))
    if polarization == 'TE' and rule == 'tensor' and not homogeneous and cross != 'naive':
        lo, hi = spectral_bounds(a)
        out.update(min_eigenvalue=lo, max_eigenvalue=hi, min_over_max=lo / hi)
    return out


def fit(rows):
    """A descriptive log fit with scatter; neither three points nor their fit proves order."""
    errors = np.asarray([r['error'] for r in rows])
    if not np.all(np.isfinite(errors)) or np.any(errors == 0):
        return dict(order=None, scatter=None, monotone=False)
    x = np.log([r['cells_actual'] for r in rows])
    y = np.log(np.abs(errors))
    line = np.polyfit(x, y, 1)
    return dict(order=float(-line[0]), scatter=float(np.expm1(np.max(abs(y - np.polyval(line, x))))),
                monotone=bool(np.all(np.diff(np.abs(errors)) < 0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='JSON output; choose a scratch path for a probe')
    parser.add_argument('--cells', default=','.join(map(str, CELLS)))
    args = parser.parse_args()
    cells = tuple(int(v) for v in args.cells.split(','))
    results = {}
    start = time.perf_counter()
    for tilt in TILTS:
        for contrast in CONTRASTS:
            arms = [('control', 'tensor', 'TE', 'self-adjoint', True),
                    ('control-naive', 'tensor', 'TE', 'naive', True),
                    ('tm', 'arithmetic', 'TM', 'none', False)]
            arms += [(rule, rule, 'TE', 'self-adjoint', False)
                     for rule in ('staircase', 'arithmetic', 'diagonal', 'tensor')]
            arms += [(cross, 'tensor', 'TE', cross, False)
                     for cross in ('corner-averaged', 'naive')]
            for name, rule, pol, cross, uniform in arms:
                rows = [measured_case(rule, tilt, contrast, c, pol, cross, uniform) for c in cells]
                key = f'{tilt[0]}:{tilt[1]}|{contrast:g}|{name}'
                results[key] = dict(rows=rows, **fit(rows))
                print(key, 'errors', '/'.join(f"{r['error']:+.4e}" for r in rows),
                      'fit', fit(rows), 'imag', max(abs(r['imag_relative']) for r in rows),
                      'residual', max(r['eigen_residual'] for r in rows),
                      'min/max', min((r['min_over_max'] for r in rows if 'min_over_max' in r),
                                     default=None), flush=True)
    args.output.write_text(json.dumps(dict(cells=cells, results=results,
                                          seconds=time.perf_counter() - start), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
