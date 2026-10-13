"""Where does a gain half-space's plane-wave answer stop describing a beam? What does c = 2 add?

A half-space takes Maystre's root, ``Re(kz) + Im(kz) > 0`` (the "coupure de Maystre" of J. P.
Hugonin and P. Lalanne, RETICOLO software for grating analysis, Institut d'Optique 2005,
arXiv:2101.00901). For a gain half-space at real ``kx`` this is the root of switching the gain
on from zero at fixed frequency. ``GainCutoffWarning`` fires per order where
``|Re kz**2| < c |Im kz**2|``, with c = 2 and ``kz**2 = eps*mu - k_par**2`` in units of k0**2.
Here ``rho = |Re kz**2| / |Im kz**2|`` is called the ratio.

The question is where a finite beam's response depends on its width. For a Gaussian beam that
is ``|kx0 - Re sqrt(eps)| < |Im sqrt(eps)|``, a ratio of about 1: there the beam integral's
contour, pushed under the branch point ``sqrt(eps)``, weighs that point more as the beam
widens. Outside it the beam converges to Maystre's plane-wave value as W grows. c = 2 adds the
zone 1 < rho < 2, where beams a few gain lengths wide have not converged yet. This study makes
those numbers reproducible.

  1 flips    A record of where the fixed-kx causal root flips. It is not the reason for c. A
             real gain medium is dispersive. Its causal root at fixed real kx is the decaying
             root at large Im(w), continued down to the real axis (J. Skaar, Phys. Rev. E 73,
             026605, 2006; J. O. Grepstad and J. Skaar, Opt. Express 19, 21404, 2011). That root
             steps at an angle the line shape sets. Single Lorentz gain lines (J. D. Jackson,
             Classical Electrodynamics, 3rd ed., Wiley 1999, sec. 7.5, with the strength's sign
             flipped): eps(w) = eps_inf + f wc^2 / (wc^2 - w^2 - i gam w), pinned to
             eps(1) = 1 - i g. A line is plausible if it is stable (eps_inf > 0, no zero of eps
             in the upper half w plane) and its peak |Im eps| on the real axis is at most 1. The
             coarse grid is wc in {0.8 .. 1.2}, gam in {0.02, 0.1, 0.3}, g in {0.005 .. 0.2},
             n_inc in {1.5, 3.5}; a finer grid follows. Three things keep the flips from setting
             c: a finite beam does not see them (section 4), how many a band holds depends on
             the grid, and the flip is itself a convention, since at the flip's kx the branch
             point of kz(w) sits in the upper half w plane.
  2 beam     A Gaussian beam from glass (n 1.5) onto the centred line (wc 1, gam 0.1, g 0.05),
             s polarization. Its field at the interface is exp(-x^2 / W^2) times the carrier, W
             in units of 1/k0. The steady response at real w follows Briggs and Bers (R. J.
             Briggs, Electron-Stream Interaction with Plasmas, MIT Press 1964; A. Bers,
             "Space-time evolution of plasma instabilities -- absolute and convective", in
             Handbook of Plasma Physics vol. 1, eds. A. A. Galeev and R. N. Sudan, North-Holland
             1983, p. 451). As Im(w) falls to 0, the real kx contour is pushed ahead of any
             branch point that crosses it. kx_b(w) = w sqrt(eps(w)) crosses the real axis and
             ends below it, so the contour dips under kx_b(1). The reflected field at the beam
             centre is a real-axis integral plus a finger round that branch point, computed with
             the finger along the trajectory (A) and straight down (B). The deformation's
             direction is checked by carrying a lowered contour in w from Im(w) above the
             crossing, where it must match the undeformed real-axis integral, down to Im(w) = 0.
             Angles 42 to 45 deg, W 100 .. 800. The field at the beam centre is compared with
             the same beam under Maystre's kernel and with the plane wave's r. The mirror point
             -kx_b(1) and the incidence branch point at kx = 1.5 sit where the beam spectrum is
             below 1e-40, and are left out. The incidence branch point is common to both kernels,
             so it never enters their difference.
  3 margin   At a fixed ratio past the c = 2 edge, how the beam's distance from Maystre's beam
             falls with W, for g in {0.005, 0.05, 0.2}, in units of the gain length; and how wide
             a beam must be at the edge itself.
  4 far flip The two lines whose fixed-kx flip lies furthest out (on each grid), with beams
             between Maystre's step and that flip: the beam goes to Maystre's value, not to the
             fixed-kx causal one.
  5 top hat  A compact-support profile at 44.5 and 50 deg, mirror point included: the window of
             ratio ~1 is a property of the Gaussian.
  6 firing   How many of the 900 angles 0.0 .. 89.9 deg fall in the c = 2 band, for glass onto
             eps = 1 - i g, g in {0.005, 0.05, 0.2}? Counted from the formula, and from the
             solver's GainCutoffWarning with one zeroth-order solve per angle.

Units: w0 = 1 and the speed of light is 1, so kx is kx/k0 at the operating frequency and
kz^2(w) = w^2 eps(w) - kx^2. Time convention exp(-i w t): Im(eps) < 0 is gain, and the Lorentz
poles sit in the lower half plane. Sections 2 and 6 run wavebit (NumPy backend) for Maystre's
plane-wave value and for the warning; the rest is NumPy only. Run from the repository
root: ``python studies/rcwa/gain_cutoff_band/gain_cutoff_band.py``. It prints the log committed as
``studies/rcwa/gain_cutoff_band/records/gain_cutoff_band.log``, in about ten seconds.
"""

from wavebit.compat import legacy as _legacy_wavebit
import textwrap
import warnings

import numpy as np

from wavebit.compat import legacy as wavebit

WCS = (0.8, 0.9, 1.0, 1.1, 1.2)
GAMS = (0.02, 0.1, 0.3)
GS = (0.005, 0.02, 0.05, 0.1, 0.2)
NS = (1.5, 3.5)
CS = (0.5, 1.0, 2.0)
FINE_WCS = np.round(np.arange(111) * 0.01 + 0.5, 2)    # 0.50 .. 1.60
FINE_GAMS = np.geomspace(0.005, 2.0, 70)
ANGLES = np.arange(900) * 0.1                         # 0.0 .. 89.9 deg
DOCUMENTED = {0.9: 41.246, 1.0: 41.821, 1.1: 42.483}  # n_inc 1.5, g 0.05, gam 0.1
N_INC = 1.5                                           # glass, for sections 2 to 6
UNDERFLOW = -690.0                                    # exp() of less is below 1e-299


class Line:
    """A single Lorentz line pinned so that eps(1) = 1 - i g; f < 0 is gain."""

    def __init__(self, wc, gam, g):
        self.wc, self.gam, self.g = wc, gam, g
        lorentz_1 = wc**2 / (wc**2 - 1.0 - 1j * gam)
        self.f = -g / lorentz_1.imag
        self.eps_inf = 1.0 - self.f * lorentz_1.real

    def eps(self, w):
        w = np.asarray(w, dtype=complex)
        return self.eps_inf + self.f * self.wc**2 / (self.wc**2 - w * w - 1j * self.gam * w)

    def deps(self, w):
        u = self.wc**2 - w * w - 1j * self.gam * w
        return self.f * self.wc**2 * (2 * w + 1j * self.gam) / u**2

    def stable(self):
        """eps_inf > 0 and no zero of eps in the upper half w plane."""
        zeros = np.roots([-self.eps_inf, -1j * self.gam * self.eps_inf,
                          (self.eps_inf + self.f) * self.wc**2])
        return bool(self.eps_inf > 0 and np.all(zeros.imag < 0))

    def peak(self):
        """max over real w of |Im eps| = |f| wc^2 gam w / ((wc^2 - w^2)^2 + gam^2 w^2).

        Its stationary point solves 3 y^2 - (2 wc^2 - gam^2) y - wc^4 = 0, with y = w^2.
        """
        b = 2 * self.wc**2 - self.gam**2
        y = (b + np.sqrt(b * b + 12 * self.wc**4)) / 6
        return float(abs(self.eps(np.sqrt(y)).imag))

    def plausible(self):
        return self.stable() and self.peak() <= 1.0

    def causal_kz(self, kx, w0=1.0):
        """kz at w0 = 1 + i eta (eta >= 0), continued at fixed real kx down from w = 1 + i inf.

        kz^2(w) = eps_inf prod(w - r_j) / prod(w - d_k). The r_j are the four branch points
        (zeros of w^2 eps - kx^2), the d_k the two Lorentz poles. On the straight path from
        1 + i inf to w0, each factor's argument starts at pi/2 and turns by less than pi, so
        it ends at pi/2 + Arg(-i (w0 - r)). Half of each, summed, is the exact continuation of
        the root that decays at large s (kz -> i sqrt(eps_inf) s). Needs eps_inf > 0.
        """
        kx = np.atleast_1d(np.asarray(kx, dtype=float))
        K = kx * kx
        e, f, wc, gam = self.eps_inf, self.f, self.wc, self.gam
        one = np.ones_like(K)
        # w^2 (e D + f wc^2) - K D with D = wc^2 - w^2 - i gam w, descending powers of w
        coeffs = np.stack([-e * one, -1j * gam * e * one, (e + f) * wc**2 + K, 1j * gam * K,
                           -K * wc**2], axis=-1)
        companion = np.zeros((K.size, 4, 4), dtype=complex)
        companion[:, 0, :] = -coeffs[:, 1:] / coeffs[:, :1]
        companion[:, 1, 0] = companion[:, 2, 1] = companion[:, 3, 2] = 1.0
        branch = np.linalg.eigvals(companion)
        poles = np.roots([-1.0, -1j * gam, wc**2])

        def half_log(z):
            return 0.5 * (np.log(np.abs(z)) + 1j * (np.pi / 2 + np.angle(-1j * z)))

        total = half_log(w0 - branch).sum(axis=-1) - half_log(w0 - poles).sum()
        return np.sqrt(e) * np.exp(total)

    def crossings(self):
        """(s, kx) where kx_b(1 + i s) = (1 + i s) sqrt(eps(1 + i s)) lies on the real kx axis.

        There w^2 eps(w) is real and positive. Im[w^2 (e D + f wc^2) conj(D)] is a real
        polynomial of degree 6 in s; its positive real roots with Re(w^2 eps) > 0 are the
        crossings. At each, a branch point in w sits on the path w = 1 + i s for that kx.
        """
        P = np.polynomial.polynomial
        e, f, wc, gam = self.eps_inf, self.f, self.wc, self.gam
        w = np.array([1.0, 1j])                       # 1 + i s, ascending powers of s
        D = P.polysub(P.polysub([wc**2], P.polymul(w, w)), 1j * gam * w)
        num = P.polymul(P.polymul(w, w), P.polyadd(e * D, [f * wc**2]))
        out = []
        for s in P.polyroots(P.polymul(num, np.conj(D)).imag):
            if abs(s.imag) < 1e-9 * max(1.0, abs(s)) and s.real > 0:
                s = s.real
                F = complex((1 + 1j * s)**2 * self.eps(1 + 1j * s))
                if F.real > 0:
                    out.append((s, np.sqrt(F.real)))
        return sorted(out)


def tracked_kz(line, kx, top=40.0, rel=0.02):
    """The same continuation in small steps: |d(kz^2)| < rel |kz^2|, the nearer root each step."""
    def kz2(s):
        w = 1.0 + 1j * s
        return complex(w * w * line.eps(w) - kx * kx)
    s, q_old, h = top, kz2(top), top / 50
    v = np.sqrt(q_old)
    root = v if v.imag > 0 else -v
    while s > 0:
        h = min(h, s)
        while True:
            q_new = kz2(s - h)
            if abs(q_new - q_old) <= rel * min(abs(q_old), abs(q_new)) or h < 1e-15:
                break
            h *= 0.5
        v = np.sqrt(q_new)
        root = v if abs(v - root) <= abs(-v - root) else -v
        s, q_old, h = s - h, q_new, 1.5 * h
    return root


def flip_angles(line, n):
    """Angles in [0, 89.9] where Re(causal kz) changes sign: a 0.1 deg scan, then bisection."""
    sign = np.sign(line.causal_kz(n * np.sin(np.radians(ANGLES))).real)
    i = np.nonzero(sign[:-1] != sign[1:])[0]
    lo, hi, s_lo = ANGLES[i].copy(), ANGLES[i + 1].copy(), sign[i]
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        left = np.sign(line.causal_kz(n * np.sin(np.radians(mid))).real) == s_lo
        lo, hi = np.where(left, mid, lo), np.where(left, hi, mid)
    return list(0.5 * (lo + hi))


def ratio(kx, g):
    """|Re kz^2| / |Im kz^2| at w = 1, where eps(1) = 1 - i g."""
    return abs(1.0 - kx * kx) / g


def flips():
    print('1. flips -- where the fixed-kx causal root flips, for single Lorentz gain lines pinned '
          'to eps(1) = 1 - i g.')
    print('   A record, not the reason for c: a finite beam does not see these flips (section 4), '
          'and how many')
    print('   a band holds depends on the grid (the finer grid below).')
    rows = {}
    for g in GS:
        for wc in WCS:
            for gam in GAMS:
                line = Line(wc, gam, g)
                row = dict(line=line, stable=line.stable(), peak=line.peak(),
                           plausible=line.plausible())
                for n in NS:
                    row[n] = flip_angles(line, n) if line.eps_inf > 0 else []
                rows[g, wc, gam] = row

    print('  documented flips, n_inc 1.5, g 0.05, gam 0.1:')
    for wc, documented in DOCUMENTED.items():
        (theta,) = rows[0.05, wc, 0.1][1.5]
        print(f'    wc {wc}: {theta:.4f} deg (documented {documented:.3f}), |Re kz^2|/|Im kz^2| '
              f'there {ratio(1.5 * np.sin(np.radians(theta)), 0.05):.3f}')

    worst, agree, count = 0.0, 0, 0
    for row in rows.values():
        if row['line'].eps_inf <= 0:
            continue
        for theta in [30.0, 60.0] + [t + d for t in row[1.5] for d in (-0.02, 0.02)]:
            kx = 1.5 * np.sin(np.radians(theta))
            closed = complex(row['line'].causal_kz(kx)[0])
            stepped = tracked_kz(row['line'], kx)
            worst = max(worst, abs(closed - stepped) / abs(closed))
            agree += bool(np.sign(closed.real) == np.sign(stepped.real))
            count += 1
    print(f'  check: closed-form continuation vs small-step tracking at {count} points (n_inc 1.5;')
    print(f'    30 and 60 deg, and 0.02 deg either side of each flip): the sign of Re agrees at '
          f'{agree}/{count},')
    print(f'    max |difference|/|kz| {worst:.1e}')

    worst, matched, total, heights, per_line = 0.0, 0, 0, [], {}
    for row in rows.values():
        if row['line'].eps_inf <= 0:
            continue
        crossings = row['line'].crossings()
        heights += [s for s, _ in crossings]
        per_line[len(crossings)] = per_line.get(len(crossings), 0) + 1
        for n in NS:
            cross = [np.degrees(np.arcsin(kx / n)) for _, kx in crossings
                     if kx / n < np.sin(np.radians(ANGLES[-1]))]
            total += len(cross)
            if len(cross) == len(row[n]):
                matched += len(cross)
                if cross:
                    gap = np.max(np.abs(np.sort(cross) - np.sort(row[n])))
                    worst = max(worst, float(gap))
    print(f'  check: bisection vs the angles where kx_b(1 + i s) crosses the real kx axis: '
          f'{matched}/{total} matched one for one,')
    print(f'    max difference {worst:.1e} deg; crossings per line with eps_inf > 0: ' +
          ', '.join(f'{k}: {v}' for k, v in sorted(per_line.items())))
    print('  the flip is a convention: at the flip\'s kx the straight path w = 1 + i s runs into '
          'a branch point of kz(w),')
    print(f'    at s {min(heights):.4f} .. {max(heights):.3f} (every crossing above), in the upper '
          f'half w plane. The root below it depends on')
    print('    which side the path passes, and the straight path is one choice of side.')
    no_root = [key for key, row in rows.items() if row['line'].eps_inf <= 0]
    print(f'  {len(no_root)} lines have eps_inf <= 0, so no decaying root at large s (unstable, '
          f'skipped):')
    print('    ' + ', '.join(f'(g {g}, wc {wc}, gam {gam})' for g, wc, gam in no_root))

    print('\n  |Re kz^2| / |Im kz^2| at each flip. It does not depend on n_inc, which only maps kx '
          'to an angle. Listed')
    print('  for n_inc 3.5, which reaches every flip with kx < 3.5. "u" unstable, "p" peak '
          '|Im eps| > 1, ">" a flip')
    print('  beyond n_inc 1.5 (kx > 1.5 sin 89.9 deg), "-" no flip in 0 .. 89.9 deg.')
    print('       g    gam' + ''.join(f'  {"wc " + str(wc):>9s}' for wc in WCS))
    reach_15 = 1.5 * np.sin(np.radians(ANGLES[-1]))
    for g in GS:
        for gam in GAMS:
            cells = []
            for wc in WCS:
                row = rows[g, wc, gam]
                if not row[3.5]:
                    cells.append('-')
                    continue
                parts = []
                for theta in row[3.5]:
                    kx = 3.5 * np.sin(np.radians(theta))
                    parts.append(f'{ratio(kx, g):.2f}' + ('' if kx < reach_15 else '>'))
                tag = '' if row['plausible'] else ('u' if not row['stable'] else 'p')
                cells.append('/'.join(parts) + tag)
            print(f'    {g:5.3f}  {gam:5.2f}' + ''.join(f'  {cell:>9s}' for cell in cells))

    print('\n  per (n_inc, g): plausible lines, their flips, the largest ratio, and how many flips '
          'c = 0.5 / 1 / 2 hold')
    held = {c: [0, 0] for c in CS}
    for n in NS:
        for g in GS:
            plaus = [row for (gg, _, _), row in rows.items() if gg == g and row['plausible']]
            ratios = [ratio(n * np.sin(np.radians(t)), g) for row in plaus for t in row[n]]
            for c in CS:
                held[c][0] += sum(q < c for q in ratios)
                held[c][1] += len(ratios)
            print(f'    n {n}  g {g:<5}: {len(plaus):2d}/15 plausible, {len(ratios):2d} flips, '
                  f'max ratio {max(ratios):.3f}, held ' +
                  ' / '.join(f'{sum(q < c for q in ratios):2d}' for c in CS))
    print('  all plausible flips, both n_inc: ' +
          ', '.join(f'c {c} holds {a}/{b}' for c, (a, b) in held.items()) + ' (this grid only)')
    implausible = [ratio(n * np.sin(np.radians(t)), g) for (g, _, _), row in rows.items()
                   if not row['plausible'] for n in NS for t in row[n]]
    print(f'  implausible lines flip up to ratio {max(implausible):.2f}: no c bounds them')
    return rows, per_line


def fine_grid():
    """Section 1 on a finer grid, the flips from the crossings (checked above against bisection)."""
    print(f'\n  finer grid: wc {FINE_WCS[0]:.2f} .. {FINE_WCS[-1]:.2f} step 0.01, '
          f'{FINE_GAMS.size} log-spaced gam {FINE_GAMS[0]} .. {FINE_GAMS[-1]:g}, the same five g '
          f'({FINE_WCS.size * FINE_GAMS.size * len(GS)} lines)')
    lines, flips_, counts = 0, [], {}
    for g in GS:
        for wc in FINE_WCS:
            for gam in FINE_GAMS:
                line = Line(wc, gam, g)
                if line.eps_inf <= 0:
                    continue
                lines += 1
                crossings = line.crossings()
                counts[len(crossings)] = counts.get(len(crossings), 0) + 1
                if line.plausible():
                    flips_ += [(ratio(kx, g), line.peak(), g, wc, gam, kx) for _, kx in crossings]
    plausible = sum(1 for _ in {(f[2], f[3], f[4]) for f in flips_})
    print(f'    {lines} lines have eps_inf > 0; crossings per line: ' +
          ', '.join(f'{k}: {v}' for k, v in sorted(counts.items())))
    ratios = np.array([f[0] for f in flips_])
    print(f'    {plausible} plausible lines, {ratios.size} flips; c 0.5 / 1 / 2 hold ' +
          ' / '.join(f'{int((ratios < c).sum())}' for c in CS) + f' (of {ratios.size})')
    out = sorted(f for f in flips_ if f[0] >= 2.0)
    print(f'    {len(out)} plausible flips lie at ratio 2 or more: {out[0][0]:.3f} .. '
          f'{out[-1][0]:.3f}, at kx {min(f[5] for f in out):.3f} .. {max(f[5] for f in out):.3f} '
          f'(all reachable from glass)')
    q, peak, g, wc, gam, kx = out[-1]
    line = Line(wc, gam, g)
    side = 3.5 * np.sin(np.radians(np.array(flip_angles(line, 3.5))))
    print(f'    the largest: ratio {q:.3f} on (g {g}, wc {wc}, gam {gam:.5f}), peak |Im eps| '
          f'{peak:.3f}, kx {kx:.4f};')
    print('      bisection on that line finds ' + ', '.join(f'{ratio(k, g):.3f}' for k in side))
    print('    the weakest plausible line with a flip at or past each c, by peak |Im eps|:')
    for c in CS:
        q, peak, g, wc, gam, _ = min((f for f in flips_ if f[0] >= c), key=lambda f: f[1])
        print(f'      c {c:<3}: peak {peak:.3f} (g {g}, wc {wc}, gam {gam:.5f}), ratio {q:.3f}')
    print('    So narrow detuned lines with peak |Im eps| <= 1 flip at ratios above 2, and the '
          'coarse grid\'s')
    print('    "c 2 holds 104/104" is a property of that grid.')
    return dict(max=out[-1], largest=out[-1][0], outside=len(out), flips=ratios.size,
                lines=lines, once=counts.get(1, 0))


# --- 2. beam ------------------------------------------------------------------------------------

def _r_s(k, kz, w=1.0):
    """s-polarized Fresnel reflection, glass onto the gain medium, mu = 1, at frequency w."""
    k1 = np.sqrt(N_INC**2 * w * w - k * k + 0j)
    return (k1 - kz) / (k1 + kz)


def _gauss(a, b, panels, order=16):
    """Composite Gauss-Legendre nodes and weights on [a, b]."""
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, panels + 1)
    half, mid = np.diff(edges) / 2, (edges[:-1] + edges[1:]) / 2
    return (mid[:, None] + half[:, None] * x).ravel(), (half[:, None] * w).ravel()


def _track(values, start):
    """Continue sqrt(values) from `start` along ordered samples; also its largest turn in degrees.

    The nearer of the two roots is the continuation while consecutive samples turn the root by
    well under 90 degrees, so the largest turn checks the sampling.
    """
    roots = np.sqrt(values.astype(complex))
    out, prev, turn = np.empty_like(roots), start, 0.0
    for i, v in enumerate(roots):
        v = v if abs(v - prev) <= abs(-v - prev) else -v
        turn = max(turn, abs(np.degrees(np.angle(v / prev))))
        out[i] = prev = v
    return out, turn


def gaussian(k0, W, x=0.0):
    """The spectrum exp(-((k - k0) W / 2)^2) of exp(-x^2 / W^2) e^{i k0 x}, seen at x.

    Its integral over real k is E_inc(0) = 2 sqrt(pi) / W.
    """
    return lambda k: np.exp(-((k - k0) * W / 2)**2 + 1j * k * x)


def top_hat(k0, L):
    """The spectrum 2 sin((k - k0) L / 2) / (k - k0) of a field e^{i k0 x} on |x| < L / 2.

    Its integral over real k is 2 pi E_inc(0) = 2 pi.
    """
    def a(k):
        z = (k - k0) * L / 2
        return L * np.sin(z) / z
    return a


def angle(kx):
    return float(np.degrees(np.arcsin(kx / N_INC)))


class Beam:
    """Glass onto one Lorentz line; the reflected field at x = 0 for a beam spectrum a(k)."""

    def __init__(self, line):
        self.line = line
        self.eps1 = complex(line.eps(1.0))
        self.k_m = np.sqrt(self.eps1.real)             # Maystre's step on the real axis
        crossings = line.crossings()
        if len(crossings) != 1:
            raise RuntimeError(f'expected one real-axis crossing, got {crossings}')
        ((self.s_c, self.kx_c),) = crossings
        self.tip = complex(np.sqrt(self.eps1))        # kx_b(1)
        self.gain_length = 2 * np.sqrt(self.eps1.real) / abs(self.eps1.imag)

    def axis_root(self, k):
        """The root continued along the real axis from kx = 0. kz^2 stays below the real axis."""
        return np.sqrt(self.eps1 - k * k + 0j)

    def maystre(self, k):
        v = self.axis_root(k)
        return np.where(v.real + v.imag < 0, -v, v)

    def tip_exponent(self, k0, W):
        """log of the Gaussian weight at the tip, (W/2)^2 (Im^2 - (k0 - Re)^2)."""
        return (W / 2)**2 * (self.tip.imag**2 - (k0 - self.tip.real)**2)

    def kb(self, s):
        w = 1.0 + 1j * s
        return w * np.sqrt(self.line.eps(w))

    def dkb(self, s):
        w = 1.0 + 1j * s
        root = np.sqrt(self.line.eps(w))
        return 1j * (root + w * self.line.deps(w) / (2 * root))

    def finger(self, k, dk, foot_root, a):
        """Down one side of the cut along k (foot to tip), round the tip, up the other side.

        Also the root's largest turn per node, and the sum of the terms' sizes.
        """
        rho, turn = _track(self.eps1 - k * k, foot_root)
        terms = (_r_s(k, rho) - _r_s(k, -rho)) * a(k) * dk
        return np.sum(terms), turn, np.sum(np.abs(terms))

    def deviation(self, a, panels):
        """E_C - E_M for the spectrum a, by both fingers (A and B).

        Also B's finger alone, A's real-axis part ("step"), the sum of the sizes of A's terms
        ("scale", which sets A's round-off), and the root's largest turn per node.
        """
        # A: the finger along the trajectory, s = s_c u^2 (u = 1 the foot, u = 0 the tip). The
        # axis kernel is the fixed-kx causal one: it steps at kx_c, Maystre's at k_m.
        u, wu = _gauss(0.0, 1.0, panels)
        u, wu = u[::-1], wu[::-1]
        s = self.s_c * u * u
        with np.errstate(over='ignore', invalid='ignore'):
            f_a, turn_a, size_a = self.finger(self.kb(s), -wu * self.dkb(s) * 2 * self.s_c * u,
                                              self.axis_root(self.kx_c), a)
        x, wx = _gauss(*sorted((self.k_m, self.kx_c)), panels)
        causal = np.where(x < self.kx_c, self.axis_root(x), -self.axis_root(x))
        terms = (_r_s(x, causal) - _r_s(x, self.maystre(x))) * a(x) * wx
        axis_a = np.sum(terms)
        # B: the finger straight down from Re kx_b(1), k = Re + i Im (1 - v^2); the axis kernel
        # steps at Re kx_b(1). It uses eps(1) alone.
        v, wv = _gauss(0.0, 1.0, panels)
        v, wv = v[::-1], wv[::-1]
        k = self.tip.real + 1j * self.tip.imag * (1 - v * v)
        f_b, turn_b, _ = self.finger(k, -wv * (-2j * self.tip.imag * v),
                                     self.axis_root(self.tip.real), a)
        x, wx = _gauss(*sorted((self.k_m, self.tip.real)), panels)
        stepped = np.where(x < self.tip.real, self.axis_root(x), -self.axis_root(x))
        axis_b = np.sum((_r_s(x, stepped) - _r_s(x, self.maystre(x))) * a(x) * wx)
        return dict(A=axis_a + f_a, B=axis_b + f_b, finger=abs(f_b), step=abs(axis_a),
                    scale=size_a + np.sum(np.abs(terms)), turn=max(turn_a, turn_b))

    def maystre_field(self, k0, W, panels):
        """E_M: the Gaussian beam under Maystre's kernel on the real axis, split at its steps."""
        reach = 2 * 9.7 / W                           # a(k) < 1e-40 beyond
        cuts = {p for p in (self.k_m, N_INC) if abs(k0 - p) < reach}
        edges = sorted({k0 - reach, k0 + reach} | cuts)
        a, e_m = gaussian(k0, W), 0.0
        for left, right in zip(edges[:-1], edges[1:]):
            x, wx = _gauss(left, right, 8 * panels)
            e_m = e_m + np.sum(_r_s(x, self.maystre(x)) * a(x) * wx)
        return e_m

    def measure(self, k0, W):
        """The Gaussian beam at x = 0, in units of E_inc(0), with its two accuracy checks.

        under: the deviation is below the double range (its tip weight under exp(-690), or it
        rounds to 0). Then only E_M is refined, and the Cauchy check is skipped.
        """
        e_inc = 2 * np.sqrt(np.pi) / W
        a = gaussian(k0, W)
        coarse, fine = self.deviation(a, 40), self.deviation(a, 80)
        m_coarse, m_fine = self.maystre_field(k0, W, 40), self.maystre_field(k0, W, 80)
        under = self.tip_exponent(k0, W) < UNDERFLOW or fine['B'] == 0
        out = dict(e_m=m_fine / e_inc, dev=0.0 if under else fine['B'] / e_inc, under=under,
                   finger=fine['finger'] / e_inc, turn=fine['turn'], step=fine['step'] / e_inc,
                   scale=fine['scale'] / e_inc)
        refine = abs(m_fine - m_coarse) / abs(m_fine)
        if under:
            out.update(cauchy=0.0, cauchy_abs=np.nan, refine=refine)
        else:
            with np.errstate(over='ignore', invalid='ignore'):
                out.update(cauchy_abs=abs(fine['A'] - fine['B']) / e_inc,
                           cauchy=abs(fine['A'] - fine['B']) / abs(fine['B']),
                           refine=max(refine, abs(fine['B'] - coarse['B']) / abs(fine['B'])))
        out['field'] = out['e_m'] + out['dev']
        return out

    def contour(self, a, w, depth, left, right):
        """E at w on the real axis with a detour to Im kx = -depth over [left, right].

        depth > 0 lowers the detour, depth < 0 raises it. No finger and no step: the root is
        carried along the contour from the real axis on its left, where it is the fixed-kx
        causal root at w. Round-off grows as exp((depth W / 2)^2).
        """
        corners = [left, left - 1j * depth, right - 1j * depth, right]
        k, dk = [], []
        for p0, p1 in zip(corners[:-1], corners[1:]):
            t, wt = _gauss(0.0, 1.0, int(np.ceil(abs(p1 - p0) / 0.002)))
            k.append(p0 + (p1 - p0) * t)
            dk.append((p1 - p0) * wt)
        k, dk = np.concatenate(k), np.concatenate(dk)
        start = complex(self.line.causal_kz(left, w)[0])
        rho, turn = _track(w * w * self.line.eps(w) - k * k, start)
        return np.sum(_r_s(k, rho, w) * a(k) * dk), turn

    def undeformed(self, a, w, lo, hi, panels=1600):
        """E at w = 1 + i eta on the real kx axis with the fixed-kx causal root at that w.

        Valid while kx_b(w) is above the axis. The incidence medium's root has Im > 0 there.
        The beam spectrum is unchanged; only the frequency moves.
        """
        x, wx = _gauss(lo, hi, panels)
        return np.sum(_r_s(x, self.line.causal_kz(x, w), w) * a(x) * wx)


def beam_table(b, rows):
    """Print section 2's table and return its measurements."""
    print('  Columns. E_r(0) and E_inc(0) are the reflected and incident fields at x = 0, the beam '
          'centre.')
    print('  |E_r/E_inc|^2 is the squared field ratio there, not a reflectance: inside the window the '
          'reflected field is a')
    print('  lateral wave that grows downstream (below). E_r takes the Briggs-Bers contour; E_M is '
          'the same beam')
    print('  under Maystre\'s kernel. |dev| = |E_r - E_M| / |E_inc|, the distance to that beam. '
          '|to r| = |E_r/E_inc - r|,')
    print('  the distance to the plane wave\'s r at the beam\'s centre angle. |finger| is finger '
          'B\'s part of dev.')
    print('  Cauchy = |dev_A - dev_B| / |dev| for the two fingers. refine = the larger relative '
          'change of dev and of')
    print('  E_M when the nodes double. A row past 1e-8 in either prints as unreliable.')
    print('   theta  ratio  |r|^2 plane     W   |E_M/E_inc|^2   |E_r/E_inc|^2      |dev|     '
          '|to r|  |finger|  Cauchy  refine')
    found = {}
    for theta, widths in rows:
        k0 = N_INC * np.sin(np.radians(theta))
        r_plane = complex(_r_s(k0, b.maystre(k0)))
        for W in widths:
            m = found[theta, W] = b.measure(k0, W)
            head = (f'   {theta:5.1f}  {ratio(k0, b.line.g):5.2f}  {abs(r_plane)**2:11.5f}  '
                    f'{W:4.0f}  {abs(m["e_m"])**2:14.6f}')
            if max(m['cauchy'], m['refine']) > 1e-8 or m['turn'] > 30.0:
                print(head + f'  unreliable: Cauchy {m["cauchy"]:.1e}, refine {m["refine"]:.1e}, '
                      f'root turn {m["turn"]:.0f} deg')
                continue
            print(head + f'  {abs(m["field"])**2:14.6g}  {abs(m["dev"]):9.2e}  '
                  f'{abs(m["field"] - r_plane):9.2e}  {m["finger"]:8.1e}  {m["cauchy"]:6.0e}  '
                  f'{m["refine"]:6.0e}')
    return found


def beam(rows):
    b = Beam(Line(1.0, 0.1, 0.05))
    print('\n2. beam -- glass (n 1.5) onto the centred line (wc 1, gam 0.1, g 0.05), '
          's polarization, field at the beam centre')
    (flip,) = rows[0.05, 1.0, 0.1][1.5]
    print(f'  kx_b(1 + i s) crosses the real axis once: s = {b.s_c:.5f}, kx_c = {b.kx_c:.6f} '
          f'= 1.5 sin {angle(b.kx_c):.4f} deg (section 1: {flip:.4f} deg)')
    print(f'  it ends at kx_b(1) = {b.tip.real:.6f} {b.tip.imag:+.6f}i, below the axis, so the '
          f'contour dips under it')
    s = np.linspace(0, b.s_c, 20001)
    path = b.kb(s)
    print(f'  check: the trajectory is continuous from s_c to 0 (largest step '
          f'{np.max(np.abs(np.diff(path))):.1e} for ds {s[1]:.1e})')
    print(f'    and below the axis there (max Im {path[:-1].imag.max():.1e})')
    k = np.linspace(0.5, 1.45, 4001)
    k = k[np.abs(k - b.kx_c) > 1e-9]
    stepped = np.where(k < b.kx_c, b.axis_root(k), -b.axis_root(k))
    print('  check: on the real axis the fixed-kx causal root of section 1 is the axis root '
          'stepped at kx_c: max |difference|')
    print(f'    {np.max(np.abs(stepped - b.line.causal_kz(k))):.1e} over kx 0.5 .. 1.45')

    rows_2 = [(42.0, (100.0, 200.0, 300.0))] + [(t, (100.0, 200.0, 300.0, 400.0, 800.0))
                                                for t in (43.3, 43.8, 44.2, 44.5, 45.0)]
    thetas = [t for t, _ in rows_2]
    plane = {}
    for theta in thetas:
        kx = N_INC * np.sin(np.radians(theta))
        plane[theta] = abs(complex(_r_s(kx, b.maystre(kx))))**2
    solved, fired = _solve_glass_onto(b.eps1, thetas)
    print(f'  check: wavebit R vs NumPy Fresnel with Maystre\'s root, max |difference| '
          f'{max(abs(solved[t] - plane[t]) for t in thetas):.1e}')
    print('    GainCutoffWarning at ' +
          ', '.join(f'{t} deg {"yes" if fired[t] else "no"}' for t in thetas))
    k_lo, k_hi = b.tip.real - abs(b.tip.imag), b.tip.real + abs(b.tip.imag)
    window = (angle(k_lo), angle(k_hi), ratio(k_lo, 0.05), ratio(k_hi, 0.05))
    print('  the window: the finger\'s Gaussian weight at the tip, exp((W/2)^2 (Im^2 - (kx0 - '
          'Re)^2)), grows with W for')
    print(f'    |kx0 - Re kx_b(1)| < |Im kx_b(1)|: kx0 {k_lo:.6f} .. {k_hi:.6f}, theta '
          f'{window[0]:.2f} .. {window[1]:.2f} deg,')
    print(f'    where the ratio is below {window[2]:.3f} and {window[3]:.3f} on the two sides')
    print(f'  the gain length L_g = 2 sqrt(Re eps) / (k0 |Im eps|) = {b.gain_length:.0f}, so W '
          f'100 .. 800 is W/L_g {100 / b.gain_length:.1f} .. {800 / b.gain_length:.1f}')
    found = beam_table(b, rows_2)

    # the lateral wave: the deviation's growth along x at 42 deg
    k0 = N_INC * np.sin(np.radians(42.0))
    rates = []
    for W in (200.0, 300.0):
        e_inc = 2 * np.sqrt(np.pi) / W
        devs = [abs(b.deviation(gaussian(k0, W, x), 80)['B']) / e_inc for x in (-50.0, 0.0, 50.0)]
        rates.append(np.log(devs[2] / devs[0]) / 100)
        print(f'  the lateral wave, 42 deg, W {W:.0f}: |E_r - E_M| / |E_inc(0)| at '
              f'x = -50 / 0 / 50 is ' + ' / '.join(f'{v:.3g}' for v in devs) + ',')
        print(f'    a growth of {rates[-1]:.4f} per unit x downstream')
    print(f'  The growth approaches |Im kx_b(1)| = {abs(b.tip.imag):.4f} as W grows. The field at '
          f'x = 0 is fed from upstream,')
    print('    so a wider beam feeds it from further away. Inside the window |E_r/E_inc|^2 is not '
          'a reflectance.')

    # the deformation's direction: the lowered contour carried in w from above the crossing
    theta, W, depth = 42.0, 100.0, 0.035
    k0 = N_INC * np.sin(np.radians(theta))
    a, e_inc, reach = gaussian(k0, W), 2 * np.sqrt(np.pi) / W, 2 * 9.7 / W
    left, right = min(k0 - reach, 0.95), max(k0 + reach, 1.05)
    print(f'  the deformation\'s direction, {theta:.0f} deg, W {W:.0f}. Fingers A and B and a '
          f'lowered contour at w = 1 share one')
    print('    topology, so their agreement tests the integration, not the direction. Only a '
          'continuation in w tests')
    print(f'    the direction. A contour lowered to Im kx = -{depth} over kx {left:.3f} .. '
          f'{right:.3f}, the root carried along it,')
    print('    must match the undeformed real-axis integral at w = 1 + i eta while kx_b(w) is '
          'above the axis (eta > s_c')
    gaps, misses, turns = [], [], []
    for eta in (0.02, 0.025, 0.03):
        w = 1.0 + 1j * eta
        und = b.undeformed(a, w, k0 - reach, k0 + reach)
        low, turn = b.contour(a, w, depth, left, right)
        high, _ = b.contour(a, w, -depth, left, right)
        gaps.append(f'{abs(low - und) / abs(und):.0e}')
        misses.append(f'{abs(high - und) / abs(und):.2f}')
        turns.append(turn)
    print(f'    = {b.s_c:.4f}). At eta 0.02 / 0.025 / 0.03 the relative gap is {" / ".join(gaps)}. '
          f'Control: the same contour')
    print(f'    raised to Im kx = +{depth} passes above kx_b(w) and misses by '
          f'{" / ".join(misses)}.')
    etas = np.linspace(0.03, 0.0, 31)
    path = b.kb(etas)
    clear = float(np.min(path.imag + depth))
    inside = bool(np.all((path.real > left) & (path.real < right)))
    for eta in etas:
        _, turn = b.contour(a, 1.0 + 1j * eta, depth, left, right)
        turns.append(turn)
    low, _ = b.contour(a, 1.0 + 0j, depth, left, right)
    m = found[theta, W]
    print('    Carried from eta 0.03 to 0 in 31 steps, kx_b(1 + i eta) stays above the lowered '
          'contour (least clearance')
    print(f'    {clear:.4f}; Re inside the detour: {"yes" if inside else "NO"}), so the '
          f'integral is analytic in eta all the way (largest root')
    print(f'    turn per node {max(turns):.1f} deg). At eta = 0 it gives E_r: |E_low - E_r| / '
          f'|E_r - E_M| = {abs(low / e_inc - m["field"]) / abs(m["dev"]):.0e}.')
    return b, found, window, rates


# --- 3. margin ----------------------------------------------------------------------------------

def _exp(log_value, width=9):
    """exp(log_value) as %.2e, or as e^N when it is below the double range."""
    if log_value < UNDERFLOW:
        return f'{"e^" + format(log_value, ".0f"):>{width}s}'
    return f'{np.exp(log_value):{width}.2e}'


def _sci(value, width=9):
    """A deviation as %.2e, or "< 1e-299" when it underflowed to 0."""
    return f'{"< 1e-299":>{width}s}' if value == 0 else f'{value:{width}.2e}'


def margin(b_centred):
    rho = ratio(N_INC * np.sin(np.radians(44.5)), 0.05)
    print(f'\n3. margin -- past the c = 2 edge, at ratio {rho:.3f} (44.5 deg at g 0.05), '
          f'high side: |E_r - E_M| / |E_inc| at x = 0')
    print('  The tip term goes as exp((W/2)^2 (Im^2 - (kx0 - Re)^2)), kx_b(1) = Re + i Im ("tip"). '
          'For weak gain that is')
    print('  exp(-(rho^2 - 1) (W/L_g)^2 / 4) ("formula"), with the gain length '
          'L_g = 2 sqrt(Re eps) / (k0 |Im eps|).')
    print('  Lines: wc 1, gam 0.1, pinned to eps(1) = 1 - i g. A row whose tip weight is below '
          'exp(-690) is skipped:')
    print('  the deviation is below the double range. Cauchy and refine as in section 2; past 1e-8 '
          'prints as unreliable.')
    print('       g    L_g   theta       W   W/L_g      |dev|    formula        tip  Cauchy  '
          'refine')
    beams = {g: Beam(Line(1.0, 0.1, g)) for g in (0.005, 0.05, 0.2)}
    table = {}
    for g, b in beams.items():
        k0 = np.sqrt(1 + rho * g)
        for W in (30.0, 100.0, 300.0, 1000.0, 3000.0):
            x = W / b.gain_length
            formula, tip = -(rho**2 - 1) * x * x / 4, b.tip_exponent(k0, W)
            head = f'   {g:5.3f}  {b.gain_length:5.0f}  {angle(k0):6.3f}  {W:6.0f}  {x:6.2f}'
            if tip < UNDERFLOW:
                print(head + f'  skipped: tip weight e^{tip:.0f}, formula e^{formula:.0f}')
                continue
            e_inc = 2 * np.sqrt(np.pi) / W
            coarse, fine = b.deviation(gaussian(k0, W), 40), b.deviation(gaussian(k0, W), 80)
            dev = table[g, W] = abs(fine['B']) / e_inc
            cauchy = abs(fine['A'] - fine['B']) / abs(fine['B'])
            refine = abs(fine['B'] - coarse['B']) / abs(fine['B'])
            if max(cauchy, refine) > 1e-8 or fine['turn'] > 30.0:
                print(head + f'  unreliable: Cauchy {cauchy:.1e}, refine {refine:.1e}')
                continue
            print(head + f'  {dev:9.2e}  {_exp(formula)}  {_exp(tip)}  {cauchy:6.0e}  '
                  f'{refine:6.0e}')
    m = b_centred.measure(N_INC * np.sin(np.radians(44.5)), 100.0)
    print(f'  check: the g 0.05 row at W 100 is section 2\'s 44.5 deg row ({abs(m["dev"]):.2e})')

    print('  how wide a beam must be at the edge itself (ratio 2, high side): the W/L_g past which '
          '|dev| stays below')
    print('  1e-3 and 1e-6. Scanned on 121 log-spaced widths from W 10 to 20 L_g, then bisected; '
          'the formula alone')
    print('  would ask for sqrt(4 ln(1/tol) / 3). The mirror point\'s Gaussian weight is at most '
          + _exp(max(-(5.0**2) * ((np.sqrt(1 + 2.0 * q) + b.tip.real)**2 - b.tip.imag**2)
                     for q, b in beams.items())).strip() + ' (W 10),')
    print('  so it stays out.')
    needs = {}
    for tol in (1e-3, 1e-6):
        cells = []
        for g, b in beams.items():
            k0 = np.sqrt(1 + 2.0 * g)

            def dev(W):
                return abs(b.deviation(gaussian(k0, W), 40)['B']) / (2 * np.sqrt(np.pi) / W)
            widths = np.geomspace(10.0, 20 * b.gain_length, 121)
            above = [i for i, W in enumerate(widths) if dev(W) >= tol]
            if not above:
                needs[tol, g] = None
                cells.append(f'g {g}: below at W 10')
                continue
            lo, hi = widths[above[-1]], widths[above[-1] + 1]
            for _ in range(40):
                mid = np.sqrt(lo * hi)
                lo, hi = (mid, hi) if dev(mid) >= tol else (lo, mid)
            needs[tol, g] = hi / b.gain_length
            cells.append(f'g {g}: {hi / b.gain_length:.2f}')
        needs[tol, 'formula'] = np.sqrt(4 * np.log(1 / tol) / (2.0**2 - 1))
        print(f'    {tol:.0e}: ' + ', '.join(cells) + f'; formula {needs[tol, "formula"]:.2f}')
    return dict(rho=rho, table=table, needs=needs)


# --- 4. far flips -------------------------------------------------------------------------------

def far_flips(rows, fine_max):
    print('\n4. far flip -- beams between Maystre\'s step and a far fixed-kx flip: the beam '
          'goes to Maystre\'s value')
    print('  B uses eps(1) alone. A runs along the line\'s own trajectory kx_b(1 + i s). A\'s '
          'real-axis part carries the')
    print('  fixed-kx causal kernel over the beam centre ("step"), and its finger cancels that. '
          'So |A - B| is read')
    print('  against "A size", the sum of the sizes of A\'s terms, which sets A\'s round-off. '
          '"overflow": the Gaussian')
    print('  on the trajectory leaves the double range; "A unresolved": past 1e-10 of A size. '
          'refine as in section 2.')
    q, g, wc, gam = max(((ratio(n * np.sin(np.radians(t)), g), g, wc, gam)
                         for (g, wc, gam), row in rows.items() if row['plausible']
                         for n in NS for t in row[n]))
    cases = [(f'the coarse grid\'s largest flip: (g {g}, wc {wc}, gam {gam}), ratio {q:.3f}',
              Line(wc, gam, g), [48.03], (100.0, 200.0, 400.0, 800.0, 1600.0))]
    q, _, g, wc, gam, kx = fine_max
    side = np.sign(kx - 1.0)
    cases.append((f'the finer grid\'s largest flip: (g {g}, wc {wc}, gam {gam:.5f}), ratio {q:.3f}',
                  Line(wc, gam, g), [angle(np.sqrt(1 + side * r * g)) for r in (3.0, 4.0)],
                  (100.0, 200.0, 400.0, 800.0)))
    out = []
    for title, line, thetas, widths in cases:
        b = Beam(line)
        print(f'  {title},')
        print(f'    kx_c {b.kx_c:.4f}, kx_b(1) = {b.tip.real:.5f} {b.tip.imag:+.5f}i')
        for theta in thetas:
            k0 = N_INC * np.sin(np.radians(theta))
            r_m = complex(_r_s(k0, b.maystre(k0)))
            r_c = complex(_r_s(k0, b.line.causal_kz(k0)[0]))
            print(f'    {theta:.2f} deg, ratio {ratio(k0, line.g):.3f}, |kx0 - Re| / |Im| '
                  f'{abs(k0 - b.tip.real) / abs(b.tip.imag):.3f}. Plane wave |r|^2: Maystre '
                  f'{abs(r_m)**2:.4f}, fixed-kx causal {abs(r_c)**2:.4f}')
            print('          W   |E_r/E_inc|^2      |dev|     |to r|   A size  |A-B|/size     '
                  'step  refine')
            beams = []
            for W in widths:
                m = b.measure(k0, W)
                beams.append(abs(m['field'])**2)
                if not np.isfinite(m['scale']):
                    check = '   overflow           '
                elif m['under']:
                    check = f'  {m["scale"]:7.1e}           -'
                else:
                    check = (f'  {m["scale"]:7.1e}  {m["cauchy_abs"] / m["scale"]:10.1e}')
                text = (f'      {W:5.0f}  {abs(m["field"])**2:14.6f}  {_sci(abs(m["dev"]))}  '
                        f'{abs(m["field"] - r_m):9.2e}{check}  {m["step"]:7.1e}  '
                        f'{m["refine"]:6.0e}')
                if m['refine'] > 1e-8 or m['turn'] > 30.0:
                    text += f'  unreliable (root turn {m["turn"]:.0f} deg)'
                elif np.isfinite(m['scale']) and not m['under'] and \
                        m['cauchy_abs'] > 1e-10 * m['scale']:
                    text += '  A unresolved'
                print(text)
            out.append((line, theta, ratio(k0, line.g), abs(r_m)**2, abs(r_c)**2, beams))
    return out


# --- 5. top hat ---------------------------------------------------------------------------------

def top_hats(b):
    print('\n5. top hat -- a field e^{i k0 x} on |x| < L/2 onto the centred line (g 0.05): '
          '|E_r - E_M| / |E_inc| at x = 0')
    print('  Its spectrum is entire and of exponential type, so on the finger it weighs up to '
          'exp(|Im kx_b(1)| L/2),')
    print('  at any angle. Its upstream edge seeds a lateral wave that grows over L/2 before it '
          'reaches x = 0.')
    print('  Unlike a Gaussian\'s, its spectrum is not small at the mirror point -kx_b(1), which '
          'the contour passes')
    print('  above. By symmetry that finger is finger B with a(k) replaced by a(-k); "mirror" is '
          'its share. The')
    print('  Gaussian column is section 2\'s deviation at W = L. Cauchy and refine as in '
          'section 2, on the sum.')
    print('   theta  ratio      L    top hat  mirror  exp(|Im| L/2)   Gaussian  Cauchy  refine')
    out = {}
    for theta in (44.5, 50.0):
        k0 = N_INC * np.sin(np.radians(theta))
        for L in (100.0, 200.0, 400.0, 800.0):
            a = top_hat(k0, L)

            def both(panels):
                main, mirror = b.deviation(a, panels), b.deviation(lambda k: a(-k), panels)
                return main['A'] + mirror['A'], main['B'] + mirror['B'], abs(mirror['B'])
            _, coarse, _ = both(40)
            fine_a, fine, mirror = both(80)
            dev = out[theta, L] = abs(fine) / (2 * np.pi)
            cauchy = abs(fine_a - fine) / abs(fine)
            refine = abs(fine - coarse) / abs(fine)
            gauss = abs(b.deviation(gaussian(k0, L), 80)['B']) / (2 * np.sqrt(np.pi) / L)
            print(f'   {theta:5.1f}  {ratio(k0, 0.05):5.2f}  {L:5.0f}  {dev:9.2e}  '
                  f'{mirror / abs(fine):6.3f}  {np.exp(abs(b.tip.imag) * L / 2):13.3g}  '
                  f'{_sci(gauss)}  {cauchy:6.0e}  {refine:6.0e}')
    return out


# --- 6. firing ----------------------------------------------------------------------------------

def _solve_glass_onto(eps, thetas):
    """R, and whether GainCutoffWarning fired, from one zeroth-order solve per angle.

    The layer is 100 nm of glass under glass; a layer of the incidence medium leaves |r| alone.
    """
    R, fired = {}, {}
    for theta in thetas:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = _legacy_wavebit.rcwa(backend='numpy', top=N_INC**2, bot=eps, wavelength=900.0,
                                  thickness=[100.0], period=[500.0],
                                  structure=np.full((1, 1, 4), N_INC**2 + 0j),
                                  theta=np.radians(theta), phi=0.0, truncation_order=(0, 0),
                                  tetm='te').solve()
        R[theta] = float(np.sum(result.R))
        fired[theta] = any(issubclass(w.category, wavebit.GainCutoffWarning) for w in caught)
    return R, fired


def firing():
    print('\n6. firing -- glass (n 1.5) onto eps = 1 - i g, the 900 angles 0.0 .. 89.9 deg, c = 2')
    kx = N_INC * np.sin(np.radians(ANGLES))
    for g in (0.005, 0.05, 0.2):
        band = np.abs(1 - kx * kx) < 2 * g
        _, fired = _solve_glass_onto(1 - 1j * g, ANGLES)
        solver = np.array([fired[t] for t in ANGLES])
        same = 'the same angles' if np.array_equal(band, solver) else 'DIFFERENT angles'
        print(f'  g {g:<5}: {band.sum():3d}/900 angles in the band, {ANGLES[band][0]:.1f} .. '
              f'{ANGLES[band][-1]:.1f} deg; the solver warned on {solver.sum():3d}, {same}')


def _point(number, text):
    print(textwrap.fill(text, width=110, initial_indent=f'  {number}. ',
                        subsequent_indent='     '))


def conclusions(fine, coarse_crossings, beam_, margin_, far, hats):
    b, found, window, rates = beam_
    g, L_g = b.line.g, b.gain_length
    print('\nconclusions')
    _point(1, f'For a Gaussian beam the field at the beam centre grows with W where |kx0 - Re sqrt(eps)| < '
              f'|Im sqrt(eps)|, a ratio below about 1 ({window[2]:.3f} and {window[3]:.3f} on the '
              f'two sides at g {g}; {window[0]:.2f} .. {window[1]:.2f} deg from glass): '
              f'|E_r/E_inc|^2 = ' +
              ' / '.join(f'{abs(found[42.0, W]["field"])**2:.3g}' for W in (100.0, 200.0, 300.0)) +
              f' at 42 deg for W 100 / 200 / 300. That field is a lateral wave growing downstream '
              f'at about |Im kx_b(1)| = {abs(b.tip.imag):.3f} per unit x ({rates[0]:.4f} and '
              f'{rates[1]:.4f} measured), not a reflectance.')
    widths = (100.0, 200.0, 400.0, 800.0)
    _point(2, f'c = 2 adds the zone 1 < ratio < 2. There a beam does converge to Maystre\'s '
              f'value, but beams a few gain lengths wide have not yet: at ratio '
              f'{ratio(N_INC * np.sin(np.radians(43.3)), g):.2f} (43.3 deg) |dev| is ' +
              ' / '.join(f'{abs(found[43.3, W]["dev"]):.2e}' for W in widths) + ' for W/L_g ' +
              ' / '.join(f'{W / L_g:g}' for W in widths) + '.')
    needs = margin_['needs']

    def span(tol):
        values = [needs[tol, q] for q in (0.005, 0.05, 0.2) if needs[tol, q] is not None]
        return f'{min(values):.1f} .. {max(values):.1f}'
    _point(3, f'At the c = 2 edge, on its high side (the slower one), a beam must be W/L_g = {span(1e-3)} wide for its field to stay '
              f'within 1e-3 of Maystre\'s beam, and {span(1e-6)} for 1e-6 (g 0.005 .. 0.2), with '
              f'L_g = 2 sqrt(Re eps) / (k0 |Im eps|). The tip weight alone, exp(-(rho^2 - 1) '
              f'(W/L_g)^2 / 4), asks for {needs[1e-3, "formula"]:.2f} and '
              f'{needs[1e-6, "formula"]:.2f}.')
    (_, _, rho_a, rm_a, rc_a, _), *rest = far
    _point(4, f'The fixed-kx causal flips of section 1 are not what a finite beam sees. Between '
              f'Maystre\'s step and a flip at ratio {fine["largest"]:.2f}, beams at ratio ' +
              ' and '.join(f'{r[2]:.2f}' for r in rest) + ' go to Maystre\'s |r|^2 (' +
              ', '.join(f'{r[3]:.4f}' for r in rest) + '), not to the fixed-kx causal one (' +
              ', '.join(f'{r[4]:.4f}' for r in rest) + f'). So does a beam at ratio {rho_a:.2f} '
              f'on the coarse grid\'s furthest flip ({rm_a:.4f}, not {rc_a:.4f}). How many flips '
              f'a band holds depends on the grid: c = 2 holds 104/104 on the coarse one, and '
              f'{fine["outside"]} of {fine["flips"]} flips lie past 2 on the finer one.')
    _point(5, f'No single-frequency check can see the contour\'s topology. The beam\'s answer '
              f'depends on eps(1) and on kx_b(w) crossing the real axis once as Im(w) falls to 0 '
              f'(checked in section 2 by continuing in w, which tests the direction; section 4\'s '
              f'A against B is an integration check). Every line here with eps_inf > 0 crosses once '
              f'({coarse_crossings.get(1, 0)} of {sum(coarse_crossings.values())} on the coarse '
              f'grid, {fine["once"]} of {fine["lines"]} on the finer one). A medium whose branch '
              f'point does not cross the real axis once needs another contour, and nothing at one '
              f'frequency, the warning included, can tell.')
    _point(6, f'The window of ratio ~1 is a property of the Gaussian. A top hat deviates by '
              f'{hats[44.5, 100.0]:.1e} .. {hats[44.5, 800.0]:.1e} at ratio '
              f'{ratio(N_INC * np.sin(np.radians(44.5)), g):.2f} for L 100 .. 800, and by '
              f'{hats[50.0, 100.0]:.1e} .. {hats[50.0, 800.0]:.1e} at ratio '
              f'{ratio(N_INC * np.sin(np.radians(50.0)), g):.2f}: an edge seeds a growing lateral '
              f'wave at any angle.')


if __name__ == '__main__':
    coarse_rows, coarse_crossings = flips()
    fine = fine_grid()
    beam_ = beam(coarse_rows)
    margin_ = margin(beam_[0])
    far = far_flips(coarse_rows, fine['max'])
    hats = top_hats(beam_[0])
    firing()
    conclusions(fine, coarse_crossings, beam_, margin_, far, hats)
