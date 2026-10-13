"""Figures for appendices A and B on choosing kz at a complex kx, and the numbers their pages quote.

Appendix A (learn/complex-kx-roots.md) tells the story in a lossless medium: figures a01..a10.
Appendix B (learn/complex-kx-lossy-gain.md) extends it to absorbing and gain media: b01..b04.

A port of the figure part of the lab script ``_complex_kx_root_figs.py`` (next to the lab notes
``complex_kx_root_choice.ko.md`` and ``complex_kx_root_choice_lossy_gain.ko.md``), restyled for the
guide. a01..a10 say what the lab's ckr01..ckr10 say; b01..b04 say what ckr13, ckr14, ckr11, ckr12 say.

The four root rules (path D, Wavebit's rule before 2026-10-02; path E; Maystre's cut, Wavebit's
rule since; fmmax) are implemented here in plain numpy, so the figures keep making the same claim
whatever the installed solver does later. Nothing here imports wavebit or scipy. The solver-backed
numbers (the Wavebit gain-substrate R, the grating table) are recorded by
tools/measure_complex_kx_roots.py; the lab note's audit re-walk is not reproduced here.

Before drawing, six things are checked. If any disagrees, nothing is drawn:
  - the closed forms for paths D and E equal a slow step-by-step walk;
  - each rule's branch-cut formula equals where the rule really jumps on a grid, in both
    directions, for eps = 1 and eps = 4 + 3i;
  - the four rules agree wherever Im(kz^2) > 0;
  - in a lossless dielectric path D is the principal root (the continuity fix never switches on);
  - for a04 (c), the growing outgoing root at kx = 1.5 + 0.3i: Re k is perpendicular to Im k, the
    intensity does not change along Re k, and a point at height z has the surface intensity of the
    point its ray starts from;
  - for b01, the phase-angle kx really makes the phase direction theta, the formal value at
    theta +- 1e-12i is n sin(theta), and the two rules coincide in a lossless medium.

    conda run -n fieldwave python tools/draw_complex_kx_roots.py
    conda run -n fieldwave python tools/draw_complex_kx_roots.py --png DIR   # PNG previews, DIR only

Units: every wavenumber is divided by k0 = 2 pi / lambda. Time convention exp(-i omega t).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap           # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Polygon, Rectangle  # noqa: E402
import numpy as np                                               # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/docs/learning/ko'
PNG_DPI = 110

INK, MUTED, HAIR = '#182b3e', '#546779', '#b8c6ce'
BLUE, TEAL, ORANGE, PURPLE, RED = '#2d60a1', '#117c77', '#b56626', '#745e97', '#b33948'
plt.rcParams.update({
    'font.family': ['Apple SD Gothic Neo', 'DejaVu Sans'],
    'font.size': 12, 'svg.fonttype': 'none', 'axes.unicode_minus': False,
    'svg.hashsalt': 'wavebit-guide-appendix-a-complex-kx',
    'text.color': INK, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
    'axes.titlesize': 14, 'axes.titlepad': 9, 'axes.labelsize': 12,
    'axes.edgecolor': HAIR, 'axes.linewidth': 0.9,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11.5,
})
QUAD = {1: TEAL, 2: BLUE, 3: MUTED, 4: ORANGE}
QUAD_TEXT = {1: '나감 · 줄어듦', 2: '들어옴 · 줄어듦', 3: '들어옴 · 커짐', 4: '나감 · 커짐'}
RULE_COLOR = {'D': RED, 'E': PURPLE, 'M': BLUE, 'F': MUTED}
RULE_NAME = {'D': '경로 D (2026년 10월 2일 전의 Wavebit)', 'E': '경로 E', 'M': 'Maystre (Wavebit · RETICOLO · S4)',
             'F': 'fmmax'}
RE_KZ = 'Re(kz)   +이면 구조에서 멀어지며 나아감'
IM_KZ = 'Im(kz)   +이면 멀어질수록 줄어듦'
REPORT = []            # the numbers the page quotes, printed after the figures are written


def report(line=''):
    REPORT.append(line)


# ============================================================================ the four rules
FLAT = 1e-12      # the principal root's tie-break (Wavebit's half-space `_FLAT` before 2026-10-02):
                  # a real part below this counts as 0 and Im(kz) > 0 wins


def principal(kz2):
    """Principal root: Re(kz) > 0, and Im(kz) > 0 when the real part is 0. `boundary_kz`'s first step before
    2026-10-02."""
    v = np.sqrt(np.asarray(kz2, dtype=complex))
    flip = np.where(np.abs(v.real) > FLAT, v.real < 0, v.imag < 0)
    return np.where(flip, -v, v)


def rule_D(eps, kx):
    """Path D: continue straight from normal incidence kx = 0. The closed form of the step walk.

    In the kz^2 plane the path is the segment eps -> eps - kx^2. The principal root breaks only on
    the negative real axis, so flip only when the segment crosses it (Wavebit's `_off_principal_branch`, removed
    on 2026-10-02 with the rule).
    """
    kx = np.asarray(kx, dtype=complex)
    eps = np.asarray(eps, dtype=complex) + 0 * kx
    kp2 = kx * kx
    v = principal(eps - kp2)
    down = kp2.imag > 0
    t = eps.imag / np.where(down, kp2.imag, 1.0)
    flip = down & (t < 1) & (eps.real - t * kp2.real < 0)
    return np.where(flip, -v, v)


def rule_E(eps, kx):
    """Path E: start at the real kx with the same real part (the nearest real case), raise Im only.

    In the kz^2 plane the path is eps - (a + i s b)^2, s: 0 -> 1. Its imaginary part eps_i - 2abs is
    linear in s, so it crosses the real axis at most once; flip when that crossing is negative.
    """
    kx = np.asarray(kx, dtype=complex)
    eps = np.asarray(eps, dtype=complex) + 0 * kx
    a, b = kx.real, kx.imag
    v = principal(eps - kx * kx)
    ab = a * b
    down = ab > 0
    s = eps.imag / np.where(down, 2 * ab, 1.0)
    flip = down & (s < 1) & (eps.real - a * a + (s * b) ** 2 < 0)
    return np.where(flip, -v, v)


def rule_M(eps, kx):
    """Maystre's cut, Re(kz) + Im(kz) > 0: Wavebit's `boundary_kz` since 2026-10-02; RETICOLO's `retsqrt(x, -1)`
    default; S4 gives the same."""
    v = np.sqrt(np.asarray(eps, dtype=complex) - np.asarray(kx, dtype=complex) ** 2)
    return np.where(v.real + v.imag < 0, -v, v)


def rule_F(eps, kx):
    """fmmax 1.7.1 `_select_eigenvalues_sign`: Im(kz) > 0."""
    v = np.sqrt(np.asarray(eps, dtype=complex) - np.asarray(kx, dtype=complex) ** 2)
    return np.where(v.imag < 0, -v, v)


RULES = {'D': rule_D, 'E': rule_E, 'M': rule_M, 'F': rule_F}


def track(eps, path, start):
    """Step walk: at each step keep the root nearer the previous value. Returns kz along the path."""
    out = np.empty(len(path), dtype=complex)
    root = out[0] = complex(start)
    for i, k in enumerate(path[1:], 1):
        v = complex(np.sqrt(complex(eps) - complex(k) ** 2))
        root = out[i] = v if abs(v - root) <= abs(-v - root) else -v
    return out


def walk_D(eps, kx, n=4000):
    return track(eps, np.linspace(0, 1, n + 1) * kx, principal(eps))[-1]


def walk_E(eps, kx, n=4000):
    a = complex(kx.real)
    return track(eps, a + 1j * np.linspace(0, 1, n + 1) * kx.imag, principal(eps - a * a))[-1]


def cut_curves(eps, key, reach=7.0, m=20000):
    """Branch cut of rule `key`: two curves from the branch points +-n out to infinity.

    'P' is the line of the principal root before the continuity flip.
    """
    eps = complex(eps)
    n = complex(np.sqrt(eps))
    s = np.linspace(0, 1, m)
    if key == 'D':                     # ray from the origin through the branch point, outward
        c = n * (1 + s * reach)
        return [c, -c]
    if key == 'E':                     # straight up from +n, straight down from -n
        c = n.real + 1j * (n.imag + s * reach)
        return [c, -c]
    u = s * reach ** 2
    if key == 'M':                     # kz on the -45 degree line: kz^2 negative imaginary, kx^2 = eps + iu
        c = np.sqrt(eps + 1j * u)
    elif key == 'F':                   # kz real: kz^2 >= 0, kx^2 = eps - u
        # Near u = Re(eps) kx grazes 0 and the square root squashes the samples (a right-angle corner
        # when lossless); sample that stretch densely so the curve reaches the corner.
        u = np.union1d(u, eps.real + np.linspace(-0.02, 0.02, m + 1))
        c = np.sqrt(eps - u + 0j)
    elif key == 'P':                   # kz imaginary: kz^2 <= 0, kx^2 = eps + u
        c = np.sqrt(eps + u + 0j)
    return [c, -c]


# ============================================================================ checks before drawing
def sample_media(rng, N):
    lossless = rng.random(N) < 0.5
    eps = rng.uniform(1.0, 6.0, N) + 1j * np.where(lossless, 0.0, rng.uniform(0.01, 3.0, N))
    return eps, rng.uniform(0.0, 3.0, N) + 0j


def check_closed_forms(rng):
    """Closed forms D and E against the slow walk. Samples whose path nears a branch point are skipped.

    Returns the number of samples used and how many of them are lossless.
    """
    def seg_dist(p, a, b):
        ab = b - a
        t = np.clip(((p - a) * np.conj(ab)).real / max(abs(ab) ** 2, 1e-300), 0, 1)
        return abs(p - (a + t * ab))
    N, same, used, lossless = 600, {'D': 0, 'E': 0}, 0, 0
    eps_all, kx_all = sample_media(rng, N)
    kx_all = kx_all + 1j * rng.uniform(-0.5, 0.5, N)
    for e, k in zip(eps_all, kx_all):
        n = complex(np.sqrt(e))
        if min(seg_dist(q, 0, k) for q in (n, -n)) < 1e-2 or \
           min(seg_dist(q, complex(k.real), k) for q in (n, -n)) < 1e-2:
            continue
        used += 1
        lossless += e.imag == 0
        same['D'] += abs(complex(rule_D(e, k)) - walk_D(e, k)) < 1e-9
        same['E'] += abs(complex(rule_E(e, k)) - walk_E(e, k)) < 1e-9
    assert same['D'] == used and same['E'] == used, (same, used)
    return used, int(lossless)


def check_lossless_is_principal(rng):
    """In a lossless dielectric (eps > 0) path D is the principal root: the continuity fix never fires.

    In the kz^2 plane path D is the segment from the positive real eps to eps - kx^2. When
    Im(kx^2) != 0 its imaginary part -t Im(kx^2) vanishes only at t = 0, so the segment meets the
    real axis only at its start, on the positive side; the principal root breaks only on the negative
    side, so nothing flips. When Im(kx^2) = 0 the segment lies on the real axis and, for a real kx
    beyond n, steps on the branch point (the tie of appendix A). Checked on a grid and random samples.
    """
    points = 0
    for eps in (1.0, 2.25, 4.0):
        K, _ = grid((-3.0, 3.0, -1.5, 1.5))
        K = np.concatenate([K.ravel(), rng.uniform(0, 3, 4000) + 1j * rng.uniform(-0.5, 0.5, 4000),
                            np.linspace(0, 3, 601) + 0j])            # real kx too (the tie rule settles the tie)
        assert np.array_equal(rule_D(eps, K), principal(eps - K * K)), eps
        points += K.size
    return points


def grid(extent, shape=(720, 380)):
    x = np.linspace(extent[0], extent[1], shape[0])
    y = np.linspace(extent[2], extent[3], shape[1])
    dx, dy = x[1] - x[0], y[1] - y[0]
    x, y = x + 0.37 * dx, y + 0.29 * dy          # keep every sample off the axes
    return x[None, :] + 1j * y[:, None], max(dx, dy)


def nearest_distance(points, curve, chunk=32):
    """Distance from each point to its nearest curve sample, in chunks (plain numpy, no k-d tree)."""
    out = np.empty(len(points))
    for i in range(0, len(points), chunk):
        out[i:i + chunk] = np.abs(points[i:i + chunk, None] - curve[None, :]).min(axis=1)
    return out


def check_cuts(eps, keys, extent):
    """Find the jumps on a grid and hold them against the formulas: every jump lies on a formula
    curve, and every formula point really jumps when nudged to either side."""
    K, h = grid(extent)
    counts = {}
    for key in keys:
        Z = RULES[key](eps, K)
        J = []
        for a, b, ka, kb in ((Z[:, :-1], Z[:, 1:], K[:, :-1], K[:, 1:]),
                             (Z[:-1, :], Z[1:, :], K[:-1, :], K[1:, :])):
            jump = (a * np.conj(b)).real < 0             # neighbours sit nearer opposite roots
            J.append(((ka + kb) / 2)[jump])
        J = np.concatenate(J)
        curve = np.concatenate(cut_curves(eps, key))
        d = nearest_distance(J, curve)
        assert d.max() < 1.5 * h, (key, eps, d.max() / h)
        # the other direction: nudge points of the formula sideways, the root must really swap
        n = complex(np.sqrt(complex(eps)))
        tested = 0
        for c in cut_curves(eps, key, m=4000):
            inside = (c.real > extent[0]) & (c.real < extent[1]) & (c.imag > extent[2]) & (c.imag < extent[3])
            idx = np.flatnonzero(inside & (np.minimum(abs(c - n), abs(c + n)) > 0.05))[::40]
            for i in idx:
                if i == 0 or i == len(c) - 1:
                    continue
                tangent = c[i + 1] - c[i - 1]
                normal = 1j * tangent / abs(tangent)
                a = complex(RULES[key](eps, c[i] + 1e-6 * normal))
                b = complex(RULES[key](eps, c[i] - 1e-6 * normal))
                assert (a * np.conj(b)).real < 0, (key, eps, c[i])
                tested += 1
        counts[key] = (len(J), tested)
    # where Im(kz^2) > 0 all four rules agree
    kz2 = eps - K * K
    agree = kz2.imag > 1e-9
    ref = rule_D(eps, K)[agree]
    for key in 'EMF':
        assert np.max(np.abs(RULES[key](eps, K)[agree] - ref)) < 1e-12, key
    return counts


# Panel (c) of a04: the outgoing root of (b) in real space. x runs along the surface, z is the distance
# from the structure, both in units of 1/k0. The rays start on the surface every RAY_STEP in x, which is
# how far a ray runs in x per unit of z, so they cross the line x = X0 at z = 0, 1, 2 and the two marked
# points z = 0.5, 1.5 fall halfway between them.
GROW_KX = 1.5 + 0.3j
GROW_WINDOW = (-5.0, 3.0, 0.0, 2.6)          # x from, x to, z from, z to
GROW_X0, GROW_MARKS = 1.5, (0.5, 1.5)
GROW_STRIP = 0.42                            # depth of the structure strip drawn under z = 0
A04C_TEXT = ('(c)는 (b)의 나가는 근 0.393 − 1.146i로 푼 장의 세기를 x–z 평면에 그린 것이다. 가로는 표면을 '
             '따라가는 k0·x, 세로는 구조에서 떨어진 거리 k0·z이고, 색이 진할수록 세다. 표면의 세기는 왼쪽일수록 '
             '세다(∝ exp(−0.6x)). 표면에서 비스듬히 뻗는 광선은 같은 세기의 선과 나란해서, 광선을 따라서는 '
             '세기가 일정하다. x = 1.5의 점선을 따라 위로 가면 세기가 커진다. z = 0.5와 1.5의 점은 표면의 '
             'x = −0.41과 −4.23에서 출발한 광선 위에 있어 그 점과 세기가 같다. 멀어질수록 더 왼쪽의 센 곳에서 '
             '출발한 광선을 만나는 것이다.')


def plane_wave(kx, kz, x, z):
    """The field exp(i(kx x + kz z)) above the structure."""
    return np.exp(1j * (kx * x + kz * z))


def check_growing_root():
    """Panel (c) before drawing: the outgoing root of kx = 1.5 + 0.3i, read in real space.

    (1) Re k is perpendicular to Im k; (2) on the drawn window, log I does not change along Re k;
    (3) the marked points on x = X0 have the surface intensity of the points their rays start from.
    """
    kx = GROW_KX
    kz = complex(principal(1 - kx * kx))
    assert kz.real > 0 > kz.imag, kz                              # outgoing and growing, as in (b)
    ortho = kx.real * kx.imag + kz.real * kz.imag
    assert abs(ortho) < 1e-12, ortho
    x0, x1, z0, z1 = GROW_WINDOW
    x, z = np.linspace(x0, x1, 161), np.linspace(z0, z1, 53)
    X, Z = np.meshgrid(x, z)
    log_i = np.log(np.abs(plane_wave(kx, kz, X, Z)) ** 2)
    assert np.max(np.abs(log_i + 2 * (kx.imag * X + kz.imag * Z))) < 1e-12   # the closed form the map uses
    d_z, d_x = np.gradient(log_i, z, x)
    along = np.max(np.abs(d_x * kx.real + d_z * kz.real)) / abs(complex(kx.real, kz.real))
    assert along < 1e-10, along
    run = kx.real / kz.real                                       # x a ray covers per unit of z
    same = max(abs(abs(plane_wave(kx, kz, GROW_X0, zm)) ** 2
                   / abs(plane_wave(kx, kz, GROW_X0 - zm * run, 0.0)) ** 2 - 1) for zm in GROW_MARKS)
    assert same < 1e-10, same
    growth, via = 2 * abs(kz.imag), 2 * kx.imag * run
    assert abs(growth - via) < 1e-12, (growth, via)
    return dict(kz=kz, run=run, growth=growth, via=via, ortho=abs(ortho), along=along, same=same)


# ============================================================================ drawing helpers
def blend(hex_colour, alpha):
    rgb = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(alpha * rgb + (1 - alpha))


def quadrant(z):
    re, im = np.real(z), np.imag(z)
    return np.where(re >= 0, np.where(im >= 0, 1, 4), np.where(im >= 0, 2, 3))


def style(ax, frame=True, grid_on=True):
    for side in ('left', 'bottom', 'top', 'right'):
        ax.spines[side].set_color(HAIR)
    if not frame:
        ax.spines[['top', 'right']].set_visible(False)
    if grid_on:
        ax.grid(alpha=.16)


def kz_plane(ax, lim, shade=True, qtext=True, labels=True):
    if shade:
        for q, (x0, y0) in {1: (0, 0), 2: (-lim, 0), 3: (-lim, -lim), 4: (0, -lim)}.items():
            ax.add_patch(Rectangle((x0, y0), lim, lim, color=blend(QUAD[q], 0.11), lw=0, zorder=0))
    if qtext:
        pad = 0.035 * lim
        for q, (x, y, ha, va) in {1: (lim - pad, lim - pad, 'right', 'top'),
                                  2: (-lim + pad, lim - pad, 'left', 'top'),
                                  3: (-lim + pad, -lim + pad, 'left', 'bottom'),
                                  4: (lim - pad, -lim + pad, 'right', 'bottom')}.items():
            ax.text(x, y, f'{q}사분면\n{QUAD_TEXT[q]}', ha=ha, va=va, fontsize=11,
                    color=QUAD[q], linespacing=1.25, zorder=1)
    ax.axhline(0, color=HAIR, lw=0.9, zorder=1)
    ax.axvline(0, color=HAIR, lw=0.9, zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    style(ax)
    if labels:
        ax.set_xlabel(RE_KZ)
        ax.set_ylabel(IM_KZ)


def kx_plane(ax, xlim, ylim, points=(1.0,), labels=True, equal=True):
    ax.axhline(0, color=HAIR, lw=0.9, zorder=1)
    ax.axvline(0, color=HAIR, lw=0.9, zorder=1)
    for p in points:
        p = complex(p)
        ax.plot(p.real, p.imag, 'o', ms=9, color=INK, mec='white', mew=1.3, zorder=6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if equal:
        ax.set_aspect('equal')
    style(ax)
    if labels:
        ax.set_xlabel('Re(kx)')
        ax.set_ylabel('Im(kx)')


def arrow_on(ax, z, frac, colour, ms=15, zorder=5):
    i = int(frac * (len(z) - 1))
    j = min(i + max(2, len(z) // 150), len(z) - 1)
    i = max(0, min(i, j - 1))
    ax.annotate('', xy=(z[j].real, z[j].imag), xytext=(z[i].real, z[i].imag), zorder=zorder,
                arrowprops=dict(arrowstyle='-|>', color=colour, lw=1.4, mutation_scale=ms,
                                shrinkA=0, shrinkB=0))


def line(ax, z, colour, lw=2.4, ls='-', zorder=4, alpha=1.0, **kw):
    return ax.plot(np.real(z), np.imag(z), color=colour, lw=lw, ls=ls, zorder=zorder, alpha=alpha,
                   solid_capstyle='round', **kw)


def dot(ax, z, colour, filled=True, ms=9, zorder=7):
    z = complex(z)
    ax.plot(z.real, z.imag, 'o', ms=ms, zorder=zorder, color=colour,
            mfc=colour if filled else 'white', mec=colour, mew=2.0)


def cz(z, digits=3):
    """A complex number in the page's notation: 0.393 − 1.146i."""
    z = complex(z)
    re, im = z.real, z.imag
    f = f'{{:.{digits}f}}'
    if abs(re) < 0.5 * 10 ** -digits:
        return (f.format(im) if im >= 0 else '−' + f.format(-im)) + 'i'
    sign = '+' if im >= 0 else '−'
    head = f.format(re) if re >= 0 else '−' + f.format(-re)
    return f'{head} {sign} {f.format(abs(im))}i'


LABEL_BOX = dict(fc='white', ec='none', pad=1.5)     # hides an axis line that runs behind a label


def below_axes(fig, inches=0.62):
    """Figure y just under the lowest axes' tick labels and x label."""
    return min(ax.get_position().y0 for ax in fig.axes if ax.axison) - inches / fig.get_figheight()


def note(fig, text, inches=0.62):
    fig.text(0.5, below_axes(fig, inches), text, ha='center', va='top', fontsize=12, color=MUTED,
             linespacing=1.5)


def save(fig, name, title, description, png_dir):
    fig.savefig(OUT / f'{name}.svg', format='svg',
                metadata={'Title': title, 'Description': description, 'Date': None},
                bbox_inches='tight', pad_inches=0.12, facecolor='white')
    if png_dir:
        fig.savefig(Path(png_dir) / f'{name}.png', dpi=PNG_DPI, bbox_inches='tight',
                    pad_inches=0.12, facecolor='white')
    plt.close(fig)


# ============================================================================ figures
TARGET = 1.5 + 0.3j        # the destination in the figures; Im is large to be visible (0.05, 1e-9 behave alike)


def fig_a01(png):
    fig = plt.figure(figsize=(11.2, 5.7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1], wspace=0.22, hspace=0.95, left=0.07, right=0.99)
    ax = fig.add_subplot(gs[:, 0])
    kz_plane(ax, 1.45, shade=False, qtext=False)
    ax.set_xlabel('Re(kz)   +이면 멀어지며 나아감')
    a, b = np.sqrt(1 - 0.25), 1j * np.sqrt(1.5 ** 2 - 1)
    for z in (a, b):
        ax.plot([z.real, -z.real], [z.imag, -z.imag], ':', color=MUTED, lw=1.5, zorder=2)
    dot(ax, a, TEAL)
    dot(ax, -a, RED, filled=False)
    dot(ax, b, TEAL)
    dot(ax, -b, RED, filled=False)
    ax.text(a, 0.13, f'+{a:.3f}\n(kx = 0.5)', ha='center', va='bottom', fontsize=11.5, color=TEAL)
    ax.text(-a, 0.13, f'−{a:.3f}', ha='center', va='bottom', fontsize=11.5, color=RED)
    ax.text(0.13, b.imag, f'+{b.imag:.3f}i\n(kx = 1.5)', ha='left', va='center', fontsize=11.5, color=TEAL)
    ax.text(0.13, -b.imag, f'−{b.imag:.3f}i', ha='left', va='center', fontsize=11.5, color=RED)
    ax.text(-1.38, -1.38, '점선: 두 근은 언제나\n원점 대칭 (kz와 −kz)', fontsize=11, color=MUTED,
            va='bottom', linespacing=1.3)
    ax.set_title('kz 평면: 같은 kx에 근이 둘', loc='left')

    d = np.linspace(0, 5.0, 500)
    cases = [(0, 1, +a, '+0.866 : 멀어지며 나아간다', TEAL),
             (0, 2, -a, '−0.866 : 구조로 다가온다', RED),
             (1, 1, +b, '+1.118i : 멀어질수록 줄어든다', TEAL),
             (1, 2, -b, '−1.118i : 멀어질수록 커진다', RED)]
    for r, c, kz, title, colour in cases:
        axm = fig.add_subplot(gs[r, c])
        style(axm, frame=False)
        if kz.imag == 0:
            k = kz.real                                  # Re e^{i(k d − ωt)} at ωt = 0 and 0.9
            axm.plot(d, np.cos(k * d), color=colour, lw=2.2)
            axm.plot(d, np.cos(k * d - 0.9), color=colour, lw=1.4, alpha=0.35)
            trough0 = np.pi / abs(k)                     # where one trough went a moment later
            trough1 = (np.pi + 0.9) / k if k > 0 else (np.pi - 0.9) / abs(k)
            axm.annotate('', (trough1, -1.3), (trough0, -1.3),
                         arrowprops=dict(arrowstyle='-|>', color=colour, lw=1.5, mutation_scale=13))
            axm.set_ylim(-1.5, 1.75)
            axm.text(4.95, 1.72, '흐린 선: 조금 뒤', ha='right', va='top', fontsize=11, color=MUTED)
        else:
            axm.plot(d, np.exp(-kz.imag * d), color=colour, lw=2.2)
            axm.set_ylim(0, 3.2)
        axm.set_title(title, fontsize=13, color=colour, loc='left')
        axm.set_xlim(0, 5.0)
        axm.set_yticks([])
        axm.spines['left'].set_visible(False)
        axm.set_xlabel('구조에서 떨어진 거리 k0·d', fontsize=11)
    save(fig, 'a01-two-roots', '같은 kx에 kz가 둘',
         '공기(ε = 1)에서 kx = 0.5이면 kz = ±0.866(실수), kx = 1.5이면 kz = ±1.118i(순허수). '
         '두 근은 원점 대칭이고, 작은 그래프는 각 근이 구조에서 멀어지는 거리에 따라 어떻게 나아가거나 '
         '줄어들고 커지는지 보인다.', png)
    return a, b


def fig_a02(png):
    """The square root halves the angle: in a lossless medium a real kx puts kz^2 on the real axis (0 or 180
    degrees), and its roots on the boundary of quadrant 1 (0 or 90 degrees) and of quadrant 3."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 5.8))
    fig.subplots_adjust(left=0.07, right=0.99, wspace=0.62)
    lim1 = 1.75
    a1.axhline(0, color=HAIR, lw=0.9, zorder=1)
    a1.axvline(0, color=HAIR, lw=0.9, zorder=1)
    a1.plot([0, lim1], [0, 0], color=TEAL, lw=4.0, zorder=2, solid_capstyle='butt')
    a1.plot([-lim1, 0], [0, 0], color=TEAL, lw=4.0, zorder=2, solid_capstyle='butt')
    a1.set_xlim(-lim1, lim1)
    a1.set_ylim(-lim1, lim1)
    a1.set_aspect('equal')
    style(a1)
    a1.text(1.0, 0.13, '전파 |kx| < n\nkz² > 0 (0°)', ha='center', va='bottom', fontsize=11.5, color=INK,
            linespacing=1.3)
    a1.text(-1.0, -0.13, '소멸 |kx| > n\nkz² < 0 (180°)', ha='center', va='top', fontsize=11.5, color=INK,
            linespacing=1.3)
    t = np.radians(np.linspace(0, 180, 120))
    line(a1, 0.5 * np.exp(1j * t), MUTED, lw=1.1)
    a1.text(0.0, 0.58, '180°', ha='center', va='bottom', fontsize=11.5, color=MUTED)
    examples = [(0.75 + 0j, 'o', 'kx = 0.5\nkz² = 0.75', (0.0, -0.22, 'center', 'top')),
                (-1.25 + 0j, 's', 'kx = 1.5\nkz² = −1.25', (0.0, 0.2, 'center', 'bottom'))]
    for z, marker, text, (dx, dy, ha, va) in examples:
        a1.plot(z.real, z.imag, marker, ms=9, color=INK, zorder=6)
        a1.text(z.real + dx, z.imag + dy, text, ha=ha, va=va, fontsize=11, color=INK, linespacing=1.3,
                bbox=LABEL_BOX, zorder=7)
    a1.text(-lim1 + 0.08, lim1 - 0.08, '공기 ε = 1\nIm(kz²) = Im(ε) = 0', ha='left', va='top', fontsize=11.5,
            color=INK, linespacing=1.3)
    a1.set_xlabel('Re(kz²)')
    a1.set_ylabel('Im(kz²)')
    a1.set_title('(a) kz² 평면: 실수 kx이면 실수축 위', loc='left')

    lim2 = 1.75
    kz_plane(a2, lim2)
    a2.plot([0, lim2], [0, 0], color=TEAL, lw=4.0, zorder=2, solid_capstyle='butt')
    a2.plot([0, 0], [0, lim2], color=TEAL, lw=4.0, zorder=2, solid_capstyle='butt')
    t = np.radians(np.linspace(0, 90, 60))
    line(a2, 0.4 * np.exp(1j * t), MUTED, lw=1.1)
    a2.text(0.33, 0.33, '90°', fontsize=11.5, color=MUTED)
    for z, marker, _, _ in examples:
        root = complex(principal(z))
        a2.plot(root.real, root.imag, marker, ms=9, color=INK, zorder=6)
        a2.plot(-root.real, -root.imag, marker, ms=9, color=INK, mfc='white', mew=1.6, zorder=6)
    a2.text(0.866, -0.14, '0.866 (0°)', ha='center', va='top', fontsize=11, color=INK)
    a2.text(-0.866, 0.14, '−0.866', ha='center', va='bottom', fontsize=11, color=INK)
    a2.text(-0.12, 1.118, '1.118i (90°)', ha='right', va='center', fontsize=11, color=INK)
    a2.text(0.12, -1.118, '−1.118i', ha='left', va='center', fontsize=11, color=INK)
    a2.set_title('(b) kz 평면: 각도가 절반 → 0° 또는 90°', loc='left')
    a2.text(0.5, -0.19, '꽉 찬 점: 고르는 근 (1사분면의 경계)    빈 점: 그 반대편 −kz', transform=a2.transAxes,
            ha='center', va='top', fontsize=11.5, color=INK)
    p1, p2 = a1.get_position(), a2.get_position()
    y = p1.y0 + 0.62 * p1.height
    x0, x1 = p1.x1 + 0.014, p2.x0 - 0.085
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle='-|>',
                                   mutation_scale=18, color=INK, lw=1.8))
    xm = (x0 + x1) / 2
    fig.text(xm, y + 0.03, '제곱근', ha='center', va='bottom', fontsize=12.5, color=INK)
    fig.text(xm, y - 0.03, '각도 ÷ 2', ha='center', va='top', fontsize=12.5, color=INK)
    save(fig, 'a02-half-angle', '제곱근은 각도를 절반으로',
         '무손실 매질에서 실수 kx이면 kz²은 실수축 위(전파 0°, 소멸 180°)에 있고, 제곱근은 각도를 절반으로 '
         '만들어 고르는 근을 1사분면의 경계(0° 또는 90°)에, 반대 근 −kz를 그 맞은편에 둔다. '
         '예: 0.75 → 0.866, −1.25 → 1.118i.', png)
    return [complex(principal(z)) for z, *_ in examples]


def fig_a03(png):
    fig = plt.figure(figsize=(10.8, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.06, left=0.08, right=0.99)
    ax = fig.add_subplot(gs[0, 0])
    side = fig.add_subplot(gs[0, 1])
    side.axis('off')
    kz_plane(ax, 2.2)
    k = np.linspace(0, 2.0, 1200)
    air = principal(1 - k ** 2)
    line(ax, -air, INK, lw=1.7, ls=(0, (3, 3)), alpha=0.45)
    line(ax, air, INK, lw=2.8)
    marks = {0.0: (0.0, 0.07, 'center', 'bottom'), 0.5: (0.0, -0.09, 'center', 'top'),
             0.9: (0.0, 0.09, 'center', 'bottom'), 1.0: (-0.07, -0.09, 'right', 'top'),
             1.2: (-0.09, 0.0, 'right', 'center'), 1.5: (-0.09, 0.0, 'right', 'center'),
             2.0: (-0.09, 0.0, 'right', 'center')}
    for kv, (dx, dy, ha, va) in marks.items():
        z = complex(principal(1 - kv ** 2))
        dot(ax, z, INK, ms=6.5)
        ax.text(z.real + dx, z.imag + dy, f'kx = {kv:g}', fontsize=11, color=INK, ha=ha, va=va)
    ax.set_title('실수 kx: 고른 근은 1사분면의 경계를 따라간다', loc='left')

    side.legend(handles=[Line2D([], [], color=INK, lw=2.8, label='공기 ε = 1\n(kx 0 → 2)'),
                         Line2D([], [], color=INK, lw=1.7, ls=(0, (3, 3)), alpha=0.45, label='버린 근 −kz')],
                loc='upper left', bbox_to_anchor=(0.02, 0.97), frameon=False, fontsize=12,
                handlelength=2.6, labelspacing=1.0)
    notes = [('공기', INK, 'kx < 1: 실수축을 따라 1 → 0 (전파)\nkx > 1: 허수축을 타고 위로 (소멸)'),
             ('버린 근 −kz', MUTED, '그 맞은편: 음의 실수축과\n음의 허수축')]
    y = 0.62
    for head, colour, body in notes:
        side.text(0.05, y, head, transform=side.transAxes, fontsize=12.5, color=colour, va='top')
        side.text(0.05, y - 0.055, body, transform=side.transAxes, fontsize=11.5, color=INK,
                  va='top', linespacing=1.45)
        y -= 0.07 + 0.055 * (body.count('\n') + 1) + 0.03
    save(fig, 'a03-real-kx', '실수 kx에서 고른 근은 1사분면의 경계',
         '공기에서 kx를 0에서 2로 키우면 고른 근은 실수축을 따라 1에서 0으로 온 뒤(전파) 허수축을 타고 '
         '올라간다(소멸). 곧 1사분면의 경계를 따라간다. 버린 근 −kz는 그 맞은편.', png)


def growing_root_map(ax):
    """Panel (c): log10 of the intensity of (b)'s outgoing root over x-z, the rays, and the line x = X0."""
    kx = GROW_KX
    kz = complex(principal(1 - kx * kx))
    run = kx.real / kz.real
    x0, x1, z0, z1 = GROW_WINDOW
    # log I is linear in x and z, so a coarse grid already gives exact straight contours
    X, Z = np.meshgrid(np.linspace(x0, x1, 33), np.linspace(z0, z1, 14))
    log10_i = np.log10(np.abs(plane_wave(kx, kz, X, Z)) ** 2)
    cmap = LinearSegmentedColormap.from_list('a04-intensity', [blend(ORANGE, 0.03), ORANGE])
    fill = ax.contourf(X, Z, log10_i, levels=np.arange(-1.0, 4.01, 0.25), cmap=cmap, zorder=0)
    fill.set_edgecolor('face')                                  # no hairline seams between the bands
    fill.set_linewidth(0.4)
    ax.add_patch(Rectangle((x0, -GROW_STRIP), x1 - x0, GROW_STRIP, color=blend(MUTED, 0.16), lw=0, zorder=1))
    ax.axhline(0, color=INK, lw=1.2, zorder=3)
    for k in range(6):                                          # rays from x_s = X0 - k·run, clipped to the window
        xs = GROW_X0 - k * run
        t0, t1 = max(0.0, (x0 - xs) / kx.real), min((x1 - xs) / kx.real, z1 / kz.real)
        if t1 <= t0:
            continue
        t = np.linspace(t0, t1, 301)
        ray = xs + t * kx.real + 1j * t * kz.real
        line(ax, ray, INK, lw=1.1, zorder=4)
        arrow_on(ax, ray, 0.55, INK, ms=11)
    ax.plot([GROW_X0, GROW_X0], [0, z1], ls=(0, (4, 3)), color=INK, lw=1.4, zorder=5)
    ax.annotate('', (GROW_X0, z1 - 0.03), (GROW_X0, z1 - 0.45), zorder=6,
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.6, mutation_scale=15, shrinkA=0, shrinkB=0))
    for zm in GROW_MARKS:                                       # each mark, back along its ray to the surface
        foot = GROW_X0 - zm * run
        ax.plot([GROW_X0, foot], [zm, 0], ls=(0, (1.5, 2)), color=BLUE, lw=1.8, zorder=6)
        dot(ax, complex(GROW_X0, zm), BLUE, ms=8)
        dot(ax, complex(foot, 0), BLUE, filled=False, ms=8)
    ray2 = lambda x: (x - (GROW_X0 - 2 * run)) * kz.real / kx.real    # the ray that crosses x = X0 at z = 2
    ax.text(GROW_X0 - 0.12, ray2(GROW_X0 - 0.12) + 0.08, '위로 갈수록 셈:\n더 왼쪽에서 출발한 광선을 만난다',
            ha='right', va='bottom', fontsize=11.5, color=INK, linespacing=1.35, zorder=7)
    ax.text(-2.6, ray2(-2.0) + 0.34, '광선을 따라서는 세기가 일정', ha='center', va='bottom', fontsize=11.5,
            color=INK, zorder=7)
    ax.annotate('', (-2.0, ray2(-2.0) + 0.02), (-2.35, ray2(-2.0) + 0.29), zorder=7,
                arrowprops=dict(arrowstyle='->', color=INK, lw=1.0, shrinkA=0, shrinkB=0))
    ax.text(x0 + 0.1, -GROW_STRIP / 2, '표면의 세기: 왼쪽일수록 셈 (∝ exp(−0.6x))', ha='left', va='center',
            fontsize=11.5, color=INK, zorder=7)
    ax.text(x1 - 0.1, -GROW_STRIP / 2, '구조', ha='right', va='center', fontsize=11, color=MUTED, zorder=7)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-GROW_STRIP, z1)
    ax.set_yticks(np.arange(0, z1 + 1e-9, 0.5))
    ax.set_aspect('equal')
    ax.set_anchor('W')
    style(ax, grid_on=False)
    ax.set_xlabel('k0·x')
    ax.set_ylabel('k0·z (구조에서 떨어진 거리)')
    ax.set_title(f'(c) (b)의 나가는 근 {cz(kz)}: x–z 평면의 세기', loc='left')
    strip = GROW_STRIP / (z1 + GROW_STRIP)
    cax = ax.inset_axes([1.02, strip, 0.012, 1 - strip])
    bar = ax.figure.colorbar(fill, cax=cax, ticks=range(-1, 5))
    bar.outline.set_edgecolor(HAIR)
    bar.set_label('log₁₀ 세기', fontsize=11.5)
    cax.tick_params(labelsize=11, color=MUTED)
    return kz


def fig_a04(png):
    fig = plt.figure(figsize=(11.2, 10.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.41, 1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ref = 1j * np.sqrt(1.25)
    pairs = []
    for ax, kx, tag in zip(axes, (1.5 - 0.3j, 1.5 + 0.3j), 'ab'):
        kz_plane(ax, 1.6)
        v = complex(principal(1 - kx * kx))
        pairs.append(v)
        ax.plot([ref.real, -ref.real], [ref.imag, -ref.imag], 'x', color=MUTED, ms=8, mew=1.8, zorder=5)
        ax.plot([v.real, -v.real], [v.imag, -v.imag], ':', color=INK, lw=1.3, zorder=3)
        for z in (v, -v):
            q = int(quadrant(z))
            dot(ax, z, QUAD[q], ms=10)
            up = z.imag < 0                                    # label on the side away from the dotted line
            ax.text(z.real + (0.1 if z.real > 0 else -0.1), z.imag + (0.14 if up else -0.14), cz(z),
                    ha='left' if z.real > 0 else 'right', va='bottom' if up else 'top',
                    fontsize=11.5, color=QUAD[q])
        sign = '−' if kx.imag < 0 else '+'
        ax.set_title(f'({tag}) kx = 1.5 {sign} 0.3i', loc='left')
    axes[0].text(-0.1, ref.imag, 'kx = 1.5일 때', ha='right', va='center', fontsize=11, color=MUTED)
    axes[1].text(0.1, ref.imag, 'kx = 1.5일 때', ha='left', va='center', fontsize=11, color=MUTED)
    box = dict(fc='white', ec=HAIR, lw=0.8, pad=5)
    axes[0].text(0.93, -0.42, '두 근이 1 · 3사분면\n한 근이 두 기준을\n다 만족한다',
                 ha='center', va='center', fontsize=11.5, color=INK, bbox=box, linespacing=1.35)
    axes[1].text(0.93, -0.42, '두 근이 2 · 4사분면\n나가는 근은 커지고\n줄어드는 근은 들어온다',
                 ha='center', va='center', fontsize=11.5, color=INK, bbox=box, linespacing=1.35)
    axc = fig.add_subplot(gs[1, :])
    growing_root_map(axc)
    fig.tight_layout(w_pad=2.4, h_pad=2.0)
    for ax in (*axes, axc):
        ax.apply_aspect()
    left, box = axes[0].get_position().x0, axc.get_position()      # (c) starts where (a) starts
    axc.set_position([left, box.y0, box.width, box.height])
    save(fig, 'a04-complex-pairs', '복소 kx에서 두 근의 자리',
         '공기에서 kx = 1.5 − 0.3i이면 두 근 ±(0.393 + 1.146i)가 1·3사분면에 있어 한 근이 두 기준을 다 '
         '만족한다. kx = 1.5 + 0.3i이면 두 근 ±(0.393 − 1.146i)가 2·4사분면에 있어 나가는 근은 커지고 '
         '줄어드는 근은 들어온다. 회색 ×는 kx = 1.5일 때의 두 근 ±1.118i. ' + A04C_TEXT, png)
    return pairs


def fig_a05(png):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.9), gridspec_kw=dict(width_ratios=[1.3, 1]))
    kx_plane(a1, (-0.1, 1.7), (-0.55, 0.62))
    phi = np.linspace(np.pi, 3 * np.pi, 1500)
    loopA = 1 + 0.35 * np.exp(1j * phi)
    loopB = 0.3 + 0.2 * np.exp(1j * phi)
    line(a1, loopA, RED)
    line(a1, loopB, BLUE)
    for z, colour in ((loopA, RED), (loopB, BLUE)):
        arrow_on(a1, z, 0.28, colour)
        arrow_on(a1, z, 0.78, colour)
        dot(a1, z[0], colour, ms=8)
    a1.text(1.0, 0.06, 'branch point\nkx = 1', ha='center', va='bottom', fontsize=11, color=INK,
            linespacing=1.25)
    a1.text(1.0, 0.4, '고리 A: branch point를 감싼다', ha='center', va='bottom', fontsize=12, color=RED)
    a1.text(0.3, 0.25, '고리 B: 안 감싼다', ha='center', va='bottom', fontsize=12, color=BLUE)
    a1.text(loopA[0].real - 0.03, -0.05, '출발', ha='right', va='top', fontsize=11, color=RED)
    a1.text(loopB[0].real - 0.03, -0.05, '출발', ha='right', va='top', fontsize=11, color=BLUE, bbox=LABEL_BOX,
            zorder=7)
    a1.set_title('kx 평면: 한 바퀴 도는 두 고리', loc='left')

    kz_plane(a2, 1.15, shade=False, qtext=False)
    a2.set_xlabel('Re(kz)')
    a2.set_ylabel('Im(kz)')
    tA = track(1.0, loopA, principal(1 - loopA[0] ** 2))
    tB = track(1.0, loopB, principal(1 - loopB[0] ** 2))
    line(a2, tA, RED)
    line(a2, tB, BLUE)
    arrow_on(a2, tA, 0.3, RED)
    arrow_on(a2, tA, 0.8, RED)
    dot(a2, tA[0], RED, ms=8)
    dot(a2, tA[-1], RED, filled=False, ms=10)
    dot(a2, tB[0], BLUE, ms=7)
    a2.text(tA[0].real, -0.1, f'출발 +{tA[0].real:.3f}', ha='center', va='top', fontsize=11, color=RED)
    a2.text(tA[-1].real, -0.1, f'도착 −{-tA[-1].real:.3f}', ha='center', va='top', fontsize=11, color=RED)
    a2.text(1.1, -0.3, '고리 B: 제자리', ha='right', va='top', fontsize=11, color=BLUE)
    a2.set_title('kz 평면: 따라간 근', loc='left')
    fig.tight_layout(w_pad=2.5)
    save(fig, 'a05-loop', 'branch point를 감싼 고리와 안 감싼 고리',
         '공기에서 kx가 branch point kx = 1을 감싸는 고리 A를 한 바퀴 돌면 따라간 근이 +0.760에서 −0.760 '
         '으로 바뀌고, 감싸지 않는 고리 B는 +0.995로 제자리에 돌아온다.', png)
    return tA[0], tA[-1], tB[0], tB[-1]


def branch_point_walkthrough():
    """Numbers for the branch-point section (air): the two candidates as a real kx walks onto the branch point,
    and loop A followed by quarter turns (the same loop as figure a05, sampled at exact quarter angles)."""
    approach = [(k, complex(principal(1.0 - k * k))) for k in (0.0, 0.9, 0.99, 1.0, 1.01, 1.5)]
    phi = np.linspace(np.pi, 3 * np.pi, 2001)
    loop = 1 + 0.35 * np.exp(1j * phi)
    kept = track(1.0, loop, principal(1 - loop[0] ** 2))
    def clean(z):                      # drop round-off below 1e-9 so a real number prints as real
        z = complex(z)
        return complex(0.0 if abs(z.real) < 1e-9 else z.real, 0.0 if abs(z.imag) < 1e-9 else z.imag)
    approach = [(k, clean(v)) for k, v in approach]
    quarters = [(clean(loop[i]), clean(principal(1 - loop[i] ** 2)), clean(kept[i]),
                 round(float(np.degrees(np.angle(clean(kept[i]))))) % 360) for i in (0, 500, 1000, 1500, 2000)]
    return approach, quarters


def fig_a06(png):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    kx_plane(a1, (-0.14, 1.95), (-0.45, 0.62))
    pD = np.linspace(0, 1, 1500) * TARGET
    pE = 1.5 + 1j * np.linspace(0, 1, 1500) * TARGET.imag
    line(a1, pD, RED)
    line(a1, pE, PURPLE)
    arrow_on(a1, pD, 0.55, RED)
    arrow_on(a1, pE, 0.6, PURPLE)
    dot(a1, 0j, RED, ms=8)
    dot(a1, 1.5 + 0j, PURPLE, ms=8)
    dot(a1, TARGET, INK, ms=8)
    a1.text(0.0, -0.08, '수직 입사\nkx = 0', ha='center', va='top', fontsize=11, color=RED, linespacing=1.25,
            bbox=LABEL_BOX, zorder=7)
    a1.text(1.5, -0.08, '가장 가까운\n실제 상황\nkx = 1.5', ha='center', va='top', fontsize=11, color=PURPLE,
            linespacing=1.25)
    a1.text(TARGET.real + 0.06, TARGET.imag + 0.03, '목적지\n1.5 + 0.3i', va='bottom', fontsize=11, color=INK,
            linespacing=1.25)
    a1.text(1.0, -0.08, 'branch\npoint', ha='center', va='top', fontsize=11, color=INK, linespacing=1.25)
    a1.text(0.55, 0.2, '경로 D', fontsize=12.5, color=RED)
    a1.text(1.57, 0.1, '경로 E', fontsize=12.5, color=PURPLE)
    a1.set_title('kx 평면: 출발점이 다른 두 길', loc='left')

    kz_plane(a2, 1.45, qtext=False)
    a2.set_xlabel('Re(kz)')
    a2.set_ylabel('Im(kz)')
    tD = track(1.0, pD, 1.0)
    tE = track(1.0, pE, principal(1 - 2.25))
    line(a2, tD, RED)
    line(a2, tE, PURPLE)
    arrow_on(a2, tD, 0.55, RED)
    arrow_on(a2, tE, 0.6, PURPLE)
    dot(a2, tD[0], RED, ms=7)
    dot(a2, tE[0], PURPLE, ms=7)
    dot(a2, tD[-1], RED, ms=10)
    dot(a2, tE[-1], PURPLE, ms=10)
    a2.text(tD[-1].real + 0.13, tD[-1].imag, f'D 도착\n{cz(tD[-1])}\n나감 · 커짐', fontsize=11,
            color=RED, va='center', linespacing=1.3)
    a2.text(tE[-1].real - 0.02, tE[-1].imag - 0.16, f'E 도착\n{cz(tE[-1])}\n들어옴 · 줄어듦', fontsize=11,
            color=PURPLE, ha='right', va='top', linespacing=1.3)
    a2.text(1.0, 0.08, 'D 출발 1', ha='center', va='bottom', fontsize=11, color=RED)
    a2.text(0.09, tE[0].imag - 0.04, 'E 출발 +1.118i', fontsize=11, color=PURPLE, va='top')
    a2.set_title('kz 평면: 도착점이 서로 반대', loc='left')
    fig.tight_layout(w_pad=2.5)
    save(fig, 'a06-paths-d-e', '출발점이 다른 두 길과 도착한 근',
         '목적지 kx = 1.5 + 0.3i로 가는 두 길. 경로 D는 수직 입사 kx = 0에서 출발해 branch point 위쪽을 '
         '지나 0.393 − 1.146i(나감 · 커짐)에, 경로 E는 가장 가까운 실제 상황 kx = 1.5에서 허수부만 올려 '
         '−0.393 + 1.146i(들어옴 · 줄어듦)에 도착한다. 두 답은 서로 반대 근.', png)
    return tD[-1], tE[-1]


def fig_a07(png):
    fig = plt.figure(figsize=(11.2, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.6, 1], width_ratios=[1, 1.6], hspace=0.42, wspace=0.26,
                          left=0.07, right=0.99)
    a1 = fig.add_subplot(gs[0, :])
    a2 = fig.add_subplot(gs[1, 0])
    a3 = fig.add_subplot(gs[1, 1])
    kx_plane(a1, (-0.17, 1.95), (-0.42, 0.42), equal=False)
    ends = {'위': (TARGET, ORANGE), '가운데': (1.5 + 0j, INK), '아래': (TARGET.conjugate(), TEAL)}
    paths = {k: np.linspace(0, 1, 1500) * z for k, (z, _) in ends.items()}
    for key, (z, colour) in ends.items():
        line(a1, paths[key], colour, lw=2.6 if key == '가운데' else 2.3)
        arrow_on(a1, paths[key], 0.82, colour)
        dot(a1, z, colour, ms=8)
    dot(a1, 0j, INK, ms=7)
    a1.text(1.54, 0.3, '위로 지남 (허수부 +)', va='center', fontsize=12, color=ORANGE)
    a1.text(1.54, 0.0, '밟고 지남 (허수부 0)', va='center', fontsize=12, color=INK, bbox=LABEL_BOX, zorder=7)
    a1.text(1.54, -0.3, '아래로 지남 (허수부 −)', va='center', fontsize=12, color=TEAL)
    a1.text(1.0, 0.035, 'branch point', ha='center', va='bottom', fontsize=11, color=INK)
    a1.text(-0.02, -0.04, 'kx = 0', ha='right', va='top', fontsize=11, color=INK)
    a1.set_title('(a) 경로 D 세 개: 목적지의 허수부만 다르다', loc='left')

    kz_plane(a2, 1.45, qtext=False)
    a2.set_xlabel('Re(kz)')
    a2.set_ylabel('Im(kz)')
    up = track(1.0, paths['위'], 1.0)
    dn = track(1.0, paths['아래'], 1.0)
    line(a2, up, ORANGE)
    line(a2, dn, TEAL)
    k = np.linspace(0, 1, 400)
    line(a2, np.sqrt(1 - k ** 2) + 0j, INK, lw=2.6)
    kk = np.linspace(1, 1.5, 300)
    line(a2, 1j * np.sqrt(kk ** 2 - 1), INK, lw=2.2, ls=(0, (4, 3)))
    line(a2, -1j * np.sqrt(kk ** 2 - 1), INK, lw=2.2, ls=(0, (4, 3)))
    dot(a2, up[-1], ORANGE, ms=9)
    dot(a2, dn[-1], TEAL, ms=9)
    dot(a2, 1j * np.sqrt(1.25), INK, filled=False, ms=9)
    dot(a2, -1j * np.sqrt(1.25), INK, filled=False, ms=9)
    a2.text(0.12, 0.12, '?', fontsize=20, color=INK)
    a2.text(up[-1].real + 0.1, up[-1].imag, cz(up[-1]), fontsize=11, color=ORANGE, va='center')
    a2.text(dn[-1].real + 0.1, dn[-1].imag, cz(dn[-1]), fontsize=11, color=TEAL, va='center')
    a2.text(-0.1, 1.118, '+1.118i', ha='right', va='center', fontsize=11, color=INK)
    a2.text(-0.1, -1.118, '−1.118i', ha='right', va='center', fontsize=11, color=INK)
    a2.set_title('(b) 따라간 근', loc='left')

    h = 1e-4
    prev = np.sqrt(1 - (1 - h) ** 2)
    cand = np.sqrt((1 + h) ** 2 - 1)
    kz_plane(a3, 0.03, shade=False, qtext=False, labels=False)
    a3.set_xlim(-0.048, 0.05)
    a3.set_xticks([-0.04, -0.02, 0, 0.02, 0.04])
    a3.set_yticks([-0.02, 0, 0.02])
    a3.set_xlabel('Re(kz)')
    a3.set_ylabel('Im(kz)')
    for s in (+1, -1):
        a3.plot([prev, 0], [0, s * cand], '--', color=MUTED, lw=1.4, zorder=3)
        dot(a3, 1j * s * cand, INK, filled=False, ms=10)
    dot(a3, prev + 0j, INK, ms=9)
    a3.plot(0, 0, 'x', color=INK, ms=9, mew=2.0, zorder=6)
    dist = abs(prev - 1j * cand)
    a3.text(prev + 0.003, 0, f'kx = 0.9999\nkz = {prev:.5f}', ha='left', va='center', fontsize=11, color=INK,
            linespacing=1.3, bbox=LABEL_BOX, zorder=7)
    a3.text(0, cand + 0.0033, f'kx = 1.0001의 후보\n+{cand:.5f}i', ha='center', va='bottom', fontsize=11,
            color=INK, linespacing=1.3, bbox=LABEL_BOX, zorder=7)
    a3.text(0, -cand - 0.0033, f'−{cand:.5f}i\nkx = 1.0001의 후보', ha='center', va='top', fontsize=11,
            color=INK, linespacing=1.3, bbox=LABEL_BOX, zorder=7)
    a3.text(0.0098, 0.0082, f'거리 {dist:.4f}', fontsize=11, color=MUTED, va='bottom')
    a3.text(0.0098, -0.0082, f'거리 {dist:.4f}', fontsize=11, color=MUTED, va='top')
    a3.text(-0.0035, 0, 'kx = 1 (branch point)\nkz = 0', fontsize=11, color=INK, ha='right', va='center',
            linespacing=1.3, bbox=LABEL_BOX, zorder=7)
    a3.set_title('(c) branch point 바로 옆 확대', loc='left')
    save(fig, 'a07-through-branch-point', '실수축을 밟고 지나는 경로 D',
         '공기에서 목적지 1.5 + 0.3i, 1.5, 1.5 − 0.3i로 가는 경로 D 세 개. 허수부가 +이면 branch point 위쪽으로 '
         '지나 0.393 − 1.146i에, −이면 아래쪽으로 지나 0.393 + 1.146i에 도착한다. 허수부 0이면 branch point를 '
         '밟는다: kx = 0.9999에서 kz = 0.01414, kx = 1.0001의 두 후보 ±0.01414i가 똑같이 0.0200 떨어져 있어 '
         '따라가기가 비긴다.', png)
    return up[-1], dn[-1], prev, cand, dist


def fig_a08(png):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.0))
    v = complex(principal(1 - TARGET ** 2))
    specs = [('D', '주근 = 경로 D (무손실 매질)\n2026년 10월 2일 전의 Wavebit\nRe(kz) > 0', (0, 1), lambda z: z.real > 0),
             ('M', 'Maystre\nWavebit · RETICOLO · S4\nRe(kz) + Im(kz) > 0', (1, -1), lambda z: z.real + z.imag > 0),
             ('F', 'fmmax\nIm(kz) > 0', (1, 0), lambda z: z.imag > 0)]
    lim = 1.45
    chosen = {}
    for ax, (key, title, direction, keep) in zip(axes, specs):
        kz_plane(ax, lim, shade=False, qtext=False, labels=False)
        colour = RULE_COLOR[key]
        dx, dy = direction                           # direction of the boundary line
        nx, ny = dy, -dx                             # normal pointing to the kept side
        if not keep(complex(nx, ny)):
            nx, ny = -nx, -ny
        big = 4 * lim
        poly = np.array([[-dx * big, -dy * big], [dx * big, dy * big],
                         [dx * big + nx * big, dy * big + ny * big], [-dx * big + nx * big, -dy * big + ny * big]])
        ax.add_patch(Polygon(poly, color=blend(colour, 0.13), lw=0, zorder=0))
        ax.plot([-dx * big, dx * big], [-dy * big, dy * big], color=colour, lw=2.4, zorder=2)
        kept, dropped = (v, -v) if keep(v) else (-v, v)
        chosen[key] = kept
        dot(ax, kept, colour, ms=11)
        dot(ax, dropped, colour, filled=False, ms=11)
        q = int(quadrant(kept))
        ax.text(0.965, 0.965, f'고른 근 (꽉 찬 점)\n{cz(kept)}\n{QUAD_TEXT[q]}', transform=ax.transAxes,
                ha='right', va='top', fontsize=11, color=colour, linespacing=1.3,
                bbox=dict(fc='white', ec='none', alpha=0.9, pad=2.5), zorder=8)
        ax.set_title(title, loc='left', color=colour, linespacing=1.3)
        ax.set_xlabel('Re(kz)')
        ax.set_ylabel('Im(kz)')
    fig.tight_layout(w_pad=1.6)
    note(fig, '색칠한 쪽 = 그 규칙이 근을 고르는 쪽.  진한 선 = 경계: 근이 이 선 위에 오면 두 근이 비기고,\n'
                   '선을 건너면 고른 근이 반대편으로 바뀐다 (튄다).  kx = 1.5 + 0.3i, 공기')
    save(fig, 'a08-half-planes', '반평면으로 한 번에 고르는 세 규칙',
         'kx = 1.5 + 0.3i의 두 근 ±(0.393 − 1.146i)에 세 규칙을 적용한다. 주근(Re(kz) > 0, 무손실 매질에서 '
         '경로 D, 2026년 10월 2일 전의 Wavebit)은 0.393 − 1.146i(나감 · 커짐), Maystre(Re + Im > 0, Wavebit · '
         'RETICOLO · S4)와 fmmax(Im(kz) > 0)는 −0.393 + 1.146i(들어옴 · 줄어듦)를 고른다. 경계선 위에서 두 근이 '
         '비기고, 선을 건너면 고른 근이 튄다.', png)
    return v, chosen


def cut_map(ax, eps, key, extent, title):
    K, _ = grid(extent, (560, 290))
    q = quadrant(RULES[key](eps, K))
    rgb = np.zeros(q.shape + (3,))
    for k_, colour in QUAD.items():
        rgb[q == k_] = blend(colour, 0.32)
    ax.imshow(rgb, extent=extent, origin='lower', interpolation='nearest', zorder=0)
    n = complex(np.sqrt(complex(eps)))
    for c in cut_curves(eps, key):
        line(ax, c, INK, lw=3.2, zorder=4)
    kx_plane(ax, extent[:2], extent[2:], points=(n, -n), labels=False)
    ax.grid(False)
    ax.set_title(title, loc='left', linespacing=1.3)


def fig_a09(png):
    extent = (-2.2, 2.2, -1.1, 1.1)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.3))
    titles = {'D': '경로 D (2026년 10월 2일 전의 Wavebit)\n실수축의 소멸 구간 위',
              'E': '경로 E\nbranch point에서 곧게 위(아래)로',
              'M': 'Maystre (Wavebit · RETICOLO · S4)\nbranch point에서 휘어 나감',
              'F': 'fmmax\n실수축의 전파 구간 + 허수축'}
    for ax, key in zip(axes.ravel(), 'DEMF'):
        cut_map(ax, 1.0, key, extent, titles[key])
    ax0 = axes[0, 0]
    for x, text in ((0.5, '전파 구간'), (1.6, '소멸 구간'), (-1.6, '소멸 구간')):
        ax0.text(x, -0.1, text, ha='center', va='top', fontsize=11, color=INK)
    for ax in axes[1]:
        ax.set_xlabel('Re(kx)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Im(kx)')
    handles = [Patch(color=blend(QUAD[q], 0.32), label=f'고른 근이 {q}사분면: {QUAD_TEXT[q]}') for q in (1, 2, 4, 3)]
    handles.append(Line2D([], [], color=INK, lw=3.2, label='branch cut (근이 반대편으로 튄다)'))
    handles.append(Line2D([], [], color=INK, marker='o', ls='', ms=9, mec='white', label='branch point kx = ±1'))
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, below_axes(fig, 0.5)),
               columnspacing=1.6)
    save(fig, 'a09-cut-maps', '네 규칙의 kx 평면 지도, 공기',
         '공기에서 네 규칙이 kx 평면의 각 점에서 고르는 근의 사분면(색)과 branch cut(굵은 선). 경로 D(2026년 '
         '10월 2일 전의 Wavebit)는 실수축의 소멸 구간, 경로 E는 branch point에서 곧게 위(아래)로, Maystre(지금의 '
         'Wavebit)는 branch point에서 휘어 나가는 쌍곡선의 반쪽, fmmax는 실수축의 전파 구간과 허수축에 선이 있다.', png)


def jumps_along(values, params):
    v = np.asarray(values)
    idx = np.flatnonzero((v[:-1] * np.conj(v[1:])).real < 0)
    return [(params[i], params[i + 1], v[i], v[i + 1]) for i in idx]


def fig_a10(png):
    fig = plt.figure(figsize=(11.5, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.42, left=0.07, right=0.99)
    a0, a1, a2 = (fig.add_subplot(gs[0, i]) for i in range(3))
    kx_plane(a0, (0.72, 1.62), (-0.14, 0.5), equal=False)
    for key, lw in (('D', 3.6), ('E', 3.6), ('M', 2.2)):
        for c in cut_curves(1.0, key):
            line(a0, c, RULE_COLOR[key], lw=lw, zorder=3)
    a0.text(1.155, 0.47, 'E와 Maystre의 선\nbranch point\n근처에서 거의 겹침', fontsize=11, color=INK,
            va='top', linespacing=1.3)
    a0.annotate('', (1.2, 0.05), (0.8, 0.05), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.7,
                                                            mutation_scale=15))
    a0.annotate('', (1.5, 0.05), (1.5, -0.05), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.7,
                                                             mutation_scale=15))
    a0.text(0.8, 0.065, '훑기 A', fontsize=11.5, color=INK, va='bottom')
    a0.text(1.47, -0.05, '훑기 B', fontsize=11.5, color=INK, va='center', ha='right')
    a0.set_title('(a) branch point 근처\n두 훑기와 branch cut 셋', loc='left', linespacing=1.3)

    xa = np.linspace(0.8, 1.2, 801)
    xb = np.linspace(-0.05, 0.05, 801)
    styles = {'D': dict(lw=4.6, alpha=0.45), 'E': dict(lw=2.2, ls=(0, (5, 2))), 'M': dict(lw=1.8, ls=(0, (1.2, 1.6)))}
    found = {}
    for ax, xs, kx, label in ((a1, xa, xa + 0.05j, 'A'), (a2, xb, 1.5 + 1j * xb, 'B')):
        style(ax, frame=False)
        for key in 'DEM':
            v = RULES[key](1.0, kx)
            y = v.imag.copy()
            js = jumps_along(v, xs)
            found[(label, key)] = js
            for x0, x1, _, _ in js:
                y[np.searchsorted(xs, x1)] = np.nan
            ax.plot(xs, y, color=RULE_COLOR[key], label=RULE_NAME[key], **styles[key])
            for x0, x1, v0, v1 in js:
                ax.plot([x0, x1], [v0.imag, v1.imag], ':', color=RULE_COLOR[key], lw=1.1)
        ax.axhline(0, color=HAIR, lw=0.9)
    a1.set_ylabel('Im(kz)   +이면 줄어듦, −이면 커짐')
    a2.set_ylabel('Im(kz)')
    a1.set_xlabel('Re(kx)   (Im(kx) = 0.05 고정)')
    a2.set_xlabel('Im(kx)   (Re(kx) = 1.5 고정)')
    a2.set_xticks([-0.04, 0, 0.04])
    a1.set_title('(b) 훑기 A\n복소 각도에서 각도를 훑는다', loc='left', linespacing=1.3)
    a2.set_title('(c) 훑기 B\n실제 상황을 살짝 비튼다', loc='left', linespacing=1.3)
    a1.annotate('E, Maystre가\n여기서 뒤집힘', (1.0, 0.0), (0.81, 0.42), fontsize=11, color=INK,
                linespacing=1.3, arrowprops=dict(arrowstyle='->', color=INK, lw=1.0, shrinkB=3))
    a2.annotate('D가 뒤집힘\n(허수부 0)', (0.0, 0.0), (-0.047, 0.45), fontsize=11, color=INK,
                linespacing=1.3, arrowprops=dict(arrowstyle='->', color=INK, lw=1.0, shrinkB=3))
    handles = [Line2D([], [], color=RULE_COLOR[k], label=RULE_NAME[k], **styles[k]) for k in 'DEM']
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, below_axes(fig, 0.6)),
               handlelength=3.0, columnspacing=2.2, title='선 색 = 규칙 ((a)에서는 그 규칙의 branch cut)',
               title_fontsize=11.5)
    save(fig, 'a10-sweeps', 'branch point 근처의 두 훑기',
         '공기에서 두 훑기를 따라 각 규칙이 고른 Im(kz). 훑기 A(Im(kx) = 0.05 고정, Re(kx) 0.8 → 1.2)에서는 '
         '경로 E와 Maystre(지금의 Wavebit)가 branch point 옆(Re(kx) ≈ 1.000, 1.001)에서 한 번 뒤집히고 경로 D는 '
         '매끄럽다. 훑기 B(Re(kx) = 1.5 고정, Im(kx) −0.05 → 0.05)에서는 경로 D(2026년 10월 2일 전의 Wavebit)만 '
         '허수부 0에서 +1.118i에서 −1.118i로 뒤집힌다.', png)
    return found


ABSORB_EPS = 2.25 + 0.5j                   # appendix B figure 2: an absorbing medium at real kx


def fig_b02(png):
    """Appendix B, figure 2: an absorbing medium at real kx. kz^2 = eps - kx^2 runs along the horizontal line
    Im = Im(eps) > 0, inside the upper half, so the chosen root runs inside quadrant 1 (no tie). The lossless
    medium with the same Re(eps) is drawn faintly for comparison: its kz^2 stays on the real axis."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.8, 5.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.subplots_adjust(left=0.07, right=0.99, wspace=0.3)
    k = np.linspace(0, 2.5, 1200)
    kz2 = ABSORB_EPS - k ** 2
    kz2_ll = ABSORB_EPS.real - k ** 2 + 0j
    xlim, ylim = (-4.95, 2.9), (-1.6, 2.2)
    a1.add_patch(Rectangle((xlim[0], 0), xlim[1] - xlim[0], ylim[1], color=blend(TEAL, 0.10), lw=0, zorder=0))
    a1.axhline(0, color=HAIR, lw=0.9, zorder=1)
    a1.axvline(0, color=HAIR, lw=0.9, zorder=1)
    line(a1, kz2_ll, MUTED, lw=2.0, ls=(0, (3, 2.5)), alpha=0.8, zorder=3)
    line(a1, kz2, INK, lw=2.8, zorder=4)
    arrow_on(a1, kz2, 0.55, INK)
    a1.set_xlim(*xlim)
    a1.set_ylim(*ylim)
    style(a1)
    for kv, (dx, dy, ha, va) in {0.0: (0.0, 0.14, 'center', 'bottom'), 1.5: (0.0, 0.14, 'center', 'bottom'),
                                 2.5: (0.0, 0.14, 'center', 'bottom')}.items():
        z = complex(ABSORB_EPS - kv ** 2)
        dot(a1, z, INK, ms=7)
        a1.text(z.real + dx, z.imag + dy, f'kx = {kv:g}\n{cz(z, 2)}', ha=ha, va=va, fontsize=11, color=INK,
                linespacing=1.3)
    a1.text(xlim[1] - 0.1, ylim[1] - 0.08, '위쪽 반: Im(kz²) > 0', ha='right', va='top', fontsize=11.5, color=TEAL)
    a1.text(-2.2, -0.2, '무손실 ε = 2.25: 실수축 위', ha='center', va='top', fontsize=11, color=MUTED)
    a1.set_xlabel('Re(kz²)')
    a1.set_ylabel('Im(kz²)')
    a1.set_title('(a) kz² 평면: Im(kz²) = Im(ε) = 0.5인 수평선', loc='left')

    lim2 = 2.2
    kz_plane(a2, lim2, qtext=False)
    roots = principal(kz2)
    roots_ll = principal(kz2_ll)
    line(a2, roots_ll, MUTED, lw=2.0, ls=(0, (3, 2.5)), alpha=0.8, zorder=3)
    line(a2, -roots, INK, lw=1.7, ls=(0, (3, 3)), alpha=0.45, zorder=3)
    line(a2, roots, INK, lw=2.8, zorder=4)
    arrow_on(a2, roots, 0.55, INK)
    for kv, (dx, dy, ha, va) in {0.0: (0.0, -0.12, 'center', 'top'), 1.5: (0.08, 0.1, 'left', 'bottom'),
                                 2.5: (0.1, 0.0, 'left', 'center')}.items():
        z = complex(principal(ABSORB_EPS - kv ** 2))
        dot(a2, z, INK, ms=7)
        a2.text(z.real + dx, z.imag + dy, f'kx = {kv:g}\n{cz(z, 3)}', ha=ha, va=va, fontsize=11, color=INK,
                linespacing=1.3, bbox=LABEL_BOX, zorder=7)
    a2.text(-lim2 + 0.1, -lim2 + 0.1, '점선: 버린 근 −kz\n(3사분면 안쪽)', ha='left', va='bottom', fontsize=11,
            color=MUTED, linespacing=1.3)
    a2.set_title('(b) kz 평면: 근은 1사분면 안쪽', loc='left')
    handles = [Line2D([], [], color=INK, lw=2.8, label='흡수 매질 ε = 2.25 + 0.5i, kx 0 → 2.5'),
               Line2D([], [], color=MUTED, lw=2.0, ls=(0, (3, 2.5)), alpha=0.8,
                      label='무손실 ε = 2.25 (비교용): 실수축과 1사분면의 경계')]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, below_axes(fig, 0.62)), columnspacing=1.8)
    save(fig, 'b02-real-kx-absorbing', '흡수 매질의 실수 kx',
         '흡수 매질 ε = 2.25 + 0.5i에서 실수 kx를 0에서 2.5로 키우면 kz²은 Im(kz²) = 0.5인 수평선을 따라 '
         '2.25 + 0.5i에서 −4 + 0.5i로 가고, 고른 근은 1.509 + 0.166i에서 1사분면 안쪽을 지나 올라간다. '
         '같은 실수부의 무손실 매질은 실수축과 1사분면의 경계를 따라간다.', png)
    return complex(principal(ABSORB_EPS)), complex(principal(ABSORB_EPS - 2.5 ** 2))


def fig_b03(png):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.8))
    extent = (-0.45, 3.45, -0.6, 2.45)
    for ax, eps, title in ((axes[0], 1.0 + 0j, '(a) 공기 ε = 1: branch point가 실수축 위'),
                           (axes[1], 4.0 + 3.0j, '(b) 흡수 매질 ε = 4 + 3i: branch point가 떠 있다')):
        K, _ = grid(extent, (560, 440))
        conflict = (eps - K * K).imag < 0
        rgb = np.ones(K.shape + (3,))
        rgb[conflict] = blend(ORANGE, 0.15)
        ax.imshow(rgb, extent=extent, origin='lower', interpolation='nearest', zorder=0)
        n = complex(np.sqrt(eps))
        if eps.imag > 0:
            for c in cut_curves(eps, 'P'):
                line(ax, c, MUTED, lw=1.5, ls=(0, (2, 2)), zorder=3)
        for key in 'FMED':
            for c in cut_curves(eps, key):
                line(ax, c, RULE_COLOR[key], lw=3.6 if key in 'DF' else 2.6, zorder=4 if key in 'DF' else 5)
        ax.plot([extent[0], extent[1]], [0, 0], color=INK, lw=1.2, zorder=3)
        kx_plane(ax, extent[:2], extent[2:], points=(n,))
        ax.grid(False)
        if n.imag:
            ax.text(n.real - 0.12, n.imag - 0.12, f'branch point\nn = {cz(n, 3)}', fontsize=11, color=INK,
                    ha='right', va='top', linespacing=1.3)
        else:
            ax.text(n.real + 0.1, n.imag - 0.12, f'branch point n = {n.real:g}', fontsize=11, color=INK, va='top')
        ax.set_title(title, loc='left')
    axes[0].text(2.35, 0.1, '실수 kx', fontsize=11.5, color=INK, va='bottom')
    axes[1].text(0.2, 0.1, '실수 kx: 네 규칙이 모두 같다', fontsize=11.5, color=INK, va='bottom')
    handles = [Line2D([], [], color=RULE_COLOR[k], lw=3.2, label=RULE_NAME[k]) for k in 'DEMF']
    handles.append(Line2D([], [], color=MUTED, lw=1.5, ls=(0, (2, 2)), label='뒤집기 전 주근의 선'))
    handles.append(Patch(color=blend(ORANGE, 0.15), label='Im(kz²) < 0: 두 기준이 갈리는 곳'))
    fig.tight_layout(w_pad=2.2)
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, below_axes(fig, 0.5)),
               columnspacing=1.8)
    save(fig, 'b03-air-vs-lossy', '공기와 흡수 매질의 branch point와 branch cut',
         '네 규칙의 branch cut을 한 장에 겹친 kx 평면. 공기에서는 branch point n = 1이 실수축 위에 있고, '
         '흡수 매질 ε = 4 + 3i에서는 n = 2.121 + 0.707i로 떠 있어 실수축 전체가 Im(kz²) < 0 영역(옅은 '
         '주황) 밖에 놓이고 실수 kx에서 네 규칙이 모두 같다.', png)


# ============================================================================ gain medium (appendix B)
GAIN_EPS, GLASS = 1.0 - 0.05j, 2.25        # from glass (n = 1.5) into a gain medium, TE


def gain_turned_on(eps, kx, n=20000):
    """Fix kx and turn the gain on from 0, following the root. Starts at the sure root of zero gain."""
    root = complex(principal(eps.real - kx * kx))
    for s in np.linspace(0, 1, n + 1)[1:]:
        v = complex(np.sqrt(eps.real + 1j * eps.imag * s - kx * kx))
        root = v if abs(v - root) <= abs(-v - root) else -v
    return root


def te_r(k1, k2):
    return (k1 - k2) / (k1 + k2)


GAIN_COMPLEX_EXAMPLES = ((1 - 0.05j, 0.1 + 5j), (1 - 0.05j, 3 - 0.05j))


def gain_complex_kx(N=400):
    """A gain half-space at a complex kx: the continuity-fix closed form (rule_D, Wavebit's before 2026-10-02)
    against a step walk.

    The closed form assumes Im(eps) >= 0: it looks only at Im(kx^2) > 0 and never checks t* = Im(eps)/Im(kx^2) >= 0.
    With gain it can flip at t* < 0, or miss a crossing of the negative real axis when Im(kx^2) < 0. Paths too close
    to a branch point are skipped, as in check_closed_forms.
    """
    def seg_dist(p, a, b):
        ab = b - a
        t = np.clip(((p - a) * np.conj(ab)).real / max(abs(ab) ** 2, 1e-300), 0, 1)
        return abs(p - (a + t * ab))
    rng = np.random.default_rng(5)
    eps = rng.uniform(1.0, 6.0, N) - 1j * rng.uniform(0.01, 3.0, N)
    kx = rng.uniform(0.0, 3.0, N) + 1j * rng.uniform(-0.5, 0.5, N)
    used, differ = 0, {'flips at t* < 0': 0, 'misses a crossing': 0}
    for e, k in zip(eps, kx):
        n = complex(np.sqrt(e))
        if min(seg_dist(q, 0, k) for q in (n, -n)) < 1e-2:
            continue
        used += 1
        if abs(complex(rule_D(e, k)) - walk_D(e, k, n=20000)) > 1e-6:
            differ['flips at t* < 0' if (k * k).imag > 0 else 'misses a crossing'] += 1
    examples = [(e, k, complex(rule_D(e, k)), walk_D(e, k, n=20000)) for e, k in GAIN_COMPLEX_EXAMPLES]
    return used, differ, examples


def slab_R(theta, d_um, eps2=GAIN_EPS, wavelength_um=1.0):
    """Glass | gain layer (thickness d) | air, TE. The Airy formula needs no +-kz choice inside the layer."""
    kx = np.sqrt(GLASS) * np.sin(theta)
    k1 = complex(principal(GLASS - kx * kx))
    k2 = complex(principal(eps2 - kx * kx))
    k3 = complex(principal(1.0 - kx * kx))
    phase = np.exp(2j * k2 * 2 * np.pi / wavelength_um * d_um)
    r12, r23 = te_r(k1, k2), te_r(k2, k3)
    return abs((r12 + r23 * phase) / (1 + r12 * r23 * phase)) ** 2, abs(r12 * r23), k2


def gain_numbers():
    rows = []
    for deg in (0.0, 30.0, 41.81, 41.82, 45.0):
        th = np.radians(deg)
        kx = np.sqrt(GLASS) * np.sin(th)
        k1 = complex(principal(GLASS - kx * kx))
        roots = {'gain on': gain_turned_on(GAIN_EPS, kx), 'path D': complex(rule_D(GAIN_EPS, kx)),
                 'Maystre (Wavebit)': complex(rule_M(GAIN_EPS, kx)), 'fmmax': complex(rule_F(GAIN_EPS, kx))}
        rows.append((deg, {k: (v, abs(te_r(k1, v)) ** 2) for k, v in roots.items()}))
    slabs = {deg: [(d, slab_R(np.radians(deg), d)[0]) for d in (2, 5, 10, 20, 100)] for deg in (30.0, 45.0)}
    rr, k2 = slab_R(np.radians(30.0), 1.0)[1:]
    threshold = np.log(1 / rr) / (-2 * k2.imag * 2 * np.pi)       # round-trip gain |r12 r23| e^{2|Im k2| k0 d} = 1
    return rows, slabs, threshold


def fig_b04(png):
    """Gain half-space (glass -> eps = 1 - 0.05i, TE): the two roots, the causal one, each rule's pick."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.9))
    # label anchors (x, y, ha, va) for the causal root and the opposite root
    cases = [(30.0, 0.9, '(a) 30° 전파\n인과적인 근 = 나가면서 커지는 파',
              (0.86, -0.11, 'right', 'top'), (-0.78, -0.1, 'left', 'top')),
             (45.0, 0.55, '(b) 45° 전반사\n인과적인 근 = 줄지만 계면으로 들어오는 파',
              (-0.1, 0.33, 'right', 'top'), (0.1, -0.33, 'left', 'bottom'))]
    out = []
    for ax, (deg, lim, title, pc, po) in zip(axes, cases):
        kz_plane(ax, lim)
        kx = np.sqrt(GLASS) * np.sin(np.radians(deg))
        k1 = complex(principal(GLASS - kx * kx))
        causal = gain_turned_on(GAIN_EPS, kx)
        picks = {'Wavebit': complex(rule_M(GAIN_EPS, kx)), 'Maystre': complex(rule_M(GAIN_EPS, kx)),
                 'fmmax': complex(rule_F(GAIN_EPS, kx)), '경로 D': complex(rule_D(GAIN_EPS, kx))}

        def R(z):
            return abs(te_r(k1, z)) ** 2

        def who(z):
            names = [name for name, p in picks.items() if abs(p - z) < 1e-12]
            if not names:
                return '고르는 규칙 없음'
            return ' · '.join(names) + ('이 고름' if names[-1] == 'Wavebit' else '가 고름')
        other = -causal
        ax.plot([causal.real, other.real], [causal.imag, other.imag], ':', color=INK, lw=1.2, zorder=3)
        dot(ax, causal, INK, ms=12)
        dot(ax, other, INK, filled=False, ms=12)
        ax.text(pc[0], pc[1], f'인과적인 근\n{cz(causal, 4)}\n{who(causal)}\nR = {R(causal):.4f}',
                ha=pc[2], va=pc[3], fontsize=11, color=INK, linespacing=1.35)
        ax.text(po[0], po[1], f'반대 근\n{cz(other, 4)}\n{who(other)}\nR = {R(other):.4f}',
                ha=po[2], va=po[3], fontsize=11, color=RED, linespacing=1.35)
        ax.set_title(title, loc='left', linespacing=1.3)
        out.append((deg, causal, R(causal), R(other)))
    fig.tight_layout(w_pad=2.4)
    note(fig, '유리 (n = 1.5) → 이득 반공간 ε = 1 − 0.05i, TE.  꽉 찬 점: 이득을 0에서 켜서 이은 근 '
                   '(보통의 약한 이득에서 인과적인 근)\n빈 점: 그 반대 근.  R은 그 근을 골랐을 때의 TE 반사율.  '
                   'Wavebit = Maystre, 경로 D = 2026년 10월 2일 전의 Wavebit')
    save(fig, 'b04-gain', '이득 반공간의 두 근과 인과적인 근',
         '유리에서 이득 매질 ε = 1 − 0.05i로 TE 입사. 30° 전파에서 인과적인 근은 0.6625 − 0.0377i '
         '(나가면서 커짐, R = 0.1056)이고 Wavebit(Maystre)과 경로 D가 고르며, fmmax는 반대 근(R = 9.4669)을 '
         '고른다. 45° 전반사에서 인과적인 근은 −0.0694 + 0.3603i(줄지만 들어옴, R = 1.2646)이고 Wavebit(Maystre)과 '
         'fmmax가 고르며, 경로 D(2026년 10월 2일 전의 Wavebit)는 반대 근(R = 0.7907)을 고른다.', png)
    return out


def phase_angle_kx(eps, th):
    """Real theta, phase-angle incidence: the real kx whose phase direction (Re k) makes exactly theta.

    docs/reference/common/conventions.md, Phase incidence: with eps = a + ib, N^2 = (a + sqrt(a^2 + b^2/cos^2 theta))/2
    and kx = N sin(theta). The a >= 0 side of Wavebit's `incident_phase_index`.
    """
    a, b = eps.real, eps.imag
    N = np.sqrt((a + np.sqrt(a * a + b * b / np.cos(th) ** 2)) / 2)
    return N * np.sin(th)


def formal_kx(eps, th):
    """Complex theta, the formal rule: n sin(theta) with the incident index n kept complex."""
    return complex(np.sqrt(complex(eps))) * np.sin(th)


THETA_EPS, THETA_MARK, THETA_TINY = 4.0 + 3.0j, 0.4, 1e-12    # appendix B: absorbing incident medium, angle
THETA_LOSSLESS = 4.0 + 0j                                    # the lossless medium with the same real part (n = 2)


def check_theta_rules():
    """Before b01: check what the two angle-to-kx rules claim.

    (1) The phase-angle kx is real, and the phase direction atan(kx / Re kz) of its outgoing root is
    exactly theta. (2) At theta +- 1e-12i the formal value differs from the real-theta n sin(theta)
    only at the 1e-12 scale, so the drawn ray is where a complex theta goes. (3) In a lossless medium
    the phase-angle kx is n sin(theta), so the two rules coincide.
    """
    th = np.linspace(0.02, 1.5, 300)
    for eps in (THETA_EPS, THETA_LOSSLESS):
        kx = phase_angle_kx(eps, th)
        assert np.isrealobj(kx), eps
        kz = principal(eps - kx * kx)
        assert np.max(np.abs(np.arctan(kx / kz.real) - th)) < 1e-12, eps
        n = complex(np.sqrt(eps))
        for s in (+1, -1):
            gap = np.abs(formal_kx(eps, th + s * 1j * THETA_TINY) - n * np.sin(th))
            assert np.max(gap) < 2 * abs(n) * THETA_TINY, (eps, s, gap.max())
    assert np.max(np.abs(phase_angle_kx(THETA_LOSSLESS, th) - 2 * np.sin(th))) < 1e-14
    t = THETA_MARK
    kx = phase_angle_kx(THETA_EPS, t)
    kz = complex(principal(THETA_EPS - kx * kx))
    n = complex(np.sqrt(THETA_EPS))
    return dict(kx=kx, kz=kz, deg=np.degrees(np.arctan(kx / kz.real)), n=n,
                formal={s: formal_kx(THETA_EPS, t + s * 1j * THETA_TINY) for s in (+1, -1)},
                jump=n.imag * np.sin(t),
                lossless=(phase_angle_kx(THETA_LOSSLESS, t), formal_kx(THETA_LOSSLESS, t + 1j * THETA_TINY)))


def fig_b01(png):
    """Appendix B, figure 1: the two rules that turn theta into kx. They split in an absorbing incident
    medium and coincide in a lossless one."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ticks = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
    for ax, eps, xlim, title in ((axes[0], THETA_EPS, (-0.12, 2.85), '(a) 흡수 입사 매질 ε = 4 + 3i: 두 규칙이 갈린다'),
                                 (axes[1], THETA_LOSSLESS, (-0.12, 2.3), '(b) 무손실 ε = 4 (n = 2): 두 규칙이 겹친다')):
        n = complex(np.sqrt(eps))
        # Absorbing: the phase-angle kx grows without bound as theta -> 90 degrees. Lossless: it ends at n.
        th = np.linspace(0, 1.35, 800) if eps.imag else np.linspace(0, np.pi / 2, 800)
        phase = phase_angle_kx(eps, th) + 0j
        formal = formal_kx(eps, np.linspace(0, np.pi / 2, 800))     # real theta 0 -> 90 degrees: from 0 to n
        kx_plane(ax, xlim, (-0.36, 1.0), points=(n,))
        line(ax, phase, TEAL, lw=4.4 if not eps.imag else 2.8, zorder=4)
        if eps.imag:
            arrow_on(ax, phase, 0.97, TEAL)
        line(ax, formal, ORANGE, lw=2.0, ls=(0, (5, 2.5)), zorder=5)
        for t in ticks:
            p, f = phase_angle_kx(eps, t), formal_kx(eps, t)
            ax.plot(p, 0, '|', ms=10, mew=1.8, color=TEAL, zorder=6)
            if eps.imag:
                ax.plot(f.real, f.imag, 'o', ms=4, color=ORANGE, zorder=6)
                if t in (0.8, 1.2):
                    ax.text(f.real - 0.05, f.imag + 0.03, f'θ = {t:g}', ha='right', va='bottom', fontsize=11,
                            color=ORANGE)
            if t in (0.8, 1.2):
                ax.text(p, -0.08, f'{t:g}', ha='center', va='top', fontsize=11, color=TEAL)
        p, f = phase_angle_kx(eps, THETA_MARK), formal_kx(eps, THETA_MARK + 1j * THETA_TINY)
        dot(ax, p, TEAL, ms=9)
        dot(ax, f, ORANGE, ms=9)
        if eps.imag:
            ax.annotate('', (f.real, f.imag - 0.02), (p, 0.02), zorder=7,
                        arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.4, mutation_scale=14, shrinkA=2, shrinkB=4))
            ax.text(p + 0.07, 0.13, 'θ에 허수부\n10⁻¹²만 줘도', ha='left', va='center', fontsize=11, color=INK,
                    linespacing=1.3)
            ax.text(p, -0.08, f'θ = 0.4\n{p:.6f}', ha='center', va='top', fontsize=11, color=TEAL, linespacing=1.3)
            ax.text(f.real - 0.05, f.imag + 0.04, f'θ = 0.4 ± 10⁻¹²i\n{cz(f, 6)}', ha='right', va='bottom',
                    fontsize=11, color=ORANGE, linespacing=1.3)
            ax.text(n.real + 0.07, n.imag + 0.02, f'branch point\nn = {cz(n, 3)}\n(θ = 90°)', ha='left',
                    va='bottom', fontsize=10.5, color=INK, linespacing=1.3)
            ax.text(xlim[1] - 0.05, 0.1, 'θ → 90°이면\n끝없이', ha='right', va='bottom', fontsize=10.5, color=TEAL,
                    linespacing=1.3)
        else:
            ax.text(p, -0.08, f'θ = 0.4: {p:.6f}\nθ = 0.4 + 10⁻¹²i:\n{p:.6f} + {f.imag / 1e-12:.1f}×10⁻¹²i',
                    ha='center', va='top', fontsize=11, color=INK, linespacing=1.3)
            ax.text(n.real + 0.22, 0.09, 'branch point n = 2\n(θ = 90°)', ha='right', va='bottom', fontsize=10.5,
                    color=INK, linespacing=1.3)
        ax.set_title(title, loc='left')
    handles = [Line2D([], [], color=TEAL, lw=2.8, label='실수 θ: 위상각 규약 (실수 kx, 눈금은 θ = 0.2, 0.4, …, 1.2)'),
               Line2D([], [], color=ORANGE, lw=2.0, ls=(0, (5, 2.5)),
                      label='복소 θ: 형식적인 n·sinθ (실수 θ를 아주 조금 비틀면 이 선 위로 간다)'),
               Line2D([], [], color=INK, marker='o', ls='', ms=9, mec='white', label='branch point kx = n')]
    fig.tight_layout(w_pad=2.2)
    for ax in axes:
        ax.apply_aspect()
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, below_axes(fig, 0.5)), columnspacing=1.8)
    save(fig, 'b01-theta-to-kx', '각도 θ를 kx로 바꾸는 두 규칙',
         'kx 평면. 흡수 입사 매질 ε = 4 + 3i에서 실수 θ의 위상각 규약은 실수축 위의 kx(θ = 0.4에서 0.833318)를 '
         '주고, 복소 θ의 형식적인 n·sinθ는 원점에서 branch point n = 2.121 + 0.707i까지 가는 선분 위의 kx'
         '(θ = 0.4 ± 10⁻¹²i에서 0.826081 + 0.275360i)를 준다. 무손실 ε = 4에서는 두 규칙이 실수축 위에서 겹친다.',
         png)
    return p


def angle_to_kx_in_lossy(eps=4 + 3j, deg=30.0):
    """Which kx each code makes of the same '30 degrees' inside an absorbing medium (appendix B table)."""
    a = eps.real

    def wavebit_kx(th):
        return phase_angle_kx(eps, th)
    th = np.radians(deg)
    kx_w = wavebit_kx(th)
    kx_s4 = complex(np.sqrt(eps)).real * np.sin(th)          # S4: Re(n) sin(theta)
    kx_fm = np.sqrt(a) * np.sin(th)                          # fmmax helper: sqrt(Re eps) sin(theta)
    lo, hi = 0.0, th                                         # kx_fm as an angle in Wavebit's convention
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if wavebit_kx(mid) < kx_fm else (lo, mid)
    return kx_w, kx_s4, kx_fm, np.degrees(lo), complex(np.sqrt(eps)) * np.sin(th)


# ============================================================================ statistics (appendices A and B)
def stats(rng):
    N = 3000
    eps, kr = sample_media(rng, N)
    lossless = eps.imag == 0
    out = {}
    base = {k: RULES[k](eps, kr) for k in RULES}
    same = {k: np.abs(base[k] - base['D']) < 1e-14 for k in RULES}
    out['real_same'] = {k: int(np.sum(s)) for k, s in same.items()}
    out['real_same_split'] = {k: (int(np.sum(s & lossless)), int(lossless.sum()),
                                  int(np.sum(s & ~lossless)), int((~lossless).sum())) for k, s in same.items()}
    jumps = {}
    for k in RULES:
        flipped = np.zeros(N, bool)
        for d in (1e-8, -1e-8):
            flipped |= np.abs(RULES[k](eps, kr + 1j * d) - base[k]) > 1e-4
        jumps[k] = (int(np.sum(flipped & lossless)), int(lossless.sum()),
                    int(np.sum(flipped & ~lossless)), int((~lossless).sum()))
    out['perturb'] = jumps
    # in the lossless sample, path D must flip exactly the evanescent orders and fmmax exactly the propagating ones
    evan = lossless & (kr.real > np.sqrt(eps.real))
    prop = lossless & (kr.real < np.sqrt(eps.real))
    flip = {}
    for k in 'DF':
        f = np.zeros(N, bool)
        for d in (1e-8, -1e-8):
            f |= np.abs(RULES[k](eps, kr + 1j * d) - base[k]) > 1e-4
        flip[k] = f & lossless
    assert np.array_equal(flip['D'], evan) and np.array_equal(flip['F'], prop)
    out['flip_sets'] = (int(evan.sum()), int(prop.sum()))
    M = 600
    kc = kr[:M] + 1j * rng.uniform(-0.5, 0.5, M)
    D = np.array([walk_D(e, k) for e, k in zip(eps[:M], kc)])
    E = np.array([walk_E(e, k) for e, k in zip(eps[:M], kc)])
    out['agree'] = {k: (int(np.sum(np.abs(RULES[k](eps[:M], kc) - D) < 1e-6)),
                        int(np.sum(np.abs(RULES[k](eps[:M], kc) - E) < 1e-6))) for k in RULES}
    # the same 600 split into lossless and absorbing media: (same as D, same as E, samples)
    ll = lossless[:M]
    out['agree_split'] = {}
    for part, mask in (('lossless', ll), ('absorbing', ~ll)):
        out['agree_split'][part] = {
            k: (int(np.sum((np.abs(RULES[k](eps[:M], kc) - D) < 1e-6) & mask)),
                int(np.sum((np.abs(RULES[k](eps[:M], kc) - E) < 1e-6) & mask)), int(mask.sum())) for k in RULES}
    return out


# ============================================================================ main
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--png', metavar='DIR', default=None,
                        help='also write PNG previews to DIR (never inside the guide)')
    args = parser.parse_args()
    if args.png:
        png = Path(args.png).resolve()
        if png == ROOT or ROOT in png.parents:
            parser.error(f'--png must be outside the guide ({ROOT})')
        png.mkdir(parents=True, exist_ok=True)
        args.png = png
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- checks before drawing; an AssertionError stops everything
    used, used_lossless = check_closed_forms(np.random.default_rng(7))
    print(f'[check] closed form = step walk: paths D and E both {used}/{used} '
          f'({used_lossless} lossless, {used - used_lossless} absorbing)')
    for eps, extent in ((1.0 + 0j, (-2.2, 2.2, -1.1, 1.1)), (4.0 + 3.0j, (-0.45, 3.45, -0.6, 2.45))):
        counts = check_cuts(eps, 'DEMF', extent)
        print(f'[check] eps = {eps}: branch-cut formula = grid jumps (jump points / nudged formula points) '
              + ', '.join(f'{k} {j}/{t}' for k, (j, t) in counts.items())
              + '; all four rules equal where Im(kz^2) > 0')
    points = check_lossless_is_principal(np.random.default_rng(11))
    print(f'[check] lossless dielectric eps = 1, 2.25, 4: path D = principal root at {points}/{points} points '
          '(the continuity fix never switches on)')
    grow = check_growing_root()
    theta = check_theta_rules()

    # ---- figures
    a, b = fig_a01(args.png)
    report('[a01 two roots] air: kx = 0.5 -> kz = ±%.3f ; kx = 1.5 -> kz = ±%.3fi' % (a, b.imag))
    report(f'[premise 3] passive negative index eps = mu = −1 + 0.1i -> eps·mu = {cz((-1 + 0.1j) ** 2, 4)}')
    r05, r15 = fig_a02(args.png)
    report(f'[a02 half angle] air: kz^2 = 0.75 (0°) -> {cz(r05)} ; −1.25 (180°) -> {cz(r15)} '
           f'({np.degrees(np.angle(r15)):.0f}°)')
    fig_a03(args.png)
    pairs = fig_a04(args.png)
    im_up, im_dn = ((1 - k ** 2).imag for k in (1.5 - 0.3j, 1.5 + 0.3j))
    report(f'[a04 complex pairs] kx = 1.5 − 0.3i: ±({cz(pairs[0])}), Im(kz^2) = {im_up:+.1f} ; '
           f'kx = 1.5 + 0.3i: ±({cz(pairs[1])}), Im(kz^2) = {im_dn:+.1f}'.replace('-', '−'))
    report(f'[a04 (c) x-z intensity] kz = {cz(grow["kz"], 6)}: growth 2·|Im kz| = {grow["growth"]:.4f} = '
           f'0.6·Re kx/Re kz = {grow["via"]:.4f} ; ray from x0 at height z starts at x0 − {grow["run"]:.3f}·z ; '
           f'checked |Re k·Im k| = {grow["ortho"]:.1e}, |d log I along Re k| ≤ {grow["along"]:.1e}, '
           f'|I(x0, z)/I(x_s, 0) − 1| ≤ {grow["same"]:.1e}')
    a0, a1, b0, b1 = fig_a05(args.png)
    report(f'[a05 loop] loop A: {cz(a0, 6)} -> {cz(a1, 6)} ; loop B: {cz(b0, 6)} -> {cz(b1, 6)}')
    approach, quarters = branch_point_walkthrough()
    report('[a05 approach] air, real kx -> the two candidates ±kz: ' + ', '.join(
        f'{k:g}: ' + ('both 0' if abs(v) < 1e-12 else f'±{cz(v, 3)}') for k, v in approach))
    report('[a05 loop A by quarter turns] ' + ' ; '.join(
        f'{q * 90}°: kx {cz(k, 3)}, candidates ±({cz(v, 3)}), kept {cz(r, 3)} ({ang:.0f}°)'
        for q, (k, v, r, ang) in enumerate(quarters)))
    d_end, e_end = fig_a06(args.png)
    report(f'[a06 paths D, E] target {cz(TARGET, 1)}: D ends {cz(d_end, 6)}, E ends {cz(e_end, 6)}')
    up, dn, prev, cand, dist = fig_a07(args.png)
    report(f'[a07 through the branch point] above {cz(up, 6)}, below {cz(dn, 6)} ; tie: kz(0.9999) = {prev:.6f}, '
           f'kz(1.0001) candidates ±{cand:.6f}i, distances {abs(prev - 1j * cand):.6f} / {abs(prev + 1j * cand):.6f}')
    v, chosen = fig_a08(args.png)
    report(f'[a08 half planes] roots at kx = {cz(TARGET, 1)}: ±({cz(v, 6)}) ; picks: '
           + ', '.join(f'{RULE_NAME[k]} {cz(z, 4)}' for k, z in chosen.items())
           + f', {RULE_NAME["E"]} {cz(rule_E(1.0, TARGET), 4)}')
    for small in (1.5 + 0.05j, 0.5 + 0.05j):
        report(f'                   kx = {cz(small, 2)}: ' + ', '.join(f'{k} {cz(RULES[k](1.0, small), 4)}' for k in 'DEMF'))
    report('                   weak absorption eps = 2.25 + 1e-6i, path D: '
           + ', '.join(f'kx = 1.6 {"+" if d > 0 else "−"} 1e-3i -> {cz(rule_D(2.25 + 1e-6j, 1.6 + 1j * d), 4)}'
                       for d in (-1e-3, 1e-3)))
    fig_a09(args.png)
    found = fig_a10(args.png)
    report('[a10 sweeps] where each rule flips (sweep A: Re(kx) with Im(kx) = 0.05 ; sweep B: Im(kx) with Re(kx) = 1.5)')
    for (label, key), js in found.items():
        where = ', '.join(f'{x0:+.5f} -> {x1:+.5f}: {cz(v0)} -> {cz(v1)}' for x0, x1, v0, v1 in js) or 'none'
        report(f'                   sweep {label} {RULE_NAME[key]:36s} {where}')
    report(f'                   Maystre cut on sweep A exactly at Re(kx) = sqrt(1 + 0.05^2) = {np.sqrt(1 + 0.05 ** 2):.5f}')
    fig_b01(args.png)
    report(f'[b01 theta to kx] eps = 4 + 3i, theta = 0.4: phase-angle kx = {theta["kx"]:.6f}, '
           f'kz = {cz(theta["kz"], 4)}, phase direction atan(kx / Re kz) = {theta["deg"]:.2f}° '
           f'= {np.radians(theta["deg"]):.6f} rad ; formal at theta ± 1e-12i: {cz(theta["formal"][1], 6)} and '
           f'{cz(theta["formal"][-1], 6)} ; Im(n)·sin 0.4 = {theta["jump"]:.6f} (n = {cz(theta["n"], 6)})')
    report(f'                   lossless eps = 4: phase angle {theta["lossless"][0]:.6f}, formal at theta + 1e-12i '
           f'{theta["lossless"][1].real:.6f} + {theta["lossless"][1].imag:.2e}i')
    e3, e_end = fig_b02(args.png)
    report(f'[b02 absorbing real kx] eps = 2.25 + 0.5i: kz^2 = 2.25 + 0.5i -> {cz(e3)} '
           f'({np.degrees(np.angle(2.25 + 0.5j)):.1f}° -> {np.degrees(np.angle(e3)):.1f}°) ; kx = 2.5 -> {cz(e_end)}')
    fig_b03(args.png)
    report(f'[b03 air vs lossy] eps = 4 + 3i: branch point n = {cz(np.sqrt(4 + 3j), 6)} ; '
           f'boundary hyperbola Re(kx)·Im(kx) = Im(eps)/2 = 1.5')
    rows, slabs, threshold = gain_numbers()
    for deg, c, Rc, Ro in fig_b04(args.png):
        report(f'[b04 gain] {deg:g}°: causal root {cz(c, 4)} R = {Rc:.4f} ; opposite root R = {Ro:.4f}')

    out = stats(np.random.default_rng(7))
    report()
    report('[A and B statistics] 3000 random media (half lossless, Re eps 1..6, Im eps 0.01..3), real kx 0..3')
    report('   same root as path D at real kx: ' + ', '.join(f'{k} {n}/3000' for k, n in out['real_same'].items()))
    report('      split (lossless / absorbing): ' + ', '.join(f'{k} {a}/{b} · {c}/{d}'
                                                        for k, (a, b, c, d) in out['real_same_split'].items()))
    report('   roots flipped by real kx ± 1e-8i (lossless / absorbing):')
    for k, (jl, nl, jy, ny) in out['perturb'].items():
        report(f'      {RULE_NAME[k]:36s} {jl:4d}/{nl}   {jy:4d}/{ny}')
    report(f'      lossless sample = {out["flip_sets"][0]} evanescent + {out["flip_sets"][1]} propagating: path D flips '
           'exactly the evanescent ones, fmmax exactly the propagating ones (checked)')
    report('   600 complex kx (Im −0.5..0.5): same root as the walk (path D / path E):')
    for k, (aD, aE) in out['agree'].items():
        report(f'      {RULE_NAME[k]:36s} {aD}/600   {aE}/600')
    for part, table in out['agree_split'].items():
        report(f'   the same 600, {part} media only (path D / path E):')
        for k, (aD, aE, m) in table.items():
            report(f'      {RULE_NAME[k]:36s} {aD}/{m}   {aE}/{m}')

    report()
    report('[B gain] glass -> eps = 1 − 0.05i, TE; root (quadrant) and R = |r|^2; critical angle '
           f'{np.degrees(np.arcsin(1 / 1.5)):.2f}°')
    for deg, d in rows:
        report(f'   {deg:6.2f}° ' + ' | '.join(f'{k} {cz(z, 4)} (Q{int(quadrant(z))}) R={R:.4f}' for k, (z, R) in d.items()))
    for deg, lst in slabs.items():
        report(f'   thick gain layer {deg:.0f}° (Airy): ' + ', '.join(f'd={d}µm R={R:.6f}' for d, R in lst))
    report(f'   thickness where the 30° layer round-trip gain reaches 1 ≈ {threshold:.2f} µm (phase ignored)')
    used_g, differ_g, examples_g = gain_complex_kx()
    report(f'   gain half-space at complex kx (eps: Re 1..6, Im −3..−0.01; kx: Re 0..3, Im ±0.5; seed 5): closed form '
           f'!= walked path D at {sum(differ_g.values())}/{used_g} (' + ', '.join(f'{k} {v}' for k, v in differ_g.items()) + ')')
    for e, k, closed, walked in examples_g:
        report(f'      eps = {cz(e, 2)}, kx = {cz(k, 2)}: closed form {cz(closed, 4)}, walked path D {cz(walked, 4)}')

    kx_w, kx_s4, kx_fm, deg_fm, kx_formal = angle_to_kx_in_lossy()
    report()
    report(f'[B codes] "30°" inside eps = 4 + 3i: Wavebit (phase angle) kx = {kx_w:.6f}, S4 {kx_s4:.6f}, '
           f'fmmax helper {kx_fm:.6f} ({deg_fm:.2f}° in Wavebit\'s convention), formal n·sin(theta) = {cz(kx_formal, 6)}')

    print('\nWrote ' + ', '.join(sorted(p.name for p in OUT.glob('[ab][01][0-9]-*.svg'))))
    if args.png:
        print(f'PNG previews in {args.png}')
    print('\n'.join([''] + REPORT))


if __name__ == '__main__':
    main()
