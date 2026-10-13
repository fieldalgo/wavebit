"""Render the fifteen recorded charts for the metal inverse-rule learning chapter.

    python tools/docs/learning/draw_metal_records.py --output artifacts/docs/learning/en

The numbers are the ones the write-up quotes: wavebit sweeps before and after the fix
(`data/sweep_*.csv`), the comparison bundle of the wavebit-b0 session
(`data/A_*.csv`, `data/B_*.csv`, `data/DE_ladder.csv`, `data/J_dips.csv`) and a handful of
after-fix points computed for the write-up. They are written out here as literals so the
figure and the text can never disagree; the CSVs are the provenance.

English rendering of preserved records; no numerical solver is executed.
Source: wavebit-lab/nvf-metal-inverse-rule/make_figures.py
Original source SHA-256: ab1c812fb1b30e667e37fc9c2002fed9c5355e7672a56fddd76a9a9935c6dc54
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter

from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Render recorded metal-study values with English labels; no solver runs.')
parser.add_argument('--output', type=Path, required=True, help='new output folder for review')
args = parser.parse_args()
OUT = str(args.output.resolve())
if args.output.exists() and any(args.output.iterdir()):
    parser.error('output must be empty; preserved figures are never overwritten')
args.output.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# -- palette: four hues of the dataviz reference instance (light mode), validated as a set. One hue per
#    code everywhere: aqua = wavebit NVF as it was, orange = fmmax, violet = S4, blue = wavebit after the
#    fix and, where no after-fix curve is drawn, the rules that are not NVF (wavebit Laurent solid, Li rule
#    dotted for RETICOLO and dashed for wavebit). Dashed = circular truncation.
SLOT = {1: '#2a78d6', 2: '#eb6834', 3: '#1baf7a', 7: '#4a3aa7'}
INK, QUIET, MUTED, GRID, AXIS, SURFACE, REF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb', '#898781'


def chart(name, title, subtitle, xlabel, ylabel, series, xlim, ylim, xticks, yticks=None, log=False,
          hlines=(), vlines=(), notes=(), ncol=3):
    fig = plt.figure(figsize=(9.6, 5.2), dpi=150, facecolor=SURFACE)
    fig.text(0.02, 0.955, title, fontsize=11.5, weight='bold', color=INK, va='center')
    fig.text(0.02, 0.912, subtitle, fontsize=9.5, color=QUIET, va='center')
    ax = fig.add_axes([0.085, 0.135, 0.885, 0.655], facecolor=SURFACE)
    for s in series:
        line = s.get('line', True)
        ax.plot(s['x'], s['y'], color=SLOT[s['slot']], linestyle=s.get('ls', '-') if line else 'none',
                linewidth=s.get('lw', 1.6), marker='o', markersize=s.get('ms', 3.6),
                markeredgecolor=SURFACE, markeredgewidth=0.6, label=s['label'], solid_joinstyle='round', zorder=3)
    for y, label, kw in hlines:
        ax.axhline(y, color=REF, linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
        if label:
            ax.text(kw.get('x', xlim[0] + 0.01 * (xlim[1] - xlim[0])), y, label, fontsize=8.5, color=QUIET,
                    ha=kw.get('ha', 'left'), va='bottom', zorder=4)
    for x, label in vlines:
        ax.axvline(x, color=REF, linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
        if label:
            ax.text(x, ylim[1], ' ' + label, fontsize=8.5, color=QUIET, ha='left', va='top', zorder=4)
    for x, y, text, kw in notes:
        ax.annotate(text, (x, y), xytext=kw.get('offset', (8, 0)), textcoords='offset points',
                    fontsize=9, color=INK, weight=kw.get('weight', 'bold'), ha=kw.get('ha', 'left'), va='center', zorder=5)
    if log:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
    elif yticks is not None:
        ax.set_yticks(yticks)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.grid(True, axis='y', color=GRID, linewidth=0.8, zorder=0)
    ax.grid(False, axis='x')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, color=AXIS)
    ax.set_xlabel(xlabel, fontsize=9.5, color=QUIET)
    ax.set_ylabel(ylabel, fontsize=9.5, color=QUIET)
    ax.legend(loc='lower left', bbox_to_anchor=(-0.01, 1.005), ncol=ncol, frameon=False, fontsize=9,
              handlelength=2.4, columnspacing=1.6, labelcolor=INK)
    path = os.path.join(OUT, name + '.png')
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    fig.savefig(os.path.join(OUT, name+'.svg'), facecolor=SURFACE)
    plt.close(fig)
    print(path)


def cols(rows, *keys):
    return [[r[k] for r in rows] for k in keys]


def pts(pairs):
    return [p[0] for p in pairs], [p[1] for p in pairs]


N_TICKS = [121, 441, 961, 1225, 1681]

# ---------------------------------------------------------------- 1. A 441 spectrum
rows = [dict(wl=w, before=b, after=a, sq=s, ci=c) for w, b, a, s, c in [
    (796, .0008, .0008, .0007, .0007), (798, .0014, .0013, .0012, .0013), (800, .0023, .0022, .0019, .0021),
    (802, .0038, .0033, .0029, .0033), (804, .0068, .005, .0044, .0052), (806, .0174, .0072, .0066, .0079),
    (808, .0117, .0103, .0096, .0121), (810, .0127, .0145, .014, .0182), (812, .0188, .02, .0201, .0274),
    (814, .0278, .0272, .0283, .0403), (816, .04, .0359, .0384, .057), (818, .0543, .0453, .049, .0746),
    (820, .0678, .0537, .0575, .0866), (822, .0764, .0589, .0618, .0883), (824, .0782, .0598, .0616, .0812),
    (826, .0743, .057, .0581, .0704), (828, .0677, .0521, .0529, .0596), (830, .0606, .0466, .0471, .0504),
    (832, .054, .0412, .0417, .0429), (834, .0483, .0365, .0368, .0369), (836, .0435, .0325, .0325, .0322),
    (838, .0394, .0291, .029, .0285), (840, .0358, .0263, .026, .0255)]]
wl, before, after, sq, ci = cols(rows, 'wl', 'before', 'after', 'sq', 'ci')
chart('a-hole-441-spectrum',
      'A 441: correction removes the 806 nm feature and approaches square-truncated fmmax',
      'Transmission T — gold nanohole A, 441 harmonics', 'Wavelength (nm)', 'Transmittance T',
      [dict(label='Wavebit before correction', x=wl, y=before, slot=3, lw=2.0),
       dict(label='Wavebit after correction', x=wl, y=after, slot=1, lw=2.0),
       dict(label='fmmax square truncation', x=wl, y=sq, slot=2, lw=1.2, ms=3),
       dict(label='fmmax circular truncation', x=wl, y=ci, slot=2, lw=1.2, ls='--', ms=3)],
      (796, 840), (0, 0.1), [800, 810, 820, 830, 840], yticks=[0, 0.025, 0.05, 0.075, 0.1],
      notes=[(806, .0174, '806 nm: spurious pre-correction peak', dict(offset=(0, 11), ha='center'))], ncol=4)

# ---------------------------------------------------------------- 2. B 121 R+T
rows = [(820, .9531, .8746), (822, .9599, .8744), (824, .9691, .8739), (826, .984, .8727), (828, 1.0034, .8709),
        (830, 1.0219, .8685), (832, 1.0378, .8652), (834, 1.0518, .8612), (836, 1.0647, .8562), (838, 1.0768, .8504),
        (840, 1.0881, .844), (842, 1.0984, .837), (844, 1.1074, .8302), (846, 1.115, .8241), (848, 1.1208, .8193),
        (850, 1.1245, .8164), (852, 1.1259, .8158), (854, 1.1249, .8173), (856, 1.1215, .8207), (858, 1.1158, .8255),
        (860, 1.1081, .8312), (862, 1.0988, .8375), (864, 1.0884, .844), (866, 1.0774, .8504), (868, 1.066, .8567),
        (870, 1.0548, .8627)]
wl = [r[0] for r in rows]
chart('b-disk-121-rt',
      'Correction brings all recorded B 121 power sums below 1',
      'Reflected + transmitted power R+T — gold disk B, 121 harmonics', 'Wavelength (nm)', 'R + T',
      [dict(label='Wavebit before correction', x=wl, y=[r[1] for r in rows], slot=3, lw=2.0),
       dict(label='Wavebit after correction', x=wl, y=[r[2] for r in rows], slot=1, lw=2.0)],
      (820, 870), (0, 1.2), [820, 830, 840, 850, 860, 870], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
      hlines=[(1.0, 'R+T = 1 — exceeding this violates passive power balance', {})],
      notes=[(852, 1.1259, '852 nm: maximum 1.126', dict(offset=(0, 10), ha='center'))])

# ---------------------------------------------------------------- 3–5. peak / dip vs harmonics
def ladder(name, title, subtitle, ylabel, data, ylim, yticks, ncol=3):
    chart(name, title, subtitle, 'Harmonic count', ylabel,
          [dict(label=lab, x=pts(p)[0], y=pts(p)[1], slot=slot, ls=ls, lw=lw, ms=4) for lab, slot, ls, lw, p in data],
          (100, 1700), ylim, N_TICKS, yticks=yticks, ncol=ncol)

ladder('a-peak-vs-harmonics',
       'A peak: square results shift 2–5 nm; circular fmmax clusters at 822.5–822.8 nm',
       'Transmission-peak wavelength — gold nanohole A, versus harmonic count', 'Peak wavelength (nm)', [
           ('fmmax circular', 2, '--', 2.0, [(121, 836.31), (441, 821.31), (961, 822.47), (1225, 822.63), (1681, 822.77)]),
           ('fmmax square', 2, '-', 1.6, [(121, 824.51), (441, 822.87), (961, 824.43), (1225, 820.93), (1681, 823.01)]),
           ('S4 Pol circular', 7, '--', 1.6, [(121, 838.08), (441, 822.78), (961, 819.12), (1225, 820.56)]),
           ('S4 Pol square', 7, '-', 1.6, [(121, 821.44), (441, 825), (961, 824.88), (1225, 821.87)]),
           ('Wavebit NVF (before correction)', 3, '-', 2.0, [(121, 826.03), (441, 823.51), (961, 825.63), (1089, 821.53), (1225, 820.78), (1369, 818.59), (1521, 822.43), (1681, 822.76)]),
           ('RETICOLO Li', 1, ':', 1.8, [(121, 831.25), (441, 825.93), (961, 829.36)])],
       (815, 840), [815, 820, 825, 830, 835, 840])

ladder('b-dip-vs-harmonics',
       'B dip: circular fmmax at 782–786 nm; S4 Pol 20–40 nm shorter; square results vary 10–25 nm',
       'Transmission-dip wavelength — gold disk B, versus harmonic count', 'Dip wavelength (nm)', [
           ('fmmax circular', 2, '--', 2.0, [(121, 783.4), (441, 781.94), (961, 785.5), (1225, 784.36), (1681, 784.66)]),
           ('fmmax square', 2, '-', 1.6, [(121, 777.52), (441, 767.65), (961, 787.89), (1225, 794.81), (1681, 785.74)]),
           ('S4 Pol circular', 7, '--', 1.6, [(121, 745.02), (441, 746.46), (961, 753.52), (1225, 764.28)]),
           ('S4 Pol square', 7, '-', 1.6, [(121, 788.98), (441, 770.82), (961, 774.49)]),
           ('Wavebit NVF (before correction)', 3, '-', 2.0, [(121, 754.07), (441, 778.37), (961, 787.99), (1225, 782.35), (1681, 787.67)]),
           ('RETICOLO Li', 1, ':', 1.8, [(121, 797.45), (441, 792.34), (961, 793.25)])],
       (740, 800), [740, 750, 760, 770, 780, 790, 800])

ladder('j-dip-vs-harmonics',
       'Dielectric J: NVF clusters near 890.0 nm from 961 harmonics; Laurent and Li still move',
       'Transmission-dip wavelength — air hole J in an n=3.5 film, versus harmonic count', 'Dip wavelength (nm)', [
           ('wavebit NVF', 3, '-', 2.0, [(121, 888.64), (441, 889.83), (961, 890), (1089, 890.05), (1225, 890.07), (1369, 890.06), (1521, 890.05), (1681, 890.05)]),
           ('fmmax Jones square', 2, '-', 1.6, [(121, 887.52), (441, 889.84), (961, 889.99), (1225, 890.02), (1681, 890.03)]),
           ('S4 Pol square', 7, '-', 1.6, [(121, 889.56), (441, 889.99), (961, 890), (1225, 890.02), (1681, 890.04)]),
           ('wavebit Laurent', 1, '-', 1.6, [(121, 893.12), (441, 891.71), (961, 891.2), (1225, 891.09), (1681, 890.94)]),
           ('RETICOLO Li', 1, ':', 1.8, [(121, 883.96), (441, 886.5), (961, 887.52), (1225, 887.77), (1681, 888.08)]),
           ('wavebit Li', 1, '--', 1.6, [(121, 884.44), (441, 886.83), (961, 887.79), (1225, 888.02), (1681, 888.33)])],
       (883, 894), [883, 885, 887, 889, 891, 893])

# ---------------------------------------------------------------- 6. A 1369
bx, by = pts([(816, .0181), (817.5, .1048), (818, .1815), (818.5, .2198), (819, .2018), (820, .1394), (822, .076)])
ax_, ay_ = pts([(814, .033), (815, .039), (816, .0456), (817, .0524), (817.5, .0558), (818, .0592), (818.5, .0624),
                (819, .0653), (820, .0704), (821, .0741), (822, .076), (823, .0763)])
chart('a-hole-1369-spectrum',
      'A 1369: pre-correction T peaks at 0.22 (R+T≈1.27); corrected values approach the fmmax point',
      'Transmission T — gold nanohole A, 1369 harmonics', 'Wavelength (nm)', 'Transmittance T',
      [dict(label='Wavebit before correction', x=bx, y=by, slot=3, lw=2.0),
       dict(label='Wavebit after correction', x=ax_, y=ay_, slot=1, lw=2.0),
       dict(label='fmmax square (one point at 818 nm)', x=[818], y=[.0686], slot=2, line=False, ms=7)],
      (814, 824), (0, 0.25), [814, 816, 818, 820, 822, 824], yticks=[0, .05, .1, .15, .2, .25],
      notes=[(818.5, .2198, 'Before: maximum 0.2198 (R+T≈1.27)', {}),
             (818, .0686, 'fmmax 0.0686', dict(offset=(10, -10), weight='normal'))])

# ---------------------------------------------------------------- 7. A 1681
bx, by = pts([(814, .0309), (815, .0371), (816, .0443), (817, .0525), (818, .0615), (819, .0712), (820, .0812), (821.5, .0985),
              (822, .1082), (822.5, .1253), (823, .126), (823.5, .0988), (824, .092), (826, .0869), (828, .078)])
ax_, ay_ = pts([(820, .0769), (821, .0822), (822, .0855), (822.5, .0862), (823, .0864), (823.5, .0861), (824, .0852)])
sx, sy = pts([(814, .0325), (816, .0466), (818, .064), (820, .0812), (822, .0918), (822.5, .0929), (823, .0932), (823.5, .0929),
              (824, .092), (824.5, .0905), (826, .0841)])
cx_, cy_ = pts([(820, .0855), (821.5, .0938), (822, .0953), (822.5, .096), (823, .0961), (823.5, .0954), (824, .0942)])
chart('a-hole-1681-spectrum',
      'A 1681: the narrow T=0.126 peak is reduced; corrected T≈0.086 at 823 nm',
      'Transmission T — gold nanohole A, 1681 harmonics', 'Wavelength (nm)', 'Transmittance T',
      [dict(label='Wavebit before correction', x=bx, y=by, slot=3, lw=2.0),
       dict(label='Wavebit after correction', x=ax_, y=ay_, slot=1, lw=2.0),
       dict(label='fmmax square', x=sx, y=sy, slot=2, lw=1.2, ms=3),
       dict(label='fmmax circular', x=cx_, y=cy_, slot=2, lw=1.2, ls='--', ms=3)],
      (814, 828), (0, 0.14), [814, 816, 818, 820, 822, 824, 826, 828], yticks=[0, .035, .07, .105, .14],
      notes=[(823, .126, 'Before: narrow peak 0.126', {})], ncol=4)

# ---------------------------------------------------------------- 8. B 1225 reflection
bx, by = pts([(780, .4072), (781.5, .4652), (782, .4836), (782.5, .4878), (783, .4764), (784, .4341), (786, .3628), (788, .2997),
              (790, .2386), (792, .5945)])
ax_, ay_ = pts([(786, .433), (787, .4334), (788, .4339), (789, .4345), (790, .4351), (791, .4359), (792, .4367)])
sx, sy = pts([(790, .4551), (792, .4679), (793.5, .4765), (794, .4792), (794.5, .4818), (795, .4843), (795.5, .4868), (796, .4892)])
cx_, cy_ = pts([(782, .6608), (783.5, .6605), (784, .66), (784.5, .6591), (785, .6581), (786, .6554), (788, .6472)])
chart('b-disk-1225-reflection',
      'B 1225: pre-correction R kinks 0.30→0.24→0.59; corrected values vary near 0.43–0.44',
      'Reflection R — gold disk B, 1225 harmonics', 'Wavelength (nm)', 'Reflectance R',
      [dict(label='Wavebit before correction', x=bx, y=by, slot=3, lw=2.0),
       dict(label='Wavebit after correction', x=ax_, y=ay_, slot=1, lw=2.0),
       dict(label='fmmax square', x=sx, y=sy, slot=2, lw=1.2, ms=3),
       dict(label='fmmax circular', x=cx_, y=cy_, slot=2, lw=1.2, ls='--', ms=3)],
      (780, 796), (0.2, 0.7), [780, 782, 784, 786, 788, 790, 792, 794, 796], yticks=[.2, .3, .4, .5, .6, .7],
      notes=[(792, .5945, 'Before: kink at 0.5945', dict(offset=(-8, 0), ha='right'))], ncol=4)

# ---------------------------------------------------------------- 9. A 441 Laurent
rows = [(780, .0004, .0007, .0005, .0006), (782, .0003, .0007, .0004, .0004), (784, .0002, .0009, .0005, .0003),
        (786, .0001, .0011, .0008, .0002), (788, .0001, .0013, .0014, .0001), (790, .0001, .0014, .0021, .0001),
        (792, .0002, .0015, .0029, .0001), (794, .0004, .0015, .0036, .0001), (796, .0007, .0014, .004, .0002),
        (798, .0013, .0013, .0041, .0004), (800, .0021, .0013, .004, .0007), (802, .0033, .0012, .0038, .0012),
        (804, .0052, .0011, .0035, .0018), (806, .0079, .0011, .0032, .0027), (808, .0121, .0011, .003, .0039),
        (810, .0182, .0011, .0027, .0056), (812, .0274, .0012, .0025, .0078), (814, .0403, .0013, .0023, .0109),
        (816, .057, .0015, .0021, .015), (818, .0746, .0018, .002, .0205), (820, .0866, .0022, .0019, .0276),
        (822, .0883, .0027, .0018, .0364), (824, .0812, .0033, .0018, .0464), (826, .0704, .0042, .0018, .0562),
        (828, .0596, .0053, .0019, .0635), (830, .0504, .0067, .0021, .0668), (832, .0429, .0085, .0023, .0656),
        (834, .0369, .0105, .0027, .0612), (836, .0322, .0129, .0031, .0552), (838, .0285, .0154, .0038, .0488),
        (840, .0255, .0179, .0046, .0429), (842, .023, .0201, .0056, .0377), (844, .021, .0218, .0068, .0332),
        (846, .0193, .0227, .0083, .0294), (848, .0179, .0229, .01, .0262), (850, .0166, .0224, .0119, .0235),
        (852, .0156, .0214, .0139, .0212), (854, .0146, .0202, .0159, .0193), (856, .0138, .0189, .0176, .0176),
        (858, .013, .0175, .019, .0162), (860, .0123, .0162, .0198, .015), (862, .0117, .015, .0201, .014),
        (864, .0112, .0139, .0199, .0131), (866, .0106, .013, .0194, .0123), (868, .0102, .0122, .0186, .0116),
        (870, .0098, .0115, .0177, .0111)]
wl = [r[0] for r in rows]
chart('a-441-laurent',
      'A 441 Laurent: circular peak falls to T≈0.02; square peak is 5 nm longer and 60% as high',
      'Transmission T — gold nanohole A, 441 harmonics, NVF off (Laurent only)', 'Wavelength (nm)', 'Transmittance T',
      [dict(label='fmmax NVF circular (comparison)', x=wl, y=[r[1] for r in rows], slot=2, lw=1.6, ls='--', ms=3),
       dict(label='fmmax Laurent circular', x=wl, y=[r[2] for r in rows], slot=2, lw=1.6, ms=3),
       dict(label='S4 Laurent circular', x=wl, y=[r[3] for r in rows], slot=7, lw=1.6, ms=3),
       dict(label='Wavebit Laurent square', x=wl, y=[r[4] for r in rows], slot=1, lw=2.0, ms=3)],
      (780, 870), (0, 0.1), [780, 800, 820, 840, 860], yticks=[0, .025, .05, .075, .1], ncol=4)

# ---------------------------------------------------------------- 10. B 441 Laurent
rows = [(720, .8764, .9123, .9072, .8678, .8361), (724, .8546, .8999, .8932, .8395, .8076), (728, .829, .8858, .8771, .8035, .7739),
        (732, .7989, .8696, .8586, .7595, .7345), (736, .7634, .8508, .837, .7105, .689), (740, .7216, .829, .8118, .666, .6394),
        (744, .6724, .8036, .7824, .6391, .5904), (748, .615, .774, .7478, .6351, .5468), (752, .5486, .7394, .7074, .6452, .5096),
        (756, .4734, .6993, .6599, .6549, .4782), (760, .3916, .6532, .605, .6547, .4507), (764, .3065, .6006, .5419, .6429, .4225),
        (768, .224, .5413, .4708, .6218, .3886), (772, .1519, .4748, .3932, .5974, .3471), (776, .0987, .4019, .3123, .5782, .2996),
        (780, .0709, .325, .2337, .5701, .2499), (784, .0713, .2483, .1651, .5713, .2038), (788, .0978, .1785, .1152, .5737, .1698),
        (792, .145, .1231, .0908, .5701, .1559), (796, .2055, .0892, .0946, .5566, .1655), (800, .2722, .081, .1242, .5327, .1945),
        (804, .3394, .0986, .1727, .4991, .2344), (808, .4034, .138, .2319, .4564, .2774), (812, .4619, .1925, .2939, .4057, .3198),
        (816, .514, .2553, .3538, .3488, .3619), (820, .5595, .3206, .4095, .2892, .4051), (824, .5987, .3843, .4615, .2324, .4494),
        (828, .632, .4442, .5109, .1838, .4939), (832, .66, .4992, .5578, .1493, .5365), (836, .6827, .5486, .6014, .1334, .5756),
        (840, .7006, .5924, .6409, .1379, .6104), (844, .7136, .631, .6759, .1611, .6407), (848, .722, .665, .7067, .1985, .6667),
        (852, .726, .6947, .7335, .2444, .689), (856, .7259, .7206, .7569, .2932, .7087), (860, .7216, .7429, .7772, .3417, .7267)]
wl = [r[0] for r in rows]
chart('b-441-laurent',
      'B 441 Laurent: dips at 799, 793 and 836 nm; NVF comparison at 782 nm',
      'Transmission T — gold disk B, 441 harmonics, NVF off (Laurent/Li)', 'Wavelength (nm)', 'Transmittance T',
      [dict(label='fmmax NVF circular (comparison)', x=wl, y=[r[1] for r in rows], slot=2, lw=1.6, ls='--', ms=3),
       dict(label='fmmax Laurent circular', x=wl, y=[r[2] for r in rows], slot=2, lw=1.6, ms=3),
       dict(label='S4 Laurent circular', x=wl, y=[r[3] for r in rows], slot=7, lw=1.6, ms=3),
       dict(label='Wavebit Laurent square', x=wl, y=[r[4] for r in rows], slot=1, lw=2.0, ms=3),
       dict(label='RETICOLO Li square', x=wl, y=[r[5] for r in rows], slot=1, lw=1.8, ls=':', ms=3)],
      (720, 860), (0, 1), [720, 740, 760, 780, 800, 820, 840, 860], yticks=[0, .25, .5, .75, 1])

# ---------------------------------------------------------------- 11. E energy deviation
dev = {'fc': [1.401, -.0998, .1368, .006, .0022], 'fs': [5.24, -.5505, .1993, .1872, .0909],
       'sc': [.2483, .1678, .0468, .0332, .0233], 'ss': [-.5432, -.0501, -.0134, .0448, .0138], 'wb': [0, 0, 0, 0, 0]}
chart('e-energy-deviation',
      'Lossless E: fmmax/S4 deviations reach 5×10⁻³; Wavebit deviations are about 10⁻¹³',
      'R+T−1 (×10⁻³) — a-Si ellipse E, 915 nm; positive values violate passive power balance', 'Harmonic count', 'R + T − 1 (×10⁻³)',
      [dict(label='fmmax Jones circular', x=N_TICKS, y=dev['fc'], slot=2, ls='--', ms=4),
       dict(label='fmmax Jones square', x=N_TICKS, y=dev['fs'], slot=2, ms=4),
       dict(label='S4 Pol circular', x=N_TICKS, y=dev['sc'], slot=7, ls='--', ms=4),
       dict(label='S4 Pol square', x=N_TICKS, y=dev['ss'], slot=7, ms=4),
       dict(label='Wavebit NVF (symmetric; unchanged by correction)', x=N_TICKS, y=dev['wb'], slot=3, lw=2.0, ms=4)],
      (100, 1700), (-1, 6), N_TICKS, yticks=[-1, 0, 1, 2, 3, 4, 5, 6])

# ---------------------------------------------------------------- 12. the pole
kap = [i * 0.002 for i in range(61)]
series = [25.6, 27, 28.6, 30.5, 32.5, 34.8, 37.6, 40.7, 44.5, 49, 54.5, 61.3, 70.2, 81.9, 98.3, 122.4, 161.1, 229.4, 351.6,
          414.4, 286.4, 191.8, 140.3, 109.7, 89.8, 75.9, 65.7, 57.9, 51.7, 46.7, 42.6, 39.2, 36.2, 33.7, 31.5, 29.6, 27.9, 26.3,
          25, 23.7, 22.6, 21.6, 20.7, 19.8, 19, 18.3, 17.7, 17, 16.4, 15.9, 15.4, 14.9, 14.5, 14, 13.6, 13.3, 12.9, 12.6, 12.3, 12, 11.7]
parallel = [25.6, 25.5, 25.5, 25.4, 25.4, 25.3, 25.3, 25.2, 25.2, 25.1, 25.1, 25, 25, 24.9, 24.9, 24.8, 24.8, 24.7, 24.6, 24.6, 24.5,
            24.5, 24.4, 24.4, 24.3, 24.3, 24.2, 24.2, 24.1, 24.1, 24, 24, 23.9, 23.8, 23.8, 23.7, 23.7, 23.6, 23.6, 23.5, 23.5, 23.4,
            23.4, 23.3, 23.3, 23.2, 23.2, 23.1, 23, 23, 22.9, 22.9, 22.8, 22.8, 22.7, 22.7, 22.6, 22.6, 22.5, 22.5, 22.4]
assert len(series) == len(parallel) == 61
floor_51 = 51.1
apex = max(range(len(kap)), key=lambda i: series[i])
chart('inverse-rule-pole',
      'Inverse-rule response reaches 414 at κ=0.038; Laurent stays near 25',
      'Effective-permittivity magnitude — gold (818 nm, ε=−25.6+1.6i) and air', 'κ = dielectric (air) weight in the mode', '|ε_eff|',
      [dict(label='Inverse rule = series formula (with pole)', x=kap, y=series, slot=3, lw=2.0, ms=2.5),
       dict(label='Laurent rule = parallel formula', x=kap, y=parallel, slot=1, lw=2.0, ms=2.5)],
      (0, 0.12), (0, 450), [0, .02, .04, .06, .08, .1, .12], yticks=[0, 100, 200, 300, 400],
      hlines=[(floor_51, 'Corrected cap: 2 × |ε_gold| ≈ 51', dict(x=0.119, ha='right'))],
      notes=[(kap[apex], series[apex], 'Pole %.0f' % series[apex], {})])

# ---------------------------------------------------------------- 13. B 625 R+T (log)
rows = [(745, .897, .8652), (746, .872, .8605), (747, .853, .8557), (748, .839, .8506), (749, .833, .8452), (750, .837, .8396),
        (751, .87, .8337), (752, .921, .8276), (753, 1.021, .8211), (754, 1.202, .8144), (755, 1.517, .8073), (756, 2.058, .7999),
        (757, 3.065, .7922), (758, 5.682, .7843), (759, 17.866, .7761), (760, 45.258, .7676), (761, 9.089, .759), (762, 3.485, .7502),
        (763, 2.071, .7413), (764, 1.538, .7323), (765, 1.284, .7232), (766, 1.142, .7142), (767, 1.051, .7052), (768, .987, .6963),
        (769, .937, .6876), (770, .897, .6791), (771, .864, .6709), (772, .835, .663), (773, .81, .6555), (774, .789, .6485), (775, .77, .642)]
wl = [r[0] for r in rows]
chart('b-disk-625-rt',
      'B 625: R+T at 760 nm falls from 45 to 0.77 after correction',
      'Reflected + transmitted power R+T — gold disk B, 625 harmonics', 'Wavelength (nm)', 'R + T (log scale)',
      [dict(label='Wavebit before correction', x=wl, y=[r[1] for r in rows], slot=3, lw=2.0),
       dict(label='Wavebit after correction', x=wl, y=[r[2] for r in rows], slot=1, lw=2.0)],
      (745, 775), (0.5, 60), [745, 750, 755, 760, 765, 770, 775], yticks=[0.5, 1, 2, 5, 10, 20, 50], log=True,
      hlines=[(1.0, 'R+T = 1 (passive bound)', {})],
      notes=[(760, 45.258, 'Before: maximum 45.26', {})])

# ---------------------------------------------------------------- 14. B 441 R+T, fmmax and S4 (log)
rows = [(740, .8814, .8701, .8989, .8908, .9835), (744, .841, .8523, .8971, .8726, 1.008), (748, .7547, .8297, .8976, .8513, 1.0537),
        (752, .794, .8016, .9015, .8266, 1.1404), (756, .8958, .7673, .9102, .7987, 1.3142), (760, .9297, .7267, .9221, .7679, 1.6962),
        (764, .9234, .6821, .9403, .7356, 2.7709), (768, .8688, .6379, .9863, .7041, 7.6883), (772, .7754, .5997, 1.1239, .6763, 48.9589),
        (776, .6745, .5719, 1.538, .6552, 19.7241), (780, .5838, .5554, 3.2161, .6433, 4.2531), (784, .5211, .5494, 16.1245, .6418, 2.3269),
        (788, .5049, .5525, 6.0978, .6499, 1.6913), (792, .5316, .5621, 2.0336, .6656, 1.3966), (796, .5744, .5753, 1.2714, .6862, 1.2365),
        (800, .6094, .5895, 1.0175, .7092, 1.1422), (804, .6293, .6032, .9052, .7322, 1.0834), (808, .6357, .6162, .8473, .754, 1.0446),
        (812, .6323, .6292, .8163, .7736, 1.0177), (816, .622, .6433, .8022, .7906, .9987), (820, .6092, .6599, .8005, .8049, .9856),
        (824, .6004, .6793, .8071, .8169, .977), (828, .6, .7019, .8183, .8263, .9715)]
wl = [r[0] for r in rows]
chart('b-disk-441-rt-fmmax',
      'B 441: square fmmax reaches R+T≈16, circular S4 Pol ≈49; pre-correction Wavebit ≈0.93',
      'Reflected + transmitted power R+T — gold disk B, 441 harmonics', 'Wavelength (nm)', 'R + T (log scale)',
      [dict(label='S4 Pol circular', x=wl, y=[r[5] for r in rows], slot=7, lw=1.6, ls='--'),
       dict(label='fmmax square', x=wl, y=[r[3] for r in rows], slot=2, lw=2.0),
       dict(label='fmmax circular', x=wl, y=[r[4] for r in rows], slot=2, lw=1.2, ls='--', ms=3),
       dict(label='Wavebit before correction', x=wl, y=[r[1] for r in rows], slot=3, lw=2.0),
       dict(label='Wavebit after correction', x=wl, y=[r[2] for r in rows], slot=1, lw=2.0)],
      (740, 828), (0.5, 60), [740, 760, 780, 800, 820], yticks=[0.5, 1, 2, 5, 10, 20, 50], log=True,
      hlines=[(1.0, 'R+T = 1', {})], vlines=[(780, 'Predicted pole (square 441): 780 nm')],
      notes=[(784, 16.1245, 'fmmax square maximum 16.12', {}), (772, 48.9589, 'S4 Pol maximum 48.96', dict(offset=(-8, 0), ha='right'))])

# ---------------------------------------------------------------- 15. E P_yx ladder
ladder('e-ellipse-pyx-ladder',
       'Ellipse P_yx: Wavebit oscillates; S4 Pol is about 0.07 lower; fmmax is smoother',
       'Cross-polarized transmission P_yx — a-Si ellipse E, 915 nm, y-polarized incidence', 'P_yx', [
           ('Wavebit NVF (exact shape normals)', 3, '-', 2.0, [(121, .2814), (441, .3026), (961, .2907), (1089, .2937), (1225, .2956), (1369, .2958), (1521, .2948), (1681, .2936)]),
           ('fmmax circular', 2, '--', 1.4, [(121, .2786), (441, .2904), (961, .2936), (1225, .2934), (1681, .2935)]),
           ('fmmax square', 2, '-', 1.4, [(121, .2826), (441, .2913), (961, .2933), (1225, .2936), (1681, .2938)]),
           ('S4 Pol circular', 7, '--', 1.6, [(121, .2824), (441, .2067), (961, .2132), (1225, .2149), (1681, .2215)])],
       (0.2, 0.31), [.2, .22, .24, .26, .28, .3], ncol=4)
