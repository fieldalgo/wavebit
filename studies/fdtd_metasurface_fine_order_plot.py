"""Display every registered fine-ridge self and final-step direct classification.

This presentation step reads the complete report; it adds no numerical run,
selection rule, threshold or classification. Unresolved cells stay unresolved.
"""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


RESULTS = Path(__file__).resolve().parent / 'results'
FIGURES = RESULTS.parents[1] / 'docs' / 'figures'
REPORT = RESULTS / 'fdtd_metasurface_fine_report.json'
OBSERVABLES = ('r|r_s', 'r|r_p', 'r|R', 't|t_s', 't|t_p', 't|T')
CLASSES = ('low', 'second', 'mixed', 'unresolved')
COLOURS = ('#F5C5A9', '#BFE1D4', '#F7E4A3', '#E8E8E8')


def draw(report, kind):
    matrix, labels, annotations = [], [], []
    for scene in ('Y12', 'Y4', 'D12', 'F12'):
        entries = report['scenes'][scene]
        lookup = {(c['wavelength'], c['pol'], c['observable']): c for c in entries}
        if len(lookup) != 36 or any(c['status'] != 'ok' for c in entries):
            raise RuntimeError(f'{scene}: expected all 36 valid cells.')
        for wl in (.95, 1., 1.05):
            group = [lookup[(wl, pol, obs)] for pol in ('s', 'p') for obs in OBSERVABLES]
            classes = [c['self_class'] if kind == 'self' else c['direct_classes'][-1] for c in group]
            orders = [c['self_order'] if kind == 'self' else c['direct_orders'][-1] for c in group]
            matrix.append([CLASSES.index(name) for name in classes])
            annotations.append([f'{order:.2f}' if name != 'unresolved' else '--'
                                for order, name in zip(orders, classes)])
            top = '' if kind == 'self' else f"   N={group[0]['reference_top']}"
            labels.append(f'{scene}   {wl:.2f}'+top)
    fig, ax = plt.subplots(figsize=(11.4 if kind == 'self' else 13., 8.5), layout='constrained')
    ax.imshow(matrix, cmap=ListedColormap(COLOURS), vmin=-.5, vmax=3.5, aspect='auto')
    names = [obs.split('|')[-1] for obs in OBSERVABLES]
    ax.set_xticks(range(12), ['s: '+name for name in names] + ['p: '+name for name in names])
    ax.set_yticks(range(12), labels)
    title = ('Self-orders from 160 / 320 / 640 cells per wavelength' if kind == 'self'
             else 'Reference-based orders over 320 to 640 cells per wavelength')
    ax.set(ylabel='Scene and vacuum wavelength', xlabel='Incident polarisation: outgoing observable',
           title='All 144 cells: '+title)
    ax.tick_params(length=0, pad=8)
    for y, row in enumerate(annotations):
        for x, label in enumerate(row):
            ax.text(x, y, label, ha='center', va='center', fontsize=10, color='#202020')
    for y in (2.5, 5.5, 8.5):
        ax.axhline(y, color='white', linewidth=3)
    ax.axvline(5.5, color='white', linewidth=3)
    ax.legend(handles=[Patch(facecolor=colour, label=name)
                       for name, colour in zip(CLASSES, COLOURS)],
              loc='upper center', bbox_to_anchor=(.5, -.095), ncol=4, frameon=False)
    note = ('No RCWA reference used. A negative self-order can coexist with decreasing reference error.'
            if kind == 'self' else 'Intervals include reference spread and the time/absorber noise proxy.')
    fig.suptitle('Colour follows the registered empirical interval; number is its raw point estimate\n'
                 '-- = unresolved (noise, reference precision or vanishing difference)\n'+note, fontsize=11)
    for suffix in ('.png', '.svg'):
        destination = FIGURES if suffix == '.png' else RESULTS
        fig.savefig(destination / ('fdtd_metasurface_fine_'+kind+'_orders'+suffix), dpi=180,
                    metadata={'Creator': 'wavebit fine-ridge registered report'})
    plt.close(fig)


def main():
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    if not report['complete'] or not report['controls_valid']:
        raise RuntimeError('A complete report with valid controls is required.')
    plt.rcParams.update({'font.size': 11, 'svg.hashsalt': 'wavebit-fine-orders'})
    for kind in ('self', 'reference'):
        draw(report, kind)
    print(json.dumps({'matplotlib': matplotlib.__version__, 'fingerprint': report['fingerprint'],
                      'source': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                      'report': hashlib.sha256(REPORT.read_bytes().replace(b'\r\n', b'\n')).hexdigest()},
                     indent=2))


if __name__ == '__main__':
    main()
