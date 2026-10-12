"""Plot the preserved reference-only scans used to choose the 640-cell controls.

This is a descriptive view of the registered inputs, not another selection rule.
It does not read the new FDTD values or change the registered campaign.
"""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


RESULTS = Path(__file__).resolve().parent / 'results'
FIGURES = RESULTS.parents[1] / 'docs' / 'figures'
INPUTS = [RESULTS / 'fdtd_metasurface_fine_scan.jsonl',
          RESULTS / 'fdtd_metasurface_fine_scan_more.jsonl']


def main():
    rows = [json.loads(line) for path in INPUTS for line in path.read_text().splitlines()]
    plt.rcParams.update({'font.size': 11, 'svg.hashsalt': 'wavebit-fine-prescan'})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), layout='constrained')
    colours = ['#0072B2', '#CC79A7', '#D55E00', '#009E73']
    for height, colour in zip((.30, .20, .15, .10), colours):
        group = sorted((r for r in rows if r['eps'] == 12. and r['height'] == height
                        and r['N'] == 80), key=lambda r: r['wavelength'])
        axes[0].plot([r['wavelength'] for r in group],
                     [r['values']['p|r|R'][0] for r in group],
                     '.-', color=colour, label=f'Height {height:.2f}')
    group = sorted((r for r in rows if r['eps'] == 4. and r['N'] == 80),
                   key=lambda r: r['wavelength'])
    for pol, colour in zip(('s', 'p'), colours):
        axes[1].plot([r['wavelength'] for r in group],
                     [r['values'][pol+'|r|R'][0] for r in group],
                     '.-', color=colour, label=pol+' incidence')
    for ax in axes:
        ax.axvspan(.95, 1.05, color='#999999', alpha=.13, zorder=-10)
        ax.set(xlim=(.90, 1.10), xlabel='Vacuum wavelength (band-centre units)',
               ylabel='Zeroth-order reflectance')
        ax.set_xticks((.90, .95, 1., 1.05, 1.10))
        ax.grid(alpha=.2)
        ax.legend(fontsize=10)
    axes[0].set(title='Epsilon 12: height moves the sampled p peak', ylim=(0., 1.02))
    axes[1].set(title='Epsilon 4 control: height 0.30', ylim=(.02, .043))
    fig.suptitle('Reference-only control selection; shaded band: 0.95–1.05\n'
                 'N = 80 samples, before the new FDTD campaign', fontsize=13)
    for suffix in ('.png', '.svg'):
        destination = FIGURES if suffix == '.png' else RESULTS
        fig.savefig(destination / ('fdtd_metasurface_fine_prescan'+suffix), dpi=180,
                    metadata={'Creator': 'wavebit fine-ridge prescan'})
    plt.close(fig)
    record = {'matplotlib': matplotlib.__version__,
              'source': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
              'inputs': {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                         for p in INPUTS}}
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
