"""Display every calibrated 40/60/80 height-study cell, without outcome selection."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

import studies.cross_solver.metasurface_tall.fdtd_metasurface_tall_settled as study


def main():
    study.verify()
    result = json.loads(study.REPORT.read_text())
    assert result['complete'] and result['controls_valid']
    classes = ['low','second','mixed','unresolved']
    colors = ['#8f3343','#186961','#99711b','#e1e4e8']
    obs = study.c7.cell_keys('M3')
    grid = np.zeros((18,6),dtype=int)
    texts = {}
    for col,(wl,name) in enumerate((wl,name) for wl in study.c7.WAVELENGTHS for name in study.HEIGHTS):
        for row,o in enumerate(obs):
            c = next(c for c in result['scenes'][name] if c['observable']==o and c['wavelength']==wl)
            grid[row,col] = classes.index(c['self_class'])
            bounds = c['self_interval']
            texts[row,col] = f'{c["self_order"]:.2f}\n[{bounds[0]:.2f}, {bounds[1]:.2f}]' if bounds else 'unresolved'
    fig,ax = plt.subplots(figsize=(12,13))
    fig.patch.set_facecolor('#faf9f6')
    ax.imshow(grid,cmap=ListedColormap(colors),vmin=-.5,vmax=3.5,aspect='auto')
    ax.set_xticks(range(6),[f'lambda {wl:g}\nh = {study.HEIGHTS[name]:.2f}'
                          for wl in study.c7.WAVELENGTHS for name in study.HEIGHTS])
    ax.xaxis.tick_top()
    ax.set_yticks(range(18),[o.replace('|',' / ') for o in obs])
    ax.tick_params(length=0,pad=9)
    for (row,col),label in texts.items():
        ax.text(col,row,label,ha='center',va='center',fontsize=9,
                color='#20252b' if grid[row,col]==3 else 'white')
    for x in (1.5,3.5):
        ax.axvline(x,color='white',lw=4)
    for y in np.arange(.5,18,1):
        ax.axhline(y,color='white',lw=.5,alpha=.5)
    rungs = ' / '.join(map(str,result['rungs']))
    ax.set_title(f'M3 height probe: every s-incidence cell\nSelf-order from {rungs} cells per wavelength',
                 fontsize=16,loc='left',pad=50)
    ax.legend(handles=[Patch(color=color,label=label) for color,label in zip(colors,classes)],
              loc='upper center',bbox_to_anchor=(.5,-.012),ncol=4,frameon=False)
    fig.text(.16,.025,'Rounded point estimate and interval; classes use full precision. Noise proxy: 60 cells.\n'
             'Low: upper <= 1.75; second: entire interval in [1.8, 2.3]. Finite-grid classes, not asymptotes.\n'
             'Height also changes corner fields and resonances; this comparison cannot isolate a cause.',
             fontsize=10,color='#40454b',va='bottom')
    fig.subplots_adjust(left=.17,right=.98,top=.85,bottom=.11)
    png = study.ROOT/'docs/figures/fdtd_metasurface_tall_settled_orders.png'
    fig.savefig(png,dpi=160,facecolor=fig.get_facecolor())
    fig.savefig(study.RESULTS/'fdtd_metasurface_tall_settled_orders.svg',facecolor=fig.get_facecolor())
    print(png)


if __name__ == '__main__':
    main()
