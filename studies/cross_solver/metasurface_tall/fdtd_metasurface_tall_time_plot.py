"""Plot the disclosed time calibration, without inferring a spatial order."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import studies.cross_solver.metasurface_tall.fdtd_metasurface_tall_time as study


def main():
    study.verify()
    result = json.loads((study.RESULTS/'fdtd_metasurface_tall_time_report.json').read_text())
    ps = result['periods']
    rows = [result['rows'][str(p)] for p in ps]
    fig,axes = plt.subplots(1,2,figsize=(11,5.4))
    fig.patch.set_facecolor('#faf9f6')
    for label,color in (('r','#8f3343'),('t','#186961')):
        axes[0].semilogy(ps,[r['decay'][label] for r in rows],'o-',label=label,color=color,lw=2)
    axes[0].axhline(1e-5,color='#494949',ls='--',label='registered decay line')
    axes[0].set_ylabel('Monitor energy / peak energy')
    axes[0].set_title('Residual field at the monitor')
    axes[0].legend(frameon=False,fontsize=9)
    axes[1].semilogy(ps,[r['max_energy_residual'] for r in rows],'o-',color='#345e8d',lw=2)
    axes[1].set_ylabel('Max over wavelength of |R + T - 1|')
    axes[1].set_title('Energy residual is a separate diagnostic')
    for ax in axes:
        ax.set_xlabel('Run duration, periods')
        ax.set_xticks(ps)
        ax.grid(alpha=.2,which='both')
        ax.set_facecolor('white')
    fig.suptitle('Tall M3, s incidence, 40 cells per wavelength',fontsize=16,x=.08,ha='left')
    fig.text(.08,.025,'The original 120/180-period experiment failed its decay gate.\n'
             '240 periods passes that line but retains an energy residual near 0.5%.\n'
             'These are time-calibration data, not a spatial-convergence result or an error bound.',
             fontsize=10,color='#40454b')
    fig.subplots_adjust(left=.085,right=.98,top=.84,bottom=.24,wspace=.34)
    png = study.ROOT/'docs/figures/fdtd_metasurface_tall_time_calibration.png'
    fig.savefig(png,dpi=170,facecolor=fig.get_facecolor())
    fig.savefig(study.RESULTS/'fdtd_metasurface_tall_time_calibration.svg',facecolor=fig.get_facecolor())
    print(png)


if __name__ == '__main__':
    main()
