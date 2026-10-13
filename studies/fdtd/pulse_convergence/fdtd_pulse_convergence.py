"""Compare temporal diagnostics with slab Airy error at fixed spatial resolution."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import platform
import numpy as np


from studies._support._fdtd_legacy import legacy_fdtd  # noqa: E402


def main():
    print(f'CPU: {platform.machine()}; NumPy {np.__version__}')
    print('| index | periods | field decay | max amplitude change | t Airy error |')
    print('| --- | --- | --- | --- | --- |')
    for index in (1.5, 4.):
        depth = .5/index
        r = (1-index)/(1+index)
        phase = 2*np.pi*index*depth
        airy = (1-r*r)*np.exp(1j*phase)/(1-r*r*np.exp(2j*phase))
        for periods in (20, 30, 60):
            result, diag = legacy_fdtd(
                structure=np.array([[[index**2]]]), thickness=[depth], top=1., bot=1.,
                period=.2, wavelength=1., backend='numpy', resolution=30, n_periods=periods,
                n_pml=12, truncation_order=0, tetm='te', return_diagnostics=True,
                convergence_window=100)
            c = diag['convergence']
            decay = max(float(run[kind]['field_decay']) for run in c['bases']
                        for kind in ('actual', 'reference'))
            change = max(float(np.max(np.abs(v))) for v in c['amplitude_change'].values())
            error = abs(result.incident.t_s.item()-airy)
            print(f'| {index} | {periods} | {decay:.6g} | {change:.6g} | {error:.6g} |', flush=True)


if __name__ == '__main__':
    main()
