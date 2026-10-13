"""Strong Lorentz slab spectra and material activity versus runtime/resolution."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import numpy as np
from wavebit.compat import legacy as wavebit


from studies._support._fdtd_legacy import legacy_fdtd  # noqa: E402


def hidden_material_motion():
    e = np.zeros((3, 4, 2, 2))
    e[0] = 1
    c = wavebit.fdtd_grid(np.ones((4, 2, 2)), spacing=(1.,)*3, dt=.02, steps=2000,
                         electric=e, conductivity=10., poles=[(1., .01, .01)],
                         convergence_window=100)['convergence']
    print({k: np.asarray(c[k]).tolist() for k in
           ('field_decay', 'material_decay', 'field_decayed', 'material_decayed', 'state_decayed')}, flush=True)


def measure(resolution, periods):
    band = np.array([.9, 1., 1.1])
    omega0, gamma, strength = 2*np.pi, .05, 16.
    omega = 2*np.pi/band
    epsilon = 1.5+strength/(omega0**2-omega**2-1j*gamma*omega)
    index = np.sqrt(epsilon+0j)
    interface = (1-index)/(1+index)
    phase = omega*index*.2
    denominator = 1-interface**2*np.exp(2j*phase)
    analytic_r = interface*(1-np.exp(2j*phase))/denominator
    analytic_t = (1-interface**2)*np.exp(1j*phase)/denominator
    result, diag = legacy_fdtd(
        structure=np.ones((1, 1, 1))*1.5, thickness=[.2], top=1., bot=1., period=.05,
        wavelength=band, backend='numpy', resolution=resolution, n_periods=periods,
        n_pml=round(.3*resolution), tetm='te', truncation_order=0,
        dispersion=np.ones((1, 1, 1), int), poles={1: [(omega0, gamma, strength)]},
        return_diagnostics=True, convergence_window=round(2*resolution*np.sqrt(3)/.8))
    r = np.array([v.incident.r_s.item() for v in result])
    t = np.array([v.incident.t_s.item() for v in result])
    c = diag['convergence']
    print(dict(resolution=resolution, periods=periods,
               max_r_error=float(max(abs(r-analytic_r))), max_t_error=float(max(abs(t-analytic_t))),
               field_decay=max(float(b[k]['field_decay']) for b in c['bases'] for k in ('actual', 'reference')),
               material_decay=max(float(b['actual']['material_decay'].max()) for b in c['bases']),
               amplitude_change=max(float(abs(v).max()) for v in c['amplitude_change'].values()),
               field_decayed=bool(c['field_decayed']), material_decayed=bool(c['material_decayed']),
               state_decayed=bool(c['state_decayed']), amplitudes_settled=bool(c['amplitudes_settled'])), flush=True)


if __name__ == '__main__':
    hidden_material_motion()
    for resolution, periods in [(40, 40), (40, 160), (80, 160), (40, 320), (80, 320)]:
        measure(resolution, periods)
