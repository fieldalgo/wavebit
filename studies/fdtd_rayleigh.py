"""Separate a discrete cutoff shift from unsettled near-Rayleigh scattering."""
import argparse
import json

import numpy as np
import wavebit

BAND = .96*np.array([.998, 1.002])
PERIOD = .8
THETA = np.arcsin(.2)
STRUCTURE = np.array([[[2.25, 1.]]])
# Keep the same physical two orders at every reference truncation. Strongly
# evanescent m=+1 extrapolation is excluded from the amplitude error metric.
ORDERS = (-1, 0)
CASES = {
    'far20-short': dict(resolution=20, periods=100),
    'far20': dict(resolution=20),
    'far40': dict(resolution=40),
    'near20': dict(resolution=20, distance=.2),
    'near40': dict(resolution=40, distance=.2),
    'near40-long': dict(resolution=40, distance=.2, periods=600),
    'near40-pml': dict(resolution=40, distance=.2, pml_width=.6),
    'narrow40': dict(resolution=40, distance=.2, width=10.),
    'narrow80': dict(resolution=80, distance=.2, width=10.),
    'narrow40-pml': dict(resolution=40, distance=.2, width=10., pml_width=.6),
}


def amplitudes(results, half, pol, basis=None):
    component = 's' if pol == 'te' else 'p'
    basis = basis or ('s_inc' if pol == 'te' else 'p_inc')
    return np.array([[np.asarray(getattr(getattr(result, basis), key+'_'+component))[0, half+np.array(ORDERS)]
                      for key in ('r', 't')] for result in results])


def reference(order):
    answer = {}
    for pol in ('te', 'tm'):
        results = [wavebit.rcwa(
            backend=0, structure=STRUCTURE, thickness=[.2], top=1., bot=1.,
            period=PERIOD, wavelength=wl, theta=np.arcsin(wl/.96*np.sin(THETA)),
            tetm=pol, truncation_order=order).solve() for wl in BAND]
        answer[pol] = amplitudes(results, order, pol, basis='given')
    return answer


def run(expected, backend, resolution, periods=300, distance=None, width=3., pml_width=.3,
        method='electric', pad_scale=1., pml_reflection=1e-8, convergence_window=200,
        pulse_notches=None, source_distance=None):
    results, d = wavebit.fdtd(
        structure=STRUCTURE, thickness=[.2], top=1., bot=1., period=PERIOD,
        wavelength=BAND, theta=THETA, tetm='te', truncation_order=1,
        backend=backend, resolution=resolution, n_pml=round(pml_width*resolution),
        n_periods=periods, align_layers=True, pulse_width=width,
        pulse_notches=pulse_notches,
        source_distance=source_distance,
        diffraction_method=method, pad_scale=pad_scale, pml_reflection=pml_reflection,
        diffraction_distance=distance, convergence_window=convergence_window, return_diagnostics=True)
    info, c = d['diffraction'], d['convergence']
    earlier = {}
    for pol, component, basis in (('te', 's', 's_inc'), ('tm', 'p', 'p_inc')):
        delta = np.stack([np.asarray(c[basis]['amplitude_change'][key+'_'+component])
                          for key in ('r', 't')], axis=1)[:, :, 0, 1+np.array(ORDERS)]
        earlier[pol] = amplitudes(results, 1, pol)-delta if np.isfinite(delta).all() else None
    return dict(
        resolution=resolution, periods=periods, distance=distance, pulse_width=width,
        method=method, pml_width=pml_width, pml_reflection=d['pml_reflection'],
        pad_scale=pad_scale, shape=d['shape'], dt=d['dt'], steps=d['steps'],
        pulse_notches=np.asarray(d['pulse']['notches']).tolist(),
        source_relative_spectrum=np.asarray(d['pulse']['relative_spectrum']).tolist(),
        source_distance=d['source_distance'], source_z=float(d['source_z']),
        pml_bounds=d['pml_bounds'],
        convergence_window=convergence_window, comparison_duration=convergence_window*d['dt'],
        monitor_z=[info[name] for name in ('reflection_monitor_z', 'transmission_monitor_z')],
        error={pol: float(abs(amplitudes(results, 1, pol)-expected[pol]).max()) for pol in expected},
        amplitude_real={pol: amplitudes(results, 1, pol).real.tolist() for pol in expected},
        amplitude_imag={pol: amplitudes(results, 1, pol).imag.tolist() for pol in expected},
        prefix_amplitude_real={pol: value.real.tolist() if value is not None else None for pol, value in earlier.items()},
        prefix_amplitude_imag={pol: value.imag.tolist() if value is not None else None for pol, value in earlier.items()},
        energy={pol: [float(np.asarray(getattr(r, basis).R+getattr(r, basis).T).sum()) for r in results]
                for pol, basis in (('te', 's_inc'), ('tm', 'p_inc'))},
        cutoff_resolved=bool(info['cutoff_resolved']),
        propagating=np.asarray(info['reflection']['propagating']).tolist(),
        q_real=np.asarray(info['reflection']['q']).real.tolist(),
        q_imag=np.asarray(info['reflection']['q']).imag.tolist(),
        yee_cutoff_parameter=np.asarray(info['reflection']['yee_cutoff_parameter']).tolist(),
        state_decayed=bool(c['state_decayed']), scattering_settled=bool(c['scattering_settled']),
        amplitude_change={basis: max(float(abs(v).max()) for v in c[basis]['amplitude_change'].values())
                          for basis in ('s_inc', 'p_inc')},
        tail_by_order={basis: np.max(np.abs(np.stack(list(c[basis]['amplitude_change'].values()))), axis=(0, 1, 2)).tolist()
                       for basis in ('s_inc', 'p_inc')},
        incoming_by_order={basis: {key: np.max(abs(np.asarray(value)), axis=(0, 1)).tolist()
                                    for key, value in components.items()}
                           for basis, components in info.get('incoming', {}).items()},
        field_decay=[float(b[k]['field_decay']) for b in c['bases'] for k in ('actual', 'reference')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy', 'jax'], default='jax')
    parser.add_argument('--case', choices=['all', 'reference', *CASES], default='all')
    args = parser.parse_args()
    if args.backend == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
    expected = reference(32)
    for order in (16, 64):
        other = reference(order)
        print(json.dumps(dict(reference_order=order, compared_with=32,
                              change={pol: float(abs(other[pol]-expected[pol]).max()) for pol in expected})), flush=True)
    for name, case in CASES.items():
        if args.case in ('all', name):
            print(json.dumps(dict(case=name, **run(expected, args.backend, **case))), flush=True)
