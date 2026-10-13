"""Separate temporal, spatial and ADE errors near a lossy Drude zero crossing."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json

import numpy as np

from studies.fdtd.layer_alignment.fdtd_layer_alignment import multilayer


BAND = np.array([.98, 1., 1.02])
GAMMA = .05
STRENGTH = (2*np.pi)**2+GAMMA**2
DEPTH = .1
THETA = .35


from studies._support._fdtd_legacy import legacy_fdtd  # noqa: E402


def reference(factors=(1., 1., 1.)):
    """Continuum Drude film, vacuum leads, fixed transverse k and unit p input."""
    omega = 2*np.pi/BAND
    material = factors[0]-STRENGTH*factors[2]/(omega**2+1j*GAMMA*factors[1]*omega)
    amplitudes = multilayer(BAND, [material], [DEPTH], theta=THETA)
    transverse = np.sin(THETA)*BAND
    q0 = np.sqrt(1-transverse**2)
    q = np.sqrt(material-transverse**2)
    phase = omega*q*DEPTH/2
    # Propagate the transmitted tangential E/H back to the middle plane,
    # then use Ez = -(kx/omega)*Hy/epsilon from Maxwell's curl equation.
    ez = transverse*amplitudes['tm'][1]/material*(np.cos(phase)-1j*material/q*q0*np.sin(phase))
    return material, amplitudes, ez


def solve(factors, backend, resolution, *, periods=120, courant=.8, side='top',
          pml_width=.3, pad_scale=1., diagnostics=True):
    # Scalar factors stay in the backend graph. Geometry and time policy are static.
    return legacy_fdtd(
        structure=factors[:1].reshape(1, 1, 1), thickness=[DEPTH], top=1., bot=1.,
        period=.05, wavelength=BAND, theta=THETA, tetm='tm', backend=backend,
        resolution=resolution, n_periods=periods, courant=courant,
        n_pml=round(pml_width*resolution), pad_scale=pad_scale, incidence_side=side,
        dispersion=np.ones((1, 1, 1), int), poles={1: [(0., GAMMA*factors[1], STRENGTH*factors[2])]},
        align_layers=True, pulse_width=3., truncation_order=0, return_diagnostics=diagnostics,
        field_monitors={'middle': DEPTH/2} if diagnostics else None,
        convergence_window=round(2/(courant/(resolution*np.sqrt(3)))) if diagnostics else None)


def objective(results):
    return sum((v.p_inc.r_p+.4j*v.p_inc.t_p).real.sum() for v in results)/len(results)


def run(resolution, backend, **changes):
    results, d = solve(np.ones(3), backend, resolution, **changes)
    material, expected, expected_ez = reference()
    values = {pol: np.array([[getattr(getattr(v, basis), key+'_'+component).item()
                             for v in results] for key in ('r', 't')])
              for pol, basis, component in [('te', 's_inc', 's'), ('tm', 'p_inc', 'p')]}
    ez = np.asarray(d['monitors']['middle']['fields']['ez'])[:, 0, 0]
    if changes.get('side') == 'bottom':
        expected_ez = -expected_ez
    c = d['convergence']
    row = dict(resolution=resolution, backend=backend, periods=changes.get('periods', 120),
               courant=changes.get('courant', .8), side=changes.get('side', 'top'),
               pml_width=changes.get('pml_width', .3), pad_scale=changes.get('pad_scale', 1.),
               shape=d['shape'], dt=d['dt'], steps=d['steps'],
               epsilon_real=material.real.tolist(), epsilon_imag=material.imag.tolist(),
               error={pol: float(np.max(abs(values[pol]-expected[pol]))) for pol in values},
               ez_error=float(np.max(abs(ez-expected_ez))),
               ez_center_abs=float(abs(ez[1])), ez_reference_abs=float(abs(expected_ez[1])),
               state_decayed=bool(c['state_decayed']), scattering_settled=bool(c['scattering_settled']),
               amplitude_change={basis: max(float(abs(v).max()) for v in c[basis]['amplitude_change'].values())
                                 for basis in ('s_inc', 'p_inc')},
               amplitudes_settled={basis: bool(c[basis]['amplitudes_settled']) for basis in ('s_inc', 'p_inc')},
               field_decay=[float(b[k]['field_decay']) for b in c['bases'] for k in ('actual', 'reference')],
               material_decay=[np.asarray(b['actual']['material_decay']).tolist() for b in c['bases']])
    return values, row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy', 'jax'], default='jax')
    parser.add_argument('--sensitivity', action='store_true')
    args = parser.parse_args()
    if args.backend == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
    if args.sensitivity:
        delta = 1e-4
        def expected(factors):
            a = reference(factors)[1]['tm']
            return float(np.mean((a[0]+.4j*a[1]).real))
        offset = np.array([0., 0., delta])
        derivative = (expected(np.ones(3)+offset)-expected(np.ones(3)-offset))/(2*delta)
        for resolution in (40, 80):
            samples = [objective(solve(np.ones(3)+sign*offset, args.backend, resolution,
                                       diagnostics=False, periods=240)) for sign in (1, -1)]
            finite = float((samples[0]-samples[1])/(2*delta))
            print(json.dumps(dict(resolution=resolution, periods=240, delta=delta,
                                  strength_derivative=finite, continuum=derivative,
                                  absolute_error=abs(finite-derivative))), flush=True)
    else:
        cases = [dict(resolution=40, periods=60), dict(resolution=40), dict(resolution=40, periods=240),
                 dict(resolution=80), dict(resolution=160), dict(resolution=80, courant=.4),
                 dict(resolution=80, pml_width=.6), dict(resolution=80, pad_scale=1.5),
                 dict(resolution=80, side='bottom'), dict(resolution=40, courant=.4)]
        baseline = None
        for case in cases:
            values, row = run(backend=args.backend, **case)
            if case == dict(resolution=80):
                baseline = values
            elif baseline is not None and case['resolution'] == 80 and any(k in case for k in ('courant', 'pml_width', 'pad_scale')):
                row['change_from_baseline'] = {pol: float(np.max(abs(values[pol]-baseline[pol]))) for pol in values}
            print(json.dumps(row), flush=True)
