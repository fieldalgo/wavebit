"""Separate source-bandwidth tails from metal-interface and diffraction errors."""
import argparse
import json

import numpy as np
import wavebit


POLES = {1: [(0., .08, 160.), (10., .12, 40.)]}
BAND = np.array([.9, 1., 1.1])


def epsilon(wavelength):
    omega = 2*np.pi/np.asarray(wavelength)
    return 1.5+160/(-omega**2-1j*.08*omega)+40/(100-omega**2-1j*.12*omega)


def airy(thickness):
    omega = 2*np.pi/BAND
    material = epsilon(BAND)
    q0 = np.sqrt(1-(np.sin(.45)*BAND)**2)
    q1 = np.sqrt(material-(np.sin(.45)*BAND)**2)
    phase = np.exp(1j*omega*q1*thickness)
    result = {}
    for pol, interface in [('te', (q0-q1)/(q0+q1)),
                            ('tm', (material*q0-q1)/(material*q0+q1))]:
        denominator = 1-interface**2*phase**2
        result[pol] = np.array([interface*(1-phase**2)/denominator,
                                (1-interface**2)*phase/denominator])
    return result


def run(resolution, periods, width, backend, *, thickness=.2, grading=None, patterned=False):
    result, d = wavebit.fdtd(
        structure=np.array([[[1.5, 1.]]]) if patterned else np.full((1, 1, 1), 1.5),
        dispersion=np.array([[[1, 0]]]) if patterned else np.ones((1, 1, 1), int), poles=POLES,
        thickness=[thickness], top=1., bot=1., period=.8 if patterned else .05,
        wavelength=1. if patterned else BAND, theta=.45, tetm='te',
        backend=backend, resolution=resolution, n_periods=periods, n_pml=round(.3*resolution),
        truncation_order=1 if patterned else 0, pulse_width=width, grading=grading,
        convergence_window=200, return_diagnostics=True)
    c = d['convergence']
    values = {}
    for pol, basis, component in [('te', 's_inc', 's'), ('tm', 'p_inc', 'p')]:
        if patterned:
            # Orders -1 and 0 propagate; +1 is evanescent and excluded.
            values[pol] = np.array([np.asarray(getattr(getattr(result, basis), key+'_'+component))[0, :2]
                                    for key in ('r', 't')])
        else:
            values[pol] = np.array([[getattr(getattr(v, basis), key+'_'+component).item()
                                     for v in result] for key in ('r', 't')])
    row = dict(resolution=resolution, periods=periods, width=width, thickness=thickness,
               grading=grading, patterned=patterned, backend=backend, shape=d['shape'], dt=d['dt'],
               pml_width=float(d['pml_bounds'][1]-d['pml_bounds'][0]),
               state_decayed=bool(c['state_decayed']), amplitudes_settled=bool(c['amplitudes_settled']),
               amplitude_change=max(float(abs(v).max()) for v in c['amplitude_change'].values()),
               field_decay=[float(b[k]['field_decay']) for b in c['bases'] for k in ('actual', 'reference')],
               material_decay=[np.asarray(b['actual']['material_decay']).tolist() for b in c['bases']])
    return values, row


def patterned(backend):
    references = {}
    for order in (12, 24, 48, 96):
        references[order] = {}
        for pol, component in [('te', 's'), ('tm', 'p')]:
            result = wavebit.rcwa(
                structure=np.array([[[epsilon(1.), 1.]]]), thickness=[.2], top=1., bot=1.,
                period=.8, wavelength=1., theta=.45, tetm=pol, backend=0,
                truncation_order=order).solve().given
            # Also retain the first evanescent order for the reference check.
            references[order][pol] = np.array([getattr(result, key+'_'+component)[0, order-1:order+2]
                                               for key in ('r', 't')])
    for lower, upper in [(12, 24), (48, 96)]:
        print(json.dumps(dict(reference_orders=[lower, upper],
                              change={pol: float(np.max(abs(references[lower][pol]-references[upper][pol])))
                                      for pol in ('te', 'tm')})), flush=True)
    for resolution, periods, width, reference_order in [
        (20, 80, 1., 24), (20, 80, 3., 24), (20, 160, 3., 24), (40, 160, 3., 24),
        (20, 160, 6., 96), (40, 160, 6., 96), (80, 160, 6., 96),
    ]:
        values, row = run(resolution, periods, width, backend, patterned=True)
        row.update(reference_order=reference_order,
                   te_error=float(np.max(abs(values['te']-references[reference_order]['te'][:, :2]))),
                   tm_difference_from_unconverged_reference=float(np.max(
                       abs(values['tm']-references[reference_order]['tm'][:, :2]))))
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy', 'jax'], default='jax')
    parser.add_argument('--pattern', action='store_true', help='Run the patterned case instead of planar cases.')
    args = parser.parse_args()
    if args.backend == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
    if args.pattern:
        patterned(args.backend)
    else:
        for thickness, resolution, width, grading in [
            (.2, 40, 1., None), (.2, 80, 1., None), (.2, 40, 3., None), (.2, 80, 3., None),
            (.213, 40, 3., None), (.213, 80, 3., None), (.213, 40, 3., True), (.213, 80, 3., True),
        ]:
            values, row = run(resolution, 160, width, args.backend, thickness=thickness, grading=grading)
            reference = airy(thickness)
            row['error'] = {pol: float(np.max(abs(values[pol]-reference[pol]))) for pol in ('te', 'tm')}
            print(json.dumps(row), flush=True)
