"""Resolve planar dispersive interfaces with a static layer-aligned z mesh."""
import argparse
import json

import numpy as np
import wavebit

from studies.fdtd_oblique_pulse import BAND, POLES, epsilon


THICKNESS = np.array([.113, .213])


def multilayer(wavelength, materials, thickness, theta=.45):
    """Independent tangential E/H characteristic matrix, vacuum half spaces.

    exp(-i*omega*t), fixed transverse k=2*pi*sin(theta), unit center wavelength.
    Matrices propagate fields from the bottom face to the top face of a layer.
    Wavebit's reflected p basis reverses its tangential electric direction.
    """
    wavelength = np.atleast_1d(wavelength)
    omega = 2*np.pi/wavelength
    transverse = np.sin(theta)*wavelength
    q0 = np.sqrt(1-transverse**2)
    result = {}
    for pol in ('te', 'tm'):
        y0 = q0 if pol == 'te' else 1/q0
        matrix = np.broadcast_to(np.eye(2, dtype=complex), (len(wavelength), 2, 2)).copy()
        for material, depth in zip(materials, thickness):
            q = np.sqrt(np.asarray(material, complex)-transverse**2)
            admittance = q if pol == 'te' else material/q
            phase = omega*q*depth
            layer = np.empty_like(matrix)
            layer[:, 0, 0] = layer[:, 1, 1] = np.cos(phase)
            layer[:, 0, 1] = -1j*np.sin(phase)/admittance
            layer[:, 1, 0] = -1j*np.sin(phase)*admittance
            matrix = matrix @ layer
        electric = matrix[:, 0, 0]+matrix[:, 0, 1]*y0
        magnetic = matrix[:, 1, 0]+matrix[:, 1, 1]*y0
        denominator = y0*electric+magnetic
        reflection = (y0*electric-magnetic)/denominator
        result[pol] = np.array([reflection if pol == 'te' else -reflection, 2*y0/denominator])
    return result


def run(resolution, *, backend='jax', align_layers=True, grading=None, side='top',
        periods=160, courant=.8, pml_width=.3, pad_scale=1.):
    result, d = wavebit.fdtd(
        structure=np.array([[[2.25]], [[1.5]]]), thickness=THICKNESS,
        dispersion=np.array([[[0]], [[1]]]), poles=POLES,
        top=1., bot=1., period=.05, wavelength=BAND, theta=.45, tetm='te',
        resolution=resolution, backend=backend, n_periods=periods, courant=courant,
        n_pml=round(pml_width*resolution), pad_scale=pad_scale,
        align_layers=align_layers, grading=grading, incidence_side=side,
        pulse_width=3., convergence_window=200, return_diagnostics=True, truncation_order=0)
    materials, thickness = [np.full_like(BAND, 2.25), epsilon(BAND)], THICKNESS
    if side == 'bottom':
        materials, thickness = materials[::-1], thickness[::-1]
    reference = multilayer(BAND, materials, thickness)
    values = {pol: np.array([[getattr(getattr(v, basis), key+'_'+component).item()
                             for v in result] for key in ('r', 't')])
              for pol, basis, component in [('te', 's_inc', 's'), ('tm', 'p_inc', 'p')]}
    convergence = d['convergence']
    row = dict(resolution=resolution, backend=backend, align_layers=align_layers,
               grading=grading, side=side, periods=periods, courant=courant, pad_scale=pad_scale,
               shape=d['shape'], dt=d['dt'], steps=d['steps'],
               pml_widths=[float(d['pml_bounds'][1]-d['pml_bounds'][0]),
                           float(d['pml_bounds'][3]-d['pml_bounds'][2])],
               boundary_distance=[float(np.min(abs(np.asarray(d['z_nodes'])-plane)))
                                  for plane in d['layer_planes']],
               error={pol: float(np.max(abs(values[pol]-reference[pol]))) for pol in values},
               state_decayed=bool(convergence['state_decayed']),
               amplitudes_settled=bool(convergence['amplitudes_settled']),
               amplitude_change=max(float(abs(v).max()) for v in convergence['amplitude_change'].values()),
               field_decay=[float(b[k]['field_decay']) for b in convergence['bases']
                            for k in ('actual', 'reference')],
               material_decay=[np.asarray(b['actual']['material_decay']).tolist()
                               for b in convergence['bases']])
    return values, row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy', 'jax'], default='jax')
    args = parser.parse_args()
    if args.backend == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
    baseline = None
    cases = [dict(resolution=r, align_layers=aligned, grading=grade)
             for r in (40, 80) for aligned, grade in ((False, None), (False, True), (True, None), (True, True))]
    cases += [dict(resolution=80, side='bottom'), dict(resolution=160)]
    cases += [dict(resolution=80, **control) for control in (
        dict(periods=320), dict(courant=.4), dict(pml_width=.6), dict(pad_scale=1.5))]
    for settings in cases:
        values, row = run(backend=args.backend, **settings)
        if settings == dict(resolution=80, align_layers=True, grading=None):
            baseline = values
        if baseline is not None and any(key in settings for key in ('periods', 'courant', 'pml_width', 'pad_scale')):
            row['change_from_baseline'] = {pol: float(np.max(abs(values[pol]-baseline[pol]))) for pol in values}
        print(json.dumps(row), flush=True)
