"""A Bragg cavity must not be certified while resonant transmission is filling."""
import numpy as np
import wavebit


def test_bragg_cavity_rejects_short_transients_against_unit_resonant_transmission():
    # Three quarter-wave n=2/1 pairs, half-wave vacuum defect, mirrored pairs.
    # Each quarter-wave matrix has zero diagonal. The palindromic product at
    # wavelength 1 is -I, giving analytic vacuum-to-vacuum t=-1 and T=1.
    indices = np.r_[np.tile([2., 1.], 3), 1., np.tile([1., 2.], 3)]
    thickness = np.r_[np.tile([.125, .25], 3), .5, np.tile([.25, .125], 3)]
    errors = []
    transmissions = []
    for periods in (25, 80):
        result, diag = wavebit.fdtd(
            structure=(indices**2)[:, None, None], thickness=thickness,
            top=1., bot=1., period=.05, wavelength=1., backend=0, resolution=40,
            n_periods=periods, truncation_order=0, tetm='te',
            return_diagnostics=True, convergence_window=200)
        c = diag['convergence']
        assert not bool(c['field_decayed'])
        assert not bool(c['amplitudes_settled'])
        assert c['calibration_valid'].all()
        assert all(bool(v[k]['source_inactive']) for v in c['bases'] for k in ('actual', 'reference'))
        assert abs(c['amplitude_change']['t_s']).max() > 1e-3
        errors.append(abs(result.given.t_s.item()+1))
        transmissions.append(result.T.item())
    assert transmissions[1] > transmissions[0]+.3
    assert errors[1] < errors[0]*.6
    # Passing the time-step/CFL check is not a convergence certificate.
    assert bool(diag['stable'])
