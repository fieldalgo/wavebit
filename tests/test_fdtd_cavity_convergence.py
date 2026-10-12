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
            top=1., bot=1., period=.05, wavelength=1., backend='numpy', resolution=40,
            n_periods=periods, truncation_order=0, tetm='te',
            return_diagnostics=True, convergence_window=200)
        c = diag['convergence']
        assert not bool(c['field_decayed'])
        assert not bool(c['amplitudes_settled'])
        assert c['calibration_valid'].all()
        assert all(bool(v[k]['source_inactive']) for v in c['bases'] for k in ('actual', 'reference'))
        assert abs(c['amplitude_change']['t_s']).max() > 1e-3
        errors.append(abs(result.incident.t_s.item()+1))
        transmissions.append(result.T.item())
    assert transmissions[1] > transmissions[0]+.3
    assert errors[1] < errors[0]*.6
    # Passing the time-step/CFL check is not a convergence certificate.
    assert bool(diag['stable'])


def test_long_cavity_matches_discrete_outgoing_boundary_problem():
    # Independent harmonic finite-difference boundary problem, not an FDTD run
    # or a continuum transfer matrix. It isolates truncation/PML errors from
    # the spatial Yee dispersion already present in the reference itself.
    indices = np.r_[np.tile([2., 1.], 3), 1., np.tile([1., 2.], 3)]
    thickness = np.r_[np.tile([.125, .25], 3), .5, np.tile([.25, .125], 3)]
    h, dt = 1/40, .8/(40*np.sqrt(3))
    depth = thickness.sum()
    z = np.arange(-1, round(depth/h)+2)*h
    cuts = np.r_[0., np.cumsum(thickness)]
    fraction = np.maximum(0., np.minimum(z[:, None]+h/2, cuts[1:])
                          - np.maximum(z[:, None]-h/2, cuts[:-1]))/h
    epsilon = 1+fraction@(indices**2-1)
    phase = 2*np.arcsin(h*np.sin(np.pi*dt)/dt)
    matrix = np.diag((4*(h*np.sin(np.pi*dt)/dt)**2*epsilon-2).astype(complex))
    matrix += np.diag(np.ones(len(z)-1), 1)+np.diag(np.ones(len(z)-1), -1)
    matrix[0, 0] += np.exp(1j*phase)
    matrix[-1, -1] += np.exp(1j*phase)
    drive = np.zeros(len(z), complex)
    drive[0] = 2j*np.sin(phase)*np.exp(1j*phase*z[0]/h)
    field = np.linalg.solve(matrix, drive)
    expected_r = (field[0]-np.exp(1j*phase*z[0]/h))*np.exp(1j*phase*z[0]/h)
    expected_t = field[-1]*np.exp(-1j*phase*(z[-1]-depth)/h)
    np.testing.assert_allclose(abs(expected_r)**2+abs(expected_t)**2, 1., atol=1e-12)
    result, diagnostic = wavebit.fdtd(
        structure=(indices**2)[:, None, None], thickness=thickness,
        top=1., bot=1., period=.05, wavelength=1., backend='numpy', resolution=40,
        n_periods=400, truncation_order=0, tetm='te',
        return_diagnostics=True, convergence_window=433)
    np.testing.assert_allclose(diagnostic['dt'], dt, atol=1e-16)
    np.testing.assert_allclose([result.incident.r_s.item(), result.incident.t_s.item()],
                               [expected_r, expected_t], atol=1e-3, rtol=0)
    assert diagnostic['convergence']['amplitudes_settled']
    assert diagnostic['convergence']['field_decayed']
