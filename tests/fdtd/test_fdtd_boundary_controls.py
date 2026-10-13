"""Vacuum supplies an analytic phase reference for padding and grid refinement."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np



def test_crossed_vacuum_phase_and_padding_sensitivity_refine():
    # All structure faces lie on nodes at both grids. A longer padding region is
    # not automatically more accurate: the propagating field also accumulates
    # numerical phase. Test complex coefficients, not only |t| or R+T.
    theta = np.deg2rad(18.)
    expected = np.exp(2j*np.pi/.8*np.cos(theta)*.2)
    measured = {}
    for resolution in (25., 50.):
        for pad in (1., 1.5):
            result = _legacy_wavebit.fdtd(
                structure=np.ones((1, 2, 2)), thickness=[.2], top=1., bot=1.,
                period=[.16, .16], wavelength=.8, psi=.4, theta=theta,
                phi=np.deg2rad(40.), resolution=resolution, n_periods=120,
                n_pml=12, pad_scale=pad, courant=.99, subpixel=True, grading=None)
            measured[resolution, pad] = np.array([result.s_inc.t_s.item(),
                                                  result.p_inc.t_p.item()])
    for pad in (1., 1.5):
        coarse = np.abs(measured[25., pad] - expected)
        fine = np.abs(measured[50., pad] - expected)
        assert np.all(fine < .003)
        assert np.all(fine < .4*coarse)
    coarse_change = np.abs(measured[25., 1.5] - measured[25., 1.])
    fine_change = np.abs(measured[50., 1.5] - measured[50., 1.])
    assert np.all(fine_change < .4*coarse_change)
