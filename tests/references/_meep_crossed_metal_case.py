"""A lossy negative-epsilon Drude patch in a subwavelength rectangular 3D cell.

Lengths are micrometres. This deliberately damped case is not a low-loss plasmonic
resonance benchmark. Only the zeroth order propagates; nonzero orders remain near fields.
"""
import numpy as np

OMEGA = 2*np.pi/.9
MASK = np.array([[[1, 0], [1, 1], [0, 0]]], dtype=int)
CASE = dict(wavelength=.9, period=(.2, .3), thickness=(.1,),
            theta_deg=20., phi_deg=35., n_top=1., n_bot=1.4,
            index=np.ones(MASK.shape), mu=np.ones(MASK.shape),
            dispersion=MASK, poles={1: [(0., .5*OMEGA, 4*OMEGA**2)]},
            orders_u=(-1, 0, 1), orders_v=(-1, 0, 1),
            meep_dpml=.5, meep_pad=.6, meep_until=120.)
