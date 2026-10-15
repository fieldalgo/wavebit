"""Run with a clean venv Python from /tmp to test the installed wheel.

36 planar cases against RCWA; not evidence for patterned magnetic/dispersive 3D.
"""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd

from wavebit.fdtd._periodic.legacy import permittivity_at as _legacy_wavebit_permittivity_at

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import importlib.metadata
import json
import sys

import numpy as np
import wavebit


def main():
    rows = []
    for crossed in (False, True):
        shape = (1, 2 if crossed else 1, 2)
        for material in ('dielectric', 'magnetic', 'drude'):
            for theta, phi in ((0., 0.), (.3, 0.), (.3, .4)):
                for psi in (0., np.pi/2):
                    common = dict(structure=np.full(shape, 2.25), thickness=[100.],
                                  top=1., bot=1.44, period=[100., 140.] if crossed else [100.],
                                  wavelength=900., theta=theta, phi=phi, psi=psi,
                                  truncation_order=dict(x=0, y=0))
                    extra = {}
                    ref_kw = dict(common)
                    if material == 'magnetic':
                        common['mu'] = np.full(shape, 1.3)
                        ref_kw['mu'] = common['mu']
                    if material == 'drude':
                        omega = 2*np.pi/900.
                        poles = [(0., .2*omega, .4*omega**2)]
                        extra = dict(dispersion=np.ones(shape, dtype=int), poles={1: poles})
                        ref_kw['structure'] = np.full(shape, _legacy_wavebit_permittivity_at(
                            poles, 900., eps_infinity=2.25))
                    result, diag = _legacy_wavebit_fdtd(**common, **extra, resolution=.04,
                                               n_periods=80, return_diagnostics=True)
                    ref = _legacy_wavebit_rcwa(backend='numpy', **ref_kw).solve()
                    gap = max(float(np.max(np.abs(getattr(result.incident, key)
                                                  - getattr(ref.incident, key))))
                              for key in ('r_s', 'r_p', 't_s', 't_p'))
                    assert np.isfinite(gap) and gap < .02, (crossed, material, theta, phi, psi, gap)
                    assert diag['residual'] < 1e-3, diag['residual']
                    rows.append(dict(crossed=crossed, material=material, theta=theta, phi=phi,
                                     psi=psi, amplitude_gap=gap, residual=diag['residual']))
    assert 'matplotlib' not in sys.modules
    assert 'jax' not in sys.modules and 'torch' not in sys.modules
    print(json.dumps(dict(package=wavebit.__file__,
                          versions={n: importlib.metadata.version(n)
                                    for n in ('wavebit', 'numpy', 'scipy', 'array-api-compat')},
                          cases=rows), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
