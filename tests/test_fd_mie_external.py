"""Independent radiation fields pin the sphere and external measuring instruments."""
from types import SimpleNamespace
import numpy as np
import pytest
from studies.fdtd_subpixel_mie_sphere import read_result
from references.interface.measure_fd_stage4_fdtdx import surface_power
from references.interface.measure_fd_stage4_meep import scattering_efficiency


def dipole_result():
    # Independent radiating electric dipole, with radiation-reaction correction
    # Im(alpha) = k^3 |alpha|^2 / (6 pi). This makes the optical theorem exact.
    k=2*np.pi
    radius=.001/k
    alpha0=4*np.pi*radius**3*(2.25-1)/(2.25+2)
    alpha=alpha0/(1-1j*k**3*alpha0/(6*np.pi))
    sigma=k**4*abs(alpha)**2/(6*np.pi)
    A=2j
    I=abs(A)**2/2
    class Source:
        def intensity(self,ws): return np.array([I])
        def incident_amplitude(self,ws,plane=None):
            return np.array([A if plane==0. else A*np.exp(1j*np.pi/3)])
    class Radiation:
        propagation = 'continuum'  # This analytic dipole is a continuum oracle.
        flux=np.array([sigma*I])
        def fields(self,theta,phi):
            t,p=np.radians(theta),np.radians(phi)
            f=A*k**2*alpha/(4*np.pi)
            return np.array([f*np.cos(t)*np.cos(p)]),np.array([-f*np.sin(p)])
        def intensity(self,theta,phi):
            a,b=self.fields(theta,phi)
            return .5*(abs(a)**2+abs(b)**2)
    class Result(dict): pass
    res=Result(sca=SimpleNamespace(flux=np.array([I*sigma])),
               abs=SimpleNamespace(flux=np.array([0.])),ff=Radiation())
    res.sources={'t':Source()}
    res.diagnostics={'decay':{'abs':0.,'sca':0.,'ff':0.}}
    res.plan=SimpleNamespace(grid=SimpleNamespace(n=(1,1,1)))
    return res,radius,sigma/(np.pi*radius**2)


def test_sphere_instrument_closes_independent_dipole_power_and_phase():
    result, radius, exact = dipole_result()
    measured = read_result(result, 2.25, radius)
    # Measured relative angular quadrature gap 6.7e-16 and exact forward closure.
    # The incident amplitude is deliberately imaginary and its entry plane has
    # a different phase, so omitting plane=0 cannot pass by normalization alone.
    for name in ('qsca', 'ffq', 'farfield_integral', 'optical'):
        assert abs(measured[name]/exact-1) < 2e-14
    assert abs(measured['optical_closure']) < 2e-14
    # At x=.001 the exact Mie angular pattern differs from the radiating dipole
    # by 5.4282e-7 of its peak. Swapping the two polarization cuts gives O(1).
    assert max(v['peak_error'] for v in measured['pattern'].values()) < 6e-7
    # The same Rayleigh fixture differs from Mie by 2.71406e-7 in complex
    # amplitude. Unlike optical power alone, this also checks the real-part
    # sign: -conj(f) keeps Im(f) and every intensity unchanged.
    assert measured['forward']['relative_error'] < 3e-7
    assert measured['incident_flux_ratio'] == 1.


def test_sphere_instrument_counts_inward_total_power_as_absorption():
    result, radius, _ = dipole_result()
    # Manufactured positive absorbed efficiency, independent of scattered power.
    exact = .125
    incident = result.sources['t'].intensity((1.,))[0]
    result['abs'].flux = np.array([-incident * np.pi * radius**2 * exact])
    assert read_result(result, 2.25, radius)['qabs'] == exact


@pytest.mark.parametrize('dimension', (2, 3))
def test_meep_face_normalization_integrates_isotropic_outward_radiation(dimension):
    # Cubic symmetry puts equal outward power on each of the 2*d faces. Their
    # reported signs are along +axis, not along the outward normal. Entry power
    # is intensity times length (2-D) or area (3-D), not total emitted power.
    total, intensity, half, radius = 2.4, 3., .7, .25
    fluxes = [-total/(2*dimension), total/(2*dimension)] * dimension
    expected = total/intensity
    if dimension == 3:
        expected /= np.pi * radius**2
    actual = scattering_efficiency(fluxes, intensity*(2*half)**(dimension-1),
                                   half, radius, dimension)
    assert actual == pytest.approx(expected, rel=2e-15)


@pytest.mark.parametrize('pol', ('TM', 'TE'))
def test_fdtdx_collocated_contour_integrates_independent_radial_power(pol):
    errors = []
    for cpw in (20, 40, 80):
        a = np.arange(-round(.55*cpw), round(.55*cpw)+1)/cpw
        y, z = np.meshgrid(a, a, indexing='ij')
        r = np.hypot(y, z)
        r = np.where(r > 0, r, 1.)
        u = np.exp(2j*np.pi*r)/np.sqrt(r)
        v = np.zeros((6,)+r.shape, dtype=complex)
        if pol == 'TE':
            v[3] = u
            v[1] = -u*z/r*(1+1j/(2*np.pi*r))
            v[2] = u*y/r*(1+1j/(2*np.pi*r))
        else:
            v[0] = u
            v[4] = u*z/r*(1+1j/(2*np.pi*r))
            v[5] = -u*y/r*(1+1j/(2*np.pi*r))
        errors.append(surface_power(v, 1/cpw)/np.pi-1)
    # Radial flux 1/(2r) integrates pi. Measured errors 4.3845e-4, 1.0962e-4,
    # 2.7403e-5 give order 2.0000. Budget covers the measured quadrature error.
    order = -np.polyfit(np.log((20, 40, 80)), np.log(abs(np.array(errors))), 1)[0]
    assert 1.99 < order < 2.01
    assert abs(errors[-1]) < 3e-5
