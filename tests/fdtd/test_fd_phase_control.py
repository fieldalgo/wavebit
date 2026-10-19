"""Independent source modes and power identities for the C3 phase diagnostic."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from studies.fdtd.subpixel_phase_control import fdtd_subpixel_phase_control as study
from wavebit.fdtd._legacy_results import SourceSeries
from wavebit.fdtd._kernel.grid import Grid
from wavebit.fdtd._prepare.monitors import box_faces


def manufactured_source(backend, cpw):
    """Raw records with specified analytic forward/backward phasors at z=0."""
    rt = study.runtime(backend.split('.')[0])
    dx, dt = 1/cpw, .5/cpw/np.sqrt(3)
    grid = Grid.from_domain((2*(.95+12*dx),)*3, (0.,)*3, dx)
    plan = NS(dx=dx, dt=dt, grid=grid)
    context = NS(rt=rt, plan=plan, scene=study.scene(cpw, True), term=NS(axis=2))
    plane = grid.low[2]+round((-.55-grid.low[2])/dx)*dx
    plane_H = plane-dx/2
    kappa = 2/dx*np.arcsin(dx/dt*np.sin(np.pi*dt))
    omega = 2*np.pi
    # Arbitrary nonzero complex modes, not empirical tolerances. Include a
    # backward wave and nonzero final samples to exercise both split and tail.
    A, B = 1.3-.7j, -.03+.04j
    wantE = A*(np.exp(1j*kappa*plane)+B*np.exp(-1j*kappa*plane))
    wantH = A*(np.exp(1j*kappa*plane_H)-B*np.exp(-1j*kappa*plane_H))
    t = dt*np.arange(1, 8)
    th = t-dt/2
    E = np.zeros((7, 3), complex)
    H = E.copy()
    E[-1, 0], H[-1, 1] = .07+.02j, -.05+.01j
    Omega = 2*np.sin(omega*dt/2)/dt
    for v, times, c, want, index in ((E, t, 0, wantE, 2), (H, th, 1, wantH, 3)):
        end = (dt*v[-1, c]*np.exp(1j*omega*times[-1])
               +1j/Omega*np.exp(1j*omega*(times[-1]+dt/2))*v[-1, c])
        v[index, c] = (want-end)*np.exp(-1j*omega*times[index])/dt
    src = SourceSeries('t', 'TFSF', t, rt.array(E, 'complex128'), th,
                       rt.array(H, 'complex128'), plane, plane_H, context=context)
    return src, plan, A, B, kappa


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'jax.jit', 'torch'))
def test_source_phase_and_power_recover_independent_modes(backend):
    for cpw in (20, 28, 40):
        src, plan, A, B, kappa = manufactured_source(backend, cpw)
        def transform(e, h):
            one = replace(src, incident=e, incident_H=h)
            return one.incident_amplitude((1.,), plane=0.), one.intensity((1.,))
        if backend == 'jax.jit':
            import jax
            transform = jax.jit(transform)
        amplitude, intensity = map(study.host, transform(src.incident, src.incident_H))
        audit = study.column_audit(NS(sources={'t': src}, plan=plan))
        # Preliminary 3-grid x 4-mode probe: max absolute amplitude/intensity/
        # independent split discrepancies 1.70e-15/4.45e-16/4.97e-16.
        assert abs(amplitude[0]-A) < 2e-14
        assert abs(intensity[0]-.5*abs(A)**2*np.cos(kappa*plan.dx/2)) < 2e-14
        assert abs(complex(*audit['A0'])-A) < 2e-14
        assert abs(complex(*audit['backward_ratio'])-B) < 2e-14


def test_cross_flux_equals_analytic_counterpropagating_mode_power():
    grid = Grid.from_domain((3.1,)*3, (0.,)*3, .05)
    faces = box_faces('plane modes', grid, study.fd.Region(size=(1.5,)*3))
    dt = .5*grid.dx/np.sqrt(3)
    kappa = 2/grid.dx*np.arcsin(grid.dx/dt*np.sin(np.pi*dt))
    # Specified incoming/outgoing amplitudes; no Mie or production transform
    # is used to obtain the expected interference power on either z face.
    Ai, Bi, As, Bs = 1.3-.7j, -.03+.04j, .17+.31j, -.23+.13j
    def modes(a, b):
        return {f: {label: a*v for label, v in part.items()}
                for f, part in study.incident_records(grid, faces, kappa, b/a).items()}
    inc, sca = modes(Ai, Bi), modes(As, Bs)
    for name, sign, layout in faces:
        if layout.P != 2:
            continue
        actual = study.mixed_flux(sca, inc, ((name, sign, layout),), grid.dx)
        expected = sign*1.5**2*np.cos(kappa*grid.dx/2)*np.real(As*np.conj(Ai)-Bs*np.conj(Bi))
        # Preliminary independent plane-mode probe: max 5.90e-17 absolute.
        assert abs(actual-expected) < 1e-15


@pytest.mark.slow
def test_fresh_vacuum_phase_and_weak_sphere_interference_energy():
    empty = study.fd.run(study.scene(20, True), backend='jax')
    result = study.fd.run(study.scene(20), backend='jax')
    line = study.line_audit(empty)
    surfaces, _ = study.surface_audit(result, empty, 1.1)
    # 20/28/40 grids, reflection-target change, and 24->48 periods measured
    # <=3.89e-7 vacuum two-wave discrepancy and <=8.65e-17 empty leakage.
    assert max(row['two_wave_gap'] for row in line['rows']) < 1e-6
    assert max(row['empty_field_max'] for row in surfaces) < 2e-15
    # Measured closure <=1.67e-4; the 24->48-period change reaches 2.46e-4.
    # A 5e-4 instrument budget covers twice that measured temporal change.
    # This is NOT the continuum optical-theorem gate or a C3 accuracy bound.
    assert max(abs(row['two_wave_closure']) for row in surfaces) < 5e-4
