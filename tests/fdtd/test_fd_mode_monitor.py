"""``fdtd.ModeMonitor``: modal amplitudes of a run's fields, against exact and conserved answers.

The criteria are MM1-MM5 of ``docs/validation/fdtd/fd-waveguide-modes.md``, registered before
the monitor was first run; every limit here is quoted from there.
"""
import math
import warnings

import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.errors import WavebitError
from wavebit.units import C0


def _quiet():
    warnings.simplefilter('ignore')


def _problem(domain, center, boundaries, structures=(), background=1.0, **kw):
    return fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(domain=fdtd.Domain(size=domain, center=center),
                         background=Material(epsilon=background), boundaries=boundaries,
                         structures=tuple(structures)), **kw)


# ---------------------------------------------------------------------------------------------
# MM1: exact direction separation in Bloch guides
# ---------------------------------------------------------------------------------------------

N1, N2 = 2.0, 1.444


def _bloch_guide(kind, dx, k):
    z = fdtd.Bloch(k=k) if k else fdtd.Periodic()
    if kind in ('box', 'smoothed'):
        shape = (fdtd.SnappedBox(size=(1., .5, 10.)) if kind == 'box'
                 else fdtd.Box(size=(1.03, .51, 10.)))
        return _problem((3., 3., 8 * dx), (0., 0., 0.),
                        fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=z),
                        structures=(fdtd.Structure('core', shape=shape,
                                                   material=Material(epsilon=4.0)),))
    return _problem((9., dx, 8 * dx), (0., 0., 0.),
                    fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.InvariantAxis(), z=z),
                    structures=(fdtd.Structure('core', shape=fdtd.SnappedSlab(x=(-.75, .75)),
                                               material=Material(epsilon=N1 ** 2)),),
                    background=N2 ** 2)


def _numerics(dx, steps, wavelength):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=dx), duration=fdtd.Duration(steps=steps),
                         absorber=fdtd.PML(wavelength=wavelength))


def _launch(kind, lam, index, sign, span=None):
    """A Bloch run started from mode ``index`` travelling ``sign`` z, read on z = 0.

    Returns ``(launched, other_way, other_mode_max, energy, flux)`` from a ``ModeMonitor`` and a
    ``FluxMonitor`` whose planes take the transverse ``span`` (``{'x': (lo, hi)}``) if given.
    """
    _quiet()
    from wavebit.fdtd._kernel.grid import OFFSETS
    dx, steps, nz = 0.05, 200, 8
    modes = fdtd.solve_modes(_bloch_guide(kind, dx, 0.0), numerics=_numerics(dx, 1, lam),
                             plane=fdtd.Plane(z=0.), wavelength=lam, count=2)
    mode = modes[index]
    beta = sign * mode.beta_per_meter * 1e-6
    problem = _bloch_guide(kind, dx, beta)
    numerics = _numerics(dx, steps, lam)
    dt = fdtd.check(problem, numerics=numerics).resolved['time_step_seconds'] * C0 / 1e-6
    omega = 2 * math.pi / lam
    # A backward mode keeps E_t and H_z and flips H_t and E_z.
    flip = {'E': (1, 1, sign), 'H': (sign, sign, 1)}

    def volume(field, values):
        out = np.zeros((3, nz) + np.shape(values[field + 'x']), dtype=complex)
        for c, axis in enumerate('xyz'):
            z = -4 * dx + (np.arange(nz) + 0.5 * OFFSETS[field][c][2]) * dx
            out[c] = flip[field][c] * np.asarray(values[field + axis])[None] \
                * np.exp(1j * beta * z)[:, None, None]
        return out
    initial = fdtd.InitialField(E=volume('E', mode.E),
                                H=volume('H', mode.H) * np.exp(1j * omega * dt / 2))
    plane = fdtd.Plane(z=0., **(span or {}))
    outputs = fdtd.Outputs(monitors=(
        fdtd.ModeMonitor('m', plane=plane, wavelengths=(lam,), modes=2),
        fdtd.FluxMonitor('f', plane=plane, wavelengths=(lam,))))
    result = fdtd.run(problem.replace(initial=initial, outputs=outputs), numerics=numerics)
    m, flux = result['m'], result['f']
    launched = (m.forward if sign > 0 else m.backward)[0, index]
    other_way = (m.backward if sign > 0 else m.forward)[0, index]
    other_mode = max(abs(m.forward[0, 1 - index]), abs(m.backward[0, 1 - index]))
    energy = (m.forward_energy if sign > 0 else -m.backward_energy)[0, index]
    return launched, other_way, other_mode, energy, flux.spectral_energy[0]


@pytest.mark.parametrize('kind,lam', [('box', 1.0), ('slab', 1.55), ('smoothed', 1.0)])
@pytest.mark.parametrize('index', [0, 1])
@pytest.mark.parametrize('sign', [1, -1])
def test_mm1_a_launched_mode_is_read_back_in_its_own_direction_only(kind, lam, index, sign):
    launched, other_way, other_mode, energy, flux = _launch(kind, lam, index, sign)
    assert abs(launched) > 0
    assert abs(other_way) <= 1e-10 * abs(launched)
    assert other_mode <= 1e-10 * abs(launched)
    assert abs(energy - flux) <= 1e-10 * abs(flux)


@pytest.mark.parametrize('index', [0, 1])
@pytest.mark.parametrize('sign', [1, -1])
def test_mm6_a_finite_span_plane_reads_the_same_mode(index, sign):
    launched, other_way, other_mode, energy, flux = _launch('slab', 1.55, index, sign,
                                                           span={'x': (-3., 3.)})
    assert abs(launched) > 0
    assert abs(other_way) <= 1e-8 * abs(launched)
    assert other_mode <= 1e-8 * abs(launched)
    assert abs(energy - flux) <= 1e-8 * abs(flux)


# ---------------------------------------------------------------------------------------------
# MM2-MM4: a dipole in a hollow metal guide
# ---------------------------------------------------------------------------------------------

LAM2 = 1.05
PULSE = fdtd.GaussianPulse(wavelength=LAM2, frequency_sigma_hz=2e13)


def _hollow(block=None, periods=150, monitors=True, pulse=None, wavelength=None, steps=None):
    pulse = pulse or PULSE
    wavelength = wavelength or LAM2
    structures = ()
    if block is not None:
        structures = (fdtd.Structure('block', shape=fdtd.SnappedBox(size=(.4, .3, .5),
                                                                    center=(.4, .25, 1.25)),
                                     material=Material(epsilon=block)),)
    outs = ()
    if monitors:
        outs = tuple(fdtd.ModeMonitor('m' + s, plane=fdtd.Plane(z=z), wavelengths=(wavelength,),
                                      modes=7)
                     for s, z in (('+', 3.), ('-', -3.))) + tuple(
            fdtd.FluxMonitor('f' + s, plane=fdtd.Plane(z=z), wavelengths=(wavelength,))
            for s, z in (('+', 3.), ('-', -3.)))
    else:
        outs = (fdtd.ModeMonitor('m+', plane=fdtd.Plane(z=3.), wavelengths=(wavelength,), modes=1),)
    problem = _problem((1., .5, 8.), (.5, .25, 0.),
                       fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=fdtd.Open()),
                       structures=structures, background=2.25,
                       sources=(fdtd.CurrentDipole('d', center=(.3, .2, 0.), component='Ey',
                                                   pulse=pulse),),
                       outputs=fdtd.Outputs(monitors=outs))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
                             duration=(fdtd.Duration(steps=steps) if steps else
                                       fdtd.Duration(periods=periods, source='d')))
    return problem, numerics


@pytest.fixture(scope='module')
def hollow_runs():
    _quiet()
    import jax
    jax.config.update('jax_enable_x64', True)
    problem, numerics = _hollow()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        runs = {b: fdtd.run(problem, numerics=numerics, backend=b) for b in ('numpy', 'jax', 'torch')}
    return runs, [type(w.message).__name__ for w in caught]


@pytest.mark.xfail(strict=True, reason='registered MM2 failed: a broadband pulse in a closed metal '
                   'guide never settles (cutoff content is trapped); see MM2b and the record')
def test_mm2_seven_modes_carry_the_whole_flux(hollow_runs):
    runs, caught = hollow_runs
    assert 'UnsettledFieldWarning' not in caught, caught
    _complete(runs['numpy'])


def _complete(result):
    for side in '+-':
        m, f = result['m' + side], result['f' + side]
        modal = np.sum(np.asarray(m.forward_energy) - np.asarray(m.backward_energy), axis=1)
        flux = np.asarray(f.spectral_energy)
        assert np.all(np.abs(modal - flux) <= 1e-6 * np.abs(flux)), (modal, flux)


#: MM2b: a narrowband pulse inside the band where exactly seven modes propagate (discrete
#: cutoffs 1.0087 and 1.0644 um on this grid), so nothing is left at a cutoff.
LAM2B = 1.036
PULSE2B = fdtd.GaussianPulse(wavelength=LAM2B, frequency_sigma_hz=1.3e12, cutoff=1e-9)


@pytest.mark.slow
def test_mm2b_seven_modes_carry_the_whole_flux_of_a_settled_run():
    _quiet()
    problem, numerics = _hollow(periods=600, pulse=PULSE2B, wavelength=LAM2B)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fdtd.run(problem, numerics=numerics)
    names = [type(w.message).__name__ for w in caught]
    assert 'UnsettledFieldWarning' not in names, names
    _complete(result)


def test_mm3_backends_agree(hollow_runs):
    runs, _ = hollow_runs
    for side in '+-':
        ref = runs['numpy']['m' + side]
        scale = max(np.max(np.abs(np.asarray(ref.forward))), np.max(np.abs(np.asarray(ref.backward))))
        for b in ('jax', 'torch'):
            got = runs[b]['m' + side]
            for name in ('forward', 'backward'):
                diff = np.max(np.abs(np.asarray(getattr(got, name)) - np.asarray(getattr(ref, name))))
                assert diff <= 1e-12 * scale, (b, name, diff / scale)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason='registered MM4 failed: the central difference at the '
                   'registered step carries a truncation error of 9.6e-6 (> 1e-6); see MM4b')
def test_mm4_derivative_through_a_scatterer_matches_central_differences():
    _quiet()
    import jax
    import torch
    jax.config.update('jax_enable_x64', True)

    def energy(eps, backend):
        problem, numerics = _hollow(block=eps, steps=MM4_STEPS, monitors=False)
        return fdtd.run(problem, numerics=numerics, backend=backend)['m+'].forward_energy[0, 0]

    eps0, h = 3.0, 3.0e-4
    central = (float(energy(eps0 + h, 'numpy')) - float(energy(eps0 - h, 'numpy'))) / (2 * h)
    by_jax = float(jax.grad(lambda e: energy(e, 'jax'))(eps0))
    e = torch.tensor(eps0, dtype=torch.float64, requires_grad=True)
    energy(e, 'torch').backward()
    by_torch = float(e.grad)
    for got in (by_jax, by_torch):
        assert abs(got - central) <= 1e-6 * abs(central), (got, central)


def _mm4_energy(eps, backend):
    problem, numerics = _hollow(block=eps, steps=MM4_STEPS, monitors=False)
    return fdtd.run(problem, numerics=numerics, backend=backend)['m+'].forward_energy[0, 0]


@pytest.mark.slow
def test_mm4b_derivative_through_a_scatterer_matches_extrapolated_differences():
    _quiet()
    import jax
    import torch
    jax.config.update('jax_enable_x64', True)
    eps0, h = 3.0, 3.0e-4

    def central(step):
        return (float(_mm4_energy(eps0 + step, 'numpy'))
                - float(_mm4_energy(eps0 - step, 'numpy'))) / (2 * step)
    reference = (4 * central(h / 2) - central(h)) / 3
    by_jax = float(jax.grad(lambda e: _mm4_energy(e, 'jax'))(eps0))
    e = torch.tensor(eps0, dtype=torch.float64, requires_grad=True)
    _mm4_energy(e, 'torch').backward()
    by_torch = float(e.grad)
    for got in (by_jax, by_torch):
        assert abs(got - reference) <= 1e-7 * abs(reference), (got, reference)


#: The MM4 run length: the pulse's leading part has crossed the +z monitor. Reverse-mode
#: derivatives keep every step (about 3.5 MB a step on JAX here), so the run is kept short; the
#: derivative and the central difference are of the same truncated function.
MM4_STEPS = 800


# ---------------------------------------------------------------------------------------------
# MM5: refusals
# ---------------------------------------------------------------------------------------------

def _run_refused(code, problem, numerics, backend='numpy', at_check=True):
    """Refused by ``run``, and by ``check`` before any step when the refusal is static."""
    _quiet()
    report = fdtd.check(problem, numerics=numerics, backend=backend)
    assert [issue.code for issue in report.issues] == ([code] if at_check else []), report.issues
    with pytest.raises(WavebitError) as info:
        fdtd.run(problem, numerics=numerics, backend=backend)
    assert info.value.code == code, (info.value.code, str(info.value))


def test_mm5_a_nonuniform_monitor_plane_is_refused():
    problem, numerics = _hollow(block=3.0, periods=1, monitors=False)
    problem = problem.replace(outputs=fdtd.Outputs(monitors=(
        fdtd.ModeMonitor('m', plane=fdtd.Plane(z=1.0), wavelengths=(LAM2,)),)))
    _run_refused('mode.nonuniform', problem, numerics)


def test_mm5_a_traced_structure_on_the_plane_is_refused():
    import torch
    eps = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    problem, numerics = _hollow(block=eps, periods=1, monitors=False)
    problem = problem.replace(outputs=fdtd.Outputs(monitors=(
        fdtd.ModeMonitor('m', plane=fdtd.Plane(z=1.25), wavelengths=(LAM2,)),)))
    _run_refused('mode.traced', problem, numerics, backend='torch')


def test_mm5_more_modes_than_confined_are_refused():
    problem, numerics = _hollow(periods=1, monitors=False)
    problem = problem.replace(outputs=fdtd.Outputs(monitors=(
        fdtd.ModeMonitor('m', plane=fdtd.Plane(z=3.0), wavelengths=(LAM2,), modes=8),)))
    # The count is a per-wavelength solve, made when the result is adapted.
    _run_refused('mode.count', problem, numerics, at_check=False)


@pytest.mark.parametrize('modes', [0, -1, True, 1.5])
def test_mm5_modes_is_a_positive_integer(modes):
    with pytest.raises(WavebitError) as info:
        fdtd.ModeMonitor('m', plane=fdtd.Plane(z=0.), wavelengths=(1.,), modes=modes)
    assert info.value.code == 'monitor.modes'


def test_mm5_a_plane_outside_the_domain_is_refused():
    problem, numerics = _hollow(periods=1, monitors=False)
    problem = problem.replace(outputs=fdtd.Outputs(monitors=(
        fdtd.ModeMonitor('m', plane=fdtd.Plane(z=5.0), wavelengths=(LAM2,)),)))
    _quiet()
    with pytest.raises(WavebitError) as info:
        fdtd.run(problem, numerics=numerics)
    assert 'outside' in str(info.value)


# ---------------------------------------------------------------------------------------------
# Across an Open face
# ---------------------------------------------------------------------------------------------

def test_a_monitor_across_open_faces_reads_the_window_solve_modes_solves():
    """The window stops one cell inside an Open face's physical edge, for both, so the monitor's
    flux plane reads no absorbing cell and its modes are exactly the ones solve_modes returns."""
    _quiet()
    problem = _problem((9., .05, .4), (0., 0., 0.),
                       fdtd.Boundaries(x=fdtd.Open(), y=fdtd.InvariantAxis(), z=fdtd.Periodic()),
                       structures=(fdtd.Structure('core', shape=fdtd.SnappedSlab(x=(-.75, .75)),
                                                  material=Material(epsilon=N1 ** 2)),),
                       background=N2 ** 2,
                       initial=fdtd.InitialField(E=(None, lambda x, y, z: np.exp(-(x / .5) ** 2), None)),
                       outputs=fdtd.Outputs(monitors=(
                           fdtd.ModeMonitor('m', plane=fdtd.Plane(z=.1), wavelengths=(1.55,), modes=2),)))
    numerics = _numerics(.05, 20, 1.55)
    spectrum = fdtd.run(problem, numerics=numerics)['m']
    solved = fdtd.solve_modes(problem, numerics=numerics, plane=fdtd.Plane(z=.1), wavelength=1.55,
                              count=2)
    assert [m.effective_index for m in spectrum.modes[0]] == [m.effective_index for m in solved]
    assert solved.metadata['window']['x'] == (13, 191)   # physical cells [12, 192), one off each side
