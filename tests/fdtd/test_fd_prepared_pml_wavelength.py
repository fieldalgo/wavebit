"""Prepared reuse when the PML takes its tuning wavelength from the sources' pulse.

A ``PML`` without ``wavelength=`` tunes its complex-frequency shift to the band centre the
sources share, so that wavelength is fixed by ``prepare``. Reusing the preparation with a
changed concrete ``GaussianPulse.wavelength`` was refused with the generic
``prepared.dependencies`` at path ``problem``, and with a traced one (``jax.grad``,
``jax.jit``) with the catch-all ``fdtd.invalid``. Both are now ``prepared.pml_wavelength`` at
the pulse's path, and the message names the fix: an explicit ``PML(wavelength=...)``. The
reference is that fix itself -- the same reuse with an explicit PML wavelength, which runs,
matches a fresh ``fdtd.run`` and differentiates like a central difference.
"""
import warnings

import numpy as np
import pytest

from wavebit import Units, fdtd
from wavebit.errors import IncompatibleProblemError
from wavebit.units import C0

jax = pytest.importorskip('jax')
jnp = jax.numpy
jax.config.update('jax_enable_x64', True)


def problem(wavelength, x=None):
    x = fdtd.Open() if x is None else x
    return fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(domain=fdtd.Domain(size=(1.6,)*3),
                         boundaries=fdtd.Boundaries(x=x, y=fdtd.PEC(), z=fdtd.PEC())),
        sources=(fdtd.CurrentDipole('d', center=(0., 0., 0.), current_moment=1e-16,
                                    pulse=fdtd.GaussianPulse(wavelength=wavelength,
                                                             frequency_sigma_hz=.03*C0/1e-6)),),
        outputs=fdtd.Outputs(final_fields=True))


def numerics(pml_wavelength=None):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.1), duration=fdtd.Duration(steps=8),
                         absorber=fdtd.PML(cells=4, wavelength=pml_wavelength, allow_thin=True))


def energy(result):
    return (result.final_fields.values['E']**2).sum()


def prepare(pml_wavelength=None, backend='numpy', x=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fdtd.prepare(problem(1.0, x), numerics=numerics(pml_wavelength), backend=backend)


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def assert_refusal(error, live):
    assert error.code == 'prepared.pml_wavelength'
    assert error.path == 'sources.d.pulse.wavelength'
    text = str(error)
    assert 'prepared at 1.0' in text and live in text
    assert 'fdtd.PML(..., wavelength=1.0)' in text


def test_a_changed_concrete_pulse_wavelength_names_the_pml_tuning():
    prepared = prepare()
    with pytest.raises(IncompatibleProblemError) as caught:
        quiet(prepared.run, problem(1.02))
    assert_refusal(caught.value, '1.02')
    report = quiet(prepared.validate, problem(1.02))
    assert not report.ok and report.issues[0].code == 'prepared.pml_wavelength'
    assert report.issues[0].path == 'sources.d.pulse.wavelength'


@pytest.mark.parametrize('transform', ['grad', 'jit'])
def test_a_traced_pulse_wavelength_names_the_pml_tuning(transform):
    prepared = prepare(backend='jax')

    def f(w):
        return energy(quiet(prepared.run, problem(w)))
    run = jax.grad(f) if transform == 'grad' else jax.jit(f)
    with pytest.raises(IncompatibleProblemError) as caught:
        run(jnp.asarray(1.02))
    assert_refusal(caught.value, 'a traced value')


def test_the_unchanged_pulse_wavelength_still_reuses_the_inferred_tuning():
    prepared = prepare()
    reused = quiet(prepared.run, problem(1.0))
    fresh = quiet(fdtd.run, problem(1.0), numerics=numerics())
    np.testing.assert_array_equal(reused.final_fields.values['E'], fresh.final_fields.values['E'])


def test_the_advised_explicit_pml_wavelength_reuses_and_differentiates():
    inferred = prepare()
    explicit = prepare(pml_wavelength=1.0)
    # At the prepared centre the advice changes nothing: the same absorber, bit for bit.
    np.testing.assert_array_equal(quiet(explicit.run, problem(1.0)).final_fields.values['E'],
                                  quiet(inferred.run, problem(1.0)).final_fields.values['E'])
    # Away from it the explicit layer stays put and the run is a fresh run's.
    reused = quiet(explicit.run, problem(1.02)).final_fields.values['E']
    fresh = quiet(fdtd.run, problem(1.02), numerics=numerics(1.0)).final_fields.values['E']
    np.testing.assert_array_equal(reused, fresh)
    assert float(energy(quiet(explicit.run, problem(1.02)))) != float(
        energy(quiet(explicit.run, problem(1.0))))

    traced = prepare(pml_wavelength=1.0, backend='jax')
    gradient = float(jax.grad(lambda w: energy(quiet(traced.run, problem(w))))(jnp.asarray(1.02)))
    h = 1e-6
    central = (float(energy(quiet(explicit.run, problem(1.02 + h))))
               - float(energy(quiet(explicit.run, problem(1.02 - h)))))/(2*h)
    assert abs(central) > 0
    np.testing.assert_allclose(gradient, central, rtol=1e-5, atol=0)


def test_a_scene_without_a_pml_still_reuses_a_changed_pulse_wavelength():
    prepared = prepare(x=fdtd.PEC())
    reused = quiet(prepared.run, problem(1.02, fdtd.PEC())).final_fields.values['E']
    fresh = quiet(fdtd.run, problem(1.02, fdtd.PEC()), numerics=numerics()).final_fields.values['E']
    np.testing.assert_array_equal(reused, fresh)
