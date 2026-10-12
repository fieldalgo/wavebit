"""Gate 1 of stage 2 -- the waveform table every source reads its time signal from.

What is pinned, each against something the engine did not compute:

* **The table is the closed form.** ``waveform_table`` is built inside the traced function
  from traced leaves; each row is ``A exp(-s^2/(2 tau^2)) sin(2 pi f0 s + phi)`` with
  ``s = t - t0 - delay``, sampled at the half step its slot reads it -- ``(n + 1/2) dt`` for an
  electric current (``before_E``), ``n dt`` for a magnetic one (``before_H``), ``(n + 1) dt``
  for a column's hard source. Checked row by row against ``fd.Band.waveform`` written out
  independently here, and through a run: a dipole's ``drive`` series on every backend.
* **The spectrum is below ``cutoff`` outside the band.** The DFT of the run's own drive
  series, over the whole pulse, reaches ``cutoff`` of its peak at ``f0 +- k df`` and stays
  under it beyond -- to within the floor the table's own start puts under every frequency
  (the envelope starts at ``cutoff``, a step worth 7e-5 of the peak spectrum, measured).
* **``Run(periods=)``** takes ``ceil(periods * wavelength / dt)`` steps.
* **Amplitude, phase and delay differentiate**, on JAX and Torch, against central differences
  of the NumPy run -- the table is built from the leaves inside the traced function, so a
  gradient reaches them.

**Mutant**: the delay applied with the wrong sign (``s = t - t0 + delay``), swapped into the
executor; the closed-form comparison sees it at O(1) for any nonzero delay, and a delay of
zero -- the case every other gate runs -- would not.
"""
import math

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd import _run
from wavebit.fd._kernel import sources as kernel_sources
from wavebit.fd._lower import lower
from wavebit.fd._runtime import runtime
from fd_helpers import backends, band, dipole_box, tfsf_box, to_numpy

#: Agreement of the table with the closed form: the same arithmetic in a different order.
TABLE_TOL = 1e-13


def closed_form(t, wavelength, df, cutoff=1e-3, amplitude=1.0, phase=0.0, delay=0.0):
    """``w(t)`` written from the definition, not from ``fd.Band.waveform``."""
    k = math.sqrt(2.0 * math.log(1.0 / cutoff))
    tau = 1.0 / (2.0 * math.pi * df)
    s = np.asarray(t) - k * tau - delay
    return amplitude * np.exp(-0.5 * s ** 2 / tau ** 2) * np.sin(2.0 * math.pi * s / wavelength
                                                                   + phase)


def _two_source_scene(amplitude=0.7, phase=0.3, delay=0.4):
    """An electric and a magnetic dipole and a TFSF box: all three row shifts in one table."""
    b = band()
    scene = tfsf_box(steps=40, src_band=b, amplitude=1.1, phase=-0.2, delay=0.1)
    extra = (fd.Dipole('e', center=(0.3, 0.3, 0.3), component='Ez', band=b,
                       amplitude=amplitude, phase=phase, delay=delay),
             fd.Dipole('m', center=(-0.3, 0.3, -0.3), component='Hy', kind='magnetic',
                       band=fd.Band(wavelength=1.0, df=0.1, cutoff=1e-4), amplitude=-amplitude,
                       phase=-phase, delay=2 * delay))
    return scene.replace(sources=scene.sources + extra)


def test_the_table_is_the_closed_form_row_by_row():
    """Every row equals ``w`` at its own half step: 0.5 for J, 0.0 for M, 1.0 for a column."""
    scene = _two_source_scene()
    plan = fd.plan(scene)
    rt = runtime('numpy')
    W = np.asarray(kernel_sources.waveform_table(plan, lower(scene, plan, rt), rt))
    n = np.arange(plan.steps)
    shifts = {'t': 1.0, 'e': 0.5, 'm': 0.0}
    rows = [t for t in plan.terms if getattr(t, 'rows', ())]
    assert [t.name for t in rows] == ['t', 'e', 'm'] and W.shape == (3, plan.steps)
    for term, source in zip(rows, scene.sources):
        assert term.rows == (shifts[term.name],)
        b = source.band
        want = closed_form((n + shifts[term.name]) * plan.dt, b.wavelength, b.df, b.cutoff,
                           source.amplitude, source.phase, source.delay)
        assert np.max(np.abs(W[term.row] - want)) <= TABLE_TOL * np.max(np.abs(want))
        # and the public closed form agrees with the one written here
        assert np.allclose(b.waveform((n + shifts[term.name]) * plan.dt, source.amplitude,
                                      source.phase, source.delay), want, rtol=0, atol=1e-15)


@pytest.mark.parametrize('backend', backends())
def test_a_dipoles_drive_series_is_the_closed_form_on_every_backend(backend):
    """Through a run: ``res.sources['d'].drive`` at ``t`` equals ``w(t)``, on each backend."""
    scene = dipole_box(size=0.6, steps=120, amplitude=1.7, phase=0.9, delay=0.35)
    res = fd.run(scene, backend=backend)
    s = res.sources['d']
    assert np.allclose(s.t, (np.arange(120) + 0.5) * res.plan.dt, rtol=0, atol=1e-15)
    want = closed_form(s.t, 1.0, 0.15, amplitude=1.7, phase=0.9, delay=0.35)
    got = to_numpy(s.drive)
    assert np.max(np.abs(got - want)) <= TABLE_TOL * np.max(np.abs(want))


def test_the_envelope_starts_at_the_cutoff():
    """At ``t = 0`` with no delay the envelope is exactly ``cutoff`` of its peak."""
    for cutoff in (1e-3, 1e-6):
        b = fd.Band(wavelength=0.55, df=0.3, cutoff=cutoff)
        tau = 1.0 / (2.0 * math.pi * b.df)
        k = b.sigmas
        assert math.isclose(math.exp(-0.5 * k ** 2), cutoff, rel_tol=1e-12)
        assert math.isclose(k * tau, b.sigmas / (2 * math.pi * 0.3), rel_tol=1e-15)
        assert math.isclose(b.shortest_wavelength, 1.0 / (1 / 0.55 + k * 0.3), rel_tol=1e-15)
        assert math.isclose(b.longest_wavelength, 1.0 / (1 / 0.55 - k * 0.3), rel_tol=1e-15)


def test_the_spectrum_is_below_the_cutoff_outside_the_band():
    """The DFT of the whole drive series, beyond ``f0 +- k df``, stays under ``cutoff``.

    The series is the run's own (``res.sources['d'].drive``) and covers the whole pulse --
    ``2 t0`` plus a margin -- so the transform is of what the grid was driven with. The table
    starts at ``t = 0`` with the envelope at ``cutoff``, and that truncation lays a floor of
    ~7e-5 of the peak under every frequency (measured: 6.7e-5 at 0.02 f0, 1.8e-5 at 3 f0), so
    the edges read 1.06e-3 rather than 1.00e-3; half a ``df`` beyond them the spectrum is under
    ``cutoff`` outright, and a ``df`` beyond them it is at that floor.
    """
    b = band()
    tau = 1.0 / (2 * math.pi * 0.15)
    scene = dipole_box(size=0.6, steps=10, src_band=b)
    dt = fd.plan(scene).dt
    steps_needed = int(math.ceil((2 * b.sigmas * tau + 2.0) / dt))
    res = fd.run(scene.replace(run=fd.Run(steps=steps_needed)))
    s = res.sources['d']
    w, t = to_numpy(s.drive), s.t
    f0, k = 1.0, b.sigmas
    lo, hi = f0 - k * 0.15, f0 + k * 0.15

    def spectrum(f):
        return np.abs(np.exp(2j * np.pi * np.outer(f, t)) @ w) * dt

    peak = spectrum(np.array([f0]))[0]
    assert math.isclose(peak, tau * math.sqrt(2 * math.pi) / 2, rel_tol=1e-3)
    floor = 1e-4
    edges = spectrum(np.array([lo, hi])) / peak
    assert np.all(np.abs(edges - b.cutoff) < floor), edges
    outside = np.concatenate([np.linspace(0.02, lo, 200), np.linspace(hi, 3.0 * f0, 400)])
    assert (spectrum(outside) / peak).max() <= b.cutoff + floor
    beyond = np.concatenate([np.linspace(0.02, lo - 0.075, 200),
                             np.linspace(hi + 0.075, 3.0 * f0, 400)])
    assert (spectrum(beyond) / peak).max() <= b.cutoff
    far = np.concatenate([np.linspace(0.02, lo - 0.15, 200), np.linspace(hi + 0.15, 3.0, 400)])
    assert (spectrum(far) / peak).max() <= floor


@pytest.mark.filterwarnings('ignore::wavebit.fd.CoarseMeshWarning')
def test_run_periods_gives_the_step_count():
    """``Run(periods=p)`` is ``ceil(p * wavelength / dt)`` steps, from the band centre."""
    for periods, wavelength in ((10, 1.0), (7.5, 0.8), (40, 1.2)):
        scene = dipole_box(size=0.6, periods=periods, src_band=band(wavelength=wavelength),
                           cells_per_wavelength=20)
        plan = fd.plan(scene)
        assert plan.steps == math.ceil(periods * wavelength / plan.dt - 1e-9)
        assert (plan.steps - 1) * plan.dt < periods * wavelength <= plan.steps * plan.dt + 1e-12


def _objective(backend, amplitude, phase, delay):
    scene = dipole_box(size=0.6, steps=120, amplitude=amplitude, phase=phase, delay=delay,
                       monitors=(fd.Record('p', at=(0.15, 0.1, 0.05), component='Ez'),))
    res = fd.run(scene, backend=backend)
    return (res['p'].values ** 2).sum()


X0 = (1.3, 0.4, 0.2)


def _central_differences(h=1e-5):
    out = []
    for i in range(3):
        up, down = list(X0), list(X0)
        up[i] += h
        down[i] -= h
        out.append((float(_objective('numpy', *up)) - float(_objective('numpy', *down))) / (2 * h))
    return np.array(out)


@pytest.fixture(scope='module')
def reference_gradient():
    return _central_differences()


@pytest.mark.parametrize('backend', [b for b in backends() if b.values[0] != 'numpy'])
def test_amplitude_phase_and_delay_differentiate(backend, reference_gradient):
    """d/d(amplitude, phase, delay) of a probe's energy, against central differences."""
    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        runtime('jax')                  # x64 lifted before the leaves are built, or they are
                                        # float32 and the gradient loses seven digits
        grad = jax.grad(lambda a, p, d: _objective('jax', a, p, d), argnums=(0, 1, 2))(
            *[jnp.asarray(v) for v in X0])
        got = np.array([float(g) for g in grad])
    else:
        import torch
        leaves = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in X0]
        _objective('torch', *leaves).backward()
        got = np.array([float(v.grad) for v in leaves])
    assert np.all(np.abs(got) > 1e-3), got                  # every leaf actually reaches it
    assert np.allclose(got, reference_gradient, rtol=1e-6, atol=0), (got, reference_gradient)


def test_mutant_the_delay_with_the_wrong_sign_is_seen(monkeypatch):
    """The table with ``s = t - t0 + delay`` fails the closed form; with no delay it would not."""
    def mutated(plan, params, rt):
        flipped = {name: dict(p, delay=-p['delay']) for name, p in params['sources'].items()}
        return kernel_sources.waveform_table(plan, dict(params, sources=flipped), rt)

    scene = dipole_box(size=0.6, steps=120, delay=0.35)
    monkeypatch.setattr(_run, 'waveform_table', mutated)
    s = fd.run(scene).sources['d']
    want = closed_form(s.t, 1.0, 0.15, delay=0.35)
    assert np.max(np.abs(to_numpy(s.drive) - want)) > 0.1 * np.max(np.abs(want))
    s0 = fd.run(dipole_box(size=0.6, steps=120, delay=0.0)).sources['d']
    want0 = closed_form(s0.t, 1.0, 0.15)
    assert np.max(np.abs(to_numpy(s0.drive) - want0)) <= TABLE_TOL * np.max(np.abs(want0))
