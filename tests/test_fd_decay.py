"""Gate 5 of stage 3 -- the decay diagnostic, and ``Run(time=)``.

Every run reports, per frequency-domain monitor, how much the energy ``sum |F|^2`` in its region
still changes over the run's last window (one period of the longest wavelength it asks for),
over that energy's peak (``wavebit/fd/_post.py::decay_ratio``); above
``_assemble.UNSETTLED_THRESHOLD = 1e-5`` a ``fd.UnsettledFieldWarning`` names the monitor and the
number. What is pinned here:

* **A cavity stopped early warns, by name and number.** A dipole in a closed PEC box
  (``fd_helpers.pec_cavity_dipole``) rings forever: its flux box's ratio is 0.98 after 3 periods
  and 0.23 after 6 (measured), and the warning's text carries ``monitor 'in'`` and the ratio
  exactly as ``res.diagnostics['decay']`` holds it.
* **A settled PML box does not.** The same dipole in an open box (0.8 of interior, PML all round,
  ``fd_helpers.flux_box_dipole``) at 10 periods: every ratio <= 3.1e-9 (measured), no warning.
* **The line means what it says.** The ratio is read against the truncation error it is a proxy
  for: the flux of each box at 8 and at 10 periods against a 14-period run of the same scene.
  Measured (NumPy, 20 cells per wavelength), error relative to the dipole's power:

      periods   ratio / flux error: 'in'    'empty'            'big'
      8         2.0e-4 / 1.5e-4             6.0e-4 / 1.2e-5    1.8e-3 / 7.4e-4
      10        1.8e-9 / 1.3e-7             3.1e-9 / 8.5e-9    2.2e-9 / 9.6e-8

  so a run that warns is off by up to its ratio (the empty box's flux is zero, so its error is
  small against the power whatever its ratio), and one that does not is off by less than the
  line. The dipole's own ``power()`` at 10 periods is the 14-period one to 2.4e-8.
* **``Run(time=T)`` takes ``ceil(T / dt)`` steps**: the last E sample is at or just past ``T``
  and never a whole step past it, a ``T`` on a step boundary takes exactly that many, and the
  run is bit-identical to ``Run(steps=)`` with the same count; ``Run(periods=P)`` is
  ``Run(time=P * wavelength)``.
* **Under ``jax.grad``** the energy series come back concrete (they are off the differentiated
  path), so the ratio is read and warns exactly as NumPy's does; under ``jax.jit`` they are
  tracers, the ratio is ``None`` and nothing warns. (``_assemble.decay_report``'s docstring says
  ``None`` under ``jax.grad`` too; that is only true under a trace that has no values.)

**Mutant.** The diagnostic reading the run's *first* window instead of its last
(``decay_ratio`` over ``energy[:window]``): the cavity then reads 3.7e-4 (the pulse's onset) and
no longer exceeds 1e-2, and the settled box reads 8e-5 to 1.7e-4 and warns -- both gates catch
it.

**A kernel defect, pinned as ``xfail(strict=True)``** (slow): a lossless lamellar ``Box``
grating under the default ``PML(cells=12, reflection=1e-8)`` does not settle -- its late field
*grows* (a probe below the tooth, p at normal incidence: 8.4e-4 at 192 periods, 1.7e-3 at 288,
2.6e-3 at 384, and Build B's 960-period run reaches 5.9e-2; an e-fold every ~180 periods). The
late spectrum sits at the ``(+-1, 0)`` Rayleigh wavelengths inside the source band. A passive
lossless scene cannot gain energy, so this is the CPML's late-time behaviour at a grazing,
near-zero-group-velocity frequency, not physics; with ``PML(cells=24, reflection=1e-12)`` the
same probe falls instead (4.8e-4 -> 3.0e-4, the control below). Until the CPML is fixed the
decay diagnostic reports it as an unsettled monitor, which is right.
"""
import functools
import math
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd import _post
from wavebit.fd._assemble import UNSETTLED_THRESHOLD
from fd_helpers import box_grating, dipole_box, flux_box_dipole, pec_cavity_dipole, to_numpy

#: The open box every settled-side test runs: small enough to take seconds.
OPEN = dict(size=0.8, box=0.3, empty_center=(0.2, 0.2, 0.15), empty_size=0.15)


def _run_collecting(scene, backend='numpy'):
    """``(res, [UnsettledFieldWarning, ...])``; every other warning is silenced."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('ignore')
        warnings.simplefilter('always', fd.UnsettledFieldWarning)
        res = fd.run(scene, backend=backend)
    return res, [w for w in caught if issubclass(w.category, fd.UnsettledFieldWarning)]


@functools.lru_cache(maxsize=None)
def _open(periods):
    """The open box run for ``periods``: ``(res, warnings)``, run once per session."""
    return _run_collecting(flux_box_dipole(periods=periods, **OPEN))


# ---------------------------------------------------------------------------------------------
# The warning
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('periods', [3, 6])
def test_a_cavity_stopped_early_warns_naming_the_monitor_and_its_number(periods):
    res, caught = _run_collecting(pec_cavity_dipole(periods=periods))
    ratio = res.diagnostics['decay']['in']
    assert ratio > 1e-2, ratio                      # a lossless cavity: measured 0.98 / 0.23
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "monitor 'in'" in message
    assert f'{ratio:.2e}' in message                # the number, as the diagnostic holds it
    assert f'{UNSETTLED_THRESHOLD:.0e}' in message  # and the line it is over
    assert res.diagnostics['threshold'] == UNSETTLED_THRESHOLD == 1e-5


def test_a_settled_pml_box_does_not_warn():
    res, caught = _open(10)
    assert caught == []
    decay = res.diagnostics['decay']
    assert set(decay) == {'in', 'empty', 'big'}
    assert all(0.0 <= v < 1e-7 for v in decay.values()), decay     # measured <= 3.1e-9


@pytest.mark.slow
def test_the_line_separates_a_cut_off_flux_from_a_settled_one():
    """A run that warns is off by about its ratio; one that does not, by less than the line."""
    ref = _open(14)[0]
    power = np.abs(to_numpy(ref.sources['d'].power()))
    for periods in (8, 10):
        res, caught = _open(periods)
        warned = {str(w.message).split("'")[1] for w in caught}
        for name in ('in', 'empty', 'big'):
            ratio = res.diagnostics['decay'][name]
            error = float(np.max(np.abs(to_numpy(res[name].flux) - to_numpy(ref[name].flux))
                                 / power))
            if periods == 8:
                assert ratio > UNSETTLED_THRESHOLD and name in warned, (name, ratio)
                assert error < 2 * ratio, (name, ratio, error)
                if name != 'empty':                 # 'in' 0.75x its ratio, 'big' 0.41x
                    assert 0.1 * ratio < error, (name, ratio, error)
            else:
                assert ratio < UNSETTLED_THRESHOLD and name not in warned, (name, ratio)
                assert error < UNSETTLED_THRESHOLD, (name, error)
    # The settled run's power is the long run's too: the source series settles with the field.
    power_10 = to_numpy(_open(10)[0].sources['d'].power())
    assert np.max(np.abs(power_10 - to_numpy(ref.sources['d'].power())) / power) < 1e-6


def _short_box(amplitude=1.0):
    """A dipole cut off at 3 periods in a small open box: its flux box has not settled."""
    return dipole_box(size=0.4, periods=3, amplitude=amplitude, monitors=(
        fd.Flux('in', box=fd.Box(size=(0.3,) * 3), wavelengths=(1.0,)),))


def test_a_run_under_jax_grad_reads_the_values_and_under_jit_reports_none():
    """Under a bare ``jax.grad`` the energy series come back concrete (they are not on the
    differentiated path), so the diagnostic reads them and warns exactly as NumPy does; under
    ``jax.jit`` they are tracers, the ratio is ``None`` and nothing warns."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    import wavebit
    wavebit.enable_x64()
    reference, caught = _run_collecting(_short_box())
    assert len(caught) == 1

    def flux(amplitude):
        res = fd.run(_short_box(amplitude), backend='jax')
        seen.append(res.diagnostics['decay']['in'])
        return jnp.sum(jnp.real(res['in'].flux))

    seen = []
    with warnings.catch_warnings(record=True) as grad_caught:
        warnings.simplefilter('ignore')
        warnings.simplefilter('always', fd.UnsettledFieldWarning)
        g = jax.grad(flux)(jnp.asarray(1.0, dtype=jnp.float64))
    assert np.isfinite(float(g))
    assert abs(seen[-1] - reference.diagnostics['decay']['in']) < 1e-12
    assert [str(w.message) for w in grad_caught
            if issubclass(w.category, fd.UnsettledFieldWarning)] == [str(caught[0].message)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        warnings.simplefilter('error', fd.UnsettledFieldWarning)
        g_jit = jax.jit(jax.grad(flux))(jnp.asarray(1.0, dtype=jnp.float64))
    assert seen[-1] is None
    assert abs(float(g_jit) - float(g)) < 1e-12 * abs(float(g))


# ---------------------------------------------------------------------------------------------
# Run(time=)
# ---------------------------------------------------------------------------------------------

def _record_scene(run):
    return dipole_box(size=0.4, monitors=(fd.Record('p', at=(0.1, 0.0, 0.0), component='Ez'),),
                      steps=1).replace(run=run)


@pytest.mark.parametrize('time', [0.37, 1.0, 2.5])
def test_run_time_steps_to_the_first_e_sample_at_or_past_the_end_time(time):
    res = fd.run(_record_scene(fd.Run(time=time)))
    dt = res.plan.dt
    assert res.plan.steps == math.ceil(time / dt)
    t = np.asarray(res['p'].t)
    assert len(t) == res.plan.steps
    assert time <= t[-1] + 1e-12 and t[-1] - dt < time      # at or past, never a step past
    same = fd.run(_record_scene(fd.Run(steps=res.plan.steps)))
    assert np.array_equal(to_numpy(res['p'].values), to_numpy(same['p'].values))


def test_run_time_on_a_step_boundary_takes_exactly_that_many_steps():
    dt = fd.plan(_record_scene(fd.Run(steps=1))).dt
    for n in (1, 7, 100):
        assert fd.plan(_record_scene(fd.Run(time=n * dt))).steps == n


def test_run_periods_is_run_time_of_that_many_wavelengths():
    wavelength = _record_scene(fd.Run(steps=1)).sources[0].band.concrete_wavelength
    assert wavelength == 1.0
    for periods in (0.5, 3.0, 8.25):
        a = fd.plan(_record_scene(fd.Run(periods=periods)))
        b = fd.plan(_record_scene(fd.Run(time=periods * wavelength)))
        assert a.steps == b.steps


# ---------------------------------------------------------------------------------------------
# The mutant: the first window read instead of the last
# ---------------------------------------------------------------------------------------------

def _first_window(energy, window):
    e = np.asarray(energy, dtype=np.float64)
    peak = float(np.max(e)) if e.size else 0.0
    if not peak > 0.0:
        return 0.0
    head = e[:max(1, min(int(window), e.size))]
    return float((np.max(head) - np.min(head)) / peak)


def test_mutant_the_first_window_is_caught_by_both_sides(monkeypatch):
    monkeypatch.setattr(_post, 'decay_ratio', _first_window)
    res, _ = _run_collecting(pec_cavity_dipole(periods=3))
    assert res.diagnostics['decay']['in'] < 1e-2            # the cavity gate would fail
    res, caught = _run_collecting(flux_box_dipole(periods=10, **OPEN))
    assert caught, 'the settled box would warn under the mutant'   # the settled gate would fail
    assert max(res.diagnostics['decay'].values()) > UNSETTLED_THRESHOLD


# ---------------------------------------------------------------------------------------------
# A kernel defect: late-time growth under the default PML at in-band Rayleigh frequencies
# ---------------------------------------------------------------------------------------------

def _grating_probe(cells, reflection, periods=384):
    """``(peak over the period ending at half the run, peak over the last period)`` of a probe
    below a lamellar tooth, p at normal incidence, the z faces ``PML(cells, reflection)``."""
    scene, _ = box_grating('lamellar', polarization='p', periods=periods, wavelengths=(1.0,))
    scene = scene.replace(
        monitors=(fd.Record('b', at=(0.3, 0.0, -0.1), component='Ez'),),
        boundaries=fd.Boundaries(x=scene.boundaries.face('x-'), y=scene.boundaries.face('y-'),
                                 z=fd.PML(cells=cells, reflection=reflection)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = fd.run(scene)
    v = np.abs(to_numpy(res['b'].values))
    per = int(round(1.0 / res.plan.dt))
    half = len(v) // 2
    return float(v[half - per:half].max()), float(v[-per:].max())


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    'kernel defect (wavebit/fd CPML, stage 1): a lossless lamellar Box grating under the '
    'default PML(12, 1e-8) grows at late time at its in-band (+-1, 0) Rayleigh wavelengths '
    '(probe 8.4e-4 at 192 periods -> 2.6e-3 at 384); PML(24, 1e-12) decays instead'))
def test_a_lossless_grating_under_the_default_pml_does_not_grow():
    middle, last = _grating_probe(12, 1e-8)
    assert last <= middle, (middle, last)


@pytest.mark.slow
def test_the_same_grating_under_a_24_cell_1e_12_pml_does_not_grow():
    middle, last = _grating_probe(24, 1e-12)
    assert last <= middle, (middle, last)                   # measured 4.8e-4 -> 3.0e-4
