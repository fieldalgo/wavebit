"""Gate 5 of stage 3 -- the decay and growth diagnostics, and ``Run(time=)``.

Every run reports, per frequency-domain monitor, how much the energy ``sum |F|^2`` in its region
still changes over the run's last window (one period of the longest wavelength it asks for),
over that energy's peak (``src/wavebit/fdtd/_post.py::decay_ratio``); above
``_assemble.UNSETTLED_THRESHOLD = 1e-5`` a ``fd.UnsettledFieldWarning`` names the monitor and the
number and says to run longer. It also reports whether the energy of the scene's **interior**
(every cell but the PML layers', ``src/wavebit/fdtd/_kernel/observe.py::InteriorEnergy``) still
**rises** after every source has ended (``_post.growth_ratio``: its mean over the last quarter
of that span over its mean over the quarter before, ``res.diagnostics['interior_growth']``,
and how steadily its last half rose, ``interior_spread``); above
``_assemble.GROWTH_THRESHOLD = 1.1``, rising as an exponential does (a spread under
``_assemble.GROWTH_STEADY = 2``), every monitor that sees it warns
``fd.GrowingFieldWarning`` instead, which says a longer run makes it worse and names the PML's
``stabilize=`` knob (``tests/fdtd/test_fd_pml_stability.py`` pins the message on the slab column).
The scene is judged and not a monitor, because a passive scene's interior energy cannot rise
while a monitor's region energy can: an echo or a slow beat moves energy through it. What is
pinned here:

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
* **A run cut off early is told to run longer, and nothing else.** A dipole stopped at 3
  periods, before its own pulse has ended, warns the plain ``UnsettledFieldWarning`` with no word
  of ``stabilize`` whatever the layer, and its growth is not judged (``None``). A closed PEC
  cavity at 40 periods is unsettled forever and never growing: its interior reads 0.99999, and
  its sampled energy holds to 3.2e-3 after the dipole.
* **What the growth statistic separates**, on series built here: an energy that holds (1, its
  ripple and all), a ring-down (``e^-4`` over a quarter), one under ``GROWTH_FLOOR`` and noise
  there (not judged), a mode growing under a ring-down (its exact ``e^2`` per quarter), one
  growing from the first sample with no dip at all (``e^2`` too -- a first version could not
  judge a series with no first peak), the quarters of the span after the sources and not of the
  run (pinned by their bounds), and a span half a period short of ``GROWTH_WINDOWS`` (not
  judged).
  An arrival is not a growth: a pulse arriving over a quiet floor steepens as it comes (spread
  4.2), one that has arrived and holds does not rise throughout (``inf``).
* **Passive scenes that fooled a per-monitor statistic do not warn**: behind a thick glass slab
  a monitor's own series rises 1e6-fold from quarter to quarter (the second echo over a quiet
  gap) while the interior reads 0.40. A second dipole fired at 38 of 50 periods leaves 4.1
  periods after it, over which the interior holds (1.00004, the static residue).
* **Passive scenes that fooled the first version of this one do not warn either** (the review of
  2026-09-20). A TFSF box over glass 3.5 below it: the column's echo comes back up at
  t = 9 .. 13 and the box re-injects it -- 4.04 from the closed-form end, 129 from the envelope's
  -- so a PlaneWave's or TFSF's end is read off its own column (15.6 there; 16.0 with a glass
  layer inside the box; 17.7 for a PlaneWave column ended by ``Absorber(12)``). A pulse kicked
  into ``Absorber(32)`` faces: the layer's return re-enters an empty interior (2.57 with the
  layer left out), so the Absorber's cells are judged too, and the energy only falls. A deep
  weak ``PML(64, 1e-3)`` hands back 1e-3 of the pulse after a quiet gap -- 984 by the ratio,
  but it arrives, spread 8.2, and nothing warns. A strong lossless Lorentz (40 ``omega0^2``,
  the default courant) and Drude (6 ``omega``, courant 0.9) cavity read 1.47 and 1.20 with the
  first version's pole reading; now 1.0002 and 0.99997.
* **What is read is the leapfrog's own conserved energy**, ``eps E^n . E^n + mu H^(n-1/2) .
  H^(n+1/2)``, read around the H update from the two H's (or, where the update is in place,
  from the curl it is about to use; ``InteriorEnergy``). Every step of a closed lossless box
  holds it to 1e-12: PEC, PMC and ``Symmetry`` walls (both parities on both kinds of face,
  three mirrors meeting at edges and a corner, the plane's own components counted at half
  weight), a Bloch box of complex field, random grid-scale data, courant 0.5 and 0.9, on
  NumPy, JAX and Torch alike. The stride's two alias modes -- which the first version's H pair
  read as 1.165 and 1.38 -- read 1.0000 and never warn. An ``eps = (2, 3, 5)`` slab flush
  against a mirror plane holds it too (4.7e-15 .. 7.5e-15), which is the one place the plane's
  per-component weight is a number of its own.
* **A term's drive is in that reading, in both writings of it.** NumPy takes its H half off the
  curl the update is about to use, so a source's contribution to the curl is in the product
  exactly as the update applies it; JAX and Torch take it off the two fields after. A
  **magnetic** dipole is the scene that separates them -- an electric one drives ``before_E``
  -- and there the three backends' every-step series agree to 1.5e-15 of the run's peak, where
  NumPy reading the bare curl is out by 11%.
* **A pole's stored energy is the interior's too, exactly.** ``T^n = F^n + V^(n+1/2) - E^n .
  (P^(n+1) - P^n)``, read by the observer itself at every step, is conserved to 3e-15 for
  strong Lorentz and Drude poles at coarse ``dt``, and a Debye's or a damped Lorentz's never
  rises; the first version's reading (half a step late, no coupling) and the coupling dropped
  swing 0.80, and Debye read a step late rises within a step. Through ``fd.run`` a lossless
  Lorentz or Drude cavity holds to 4e-14 (the field's alone swings 1.6e4-fold and 4.3-fold),
  and a Debye cavity never rises (with the Debye energy left out, 1.8e-2 in one row). On JAX
  the pole coefficients are tracers, so both kinds are computed and a ``where`` picks: the same
  series as NumPy's static path, to 2e-15.
* **The weights are ``eps`` and ``mu``, in a lossy cell too**: 4 inside a ``Conductor(eps=4)``,
  1 through an Absorber; a closed column half filled with ``eps = 12`` holds to round-off and
  swings 6.8 with weights of one. The judged cells are every cell but a PML's (pinned on a
  column with a PML below and an Absorber above).
* **A column that never goes quiet does not hide a growth** (the review of 2026-09-20). The
  growing slab column of ``test_fd_pml_stability.py`` driven by an oblique ``PlaneWave``: its
  column holds an evanescent field the stretch keeps, whose one-step change stays at 1.8e-8 of
  its peak -- just over ``COLUMN_LEVEL`` -- so the column end was the run's end and a scene
  growing 14000-fold said only "run longer". A column that holds is judged from its envelope's
  end instead (4.57 at t = 200, warning), one that grows makes the scene growing by itself,
  and one that is still ringing down or passing an echo keeps its column end
  (``_assemble._column_state``; the TFSF and Absorber scenes above are that case). Both of
  that rule's constants are pinned by value: ``_round_trip`` against its closed form
  ``2 L / v`` (one crossing fails it, and a built echo says why -- over quarters of a crossing
  it reads growing at 22.6, spread 1.002, where over the round trip it is not judged until it
  has come and gone), and ``'holding'`` against ``GROWTH_THRESHOLD`` (a column whose eighth
  means bump by 1.40 is not holding, though a band of ``GROWTH_STEADY`` would take it).
* **An overflow is growth**: an interior energy past float64's range warns ``GrowingFieldWarning``
  once, saying so, and ``interior_growth`` is ``inf`` -- unless the stability guard voided the run.

**Mutant.** The diagnostic reading the run's *first* window instead of its last
(``decay_ratio`` over ``energy[:window]``): the cavity then reads 3.7e-4 (the pulse's onset) and
no longer exceeds 1e-2, and the settled box reads 8e-5 to 1.7e-4 and warns -- both gates catch
it.

**The growth mutants.** The statistic judging the whole run instead of the span after every
source (``sources_end`` 0): the late second dipole then lands in the last quarter over a third
quarter where the first had long left (2.9e5), and the passive gate fails on its ratio (it no
longer warns: the pulse arrives, spread over the line). The ratio alone, the spread ignored:
the deep layer's return warns growing. A TFSF ended with its envelope: 129 over the line. A
column ended by its level instead of its change: the oblique PlaneWave's static residue never
ends and its scene is never judged. A holding column's end kept: the growing PlaneWave scene
says only "run longer". The Absorber's cells left out: a row rises 1.6e-3 of the peak. Weights
of one, the Debye energy left out, the pole read half a step late or without its coupling,
Debye read a step late, a mirror plane counted whole, the first version's H pair: each fails
its gate above. So do the four this fixup added a gate for: NumPy's reading taken from the
*bare* curl (above the terms and the lift), the mirror plane's weight taken from component 0,
half a round trip in the column's quarter, and the holding band doubled to
``GROWTH_STEADY``. The naive energy ``eps |E^n|^2 + mu |H^(n-1/2)|^2``, off at first order: the
cavity's energy ripples by 8.2e-2 and the flatness gate fails. (The growing gate's own mutant
is in ``tests/fdtd/test_fd_pml_stability.py``; the source mutants this fixup ran, including the
reading taken after the update and the sampling started at step 0, are in its commit message.)

**A kernel defect, pinned, and reported** (slow). A lossless lamellar ``Region`` grating under the
default ``PML(cells=12, reflection=1e-8)`` does not settle -- its late field *grows* (a probe
below the tooth, p at normal incidence: 8.4e-4 at 192 periods, 1.7e-3 at 288, 2.6e-3 at 384, and
Build B's 960-period run reaches 5.9e-2; an e-fold every ~180 periods), at the ``(+-1, 0)``
Rayleigh wavelengths, where the tooth's guided resonances sit just on the evanescent side of the
light line. A stretch backed by a PEC wall is an active termination for such a mode, whatever
its constants (``src/wavebit/fdtd/_prepare/cpml.py::stabilizing_rate``;
``tests/fdtd/test_fd_pml_stability.py`` pins it on a slab column). ``PML(stabilize=3e-3)`` -- a small
matched loss on the normal component -- removes it: the same probe falls, 4.4e-4 -> 9.7e-5. It
is not the default because of what it costs at oblique incidence (``stabilizing_rate``, "Why it
is not the default"), so the default layer keeps growing and the run says so: the interior's
growth reads 2.64 at 384 periods (0.40 at 192, where the ring-down still dominates), the probe
sees it (2.29), and the run warns ``GrowingFieldWarning`` naming ``PML(stabilize=3e-3)``. The
stabilized twin decays under the floor (not judged) and ``PML(24, 1e-12)``, the control, reads
0.52 (its probe falls, 4.8e-4 -> 3.0e-4); neither warns growing.
"""

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64
import dataclasses
import functools
import importlib
import math
import warnings
from unittest import mock

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd import _assemble, _post
from wavebit.fdtd._assemble import GROWTH_FLOOR, GROWTH_STEADY, GROWTH_THRESHOLD, GROWTH_WINDOWS, UNSETTLED_THRESHOLD
from wavebit.fdtd._kernel import step as _kernel_step
from wavebit.fdtd._kernel.observe import InteriorEnergy
from wavebit.fdtd._prepare import lower, plan as make_plan
from wavebit.fdtd._runtime import runtime
from tests.fixtures.fdtd.fd_helpers import bloch_plane_wave, box_grating, cavity_box, dipole_box, flux_box_dipole, pec_cavity_dipole, symmetry_pair, to_numpy

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
        fd.Flux('in', box=fd.Region(size=(0.3,) * 3), wavelengths=(1.0,)),))


def test_a_run_cut_off_early_is_told_to_run_longer_and_names_no_knob():
    """A dipole cut off at 3 periods -- before its own pulse has ended -- is unsettled, not
    growing: "run longer" is the right advice and the PML's stabilizing knob is noise, whatever
    the layer (the pure stretch, the default; a stabilized one; a weaker target). Its growth is
    not judged at all: the source is still on at the end of the run."""
    layers = (None, fd.PML(cells=12, wavelength=1.0, stabilize=3e-3),
              fd.PML(cells=12, wavelength=1.0, reflection=1e-4))
    for layer in layers:
        box = _short_box()
        if layer is not None:
            box = box.replace(boundaries=fd.Boundaries.open(layer))
        res, caught = _run_collecting(box)
        assert len(caught) == 1, (layer, caught)
        assert caught[0].category is fd.UnsettledFieldWarning, (layer, caught[0].category)
        assert 'stabilize' not in str(caught[0].message), layer
        assert 'Run longer' in str(caught[0].message)
        assert res.diagnostics['growth'] == {'in': None}, res.diagnostics['growth']


def test_a_run_under_jax_grad_reads_the_values_and_under_jit_reports_none():
    """Under a bare ``jax.grad`` the energy series come back concrete (they are not on the
    differentiated path), so the diagnostic reads them and warns exactly as NumPy does; under
    ``jax.jit`` they are tracers, the ratio is ``None`` and nothing warns."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit_enable_x64()
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
# Growth: what the statistic separates (``_post.growth_ratio`` on series built here)
# ---------------------------------------------------------------------------------------------

#: A synthetic interior-energy series: 64 periods of a unit wavelength, 40 samples a period.
T = np.arange(1, 64 * 40 + 1) / 40.0


def _judged(e, start=0.0):
    return _post.growth_ratio(T, e, start, 1.0, GROWTH_WINDOWS, GROWTH_FLOOR)


def _growth(e, start=0.0):
    got = _judged(e, start)
    return None if got is None else got[0]


def _ripple():
    """A closed lossless box's sampled energy: constant, with the second-order ripple at twice
    the carrier (``InteriorEnergy``; 1e-3 is several times what the cavity below shows)."""
    return 1.0 + 1e-3 * np.cos(4 * np.pi * T)


def test_the_statistic_separates_growth_from_holding_decay_and_the_floor():
    assert abs(_growth(_ripple()) - 1.0) < 3e-3              # a closed lossless box: holds
    ring = np.exp(-T / 4.0) * _ripple()                      # a ring-down
    assert abs(_growth(ring) / math.exp(-4.0) - 1.0) < 3e-3  # a quarter is 16 periods: e^-4
    assert _growth(np.exp(-T) * _ripple()) is None           # decays under the floor
    noise = np.random.default_rng(0).random(T.size) * np.linspace(1, 5, T.size) * 1e-20
    assert _growth(np.where(T < 1.0, 1.0, noise)) is None    # rising noise at the floor
    # A mode seeded under a ring-down, growing an e-fold every 8 periods in energy: each
    # 16-period quarter is e^2 = 7.4x the one before, whatever the ring-down did first.
    grow = ring + 1e-4 * np.exp(T / 8.0)
    assert abs(_growth(grow) / math.exp(2.0) - 1.0) < 1e-2, _growth(grow)
    # And one that rises from the start, with no dip at all -- the fastest growth there is.
    assert abs(_growth(1e-4 * np.exp(T / 8.0)) / math.exp(2.0) - 1.0) < 1e-2


def _spread(e, start=0.0):
    got = _judged(e, start)
    return None if got is None else got[4]


def test_a_growth_rises_steadily_and_an_arrival_does_not():
    """The spread: the last half's eighth-to-eighth log rises, largest over smallest. An
    exponential's are equal (1.002, the binning), and a mode seeded under a ring-down comes to
    it as the ring-down fades (1.017); a pulse arriving over a quiet floor -- energy handed back
    -- steepens as it comes (4.2), and one that has arrived and holds does not rise throughout
    (``inf``). Both are far over the line by their ratio. (A pulse's edge slow enough to be
    exponential over the whole last half is not told apart: centred 72 and 10 wide it reads
    1.5 -- ``GROWTH_THRESHOLD``'s "what still reads as growth".)"""
    ring = np.exp(-T / 4.0) * _ripple()
    assert _spread(1e-4 * np.exp(T / 8.0)) < 1.01
    assert _spread(ring + 1e-4 * np.exp(T / 8.0)) < 1.05
    arrival = np.exp(-T / 2.0) * _ripple() + 1e-3 * np.exp(-((T - 70.0) / 8.0) ** 2)
    assert _growth(arrival) > 1e3 and _spread(arrival) > GROWTH_STEADY, _judged(arrival)
    held = ring + 1e-2 * (T > 54.0)
    assert _growth(held) > GROWTH_THRESHOLD and _spread(held) == math.inf


def _deep_layer(time=13.0):
    """A pulse (no zero-frequency content to speak of) kicked into a vacuum column under a deep
    weak layer, ``PML(64, 1e-3)``: it leaves, and the wall behind the layer sends 1e-3 of it
    back after a 6.4-wavelength round trip, into an interior the pulse has left."""
    dx, cells = 1 / 20, 64
    return fd.Scene(
        domain=fd.Domain(size=(dx, dx, 6.0 + 2 * cells * dx)),
        boundaries=fd.Boundaries(x=fd.Axis1D(k=0.0), y=fd.Axis1D(k=0.0),
                                 z=fd.PML(cells=cells, reflection=1e-3, wavelength=1.0,
                                          allow_thin=True)),
        resolution=fd.Resolution(dx=dx), run=fd.Run(time=time),
        initial=fd.InitialField(E=(lambda x, y, z: np.exp(-z ** 2) * np.cos(2 * math.pi * z),
                                   None, None)),
        monitors=(fd.Record('r', at=(0, 0, 0), component='Ex'),
                  fd.Flux('f', plane=fd.Plane(z=0.5), wavelengths=(1.0,))))


def test_a_layers_return_is_an_arrival_and_not_growth():
    """Energy a PML took and hands back is not in the interior's account, and by its size alone
    it is growth: the deep layer's echo makes the last quarter 984 times the one before (at
    t = 13; 1.8e4 at 12.25). By its shape it is not -- it steepens as it arrives, spread 8.2
    against the line's 2 -- and nothing warns."""
    res, caught = _run_collecting(_deep_layer())
    assert res.diagnostics['interior_growth'] > 100.0, res.diagnostics['interior_growth']
    assert res.diagnostics['interior_spread'] > GROWTH_STEADY, res.diagnostics
    assert not [w for w in caught if w.category is fd.GrowingFieldWarning], caught


def test_a_kick_with_zero_frequency_content_still_reads_as_growth_and_the_warning_says_why():
    """The case no statistic of the interior's energy tells apart, pinned so the documentation
    stays true (``_assemble.GROWTH_THRESHOLD``, "what still reads as growth"): a Gaussian kick
    ``exp(-(z/0.15)^2) cos(2 pi z)`` -- most of its content near zero frequency -- in a vacuum
    column under the default PML. The layer's CFS term leaves that content unabsorbed, and it
    returns off the wall behind as a slow, steady rise: 4.42 at t = 6.25, spread 1.18 (it warns
    at 8 of the 52 run lengths to 16). The warning says an InitialField can do this."""
    dx, cells = 1 / 20, 12
    scene = fd.Scene(
        domain=fd.Domain(size=(dx, dx, 2.0 + 2 * cells * dx)),
        boundaries=fd.Boundaries(x=fd.Axis1D(k=0.0), y=fd.Axis1D(k=0.0),
                                 z=fd.PML(cells=cells, wavelength=1.0)),
        resolution=fd.Resolution(dx=dx), run=fd.Run(time=6.25),
        initial=fd.InitialField(E=(lambda x, y, z: np.exp(-(z / 0.15) ** 2)
                                   * np.cos(2 * math.pi * z), None, None)),
        monitors=(fd.Record('r', at=(0, 0, 0), component='Ex'),))
    res, caught = _run_collecting(scene)
    growing = [str(w.message) for w in caught if w.category is fd.GrowingFieldWarning]
    assert growing and all('starts from an InitialField' in m for m in growing), growing
    assert res.diagnostics['interior_spread'] < GROWTH_STEADY, res.diagnostics


def test_mutant_the_ratio_alone_calls_the_layers_return_growth(monkeypatch):
    """Judged by its ratio alone (the review's case: "a single returning echo must not
    trigger"), the deep layer's echo warns growing and the gate above fails."""
    monkeypatch.setattr(_assemble, 'GROWTH_STEADY', math.inf)
    _, caught = _run_collecting(_deep_layer())
    assert [w for w in caught if w.category is fd.GrowingFieldWarning], caught


def test_the_span_starts_where_the_sources_end_and_its_quarters_are_its_own():
    """The span is ``[start, end]`` and the two quarters are that span's, not the run's: with
    the sources on until 40 and the energy decaying after, the quarters are [52, 58) and
    [58, 64] (the run's would be [32, 48) and [48, 64], reaching back to when it was driven)."""
    driven = np.where(T < 40.0, T / 40.0, np.exp(-(T - 40.0) / 4.0))
    got = _judged(driven, start=40.0)
    assert got[1:4] == pytest.approx((52.0, 58.0, 64.0)), got
    assert abs(got[0] / math.exp(-1.5) - 1.0) < 1e-2, got   # 6 periods at e^-1/4 each


def test_a_span_shorter_than_the_line_is_not_judged():
    grow = np.exp((T - 64.0) * 4.0 / GROWTH_WINDOWS)        # an e-fold per quarter
    start = 64.0 - GROWTH_WINDOWS
    assert abs(_growth(grow, start) / math.e - 1.0) < 3e-2   # exactly GROWTH_WINDOWS periods
    assert _growth(grow, start + 0.5) is None                # half a period short


def test_a_passive_echo_does_not_read_as_growth():
    """A monitor behind a thick glass slab sees the second reflection arrive long after the
    first, over a quiet gap: its own series rises by 1e6 from one quarter to the next at 60
    (the first version of this diagnostic judged monitors and warned growing here). The
    interior's energy does not rise -- the echo is energy the interior already held: 0.40 -- so
    nothing warns growing. (The run is 60 long because the PlaneWave's end is read off its
    column, whose own layer returns a trace of the pulse at t = 24 .. 28.)"""
    dx = 0.05
    pw = fd.PlaneWave('pw', z=1.0, direction='-z', polarization='s',
                      band=fd.Band(wavelength=1.0, df=0.15))
    scene = fd.Scene(
        domain=fd.Domain(size=(0.1, 0.1, 12.0 + 24 * dx), center=(0, 0, -3.5)),
        boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=fd.PML(cells=12)),
        resolution=fd.Resolution(dx=dx), run=fd.Run(time=60.0),
        structures=(fd.Structure(fd.Slab(z=(-8.0, 0.0)), fd.Material(eps=2.25)),),
        sources=(pw,),
        monitors=(fd.Record('rec', at=(0, 0, 2.0), component='Ey'),
                  fd.Flux('r', plane=fd.Plane(z=2.0), wavelengths=(1.0,))))
    res, caught = _run_collecting(scene)
    growth = res.diagnostics['growth']
    assert growth['rec'] > 1e5 and growth['r'] > 1e5, growth         # measured 1.0e6 / 1.3e6
    assert res.diagnostics['interior_growth'] < 1.0, res.diagnostics  # measured 0.40
    assert not [w for w in caught if w.category is fd.GrowingFieldWarning], caught


@functools.lru_cache(maxsize=None)
def _cavity(periods=40):
    return _run_collecting(pec_cavity_dipole(periods=periods))


def test_a_long_lossless_cavity_is_unsettled_but_not_growing():
    """A closed PEC box rings forever at constant energy, its modes beating: unsettled at any
    run length (the plain warning, "run longer" being the only advice there is), never
    growing: its interior's energy reads 1.0000 over two quarters, and it holds to 1.7e-4
    (max / min - 1) from the moment the dipole's envelope has ended -- what the drive's tail
    still does past its cutoff, the reading itself being the leapfrog's exactly conserved
    energy (``InteriorEnergy``; the closed boxes below hold to 1e-13). The gate is 6x that."""
    res, caught = _cavity()
    assert [w.category for w in caught] == [fd.UnsettledFieldWarning], caught
    assert abs(res.diagnostics['interior_growth'] - 1.0) < 1e-6, res.diagnostics
    t, W = res.diagnostics['interior_energy']
    after = W[t >= _assemble.sources_end(pec_cavity_dipole(periods=40))]
    assert after.max() / after.min() - 1.0 < FLAT, after.max() / after.min()
    assert res.diagnostics['growth']['in'] is not None
    assert res.diagnostics['growth_threshold'] == GROWTH_THRESHOLD


#: How flat the cavity's sampled interior energy is after its dipole (max / min - 1).
FLAT = 1e-3

_PLAN_MODULE = importlib.import_module('wavebit.fdtd._prepare.plan')
_REAL_WEIGHT = InteriorEnergy._weight


def _first_order(self, params, field, rt, comps=slice(None), box=None, times=None, key=''):
    """Mutant: the naive energy ``eps |E^n|^2 + mu |H^(n-1/2)|^2``, off at first order -- the
    product with ``H^(n+1/2)`` replaced by the old H's own square (no ``minus_bh cE`` term,
    ``ah`` read as 1)."""
    if times == 'minus_bh':
        return 0.0
    return _REAL_WEIGHT(self, params, field, rt, comps, box, None, key)


def test_mutant_the_first_order_energy_is_caught_by_the_flatness_gate(monkeypatch):
    """Without the product ``H^(n-1/2) . H^(n+1/2)`` the sampled energy ripples at first order
    in ``omega dt`` (measured 8.2e-2 in the cavity at 12 periods): the flatness gate above
    fails."""
    monkeypatch.setattr(InteriorEnergy, '_weight', _first_order)
    res, _ = _run_collecting(pec_cavity_dipole(periods=12))
    t, W = res.diagnostics['interior_energy']
    after = W[t >= _assemble.sources_end(pec_cavity_dipole(periods=12))]
    assert after.max() / after.min() - 1.0 > FLAT, after.max() / after.min()


# ---------------------------------------------------------------------------------------------
# The reading is the leapfrog's exactly conserved energy: closed lossless boxes, every step
# ---------------------------------------------------------------------------------------------

def _every_step(scene, backend='numpy'):
    """``(t, W)``: the interior energy read at every step from step 1 (``ENERGY_STRIDE`` set to
    1 for this run), exactly as the observer reads it at its stride."""
    with mock.patch.object(_PLAN_MODULE, 'ENERGY_STRIDE', 1), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fd.run(scene, backend=backend).diagnostics['interior_energy']


def _random_box(courant, dtype='real', seed=1, n=10):
    """A closed box of random (grid-scale) field. ``'real'``: every face a different wall --
    PEC, PMC and ``Symmetry`` planes of both parities on both kinds of face, three mirrors
    meeting at edges and a corner. ``'complex'``: a Bloch box of complex field."""
    dx = 1.0 / 16
    rng = np.random.default_rng(seed)
    shape = (3, n - 2, n - 1, n)
    E, H = rng.standard_normal(shape), rng.standard_normal(shape)
    if dtype == 'complex':
        E, H = E + 1j * rng.standard_normal(shape), H + 1j * rng.standard_normal(shape)
        faces = dict(x=fd.Bloch(phase=0.7), y=fd.Bloch(phase=-1.1), z=fd.Periodic())
    else:
        faces = dict(x=(fd.Symmetry(parity=1), fd.PMC()), y=(fd.PEC(), fd.Symmetry(parity=-1)),
                     z=(fd.PMC(), fd.Symmetry(parity=-1)))
    return fd.Scene(domain=fd.Domain(size=(n * dx, (n - 1) * dx, (n - 2) * dx)),
                    boundaries=fd.Boundaries(**faces), resolution=fd.Resolution(dx=dx),
                    run=fd.Run(steps=300, courant=courant),
                    initial=fd.InitialField(E=E, H=H),
                    monitors=(fd.Record('p', at=(0, 0, 0), component='Ex'),))


#: How flat the observer's own reading is in a closed lossless box, every step (max / min - 1):
#: measured 1.3e-15 .. 1.4e-14 (``Symmetry`` on either face and of either parity, PEC and PMC
#: far walls, three mirrors meeting, a Bloch box of complex field, courant 0.5 and 0.9, random
#: grid-scale field), and up to 4e-13 for the stride's alias modes below over 3000 steps.
EXACT = 1e-12


@pytest.mark.parametrize('courant', [0.5, 0.9])
@pytest.mark.parametrize('side, parity', [('-', 1), ('-', -1), ('+', 1), ('+', -1)])
def test_the_reading_is_exact_in_a_half_box_and_its_full_box(side, parity, courant):
    """``eps E^n . E^n + mu H^(n-1/2) . H^(n+1/2)``, read before the H update from the curl the
    update uses (``InteriorEnergy``): exactly conserved, every step, in a mirror's half box
    (the plane's components counted at half weight, ``plan._mirror_planes``) and in the full
    box it is half of, from random grid-scale field."""
    half, full, _ = symmetry_pair(side, parity, far=fd.PMC() if parity > 0 else fd.PEC(),
                                  courant=courant, steps=300)
    for scene in (half, full):
        t, W = _every_step(scene)
        assert W.size == 299 and W.max() / W.min() - 1.0 < EXACT, W.max() / W.min() - 1.0


@pytest.mark.parametrize('courant', [0.5, 0.9])
@pytest.mark.parametrize('dtype', ['real', 'complex'])
def test_the_reading_is_exact_with_mirrors_meeting_and_in_a_bloch_box(dtype, courant):
    t, W = _every_step(_random_box(courant, dtype))
    assert W.max() / W.min() - 1.0 < EXACT, W.max() / W.min() - 1.0


def _aniso_mirror(side, parity, magnetic=False, n=10, steps=300, seed=7):
    """A closed lossless box with a ``Symmetry`` plane on face ``side`` of x and an
    anisotropic slab flush against that plane, from random grid-scale field.

    The mirror plane's own components are weighed one by one
    (``InteriorEnergy._cut``'s ``comps``), and a structure **on** the plane is the only
    configuration in which that slicing is a different number from component 0's: everywhere
    else the plane sits in vacuum, where ``eps`` and ``mu`` are scalars and ``_cut`` returns
    at once.
    """
    dx = 1.0 / 16
    L = n * dx
    faces = ((fd.Symmetry(parity=parity), fd.PEC()) if side == '-'
             else (fd.PEC(), fd.Symmetry(parity=parity)))
    span = (-L / 2, -L / 2 + 0.2) if side == '-' else (L / 2 - 0.2, L / 2)
    material = fd.Material(eps=(2.0, 3.0, 5.0),
                           **(dict(mu=(1.5, 2.5, 4.0)) if magnetic else {}))
    rng = np.random.default_rng(seed)
    shape = (3, n, n, n)
    return fd.Scene(domain=fd.Domain(size=(L, L, L)),
                    boundaries=fd.Boundaries(x=faces, y=fd.PMC(), z=fd.PEC()),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps, courant=0.5),
                    structures=(fd.Structure(fd.Slab(x=span), material),),
                    initial=fd.InitialField(E=rng.standard_normal(shape),
                                            H=rng.standard_normal(shape)),
                    monitors=(fd.Record('p', at=(0, 0, 0), component='Ex'),))


@pytest.mark.parametrize('magnetic', [False, True], ids=['eps', 'eps+mu'])
@pytest.mark.parametrize('side, parity', [('-', 1), ('+', -1)])
def test_the_reading_is_exact_with_an_anisotropic_structure_on_the_mirror_plane(
        side, parity, magnetic):
    """A ``Material(eps=(2, 3, 5))`` slab (and its ``mu=(1.5, 2.5, 4)`` twin) sitting on the
    mirror plane, both kinds of face: the plane's half-weighted components each take their own
    ``eps`` or ``mu``, and the reading is still the leapfrog's conserved energy at every step.
    Measured 4.7e-15 .. 7.5e-15; read off component 0 instead it swings up to 1.7e-2 (below)."""
    t, W = _every_step(_aniso_mirror(side, parity, magnetic))
    assert W.max() / W.min() - 1.0 < EXACT, W.max() / W.min() - 1.0


def test_mutant_a_mirror_planes_weight_taken_from_component_zero_is_caught(monkeypatch):
    """``_cut`` giving every plane part component 0's coefficient in place of its own: a
    defect no vacuum-plane scene can see (every ``Symmetry`` scene in the suite has scalar
    coefficients at its plane, where ``_cut`` returns early), and the gate above fails at
    1.4e-2 and 1.3e-2 against its 7.5e-15. The ``mu`` twin is what makes both faces bite: on a
    high face the plane's own E component is the normal one, component 0, which this mutant
    leaves alone, so with ``eps`` anisotropic and ``mu`` scalar that case still reads 4.7e-15
    -- right by accident. A low face is caught either way (1.7e-2 with ``eps`` alone)."""
    real = InteriorEnergy._cut

    def component_zero(b, comps, box):
        if getattr(b, 'ndim', 0) == 4 and b.shape[0] > 1 and comps.start:
            comps = slice(0, 1)
        return real(b, comps, box)

    monkeypatch.setattr(InteriorEnergy, '_cut', staticmethod(component_zero))
    for side, parity in (('-', 1), ('+', -1)):
        t, W = _every_step(_aniso_mirror(side, parity, magnetic=True))
        assert W.max() / W.min() - 1.0 > 1e-3, (side, parity, W.max() / W.min() - 1.0)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('dtype', ['real', 'complex'])
def test_the_functional_backends_read_the_same_exact_energy(dtype, backend):
    """JAX and Torch keep ``H^(n-1/2)`` through the update, so they read the product of the two
    fields after it (``step.py``) where NumPy reads it from the curl before: the same energy,
    exactly conserved (measured 1e-15 .. 1e-14), and NumPy's series to round-off."""
    pytest.importorskip(backend)
    if backend == 'jax':
        _legacy_wavebit_enable_x64()
    scene = _random_box(0.9, dtype)
    _, reference = _every_step(scene)
    _, W = _every_step(scene, backend)
    W = to_numpy(W)
    assert W.max() / W.min() - 1.0 < EXACT, W.max() / W.min() - 1.0
    assert np.max(np.abs(W - reference)) < EXACT * reference.max()


def _magnetic_dipole():
    """An open box driven by a **magnetic** dipole: the one scene here whose term writes into
    the curl the H update uses, inside the judged cells.

    A term's drive reaches ``H`` through ``before_H`` (``step.py``'s slot order), so it is in
    the curl NumPy reads its half of the row from and in the field JAX and Torch read theirs
    from -- the two writings must carry it alike. An electric ``Dipole`` drives ``before_E``
    and never touches the H half at all, and a ``PlaneWave``'s and a ``TFSF``'s correction
    lands on faces where ``H^(n-1/2)`` is zero, so a magnetic dipole is what separates them.
    """
    return dipole_box(component='Hz', kind='magnetic', size=0.5, cells_per_wavelength=14,
                      steps=110)


@functools.lru_cache(maxsize=None)
def _magnetic_dipole_series():
    """NumPy's every-step reading of :func:`_magnetic_dipole`, run once per session."""
    return _every_step(_magnetic_dipole())


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_the_backends_read_the_same_energy_where_a_term_adds_to_the_curl(backend):
    """The gate above runs a closed box with no term in it, so nothing there tells the two
    writings apart where a source drives the curl. A magnetic dipole does
    (:func:`_magnetic_dipole`): NumPy's ``mu ah |H^(n-1/2)|^2 + mu minus_bh H^(n-1/2) . cE``
    carries the dipole's ``M`` because it is already in ``cE``, and JAX's and Torch's product
    of the two fields carries it because the update applied it -- measured to 1.5e-15 (JAX)
    and 6.3e-16 (Torch) of the run's peak, where dropping it is 1.1e-1 (the mutant below)."""
    pytest.importorskip(backend)
    if backend == 'jax':
        _legacy_wavebit_enable_x64()
    _, reference = _magnetic_dipole_series()
    _, W = _every_step(_magnetic_dipole(), backend)
    W = to_numpy(W)
    assert np.max(np.abs(W - reference)) < EXACT * np.max(np.abs(reference))


def test_mutant_numpys_reading_of_the_bare_curl_is_caught(monkeypatch):
    """The in-place reading moved above the term loop, so NumPy reads the bare curl and the
    dipole's own drive is not in its H half. Nothing else in the fast tier notices -- the
    whole tier passes under it (review, 2026-09-20) -- and the gate above fails: the row is
    out by 11% of the run's peak (measured 1.06e-1).

    The stabilizing ``lift`` runs in that same gap and needs no gate of its own: it scales one
    face's ``normal_slab``, which is that face's own PML cells (``cpml.py::CpmlFace``), and the
    judged cells ``lo .. hi`` are every cell but a PML's (``src/wavebit/fdtd/_prepare/plan.py``), so
    it cannot reach a cell this observer sums.
    """
    bare = {}
    real_curl, real_magnetic = _kernel_step.curl, InteriorEnergy.magnetic
    # Populate the reference before patching even when this test runs alone.
    _, reference = _magnetic_dipole_series()

    def spy(F, field, plan, params, psi, rt):
        cF, psi = real_curl(F, field, plan, params, psi, rt)
        if field == 'E':
            bare['cE'] = np.array(cF)           # before the terms and the lift write into it
        return cF, psi

    def from_the_bare_curl(self, H, new, cE, params, rt, mem=None):
        return real_magnetic(self, H, new, bare['cE'] if new is None else None, params, rt, mem)

    monkeypatch.setattr(_kernel_step, 'curl', spy)
    monkeypatch.setattr(InteriorEnergy, 'magnetic', from_the_bare_curl)
    _, W = _every_step(_magnetic_dipole())
    out = np.max(np.abs(W - reference)) / np.max(np.abs(reference))
    assert out > 1e-2, out                       # measured 1.2e-1


def test_mutant_a_mirror_plane_counted_whole_is_caught():
    """The plane's components counted whole: the half box's reading carries the plane's own
    energy, which is not conserved -- it swings 4.5e-2 (``p = +1`` on the low face) and 4.1e-2
    (``p = -1`` on the high face), and the gate above fails."""
    real = InteriorEnergy.observe

    def whole(self, acc, *args):
        return real(dataclasses.replace(self, planes=()), acc, *args)

    for side, parity in (('-', 1), ('+', -1)):
        half = symmetry_pair(side, parity, far=fd.PEC(), steps=300)[0]
        with mock.patch.object(InteriorEnergy, 'observe', whole):
            t, W = _every_step(half)
        assert W.max() / W.min() - 1.0 > 1e-3, (side, parity, W.max() / W.min() - 1.0)


def test_mutant_the_first_versions_h_pair_is_caught(monkeypatch):
    """The first version read ``eps E . E + mu (|H^(n-1/2)|^2 + |H^(n+1/2)|^2) / 2``, off by
    ``dt^2 |curl E|^2 / (2 mu)``: second order, rippling at twice the carrier, which the stride
    aliased into a steady rise (the exact Yee mode (14, 12, 12) of a 16^3 PEC box warned growing
    at 3000 steps; review, 2026-09-20). Read that way, the random box of grid-scale field swings
    by far more than the exact gate allows."""
    def pair(self, H, new, cE, params, rt, mem=None):
        view = (slice(None),) + tuple(slice(lo, hi) for lo, hi in self.box)
        new = params['ah'] * H + params['minus_bh'] * cE if new is None else new
        mu = self._weight(params, 'H', rt)
        return 0.5 * (rt.weighted_energy(H[view], mu) + rt.weighted_energy(new[view], mu))

    monkeypatch.setattr(InteriorEnergy, 'magnetic', pair)
    t, W = _every_step(_random_box(0.5))
    assert W.max() / W.min() - 1.0 > 1e-2, W.max() / W.min() - 1.0


@pytest.mark.parametrize('lmn', [(14, 12, 12), (14, 13, 11)])
def test_the_strides_alias_modes_read_flat_and_never_warn(lmn):
    """The exact Yee modes the first version's H pair aliased into growth at the stride
    (``ENERGY_STRIDE``; (14, 12, 12) warned at 3000 steps, (14, 13, 11) read 1.38 at 2000):
    with no ripple left to alias, the sampled energy holds to 4e-13 at courant 0.5 and 0.9 and
    nothing warns growing."""
    for courant in (0.5, 0.9):
        scene = cavity_box('pec', N=16, steps=3000, lmn=lmn)[0].replace(
            run=fd.Run(steps=3000, courant=courant))
        res, caught = _run_collecting(scene)
        t, W = res.diagnostics['interior_energy']
        assert W.max() / W.min() - 1.0 < EXACT, W.max() / W.min() - 1.0
        assert abs(res.diagnostics['interior_growth'] - 1.0) < 1e-10, res.diagnostics
        assert not [w for w in caught if w.category is fd.GrowingFieldWarning], caught


def _pole_cavity(pole, dx=0.02, time=40.0, courant=0.5):
    """A closed lossless dispersive cavity, one wavelength long, kicked in its field alone."""
    return fd.Scene(
        domain=fd.Domain(size=(dx, dx, 1.0)),
        boundaries=fd.Boundaries(x=fd.Axis1D(k=0.0), y=fd.Axis1D(k=0.0), z=fd.PEC()),
        resolution=fd.Resolution(dx=dx), run=fd.Run(time=time, courant=courant),
        structures=(fd.Structure(fd.Slab(z=(-0.5, 0.5)),
                                 fd.Dispersive(eps_inf=1.0, poles=(pole,))),),
        initial=fd.InitialField(E=(lambda x, y, z: np.cos(math.pi * z), None, None)),
        monitors=(fd.Record('p', at=(0.0, 0.0, 0.1), component='Ex'),))


POLES = [fd.Lorentz(omega_0=2 * math.pi, strength=(2 * math.pi) ** 2),
         fd.Drude(omega_p=2 * math.pi)]

#: How flat a lossless pole cavity's sampled energy is, the pole's own energy counted
#: (max / min - 1): measured 4.3e-14 (Lorentz) and 2.7e-14 (Drude) over 40 time units at dx
#: 0.02 -- the reading is exact, the field's part and the pole's alike. (The first version's
#: half-step-late pole reading swung 3.5% here; the field's H pair alone, 1.6e-4.)
POLE_FLAT = 1e-12


def _swing(scene):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t, W = fd.run(scene).diagnostics['interior_energy']
    return W.max() / W.min() - 1.0


@pytest.mark.parametrize('pole', POLES, ids=['lorentz', 'drude'])
def test_a_poles_stored_energy_is_part_of_the_interiors(pole):
    """A lossless Lorentz or Drude cavity kicked in its field hands its energy back and forth
    between the field and the polarisation; counted together (``InteriorEnergy``: the pole's
    ``V^(n+1/2) - E^n . (P^(n+1) - P^n)``) the sampled energy holds to round-off over 40 time
    units."""
    assert _swing(_pole_cavity(pole)) < POLE_FLAT


@pytest.mark.parametrize('pole', POLES, ids=['lorentz', 'drude'])
def test_mutant_the_field_alone_is_caught_by_the_pole_gate(pole, monkeypatch):
    """Without the pole's energy the field's alone swings by a factor of 1.6e4 (Lorentz: the
    field empties into the oscillator) and 4.3 (Drude): the gate above fails."""
    monkeypatch.setattr(InteriorEnergy, '_pole', staticmethod(lambda *args: 0.0))
    assert _swing(_pole_cavity(pole)) > POLE_FLAT


def _largest_rise(scene):
    """The largest rise of the sampled interior energy from one row to the next, over its
    peak."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t, W = fd.run(scene).diagnostics['interior_energy']
    return float(np.max(np.diff(W)) / W.max())


#: Debye cavities: ``(pole, dx, courant, time)``. The second is the review's ``tau`` near ``dt``
#: (``dt = 3.5e-3``), where the first version's reading swung 12-20% per sample.
DEBYE = [(fd.Debye(tau=0.05, delta_eps=5.0), 0.02, 0.5, 40.0),
         (fd.Debye(tau=0.005, delta_eps=5.0), 0.02, 0.3, 10.0)]

#: The most a lossy (Debye) cavity's sampled energy may rise from one row to the next, over
#: its peak. Measured: it never rises (the largest step is a fall, -4.6e-7 and -2.2e-7); with
#: the Debye energy left out it rises 1.8e-2 and 1.7e-2. Round-off is the only allowance.
DEBYE_RISE = 1e-12


@pytest.mark.parametrize('case', DEBYE, ids=['tau=0.05', 'tau=0.005'])
def test_a_debye_cavity_never_gains_energy(case):
    """A Debye pole is a loss: the field plus ``|P|^2 / delta_eps`` never rises, at any
    ``tau`` (``InteriorEnergy``). Read at the right slot of the kernel's memory, the sampled
    energy only falls."""
    pole, dx, courant, time = case
    assert _largest_rise(_pole_cavity(pole, dx=dx, time=time, courant=courant)) < DEBYE_RISE


@pytest.mark.parametrize('case', DEBYE, ids=['tau=0.05', 'tau=0.005'])
def test_mutant_a_debye_pole_read_as_storing_nothing_is_caught(case, monkeypatch):
    """The Debye branch's energy zeroed (the review's surviving mutant: no fast-tier scene had
    a Debye pole): the field alone rises as the polarisation relaxes and hands its energy back
    -- the gate above fails."""
    real = InteriorEnergy._pole

    def zero_debye(mem, E, coefficients, *args):
        if float(np.asarray(coefficients['hold'][0])) == 0.0:
            return 0.0
        return real(mem, E, coefficients, *args)

    monkeypatch.setattr(InteriorEnergy, '_pole', staticmethod(zero_debye))
    pole, dx, courant, time = case
    assert _largest_rise(_pole_cavity(pole, dx=dx, time=time, courant=courant)) > 1e-3


def test_a_pole_with_no_strength_stores_nothing():
    """``Lorentz(strength=0)`` and ``Debye(delta_eps=0)`` are legal and never driven; the pole
    energy divides by the strength, so the reading skips them (and under a trace zeroes them)
    rather than warn an overflow for a ``0/0``. The cavity holds as a vacuum one does (1.3e-15)."""
    poles = (fd.Lorentz(omega_0=2 * math.pi, strength=0.0), fd.Debye(tau=0.05, delta_eps=0.0))
    scene = _pole_cavity(poles[0], dx=0.05, time=6.0).replace(structures=(
        fd.Structure(fd.Slab(z=(-0.5, 0.5)), fd.Dispersive(eps_inf=1.0, poles=poles)),))
    res, caught = _run_collecting(scene)
    t, W = res.diagnostics['interior_energy']
    assert np.all(np.isfinite(W)) and W.max() / W.min() - 1.0 < EXACT
    assert not [w for w in caught if w.category is fd.GrowingFieldWarning], caught


def _lorentz_and_debye(time=6.0):
    """A closed cavity of one material with a Lorentz and a Debye pole, kicked in its field."""
    poles = (fd.Lorentz(omega_0=2 * math.pi, strength=10 * (2 * math.pi) ** 2),
             fd.Debye(tau=0.05, delta_eps=2.0))
    return _pole_cavity(poles[0], dx=0.05, time=time).replace(structures=(
        fd.Structure(fd.Slab(z=(-0.5, 0.5)), fd.Dispersive(eps_inf=1.0, poles=poles)),))


def test_the_traced_pole_reading_is_numpys():
    """On JAX the executor is jitted with the params as its argument, so the pole coefficients
    are tracers and ``InteriorEnergy._pole`` takes its traced path: both kinds computed and
    ``where(hold == 0, debye, second)`` choosing per pole. A Lorentz and a Debye pole in one
    material read the same series as NumPy's static path (measured 2e-15), and the reading
    never rises. (Mutant: the ``where``'s branches swapped -- each pole read as the other kind
    -- moves the JAX series off NumPy's by order one.)"""
    pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    scene = _lorentz_and_debye()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, reference = fd.run(scene).diagnostics['interior_energy']
        _, traced = fd.run(scene, backend='jax').diagnostics['interior_energy']
    assert np.max(np.abs(traced - reference) / np.abs(reference)) < 1e-12
    assert np.max(np.diff(reference)) <= 1e-12 * reference.max()


# ---------------------------------------------------------------------------------------------
# The pole energy is the recurrence's own conserved one, exactly
# ---------------------------------------------------------------------------------------------

#: Strong lossless poles, at coarse ``dt``: ``(pole, dx, courant)``. The first two are the
#: review's scenes that read 1.47 (the default courant) and 1.20 under the first version.
STRONG = {'lorentz 40': (fd.Lorentz(omega_0=2 * math.pi, strength=40 * (2 * math.pi) ** 2),
                         0.05, 0.5),
          'drude 6': (fd.Drude(omega_p=6 * 2 * math.pi), 0.05, 0.9),
          'lorentz 10, dx 0.1': (fd.Lorentz(omega_0=2 * math.pi,
                                            strength=10 * (2 * math.pi) ** 2), 0.1, 0.9)}


def _strong(name, time=6.0):
    pole, dx, courant = STRONG[name]
    return _every_step(_pole_cavity(pole, dx=dx, time=time, courant=courant))[1]


@pytest.mark.parametrize('name', list(STRONG))
def test_the_pole_energy_is_the_recurrences_own_conserved_one(name):
    """``T^n = F^n + V^(n+1/2) - E^n . (P^(n+1) - P^n)`` (``InteriorEnergy``) is conserved by
    the kernel's recurrence exactly: read by the observer at every step, to round-off (1.3e-15
    .. 2.9e-15) for a Lorentz at 40 ``omega0^2``, a Drude at 6 ``omega``, and a Lorentz at dx
    0.1 and courant 0.9."""
    T = _strong(name)
    assert T.max() / T.min() - 1.0 < EXACT, T.max() / T.min() - 1.0


@pytest.mark.parametrize('pole', [fd.Debye(tau=0.01, delta_eps=1.0),
                                  fd.Lorentz(omega_0=2 * math.pi,
                                             strength=10 * (2 * math.pi) ** 2, gamma=2.0)],
                         ids=['debye', 'damped lorentz'])
def test_a_lossy_poles_energy_never_rises_exactly(pole):
    """With loss the same ``T`` (Debye's ``F + |P|^2 / delta_eps``) never rises, step by step,
    to round-off (measured: every step a fall, the smallest 8.1e-9 and 1.5e-9 of the start).
    The observer's rows start at step 1: the pole memory starts at rest (``poles.py``), which
    drops one step's drive, so ``T^0`` is off the identity's path -- the Debye cavity's reading
    rises 6.0e-4 from ``T^0`` to ``T^1`` and falls at every step after."""
    T = _every_step(_pole_cavity(pole, dx=0.05, time=6.0))[1]
    assert np.max(np.diff(T)) <= EXACT * T[0], np.max(np.diff(T)) / T[0]


def _half_step_late(mem, E, coefficients, n_poles, cells, offsets, dt, rt, comps=slice(None)):
    """Mutant: the first version's reading, ``(|P'|^2 + omega0^2 |P|^2)/s`` at the midpoint of
    the last two values, with no coupling (Lorentz and Drude only)."""
    inside = (comps,) + tuple(slice(o, o + (c1 - c0)) for o, (c0, c1) in zip(offsets, cells))
    gain, hold, drive = (float(np.asarray(coefficients[k][0]))
                         for k in ('step_gain', 'hold', 'drive'))
    new, old = mem['P'][0][inside], mem['previous'][0][inside]
    scale = 2.0 / (dt * dt * (1.0 + hold))
    return (rt.weighted_energy(new - old, 1.0) / (dt * dt)
            + (2.0 / (dt * dt) - gain * scale)
            * rt.weighted_energy(0.5 * (new + old), 1.0)) / (drive * scale)


def _without_coupling(mem, E, coefficients, n_poles, cells, offsets, dt, rt, comps=slice(None)):
    """Mutant: the exact ``V`` with the coupling ``- E^n . (P^(n+1) - P^n)`` dropped."""
    inside = (comps,) + tuple(slice(o, o + (c1 - c0)) for o, (c0, c1) in zip(offsets, cells))
    field = E[(comps,) + tuple(slice(c0, c1) for c0, c1 in cells)]
    new, old = mem['P'][0][inside], mem['previous'][0][inside]
    return (_REAL_POLE(mem, E, coefficients, n_poles, cells, offsets, dt, rt, comps)
            + rt.real_inner(field, new - old))


_REAL_POLE = InteriorEnergy._pole


@pytest.mark.parametrize('mutant', [_half_step_late, _without_coupling],
                         ids=['half a step late', 'no coupling'])
def test_mutant_an_approximate_pole_reading_is_caught_by_the_exact_gate(mutant, monkeypatch):
    """Read half a step late at the midpoint (the first version, first order in ``dt``) or
    with the coupling term dropped, the strong Lorentz's ``T`` swings by order one, not
    round-off: the exact gate fails."""
    monkeypatch.setattr(InteriorEnergy, '_pole', staticmethod(mutant))
    T = _strong('lorentz 40')
    assert T.max() / T.min() - 1.0 > 1e-2, T.max() / T.min() - 1.0


def test_mutant_a_debye_pole_read_a_step_late_is_caught(monkeypatch):
    """Debye's ``P~`` read at ``mem['P']`` (a step ahead of ``E^n``) instead of
    ``mem['previous']``: ``T`` rises within a step, and the lossy gate fails."""
    def late(mem, E, coefficients, n_poles, cells, offsets, dt, rt, comps=slice(None)):
        shifted = dict(mem, previous=mem['P'])
        return _REAL_POLE(shifted, E, coefficients, n_poles, cells, offsets, dt, rt, comps)

    monkeypatch.setattr(InteriorEnergy, '_pole', staticmethod(late))
    T = _every_step(_pole_cavity(fd.Debye(tau=0.01, delta_eps=1.0), dx=0.05, time=6.0))[1]
    assert np.max(np.diff(T)) > 1e-3 * T[0], np.max(np.diff(T)) / T[0]


@pytest.mark.parametrize('name, time', [('lorentz 40', 26.82), ('drude 6', 9.18)])
def test_a_strong_lossless_pole_cavity_is_not_growth(name, time):
    """The review's two scenes, which read 1.47 (a Lorentz at 40 ``omega0^2``, the default
    courant) and 1.20 (a Drude at 6 ``omega``, courant 0.9) and warned growing under the first
    version's pole reading: now 1.0000 both, the sampled energy flat to 6e-15, and nothing
    warns."""
    pole, dx, courant = STRONG[name]
    res, caught = _run_collecting(_pole_cavity(pole, dx=dx, time=time, courant=courant))
    assert not [w for w in caught if w.category is fd.GrowingFieldWarning], caught
    assert abs(res.diagnostics['interior_growth'] - 1.0) < 1e-12, res.diagnostics
    t, W = res.diagnostics['interior_energy']
    assert W.max() / W.min() - 1.0 < EXACT, W.max() / W.min() - 1.0


# ---------------------------------------------------------------------------------------------
# What is weighed, and where
# ---------------------------------------------------------------------------------------------

def _half_filled(eps=12.0, mu=1.0):
    """A closed lossless 1-D PEC column, half of it ``(eps, mu)``, kicked in its field."""
    dx = 0.02
    return fd.Scene(
        domain=fd.Domain(size=(dx, dx, 1.0)),
        boundaries=fd.Boundaries(x=fd.Axis1D(k=0.0), y=fd.Axis1D(k=0.0), z=fd.PEC()),
        resolution=fd.Resolution(dx=dx), run=fd.Run(time=20.0),
        structures=(fd.Structure(fd.Slab(z=(-0.5, 0.0)), fd.Material(eps=eps, mu=mu)),),
        initial=fd.InitialField(E=(lambda x, y, z: np.cos(math.pi * z), None, None)),
        monitors=(fd.Record('p', at=(0.0, 0.0, 0.1), component='Ex'),))


#: How flat the half-filled column's sampled energy is (max / min - 1): measured 1.6e-15 with
#: ``eps = 12`` and 1.8e-15 with ``mu = 12``; with weights of 1 they swing 6.8 and (H alone
#: unweighted) 6.6, the wave carrying ``eps |E|^2`` or ``mu |H|^2`` on one side. The ``mu``
#: half is what sees the H weight: with ``mu = 1`` everywhere, an Absorber's included (the
#: formula reads 1 there), an H weight of 1 is the right one.
WEIGHT_FLAT = 1e-12


def test_the_weights_are_eps_and_mu_even_in_a_lossy_cell():
    """``eps`` and ``mu`` per component are read back off the update's own ``(a, b)``:
    ``(dt/dx)(1 + a)/(2 b)``, exactly -- 4 inside a ``Conductor(eps=4)`` however lossy (the
    first version's ``(dt/dx)/b`` read ``eps (1 + q)`` there), 1 in vacuum and all through an
    ``Absorber``'s graded loss. And a closed column half filled with ``eps = 12``, or with
    ``mu = 12``, holds to round-off."""
    dx = 0.05
    scene = fd.Scene(
        domain=fd.Domain(size=(dx, dx, 2.0 + 16 * dx)),
        boundaries=fd.Boundaries(x=fd.Axis1D(k=0.0), y=fd.Axis1D(k=0.0),
                                 z=fd.Absorber(cells=8)),
        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=4),
        structures=(fd.Structure(fd.Slab(z=(-0.5, 0.5)), fd.Conductor(eps=4.0, sigma=20.0)),),
        initial=fd.InitialField(E=(lambda x, y, z: np.exp(-(z / 0.2) ** 2), None, None)),
        monitors=(fd.Record('p', at=(0.0, 0.0, 0.1), component='Ex'),))
    rt = runtime('numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = make_plan(scene)
    params = lower(scene, p, rt)
    eps = np.broadcast_to(p.energy._weight(params, 'E', rt), (3, p.nz, 1, 1))[:, :, 0, 0]
    mu = np.broadcast_to(p.energy._weight(params, 'H', rt), (3, p.nz, 1, 1))[:, :, 0, 0]
    inside = [k for k in range(p.nz) if abs(np.asarray(params['ae'])[0, k, 0, 0] - 1.0) > 1e-3
              and 8 + 2 < k < p.nz - 8 - 2]
    assert inside, 'the conductor is lossy'
    assert np.allclose(eps[0, inside], 4.0, rtol=0, atol=1e-12), eps[0, inside]
    outside = [k for k in range(p.nz) if k not in inside and k not in range(9, p.nz - 9)]
    assert np.allclose(eps[:, outside], 1.0, rtol=0, atol=1e-12)
    assert np.allclose(mu, 1.0, rtol=0, atol=1e-12)
    assert _swing(_half_filled()) < WEIGHT_FLAT
    assert _swing(_half_filled(eps=1.0, mu=12.0)) < WEIGHT_FLAT


def test_mutant_weights_of_one_are_caught_by_the_flatness_gate(monkeypatch):
    """Unweighted (the review's surviving mutant: nothing in the fast tier moved under it), the
    ``eps`` half-filled column swings 6.8; with only H unweighted the ``mu`` one swings 6.6. The
    gate above fails either way."""
    def ones(self, params, field, rt, comps=slice(None), box=None, times=None, key='',
             only=('E', 'H')):
        if field not in only:
            return _REAL_WEIGHT(self, params, field, rt, comps, box, times, key)
        return 1.0 if times is None else self._cut(params[times], comps,
                                                   self.box if box is None else box)

    monkeypatch.setattr(InteriorEnergy, '_weight', ones)
    assert _swing(_half_filled()) > 1.0
    monkeypatch.setattr(InteriorEnergy, '_weight', functools.partialmethod(ones, only=('H',)))
    assert _swing(_half_filled(eps=1.0, mu=12.0)) > 1.0


def test_the_judged_cells_are_all_but_the_pmls():
    """``plan.energy`` judges every cell but a PML's: an ``Absorber`` is a passive loss backed
    by its wall, whose energy is physical (a PML's stretch is not). A column with a 12-cell PML
    below and an 8-cell Absorber above judges from cell 12 to the top."""
    dx = 0.05
    scene = fd.Scene(
        domain=fd.Domain(size=(dx, dx, 1.0 + 20 * dx)),
        boundaries=fd.Boundaries(x=fd.Axis1D(k=0.0), y=fd.Axis1D(k=0.0),
                                 z=(fd.PML(cells=12, wavelength=1.0), fd.Absorber(cells=8))),
        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=4))
    p = make_plan(scene)
    assert (p.energy.lo, p.energy.hi) == ((12, 0, 0), (p.nz, 1, 1)), (p.energy.lo, p.energy.hi)


def _kicked(cells=32, time=12.0):
    """A pulse kicked in a vacuum column, leaving through ``Absorber(cells)`` faces: part of
    it returns after the layer's round trip (``2 cells dx``, 3.2 wavelengths at 32 cells)."""
    dx = 1 / 20
    return fd.Scene(
        domain=fd.Domain(size=(dx, dx, 2.0 + 2 * cells * dx)),
        boundaries=fd.Boundaries(x=fd.Axis1D(k=0.0), y=fd.Axis1D(k=0.0),
                                 z=fd.Absorber(cells=cells)),
        resolution=fd.Resolution(dx=dx), run=fd.Run(time=time),
        initial=fd.InitialField(E=(lambda x, y, z: np.exp(-(z / 0.15) ** 2)
                                   * np.cos(2 * math.pi * z), None, None)),
        monitors=(fd.Record('r', at=(0, 0, 0), component='Ex'),
                  fd.Flux('f', plane=fd.Plane(z=0.5), wavelengths=(1.0,))))


#: The most the kicked column's sampled energy may rise from one row to the next, over its
#: peak. Measured 1.4e-16 with the Absorber counted (round-off: the exact energy only falls
#: in its graded electric and magnetic loss); 1.6e-3 without it, when the layer's return
#: re-enters an interior the pulse has left.
ABSORBER_RISE = 1e-10


def test_an_absorbers_return_is_not_growth():
    """The review's second passive false alarm: an ``Absorber`` holds what it has not yet
    absorbed and hands part of it back (``Absorber(32)`` read 2.57 and warned). Counted, the
    energy only falls: no row rises by more than the H pair's ripple, and judged at every run
    length it reads at most 0.96."""
    res, caught = _run_collecting(_kicked())
    assert not [w for w in caught if w.category is fd.GrowingFieldWarning], caught
    t, W = res.diagnostics['interior_energy']
    assert float(np.max(np.diff(W)) / W.max()) < ABSORBER_RISE
    period = _assemble._scene_period(_kicked(), res.plan)
    ratios = [_post.growth_ratio(t[:j], W[:j], 0.0, period, GROWTH_WINDOWS, GROWTH_FLOOR)
              for j in range(8, t.size + 1)]
    assert max(r[0] for r in ratios if r is not None) < 1.0


def test_mutant_leaving_the_absorber_out_is_caught(monkeypatch):
    """The judged cells cut at the Absorber as the first version cut them: the layer's return
    re-enters an empty interior, a row rises 1.6e-3 of the peak, and the gate above fails."""
    from wavebit.fdtd._prepare.plan import Plan
    real = Plan.energy.fget

    def without(self):
        cells = dict(self.absorbing_faces)
        n = {'x': self.nx, 'y': self.ny, 'z': self.nz}
        return dataclasses.replace(real(self), lo=tuple(cells.get(f'{a}-', 0) for a in 'zyx'),
                                   hi=tuple(n[a] - cells.get(f'{a}+', 0) for a in 'zyx'))

    monkeypatch.setattr(Plan, 'energy', property(without))
    res, _ = _run_collecting(_kicked())
    t, W = res.diagnostics['interior_energy']
    assert float(np.max(np.diff(W)) / W.max()) > ABSORBER_RISE


# ---------------------------------------------------------------------------------------------
# When a source stops driving
# ---------------------------------------------------------------------------------------------

def _two_pulses(delay_second=38.0, periods=50):
    """Two dipoles in an open box, the second delayed to fire in the run's last quarter."""
    box = dipole_box(size=0.4, periods=periods, monitors=(
        fd.Flux('in', box=fd.Region(size=(0.3,) * 3), wavelengths=(1.0,)),
        fd.Record('p', at=(0.1, 0.05, 0.0), component='Ez')))
    first = box.sources[0]
    second = first.replace(name='d2', delay=delay_second)
    return box.replace(sources=(first, second))


def test_a_source_that_fires_late_is_not_growth():
    """The span is judged only after **every** source has ended (``_assemble.sources_end``):
    a second dipole fired at 38 (its envelope ends at 45.9) leaves 4.1 periods of a 50-period
    run, over which the interior's energy holds (measured 1.00004: the dipoles' static
    residue, the radiated pulses gone): nothing warns growing."""
    scene = _two_pulses()
    assert abs(_assemble.sources_end(scene) - (38.0 + 2 * scene.sources[0].band.sigmas
                                               / (2 * math.pi * 0.15))) < 1e-12
    res, caught = _run_collecting(scene)
    assert not [w for w in caught if w.category is fd.GrowingFieldWarning], caught
    assert res.diagnostics['interior_growth'] < 1.001, res.diagnostics['interior_growth']


def test_mutant_judging_the_whole_run_calls_a_late_source_growth(monkeypatch):
    """The statistic reading the whole run instead of the span after every source ended
    (``sources_end`` 0): the second pulse lands in the last quarter, over a third quarter where
    the first had long left the box (measured 2.9e5), and the passive gate above fails on its
    ratio. (It no longer warns growing: the pulse arrives, it does not rise as an exponential,
    and its spread -- ``_post.growth_ratio`` -- is over ``GROWTH_STEADY``. The first version,
    which read the ratio alone, warned for both monitors.)"""
    monkeypatch.setattr(_assemble, 'sources_end', lambda *args: 0.0)
    res, caught = _run_collecting(_two_pulses())
    assert res.diagnostics['interior_growth'] > 1e3, res.diagnostics['interior_growth']
    assert res.diagnostics['interior_spread'] > GROWTH_STEADY, res.diagnostics


def _tfsf_echo(time, gap=3.5, inside=None):
    """The review's scene: a vacuum TFSF box (0.5 on a side), ``-z``, over a glass half-space
    ``gap`` below it, PML all round; ``inside`` adds a ``(z-span, eps)`` layer across the box."""
    dx, pml, margin = 0.04, 12, 0.3
    lo, hi = -0.25 - gap - 0.5, 0.25 + margin
    size = (0.5 + 2 * margin + 2 * pml * dx,) * 2 + (hi - lo + 2 * pml * dx,)
    structures = [fd.Structure(fd.Slab(z=(lo - pml * dx - dx, -0.25 - gap)),
                               fd.Material(eps=2.25))]
    if inside is not None:
        structures.append(fd.Structure(fd.Slab(z=inside[0]), fd.Material(eps=inside[1])))
    source = fd.TFSF('t', box=fd.Region(size=(0.5, 0.5, 0.5)), direction='-z', polarization='x',
                     band=fd.Band(wavelength=1.0, df=0.2))
    # The internal-layer case measures field decay and the source-column echo,
    # not whole-object power. Its transverse slab cuts the box, so C3 correctly
    # refuses a closed Flux surface there; a Field volume retains that purpose.
    monitor_type = fd.Flux if inside is None else fd.Field
    return fd.Scene(domain=fd.Domain(size=size, center=(0, 0, 0.5 * (hi + lo))),
                    boundaries=fd.Boundaries.open(fd.PML(cells=pml)),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(time=time),
                    structures=tuple(structures), sources=(source,),
                    monitors=(monitor_type('f', box=fd.Region(size=(0.7, 0.7, 0.7)),
                                           wavelengths=(1.0,)),))


def _every_length(scene, res, end=None):
    """``[(run length, growth ratio)]`` at every run length of ``res`` (a run's prefix is the
    shorter run): the column series cut to that length, the end read off them as
    ``decay_report`` does, or ``end(prefix sources)`` when given."""
    t, W = res.diagnostics['interior_energy']
    period = _assemble._scene_period(scene, res.plan)
    rt = runtime('numpy')
    out = []
    for j in range(8, t.size + 1):
        m = int(round(t[j - 1] / res.plan.dt)) + 1
        cut = {name: dataclasses.replace(s, t=s.t[:m], incident=s.incident[:m],
                                         t_H=s.t_H[:m], incident_H=s.incident_H[:m])
               for name, s in res.sources.items()}
        start = _assemble.sources_end(scene, cut, rt) if end is None else end(cut)
        got = _post.growth_ratio(t[:j], W[:j], start, period, GROWTH_WINDOWS, GROWTH_FLOOR)
        if got is not None:
            out.append((float(t[j - 1]), got[0]))
    return out


@functools.lru_cache(maxsize=None)
def _tfsf_run(inside=None):
    scene = _tfsf_echo(16.0, inside=inside)
    return scene, _run_collecting(scene)


def _envelope_end(scene):
    band = scene.sources[0].band
    return 2 * band.sigmas / (2 * math.pi * float(band.df))


def test_a_tfsf_box_drives_while_its_column_changes():
    """The review's first passive false alarm: glass 3.5 below a vacuum TFSF box reflects 4%
    of the pulse back up the column at t = 9 .. 13, and the box re-injects it into an interior
    the pulse had left -- judged from the closed-form end (6.7) it read 4.04 and warned. The
    end is read off the column's own series (``sources_end``): the last time its entry-plane
    E or H still changes by ``COLUMN_LEVEL`` of its largest change, 15.6, after the echo. At
    every run length to 16 nothing is judged over the line, and nothing warns."""
    scene, (res, caught) = _tfsf_run()
    end = _assemble.sources_end(scene, res.sources, runtime('numpy'))
    assert end > 15.0, end                                   # measured 15.58
    assert not [w for w in caught if w.category is fd.GrowingFieldWarning], caught
    assert all(r <= GROWTH_THRESHOLD for _, r in _every_length(scene, res))


def test_mutant_a_tfsf_ended_with_its_envelope_calls_its_echo_growth():
    """Ended with its waveform's envelope (5.9), as a Dipole is, the same run is judged over its
    echo and reads over the line -- 129 at t = 10.4 (the review's closed-form end, 6.7, read
    4.04 at 11.6): the gate above fails."""
    scene, (res, _) = _tfsf_run()
    ratios = _every_length(scene, res, end=lambda cut: _envelope_end(scene))
    assert max(r for _, r in ratios) > GROWTH_THRESHOLD, max(r for _, r in ratios)


def test_the_entry_plane_sees_an_echo_through_a_layer_inside_the_box():
    """A glass layer across the box and the half-space below it: the echo from below crosses
    the in-box layer and the entry plane reads what it lets through, so the end moves past the
    echo (measured 15.6 without the layer, 16.0 with it) and nothing is judged over the line at
    any run length -- where, ended with its envelope, it reads 143 at t = 10.0."""
    inside = ((-0.1, 0.0), 2.25)
    scene, (res, caught) = _tfsf_run(inside)
    bare_scene, (bare, _) = _tfsf_run()
    rt = runtime('numpy')
    end = _assemble.sources_end(scene, res.sources, rt)
    assert end >= _assemble.sources_end(bare_scene, bare.sources, rt), end
    assert not [w for w in caught if w.category is fd.GrowingFieldWarning], caught
    assert all(r <= GROWTH_THRESHOLD for _, r in _every_length(scene, res))


def test_a_plane_waves_end_is_read_off_its_column_too():
    """A PlaneWave sheet injects its column's field wherever it is, and an ``Absorber`` end of
    the column returns 5% of it: the sheet drives past its envelope's end (4.73) until the
    column's echoes have died: 17.7 (8.97 with ``PML(12)`` ends)."""
    dx = 1 / 20
    pw = fd.PlaneWave('pw', z=0.8, direction='-z', polarization='s',
                      band=fd.Band(wavelength=1.0, df=0.25))
    scene = fd.Scene(domain=fd.Domain(size=(dx, dx, 2.0 + 24 * dx)),
                     boundaries=fd.Boundaries(**fd.Bloch.from_source(pw),
                                              z=fd.Absorber(cells=12)),
                     resolution=fd.Resolution(dx=dx), run=fd.Run(time=30.0), sources=(pw,),
                     monitors=(fd.Record('r', at=(0, 0, 0), component='Ey'),))
    res, _ = _run_collecting(scene)
    end = _assemble.sources_end(scene, res.sources, runtime('numpy'))
    assert end > _envelope_end(scene) + 5.0, end


def test_a_column_source_without_values_has_no_end():
    """Under a trace the column's series has no value, so the end is unknown and nothing is
    judged; with no series at all, the same."""
    scene = _tfsf_echo(4.0)
    assert _assemble.sources_end(scene) is None
    assert _assemble.sources_end(scene, {}, runtime('numpy')) is None


def _oblique(time=20.0):
    """``fd_helpers.bloch_plane_wave`` at 30 degrees over a glass slab: the column holds a
    static H at 1.2e-7 of its peak level for good, and the slab keeps the interior over the
    floor long enough to be judged."""
    scene, _ = bloch_plane_wave(
        theta=30.0, length=3.0, steps=10,
        structures=(fd.Structure(fd.Slab(z=(-1.0, -0.5)), fd.Material(eps=2.25)),),
        monitors=(fd.Flux('f', plane=fd.Plane(z=0.8), wavelengths=(1.0,)),))
    return scene.replace(run=fd.Run(time=time))


def test_an_oblique_plane_waves_end_is_read_from_its_columns_change():
    """The column's change, not its level: an oblique column's static H residue (1.2e-7 of its
    peak level, over :data:`COLUMN_LEVEL`) is injected at constant energy and feeds nothing,
    and the change does not see it -- the end is 11.02, well before the run's 20, and the
    interior is judged (0.72)."""
    res, caught = _run_collecting(_oblique())
    end = res.diagnostics['sources_end']
    assert end < 12.0, end
    assert res.diagnostics['interior_growth'] is not None, res.diagnostics
    assert not [w for w in caught if w.category is fd.GrowingFieldWarning], caught


def _level_end(series, rt):
    """Mutant of ``_assemble._column_end``: the column's level ``|x|^2`` read for its change."""
    end, (t, change) = _REAL_COLUMN_END(series, rt)
    for values, times in ((series.incident, series.t), (series.incident_H, series.t_H)):
        level = np.sum(np.abs(np.asarray(values)) ** 2, axis=1)
        last = int(np.nonzero(level > _assemble.COLUMN_LEVEL * level.max())[0][-1])
        end = max(end, float(np.asarray(times)[last]))
    return end, (t, change)


_REAL_COLUMN_END = _assemble._column_end


def test_mutant_a_column_ended_by_its_level_never_ends(monkeypatch):
    """Ended by its level, the oblique column's static residue keeps it driving to the run's
    last step: the end is the run's, and the interior is never judged -- the gate above
    fails."""
    monkeypatch.setattr(_assemble, '_column_end', _level_end)
    res, _ = _run_collecting(_oblique())
    assert res.diagnostics['sources_end'] > 19.9, res.diagnostics['sources_end']
    assert res.diagnostics['interior_growth'] is None


def _slab_plane_wave(time=200.0, stabilize=None):
    """The growing slab column of ``tests/fdtd/test_fd_pml_stability.py`` (eps 1.5, 0.08 thick,
    ``Axis1D`` kx = 2 pi / 0.8, ``PML(12)`` z faces) driven by an oblique PlaneWave in place
    of its kick: wavelength 0.6 at 48.6 degrees (the review's ``pw_growth.py``)."""
    dx, gap, cells, thickness, wavelength = 0.04, 0.24, 12, 0.08, 0.6
    theta = math.degrees(math.asin(wavelength / 0.8))
    pw = fd.PlaneWave('pw', z=gap / 2, direction='-z', theta=theta, polarization='p',
                      band=fd.Band(wavelength=wavelength, df=0.1))
    return fd.Scene(
        domain=fd.Domain(size=(dx, dx, thickness + 2 * gap + 2 * cells * dx),
                         center=(0.0, 0.0, -thickness / 2)),
        boundaries=fd.Boundaries(**fd.Bloch.from_source(pw, axis1d=('x', 'y')),
                                 z=fd.PML(cells=cells, wavelength=1.0, stabilize=stabilize)),
        resolution=fd.Resolution(dx=dx), run=fd.Run(time=time),
        structures=(fd.Structure(fd.Slab(z=(-thickness, 0.0)), fd.Material(eps=1.5)),),
        sources=(pw,),
        monitors=(fd.Flux('f', plane=fd.Plane(z=-thickness - 0.1), wavelengths=(wavelength,)),
                  fd.Record('r', at=(0, 0, -thickness / 2), component='Ez')))


def _growing(caught):
    return [str(w.message) for w in caught if w.category is fd.GrowingFieldWarning]


def test_a_plane_wave_whose_column_holds_is_judged_from_its_envelope():
    """The review's regression: the growing slab column driven by a PlaneWave. Its column (the
    uniform background under the same pure-stretch faces) never goes quiet -- the stretch
    holds an evanescent field at its entry plane, 1.5e-4 in amplitude, whose one-step change
    stays at 1.8e-8 of its peak, just over :data:`COLUMN_LEVEL` -- so its end was the run's
    and the scene, growing 14000-fold to t = 400, said only "run longer". A column that
    holds (its change's eighths within ``GROWTH_THRESHOLD`` of each other, over quarters of a
    round trip each: ``_assemble._column_state``) injects at constant energy, which cannot
    lift a passive interior steadily: the interior is judged from the envelope's end (11.83),
    reads 4.57 at t = 200 (spread 1.05) and warns, naming the knob. The stabilized twin's
    column falls silent at 34.7 and nothing warns."""
    res, caught = _run_collecting(_slab_plane_wave())
    assert abs(res.diagnostics['sources_end'] - _envelope_end(_slab_plane_wave())) < 1e-9
    assert res.diagnostics['interior_growth'] > 3.0, res.diagnostics['interior_growth']
    growing = _growing(caught)
    assert growing and all('PML(stabilize=3e-3)' in m for m in growing), growing
    res, caught = _run_collecting(_slab_plane_wave(stabilize=3e-3))
    assert not _growing(caught), caught
    assert res.diagnostics['sources_end'] > 30.0, res.diagnostics['sources_end']


def test_mutant_a_holding_column_kept_driving_hides_the_growth(monkeypatch):
    """The column's state ignored (its end kept even when it holds): the end is the run's, the
    interior is never judged, and the growing scene warns nothing but "run longer" -- the gate
    above fails."""
    monkeypatch.setattr(_assemble, '_column_state', lambda *args: (None, None))
    res, caught = _run_collecting(_slab_plane_wave())
    assert res.diagnostics['sources_end'] > 199.0 and not _growing(caught)


def test_a_columns_state_is_growing_holding_or_neither():
    """``_assemble._column_state`` on built series, quarters of 2 time units: an exponential rise
    (``e^2`` a quarter, spread 1.00) is growing; a constant oscillation holds; a ring-down and a
    pulse passing (an echo) are neither. A growing column warns, naming the source's column
    (below)."""
    t = np.arange(0.0, 20.0, 0.01)
    wiggle = 1.0 + 0.5 * np.cos(40.0 * t)
    grow = 1e-4 * np.exp(t) * wiggle
    assert _assemble._column_state((t, grow), 4.0, 1.0, 2.0)[0] == 'growing'
    assert _assemble._column_state((t, 1e-4 * wiggle), 4.0, 1.0, 2.0)[0] == 'holding'
    assert _assemble._column_state((t, np.exp(-t) * wiggle), 4.0, 1.0, 0.5)[0] is None
    echo = 1e-6 + 1e-3 * np.exp(-((t - 16.0) / 1.0) ** 2) * wiggle
    assert _assemble._column_state((t, echo), 4.0, 1.0, 2.0)[0] is None


def _column_scene(theta=30.0, eps=2.25, length=1.6):
    """``(scene, source)``: a ``length``-long column under a ``PlaneWave`` at ``theta`` over an
    ``eps`` slab -- the scene :func:`_assemble._round_trip` is pinned on below."""
    pw = fd.PlaneWave('pw', z=0.5, direction='-z', theta=theta, polarization='p',
                      band=fd.Band(wavelength=0.6, df=0.1))
    scene = fd.Scene(
        domain=fd.Domain(size=(0.05, 0.05, length)),
        boundaries=fd.Boundaries(**fd.Bloch.from_source(pw, axis1d=('x', 'y')),
                                 z=fd.PML(cells=12, wavelength=1.0)),
        resolution=fd.Resolution(dx=0.05), run=fd.Run(time=10.0),
        structures=(fd.Structure(fd.Slab(z=(-0.2, 0.0)), fd.Material(eps=eps)),),
        sources=(pw,))
    return scene, pw


def test_a_columns_round_trip_is_down_the_column_and_back():
    """``_assemble._round_trip`` is ``2 L / v``, not one crossing: the closed form at the
    slowest axial group speed ``v = sqrt(n^2 - s^2) / n^2`` over the scene's materials
    (``s = sin theta``), for the 1.6-long column above at 30 degrees over ``n = 1.5``.
    The factor of 2 is the value the test below is about; halved, it is 2.5456."""
    scene, pw = _column_scene()
    s = math.sin(math.radians(30.0))
    slowest = min(math.sqrt(n * n - s * s) / (n * n) for n in (1.0, 1.5))
    assert _assemble._round_trip(scene, pw, None) == pytest.approx(2.0 * 1.6 / slowest,
                                                                   rel=1e-12)
    assert _assemble._round_trip(scene, pw, None) == pytest.approx(5.091168824543142,
                                                                   rel=1e-12)


#: The round trip of :func:`_column_scene`, the time unit the echo below is judged by.
ROUND_TRIP = 5.091168824543142


def _arriving_echo(stop, arrival=21.0, width=1.2):
    """A column whose echo arrives at ``arrival``: a one-sided exponential, so its leading edge
    rises as steadily as a growth does (spread 1.00) and it is the shape the span has to be
    long enough to see the whole of."""
    t = np.arange(0.0, stop, 0.004)
    return t, (1e-12 + np.exp(-np.abs(t - arrival) / width)) * (1.0 + 0.5 * np.cos(40.0 * t))


def test_a_column_is_judged_over_a_whole_round_trip_so_an_arriving_echo_is_not_growth():
    """A quarter of at least ``2 L / v`` is what makes an echo come *and go* inside the judged
    span (``_column_state``). The echo above, at a run length whose span is 15 time units:
    over quarters of one crossing (2.55) it is judged at once and reads growing, 22.6 with a
    spread of 1.00 -- steadier than any growth measured -- and the scene would warn of a
    passive echo; over quarters of the round trip (5.09) the span is short of four windows and
    nothing is judged, so the column end stands. Run to 30 the echo has passed and the same
    series reads 0.068 over the round trip: neither."""
    change = _arriving_echo(19.0)
    assert _assemble._column_state(change, 4.0, 0.6, ROUND_TRIP) == (None, None)
    kind, judged = _assemble._column_state(change, 4.0, 0.6, ROUND_TRIP / 2.0)
    assert kind == 'growing' and judged[0] > 20.0, (kind, judged)
    assert judged[4] < 1.1, judged[4]                       # spread 1.002
    kind, judged = _assemble._column_state(_arriving_echo(30.0), 4.0, 0.6, ROUND_TRIP)
    assert kind is None and judged is not None and judged[0] < 0.1, (kind, judged)


def _bumped_column(stop=40.0, start=4.0):
    """A column whose change bumps by half through the second eighth of the last half and
    settles back: eighth means 1.00, 1.34, 1.00, 0.95."""
    t = np.arange(0.0, stop, 0.005)
    q = (t[-1] - start) / 4.0
    centre = t[-1] - 2.0 * q + 0.75 * q
    return t, 1e-4 * (1.0 + 0.5 * np.exp(-((t - centre) / (0.28 * q)) ** 2)) * (
        1.0 + 0.5 * np.cos(40.0 * t))


def test_a_holding_column_holds_to_the_growth_line_and_not_to_a_factor_of_two():
    """``'holding'`` means *steady*, to within ``GROWTH_THRESHOLD`` (1.1) across the four
    eighths of the last half -- that is the whole of its licence to judge the interior from
    the envelope's end instead. The bumped column above spreads its eighths by 1.40: inside a
    band of ``GROWTH_STEADY`` (2) and outside the committed one, so a wider band would call it
    a steady injection. Its quarter ratio is 0.84, under the growth line, so ``'growing'`` is
    not what saves it: at 1.1 it is ``None`` and the column end stands."""
    change = _bumped_column()
    kind, judged = _assemble._column_state(change, 4.0, 2.0, 2.0)
    assert kind is None, kind
    assert judged is not None and judged[0] < GROWTH_THRESHOLD, judged
    t, e = change
    edges = np.linspace(judged[1], judged[3], 5)
    means = [float(np.mean(e[(t >= a) & (t <= b)])) for a, b in zip(edges[:-1], edges[1:])]
    assert GROWTH_THRESHOLD * min(means) < max(means) <= GROWTH_STEADY * min(means), means


def test_a_growing_column_warns_naming_the_source(monkeypatch):
    """A column that grows is a growing scene on its own, whatever the interior reads: made to
    say so (``_column_state`` returning ``'growing'`` for whatever it judged) on the stabilized
    slab PlaneWave, whose interior falls, the run warns ``GrowingFieldWarning`` naming the
    source's own column and the knob, and ``column_growth`` holds the column's ratio."""
    real = _assemble._column_state

    def growing(change, start, wavelength, round_trip):
        kind, judged = real(change, start, wavelength, round_trip)
        return ('growing', judged) if judged is not None else (kind, judged)

    monkeypatch.setattr(_assemble, '_column_state', growing)
    res, caught = _run_collecting(_slab_plane_wave(time=33.0, stabilize=3e-3))
    interior = res.diagnostics['interior_growth']
    assert interior is None or interior < 1.0, interior
    messages = _growing(caught)
    assert messages and all("source 'pw''s own column" in m for m in messages), messages
    assert all('Raise stabilize' in m for m in messages), messages
    assert set(res.diagnostics['column_growth']) == {'pw'}


# ---------------------------------------------------------------------------------------------
# An overflow is growth
# ---------------------------------------------------------------------------------------------

def _huge(amplitude=1e160):
    """A closed PEC column whose energy is past float64's range from the first sample."""
    dx = 0.05
    return fd.Scene(
        domain=fd.Domain(size=(dx, dx, 1.0)),
        boundaries=fd.Boundaries(x=fd.Axis1D(k=0.0), y=fd.Axis1D(k=0.0), z=fd.PEC()),
        resolution=fd.Resolution(dx=dx), run=fd.Run(time=4.0),
        initial=fd.InitialField(E=(lambda x, y, z: amplitude * np.cos(math.pi * z), None,
                                   None)),
        monitors=(fd.Record('p', at=(0.0, 0.0, 0.1), component='Ex'),))


def test_an_overflowed_run_warns_growing():
    """An interior energy that is no longer finite is the largest growth there is: one
    ``GrowingFieldWarning`` saying the field overflowed and from when, ``interior_growth``
    infinite -- not ``None`` and silent, as the first version left it."""
    res, caught = _run_collecting(_huge())
    growing = [str(w.message) for w in caught if w.category is fd.GrowingFieldWarning]
    assert len(growing) == 1 and 'OVERFLOWED' in growing[0], growing
    assert res.diagnostics['interior_growth'] == math.inf
    assert not [w for w in caught if w.category is fd.UnsettledFieldWarning]
    quiet, caught = _run_collecting(_huge(1.0))
    assert not caught and quiet.diagnostics['interior_growth'] != math.inf


def test_a_run_the_stability_guard_voided_is_not_called_an_overflow(monkeypatch):
    """A run past its stability bound under a traced material comes back ``nan`` on purpose
    (``diagnostics['stable']`` False); that is not an overflow and does not warn one."""
    from wavebit.fdtd import _execution as _run
    monkeypatch.setattr(_run, '_stable', lambda params, rt: False)
    res, caught = _run_collecting(_huge())
    assert not [w for w in caught if w.category is fd.GrowingFieldWarning], caught
    assert res.diagnostics['stable'] is False


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

def _grating_probe(cells, reflection, periods=384, stabilize=None):
    """``(peak over the period ending at half the run, peak over the last period, the
    interior's growth, the probe's own, [GrowingFieldWarning text, ...])`` of a probe below a
    lamellar tooth, p at normal incidence, the z faces ``PML(cells, reflection)``."""
    scene, _ = box_grating('lamellar', polarization='p', periods=periods, wavelengths=(1.0,))
    scene = scene.replace(
        monitors=(fd.Record('b', at=(0.3, 0.0, -0.1), component='Ez'),),
        boundaries=fd.Boundaries(x=scene.boundaries.face('x-'), y=scene.boundaries.face('y-'),
                                 z=fd.PML(cells=cells, reflection=reflection,
                                          stabilize=stabilize)))
    res, caught = _run_collecting(scene)
    v = np.abs(to_numpy(res['b'].values))
    per = int(round(1.0 / res.plan.dt))
    half = len(v) // 2
    growing = [str(w.message) for w in caught if w.category is fd.GrowingFieldWarning]
    return (float(v[half - per:half].max()), float(v[-per:].max()),
            res.diagnostics['interior_growth'], res.diagnostics['growth']['b'], growing)


@pytest.mark.slow
def test_a_lossless_grating_under_the_default_pml_grows_and_says_so():
    """The kernel defect, pinned: under the default pure ``PML(12, 1e-8)`` the probe grows
    (measured 8.4e-4 at 192 periods -> 2.6e-3 at 384), and the run says so -- the interior's
    energy reads 2.64 (its mean over the last 94 periods, against the 94 before, with the
    growing mode still at 1e-5 of the energy the pulse left behind), the probe's own 2.29,
    and the warning names the stabilized layer."""
    middle, last, interior, own, growing = _grating_probe(12, 1e-8)
    assert last > 2 * middle, (middle, last)                # measured 8.4e-4 -> 2.6e-3
    assert interior > GROWTH_THRESHOLD, interior            # measured 2.64
    assert own > 1.0, own                                   # measured 2.29
    assert len(growing) == 1 and "monitor 'b'" in growing[0], growing
    assert 'PML(stabilize=3e-3)' in growing[0] and 'longer run does not settle' in growing[0]


@pytest.mark.slow
def test_the_same_grating_under_a_stabilized_pml_does_not_grow():
    """``PML(stabilize=3e-3)`` on the same layer: the growth is gone, and nothing warns it --
    the interior's energy is not even judged, having fallen to 3.3e-8 of its peak (under
    ``GROWTH_FLOOR``; over the floor it would read 0.10)."""
    middle, last, interior, _, growing = _grating_probe(12, 1e-8, stabilize=3e-3)
    assert last <= middle, (middle, last)                   # measured 4.4e-4 -> 9.7e-5
    assert growing == [] and interior is None, (interior, growing)


@pytest.mark.slow
def test_the_same_grating_under_a_24_cell_1e_12_pml_does_not_grow():
    middle, last, interior, _, growing = _grating_probe(24, 1e-12)
    assert last <= middle, (middle, last)                   # measured 4.8e-4 -> 3.0e-4
    assert growing == [] and interior < 1.0, (interior, growing)   # measured 0.52
