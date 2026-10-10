"""``GainCutoffWarning``: an order of a gain half-space near grazing, and nowhere else.

A gain half-space takes Maystre's root, the root of switching the gain on from zero
(``boundary_kz``). Near grazing no plane-wave value is the answer: a finite beam's response
depends on its width there. The band is ``|Re(kz**2)| < 2*|Im(kz**2)|`` per order, and the
measurements behind the 2 are in ``studies/gain_cutoff_band.py``. These tests pin where the
warning fires, what it never changes, and that it stays one report per sweep.

The planar scene is glass (``eps = 2.25``) onto ``eps = 1 - 0.05i``, TE: ``kz**2 = 1 - 0.05i -
2.25*sin(theta)**2``, so the band runs from 39.23 to 44.36 degrees, and Maystre's root flips at
``asin(2/3)`` = 41.81 degrees. Each threshold test names the mutant it kills.
"""
import warnings

import numpy as np
import pytest

import wavebit
from wavebit import GainCutoffWarning, Material

GAIN = 1 - 0.05j
CRITICAL = float(np.degrees(np.arcsin(2 / 3)))


def _planar(theta_deg, bot=GAIN, tetm='te', backend='numpy', **kw):
    return wavebit.rcwa(backend=backend, tetm=tetm, top=2.25, bot=bot, theta=np.deg2rad(theta_deg),
                        truncation_order=0, wavelength=500.0, period=[1000.0],
                        structure=np.full((1, 1, 2), 2.25), thickness=[100.0], **kw)


def _gain_warnings(solver, **solve_kw):
    """The solve's ``GainCutoffWarning`` messages, recorded with every filter open."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        result = solver.solve(**solve_kw)
    return [str(r.message) for r in rec if issubclass(r.category, GainCutoffWarning)], result


@pytest.mark.parametrize('tetm', ['te', 'tm'])
@pytest.mark.parametrize('theta_deg, fires', [
    (20.0, False), (30.0, False), (39.0, False),
    (41.0, True), (CRITICAL, True), (42.5, True),
    (44.0, True),            # ratio 1.71: kills a band of 1
    (45.0, False),           # ratio 2.5, the pinned R = 1.2646: kills a band of 3
    (60.0, False),
])
def test_the_band_on_one_interface(theta_deg, fires, tetm):
    """In the band and only there, on either polarisation.

    42.5 deg sits at ratio 0.53, so a band of 0.5 goes silent there. The exact tie
    ``asin(2/3)`` puts ``kz**2`` on Maystre's cut itself, which the band must contain.
    """
    messages, _ = _gain_warnings(_planar(theta_deg, tetm=tetm))
    assert (len(messages) == 1) == fires, (theta_deg, messages)
    assert all('bottom' in m for m in messages)


@pytest.mark.parametrize('theta_deg, fires', [(39.18, False), (39.28, True), (44.31, True),
                                              (44.41, False)])
def test_the_band_edges(theta_deg, fires):
    """The edges sit at 39.23 and 44.36 deg; a constant moved by more than ~2% crosses one."""
    messages, _ = _gain_warnings(_planar(theta_deg))
    assert bool(messages) == fires, (theta_deg, messages)


def test_the_warning_changes_no_value():
    """Recorded, ignored or escalated outside the band, the numbers are the same numbers."""
    _, recorded = _gain_warnings(_planar(44.0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', GainCutoffWarning)
        ignored = _planar(44.0).solve()
    for name in ('R', 'T'):
        assert np.array_equal(np.asarray(getattr(recorded, name)), np.asarray(getattr(ignored, name)))
    with warnings.catch_warnings():
        warnings.simplefilter('error', GainCutoffWarning)
        flagship = _planar(45.0).solve()                    # outside the band: no raise
        with pytest.raises(GainCutoffWarning):
            _planar(44.0).solve()
    assert float(np.sum(flagship.R)) == pytest.approx(1.2646, abs=1e-4)


@pytest.mark.parametrize('theta_deg', [CRITICAL, 42.5])
def test_a_lossy_twin_is_silent(theta_deg):
    """The same interface with ``1 + 0.05i``: no gain, no warning.

    Its ``kz**2`` sits above the cut, so the radicand test alone is silent here too; the gain
    gate itself is pinned by :func:`test_a_passive_magnetic_metal_is_silent`."""
    messages, _ = _gain_warnings(_planar(theta_deg, bot=1 + 0.05j))
    assert messages == []


#: Passive (``Im(eps), Im(mu) > 0``) yet ``Im(eps*mu) < 0``: ``eps*mu = -0.205 - 0.999i``.
#: One real part is negative, so it is not a double negative.
_MAGNETIC_METAL = Material(-2 + 0.01j, mu=0.1 + 0.5j)


@pytest.mark.parametrize('theta_deg', [0.0, 30.0, 60.0])
def test_a_passive_magnetic_metal_is_silent(theta_deg):
    """Its ``kz**2`` sits inside the band at every angle (ratio 0.21 to 1.9), so only the gain
    gate keeps the solve silent and the mask all False.

    Kills dropping the gain gate on the solve path, and the public ``gain_cutoff_band()``
    reading ``n**2`` as a non-magnetic ``eps`` (which reads ``Im(eps*mu) < 0`` as gain).
    """
    solver = _planar(theta_deg, bot=_MAGNETIC_METAL)
    messages, _ = _gain_warnings(solver)
    assert messages == []
    masks = solver.gain_cutoff_band()
    assert not np.asarray(masks['bottom']).any() and not np.asarray(masks['top']).any()


def test_the_band_reads_the_material_not_the_radicand():
    """A passive medium with ``Re(eps), Re(mu) < 0`` has ``Im(eps*mu) < 0`` and no gain.

    The function answers for it, and a solve with such a far side stays silent
    (``tests/test_negative_index_half_space.py``)."""
    from wavebit._core.conventions import gain_cutoff_band
    nim = -1 + 0.1j
    k_par_sq = np.linspace(0.0, 3.0, 301) + 0j
    assert not gain_cutoff_band(np, nim, nim, k_par_sq).any()
    # The same radicand from gain does fire somewhere on this sweep.
    assert gain_cutoff_band(np, 0.99 - 0.2j, 1.0, k_par_sq).any()


def test_gain_in_mu_alone_counts():
    """``eps = 1``, ``mu = 1 - 0.05i`` is a gain medium too. Kills an epsilon-only gate."""
    solver = _planar(44.0, tetm='te', bot=Material(1.0, 1 - 0.05j))
    messages, _ = _gain_warnings(solver)
    assert len(messages) == 1
    assert np.asarray(solver.gain_cutoff_band()['bottom']).all()


def test_gain_eps_with_lossier_mu_is_silent():
    """``eps*mu`` above the real axis takes the passive root: nothing steps, nothing to say."""
    from wavebit._core.conventions import gain_cutoff_band
    k_par_sq = np.linspace(0.0, 3.0, 301) + 0j
    assert not gain_cutoff_band(np, 1 - 0.05j, 1 + 0.06j, k_par_sq).any()


# ----------------------------------------------------------------------- per order, per side

_GRATING = np.array([[[2.25] * 4 + [1.0] * 4]])


def _grating(wavelength, **kw):
    scene = dict(backend='numpy', tetm='te', top=1.0, bot=2.25 - 0.05j, theta=0.0,
                 truncation_order=5, period=[1000.0], structure=_GRATING, thickness=[100.0])
    scene.update(kw)
    return wavebit.rcwa(wavelength=wavelength, **scene)


@pytest.mark.parametrize('wavelength, fires', [(1400.0, False), (1500.005, True), (1600.0, False)])
def test_a_grating_order_grazing_in_the_gain_substrate(wavelength, fires):
    """At 1500.005 nm the +-1 orders graze the substrate; the specular order's ratio is 45.

    Kills a specular-only check.
    """
    messages, _ = _gain_warnings(_grating(wavelength))
    assert bool(messages) == fires, (wavelength, messages)


def test_the_gain_side_is_named_whichever_side_is_lit():
    """Gain on top, lit from below: the transmitted +-1 orders graze the top medium."""
    messages, _ = _gain_warnings(_grating(1500.005, top=2.25 - 0.05j, bot=1.0,
                                          incidence_side='bottom'))
    assert len(messages) == 1 and 'gain top half-space' in messages[0], messages


def test_the_mask_names_the_orders():
    """``gain_cutoff_band()`` marks exactly the two grazing orders of the substrate."""
    masks = _grating(1500.005).gain_cutoff_band()
    bottom = np.asarray(masks['bottom']).ravel()
    assert bottom.sum() == 2 and not np.asarray(masks['top']).any()
    assert list(np.flatnonzero(bottom)) == [4, 6]              # orders -1 and +1 of -5..5


def test_a_complex_angle_is_left_to_its_own_warning():
    """Formal R and T already carry ComplexIncidenceWarning; the band is about real k_par."""
    solver = wavebit.rcwa(backend='numpy', tetm='te', top=2.25, bot=GAIN, theta=np.deg2rad(44.0) + 0.01j,
                          truncation_order=0, wavelength=500.0, period=[1000.0],
                          structure=np.full((1, 1, 2), 2.25), thickness=[100.0])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        solver.solve()
    names = {r.category.__name__ for r in rec}
    assert 'ComplexIncidenceWarning' in names and 'GainCutoffWarning' not in names


# --------------------------------------------------------------------- sweeps and the ladder

def test_a_sweep_through_the_band_reports_once():
    """Constant text, so the default filter collapses a sweep on the default solve().

    Twenty in-band points under ``auto_detune=True``: one report. This also needs the ladder
    not to catch warnings, which ``test_rayleigh_guard`` pins separately.
    """
    solver = _planar(41.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('default')
        for theta_deg in np.linspace(41.0, 44.0, 20):
            solver.solve(theta=np.deg2rad(theta_deg))
    assert len([r for r in rec if issubclass(r.category, GainCutoffWarning)]) == 1


def test_the_band_is_not_a_detune_trigger():
    """A one-ULP nudge cannot leave a band a degree wide, so the ladder must not try."""
    solver = _planar(44.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        default = solver.solve()
    assert solver.wavelength == 500.0
    assert not any(r.category.__name__ == 'AutoDetuneWarning' for r in rec)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', GainCutoffWarning)
        exact = _planar(44.0).solve(auto_detune=False)
    assert np.array_equal(np.asarray(default.R), np.asarray(exact.R))


# --------------------------------------------------------------------------------- backends

@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_eager_jax_and_torch_warn_too(backend):
    if backend == 'jax':
        pytest.importorskip('jax')
    else:
        pytest.importorskip('torch')
    messages, _ = _gain_warnings(_planar(44.0, backend=backend))
    assert len(messages) == 1


def test_a_traced_solve_is_silent_and_the_mask_traces():
    """Under ``jax.jit`` the warning cannot read a value; the mask is the per-point verdict."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()
    solver = _planar(44.0, backend='jax')

    def bottom_mask(theta):
        solver.theta = theta
        return jnp.asarray(solver.gain_cutoff_band()['bottom']).ravel()

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        R = jax.jit(lambda t: jnp.sum(jnp.asarray(solver.solve(theta=t).R)))(jnp.deg2rad(44.0))
        inside = jax.jit(bottom_mask)(jnp.deg2rad(44.0))
        outside = jax.jit(bottom_mask)(jnp.deg2rad(45.0))
    assert not any(issubclass(r.category, GainCutoffWarning) for r in rec)
    assert bool(inside[0]) and not bool(outside[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', GainCutoffWarning)
        eager = float(np.sum(_planar(44.0).solve().R))
    assert float(R) == pytest.approx(eager, abs=1e-12)


# ------------------------------------------------------------------------------------ reach

@pytest.mark.parametrize('scene', ['te', 'conical', '2d', 'smm'])
def test_every_solve_path_checks_once(scene, monkeypatch):
    """One check per solve on each path the engine dispatches to; counted, not inferred."""
    from wavebit._core import engine
    calls = []
    original = engine.CoreFactorizationMixin._warn_gain_cutoff

    def counted(self, **kw):
        calls.append(1)
        return original(self, **kw)

    monkeypatch.setattr(engine.CoreFactorizationMixin, '_warn_gain_cutoff', counted)
    common = dict(backend='numpy', top=2.25, bot=GAIN, theta=np.deg2rad(44.0), wavelength=500.0,
                  thickness=[100.0])
    if scene == 'te':
        solver = _planar(44.0)
    elif scene == 'conical':
        solver = wavebit.rcwa(truncation_order=2, phi=0.3, period=[1000.0],
                              structure=np.full((1, 1, 2), 2.25), **common)
    elif scene == '2d':
        solver = wavebit.rcwa(truncation_order=(1, 1), period=(1000.0, 1000.0),
                              structure=np.full((1, 2, 2), 2.25), **common)
    else:
        solver = wavebit.rcwa(truncation_order=2, phi=0.3, period=[1000.0], layer_algorithm='SMM',
                              structure=np.full((1, 1, 2), 2.25), **common)
    messages, _ = _gain_warnings(solver)
    assert len(calls) == 1 and len(messages) == 1, (scene, len(calls), messages)


# ------------------------------------------------------------------------- CoherentSolver

def _coherent(source, bot=GAIN, **kw):
    return wavebit.rcwa(backend='numpy', top=2.25, bot=bot, truncation_order=(0, 0), wavelength=500.0,
                        period=(1000.0, 1000.0), structure=np.full((1, 2, 2), 2.25),
                        thickness=[100.0], excitation={'top': source}, **kw)


@pytest.mark.parametrize('theta_deg, fires', [(30.0, False), (44.0, True), (45.0, False)])
def test_coherent_plane_wave_checks_the_same_band(theta_deg, fires):
    """``CoherentSolver`` builds its own ports and runs the same test on them, with the same
    text. Kills a facade that skips the check."""
    source = wavebit.PlaneWave(theta=np.deg2rad(theta_deg), polarization=(1., 0.))
    messages, _ = _gain_warnings(_coherent(source))
    assert len(messages) == (1 if fires else 0), (theta_deg, messages)
    assert all('gain bottom half-space' in m for m in messages)


def test_coherent_gaussian_beam_reports_once_across_its_sectors():
    """A beam spread over several Bloch sectors still says it once per side.

    Centred at 42 deg with a 20 um waist, its three samples sit within about 1.4 deg of the
    centre, all inside the band and each in its own sector. Kills a warning per sector."""
    beam = wavebit.GaussianBeam(theta=np.deg2rad(42.0), polarization=(1., 0.), waist=20000.,
                                sampling=(3, 1))
    solver = _coherent(beam, power_region=((-40000., 40000.), (-40000., 40000.)))
    messages, _ = _gain_warnings(solver)
    assert len(messages) == 1, messages

