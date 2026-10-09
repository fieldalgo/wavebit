"""SMM's field state at a complex in-plane wavevector: the cascade's labels are not the state's.

SMM recovers the per-layer state field reconstruction walks from its gap-basis cascade
(:func:`~wavebit._core.scattering._field_state`). That state is the ETM's: one vector per layer,
the layer's ``+`` amplitude at its top, advanced into the next layer. It describes the field only
when ``+`` is the wave the region below receives -- the ETM's label,
:func:`~wavebit._core.conventions.etm_mode_q`. SMM built its layers on
:func:`~wavebit._core.conventions.mode_q`'s decaying root, which is that wave at every real
``k_par`` in a passive layer and is not once ``q**2`` lies above the real axis: at a complex angle,
or in a gain layer, a propagating mode's decaying root travels *up*.

The reported case was a vacuum layer between vacuum half-spaces at ``theta = 0.5+0.3j``. Every
``+`` amplitude is then zero and the whole field sits in the ``-`` amplitudes, which the state has
no slot for: SMM's ``A_i`` came out at 1e15, ``T1`` at zero, and the s-polarized field with
``|E_z|/|E| = 6e-2`` where it is zero. ``R`` and ``T`` were right throughout, because the cascade
does not care which root is called ``+``; 2ff ETM, and 4ff under either algorithm, were right too.

The fix rebuilds a layer on the ETM's labels for the field state where the two rules can differ
(:func:`~wavebit._core.scattering._field_labels_differ`) and leaves the cascade on its own. So:

* the reflectionless uniform scenes below have a closed-form field, a single plane wave in each
  layer, and every path reconstructs it at a complex angle (and in a gain layer at a real one);
* on a grating, SMM's state is the ETM's matrix for matrix, its modes to the bit;
* a layer that may hold a double negative is rebuilt without asking, and has to be;
* the rebuilt state differentiates as the ETM's does, on JAX and torch;
* at a real angle nothing is rebuilt, so the real-angle field is the one it was;
* declaring ``answers='field'`` does not move an amplitude: the cascade keeps its labels, which
  is not a free choice -- handed the ETM's, its ``R`` moved 4.7e-10 at 200 um and
  ``theta = 1.2+0.5j``.

Before the fix every SMM 2ff and pure TE/TM case of the closed form failed, by 0.24 to 1.5e6
relative; ETM and 4ff passed on the same scenes.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core import conventions, scattering

pytestmark = pytest.mark.filterwarnings('ignore::wavebit.ComplexIncidenceWarning')

WL, THICK, PHI = 900.0, 100.0, 0.3
THETA = 0.5 + 0.3j


def _maystre(x):
    """``kz`` of a half-space over ``k0``: ``Re(kz) + Im(kz) > 0`` (``boundary_kz``'s cut)."""
    v = np.sqrt(complex(x))
    return v if v.real + v.imag > 0 else -v


# ------------------------------------------------------------------- the closed-form field

#: (label, theta, layer eps, layer mu, incidence side). The substrate is the layer's own medium
#: and the superstrate vacuum, so the only interface that reflects is the one the light enters
#: through -- none at all on the backside vacuum scene -- and each layer holds one plane wave.
SCENES = [
    ('vacuum, complex angle', THETA, 1.0, 1.0, 'top'),
    ('eps 25 on its own substrate, complex angle', THETA, 25.0, 1.0, 'top'),
    ('gain on its own substrate, real angle', 0.5, 25.0 - 0.5j, 1.0, 'top'),
    ('vacuum lit from below, complex angle', THETA, 1.0, 1.0, 'bottom'),
    ('magnetic on its own substrate, complex angle', THETA, 4.0, 2.25, 'top'),
]
SCENE_IDS = [s[0] for s in SCENES]

#: (kind, formulation, algorithm). ETM and 4ff were right before the fix and are here as the
#: twins the SMM 2ff rows must agree with.
PATHS = [(kind, form, algo) for kind in ('conical', '2d') for form in ('2ff', '4ff')
         for algo in ('ETM', 'SMM')] + [(k, None, a) for k in ('te', 'tm') for a in ('ETM', 'SMM')]
PATH_IDS = [f'{k}-{f}-{a}' if f else f'{k}-{a}' for k, f, a in PATHS]


def _sampled_cell(kind, form, algo, theta, eps, mu, side, z):
    """``{polarization: (Ex, Ey, Ez, Hx, Hy, Hz)}`` at the cell centre and heights ``z``."""
    medium = eps if mu == 1.0 else wavebit.Material(eps, mu=mu)
    kw = dict(backend=0, top=1.0, bot=medium, theta=theta, wavelength=WL, thickness=[THICK],
              layer_algorithm=algo, answers='field', incidence_side=side)
    if kind == '2d':
        # Two rows in y, so a uniform layer still routes to the 2D solve (the one that takes mu).
        kw.update(psi=np.pi / 2, phi=PHI, truncation_order=[0, 0], period=[2000., 2000.])
        grid = (1, 2, 2)
    elif kind == 'conical':
        kw.update(psi=np.pi / 2, phi=PHI, truncation_order=[0], period=[2000.])
        grid = (1, 1, 4)
    else:
        kw.update(tetm=kind, truncation_order=[0], period=[2000.])
        grid = (1, 1, 4)
    solver = wavebit.rcwa(**kw)
    solver.structure = np.full(grid, eps, dtype=complex)
    if mu != 1.0:
        solver.mu = np.full(grid, mu, dtype=complex)
    if form is not None:
        solver.formulation = form
    solver.solve()
    if kind in ('te', 'tm'):
        c = np.asarray(solver.reconstruct_field(sample_x=[0.], sample_y=[0.], sample_z=z))[:, 0, 0]
        zero = np.zeros_like(c[:, 0])
        if kind == 'te':                                      # (Ey, Hx, Hz)
            return {'s': (zero, c[:, 0], zero, c[:, 1], zero, c[:, 2])}
        return {'p': (c[:, 1], zero, c[:, 2], zero, c[:, 0], zero)}   # (Hy, Ex, Ez)
    c = np.asarray(solver.reconstruct_field(sample_x=[0.], sample_y=[0.], sample_z=z,
                                            incidences=('s', 'p')))[:, :, 0, 0]
    return {pol: tuple(c[i, :, j] for j in range(6)) for i, pol in enumerate('sp')}


def _plane_wave_residuals(pol, fields, kx, ky, kz, eps, mu, lead):
    """Each relation a single plane wave obeys, as a residual that is zero when it does.

    ``(kx, ky, kz)`` is the wavevector over ``k0`` in the public frame (``z`` up), ``lead`` the
    transverse amplitude along ``z``: of ``E`` for s, of ``H`` for p (unit incident ``E`` makes
    the incident ``H`` the vacuum index, 1). The transverse amplitude is compared squared,
    ``Ex**2 + Ey**2``, so the comparison holds without fixing the sign of the s/p frame at a
    complex azimuth; its ratio and the longitudinal component pin the rest. ``H = k x E / mu``
    and ``E = -k x H / eps`` are Faraday's and Ampere's laws for a plane wave, with no wavebit in
    them.
    """
    Ex, Ey, Ez, Hx, Hy, Hz = fields
    if pol == 's':
        return {'Ez': Ez, 'k.E': kx * Ex + ky * Ey, 'E_t**2': Ex ** 2 + Ey ** 2 - lead ** 2,
                'Hx': Hx - (ky * Ez - kz * Ey) / mu, 'Hy': Hy - (kz * Ex - kx * Ez) / mu,
                'Hz': Hz - (kx * Ey - ky * Ex) / mu}
    return {'Hz': Hz, 'k.H': kx * Hx + ky * Hy, 'H_t**2': Hx ** 2 + Hy ** 2 - lead ** 2,
            'Ex': Ex + (ky * Hz - kz * Hy) / eps, 'Ey': Ey + (kz * Hx - kx * Hz) / eps,
            'Ez': Ez + (kx * Hy - ky * Hx) / eps}


@pytest.mark.parametrize('kind,form,algo', PATHS, ids=PATH_IDS)
@pytest.mark.parametrize('name,theta,eps,mu,side', SCENES, ids=SCENE_IDS)
def test_a_reflectionless_uniform_scene_reconstructs_its_plane_wave(
        name, theta, eps, mu, side, kind, form, algo):
    """Inside the layer the field is one plane wave, absolutely: amplitude, phase and polarization.

    Lit from above, it is the wave transmitted at the top interface,
    ``t exp(i k0 kz (h - z))``, with ``t_s = 2 kz0 / (kz0 + kz/mu)`` on the electric field and
    ``t_p = 2 kz0 / (kz0 + kz/eps)`` on the magnetic one. Lit from below through vacuum, it is the
    incident wave itself, ``exp(i k0 kz z)``.
    """
    if mu != 1.0 and (kind != '2d' or form == '4ff'):
        pytest.skip('a raster mu is the 2D 2ff path only; the others refuse it by name')
    kpar = np.sin(theta)                                      # both incidence media are vacuum
    kx, ky = (kpar * np.cos(PHI), kpar * np.sin(PHI)) if kind in ('conical', '2d') else (kpar, 0.0)
    kz0, kz = _maystre(1.0 - kpar ** 2), _maystre(eps * mu - kpar ** 2)
    k0 = 2 * np.pi / WL
    z = np.linspace(5.0, THICK - 5.0, 5)
    if side == 'top':
        travel, kz_up = np.exp(1j * k0 * kz * (THICK - z)), -kz
        t = {'s': 2 * kz0 / (kz0 + kz / mu), 'p': 2 * kz0 / (kz0 + kz / eps)}
    else:
        travel, kz_up = np.exp(1j * k0 * kz * z), kz
        t = {'s': 1.0, 'p': 1.0}

    cells = _sampled_cell(kind, form, algo, theta, eps, mu, side, z)
    for pol, fields in cells.items():
        lead = t[pol] * travel
        scale = float(np.max(np.abs(lead)))
        for what, r in _plane_wave_residuals(pol, fields, kx, ky, kz_up, eps, mu, lead).items():
            worst = float(np.max(np.abs(r))) / (scale ** 2 if what.endswith('**2') else scale)
            assert worst < 1e-12, (pol, what, worst)


# ------------------------------------------------------------ a grating: the ETM's state

#: The scene ``docs/connecting-algorithms.md`` section 10 found ETM's own label floor on: wide
#: ``eps = 25`` blocks resting on an ``eps = 25`` substrate, so modes confined to them cross into
#: it unreflected. Two layers, so the walk crosses a layer-to-layer interface as well as the two
#: half-space ones. SMM's field was 13x off the ETM's here at one layer.
GRATING = np.array([[[0, 1, 0, 1, 1, 1]], [[1, 0, 0, 1, 1, 1]]]) * 24 + 1.
RASTER = np.ones((2, 6, 6)) * 25.0
RASTER[:, 1:3, 2:5] = 1.0
RASTER[1, 3:5, 0:2] = 1.0

GRATING_CASES = {
    'conical': dict(psi=np.pi / 2, phi=PHI, structure=GRATING, truncation_order=[15],
                    period=[2000.]),
    'te': dict(tetm='te', structure=GRATING, truncation_order=[15], period=[2000.]),
    'tm': dict(tetm='tm', structure=GRATING, truncation_order=[15], period=[2000.]),
    '2d': dict(psi=np.pi / 2, phi=PHI, structure=RASTER, truncation_order=[3, 3],
               period=[1200., 1200.]),
}


def _grating(algo, case, theta=THETA, answers='field'):
    solver = wavebit.rcwa(backend=0, top=1.0, bot=25.0, theta=theta, wavelength=WL,
                          thickness=[100., 60.], layer_algorithm=algo, answers=answers,
                          **GRATING_CASES[case])
    return solver, solver.solve()


@pytest.mark.parametrize('case', list(GRATING_CASES))
def test_a_grating_field_state_is_the_etms_at_a_complex_angle(case):
    """Layer by layer, matrix for matrix: ``test_scattering_core``'s property, at a complex angle.

    ``W``, ``V`` and ``q`` are compared exactly: the rebuilt layer is the same builder on the same
    labels as the ETM's. ``T1`` and the cascade-recovered blocks are compared to a tolerance;
    the bottom layer's two blocks only through their product, as there (``_field_state``).
    Before the fix ``q`` differed by sign and the field by up to 1.5e5 relative.
    """
    etm, _ = _grating('ETM', case)
    smm, _ = _grating('SMM', case)

    a, b = np.asarray(etm.T1), np.asarray(smm.T1)
    assert np.max(np.abs(a - b)) < 1e-10 * np.max(np.abs(a))
    for i, (le, ls) in enumerate(zip(etm.layer_info_list, smm.layer_info_list)):
        for j, what in ((1, 'W'), (2, 'V'), (3, 'q')):
            assert np.array_equal(np.asarray(le[j]), np.asarray(ls[j])), (i, what)
        pairs = ([(le[6] @ le[5], ls[6] @ ls[5], 'B@A_i')] if i == 0
                 else [(le[5], ls[5], 'A_i'), (le[6], ls[6], 'B')])
        for x, y, what in pairs:
            scale = max(float(np.max(np.abs(x))), 1.0)
            assert np.max(np.abs(x - y)) < 1e-10 * scale, (i, what)

    incidences = ('incident',) if case in ('te', 'tm') else ('incident', 's', 'p')
    samples = dict(sample_x=6, sample_y=6, sample_z=4, incidences=incidences)
    ce = np.asarray(etm.reconstruct_field(**samples))
    cs = np.asarray(smm.reconstruct_field(**samples))
    assert np.max(np.abs(ce - cs)) < 1e-10 * np.max(np.abs(ce))


def _double_negative_cell(algo, theta, form='2ff'):
    eps, mu = -4.0, -1.0
    solver = wavebit.rcwa(backend=0, top=1.0, bot=wavebit.Material(eps, mu=mu), theta=theta,
                          wavelength=WL, thickness=[THICK], layer_algorithm=algo,
                          answers='field', psi=np.pi / 2, phi=PHI, truncation_order=[0, 0],
                          period=[2000., 2000.])
    solver.structure = np.full((1, 2, 2), eps, dtype=complex)
    solver.mu = np.full((1, 2, 2), mu, dtype=complex)
    solver.formulation = form
    solver.solve()
    return np.asarray(solver.reconstruct_field(sample_x=3, sample_y=3, sample_z=4,
                                               incidences=('s', 'p')))


def test_a_double_negative_is_rebuilt_where_the_plain_check_would_not_see_it():
    """A uniform double negative on its own medium, ``theta = 0.5-0.3j``: SMM matches the ETM.

    The ETM labels such a layer by the mirror of Maystre's cut, ``Re(q) + Im(q) > 0``, and the
    check that decides a rebuild restates only the plain rule, ``Re(q) > Im(q)``. Here ``q**2``
    lies below the real axis, the decaying root has ``Im(q) < -Re(q)``, and the plain rule calls
    the labels equal while the mirror flips them -- so a layer that may hold a double negative
    is rebuilt without asking. Without that, SMM came back 2x off the ETM, as it did before the
    fix. The 4ff path, which labels by the flux and shares neither rule, vouches for the ETM.
    """
    theta = 0.5 - 0.3j
    etm = _double_negative_cell('ETM', theta)
    scale = np.max(np.abs(etm))
    assert np.max(np.abs(_double_negative_cell('ETM', theta, '4ff') - etm)) < 1e-12 * scale
    assert np.max(np.abs(_double_negative_cell('SMM', theta) - etm)) < 1e-12 * scale


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('case', ['te', 'conical'])
def test_the_rebuilt_state_differentiates_as_the_etms_does(case, backend):
    """``d sum|E|**2 / ds`` through the rebuilt state, ``s`` scaling the grating's permittivity.

    Why the state is rebuilt by the mode builder rather than relabelled afterwards: under
    autodiff a layer's ``q`` is a modal generator whose off-diagonal entries divide by
    ``q_i + q_j``, and flipping some roots and not others changes those denominators. Measured
    on TE: negating the flipped roots and their ``V`` columns after the fact reconstructed every
    closed-form field above to 1e-15, and gave this derivative 15x off on JAX and 20% off on
    torch. Built afresh on the ETM's labels the state is the ETM's, derivative included.
    """
    pytest.importorskip(backend)
    wavebit.enable_x64()
    base = GRATING_CASES[case]['structure']
    samples = dict(sample_x=4, sample_y=3, sample_z=3)

    def gradient(algo):
        kw = {k: v for k, v in GRATING_CASES[case].items() if k != 'structure'}
        if backend == 'jax':
            import jax
            import jax.numpy as jnp

            def loss(s):
                solver = wavebit.rcwa(backend=1, top=1.0, bot=25.0, theta=THETA, wavelength=WL,
                                      thickness=[100., 60.], layer_algorithm=algo,
                                      answers='field', structure=jnp.asarray(base) * (1 + s), **kw)
                return jnp.sum(jnp.abs(solver.solve_field_eager(**samples)[1]) ** 2)
            return float(jax.grad(loss)(0.0))
        import torch
        s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        solver = wavebit.rcwa(backend=2, top=1.0, bot=25.0, theta=THETA, wavelength=WL,
                              thickness=[100., 60.], layer_algorithm=algo, answers='field',
                              structure=torch.as_tensor(base) * (1 + s), **kw)
        torch.sum(torch.abs(solver.solve_field(**samples)[1]) ** 2).backward()
        return float(s.grad)

    etm, smm = gradient('ETM'), gradient('SMM')
    assert abs(smm - etm) < 1e-9 * abs(etm), (smm, etm)


# ----------------------------------------------------- what the fix leaves where it was

def _count_relabelled_builds(monkeypatch):
    """Count the layer-mode builds SMM asks for on the ETM's labels (``k0_d`` given)."""
    calls = []
    for name in ('layer_modes_1d', 'layer_modes_1d_conical', 'layer_modes_2d'):
        build = getattr(scattering, name)

        def counted(*args, _build=build, **kwargs):
            if kwargs.get('k0_d') is not None:
                calls.append(kwargs['k0_d'])
            return _build(*args, **kwargs)
        monkeypatch.setattr(scattering, name, counted)
    return calls


@pytest.mark.parametrize('case', list(GRATING_CASES))
def test_a_real_angle_field_state_is_the_cascades_own(case, monkeypatch):
    """At a real angle no layer is rebuilt, so the real-angle field is the one it always was.

    There the two label rules pick the same root for every mode of a passive layer, to the bit
    (``etm_mode_q``'s docstring), and the check that decides a rebuild sees it: the state is the
    cascade's own modes, unchanged. The same scene at a complex angle has to rebuild every
    layer, or this counter could not see a rebuild at all. Measured byte for byte against the
    code before the fix: 134 arrays over TE, TM, conical and 2D, both incidence sides, a
    magnetic raster and a uniform double negative -- every field cell, ``T1``, layer record and
    amplitude identical.
    """
    calls = _count_relabelled_builds(monkeypatch)
    _grating('SMM', case, theta=0.5)
    assert calls == []

    _grating('SMM', case, theta=THETA)
    assert len(calls) == 2, calls


@pytest.mark.parametrize('case', list(GRATING_CASES))
def test_declaring_the_field_does_not_move_an_amplitude_at_a_complex_angle(case, monkeypatch):
    """The cascade keeps its own labels when the field state is rebuilt, so the rungs agree exactly.

    Not a free choice. A cascade handed the ETM's labels answers differently in the last bits at
    100 nm, and at 200 um and ``theta = 1.2+0.5j`` its ``R`` moved 4.7e-10 (2.8e-9 under a 1e-14
    nudge of ``theta``) against its own labels' 1.9e-12 from the ETM. The control below hands
    the cascade the ETM's labels and has to move the bytes, or this comparison could not see
    the difference.
    """
    _, amplitude = _grating('SMM', case, answers='amplitude')
    _, field = _grating('SMM', case, answers='field')
    fields = ('r_s', 'r_p', 't_s', 't_p')
    for f in fields:
        assert np.array_equal(np.asarray(getattr(amplitude.incident, f)),
                              np.asarray(getattr(field.incident, f))), f

    k0_d = 2 * np.pi / WL * 100.0
    monkeypatch.setattr(conventions, 'mode_q',
                        lambda xp, eigenvalue, mirrored=None:
                        conventions.etm_mode_q(xp, eigenvalue, k0_d, mirrored))
    _, relabelled = _grating('SMM', case, answers='amplitude')
    assert not all(np.array_equal(np.asarray(getattr(amplitude.incident, f)),
                                  np.asarray(getattr(relabelled.incident, f))) for f in fields)
