"""SMM's field state is read off the gap amplitude between layers, so it reads no labels.

SMM recovers the per-layer state field reconstruction walks from its gap-basis cascade
(:func:`~wavebit.rcwa._engine.scattering._field_state`). Until 2026-10-10 that state was the ETM's: one
vector per layer, the layer's ``+`` amplitude at its top, advanced through the inverse of ``M``,
the map from the gap below a layer onto those amplitudes. Inverting ``M`` presumes the layer's
``+`` modes are the waves the region below receives -- the ETM's labels -- and the cascade is
built on :func:`~wavebit._common.conventions.mode_q`'s decaying ones. At a complex angle (or in a
gain layer, at single precision, or on a truncated metal operator's spurious root) a propagating
mode's decaying root travels up, and a vacuum layer between vacuum half-spaces at
``theta = 0.5+0.3j`` had ``M`` exactly singular: ``|E_z|/|E| = 6e-2`` where it is zero, with
``R`` and ``T`` exact.

The first repair (429f4ac) rebuilt each such layer on the ETM's labels. It fixed that scene and
not the inversion: it ran a second ``eig`` on an identical operator, stayed O(1) wrong where a
root grows past the ETM's cap and meets an interface that reflects nothing, lost digits where a
root grows below it (``M``'s two terms cancel), and made real-angle metal rasters 1e3-1e6x worse.
The state is now each layer's own ``+`` and ``-`` amplitudes, read off the gap amplitude ``g``
at its faces by forward maps; nothing that maps into a layer is inverted. A mode with
``|k0 q d| < 1`` has both read off the top face, because there they cancel in the field. So:

* the reflectionless uniform scenes below have a closed-form field, one plane wave in the layer,
  and every path reconstructs it -- thin layers, a gain layer at a real angle, a magnetic one,
  and layers thick enough that the ETM's cap turns a root;
* on gratings, at a complex angle and thick, SMM's field is the ETM's;
* the scenes the relabel got wrong or made worse are pinned: a capped layer on its own medium,
  a root growing below another layer, metal rasters and an active double negative at a real
  angle, single precision at a real angle;
* an order on a layer's light line costs no digits, and the field is continuous at every
  interface and meets the half-space fields its ``R`` and ``T`` imply -- an oracle with no label;
* each layer's modes are built once, and declaring ``answers='field'`` moves no amplitude;
* the state differentiates as the ETM's does, and past the ETM's cap as a finite difference does.
"""

from wavebit._common.backend import AutoDetuneWarning as _legacy_wavebit_AutoDetuneWarning

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import warnings

import numpy as np
import pytest

import wavebit
from wavebit.rcwa._engine import engine
from wavebit.rcwa._engine import scattering

pytestmark = pytest.mark.filterwarnings('ignore::wavebit._common.backend.ComplexIncidenceWarning')

WL, PHI = 900.0, 0.3
THETA = 0.5 + 0.3j
#: The capped layer on its own medium that the relabel left 0.75 off (and 0.92 under a 1e-14
#: nudge of ``theta``): eps = 2.25 throughout, lit from below.
CHIP_THETA = 0.611565803730562 + 0.34349515827119764j


def _maystre(x):
    """``kz`` of a half-space over ``k0``: ``Re(kz) + Im(kz) > 0`` (``boundary_kz``'s cut)."""
    v = np.sqrt(complex(x))
    return v if v.real + v.imag > 0 else -v


def _relative(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b))))


@pytest.fixture(params=['merged', 'unmerged'])
def merge(request, monkeypatch):
    """A layer of a half-space's own medium, both ways: merged into it, which a solve does at a
    complex angle (``tests/rcwa/test_half_space_merge.py``), and solved as a layer -- the
    kernels' own field state, which a jitted solve or an ``excitation`` still takes.
    """
    if request.param == 'unmerged':
        monkeypatch.setattr(engine.CoreFactorizationMixin, '_half_space_merge_plan',
                            lambda self: None)
    return request.param


# ------------------------------------------------------------------- the closed-form field

#: (label, theta, layer eps, layer mu, incidence side, incidence-medium eps, thickness nm, bound).
#: Lit from above, the superstrate is the incidence medium and the substrate is the layer's own,
#: so only the top interface reflects; lit from below, every medium is the layer's, so nothing
#: does. Either way the layer holds one plane wave. The bound is relative to the wave where it is
#: sampled, not to the largest one: past the cap the two differ by the growth ``G``, and an error
#: near the lit face measured against the far one would shrink by it (a spurious O(1) wave at the
#: entrance passed a bound so scaled). The thick rows grow past the ETM's cap -- by ``2.7e4``,
#: ``6.3e5`` and ``2.8e4`` -- and there every path sits near ``eps * G**2`` at the lit face, 1.6e-7,
#: 8.9e-5 and 1.8e-7: over seven 1e-13 nudges of ``theta`` the worst path reached 3.2e-7, 1.8e-4
#: and 3.5e-7 (first-order), SMM's pure TE/TM paths 7.1e-8, 4.4e-5 and 1.3e-7. The ``M`` walk
#: left SMM 0.2 to 1.0 off on these rows and the relabel 0.15 to 2.4. The thin rows are at 1.3e-14
#: at most.
SCENES = [
    ('vacuum, complex angle', THETA, 1.0, 1.0, 'top', 1.0, 100.0, 1e-12),
    ('eps 4 on its own substrate, real angle', 0.5, 4.0, 1.0, 'top', 1.0, 100.0, 1e-12),
    ('eps 25 on its own substrate, complex angle', THETA, 25.0, 1.0, 'top', 1.0, 100.0, 1e-12),
    ('gain on its own substrate, real angle', 0.5, 25.0 - 0.5j, 1.0, 'top', 1.0, 100.0, 1e-12),
    ('vacuum lit from below, complex angle', THETA, 1.0, 1.0, 'bottom', 1.0, 100.0, 1e-12),
    ('magnetic on its own substrate, complex angle', THETA, 4.0, 2.25, 'top', 1.0, 100.0, 1e-12),
    ('10 um of vacuum on vacuum, past the cap', THETA, 1.0, 1.0, 'top', 1.0, 1e4, 5e-6),
    ('20 um of eps 2.13 on itself, past the cap', THETA, 2.13, 1.0, 'top', 1.0, 2e4, 2e-3),
    ('4869 nm of eps 2.25 in eps 2.25, lit from below', CHIP_THETA, 2.25, 1.0, 'bottom', 2.25,
     4869.0, 5e-6),
]
SCENE_IDS = [s[0] for s in SCENES]

#: (kind, formulation, algorithm). ETM and the first-order path were right throughout and are
#: here as the twins the SMM second-order rows must agree with.
PATHS = [(kind, form, algo) for kind in ('conical', '2d')
         for form in ('second_order', 'first_order')
         for algo in ('ETM', 'SMM')] + [(k, None, a) for k in ('te', 'tm') for a in ('ETM', 'SMM')]
PATH_IDS = [f'{k}-{f}-{a}' if f else f'{k}-{a}' for k, f, a in PATHS]


def _medium(eps, mu):
    return eps if mu == 1.0 else wavebit.Material(eps, mu=mu)


def _sampled_cell(kind, form, algo, theta, eps, mu, side, eps_inc, thick, z):
    """``{polarization: (Ex, Ey, Ez, Hx, Hy, Hz)}`` at the cell centre and heights ``z``."""
    layer = _medium(eps, mu)
    top, bot = (eps_inc, layer) if side == 'top' else (layer, layer)
    kw = dict(backend='numpy', top=top, bot=bot, theta=theta, wavelength=WL, thickness=[thick],
              layer_algorithm=algo, answers='field', incidence_side=side)
    if kind == '2d':
        kw.update(psi=np.pi / 2, phi=PHI, truncation_order=[0, 0], period=[2000., 2000.])
        grid = (1, 2, 2)
    elif kind == 'conical':
        kw.update(psi=np.pi / 2, phi=PHI, truncation_order=[0], period=[2000.])
        grid = (1, 1, 4)
    else:
        kw.update(tetm=kind, truncation_order=[0], period=[2000.])
        grid = (1, 1, 4)
    solver = _legacy_wavebit_rcwa(**kw)
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
    the incident ``H`` the incidence medium's index). The transverse amplitude is compared
    squared, ``Ex**2 + Ey**2``, so the comparison holds without fixing the sign of the s/p frame
    at a complex azimuth; its ratio and the longitudinal component pin the rest. ``H = k x E / mu``
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
@pytest.mark.parametrize('name,theta,eps,mu,side,eps_inc,thick,bound', SCENES, ids=SCENE_IDS)
def test_a_reflectionless_uniform_scene_reconstructs_its_plane_wave(
        name, theta, eps, mu, side, eps_inc, thick, bound, kind, form, algo, merge):
    """Inside the layer the field is one plane wave, absolutely: amplitude, phase and polarization.

    Lit from above, it is the wave transmitted at the top interface, ``t exp(i k0 kz (h - z))``,
    with ``t_s = 2 kz_i / (kz_i + kz/mu)`` on the electric field and
    ``t_p = n_i 2 (kz_i/eps_i) / (kz_i/eps_i + kz/eps)`` on the magnetic one. Lit from below
    through the layer's own medium, it is the incident wave itself, ``exp(i k0 kz z)``.

    At a complex angle or with gain the wave sits on the cascade's ``-`` label; the passive
    real-angle row is the one whose wave is the layer's ``+`` mode, so the closed form reads both.
    Before the gap walk every SMM second-order and pure TE/TM case failed: by 0.24 to 1.5e6 on
    the thin rows before 429f4ac, and on the thick ones under the relabel (07ae0f6) by 0.15 to
    2.4 of the largest field, or with ``LinAlgError``.
    """
    n_inc = np.sqrt(complex(eps_inc))
    kpar = n_inc * np.sin(theta)
    kx, ky = (kpar * np.cos(PHI), kpar * np.sin(PHI)) if kind in ('conical', '2d') else (kpar, 0.0)
    kz_i, kz = _maystre(eps_inc - kpar ** 2), _maystre(eps * mu - kpar ** 2)
    k0 = 2 * np.pi / WL
    z = np.linspace(0.05, 0.95, 5) * thick
    if side == 'top':
        travel, kz_up = np.exp(1j * k0 * kz * (thick - z)), -kz
        t = {'s': 2 * kz_i / (kz_i + kz / mu),
             'p': n_inc * 2 * (kz_i / eps_inc) / (kz_i / eps_inc + kz / eps)}
    else:
        travel, kz_up = np.exp(1j * k0 * kz * z), kz
        t = {'s': 1.0, 'p': n_inc}

    cells = _sampled_cell(kind, form, algo, theta, eps, mu, side, eps_inc, thick, z)
    for pol, fields in cells.items():
        lead = t[pol] * travel
        local = np.abs(lead)
        for what, r in _plane_wave_residuals(pol, fields, kx, ky, kz_up, eps, mu, lead).items():
            worst = float(np.max(np.abs(r) / (local ** 2 if what.endswith('**2') else local)))
            assert worst < bound, (pol, what, worst)


# ------------------------------------------------------------ gratings: SMM's field is the ETM's

#: Wide ``eps = 25`` blocks on an ``eps = 25`` substrate, so modes confined to them cross into it
#: unreflected (``docs/theory/rcwa/connecting-algorithms.md`` §10). Two layers, so the walk crosses a
#: layer-to-layer plane as well as the two half-space ones.
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


def _grating(algo, case, theta=THETA, answers='field', thickness=(100., 60.), **extra):
    solver = _legacy_wavebit_rcwa(backend='numpy', top=1.0, bot=25.0, theta=theta, wavelength=WL,
                          thickness=list(thickness), layer_algorithm=algo, answers=answers,
                          **GRATING_CASES[case], **extra)
    return solver, solver.solve()


def _field(solver, case):
    incidences = ('incident',) if case in ('te', 'tm') else ('incident', 's', 'p')
    return np.asarray(solver.reconstruct_field(sample_x=6, sample_y=6, sample_z=4,
                                               incidences=incidences))


@pytest.mark.parametrize('thickness', [(100., 60.), (3000., 1800.)], ids=['thin', 'thick'])
@pytest.mark.parametrize('case', list(GRATING_CASES))
def test_a_grating_field_is_the_etms_at_a_complex_angle(case, thickness):
    """SMM's field on the grating at ``theta = 0.5+0.3j``: the ETM's, thin and 4.8 um thick.

    Worst of the four paths, ``studies/rcwa/smm_field_state/smm_field_state.py`` section 2: before 429f4ac 1.5e5 (thin)
    and 18 (thick); under the relabel 1.3e-13 and 4.1e-8, since it still inverted ``M``, whose two
    terms cancel where a root grows; now 8.0e-14 and 4.9e-11.
    """
    etm, _ = _grating('ETM', case, thickness=thickness)
    smm, _ = _grating('SMM', case, thickness=thickness)
    assert _relative(_field(smm, case), _field(etm, case)) < 5e-10


def test_a_root_growing_below_another_layer_costs_no_digits():
    """A thick vacuum layer below a thin one, at a complex angle: SMM's field is the ETM's.

    The vacuum's propagating root grows by ``e**8.66`` downward across 8.5 um. On the ETM's labels
    the relabelled state's ``M`` at the plane above it cancelled to about ``1/G**2``: its TE field
    was 7.3e-11 off a 60-digit closed form where the ETM's is 3.8e-15. The walk takes no label and
    is at 5.2e-16 (``studies/rcwa/smm_field_state/smm_field_state.py`` section 3).
    """
    def field(algo, kind):
        kw = dict(tetm=kind) if kind in ('te', 'tm') else dict(psi=np.pi / 2, phi=PHI)
        solver = _legacy_wavebit_rcwa(backend='numpy', top=1.0, bot=2.25, theta=THETA, wavelength=WL,
                              thickness=[8500., 500.], truncation_order=[0], period=[2000.],
                              structure=np.array([[[1.] * 4], [[4.] * 4]], dtype=complex),
                              layer_algorithm=algo, answers='field', **kw)
        solver.solve()
        incidences = {} if kind in ('te', 'tm') else dict(incidences=('s', 'p'))
        return np.asarray(solver.reconstruct_field(sample_x=[0.], sample_y=[0.], sample_z=9,
                                                   **incidences))
    for kind in ('te', 'tm', 'conical'):
        assert _relative(field('SMM', kind), field('ETM', kind)) < 1e-12, kind


def _chip(algo, form='second_order', nudge=1.0):
    solver = _legacy_wavebit_rcwa(backend='numpy', top=2.25, bot=2.25, theta=CHIP_THETA * nudge, phi=0.41,
                          psi=0.27, wavelength=1000., period=[1654.489947568981],
                          truncation_order=[7], thickness=[4869.],
                          structure=np.full((1, 1, 8), 2.25 + 0j), layer_algorithm=algo,
                          answers='field', incidence_side='bottom')
    solver.formulation = form
    solver.solve(auto_detune=False)
    return np.asarray(solver.reconstruct_field(sample_x=5, sample_y=1, sample_z=6,
                                               incidences=('s', 'p')))


def test_a_capped_layer_on_its_own_medium_is_the_etms_and_steady(merge):
    """4869 nm of ``eps = 2.25`` inside ``eps = 2.25``, lit from below at a complex angle.

    The ETM's cap turns a growing root back to the decaying one there, and the interface below
    reflects nothing: on the ETM's labels ``M`` was a rounding residue, and the relabelled SMM
    field came out 0.75 off and moved 0.92 under a 1e-14 nudge of ``theta``. Now it is the ETM's
    to 1.2e-12 and the nudge moves it 1.3e-12. The first-order path, at its own floor here
    (3e-6), vouches for both.
    """
    smm = _chip('SMM')
    assert _relative(smm, _chip('ETM')) < 1e-10
    assert _relative(_chip('SMM', nudge=1 + 1e-14), smm) < 1e-10
    assert _relative(smm, _chip('ETM', 'first_order')) < 1e-4


#: A grating over an ``eps = 2.25`` spacer, 900 nm on a 600 nm period at normal incidence: orders
#: +-1 sit exactly on the spacer's light line, ``q = 0``, and ``auto_detune`` lands ``1e-12`` off it.
GRAZE_GRATING = np.ones((2, 1, 8), complex)
GRAZE_GRATING[0, 0, 2:5] = 6.0
GRAZE_GRATING[1] = 2.25
GRAZE_RASTER = np.ones((2, 6, 6), complex)
GRAZE_RASTER[0, 1:4, 1:4] = 6.0
GRAZE_RASTER[1] = 2.25


@pytest.mark.parametrize('kind', ['te', 'tm', 'conical', '2d'])
def test_a_layer_light_line_costs_no_digits(kind):
    """An order on a layer's light line, ``auto_detune``d off it: SMM's field is the ETM's.

    There the mode's two amplitudes are each ``O(1/q)`` and cancel to an ``O(1)`` field, so where
    they are read matters. Read off opposite faces, as the first gap walk read them, their rounding
    did not cancel with them: 8e-7 (TE) to 7e-6 (conical) off the ETM, which is 3e-11 from an
    analytic continuation across the graze. Reading such a mode's ``-`` amplitude off the top face
    too (:func:`~wavebit.rcwa._engine.modal_gradient.two_face`) puts SMM at 2e-11 to 6e-11 from the ETM.
    """
    kw = dict(tetm=kind) if kind in ('te', 'tm') else dict(psi=0.7, phi=0.0)
    two = kind == '2d'

    def field(algo):
        solver = _legacy_wavebit_rcwa(backend='numpy', top=1.0, bot=4.0, theta=0.0, wavelength=WL,
                              thickness=[150., 400.], period=[600.] * (2 if two else 1),
                              structure=GRAZE_RASTER if two else GRAZE_GRATING,
                              truncation_order=[2, 2] if two else [4], layer_algorithm=algo,
                              answers='field', **kw)
        with pytest.warns(_legacy_wavebit_AutoDetuneWarning):
            solver.solve()
        incidences = ('incident',) if kind in ('te', 'tm') else ('incident', 's', 'p')
        return np.asarray(solver.reconstruct_field(sample_x=5, sample_y=4 if two else 1,
                                                   sample_z=[30., 200., 400., 520.],
                                                   incidences=incidences))
    assert _relative(field('SMM'), field('ETM')) < 1e-9


# --------------------------------------------- the field meets itself and its ports, label-free

def _tangential(cell):
    return cell[..., [0, 1, 3, 4]]


PORT_SCENES = {
    'three layers, complex angle': dict(theta=THETA, thickness=[200., 300., 150.],
                                        layers=[(4., 1.), (2.25, 6.), (9., 1.)]),
    '10 um of vacuum between gratings, complex angle': dict(
        theta=THETA, thickness=[200., 1e4, 150.], layers=[(4., 1.), (1., 1.), (9., 1.)]),
    'metal grating, real angle': dict(theta=0.3, thickness=[100., 300., 100.],
                                      layers=[(4., 1.), (-20 + 1j, 1.), (2.25, 1.)]),
}


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('kind', ['te', 'tm', 'conical', '2d'])
@pytest.mark.parametrize('scene', list(PORT_SCENES))
def test_the_field_meets_itself_and_its_ports(scene, kind, side):
    """Tangential ``E`` and ``H`` are continuous at every interface, the two outer ones included.

    An oracle that names no label: inside a layer the modes satisfy Maxwell by construction, so a
    field continuous at every interior interface and equal at the outer faces to the half-space
    fields ``R`` and ``T`` imply (``boundary_fields``) is the truncated problem's unique solution.
    Every layer's ``+`` and ``-`` amplitudes enter it. SMM sits at 1e-15 to 7e-15 of the field;
    reading every mode that keeps a hundredth of itself across the layer off the top face (rather
    than only those with ``|k0 q d| < 1``) put it at up to 2e-13, the top face's rounding carried
    down by up to ``exp(4.6)``.
    """
    sc = PORT_SCENES[scene]
    two = kind == '2d'
    kw = dict(psi=0.7, phi=0.3) if kind in ('conical', '2d') else dict(tetm=kind)
    grid = (len(sc['thickness']), 4, 6) if two else (len(sc['thickness']), 1, 14)
    structure = np.ones(grid, dtype=complex)
    for i, (background, block) in enumerate(sc['layers']):
        structure[i] = background
        structure[i, ..., :grid[-1] // 2] = block
    solver = _legacy_wavebit_rcwa(backend='numpy', top=1.0, bot=2.25, theta=sc['theta'], wavelength=WL,
                          thickness=sc['thickness'], layer_algorithm='SMM', answers='field',
                          incidence_side=side, structure=structure,
                          truncation_order=[2, 2] if two else [4],
                          period=[700., 600.] if two else [700.], **kw)
    result = solver.solve(auto_detune=False)
    xs, ys = list(np.linspace(0., 690., 23)), [0., 211.]
    total = float(sum(sc['thickness']))
    faces = np.asarray(result.reconstruct_field(sample_x=xs, sample_y=ys, sample_z=[0., total]))
    ports = [np.asarray(result.boundary_fields(face, sample_x=xs, sample_y=ys, part='total'))
             for face in ('bottom', 'top')]
    scale = max(float(np.max(np.abs(_tangential(c)))) for c in [faces] + ports)
    for i, port in enumerate(ports):
        gap = float(np.max(np.abs(_tangential(faces[i]) - _tangential(port)))) / scale
        assert gap < 5e-14, (('bottom', 'top')[i], gap)
    # Two samples per layer land on its faces; neighbours meet at each interior interface.
    counts = np.asarray(result.reconstruct_field(sample_x=xs, sample_y=ys, sample_z=2))
    for k in range(len(sc['thickness']) - 1):
        below, above = _tangential(counts[2 * k + 1]), _tangential(counts[2 * k + 2])
        local = max(float(np.max(np.abs(below))), float(np.max(np.abs(above))))
        assert float(np.max(np.abs(below - above))) / local < 5e-14, k


# ----------------------------------------------- real angles: what the relabel made worse

def _raster(eps_hi, eps_lo=1.0, n=16):
    """A disc over a bar, ``n`` pixels a side, in two layers."""
    yy, xx = np.mgrid[0:n, 0:n] + 0.5
    grid = np.full((2, n, n), eps_lo, dtype=complex)
    grid[0][np.hypot(xx - n / 2, yy - n / 2) < n * 0.35] = eps_hi
    grid[1][(np.abs(xx - n / 2) < n * 0.3) & (np.abs(yy - n / 2) < n * 0.15)] = eps_hi
    return grid


def _metal_field(algo, form, theta, thickness, eps_hi=-15 + 0.8j, factorization='li', to=5):
    solver = _legacy_wavebit_rcwa(backend='numpy', top=1.0, bot=2.25, theta=theta, wavelength=600.,
                          psi=np.pi / 3, phi=0.4, structure=_raster(eps_hi),
                          truncation_order=[to, to], period=[400., 400.], thickness=thickness,
                          layer_algorithm=algo, answers='field', factorization=factorization)
    solver.formulation = form
    solver.solve()
    return np.asarray(solver.reconstruct_field(sample_x=8, sample_y=8, sample_z=8,
                                               incidences=('s', 'p')))


@pytest.mark.parametrize('theta,thickness', [(0.35, [120., 99.]), (0.0, [120., 95.])],
                         ids=['oblique', 'normal'])
def test_a_metal_raster_at_a_real_angle(theta, thickness):
    """A lossy metal raster at a real angle, normal incidence included: SMM's field is the ETM's.

    A truncated metal operator can hold a root with ``Im(q**2) > 0`` at a real ``k_par``, so the
    two label rules part there too. The relabel rebuilt such layers on growing roots and left
    SMM's field 3.2e-8 (oblique) and 1.2e-8 (normal) off where it had been at 1e-12 (3.5e-6 and
    2.8e-6 on the gap before 9a92bf0). The first-order path, which shares neither second-order
    algorithm's modes, agrees with both.
    """
    etm = _metal_field('ETM', 'second_order', theta, thickness)
    smm = _metal_field('SMM', 'second_order', theta, thickness)
    assert _relative(smm, etm) < 1e-10
    assert _relative(smm, _metal_field('ETM', 'first_order', theta, thickness)) < 1e-10


@pytest.mark.parametrize('kind', ['tm', 'conical'])
@pytest.mark.parametrize('theta', [0.0, 0.5])
def test_single_precision_at_a_real_angle(kind, theta):
    """``complex_dtype=1`` at a real angle: SMM's field is as good as the ETM's at that width.

    At single precision a propagating mode's ``Re(q)`` is round-off, so ``mode_q`` labels some of
    them upward by chance. The ``M`` walk made that an O(1) error (1.4 before 429f4ac, 49 on a
    one-layer version of this grating); the walk over ``g`` never asks.
    """
    kw = dict(tetm='tm') if kind == 'tm' else dict(psi=np.pi / 2, phi=PHI)

    def field(algo, width):
        solver = _legacy_wavebit_rcwa(backend='numpy', top=1.0, bot=25.0, theta=theta, wavelength=WL,
                              thickness=[100., 60.], structure=GRATING, truncation_order=[15],
                              period=[2000.], layer_algorithm=algo, answers='field',
                              complex_dtype=width, **kw)
        solver.solve()
        return np.asarray(solver.reconstruct_field(sample_x=6, sample_y=1, sample_z=4))
    reference = field('ETM', 0)
    etm32 = _relative(field('ETM', 1), reference)
    assert _relative(field('SMM', 1), reference) < max(10 * etm32, 1e-5)


def test_an_active_patterned_double_negative_at_a_real_angle():
    """Gain double-negative cells in a dielectric raster, ``theta = 0.5``: SMM's field is the
    first-order one.

    The ETM's power-down labels flip 41 of 98 modes per layer to growing roots there, and the
    relabelled state was 5.9e-11 off where the ``M`` walk on the cascade's labels had been at
    4.0e-12; the walk over ``g`` is at 5.4e-13. (The ETM's own second-order field is 5.6e-12
    from the first-order one.)
    """
    eps = RASTER.astype(complex)
    mu = np.ones_like(eps)
    eps[RASTER == 25.0], mu[RASTER == 25.0] = -4 - 0.1j, -1 - 0.05j

    def field(algo, form):
        solver = _legacy_wavebit_rcwa(backend='numpy', top=1.0, bot=2.0, theta=0.5, phi=PHI, psi=np.pi / 2,
                              wavelength=WL, thickness=[100., 60.], truncation_order=[3, 3],
                              period=[1200., 1200.], layer_algorithm=algo, answers='field')
        solver.structure, solver.mu = eps, mu
        solver.formulation = form
        solver.solve()
        return np.asarray(solver.reconstruct_field(sample_x=5, sample_y=5, sample_z=4,
                                                   incidences=('s', 'p')))
    assert _relative(field('SMM', 'second_order'), field('ETM', 'first_order')) < 1e-11


def _double_negative_cell(algo, theta, form='second_order'):
    eps, mu = -4.0, -1.0
    solver = _legacy_wavebit_rcwa(backend='numpy', top=1.0, bot=wavebit.Material(eps, mu=mu), theta=theta,
                          wavelength=WL, thickness=[100.], layer_algorithm=algo,
                          answers='field', psi=np.pi / 2, phi=PHI, truncation_order=[0, 0],
                          period=[2000., 2000.])
    solver.structure = np.full((1, 2, 2), eps, dtype=complex)
    solver.mu = np.full((1, 2, 2), mu, dtype=complex)
    solver.formulation = form
    solver.solve()
    return np.asarray(solver.reconstruct_field(sample_x=3, sample_y=3, sample_z=4,
                                               incidences=('s', 'p')))


def test_a_uniform_double_negative_on_itself_below_the_real_axis(merge):
    """A uniform double negative on its own medium at ``theta = 0.5-0.3j``: SMM's field is the ETM's.

    The mirror label (:func:`~wavebit._common.conventions.etm_mode_q`) and the decaying one part
    here, and the ``M`` walk was 2.0 off. The first-order path, which labels by flux, vouches
    for the ETM.
    """
    theta = 0.5 - 0.3j
    etm = _double_negative_cell('ETM', theta)
    scale = np.max(np.abs(etm))
    assert np.max(np.abs(_double_negative_cell('ETM', theta, 'first_order') - etm)) < 1e-12 * scale
    assert np.max(np.abs(_double_negative_cell('SMM', theta) - etm)) < 1e-12 * scale


# ----------------------------------------------------------------------- what the state costs

def _count_builds(monkeypatch):
    """Count SMM's calls to the three layer-mode builders."""
    calls = []
    for name in ('layer_modes_1d', 'layer_modes_1d_conical', 'layer_modes_2d'):
        build = getattr(scattering, name)

        def counted(*args, _build=build, **kwargs):
            calls.append(kwargs)
            return _build(*args, **kwargs)
        monkeypatch.setattr(scattering, name, counted)
    return calls


@pytest.mark.parametrize('theta', [0.5, THETA], ids=['real', 'complex'])
@pytest.mark.parametrize('case', list(GRATING_CASES))
def test_each_layer_is_built_once(case, theta, monkeypatch):
    """A field solve builds each layer's modes once -- one ``eig`` -- at any angle.

    The relabel built a second time, on an identical operator, wherever the two label rules
    could part: on the 2D raster of ``studies/rcwa/smm_field_state/smm_field_state.py`` section 8 an SMM field solve
    cost 2.5x the ETM's at a complex angle against 1.6x at a real one, for an ``eig`` whose
    answer it already had. It is 1.6x at both now.
    """
    calls = _count_builds(monkeypatch)
    _grating('SMM', case, theta=theta)
    assert len(calls) == 2
    assert all(kw.get('k0_d') is None for kw in calls)


@pytest.mark.parametrize('case', list(GRATING_CASES))
def test_declaring_the_field_does_not_move_an_amplitude(case):
    """``answers='amplitude'`` and ``answers='field'`` give the same bytes, at a complex angle.

    The cascade is the same call on either rung, and the walk inverts nothing that can be
    singular, so a field solve cannot raise where an amplitude solve does not -- which on
    9a92bf0's gap, before the walk (07ae0f6), it did, for 10 um of vacuum on vacuum, and
    ``auto_detune`` then moved the field rung's amplitudes by 8e-10.
    """
    fields = ('r_s', 'r_p', 't_s', 't_p')
    _, amplitude = _grating('SMM', case, answers='amplitude')
    _, field = _grating('SMM', case, answers='field')
    for f in fields:
        assert np.array_equal(np.asarray(getattr(amplitude.incident, f)),
                              np.asarray(getattr(field.incident, f))), f


@pytest.mark.parametrize('kind', ['te', 'tm'])
def test_a_field_solve_past_the_cap_neither_raises_nor_detunes(kind, merge):
    """10 um of vacuum on vacuum at ``theta = 0.5+0.3j``: the field rung solves at the angle asked.

    There ``M`` is singular in exact arithmetic on any gap, and with the gap's waves of 9a92bf0 it
    rounds to exactly zero on the pure TE/TM paths: the field solve raised ``LinAlgError``, and
    the default ``auto_detune`` re-solved at ``theta * (1 + 1e-12)``. (The conical path's ``M``
    rounded to a residue instead, a wrong field rather than a refusal; the closed-form test above
    is the one that catches it.)
    """
    kw = dict(tetm=kind) if kind in ('te', 'tm') else dict(psi=np.pi / 2, phi=PHI)

    def solve(answers):
        solver = _legacy_wavebit_rcwa(backend='numpy', top=1.0, bot=1.0, theta=THETA, wavelength=WL,
                              thickness=[1e4], truncation_order=[0], period=[2000.],
                              structure=np.ones((1, 1, 4), dtype=complex), layer_algorithm='SMM',
                              answers=answers, **kw)
        return solver.solve()
    with warnings.catch_warnings():
        warnings.simplefilter('error', _legacy_wavebit_AutoDetuneWarning)
        field = solve('field')
    amplitude = solve('amplitude')
    for f in ('r_s', 'r_p', 't_s', 't_p'):
        assert np.array_equal(np.asarray(getattr(amplitude.incident, f)),
                              np.asarray(getattr(field.incident, f))), f


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('case', ['te', 'conical'])
def test_the_state_differentiates_as_the_etms_does(case, backend):
    """``d sum|E|**2 / ds`` through the state, ``s`` scaling the grating's permittivity.

    The walk is built from the cascade's own blocks, so its derivative is the cascade's. (Of the
    relabel's alternatives, negating flipped roots and their ``V`` columns after the fact was
    right in value and 2.5x off in this derivative on both backends: under autodiff a layer's
    ``q`` is a modal generator whose off-diagonals divide by ``q_i + q_j``.)
    """
    pytest.importorskip(backend)
    _legacy_wavebit_enable_x64()
    base = GRATING_CASES[case]['structure']
    kw = {k: v for k, v in GRATING_CASES[case].items() if k != 'structure'}
    samples = dict(sample_x=4, sample_y=3, sample_z=3)

    def gradient(algo):
        if backend == 'jax':
            import jax
            import jax.numpy as jnp

            def loss(s):
                solver = _legacy_wavebit_rcwa(backend='jax', top=1.0, bot=25.0, theta=THETA, wavelength=WL,
                                      thickness=[100., 60.], layer_algorithm=algo,
                                      answers='field', structure=jnp.asarray(base) * (1 + s), **kw)
                return jnp.sum(jnp.abs(solver.solve_field_eager(**samples)[1]) ** 2)
            return float(jax.grad(loss)(0.0))
        import torch
        s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        solver = _legacy_wavebit_rcwa(backend='torch', top=1.0, bot=25.0, theta=THETA, wavelength=WL,
                              thickness=[100., 60.], layer_algorithm=algo, answers='field',
                              structure=torch.as_tensor(base) * (1 + s), **kw)
        torch.sum(torch.abs(solver.solve_field(**samples)[1]) ** 2).backward()
        return float(s.grad)

    etm, smm = gradient('ETM'), gradient('SMM')
    assert abs(smm - etm) < 1e-9 * abs(etm), (smm, etm)


#: ``(kind, theta, structure eps, thickness, incidence side, half-space eps)``: a layer whose root
#: grows past the ETM's cap, where the ETM's own derivative is no referee.
PAST_CAP = {
    'vacuum-te': ('te', THETA, 1.0, 1e4, 'top', 1.0),
    'vacuum-tm': ('tm', THETA, 1.0, 1e4, 'top', 1.0),
    'vacuum-conical': ('conical', THETA, 1.0, 1e4, 'top', 1.0),
    'chip-conical': ('conical', CHIP_THETA, 2.25, 4869., 'bottom', 2.25),
}


@pytest.mark.parametrize('case', list(PAST_CAP))
def test_the_state_differentiates_past_the_cap(case):
    """``d sum|E|**2 / ds`` past the ETM's cap, ``s`` scaling the layer's permittivity.

    Against a fourth-order central difference of the field itself: SMM's ``jax.grad`` is within
    1.5e-9 to 3.9e-9 of it. The ETM's has the wrong sign on the vacuum TE row, is 16% off on the
    conical one and 2% off on the chip -- a defect of the ETM's own record, which is why
    :func:`test_the_state_differentiates_as_the_etms_does` stops below the cap.
    """
    pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    import jax
    import jax.numpy as jnp
    kind, theta, eps, thickness, side, medium = PAST_CAP[case]
    kw = dict(tetm=kind) if kind in ('te', 'tm') else dict(psi=0.27, phi=0.41)
    order, period = ([7], [1654.489947568981]) if case == 'chip-conical' else ([0], [2000.])
    wavelength = 1000. if case == 'chip-conical' else WL

    def loss(s):
        solver = _legacy_wavebit_rcwa(backend='jax', top=medium, bot=medium, theta=theta,
                              wavelength=wavelength, thickness=[thickness], period=period,
                              truncation_order=order, incidence_side=side,
                              structure=jnp.full((1, 1, 8), eps + 0j) * (1 + s),
                              layer_algorithm='SMM', answers='field', **kw)
        cell = solver.solve_field_eager(sample_x=5, sample_y=1, sample_z=6)[1]
        return jnp.sum(jnp.abs(cell) ** 2)

    h = 1e-7
    difference = (8 * (float(loss(h)) - float(loss(-h)))
                  - (float(loss(2 * h)) - float(loss(-2 * h)))) / (12 * h)
    gradient = float(jax.grad(loss)(0.0))
    assert abs(gradient - difference) < 1e-6 * abs(difference), (gradient, difference)
