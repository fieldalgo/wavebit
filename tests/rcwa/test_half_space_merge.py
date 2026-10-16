"""A layer of a half-space's own medium is merged into that half-space.

The plane between a layer and a half-space of the same material reflects nothing, so the stack
with the layer is the stack without it, with the reference planes moved back to the faces. That
is the theory value, and at a complex ``k_par`` it is the only reliable one: solved as a layer, a
wave growing across it turns the last bit of rounding into an O(1) change -- a knife edge
(``docs/theory/rcwa/connecting-algorithms.md`` section 10) -- and SMM and ETM alike return a
plausible wrong number or raise. :mod:`~wavebit.rcwa._engine.half_space_merge` solves the remaining
stack and moves the answer to the faces analytically.

* closed forms: a thick layer of the substrate is the bare interface, lit from either side, on
  knife edges where the unmerged solve is O(1) off (each test runs its own control); a vacuum
  layer in vacuum is the incident plane wave, field included;
* the coupled paths against the stack with the layer deleted, carried by phases written here;
* every rung answers with one set of bits, the scattering matrix and the field move with the
  amplitudes, a merged layer's thickness differentiates;
* real angles keep their bits; ``half_space_merge_tolerance`` takes a near material to be the
  half-space's, at any angle, and is refused where nothing can read it.
"""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import warnings

import numpy as np
import pytest

import wavebit
from wavebit.rcwa._engine import engine

pytestmark = pytest.mark.filterwarnings('ignore::wavebit._common.backend.ComplexIncidenceWarning')

WL = 1000.0


def _maystre(x):
    """``kz`` over ``k0`` on Maystre's cut, ``Re(kz) + Im(kz) > 0``, written here."""
    v = np.sqrt(np.asarray(x, dtype=complex))
    return np.where(v.real + v.imag > 0, v, -v)


def _relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    scale = np.max(np.abs(b))
    return float(np.max(np.abs(a - b)) / scale) if scale else float(np.max(np.abs(a)))


@pytest.fixture
def unmerged(monkeypatch):
    """Solve every layer as a layer: the control each knife-edge test runs."""
    def off():
        monkeypatch.setattr(engine.CoreFactorizationMixin, '_half_space_merge_plan',
                            lambda self: None)
    return off


def _uniform(tetm, side, theta, d, eps=9.0, **kw):
    """A layer of the substrate's permittivity over it, vacuum above."""
    return _legacy_wavebit_rcwa(backend='numpy', tetm=tetm, top=1.0, bot=eps, theta=theta,
                                wavelength=WL, period=[700.], truncation_order=[2], thickness=[d],
                                structure=np.full((1, 1, 4), eps + 0j), incidence_side=side, **kw)


def _interface(tetm, side, theta, d, eps=9.0):
    """The closed form: Fresnel's interface, its faces moved ``d`` into the substrate."""
    e_in, e_out = (1.0, eps) if side == 'top' else (eps, 1.0)
    kx = np.sqrt(e_in) * np.sin(theta)
    k1, k2 = _maystre(e_in - kx ** 2), _maystre(e_out - kx ** 2)
    k0 = 2 * np.pi / WL
    if tetm == 'te':
        r, t = (k1 - k2) / (k1 + k2), 2 * k1 / (k1 + k2)
    else:                                        # r_p the H ratio, t_p the E one
        r = (e_out * k1 - e_in * k2) / (e_out * k1 + e_in * k2)
        t = 2 * e_out * k1 / (e_out * k1 + e_in * k2) * np.sqrt(e_in) / np.sqrt(e_out)
    if side == 'top':                            # the merged layer is the far medium
        return r, t * np.exp(1j * k0 * k2 * d)
    return r * np.exp(2j * k0 * k1 * d), t * np.exp(1j * k0 * k1 * d)


#: Knife edges: a thick layer of the substrate at a complex angle, lit from the vacuum (the layer
#: is the far medium) or through the substrate (the incident one). Unmerged, SMM and ETM are
#: 0.38 to 1.0 off these closed forms.
KNIVES = [('te', 'top', 1.2 + 0.4j, 6e4), ('te', 'bottom', 0.9 + 0.2j, 2e4),
          ('tm', 'top', 1.2 + 0.4j, 6e4), ('tm', 'bottom', 0.5 + 0.3j, 2e4)]


@pytest.mark.parametrize('algo', ['SMM', 'ETM'])
@pytest.mark.parametrize('tetm,side,theta,d', KNIVES)
def test_a_thick_layer_of_the_substrate_is_the_bare_interface(tetm, side, theta, d, algo, unmerged):
    def answer():
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            try:
                res = _uniform(tetm, side, theta, d, layer_algorithm=algo).solve(auto_detune=False)
            except np.linalg.LinAlgError:
                return None
        sub = res.incident
        return (sub.r_s[0, 2], sub.t_s[0, 2]) if tetm == 'te' else (sub.r_p[0, 2], sub.t_p[0, 2])
    r, t = _interface(tetm, side, theta, d)
    scale = max(abs(r), abs(t))
    got = answer()
    assert max(abs(got[0] - r), abs(got[1] - t)) / scale < 1e-12
    unmerged()
    control = answer()
    assert control is None or max(abs(control[0] - r), abs(control[1] - t)) / scale > 1e-6


def test_a_vacuum_layer_in_vacuum_is_the_incident_wave_field_and_all():
    """10 um of vacuum between vacuum half-spaces at ``theta = 0.5+0.3j``: nothing reflects, the
    transmitted wave is the incident one ``d`` further on, and the field is that wave throughout
    -- ``E_y`` and Faraday's ``H = k x E`` of it. Unmerged, SMM left ``r_s = -1.2e-8 + 4.9e-9j``.
    """
    theta, d = 0.5 + 0.3j, 1e4
    solver = _legacy_wavebit_rcwa(backend='numpy', tetm='te', top=1.0, bot=1.0, theta=theta,
                                  wavelength=WL, period=[700.], truncation_order=[2],
                                  thickness=[d], structure=np.ones((1, 1, 4), complex),
                                  answers='field')
    res = solver.solve(auto_detune=False)
    kx = np.sin(theta)
    kz = _maystre(1 - kx ** 2)
    k0 = 2 * np.pi / WL
    assert np.all(res.incident.r_s == 0)
    t = np.asarray(res.incident.t_s)
    assert abs(t[0, 2] - np.exp(1j * k0 * kz * d)) / abs(t[0, 2]) < 1e-13
    assert np.all(np.delete(t, 2) == 0)
    z = np.linspace(0, d, 7)
    cell = np.asarray(solver.reconstruct_field(sample_x=[0.], sample_y=[0.], sample_z=z))[:, 0, 0]
    ey = np.exp(1j * k0 * kz * (d - z))           # down from the lit face, unit there
    assert _relative(cell[:, 0], ey) < 1e-13
    assert _relative(cell[:, 1], kz * ey) < 1e-13   # H_x
    assert _relative(cell[:, 2], kx * ey) < 1e-13   # H_z


# ---------------------------------------------------------------- the coupled paths, by deletion

def _grating(kind, side, structure, thickness, *, top=1.0, bot=2.25, theta=0.3 + 0.08j, mu=None,
             backend='numpy', **kw):
    args = dict(backend=backend, top=top, bot=bot, theta=theta, wavelength=WL,
                thickness=thickness, structure=structure, incidence_side=side, **kw)
    if mu is not None:
        args['mu'] = mu
    if kind == '2d':
        args.update(psi=0.4, phi=0.3, truncation_order=[2, 1], period=[700., 600.])
    elif kind == 'conical':
        args.update(psi=0.4, phi=0.3, truncation_order=[3], period=[700.])
    else:
        args.update(tetm=kind, truncation_order=[3], period=[700.])
    return _legacy_wavebit_rcwa(**args)


def _raster(kind, values):
    grid = (len(values), 4, 6) if kind == '2d' else (len(values), 1, 8)
    out = np.ones(grid, complex)
    for i, v in enumerate(values):
        if v is None:                             # a grating
            out[i, ..., :grid[-1] // 2] = 4.0
        else:
            out[i] = v
    return out


KINDS = ['te', 'tm', 'conical', '2d']


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('kind', KINDS)
def test_a_merged_stack_is_the_deleted_one_moved_to_its_faces(kind, side):
    """The substrate's medium below a grating and the cover's above it, 0.7 and 0.4 um at
    ``theta = 0.9+0.25j``; the theory is the grating alone, its reflection carried back through
    the lit layer and its transmission on through the far one by phases computed here.
    """
    theta, th = 0.9 + 0.25j, [700., 220., 400.]
    full = _grating(kind, side, _raster(kind, [2.25, None, 1.0]), th, theta=theta)
    alone = _grating(kind, side, _raster(kind, [None]), th[1:2], theta=theta)
    got, ref = full.solve(auto_detune=False), alone.solve(auto_detune=False)
    assert full._half_space_merged[:2] == (1, 1)
    kx, ky = (np.asarray(a) for a in full.order_kxy())
    gx, gy = np.meshgrid(kx, np.atleast_1d(ky))
    k0 = 2 * np.pi / WL
    kz = {'top': _maystre(1.0 - gx ** 2 - gy ** 2), 'bottom': _maystre(2.25 - gx ** 2 - gy ** 2)}
    d = {'top': th[2], 'bottom': th[0]}
    lit, far = (side, 'bottom' if side == 'top' else 'top')
    spec = (gx.shape[0] // 2, gx.shape[1] // 2)  # the specular order: the window's centre
    lead = np.exp(1j * k0 * kz[lit][spec] * d[lit])
    f_r = lead * np.exp(1j * k0 * kz[lit] * d[lit])
    f_t = lead * np.exp(1j * k0 * kz[far] * d[far])
    variants = ['incident'] if kind in ('te', 'tm') else ['s_inc', 'p_inc']
    for v in variants:
        a, b = getattr(got, v), getattr(ref, v)
        theory = np.concatenate([np.ravel(b.r_s * f_r), np.ravel(b.r_p * f_r),
                                 np.ravel(b.t_s * f_t), np.ravel(b.t_p * f_t)])
        assert _relative(np.concatenate([np.ravel(a.r_s), np.ravel(a.r_p), np.ravel(a.t_s),
                                         np.ravel(a.t_p)]), theory) < 1e-12, v


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('kind', KINDS)
def test_every_rung_and_output_moves_with_the_amplitudes(kind, side, unmerged):
    """One merged scene, three rungs, one set of bits for ``r``/``t``/``R``/``T``; and against the
    unmerged solve at a mild complex angle, where both are accurate, the field (counts and points,
    both doors), the faces' fields and the scattering matrix agree to rounding.
    """
    layers, th = _raster(kind, [2.25, None, 1.0]), [150., 220., 130.]
    solves = {a: _grating(kind, side, layers, th, answers=a, layer_algorithm='SMM')
              for a in ('efficiency', 'amplitude', 'field')}
    res = {a: s.solve(auto_detune=False) for a, s in solves.items()}
    for a in ('efficiency', 'field'):
        names = ('R', 'T') if a == 'efficiency' else ('r_s', 'r_p', 't_s', 't_p', 'R', 'T')
        for name in names:
            assert np.array_equal(np.asarray(getattr(res[a].incident, name)),
                                  np.asarray(getattr(res['amplitude'].incident, name))), (a, name)
    merged, field = solves['field'], res['field']
    zs = list(np.linspace(0., 500., 11))
    incidences = ('incident',) if kind in ('te', 'tm') else ('incident', 's', 'p')
    cells = [merged.reconstruct_field(sample_x=5, sample_y=3, sample_z=zs, incidences=incidences),
             merged.reconstruct_field(sample_x=5, sample_y=3, sample_z=3, incidences=incidences),
             field.reconstruct_field(sample_x=5, sample_y=3, sample_z=zs)]
    faces = [field.boundary_fields(f, sample_x=[0., 90.]) for f in ('top', 'bottom')]
    scattering = merged.scattering
    unmerged()
    plain = _grating(kind, side, layers, th, answers='field', layer_algorithm='SMM')
    ref = plain.solve(auto_detune=False)
    expected = [plain.reconstruct_field(sample_x=5, sample_y=3, sample_z=zs, incidences=incidences),
                plain.reconstruct_field(sample_x=5, sample_y=3, sample_z=3, incidences=incidences),
                ref.reconstruct_field(sample_x=5, sample_y=3, sample_z=zs)]
    for got, want in zip(cells, expected):
        assert np.shape(got) == np.shape(want)
        assert _relative(got, want) < 1e-13
    for got, f in zip(faces, ('top', 'bottom')):
        assert _relative(got, ref.boundary_fields(f, sample_x=[0., 90.])) < 1e-13
    for block in ('S11', 'S12', 'S21', 'S22'):
        assert _relative(getattr(scattering, block), getattr(plain.scattering, block)) < 1e-13


@pytest.mark.parametrize('kind', KINDS)
def test_the_field_meets_the_ports_and_itself_across_the_merged_faces(kind):
    """Inside a merged layer the field is the half-space's plane waves: at the stack's faces it
    is ``boundary_fields``' total, and at the plane where the merged layer meets the solved stack
    tangential ``E`` and ``H`` are continuous -- one side from the kernels' records, the other
    from the plane waves.
    """
    th = [700., 220., 400.]
    solver = _grating(kind, 'top', _raster(kind, [2.25, None, 1.0]), th, theta=0.9 + 0.25j,
                      answers='field')
    res = solver.solve(auto_detune=False)
    xs, ys = [0., 150.], [0., 70.]
    faces = np.asarray(res.reconstruct_field(sample_x=xs, sample_y=ys, sample_z=[0., sum(th)]))
    for i, face in enumerate(('bottom', 'top')):
        port = np.asarray(res.boundary_fields(face, sample_x=xs, sample_y=ys))
        assert _relative(faces[i][..., [0, 1, 3, 4]], port[..., [0, 1, 3, 4]]) < 1e-13
    for plane in (th[0], th[0] + th[1]):
        below, above = (np.asarray(res.reconstruct_field(sample_x=xs, sample_y=ys,
                                                         sample_z=[plane + e]))[0]
                        for e in (-1e-9, 0.))
        assert _relative(below[..., [0, 1, 3, 4]], above[..., [0, 1, 3, 4]]) < 1e-9


def test_every_layer_merged_is_the_bare_interface_and_one_medium():
    for layers, top in (([2.25, 1.0], 1.0), ([2.25, 2.25], 2.25)):
        for algo in ('SMM', 'ETM'):
            solver = _grating('conical', 'top', _raster('conical', layers), [300., 200.], top=top,
                              answers='field', layer_algorithm=algo)
            res = solver.solve(auto_detune=False)
            assert sum(solver._half_space_merged[:2]) == 2
            cell = res.reconstruct_field(sample_x=3, sample_y=2, sample_z=[0., 250., 500.])
            assert np.all(np.isfinite(cell))
    one = _grating('te', 'top', _raster('te', [2.25, 2.25]), [300., 200.], top=2.25,
                   theta=0.3 + 0.2j).solve(auto_detune=False)
    assert np.all(one.incident.r_s == 0)


# ------------------------------------------------------------------- media the decision reads

@pytest.mark.parametrize('medium', [wavebit.Material(2.25, mu=1.5),
                                    wavebit.Material(-4.0 + 0.1j, mu=-1.0 + 0.05j)],
                         ids=['magnetic', 'double negative'])
@pytest.mark.parametrize('kind', KINDS)
def test_magnetic_and_double_negative_half_spaces_merge_on_their_own_root(kind, medium, unmerged):
    """Merged into a magnetic or a double-negative substrate, the far layer's phase is that
    medium's own root -- the mirror cut for the double negative -- so the answer is the unmerged
    one at a mild complex angle. The shift once took Maystre's root there and was 1.35 off.
    """
    layers = _raster(kind, [medium.epsilon, None])
    mu = np.ones_like(layers)
    mu[0] = medium.mu
    def solve():
        s = _grating(kind, 'top', layers, [180., 150.], bot=medium, theta=0.3 + 0.05j, mu=mu,
                     answers='field')
        return s, s.solve(auto_detune=False)
    solver, got = solve()
    assert solver._half_space_merged.bottom == 1
    unmerged()
    _, want = solve()
    for name in ('r_s', 'r_p', 't_s', 't_p'):
        assert _relative(getattr(got.incident, name), getattr(want.incident, name)) < 1e-13
    assert _relative(got.reconstruct_field(sample_x=3, sample_y=2, sample_z=4),
                     want.reconstruct_field(sample_x=3, sample_y=2, sample_z=4)) < 1e-13


def test_a_shape_layer_merges_when_every_region_is_the_half_space():
    medium = wavebit.Material(2.25)

    def solver(shape_material):
        layers = [{'background': medium, 'shapes': [
                      {'type': 'rectangle', 'cx': 0., 'cy': 0., 'lx': 200., 'ly': 200.,
                       'material': shape_material}]},
                  {'background': 1.0, 'shapes': [
                      {'type': 'rectangle', 'cx': 0., 'cy': 0., 'lx': 300., 'ly': 250.,
                       'material': 4.0}]}]
        s = _grating('2d', 'top', layers, [180., 150.], bot=medium)
        s.solve(auto_detune=False)
        return s._half_space_merged
    assert solver(medium).bottom == 1
    assert solver(2.25).bottom == 1               # a value, not the same object
    assert solver(2.25 * (1 + 1e-15)) is None     # a different material


def test_a_layer_merges_only_when_its_permeability_is_the_half_spaces_too():
    layers = _raster('conical', [2.25, None])
    mu = np.ones_like(layers)
    mu[0] = 1.5
    for bot, mu_layers in ((2.25, mu), (wavebit.Material(2.25, mu=1.5), None),
                           (wavebit.Material(2.25, mu=1.5), mu)):
        s = _grating('conical', 'top', layers, [180., 150.], bot=bot, mu=mu_layers)
        s.solve(auto_detune=False)
        merged = s._half_space_merged is not None
        assert merged == (mu_layers is not None and not isinstance(bot, float)), (bot, mu_layers)


def test_real_angles_keep_their_bits():
    """A layer of a half-space's medium at a real ``k_par`` is solved as a layer, as before: the
    answer is the same either way, and the bits of every real-angle solve stay where they were --
    with a tolerance too, which merges only a run it snaps.
    """
    for kind in KINDS:
        for tolerance in (0., 1e-8):
            s = _grating(kind, 'top', _raster(kind, [2.25, None, 1.0]), [150., 220., 130.],
                         theta=0.3, half_space_merge_tolerance=tolerance)
            s.solve(auto_detune=False)
            assert s._half_space_merged is None


def test_an_excitation_solves_every_layer():
    """The face shift carries one incident order; an excitation lights several."""
    s = _legacy_wavebit_rcwa(backend='numpy', tetm='te', top=1.0, bot=2.25, theta=0.3 + 0.08j,
                             wavelength=WL, period=[1500.], truncation_order=[3],
                             thickness=[150., 220.], structure=_raster('te', [2.25, None]),
                             excitation={0: 1.0, -1: 0.5})
    s.solve(auto_detune=False)
    assert s._half_space_merged is None


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_merged_layer_thickness_differentiates(side):
    torch = pytest.importorskip('torch')
    th = np.array([400., 220., 300.])

    def objective(thickness, backend):
        lib = torch if backend == 'torch' else np
        s = _grating('conical', side, _raster('conical', [2.25, None, 1.0]), thickness,
                     backend=backend, answers='field')
        res = s.solve(auto_detune=False)
        assert s._half_space_merged is not None
        cell = res.reconstruct_field(sample_x=[50.], sample_y=[0.], sample_z=[100., 500., 800.])
        return (lib.sum(lib.abs(res.incident.r_s) ** 2) + lib.sum(lib.abs(res.incident.t_p) ** 2)
                + 1e-2 * lib.sum(lib.abs(cell) ** 2))
    t = torch.tensor(th, dtype=torch.float64, requires_grad=True)
    objective(t, 'torch').backward()
    for i in range(3):
        h = np.zeros(3)
        h[i] = 1e-3
        fd = (objective(list(th + h), 'numpy') - objective(list(th - h), 'numpy')) / 2e-3
        assert abs(float(t.grad[i]) - fd) < 1e-8 * max(abs(fd), 1e-3), i


# ----------------------------------------------------------------- half_space_merge_tolerance

@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('kind', KINDS)
def test_a_tolerance_takes_a_near_material_to_be_the_half_spaces(kind, side):
    """``2.25 (1 + 3e-9)`` on a substrate of 2.25 and ``1 - 2e-9`` under a vacuum cover, with a
    tolerance of ``1e-8``, at a real angle: the answer is the scene with exactly 2.25 and 1, not
    the scene asked for (which differs from it by 4e-9).
    """
    th = [400., 220., 300.]
    exact = _grating(kind, side, _raster(kind, [2.25, None, 1.0]), th, theta=0.3,
                     answers='field').solve(auto_detune=False)
    near = _raster(kind, [2.25 * (1 + 3e-9), None, 1.0 - 2e-9])
    solver = _grating(kind, side, near, th, theta=0.3, answers='field',
                      half_space_merge_tolerance=1e-8)
    snapped = solver.solve(auto_detune=False)
    assert solver._half_space_merged.snapped == (0, 2)
    for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
        assert _relative(getattr(snapped.incident, name), getattr(exact.incident, name)) < 1e-13
    assert abs(snapped.incident.power_entering - exact.incident.power_entering) < 1e-14
    assert _relative(snapped.reconstruct_field(sample_x=3, sample_y=2, sample_z=4),
                     exact.reconstruct_field(sample_x=3, sample_y=2, sample_z=4)) < 1e-13
    asked = _grating(kind, side, near, th, theta=0.3).solve(auto_detune=False)
    assert _relative(asked.incident.T, exact.incident.T) > 1e-10


@pytest.mark.parametrize('answers', ['efficiency', 'amplitude'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_snapped_lossy_incident_medium_absorbs_between_the_faces(side, answers):
    """A lossy incident medium at a real ``k_par`` (a phase angle): the merged layer absorbs, so
    ``power_entering`` is read again on the full stack's lit face rather than moved. It is the
    exact scene's, on the efficiency rung too.
    """
    lit = 2.25 + 0.2j
    top, bot = (lit, 1.0) if side == 'top' else (1.0, lit)
    theta = np.arcsin(0.4 / np.sqrt(lit))
    th = [400., 220., 300.]
    exact = _grating('2d', side, _raster('2d', [bot, None, top]), th, top=top, bot=bot,
                     theta=theta, answers=answers).solve(auto_detune=False)
    near = _grating('2d', side, _raster('2d', [bot * (1 + 2e-10), None, top * (1 + 2e-10)]), th,
                    top=top, bot=bot, theta=theta, answers=answers,
                    half_space_merge_tolerance=1e-8).solve(auto_detune=False)
    assert abs(near.incident.power_entering - exact.incident.power_entering) < 1e-14
    assert _relative(near.incident.T, exact.incident.T) < 1e-13


@pytest.mark.parametrize('value', [-1e-9, 1.0, 2.0, float('nan'), float('inf'), True, 1e-3j, 'x'])
def test_the_tolerance_is_a_relative_number_below_one(value):
    with pytest.raises((ValueError, TypeError)):
        _legacy_wavebit_rcwa(backend='numpy', half_space_merge_tolerance=value)


def test_a_tolerance_nothing_can_read_is_refused():
    """Where no merge can read it -- an excitation, matrices handed to ``solve_conv``, a
    differentiated material -- a positive tolerance is refused: the scene asked for is the
    snapped one, and answering the other would be a silent wrong answer. 0 is never refused.
    """
    torch = pytest.importorskip('torch')
    layers = _raster('te', [2.25, None])
    s = _legacy_wavebit_rcwa(backend='numpy', tetm='te', top=1.0, bot=2.25, theta=0.3,
                             wavelength=WL, period=[1500.], truncation_order=[3],
                             thickness=[100., 100.], structure=layers,
                             excitation={0: 1.0, -1: 0.5}, half_space_merge_tolerance=1e-8)
    with pytest.raises(NotImplementedError, match='excitation'):
        s.solve()
    s = _grating('te', 'top', layers, [100., 100.], half_space_merge_tolerance=1e-8)
    with pytest.raises(NotImplementedError, match='solve_conv'):
        s.solve_conv(*s.eps_conv_matrices())
    s = _grating('te', 'top', torch.tensor(layers, requires_grad=True), [100., 100.],
                 backend='torch', half_space_merge_tolerance=1e-8)
    with pytest.raises(NotImplementedError, match='differentiate'):
        s.solve()
    s.half_space_merge_tolerance = 0.
    s.solve()
    assert s._half_space_merged is None           # a differentiated layer is a layer


# ------------------------------------------------------------------ the solver's own field door

@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_height_on_a_merged_face_keeps_its_whole_derivative(side, unmerged):
    """A sample on a merged layer's face -- the stack's top, or where the merged layer meets the
    solved ones -- differentiates as the unmerged solve's does. Clipping the selected sample into
    its zone once split the tangent with the bound: half the derivative, on both backends.
    """
    torch = pytest.importorskip('torch')
    th = [60., 120., 80.]

    def gradient():
        t = torch.tensor(th, dtype=torch.float64, requires_grad=True)
        s = _grating('te', side, _raster('te', [2.25, None, 1.0]), t, backend='torch',
                     theta=0.3 + 0.05j, answers='field')
        res = s.solve(auto_detune=False)
        cell = res.reconstruct_field(sample_x=[0., 37.], sample_y=[0.], sample_z=[180., 260.])
        torch.sum(cell.real + cell.imag).backward()
        return s._half_space_merged, t.grad.numpy()
    plan, merged = gradient()
    assert plan is not None
    unmerged()
    plan, plain = gradient()
    assert plan is None
    assert np.max(np.abs(merged - plain)) < 1e-12 * np.max(np.abs(plain))


def test_the_solver_door_keeps_its_sampling_rules_after_a_merge():
    """Heights within 1e-9 outside the stack, where a face measured from the other one lands,
    and on a hexagonal cell a count along ``x`` with explicit ``y``: the door answers both on an
    unmerged solve, and still does on a merged one.
    """
    th = [60., 120., 80.]
    solver = _grating('te', 'bottom', _raster('te', [2.25, None, 1.0]), th, answers='field')
    solver.solve(auto_detune=False)
    faces = [-1e-12, 60., sum(th) + 1e-12]                 # faces one rounding outside
    assert np.all(np.isfinite(solver.reconstruct_field(sample_x=4, sample_z=faces)))
    with pytest.raises(ValueError):
        solver.reconstruct_field(sample_x=4, sample_z=[-1e-6])

    period = 400.
    hexagon = np.column_stack(([period, 0.], [period / 2, period * np.sqrt(3) / 2]))
    layers = np.ones((3, 12, 12), complex)
    layers[0], layers[1, 3:8, 4:9], layers[2] = 2.25, 5.76, 1.0
    s = _legacy_wavebit_rcwa(backend='numpy', structure=layers, truncation_order=[2, 2],
                             thickness=th, wavelength=WL, top=1.0, bot=2.25, theta=0.3 + 0.05j,
                             phi=0.2, psi=0.4, lattice=hexagon, factorization='nvf',
                             answers='field')
    s.solve(auto_detune=False)
    assert s._half_space_merged is not None
    cell = s.reconstruct_field(4, np.array([-50.0, 0.0, 80.0]), np.array([10.0, 100.0, 250.0]))
    assert np.shape(cell) == (1, 3, 3, 4, 6)


def test_a_traced_angle_merges_and_snaps():
    """Under ``jax.jit`` or ``vmap`` over the angle, with the materials concrete, whether
    ``k_par`` is real cannot be read; the runs merge, which is the theory value at any angle, and
    a tolerance snaps as it does eagerly. Taking no merge there once answered the unsnapped
    scene in silence, and a knife edge with ``NaN`` from ETM.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    layers = np.stack([np.full((2, 2), 2.25 * (1 + 1e-3)), np.full((2, 2), 3.0)])
    s = _legacy_wavebit_rcwa(backend='jax', top=1.0, bot=2.25, theta=0.29, phi=0.0,
                             polarization=(1, 0), truncation_order=[1, 1], period=[700., 800.],
                             wavelength=800., thickness=[180., 100.], structure=layers,
                             half_space_merge_tolerance=1e-2)
    eager = float(jnp.sum(s.solve().incident.R))
    jitted = jax.jit(lambda t: jnp.sum(s.solve(theta=t).incident.R))(jnp.asarray(0.29))
    mapped = jax.vmap(lambda t: jnp.sum(s.solve(theta=t).incident.R))(jnp.asarray([0.29]))
    assert abs(float(jitted) - eager) < 1e-13 and abs(float(mapped[0]) - eager) < 1e-13
    s.half_space_merge_tolerance = 0.
    assert abs(float(jnp.sum(s.solve().incident.R)) - eager) > 1e-6   # the scene asked for


def test_unreadable_and_non_finite_materials():
    """A PyTorch conjugate view is read through, as the solve reads it; a NaN is no material and
    so not the half-space's, whatever the tolerance."""
    torch = pytest.importorskip('torch')
    from wavebit.rcwa._engine.half_space_merge import material_match
    layers = torch.conj(torch.tensor(_raster('te', [2.25 - 0j, None]), dtype=torch.complex128))
    s = _grating('te', 'top', layers, [100., 100.], backend='torch')
    s.solve(auto_detune=False)
    assert s._half_space_merged is not None
    assert material_match([(np.array([2.25, np.nan]), None)], (2.25, 1.0), 1e-3) is None
    assert material_match([(np.array([2.25, np.inf]), None)], (2.25, 1.0), 1e-3) is None


def test_an_unlit_order_that_overflows_leaves_every_derivative_finite():
    """130 um of the cover merged at ``theta = 0.001-0.3j``: the unlit orders' phases across it
    overflow, and their amplitudes are exactly zero. Masking the factor after its ``exp`` kept the
    values finite and sent ``0 * inf`` into the derivative; the exponent is masked now, so the
    thickness gradient of ``R`` and of the field is the central difference's.
    """
    torch = pytest.importorskip('torch')
    d0 = np.array([100.0, 130000.0])

    def solve(backend, d):
        return _legacy_wavebit_rcwa(
            backend=backend, tetm='te', top=2.25, bot=1.0, theta=0.001 - 0.3j,
            truncation_order=1, wavelength=500.0, period=[500.0 / 1.2],
            structure=np.stack([np.full((1, 4), 4.0 + 0j), np.full((1, 4), 2.25 + 0j)]),
            thickness=d, answers='field').solve(auto_detune=False)
    d = torch.tensor(d0, dtype=torch.float64, requires_grad=True)
    res = solve('torch', d)
    cell = res.reconstruct_field(sample_x=[0.], sample_y=[0.], sample_z=[50.])
    for value in (torch.sum(res.R), torch.sum(torch.abs(cell) ** 2)):
        grad, = torch.autograd.grad(value, d, retain_graph=True)
        assert torch.all(torch.isfinite(grad))
    grad, = torch.autograd.grad(torch.sum(solve('torch', d).R), d)
    h = np.array([1e-3, 0.])
    fd = (np.sum(solve('numpy', d0 + h).R) - np.sum(solve('numpy', d0 - h).R)) / 2e-3
    assert abs(float(grad[0]) - fd) < 1e-7 * abs(fd)


def test_a_job_that_does_not_use_the_tolerance_keeps_its_key():
    """The field joined ``rcwa.Numerics`` after ``wavebit.job/v1`` was fixed: written only off
    its default, so every saved run record keeps its identity, and a set value round-trips."""
    from wavebit import Job
    plain, snapped = wavebit.rcwa.Numerics(), wavebit.rcwa.Numerics(half_space_merge_tolerance=1e-6)
    layer = wavebit.rcwa.Layer(thickness=.2, background=wavebit.Material(2.25))
    problem = wavebit.rcwa.Problem(
        units=wavebit.Units(length='um'),
        stack=wavebit.rcwa.Stack(period=(.8, .9), layers=(layer,)),
        excitation=wavebit.rcwa.Excitation(wavelength=np.array([.6]),
                                           sources=(wavebit.rcwa.PlaneWave(),)))
    job = Job(solver='rcwa', problem=problem, numerics=plain)
    assert 'half_space_merge_tolerance' not in job.to_json()
    other = Job(solver='rcwa', problem=problem, numerics=snapped)
    assert other.key != job.key
    assert Job.from_json(other.to_json()).numerics.half_space_merge_tolerance == 1e-6
