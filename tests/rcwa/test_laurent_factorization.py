"""``factorization='laurent'``: the direct rule for every product, offered as an opt-in.

Laurent's rule closes every product of the permittivity with a field component with the plain
convolution matrix ``[[eps]]``, and the z slot with ``[[eps]]^-1`` -- the formulation RCWA
started from (Moharam & Gaylord, JOSA 71, 811, 1981), and the one L. Li (JOSA A 13, 1870, 1996)
showed converges slowly wherever a field component jumps with the permittivity. It is here so a
result can be compared like for like with solvers that use it; nothing resolves to it.

What is checked here:

* **The definition**, at the matrix level: both in-plane slots are one ``[[eps]]`` built by index
  loops from the exact pixel integrals, the z slot is its inverse and is Li's z slot, and it is
  not ``[[1/eps]]``.
* **Where it must equal Li**: a uniform layer, and TE on a grating that varies along x only;
  and where it must not -- 1D TM, where its error is first order in the truncation while Li's
  falls faster (Li 1996's result, reproduced).
* **Where it is built**: rasters with either series, shapes on the closed form and painted
  overlaps (the same ``[[eps]]`` as a pixel-exact raster of the same painting), and skewed
  cells, since the rule names no axis. **Where it is refused, by name**: tensor materials, and
  every option only another rule reads.
* **Every door**: the constructor, assignment, the per-call conditions, the spec, the JAX
  pytree, and ``freeze_geometry`` -- whose records are stamped with their rule and refused under
  another.
* **Backends and gradients**: three backends agree, and JAX and PyTorch differentiate it.

The comparison with another solver is ``tests/references/test_fmmax_laurent_golden.py``:
fmmax's ``Formulation.FFT`` assembles the same operator, and agrees to round-off.
"""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
from tests.fixtures.common._conv_frame import slots_about_centre
import pytest

import array_api_compat.numpy as xnp

import wavebit
from wavebit.compat.legacy import Factorization
from wavebit.rcwa._engine import factorization as F
from wavebit.rcwa._engine.geometry import StaleGeometryError, StaticGeometry

try:
    _legacy_wavebit.enable_x64()
except ImportError:
    pass


# ------------------------------------------------------------------------------ the scenes

_SCENE = dict(backend='numpy', top=1.0, bot=2.25, theta=0.21, phi=0.37, psi=0.6, wavelength=0.9,
              period=[1.0, 0.8], thickness=[0.35], truncation_order=[4, 3])

GRID = 20


def _crossed():
    """A lossless crossed grating with no symmetry, contrast 12.25 against 1."""
    eps = np.ones((1, GRID, GRID))
    eps[0, 3:12, 2:10] = 12.25
    eps[0, 8:17, 7:16] = 12.25
    return eps


def _box(c0, c1, r0, r1, material):
    """The shape twin of ``raster[r0:r1, c0:c1] = material`` on ``_SCENE``'s cell, in absolute
    lengths from the cell centre."""
    px, py = _SCENE['period']
    return dict(type='rectangle', material=material,
                cx=px * ((c0 + c1) / 2 / GRID - 0.5), cy=py * ((r0 + r1) / 2 / GRID - 0.5),
                lx=px * (c1 - c0) / GRID, ly=py * (r1 - r0) / GRID)


#: Two rectangles painted one over the other -- the second covers part of the first -- and the
#: raster that paints the same thing pixel for pixel. The shape path resolves the overlap into
#: disjoint polygons (the boundary route); the raster is a staircase with the same steps.
_PAINTED_SHAPES = [dict(background=1.0, shapes=[_box(2, 10, 3, 12, 4.0), _box(7, 16, 8, 17, 9.0)])]


def _painted_raster():
    eps = np.ones((1, GRID, GRID))
    eps[0, 3:12, 2:10] = 4.0
    eps[0, 8:17, 7:16] = 9.0
    return eps


_CIRCLE = [dict(background=1.0, shapes=[dict(type='circle', cx=0.05, cy=-0.03, lx=0.46,
                                             material=6.25)])]

#: A hexagonal cell: Li is refused on it, Laurent is not.
_HEX = np.column_stack(([1.0, 0.0], [0.5, np.sqrt(3.0) / 2]))
_HEX_SCENE = {k: v for k, v in _SCENE.items() if k != 'period'}


def _solve(**kw):
    return _legacy_wavebit.rcwa(**{**_SCENE, **kw}).solve()


def _both(result):
    return np.asarray(result.R), np.asarray(result.T)


def _gap(a, b):
    return max(np.abs(np.asarray(x) - np.asarray(y)).max() for x, y in zip(_both(a), _both(b)))


def _energy(result):
    return float(np.sum(result.R) + np.sum(result.T))


# ---------------------------------------------------------------------------- the definition

def _literal_direct(eps, Mu, Mv):
    """``[[eps]]`` by index loops: each coefficient as the integral over the pixels, laid out
    ``M[(n, m), (n', m')] = c(m - m', n - n')`` with rows ``n * (2 Mu + 1) + m``. Shares nothing
    with the builders -- no FFT, no correction factor, no gather."""
    Nv, Nu = eps.shape

    def weights(N, K):
        left = np.arange(N) / N
        W = np.empty((2 * K + 1, N), dtype=complex)
        for k in range(-K, K + 1):
            W[k + K] = (1.0 / N if k == 0 else
                        (np.exp(-2j * np.pi * k * left) - np.exp(-2j * np.pi * k * (left + 1.0 / N)))
                        / (2j * np.pi * k))
        return W

    c = weights(Nv, 2 * Mv) @ eps @ weights(Nu, 2 * Mu).T          # [kv, ku]
    nu, nv = 2 * Mu + 1, 2 * Mv + 1
    M = np.empty((nv * nu, nv * nu), dtype=complex)
    for n in range(nv):
        for m in range(nu):
            for n2 in range(nv):
                for m2 in range(nu):
                    M[n * nu + m, n2 * nu + m2] = c[n - n2 + 2 * Mv, m - m2 + 2 * Mu]
    return M


@pytest.mark.parametrize('coefficients', ['sampled', 'integrated'])
def test_the_slots_are_the_direct_rule_in_plane_and_its_inverse_in_z(coefficients):
    eps = _crossed()[0] * (1 + 0.1j)        # lossy, so no symmetry of a real matrix helps
    Mu, Mv = 4, 3
    if coefficients == 'sampled':
        slots = F.laurent_conv_matrices(xnp, eps, Mu, Mv)
    else:
        row, col, x, y = F.compression_indices(eps)
        slots = F.laurent_conv_matrices_integrated(
            xnp, F.compress_cell(xnp, eps, row, col), x, y, Mu, Mv)
    direct = _literal_direct(eps, Mu, Mv)
    assert np.array_equal(slots.epx, slots.epy)             # one operator, to the bit
    assert np.abs(slots.epx - direct).max() < 1e-12 * np.abs(direct).max()
    assert np.abs(slots.epz_i - np.linalg.inv(direct)).max() < 1e-12
    assert slots.epxy is None and slots.epyx is None


def test_the_z_slot_is_lis_and_not_the_transform_of_the_reciprocal():
    """``E_z`` is continuous across every vertical interface, so ``D_z = eps E_z`` is a product
    the direct rule closes correctly -- Li's rules take it too, and the two z slots agree. The
    other candidate, ``[[1/eps]]``, is the direct rule applied to ``E_z = (1/eps) D_z``, whose
    factors jump together; it is measurably another matrix, which is what makes this a choice."""
    eps = _crossed()[0]
    Mu, Mv = 4, 3
    laurent = F.laurent_conv_matrices(xnp, eps, Mu, Mv)
    li = F.li_conv_matrices(xnp, eps, Mu, Mv)
    assert np.abs(laurent.epz_i - li.epz_i).max() < 1e-13
    reciprocal = _literal_direct(1.0 / eps, Mu, Mv)
    assert np.abs(laurent.epz_i - reciprocal).max() > 1e-2


@pytest.mark.parametrize('coefficients', ['sampled', 'integrated'])
def test_a_uniform_layer_is_the_same_under_every_rule(coefficients):
    """``eps * I`` exactly in both in-plane slots, and the solve Li's and NVF's to round-off."""
    uniform = np.full((1, 12, 12), 5.0 + 0.2j)
    conv = _legacy_wavebit.rcwa(**_SCENE, structure=uniform, factorization='laurent',
                        fourier_coefficients=coefficients).eps_conv_matrices()
    n = conv.epx.shape[-1]
    assert np.array_equal(np.asarray(conv.epx)[0], (5.0 + 0.2j) * np.eye(n))
    assert np.abs(np.asarray(conv.epz_i)[0] - np.eye(n) / (5.0 + 0.2j)).max() < 1e-16
    laurent = _solve(structure=uniform, factorization='laurent', fourier_coefficients=coefficients)
    for rule in ('li', 'nvf'):
        other = _solve(structure=uniform, factorization=rule, fourier_coefficients=coefficients)
        assert _gap(laurent, other) < 1e-13, rule


# ------------------------------------------------------------------ one-dimensional gratings

_1D = dict(backend='numpy', top=1.0, bot=2.25, theta=0.2, phi=0.0, period=[700.0], wavelength=900.0,
           thickness=[300.0])


def _grating():
    eps = np.ones((1, 1, 40))
    eps[0, 0, :20] = 12.25
    return eps


def _1d_T(rule, order, tetm):
    result = _legacy_wavebit.rcwa(**_1D, structure=_grating(), truncation_order=(order, 0), tetm=tetm,
                          factorization=rule).solve()
    return result, float(np.sum(result.T))


def test_te_on_a_one_dimensional_grating_is_lis_answer():
    """TE reads ``epy`` alone, where both rules build ``[[eps]]`` (Li's nested rule reduces to
    it on a y-invariant layer), so the two answers are one to round-off, order by order."""
    for order in (5, 20, 40):
        li, _ = _1d_T('li', order, 'te')
        laurent, _ = _1d_T('laurent', order, 'te')
        assert _gap(li, laurent) < 1e-13, order


def test_tm_converges_first_order_where_li_converges_faster():
    """Li (1996), reproduced. TM reads ``epx`` -- Li's ``[[1/eps]]^-1`` against Laurent's
    ``[[eps]]`` -- and ``E_x`` jumps where ``eps`` does. Measured against Li at order 200: Laurent
    2.7e-2, 1.4e-2, 7.4e-3 at orders 10, 20, 40 (halving per doubling, first order), Li 1.9e-3,
    2.9e-4, 4.0e-5. Both conserve energy: the slow convergence is not a loss."""
    _, reference = _1d_T('li', 200, 'tm')
    laurent_errors, li_errors = [], []
    for order in (10, 20, 40):
        result, T = _1d_T('laurent', order, 'tm')
        laurent_errors.append(abs(T - reference))
        assert abs(_energy(result) - 1.0) < 1e-12
        li_errors.append(abs(_1d_T('li', order, 'tm')[1] - reference))
    ratios = np.asarray(laurent_errors[:-1]) / np.asarray(laurent_errors[1:])
    assert np.all((ratios > 1.7) & (ratios < 2.3)), ratios
    assert all(a > 10 * b for a, b in zip(laurent_errors, li_errors)), (laurent_errors, li_errors)


def test_the_conical_path_builds_it_and_conserves_energy():
    """The coupled 1D solve reads all three slots; Laurent moves it off Li and keeps it lossless."""
    kw = dict(_1D, phi=0.4, psi=0.3)
    li = _legacy_wavebit.rcwa(**kw, structure=_grating(), truncation_order=(10, 0)).solve()
    laurent = _legacy_wavebit.rcwa(**kw, structure=_grating(), truncation_order=(10, 0),
                           factorization='laurent').solve()
    assert abs(_energy(laurent) - 1.0) < 1e-12
    assert _gap(li, laurent) > 1e-3


# --------------------------------------------------------------------- where it is built

@pytest.mark.parametrize('coefficients', ['sampled', 'integrated'])
def test_a_crossed_grating_is_a_different_answer_that_conserves_energy(coefficients):
    laurent = _solve(structure=_crossed(), factorization='laurent',
                     fourier_coefficients=coefficients)
    assert abs(_energy(laurent) - 1.0) < 1e-12
    for rule in ('li', 'nvf'):
        other = _solve(structure=_crossed(), factorization=rule, fourier_coefficients=coefficients)
        assert _gap(laurent, other) > 1e-3, rule


def test_the_two_coefficient_series_agree_under_laurent():
    """Both are exact for the staircase, and Laurent asks the coefficients for nothing else."""
    sampled = _solve(structure=_crossed(), factorization='laurent')
    integrated = _solve(structure=_crossed(), factorization='laurent',
                        fourier_coefficients='integrated')
    assert _gap(sampled, integrated) < 1e-12


def test_shapes_build_the_same_matrix_as_a_pixel_exact_raster():
    """Rectangles on pixel edges, painted one over the other: the shape path resolves the overlap
    into polygons and transforms them in closed form, the raster integrates its pixels -- two
    routes to one ``[[eps]]``, and under Laurent that is the whole build."""
    shapes = _solve(structure=_PAINTED_SHAPES, factorization='laurent')
    raster = _solve(structure=_painted_raster(), factorization='laurent')
    assert _gap(shapes, raster) < 1e-11
    single = [dict(background=1.0, shapes=[_box(2, 10, 3, 12, 12.25)])]
    one_box = np.ones((1, GRID, GRID))
    one_box[0, 3:12, 2:10] = 12.25
    assert _gap(_solve(structure=single, factorization='laurent'),
                _solve(structure=one_box, factorization='laurent')) < 1e-11


def test_the_shape_builders_refuse_a_normal_field_option_under_laurent():
    """Called directly, the shape builders refuse what they would otherwise ignore -- the same
    rule the solver's keywords follow."""
    from wavebit.rcwa._engine import boundary, parametric
    from wavebit.lattice import as_lattice
    lattice = as_lattice(period=_SCENE['period'])
    with pytest.raises(ValueError, match='no reader'):
        parametric.to_conv_mat_parametric_layers(_CIRCLE, 2, 2, lattice, xp=np,
                                                 factorization='laurent', nvf_projector='linear')
    with pytest.raises(ValueError, match='no reader'):
        parametric.to_conv_mat_parametric_layers(_CIRCLE, 2, 2, lattice, xp=np,
                                                 factorization='laurent', gamma_plans=[None])
    with pytest.raises(ValueError, match='no normal projector'):
        boundary.to_conv_mat_boundary_layer(np, _PAINTED_SHAPES[0]['shapes'], 1.0, 2, 2, lattice,
                                            np.complex128, laurent=True, nvf_projector='linear')
    with pytest.raises(ValueError, match="rectilinear"):
        parametric.to_conv_mat_parametric_layers(_CIRCLE, 2, 2, lattice, xp=np,
                                                 factorization='li')


def test_a_curved_shape_is_built_from_its_closed_form_transform():
    """No normal field, so no grid: ``shape_nvf_grid_size`` is not read, and the circle's
    ``[[eps]]`` is the one NVF starts from."""
    solver = _legacy_wavebit.rcwa(**_SCENE, structure=_CIRCLE, factorization='laurent')
    laurent = solver.eps_conv_matrices()
    nvf = _legacy_wavebit.rcwa(**_SCENE, structure=_CIRCLE).eps_conv_matrices()
    assert np.abs(np.linalg.inv(np.asarray(laurent.epz_i)) - np.linalg.inv(np.asarray(nvf.epz_i))
                  ).max() < 1e-12
    assert np.array_equal(np.asarray(laurent.epx), np.asarray(laurent.epy))
    assert laurent.epxy is None and laurent.epyx is None
    assert abs(_energy(solver.solve()) - 1.0) < 1e-12


@pytest.mark.parametrize('structure', ['raster', 'shapes'])
def test_a_skewed_cell_takes_it_because_the_rule_names_no_axis(structure):
    """Li is refused on a hexagonal cell -- its inverse rule runs along a1 and a2 and would be
    read as x and y. Laurent's two in-plane slots are one scalar convolution, so it builds, and a
    uniform layer there is NVF's answer to round-off."""
    struct = _crossed() if structure == 'raster' else _CIRCLE
    with pytest.raises(NotImplementedError, match="factorization='li'"):
        _legacy_wavebit.rcwa(**_HEX_SCENE, lattice=_HEX, structure=struct, factorization='li').solve()
    result = _legacy_wavebit.rcwa(**_HEX_SCENE, lattice=_HEX, structure=struct,
                          factorization='laurent').solve()
    assert abs(_energy(result) - 1.0) < 1e-12
    uniform = (np.full((1, 8, 8), 3.0) if structure == 'raster'
               else [dict(background=3.0, shapes=[])])
    a = _legacy_wavebit.rcwa(**_HEX_SCENE, lattice=_HEX, structure=uniform, factorization='laurent').solve()
    b = _legacy_wavebit.rcwa(**_HEX_SCENE, lattice=_HEX, structure=uniform, factorization='nvf').solve()
    assert _gap(a, b) < 1e-13


@pytest.mark.parametrize('structure', ['raster', 'circle'])
def test_the_first_order_formulation_reads_the_same_slots(structure):
    """Forcing ``'first_order'`` is how the two formulations are cross-checked: they consume the
    same five slots, so under Laurent's rule they agree to round-off -- and the field is
    reconstructed from them either way."""
    struct = _crossed() if structure == 'raster' else _CIRCLE
    second = _legacy_wavebit.rcwa(**_SCENE, structure=struct, factorization='laurent')
    first = _legacy_wavebit.rcwa(**_SCENE, structure=struct, factorization='laurent', answers='field')
    first.formulation = 'first_order'
    assert _gap(second.solve(), first.solve()) < 1e-13
    assert first.formulation_assigned == _legacy_wavebit.Formulation.FIRST_ORDER
    field = np.asarray(first.reconstruct_field(sample_x=4, sample_y=4, sample_z=3))
    assert np.isfinite(field).all() and np.abs(field).max() > 0


def test_the_permeability_takes_the_same_rule():
    """``mu`` goes through the same builder by duality: ``[[mu]]`` in plane, ``[[mu]]^-1`` in z."""
    eps = _crossed()
    mu = np.ones_like(eps)
    mu[0, 2:9, 5:18] = 2.5
    solver = _legacy_wavebit.rcwa(**_SCENE, structure=eps, mu=mu, factorization='laurent')
    Mu, Mv = _SCENE['truncation_order']
    want = slots_about_centre(F.laurent_conv_matrices(xnp, mu[0].astype(complex), Mu, Mv), (Mu, Mv))
    got = solver.mu_conv_matrices()
    for slot in ('epx', 'epy', 'epz_i'):
        assert np.abs(np.asarray(getattr(got, slot))[0] - getattr(want, slot)).max() < 1e-14
    assert abs(_energy(solver.solve()) - 1.0) < 1e-12


def test_nothing_resolves_to_it():
    """Opt-in: the raster default is still Li and the shape default still NVF, to the bit."""
    assert _legacy_wavebit.rcwa(**_SCENE, structure=_crossed()).factorization == Factorization.LI
    assert _legacy_wavebit.rcwa(**_SCENE, structure=_CIRCLE).factorization == Factorization.NVF
    assert _gap(_solve(structure=_crossed()), _solve(structure=_crossed(), factorization='li')) == 0
    assert _gap(_solve(structure=_CIRCLE), _solve(structure=_CIRCLE, factorization='nvf')) == 0


# ------------------------------------------------------------------ where it is refused

@pytest.mark.parametrize('option', [
    dict(li_nesting='fixed'), dict(li_nesting='symmetric'), dict(raster_normal_method='contour'),
    dict(nvf_projector='linear'), dict(nvf_projector='circular')])
@pytest.mark.parametrize('structure', ['raster', 'shapes'])
@pytest.mark.parametrize('door', ['solve', 'freeze_geometry'])
def test_an_option_only_another_rule_reads_is_refused_by_name(option, structure, door):
    """Laurent nests no rules and builds no normal field, so every option that tunes one is
    refused rather than ignored -- on the solve, and on the freeze that would record it."""
    struct = _crossed() if structure == 'raster' else _CIRCLE
    if structure == 'shapes' and 'raster_normal_method' in option:
        option = dict(shape_nvf_grid_size=64)
    solver = _legacy_wavebit.rcwa(**_SCENE, structure=struct, factorization='laurent', **option)
    (name, value), = option.items()
    with pytest.raises(NotImplementedError, match=rf"{name}={value!r} has no reader here: "
                                                   rf"factorization='laurent'"):
        getattr(solver, door)()


@pytest.mark.parametrize('structure', ['raster', 'shapes', 'uniform shapes'])
def test_tensor_materials_refuse_it_by_name(structure):
    """Tensor layers are built by tensor Li or Popov-Neviere NVF; a Laurent tensor build is not
    written, and answering with the scalar one would be wrong silently. A uniform tensor needs no
    rule at all, which the message says -- and the scene is still refused rather than guessed."""
    if structure == 'raster':
        struct = np.ones((1, 8, 8, 3, 3)) * np.eye(3)
        struct[0, 2:6, 1:5] *= 4.0
    else:
        tensor = wavebit.Material([2.0, 3.0, 4.0])
        struct = [dict(background=1.0, shapes=[] if structure == 'uniform shapes' else
                       [dict(type='rectangle', cx=0.0, cy=0.0, lx=0.3, ly=0.2, material=tensor)])]
        if structure == 'uniform shapes':
            struct = [dict(background=tensor, shapes=[])]
    for door in ('solve', 'freeze_geometry', 'eps_conv_matrices'):
        with pytest.raises(NotImplementedError, match="factorization='laurent' is not built for "
                                                      "tensor materials.*uniform"):
            getattr(_legacy_wavebit.rcwa(**_SCENE, structure=struct, factorization='laurent'), door)()


# ------------------------------------------------------------------------------ the doors

def test_every_way_of_asking_builds_the_same_thing():
    direct = _solve(structure=_crossed(), factorization='laurent')
    by_enum = _solve(structure=_crossed(), factorization=Factorization.LAURENT)
    by_case = _solve(structure=_crossed(), factorization='Laurent')
    solver = _legacy_wavebit.rcwa(**_SCENE, structure=_crossed())
    solver.factorization = 'laurent'
    assigned = solver.solve()
    per_call = _legacy_wavebit.rcwa(**_SCENE, structure=_crossed()).solve(factorization='laurent')
    for other in (by_enum, by_case, assigned, per_call):
        assert _gap(direct, other) == 0
    conv = _legacy_wavebit.rcwa(**_SCENE, structure=_crossed()).eps_conv_matrices(factorization='laurent')
    assert np.array_equal(np.asarray(conv.epx), np.asarray(conv.epy))
    assert Factorization.LAURENT == 'laurent'


def test_a_misspelling_is_refused_at_every_door():
    with pytest.raises(ValueError, match=r"factorization='laurant' is not valid.*'laurent'"):
        _legacy_wavebit.rcwa(**_SCENE, structure=_crossed(), factorization='laurant')
    solver = _legacy_wavebit.rcwa(**_SCENE, structure=_crossed())
    with pytest.raises(ValueError, match=r"factorization='laurant'"):
        solver.factorization = 'laurant'
    with pytest.raises(ValueError, match=r"factorization='laurant'"):
        solver.solve(factorization='laurant')
    with pytest.raises(AttributeError, match=r"no attribute 'factorisation'.*'factorization'"):
        solver.factorisation = 'laurent'
    assert solver.factorization == Factorization.LI
    from wavebit.compat.legacy import Numerics, SpecError
    with pytest.raises(SpecError, match='numerics.factorization'):
        Numerics.from_dict(dict(truncation_order=[2, 2], factorization='laurant'))


def test_the_spec_carries_it():
    from wavebit.compat.legacy import JobSpec, Numerics
    numerics = Numerics.from_dict(dict(truncation_order=[4, 3], factorization='laurent'))
    assert numerics.factorization == Factorization.LAURENT
    assert numerics.as_kwargs()['factorization'] == 'laurent'
    assert Numerics.from_dict(numerics.to_dict()).factorization == Factorization.LAURENT
    spec = JobSpec.from_dict({
        'version': '10',
        'numerics': {'backend': 'numpy', 'truncation_order': [10, 0], 'factorization': 'laurent'},
        'stack': {'top': 1.0, 'bot': 2.25, 'period': [700.0],
                  'layers': [{'thickness': 300.0, 'structure': [[1.0, 1.0, 12.25, 12.25]]}]},
        'incidence': {'wavelength': 900.0, 'theta': 0.2, 'phi': 0.0, 'tetm': 'tm'},
    })
    ran = _legacy_wavebit.run(JobSpec.from_json(spec.to_json()))
    direct = _legacy_wavebit.rcwa(backend='numpy', wavelength=900., period=[700.], truncation_order=[10, 0],
                          top=1., bot=2.25, theta=0.2, phi=0., tetm='tm', thickness=[300.],
                          structure=np.array([[[1.0, 1.0, 12.25, 12.25]]]),
                          factorization='laurent').solve()
    assert _gap(ran, direct) == 0


def test_the_manifest_publishes_it_and_where_it_is_refused():
    from wavebit.manifest import manifest
    doc = manifest()
    assert 'laurent' in doc['enums']['factorization']
    paths = {p['path']: p for p in doc['paths']}
    assert 'laurent' in paths['raster']['factorization']
    assert paths['parametric']['factorization'][0] == 'nvf'           # the default, first
    assert 'laurent' in paths['parametric']['factorization']
    assert 'laurent' not in doc['lattice']['factorizations_refused_on_oblique']
    assert doc['tensor_materials']['factorizations_refused'] == ['laurent']


# -------------------------------------------------------------------- frozen geometry

def test_a_record_carries_the_rule_it_was_frozen_under():
    raster = _legacy_wavebit.rcwa(**_SCENE, structure=_crossed(), fourier_coefficients='integrated',
                          factorization='laurent').freeze_geometry()
    assert raster.factorization == 'laurent' and raster.compression is not None
    assert raster.normal_field is None
    shapes = _legacy_wavebit.rcwa(**_SCENE, structure=_PAINTED_SHAPES,
                          factorization='laurent').freeze_geometry()
    # The painting is planned, the interface set is not: Laurent reads no normal field.
    assert shapes.factorization == 'laurent'
    assert shapes.boundary_plans[0] is not None and shapes.gamma_plans is None
    nvf = _legacy_wavebit.rcwa(**_SCENE, structure=_PAINTED_SHAPES).freeze_geometry()
    assert nvf.factorization == 'nvf' and nvf.gamma_plans is not None
    # The stamp is part of the content key a jitted call caches on: the same plans under two
    # rules are two records.
    assert (StaticGeometry(boundary_plans=[None], factorization='laurent')
            != StaticGeometry(boundary_plans=[None], factorization='nvf'))
    assert (StaticGeometry(boundary_plans=[None], factorization='laurent')
            == StaticGeometry(boundary_plans=[None], factorization='laurent'))


@pytest.mark.parametrize('structure', ['raster', 'painted', 'circle'])
def test_a_record_replays_under_its_own_rule(structure):
    struct, extra = {'raster': (_crossed(), dict(fourier_coefficients='integrated')),
                     'painted': (_PAINTED_SHAPES, {}), 'circle': (_CIRCLE, {})}[structure]
    solver = _legacy_wavebit.rcwa(**_SCENE, structure=struct, factorization='laurent', **extra)
    record = solver.freeze_geometry()
    assert _gap(solver.solve(geometry=record), solver.solve()) == 0


@pytest.mark.parametrize('held, now', [('laurent', 'nvf'), ('nvf', 'laurent'),
                                       ('laurent', 'li'), ('li', 'laurent')])
@pytest.mark.parametrize('structure', ['raster', 'shapes'])
def test_a_record_frozen_under_another_rule_is_refused(held, now, structure):
    """A record holds what its own rule's build decides, and a build reads a missing slot as an
    answer: a Laurent record of shapes has no interface set, which the NVF replay would take as
    "plan blind". So the rule is compared, not the slots, and a mismatch is refused by name --
    eagerly and, since the stamp is a string, under ``jax.jit`` alike."""
    if structure == 'raster':
        struct, extra = _crossed(), dict(fourier_coefficients='integrated')
    else:
        struct, extra = [dict(background=1.0, shapes=[_box(2, 10, 3, 12, 4.0)])], {}
    record = _legacy_wavebit.rcwa(**_SCENE, structure=struct, factorization=held,
                          **extra).freeze_geometry()
    replay = _legacy_wavebit.rcwa(**_SCENE, structure=struct, factorization=now, **extra)
    with pytest.raises(StaleGeometryError, match=rf"taken under factorization='{held}' and is "
                                                 rf"being replayed under factorization='{now}'"):
        replay.solve(geometry=record)
    # The per-call door: a solver configured under the record's rule, asked for another.
    same = _legacy_wavebit.rcwa(**_SCENE, structure=struct, factorization=held, **extra)
    with pytest.raises(StaleGeometryError, match=f"replayed under factorization='{now}'"):
        same.solve(geometry=record, factorization=now)
    with pytest.raises(StaleGeometryError, match=f"replayed under factorization='{now}'"):
        same.eps_conv_matrices(geometry=record, factorization=now)


def test_a_hand_built_record_is_replayed_on_trust_as_before():
    """No stamp, no comparison: a record written by hand says nothing about its rule."""
    solver = _legacy_wavebit.rcwa(**_SCENE, structure=_crossed(), factorization='laurent')
    assert _gap(solver.solve(geometry=StaticGeometry()), solver.solve()) == 0


# ------------------------------------------------------------------ backends and gradients

def _backend_result(backend, structure, **kw):
    return _legacy_wavebit.rcwa(**{**_SCENE, 'backend': backend}, structure=structure,
                        factorization='laurent', **kw).solve()


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('case', ['sampled', 'integrated', 'circle', 'painted'])
def test_every_backend_answers_the_same(backend, case):
    pytest.importorskip(backend)
    struct, extra = {'sampled': (_crossed(), {}),
                     'integrated': (_crossed(), dict(fourier_coefficients='integrated')),
                     'circle': (_CIRCLE, {}), 'painted': (_PAINTED_SHAPES, {})}[case]
    want = _backend_result('numpy', struct, **extra)
    got = _backend_result(backend, struct, **extra)
    to_np = lambda a: a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)  # noqa: E731
    for a, b in zip(_both(want), (to_np(got.R), to_np(got.T))):
        assert np.abs(a - b).max() < 1e-12


def _split_block():
    """The crossed grating as ``rest + value * mask``, ``mask`` its first block."""
    base = _crossed()
    mask = np.zeros_like(base)
    mask[0, 3:12, 2:10] = 1.0
    return base * (1 - mask), mask


def test_jax_and_torch_differentiate_the_raster_and_agree_with_a_difference():
    """Every raster cell is a variable -- Laurent decides nothing from the values -- so a
    gradient reaches the pattern through ``[[eps]]`` and its inverse, on both backends."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp
    rest, mask = _split_block()

    def total(backend, eps):
        return _legacy_wavebit.rcwa(**{**_SCENE, 'backend': backend}, structure=eps,
                            factorization='laurent').solve().T.sum()

    g_jax = float(jax.grad(lambda v: jnp.real(total('jax', jnp.asarray(rest) + v * jnp.asarray(mask))))(
        jnp.asarray(9.0)))
    v = torch.tensor(9.0, dtype=torch.float64, requires_grad=True)
    total('torch', torch.as_tensor(rest) + v * torch.as_tensor(mask)).real.backward()
    g_torch = float(v.grad)
    h = 1e-5
    value = lambda x: float(np.real(total('numpy', rest + x * mask)))  # noqa: E731
    fd = (value(9.0 + h) - value(9.0 - h)) / (2 * h)
    assert abs(g_jax - g_torch) < 1e-10 * max(1.0, abs(g_jax))
    assert abs(g_jax - fd) < 1e-6 * max(1.0, abs(fd))
    assert abs(g_jax) > 1e-4


def test_jax_and_torch_differentiate_a_shape_parameter():
    """A shape's size enters through the closed-form transform alone -- no normal field to
    differentiate -- and both backends agree with a central difference."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')

    def total(backend, lx):
        layers = [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.04, cy=-0.02, lx=lx,
                                                     ly=0.3, angle=0.2, material=6.25)])]
        return _legacy_wavebit.rcwa(**{**_SCENE, 'backend': backend}, structure=layers,
                            factorization='laurent').solve().T.sum()

    import jax.numpy as jnp
    g_jax = float(jax.grad(lambda x: jnp.real(total('jax', x)))(jnp.asarray(0.42)))
    x = torch.tensor(0.42, dtype=torch.float64, requires_grad=True)
    total('torch', x).real.backward()
    h = 1e-6
    fd = (float(np.real(total('numpy', 0.42 + h))) - float(np.real(total('numpy', 0.42 - h)))) / (2 * h)
    assert abs(g_jax - float(x.grad)) < 1e-9 * max(1.0, abs(g_jax))
    assert abs(g_jax - fd) < 1e-6 * max(1.0, abs(fd))


def test_a_jitted_solve_and_a_jitted_replay_keep_the_rule():
    """The rule rides in the JAX treedef's aux data, so a compiled solve builds Laurent's matrix
    -- not the path's default -- and a frozen Laurent record replays under ``jax.jit``."""
    jax = pytest.importorskip('jax')
    from jax import tree_util
    solver = _legacy_wavebit.rcwa(**{**_SCENE, 'backend': 'jax'}, structure=_crossed(),
                          fourier_coefficients='integrated', factorization='laurent')
    leaves, treedef = tree_util.tree_flatten(solver)
    assert tree_util.tree_unflatten(treedef, leaves).factorization == Factorization.LAURENT
    record = solver.freeze_geometry()
    eager = np.asarray(solver.solve().T)
    compiled = np.asarray(jax.jit(lambda s: s.solve(geometry=record).T)(solver))
    np.testing.assert_allclose(compiled, eager, atol=1e-12, rtol=1e-12)
    li = np.asarray(_legacy_wavebit.rcwa(**{**_SCENE, 'backend': 'jax'}, structure=_crossed(),
                                 fourier_coefficients='integrated').solve().T)
    assert np.abs(compiled - li).max() > 1e-3
    shapes = _legacy_wavebit.rcwa(**{**_SCENE, 'backend': 'jax'}, structure=_PAINTED_SHAPES,
                          factorization='laurent')
    record = shapes.freeze_geometry()
    compiled = np.asarray(jax.jit(lambda s: s.solve(geometry=record).T)(shapes))
    np.testing.assert_allclose(compiled, np.asarray(shapes.solve().T), atol=1e-12, rtol=1e-12)
