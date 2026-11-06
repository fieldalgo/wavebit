"""A polarization word on a one-row raster takes the pure 1D TE/TM path, and nothing else moves.

``PlaneWave(polarization='s')`` / ``'p'`` is a declaration that the polarization is fixed, and
the engine's routing (``CoreSolveMixin._polarization_word_holds``) takes the pure TE/TM path
for it wherever that path answers the same light. A Jones pair is a value and keeps the conical
path. Since 2026-11-03; before, every public 1D run was conical.

Two families of checks. The **regression matrix** runs each feature twice, once with the word
(the pure path) and once with the pair that names the same light (the conical path, which
every public 1D run took until this change), and holds every observable to round-off between
the two: two formulations of one problem, written independently, agree. The **fallback pins**
hold the cases the word does not take pure -- a skewed cell, a traced or differentiated azimuth,
a nonzero ``phi`` or ``ky``, ``formulation='first_order'``, a requested scattering matrix -- to
the pair *bit for bit*, which is what the word was lowered to before. Both also check the path
the run reports it took (``Result.diagnostics['solve_path']``) against what ``check`` planned
(``CheckReport.resolved['solve_path']``) and, where a transform could hide it, against the
kernel that was actually called.
"""
import collections

import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.rcwa._engine import engine as _engine

NX = 40
PERIOD, THICKNESS, WAVELENGTH = 1000.0, 200.0, 1363.0
#: The word and the pair that names the same light, and the pure path the word takes.
WORDS = {'s': ((1.0, 0.0), '1d_te'), 'p': ((0.0, 1.0), '1d_tm')}
ORDERS = rcwa.Numerics(orders=(8, 0))
X = np.linspace(-480.0, 480.0, 9)
Y = np.array([0.0, 130.0])


def raster(epsilon=12.25):
    """One row, 40 px: a bar of 18 px, off the cell centre so no pair of orders is mirror-equal."""
    grid = np.ones((1, NX), dtype=complex)
    grid[0, 9:27] = epsilon
    return grid


def grating(polarization='s', *, theta=0.2, phi=None, kx=None, ky=None, side='top', top=1.0,
            bottom=2.25, wavelength=WAVELENGTH, layers=None, lattice=None, mu=None,
            epsilon=None, thickness=THICKNESS, fields=True, scattering=False):
    layer = rcwa.Layer(thickness=thickness, epsilon=raster() if epsilon is None else epsilon, mu=mu)
    stack = dict(top=top if isinstance(top, Material) else Material(top),
                 bottom=bottom if isinstance(bottom, Material) else Material(bottom),
                 layers=layers or (layer,))
    stack.update({'period': (PERIOD, PERIOD)} if lattice is None else {'lattice': lattice})
    source = dict(polarization=polarization, side=side)
    source.update(dict(kx=kx, ky=ky) if kx is not None or ky is not None else dict(theta=theta, phi=phi))
    return rcwa.Problem(units=Units(length='nm'), stack=rcwa.Stack(**stack),
                        excitation=rcwa.Excitation(wavelength=wavelength,
                                                   sources=(rcwa.PlaneWave(**source),)),
                        outputs=rcwa.Outputs(fields=fields, scattering=scattering))


def host(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def observables(result, problem):
    """Every number a single-plane-wave result carries, by name."""
    out = {}
    for port in ('reflected', 'transmitted'):
        o = result.outputs[port]
        for name in ('amplitude_s', 'amplitude_p', 'amplitude_xyz', 'efficiency', 'power',
                     'k_parallel', 'k_normal'):
            out[f'{port}.{name}'] = host(getattr(o, name))
    out['absorption'] = host(result.absorption)
    out['power_entering'] = host(result.outputs['power_entering'])
    if problem.outputs.fields:
        total = sum(float(host(layer.thickness)) for layer in problem.stack.layers)
        z = np.array([0.0, 0.31*total, 0.77*total, total])
        out['field'] = host(result.field(x=X, y=Y, z=z).values)
        for side in ('top', 'bottom'):
            out[f'boundary.{side}'] = host(result.boundary_fields(side=side, x=X, y=Y).values)
    return out


def _family(name):
    """The observables one scale is read from: a quantity that is round-off on one path has no
    scale of its own. A cross-polarized amplitude is an exact zero on the pure path and noise on
    the conical one, measured against the co-polarized amplitude of the same port; a lossless
    stack's absorption is measured against the power entering it."""
    port, _, quantity = name.rpartition('.')
    if quantity in ('amplitude_s', 'amplitude_p'):
        return f'{port}.amplitude_sp'
    return 'budget' if name in ('absorption', 'power_entering') else name


def assert_round_off(pure, conical, rtol):
    """Each observable to ``rtol`` of its family's largest entry, NaN where the other is NaN."""
    assert pure.keys() == conical.keys()
    scales = collections.defaultdict(float)
    for name in pure:
        for values in (pure[name], conical[name]):
            scales[_family(name)] = max(scales[_family(name)],
                                        float(np.nanmax(np.abs(values), initial=0.0)))
    for name in pure:
        a, b = pure[name], conical[name]
        assert a.shape == b.shape, name
        assert np.array_equal(np.isnan(a), np.isnan(b)), name
        gap = float(np.nanmax(np.abs(a - b), initial=0.0))
        scale = scales[_family(name)]
        assert gap <= rtol*scale, f'{name}: {gap:.2e} of {scale:.2e}'


def assert_bits(word, pair):
    assert word.keys() == pair.keys()
    for name in word:
        assert word[name].dtype == pair[name].dtype and np.array_equal(word[name], pair[name],
                                                                        equal_nan=True), name


def paths(result):
    return result.diagnostics['solve_path'], result.diagnostics['report'].resolved['solve_path']


def run_both(word, make, numerics=ORDERS, backend='numpy', precision='double'):
    """The word's run and the pair's, with the paths each reported."""
    pair, _ = WORDS[word]
    out = []
    for polarization in (word, pair):
        problem = make(polarization)
        result = rcwa.run(problem, numerics=numerics, backend=backend, precision=precision)
        out.append((observables(result, problem), result))
    return out


# ---------------------------------------------------------------- the regression matrix

k0 = 2*np.pi/WAVELENGTH
_mu = np.ones((1, NX), dtype=complex)
_mu[0, 9:27] = 1.44

#: Each feature the pure path has to answer as the conical one does.
CASES = {
    'normal incidence': (lambda pol: grating(pol, theta=0.0), ORDERS),
    'theta 0.3': (lambda pol: grating(pol, theta=0.3), ORDERS),
    'theta -0.3': (lambda pol: grating(pol, theta=-0.3), ORDERS),
    'lit from below, glass below': (lambda pol: grating(pol, theta=0.3, side='bottom'), ORDERS),
    'a stated zero azimuth': (lambda pol: grating(pol, theta=0.3, phi=0.0), ORDERS),
    'kx spelling, ky = 0': (lambda pol: grating(pol, kx=0.3*k0, ky=0.0), ORDERS),
    'kx spelling alone': (lambda pol: grating(pol, kx=-0.2*k0), ORDERS),
    'raster mu': (lambda pol: grating(pol, mu=_mu), ORDERS),
    'lossy bar': (lambda pol: grating(pol, epsilon=raster(12.25+1j)), ORDERS),
    'magnetic half-spaces': (lambda pol: grating(pol, top=Material(2.25, mu=1.4),
                                                 bottom=Material(3.24, mu=1.1)), ORDERS),
    'a negative-index substrate': (lambda pol: grating(pol, bottom=Material(-2.0+0.05j, mu=-1.2+0.05j)),
                                   ORDERS),
    'layers merged into the half-spaces': (lambda pol: grating(pol, layers=(
        rcwa.Layer(thickness=30.0, epsilon=np.full((1, NX), 2.25+0j)),
        rcwa.Layer(thickness=THICKNESS, epsilon=raster()),
        rcwa.Layer(thickness=50.0, epsilon=np.full((1, NX), 1.0+0j)))), ORDERS),
    'a spectrum': (lambda pol: grating(pol, wavelength=(1300.0, 1363.0, 1420.0)), ORDERS),
    'ETM': (lambda pol: grating(pol), rcwa.Numerics(orders=(8, 0), algorithm='etm')),
    'detuning': (lambda pol: grating(pol), rcwa.Numerics(orders=(8, 0), detuning=1e-12)),
    'nvf': (lambda pol: grating(pol), rcwa.Numerics(orders=(8, 0), factorization='nvf')),
    'laurent, integrated': (lambda pol: grating(pol), rcwa.Numerics(
        orders=(8, 0), factorization='laurent', fourier_coefficients='integrated')),
    'second_order stated': (lambda pol: grating(pol), rcwa.Numerics(orders=(8, 0),
                                                                    formulation='second_order')),
    # The -1 order exactly grazing in air: a Rayleigh point.
    'a Rayleigh point': (lambda pol: grating(pol, theta=float(np.arcsin(WAVELENGTH/PERIOD - 1.0)),
                                             bottom=1.0), ORDERS),
    # theta = pi/2 + 0.2j from glass: an evanescent incident wave, which has no power to
    # normalize by, so efficiencies are NaN on both paths and the amplitudes carry the answer.
    'evanescent incidence': (lambda pol: grating(pol, theta=np.pi/2 + 0.2j, top=2.25, fields=False),
                             ORDERS),
}


@pytest.mark.parametrize('word', sorted(WORDS))
@pytest.mark.parametrize('case', sorted(CASES))
def test_the_pure_path_answers_as_the_conical_one(case, word):
    make, numerics = CASES[case]
    (pure, word_run), (conical, pair_run) = run_both(word, make, numerics)
    expected = WORDS[word][1]
    count = len(np.atleast_1d(make(word).excitation.wavelength))
    assert paths(word_run) == ((expected,)*count, expected)
    assert paths(pair_run) == (('1d_conical',)*count, '1d_conical')
    assert_round_off(pure, conical, rtol=1e-11)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('word', sorted(WORDS))
def test_the_pure_path_answers_on_every_backend(backend, word):
    pytest.importorskip(backend)
    if backend == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
    (pure, word_run), (conical, _) = run_both(word, lambda pol: grating(pol, theta=0.25),
                                              backend=backend)
    assert paths(word_run) == ((WORDS[word][1],), WORDS[word][1])
    assert_round_off(pure, conical, rtol=1e-11)


@pytest.mark.parametrize('word', sorted(WORDS))
def test_single_precision_agrees_to_its_own_round_off(word):
    (pure, word_run), (conical, _) = run_both(word, lambda pol: grating(pol, theta=0.25),
                                              precision='single')
    assert paths(word_run) == ((WORDS[word][1],), WORDS[word][1])
    assert_round_off(pure, conical, rtol=1e-4)


def test_check_and_prepare_report_the_path_before_any_solve():
    for word, (pair, expected) in WORDS.items():
        assert rcwa.check(grating(word), numerics=ORDERS).resolved['solve_path'] == expected
        prepared = rcwa.prepare(grating(word), numerics=ORDERS)
        assert prepared.report.resolved['solve_path'] == expected
        assert prepared.validate(grating(word)).resolved['solve_path'] == expected
        assert rcwa.check(grating(pair), numerics=ORDERS).resolved['solve_path'] == '1d_conical'
    skew = np.array([[PERIOD, 0.3*PERIOD], [0.0, PERIOD]])
    nvf = rcwa.Numerics(orders=(8, 0), factorization='nvf')
    assert rcwa.check(grating('s', lattice=skew), numerics=nvf).resolved['solve_path'] == '2d'
    assert rcwa.check(grating('s', phi=0.3), numerics=ORDERS).resolved['solve_path'] == '1d_conical'
    assert rcwa.check(grating('s'), numerics=rcwa.Numerics(orders=(8, 0), formulation='first_order')
                      ).resolved['solve_path'] == '2d'
    assert rcwa.check(grating('s', scattering=True), numerics=ORDERS
                      ).resolved['solve_path'] == '1d_conical'


# ---------------------------------------------------------------- derivatives and transforms

def _jax():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    return jax


@pytest.mark.parametrize('word', sorted(WORDS))
def test_jax_derivatives_through_the_pure_path_are_the_conical_ones(word):
    jax = _jax()
    import jax.numpy as jnp

    def total_R(polarization, variable):
        def R(v):
            kw = {'thickness': dict(thickness=v), 'theta': dict(theta=v),
                  'epsilon': dict(epsilon=jnp.where(jnp.asarray(raster()) == 1.0, 1.0+0j, v+0j))}[variable]
            return jnp.sum(rcwa.run(grating(polarization, fields=False, **kw), numerics=ORDERS,
                                    backend='jax').R)
        return R

    pair = WORDS[word][0]
    for variable, at in (('thickness', THICKNESS), ('theta', 0.2), ('epsilon', 12.25)):
        pure = jax.grad(total_R(word, variable))(jnp.float64(at))
        conical = jax.grad(total_R(pair, variable))(jnp.float64(at))
        assert abs(float(pure) - float(conical)) <= 1e-11*max(abs(float(conical)), 1e-3), variable


@pytest.mark.parametrize('word', sorted(WORDS))
def test_torch_autograd_through_the_pure_path_is_the_conical_one(word):
    torch = pytest.importorskip('torch')

    def gradients(polarization):
        thickness = torch.tensor(THICKNESS, dtype=torch.float64, requires_grad=True)
        bar = torch.tensor(12.25, dtype=torch.float64, requires_grad=True)
        epsilon = torch.where(torch.as_tensor(raster().real) == 1.0, torch.ones(1, NX, dtype=torch.float64),
                              bar).to(torch.complex128)
        result = rcwa.run(grating(polarization, thickness=thickness, epsilon=epsilon, fields=False),
                          numerics=ORDERS, backend='torch')
        torch.sum(result.R).backward()
        return result.diagnostics['solve_path'], float(thickness.grad), float(bar.grad)

    path, *pure = gradients(word)
    assert path == (WORDS[word][1],)
    _, *conical = gradients(WORDS[word][0])
    np.testing.assert_allclose(pure, conical, rtol=1e-11)


@pytest.fixture
def kernels(monkeypatch):
    """Which kernel each solve called: a transform traces the call even when it hides a result."""
    calls = collections.Counter()

    def counted(kind, kernel):
        def call(*args, **kwargs):
            calls[kind] += 1
            return kernel(*args, **kwargs)
        return call
    monkeypatch.setattr(_engine, '_SOLVES', {
        algorithm: {kind: counted(kind, kernel) for kind, kernel in table.items()}
        for algorithm, table in _engine._SOLVES.items()})
    return calls


@pytest.mark.parametrize('word', sorted(WORDS))
def test_a_prepared_run_takes_the_pure_path_under_jit_and_vmap(word, kernels):
    """A stated `phi=0.` is a fixed zero inside the trace too, so the word stays pure there."""
    jax = _jax()
    import jax.numpy as jnp
    make = lambda t: grating(word, phi=0.0, thickness=t, fields=False)    # noqa: E731
    prepared = rcwa.prepare(make(THICKNESS), numerics=ORDERS, backend='jax')
    R = lambda t: jnp.sum(prepared.run(make(t)).R)                        # noqa: E731
    thicknesses = jnp.asarray([190.0, 200.0, 210.0])
    eager = [float(R(t)) for t in thicknesses]
    eager_grad = [float(jax.grad(R)(t)) for t in thicknesses]
    kernels.clear()
    jitted = jax.jit(lambda t: prepared.run(make(t)))(jnp.float64(THICKNESS))
    assert jitted.diagnostics['solve_path'] == (WORDS[word][1],)
    np.testing.assert_allclose(jax.vmap(R)(thicknesses), eager, rtol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.vmap(jax.grad(R)))(thicknesses), eager_grad, rtol=1e-10)
    assert kernels['1d'] > 0 and kernels['conical'] == 0, kernels


# ---------------------------------------------------------------- the fallback pins

def _skew():
    return np.array([[PERIOD, 0.3*PERIOD], [0.0, PERIOD]])


#: Where the word does not take the pure path, and the layout it takes instead.
FALLBACKS = {
    # A one-row raster on a skewed cell is a 1D grating whose grooves do not run along y; the
    # 2D layout is the one that expresses it (the scratch prototype that passed `tetm` from the
    # adapter turned this solve into a ValidationError).
    'a skewed cell': (lambda pol: grating(pol, lattice=_skew()),
                      rcwa.Numerics(orders=(8, 0), factorization='nvf'), '2d'),
    'a nonzero azimuth': (lambda pol: grating(pol, phi=0.3), ORDERS, '1d_conical'),
    'a nonzero ky': (lambda pol: grating(pol, kx=0.2*k0, ky=0.1*k0), ORDERS, '1d_conical'),
    'first_order': (lambda pol: grating(pol), rcwa.Numerics(orders=(8, 0), formulation='first_order'),
                    '2d'),
    # The public scattering matrix keeps both polarizations' ports, 2N x 2N with channel 'sp';
    # the pure path's is N x N, one polarization's, so a run that asks for one stays conical.
    'a scattering request': (lambda pol: grating(pol, scattering=True, fields=False), ORDERS,
                             '1d_conical'),
}


def _scattering_blocks(result):
    if result._scattering is None:
        return {}
    return {f'S{i}.{block}': host(getattr(matrix, block))
            for i, matrix in enumerate(result.scattering) for block in ('S11', 'S12', 'S21', 'S22')}


@pytest.mark.parametrize('word', sorted(WORDS))
@pytest.mark.parametrize('case', sorted(FALLBACKS))
def test_a_word_the_pure_path_cannot_answer_solves_bit_for_bit_as_its_pair(case, word):
    make, numerics, expected = FALLBACKS[case]
    (a, word_run), (b, pair_run) = run_both(word, make, numerics)
    assert paths(word_run) == paths(pair_run) == ((expected,), expected)
    a.update(_scattering_blocks(word_run))
    b.update(_scattering_blocks(pair_run))
    assert_bits(a, b)
    if case == 'a scattering request':
        modes = 2*8 + 1
        assert [np.shape(m.S11) for m in word_run.scattering] == [(2*modes, 2*modes)]
        assert word_run.scattering[0].entry_port.channel == 'sp'


def test_a_coherent_source_run_reports_the_2d_cascade_it_takes():
    """Several sources solve on the side-resolved first-order cascade, a word among them."""
    problem = grating('s', wavelength=700.0, fields=False)
    problem = problem.replace(
        outputs=rcwa.Outputs(diffraction=False),
        excitation=problem.excitation.replace(sources=(
            rcwa.PlaneWave(polarization='s', theta=0.1), rcwa.PlaneWave(polarization='p', theta=0.1))))
    assert rcwa.check(problem, numerics=ORDERS).resolved['solve_path'] == '2d'
    assert paths(rcwa.run(problem, numerics=ORDERS)) == (('2d',), '2d')


def test_a_jones_pair_is_a_value_and_stays_conical():
    for pair in ((1.0, 0.0), (0.0, 1.0), (0.6, 0.8j)):
        result = rcwa.run(grating(pair, fields=False), numerics=ORDERS)
        assert paths(result) == (('1d_conical',), '1d_conical')


def test_a_traced_or_differentiated_azimuth_stays_conical(kernels):
    """`phi` swept or differentiated is a degree of freedom only the conical path carries."""
    jax = _jax()
    import jax.numpy as jnp
    torch = pytest.importorskip('torch')
    transforms = {'jit': jax.jit, 'grad': jax.grad, 'jit(grad)': lambda f: jax.jit(jax.grad(f)),
                  'vmap': jax.vmap}

    def total_R(polarization):
        prepared = rcwa.prepare(grating(polarization, phi=0.0, fields=False), numerics=ORDERS,
                                backend='jax')
        return lambda ph: jnp.sum(prepared.run(grating(polarization, phi=ph, fields=False)).R)

    for word, (pair, _) in WORDS.items():
        R = {polarization: total_R(polarization) for polarization in (word, pair)}
        kernels.clear()
        for name, transform in transforms.items():
            at = jnp.asarray([0.0, 0.1]) if name == 'vmap' else jnp.float64(0.0)
            np.testing.assert_array_equal(np.asarray(transform(R[word])(at)),
                                          np.asarray(transform(R[pair])(at)), err_msg=name)
        assert kernels['1d'] == 0 and kernels['conical'] > 0, kernels
        # The plan is read from the concrete problem the preparation was given; the record is
        # the path the traced run took.
        prepared = rcwa.prepare(grating(word, phi=0.0, fields=False), numerics=ORDERS, backend='jax')
        traced = jax.jit(lambda ph: prepared.run(grating(word, phi=ph, fields=False)))(jnp.float64(0.0))
        assert paths(traced) == (('1d_conical',), WORDS[word][1])

        phi = {pol: torch.tensor(0.0, dtype=torch.float64, requires_grad=True) for pol in (word, pair)}
        values = {}
        for pol in (word, pair):
            result = rcwa.run(grating(pol, phi=phi[pol], fields=False), numerics=ORDERS, backend='torch')
            torch.sum(result.R).backward()
            values[pol] = (result.diagnostics['solve_path'], host(result.R), float(phi[pol].grad))
        assert values[word][0] == ('1d_conical',)
        np.testing.assert_array_equal(values[word][1], values[pair][1])
        assert values[word][2] == values[pair][2]
