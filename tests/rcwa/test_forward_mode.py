"""Forward-mode autodiff through ``solve()``: the reverse-mode derivative, on every path.

``jax.jvp``, ``jax.jacfwd`` and ``torch.autograd.forward_ad`` are checked against ``jax.grad`` /
``torch.autograd`` (to 1e-11 relative) and against a central difference on NumPy, extrapolated in
the step, for ``R``, ``T``, an amplitude and the field at a point -- on the 1D TE and TM paths,
the conical path, the 2D Li, Laurent and NVF factorizations, the z-coupled (``'first_order'``)
formulation, analytic shapes (a rectangle, a disc, an oblique cell, a two-shape layer), a metal
where the NVF inverse-rule floor binds, a magnetic material, tensor layers, the scattering matrix,
an ``excitation`` and the jitted ``solve_field`` door, in geometry, material and incidence -- the
cell included: an NVF raster's period and skew, a hexagonal cell's length, and a cell handed in
as a bare array or as nested lists.

Until 2026-10-02 forward mode was unsupported. The modal operator's first-order rule
(``modal_gradient._variation``) was a JAX ``custom_vjp``, so ``jax.jvp`` raised (``can't apply
forward-mode autodiff (jvp) to a custom_vjp function``), and on PyTorch an ``autograd.Function``
entered only for ``requires_grad``, which a ``forward_ad`` dual does not set: the tangent came
back wrong with no error. Measured on these scenes at 43ae852, the weighted read-out's tangent
on PyTorch against reverse mode: 0.0 against 0.272 in the 1D grating's permittivity, -1.11
against -3.02 in the 2D raster's ``theta``, 3.58 against 1.19 in the rectangle's width, 0.0154
against 0.0179 in the gold's ``Re eps``, 0.0 against 0.0573 in the first-order tensor's
permittivity. Several dispatches outside the modal rule were as blind to a dual -- each asked
``requires_grad``, or took a successful NumPy conversion for a constant -- and each is pinned by
a scene that reaches it: the integrated raster's pixel compression, a shape layer's projector
cache (a two-shape layer's material), the tensor-coupling test, the lattice built from a
``period`` (and with it the result's sample grid), a cell given as an array or as lists, the NVF
raster's cell metric, the solver's own field sampler, and the ``auto_detune`` retry.

Second order stays refused by name wherever the modal operator moves under both derivatives:
the modal rule holds the basis fixed, so differentiating it again would drop the eigenvectors'
own variation and return a partial Hessian.

``torch.autograd`` through ``torch.func.vmap`` -- reverse mode, ``forward_ad`` and a captured
parameter beside a batched input -- is checked against eager autograd lane by lane in
``test_torch_vmap_autograd.py``, which builds on the scenes here; the ``torch.func`` grad and jvp
transforms are refused by name, here and there.
"""

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import functools

import numpy as np
import pytest

import wavebit

# CI may run each test of this file in a different shard (tools/run_verification.py
# --shard); tests that share a cache carry one shard_unit name.
pytestmark = [pytest.mark.filterwarnings('ignore'), pytest.mark.shard_unit(per='test')]


# ------------------------------------------------------------------------------- the scenes

def _lift(backend, a):
    if backend == 'numpy':
        return np.asarray(a)
    if backend == 'jax':
        import jax.numpy as jnp
        return jnp.asarray(a)
    import torch
    return torch.as_tensor(np.asarray(a))


def _stack(backend, items):
    if backend == 'numpy':
        return np.array([float(np.real(np.asarray(i))) for i in items])
    if backend == 'jax':
        import jax.numpy as jnp
        return jnp.stack([jnp.real(i) for i in items])
    import torch
    return torch.stack([torch.real(i) if torch.is_complex(i) else i for i in items])


def _read_out(backend, result, amplitude, index, point=None, counted=None):
    """``R``, ``T``, the real and imaginary parts of one amplitude, and three field values."""
    a = getattr(result, amplitude).reshape(-1)[index]
    items = [result.R.sum(), result.T.sum(), a.real, a.imag]
    if point is not None:
        f = result.reconstruct_field(*point)[0, 0, 0]
        items += [f[1].real, f[3].imag, f[4].real]
    if counted is not None:
        cell = result.reconstruct_field(*counted)
        items += [cell[0, 1, 2, 1].real, cell[0, 0, 1, 3].imag]
    return _stack(backend, items)


GRATING = np.array([[[1.0, 4.0 + .08j, 4.0 + .08j, 1.0]]])
GRATING_DIRECTION = np.array([[[1.0, -.3, .2, .5]]])
ONE_D = dict(period=(1400., 1500.), wavelength=900., thickness=(170.,), truncation_order=(3, 0),
             theta=0.2, top=1.0, bot=2.25, answers='field')


def one_d(parameter, algorithm, amplitude, **polarization):
    def f(b, p):
        cfg = dict(ONE_D, layer_algorithm=algorithm, **polarization)
        eps = _lift(b, GRATING)
        if parameter == 'eps':
            eps = eps + p * _lift(b, GRATING_DIRECTION)
        elif parameter == 'mu':
            cfg['mu'] = _lift(b, np.full((1, 1, 4), 1.3 + .02j)) + p * _lift(b, GRATING_DIRECTION)
        else:
            cfg[parameter] = (p,) if parameter == 'thickness' else p
        solver = _legacy_wavebit_rcwa(backend=b, structure=eps, **cfg)
        return _read_out(b, solver.solve(), amplitude, 3, point=([57.], [0.], [83.]))
    return f


RASTER = np.ones((1, 8, 8), dtype=complex)
RASTER[0, 2:6, 2:6] = 9.0 + 0.2j
#: Two pixels of the inner square, moved apart: under integrated coefficients the equal pixels
#: around them are merged unless the build sees a derivative coming.
PIXELS = np.zeros((1, 8, 8))
PIXELS[0, 2, 3], PIXELS[0, 5, 5] = 1.0, -0.5
#: The inner square's material as a whole: the NVF raster's interfaces stay where they are.
SQUARE = np.zeros((1, 8, 8))
SQUARE[0, 2:6, 2:6] = 1.0
TWO_D = dict(period=[0.77, 0.77], thickness=[0.2], theta=0.2, phi=0.3, psi=0.4, wavelength=0.777,
             top=1.0, bot=2.25, truncation_order=[2, 2], answers='field', layer_algorithm='SMM')


def raster(parameter, direction=PIXELS, **options):
    def f(b, p):
        cfg = dict(TWO_D, **options)
        eps = _lift(b, RASTER)
        if parameter == 'eps':
            eps = eps + p * _lift(b, direction)
        elif parameter == 'period':
            cfg['period'] = [p, _lift(b, 0.77)]
        else:
            cfg[parameter] = [p] if parameter == 'thickness' else p
        solver = _legacy_wavebit_rcwa(backend=b, structure=eps, **cfg)
        # A counted sample grid spans the cell, so its points move with the period.
        return _read_out(b, solver.solve(), 'r_s', 6, point=([0.1], [-0.05], [0.07]),
                         counted=(3, 2, [0.07]))
    return f


def scattering_matrix(b, p):
    cfg = dict(TWO_D, answers='amplitude')
    solver = _legacy_wavebit_rcwa(backend=b, structure=_lift(b, RASTER) + p * _lift(b, PIXELS), **cfg)
    solver.solve()
    S = solver.scattering
    blocks = [S.S11[3, 7], S.S21[5, 5], S.S12[0, 9], S.S22[12, 2]]
    return _stack(b, [v for z in blocks for v in (z.real, z.imag)])


SHAPES = dict(period=[0.77, 0.77], thickness=[0.25], theta=0.35, phi=0.4, psi=0.1, wavelength=0.777,
              top=1.0, bot=2.25, truncation_order=[2, 2], answers='field')


def shape(parameter, kind='rectangle', **options):
    def f(b, p):
        cfg = dict(SHAPES, **options)
        piece = dict(type=kind, cx=0.02, cy=-0.01, lx=0.385, ly=0.231, material=12.25 + 0.3j)
        if kind == 'rectangle':
            piece['angle'] = 0.4
        else:
            piece.pop('ly')
        if parameter in ('lx', 'cx'):
            piece[parameter] = p
        elif parameter == 'Im eps':
            piece['material'] = 12.25 + 1j * p
        elif parameter == 'mu':
            piece['material'] = wavebit.Material(12.25 + 0.3j, mu=p)
        else:
            cfg[parameter] = p
        solver = _legacy_wavebit_rcwa(backend=b, structure=[dict(background=1.44, shapes=[piece])], **cfg)
        return _read_out(b, solver.solve(), 't_p', 4, point=([0.12], [0.03], [0.11]))
    return f


OBLIQUE = np.radians(100.0)


def _oblique_solver(b, angle=OBLIQUE, eps=5.76):
    piece = dict(type='circle', cx=-50.0, cy=-20.0, lx=140.0, material=eps)
    return _legacy_wavebit_rcwa(backend=b, wavelength=900.0, lattice=wavebit.Lattice.oblique(400.0, 400.0, angle),
                        truncation_order=[2, 2], top=1.0, bot=1.0, theta=0.07, phi=0.31, psi=0.2,
                        thickness=[300.0], answers='field', shape_nvf_grid_size=64,
                        structure=[dict(background=1.0, shapes=[piece])])


def oblique_cell(parameter):
    """A disc on a 100-degree cell; ``parameter`` is the angle between the lattice vectors or the
    disc's permittivity."""
    def f(b, p, geometry=None):
        solver = _oblique_solver(b, **{'angle' if parameter == 'angle' else 'eps': p})
        return _read_out(b, solver.solve(geometry=geometry), 'r_s', 7, point=([30.], [10.], [120.]))
    return f


GOLD = -28.443519 + 1.753958j        # Johnson & Christy, 852 nm


def gold_disks(parameter):
    """Gold disks 140 nm across in glass at 81 harmonics: the NVF floor binds on two modes."""
    def f(b, p):
        lx, eps = (p, GOLD) if parameter == 'lx' else (140.0, p + 1j * GOLD.imag)
        piece = dict(type='circle', cx=0.0, cy=0.0, lx=lx, material=eps)
        solver = _legacy_wavebit_rcwa(backend=b, structure=[dict(background=2.25, shapes=[piece])],
                              top=2.25, bot=2.25, theta=0.3, phi=0.4, psi=0.7, wavelength=852.0,
                              period=(450.0, 450.0), thickness=(50.0,), truncation_order=(4, 4))
        return _read_out(b, solver.solve(), 'r_s', 40)
    return f


def _tensor(coupled):
    base = np.zeros((1, 4, 4, 3, 3), dtype=complex)
    base[..., 0, 0], base[..., 1, 1], base[..., 2, 2] = 2.5, 2.2, 2.0
    base[0, 1:3, 1:3, 0, 0], base[0, 1:3, 1:3, 1, 1] = 6.0 + 0.1j, 5.0
    base[0, 1:3, 1:3, 0, 1] = base[0, 1:3, 1:3, 1, 0] = 0.7
    if coupled:
        base[0, 1:3, 1:3, 0, 2] = base[0, 1:3, 1:3, 2, 0] = 0.4
    # The inner phase as a whole, so its interfaces stay where they are; with a z-coupling that,
    # on the uncoupled tensor, is zero where the derivative is taken and moves with it.
    direction = np.zeros_like(base)
    direction[0, 1:3, 1:3, 0, 0] = 1.0
    direction[0, 1:3, 1:3, 0, 1] = direction[0, 1:3, 1:3, 1, 0] = 0.5
    direction[0, 1:3, 1:3, 0, 2] = direction[0, 1:3, 1:3, 2, 0] = 0.3
    return base, direction


def tensor(parameter, coupled, formulation=None):
    base, direction = _tensor(coupled)

    def f(b, p):
        cfg = dict(period=[600., 750.], wavelength=900., thickness=[170.], theta=0.3, phi=0.2,
                   psi=0.5, top=1.0, bot=2.25, truncation_order=[1, 1], answers='field')
        eps = _lift(b, base)
        if parameter == 'eps':
            eps = eps + p * _lift(b, direction)
        else:
            cfg[parameter] = p
        solver = _legacy_wavebit_rcwa(backend=b, structure=eps, **cfg)
        if formulation:
            solver.formulation = formulation
        # JAX traces a tensor raster only against a record of its phases.
        geometry = (_legacy_wavebit_rcwa(backend=b, structure=_lift(b, base), **cfg).freeze_geometry()
                    if b == 'jax' else None)
        return _read_out(b, solver.solve(geometry=geometry), 'r_s', 4, point=([10.], [20.], [60.]))
    return f


def excitation(b, p):
    cfg = dict(ONE_D, tetm='tm', layer_algorithm='SMM')
    eps = _lift(b, GRATING) + p * _lift(b, GRATING_DIRECTION)
    solver = _legacy_wavebit_rcwa(backend=b, structure=eps, excitation={0: 1.0, 1: 0.4j}, **cfg)
    return _read_out(b, solver.solve(), 'r_p', 3, point=([57.], [0.], [83.]))


def field_door(b, p):
    """``solve_field``: on JAX the jitted door, the solver crossing as a pytree."""
    cfg = {k: v for k, v in TWO_D.items() if k != 'answers'}
    solver = _legacy_wavebit_rcwa(backend=b, structure=_lift(b, RASTER), answers='field', **cfg)
    structure = _lift(b, RASTER) + p * _lift(b, PIXELS)
    if b == 'jax':
        result, cell = solver.solve_field(3, 2, 2, geometry=solver.freeze_geometry(),
                                          structure=structure)
    else:
        result, cell = solver.solve_field(3, 2, 2, structure=structure)
    values = [cell[0, 0, 1, 2, 1], cell[0, 1, 0, 1, 3]]          # (incidence, z, y, x, component)
    return _stack(b, [result.R.sum(), result.T.sum()] + [v for z in values for v in (z.real, z.imag)])


def legacy_field(b, p):
    """The solver's own ``reconstruct_field``, on a counted grid, in the period: its sampler
    spans the period, and PyTorch's ``linspace`` does not differentiate its endpoints, so the
    sampler scales a constant grid instead wherever the period carries a derivative."""
    cfg = dict(TWO_D, period=[p, _lift(b, 0.77)])
    solver = _legacy_wavebit_rcwa(backend=b, structure=_lift(b, RASTER), **cfg)
    result = solver.solve()
    cell = solver.reconstruct_field(3, 2, 2)
    values = [cell[0, 1, 1, 2, 1], cell[0, 0, 0, 0, 3], cell[0, 1, 1, 0, 4]]
    return _stack(b, [result.R.sum(), result.T.sum()] + [v for z in values for v in (z.real, z.imag)])


NVF_ROD = np.ones((1, 8, 8), dtype=complex)
NVF_ROD[0, 2:6, 2:5] = 9.0 + 0.2j


def nvf_cell(parameter):
    """An NVF raster whose cell moves, so its ``'distance'`` normal field moves with it: the
    period of a 0.77 x 0.61 cell (not square, where the nearest interface pixel is a tie and the
    loss has a kink), or the angle of a 0.77 x 0.77 one."""
    def f(b, p):
        cfg = {k: v for k, v in TWO_D.items() if k != 'period'}
        if parameter == 'period':
            cfg['period'] = [p, _lift(b, 0.61)]
        else:
            cfg['lattice'] = wavebit.Lattice.oblique(0.77, 0.77, p)
        solver = _legacy_wavebit_rcwa(backend=b, structure=_lift(b, NVF_ROD), factorization='nvf', **cfg)
        return _read_out(b, solver.solve(), 'r_s', 6, point=([0.1], [-0.05], [0.07]))
    return f


def _xp(b):
    if b == 'numpy':
        return np
    if b == 'jax':
        import jax.numpy as jnp
        return jnp
    import torch
    return torch


CELL_DISC = dict(type='circle', cx=-30.0, cy=10.0, lx=120.0, material=6.0 + 0.1j)
CELL_SCENE = dict(wavelength=860.0, truncation_order=[1, 1], top=1.0, bot=1.5, theta=0.1, phi=0.2,
                  psi=0.6, thickness=[220.0], factorization='laurent', answers='field')


def spelled_cell(spelling):
    """A disc on a cell given three ways, the parameter a length or an angle inside it:
    ``hexagonal(a)`` (a traced length beside a plain angle), and a skewed basis built from its
    angle as a bare ``(2, 2)`` array or as nested lists. Laurent, so that no normal field steps as
    the cell moves."""
    def f(b, p):
        xp = _xp(b)
        if spelling == 'hexagonal':
            lattice = wavebit.Lattice.hexagonal(p)
        else:
            rows = [[_lift(b, 400.0), 400.0 * xp.cos(p)], [_lift(b, 0.0), 400.0 * xp.sin(p)]]
            if spelling == 'lists':
                lattice = rows
            elif b == 'numpy':
                lattice = np.array(rows, dtype=float)
            else:
                lattice = xp.stack([xp.stack(row) for row in rows])
        solver = _legacy_wavebit_rcwa(backend=b, lattice=lattice, structure=[dict(background=1.2, shapes=[CELL_DISC])],
                              **CELL_SCENE)
        return _read_out(b, solver.solve(), 'r_s', 2, point=([30.], [10.], [120.]))
    return f


def two_shapes(b, p):
    """Two shapes in a layer, the circle's material the parameter: the projector cache keys a
    layer of several shapes by their materials (the seam between them is drawn by contrast), and
    a dual read as a plain value there was served the plain solve's build."""
    pieces = [dict(type='circle', cx=-0.15, cy=0.0, lx=0.2, material=p + 0.3j),
              dict(type='rectangle', cx=0.15, cy=0.05, lx=0.2, ly=0.25, angle=0.3, material=4.0)]
    solver = _legacy_wavebit_rcwa(backend=b, structure=[dict(background=1.44, shapes=pieces)], **SHAPES)
    return _read_out(b, solver.solve(), 't_p', 4, point=([0.12], [0.03], [0.11]))


#: name: (read-out, point, difference step). Each moves a layer's modal operator, so each went
#: through the modal rule that forward mode could not take before 2026-10-02.
PATHS = {
    '1d TE, eps, ETM': (one_d('eps', 'ETM', 'r_s', tetm='te'), 0.0, 1e-4),
    '1d TM, theta, SMM': (one_d('theta', 'SMM', 'r_p', tetm='tm'), 0.2, 1e-5),
    '1d TE, wavelength, SMM': (one_d('wavelength', 'SMM', 'r_s', tetm='te'), 900.0, 1e-2),
    'conical, eps, ETM': (one_d('eps', 'ETM', 'r_s', psi=0.37, phi=0.3), 0.0, 1e-4),
    'conical, mu, SMM': (one_d('mu', 'SMM', 'r_s', psi=0.37, phi=0.3), 0.0, 1e-4),
    '2d Li, pixel eps, integrated': (raster('eps', fourier_coefficients='integrated'), 0.0, 1e-4),
    '2d Li, period, counted field': (raster('period'), 0.77, 1e-6),
    '2d Li, period, solver field': (legacy_field, 0.77, 1e-6),
    '2d Li, theta': (raster('theta'), 0.2, 1e-5),
    '2d Laurent, pixel eps': (raster('eps', factorization='laurent'), 0.0, 1e-4),
    '2d NVF raster, material': (raster('eps', SQUARE, factorization='nvf'), 0.0, 1e-4),
    'scattering matrix, pixel eps': (scattering_matrix, 0.0, 1e-4),
    'shape, rectangle lx': (shape('lx'), 0.385, 1e-5),
    'shape, rectangle cx': (shape('cx'), 0.02, 1e-5),
    'shape, Im eps': (shape('Im eps'), 0.3, 1e-4),
    'shape, magnetic mu': (shape('mu'), 1.2, 1e-4),
    'shape, oblique cell, eps': (oblique_cell('eps'), 5.76, 1e-4),
    'shape, Laurent lx': (shape('lx', factorization='laurent'), 0.385, 1e-5),
    'shape, disc lx': (shape('lx', 'circle'), 0.385, 1e-5),
    'shape, disc wavelength': (shape('wavelength', 'circle'), 0.777, 1e-6),
    'shape, two shapes, material': (two_shapes, 12.25, 1e-5),
    'cell, hexagonal length': (spelled_cell('hexagonal'), 400.0, 1e-3),
    'cell, a bare array, angle': (spelled_cell('array'), OBLIQUE, 1e-6),
    'cell, nested lists, angle': (spelled_cell('lists'), OBLIQUE, 1e-6),
    '2d NVF raster, period': (nvf_cell('period'), 0.77, 1e-6),
    '2d NVF raster, cell angle': (nvf_cell('angle'), OBLIQUE, 1e-6),
    'gold disks, floor binds, lx': (gold_disks('lx'), 140.0, 1e-3),
    'gold disks, floor binds, Re eps': (gold_disks('Re eps'), GOLD.real, 1e-4),
    'tensor Li, zero coupling moved': (tensor('eps', coupled=False), 0.0, 1e-4),
    'tensor first-order, eps': (tensor('eps', coupled=True, formulation='first_order'), 0.0, 1e-4),
    'tensor first-order, theta': (tensor('theta', coupled=True, formulation='first_order'), 0.3,
                                  1e-5),
    'excitation, eps': (excitation, 0.0, 1e-4),
    'solve_field door, pixel eps': (field_door, 0.0, 1e-4),
}


#: Paths whose parameter moves no modal operator. PyTorch's ``forward_ad`` was already right on
#: them; JAX's ``jacfwd`` was not, a thickness reaching the modal exponential's
#: ``asarray(..., device=)``, which JAX cannot batch.
JAX_ONLY = {
    '1d TM, thickness, SMM': (one_d('thickness', 'SMM', 'r_p', tetm='tm'), 170.0, 1e-2),
    '2d Li, thickness, ETM': (raster('thickness', layer_algorithm='ETM'), 0.2, 1e-5),
}
PATHS_AND_JAX_ONLY = {**PATHS, **JAX_ONLY}


@functools.lru_cache(maxsize=None)
def _difference(name):
    """The read-out and its central difference on NumPy, Richardson-extrapolated: O(h^4)."""
    f, at, h = PATHS_AND_JAX_ONLY[name]
    near = (f('numpy', at + h) - f('numpy', at - h)) / (2 * h)
    far = (f('numpy', at + 2 * h) - f('numpy', at - 2 * h)) / (4 * h)
    return f('numpy', at), (4 * near - far) / 3


def _weights(n):
    return np.random.default_rng(5).standard_normal(n)


def _close(got, want, rel):
    got, want = np.asarray(got, float), np.asarray(want, float)
    return float(np.max(np.abs(got - want))) <= rel * float(np.max(np.abs(want)))


# ------------------------------------------------------------------------------- first order

#: Where ``jax.jacfwd`` -- ``vmap`` over the tangent -- is checked as well as ``jax.jvp``: the
#: modal rule, the NVF floor's rule, and the thicknesses that ``vmap`` could not batch.
JACFWD = {'1d TE, eps, ETM', 'gold disks, floor binds, Re eps', '2d Li, thickness, ETM',
          '1d TM, thickness, SMM'}

#: The JAX share of the paths. The tangent rule is one body for all of them, so JAX runs one
#: path per formulation and rule, the doors only JAX has (the jitted ``solve_field``) and the
#: cell. Eager JAX costs 5-35 s a path here, where PyTorch's dispatches, which a dual can slip
#: past one by one, are what the full list is for.
JAX_PATHS = sorted(JACFWD | {'conical, mu, SMM', 'tensor first-order, eps', 'shape, disc lx',
                             'solve_field door, pixel eps', 'scattering matrix, pixel eps',
                             '2d NVF raster, material', '2d NVF raster, period',
                             '2d NVF raster, cell angle', 'cell, nested lists, angle'})


@pytest.mark.parametrize('name', JAX_PATHS)
@pytest.mark.shard_unit('difference')
def test_jax_forward_mode_is_the_reverse_mode_derivative(name):
    """``jax.jvp`` (and on :data:`JACFWD`, ``jax.jacfwd``) against ``jax.grad`` and a central
    difference."""
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    import jax.numpy as jnp
    f, at, _ = PATHS_AND_JAX_ONLY[name]
    value, finite = _difference(name)
    w = _weights(value.size)
    x = jnp.asarray(at, dtype=jnp.float64)
    primal, tangent = jax.jvp(lambda p: f('jax', p), (x,), (jnp.asarray(1.0),))
    gradient = jax.grad(lambda p: jnp.sum(jnp.asarray(w) * f('jax', p)))(x)
    np.testing.assert_allclose(np.asarray(primal), value, rtol=0, atol=1e-10)
    if name in JACFWD:
        assert _close(jax.jacfwd(lambda p: f('jax', p))(x), tangent, 1e-13)
    assert float(w @ np.asarray(tangent)) == pytest.approx(float(gradient), rel=1e-11)
    assert _close(tangent, finite, 1e-6), (np.asarray(tangent), finite)


@pytest.mark.parametrize('parameter, at', [('angle', OBLIQUE), ('eps', 5.76)])
def test_jax_forward_mode_on_an_oblique_cell_under_jit(parameter, at):
    """An oblique cell under ``jax.jit`` with a frozen geometry, the way an optimizer runs one:
    an eager JAX solve whose oblique lattice is a JAX array takes minutes (43ae852 too), and
    the traced normal field is not NumPy's to the last digits (5e-6 in this read-out), so the
    difference is taken on the same compiled function. The cell angle is checked against reverse
    mode only: it moves every sample of the NVF normal field, so the loss steps where a sample
    crosses an interface, and a central difference straddling those steps read an order of
    magnitude off both modes here (``docs/guide/common/autodiff.md`` on the NVF steps)."""
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    import jax.numpy as jnp
    geometry = _oblique_solver('jax').freeze_geometry()
    f = jax.jit(lambda p: oblique_cell(parameter)('jax', p, geometry))
    x = jnp.asarray(at, dtype=jnp.float64)
    w = _weights(np.asarray(f(x)).size)
    _, tangent = jax.jit(lambda p: jax.jvp(f, (p,), (jnp.ones_like(p),)))(x)
    jacobian = jax.jit(jax.jacfwd(f))(x)
    gradient = jax.jit(jax.grad(lambda p: jnp.sum(jnp.asarray(w) * f(p))))(x)
    assert _close(jacobian, tangent, 1e-13)
    assert float(w @ np.asarray(tangent)) == pytest.approx(float(gradient), rel=1e-11)
    if parameter == 'eps':
        h = 1e-4
        near = (np.asarray(f(x + h)) - np.asarray(f(x - h))) / (2 * h)
        far = (np.asarray(f(x + 2 * h)) - np.asarray(f(x - 2 * h))) / (4 * h)
        assert _close(tangent, (4 * near - far) / 3, 1e-6), np.asarray(tangent)


def test_torch_forward_ad_on_an_oblique_cell_angle():
    """The cell angle on PyTorch, against reverse mode only (see the JAX test above for why not
    a central difference). The basis is a dual here, and the cell's lengths and raster Jacobian
    are read from it through ``constant_array``, which a dual now refuses as reverse mode's
    ``requires_grad`` always was."""
    torch = pytest.importorskip('torch')
    from torch.autograd import forward_ad
    f = oblique_cell('angle')
    w = _weights(f('numpy', OBLIQUE).size)
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(torch.tensor(OBLIQUE, dtype=torch.float64),
                                    torch.tensor(1.0, dtype=torch.float64))
        tangent = forward_ad.unpack_dual(f('torch', dual)).tangent.numpy()
    x = torch.tensor(OBLIQUE, dtype=torch.float64, requires_grad=True)
    (torch.as_tensor(w) * f('torch', x)).sum().backward()
    assert float(w @ tangent) == pytest.approx(float(x.grad), rel=1e-11)


@pytest.mark.parametrize('name', list(PATHS))
@pytest.mark.shard_unit('difference')
def test_torch_forward_ad_is_the_reverse_mode_derivative(name):
    """``torch.autograd.forward_ad`` against ``backward()`` and a central difference.

    A plain solve at the same point runs first, as an optimizer's line search would: a shape
    layer's projector is cached by the values it was built at, and a dual read as a plain value
    there was served the plain solve's projector, its geometry tangent dropped.
    """
    torch = pytest.importorskip('torch')
    from torch.autograd import forward_ad
    f, at, _ = PATHS[name]
    value, finite = _difference(name)
    w = _weights(value.size)
    plain = f('torch', torch.tensor(at, dtype=torch.float64))
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(torch.tensor(at, dtype=torch.float64),
                                    torch.tensor(1.0, dtype=torch.float64))
        primal, tangent = forward_ad.unpack_dual(f('torch', dual))
        tangent = np.zeros(value.shape) if tangent is None else tangent.numpy()
    x = torch.tensor(at, dtype=torch.float64, requires_grad=True)
    (torch.as_tensor(w) * f('torch', x)).sum().backward()
    np.testing.assert_allclose(plain.numpy(), value, rtol=0, atol=1e-10)
    np.testing.assert_allclose(primal.detach().numpy(), value, rtol=0, atol=1e-10)
    assert float(w @ tangent) == pytest.approx(float(x.grad), rel=1e-11)
    assert _close(tangent, finite, 1e-6), (tangent, finite)


def test_a_sampled_tensor_shape_refuses_a_forward_mode_geometry_derivative():
    """Sampled tensor shapes paint a fixed mask, so a geometry derivative through them would be a
    plausible zero; they refuse one by name. The check asked ``requires_grad``, which a
    ``forward_ad`` dual does not set, so a dual diameter went through."""
    torch = pytest.importorskip('torch')
    from torch.autograd import forward_ad
    with forward_ad.dual_level():
        lx = forward_ad.make_dual(torch.tensor(270.0, dtype=torch.float64),
                                  torch.tensor(1.0, dtype=torch.float64))
        solver = _legacy_wavebit_rcwa(
            backend='torch', wavelength=700., period=(600., 600.), thickness=[170.],
            structure=[dict(background=1., shapes=[dict(type='circle', lx=lx,
                                                          material=wavebit.Material([2., 3., 4.]))])],
            truncation_order=(1, 1), shape_nvf_grid_size=24, phi=.1, fourier_coefficients='sampled')
        with pytest.raises(ValueError, match='geometry derivatives'):
            solver.solve()


RAYLEIGH = dict(period=[1000.], wavelength=500., truncation_order=(3, 0), top=1.21, bot=1.69,
                theta=0., phi=0., psi=np.pi / 2, thickness=[300., 200.])
RAYLEIGH_PILLAR = np.ones((1, 20))
RAYLEIGH_PILLAR[:, 6:14] = 12.25


def _rayleigh(b, eps):
    """``tests/rcwa/test_rayleigh_gradient_retry.py``'s stack: the spacer's order 3 grazes at
    ``eps = 2.25``, where the solve retries detuned (``auto_detune``)."""
    xp = _xp(b)
    structure = xp.stack([xp.ones((1, 20), dtype=xp.float64) * eps, xp.asarray(RAYLEIGH_PILLAR)])
    return xp.real(_legacy_wavebit_rcwa(backend=b, structure=structure, **RAYLEIGH).solve().R).sum()


def test_forward_mode_reaches_the_detuned_retry():
    """At a Rayleigh point the solve retries detuned, by a wider step where the scene carries a
    derivative, so that the derivative survives the detune. Whether it does is
    ``engine._holds_autodiff``, a walk of the solver's conditions through ``carries_autodiff`` --
    blind to a dual until 2026-10-02. And once it saw one, ``forward_tangent`` raised
    ``TypeError`` for the ``torch.dtype`` the solver also holds, so a plain solve here raised
    whenever a dual level was open; it returns the plain value now."""
    torch = pytest.importorskip('torch')
    from torch.autograd import forward_ad
    h = 1e-4
    finite = (float(_rayleigh('numpy', 2.25 + h)) - float(_rayleigh('numpy', 2.25 - h))) / (2 * h)
    plain = float(_rayleigh('torch', torch.tensor(2.25, dtype=torch.float64)))
    with forward_ad.dual_level():
        assert float(_rayleigh('torch', torch.tensor(2.25, dtype=torch.float64))) == plain
        dual = forward_ad.make_dual(torch.tensor(2.25, dtype=torch.float64),
                                    torch.tensor(1.0, dtype=torch.float64))
        tangent = float(forward_ad.unpack_dual(_rayleigh('torch', dual)).tangent)
    x = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
    reverse = float(torch.autograd.grad(_rayleigh('torch', x), x)[0])
    # The detuned solve sits next to a grazing order and loses digits: the two modes agree to
    # 2e-10 here, not to the 1e-11 of an ordinary point.
    assert tangent == pytest.approx(reverse, rel=1e-8)
    assert tangent == pytest.approx(finite, rel=2e-4, abs=2e-6)


# ------------------------------------------------------------------------------- second order

def _second_order_scene(b, p, geometry=None):
    """A small raster scene for the refusals: one pixel's permittivity moves the modal operator."""
    cfg = dict(TWO_D, truncation_order=[1, 1])
    solver = _legacy_wavebit_rcwa(backend=b, structure=_lift(b, RASTER), **cfg)
    result = solver.solve(structure=_lift(b, RASTER) + p * _lift(b, PIXELS), geometry=geometry)
    return (result.R.sum() + 0.3 * result.T.sum()).real


JAX_ROUTES = {
    'hessian': lambda jax, g: jax.hessian(g),
    # `jax.checkpoint` stages the tangent rule and drops what it computes and does not read
    # before the outer derivative arrives: the refusal has to be read to survive.
    'hessian(checkpoint)': lambda jax, g: jax.hessian(jax.checkpoint(g)),
    'grad(grad(checkpoint))': lambda jax, g: jax.grad(jax.grad(jax.checkpoint(g))),
    'jacfwd(jacrev)': lambda jax, g: jax.jacfwd(jax.jacrev(g)),
    'jvp(grad)': lambda jax, g: (lambda x: jax.jvp(jax.grad(g), (x,), (x * 0 + 1,))[1]),
    'grad(jvp)': lambda jax, g: jax.grad(lambda x: jax.jvp(g, (x,), (x * 0 + 1,))[1]),
    'jvp(jvp)': lambda jax, g: (lambda x: jax.jvp(lambda y: jax.jvp(g, (y,), (y * 0 + 1,))[1],
                                                  (x,), (x * 0 + 1,))[1]),
}


@pytest.mark.parametrize('jit', [False, True], ids=['eager', 'jit'])
@pytest.mark.parametrize('route', list(JAX_ROUTES))
def test_jax_forward_mode_stops_at_first_order(route, jit):
    """The modal rule is first order: its tangent holds the basis fixed, so a derivative of that
    tangent drops the eigenvectors' variation. ``jax.jvp`` takes the first derivative, which is
    reverse mode's, and every JAX route to a second -- each order of forward and reverse --
    reaches ``first_order_only`` with the operator carrying an outer tangent and raises
    ``NotImplementedError``. Before 2026-10-02 ``jax.jvp`` itself raised the ``custom_vjp``
    ``TypeError``, as did the second-order routes that end in forward mode; the reverse-mode
    ``custom_vjp`` already refused ``hessian``, ``jacfwd(jacrev)`` and ``jvp(grad)`` by name.
    Without the guard the ``custom_jvp`` returns a partial Hessian with no error -- and with a
    guard whose result goes unread, under ``jax.checkpoint``: its partial evaluation removed the
    call as dead code, and ``hessian(checkpoint)`` returned 1.0385e-5 here where a difference of
    ``jax.grad`` reads -6.352e-6."""
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    import jax.numpy as jnp
    geometry = None
    if jit:
        geometry = _legacy_wavebit_rcwa(backend='jax', structure=jnp.asarray(RASTER),
                                **dict(TWO_D, truncation_order=[1, 1])).freeze_geometry()
    scene = lambda p: _second_order_scene('jax', p, geometry)
    first = lambda p: jax.jvp(scene, (p,), (jnp.ones_like(p),))[1]
    reverse = jax.grad(scene)
    second = JAX_ROUTES[route](jax, scene)
    if jit:
        first, reverse, second = jax.jit(first), jax.jit(reverse), jax.jit(second)
    x = jnp.asarray(0.0)
    assert float(first(x)) == pytest.approx(float(reverse(x)), rel=1e-11)
    with pytest.raises(NotImplementedError, match='first order'):
        second(x)


@pytest.mark.parametrize('route', ['forward_ad over backward', 'backward over forward_ad'])
def test_a_torch_mixed_second_derivative_is_refused_by_name(route):
    """PyTorch's two mixed orders: a ``forward_ad`` tangent carried through a backward pass, and
    a backward pass through a ``forward_ad`` tangent. Each differentiates the modal rule's
    first-order map and would return a partial Hessian; each raises ``RuntimeError`` naming the
    first-order limit. (``create_graph=True``, the pure reverse order, raised before.)"""
    torch = pytest.importorskip('torch')
    from torch.autograd import forward_ad
    x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    with forward_ad.dual_level(), pytest.raises(RuntimeError, match='first order'):
        dual = forward_ad.make_dual(x, torch.tensor(1.0, dtype=torch.float64))
        if route == 'forward_ad over backward':
            gradient, = torch.autograd.grad(_second_order_scene('torch', dual), x)
            forward_ad.unpack_dual(gradient)
        else:
            tangent = forward_ad.unpack_dual(_second_order_scene('torch', dual)).tangent
            torch.autograd.grad(tangent, x)


def test_a_hessian_that_moves_no_modal_operator_is_exact():
    """The refusal is keyed on the modal operator moving under both derivatives, not on the
    order asked for: a thickness moves no layer's operator, only its exponential, so its Hessian
    is exact -- ``jax.hessian`` and PyTorch's ``create_graph=True`` alike, against a central
    difference of the first derivative. ``jax.hessian`` raised here before 2026-10-02, on the
    modal exponential's ``asarray(..., device=)``.

    PyTorch's ``forward_ad`` over a backward pass, and a backward pass over ``forward_ad``, too:
    this scene's ETM matching solves through ``backend.linear_solve``, whose derivative PyTorch
    can take again. Through ``torch.linalg.solve`` itself the first read 1.284 against 11.18 here
    (its backward reuses non-differentiable LU factors with grad mode off; torch 2.13)."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    _legacy_wavebit_enable_x64()
    import jax.numpy as jnp
    f, at, h = JAX_ONLY['2d Li, thickness, ETM']

    def scalar(b, d):
        v = f(b, d)
        return v[0] + 0.3 * v[1] + 0.2 * v[2]

    first = jax.grad(lambda d: scalar('jax', d))
    finite = (float(first(jnp.asarray(at + h))) - float(first(jnp.asarray(at - h)))) / (2 * h)
    assert float(jax.hessian(lambda d: scalar('jax', d))(jnp.asarray(at))) == pytest.approx(finite, rel=1e-6)

    x = torch.tensor(at, dtype=torch.float64, requires_grad=True)
    g, = torch.autograd.grad(scalar('torch', x), x, create_graph=True)
    assert float(torch.autograd.grad(g, x)[0]) == pytest.approx(finite, rel=1e-6)

    from torch.autograd import forward_ad
    one = torch.tensor(1.0, dtype=torch.float64)
    with forward_ad.dual_level():
        x = forward_ad.make_dual(torch.tensor(at, dtype=torch.float64), one).requires_grad_()
        g, = torch.autograd.grad(scalar('torch', x), x)
        assert float(forward_ad.unpack_dual(g).tangent) == pytest.approx(finite, rel=1e-6)
    x = torch.tensor(at, dtype=torch.float64, requires_grad=True)
    with forward_ad.dual_level():
        tangent = forward_ad.unpack_dual(scalar('torch', forward_ad.make_dual(x, one))).tangent
    assert float(torch.autograd.grad(tangent, x)[0]) == pytest.approx(finite, rel=1e-6)


def _mixed_scene(b, eps, d, algorithm):
    cfg = dict(TWO_D, truncation_order=[1, 1], thickness=[d], layer_algorithm=algorithm)
    solver = _legacy_wavebit_rcwa(backend=b, structure=_lift(b, RASTER) + eps * _lift(b, PIXELS), **cfg)
    result = solver.solve()
    return (result.R.sum() + 0.3 * result.T.sum()).real


def test_a_mixed_second_derivative_differentiates_the_rule_once():
    """``d2/(d eps d thickness)``: the thickness moves no modal operator, so the rule is
    differentiated once, in the permittivity, and the mixed derivative is exact -- on JAX in
    each order (``jvp`` of ``grad`` either way round, ``grad`` of ``grad``) and on PyTorch through
    ``create_graph=True`` taken in the thickness first, against a central difference of the first
    derivative. Not refused, because the number is right; the refusal is for the operator moving
    under both derivatives. (``create_graph=True`` in the permittivity first runs the rule's
    backward with grad mode on, which is the refusal's signal, and is refused.)

    PyTorch's two mixed orders are exact too, on ETM and SMM alike: the matching solves through
    ``backend.linear_solve``. Through ``torch.linalg.solve`` itself, ``forward_ad`` in the
    thickness over a backward pass in the permittivity read 0.004266 on this scene's ETM matching
    and a backward pass over it 0.004953, where the difference reads 0.002997 -- 42% and 65% off,
    with no error.
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    from torch.autograd import forward_ad
    _legacy_wavebit_enable_x64()
    import jax.numpy as jnp
    eps0, d0, h = 0.0, 0.2, 1e-5

    d_eps = jax.grad(lambda e, d: _mixed_scene('jax', e, d, 'ETM'))
    finite = (float(d_eps(jnp.asarray(eps0), jnp.asarray(d0 + h)))
              - float(d_eps(jnp.asarray(eps0), jnp.asarray(d0 - h)))) / (2 * h)
    routes = {
        'jvp in d of grad in eps':
            lambda: jax.jvp(lambda d: d_eps(jnp.asarray(eps0), d), (jnp.asarray(d0),), (jnp.asarray(1.0),))[1],
        'jvp in eps of grad in d':
            lambda: jax.jvp(lambda e: jax.grad(lambda d: _mixed_scene('jax', e, d, 'ETM'))(jnp.asarray(d0)),
                            (jnp.asarray(eps0),), (jnp.asarray(1.0),))[1],
        'grad in eps of grad in d':
            lambda: jax.grad(lambda e: jax.grad(lambda d: _mixed_scene('jax', e, d, 'ETM'))(jnp.asarray(d0)))(
                jnp.asarray(eps0)),
    }
    for name, run in routes.items():
        assert float(run()) == pytest.approx(finite, rel=1e-6), name

    e = torch.tensor(eps0, dtype=torch.float64, requires_grad=True)
    d = torch.tensor(d0, dtype=torch.float64, requires_grad=True)
    g, = torch.autograd.grad(_mixed_scene('torch', e, d, 'ETM'), d, create_graph=True)
    assert float(torch.autograd.grad(g, e)[0]) == pytest.approx(finite, rel=1e-6)

    for algorithm in ('ETM', 'SMM'):
        def d_eps_torch(d):
            e = torch.tensor(eps0, dtype=torch.float64, requires_grad=True)
            return float(torch.autograd.grad(
                _mixed_scene('torch', e, torch.tensor(d, dtype=torch.float64), algorithm), e)[0])

        finite = (d_eps_torch(d0 + h) - d_eps_torch(d0 - h)) / (2 * h)
        one = torch.tensor(1.0, dtype=torch.float64)
        e = torch.tensor(eps0, dtype=torch.float64, requires_grad=True)
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(torch.tensor(d0, dtype=torch.float64), one)
            gradient, = torch.autograd.grad(_mixed_scene('torch', e, dual, algorithm), e)
            forward_over_backward = float(forward_ad.unpack_dual(gradient).tangent)
        e = torch.tensor(eps0, dtype=torch.float64, requires_grad=True)
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(torch.tensor(d0, dtype=torch.float64), one)
            tangent = forward_ad.unpack_dual(_mixed_scene('torch', e, dual, algorithm)).tangent
        backward_over_forward = float(torch.autograd.grad(tangent, e)[0])
        assert forward_over_backward == pytest.approx(finite, rel=1e-6), algorithm
        assert backward_over_forward == pytest.approx(finite, rel=1e-6), algorithm


# ------------------------------------------------------------------------------- the other rules

@pytest.mark.shard_unit('difference')
def test_native_j1_forward_mode():
    """PyTorch's own ``bessel_j1`` (``set_j1_implementation('native')``, and the default below
    float64) is wrapped in an ``autograd.Function`` for the derivative PyTorch does not ship. It
    had a ``backward`` only, so a ``forward_ad`` dual through a disc raised PyTorch's "jvp is
    not implemented"; it now carries the same ``J0 - J1/x`` as a ``jvp``, and a ``setup_context``
    so that ``torch.func`` takes it too."""
    torch = pytest.importorskip('torch')
    from torch.autograd import forward_ad
    from wavebit._common.backend import j1, set_j1_implementation
    previous = set_j1_implementation('native')
    try:
        f, at, _ = PATHS['shape, disc lx']
        value, finite = _difference('shape, disc lx')
        w = _weights(value.size)
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(torch.tensor(at, dtype=torch.float64),
                                        torch.tensor(1.0, dtype=torch.float64))
            tangent = forward_ad.unpack_dual(f('torch', dual)).tangent.numpy()
        x = torch.tensor(at, dtype=torch.float64, requires_grad=True)
        (torch.as_tensor(w) * f('torch', x)).sum().backward()
        assert float(w @ tangent) == pytest.approx(float(x.grad), rel=1e-11)
        # The native J1 is accurate to ~4e-7 in float64 (docs/validation/rcwa/bessel-j1.md), so the scene's
        # derivative is too.
        assert _close(tangent, finite, 1e-5), (tangent, finite)

        points = torch.tensor([0.0, 0.3, 2.0, 9.0], dtype=torch.float64)
        exact = (torch.special.bessel_j0(points[1:]) - torch.special.bessel_j1(points[1:]) / points[1:])
        _, through_func = torch.func.jvp(j1, (points,), (torch.ones_like(points),))
        np.testing.assert_allclose(through_func[1:].numpy(), exact.numpy(), rtol=1e-15)
        assert float(through_func[0]) == 0.5
        np.testing.assert_allclose(torch.func.vmap(torch.func.grad(j1))(points).numpy(),
                                   through_func.numpy(), rtol=1e-15)
    finally:
        set_j1_implementation(previous)


@pytest.mark.parametrize('transform', ['jvp', 'grad', 'jacrev', 'jacfwd'])
def test_torch_func_is_refused_by_name(transform):
    """``torch.func``'s grad and jvp transforms through ``solve()`` are refused by name, pointing
    at ``torch.autograd``. The modal ``autograd.Function`` has a functorch ``vmap`` rule now; the
    refusal is its first-order guard's: that reads ``torch.is_grad_enabled()``, which
    ``torch.func.grad`` sets on every backward pass, and its ``jvp`` has none, so a ``torch.func``
    grad or jvp level is turned away at the door (``backend.functorch_differentiates``). Before
    2026-10-02 every ``torch.func`` transform of a solve died on PyTorch's own ``setup_context``
    error at the eigensolve, a function no derivative ever flows through. Every composition with
    ``vmap`` is refused as well: ``test_torch_vmap_autograd.py``."""
    torch = pytest.importorskip('torch')
    x = torch.tensor(0.0, dtype=torch.float64)
    scene = lambda p: _second_order_scene('torch', p)
    run = {'jvp': lambda: torch.func.jvp(scene, (x,), (torch.ones_like(x),)),
           'grad': lambda: torch.func.grad(scene)(x),
           'jacrev': lambda: torch.func.jacrev(scene)(x),
           'jacfwd': lambda: torch.func.jacfwd(scene)(x)}[transform]
    with pytest.raises(NotImplementedError, match='torch.func'):
        run()
