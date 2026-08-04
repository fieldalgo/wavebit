"""Every design parameter, every path, against a finite difference.

The suite had gradient tests for the raster ``ucell`` and for a few shape parameters, and they
all passed while ``wavelength`` was 38-85% wrong, ``n_top`` 49% wrong, and ``theta`` / ``phi`` /
``psi`` / ``period`` / ``n_index`` / ``n_base`` / ``angle`` carried no gradient at all on
PyTorch. Nothing was checking the scalars.

Two failure modes to keep apart, because only one of them announces itself:

* a severed graph raises on PyTorch (``element 0 of tensors does not require grad``) and
  silently returns zero on JAX;
* a *partially* severed graph returns a plausible number. ``wavelength`` used to come back as
  exactly the ``k0 = 2*pi/lambda`` part of its own derivative, with the ``kx``/``ky`` path cut
  — no error, no zero, just wrong by a factor.

So each case compares against a central difference rather than a stored value: the claim is
that the derivative is the real one. Any parameter whose finite difference is itself zero is
reported as such rather than counted as a pass.
"""
import importlib

import numpy as np
import pytest

import fieldwave

RASTER = np.ones((1, 8, 8))
RASTER[0, 2:6, 2:6] = 3.0

BASE = dict(period=[0.77, 0.77], thickness=[0.2], theta=0.2, phi=0.3, psi=0.4,
            wavelength=0.777, n_top=1.0, n_bot=1.5, fto=[3, 3])

SHAPE = dict(type='rectangle', cx=0.385, cy=0.385, lx=0.385, ly=0.231, angle=0.4, n_index=3.5)
ANALYTIC_BASE = dict(period=[0.77, 0.77], thickness=[0.25], theta=0.35, phi=0.4, psi=0.0,
                     wavelength=0.777, n_top=1.0, n_bot=1.5, fto=[4, 4])

def _frozen_raster():
    global FROZEN
    if FROZEN is None:
        FROZEN = fieldwave.rcwa(backend=0, ucell=RASTER, fourier_type=0, **BASE).freeze_geometry()
    return FROZEN


FROZEN = None

VECTOR_BASE = dict(period=[300.0, 300.0], thickness=[205.0], theta=0.1, fto=[3, 3],
                   wavelength=900.0, n_top=1.0, n_bot=1.0, pol=0.5)


def _backend(name):
    pytest.importorskip(name)
    if name == 'jax':
        importlib.import_module('fieldwave.on_jax')       # enables x64
        import jax.numpy as jnp
        return 1, (lambda v: jnp.asarray(v, dtype=jnp.float64))
    import torch
    return 2, (lambda v: torch.tensor(v, dtype=torch.float64))


def _derivative(backend, objective, value, rel_step=1e-6):
    """(autograd, central difference) of ``objective`` at ``value``.

    ``rel_step`` matters more than it looks on the analytic path: the normal field is sampled
    on a grid of ``period / analytic_normal_grid``, so the objective carries a sawtooth of that
    period. A difference step that straddles one of those steps reports the jump, not the
    derivative — at ``cx = period/2``, which sits exactly on a grid line, ``h = 1e-6`` gives
    3.8 where the derivative is 3.2e-3. Stay well inside one cell.
    """
    if backend == 2:
        import torch
        p = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        objective(p).backward()
        auto = float(p.grad)
    else:
        import jax
        auto = float(jax.grad(lambda v: objective(v).real)(value))

    h = rel_step * max(abs(value), 1.0)
    scalar = (lambda v: float(objective(_backend_scalar(backend)(v))))
    return auto, (scalar(value + h) - scalar(value - h)) / (2 * h)


def _backend_scalar(backend):
    if backend == 2:
        import torch
        return lambda v: torch.tensor(v, dtype=torch.float64)
    import jax.numpy as jnp
    return lambda v: jnp.asarray(v, dtype=jnp.float64)


def _total(res):
    import torch
    d = res.de_ti
    return torch.sum(d) if hasattr(d, 'backward') or hasattr(d, 'grad_fn') else d.sum()


# --------------------------------------------------------------------------- raster (topology)

RASTER_SCALARS = ['theta', 'phi', 'psi', 'wavelength', 'n_top', 'n_bot', 'thickness', 'period']


@pytest.mark.parametrize('name', ['torch', 'jax'])
@pytest.mark.parametrize('param', RASTER_SCALARS)
def test_raster_scalar_gradients(param, name):
    backend, cast = _backend(name)

    def objective(p):
        cfg = dict(BASE)
        if param == 'thickness':
            cfg['thickness'] = [p]
        elif param == 'period':
            cfg['period'] = [p, cast(0.77)]
        else:
            cfg[param] = p
        solver = fieldwave.rcwa(backend=backend, ucell=cast(RASTER), fourier_type=0, **cfg)
        # the same-material auto-detection reads concrete values, so under a JAX trace the
        # discrete geometry has to be frozen first (this is the documented contract).
        return solver.conv_solve(geometry=_frozen_raster() if backend == 1 else None).de_ti.sum()

    start = {'theta': 0.2, 'phi': 0.3, 'psi': 0.4, 'wavelength': 0.777, 'n_top': 1.0,
             'n_bot': 1.5, 'thickness': 0.2, 'period': 0.77}[param]
    auto, fd = _derivative(backend, objective, start)
    assert abs(fd) > 1e-8, f'the objective does not depend on {param}; test is vacuous'
    assert abs(auto - fd) / abs(fd) < 1e-4, (param, auto, fd)


@pytest.mark.parametrize('name', ['torch', 'jax'])
def test_raster_per_cell_gradient_reaches_every_cell(name):
    """DFS is the raster transform topology optimization needs: one gradient per cell.

    CFS legitimately gives 9 of 64 — it compresses the raster into blocks and the other cells
    genuinely stop being variables — which is why the choice is not an implementation detail.
    """
    backend, cast = _backend(name)

    def cells(fourier_type, geometry=None):
        if backend == 2:
            import torch
            uc = torch.tensor(RASTER, dtype=torch.float64, requires_grad=True)
            res = fieldwave.rcwa(backend=2, ucell=uc, fourier_type=fourier_type,
                                 **BASE).conv_solve(geometry=geometry)
            torch.sum(res.de_ti).backward()
            return int((uc.grad.abs() > 1e-14).sum())
        import jax, jax.numpy as jnp
        g = jax.grad(lambda u: jnp.sum(fieldwave.rcwa(backend=1, ucell=u,
                     fourier_type=fourier_type, **BASE).conv_solve(geometry=geometry).de_ti).real)
        return int(np.sum(np.abs(np.asarray(g(jnp.asarray(RASTER)))) > 1e-14))

    assert cells(0) == RASTER.size

    solver = fieldwave.rcwa(backend=backend, ucell=cast(RASTER), fourier_type=1, **BASE)
    assert cells(1, solver.freeze_geometry()) < RASTER.size


# ------------------------------------------------------------------------- analytic (shape/OCD)

ANALYTIC_PARAMS = ['lx', 'ly', 'angle', 'n_index', 'n_base',
                   'theta', 'wavelength', 'thickness']


@pytest.mark.parametrize('name', ['torch', 'jax'])
@pytest.mark.parametrize('param', ANALYTIC_PARAMS)
def test_analytic_shape_gradients(param, name):
    backend, cast = _backend(name)

    def objective(p):
        shape, cfg, n_base = dict(SHAPE), dict(ANALYTIC_BASE), 1.0
        if param == 'n_base':
            n_base = p
        elif param == 'thickness':
            cfg['thickness'] = [p]
        elif param in shape:
            shape[param] = p
        else:
            cfg[param] = p
        res = fieldwave.rcwa(backend=backend, ucell=[dict(n_base=n_base, shapes=[shape])],
                             **cfg).conv_solve()
        return res.de_ri.sum()

    start = dict(SHAPE, n_base=1.0, theta=0.35, wavelength=0.777, thickness=0.25)[param]
    auto, fd = _derivative(backend, objective, float(start), rel_step=1e-8)
    assert abs(fd) > 1e-8, f'the objective does not depend on {param}; test is vacuous'
    assert abs(auto - fd) / abs(fd) < 1e-3, (param, auto, fd)


# ------------------------------------------------------------------------------ vector (OCD)

@pytest.mark.parametrize('name', ['torch', 'jax'])
@pytest.mark.parametrize('param,index,start', [('lx', 3, 80.0), ('ly', 4, 60.0),
                                               ('angle', 6, 0.4)])
def test_vector_edge_gradients(param, index, start, name):
    """OCD differentiates the edges: CD, position, sidewall angle."""
    backend, cast = _backend(name)

    def objective(p):
        rect = ['rectangle', cast(150.0), cast(150.0), cast(80.0), cast(60.0), 2.0, cast(0.4)]
        rect[index] = p
        res = fieldwave.rcwa(backend=backend, ucell=[[3.5, [rect]]], **VECTOR_BASE).conv_solve()
        return res.de_ti.sum()

    auto, fd = _derivative(backend, objective, start)
    assert abs(fd) > 1e-8, f'the objective does not depend on {param}; test is vacuous'
    assert abs(auto - fd) / abs(fd) < 1e-3, (param, auto, fd)


# ------------------------------------------------------------------- shape gradients near zero

def test_rectangle_margin_is_symmetric_about_zero():
    """A rectangle at -1e-6 rad must take the same branch as one at +1e-6.

    ``angle`` is wrapped into [0, 2*pi) before the near-alignment test, so a small negative
    angle arrives as ``2*pi - eps``. The targets used to stop at ``3*pi/2``, so the negative
    side fell through to the rotated decomposition while the positive side got the exact
    single block, and the two disagreed by 2e-7 in efficiency — a step sitting exactly on the
    unrotated case, which is where any sidewall-angle fit starts.
    """
    m = fieldwave.rcwa(backend=0, period=[300.0, 300.0]).modeler
    blocks = [m.rectangle(150.0, 150.0, 80.0, 60.0, 2.0, a) for a in (-1e-6, 0.0, 1e-6)]
    assert [len(b) for b in blocks] == [1, 1, 1], [len(b) for b in blocks]
    for side in (0, 2):
        assert np.allclose(np.asarray(blocks[side][0][:2], dtype=float),
                           np.asarray(blocks[1][0][:2], dtype=float))


def test_analytic_angle_is_smooth_through_normal():
    """The analytic path is the one that can fit a sidewall angle near normal; the vector one cannot.

    Rotating a rectangle moves its Fourier coefficients analytically, so the efficiency is
    smooth in ``angle`` and the two one-sided slopes at zero must agree. They do, on the
    analytic path. The block decomposition instead misses the shape by a staircase of area
    ``~|angle|`` that ``n_split`` does not remove (measured stable from 4 to 64), which shows
    up as a kink about twenty times the true slope. A central difference hides it — the
    ``|angle|`` term cancels — so this only bites an optimizer, which sees one side at a time.
    """
    base = dict(period=[300.0, 300.0], thickness=[205.0], fto=[5, 5], n_top=1.0, n_bot=1.0,
                wavelength=900.0, pol=0.5, theta=0.1, phi=0.3)

    def analytic(angle):
        shape = dict(type='rectangle', cx=150.0, cy=150.0, lx=80.0, ly=60.0,
                     angle=angle, n_index=2.0)
        res = fieldwave.rcwa(backend=0, ucell=[dict(n_base=3.5, shapes=[shape])],
                             **base).conv_solve()
        return float(np.sum(res.de_ti))

    h = 1e-5
    f0, fp, fm = analytic(0.0), analytic(h), analytic(-h)
    left, right = (f0 - fm) / h, (fp - f0) / h
    assert abs(right - left) < 0.02 * abs(left), (left, right)


def test_translation_leaves_only_sampling_noise_in_the_shape_gradient():
    """Sliding a single layer along x cannot change the efficiency, so d/dcx is exactly zero.

    What is left is the normal field's sampling grid, and its cell size sets the noise floor
    under every shape gradient on this path: 5e-4 at ``analytic_normal_grid=256``, which is
    the reason the default is no longer 256.
    """
    base = dict(period=[300.0, 300.0], thickness=[205.0], fto=[5, 5], n_top=1.0, n_bot=1.0,
                wavelength=900.0, pol=0.5, theta=0.1, phi=0.3)
    solver = fieldwave.rcwa(backend=0, period=[300.0, 300.0])
    cell = 300.0 / solver.analytic_normal_grid

    def total(cx):
        shape = dict(type='rectangle', cx=cx, cy=150.0, lx=80.0, ly=60.0,
                     angle=0.4, n_index=2.0)
        return float(np.sum(fieldwave.rcwa(backend=0, ucell=[dict(n_base=3.5, shapes=[shape])],
                                           **base).conv_solve().de_ti))

    values = [total(150.0 + f * cell) for f in (0.05, 0.35, 0.65, 0.95)]
    assert np.ptp(values) < 5e-5, np.ptp(values)
