"""``torch.compile`` runs wavebit eagerly, so a compiled caller gets eager torch's values and gradients.

Until 2026-10 nothing kept wavebit out of the compiled graph, and default-mode ``torch.compile``
returned wrong values *and* gradients with no error: a 1D TE loss of -4.43 against an eager
-0.895, its permittivity gradient -0.118 against 0.272, the 2D raster's field 2.9 off. Dynamo alone
and AOTAutograd alone were exact; Inductor (torch 2.13, CPU, complex128 with an input that requires
grad) emitted the in-place diagonal write of ``array_api_compat``'s ``eye`` after the kernels that
read it, so the modal exponential's identity read as zero. A refusal cannot be raised from inside
the trace -- Dynamo turns it into a graph break and reruns the frame natively -- so every public
callable is a door that ``torch.compiler.disable`` keeps out of the graph
(``backend.eager_under_torch_compile``).

What each test is there to catch:

* the doors cover the surface: a class or function added without the decorator, or a member kind
  the decorator skips (a property, a classmethod, a dunder), fails the walk, which builds a
  solver on every installed backend;
* the doors work: a door that calls through while Dynamo traces, picks its route by
  ``is_compiling()``, or skips the wrapper when already inside a door (a compiled ``forward``
  handed to ``solver.grad``), gives the wrong numbers the tripwire compares;
* the doors compose: ``torch.compile`` around ``torch.func`` second-order and vmapped routes
  is the eager derivative, against NumPy (two of the routes, ``jvp(jvp)`` and ``vmap(jacfwd)``,
  are exact eagerly because ``backend.linear_solve`` is);
* the doors keep their fast path: Dynamo is never imported by an eager solve and a backward;
* the refusals are loud and named: ``fullgraph=True``, either ``torch.export`` mode, and
  ``torch.jit.trace`` (refused at every door; its values are concrete, so nothing else noticed
  it, and it returned a gradient without wavebit's share -- of a solve, of a solved result's
  field at a traced height, of a cell built from a traced length);
* the reason still holds: the canary reproduces the Inductor defect on a five-line function, so a
  torch that fixes it is noticed where it was measured, macOS arm64 (the doors stay; the docs
  should say so).
"""

from wavebit.compat import legacy as _legacy_wavebit
import enum
import functools
import importlib.util
import inspect
import platform
import subprocess
import sys
import textwrap
import types

import numpy as np
import pytest

import wavebit as _public_wavebit
wavebit = _legacy_wavebit
from wavebit._common import backend as B

torch = pytest.importorskip('torch')

from tests.rcwa.test_forward_mode import GRATING, ONE_D, RASTER, PIXELS, SHAPES, TWO_D, _lift, _read_out, one_d, raster

pytestmark = pytest.mark.filterwarnings('ignore')


def _reset_dynamo():
    """Forget every compiled frame, importing Dynamo here rather than at module level: once
    ``torch._dynamo`` is in ``sys.modules`` every door in the process takes its
    ``torch.compiler.disable`` route, and a module-level import would put every test collected
    with this file on that slower route (the numbers are the same either way)."""
    import torch._dynamo
    torch._dynamo.reset()


def _f64(value, grad=False):
    """A float64 input: a bare ``torch.tensor(0.2)`` is float32, which is not the number written,
    and whose gradient comes back rounded to float32 (to 1e-8, under a 1e-12 comparison)."""
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


# ---------------------------------------------------------------------------- the surface
def _is_door(f):
    return hasattr(f, '_wavebit_eager_door')


def _public(name):
    return not name.startswith('_') or (name.startswith('__') and name.endswith('__'))


def _members(cls):
    for k in cls.__mro__:
        if not k.__module__.startswith('wavebit'):
            continue
        for name, m in vars(k).items():
            if not _public(name) or name in B._NOT_DOORS:
                continue
            if isinstance(m, property):
                yield from ((k, name, f) for f in (m.fget, m.fset, m.fdel) if f is not None)
            elif isinstance(m, (classmethod, staticmethod)):
                yield k, name, m.__func__
            elif isinstance(m, types.FunctionType):
                yield k, name, m


def _installed_backends():
    return ['numpy', 'torch'] + (['jax'] if importlib.util.find_spec('jax') else [])


def _reached_types():
    """The wavebit types a caller is handed by public calls, on every installed backend.

    A NumPy or JAX solve inside a compiled function needs its doors as much as a torch one: on
    torch 2.13 a NumPy solve there raised ``'ndarray' object has no attribute 'detach'`` without
    them. So each backend builds a 1D ETM solve, a conical SMM solve with its scattering matrix,
    a shape structure and an ``excitation=`` solve, and the walk follows, through lists, tuples
    and dicts: each public attribute; the children and aux data of a public ``tree_flatten()``
    (how a result hands over its field state); the nodes of a JAX pytree (how a JAX solver hands
    over its shape structure); and ``Lattice.drawn``'s return.
    """
    seen = set()
    tree_util = None

    def walk(obj, depth):
        t = type(obj)
        if t in (list, tuple, dict):           # a wavebit namedtuple is a type of its own
            for v in (obj.values() if t is dict else obj):
                walk(v, depth)
            return
        if not t.__module__.startswith('wavebit'):
            return
        seen.add(t)
        if depth == 0:
            return
        for name in dir(obj):
            if name.startswith('_'):
                continue
            try:
                v = getattr(obj, name)
            except Exception:          # noqa: BLE001 - a refusing property is checked as a member
                continue
            if inspect.ismethod(v):
                if name == 'tree_flatten':
                    walk(v(), depth - 1)
            elif not isinstance(v, (torch.Tensor, np.ndarray)):
                walk(v, depth - 1)
        if tree_util is not None:
            nodes = tree_util.tree_leaves(obj, is_leaf=lambda x: x is not obj and
                                          type(x).__module__.startswith('wavebit'))
            walk([x for x in nodes if x is not obj], depth - 1)

    for backend in _installed_backends():
        if backend == 'jax':
            from jax import tree_util
        lift = functools.partial(_lift, backend)
        for algorithm, extra in (('ETM', {}), ('SMM', dict(psi=0.37, phi=0.3))):
            solver = _legacy_wavebit.rcwa(backend=backend, structure=lift(GRATING),
                                  **dict(ONE_D, layer_algorithm=algorithm, **extra))
            result = solver.solve()
            for obj in (solver, result, result.order(1), solver.lattice, solver.freeze_geometry(),
                        solver.eps_conv_matrices(), solver.lattice.drawn(lift(GRATING))):
                walk(obj, 2)
            if algorithm == 'SMM':
                walk(solver.scattering, 2)
        piece = dict(type='rectangle', cx=0.02, cy=-0.01, lx=0.385, ly=0.231, angle=0.4,
                     material=12.25 + 0.3j)
        walk(_legacy_wavebit.rcwa(backend=backend, structure=[dict(background=1.44, shapes=[piece])],
                          **SHAPES), 2)
        walk(_legacy_wavebit.rcwa(backend=backend, structure=lift(GRATING), wavelength=900.,
                          period=[1400.], thickness=[170.], truncation_order=(3, 0), top=1.0,
                          bot=2.25, excitation={'top': wavebit.PlaneWave(
                              theta=0.2, phi=0.6, polarization=(0.6, 0.8j))}).solve(), 2)
    return {t for t in seen if not issubclass(t, (enum.Enum, BaseException))}


def test_every_public_callable_is_a_compile_door():
    classes, missing, seen_modules = set(), [], set()

    def inspect_module(module):
        if module in seen_modules:
            return
        seen_modules.add(module)
        for name in module.__all__:
            obj = getattr(module, name)
            if isinstance(obj, types.ModuleType):
                inspect_module(obj)
            elif inspect.isclass(obj):
                classes.add(obj)
            elif callable(obj) and not _is_door(obj):
                missing.append(f'{module.__name__}.{name}')

    # Keep the full legacy reachable-object gate while inspecting real public
    # modules recursively. A namespace is not a function and is never exempt.
    inspect_module(_public_wavebit)
    inspect_module(_legacy_wavebit)
    assert {_public_wavebit.rcwa, _public_wavebit.fdtd, _public_wavebit.materials,
            _legacy_wavebit.fd} <= seen_modules
    assert {_public_wavebit.rcwa.Problem, _public_wavebit.fdtd.Problem,
            _public_wavebit.Material, _public_wavebit.DiffractionOrders} <= classes
    from wavebit.observations import DiffractionOrder
    classes.add(DiffractionOrder)
    classes |= _reached_types()
    for cls in classes:
        if issubclass(cls, (enum.Enum, BaseException)):
            continue
        missing += [f'{k.__module__}.{k.__qualname__}.{n}' for k, n, f in _members(cls) if not _is_door(f)]
    assert not missing, f'{len(missing)} public callables torch.compile would trace into: {missing[:10]}'


def test_a_door_keeps_the_signature_and_the_docstring():
    assert str(inspect.signature(_legacy_wavebit.rcwa)).startswith("(backend='numpy', *args, stack=None")
    assert wavebit.rcwa.__doc__ and 'decide backend' in wavebit.rcwa.__doc__
    assert wavebit.Lattice.hexagonal.__func__.__name__ == 'hexagonal'
    assert inspect.getdoc(wavebit.ScatteringMatrix.cascade) == inspect.getdoc(
        wavebit.ScatteringMatrix.cascade._wavebit_eager_door)


# ---------------------------------------------------------------------------- the tripwire
def _outside_1d(**over):
    return _legacy_wavebit.rcwa(backend='torch', structure=_lift('torch', GRATING), **dict(ONE_D, **over))


CASES = {
    # name: (read-out of a torch scalar p, the point)
    '1d TE eps ETM, solver inside': (lambda p: one_d('eps', 'ETM', 'r_s', tetm='te')('torch', p), 0.0),
    'conical eps SMM, solver inside': (lambda p: one_d('eps', 'SMM', 'r_s', psi=0.37, phi=0.3)('torch', p), 0.0),
    '2d Li pixel with fields, solver inside': (lambda p: raster('eps', fourier_coefficients='integrated')('torch', p), 0.0),
}
_S_TM = _outside_1d(layer_algorithm='SMM', tetm='tm')


def _theta_setter(p):
    _S_TM.theta = p
    return _read_out('torch', _S_TM.solve(), 'r_p', 3, point=([57.], [0.], [83.]))


_S_2D = _legacy_wavebit.rcwa(backend='torch', structure=_lift('torch', RASTER), **TWO_D)


def _outside_2d(p):
    result = _S_2D.solve(structure=_lift('torch', RASTER) + p * _lift('torch', PIXELS))
    return _read_out('torch', result, 'r_s', 6, point=([0.1], [-0.05], [0.07]), counted=(3, 2, [0.07]))


CASES['1d TM theta setter, solver outside'] = (_theta_setter, 0.2)
CASES['2d Li pixel with fields, solver outside'] = (_outside_2d, 0.0)


@pytest.mark.parametrize('name', sorted(CASES))
def test_a_compiled_solve_is_the_eager_solve(name):
    read_out, at = CASES[name]
    x = _f64(at, grad=True)
    want = read_out(x)
    w = torch.as_tensor(np.random.default_rng(5).standard_normal(want.shape[0]))
    (gw,) = torch.autograd.grad((w * want).sum(), x)
    _reset_dynamo()
    x = _f64(at, grad=True)
    got = torch.compile(read_out)(x)
    (gg,) = torch.autograd.grad((w * got).sum(), x)
    np.testing.assert_allclose(got.detach().numpy(), want.detach().numpy(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(float(gg), float(gw), rtol=1e-12)


def test_a_compiled_forward_called_from_inside_a_door_stays_eager():
    """``solver.grad`` is a door that calls the caller's ``forward``. Compiled, that callable sets
    Dynamo's frame hook back inside the door, so the ``solve`` door it calls must take its
    ``disable`` route again. A door that skips the wrapper when already inside one let Dynamo
    trace the solve: the permittivity gradient read up to 0.8165 off (eager's largest entry is
    0.1539) with the loss right to every printed digit, and nothing raised."""
    def loss_fn(result):
        return result.R.sum() + 0.3 * result.T.sum() + result.r_s.reshape(-1)[3].real

    eager = _outside_1d(layer_algorithm='ETM', tetm='te')
    want = eager.grad(['structure'], eager.solve, loss_fn)['structure'].detach().numpy()
    _reset_dynamo()
    solver = _outside_1d(layer_algorithm='ETM', tetm='te')
    got = solver.grad(['structure'], torch.compile(lambda: solver.solve()), loss_fn)['structure']
    np.testing.assert_allclose(got.detach().numpy(), want, rtol=1e-12,
                               atol=1e-14 * float(np.abs(want).max()))


# ---------------------------------------------------------------------------- compile x torch.func
#: ``torch.compile`` around these routes returns the eager derivative, so it is exact where eager
#: is. Eagerly, ``jvp(jvp)`` and ``vmap(jacfwd)`` are exact because every solve goes through
#: ``backend.linear_solve``, which keeps ``torch.linalg.solve``'s derivatives exact when a forward
#: rule is differentiated again or runs under two ``vmap`` levels; through ``torch.linalg.solve``
#: itself they read 1.1801616522e-04 for 1.1001006765e-04, and lanes 196.4 and -2.08e5 for 0.01146
#: and 0.01160 (``tests/common/test_linear_solve_derivatives.py`` holds that rule to a closed form).
_LANES = (170.0, 170.7, 172.0)


@pytest.fixture(scope='module')
def thickness_scene():
    """1D TM SMM through a thickness: the scalar ``w . f`` in torch, and its NumPy oracles."""
    f, at, h = one_d('thickness', 'SMM', 'r_p', tetm='tm'), 170.0, 0.5
    w = np.random.default_rng(5).standard_normal(f('numpy', at).size)
    g_numpy = lambda p: float(np.dot(w, f('numpy', p)))
    second = lambda h: (g_numpy(at + h) - 2 * g_numpy(at) + g_numpy(at - h)) / h ** 2
    wt = torch.as_tensor(w)
    g = lambda p: (wt * f('torch', p)).sum()
    first = []
    for p in _LANES:
        x = _f64(p, grad=True)
        (d,) = torch.autograd.grad(g(x), x)
        first.append(float(d))
    # Richardson: (4 D(h) - D(2h)) / 3 removes the h^2 term of the central second difference.
    return g, at, (4 * second(h) - second(2 * h)) / 3, np.array(first)


def _jvp(q):
    return lambda x: torch.func.jvp(q, (x,), (torch.ones_like(x),))[1]


def _create_graph_twice(g):
    def d2(x):
        (d,) = torch.autograd.grad(g(x), x, create_graph=True)
        return torch.autograd.grad(d, x)[0]
    return d2


@pytest.mark.parametrize('route', ['jvp(jvp)', 'create_graph x2', 'vmap(jacfwd)'])
def test_compiling_around_a_second_order_route_is_exact(route, thickness_scene):
    """``torch.compile`` around ``torch.func`` and double backward: wavebit runs eagerly inside,
    so the compiled derivative is the eager one -- checked against NumPy, which shares no
    autodiff rule with torch (second difference; eager reverse mode per vmap lane)."""
    g, at, second, first = thickness_scene
    _reset_dynamo()
    if route == 'jvp(jvp)':
        got, want, rtol = torch.compile(_jvp(_jvp(g)))(_f64(at)), second, 1e-7
    elif route == 'create_graph x2':
        got, want, rtol = (torch.compile(_create_graph_twice(g))(_f64(at, grad=True)),
                           second, 1e-7)
    else:
        got, want, rtol = (torch.compile(torch.func.vmap(torch.func.jacfwd(g)))(_f64(_LANES)),
                           first, 1e-12)
    np.testing.assert_allclose(np.atleast_1d(got.detach().numpy()), np.atleast_1d(want), rtol=rtol)


#: Both tests below warm the doors with Dynamo already imported: only then does a door build its
#: cached ``disable`` wrapper, which is what carries the reason into Dynamo's message. Warmed
#: before Dynamo is loaded, a door takes its fast path and caches nothing; run alone (``-k``, a
#: node id, another xdist split) the message then named ``disable`` without the reason.


def test_fullgraph_names_the_reason():
    _reset_dynamo()
    _outside_1d(layer_algorithm='ETM', tetm='te').solve()          # warms the door
    with pytest.raises(Exception, match='wavebit runs eagerly under torch.compile') as info:
        torch.compile(lambda p: _S_TM.solve(theta=p).R.sum(), fullgraph=True)(_f64(0.2))
    assert type(info.value).__name__ == 'Unsupported'


@pytest.mark.parametrize('strict', [True, False])
def test_export_is_refused_by_name(strict):
    class Module(torch.nn.Module):
        def forward(self, p):
            return one_d('eps', 'ETM', 'r_s', tetm='te')('torch', p).sum()

    _reset_dynamo()
    Module()(_f64(0.0))       # warms the doors, so a strict export names the reason
    with pytest.raises(Exception, match='wavebit runs eagerly|cannot be exported'):
        torch.export.export(Module(), (_f64(0.0),), strict=strict)


def _field_across_an_interface():
    """A solved two-layer grating's field at a traced height: no solve, so ``_concrete`` is
    never reached."""
    two_layers = np.concatenate([GRATING, [[[2.0, 2.0, 1.0, 1.0]]]])
    result = _legacy_wavebit.rcwa(backend='torch', structure=_lift('torch', two_layers),
                          **dict(ONE_D, thickness=(170., 100.), layer_algorithm='SMM',
                                 tetm='te')).solve()
    return lambda z: result.reconstruct_field([57.], [0.], z.reshape(1))[0, 0, 0][1].real


def _cell_from_a_traced_length():
    return lambda a: sum((m * m).sum() for m in wavebit.meshgrid_xy(
        (3, 4), lattice=wavebit.Lattice.hexagonal(a), backend='torch'))


#: name: (the function to trace, built on demand; the example input). Each was silent until
#: 2026-10 -- TracerWarnings at most, then the example's branches replayed:
#:
#: * a solve (66 TracerWarnings) returned no ``grad_fn``, so a loss mixing it with another term
#:   lost wavebit's share of the gradient: 0.0 at p = 0, where eager torch reads 0.2721483665790;
#: * a solved two-layer grating's field, traced at z = 83 in the lower layer (z = 0 is the
#:   stack's bottom; the 100-thick layer spans 0..100) and replayed at z = 200 in the upper one,
#:   read d/dz -0.004063805947 where eager and a central difference read 0.003442581948, and
#:   at z = 400, outside the stack, 0.9016 where eager refuses;
#: * a hexagonal cell traced at a = 400 became a NumPy constant: replayed at 431 it read
#:   +2.922e5 for an eager +3.393e5, with no gradient for eager's 1574.35.
TRACED = {
    'a solve': (lambda: lambda p: one_d('eps', 'ETM', 'r_s', tetm='te')('torch', p).sum(), 0.0),
    'a solved field across a layer interface': (_field_across_an_interface, 83.0),
    'a cell built from a traced length': (_cell_from_a_traced_length, 400.0),
}


@pytest.mark.parametrize('name', sorted(TRACED))
def test_jit_trace_is_refused_by_name(name):
    """Every door refuses ``torch.jit.trace``. A check only where a solve reads a value at a
    decision (``_concrete``) missed the read-out of a solved result and a cell built from a
    traced length, which never reach it.
    Dynamo is imported first, so every door is on its ``disable`` route whichever tests ran
    before; the fast route is the subprocess test below."""
    _reset_dynamo()
    build, at = TRACED[name]
    traced = build()
    with pytest.raises(NotImplementedError, match=r'cannot be traced by torch\.jit\.trace'):
        torch.jit.trace(traced, (_f64(at),))
    assert not torch.jit.is_tracing()
    x = _f64(at, grad=True)               # the refusal leaves eager calls as they were
    (g,) = torch.autograd.grad(traced(x), x)
    assert bool(torch.isfinite(g))


# ---------------------------------------------------------------------------- the eager cost
def test_an_eager_solve_never_imports_dynamo():
    code = textwrap.dedent("""
        import sys, torch
        from wavebit.compat import legacy as wavebit
        x = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        eps = torch.tensor([[[1.0, 4.0, 4.0, 1.0]]], dtype=torch.complex128) + x
        r = wavebit.rcwa(backend='torch', structure=eps, period=(1400., 1500.), wavelength=900.,
                         thickness=(170.,), truncation_order=(3, 0), theta=0.2, top=1.0, bot=2.25,
                         answers='field').solve()
        (r.R.sum() + r.reconstruct_field([57.], [0.], [83.]).abs().sum()).backward()
        print('torch._dynamo' in sys.modules)
    """)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300)
    assert out.stdout.strip().splitlines()[-1] == 'False', out.stderr[-2000:]


def test_jit_trace_is_refused_while_dynamo_is_not_loaded():
    """A door's fast path, which an eager caller takes until something imports Dynamo, refuses
    ``torch.jit.trace`` too: here a solved result's field at a traced height."""
    code = textwrap.dedent("""
        import sys, torch
        from wavebit.compat import legacy as wavebit
        eps = torch.tensor([[[1.0, 4.0, 4.0, 1.0]], [[2.0, 2.0, 1.0, 1.0]]], dtype=torch.complex128)
        r = wavebit.rcwa(backend='torch', structure=eps, period=(1400., 1500.), wavelength=900.,
                         thickness=(170., 100.), truncation_order=(3, 0), theta=0.2, top=1.0,
                         bot=2.25, layer_algorithm='SMM', answers='field').solve()
        field = lambda z: r.reconstruct_field([57.], [0.], z.reshape(1)).real.sum()
        try:
            torch.jit.trace(field, (torch.tensor(83.0, dtype=torch.float64),))
        except NotImplementedError as e:
            print('refused' if 'cannot be traced by torch.jit.trace' in str(e) else 'unnamed')
        else:
            print('traced')
        print('torch._dynamo' in sys.modules)
    """)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300)
    assert out.stdout.split()[-2:] == ['refused', 'False'], (out.stdout, out.stderr[-2000:])


# ---------------------------------------------------------------------------- the canary
def _eye_then_read(q):
    """``array_api_compat``'s torch ``eye`` (a zeros tensor with its diagonal written in place),
    read twice: the shape of ``modal_gradient.exponential``'s identity."""
    eye = torch.zeros(5, 5, dtype=q.dtype)
    eye.diagonal(0).fill_(1)
    return eye * q.exp() + (q - eye * q)


#: Where the reorder was measured: macOS arm64, CPU, torch 2.13.0. Inductor generates its own C++
#: per platform, and nobody has run this on CI's Linux x86-64 with the CPU index's latest torch.
_CANARY_MEASURED_ON = ('darwin', 'arm64')


def test_canary_inductor_still_reorders_the_eye_write():
    """Fails when torch stops miscompiling this: then say so in docs/development/torch-compile.md §6. The doors
    stay -- they keep wavebit out of every future defect, at the cost measured there.

    Asserted only where the reorder was measured. Elsewhere a gap that does not reproduce skips
    with what it read, for the docs, rather than failing a run for a reason nobody has measured.
    """
    q = torch.complex(torch.randn(5, 5, generator=torch.Generator().manual_seed(0), dtype=torch.float64),
                      torch.randn(5, 5, generator=torch.Generator().manual_seed(1), dtype=torch.float64))
    q.requires_grad_()
    _reset_dynamo()
    gap = float((torch.compile(_eye_then_read)(q) - _eye_then_read(q)).abs().max())
    here = (sys.platform, platform.machine())
    if gap <= 1e-3 and here != _CANARY_MEASURED_ON:
        pytest.skip(f'the eye-write reorder does not reproduce on {here} with torch '
                    f'{torch.__version__} (gap {gap:.1e}); it was measured on {_CANARY_MEASURED_ON} '
                    f'with torch 2.13.0 -- record this platform in docs/development/torch-compile.md §6')
    assert gap > 1e-3, f'torch {torch.__version__} compiles the eye write in order now (gap {gap:.1e})'
