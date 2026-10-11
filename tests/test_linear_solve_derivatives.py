"""``backend.linear_solve``: every derivative of a linear solve exact on PyTorch, at every order.

PyTorch 2.13 gets two families of derivatives of ``torch.linalg.solve`` wrong with no error
(``backend``'s note above :func:`~wavebit._core.backend.linear_solve`): a forward-mode rule
differentiated again (``jvp(jvp)``, ``grad(jvp)``, ``jacfwd(jacfwd)``, a backward pass over a
``forward_ad`` tangent, a tangent through a backward pass without ``create_graph``), because the
rule reuses non-differentiable LU factors -- as does the backward pass with grad mode off, which
is also ``torch.func.hessian``'s under ``torch.no_grad()`` -- and a forward-mode rule under two
``vmap`` levels (``vmap(jacfwd)``, a first derivative, and ``vmap(hessian)``), because
``lu_solve`` batches its factors and its right side wrongly there. Measured before the fix: the 1D
TM grating's thickness second derivative by ``jvp(jvp)`` read 1.1801616522e-04 where it is
1.1001006774e-04, a 2D ETM raster's by the same route 760.746214 where it is 812.595272, and
``vmap(jacfwd)`` of the 1D TM grating 197.0 for 0.01147 on its second lane.

The unit tests here hold the solve to a closed form, not a difference: with ``A(x) = A0 + x A1``
and ``B`` a cubic, ``A X^(k) = B^(k) - k A1 X^(k-1)`` gives every derivative exactly. The scene
tests hold each site that solves through it to PyTorch's ``create_graph=True``, which re-solves
and is exact (and to ``jax.hessian`` on one scene, which shares no rule with it).
"""
import ast
import functools
import itertools
import pathlib

import numpy as np
import pytest

import wavebit

torch = pytest.importorskip('torch')
from torch.autograd import forward_ad as fa  # noqa: E402

from wavebit._core import backend as WB  # noqa: E402

pytestmark = pytest.mark.filterwarnings('ignore')
fn = torch.func
XS = (0.1, 0.3, 0.5)


@pytest.fixture(autouse=True)
def _float64():
    """Every bare ``torch.tensor(x)`` here is a float64 parameter, as the closed form's ``x``."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _xp():
    import array_api_compat
    return array_api_compat.array_namespace(torch.zeros(1))


def _problem(dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, dtype=dtype, generator=g)  # noqa: E731
    return dict(A0=r(4, 4) + 3 * torch.eye(4, dtype=dtype), A1=r(4, 4), B=[r(4, 3) for _ in range(4)],
                W=torch.randn(4, 3, dtype=torch.complex128, generator=g))


def _operands(P, x):
    b = P['B']
    return P['A0'] + x * P['A1'], b[0] + x * b[1] + x ** 2 * b[2] + x ** 3 * b[3]


def _exact(P, x, order):
    """``sum(Re(W X^(order)))`` from ``A X^(k) = B^(k) - k A1 X^(k-1)``."""
    A, B = _operands(P, torch.tensor(x))
    b = P['B']
    dB = [B, b[1] + 2 * x * b[2] + 3 * x ** 2 * b[3], 2 * b[2] + 6 * x * b[3], 6 * b[3]]
    X = [torch.linalg.solve(A, B)]
    for k in range(1, order + 1):
        X.append(torch.linalg.solve(A, dB[k] - k * P['A1'] @ X[k - 1]))
    return float((P['W'] * X[order]).real.sum())


def _readout(P, solve, constant_matrix=False):
    """``sum(Re(W X))`` as a function of ``x``; with ``constant_matrix``, ``A = A0`` is captured
    outside every transform and only the right side moves."""
    if constant_matrix:
        return lambda x: (P['W'] * solve(P['A0'], _operands(P, x)[1])).real.sum()
    return lambda x: (P['W'] * solve(*_operands(P, x))).real.sum()


def _forward(h):
    return lambda x: fn.jvp(h, (x,), (torch.ones_like(x),))[1]


def _lanes(t):
    return lambda: torch.stack([t(torch.tensor(x)) for x in XS])


def _vmapped(t):
    return lambda: fn.vmap(t)(torch.tensor(XS))


def _per_lane(run):
    return lambda: torch.stack([torch.as_tensor(run(x)) for x in XS])


def _create_graph(g, order):
    def run(at):
        x = torch.tensor(at, requires_grad=True)
        d = g(x)
        for k in range(order):
            (d,) = torch.autograd.grad(d, x, create_graph=k < order - 1)
        return d
    return run


def _forward_over_backward(g, create_graph):
    def run(at):
        with fa.dual_level():
            x = fa.make_dual(torch.tensor(at), torch.tensor(1.0)).requires_grad_()
            (d,) = torch.autograd.grad(g(x), x, create_graph=create_graph)
            return fa.unpack_dual(d).tangent
    return run


def _backward_over_forward(g):
    def run(at):
        x = torch.tensor(at, requires_grad=True)
        with fa.dual_level():
            t = fa.unpack_dual(g(fa.make_dual(x, torch.tensor(1.0)))).tangent
        return torch.autograd.grad(t, x)[0]
    return run


def _backward_twice_over_forward(g):
    def run(at):
        x = torch.tensor(at, requires_grad=True)
        with fa.dual_level():
            t = fa.unpack_dual(g(fa.make_dual(x, torch.tensor(1.0)))).tangent
        (d,) = torch.autograd.grad(t, x, create_graph=True)
        return torch.autograd.grad(d, x)[0]
    return run


def _grad_mode_off(run, mode=torch.no_grad):
    """``run`` under ``torch.no_grad()`` (or ``mode``): ``torch.func.vjp``, ``jacrev`` and
    ``hessian`` then run their backward with grad mode off, which PyTorch's rule takes as leave
    to reuse the factors."""
    def wrapped():
        with mode():
            return run()
    return wrapped


def _vjp(h):
    """``torch.func.vjp`` of ``h``, its ``vjp_fn`` called in the grad mode ``vjp`` was entered with."""
    return lambda x: fn.vjp(h, x)[1](torch.ones(()))[0]


def _dual_inside(h):
    """``h``'s derivative by a ``forward_ad`` dual made inside the function -- inside whatever
    ``torch.func`` level calls it, whose wrapper then holds the tangent."""
    def tangent(x):
        with fa.dual_level():
            return fa.unpack_dual(h(fa.make_dual(x, torch.ones_like(x)))).tangent
    return tangent


def _forward_over(t):
    """A plain ``forward_ad`` dual outside ``t``, lane by lane."""
    def run():
        out = []
        for x in XS:
            with fa.dual_level():
                out.append(fa.unpack_dual(t(fa.make_dual(torch.tensor(x), torch.tensor(1.0)))).tangent)
        return torch.stack(out)
    return run


F, R = _forward, fn.grad
#: name: (order, route builder, whether the solve stays PyTorch's own there -- and so must be
#: right natively on torch 2.13; every other route is rewritten, and wrong natively but three)
ROUTES = {
    'jvp(jvp)': (2, lambda g: _lanes(F(F(g))), False),
    'jacfwd(jacfwd)': (2, lambda g: _lanes(fn.jacfwd(fn.jacfwd(g))), False),
    'jacrev(jacfwd)': (2, lambda g: _lanes(fn.jacrev(fn.jacfwd(g))), False),
    'grad(jvp)': (2, lambda g: _lanes(R(F(g))), False),
    'backward over forward_ad': (2, lambda g: _per_lane(_backward_over_forward(g)), False),
    'forward_ad over backward': (2, lambda g: _per_lane(_forward_over_backward(g, False)), False),
    'vmap(jvp(jvp))': (2, lambda g: _vmapped(F(F(g))), False),
    'vmap(jacfwd)': (1, lambda g: _vmapped(fn.jacfwd(g)), False),
    'vmap(hessian)': (2, lambda g: _vmapped(fn.hessian(g)), False),
    'vmap(jacfwd(jacfwd))': (2, lambda g: _vmapped(fn.jacfwd(fn.jacfwd(g))), False),
    'jvp(jvp(grad))': (3, lambda g: _lanes(F(F(R(g)))), False),
    'grad(jvp(grad))': (3, lambda g: _lanes(R(F(R(g)))), False),
    'jvp(jvp(jvp))': (3, lambda g: _lanes(F(F(F(g)))), False),
    'backward x2 over forward_ad': (3, lambda g: _per_lane(_backward_twice_over_forward(g)), False),
    # Both plain levels outside a torch.func level: the dual is read beneath its wrapper
    'backward over forward_ad over grad': (3, lambda g: _per_lane(_backward_over_forward(R(g))), False),
    'forward_ad over backward over grad': (3, lambda g: _per_lane(_forward_over_backward(R(g), False)),
                                           False),
    # A grad level entered with grad mode off runs its backward so (vjp, jacrev, hessian): its
    # rule reuses the factors under a forward level outside it, or batches them under two vmaps
    'no_grad: hessian': (2, lambda g: _grad_mode_off(_lanes(fn.hessian(g))), False),
    'no_grad: jacfwd(jacrev)': (2, lambda g: _grad_mode_off(_lanes(fn.jacfwd(fn.jacrev(g)))), False),
    'no_grad: jvp(jacrev)': (2, lambda g: _grad_mode_off(_lanes(F(fn.jacrev(g)))), False),
    'no_grad: jvp(vjp)': (2, lambda g: _grad_mode_off(_lanes(F(_vjp(g)))), False),
    'no_grad: forward_ad over jacrev': (2, lambda g: _grad_mode_off(_forward_over(fn.jacrev(g))), False),
    'no_grad: vmap(jacrev)': (1, lambda g: _grad_mode_off(_vmapped(fn.jacrev(g))), False),
    'set_grad_enabled(False): hessian': (
        2, lambda g: _grad_mode_off(_lanes(fn.hessian(g)), lambda: torch.set_grad_enabled(False)), False),
    'inference_mode: jvp(vjp)': (2, lambda g: _grad_mode_off(_lanes(F(_vjp(g))), torch.inference_mode), False),
    # PyTorch is right here -- torch.func.grad always re-solves, and a vjp_fn called inside its
    # own vmap sees no second one -- but neither level can be told from jacrev's, so each is
    # rewritten too, moving in the last digits
    'no_grad: jvp(grad)': (2, lambda g: _grad_mode_off(_lanes(F(R(g)))), False),
    'no_grad: vmap(grad)': (1, lambda g: _grad_mode_off(_vmapped(R(g))), False),
    'no_grad: vmap(vjp)': (1, lambda g: _grad_mode_off(_vmapped(_vjp(g))), False),
    # A forward_ad dual made inside a grad level: the level holds its tangent and differentiates
    # the forward rule
    'grad(dual-inside)': (2, lambda g: _lanes(R(_dual_inside(g))), False),
    'jacrev(dual-inside)': (2, lambda g: _lanes(fn.jacrev(_dual_inside(g))), False),
    'vjp(dual-inside)': (2, lambda g: _lanes(_vjp(_dual_inside(g))), False),
    'grad(grad(dual-inside))': (3, lambda g: _lanes(R(R(_dual_inside(g)))), False),
    # ... the tangent beneath another grad level's wrapper, or a vmap's: read at its own level
    'grad(dual-inside(grad))': (3, lambda g: _lanes(R(_dual_inside(R(g)))), False),
    'grad(dual-inside(vmap))': (2, lambda g: _lanes(R(_dual_inside(lambda x: fn.vmap(g)(x[None])[0]))), False),
    'vmap(grad(dual-inside))': (2, lambda g: _vmapped(R(_dual_inside(g))), False),
    'vmap(jacrev(dual-inside))': (2, lambda g: _vmapped(fn.jacrev(_dual_inside(g))), False),
    # PyTorch's own rule is exact here, and the solve must stay its own (test_native_where_exact)
    'hessian': (2, lambda g: _lanes(fn.hessian(g)), True),
    'jacrev(jacrev)': (2, lambda g: _lanes(fn.jacrev(fn.jacrev(g))), True),
    'jvp(grad)': (2, lambda g: _lanes(F(R(g))), True),
    'create_graph x2': (2, lambda g: _per_lane(_create_graph(g, 2)), True),
    'create_graph x3': (3, lambda g: _per_lane(_create_graph(g, 3)), True),
    'jvp(grad(grad))': (3, lambda g: _lanes(F(R(R(g)))), True),
    'vmap(jvp)': (1, lambda g: _vmapped(F(g)), True),
    'vmap(grad)': (1, lambda g: _vmapped(R(g)), True),
    'jacfwd(vmap)': (1, lambda g: (lambda: torch.diagonal(fn.jacfwd(fn.vmap(g))(torch.tensor(XS)))), True),
    'vmap(jacrev(jacrev))': (2, lambda g: _vmapped(fn.jacrev(fn.jacrev(g))), True),
    # The dual outside the grad level sits on the base, and the level's backward re-solves
    'forward_ad over grad': (2, lambda g: _forward_over(R(g)), True),
    # Under torch.no_grad() the outermost level's backward reuses the factors, but nothing
    # differentiates it; the inner jacrev is called inside the outer one, with grad mode on
    'no_grad: jacrev': (1, lambda g: _grad_mode_off(_lanes(fn.jacrev(g))), True),
    'no_grad: jacrev(jacrev)': (2, lambda g: _grad_mode_off(_lanes(fn.jacrev(fn.jacrev(g)))), True),
    'no_grad: grad(grad)': (2, lambda g: _grad_mode_off(_lanes(R(R(g)))), True),
}


def _relative(got, want):
    return float((got.detach().reshape(-1) - want).abs().max() / want.abs().max())


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128], ids=['real', 'complex'])
@pytest.mark.parametrize('route', list(ROUTES))
def test_every_route_matches_the_closed_form(route, dtype):
    """Each route through ``linear_solve`` gives the exact derivative on every lane, where
    ``torch.linalg.solve`` is 0.34 to 1.7e3 off on the routes it rewrites (first, second and
    third order, with and without an outer ``vmap``, under ``torch.no_grad()``, and with a
    ``forward_ad`` dual made inside a grad level) -- all but ``no_grad: jvp(grad)``,
    ``no_grad: vmap(grad)`` and ``no_grad: vmap(vjp)``, which are rewritten though PyTorch is right
    there."""
    order, build, native_exact = ROUTES[route]
    P = _problem(dtype)
    want = torch.tensor([_exact(P, x, order) for x in XS])
    got = build(_readout(P, lambda A, B: WB.linear_solve(_xp(), A, B)))()
    assert _relative(got, want) < 1e-10, route
    native = _relative(build(_readout(P, torch.linalg.solve))(), want)
    if native_exact:
        assert native < 1e-10, route


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128], ids=['real', 'complex'])
def test_two_parameters_one_per_operand(dtype):
    """``A`` moved by ``x`` alone and ``B`` by ``y`` alone: the forward level inside moves only
    the right side, so ``A`` carries the outer level and not the inner one -- PyTorch's rule drops
    the whole ``-A^-1 A1 X_y`` there (relative error 1.0). And under a ``vmap`` over ``x``, the
    first derivative in ``y`` by ``jacfwd``: the factors batched outside, the right side inside.
    So too a grad level in ``y`` alone under that ``vmap`` whose rule reuses the factors -- the
    ``jacrev`` of a ``forward_ad`` tangent taken inside it, or ``jacrev`` under
    ``torch.no_grad()``: ``A`` carries no derivative there, but its backward runs ``lu_solve`` on
    factors batched by the outer ``vmap`` and a right side batched by ``jacrev``'s own."""
    P = _problem(dtype)
    solve = lambda A, B: WB.linear_solve(_xp(), A, B)  # noqa: E731
    h = lambda A, y: (P['W'] * solve(A, P['B'][0] + y * P['B'][1])).real.sum()  # noqa: E731
    y0 = torch.tensor(0.3)
    b = P['B']
    h2 = lambda A, y: (P['W'] * solve(A, b[0] + y * b[1] + y ** 2 * b[2])).real.sum()  # noqa: E731

    def under_x(inner):
        def lane(x):
            A = P['A0'] + x * P['A1']        # batched by the vmap, constant in y
            return inner(lambda y: h2(A, y))(y0)
        return _vmapped(lane)
    at_x = lambda right: torch.tensor([float((P['W'] * torch.linalg.solve(  # noqa: E731
        P['A0'] + x * P['A1'], right)).real.sum()) for x in XS])
    assert _relative(under_x(lambda f: fn.jacrev(_dual_inside(f)))(), at_x(2 * b[2])) < 1e-10
    assert _relative(_grad_mode_off(under_x(fn.jacrev))(), at_x(b[1] + 2 * y0 * b[2])) < 1e-10

    def g(x, inner):
        # The matrix is built before the inner transform is entered, so it carries the outer
        # level only (built inside, functorch would lift it to the inner level as well).
        A = P['A0'] + x * P['A1']
        return inner(lambda y: h(A, y))(y0)
    mixed = torch.tensor([float((P['W'] * torch.linalg.solve(
        P['A0'] + x * P['A1'], -P['A1'] @ torch.linalg.solve(P['A0'] + x * P['A1'], P['B'][1]))).real.sum())
        for x in XS])
    first = torch.tensor([float((P['W'] * torch.linalg.solve(P['A0'] + x * P['A1'], P['B'][1])).real.sum())
                          for x in XS])
    inner_moves_b = _lanes(F(lambda x: g(x, F)))()
    reverse_of_it = _lanes(R(lambda x: g(x, F)))()
    batched_first = _vmapped(lambda x: g(x, fn.jacfwd))()
    assert _relative(inner_moves_b, mixed) < 1e-10
    assert _relative(reverse_of_it, mixed) < 1e-10
    assert _relative(batched_first, first) < 1e-10


#: One ``torch.func`` transform of a scalar function of a scalar: ``V`` maps over two distinct
#: lanes, ``x`` and ``x + SHIFT``, weighted 1 and 1/2, so that a wrong second lane shows.
SHIFT = 0.1
NEST_OPS = {
    'F': lambda h: (lambda x: fn.jvp(h, (x,), (torch.ones_like(x),))[1]),
    'R': fn.grad, 'J': fn.jacfwd, 'K': fn.jacrev,
    'V': lambda h: (lambda x: (fn.vmap(h)(torch.stack([x, x + SHIFT])) * torch.tensor([1.0, 0.5])).sum()),
}
#: The plain levels outside every transform: none, a backward pass, a ``forward_ad`` dual, a
#: backward pass over a dual's tangent, a dual's tangent of a backward pass (no ``create_graph``).
PLAIN = ('', 'b', 'd', 'bd', 'db')


def _nest_cell(P, nest, plain, outer_vmap):
    """``nest`` (outermost first) of ``linear_solve``'s read-out, under ``plain``, on every lane of
    ``XS`` -- by an outermost ``vmap`` or one lane at a time -- and the closed form for it."""
    h = _readout(P, lambda A, B: WB.linear_solve(_xp(), A, B))
    shifts = [(0.0, 1.0)]
    for name in reversed(nest):
        h = NEST_OPS[name](h)
    for name in nest:
        if name == 'V':
            shifts += [(c + SHIFT, w / 2) for c, w in shifts]
    order = len(nest) - nest.count('V') + len(plain)
    want = torch.tensor([sum(w * _exact(P, x + c, order) for c, w in shifts) for x in XS])
    f = fn.vmap(h) if outer_vmap else (lambda x: torch.stack([h(x[i]) for i in range(len(XS))]))
    x0, one = torch.tensor(XS), torch.ones(len(XS))
    if plain == '':
        got = f(x0)
    elif plain == 'd':
        with fa.dual_level():
            got = fa.unpack_dual(f(fa.make_dual(x0, one))).tangent
    elif plain == 'b':
        x = x0.clone().requires_grad_()
        got = torch.autograd.grad(f(x).sum(), x)[0]
    elif plain == 'bd':
        x = x0.clone().requires_grad_()
        with fa.dual_level():
            t = fa.unpack_dual(f(fa.make_dual(x, one))).tangent
        got = torch.autograd.grad(t.sum(), x)[0]
    else:
        with fa.dual_level():
            x = fa.make_dual(x0, one).requires_grad_()
            got = fa.unpack_dual(torch.autograd.grad(f(x).sum(), x)[0]).tangent
    return got, want


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128], ids=['real', 'complex'])
def test_every_nest_of_three_transforms_matches_the_closed_form(dtype):
    """A census: every nest of up to three of ``jvp``, ``grad``, ``jacfwd``, ``jacrev`` and a
    two-lane ``vmap``, under each of the five plain orders above, with and without an outermost
    ``vmap``, of first to third order -- 784 cells. Each is the closed form on every lane, or
    PyTorch's own refusal of a ``forward_ad`` dual outside a ``jvp`` level ("Nested forward mode
    AD is not supported"; 168 cells on torch 2.13). ``torch.linalg.solve`` gets 360 of them
    wrong. A routing that read the plain dual beneath a ``torch.func`` level plainly, where it
    reads None, got 48 of them wrong, 0.34 to 0.52 relative off: a backward pass over a dual's
    tangent of one ``torch.func.grad`` or ``jacrev`` level, and the reverse, at third order."""
    P = _problem(dtype)
    wrong, checked, refused = [], 0, 0
    for n in range(4):
        for nest, plain, outer_vmap in itertools.product(
                itertools.product(NEST_OPS, repeat=n), PLAIN, (False, True)):
            if not 1 <= n - nest.count('V') + len(plain) <= 3:
                continue
            try:
                got, want = _nest_cell(P, nest, plain, outer_vmap)
            except RuntimeError as error:
                assert 'Nested forward mode AD is not supported' in str(error), (nest, plain)
                refused += 1
                continue
            checked += 1
            if not _relative(got, want) < 1e-10:
                wrong.append((''.join(nest), plain, outer_vmap, _relative(got, want)))
    assert not wrong, wrong
    assert checked + refused == 784 and checked >= 600, (checked, refused)


def _lu_calls(monkeypatch):
    calls = [0]
    original = torch.linalg.lu_factor

    def counting(*a, **k):
        calls[0] += 1
        return original(*a, **k)
    monkeypatch.setattr(torch.linalg, 'lu_factor', counting)
    return calls


#: Routes rewritten where the matrix moves, which a constant matrix leaves PyTorch's own: no rule
#: then holds a moving factor fixed. (Those that put a forward or reusing grad level under two
#: vmap levels are rewritten either way -- the batching defect needs no moving matrix.)
CONSTANT_MATRIX = ('jvp(jvp)', 'grad(jvp)', 'backward over forward_ad', 'forward_ad over backward',
                   'grad(dual-inside)', 'no_grad: jvp(jacrev)')


@pytest.mark.parametrize('route', [r for r, (_, _, exact) in ROUTES.items() if exact]
                         + ['first order', 'constant matrix'])
def test_native_where_exact(route, monkeypatch):
    """Where PyTorch's rule is exact the solve is ``torch.linalg.solve`` itself, and the bytes are
    its own: an eager value, first derivatives in each mode, the second-derivative routes whose
    backward re-solves (``hessian``, ``jvp(grad)``, ``create_graph=True``), and every route of
    :data:`CONSTANT_MATRIX` where ``A`` is captured as a constant outside the transforms and only
    ``B`` moves (there the closed form holds too, with ``A1 = 0``). Only ``linalg.solve`` runs --
    no ``lu_factor``."""
    P = _problem(torch.complex128)
    calls = _lu_calls(monkeypatch)
    constant = route == 'constant matrix'
    ours = _readout(P, lambda A, B: WB.linear_solve(_xp(), A, B), constant)
    native = _readout(P, torch.linalg.solve, constant)
    if route == 'first order':
        builds = [lambda g: _lanes(g), lambda g: _lanes(R(g)), lambda g: _lanes(F(g)),
                  lambda g: _per_lane(_create_graph(g, 1)), lambda g: _lanes(fn.jacfwd(g)),
                  lambda g: _per_lane(lambda at: _forward_over_backward_first(g, at))]
    elif constant:
        builds = [ROUTES[name][1] for name in CONSTANT_MATRIX]
        flat = dict(P, A1=torch.zeros_like(P['A1']))
        for name in CONSTANT_MATRIX:
            order, build, _ = ROUTES[name]
            want = torch.tensor([_exact(flat, x, order) for x in XS])
            assert _relative(build(ours)(), want) < 1e-10, name
    else:
        builds = [ROUTES[route][1]]
    for build in builds:
        a, b = build(ours)(), build(native)()
        assert a.dtype == b.dtype and a.detach().numpy().tobytes() == b.detach().numpy().tobytes(), route
    assert calls[0] == 0, route


def _forward_over_backward_first(g, at):
    with fa.dual_level():
        return fa.unpack_dual(g(fa.make_dual(torch.tensor(at), torch.tensor(1.0)))).tangent


def test_the_rewritten_solve_reads_operands_as_xp_linalg_solve():
    """Where it is rewritten, the value is ``xp.linalg.solve``'s: the same bytes outside ``vmap``
    (``lu_factor`` + ``lu_solve`` are the LAPACK calls ``linalg.solve`` makes) and the same shapes
    and dtypes for a matrix, a vector and a stacked right side, a real one beside a complex
    matrix, and a batch."""
    xp = _xp()
    g = torch.Generator().manual_seed(1)
    A = torch.randn(5, 5, dtype=torch.complex128, generator=g) + 3 * torch.eye(5)
    cases = {'matrix': torch.randn(5, 2, dtype=torch.complex128, generator=g),
             'vector': torch.randn(5, dtype=torch.complex128, generator=g),
             'real right side': torch.randn(5, 3, generator=g),
             'batched': (torch.randn(3, 5, 5, dtype=torch.complex128, generator=g) + 3 * torch.eye(5),
                         torch.randn(3, 5, 2, dtype=torch.complex128, generator=g)),
             # A's shape less its last axis: one matrix, broadcast over A's batch, not a stack of
             # vectors as torch.linalg.solve itself would read it
             'stacked right side': (torch.randn(5, 5, 5, dtype=torch.complex128, generator=g)
                                    + 3 * torch.eye(5), torch.randn(5, 5, dtype=torch.complex128, generator=g))}
    for name, case in cases.items():
        A_, B_ = case if isinstance(case, tuple) else (A, case)
        want = xp.linalg.solve(A_, B_)
        plain = WB._differentiable_solve(A_, B_, batched=False)
        unpacked = WB._differentiable_solve(A_, B_, batched=True)
        assert plain.shape == want.shape and plain.dtype == want.dtype, name
        assert plain.numpy().tobytes() == want.numpy().tobytes(), name
        assert unpacked.shape == want.shape and unpacked.dtype == want.dtype, name
        np.testing.assert_allclose(unpacked.numpy(), want.numpy(), rtol=1e-13, atol=1e-14)


def test_numpy_and_jax_are_their_own_solve():
    """No branch for NumPy or JAX: ``linear_solve`` is ``xp.linalg.solve``, byte for byte."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6)) + 3 * np.eye(6)
    B = rng.standard_normal((6, 2)) + 0j
    assert WB.linear_solve(np, A, B).tobytes() == np.linalg.solve(A, B).tobytes()
    pytest.importorskip('jax')
    wavebit.enable_x64()
    import jax.numpy as jnp
    Aj, Bj = jnp.asarray(A), jnp.asarray(B)
    assert np.asarray(WB.linear_solve(jnp, Aj, Bj)).tobytes() == np.asarray(jnp.linalg.solve(Aj, Bj)).tobytes()


@pytest.mark.parametrize('missing', ['get_interpreter_stack', 'TransformType', 'CGradInterpreterPtr'])
def test_a_missing_private_call_is_refused_by_name(missing):
    """The level reading uses ``torch._C._functorch``'s ``maybe_get_level``,
    ``get_interpreter_stack``, ``TransformType`` and ``CGradInterpreterPtr`` (a grad level's grad
    mode at entry): a PyTorch without one is refused by name, not met with an ``AttributeError``
    inside a solve -- and not taken as "no level", or as grad mode on."""
    assert missing in WB._LEVEL_EXTRA
    # Every call the vmap fix's walks already need is present; only the one named is not.
    names = WB._FUNCTORCH_CALLS + tuple(name for name in WB._LEVEL_EXTRA if name != missing)
    functorch = type('_functorch', (), {name: None for name in names})

    class Fake:
        __version__ = '9.9'
        _C = type('_C', (), {'_functorch': functorch})
    with pytest.raises(RuntimeError, match=missing):
        WB._functorch(Fake, WB._LEVEL_EXTRA)


def test_an_interpreter_key_that_is_not_a_transform_type_is_refused_by_name(monkeypatch):
    """A PyTorch whose interpreters key their transform by anything but a ``TransformType``
    member -- here the same stack, keyed by plain ints -- would read no level as forward mode,
    and ``jvp(jvp)`` would take ``torch.linalg.solve``'s rule: 0.57 relative off, with no error.
    It is refused by name instead."""
    functorch = torch._C._functorch
    stack = functorch.get_interpreter_stack

    class Renumbered:
        def __init__(self, interpreter):
            self.interpreter = interpreter

        def level(self):
            return self.interpreter.level()

        def key(self):
            return int(self.interpreter.key())

    P = _problem(torch.float64)
    g = _readout(P, lambda A, B: WB.linear_solve(_xp(), A, B))
    route = ROUTES['jvp(jvp)'][1](g)
    monkeypatch.setattr(functorch, 'get_interpreter_stack',
                        lambda: [Renumbered(i) for i in stack() or ()])
    with pytest.raises(RuntimeError, match='not a member of torch._C._functorch.TransformType'):
        route()


def test_a_grad_level_is_read_by_the_grad_mode_it_was_entered_with():
    """Each grad level is read by its grad mode at entry (``CGradInterpreterPtr``'s
    ``prevGradMode``), not the mode inside it, which ``torch.func`` turns on: ``hessian``'s grad
    level -- level 3, inside ``jacfwd``'s ``vmap`` and ``jvp`` -- is entered with it off under
    ``torch.no_grad()``, and a ``jacrev`` inside another with it on. Should the reading drift to
    "on" everywhere, the no-grad routes above would take PyTorch's rule, silently."""
    seen = []

    def probe(x):
        seen.append(WB._transform_modes()[2])
        return (x ** 3).sum()
    x = torch.tensor(0.3)
    fn.hessian(probe)(x)
    entered_on = seen[-1]
    with torch.no_grad():
        fn.hessian(probe)(x)
        entered_off = seen[-1]
        fn.jacrev(fn.jacrev(probe))(x)
        outer_off = seen[-1]
    assert (entered_on, entered_off, outer_off) == (set(), {3}, {1})


def _vjp_fn_called(call):
    """``h`` through ``torch.func.vjp`` entered in the caller's grad mode, its ``vjp_fn`` called by
    ``call``."""
    def build(h):
        return lambda x: call(fn.vjp(h, x)[1])
    return build


def _under_no_grad(vjp_fn):
    with torch.no_grad():
        return vjp_fn(torch.ones(()))[0]


def _without_create_graph(vjp_fn):
    return vjp_fn(torch.ones(()), create_graph=False)[0]


def _over_a_basis_under_no_grad(vjp_fn):
    """As ``jacrev`` calls it -- under a ``vmap`` over cotangents -- but with grad mode off."""
    with torch.no_grad():
        return fn.vmap(lambda c: vjp_fn(c)[0])(torch.ones(2)).mean()


def _entered_without_grad(h):
    def g(x):
        with torch.no_grad():
            vjp_fn = fn.vjp(h, x)[1]
        return vjp_fn(torch.ones(()))[0]
    return g


#: route: (order, build, refused). A ``vjp_fn`` called in a grad mode other than the one
#: ``torch.func.vjp`` was entered with, by which ``linear_solve`` read the level.
VJP_FN_ROUTES = {
    'jvp(vjp), vjp_fn under no_grad': (2, lambda g: _lanes(F(_vjp_fn_called(_under_no_grad)(g))), True),
    'jvp(vjp), vjp_fn(create_graph=False)': (
        2, lambda g: _lanes(F(_vjp_fn_called(_without_create_graph)(g))), True),
    'forward_ad over vjp, vjp_fn under no_grad': (
        2, lambda g: _forward_over(_vjp_fn_called(_under_no_grad)(g)), True),
    'vmap(vjp, vmap(vjp_fn) under no_grad)': (
        1, lambda g: _vmapped(_vjp_fn_called(_over_a_basis_under_no_grad)(g)), True),
    # Nothing differentiates the backward again, or batches the factors apart: exact as it is
    'vmap(vjp, vjp_fn under no_grad)': (1, lambda g: _vmapped(_vjp_fn_called(_under_no_grad)(g)), False),
    'vjp, vmap(vjp_fn) under no_grad': (
        1, lambda g: _lanes(_vjp_fn_called(_over_a_basis_under_no_grad)(g)), False),
    # Entered with grad mode off, the level is rewritten: exact in whatever mode vjp_fn then runs
    'jvp(vjp entered under no_grad), vjp_fn with grad on': (2, lambda g: _lanes(F(_entered_without_grad(g))),
                                                            False),
}


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128], ids=['real', 'complex'])
@pytest.mark.parametrize('route', list(VJP_FN_ROUTES))
def test_a_vjp_fn_in_another_grad_mode_is_exact_or_refused(route, dtype):
    """A grad level is read by the grad mode it was entered with, the one ``jacrev`` and
    ``hessian`` run its backward in. A ``vjp_fn`` called in another cannot be seen when the
    solve is taken. Where PyTorch's backward would then be wrong -- differentiated again by a
    forward level outside, or batched by a ``vmap`` added after the level beside one under it
    -- that backward raises, by name (``torch.linalg.solve`` reads 0.44 to 94 relative off there,
    with no error); elsewhere it is exact."""
    order, build, refused = VJP_FN_ROUTES[route]
    P = _problem(dtype)
    want = torch.tensor([_exact(P, x, order) for x in XS])
    run = build(_readout(P, lambda A, B: WB.linear_solve(_xp(), A, B)))
    if refused:
        with pytest.raises(RuntimeError, match='reaches a linear solve with grad mode off'):
            run()
        assert _relative(build(_readout(P, torch.linalg.solve))(), want) > 1e-3, route
    else:
        assert _relative(run(), want) < 1e-10, route


def test_pytorchs_own_rule_is_still_wrong():
    """Canary: torch.linalg.solve's ``jvp(jvp)`` and ``vmap(jacfwd)`` on this PyTorch. When this
    fails PyTorch has fixed one of them (pytorch#192667 fixes the first, from 2.15; nothing
    upstream names the second): measure every route of ``test_every_route_matches_the_closed_form``
    on the new version before narrowing ``linear_solve`` -- the backward's factor reuse with grad
    mode off is not part of that fix."""
    P = _problem(torch.complex128)
    native = _readout(P, torch.linalg.solve)
    second = torch.tensor([_exact(P, x, 2) for x in XS])
    first = torch.tensor([_exact(P, x, 1) for x in XS])
    assert _relative(ROUTES['jvp(jvp)'][1](native)(), second) > 1e-3
    assert _relative(ROUTES['vmap(jacfwd)'][1](native)(), first) > 1e-3


# ------------------------------------------------------------------------------- the scenes
from test_forward_mode import GRATING, ONE_D, TWO_D, RASTER, _lift, _read_out, _stack, one_d, raster  # noqa: E402


def _first_order_2d(algorithm, parameter='thickness'):
    def f(b, p):
        cfg = dict(TWO_D, layer_algorithm=algorithm)
        cfg[parameter] = [p] if parameter == 'thickness' else p
        solver = wavebit.rcwa(backend=b, structure=_lift(b, RASTER), **cfg)
        solver.formulation = 'first_order'
        return _read_out(b, solver.solve(), 'r_s', 6, point=([0.1], [-0.05], [0.07]))
    return f


def _efficiency_2d(b, p):
    """The 2D raster's ``R`` and ``T`` alone (``answers='efficiency'``), the read-out of a
    thickness Hessian of ``R`` and ``T``: the ETM final matching then solves the declared column
    alone, in its own branch, where the amplitude rung adds the pure-TE and pure-TM columns."""
    cfg = dict(TWO_D, layer_algorithm='ETM', answers='efficiency', thickness=[p])
    r = wavebit.rcwa(backend=b, structure=_lift(b, RASTER), **cfg).solve()
    return _stack(b, [r.R.sum(), r.T.sum(), r.R.reshape(-1)[4], r.T.reshape(-1)[2]])


def _coherent(b, p):
    assert b == 'torch'
    sources = [wavebit.PlaneWave(), wavebit.PlaneWave(side='bottom', amplitude=.3j)]
    solver = wavebit.rcwa(backend=b, excitation=sources, period=(600., 750.), wavelength=900.,
                          thickness=(p,), truncation_order=(1, 1), answers='field',
                          structure=_lift(b, RASTER))
    f = solver.solve().reconstruct_field([23.], [-19.], [83.])[0, 0, 0]
    return torch.stack([f[0].real, f[1].imag, f[3].real, f[4].imag])


def _excitation_conical(b, p):
    cfg = dict(period=(1400., 1500.), wavelength=900., thickness=(p,), truncation_order=(3, 0),
               theta=0.2, phi=0.3, psi=0.37, top=1.0, bot=2.25, answers='field', layer_algorithm='ETM',
               excitation={0: 1.0, 1: 0.4j})
    r = wavebit.rcwa(backend=b, structure=_lift(b, GRATING), **cfg).solve()
    return torch.stack([r.R.sum().real, r.T.sum().real])


#: scene: (builder, its parameter's value -- a thickness, but for the substrate ``bot`` where
#: named -- and the solve sites it reaches)
SCENES = {
    '1D TM SMM field': (one_d('thickness', 'SMM', 'r_p', tetm='tm'), 170.0, 'scattering.py _below_reflections, _field_state'),
    '2D Li ETM': (raster('thickness', layer_algorithm='ETM'), 0.2, 'solve_1d.py final_matching'),
    '2D Li ETM, R and T only': (_efficiency_2d, 0.2, 'solve_1d.py final_matching (the declared column alone)'),
    '2D first_order SMM field': (_first_order_2d('SMM'), 0.2, 'solve_2d_first_order.py _smm_cascade'),
    '2D first_order SMM field, bot': (_first_order_2d('SMM', 'bot'), 2.25,
                                      'solve_2d_first_order.py _interface_s (the substrate seeds the tail)'),
    'conical ETM excitation=': (_excitation_conical, 170.0, 'solve_1d.py final_matching (excitation columns)'),
    'CoherentSolver PlaneWave': (_coherent, 170.0, 'illumination.py _affine_star, _inside_amplitudes'),
}


def _hessian_under_no_grad(g, x):
    with torch.no_grad():
        return fn.hessian(g)(x)


SCENE_ROUTES = {
    'jvp(jvp)': lambda g, x: F(F(g))(x),
    'backward over forward_ad': lambda g, x: _backward_over_forward(g)(float(x)),
    'forward_ad over backward': lambda g, x: _forward_over_backward(g, False)(float(x)),
    'no_grad: hessian': _hessian_under_no_grad,
    'grad(dual-inside)': lambda g, x: R(_dual_inside(g))(x),
}
#: The routes through torch.func, which refuses the CoherentSolver
TORCH_FUNC_ROUTES = {'jvp(jvp)', 'no_grad: hessian', 'grad(dual-inside)'}


@pytest.mark.parametrize('route', list(SCENE_ROUTES))
@pytest.mark.parametrize('scene', list(SCENES))
def test_a_thickness_second_derivative_is_exact(scene, route):
    """A thickness moves no modal operator (nor does the substrate ``bot``), so nothing refuses
    its second derivative, and every route through each site that solves through
    ``linear_solve`` matches ``create_graph=True`` (exact: its backward re-solves). Before, each
    of the first three routes read 7.3e-2 relative off on the 1D TM field, 6.4e-2 on the 2D ETM
    raster, 6.8e-2 on the first-order SMM field, 5.4 on the conical ``excitation=`` scene (the
    sign wrong) and 0.74 on the CoherentSolver; ``torch.func.hessian`` under ``torch.no_grad()``,
    and ``torch.func.grad`` of a ``forward_ad`` tangent, read 7.3e-2 and 6.4e-2 off on the first
    two."""
    if scene == 'CoherentSolver PlaneWave' and route in TORCH_FUNC_ROUTES:
        pytest.skip('torch.func refuses the CoherentSolver ("Cannot access data pointer"), loudly')
    f, at, _ = SCENES[scene]
    n = f('torch', torch.tensor(at)).numel()
    w = torch.as_tensor(np.random.default_rng(5).standard_normal(n))
    g = lambda p: (w * f('torch', p)).sum()  # noqa: E731
    want = float(_create_graph(g, 2)(at))
    got = float(SCENE_ROUTES[route](g, torch.tensor(at)))
    assert got == pytest.approx(want, rel=1e-9), (scene, route)


@pytest.mark.parametrize('scene', ['1D TM SMM field', '2D Li ETM'])
def test_vmap_of_jacfwd_and_hessian_are_exact_on_every_lane(scene):
    """``vmap(jacfwd)`` -- a first derivative -- and ``vmap(hessian)`` over thickness lanes, and
    ``vmap(jacrev)`` under ``torch.no_grad()``, against eager autograd and ``create_graph=True``
    lane by lane. Before, lane 0 was right and the others read 197.0 for 0.01147 (1D TM) or NaN
    (2D ETM)."""
    f, at, _ = SCENES[scene]
    lanes = [at, at * 1.005, at * 1.0125]
    n = f('torch', torch.tensor(at)).numel()
    w = torch.as_tensor(np.random.default_rng(5).standard_normal(n))
    g = lambda p: (w * f('torch', p)).sum()  # noqa: E731
    first = torch.tensor([float(_create_graph(g, 1)(p)) for p in lanes])
    second = torch.tensor([float(_create_graph(g, 2)(p)) for p in lanes])
    xs = torch.tensor(lanes)
    assert _relative(fn.vmap(fn.jacfwd(g))(xs), first) < 1e-9
    assert _relative(fn.vmap(fn.hessian(g))(xs), second) < 1e-9
    with torch.no_grad():
        # jacrev's backward with grad mode off, under the outer vmap and its own: NaN on lanes 1
        # and 2 of the 2D ETM raster before
        assert _relative(fn.vmap(fn.jacrev(g))(xs), first) < 1e-9


def _at_wavelength(algorithm, parameter, **polarization):
    """``test_forward_mode``'s 1D grating as a function of the wavelength and ``parameter``."""
    def f(wavelength, p):
        cfg = dict(ONE_D, layer_algorithm=algorithm, wavelength=wavelength, **polarization)
        cfg[parameter] = p
        solver = wavebit.rcwa(backend='torch', structure=_lift('torch', GRATING), **cfg)
        return _read_out('torch', solver.solve(), 'r_s', 3, point=([57.], [0.], [83.]))
    return f


#: scene: (read-out of (wavelength, parameter), the parameter's value)
SPECTRAL = {
    'conical SMM, psi': (_at_wavelength('SMM', 'psi', phi=0.3), 0.37),
    'conical ETM, psi': (_at_wavelength('ETM', 'psi', phi=0.3), 0.37),
    '1D TM SMM, bot': (_at_wavelength('SMM', 'bot', tetm='tm'), 2.25),
    'conical ETM, bot': (_at_wavelength('ETM', 'bot', psi=0.37, phi=0.3), 2.25),
}


@pytest.mark.parametrize('scene', list(SPECTRAL))
def test_a_wavelength_vmap_of_jacfwd_is_exact_on_every_lane(scene):
    """``vmap`` over wavelength lanes of ``jacfwd`` in the polarization angle ``psi``, which moves
    only the incident right side of the matching, or the substrate ``bot``. The wavelength moves
    every modal operator, but under ``vmap`` alone, which differentiates nothing; the forward
    level sits under two ``vmap`` levels (the outer one and ``jacfwd``'s own), so PyTorch's rule
    would call ``lu_solve`` with factors batched by one and a right side by the other. A first
    derivative, against eager autograd lane by lane. Before, lane 0 was right and the others
    wrong by up to 6e6 relative, or NaN."""
    f, p0 = SPECTRAL[scene]
    wavelengths = [900., 905., 912.]
    n = f(torch.tensor(wavelengths[0]), torch.tensor(p0)).numel()
    w = torch.as_tensor(np.random.default_rng(5).standard_normal(n))
    g = lambda wavelength, p: (w * f(wavelength, p)).sum()  # noqa: E731
    eager = []
    for wavelength in wavelengths:
        x = torch.tensor(p0, requires_grad=True)
        eager.append(float(torch.autograd.grad(g(torch.tensor(wavelength), x), x)[0]))
    got = fn.vmap(lambda wavelength: fn.jacfwd(lambda p: g(wavelength, p))(torch.tensor(p0)))(
        torch.tensor(wavelengths))
    assert _relative(got, torch.tensor(eager)) < 1e-9


def _nest(names):
    """``names`` (outermost first) of ``torch.func.jvp`` and ``grad``, at a float."""
    def build(g):
        for name in reversed(names):
            g = {'jvp': F, 'grad': R}[name](g)
        return lambda at: g(torch.tensor(at))
    return build


THIRD_ROUTES = {
    **{'('.join(names) + '))': _nest(names) for names in itertools.product(('jvp', 'grad'), repeat=3)},
    'backward x2 over forward_ad': _backward_twice_over_forward,
    'backward over forward_ad over grad': lambda g: _backward_over_forward(R(g)),
    'forward_ad over backward over grad': lambda g: _forward_over_backward(R(g), False),
}


@functools.lru_cache(maxsize=None)
def _third_scene():
    """The conical ETM grating's field read-out, weighted, and its third derivative in the
    thickness by ``create_graph=True`` three times (each backward re-solves: exact)."""
    f = one_d('thickness', 'ETM', 'r_s', psi=0.37, phi=0.3)
    w = torch.as_tensor(np.random.default_rng(5).standard_normal(f('numpy', 170.0).size))
    g = lambda p: (w * f('torch', p)).sum()  # noqa: E731
    return g, float(_create_graph(g, 3)(170.0))


@pytest.mark.parametrize('route', list(THIRD_ROUTES))
def test_a_thickness_third_derivative_is_exact(route):
    """Every nest of three ``torch.func`` ``jvp`` and ``grad`` levels, and the plain orders around
    a backward pass and a ``forward_ad`` dual, in the thickness of a conical ETM grating's field
    read-out, against ``create_graph=True`` three times. Before, every nest with a ``jvp`` level
    inside another level read 0.4% to 14% off, with no error; ``grad(grad(grad))`` and
    ``jvp(grad(grad))`` were right."""
    g, want = _third_scene()
    assert float(THIRD_ROUTES[route](g)(170.0)) == pytest.approx(want, rel=1e-9), route


def test_jax_agrees_on_the_2d_etm_raster():
    """An oracle that shares no rule with ``linear_solve``: ``jax.hessian`` (``jnp.linalg.solve``'s
    own rules, exact) against PyTorch's ``jvp(jvp)`` through it."""
    jax = pytest.importorskip('jax')
    wavebit.enable_x64()
    import jax.numpy as jnp
    f, at, _ = SCENES['2D Li ETM']
    w = np.random.default_rng(5).standard_normal(f('numpy', at).size)
    want = float(jax.hessian(lambda d: jnp.sum(jnp.asarray(w) * f('jax', d)))(jnp.asarray(at)))
    g = lambda p: (torch.as_tensor(w) * f('torch', p)).sum()  # noqa: E731
    assert float(F(F(g))(torch.tensor(at))) == pytest.approx(want, rel=1e-9)


# ------------------------------------------------------------------------------- structure
#: The references to a ``linalg`` module's ``solve`` that do not go through linear_solve, and why
#: (paths relative to the package).
EXEMPT = {
    ('_core/backend.py', '_eig_grad'): "inside eig's custom backward; no forward rule, so a forward level raises",
    ('_core/modal_gradient.py', 'jvp'): 'the modal rule is first order; a second derivative through it raises',
    ('_core/modal_gradient.py', 'backward'): 'the modal rule is first order; a second derivative through it raises',
    ('_core/modal_gradient.py', 'tangent'): "JAX's rule (jnp.linalg.solve's derivatives are exact)",
    ('_core/backend.py', 'linear_solve'): 'the one place it is spelled',
}


def _solve_references(source):
    """``(innermost function, line)`` of each reference to a ``linalg`` module's ``solve`` in
    ``source``, called or not: ``<x>.linalg.solve`` (``solve = xp.linalg.solve`` is one),
    ``getattr(<x>.linalg, 'solve')``, ``solve`` imported from a ``*.linalg`` module, and
    ``<name>.solve`` where ``<name>`` is bound to a ``linalg`` module (``import torch.linalg as
    la``, ``from torch import linalg``, ``la = xp.linalg``). Host NumPy's ``np.linalg`` is
    skipped: it never sees a tensor."""
    tree = ast.parse(source)
    parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def owner(node):
        while node in parent:
            node = parent[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return '<module>'

    def linalg(node):
        return (isinstance(node, ast.Attribute) and node.attr == 'linalg'
                and not (isinstance(node.value, ast.Name) and node.value.id == 'np'))

    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {a.asname for a in node.names if a.asname and a.name.split('.')[-1] == 'linalg'}
        elif isinstance(node, ast.ImportFrom):
            aliases |= {a.asname or a.name for a in node.names if a.name == 'linalg'}
        elif isinstance(node, ast.Assign) and linalg(node.value):
            aliases |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == 'solve' and (
                linalg(node.value) or isinstance(node.value, ast.Name) and node.value.id in aliases):
            found.append((owner(node), node.lineno))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'getattr'
              and len(node.args) >= 2 and linalg(node.args[0])
              and isinstance(node.args[1], ast.Constant) and node.args[1].value == 'solve'):
            found.append((owner(node), node.lineno))
        elif (isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[-1] == 'linalg'
              and any(a.name == 'solve' for a in node.names)):
            found.append((owner(node), node.lineno))
    return sorted(found)


@pytest.mark.parametrize('source, found', [
    ('def f(xp, a, b):\n    return xp.linalg.solve(a, b)\n', [('f', 2)]),
    ('def f(xp, a, b):\n    solve = xp.linalg.solve\n    return solve(a, b)\n', [('f', 2)]),
    ("def f(xp, a, b):\n    return getattr(xp.linalg, 'solve')(a, b)\n", [('f', 2)]),
    ('from torch.linalg import solve\n', [('<module>', 1)]),
    ('import torch.linalg as la\n\ndef f(a, b):\n    return la.solve(a, b)\n', [('f', 4)]),
    ('from torch import linalg\n\ndef f(a, b):\n    return linalg.solve(a, b)\n', [('f', 4)]),
    ('def f(xp, a, b):\n    la = xp.linalg\n    return la.solve(a, b)\n', [('f', 3)]),
    ('def f(xp, a, b):\n    def g():\n        return xp.linalg.solve(a, b)\n    return g\n', [('g', 3)]),
    ('def f(a, b):\n    return np.linalg.solve(a, b)\n', []),
    ('def f(solver):\n    return solver.solve()\n', []),
    ('def f(xp, a, b):\n    return xp.linalg.solve_triangular(a, b)\n', []),
])
def test_the_scan_sees_each_spelling_it_names(source, found):
    """The scan below reads references, not calls: an alias of ``xp.linalg.solve`` taken before
    its call (``solve = xp.linalg.solve; solve(A, B)`` in ``final_matching`` or ``_interface_s``)
    passed a scan of call nodes, and every other test."""
    assert _solve_references(source) == found


def test_every_solve_goes_through_linear_solve():
    """Every reference to a ``linalg`` module's ``solve`` in the product code (all of ``wavebit``
    outside the research tier's ``_core/conforming/``) is ``linear_solve`` or a listed exemption:
    a new site spelled ``xp.linalg.solve`` would bring PyTorch's second derivative back,
    silently, the first time a parameter moves its matrix. And every exemption names a reference
    the scan sees, so the scan is seen to read the files -- and since every exemption is under
    ``_core/``, its reach beyond it is asserted by name: no file outside ``_core/`` refers to a
    solve today, so a walk narrowed back to ``_core/*.py`` would pass everything else."""
    package = pathlib.Path(wavebit.__file__).parent
    found, scanned = [], []
    for path in sorted(package.rglob('*.py')):
        name = path.relative_to(package).as_posix()
        if not name.startswith('_core/conforming/'):
            scanned.append(name)
            found += [(name, func, line) for func, line in _solve_references(path.read_text())]
    assert {'main.py', 'materials.py', 'backends/torch_backend.py', 'backends/jax_backend.py',
            '_core/solve_1d.py'} <= set(scanned), scanned
    assert not any(name.startswith('_core/conforming/') for name in scanned), scanned
    stray = [(name, func, line) for name, func, line in found if (name, func) not in EXEMPT]
    assert not stray, f'solve outside linear_solve: {stray}'
    unseen = set(EXEMPT) - {(name, func) for name, func, _ in found}
    assert not unseen, f'exemptions that name no solve: {sorted(unseen)}'
