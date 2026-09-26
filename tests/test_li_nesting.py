"""Li's nested rules, rebuilt from the paper: the definition, and what the nesting changes.

The raster ``factorization='li'`` builders were replaced on 2026-09-25 by an implementation
written from the published mathematics -- L. Li, JOSA A 14, 2758 (1997), with JOSA A 13,
1870 (1996) for the one-dimensional rules. The earlier builders came from meent, whose Li-rule
code was written with reference to RETICOLO's MATLAB source. The rewrite also added the second
of Li's two nestings (:class:`~wavebit._core.enums.Nesting`).

What is checked here:

* **The definition.** :func:`_literal` below builds every slot by index loops over the
  formulas -- the explicit segment integral for every coefficient, ``[T]_{m,m'} = c_{m-m'}``
  entry by entry, and the table in :class:`~wavebit._core.enums.Nesting` -- sharing no code and
  no shortcut with the builders (no FFT, no jump form, no gather). Both nestings, both
  coefficient paths, to 1e-12.
* **Properties of the nesting itself**: the symmetric nesting is covariant under relabelling x
  and y at the matrix level, the fixed one is not, and both reduce to the 1D rules on a
  y-invariant layer.
* **Backends and gradients**: three backends agree, and the raster gradient is the finite
  difference's, in both nestings.
* **The keyword** (``li_nesting=``): read by scalar Li on a raster and on axis-aligned rectangles,
  and refused by name where nothing honours it -- NVF, and ``'symmetric'`` on tensor Li.

A fourth kind lived here for one commit and went with its subject: the new builders against
the old ones, black-box, before the old were deleted. It measured the sampled path bit-identical
on NumPy, JAX and PyTorch, the integrated path within 4.3e-14, the raster gradients within
2.8e-14, and ``fourier2d_pointwise``'s replacement bit-identical on all three backends -- see
the commit that added this file. On the machine the rewrite was rebased on, JAX moved the
pointwise transform by round-off (a few ulps, never on a one-pixel or single-row grid) while
NumPy and PyTorch stayed bit-identical.
"""
import numpy as np
import pytest

import array_api_compat.numpy as xnp

import wavebit
from wavebit._core import factorization as F

try:
    wavebit.enable_x64()
except ImportError:
    pass


# ---------------------------------------------------------------------------- the definition

def _segment_weights(right_edges, K):
    """``W[k + K, j]`` = integral over segment ``j`` of ``exp(-2 pi i k t)``, ``k = -K..K``."""
    right = np.asarray(right_edges, dtype=float).ravel()
    right = right / right[-1]
    left = np.concatenate([[0.0], right[:-1]])
    W = np.empty((2 * K + 1, right.size), dtype=complex)
    for k in range(-K, K + 1):
        if k == 0:
            W[k + K] = right - left
        else:
            W[k + K] = (np.exp(-2j * np.pi * k * left)
                        - np.exp(-2j * np.pi * k * right)) / (2j * np.pi * k)
    return W


def _toeplitz(c, M):
    """``(..., 4M+1)`` coefficients -> ``(..., 2M+1, 2M+1)`` with ``T[m, m'] = c_{m - m'}``."""
    n = 2 * M + 1
    T = np.empty(c.shape[:-1] + (n, n), dtype=complex)
    for r in range(n):
        for s in range(n):
            T[..., r, s] = c[..., (r - s) + 2 * M]
    return T


def _literal_nested(g, x, y, Mu, Mv, inner, invert_inner, invert_outer):
    """One nested operator, in the layout ``i = n*n_orders_u + m``, by index loops."""
    g = np.asarray(g, dtype=complex)
    nu, nv = 2 * Mu + 1, 2 * Mv + 1
    Wu, Wv = _segment_weights(x, 2 * Mu), _segment_weights(y, 2 * Mv)
    out = np.empty((nv, nu, nv, nu), dtype=complex)
    if inner == 'v':
        A = _toeplitz((Wv @ g).T, Mv)                        # T_v(g)(u), one per column
        A = np.linalg.inv(A) if invert_inner else A
        A_hat = np.einsum('kj,jab->kab', Wu, A)
        for n in range(nv):
            for m in range(nu):
                for n2 in range(nv):
                    for m2 in range(nu):
                        out[n, m, n2, m2] = A_hat[(m - m2) + 2 * Mu, n, n2]
    else:
        B = _toeplitz(g @ Wu.T, Mu)                          # T_u(g)(v), one per row
        B = np.linalg.inv(B) if invert_inner else B
        B_hat = np.einsum('kj,jab->kab', Wv, B)
        for n in range(nv):
            for m in range(nu):
                for n2 in range(nv):
                    for m2 in range(nu):
                        out[n, m, n2, m2] = B_hat[(n - n2) + 2 * Mv, m, m2]
    out = out.reshape(nv * nu, nv * nu)
    return np.linalg.inv(out) if invert_outer else out


def _literal(eps, x, y, Mu, Mv, nesting):
    """``(epx, epy, epz_i)`` straight from the table in :class:`~wavebit._core.enums.Nesting`."""
    eps = np.asarray(eps, dtype=complex)
    if nesting == 'fixed':
        epx = _literal_nested(eps, x, y, Mu, Mv, 'v', True, True)
    else:
        epx = _literal_nested(1 / eps, x, y, Mu, Mv, 'u', True, False)
    epy = _literal_nested(1 / eps, x, y, Mu, Mv, 'v', True, False)
    epz_i = _literal_nested(eps, x, y, Mu, Mv, 'v', False, True)
    return epx, epy, epz_i


def _pixels(n):
    return np.arange(1, n + 1) / n


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


METAL = -16.49 + 1.06j          # gold at 700 nm, Johnson-Christy: Re(eps) < 0


def _raster(shape, phases, seed, block=(1, 1)):
    """A random raster over ``phases``, optionally built from ``block``-sized constant cells so
    that the integrated path's compression has something to merge."""
    rng = np.random.default_rng(seed)
    coarse = (-(-shape[0] // block[0]), -(-shape[1] // block[1]))
    idx = rng.integers(0, len(phases), size=coarse)
    idx = np.repeat(np.repeat(idx, block[0], axis=0), block[1], axis=1)[:shape[0], :shape[1]]
    return np.asarray(phases, dtype=complex)[idx]


#: name -> (raster, half_width_u, half_width_v). Two to five phases, metal among them; unequal
#: truncation; sizes that are prime, odd, a single row or column, and smaller than
#: ``4*half_width+1`` (so the DFT wraps and only the exact correction keeps the orders right).
CASES = {
    'two_phase_metal': (_raster((7, 11), (1.0, METAL), 0), 3, 2),
    'three_phase': (_raster((13, 5), (1.0, 6.0, 2.1 + 0.1j), 1), 1, 4),
    'four_phase_primes': (_raster((23, 29), (1.0, 12.25, METAL, 3.0 + 0.4j), 2), 5, 2),
    'five_phase': (_raster((17, 19), (1.0, 2.25, 6.0, METAL, 4.0 + 1.0j), 3), 4, 3),
    'wrapped_orders': (_raster((5, 6), (1.0, 9.0, 2.0 + 0.3j), 4), 4, 3),
    'one_row': (_raster((1, 13), (1.0, METAL, 6.0), 5), 4, 0),
    'one_column': (_raster((11, 1), (1.0, 6.0), 6), 0, 3),
    'blocks_metal': (_raster((12, 18), (1.0, METAL, 6.0, 2.5), 7, block=(3, 2)), 3, 3),
}

NESTINGS = ('fixed', 'symmetric')


@pytest.mark.parametrize('nesting', NESTINGS)
@pytest.mark.parametrize('name', CASES)
def test_the_sampled_slots_are_the_definition(name, nesting):
    """Pixel edges on the unit cell: the sampled path must be the literal staircase integral."""
    eps, Mu, Mv = CASES[name]
    got = F.li_conv_matrices(xnp, eps, Mu, Mv, nesting=nesting)
    want = _literal(eps, _pixels(eps.shape[1]), _pixels(eps.shape[0]), Mu, Mv, nesting)
    for slot, a, b in zip(('epx', 'epy', 'epz_i'), got, want):
        assert _rel(a, b) < 1e-12, (name, nesting, slot, _rel(a, b))
    assert got.epxy is None and got.epyx is None


@pytest.mark.parametrize('nesting', NESTINGS)
@pytest.mark.parametrize('name', CASES)
def test_the_integrated_slots_are_the_definition(name, nesting):
    """The compressed layer on its own segment edges -- and on edges no pixel grid could hold."""
    eps, Mu, Mv = CASES[name]
    row, col, x, y = F.compression_indices(eps)
    comp = np.asarray(F.compress_cell(xnp, eps, row, col))
    for edges in ((x, y), _arbitrary_edges(x, y, seed=len(name))):
        got = F.li_conv_matrices_integrated(xnp, comp, *edges, Mu, Mv, nesting=nesting)
        want = _literal(comp, *edges, Mu, Mv, nesting)
        for slot, a, b in zip(('epx', 'epy', 'epz_i'), got, want):
            assert _rel(a, b) < 1e-12, (name, nesting, slot, _rel(a, b))


def _arbitrary_edges(x, y, seed):
    """The same number of segments on edges that fall nowhere near a pixel boundary, and in a
    unit that is not the period -- the builder reads the last edge as the period."""
    rng = np.random.default_rng(seed)
    out = []
    for e, scale in ((x, 325.0), (y, 0.37)):
        inner = np.sort(rng.uniform(0.02, 0.98, size=len(e) - 1))
        out.append((np.append(inner, 1.0) * scale).reshape(-1, 1))
    return tuple(out)


def _assert_eps_times_identity(conv, eps, where):
    """Every slot of a uniform layer: identically zero off the diagonal, and ``eps`` on it
    (``1/eps`` for ``epz_i``) -- to the bit for a real ``eps``, to a few ulps for a complex one."""
    for slot, want in (('epx', eps), ('epy', eps), ('epz_i', 1 / eps)):
        got = np.asarray(getattr(conv, slot))
        off = ~np.eye(got.shape[0], dtype=bool)
        assert np.max(np.abs(got[off])) == 0.0, where + (slot, 'off the diagonal')
        miss = np.max(np.abs(np.diag(got) - want))
        limit = 0.0 if isinstance(eps, float) else 4 * np.finfo(float).eps * abs(want)
        assert miss <= limit, where + (slot, miss)


def test_a_uniform_layer_is_exactly_eps_times_identity_in_both_nestings():
    """Nothing to factorize, so every slot is ``eps * I`` exactly -- and ``epz_i`` is ``I/eps``.

    The integrated path gets this only because its coefficients are taken in the jump form
    (``_segment_series``): the segment form leaves ``exp(-2 pi i k)`` of round-off off ``k = 0``.
    The sampled path gets it at every grid size only because ``_raster_series`` transforms a
    staircase as ``a - a_0``: the FFT of a constant is exact on 8x8 but not on 20 pixels, where
    the matrix carried 1.3e-16 off the diagonal until 2026-09-26. Hence the sizes below, which
    include grids shorter than ``4*3 + 1``.

    Exact is a claim about the series, so it is held exactly where the series decides it: off
    the diagonal every slot is identically zero, for every value. On the diagonal a real
    ``eps`` is exact too, but a complex one passes through LAPACK's inverse, which rounds a
    complex reciprocal its own way -- on macOS arm64 with NumPy 2.5.1, ``inv(inv(z * I))``
    misses ``z = 3.1+0.7j`` by 5.0e-16 -- so there the diagonal is held to a few ulps.
    """
    for shape in ((8, 8), (7, 5), (1, 20), (13, 14), (5, 3)):
        for eps in (1.0, 2.25, 7.290000000000001, 3.1 + 0.7j, METAL):
            cell = np.full(shape, eps, dtype=complex)
            row, col, x, y = F.compression_indices(cell)
            comp = np.asarray(F.compress_cell(xnp, cell, row, col))
            for nesting in NESTINGS:
                for conv in (F.li_conv_matrices(xnp, cell, 3, 3, nesting=nesting),
                             F.li_conv_matrices_integrated(xnp, comp, x, y, 3, 3, nesting=nesting)):
                    _assert_eps_times_identity(conv, eps, (shape, eps, nesting))


def test_an_unknown_nesting_is_refused_by_name():
    eps = CASES['two_phase_metal'][0]
    with pytest.raises(ValueError, match=r"nesting='inner'.*'fixed', 'symmetric'"):
        F.li_conv_matrices(xnp, eps, 2, 2, nesting='inner')


# --------------------------------------------------------------------- properties of the nesting

def _swap_orders(n_orders_u, n_orders_v):
    """What relabelling x and y does to the order grid, as a gather index.

    The transposed layer's flat index ``n'*n_orders_v + m'`` -- its ``v`` orders are the
    original's ``u`` orders, ``n_orders_u`` of them -- holds the original's order
    ``(n, m) = (m', n')``, at flat index ``m'*n_orders_u + n'``. Listed in the transposed
    layer's order, so ``A[np.ix_(p, p)]`` is ``A`` relabelled.
    """
    return np.array([m * n_orders_u + n for n in range(n_orders_u) for m in range(n_orders_v)])


def _build(coefficients, eps, Mu, Mv, nesting):
    """One layer's slots on either coefficient path, the integrated one on its own compression."""
    if coefficients == 'sampled':
        return F.li_conv_matrices(xnp, eps, Mu, Mv, nesting=nesting)
    row, col, x, y = F.compression_indices(eps)
    return F.li_conv_matrices_integrated(xnp, np.asarray(F.compress_cell(xnp, eps, row, col)),
                                         x, y, Mu, Mv, nesting=nesting)


@pytest.mark.parametrize('coefficients', ['sampled', 'integrated'])
@pytest.mark.parametrize('name', ['two_phase_metal', 'five_phase', 'blocks_metal'])
def test_the_symmetric_nesting_is_covariant_under_relabelling_x_and_y(name, coefficients):
    """``epx`` of the transposed layer is ``epy`` of the layer, relabelled -- to round-off.

    That is the whole content of the symmetric nesting: each slot puts its own inverse-rule axis
    inside, so relabelling the axes exchanges the two slots. The fixed nesting keeps ``v``
    inside for both, and the same comparison lands far from round-off -- asserted too, so a
    nesting that silently became the other one fails here rather than in a golden.
    """
    eps, Mu, Mv = CASES[name]
    p = _swap_orders(2 * Mu + 1, 2 * Mv + 1)
    sym = _build(coefficients, eps, Mu, Mv, 'symmetric')
    sym_t = _build(coefficients, eps.T.copy(), Mv, Mu, 'symmetric')
    for mine, theirs in ((sym_t.epx, sym.epy), (sym_t.epy, sym.epx), (sym_t.epz_i, sym.epz_i)):
        relabelled = np.asarray(theirs)[np.ix_(p, p)]
        assert _rel(mine, relabelled) < 1e-13, _rel(mine, relabelled)

    fixed = _build(coefficients, eps, Mu, Mv, 'fixed')
    fixed_t = _build(coefficients, eps.T.copy(), Mv, Mu, 'fixed')
    assert _rel(fixed_t.epx, np.asarray(fixed.epy)[np.ix_(p, p)]) > 1e-3


@pytest.mark.parametrize('nesting', NESTINGS)
@pytest.mark.parametrize('coefficients', ['sampled', 'integrated'])
def test_a_y_invariant_layer_reduces_to_the_1d_rules_in_both_nestings(nesting, coefficients):
    """Rows all equal: every slot is block-diagonal over the ``v`` orders, each block the 1D one.

    ``T_v`` of a function of ``u`` alone is that function times the identity, so the nesting
    has nothing to act on: ``epx`` is Li's 1D inverse rule ``[[1/eps]]_u^-1`` and ``epy`` the
    direct rule ``[[eps]]_u``, in every ``v`` block -- each exactly what the one-row raster, the
    1D path's input, builds with ``half_width_v = 0``.
    """
    row = CASES['one_row'][0]
    Mu, Mv = 4, 3
    conv = _build(coefficients, np.repeat(row, 5, axis=0), Mu, Mv, nesting)
    one_d = _build(coefficients, row, Mu, 0, nesting)
    eye_v = np.eye(2 * Mv + 1)
    for slot, a, b in zip(('epx', 'epy', 'epz_i'), conv, one_d):
        want = np.kron(eye_v, np.asarray(b))
        assert np.max(np.abs(np.asarray(a) - want)) < 1e-12 * np.max(np.abs(want)), slot

    # ...and the 1D blocks are the 1D rules themselves, from the definition
    if coefficients == 'sampled':
        values, edges = row[0], _pixels(row.shape[1])
    else:
        _, col, edges, _ = F.compression_indices(row)
        values = row[0, col]
    series = lambda g: _segment_weights(edges, 2 * Mu) @ g          # noqa: E731
    assert _rel(one_d.epx, np.linalg.inv(_toeplitz(series(1 / values), Mu))) < 1e-12
    assert _rel(one_d.epy, _toeplitz(series(values), Mu)) < 1e-12


# --------------------------------------------------------------- backends, gradients, wrappers

def _namespace(backend):
    if backend == 'numpy':
        return xnp, np.asarray
    if backend == 'jax':
        import jax.numpy as jnp
        return jnp, jnp.asarray
    import torch
    import array_api_compat.torch as xt
    return xt, lambda a: torch.as_tensor(np.asarray(a))


def _to_numpy(a):
    return a.detach().numpy() if hasattr(a, 'detach') else np.asarray(a)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('nesting', NESTINGS)
@pytest.mark.parametrize('name', ['four_phase_primes', 'blocks_metal', 'wrapped_orders'])
def test_every_backend_builds_the_same_slots(name, nesting, backend):
    """One implementation behind the array-API adapter, so three backends agree to round-off."""
    pytest.importorskip(backend)
    xp, lift = _namespace(backend)
    eps, Mu, Mv = CASES[name]
    row, col, x, y = F.compression_indices(eps)
    pairs = (
        (F.li_conv_matrices(xnp, eps, Mu, Mv, nesting=nesting),
         F.li_conv_matrices(xp, lift(eps), Mu, Mv, nesting=nesting)),
        (F.li_conv_matrices_integrated(xnp, F.compress_cell(xnp, eps, row, col), x, y, Mu, Mv,
                                       nesting=nesting),
         F.li_conv_matrices_integrated(xp, F.compress_cell(xp, lift(eps), row, col), x, y,
                                       Mu, Mv, nesting=nesting)),
    )
    for want, got in pairs:
        for slot, a, b in zip(('epx', 'epy', 'epz_i'), want, got):
            assert _rel(_to_numpy(b), a) < 1e-12, (name, nesting, backend, slot)


def _functional(xp, mats, weights):
    """A fixed real linear functional of the three slots -- one number to differentiate."""
    return sum(xp.real(xp.sum(w * m)) for w, m in zip(weights, mats))


@pytest.mark.parametrize('nesting', NESTINGS)
@pytest.mark.parametrize('coefficients', ['sampled', 'integrated'])
def test_the_raster_is_a_design_variable_in_both_nestings(coefficients, nesting):
    """d(functional)/d(raster): JAX == PyTorch per cell, and both == a central difference.

    The raster is what an inverse design moves, so the gradient has to reach it through every
    inversion and transform here -- the inner ones the nesting adds included.
    """
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    eps, Mu, Mv = CASES['blocks_metal']
    n = (2 * Mu + 1) * (2 * Mv + 1)
    rng = np.random.default_rng(21)
    weights = [rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)) for _ in range(3)]
    row, col, x, y = F.compression_indices(eps)

    def slots(xp, s, imag):
        cell = s + 1j * imag
        if coefficients == 'sampled':
            return F.li_conv_matrices(xp, cell, Mu, Mv, nesting=nesting)[:3]
        return F.li_conv_matrices_integrated(xp, F.compress_cell(xp, cell, row, col),
                                             x, y, Mu, Mv, nesting=nesting)[:3]

    jxp, jlift = _namespace('jax')
    g_jax = np.asarray(jax.grad(lambda s: _functional(
        jxp, slots(jxp, s, jlift(eps.imag)), [jlift(w) for w in weights]))(jlift(eps.real)))

    txp, tlift = _namespace('torch')
    s = torch.tensor(eps.real, requires_grad=True)
    _functional(txp, slots(txp, s, tlift(eps.imag)), [tlift(w) for w in weights]).backward()
    g_torch = s.grad.numpy()

    direction = rng.standard_normal(eps.shape)
    h = 1e-6
    value = lambda d: float(_functional(np, slots(xnp, eps.real + d, eps.imag), weights))  # noqa: E731
    fd = (value(h * direction) - value(-h * direction)) / (2 * h)

    assert _rel(g_jax, g_torch) < 1e-12, _rel(g_jax, g_torch)
    assert abs(float(np.sum(g_jax * direction)) - fd) < 1e-6 * max(1.0, abs(fd))
    assert np.abs(g_jax).max() > 0


@pytest.mark.parametrize('nesting', NESTINGS + (None,))
def test_the_raster_builders_pass_the_nesting_through(nesting):
    """The stacked builders and the one-slot grid builders say the same as the per-layer ones.

    ``None`` is the default, and has to be the fixed nesting -- every number measured before
    the choice existed was taken on it.
    """
    eps, Mu, Mv = CASES['five_phase']
    stack = np.stack([eps, eps[::-1].copy()])
    per_layer = [F.li_conv_matrices(xnp, layer, Mu, Mv, nesting=nesting or 'fixed')
                 for layer in stack]
    stacked = F.to_conv_mat_raster_sampled(stack, Mu, Mv, nesting=nesting)
    assert stacked.epxy is None and stacked.epyx is None
    for k in range(3):
        for i in range(2):
            assert np.array_equal(np.asarray(stacked[k])[i], np.asarray(per_layer[i][k]))
    assert np.array_equal(np.asarray(F.fourier2d_piecewise_constant(
        xnp, eps, Mu, Mv, inverse_rule='u', nesting=nesting)), np.asarray(per_layer[0].epx))

    records = [F.compression_indices(layer) for layer in stack]
    stacked = F.to_conv_mat_raster_integrated(stack, Mu, Mv, compression=records,
                                              nesting=nesting)
    for i, (row, col, x, y) in enumerate(records):
        comp = F.compress_cell(xnp, stack[i], row, col)
        one = F.li_conv_matrices_integrated(xnp, comp, x, y, Mu, Mv, nesting=nesting or 'fixed')
        for k in range(3):
            assert np.array_equal(np.asarray(stacked[k])[i], np.asarray(one[k]))
        assert np.array_equal(np.asarray(F.fourier2d_integrated(
            xnp, comp, x, y, Mu, Mv, inverse_rule='u', nesting=nesting)), np.asarray(one.epx))


# ------------------------------------------------------------------------------- the keyword

#: A small crossed grating, and the scene around it. 2.4**2 is spelled to the bit.
_SCENE = dict(backend=0, top=1.0, bot=2.25, theta=0.21, phi=0.37, psi=0.6, wavelength=0.9,
              period=[1.0, 0.8], thickness=[0.35], truncation_order=[4, 3])


def _crossed(N=24):
    eps = np.ones((1, N, N))
    eps[0, N // 6:N // 2, N // 5:3 * N // 4] = 5.76
    eps[0, N // 2:5 * N // 6, N // 5:N // 3] = 5.76
    return eps


def _totals(**kw):
    res = wavebit.rcwa(**{**_SCENE, **kw}).solve()
    return np.asarray(res.R), np.asarray(res.T)


def test_the_default_is_fixed_and_naming_it_changes_nothing():
    """Opt-in: the untouched default *is* ``'fixed'``, to the bit, on both coefficient paths."""
    for coefficients in ('sampled', 'integrated'):
        base = _totals(structure=_crossed(), fourier_coefficients=coefficients)
        named = _totals(structure=_crossed(), fourier_coefficients=coefficients, li_nesting='fixed')
        assert all(np.array_equal(a, b) for a, b in zip(base, named)), coefficients
    assert wavebit.rcwa(**_SCENE, structure=_crossed()).li_nesting == wavebit.Nesting.FIXED


def test_the_symmetric_nesting_moves_a_crossed_grating_and_conserves_energy():
    """A different truncation of the same problem: it has to answer differently, and still
    conserve energy on a lossless layer -- the nesting is a factorization, not a loss."""
    for coefficients in ('sampled', 'integrated'):
        R_f, T_f = _totals(structure=_crossed(), fourier_coefficients=coefficients)
        R_s, T_s = _totals(structure=_crossed(), fourier_coefficients=coefficients,
                           li_nesting='symmetric')
        assert np.abs(T_s - T_f).max() > 1e-5, coefficients
        assert abs(R_s.sum() + T_s.sum() - 1.0) < 1e-10, coefficients


def test_on_a_y_invariant_grating_the_two_nestings_agree():
    """The 1D reduction, through the whole solve: nothing is nested when ``T_v`` is diagonal."""
    rows = np.repeat(_crossed()[:, 5:6, :], 4, axis=1)
    kw = dict(structure=rows, truncation_order=[6, 2])
    for coefficients in ('sampled', 'integrated'):
        fixed = _totals(fourier_coefficients=coefficients, **kw)
        sym = _totals(fourier_coefficients=coefficients, li_nesting='symmetric', **kw)
        for a, b in zip(fixed, sym):
            assert np.abs(a - b).max() < 1e-12, (coefficients, np.abs(a - b).max())


def _tensor(eps):
    """A scalar raster as an in-plane anisotropic tensor raster ``(L, Ny, Nx, 3, 3)``."""
    t = np.zeros(eps.shape + (3, 3), dtype=complex)
    t[..., 0, 0], t[..., 1, 1], t[..., 2, 2] = eps, 1.25 * eps - 0.25, eps
    return t


#: An axis-aligned rectangle, which the parametric path builds with Cartesian Li, and the same
#: rectangle painted on a commensurate raster: in the (1.0, 0.8) cell its edges sit at
#: x = 0.3, 0.8 and y = 0.1, 0.5, which are pixel lines of a 40 x 40 grid.
_RECTANGLE = [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.05, cy=-0.1, lx=0.5,
                                                ly=0.4, material=5.76)])]
_TENSOR_RECTANGLE = [dict(background=1.0, shapes=[dict(
    type='rectangle', cx=0.05, cy=-0.1, lx=0.5, ly=0.4,
    material=wavebit.Material(np.diag([5.76, 4.0, 5.76])))])]


def _painted_rectangle():
    eps = np.ones((1, 40, 40))
    eps[0, 5:25, 12:32] = 5.76
    return eps


_UNREAD = {
    'nvf': (dict(structure=_crossed(), factorization='nvf'),
            "factorization='nvf' nests no rules"),
    'parametric-nvf': (dict(structure=[dict(background=1.0, shapes=[dict(
        type='circle', cx=0.0, cy=0.0, lx=0.4, material=5.76)])]),
        "parametric path's factorization='nvf' nests no rules"),
    'tensor-raster-nvf': (dict(structure=_tensor(_crossed())),
                          "factorization='nvf' nests no rules"),
    'tensor-raster-li': (dict(structure=_tensor(_crossed()), factorization='li'),
                         "tensor factorization='li' nests its rules in the fixed order only"),
    'tensor-rectangle-li': (dict(structure=_TENSOR_RECTANGLE, factorization='li'),
                            "tensor factorization='li' nests its rules in the fixed order only"),
}


@pytest.mark.parametrize('where', sorted(_UNREAD))
def test_a_nesting_nothing_honours_is_refused_not_ignored(where):
    """Every build that would not honour an explicit ``'symmetric'`` refuses it by name, on the
    solve and on ``freeze_geometry``: NVF on either path nests no rules, and tensor Li -- whose
    scenes skip the scalar refusals and go through ``anisotropy.validate_options`` -- builds the
    fixed nesting only. Before these doors a tensor scene built ``'fixed'`` in silence."""
    kw, match = _UNREAD[where]
    solver = wavebit.rcwa(**_SCENE, **kw, li_nesting='symmetric')
    with pytest.raises(NotImplementedError, match=match) as excinfo:
        solver.solve()
    assert "li_nesting='symmetric'" in str(excinfo.value)
    with pytest.raises(NotImplementedError, match='no reader'):
        solver.freeze_geometry()


def test_tensor_li_takes_the_fixed_nesting_it_builds():
    """The refusal is for what tensor Li cannot build: naming the one it does is honoured, to
    the bit, on a raster and on rectangles."""
    for structure in (_tensor(_crossed()), _TENSOR_RECTANGLE):
        base = wavebit.rcwa(**_SCENE, structure=structure, factorization='li').solve()
        named = wavebit.rcwa(**_SCENE, structure=structure, factorization='li',
                             li_nesting='fixed').solve()
        assert np.array_equal(np.asarray(base.T), np.asarray(named.T))


@pytest.mark.parametrize('nesting', NESTINGS)
def test_rectangles_under_li_read_the_nesting_as_the_raster_does(nesting):
    """The parametric path's Cartesian Li runs the raster path's integrated builder on exact
    edges, so it is a reader too: nesting for nesting, its matrices are the painted raster's."""
    shape = wavebit.rcwa(**_SCENE, structure=_RECTANGLE, factorization='li', li_nesting=nesting)
    raster = wavebit.rcwa(**_SCENE, structure=_painted_rectangle(),
                          fourier_coefficients='integrated', li_nesting=nesting)
    got, want = shape.eps_conv_matrices(), raster.eps_conv_matrices()
    for a, b in zip(got[:3], want[:3]):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() < 2e-12, nesting
    assert np.abs(np.asarray(shape.solve().T) - np.asarray(raster.solve().T)).max() < 1e-11


def test_the_nesting_moves_rectangles_and_one_frozen_plan_serves_both():
    """``'symmetric'`` changes the shape build (so it is read, not dropped), and the frozen
    plan -- edge order and ownership only -- replays under either nesting."""
    solver = wavebit.rcwa(**_SCENE, structure=_RECTANGLE, factorization='li')
    plan = solver.freeze_geometry()
    fixed = np.asarray(solver.solve().T)
    solver.li_nesting = 'symmetric'
    replayed = np.asarray(solver.solve(geometry=plan).T)
    rebuilt = np.asarray(wavebit.rcwa(**_SCENE, structure=_RECTANGLE, factorization='li',
                                      li_nesting='symmetric').solve().T)
    assert np.array_equal(replayed, rebuilt)
    assert np.abs(rebuilt - fixed).max() > 1e-5


def test_an_unknown_nesting_is_refused_at_the_keyword():
    with pytest.raises(ValueError, match=r"li_nesting='inner'.*'fixed', 'symmetric'"):
        wavebit.rcwa(**_SCENE, structure=_crossed(), li_nesting='inner')


def test_the_nesting_is_a_solve_condition():
    """``solve(li_nesting=...)`` reaches the build, like every other settable condition."""
    solver = wavebit.rcwa(**_SCENE, structure=_crossed())
    via_solve = np.asarray(solver.solve(li_nesting='symmetric').T)
    direct = _totals(structure=_crossed(), li_nesting='symmetric')[1]
    assert np.array_equal(via_solve, direct)
    assert solver.li_nesting == wavebit.Nesting.SYMMETRIC


def test_one_frozen_geometry_serves_both_nestings():
    """The integrated path's compression says where the blocks are, not how they combine."""
    solver = wavebit.rcwa(**_SCENE, structure=_crossed(), fourier_coefficients='integrated')
    record = solver.freeze_geometry()
    solver.li_nesting = 'symmetric'
    replayed = np.asarray(solver.solve(geometry=record).T)
    rebuilt = _totals(structure=_crossed(), fourier_coefficients='integrated',
                      li_nesting='symmetric')[1]
    assert np.array_equal(replayed, rebuilt)


def test_the_permeability_is_nested_the_same_way():
    """``mu`` goes through the same builders by duality, so it takes the same nesting."""
    eps = _crossed()
    mu = np.ones_like(eps)
    mu[0, 3:9, 2:17] = 2.5
    solver = wavebit.rcwa(**_SCENE, structure=eps, mu=mu, li_nesting='symmetric')
    Mu, Mv = _SCENE['truncation_order']
    want = F.li_conv_matrices(xnp, mu[0], Mu, Mv, nesting='symmetric')
    got = solver.mu_conv_matrices()
    assert _rel(np.asarray(got.epx)[0], want.epx) < 1e-14
    fixed = F.li_conv_matrices(xnp, mu[0], Mu, Mv, nesting='fixed')
    assert _rel(np.asarray(got.epx)[0], fixed.epx) > 1e-6


def test_the_per_call_doors_reach_the_build():
    """``eps_conv_matrices`` / ``mu_conv_matrices`` take ``li_nesting=`` like ``solve`` does,
    and ``freeze_geometry(li_nesting=...)`` is refused where the solve would be."""
    eps = _crossed()
    mu = np.ones_like(eps)
    mu[0, 3:9, 2:17] = 2.5
    Mu, Mv = _SCENE['truncation_order']
    solver = wavebit.rcwa(**_SCENE, structure=eps, mu=mu)
    got = solver.eps_conv_matrices(li_nesting='symmetric')
    assert _rel(np.asarray(got.epx)[0],
                F.li_conv_matrices(xnp, eps[0], Mu, Mv, nesting='symmetric').epx) < 1e-14
    got = solver.mu_conv_matrices(li_nesting='symmetric')
    assert _rel(np.asarray(got.epx)[0],
                F.li_conv_matrices(xnp, mu[0], Mu, Mv, nesting='symmetric').epx) < 1e-14
    nvf = wavebit.rcwa(**_SCENE, structure=eps, factorization='nvf')
    with pytest.raises(NotImplementedError, match="li_nesting='symmetric' has no reader"):
        nvf.freeze_geometry(li_nesting='symmetric')


def test_a_jitted_replay_keeps_the_nesting():
    """The nesting rides in the JAX treedef, so a compiled solve on a frozen record builds the
    nesting the solver holds -- not the default."""
    jax = pytest.importorskip('jax')
    kw = dict(_SCENE, backend=1, structure=_crossed(), fourier_coefficients='integrated')
    solver = wavebit.rcwa(**kw, li_nesting='symmetric')
    geometry = solver.freeze_geometry()
    direct = np.asarray(solver.solve().T)
    compiled = np.asarray(jax.jit(lambda s: s.solve(geometry=geometry).T)(solver))
    np.testing.assert_allclose(compiled, direct, atol=1e-12, rtol=1e-12)
    fixed = np.asarray(wavebit.rcwa(**kw).solve().T)
    assert np.abs(compiled - fixed).max() > 1e-5


def test_the_spec_carries_the_nesting():
    from wavebit import Numerics
    numerics = Numerics.from_dict(dict(truncation_order=[2, 2], li_nesting='symmetric'))
    assert numerics.li_nesting == wavebit.Nesting.SYMMETRIC
    assert numerics.as_kwargs()['li_nesting'] == 'symmetric'
    assert Numerics.from_dict(numerics.to_dict()).li_nesting == wavebit.Nesting.SYMMETRIC
    assert 'li_nesting' not in Numerics.from_dict(dict(truncation_order=[2, 2])).as_kwargs()


def test_the_unscoped_spelling_is_refused_like_any_unknown_option():
    """The option is ``li_nesting``: option names state their scope, as ``nvf_projector``
    and ``raster_normal_method`` do. Nothing has been released, so the bare ``nesting`` is no
    alias -- the constructor, the attribute, the spec and the manifest treat it as a name they
    have never heard of."""
    from wavebit import Numerics, SpecError
    from wavebit.manifest import manifest
    with pytest.raises(TypeError, match=r"unexpected keyword argument.*'nesting'"):
        wavebit.rcwa(**_SCENE, structure=_crossed(), nesting='symmetric')
    solver = wavebit.rcwa(**_SCENE, structure=_crossed())
    with pytest.raises(AttributeError, match=r"no attribute 'nesting' to set"):
        solver.nesting = 'symmetric'
    assert not hasattr(solver, 'nesting')
    assert solver.li_nesting == wavebit.Nesting.FIXED
    with pytest.raises(SpecError, match=r"unknown field\(s\) \['nesting'\]"):
        Numerics.from_dict(dict(truncation_order=[2, 2], nesting='symmetric'))
    for path in manifest()['paths']:
        assert 'li_nesting' in path and 'nesting' not in path


@pytest.mark.parametrize('door', ['solve', 'freeze_geometry', 'eps_conv_matrices',
                                  'mu_conv_matrices'])
def test_the_unscoped_spelling_is_refused_at_every_call_door(door):
    """The per-call doors refuse it by name too, before any condition moves."""
    solver = wavebit.rcwa(**_SCENE, structure=_crossed())
    with pytest.raises(TypeError, match=r"unexpected keyword argument.*'nesting'"):
        getattr(solver, door)(wavelength=1.3, nesting='symmetric')
    assert float(solver.wavelength) == _SCENE['wavelength']
    assert solver.li_nesting == wavebit.Nesting.FIXED
