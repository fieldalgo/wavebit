"""Contact shape derivatives from independent trace and body weak problems.

Reconstruct the integrated flux basis by piecewise physical quadrature, then
solve the limiting trace and the first body variation separately. This oracle
uses no production weak assembly, Fourier moments, projection, or autodiff.
The geometry helper only supplies the finite space to be checked. Compensated
weak products and residuals control roundoff on the smallest graded elements.
"""
from fractions import Fraction

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.linalg import lu_factor, lu_solve
from scipy.special import eval_legendre
from wavebit._core.conforming.affine_cell import AffineCellGapModel

pytestmark = pytest.mark.research


def x_samples(model, profile, width, quadrature):
    """Integrate periodic flux hats directly across mesh and material cuts."""
    _, edges, values, nodes, lengths, _ = model._profile_geometry(0., profile, width, np)
    seams = edges[1:-1]
    count = len(nodes)
    q, qw = leggauss(quadrature)
    cumulative = np.zeros(count, dtype=complex)
    xs, ws, es, fs, dfs = [], [], [], [], []
    cuts = np.unique(np.r_[nodes, .5, seams])
    for a, b in zip(cuts[:-1], cuts[1:]):
        middle = (a+b)/2
        element = np.searchsorted(nodes, middle, side='right')-1
        epsilon = values[np.searchsorted(seams, middle)] if element//model.per_region == 1 else 1.
        x = middle+(b-a)/2*q
        left = np.eye(count)[element]
        change = np.eye(count)[(element+1) % count]-left
        length = lengths[element]
        ra, rb = (a-nodes[element])/length, (b-nodes[element])/length
        # Form the local coordinate before adding the physical origin. Recovering
        # it from x-node loses digits on the smallest graded elements.
        r = (ra+rb)/2+(rb-ra)/2*q
        fs.extend(cumulative+length/epsilon*(np.outer(r-ra, left)+np.outer((r*r-ra*ra)/2, change)))
        dfs.extend((left+np.outer(r, change))/epsilon)
        cumulative += length/epsilon*((rb-ra)*left+(rb*rb-ra*ra)/2*change)
        xs.extend(x)
        ws.extend((b-a)/2*qw)
        es.extend(np.full(x.shape, epsilon))
    x, w, eps, f, df = map(np.asarray, (xs, ws, es, fs, dfs))
    lift = 1/eps/cumulative.sum()
    pivot = model.per_region//2 if model.flux_basis == 'local' else model.per_region+model.per_region//2
    keep = np.arange(count) != pivot
    if model.flux_basis == 'local':
        first = x < nodes[1]
        origin = np.sum(w[first]*df[first, 0])/cumulative[0]
        offsets = (1-origin)*np.eye(count)[0]+origin*np.eye(count)[-1]
        f, df = f/cumulative, df/cumulative
        f, df = (np.roll(f, -1, axis=1)-f-offsets)[:, keep], (np.roll(df, -1, axis=1)-df)[:, keep]
    else:
        f = (f/cumulative-f.sum(axis=1)[:, None]/cumulative.sum())[:, keep]
        df = (df/cumulative-lift[:, None])[:, keep]
    return x, w, eps, lift, np.column_stack([np.ones(len(x)), f]), np.column_stack([np.zeros(len(x)), df])


def y_samples(model, quadrature):
    """Evaluate periodic nodal functions and integrated Legendre bubbles."""
    q, w = leggauss(quadrature)
    r = (q+1)/2
    elements = len(model.yedges)-1
    h = np.zeros((elements*quadrature, model.rows))
    dh = np.zeros_like(h)
    times, weights = [], []
    for e, (left, right) in enumerate(zip(model.yedges[:-1], model.yedges[1:])):
        selection = slice(e*quadrature, (e+1)*quadrature)
        h[selection, e] += 1-r
        h[selection, (e+1) % elements] += r
        dh[selection, e] -= 1/(right-left)
        dh[selection, (e+1) % elements] += 1/(right-left)
        for k in range(1, model.body_degree):
            index = elements+e*(model.body_degree-1)+k-1
            h[selection, index] = (eval_legendre(k+1, q)-eval_legendre(k-1, q))/(2*(2*k+1))
            dh[selection, index] = eval_legendre(k, q)/(right-left)
        times.extend(left+(right-left)*r-.5)
        weights.extend((right-left)*w/2)
    return np.array(times), np.array(weights), h, dh


def compensated_dot(a, b):
    """Kahan accumulation of the unchanged weak inner products."""
    a, b = np.asarray(a), np.asarray(b)
    vector = a.ndim == 1
    if vector:
        a = a[None]
    total = np.zeros((a.shape[0], b.shape[1]), dtype=np.result_type(a, b))
    correction = np.zeros_like(total)
    for k in range(a.shape[1]):
        term = a[:, k, None]*b[k, None, :]-correction
        updated = total+term
        correction = (updated-total)-term
        total = updated
    return total[0] if vector else total


def product_with_error(a, b):
    """Split a real double product into its rounded value and rounding error.

    Oracle matrices are bounded finite doubles, away from overflow/underflow;
    this arithmetic helper is not a general extended-precision array backend.
    """
    product = a*b
    split_a, split_b = 134217729.*a, 134217729.*b
    a_hi, b_hi = split_a-(split_a-a), split_b-(split_b-b)
    a_lo, b_lo = a-a_hi, b-b_hi
    return product, ((a_hi*b_hi-product)+a_hi*b_lo+a_lo*b_hi)+a_lo*b_lo


def compensated_residual(matrix, solution, right):
    """Accumulate right-M solution with compensated real products and sums."""
    result = right.copy()
    for column in range(solution.shape[1]):
        totals = [np.zeros(matrix.shape[0]) for _ in range(2)]
        errors = [np.zeros(matrix.shape[0]) for _ in range(2)]
        for k in range(matrix.shape[1]):
            a, b = matrix[:, k], solution[k, column]
            ac, ace = product_with_error(a.real, b.real)
            bd, bde = product_with_error(a.imag, b.imag)
            ad, ade = product_with_error(a.real, b.imag)
            bc, bce = product_with_error(a.imag, b.real)
            for part, high_terms, low in ((0, (ac, -bd), ace-bde), (1, (ad, bc), ade+bce)):
                for term in high_terms:
                    old = totals[part]
                    updated = old+term
                    remainder = updated-old
                    errors[part] += (old-(updated-remainder))+(term-remainder)
                    totals[part] = updated
                errors[part] += low
        result[:, column] = ((right[:, column].real-totals[0])-errors[0]
                             +1j*((right[:, column].imag-totals[1])-errors[1]))
    return result


def balanced_solve(matrix, right):
    """One balanced LU, refined against the original independent weak system."""
    scale = 1/np.sqrt(abs(np.diag(matrix)))
    factor = lu_factor(scale[:, None]*matrix*scale[None])
    solution = scale[:, None]*lu_solve(factor, scale[:, None]*right)
    for _ in range(3):
        defect = compensated_residual(matrix, solution, right)
        solution += scale[:, None]*lu_solve(factor, scale[:, None]*defect)
    return solution


def contact_derivative(model, profile, width, basis, quadrature=24):
    """Return d(coefficients)/dg at contact, with dimensionless gap g."""
    basis = basis/np.linalg.norm(basis[:, 0])
    inverse = np.linalg.inv(basis)
    metric = inverse@inverse.T
    x, wx, epsilon, lift, f, df = x_samples(model, profile, width, quadrature)
    t, wy, h, dh = y_samples(model, quadrature)
    base = np.tile(np.eye(2, dtype=complex), (len(x), 1, 1))
    base[:, 0, 0] = lift
    base[:, 0, 1] = metric[0, 1]/metric[0, 0]*(lift-1)
    flux = metric@base
    mass = compensated_dot(f.conj().T*wx, f)
    forcing = compensated_dot(f.conj().T, (wx*(epsilon-1))[:, None]*flux[:, 1])
    a = balanced_solve(metric[1, 1]*mass, forcing)
    trace = f@a
    # Differentiated periodic body problem, with the known linear trace ramp.
    matrix = np.zeros((model.rows*model.nx,)*2, complex)
    right = np.zeros((model.rows, model.nx, 2), complex)
    for i, hy in enumerate((h, dh)):
        for j, ky in enumerate((h, dh)):
            fx, kx = (df, f)[i], (df, f)[j]
            pack = compensated_dot(fx.conj().T*(wx*epsilon), kx)
            matrix += metric[i, j]*np.kron(compensated_dot(hy.T*wy, ky), pack)
            weight = wy*(t if j == 0 else 1)
            right += metric[i, j]*compensated_dot(weight, hy)[:, None, None]*(pack@a)[None]
    gap_flux = flux[:, 0]+metric[0, 1]*trace
    right[0] -= compensated_dot(df.conj().T, wx[:, None]*gap_flux)
    solution = np.zeros((model.rows*model.nx, 2), complex)
    solution[1:] = balanced_solve(matrix[1:, 1:], right.reshape(-1, 2)[1:])
    v = solution.reshape(model.rows, model.nx, 2)
    derivative = np.stack([np.einsum('yr,ric,xi->yxc', h, v, df)-t[:, None, None]*(df@a)[None],
                           np.einsum('yr,ric,xi->yxc', dh, v, f)-trace[None]], axis=-2)
    gram_derivative = epsilon[None, :, None, None]*(np.swapaxes(derivative.conj(), -1, -2)@metric@base
                                                   +np.swapaxes(base.conj(), -1, -2)@metric@derivative)
    gap_field = base.copy()
    gap_field[:, 1] += trace
    gap_energy = np.swapaxes(gap_field.conj(), -1, -2)@metric@gap_field
    body_energy = epsilon[:, None, None]*(np.swapaxes(base.conj(), -1, -2)@metric@base)
    xwave = wx[None]*np.exp(-2j*np.pi*model.modes[:, None]*x[None])
    ywave = wy[None]*np.exp(-2j*np.pi*model.modes[:, None]*t[None])
    boundary_phase = (-1.)**model.modes
    field = (np.einsum('ny,mx,yxcd->nmcd', ywave, xwave, derivative)
             +boundary_phase[:, None, None, None]*np.einsum('mx,xcd->mcd', xwave, gap_field-base)[None])
    gram = (np.einsum('ny,mx,yxcd->nmcd', ywave, xwave, gram_derivative)
            +boundary_phase[:, None, None, None]*np.einsum('mx,xcd->mcd', xwave, gap_energy-body_energy)[None])
    field = inverse.T@field@basis.T
    gram = basis@gram@basis.T
    eps = boundary_phase[:, None]*(xwave@(1-epsilon))[None]
    return np.stack([field[..., 0, 0], field[..., 0, 1], field[..., 1, 0], field[..., 1, 1],
                     gram[..., 0, 0], gram[..., 0, 1], gram[..., 1, 1], eps, gram[..., 1, 0]])


def test_compensated_complex_residual_matches_exact_rational_cancellation():
    small, large = 2.**-27, 2.**53
    matrix = np.array([[1+small, 1, -1, 0],
                       [0, large+1j, 1+1j, -large+1j],
                       [1j*(1+small), -1j, 0, 0]], complex)
    solution = np.outer([1-small, 1, 1, 1], [1, 1j])
    right = np.outer([1, 0, 0], [1, 1j])
    expected = np.empty(right.shape, complex)
    # Exact rational arithmetic on the original double operands independently
    # retains both a sub-ulp product term and an order-one cancelled sum.
    for row in range(len(matrix)):
        for column in range(solution.shape[1]):
            real, imag = Fraction(float(right[row, column].real)), Fraction(float(right[row, column].imag))
            for a, b in zip(matrix[row], solution[:, column]):
                ar, ai = Fraction(float(a.real)), Fraction(float(a.imag))
                br, bi = Fraction(float(b.real)), Fraction(float(b.imag))
                real -= ar*br-ai*bi
                imag -= ar*bi+ai*br
            expected[row, column] = complex(float(real), float(imag))
    np.testing.assert_array_equal(expected, np.outer([2.**-54, -1-3j, 1j*2.**-54], [1, 1j]))
    np.testing.assert_array_equal(compensated_residual(matrix, solution, right), expected)


@pytest.mark.parametrize('profile,body_degree,gap_degree,grade_y,shear,batch_size', [
    (4.+.8j, 2, 2, False, .45, 2),
    (dict(seams=[31.], values=[4.+.8j, 9.+2.1j]), 4, 6, False, .45, 16),
    (dict(seams=[-73., 31.], values=[4., 9., 6.25]), 1, 4, False, -.8, 16),
    (4., 2, 1, True, -.8, 16),
])
def test_contact_fourier_opening_derivative_matches_independent_weak_variation(
        profile, body_degree, gap_degree, grade_y, shear, batch_size, monkeypatch):
    # Five Fourier labels cross three batches in the first contact-opening case.
    monkeypatch.setattr('wavebit._core.conforming.affine_cell._PROJECTION_BATCH_SIZE', batch_size)
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = AffineCellGapModel(1, rows=2, body_degree=body_degree, gap_degree=gap_degree,
                               grade_y=grade_y, flux_basis='local')
    basis = np.array([[1., shear], [0., .7]])
    expected = contact_derivative(model, profile, 350., basis, 16)
    refined = contact_derivative(model, profile, 350., basis, 24)
    np.testing.assert_allclose(expected, refined, atol=3e-10, rtol=0)

    def coefficients(g):
        return model.coefficients(700*g, profile, xp=jnp, basis=jnp.asarray(basis))[0]

    _, derivative = jax.jit(lambda g: jax.jvp(coefficients, (g,), (jnp.ones_like(g),)))(jnp.asarray(0.))
    np.testing.assert_allclose(derivative, refined, atol=3e-9, rtol=0)
    jax.clear_caches()
