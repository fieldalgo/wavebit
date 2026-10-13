"""Complex weak equations checked by independent physical tensor quadrature."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre

from wavebit.rcwa._engine.conforming.continuous_flux import ContinuousFluxGapModel
from wavebit.rcwa._engine.conforming.affine_cell import AffineCellGapModel

pytestmark = pytest.mark.research


@pytest.mark.parametrize('gap_degree', [1, 2, 4])
@pytest.mark.parametrize('g', [.07, .31])
@pytest.mark.parametrize('cell', ['diagonal', 'affine', 'affine-contact-factor'])
def test_complex_cell_solve_satisfies_independently_assembled_weak_equations(gap_degree, g, cell):
    affine, contact_factor = cell != 'diagonal', cell == 'affine-contact-factor'
    cls, settings = (AffineCellGapModel, dict(body_degree=1)) if affine else (ContinuousFluxGapModel, {})
    model = cls(0, rows=3, grade_y=False, gap_degree=gap_degree, quadrature=24, **settings)
    # A small periodic complex basis isolates the weak-system assembly from
    # the material-adapted x integral implementation and its mesh density.
    model.nx = 3
    q, weights = leggauss(64)
    x, wx = q/2, weights/2
    change = np.array([[1., .1+.2j, 0.], [0., 1.+.3j, .2j], [0., .1j, 1.-.2j]])
    f = np.column_stack([np.ones_like(x), np.cos(2*np.pi*x), np.sin(2*np.pi*x)])@change
    df = np.column_stack([np.zeros_like(x), -2*np.pi*np.sin(2*np.pi*x), 2*np.pi*np.cos(2*np.pi*x)])@change
    lift = 1.+.17j*np.cos(2*np.pi*x)
    epsilon = 2.+.8*np.cos(2*np.pi*x)+1j*(.6+.2*np.sin(2*np.pi*x))
    if contact_factor:
        # Factoring the body correction by g uses the exact laminated lift.
        # Its weak x load vanishes for every periodic test basis.
        lift = (1/epsilon)/np.sum(wx/epsilon)
    metric_y = 2.4
    metric = np.array([[1., -.4], [-.4, metric_y]]) if affine else np.diag([1., metric_y])
    def pack(material):
        w = wx*material
        if affine:
            return dict(mass=(f.conj().T*w)@f, stiffness=(df.conj().T*w)@df,
                        mixed=(df.conj().T*w)@f, reverse_mixed=(f.conj().T*w)@df,
                        dual_load=w@f.conj(), dual_derivative=w@df.conj(),
                        force=(w*lift)@df.conj(), potential_force=(w*lift)@f.conj())
        return (metric_y*(f.conj().T*w)@f, (df.conj().T*w)@df,
                metric_y*w@f.conj(), (w*lift)@df.conj())
    if affine:
        v, a, b, residual = model._solve_metric_cell(g, pack(epsilon), pack(np.ones_like(x)), metric, np,
                                                   contact_factor=contact_factor)
    else:
        v, a, b, residual = model._solve_flux_cell(g, *pack(epsilon), *pack(np.ones_like(x)), np)
    assert residual < 1e-12
    solution = np.concatenate([v.reshape(-1, 2), a, b.reshape(-1, 2)])[1:]
    count = len(solution)
    matrix, right = np.zeros((count, count), complex), np.zeros((count, 2), complex)

    def accumulate(gx, gy, material, wy):
        nonlocal matrix, right
        gradient = np.stack([gx, gy], axis=2)[..., 1:]
        w = wy[:, None]*wx[None]*material[None]
        matrix += np.einsum('yx,yxci,cd,yxdj->ij', w, gradient.conj(), metric, gradient)
        affine_field = np.zeros((*w.shape, 2, 2), complex)
        affine_field[:, :, 0, 0], affine_field[:, :, 1, 1] = lift[None], 1.
        affine_field[:, :, 0, 1] = metric[0, 1]/metric[0, 0]*(lift[None]-1)
        right -= np.einsum('yx,yxci,cd,yxdj->ij', w, gradient.conj(), metric, affine_field)

    h, dh, t = model.yvalue, model.yderivative, model.yt
    vx = np.einsum('yr,xi->yxri', h, df).reshape(len(t), len(x), -1)
    vy = np.einsum('yr,xi->yxri', dh, f).reshape(len(t), len(x), -1)/(1-g)
    ax = -g*t[:, None, None]*df[None]
    ay = np.broadcast_to(-g/(1-g)*f[None], ax.shape)
    zeros = np.zeros((len(t), len(x), (gap_degree-1)*model.nx))
    accumulate(np.concatenate([vx, ax, zeros], axis=-1), np.concatenate([vy, ay, zeros], axis=-1),
               epsilon, (1-g)*model.yw)

    r, wr = (q+1)/2, weights/2
    beta = np.array([(eval_legendre(k+1, q)-eval_legendre(k-1, q))/(2*(2*k+1))
                     for k in range(1, gap_degree)]).reshape(gap_degree-1, len(r)).T
    dbeta = np.array([eval_legendre(k, q) for k in range(1, gap_degree)]).reshape(gap_degree-1, len(r)).T
    vx = np.broadcast_to(np.einsum('r,xi->xri', model.boundary, df).reshape(len(x), -1)[None],
                          (len(r), len(x), model.rows*model.nx))
    ax, ay = g*(r-.5)[:, None, None]*df[None], np.broadcast_to(f[None], (len(r), *f.shape))
    bx = g*np.einsum('yb,xi->yxbi', beta, df).reshape(len(r), len(x), -1)
    by = np.einsum('yb,xi->yxbi', dbeta, f).reshape(len(r), len(x), -1)
    accumulate(np.concatenate([vx, ax, bx], axis=-1),
               np.concatenate([np.zeros_like(vx), ay, by], axis=-1), np.ones_like(x), g*wr)

    # The production system analytically divides enrichment test rows by g.
    # Physical quadrature assembles the undivided equations independently.
    assert np.linalg.norm(matrix@solution-right)/np.linalg.norm(right) < 2e-12
    np.testing.assert_allclose(solution, np.linalg.solve(matrix, right), atol=3e-12, rtol=0)
