"""Conforming rectangular NVF cell: polynomial body."""
import numpy as np
from numpy.polynomial import Legendre
from numpy.polynomial import Polynomial

from wavebit.rcwa._engine.conforming.continuous_flux import polynomial_moments
from wavebit.rcwa._engine.conforming.material_grading import MaterialGradedGapModel


def body_space(edges, degree):
    """Periodic P1 nodes plus integrated Legendre bubbles on each y interval."""
    elements = len(edges)-1
    count = elements*degree
    polynomials = np.zeros((degree+1, degree+1))
    polynomials[0, :2], polynomials[1, :2] = [1., -1.], [0., 1.]
    for k in range(1, degree):
        beta = Legendre.basis(k, domain=[0., 1.]).convert(kind=Polynomial).integ().coef
        polynomials[k+1, :len(beta)] = beta
    # Legendre orthogonality keeps exactly zero derivative couplings zero on
    # the tiny contact intervals, avoiding monomial cancellation divided by dy.
    mass, stiffness = np.zeros((degree+1, degree+1)), np.zeros((degree+1, degree+1))
    mass[:2, :2] = np.array([[2., 1.], [1., 2.]])/6
    stiffness[:2, :2] = [[1., -1.], [-1., 1.]]
    mean, first = np.zeros(degree+1), np.zeros(degree+1)
    mean[:2], first[:2] = [.5, .5], [1/6, 1/3]
    for k in range(1, degree):
        mean[k+1] = -1/6 if k == 1 else 0.
        first[k+1] = -1/12 if k == 1 else -1/60 if k == 2 else 0.
        mass[0, k+1] = mass[k+1, 0] = mean[k+1]-first[k+1]
        mass[1, k+1] = mass[k+1, 1] = first[k+1]
        stiffness[k+1, k+1] = 1/(2*k+1)
        for l in range(1, degree):
            mass[k+1, l+1] = sum(sa*sb/(2*a+1) for a, sa in ((k+1, 1), (k-1, -1))
                                  for b, sb in ((l+1, 1), (l-1, -1)) if a == b)/(4*(2*k+1)*(2*l+1))
    indices = np.array([[e, (e+1) % elements,
                         *(elements+e*(degree-1)+np.arange(degree-1))] for e in range(elements)], int)
    matrices = dict(mass=np.zeros((count, count)), stiffness=np.zeros((count, count)),
                    load=np.zeros(count), cross=np.zeros(count), boundary=np.eye(count)[0])
    for e, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        length = right-left
        selection = np.ix_(indices[e], indices[e])
        matrices['mass'][selection] += length*mass
        matrices['stiffness'][selection] += stiffness/length
        matrices['load'][indices[e]] += length*mean
        matrices['cross'][indices[e]] -= length*((left-.5)*mean+length*first)
    return indices, polynomials, matrices


def project_body_polynomial(potential, normal, moments, body, background, xp=np, *, metric_y=1., sesquilinear=False):
    """Integrate the actual polynomial field/energy over all body intervals."""
    def quadratic(matrix, coefficients):
        dual = xp.conj(coefficients) if sesquilinear else coefficients
        return sum(xp.einsum('ne,emcd->nmcd', moments[..., i+j],
                             xp.einsum('mij,eic,ejd->emcd', matrix, dual[:, i], coefficients[:, j]))
                   for i in range(coefficients.shape[1]) for j in range(coefficients.shape[1]))

    def linear(weight, coefficients):
        return sum(xp.einsum('ne,mi,eic->nmc', moments[..., i], weight, coefficients[:, i])
                   for i in range(coefficients.shape[1]))

    field_linear = xp.stack([linear(background['derivative'], potential), linear(background['load'], normal)], axis=-2)
    energy_linear = xp.stack([linear(body['force'], potential), metric_y*linear(body['load'], normal)], axis=-2)
    energy_dual = energy_linear
    if sesquilinear:
        energy_linear = xp.stack([linear(body['reverse_force'], potential),
                                  metric_y*linear(body['load'], normal)], axis=-2)
        energy_dual = xp.stack([linear(body['force'], xp.conj(potential)),
                                metric_y*linear(body['dual_load'], xp.conj(normal))], axis=-2)
    energy = quadratic(body['stiffness'], potential)+metric_y*quadratic(body['mass'], normal)
    scale = xp.sum(moments[..., 0], axis=1)[:, None]
    ex, ey = xp.diag(xp.asarray([1., 0.])), xp.diag(xp.asarray([0., 1.]))
    field = (field_linear+scale[..., None, None]*background['lift'][None, :, None, None]*ex
             +scale[..., None, None]*background['material'][None, :, None, None]*ey)
    material = scale*body['material'][None]
    gram = (energy+energy_linear+xp.swapaxes(energy_dual, -1, -2)
            +(scale*body['lift_squared'][None])[..., None, None]*ex+metric_y*material[..., None, None]*ey)
    return field, gram, material


class PolynomialBodyGapModel(MaterialGradedGapModel):
    """Fixed conforming y bubbles; x material integration remains unchanged."""

    def __init__(self, *args, body_degree=2, **kwargs):
        if not isinstance(body_degree, int) or not 1 <= body_degree <= 4:
            raise ValueError('body_degree must be an integer between one and four')
        super().__init__(*args, **kwargs)
        self.body_degree = body_degree
        if body_degree == 1:
            return
        self.body_indices, self.body_polynomials, matrices = body_space(self.yedges, body_degree)
        self.rows = len(self.yedges[:-1])*body_degree
        self.ymass, self.ystiffness = matrices['mass'], matrices['stiffness']
        self.yload, self.ycross, self.boundary = matrices['load'], matrices['cross'], matrices['boundary']

    def _project_body(self, v, a, g, body, background, xp, *, metric_y=1.):
        if self.body_degree == 1:
            return super()._project_body(v, a, g, body, background, xp, metric_y=metric_y)
        dt = xp.asarray(np.diff(self.yedges))
        t = xp.asarray(self.yedges[:-1])-.5
        potential = xp.einsum('jk,ejic->ekic', xp.asarray(self.body_polynomials), v[self.body_indices])
        ramp = xp.stack([t, dt]+[xp.zeros_like(t)]*(self.body_degree-1), axis=1)
        potential = potential-g*ramp[..., None, None]*a[None, None]
        normal = (xp.arange(1, self.body_degree+1)[None, :, None, None]*potential[:, 1:]
                  /((1-g)*dt[:, None, None, None]))
        moments = polynomial_moments((1-g)*t, (1-g)*dt, xp.zeros_like(t), xp.ones_like(t),
                                      self.modes, xp, degree=2*self.body_degree)
        return project_body_polynomial(potential, normal, moments, body, background, xp,
                                        metric_y=metric_y, sesquilinear=self._complex_materials)

    def coefficients(self, *args, **kwargs):
        c, metadata = super().coefficients(*args, **kwargs)
        return c, dict(**metadata, body_degree=self.body_degree)


class ComplexMaterialGapModel(PolynomialBodyGapModel):
    """Complex trial fields and sesquilinear projection for positive-real profiles.

    Values are relative to the layer background, so their imaginary parts
    may have either sign. The public adapter checks physical passivity before
    normalization. The ninth coefficient is the independent yx energy term;
    the preceding eight indices retain the real kernel's layout.
    """

    _complex_materials = True
