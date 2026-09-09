"""Element-local triangle integration for optional static FEM references."""
import ngsolve as ng
import numpy as np


def triangle_quadrature(mesh, quadrature, *, period, area_fraction, batch_size=24000):
    """Yield element-owned points and normalized Duffy weights in batches.

    Global point location can choose a neighboring material for nodes very
    close to a fitted interface. Each point here retains its known element
    through the element transformation, including its coefficient evaluation.
    """
    if not isinstance(quadrature, int) or quadrature < 1 or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('quadrature and batch size must be positive integers')
    if not np.isfinite([period, area_fraction]).all() or period <= 0 or area_fraction <= 0:
        raise ValueError('period and area fraction must be finite and positive')
    elements = list(mesh.Elements(ng.VOL))
    coordinates = np.array([[mesh[v].point for v in element.vertices] for element in elements])
    if coordinates.shape[1:] != (3, 2):
        raise ValueError('projection requires straight triangular elements')
    a = coordinates[:, 0]
    edge1, edge2 = coordinates[:, 1]-a, coordinates[:, 2]-a
    determinant = abs(edge1[:, 0]*edge2[:, 1]-edge1[:, 1]*edge2[:, 0])
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    nodes, weights = (nodes+1)/2, weights/2
    u, v = np.meshgrid(nodes, nodes, indexing='ij')
    wu, wv = np.meshgrid(weights, weights, indexing='ij')
    reference_weights = (wu*wv*(1-u)).ravel()/period**2
    if not np.isclose(determinant.sum()*reference_weights.sum(), area_fraction, rtol=0, atol=1e-12):
        raise ValueError('invalid cell integration weights')
    # NGSolve's triangle coordinates are the weights of vertices 0 and 1;
    # vertex 2 has weight 1-xi-eta. Preserve the previous physical Duffy nodes.
    rule = ng.IntegrationRule(points=list(zip((1-u-(1-u)*v).ravel(), u.ravel())))
    stride = max(1, batch_size//quadrature**2)
    for start in range(0, len(elements), stride):
        mapped = np.concatenate([mesh.GetTrafo(element)(rule) for element in elements[start:start+stride]])
        xy = np.column_stack([ng.x(mapped).ravel(), ng.y(mapped).ravel()])
        weights = (determinant[start:start+stride, None]*reference_weights).ravel()
        yield mapped, xy, weights
