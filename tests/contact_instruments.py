"""Independent Cartesian Li reference for one centered rectangular inclusion.

Integrate nonuniform material intervals exactly: even a sub-pixel gap is retained.
This instrument bypasses the NVF and its quadrature, while sharing the Maxwell solve.
It is not a production dispatch and does not support tilted or multi-shape scenes.
"""
from wavebit._core import factorization as F
from wavebit._core.domain import as_material
from wavebit._core.lattice import is_axis_aligned


def rectangular_li(xp, shapes, background, hu, hv, lattice, dtype, **kwargs):
    if len(shapes) != 1:
        raise ValueError('the reference requires one rectangle')
    s = shapes[0]
    if s['type'] != 'rectangle' or s.get('angle', 0.) != 0. or s.get('cx', 0.) != 0. or s.get('cy', 0.) != 0.:
        raise ValueError('the reference requires a centered axis-aligned rectangle')
    if not is_axis_aligned(lattice):
        raise ValueError('the reference requires a rectangular cell')
    bg, hi = as_material(background).epsilon, as_material(s['material']).epsilon
    x = xp.stack([.5-s['lx']/(2*lattice[0][0]), .5+s['lx']/(2*lattice[0][0]),
                  xp.asarray(1.)]).reshape((-1, 1))
    y = xp.stack([.5-s['ly']/(2*lattice[1][1]), .5+s['ly']/(2*lattice[1][1]),
                  xp.asarray(1.)]).reshape((-1, 1))
    cell = xp.asarray([[bg,bg,bg], [bg,hi,bg], [bg,bg,bg]], dtype=dtype)
    ex = F.fourier2d_integrated(xp,cell,x,y,hu,hv,inverse_rule='u')
    ey = F.fourier2d_integrated(xp,cell,x,y,hu,hv,inverse_rule='v')
    ez = xp.linalg.inv(F.fourier2d_integrated(xp,cell,x,y,hu,hv,inverse_rule=None))
    return ex,ey,ez,xp.zeros_like(ex),xp.zeros_like(ex)


def split_nvf_derivative(reference_height, component):
    """Return a builder for a partial derivative at one rectangle's height.

    ``material`` varies epsilon/inverse-epsilon coefficients with a fixed normal
    projector; ``normal`` does the reverse. Both evaluate the same forward model
    at reference_height, so their derivatives must add to the full derivative.
    These partial derivatives are diagnostics, not replacement solver gradients.
    """
    if component not in ('material', 'normal'):
        raise ValueError('component must be material or normal')

    def build(xp, shapes, background, hu, hv, lattice, dtype,
              normal_grid=256, plan=None):
        from wavebit._core import parametric as P
        if len(shapes) != 1 or shapes[0]['type'] != 'rectangle':
            raise ValueError('the derivative instrument requires one rectangle')
        fixed = [dict(shapes[0], ly=reference_height)]
        material_shapes = shapes if component == 'material' else fixed
        normal_shapes = shapes if component == 'normal' else fixed
        eps, inv_eps = P.assemble_eps_coeffs(
            xp, material_shapes, background, hu, hv, lattice, dtype)
        xx, xy, yy = P.normal_projector_coeffs(
            xp, normal_shapes, (normal_grid, normal_grid), lattice, hu, hv,
            dtype, plan=plan)
        return P.nvf_from_coeffs(xp, eps, inv_eps, xx, xy, yy, hu, hv)

    return build


def paired_rectangles_li(xp, shapes, background, hu, hv, lattice, dtype, **kwargs):
    """Exact Li instrument for two 300-square boxes side by side in a 700 cell.

    Shapes must be ordered left to right and remain disjoint inside the cell.
    The x coordinates and both materials stay live, including a zero-width gap.
    Keeping the zero-width interval avoids a traced topology-dependent sort.
    """
    if len(shapes) != 2 or not is_axis_aligned(lattice):
        raise ValueError('the reference requires two boxes in a rectangular cell')
    for s in shapes:
        if (s['type'] != 'rectangle' or s['lx'] != 300. or s['ly'] != 300.
                or s.get('cy', 0.) != 0. or s.get('angle', 0.) != 0.):
            raise ValueError('the reference requires centered 300-square boxes')
    px, py = lattice[0][0], lattice[1][1]
    bg = as_material(background).epsilon
    left, right = (as_material(s['material']).epsilon for s in shapes)
    x = xp.stack([.5+(shapes[0]['cx']-150.)/px,
                  .5+(shapes[0]['cx']+150.)/px,
                  .5+(shapes[1]['cx']-150.)/px,
                  .5+(shapes[1]['cx']+150.)/px, xp.asarray(1.)]).reshape((-1, 1))
    y = xp.stack([.5-150./py, .5+150./py, xp.asarray(1.)]).reshape((-1, 1))
    cell = xp.asarray([[bg]*5, [bg, left, bg, right, bg], [bg]*5], dtype=dtype)
    ex = F.fourier2d_integrated(xp, cell, x, y, hu, hv, inverse_rule='u')
    ey = F.fourier2d_integrated(xp, cell, x, y, hu, hv, inverse_rule='v')
    ez = xp.linalg.inv(F.fourier2d_integrated(xp, cell, x, y, hu, hv, inverse_rule=None))
    return ex, ey, ez, xp.zeros_like(ex), xp.zeros_like(ex)
