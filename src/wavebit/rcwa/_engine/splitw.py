"""What the split rules read off a structure, for every door that reconstructs a field.

Across a wall it is the normal ``D`` and the tangential ``E`` that are continuous, so the rule
splits the in-plane field with the wall's own normal ``n`` and sums each part's series where it
is continuous (M. Weismann, D. F. G. Gallagher and N. C. Panoiu, J. Opt. 17, 125612 (2015))::

    D_n = 1/2 (P Q + Q P) E,   E_t = (I - P) E,   E(r) = sum(D_n)(r) / eps(r) + sum(E_t)(r)

``P`` is the convolution matrix of ``w n n^T`` and ``Q = [[1/eps]]^-1`` the inverse rule, floored
as the NVF solve floors it. There are two rules, one for each ``w``.

``'split'`` takes ``w = 1``: the circle's radial normal over the whole cell. That field is right
on the wall and jumps where nothing needs it, at the circle's axis and on the cell's edge.

``'splitw'`` windows it: ``w`` is 1 within ``a`` of the wall and falls as ``cos^2`` to 0 at ``b``,
so the radial normal is used only near the wall. Where ``w < 1`` the rest of the normal field
stays in the plain sum, in one material. ``b`` is the smaller of the radius and half the gap to
the nearest image's wall, and ``a = b / 4``.

**Where the permittivity changes sign across the wall the rules answer on its negative side
only**, and the other side keeps the plain sum (:func:`_applies`). There the solve's high
orders carry an error the inverse rule amplifies; divided by a metal's ``eps`` it is small, and
on the dielectric side it is not. Against the finite-element reference of six metal circles the
rules were 2.4 to 3.9 times closer than the sum inside the metal and further than it outside,
at every distance from the wall.

Both answer what they were measured on, in a rectangular cell: a layer holding one circle, one
ellipse along the cell's axes, or several circles of one material that do not touch, of a scalar
permittivity. :func:`refusal` names everything else. Like the Lalanne rule
(:mod:`~wavebit.rcwa._engine.lalanne`) they read the solve's own copy of the structure, never
the live one.

**One circle and the other layers are built differently, and windowed differently.** One circle
keeps what it was first measured with: the radial normal, sampled with the circle at the cell's
centre and carried by a phase, and ``b`` the smaller of the radius and half the gap to the
nearest image. An ellipse or several circles take the solve's own analytic normal field of the
shapes (:func:`~wavebit.rcwa._engine.parametric.normal_vector_field_analytical`) and a window
on the distance to the nearest wall, with ``b`` the smallest inradius of the shapes: the
radius of a circle, the shorter semi-axis of an ellipse. With no window, or with the first
``b``, that construction gives one circle the field it has here to four digits on sixteen
cells. The second ``b`` is the one that held on the ellipse and the two circles measured (it
is never behind ``'split'`` there, where the first is at order 8); on one circle it is closer
on five of the sixteen cells and 10 to 29% further on one, so one circle keeps the first.
"""

from collections import namedtuple

import numpy as np

from wavebit._common.backend import as_array
from wavebit._common.backend import carries_autodiff
from wavebit._common.geometry.recipes import shape_geometry
from wavebit.lattice import periods_of
from wavebit.lattice import sample_grid
from wavebit.materials.response import as_material
from wavebit.materials.tensors import tensor_shape
from wavebit.rcwa._engine.factorization import INVERSE_RULE_BOUND
from wavebit.rcwa._engine.factorization import _floor_singular_values
from wavebit.rcwa._engine.factorization import inverse_rule_bound
from wavebit.rcwa._engine.fields import SplitHalf
from wavebit.rcwa._engine.fields import _axis_samples
from wavebit.rcwa._engine.lalanne import LayerRule
from wavebit.rcwa._engine.lalanne import _device_of
from wavebit.rcwa._engine.lalanne import _host
from wavebit.rcwa._engine.lalanne import _kept
from wavebit.rcwa._engine.lalanne import _masked
from wavebit.rcwa._engine.lalanne import _negative_side
from wavebit.rcwa._engine.lalanne import _positive_sides
from wavebit.rcwa._engine.line_crossings import is_magnetic
from wavebit.rcwa._engine.parametric import _ellipse_distance
from wavebit.rcwa._engine.parametric import _grid
from wavebit.rcwa._engine.parametric import _hypot_safe
from wavebit.rcwa._engine.parametric import _sampled_coeffs
from wavebit.rcwa._engine.parametric import assemble_eps_coeffs
from wavebit.rcwa._engine.parametric import conv_matrix_2d
from wavebit.rcwa._engine.parametric import layer_epsilons
from wavebit.rcwa._engine.parametric import normal_vector_field_analytical


#: Points per axis the windowed projector ``w n n^T`` is sampled on before its transform. The
#: field is once differentiable, so its coefficients converge at about the third order of the
#: grid: against 4096 points, the in-plane field over a plane moves 1.5e-8 of its RMS at 512 and
#: 1e-10 at 2048 for a radius-100 circle in a 400 cell (orders 8 and 12), and 1e-5 and 1.5e-7
#: where the window is narrow, a radius-470 circle in a 1000 cell. 2048 costs about 1.5 s of
#: each reconstruction on four cores, 1024 a fifth of that.
WINDOW_GRID = 2048

_CIRCLES = ('circle', 'disk')

#: Points per axis for an ellipse or several circles, whose normal field is the solve's analytic
#: one, evaluated shape by shape: the grid their measurements were taken on.
SHAPES_GRID = 1024

#: One layer's geometry as the rule reads it: the circle's centre from the cell's corner, its
#: radius, the cell's two sides and the window's reach ``b``.
_Circle = namedtuple('_Circle', 'cx cy radius side_x side_y reach')


def refusal(kind, structure, lattice, kx, ky, mu=None, merged=None, rule='splitw'):
    """Why ``rule``, ``'split'`` or ``'splitw'``, cannot answer this solve, by name, or ``None``.

    The other arguments are :func:`wavebit.rcwa._engine.lalanne.refusal`'s. What is refused is
    what the rules were not measured on, against a 3D finite-element reference, rather than what
    their algebra cannot be written for: shapes other than a circle, several shapes in a layer,
    a raster, a tensor, a patterned permeability.
    """
    name = f'reconstruction={rule!r}'
    if kind in ('te', 'tm', 'conical'):
        return (f"{name} splits the in-plane field along the normal of a curved wall in a 2D "
                "cell. A 1D grating's walls all face x, where that split is the line rule: use "
                "'lalanne'.")
    skewed = not getattr(lattice, 'axis_aligned', False) if kind == 'source' else np.ndim(kx) != 1
    if skewed or _negative_side(lattice):
        return (f'{name} is built for a rectangular cell with positive sides; a skewed lattice, '
                'one written with a negative side, and a lattice differentiated as a whole '
                "matrix are refused. Keep 'plain'.")
    if structure is None:
        return (f'{name} reads the wall and its normal off the structure, and solve_conv solved '
                'matrices that need not be that structure. Use solve().')
    if not isinstance(structure, list):
        return (f"{name} reads the wall's normal off a shape, and a raster names none. Give the "
                "circle as a shape and solve it with factorization='nvf', or keep 'plain'; "
                "'lalanne' takes a raster's walls along x and y.")
    if mu is not None:
        return f"{name} rebuilds E alone, and this solve carries a permeability field. Keep 'plain'."
    for index in _kept(merged, len(structure)):
        message = _layer_refusal(structure[index], lattice, index, name)
        if message is not None:
            return message
    return None


def _one_circle(layer):
    """Is the layer one circle, the case that keeps its first construction (:func:`_projector`)?"""
    shapes = layer['shapes']
    return len(shapes) == 1 and shapes[0].get('type', 'rectangle') in _CIRCLES


def _layer_refusal(layer, lattice, index, name):
    """One layer's refusal: anything patterned other than what the module's docstring accepts."""
    shapes = layer.get('shapes')
    if not shapes:
        return None
    kinds = [shape.get('type', 'rectangle') for shape in shapes]
    circles, ellipse = all(kind in _CIRCLES for kind in kinds), kinds == ['ellipse']
    if not (circles or ellipse):
        held = 'holds ' + ', '.join(f"{kinds.count(k)} {k}{'s' * (kinds.count(k) > 1)}" for k in dict.fromkeys(kinds))
        return (f'{name} is measured on a layer that holds circles, or one ellipse along the '
                f"cell's axes, whose normal is the solve's own; layer {index} {held}. Its normal "
                'near a corner, or between an ellipse and another shape, has no measurement '
                "behind it. Keep 'plain', or 'lalanne' where every wall is along x or y.")
    if is_magnetic(layer):
        return (f'{name} rebuilds E alone: its reference stored no H, so a patterned '
                f"permeability, which layer {index} names, has no measurement behind it. Keep "
                "'plain'.")
    values = [as_material(item).epsilon for item in [layer.get('background', 1.0)] + [
        shape.get('material', 1.0) for shape in shapes]]
    if any(tensor_shape(value) != () for value in values):
        return (f'{name} divides D_n by a scalar eps, and layer {index} holds a tensor. '
                "Keep 'plain'.")
    materials = [_host(value) for value in values[1:]]
    if None not in materials and any(not np.all(value == materials[0]) for value in materials[1:]):
        return (f'{name} is measured on circles of one material, and the circles of layer '
                f"{index} are of several. Keep 'plain'.")
    geometry = [shape_geometry(shape, lattice) for shape in shapes]
    numbers = [[_host(value) for value in row[1:]] for row in geometry]
    sides = [_host(p) for p in periods_of(lattice)]
    if None in sides or any(None in row for row in numbers):
        return None                 # a trace hides the fit; the samples then answer NaN where it fails
    sides = [float(np.real(side)) for side in sides]
    numbers = [[float(np.real(value)) for value in row] for row in numbers]
    if ellipse and numbers[0][4] != 0:
        return (f'{name} is measured on an ellipse along the axes of the cell, and the ellipse '
                f"of layer {index} is turned. Keep 'plain'.")
    for _, _, lx, ly, _ in numbers:
        if ellipse and not (0 < lx < sides[0] and 0 < ly < sides[1]):
            return (f'{name} reads the normal between the wall and the nearest image of it, and '
                    f'the ellipse of layer {index} leaves no room: an axis is not below the '
                    "cell's side along it. Keep 'plain'.")
        if not ellipse and not 0 < lx < min(sides):
            return (f'{name} reads the normal between the wall and the nearest image of it, and '
                    f"a circle of layer {index} leaves no room: its diameter is not below the "
                    "cell's shorter side. Keep 'plain'.")
    for i, (cx, cy, lx, _, _) in enumerate(numbers):
        for ox, oy, other, _, _ in numbers[i + 1:]:
            dx, dy = cx - ox, cy - oy
            dx, dy = dx - sides[0] * round(dx / sides[0]), dy - sides[1] * round(dy / sides[1])
            if np.hypot(dx, dy) <= 0.5 * (lx + other):
                return (f'{name} reads the normal between two walls, and two circles of layer '
                        f"{index} touch or overlap. Keep 'plain'.")
    return None


def _circle(xp, layer, lattice, real, dev):
    """The layer's circle and cell as ``real`` scalars of ``xp``, on the derivative's path."""
    _, cx, cy, diameter, _, _ = shape_geometry(layer['shapes'][0], lattice)
    # At the solve's precision in one step: a Python float cast afterwards has already been
    # rounded to the backend's default, float32 on PyTorch.
    cx, cy, diameter, side_x, side_y = (
        as_array(xp, value, dtype=real, device=dev) for value in (cx, cy, diameter, *periods_of(lattice)))
    radius = 0.5 * diameter
    reach = xp.minimum(radius, 0.5 * xp.minimum(side_x, side_y) - radius)
    return _Circle(cx, cy, radius, side_x, side_y, reach)


def _window(xp, distance, reach):
    """``'splitw'``'s window: 1 within ``reach / 4`` of the wall, falling as ``cos^2`` to 0 at ``reach``."""
    inner = 0.25 * reach
    ramp = (distance - inner) / (reach - inner)
    ramp = xp.minimum(xp.maximum(ramp, xp.zeros_like(ramp)), xp.ones_like(ramp))
    return xp.cos(0.5 * np.pi * ramp) ** 2


def _projector(xp, circle, lattice, half_u, half_v, real, C, grid, windowed):
    """The convolution matrices of ``w n n^T``: ``(xx, xy, yy)``, in the kernels' frame.

    Sampled with the circle at the cell's centre and carried to its place by a phase, as the NVF
    solve builds its own projector (:func:`~wavebit.rcwa._engine.parametric.
    normal_projector_coeffs`), so a circle moved across the cell gives the same field moved.
    """
    X, Y = sample_grid(xp, (grid, grid), lattice, real)
    dx, dy = X - 0.5 * circle.side_x, Y - 0.5 * circle.side_y
    rho = _hypot_safe(xp, dx, dy)
    nx, ny = dx / rho, dy / rho
    w = _window(xp, xp.abs(rho - circle.radius), circle.reach) if windowed else xp.ones_like(rho)
    KKY, KKX = _grid(xp, half_u, half_v, _device_of(X), real)
    shift_u = (circle.cx - 0.5 * circle.side_x) / circle.side_x
    shift_v = (circle.cy - 0.5 * circle.side_y) / circle.side_y
    phase = xp.exp(xp.astype(KKY * shift_v + KKX * shift_u, C) * (-2j * np.pi))
    return tuple(conv_matrix_2d(xp, _sampled_coeffs(xp, xp.astype(field, C), half_u, half_v) * phase,
                                half_u, half_v)
                 for field in (w * nx * nx, w * nx * ny, w * ny * ny))


def _inverse_rule(xp, layer, lattice, half_u, half_v, C):
    """``Q = [[1/eps]]^-1`` of the true shapes, floored where the NVF solve floors its own."""
    background, shapes = layer.get('background', 1.0), layer['shapes']
    _, inverse = assemble_eps_coeffs(xp, list(shapes), background, half_u, half_v, lattice, C)
    inverse = conv_matrix_2d(xp, inverse, half_u, half_v)
    bound = inverse_rule_bound(xp, layer_epsilons(xp, [background, shapes[0]['material']], C))
    if bound is not None:
        inverse = _floor_singular_values(xp, inverse, 1.0 / (INVERSE_RULE_BOUND * bound))
    return xp.linalg.inv(inverse)


def _eps_at(xp, circle, layer, sample_x, sample_y, C, dev):
    """``eps`` at the samples, ``(n_y, n_x)``: the circle's inside, wall included, or the background.

    A sample is read against the nearest image of the circle, so it need not be in the cell.
    A circle a trace hides the fit of reads NaN where it reaches its own image, which is the
    refusal for a trace that cannot read one.
    """
    x = xp.real(_axis_samples(xp, sample_x, circle.side_x, C, dev)) - circle.cx
    y = xp.real(_axis_samples(xp, sample_y, circle.side_y, C, dev)) - circle.cy
    x = x - circle.side_x * xp.round(x / circle.side_x)
    y = y - circle.side_y * xp.round(y / circle.side_y)
    inside = x[None, :] ** 2 + y[:, None] ** 2 <= circle.radius ** 2
    background, material = layer_epsilons(
        xp, [layer.get('background', 1.0), layer['shapes'][0]['material']], C)
    eps = xp.where(inside, material, background)
    return xp.where(circle.reach > 0, eps, xp.full_like(eps, float('nan')))


def _round_shapes(xp, layer, lattice, real, dev):
    """``[(cx, cy, a, b)]`` of an ellipse or of several circles, and the cell's two sides.

    Centres from the cell's corner, ``a`` and ``b`` the semi-axes along x and y (one radius for
    a circle), as ``real`` scalars of ``xp`` on the derivative's path.
    """
    sides = tuple(as_array(xp, side, dtype=real, device=dev) for side in periods_of(lattice))
    rows = []
    for shape in layer['shapes']:
        kind, cx, cy, lx, ly, _ = shape_geometry(shape, lattice)
        cx, cy, lx, ly = (as_array(xp, value, dtype=real, device=dev) for value in (cx, cy, lx, ly))
        rows.append((cx, cy, 0.5 * lx, 0.5 * (lx if kind in _CIRCLES else ly)))
    return rows, sides


def _reach(xp, rows):
    """The window's reach for these shapes: their smallest inradius."""
    reach = None
    for _, _, a, b in rows:
        inradius = xp.minimum(a, b)
        reach = inradius if reach is None else xp.minimum(reach, inradius)
    return reach


def _clearance(xp, rows, sides):
    """What is left between the walls, positive where every shape fits: the smallest of each
    shape's gap to its own image and of each pair's gap, nearest images counted."""
    clear = None
    for i, (cx, cy, a, b) in enumerate(rows):
        gaps = [sides[0] - 2 * a, sides[1] - 2 * b] if len(rows) == 1 else [xp.minimum(sides[0], sides[1]) - 2 * a]
        for ox, oy, other, _ in rows[i + 1:]:
            dx, dy = cx - ox, cy - oy
            dx, dy = dx - sides[0] * xp.round(dx / sides[0]), dy - sides[1] * xp.round(dy / sides[1])
            gaps.append(_hypot_safe(xp, dx, dy) - (a + other))
        for gap in gaps:
            clear = gap if clear is None else xp.minimum(clear, gap)
    return clear


def _wall_distance(xp, rows, sides, x, y):
    """``(distance to the nearest wall, inside a shape)`` at points from the cell's corner.

    Each shape is read at its nearest image. An ellipse's distance is the exact one
    (:func:`~wavebit.rcwa._engine.parametric._ellipse_distance`).
    """
    distance, inside = None, None
    for cx, cy, a, b in rows:
        dx, dy = x - cx, y - cy
        dx, dy = dx - sides[0] * xp.round(dx / sides[0]), dy - sides[1] * xp.round(dy / sides[1])
        d = xp.abs(_ellipse_distance(xp, xp.abs(dx), xp.abs(dy), a, b))
        within = (dx / a) ** 2 + (dy / b) ** 2 <= 1.0
        distance = d if distance is None else xp.minimum(distance, d)
        inside = within if inside is None else xp.logical_or(inside, within)
    return distance, inside


def _shapes_projector(xp, layer, rows, sides, lattice, half_u, half_v, real, C, grid, windowed):
    """The convolution matrices of ``w n n^T`` for an ellipse or several circles: ``(xx, xy, yy)``.

    ``n`` is the solve's analytic normal field of the layer's shapes, sampled where they are, and
    ``w`` the window on the distance to the nearest wall, reaching :func:`_reach`.
    """
    nx, ny = normal_vector_field_analytical(xp, (grid, grid), list(layer['shapes']), real, lattice)
    if windowed:
        X, Y = sample_grid(xp, (grid, grid), lattice, real)
        w = _window(xp, _wall_distance(xp, rows, sides, X, Y)[0], _reach(xp, rows))
    else:
        w = xp.ones_like(nx)
    return tuple(conv_matrix_2d(xp, _sampled_coeffs(xp, xp.astype(field, C), half_u, half_v), half_u, half_v)
                 for field in (w * nx * nx, w * nx * ny, w * ny * ny))


def _shapes_eps_at(xp, rows, sides, layer, sample_x, sample_y, C, dev):
    """``eps`` at the samples for an ellipse or several circles, as :func:`_eps_at` gives a circle's.

    NaN everywhere where a trace hid shapes that touch each other or their own images.
    """
    x = xp.real(_axis_samples(xp, sample_x, sides[0], C, dev))
    y = xp.real(_axis_samples(xp, sample_y, sides[1], C, dev))
    X, Y = x[None, :] + 0 * y[:, None], y[:, None] + 0 * x[None, :]
    _, inside = _wall_distance(xp, rows, sides, X, Y)
    background, material = layer_epsilons(
        xp, [layer.get('background', 1.0), layer['shapes'][0]['material']], C)
    eps = xp.where(inside, material, background)
    return xp.where(_clearance(xp, rows, sides) > 0, eps, xp.full_like(eps, float('nan')))


def _applies(xp, layer, eps, C):
    """The samples that take the rule, ``(n_y, n_x)``, or ``None`` where all of them do.

    All of them where both materials can be read as ``Re eps > 0``, the case that needs no
    array. Otherwise the samples on the negative side where ``Re eps`` changes sign across the
    wall, and all of them where it does not; asked in ``xp``, so a trace answers it. A sample
    whose ``eps`` is NaN, the refusal under a trace, takes the rule and so keeps the NaN: left
    to the sum it would read as an answer.
    """
    values = [as_material(item).epsilon for item in (layer.get('background', 1.0), layer['shapes'][0]['material'])]
    host = [_host(value) for value in values]
    if all(value is not None and bool(np.all(np.real(value) > 0)) for value in host):
        return None
    background, material = (xp.real(value) < 0 for value in layer_epsilons(xp, values, C))
    negative = ~(xp.real(eps) >= 0)
    return xp.where(xp.logical_xor(background, material), negative, xp.ones_like(negative))


def _uniform(layer):
    """Is every shape its background, by value, with no derivative to tell them apart?"""
    values = [as_material(item).epsilon for item in [layer.get('background', 1.0)] + [
        shape['material'] for shape in layer['shapes']]]
    if any(carries_autodiff(value) for value in values):
        return False
    host = [_host(value) for value in values]
    return None not in host and all(bool(np.all(host[0] == value)) for value in host[1:])


def split_rows(xp, structure, lattice, kx, ky, real, C, sample_x, sample_y, held=None, windowed=True,
               grid=None):
    """What ``'splitw'`` (``windowed``) or ``'split'`` reads off each layer, in the public order of the layers.

    ``None`` where the layer has no wall, which keeps the Fourier sum bit for bit, and otherwise
    a :data:`~wavebit.rcwa._engine.lalanne.LayerRule` whose ``electric`` half is a
    :data:`~wavebit.rcwa._engine.fields.SplitHalf` and whose ``magnetic`` half is ``None``.
    ``held``, where given, is the layers to read (:func:`~wavebit.rcwa._engine.lalanne.
    layers_holding`). The operators do not depend on the samples; ``eps`` at them does.
    """
    half_u, half_v = (kx.shape[0] - 1) // 2, (ky.shape[0] - 1) // 2
    dev = _device_of(kx)
    # A side whose sign a trace hides cannot be refused by name: the operators are built for
    # positive sides, so a negative one answers NaN where the rule reads, as 'lalanne' does.
    positive = _positive_sides(xp, lattice, dev)
    rows = []
    for index, layer in enumerate(structure):
        if not layer.get('shapes') or (held is not None and index not in held) or _uniform(layer):
            rows.append(None)
            continue
        if _one_circle(layer):
            circle = _circle(xp, layer, lattice, real, dev)
            xx, xy, yy = _projector(xp, circle, lattice, half_u, half_v, real, C, grid or WINDOW_GRID, windowed)
            eps = _eps_at(xp, circle, layer, sample_x, sample_y, C, dev)
        else:
            shapes, sides = _round_shapes(xp, layer, lattice, real, dev)
            xx, xy, yy = _shapes_projector(xp, layer, shapes, sides, lattice, half_u, half_v, real, C,
                                           grid or SHAPES_GRID, windowed)
            eps = _shapes_eps_at(xp, shapes, sides, layer, sample_x, sample_y, C, dev)
        Q = _inverse_rule(xp, layer, lattice, half_u, half_v, C)
        normal = 0.5 * xp.concat([xp.concat([xx @ Q + Q @ xx, xy @ Q + Q @ xy], axis=1),
                                  xp.concat([xy @ Q + Q @ xy, yy @ Q + Q @ yy], axis=1)], axis=0)
        projector = xp.concat([xp.concat([xx, xy], axis=1), xp.concat([xy, yy], axis=1)], axis=0)
        tangential = xp.eye(projector.shape[0], dtype=C, device=_device_of(projector)) - projector
        eps = _masked(xp, eps, positive)
        rows.append(LayerRule(SplitHalf(normal, tangential, eps, _applies(xp, layer, eps, C)), None))
    return rows
