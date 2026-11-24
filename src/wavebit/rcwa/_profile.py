"""Slicing of layers whose cross-section changes with height (:class:`~.problem.ProfiledLayer`).

A profiled layer is solved as a stack of ordinary layers, each holding the cross-section at its
midpoint height. Nothing below this module knows about profiles: every slice is a
:class:`~.problem.Layer` of :class:`~.problem.Rectangle` (a :class:`~.problem.Trapezoid` line) or
:class:`~.problem.Polygon` (a :class:`~.problem.Loft` or :class:`~.problem.Profile`) regions, so
the slices take the closed-form and painting routes as written, with their factorizations,
backends and derivatives. There is no raster form: a raster quantizes each width to pixels.

The slice boundaries are fractions of the layer's thickness, planned on concrete values when the
problem is prepared and fixed afterwards. A run then moves only the thickness and the widths, so
the derivative with respect to a sidewall angle, a rounding radius or the height is the
derivative of a fixed-topology staircase, smooth in every parameter.

The midpoint rule's staircase error falls as ``N**-2`` for a smooth profile; the error is
estimated by Richardson's deferred approach to the limit (L. F. Richardson, Phil. Trans. R. Soc.
A 210, 307 (1911)) from solves at ``N`` and ``N/2`` slices.
"""
import dataclasses
import math
import warnings
from types import MappingProxyType

import numpy as np

from .._common.backend import SlicingWarning, _device_of, array_namespace, as_array, in_namespace
from ..errors import IncompatibleProblemError, UnsupportedFeatureError, ValidationError, WavebitError
from . import _adapt
from .numerics import Slicing
from .problem import Graded, Layer, Loft, Polygon, Problem, PROFILE_SHAPES, ProfiledLayer, Rectangle, Region, Trapezoid

#: Samples along the height used to measure a sidewall's arc length when planning the slices.
_PLAN_SAMPLES = 2048

_HALF_PI = 1.5707963267948966


def holds_profile(problem):
    return isinstance(problem, Problem) and any(isinstance(layer, ProfiledLayer)
                                                for layer in problem.stack.layers)


def _foreign(value):
    """A JAX or PyTorch array (a tracer included): the values that pick the namespace."""
    return (hasattr(value, 'shape') and hasattr(value, 'dtype')
            and not isinstance(value, (np.ndarray, np.generic)))


def _differentiated(value):
    """Whether a leaf carries a derivative: a JAX tracer, or a tensor that requires grad."""
    return _adapt.abstract(value) or bool(getattr(value, 'requires_grad', False))


class _Math:
    """Scalar arithmetic on Python floats, or on the namespace of the arrays among ``values``.

    Python floats stay Python floats, so a vertical wall's zero offset is an exact ``0.0`` and a
    vertical line is the same Rectangle, to the bit, as one written by hand.
    """

    def __init__(self, *values):
        refs = [v for v in _adapt.leaves(values) if _foreign(v)]
        self.ref = refs[0] if refs else None
        self.xp = array_namespace(self.ref) if refs else None
        if self.xp is not None:
            dtype = self.ref.dtype
            kind = getattr(dtype, 'is_complex', None)
            complex_ = kind if kind is not None else np.dtype(str(dtype)).kind == 'c'
            self.dtype = None if complex_ or 'int' in str(dtype) else dtype

    def c(self, value):
        if self.xp is None or in_namespace(self.xp, value):
            return value
        return as_array(self.xp, value, dtype=self.dtype, device=_device_of(self.ref))

    def tan(self, v):
        return math.tan(v) if self.xp is None else self.xp.tan(self.c(v))

    def sin(self, v):
        return math.sin(v) if self.xp is None else self.xp.sin(self.c(v))

    def where(self, condition, a, b):
        if self.xp is None:
            return a if condition else b
        return self.xp.where(self.c(condition), self.c(a), self.c(b))

    def arc(self, zone, radius, offset):
        """``sqrt(radius**2 - offset**2)`` inside ``zone``; a safe 1 outside it, for the derivative."""
        squared = radius*radius - offset*offset
        if self.xp is None:
            return math.sqrt(max(squared, 0.)) if zone else 1.
        safe = self.xp.where(self.c(zone), self.c(squared), self.c(1.))
        return self.xp.sqrt(self.xp.maximum(safe, self.c(0.)))


def _walls(g):
    return g.swa if isinstance(g.swa, tuple) else (g.swa, g.swa)


def _straight(m, g, h, z):
    """Width and centre of the straight-walled trapezoid at height ``z``, before any rounding."""
    left, right = _walls(g)
    tl, tr = m.tan(_HALF_PI - left), m.tan(_HALF_PI - right)
    z0 = {'bottom': 0., 'middle': h/2, 'top': h}[g.cd_at]
    dz = z - z0
    return g.cd - dz*(tl + tr), g.center + dz*(tl - tr)/2


def _corner(m, g, h, z, side):
    """``(zone, x)`` of one wall's rounded parts at height ``z``: the rounding's x where it applies.

    The arc of radius ``r`` filling a corner of interior angle ``theta`` is tangent to both faces
    at ``r / tan(theta/2)`` from the corner. A top corner's interior angle is ``pi - swa`` and a
    footing's exterior angle is the same, so both tangent lengths are ``r * tan(swa/2)``.
    """
    left, right = _walls(g)
    swa = right if side > 0 else left
    rt, rf = g.top_rounding, g.footing
    half = m.tan(swa/2)
    lift = m.sin(swa)
    w_top, c_top = _straight(m, g, h, h)
    w_bottom, c_bottom = _straight(m, g, h, 0.)
    edge_top, edge_bottom = c_top + side*w_top/2, c_bottom + side*w_bottom/2
    reach_top, reach_foot = rt*half, rf*half
    in_top = z > h - reach_top*lift
    in_foot = z < reach_foot*lift
    x_top = edge_top - side*reach_top + side*m.arc(in_top, rt, z - (h - rt))
    x_foot = edge_bottom + side*reach_foot - side*m.arc(in_foot, rf, z - rf)
    return in_top, x_top, in_foot, x_foot


def trapezoid_section(m, g, h, z):
    """Width and centre of a :class:`Trapezoid` line at height ``z`` above the layer's bottom face."""
    width, centre = _straight(m, g, h, z)
    if m.xp is None and not g.top_rounding and not g.footing:
        return width, centre
    edges = []
    rounded = False
    for side in (-1, 1):
        in_top, x_top, in_foot, x_foot = _corner(m, g, h, z, side)
        straight = centre + side*width/2
        edges.append(m.where(in_top, x_top, m.where(in_foot, x_foot, straight)))
        rounded = (in_top or in_foot or rounded) if m.xp is None else (
            m.xp.logical_or(m.xp.logical_or(m.c(in_top), m.c(in_foot)), m.c(rounded)))
    left, right = edges
    return m.where(rounded, right - left, width), m.where(rounded, (right + left)/2, centre)


def _lerp(m, bottom, top, fraction):
    if isinstance(bottom, tuple) and isinstance(top, tuple):
        return tuple(_lerp(m, b, t, fraction) for b, t in zip(bottom, top))
    if m.xp is None:
        if isinstance(bottom, tuple) or isinstance(top, tuple):
            bottom, top = np.asarray(bottom, dtype=float), np.asarray(top, dtype=float)
        return bottom + (top - bottom)*fraction
    b, t = m.c(bottom), m.c(top)
    return b + (t - b)*fraction


def _period_y(problem, path):
    stack = problem.stack
    if stack.period is None:
        raise ValidationError('A Trapezoid line spans the cell along y; give the cell as '
                              'Stack(period=(Lx, Ly)).', code='shape.profile_cell', path=path)
    return stack.period[1]


def material_at(material, z, u):
    """A :class:`Graded` material at height ``z`` (fraction ``u`` of the thickness); any other as it is."""
    if isinstance(material, Graded):
        return material.material(u if material.relative else z)
    return material


def slice_regions(problem, layer, m, z, path, u=None):
    """The regions of ``layer`` at height ``z``: each profile shape as its cross-section.

    ``u`` is that height as a fraction of the thickness, which a relative :class:`Graded`
    material reads; given as a number, it stays one where ``z`` is traced through the thickness.
    """
    if u is None:
        u = z/layer.thickness
    regions = []
    for j, region in enumerate(layer.regions):
        g = region.geometry
        if isinstance(g, Trapezoid):
            width, centre = trapezoid_section(m, g, layer.thickness, z)
            size_y = _period_y(problem, f'{path}.regions[{j}].geometry')
            geometry = Rectangle(size=(width, size_y), center=(centre, 0.))
        elif isinstance(g, Loft):
            fraction = z/layer.thickness
            geometry = Polygon(vertices=_lerp(m, g.bottom, g.top, fraction))
        elif isinstance(g, PROFILE_SHAPES):
            geometry = Polygon(vertices=g.section(z))
        else:
            geometry = g
        regions.append(Region(geometry=geometry, material=material_at(region.material, z, u)))
    return tuple(regions)


def slice_layer(problem, layer, m, z, u, thickness, path):
    """One slice of ``layer``: an ordinary layer of its cross-section and materials at ``z``."""
    return Layer(thickness=thickness, background=material_at(layer.background, z, u),
                 regions=slice_regions(problem, layer, m, z, path, u))


def _layer_math(problem, layer):
    geometry = tuple(r.geometry for r in layer.regions if not callable(getattr(r.geometry, 'section', None)))
    return _Math(layer.thickness, geometry, problem.stack.period)


def expand(problem, fractions, merged):
    """``problem`` with each profiled layer replaced by its slices, and each slice's public layer.

    ``fractions[i]`` is the slice boundaries of layer ``i`` as fractions of its thickness, and
    ``merged[i]`` says to solve it as one layer of its (height-independent) cross-section.
    """
    layers, owner = [], []
    for i, layer in enumerate(problem.stack.layers):
        if not isinstance(layer, ProfiledLayer):
            layers.append(layer)
            owner.append(i)
            continue
        h = layer.thickness
        m = _layer_math(problem, layer)
        path = f'stack.layers[{i}]'
        if merged[i]:
            layers.append(slice_layer(problem, layer, m, h/2, .5, h, path))
            owner.append(i)
            continue
        t = fractions[i]
        for t0, t1 in zip(t[:-1], t[1:]):
            layers.append(slice_layer(problem, layer, m, h*((t0 + t1)/2), (t0 + t1)/2, h*(t1 - t0), path))
            owner.append(i)
    stack = dataclasses.replace(problem.stack, layers=tuple(layers))
    return dataclasses.replace(problem, stack=stack), tuple(owner)


# ----------------------------------------------------------------------------------------------
# Planning on concrete values


def _host(value, path):
    return np.asarray(_adapt.host(value, path), dtype=float)


def _concrete(problem):
    """``problem`` with the period and each profiled layer's thickness and geometry read as host values.

    Planning reads numbers only; a JAX primal or a detached tensor value is the design being
    prepared, and nothing planned on it reaches the run's derivative.
    """
    def read(value, path):
        out = _host(value, path)
        return float(out) if out.shape == () else out

    def geometry(g, path):
        if isinstance(g, Trapezoid):
            return dataclasses.replace(g, **{f: (tuple(read(v, path) for v in getattr(g, f))
                                                 if isinstance(getattr(g, f), tuple) else read(getattr(g, f), path))
                                             for f in ('cd', 'swa', 'top_rounding', 'footing', 'center')})
        if isinstance(g, Loft):
            return dataclasses.replace(g, bottom=read(g.bottom, path), top=read(g.top, path))
        return g

    layers = []
    for i, layer in enumerate(problem.stack.layers):
        if isinstance(layer, ProfiledLayer):
            path = f'stack.layers[{i}]'
            layer = dataclasses.replace(layer, thickness=read(layer.thickness, path), regions=tuple(
                dataclasses.replace(r, geometry=geometry(r.geometry, path)) for r in layer.regions))
        layers.append(layer)
    stack = problem.stack
    cell = ({'period': tuple(read(v, 'stack.period') for v in stack.period)} if stack.period is not None else {})
    return dataclasses.replace(problem, stack=dataclasses.replace(stack, layers=tuple(layers), **cell))


def _edges(problem, layer, z, path):
    """Concrete x of every moving edge or vertex of ``layer`` at height ``z``, flattened."""
    m = _Math()
    out = []
    for j, region in enumerate(layer.regions):
        g = region.geometry
        if isinstance(g, Trapezoid):
            width, centre = trapezoid_section(m, g, float(layer.thickness), z)
            out += [centre - width/2, centre + width/2]
        elif isinstance(g, Loft):
            out += list(np.ravel(_lerp(m, g.bottom, g.top, z/float(layer.thickness))))
        elif isinstance(g, PROFILE_SHAPES):
            out += list(np.ravel(_host(g.section(z), f'{path}.regions[{j}].geometry.section')))
    return np.asarray(out, dtype=float)


def plan_fractions(problem, i, count, rule):
    """Slice boundaries of profiled layer ``i`` as ``count + 1`` fractions of its thickness."""
    layer = problem.stack.layers[i]
    uniform = tuple(k/count for k in range(count)) + (1.,)
    # A layer whose outline does not move -- a graded material alone -- is cut uniformly: its
    # midpoint staircase converges at second order in the smooth material profile as it is.
    if rule == 'uniform' or not any(isinstance(r.geometry, PROFILE_SHAPES) for r in layer.regions):
        return uniform
    h = float(layer.thickness)
    z = np.linspace(0., h, _PLAN_SAMPLES + 1)
    x = np.stack([_edges(problem, layer, float(v), f'stack.layers[{i}]') for v in z])
    dz = h/_PLAN_SAMPLES
    slope = np.max(np.abs(np.diff(x, axis=0)), axis=1)/dz if x.shape[1] else np.zeros(_PLAN_SAMPLES)
    # Arc length of the fastest-moving edge in the x-z plane: a straight taper is cut uniformly,
    # and a rounded corner, where the edge turns horizontal, gets the slices its length needs.
    arc = np.concatenate([[0.], np.cumsum(np.hypot(dz, slope*dz))])
    targets = arc[-1]*np.arange(count + 1)/count
    t = np.interp(targets, arc, z)/h
    t[0], t[-1] = 0., 1.
    if np.any(np.diff(t) <= 0):
        return uniform
    return tuple(float(v) for v in t)


def _check_trapezoid(problem, layer, g, path):
    """Refuse a concrete Trapezoid whose rounding does not fit, or that reaches its own image."""
    def value(v):
        return float(_host(v, path))
    g = dataclasses.replace(g, cd=value(g.cd), center=value(g.center), top_rounding=value(g.top_rounding),
                            footing=value(g.footing), swa=tuple(value(v) for v in _walls(g)))
    h = value(layer.thickness)
    m = _Math()
    left, right = _walls(g)
    rt, rf = g.top_rounding, g.footing
    w_bottom, _ = _straight(m, g, h, 0.)
    w_top, _ = _straight(m, g, h, h)
    if not (w_bottom > 0 and w_top > 0):
        raise ValidationError(f'This Trapezoid closes inside the layer: its straight walls give width '
                              f'{w_bottom:.6g} at the bottom and {w_top:.6g} at the top. Reduce the '
                              f'taper or the thickness.', code='shape.profile', path=path)
    top_reach = rt*(math.tan(left/2) + math.tan(right/2))
    if top_reach > w_top*(1 + 1e-12):
        raise ValidationError(f'top_rounding={rt:.6g} needs {top_reach:.6g} of top width, and the top is '
                              f'{w_top:.6g} wide.', code='shape.profile', path=f'{path}.top_rounding')
    for swa in (left, right):
        used = (rt + rf)*math.tan(swa/2)*math.sin(swa)
        if used > h*(1 + 1e-12):
            raise ValidationError(f'top_rounding and footing together reach {used:.6g} up a sidewall '
                                  f'of height {h:.6g}.', code='shape.profile', path=path)
    widest = max(w_top, w_bottom + rf*(math.tan(left/2) + math.tan(right/2)))
    period = float(_host(problem.stack.period[0], path))
    if widest >= period:
        raise ValidationError(f'This Trapezoid is {widest:.6g} wide at its widest, which reaches its periodic '
                              f'image in a period of {period:.6g}.', code='shape.profile', path=path)


def refuse(problem):
    """Concrete refusals of a problem holding profiled layers."""
    layers = problem.stack.layers
    for i, layer in enumerate(layers):
        if isinstance(layer, Layer) and layer.epsilon is not None:
            raise ValidationError(f'stack.layers[{i}] is a sampled (raster) layer, and a stack with a '
                                  f'ProfiledLayer is built on the region path only: a raster '
                                  f'quantizes each slice width to pixels, so the width would move in '
                                  f'steps as a sidewall angle changes and its derivative would be zero '
                                  f'between them. Describe the layer with regions.',
                                  code='layer.profile_raster', path=f'stack.layers[{i}]')
    for i, layer in enumerate(layers):
        if not isinstance(layer, ProfiledLayer):
            continue
        for j, region in enumerate(layer.regions):
            if isinstance(region.geometry, Trapezoid):
                path = f'stack.layers[{i}].regions[{j}].geometry'
                _period_y(problem, path)
                if not _adapt.opaque((region.geometry, layer.thickness, problem.stack.period)):
                    _check_trapezoid(problem, layer, region.geometry, path)


def invariant(problem, i, fractions):
    """Whether profiled layer ``i`` is the same at every slice, on values carrying no derivative.

    Read on concrete values: a geometry value that carries a derivative keeps the layer sliced,
    so the derivative sees the profile, and a value with no number (under ``jax.jit``) cannot be
    compared at all.
    """
    layer = problem.stack.layers[i]
    if _adapt.opaque(problem.stack.layers[i]):
        return False
    if any(_differentiated(v) for r in layer.regions for v in _adapt.leaves(r.geometry)):
        return False
    m = _layer_math(problem, layer)
    h = layer.thickness
    path = f'stack.layers[{i}]'
    first = None
    for t0, t1 in zip(fractions[:-1], fractions[1:]):
        sliced = slice_layer(problem, layer, m, h*((t0 + t1)/2), (t0 + t1)/2, h, path)
        parts = [sliced.background] + [r.material for r in sliced.regions] + [r.geometry for r in sliced.regions]
        leaves = [v for part in parts for v in _adapt.leaves(part)]
        if any(_differentiated(v) for v in leaves):
            return False
        try:
            values = [(type(part).__name__,) for part in parts] + [np.asarray(_adapt.host(v, path)) for v in leaves]
        except (TypeError, ValueError):
            # A material model with no array leaves to read (a tabulated table): compare it by identity.
            values = [id(part) for part in parts]
        if first is None:
            first = values
        elif len(values) != len(first) or not all(np.array_equal(a, b) for a, b in zip(values, first)):
            return False
    return True


# ----------------------------------------------------------------------------------------------
# Public paths and diagnostics


def _public_path(path, owner):
    """Rewrite ``stack.layers[k]`` of the sliced problem as the public layer and its slice."""
    import re
    match = re.match(r'stack\.layers\[(\d+)\](.*)', path or '')
    if not match:
        return path
    k = int(match.group(1))
    if k >= len(owner):
        return path
    i = owner[k]
    first = owner.index(i)
    count = owner.count(i)
    where = f'stack.layers[{i}]'
    if count > 1:
        where += f' (slice {k - first} of {count})'
    return where + match.group(2)


def _publish(error, owner):
    if isinstance(error, WavebitError) and error.path:
        public = _public_path(error.path, owner)
        if public != error.path:
            text = str(error)
            if text.startswith(error.path + ': '):
                text = public + ': ' + text[len(error.path) + 2:]
            error.path = public
            error.issue = dataclasses.replace(error.issue, path=public)
            error.args = (text,) + error.args[1:]
    return error


def publish_report(report, owner):
    issues = tuple(dataclasses.replace(issue, path=_public_path(issue.path, owner))
                   for issue in report.issues)
    return dataclasses.replace(report, issues=issues)


def _covered(owner, count, from_top):
    """How many whole public layers the first ``count`` slices from one side make up."""
    sequence = owner[::-1] if from_top else owner
    covered = consumed = 0
    for public in dict.fromkeys(sequence):
        consumed += sequence.count(public)
        if consumed > count:
            break
        covered += 1
    return covered


def _public_merges(merges, owner):
    """The half-space merge record counted in public layers: a layer counts when all its slices do."""
    if merges is None:
        return None
    out = []
    for plan in merges:
        if plan is None:
            out.append(None)
            continue
        snapped = plan['snapped']
        if all(isinstance(k, int) for k in snapped):
            snapped = tuple(sorted({owner[k] for k in snapped}))
        out.append({'bottom': _covered(owner, plan['bottom'], False),
                    'top': _covered(owner, plan['top'], True), 'snapped': snapped})
    return tuple(out)


def _efficiencies(result, xp=None):
    """Every reflected and transmitted efficiency of a result, flattened."""
    if xp is None:
        return np.concatenate([np.ravel(np.asarray(_adapt.host(result.R, 'result'), dtype=float)),
                               np.ravel(np.asarray(_adapt.host(result.T, 'result'), dtype=float))])
    return xp.concat([xp.reshape(result.R, (-1,)), xp.reshape(result.T, (-1,))])


# ----------------------------------------------------------------------------------------------
# The preparation


class Profiled:
    """The slicing plan of a prepared problem, and the preparations of its sliced layouts."""

    def __init__(self, problem, numerics, prepare):
        self.numerics = numerics
        self.slicing = numerics.slicing or Slicing()
        self._prepare = prepare
        self.profiled = tuple(i for i, layer in enumerate(problem.stack.layers)
                              if isinstance(layer, ProfiledLayer))
        self.layer_count = len(problem.stack.layers)
        self._fractions = {}
        self._cache = {}
        self._plan_problem = _concrete(problem)
        self.levels, self.error, self.observed_order, self.order_used = (), None, None, 2.
        self.converged = None
        refuse(problem)
        slicing = self.slicing
        start = slicing.count if slicing.count != 'auto' else slicing.min_count
        merged = self.merge_flags(problem, start)
        if slicing.count == 'auto':
            if not problem.outputs.diffraction:
                raise UnsupportedFeatureError(
                    "Slicing(count='auto') estimates the staircase error from the diffraction "
                    'efficiencies, and this problem asks for none. Give Slicing(count=N).',
                    code='numerics.slicing', path='numerics.slicing.count')
            self.count = self._auto(problem, merged)
        else:
            self.count = slicing.count
            if slicing.estimate is not None and problem.outputs.diffraction:
                self._estimate_at_prepare(problem, merged)
        unmerged = tuple(False for _ in problem.stack.layers)
        self.inner(problem, self.count, unmerged)
        if any(merged):
            self.inner(problem, self.count, merged)
        if slicing.estimate == 'run' and self.count >= 2:
            self.inner(problem, self.count//2, unmerged)
        self.prepared_merged = merged

    # The plan --------------------------------------------------------------------------------

    def fractions(self, count):
        if count not in self._fractions:
            problem = self._plan_problem
            self._fractions[count] = tuple(
                plan_fractions(problem, i, count, self.slicing.rule) if i in self.profiled else None
                for i in range(self.layer_count))
        return self._fractions[count]

    def merge_flags(self, problem, count):
        fractions = self.fractions(count)
        return tuple(i in self.profiled and invariant(problem, i, fractions[i])
                     for i in range(self.layer_count))

    def expand(self, problem, count, merged):
        return expand(problem, self.fractions(count), merged)

    def inner(self, problem, count, merged):
        """The preparation of one sliced layout, prepared on concrete values when first needed."""
        key = (count, merged)
        if key not in self._cache:
            if _adapt.opaque(problem):
                raise IncompatibleProblemError(
                    'This sliced layout was not prepared and cannot be prepared on traced values. '
                    'Prepare outside the transform.', code='prepared.structure', path='stack')
            expanded, owner = self.expand(_adapt.primal_problem(problem), count, merged)
            try:
                self._cache[key] = (self._prepare(expanded), owner)
            except WavebitError as error:
                raise _publish(error, owner)
        return self._cache[key]

    def _solve(self, problem, count, merged):
        prepared, owner = self.inner(problem, count, merged)
        expanded, _ = self.expand(problem, count, merged)
        try:
            return prepared.run(expanded)
        except WavebitError as error:
            raise _publish(error, owner)

    def _order(self, differences):
        """Observed order from the last three levels, and the order the estimate uses."""
        if len(differences) < 2 or differences[-1] <= 0 or differences[-2] <= 0:
            return None, 2.
        observed = math.log2(differences[-2]/differences[-1])
        # Never assume faster convergence than the midpoint rule's second order; a slower one
        # observed outside the asymptotic range makes the estimate more cautious, not less.
        return observed, min(2., max(.5, observed))

    def _auto(self, problem, merged):
        slicing = self.slicing
        if all(merged[i] for i in self.profiled):
            # Every profiled layer is the same at every height: any count solves the same physics.
            self.error, self.converged, self.levels = 0., True, ()
            return slicing.min_count
        unmerged = tuple(False for _ in problem.stack.layers)
        count, previous, differences, levels = slicing.min_count, None, [], []
        while True:
            values = _efficiencies(self._solve(problem, count, unmerged))
            levels.append(count)
            if previous is not None:
                differences.append(float(np.max(np.abs(values - previous))))
                self.observed_order, self.order_used = self._order(differences)
                self.error = differences[-1]/(2**self.order_used - 1)
                if self.error <= slicing.tolerance:
                    self.converged = True
                    break
            if 2*count > slicing.max_count:
                self.converged = False
                break
            previous, count = values, 2*count
        self.levels = tuple(levels)
        if not self.converged:
            warnings.warn(f'Slicing did not reach tolerance={slicing.tolerance:g}: with {count} slices '
                          f'per profiled layer the estimated staircase error is {self.error:.3g}. Raise '
                          f'Slicing(max_count=...) or loosen the tolerance.', SlicingWarning, stacklevel=4)
        return count

    def _estimate_at_prepare(self, problem, merged):
        if self.count < 2 or all(merged[i] for i in self.profiled):
            self.error = 0. if all(merged[i] for i in self.profiled) else None
            return
        unmerged = tuple(False for _ in problem.stack.layers)
        half = _efficiencies(self._solve(problem, self.count//2, unmerged))
        full = _efficiencies(self._solve(problem, self.count, unmerged))
        self.levels = (self.count//2, self.count)
        self.error = float(np.max(np.abs(full - half)))/(2**self.order_used - 1)

    # Runs ------------------------------------------------------------------------------------

    def run_flags(self, problem):
        if _adapt.opaque(problem):
            return tuple(False for _ in range(self.layer_count))
        return self.merge_flags(problem, self.count)

    def run(self, problem):
        if not isinstance(problem, Problem):
            raise ValidationError('Prepared.run/validate require rcwa.Problem.', code='problem.type', path='problem')
        if len(problem.stack.layers) != self.layer_count or tuple(
                i for i, layer in enumerate(problem.stack.layers) if isinstance(layer, ProfiledLayer)) != self.profiled:
            raise IncompatibleProblemError('The problem declaration layout changed. Prepare it again.',
                                           code='prepared.structure', path='stack.layers')
        refuse(problem)
        merged = self.run_flags(problem)
        result = self._solve(problem, self.count, merged)
        error, source = self.error, 'prepare' if self.error is not None else None
        if self.slicing.estimate == 'run' and self.count >= 2 and problem.outputs.diffraction:
            half = self._solve(problem, self.count//2, tuple(False for _ in range(self.layer_count)))
            xp = array_namespace(result.R)
            difference = xp.max(xp.abs(_efficiencies(result, xp) - _efficiencies(half, xp)))
            error, source = difference/(2**self.order_used - 1), 'run'
        return self._attach(result, merged, error, source)

    def validate(self, problem):
        """Whether a concrete problem can reuse the sliced layout its values select."""
        from .._common.execution import CheckReport
        from ._lifecycle import _issue
        try:
            if _adapt.opaque(problem):
                raise UnsupportedFeatureError('validate requires concrete inputs; call it outside jax.jit or jax.vmap.',
                                              code='validate.concrete', path='problem')
            problem = _adapt.primal_problem(problem)
            refuse(problem)
            merged = self.run_flags(problem)
            prepared, owner = self.inner(problem, self.count, merged)
            expanded, _ = self.expand(problem, self.count, merged)
            return publish_report(prepared.validate(expanded), owner)
        except (ValidationError, TypeError, ValueError, NotImplementedError, RuntimeError) as error:
            return CheckReport(issues=(_issue(error),))

    def _attach(self, result, merged, error, source):
        _, owner = self._cache[(self.count, merged)]
        if all(merged[i] for i in self.profiled):
            # Each profiled layer is solved as itself, one layer: there is no staircase.
            error, source = 0., 'merged'
        per_layer = []
        for i in range(self.layer_count):
            if i not in self.profiled:
                per_layer.append(None)
                continue
            per_layer.append(MappingProxyType({
                'count': 1 if merged[i] else self.count,
                'planned_count': self.count,
                'rule': self.slicing.rule,
                'merged': bool(merged[i]),
                'fractions': (0., 1.) if merged[i] else self.fractions(self.count)[i],
                'staircase_error': error,
                'observed_order': self.observed_order,
                'order_used': self.order_used,
                'levels': self.levels,
                'converged': self.converged,
                'tolerance': self.slicing.tolerance,
                'source': source}))
        diagnostics = dict(result.diagnostics)
        diagnostics['z_slicing'] = tuple(per_layer)
        if diagnostics.get('half_space_merge') is not None:
            diagnostics['half_space_merge'] = _public_merges(diagnostics['half_space_merge'], owner)
        validity = result.validity
        if self.converged is not None:
            checks = dict(validity.checks)
            checks['z_slicing_converged'] = np.bool_(self.converged)
            validity = dataclasses.replace(validity, checks=checks)
        return dataclasses.replace(result, diagnostics=diagnostics, validity=validity)
