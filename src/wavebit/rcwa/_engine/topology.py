"""Concrete scalar-shape replay validation, shared by eager and callback execution."""


def check_shape_record(layers, lattice, geometry, factorization, projector):
    from . import boundary, parametric, rectilinear
    from .geometry import StaleGeometryError

    if factorization == 'li':
        rectilinear.validate(layers, lattice)
        plans = geometry.rectilinear_plans
        if plans is None or len(plans) != len(layers):
            raise StaleGeometryError('Cartesian Li layer plans changed; freeze_geometry() again')
        for layer, plan in zip(layers, plans):
            rectilinear.check_plan(layer.get('shapes', []), lattice, plan)
        return
    plans, gammas = geometry.boundary_plans, geometry.gamma_plans
    for records in (plans, gammas if factorization == 'nvf' else None):
        if records is not None and len(records) != len(layers):
            raise StaleGeometryError('shape layer count changed; freeze_geometry() again')
    for index, layer in enumerate(layers):
        parametric.check_layer_spec(layer, index)
        parametric.check_layer_shapes(layer, lattice, index)
        plan = None if plans is None else plans[index]
        if plans is not None:
            if plan is None:
                parametric.check_no_plan_still_fits(layer, lattice, index)
            else:
                boundary.check_plan_still_fits(layer.get('shapes', []), lattice, plan)
        if factorization == 'nvf' and gammas is not None and (plan is None or projector == 'circular'):
            parametric.check_gamma_plan_still_fits(
                layer, lattice, gammas[index], index, dyadic=projector == 'circular')
