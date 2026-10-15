"""Explicit host validation for scalar shape geometry inside JAX transformations.

Only geometry is copied to the host. The callback returns a boolean, never a
replacement geometry or an optical answer. All numerical leaves retain their
original JAX graphs; invalid reuse is masked by the result validity contract.
"""
from dataclasses import fields

import numpy as np

from . import _adapt
from ._engine.topology import check_shape_record
from ..lattice import as_lattice


def shape_payload(problem):
    return (problem.stack.lattice if problem.stack.lattice is not None else problem.stack.period,
            tuple(tuple(tuple(getattr(r.geometry, f.name) for f in fields(r.geometry))
                        for r in layer.regions) for layer in problem.stack.layers))


def make_check(problem, records, factorization, projector):
    """Capture only immutable topology metadata; do not capture live material values."""
    layouts = tuple(tuple((type(r.geometry), tuple(f.name for f in fields(r.geometry)))
                          for r in layer.regions) for layer in problem.stack.layers)
    use_lattice = problem.stack.lattice is not None

    def check(cell, shapes):
        # This function receives concrete host arrays, including on every vmap
        # lane. No planner may interpret an unreadable tracer as permission.
        try:
            basis = as_lattice(lattice=cell) if use_lattice else as_lattice(period=cell)
            layers = []
            for layout, values in zip(layouts, shapes):
                recipes = []
                for (cls, names), coordinates in zip(layout, values):
                    shape = cls(**dict(zip(names, coordinates)))
                    recipes.append(_adapt.geometry_recipe(shape))
                layers.append(dict(shapes=recipes))
            # The engine uses one uniform zero-thickness layer for a bare interface.
            layers = layers or [dict(shapes=[])]
            for record in records:
                check_shape_record(layers, basis, record, factorization, projector)
        except (TypeError, ValueError, RuntimeError, NotImplementedError):
            return np.asarray(False)
        return np.asarray(True)

    return check


def traced_check(problem, check):
    import jax
    import jax.numpy as jnp

    payload = shape_payload(problem)
    # The validity predicate is discrete. Detaching its inputs does not detach
    # the independently executed solver or its tangent.
    values = jax.tree.map(lambda v: jax.lax.stop_gradient(jnp.asarray(v)), payload)

    def finite_check(cell, shapes):
        if not all(np.isfinite(v).all() for v in jax.tree.leaves((cell, shapes))):
            return np.asarray(False)
        return check(cell, shapes)

    return jax.pure_callback(finite_check, jax.ShapeDtypeStruct((), jnp.bool_),
                             *values, vmap_method='sequential')
