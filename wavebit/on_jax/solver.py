from .emsolver.rcwa import RCWAJax
from .optimizer.optimizer import OptimizerJax


class SolverJax(RCWAJax, OptimizerJax):

    def __init__(self, *args, **kwargs):
        RCWAJax.__init__(self, *args, **kwargs)
        OptimizerJax.__init__(self, *args, **kwargs)

    def _tree_flatten(self):
        # `polarization` is a *value* of the incidence, so it is a child rather than aux data.
        # It used to ride in aux, which made it a compile-time constant: differentiating a
        # solver whose incidence was a pair returned exactly `0j` for it -- not an error, a
        # zero, and indistinguishable from the true zeros the same slot produces (a pure-s
        # incidence on a mirror-symmetric structure gives one). It rides last so the leading
        # tuple still lines up with `__init__`'s positional parameters; `None` is an empty
        # subtree, so a solver without a pair flattens exactly as it did before.
        #
        # `top` and `bot` are the two Materials, not the derived indices. They ride as
        # *children* because a Material is a registered pytree whose children are
        # `(epsilon, mu)`, so a swept or differentiated half-space permittivity is a traced
        # leaf here rather than a compile-time constant hashed into the cache key -- the same
        # argument `polarization` makes just below. Handing back the derived `n_top` instead
        # would also be lossy: `sqrt(eps*mu)` cannot be taken apart again, so a magnetic
        # half-space would come back through `_tree_unflatten` as a non-magnetic one of a
        # different permittivity, silently.
        #
        # `mu` is the raster permeability field, a child for the same reason `structure` is:
        # it is material values, and material values are what gradients are for. It rides
        # last, after `polarization`, so the leading tuple still lines up with `__init__`'s
        # positional parameters; `None` is an empty subtree, so a non-magnetic solver
        # flattens exactly as it did before.
        children = (self.top, self.bot, self.theta, self.phi, self.psi,
                    self.period, self.wavelength, self.structure, self.thickness,
                    self._chi, self._polarization, self._mu)
        aux_data = {
            'backend': self.backend,
            'tetm': self.tetm,
            'truncation_order': self.truncation_order,
            'layer_algorithm': self.layer_algorithm,
            'device': self.device,
            'complex_dtype': self.complex_dtype,
            # The *requested* values, not the resolved ones the properties report. Both axes
            # resolve against the modeling path, and a pytree round-trip has to reconstruct the
            # solver it flattened: writing the resolved value here would hand the rebuilt solver
            # an explicit request nobody made, so a structure that had merely accepted its
            # path's only option would come back out asserting it -- and, on a solver that did
            # name an unavailable one, the refusal would fire here, inside JAX's tree
            # machinery, rather than at the solve that is actually being refused.
            'fourier_coefficients': self._fourier_coefficients,
            'factorization': self._factorization,
            'normal_method': self._normal_method,
            # A dispatch key like `pol`: the solves branch on it (which cascade blocks are
            # read, which way the stack is walked), so it must come back static, never traced.
            'incidence_side': self.incidence_side,
            # Aux data, and it has to be. A plain instance attribute survives __init__ but
            # nothing puts it here, so `_tree_unflatten` never passes it to the constructor
            # and the rebuilt solver takes the default: measured on this pair, a solver that
            # declared 'field' comes back reporting 'amplitude' and retaining 0.000 MB where
            # it should hold 1.253. And with the key absent the two declarations produce
            # *identical* aux_data, so their treedefs coincide and one jit compilation serves
            # both -- the loss cannot even be recovered by recompiling. Static here, so the
            # core's `covers` branches are read at trace time, the only time they can be.
            # See docs/what-a-solve-answers.md section 6.
            'answers': self.answers,
            # Not the pair itself (that is a child now) but the fact that `psi` was given
            # explicitly, which the always-present psi child would otherwise overwrite.
            'psi_given': self._psi_given,
            # Both of these were missing, and both are silent when they are. `solve_field` is
            # `jax.jit`-decorated at class level, so `self` is flattened on every call -- which
            # means the solve ran on whatever the *rebuilt* solver carries, not on what the
            # caller configured. Measured on a 60-degree cell: the basis came back `None`,
            # `lattice_is_axis_aligned` flipped False -> True (taking
            # `_reject_unsupported_lattice`'s guard with it), and the jitted call answered the
            # rectangular-cell problem -- T = 0.6565 against a correct 0.5460, energy conserving
            # throughout. `normal_grid` fell 4096 -> 1024, which moves a shape gradient's
            # sawtooth floor by an order of magnitude.
            #
            # The basis rides as a nested tuple because aux_data has to be hashable and this is
            # a `(2, 2)` ndarray; `lattice.as_lattice` takes it back through `np.asarray`, so
            # the round trip is exact. It belongs in aux rather than in children for the reason
            # its own docstring gives -- a basis decides control flow and reaches SciPy, so it
            # is concrete by construction and can never be a tracer.
            #
            # Ordering note for anyone adding to this dict: `__init__` assigns `lattice` after
            # `period` because the setter derives `period` from the basis. That still holds on
            # the way back in, and it is correct: with a basis in play `period` is a reported
            # quantity rather than an independent one.
            # `.columns`, not `.tolist()` on the basis itself: `_lattice_basis` holds a
            # `Lattice` since the cell became a design variable, and a `Lattice` has no
            # `tolist`. Flattening an oblique solver raised `AttributeError` -- and because
            # `solve_field` is jitted at class level, `self` is flattened on every call, so
            # the *entire jitted field path* was unreachable on a skewed cell. Nothing caught
            # it: jax, an oblique cell and a field reconstruction are three features the suite
            # covered one at a time and never together.
            'lattice': (None if self._lattice_basis is None
                        else tuple(map(tuple, self._lattice_basis.columns.tolist()))),
            'normal_grid': self.normal_grid,
        }

        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        # `polarization` cannot ride through __init__: the psi child is always present (it
        # is the traced leaf, non-None even when it was merely derived from `pol`), so the
        # constructor would see both spellings of the incidence and refuse a state that was
        # legal when it was flattened. Rebuild without it, then restore the slot directly.
        aux_data = dict(aux_data)
        psi_given = aux_data.pop('psi_given')
        *core, chi, polarization, mu = children
        solver = cls(*core, chi=chi, mu=mu, **aux_data)
        # Straight into the slot, bypassing the setter's validation: the pair may now be a
        # tuple of tracers, and the setter's `(0, 0)` check reads concrete values. It already
        # guards that with a try/except, but there is nothing left to validate here anyway --
        # this pair was validated when the solver it came from was built.
        solver._polarization = polarization
        solver._psi_given = psi_given

        return solver
