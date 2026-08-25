from .emsolver.rcwa import RCWAJax
from .optimizer.optimizer import OptimizerJax


#: The flattened children, in order, by the name each takes on `__init__`. `polarization` is
#: the one that cannot ride through the constructor -- see `_tree_unflatten` -- and is popped
#: there; everything else goes in as a keyword, so this order is a *listing* rather than a
#: contract with `__init__`'s signature.
CHILD_NAMES = ('top', 'bot', 'theta', 'phi', 'psi', 'period', 'wavelength', 'structure',
               'thickness', 'chi', 'polarization', 'mu', 'excitation')

#: Read through the private slot rather than the property: these three report a *resolved*
#: value, and a round trip has to carry what was asked for.
_PRIVATE = frozenset({'chi', 'polarization', 'mu'})


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
        # `excitation` is a child, and JAX's own dict handling is what makes that the right
        # slot: a dict flattens with its **keys in the treedef** and its **values as leaves**,
        # which is exactly the split the property documents -- "they may be traced ... while
        # the keys may not, because which entries of the incident column are non-zero decides
        # array structure rather than a value". Amplitudes are a legitimate design variable,
        # so putting the whole dict in aux would have made a differentiated beam weight
        # return a compile-time constant, the way `polarization` did above.
        #
        # It was in *neither* slot until now, which is worse than the wrong slot: the jitted
        # `solve_field` rebuilds `self` on every call, so a multimode scene came back with
        # `excitation = None` and the solve answered the single-specular-wave problem.
        # Measured on a 1D grating, eager against jitted: the field parted by 1.22 and R read
        # 0.148814934 where eager gave 0.164651118 -- and 0.148814934 is exactly the R of the
        # same scene built with no excitation at all, which is the tell. `order_window`
        # narrowed with it, ((0, 3), (0, 0)) -> ((0, 2), (0, 0)).
        #
        # `mu` is the raster permeability field, a child for the same reason `structure` is:
        # it is material values, and material values are what gradients are for. It rides
        # last, after `polarization`, so the leading tuple still lines up with `__init__`'s
        # positional parameters; `None` is an empty subtree, so a non-magnetic solver
        # flattens exactly as it did before.
        # The names, beside the values, because `_tree_unflatten` rebuilds by keyword. They
        # used to be matched by *position* against `__init__`'s leading parameters, which made
        # the two orders one undeclared contract: adding a positional parameter anywhere but
        # the end would have shifted every child by one and rebuilt a different solver, in
        # silence. Two comments below warned about it; now nothing has to.
        children = tuple(getattr(self, '_' + n, None) if n in _PRIVATE else getattr(self, n)
                         for n in CHILD_NAMES)
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
        named = dict(zip(CHILD_NAMES, children))
        polarization = named.pop('polarization')
        solver = cls(**named, **aux_data)
        # Straight into the slot, bypassing the setter's validation: the pair may now be a
        # tuple of tracers, and the setter's `(0, 0)` check reads concrete values. It already
        # guards that with a try/except, but there is nothing left to validate here anyway --
        # this pair was validated when the solver it came from was built.
        solver._polarization = polarization
        solver._psi_given = psi_given

        return solver
