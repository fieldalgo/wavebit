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
        children = (self.n_top, self.n_bot, self.theta, self.phi, self.psi,
                    self.period, self.wavelength, self.ucell, self.thickness,
                    self._chi, self._polarization)
        aux_data = {
            'backend': self.backend,
            'tetm': self.tetm,
            'fto': self.fto,
            'ucell_materials': self.ucell_materials,
            'connecting_algo': self.connecting_algo,
            'perturbation': self.perturbation,
            'device': self.device,
            'type_complex': self.type_complex,
            # The *requested* values, not the resolved ones the properties report. Both axes
            # resolve against the modeling path, and a pytree round-trip has to reconstruct the
            # solver it flattened: writing the resolved value here would hand the rebuilt solver
            # an explicit request nobody made, so a structure that had merely accepted its
            # path's only option would come back out asserting it -- and, on a solver that did
            # name an unavailable one, the refusal would fire here, inside JAX's tree
            # machinery, rather than at the solve that is actually being refused.
            'fourier_coefficients': self._fourier_coefficients,
            'factorization': self._factorization,
            # A dispatch key like `pol`: the solves branch on it (which cascade blocks are
            # read, which way the stack is walked), so it must come back static, never traced.
            'incidence_side': self.incidence_side,
            # Not the pair itself (that is a child now) but the fact that `psi` was given
            # explicitly, which the always-present psi child would otherwise overwrite.
            'psi_given': self._psi_given,
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
        *core, chi, polarization = children
        solver = cls(*core, chi=chi, **aux_data)
        # Straight into the slot, bypassing the setter's validation: the pair may now be a
        # tuple of tracers, and the setter's `(0, 0)` check reads concrete values. It already
        # guards that with a try/except, but there is nothing left to validate here anyway --
        # this pair was validated when the solver it came from was built.
        solver._polarization = polarization
        solver._psi_given = psi_given

        return solver
