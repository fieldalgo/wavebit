from .emsolver.rcwa import RCWAJax
from .optimizer.optimizer import OptimizerJax


class SolverJax(RCWAJax, OptimizerJax):

    def __init__(self, *args, **kwargs):
        RCWAJax.__init__(self, *args, **kwargs)
        OptimizerJax.__init__(self, *args, **kwargs)

    def _tree_flatten(self):
        children = (self.n_top, self.n_bot, self.theta, self.phi, self.psi,
                    self.period, self.wavelength, self.ucell, self.thickness)
        aux_data = {
            'backend': self.backend,
            'pol': self.pol,
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
            # Static, like `pol`: the pair is a plain tuple, not a traced leaf — `psi` is the
            # child that carries polarization gradients. Restored directly (see below),
            # along with the psi-was-explicit flag the psi child would otherwise overwrite.
            'polarization': self._polarization,
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
        polarization = aux_data.pop('polarization')
        psi_given = aux_data.pop('psi_given')
        solver = cls(*children, **aux_data)
        solver._polarization = polarization
        solver._psi_given = psi_given

        # ``pol`` and ``psi`` are one degree of freedom held twice: ``psi`` rides along as a
        # traced child (it enters the solve as cos/sin amplitudes, so gradients flow through
        # it), while ``pol`` is aux data because the solve *dispatches* on it -- 1D TE vs TM
        # are different code paths, and `field_1d` branches on it too. ``__init__`` applies
        # ``pol`` first and then ``psi``, whose setter recomputes ``_pol`` from it; under a
        # trace that silently turns the static dispatch key into a tracer and every
        # ``if pol == 0`` downstream raises TracerBoolConversionError. Restore the static
        # value the aux data already carries.
        solver._pol = aux_data['pol']
        return solver
