from .._core.modeler import Modeling
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
            'fourier_type': self.fourier_type,
            'enhanced_dfs': self.enhanced_dfs,
        }

        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        solver = cls(*children, **aux_data)

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
