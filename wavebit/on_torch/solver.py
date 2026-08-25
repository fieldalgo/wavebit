from .emsolver.rcwa import RCWATorch
from .optimizer.optimizer import OptimizerTorch


class SolverTorch(RCWATorch, OptimizerTorch):

    # `(*args, **kwargs)` and nothing else, which is what `SolverNumpy` and `SolverJax` are.
    # This one used to name `device` and `complex_dtype` before them, and naming them here is
    # not a default -- it is a claim on the first two POSITIONS. `rcwa()` forwards `*args`
    # positionally, so `rcwa(0, 1.5, 2.25)` and `rcwa(1, 1.5, 2.25)` build a solver with
    # `top=1.5, bot=2.25` while `rcwa(2, 1.5, 2.25)` read 1.5 as a device and refused. Same
    # call, one backend answering a different question -- and the two names it intercepted are
    # the two that already had defaults one layer down, so it was not adding anything either.
    #
    # The forwarding was equally redundant: `device=0` and `complex_dtype=0` resolve to exactly
    # what `RCWATorch.__init__`'s own `'cpu'` and `torch.complex128` resolve to, through
    # `_resolve_device` and `_resolve_precision`. Removing them leaves construction identical.
    def __init__(self, *args, **kwargs):
        RCWATorch.__init__(self, *args, **kwargs)
        OptimizerTorch.__init__(self, *args, **kwargs)
