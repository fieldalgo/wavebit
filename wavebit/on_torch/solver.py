from .emsolver.rcwa import RCWATorch
from .optimizer.optimizer import OptimizerTorch


class SolverTorch(RCWATorch, OptimizerTorch):

    def __init__(self, device=0, complex_dtype=0, *args, **kwargs):

        # The `complex_dtype` block that used to sit here was the seventh copy of the width
        # resolution and it was dead: `RCWATorch.__init__` below reaches `_BaseRCWA.__init__`,
        # which resolves the width again, and nothing between the two reads a dtype.
        self.device = device

        RCWATorch.__init__(self, device=device, complex_dtype=complex_dtype, *args, **kwargs)
        OptimizerTorch.__init__(self, device=device, complex_dtype=complex_dtype, *args, **kwargs)
