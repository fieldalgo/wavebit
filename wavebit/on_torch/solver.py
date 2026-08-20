import torch
import numpy as np

from .emsolver.rcwa import RCWATorch
from .optimizer.optimizer import OptimizerTorch


class SolverTorch(RCWATorch, OptimizerTorch):

    def __init__(self, device=0, complex_dtype=0, *args, **kwargs):

        # device
        if device in (0, 'cpu'):
            self._device = torch.device('cpu')
        elif device in (1, 'gpu', 'cuda'):
            self._device = torch.device('cuda')
        elif type(device) is torch.device:
            self._device = device
        else:
            raise ValueError('device')
        #
        # complex_dtype
        if complex_dtype in (0, torch.complex128, np.complex128):
            self._complex_dtype = torch.complex128
        elif complex_dtype in (1, torch.complex64, np.complex64):
            self._complex_dtype = torch.complex64
        else:
            raise ValueError('Torch complex_dtype')
        self._float_dtype = (torch.float64 if self._complex_dtype is not torch.complex64
                             else torch.float32)
        self._int_dtype = torch.int64 if self._complex_dtype is not torch.complex64 else torch.int32
        self.device = device

        RCWATorch.__init__(self, device=device, complex_dtype=complex_dtype, *args, **kwargs)
        OptimizerTorch.__init__(self, device=device, complex_dtype=complex_dtype, *args, **kwargs)
