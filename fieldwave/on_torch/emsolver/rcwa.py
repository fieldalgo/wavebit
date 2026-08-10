import torch

import numpy as np

from ._base import _BaseRCWA
from ..._core.engine import CoreFactorizationMixin
from ..._core.fields import field_1d, field_1d_conical, field_2d, field_plot
from ..._core.modeler import Modeling


class ResultTorch:
    def __init__(self, res=None, res_te_inc=None, res_tm_inc=None):

        self.res = res
        self.res_te_inc = res_te_inc
        self.res_tm_inc = res_tm_inc

    @property
    def de_ri(self):
        if self.res is not None:
            return self.res.de_ri
        elif self.res_te_inc is not None and self.res_tm_inc is not None:
            return 0.5 * (self.res_te_inc.de_ri + self.res_tm_inc.de_ri)
        elif self.res_te_inc is not None:
            return self.res_te_inc.de_ri
        elif self.res_tm_inc is not None:
            return self.res_tm_inc.de_ri
        else:
            return None

    @property
    def de_ti(self):
        if self.res is not None:
            return self.res.de_ti
        elif self.res_te_inc is not None and self.res_tm_inc is not None:
            return 0.5 * (self.res_te_inc.de_ti + self.res_tm_inc.de_ti)
        elif self.res_te_inc is not None:
            return self.res_te_inc.de_ti
        elif self.res_tm_inc is not None:
            return self.res_tm_inc.de_ti
        else:
            return None


class ResultSubTorch:
    def __init__(self, R_s, R_p, T_s, T_p, de_ri, de_ri_s, de_ri_p, de_ti, de_ti_s, de_ti_p):
        self.R_s = R_s
        self.R_p = R_p
        self.T_s = T_s
        self.T_p = T_p
        self.de_ri = de_ri
        self.de_ri_s = de_ri_s
        self.de_ri_p = de_ri_p

        self.de_ti = de_ti
        self.de_ti_s = de_ti_s
        self.de_ti_p = de_ti_p


class RCWATorch(CoreFactorizationMixin, _BaseRCWA):
    def modeling_block_instruction(self, *args, **kwargs):
        return self.modeler.modeling_block_instruction(*args, **kwargs)

    def rectangle(self, *args, **kwargs):
        return self.modeler.rectangle(*args, **kwargs)

    def ellipse(self, *args, **kwargs):
        return self.modeler.ellipse(*args, **kwargs)

    def __init__(self,
                 n_top=1.,
                 n_bot=1.,
                 theta=0.,
                 phi=None,
                 psi=None,
                 period=(1., 1.),
                 wavelength=1.,
                 ucell=None,
                 thickness=(0.,),
                 backend=2,
                 pol=0.,
                 fto=(0, 0),
                 ucell_materials=None,
                 connecting_algo='TMM',
                 perturbation=1E-20,
                 device='cpu',
                 type_complex=torch.complex128,
                 fourier_coefficients=None,  # 'sampled' (DFS) | 'integrated'
                 # (closed-form series); the default resolves per modeling path --
                 # 'sampled' on a raster, 'integrated' on both shape paths, which is
                 # the only series either can form. Naming the other one raises.
                 factorization=None,         # 'li' | 'nvf' (normal vector field); the
                 # default resolves per modeling path -- 'li' on a raster and on the
                 # vector modeler, 'nvf' on analytic shapes, which is the only rule
                 # each of those last two implements. Naming the other one raises.
                 fourier_type=None,          # deprecated alias for the two keywords above
                 normal_grid=1024,  # normal-field sampling; sets the floor on shape gradients
                 # (a translation-invariant structure should give d/dcx == 0 exactly; the
                 #  sampling leaves a sawtooth instead -- 5e-4 at 256, 4.2e-5 at 1024, for
                 #  ~1.2-2x the runtime. Raise further to fit positions below that.)
                 lattice=None,               # 2x2 cell basis (columns a1, a2); None = diag(period).
                 # Kept last on purpose: SolverJax._tree_unflatten passes the children
                 # positionally, so a new parameter anywhere earlier silently shifts them.
                 ):

        super().__init__(n_top=n_top, n_bot=n_bot, theta=theta, phi=phi, psi=psi, pol=pol,
                         fto=fto, period=period, wavelength=wavelength,
                         thickness=thickness, connecting_algo=connecting_algo, perturbation=perturbation,
                         device=device, type_complex=type_complex)

        self.modeler = Modeling(period=self.period, device=device, type_complex=type_complex)

        self._modeling_type_assigned = None
        self._grating_type_assigned = None
        self._ucell = None

        self.ucell = ucell
        self.ucell_materials = ucell_materials

        self.backend = backend
        self._set_fourier_options(fourier_coefficients, factorization, fourier_type)
        self.normal_grid = normal_grid
        self.lattice = lattice      # after period: assigning a basis overwrites period

    @property
    def ucell(self):
        return self._ucell

    @ucell.setter
    def ucell(self, ucell):

        if isinstance(ucell, torch.Tensor):  # Raster
            self._modeling_type_assigned = 0
            if ucell.dtype in (torch.complex128, torch.complex64):
                dtype = self.type_complex
                self._ucell = ucell.to(device=self.device, dtype=dtype)
            elif ucell.dtype in (torch.float64, torch.float32, torch.int64, torch.int32):
                dtype = self.type_float
                self._ucell = ucell.to(device=self.device, dtype=dtype)
            else:
                raise ValueError

        elif isinstance(ucell, np.ndarray):  # Raster
            self._modeling_type_assigned = 0
            if ucell.dtype in (np.int64, np.float64, np.int32, np.float32):
                dtype = self.type_float
                self._ucell = torch.tensor(ucell, device=self.device, dtype=dtype)
            elif ucell.dtype in (np.complex128, np.complex64):
                dtype = self.type_complex
                self._ucell = torch.tensor(ucell, device=self.device, dtype=dtype)
            else:
                raise ValueError

        elif type(ucell) is list:
            # Analytic: one dict(n_base=..., shapes=[...], combine=...) per layer.
            # Shape coordinates are absolute lengths in the units of `period`, not
            # fractions of it, and `combine` picks what overlapping shapes mean --
            # this path adds them where the block path paints. See
            # fieldwave._core.analytic; overlaps are refused until it is set.
            if len(ucell) > 0 and isinstance(ucell[0], dict):
                self._modeling_type_assigned = 2
                self._ucell = ucell
            else:  # Vector
                self._modeling_type_assigned = 1
                self._ucell = ucell
        elif ucell is None:
            self._ucell = ucell
            self._modeling_type_assigned = None
        else:
            raise ValueError

    @property
    def modeling_type_assigned(self):
        return self._modeling_type_assigned

    # @modeling_type_assigned.setter
    # def modeling_type_assigned(self, modeling_type_assigned):
    #     self._modeling_type_assigned = modeling_type_assigned

    def _assign_grating_type(self):
        # self.modeling_type_assigned = 0
        # self._grating_type_assigned = 1  # else

        if self.ucell is None:
            if self._grating_type_assigned is None:
                self._grating_type_assigned = 2
            return

        if self.modeling_type_assigned == 0:
            if self.ucell.shape[1] == 1:
                if (self.pol in (0, 1)) and (self.phi is None) and (self.fto[1] == 0):
                    self._grating_type_assigned = 0  # 1D TE and TM only
                else:
                    self._grating_type_assigned = 1  # 1D conical
            else:
                self._grating_type_assigned = 2  # else

        elif self.modeling_type_assigned == 1:
            self.grating_type_assigned = 2

        elif self.modeling_type_assigned == 2:  # Analytic (staircase-free shapes)
            self._grating_type_assigned = 2

    @property
    def grating_type_assigned(self):
        return self._grating_type_assigned

    @grating_type_assigned.setter
    def grating_type_assigned(self, grating_type_assigned):
        self._grating_type_assigned = grating_type_assigned

    def solve_for_conv(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all,
                       epxy_conv_all=None, epyx_conv_all=None):
        self._assign_grating_type()

        if self._grating_type_assigned == 0:
            result_dict = self.solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        elif self._grating_type_assigned == 1:
            result_dict = self.solve_1d_conical(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        else:
            result_dict = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all,
                                        epxy_conv_all, epyx_conv_all)

        res_psi = ResultSubTorch(**result_dict['res']) if 'res' in result_dict else None
        res_te_inc = ResultSubTorch(**result_dict['res_te_inc']) if 'res_te_inc' in result_dict else None
        res_tm_inc = ResultSubTorch(**result_dict['res_tm_inc']) if 'res_tm_inc' in result_dict else None

        result = ResultTorch(res_psi, res_te_inc, res_tm_inc)

        return result

    def calculate_field(self, res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False)):
        kx, ky = self._kxy_2d(self.wavelength)

        if self._grating_type_assigned == 0:
            res_y = 1
            field_cell = field_1d(self.wavelength, kx, self.T1, self.layer_info_list, self.period, self.pol,
                                       res_x=res_x, res_y=res_y, res_z=res_z)
        elif self._grating_type_assigned == 1:
            field_cell = field_1d_conical(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                               res_x=res_x, res_y=res_y, res_z=res_z, set_field_input=set_field_input,
                                       lattice=self.lattice)
        else:
            field_cell = field_2d(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                       res_x=res_x, res_y=res_y, res_z=res_z, set_field_input=set_field_input,
                                       lattice=self.lattice)

        return field_cell

    def conv_solve_field(self, res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False),
                         geometry=None, **kwargs):
        """``geometry`` is optional here — PyTorch traces eagerly, so auto-detection works. It is
        accepted so backend-agnostic code can pass the same frozen geometry JAX requires."""
        [setattr(self, k, v) for k, v in kwargs.items()]  # needed for optimization

        res = self.conv_solve(geometry=geometry)
        field_cell = self.calculate_field(res_x, res_y, res_z, set_field_input)
        return res, field_cell

    def field_plot(self, field_cell):
        field_plot(field_cell, self.pol)
