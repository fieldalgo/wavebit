import numpy as np

from ._base import _BaseRCWA
from ..._core.engine import CoreFactorizationMixin
from .field_distribution import field_plot, field_dist_1d, field_dist_1d_conical, field_dist_2d


class ResultNumpy:
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


class ResultSubNumpy:
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


from ..modeler.modeling import ModelingNumpy


class RCWANumpy(CoreFactorizationMixin, _BaseRCWA):
    def modeling_vector_instruction(self, *args, **kwargs):
        return self.modeler.modeling_vector_instruction(*args, **kwargs)

    def rectangle(self, *args, **kwargs):
        return self.modeler.rectangle(*args, **kwargs)

    def ellipse(self, *args, **kwargs):
        return self.modeler.ellipse(*args, **kwargs)

    def run_ucell_vmap(self, ucell_list):
        """
        under development
        """
        pass

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
                 backend=0,
                 pol=0.,
                 fto=(0, 0),
                 ucell_materials=None,
                 connecting_algo='TMM',
                 perturbation=1E-20,
                 type_complex=np.complex128,
                 fourier_type=0,  # 0 DFS, 1 CFS, 2 NVF
                 enhanced_dfs=True,
                 use_pinv=False,
                 analytic_normal_grid=256,  # grid for sampling analytic normal field (modeling_type 2)
                 ):

        super().__init__(n_top=n_top, n_bot=n_bot, theta=theta, phi=phi, psi=psi, pol=pol,
                         fto=fto, period=period, wavelength=wavelength,
                         thickness=thickness, connecting_algo=connecting_algo, perturbation=perturbation,
                         type_complex=type_complex, use_pinv=use_pinv)

        self.modeler = ModelingNumpy(period=self.period, type_complex=type_complex)

        self._modeling_type_assigned = None
        self._grating_type_assigned = None
        self._ucell = None

        self.ucell = ucell
        self.ucell_materials = ucell_materials

        self.backend = backend
        self.fourier_type = fourier_type
        self.enhanced_dfs = enhanced_dfs
        self.use_pinv = use_pinv
        self.analytic_normal_grid = analytic_normal_grid

    @property
    def ucell(self):
        return self._ucell

    @ucell.setter
    def ucell(self, ucell):

        if isinstance(ucell, np.ndarray):  # Raster
            self._modeling_type_assigned = 0
            if ucell.dtype in (np.int64, np.float64, np.int32, np.float32):
                dtype = self.type_float
            elif ucell.dtype in (np.complex128, np.complex64):
                dtype = self.type_complex
            else:
                raise ValueError
            # self._ucell = np.array(ucell, dtype=dtype)
            self._ucell = ucell.astype(dtype)

        elif type(ucell) is list:
            if len(ucell) > 0 and isinstance(ucell[0], dict):  # Analytic (list of layer dicts)
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
        """
        Select the grating type for RCWA simulation. This decides the efficient formulation for given case.

        `_grating_type_assigned` == 0(1D TETM)    is for 1D grating, no rotation (phi or azimuth), and either TE or TM.
        `_grating_type_assigned` == 1(1D conical) is for 1D grating with generality.
        `_grating_type_assigned` == 2(2D)         is for 2D grating with generality.

        Note that no rotation means 'phi' is `None`. If phi is given as '0', then it takes 1D conical form
         even though when the case itself is 1D TETM.

        1D conical is under implementation.

        Returns:

        """
        if self.ucell is None:
            if self._grating_type_assigned is None:
                self._grating_type_assigned = 2
            return

        if self.modeling_type_assigned == 0:  # Raster
            if self.ucell.shape[1] == 1:
                if (self.pol in (0, 1)) and (self.phi is None) and (self.fto[1] == 0):
                    self._grating_type_assigned = 0  # 1D TE and TM only
                else:
                    self._grating_type_assigned = 1  # 1D conical
            else:
                self._grating_type_assigned = 2  # else

        elif self.modeling_type_assigned == 1:  # Vector
            self.grating_type_assigned = 2  # TODO: 1d conical case

        elif self.modeling_type_assigned == 2:  # Analytic (shape-based, staircase-free)
            self._grating_type_assigned = 2

    @property
    def grating_type_assigned(self):
        return self._grating_type_assigned

    @grating_type_assigned.setter
    def grating_type_assigned(self, grating_type_assigned):
        self._grating_type_assigned = grating_type_assigned

    def solve_for_conv(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all, epxy_conv_all=None, epyx_conv_all=None):
        self._assign_grating_type()

        if self.grating_type_assigned == 0:
            result_dict = self.solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        elif self._grating_type_assigned == 1:
            result_dict = self.solve_1d_conical(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        else:
            result_dict = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all, epxy_conv_all, epyx_conv_all)

        res_psi = ResultSubNumpy(**result_dict['res']) if 'res' in result_dict else None
        res_te_inc = ResultSubNumpy(**result_dict['res_te_inc']) if 'res_te_inc' in result_dict else None
        res_tm_inc = ResultSubNumpy(**result_dict['res_tm_inc']) if 'res_tm_inc' in result_dict else None

        result = ResultNumpy(res_psi, res_te_inc, res_tm_inc)

        return result

    def calculate_field(self, res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False)):
        # TODO: change res_ to accept array of points.
        kx, ky = self.get_kx_ky_vector(wavelength=self.wavelength)

        if self._grating_type_assigned == 0:
            res_y = 1
            field_cell = field_dist_1d(self.wavelength, kx, self.T1, self.layer_info_list, self.period, self.pol,
                                       res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
        elif self._grating_type_assigned == 1:
            # TODO other bds
            field_cell = field_dist_1d_conical(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                               res_x=res_x, res_y=res_y, res_z=res_z, set_field_input=set_field_input,
                                               type_complex=self.type_complex)
        else:
            field_cell = field_dist_2d(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                       res_x=res_x, res_y=res_y, res_z=res_z, set_field_input=set_field_input,
                                       type_complex=self.type_complex)

        return field_cell

    def conv_solve_field(self, res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False)):
        res = self.conv_solve()
        field_cell = self.calculate_field(res_x, res_y, res_z, set_field_input)
        return res, field_cell

    def field_plot(self, field_cell):
        field_plot(field_cell, self.pol)
