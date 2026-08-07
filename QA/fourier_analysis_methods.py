import numpy as np

from fieldwave import rcwa


def compare_conv_mat_method(backend, type_complex, device):

    pol = 1  # 0: TE, 1: TM
    n_top = 1.45  # n_incidence
    n_bot = 1  # n_transmission

    theta = 0
    phi = 0

    wavelength = 900
    fto = 80

    ucell = [1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1,
             1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1]
    ucell = np.array(ucell).reshape((1, 1, -1))
    ucell = (ucell + 1) * 1.5 + 1

    thickness = [325]
    period = [abs(wavelength / np.sin(60 / 180 * np.pi))]
    solver = rcwa(backend, pol=pol, n_top=n_top, n_bot=n_bot, theta=theta, phi=phi,
                   fto=fto, wavelength=wavelength, period=period, ucell=ucell,
                   thickness=thickness, device=device, type_complex=type_complex, )

    solver.fourier_coefficients = 'sampled'
    res_dfs = solver.conv_solve().res
    de_ri_dfs, de_ti_dfs = res_dfs.de_ri, res_dfs.de_ti

    res_efs = solver.conv_solve().res
    de_ri_efs, de_ti_efs = res_efs.de_ri, res_efs.de_ti

    solver.fourier_coefficients = 'integrated'
    res_cfs = solver.conv_solve().res
    de_ri_cfs, de_ti_cfs = res_cfs.de_ri, res_cfs.de_ti

    a = np.linalg.norm(de_ri_dfs - de_ri_efs)
    b = np.linalg.norm(de_ti_dfs - de_ti_efs)
    c = np.linalg.norm(de_ri_dfs - de_ri_cfs)
    d = np.linalg.norm(de_ti_dfs - de_ti_cfs)
    e = np.linalg.norm(de_ri_efs - de_ri_cfs)
    f = np.linalg.norm(de_ti_efs - de_ti_cfs)

    print('Norm of DFS-EFS: ', a, b)
    print('Norm of DFS-CFS: ', c, d)
    print('Norm of EFS-CFS: ', e, f)


if __name__ == '__main__':
    dtype = 0
    device = 0

    print('NumpyMeent')
    compare_conv_mat_method(0, type_complex=dtype, device=device)
    print('JaxMeent')
    compare_conv_mat_method(1, type_complex=dtype, device=device)
    print('TorchMeent')
    compare_conv_mat_method(2, type_complex=dtype, device=device)
