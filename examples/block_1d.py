import numpy as np

import wavebit


def run():
    # Absorbing Si would be 3.638751670074983+0.007498295841854125j — Im(n) > 0 is loss
    # here, the same convention the bundled nk tables use (n + 1j*k). n_bot takes a complex
    # index just as a layer does.
    si = 3.638751670074983
    sio2 = 1.4518
    si3n4 = 2.0056

    # The silicon underneath is the substrate, so it goes in n_bot.
    rcwa_options = dict(backend=2, thickness=[205], period=[300, 300],
                        fto=[3, 0],
                        n_top=1, n_bot=si,
                        wavelength=900,
                        psi=np.pi / 4,
                        )

    ucell = [
        # layer 1
        [si3n4,
            [
                # obj 1
                ['rectangle', 50, 150, 31, 300, si, 0, 0, 0],
                # obj 2
                ['rectangle', 200, 150, 49.5, 300, si, 0, 0, 0],
            ],
        ],
    ]

    solver = wavebit.rcwa(**rcwa_options)
    solver.ucell = ucell

    result = solver.conv_solve()

    result_given_pol = result.res
    result_te_incidence = result.res_te_inc
    result_tm_incidence = result.res_tm_inc

    R, T = result_given_pol.R, result_given_pol.T
    de_ri1, de_ti1 = result_te_incidence.R, result_te_incidence.T
    de_ri2, de_ti2 = result_tm_incidence.R, result_tm_incidence.T

    print(R.sum(), T.sum())
    print(de_ri1.sum(), de_ti1.sum())
    print(de_ri2.sum(), de_ti2.sum())

    return


if __name__ == '__main__':

    run()

    print(0)
