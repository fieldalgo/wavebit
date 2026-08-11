import torch

import fieldwave


def run():
    # Absorbing Si would be 3.638751670074983+0.007498295841854125j — Im(n) > 0 is loss
    # here, the same convention the bundled nk tables use (n + 1j*k). n_bot takes a complex
    # index just as a layer does.
    si = 3.638751670074983
    sio2 = 1.4518
    si3n4 = 2.0056

    # The silicon underneath is the substrate, so it goes in n_bot.
    rcwa_options = dict(backend=1, thickness=[205, 305], period=[300, 300],
                        fto=[3, 3],
                        n_top=1, n_bot=si,
                        wavelength=900,
                        pol=0.5,
                        )

    ucell = [
        # layer 1
        [sio2,
            [
                # obj 1
                ['ellipse', 75, 225, 101.5, 81.5, si, 20 * torch.pi / 180, 40, 40],
                # obj 2
                ['rectangle', 225, 75, 98.5, 81.5, si, 0, 0, 0],
            ],
        ],
        # layer 2
        [si3n4,
            [
                # obj 1
                ['rectangle', 50, 150, 31, 300, si, 0, 0, 0],
                # obj 2
                ['rectangle', 200, 150, 49.5, 300, si, 0, 0, 0],
            ],
        ],
    ]

    solver = fieldwave.rcwa(**rcwa_options)
    solver.ucell = ucell

    result = solver.conv_solve()

    result_given_pol = result.res
    result_te_incidence = result.res_te_inc
    result_tm_incidence = result.res_tm_inc

    de_ri, de_ti = result_given_pol.de_ri, result_given_pol.de_ti
    de_ri1, de_ti1 = result_te_incidence.de_ri, result_te_incidence.de_ti
    de_ri2, de_ti2 = result_tm_incidence.de_ri, result_tm_incidence.de_ti

    print(de_ri.sum(), de_ti.sum())
    print(de_ri1.sum(), de_ti1.sum())
    print(de_ri2.sum(), de_ti2.sum())

    return


if __name__ == '__main__':

    run()

    print(0)
