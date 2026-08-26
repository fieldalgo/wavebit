import numpy as np

import wavebit


def run():
    # A material carries permittivity and permeability together, and permittivity is the
    # quantity the API takes. These three are non-magnetic, so `Material(eps)` is enough --
    # and `eps_from_index` is here because every nk datasheet hands you the other quantity.
    # Writing the conversion out is the point: 3.638... reread as a permittivity is a legal,
    # energy-conserving, different silicon, and nothing downstream can flag it.
    #
    # Absorbing Si would be eps_from_index(3.638751670074983+0.007498295841854125j).
    # Im(eps) > 0 is loss here, which is the same convention the bundled nk tables use
    # (n + 1j*k) -- eps_from_index carries it over, since the principal square of the upper
    # half plane stays in it.
    si = wavebit.Material(wavebit.eps_from_index(3.638751670074983))
    sio2 = wavebit.Material(wavebit.eps_from_index(1.4518))
    si3n4 = wavebit.Material(wavebit.eps_from_index(2.0056))

    # The silicon underneath is the substrate, so it goes in `bot`. Both half-spaces take a
    # Material too -- `top=1` is the scalar shorthand for Material(epsilon=1), i.e. vacuum.
    rcwa_options = dict(backend=2, thickness=[205], period=dict(x=300, y=300),
                        truncation_order=dict(x=3, y=0),
                        top=1, bot=si,
                        wavelength=900,
                        psi=np.pi / 4,
                        )

    # One dict per layer: a background material, and the shapes drawn on it. `cx`, `cy`, `lx`
    # and `ly` are absolute lengths in the units of `period` -- not fractions of it -- so these
    # rods are 31 and 49.5 wide on a 300 period and run the full height of the cell.
    #
    # Each shape's Fourier transform is taken in closed form, so a tilted or curved edge carries
    # no staircase error. Shapes paint, later over earlier, where they overlap; these two do not,
    # and a layer with nothing overlapping stays on the closed-form builder.
    structure = [
        # layer 1
        dict(background=si3n4, shapes=[
            # obj 1
            dict(type='rectangle', cx=50, cy=150, lx=31, ly=300, angle=0, material=si),
            # obj 2
            dict(type='rectangle', cx=200, cy=150, lx=49.5, ly=300, angle=0, material=si),
        ]),
    ]

    solver = wavebit.rcwa(**rcwa_options)
    solver.structure = structure

    result = solver.solve()

    result_given_pol = result.given
    result_te_incidence = result.s_inc
    result_tm_incidence = result.p_inc

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
