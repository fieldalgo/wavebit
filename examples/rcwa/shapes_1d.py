"""Typed shape declarations, mixed/s/p illumination and lossless power balance."""
import numpy as np

from wavebit import Material, Units, eps_from_refractive_index, rcwa


def make_problem():
    # Datasheet indices are converted explicitly to relative permittivity.
    si = Material(epsilon=eps_from_refractive_index(3.638751670074983))
    si3n4 = Material(epsilon=eps_from_refractive_index(2.0056))
    return rcwa.Problem(
        units=Units(length="nm"),
        stack=rcwa.Stack(period=(300., 300.), bottom=si, layers=(
            rcwa.Layer(thickness=205., background=si3n4, regions=(
                rcwa.Region(geometry=rcwa.Rectangle(size=(31., 300.), center=(-100., 0.)), material=si),
                rcwa.Region(geometry=rcwa.Rectangle(size=(49.5, 300.), center=(50., 0.)), material=si),
            )),
        )),
        excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization=(1., 1.)),)),
    )


def run():
    problem = make_problem()
    numerics = rcwa.Numerics(orders=(3, 0), algorithm="etm")
    prepared = rcwa.prepare(problem, numerics=numerics, backend="torch")
    results = {}
    # Each physical illumination is explicit; no hidden result.incident/s_inc/p_inc state.
    for label, polarization in (("mixed", (1., 1.)), ("s", "s"), ("p", "p")):
        source = rcwa.PlaneWave(polarization=polarization)
        live = problem.replace(excitation=problem.excitation.replace(sources=(source,)))
        # Changing the polarization declaration's layout requires its own preparation.
        result = (prepared.run(live) if label == "mixed"
                  else rcwa.run(live, numerics=numerics, backend="torch"))
        r, t = float(result.R.sum()), float(result.T.sum())
        assert np.isfinite([r, t]).all()
        np.testing.assert_allclose(r+t, 1., atol=2e-11, rtol=0)
        print(label, r, t)
        results[label] = result
    return results


if __name__ == "__main__":
    run()
