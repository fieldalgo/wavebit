"""The periodic reference route warns the public ``UnsettledFieldWarning`` (issue #55).

It used to define a second class of that name in a private module, which neither
``warnings.filterwarnings(category=wavebit.fdtd.diagnostics.UnsettledFieldWarning)`` nor
``pytest.warns`` saw, and its advice named the engine's ``n_periods``. The scene is the issue's:
a 300 nm grating period, a 100 nm layer on glass, 50 optical periods of a 900 nm wave, which
leaves 1.0e-2 and 1.1e-2 of the peak at the transmission monitor against a 1e-2 target.
"""
import warnings

import numpy as np
import pytest

from wavebit import Material, Units
from wavebit.fdtd import periodic
from wavebit.fdtd.diagnostics import UnsettledFieldWarning

UNITS = Units(length='nm')


def _short_run():
    problem = periodic.Problem(units=UNITS,
        stack=periodic.Stack(period=(300.,), bottom=Material(epsilon=1.44 ** 2),
            layers=(periodic.Layer(thickness=100., epsilon=np.array([[2.25, 2.25]])),)),
        excitation=periodic.Excitation(wavelength=900., sources=(periodic.PlaneWave(),)))
    numerics = periodic.Numerics(spacing=20., duration_seconds=50 * 900 * UNITS.time_unit_seconds,
                                 algorithm='reference')
    return problem, numerics


def test_the_reference_route_warns_the_public_class_with_public_advice():
    problem, numerics = _short_run()
    with pytest.warns(UnsettledFieldWarning, match='had not settled') as caught:
        periodic.run(problem, numerics=numerics)
    unsettled = [w for w in caught if issubclass(w.category, UnsettledFieldWarning)]
    assert unsettled
    for w in unsettled:
        assert w.category is UnsettledFieldWarning
        text = str(w.message)
        assert 'Raise periodic.Numerics(duration_seconds=...) (it is 1.5e-13 s)' in text, text
        assert 'n_periods' not in text, text
        assert w.filename == __file__, w.filename             # the caller's line


def test_a_public_filter_sees_it():
    problem, numerics = _short_run()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        warnings.filterwarnings('error', category=UnsettledFieldWarning)
        with pytest.raises(UnsettledFieldWarning):
            periodic.run(problem, numerics=numerics)


def test_the_engine_names_the_same_class():
    from wavebit.fdtd._periodic import legacy
    assert legacy.UnsettledFieldWarning is UnsettledFieldWarning
