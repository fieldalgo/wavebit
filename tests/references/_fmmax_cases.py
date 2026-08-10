"""Cases shared by ``gen_goldens_fmmax.py`` and ``test_fmmax_golden.py``.

They live here rather than in the test module — the pattern the RETICOLO pair uses — because
the two run in *different environments*. fmmax pins its own JAX, so the generator runs in a
separate ``fwref`` env that has no ``fieldwave`` in it, and the test runs in the project env
that has no ``fmmax``. Neither can import the other's module. This one imports numpy and
nothing else, so both can read it.

The reason for the case list living in one place at all is the one
``gen_goldens_reticolo.py`` gives: a copy that must stay identical is a copy that will not.
"""
import numpy as np

WL = 620.0
PERIOD = 340.0
THICK = 185.0


def tensor(exx, eyy, ezz, exy=0.0, eyx=0.0):
    """The physical permittivity tensor. Slot negation happens at the caller, not here."""
    return np.array([[exx, exy, 0.0], [eyx, eyy, 0.0], [0.0, 0.0, ezz]], dtype=complex)


def _rotated_uniaxial(eo, ee, phi):
    c, s = np.cos(phi), np.sin(phi)
    return tensor(ee * c ** 2 + eo * s ** 2, ee * s ** 2 + eo * c ** 2, eo,
                  (ee - eo) * s * c, (ee - eo) * s * c)


# name -> (tensor, theta in degrees). Deliberately not the RETICOLO set: different wavelength,
# period and thickness, different tensors, and angles chosen off that grid. Two oracles that
# agree only where they were handed the same numbers would not be telling us much.
CASES = {
    'isotropic_normal': (tensor(2.56, 2.56, 2.56), 0.0),
    'isotropic_28deg': (tensor(2.56, 2.56, 2.56), 28.0),
    'uniaxial_z_50deg': (tensor(2.4, 2.4, 3.7), 50.0),
    'biaxial_normal': (tensor(2.2, 3.1, 2.75), 0.0),
    'biaxial_62deg': (tensor(2.2, 3.1, 2.75), 62.0),
    'rotated_uniaxial_18deg_normal': (_rotated_uniaxial(2.3, 3.9, np.deg2rad(18.0)), 0.0),
    'rotated_uniaxial_18deg_44deg': (_rotated_uniaxial(2.3, 3.9, np.deg2rad(18.0)), 44.0),
    'rotated_uniaxial_67deg_31deg': (_rotated_uniaxial(2.3, 3.9, np.deg2rad(67.0)), 31.0),
    'gyrotropic_g030_normal': (tensor(2.56, 2.56, 2.56, 0.30j, -0.30j), 0.0),
    'gyrotropic_g030_37deg': (tensor(2.56, 2.56, 2.56, 0.30j, -0.30j), 37.0),
    'gyrotropic_g065_58deg': (tensor(2.56, 2.56, 2.56, 0.65j, -0.65j), 58.0),
    'absorbing_gyrotropic_25deg': (tensor(2.56 + 0.12j, 2.56 + 0.12j, 2.56 + 0.12j,
                                          0.30j, -0.30j), 25.0),
}
