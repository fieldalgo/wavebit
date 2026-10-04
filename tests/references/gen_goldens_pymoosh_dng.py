"""Freeze PyMoosh's answers for flat multilayers with **lossless** double negatives.

Writes ``golden_pymoosh_dng.npz``. PyMoosh pulls in matplotlib, scipy and refractiveindex, so it
gets an environment of its own, used by nothing but this script::

    python -m venv pymoosh-ref
    pymoosh-ref/bin/pip install PyMoosh==4.0.1
    pymoosh-ref/bin/python tests/references/gen_goldens_pymoosh_dng.py

PyMoosh is MIT-licensed (its ``LICENSE.txt`` and package metadata; the header of its
``__init__.py`` carries GPL boilerplate that ends by pointing to the MIT licence). Nothing of it is
vendored here: the golden holds its numbers, as the other goldens hold theirs.

Frozen with PyMoosh 4.0.1 and numpy 2.5.3 (Python 3.12), 2026-10-04; both are recorded in the
golden (``__pymoosh_version``, ``__numpy_version``). Another PyMoosh is refused: the
double-negative switch this golden exists to check lives in its ``coefficient_S``
(``_pymoosh_dng_cases.py`` quotes it), and a release that moved it would freeze a different
rule.

What is frozen, per case: ``{name}__kx`` (PyMoosh's ``alpha/k0`` at each angle) and
``{name}__pol{0,1}``, rows ``(r, t, R, T)`` per angle, from ``PyMoosh.coefficient_S`` (the
scattering-matrix method, the one with the switch). Refuses to freeze unless every row conserves
energy, ``|R + T - 1| < 1e-12``: every medium is lossless, lit or not, so nothing else is
possible, and a root that grew or ran toward the interface would break it.
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / 'golden_pymoosh_dng.npz'


def main():
    from importlib.metadata import version
    import PyMoosh
    from _pymoosh_dng_cases import CASES, PYMOOSH_VERSION, WL, media, pymoosh_kx, thicknesses

    if version('PyMoosh') != PYMOOSH_VERSION:
        raise SystemExit(f'PyMoosh {version("PyMoosh")} is not {PYMOOSH_VERSION}; read its '
                         'coefficient_S for the double-negative switch before freezing')
    out = {'__pymoosh_version': np.asarray(version('PyMoosh')),
           '__numpy_version': np.asarray(np.__version__)}
    for name, case in CASES.items():
        structure = PyMoosh.Structure([list(m) for m in media(name)],
                                      list(range(len(media(name)))), thicknesses(name),
                                      verbose=False)
        out[f'{name}__kx'] = np.asarray([pymoosh_kx(name, theta) for theta in case['thetas']])
        for pol in (0, 1):
            rows = np.asarray([PyMoosh.coefficient_S(structure, WL, theta, pol)
                               for theta in case['thetas']], dtype=complex)
            leak = np.abs(rows[:, 2].real + rows[:, 3].real - 1).max()
            if leak > 1e-12:
                raise SystemExit(f'{name}, polarization {pol}: R + T - 1 = {leak:.1e} in a '
                                 'lossless stack')
            out[f'{name}__pol{pol}'] = rows
        print(f'{name}: {len(case["thetas"])} angles, kx {out[name + "__kx"].round(4)}')
    np.savez(OUT, **out)
    print(f'wrote {OUT} ({len(out)} arrays)')


if __name__ == '__main__':
    main()
