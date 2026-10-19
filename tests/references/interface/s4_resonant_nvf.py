"""S4 side of the resonant-NVF contact comparison, run in S4's own interpreter.

S4 is GPL-licensed, so it never shares a process with Wavebit. This bridge
imports only NumPy and S4, never Wavebit. The Wavebit side reads what it writes:

- ``--output FILE.npz`` freezes the cases in :data:`CASES` for
  ``tests/references/test_s4_resonant_nvf_golden.py``;
- ``--orders ... --methods ...`` prints one JSON line per solve for
  ``studies/rcwa/nvf_resonant_spectrum/resonant_nvf_probe.py``, which starts
  this script with the interpreter named by ``S4_PYTHON``.

Run from the repository root with an interpreter that has S4 installed:

    "$S4_PYTHON" tests/references/interface/s4_resonant_nvf.py --output artifacts/golden_s4_resonant_nvf.npz
"""

import argparse
import json
import sys
import time

import numpy as np

BASES = {'s4-direct': None, 's4-default': 'Default', 's4-normal': 'Normal', 's4-jones': 'Jones'}


def fixture_shape():
    angle = np.arctan2(1., 2.)
    return dict(type='rectangle', cx=0., cy=0., lx=640.,
                ly=700.*np.sin(angle), angle=angle, material=12.25)


def uniform_slab():
    """A full-cell rectangle: the oblique uniform-slab control."""
    return dict(type='rectangle', lx=700., ly=700., material=4.)


def rotated_equivalent(shape):
    """The same rectangle stated with its axes swapped and turned a quarter turn."""
    return dict(shape, lx=shape['ly'], ly=shape['lx'], angle=shape['angle']+np.pi/2)


def shifted_fixture(shift):
    return dict(fixture_shape(), cx=shift[0], cy=shift[1])


SHIFTS = {'centred': (0., 0.), 'shifted': (41., -27.)}

# name -> (order, shape, method). The Wavebit test pairs each with its own solve.
CASES = {
    **{f'uniform_{method}': (1, uniform_slab(), method) for method in BASES},
    'rotation_fixture': (2, fixture_shape(), 's4-direct'),
    'rotation_equivalent': (2, rotated_equivalent(fixture_shape()), 's4-direct'),
    **{f'contact_{label}_order{order}': (order, shifted_fixture(shift), 's4-direct')
       for order in (2, 4) for label, shift in SHIFTS.items()},
}


def s4_solve(order, shape=None, *, basis=None):
    """Independent S4 solve on a checked square order set; None selects direct rule."""
    import S4
    shape = fixture_shape() if shape is None else shape
    s = S4.New(Lattice=((700., 0.), (0., 700.)), NumBasis=(2*order+1)**2)
    if basis not in (None, 'Default', 'Normal', 'Jones'):
        raise ValueError(f'unknown S4 polarization basis: {basis!r}')
    options = dict(LatticeTruncation='Parallelogramic', PolarizationDecomposition=basis is not None)
    if basis is not None:
        options['PolarizationBasis'] = basis
    s.SetOptions(**options)
    for name, epsilon in (('air', 1.), ('substrate', 2.25), ('inclusion', float(shape['material']))):
        s.SetMaterial(Name=name, Epsilon=epsilon)
    s.AddLayer(Name='top', Thickness=0., Material='air')
    uniform = shape['lx'] == 700. and shape['ly'] == 700. and shape.get('angle', 0.) == 0.
    # A full-cell rectangle is a homogeneous periodic layer. Stating it as such
    # avoids S4's Default/Jones zero-vector failure on this validation control.
    s.AddLayer(Name='film', Thickness=300., Material='inclusion' if uniform else 'air')
    if not uniform:
        s.SetRegionRectangle(Layer='film', Material='inclusion',
                            Center=(float(shape.get('cx', 0.)), float(shape.get('cy', 0.))),
                            Angle=float(np.rad2deg(shape.get('angle', 0.))),
                            Halfwidths=(float(shape['lx'])/2, float(shape['ly'])/2))
    s.AddLayer(Name='bottom', Thickness=0., Material='substrate')
    actual = set(s.GetBasisSet())
    expected = {(m, n) for m in range(-order, order+1) for n in range(-order, order+1)}
    if actual != expected:
        raise ValueError('S4 reciprocal-order set differs from the requested square truncation')
    s.SetFrequency(1./900.)
    s.SetExcitationPlanewave(IncidenceAngles=(float(np.rad2deg(.2)), 0.),
                            sAmplitude=0., pAmplitude=1.)
    incoming, reflected = s.GetPowerFlux(Layer='top')
    transmitted, _ = s.GetPowerFlux(Layer='bottom')
    fluxes = np.asarray([incoming, reflected, transmitted], dtype=complex)
    if not np.isfinite(fluxes).all() or incoming.real <= 0:
        raise FloatingPointError('S4 returned invalid fluxes; this solve is not a reference')
    return float(transmitted.real/incoming.real), float(-reflected.real/incoming.real)


def solve(order, method, shape=None):
    """Total (T, R) for one of the probe's S4 methods."""
    if method not in BASES:
        raise ValueError(f'unknown S4 method: {method!r}')
    return s4_solve(order, shape, basis=BASES[method])


def freeze(output):
    import S4
    arrays = {name: np.array(solve(order, method, shape)) for name, (order, shape, method) in CASES.items()}
    arrays['__cases'] = np.array(json.dumps(
        {name: dict(order=order, method=method, shape={k: float(v) if k != 'type' else v
                                                       for k, v in shape.items()})
         for name, (order, shape, method) in CASES.items()}, sort_keys=True))
    arrays['__s4_version'] = np.array(str(getattr(S4, '__version__', 'unknown')))
    arrays['__numpy_version'] = np.array(np.__version__)
    np.savez(output, **arrays)
    print(f'wrote {output} ({len(CASES)} cases)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', help='freeze CASES into this .npz')
    parser.add_argument('--orders', nargs='+', type=int, default=[])
    parser.add_argument('--methods', nargs='+', choices=sorted(BASES), default=['s4-direct'])
    args = parser.parse_args()
    if args.output:
        freeze(args.output)
    for order in args.orders:
        for method in args.methods:
            start = time.monotonic()
            transmission, reflection = solve(order, method)
            print(json.dumps(dict(order=order, method=method, T=transmission, R=reflection,
                                  seconds=time.monotonic()-start)), flush=True)


if __name__ == '__main__':
    sys.exit(main())
