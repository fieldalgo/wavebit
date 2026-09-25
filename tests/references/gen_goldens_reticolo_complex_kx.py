"""Freeze RETICOLO at a complex transverse wavevector into ``golden_reticolo_complex_kx.npz``.

Run once, on a machine that has Octave and RETICOLO; ``test_reticolo_complex_kx_golden.py`` then
runs anywhere. The rules of ``gen_goldens_reticolo.py`` hold: rerun only to add cases or after
deliberately changing what is compared -- never to make a failing test pass.

    RETICOLO_PATH=/path/to/reticolo_allege_v10 python tests/references/gen_goldens_reticolo_complex_kx.py

The cases are imported from the test module rather than restated here.

The one thing this generator does that the others do not is open RETICOLO's mode selection.
``res2`` reports only the modes it counts as propagating, and at a complex ``k_parallel`` it
counts none: the order lists come back empty, with no error. The bridge's
``run_res2_all_modes`` raises ``tolh`` / ``tolb`` and hands them to ``res2`` as its third
argument -- on the parm given to ``res1`` they are ignored without a word, which is how the first
attempt returned empty lists at a tolerance of 100. With the selection open every retained order
comes back, and that count is checked here, so a silent disagreement about which orders exist
cannot become a golden.
"""
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # repo root, for `import wavebit`
sys.path.insert(0, str(HERE))              # the bridge and the test module beside it

# No default, for the reason gen_goldens_reticolo.py gives: where RETICOLO lives is a property of
# the machine, not of this project.
RETICOLO_PATH = os.environ.get('RETICOLO_PATH')
if not RETICOLO_PATH:
    raise SystemExit('set RETICOLO_PATH to a reticolo_allege_v10 directory to regenerate this '
                     'golden; the committed .npz is what the tests read.')
RETICOLO_PATH = os.path.expanduser(RETICOLO_PATH)

OUT = HERE / 'golden_reticolo_complex_kx.npz'


def _real(values, what):
    """RETICOLO's efficiencies, as the real numbers they are -- checked, not assumed."""
    values = np.asarray(values).ravel()
    if np.abs(np.imag(values)).max(initial=0.0) > 1e-12:
        raise SystemExit(f'{what}: RETICOLO returned complex efficiencies {values}')
    return np.real(values).astype(float)


def main():
    from oct2py import octave
    octave.addpath(octave.genpath(RETICOLO_PATH))

    from interface.Reticolo import Reticolo
    from test_reticolo_complex_kx_golden import CASES

    reti = Reticolo()
    out = {}
    for name, kw in CASES.items():
        pol = 0 if kw['tetm'] == 'te' else 1
        refl, tran = reti.run_res2_all_modes(
            period=kw['period'], truncation_order=kw['truncation_order'],
            structure=kw['structure'], thickness=kw['thickness'], theta=kw['theta'], pol=pol,
            wavelength=kw['wavelength'], top=kw['top'], bot=kw['bot'])

        n = kw['truncation_order']
        for kind, info in (('r', refl), ('t', tran)):
            order = np.rint(np.real(np.asarray(info.order).ravel())).astype(np.int64)
            if sorted(order.tolist()) != list(range(-n, n + 1)):
                raise SystemExit(f'{name}: RETICOLO reported orders {order.tolist()} for {kind}, '
                                 f'not the {2 * n + 1} retained ones -- is the selection open?')
            out[f'{name}__order_{kind}'] = order
            out[f'{name}__eff_{kind}'] = _real(info.efficiency, f'{name} {kind}')
        amplitude = np.asarray(refl.amplitude).ravel()
        out[f'{name}__r0'] = np.complex128(
            amplitude[int(np.flatnonzero(out[f'{name}__order_r'] == 0)[0])])
        print(f'{name:34s} orders={2 * n + 1:2d}  r0={complex(out[f"{name}__r0"]):.6f}  '
              f'sum R={out[f"{name}__eff_r"].sum():.6f}  sum T={out[f"{name}__eff_t"].sum():.6f}')

    np.savez_compressed(OUT, **out)
    print(f'\nwrote {OUT} ({OUT.stat().st_size} bytes, {len(CASES)} cases)')


if __name__ == '__main__':
    main()
