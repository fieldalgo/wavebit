"""Freeze new black-box Meep magnetic/Drude/Lorentz crossed references.

    conda run -n meep python tests/references/gen_goldens_meep_crossed_materials.py --output artifacts/materials.npz

Never read Meep source. This harness only calls its public API and reuses the existing
crossed monitor/frame bridge. No Wavebit solver is imported. Output must not exist;
existing goldens are never overwritten to make tests pass.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import time

sys.tracebacklimit = 0
import numpy as np

from _meep_crossed_material_cases import CASES, epsilon
from gen_goldens_meep_crossed import _run, _amplitudes


def material_factory(case):
    import meep as mp
    # Public API specifies frequency and gamma as angular values divided by 2*pi.
    # https://meep.readthedocs.io/en/master/Python_User_Interface/#lorentziansusceptibility
    # Drude frequency is a numerator scale, not the zero resonance frequency.
    # https://meep.readthedocs.io/en/master/Python_User_Interface/#drudesusceptibility
    def material_at(layer, j, i):
        eps_inf = float(case['index'][layer, j, i]**2)
        terms = []
        if case['poles']:
            identity = int(case['dispersion'][layer, j, i])
            for omega0, gamma, strength in case['poles'].get(identity, []):
                if omega0 == 0.:
                    terms.append(mp.DrudeSusceptibility(frequency=1., gamma=gamma/(2*np.pi),
                                                        sigma=strength/(2*np.pi)**2))
                else:
                    terms.append(mp.LorentzianSusceptibility(frequency=omega0/(2*np.pi),
                                                             gamma=gamma/(2*np.pi),
                                                             sigma=strength/omega0**2))
        return mp.Medium(epsilon=eps_inf, mu=float(case['mu'][layer, j, i]),
                         E_susceptibilities=terms)
    # Independently check the API's actual material response, not just our conversion.
    for wavelength in (.75, .9, 1.05):
        want = epsilon(case, wavelength)
        for index in np.ndindex(want.shape):
            material = material_at(*index)
            np.testing.assert_allclose(np.diag(material.epsilon(1/wavelength)), want[index],
                                       rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(np.diag(material.mu(1/wavelength)), case['mu'][index],
                                       rtol=1e-12, atol=1e-12)
    return material_at


def main():
    import meep as mp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; choose a new path. Frozen references are immutable.')
    files = [Path(__file__), Path(__file__).with_name('_meep_crossed_material_cases.py'),
             Path(__file__).with_name('gen_goldens_meep_crossed.py')]
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    out = dict(meep_version=np.array(mp.__version__), generator_sha256=np.array(digest),
               complete=np.array(False), resolutions=np.array([20, 40]),
               gap=np.array(.1), df_frac=np.array(.2))
    # All three cases share incident half-space, lattice, angles and wavelength.
    empty_cache = {}
    for name, case in CASES.items():
        factory = material_factory(case)
        for resolution in (20, 40):
            for leg in ('s', 'p'):
                start = time.perf_counter()
                cache_key = (resolution, leg)
                if cache_key not in empty_cache:
                    empty_cache[cache_key] = _run(case, leg, resolution, empty=True)
                empty = empty_cache[cache_key]
                struct = _run(case, leg, resolution, empty=False, material_at=factory)
                data, leak, purity = _amplitudes(case, leg, struct, empty)
                if leak > 1e-9 or purity > .01:
                    raise SystemExit(f'Incident control failed: {name} {leg} {leak=} {purity=}')
                total = float(data['R'].sum() + data['T'].sum())
                if not .4 < total < 1.1:
                    raise SystemExit(f'Unexpected energy balance: {name} {leg} {total=}')
                prefix = f'{name}__{resolution}__{leg}__'
                out.update({prefix+k: v for k, v in data.items()})
                out.update({prefix+'leak': np.array(leak), prefix+'purity': np.array(purity),
                            prefix+'seconds': np.array(time.perf_counter()-start)})
                np.savez(args.output, **out)
                print('MATERIAL_RESULT', name, resolution, leg, total, leak, purity, flush=True)
    out['complete'] = np.array(True)
    np.savez(args.output, **out)


if __name__ == '__main__':
    main()
