"""Public split at exact node-plane ties, against the registered RCWA consistency rule.

This driver uses the existing study's scene builder and archived RCWA references,
but no lowering patch. Failed rows are retained. It never changes a golden,
reference, or acceptance threshold. Run one process at a time.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import wavebit.fdtd._legacy as fd
from studies.fdtd.subpixel_junction import fdtd_subpixel_junction as h


def main():
    log = Path('studies/cross_solver/junction/records/fdtd_junction_production_ties.log')
    source = hashlib.sha256()
    for path in sorted(Path(fd.__file__).parent.rglob('*.py')):
        source.update(path.relative_to(Path(fd.__file__).parent).as_posix().encode())
        source.update(path.read_bytes().replace(b'\r\n', b'\n'))
    digest = source.hexdigest()
    h.PROV = h.provenance()
    references = h.load(h.RCWA_LOG)
    names = sys.argv[1:] or ["C'", "D'"]
    def write(row):
        row.update(source=digest, date=time.time())
        with log.open('a', encoding='utf-8') as output:
            output.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)
    for name in names:
        reference = h.rcwa_ladder(references, name)
        assert not reference['needs'], reference['needs']
        results = {}
        for pol in ('s', 'p'):
            for n in (40, 80):
                started = time.perf_counter()
                scene, _ = h.build(name, n, 0., pol=pol)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    result = fd.run(scene, backend=h.engine(name, n))
                amplitudes = h.amplitudes(result, pol)
                values = np.asarray(amplitudes['r']+amplitudes['t'])
                decay = {k: None if v is None else float(v)
                         for k,v in result.diagnostics['decay'].items()}
                finite = bool(np.isfinite(values).all()
                              and np.isfinite(np.asarray(result.final.E)).all()
                              and np.isfinite(np.asarray(result.final.H)).all())
                passed = (finite and len(decay) == 2
                          and all(v is not None and v < 1e-5 for v in decay.values())
                          and not any('GrowingField' in w.category.__name__ for w in caught))
                if result.plan.constitutive:
                    passed = passed and result.diagnostics.get('stable') is True
                write(dict(kind='solve', scene=name, zeta=0., n=n, pol=pol,
                           passed=bool(passed), decay=decay,
                           seconds=time.perf_counter()-started, **amplitudes))
                assert passed, 'Tie solve invalid; retained the row and stopped.'
                results[n, pol] = amplitudes
        comparisons = []
        for pol in ('s', 'p'):
            for k, wl in enumerate(h.WAVELENGTHS):
                for monitor in ('r', 't'):
                    # Read each polarisation's actual archived complex value;
                    # do not infer its reflection-phase convention by symmetry.
                    available = [row for row in references.values()
                                 if row.get('kind') == 'rcwa' and row.get('scene') == name
                                 and row.get('variant') == 'flush' and row.get('wl') == wl
                                 and row.get('theta') == 0.]
                    ref = h._rc(max(available, key=lambda row: row['N']), pol, monitor)
                    fine = complex(*results[80, pol][monitor][k])
                    coarse = complex(*results[40, pol][monitor][k])
                    passed, miss, margin = h.consistent(ref, fine, coarse, 80, 40, reference['u'])
                    comparisons.append(dict(wavelength=wl, monitor=monitor, pol=pol,
                                            passed=passed, miss=miss, margin=margin))
        passed = all(row['passed'] for row in comparisons)
        write(dict(kind='consistency', scene=name, passed=passed,
                   reference_uncertainty=reference['u'], comparisons=comparisons))
        assert passed, 'Exact-tie consistency failed; do not ship the default.'


if __name__ == '__main__':
    main()
