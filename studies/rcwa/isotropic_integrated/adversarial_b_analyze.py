import json, pathlib, numpy as np
d = json.loads(pathlib.Path(__file__).with_name('adversarial_b.json').read_text())
for case, v in d.items():
    refs = {'S13': v['scalar|13'], 'I13': v['integrated|13'], 'joint': (v['scalar|13'] + v['integrated|13'])/2}
    print(case, 'gap13', v['scalar|13'] - v['integrated|13'], 'S12-13', v['scalar|12']-v['scalar|13'],
          'I12-13', v['integrated|12']-v['integrated|13'])
    for rn, ref in refs.items():
        es = [abs(v[f'scalar|{M}'] - ref) for M in range(3, 11)]
        ei = [abs(v[f'integrated|{M}'] - ref) for M in range(3, 11)]
        wins = sum(a < b for a, b in zip(es, ei))
        print(f'  ref {rn}: geomean M3..10 S {np.exp(np.mean(np.log(es))):.2e} I {np.exp(np.mean(np.log(ei))):.2e}  S wins {wins}/8')
