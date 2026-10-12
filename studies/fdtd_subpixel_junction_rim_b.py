"""Post-hoc diagnostic, not part of the rule: the one rim escalation (scene B, placement 0.25,
s) was an order fit over three rungs (24/48/96) with RMS 0.20 and a difference 1.09 times the
materiality line. Rungs 32 and 64 keep B's placement exact (P n and H n integers), so arms 2
and 1b there resolve the order of max|T_1b - T_2| over five rungs. Rows go to their own log;
the registered rows at 24/48/96 are read from the study's log, not solved again."""
import json
import math


import fdtd_subpixel_junction as S

S.PROV = S.provenance()
DIAG = S.RESULTS / 'fdtd_subpixel_junction_rim_b.log'
RUNGS = (24, 32, 48, 64, 96)
specs = [S.spec('B', n, 0.25, arm) for n in (32, 64) for arm in ('2', '1b')]
S.run_specs(specs, log=DIAG)

rows = dict(S.load(S.LOG))
rows.update(S.load(DIAG))
d, s2 = [], []
for n in RUNGS:
    two = S.cells(rows, 'B', '2', n, (0.25,), ('s',))
    one = S.cells(rows, 'B', '1b', n, (0.25,), ('s',))
    d.append(S.dmax(one, two))
    s2.append(two)
q, rms = S.fit(list(RUNGS), d)
local = [math.log(d[i] / d[i + 1]) / math.log(RUNGS[i + 1] / RUNGS[i]) for i in range(len(d) - 1)]
sigma = S.dmax(s2[-1], s2[-3])            # |T2(96) - T2(48)|, the rule's sigma pair
out = dict(rungs=RUNGS, d=d, order=q, rms=rms, local=local, sigma_96_48=sigma,
           material_ratio=d[-1] / (0.05 * sigma))
print(json.dumps(out, indent=1))
with open(S.RESULTS / 'fdtd_subpixel_junction_rim_b_summary.log', 'w', encoding='utf-8') as f:
    f.write(json.dumps(out) + '\n')
