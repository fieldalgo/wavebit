"""Rerun every probe mode at the final fd source; keep the earlier JSON under before_final/ and
compare the two bit for bit (timings excluded)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0] if False else Path.cwd()
RESULTS = ROOT / 'studies/results'
CONTROLS = RESULTS / 'fdtd_traced_material_controls'
BEFORE = CONTROLS / 'before_final'
BEFORE.mkdir(parents=True, exist_ok=True)
MODES = [('equal', [], ['equal']), ('measure', ['cylinder'], ['measure_cylinder_-0.2375']),
         ('measure', ['box'], ['measure_box_-0.2375']), ('multi', [], ['multi']),
         ('sphere', [], ['sphere']), ('guards', [], ['guards']), ('extras', [], ['extras']),
         ('settled', [], ['settled'])]


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in ('seconds', 'numpy_seconds')}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


report = {}
for mode, args, names in MODES:
    for name in names:
        src = RESULTS / f'fdtd_traced_material_{name}.json'
        shutil.copy2(src, BEFORE / src.name)
    log = CONTROLS / ('probe_' + '_'.join([mode] + args) + '_final.log')
    with log.open('w', encoding='utf-8') as stream:
        code = subprocess.run([sys.executable, 'studies/fdtd_traced_material_probe.py', mode, *args],
                              cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
    for name in names:
        before = strip(json.loads((BEFORE / f'fdtd_traced_material_{name}.json').read_text()))
        after = strip(json.loads((RESULTS / f'fdtd_traced_material_{name}.json').read_text()))
        report[name] = dict(exit=code, identical=before == after)
        print(name, report[name], flush=True)
(CONTROLS / 'rerun_at_final_source.json').write_text(json.dumps(report, indent=2) + '\n',
                                                      encoding='utf-8')
assert all(r['exit'] == 0 and r['identical'] for r in report.values()), report
print('every probe mode reproduced bit for bit', flush=True)
