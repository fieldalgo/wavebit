"""Validate study/docs; explicitly reuse recent unchanged-source regression evidence.

No production or test source changes are part of this experiment. The recent
fine campaign already ran the full/slow tiers and four read-only freezers on
this exact package. Do not describe those checks as newly run for this study.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import json
import re
import subprocess
import sys
import time
import uuid

import studies.cross_solver.metasurface_tall.fdtd_metasurface_tall as study
from studies.fdtd.polygon_tie.fdtd_polygon_tie_validation import source_hash


def main():
    reg = study.verify()
    previous = study.RESULTS/'fdtd_metasurface_fine_validation.json'
    prior = json.loads(previous.read_text())
    previous_reg = json.loads((study.RESULTS/'fdtd_metasurface_fine_registration.json').read_text())
    assert prior['source'] == source_hash()
    assert previous_reg['identity']['package'] == reg['identity']['package']
    assert all(c['exit']==0 for c in prior['checks'])
    diff = subprocess.check_output(['git','diff','33aab0f9','--','wavebit','tests'],cwd=study.ROOT,text=True)
    untracked = subprocess.check_output(['git','ls-files','--others','--exclude-standard','wavebit','tests'],
                                       cwd=study.ROOT,text=True)
    assert not diff and not untracked, 'Production or tests changed since prior validation'
    folder = study.RESULTS/'fdtd_metasurface_tall_controls'
    folder.mkdir(exist_ok=True)
    manifest = study.RESULTS/'fdtd_metasurface_tall_validation.json'
    assert not manifest.exists(), 'Preserve any prior validation attempt before rerunning'
    record = dict(fd_source=source_hash(),package=reg['identity']['package'],
                  production_and_tests_unchanged_since='33aab0f9',checks=[],
                  prior_full_slow_freezers=dict(file=previous.name,sha256=study.sha(previous),
                                               rerun=False,checks=prior['checks']))
    temp = study.ROOT/'.metasurface_tall_work'/('docs-'+uuid.uuid4().hex)
    temp.parent.mkdir(parents=True,exist_ok=True)
    commands = [
        ('audit',['studies/cross_solver/metasurface_tall/fdtd_metasurface_tall_audit.py']),
        ('settled_audit',['studies/cross_solver/metasurface_tall/fdtd_metasurface_tall_settled_audit.py']),
        ('docs_collection',['-m','pytest','-q','-p','no:cacheprovider','--collect-only','tests/api/test_docs_contract.py']),
        ('docs',['-m','pytest','-q','-p','no:cacheprovider','--basetemp',str(temp),'tests/api/test_docs_contract.py'])]
    collected = None
    for name,args in commands:
        started = time.perf_counter()
        with (folder/(name+'.log')).open('x',encoding='utf-8') as stream:
            run = subprocess.run([sys.executable,*args],cwd=study.ROOT,stdout=stream,stderr=subprocess.STDOUT)
        text = (folder/(name+'.log')).read_text(encoding='utf-8')
        check = dict(name=name,exit=run.returncode,seconds=time.perf_counter()-started)
        record['checks'].append(check)
        manifest.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        assert run.returncode==0 and not any(s in text for s in ('INTERNALERROR','crashed','node down','ERROR collecting'))
        if name=='docs_collection':
            collected = sum(s.startswith('tests/') and '::' in s for s in text.splitlines())
            assert collected>0
        if name=='docs':
            passed = int(re.findall(r'(\d+) passed',text)[-1])
            assert passed==collected
            check.update(collected=collected,passed=passed)
        assert source_hash()==record['fd_source']
        manifest.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        print(name,check,flush=True)
    print('Study and docs validation passed; prior full/slow/freezer evidence reused explicitly.',flush=True)


if __name__ == '__main__':
    main()
