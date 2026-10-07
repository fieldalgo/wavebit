"""Disclose fixed independent C9 algebra/step rounding before accepting budgets.

Draft only. Run exclusively under the owner's numerical queue after application.
All production/test/helper snapshots are written before importing the test module.
The tests' near() function is replaced only in memory with a recorder, leaving all
structural, positivity, type and finite assertions active. No tolerance is chosen.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'fdtd_constitutive_absorber_'
TEST = ROOT / 'tests/test_fd_constitutive_absorber.py'
SCHEDULE = (
    ('test_damped_flux_and_field_match_independent_dense_operator',
     ('backend','axis','complex_'), (('numpy','jax','torch'), (0,1,2), (False,True))),
    ('test_staggered_storage_obeys_independent_dissipation_identity',
     ('backend','complex_','damped','work'), (('numpy','jax','torch'), (False,True), (False,True), (False,True))),
    ('test_observer_uses_whole_absorber_and_both_magnetic_clocks',
     ('backend','complex_','large_rate'), (('numpy','jax','torch'), (False,True), (False,True))),
    ('test_lift_and_cfl_read_physical_not_damped_diagonal',
     ('backend','axis'), (('numpy','jax','torch'), (0,1,2))),
    ('test_projected_yee_step_and_sampled_energy_match_independent_flux_oracle',
     ('backend','wall','axis'), (('numpy','jax','torch'), ('PEC','PMC'), (0,1,2))),
)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def paths():
    return sorted(set(ROOT.joinpath('wavebit').rglob('*.py'))
                  | {TEST, ROOT/'tests/fd_helpers.py'}
                  | set((ROOT/'studies').glob(PREFIX+'*.py')))


def identity():
    return {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in paths()}


def write(path, value):
    path.write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=int,required=True)
    parser.add_argument('--expected-base',required=True)
    parser.add_argument('--expected-branch',required=True)
    args = parser.parse_args()
    assert args.attempt > 0
    assert len(args.expected_base) == 40 and all(c in '0123456789abcdef' for c in args.expected_base)
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch = subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    assert (head,branch) == (args.expected_base,args.expected_branch)
    directory = ROOT/'studies/results'/f'{PREFIX}algebra_attempt{args.attempt}'
    directory.mkdir(parents=True,exist_ok=False)
    started = utc()
    before = identity()
    schedule = [{'function':name,'arguments':dict(zip(names,values))}
                for name,names,choices in SCHEDULE for values in itertools.product(*choices)]
    with zipfile.ZipFile(directory/'sources.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in before:
            archive.writestr(name,(ROOT/name).read_bytes())
    snapshot = {'started_utc':started,'base':head,'branch':branch,'before':before,
                'schedule':schedule,'sources_archive':'sources.zip',
                'sources_sha256':digest((directory/'sources.zip').read_bytes()),
                'qualification':'Written before importing tests/project or executing any instrument.'}
    write(directory/'before.json',snapshot)
    (directory/'test.py.raw.gz').write_bytes(gzip.compress(TEST.read_bytes(),mtime=0))
    arrays,metrics,rows = {},[],[]
    result = {'complete':False,'passed':False,'started_utc':started,'base':head,'branch':branch,
              'attempt':args.attempt,'guard_before':before,'schedule':schedule,'rows':rows,
              'metrics':metrics,'allowances_accepted':False,
              'qualification':'Fixed instruments; no fit or production mutation; saved actual/expected arrays.'}
    active = None
    status = 1
    clock = time.perf_counter()
    try:
        import numpy as np
        sys.path.insert(0,str(ROOT))
        sys.path.insert(0,str(ROOT/'tests'))
        spec = importlib.util.spec_from_file_location('c9_algebra_instruments',TEST)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.ACCEPTED_BUDGETS is None, 'calibration must precede budget acceptance'
        original_near = module.near
        def record(actual, expected, key='algebra'):
            actual,expected = np.asarray(actual),np.asarray(expected)
            number = len(metrics)
            stem = f'check{number:05d}'
            arrays[stem+'_actual'] = actual.copy()
            arrays[stem+'_expected'] = expected.copy()
            same_shape = actual.shape == expected.shape
            finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
            absolute = (float(np.max(np.abs(actual-expected),initial=0.0))
                        if same_shape and finite else None)
            scale = (max(1.0,float(np.max(np.abs(expected),initial=0.0)))
                     if finite else None)
            metrics.append({'check_index':number,'row_index':active,'kind':key,
                            'absolute':absolute,'scale':scale,
                            'normalization': ('absolute residual; zero reference, scale=1'
                                if finite and not np.any(expected)
                                else 'absolute error / max(1, reference component magnitude)'),
                            'scaled':absolute/scale if absolute is not None else None,
                            'finite':finite,'same_shape':same_shape,
                            'actual':stem+'_actual','expected':stem+'_expected',
                            'shape_actual':list(actual.shape),'shape_expected':list(expected.shape),
                            'dtype_actual':str(actual.dtype),'dtype_expected':str(expected.dtype)})
            assert same_shape and finite, 'raw failing arrays retained in calibration archive'
        module.near = record
        try:
            for index,row in enumerate(schedule):
                active = index
                t0 = time.perf_counter()
                entry = {'index':index,**row,'started_utc':utc(),'passed':False,
                         'first_check':len(metrics)}
                rows.append(entry)
                getattr(module,row['function'])(**row['arguments'])
                entry.update(passed=True,ended_utc=utc(),seconds=time.perf_counter()-t0,
                             checks=len(metrics)-entry['first_check'])
                print(json.dumps({'row':index,'passed':True,'checks':entry['checks']}),flush=True)
        finally:
            module.near = original_near
        result['complete'] = len(rows)==len(schedule) and all(row['passed'] for row in rows)
        result['max_scaled'] = {key:max(m['scaled'] for m in metrics if m['kind']==key)
                                for key in sorted({m['kind'] for m in metrics})}
        result['passed'] = result['complete']
        status = 0
    except Exception:
        result['failure'] = traceback.format_exc()
        print(result['failure'],file=sys.stderr,flush=True)
    finally:
        result['guard_after'] = identity()
        result['source_unchanged'] = result['guard_after']==before
        if not result['source_unchanged']:
            status = 1
            result['passed'] = False
        if arrays:
            np.savez_compressed(directory/'comparisons.npz',**arrays)
            result['arrays'] = {'file':'comparisons.npz','sha256':digest((directory/'comparisons.npz').read_bytes()),
                                'members':len(arrays)}
        result.update(ended_utc=utc(),seconds=time.perf_counter()-clock,exit_code=status)
        write(directory/'result.json',result)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
