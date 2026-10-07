"""Unexecuted six-pair development calibration; no empirical ceiling selected here."""
import argparse
import hashlib
import itertools
import json
import traceback
import zipfile

from fdtd_constitutive_absorber_support import (
    ROOT, RESULTS, PREFIX, guard, identity, now, relative, require_base, run_worker, sha, versions, write)

FAMILIES = ('cylinder', 'rotated-box')
BACKENDS = ('numpy', 'jax', 'torch')


def config(family, backend, *, n=20, placement=.3, polarization='u', length='short',
           time=16., contrast=True, homogeneous=False):
    return dict(family=family, backend=backend, n=n, placement=placement,
                polarization=polarization, length=length, time=time,
                contrast=contrast, homogeneous=homogeneous)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def freeze(directory):
    before = identity()
    destination = directory/'sources.zip'
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path, expected in before.items():
            data = (ROOT/path).read_bytes()
            require(hashlib.sha256(data).hexdigest() == expected, 'Source changed while freezing.')
            archive.writestr(path, data)
    with zipfile.ZipFile(destination) as archive:
        require(set(archive.namelist()) == set(before), 'Source archive inventory differs.')
        require(all(hashlib.sha256(archive.read(p)).hexdigest() == h for p, h in before.items()),
                'Source archive bytes differ.')
    return before, {'path': relative(destination), 'sha256': sha(destination)}


def read_worker(row, expected_config, expected_identity):
    """Reopen only saved outputs, checking every byte and named array."""
    import numpy as np
    require(row['exit_code'] == 0, 'Field worker failed; preserve it and stop.')
    log, result_path = ROOT/row['log'], ROOT/row['result']
    require(sha(log) == row['log_sha256'] and sha(result_path) == row['result_sha256'],
            'Worker raw log/result hash differs.')
    require(sha(ROOT/row['config']) == row['config_sha256'], 'Worker configuration hash differs.')
    require(json.loads((ROOT/row['config']).read_text()) == expected_config, 'Written worker config differs.')
    result = json.loads(result_path.read_text())
    require(result['config'] == expected_config and result['complete']
            and all(result['checks'].values()), 'Worker did not complete the declared fixture.')
    require(result['source_before'] == result['source_after'] == expected_identity,
            'Worker production/test/helper identity differs.')
    raw = ROOT/result['raw']
    require(sha(raw) == result['raw_sha256'], 'Raw NPZ hash differs.')
    with np.load(raw, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    require(set(arrays) == set(result['array_sha256']), 'Raw array inventory differs.')
    require(all(hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                == result['array_sha256'][key] for key, value in arrays.items()), 'Raw array payload differs.')
    require(all(np.isfinite(value).all() for value in arrays.values()), 'Nonfinite saved array.')
    return result, arrays


def run(attempt):
    require(attempt > 0, 'Positive attempt number required.')
    base = require_base()
    folder = RESULTS/f'{PREFIX}_calibration_attempt{attempt}'
    folder.mkdir(parents=True, exist_ok=False)
    before, archive = freeze(folder)
    frozen_guard = guard()
    require(frozen_guard['source'] == before, 'Source changed during initial guard capture.')
    schedule = []
    for family, backend in itertools.product(FAMILIES, BACKENDS):
        schedule.extend([config(family, backend), config(family, backend, length='reference'),
                         config(family, backend, contrast=False),
                         config(family, backend, contrast=False, homogeneous=True)])
    manifest = {'base': base, 'attempt': attempt, 'started_utc': now(), 'source_before': before,
                'source_archive': archive, 'guard_before': frozen_guard, 'versions': versions(), 'schedule': schedule,
                'workers': [], 'complete': False, 'ceilings': None,
                'qualification': 'Six disclosed family/CPU rows each contain positive short/reference and zero-shaped/legacy pairs (24 solves). No termination or continuum accuracy ceiling accepted.'}
    write(folder/'before.json', manifest)
    status = 1
    try:
        from fdtd_constitutive_absorber_probe import preflight, compare_pair, difference, zero_comparison
        manifest['analytic_preflight'] = preflight()
        write(folder/'analytic_preflight.json', manifest['analytic_preflight'])
        for index, item in enumerate(schedule):
            require(guard() == frozen_guard, 'Source/golden/prior evidence changed before calibration worker.')
            row = run_worker(item, folder/f'worker{index:02d}')
            manifest['workers'].append(row)
            write(folder/f'ledger{index:02d}.json', manifest['workers'])
            read_worker(row, item, before)
        loaded = [read_worker(row, cfg, before) for row, cfg in zip(manifest['workers'], schedule)]
        manifest['pairs'] = []
        for index in range(0, len(loaded), 4):
            (short, a), (reference, b) = loaded[index:index+2]
            report = compare_pair(short, reference, a, b)
            require(all(report['mapping_checks'].values()), 'Calibration shared-grid mapping failed.')
            shaped, legacy = loaded[index+2:index+4]
            zero = zero_comparison(shaped[0], legacy[0], shaped[1], legacy[1])
            require(all(zero['checks'].values()), 'Zero-control structural validation failed.')
            manifest['pairs'].append({'family': short['config']['family'], 'backend': short['config']['backend'],
                                      'worker_indices': list(range(index,index+4)), 'comparison': report, 'zero_comparison': zero})
        comparisons = []
        for family in FAMILIES:
            for role, length, contrast, homogeneous in (
                    ('short','short',True,False), ('reference','reference',True,False),
                    ('zero-shaped','short',False,False), ('zero-legacy','short',False,True)):
                selected = {r['config']['backend']: a for r, a in loaded
                            if r['config']['family'] == family and r['config']['length'] == length
                            and r['config']['contrast'] == contrast and r['config']['homogeneous'] == homogeneous}
                for backend in ('jax', 'torch'):
                    require(set(selected[backend]) == set(selected['numpy']), 'Backend array roster differs.')
                    comparisons.append({'family': family, 'role': role, 'length': length, 'backend': backend,
                        'arrays': {k: difference(selected[backend][k], selected['numpy'][k])
                                   for k in selected['numpy']}})
        manifest['backend_comparisons'] = comparisons
        manifest['max_backend_scaled'] = max(v['roundoff_scaled'] for row in comparisons for v in row['arrays'].values())
        manifest['max_shared_alignment_scaled'] = max(v['roundoff_scaled'] for pair in manifest['pairs']
            for key in ('coordinate_errors', 'shared_lowering', 'plane_coordinates')
            for v in pair['comparison'][key].values())
        manifest['max_zero_scaled'] = max(pair['zero_comparison']['max_roundoff_scaled'] for pair in manifest['pairs'])
        manifest['complete'] = len(loaded) == 24 and guard() == frozen_guard
        status = 0 if manifest['complete'] else 1
    except BaseException:
        manifest['failure'] = traceback.format_exc()
        raise
    finally:
        manifest.update(ended_utc=now(), source_after=identity(), guard_after=guard(), exit_code=status)
        write(RESULTS/f'{PREFIX}_calibration_attempt{attempt}.json', manifest)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.attempt))
