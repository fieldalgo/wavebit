"""Actual production assertion witnesses; root owns the exclusive execution window.

Every attempt is retained. Original/mutant source archives are deterministic gzip
of raw bytes, preventing documentation scanners from treating archived code as
prose. Each temporary write is restored in finally before any acceptance decision.
No numerical module is imported by this controller.
"""
import argparse
import ast
import gzip
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import fdtd_decay_stop_support as support

ROOT, RESULTS, PREFIX, SCRATCH = support.ROOT, support.RESULTS, support.PREFIX, support.SCRATCH
TEST = support.NEW_TEST
OLD = (
    'tests/test_fd_monitors_doors.py::test_mode_monitors_and_decay_stopping_are_stage_7',
    'tests/test_fd_monitors_doors.py::test_stopping_on_decay_is_refused_naming_stage_7',
    'tests/test_fd_contract.py::test_later_stages_are_refused_by_name',
    'tests/test_fd_scene.py::test_a_later_stage_keyword_on_scene_is_refused_naming_the_stage',
)
TARGETS = (TEST, *OLD)
FILES = tuple(dict.fromkeys(name.split('::')[0] for name in TARGETS))
DECAY = 'wavebit/fd/_decay.py'


def mutations():
    return [
        ('closed-upper-threshold-admitted', 'wavebit/fd/monitors.py',
         '        if not (math.isfinite(threshold) and 0 < threshold < 1):\n',
         '        if not (math.isfinite(threshold) and 0 < threshold <= 1):\n',
         TEST+'::test_decay_requires_explicit_static_finite_amplitude_ratio_and_window[threshold-one]'),
        ('complex-modulus-overflow', 'wavebit/fd/_runtime.py',
         '    value = xp.max(xp.abs(xp.real(array)))\n',
         '    value = xp.max(xp.abs(array))\n',
         TEST+'::test_component_amplitude_keeps_subnormals_and_extreme_complex_values[large-complex-eager-numpy]'),
        ('jax-window-subnormal-maximum-flushed', 'wavebit/fd/_runtime.py',
         '        keys = self.xp.maximum(self._amplitude_keys(a), self._amplitude_keys(b))\n',
         '        keys = self.jax.lax.bitcast_convert_type(self.xp.maximum(a, b), self.xp.int64)\n',
         TEST+'::test_component_amplitude_keeps_subnormals_and_extreme_complex_values[subnormal-actual-window-jax]'),
        ('rounded-ratio-false-success', DECAY,
         '                    and all(Fraction(window_metrics[k]) <= Fraction(self.plan.decay.threshold)\n'
         '                            * Fraction(self.peaks[k]) for k in self.groups))\n',
         '                    and all(self.peaks[k] == 0 or window_metrics[k] / self.peaks[k]\n'
         '                            <= self.plan.decay.threshold for k in self.groups))\n',
         TEST+'::test_complete_window_tracker_matches_independent_scalar_history[rounded-ratio-boundary]'),
        ('weak-group-global-normalization', DECAY,
         '                            * Fraction(self.peaks[k]) for k in self.groups))\n',
         '                            * Fraction(max(self.peaks.values())) for k in self.groups))\n',
         TEST+'::test_complete_window_tracker_matches_independent_scalar_history[weak-memory]'),
        ('hybrid-aux-group-omitted', DECAY,
         '            for key in keys:\n',
         "            for key in tuple(k for k in keys if k != 'aux'):\n",
         TEST+'::test_each_evolving_array_is_a_separate_group[hybrid-film]'),
        ('cpml-group-omitted', DECAY,
         "        out['psi:' + name] = state['psi'][name]\n",
         "        if False:\n            out['psi:' + name] = state['psi'][name]\n",
         TEST+'::test_each_evolving_array_is_a_separate_group[cpml]'),
        ('only-last-step-of-window-read', DECAY,
         '                        return current, rt.positive_maximum(maxima, values)\n',
         '                        return current, values\n',
         TEST+'::test_uniform_conductor_stops_at_first_independently_predicted_window[real-numpy]'),
        ('record-cap-padding-retained', DECAY,
         "        out['obs'][observer.name] = (own(acc[:steps]) if type(observer) is PointRecord\n",
         "        out['obs'][observer.name] = (own(acc) if type(observer) is PointRecord\n",
         TEST+'::test_prefix_outputs_use_actual_count_and_complete_energy_clock[3-numpy]'),
        ('energy-prefix-one-clock-late', DECAY,
         "    out['energy'] = own(state['energy'][:max(steps - 1, 0)])\n",
         "    out['energy'] = own(state['energy'][1:steps])\n",
         TEST+'::test_prefix_outputs_use_actual_count_and_complete_energy_clock[3-numpy]'),
        ('decay-dft-endpoint-one-step-late', 'wavebit/fd/_assemble.py',
         '    dt, N = plan.dt, plan.steps\n',
         '    dt, N = plan.dt, plan.steps + (plan.decay is not None)\n',
         TEST+'::test_prefix_outputs_use_actual_count_and_complete_energy_clock[3-numpy]'),
        ('last-gaussian-source-ignored', DECAY,
         '            latest = max(latest, time)\n',
         '            latest = latest if latest else time\n',
         TEST+'::test_source_eligibility_uses_every_gaussian_tail_and_conservative_clock[Ex-0.1]'),
        ('uncovered-runtime-admitted', DECAY,
         '    if type(rt) not in (NumpyRuntime, JaxRuntime, TorchRuntime):\n',
         '    if False and type(rt) not in (NumpyRuntime, JaxRuntime, TorchRuntime):\n',
         TEST+'::test_decay_refuses_uncovered_sources_memory_backends_and_transforms[unknown-runtime]'),
        ('callable-dual-primal-conversion-admitted', 'wavebit/fd/_lower/fields.py',
         '    if concrete_only:\n        from .._decay import reject_carried\n        reject_carried(entry)\n',
         '    if False and concrete_only:\n        from .._decay import reject_carried\n        reject_carried(entry)\n',
         TEST+'::test_decay_refuses_uncovered_sources_memory_backends_and_transforms[torch-callable-dual]'),
        ('actual-prefix-mislabeled-as-cap', 'wavebit/fd/_check.py',
         '        stopping = (run_steps(scene, p.dt), p.decay.threshold, p.decay.window,\n',
         '        stopping = (p.steps, p.decay.threshold, p.decay.window,\n',
         TEST+'::test_prepared_plan_and_report_bind_cap_criterion_and_effective_prefix[steps]'),
        ('fixed-energy-schedule-changed', 'wavebit/fd/_lower/plan.py',
         'stride=1 if self.decay is not None else ENERGY_STRIDE',
         'stride=1',
         TEST+'::test_fixed_runs_keep_their_energy_cadence_and_amplitude_derivative[numpy]'),
        ('implicit-window-default', 'wavebit/fd/monitors.py',
         '    window: float\n', '    window: float = 1.0\n',
         OLD[0]),
        ('bare-run-ratio-admitted', 'wavebit/fd/scene.py',
         '        if self.decay is not None and type(self.decay) is not Decay:\n',
         '        if False and self.decay is not None and type(self.decay) is not Decay:\n',
         OLD[1]+'[decay]'),
        ('scene-contract-keeps-stale-stage-door', 'wavebit/fd/scene.py',
         "    'decay': 'use Run(decay=Decay(threshold=..., window=...)) with an explicit hard cap',\n",
         "    'decay': 'stage 7 placeholder',\n",
         OLD[2]),
        ('scene-replace-keeps-stale-stage-door', 'wavebit/fd/scene.py',
         "    'decay': 'use Run(decay=Decay(threshold=..., window=...)) with an explicit hard cap',\n",
         "    'decay': 'stage 7 placeholder',\n",
         OLD[3]+'[decay]'),
    ]


def archived(raw, destination):
    data = gzip.compress(raw, compresslevel=9, mtime=0)
    support.require(gzip.decompress(data) == raw, 'Source compression changed raw bytes.')
    with destination.open('xb') as stream:
        stream.write(data)
    return support.relative(destination), support.digest(raw), support.digest(data)


def run_pytest(targets, log, tag, *, collect=False):
    env = dict(os.environ)
    env.update(PYTHONUTF8='1', PYTHONPATH=os.pathsep.join((str(ROOT / 'src'), str(ROOT))), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPYCACHEPREFIX=str(SCRATCH/('mutation-pycache-'+tag+'-'+uuid.uuid4().hex)))
    temporary = SCRATCH/('mutation-pytest-'+tag+'-'+uuid.uuid4().hex)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-B', '-m', 'pytest', '-q' if collect else '-vv',
               '-p', 'no:cacheprovider', '--tb=short', '--basetemp', str(temporary), *targets]
    if collect:
        command += ['--collect-only']
    begin, started = time.perf_counter(), support.now()
    with log.open('xb') as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return {'command': command, 'exit_code': process.returncode,
            'started_utc': started, 'ended_utc': support.now(), 'wall_s': time.perf_counter()-begin,
            'raw_log': support.relative(log), 'raw_log_sha256': support.sha(log),
            'environment': {key: env.get(key) for key in ('PYTHONUTF8', 'PYTHONPATH',
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONPYCACHEPREFIX', 'XLA_FLAGS')}}


def function_set():
    tree = ast.parse((ROOT/TEST).read_text(encoding='utf-8'))
    return {TEST+'::'+node.name for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')} | set(OLD)


def assertion_failure(text):
    return bool(re.search(r'AssertionError|Failed: DID NOT RAISE|^E\s+assert\b',
                          text, flags=re.MULTILINE))


def source_replacement(raw, before, after):
    before_raw = before.replace('\n', '\r\n').encode() if b'\r\n' in raw else before.encode()
    after_raw = after.replace('\n', '\r\n').encode() if b'\r\n' in raw else after.encode()
    support.require(raw.count(before_raw) == 1, 'Expected exactly one raw mutation anchor.')
    return raw.replace(before_raw, after_raw)


def run(attempt, baseline_log):
    support.start()
    targeted = support.load(RESULTS/(PREFIX+'targeted.json'))
    support.require(targeted['complete'], 'A completed targeted baseline is required.')
    support.verify_guard(targeted['guard_before'])
    baseline = support.inside(str(baseline_log))
    matches = [row for row in targeted['checks'] if support.inside(row['log']) == baseline]
    support.require(len(matches) == 1 and matches[0]['exit'] == 0
                    and matches[0]['sha256'] == support.sha(baseline),
                    'The baseline must be the exact passed targeted command log.')
    text = baseline.read_text(encoding='utf-8')
    support.check_log(text)
    summary = support.summary(text)
    support.require(summary['passed'] > 0 and sum(summary.values()) == summary['passed'],
                    'The baseline must have only actual passed test statuses.')
    directory = support.CONTROL/f'mutations_attempt{attempt}'
    directory.mkdir(exist_ok=False)
    path = RESULTS/f'{PREFIX}mutations_attempt{attempt}.json'
    support.require(not path.exists(), 'Preserve the earlier mutation attempt.')
    guard_before, source_before = support.guard(), support.identity()
    roster, wanted = mutations(), function_set()
    support.require(len({row[0] for row in roster}) == len(roster), 'Repeated mechanism name.')
    support.require({row[4].split('[')[0] for row in roster} == wanted,
                    'Every new or changed function needs an actual assertion witness.')
    source_snapshots = {}
    for name in (*FILES, support.relative(Path(__file__))):
        file, raw_hash, archive_hash = archived((ROOT/name).read_bytes(),
            directory/(name.replace('/', '__')+'.raw.gz'))
        source_snapshots[name] = {'file': file, 'sha256': raw_hash, 'archive_sha256': archive_hash}
    record = {'base': support.BASE, 'attempt': attempt, 'started_utc': support.now(),
              'source_before': source_before, 'guard_before': guard_before,
              'harness_sha256': support.sha(Path(__file__)),
              'test_sha256': support.sha(ROOT/TEST),
              'test_file_sha256': {name: support.sha(ROOT/name) for name in FILES},
              'source_snapshots': source_snapshots,
              'baseline_log': support.relative(baseline), 'baseline_log_sha256': support.sha(baseline),
              'targeted_manifest_sha256': support.sha(RESULTS/(PREFIX+'targeted.json')),
              'expected_functions': sorted(wanted), 'expected_mechanisms': [row[0] for row in roster],
              'witnesses': [], 'restored': True, 'passed': False,
              'all_function_assertion_witnesses': False}
    support.write(path, record, exclusive=True)
    # All anchors are checked before the first temporary edit or pytest invocation.
    for name, relative, before, after, node in roster:
        support.require(node.split('[')[0] in wanted, 'Unexpected mutation target: '+name)
        original = (ROOT/relative).read_bytes()
        changed = source_replacement(original, before, after)
        ast.parse(changed.decode('utf-8'), filename=relative)
    try:
        collect_log = directory/'selected_collection.log'
        collected = run_pytest(TARGETS, collect_log, f'{attempt}-collection', collect=True)
        collection_text = collect_log.read_text(encoding='utf-8')
        support.check_log(collection_text)
        nodes = support.nodes(collection_text)
        support.require(collected['exit_code'] == 0, 'Mutation target collection failed.')
        support.require({node.split('[')[0] for node in nodes} == wanted,
                        'Collection does not cover exactly the intended functions.')
        support.require(all(row[4] in nodes for row in roster), 'Mutation node id is absent.')
        record['collection'] = dict(collected, collected=len(nodes), nodes=nodes)
        support.write(path, record)
        for name, relative, before, after, node in roster:
            support.verify_guard(guard_before)
            source = ROOT/relative
            original = source.read_bytes()
            changed = source_replacement(original, before, after)
            a_file, a_hash, a_gzip = archived(original, directory/(name+'.original.raw.gz'))
            b_file, b_hash, b_gzip = archived(changed, directory/(name+'.mutant.raw.gz'))
            witness = {'name': name, 'production_file': relative, 'test': node,
                'replacement': {'before': before, 'after': after},
                'original_snapshot': a_file, 'original_sha256': a_hash, 'original_archive_sha256': a_gzip,
                'mutant_snapshot': b_file, 'mutant_sha256': b_hash, 'mutant_archive_sha256': b_gzip,
                'source_before': support.identity(), 'accepted': False, 'assertion_failure': False}
            print(support.now(), 'START', name, flush=True)
            command_error = None
            try:
                source.write_bytes(changed)
                support.require(source.read_bytes() == changed, 'Mutant write changed raw bytes.')
                witness['source_on'] = support.identity()
                expected_guard = dict(guard_before)
                expected_guard['maps'] = dict(guard_before['maps'])
                expected_guard['maps']['wavebit'] = dict(guard_before['maps']['wavebit'])
                expected_guard['maps']['wavebit'][relative] = b_hash
                support.require(support.guard() == expected_guard,
                                'The mutation window changed an unowned file.')
                witness.update(run_pytest([node], directory/(name+'.log'), f'{attempt}-{name}'))
            except BaseException as error:
                command_error = error
                witness['launcher_error'] = repr(error)
            finally:
                source.write_bytes(original)
                support.require(source.read_bytes() == original, 'Exact restoration failed: '+name)
                witness['restored_sha256'] = support.sha(source)
                witness['source_restored'] = support.identity()
                support.verify_guard(guard_before)
                record['restored'] = witness['source_restored'] == source_before
                record['witnesses'].append(witness)
                support.write(path, record)
            if command_error is not None:
                raise command_error
            output = support.inside(witness['raw_log']).read_text(encoding='utf-8')
            witness['assertion_failure'] = assertion_failure(output)
            try:
                support.check_log(output)
                statuses, totals = support.reports(output), support.summary(output)
                witness.update(node_status=statuses, totals=totals)
                witness['accepted'] = bool(witness['exit_code'] == 1
                    and witness['assertion_failure'] and statuses == {node: 'failed'}
                    and totals['failed'] == 1 and sum(totals.values()) == 1)
            except ValueError as error:
                witness['acceptance_error'] = str(error)
            support.write(path, record)
            print(support.now(), 'END', name, 'accepted', witness['accepted'],
                  'restored', record['restored'], flush=True)
        final_log = directory/'restored_selected_baseline.log'
        final = run_pytest(TARGETS, final_log, f'{attempt}-restored')
        output = final_log.read_text(encoding='utf-8')
        final['observed'] = support.audit_pytest(collection_text, output)
        totals = final['observed']['totals']
        final.update(passed=totals['passed'], collected=len(nodes))
        record['final_baseline'] = final
        support.require(final['exit_code'] == 0 and totals['passed'] == len(nodes)
                        and sum(totals.values()) == len(nodes), 'Restored baseline is not all-pass.')
        support.verify_guard(guard_before)
        accepted_functions = {w['test'].split('[')[0] for w in record['witnesses'] if w['accepted']}
        record['all_function_assertion_witnesses'] = bool(
            len(record['witnesses']) == len(roster) and accepted_functions == wanted
            and all(w['accepted'] for w in record['witnesses']))
        record['source_after'] = support.identity()
        record['restored'] = record['source_after'] == source_before
        record['passed'] = record['restored'] and record['all_function_assertion_witnesses']
    except BaseException as error:
        record['controller_error'] = repr(error)
        record['source_after'] = support.identity()
        record['restored'] = record['source_after'] == source_before
        raise
    finally:
        record['ended_utc'] = support.now()
        support.write(path, record)
    if record['passed']:
        support.write(RESULTS/(PREFIX+'mutations.json'), record, exclusive=True)
    print({'passed': record['passed'], 'restored': record['restored'],
           'accepted': sum(w['accepted'] for w in record['witnesses']),
           'functions': len(wanted), 'manifest': support.relative(path)}, flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--baseline-log', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.attempt, args.baseline_log))
