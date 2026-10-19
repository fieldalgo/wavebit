#!/usr/bin/env python3
"""Record and safely resume explicit verification profiles. See --help."""
import argparse
import codecs
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

from _verification.events import read_events, summarize
from _verification.inputs import digest, environment, hashes, snapshot

QUICK = (
    'tests/api/test_installation_metadata.py',
    'tests/api/test_unified_rcwa.py::test_magnetic_conical_slab_matches_independent_fresnel',
    'tests/api/test_unified_rcwa.py::test_physical_amplitude_scales_power_and_si_fields_but_not_efficiency',
    'tests/api/test_unified_rcwa.py::test_si_conductivity_and_physical_power_are_independent_of_length_unit',
    'tests/api/test_unified_jobs.py::test_table_roundtrip_preserves_unit_interpolation_and_permeability',
    'tests/api/test_unified_jobs.py::test_fdtd_job_roundtrip_preserves_physical_source_geometry_and_monitor_declarations',
    'tests/materials/test_common_material_models.py::test_import_and_new_execution_do_not_mutate_jax_precision',
    'tests/fdtd/test_lifecycle.py::test_check_never_advances_fields',
    'tests/fdtd/test_lifecycle.py::test_public_h_and_conductivity_obey_si_update_equation',
    'tests/fdtd/test_lifecycle.py::test_same_physical_pulse_in_nm_and_um_has_equal_fields_and_seconds',
)
EXPRESSIONS = dict(quick='not research and not slow', full='not research', research='research')
REUSABLE = {'passed', 'skipped', 'xfailed', 'xpassed'}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def load(path):
    return json.loads(path.read_text())


def validate_paths(root, paths):
    for value in paths:
        path = (root / value.split('::', 1)[0]).resolve()
        if not path.is_relative_to(root) or not path.exists():
            raise ValueError(f'Test selection must exist inside the checkout: {value}')
    return list(paths)


def parse_shard(value):
    """'INDEX/COUNT' with 1 <= INDEX <= COUNT, as a (index, count) pair."""
    try:
        index, count = (int(part) for part in value.split('/'))
    except ValueError:
        raise ValueError(f'--shard must be INDEX/COUNT, for example 2/4: {value!r}') from None
    if count < 1 or not 1 <= index <= count:
        raise ValueError(f'--shard needs 1 <= INDEX <= COUNT: {value!r}')
    return index, count


def shard_files(nodes, count, timings=None):
    """Assign whole files by recorded duration, falling back to estimated node cost.

    Files, not nodes, because CI distributes with `--dist loadfile` and because test modules
    set process-wide state at import (see tests.yml), so a file must never be split.
    The same collection and timing table always give the same assignment. Without
    a timing table, retain the node-count policy for local and research runs.
    """
    sizes = {}
    for node in nodes:
        name = node.split('::', 1)[0]
        sizes[name] = sizes.get(name, 0) + 1
    timings = timings or {}
    fallback = max(1.0, sum(row['seconds'] for row in timings.values()) /
                   max(1, sum(row['nodes'] for row in timings.values())))
    weights = {name: max(0.001, timings[name]['seconds'] / timings[name]['nodes']) * size
               if name in timings else fallback * size for name, size in sizes.items()}
    assignment, load = {}, [0.0] * count
    for name in sorted(sizes, key=lambda name: (-weights[name], name)):
        shard = load.index(min(load))
        assignment[name] = shard + 1
        load[shard] += weights[name]
    return assignment


def shard_nodes(nodes, index, count, timings=None):
    """The nodes of shard `index` of `count`, in collection order, and its files."""
    assignment = shard_files(nodes, count, timings)
    files = sorted(name for name, shard in assignment.items() if shard == index)
    selected = [node for node in nodes if assignment[node.split('::', 1)[0]] == index]
    return selected, files


def collections(directory):
    records = [load(path) for path in sorted(directory.glob('collection-*.json'))]
    if not records:
        return [], [], ['No collection record was produced.']
    errors = []
    first = records[0]
    if any(record != first for record in records[1:]):
        errors.append('Workers collected different test lists.')
    for path in directory.glob('collection-errors-*.jsonl'):
        errors.extend(json.loads(line)['message'] for line in path.read_text().splitlines())
    return first['nodeids'], first['mutation_nodes'], errors


def relay_output(stream, log, errors):
    """Tee available bytes immediately; do not wait for a full line or pipe buffer."""
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    # A text-mode console writes each '\n' as os.linesep. On Windows the child already wrote
    # '\r\n', which would reach the console as '\r\r\n'; collapse it first, holding back a
    # trailing '\r' whose '\n' may arrive in the next block, so the console matches the log.
    collapse, pending = os.linesep == '\r\n', ''
    def text(data, final=False):
        nonlocal pending
        value = pending + decoder.decode(data, final=final)
        if not collapse:
            return value
        pending = '' if final or not value.endswith('\r') else '\r'
        return value[:len(value)-len(pending)].replace('\r\n', '\n')
    console = True
    try:
        while block := stream.read1(65536):
            log.write(block)
            log.flush()
            if console:
                try:
                    sys.stdout.write(text(block))
                    sys.stdout.flush()
                except BrokenPipeError:
                    console = False
        if console:
            sys.stdout.write(text(b'', final=True))
            sys.stdout.flush()
    except (OSError, ValueError) as exc:
        errors.append(f'Live output recording failed: {exc}')
    finally:
        stream.close()


def phase_run(output, name, root, config, *, expected=None, selected=None,
              collect=False, isolated=False, paths=None):
    directory = output / name
    directory.mkdir()
    contract = dict(directory=str(directory), root=str(root), expected=expected,
                    selected=selected, snapshot=isolated)
    contract_path = directory / 'phase.json'
    save(contract_path, contract)
    args = [sys.executable, '-u', '-m', 'pytest', *(paths or config['paths']),
            '-v' if config['verbose'] and not collect else '-q',
            '-m', config['expression'], '-p', '_verification.plugin',
            '--junitxml=' + str(directory / 'junit.xml'), '-o',
            'cache_dir=' + str(directory / 'cache'), '--basetemp=' + str(directory / 'temporary')]
    if config['timeout'] is not None:
        args += ['-p', 'pytest_timeout', '--timeout=' + str(config['timeout']),
                 '--timeout-method=' + config['timeout_method']]
    if collect:
        args.append('--collect-only')
    elif config['workers'] and not isolated:
        args += ['-p', 'xdist.plugin', '-n', str(config['workers']), '--dist', 'loadfile']
    env = dict(os.environ)
    for key in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'PYTHONHOME', 'PYTEST_XDIST_WORKER', 'PYTEST_XDIST_WORKER_COUNT'):
        env.pop(key, None)
    env.update(PYTHONPATH=os.pathsep.join((str(root/'src'), str(root), str(root/'tools'))),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', WAVEBIT_VERIFICATION_PHASE=str(contract_path))
    started = time.monotonic()
    interrupted = False
    live = config['live_output'] and not collect
    stream_errors = []
    with (directory / 'pytest.log').open('wb') as log:
        group = (dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                 if os.name == 'nt' else dict(start_new_session=True))
        process = subprocess.Popen(args, cwd=root, env=env,
                                   stdout=subprocess.PIPE if live else log,
                                   stderr=subprocess.STDOUT, **group)
        if live:
            relay = threading.Thread(target=relay_output, args=(process.stdout, log, stream_errors),
                                     daemon=True)
            relay.start()
        try:
            exit_code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            if os.name == 'nt':
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGINT)
            try:
                exit_code = process.wait(timeout=30)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if os.name == 'nt':
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                exit_code = process.wait()
        finally:
            if live:
                relay.join(timeout=5)
                if relay.is_alive():
                    stream_errors.append('Pytest exited but a child kept its output pipe open.')
    nodes, mutations, errors = collections(directory)
    errors.extend(stream_errors)
    result = dict(name=name, directory=str(directory), command=args, exit_code=exit_code,
                  seconds=round(time.monotonic()-started, 3), interrupted=interrupted,
                  collected=nodes, mutation_nodes=mutations, collection_errors=errors)
    save(directory / 'result.json', result)
    return result


def run(args):
    root = args.root.resolve()
    if args.timeout is not None and importlib.util.find_spec('pytest_timeout') is None:
        raise ValueError('--timeout requires pytest-timeout; install it in this Python environment.')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    os.environ.setdefault('PYTHONHASHSEED', '0')
    paths = validate_paths(root, args.path or (QUICK if args.profile == 'quick' else ['tests']))
    expression = EXPRESSIONS[args.profile]
    if args.profile != 'quick' and args.slow != 'include':
        expression += ' and ' + ('slow' if args.slow == 'only' else 'not slow')
    shard = parse_shard(args.shard) if args.shard else None
    timings = None
    if args.shard_timings:
        if not shard or args.profile != 'full' or args.slow not in ('exclude', 'only'):
            raise ValueError('--shard-timings requires --shard, --profile full and --slow exclude/only')
        table = load(args.shard_timings)
        if table.get('schema') != 'wavebit.shard-timings/v1':
            raise ValueError('Unsupported shard timing schema')
        timings = table['partitions'][args.slow]
        if not timings:
            raise ValueError('Shard timing partition is empty')
        for row in timings.values():
            if (type(row['nodes']) is not int or row['nodes'] <= 0 or
                    type(row['seconds']) not in (int, float) or
                    not math.isfinite(row['seconds']) or row['seconds'] < 0):
                raise ValueError('Shard timings need positive node counts and finite nonnegative seconds')
    config = dict(profile=args.profile, expression=expression, paths=paths,
                  workers=args.workers, plugin_autoload=False, distribution='loadfile',
                  slow=args.slow, timeout=args.timeout, timeout_method=args.timeout_method,
                  verbose=args.verbose, live_output=args.live_output,
                  shard_timings=timings,
                  shard=None if shard is None else dict(index=shard[0], count=shard[1]))
    before = hashes(root)
    inputs = dict(root=str(root), files=before, environment=environment(), config=config)
    fingerprint = digest(inputs)
    previous = None
    if args.resume:
        previous = load(args.resume.resolve() / 'receipt.json')
        if previous['fingerprint'] != fingerprint or not previous.get('inputs_unchanged'):
            raise ValueError('Resume refused: source, tests, inputs, environment or configuration changed.')
        previous_nodes = previous.get('collected_nodeids', [])
        if (len(previous_nodes) != len(set(previous_nodes)) or
                set(previous.get('outcomes', {})) != set(previous_nodes)):
            raise ValueError('Resume refused: receipt outcomes do not match its exact collection.')
        if previous.get('integrity_errors'):
            raise ValueError('Resume refused: previous receipt has collection or recording integrity errors.')
    output = (args.output or root / 'artifacts' / 'verification' /
              (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # The operating system's own message is localized; name the refusal in English.
        raise ValueError(f'Output directory already exists: {output}') from None
    save(output / 'inputs.json', inputs)
    started = time.monotonic()
    phases, nodes, inherited, errors = [], [], {}, []
    receipt = dict(schema='wavebit.verification/v1', directory=str(output), fingerprint=fingerprint,
                   profile=args.profile, start_utc=datetime.now(timezone.utc).isoformat(),
                   resumed_from=str(args.resume.resolve()) if args.resume else None)
    print(f'Verification record: {output}', flush=True)
    status = 2
    try:
        collection = phase_run(output, 'collection', root, config, collect=True)
        phases.append(collection)
        nodes = collection['collected']
        errors.extend(collection['collection_errors'])
        if collection['exit_code'] or errors or not nodes:
            raise ValueError('Collection failed or selected no tests; see collection/pytest.log.')
        all_nodes = nodes
        if shard is not None:
            # The phases still expect the whole collection -- a shard is a selection within
            # it, not a different profile -- so every shard of a run collects the same list,
            # and tools/verify_shards.py can check that the shards add up to exactly that list.
            index, count = shard
            nodes, files = shard_nodes(all_nodes, index, count, timings)
            save(output / 'collected-nodeids-all.json', all_nodes)
            receipt['shard'] = dict(index=index, count=count, files=files,
                                    timings_digest=digest(timings),
                                    nodes_total=len(all_nodes), nodes_in_shard=len(nodes))
            save(output / 'shard.json', receipt['shard'])
            if not nodes:
                raise ValueError(f'Shard {index}/{count} selects no tests; use fewer shards.')
        save(output / 'collected-nodeids.json', nodes)
        if previous:
            if previous['collected_nodeids'] != nodes:
                raise ValueError('Resume refused: collected nodeids changed (new, missing, reordered or duplicated nodes).')
            inherited = {node: dict(row, reused_from=previous['directory'])
                         for node, row in previous['outcomes'].items()
                         if row.get('complete') and row['outcome'] in REUSABLE}
        if not args.collect_only:
            pending = [node for node in nodes if node not in inherited]
            mutations = [node for node in pending if node in collection['mutation_nodes']]
            ordinary = [node for node in pending if node not in mutations]
            if ordinary:
                result = phase_run(output, 'tests', root, config, expected=all_nodes, selected=ordinary)
                phases.append(result)
                if result['interrupted'] or result['exit_code'] == 2:
                    raise KeyboardInterrupt
            if mutations:
                copy = output / 'snapshot'
                snapshot(root, copy, before)
                # Only source-writing nodes enter this private serial checkout.
                result = phase_run(output, 'mutations', copy, config, selected=mutations,
                                   isolated=True, paths=mutations)
                phases.append(result)
                restored = hashes(copy) == before
                receipt['snapshot'] = dict(path=str(copy), initially_byte_identical=True,
                                            restored=restored, selected=mutations)
                if not restored:
                    errors.append('Mutation snapshot was not restored to its original bytes.')
                if result['interrupted'] or result['exit_code'] == 2:
                    raise KeyboardInterrupt
        status = 0
    except KeyboardInterrupt:
        receipt['interrupted'] = True
        status = 130
    except (ValueError, OSError) as exc:
        errors.append(str(exc))
        status = 2
    finally:
        outcomes, counts, report_errors = summarize(nodes, phases, inherited)
        errors.extend(report_errors)
        for phase in phases:
            errors.extend(phase['collection_errors'])
            if phase['exit_code'] not in (0, 1, 2, 5, 130):
                errors.append(f'{phase["name"]} exited with unexpected code {phase["exit_code"]}.')
            if phase['exit_code'] in (1, 5):
                rows, _ = read_events(Path(phase['directory']) / 'events.jsonl')
                if not any(row.get('event') == 'report' and row.get('outcome') == 'failed'
                           for row in rows):
                    errors.append(f'{phase["name"]} exited with code {phase["exit_code"]} without a failed test report.')
        try:
            unchanged = before == hashes(root)
        except (ValueError, OSError) as exc:
            unchanged = False
            errors.append(f'Final input check failed: {exc}')
        if not unchanged:
            errors.append('Recorded inputs changed during this run; results cannot be resumed.')
        if errors:
            status = 2
        elif status == 0 and not args.collect_only:
            if (any(phase['exit_code'] != 0 for phase in phases) or
                    any(row['outcome'] in ('error', 'failed', 'interrupted', 'not_run')
                        for row in outcomes.values())):
                status = 1
        receipt.update(end_utc=datetime.now(timezone.utc).isoformat(),
                       seconds=round(time.monotonic()-started, 3), exit_code=status,
                       collect_only=args.collect_only, collected_nodeids=nodes,
                       outcomes=outcomes, counts=counts, reused_nodeids=list(inherited),
                       integrity_errors=list(dict.fromkeys(errors)), inputs_unchanged=unchanged,
                       phases=phases)
        save(output / 'receipt.json', receipt)
        print(json.dumps(dict(counts=counts, reused=len(inherited), exit_code=status,
                              receipt=str(output/'receipt.json'))), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=EXPRESSIONS, default='quick',
                        help='quick: selected smoke checks; full: all non-research including slow; research: research only')
    parser.add_argument('--slow', choices=('include', 'exclude', 'only'), default='include',
                        help='split full/research profiles by the slow marker; quick always excludes slow')
    parser.add_argument('--timeout', type=float, help='positive per-test seconds; requires pytest-timeout')
    parser.add_argument('--timeout-method', choices=('thread', 'signal'), default='thread')
    parser.add_argument('--verbose', action='store_true', help='show test nodeids with pytest -v')
    parser.add_argument('--live-output', action='store_true', help='flush test output to both console and phase log')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--path', action='append', help='restrict the profile to a repository path or nodeid (repeatable)')
    parser.add_argument('--workers', type=int, default=0, help='xdist workers; 0 is serial (default)')
    parser.add_argument('--output', type=Path, help='new output directory; an existing path is always refused')
    parser.add_argument('--resume', type=Path, help='reuse completed successful/skip/xfail nodes from an identical run')
    parser.add_argument('--shard', metavar='INDEX/COUNT',
                        help='run only shard INDEX of COUNT, whole test files assigned deterministically; '
                             'every shard records the full collection, tools/verify_shards.py checks they add up')
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--shard-timings', type=Path,
                        help='fixed file-duration JSON for time-balanced full-profile CI shards')
    args = parser.parse_args()
    if args.profile == 'quick' and args.slow == 'only':
        parser.error('--profile quick cannot select --slow only')
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error('--timeout must be a positive finite number of seconds')
    if args.workers < 0:
        parser.error('--workers cannot be negative')
    try:
        return run(args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f'Verification refused: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
