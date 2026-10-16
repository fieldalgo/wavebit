#!/usr/bin/env python3
"""Check that the shards of one verification run add up to its whole collection. See --help.

`run_verification.py --shard INDEX/COUNT` runs one shard of a profile and records, beside the
shard's own receipt, the full collection every shard saw (`collected-nodeids-all.json`). This
tool reads COUNT such run directories and refuses to call the profile verified unless every
shard is present exactly once, all of them collected the same list, their shards partition
that list (every node in exactly one shard), and every shard finished with exit code 0 and
no integrity errors. CI runs it after the sharded jobs; a missing or failed shard fails here
by name instead of disappearing into a green checkmark.
"""
import argparse
import json
from pathlib import Path
import sys


def read(directory, name):
    return json.loads((directory / name).read_text())


def check(directories, count):
    """Return (errors, counts): the problems found, and the summed outcome counts."""
    errors, seen, counts, shards = [], {}, {}, []
    for directory in directories:
        try:
            receipt = read(directory, 'receipt.json')
        except (OSError, ValueError) as exc:
            errors.append(f'{directory}: unreadable receipt ({exc})')
            continue
        shard = receipt.get('shard')
        if not shard:
            errors.append(f'{directory}: receipt is not from a --shard run')
            continue
        try:
            collection = read(directory, 'collected-nodeids-all.json')
        except (OSError, ValueError) as exc:
            errors.append(f'{directory}: unreadable full collection ({exc})')
            continue
        if shard['count'] != count:
            errors.append(f'{directory}: shard {shard["index"]}/{shard["count"]}, expected COUNT {count}')
            continue
        if shard['index'] in seen:
            errors.append(f'{directory}: shard {shard["index"]} also recorded by {seen[shard["index"]]}')
            continue
        seen[shard['index']] = directory
        shards.append((directory, receipt, collection))
    missing = sorted(set(range(1, count + 1)) - set(seen))
    if missing:
        errors.append('missing shard(s): ' + ', '.join(f'{index}/{count}' for index in missing))
    if not shards:
        return errors, counts
    reference = shards[0][2]
    timings_digest = shards[0][1]['shard'].get('timings_digest')
    union, duplicates = set(), set()
    for directory, receipt, collection in shards:
        if receipt['shard'].get('timings_digest') != timings_digest:
            errors.append(f'{directory}: shards used different timing tables')
        if collection != reference:
            errors.append(f'{directory}: collected a different full list than {shards[0][0]}')
        own = receipt.get('collected_nodeids', [])
        duplicates |= union & set(own)
        union |= set(own)
        if receipt.get('exit_code') != 0:
            errors.append(f'{directory}: shard {receipt["shard"]["index"]} exit code {receipt.get("exit_code")}')
        for problem in receipt.get('integrity_errors', []):
            errors.append(f'{directory}: {problem}')
        for outcome, number in receipt.get('counts', {}).items():
            counts[outcome] = counts.get(outcome, 0) + number
    if duplicates:
        errors.append(f'{len(duplicates)} node(s) ran in more than one shard, e.g. {sorted(duplicates)[0]}')
    unrun = set(reference) - union
    if unrun and not missing:
        errors.append(f'{len(unrun)} collected node(s) belong to no shard, e.g. {sorted(unrun)[0]}')
    if union - set(reference):
        errors.append('a shard ran nodes outside the full collection')
    return errors, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--count', type=int, required=True, help='the COUNT the shards were run with')
    parser.add_argument('directories', nargs='+', type=Path,
                        help='run directories (each holds receipt.json and collected-nodeids-all.json)')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be at least 1')
    errors, counts = check(args.directories, args.count)
    print(json.dumps(dict(shards=args.count, counts=counts, errors=errors), indent=2))
    if errors:
        print(f'Verification incomplete: {len(errors)} problem(s); see errors above.', file=sys.stderr)
        return 1
    print(f'All {args.count} shards present and consistent: {sum(counts.values())} nodes recorded.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
