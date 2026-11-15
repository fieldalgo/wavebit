#!/usr/bin/env python3
"""Build a fixed CI timing table from a complete successful run's downloaded artifacts.

Rows are shard units: a test file, or a `shard_unit`-marked unit within one, as each
shard's receipt records it. `--units` re-keys the baseline's nodes by a newer collection's
units, for markers added after the baseline ran: its durations are per node, so a file that
has since been split into units gets one measured row per unit instead of its file's mean."""
import argparse
import json
import math
from pathlib import Path

from verify_shards import check


def units_of(paths):
    """Node to unit, merged from `shard.json` files of `--collect-only` runs (or receipts)."""
    units = {}
    for path in paths:
        record = json.loads(Path(path).read_text(encoding='utf-8'))
        record = record.get('shard', record)
        if not isinstance(record.get('units'), dict):
            raise ValueError(f'{path}: no shard units; pass the shard.json of a --collect-only run')
        units.update(record['units'])
    return units


def build(directory, source_run, units=None):
    partitions = {}
    for label, partition in [('default', 'exclude'), ('slow', 'only')]:
        directories = sorted(directory.glob(f'verification-{label}-*'))
        errors, _ = check(directories, len(directories))
        if not directories or errors:
            raise ValueError(f'{label}: incomplete baseline: {errors}')
        files = {}
        for folder in directories:
            receipt = json.loads((folder / 'receipt.json').read_text(encoding='utf-8'))
            if receipt.get('collect_only') or not receipt.get('inputs_unchanged'):
                raise ValueError(f'{folder}: baseline must be an unchanged execution')
            if set(receipt['outcomes']) != set(receipt['collected_nodeids']):
                raise ValueError(f'{folder}: baseline outcomes differ from collection')
            for node, outcome in receipt['outcomes'].items():
                seconds = outcome.get('duration')
                if (not outcome.get('complete') or outcome.get('outcome') not in
                        ('passed', 'skipped', 'xfailed', 'xpassed') or
                        not isinstance(seconds, (int, float)) or
                        not math.isfinite(seconds) or seconds < 0):
                    raise ValueError(f'{node}: incomplete or invalid timing')
                recorded = receipt['shard'].get('units', {}) if units is None else units
                unit = recorded.get(node) or node.split('::', 1)[0]
                row = files.setdefault(unit, dict(nodes=0, seconds=0.0))
                row['nodes'] += 1
                row['seconds'] += seconds
        for row in files.values():
            row['seconds'] = round(row['seconds'], 6)
        partitions[partition] = files
    return dict(schema='wavebit.shard-timings/v1', source_run=source_run,
                partitions=partitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--source-run', required=True, help='GitHub Actions run URL for provenance')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--units', type=Path, action='append', metavar='SHARD_JSON',
                        help="shard.json of a --collect-only run on the tree the table is for, "
                             "once per partition; its units replace the baseline's own")
    args = parser.parse_args()
    table = build(args.directory, args.source_run, units_of(args.units) if args.units else None)
    args.output.write_text(json.dumps(table, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
