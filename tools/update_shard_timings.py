#!/usr/bin/env python3
"""Build a fixed CI timing table from a complete successful run's downloaded artifacts."""
import argparse
import json
import math
from pathlib import Path

from verify_shards import check


def build(directory, source_run):
    partitions = {}
    for label, partition in [('default', 'exclude'), ('slow', 'only')]:
        directories = sorted(directory.glob(f'verification-{label}-*'))
        errors, _ = check(directories, len(directories))
        if not directories or errors:
            raise ValueError(f'{label}: incomplete baseline: {errors}')
        files = {}
        for folder in directories:
            receipt = json.loads((folder / 'receipt.json').read_text())
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
                row = files.setdefault(node.split('::', 1)[0], dict(nodes=0, seconds=0.0))
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
    args = parser.parse_args()
    table = build(args.directory, args.source_run)
    args.output.write_text(json.dumps(table, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
