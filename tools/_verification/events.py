"""Reduce exact pytest phases; a passed call without teardown is incomplete."""
from collections import Counter
import json
from pathlib import Path


def read_events(path):
    path = Path(path)
    if not path.exists():
        return [], []
    rows, errors = [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            errors.append(f'{path.name}:{number}: incomplete JSON event')
    return rows, errors


def summarize(nodes, phases, inherited=None):
    outcomes = dict(inherited or {})
    errors = []
    seen = {}
    for phase in phases:
        rows, failures = read_events(Path(phase['directory']) / 'events.jsonl')
        errors.extend(failures)
        for row in rows:
            if row['event'] != 'report':
                continue
            node = row['nodeid']
            if node not in nodes:
                errors.append(f'Unexpected report: {node}')
                continue
            key = node, row['when']
            if key in seen:
                errors.append(f'Duplicate report: {node} ({row["when"]})')
            seen[key] = row
    for node in nodes:
        reports = {when: seen[node, when] for when in ('setup', 'call', 'teardown')
                   if (node, when) in seen}
        if not reports:
            continue
        complete = 'teardown' in reports
        failed_phase = next((when for when in ('setup', 'teardown')
                             if reports.get(when, {}).get('outcome') == 'failed'), None)
        call = reports.get('call', {})
        skipped = next((r for r in reports.values() if r['outcome'] == 'skipped'), None)
        if failed_phase:
            outcome = 'error'
        elif call.get('outcome') == 'failed':
            outcome = 'failed'
        elif skipped:
            outcome = 'xfailed' if skipped.get('wasxfail') is not None else 'skipped'
        elif call.get('outcome') == 'passed':
            outcome = 'xpassed' if call.get('wasxfail') is not None else 'passed'
        else:
            outcome = 'interrupted'
        if not complete:
            outcome = 'interrupted'
        outcomes[node] = dict(outcome=outcome, complete=complete,
                              failed_phase=failed_phase,
                              wasxfail=(skipped or call).get('wasxfail'),
                              duration=sum(r['duration'] for r in reports.values()))
    for node in nodes:
        outcomes.setdefault(node, dict(outcome='not_run', complete=False))
    counts = dict(Counter(row['outcome'] for row in outcomes.values()))
    return outcomes, counts, errors
