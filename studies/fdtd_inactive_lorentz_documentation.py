"""Final docs contract and Sphinx -W with immutable document-source records."""
import argparse
import os
from fdtd_inactive_lorentz_support import (ROOT, RESULTS, PREFIX, load, require, verify_guard,
    attempt_record, docs_map, pytest_pair, run_command, write, finish)


def main(attempt):
    setup = load(RESULTS/(PREFIX+'prevalidation.json'))
    verify_guard(setup['guard'])
    validation = load(RESULTS/(PREFIX+'validation.json'))
    require(validation['complete'], 'Finish numerical gates before final documentation checks.')
    executable = os.environ.get('WAVEBIT_DOCS_SPHINX')
    require(executable and os.path.isfile(executable), 'Set the existing Sphinx executable path.')
    directory, manifest, record = attempt_record('final_docs', attempt)
    record['sources'] = docs_map()
    write(manifest, record)
    pytest_pair(record, manifest, 'docs', ['tests/test_docs_contract.py'], directory)
    require(docs_map() == record['sources'], 'Docs changed during contract test.')
    output = ROOT/'.inactive_lorentz_work'/f'docs_html_attempt{attempt}'
    require(not output.exists(), 'Preserve earlier HTML output.')
    row, _ = run_command(record, manifest, 'sphinx', [executable, '-W', '--keep-going',
        '-b', 'html', 'docs', str(output)], directory)
    require(row['exit'] == 0, 'Sphinx failed.')
    require(docs_map() == record['sources'], 'Docs changed during build.')
    finish('final_docs', manifest, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=int)
    main(parser.parse_args().attempt)
