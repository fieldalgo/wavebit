"""Execute the two current API notebooks in separate installed-package interpreters.

Only ordinary Python code cells are supported. Each notebook starts in an empty
working directory with no previous notebook state. Markdown and stored outputs
are not executed; scientific assertions in the cells must pass.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ('tutorials/rcwa/00-unified-lifecycle.ipynb',
           'tutorials/fdtd/00-unified-lifecycle.ipynb')
CHILD = '''import json, sys
notebook = json.load(sys.stdin)
namespace = {"__name__": "__main__"}
for number, source in notebook["cells"]:
    exec(compile(source, notebook["path"] + "#cell-" + str(number), "exec"), namespace)
print("Executed", len(notebook["cells"]), "code cells")
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='*', type=Path)
    args = parser.parse_args()
    paths = args.notebooks or [ROOT / name for name in CURRENT]
    for path in paths:
        path = path.resolve()
        notebook = json.loads(path.read_text(encoding='utf-8'))
        cells = [(i, ''.join(cell['source'])) for i, cell in enumerate(notebook['cells'], 1)
                 if cell['cell_type'] == 'code' and ''.join(cell['source']).strip()]
        if not cells:
            raise ValueError(f'{path}: no executable code cells')
        payload = json.dumps({'path': str(path), 'cells': cells})
        with tempfile.TemporaryDirectory(prefix='wavebit-tutorial-') as work:
            env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=work)
            result = subprocess.run([sys.executable, '-I', '-c', CHILD], input=payload,
                text=True, capture_output=True, check=False, cwd=work, env=env, timeout=120)
        if result.returncode:
            print(result.stdout, end='')
            print(result.stderr, file=sys.stderr, end='')
            raise RuntimeError(f'{path}: code-cell execution failed')
        print(f'{path.relative_to(ROOT)}: {len(cells)} code cells passed', flush=True)


if __name__ == '__main__':
    main()
