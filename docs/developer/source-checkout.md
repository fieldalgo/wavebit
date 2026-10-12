# Source checkout and debugging

Use a separate virtual environment for each worktree. An editable installation
points at one checkout; another worktree does not change that pointer.

From the checkout you want to edit, with Python 3.12 or newer:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[jax,pytorch,contour,plot,docs]'
python -m pip install pytest
python -c "import wavebit; print(wavebit.__file__)"
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. The module path must end
in this checkout's `src/wavebit/__init__.py`. Install only the optional backends you
need. Restart notebook kernels after structural changes.

## Step through the source

Select this checkout's `.venv` interpreter in your IDE. Set a breakpoint in
`src/wavebit/`, then debug your script or a named test with that interpreter.
The standard Python debugger also works:

```sh
python -m pdb examples/shapes_1d.py
python -m pytest tests/test_manifest.py -q
```

With `pdb`, `s` steps into a Python call and `n` advances to the next line. NumPy is
useful for following solver logic line by line. JAX JIT and compiled kernels do
not expose every operation as a Python step; disable JIT for inspection, then
validate the compiled calculation separately.

## Using a shared scientific environment

A shared environment is an explicit alternative when duplicating large dependencies
is inconvenient. Prepend this checkout's source directory for each command:

```sh
PYTHONPATH="$PWD/src" python -c "import wavebit; print(wavebit.__file__)"
PYTHONPATH="$PWD/src" python -m pytest tests/test_manifest.py -q
```

The working directory alone does not select the package in a `src` layout.
For IDE debugging, set the launch environment's `PYTHONPATH` to the workspace's
`src` directory and verify the import path. Changing a shared editable installation
affects every process started from that environment, so avoid it for concurrent
worktree development.
