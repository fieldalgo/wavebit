# Installation

This is **0.0.4**, an alpha. It is on PyPI:

```shell
pip install wavebit
```

Or from the source checkout for the version you want to evaluate:

```shell
pip install .
```

A source archive can be extracted and installed with the same command from its root.
Installing the repository's default branch may give a different version. Check the installed
version with `python -c "import wavebit; print(wavebit.__version__)"`.
The alpha API may change; the experimental auxiliary solver has the narrower scope in
[method selection](../guide/rcwa/nvf-method-selection.md).

The base installation runs both public solvers on NumPy. JAX and PyTorch add supported
automatic differentiation. RCWA also supports compatible accelerator execution. General-scene
FDTD requires double precision: NumPy and JAX use CPU, while PyTorch also admits the
[bounded CUDA scene contract](../validation/fdtd/general-cuda.md). Backend availability does
not widen a solver's physical or numerical scope; use `check` before preparing a problem
and read the [execution guide](../guide/common/backends.md).

Install an optional backend when you need it:

```shell
pip install '.[jax]'
```

The extras are `jax`, `pytorch`, `contour`, `plot` and `docs`, and they combine —
`[jax,pytorch]` installs both backends. Use `[plot]` for the plotting helpers.

## For a checkout you intend to edit

```shell
git clone https://github.com/fieldalgo/wavebit.git
cd wavebit
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -c "import wavebit; print(wavebit.__file__)"
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Confirm that the import path is
this checkout's `src/wavebit/__init__.py`. Then run the NumPy [RCWA first
calculation](rcwa.md) and [FDTD pulse calculation](fdtd.md). These small examples verify that
you can execute the current API; their resolution is not a convergence recommendation.

Install `pytest` separately when developing the library:

```shell
python -m pip install pytest
python tools/run_verification.py --profile quick
```

The [verification runner](../development/verification.md) records this small smoke profile and
provides explicit full/research profiles with safe resumption. For direct pytest commands,
`python -m pytest tests/ -q` deselects research and skips slow cases.
`--run-slow` enables slow cases within the selected set; it does **not** select research
cases. `python -m pytest -m research tests/ -q` selects research alone, and
`python -m pytest -m "research or not research" tests/ -q` selects both tiers, including
slow cases. A named test file opts in to its research and slow cases. See the
[test guide](https://github.com/fieldalgo/wavebit/blob/main/tests/README.md) before starting
a larger validation run and the [source guide](../development/source-checkout.md) for debugging.

`environment.yml` builds the full development environment, including the tools the studies need:

```shell
conda env create -f environment.yml    # run from the repo root
conda activate wavebit
```

## Requirements, and why the floors are high

Python ≥ 3.12, NumPy ≥ 2.5.1, SciPy ≥ 1.18 and array-api-compat. With the optional backends,
`jax >= 0.11` or `torch >= 2.14`.

The floors are the versions the suite is **actually verified against**, not the oldest that
might work. Loosening one later is safe; tightening one after a release is not, so they start
strict. Two of them also sit above a known defect — NumPy 2.2.5 through 2.3.0 mis-report the
floating-point status of a complex `matmul`, printing hundreds of `RuntimeWarning: ...
encountered in matmul` while returning correct efficiencies, and torch below 2.13 cannot compile
the solver at all. Every floor is commented in
[`pyproject.toml`](https://github.com/fieldalgo/wavebit/blob/main/pyproject.toml) with the
reason it sits where it does. If you need an older one, verify it and the floor can come down.

`contourpy` is optional because only `raster_normal_method='contour'` reads it; it ships with
matplotlib, so most environments already have it.
