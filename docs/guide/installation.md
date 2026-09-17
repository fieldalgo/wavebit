# Installation

This checkout prepares **0.1.0**, an alpha candidate. The solver release has not been
published to PyPI. From the source checkout for the version you want to evaluate:

```shell
pip install .
```

A source archive can be extracted and installed with the same command from its root.
Installing the repository's default branch may give a different version. Check the installed
version with `python -c "import wavebit; print(wavebit.__version__)"`.
The alpha API may change; the experimental auxiliary solver has the narrower scope in
[method selection](nvf-method-selection.md).

NumPy alone runs the solver. JAX and PyTorch are optional and bring automatic differentiation
and GPU support:

```shell
pip install '.[jax]'
```

The extras are `jax`, `pytorch`, `contour` and `docs`, and they combine —
`[jax,pytorch]` installs both backends.

## For a checkout you intend to edit

```shell
git clone https://github.com/fieldalgo/wavebit.git && cd wavebit
pip install -e '.[jax]'                # or '.[pytorch]', or plain -e . for NumPy only
pytest tests/ -q
```

`environment.yml` builds the full development environment, including the tools the studies need:

```shell
conda env create -f environment.yml    # run from the repo root
conda activate wavebit
```

## Requirements, and why the floors are high

Python ≥ 3.12, NumPy ≥ 2.5.1, SciPy ≥ 1.18 and array-api-compat. With the optional backends,
`jax >= 0.11` or `torch >= 2.13`.

The floors are the versions the suite is **actually verified against**, not the oldest that
might work. Loosening one later is safe; tightening one after a release is not, so they start
strict. Two of them also sit above a known defect — NumPy 2.2.5 through 2.3.0 mis-report the
floating-point status of a complex `matmul`, printing hundreds of `RuntimeWarning: ...
encountered in matmul` while returning correct efficiencies, and torch below 2.13 cannot compile
the solver at all. Every floor is commented in
[`pyproject.toml`](https://github.com/fieldalgo/wavebit/blob/develop/pyproject.toml) with the
reason it sits where it does. If you need an older one, verify it and the floor can come down.

`contourpy` is optional because only `normal_method='contour'` reads it; it ships with
matplotlib, so most environments already have it.
