<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/wavebit_logo_dark.svg">
    <img src="images/wavebit_logo.svg" width="45%" alt="Wavebit">
  </picture>
</p>

# Wavebit

[![tests](https://github.com/fieldalgo/wavebit/actions/workflows/tests.yml/badge.svg)](https://github.com/fieldalgo/wavebit/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

**Differentiable electromagnetic simulation suite — by [FieldAlgo](https://fieldalgo.com)**

Wavebit simulates how light interacts with nanostructures. Its RCWA path differentiates
through the calculation, so the same model can predict and optimize a device response.

**Alpha candidate: 0.1.0.** The API may change. The ordinary RCWA/NVF solver and the
experimental contact auxiliary solver have different validation scopes. The auxiliary solver
is opt-in, NumPy-only, and limited to one supported rectangle pair; its optical convergence,
contact derivatives and practical runtime remain under investigation. Ordinary NVF does not
call it automatically. See [method selection](docs/guide/nvf-method-selection.md) before using
that experimental path.

* **Modeling** — describe a structure parametrically or as a raster
* **EM simulation** — RCWA solver, rigorous and fully vectorial
* **Optimization** — gradients flow end to end, from geometry to diffraction efficiency

The ordinary solver uses one implementation of the physics on NumPy, JAX and PyTorch;
cross-backend agreement is checked in the test suite. Experimental auxiliary families have
the narrower backend and derivative support stated in their guides.

**The periodic-stack FDTD entry point, `wavebit.fdtd(...)`, is an experimental
research-alpha path.** It supports periodic raster scenes and shares the RCWA
scene and result interfaces. An explicit `backend`
selects the common NumPy/JAX/PyTorch engine; JAX and PyTorch support angle,
material and constant-total layer-thickness derivatives on a fixed mesh. Omitting `backend` selects the legacy
NumPy path. CPU execution is tested; GPU validation remains outstanding.
See the [FDTD scope and validation report](docs/fdtd-alpha.md) before relying on a new
geometry, angle or material combination.

The separate general engine, `import wavebit.fd as fd`, uses explicit `fd.Scene`
domains, boundaries, sources and monitors for finite or periodic problems, on
NumPy, JAX and PyTorch CPU backends. Its own material, coordinate and angle
contracts are in the [general FDTD guide](docs/guide/general-fdtd.md), including
its current geometry and validation limits.

## Install

The solver release has not been published to PyPI. Install this alpha candidate from its
source checkout:

```shell
pip install .
```

Use the source checkout or distribution attached to the version you want to evaluate;
a Git installation from the repository's default branch can contain a different version.

NumPy alone runs the solver. JAX and PyTorch are optional and bring autodiff and GPU support:

```shell
pip install '.[jax]'
```

For a checkout you intend to edit, or the full development environment:

```shell
git clone https://github.com/fieldalgo/wavebit.git && cd wavebit
pip install -e '.[jax]'                # or '.[pytorch]', or plain -e . for NumPy only
conda env create -f environment.yml    # everything, including what the studies need
```

Python ≥ 3.12. The version floors in [`pyproject.toml`](pyproject.toml) are the ones the suite
is verified against rather than the oldest that might work, and each is commented with why it
sits where it does.

## Quickstart

A binary grating, solved on the NumPy backend:

```python
import numpy as np
import wavebit

# One 300 nm layer, half air and half n = 3.5, on a glass substrate. Materials are relative
# permittivity, so an index off a datasheet goes through `eps_from_index` first.
eps_hi = wavebit.eps_from_index(3.5)                     # 12.25
structure = np.array([[[1.0, 1.0, eps_hi, eps_hi]]])     # (layer, y, x)

solver = wavebit.rcwa(
    backend='numpy',                          # or 'jax' / 'torch'
    wavelength=900., period=[700.],           # one shared length unit; nm here
    truncation_order=dict(x=10, y=0),         # named, so there is no order to recall
    top=1., bot=wavebit.eps_from_index(1.5),  # the two half-spaces, as materials
    theta=0., phi=0., tetm='te',
    structure=structure, thickness=[300.],
)

result = solver.solve()
print(result.R.sum(), result.T.sum())    # 0.20533162825847726 0.7946683717415197
print(result.order(0, 0).T)              # 0.3531011230787259 — the zeroth transmitted order
```

`R` and `T` are the reflected and transmitted diffraction efficiencies, one entry per order.
Here they sum to 1 to within a few times 1e-15, as a lossless structure must.

[`tutorials/01-modeling-and-emsolver.ipynb`](tutorials/01-modeling-and-emsolver.ipynb) walks the
same scene in full, and the tutorials after it take one topic each.

## Conventions

Seven rows decide whether your input means what you think it means. Every one of them is legal,
energy-conserving and *wrong* when read backwards — no error, no warning — which is why they are
worth the two minutes. [`docs/conventions.md`](docs/conventions.md) is the normative contract.
This table describes the RCWA and periodic-stack interfaces; the separate
[general FDTD conventions](docs/guide/general-fdtd.md#units-coordinates-and-backends)
include degrees for incidence angles and normalized time units.

| | |
| --- | --- |
| **Materials are relative permittivity** | not refractive index. `eps_from_index` / `index_from_eps` convert, and the retired index-shaped keywords (`ucell`, `n_top`, `n_bot`) are refused by name because the quantity moved with them |
| **Time convention is `e^{-iωt}`** | `Im(eps) > 0` is loss — equivalently `Im(n) > 0`, how nk databases tabulate it — and a wave travelling in `+z` carries `exp(+i*k0*kz*z)` |
| **Layers run bottom to top; `+z` points upward** | `structure[0]` touches `bot`; the bottom interface is `z=0`. Raster row `0` is at the negative-y edge. Select the light with `incidence_side` |
| **Uppercase is power, lowercase is amplitude** | `R` / `T` are per-order efficiencies; `r_s` / `r_p` / `t_s` / `t_p` are the complex Fresnel amplitudes, each order in its own s/p frame; `r_xyz` / `t_xyz` are the same waves as Cartesian E vectors, with no frame to turn over |
| **Arrays run `v` first, written pairs run `x` first** | `truncation_order=[4, 1]` gives `R.shape == (3, 9)`. `result.order(m, n)` indexes by order and sidesteps the layout entirely |
| **The cell is centred at the origin** | a centred feature is `0` in a shape's `cx` / `cy`, in `reconstruct_field`'s `sample_x` / `sample_y` and in `wavebit.meshgrid_xy` alike. Shape lengths are absolute, not fractions of the period |
| **Units are yours** | `wavelength`, `period` and `thickness` only have to share one. Angles are radians |

## Differentiating a solve

`jax.grad` and PyTorch autograd work straight off the public API — a grad tracer carries a
concrete value, so the pipeline's geometric decisions read it one at a time:

```python
import jax, jax.numpy as jnp

solver = wavebit.rcwa(backend='jax', structure=structure, ...)
grad = jax.grad(lambda u: jnp.sum(solver.solve(structure=u).T))(structure)
```

`jax.jit` and `vmap` are the harder bar, because those same decisions branch on values that a
jit tracer does not have. `freeze_geometry()` takes them once, against the concrete structure,
and hands them back as data:

```python
geometry = solver.freeze_geometry()
step = jax.jit(lambda u: jnp.sum(solver.solve(structure=u, geometry=geometry).T))
```

Re-take the freeze every step rather than holding one. It costs 0–1 ms, compares by content so
an unchanged record is free, and it is the only form that cannot carry a stale one forward.
Which backend to use, and whether to jit at all, differs by modeling path and is measured in
[`docs/backend-choice.md`](docs/backend-choice.md).

## Validation

The ordinary solver is checked against references, not only against itself. These checks
cover their recorded cases; they do not certify the experimental contact auxiliary solver's
optical convergence or opening derivative:

* **Three outside RCWA solvers** — RETICOLO through an Octave bridge, plus fmmax and S4. Their
  numbers are committed as goldens and checked in CI, so the half that needs software this
  project does not ship runs once, by hand. RETICOLO agrees to ~1e-13 and is the only reference
  here that validates individual diffraction *orders* rather than totals.
* **A different method entirely** — Wavebit's own FDTD ([`docs/fdtd.md`](docs/fdtd.md)) solves
  the same scene in the time domain, and Meep is the outside reference for it. RCWA truncates a
  Fourier series and FDTD discretises space and time, so a defect surviving both belongs to
  neither. Meep's answers are committed as
  [`tests/references/golden_meep_1d.npz`](tests/references/golden_meep_1d.npz) (the 1D
  benchmark, both polarisations, two resolutions, with a planar closed-form calibration
  gating the run) and `golden_meep_crossed.npz` (a crossed grating at a conical mount,
  per-order complex amplitudes), and CI reads them on every push: on the benchmark grating at
  the 5 nm grid, our worst per-order gap against the converged RCWA answer is 6.6e-4 (TE) /
  6.9e-4 (TM), against Meep's own 4.9e-4 / 7.4e-4 on that grid — read both against the fact
  that the Meep column moves by 1.6e-3 between two of its own PML settings at that same
  resolution, for reasons §2 records as unexplained, so neither figure is tighter than the
  oracle's own spread. Those four numbers, and every
  cell of the table they summarise, are read back out of the markdown and compared to
  [`golden_fdtd_table.npz`](tests/references/golden_fdtd_table.npz) — with the conditions all
  three columns were taken at written out by value in `tests/references/_meep_1d_cases.py`,
  because until 2026-09-07 they were the phrase "at the shipped defaults" and one column had
  quietly gone stale behind it. Meep is GPL and this project is Apache-2.0, so it is run as a
  black-box oracle and is not a dependency of anything here.
* **Published benchmark tables** — Li's 1997 crossed-grating checkerboard as tabulated by
  Demésy et al. (2012). All nine transmitted orders agree to a worst 2.06e-4, on the oblique
  primitive cell, which puts that path against numbers from outside this project.
* **Energy conservation** for lossless structures, at machine precision (|R+T−1| ~ 1e-15).
* **Cross-backend agreement** — NumPy, JAX and PyTorch on the same input.
* **Analytic gradients against finite differences**, for the parameters an optimizer moves.

The suite is [`tests/`](tests); everything with a statable criterion lives there, including the
slower cross-backend and optimizer checks. CI runs its default set on every push. The tests of
the experimental conforming NVF module and its study instruments form a
[research tier](tests/README.md#the-research-tier) that CI does not run; it is run while that
research code is being developed. Every eager NumPy solve also checks what it is
about to hand back — a NaN or infinity raises `FloatingPointError` naming the contaminated
array.

There is deliberately no generic "R+T above 1" warning. It cannot tell a broken solve from a
structure that legitimately amplifies: gain is `Im(eps) < 0` here, and `R + T > 1`, or a
negative order under total internal reflection, is then the right answer. Energy is asserted
per scene in the suite, where the expected number is known; see
[`docs/result-guards.md`](docs/result-guards.md).

## Documentation

The prose is split by **who is reading**, not by subject — the same three-way split NumPy's
manual uses, with the decision records outside the manual the way NEPs are.

| | For a reader who | |
| --- | --- | --- |
| [`docs/guide/`](docs/guide) | needs the contract for one keyword | **The manual.** What Wavebit solves and what it is bad at, installation, a quickstart; then describing a scene, saying what the light is, choosing the numerics, near fields, autodiff, backends, and a run described as data |
| [`docs/reference/`](docs/reference) | needs a signature or a guarantee | [`conventions.md`](docs/conventions.md) is the **normative** contract; `keywords.md` is all 27 `rcwa()` keywords with their defaults and refusals; `api.md` is generated from the docstrings |
| [`docs/`](docs) | is questioning a default | **Why it is what it is**, and what was measured to decide it — the write-ups, cited from the code they explain. [`docs/decisions/`](docs/decisions) groups them |
| [`tutorials/`](tutorials) | has not used the API yet | Lessons in order, as runnable notebooks: a first solve, autograd and optimization, device and precision, NVF factorization, oblique cells, polarization, multimode incidence |
| [`examples/`](examples) | wants it applied | Wavebit against outside problems |
| [`studies/`](studies) | wants the raw measurement | The scripts behind the numbers in `docs/` |

All of it also builds as a site:

```shell
pip install -e '.[docs]'
sphinx-build -b html docs docs/_build/html      # then open docs/_build/html/index.html
```

Every page reads the same in both places, which is a constraint rather than a coincidence: a
link in `docs/` is a relative link to a real file, the navigation lives in `docs/_toc.yml`
rather than in directives inside the pages, and `tests/test_docs_contract.py` checks that every
path a document cites exists. The one thing the site has that the repository cannot show is the
API reference, which is read off the live objects at build time.

Nothing is deployed yet. `python -m wavebit.manifest` is the other machine-readable surface —
every option, refusal and constraint as one generated JSON document, for a program that needs to
know without reading the source ([`docs/capability-manifest.md`](docs/capability-manifest.md)).

## Contributing

Bug reports, questions and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

**Adding a file? Start there.** It opens with the rule for which directory a new file belongs
in, answerable without reading what is already in each one.

## Citation

If you use Wavebit in your research, please cite the software:

```bibtex
@software{wavebit,
    title  = {Wavebit: Differentiable Electromagnetic Simulation Suite},
    author = {Kim, Yongha},
    year   = {2026},
    url    = {https://github.com/fieldalgo/wavebit},
}
```

## License

Wavebit is licensed under Apache-2.0. See [LICENSE](LICENSE), which also records the
third-party notices this project carries.

## Contact

[📩 FieldAlgo](mailto:dev@fieldalgo.com)
