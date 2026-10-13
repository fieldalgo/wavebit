<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/branding/wavebit_logo_dark.svg">
    <img src="assets/branding/wavebit_logo.svg" width="45%" alt="Wavebit">
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
call it automatically. See [method selection](docs/guide/rcwa/nvf-method-selection.md) before using
that experimental path.

* **Modeling** — describe a structure parametrically or as a raster
* **EM simulation** — RCWA solver, rigorous and fully vectorial
* **Optimization** — gradients flow end to end, from geometry to diffraction efficiency

The ordinary solver uses one implementation of the physics on NumPy, JAX and PyTorch;
cross-backend agreement is checked in the test suite. Experimental auxiliary families have
the narrower backend and derivative support stated in their guides.

The public `wavebit.rcwa` and `wavebit.fdtd` modules share an immutable lifecycle:
`check(problem, numerics=...)`, `prepare(problem, numerics=...)`, and
`run(problem, numerics=...)`. A prepared calculation always receives the complete
live problem as `prepared.run(problem)`. Both use shared materials, explicit
length units, radian angles, electric fields in V/m, and magnetic fields in A/m.
FDTD times and material rates use SI seconds and inverse seconds.

FDTD currently supports CPU execution in double precision on NumPy, JAX, and
PyTorch. Geometry, material and derivative support depend on the admitted
problem; preparation reports its limits. Older periodic factories and the
normalized general FDTD dialect remain explicit compatibility interfaces under
`wavebit.compat.legacy`. Historical tutorials and measurements label that import.

## Install

The solver release has not been published to PyPI. Install this alpha candidate from its
source checkout:

```shell
pip install .
```

Use the source checkout or distribution attached to the version you want to evaluate;
a Git installation from the repository's default branch can contain a different version.

NumPy runs both solvers. JAX and PyTorch are optional differentiation backends:

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

A binary grating on glass, using the NumPy backend:

```python
import numpy as np
from wavebit import Material, Units, rcwa

problem = rcwa.Problem(
    units=Units(length="nm"),
    stack=rcwa.Stack(
        period=(700., 700.),
        bottom=Material(epsilon=2.25),
        layers=(rcwa.Layer(thickness=300., epsilon=np.array([[1., 1., 12.25, 12.25]])),),
    ),
    excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization="s"),)),
)
numerics = rcwa.Numerics(orders=(10, 0))
result = rcwa.run(problem, numerics=numerics)
print(float(result.R.sum()), float(result.T.sum()))
print(result.transmission.order(0, 0).efficiency)
```

The diffraction arrays retain `(wavelength, order_v, order_u)` axes, including
single-wavelength solves. Reflection and transmission here are efficiencies;
absolute port power and its integration measure are recorded separately.

The corresponding FDTD entry point is demonstrated in
[`examples/fdtd/pulse_slab.py`](examples/fdtd/pulse_slab.py). Its physical domain is
kept separate from the PML cells added outside open faces. Time records use
seconds; raw Fourier fields retain their pulse-transform units rather than
claiming monochromatic power.

## Conventions

* `Material(epsilon=...)` declares relative permittivity, with shared permeability
  and supported conductive/dispersive models. Tabulated optical data are not
  automatically converted into a causal FDTD material.
* The phasor convention is `exp(-i omega t)`. Positive imaginary permittivity is
  loss. The common Cartesian frame has `+z` upward; RCWA finite layers are
  declared from bottom to top, with the bottom interface at `z=0`.
* All geometry lengths and wavelengths follow `Problem.units`. Angles are
  radians, FDTD time is seconds, and field units are E in V/m and H in A/m.
* RCWA cells are centered laterally. Diffraction observations identify their
  reference plane, medium, lateral origin, propagation and polarization frame.
  FDTD Yee samples additionally retain component-specific positions and times.
* Raw pulse spectra, normalized scattering efficiency and calibrated power are
  separate quantities. A missing or unusable incident reference cannot define
  a normalized efficiency.

The current contract is described in [`docs/design/unified-api.md`](docs/design/unified-api.md).
Compatibility examples preserve their historical conventions explicitly.

## Differentiating a solve

A prepared calculation fixes structural decisions while eligible numerical
leaves remain live. Its `contract` and validation report identify reuse limits.
Enable JAX double precision explicitly before preparing; importing Wavebit does
not change the process-wide JAX configuration.

```python
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

prepared = rcwa.prepare(problem, numerics=numerics, backend="jax")

def transmitted(thickness):
    layer = problem.stack.layers[0].replace(thickness=thickness)
    live = problem.replace(stack=problem.stack.replace(layers=(layer,)))
    return jnp.sum(prepared.run(live).T)

gradient = jax.grad(transmitted)(jnp.asarray(300.))
```

FDTD also validates fixed grid, duration, source and observation dependencies.
Geometry derivatives require explicitly reserved support where the engine
supports them. Runtime guards remain active under transformations; preparation
alone is not a convergence certificate.

## Validation

The ordinary solver is checked against references, not only against itself. These checks
cover their recorded cases; they do not certify the experimental contact auxiliary solver's
optical convergence or opening derivative:

* **Three outside RCWA solvers** — RETICOLO through an Octave bridge, plus fmmax and S4. Their
  numbers are committed as goldens and checked in CI, so the half that needs software this
  project does not ship runs once, by hand. RETICOLO agrees to ~1e-13 and is the only reference
  here that validates individual diffraction *orders* rather than totals.
* **A different method entirely** — Wavebit's own FDTD ([`docs/guide/fdtd/fdtd.md`](docs/guide/fdtd/fdtd.md)) solves
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
[`docs/reference/common/result-guards.md`](docs/reference/common/result-guards.md).

## Documentation

The prose is split by **who is reading**, not by subject — the same three-way split NumPy's
manual uses, with the decision records outside the manual the way NEPs are.

| | For a reader who | |
| --- | --- | --- |
| [`docs/guide/`](docs/guide) | needs the contract for one keyword | **The manual.** What Wavebit solves and what it is bad at, installation, a quickstart; then describing a scene, saying what the light is, choosing the numerics, near fields, autodiff, backends, and a run described as data |
| [`docs/reference/`](docs/reference) | needs a signature or a guarantee | [`conventions.md`](docs/reference/common/conventions.md) is the **normative** contract; the current solver modules expose typed physical declarations and execution contracts; compatibility references retain older keyword details |
| [`docs/`](docs) | is questioning a default | **Why it is what it is**, and what was measured to decide it — the write-ups, cited from the code they explain. [`docs/decisions/`](docs/decisions) groups them |
| [`tutorials/`](tutorials) | has not used the API yet | Start with the current [RCWA](tutorials/rcwa/00-unified-lifecycle.ipynb) or [FDTD](tutorials/fdtd/00-unified-lifecycle.ipynb) notebook. Historical lessons use explicit compatibility imports |
| [`examples/`](examples) | wants it applied | Wavebit against outside problems |
| [`studies/`](studies) | wants the raw measurement | The scripts behind the numbers in `docs/` |

All of it also builds as a site:

```shell
pip install -e '.[docs]'
sphinx-build -b html docs docs/_build/html      # then open docs/_build/html/index.html
```

Every page reads the same in both places, which is a constraint rather than a coincidence: a
link in `docs/` is a relative link to a real file, the navigation lives in `docs/_toc.yml`
rather than in directives inside the pages, and `tests/api/test_docs_contract.py` checks that every
path a document cites exists. The one thing the site has that the repository cannot show is the
API reference, which is read off the live objects at build time.

Nothing is deployed yet. `python -m wavebit.manifest` is the other machine-readable surface —
every option, refusal and constraint as one generated JSON document, for a program that needs to
know without reading the source ([`docs/reference/common/capability-manifest.md`](docs/reference/common/capability-manifest.md)).

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
