# A run as data

`wavebit.rcwa()` builds a *solver object* from Python values: arrays, enums, floats. That is the
right shape for a script, and the wrong shape for anything that has to cross a process, a queue
or a network. `JobSpec` is the other shape:

```
JobSpec (plain data)  --->  run(spec)  --->  Result
```

It is JSON-round-trippable and fully determines a run, which buys three things a solver object
cannot:

* **A contract.** One schema is shared by every caller — a script, a distributed worker, an HTTP
  client — so they cannot drift apart.
* **Reproducibility.** `JobSpec.key()` hashes the normalised spec, so an identical run is
  identifiable as identical without re-solving it.
* **Safety.** Accepting a declarative spec is not accepting arbitrary Python. A service that
  takes specs is not executing its callers' code.

## The shape

Four parts. The physical scene reuses the same `Stack` / `Incidence` objects
[`scene.md`](scene.md) describes; `numerics` holds the discretisation choices, which are not
part of the scene; `sweep` declares what varies.

| Field | Holds |
| --- | --- |
| `stack` | `layers` (each a `Layer(thickness, structure, mu)`), `top`, `bot`, `period`, `lattice` |
| `incidence` | `wavelength`, `theta`, `phi`, `kx`, `ky`, `tetm`, `psi`, `chi`, `incidence_side`, `polarization` |
| `numerics` | `backend`, `truncation_order`, `fourier_coefficients`, `factorization`, `raster_normal_method`, `nvf_projector`, `li_nesting`, `complex_dtype`, `layer_algorithm`, `shape_nvf_grid_size`, `answers` |
| `sweep` | `axes` — `{field: [values]}` after resolution |

`solver` (`'rcwa'`) and `version` complete it.

The current schema version is **10**. Version 7 used `normal_method`, `normal_grid`
and `projector` inside `numerics`. Rename those keys to `raster_normal_method`,
`shape_nvf_grid_size` and `nvf_projector`, respectively, then apply the bottom-to-top coordinate migration below before setting `version='10'`.
The NVF option values themselves are unchanged.

Two differences from the constructor are deliberate. `numerics.complex_dtype` is the **bit
width** (`128` / `64`) rather than a dtype object, so the spec stays backend-neutral. And
`answers` lives in `numerics` rather than in `incidence`, because it describes the job and not
the light.

The enum-typed `numerics` fields (`factorization`, `li_nesting` and the rest) take the enum or
its name in any case, as in `Numerics(li_nesting='symmetric')`. `backend` takes the enum or its
name exactly as `wavebit.rcwa` does -- `'numpy'`, `'jax'`, `'torch'`, and no integer code.
Construction normalises them exactly as `Numerics.from_dict` does, and so does a later
assignment (`numerics.li_nesting = 'symmetric'`), so a spec built in Python, the same spec
edited after it was built, and the same spec parsed from JSON are equal and share one `key()`.
A name that is no member is refused at construction or assignment, with the message the JSON
door gives, and an assignment refused leaves the field as it was.

## Running one

```python
import numpy as np, wavebit
from wavebit import JobSpec, Numerics, Sweep, Layer, Stack, Incidence, run

eps_hi = 12.25
spec = JobSpec(
    stack=Stack(layers=[Layer(thickness=300.,
                              structure=np.array([[1.0, 1.0, eps_hi, eps_hi]]))],
                top=1.0, bot=2.25, period=(700., 700.)),
    incidence=Incidence(wavelength=900., theta=0., phi=0., tetm='te'),
    numerics=Numerics(truncation_order=(10, 0)),
    sweep=Sweep.from_dict({'wavelength': {'linspace': [800., 1000., 3]}}),
)

for point in spec.expand():
    result = run(point)
    print(point.incidence.wavelength, result.R.sum(), result.T.sum())
```

```
800.0   0.460323   0.539677
900.0   0.205332   0.794668
1000.0  0.171514   0.828486
```

A raster layer must be an **array**. A nested `list` in `Layer.structure` is read as the
analytic spelling and refused with an explanation, which is the right refusal and a surprising
one if you meant a grid.

## Sweeps

Declared *inside* the spec, so "501 wavelengths" travels as one small object rather than 501
near-identical ones. `expand()` turns that back into one sweep-free spec per grid point.

```python
Sweep.from_dict({
    'wavelength': {'linspace': [800., 1000., 3]},   # start, stop, num
    'theta':      [0.0, 0.1, 0.2],                  # a bare list is shorthand for {'values': [...]}
})
```

Seven fields are sweepable — `wavelength`, `theta`, `phi`, `psi`, `chi`, `kx`, `ky` — and anything else is
refused by name. Multiple axes form the full **Cartesian product**, an outer product rather
than a zip, because that is what a parameter sweep means; `len(spec)` is the number of points.

Distributing those points across processes or nodes is ordinary work — they are independent
solves — and works identically on all three backends.

## Serializing

```python
text = spec.to_json(indent=2)
back = JobSpec.from_json(text)
assert back.key() == spec.key()          # normalised hash, so a round trip is identifiable
```

`key()` hashes the normalised spec rather than the text, so formatting does not move it. That
is what makes it usable as a cache key.

## What a document must say, and what is refused

The schema is versioned, and the gate is strict on purpose:

| Written | Result |
| --- | --- |
| no `version` | `SpecError` — a parsed document has to say which schema it is written in. Defaulting the absence to "current" would make the gate a no-op for exactly the documents most likely to be stale |
| `version: '2'` | `SpecError` — this build understands version `'10'` |
| `outputs: [...]` | `SpecError` — removed in version 4, and refused by name rather than as an unknown field |
| any unknown field | `SpecError`, naming the nearest field (`thet` -> `theta`) and listing what is allowed |
| `stack.n_top: 1.5` | `SpecError` *without* a nearest field: a name spelled like a refractive index (`n_top`, `ntop`, `n_sub`, `n1`, `RI_top`, `*_index`, `refractive*`) or a permeability (`mu_*`) is never pointed at a slot that holds another quantity, since `top` would take the `1.5` as a permittivity and solve. It is told instead that the material slots are relative permittivity (eps = n**2) or a `Material`. An `n` reads as an index only alone, numbered, or beside a name a medium goes by (`top`, `sub`, `bg`, ...) or a part of an index (`eff`, `real`): `n_orders` or `n_theta` is a count, and keeps its nearest field |
| a key that is not a string | `SpecError` — field names are strings |

Building a spec in Python is the unceremonious path: `JobSpec(...)` is versioned by the build
it is constructed in, because a spec constructed against these classes cannot be stale. Only a
*parsed* document has to declare itself.

The objects are held to the same vocabulary after construction. Assigning a name that is not a
field — `spec.incidence.thet = 0.2`, `spec.numerics.nesting = 'symmetric'` — raises
`AttributeError` with the same near-match hint, exactly as the solver does for
`solver.thicknes = ...`, and with the same refusal to guess across quantities
(`stack.n_top = 1.5` is not sent to `top`). Storing it would be silent: the run reads the fields it knows, so it
would answer for the value you meant to change. A name starting with an underscore is yours to
use, and the fields themselves assign as before.

The version bumps whenever the meaning of a value moves, not only when a key does. Version 3 is
the material convention: the keys moved (`eps` to `structure`, `n_top` / `n_bot` to `top` /
`bot`) but so did the **quantity**, and a version-2 document whose keys are renamed and whose
numbers are kept describes a different structure that solves and conserves energy. That is why
it is refused rather than migrated by guess.

A `Material` has a JSON form — `{"epsilon": e}`, plus `"mu"` when one is named — so it survives
the round trip. A bare number stays a bare number, since respelling it would change the `key()`.

A `TabulatedMaterial` is stored with its full table in a `dispersion` object.
Wavelength sweeps reevaluate it at each point; numerical `Material` values stay
constant. The table is included in `key()`. See [Materials](materials.md).

## Telling a consumer what is possible

A spec says what to run; the **capability manifest** says what may be asked for. It is a
generated JSON document covering which path implements which option, what a skewed cell
refuses, and which material cases are unsolvable:

```shell
python -m wavebit.manifest > wavebit-manifest.json
```

It carries `material_quantity` as a machine-readable key, so a generator can *test* which
quantity the materials are in rather than parse prose for it — and the document's version
number moves whenever that answer does, which is what lets a consumer that only knows the old
answer refuse instead of emitting legal, energy-conserving, wrong scenes.

**Capabilities and constraints only** — the manifest never recommends anything. See
[`../capability-manifest.md`](../capability-manifest.md).

Version **10** uses bottom-to-top `stack.layers` and upward-positive height from the bottom
interface. Reverse older layer lists and review z/tensor conventions before changing the
version. See [coordinate migration](../conventions.md#updating-a-pre-release-downward-positive-scene).
