# The capability manifest — what the solver says about itself, and where the line is

Written: 2026-08-11.

Updated 2026-09-24: manifest version **12** names the specified-incidence result
`incident` and publishes direct amplitude access through `result.r_s`, `result.r_p`,
`result.t_s`, and `result.t_p`. The field-reconstruction selector is also `'incident'`.

Updated 2026-09-21: manifest version **11** records automatic material dispersion
evaluation and the numerical axes `raster_normal_method` and `nvf_projector`;
JobSpec version **10** uses bottom-to-top layers with +z upward, retains material tables and names the
shape sampling count `shape_nvf_grid_size`. Update consumers of the old option keys
using the [keyword migration](reference/keywords.md#nvf-names-changed-on-2026-09-21).

Targets:
- `wavebit/manifest.py` — the document
- `tests/test_manifest.py` — what keeps it true
- `wavebit/_core/engine.py` — the refusal tables it reads, and the `structure` setter whose
  type dispatch it describes. Those were two entries until the per-backend copies folded into
  `_core`; the second pointed at the NumPy backend, which no longer holds that setter.

---

## TL;DR

Wavebit has five independent numerical choices — modeling path, Fourier series,
factorization rule, backend, lattice spelling — and the rules governing them are scattered
across nine documents and two refusal tables in `_core/engine.py`. Programs outside this
process need those rules: documentation, configuration UIs, an assistant that recommends
conditions. Until now the only way to get them was to read the source and retype what you
found, which is how a consumer ends up confidently telling users about a flag that was
deleted a year ago.

`wavebit.manifest` publishes them instead, as one JSON document generated from the code
and checked against it by the suite.

```bash
python -m wavebit.manifest > wavebit-manifest.json
```

**Decision: capabilities and constraints only.** The manifest says what each path implements,
what is refused and why, and which spelling of `structure` selects which path. It does not
recommend anything.

## The line, and why it is where it is

| In | Out |
| --- | --- |
| What each path implements | Which path to use |
| What is refused, and why | "PyTorch below `truncation_order≈5`" |
| How a `structure` selects a path | A vocabulary for describing an experiment |
| Which material cases are unsolvable | An extraction schema, a system prompt |

The test is whether the fact can be checked against the code. `_FACTORIZATION_IMPLEMENTED_BY`
can be — build a parametric solver, ask for `li`, watch it refuse. *"PyTorch wins below
`truncation_order≈5`"* cannot: it is a measurement on one CPU host about a workload, and the solver has no
opinion about it. A fact that cannot be checked that way does not belong in the module, because
publishing it would extend the manifest's guarantee to something the guarantee does not cover.

That line also settles a scope question that came up while this was being built. An advisor
that maps *"my pillars are round and I want to optimise their diameters"* onto a condition
set is a useful thing, but it is advice about a user's problem rather than a property of the
solver, and a library that grows a recommendation engine has stopped being a library. It
lives outside this repository and consumes the manifest. What stays here is the half only
this repository can know.

### One later admission: the results contract (2026-08-12)

The first cut carried inputs and constraints only, and drew the line at "what the solver
returns" as well. That half was admitted when the result surface was renamed for the first
release, because it passes the same test the line is made of: *can the fact be checked
against the code?* The `results` section's every claim — field names, power-vs-amplitude
kind, the order layout, the accessor, the JobSpec defaults — is held against a **real
result object** by `tests/test_manifest.py`, so a rename that misses the manifest fails the
suite. What still stays out is unchanged: recommendations, and vocabulary for describing a
user's experiment. The section is data *about* the contract; the contract itself is
[`conventions.md`](conventions.md), which stays normative.

## Why the tests are not mirror tests

`wavebit/manifest.py` reads `_core/engine.py`'s tables at import, so asserting that the two
agree would assert that a variable equals itself. `tests/test_manifest.py` checks the
manifest against **runtime behaviour** instead:

| Test | What would have to break for it to fail |
| --- | --- |
| Each spelling selects the modeling path it claims | The `structure` setter's type dispatch changes, or the prose describing a spelling stops matching what it accepts |
| An option the manifest withholds is refused | A path stops refusing what the manifest says it cannot express |
| An option the manifest offers actually builds | A path is described as implementing something it refuses |
| The skewed-cell refusals hold | `_reject_unsupported_lattice` loosens or tightens |
| Every cited source exists | A file a claim was anchored to is renamed or deleted |

The middle two are the pair that matters: they close the loop the refusal tables sit in the
middle of, so the manifest is tied to what the solver *does* rather than to what one table
*says*.

## The constraint that is stated twice

`lattice.refused_on_oblique` describes the two refusals a skewed cell carries, in prose. The
same two are also listed by name in `paths_refused_on_oblique` and
`factorizations_refused_on_oblique`, because a consumer deciding what to recommend needs a
membership test and prose is not one.

A third list, `normal_methods_refused_on_oblique`, is **empty** and kept rather than dropped —
it held three entries until 2026-08-21, when all four normal-field constructions gained the
cell metric. The key is the membership test a consumer already asks, and an empty answer is an
answer; removing it would turn "nothing is refused on this axis" into "this consumer is talking
to a document that predates the axis".

Two spellings of one fact drift, and the machine-readable spelling is the one a consumer
acts on — so a mismatch would leave the document reading correctly and behaving wrongly.
`test_the_named_and_described_skewed_refusals_agree` is what stops that.

## The one fact that had to become machine-readable (2026-08-20)

`material_quantity` says what a material number *is*: `quantity: 'permittivity'`, relative,
complex, `Im(eps) > 0` for loss, the `Material(epsilon, mu=1.0)` type it is written as, the
scalar shorthand, the two conversion helpers, and a `retired` map from every old spelling to
its replacement.

It is a dict rather than a sentence in `summary` for one reason: the answer **changed**. Every
material slot took a refractive index until this date. A consumer that had learned the old
answer and kept generating scenes against it would keep emitting numbers that are still
*legal* — `3.5` is a perfectly good permittivity — so nothing raises, nothing warns, and
energy conserves. Prose cannot be tested for; membership can, so a consumer can assert on the
key and refuse the run instead of quietly solving a different structure.

`retired` is *derived* from the same table the solver's own refusals are generated from
(`_core/domain.RETIRED_MATERIAL_NAMES`) rather than restated beside it, which is what keeps
the promise true of the list rather than of a copy of it. Same reasoning as "the constraint
that is stated twice" above, applied before the second spelling had a chance to drift.

`MANIFEST_VERSION` went `4` -> `5` for this, and the bump is the point: a rename alone would
not have earned one, since a consumer reading unknown keys degrades gracefully. A changed
*quantity* under a recognisable shape does not.

`6` (2026-08-20) is the same rule applied one level down, not an exception to it. The
pre-release rename set moved `results.incidences.fields` from `('res', 'res_te_inc',
'res_tm_inc')` to `('given', 's_inc', 'p_inc')`. Those are not key names a consumer can skip —
they are attribute names it indexes a result object with, published under a key that still
looks exactly the same. The contents kept their shape and stopped resolving, which is the
`material_quantity` failure in miniature.

## Two paths, after publishing three (2026-09-03)

`paths` carried three entries — `raster`, `smooth`, `boundary` — and the last two were one
path published twice. Every machine-readable key on them was identical, they take the same
`structure` spelling, and the choice between them is made **per layer** from the shapes'
coordinates, so one structure can occupy both at once. Two sibling entries cannot say that,
and the suite could not see the difference either: both tests that distinguished them
compared `modeling_type`, which was `2` for both.

They are now `parametric` with an `overlap` property, and `raster` is unchanged. The property
is held against runtime like every other claim here — that the overlapping spelling selects
the same path, that a mixed structure resolves one layer and not the other, and that the
option battery the `boundary` entry used to carry still runs against an overlapping layer.

`smooth` also collided with `NormalMethod.SMOOTH`, which the published JSON showed plainly:
`paths[0].raster_normal_method` (the **raster**'s options) contained `'smooth'` while `paths[1].path`
*was* `'smooth'`. A caller on the overlapping route asking for `raster_normal_method='smooth'` was
told "the smooth shape path builds its normal field from the geometry itself" — one word
meaning two things in one sentence, naming a path they were not on.

**The version bump, and which argument carries it.** `MANIFEST_VERSION` went `8` -> `9`, but
not because a value a consumer branches on changed. `path` is a label, not something fed back
to the solver — there is no `path=` keyword — so a branch that stops matching falls through
rather than solving the wrong scene. That is the "reading unknown keys degrades gracefully"
case, and on its own it would not earn a bump.

What earns it is `shape_coordinates.applies_to`, which held `('smooth', 'boundary')`. That is
a **membership test keyed on these names**, and what it gates is a coordinate convention. A
consumer with `'smooth'` hardcoded in its own source now gets `False`, concludes absolute
lengths and a cell-centre origin do not apply, and emits fractions or a corner-relative
centre. The scene it builds solves and conserves energy — the `'8'` failure exactly, on the
convention `'8'` was about. The lesson generalises: when path names are used as keys
*elsewhere in the document*, renaming them is not a rename.

## Refusals are facts, not roadmap

`materials[].supported` is the solver's own answer. `false` means a run cannot be done today,
and `why` says what would have to change — a new factorization for a patterned anisotropic
layer, a second site for `mu != 1`, a first-order solver for `eps_xz`. Publishing that is
deliberate: a consumer that cannot tell "unsupported" from "you spelled it wrong" will send
users to debug their own input.

The `z_coupled` entry is the one to watch, and it has already turned over once: it used to
describe work that existed but was not reachable — the 4ff operator written and pinned with
nothing dispatching to it. It now reads `supported: true`, and its `why` carries the two limits
that ride with it (`layer_algorithm` must be `ETM`, no `mu`) — and it has already been edited
once more since, when field reconstruction landed and stopped being one of them. That is the shape
these entries are meant to take: a `true` with its edges named, rather than a `false` that
quietly becomes stale the day the capability lands.
