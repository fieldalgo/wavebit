# The capability manifest — what the solver says about itself, and where the line is

Written: 2026-08-11.

Targets:
- `wavebit/manifest.py` — the document
- `tests/test_manifest.py` — what keeps it true
- `wavebit/_core/engine.py` — the refusal tables it reads
- `wavebit/on_numpy/emsolver/rcwa.py` — the `structure` setter whose type dispatch it describes

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
| What is refused, and why | "PyTorch below `max_order≈5`" |
| How a `structure` selects a path | A vocabulary for describing an experiment |
| Which material cases are unsolvable | An extraction schema, a system prompt |

The test is whether the fact can be checked against the code. `_FACTORIZATION_IMPLEMENTED_BY`
can be — build a block-path solver, ask for NVF, watch it refuse. *"PyTorch wins below
`max_order≈5`"* cannot: it is a measurement on one CPU host about a workload, and the solver has no
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
| Each spelling selects the modeling type it claims | The `structure` setter's type dispatch changes, or the prose describing a spelling stops matching what it accepts |
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

## Refusals are facts, not roadmap

`materials[].supported` is the solver's own answer. `false` means a run cannot be done today,
and `why` says what would have to change — a new factorization for a patterned anisotropic
layer, a second site for `mu != 1`, a first-order solver for `eps_xz`. Publishing that is
deliberate: a consumer that cannot tell "unsupported" from "you spelled it wrong" will send
users to debug their own input.

The `z_coupled` entry is the one to watch, and it has already turned over once: it used to
describe work that existed but was not reachable — the 4ff operator written and pinned with
nothing dispatching to it. It now reads `supported: true`, and its `why` carries the two limits
that ride with it (`connecting_algo` must be `ETM`, no `mu`) — and it has already been edited
once more since, when field reconstruction landed and stopped being one of them. That is the shape
these entries are meant to take: a `true` with its edges named, rather than a `false` that
quietly becomes stale the day the capability lands.
