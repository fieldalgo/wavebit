# Repository layout — the target structure, and the rule that keeps it

Written: 2026-08-09. **Proposal. No tracked file has been moved**, and none moves until the
concurrent sessions land — that is a decision, not an oversight. The one thing already done is
`lab/`, which is untracked and so cost the repository nothing; it is marked **Done** below.

Counts were measured against `develop` at `ae07ffa` and this document is based on `3c73d3d`;
`develop` moves under concurrent work, so treat them as accurate to within a few files rather
than exact. Where a count matters to a decision it is stated with its commit.

This is designed from what the project contains today, not derived from what the tree happens to
look like. Where the current layout already matches, that is a result, not an assumption.

Targets:
- `.github/workflows/tests.yml`, `setup.py`, `.gitattributes` — the packaging and CI files the
  structure has to keep true
- `tests_verification/gen_goldens_reticolo.py`, `benchmarks/interface/` — the external-oracle
  machinery, currently split across two directories
- `examples/electric-field-fno/`, `examples/metasurface-design/` — 341 files of vendored
  third-party ML code, 69% of the repository
- the root layout of **104 well-known repositories**, surveyed 2026-08-09 — so the rule is
  checked against the field, not only against this project

---

## The problem being solved

An agent filed a measurement study in `QA/` when it belonged in `benchmarks/`. A human caught
it by asking "is that structure right?" Nothing in the repository could have.

The failure is not that a directory was wrong. It is that **the rule was only recoverable by
reading what each directory already contained and inferring backwards.** A layout whose rule has
to be reconstructed from precedent will keep being violated — by agents, and by anyone new.

### It happened again while this document was being written

A second session added `benchmarks/benchmark_oblique_lattice.py` — a study by the definition
below: run once by hand, ending in `docs/oblique-lattice.md`. It landed in `benchmarks/` for the
same reason the first one landed in `QA/`: the author read `benchmarks/README.md`, judged it the
closest model, and **inferred backwards from precedent.** A human caught it again, with the same
shape of question — *"isn't there no benchmarks folder?"*

The second incident is the more useful one, because by then **the rule already existed and said
exactly where the file should go.** It did not help. It was sitting uncommitted in one worktree
where no other session could read it.

So the deliverable is not the structure, and not even the written rule. It is:

1. a rule applicable **without reading precedent**, and
2. that rule **reachable from inside the repository** by anyone about to add a file.

Point 2 is what both incidents actually failed. It is why `CONTRIBUTING.md` is move 1 rather
than a tidy-up at the end, why the rule must land on `develop` rather than live in a branch, and
why something every session already opens — the root `README.md`, or a `CLAUDE.md` if this repo
grows one — has to point at it. A rule nobody can reach is indistinguishable from no rule.

---

## What actually has to be organized

Start from the artifacts, not the directories. FieldWave produces eight distinct kinds of thing,
and they separate cleanly on three questions: **who reads it**, **what makes it wrong**, and
**can CI run it**.

| Artifact | Read by | Goes stale when | CI can run it? |
|---|---|---|---|
| the implementation | anyone importing the package | — | yes |
| assertions with a statable criterion | CI itself | the code changes | it *is* the gate |
| frozen reference data | the assertions | regenerated on purpose | it is data |
| external-solver bridges | whoever has Octave/MATLAB | the external tool moves | **no** |
| one-shot measurement studies | "which backend should I use?" | hardware and versions drift | no, and pointless |
| decision records | future maintainers; `_core/` cites them by path | a decision is revisited | no, it is prose |
| ordered lessons | someone who has not used the API | the API drifts | should |
| applications | someone with a similar problem | the API drifts | should |

Two rows earn their own home for reasons that are easy to miss:

- **External-solver bridges** are neither tests nor studies. Their expected answer is known
  (agreement to ~1e-13), so they are not open questions — but they need Octave and a licensed
  RETICOLO tree, so CI cannot run them. Anything that ignores this row ends up scattering it,
  which is what happened.
- **Frozen reference data** is 2.7 MB of `.npz`. It is not source, it changes only when
  deliberately regenerated, and it is worthless without the generator that produced it. Data and
  generator belong together.

The bottom two rows share a failure mode — **API drift** — and nothing currently catches it.
That is a structural fact, not an accident, and it drives one of the open decisions below.

---

## The target structure

```
fieldwave/              the implementation
tests/                  everything whose criterion can be stated up front; CI gates it all
    data/               frozen goldens + the generators that produce them
    references/<case>/   external-solver validation: bridge, recorded output, assertion
studies/                no statable criterion yet -> numbers -> a docs/ write-up
docs/                   prose that gets cited, including by code comments
tutorials/              ordered lessons, notebooks
examples/               FieldWave applied to outside problems
images/                 README assets

lab -> ../<your-scratch-dir>   scratch and R&D. outside the repo; a symlink, never committed
```

Root: `README.md`, `CONTRIBUTING.md`, `pyproject.toml`, `LICENSE`, `LICENSE.meent`,
`CITATION.bib`, `environment.yml`.

Eight tracked directories. The survey's mean is 9.9, so this is neither unusually flat nor
unusually deep.

### The rule

Route a new file by the **first yes**. Stop there.

| | Question | Destination |
|---|---|---|
| **0** | **Is this going into git at all?** — a fragment, a probe, an idea being tried out | **no → `lab/`.** Stop; the rest of the table is about committed files |
| 1 | Is it the implementation? | `fieldwave/` |
| 2 | **Can you write the `assert` line before you run it?** | `tests/` — and if the criterion comes from an *outside solver*, `tests/references/<case>/` |
| 3 | **Would writing that `assert` mean asserting the thing you are trying to find out?** | `studies/` — **and** a `docs/` write-up |
| 4 | Is it prose someone will cite later? | `docs/` |
| 5 | Is it a lesson, read in order, for someone who has not used the API? | `tutorials/` |
| 6 | Is it FieldWave applied to an outside problem? | `examples/` |

If 0 is a no and none of 1–6 is a yes, **the file has no home yet.** Say so in the PR instead of
inventing a directory. Eight is the budget.

Two asymmetries the table cannot show:

- **Every study owes a write-up; not every write-up owes a study.** Step 3 produces a pair —
  a study whose result is not written down was a wasted run. Step 4 stands alone; this document
  is a step-4 file with no study behind it.
- **`examples/` is the only destination CI does not keep green, which is exactly why things
  drift into it.** That is a hazard, not a convenience — see the open decisions.

### Every directory carries a README whose first table maps each entry to the question it answers

Not "what this script does" — that is recoverable by reading the script. **What question it
exists to answer**, which is not, and which is the thing that decides whether a new file belongs
beside it.

`benchmarks/README.md` already does this and is the model:

| Script | Question it answers |
|---|---|
| `benchmark_backends.py` | Which backend, and whether to freeze the geometry, for a *differentiated* run. … Write-up: `../docs/backend-choice.md`. |

Coverage today: `benchmarks/` yes (the model), `tutorials/` yes, `examples/` yes but without the
question column, `docs/` none, `tests_verification/` none, `QA/` none.

**A README says what is there; it cannot say where a new file goes.** So the routing rule lives
at the root, in `CONTRIBUTING.md`. That is the half the existing precedent was missing.

---

## Why each boundary is where it is

### `lab/` — where an idea lives before it has a home

Every question in the table above presupposes that the file is going into the repository. Most
work does not start there. A fragment written to check one thing, a probe, a half-formed idea,
a plot made once to see a shape — none of these is an implementation, a test, a study or a
document, and forcing them through the table produces "no home yet", which is wrong advice.

So the table starts at 0, and the answer is `lab/`. **Done 2026-08-09:** the directory formerly
called `dev/` now lives *outside* the repository as `../<your-scratch-dir>`, with `lab` inside the
checkout as a symlink to it. **22 MB, 357 files:**

| What is in there | Count |
|---|---:|
| figures, regenerated constantly | 89 `.png` |
| probes, experiment scripts and notebooks | 70 `.py`, 7 `.ipynb` |
| working notes, mostly Korean | 31 `.md` |
| cached results and derived data | `.npz`, `.json`, `.jsonl`, `.log`, `.txt` |
| build residue | 16 `.pyc` |
| sub-projects, one with its own `.git` | `research/`, `s4-oracle/`, `s4-boundary/`, `boundary/`, `figs/` |

**Out of version control is the design, not an oversight.** Two reasons, and the second settles
it:

- Committed prose here is English. `lab/` is where thinking happens in whichever language is
  fastest — most of those notes are Korean. A scratch directory under review pressure stops
  being a scratch directory.
- **It is not a code directory.** It is a workspace: reference material, figures, cached
  sweeps, downloaded data, scratch copies. Git is the wrong tool for all of that — 89 PNGs that
  change every time a plot is retouched, `.npz` caches that are derived, a nested repository
  that would become a submodule by accident. Versioning it would cost binary churn forever and
  buy nothing, because none of it is the source of truth for anything.

**Why it is outside the repository rather than merely ignored.** An ignored directory is not
safe — it is exactly what `git clean -xdf` exists to delete, and a dry run confirmed it: 97
entries, including `lab/research`. With 16 worktrees driven by agents, that command will be run.
Two mitigations were measured:

| arrangement | `git clean -xdf` | `git clean -xdff` |
|---|---|---|
| plain ignored directory | **deletes it** | deletes it |
| ignored, with a nested `.git` (even with zero commits) | skips it | **deletes it** |
| outside the repo, reached by a symlink | removes the *symlink*; contents untouched | same |

Only the third is safe against everything, and it costs nothing: `ln -s ../<your-scratch-dir> lab`
restores the link if it is ever cleaned. It also matches how this project already works —
`<a path outside this repository>` holds `docs/` (paper drafts, plans, backlog) beside the checkout, and
already uses symlinks for build trees.

**One gotcha, which cost a test to find.** The ignore pattern must be `lab`, **not** `lab/`.
A trailing slash matches directories only, and `lab` is now a symlink — with `lab/` it shows up
as untracked in every `git status`. Until the `.gitignore` change lands it is carried in
`.git/info/exclude`, which sits in the common git directory and so covers all 16 worktrees
without touching a tracked file.

**Why `lab/` and not `dev/` or `research/`.** There is no convention to copy — of 60 `.gitignore`
files fetched from the survey repositories, only **4** ignore a personal workspace at all
(`tmp` 3, `dev` 3, `temp` 1, `local` 1). So this is decided locally, on two concrete grounds:

- **`dev` collides with refs in this repository.** `origin` and `upstream` still carry
  `DEV/main`, `DEV/examples`, `DEV/tutorials`, `DEV/vector` and `DEV_QA_2D`; on a
  case-insensitive filesystem those share a path with a `dev` ref. And with a `dev/` directory
  present, `git log dev` is ambiguous — *"both revision and filename"*. This is the same
  collision that already pushed the trunk's name to `develop`; it applies to the directory too.
  `dev` is also ambiguous in the wider world, where it usually means *committed* developer
  tooling — exactly the wrong signal for a directory that is never committed.
- **`research/` is already taken, one level down, and earning its keep.**
  `research/20240909_numpy_vs_reticolo/` holds a README, a notebook, figures and a script — a
  dated, organised study. Naming the outer directory `research/` would either nest it in itself
  or cost that distinction. The outer layer is the messy one: 89 loose PNGs, `.pyc`, drafts,
  one-off probes. Calling all of that "research" oversells it.

`lab/` fits the two layers as they already are — a lab is where the experiments run *and* where
the mess accumulates, and organised work sits in `lab/research/` inside it. It is also three
characters, so nothing gets more expensive to type.

**Nothing is meant to stay there.** The graduation path is already being walked:

| started in `lab/` | ended up in |
|---|---|
| `rcwa_proto.py` — proving the single-source core was feasible | `fieldwave/_core/` + `tests/` |
| `NVF_FINDINGS.md` | `docs/nvf-upsampling.md` |
| `draft-README-describing-a-structure.md` | a `README.md` section |

The trigger is the same rule read forwards: **when a fragment produces a number somebody will
act on, it becomes a study plus a write-up; when it becomes behaviour, it becomes code plus a
test.** Until then it stays put.

**One consequence to know about.** `lab/` is per-checkout: of 17, only 3 have one, and they cannot see each other. The main checkout's is the
real one. Anything working inside a worktree — an agent, most often — will not find
`lab/DESIGN.md` or a prototype that a note refers to, and should ask rather than assume it was
deleted. That is a fact to know, not a problem to fix; the graduation path above is what moves
something out of `lab/` when it needs to be seen.

### `tests/` vs `studies/` — the boundary that was violated

"Do you already know the answer?" is the wrong phrasing, because it is subjective: a newcomer
knows nothing and an expert knows everything, and a rule whose destination depends on who is
asking is not a rule. Two operational forms that do not have that problem:

**Write the `assert` line first.**

| | Can you write it? | Example |
|---|---|---|
| test | **yes** — the criterion comes from physics or from a reference | `assert abs(R + T - 1) < 1e-9`, `assert norm(numpy - jax) < 1e-12`, `assert matches_reticolo(1e-13)` |
| study | **no** — writing it would require the thing you are trying to find out | `assert jax_is_faster` — that *is* the question |

The obvious objection is "I don't know whether my test will pass either." Not knowing whether
**the code is right** is different from not knowing **what right means**. A test states the
criterion and the run checks the code. A study has no criterion to state yet.

**Or, equivalently: when the number moves, what do you fix?**

| the number moved | you fix | so it is a |
|---|---|---|
| `R + T` is no longer 1 | the **code** | test |
| JAX is now slower than PyTorch | the **write-up** — the hardware or a version changed | study |

This is the form that does not depend on who is asking. It also explains why a study must not be
turned into a test by picking a threshold: `assert jax_is_faster` inside
`benchmark_backends.py` would become a *lie* the day the machine changes, because what actually
needed updating was prose.

**The two are connected, and that is not a contradiction.** A study finds the number, the
write-up records the decision, and a test pins it:

> study → `docs/` write-up → (sometimes) a test that holds the decision in place

`docs/nvf-upsampling.md` went the whole way: the study found that the fine normal-field grid was
not paying for itself, a constant changed, and tests now hold it. `docs/backend-choice.md`
stopped at the write-up, because its conclusion is advice to a human choosing a backend — there
is nothing to pin.

**Checked against the case that went wrong.** Could the truncation study have been written as an
assertion? `assert disc_cost < rect_cost` — no: that *was* the question, and it turned out
**false** in one of the three structures measured (0.78x on an axis-aligned square). So it is a
study, and the rule routes it out of `QA/` without being told.

### Why `studies/` and not `benchmarks/`

In this ecosystem **"benchmarks" means continuous performance-regression tracking.** The survey
is unambiguous: numpy, scipy, pandas, xarray and networkx all use `asv`; Cirq uses
`pytest-benchmark`. I read six of those READMEs and every one explains *how to run the suite*.

That is a different genre from what this directory holds. These run **once**, by hand, to settle
a question, and end in a written decision — `docs/backend-choice.md` ends in a recommendation,
`docs/fourier-truncation-shape.md` ends in *"Decision: not implemented."* `asv` has no
equivalent, and someone arriving from numpy will look for a dashboard that does not exist.

The name is also only about a third accurate: of the three scripts, `benchmark_backends.py` is a
timing study, `benchmark_analytic.py` measures **convergence**, and the truncation study measures
**cost at equal accuracy**.

Cost of the rename: **zero code changes.** Nothing outside the directory imports it — the only
references are `reti_1D/1Dc/2D.py` importing `interface/`, and those move to `tests/references/`
anyway.

### Where the external oracles go, and why they are not studies

RETICOLO, GRCWA and TORCWA comparisons have a known expected answer, so rule 2 applies — but
driving RETICOLO needs Octave and a licensed tree. The resolution is to split the parts by what
they need:

```
tests/references/<case>/
    bridge.py        drives the external solver      needs Octave/MATLAB
    recorded/        its output, committed           needs nothing
    test_*.py        the assertion                   needs nothing; runs in CI everywhere
```

**This pattern is already proven here.** `tests_verification/test_reticolo_golden.py` works
exactly this way — its docstring notes that the evidence used to live in a gitignored notebook
and in `benchmarks/`, "which has no assertions and does not run in CI, so the strongest result
the project had was also the one nothing checked." The only problem is that its three pieces sit
in two directories and the generator bridges the gap with
`sys.path.insert(0, str(HERE.parent / 'benchmarks'))`.

One property to build in from the start: **the test should discover cases rather than name
them**, so adding an oracle case is adding a directory. RETICOLO is the only one wired up today,
but the GRCWA and TORCWA bridges already exist.

### `tutorials/` vs `docs/` — audience, not format

> **`tutorials/` is for a reader who does not yet know the API. `docs/` is for a reader who
> already does and needs to know what was measured, or why a default is what it is.**

The tell: a tutorial is *outgrown* — you read it once. A doc is *cited* — `_core/factorization.py`
links to `docs/nvf-upsampling.md` from a comment, and it is why a constant has the value it has.

Format is the wrong discriminator and will eventually mislead. A notebook whose purpose is to
record a measurement is a doc stored in JSON: it will not diff, will not review, and will carry
stale outputs.

### One `docs/`, not two — and where the real debt is

All six files in `docs/` are the same genre: measurement and decision records. None is a user
guide. The obvious from-scratch move is to split `docs/guide/` from `docs/decisions/` — **and it
is wrong at this size.** Six files do not justify two trees; structure should match the volume it
organizes, and over-structuring early is a real cost paid by everyone who has to navigate it.

The structural debt is elsewhere: **`README.md` is 389 lines and 15 sections**, carrying install,
quickstart, backends, modeling, conventions, autodiff, validation and migration — the entire user
guide. When that is split out, `docs/guide/` and `docs/decisions/` become the right shape.
Not yet.

### `src/` — considered and rejected

A `src/fieldwave/` layout was proposed here and then dropped. Recording why, because the
argument for it is superficially strong and will be made again.

**The case for it.** In a flat layout the working directory is on `sys.path`, so the source tree
shadows the installed package whenever the cwd is a checkout. Measured, same command, two
locations:

```
$ cd <worktree>  && python -c "import fieldwave; print(fieldwave.__file__)"
   .../worktrees/amazing-vaughan-573753/fieldwave/__init__.py     <- this worktree
$ cd /tmp        && python -c "import fieldwave; print(fieldwave.__file__)"
   .../fieldalgo/fieldwave/fieldwave/__init__.py                  <- the main checkout
```

**The case against it, which decides it.** That shadowing is *load-bearing* for this project's
workflow. There are **16 worktrees sharing one environment**, and one editable install, pointing
at the main checkout. Today `cd <worktree> && pytest` imports **that worktree's** code — verified
— which is exactly what someone working in it wants. Under `src/`, the repo root would no longer
contain an importable `fieldwave/`, so the same command would import the *installed* package:
**every worktree would silently test the main checkout instead of its own edits.**

That is a worse failure than the one being fixed — inconsistent-but-usually-right becomes
consistent-and-wrong — and it would only be avoided by giving each of the 16 worktrees its own
environment.

The supporting evidence was also weaker than it first looked. The CI `minimal` job running in
`${{ runner.temp }}` is not a workaround for a layout defect; it is the correct way to verify
that the *installed* package works, and it is what already covers the real risk (a module missing
from `packages=`, a data file missing from `package_data`). The three `sys.path.insert` calls in
`benchmark_backends.py:50`, `benchmark_analytic.py:25` and `gen_goldens_reticolo.py:33` are
ordinary handling for scripts invoked by path, where `sys.path[0]` is the script's own directory
rather than the repo root — not a symptom of the layout.

The survey agrees with dropping it: of 104 repos, 29 have a top-level `src/` but most are C/C++
trees. Inspecting 12, nine are genuine Python src-layout (flask, requests, pytest, transformers,
diffusers, pytorch-lightning, sax, scikit-image, ray) and three are not (matplotlib, meep,
emopt). In scientific Python it is 2 of 25; in photonics, essentially one.

**What the real issue is, since it is not the layout.** Anything run from *outside* a worktree
gets the main checkout, whichever worktree you believe you are working in. That is a workflow
property of one shared editable install, and the fix — if it is worth one — is per-worktree
environments, not a directory rename.

### `tests/data/` — data with its generator

2.7 MB of `.npz` goldens and six `gen_goldens_*.py` scripts. The generators are tooling, not
tests, and they change in lockstep with the data they produce. Keeping them together is what
makes "regenerate this golden" a local operation rather than a scavenger hunt.

This is also the argument against putting tests *inside* the package, which is what most of
scientific Python does — see the survey.

---

## What the field does

**Method.** Root listing of 104 well-known repositories via the GitHub contents API, 2026-08-09,
default branch, 0 fetch failures. Five groups chosen to bracket the question: **photonics/EM 19**
(meep, tidy3d, S4, ceviche, legume, grcwa, gdsfactory, fdtd, simphony, mpb, lumopt, emopt, EMpy,
nlopt …), **scientific Python 25** (numpy, scipy, pandas, scikit-learn, astropy, sympy, jax,
pytorch, xarray, numba, Cirq, qutip …), **simulation/HPC 15** (dolfinx, firedrake, LAMMPS,
OpenMM, SU2, MOOSE, deal.II, Trilinos, Kokkos …), **ML 20**, **general 25** (cpython, rust, go,
node, react, django, flask, pytest, kubernetes, git, curl, postgres …).

| Question | Result | Bearing on the design |
|---|---|---|
| Is a `QA/`-style directory a thing? | **2 of 104.** The one I opened, `chriskeraly/lumopt`, holds `*_test.py` and `__init__.py` — a pytest suite under another name. | No such directory in the target. |
| `docs` or `doc`? | `docs` 49, `doc` 35 | `docs/`. |
| A decision-record directory (`rfcs/`, `adr/`, `design/`, `notes/`)? | **2 of 104** | No convention to adopt; `docs/` holding write-ups is fine. |
| `CONTRIBUTING.md` at root? | 58 of 104 | Supports putting the routing rule there. |
| Both `examples/` and `tutorials/`? | **5 of 104**, including transformers, MOOSE, mne-python and pyro | Rare, but the adopters are large and well-run. Keep both; they are different artifacts. |
| Third-party code vendored inside `examples/`? | **0.** All 28 that vendor use a named top-level directory (`third_party` 8, `contrib` 9, `subprojects` 5, `vendor` 2, …). Sampled 6 repos' `examples/` for a nested `setup.py`: no hits. | The current arrangement is unlike anything in the sample. |
| Does a benchmark README map scripts to questions? | **None of the six I read** (numpy, scipy, pytorch, scikit-image, Cirq, networkx). All explain how to run the suite. | The question column is a local invention worth keeping, not a deviation to correct. |

### Where tests live — the one that split the field

| | root `tests/` | inside the package |
|---|---:|---:|
| **photonics / EM (19)** | **14** | 5 |
| scientific Python (25) | 6 | **19** |
| simulation / HPC (15) | 7 | 8 |
| ML (20) | 10 | 10 |
| general (25) | 16 | 9 |
| **total (104)** | **53** | **51** |

Dead even overall, and the two groups that matter here point in **opposite** directions. numpy,
pandas, astropy, networkx, xarray and numba ship `<package>/tests/` so a user can verify their
own install — genuinely attractive for a physics library.

Two things decide against it: **14 of 19 photonics/EM projects use a root test directory**, and
these tests carry 2.7 MB of goldens, which in-package tests would put in every wheel. numpy
accepts that cost; a package whose test data is 7x its source has a weaker case.

No project in 104 uses a qualified name like `tests_verification`.

---

## Distance from here to there

| # | Move | Touches | Note |
|---|---|---|---|
| 1 | Add `CONTRIBUTING.md` with the routing rule, and link it from `README.md` | 1 new file + 1 line | **First, and land it on `develop`.** Two sessions have now misfiled while the rule existed but was unreachable. |
| 2 | READMEs: add `docs/`, add `tests/`, rewrite `examples/` to the question table, one line in `tutorials/` | 4 files | Additive. |
| 3 | Fix `.gitattributes`; add `lab` to `.gitignore` — **no trailing slash**, it is a symlink | 5 lines | See below. Keep the `dev/`/`devs/` lines until every checkout has renamed. |
| 4 | Dissolve `QA/` | 6 deleted, ~3 test files, `README.md:310` | Evidence below. |
| 5 | Vendored examples out of the repository | 341 files removed | 492 → **151** tracked files. Needs the destination repos to exist. |
| 6 | `setup.py` → `pyproject.toml` | 1 file | Independent of everything else. |
| 7 | `tests_verification/` → `tests/`; goldens into `tests/data/` | 42 files, 8 textual refs | No code imports it. |
| 8 | `benchmarks/` → `studies/`; oracles into `tests/references/` | 16 files + **8 prose refs** | Zero *code* changes outside the directory; the prose refs are listed below. Do after 7. |

Moves 1–6 touch nothing any live branch has. **Moves 7–8 are top-level renames and rebase
through all 16 worktrees** — they should land together, in one quiet window, with nothing else
riding along. They touch disjoint paths, so batching costs one disruption instead of two.

If the window never comes, both can be dropped without losing the rule: they remove wrong words,
not wrong behaviour. Moves 1–6 are what make the layout actually work.

### The prose references move 8 must fix

No code imports `benchmarks/`, but eight places name it in text, and a rename that leaves them
behind produces exactly the kind of dangling pointer this document exists to prevent:

| File | Lines |
|---|---|
| `docs/backend-choice.md` | 7, 311, 312, 313 — including `benchmarks/results/backends.json` |
| `docs/analytic-modeling.md` | 124 |
| `docs/oblique-lattice.md` | 64, 242 — 242 says the script wins if script and prose disagree, so it has to keep resolving |
| `README.md` | 300 |

Two scripts — `benchmark_backends.py:50` and `benchmark_analytic.py:25`, joined by
`benchmark_oblique_lattice.py:38` — reach the repo root with
`sys.path.insert(0, dirname(dirname(abspath(__file__))))`. That survives a *rename* but not a
change of *depth*, so `studies/` must stay at the top level, not become `tests/studies/` or
similar. `tests_verification/gen_goldens_reticolo.py:34` also reaches sideways into
`benchmarks/`; that one disappears when the oracles are regrouped under `tests/references/`,
which is the point of doing 7 and 8 together.

### Evidence for dissolving `QA/`

All six scripts were **run** on `develop`, with all three backends installed:

| Script | Runs? | Where it goes |
|---|---|---|
| `rcwa_backend_consistency.py` | yes | **already asserted, both halves** — solve by `test_solve_1d_core.py::test_core_cross_backend`, field by `test_field_core.py::test_field_cross_backend`. Fold in its six option dicts as parametrization; delete. |
| `1d_pattern_in_1dc_and_2d.py` | yes | `tests/`, marked slow — it runs at `fto=[100, 0]`. |
| `fourier_analysis_methods.py` | **no — crashes** | `tests/`, after repair |
| `autodiff_raster1.py`, `autodiff_raster2.py`, `autodiff_vector.py` | yes | assertion half → `tests/`; the demo half duplicates `tutorials/02-autograd-and-optimization-*.ipynb` |

`fourier_analysis_methods.py` fails twice over. It passes `device=` to the NumPy solver, which
does not accept it (`TypeError: RCWANumpy.__init__() got an unexpected keyword argument
'device'`). Drop that kwarg and run it, and the check underneath is empty:

```
Norm of DFS-EFS:  0.0 0.0
Norm of DFS-CFS:  1.8572843505745656e-13 5.380951361661928e-13
Norm of EFS-CFS:  1.8572843505745656e-13 5.380951361661928e-13
```

`DFS-EFS` is exactly `0.0` because the script solves twice with the same setting — the last two
rows are identical to the final digit for the same reason. It reports empty as perfect
agreement.

**An unasserted script cannot fail, so it cannot tell you it has stopped testing anything — and
since nothing runs it, it cannot tell you it has stopped running either.**

### Evidence for taking the vendored trees out

| Tree | Files | Size | Upstream | License |
|---|---:|---:|---|---|
| `examples/metasurface-design/dv3/` | 240 | 1.6 MB | sheeprl | Apache-2.0 |
| `examples/electric-field-fno/neuraloperator/` | 78 | 390 KB | neuraloperator | MIT |
| `examples/metasurface-design/dqn/` | 8 | 21 KB | jLabKAIST | MIT |

Both large trees are **forks with local modifications**, which is why they were vendored rather
than pinned — a plain `pip install neuraloperator` would not carry the BatchNorm change its
README describes.

The grep cost is real but narrow, and worth stating precisely rather than alarmingly:

| term | total | from vendored |
|---|---:|---:|
| `fto`, `conv_solve`, `period` | 927 / 293 / 781 | 1 / 0 / 1 |
| `import torch` | 201 | 153 (76%) |
| `def forward` | 111 | 87 (78%) |

FieldWave's own vocabulary is clean; generic Python and ML vocabulary is not — which is exactly
what someone unfamiliar with the codebase searches first.

**The stronger argument is staleness.** `examples/electric-field-fno/data.py:14` reads
`from fieldwave.on_numpy.modeler.modeling import ...`; `fieldwave/on_numpy/` holds only an empty
`__init__.py`, `emsolver/` and `solver.py`. **This example cannot import**, and nothing noticed.

**Decision: the trees leave.** B for `neuraloperator` (pin the dependency, carry the three named
edits as a patch), A for `dv3` (240 files is a project, not a patch). `examples/` drops to 16
files — `rcwa/` (7 textbook cases), `ocd/` (4, the arXiv optimization study), `vector_*.py` (4),
two READMEs — all pure FieldWave.

`examples/ocd/` **stays**: it is maintainable FieldWave code. Reproduction material for the
published paper living elsewhere is not a licence to drop everything with `arxiv` in its name.

Removing files does not shrink `.git`; history keeps the bytes. The benefit is to the working
tree, the greps, and the reader.

### `.gitattributes`

Currently `examples/**`, `benchmarks/**`, `QA/**` and `tutorials/**` are all marked
`linguist-vendored`. Three of those four are our own code — the tag means "not ours", which is
false, and it also hides them from code search. I read the intent as suppressing GitHub's
language bar; that is legitimate, reached with the wrong tag. Once the vendored trees are gone:

```
tutorials/** linguist-documentation
```

That is the whole file.

---

## Open decisions

1. **What do we promise about `examples/`?** One of them cannot import today, and after the
   reshuffle all 16 remaining files are pure FieldWave and fast — so a CI smoke job is cheap
   enough that "we do not maintain these" is hard to justify. **Recommendation: add the smoke
   job.** The alternative — `examples/README.md` saying out loud that none of it is checked — is
   defensible. Silently shipping a broken example is not. The same question covers `tutorials/`,
   which shares the API-drift failure mode and has no coverage either.

2. **Does `CONTRIBUTING.md` belong at the root, or as a `README.md` section?** Root file:
   `README.md` is user-facing and already 389 lines, and the routing rule is for contributors.
   If the preference is a single entry point it becomes a section — the rule is unchanged.

3. **When does `docs/` split into `guide/` and `decisions/`?** Not now, at six files. The
   trigger is splitting the user guide out of `README.md`; at that point the two genres stop
   sharing a shape and the split pays for itself.

---

## Summary

> **Not going into git → `lab/`. Otherwise route by the first yes: implementation →
> `fieldwave/`; you can write the `assert` line →
> `tests/` (criterion from an outside solver → `tests/references/`); writing it would assert the
> very thing you are measuring → `studies/` plus a `docs/` write-up; prose worth citing →
> `docs/`; a lesson → `tutorials/`; an outside application → `examples/`. Every directory
> carries a README whose first table maps each entry to the question it answers. The routing
> rule itself lives in `CONTRIBUTING.md`, because a README can say what is there but not where a
> new file goes.**
