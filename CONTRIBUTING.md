# Contributing to Wavebit

Thanks for your interest in Wavebit! Bug reports, questions, and pull requests are all welcome. Wavebit is currently maintained by a single developer at FieldAlgo — responses are usually quick, but please bear with the occasional delay.

## Where does a new file go?

Answer these in order and stop at the first **yes**. The point of the order is that you never have to look at what is already in a directory and guess.

| | Question | Where it goes |
|---|---|---|
| **0** | Is this going into git at all? A fragment, a probe, an idea being tried out. | **No → your scratch space, outside the repository.** Stop here. |
| 1 | Is it the implementation? | `src/wavebit/` |
| 2 | Can you write the `assert` line *before* you run it? | `tests/` |
| 3 | …and does that criterion come from an **outside solver** (RETICOLO, GRCWA, TORCWA, Meep, FDTDX)? | the external-oracle machinery — `tests/references/interface/` plus the golden and its test |
| 4 | Would writing that `assert` mean asserting the very thing you are trying to find out? | `studies/` — **and** a write-up in `docs/` |
| 5 | Is it prose telling a user **how to use** something — what a keyword takes, what it defaults to, what it refuses? | `docs/guide/` |
| 6 | Is it prose **about the code** — why a default is what it is, what was measured to decide it? | `docs/theory/`, `docs/validation/` or `docs/design/`, according to its role |
| 7 | Is it maintained guidance for contributors or an accepted implementation decision? | `docs/development/` or `docs/design/`. Private coordination and unfinished plans stay outside the repository. |
| 8 | Is it a lesson, read in order, for someone who has not used the API? | `docs/learn/` for the course; `tutorials/rcwa/` or `tutorials/fdtd/` for runnable notebooks |
| 9 | Is it Wavebit applied to an outside problem? | `examples/` |

If none of them is a yes, **the file has no home yet.** Say so in the PR rather than inventing a directory.

**Questions 6 and 7 separate scientific explanation from contributor guidance.**
Published documentation includes measured evidence, architectural decisions, source maps and
repeatable development workflows. Private coordination notes and unfinished plans remain
outside the checkout. A design page distinguishes a proposal from implemented behavior and
links the evidence supporting its implementation checkpoint.

Because it is published, **nothing committed here may name a path outside the checkout** — not a maintained document, a script's default argument, or a new notebook's stored output. State what a private record measured; never where it sits. [`docs/README.md`](docs/README.md) states the contract, and `tests/api/test_docs_contract.py` fails the build on a violation. Exact, hash-verified historical source records have the narrow exceptions listed there; their paths are evidence, not current navigation.

**Questions 5 and 6 split what used to be one row.** `docs/guide/` is the manual: it answers *how do I use this*, for a reader who already knows the API exists and needs the contract for one keyword. `docs/` answers *why is it built this way*, for a reader who is questioning a default. The test is whether the sentence would change if a measurement came out differently — if it would, it is a `docs/` sentence. Question 5 also sits above question 8 on purpose: a manual page is looked up, a tutorial is read once and outgrown, and when a draft is both it is a tutorial.

**Questions 2 and 4 are the pair that gets misfiled**, so they are worth restating. If you can state the criterion up front — `assert abs(R + T - 1) < 1e-9`, `assert norm(numpy - jax) < 1e-12` — it is a test. If writing the assertion would require the answer you are running it to discover — `assert jax_is_faster` — it is a study, and it is not finished until `docs/` records what the numbers decided. Equivalently: when the number moves, would you fix the *code* or the *write-up*?

Two more rules that go with the table:

- **Every directory carries a `README.md` whose first table maps each entry to the question it answers.** Not what the script does — what it exists to find out. `studies/README.md` is the model.
- **English is canonical** for code, comments, documentation and commit messages. Optional Korean translations live in `docs/locales/ko/`; keep numerical examples in their canonical source files.
- **Generated outputs are local.** Build, comparison and verification commands may write portable outputs under ignored `artifacts/`. These are not committed reference evidence; promote a reviewed measurement into an explicitly documented record when needed.
- **Scratch lives outside the checkout.** An ignored directory inside the repository is not safe — it is what `git clean -xdf` exists to delete. Keep it as a sibling of the checkout and symlink it in as `lab` if you want it on hand (`ln -s ../<your-scratch-dir> lab`); the symlink is ignored, and losing it costs one `ln -s`.

### Why the directories are named this way

Three of them are worth a sentence, because the obvious guess is wrong:

- **`studies/`, not `benchmarks/`.** In this ecosystem "benchmarks" means a continuous performance-regression suite — numpy, scipy, pandas and networkx all run `asv`. These run once, by hand, to settle a question, and end in a `docs/` write-up. Only one of them is even about speed; the others measure convergence and cost at equal accuracy.
- **`tests/`, not a qualified name.** The suite holds verification (frozen goldens) and validation (an outside solver) alike, so any qualifier would name half of it.
- **`tests/references/`** exists because external-solver comparison has a known expected answer — so it is a test, not a study — but producing the reference needs Octave and a RETICOLO V10 tree from its Zenodo record (CC BY 4.0; see `tests/references/README.md`). The directory splits by what each piece needs: the bridge and generator need those, the committed numbers and the assertion need nothing and run in CI everywhere.

The package uses a `src/` layout so installed-package checks and source-tree tests
have explicit import paths. Use one editable environment per worktree; see the
[source checkout guide](docs/development/source-checkout.md) for setup and line debugging,
and the [current source map](docs/development/source-map.md) for change locations and regression checks.
`studies/` keeps executable investigations; their explanations live in `docs/`.

**This file is the operative one.** If a directory name anywhere else disagrees with the table above, the table wins.

## Reporting bugs

Suspected bugs, accuracy questions and research observations are welcome even
before a defect is confirmed. A complete reproducer or reference is helpful but
is not required. Search existing open and closed issues first; include available
evidence and mark unknown details honestly. Helpful details include:

- A **minimal script** that reproduces the issue (structure, wavelength, backend, Fourier order).
- The **backend** (NumPy / JAX / PyTorch) and its version, plus your Python and Wavebit versions.
- What you expected vs. what you got. For convergence or accuracy issues, a reference value and where it comes from (analytic result, another solver, a paper) is enormously helpful.

If the same case behaves differently on two backends, say so — cross-backend discrepancies are treated as high-priority bugs.

Use the closest **Bug report**, **Accuracy report** or **Enhancement** form.
Pin the version or public commit, disclose local changes, and distinguish main
behavior from PR-only behavior. Reporting does not relax acceptance of a fix.

## Issues are the work list

Planned work is tracked as issues too, one item per issue, so the open issues are the project's
to-do list. A label says what kind of item it is: `bug`, `accuracy` (it runs, but a number is less
accurate than it should be), `enhancement`, `performance`, `documentation` or `validation` (a
comparison or check not yet made); and which part it touches: `rcwa`, `fdtd`, `autodiff`. A pull
request that resolves an issue says `Fixes #<number>`, so landing it closes the issue.

## Questions and discussions

Not sure whether something is a bug, a usage question, or a feature idea? Open an issue anyway and label it as a question. Questions about RCWA methodology (Fourier factorization choices, convergence behavior, conventions) are welcome — they often turn into documentation improvements.

## Pull requests

`main` is the only development trunk. Open draft PRs against it, run relevant
checks in a suitable environment and mark them Ready after reviewing the change.
Documentation changes follow the same required CI as code changes. Reviewed Ready
PRs use automatic **Rebase and merge** after every required check passes.

- **Keep PRs small and focused.** One fix or one feature per PR reviews much faster than a grab-bag.
- **Add or extend a test.** Solver changes need a numerical test against a reference value; see `tests/` for the pattern.
- **Mind the conventions.** All sign and branch-cut choices live in `src/wavebit/_common/conventions.py` — never introduce a local `sqrt`/`kz` branch decision elsewhere. Wavebit uses the `e^{-iωt}` time convention (`Im(n) > 0` = loss).
- **Backend neutrality.** New solver code should be written against the Array API (`src/wavebit/_common/backend.py`) rather than importing NumPy/JAX/Torch directly, unless there is a documented reason.
- For anything larger than a bug fix, please open an issue first to discuss the design — it protects your time as much as the maintainer's.

### Branches and releases

Start new work from current `origin/main`. Preserve published and shared history;
use a new branch if replaying submitted work is necessary. Do not push directly to
main, force-push shared work, squash on landing or create merge commits. The same
PR/Rebase procedure applies to every solver and integration change.

Only conflicting PRs need a manual rebase and another validation run. There is no
PR merge queue. GitHub Rebase and merge creates new commit IDs: record the landed
main SHA. Preserve unfinished work before merged branches are deleted automatically.
Main push CI validates the combined state; investigate failures and fix or revert
through a PR before releasing that state.

When a commit lands work first written in another commit (replayed onto a new base,
rewritten, squashed before submission, or ported by another contributor), name each
original in the commit or PR body with one `Ports: <sha>` line. A PR's own commits
landed by Rebase and merge need no such line; the PR records them. Give the SHA only,
without private branch names or paths. A commit that is not an ancestor of main may
still have landed under a new ID: check `Ports:` lines, the PR record or the behavior
on main before reporting it missing.

Feature pushes and draft PRs allocate no validation runners. Ready PRs, main pushes
and explicit dispatch run the checks. Main validation retains runs in a stable
workflow/ref queue; non-main validation supersedes older runs for the same ref.
See the [verification guide](docs/development/verification.md) for exact scope,
platform limits and queue behavior.

Create `maintenance/X.Y.x` from a release tag only when that release needs a patch.
Land version changes through the same PR process. Tag `vX.Y.Z` and publish only after
the resulting main commit passes validation; `release-gate.yml` checks that exact
commit before publication. A branch update alone does not publish a release.

### Portable changes and agent entry points

Use portable path, temporary-directory and subprocess APIs with explicit text
encodings. Check OS-sensitive behavior on the supported platforms; do not weaken
numerical criteria or add unexplained skips to conceal failures. Keep Linux full
acceptance and Linux/macOS/Windows installed-distribution checks distinct.

This document and the development guides are the shared contributor contract.
`AGENTS.md` provides short repository navigation and essential commands;
`CLAUDE.md` imports it. Keep personal preferences, account settings, private paths,
session coordination and operational history outside the public repository.

## Development setup

```shell
git clone https://github.com/fieldalgo/wavebit.git
cd wavebit
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[jax]'  # or .[pytorch], or plain -e . for NumPy only
python -m pip install pytest
python -c "import wavebit; print(wavebit.__file__)"
python -m pytest tests/ -q  # default: research deselected, slow skipped
```

That runs the default set, which is also what CI runs. `--run-slow` enables slow
cases in that set; it does not select the research tier. Tests of the NVF research code -- the
experimental conforming module and the study instruments built on it -- are marked `research`
and run only when asked for, because they study that experimental code rather than the ordinary
solver and they are where the suite's time goes:

```shell
pytest -m research tests/ -q                     # the research tier alone
pytest -m "research or not research" tests/ -q   # everything
pytest tests/rcwa/test_nvf_pair_galerkin.py -q        # a file you name runs whole
```

Run the tier when you work on the research code itself -- `src/wavebit/rcwa/_engine/conforming/`,
`src/wavebit/rcwa/_engine/passive_nvf.py`, an NVF model under `studies/rcwa/` or a research test. A change to the rest
of the package lands on the default set alone. Many research tests drive the ordinary engine, so
such a change can break one; that does not hold the change back, and the test is repaired when
research work next runs the tier.

A new test of research code carries `pytestmark = pytest.mark.research`. What belongs in the
tier and what stays out of it, with the counts, is in `tests/README.md`.

### Documentation

`docs/` is the source of a Sphinx site as well as a folder of markdown, and a page has to read
the same in both -- so a link there is a relative link to a real file, or a full `github.com`
URL when the target is outside `docs/`. **Prefer plain markdown to a MyST directive** for the
same reason: a directive renders on the site and arrives as its own source text on GitHub. The
navigation lives in `docs/_toc.yml` rather than in `toctree` blocks inside the pages, which is
that rule applied to the nav itself.

```shell
pip install -e '.[docs]'
sphinx-build -b html docs docs/_build/html      # then open docs/_build/html/index.html
```

The `docs` CI job runs that with `-W`, so **any warning fails the build**. That is the check,
not a tidiness rule: `docs/reference/api.md` names objects by hand, and autodoc reports one it
cannot import as a warning -- so a renamed public symbol fails CI instead of leaving a quietly
shorter page. A `github.com` URL into this repository is checked against the tree separately, by
`tests/api/test_docs_contract.py`.

Nothing is deployed. `.github/workflows/docs-deploy.yml` is written and dormant; its header
says what has to be true before it runs.

## Licensing

Wavebit is licensed under Apache-2.0. By submitting a contribution, you agree that it is your own work and is provided under the same license (inbound = outbound). The third-party notices recorded in [LICENSE](LICENSE) must be preserved.

Before using or upgrading third-party libraries, external validation tools, or
copied source, check the exact version's official license and any relevant
transitive or bundled components. Record the source of the license, intended
use, and applicable obligations with the integration or reference instrument.
Preserve Wavebit's Apache-2.0 license and commercial use; do not introduce
terms requiring FieldAlgo code to be relicensed or proprietary source to be
disclosed. Unclear or incompatible terms require a verified alternative.

Internal validation, runtime integration, and redistribution have different
obligations. Approval for one use does not cover another. Optional installation
or a separate environment does not by itself establish compatibility. Recheck
before copying, modifying, linking, or bundling a reference tool into a product,
wheel, container, or installer, and preserve all required third-party notices.
