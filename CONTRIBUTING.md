# Contributing to FieldWave

Thanks for your interest in FieldWave! Bug reports, questions, and pull requests are all welcome. FieldWave is currently maintained by a single developer at FieldAlgo — responses are usually quick, but please bear with the occasional delay.

## Where does a new file go?

Answer these in order and stop at the first **yes**. The point of the order is that you never have to look at what is already in a directory and guess.

| | Question | Where it goes |
|---|---|---|
| **0** | Is this going into git at all? A fragment, a probe, an idea being tried out. | **No → your scratch space, outside the repository.** Stop here. |
| 1 | Is it the implementation? | `fieldwave/` |
| 2 | Can you write the `assert` line *before* you run it? | `tests/` |
| 3 | …and does that criterion come from an **outside solver** (RETICOLO, GRCWA, TORCWA)? | the external-oracle machinery — `tests/references/interface/` plus the golden and its test |
| 4 | Would writing that `assert` mean asserting the very thing you are trying to find out? | `studies/` — **and** a write-up in `docs/` |
| 5 | Is it prose **about the code** — what it does, why a default is what it is, what was measured? | `docs/` |
| 6 | Is it prose **about how the project is run** — a plan, a process, a decision about the repository rather than about the physics? | Not here. It is finished work, but it is not documentation of the library. |
| 7 | Is it a lesson, read in order, for someone who has not used the API? | `tutorials/` |
| 8 | Is it FieldWave applied to an outside problem? | `examples/` |

If none of them is a yes, **the file has no home yet.** Say so in the PR rather than inventing a directory.

**Questions 5 and 6 are the other pair worth restating.** `docs/` is published: anyone who clones this repository reads it, so it holds what a user or contributor needs — measurements, and the reasoning behind a default. A record of *how the project is run* is finished work rather than scratch, but it is not part of the library, and putting it here means publishing it. Keep those alongside the checkout instead of inside it.

Because it is published, **nothing committed here may name a path outside the checkout** — not a document, not a script's default argument, not a notebook's stored output. State what a private record measured; never where it sits. [`docs/README.md`](docs/README.md) states the contract, and `tests/test_docs_contract.py` fails the build on a violation. A draft that keeps wanting to point outward is usually a question-6 file that landed in the wrong place.

**Questions 2 and 4 are the pair that gets misfiled**, so they are worth restating. If you can state the criterion up front — `assert abs(R + T - 1) < 1e-9`, `assert norm(numpy - jax) < 1e-12` — it is a test. If writing the assertion would require the answer you are running it to discover — `assert jax_is_faster` — it is a study, and it is not finished until `docs/` records what the numbers decided. Equivalently: when the number moves, would you fix the *code* or the *write-up*?

Two more rules that go with the table:

- **Every directory carries a `README.md` whose first table maps each entry to the question it answers.** Not what the script does — what it exists to find out. `studies/README.md` is the model.
- **Everything committed is in English**: code, comments, docs and commit messages. Your scratch space is exempt, which is part of why it is kept separate.
- **Scratch lives outside the checkout.** An ignored directory inside the repository is not safe — it is what `git clean -xdf` exists to delete. Keep it as a sibling of the checkout and symlink it in as `lab` if you want it on hand (`ln -s ../<your-scratch-dir> lab`); the symlink is ignored, and losing it costs one `ln -s`.

### Why the directories are named this way

Three of them are worth a sentence, because the obvious guess is wrong:

- **`studies/`, not `benchmarks/`.** In this ecosystem "benchmarks" means a continuous performance-regression suite — numpy, scipy, pandas and networkx all run `asv`. These run once, by hand, to settle a question, and end in a `docs/` write-up. Only one of them is even about speed; the others measure convergence and cost at equal accuracy.
- **`tests/`, not a qualified name.** The suite holds verification (frozen goldens) and validation (an outside solver) alike, so any qualifier would name half of it.
- **`tests/references/`** exists because external-solver comparison has a known expected answer — so it is a test, not a study — but producing the reference needs Octave and a licensed RETICOLO tree. The directory splits by what each piece needs: the bridge and generator need those, the committed numbers and the assertion need nothing and run in CI everywhere.

Two things were considered and rejected, so that they are not proposed again: a `src/` layout, which with many worktrees sharing one editable install would make each worktree silently test the main checkout rather than its own edits; and merging `studies/` into `docs/`, which only three of the write-ups could support since the rest have no script behind them.

**This file is the operative one.** If a directory name anywhere else disagrees with the table above, the table wins.

## Reporting bugs

Numerical software bugs are often subtle, so a good report makes all the difference. Please include:

- A **minimal script** that reproduces the issue (structure, wavelength, backend, Fourier order).
- The **backend** (NumPy / JAX / PyTorch) and its version, plus your Python and FieldWave versions.
- What you expected vs. what you got. For convergence or accuracy issues, a reference value and where it comes from (analytic result, another solver, a paper) is enormously helpful.

If the same case behaves differently on two backends, say so — cross-backend discrepancies are treated as high-priority bugs.

## Questions and discussions

Not sure whether something is a bug, a usage question, or a feature idea? Open an issue anyway and label it as a question. Questions about RCWA methodology (Fourier factorization choices, convergence behavior, conventions) are welcome — they often turn into documentation improvements.

## Pull requests

- **Keep PRs small and focused.** One fix or one feature per PR reviews much faster than a grab-bag.
- **Add or extend a test.** Solver changes need a numerical test against a reference value; see `tests/` for the pattern.
- **Mind the conventions.** All sign and branch-cut choices live in `fieldwave/_core/conventions.py` — never introduce a local `sqrt`/`kz` branch decision elsewhere. FieldWave uses the `e^{-iωt}` time convention (`Im(n) > 0` = loss).
- **Backend neutrality.** New solver code should be written against the Array API (`fieldwave/_core/backend.py`) rather than importing NumPy/JAX/Torch directly, unless there is a documented reason.
- For anything larger than a bug fix, please open an issue first to discuss the design — it protects your time as much as the maintainer's.

## Development setup

```shell
git clone https://github.com/fieldalgo/fieldwave.git
cd fieldwave
pip install -e .[jax]      # or .[pytorch], or plain -e . for NumPy only
pytest tests/ -q
```

## Licensing

FieldWave is licensed under Apache-2.0. By submitting a contribution, you agree that it is your own work and is provided under the same license (inbound = outbound). The third-party notices recorded in [LICENSE](LICENSE) must be preserved.
