# Contributing to FieldWave

Thanks for your interest in FieldWave! Bug reports, questions, and pull requests are all welcome. FieldWave is currently maintained by a single developer at FieldAlgo — responses are usually quick, but please bear with the occasional delay.

## Where does a new file go?

Answer these in order and stop at the first **yes**. The point of the order is that you never have to look at what is already in a directory and guess.

| | Question | Where it goes |
|---|---|---|
| **0** | Is this going into git at all? A fragment, a probe, an idea being tried out. | **No → your scratch space, outside the repository.** Stop here. |
| 1 | Is it the implementation? | `fieldwave/` |
| 2 | Can you write the `assert` line *before* you run it? | `tests_verification/` |
| 3 | …and does that criterion come from an **outside solver** (RETICOLO, GRCWA, TORCWA)? | the external-oracle machinery — `benchmarks/interface/` plus the golden and its test |
| 4 | Would writing that `assert` mean asserting the very thing you are trying to find out? | `benchmarks/` — **and** a write-up in `docs/` |
| 5 | Is it prose someone will cite later? | `docs/` |
| 6 | Is it a lesson, read in order, for someone who has not used the API? | `tutorials/` |
| 7 | Is it FieldWave applied to an outside problem? | `examples/` |

If none of them is a yes, **the file has no home yet.** Say so in the PR rather than inventing a directory.

**Questions 2 and 4 are the pair that gets misfiled**, so they are worth restating. If you can state the criterion up front — `assert abs(R + T - 1) < 1e-9`, `assert norm(numpy - jax) < 1e-12` — it is a test. If writing the assertion would require the answer you are running it to discover — `assert jax_is_faster` — it is a study, and it is not finished until `docs/` records what the numbers decided. Equivalently: when the number moves, would you fix the *code* or the *write-up*?

Two more rules that go with the table:

- **Every directory carries a `README.md` whose first table maps each entry to the question it answers.** Not what the script does — what it exists to find out. `benchmarks/README.md` is the model.
- **Everything committed is in English**: code, comments, docs and commit messages. Your scratch space is exempt, which is part of why it is kept separate.
- **Scratch lives outside the checkout.** An ignored directory inside the repository is not safe — it is what `git clean -xdf` exists to delete. Keep it as a sibling of the checkout (this project uses `../<your-scratch-dir>`) and symlink it in as `lab` if you want it on hand; the symlink is ignored, and losing it costs one `ln -s`.

The reasoning behind all of this, the survey it was checked against, and the renames still planned (`tests_verification/` → `tests/`, `benchmarks/` → `studies/`, external oracles regrouped under `tests/references/`) are in [`docs/repository-layout.md`](docs/repository-layout.md). **The table above uses today's directory names, so it is correct as written**; if a name here disagrees with that document, the rename has not happened yet and this file wins.

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
- **Add or extend a test.** Solver changes need a numerical test against a reference value; see `tests_verification/` for the pattern.
- **Mind the conventions.** All sign and branch-cut choices live in `fieldwave/_core/conventions.py` — never introduce a local `sqrt`/`kz` branch decision elsewhere. FieldWave uses the `e^{-iωt}` time convention (`Im(n) > 0` = loss).
- **Backend neutrality.** New solver code should be written against the Array API (`fieldwave/_core/backend.py`) rather than importing NumPy/JAX/Torch directly, unless there is a documented reason.
- For anything larger than a bug fix, please open an issue first to discuss the design — it protects your time as much as the maintainer's.

## Development setup

```shell
git clone https://github.com/fieldalgo/fieldwave.git
cd fieldwave
pip install -e .[jax]      # or .[pytorch], or plain -e . for NumPy only
pytest tests_verification/ -q
```

## Licensing

FieldWave is licensed under Apache-2.0. By submitting a contribution, you agree that it is your own work and is provided under the same license (inbound = outbound). FieldWave is derived from [meent](https://github.com/kc-ml2/meent) (MIT License, © 2022 ML2); the original license notice must be preserved.
