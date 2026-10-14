# Repository tools

These tools maintain the checkout; they are not part of the installed package.

- `run_verification.py` records quick/full/research pytest profiles and resumes only identical inputs; see [recorded verification](../docs/development/verification.md).
- `run_examples_smoke.py` runs the executable examples from any working directory.
- `run_tutorials_smoke.py` executes the current RCWA/FDTD notebook code cells in fresh interpreters.
- `run_distribution_smoke.py` builds and validates an isolated installed distribution.
- `repository_paths.py` resolves the historical identities recorded in `relocations.json`.

The relocation ledger preserves the exact original Git blob, file mode, and SHA-256
for reference data and archived experiment records. It records moves, not new runs.
Historical records keep their embedded paths and metadata; maintained instruments
use the resolver when following those paths. The ledger belongs with the checkout.

Write new experiment outputs under `artifacts/` and pass an explicit output path
when replaying an instrument that would replace an existing record. The resolver
places unrecorded historical `studies/results/` destinations under `artifacts/studies/`.
That includes the old names of the records archived outside the repository on 2026-10-14,
whose ledger rows were removed with them. The test goldens and oracles under `tests/references/` are not in the
ledger either (since 2026-10-14): they are regenerated and extended on purpose, so the
byte pin would fail at every legitimate change.
Only review and add a result to a question's `records/` directory deliberately.
JSON, CSV, logs, and images are not globally ignored formats.
