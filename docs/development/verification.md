# Recorded verification and safe resumption

## Repository validation policy

Open PRs as drafts; draft PRs allocate no CI runners and feature-branch pushes
do not run validation. Mark a reviewed PR Ready after relevant checks pass in
a suitable local or remote environment. The workflow runs on main, Ready PR
events and explicit manual dispatch. Automatic Rebase and merge preserves every
required check. Main runs use a stable workflow/ref queue (`queue: max`, no
cancellation); non-main runs use `queue: single` and replace older runs. This
Actions queue is separate from a PR merge queue.

Full Linux acceptance includes the default/slow partitions, examples and docs.
The independent distribution matrix builds an sdist and wheel, installs outside
the source tree and checks public NumPy RCWA/FDTD use, bundled data and plotting
on Linux, macOS and Windows with Python 3.12. A platform smoke pass does not imply
that all optional backends or the full numerical suite passed on that OS. Inspect
the exact commit's checks and artifacts before reporting validation or release
readiness. Contributor instructions live in AGENTS.md; CLAUDE.md imports them.

Run the repository tool with the Python interpreter used for development:

```shell
python tools/run_verification.py --profile quick
python tools/run_verification.py --profile full --workers 4
python tools/run_verification.py --profile research --workers 2
```

`quick` selects a small set of installed metadata/data, public API, material-unit,
import, job round-trip,
Fresnel and FDTD update checks. It is a smoke test, not a convergence claim.
`full` runs **all non-research tests, including slow tests**. `research` runs
only tests marked `research`, including their slow cases. These are explicit
marker selections; ordinary `pytest tests` continues to use the existing default
policy, which omits research and skips slow tests.

Workers are optional. The default is serial; `--workers N` requires pytest-xdist.
The runner loads its own recorder and xdist explicitly, without auto-loading
unrelated installed pytest plugins. Existing repository conftest hooks and
pytest configuration still apply. It sets unspecified BLAS thread counts to one.

The optional `--slow` filter narrows a profile without changing its research policy:
`include` (the default) keeps the profile's existing selection, `exclude` removes
slow cases, and `only` selects its slow cases. `quick --slow only` is refused.
Thus `full` still includes slow cases unless you explicitly narrow it.

`--timeout SECONDS` sets a positive per-test limit and requires pytest-timeout.
The runner loads that plugin explicitly when requested and refuses the run if it
is unavailable. `--timeout-method` accepts `thread` (the default) or `signal`.
The same limit applies to the separate serial mutation phase. No timeout is added
when the option is omitted. `--verbose` requests verbose test progress, while
`--live-output` streams the output as well as saving it to the phase log.
Collection remains quiet and file-only; streaming and verbosity apply to the
executing test phases. A thread timeout can terminate pytest before its final
JUnit XML or session summary is complete. Flushed JSONL events remain available;
the interrupted node stays unfinished and the run returns a nonzero status.
These options affect verification subprocesses, not ordinary solver execution.

Each invocation creates a new directory under `artifacts/verification/`.
`--output` may choose a different new directory; an existing directory is refused.
The tool prints the location before running tests. Inspect:

- `inputs.json`: hashes of repository source, tests, fixtures and other recorded
  inputs, Python and installed distribution versions, selected configuration,
  and a short allowlist of numerical environment controls, including `XLA_FLAGS`. Arbitrary environment
  variables, credentials and package index URLs are not recorded.
- `collected-nodeids.json`: the exact ordered profile membership.
- `receipt.json`: per-node outcomes, reused nodes, integrity checks, phase exit
  codes and elapsed time. Passed, failed, setup/teardown errors, skipped, expected
  failures, unexpected passes and unfinished nodes remain distinct.
- Each phase directory: pytest output, JUnit XML, ordered JSONL phase events,
  worker collection lists and the actual imported package origin.

Collection errors, duplicate node IDs, differing worker collections and changed
inputs are errors, not successful empty runs. A passed call is reusable only
after its teardown completes. An interrupted run retains completed events and
returns a nonzero exit status; it does not claim that unreported nodes passed.

## Queued main validation

Each `main` run of a validation workflow shares a stable workflow/ref concurrency
group with `queue: max` and `cancel-in-progress: false`. One run executes at a time
in that group, preserving up to 100 pending runs instead of replacing them on a
new push. Further runs can be cancelled when that queue is full. Queue entry order
determines processing order; commit or push order is not guaranteed.

Other refs use `queue: single` and cancel superseded runs for the same ref. Different
validation workflows and repositories can still run concurrently, subject to runner
capacity; this does not reserve slots or give PRs scheduling priority. Existing runs
keep their original workflow configuration. Test selection, numerical criteria,
required checks and release gates are unchanged. A release still requires successful
validation of its exact commit. This is a CI execution queue, not a PR merge queue.

## CI runs the two partitions in shards

The [test workflow](https://github.com/fieldalgo/wavebit/blob/main/.github/workflows/tests.yml)
uses the runner for both non-research partitions, each split across runners with
`--shard INDEX/COUNT`: six shards for the ordinary partition, two for the slow one.

```shell
python tools/run_verification.py --profile full --slow exclude --shard 1/6 --workers 2 \
  --timeout 1200 --timeout-method thread --verbose --live-output \
  --output artifacts/verification/default
python tools/run_verification.py --profile full --slow only --shard 1/2 --workers 2 \
  --timeout 2400 --timeout-method thread --verbose --live-output \
  --output artifacts/verification/slow
```

A shard is a selection within the profile, not a different profile. Every shard collects
the whole partition and records it as `collected-nodeids-all.json`; whole test files are
then assigned to shards deterministically, largest file first to the lightest shard, so
a file is never split and the same collection always gives the same shards. After the
shard jobs, `tools/verify_shards.py --count N` reads their receipts and fails unless every
shard is present once, all collected the same list, the shards partition it, and each
exited 0. That job is what makes the partition verified; a shard whose runner died is a
named failure there, not an absence.

Sharding is a response to the runners' memory, not to the clock: a single two-worker run
of the ordinary partition grew past the 7.9 GB available and was shut down at 50 to 61 %
every time. The per-test limits stay at 1,200 seconds (ordinary) and 2,400 seconds (slow);
the job limits are 120 minutes per shard, against healthy lengths of about 50 (ordinary)
and 40 (slow) minutes. Source-mutating
cases within a shard run in the runner's separate serial snapshot, not in the shared
parallel checkout.

GitHub artifacts `verification-default-N` and `verification-slow-N` retain each shard's
JSON receipts and inventories, JSONL events, JUnit XML and logs. The writable source
snapshot is excluded from upload to avoid copying the repository into every
artifact. These are verification records, not full source snapshots.

## Resume an identical run

```shell
python tools/run_verification.py --profile full --workers 4 --resume RUN_DIRECTORY
```

Keep the profile, slow filter, shard, paths, worker and timeout settings, output mode,
checkout inputs and environment unchanged. The runner fingerprints them again and recollects the whole selection.
It refuses reuse if any fingerprint or collected node ID changed, including new,
missing or reordered tests. Failures and unfinished tests run again by default;
completed passes, skips, expected failures and non-strict unexpected passes are
reused with their original
receipt location. Changing `XLA_FLAGS`, including setting or removing it, changes
the environment fingerprint and refuses reuse; start a new run instead.
Unexpected passes remain separately reported; strict XPASS is
a failure and is rerun. A new receipt records the combined result, without overwriting
the previous run. This is local reproducibility evidence, not a claim that remote
services, hardware or unrecorded external state cannot change.

Use `--path` repeatedly to restrict a profile to named repository paths or node IDs.
`--collect-only` records the selection without executing it. Neither changes the
meaning of the profile's research/slow marker expression.

## Tests that temporarily modify source

The known boundary and Gamma mutation tests, and tests marked `source_mutation`,
are removed from the ordinary run. They run serially in a separate copy of the
recorded inputs. The recorder checks byte identity before execution, verifies the
package import comes from the copy, and checks that the copy was restored and the
original checkout stayed unchanged. New tests that write package source must use
this marker and restore their changes even when an assertion fails. Do not use
parallel workers to mutate a shared checkout.

The quick profile does not replace the separate distribution validation:

```shell
python tools/run_distribution_smoke.py
```

That tool builds a wheel and source distribution and checks fresh installed
packages outside the checkout, including package data and optional plotting.
