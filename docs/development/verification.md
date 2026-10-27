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
readiness. The [contribution guide](https://github.com/fieldalgo/wavebit/blob/main/CONTRIBUTING.md) and development documents
hold the shared policy. AGENTS.md is a short technical entry point; CLAUDE.md imports it.
Personal preferences, account settings, private paths and session history stay outside
the public repository.

Changes target main through draft PRs, relevant checks and review, then required CI
and GitHub Rebase and merge. Only conflicts require a manual rebase and another run.
Preserve published history; do not directly push main, force-push shared branches,
squash on landing or create integration merge commits. Record the resulting main SHA
and retain main validation. Fix or revert failed main changes through a PR before
release. Release tags require successful validation of the exact resulting commit.
A commit or PR that lands work replayed, rewritten, squashed or ported from another
commit names each original in its body as `Ports: <sha>` (SHA only); do not report a
commit missing from SHA ancestry alone.

Use portable path, temporary-directory and subprocess APIs with explicit encodings.
Verify OS-sensitive behavior without weakening numerical gates or hiding failures
with unexplained skips. English is canonical; update Korean documentation alongside
its English counterpart. Keep executable examples and code in English.

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

## Dependency versions in CI

Required CI does not install the newest releases. Every job of the
[test workflow](https://github.com/fieldalgo/wavebit/blob/main/.github/workflows/tests.yml)
except `lint` installs through `tools/ci_constraints.py`, which applies the exact versions in
`.github/constraints/`: `tests.txt` for the test, slow and examples jobs, `docs.txt` for the
docs build and its deployment, `distribution.txt` for the distribution smoke, the release
build and their isolated builds. A release upstream therefore cannot turn a run red on a commit that changed nothing.
`pyproject.toml` is unchanged by this: its floors are still the oldest versions verified,
users still install the newest releases, and the files are not an install requirement.

There is one file per environment because the environments disagree. The PyTorch CPU index
mirrors only some versions of torch's own dependencies, and the docs extra and the packaging
tools hold different versions of shared distributions.

The [latest dependencies workflow](https://github.com/fieldalgo/wavebit/blob/main/.github/workflows/latest-deps.yml)
watches the newest releases. It is not a required check. Daily, it installs the same three
environments without the files and runs the license check, the quick profile, the
backend-sensitive files, the `-W` docs build and the distribution smoke on Linux, macOS and
Windows. A scheduled
failure opens or comments on one issue, which the next passing run closes. On Mondays, or on
a manual run with `refresh`, it compares the versions it installed with the recorded ones. If
they differ, it pushes a branch from main that replaces the files. It then opens or updates
one issue that lists the changes and links to a pull request from that branch. A person
opens that pull request: Actions may not open pull requests in this repository, and one its
token opened would run no CI.

A version moves only through that pull request. Its required CI is the full validation on the
new versions, including macOS and Windows. Merge it when it is green. If it is red, close it:
the recorded versions stay, and the next refresh offers newer ones. To land part of it, keep
the failing distribution at its previous version in a follow-up pull request and record why.
A tolerance is not loosened and a golden is not regenerated to make a version pass. A newly
installed distribution needs its license review in the same pull request.

To reproduce a CI environment locally, in a fresh virtual environment:

```shell
python tools/ci_constraints.py install tests            # the recorded versions
python tools/ci_constraints.py install tests --latest   # the newest releases
python tools/ci_constraints.py freeze tests             # record what is installed
```

## CI runs the two partitions in shards

The [test workflow](https://github.com/fieldalgo/wavebit/blob/main/.github/workflows/tests.yml)
uses the runner for both non-research partitions, each split across runners with
`--shard INDEX/COUNT`: thirty-two shards for the ordinary partition, twelve for the slow one.

CI stores both verification outputs and their pytest temporary fixtures under
`RUNNER_TEMP`, outside the checkout, so study fixtures can enforce their raw-output
isolation contract. GitHub provides this variable; for local reproduction, set it
to an external scratch directory first. Artifact uploads use the same output root.

```shell
python tools/run_verification.py --profile full --slow exclude --shard 1/32 --workers 2 \
  --shard-timings tools/ci_shard_timings.json \
  --timeout 1200 --timeout-method thread --verbose --live-output \
  --output "$RUNNER_TEMP/wavebit-verification/default"
python tools/run_verification.py --profile full --slow only --shard 1/12 --workers 2 \
  --shard-timings tools/ci_shard_timings.json \
  --timeout 2400 --timeout-method thread --verbose --live-output \
  --output "$RUNNER_TEMP/wavebit-verification/slow"
```

A shard is a selection within the profile, not a different profile. Every shard collects
the whole partition and records it as `collected-nodeids-all.json`; whole shard units are
then assigned to shards deterministically, largest estimated duration first to the lightest shard,
so the same collection and timing table give the same shards. A unit is a test file, except
for tests marked `shard_unit`: a bare `@pytest.mark.shard_unit` lets each of its cases run
apart from the rest of the file, `shard_unit(per='test')` makes each test function one unit
with all its parameter cases (a whole file opts in with `pytestmark`), and `shard_unit('name')`
keeps the cases of one file that carry that name together, for cases that share a cache or a
module fixture. The marker closest to a case wins, so a named group overrides a file's mark. A file that sets
process-wide state at import is safe either way, since every shard is a separate process; within
a shard `--dist loadfile` still keeps one file's cases on one worker. Mark only cases that do not
rely on another case of their file having run first. After the
shard jobs, `tools/verify_shards.py --count N` reads their receipts and fails unless every
shard is present once, all collected the same list, the shards partition it, and each
exited 0. That job is what makes the partition verified; a shard whose runner died is a
named failure there, not an absence.

Sharding is a response to the runners' memory, not to the clock: a single two-worker run
of the ordinary partition grew past the 7.9 GB available and was shut down at 50 to 61 %
every time. The slow partition followed: as three shards, shard 3 lost its runner
at about 95 % on both attempts of run 39098232669, so it ran as six. Twelve since 2026-10-20
are for the clock: whole files could not bring a slow shard under the largest file, 21.7
worker-minutes, and with `shard_unit` the largest unit is 8.1. The ordinary partition went
from sixteen to thirty-two the same day: as sixteen it was bound both by load (585
worker-minutes) and by its largest file (18.3), and its fourteen files above 9 worker-minutes
are now marked per test. The per-test limits stay at 1,200 seconds (ordinary) and 2,400 seconds (slow);
the job limits are 120 minutes per shard. Measured durations guide balancing;
actual elapsed times still depend on collection, compilation and worker scheduling. Source-mutating
cases within a shard run in the runner's separate serial snapshot, not in the shared
parallel checkout.

GitHub artifacts `verification-default-N` and `verification-slow-N` retain each shard's
JSON receipts and inventories, JSONL events, JUnit XML and logs. The writable source
snapshot is excluded from upload to avoid copying the repository into every
artifact. These are verification records, not full source snapshots.

## Runner diagnostics

A manual dispatch with the `slow_runner_diagnostics` input runs only the slow shards, each
with a Linux-only sidecar that flushes host memory/swap, memory pressure, process RSS
and cgroup v2 limits/events every five seconds to both the live CI log and
`runner-diagnostics-slow-N`. It does not collect command lines or environment
variables. With passwordless sudo and tracefs support, a private tracing instance
also records kernel signal generation (sender task/PID and target) and OOM victims.
A harmless SIGCONT self-probe checks the signal stream. Missing tracing permissions
or capabilities are explicitly logged; absence of an event does not prove absence
of a signal or an OOM. SIGKILL or host loss can prevent final samples and uploads.

Such a run skips the other jobs and the ordinary partition's aggregator; `slow tests
complete` still checks the slow shards add up. A `diagnostics run, not a CI gate` job
fails the run on purpose, so the release gate, which accepts any successful `tests`
run on the release commit, never counts it. With the input omitted or false, the
normal CI gates remain and no sidecar runs. The numerical tests, worker count and
timeout limits are unchanged. Compare memory counters and signal timestamps with
test progress before attributing a shutdown to a test, a resource limit or a manual
cancellation.

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

The fixed `tools/ci_shard_timings.json` table scales each unit's summed setup/call/teardown
cost by the current case count; its rows are files, and `shard_unit` units by the names the
receipts record (the case's node id, or `FILE::@name`). A unit without a row uses its file's
mean per-case cost, and a new file the partition mean per-case cost (at least one second). Timing data participates in resume fingerprints and cross-shard consistency
checks. Without the option, assignment remains count-based. Refresh from all artifacts of
one successful run, review and commit the result:

```shell
python tools/update_shard_timings.py artifacts/ci-timing-baseline --source-run https://github.com/fieldalgo/wavebit/actions/runs/RUN_ID --output tools/ci_shard_timings.json
```
