# Recorded verification and safe resumption

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

Each invocation creates a new directory under `artifacts/verification/`.
`--output` may choose a different new directory; an existing directory is refused.
The tool prints the location before running tests. Inspect:

- `inputs.json`: hashes of repository source, tests, fixtures and other recorded
  inputs, Python and installed distribution versions, selected configuration,
  and a short allowlist of numerical environment controls. Arbitrary environment
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

## Resume an identical run

```shell
python tools/run_verification.py --profile full --workers 4 --resume RUN_DIRECTORY
```

Keep the profile, paths, worker configuration, checkout inputs and environment
unchanged. The runner fingerprints them again and recollects the whole selection.
It refuses reuse if any fingerprint or collected node ID changed, including new,
missing or reordered tests. Failures and unfinished tests run again by default;
completed passes, skips, expected failures and non-strict unexpected passes are
reused with their original
receipt location. Unexpected passes remain separately reported; strict XPASS is
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
