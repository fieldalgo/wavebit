# Wavebit — conventions for agent sessions

This is the canonical source of repository instructions; `CLAUDE.md` imports it.
This file holds the branching and landing rules of the repository. Maintainers keep their private notes
outside this repository; nothing private belongs in any branch, commit message or file here.

## Branches: one trunk, `main`

| Branch | What it is |
|---|---|
| `main` | The only long-lived branch and the default. Every change lands here, RCWA, FDTD and shared work alike, each from its own feature branch, after its pull request's CI is green. A release is a tag on one of its commits. |
| `maintenance/X.Y.x` | Only if a patch has to go out for a released version without the trunk's newer work: cut from that release's tag when first needed, fixes cherry-picked from `main`. None exists yet. |

`develop` (2026-10-15 to 16) and the per-solver trunks `rcwa` and `fdtd` (2026-10-15) are
retired; everything they held is on `main`. Solver work is told apart by the commit scope
(`rcwa:`, `fdtd:`) and the paths it touches, not by a branch. This is how NumPy, PyTorch and
Matplotlib work, and it leaves nothing to keep in sync.

### Landing a feature branch

Branch from `main`. `main` keeps its history linear, and `origin/main` is the only `main` that
counts: a landing is finished when it is pushed.

1. `git fetch origin` and `git rebase origin/main` in the branch's own worktree. Squash
   WIP/fixup commits here.
2. Push the branch and open a **draft** pull request against `main` (`gh pr create --draft`).
   A draft runs no CI, so push as often as you like while the work is in progress. Run the
   tests the change touches in a suitable local or remote environment; when they pass, mark the pull request ready
   (`gh pr ready <number>`). That starts the Linux suite (free on this public repository, about
   seventy minutes; shared runner capacity can add queue time). Fix what fails and wait for
   green. A docs-only change goes the same way; there are no exceptions by size.
3. Land through the pull request: **Rebase and merge** on GitHub, or
   `gh pr merge <number> --rebase --delete-branch`. GitHub replays the commits onto the tip of
   `main` (same content and messages, new hashes) and nothing is run again; the branch ruleset
   allows no other way in -- no direct push, no merge commit, no squash, no force push. Only
   a pull request that conflicts with `main` needs a manual rebase and a new run.
4. Cite the hash `main` holds, not the pull request's: read it after landing.

This is how NumPy and PyTorch land (the pull request's checks gate it; the merged state is not
run again first), chosen 2026-10-16 over a merge queue for speed while the project is young.
So `main` can go red when two landings interact. The run on `main` after each landing is the
record; if it is red, fix it the same day or revert the landing through a pull request
(`git revert`) and come back later -- a revert is procedure, not failure. Releases are only
ever cut from a green `main` commit (`release-gate.yml`), so a red `main` never reaches a
user. Never land by fast-forwarding a local `main` and pushing: on 2026-10-15 twenty commits
were landed that way on one machine and had to be rebased by hand. When several branches
must land together because they fix each other's failures, one session stacks them on one
branch and lands that pull request.

A release: bump the version, land it like anything else, then tag that commit `vX.Y.Z` and
publish the GitHub release; `release-gate.yml` refuses to publish unless `tests` was green on
that commit, and publishing uploads the package and the documentation site.

`merge.ff=only` is set in the repo config as a guard (`git config merge.ff only` on a fresh
clone), so a plain `git merge` that would create a merge commit refuses. Do not override it.

Keep commits atomic with meaningful messages — the commit, not the branch, is the unit of
history here.

Never rewrite `main` once pushed: docs, tests and studies cite their commits by hash. Commits that exist only on one machine may be rebased onto the updated remote branch
before they are pushed.

## Issues

Public-safe observations, suspected bugs, accuracy and research questions,
enhancements and validation gaps are welcome, one topic per issue. Search open
and closed issues first. Use the closest form, state available evidence and mark
unknown details honestly. A complete reproducer, independent reference or confirmed
defect is helpful but not a prerequisite. Pin the release or public commit and
distinguish main, released, nightly and PR-only behavior. Agents prepare a concrete
draft and post only after the maintainer approves it. No private information goes
into public issues. A resolving PR uses `Fixes #<number>`; partial work references
the issue without claiming closure. Intake does not relax review or acceptance.

## Shared validation and contribution policy

Reviewed Ready PRs use automatic Rebase and merge after relevant validation and
all required checks. Never bypass protections or auto-merge an unreviewed draft.
Main validation uses one stable workflow/ref group with `queue: max` and no
cancellation. Non-main runs use `queue: single` and supersede older runs. These
Actions queues are distinct from a PR merge queue.

Use portable path, temporary-directory and subprocess APIs with explicit text
encodings. Validate OS-sensitive contracts on the supported systems, retaining
failures and artifacts. Full Linux tests and isolated distribution checks on
Linux, macOS and Windows have different scopes; a pass on one does not establish
the others. Do not weaken numerical gates or add unexplained skips to hide a
platform failure.

English is canonical for code, contribution guidance and README files. Update
English and Korean product documentation together under docs, preserving units,
commands and limits. Keep private planning and progress records outside the
public repository. Record implementation, exact validation and landing separately.
Verify dependency licenses and provenance before adoption. Agent commits carry
their co-author trailer, for example `Co-authored-by: Codex <noreply@openai.com>`.
