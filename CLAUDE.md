# Wavebit — conventions for agent sessions

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
2. Push the branch and open a pull request against `main`. The Linux suite runs on it (free on
   this public repository; about seventy minutes). Fix what fails, rebase again if `main`
   moved, and wait for green. A docs-only change goes the same way; there are no exceptions
   by size.
3. Land the exact commit the green run tested: `git push origin <branch>:main` (a fast-forward;
   the branch ruleset refuses anything else, and refuses a commit without a green `tests`
   run). If it is refused because `main` moved, go back to step 1.
4. Delete the branch.

That retry loop is the concurrency control between sessions; do not work around it, and never
land through the merge button or by fast-forwarding a local `main` without pushing (on
2026-10-15 twenty commits were landed that way on one machine and had to be rebased by hand).
When several branches must land together because they fix each other's failures, one session
stacks them on one branch, runs one pull request, and lands that.

A release: bump the version, land it like anything else, then tag that commit `vX.Y.Z` and
publish the GitHub release; `release-gate.yml` refuses to publish unless `tests` was green on
that commit, and publishing uploads the package and the documentation site.

`merge.ff=only` is set in the repo config as a guard (`git config merge.ff only` on a fresh
clone), so a plain `git merge` that would create a merge commit refuses. Do not override it.

Keep commits atomic with meaningful messages — the commit, not the branch, is the unit of
history here.

Never rewrite `main` once pushed: docs, tests and studies cite their commits by hash. Commits that exist only on one machine may be rebased onto the updated remote branch
before they are pushed.
