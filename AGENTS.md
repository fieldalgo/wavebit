# Wavebit — conventions for agent sessions

This file holds the branching and landing rules of the repository. Maintainers keep their private notes
outside this repository; nothing private belongs in any branch, commit message or file here.

## Branches: feature branches land on `develop`

| Branch | What lands there |
|---|---|
| `main` | The last release, and the branch a visitor lands on. Nothing lands here directly: at each release `main` is fast-forwarded to the `develop` commit being released and tagged. Until the first release it simply tracks `develop`. |
| `develop` | Everything: RCWA, FDTD and shared work alike, each from its own feature branch. The public mirror is cut from `develop` alone. |

Solver work is told apart by the commit scope (`rcwa:`, `fdtd:`) and the paths it touches, not
by a branch. The per-solver trunks `rcwa` and `fdtd` (2026-10-15) are retired: everything they
held is on `develop`.

### Landing a feature branch

Branch from `develop`. `develop` keeps its history linear:

1. In the branch's own worktree: `git rebase develop`. Squash WIP/fixup commits here.
2. In the main checkout, which holds `develop`: `git merge --ff-only <branch>`.

If step 2 fails, another session landed first — go back to step 1 and retry. This retry
loop is the concurrency control between sessions; do not work around it.

`merge.ff=only` is set in the repo config as a guard (`git config merge.ff only` on a fresh
clone), so a plain `git merge` that would create a merge commit refuses. Do not override it.

Keep commits atomic with meaningful messages — the commit, not the branch, is the unit of
history here.

Never rewrite `main` or `develop` once pushed: docs, tests and studies cite their commits by
hash. Commits that exist only on one machine may be rebased onto the updated remote branch
before they are pushed.
