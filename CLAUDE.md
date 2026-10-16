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

Branch from `develop`. `develop` keeps its history linear, and `origin/develop` is the only
`develop` that counts: a landing is finished when it is pushed.

1. `git fetch origin` and, in the main checkout, `git merge --ff-only origin/develop`, so the
   local `develop` is origin's.
2. In the branch's own worktree: `git rebase develop`. Squash WIP/fixup commits here.
3. In the main checkout, which holds `develop`: `git merge --ff-only <branch>`.
4. At once: `git push origin develop`.

If step 3 or 4 is refused, another session landed first — go back to step 1 and retry. This
retry loop is the concurrency control between sessions; do not work around it. A landing that
stops before step 4 is not a landing: on 2026-10-15 twenty commits were fast-forwarded onto
one machine's `develop` and never pushed while another clone pushed eleven to origin, and the
two lines had to be rebased together by hand, conflicts and all.

`merge.ff=only` is set in the repo config as a guard (`git config merge.ff only` on a fresh
clone), so a plain `git merge` that would create a merge commit refuses. Do not override it.

Keep commits atomic with meaningful messages — the commit, not the branch, is the unit of
history here.

Never rewrite `main` or `develop` once pushed: docs, tests and studies cite their commits by
hash. Commits that exist only on one machine may be rebased onto the updated remote branch
before they are pushed.
