# Wavebit — conventions for agent sessions

This file holds the branching and landing rules of the repository. Maintainers keep their private notes
outside this repository; nothing private belongs in any branch, commit message or file here.

## Branches: a trunk per solver, `develop` integrates them

Since 2026-10-15 each solver has its own long-lived trunk:

| Branch | What lands there |
|---|---|
| `rcwa` | RCWA work: `src/wavebit/rcwa/` and its tests, docs, studies and references |
| `fdtd` | FDTD work: `src/wavebit/fdtd/`, `tests/fdtd/`, `docs/*/fdtd/`, `studies/fdtd/` and their references |
| `main` | The last release, and the branch a visitor lands on. Nothing lands here directly: at each release `main` is fast-forwarded to the `develop` commit being released and tagged. Until the first release it simply tracks `develop`. |
| `develop` | Integration. Shared work lands here directly: `src/wavebit/_common/`, materials, the package root and the public lifecycle, `tools/`, CI, the docs skeleton, and anything both solvers need at once. The public mirror is cut from `develop` alone. |

A change belongs to the solver whose results it can move; a change that can move both
solvers' results is shared.

### Landing a feature branch

Branch from the trunk the work belongs to (`develop` for shared work). Each trunk keeps its
own history linear:

1. In the branch's own worktree: `git rebase <trunk>`. Squash WIP/fixup commits here.
2. Fast-forward the trunk to the branch:
   - where `<trunk>` is checked out (the main checkout holds `develop`): `git merge --ff-only <branch>`;
   - where it is checked out nowhere: `git fetch . <branch>:<trunk>` from any checkout of this
     clone, which refuses anything but a fast-forward.

If step 2 fails, another session landed first — go back to step 1 and retry. This retry
loop is the concurrency control between sessions; do not work around it.

`merge.ff=only` is set in the repo config as a guard (`git config merge.ff only` on a fresh
clone), so a plain `git merge` that would create a merge commit refuses. Do not override it
for feature work.

Keep commits atomic with meaningful messages — the commit, not the branch, is the unit of
history here.

### Between a trunk and `develop`: true merges only

The trunks and `develop` exchange work by merge commits, never by rebase or squash, so each
solver's own history stays reachable. This continues the FDTD integration the owner asked
for: `729b898d` merged `fdtd-general`, which is retired; `fdtd` is its successor.

- Trunk into `develop`, when the owner asks: in a checkout of `develop`,
  `git merge --no-ff <trunk>`.
- `develop` into a trunk, to take shared work and the other solver: in a checkout of the
  trunk, `git merge --ff develop` (a fast-forward while the trunk has nothing of its own,
  a merge commit otherwise). Resolve conflicts on the trunk, never on `develop`.
- These merges are the only exception to `merge.ff=only`.
- Never rewrite `main`, `develop`, `rcwa` or `fdtd` once pushed: docs, tests and studies cite their
  commits by hash. Commits that exist only on one machine may be rebased onto the updated
  remote branch before they are pushed.
