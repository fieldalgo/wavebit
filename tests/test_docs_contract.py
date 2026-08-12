"""The contract `docs/README.md` states about the documents beside it, enforced.

A rule nobody checks is a rule that has already been broken somewhere. This one had been:
documents in `docs/` pointed at directories on one contributor's machine -- a private study, a
folder of unpublished drafts, an old project name -- and nothing noticed, because a dangling
pointer typesets exactly as well as a live one. What it costs is two things at once. A reader
who clones the repository cannot follow the link, and the author discloses the shape of their
own filesystem without deciding to.

So two checks, matching the two halves of the contract:

* **Nothing in the published tree names a path outside it.** Not just `docs/` -- the same leak
  reached a generator's default argument, a results log and a notebook's stored output, so the
  check reads every text file that ships.
* **Every `Targets:` path in a document resolves.** A document here exists to explain named
  code; if the name is wrong the document is decorative, and it rots silently because prose
  does not fail to import.

Both are deliberately dumb pattern checks. The failure they catch is not subtle -- it is
someone pasting a path they can see from where they are sitting -- and a dumb check is one that
still works when nobody remembers it exists.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Directories that ship. `studies/` and `tutorials/` are here because the leak reached both.
PUBLISHED = ('fieldwave', 'docs', 'tests', 'studies', 'tutorials', 'examples')
TEXT_SUFFIXES = {'.py', '.md', '.ipynb', '.toml', '.cfg', '.yml', '.yaml', '.txt', '.m'}

# A home directory or a user-rooted absolute path. `~/` alone is not enough to match: it is
# ordinary in prose ("keep it as a sibling, `../<your-scratch-dir>`"), and what makes a pointer a
# leak is that it descends into somebody's tree.
OUTWARD = re.compile(
    r'(?:/(?:Users|home)/[A-Za-z0-9._-]+/'          # /Users/someone/... or /home/someone/...
    r'|~/[A-Za-z0-9._-]+/'                          # ~/project/..., ~/miniconda3/...
    r'|[A-Za-z]:\\\\?[A-Za-z0-9._-]+\\)'            # C:\Users\...
)

# This file quotes the patterns it forbids, and so does the contract that describes them.
EXEMPT = {'tests/test_docs_contract.py', 'docs/README.md', 'CONTRIBUTING.md'}


def published_files():
    for d in PUBLISHED:
        for p in sorted((ROOT / d).rglob('*')):
            if p.is_file() and p.suffix in TEXT_SUFFIXES and '__pycache__' not in p.parts:
                yield p


def test_nothing_published_points_outside_the_repository():
    """No file that ships names a path on somebody's machine."""
    bad = []
    for p in published_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in EXEMPT:
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            m = OUTWARD.search(line)
            if m:
                bad.append(f'{rel}:{n}: {line.strip()[:100]}')

    assert not bad, (
        'these point outside the repository, so they are dangling links for anyone who clones '
        'it and a disclosure of the author\'s filesystem:\n  ' + '\n  '.join(bad) +
        '\n\nState the substance instead of the address -- see the contract in docs/README.md.')


def docs_with_targets():
    for p in sorted((ROOT / 'docs').glob('*.md')):
        text = p.read_text(encoding='utf-8')
        if 'Targets' in text:
            yield p, text


# A repo-relative path in backticks. This catches the other half of the same mistake: not
# `/Users/someone/...` but `<a survey of the available solvers>` or `<a scratch note outside this repository>` -- a path that
# looks local, resolves on the author's disk, and is simply absent from the clone. It reads as
# an in-repo reference, which makes it worse than an obvious outward one.
CITED_PATH = re.compile(r'`([A-Za-z0-9_][A-Za-z0-9._/-]*\.(?:md|py|json|npz|toml|yml|cfg))`')


@pytest.mark.parametrize('doc', sorted(p.name for p in (ROOT / 'docs').glob('*.md')))
def test_paths_a_document_cites_are_in_the_repository(doc):
    """Every backticked path resolves, from the repo root or as `_core/...` shorthand."""
    p = ROOT / 'docs' / doc
    missing = []
    for m in CITED_PATH.finditer(p.read_text(encoding='utf-8')):
        t = m.group(1)
        if '/' not in t:
            continue                                    # a bare filename is prose, not a path
        if any((base / t).exists() for base in (ROOT, ROOT / 'fieldwave', p.parent)):
            continue
        missing.append(t)
    assert not missing, (
        f'docs/{doc} cites paths that are not in the repository: {sorted(set(missing))}. '
        'A path that resolves only on the author\'s disk reads as an in-repo reference and is '
        'the harder half of the same leak -- see the contract in docs/README.md.')


@pytest.mark.parametrize('doc', [p.name for p, _ in docs_with_targets()])
def test_every_target_a_document_names_exists(doc):
    """A document explains named code; the names have to be real."""
    text = (ROOT / 'docs' / doc).read_text(encoding='utf-8')
    missing = []
    for m in re.finditer(r'`(fieldwave/[A-Za-z0-9_./*-]+\.py)', text):
        target = m.group(1)
        found = list(ROOT.glob(target)) if '*' in target else [ROOT / target]
        if not any(f.exists() for f in found):
            missing.append(target)
    assert not missing, (
        f'docs/{doc} explains files that do not exist: {sorted(set(missing))}. '
        'Either the document is stale or the path is a typo; both make it decorative.')
