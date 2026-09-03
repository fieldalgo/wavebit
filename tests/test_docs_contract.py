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
PUBLISHED = ('wavebit', 'docs', 'tests', 'studies', 'tutorials', 'examples')
TEXT_SUFFIXES = {'.py', '.md', '.ipynb', '.toml', '.cfg', '.yml', '.yaml', '.txt', '.m'}

# A home directory or a user-rooted absolute path. `~/` alone is not enough to match: it is
# ordinary in prose ("keep it as a sibling of the checkout"), and what makes a pointer a leak
# is that it descends into somebody's tree.
OUTWARD = re.compile(
    r'(?:/(?:Users|home)/[A-Za-z0-9._-]+/'          # /Users/someone/... or /home/someone/...
    r'|~/[A-Za-z0-9._-]+/'                          # ~/project/..., ~/miniconda3/...
    r'|[A-Za-z]:\\\\?[A-Za-z0-9._-]+\\)'            # C:\Users\...
)

# This file quotes the patterns it forbids, and so does the contract that describes them.
EXEMPT = {'tests/test_docs_contract.py', 'docs/README.md', 'CONTRIBUTING.md'}

# Paths that do not resolve and are named on purpose. Listed rather than pattern-matched, so
# that adding one is a decision somebody writes down: the whole failure this file exists to
# catch is a reference that looks fine and is not.
CITED_BUT_ABSENT = {
    # A file the script writes rather than reads. `*.json` is ignored on purpose -- the log
    # beside it is the committed record, and a second copy would be a second thing to keep true.
    'studies/results/backends.json',
    # Deliberately named as gone. Each appears in prose that says so ("used to live in",
    # "the legacy", "the deleted"), explaining where a test came from or what it replaced.
    'QA/1d_pattern_in_1dc_and_2d.py',
    'QA/autodiff_raster1.py',
    'on_numpy/emsolver/nvf.py',
}


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


# `path.py::symbol` -- how this codebase points at one function inside a file. The path half
# was already checked by CITED_PATH above; the *symbol* half was not, because that regex stops
# at `.py`. All 78 of these went stale silently, which is the same shape as the bare-filename
# gap the comment above measures and rejects -- except that a `::symbol` is never prose. It is
# always a claim about a definition, so it can be checked without exemption pressure, and the
# 2026-08-20 rename set is what made it worth checking: every one of them named a method or a
# function, and several named ones that were about to move.
SYMBOL_REF = re.compile(r'`([A-Za-z0-9_./-]+\.py)::([A-Za-z0-9_.]+)')

# Symbols named on purpose that no definition backs. Listed rather than pattern-matched, for
# the same reason CITED_BUT_ABSENT is.
SYMBOL_EXEMPT = {
    # a glob over the `dfs2d_*` builder family, not one name
    '_core/factorization.py::dfs2d_',
    # `_block` is defined in solve_1d.py and imported into solve_2d.py, which is where the
    # document is describing its use
    'solve_2d.py::_block',
    # a meent file the prose marks as gone ("meent carried a regularized rule for this")
    'on_torch/emsolver/primitives.py::Eig',
}


def test_every_symbol_a_document_names_exists():
    """`file.py::name` has to name a real definition, not only a real file.

    A stale one reads exactly like a live one and sends the reader looking for a function that
    was renamed out from under the sentence.
    """
    bad = []
    for p in published_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in EXEMPT:            # this file quotes the shapes it checks for
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        for m in SYMBOL_REF.finditer(text):
            path, symbol = m.group(1), m.group(2)
            if f'{path}::{symbol}' in SYMBOL_EXEMPT:
                continue
            sources = _sources_named(path, p)
            if not sources:
                bad.append(f'{rel}: no such file: {path}::{symbol}')
                continue
            if not any(_defines(src, symbol) for src in sources):
                bad.append(f'{rel}: no such symbol: {path}::{symbol}')

    assert not bad, (
        'these name symbols that do not exist:\n  ' + '\n  '.join(sorted(set(bad))) +
        '\n\nA `file.py::name` is a claim about a definition. Renaming the definition without '
        'the sentence leaves prose that typesets perfectly and points nowhere.')


def _sources_named(path, citing):
    """Every file the citation could mean, as text. Same three bases as the path check, plus
    a basename match, because `factorization.py::x` is how this repo writes a `_core` file."""
    out = []
    for base in (ROOT, ROOT / 'wavebit', citing.parent):
        candidate = base / path
        if candidate.is_file():
            out.append(candidate)
    if not out:
        out = [q for q in ROOT.rglob(pathlib.Path(path).name)
               if q.is_file() and q.as_posix().endswith(path) and '__pycache__' not in q.parts]
    return [q.read_text(encoding='utf-8', errors='ignore') for q in out]


def _defines(src, symbol):
    """A definition, an assignment, or -- for `file.py::214` -- a line that exists."""
    last = symbol.split('.')[-1]
    if last.isdigit():
        return int(last) <= len(src.splitlines())
    return bool(re.search(rf'^\s*(def|class)\s+{re.escape(last)}\b', src, re.M)
                or re.search(rf'^\s*{re.escape(last)}\s*[:=]', src, re.M))


# A link into this repository written as a full URL. `docs/` grew these when the documentation
# site landed: a page there may legitimately point at a notebook to run or a test to read, and a
# relative `../tests/...` is correct in a clone while being unresolvable in a site whose root is
# `docs/` -- a reader on the site clicks it and gets a 404. Absolute URLs work in both.
#
# What that costs is the thing this file exists to prevent. A relative link is checked by the
# clone itself the moment someone follows it; a URL is checked by nobody, and a renamed tutorial
# leaves a link that still looks live and 404s only for a reader who is not here to complain.
# So the path half is checked the same way every other citation is -- offline, against the tree,
# with no network involved.
REPO_URL = re.compile(
    r'https://github\.com/fieldalgo/wavebit/(?:blob|tree)/[A-Za-z0-9._/-]+?/'
    r'([A-Za-z0-9_][A-Za-z0-9._/-]*)')


def test_repository_urls_name_something_in_the_tree():
    """A github.com URL into this repository has to point at a path that exists here.

    The branch in the URL is not checked and cannot be: it names a ref that may not be the one
    checked out. What is checked is the path after it, which is what goes stale in a rename.
    """
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
            for m in REPO_URL.finditer(line):
                cited = m.group(1).rstrip('/).,')
                if not (ROOT / cited).exists():
                    bad.append(f'{rel}:{n}: {cited}')

    assert not bad, (
        'these URLs point into this repository at paths that are not in it, so they 404 for '
        'every reader and for nobody who can fix them:\n  ' + '\n  '.join(bad))


def docs_with_targets():
    for p in sorted((ROOT / 'docs').glob('*.md')):
        text = p.read_text(encoding='utf-8')
        if 'Targets' in text:
            yield p, text


# A repo-relative path in backticks. This catches the other half of the same mistake: not
# `/Users/someone/...` but `third_party/rcwa/ANALYSIS.md` or `dev/bench_jit.py` -- a path that
# looks local, resolves on the author's disk, and is simply absent from the clone. It reads as
# an in-repo reference, which makes it worse than an obvious outward one.
CITED_PATH = re.compile(r'`([A-Za-z0-9_][A-Za-z0-9._/-]*\.(?:md|py|json|npz|toml|yml|cfg))`')


def test_paths_cited_anywhere_are_in_the_repository():
    """Every backticked path in a shipped file resolves.

    Scoped to `docs/` at first, which is how `third_party/rcwa/ANALYSIS.md` survived twice in
    `tests/references/`: the rule was right and the check was looking in one place. A cited
    path is a promise the reader can follow it, and that promise is made wherever it is
    written.
    """
    bad = []
    for p in published_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in EXEMPT:
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        for m in CITED_PATH.finditer(text):
            t = m.group(1)
            if '/' not in t or t in CITED_BUT_ABSENT:
                # A bare filename is prose, not a path -- and that skip is a measured choice,
                # not an oversight. It does let a stale name through: `tests/data/README.md`
                # described `gen_goldens_analytic.py` as still present for months after
                # `ee70470` deleted it. Widening the rule was tried and rejected on the
                # numbers. Resolving a bare name *beside the citing file* flags 84 of 330
                # citations, almost all correct prose (`docs/` saying `boundary.py` for
                # `wavebit/_core/boundary.py`). Resolving it *anywhere in the tree* is far
                # better at 17, but 14 of those 17 are legitimate too: files of other projects
                # (GRCWA's `kbloch.py`, fmmax's `utils.py`), names the prose itself marks as
                # former (`golden_legacy_field.npz`), things deliberately deleted, and a script
                # that lives on an unmerged branch. Buying one real catch with fourteen
                # exemptions to write and maintain is a worse document than the one stale line.
                continue
            # Three bases, and the middle one is a deliberate house style rather than slack:
            # `_core/parametric.py` is how this codebase refers to its own internals, in 13 files
            # and counting, and there is only one `_core/` for it to mean. Spelling it
            # `wavebit/_core/...` everywhere would be longer and no clearer. The third base
            # accepts a link relative to the citing file, which is how one doc points at another.
            if any((base / t).exists() for base in (ROOT, ROOT / 'wavebit', p.parent)):
                continue
            bad.append(f'{rel}: {t}')

    assert not bad, (
        'these cite paths that are not in the repository:\n  ' + '\n  '.join(sorted(set(bad))) +
        '\n\nA path that resolves only on the author\'s disk reads as an in-repo reference, '
        'which is the harder half of the same leak -- nothing about it looks wrong. State the '
        'substance instead of the address; see the contract in docs/README.md.')


@pytest.mark.parametrize('doc', [p.name for p, _ in docs_with_targets()])
def test_every_target_a_document_names_exists(doc):
    """A document explains named code; the names have to be real."""
    text = (ROOT / 'docs' / doc).read_text(encoding='utf-8')
    missing = []
    for m in re.finditer(r'`(wavebit/[A-Za-z0-9_./*-]+\.py)', text):
        target = m.group(1)
        found = list(ROOT.glob(target)) if '*' in target else [ROOT / target]
        if not any(f.exists() for f in found):
            missing.append(target)
    assert not missing, (
        f'docs/{doc} explains files that do not exist: {sorted(set(missing))}. '
        'Either the document is stale or the path is a typo; both make it decorative.')


def test_every_toc_entry_names_a_document_that_exists():
    """The navigation is the one place a renamed document fails silently.

    Every other reference to a document is prose a reader can still follow, or a path
    `test_paths_cited_anywhere_are_in_the_repository` already checks. `_toc.yml` is neither:
    it is consumed by the site build rather than read, its entries carry no suffix, and a
    stale one points at a page that is simply absent. Nothing in the suite noticed when
    `analytic-modeling` was renamed to `parametric-modeling` (2026-09-03), because the sweep
    that renamed it covered `.py`, `.md`, `.ipynb` and `.rst` and the navigation is `.yml`.

    So the check is on the *extension the sweep did not have*, which is the general form of
    that miss rather than the instance.
    """
    toc = (ROOT / 'docs' / '_toc.yml').read_text(encoding='utf-8')
    missing = [name for name in re.findall(r'-\s*file:\s*(\S+)', toc)
               if not any((ROOT / 'docs' / f'{name}{ext}').exists()
                          for ext in ('.md', '.ipynb', '.rst'))]
    assert not missing, (
        f'docs/_toc.yml lists pages that do not exist: {sorted(set(missing))}. '
        'The site build resolves these, so a stale entry is a missing page rather than a '
        'broken link -- add the document, or drop the entry.')
