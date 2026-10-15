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

What they read is what a clone gets: the files git tracks, not whatever is on the disk they run
on. See `tree` for the two ways reading the disk went wrong.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_root
import functools
import json
import os
import pathlib
import posixpath
import re
import shutil
import subprocess

import pytest

ROOT = repository_root()

# Directories that ship. `studies/` and `tutorials/` are here because the leak reached both.
# The files at the top of the tree ship too; `published_files` adds them.
PUBLISHED = ('src', 'docs', 'tests', 'studies', 'tutorials', 'examples', 'tools', 'assets', 'artifacts')
# `.log` because the results logs are committed records, and one carried a build machine's
# temporary directories.
TEXT_SUFFIXES = {'.py', '.md', '.ipynb', '.toml', '.cfg', '.yml', '.yaml', '.txt', '.m', '.log'}

# A home directory or a user-rooted absolute path. `~/` alone is not enough to match: it is
# ordinary in prose ("keep it as a sibling of the checkout"), and what makes a pointer a leak
# is that it descends into somebody's tree.
#
# A scratch directory is a path on one machine too. `docs/validation/rcwa/half-space-power-audit.md` cited its
# scripts, logs and licence records under `/private/tmp/...`, and a results log carried a build
# tool's `/private/var/folders/...`; the pattern read neither until 2026-09-26. So `/Users/`
# is caught however it continues, and `/private/` or `/tmp/` wherever it starts a path -- not
# inside a relative one like `tests/tmp/`. A drive letter has to start a word, so the JSON-escaped
# newlines of a results log (`occurred:\n\n`) do not read as `d:\n\`.
OUTWARD = re.compile(
    r'(?:/Users/'                                   # /Users/someone/..., or /Users/ itself
    r'|/home/[A-Za-z0-9._-]+/'                      # /home/someone/...
    r'|(?<![A-Za-z0-9_-])/(?:private|tmp)/'         # /private/tmp/..., /tmp/..., ../private/var/...
    r'|~/[A-Za-z0-9._-]+/'                          # ~/project/..., ~/miniconda3/...
    r'|(?<![A-Za-z0-9_])[A-Za-z]:\\\\?[A-Za-z0-9._-]+\\)'  # C:\Users\..., at a word start
)

# This file quotes the patterns it forbids, and so does the contract that describes them.
# CLAUDE.md and AGENTS.md, the agent instructions of a working checkout (the second is the copy
# other agent tools read), name local folders; whether they ship is settled when the public mirror
# is cut, not here.
# Imported historical learning records retain their original bytes and provenance.
# Their exact content hashes are checked by test_repository_layout.py.
PRESERVED_LEARNING_RECORDS = {'docs/validation/learning-records/complex-kx-roots.json', 'docs/development/learning-provenance/example-verification.json', 'docs/validation/learning-records/power-example.json', 'docs/validation/learning-records/tensor-material-api-tests.log', 'docs/development/learning-provenance/migration.json', 'docs/validation/learning-records/tensor-shape-integration.json', 'docs/validation/learning-records/tensor-integrated-regression.log', 'docs/validation/learning-records/chapter03-material-examples.json', 'docs/validation/learning-records/tensor-material-api.json', 'docs/validation/learning-records/tensor-integrated-expanded.log', 'docs/development/learning-provenance/direct-k-verification.json', 'docs/development/learning-provenance/toolchain.json', 'docs/validation/learning-records/nvf-measurements.json'}

EXEMPT = ({'tests/api/test_docs_contract.py', 'docs/README.md', 'CONTRIBUTING.md', 'CLAUDE.md', 'AGENTS.md'}
          | PRESERVED_LEARNING_RECORDS)

# Paths that do not resolve and are named on purpose. Listed rather than pattern-matched, so
# that adding one is a decision somebody writes down: the whole failure this file exists to
# catch is a reference that looks fine and is not.
CITED_BUT_ABSENT = {
    # A file the script writes rather than reads. `*.json` is ignored on purpose -- the log
    # beside it is the committed record, and a second copy would be a second thing to keep true.
    'studies/results/backends.json',
    'artifacts/studies/backends.json',
    'studies/results/fdtd_cpml_slab_cost.json',
    'studies/results/fdtd_slab_add_cost.json',
    'studies/results/fdtd_tfsf_oblique_leak.json',
    'artifacts/studies/fdtd_tfsf_oblique_leak.json',
    # Deliberately named as gone. Each appears in prose that says so ("used to live in",
    # "the legacy", "the deleted"), explaining where a test came from or what it replaced.
    'QA/1d_pattern_in_1dc_and_2d.py',
    'QA/autodiff_raster1.py',
    'on_numpy/emsolver/nvf.py',
}


@functools.lru_cache(maxsize=None)
def tree(root=ROOT):
    """Every file a clone of `root` contains, as repository-relative POSIX paths.

    Every check below finds its files and resolves its citations here, never on the disk
    directly, because the disk also holds whatever this machine put there, and that is wrong in
    both directions. An untracked `AGENTS.md` -- an agent tool's copy of `CLAUDE.md`, which is
    exempt under its own name -- failed the contract on one machine while a clean checkout
    passed. The other direction is quieter: an untracked file satisfies a citation as well as a
    tracked one does, so a reference nobody who clones can follow would pass. `docs/README.md`
    allows only "things that arrive with the clone", and an untracked file does not arrive.

    In a git checkout that is `git ls-files`, less any tracked file deleted from the disk: the
    tree `git commit -a` would record, read as it is on disk. Outside one -- an sdist, an
    unpacked archive -- nothing is tracked and what is on disk is what shipped, so the tree is
    walked.
    """
    tracked = _tracked(root)
    if tracked is not None:
        return tracked
    return frozenset(p.relative_to(root).as_posix() for p in root.rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)


def _tracked(root):
    """`git ls-files` at `root`, or None when `root` is not the top of a git checkout."""
    try:
        top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root, env=_git_env(),
                             capture_output=True, text=True)
    except OSError:                                  # no git on this machine
        return None
    # The top, not merely inside one: an sdist unpacked in some other project's checkout is not
    # tracked by it, and `ls-files` there would answer that nothing ships. The same directory,
    # not the same string: git prints the case the disk stores, so on a case-insensitive disk a
    # `root` spelled in another case is the top all the same, and compared as strings it would
    # fall back to the disk walk -- strays and all -- without a word.
    try:
        # rstrip the newline only: a directory name may end in whitespace, and `strip()` would
        # trim it into a path that is not there, falling back to the disk walk silently.
        at_top = top.returncode == 0 and pathlib.Path(top.stdout.rstrip('\n')).samefile(root)
    except OSError:                                  # git printed a path that is not there
        at_top = False
    if not at_top:
        return None
    listed = subprocess.run(['git', 'ls-files', '-z'], cwd=root, env=_git_env(),
                            capture_output=True, check=True).stdout
    # Tracked and present: a tracked file deleted from the working tree is leaving with the
    # next commit, and there is nothing on disk to read.
    return frozenset(n for n in os.fsdecode(listed).split('\0') if n and (root / n).is_file())


# The variables that tell git which repository, index and configuration to use. githooks(5): a
# hook runs with them exported, and one that runs git somewhere else should clear them. Here the
# somewhere else is a temporary repository a self-test builds, and inherited from a pre-commit
# hook, GIT_INDEX_FILE would stage that repository's files into the commit being made. The list
# is git's own, `git rev-parse --local-env-vars`, so a variable a later git adds is cleared too;
# this copy of git 2.39's answer is for a git that cannot give one.
_LOCAL_ENV_VARS = (
    'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_CONFIG', 'GIT_CONFIG_PARAMETERS', 'GIT_CONFIG_COUNT',
    'GIT_OBJECT_DIRECTORY', 'GIT_DIR', 'GIT_WORK_TREE', 'GIT_IMPLICIT_WORK_TREE', 'GIT_GRAFT_FILE',
    'GIT_INDEX_FILE', 'GIT_NO_REPLACE_OBJECTS', 'GIT_REPLACE_REF_BASE', 'GIT_PREFIX',
    'GIT_INTERNAL_SUPER_PREFIX', 'GIT_SHALLOW_FILE', 'GIT_COMMON_DIR')


@functools.lru_cache(maxsize=None)
def _local_env_vars():
    env = {k: v for k, v in os.environ.items() if k not in _LOCAL_ENV_VARS}
    try:
        asked = subprocess.run(['git', 'rev-parse', '--local-env-vars'], env=env,
                               capture_output=True, text=True)
    except OSError:                                  # no git on this machine
        return frozenset(_LOCAL_ENV_VARS)
    names = asked.stdout.split()
    return frozenset(names if asked.returncode == 0 and names else _LOCAL_ENV_VARS)


def _git_env():
    """The environment, less every variable that would point git at another repository."""
    local = _local_env_vars()
    return {k: v for k, v in os.environ.items() if k not in local}


@functools.lru_cache(maxsize=None)
def _directories(root=ROOT):
    return frozenset(str(d) for rel in tree(root) for d in pathlib.PurePosixPath(rel).parents)


def in_tree(rel, root=ROOT):
    """Whether a clone of `root` has `rel`, a repository-relative path, as a file or a directory."""
    rel = posixpath.normpath(rel)
    return rel in tree(root) or rel in _directories(root)


def published_files(root=ROOT):
    """The text files that ship: those at the top of the tree, and those under PUBLISHED."""
    for rel in sorted(tree(root)):
        if _published(rel):
            yield root / rel


def _published(rel):
    path = pathlib.PurePosixPath(rel)
    if '__pycache__' in path.parts:
        return False
    if len(path.parts) == 1:
        return (path.suffix in TEXT_SUFFIXES or path.suffix == '.bib'
                or path.name.startswith('LICENSE'))
    return path.parts[0] in PUBLISHED and path.suffix in TEXT_SUFFIXES


def outward_lines(root=ROOT):
    """Each line of a shipped file that names a path on somebody's machine, as `file:line: text`."""
    bad = []
    for p in published_files(root):
        rel = p.relative_to(root).as_posix()
        if rel in EXEMPT:
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if OUTWARD.search(line):
                bad.append(f'{rel}:{n}: {line.strip()[:100]}')
    return bad


def test_nothing_published_points_outside_the_repository():
    """No file that ships names a path on somebody's machine."""
    bad = outward_lines()
    assert not bad, (
        'these point outside the repository, so they are dangling links for anyone who clones '
        'it and a disclosure of the author\'s filesystem:\n  ' + '\n  '.join(bad) +
        '\n\nState the substance instead of the address -- see the contract in docs/README.md.')


@pytest.mark.parametrize('line', [
    'records in /Users/someone/project/notes.md',
    'logs: `/private/tmp/wavebit-run/regression.log`',
    'git archive <rev> | tar -x -C /tmp/before',
    'include directory: ../../../private/var/folders/x/T/build-env/lib',
    'see ~/project/example/wavebit',
    r'C:\Users\someone\wavebit',
    '  path: /private/var/folders/x/T/build-env/bin/python',
])
def test_the_outward_pattern_catches_each_form_it_names(line):
    """A pattern check is a check only if it fires: one line for each form it forbids."""
    assert OUTWARD.search(line), line


@pytest.mark.parametrize('line', [
    'tests/tmp_cases.py',
    'studies/results/tmp/run.log',
    'keep it beside the checkout, ~/ is fine',
    'https://example.org/privateer/x',
    r'another exception occurred:\n\nTraceback',
])
def test_the_outward_pattern_leaves_inward_paths_alone(line):
    assert not OUTWARD.search(line), line


def test_the_checks_read_the_top_of_the_tree_and_the_results_logs():
    """The files at the root and the committed `.log` records ship, so the checks read them."""
    read = {p.relative_to(ROOT).as_posix() for p in published_files()}
    assert {'README.md', 'LICENSE', 'CITATION.bib', 'pyproject.toml', 'environment.yml'} <= read
    _, records = _relocation_metadata(ROOT)
    logs = {r for r in records if r.endswith('.log')}
    assert logs, 'The relocation ledger must include the preserved experiment logs.'
    assert logs <= read, 'Every preserved results log must still be checked.'


# The self-tests of `tree` build a small repository, because what they test is what git answers.
needs_git = pytest.mark.skipif(shutil.which('git') is None, reason='needs git')
LEAK = 'notes kept in ~/project/elsewhere/notes.md'


# The configuration of whoever runs them must not change what `init` and `add` do in such a
# repository. A global ignore file listing `CLAUDE.md` -- a common one -- would refuse the `add`;
# so would the `info/exclude` of a template directory, which is how global hooks are installed,
# and so would `core.safecrlf` beside `core.autocrlf`, for a file with LF endings. So the
# self-tests' git reads no global or system configuration; is given an empty ignore file, because
# git reads the default one, `$XDG_CONFIG_HOME/git/ignore`, even then; and initializes from an
# empty template, because GIT_TEMPLATE_DIR names one from the environment. `_tracked` keeps the
# user's configuration: a checkout owned by another user is readable only through the
# `safe.directory` one of them grants.
_ISOLATED = {'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1'}


def _git(root, *args):
    subprocess.run(['git', '-c', f'core.excludesFile={os.devnull}', *args], cwd=root,
                   env={**_git_env(), **_ISOLATED}, check=True, capture_output=True)


def _init(root):
    """`git init` at `root` from an empty template, whatever GIT_TEMPLATE_DIR names."""
    _git(root, 'init', '-q', '--template=')


def _write(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text + '\n', encoding='utf-8')


# One outward line, five times: in a tracked file, in an untracked one at the top of the tree,
# in an untracked one inside a published directory, and in the exempt `CLAUDE.md` and `AGENTS.md`
# (the agent instructions; `AGENTS.md` arrived untracked from agent tools until it was committed).
STRAYS = {'README.md': LEAK, 'NOTES.md': LEAK, 'docs/stray.md': LEAK, 'CLAUDE.md': LEAK,
          'AGENTS.md': LEAK}


@needs_git
def test_an_untracked_file_is_not_read_and_a_tracked_one_is(tmp_path):
    """The contract judges what a clone gets: the same line fails tracked and passes untracked.

    Both halves are asserted by one equality, so reading too much (the strays) and reading too
    little (the tracked file) each fail it -- and the tracked `CLAUDE.md` stays exempt.
    """
    _write(tmp_path, STRAYS)
    _init(tmp_path)
    _git(tmp_path, 'add', 'README.md', 'CLAUDE.md')
    assert outward_lines(tmp_path) == [f'README.md:1: {LEAK}']


@needs_git
def test_a_checkout_named_in_another_case_is_still_its_own_top(tmp_path):
    """On a case-insensitive disk, a checkout named in another case is read from git all the same.

    git prints its top in the case the disk stores, so a string compare against a `root` typed
    in another case says "not the top", and the strays are read from the disk.
    """
    root, respelled = tmp_path / 'repo', tmp_path / 'REPO'
    _write(root, STRAYS)
    if not respelled.exists():
        pytest.skip('the disk is case-sensitive')
    _init(root)
    _git(root, 'add', 'README.md', 'CLAUDE.md')
    assert outward_lines(respelled) == [f'README.md:1: {LEAK}']


@pytest.mark.parametrize('where', ['outside any checkout', 'untracked inside another checkout'])
def test_without_a_checkout_of_its_own_the_tree_on_disk_is_read(tmp_path, where):
    """An sdist has no index to ask, so what is on disk is what shipped, strays and all.

    The second case is an archive unpacked inside some other project's checkout: git answers
    there, for the wrong tree, and it answers that nothing is tracked.
    """
    root = tmp_path
    if where != 'outside any checkout':
        if shutil.which('git') is None:
            pytest.skip('needs git')
        _init(tmp_path)
        root = tmp_path / 'unpacked'
    _write(root, STRAYS)
    assert outward_lines(root) == [f'{rel}:1: {LEAK}'
                                   for rel in ('NOTES.md', 'README.md', 'docs/stray.md')]


# One citation of each kind the checks resolve -- a `Targets:` file and glob, a path, a
# `file.py::name`, a repository URL to a directory, and a navigation entry -- and the files
# they name.
CITING = {
    'docs/a.md': ('Targets: `src/wavebit/x.py`, `src/wavebit/*.py`. See `docs/b.md`, `docs/b.py::f` and '
                  'https://github.com/fieldalgo/wavebit/tree/develop/docs/sub.'),
    'docs/_toc.yml': '- file: b',
}
CITED = {
    'src/wavebit/x.py': 'x = 1',
    'docs/b.md': 'b',
    'docs/b.py': 'def f():\n    pass',
    'docs/sub/c.md': 'c',
}


@needs_git
@pytest.mark.parametrize('state', ['untracked', 'tracked', 'tracked, then deleted'])
def test_a_citation_resolves_only_to_what_a_clone_gets(tmp_path, state):
    """Every citation check resolves against tracked files that are present, and only those.

    Untracked, the cited files are on disk and a clone does not get them. Tracked and then
    deleted, git still lists them and the next commit drops them; the empty `docs/sub/` stays
    on disk and names nothing.
    """
    _write(tmp_path, {**CITING, **CITED})
    _init(tmp_path)
    _git(tmp_path, 'add', *CITING)
    if state != 'untracked':
        _git(tmp_path, 'add', *CITED)
    if state == 'tracked, then deleted':
        for rel in CITED:
            (tmp_path / rel).unlink()
    found = (missing_targets(CITING['docs/a.md'], tmp_path), missing_cited_paths(tmp_path),
             missing_symbols(tmp_path), dangling_repo_urls(tmp_path), missing_toc_pages(tmp_path))
    assert found == (([],) * 5 if state == 'tracked' else (
        ['src/wavebit/*.py', 'src/wavebit/x.py'],
        ['docs/a.md: docs/b.md', 'docs/a.md: src/wavebit/x.py'],
        ['docs/a.md: no such file: docs/b.py::f'],
        ['docs/a.md:1: docs/sub'],
        ['b']))


@needs_git
def test_only_a_tracked_document_has_its_targets_checked(tmp_path):
    """The documents whose `Targets:` are checked come from the tracked set too.

    `docs_with_targets` decides which documents get a test case at all, so read from the disk
    it would give an untracked draft in `docs/` a case of its own, failing or passing on a
    document no clone has.
    """
    _write(tmp_path, {**CITING, 'docs/draft.md': CITING['docs/a.md']})
    _init(tmp_path)
    _git(tmp_path, 'add', *CITING)
    assert [p.relative_to(tmp_path).as_posix() for p, _ in docs_with_targets(tmp_path)] == [
        'docs/a.md']


@needs_git
def test_the_self_tests_neither_reach_nor_read_the_git_running_them(tmp_path, monkeypatch):
    """What the machine's git carries does not cross into a temporary repository, either way.

    A pre-commit hook exports GIT_INDEX_FILE, which would stage the temporary repository's files
    into the commit being made. And the user's settings would refuse the `git add`. Of a file
    they list: an ignore file in its default place, or the `info/exclude` of the template
    directory GIT_TEMPLATE_DIR names -- how global hooks are installed. Of a file with LF
    endings: `core.safecrlf` beside `core.autocrlf`, in the global file, in the system file, or
    in the GIT_CONFIG_PARAMETERS a hook inherits from `git -c ... commit`.
    """
    outer_index = tmp_path / 'outer-index'
    monkeypatch.setenv('GIT_INDEX_FILE', str(outer_index))
    _write(tmp_path, {'config/git/ignore': '\n'.join(STRAYS),
                      'config/git/config': '[core]\n\tautocrlf = true\n\tsafecrlf = true',
                      'template/info/exclude': '\n'.join(STRAYS)})
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('GIT_CONFIG_SYSTEM', str(tmp_path / 'config' / 'git' / 'config'))
    monkeypatch.setenv('GIT_CONFIG_PARAMETERS', "'core.autocrlf'='true' 'core.safecrlf'='true'")
    monkeypatch.setenv('GIT_TEMPLATE_DIR', str(tmp_path / 'template'))
    root = tmp_path / 'repo'
    _write(root, STRAYS)
    _init(root)
    _git(root, 'add', 'README.md', 'CLAUDE.md')
    assert outward_lines(root) == [f'README.md:1: {LEAK}']
    assert not outer_index.exists()


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


def missing_symbols(root=ROOT):
    """Each `file.py::name` in a shipped file that names no file, or no such definition."""
    bad = []
    for p in published_files(root):
        rel = p.relative_to(root).as_posix()
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
            sources = _sources_named(path, rel, root)
            if not sources:
                bad.append(f'{rel}: no such file: {path}::{symbol}')
                continue
            if not any(_defines(src, symbol) for src in sources):
                bad.append(f'{rel}: no such symbol: {path}::{symbol}')
    return sorted(set(bad))


def test_every_symbol_a_document_names_exists():
    """`file.py::name` has to name a real definition, not only a real file.

    A stale one reads exactly like a live one and sends the reader looking for a function that
    was renamed out from under the sentence.
    """
    bad = missing_symbols()
    assert not bad, (
        'these name symbols that do not exist:\n  ' + '\n  '.join(bad) +
        '\n\nA `file.py::name` is a claim about a definition. Renaming the definition without '
        'the sentence leaves prose that typesets perfectly and points nowhere.')


def _sources_named(path, citing, root=ROOT):
    """Every file the citation could mean, as text. Same three bases as the path check, plus
    a basename match, because `factorization.py::x` is how this repo writes a `_core` file.
    `citing` is the repository-relative path of the file that cites it."""
    files = tree(root)
    out = [c for c in _citation_candidates(path, citing, root) if c in files]
    if not out:
        name = posixpath.basename(path)
        out = [q for q in sorted(files)
               if posixpath.basename(q) == name and q.endswith(path)
               and '__pycache__' not in q.split('/')]
    return [(root / q).read_text(encoding='utf-8', errors='ignore') for q in out]


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
    r'((?:\.[A-Za-z0-9_]|[A-Za-z0-9_])[A-Za-z0-9._/-]*)')


@functools.lru_cache(maxsize=None)
def _historical_repo_links(root=ROOT):
    """Exact pinned URLs whose original Git objects were verified at import time."""
    name = 'docs/development/learning-provenance/historical-links.json'
    if name not in tree(root):
        return frozenset()
    rows = json.loads((root / name).read_text())['links']
    for row in rows:
        assert re.fullmatch(r'[0-9a-f]{40}', row['commit'])
        assert re.fullmatch(r'[0-9a-f]{40}', row['git_blob'])
        assert re.fullmatch(r'[0-9a-f]{64}', row['sha256'])
        match = re.fullmatch(r'https://github\.com/fieldalgo/wavebit/blob/([0-9a-f]{7,40})/(.+)', row['url'])
        assert match and row['commit'].startswith(match[1]) and row['path'] == match[2]
    return frozenset(row['url'] for row in rows)


def dangling_repo_urls(root=ROOT):
    """Each github.com URL into this repository, in a shipped file, whose path is not tracked."""
    bad = []
    for p in published_files(root):
        rel = p.relative_to(root).as_posix()
        if rel in EXEMPT:
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for m in REPO_URL.finditer(line):
                cited = m.group(1).rstrip('/).,')
                historical = _historical_repo_links(root)
                exact_url = m.group(0).rstrip('/).,')
                if exact_url in historical:
                    continue
                if not in_tree(cited, root):
                    bad.append(f'{rel}:{n}: {cited}')
    return bad


def test_only_exact_verified_historical_urls_survive_a_source_move(tmp_path):
    """A pinned evidence URL is different from an arbitrary SHA or stale branch link."""
    base = 'https://github.com/fieldalgo/wavebit/blob/'
    verified = base + 'abcdef0/old/source.py'
    _write(tmp_path, {
        'docs/development/learning-provenance/historical-links.json': json.dumps({'links': [{
            'url': verified, 'commit': 'abcdef0' + '1' * 33,
            'path': 'old/source.py', 'git_blob': '2' * 40, 'sha256': '3' * 64}]}),
        'docs/page.md': '\n'.join((verified, base + 'abcdef0/old/other.py',
                                   base + 'develop/old/source.py')),
    })
    assert dangling_repo_urls(tmp_path) == [
        'docs/page.md:2: old/other.py', 'docs/page.md:3: old/source.py']


@needs_git
def test_hidden_repository_links_keep_dotted_refs_and_missing_targets_distinct(tmp_path):
    """A dot-prefixed repository path is not part of the preceding branch name."""
    base = 'https://github.com/fieldalgo/wavebit/'
    urls = [base + route + '/' + ref + '/' + target
            for route in ('blob', 'tree') for ref in ('develop', 'release.v1')
            for target in ('.github/workflows/tests.yml', '.gitignore', 'src/example.py')]
    urls += [base + 'blob/develop/.github/workflows/missing.yml',
             base + 'blob/release.v1/src/missing.py']
    _write(tmp_path, {'docs/page.md': '\n'.join(urls),
                      '.github/workflows/tests.yml': 'name: Tests',
                      '.gitignore': 'build/', 'src/example.py': 'value = 1'})
    _init(tmp_path)
    _git(tmp_path, 'add', 'docs/page.md', '.github/workflows/tests.yml', '.gitignore', 'src/example.py')
    assert dangling_repo_urls(tmp_path) == [
        'docs/page.md:13: .github/workflows/missing.yml',
        'docs/page.md:14: src/missing.py']


def test_repository_urls_name_something_in_the_tree():
    """A github.com URL into this repository has to point at a path that exists here.

    Branch links resolve against this checkout. Exact pinned historical links may resolve
    through the published provenance receipt, whose original Git objects were verified
    during import; an arbitrary SHA is not an exemption from the current-tree check.
    """
    bad = dangling_repo_urls()
    assert not bad, (
        'these URLs point into this repository at paths that are not in it, so they 404 for '
        'every reader and for nobody who can fix them:\n  ' + '\n  '.join(bad))


def docs_with_targets(root=ROOT):
    """Each document directly in `docs/` that names its `Targets`, with its text."""
    for rel in sorted(tree(root)):
        if (rel.startswith('docs/') and rel.endswith('.md')):
            text = (root / rel).read_text(encoding='utf-8')
            if 'Targets' in text:
                yield root / rel, text


# A repo-relative path in backticks. This catches the other half of the same mistake: not
# `/Users/someone/...` but `third_party/rcwa/ANALYSIS.md` or `dev/bench_jit.py` -- a path that
# looks local, resolves on the author's disk, and is simply absent from the clone. It reads as
# an in-repo reference, which makes it worse than an obvious outward one.
CITED_PATH = re.compile(r'`([A-Za-z0-9_][A-Za-z0-9._/-]*\.(?:md|py|json|npz|toml|yml|cfg))`')


@functools.lru_cache(maxsize=None)
def _relocation_metadata(root=ROOT):
    """Exact published aliases and preserved record identities, never prefix exemptions."""
    maps, records = {}, {}
    files = tree(root)
    for name in ('tools/relocations.json', 'docs/development/source-relocations.json',
                 'docs/development/document-relocations.json'):
        if name not in files:
            continue
        data = json.loads((root / name).read_text(encoding='utf-8'))
        maps.update(data.get('paths', {}))
        maps.update({m['source']: m['target'] for m in data.get('moves', [])})
        records.update({m['target']: old for old, m in data.get('immutable', {}).items()})
    return maps, records


def _citation_candidates(path, citing, root=ROOT):
    """Resolve live citations directly and immutable evidence through exact ledgers.

    Old paths in records stay auditable without rewriting historical bytes. The
    resulting files must still be tracked and present; live prose gets no alias.
    """
    bases = ('', 'src/wavebit', posixpath.dirname(citing))
    overlay = 'docs/locales/ko/pages/'
    if citing.startswith(overlay):
        bases += (posixpath.dirname('docs/' + citing.removeprefix(overlay)),)
    candidates = {posixpath.normpath(posixpath.join(base, path)) for base in bases}
    maps, records = _relocation_metadata(root)
    if citing not in records:
        return sorted(candidates)
    candidates.add(posixpath.normpath(posixpath.join('src', path)))
    # Relative paths in immutable records are relative to their original directory.
    candidates.add(posixpath.normpath(posixpath.join(posixpath.dirname(records[citing]), path)))
    return sorted(candidates | {maps[c] for c in candidates if c in maps})


def test_only_archived_citations_use_published_relocation_maps(tmp_path):
    """An old path in evidence remains auditable; the same stale live citation fails."""
    _write(tmp_path, {
        'tools/relocations.json': json.dumps({
            'paths': {'studies/old.py': 'studies/new.py',
                      'studies/results/relative.py': 'studies/new.py'},
            'immutable': {'studies/results/run.log': {'target': 'studies/case/records/run.log'}}}),
        'studies/new.py': 'value = 1',
        'studies/case/records/run.log': 'source `studies/old.py` and `./relative.py`',
        'studies/case/records/new.log': 'source `studies/old.py`',
        'docs/current.md': 'source `studies/old.py`',
    })
    assert missing_cited_paths(tmp_path) == [
        'docs/current.md: studies/old.py', 'studies/case/records/new.log: studies/old.py']


def missing_cited_paths(root=ROOT):
    """Each backticked path in a shipped file that resolves to nothing tracked."""
    bad = []
    for p in published_files(root):
        rel = p.relative_to(root).as_posix()
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
                # not an oversight. It does let a stale name through: `tests/fixtures/legacy_rcwa/README.md`
                # described `gen_goldens_analytic.py` as still present for months after
                # `ee70470` deleted it. Widening the rule was tried and rejected on the
                # numbers. Resolving a bare name *beside the citing file* flags 84 of 330
                # citations, almost all correct prose (`docs/` saying `boundary.py` for
                # `src/wavebit/rcwa/_engine/boundary.py`). Resolving it *anywhere in the tree* is far
                # better at 17, but 14 of those 17 are legitimate too: files of other projects
                # (GRCWA's `kbloch.py`, fmmax's `utils.py`), names the prose itself marks as
                # former (`golden_legacy_field.npz`), things deliberately deleted, and a script
                # that lives on an unmerged branch. Buying one real catch with fourteen
                # exemptions to write and maintain is a worse document than the one stale line.
                continue
            if any(in_tree(candidate, root)
                   for candidate in _citation_candidates(t, rel, root)):
                continue
            bad.append(f'{rel}: {t}')
    return sorted(set(bad))


def test_paths_cited_anywhere_are_in_the_repository():
    """Every backticked path in a shipped file resolves.

    Scoped to `docs/` at first, which is how `third_party/rcwa/ANALYSIS.md` survived twice in
    `tests/references/`: the rule was right and the check was looking in one place. A cited
    path is a promise the reader can follow it, and that promise is made wherever it is
    written.
    """
    bad = missing_cited_paths()
    assert not bad, (
        'these cite paths that are not in the repository:\n  ' + '\n  '.join(bad) +
        '\n\nA path that resolves only on the author\'s disk reads as an in-repo reference, '
        'which is the harder half of the same leak -- nothing about it looks wrong. State the '
        'substance instead of the address; see the contract in docs/README.md.')


# The licence files name this tree's files in plain text, not in backticks, so CITED_PATH never
# read them: LICENSE kept citing four `wavebit/_core/` files after 4272c7ce moved them under
# `src/wavebit/`, in the one file the public mirror must ship correct. A path into this package
# or its tests has to resolve. The other projects' files a notice names -- meent's
# `meent/integ/smm_util.py`, fmmax's `src/fmmax/utils.py` -- carry neither prefix.
LICENCE_PATH = re.compile(r'(?<![\w./-])((?:src/)?wavebit/[\w./-]*\w|tests/[\w./-]*\w)')


def missing_licence_paths(root=ROOT):
    """Each path into this tree that a top-level licence file names and a clone lacks."""
    bad = []
    for rel in sorted(tree(root)):
        if '/' in rel or not rel.startswith('LICENSE'):
            continue
        for m in LICENCE_PATH.finditer((root / rel).read_text(encoding='utf-8')):
            if not in_tree(m.group(1), root):
                bad.append(f'{rel}: {m.group(1)}')
    return bad


def test_every_path_a_licence_names_is_in_the_repository():
    bad = missing_licence_paths()
    assert not bad, ('the licence names files that are not in the repository:\n  '
                     + '\n  '.join(bad) + '\n\nA notice that points at a moved file attributes '
                     'nothing; name where the code lives now.')


@pytest.mark.parametrize('text,found', [
    ('gradient in wavebit/_core/backend.py is a port', 'wavebit/_core/backend.py'),
    ('the Cartesian shape path (src/wavebit/rcwa/_engine/rectilinear.py) through',
     'src/wavebit/rcwa/_engine/rectilinear.py'),
    ('the comparisons in tests/references/ call a copy', 'tests/references'),
    ('script, tests/references/interface/reti_aniso.m, lays', 'tests/references/interface/reti_aniso.m'),
    ('https://github.com/fieldalgo/wavebit/blob/develop/README.md', None),
    ("from that project's src/fmmax/utils.py.", None),
    ('meent/integ/smm_util.py and fieldwave/on_numpy/', None),
])
def test_the_licence_path_pattern_reads_this_tree_only(text, found):
    m = LICENCE_PATH.search(text)
    assert (m.group(1) if m else None) == found


def missing_targets(text, root=ROOT):
    """Each `wavebit/...py` a document's text names that is not in the tree."""
    missing = []
    for m in re.finditer(r'`(src/wavebit/[A-Za-z0-9_./*-]+\.py)', text):
        target = m.group(1)
        found = ([f.relative_to(root).as_posix() for f in root.glob(target)] if '*' in target
                 else [target])
        if not any(in_tree(f, root) for f in found):
            missing.append(target)
    return sorted(set(missing))


@pytest.mark.parametrize('doc', [p.relative_to(ROOT / 'docs').as_posix() for p, _ in docs_with_targets()])
def test_every_target_a_document_names_exists(doc):
    """A document explains named code; the names have to be real."""
    missing = missing_targets((ROOT / 'docs' / doc).read_text(encoding='utf-8'))
    assert not missing, (
        f'docs/{doc} explains files that do not exist: {missing}. '
        'Either the document is stale or the path is a typo; both make it decorative.')


def missing_toc_pages(root=ROOT):
    """Each `_toc.yml` entry with no page behind it."""
    toc = (root / 'docs' / '_toc.yml').read_text(encoding='utf-8')
    return sorted({name for name in re.findall(r'-\s*file:\s*(\S+)', toc)
                   if not any(in_tree(f'docs/{name}{ext}', root)
                              for ext in ('.md', '.ipynb', '.rst'))})


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
    missing = missing_toc_pages()
    assert not missing, (
        f'docs/_toc.yml lists pages that do not exist: {missing}. '
        'The site build resolves these, so a stale entry is a missing page rather than a '
        'broken link -- add the document, or drop the entry.')


def test_private_lab_originals_are_not_in_the_public_tree():
    prefix = 'docs/development/learning-provenance/'
    private_files = {prefix + 'supplementary-notes.json', prefix + 'additional-originals.json',
                     'docs/development/research-note-archive.md'}
    assert not private_files & tree()
    assert not any(name.startswith(prefix + 'source-evidence/') for name in tree())


def test_public_documentation_does_not_link_private_conversation_artifacts():
    bad = []
    for name in tree():
        if not name.startswith('docs/'):
            continue
        path = ROOT / name
        if path.suffix in ('.md', '.rst', '.json', '.html', '.yml', '.ipynb', '.po'):
            if re.search(r'https?://claude\.ai/(?:code/artifact|share|chat)/',
                         path.read_text(encoding='utf-8')):
                bad.append(name)
    assert not bad, f'Private conversation links belong in the lab: {bad}'


@pytest.mark.parametrize('suffix', ['.md', '.py'])
def test_private_notes_do_not_receive_public_path_exemptions(tmp_path, suffix):
    name = 'docs/research-notes/new-note' + suffix
    _write(tmp_path, {name: 'an unreviewed path /home/someone/private/note.md'})
    assert outward_lines(tmp_path) == [
        name + ':1: an unreviewed path /home/someone/private/note.md']
