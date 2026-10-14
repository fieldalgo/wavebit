"""Resolve repository relocations without rewriting historical experiment records.

This module is repository tooling, never an installed wavebit dependency. The
ledger records a file move, not a new experiment or an updated historical hash.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _ledger():
    return json.loads((_ROOT / 'tools' / 'relocations.json').read_text())


@lru_cache(maxsize=1)
def _source_paths():
    """Final engine relocations, separate from immutable experiment records."""
    return json.loads((_ROOT / 'docs/development/source-relocations.json').read_text())['paths']


def resolve_repository_path(value):
    """Resolve a historical repository path; leave external/current paths alone."""
    path = Path(value)
    absolute = path if path.is_absolute() else _ROOT / path
    try:
        relative = absolute.relative_to(_ROOT).as_posix()
    except ValueError:
        return absolute
    ledger = _ledger()
    if relative in ledger['paths']:
        return _ROOT / ledger['paths'][relative]
    source = relative if relative.startswith('src/') else 'src/' + relative
    if source in _source_paths():
        return _ROOT / _source_paths()[source]
    for old, new in sorted(ledger['directory_prefixes'].items(), key=lambda pair: -len(pair[0])):
        if relative == old or relative.startswith(old + '/'):
            return _ROOT / new / relative[len(old):].lstrip('/')
    if relative == 'studies/results' or relative.startswith('studies/results/'):
        return _ROOT / 'artifacts/studies' / relative.removeprefix('studies/results').lstrip('/')
    return absolute


class RepositoryPath(type(Path())):
    """A repository-only path retaining old identity through incremental joins.

    A historical root such as studies/results spans several new question
    folders. Resolve when its filename is joined, not by inventing one new root.
    Only construction and joins resolve: ``/``, ``joinpath``, ``key / path``
    and ``absolute()``, which joins the working directory. A path pathlib
    derives from another (parent, with_name, iterdir and glob children,
    relative_to's argument) keeps the string it is given, as its identity.
    """
    if sys.version_info >= (3, 12):
        def __init__(self, *parts):
            self._historical = Path(*parts)
            resolved = (resolve_repository_path(self._historical)
                        if self._historical.is_absolute() else self._historical)
            super().__init__(resolved)

        def with_segments(self, *pathsegments):
            # Every derived path comes through here. Built through __init__ it
            # would resolve, and pathlib would then overwrite its string with the
            # unresolved one: str() and Path() of one path disagreed, and
            # relative_to(<old location>) resolved its argument and passed for a
            # path that is not under it.
            path = super().__new__(type(self))
            super(RepositoryPath, path).__init__(*pathsegments)
            path._historical = Path(*pathsegments)
            return path
    else:
        # Python 3.11 (fwref, the env the fmmax generators run in) has no
        # with_segments(): Path.__new__ builds every instance in _from_parts and
        # never passes the parts to __init__, and derived paths come from
        # _from_parts (absolute, resolve) or _from_parsed_parts (parent,
        # with_name, iterdir and glob children). Construction resolves here, in
        # __new__; both classmethods keep their string, as with_segments does.
        def __new__(cls, *parts):
            historical = Path(*parts)
            resolved = (resolve_repository_path(historical)
                        if historical.is_absolute() else historical)
            self = super().__new__(cls, resolved)
            self._historical = historical
            return self

        @classmethod
        def _from_parts(cls, args):
            self = super()._from_parts(args)
            self._historical = Path(self)
            return self

        @classmethod
        def _from_parsed_parts(cls, drv, root, parts):
            self = super()._from_parsed_parts(drv, root, parts)
            self._historical = Path(self)
            return self

    def __truediv__(self, key):
        return type(self)(self._historical / key)

    def __rtruediv__(self, key):
        # A join, so it resolves; pathlib would build it as a derived path.
        try:
            return type(self)(key, self)
        except TypeError:
            return NotImplemented

    def absolute(self):
        # The working directory joined with this path, so it resolves too.
        return self if self.is_absolute() else type(self)(super().absolute())

    def joinpath(self, *parts):
        return type(self)(self._historical.joinpath(*parts))

    def relative_to(self, other, /, *parts, **kwargs):
        try:
            return super().relative_to(other, *parts, **kwargs)
        except ValueError:
            # A historical evidence root can span several current folders. Only
            # two ledger-aware paths may fall back to their original ancestry;
            # ordinary Path callers retain the normal containment guarantee.
            if not isinstance(other, RepositoryPath) or parts or kwargs.get('walk_up'):
                raise
            return self._historical.relative_to(other._historical)

    def rglob(self, pattern, **kwargs):
        """Recurse over both current files and the historical ledger roster."""
        yield from self.glob('**/' + str(pattern), **kwargs)

    def glob(self, pattern, **kwargs):
        yielded = set()
        for path in super().glob(pattern, **kwargs):
            yielded.add(str(path))
            yield path
        try:
            prefix = self._historical.relative_to(_ROOT).as_posix()
        except ValueError:
            return
        # Historical evidence enumerators retain the old roster even though the
        # records now live in several question folders. New/current scans should
        # explicitly use their populated current directories.
        ledger = _ledger()
        candidates = ledger['paths'] | ledger['directory_prefixes']
        for old, new in candidates.items():
            original = Path(old)
            try:
                relative = original.relative_to(prefix)
            except ValueError:
                continue
            matches = relative.match(pattern)
            if pattern.startswith('**/'):
                matches = matches or relative.match(pattern[3:])
            if matches and ('**' in pattern or len(relative.parts) == len(Path(pattern).parts)):
                actual = _ROOT / new
                if actual.exists() and str(actual) not in yielded:
                    yielded.add(str(actual))
                    yield RepositoryPath(_ROOT / old)


def repository_path(*parts):
    """Build a relocatable path from a repository-relative historical identity."""
    return RepositoryPath(_ROOT.joinpath(*parts))


def repository_root():
    """Return the actual checkout root, carrying historical join compatibility."""
    return RepositoryPath(_ROOT)


def relocate_nodeid(nodeid):
    """Translate only a collected file path; preserve all parameter/test suffixes."""
    path, separator, suffix = nodeid.partition('::')
    return _ledger()['paths'].get(path, path) + separator + suffix
