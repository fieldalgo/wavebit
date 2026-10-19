# Documentation and translations

English is the canonical language. Navigation follows six reader tasks: Getting
started, Learn, User guide, API reference, Theory & validation, and Developer guide.
Source pages, docstrings and executable examples are maintained once.

## Build the English site

Activate the checkout's development environment first:

```sh
python -m pip install -e '.[docs]'
python -m sphinx -W -b html docs docs/_build/html
```

The build reads API docstrings from this checkout and refuses a different installed
copy. Output stays local; building does not publish the site.

## Build the bilingual site

The published artifact keeps English at its existing root URLs and adds Korean
under `ko/`. A language selector in the header opens the same page in the other
language. Its links are relative, so hosting under a project or version prefix
does not change them. Build both trees together with the explicit bilingual flag:

```sh
python -m sphinx -W -b html -D wavebit_bilingual=1 -d docs/_build/doctrees/en docs docs/_build/html
python -m sphinx -W -b html -D wavebit_bilingual=1 -D language=ko -d docs/_build/doctrees/ko docs docs/_build/html/ko
python -m http.server --directory docs/_build/html 8000
```

Open `http://localhost:8000/` or `http://localhost:8000/ko/`. Publish the entire
`docs/_build/html` directory to keep both language trees together. The docs CI
job builds both with warnings as errors; the deployment workflow uses that same
layout. Building or serving locally does not deploy anything.

The flag defaults to false. A standalone English build, or a standalone Korean
build without `-D wavebit_bilingual=1`, shows no links to an absent language tree.
Use separate doctree directories for the two languages, as above. To inspect
only Korean locally:

```sh
python -m sphinx -W -b html -D language=ko docs docs/_build/ko
```

## Contribute Korean translations

Sphinx gettext catalogs live under `docs/locales/ko/LC_MESSAGES/`. Untranslated
passages fall back to English on the Korean site, so every page remains usable.
The homepage and all forty imported course chapters have complete Korean Markdown
overlays under `docs/locales/ko/pages/`. Their source is loaded only in the Korean
build; English pages remain canonical. Maintained guide and developer pages may also provide
complete overlays — currently the [current implementation map](source-map.md),
[recorded verification](verification.md), [run records](../guide/run-records.md),
[exceptional points](../guide/rcwa/exceptional-points.md),
[how the FDTD solver models a problem](../guide/fdtd/model.md) and
[physical periodic-stack FDTD](../guide/fdtd/periodic-problems.md) — recorded separately from the
imported course translations in the language inventory. Other pages use gettext where available
and otherwise display an explicit English-fallback notice. Localized SVG and PNG
assets live under `docs/locales/ko/assets/` with the same canonical relative paths.
The source-read extension selects them without changing English content.

Extract current English messages for gettext-based pages with:

```sh
python -m sphinx -b gettext docs docs/_build/gettext
```

When English changes, update the corresponding `msgid` and review `msgstr` against
the new source. Preserve parameter names, code, math identifiers and links. Build
both languages before submitting a translation. A translation of an old solver
convention must not be presented as the current contract.

## Research notes and learning material

Keep examples in `examples/` or `tutorials/` and reference their canonical source.
Bring learning material into `docs/learn/` after its API, conventions, figures and
licensing have been checked against the current package. Preserve validation
provenance and distinguish historical results from current instructions.

Keep raw author notes, draft designs, session records, private conversation links and
unreviewed notebooks or prototypes in the separate private lab repository. Do not
attach them as public downloads. A published chapter should stand on its own and cite
reviewed evidence that readers can access. Preserve originals privately while editing
the public English explanation and optional Korean translation.

Build release artifacts into an empty output directory. Incremental Sphinx builds can
leave removed pages, source copies and downloads on disk; hiding navigation does not
remove those files from a deployment. Replace the previously served artifact with the
complete clean build after checking both languages.
