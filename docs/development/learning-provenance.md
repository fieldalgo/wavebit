# Learning source provenance

The RCWA fundamentals and NVF courses were imported from the editable Markdown sources of the separate lab repository, `wavebit-sphinx-guide/source`, at base commit `89bff30b31dd37455d9c3292ecf030522fd36b05`. Exact source hashes include uncommitted author changes present at import time; the lab checkout was not modified.

The import covers forty complete chapter and index pages, one hundred figures, fifteen companion Python examples, and twelve historical numerical records. English translations form the canonical pages. Full Korean source text is retained as a same-route language overlay, with current API corrections distinguished from historical numerical evidence.

- [Completed translation receipt](learning-provenance/translation-receipt.json): all forty full English bodies and Korean overlays.
- [English SVG text translations](learning-provenance/text-figure-translations.json) and [glyph-figure translations](learning-provenance/glyph-figure-translations.json).
- [English metal plots from preserved data](learning-provenance/metal-figures.json) and [original English composite plots](learning-provenance/metal-original-plots.json).
- [Verified pinned historical links](learning-provenance/historical-links.json): original paths at their original commits.
- [Import receipt](learning-provenance/import-receipt.json): exact source hashes, original Git status, destination and decision for every inventory entry.
- [Original chapter migration inventory](learning-provenance/migration.json).
- [Original documentation verification](learning-provenance/verification.json).
- [Original example verification](learning-provenance/example-verification.json).
- [Original direct-wavevector verification](learning-provenance/direct-k-verification.json).
- [Original toolchain](learning-provenance/toolchain.json).
- [Preserved license notices](learning-licenses.md).

## Meaning of the records

Original measurements, logs and numerical figure data are preserved rather than rerun or relabeled. The old example verification covers fourteen scripts and does not certify the newer boundary-power example. The old documentation verification describes its own earlier lesson count and build, not this integrated site. Translated labels or relocated links do not change a numerical record.

## Publication scope

The public collection contains the maintained course, reviewed figures and examples,
and the explicitly identified historical numerical records. Raw author notes, draft
designs, session records, exploratory notebooks and independent prototype bundles
remain in the separate private lab repository. They are not public downloads.

The import receipt covers the material retained in this repository. Source hashes
identify the inputs to an edited chapter or figure; they do not require publishing
the underlying private research archive. A reader can use the public documentation
without access to that archive or to an author's conversation links.

## Translation and fallback

Korean chapter bodies live under `docs/locales/ko/pages` and are loaded only for Korean builds. Other pages use gettext where a catalog exists; otherwise the Korean site explicitly identifies the English fallback. Localized SVG and PNG labels are selected independently of the preserved numerical data. The language switch always retains the current route.
