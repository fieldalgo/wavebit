# Documentation extensions

`wavebit_i18n.py` overlays complete Korean Markdown chapters, selects localized teaching assets and labels untranslated pages. It does not translate text or execute numerical examples.

`wavebit_pycode.py` works around a Sphinx source-analyzer failure on a `**`-unpacked dict in a decorator argument, which otherwise leaves `wavebit.observations.contracts` listed in the module index without a `[source]` page. It patches nothing once Sphinx unparses the construct itself.
