# Learning figure and evidence tools

These seven instruments were imported from the laboratory learning guide. The
source identities and hashes are recorded in the documentation import receipt.
They preserve the historical computations and Korean figure labels. They are
not run during documentation builds and do not regenerate numerical evidence.

The figure tools write to an explicit output path, or to
`artifacts/docs/learning/ko/`. `draw_power_chapter.py` reads the preserved
`docs/validation/learning-records/power-example.json` and accepts `--output-dir`.
The wavevector measurement tool uses the current checkout through the exact private
RCWA engine it measures (not a stable public interface); its default output is under `artifacts/docs/learning/`
and it refuses to overwrite the preserved learning-records directory. Existing
historical records are never silently relabeled as new measurements.

Use `check_site.py PATH_TO_BUILT_HTML` to check generated links and image alt
text. Its `--allow-remote-asset` option names each deliberate external resource.

`draw_metal_records.py --output NEW_DIRECTORY` renders the recorded metal-study
charts with English labels. Its measurement literals and calculations match the
laboratory source; it does not run a solver. It refuses a nonempty output folder.
Original Korean plots and the translation provenance are preserved in the docs.
