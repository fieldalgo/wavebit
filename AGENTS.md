# Working on Wavebit

Read [CONTRIBUTING.md](CONTRIBUTING.md) for repository policy and change placement.
Use the [source checkout guide](docs/development/source-checkout.md) for setup,
the [source map](docs/development/source-map.md) to locate implementation and tests,
and the [verification guide](docs/development/verification.md) for required checks.

- Implementation lives in `src/wavebit/`; regression tests live in `tests/`.
- Keep solver changes backend-neutral and follow the shared sign/branch conventions
  in `src/wavebit/_common/conventions.py`. Preserve numerical acceptance criteria.
- Use one editable environment per checkout and confirm the imported source path.
- Update English and Korean documentation under `docs/` together.

From the repository root in a configured environment:

```sh
python tools/run_verification.py --profile quick
```

This is a smoke check. Select the relevant full/research checks from the verification
and test guides; it does not replace required CI. Follow CONTRIBUTING for PRs and releases.
