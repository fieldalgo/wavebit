"""The suite's one collection rule: research tests run when they are asked for.

`research` (registered in `pyproject.toml`) marks the tests whose subject is NVF research code
-- the experimental conforming module `wavebit/_core/conforming/`, the NVF study instruments
under `studies/`, and the unused passive closure kernel. They are asserted like everything else
here. They are left out of the default run because they study that experimental code rather
than the ordinary solver and because they are where the suite's time goes. They are a
development aid for that code: work on it runs them, work on the rest of the package does not.
Many drive the ordinary engine (`solve_conv`, `ConvMatrices`, `parametric`), and so do the
`studies/nvf_*` models they import, so a product change can break one; it is repaired when
research work next runs the tier. `tests/README.md` says which files are in the tier and why.

A run that names only directories -- `pytest tests`, which is what CI runs -- deselects them.
They run when asked for, in any of three ways:

    pytest -m research tests                      # the research tier alone
    pytest -m "research or not research" tests    # everything
    pytest tests/test_nvf_pair_galerkin.py         # a file named on the command line, whole

Any explicit `-m` decides on its own, so this hook only supplies the default. Naming a file
counts as asking for it: the published write-ups under `docs/` reproduce their numbers with
`pytest <file>`, and those commands must keep running what they name.
"""
import pathlib


def _named_files(config):
    here = pathlib.Path(config.invocation_params.dir)
    named = set()
    for arg in config.args:
        path = (here / str(arg).split('::', 1)[0]).resolve()
        if path.is_file():
            named.add(path)
    return named


def pytest_collection_modifyitems(config, items):
    if config.option.markexpr:
        return
    named = _named_files(config)
    selected, deselected = [], []
    for item in items:
        asked = pathlib.Path(item.path).resolve() in named
        research = item.get_closest_marker('research') is not None
        (deselected if research and not asked else selected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
