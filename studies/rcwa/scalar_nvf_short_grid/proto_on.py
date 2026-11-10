"""pytest plugin that turns the prototype's switch on for a whole test run (section (d)).

    PYTHONPATH=studies/rcwa/scalar_nvf_short_grid python -m pytest -p proto_on -m "not slow" <file>
"""


def pytest_configure(config):
    from wavebit.rcwa._engine import factorization
    factorization._NVF_SHORT_GRID_INTERPOLANT = True
