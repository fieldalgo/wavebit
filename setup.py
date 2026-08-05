import os

from setuptools import setup, find_packages

# Floors are the versions the suite is actually verified against, not the
# oldest that might happen to work. The jax floor is a hard requirement, not a
# preference: _core/backend.py calls jax.lax.linalg.eig(enable_eigvec_derivs=True),
# which does not exist before 0.11 — an older jax installs fine and then fails
# at the first eigendecomposition.
extras = {
    'jax': ['jax>=0.11',
            'jaxlib>=0.11',
            'optax>=0.2',
            'tqdm>=4.67',
            ],
    'pytorch': ['torch>=2.7',
                'tqdm>=4.67',
                ],
}
# Read in README.md for our long_description
cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='fieldwave',
    version='0.1.0',
    url='https://github.com/fieldalgo/fieldwave',
    author='FieldAlgo',
    author_email='dev@fieldalgo.com',
    license='Apache-2.0',
    packages=['fieldwave'] + find_packages(include=['fieldwave.*']),
    install_requires=[
        # 2.3.1, not 2.2, because of a NumPy bug in the complex `matmul` FP-status check:
        # on 2.2.x and 2.3.0 an ordinary complex matrix product raises a cascade of
        # `RuntimeWarning: overflow / divide by zero / invalid value encountered in matmul`
        # on values nowhere near overflow -- measured here on a convolution matrix with
        # condition number 12 whose largest entry was 2.1. A single 2D solve emitted 456 of
        # them. The values were never affected (energy conserved to ~1e-14, and the results
        # are bit-identical across versions); only the reporting was. 2.3.1 fixes it, so the
        # floor is the fix rather than a suppression layer inside the library.
        'numpy>=2.3.1',
        'scipy>=1.15',
        'array-api-compat>=1.15',
    ],
    extras_require=extras,
    # Driven by the base install alone (numpy, scipy and array-api-compat all
    # declare >=3.10), not by the optional backends: jax 0.11 needs 3.12, but
    # pip enforces that from jax's own metadata when the extra is requested.
    # Encoding it here would refuse the install to numpy-only users who never
    # asked for jax. The `minimal` CI job runs on every version listed here.
    python_requires='>=3.10',
    description=(
        "Electromagnetic simulation (RCWA) & optimization package in Python"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={
        'fieldwave': ['nk_data/filmetrics/*.txt', 'nk_data/matlab/*.mat'],
    },
)
