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
        'numpy>=2.2',
        'scipy>=1.15',
        'array-api-compat>=1.15',
    ],
    extras_require=extras,
    # 3.12 is what CI runs and what jax 0.11 itself requires. Widen this only
    # together with a CI matrix that proves the older version works.
    python_requires='>=3.12',
    description=(
        "Electromagnetic simulation (RCWA) & optimization package in Python"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={
        'fieldwave': ['nk_data/filmetrics/*.txt', 'nk_data/matlab/*.mat'],
    },
)
