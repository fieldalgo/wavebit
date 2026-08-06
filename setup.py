import os

from setuptools import setup, find_packages

# Floors are the versions the suite is actually verified against, not the
# oldest that might happen to work. That rule is applied literally to the numerical
# stack -- numpy, scipy, array-api-compat, jax, torch -- whose exact versions the
# frozen goldens are pinned against; each floor below is a version the full suite has
# been run on, and nothing between it and the previous floor is claimed. Loosening a
# floor later is safe and cheap; tightening one after a release is not, so they start
# strict. `tqdm` and `optax` are exempt: neither touches a number the goldens cover.
#
# The jax floor is a hard requirement on top of that, not just a verification record:
# _core/backend.py calls jax.lax.linalg.eig(enable_eigvec_derivs=True), which does not
# exist before 0.11 — an older jax installs fine and then fails at the first
# eigendecomposition.
extras = {
    'jax': ['jax>=0.11',
            'jaxlib>=0.11',
            'optax>=0.2',
            'tqdm>=4.67',
            ],
    # 2.13, not 2.7, because 2.7 cannot compile this solver: `torch.compile` dies with
    # `InternalTorchDynamoError` on every path, from four separate PyTorch defects -- a
    # `torch._numpy` shim missing `np._CopyMode`, an Inductor meta kernel that declares
    # row-major eigenvectors where LAPACK returns column-major, a Dynamo resume-bytecode
    # failure on cell variables, and a complex-view stride error in the backward pass.
    # All four are gone by 2.13, where the suite is 611 green. Eager and autograd work on
    # 2.7, so this floor is about matching the version the suite is verified against, not
    # about enabling `torch.compile` -- which is not worth using either way; see
    # docs/torch-compile.md. Versions between 2.8 and 2.12 were not tested.
    'pytorch': ['torch>=2.13',
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
        # History behind this one, which the floor now sits well above: NumPy 2.2.x and
        # 2.3.0 have a bug in the complex `matmul` FP-status check, where an ordinary
        # complex matrix product raises a cascade of `RuntimeWarning: overflow / divide by
        # zero / invalid value encountered in matmul` on values nowhere near overflow --
        # measured here on a convolution matrix with condition number 12 whose largest
        # entry was 2.1. A single 2D solve emitted 456 of them. The values were never
        # affected (energy conserved to ~1e-14, and the results are bit-identical across
        # versions); only the reporting was. 2.3.1 fixed it, and was the floor until the
        # floors were aligned with what is actually verified.
        'numpy>=2.5.1',
        'scipy>=1.18',
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
