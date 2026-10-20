# Third-party software, data and references

Wavebit's own code is licensed `Apache-2.0 AND MIT`: Apache-2.0 for this project's
work, with the MIT notices of the code it derives from kept in
[LICENSE](https://github.com/fieldalgo/wavebit/blob/main/LICENSE),
`LICENSE.meent` and `LICENSE.fmmax`.
[NOTICE](https://github.com/fieldalgo/wavebit/blob/main/NOTICE) summarises the
attributions. All four files ship in the sdist and in the wheel's
`.dist-info/licenses/`, and `tools/run_distribution_smoke.py` checks that both
carry them unchanged. This page lists every third-party component
the repository uses, how it is used and under which terms. Wavebit's license
does not relicense any of them.

The machine-readable inventory is `tools/third_party_licenses.json`.
`tools/check_licenses.py` compares it with the installed metadata of an
environment; see [Automated check](#automated-check).

## How a component is used

| Use | Meaning |
| --- | --- |
| runtime | Base requirement of `wavebit`, imported at run time. Installed separately by pip, never bundled. |
| optional | Reached through a Wavebit extra (`jax`, `pytorch`, `contour`, `plot`, `docs`). Same terms as runtime. |
| tool | Test runner, linter, build or upload tool; not required by Wavebit. |
| external tool | Separately installed program or interpreter whose output numbers are frozen into `tests/references/`. Never a Wavebit dependency. |
| bundled | Code or data inside this repository that came from elsewhere. |

Policy:

- Runtime dependencies are permissive.
- LGPL or MPL is accepted for optional or tool components that are installed
  separately by pip, unmodified and never bundled; the inventory records the
  review for each.
- LGPL or MPL is never a runtime requirement, and Wavebit's code never loads
  such a library directly (ctypes, cffi or a compiled extension); it uses the
  component's public API only.
- GPL code is never imported, linked or bundled by what Wavebit ships (the
  `src/wavebit/` package in the sdist and wheel), and is never a requirement.
  Unshipped test, study and reference scripts may run a GPL tool such as S4 or
  Meep, also in the same process as Wavebit, provided they stay outside the
  sdist and wheel and the installation and the default test run work without
  the tool (tests that need it skip). GPL source is not read for implementation.
- AGPL and SSPL are not used anywhere.

## Python distributions

Version floors are in `pyproject.toml`; the table names what CI installs.

### Runtime

| Distribution | SPDX | Notes |
| --- | --- | --- |
| `numpy` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Wheels bundle OpenBLAS (BSD-3-Clause) and, on Linux and Windows, the GCC Fortran runtime (GPL-3.0-or-later WITH GCC-exception-3.1, which permits this use) |
| `scipy` | BSD-3-Clause | Same bundled libraries as numpy |
| `array-api-compat` | MIT | |

### Optional extras

| Distribution | Extra | SPDX | Notes |
| --- | --- | --- | --- |
| `jax`, `jaxlib`, `optax`, `absl-py`, `ml-dtypes` | `jax` | Apache-2.0 | |
| `opt-einsum` | `jax` | MIT | |
| `tqdm` | `jax`, `pytorch` | MPL-2.0 AND MIT | Progress bars only. Installed separately by pip, unmodified, never bundled |
| `torch` | `pytorch` | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | CPU wheel in CI |
| `sympy`, `mpmath`, `networkx`, `fsspec` | torch | BSD-3-Clause | |
| `filelock` | torch | MIT | |
| `jinja2`, `markupsafe` | torch, sphinx | BSD-3-Clause | |
| `typing-extensions` | torch, others | PSF-2.0 | |
| `setuptools` | torch | MIT | |
| `contourpy` | `contour`, matplotlib | BSD-3-Clause | |
| `matplotlib` | `plot` | LicenseRef-Matplotlib | PSF-based permissive license agreement |
| `cycler`, `kiwisolver` | matplotlib | BSD-3-Clause | |
| `fonttools`, `pyparsing`, `six` | matplotlib | MIT | |
| `pillow` | matplotlib | MIT-CMU | |
| `packaging` | matplotlib, others | Apache-2.0 OR BSD-2-Clause | |
| `python-dateutil` | matplotlib | Apache-2.0 AND BSD-3-Clause | |
| `sphinx`, `sphinxcontrib-mermaid`, `sphinxcontrib-applehelp`, `sphinxcontrib-devhelp`, `sphinxcontrib-htmlhelp`, `sphinxcontrib-jsmath`, `sphinxcontrib-qthelp`, `sphinxcontrib-serializinghtml` | `docs` | BSD-2-Clause | |
| `pydata-sphinx-theme`, `accessible-pygments`, `babel`, `snowballstemmer`, `alabaster`, `click` | `docs` | BSD-3-Clause | |
| `myst-parser`, `mdit-py-plugins`, `markdown-it-py`, `mdurl`, `sphinx-design`, `sphinx-external-toc`, `sphinx-multitoc-numbering`, `beautifulsoup4`, `soupsieve`, `pyyaml`, `imagesize`, `charset-normalizer`, `urllib3` | `docs` | MIT | |
| `docutils` | `docs`, twine | LicenseRef-Public-Domain AND BSD-2-Clause AND 0BSD | Its GPL classifier names files outside the wheel; see the inventory's review |
| `roman-numerals` | sphinx | 0BSD OR CC0-1.0 | |
| `pygments` | sphinx, pytest | BSD-2-Clause | |
| `requests` | sphinx, twine | Apache-2.0 | |
| `idna` | requests | BSD-3-Clause | |
| `certifi` | requests | MPL-2.0 | Unmodified CA bundle, installed by pip; never imported or redistributed by Wavebit |

### Tools

| Distribution | Reached through | SPDX |
| --- | --- | --- |
| `pytest`, `pytest-timeout`, `pytest-xdist`, `execnet`, `iniconfig`, `pluggy` | CI test jobs | MIT |
| `colorama` | pytest on Windows | BSD-3-Clause |
| `ruff` | lint job | MIT |
| `build`, `pyproject-hooks` | distribution job | MIT |
| `twine`, `id`, `readme-renderer`, `requests-toolbelt`, `rfc3986` | distribution job | Apache-2.0 |
| `nh3`, `rich`, `keyring`, `jaraco-classes`, `jaraco-context`, `jaraco-functools`, `more-itertools`, `jeepney`, `pip` | twine, environment | MIT |
| `secretstorage`, `cryptography`, `cffi`, `pycparser` | keyring | BSD-3-Clause, Apache-2.0 OR BSD-3-Clause, MIT-0, BSD-3-Clause |

## External reference tools

None of these is a Wavebit dependency or part of the sdist and wheel. Each is
installed separately; the numbers they produce are frozen into
`tests/references/`, and CI, which installs none of them, reads the numbers. [Reference cases](https://github.com/fieldalgo/wavebit/blob/main/tests/references/README.md)
describes each comparison.

| Tool | SPDX | How it is used |
| --- | --- | --- |
| `Meep` | GPL-2.0-or-later | FDTD reference through its documented Python API; the generators and bridges under `tests/references/` run by hand |
| `S4` | GPL-2.0-or-later | RCWA reference; the `gen_goldens_s4*.py` generators run by hand, and a few unshipped tests and studies call it directly and skip when it is absent |
| `Octave` | GPL-3.0-or-later | Runs RETICOLO as a separate process. Our `.m` wrappers run inside Octave and never call Wavebit |
| `RETICOLO` | CC-BY-4.0 | RCWA reference run inside Octave; not shipped. Attribution and history in [LICENSE](https://github.com/fieldalgo/wavebit/blob/main/LICENSE) |
| `oct2py` | MIT | Python driver for the Octave process |
| `fmmax` | MIT | RCWA reference in a separate environment |
| `FDTDX` | MIT | FDTD reference in a separate environment |
| `PyMoosh` | MIT | Multilayer reference in a throwaway environment |
| `NGSolve` | LGPL-2.1-only | Optional FEM reference for NVF research studies. Its installed binaries also carry GPL-2.0-or-later UMFPACK code, so that environment is never redistributed |

## Code and data in this repository

| Component | SPDX | Where | Notes |
| --- | --- | --- | --- |
| `meent` | MIT | parts of `src/wavebit/rcwa/` | Origin of part of the RCWA code; notice in `LICENSE` and `LICENSE.meent` |
| `fmmax-eig-bwd` | MIT | `src/wavebit/_common/backend.py` | Eigendecomposition gradient ported from fmmax; notice in `LICENSE` and `LICENSE.fmmax` |
| `refractiveindex-info` | CC0-1.0 | `src/wavebit/data/refractiveindex/` | Refractive-index tables for Si, SiO2, Si3N4 and Al2O3 shipped in the wheel and sdist, from the public-domain refractiveindex.info database; the directory's README cites each table's publications and the retrieval date |

## Automated check

```sh
python tools/check_licenses.py                    # every installed distribution
python tools/check_licenses.py --reachable-only   # only what Wavebit's requirements reach
```

The check reads installed metadata with `importlib.metadata`. It fails when an
installed distribution has no reviewed entry, when its declared license differs
from the entry, when a component's license is not allowed for the way Wavebit
reaches it, when Wavebit's package code imports an external tool, or when this
page or its Korean translation does not name an entry. CI runs it in the full
backends environment and in the docs environment;
`tests/api/test_third_party_licenses.py` runs the reachable-only form in every
test environment.

Before adding a dependency, add its reviewed entry to the inventory and a row
here and in the Korean page.
