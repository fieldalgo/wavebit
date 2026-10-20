# 제3자 소프트웨어, 데이터와 레퍼런스

Wavebit 자체 코드의 라이선스는 `Apache-2.0 AND MIT`입니다. 이 프로젝트가 작성한 부분은
Apache-2.0이고, 파생 원본 코드의 MIT 고지는
[LICENSE](https://github.com/fieldalgo/wavebit/blob/main/LICENSE),
`LICENSE.meent`, `LICENSE.fmmax`에 보존합니다. 이 페이지는 저장소가 쓰는 모든
제3자 구성요소와 그 쓰임새, 적용 조건을 정리합니다. Wavebit의 라이선스가 이들의
라이선스를 바꾸지는 않습니다.

기계용 목록은 `tools/third_party_licenses.json`입니다. `tools/check_licenses.py`가
환경에 설치된 메타데이터를 이 목록과 대조합니다. [자동 검사](#자동-검사)를 보세요.

## 쓰임새 구분

| 쓰임새 | 뜻 |
| --- | --- |
| runtime | `wavebit`의 기본 의존성으로, 실행 중에 import합니다. pip이 따로 설치하며 번들하지 않습니다. |
| optional | Wavebit extra(`jax`, `pytorch`, `contour`, `plot`, `docs`)를 통해 설치됩니다. 조건은 runtime과 같습니다. |
| tool | 시험 실행기, 린터, 빌드·업로드 도구로, Wavebit이 요구하지 않습니다. |
| external tool | 따로 설치한 프로그램이나 인터프리터로, 출력한 수치만 `tests/references/`에 고정합니다. Wavebit 의존성이 아닙니다. |
| bundled | 다른 곳에서 온 코드나 데이터로, 이 저장소 안에 있습니다. |

정책:

- runtime 의존성은 허용형(permissive) 라이선스만 씁니다.
- LGPL과 MPL은 pip이 따로 설치하고, 수정하지 않으며, 번들하지 않는 optional·tool
  구성요소에만 허용합니다. 각 항목의 검토 내용은 목록에 기록합니다.
- LGPL·MPL은 runtime 필수 의존성으로 두지 않고, Wavebit 코드가 그런 라이브러리를
  직접 로드(ctypes, cffi, 컴파일 확장)하지 않으며 공개 API로만 씁니다.
- Wavebit이 배포하는 것(sdist·wheel의 `src/wavebit/` 패키지)은 GPL 코드를 import·
  링크·번들하지 않고, 필수 의존성으로도 두지 않습니다. 배포하지 않는 시험·연구·
  레퍼런스 스크립트는 S4나 Meep 같은 GPL 도구를 Wavebit과 같은 프로세스에서도
  실행할 수 있습니다. 조건은 sdist·wheel 밖에 있고, 도구가 없어도 설치와 기본 시험이
  동작하는 것(필요한 시험은 건너뜀)입니다. GPL 소스는 구현 참고용으로 읽지 않습니다.
- AGPL과 SSPL은 의존성·CI에 두지 않고, 이 저장소의 시험·스크립트가 실행하지도
  않습니다. 이 규칙 전에 AGPL 도구 inkstone을 블랙박스로 한 번 비교했으며,
  [과거 비교](#과거-비교)에 적었습니다. 이후 실행은 이 저장소 밖에서만 합니다.

## Python 배포 패키지

버전 하한은 `pyproject.toml`에 있고, 표는 CI가 설치하는 패키지를 적습니다.

### Runtime

| 배포 패키지 | SPDX | 비고 |
| --- | --- | --- |
| `numpy` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | 휠에 OpenBLAS(BSD-3-Clause)와, Linux·Windows에서는 GCC Fortran 런타임(GPL-3.0-or-later WITH GCC-exception-3.1, 이 사용을 허용함)이 들어 있음 |
| `scipy` | BSD-3-Clause | numpy와 같은 라이브러리를 번들 |
| `array-api-compat` | MIT | |

### Optional extra

| 배포 패키지 | Extra | SPDX | 비고 |
| --- | --- | --- | --- |
| `jax`, `jaxlib`, `optax`, `absl-py`, `ml-dtypes` | `jax` | Apache-2.0 | |
| `opt-einsum` | `jax` | MIT | |
| `tqdm` | `jax`, `pytorch` | MPL-2.0 AND MIT | 진행 표시줄 전용. pip이 따로 설치하며 수정·번들하지 않음 |
| `torch` | `pytorch` | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | CI는 CPU 휠 |
| `sympy`, `mpmath`, `networkx`, `fsspec` | torch | BSD-3-Clause | |
| `filelock` | torch | MIT | |
| `jinja2`, `markupsafe` | torch, sphinx | BSD-3-Clause | |
| `typing-extensions` | torch 외 | PSF-2.0 | |
| `setuptools` | torch | MIT | |
| `contourpy` | `contour`, matplotlib | BSD-3-Clause | |
| `matplotlib` | `plot` | LicenseRef-Matplotlib | PSF 기반 허용형 라이선스 |
| `cycler`, `kiwisolver` | matplotlib | BSD-3-Clause | |
| `fonttools`, `pyparsing`, `six` | matplotlib | MIT | |
| `pillow` | matplotlib | MIT-CMU | |
| `packaging` | matplotlib 외 | Apache-2.0 OR BSD-2-Clause | |
| `python-dateutil` | matplotlib | Apache-2.0 AND BSD-3-Clause | |
| `sphinx`, `sphinxcontrib-mermaid`, `sphinxcontrib-applehelp`, `sphinxcontrib-devhelp`, `sphinxcontrib-htmlhelp`, `sphinxcontrib-jsmath`, `sphinxcontrib-qthelp`, `sphinxcontrib-serializinghtml` | `docs` | BSD-2-Clause | |
| `pydata-sphinx-theme`, `accessible-pygments`, `babel`, `snowballstemmer`, `alabaster`, `click` | `docs` | BSD-3-Clause | |
| `myst-parser`, `mdit-py-plugins`, `markdown-it-py`, `mdurl`, `sphinx-design`, `sphinx-external-toc`, `sphinx-multitoc-numbering`, `beautifulsoup4`, `soupsieve`, `pyyaml`, `imagesize`, `charset-normalizer`, `urllib3` | `docs` | MIT | |
| `docutils` | `docs`, twine | LicenseRef-Public-Domain AND BSD-2-Clause AND 0BSD | GPL 분류자는 휠 밖의 파일을 가리킴. 목록의 검토 내용 참고 |
| `roman-numerals` | sphinx | 0BSD OR CC0-1.0 | |
| `pygments` | sphinx, pytest | BSD-2-Clause | |
| `requests` | sphinx, twine | Apache-2.0 | |
| `idna` | requests | BSD-3-Clause | |
| `certifi` | requests | MPL-2.0 | 수정하지 않은 CA 묶음. pip이 설치하며 Wavebit은 import·재배포하지 않음 |

### 도구

| 배포 패키지 | 경로 | SPDX |
| --- | --- | --- |
| `pytest`, `pytest-timeout`, `pytest-xdist`, `execnet`, `iniconfig`, `pluggy` | CI 시험 작업 | MIT |
| `colorama` | Windows의 pytest | BSD-3-Clause |
| `ruff` | lint 작업 | MIT |
| `build`, `pyproject-hooks` | 배포 작업 | MIT |
| `twine`, `id`, `readme-renderer`, `requests-toolbelt`, `rfc3986` | 배포 작업 | Apache-2.0 |
| `nh3`, `rich`, `keyring`, `jaraco-classes`, `jaraco-context`, `jaraco-functools`, `more-itertools`, `jeepney`, `pip` | twine, 환경 | MIT |
| `secretstorage`, `cryptography`, `cffi`, `pycparser` | keyring | BSD-3-Clause, Apache-2.0 OR BSD-3-Clause, MIT-0, BSD-3-Clause |

## 외부 레퍼런스 도구

모두 Wavebit 의존성이 아니고 sdist·wheel에 들어가지 않습니다. 각각 따로 설치하며,
결과 수치를 `tests/references/`에 고정합니다. 이 도구들을 설치하지 않는 CI는 그 수치를
읽습니다. 비교 내용은
[레퍼런스 사례](https://github.com/fieldalgo/wavebit/blob/main/tests/references/README.md)에 있습니다.

| 도구 | SPDX | 쓰임새 |
| --- | --- | --- |
| `Meep` | GPL-2.0-or-later | 문서화된 Python API로 쓰는 FDTD 레퍼런스. `tests/references/`의 생성기·브리지는 손으로 실행 |
| `S4` | GPL-2.0-or-later | RCWA 레퍼런스. `gen_goldens_s4*.py` 생성기는 손으로 실행하고, 배포하지 않는 일부 시험·연구는 S4를 직접 부르며 없으면 건너뜀 |
| `Octave` | GPL-3.0-or-later | RETICOLO를 별도 프로세스로 실행. 우리 `.m` 래퍼는 Octave 안에서 돌며 Wavebit을 부르지 않음 |
| `RETICOLO` | CC-BY-4.0 | Octave 안에서 실행하는 RCWA 레퍼런스로, 배포하지 않음. 저작자 표시와 이력은 [LICENSE](https://github.com/fieldalgo/wavebit/blob/main/LICENSE) |
| `oct2py` | MIT | Octave 프로세스를 구동하는 Python 드라이버 |
| `fmmax` | MIT | 별도 환경의 RCWA 레퍼런스 |
| `FDTDX` | MIT | 별도 환경의 FDTD 레퍼런스 |
| `PyMoosh` | MIT | 일회용 환경의 다층막 레퍼런스 |
| `NGSolve` | LGPL-2.1-only | NVF 연구용 선택 FEM 레퍼런스. 설치 바이너리에 GPL-2.0-or-later UMFPACK 코드도 들어 있어 그 환경은 재배포하지 않음 |

### 과거 비교

공개 동작을 재려고 손으로 한 번 실행한 도구입니다. 수치는
[레퍼런스 사례](https://github.com/fieldalgo/wavebit/blob/main/tests/references/README.md)와
RCWA 이론 노트가 인용하며, 이 저장소의 생성기·골든·시험은 이 도구들을 실행하거나
import하지 않습니다.

| 도구 | 버전 | SPDX | 쓰임새 |
| --- | --- | --- | --- |
| `inkstone` | 0.3.15 | AGPL-3.0-or-later | 공개 API와 반환 수치만으로 한 블랙박스 RCWA 비교. 소스는 읽지 않음. 이 저장소에서는 다시 실행하지 않으며, 이후 비교는 저장소 밖에서만 하고 Wavebit 시험·CI·의존성에 두지 않음 |
| `EMpy` | 2.2.3 (PyPI 이름 `ElectroMagneticPython`) | MIT | 이방성 전달 행렬의 블랙박스 비교 |

## 저장소 안의 코드와 데이터

| 구성요소 | SPDX | 위치 | 비고 |
| --- | --- | --- | --- |
| `meent` | MIT | `src/wavebit/rcwa/`의 일부 | RCWA 코드 일부의 원본. 고지는 `LICENSE`와 `LICENSE.meent` |
| `fmmax-eig-bwd` | MIT | `src/wavebit/_common/backend.py` | fmmax에서 옮긴 고윳값 분해 기울기. 고지는 `LICENSE`와 `LICENSE.fmmax` |
| `refractiveindex-info` | CC0-1.0 | `src/wavebit/data/refractiveindex/` | 휠과 sdist에 들어가는 Si, SiO2, Si3N4, Al2O3 굴절률 표. 퍼블릭 도메인인 refractiveindex.info 데이터베이스에서 가져왔고, 같은 디렉터리의 README가 표마다 원 논문과 검색 날짜를 밝힘 |

## 자동 검사

```sh
python tools/check_licenses.py                    # 설치된 모든 배포 패키지
python tools/check_licenses.py --reachable-only   # Wavebit 의존성이 닿는 것만
```

검사는 `importlib.metadata`로 설치 메타데이터를 읽습니다. 설치된 패키지에 검토 항목이
없을 때, 선언한 라이선스가 항목과 다를 때, 라이선스가 Wavebit이 그 패키지에 닿는
방식에 허용되지 않을 때, Wavebit 패키지 코드가 외부 도구를 import할 때, AGPL·SSPL
외부 도구에 저장소 밖 실행 표시가 없을 때, 이 페이지나
영어 페이지가 항목을 적지 않았을 때 실패합니다. CI는 full backends 환경과 docs
환경에서 실행하고, `tests/api/test_third_party_licenses.py`는 모든 시험 환경에서
`--reachable-only` 형태를 실행합니다.

의존성을 추가하기 전에 목록에 검토 항목을 넣고, 이 페이지와 영어 페이지에 행을
추가하세요.
