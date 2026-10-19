# 설치

현재 버전은 alpha인 **0.0.4**이며 PyPI에서 설치할 수 있습니다.

```shell
pip install wavebit
```

평가할 버전의 소스 checkout에서도 설치할 수 있습니다.

```shell
pip install .
```

소스 archive를 풀었다면 그 루트에서 같은 명령을 사용합니다. 저장소의 기본
브랜치는 다른 버전일 수 있습니다. 설치 버전은
`python -c "import wavebit; print(wavebit.__version__)"`으로 확인하세요.
Alpha API는 바뀔 수 있습니다. 실험적 보조 solver의 제한된 범위는
[방법 선택](../guide/rcwa/nvf-method-selection.md)을 참고하세요.

기본 설치는 두 공개 solver를 NumPy로 실행합니다. JAX와 PyTorch는 지원되는
자동 미분을 추가합니다. RCWA는 호환 accelerator 실행도 지원합니다. 일반 장면
FDTD는 double 정밀도가 필요합니다. NumPy/JAX는 CPU를 사용하고 Torch는
[제한된 CUDA 장면 계약](../validation/fdtd/general-cuda.md)도 허용합니다.
백엔드가 설치되어 있어도 solver의 물리·수치 범위가 넓어지는 것은 아닙니다.
문제를 준비하기 전에 `check`를 사용하고
[실행 가이드](../guide/common/backends.md)를 읽으세요.

필요한 선택적 백엔드를 설치하세요.

```shell
pip install '.[jax]'
```

Extra는 `jax`, `pytorch`, `contour`, `plot`, `docs`이며 함께 사용할 수 있습니다.
`[jax,pytorch]`는 두 백엔드를 설치합니다. 그래프 도우미에는 `[plot]`을 사용하세요.

## 수정할 checkout 준비

```shell
git clone https://github.com/fieldalgo/wavebit.git
cd wavebit
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -c "import wavebit; print(wavebit.__file__)"
```

Windows에서는 `.venv\Scripts\Activate.ps1`로 활성화합니다. Import 경로가
이 checkout의 `src/wavebit/__init__.py`인지 확인한 뒤 NumPy
[RCWA 첫 계산](rcwa.md)과 [FDTD 펄스 계산](fdtd.md)을 실행하세요.
작은 예제는 현재 API를 실행할 수 있는지 확인하며, 그 해상도를 수렴 권장값으로
해석해서는 안 됩니다.

라이브러리를 개발할 때는 `pytest`를 별도로 설치합니다.

```shell
python -m pip install pytest
python tools/run_verification.py --profile quick
```

[검증 runner](../development/verification.md)는 이 작은 smoke profile을 기록하고
안전한 재개가 가능한 명시적 full/research profile을 제공합니다. 직접 pytest를
사용하면 `python -m pytest tests/ -q`는 research를 선택하지 않고 slow를 건너뜁니다.
`--run-slow`는 선택된 집합의 slow 사례를 켜며 research를 선택하지는 않습니다.
`python -m pytest -m research tests/ -q`는 research만,
`python -m pytest -m "research or not research" tests/ -q`는 slow를 포함한 두
계층을 모두 선택합니다. 시험 파일을 직접 지정하면 그 파일의 research와 slow
사례도 선택합니다. 큰 검증을 시작하기 전에는
[시험 가이드](https://github.com/fieldalgo/wavebit/blob/main/tests/README.md),
디버깅에는 [소스 가이드](../development/source-checkout.md)를 참고하세요.

`environment.yml`은 study에 필요한 도구를 포함한 전체 개발 환경을 만듭니다.

```shell
conda env create -f environment.yml    # 저장소 루트에서 실행
conda activate wavebit
```

## 요구 버전과 높은 하한의 이유

Python ≥ 3.12, NumPy ≥ 2.5.1, SciPy ≥ 1.18, array-api-compat가 필요합니다.
선택적 백엔드는 `jax >= 0.11, < 0.12` 또는 `torch >= 2.14`입니다. jax 상한은 임시입니다. jax 0.12.0에서는
FDTD 장면을 다시 풀 때 다시 컴파일되고, 0.12에서는 아직 아무것도 검증하지 않았습니다.

이 하한은 작동할 가능성이 있는 최저 버전이 아니라 시험을 실제로 검증한 버전입니다.
나중에 낮출 수는 있지만 release 뒤 올리는 것은 안전하지 않으므로 엄격하게
시작합니다. 일부 하한은 알려진 결함보다 높습니다. NumPy 2.2.5–2.3.0은
복소수 `matmul`의 부동소수점 상태를 잘못 보고하여 올바른 효율을 반환하면서도
`RuntimeWarning: ... encountered in matmul`을 반복 출력하고, Torch 2.13
미만은 solver를 compile할 수 없습니다. 각 하한의 이유는
[`pyproject.toml`](https://github.com/fieldalgo/wavebit/blob/main/pyproject.toml)에
주석으로 기록되어 있습니다. 더 오래된 버전이 필요하면 검증 후 하한을 낮출 수 있습니다.

`contourpy`는 `raster_normal_method='contour'`만 사용하므로 선택적입니다.
Matplotlib에 포함되어 대부분의 환경에 이미 설치되어 있습니다.
