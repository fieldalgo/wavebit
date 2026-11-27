# 설치

현재 버전은 alpha인 **0.0.5**이며 PyPI에서 설치할 수 있습니다.

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
conda env create -f environment.yml    # run from the repo root
conda activate wavebit
```

## 요구 버전과 높은 하한의 이유

Python ≥ 3.12, NumPy ≥ 2.5.1, SciPy ≥ 1.18, array-api-compat가 필요합니다.
선택적 백엔드는 `jax >= 0.11.1`과 `jaxlib >= 0.11.1`, 또는 `torch >= 2.14`입니다.

하한은 필요한 백엔드 기능과 확인된 결함의 경계를 반영합니다.
[검증 가이드](../development/verification.md)는 전체 CI 환경과 버전·플랫폼별
개별 검증을 구분합니다. JAX/jaxlib 0.11.0은 물리적으로 유효한 스펙트럼 묶음에서도
CPU의 배치 LU/SVD 계산이 교착할 수 있습니다. 공개 0.11.1 wheel은 중첩된 스레드 풀
경로를 비활성화했습니다. Wavebit는 장치 설정 전에 실제 로드된 JAX와 jaxlib 버전을
확인하고, 오래된 환경에는 `execution.jax_version`을 반환합니다. 소스 checkout이나
의존성 설치를 생략한 환경도 같은 검사를 거칩니다.
두 패키지를 갱신한 뒤 Python 프로세스를 다시 시작하세요. 디스크의 파일만 바꾸어서는
이미 로드된 네이티브 라이브러리가 교체되지 않습니다. 현재 전체 CI는 0.12.0.1을
사용합니다. 원 모델·플랫폼 범위·수치 대조는
[묶음 실행 측정](../validation/rcwa/fitting-grid-convergence.md#묶음-jit-실행과-jax-버전)을 참고하세요.

다른 하한도 알려진 결함을 제외합니다. NumPy 2.2.5–2.3.0은 복소수 `matmul`의
부동소수점 상태를 잘못 보고하여 올바른 효율을 반환하면서도 경고를 반복하며,
Torch 2.13 미만은 solver를 compile할 수 없습니다. 각 하한의 이유는
[`pyproject.toml`](https://github.com/fieldalgo/wavebit/blob/main/pyproject.toml)에 적혀 있습니다.
이전 버전의 지원을 재검토하려면 별도 검증이 필요합니다.

`contourpy`는 `raster_normal_method='contour'`만 사용하므로 선택적입니다.
Matplotlib에 포함되어 대부분의 환경에 이미 설치되어 있습니다.
