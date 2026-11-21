# 소스 체크아웃과 디버깅

worktree마다 별도의 가상 환경을 쓰세요. editable 설치는 체크아웃 하나를 가리키며, 다른
worktree가 그 지정을 바꾸지 않습니다.

수정하려는 체크아웃에서, Python 3.12 이상으로:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .  # NumPy first; add only the extras you need
python -m pip install pytest
python -c "import wavebit; print(wavebit.__file__)"
```

Windows에서는 `.venv\Scripts\Activate.ps1`로 활성화합니다. 모듈 경로는 이 체크아웃의
`src/wavebit/__init__.py`로 끝나야 합니다. 선택형 백엔드는 필요한 것만 설치하세요. 구조를 바꾼
뒤에는 노트북 커널을 다시 시작하세요.

재료 모델, 준비 결과 재사용, 필드나 수치 변경을 다룰 때는 [현재 구현 구조와 수정 지점](source-map.md)에서
진입점과 처음 돌릴 회귀 검사를 고르세요. 반입한 강좌의 코드 지도는 그 강좌의 호환 작업 흐름을 설명합니다.

## 소스를 한 단계씩 따라가기

IDE에서 이 체크아웃의 `.venv` 인터프리터를 선택하세요. `src/wavebit/` 안에 중단점을 걸고, 그
인터프리터로 스크립트나 이름을 지정한 테스트를 디버깅합니다. 표준 Python 디버거도 됩니다.

```sh
python -m pdb examples/fdtd/pulse_slab.py
python -m pytest tests/api/test_unified_rcwa.py::test_check_does_not_run_scattering -q
```

`pdb`에서 `s`는 Python 호출 안으로 들어가고 `n`은 다음 줄로 넘어갑니다. 솔버 논리를 한 줄씩
따라갈 때는 NumPy가 유용합니다. JAX JIT과 컴파일된 커널은 모든 연산을 Python 단계로 드러내지
않으므로, 살펴볼 때는 JIT을 끄고, 컴파일된 계산은 따로 검증하세요.

## 공유 과학 계산 환경 쓰기

큰 의존성을 중복 설치하기 번거롭다면 공유 환경을 명시적인 대안으로 쓸 수 있습니다. 명령마다 이
체크아웃의 소스 디렉터리를 앞에 붙이세요.

```sh
PYTHONPATH="$PWD/src" python -c "import wavebit; print(wavebit.__file__)"
PYTHONPATH="$PWD/src" python -m pytest tests/api/test_unified_rcwa.py::test_check_does_not_run_scattering -q
```

`src` 구조에서는 작업 디렉터리만으로 패키지가 선택되지 않습니다. IDE로 디버깅할 때는 실행 환경의
`PYTHONPATH`를 워크스페이스의 `src` 디렉터리로 설정하고 import 경로를 확인하세요. 공유 editable
설치를 바꾸면 그 환경에서 시작하는 모든 프로세스가 영향을 받으므로, 여러 worktree를 동시에 개발할
때는 피하세요.
