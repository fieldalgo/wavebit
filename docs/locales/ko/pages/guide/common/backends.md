# 백엔드, 장치와 정밀도

실행 설정은 물리 문제 및 수치 설정과 구분합니다.

```python
result = rcwa.run(problem, numerics=numerics,
                  backend="numpy", device="cpu", precision="double")
```

같은 키워드를 `fdtd.run`, `check`, `prepare`에도 사용합니다. 백엔드 이름은
`"numpy"`, `"jax"`, `"torch"`이며 정밀도는 `"single"` 또는 `"double"`입니다.
NumPy는 CPU가 필요합니다. JAX는 사용 가능한 `cpu`, `gpu`, `tpu` 장치를
선택적 인덱스와 함께 받으며 Torch는 CPU 또는 사용 가능한 CUDA 장치를 받습니다.
이 장치 선택은 RCWA 실행을 설명합니다. 일반 장면 `fdtd`는 double 정밀도가
필요합니다. NumPy와 JAX는 CPU를 사용하고 Torch는
[제한된 CUDA 장면 계약](../../validation/fdtd/general-cuda.md)도 허용합니다.
명시적 `fdtd.native`와 공통 `fdtd.periodic` 경로도 double 정밀도가 필요하며
사용 가능한 백엔드 장치를 받습니다. Periodic reference 알고리즘은 NumPy/CPU가
필요합니다. 장치 허용은 GPU 성능이나 정확도 측정이 아니며 물리 기능의 지원
범위는 solver마다 다릅니다.

Wavebit를 import해도 JAX x64를 전역으로 켜지 않습니다. JAX 배열을 만들기 전에
double 정밀도를 설정하세요.

```python
import jax
jax.config.update("jax_enable_x64", True)
from wavebit import rcwa
```

x64를 켜지 않고 JAX double 정밀도를 요청하면 구조화된 오류가 발생합니다.
선택적 의존성은 필요할 때만 import합니다. Live 배열은 선택한 백엔드와 장치에
두세요. 미분 가능한 입력을 NumPy를 거쳐 변환하면 미분 경로가 끊어집니다.

준비와 tracing은 [자동 미분](autodiff.md), 이전 엔진의 성능 설명과 당시 비교
측정은 [호환 backend 매뉴얼](../../migration/backends-v1.md)을 참고하세요.

## JAX 변환에서 같은 재질 입력 공유하기

단일 평면파를 쓰는 `Prepared.run`은 스칼라 영역 층과 인접 반공간이 **동일한 JAX 입력 변수**를
공유하면 그 층을 반공간에 병합할 수 있습니다. 변환할 함수 안에서 층의 배경과 반공간을 모두
`Material(eps)`로 만드세요. 투자율도 변수라면 같은 입력을 공유해야 하며, 층의 모든 영역이
같은 재질이어야 합니다. 불필요한 경계를 없애되, 공유 변수의 미분은 반공간과 원래 기준면으로
되돌리는 위상을 통해 보존됩니다.

이 조건은 배열 변환 전에 매 실행의 입력에서 새로 확인합니다. `prepare` 시점에 값이 같았다는
이유만으로 병합을 고정하지 않습니다. 서로 다른 입력은 값이 같아도 독립적입니다. `eps + 0`은
별도 식이며, `Problem` 전체를 JIT 인수로 전달하면 재질들이 별도 입력으로 펼쳐집니다.
이 경우들과 래스터 재질, 분산 재질 모델, PyTorch 재질 미분, 내부 `solve_field` 경로는 기존
병합 한계가 유지됩니다. 양수 `half_space_merge_tolerance`도 추적 재질에서는 계속 거부합니다.

이 처리는 복소 입사각에서 특히 중요합니다. 반공간과 같은 재질의 두꺼운 층을 그대로 풀면
반올림 오차가 크게 증폭될 수 있습니다. 실제로 다른 재질이나 패턴이 있는 층의 안정성까지
보장하는 기능은 아닙니다. 구체적인 재질 값을 사용하는 일반 eager 실행은 기존 경로를 유지합니다.
자세한 수치 근거는 [연결 알고리즘 설명](../../theory/rcwa/connecting-algorithms.md#10-etms-labels-at-a-complex-in-plane-wavevector)을 참고하세요.
