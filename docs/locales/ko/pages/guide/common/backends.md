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
