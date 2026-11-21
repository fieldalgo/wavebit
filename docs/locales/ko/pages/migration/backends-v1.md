# 백엔드, 장치와 정밀도

> **과거 API:** 이 페이지에서 `wavebit.compat`을 쓰는 코드 조각과 설명은 과거 체크아웃이 있어야 동작합니다.
> 현재 설치본에는 이 네임스페이스가 없으며, 퇴역 직전 기준은
> [리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
> 기록된 실험은 각자 인용한 커밋과 환경으로 재현하세요. 그 측정은 다시 실행하지 않았습니다.


:::{note}
이 페이지는 예전 인터페이스의 전체 호환성 레퍼런스입니다. `wavebit.compat.legacy`를 명시적으로 쓰세요. 새 애플리케이션은 [현재 생애주기](../guide/common/lifecycle.md)와 [현재 API 레퍼런스](../reference/api.md)로 시작하세요. 여기에 인용한 수치 근거는 원래의 적용 범위를 그대로 유지합니다.
:::

물리 구현은 하나이며, 얇은 어댑터 뒤에서 세 백엔드 모두에서 돌아갑니다. 따라서 어떤 백엔드를
고르든 결과는 달라지지 않습니다. 백엔드 간 일치는 주장이 아니라 테스트 모음의 일부입니다.

이 장은 `rcwa()`에 관한 것입니다. [주기 적층 시간 영역 가이드](../guide/fdtd/time-domain.md)는
예전 NumPy 기본값과 명시적 백엔드 경로를 구분합니다. 별도의
[현재 FDTD 가이드](../guide/fdtd/general-fdtd.md)는 물리 단위의 `fdtd.Problem`,
그와 분리된 수치 설정, 그리고 NumPy/JAX/PyTorch의 CPU 배정밀도 실행을 다룹니다.
아래의 장치와 정밀도 계약은 예전 RCWA 팩토리에 속합니다.

![NumPy, JAX, PyTorch](../figures/backends.png "Backends")

* [NumPy](https://github.com/numpy/numpy) — 기준선입니다. 자동 미분도, GPU도, 컴파일할 것도
  없습니다.
* [JAX](https://github.com/google/jax) — 자동 미분과 XLA입니다. `jax.grad`는 공개 API에서 바로
  동작하고, `jax.jit`에는 추가로 고정된(frozen) 형상이 필요합니다([`autodiff.md`](../guide/common/autodiff.md)).
* [PyTorch](https://github.com/pytorch/pytorch) — 즉시 실행(eager) 방식의 자동 미분입니다. 고정할
  기록이 없으므로 낡을 기록도 없습니다. `torch.autograd`는 `torch.func.vmap`을 거쳐서도 동작하지만,
  `torch.func`의 grad와 jvp 변환은 그 입력이 모드 연산자에 닿는 곳에서, 중첩된 변환은 연산자가 미분을
  지니는 모든 곳에서 이름을 들어 거부됩니다(두께에 대한 미분은 정확합니다)
  ([`autodiff.md`](autodiff-v1.md#forward-mode)).
  `torch.compile` 아래에서 wavebit은 컴파일된 호출자 안에서 즉시 실행 방식으로 돌아가므로, 그 값과
  그래디언트는 즉시 실행 torch의 것과 같습니다. `fullgraph=True`, `torch.export`, `torch.jit.trace`는
  이름을 들어 거부합니다([`../torch-compile.md`](../development/torch-compile.md)).

```python
solver = wavebit.rcwa(backend='numpy', ...)               # 'numpy', 'jax' or 'torch'
solver = wavebit.rcwa(backend=wavebit.Backend.JAX, ...)   # or the enum member, the same thing
```

이름은 각 라이브러리의 import 이름이며 철자도 정확히 그대로입니다. `'JAX'`, `'pytorch'`, 그리고
2026-10-11까지의 표기였던 정수 `0` / `1` / `2`는 허용되는 이름 목록과 함께 거부됩니다.
`solver.backend`는 `wavebit.Backend` 멤버를 담으며, 이 멤버는 자신의 이름과 같습니다
(`solver.backend == 'jax'`).

## 백엔드별로 할 수 있는 것

|                               | NumPy | JAX | PyTorch |
| ----------------------------- | :---: | :-: | :-----: |
| float64 / complex128          |   ●   |  ●  |    ●    |
| float32 / complex64           |   ●   |  ●  |    ●    |
| 자동 미분     |   ○   |  ●  |    ●    |
| GPU                           |   ○   | ●*  |   ●*    |
| 프로세스 내 장치 병렬화 |   ○   | ● (`pmap`) | ○ |
| TPU                           |   ○   |  ○  |    ○    |

\*GPU에서의 사정을 따로 밝혀야 하는 연산은 비대칭 고유값 분해이며, 최근에 상황이 바뀌었습니다.
다음은 2026-08 기준입니다. `jax.lax.linalg.eig`는 호스트 CPU의 LAPACK, GPU의 MAGMA, GPU의 cuSOLVER
가운데 하나를 cuSOLVER 버전, 왼쪽 고유벡터를 요청했는지 여부(이 솔버는 요청하지 않습니다),
`jax_use_magma`에 따라 자동으로 고릅니다. PyTorch는 여전히 `torch.linalg.eig`에 CUDA 입력을 주면 그
장치가 CPU와 동기화된다고 문서화하지만, 2.10에서 CUDA 12.8 이상용 네이티브 `cusolverDnXgeev` 경로가
추가되었습니다. 둘 다 이 프로젝트가 지원하는 모든 버전(`jax>=0.11`, `torch>=2.14`)에 해당합니다.
**이 경로들은 여기서 하나도 실행해 보지 않았습니다.** 이 저장소의 어떤 것도 GPU에서 측정한 적이
없으므로, 그 열은 성능 주장이 아니라 "프레임워크가 지원한다"로 읽으세요. TPU는 아예 제외입니다.
32비트 고유값 분해가 그곳으로 가는 경로가 전혀 없기 때문입니다.

`pmap` 행은 *하나의* 계산을 여러 장치에 나누는 것에 관한 것이며, 이것은 JAX만 합니다. 스윕에 필요한
병렬화에 대해서는 아무것도 말하지 않습니다. 파장, 각도, 매개변수 스윕은 서로 독립적인 풀이의
집합이며, 이를 프로세스나 노드에 나누어 보내는 것(Ray, MPI, Slurm 배열 작업)은 세 백엔드 모두에서
똑같이 동작합니다. [`specs.md`](../guide/common/specs.md)는 실행이 그런 경계 중 하나를 건너야 할 때
취하는 형태입니다.

## 장치와 정밀도 요청하기

두 키워드는 어느 한 네임스페이스의 어휘가 아니라 wavebit 자체의 어휘로 씁니다. 그래서 하나의
시그니처가 세 백엔드를 모두 지원할 수 있습니다.

```python
wavebit.rcwa(backend='jax', device=1, complex_dtype=1, ...)     # JAX, GPU, single precision
```

| | 받는 값 |
| --- | --- |
| `device` | `0` / `'cpu'`, `1` / `'gpu'` / `'cuda'`, 또는 백엔드 자체의 장치 객체. NumPy에는 장치가 하나뿐이므로 GPU 요청을 메시지만 출력하고 넘어가지 않고 **거부합니다** |
| `complex_dtype` | `0`(배정밀도, 기본값) / `1`(단정밀도), 또는 세 네임스페이스의 복소 dtype 표기 중 무엇이든 — `np.complex128`, `jnp.complex64`, `torch.complex128` |

32비트는 메모리를 절반으로 줄이지만 항상 안전하지는 않습니다. 여덟 자리 이상 차이 나는 두 수는
경고도 오류도 없이 조용히 상쇄되며, 합성곱 행렬은 흔히 둘을 함께 담습니다. 문제에 그런 일이 없음을
확인한 뒤에만 쓰세요. 소멸파(복소 각도) 입사는 2026-10-02까지 32비트에서 완전히 실패하던 유일한
경우였습니다. 이제 그 두 가지 가지 결정, 즉 반공간 근과 ETM 층 라벨이 모두 32비트에서도 해결됩니다.
측정한 한 점은 ETM에서 배정밀도와 1.9e-6, SMM에서 3.4e-6 차이가 나며, 이전에는 ETM에서 5.1e-2
차이였습니다. 솔버는 어떤 가지가 여전히 반올림 오차로 결정되었을 때 `UnresolvedBranchWarning` 경고를
냅니다. 이런 일은 근이 분지 절단 위에 놓여야 일어납니다. [`../branch-cuts.md`](../theory/rcwa/branch-cuts.md)를 보세요.

튜토리얼 [`03-jax`](https://github.com/fieldalgo/wavebit/blob/main/tutorials/rcwa/03-device-and-datatype-jax.ipynb)와
[`03-torch`](https://github.com/fieldalgo/wavebit/blob/main/tutorials/rcwa/03-device-and-datatype-torch.ipynb)에서 두 키워드를 차례로 다룹니다.

## JAX 64비트 상한과 import 순서

JAX는 `jax_enable_x64`가 켜져 있지 않으면 배열을 32비트로 제한합니다. 이것은 **기본값이 아니라
상한**입니다. 꺼져 있으면 `jnp.asarray(x, jnp.float64)`도 float32를 내므로, 배정밀도를 아예 선택할 수
없습니다. 이미 여러 곳에서 이 상한을 풀지만, 모든 순서를 다루는 것은 없습니다.

* `from wavebit.compat import legacy as wavebit`는 JAX가 **이미** import되어 있으면 상한을 풉니다.
  `import wavebit`로 새 루트 패키지를 import하면 이 프로세스 전역 설정은 바뀌지 않습니다.
* JAX RCWA 백엔드는 처음 **사용될** 때 상한을 풉니다.
* JAX FDTD 실행(`fdtd` 또는 `fdtd_grid`)은 시간 루프를 만들 때, 그리고
  `meshgrid_xy(..., backend='jax')`는 샘플링하기 전에 상한을 풉니다. 둘 다 첫 배열을 만들기 전입니다.

남은 호환성 틈은 JAX가 예전 모듈보다 늦게 import되고 첫 풀이 전에 쓰이는 경우입니다. 그 사이에 만든
배열은 `complex_dtype`이 모든 곳에서 여전히 128로 읽히는데도 조용히 `complex64`가 됩니다. 그 순서는
한때 모든 dtype이 배정밀도라고 보고하는 가운데 참값 0.67 대신 `|R| = 3.94`를 냈습니다.
`wavebit`이 아래처럼 명시적인 예전 별칭이면 `wavebit.enable_x64()`가 이 틈을 막습니다.

```python
from wavebit.compat import legacy as wavebit
wavebit.enable_x64()          # before building any JAX array wavebit will be handed
import jax.numpy as jnp
theta = jnp.asarray(0.4 + 0.3j)        # complex128, not complex64
```

여러 번 불러도 결과가 같으며, JAX가 설치되어 있지 않으면 **조용히 아무것도 하지 않는 대신 예외를
냅니다**. 배정밀도를 요청했는데 조용히 얻지 못하는 것이야말로 이 경로 전체가 막으려는 실패이기
때문입니다. 어떤 방법으로도 닿을 수 없는 순서에 대해서는 실행 시점에 `PrecisionLossWarning`이 그렇다고
알려 줍니다.

## 어떤 백엔드로 실행할까

취향의 문제가 아니며, 순방향 풀이와 미분하는 풀이에 대한 답도 같지 않습니다.

| 상황 | 백엔드 |
| --- | --- |
| 작거나 중간 크기의 순방향 풀이 | **NumPy** — 설치가 가장 가볍습니다 |
| 형상 레시피 미분 | 측정한 모든 크기에서 **jit한 JAX** — `truncation_order=3`에서 PyTorch의 2.3x, 13에서 1.3x이며, 최악의 경우에도 13 스텝 안에 컴파일 비용을 회수합니다 |
| 래스터 미분, `truncation_order ≲ 5` | **즉시 실행 PyTorch**가 확실히 낫습니다. 그 크기에서는 JAX의 컴파일 비용이 회수되지 않습니다 |
| 래스터 미분, `truncation_order ≳ 7`, 긴 실행 | **jit한 JAX** — PyTorch의 대략 1.5x이며, 열두 스텝이 채 안 되어 컴파일 비용을 회수합니다 |

형상 레시피에서 이 중 무엇이든 쓰기 전에 `shape_nvf_grid_size`를 보세요.
`truncation_order=3`에서 기본값은 풀이 자체에 드는 비용보다 훨씬 많은 시간을 스텝마다 법선장을 만드는
데 씁니다. 이 값을 낮추는 것이 어떤 백엔드 선택보다도 효과가 큽니다. 다만 격자를 정하는 것은
직사각형의 *각도*이므로, 그 한계는 취향이 아니라 정확도가 정합니다.

이 모든 것은 [`../backend-choice.md`](../validation/rcwa/backend-choice.md)에서 CPU 호스트 하나로 측정했으며,
실행 자체의 반복 재현성을 숫자 옆에 함께 적었습니다.

`torch.compile`은 속도를 위한 선택이 아닙니다. 컴파일된 호출자는 정확합니다. 모든 공개 wavebit 호출
가능 객체가 스스로를 그래프 밖에 두므로(`torch.compiler.disable`), wavebit은 그 안에서 즉시 실행
방식으로 돌아가고 그 출력에 대한 사용자 자신의 연산만 컴파일됩니다. 그래서 wavebit 부분에는 아무
이득이 없습니다. 어차피 이득은 적었을 것입니다. 풀이는 거의 전부가 컴파일러가 손대지 않고 넘기는
LAPACK 호출이기 때문입니다. 2026-10까지는 wavebit을 그래프 밖에 두는 장치가 없었고, 기본 모드는 오류
없이 틀린 값과 그래디언트를 돌려주었습니다. Inductor(torch 2.13)가 모드 지수 함수의 바탕이 되는 복소
항등식을 잘못 컴파일했기 때문입니다. `fullgraph=True`, `torch.export`, `torch.jit.trace`는 이름을 들어
거부합니다. 2.13 미만의 torch에서는 컴파일 자체가 실행되지 않으며, 2.14가 그 잘못된 컴파일을
고쳤습니다. 그래서 2.14가 최소 버전입니다. [`../torch-compile.md`](../development/torch-compile.md)를 보세요.
