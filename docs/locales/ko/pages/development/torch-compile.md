# `torch.compile` — 왜 실패했는지, 왜 쫓을 가치가 없는지, 왜 그 아래에서 wavebit이 즉시 실행되는지

> **과거 API:** 이 페이지에서 `wavebit.compat`을 쓰는 코드 조각과 설명은 과거 체크아웃이 필요합니다.
> 이 네임스페이스는 현재 설치본에 없으며, 퇴역 직전 기준은
> [리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
> 기록된 실험은 각자 인용한 커밋과 환경으로 재현하세요. 그 측정은 다시 실행하지 않았습니다.


작성: 2026-08. macOS arm64(CPU 전용)에서 `develop`의 `8257e2c`를 대상으로,
NumPy 2.5.1 / SciPy 1.18.0 / jax 0.11.0으로 측정했습니다. **갱신 2026-10 (§6)**, torch 2.13.0에서; §6의 결함은 torch 2.14.0에서 고쳐졌습니다(2026-10-16 기록).

대상:
- `src/wavebit/rcwa/_engine/engine.py::CoreSolveMixin.period` — 첫 실패가 발생한 곳(그 경로가 제거되기 전까지는 block 모델러 자체의 `period` 설정자였음)
- `src/wavebit/_common/backend.py::eig` — 나머지 실패들이 다루는 기본 연산
- `pyproject.toml` — 그 결과로 2.7에서 2.13으로(§3), 그리고 §6의 상류 수정을 위해 2.14로 올린 torch 하한
- `src/wavebit/_common/backend.py::eager_under_torch_compile` — 2026-10부터 wavebit을 컴파일된 그래프 밖에
  두는 입구(door); §6 참고

---

## 요약

**2026-10부터 컴파일된 호출자는 정확합니다. wavebit은 결코 컴파일되지 않기 때문입니다(§6).** 모든 공개
wavebit 호출 가능 객체는 `torch.compiler.disable` 입구이므로, wavebit은 `torch.compile`된 함수 안에서 즉시
실행되고 그 값과 그래디언트는 즉시 실행 torch의 것입니다. 컴파일되는 것은 호출자가 wavebit의 출력에 대해
수행하는 자기 산술뿐입니다. `fullgraph=True`와 `torch.export`는 이름으로 거부되며, `torch.jit.trace`도 모든
공개 호출에서(풀린 결과의 읽기 포함) 거부됩니다.
그 전까지 torch 2.13의 기본 모드 `torch.compile`은 **오류 없이 틀린 값과 틀린 그래디언트**를
돌려주었습니다(1D TE 손실이 즉시 실행의 -0.895 대신 -4.43, 그 그래디언트가 0.272 대신 -0.118). Inductor가
`array_api_compat`의 복소 `eye`의 제자리 대각 쓰기를 그것을 읽는 커널 뒤에 내보냈기 때문입니다. 그 결함은
§1–§5가 측정한 모든 것 뒤에 모달 지수 함수(`modal_gradient.py`, 2026-09-29)와 함께 들어왔으므로, §3과 §4의
"비트 단위로 동일"은 과거 기록입니다.

2026-08 조사인 §1–§5는 아래에 쓴 그대로 유효합니다. `torch.compile`은 **모든** Wavebit 경로에서
`InternalTorchDynamoError`로 실패했습니다. 네 원인 중 어느 것도 Wavebit에 있지 않았습니다. 넷 모두 PyTorch
결함이며, **torch 2.7.0**(예전 하한이 허용하던 버전)에 있었고 **torch 2.13.0에서 모두 고쳐졌습니다**.
2.13.0에서는 실패하던 코드 조각이 수정 없이 실행되어 비트 단위로 동일한 값을 돌려줍니다.

그러니 작동하게 만들 것이 없습니다. 그럴 이유도 없습니다. 2.13.0에서 모든 것이 작동하는 상태로 컴파일해도
**순방향에서 0.92–0.99x, 순방향+역방향에서 0.85–1.02x** — 어느 쪽으로든 아무것도 얻지 못합니다. 이유는
버그가 아니라 구조적입니다. `truncation_order=10`에서 **실행 시간의 98%가 Inductor가 그대로 넘기는
LAPACK/BLAS 호출 안에 있고**, Inductor는 복소 연산자에 대해서는 아예 코드를 생성하지 않는데, 사실상 나머지
전부가 그것입니다.

**권고(2026-08): `torch.compile` 지원을 추가하지 마세요.** 속도에 관해서는 여전히 유효합니다 —
§6은 컴파일러 아래에서 wavebit을 즉시 실행 상태로 두므로, 구성상 wavebit 부분은 아무것도 얻지 못합니다.
그럼에도 torch 하한은 2.13으로 올렸습니다 — 컴파일을 가능하게 하려는 것이 아니라, 이 프로젝트가 밝힌
규칙이 하한은 테스트 모음을 검증한 버전이라는 것이고(`pyproject.toml`), 당시 검증한 버전이 2.13이었기
때문입니다. 2026-10-16부터는 §6의 Inductor 결함이 고쳐진 첫 릴리스인 2.14이므로, 그 결함을 재현하는
카나리아가 지원하는 모든 torch에서 단언합니다. 즉시 실행 torch와 `.backward()`는 2.7.0에서도 잘 작동합니다.
작동하지 않는 것은 `torch.compile`뿐입니다.

---

## 1. 실패, 그리고 그것이 모든 경로에 닿은 이유

```python
total = lambda x: torch.sum(wavebit.rcwa(backend='torch', structure=x, **cfg).solve().T)
torch.compile(total)(u)      # InternalTorchDynamoError on torch 2.7
```

```
torch._dynamo.exc.InternalTorchDynamoError: AttributeError: 'str' object has no attribute 'IF_NEEDED'
from user code:
   File ".../array_api_compat/numpy/_aliases.py", line 94, in asarray
    return np.array(obj, copy=copy, dtype=dtype, **kwargs)
```

가장 안쪽부터 본 연쇄:

1. `Modeling.__init__`은 `self._xp`를 **즉시 실행 NumPy** 네임스페이스로 기본 설정한 다음 `period` 설정자를
   실행합니다. `period=[300., 300.]` — 배열이라고는 보이지 않는 평범한 Python 리스트 — 로 설정자는 NumPy
   네임스페이스에서 `xp.asarray(vals, dtype=F, device=dev)` 가지를 탑니다.
   이것은 평범한 설정 장부 정리이며, 텐서 수학은 관여하지 않습니다.
2. 그 코드가 컴파일 영역 안에 있으므로 Dynamo가 그것을 추적합니다. Dynamo의 `trace_numpy` 기능은 추적하는
   동안 `numpy` 대신 `torch._numpy`를 넣습니다.
3. `array_api_compat`의 NumPy `asarray`는 NumPy 비공개 열거형인 `np._CopyMode.IF_NEEDED`를 찾습니다.
   `torch._numpy`는 이를 구현하지 않고(`hasattr(torch._numpy, '_CopyMode')`가
   `False`), Dynamo는 없는 속성을 `str`로 모델링하며, `getattr(str, 'IF_NEEDED')`가 **Dynamo 안에서** 예외를
   일으킵니다 — 그래서 Dynamo가 추적할 수 없는 것에 보통 쓰는 그래프 분리 대신 내부 오류가 납니다.

모든 경로가 똑같이 실패한 이유가 이것입니다. 실패는 객체 생성에 있고, 경로별 처리보다 앞에 있습니다. 모든
경로는 리스트 값 `period`로 `Modeling`을 만듭니다.

Wavebit 없는 최소 재현(2.7.0에서 실패, 2.13.0에서 통과):

```python
import torch, array_api_compat.numpy as anp
torch.compile(lambda t: t + float(anp.asarray([300.0, 300.0])[0]))(torch.zeros(1))
```

## 2. 그 뒤에 있던 세 장애물

하나를 치우면 다음 것이 드러납니다. 숫자 하나를 얻으려면 넷을 모두 치워야 했습니다.

| # | 실패 | 치운 방법 | 누구의 버그 |
|---|---|---|---|
| 1 | `AttributeError: 'str' object has no attribute 'IF_NEEDED'` | `torch._dynamo.config.trace_numpy = False` | PyTorch (`torch._numpy`) |
| 2 | `AssertionError: expected size 98==98, stride 1==98 ... 98==1` | `torch._inductor.config.size_asserts = False` | PyTorch (Inductor) |
| 3 | `InternalTorchDynamoError: KeyError: i` | 한 글자 소스 수정, 아래 참고 | PyTorch (Dynamo) |
| 4 | `RuntimeError: self.stride(-1) must be 1 to view ComplexDouble as Double` | 찾지 못함 | PyTorch (Inductor/AOTAutograd) |

**#2는 RCWA가 없어서는 안 되는 유일한 기본 연산인 `linalg.eig`에 관한 것입니다.** Wavebit 없이 세 줄로
줄어듭니다.

```python
A = torch.randn(8, 8, dtype=torch.complex128)
torch.compile(lambda A: torch.linalg.eig(A)[1])(A)   # AssertionError on 2.7.0
```

즉시 실행 `torch.linalg.eig`는 stride `(1, 8)`인 고유벡터를 돌려줍니다 — LAPACK 고유의 배치인 **열 우선**
입니다. Inductor의 메타 커널은 `(8, 1)`을 선언하고, `assert_size_stride`가 그 불일치를 잡습니다. 고유*값*은
잘 컴파일되며(1차원이라 틀릴 배치가 없음), `size_asserts`를 끄면 비트 단위로 정확한 결과가 나옵니다 — 따라서
틀린 숫자가 아니라 배치에 관한 거짓 경보였습니다. `linalg.eigh`도 실패했고, `linalg.svd`는 괜찮았습니다.

**#3**은 셀 변수가 있는 프레임에 그래프 분리가 떨어질 때 Dynamo가 재개 바이트코드를 생성하지 못하는
문제입니다. `src/wavebit/rcwa/_engine/solve_1d.py::final_matching`에서 그 계기는
`variants = [_pack(..., *(e[i] for e in eff2)) for i in (0, 1)]`의 제너레이터 표현식입니다 — PEP 709에서도
제너레이터 표현식은 실제 프레임을 만들므로 `i`를 포획하고, 그것을 `final_matching`의 셀 변수로 만듭니다.
`*(...)`를 `*[...]`로 고쳐 쓰면 셀 변수와 오류가 사라집니다. **그 수정은 장애물을 넘기 위해서만 했고 되돌렸습니다.**
이 모든 일에 필요했을 유일한 Wavebit 쪽 변경이며, 동작에 중립적이었고(결과가 비트 단위로 동일),
2.13.0에서는 필요 없습니다.

**#4**는 역방향 패스에서만 나타납니다 — 최적화에 중요한 절반입니다. #2와 같은 복소 배치 문제이며, 이번에는
autograd 중에 연속이 아닌 `ComplexDouble`을 `Double`로 보는 경우입니다. 2.7.0에서는 우회책을 찾지 못했습니다.

## 3. 넷 모두 상류에서 고쳐졌습니다

torch 2.13.0을 넣은 일회용 venv에서는 실패하던 코드 조각이 **수정 없이** — 설정 플래그도, 소스 수정도
없이 — 실행되어 즉시 실행과 같은 값을 돌려줍니다.

| | torch 2.7.0 | torch 2.13.0 |
|---|---|---|
| 장애물 1 (`np._CopyMode`) | 실패 | 통과 |
| 장애물 2 (`linalg.eig` stride) | 실패 | 통과 |
| 장애물 4 (역방향 복소 뷰) | 실패 | 통과 |
| 전체 Wavebit 풀이, 컴파일 | 실패 | 통과, `diff = 0.0` |
| 컴파일된 `.backward()` | 실패 | 통과, `max\|Δgrad\| ≤ 8.5e-20` |

그 두 "통과"는 `modal_gradient.py`가 생기기 전(`8ac44120`, 2026-09-29에 생성)인 `8257e2c`에서 측정한
것입니다. 이제 컴파일된 wavebit에 대해서는 참이 아닙니다. 모달 지수 함수의 복소 `eye`가 그래프에 들어오자
Inductor가 값과 그래디언트를 틀리게 구했고(§6), 그 치료법은 wavebit을 컴파일하지 않는 것이었습니다.

장애물 3은 독립된 재현 코드가 없으므로, 전체 2.13.0 실행에서 나타나지 않는다는 것만 알려져 있습니다.
어느 릴리스가 각각을 고쳤는지는 이분 탐색하지 않았습니다 — 쓸모 있는 사실은 2.7은 실패하고 2.13은 실패하지
않는다는 것입니다. 같은 재료 지름길을 제거한 뒤(`5c73bda`) `8257e2c`에서 두 절반을 다시 확인했습니다. 그
제거는 그래프 분리 원인 하나를 지웠고, 여기서는 다른 아무것도 바꾸지 않았습니다.

**이 때문에 하한이 `torch>=2.13`으로 옮겨졌습니다.** 컴파일을 가능하게 하려는 것이 아니라 — §4가 보여
주듯 그것은 아무것도 주지 않습니다 — `pyproject.toml` 자체의 규칙이 하한은 테스트 모음을 검증한 버전이라는
것이기 때문입니다. 전체 테스트 모음을 그 조합(torch 2.13.0, jax 0.11.0, NumPy
2.5.1, SciPy 1.18.0, array-api-compat 1.15.0)에서 실행했습니다. 녹색이었고, 건너뛴 것이 없었으며, 고정된
골든 값에 대한 회귀도 없었습니다. 하한은 2026-10-16부터 2.14입니다(§6).

버전 2.8–2.12는 테스트하지 않았으므로 제외됩니다. 누군가 하나를 검증하면 나중에 하한을 완화할 수 있으며,
그것이 릴리스 후에 조이는 것보다 쌉니다. 이 글을 쓸 당시 `wavebit`은 아직 PyPI에 없었고 `develop`은 push되지
않았으므로, 이 제외로 배제된 기존 사용자는 없었습니다.

이를 점검하다가 하한이 하나의 규칙으로 정해지고 있지 않다는 것도 드러났습니다 — `numpy>=2.3.1`과
`scipy>=1.15`는 알려진 결함이 없는 가장 오래된 버전이었지만, 테스트 모음은 실제로 2.5.1과 1.18.0에서
검증되었습니다. 이들도 같은 때에 밝힌 규칙에 맞췄습니다. `tqdm`과 `optax`는 일부러 느슨하게 둡니다. 둘 다
골든 값이 다루는 숫자를 건드리지 않습니다.

## 4. 얻을 수 있는 것, 측정

래스터 경로, 64×64 유전율 `structure`, CPU 하나, 칸마다 3..7회 중 최고값, 모든 것이 작동하는
**torch 2.13.0**:

| `truncation_order` | 순방향 즉시 실행 | 순방향 컴파일 | | 순방향+역방향 즉시 실행 | 순방향+역방향 컴파일 | |
|---|---|---|---|---|---|---|
| 3 | 0.0126 s | 0.0137 s | **0.92x** | 0.0186 s | 0.0219 s | **0.85x** |
| 6 | 0.1810 s | 0.1905 s | **0.95x** | 0.2992 s | 0.2940 s | **1.02x** |
| 10 | 1.9800 s | 1.9996 s | **0.99x** | 3.0386 s | 3.0172 s | **1.01x** |

값은 비트 단위로 동일했고(`diff = 0.0`) 그래디언트는 `8.5e-20`까지 일치했습니다 — `modal_gradient.py`가
생기기 전인 `8257e2c`에서이며, 그 뒤의 컴파일된 wavebit에 대해서는 참이 아닙니다(§6). 시간 측정도 과거
기록입니다. 이제 wavebit은 `torch.compile` 아래에서 즉시 실행되므로, "컴파일" 열은 즉시 실행 값으로 읽힐
것입니다.

비교하자면, 실제로 작동하는 대응물인 `jax.jit`은 마지막으로 측정했을 때 `truncation_order=3`에서 약 1.9x,
`truncation_order=10`에서 1.3x로 떨어지는 이득을 냈습니다 -- 같은 방식으로 측정한, 같은 무너지는 추세입니다.
컴파일된 torch는 거기에 이르지 못하고, 평범한 NumPy에도 이르지 못합니다.

## 5. 얻을 것이 0인 이유와, 그것을 바꿀 수 있는 것

버그가 아니며, 장애물을 치워서 고칠 수 있는 것도 아닙니다. 시간이 어디에 쓰이는지(torch 프로파일러, aten
자체 시간, 래스터 경로):

| | `truncation_order=3` | `truncation_order=10` |
|---|---|---|
| BLAS / LAPACK | 83.7% | **97.9%** |
| 그 밖의 모든 것 | 16.3% | 2.1% |

`truncation_order=10`에서 상위 네 연산은 `linalg_eig`(41.5%), `linalg_lu_solve`(25.1%), `mm`(17.5%),
`linalg_lu_factor_ex`(13.8%)입니다. Inductor는 이 중 어느 것도 구현하지 않으며, 즉시 실행이 호출하는 것과
같은 ATen 커널로의 폴백을 내보냅니다. 나머지 ~2%는 복소수 값 원소별 연산인데, Inductor는 이를 컴파일하지
않겠다고 알립니다.

```
UserWarning: Torchinductor does not support code generation for complex operators.
Performance may be worse than eager.
```

이 조사 도중 그 비중이 커졌는데, 기록해 둘 만한 이유가 있습니다. `c992b72`에서 한 이전 프로파일은
`truncation_order=10`에서 BLAS/LAPACK을 89%로 잡았고, `_fft_c2c`가 거기서 5%, `truncation_order=3`에서
30%를 차지했습니다. 그 뒤 `072710d perf(factorization): compute the raster's Fourier coefficients exactly`가
들어왔습니다. 이 커밋은 FFT를 없애지 않았습니다 — 같은 변환 여섯 개가 여전히 실행됩니다 — 그러나 그 앞의
**업샘플링**을 없앴습니다. 표본 경로는 격자가 `4*truncation_order+1`개의 고조파를 분해할 때까지 픽셀을
반복한 뒤 *그것을* 변환했지만, 정확한 계수는 원래 격자의 N점 DFT에 `sinc(k/N) * exp(-i*pi*k/N)`를 곱한
것입니다. 작은 격자를 대신 변환하자 FFT가 `truncation_order=3`에서 8.4 ms에서 0.05 ms로(작업의 0.34%),
`truncation_order=10`에서 157.9 ms에서 0.27 ms로(0.01%) 줄었습니다.

어느 쪽이든 Inductor가 손대지 않고 넘기는 라이브러리 호출이었으므로, 결론은 바뀌지 않았습니다 — 남은 것이
LAPACK에 더 몰렸을 뿐입니다.

그래프는 Wavebit이 일부러 내리는 결정들 — 그중에서도 `array_namespace` 분기 — 때문에도 반복해서 분리되므로,
컴파일러에 닿는 것은 라이브러리 호출 사이를 잇는 접착 코드의 조각들입니다.

다음 중 하나가 바뀌면 다시 살펴볼 가치가 있습니다.

- **GPU.** 위의 모든 것은 CPU 전용입니다. 이 기계에 GPU가 없기 때문입니다. GPU에서는 커널 실행 오버헤드와
  메모리 트래픽의 비중이 훨씬 크고, 순위가 달라질 수 있습니다.
- **Inductor가 복소 연산자의 코드 생성을 배움.** 그것이 LAPACK 밖의 비중을 문제 삼게 만들 것이며 — 그 비중이
  가장 큰 낮은 `truncation_order`에서 가치가 가장 큽니다.
- **비용이 더 이상 조밀 고유값 분해가 아니게 됨** — 다른 모드 솔버, 또는 작은 풀이 여러 개를 배치로 묶어 호출당
  오버헤드가 다시 지배하게 되는 경우.

그때까지 추적되는 풀이에 대한 작동하는 답은 `jax.jit`입니다.
[풀이 미분하기](../migration/autodiff-v1.md)를 보세요(이 글을 쓸 때는 README의 "Differentiating a
simulation").

## 6. 2026-10: Inductor가 복소 단위 행렬을 잘못 컴파일하므로 wavebit은 즉시 실행됩니다

2026-10-12에 torch 2.13.0, Python 3.12.9, macOS arm64(CPU)에서, `develop`의 `01815aa0`과 이 변경에서
측정했습니다.

### 결함

`array_api_compat`의 torch `eye`는 영 텐서를 만든 뒤 그 대각을 제자리에서 씁니다
(`diagonal().fill_(1)`). 그 텐서가 복소수이고 입력이 grad를 요구하면, Inductor는 제자리 쓰기를 그 텐서를 읽는
커널들 *뒤에* 내보내므로 커널들은 영을 읽습니다. 모달 지수 함수(`src/wavebit/rcwa/_engine/modal_gradient.py`,
2026-09-29 생성)가 정확히 그렇게 만들어져 있습니다.
`eye*phases + (q - eye*values)*divided`. 이를 보는 데 wavebit은 필요 없습니다.

```python
import torch, array_api_compat.torch as act

def f(q):
    eye = act.eye(q.shape[0], dtype=q.dtype)
    return eye * torch.exp(q) + (q - eye * q)

q = torch.randn(4, 4, dtype=torch.complex128, generator=torch.Generator().manual_seed(11))
print((torch.compile(f)(q.clone().requires_grad_()) - f(q)).abs().max())   # 1.381; eagerly 0
```

| Inductor로 컴파일한 `f` | max \|컴파일 − 즉시 실행\| |
|---|---|
| complex128, 입력이 grad를 요구함 | **1.381** |
| complex64, 입력이 grad를 요구함 | **5.709** |
| grad 없는 complex128; grad 유무와 상관없이 float64 | 0 |
| `backend='aot_eager'` 또는 `backend='eager'` | 0 |
| `array_api_compat`의 `eye` 대신 `torch.eye` | 0 |
| `torch._inductor.config.reorder_for_peak_memory = False` | 0 |

따라서 Dynamo와 AOTAutograd는 정확하고, Inductor의 최대 메모리 재배치가 쓰기를 옮깁니다. 그 패스를 끄면 이
재현 코드가 고쳐집니다(권고가 아니라 사실입니다. 여기서 그것에 의존하는 것은 없습니다). 2026-10-11에
PyTorch 이슈를 검색했을 때 일치하는 것은 없었습니다.

wavebit에서는 값 **그리고** 그래디언트에서 조용했습니다. 가중 읽기 값(`tests/rcwa/test_forward_mode.py`의
장면)에 대한 풀이 한 번과 역전파, 기본 모드, 이 변경 전:

| 장면 | 컴파일 손실 | 즉시 실행 손실 | 컴파일 그래디언트 | 즉시 실행 그래디언트 |
|---|---|---|---|---|
| 1D TE, 유전율, ETM | **-4.434983049888** | -0.8953932051167 | **-0.1182433628003** | 0.2721483665790 |
| 필드가 있는 2D Li 래스터 | **-5.512585863438** | -0.8677333191407 | **0.02838407891956** | 0.01971995024377 |

같은 1D 장면은 `backend='aot_eager'`(그래디언트 상대 오차 6.1e-16)와 `backend='eager'`(0)에서 정확했습니다.
TM 두께와, 밖에서 만든 솔버의 theta 설정자는
`InternalTorchDynamoError: TypeError: Cannot interpret 'torch.complex128' as a data type`를 일으켰고,
`fullgraph=True`는 `cmath.sqrt`에서 `Unsupported`를 일으켰습니다. 틀린 두 행은 아무것도 일으키지 않았습니다.

### 입구

거부는 추적 안에서 일으킬 수 없습니다. Dynamo는 추적 중에 일어난 예외를 그래프 분리로 바꾸고 프레임을 기본
방식으로 다시 실행하며, 거기서 예외가 삼켜집니다(wavebit 없는 장난감 예제가 오류 없이 5.0을 돌려주었습니다).
그리고 산술이 틀린 것이 아니라 컴파일러가 틀린 것입니다. 그래서 wavebit은 자기 프레임을 그래프 밖에 둡니다.
`src/wavebit/_common/backend.py::eager_under_torch_compile`는 `wavebit.__all__`의 모든 클래스, 공개 호출이
돌려주는 모든 타입, 그리고 그 둘 중 어느 것의 모든 wavebit 기반 클래스 — 클래스 자체 네임스페이스의 각 공개
메서드, 속성 접근자, `classmethod`/`staticmethod`와 dunder — 그리고 공개 함수들을 꾸밉니다. 65개 지점입니다.
2026-10-13부터는 `wavebit.rcwa`와 `wavebit.fdtd` 생애주기 함수와 그 선언 및 결과 타입도 입구입니다. 입구 안에서만
만들어 쓰고 호출자에게 넘기지 않는 클래스는 입구가 필요 없습니다. 입구가 호출하는 모든 것은 이미 밖에 있기
때문입니다. 입구는

- `torch._dynamo`가 `sys.modules`에 없는 동안에는 그대로 호출합니다. 그때는 아무것도 컴파일 중일 수 없습니다.
  `torch.compile`이 그것을 가져오기 때문이며, 즉시 실행 풀이와 역전파는 그것을 가져오지 않습니다;
- 그 밖에는 캐시된 `torch.compiler.disable(fn, reason=...)` 래퍼를 거쳐 호출합니다. 이는 함수와 그것이
  호출하는 모든 것을 `torch.compile` 밖에 두는 PyTorch의 문서화된 방법입니다
  (docs.pytorch.org/docs/2.13/generated/torch.compiler.disable.html). 첫 호출 때 지연 생성하는 것은 PyTorch
  자체의 `torch._compile._disable_dynamo`의 발상입니다.

같아 보이는 두 지름길은 틀립니다.
`torch.compiler.is_compiling()`으로 경로를 고르면 `tests/api/test_torch_compile.py`의 다섯 경보 장면이 모두
틀립니다. Dynamo는 프레임이 아무것도 아닌 것으로 컴파일된 래퍼의 추적을 멈추고 그것을 기본 방식으로
실행하며, 입구 자체의 프레임은 여전히 컴파일된 채로 남습니다. 이미 입구 안에 있을 때 래퍼를 건너뛰면,
`torch.compile`로 컴파일한 사용자 호출 가능 객체가 그 안에서 호출될 때마다 Dynamo가 wavebit을 다시
추적하게 됩니다. 그 호출 가능 객체가 프레임 훅을 다시 설정하기 때문입니다. 그 결과 컴파일된 `forward`를 쓴
`solver.grad`가 유전율 그래디언트를 최대 0.8165 어긋나게 돌려주었고(즉시 실행의 가장 큰 항목은
0.1539), 손실은 맞았으며 오류도 없었습니다.

**이것이 주는 것.** 값은 즉시 실행과 2.6e-16 이내, 그래디언트는 9.5e-16 이내입니다 — 남은 차이는 호출자 자신의
컴파일된 가중합입니다. 기본 모드, `mode='reduce-overhead'`와
`dynamic=True`에서, 컴파일된 함수 안이나 밖에서 만든 솔버, 속성 설정자,
`solve(...)` 키워드, 결과의 읽기, `ScatteringMatrix`, `Lattice`, `Material`과
`TabulatedMaterial`, `CoherentSolver`와 `excitation=`에 대해 그렇습니다. `torch.func`
경로와 이중 역전파를 감싼 `torch.compile`은 즉시 실행 미분을 돌려줍니다. `create_graph=True`를 두 번 감싸면
NumPy 2차 차분 1.1001006765e-04에 대해 1.1001006774e-04를 읽으며, 전에는
`InternalTorchDynamoError`를 일으켰습니다. `jvp(jvp)`와 `vmap(jacfwd)`를 감싸도 즉시 실행 미분이며, 따라서
정확합니다. 이 둘은 선형 풀이의 순방향 규칙을 다시 미분하거나 두 `vmap` 수준 아래에서 실행하는데,
`torch.linalg.solve` 자체의 규칙은 이를 틀리고(torch 2.13과 2.14), 모든 풀이가 거치는 `backend.linear_solve`는
맞게 구합니다(자동 미분 안내의
[Forward mode](../migration/autodiff-v1.md#forward-mode)). `tests/api/test_torch_compile.py`가 세 가지를
모두 NumPy와 대조해 검사합니다.

**이름으로 거부하는 것.**

- `fullgraph=True`와 엄격한 `torch.export`는 `torch._dynamo.exc.Unsupported: Skip
  calling torch.compiler.disable()d function ... (reason: wavebit runs eagerly under
  torch.compile: compiling it buys no speed, and Inductor has silently miscompiled its complex
  arithmetic before (torch 2.13, fixed in 2.14) -- see docs/development/torch-compile.md)`를 일으킵니다. 입구는
  Dynamo가 로드된 상태에서 실행된 뒤에야 이유를 담습니다. 한 프로세스에서 첫 `fullgraph` 컴파일이나 엄격한
  export는 즉시 실행 wavebit 호출 뒤라도(그 호출은 빠른 경로를 타고 아무것도 캐시하지 않습니다) 이유 없이
  `torch.compiler.disable`만 지명합니다 -- 그래도 `Unsupported`로 크게 알립니다.
- 엄격하지 않은 `torch.export`는 `disable`이 적용되지 않는 기본 방식으로 입구를 실행하므로, 입구는
  `torch.compiler.is_exporting()`이 참인 동안 `NotImplementedError`("wavebit cannot be exported: ...")로
  거부합니다. 그것이 없으면 내보낸 모듈은 즉시 실행 값과 `grad_fn` 없음을 돌려주었습니다.
- `torch.jit.trace`는 모든 공개 wavebit 호출에서 `NotImplementedError`("wavebit cannot be traced by
  torch.jit.trace: ...")를 일으킵니다. Dynamo가 관여하지 않고 값이 구체적이므로 `disable`이 잡지 못합니다.
  각 입구는 공개 `torch.jit.is_tracing()`이 참인 동안 거부합니다. 그 전에는 닿는 곳마다 조용했습니다.
  - 풀이 하나가 `TracerWarning` 66개와 함께 추적되어 `grad_fn` 없는 출력을 돌려주었으므로, 추적한 풀이를 다른
    항과 섞은 손실은 그래디언트에서 wavebit의 몫을 잃었습니다. p = 0에서 즉시 실행이 0.2721483665790을 읽는
    곳에서 0.0, p = 0.05에서 즉시 실행이 0.3733920308334를 읽는 곳에서 0.1(즉 2p뿐)이었습니다;
  - 풀린 2층 격자의 `reconstruct_field`를 한 층 안의 높이에서 추적하고 다른 층에서 재생하면, 즉시 실행과
    중앙 차분이 0.003442581948을 읽는 곳에서 d/dz -0.004063805947을 읽었고, 적층 밖에서 재생하면 즉시
    실행이 거부하는 곳에서 0.9016을 답했습니다;
  - 추적한 길이로 만든 `Lattice.hexagonal(a)`는 NumPy 상수가 되었으므로
    `meshgrid_xy(lattice=...)`가 예제의 셀을 재생했습니다. 즉시 실행의 +3.393e5 대신 +2.922e5였고, 즉시
    실행의 1574.35에 해당하는 그래디언트는 없었습니다.

  첫 판은 풀이가 결정 지점에서 값을 읽는 곳(`backend._concrete`)에서만 거부했습니다. 마지막 둘은 거기에
  닿지 않으므로, 검사는 모든 공개 호출이 지나는 입구에 있습니다.

**다루지 않는 것.**

- 실제 텐서에 대한 `make_fx`는 `grad_fn` 없는 값을 돌려줍니다. 거부할 근거가 되는 공개 플래그가 없습니다.
- 컴파일된 autograd(`torch._dynamo.config.compiled_autograd = True`)는 입구가 아닌 wavebit의 사용자 정의
  `autograd.Function` 역전파 규칙을 컴파일합니다. 즉시 실행과 7.2e-13 이내로 측정했지만(2026-10-11), 그것은
  계약이 아닙니다.
- 컴파일된 함수를 거친 `forward_ad` 접선은 `None`으로 돌아옵니다. wavebit 없이도 그러므로, torch의
  문제입니다.

### 비용과, 그것이 기대는 것

Dynamo가 로드되지 않은 동안 입구 호출당 약 90 ns, 로드된 뒤에는 약 490 ns이며, 순방향 한 번에 374–548회의 입구
호출이 있습니다(2026-10-11에 67개 지점으로 셈). 번갈아 실행한 세 라운드의 최솟값, 스레드 하나,
`develop` → 이 변경: 1D TE (3,0) 값만 구하는 풀이는 Dynamo 없이 2.14 → 2.17 ms, 있을 때
2.17 → 2.40 ms; 필드가 있는 2D (2,2) 풀이는 7.66 → 8.04 ms와 7.64 → 7.96 ms입니다.
순방향 모드 이원수, `torch.func.grad`, 모든 `torch.optim` 단계는 Dynamo를 가져오므로, 최적화 루프는 첫 단계
뒤로 두 번째 수치의 비용을 치릅니다. 순방향당 고정된 ~0.2–0.3 ms이며, 풀이 하나가 수백 밀리초 걸리는 차수 (15,15)
이상에서는 0.1% 미만입니다. `torch.jit.trace` 검사는 어느 경로에서든 모든 입구 호출에 약 100 ns를 더합니다.
아무것도 하지 않는 함수를 감싼 입구는 Dynamo 없이 61에서 158 ns로, 있을 때 346에서 449 ns로 늘었습니다
(2026-10-12, 두 실행의 평균, 각 실행은 백만 회 호출 반복 일곱 번의 최솟값). 1D TE, conical, 2D 장면에서
순방향당 입구 호출은 373, 390, 392회이며 — 순방향당 약 0.04 ms입니다.

두 가지 구현 사실에 기대며, 둘 다 바뀌면 조용히 실패할 것입니다.

1. `torch.compile`과 엄격하지 않은 `torch.export`는 무엇이든 컴파일하기 전에 `torch._dynamo`라는 모듈을
   가져옵니다 — 빠른 경로의 전제입니다.
2. `torch.compiler.disable`은 재귀적이므로(문서화된 기본값 `recursive=True`), 입구가 호출하는 모든 것은 즉시
   실행으로 남습니다.

`tests/api/test_torch_compile.py`가 둘 중 어느 것이든 잡으므로(그 경보는 컴파일된 풀이를 즉시 실행 autograd와
1e-12에서 비교합니다), torch를 올릴 때마다 실행하세요. 그 카나리아
`test_canary_inductor_compiles_the_eye_write_in_order`는 위의 다섯 줄짜리 `f`를 실행하고 차이가 0이기를
요구합니다. 재배치를 되살린 torch에서는 실패할 것입니다.

**상류에서 수정됨, 2026-10-16.** 위의 결함은 torch 2.13.0, macOS arm64(CPU)에서 측정했습니다. PyTorch는
2.14.0에서 이를 고쳤습니다. [pytorch/pytorch#182012](https://github.com/pytorch/pytorch/issues/182012)
(`torch.compile`이 `diagonal_scatter`의 그래디언트를 조용히 0으로 만드는 문제. 제자리 대각 `fill_`은
함수화되면 이것이 됩니다)는 `torch/_refs`의
[PR #182292](https://github.com/pytorch/pytorch/pull/182292)로 닫혔고, 2.14.0 릴리스 노트에 "Fix incorrect
compiled output and gradients for overlapping-input `torch.diagonal_scatter` operations"로 실렸습니다.
코드 조각을 쓰인 그대로 여기서 다시 측정했습니다. 2.13.0은 grad가 있을 때 complex128에서 1.381,
complex64에서 5.709를 주고, 2.14.0과 2.14.1은 macOS arm64와 CI의 Linux x86-64에서 모든 dtype에 대해 0을 줍니다. 그날부터
하한이 2.14이므로 카나리아는 모든 플랫폼에서 단언합니다. 입구는 위의 이유로 남습니다. 측정한 비용으로 앞으로의
모든 Inductor 결함에서 wavebit을 지켜 주며, wavebit 부분을 컴파일해도 얻는 것이 없습니다(§4).
