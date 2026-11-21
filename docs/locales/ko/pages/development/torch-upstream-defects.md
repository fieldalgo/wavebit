# wavebit이 우회하는 PyTorch 결함 — 각각 무엇인지, wavebit에 어디서 닿는지, 우회책을 언제 걷어낼 수 있는지

작성: 2026-10-12. macOS arm64(CPU 전용), Python 3.12.9, NumPy 2.5.1에서, 순수 torch 재현 코드(wavebit도
`array_api_compat`도 쓰지 않음)로 torch 2.13.0, 2.14.1, 2.15.0(test 채널의 릴리스 후보)과
2.16.0.dev20261011 nightly에서 측정했습니다.

대상:
- `src/wavebit/_common/backend.py::linear_solve` — 미분에 민감한 모든 지점이 거치는 선형 풀이
- `src/wavebit/_common/backend.py::eager_under_torch_compile` — wavebit을 컴파일된 그래프 밖에 두는 문
- `src/wavebit/rcwa/_engine/factorization.py::_svd` — NVF 하한의 SVD로, `?gesvd`로 다시 시도함(결함 6)
- `pyproject.toml` — torch 하한(`torch>=2.14`)으로, 사용자가 이 결함들 중 어느 것을 만날 수 있는지 정함
- `tests/common/test_linear_solve_derivatives.py` — 결함 2와 4의 카나리아인 `test_pytorchs_own_rule_is_still_wrong`. 2.15 수정 전후로 어느 경로가 기본 상태에서 틀리는지 단언함
- `tests/api/test_torch_compile.py` — 결함 1의 카나리아인 `test_canary_inductor_compiles_the_eye_write_in_order`

---

## 요약

torch 결함 네 개가 wavebit을 거치면서 오류 없이 틀린 숫자를 돌려주었습니다. 각 결함은 wavebit 안에서
우회되며, 어느 것도 평범한 즉시 실행(eager) `backward()`, JAX, `hessian`, `create_graph=True`에는 닿지
않습니다. 둘은 상류에서 릴리스로 고쳐졌고, 둘은 아직 어디서도 고쳐지지 않았습니다.
`torch.jit.trace`는 따로 적었습니다. 이것은 어떤 torch 버전도 바꾸지 못할 이유로 거부됩니다.

| # | 결함 (torch) | 2.13.0 | 2.14.1 | 2.15.0 | 2.16 nightly | 상류 | wavebit에 닿는 경우 | 우회책 | 걷어낼 수 있는 때 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 입력이 grad를 요구할 때, Inductor가 복소 단위 행렬의 제자리 대각 쓰기를 그것을 읽는 커널 뒤에 내보냄 | 틀림 | 맞음 | 맞음 | 맞음 | [#182012](https://github.com/pytorch/pytorch/issues/182012), 2.14.0에서 [PR #182292](https://github.com/pytorch/pytorch/pull/182292)로 수정 | 어떤 풀이든 `torch.compile`(Inductor)로 감쌀 때 | 문: `eager_under_torch_compile` | 하한 ≥ 2.14(2026-10-16 완료) **그리고** 문 없이 컴파일한 wavebit이 정확하다고 측정됨(아직 측정 안 함) |
| 2 | `linalg.solve`의 JVP를 다시 미분하면 항 하나를 잃음(여기서는 2차 미분의 절반) | 틀림 | 틀림 | 맞음 | 맞음 | [#192540](https://github.com/pytorch/pytorch/issues/192540)([PR #192667](https://github.com/pytorch/pytorch/pull/192667)로 수정); `left=False`는 [#196695](https://github.com/pytorch/pytorch/issues/196695) 열려 있음 | 다른 수준 안의 순방향 모드, 어느 풀이 지점을 거치든 | `linear_solve`가 LU로 우회 | 하한 ≥ 2.15(그러면 이 경우의 우회를 좁힐 수 있음) |
| 3 | CPU `lu_solve`가 LU를 `LU.stride(-3)`에서 읽어, 그 축의 크기가 1이고 stride가 정규형이 아니면 틀림 | 틀림 | 틀림 | 틀림 | 틀림 | [#200599](https://github.com/pytorch/pytorch/issues/200599), 열려 있음 | 간접적으로, `vmap(jacfwd)` 아래 결함 2의 JVP를 통해(2.13–2.14) | `linear_solve`는 stride를 바꾼 LU를 `lu_solve`에 넘기지 않음 | 하한에 포함되는 릴리스에서 #200599가 수정됨 |
| 4 | `linalg.solve`의 역전파가 `create_graph` 없이 실행되면 행렬의 순방향 접선을 버림 | 틀림 | 틀림 | 틀림 | 틀림 | [#200600](https://github.com/pytorch/pytorch/issues/200600), 열려 있음 | `forward_ad` 이원수에 `create_graph` 없는 역전파가 더해질 때(역방향 위의 순방향) | `linear_solve`가 LU로 우회 | 하한에 포함되는 릴리스에서 #200600이 수정됨 |

위의 두 카나리아는 torch가 해당 결함을 고치면 일부러 실패합니다. Inductor 카나리아는 측정한 곳인 macOS
arm64에서만 단언하고 다른 곳에서는 건너뜁니다. 따라서 torch ≥ 2.14.1에서는 macOS arm64에서 실패합니다. 이는
문 없는 컴파일을 측정하라는 신호(1행)이지 회귀가 아닙니다.

다섯 번째 결함은 틀린 숫자를 돌려주는 대신 예외를 일으키며, torch 자체의 코드도 아닙니다. CPU wheel이
번들하는 MKL이 잘 조건화된 일부 행렬에서 분할 정복 SVD를 수렴시키지 못합니다(6절,
[#202596](https://github.com/pytorch/pytorch/issues/202596)).

---

## 1. Inductor의 단위 행렬 순서 바꾸기

모달 지수 함수는 `array_api_compat`의 torch `eye`와 같은 방식으로 단위 행렬을 만듭니다. 영 행렬에 제자리
대각 `fill_`을 더하는 방식입니다. Inductor로 컴파일하고 입력이 grad를 요구하는 복소수이면, 이 쓰기가 단위
행렬을 읽는 커널 뒤로 배치되어 그 커널들은 영을 읽습니다. wavebit이 필요 없는 재현 코드는 다음과 같습니다.

```python
q = torch.tensor([[1 + 1j, 0], [0, 4 + 4j]], dtype=torch.complex128, requires_grad=True)
def f(q):
    eye = torch.zeros(2, 2, dtype=q.dtype); eye.diagonal().fill_(1)
    return eye * torch.exp(q) + (q - eye * q)
torch.compile(f)(q)   # on 2.13.0 differs from eager f(q) by max|diff| 60.24; 0.0 from 2.14.0
```

다음 중 어느 것이든 2.13.0에서 결과를 정확하게 만듭니다.

- 복소수 대신 float64;
- grad 없음;
- 제자리 채우기 대신 `torch.eye`;
- `backend='aot_eager'`;
- `torch._inductor.config.reorder_for_peak_memory=False`.

2.14.0, 2.14.1, 2.15.0과 nightly에서는 쓰인 그대로 정확합니다. 수정은
[#182012](https://github.com/pytorch/pytorch/issues/182012) / [PR #182292](https://github.com/pytorch/pytorch/pull/182292)
(`torch/_refs`의 `diagonal_scatter` 분해, 2.14.0에 포함)이며, 2026-10-16에 찾았습니다. 재현 코드의 대각
뷰에 대한 `fill_`은 함수화되면 `diagonal_scatter`가 됩니다. 2.13.0에서 wavebit을 거치면 1D TE 손실이 즉시
실행의 -0.895 대신 -4.43으로, 그 그래디언트가 0.272 대신 -0.118로 읽혔습니다.

문은 컴파일된 호출자 안에서 wavebit을 즉시 실행시킵니다([`torch-compile.md`](torch-compile.md)
§6). 문을 걷어낼 수 있는 것은 문 *없이* wavebit을 컴파일한 결과가 하한 버전에서 정확하다고 측정된 뒤뿐입니다.
어느 쪽이든 컴파일은 wavebit 부분에 속도를 주지 않습니다(`torch-compile.md` §4). 같은 문에서의
`torch.jit.trace` 거부는 그와 상관없이 남습니다(§5).

## 2. 다시 미분한 `linalg.solve`의 JVP

`A(x)`가 아핀이고 `B(x)`가 이차이면, `X = A⁻¹B`의 정확한 2차 미분은 `A X'' = B'' − 2 A' X'`를 만족합니다.
2.13.0과 2.14.1에서 `jvp(jvp)`, `jacfwd(jacfwd)`, `jacrev(jacfwd)`, `grad(jvp)`, `forward_ad` 위의 역전파,
그리고 `create_graph` 없는 역전파 위의 `forward_ad`는 재현 코드에서 각각 그 절반을 돌려줍니다(0.25 대신
0.125). `hessian`, `jvp(grad)`, `grad(grad)`, `create_graph=True`는 정확합니다.

원인은 `linalg_solve_jvp`가 원래 계산에서 저장한 LU로 풀었다는 것이며, 그 LU는 미분을 담고 있지 않습니다.
2.15에서는 JVP를 다시 미분할 수 있을 때마다
(`(GradMode::is_enabled() && A.requires_grad()) || areAnyTensorSubclassLike({A, dA, dB})`) `A`로 다시
풉니다.

2.13.0에서 wavebit을 거친 결과:

- 1D TM 격자의 두께 2차 미분이 1.1001006774e-04 대신 1.1801616522e-04로 읽혔습니다;
- 2D 래스터는 ETM에서 812.595272 대신 760.746214, 필드 값이 있는 SMM에서 725.018988로 읽혔습니다.

전체 목록은 [`docs/migration/autodiff-v1.md`](../migration/autodiff-v1.md#forward-mode)에 있습니다. 관련된
순방향 위의 역방향 경로(`torch.func.jvp` 다음, `requires_grad`로 잡은 매개변수까지 `backward()`)도 전후가
같습니다. 2.13.0과 2.14.1에서 틀리고, 2.15.0과 nightly에서 맞습니다.

릴리스된 2.15.0+cpu wheel에서(2026-11-17 측정, Linux x86-64,
`tests/common/test_linear_solve_derivatives.py`의 닫힌 형식에 대해): `linear_solve`가 다시 쓰는 35개 경로
중 `torch.linalg.solve` 자체가 이제 25개에서 4e-15까지 정확합니다. 순방향 수준이 다른 수준 안에 있는 모든
경로, `vmap(jacfwd)`, `vmap(hessian)`, grad 수준 안에서 만든 `forward_ad` 이원수가 여기에 해당하며, 25개 중
셋은 전에도 정확했습니다. 나머지 열 개에서는 여전히 틀리며, 상대 오차가 0.34에서 94입니다.
`create_graph` 없는 역전파 위의 `forward_ad`(결함 4), 그리고 `torch.no_grad()`, `set_grad_enabled(False)`
또는 `inference_mode` 아래에서 역전파가 인수분해를 재사용하는 경로들이며, `vmap(jacrev)`도 그중
하나입니다. `linear_solve`는 2.14.1과 2.15.0에서 35개 모두 정확하며, 하한이 2.14인 동안은 모든 경로를
유지합니다.

## 3. stride를 바꾼 배치 축을 가진 CPU `lu_solve` — #200599

`apply_lu_solve`(`aten/src/ATen/native/BatchLinearAlgebraKernel.cpp`)는 연속한 LU 행렬 사이의 거리를
`LU.stride(-3)`에서 가져오고, 평탄화한 배치 색인으로 접근합니다. `LU[:, None]`(또는 바깥 `vmap`으로
배치된 LU 인수와 안쪽 것으로 배치된 우변)은 그 자리에 stride가 행렬 stride가 아닌 크기 1의 축을 두므로,
나중 레인이 다른 레인의 인수를 읽게 됩니다. 이슈의 2×2 예에서는 두 번째 해가 `[1/6, 1/3]` 대신
`[5, −1]`로 나왔으며, 네 버전 모두에서 그랬습니다. `.contiguous()`로 고쳐집니다.

wavebit은 `lu_solve`를 직접 호출한 적이 없습니다. 이 결함은 결함 2를 통해 wavebit에 닿았습니다.
2.13–2.14에서 `linalg.solve`의 JVP는 저장한 LU로 이 커널을 거쳐 풀기 때문에, 두께의 레인에 대한
`vmap(jacfwd)`가 0.011455 대신 196.4(1D TM)로 읽혔고, conical과 2D에서는 NaN이었습니다.

`linear_solve`는 두 가지 방법으로 이 배치를 피합니다.

- `vmap` 밖에서는 LU를 `lu_factor`에서 바로 가져오므로 stride가 정규형입니다;
- 어떤 `vmap` 아래에서든 `lu_solve` 대신 `lu_unpack`과 `solve_triangular` 두 번으로 풉니다
  (`backend.py::_differentiable_solve`).

## 4. 순방향 모드 아래에서의 `linalg.solve` 역전파 — #200600

`linalg_solve_backward`는 grad 모드가 켜져 있을 때만 `A`로 다시 풉니다. `create_graph` 없는 역전파는 grad
모드가 꺼진 채 실행되므로, 순방향 접선을 담지 않은 저장된 LU와 피벗을 씁니다. 순방향 AD는 grad 모드에
의존하지 않으므로, `A`의 살아 있는 접선이 버려집니다. 이슈의 1×1 예에서 `∂X/∂B`의 접선은 (−0.25 대신)
`None`으로, `∂X/∂A`의 접선은 (0.25 대신) 0.125로 나왔으며, 네 버전 모두에서 그랬습니다.

`lu_factor` + `lu_solve`, `inv(A) @ B`, `create_graph=True`, `torch.func.jvp(torch.func.grad)`는
정확합니다. 결함 2의 JVP 수정은 이 가지를 건드리지 않았습니다.

wavebit에서는 conical과 2D ETM 최종 정합(`solve_1d.py::final_matching`)과 SMM의 필드 상태
(`scattering.py::_below_reflections`, `scattering.py::_field_state`)에 닿았습니다. 역전파 위의
`forward_ad`로 구한 2D 래스터의 두께 2차 미분이 11.18 대신 1.284로 읽혔습니다.
1D TE/TM ETM 연쇄는 `linalg.inv`로 역행렬을 구하므로 영향을 받은 적이 없습니다.

## 5. `torch.jit.trace` — 거부하며, torch 결함이 아님

추적은 값에 따라 달라지는 각 결정을 상수로 기록하고, `grad_fn`이 없는 출력을 돌려줍니다. 거부 전 develop
에서 wavebit을 거친 결과:

- 추적한 풀이 하나가 `TracerWarning` 66개를 냈습니다;
- 그것을 다른 항과 섞은 손실은 즉시 실행이 0.2721483665790을 읽는 곳에서 wavebit 몫의 그래디언트를 0.0으로
  읽었습니다;
- 풀린 결과의 `reconstruct_field`는 한 층의 식을 다른 층에서 재생했습니다.

이것이 추적의 본성이므로 어떤 torch 릴리스도 이를 바꾸지 않습니다. 모든 문이 이것을 이름으로 거부하며,
그 거부는 남습니다.

## 6. 번들된 MKL의 분할 정복 SVD — #202596

2026-10-29에 Linux x86-64(Intel Xeon, AVX-512)에서 torch 2.13.0, 2.14.1, 2.15.0과
2.16.0.dev20261029 nightly로 측정했으며, 모두 MKL 2024.2를 번들합니다. `torch.linalg.svd`(LAPACK `?gesdd`)가
공기 중 원 하나(반지름 161.48 nm, 600 nm 셀, ε 12.25, 조건수 12.25)의 961×961 `<1/eps>` 행렬에서
`LinAlgError`를 일으켰는데, `svdvals`, NumPy, SciPy는 성공했습니다.

원인은 torch가 아니라 LAPACK에 있습니다. `DLASD2`의 축약 검사 `ABS(D(J)-D(JPREV)).LE.TOL`이 거의 같은 특잇값의
뒤바뀐 쌍을 통과시키고, 그러면 `DLASD4`가 반복 한계에 걸립니다. 이것은 Reference-LAPACK #1138이며, 그
PR #1140(2026년 3월, 아직 어느 reference 릴리스에도 포함되지 않음)으로 고쳐졌습니다. 같은 이중 대각 행렬을
넣으면 `dgesdd`는 MKL 2021.4에서 2026.1까지와 reference LAPACK 3.12.0에서 실패했고, MKL 2026.2와 OpenBLAS
0.3.34에서 성공했습니다. 소스에는 reference LAPACK 3.12.1과 OpenBLAS 0.3.30까지 옛 줄이 있습니다. 조밀
행렬이 이 결함에 걸리는지는 이중 대각화의 마지막 비트 반올림에 달려 있으므로, ISA 경로와 스레드 수에 따라
바뀌며 호출자에게는 무작위처럼 보입니다.

이 결함은 NVF 역규칙 하한을 통해서만 wavebit에 닿습니다. 이 하한은 그것이 걸리는 층에서 특이 벡터가
필요합니다(`rcwa/_engine/factorization.py::_svd`).

- 수동 유전체로 된 층은 유전율이나 특잇값을 읽을 수 있는 곳에서는 벡터 SVD를 건너뜁니다(`jax.jit`
  추적의 상수인 재료 포함). 유전율이 jit 인수이거나 vmap 레인인 층은 알 수 없으므로 SVD를 수행합니다;
- 유한 행렬에서 즉시 실행이 실패하면, 어느 백엔드에서든 QR 기반 `?gesvd`로 다시 계산하고(NumPy와 torch는
  호스트에서 SciPy의 것, JAX는 `SvdAlgorithm.QR`. JAX의 `?gesdd`는 예외 대신 NaN 인수를 돌려줍니다)
  `wavebit.rcwa.diagnostics.SvdConvergenceWarning`으로 알립니다;
- `jax.jit` 아래에서는 아무것도 다시 계산할 수 없습니다. NaN 인수 때문에 층에 하한이 적용되지 않으며,
  `jax.debug.callback`이 같은 경고를 냅니다.

같은 버그는 NumPy, SciPy, JAX(CPU LAPACK이 SciPy의 것)가 영향을 받는 LAPACK에 링크되는 곳이면 어디서든
닿습니다. Anaconda `defaults` 채널의 기본 MKL 빌드가 그런 예입니다. PyPI wheel은 고쳐진 OpenBLAS를
담습니다. 재시도는 지원하는 설치가 가져올 수 있는 모든 LAPACK이 수정을 담게 되면 걷어낼 수 있습니다 —
그 전에는 안 됩니다. NumPy와 SciPy 빌드가 하나의 LAPACK에 고정되어 있지 않기 때문입니다.

## 영향을 받지 않는 것

- 즉시 실행 `backward()`(평범한 최적화 루프), 모든 JAX 변환, `hessian`, `create_graph=True`, 그리고
  `torch.func.vmap`의 값: 이 중 어느 것도 결함 1–4에 닿지 않습니다.
- 결함에 닿는 미분 경로들(다른 수준 안의 순방향 모드, `create_graph` 없는 역방향 위의 순방향,
  `vmap(jacfwd)`, `torch.compile`)은 위에서 측정한 모든 버전에서, `4938026b`(결함 1)와
  `1d47957e`(결함 2–4) 이후로 wavebit을 거쳐 정확합니다.

## torch를 올릴 때 다시 확인하기

1. `tests/common/test_linear_solve_derivatives.py`와 `tests/api/test_torch_compile.py`를 실행하세요. 카나리아가
   실패하면 torch가 그 결함을 고쳤다는 뜻입니다.
2. 새 버전에서 상류 이슈의 재현 코드(#200599, #200600, #202596)를 다시 실행하세요.
3. 그런 다음에야 `linear_solve`의 우회 조건을 좁히거나 문 없는 컴파일을 측정하고, `pyproject.toml`의 하한을
   측정한 버전으로 올리세요. 우회책은 하한이 결함이 있는 모든 버전을 배제할 때까지 남습니다.
