# 계산이 레일리 점을 다루는 방식 — 처음부터 끝까지

:::{note}
두 인터페이스 모두 같은 엔진이 이 과정을 실행하며, 기본값만 다릅니다. 현재 API는 재시도 사다리(retry
ladder)를 `rcwa.Numerics(detuning=...)`로 제공하며 기본적으로 꺼져 있고, 그렇지 않으면 요청한 점을 정확히
그대로 풉니다. 2026-10-15에 퇴역한 이전 `wavebit.compat.legacy` 인터페이스는 `solve(auto_detune=True)`로
기본적으로 사다리를 실행했습니다.
:::

작성일: 2026-08-16. `rayleigh-points.md`의 짝 문서입니다. 그 문서에는 측정과 경로별 이력이 담겨 있고, 이
문서는 *과정*을 설명합니다. 즉 계산이 그 조건 위나 근처에 놓일 때 실제로 어떤 일이 어떤 순서로 일어나는지,
각 결정을 어느 구성 요소가 맡는지, 그리고 각 부분이 서로에게 넘겨주도록 어떻게 조정되어 있는지를
설명합니다. 여기 나오는 임계값을 하나라도 바꾼다면 먼저 구성(composition) 절을 읽으세요. 임계값들은 서로를
기준으로 크기가 정해져 있으며, 독립적으로 정해진 것이 아닙니다.

전체에 걸쳐 쓰는 예는 재현용 적층입니다. 균일한 스페이서 위의 패턴 기둥 층, 주기 1000 nm, 파장 500 nm,
수직 입사입니다. 이때 유지 차수는 `kx = 0.5 * m`을 가지므로, 굴절률 `n = 2.0`인 스페이서는 차수
`m = +-4`를 `kx = +-2.0`에 놓습니다. 이는 그 층의 라이트 라인(light line) 위에 **정확히** 놓이는 것입니다. 그 차수는
층 안에서 전파하지도 소멸하지도 않으며, 그에 대한 층 고유값은 수학적으로 0입니다. 파장을 어느 쪽으로든
ULP 하나만 옮기면 문제는 평범해집니다. 특이점은 물리가 아니라 좌표에 있습니다.

## 등장 요소

| 구성 요소 | 역할 |
| --- | --- |
| `solve_2d.layer_modes_2d` | 각 층의 모드를 만들고, 스치듯 입사를 감지해 분기합니다 |
| `solve_2d._rebuild_grazing_cluster` | 결합된 1차 문제로부터 스치는 모드를 다시 유도합니다 — 해결책 |
| `backend.check_mode_cutoff` / `RayleighPointWarning` | 재구성을 실행할 수 없을 때 대체 경로를 알리는 역할 |
| `backend.check_finite` | 유한하지 않은 결과에 대한 최후의 경보 장치 |
| `rcwa.Numerics(detuning=1e-12)` -> `engine.solve(auto_detune=...)` | 선택해서 켜는 재시도 사다리 — "엔지니어링 모드"(이전 인터페이스의 기본값) |

## 결정 흐름

모든 2D 층에서, 모든 계산마다:

    eig(Omega2) -> (w, W);  q = sqrt(w);  V = (Omega_R @ W) / q      [the ordinary construction]

    is min|w| / max|w| < 1e-9 ?                                      [_COUPLED_REL]
    |
    +-- no ................. ordinary path. Bit-identical to what the solver always
    |                        returned; zero added cost. This is every solve that is not
    |                        within ~1e-9 of a graze — in practice, essentially all of them
    |                        (0 engagements over 40 random configs).
    |
    +-- unreadable ......... traced (jax.jit) or grad-carrying values: the rebuild cannot
    |                        materialize them, so the old detect-and-warn contract stands —
    |                        `check_mode_cutoff` warns below 1e-13 where it can read, and
    |                        the ladder (which runs eagerly around the trace) picks it up:
    |                        through the current API only when `detuning` is set, on
    |                        PyTorch and eager jax.grad, and never under jax.jit or
    |                        jax.vmap, where `detuning` is refused
    |                        (code `prepared.detuning_transform`).
    |
    +-- yes ................ rebuild the grazing cluster:
                             restrict Omega_L / Omega_R to the cluster's invariant subspace
                             pair, eig the small coupled [[0, L_r], [R_r, 0]], and take one
                             mode per +-pair. Three exits:

                             (a) cluster representable  -> exact modes. No warning, no
                                 detune, the requested point solves. Done.
                             (b) coupled rates split below 1e-10 of the modal scale
                                 [_DEFECTIVE_REL] -> no exponential basis exists at working
                                 precision (the {1, z} regime). Raise FloatingPointError
                                 (public: NumericalError, code rcwa.singular) with the
                                 detune advice. LOUD, never quiet.
                             (c) E/H cluster sizes disagree (structure not understood)
                                 -> keep the ordinary construction + RayleighPointWarning.
                                 Detected, not guessed.

    Numerics(detuning=1e-12) (opt-in; the earlier default auto_detune=True) catches (b), (c):
        retry at wavelength * (1 + 1e-12)   [then theta * (1 + 1e-12) if needed]
        -> the retried solve has ratio ~1e-11, which is INSIDE the 1e-9 gate,
           so the rebuild solves the nearby point exactly -> AutoDetuneWarning
           names the knob and the value actually used; result.wavelengths keeps it.

## 세 가지 풀이 시나리오

**1. 표현 가능한 스침(스페이서 n = 2.0, 흔한 경우).** 클러스터는 모드 네 개(차수 +-4, 편광 두 개)입니다.
결합된 비율은 |mu| ~ 2e-9로 나옵니다. 작지만 결함(defective) 경계선에서 충분히 떨어져 있으며, 재구성은
두 1차 관계를 eps 수준까지 만족하는 모드를 반환합니다. 정확히 500 nm에서의 결과: 외삽한 참값 0.27200474에
대해 R = 0.272004739, R+T-1 ~ 1e-10이며, 어떤 경고도 없습니다. 일반 구성은 여기서 R = 0.281, R+T = 0.989를
반환했습니다. 무손실 적층에서 그럴듯해 보이는 흡수가 나타난 것입니다. `detuning` 사다리는 전혀 깨어나지
않습니다. 이 부류는 사다리에서 *졸업*했습니다.

**2. 결함 있는 스침(이 플랫폼에서 스페이서 n = 1.5).** 기하는 같지만, 반올림 때문에
결합된 비율이 |mu| ~ 5e-16에 남게 됩니다. 모드 쌍의 두 지수 함수가 수치적으로 같은 함수가 되어, 모드 기저가 존재하지
않습니다(측정: 시도한 모든 구성이 1e-4..1e-3을 남깁니다). 재구성은 요란하게 거부하며, 현재 기본값에서는
`NumericalError`(코드 `rcwa.singular`)입니다. `Numerics(detuning=1e-12)`를 쓰면 사다리가 그 거부를 받아 파장을
상대적으로 1e-12만큼 옮기고, 이제 표현 가능해진 재시도 점이 재구성을 통해 풀립니다. 결과는 참값 0.19600190에
대해 R = 0.196001919이며, 사용한 파장을 알려 주는 `AutoDetuneWarning`이 함께 나옵니다. 어떤 적층이 부류 1에
들고 어떤 적층이 부류 2에 드는지는 *반올림 복권*입니다(LAPACK 빌드마다 다를 수 있습니다). 복권이 아닌 것은
계약입니다. 정확하거나 요란하거나, 결코 조용히 틀리지 않습니다.

**3. 그 밖의 모든 경우.** 520 nm에서의 계산이나 임의의 구조에서는 관문이 False이고, 코드 경로는 항상
실행되던 그 경로이며, 결과는 수정 이전과 비트 단위로 같습니다. 재구성은 실행되지 않을 때 비용이 전혀 없고,
특이 집합에서만 고유값 분해를 약 한 번 더(층 모드 비용의 ~30%) 수행합니다.

## 임계값이 지금 위치에 있는 이유 — 구성

이 네 숫자는 네 개의 개별 선택이 아니라 하나의 시스템입니다.

| 상수 | 값 | 크기를 정한 근거 |
| --- | --- | --- |
| `solve_2d._COUPLED_REL` (재구성 관문) | 1e-9 | 사다리의 착지점(1e-12 detune 후 비율 ~1e-11)을 포함해야 합니다. 관문이 1e-12였을 때는 구조된 재시도가 일반 구성의 열화 대역으로 다시 떨어져 1e-6..8e-6 오차가 남았습니다 — 측정한 뒤 넓혔습니다. |
| `solve_2d._DEFECTIVE_REL` (거부 경계선) | 1e-10 | 지수 기저의 고유 오차는 ~eps/(k0*d*mu)입니다. 모드 스케일의 1e-10이면 k0*d >~ 0.4에서 표현 가능한 쪽의 오차가 <= ~1e-6으로 유지됩니다. |
| `backend._CUTOFF_REL` (대체 경로 경고) | 1e-13 | 일반 구성의 오차가 ~1e-5를 넘는 지점(err ~ eps/ratio)입니다. 이제는 재구성의 대체 경로에서만 의미가 있습니다. |
| `engine._AUTO_DETUNE_REL` (사다리 단계) | 1e-12 | 경고 대역에서 한 자릿수 떨어져 있고, 물리적으로 보이지 않을 만큼 작으며(500 nm에서 5e-10 nm), 재구성 관문 안에 착지합니다 — 첫 번째 행을 보세요. |

지켜야 할 의존 관계: **사다리 단계 -> 착지 비율 -> 반드시 재구성 관문 안이어야 함**. 관문을 줄이거나
단계를 독립적으로 키우면 구조된 계산이 조용히 다섯 자릿수를 잃습니다.

현재 API에서 단계는 `Numerics(detuning=...)`에 준 값이며, 준 그대로 사용합니다. `1e-12`가 이 구성의 크기를
정한 기준 단계입니다. 이전 기본값은 입력이 자동 미분 그래프를 지니면 안정적인 미분을 위해 단계를 1e-8로
넓히기도 했습니다. PyTorch 자동 미분에서 같은 동작을 원하면 `1e-8`을 주세요.

## 사용자에게 의미하는 것

- **`Numerics(detuning=1e-12)`**(이전 인터페이스의 기본값, `auto_detune=True`):
  모든 점이 올바른 답을 반환합니다. 정확한 레일리 점은 (조용히) 정확하게 풀리거나, 1e-12만큼 detune한 이웃
  점으로 돌아오며 이때 그 사실을 알리는 `AutoDetuneWarning`이 함께 나옵니다. `result.wavelengths`에는 실제로
  사용한 파장이 들어 있습니다. 어떤 것도 조용히 옮겨지지 않습니다.
- **기본값(`detuning=None`, 이전의 `auto_detune=False`)**: 요청한 점을 정확히 그대로 풀며, 결과는
  *정확하거나 요란합니다*. 거부(`NumericalError`, 코드 `rcwa.singular`)는 그 정확한 부동소수점 값에서 지수
  모드 기저가 존재하지 않음이 증명된다는 뜻이며, 메시지는 ULP 하나만큼의 detune을 처방합니다.
- **`Numerics(formulation='first_order')`** 도 층의 라이트 라인 위에 놓인 차수를 같은 방식으로, `rcwa.singular`와 같은
  조언으로 거부하며, `detuning`이 이를 재시도합니다. 균일한 층에서 차수 (1, 1)과 (2, 2)일 때 1e-12만큼
  detune한 1차 결과는 모든 효율에서 2차 결과와 1e-13 이내로 일치합니다. 2026-11-09 이전에는 그 경우 1차
  모드 분리가 개수를 잘못 세어 `rcwa.invalid`를 냈고, 조건을 그대로 남기는 더 높은 푸리에 차수를 권했습니다.
  `detuning`은 그 거부를 재시도하지 않았습니다.
- **`jax.jit`이나 `jax.vmap` 아래에서는** 거부도 재시도도 할 수 없고, `detuning`은 그곳에서 거부됩니다.
  그래서 반공간 병합이 지울 수 없는 균일층에 정확히 0인 모드가 있으면, 값과 그래디언트가 `NaN`으로 돌아오고
  `result.validity.checks['numerical_finite']`가 False가 됩니다. 이것이 `rcwa.singular`의 트레이스 쪽 형태이며
  계약입니다(fieldalgo/wavebit#268):
  - 플래그를 읽고 결과를 가리거나 버립니다.
  - 또는 그 점을 `Numerics(detuning=...)`로 즉시 실행해 풉니다.
  - 또는 입력을 그 부동소수점 값에서 옮깁니다. 균일층이라면 `jit` 아래에서도 1 ULP면 충분합니다.

  반공간 자신의 매질로 된 층은 대신 병합되어 정확히 답합니다(`rayleigh-points.md` §5).
- **이 영역을 탐색하는 측정 스크립트는 `detuning`을 `None`으로 두어야 합니다**(이전 인터페이스에서는
  `auto_detune=False`를 주세요). 그렇지 않으면 라이브러리가 표본으로 삼으려던 정확한 점을 옮겨 버려,
  "정확 대 detune" 비교가 똑같이 나옵니다(`rayleigh-points.md` §4 참고).
- **조건 근처의 오라클(oracle)**: R(lambda(1+-h))는 R0 +- a*sqrt(h)처럼 변합니다(Wood 이상). 선형 외삽은
  잘못된 값으로 수렴하며, *이전* 구성으로 계산한 ULP 하나 떨어진 이웃 점 자체가 망가진 대역 안에 있습니다.
  `rayleigh-points.md` §6처럼 멀리 떨어진 detune(~1e-8 상대값)에서 양쪽 sqrt 외삽을 사용하세요.
- **경고 거르기**: `RayleighPointWarning`, `AutoDetuneWarning`, `ExceptionalPointWarning`은
  `wavebit.rcwa.diagnostics`에 공개되어 있고 각각 별도의 범주이므로, 스윕에서 다른 것은 그대로 두고 하나만
  오류로 올리거나 숨길 수 있습니다
  (`warnings.simplefilter('error', diagnostics.RayleighPointWarning)`).

## 이 과정이 아닌 것: 1D conical x-차단

`ky != 0`인 1D conical 경로에서, 어떤 층에서 `kx_m**2 = eps*mu`인 차수는 레일리 점이 *아닙니다*. 그 `kz`는
`abs(ky)`이며, 스치는 것이 없고, 위의 장치는 전혀 관여하지 않습니다. 그곳에서 망가지는 것은 conical 경로의
두 모드 계열이며, 이 둘이 하나의 필드로 붕괴합니다. 층은 1D 빌더 안에서 정칙 기저로 정확하게 다시
만들어지며(`solve_1d.py::_regular_conical_modes`), 경고도 detune도 없으므로 `detuning` 유무와 상관없이 계산은
같은 답을 반환합니다. 모든 트레이스 아래에서도 마찬가지입니다.
`jax.jit`, `jax.vmap`(따라서 `vmap`으로 작성한 파장 스윕), `jax.lax.map`이나 `scan` 본문, `jax.checkpoint`와
`torch.func.vmap`도 관문의 마스크를 데이터로 지닌 채 층을 다시 만듭니다. 구성, 관문, 트레이스 처리 방식과
측정은 `rayleigh-points.md` §7에 있습니다.

2D와 1차 경로에서는 같은 패턴 점이 0이 아닌 근에서 거의 평행한 고유벡터를 만듭니다. 이 경로의 빌더는
자격이 되는 각 쌍을 정칙 기저와 결합 생성자로 대체합니다(`rayleigh-points.md` §§8–9). 예외점이 일치하거나
거의 일치하는 곳에서는 쌍들이 모드 세 개에서 여덟 개로 된 결함 클러스터로 합쳐지며, 두 빌더 모두 이를
통째로 다시 만듭니다(`solve_2d.py::_rebuild_defective_clusters`; 1차 분리에서는
`solve_2d_first_order.py::_regular_clusters`, 각 방향 안에서). 즉시 실행에서도 모든 트레이스 아래에서도
그렇게 하며, 자체 검사를 거칩니다. 두 단계 분할로는 나타낼 수 없는 길이 3인 Jordan 사슬은 세 단계로(`rayleigh-points.md` §8,
*세 단계의 사슬*), 즉시 실행에서는
(미분 포함, `solve_2d.py::_chain_split`) 그리고 트레이스 아래에서는 (값, 필드와 그 미분,
`solve_2d.py::_chain_split_traced`, 지수 함수의 접선은
`modal_gradient.py::_frechet_chain_terms`) 다시 만듭니다. 어떤 재구성도 맡지 않는 클러스터(모드가 여덟 개보다
많은 것, 사슬로 이어진 것)와
검사에 실패한 공동 재구성은 즉시 실행에서
`ExceptionalPointWarning`을 내며, 이 경고는 detune을 일으키지 않습니다. 추상 트레이스 아래에서는 선별 검사가
데이터로 실행됩니다. 경고를 낼 층이나, 정적 슬롯보다 많은 자격 쌍이나 클러스터를 가진 층은 NaN을 반환합니다.

## 증명이 있는 곳

- `docs/theory/rcwa/rayleigh-points.md` — 위의 모든 숫자를 측정과 함께, 막다른 길까지 기록해 두었습니다.
- `tests/rcwa/test_rayleigh_guard.py` — 고정해 둔 계약: 스침 적층에서의 정확-또는-요란 동작, 이 플랫폼에서의
  정확한 계산, 조언과 함께하는 거부, 사다리의 착지 정확도, 조건에서 벗어났을 때의 무경고, 그리고 가드의
  구체성 규칙.
