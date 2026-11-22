# 다중 모드 입사 — 여러 차수를 한꺼번에 비추기

:::{note}
현재 API는 `rcwa.IncidentOrders`로 여러 차수를 비추며, 그 실행은 경계 전력과 필드를 보고하지만
차수별 효율은 보고하지 않습니다. 이 페이지가 측정하는 차수별 `R`, `T`와 진폭은 이전
`wavebit.compat.legacy` 인터페이스의 `excitation=` 키워드에서 나오므로, 그 값이 필요한 블록은
그 인터페이스를 쓰며 그렇다고 밝혀 둡니다.
:::

**코드**: `src/wavebit/rcwa/sources.py` — `IncidentOrders`;
`src/wavebit/_common/conventions.py` — `lit_orders`, `incident_column`;
`src/wavebit/rcwa/_engine/solve_1d.py` — 두 가지 정합; `src/wavebit/observations/power.py` —
`incident_power`; `src/wavebit/rcwa/_engine/scattering.py` — SMM 쌍둥이;
`src/wavebit/rcwa/_engine/engine.py` — `excitation`.
**테스트**: `tests/rcwa/test_excitation.py`.

```python
from wavebit import rcwa

source = rcwa.IncidentOrders(weights=(((0, 0), 1.0), ((1, 0), 0.6 - 0.8j)))  # by the (x, y) pair
```

:::{admonition} 이전 인터페이스
:class: earlier-interface

2026-10-15에 `wavebit.compat`와 함께 퇴역했으며, 이 코드는 과거 체크아웃에서만 실행됩니다.

```python
# Earlier interface: per-order R/T for lit orders are not in the current API yet.
from wavebit.compat import legacy as wavebit

solver = wavebit.rcwa(..., excitation={0: 1.0, 1: 0.6 - 0.8j})     # 1D: by the x order
solver = wavebit.rcwa(..., excitation={(0, 0): 1.0, (1, 0): 0.5j})  # 2D: by the pair
```
:::

조명은 한 번의 풀이에서 결맞게(coherently) 둘 이상의 회절 차수로 들어옵니다. 가중치는 *공간적*입니다.
`polarization`(이전 인터페이스에서는 `psi` / `tetm`도)은 여전히 어떤 편광이 들어오는지를 정하며, 비추는
모든 차수가 같은 편광을 지니되 **각 차수의 s/p 좌표계에서** 지닙니다. 가중치는 해당 차수의 `s_hat` /
`p_hat`에 곱해집니다. `phi = 0`인 1D 풀이에서 s는 `sign(kx) * y_hat`이므로, `kx < 0`인 차수 위의 TE 빔은
`-y` 방향으로 편광됩니다. 이전 인터페이스의 순수 1D `tetm` 경로도 2026-09-27부터 같은 뜻이었습니다
(그 전에는 비추는 모든 차수에 `+y`를 곱했는데, 비추는 차수 집합이 `kx = 0`을 사이에 두면 이는 다른
빛입니다. `tests/rcwa/test_excitation.py`의 장면에서 `sum(R)`이 0.3701이 아니라 0.4933이었습니다).
각 가중치는 비추는 면 위 `x = y = 0`, 즉 셀 중심에서의 그 빔의 진폭입니다 -- `IncidentOrders(weights=...)`가
받는 것과 같은 가중치이고, 같은 빛입니다. 이것도 2026-09-27부터입니다. 그 전까지는 가중치를 셀 모서리
기준으로 잡았으므로, `m + n`이 홀수인 모든 차수가 나머지에 대해 부호가 뒤집힌 채 비추어졌고, 같은
장면의 `sum(R)`이 두 번째로 0.3701 -> 0.1953으로 바뀌었습니다(§6).

## 1. 무엇에 쓰고, 무엇에 쓰지 않는가

정반사가 아닌 차수 하나만 비추는 데에는 **쓰지 않습니다**. `theta`에서 차수 `m`을 여기(excite)하는 것은
`kx`가 맞는 각도에서 정반사 차수를 여기하는 것과 *같은 평면파*입니다 — 차수 표지는 물리적 선택이 아니라
장부상의 선택입니다. 이동 항등식으로 검증했습니다. `theta`에서 풀고 차수 `+1`을 읽는 것과
`arcsin(sin(theta) + lambda/period)`에서 풀고 정반사 차수를 읽는 것이 시도한 모든 기저 크기에서
**~1e-9**까지 일치하며, 남는 차이는 두 풀이가 조금 다른 유한 창을 유지한다는 것뿐입니다. 따라서 차수
하나의 여기로 얻는 것은 표지 바꾸기입니다. (§4의 창 넓히기 전에는 9e-1 → 2.5e-4였는데, 이 자체가 창
넓히기가 하는 일을 보여 줍니다. `Numerics(orders=(N, 0))`에서 차수 +1을 비추면 이제 `[-N, N + 1]`을
유지하며, 이는 각도를 옮긴 풀이가 유지하는 차수에서 하나만 빠진 것입니다.) `ky == 0` 행에 대해 주의할
점이 하나 있습니다. 거기서 진폭은 `sign(kx) * y_hat`으로 표시되므로, 한 표기에서는 `kx`가 정확히 `0`이고
다른 표기에서는 `+-1e-17`(반올림)인 차수는 두 표기에서 반대 부호로 돌아올 수 있으며, 효율은 일치합니다.

결맞는 중첩에는 **씁니다**. 이는 어떤 단일 일반 풀이로도 표현할 수 없습니다. 구체적으로는 결맞는 빔
결합(coherent beam combining)입니다. N 대 1 결합기로 쓰는 격자는 격자 방정식이 정하는 각도의 빔 N개를
받는데, 이것이 바로 "차수 0과 차수 +1을 한꺼번에"입니다. 거꾸로 쓰는 회절 분할기도 같은 문제입니다.

합성에도 **쓰지 않습니다**. 위쪽의 다른 주기 구조에서 오는 빛은 이 계산 격자 위의 중첩이지만, 두 적층을
잇는 것은 산란 행렬에 대한 행렬 대수(`ScatteringMatrix.cascade`, `connecting-algorithms.md` §9)이지, 풀이에
넘기는 여기가 아닙니다.

## 2. 무엇을 얻는가

이 기능이 없을 때의 대안은 빔마다 일반 풀이를 한 번씩, 각 빔이 정반사가 되는 각도에서 하고, 차수 번호를
다시 맞춘 뒤 진폭을 더하는 것입니다. 이 방법은 동작하며 — 풀이는 입사 필드에 대해 선형입니다 —
**정확도도 같습니다**.

얻는 것은 풀이 횟수 — 빔 세 개에서 **2.95x**로 측정 — 와, 호출하는 쪽이 더 이상 하지 않아도 되는 장부
처리 두 단계입니다. 두 단계는 각각 조용히 틀릴 수 있는 곳입니다. 빔마다
`arcsin(sin(theta) + m*lambda/period)`를 계산하는 것, 그리고 더하기 전에 각 결과를 올바른 차수 번호만큼
옮기는 것입니다.

*더* 정확하기도 하지만, 그것은 §4의 창 넓히기 덕분일 뿐입니다. 정반사 차수에 중심을 둔 창에서는 둘이
측정상 동률입니다 — 기저 크기에 따라 승자가 제멋대로 바뀌고, `N = 8` 이상에서는 둘의 차이가 각자의 절단
오차보다 작습니다(§4의 표). 동률을 승리로 바꾸는 것은 창 넓히기입니다.

## 3. 비추는 모든 차수는 전파해야 합니다

라이트 라인(light line) 너머의 차수는 시간 평균 z 플럭스를 운반하지 않습니다. 그런 차수를 비추는 것은 거부합니다.
이전 인터페이스의 `excitation=`은 이름을 밝혀, 한 번에 차수 하나씩 거부합니다.

```
the excitation lights order 5, which is evanescent or carries no positive incident power in
this medium; a zero-power near-field excitation cannot be normalized as an incident plane wave.
```

그런 차수를 비추는 `IncidentOrders`는 `ValidationError`(코드 `rcwa.invalid`)를 일으킵니다.
`top input carries no positive incident normal power (evanescent or grazing).`

이는 까다로움이 아닙니다. 모든 효율은 `outgoing power / incident power`이고, 소멸파 입력은 분모에 아무것도
보태지 않으면서 구조를 구동합니다 — 그래서 수동 적층에서 `R + T`가 1을 넘고(거부가 생기기 전에 열세 차수
여기에서 **8.87**로 측정), "효율"은 조용히 *입사 전력의 비율*이라는 뜻을 잃습니다. 소멸파 여기는 근접장
소스이며, 이는 입사 평면파와 다른 문제이고 이 라이브러리가 모델링하지 않는 문제입니다.

실제적인 결과는 결합할 수 있는 빔 개수의 상한입니다. 차수는 `|kx| < n_inc`일 때만 전파하므로, 쓸 수 있는
차수는 대략 `2 * period / wavelength`개입니다. 전파하는 차수가 적을 만큼 촘촘한 격자는 빔을 많이 결합할 수
없습니다 — 이는 이 코드의 한계가 아니라 물리입니다.

## 4. 유지 창은 비추는 차수 둘레로 넓어집니다

비추는 차수마다 양쪽에 `Numerics(orders=...)`만큼의 이웃이 필요합니다. 그렇지 않으면 중심에서 벗어난 빔이
혼자 풀 때보다 덜 분해됩니다. 그래서 유지 집합은 정반사 차수에 중심을 둔 채로 있지 않고,
`[m_min - N, m_max + N]`에 걸칩니다. 여기서 `N`은 선언한 반폭입니다.

그 집합은 범위가 아니라 `(centre, half_width)`로 운반됩니다 — 이전 인터페이스의 `solver.order_window`이며,
현재 `rcwa.check` 보고의 `resolved['order_window']`는 이 넓힌 창이 아니라 선언한 창을 보여 줍니다 —
그래야 `n_orders = 2*half_width + 1`이 계속 참이고 뒤따르는 모든 크기 계산이 손대지 않은 채로 남기
때문입니다. 대가는 대칭 창이 반정수 중심에 걸칠 수 없다는 것이므로, 폭이 홀수이면 합집합이 엄밀히 필요로
하는 것보다 차수를 하나 더 씁니다.

**이것이 이 기능을 우회 방법과 같은 수준이 아니라 더 낫게 만드는 점입니다.** 차수 0, +1, +2에 빔 세 개를
두고, 수렴한 기준값에 대한 차수 -2..+2의 오차입니다.

:::{admonition} 이전 인터페이스
:class: earlier-interface

이전 `wavebit.compat.legacy` 인터페이스로 측정했으며, 그 `truncation_order`는 현재
`rcwa.Numerics(orders=(N, 0))`의 반폭 `N`입니다.

| truncation_order | 합집합 창 | 차수 수 | 따로 푼 세 번 | 차수 수 |
| --- | --- | --- | --- | --- |
| 6 | 4.01e-1 | 15 | 3.48e-1 | 39 |
| 8 | **2.51e-1** | 19 | 4.85e-1 | 51 |
| 12 | **2.14e-2** | 27 | 3.07e-2 | 75 |
| 16 | **8.30e-3** | 35 | 1.00e-2 | 99 |
| 20 | **4.34e-3** | 43 | 5.05e-3 | 123 |
| 24 | **2.09e-3** | 51 | 2.47e-3 | 147 |

`truncation_order` 8 이상에서는 차수의 3분의 1만 쓰고도 매번 이깁니다. (6 행은 수렴하지 않은 영역이며 —
두 오차 모두 ~0.4 — 아무 의미가 없습니다.) 창 넓히기가 없으면 둘은 측정상 동률이며, 이 기능의 이전
초안이 그랬습니다.
:::

추가 차수는 싸고, 점점 더 싸집니다. `m_max - m_min`개, 즉 **고정된** 개수이고, 이에 비해 `n_orders`는
커집니다. **3.6–9.1%** 수준으로 측정되었고, N이 커질수록 줄어듭니다.

**비추는 차수 자체는 선언한 `Numerics(orders=...)` 안에 있어야 합니다.** 이 규칙이 없으면 창 넓히기에
한계가 없어집니다 — `orders=(6, 0)`에서 비추는 차수 `(0, 0)`과 `(9, 0)`은 조용히 차수 22개를 유지할
것입니다 — 그러면 `orders`는 모델링하는 차수라는 뜻을 잃습니다. 이 규칙이 있으면 창 넓히기는 선언한 반폭의
두 배로 제한되고, 바깥의 차수를 비추면 이름을 밝혀 거부합니다.

창 넓히기가 건드리는 것은 모두 *크기*나 *오프셋*입니다. `kx` 생성기 네 개, 인수분해 진입점 두 개, 그리고
`lit_orders`의 사상입니다. **인수분해는 창이 어디에 있는지 상관하지 않습니다** — 합성곱 행렬의 `(i, j)`
성분은 조화 성분 `i - j`이므로, 같은 `orders`에서 아주 다른 각도로 푼 두 풀이는 비트 단위로 같은 칸을
만듭니다(0.0으로 측정). 인수분해로 넘어가는 것은 폭뿐입니다.

## 5. 어디로 들어가는가

입사 열인 `delta`는 정합마다 하나씩, 네 곳에서 정반사 차수 위의 단위 벡터로 만들어졌습니다.

| | |
| --- | --- |
| `solve_1d.py` 3단계 | 순수 1D TE/TM 정합 |
| `solve_1d.py::final_matching` | conical, 2D, 1차 경로가 공유하는 정합 |
| `scattering.py::smm_1d` | 첫 번째의 SMM 쌍둥이 |
| `scattering.py::_smm_sp` | 두 번째의 SMM 쌍둥이 |

이제 네 곳 모두 `incident_column`을 받습니다. `lit_orders`는 `incident_sp_amplitudes`가 편광을 해석하는
방식대로 경계에서 선언을 한 번 해석하며 — 아무것도 선언하지 않았으면 `None`을 반환하고, 이것이 일반
경로를 손대지 않은 채로 유지합니다.

**키는 구조이고 값은 구조가 아닙니다.** 열의 어느 성분이 0이 아닌지가 배열 모양을 정하므로, 키는 Python
정수로 읽고 추적한 값과는 결코 비교하지 않습니다. 그 옆의 진폭은 일반 설계 변수이며 그래디언트를
운반합니다(자동 미분끼리의 일치가 아니라 차분 몫과 비교해 확인했습니다).

## 6. 차수별 편광: 이 키워드가 아니라 행렬을 거쳐

`IncidentOrders`는 이전 `excitation=`처럼 차수마다 진폭 하나와 **모든 차수에 공통인 편광 하나**를
운반합니다 — `polarization`이 무엇이든, 각 차수의 s/p 좌표계에서입니다. 이는 정식화의 한계가 아니라 범위에
대한 결정이며, 그 기능은 이미 다른 방법으로 쓸 수 있습니다.

**`result.scattering`을 거쳐.** 결합된 경로에서 입사 열은 길이가 각각 `n_orders`인
`[s-block ; p-block]`이므로, 두 절반을 독립적으로 채우는 것이 *곧* 차수별 편광입니다. s 절반은 s 진폭
자체이고, p 절반은 **포트 변수**, 즉 비추는 매질에서 p파의 자기장 `-1j * (n/mu) * a_p`입니다. 돌아오는 p
절반은 반사된 자기장이며, 위로 가는 파에서는 뒤집힙니다.

```python
result = rcwa.run(problem.replace(outputs=rcwa.Outputs(scattering=True)),
                  numerics=rcwa.Numerics(orders=..., algorithm='smm'))
S = result.scattering[0]                      # one matrix per wavelength
h = n_lit / mu_lit                            # the lit medium's admittance, Z0 units

column = np.zeros((2 * n_orders, 1), dtype=complex)
column[n_orders // 2] = 1.0                   # order 0: s, amplitude 1
column[n_orders + n_orders // 2 + 1] = -1j * h * 0.5j   # order +1: p, amplitude 0.5j
reflected = S.S11 @ column                    # both halves respond, from one solve
r_s = reflected[:n_orders]                    # the s amplitudes
r_p = reflected[n_orders:] / (1j * h)         # the p amplitudes, out of the port variable
```

확인 결과: *공통* 편광에서 이 경로와 `excitation`은 s 절반에서 6e-17까지, p 절반은 위처럼 읽으면 4e-17까지
일치합니다(`S11 @ delta_p`는 열 자체의 p 변수 단위당 `-r_p`이고, `S21 @ delta_p`는 `t_p`에
`(n/mu)_exit / (n/mu)_lit`을 곱한 값입니다. `docs/reference/common/conventions.md`가 모든 경로에 대해 이를 한
번에 명시합니다). 행렬 경로가 주지 않는 것은 효율입니다 — 진폭을 돌려주고 정규화는 호출하는 쪽이 합니다 —
그리고 SMM 전용입니다. `Numerics(algorithm='etm')`과 함께 쓴 `Outputs(scattering=True)`는 거부합니다(코드
`rcwa.scattering_algorithm`). 이전 인터페이스의 순수 1D 경로의 행렬은 같은 변수로 나타낸 이 행렬의 s-s(TE)
또는 p-p(TM) 블록이며, 그 포트가 어느 쪽인지 알려 줍니다(`S.entry_port.channel`이 거기서는 `'te'` /
`'tm'`, 여기서는 `'sp'`).

**두 입구(door)는 일치합니다(2026-09-27부터).** `excitation`의 가중치와 결맞는 소스 경로의 `IncidentOrders` 가중치는
모두 공개 API가 측정 기준으로 삼는 셀 **중심**인 `x = y = 0`을 기준으로 하므로, 사전 하나가 두 입구를 같은
빛으로 비춥니다. `phi = 0`, `polarization=(1, 0)`인 `tests/rcwa/test_excitation.py`의 장면에서
`{0: 1, 1: 0.6-0.8j}`는 어느 쪽으로든 `sum(R)` 0.151782를 주며, 차이는 4.2e-16입니다(유리 쪽에서 비추면
0.241294). 그날까지 일반 입구는 풀이가 적분을 시작하는 푸리에 원점인 셀 **모서리**를 가중치 기준으로 삼았으므로,
차수 `(m, n)`을 나머지에 대해 추가 인자 `(-1)**(m + n)`을 붙여 비추었습니다 -- 이는 전역 위상이 아니라,
홀짝성이 반대인 두 차수를 비추면 다른 물리입니다. 같은 사전이 거기서는 0.452653을 주었습니다.
`tests/rcwa/test_centre_reference.py`에 고정되어 있고(역격자 이동 항등식으로서는
`tests/rcwa/test_coherent_illumination.py`에도), 기록은 `docs/theory/rcwa/amplitude-sp-frame.md` §7입니다.
현재 `rcwa.IncidentOrders`(이를 측정할 때는 `rcwa.OrderExcitation`)가 그 결맞는 소스 입구입니다. 같은 장면에서
위쪽의 나가는 전력 대 들어오는 전력은 같은 0.151782이며, 이전 `excitation=` 입구의 `sum(R)`과 4.6e-15
차이입니다(유리 쪽에서 비추면 0.241294, 8.9e-15. 2026-10-15 측정).

**왜 소스에 넣지 않는가.** s/p 기저는 *차수별*입니다. `order_azimuth`는 유지 차수마다 고유한 방위각을 주고,
`s_hat = z_hat x k_hat_par`는 **그 차수의** 파수 벡터로 만들어집니다. 그래서 차수 0의 "s"와 차수 +1의
"s"는 공간에서 서로 다른 방향을 가리키며, 차수별 `(a_s, a_p)`는 공통 좌표계가 아니라 각 빔을 각자의
좌표계에서 지정한다는 뜻이 됩니다. 이곳은 이 저장소에서 부호 오류가 가장 잘 나는 구석입니다 —
`order_azimuth`의 docstring에는 거기서 고친 버그 두 개(`kx < 0` 반회전과 원점의 `phi`)가 기록되어 있고,
테스트 모음은 구조상 차수별 진폭 부호를 보지 못합니다. 그 장치를 *입력* 쪽에도 두려면 별도의 외부 기준이
필요하며, 이는 별개의 작업입니다.

만약 언젠가 추가한다면 따를 것은 S4의 선택입니다. 차수별 s/p가 아니라 고정된 데카르트 `'x'` / `'y'`
좌표계를 쓰면 좌표계 문제 자체를 피할 수 있습니다.

## 7. 기준 솔버들은 어떻게 하는가

RETICOLO V10, S4, fmmax의 로컬 사본을 대상으로 조사했습니다.

| | 다중 모드 입사 | 차수별 편광 | 방식 |
| --- | --- | --- | --- |
| **fmmax** | 예 | 예 | 입사라는 개념 자체가 없음 — S 행렬을 제공하고 호출하는 쪽이 `(2n, k)` 진폭 배열을 채움. `amplitude_poynting_flux`는 *주어진 진폭으로부터* 플럭스를 계산하므로 정규화 문제가 아예 생기지 않음 |
| **S4** | 예 | 예 (`'x'` / `'y'`) | `SetExcitationExterior{{G-index, pol, {re, im}}, ...}`. `SetExcitationPlanewave` 옆의 **"low level"** 탈출구로 문서화됨. 목록 하나가 앞면-순방향과 뒷면-역방향을 다루므로 양쪽에 걸침 |
| **RETICOLO** | 아니요 | 아니요 | 고정된 입사(TE/TM x 위/아래)를 열거하고 각각을 보고함. `einc = [a_TE, a_TM]`은 파 하나의 편광 |
| **wavebit** | 예, `IncidentOrders` | `result.scattering`을 거쳐 | 흔한 경우에는 소스, 나머지는 행렬 |

여기서 세 가지가 따라 나옵니다.

**이를 지원하는 두 동료 솔버 모두 행렬을 거쳐 도달했고**, 그중 하나는 입사라는 개념이 아예 없습니다. 이는
이 저장소가 반대 방향에서 도달한 결론과 같습니다. 산란 행렬이 일반적인 답이고, `IncidentOrders`는 전력을
돌려받고 싶은 경우(이전 `excitation=`을 거친 차수별 효율)를 위해 그 위에 얹은 편의 기능입니다.

**S4는 자기 버전을 저수준이라 부르고 그 옆에 평면파 호출을 둡니다.** 여기의 구분도 같은 모양입니다 —
`IncidentOrders`는 고수준 입구이고 전력을 돌려주며, 행렬은 저수준 입구이고 진폭을 돌려줍니다.

**RETICOLO는 이 중 어느 것의 기준도 될 수 없습니다.** 이 저장소가 가장 많이 의지하는 기준이므로 중요한
점입니다. 그래서 `tests/rcwa/test_excitation.py`의 검사는 내부 검사(ETM 정합에 대한 SMM 연쇄)이며, 외부
기준은 fmmax나 S4가 되어야 할 것입니다.

## 8. 입사 전력과 실수 각도

입사 차수의 법선 방향 파수 벡터는 그 반공간 모드에서 읽습니다. 손실 없는 양의 굴절률 매질에서 실수 각도의
경우 `n_inc*cos(theta)`입니다. 흡수하거나 음의 굴절률을 가진 실수 각도 입사에는 [실수 각도
구성](../../reference/common/conventions.md#real-angle-incidence)을 쓰세요. 복소 `n_inc`로 쓴 같은 식은 그
법선 방향 파수 벡터가 아닙니다.

비추는 각 차수는 자체의 실수 전력 어드미턴스로 가중됩니다. 전기 s 진폭에는 `Re(kz/mu)`, 자기 p 진폭에는
`Re(kz/epsilon)`이며, 전기-자기 p 스케일을 포함합니다. 입사 전력은 이 가중치에 진폭 크기의 제곱을 곱한
합입니다. 단일 `PlaneWave`는 중앙 차수 하나를 비춥니다.

실수 횡방향 파수 벡터에서 유한 적층의 흡수는 `result.absorption`,
`result.outputs['power_entering'] - result.T.sum(axis=(1, 2))`입니다. 입사·반사 간섭은 `power_entering`에
포함되므로, 이를 `1-R.sum()`으로 바꾸려면 입사 매질에 손실이 없어야 합니다.
