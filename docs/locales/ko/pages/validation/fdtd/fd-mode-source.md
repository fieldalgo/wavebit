# 한쪽으로만 내보내는 도파관 모드 광원

작성일: 2026-10-27

대상: `src/wavebit/fdtd/_mode_source.py`, `src/wavebit/fdtd/_kernel/mode_sheet.py`,
`src/wavebit/fdtd/_sources.py`의 `ModeSource` 선언, `ModeMonitor`의 `source=` 옵션,
`tests/fdtd/test_fd_mode_source.py`.

`fdtd.ModeSource`는 도파관을 가로지르는 면에서 `fdtd.solve_modes`의 모드를 한쪽으로 내보냅니다.
이 면은 한쪽만 있는 전체장/산란장(TFSF) 시트입니다. 커널의 평면파 구성에서 입사장을 모드로
바꾼 것입니다. E 패턴은 시트의 노드 면에, H 패턴은 상류 쪽 반 칸 면에 있고, 각각 자기
파형 행을 갖습니다. H 행은 E 행을 모드의 반 칸 위상 `beta dx / 2`만큼, 그리고 반 칸 군지연만큼
앞당긴 것입니다. 그러면 펄스의 중심 주파수에서 두 행의 짝은 정확히 이산 모드가 되고, 시트는
그 모드를 한쪽으로만 내보냅니다. 다른 주파수에서는 단면과 반 칸 위상이 중심 주파수의 것이므로
시트는 반대쪽으로도 모드를 조금 내보냅니다. 이산 모드의 정확한 비켤레 직교성 덕분에 두 부분
모두 실제 파형 행으로부터 닫힌 식으로 나옵니다(유도는 `_mode_source.py` 모듈 설명에
있습니다). `source=`를 선언한 `ModeMonitor`가 이를 `injected`(주입)와 `leakage`(누설)로
보고합니다.

TFSF 구성은 표준적인 것입니다(Taflove와 Hagness, *Computational Electrodynamics*, 3판,
5장). 모드를 TFSF 면의 입사장으로 쓰고 H 면의 파형을 반 칸 앞당기는 것도 같은 생각을 따릅니다.
다른 솔버에서 가져온 코드는 없습니다.

## 등록한 합격 기준 (2026-10-27, 계산 전)

**설정 S.** 흡수층 통과 기록(`fd-waveguide-through-pml.md`) D1의 슬래브입니다. 유전율 2.085136의
클래딩 안에 폭 1.5 µm, 유전율 4인 코어, y는 `InvariantAxis`, x는 ±4.5 µm에서 `Open`, z는
±4 µm에서 `Open`, 코어는 양쪽 z 면을 지나 이어지는 `SnappedSlab`, 1.55 µm에 맞춘 24칸 `PML`.
z = -2 µm의 `ModeSource('src')`, 방향 `+z`, 모드 0(TE0), 진폭 1(미터당 와트의 제곱근), 펄스
중심 1.55 µm, `frequency_sigma_hz = 1.5e13`, `cutoff = 1e-12`. `source='src'`, `modes=1`,
파장 1.45, 1.55, 1.65 µm인 `ModeMonitor`: z = +1 µm의 `p1`, z = +2 µm의 `p2`, z = -3 µm의
`q`. 격자 간격 0.05와 0.025 µm. 각 실행은 `UnsettledFieldWarning`과 `GrowingFieldWarning`이
나지 않을 때까지 계속합니다. 아래의 `P(d)`는 그 파장에서 모드 자신의 격자 `beta`(`solve_modes`
값)로 거리 `d`만큼 진행하는 인자이며, 실행의 푸리에 규약을 따릅니다.

* **S1, 예측대로의 주입.** 모든 파장과 두 간격에서
  `|forward(p2) - injected * P(4 µm)| <= 1e-5 |injected|`. 거울상 실행(광원을 z = +2 µm에
  방향 `-z`로, 측정기를 z = -2 µm에 두고 그 후방 진폭을 읽음)에서도 0.05 µm에서 같은 기준.
* **S2, 예측대로의 누설과 그 값.** 모든 파장과 두 간격에서
  `|backward(q) - (leakage * P(1 µm) + backward(p2) * P(5 µm))| <= 1e-6 |injected|`.
  1.55 µm에서 예측한 누설 전력 `|leakage / injected|^2`는 1e-12 이하(중심에서는 한쪽으로만
  내보냄)이고, 1.45와 1.65 µm에서는 그 값을 보고합니다.
* **S3, 절대 전력.** 진폭 `A`인 연속파는 미터당 `A^2` 와트를 나르므로, 내보낸 모드의 순간
  전력은 단위 파형 `w`에 대해 `2 A^2 w(t)^2`이고 양측 스펙트럼 에너지는 `2 A^2 |W(f)|^2`입니다.
  `W`는 `w`의 연속 푸리에 변환입니다. 1.55 µm와 두 간격에서 `forward_energy(p2)`가 이 값과
  상대 2e-5 이내로 맞아야 합니다.
* **S4, 해상도에 따른 수렴.** 1.55 µm에서 실행 자체의 유효 굴절률
  `n_run = arg(forward(p2) / forward(p1)) / (k0 * 1 µm)`(솔버 굴절률에 가장 가까운 가지)가 같은
  간격의 솔버 격자 굴절률과 1e-9 이내로 맞고, 슬래브의 정확한 TE0 굴절률(1.9562181938, 모드 솔버
  기록의 M3)에 대한 오차가 0.05에서 0.025 µm로 갈 때 관측 차수 1.8 이상으로 줄어야 합니다.
* **S5, 맞대기 접합과 모드 정합 비교.** 설정 S에서 코어를 두 `SnappedBox`로 바꿉니다. z < 0은 폭
  1.5 µm(z- 면을 지나 이어짐), z > 0은 폭 1.0 µm(z+ 면을 지나 이어짐)이고, `p2`는 좁은 코어의
  TE0를 읽으며 `q`는 z = -1 µm로 옮깁니다. 1.55 µm에서 TE0 전력 투과
  `T = forward_energy(p2) / injected_energy`와 반사 `R = backward_energy(q) / injected_energy`를
  같은 접합의 연속체 모드 정합 값과 비교합니다. 모드 정합은 테스트 안에 직접 씁니다. 완전 도체
  상자 안 두 슬래브의 TE 모드를 1차원 연속 연산자의 미세 유한차분 풀이로 구하고, z = 0에서
  Ey와 Hx의 연속 조건으로 잇고, 모드 수와 상자 폭에 대해 수렴시킵니다. 기준:
  `|T - T_mm|`가 0.05 µm에서 5e-3 이하, 0.025 µm에서 2e-3 이하이고 더 고운 간격에서 더 작음.
  `|R - R_mm|`가 0.025 µm에서 `0.2 R_mm` 이하이고 0.05 µm보다 작음.
* **S6, 백엔드.** 0.05 µm의 설정 S를 NumPy, JAX, Torch에서 실행해 `injected`, `leakage`,
  `forward(p2)`, `backward(q)`가 `|injected|` 대비 1e-12 이내로 일치.
* **S7, 미분.** JAX와 Torch에서 0.05 µm로: 1.55 µm의 `forward_energy(p2)`를 광원 `amplitude`로,
  `Re forward(p2)`를 광원 `phase`와 `delay_seconds`로, 그리고 `forward_energy(p2)`를 광원과
  `p1` 사이에 있고 두 면 어디에도 닿지 않는 추적된 `Box` 산란체의 유전율로 미분한 값이 각각
  중앙 차분(두 간격, Richardson 외삽)과 상대 1e-6 이내로 일치.
* **S8, 코드별 거부.** 면의 법선을 따르지 않는 방향(`source.direction`), 음수이거나 정수가 아닌
  모드(`source.mode`), 찾은 갇힌 모드 수를 넘는 모드 번호(`mode.count`), 광원 면에 닿는 추적된
  구조(`mode.traced`), 물리 영역 밖의 면(`source.outside_domain`), 시트에서 한 칸 안쪽의
  측정기(`monitor.source_sheet`), `ModeSource`를 가리키지 않는 `source=`(`monitor.source`),
  추적된 펄스 파장(`mode.traced`), 엔진 자리표시자 `fd.ModeSource`(공개 광원을 알려주는
  `NotImplementedError`).

## 결과

계산 전입니다.
