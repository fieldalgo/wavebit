# 일반 FDTD 엔진의 원거리장 복사

열린 장면에서 닫힌 측정 상자에는 `fdtd.FarFieldMonitor`를 사용하세요.
상자는 모든 물체를 감싸야 하고, 흡수층과 소스 면에서 떨어진 균일한 등방성 배경 안에 있어야 합니다.
모든 구조와 모든 `CurrentDipole` 스탬프는 셀 하나의 여유를 두고 상자 안쪽에 엄격하게 들어 있어야 합니다.
변환은 상자가 감싼 것만 복사하므로, `fdtd.check`와 `fdtd.prepare`는 상자 바깥의 쌍극자를 거부합니다
(`monitor.source_outside`).

```python
from wavebit import fdtd

monitor = fdtd.FarFieldMonitor(
    'ff', wavelengths=(1.,), box=fdtd.SnappedBox(size=(1.5,)*3), propagation='yee')
```

`fdtd.run`을 호출하기 전에 모니터를 `fdtd.Outputs(monitors=...)`에 넣으세요.
`propagation`은 표본화한 표면 필드를 무한대까지 전파하는 방식을 고릅니다.

| 값 | 의미 |
| --- | --- |
| `'continuum'` (기본값) | 원래의 연속체 그린 함수 근사입니다. 이전 결과를 보존합니다. 유한 해상도에서는 위상과 전력이 이산 격자 전파와 다를 수 있습니다. |
| `'yee'` | 균질한 Yee 격자의 출사 필드로, 방향 분산, 엇갈린 역격자 필드와 격자 에너지 플럭스 가중치를 포함합니다. |

연속체 모니터는 계획 단계에서 장면 전체에 대해 `RadiationApproximationWarning`(`wavebit.fdtd.diagnostics`에서
제공)을 냅니다. `fdtd.check`는 같은 내용을 측정된 약한 구(weak-sphere) 예와 함께
`report.capabilities['warnings']`에 나열합니다. 그 예의 오차는 여러분의 장면에 대한 상한이 아닙니다.

결과 `res`에서 `res['ff'].fields(theta, phi)`는 전기장의 theta/phi 투영을 반환합니다.
각도 단위는 라디안이며, theta는 +z에서, phi는 +x에서 +y 쪽으로 잽니다. 두 각도는 함께 브로드캐스트되며,
각 출력의 모양은 `(nf, ...angles)`입니다. `res['ff'].intensity(theta, phi)`는 펄스의 입체각당 양측
스펙트럼 에너지를 `res['ff'].units`(3차원 장면에서는 J/Hz/sr) 단위로 줍니다. 원거리장 지도를 만들기 전에
상자를 지나는 알짜 플럭스가 필요하면 같은 상자에 `fdtd.FluxMonitor`를 추가하세요. 이 모니터는 같은 면을 읽습니다.

`'yee'`에서는 각도가 출사 에너지 광선을 지정합니다. 점근 위상은 `k_star dot position`이며, 여기서 `k_star`는
격자 분산 곡면 위에 있습니다. 전기 편광은 이산 파수 벡터에 수직이며, 이 파수 벡터가 광선 방향을 따를 필요는
없습니다. 전력에는 `intensity()`를 사용하세요. 두 각도 투영만 제곱하면 있을 수 있는 반지름 방향 성분이
빠집니다. 세 개의 복소 카르테시안 진폭, 즉 모양이 `(nf, ...angles, 3)`인 `vector(theta, phi)`는 아직 현재
결과에 없습니다. 이전 인터페이스인 `wavebit.compat.legacy.fd`에는 2026-10-15에 퇴역하기 전까지 있었습니다.

기록된 필드는 JAX와 Torch에서 계속 미분할 수 있습니다. Yee 배경 매개변수와 요청한 각도는 구체적인 호스트
값이어야 합니다. 배경의 epsilon과 mu는 양의 스칼라여야 합니다. 계획 단계는 `omega*dt < pi`와
`0 < Omega*sqrt(epsilon*mu)*dx < sqrt(2)`(여기서 `Omega=2*sin(omega*dt/2)/dt`)를 만족하지 않으면 모든 주파수를
거부합니다. 이 조건은 유일한 볼록 출사 분산 곡면을 고릅니다. 지원 대역을 넓히는 수치적 클램프는 없습니다.

격자 지도를 고른다고 해서 재료 경계의 정확도가 확립되지는 않습니다.
요청한 관측값에 대해 시간, 경계, 공간 수렴을 확인하세요.
[유도와 측정된 검증](../../validation/fdtd/fdtd-lattice-radiation.md)을 참고하세요.
