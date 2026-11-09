# 가우시안 빔

`fdtd.GaussianBeam`은 면에서 집속 가우시안 빔을 한쪽으로 내보냅니다.

```python
from wavebit import Units, fdtd

beam = fdtd.GaussianBeam('b', plane=fdtd.Plane(z=-4.), direction='+z', waist=1.5,
                         focus=(0., 0., 0.), polarization='y',
                         pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=2e13))
```

## 무엇을 내보내는가

초점면에서 `polarization`(가로 축) 방향의 E 성분은 `focus`를 중심으로 `exp(-r**2 / waist**2)`이고
다른 가로 성분은 0입니다. `waist`는 `rcwa.GaussianBeam`과 같이 진폭이 1/e가 되는 반지름입니다.
빔은 그 면의 각 스펙트럼(장을 평면파들의 합으로 나타낸 것) 가운데 전파하는 부분입니다. 광원 면을
채운 균일 매질에서 격자 자신의 평면파로 만들므로, 펄스 중심 파장에서 격자의 정확한 해입니다.
`focus`는 그 매질이 공간을 채웠다면 빔이 모일 곳입니다. 다른 매질로 들어가는 빔은 굴절이
데려가는 곳에 모입니다.

`amplitude`의 단위는 와트의 제곱근입니다(불변 축이 있으면 그 축 미터당). 진폭 `A`인 연속파는
`A**2` 와트를 나릅니다. `phase`는 라디안이고 `delay_seconds`는 펄스를 늦춥니다. 2차원
문제(`InvariantAxis` 하나)에서 빔은 판 모양 빔으로, 분해된 가로 축으로만 모입니다.

면은 `ModeSource`와 같은 한쪽 전체장/산란장(TFSF) 시트입니다([도파관 모드](waveguide-modes.md)
참고). 중심 파장에서는 빔을 정확히 한쪽으로만 내보냅니다. 중심에서 벗어나면 평면파마다 단면과
반 칸 위상이 중심의 것이어서 빔이 뒤로도 조금 나갑니다. [검증 기록](../../validation/fdtd/fd-gaussian-beam.md)이
그 부분을 정확한 예측과 비교해 잽니다.

## 유한 격자에서의 정확도

빔은 연속체가 아니라 격자에 대해 정확합니다. 격자의 평면파는 방향에 따라 위상 속도가 조금씩
다르므로, 빔이 초점을 지나며 얻는 위상(구이 위상)이 연속체 빔과 조금 다릅니다.
[검증 기록](../../validation/fdtd/fd-gaussian-beam.md)(진공, 파장 1 µm, 허리 1.5 µm)에서 그 차이는 초점에서
3 µm 안쪽에서 격자 간격 0.05 µm(파장당 20칸)일 때 최대 2.2e-3 rad이고, 0.025 µm일 때 5.4e-4 rad입니다.
오차는 2차로 줄어듭니다. 빔을 따라가는 위상에 기대는 결과는 격자 간격 수렴을 확인하세요. 초점면 허리,
전력, 뒤쪽 복사는 예측과 훨씬 더 가깝게 맞습니다.

## 요구 조건

- 면을 가로지르는 경계면은 `Open`(또는 `InvariantAxis`)입니다. 금속, 거울, 주기 경계면은
  거절됩니다(`source.boundary`). 빔이 그 도파관이나 격자의 모드가 되어 버리기 때문입니다.
- 면 위의 매질은 균일하고(`source.medium`) 손실과 분산이 없으며, 축을 따라 양쪽 한 칸씩 같아야
  합니다(`solve_modes`의 창 규칙).
- 단면이 빔을 담아야 합니다. 면의 가장 바깥 칸에서 빔이 최고값의 1e-3 이하여야
  합니다(`source.beam_size`). 시트는 창(`Open` 면에서 한 칸 안쪽)에서 멈추고 그 너머로는 아무것도
  내보내지 않습니다.
- `waist`와 `focus`는 구체적인 수여야 합니다(`source.traced`). 빔은 호스트에서 만듭니다.
  `amplitude`, `phase`, `delay_seconds`와 다른 곳의 구조는 추적된 값을 지닐 수 있습니다. 펄스
  파장은 안 됩니다(`mode.traced`).
- 수직 입사만 됩니다. 빔의 축은 면의 법선입니다. 시트에서 한 칸 안쪽의 플럭스 측정기는
  거절됩니다(`monitor.source_sheet`).

## 기울기와 메모리

JAX와 Torch 백엔드에서 빔의 `amplitude`, `phase`, `delay_seconds`나 다른 곳의 추적된 구조에
대한 실행의 미분은 실행 전체의 역방향 미분으로 구합니다
({ref}`실행 미분하기 <fdtd-model-differentiating>` 참고). 기본 설정에서 역방향 미분은 모든 시간
스텝을 저장하므로 메모리가 스텝 수에 따라 늘어납니다. CPU의 JAX 백엔드에서는 검증 설정만큼 긴
실행(약 1만 2천 스텝)이 메모리에 들어가지 않았습니다. 긴 실행에는 `Numerics(checkpoint_steps=K)`를
설정하세요. `K` 스텝 구간을 저장하지 않고 다시 계산하며, JAX와 Torch, CPU와 CUDA 모두에서 쓸 수
있습니다. `K**L` 스텝 실행은 약 `L * K`개 상태만 저장합니다. 고정 `Duration`이 필요하며
`DecayStop`과는 함께 쓸 수 없습니다. 구간 일정과 측정한 메모리는
[체크포인트 기록](../../validation/fdtd/checkpointing.md)에 있고, 이를 거쳐도 이 빔의 미분은
일반 실행의 값과 1e-12까지 같습니다([검증 기록](../../validation/fdtd/fd-gaussian-beam.md)).
