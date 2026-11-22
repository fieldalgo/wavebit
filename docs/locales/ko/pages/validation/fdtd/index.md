# FDTD 검증

공간, 시간, 재료, 경계에 관한 연구입니다. 각 보고서는 측정한 구성과 남은 한계를 밝힙니다.

현재 공개 API: [입사파 보조 격자를 포함한 자동 종료](fd-source-decay.md),
[Native Yee 격자의 CUDA 검증](native-cuda.md), 그리고 이와 별도의 근거로
[범위를 한정한 일반 장면 CUDA 실행](general-cuda.md)과
[JAX·Torch 역방향 미분의 메모리 상한](checkpointing.md).
[면적 적분 도형 FDTD 미분의 외부 jit 메모리](fdtd-jit-painting-memory.md)는 현재 주기 API와 연구용
엔진을 함께 측정합니다.

아래 기록들은 공개 `wavebit.fdtd` 모듈(2026-10-13)보다 먼저 작성되었고, 측정한 인터페이스의 이름을
그대로 씁니다. 기록이 `wavebit.fd`(`fd.Scene`, `fd.plan`, `fd.run`, `fd.check`)라고 부르는 일반 엔진은
`wavebit.compat.legacy.fd`로 남아 있었고, `Absorber`와 `Decay`는 거기에만 있었습니다. `fdtd-*` 기록이
측정하는 주기 적층 솔버는 `wavebit.compat.legacy.fdtd`로 남아 있었습니다. 둘 다 2026-10-15에
퇴역했으며, [퇴역 직전 리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)에서
실행됩니다. 현재 사용법은 [FDTD 가이드](../../guide/fdtd/general-fdtd.md)와
[예전 일반 엔진 설명서](../../migration/general-fdtd-v1.md)를 보세요. "`src/wavebit/fd` 기준"이라고
인용한 소스 다이제스트는 소스 트리가 `src/wavebit/fdtd/`로 옮겨지기 전에 그 실행들이 트리를
해시한 방식을 말합니다.

- [섞인 재료 매개변수와 극점 산술 거부](fd-pole-arithmetic.md)
- [한 칸에 지나가는 두 스무딩한 표면: 다면체 자르기 (4단계, C10)](fd-polytope-clip.md)
- [대각 밖 구성 칸 (4단계, C11): 등록한 관문의 결과](fd-stage4-c11.md)
- [4단계 보고: 모양과 정확도 (C0-C12)](fd-stage4-report.md)
- [C8: 유전체 안의 전도성 박막 하나](fd-conductive-film.md)
- [C8: 떨어져 있는 평행 전도성 박막들](fd-conductive-stack.md)
- [C8: 정확히 맞닿은 두 전도성 박막](fd-conductive-contact.md)
- [교환 가능한 결합 유전체의 정합 흡수](fd-constitutive-absorber.md)
- [C8: 유전체 안의 Debye 박막 하나](fd-debye-film.md)
- [C8: 떨어져 있는 Debye 박막들](fd-debye-stack.md)
- [C8: 박막 하나 안의 유한한 Debye 합](fd-debye-sum-film.md)
- [과도 응답 뒤의 자동 종료](fd-decay-stopping.md)
- [C8: Debye와 Drude/Lorentz가 섞인 박막](fd-hybrid-film.md)
- [Yee 격자와 일관된 복사](fd-lattice-radiation.md)
- [RCWA와 비교한 주기 메타표면 (4단계, C7)](fd-metasurface-validation.md)
- [Mie 기준과 곡면 계면 정확도 관문](fd-mie-validation.md)
- [C8: 떨어져 있는 단일 진동자 박막들](fd-oscillator-stack.md)
- [C8: 박막 하나 안의 유한한 Drude/Lorentz 합](fd-oscillator-sum-film.md)
- [C8: 유전체 안의 Drude 또는 Lorentz 박막 하나](fd-single-pole-film.md)
- [JIT 경계를 가로지르는 구 자르기](fd-sphere-clip-jit.md)
- [스무딩한 점유율과 기울어진 계면 측정](fd-subpixel-occupancy.md)
- [C8: Debye 극점 두 개를 가진 박막 하나](fd-two-debye-film.md)
- [이산 Yee 계의 도파관 모드 (7단계)](fd-waveguide-modes.md)
- [흡수층을 지나는 칸 맞춤 도파관 (7단계)](fd-waveguide-through-pml.md)
- [한쪽으로만 내보내는 도파관 모드 광원 (7단계)](fd-mode-source.md)
- [집속 가우시안 빔 (7단계)](fd-gaussian-beam.md)
- [수반법, 이산 미분과 물리적 정확도](fdtd-adjoint-validation.md)
- [Bragg 공진기의 시간·공간 수렴](fdtd-bragg-cavity.md)
- [원형 FDTD 형상과 지름 미분](fdtd-circle-gradients.md)
- [주파수 모니터 부분 표본화의 비용과 오차](fdtd-dft-sampling.md)
- [방향별 FDTD 추출과 레일리 근처에서 지속되는 모드](fdtd-directional.md)
- [회전한 타원 FDTD 형상과 미분](fdtd-ellipse-gradients.md)
- [손실 있는 ENZ 박막과 보정된 산란 수렴](fdtd-enz.md)
- [실험을 바꾸지 않는 반공간 미분](fdtd-halfspace-gradients.md)
- [격자·경계·유한 시간 산란 오차 분리하기](fdtd-lattice-reference.md)
- [평면 분산 층 경계 맞추기](fdtd-layer-alignment.md)
- [고정 메시 위의 내부 층 미분](fdtd-layer-gradients.md)
- [연속적인 내부 계면 표본화](fdtd-layer-smoothing.md)
- [남은 Lorentz 운동과 분산 매질의 안정화](fdtd-lorentz-decay.md)
- [저손실 Drude/Lorentz 혼합 금속](fdtd-mixed-metal.md)
- [제약이 있는 다층 최적화](fdtd-multilayer-optimization.md)
- [비스듬한 펄스의 대역폭과 금속 계면](fdtd-oblique-pulse.md)
- [회전한 FDTD 직사각형 여러 개의 주기적 칠하기](fdtd-painted-shapes.md)
- [레일리 문턱 근처의 FDTD CPML 세기](fdtd-pml-strength.md)
- [펄스 안정화와 공간 정확도](fdtd-pulse-convergence.md)
- [소스 스펙트럼의 영점과 남은 차단 오차](fdtd-pulse-notches.md)
- [레일리 근처 채널과 회절 평면 위치](fdtd-rayleigh.md)
- [가로 방향 직사각형 미분](fdtd-rectangle-gradients.md)
- [FDTD 링다운 맞춤과 지수적이지 않은 차단 꼬리](fdtd-ringdown.md)
- [회전한 직사각형의 면적 적분](fdtd-rotation-gradients.md)
- [회전한 직사각형의 정확한 평행 이동 평균](fdtd-rotation-smoothing.md)
- [FDTD 경로 일치: 두 엔진은 같은 오차를 낼까?](fdtd-route-parity.md)
- [고정 격자에서의 입사각 민감도](fdtd-scene-angles.md)
- [가로 메시 하나 위의 방위각 미분](fdtd-scene-azimuth.md)
- [독립적인 소스 위치와 경계 여유](fdtd-source-clearance.md)
- [독립적인 가로 메시와 축별로 일관된 재료 평균](fdtd-transverse-resolution.md)
- [움직이는 바깥 기준면을 쓴 전체 두께 미분](fdtd-variable-depth.md)

## 공개 입력과 준비 단계 제어

`tests/fdtd/test_public_input_followup.py`는 현재 공개 API를 시험하며, 위 연구들이 쓰는 과거
네임스페이스에 의존하지 않습니다. 소스 체크아웃에서, 그 체크아웃의 `src`를 `PYTHONPATH`에 두고
실행하세요.

```sh
python -m pytest tests/fdtd/test_public_input_followup.py -q
```

이 테스트는 `PlaneWave`에 Plane이 아닌 입력을 주었을 때의 `check`, `prepare`, `run`, 올바른 평면과
물리 도메인 밖의 평면, 스텐실 메모리 한도와 잘못되었거나 쓰이지 않는 설정, 구조물이 여러 개일 때의
공개 해결책을 다룹니다. `Numerics(memory_bound=...)`는 바이트 단위 한도를 기존 호스트 예측 보호
장치에 넘깁니다. 메모리 경로 테스트는 희소 형상 조립에 들어가는 지점에서 멈춥니다. 즉 허용 경계와
바뀌지 않은 기본 소스를 확인할 뿐, 조립된 연산자나 산란 정확도를 확인하지 않습니다. 구조물이 여러
개일 때의 해결책도 `check`로 적용해 `Numerics(discretization="voxel")`이 그 거부를 없애는지
확인합니다.

[현재 입력·메모리 계약](../../guide/fdtd/model.md)을 보세요.

### 공개 grazing 안내

`tests/fdtd/test_public_grazing_guidance.py`는 좁은 대역의 비스듬한 입사 열에 대해 `check`,
`prepare`, `run`이 실제로 내는 메시지를 검사합니다. 출력된 공개 후보를 하나씩 실행하고 같은 물리
문제를 다시 검사하며, 여기에는 더 강한 기존 제어, 따옴표로 인용한 소스 이름, 순수 거부와 안정화
거부, 서로 다른 고정 기준 매질, 측정된 하한, 공개 호출 뒤에도 바뀌지 않은 내부 엔진 안내가
포함됩니다. 흡수층 셀을 더하면 바깥 패딩이 늘어나고 다른 흡수층 옵션은 유지되는지도 확인합니다.

```sh
python -m pytest tests/fdtd/test_public_grazing_guidance.py -q
```

이것은 진단과 준비 단계를 검증할 뿐, 시간 진행이나 새로운 산란 정확도 주장은 포함하지 않습니다.
수치 규칙의 근거는 여전히 기존 grazing 반환 측정과 PML 안정성 측정입니다. 함수 단위의 제품 변이
시험과 소스의 정확한 복원은 `tests/fdtd/public_grazing_mutation_evidence.json`에 기록되어 있습니다.

### 공개 소스 허용

`tests/fdtd/test_public_source_admission.py`는 잘못된 소스 형식, 유효하지 않거나 중복된 이름,
세 소스 형식 모두에서 step·period 실행 시간과 함께 펄스 선언이 빠지거나 잘못된 경우, Region이
아닌 TFSF 상자를 시험합니다. 같은 필드 경로를 `check`, `prepare`, `run`, `Prepared.validate`,
`Prepared.run`으로 검사한 뒤, 유효한 준비 결과를 다시 사용합니다. 유효한 소스와 물리 도메인 밖의
Region은 기존 허용 동작을 그대로 유지합니다.

```sh
python -m pytest tests/fdtd/test_public_source_admission.py -q
```

이 테스트는 준비와 입력 거부 단계에서 멈추며, 전파 정확도 주장을 더하지 않습니다. 실제 보호
장치와 진입점 변이 시험, 그리고 이어진 소스의 정확한 복원은
`tests/fdtd/public_source_mutation_evidence.json`에 기록되어 있습니다.
