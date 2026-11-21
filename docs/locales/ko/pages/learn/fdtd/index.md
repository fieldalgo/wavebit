# FDTD 배우기

FDTD는 전기장과 자기장을 시간에 따라 진행시킵니다. 쓸모 있는 답은 물리 문제와, 그 문제에서 가장 빠른 변화를 분해하는 이산화 둘 다에 달려 있습니다. 결과를 역설계에 쓰기 전에 아래 순서를 따르세요.

## 1. 물리 문제 정의하기

[일반 FDTD 가이드](../../guide/fdtd/general-fdtd.md)와 [FDTD 솔버가 문제를 모델링하는 방식](../../guide/fdtd/model.md)부터 시작하세요. 유한한 도메인, 물체, 소스, 경계 조건, 요청할 모니터를 지정합니다. 물리 도메인과 그 바깥의 흡수 경계 영역을 구분하세요. 평면파로 비추는 주기 층 구조라면 [주기 FDTD 가이드](../../guide/fdtd/time-domain.md)를 따르세요.

## 2. 공간과 시간 분해하기

공간 해상도는 재료 안에서의 파장과 형상의 특징 크기로 정합니다. 안정적인 시간 간격을 정한 뒤, 요청한 스펙트럼 양이 더 이상 변하지 않을 때까지 실행 시간을 늘리세요. 안정적인 시간 간격은 수렴 결과가 아닙니다. [가로 방향 해상도](../../validation/fdtd/fdtd-transverse-resolution.md), [DFT 표본화](../../validation/fdtd/fdtd-dft-sampling.md), [펄스 수렴](../../validation/fdtd/fdtd-pulse-convergence.md)을 비교해 보세요.

## 3. 소스와 경계를 분리하기

펄스의 범위, 모니터 위치, 흡수 경계까지의 여유를 각각 따로 확인하세요. [소스 여유 연구](../../validation/fdtd/fdtd-source-clearance.md)와 [PML 세기 연구](../../validation/fdtd/fdtd-pml-strength.md)는 흡수층 세기만 올려서는 스펙트럼을 보증할 수 없는 이유를 보여 줍니다.

## 4. 요청한 관측량 읽기

필드 표본, 푸리에 진폭, 적분한 전력은 서로 다른 질문에 답합니다. 모니터의 정의와 기준면을 사용하세요. RCWA와 비교하기 전에는 반사율만이 아니라 [좌표, 층 순서, 위상 원점, 편광 좌표계](../../reference/common/conventions.md)를 맞추세요. [경로 일치 연구](../../validation/fdtd/fdtd-route-parity.md)에 비교 범위가 기록되어 있습니다.

## 5. 재료와 그래디언트 더하기

전도성 매질이나 분산 매질을 지정하기 전에 [구성 응답과 흡수층](../../guide/fdtd/fd-constitutive-absorber.md)을 읽으세요. 그다음 [미분 가이드](../../guide/fdtd/fd-differentiation.md)를 따르세요. 연속 입력을 미분하는 동안에는 이산적인 배치 선택을 고정하고, 미분을 유한 차분과 비교하고, 미분뿐 아니라 순방향 계산도 세밀하게 하세요.

## 6. 응용 예 고르기

[복사](../../guide/fdtd/fd-radiation.md), [메타표면](../../guide/fdtd/fd-metasurface.md), 또는 [다층 최적화 연구](../../validation/fdtd/fdtd-multilayer-optimization.md)로 이어 가세요. 각 문서의 지원·수렴 진술은 시험한 구성에만 해당합니다.
