# 출사 회절차수 전파

`wavebit.postprocess.propagation`은 RCWA/FDTD의 공통 결과를 받는 후처리 계층이다.
첫 구현은 continuum 회절차수의 복소 전기장을 균일 외부 매질에서 직접 전파한다.
호출 예제와 정확한 계약은 [영문 원문](../../../../guide/postprocess/propagation.md)을 따른다.

입력은 Cartesian CW 진폭(V/m), 파수, 기준면·횡방향 원점·외향 법선·단위·시간 규약을
가진 `DiffractionOrders`다. RCWA의 reflection/transmission을 연결할 수 있다.
R/T만 있는 결과, 원시 pulse 스펙트럼, 정보가 누락되거나 Yee 분산을 사용하는 FDTD 결과는 거부한다.

`Plane`의 x/y는 유한한 실수 1차원 좌표이며 등간격일 필요가 없다. distance는 기준면에서
바깥으로 향하는 0 이상의 거리다. 아래쪽 포트에서는 public z가 감소한다. 출력은
(wavelength, y, x, xyz)의 `FieldSamples`이며 E는 V/m, 성분별 절댓값 제곱은 V²/m²다.
좌표와 파장은 목표면의 길이 단위를 따른다. 정확한 전력 유속은 반환하지 않는다.
원래 metadata와 기준면 정보는 diagnostics에 보존하고 NaN 진폭을 0으로 바꾸지 않는다.

범위는 z 법선 포트, 실수 횡파수, 균일·등방·비자성·양의 무손실 외부 매질이다.
복소장을 합산하므로 간섭과 소멸파 감쇠를 유지한다. 유한 개구/가장자리 산란을 새로 계산하지 않는다.
일반 평면장 ASM과 FDTD 정규화·분산 adapter는 후속 기능이다.

NumPy/JAX/PyTorch CPU eager 계산 및 진폭·거리·상류 RCWA 설계변수 미분을 지원한다.
추상 JAX JIT 검증은 명시적으로 거부하며 GPU·배치 변환은 검증하지 않았다.
양의 정수 chunk_size는 forward 관측점 중간 배열을 나누며 전체 출력·미분 메모리를 제한하지 않는다.
잘못된 입력은 `propagation.*` 코드의 구조화된 오류로 보고한다.
