# 05. Wavebit는 NVF를 어떤 식으로 조립하나

:::{note}
다운로드 가능한 교육 스크립트는 현재 물리 Problem/Job API를 사용합니다. 일부 설명 코드와 그림에는 이전 API 표기가 남아 있으므로 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 함께 참고하세요. 과거 측정 파일과 원본 출처는 보존하며, 이전한 스크립트의 검증과 구분합니다.
:::

이제 원리와 실제 코드를 연결하겠습니다. 이 장의 식은 기준 커밋의 [`_nvf_assemble`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/factorization.py)와 [`nvf_from_coeffs`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/parametric.py)를 읽기 쉽게 다시 쓴 것입니다. 모든 행렬은 같은 유지 모드 순서를 사용합니다.

```{figure} ../../figures/learn/nvf/06-assembly.svg
:alt: 유전율 계수와 투영 계수를 결합해 다섯 슬롯을 만든다
:figclass: learning-figure

그림 5-1. 유전율 계수와 투영 계수를 결합해 다섯 슬롯을 만든다
```

```text
A = C(ε),  B = C(1/ε),  Δ = A − B⁻¹
Cij = C(Pij)

Kxx = A − ½(Δ Cxx + Cxx Δ)
Kyy = A − ½(Δ Cyy + Cyy Δ)
Kxy =   − ½(Δ Cxy + Cxy Δ)
Kyx =   − ½(Δ Cyx + Cyx Δ)
```

K는 유한 Fourier 공간에서 사용하는 횡방향 재료 응답의 블록입니다. 그림의 `δij`는 i=j일 때 1이고 다를 때 0입니다. 공간에서의 단순한 직관은 “접선에는 A, 법선에는 B⁻¹”입니다. 하지만 잘린 Fourier 행렬끼리는 곱의 순서가 바뀌면 결과도 달라집니다. Wavebit는 `Δ Cij` 하나만 쓰지 않고 **두 순서의 평균**을 사용합니다.

손실 없는 실수 재료와 적절한 Hermitian 투영 행렬에서는 이 대칭화가 전체 블록 행렬의 Hermitian 구조를 유지하도록 합니다. 손실이 있는 재료의 응답까지 억지로 Hermitian으로 만든다는 뜻은 아닙니다. 복소 손실은 그대로 남아야 합니다.

## 두 가지 손계산으로 코드의 뜻을 확인하기

법선이 셀 전체에서 x 방향이면 `Cxx=I`, `Cyy=Cxy=Cyx=0`입니다. 위 식은 곧바로 `Kxx=B⁻¹`, `Kyy=A`, `Kxy=Kyx=0`이 됩니다. x가 법선이므로 역 규칙, y가 접선이므로 직접 규칙을 적용한 것입니다.

균일 재료에서는 `A=εI`, `B=(1/ε)I`이므로 `Δ=0`입니다. 방향장을 어떻게 설명하든 보정항은 사라집니다. 이것이 NVF가 재료 경계의 불연속을 다루는 방법이라는 사실과 맞아떨어집니다.

## 다섯 슬롯과 마이너스 부호

내부적으로 다음 순서의 `ConvMatrices`를 광학 엔진에 넘깁니다.

| 슬롯 | 이 장의 기호 | 읽는 방법 |
| --- | --- | --- |
| `epx` | Kxx | x-x 재료 응답 |
| `epy` | Kyy | y-y 재료 응답 |
| `epz_i` | A⁻¹ | z 성분에 쓰는 직접 행렬의 역행렬 |
| `epxy` | −Kxy | 엔진의 부호 규약에 맞춘 교차 성분 |
| `epyx` | −Kyx | 반대 방향 교차 성분 |

여기서 `epz_i=A⁻¹`은 법선 성분의 `B⁻¹`과 다릅니다. 경계가 xy 평면 안에서 방향을 바꾸고, 층은 z 방향으로 같은 단면이라는 설정을 떠올리면 됩니다. 교차 슬롯의 마이너스 부호도 실제 재료에 음의 값을 추가한다는 뜻이 아니라 엔진의 변수·방정식 규약입니다.

일반 사용자는 이 행렬들을 직접 만들 필요가 없습니다. `wavebit.rcwa(..., factorization='nvf').solve()`가 구성과 광학 계산을 이어 줍니다. 이미 재료 행렬을 만든 연구용 경로는 `solve_conv()`로 전달하지만, 이 경우 factorization 설정을 바꾼다고 전달한 행렬이 다시 조립되지는 않습니다.
