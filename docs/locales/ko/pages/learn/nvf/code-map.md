# 12. 코드 읽기 지도와 복습

:::{note}
다운로드 가능한 교육 스크립트는 현재 물리 Problem/Job API를 사용합니다. 일부 설명 코드와 그림에는 이전 API 표기가 남아 있으므로 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 함께 참고하세요. 과거 측정 파일과 원본 출처는 보존하며, 이전한 스크립트의 검증과 구분합니다.
:::

이 교재는 공개 API로 사용하는 것을 먼저 권합니다. 내부 함수를 읽을 때는 아래 순서가 원리와 가장 잘 연결됩니다. 파일과 함수 위치는 기준 커밋에 고정된 안내이며 이후 변경될 수 있습니다.

| 궁금한 내용 | 읽을 위치 | 연결되는 장 |
| --- | --- | --- |
| 기본값·지원 조합·읽히지 않는 설정 거부 | [`engine.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/engine.py), `_resolve_normal_grid`, `_reject_unread_*` | 6장 |
| raster 법선 추정 | [`normal_methods.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/normal_methods.py), [`factorization.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/factorization.py)의 `normal_vector_field` | 4·6장 |
| shape 방향장과 투영 계수 | [`parametric.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/parametric.py), `normal_vector_field_analytical`, `normal_projector_coeffs` | 4장 |
| NVF 대칭화 조립 | [`factorization.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/factorization.py)의 `_nvf_assemble`, [`parametric.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/parametric.py)의 `nvf_from_coeffs` | 5장 |
| 슬롯 이름과 반환 규약 | [`factorization.py`](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/factorization.py)의 `ConvMatrices` | 5장 |
| 입력·단위·편광·결과 축 | [키워드](../../reference/keywords.md), [규약](../../reference/common/conventions.md) | 7·8장 |

## 직접 해볼 연습

1. 예제 1의 `material=4.`를 1로 바꾸세요. 무늬 층과 배경의 재료 차이가 사라집니다. 원의 크기에 따른 결과 변화도 사라지는지 확인하세요. 기판은 그대로 있으므로 전체 반사율까지 0이 되어야 하는 것은 아닙니다.
2. 예제 1의 `orders=(2, 2)`를 `(3, 1)`로 바꾸세요. 출력 배열 shape가 `(3,7)`인지 확인하고, 그 이유를 모드 수 식으로 설명하세요.
3. 3장의 50:50 예에서 두 유전율을 모두 4로 바꾸세요. A와 B⁻¹이 같아지고 Δ가 0이 되는지 계산하세요.
4. 원의 오른쪽 경계와 중심에서 n과 J가 어떻게 다른지 설명하세요. 조작 예제에서는 θ=30°와 −150°를 비교해, 화살표가 반대인데 P가 같은 이유도 확인하세요.
5. JAX 예제에서 shape_nvf_grid_size만 높여 T와 dT/dε를 함께 비교하세요. 다음에는 차수만 높여 비교하세요. 차분 검증의 일치와 물리 기울기의 수렴이 왜 별도인지 기록하세요.

## 작은 용어 사전

| 용어 | 이 책에서의 뜻 |
| --- | --- |
| NVF | 경계의 법선 정보를 전체 영역의 분해 기준으로 연장하고 Fourier 곱 규칙에 반영하는 방식 |
| 경계 법선 n | 매끄러운 도형 경계에 수직인 실제 단위벡터 |
| 보조벡터 J | 경계 법선에 맞추고 내부·배경으로 연장한 계산용 벡터 |
| factorization | 여기서는 불연속 함수 곱의 Fourier 표현 규칙; LU 같은 행렬 분해와 다른 맥락 |
| 투영 행렬 P | J로 정한 성분 분해를 수행하는 행렬. P=JJᴴ/(JᴴJ) |
| raster_normal_method | raster NVF의 방향장 생성 방법 |
| shape_nvf_grid_size | shape NVF의 보조장·투영장을 평가하는 한 축당 샘플 수 |
| nvf_projector | NVF 투영의 구성 방식을 고르는 API 옵션. linear / circular |
| convolution matrix | Fourier 계수의 곱을 표현하는 합성곱 행렬 C(f) |
| truncation | 무한 Fourier 모드 중 유한한 부분만 유지하는 것 |
| raster | 각 픽셀에 재료값을 넣은 구조 표현 |
| shape / parametric | 중심·폭·각도 등 연속 매개변수로 정의한 도형 표현 |
| `rcwa.prepare` | 추적 전에 구체적인 값으로 이산적인 기하 선택을 기록하는 준비 단계 |
| 수렴 | 정밀도를 높였을 때 관심 물리량의 변화가 요구 범위 안에 드는지 확인하는 과정 |

## 참고문헌과 이 교재의 검증 범위

[L. Li, “Use of Fourier series in the analysis of discontinuous periodic structures,” JOSA A 13, 1870–1876 (1996)](https://doi.org/10.1364/JOSAA.13.001870). 불연속 주기 함수 곱의 Fourier 인수분해 배경입니다.

[T. Schuster, J. Ruoff, N. Kerwien, S. Rafler, W. Osten, “Normal vector method for convergence improvement using the RCWA for crossed gratings,” JOSA A 24, 2880–2890 (2007)](https://doi.org/10.1364/JOSAA.24.002880). RCWA의 법선 벡터 방법에 관한 선행 연구입니다.

위 논문의 공식 출판사 서지·초록을 확인했습니다. 이 책은 논문 전문의 번역이나 재현물이 아닙니다. 구현식과 API 설명은 로컬 Wavebit 소스에서 확인했고, 그림과 손계산 예는 이 교재용으로 만들었습니다. 외부 구현 코드나 그림은 복사하지 않았습니다.

예제 5개는 이 API 개정 작업본에서 실행됐고, 모든 실행의 원본 수치와 환경은 {download}`measurements.json <../../validation/learning-records/nvf-measurements.json>`에 남겼습니다. 이것은 입문 예제의 실행·일관성 확인입니다. 전체 회귀 검사, 모든 구조의 광학 수렴, 고급 보조 셀의 완성도를 인증하는 문서는 아닙니다. 출처의 파일 해시는 [빌드 기준과 출처](../../development/learning-provenance.md), 문서 검토 범위는 [검토 범위](../../development/learning-provenance.md)에 있습니다.
