# 08. 픽셀 입력으로 사용하고 NVF를 선택하기

:::{note}
이 교재의 실행 예제는 원래 수치 기록을 재현할 수 있도록 `wavebit.compat.legacy` 인터페이스를 명시적으로 사용합니다. 새 작업은 [현재 시작 안내](../../getting-started/index.md)와 [API 이행 안내](../../migration/index.md)를 참고하세요. 기존 측정값은 이번 문서 이행에서 다시 계산하지 않았습니다.
:::

이미지나 기존 재료 배열이 있다면 {download}`예제 2 <../../../examples/rcwa/learning/nvf/02_raster_nvf.py>`를 사용합니다. 64×64 픽셀로 원을 만든 뒤 integrated 계수와 NVF를 적용합니다. 반지름 160 nm의 원을 픽셀 중심에서 판정하므로, 예제 1의 연속 원과 기하가 완전히 같지는 않습니다.

```{literalinclude} ../../../examples/rcwa/learning/nvf/02_raster_nvf.py
:language: python
:linenos:
```

```bash
python examples/02_raster_nvf.py
python examples/05_select_li_or_nvf.py
```

예제 2의 실제 결과는 `R_p=0.002827293719`, `T_p=0.997172706281`, 에너지 잔차 `2.22×10⁻¹⁶`입니다. 예제 1과의 차이에는 픽셀 기하와 법선 구성 방식의 차이가 함께 들어 있습니다. 이 두 값만으로 어느 경로가 참값에 더 가깝다고 판정할 수는 없습니다.

## NVF를 켜거나 끄려면

raster에서는 `factorization='nvf'` 또는 `'li'`를 선택합니다. Li를 선택할 때는 `raster_normal_method`, `nvf_projector`, `shape_nvf_grid_size` 같은 NVF 전용 설정을 함께 넣지 않습니다. 읽히지 않는 옵션을 지정하면 현재 구현은 조용히 무시하지 않고 거부합니다.

shape에서는 기본값이 NVF입니다. **직교 셀 안의 축 정렬 사각형**에는 명시적으로 `factorization='li'`를 선택할 수 있습니다. 도형이 셀 안에 들어가야 하며, 회전 도형·곡선·기울어진 셀·셀 경계를 가로지르는 사각형에는 이 shape Li 경로를 적용할 수 없습니다. 사각형이라고 자동으로 NVF가 꺼지는 것은 아닙니다.

{download}`예제 5 <../../../examples/rcwa/learning/nvf/05_select_li_or_nvf.py>`는 지원 범위 안의 같은 사각형을 두 방식으로 계산합니다. 실제 출력은 다음과 같습니다.

| 방식 | p 투과율 T | 에너지 잔차 |
| --- | --- | --- |
| shape Li | 0.995999620094 | 1.11×10⁻¹⁵ |
| shape NVF / linear | 0.995802829398 | 1.33×10⁻¹⁵ |

두 방식 모두 에너지를 잘 보존하지만 유한 차수의 답은 다릅니다. 이 표는 선택 기능을 보여 주는 예이지, 정확도나 속도 승자를 정하는 벤치마크가 아닙니다.
