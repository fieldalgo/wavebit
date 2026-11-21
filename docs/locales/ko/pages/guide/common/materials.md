# 공통 물리 재료

RCWA와 FDTD는 같은 재료 정의를 사용합니다. 재료는 물리적 구성(constitutive) 응답을 기술하며, 각 솔버는 자신이 그 기하, 텐서 형태와 주파수/시간 영역 표현을 지원하는지 검사합니다.

```python
from wavebit import Material, ConductiveMaterial, DispersiveMaterial, Lorentz, Units

vacuum = Material(epsilon=1.)
dielectric = Material(epsilon=2.25)
conductor = ConductiveMaterial(epsilon=2.25, conductivity=1e4)  # S/m
resonant = DispersiveMaterial(
    epsilon_inf=1., poles=(Lorentz(omega_0=2e15, gamma=1e13, strength=(2e15)**2),))
response = resonant.at(.8, units=Units(length="um"))
```

`Material`은 상수 비유전율과 선택적인 비투자율을 제공합니다. `MaterialResponse`는 요청한 파장에서 평가한 응답입니다. 스칼라, 대각, 전체 3×3 응답은 물리적으로 서로 다릅니다. 재료 생성자가 받아들인다고 해서 모든 솔버가 모든 텐서나 손실 모델을 지원한다는 약속은 아닙니다.

`ConductiveMaterial.conductivity`는 S/m 단위의 전기 전도율입니다. Drude와 Lorentz의 각속도 비율은 rad/s, Lorentz 세기는 s⁻², Debye 이완 시간은 초 단위입니다. 기하 단위를 바꿔도 이 물리적 재료 매개변수는 다시 축척되지 않습니다. `exp(-i omega t)` 규약에서 수동적인 전기 손실은 양의 허수부 유전율을 가집니다.

각 생성자는 잘못된 입력을 어떤 솔버가 보기 전에 `.code`와 `.path`로 해당 필드를 가리키는 `ValidationError`(`ValueError`의 일종)로 거부합니다. 유한한 실수 하나가 아닌 극점 비율, Debye 시간 또는 전도율은 `material.scalar`입니다(`material.poles.omega_p`, `material.conductivity`, ...). 숫자 하나, 대각 값 세 개, 3×3 텐서 중 어느 것도 아닌 `epsilon`, `epsilon_inf` 또는 `mu` -- 유전율에 대한 `None`, 문자열, 길이가 고르지 않은 리스트 -- 는 `material.shape`입니다. `mu=None`은 비자성 기본값입니다. `Drude`, `Lorentz`, `Debye` 모델로 이루어진 비어 있지 않은 시퀀스가 아닌 `poles`는 `material.poles`입니다. 극점 하나는 튜플에 넣어 `poles=(Drude(...),)`처럼 씁니다. 표의 `wavelengths`가 표본 두 개 이상으로 된 실수 행 하나가 아니면 `wavelength.shape`이고, 유한하고 양수이며 순증가하지 않으면 `wavelength.value`이며, 둘 다 `material.wavelengths`에서 보고됩니다. 파장마다 유한한 값 하나가 아닌 표본이나 굴절률 표 옆의 0인 `mu`는 `material.epsilon`, `material.index` 또는 `material.mu`에서 `material.shape`입니다.

한 주파수에서의 복소 유전율 하나만으로는 인과적인 광대역 시간 영역 모델이 정의되지 않습니다. FDTD는 이 구분을 검사하고 지원하지 않는 응답을 거부합니다. 시간 영역 연장을 짐작하지 말고, 지원되는 전도율 모델이나 분산 모델을 명시적으로 제공하세요.

## 표와 카탈로그 데이터

재료 표는 자체적으로 선언한 파장 단위를 가집니다. 표를 받는 문제는 `Units`를 독립적으로 선언하며, 공개 솔버는 표를 평가할 때 일관되게 변환합니다. 기하 값만 바꿔서 나노미터 단위의 표를 마이크로미터로 재해석하지 마세요. 표로 주어진 주파수 영역 데이터를 FDTD에서 쓰려면 먼저 지원되는 시간 영역 구현이 필요합니다. 이 데이터는 자동으로 피팅되지 않습니다.

[생성된 재료 레퍼런스](../../reference/common/index.md)에서 현재 생성자를 확인할 수 있습니다. [이전 재료 표 설명서](../../migration/materials-v1.md)는 명시적으로 유지되는 호환 작업 흐름과 그 과거의 텐서/NVF 시연을 기록합니다.

세기가 구체적이고 정확히 0인 Lorentz 극점은 감쇠 없는 공명에서도 특이한 분모를 만들거나 유한한 비율을 넘치게 하지 않고 0을 기여합니다. JAX나 Torch 미분을 지닌 세기는 0에서도 응답의 미분을 유지합니다. 활성 상태이거나 세기가 추적되는 감쇠 없는 공명은 여전히 특이합니다. 유효하지 않은 추적 재료 매개변수는 여전히 NaN을 만듭니다.

FDTD에서 공명이 계획이 검사하는 파장 -- `DiffractionMonitor` 파장이나 소스의 가장 짧은 대역 파장 -- 에 정확히 놓이는 활성 감쇠 없는 Lorentz 극점은 `fdtd.check`, `fdtd.prepare`, `fdtd.run`이 `fdtd.invalid`로 거부하며, 해당 극점과 파장을 알려 줍니다. 그곳에서는 유전율이 무한하므로 재료에 유한한 굴절률이 없고 어떤 메시로도 분해할 수 없습니다. 조금 어긋난 극점은 일반적인 거친 메시 거부를 받으며, 이때 요구되는 간격은 어긋남이 줄어들수록 0으로 갑니다. 파장을 공명에서 비켜 두거나 극점에 감쇠를 주세요. 시간 진행 자체는 공명 분모로 나누지 않습니다. 공명하는 소스 중심 파장에서 읽은 실행 후 열(column) 진단은 실패로 처리하지 않고 판정하지 않음으로 보고합니다.
