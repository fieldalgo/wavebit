# RCWA·FDTD 통합 API 설계와 구현 중간 점검

> **과거 API:** 이 페이지에서 `wavebit.compat`을 쓰는 코드 조각과 설명은 과거 체크아웃이 있어야 동작합니다.
> 현재 설치본에는 이 네임스페이스가 없으며, 퇴역 직전 기준은
> [리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
> 기록된 실험은 각자 인용한 커밋과 환경으로 재현하세요. 그 측정은 다시 실행하지 않았습니다.


상태(2026-10-15 갱신): 구현됨. 공통 공개 생애주기, 재료, 소스 구조, 이중 언어 학습 사이트
마이그레이션이 `4272c7ce`(2026-10-13)로 develop에 반영되었습니다. `wavebit.Job` / `wavebit.run`,
`wavebit.rcwa`, `wavebit.fdtd`입니다. `rcwa.Polygon` 영역은 `97c6d73f`(2026-10-15)로 뒤따랐고, 둘 다
0.0.2와 0.0.3 릴리스에 포함되어 있습니다. [14절](#14-what-of-the-proposal-is-implemented-2026-10-15)은
제안 가운데 오늘 구현된 것과 아직 제안으로 남은 것을 항목별로 표시합니다. 이 중 어느 것도 보편적인
물리적 수렴을 주장하지 않습니다.

설계 작성: 2026-10-07. 구현 중간 점검: 2026-10-13, `3116a50a`를 바탕으로 한 작업 트리를 대상으로 했고
`4272c7ce`로 커밋했습니다. §13의 근거는 기반 커밋만의 내용이 아니라 그 커밋을 기술합니다.

1–12절은 채택된 설계와 원래의 근거를 보존합니다. 그 안의 제안 예제와 미래 시제의 요구 사항은 설계
이력입니다. 지금 구현되고 검사된 것은 현재의
[시작하기 페이지](../getting-started/index.md), [API 레퍼런스](../reference/index.md),
[구현 중간 점검](#13-implementation-checkpoint-and-acceptance-evidence)이 정의합니다. 그 중간 점검의
능력 경계가 원래 제안의 더 넓은 포부보다 우선합니다.

원래 검토는 RCWA `b9957c8bde0f60d3ca47126996cad257fc6e20a6`과
FDTD `f5a86603e4d7b668726a0519714dc2bcae950655`를 비교했습니다. 이후 통합에서는 더 새로운 FDTD
중간 점검 `d1119445bb2839dd03d2af0b769a2797d0be2d13`을 썼고, 그 이력을 병합
`04b449754e6b226bdcd4374b2897732054468ce7`에 보존했습니다. 이 병합의 첫 번째 부모는 develop
`db37dbb8722b8993c2486c6af4f491e3b1b8bc13`이고, 두 번째 부모는 FDTD 중간 점검입니다. 소스 구조와 이후의
공개 API 변경은 이 계보를 squash로 대체하지 않고 그 위에 쌓습니다.

라이브러리 소스는 이제 `src/wavebit/`에 있습니다. 정리된 공개 문서는 `docs/`에 있으며, 영어 페이지가
기준이고 한국어 번역은 선택 사항입니다. 마이그레이션에는 선택한 교육 자료 전체와 그 자산이 포함되며,
연구 기록은 기록된 수치 결과와 출처를 그대로 유지합니다.

## 1. 결정

두 방법 모두에 하나의 공개 생애주기를 씁니다.

```text
Problem + Numerics + execution settings
                  |
                prepare
                  |
               Prepared -- run(Problem) --> Result

run(Problem, ...) is the convenience prepare-and-run path.
```

공통 계약은 생애주기, 재료 응답, 물리 규약, 검증, 비교 가능한 관측값입니다. RCWA와 FDTD는 서로 다른
문제 기술과 수치 알고리즘을 유지합니다. 공통 API가 보편적인 Scene, 하나의 내부 Plan 표현, 똑같은
지원 기능을 뜻하지는 않습니다.

현재 구현을 보존하려는 이유만으로 가변 솔버를 주 인터페이스로 남기지 마세요. 사용자가 상속해야 하는
공통 Solver 기반 클래스를 도입하지 마세요. 백엔드별 솔버 클래스는 구현 세부 사항입니다.

### 책임

| 객체 | 담당하는 것 | 담당하지 않는 것 |
| --- | --- | --- |
| 재료 모델 | 구성 응답, 분산, 재료 데이터의 원래 단위 | 격자, 시간 간격, ADE 점화식, 푸리에 인수분해 |
| Problem | 물리적 형상, 단위, 여기(excitation), 요청한 관측 | 백엔드, 캐시된 연산자, 마지막 결과 |
| Numerics | 방법별 근사와 이산화 선택 | 재료의 정체나 입사 편광 |
| 실행 설정 | 백엔드, 장치, 정밀도 | 물리 문제나 이산화 |
| Prepared | 해석된 수치 설정, 정적 배치, 재사용 계약, 안전하게 키가 매겨진 캐시 | 가변인 마지막 문제, 마지막 결과, 낡은 실시간 재료 값 |
| Result | 출력, 기준 규약, 유효성, 출처, 요청한 재구성 상태 | 가장 최근에 실행한 계산에 대한 의존 |

모든 선언 객체는 `replace(**changes)`를 지원하며 구조적으로 불변입니다. 수치 리프(leaf)는 자신의
백엔드와 자동 미분 그래프를 보존합니다. 선언이 고정되어 있다고 해서 호출자의 텐서가 마법처럼 불변이
되지는 않습니다. 실행 중에 입력 버퍼를 동시에 변경해서는 안 됩니다. 준비 단계는 정적 호스트 데이터의
스냅숏을 뜨고, 실행은 실시간 배열을 변경하거나 분리(detach)하지 않고 빌려 씁니다.

## 2. 공개 생애주기

두 공개 모듈은 모두 다음 시그니처를 제공하며, Problem과 Numerics 타입은 모듈마다 다릅니다.

```python
def check(problem, *, numerics, backend="numpy", device="cpu", precision="double"):
    """Return a validation and resource report; do not advance fields or solve scattering."""

def prepare(problem, *, numerics, backend="numpy", device="cpu", precision="double"):
    """Validate concrete inputs and return a Prepared calculation."""

def run(problem, *, numerics, backend="numpy", device="cpu", precision="double"):
    """Equivalent in meaning to prepare(problem, ...).run(problem)."""

class Prepared:
    def validate(self, problem):
        """Eagerly report whether this concrete problem satisfies the reuse contract."""

    def run(self, problem):
        """Run the supplied problem without changing this object's numerical contract."""
```

`check`는 출력문이 아니라 데이터를 돌려줍니다. 검증 로직은 `prepare`와 공유하며, 이 함수는 항상
검증합니다. 호출자가 세 함수를 모두 부를 필요는 없습니다. 자원 추정치는 임시 배열과 백엔드 런타임
메모리를 제외했는지를 포함해 자신의 범위를 밝힙니다. `prepared.report`는 해석된 격자나 모드 창,
실제 이산화, 추정 비용, 능력의 한계를 기록합니다.

`Prepared.run`은 항상 완전한 실시간 문제를 받습니다. 암묵적인 "지난번 호출의 값을 쓰기"도, Problem과
어긋날 수 있는 두 번째 매개변수 딕셔너리도 없습니다. 순수 Python 함수로 최적화기의 작은 매개변수
트리에서 Problem을 만들 수 있습니다.

`numerics`, 백엔드, 장치, 정밀도는 준비 시점에 고정됩니다. 이를 바꾸려면 다시 `prepare`해야 합니다.
재사용 가능한 Prepared는 결코 조용히 계획을 다시 세우지 않습니다. 편의용 모듈 수준 `run`은 주어진
문제를 새로 준비합니다. 다만 내용 주소 기반(content-addressed) 내부 캐시가 똑같은 작업을 피할 수는
있습니다.

백엔드 이름은 `"numpy"`, `"jax"`, `"torch"`입니다. 정밀도 이름은 `"double"`과 `"single"`이며 기본값은
double입니다. 이것은 어휘일 뿐, 모든 조합이 지원된다는 주장이 아닙니다. 지원하지 않는 정밀도/장치/
솔버 조합은 거부합니다. 조용한 정밀도 강등이나 CPU 대체 실행은 없습니다.
Wavebit을 import해도 선택적 백엔드를 import하거나 전역 정밀도를 바꿔서는 안 됩니다.
JAX 배정밀도 요청은 실제 x64 설정을 확인해야 하며, 이를 지킬 수 없을 때는 조치할 수 있는 오류를 내야
합니다.

아래의 JAX 예제는 애플리케이션이 배열을 만들기 전에 명시적으로 x64를 켰다고 가정합니다. 예를 들어
`jax.config.update("jax_enable_x64", True)`로 켭니다. 라이브러리는 import 시점에 이를 하지 않습니다.

## 3. 공통 재료 모델

공개 계열은 다음과 같습니다.

```python
from wavebit import Material
from wavebit.materials import (
    ConductiveMaterial, DispersiveMaterial, TabulatedMaterial,
    Drude, Lorentz, Debye,
)

glass = Material(epsilon=2.25)       # mu=None means structurally nonmagnetic
magnetic = Material(epsilon=2.25, mu=1.1)
```

`epsilon`과 `mu`는 진공에 대한 무차원 상대값입니다. 초기 생성자 계약은
`ConductiveMaterial(epsilon, conductivity, mu=None)`,
`DispersiveMaterial(epsilon_inf, poles, mu=None)`,
`TabulatedMaterial(wavelengths, epsilon, *, wavelength_unit, mu=None)`입니다.
TabulatedMaterial은 명시적인 `from_refractive_index` 생성자도 제공합니다. 저장된 보간 계약은 굴절률과
유전율 중 무엇을 보간하는지 기록하며, 표를 옮기더라도 그 선택이 조용히 바뀌어서는 안 됩니다.

모든 모델은 다음을 구현합니다.

```python
response = model.at(wavelength, units=units)  # MaterialResponse(epsilon, mu)
```

`MaterialResponse`는 평가된 구성 데이터이며, 재사용 가능한 광대역 재료 모델이 아닙니다. 파장은 진공
파장입니다. 모델은 장면이 고른 길이 단위와 독립적이며, `units`는 주어진 파장만 해석합니다. 솔버가
평가할 때는 그 맥락을 자동으로 제공합니다.

| 모델 | 물리 데이터 | RCWA의 사용 | FDTD의 사용 |
| --- | --- | --- | --- |
| Material | 상수 epsilon과 선택적 mu; 스칼라/대각/텐서 | 직접 평가 | 지원하는 실수·안정 구성 경우만 받아들임 |
| ConductiveMaterial | 기준 epsilon, 전도도, 선택적 mu | 주파수에서 전도도 기여를 평가 | 지원하는 전도성 갱신식을 구성 |
| DispersiveMaterial | 고주파 응답과 이름 붙은 극점 | 연속 주파수 응답을 평가 | 지원하는 보조 갱신식을 구성 |
| TabulatedMaterial | 스펙트럼, 단위, 보간과 출처 메타데이터 | 선언된 범위 안에서 평가 | 지원하는 명시적 인과적 분산 피팅을 요구 |

Drude/Lorentz/Debye 응답 공식은 공통 재료 패키지에 속합니다. 시간 이산화 계수, 메시 안정성, 푸리에
연산자는 각 솔버에 속합니다. 공통 클래스에 `fdtd_options`와 `rcwa_options` 분기가 있어서는 안 됩니다.

생성자는 미분 가능한 값을 Python float나 NumPy로 변환하지 않고 수학적 구조를 검증합니다. 솔버의
수용 단계는 능력을 검증합니다. 완전한 텐서, 능동 매질, 분산 경계면, 재료 그래디언트는 한 경로에서는
받아들여지고 다른 경로에서는 거부될 수 있습니다. `mu=None`과 명시적으로 지정한 `mu=1`은 계획 수립과
미분에서 서로 다른 구조적 의미를 유지합니다.

자동 표 피팅, 손실 제거, 복소 상수를 추측한 시간 영역 재료로 대체하는 것은 허용하지 않습니다. 명시적인
피팅 연산이라면 새 DispersiveMaterial과 함께 오차/대역/수동성(passivity) 보고서를 낼 것입니다.

## 4. 단위, 좌표, 물리적 의미

각 Problem은 `Units(length="m" | "mm" | "um" | "nm")`를 선언합니다.
모든 형상과 진공 파장은 그 길이 단위를 씁니다. 다른 물리량은 장면의 길이 단위와 무관하게 고정된 공개
단위를 씁니다.

| 물리량 | 공개 규약 |
| --- | --- |
| 각도(형상 회전 포함) | 라디안 |
| 시간, 펄스 지연, 샘플링한 시간 축 | 초 |
| 주파수 / 각주파수 | Hz / rad s^-1; 이름으로 구분 |
| Epsilon / mu | 무차원 상대 구성값 |
| 재료 극점 감쇠율 | s^-1 |
| 전도도 | S/m |
| 시간 영역 필드와 보정된 페이저 | E는 V/m, H는 A/m, 소스 정규화를 기록 |
| 전기 쌍극자 모멘트 / 전류 모멘트 | C m / A m; 서로 다른 물리량이자 타입 |

처음에 유지하는 극점 모델에는 `Drude(omega_p=..., gamma=...)`, `Lorentz(omega_0=..., gamma=..., strength=...)`,
`Debye(tau=..., delta_epsilon=...)`처럼 명시적인 이름을 씁니다. omega와 gamma는 SI 비율을, tau는 초를
쓰며, 현재 Lorentz 분자 세기의 단위는 s^-2입니다. 이 의미는 직렬화된 모델과 함께 전달됩니다. 숫자의
크기로부터 추론하는 일은 결코 없습니다.

내부적으로 FDTD는 c=epsilon0=mu0=1을 유지할 수 있습니다. 길이/시간 변환은
`t_internal = t_seconds * c0 / length_unit_meters`이고, 비율에는 `length_unit_meters / c0`를 곱하며,
전기 전도도는 `sigma_internal = sigma_SI * length_unit_meters / (epsilon0 * c0)`가 됩니다.
공개 H는 기존의 Z0*H 저장값에서 변환합니다. 소스 진폭과 모든 물리적 전력 출력은 일관되게 변환해야
합니다. 이것은 겉보기 라벨 변경이 아니라 구현 요구 사항입니다. 전문가에게 정규화된 필드 접근이
필요하다면, 이름을 붙이고 그 배율을 함께 담아야 합니다.

전역 데카르트 좌표계는 +z가 위쪽인 오른손 좌표계입니다. 적층의 층은 아래에서 위로 쌓으며, 유한
적층의 바닥이 z=0입니다. FDTD Domain은 같은 좌표계 안에서 명시적인 중심을 가집니다. 주기 셀은
중심 기준의 공개 가로 위상 원점을 씁니다. 모든 복소 주파수 응답은 exp(-i omega t)를 씁니다.

파장 단위를 바꿔도 재료의 물리적 극점 주파수가 바뀌어서는 안 됩니다. 좌표 규약을 바꿀 때는 층 목록을
뒤집는 것만이 아니라 벡터/텐서와 복소 기준 위상을 변환해야 합니다. 마이그레이션은 아래쪽을 양으로
하던 옛 주기 FDTD 경로와 더 새로운 일반 FDTD 경로를 명시적으로 따로 다룹니다.

## 5. RCWA 문제와 수치 설정

RCWA는 주기적인 층상 Stack과 단색 Excitation을 씁니다. Excitation은 파장과 소스 선언의 튜플을
담당합니다. 방향, 편광, 진폭, 면은 소스라는 하나의 담당자만 가집니다. 새 API는 서로 경쟁하는
Incidence, PlaneWave, 최상위 theta/polarization 인자를 유지하지 않습니다.

```python
from wavebit import Material, Units, rcwa

glass = Material(epsilon=2.25)
problem = rcwa.Problem(
    units=Units(length="um"),
    stack=rcwa.Stack(
        period=(0.6, 0.6),
        layers=(rcwa.Layer(thickness=0.2, background=glass),),
        top=Material(epsilon=1.0),
        bottom=Material(epsilon=1.0),
    ),
    excitation=rcwa.Excitation(
        wavelength=0.55,
        sources=(rcwa.PlaneWave(side="top", polarization="s"),),
    ),
    outputs=rcwa.Outputs(diffraction=True, fields=False, scattering=False),
)
numerics = rcwa.Numerics(orders=(5, 5), algorithm="smm")

result = rcwa.run(problem, numerics=numerics)
R = result.R.sum(axis=(-2, -1))  # one value per wavelength; spectral axis retained
T = result.T.sum(axis=(-2, -1))

prepared = rcwa.prepare(problem, numerics=numerics, backend="jax")
another = problem.replace(
    excitation=problem.excitation.replace(wavelength=0.60)
)
another_result = prepared.run(another)
```

이 예제는 제안된 균질 박막 API이며, 검토한 커밋에서 실행할 수 있는 예제가 아닙니다. (`4272c7ce`부터는
적힌 그대로 실행됩니다. §2에서 말했듯 JAX 준비에는 먼저 x64를 켜야 합니다.) Layer에는 타입이 있는 2D
영역이나 샘플링한 epsilon 필드도 담을 수 있습니다. 래스터는 명시적으로 샘플링한 물리량이며, Material
객체의 배열이 아닙니다. 자성 래스터 데이터는 별도의 mu 필드로 기술합니다.

`rcwa.Numerics`는 차수, 푸리에 계수 전략, 인수분해, 정식화, 층 알고리즘, 보조 이산화 선택을
담당합니다. 기본값은 문제로부터 해석되어 준비 보고서에 공개됩니다. 요청했지만 관련 없는 옵션은
무시하지 않고 거부합니다.

필드와 산란 행렬은 유지하는 데 상당한 메모리 비용이 들기 때문에 선택적으로 켜는 Outputs입니다.
재구성은 솔버의 가장 최근 상태가 아니라 반환된 결과를 읽습니다. 저수준 합성곱 조립과 직접 연산자
풀이는 `rcwa.operators` 아래에, 실험적인 conforming/NVF 구성은 `rcwa.experimental` 아래에 있습니다.
이들은 일반 `run`의 입력 타입을 중복 정의(overload)하지 않습니다.

결맞은 빔, 차수 여기, 내부 쌍극자도 같은 Problem과 생애주기를 씁니다. 그 관측 스키마는 경계 전력과
필드를 지원하며, 여기에 하나의 입사 포트가 없을 때 반사 계수를 만들어 내지 않습니다. 즉시 실행, grad,
jit, vmap 지원은 각각 별개의 능력입니다.

## 6. FDTD 문제와 수치 설정

FDTD Scene은 물리 영역, 재료, 구조물, 물리적 경계 조건을 담당합니다. Problem은 여기에 소스와 요청한
출력을 더합니다. Numerics는 격자, 시간 적분, 지속 시간, 흡수 구현, 형상 준비 범위를 담당합니다.

```python
from wavebit import Material, Units, fdtd

glass = Material(epsilon=2.25)
problem = fdtd.Problem(
    units=Units(length="um"),
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(0.6, 0.6, 1.6), center=(0, 0, 0.1)),
        background=Material(epsilon=1.0),
        boundaries=fdtd.Boundaries(
            x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()
        ),
        structures=(
            fdtd.Structure("film", fdtd.Slab(z=(0.0, 0.2)), glass),
        ),
    ),
    sources=(
        fdtd.PlaneWave(
            "illumination", plane=fdtd.Plane(z=0.6), direction="-z",
            theta=0.0, phi=0.0, polarization="s",
            pulse=fdtd.GaussianPulse(
                wavelength=0.55, frequency_sigma_hz=2e13
            ),
        ),
    ),
    outputs=fdtd.Outputs(
        monitors=(
            fdtd.Record("probe", at=(0.0, 0.0, -0.2), component="Ey"),
        ),
        final_fields=False,
    ),
)
numerics = fdtd.Numerics(
    grid=fdtd.UniformGrid(spacing=0.025),
    duration=fdtd.Duration(periods=40, source="illumination"),
    courant=0.5,
    absorber=fdtd.PML(cells=12),
)

result = fdtd.run(problem, numerics=numerics)
t_seconds = result.outputs["probe"].times_seconds
ey = result.outputs["probe"].values

prepared = fdtd.prepare(problem, numerics=numerics, backend="jax")
result = prepared.run(problem)
```

(`4272c7ce`부터 2026-10-19의 공개 이름 변경 전까지는 이 선언들이 공개 `wavebit.fdtd` 타입이었고 적힌
그대로 생성되었습니다. 현재 표기는
`fdtd.Structure("film", shape=fdtd.SnappedSlab(z=(0.0, 0.2)), material=glass)`와
`fdtd.TimeSeriesMonitor("probe", at=(0.0, 0.0, -0.2), component="Ey")`입니다.
[공개 이름 중간 점검](#public-name-checkpoint-2026-10-19)을 보세요.)

PlaneWave의 소스 진폭 규약은 전기장 진폭이며, 위상과 정규화는 소스 기준 정보에 명시적으로 기록합니다.
`GaussianPulse.frequency_sigma_hz`는 진폭 스펙트럼의 표준편차입니다. 비율 대역폭도, 파장 폭도, 세기
FWHM도 아닙니다. `Duration`은 초, 스텝, 주기 중 정확히 하나를 받습니다. 주기로 지정할 때는 기준 소스의
이름을 명시하므로, 소스가 여럿인 장면에서도 모호해질 수 없습니다.

Domain은 관심 있는 물리 영역을 뜻합니다. 흡수층 셀은 Domain의 Open 면 바깥에 덧붙으며, 준비 보고서는
할당된 전체 영역과 실제 샘플 위치를 드러냅니다. 패딩 안으로 재료가 어떻게 이어지는지는 지정하거나,
모호하지 않고 지원되는 경우여야 합니다. 패딩이 조용히 진공이 되는 일은 결코 없습니다.
PEC/PMC/주기/Bloch 조건은 물리적 선언으로 남습니다. PML의 단계형 분포(grading), 안정화, 셀 수는 수치적
선택입니다.

제안된 Open/PML 분리와 바깥 패딩은 현재 구현과 다르며, 전용 동등성 테스트가 필요합니다. 어댑터와
형상 계산 방식을 바꾸지 않고 API 이름만 바꿔서는 이를 주장할 수 없습니다.

표준 FDTD 문제는 시간 기록, 필드, 플럭스, 회절, 원거리장, 최종 필드를 요청할 수 있습니다. 구현된
관측값만 안정적인 공개 내보내기입니다. 감쇠 기반 종료, GPU 지원, 비균일 격자는 생성자나 키워드가
예약되어 있다는 이유만으로 지원되는 것처럼 보여서는 안 됩니다.

## 7. Prepared reuse and differentiation

준비 단계는 모든 수치 값이 아니라 계산 구조를 고정합니다.
`prepared.contract`는 구조적 의존성, 허용된 실시간 필드, 모양/dtype 제약, 실행 시점 보호 장치, 미분
능력을 보고합니다. 재사용 가능 여부와 미분 가능 여부는 별개입니다. 어떤 값은 지원되는 미분이 없어도
즉시 실행 사이에 바꿀 수 있습니다.

| 변경 | 요구 사항 |
| --- | --- |
| 재료 계수, 파장, 소스 진폭/위상 | 의존하는 계수를 다시 계산; 선언된 경로와 보호 장치가 유효한 동안에만 재사용 |
| 형상 위치/크기, 층 두께, 격자 기저 | 지원되는 위상(topology)과 공간 범위 안에서만 재사용 |
| 래스터 값 | 배열 모양이 같아도 위상 분할이나 NVF 경계면 기록을 무효화할 수 있음 |
| 물체/층/소스 개수, 재료 모델 타입이나 극점 개수 | 정적 구조가 바뀌면 다시 준비 |
| 스칼라에서 텐서로, mu=None에서 지정한 mu로 | 바뀐 구성 구조를 다시 준비 |
| 결맞은 묶음, 회절 창, 관측기 배열 배치 | 준비된 묶음/모양이 바뀌면 다시 준비 |
| 격자 크기, 수치 방법, 유지할 출력, 백엔드/장치/정밀도 | 다시 준비 |

해석된 시간 간격과 스텝 수는 준비 단계의 의존성입니다. 특히
`Duration(periods=40, source="illumination")`은 소스의 중심 파장에 의존합니다. 그 파장을 바꾸어 해석된
지속 시간이나 스텝 수가 바뀌면 다시 준비해야 합니다. 고정 지속 시간이나 고정 스텝 정책이라면 다른
보호 장치를 만족한 뒤에 다른 파장 변경을 허용할 수 있습니다. 실행이 옛 스텝 수를 쓰면서 새 소스의
40주기라고 주장해서는 안 됩니다.

RCWA는 수치적 의존성이 바뀐 경우에 걸쳐 재료 행렬, 모드, 전파 연산자를 캐시해서는 안 됩니다. FDTD는
격자 모양이 같다는 이유만으로 갱신 계수를 재사용해서는 안 됩니다. 안전한 캐시는 객체 정체나 이전
호출이 아니라 실제 의존성을 키로 삼습니다.

움직이는 FDTD 물체의 경우, 예약된 공간 범위는 안정적인 Structure 이름을 키로 하는
`fdtd.Numerics.geometry_ranges`에 속합니다. 이는 계산 지원 범위의 선언이며 Sphere나 Box의 속성이
아닙니다. 이것이 모든 재료 조합의 안정성을 증명하지도, 실행 시점 검사를 대체하지도 않습니다.

일반적인 즉시 실행 `Prepared.run`은 주어진 값을 검증하고, 호환되지 않는 문제에 대해 경로별 오류를
냅니다. `Prepared.validate`를 쓰면 최적화기가 컴파일된 영역 밖에서 그 검증을 명시적으로 수행할 수
있습니다. 이것이 임의의 미래 값을 보증하지는 않습니다.

자동 미분에서는 구체적인 Problem을 만들고, 변환 바깥에서 준비한 다음, 실행마다 새로 만든 미분 가능한
Problem을 넘기세요. 즉시 실행 `jax.grad` 아래에서는 PyTorch 텐서와 마찬가지로 준비, 검사, 검증이
추적자가 지닌 값을 읽습니다. `jax.jit`이나 `jax.vmap` 아래에서는 이를 거부합니다. 예를 들면 다음과
같습니다.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

base = design(jnp.asarray(2.25))  # design(epsilon) returns a Problem
prepared = rcwa.prepare(base, numerics=numerics, backend="jax")

def loss(epsilon):
    result = prepared.run(design(epsilon))
    return -jnp.sum(result.T)

gradient = jax.grad(loss)(jnp.asarray(2.25))
```

이 계약은 제한 없는 형상 미분을 약속하지 않습니다. 바깥 JAX 변환 아래에서는 Python 메타데이터로
감지할 수 있는 구조 변경을 거부합니다. 지원되는 값 의존 보호 장치는 백엔드 연산으로 실행되거나,
`Numerics(topology_check='host')`로 명시적으로 선택했을 때 CPU 콜백으로 실행됩니다. 실패하면 분리된
접선(tangent)이나 보고되지 않은 값 고정(clamp)이 아니라, 명시적으로 실패한 유효성 검사와 NaN 수치
출력을 냅니다. 진단 메타데이터는 계속 읽을 수 있습니다. 명시적 호스트 정책은 전송 비용과 순차적인
vmap 동작을 밝혀야 합니다. 허용된 콜백이 없는 호스트 전용 위상 결정은 트레이스를 만들 때 거부하거나,
변환 바깥에서 구체적인 설계마다 다시 준비해야 합니다.
즉시 실행 `jax.grad` 아래처럼 구체적인 primal 값을 지닌 추적자는 즉시 실행 PyTorch 텐서처럼 그 primal
값으로 검사할 수 있으며, 검사가 실패하면 즉시 실행에서 예외를 냅니다. 실행은 추적된 값과 그 접선을 유지합니다.
추적된 변경을 안전하게 검사할 수 없다면, 그것은 허용된 실시간 필드가 아닙니다.

형상 최적화는 구체적인 반복값마다 준비하고, 결과 구조가 같으면 컴파일된 커널을 재사용할 수 있습니다.
위상 경계를 넘는 경우에는 매끄러운 미분이 약속되지 않습니다. API는 고정된 메시, 성공한 컴파일, 참인
안정성 보호 장치가 물리적 정확도를 증명한다고 결코 보장하지 않습니다.

## 8. 결과와 관측

모든 Result에는 `outputs`, `validity`, `diagnostics`, `provenance`가 있습니다. 메타데이터는 요청한
입력과 실제로 적용된 입력, 단위, 소스 정규화, 수치 설정, 실행 정밀도, 코드 버전을 기록합니다. 값은
자신의 백엔드에 그대로 남습니다. 저장된 결과와 필드 재구성은 이후의 실행 뒤에도 유효합니다.

`result.outputs`는 이름에서 관측으로의 매핑입니다. FDTD의 키는 요청한 모니터 이름입니다. 한쪽 면
RCWA 여기는 이름 붙은 반사/투과 출력을 제공하고, 더 일반적인 여기는 이름 붙은 경계 출력을 씁니다.
솔버별 편의 기능으로 `R`, `T`, `scattering`, 요청한 최종 필드를 노출할 수 있지만, 그 의미는 가변 실행
상태가 아니라 관측이 정의합니다.

공통 관측 타입에는 FieldSamples와 DiffractionOrders가 있습니다. DiffractionOrders 객체는 **하나의**
기준 포트와 **하나의** 입사/Bloch 가로 파동벡터 묶음에서의 출사 차수를 기술합니다.

```text
wavelengths, orders_u, orders_v
base transverse wavevector and reciprocal lattice (or per-order wavevectors)
reference medium response and longitudinal propagation data
amplitude_ratio_s, amplitude_ratio_p, optional amplitude_ratio_xyz (dimensionless)
field_s, field_p, field_xyz in V/m, where the drive is a continuous wave in V/m
calibrated outgoing modal power, and efficiency when an incident normalization exists
reference plane, lateral origin, polarization frame, propagation convention
source normalization and incident reference plane
integration measure, cell area or invariant-length normalization, outward normal
```

스펙트럼/모드 배열은 길이가 1인 파장 축을 포함해 항상 `(wavelength, order_v, order_u)`를 씁니다.
이름으로 쓰는 `order(m, n)`과 명시적인 스펙트럼 선택으로 위치에 대한 가정을 피합니다. RCWA의 R/T는
단일 포트 경우의 반사와 투과에 대한 효율 뷰입니다. FDTD 반사 모니터 하나만으로 측정하지 않은 투과
결과를 지어내서는 안 됩니다.

같은 포트에서 Bloch 묶음이 다르면 물리적 차수 격자도 다릅니다. 이들은 별도의 묶음 기록이나 경계
필드/전력 관측으로 반환되며, 정수 차수 라벨이 같다는 이유만으로 진폭을 하나의 차수 배열에 더하지
않습니다. 요청한 유한 영역에서 적분한 전력은 그 관측에 해당하는 묶음 간 간섭을 모두 유지해야 합니다.

절대 포트 전력은 공간 적분 측도를 명시합니다. 선언한 주기 셀/면당 W이거나, 차원을 줄인 계산에서는
불변 길이당 W/m입니다. 면의 방향과 정규화 면적/길이는 명시적입니다. 효율은 입사 분모와 적분 측도가
고정된 뒤에야 무차원입니다. 절대 보정을 할 수 없으면 절대 전력도 제공하지 않으며, 정규화된 숫자에
조용히 단위를 붙이지 않습니다.

펄스 푸리에 출력은 시간 영역 샘플이나 보정된 단색 페이저와 구별됩니다. 원 변환
`E_tilde(f) = integral E(t) exp(+i*2*pi*f*t) dt`에서 필드 단위는 V*s/m와 A*s/m입니다.
그에 대응하는 스펙트럼 에너지 밀도의 단위는 공간 적분 후 J/Hz(불변 길이당이면 J/(Hz*m))이며, W 단위의
정상 전력이 아닙니다. RawFieldSpectrum과 RawDiffractionSpectrum은 변환 정규화, 창 함수, 샘플링,
단측/양측 스펙트럼 규약을 기록합니다. 파장당 스펙트럼 밀도에는 주파수에서 파장으로의 야코비안이
필요합니다. RawDiffractionSpectrum은 DiffractionOrders의 한 종류가 아니라 그 형제입니다. 그 배열은
펄스 스펙트럼, 즉 V*s/m 단위의 spectrum_s/_p/_xyz(`reference['spectrum_units']`)이며, 진폭비나 필드는
없습니다.

FDTD는 명시적으로 선택한 입사 소스 기준으로 정규화하여 비교 가능한 무차원 산란 진폭/효율을 만들 수
있습니다. 선언된 사용 가능 스펙트럼 대역은 분해되지 않았거나 사실상 0인 소스 진폭을 제외합니다. 이런
전달 관측은 DiffractionOrders를 씁니다. 정규화된 amplitude_ratio_s/p는 무차원이며, 차수는 자신을
정규화한 입사 기준을 밝힙니다. 연속파 구동이면 `reference['incident_field']`(V/m)이고, 그 차수는 출사
필드(field_s/_p, 데카르트 비율이 있으면 field_xyz)와 전력도 제공합니다. 펄스이면
`reference['incident_spectrum']`(V*s/m, 파장별)이고, 그 차수는 비율과 효율만 제공합니다. 한 생산자의
입사 필드를 다른 생산자의 입사 스펙트럼으로 나누는 일은 결코 없습니다. 솔버끼리는 비율, 효율, 전력,
필드로 비교합니다. 원 푸리에 배열에 V/m, A/m, W 라벨을 다시 붙이는 일은 결코 없습니다.
요청하면 원 형태와 정규화된 형태의 관측을 둘 다 제공합니다.

공통 타입은 연속체 전파와 Yee 전파/디임베딩(de-embedding) 메타데이터의 구분을 유지해야 합니다. 필드
이름이 같아도 기준면, 가로 원점, 입사 정규화, 편광이 일치하기 전에는 진폭을 직접 비교할 수 없습니다.
기준 변환은 명시적이며, 사용한 전파 모델을 보고합니다.

흡수는 보편적으로 `1-R-T`가 아닙니다. 흡수하는 입사 매질에는 그에 맞는 입사 전력/간섭 항이 필요합니다.
소스가 여럿이거나 내부 소스가 있는 결과는 유일한 R/T 정규화를 지어내지 않고 경계 전력을 노출합니다.
비결맞음 평균은 전력을 결합하며, 하나의 복소 진폭을 가지지 않습니다.

FieldSamples는 좌표, 성분 라벨, 샘플링/엇갈림(staggering) 메타데이터, 명시적 단위를 노출합니다.
RawFieldSpectrum은 여기에 더해 위의 원 변환 계약을 따릅니다. 요청한 동일 위치(collocated) 필드와 원
Yee 성분은 서로 다른 관측 형태이며, 이름 붙은 원 샘플에서 보간이 암묵적으로 일어나는 일은 결코
없습니다. RCWA 필드 재구성 상태는 요청했을 때만 유지합니다. FDTD 최종 필드도 선택적으로 켭니다.

특이한 풀이를 정칙화하려고 물리 입력을 바꾸는 것은 명시적 정책입니다. 기본값은 요청한 문제를 그대로
대상으로 하며 특이성을 보고합니다. 호출자가 디튜닝(detuning)을 선택하면, 결과는 실제로 쓴
파장/각도를 기록하고 그 그래디언트가 그 실제 문제에 속한다고 밝힙니다. 트레이스가 즉시 실행과 다른
재시도 정책을 조용히 고르는 일은 없습니다.

## 9. 검증, 오류, 저장된 작업

공통 오류 계열 ValidationError, UnsupportedFeatureError, IncompatibleProblemError, NumericalError를
씁니다. 문제 보고에는 안정적인 코드, 입력 경로, 설명, 관련된 기대값/실제값이 들어갑니다. 지원하지
않는 기능을 무시하지 않으며, 제공할 수 없는 출력이 0처럼 보이지 않게 합니다.

`validity`는 수치/수용 검사와 물리적 수렴을 구분합니다. 불안정성, 범위를 벗어난 재료 조회, 만료된
형상 범위, 부족한 시뮬레이션 시간이 stable이라는 하나의 불리언으로 뭉뚱그려져서는 안 됩니다.
경고와 수렴 진단은 각자의 의미를 유지합니다.

Job은 하나의 버전 붙은 봉투(envelope)와 솔버별 페이로드를 씁니다.

```text
Job:
  schema: wavebit.job/v1
  solver: rcwa | fdtd
  problem: tagged Problem data, including Units
  numerics: tagged Numerics data
  execution: backend, device, precision
```

`Job.to_json`, `Job.from_json`, `Job.key`는 복소 값, 타입이 있는 배열, 재료 모델, 스펙트럼, 소스
선언을 보존합니다. 큰 데이터셋은 명시적인 내용 해시 참조를 쓸 수 있습니다. 임의의 Python 호출 가능
객체, 컴파일된 함수, Prepared 객체, 장치 버퍼는 이식 가능한 작업 데이터가 아닙니다. 호출 가능 객체
기반 설계는 Python API의 기능으로 남으며, Job으로 저장하려면 구체적이고 직렬화 가능한 Problem이
필요합니다.

`wavebit.run(job)`은 Job만 분배합니다. `rcwa.run`과 `fdtd.run`은 JSON, 매핑, 작업, 원 연산자 배열을
뒤섞은 것이 아니라 타입이 있는 Problem을 받습니다. 스윕은 명시적으로 독립적인 Job들로 펼쳐지며,
하나의 실행이 조용히 첫 스윕 점을 고르는 일은 없습니다. RCWA 스펙트럼 스윕은 나중에 결과를 모을 수
있고, FDTD 펄스는 요청한 스펙트럼 관측을 가진 하나의 광대역 실행으로 남습니다.

## 10. 패키지 구조

다음은 설치된 패키지의 구조입니다. 저장소 안의 위치는 `src/wavebit/`이며, 설치해도 여전히
`import wavebit`을 노출하고 `import src`는 결코 노출하지 않습니다. 12절은 개발, 문서, 마이그레이션
요구 사항을 설명합니다.

```text
wavebit/
  __init__.py           # Common public types, solver modules, run(Job)
  materials/           # Material models, response evaluation, catalog, fitting
  units.py
  lattice.py
  observations/        # Common fields, diffraction, power/reference contracts
  jobs.py
  errors.py
  _common/             # Array/precision helpers and validation infrastructure
  rcwa/
    __init__.py        # Problem, Numerics, check, prepare, run
    problem.py        # Stack, Layer, Excitation, Outputs, Problem
    sources.py
    numerics.py
    results.py
    operators.py      # Explicit advanced operator API
    experimental/
    _engine/
    _backends/
  fdtd/
    __init__.py        # Problem, Numerics, check, prepare, run
    problem.py        # Scene, Outputs, Problem
    geometry.py
    boundaries.py
    sources.py
    monitors.py
    numerics.py
    results.py
    _prepare/
    _kernel/
    _runtime.py
  compat/
    legacy.py         # Explicit transitional old entry points

tests/
  materials/
  rcwa/
  fdtd/
  cross_solver/
  api/
```

공개 rcwa/fdtd 패키지는 공개 이름을 가지며, 내부 계산은 밑줄로 시작하는 모듈을 씁니다. 별도의 공개
`fd`도, 두 물리 엔진을 모두 담는 보편적인 `_core`도 없으며, 결과 클래스를 얻으려는 이유만으로 FDTD가
RCWA 백엔드를 import하는 일도 없습니다. 공통 물리 타입은 공통 배열 도우미에 의존할 수 있지만 솔버에는
결코 의존하지 않습니다.

공통 재료 패키지는 `materials.load`와 `materials.available` 같은 카탈로그 연산을 유지하면서 현재의
재료 카탈로그 모듈을 대체합니다. 피팅 API는 구현되고 자격을 갖춘 뒤에만 노출합니다. 2D 원과 3D
원기둥이 관련된 객체를 기술할 수 있다는 이유만으로 형상을 공통으로 만들지는 않습니다. 지원되는 구조
변환은 명시적인 어댑터입니다.

## 11. 마이그레이션과 수용 관문

이것은 의도적으로 호환성을 깨는 API 재설계입니다. Python에서는 `wavebit.rcwa`를 옛 호출 가능 객체이자
새 모듈로 동시에 안정적으로 노출할 수 없으며, fdtd도 마찬가지입니다. 호출 가능 모듈 같은 꼼수나 import
순서에 의존하는 호환성을 쓰지 마세요.

1. 통합 중간 점검을 고정하고 예전 엔진과 새 엔진의 능력을 기록합니다.
   정확성을 먼저 병합/적응시키고, 파일 이동을 수치 수정 안에 숨기지 않습니다.
2. 공통 재료/단위/기준 계약을 구현하고 검증합니다. 미분 가능한 리프를 보존하고, 물리적 응답을 수치적
   낮추기(lowering)와 분리합니다.
3. 두 엔진 위에 Problem/Numerics, 공통 생애주기, 결과 소유권을 추가합니다.
   임시 내부 어댑터는 옛 솔버를 쓸 수 있지만, 그 이전 호출 상태를 새 계약을 통해 노출할 수는 없습니다.
4. 새 모듈 API를 노출하고, 자체 예제, 문서, 테스트, 직렬화 스키마, capability manifest를 갱신합니다.
   구현된 기능만 내보냅니다.
5. 동작을 고정한 뒤, 따로 검토할 수 있는 커밋들로 파일을 옮깁니다.
6. 능력의 동등성을 확립한 뒤에만 호환성 계층을 퇴역시킵니다.

옛 주기 FDTD 경로에는 현재 일반 엔진에 없는 기능이 있습니다. 비균일 메시, 더 넓은 편광 지원, 일부
각도/층 깊이 미분이 그렇습니다. 전환 기간에는 이를 명시적인 호환성 진입점 아래에 보존하고, 표준
`fdtd.run` 아래에서 엔진을 조용히 바꾸지 마세요. 이름을 옮겼다고 해서 일반 엔진이 그 기능을 구현했다는
근거가 되지는 않습니다.

필요한 수용 근거:

- 무손실 및 손실이 있는 비대칭 슬랩; 양쪽 면에서의 TE/TM 및 비스듬한 입사; 공통 재료의 주파수 응답과
  지원되는 분산 모델.
- 위상 원점/기준면을 맞추고 공간/시간 수렴을 측정한 상태에서의 RCWA/FDTD 차수 전력 **및 복소 진폭**.
  셀 모서리 대 셀 중심 위상 오류를 잡아내기 위해 홀수 회절 차수를 시험합니다.
- 물리 단위 변경(m/um/nm)이 전도도, 분산, 쌍극자 정규화, 필드 H, 전력 단위를 포함해 물리적 답을
  보존합니다.
- 허용된 변경에 대해 Prepared 재사용이 새로운 준비와 일치하며, 위상, 재료 배치, 출력 배치, 소스 묶음
  변경은 올바르게 무효화됩니다.
- JAX/Torch 미분이 선언된 각 범위 안에서 독립 검사와 일치하며, 추적된 잘못된 입력은 드러나고 낡은 수치
  값을 결코 재사용하지 않습니다.
- 결과는 이후의 실행 뒤에도 살아남고, 쓰지 않는 필드 상태는 유지하지 않으며, 최소 설치에서 선택적
  백엔드는 선택 사항으로 남습니다.
- Job 왕복 변환은 의미와 키를 보존합니다. 옛 스키마의 좌표/재료/각도 변경은 명시적인 마이그레이션이
  필요하며 버전 수정만으로는 처리할 수 없습니다.

이 관문을 통과하면 시험한 범위 안에서 새 계약이 확립됩니다. 모든 재료/형상/백엔드 조합이나 보편적인
물리적 수렴이 확립되는 것은 아닙니다.

## 12. 저장소 전체 구조

패키지와 그 개발 근거는 서로 다른 독자를 위한 것입니다. 테스트, 연구 실행기, 생성된 결과를 import
가능한 라이브러리에 넣지 말고, 각각의 진입점을 명시적으로 유지하세요.

```text
wavebit-repository/
  pyproject.toml
  README.md
  CONTRIBUTING.md
  CITATION.bib
  LICENSE*
  src/
    wavebit/                     # Package layout from section 10
      data/                      # Shipped optical data and their source metadata
  tests/
    api/                         # Public lifecycle, errors, jobs, result ownership
    materials/
    common/                      # Units, references, array/precision helpers
    rcwa/
    fdtd/
    cross_solver/                # Same physical cases, normalized comparisons
    distribution/                # Built wheel/sdist and minimal installation
    references/                  # Independent oracles, generators, frozen data
    fixtures/                    # Small shared test inputs
  docs/
    index.md                     # Default English documentation site entry point
    conf.py
    _toc.yml                     # Site navigation, distinct from filesystem order
    getting-started/             # Installation and first RCWA/FDTD calculations
    learn/                       # Canonical English textbook, informed by research sources
      fundamentals/
      nvf/
    guide/
      common/
      rcwa/
      fdtd/
      how-to/                    # Short task-oriented recipes using public APIs
      troubleshooting/           # Symptoms, diagnosis, and supported remedies
    reference/                   # Public contracts and generated API documentation
      common/
      rcwa/
      fdtd/
    theory/                      # Derivations and numerical-method explanations
    validation/                  # Measured claims, accuracy, and current limits
    design/                      # Architecture proposals and decisions
    development/                 # Source setup, debugging, extension, contributing
    migration/                   # Upgrade paths, including the unified API transition
    release-notes/
    glossary.md                  # Symbols, physical terms, and API vocabulary
    locales/
      ko/                        # Optional Korean translations of the English documents
    figures/                     # Maintained diagrams and plots used by documents
    _static/
  examples/
    rcwa/
    fdtd/
    cross_solver/
  tutorials/                     # Sequential teaching notebooks
    rcwa/
    fdtd/
    cross_solver/
  studies/                       # Reproducible investigations, grouped by question
    rcwa/
    fdtd/
    cross_solver/
    _support/                    # Research-only helpers, never production imports
  benchmarks/                    # Stable performance/memory workloads
    rcwa/
    fdtd/
  tools/                         # Release, documentation, and repository utilities
  assets/                        # Project branding and other non-runtime assets
  artifacts/                     # Local generated outputs; ignored by Git
  .github/workflows/
```

모든 디렉터리를 미리 빈 채로 만들지 마세요. 담당자와 내용이 생겼을 때 분류를 도입하세요. 특히 기존의
일회성 시간 측정 연구는 benchmarks 아래로 옮긴다고 해서 성능 회귀 테스트 모음이 되지 않습니다.

### 소스와 설치 데이터

저장소 소스 구조로 `src/wavebit/`를 채택합니다. 이렇게 하면 import 가능한 라이브러리 코드가 문서,
테스트, 연구 도구와 분리되고, 배포 검사에서 체크아웃을 실수로 import하는 일이 줄어듭니다. 공개 import
이름을 바꾸거나 RCWA/FDTD 수치 아키텍처를 결정하지는 않습니다.
[Python Packaging User Guide](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)가
패키징상의 장단점을 설명합니다. 소스 구조 선택은 이제 나중의 선택적 방향이 아니라 이 제안의
일부입니다.

구현을 수정하는 연구자는 일급 사용자입니다. 일반적인 소스 작업 흐름은 활성 작업 트리마다 환경 하나를
두고 그 작업 트리에서 편집 가능 설치(editable install)를 합니다. 일반적인 Python 소스 수정은 다시
설치하지 않아도 다음 프로세스 실행에 반영되며, 오래 실행 중인 노트북은 다시 불러오거나 재시작해야
합니다. 프로젝트 메타데이터와 컴파일된 확장은
[pip의 편집 가능 설치 문서](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)에
설명된 대로 다시 설치하거나 다시 빌드해야 할 수 있습니다.
독립된 환경들은 통제된 비교를 위해 기록된 같은 의존성 버전을 쓸 수 있습니다.

의존성이 일치하면 공유 환경은 선택 사항입니다. 하나의 편집 가능 설치는 하나의 체크아웃을 가리킨다는
사실을 문서화하세요. 현재 디렉터리를 바꾼다고 다른 체크아웃의 `src`가 선택되지는 않습니다. 디버거는
import 경로를 통해 활성 작업 공간의 `src`를 명시적으로 선택할 수 있으며, 사용자는 `wavebit.__file__`을
확인해 어떤 소스가 로드되었는지 검증할 수 있습니다. 소스 중단점은 어느 방식에서든 동작합니다. 명시적
소스 경로 디버깅은 개발 작업 흐름이며, 설치된 배포본을 시험하는 일을 대체하지 않습니다.

Wavebit은 이미 체크아웃 바깥에서 `python -I`를 써서 격리된 편집 불가 설치의 스모크 검사를 수행하고,
import된 패키지가 체크아웃 아래에 있지 않음을 단언합니다. 이는 유용한 보호 장치이지만, 설치된 배포본
전체가 전체 수치 테스트 모음을 통과한다는 증명은 아닙니다. 이 검사를 유지하고, 배포 검증을 빌드한
wheel과 sdist에서 만든 wheel로 확장하세요. 패키지 데이터, 공개 API, 선택적 백엔드 격리도 포함합니다.

소스 이동은 FDTD 중간 점검과 통합 동작이 확립된 뒤, 따로 검토할 수 있는 변경으로 구현하세요. 패키지
탐색, 패키지 데이터, 연구/테스트 실행기, 문서 import, 커버리지, 경로 기반 저장소 검사를 함께
갱신하세요. `docs/conf.py`의 루트 경로 주입을 문서화된 설치 작업 흐름으로 바꾸고, autodoc이 이 작업
트리의 의도한 패키지를 로드하는지 검증하세요. 마이그레이션이 반영되면 `CONTRIBUTING.md`의 현재
평면 구조 정책을 갱신하세요. `PYTHONPATH=.`도, 저장소 루트에서 그냥 실행하는 것도 새 패키지 설치
절차가 아닙니다.

현재 런타임 광학 데이터 자산은 패키지와 함께 옮기고 패키지 리소스를 통해 로드하세요. 그 단위, 출처
정체, 이력, 라이선스 고지도 함께 옮깁니다. 테스트 골든 값과 연구 출력은 wheel 데이터가 아닙니다.
필수 루트 라이선스 고지는 유지하세요. 폴더 정리 명목으로 제거하지 마세요.

### 테스트와 기준 근거

테스트는 비공개 모듈마다 테스트 파일 하나를 두는 식이 아니라, 검증하는 계약에 따라 정리하세요. API와
공통 테스트는 공유되는 동작을, 솔버 테스트는 방법별 동작을, cross_solver 테스트는 명시한 수렴 및 정규화
조건에서의 물리적 비교를 검사합니다. 독립적인 외부 골든 값은 Wavebit 자신이 생성한 스냅숏과 구별되어야
합니다.

외부 오라클(oracle)마다 생성기, 고정 데이터, 테스트, 환경 메모, 제3자 고지를 tests/references 아래에
함께 두세요. 경로를 바꿨다고 기준 답을 다시 생성해서는 안 됩니다. 외부 솔버 설치는 재생성할 때만
선택적으로 필요하며, CI에서 커밋된 골든 값을 쓰기만 하는 데에는 필요하지 않습니다.

slow와 research 같은 실행 시간 분류는 테스트 메타데이터이지, 유일한 폴더 분류 체계가 아닙니다. 옮길
때에도 명시적 파일 선택과 마지막 실패 재실행을 포함한 수집 의미를 보존하세요. 경로 기반 마커 목록,
픽스처 경로, 문서 계약 테스트, 워크플로 명령도 갱신하세요.

### 연구와 산출물

검토한 FDTD 리비전에는 studies/results 아래에 추적되는 파일이 3,354개 있으며, 로그, JSON, 압축/원
기록, 실행 가능한 스크립트가 포함됩니다. 이것이 모두 버려도 되는 빌드 출력은 아닙니다. 조사하는 질문에
따라 다시 정리하세요.

```text
studies/fdtd/conductive_stack/
  README.md                      # Question, method, conclusion, limitations
  run.py                         # Reproducible entry point
  configurations/                # Explicit experiment inputs
  records/
    <record-id>/
      manifest.json              # Commit, command, inputs, versions, checksums
      summary.json               # Compact measured outcomes
      evidence/                  # Selected supporting outputs, including failures
```

새 탐색 실행은 무시되는 artifacts에 씁니다. 예를 들어
`artifacts/studies/fdtd/conductive_stack/<run-id>/`입니다. 실행을 인용되는 기록으로 승격할 때는 주장을
재현하고 평가하기에 충분한 근거를 보존합니다. 큰 원 출력은 그런 저장소가 있으면 안정적인 내용 해시
산출물 참조를 쓸 수 있습니다. 외부 저장소가 없다는 것이 기존 근거를 지울 이유는 아닙니다. 중단된
결과나 부정적인 결과가 한계를 뒷받침한다면 유지하세요.

실행 가능한 연구 코드는 results/evidence가 아니라 run.py 옆에 둡니다. 공개된 검증 문서는 기계마다 다른
임시 경로가 아니라 안정적인 기록 ID와 상대 manifest에 링크합니다. 같은 관측에는 하나의 정본 저장
기록이 있어야 하며, 필요하면 여러 문서가 그것을 참조합니다.

생성된 디렉터리는 경로로 무시하세요. 실험 설정, 작업 픽스처, 선별한 근거를 숨기는 JSON/CSV/log 일괄
무시 규칙을 쓰지 마세요. 현재 저장소 전역 패턴은 이 마이그레이션의 일부로 검토해야 합니다.

### 예제, 튜토리얼, 문서, 벤치마크

예제는 안정적인 공개 API만 쓰는 짧은 스크립트이고, 튜토리얼은 순서를 따라 가르치며 노트북일 수
있습니다. 둘 다 비공개 솔버 내부나 연구 도우미를 import하지 않습니다. 그 스모크 검사는 의도한 설치
패키지를 대상으로 실행합니다.

가이드 페이지는 사용법을 가르칩니다. 레퍼런스 페이지는 공개 타입/함수를 문서화합니다. 이론 페이지는
유도를 설명합니다. 검증 페이지는 측정된 근거와 한계를 밝힙니다. 설계 페이지는 제안과 아키텍처 결정을
설명합니다. 개발 페이지는 기여자에게 소스를 설치하고, 디버깅하고, 시험하고, 확장하는 법을 가르칩니다.
페이지를 옮기면 문서 목차와 들어오는 링크도 갱신합니다.

### 공개 문서와 연구 작업 공간 가져오기

문서 소스는 저장소 루트의 `docs/`에 `src/`와 나란히 두어, 소스 체크아웃 하나에 구현과 그 설명이 함께
담기게 하세요. 빌드된 HTML 사이트는 생성된 산출물이며 라이브러리 패키지 데이터가 아닙니다. 연구자는
Markdown을 로컬에서 읽고 편집할 수 있고, 같은 내용을 공개 사이트에서 읽을 수 있습니다.

문서 구조는 2026-10-07에 검토한 NumPy, JAX, PyTorch 공식 사이트를 참고합니다. 이들은 정보 구조의
참고 사례입니다. 아래에 제안한 Wavebit 구성은 프로젝트 고유의 설계이며, 세 프로젝트가 똑같은 계층
구조를 공유한다는 주장이 아닙니다.

| 참고 대상 | 관찰한 구성 | Wavebit에 적용 |
| --- | --- | --- |
| [NumPy documentation](https://numpy.org/doc/stable/)과 [user guide](https://numpy.org/doc/stable/user/index.html) | 시작하기, 사용자 가이드, API, 기여자를 위한 진입점이 구별되고, 사용 문서 안에 기초, how-to, 고급 주제가 있습니다. | 학습, 운용 안내, 정확한 계약, 소스 기여를 쉽게 구별할 수 있게 유지합니다. |
| [JAX documentation](https://docs.jax.dev/en/latest/)과 [101 course](https://docs.jax.dev/en/latest/101/index.html) | 점진적인 학습 단계가 있고, API 레퍼런스와 개발자 자료는 따로 제공됩니다. | 첫 시뮬레이션부터 미분과 고급 실행까지 이어지는 순서를 제공하고, 숙련된 사용자를 위한 직접 조회 경로도 둡니다. |
| [PyTorch tutorials](https://docs.pytorch.org/tutorials/)와 [recipes](https://docs.pytorch.org/tutorials/recipes_index.html) | 안내형 튜토리얼과 간결한 작업 중심 레시피가 라이브러리 문서와 나란히 서로 다른 진입점을 가집니다. | 일관된 학습 과정과 짧은 실용 레시피를 유지하고, 각각이 실행 가능한 예제와 정확한 API 계약에 링크하게 합니다. |

기존의 Sphinx, MyST, PyData 테마, autodoc, 외부 목차를 유지합니다. 하나의 Wavebit 사이트로 PyTorch의
다중 사이트 구성이나 JAX의 번호 붙은 단계와 시스템별 주제를 따라 하지 않고도 이 독자들을 모두 지원할 수
있습니다.

#### 웹사이트 탐색과 독자 경로

주 탐색 항목은 여섯 개를 씁니다. 웹사이트 메뉴는 독자의 작업에 따라 구성하며, 저장소의 모든
디렉터리를 최상위 탭으로 재현하지 않습니다. 기본 라벨과 랜딩 페이지 설명은 영어입니다.

| 주 항목 | 독자의 질문 | 내용과 정본 소스 |
| --- | --- | --- |
| 시작하기(Getting started) | Wavebit은 무엇을 풀고, 첫 계산은 어떻게 실행하나요? | 설치, RCWA와 FDTD 중 고르기, 구현된 솔버마다 짧은 첫 예제, 기본 규약; `docs/getting-started/`. |
| 학습(Learn) | 이해를 단계적으로 어떻게 쌓나요? | 기초, 안내형 RCWA/FDTD 과정, NVF와 역설계 강의, 튜토리얼과 예제 갤러리; `docs/learn/`, 정본 노트북과 스크립트로의 링크. |
| 사용자 가이드(User guide) | 이 작업은 어떻게 하고, 이 설정은 어떻게 고르나요? | 재료, 형상, 소스, 관측, 수치 설정, 백엔드, 미분, 준비 재사용, 성능, how-to, 문제 해결; `docs/guide/`. |
| API 레퍼런스(API reference) | 이 공개 이름은 정확히 무엇을 받고 무엇을 돌려주나요? | 공통 타입과 유틸리티, RCWA, FDTD; `docs/reference/`의 생성된 공개 심볼 페이지와 규범적 계약. |
| 이론과 검증(Theory & validation) | 이 방법은 왜 동작하며, 정확도 주장은 무엇이 뒷받침하나요? | 규약과 유도, 수렴, 독립 비교, 지원 범위와 부정적 결과; 서로 구별되는 `docs/theory/`와 `docs/validation/` 묶음. |
| 개발자 가이드(Developer guide) | 구현을 어떻게 살펴보고, 바꾸고, 시험하고, 확장하나요? | 소스 설정, 작업 트리 환경, 디버깅, 아키텍처, 확장 지점, 테스트, 문서와 릴리스 절차; `docs/development/`, 그리고 상태 라벨이 붙은 별도 하위 절인 `docs/design/`. |

랜딩 페이지는 문제 영역을 소개하고, 두 가지 첫 계산에 링크하며, 이 여섯 진입점을 보여 줍니다. 검색,
문서 버전, 영어/한국어 선택기는 일관된 사이트 컨트롤에 둡니다. 용어집, 릴리스 노트, 업그레이드 가이드,
이슈 보고, 인용 정보는 보조 탐색에서 닿을 수 있게 합니다. 릴리스된 문서와 릴리스되지 않은 문서를
구별할 수 있게 유지합니다.

독자에게는 작업에 맞는 짧은 경로를 제공합니다.

- 처음 쓰는 사용자는 설치에서 동작하는 시뮬레이션 하나로 간 다음 학습 과정을 고릅니다. 이론 교재를
  끝까지 읽는 것은 전제 조건이 아닙니다.
- 다시 찾아온 연구자는 파장 스윕, 회절 추출, 수렴 검사 같은 how-to를 열고, 링크를 따라 관련 API로
  갑니다.
- 최적화 사용자는 미분 범위, 백엔드/정밀도 요구 사항, 준비 재사용, 성능 문제 해결을 사용자 가이드에서
  한데 찾습니다.
- 과학적 주장을 평가하는 독자는 규약에서 그 유도로, 검증 페이지로, 버전이 붙은 연구나 독립 기준
  근거로 따라갑니다.
- 소스를 수정하는 사람은 설치에서 바로 편집 가능 설정과 한 줄씩 디버깅으로, 그다음 관련 엔진 개요와
  테스트로 갈 수 있습니다.

가이드 안에서는 공통 개념을 한 번만 제시하고, 계약이 다른 부분에서 RCWA/FDTD 세부로 갈라집니다. 방법별
절이 있다고 해서 다른 솔버가 같은 기능을 지원한다는 뜻은 아닙니다. 예제 갤러리는 모든 연구를 정리하지
않고 나열한 목록이 아니라, 방법, 백엔드, 전제 조건, 측정된 경우의 예상 실행 시간을 보여 주어야 합니다.

튜토리얼 페이지는 전제 조건과 달성 가능한 결과를 밝히고 완전한 예제를 차근차근 따라갑니다. How-to
페이지는 최소한의 배경지식으로 하나의 집중된 작업을 해결합니다. API 페이지는 시그니처, 단위, 모양,
기본값, 오류, 지원되는 백엔드/미분을 문서화합니다. 이론 페이지는 가정과 수학적 규약을 드러내고, 검증
페이지는 측정 조건과 한계를 제시합니다. 각 페이지는 다른 유형의 페이지 설명 전체를 중복하지 않고 관련
페이지에 연결합니다. 예를 들어 재료 튜토리얼은 사용 가이드와 `Material` 레퍼런스에 링크하고, 그다음
구성 모델과 그 독립 검증 근거에 링크합니다.

문제 해결은 관찰할 수 있는 증상에서 시작합니다. 예상치 못한 체크아웃이 import됨, 위상이나 방향이 틀림,
수렴하지 않은 결과, 유한하지 않은 값, 지원하지 않는 미분, 지나친 실행 시간 같은 것입니다. 해당하는
규약이나 측정된 한계에 링크하고, 사용자 설정 오류와 지원하지 않는 기능을 구별하세요. 검증된 해결책과
성능 주장만 문서화하세요.

학습 랜딩 페이지는 명시적인 튜토리얼과 예제 항목을 제공합니다.
`tutorials/`의 기존 실행 가능한 노트북과 `examples/`의 스크립트가 정본 소스로 남습니다. 처음에는 이들에
링크하고, 나중에 삽입한다면 의도적인 렌더링/include 단계를 쓰세요. 저장소용과 웹사이트용으로 튜토리얼
본문을 따로 복사해 관리하지 마세요. 비싼 연구 실행은 일반 HTML 빌드 밖에 두고, 선별한 실행 가능 예제는
따로 검증하세요.

연구 작업 공간의 문서도 공개 라이브러리 문서가 될 수 있습니다. 현재 위치만으로 임시 문서가 되거나
사이트에서 제외되지는 않습니다. 그 문서가 답하는 질문에 따라 가져오세요.

| 연구 문서 내용 | 공개 위치 |
| --- | --- |
| 순서대로 읽는 기초 및 NVF 교재 장 | `docs/learn/` |
| 설치, 소스 편집, 디버깅, 솔버 확장 | `docs/development/` |
| 구현된 기능 사용이나 설정 선택 | `docs/guide/` |
| Maxwell 방정식, RCWA/FDTD 유도, 재료와 위상 규약 | `docs/theory/` |
| 수렴, 독립 비교, 측정된 정확도와 한계 | `docs/validation/` |
| 유지되는 API/아키텍처 근거와 제안된 대안 | `docs/design/` |
| 재현 가능한 조사 코드와 선별한 바탕 기록 | `studies/` |

살펴본 현행 교재는 이미 Sphinx 소스 모음입니다. 20개 장과 세 개 부록으로 된 한국어 기초 과정, 15개 장의
NVF 과정, 15개 Python 예제로 이루어져 있고, 체크아웃의 영어 매뉴얼 및 API와 결합되어 있습니다. 유지되는
그 장 소스, 그림, 대화형 자산, 예제 파일을 마이그레이션의 입력으로 쓰세요. 예전의 독립 HTML 교재는 보관용
버전이며, 생성된 HTML과 생성된 영어 매뉴얼 사본은 추가 편집 소스가 아닙니다. 통합 과정에서 기존의 예제
검증, 출처, 사이트 링크 검사를 잃지 말고 재사용하거나 적응시키세요.

교재, 이론, 검증, 설계, 개발 가이드, API 레퍼런스를 포함한 모든 공개 문서 절은 영어가 정본 언어입니다.
공개된 튜토리얼 노트북과 예제 설명도 같은 기본값을 따릅니다.
기본 사이트는 영어로 열립니다. 독자는 언어 선택기로 한국어를 명시적으로 고를 수 있습니다. 한국어는
선택적 번역이며, 별도의 기능 집합이나 영어 문서 공개의 전제 조건이 아닙니다.

기존의 한국어 연구 자료와 교재 소스를 써서 검토한 영어 장을 준비한 다음, 선택적인 한국어 번역을 그 영어
페이지에 맞추세요. 양쪽 모두에서 원래의 과학적 주장, 단서, 출처 표시를 보존하세요. 기존 한국어 텍스트는
마이그레이션 입력이지, 그에 대응하는 영어 페이지가 이미 있다는 증거가 아닙니다. 정본 영어 Markdown은 위의
절 디렉터리에 두고, 한국어 번역 자원은 `docs/locales/ko/` 아래에 두세요. 안정적인 페이지 정체를 써서 언어를
바꿔도 같은 주제와 같은 문서 버전에 머물게 하세요. 한국어 번역이 없거나 낡았다면, 옛 번역이 현재 API를
설명하는 것처럼 보이게 하지 말고 영어로 대체 표시됨을 분명히 나타내세요.

API 식별자, 실행 가능한 예제, 수치 규약은 언어 간에 일관되게 유지하세요. 예제 파일과 언어 중립적인 그림은
공유하고, 산문, 캡션, 탐색, 설명용 라벨은 필요에 따라 번역하세요. API docstring은 영어로 남으며, 번역된
설명이 두 번째 API 계약을 만들지 않습니다. 각 번역이 어떤 소스 리비전을 기준으로 검토되었는지 문서화하고,
빌드할 때 제공되는 번역에 깨진 링크와 빠진 자산이 없는지 검사하세요. 기여 규칙을 갱신해 선택적 번역 자원을
허용하되, 정본 문서와 코드 주석의 기본값은 영어로 유지하세요.

기존 학습 과정은 RCWA와 현재 API를 설명합니다. 그 범위를 표시하고, 통합 API가 반영되면 예제와 규약을
갱신하세요. FDTD 강의는 구현되고 검증된 동작을 기준으로 써야 합니다.

가져오는 문서마다 저장소 상대 위치, 의존하는 그림/데이터, 관련 코드나 연구, 그리고 현재 동작을
설명하는지, 구현되지 않은 제안인지, 과거의 실험인지를 기록하세요. 출처 표시와 과학적 단서를 보존하고,
기계별 경로를 제거하며, 비공개 링크는 포함된 자산이나 오래 유지되는 공개 참조로 바꾸세요. 연구 작업
공간의 빠진 파일에 의존하는 페이지는 완성된 것이 아닙니다. 섞여 있는 노트북이나 일지는 원래 기록을 그대로
두면서 편집된 공개 설명을 만들어 낼 수 있습니다.

선별한 문서와 필요한 자산은 작고 검토 가능한 묶음으로 옮기세요. 옮겨 간 곳의 내용, 참조, 탐색을 확인하기
전에는 원본 기록을 제거하지 마세요. 선별하지 않은 과정 메모와 임시 로그는 연구 작업 공간에 남으며, 그것이
있다고 공개 문서를 가져오지 못하는 것은 아닙니다. 사이트에 나타났다는 이유만으로 구현된 API 주장으로
승격되는 문서는 없습니다.

그 마이그레이션의 일부로 문서 기여자 가이드, `_toc.yml`, 교차 링크, 새 하위 폴더에 대한 재귀적 문서 계약
검사를 갱신하세요. 코드 대상과 측정 메타데이터 요구 사항은 그런 주장을 하는 페이지에 적용하세요. 일반
교육용 장이 연구 실행 보고서를 흉내 낼 필요는 없습니다. 가능하면 리디렉션이나 명시적인 이동 안내
페이지로 공개된 URL을 보존하고, 옛 비공개 구현 클래스가 아니라 공개 이름에서 생성한 하나의 공개 API
레퍼런스를 유지하세요.
경고를 오류로 다루는 Sphinx 빌드와 공개 심볼 검증을 유지하세요. 문서는 일치하는 코드 리비전에 대해
빌드하고, 릴리스되지 않은 자료에는 분명히 라벨을 붙이세요. 공개된 릴리스 문서는 그에 맞는 릴리스를
설명해야 합니다. 소스 재구성이나 로컬 HTML 빌드만으로 웹사이트가 공개되지는 않습니다.

벤치마크는 명시적인 문제 크기, 하드웨어/백엔드 메타데이터, 비교 정책을 갖춘 안정적인 성능 및 메모리
작업 부하를 측정합니다. 탐색적인 수치 연구는 studies에 남습니다. 성능 회귀 테스트와 물리 정확도
테스트는 작은 불변 픽스처를 공유하더라도 서로 구별되게 유지합니다.

의존성 규칙은 한 방향입니다. 개발 도구, 연구, 예제, 테스트는 설치된 패키지를 import할 수 있지만,
제품 코드는 그 폴더들을 import해서는 안 됩니다. 제품에 필요한 재사용 가능한 물리 코드는 studies
import로 끌어 쓰지 말고, 공개/비공개 계약과 검증을 갖춘 알맞은 패키지 모듈로 승격시킵니다.

(implementation-checkpoint)=
## 13. Implementation checkpoint and acceptance evidence

### 구현된 계약

`wavebit.rcwa`와 `wavebit.fdtd`는 `check`, `prepare`, `run`을 가진 공개 모듈입니다. 각각은 명시적인
백엔드, 장치, 정밀도 설정과 함께 자신의 불변 `Problem`과 `Numerics`를 받습니다. 준비된 객체는
`run(problem)`을 부를 때마다 완전한 실시간 문제를 요구하고 재사용 의존성을 검사합니다. 이전 결과는 자신의
상태를 그대로 유지합니다. 공개 작업은 가변인 마지막 솔버 상태가 아니라 지원되는 물리 및 수치 선언을
직렬화합니다.

두 방법 모두 공통 재료 응답과 명시적 `Units`를 씁니다. 형상과 파장은 선언한 길이 단위를 쓰고, 시간은 초,
주파수는 Hz, 각속도는 rad/s, 전기 전도도는 S/m입니다. 공개 전기장과 자기장은 V/m와 A/m입니다. 주기
장면은 아래에서 위로 쌓는 층, 셀 중심의 가로 원점, 위쪽이 +z인 좌표, 라디안 단위 각도를 씁니다. 관측
메타데이터는 기준면과 소스 정규화를 담습니다.

[RCWA 가이드](../guide/rcwa/index.md)는 층상 적층, 재료와 형상 선언, 단일 입사 회절, 요청한 필드, 지원되는
결맞은 소스/내부 소스 관측을 다룹니다. 결맞은 소스 실행은 경계 전력과 요청한 필드를 노출하며, 유일한
단일 포트 R/T나 묶음과 무관한 산란 행렬을 지어내지 않습니다. 결맞은 조립은 현재 즉시 실행 전용입니다.
PyTorch와 즉시 실행 `jax.grad`는 되지만 `jax.jit`이나 `jax.vmap`은 안 됩니다.
산란 행렬 출력에는 SMM이 필요합니다. 호스트 전용 형상 결정과 적응형 디튜닝은 값을 지니지 않는 바깥 자동
미분 변환 안에서 안전하다고 가정할 수 없습니다.

[FDTD 가이드](../guide/fdtd/general-fdtd.md)는 물리 영역, 지원되는 구조물, 평면파와 전류 쌍극자 소스, 요청한
기록, 스펙트럼 필드, 플럭스, 회절, 원거리장 관측을 다룹니다. 현재 실행기는 CPU에서 배정밀도로 NumPy, JAX,
Torch를 지원합니다. 격자, 지속 시간, CFL, PML은 수치 입력이며, PML은 물리 영역 바깥으로 확장됩니다.
FDTD 쌍극자 진폭은 전류 모멘트(전기 A·m 또는 자기 등가 V·m)인 반면, RCWA의 전기 쌍극자 모멘트는
C·m입니다.

재료를 만들 수 있다고 해서 두 솔버가 모든 텐서, 분산, 손실, 형상, 경계 조합을 지원한다는 뜻은
아닙니다. 주파수 영역 표는 인과적 시간 영역 모델로 자동 피팅되지 않습니다. `check`는 지원되는 조합을
보고하고, 실행은 지원하지 않는 조합을 거부합니다. 구체적인 준비와 검증은 추적된 실행 시점 유효성 검사와
별개로 남습니다. 어떤 결과의 원 펄스 스펙트럼도 조용히 단색 와트나 완전히 정규화된 입사 전력으로
취급되지 않습니다.

예전 팩토리, 특화된 솔버 작업 흐름, 교육용 예제는 `wavebit.compat.legacy`를 통해 명시적으로 계속 쓸 수
있으며, 새 모듈이 그 팩토리인 척하지 않습니다. [호환성 통합 중간 점검](../development/solver-integration.md)은
그에 앞선 주기 어댑터 작업을 기록하고, [마이그레이션 가이드](../migration/index.md)는 두 인터페이스를
구분합니다.

### 저장소와 교육 자료 마이그레이션

저장소는 설치되는 소스, 솔버 테스트, 유지되는 예제, 튜토리얼, 연구, 개발 도구, 문서를 분리합니다. 과거
수치 기록과 보존된 소스 스냅숏은 이동 및 출처 장부를 통해 보존됩니다. 그 그림과 로그는 새로운 솔버
근거로 다시 생성하지 않았습니다.

선별한 교육 자료는 완전한 영어 장 40개, 그림 100개, 유지되는 Python 예제 15개를 담고 있습니다. 완전한
한국어 장 본문과 현지화된 자산은 같은 경로의 선택적 오버레이이며, 다른 페이지는 명시적으로 영어로
대체됩니다. 언어 목록은 유지되는 정본 Markdown 전체와 공개된 문서 이미지 114개를 다룹니다. 원 실험 메모,
예전 초안과 인계 기록, 원본 삽화, 탐색용 노트북, 독립 프로토타입은 비공개 lab 저장소에 남습니다. 공개
문서에는 검토된 설명과 선별한 수치 근거가 들어 있으며, 연구 원본을 보존한다고 그것을 공개하는 것은
아닙니다. 소스 해시, 번역과 그림 기록은 [학습 자료 출처](../development/learning-provenance.md)를, 빌드와
기여 규칙은 [문서 개발](../development/documentation.md)을 보세요.

현재 API를 쓰는 시작 노트북 두 개가 RCWA와 FDTD의 준비, 실시간 입력, 단위, 관측을 다룹니다. 유지되는 고급
노트북 열 개는 자신의 호환성 import를 밝히고 원래의 수치 코드와 저장된 출력을 유지합니다.

### 이 중간 점검에서 기록한 검증

아래 개수는 서로 다른 집중 실행을 기술하며 겹칠 수 있습니다. 하나의 테스트 총계로 더해서는 안 됩니다.
기존의 지원하지 않는 기능에 대한 건너뜀은 건너뜀으로 남습니다.

| 검증 | 기록된 결과와 범위 |
| --- | --- |
| API 계약 테스트 모음 | 전체 API 실행과 집중 후속 실행에 걸쳐 서로 다른 1,010개 경우가 통과했습니다. 최종 API와 선언 계약을 다루며, 모든 물리 영역을 다루지는 않습니다. |
| 공통 모듈 | 96개 통과; 기존의 지원하지 않는 기능 건너뜀 3개. |
| 공개 FDTD, 새 교차 솔버 및 저장소 계약 | 아래에서 설명하는 네 가지 공개 API 세분화 경우를 포함해 38개 통과. |
| 문서 계약 | 149개 통과; 원래의 연구 코드 대상 검사를 유지하고, 정확한 소스 보존 검사와 보관 경계 검사를 추가했습니다. |
| 유지되는 예제 스크립트 | 37개 모두 통과: 전체 예제 실행에서 36개, 수정한 이방성 재료 예제는 집중 재실행에서 통과. |
| 현재 API 노트북 | 새 노트북 두 개 모두 통과했으며, 코드 셀 6개를 서로 다른 새 NumPy 프로세스에서 순서대로 실행했습니다. |
| 배포 | 최종 wheel과 소스 배포본이 엄격한 Twine 검사를 통과했습니다. 새 가상 환경에서 기본 설치와 plotting 설치 검사를 통과했으며, 유지되는 호환성 교차 솔버 경우 36개를 포함합니다. 그 최대 복소 진폭 오차는 0.0060788554로, 그 경우들의 기존 한계 안에 있었습니다. |
| 영어와 한국어 문서 | 언어마다 소스 페이지 251개에 대해 엄격한 이중 언어 Sphinx 빌드가 통과했습니다. 공개된 HTML 페이지 506개에 빠진 로컬 링크, 프래그먼트, 자산 대상이 없었고, 모든 언어 전환이 같은 경로를 유지했습니다. |
| 전체 기본 수집 | 선택된 16,987개 경우 전부를 설명했습니다: **16,697개 통과, 289개 건너뜀, 1개 예상된 실패**. 분할 실행과 집중 재실행에 걸친 정확한 테스트 노드 대조에서 빠지거나 예상치 못한 노드도, 해결되지 않은 실패도 없었습니다. research 등급에는 선택 해제된 경우 2,721개가 있습니다. |

기본 결과는 분할 실행과 수정한 경우의 재실행에서 개별적으로 식별한 테스트 노드의 합집합입니다. 중단 없이
한 번에 통과한 실행이 아닙니다. 선택된 각 노드는 하나의 최종 결과를 냅니다. 기존의 엄격한 예상된 실패는
`tests/rcwa/test_contact_jones_probe.py::test_equal_material_split_is_equivalent_to_one_rectangle`입니다.
유지된 인공적인 이음매가 유한 차수 확장을 바꾸기 때문입니다. 이는 통과한 동등성 주장이 아니라 명시적인
한계로 남습니다. 건너뜀 289개와 research 등급 선택 해제 2,721개는 통과한 실행으로 세지 않습니다.

이 중간 점검은 완료된 릴리스, 실행을 마친 전체 research/slow 등급, 모델과 백엔드에 걸친 보편적 수렴을
주장하지 않습니다.

### 네 가지 공개 API 회절 비교

[기계가 읽을 수 있는 근거](../development/public-cross-solver-evidence.json)는 새 공개 NumPy API를 써서 균질
박막과 중심에서 벗어난 1차원 격자를 각각 s와 p 조명 아래에서 기록합니다. 테스트 소스 해시와 측정한 모든
세분화 값을 보존합니다. 반사 및 투과 차수 `(-1, 0, 1)`은 공유된 물리 기준면에서 복소 진폭과 전력 효율로
비교합니다. 격자의 평행 이동은 독립적인 홀수 차수 위상 검사를 제공합니다.

RCWA 절단은 `(7, 0)`에서 `(11, 0)`으로 바뀝니다. FDTD 간격은 70 fs에서 0.05 마이크로미터에서 0.025
마이크로미터로 바뀌고, 이어서 세밀한 격자에서 100 fs와 140 fs로 지속 시간을 따로 검사합니다. 이 네
경우에 걸쳐 간격을 절반으로 줄이면 최대 진폭 오차와 효율 오차가 성긴 격자 값의 0.234–0.258배로 줄어듭니다.
최종 설정에서 가장 큰 진폭 오차는 약 0.009234이고, 가장 큰 효율 오차는 약 0.003509입니다.

이 측정은 이 네 문제에 대해 시험한 기준 규약과 국소적인 공간 및 시간 세분화를 뒷받침합니다. 광대역,
비스듬한 입사, 손실, 텐서, 3차원, 모든 백엔드에 대한 수렴 보장은 아닙니다. 원 FDTD 펄스 에너지를 단색
RCWA 전력과 비교하지 않으며, 비교에는 각 방법이 선언하거나 측정한 입사 정규화를 씁니다.

## 14. What of the proposal is implemented (2026-10-15)

0.0.3 시점 develop의 `src/`를 대조해 확인했습니다. "구현됨"은 공개 이름이 존재하고 §13이 밝힌 범위 안에서
해당 절이 설명하는 대로 동작한다는 뜻이며, 그 범위를 넓히지 않습니다.

| 제안 | 상태 | 위치 |
| --- | --- | --- |
| 두 솔버에 하나의 생애주기: `check`, `prepare`, `run`, `Prepared.validate`, `Prepared.run(problem)` (§1, §2) | 구현됨 | `src/wavebit/rcwa/_lifecycle.py`, `src/wavebit/fdtd/_api.py` |
| 백엔드, 장치, 정밀도 어휘; x64 없이 JAX 배정밀도를 요청하면 해결책과 함께 거부; `wavebit`을 import해도 선택적 백엔드를 import하지 않고 정밀도도 바꾸지 않음 (§2) | 구현됨 | `src/wavebit/_common/execution.py` |
| `prepared.report`와 `prepared.contract` (§2, §7) | 구현됨 | 두 `Prepared` 타입 |
| `Material`, `ConductiveMaterial`, `DispersiveMaterial`, `TabulatedMaterial`(`from_refractive_index` 포함), `Drude` / `Lorentz` / `Debye`, `MaterialResponse`를 돌려주는 `model.at(...)`; `materials.load`와 `materials.available` (§3, §10) | §3의 생성자 시그니처로 구현됨 | `src/wavebit/materials/` |
| 명시적인 재료 피팅 연산 (§3, §10) | 아직 제안: 피팅 API를 노출하지 않음 | -- |
| `Units(length=...)`와 §4의 고정 공개 단위 | 구현됨 | `src/wavebit/units.py` |
| `rcwa.Problem`, `Stack`, `Layer`, `Region`, `Excitation`, `Outputs`, `Numerics`와 소스 타입들 (§5) | 구현됨; §5 예제가 실행됨. Region 형상은 `Rectangle`, `Circle`, `Ellipse`, 그리고 `97c6d73f`부터 칠하기 경로의 `Polygon`이며, 경로가 이를 만들 수 없는 곳(Li, `nvf_projector='circular'`, 텐서 재료)에서는 이름을 들어 거부 | `src/wavebit/rcwa/` |
| `rcwa.operators`와 `rcwa.experimental` (§5) | 명시적으로 import하는 하위 모듈로 구현됨 | `src/wavebit/rcwa/operators.py`, `src/wavebit/rcwa/experimental/` |
| 명시적이고 기록되는 정책으로서의 디튜닝 (§8) | 구현됨: `rcwa.Numerics(detuning=None)`이 기본값이며 요청한 점을 풂 | `src/wavebit/rcwa/numerics.py` |
| `fdtd.Problem`, `Scene`, `Domain`, `Boundaries`, `Structure`, `PlaneWave`, `GaussianPulse`, `Outputs`, `Record`, `Numerics`, `UniformGrid`, `Duration`, `PML`, PML은 물리 영역 바깥 (§6) | 구현됨; CPU와 배정밀도만 (§13) | `src/wavebit/fdtd/` |
| FDTD 감쇠 기반 종료, 비균일 격자, GPU (§6) | 구현되지 않았고 노출하지 않음 | -- |
| FDTD 이산 도파관 모드(`solve_modes`)와 `ModeMonitor` (7단계) | 구현됨 | [검증](../validation/fdtd/fdtd-waveguide-modes.md) |
| FDTD `ModeSource`, 한 방향 모드 발사 (7단계) | 구현됨 | [검증](../validation/fdtd/fdtd-mode-source.md) |
| FDTD `GaussianBeam`, 한 방향으로 발사하는 집속 빔 (7단계) | 구현됨 | [검증](../validation/fdtd/fdtd-gaussian-beam.md) |
| `fdtd.Numerics.geometry_ranges` (§7) | 구현됨 | `src/wavebit/fdtd/_numerics.py` |
| 바깥 변환 아래에서 호스트 전용 위상 결정을 거부 (§7) | 구현됨: 고정 셀 데카르트 Li 직사각형의 크기/중심은 JIT/vmap 아래에서 배열 값의 모서리 순서 보호 장치를 씀. 다른 스칼라 영역 형상은 검사되는 CPU 콜백을 위해 `topology_check='host'`를 명시적으로 선택할 수 있음. 기본 장치 정책은 `jax.jit`이나 `jax.vmap` 아래에서 이를 거부함(`prepared.host_topology`); 즉시 실행 `jax.grad` tracer의 primal 값은 PyTorch 텐서처럼 검사함 | `src/wavebit/rcwa/_lifecycle.py` |
| 모든 결과의 `outputs`, `validity`, `diagnostics`, `provenance`; `order(m, n)`을 가진 `DiffractionOrders`, `FieldSamples`, `SpectralFieldSamples`, `RawDiffractionSpectrum`, `TimeSeries` (§8) | 구현됨 | `src/wavebit/observations/` |
| 안정적인 코드와 입력 경로를 가진 공통 오류 계열 (§9) | 구현됨 | `src/wavebit/errors.py` |
| `wavebit.job/v1`, `Job.to_json` / `from_json` / `key`, 그리고 `Job`만 받는 `wavebit.run(job)` (§9) | 구현됨; 그 옆에 저장된 실행 기록을 추가함(`wavebit.records`, 2026-10-13) | `src/wavebit/jobs.py`, `src/wavebit/records.py` |
| 패키지 구조 (§10) | 그린 대로 구현됨, 여기에 나열하지 않은 비공개 모듈 추가 | `src/wavebit/` |
| 옛 주기 FDTD 경로를 명시적인 호환성 진입점 뒤에 유지 (§11) | 구현됨: `legacy.rcwa`, `legacy.fd`와 나란히 `wavebit.compat.legacy.fdtd` | `src/wavebit/fdtd/_periodic/api.py` |
| 저장소 전체 구조 (§12) | `src/`, 테스트 분류, 문서 절, `examples/`, `tutorials/`, `studies/`에 대해 구현됨. 만들지 않은 것: `benchmarks/`, 교차 솔버 튜토리얼 폴더, 가이드의 how-to와 문제 해결 폴더, 릴리스 노트, 용어집 페이지 | 저장소 루트 |
| README, 실행 스크립트, `records/<record-id>/manifest.json`으로 된 연구 기록 (§12) | 채택하지 않음: 연구는 선택적으로 커밋된 `records/` 로그를 가진 스크립트이며, 원 연구 기록은 2026-10-14에 저장소를 떠남(`0f2bb088`) | `studies/` |


## 호환성 계층 퇴역 중간 점검

위의 과거 구현 중간 점검들은 각자 날짜가 붙은 리비전을 기술합니다. 현재 패키지는 `wavebit.compat`을
제거합니다. native 샘플 격자와 주기 적층에는 SI 관측, 준비 재사용, Job을 포함한 물리 단위의
`fdtd.native`와 `fdtd.periodic` 생애주기가 있습니다. 일반 사용자 정의 형상은 `fdtd.Shape`를 쓰고, 오프라인
SI 링다운(ringdown) 피팅은 `observations.fit_modes`를 씁니다.
`migration.migrate_legacy_jobs`는 명시적 길이 단위와 함께 v10 입력을 변환합니다. Manifest v22는 공개
생애주기 색인을 비공개의 정규화된 엔진 진단과 분리합니다. 유지되는 예제는 물리 단위 API를 쓰고, 수치
계측 도구는 정확한 비공개 구현의 이름을 씁니다. 과거 데이터는 원래의 바이트와 소스/환경 출처를
유지합니다. 현재 계약과 고정된 과거 트리는 [마이그레이션 가이드](../migration/index.md)를 보세요.

## Public-name checkpoint (2026-10-19)

위의 중간 점검들은 각자 그 날짜의 이름을 씁니다. 2026-10-19에 일반 FDTD 선언은 각 이름이 객체가 무엇인지
말하도록 별칭 없이 한 번에 깔끔하게 이름이 바뀌었습니다([이슈 #57](https://github.com/fieldalgo/wavebit/issues/57)).
다섯 모니터는 `TimeSeriesMonitor`, `FieldMonitor`, `FluxMonitor`, `DiffractionMonitor`, `FarFieldMonitor`이고,
셀을 채우는 형상은 `SnappedSlab`과 `SnappedBox`이며, `Dipole`, `Axis1D`, `Symmetry`, `Absorber`, `Decay`는
`CurrentDipole`, `InvariantAxis`, `MirrorSymmetry`, `ConductiveAbsorber`, `DecayStop`입니다. 여러 키워드가
자신의 의미나 단위를 담게 되었고(`discretization`, `stopping`, `stabilizing_loss`, `delay_seconds`, `wave`,
`invariant_axes`), 이름 붙은 선언은 이름만 위치 인자로 받습니다. 각 객체에는 하나의 공개 경로
`wavebit.fdtd.<Name>`이 있으며, 예전 선언 모듈은 비공개입니다. 옛 표기는 실패하면서 그 대체 이름을
알려 줍니다. [공개 이름 마이그레이션 가이드](../migration/public-names.md)에 모든 변경이 나열되어 있습니다.

비공개 엔진은 자신의 클래스와 키워드 이름을 유지합니다. 정적 계획이 그 이름을 기록하기 때문입니다. 이름
변경은 낮춘 장면, 계획, 출력을 하나도 바꾸지 않으며, `tests/fdtd/test_public_names.py`가 변경 전에
측정한 여덟 장면으로 이를 고정합니다. 저장된 작업은 공개 이름을 기록하므로 작업 스키마는
`wavebit.job/v2`이고 실행 기록은 `wavebit.run/v2`입니다. 버전 1 파일은 더 이상 존재하지 않는 이름으로 읽지
않고 거부합니다. 이 변경으로 capability manifest는 버전 23이 되었습니다.

같은 날의 두 번째 변경은 명시적 격자 모듈 `fdtd.native`와 `fdtd.periodic`의 이름을 같은 규칙으로
바꿨습니다. 각 모듈은 자신의 클래스를 유지합니다. `Record`는 두 모듈 모두에서 `Snapshots`이며, 성분은 다른
모든 출력에서처럼 `'Ex'`부터 `'Hz'`까지로 쓰고(소문자는 거부합니다) 결과도 `Result.snapshots` 아래에 같은
방식으로 키가 매겨집니다. native `Initial`은 `InitialField`, native `Tail`은 `DecayCheck`이며, periodic은
`tail_steps`/`tail_tolerance` 키워드 대신 자신의 `DecayCheck`를 갖게 되었습니다. native 스냅숏 영역은 정수
`slices=`이고 periodic 스냅숏 영역은 물리 단위 `bounds=`입니다. `Spectrum(planes=)`는 `z_indices=`,
`RasterDispersion(models=)`는 `poles=`이고, native `polarization` 결과는 `polarization_density`입니다. 두 값은
이름만이 아니라 형태도 바뀝니다. periodic 감쇠 창은 초 단위이며 native와 공유하는 하나의 규칙으로 정수
스텝에 반올림됩니다. periodic 펄스는 `fdtd.GaussianPulse`가 이미 쓰는 스펙트럼 표준편차
`frequency_sigma_hz`로 지정하며, 이는 옛 폭 `W`에 대해 1/(sqrt(2)·pi·W)입니다.
엔진은 자신의 소문자 이름과 키워드를 유지합니다. 공개 입구가 이를 번역하며, 엔진 키워드를 언급하는 엔진의
거부는 그것을 설정한 선언으로 바꿔 표현되어 호출자에게 전달됩니다. 옛 표기와 새 표기는 native 경우와
periodic 경우 각각 하나에서 비트 단위로 같은 결과를 냈습니다. 작업 스키마는 `wavebit.job/v2`로 유지되고,
옛 native 또는 periodic 이름을 가진 v2 파일은 거부되며, capability manifest는 버전 24입니다.
[마이그레이션 가이드](../migration/public-names.md#native-grid-and-periodic-stack-names)에 native와 periodic의
모든 변경이 나열되어 있습니다.

세 번째 변경(2026-10-20)은 같은 규칙을 RCWA, 관측 객체, 공통 모듈에 적용했습니다.
`rcwa.OrderExcitation(orders=)`는 `rcwa.IncidentOrders(weights=)`, `GaussianBeam(spectral_extent=)`는
`angular_spectrum_extent=`이고, `layer_thicknesses`는 최상위로 옮겨졌습니다. RCWA 경고에는 하나의 공개 위치
`wavebit.rcwa.diagnostics`가 있습니다. 관측 객체는 자신의 단위와 내용을 이름에 담습니다.
`SpectralFieldSamples`는 `RawFieldSpectrum`, `times`는 `times_seconds`, `frequencies`는 `frequencies_hz`,
`kz`는 `k_normal`, `cartesian`은 `amplitude_xyz`입니다. `Units`는 한 단위가 무엇인지를 이름에 담고
(`length_unit_meters`, `time_unit_seconds`), 그 커널 변환은 비공개입니다. `WavebitError`는 모든 Wavebit 오류의
뿌리이며, `NumericalError`는 더 이상 `ValidationError`가 아닙니다. 엔진 키워드를 언급하던 RCWA 거부는 이제
공개 설정을 언급합니다. RCWA 경고의 문구는 바뀌지 않았으며 나중에 뒤따릅니다. 엔진은 자신의 이름을
유지하고, FDTD 계획 골격은 바뀌지 않았으며, 골든 값도 바뀌지 않았습니다. 작업 스키마는 `wavebit.job/v2`로
유지됩니다. 옛 RCWA 태그나 필드는 거부되므로, `IncidentOrders`나 `GaussianBeam`을 담은 RCWA 작업의 키는
바뀝니다. capability manifest는 버전 25입니다.
[마이그레이션 가이드](../migration/public-names.md#shared-observation-and-rcwa-names)에 모든 변경이 나열되어
있습니다.

같은 날의 네 번째 변경은 모든 솔버가 보고하는 사실마다 하나의 키를 주었습니다(이슈의 §5). 검사는 모든
FDTD 경로에서 `runtime_stability`와 `numerical_finite`이고, rcwa는 `numerical_finite`를 보고합니다(periodic의
유한성 검사는 이제 R뿐 아니라 T도 다루며, 그 reference 알고리즘은 안정성을 평가하지 않음, 즉 `None`으로
보고합니다). 시간 간격은 `time_step_seconds`이며, 감쇠 근거는 모든 FDTD 경로에서 `convergence['decay']`와
`diagnostics['decay']` 아래에 있습니다. 모든 `provenance`에는 `solver`, `execution`, `numerics`, `units`,
`coordinate_frame`, `time_convention`이 있습니다.
`DiffractionOrders.reference`에는 하나의 핵심 키 집합이 있고(`port`는 면, `wave`는 반사 또는 투과 역할,
`plane`, 좌표로서의 `lateral_origin`, 각자의 `_units`를 가진 `incident_amplitude`와 `cell_measure`), 단위 라벨
키는 `_units`로 끝나며 단위마다 하나의 ASCII 표기를 쓰고, `FieldSamples.units`는 성분과 `coordinates`를
키로 하는 하나의 매핑입니다. 모든 공개 거부에는 `area.item` 코드가 있고, 모든 솔버에 `prepared.structure`가
있으며, native와 periodic의 거부는 더 이상 평범한 `ValueError`로 빠져나가지 않습니다. `orders=`는 어디서나
명시적인 쌍 `(Mu, Mv)`입니다. 내보내는 값은 미국식 철자를 씁니다. 키는 별칭 없이 이름이 바뀌었고, 숫자,
계획, 골든 값은 바뀌지 않았으며, 실행 기록은 `wavebit.run/v2`로 유지되고, capability manifest는 버전
26입니다. [마이그레이션 가이드](../migration/public-names.md#result-and-report-keys)에 모든 키가 나열되어
있습니다.

다섯 번째 변경(2026-11-08, [이슈 #54](https://github.com/fieldalgo/wavebit/issues/54))은 회절 결과에서 하나의
이름이 하나의 물리량을 담게 했습니다. `fdtd.periodic`은 `rcwa`와 `fdtd`가 비율 자체에 쓰던 이름 아래에,
진폭비에 소스 진폭을 곱하고 V/m 라벨을 붙인 값을 공개해 왔습니다. 그리고 `incident_amplitude`는 두
솔버에서는 V/m 단위의 필드였고 세 번째 솔버에서는 V*s/m 단위의 펄스 스펙트럼이었습니다. 이제 진폭은 모든
솔버에서 `amplitude_ratio_s`, `amplitude_ratio_p`, `amplitude_ratio_xyz`이며(세 번째 변경의 `amplitude_xyz`
포함), 무차원이고 소스 진폭에 따라 바뀌지 않습니다. `field_s`, `field_p`, `field_xyz`는 구동이 연속파인
경우 그 비율에 `reference['incident_field']`를 곱한 V/m 단위의 값이며, `fdtd`의 펄스에서는 None입니다. 네
번째 변경의 `incident_amplitude`는 `incident_field`(V/m: `rcwa`, `fdtd.periodic`)와
`incident_spectrum`(V*s/m: `fdtd`)으로 나뉘고, `amplitude_units`는 항상 `'1'`인 `amplitude_ratio_units`가
됩니다. 관측 계약은 옛 키와 다른 모든 단위 라벨을 거부합니다. `fdtd.periodic` 차수에는 `power`와
`reference['incident_power']`가 추가됩니다. `RawDiffractionSpectrum`은 `DiffractionOrders`의 하위 클래스가
아니라 형제이며, 그 배열은 V*s/m 단위의 `spectrum_s`, `spectrum_p`, `spectrum_xyz`이고, 그 `order(m, n)`은
`RawDiffractionOrder`입니다. 옛 진폭 이름은 속성으로든 키워드로든 거부되며 그 대체 이름을 알려 줍니다.
`rcwa`의 숫자는 바뀌지 않았습니다. `fdtd.periodic` 진폭비는 더 이상 소스 진폭을 담지 않으며, 비스듬한 Bloch
입사에서 `fdtd`의 입사 스펙트럼은 그 비율이 이미 그랬듯이 셀 중심을 기준으로 합니다. 다시 생성한 골든 값은
없습니다. 작업은 `wavebit.job/v2`, 실행 기록은 `wavebit.run/v2`로 유지되고, capability manifest는 버전
27입니다.
[마이그레이션 가이드](../migration/public-names.md#amplitude-ratios-fields-and-spectra)에 모든 변경이 나열되어
있습니다.
