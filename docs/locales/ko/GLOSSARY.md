# Korean translation glossary

This glossary fixes one Korean rendering per English term for the overlays under `pages/`.
Use it when translating a page or changing one. It does not apply to the forty imported course
chapters under `pages/learn/fundamentals/` and `pages/learn/nvf/`, which keep the author's
original Korean. Code, identifiers and option values are never translated.

Where a term is first used on a page and the Korean is not obvious, add the English once in
parentheses, e.g. 동결 검사(freezer). After that, use the Korean alone.

## Terms whose rendering was unified

| English | 한국어 | Notes |
| --- | --- | --- |
| production (code, path, engine, module) | 제품 | 제품 코드, 제품 경로, 제품 엔진. Not 운영, 프로덕션 or 실사용. |
| door (an entry-point check) | 입구 | 입구 검사. Not 문 or 관문. The front door (the main public entry path): 정문. |
| gate (a pre-registered pass/fail criterion) | 관문 | Kept distinct from door. A time gate (a finite time window): 시간 창. |
| freezer (frozen-array regression check) | 동결 검사 | freezer arrays: 동결 배열. Not 고정기 or 고정 보관본. Plain "frozen" (a frozen plan, a frozen reference) stays 고정된. |
| oracle (independent reference that judges an answer) | 오라클 | Not 정답 or 판정 기준. |
| reference (solver, value) | 기준 | 기준 솔버, 기준값. |
| eager / eagerly (JAX, PyTorch execution) | 즉시 실행 | 즉시 실행에서, 즉시 실행 JAX. "eagerly" in a non-JAX sense (prepare once, eagerly, on the host): 즉시. |
| traced / a trace / under a trace (JAX, PyTorch) | 추적된 / 트레이스 / 트레이스 아래 | tracer: 추적자, also in prose. |
| primal (JAX JVP) | primal | Kept in English in this JAX sense only. A primal (as opposed to dual or complementary) problem, bound or tensor: 주, e.g. 주 에너지 텐서. |
| lowering / lowered (to an engine or kernel) | 낮추기 / 낮춘 | Not 하강, 하향 변환 or 변환. |
| light line | 라이트 라인 | Not 광선 (a ray) or 광원뿔. |
| medial axis | 중심축 | |
| graded mesh, grid, reference | 비균일 | 비균일 격자, 비균일 메시. grading of a mesh: 비균일화. |
| graded absorber, padding, PML | 단계형 | 단계형 흡수 패딩, as in the stopping-and-absorbers guide. PML grading (noun): 단계형 분포. |
| graded material, profile, reference, target, band, raster (a permittivity ramp, a smooth profile) | 점진 | 점진 재료, 점진 프로파일, 점진 기준. Not 비균일 (meshes) or 단계형 (absorbers). |
| enclosure (an interval bound) | 포함 구간 | Not 포위 구간 or 가둠 구간. |
| witness (in a mutation or validity check) | 증거 | Not 증인. |
| mutant, mutation (mutation testing) | 변이 | Not 돌연변이. |
| proxy (an indirect measure) | 대리 지표 | noise proxy: 잡음 대리 지표. A stand-in object (proxy normal, material proxy): 대리 법선, 대리값. |
| certificate / certify | 증명서 / 증명하다 | Not 인증서 or 인증하다. A proof is also 증명; the English decides. |
| fixture (a test structure or scene) | 픽스처 | |
| seam (material seam) | 이음매 | |
| closure (constitutive, energy) | 닫힘 | |
| smoothed, smoothing (subpixel smoothing) | 스무딩 | 스무딩 바디, 스무딩한 표면, 서브픽셀 스무딩. Not 매끈한 or 평활. A smooth (not smoothed) surface stays 매끈한. |
| rung / ladder | 단계 / 사다리 | |
| settling | 안정화 | |
| harness | 하네스 | |
| ledger | 장부 | |
| coercivity / coercive | 강압성 / 강압적 | |
| stationary (energy, functional) | 정상 | stationary point: 정류점. |
| compensated (arithmetic) | 보상 | |
| detuning | 디튜닝 | |
| de-embedding | 디임베딩 | |
| ringdown | 링다운 | |
| dyad | 다이애드 | |
| Earlier interface (admonition title) | 이전 인터페이스 | |

## Terms already used consistently

| English | 한국어 |
| --- | --- |
| diffraction order | 회절 차수 |
| convergence | 수렴 |
| validation | 검증 |
| golden values | 골든 값 |
| lifecycle | 생애주기 |
| scene | 장면 |
| unit cell / supercell | 단위 셀 / 슈퍼셀 |
| grazing incidence | 스치듯 입사 |
| normal field | 법선장 |
| Rayleigh point | 레일리 점 |
| exceptional point | 예외점 |
| absorber | 흡수층 |
| amplitude ratio | 진폭비 |
| refusal | 거부 |
| contact / gap / pair | 접촉 / 틈 / 쌍 |
| staircasing | 계단 근사 |
| painting a shape | 칠하기 |
| stage 4 | 4단계 |
