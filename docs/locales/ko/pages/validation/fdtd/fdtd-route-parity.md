# FDTD 경로 일치: 두 엔진은 같은 오차를 낼까?

> **과거 API:** 이 페이지에서 `wavebit.compat`을 쓰는 코드 조각과 설명은 과거 체크아웃이 필요합니다.
> 이 네임스페이스는 현재 설치본에 없으며, 퇴역 직전 기준은
> [리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)입니다.
> 기록된 실험은 각자 인용한 커밋과 환경으로 재현하세요. 그 측정은 다시 실행하지 않았습니다.


작성: 2026-09-17

대상:
- `src/wavebit/fdtd/_periodic/api.py`
- `src/wavebit/manifest.py`
- `src/wavebit/fdtd/_periodic/legacy.py`
- `src/wavebit/fdtd/_periodic/scene.py`
- `src/wavebit/fdtd/_periodic/grid.py`
- `studies/fdtd/route_parity/fdtd_route_parity.py`
- `tests/fdtd/test_fdtd_route_parity_capture.py`
- `tests/fdtd/test_fdtd_pole_colocation.py`

요약: 장면 40개에서 격자, 모니터 평면, CPML, 펄스 포락선과 모든 재료 배열을 맞추고, 그중 35개에서
`dt`를 맞추고, 측정할 수 있는 유일한 후처리 차이(디임베딩(de-embedding) 위상)를 없앴을 때, 레거시
엔진과 공유 엔진은 분류된 호출 80개 중 50개에서, 그리고 장면 19개에서는 진폭과 전력 모두에서
**같은 오차** 를 냅니다. 그 차이를 없애지 않으면 비교는 장면 27개에서 차이를 과장합니다. **어느
장면에서도 장면 수준으로 순위가 매겨진 경로는 없습니다.** 23개는 기준값이 반올림 오차 수준까지
정확해서 순위를 보류했는데, 이것은 측정된 동률이 아니라 규칙입니다. 나머지 17개 중 16개에서는
분해능이 부족해 적어도 한 지표가 보류되었고, 마지막 하나에서는 진폭이 취약(fragile)하다는 이유로
보류되었습니다. 전력만 보면 장면 3개에서 공유 엔진이 1.2-5.1% 차이로 앞섰지만, 그중 두 차이는
대리 지표 안정화 한계(proxy settling bound)의 1.5x 이내이며, 종료 조건을 강화한 쌍둥이 장면에서는 약
2e-6으로 줄어 분해능 아래로 내려갑니다. 재료 장면에서는 CPML 목표, 깊이, 실행 길이를 함께 강화하자
스치는 차수의 차이 대부분이 사라졌지만, 두 Lorentz 장면에는 분해된 나머지가 남습니다. 전력에서는
진공 장면 네 개 모두 패딩에 따라 오차가 더 작은 경로가 바뀌며, 균일 슬래브 하나는 원인을 돌릴 곳
없이 불일치합니다. 이 기록은 공유 엔진을 그 자체의 기본값으로 측정하지 않았고, 소스 주입을 분리하지
않았으며, 그 뒤로 바뀐 패키지에서 측정했습니다(3절). 이 기록을 근거로 `default_route`를 바꾸거나
어느 엔진을 없앨 수는 없습니다.

## 1. 질문, 그리고 두 엔진의 차이

`wavebit.fdtd`에는 경로가 두 개 있습니다(`src/wavebit/manifest.py`, `fdtd.routes`. `default_route`는
`legacy_numpy`). 2026-10-13부터 이 주기 진입점은 `wavebit.compat.legacy.fdtd`이며,
`wavebit.fdtd`는 공개 일반 FDTD 모듈을 가리킵니다. `backend`를 생략하면
`src/wavebit/fdtd/_periodic/api.py::fdtd`는 **레거시 엔진** 인
`src/wavebit/fdtd/_periodic/legacy.py::solve_fdtd`를 호출합니다. 이 엔진은 NumPy 전용이고 미분할 수
없으며 위쪽 입사만 지원합니다. `backend`를 명시하면 `src/wavebit/fdtd/_periodic/grid.py` 위에서 도는
**공유 엔진** 인 `src/wavebit/fdtd/_periodic/scene.py::solve_scene_fdtd`를 호출합니다. 이 엔진은 NumPy,
JAX, Torch를 지원하고, JAX와 Torch에서 미분할 수 있으며, 양쪽 입사를 모두 지원합니다. 두 엔진은
Maxwell 갱신을 하나도 공유하지 않습니다. 기본값과 방법은 다음과 같이 다릅니다.

| | 레거시 엔진 | 공유 엔진 |
| --- | --- | --- |
| `courant` | 0.99 (`src/wavebit/fdtd/_periodic/legacy.py::COURANT`) | 0.8 |
| `n_periods` | 80 (`src/wavebit/fdtd/_periodic/legacy.py::N_PERIODS`) | 40 |
| 펄스 최고점 | 폭의 4배 (`src/wavebit/fdtd/_periodic/legacy.py::SOURCE_DELAY`) | 폭의 5배 |
| CPML 목표 | 1e-8 (`src/wavebit/fdtd/_periodic/legacy.py::PML_TARGET_REFLECTION`) | 1e-8 (`pml_reflection`) |
| 소스 | 1D 입사 격자가 공급하는 TFSF 평면 | 전류 시트 두 장. 균일 매질 기준 실행으로 입사 진폭을 보정하고 반사 평면에서 뺍니다 |
| 디임베딩 위상 | 연속체 `kz` | 이산 Yee `kz` |
| PML 뒷면 | 끝 행에서 E를 접지 | 닫는 셀 하나를 둔 주기적 z 감싸기 |
| PML-구조물 간격 | 소스, 반사, TFSF 평면이 있는 구조물 위쪽이 아래쪽보다 큼 | 양쪽에 같은 패딩 하나, 셀 하나 이내로 같음 |

서로 독립인 두 이산화는 비트 단위로 일치하지 않으므로, 경로 차이를 단독으로 읽지는 않습니다. 각
장면과 채널에 대해, 연구는 수렴한 기준값 하나에 대해 다음을 보고합니다.

| 기호 | 의미 |
| --- | --- |
| `e1` | 레거시 자체의 오차 `abs(legacy - reference)`. 엔진 쌍 비교에서는 디임베딩 교체(6절)를 적용한 레거시 |
| `e2` | 공유 엔진 자체의 오차 |
| `d` | 경로 차이 `abs(legacy - shared)`. 기준값이 들어가지 않습니다 |
| `u` | 기준값의 불확도. 매 실행에서 RCWA 차수 열로 측정합니다. 기준값이 정확한 경우 `ROUNDOFF` = 1e-14를 하한으로 둡니다 |

기준값은 항상 차수 열에서 가장 높은 절단 차수의 `wavebit.rcwa(backend='numpy')`입니다.
닫힌 형식 해가 있으면 그 답과 대조하며, 둘 사이의 차이만큼 진폭 `u`를 넓힙니다. `d`가 `e1`, `e2`에
비해 작으면 두 엔진은 같은 오차를 내는 것이고, 둘 중 무엇을 고를지는 기능의 문제가 됩니다.

## 2. 판정 읽는 법

각 쌍은 복소 진폭과 전력에 대해 각각 한 번씩, 두 번 분류합니다. 사례는 `d`가 가장 큰 채널 하나에서
그 채널 자체의 `e1`, `e2`와 함께 읽습니다. 순위는 그 지표의 모든 채널에 걸친 두 엔진의 최악 오차를
비교합니다. 판단은 다음 순서로 합니다
(`studies/fdtd/route_parity/fdtd_route_parity.py::classify`).

| 단계 | 규칙 | 결과 |
| --- | --- | --- |
| 1 | `d`가 안정화 한계보다 작음 | `unresolved_against_settling`. 순위 없음 |
| 2 | `u`가 반올림 하한 | 순위 보류. 모든 `e1 != e2`가 10 u를 넘어설 것이기 때문 |
| 3 | 그 밖에, 최악 오차 사이의 차이가 10 u보다 작음 | 순위 보류("10 u 미만") |
| 4 | 그 밖에, 그 차이가 디임베딩 교체로 같은 최악 오차에 생긴 이동보다 작음 | 순위 보류("취약"). 전력에서는 0 |
| 5 | 두 오차 모두 분해됨(10 u 이상), 반올림 하한, 한 오차가 다른 오차의 10배보다 큼 | `one_route_near_exact` |
| 5 | 둘 다 분해됨, `d/min(e)`가 1/3 이하 | `same_error` |
| 5 | 둘 다 분해됨, `d/min(e)`가 1 이상 | `route_changes_answer` |
| 5 | 둘 다 분해됨, 그 사이 | `intermediate` |
| 5 | 한 오차만 분해되거나 둘 다 분해되지 않음 | `one_route_below_reference_resolution`, `both_below_reference_resolution`. 기준값으로 그 쌍을 판단할 수 없습니다 |
| 장면 | 재료가 다름 | `inputs_differ`. 어떤 사례도 엔진에 관한 주장이 아닙니다 |

안정화 한계는 공유 엔진의 `scattering_change_bound`와 레거시의 필드 `residual` 중 큰 값이며, 전력
한계는 그 두 배입니다(`POWER_SETTLING_FACTOR`). 진폭 변화의 상한이 되는 것은 공유 엔진 쪽뿐이고,
레거시의 잔차와 전력 배수는 대리 지표입니다. 기록은 어느 쪽이 결정했는지 표시합니다. 순위는 안정화
한계와 비교하지 않습니다.

**장면은 두 지표가 모두 같은 경로를 앞세울 때에만 그 경로에 순위를 매깁니다**
(`studies/fdtd/route_parity/fdtd_route_parity.py::scene_verdict`). 어느 지표도 특권을 갖지 않습니다. 진폭은
디임베딩 위상이 지배하는 지표이고, 전력은 그 영향을 받지 않습니다.

## 3. 재현과 출처

숫자가 설명해야 할 체크아웃의 루트에서 실행하세요(연구는 다른 곳에서 가져온 `wavebit`을
거부합니다).

```shell
python studies/fdtd/route_parity/fdtd_route_parity.py --list
python studies/fdtd/route_parity/fdtd_route_parity.py --scene all --output artifacts/fdtd_route_parity.json
```

출력은 무시 대상인 `artifacts/`에 저장됩니다. 다시 실행한 결과는 로컬에서 생성한 출력이지, 클론과
함께 오는 근거가 아닙니다. 측정한 실행의 기록 `fdtd_route_parity.json`은 저장소 밖에 보관되어 있으며,
이 페이지가 그 측정 내용을 밝힙니다. 이 실행은 `--arms matched`와 `--common-pad legacy`(둘 다
기본값)로, `--check-inert` 없이, 그리고 목표 1e-16, 셀 24개, 주기 640인 `-pml16` 쌍둥이 장면과 함께
실행했습니다(JSON의 `settings`). 기록에는 경과 시간이 없으며, 이 페이지도 비용을 밝히지 않습니다.

| 항목 | 값 |
| --- | --- |
| 기계 | Windows 11 (10.0.22631), CPU, BLAS/OpenMP 스레드 하나 |
| 소프트웨어 | Python 3.12.14, NumPy 2.5.1, SciPy 1.18.0, array-api-compat 1.15.0. NumPy 백엔드만 |
| `git_head` | `99226a8fcda0bc55c6bc153ee71ddd2c4ba0ac25`. 실행한 커밋이지만 그 뒤 다시 쓰였으므로 아래 다이제스트로 소스를 식별합니다. `package_tree_dirty`는 false |
| 장면 | 40개: 배포된 장면 36개와 `-pml16` 쌍둥이 4개 |

기록에는 소스 다이제스트가 있습니다(`studies/fdtd/route_parity/fdtd_route_parity.py::provenance`). 연구 자체의
규칙(`studies/fdtd/route_parity/fdtd_route_parity.py::_digest`, CRLF를 LF로 읽음)으로 현재 트리에 대해 다시
계산한 결과는 다음과 같습니다.

| 다이제스트 | 기록 | 현재 트리 |
| --- | --- | --- |
| `references_sha256` | `eac2a1b0…` | 일치 |
| `study_sha256` | `95b5d973…` | 다름 |
| `package_sha256` | `30ac7083…` | 다름 |
| `source_sha256` | `5018c678…` | 다름(위 세 개를 포함) |

**연구 소스가 다른** 이유는 이제 `-pml16` 쌍둥이가 두 엔진 모두에 `pml_reflection`을 넘기는데,
측정한 실행에서는 호출하는 동안 레거시의 프로필 함수를 바꿔 끼워 CPML 목표를 정했기 때문이며, 또
그 뒤에 docstring을 고쳤기 때문입니다. 키워드를 추가할 때 둘을 바이트 단위로 비교했습니다.
`material-magnetic-r20`에서의 연구 레거시 쪽 전체, 자성·Drude·Lorentz r20 쌍둥이와
`binary_grating-r50`, 그리고 네 쌍둥이 모두의 CPML 계수를 비교했으며, 진단값이 이제 목표를
`pml_reflection`이라고 부른다는 점만 빼면 동일했습니다. 그 비교는 여기 있는 파일이 아니라 키워드를
추가한 커밋에 기록되어 있습니다. 기록의 `definitions.termination_control` 문구는 레거시에 목표
키워드가 없다고 말하는데, 이것은 현재 트리가 아니라 측정한 실행을 설명합니다.

**패키지는** 파일 다섯 개에서 다릅니다. JAX와 Torch 백엔드에서 RCWA 풀이의 장치 배치
(`src/wavebit/rcwa/_engine/engine.py`, `src/wavebit/rcwa/_engine/parametric.py`, `src/wavebit/rcwa/_backends/jax_backend.py`),
공유 엔진이 이제 균일 매질 기준 실행을 셀 두 개짜리 가로 열에서 진행하는 것
(`src/wavebit/fdtd/_periodic/scene.py`, 전체 격자 기준과의 일치는
`tests/fdtd/test_fdtd_uniform_reference.py`가 확인합니다), 그리고 레거시 엔진에 추가된 `pml_reflection`
(`src/wavebit/fdtd/_periodic/legacy.py`)입니다. 이 장면 40개는 그 어떤 변경 뒤에도 다시 실행하지
않았습니다. 아래의 모든 숫자는 `git_head` 시점 패키지의 측정으로 읽으세요.

## 4. 장면

| 그룹 | 장면 수 | 내용 | 기준, `u` |
| --- | ---: | --- | --- |
| slab | 18 | 1D와 교차 평면 슬래브. 유전체, 자성, Drude. 수직, 비스듬한, conical 입사 (`tests/distribution/fdtd_install_smoke.py`, `psi` 중복 제거) | RCWA, 정확: 반올림 하한 |
| crossed | 5 | 해상도 16.67, 25, 33.33의 `square_18deg`. 20, 30의 `rect_28deg` (`studies/cross_solver/alpha_convergence/fdtd_alpha_convergence.py`) | RCWA 차수 열. 진폭 `u` 1.4e-4와 2.1e-4 |
| vacuum | 4 | 해상도 25와 50, `pad_scale` 1과 1.5 (`tests/fdtd/test_fdtd_boundary_controls.py`) | RCWA, 해석적 위상과 대조: 반올림 하한 |
| materials | 4 | 해상도 20의 교차 자성, Drude, Lorentz 래스터. Lorentz는 40에서도 (`tests/references/test_meep_crossed_materials_golden.py`) | 진폭 `u` 2.0e-5에서 2.5e-4 |
| materials-pml16 | 4 | 같은 네 장면을 CPML 목표 1e-16, 셀 24개, 주기 640으로. 이 연구에서 만듦 | 기본 장면과 같음 |
| metal | 2 | 해상도 30과 60의 감쇠 있는 음의 epsilon 금속 (`studies/fdtd/metal_validation/fdtd_metal_validation.py`) | 진폭 `u` 2.08e-3 |
| binary | 3 | `slab_uniform-r50`(RCWA, Airy 닫힌 형식과 대조)과 [`fdtd.md`](../../guide/fdtd/fdtd.md) 2절의 해상도 50과 100 이진 격자 | 반올림 하한. 격자 진폭 `u` 8.2e-6 |

## 5. 맞춘 것

레거시 엔진은 연구의 키워드로 실행하며, 쌍둥이 장면에서는 CPML 목표도 여기에 포함됩니다(3절). 레거시가
만든 격자를 그 진단값에서 다시 읽어 들이고, 공유 엔진을 그 격자에 맞춰 구동합니다
(`studies/fdtd/route_parity/fdtd_route_parity.py::matched_common`). 셀 크기와 개수, `dt`, 포락선 폭, 펄스
최고점 이후 시간, CPML 깊이와 목표, 같은 마디 위의 모니터 평면, 그리고 레거시의 위쪽 간격에 닿도록
크기를 정한 패딩을 맞춥니다. 패딩은 장면 40개 모두에서 셀 수로 일치합니다. 위쪽 간격 자체는 12개
장면에서 공유 엔진 쪽이 셀 하나만큼 더 큽니다.

- `match.matched`는 장면 40개 중 40개에서 참입니다. 각 엔진이 진행에 쓰는 성분별 `eps`, `mu`, 극점
  채움 배열이 모두 마디마다 상대 1e-9(`MATERIAL_TOL`) 안에서 일치하므로, `inputs_differ`인 장면은
  없습니다. 재료를 비교하는 이유는 일정은 정확히 맞아도 극점 채움은 맞지 않을 수 있기 때문입니다.
  이 연구의 첫 실행에서 레거시가 Ey의 극점 세기를 y 방향으로 반 셀 어긋나게 채우는 것을 발견했고
  (`studies/fdtd/route_parity/fdtd_route_parity.py::KNOWN_INPUT_DEFECTS`), 그 뒤 고쳐서
  `tests/fdtd/test_fdtd_pole_colocation.py`로 고정했습니다.
  `tests/fdtd/test_fdtd_route_parity_capture.py`는 캡처가 그 결함을 계속 볼 수 있게 유지합니다.
- 장면 5개에서는 `dt`가 예외입니다. 1D conical 슬래브 세 개는 비 0.8165로 실행합니다. 1D 격자의
  conical 입사는 `courant < 1` 안에서 레거시의 `dt`에 도달할 수 없습니다. `material-lorentz-r20`과
  그 쌍둥이는 0.9829로 실행합니다. 공유 엔진이 극점을 고려한 CFL 한계에 따라 레거시의 `dt`를
  거부했기 때문입니다. 이 경우 `dt` 비는 교란 요인(confound)이 됩니다.

## 6. 디임베딩: 엔진 차이가 아닌 차이

레거시는 각 모니터 위상을 연속체 `kz`로, 공유 엔진은 이산 Yee `kz`로 계면까지 되돌립니다. 이것은
진폭 위상만 움직입니다.
`legacy_yee_deembedded` 비교 갈래(arm)는 레거시 자신의 진폭을 레거시의 `dt`, 셀, 평면에서 공유 엔진의
공식으로 다시 디임베딩하며, 자유 매개변수는 없습니다
(`studies/fdtd/route_parity/fdtd_route_parity.py::yee_deembedded`). 이 갈래와 공유 엔진의 비교가 엔진
비교이고, `legacy`와 공유 엔진의 비교는 원시 비교입니다. 교체는 마디 평면에서 적용합니다. conical이
아닌 TM 루프에서 레거시는 모니터를 그보다 반 셀 아래에서 표본화하므로, 그 채널에는
`O((kz_yee - kz) dz)` 잔차가 남습니다. 기록에는 없는 연구 자체의 추정에 따르면, 모니터 항만 반 셀
평면으로 옮겨도 그 채널들의 `d`가 최대 약 20% 바뀌지만 바뀌는 사례는 없습니다. 입사 항은 추정하지
않았습니다.

| 진폭 사례 | 원시 (`legacy`) | 엔진 (`legacy_yee_deembedded`) |
| --- | ---: | ---: |
| `same_error` | 4 | 30 |
| `intermediate` | 24 | 1 |
| `route_changes_answer` | 9 | 0 |
| `unresolved_against_settling` | 2 | 5 |
| `both_below_reference_resolution` | 1 | 4 |

교체만으로 생기는 변화는 원시 `d`의 0.57배(`material-lorentz-r40`)에서 1.09배(`vacuum-r25-pad1`)입니다
(`deembedding_control.effect_over_raw_d`). 원시 사례는 장면 33개에서 엔진 사례와 다르며, 27개에서는
`same_error < intermediate < route_changes_answer` 척도에서 더 높은 쪽에 놓입니다(`raw_overstates`).
전력 사례는 당연히 장면 40개 모두에서 두 갈래가 같습니다. 교체는 원시 갈래에 대해서만 양성 대조입니다.
교체는 대부분의 엔진 `d`보다 훨씬 크므로, 엔진 갈래의 분해능에 대해서는 아무것도 증명하지 않습니다.

## 7. 결과

장면 40개에 걸친 엔진 사례(`verdict.engines`)의 개수입니다.

| 사례 | 진폭 | 전력 |
| --- | ---: | ---: |
| `same_error` | 30 | 20 |
| `unresolved_against_settling` | 5 | 9 |
| `both_below_reference_resolution` | 4 | 2 |
| `intermediate` | 1 | 4 |
| `route_changes_answer` | 0 | 3 |
| `one_route_near_exact` | 0 | 2 |

호출 80개 전체로는 `same_error` 50, `unresolved_against_settling` 14,
`both_below_reference_resolution` 6, `intermediate` 5, `route_changes_answer` 3,
`one_route_near_exact` 2입니다. 장면 19개에서 두 지표가 모두 `same_error`이고, 15개에서는 두 지표의
사례가 다릅니다. `same_error`인 경우 `d/min(e)`는 진폭에서 0.0070-0.27, 전력에서 0.0046-0.24입니다.

장면별 표는 실행 기록 `fdtd_route_parity.json`(저장소 밖에 보관)에서 생성했으며, 그 기록은 공유
엔진을 `common`이라고 부릅니다. `unresolved`는 `unresolved_against_settling`, `both_below`는
`both_below_reference_resolution`입니다. `d/min(e)`는 `d`가 가장 큰 채널에서의 값이며, 사례와 관계없이
보고합니다.

| 장면 | 진폭 | d/min(e) | 전력 | d/min(e) | 순위 없는 이유 (진폭; 전력) | 비고 |
| --- | --- | ---: | --- | ---: | --- | --- |
| slab-1d-dielectric-normal | same_error | 0.19 | same_error | 0.24 | 반올림 하한 |  |
| slab-1d-dielectric-oblique | same_error | 0.16 | intermediate | 0.37 | 반올림 하한 |  |
| slab-1d-dielectric-conical | same_error | 0.14 | same_error | 0.049 | 반올림 하한 | dt 예외 (conical, 비 0.8165) |
| slab-1d-magnetic-normal | same_error | 0.17 | same_error | 0.21 | 반올림 하한 |  |
| slab-1d-magnetic-oblique | same_error | 0.087 | intermediate | 0.35 | 반올림 하한 |  |
| slab-1d-magnetic-conical | same_error | 0.072 | same_error | 0.045 | 반올림 하한 | dt 예외 (conical, 비 0.8165) |
| slab-1d-drude-normal | same_error | 0.27 | intermediate | 0.35 | 반올림 하한 |  |
| slab-1d-drude-oblique | same_error | 0.24 | intermediate | 0.58 | 반올림 하한 |  |
| slab-1d-drude-conical | same_error | 0.2 | same_error | 0.057 | 반올림 하한 | dt 예외 (conical, 비 0.8165) |
| slab-crossed-dielectric-normal | same_error | 0.1 | same_error | 0.13 | 반올림 하한 |  |
| slab-crossed-dielectric-oblique | same_error | 0.088 | same_error | 0.067 | 반올림 하한 |  |
| slab-crossed-dielectric-conical | same_error | 0.088 | same_error | 0.067 | 반올림 하한 |  |
| slab-crossed-magnetic-normal | same_error | 0.052 | same_error | 0.12 | 반올림 하한 |  |
| slab-crossed-magnetic-oblique | same_error | 0.047 | same_error | 0.057 | 반올림 하한 |  |
| slab-crossed-magnetic-conical | same_error | 0.047 | same_error | 0.057 | 반올림 하한 |  |
| slab-crossed-drude-normal | same_error | 0.15 | same_error | 0.19 | 반올림 하한 |  |
| slab-crossed-drude-oblique | same_error | 0.13 | same_error | 0.094 | 반올림 하한 |  |
| slab-crossed-drude-conical | same_error | 0.13 | same_error | 0.093 | 반올림 하한 |  |
| square_18deg-r16.67 | same_error | 0.09 | unresolved | 1.7 | 취약; 안정화 |  |
| square_18deg-r25 | unresolved | 0.032 | unresolved | 0.12 | 안정화; 안정화 |  |
| square_18deg-r33.33 | unresolved | 0.07 | unresolved | 0.079 | 안정화; 안정화 |  |
| rect_28deg-r20 | unresolved | 0.11 | unresolved | 0.081 | 안정화; 안정화 |  |
| rect_28deg-r30 | unresolved | 0.16 | unresolved | 0.27 | 안정화; 안정화 |  |
| vacuum-r25-pad1 | same_error | 0.12 | one_route_near_exact | 19 | 반올림 하한 | 패딩에 따라 오차가 작은 경로가 바뀜 (전력) |
| vacuum-r25-pad1.5 | same_error | 0.12 | route_changes_answer | 1.5 | 반올림 하한 | 패딩에 따라 오차가 작은 경로가 바뀜 (전력) |
| vacuum-r50-pad1 | same_error | 0.08 | route_changes_answer | 9.6 | 반올림 하한 | 패딩에 따라 오차가 작은 경로가 바뀜 (진폭, 전력) |
| vacuum-r50-pad1.5 | same_error | 0.12 | one_route_near_exact | 138 | 반올림 하한 | 패딩에 따라 오차가 작은 경로가 바뀜 (진폭, 전력) |
| material-magnetic-r20 | same_error | 0.13 | same_error | 0.054 | 10 u 미만; common 우위 |  |
| material-drude-r20 | same_error | 0.18 | same_error | 0.027 | 10 u 미만; common 우위 |  |
| material-lorentz-r20 | same_error | 0.11 | unresolved | 0.93 | 10 u 미만; 안정화 | dt 예외 (pole_cfl, 비 0.9829) |
| material-lorentz-r40 | intermediate | 0.44 | both_below | 84 | 10 u 미만; 10 u 미만 |  |
| material-magnetic-r20-pml16 | unresolved | 0.00047 | unresolved | 0.001 | 안정화; 안정화 |  |
| material-drude-r20-pml16 | both_below | 0.24 | unresolved | 0.00012 | 10 u 미만; 안정화 |  |
| material-lorentz-r20-pml16 | same_error | 0.007 | same_error | 0.0046 | 10 u 미만; 10 u 미만 | dt 예외 (pole_cfl, 비 0.9829) |
| material-lorentz-r40-pml16 | both_below | 0.012 | unresolved | 0.0035 | 10 u 미만; 안정화 |  |
| metal-r30 | both_below | 0.017 | same_error | 0.013 | 10 u 미만; 10 u 미만 |  |
| metal-r60 | both_below | 0.0089 | both_below | 0.009 | 10 u 미만; 10 u 미만 |  |
| slab_uniform-r50 | same_error | 0.0096 | route_changes_answer | 2.1 | 반올림 하한 |  |
| binary_grating-r50 | same_error | 0.072 | same_error | 0.23 | 취약; common 우위 |  |
| binary_grating-r100 | same_error | 0.047 | same_error | 0.17 | 10 u 미만; 10 u 미만 |  |

## 8. 어느 경로도 순위가 매겨지지 않는 이유

`verdict.engines.ranking`은 장면 40개 모두에서 비어 있습니다.

**장면 23개는 측정이 아니라 규칙에 따라 보류됩니다.** 슬래브 18개, 진공 장면 4개, `slab_uniform-r50`은
두 지표 모두에서 `u`가 반올림 하한이며, 그곳에서는 어떤 `e1 != e2`도 10 u를 넘어섭니다. 이것은 측정된
동률이 아닙니다. 구성상 연구가 그곳에서 순위를 매길 수 없을 뿐입니다.

나머지 17개는 다음 이유로 보류됩니다.

| 진폭 | 전력 | 장면 수 |
| --- | --- | ---: |
| 안정화 | 안정화 | 5 |
| 10 u 미만 | 10 u 미만 | 5 |
| 10 u 미만 | 안정화 | 3 |
| 10 u 미만 | 공유 엔진 우위 | 2 |
| 취약 | 안정화 | 1 |
| 취약 | 공유 엔진 우위 | 1 |

장면 세 개에서는 전력만 공유 엔진이 더 정확하다고 순위를 매겼고 진폭은 보류되었으므로, 장면 수준
규칙이 그 순위를 버렸습니다. 아래 여유는 `ranking_margin`에서 가져온 값이며, 전력 안정화 한계
(`summary.power.settling`)와, 쌍둥이가 있는 경우 쌍둥이의 더 작은 최악 전력 오차(`families`)에 대한
것입니다.

| 장면 | 상대 차이 | 차이 / 10 u | 차이 / 안정화 한계 | 쌍둥이에서의 차이 | 진폭을 보류한 이유 |
| --- | ---: | ---: | ---: | --- | --- |
| material-magnetic-r20 | 5.1% | 3.2 | 1.2 (대리 지표) | 1.9e-6, 분해되지 않음 | 10 u 미만 |
| material-drude-r20 | 2.6% | 1.2 | 1.5 (대리 지표) | 1.4e-6, 분해되지 않음 | 10 u 미만 |
| binary_grating-r50 | 1.2% | 4.7 | 7.4 (대리 지표) | 쌍둥이 없음 | 취약 |

두 재료 장면에서는 순위가 대리 지표 한계를 겨우 넘는 수준이며, 종료 조건을 강화하면 두 최악 전력 오차가
서로 1.9e-6, 1.4e-6 이내로 가까워져 쌍둥이의 10 u와 안정화 한계 아래로 내려갑니다. 쌍둥이의 전력
판정은 `unresolved_against_settling`입니다. 그 두 순위는 쌍둥이에서 살아남지 못합니다.
`binary_grating-r50`의 순위는 안정화 한계를 7.4배 넘고 쌍둥이도 없으며, 오직 장면 수준 규칙 때문에
버려집니다.

## 9. 두 엔진이 불일치하는 곳

**진공: 오차가 더 작은 경로가 패딩을 따라갑니다.** 전력에서는 두 해상도 모두에서 `pad_scale` 1일 때
레거시의 오차가, 1.5일 때 공유 엔진의 오차가 더 작으므로, 네 장면 모두 뒤바뀜 목록에 올라 있습니다
(`families.vacuum.smaller_route_flips_with_pad`). 진폭에서도 해상도 50에서 같은 뒤바뀜이 나타납니다.
두 개는 `route_changes_answer`, 두 개는 `one_route_near_exact`이며, 후자의 `d/min(e)` 19와 138은 그보다도
큽니다. 기록은 이것을 엔진이 아니라 `pml_structure_gap` 아래에 분류합니다. 레거시의 여유 간격은
아래보다 위가 크지만 공유 엔진은 그렇지 않으며, `--common-pad legacy`는 공유 엔진이 양쪽에 쓰는 패딩
하나의 크기를 레거시의 위쪽 간격에 맞춥니다.

| 장면 | 레거시 간격, 위 / 아래 (셀) | 공유 엔진 간격, 위 / 아래 (셀) | 전력 사례 |
| --- | --- | --- | --- |
| vacuum-r25-pad1 | 18 / 10 | 18 / 18 | `one_route_near_exact` |
| vacuum-r25-pad1.5 | 25 / 16 | 26 / 26 | `route_changes_answer` |
| vacuum-r50-pad1 | 32 / 20 | 32 / 32 | `route_changes_answer` |
| vacuum-r50-pad1.5 | 48 / 30 | 48 / 48 | `one_route_near_exact` |

진공의 모든 비는 반올림 하한 `u`에 대해 계산한 것이며, 그런 경우 한 엔진이 정확한 값에 가깝게
떨어지면 그것이 `d/min(e)`를 결정합니다.

**1D 슬래브 네 개는 전력에서 `intermediate`입니다** (`d/min(e)` 0.35-0.58). 비스듬한 입사의 유전체,
자성, Drude 슬래브에서는 전력 `d`가 레거시의 TM 루프인 `p.T(0,0)`에 있고, 수직 입사의 Drude에서는
레거시의 TE 루프인 `s.T(0,0)`에 있습니다. 이들의 진폭은 `same_error`입니다.
`tm_half_cell_sampling`은 네 장면 모두에 해당하지만 TM 루프에만 작용하며, 전력에서 `same_error`인 수직
입사의 유전체·자성 슬래브에도 똑같이 해당합니다. 여기서는 아무것도 측정하지 않았습니다. 교란 요인은
장면별로 나열되어 있습니다.

**`slab_uniform-r50`은 전력에서 `route_changes_answer`이며, 아무것도 그 원인으로 지목되지 않습니다.**
패딩 뒤바뀜 플래그가 없는 것은 그 계열에 변형이 하나뿐이기 때문입니다. 여기에 해당하는 교란 요인은
`source_normalisation`, `deembedding_phase`, `tm_half_cell_sampling`, `pml_backing`,
`pml_structure_gap`(레거시 36 / 16 셀, 공유 엔진 36 / 36), `pulse_delay_and_carrier`입니다. 전력 `d`는
`tm.T(0,0)`에 있습니다. `d` 3.6e-4에 대해 `e1` 1.69e-4, `e2` 1.89e-4이므로 `d`는 대략 `e1 + e2`입니다.
`te.T(0,0)`에서 `d`는 5.0e-5입니다. 균일 슬래브의 수직 입사에서 TE와 TM은 같은 물리이며, 그곳에서
엔진을 구별하는 관찰이 하나 있습니다.

| | 레거시 | 공유 엔진 |
| --- | ---: | ---: |
| `te.T(0,0)` 오차 | 1.387e-4 | 1.886e-4 |
| `tm.T(0,0)` 오차 | 1.690e-4 | 1.886e-4 (TE와 1e-14 이내로 같음) |
| 에너지 결손, TE 쪽 | -1.284e-3 | -1.194e-3 |
| 에너지 결손, TM 쪽 | -1.592e-3 | -1.194e-3 |

공유 엔진은 TE와 TM에 같은 답을 주지만 레거시는 그렇지 않습니다. 레거시의 TE 쪽은 공유 엔진에서
5.0e-5 떨어진 곳에, TM 쪽은 3.6e-4 떨어진, 기준값의 반대편에 놓이며, 그곳에서는 공유 엔진의 오차가 더
큽니다. 각 쪽은 각자의 시간 루프입니다.
`tm_half_cell_sampling`(레거시는 공유 엔진이 E를 표본화하는 평면에서 반 셀 떨어진 곳에서 Hy를
표본화합니다)은 TM 쪽 차이 가운데 후보 하나일 뿐이며, 그중 어느 것도 실행으로 시험하지 않았습니다.
이것은 관찰이지 원인 규명이 아닙니다.

`material-lorentz-r40`은 스치는 `s.t_s(0,1)` 채널에서 진폭이 `intermediate`(`d/min(e)` 0.44)입니다.
10절에서 다룹니다.

## 10. 스치는 차수와 종료 조건

출사 매질에서 cos(theta)가 작은 각도로 나가는 차수는 종료부(CPML, 뒷면, 간격)에 의해 일부 되돌려
보내지며, 두 엔진은 그 배치가 다릅니다. 연구는 `GRAZING_COS` = 0.3
(`studies/fdtd/route_parity/fdtd_route_parity.py::GRAZING_COS`)보다 작은 채널을 `grazing`으로 표시합니다.
재료 장면에서 투과 (0, 1) 차수는 0.263으로, 반사 (0, -1) 차수는 0.242로 나갑니다.
각 재료 장면에는 CPML 목표 1e-16, 셀 24개, 주기 640인 `-pml16` 쌍둥이가 있고
(`studies/fdtd/route_parity/fdtd_route_parity.py::PML16`), `termination_control`이 그 쌍을 채널별로 비교합니다
(`studies/fdtd/route_parity/fdtd_route_parity.py::termination_control`). 쌍둥이는 목표, 깊이, 실행 길이를 한꺼번에
바꾸므로, 변화에는 기본 실행이 안정화하지 못했던 부분도 함께 들어 있습니다. 네 쌍둥이 모두 입력이
일치했습니다.

| 장면 | 스치는 채널 최악 `d`, 기본 | 스치는 채널 최악 `d`, 쌍둥이 | 쌍둥이 안정화 한계 (대리 지표) | 쌍둥이 `d` 분해됨 | 그 밖의 채널 최악 `d`, 기본 / 쌍둥이 | 기본 판정 (진폭 / 전력) | 쌍둥이 판정 |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| material-magnetic-r20 | 2.53e-3 | 1.14e-5 | 1.23e-5 | 아니오 | 1.18e-3 / 1.00e-5 | same_error / same_error | unresolved / unresolved |
| material-drude-r20 | 3.05e-4 | 5.94e-6 | 6.28e-6 | 아니오 | 7.44e-4 / 9.49e-6 | same_error / same_error | both_below / unresolved |
| material-lorentz-r20 | 2.76e-3 | 1.88e-4 | 2.16e-5 | 예 | 1.16e-3 / 9.28e-5 | same_error / unresolved | same_error / same_error |
| material-lorentz-r40 | 3.26e-3 | 2.15e-5 | 1.50e-5 | 예 | 3.82e-4 / 7.19e-6 | intermediate / both_below | both_below / unresolved |

- magnetic-r20과 두 Lorentz 장면에서 기본 진폭 `d`는 스치는 `s.t_s(0,1)` 채널에 있었고, 목표, 깊이,
  실행 길이를 함께 강화하자 스치는 채널의 최악 `d`가 221배(magnetic), 15배(lorentz-r20),
  152배(lorentz-r40) 줄었습니다. 따라서 그 대부분은 엔진 내부가 아니라 종료부, 또는 더 짧은 기본 실행이
  안정화하지 못한 부분이었습니다. 쌍둥이는 이 둘을 구별하지 않습니다. magnetic-r20과 drude-r20에서는
  쌍둥이의 스치는 채널 `d`가 쌍둥이의 안정화 한계보다 작습니다.
- 두 Lorentz 장면 모두에 분해된 스치는 채널 나머지가 남습니다. `material-lorentz-r20-pml16`의
  `p.t_p(0,1)`에서 1.9e-4인데, 이 쌍둥이의 판정은 두 지표 모두 `same_error`이지만 `dt`가
  예외입니다(극점 CFL, 비 0.9829). 그리고 `material-lorentz-r40-pml16`의 반사 `s.r_p(0,-1)`에서
  2.1e-5이며, 이 장면의 `dt`는 맞춰져 있습니다. 그 쌍둥이의 (0, 1) 최악 `d` 1.0e-5는 안정화 한계보다
  작습니다.
- drude-r20의 기본 진폭 `d`는 정반사 `s.r_s(0,0)` 채널에 있었습니다. 스치지 않는 채널의 최악 `d`는 네
  장면 모두에서 12배에서 118배 줄었습니다. 쌍둥이의 값은 drude-r20과 lorentz-r20에서는 분해되고,
  magnetic-r20과 lorentz-r40에서는 쌍둥이의 안정화 한계보다 작습니다. 스치지 않는 채널도 이 변화에서
  자유롭지 않습니다.

[CPML 세기](fdtd-pml-strength.md)는 다른 장면, 즉 공유 엔진만으로 계산한 레일리 근처 이진 격자에서
관련된 효과를 기록합니다. 폭만 늘리면 스치는 반환이 줄지 않고, 세기만 늘리면 줄지만 차수 0 반환이
커지며, 세기만 늘린 실행과 둘 다 늘린 800주기 실행은 그 산란 안정화 검사에 실패합니다.

## 11. 이 기록이 결정하지 않는 것

- **공유 엔진 자체의 기본값.** 맞춘 갈래만 실행했으며(`settings.arms`), `common_verbatim` 항목이 있는
  장면은 없습니다. 원문 그대로(verbatim) 갈래로도 "`backend='numpy'` 추가하기"를 측정할 수는 없습니다.
  `studies/fdtd/route_parity/fdtd_route_parity.py::verbatim_common`은 모든 장면을 0.8이 아니라 `courant` 0.99로
  실행하며(21개 장면은 그 자체의 기본값으로, 19개는 연구가 넘겨주는 값으로), `--common-pad default`는
  맞춘 갈래만 바꿉니다. 기록의 `scope`도 공유 엔진 자체의 기본값은 결론이 나지 않았다고 말합니다.
- **소스 주입.** `source_normalisation`(TFSF 평면 대 기준 실행으로 보정한 전류 시트)에는 갈래도
  `measured_in`도 없습니다. `pulse_delay_and_carrier`와 마찬가지로 모든 `d`에 섞여 들어 있습니다.
  `transverse_demodulation`, `subpixel_order`, `tm_half_cell_sampling`은 해당하는 장면(34개, 17개,
  9개)의 모든 `d`에 섞여 들어 있습니다.
- **뒷면과 간격.** `pml_backing`이나 `pml_structure_gap`을 바꾸는 갈래는 없습니다. 쌍둥이는 CPML의
  목표와 깊이, 실행 길이를 바꿀 뿐 뒷면이나 간격은 바꾸지 않는데도, 기록은 재료 장면 8개에서 두 교란
  요인 모두에 `measured_in` `termination_control`이라는 레이블을 붙입니다.
  진공에서는 간격이 패딩 뒤바뀜으로만 드러납니다.
- **반올림 하한 장면에서의 순위와 안정화 한계 아래의 `d`.** 구성상 결정할 수 없습니다.
- **`convergence_window`의 무영향성.** `--check-inert`를 실행하지 않았으므로, 이 기록에서는 공유 엔진에
  넘긴 진단 옵션이 비트 단위로 영향이 없다는 것을 보이지 못합니다.
- **범위 밖:** 비균일 메시(`matched_common`이 거부합니다), 아래쪽 입사, 형상 레시피, JAX와 Torch(공유
  엔진은 NumPy에서만 실행했습니다), 그래디언트, GPU, 레거시 전용 진단, 비용.
- **현재 패키지**(3절).

따라서 이 기록을 근거로 `default_route`를 `legacy_numpy`에서 바꿀 수 없고, 어느 엔진을 없앨 수도
없습니다. 기록이 쌍을 분해할 수 있었던 곳에서는 두 엔진이 대체로 같은 오차를 냈습니다. 분해할 수
없었거나 두 엔진이 불일치한 곳에서는 어느 엔진이 옳은지 말하지 않습니다.

## 12. 교란 요인

장면별로 `confounds` 아래에 나열되어 있습니다. 해당 장면 수는 장면 40개에 대해 센 것입니다.

| 교란 요인 | 장면 수 | 무엇이 다른가 | 측정 |
| --- | ---: | --- | --- |
| `source_normalisation` | 40 | 1D 입사 격자를 쓴 TFSF 평면 대 균일 매질 기준 실행으로 보정한 전류 시트 | 아니오 |
| `deembedding_phase` | 40 | 연속체 대 Yee `kz` | `legacy_yee_deembedded` 갈래로 마디 평면에서 제거. conical이 아닌 TM 루프에는 반 셀 잔차가 남음(6절) |
| `pml_backing` | 40 | 접지한 끝 행 대 주기적 감싸기 | 바꾸지 않음. 장면 8개에서 `termination_control` 레이블 |
| `pml_structure_gap` | 40 | 위가 아래보다 큰 여유 간격 대 셀 하나 이내로 같은 간격 | 바꾸지 않음. 장면 8개에서 `termination_control` 레이블. 진공에서는 패딩 뒤바뀜 |
| `pulse_delay_and_carrier` | 40 | 최고점이 폭의 4배 대 5배. 실수 사인 대 복소 반송파. 최고점 이후 시간은 같게 맞춤 | 아니오 |
| `transverse_demodulation` | 34 | 성분별 x 대 면 보간을 cos(kx dx/2)로 나눈 것 | 아니오 |
| `subpixel_order` | 17 | 같은 평균 규칙, 톱니 모서리에서 다른 순서 | 아니오 |
| `tm_half_cell_sampling` | 9 | 레거시의 2D TM 진행기는 공유 엔진의 E 평면에서 반 셀 떨어진 곳에서 Hy를 표본화 | 아니오 |
| `conical_dt_and_dy` | 3 | 연속체 d/dy 대 셀 두 개짜리 Bloch 차분. `dt` 비 0.8165 | 아니오 |
| `pole_cfl_dt` | 2 | 공유 엔진의 극점 고려 CFL 한계가 레거시의 `dt`를 거부. 비 0.9829 | 아니오 |
