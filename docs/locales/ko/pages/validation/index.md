# 검증 연구

- [RCWA](rcwa/index.md)
- [FDTD](fdtd/index.md)

## 솔버를 검사하는 방법

일반 솔버는 자기 자신과의 비교만이 아니라 외부 기준과 비교해 검사합니다. 이 검사들은 기록된
사례에 대한 것이며, 실험적 접촉 보조 솔버의 광학적 수렴이나 열림 미분을 보증하지 않습니다.

* **외부 RCWA 솔버 셋** — Octave 브리지를 거친 RETICOLO, 그리고 fmmax와 S4. 이들의 수치는
  골든 값으로 커밋되어 CI에서 검사되므로, 이 프로젝트가 배포하지 않는 소프트웨어가 필요한 절반은
  손으로 한 번만 실행합니다. 이 중 둘은 합계만이 아니라 개별 회절 *차수*를 검증합니다. RETICOLO는
  약 1e-13까지 일치하고, `factorization='laurent'`와 같은 방식으로 연산자를 조립하는 fmmax의 `FFT`
  정식화는 두 편광의 모든 차수에서 약 3e-11까지 일치합니다
  ([`test_fmmax_laurent_golden.py`](../../tests/references/test_fmmax_laurent_golden.py)).
* **완전히 다른 방법** — Wavebit 자체의 FDTD([`docs/guide/fdtd/fdtd.md`](../guide/fdtd/fdtd.md))가 같은
  장면을 시간 영역에서 풀고, Meep이 그 외부 기준입니다. RCWA는 푸리에 급수를 자르고 FDTD는 공간과
  시간을 이산화하므로, 두 방법 모두에서 살아남는 결함은 어느 쪽 탓도 아닙니다. Meep의 답은
  [`tests/references/golden_meep_1d.npz`](../../tests/references/golden_meep_1d.npz)(1D 기준 문제, 두 편광,
  두 해상도, 실행을 판정하는 평면 닫힌꼴 보정 포함)와 `golden_meep_crossed.npz`(conical 입사의 교차
  격자, 차수별 복소 진폭)로 커밋되어 있고, CI가 `main`에 대한 모든 push와 모든 pull request에서 이를
  읽습니다. 기준 격자의 5 nm 격자 간격에서 수렴한 RCWA 답 대비 우리의 최악 차수별 차이는 6.6e-4(TE) /
  6.9e-4(TM)이고, 같은 격자에서 Meep 자신은 4.9e-4 / 7.4e-4입니다. 다만 같은 해상도에서 Meep 열이
  자신의 PML 설정 두 가지 사이에서 1.6e-3만큼 움직인다는 사실과 함께 읽어야 합니다. 그 이유는 §2에
  설명되지 않은 것으로 기록되어 있으므로, 어느 수치도 기준 자체의 흔들림보다 촘촘하지 않습니다. 이 네
  수치와, 그것이 요약하는 표의 모든 칸은 markdown에서 다시 읽어
  [`golden_fdtd_table.npz`](../../tests/references/golden_fdtd_table.npz)와 비교합니다. 세 열을 얻은
  조건은 `tests/references/_meep_1d_cases.py`에 값으로 적혀 있습니다. 2026-09-07까지는 그 조건이
  "배포 기본값에서"라는 문구였고, 그 뒤에서 한 열이 조용히 낡아 있었기 때문입니다. Meep은 GPL이고 이
  프로젝트는 Apache-2.0이므로, Meep은 블랙박스 기준으로만 실행되며 여기 있는 어떤 것의 의존성도
  아닙니다.
* **공개된 기준 표** — Demésy 외(2012)가 표로 정리한 Li(1997)의 교차 격자 체커보드. 투과 차수 아홉
  개 모두가 최악 2.06e-4로 일치하며, 이는 사각이 아닌 기본 셀 위에서 얻은 것이라 그 경로를 이 프로젝트
  바깥의 수치와 맞대 봅니다.
* 무손실 구조의 **에너지 보존**: 테스트는 대부분의 장면에서 |R+T−1|을 1e-12에서 1e-10 아래로,
  레일리 점 옆에서는 1e-6에서 1e-4 아래로 유지합니다.
* **백엔드 간 일치** — 같은 입력에 대한 NumPy, JAX, PyTorch.
* 최적화기가 움직이는 매개변수에 대한 **해석적 그래디언트와 유한 차분의 비교**.

테스트 모음은 [`tests/`](https://github.com/fieldalgo/wavebit/tree/main/tests)입니다. 기준을 명시할 수
있는 것은 더 느린 백엔드 간 검사와 최적화기 검사를 포함해 모두 여기에 있습니다. CI는 `main`에 대한
모든 push와 모든 pull request에서 기본 묶음을 실행합니다. 실험적 conforming NVF 모듈과 그 연구 도구의
테스트는 CI가 실행하지 않는
[연구 단계](https://github.com/fieldalgo/wavebit/blob/main/tests/README.md#the-research-tier)를 이루며,
그 연구 코드를 개발하는 동안 실행합니다. 즉시 실행되는(eager) NumPy 계산은 모두 돌려주기 직전의
값을 검사합니다. NaN이나 무한대가 있으면 오염된 배열의 이름을 담아 `FloatingPointError`를 냅니다.

일반적인 "R+T가 1을 넘음" 경고는 일부러 두지 않습니다. 그런 경고는 망가진 계산과 실제로 증폭하는
구조를 구별하지 못합니다. 여기서 이득은 `Im(eps) < 0`이고, 이때 `R + T > 1`이나 전반사 아래의 음수
차수가 올바른 답입니다. 에너지는 기대값을 아는 테스트 모음 안에서 장면마다 단언합니다.
[`docs/reference/common/result-guards.md`](../reference/common/result-guards.md)를 보세요.

[반입한 교육 자료 기록](../development/learning-provenance.md)은 원래의 소스 리비전과 도구 체인을
그대로 유지합니다. 이는 과거의 근거이며, 지금 체크아웃이 그것을 다시 실행했다는 주장이 아닙니다.
