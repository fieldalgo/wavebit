# RCWA 이론

푸리에 영역 솔버의 바탕이 되는 유도와 수학적 규약입니다.

이 페이지들은 날짜가 적힌 기록이며, 작성 당시의 솔버 인터페이스를 기준으로 합니다. 코드 조각은
예전 인터페이스를 쓰며, 그 인터페이스는 2026-10-13부터 `wavebit.compat.legacy`였습니다. 코드 조각이
`wavebit.rcwa(...)`를 호출하면 `wavebit`은 `from wavebit.compat import legacy as wavebit`을 뜻합니다.
이 네임스페이스는 2026-10-15에 퇴역했으며, [퇴역 직전 리비전 `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a)에서 실행됩니다.
공개 `wavebit.rcwa` 모듈은 [RCWA 가이드](../../guide/rcwa/index.md)에서 소개합니다.

- [복소 진폭을 표시하는 차수별 s/p 좌표계](amplitude-sp-frame.md)
- [분지 절단 — 어느 제곱근인가, 그리고 잘못된 두 가지 경우](branch-cuts.md)
- [층 연결 알고리즘 — ETM, SMM, 그리고 단순 전달 행렬이 목록에 없는 이유](connecting-algorithms.md)
- [Laurent, direct, inverse, Toeplitz: 어느 단어가 규칙이고 어느 단어가 행렬인가](factorization-terms.md)
- [1차 경로 — 기준 솔버가 없을 때 무엇과 비교해 판단하는가](first-order-formulation.md)
- [아홉 칸 중 다섯 칸 — 모든 2차 RCWA가 갖는 형태와 그 경계에서 일어나는 일](five-slot-limit.md)
- [코드 자체의 규약으로 유도한 정식화](formulation.md)
- [사각형 대 원형 푸리에 차수 절단 — 측정](fourier-truncation-shape.md)
- [자성 재료 — 층과 반공간의 `mu`](magnetic-materials.md)
- [음굴절률 매질 — 부호 있는 굴절률, 거울 근, 그리고 `theta`의 의미](negative-index-media.md)
- [금속에서의 inverse 규칙: `<1/eps>^-1`의 극점과 그것을 막는 하한](nvf-metal-inverse-rule.md)
- [래스터 법선장 구성과 `raster_normal_method` / `nvf_projector` 옵션](nvf-normal-methods.md)
- [직교하지 않는(비스듬한) 단위 셀 — `lattice`가 하는 일과 그 비용](oblique-lattice.md)
- [비대각 유전율 칸 — 무엇을 담고, 비대칭 텐서가 왜 틀리게 나왔는가](off-diagonal-slot-convention.md)
- [패턴이 있는 이방성 층 — Popov–Nevière 인수분해](patterned-anisotropic-factorization.md)
- [레일리 점 — 정확한 스침, 어느 경로가 그것을 푸는가, 그리고 `auto_detune`이 하는 일](rayleigh-points.md)
- [이방성 재료의 Li 인수분해](tensor-li-factorization.md)
