# 개발자 가이드

| 페이지 | 답하는 질문 |
| --- | --- |
| [검증 기록과 안전한 이어 실행](verification.md) | quick/full/research 검사를 어떻게 고르고, 기록을 남기고, 같은 실행을 이어 갈까? |
| [현재 구현 구조와 수정 지점](source-map.md) | 어떤 공개 호출이 어떤 계산에 닿고, 변경에는 어떤 검사가 따를까? |
| [소스 체크아웃과 디버깅](source-checkout.md) | 어느 사본이 import되고, 솔버 코드 안으로 어떻게 들어가 볼까? |
| [RCWA와 FDTD 통합](solver-integration.md) | 어떤 솔버 계약을 공유하고, 예전 주기 구조 호출 코드는 어떻게 옮길까? |
| [제3자 소프트웨어, 데이터와 레퍼런스](third-party.md) | Wavebit은 어떤 외부 구성 요소를 어떤 조건으로 쓰고, 그것을 어떻게 확인할까? |
| [문서와 번역](documentation.md) | 영어 문서를 어떻게 빌드하고, 한국어 번역에 어떻게 기여할까? |
| [기여하기](https://github.com/fieldalgo/wavebit/blob/main/CONTRIBUTING.md) | 새 코드는 어디에 두고, 변경은 어떻게 검토·반영될까? |
| [테스트 모음](https://github.com/fieldalgo/wavebit/tree/main/tests) | 어느 검증 단계를 실행해야 할까? |

구현은 `src/wavebit/` 아래에 있습니다. 문서, 예제, 연구(studies)와
테스트는 저장소 루트에 둡니다.
