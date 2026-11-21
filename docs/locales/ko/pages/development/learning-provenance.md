# 학습 자료 출처

RCWA 기초 강좌와 NVF 강좌는 별도 lab 저장소의 편집 가능한 Markdown 소스 `wavebit-sphinx-guide/source`에서, 기준 커밋 `89bff30b31dd37455d9c3292ecf030522fd36b05`로 반입했습니다. 정확한 소스 해시에는 반입 당시 있던 저자의 커밋되지 않은 변경도 포함됩니다. lab 체크아웃은 수정하지 않았습니다.

반입 범위는 완전한 장·색인 페이지 40개, 그림 100개, 함께 쓰는 Python 예제 15개, 과거 수치 기록 12개입니다. 영어 번역이 정본 페이지이고, 한국어 원문 전체는 같은 경로의 언어 overlay로 보존하며, 현재 API에 맞춘 수정은 과거 수치 근거와 구분합니다.

- [번역 완료 기록](learning-provenance/translation-receipt.json): 영어 본문 40개 전체와 한국어 overlay.
- [영어 SVG 문자열 번역](learning-provenance/text-figure-translations.json)과 [글리프 그림 번역](learning-provenance/glyph-figure-translations.json).
- [보존한 데이터로 그린 영어 금속 그래프](learning-provenance/metal-figures.json)와 [원래의 영어 합성 그래프](learning-provenance/metal-original-plots.json).
- [확인한 고정 과거 링크](learning-provenance/historical-links.json): 원래 커밋에서의 원래 경로.
- [반입 기록](learning-provenance/import-receipt.json): 모든 목록 항목의 정확한 소스 해시, 원래 Git 상태, 대상 위치와 결정.
- [원래의 장 이전 목록](learning-provenance/migration.json).
- [원래의 문서 검증](learning-provenance/verification.json).
- [원래의 예제 검증](learning-provenance/example-verification.json).
- [원래의 직접 파수 벡터 검증](learning-provenance/direct-k-verification.json).
- [원래의 도구 체인](learning-provenance/toolchain.json).
- [보존한 라이선스 고지](learning-licenses.md).

## 기록의 의미

원래의 측정, 로그, 수치 그림 데이터는 다시 실행하거나 라벨을 바꾸지 않고 보존합니다. 예전 예제 검증은 스크립트 14개를 다루며, 더 새로운 경계 전력 예제를 보증하지 않습니다. 예전 문서 검증은 그 당시의 강좌 수와 빌드를 기술하며, 이 통합 사이트를 기술하지 않습니다. 라벨을 번역하거나 링크 위치를 옮겨도 수치 기록은 바뀌지 않습니다.

## 공개 범위

공개 모음에는 유지되는 강좌, 검토한 그림과 예제, 그리고 명시적으로 표시한 과거 수치 기록이
들어 있습니다. 저자의 원본 메모, 설계 초안, 세션 기록, 탐색용 노트북, 독립 프로토타입 묶음은
별도의 비공개 lab 저장소에 남아 있으며, 공개 다운로드 대상이 아닙니다.

반입 기록은 이 저장소에 보존한 자료를 다룹니다. 소스 해시는 편집한 장이나 그림의 입력을 식별할
뿐, 그 바탕의 비공개 연구 보관소를 공개해야 한다는 뜻이 아닙니다. 독자는 그 보관소나 저자의 대화
링크 없이도 공개 문서를 쓸 수 있습니다.

## 번역과 대체 표시

한국어 장 본문은 `docs/locales/ko/pages` 아래에 있으며 한국어 빌드에서만 읽힙니다. 다른 페이지는 카탈로그가 있으면 gettext를 쓰고, 없으면 한국어 사이트가 영어로 대체 표시한다는 것을 명시합니다. 현지화한 SVG·PNG 라벨은 보존한 수치 데이터와 별개로 선택됩니다. 언어 전환은 항상 현재 경로를 유지합니다.
