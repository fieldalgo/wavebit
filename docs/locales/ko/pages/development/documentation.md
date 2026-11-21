# 문서와 번역

영어가 정본 언어입니다. 탐색 구조는 독자의 여섯 가지 과제를 따릅니다: 시작하기, 배우기, 사용자
가이드, API 레퍼런스, 이론과 검증, 개발자 가이드. 소스 페이지, docstring, 실행 가능한 예제는 한
곳에서만 관리합니다.

## 영어 사이트 빌드

먼저 체크아웃의 개발 환경을 활성화하세요.

```sh
python -m pip install -e '.[docs]'
python -m sphinx -W -b html docs docs/_build/html
```

빌드는 이 체크아웃의 API docstring을 읽으며, 다르게 설치된 사본은 거부합니다. 출력은 로컬에만
남고, 빌드한다고 사이트가 게시되지는 않습니다.

## 두 언어 사이트 빌드

게시되는 결과물은 영어를 기존 루트 URL에 두고 한국어를 `ko/` 아래에 더합니다. 머리글의 언어 선택
메뉴는 같은 페이지를 다른 언어로 엽니다. 이 링크는 상대 경로이므로 프로젝트나 버전 접두어 아래에
호스팅해도 바뀌지 않습니다. 명시적인 두 언어 플래그로 두 트리를 함께 빌드하세요.

```sh
python -m sphinx -W -b html -D wavebit_bilingual=1 -d docs/_build/doctrees/en docs docs/_build/html
python -m sphinx -W -b html -D wavebit_bilingual=1 -D language=ko -d docs/_build/doctrees/ko docs docs/_build/html/ko
python -m http.server --directory docs/_build/html 8000
```

`http://localhost:8000/` 또는 `http://localhost:8000/ko/`를 여세요. 두 언어 트리를 함께 유지하려면
`docs/_build/html` 디렉터리 전체를 게시하세요. 문서 CI 작업은 경고를 오류로 취급하며 둘 다
빌드하고, 배포 워크플로도 같은 배치를 씁니다. 로컬에서 빌드하거나 띄워 보는 것은 아무것도 배포하지
않습니다.

이 플래그의 기본값은 false입니다. 영어만 따로 빌드하거나, `-D wavebit_bilingual=1` 없이 한국어만
따로 빌드하면 없는 언어 트리로 가는 링크를 보여 주지 않습니다. 위처럼 두 언어에 별도의 doctree
디렉터리를 쓰세요. 한국어만 로컬에서 살펴보려면:

```sh
python -m sphinx -W -b html -D language=ko docs docs/_build/ko
```

## Contribute Korean translations

완전한 한국어 Markdown overlay는 `docs/locales/ko/pages/` 아래, 번역하는 영어 페이지와 같은 상대
경로에 둡니다. overlay는 한국어 빌드에서만 그 페이지의 소스 전체를 대체합니다. 영어 페이지가 정본으로
남고, overlay 안의 상대 링크·그림·include는 모두 영어 페이지의 위치를 기준으로 해석됩니다. 홈페이지,
반입한 강좌의 장 40개 전부, 유지되는 페이지 대부분에 overlay가 있으며, 어떤 페이지가 포함되는지는 이
디렉터리 자체가 기준입니다. `docs/locales/ko/LC_MESSAGES/` 아래의 Sphinx gettext 카탈로그는 문단
단위로 번역한 페이지를 위해 남아 있습니다. 둘 다 없는 페이지는 영어로 대체 표시한다는 안내를 명시적으로
보여 주므로, 모든 페이지를 쓸 수 있습니다. 현지화한 SVG와 PNG 자산은 같은 정본 상대 경로로
`docs/locales/ko/assets/` 아래에 있습니다. source-read 확장이 영어 내용을 바꾸지 않고 이 자산을
선택합니다.

overlay는 요약이 아니라 현재 영어 페이지의 완전한 번역이며, 사람이 읽는 문장만 바꿉니다. 펜스로 감싼
코드·출력 블록, 인라인 코드, 수식, 숫자, 링크 대상, 지시문 줄, `eval-rst` 블록은 그대로 옮기세요.
어떤 페이지든 `#slug`로 링크하는 제목은 영어로 두세요. 한국어 빌드는 그 링크를 제목 자체에 맞춰 찾기
때문입니다. 영어 페이지가 바뀌면 같은 커밋에서 그 overlay도 고치세요.

gettext 기반 페이지의 현재 영어 메시지는 다음으로 추출합니다.

```sh
python -m sphinx -b gettext docs docs/_build/gettext
```

영어가 바뀌면 해당 `msgid`를 고치고 새 소스에 맞춰 `msgstr`을 검토하세요. 매개변수 이름, 코드, 수식
식별자, 링크는 보존하세요. 번역을 제출하기 전에 두 언어를 모두 빌드하세요. 예전 솔버 규약의 번역을
현재 계약인 것처럼 제시해서는 안 됩니다.

## 연구 노트와 학습 자료

예제는 `examples/`나 `tutorials/`에 두고 정본 소스를 참조하세요. 학습 자료는 API, 규약, 그림,
라이선스를 현재 패키지와 대조해 확인한 뒤 `docs/learn/`으로 가져오세요. 검증 출처를 보존하고, 과거의
결과와 현재의 사용법을 구분하세요.

저자의 원본 메모, 설계 초안, 세션 기록, 비공개 대화 링크, 검토하지 않은 노트북이나 프로토타입은 별도의
비공개 lab 저장소에 두세요. 공개 다운로드로 첨부하지 마세요. 게시한 장은 그 자체로 읽혀야 하며, 독자가
접근할 수 있는 검토된 근거를 인용해야 합니다. 공개 영어 설명과 선택적 한국어 번역을 편집하는 동안
원본은 비공개로 보존하세요.

릴리스 결과물은 빈 출력 디렉터리에 빌드하세요. 증분 Sphinx 빌드는 지운 페이지, 소스 사본, 다운로드
파일을 디스크에 남길 수 있고, 탐색에서 숨겨도 배포에서 그 파일이 지워지지 않습니다. 두 언어를 확인한
뒤, 이전에 제공하던 결과물을 깨끗한 전체 빌드로 교체하세요.
