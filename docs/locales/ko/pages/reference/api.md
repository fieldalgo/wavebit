# API 레퍼런스

공개 네임스페이스는 `wavebit.rcwa`와 `wavebit.fdtd`입니다. 둘 다 불변(immutable) 문제와 명시적인 수치·실행 설정을 받습니다. 공통 재료, 단위, 관측 형식, 구조화된 오류는 패키지 루트에 있습니다.

- [공통 형식](common/index.md): 재료 응답, 물리 단위, 실행, 작업(job), 오류.
- [RCWA](rcwa/index.md): 주기 형상, 입사, 수치 설정, 결과.
- [FDTD](fdtd/index.md): 물리 도메인, 시간 영역 소스, 수치 설정, 모니터.
- [생애주기](../guide/common/lifecycle.md): `check`, `prepare`, `run`과 호환되는 재사용.
- [예전 API](../migration/legacy-api.md): 과거의 정규화된 엔진 세부 사항과, 퇴역한 import를 위한 고정 소스.

아래 페이지들은 현재 공개 선언에서 생성됩니다. 내부 구현 모듈은 소스 참고용이며, 사용자용 대체 API가 아닙니다.
