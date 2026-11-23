# 설계와 구현 기록

이 문서들은 채택된 설계, 구현 중간 점검, 수치 연구 제안을 보존합니다. 구현된 인터페이스는 [API 레퍼런스](../reference/index.md)를 보세요.

- [RCWA·FDTD 통합 API 설계](unified-api.md) — 채택된 설계로 0.0.2부터 구현됨. §14에 제안 중 구현된 부분과 아직 제안으로 남은 부분이 표시되어 있습니다.
- [대칭 FDTD 스텐실 조사](fdtd-symmetric-stencil.md) — 수치 연구 기록. 여기서 측정한 연산자는 선택형 `fdtd.Numerics(discretization='stencil')`입니다.
