# Deep Vision Studio Desktop 11.58

11.57 통합 후 원격 CI의 웹 보안 검사에서 기존 Vite 8.2.2/source-map-js 1.2.1 문제가 확인되었습니다. Vite를 8.3.7로, source-map-js를 1.2.2로 갱신했습니다.

`npm audit --audit-level=high`에서 취약점 0건, 웹 테스트 14개 통과, TypeScript 검사 및 프로덕션 빌드 통과를 확인했습니다. 세그멘테이션 모델·SDK의 연산은 변경하지 않았습니다. 버그 수정 +0.01을 적용했습니다.
