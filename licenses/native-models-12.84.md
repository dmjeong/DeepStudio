# 기본 제공 모델 추가 — 출처와 라이선스 (2026-11-11 확인)

| 모델/가중치 | 게시 라이선스 | 원본/범위 |
|---|---|---|
| LibreDFINEn.pt | Apache-2.0 | D-FINE Authors (2024), dfine_n_coco.pth. COCO-only Nano. Objects365 가중치 사용 안 함. HGNetV2: PaddlePaddle Authors, Apache-2.0. |
| LibreMobileNetV4m-cls.pt | Apache-2.0 | Ross Wightman/timm contributors, mobilenetv4_conv_medium.e500_r224_in1k (ImageNet-1k). |
| LibreYOLO 1.5.0 runtime | MIT | YOLO 이름이 같은 다른 구현의 라이선스와 혼동하지 말 것. 배포 시 런타임 LICENSE/NOTICE/THIRD_PARTY_NOTICES 보존. |
| pycocotools 2.0.11 | BSD | COCO 평가용. 라이선스 전문은 pycocotools-LICENSE.txt. |

모델 카드와 저자의 명시된 가중치 라이선스를 확인한 결과다. 학습 데이터의 이용 조건과 사용자가 추가하는 이미지 권리는 별개다.

- https://huggingface.co/LibreYOLO/LibreDFINEn
- https://huggingface.co/LibreYOLO/LibreMobileNetV4m-cls
- https://github.com/Peterande/D-FINE/issues/357

원본 가중치는 변경하지 않고 내려받는다. 사용자의 미세조정/패치 학습 결과는 별도 체크포인트에 저장된다.
Apache-2.0 전문은 각 모델의 LICENSE 파일, 저작권 고지는 NOTICE 파일을 함께 배포한다.

이번 검증 다운로드의 SHA-256:
- LibreDFINEn.pt: a962b8cf97f2f613afcb8d4967eeb5d2f7dfe2461e06c16ab94b3c12ea633635
- LibreMobileNetV4m-cls.pt: cb03ca7a6acd102726f1aa0db09fe2a6e1995fbea097c44606168eac72bb8db3
