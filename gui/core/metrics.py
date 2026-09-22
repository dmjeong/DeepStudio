"""
Deep Vision Studio — 성능 평가 지표 엔진

태스크별 유명 평가 지표:
┌─────────────────────────────────────────────────────────────────┐
│  Classification                                                 │
│  ├── Accuracy (Top-1)        — 전체 정답률                       │
│  ├── Precision (macro)       — 양성 예측 정밀도                   │
│  ├── Recall (macro)          — 실제 양성 검출률                   │
│  ├── F1-Score (macro)        — Precision·Recall 조화 평균        │
│  ├── Confusion Matrix        — 클래스별 혼동 행렬                 │
│  └── Per-class metrics       — 클래스별 P/R/F1                   │
│                                                                  │
│  Segmentation                                                    │
│  ├── mIoU                    — Mean Intersection over Union      │
│  ├── Dice Score              — F1 for segmentation               │
│  ├── Pixel Accuracy          — 전체 픽셀 정확도                   │
│  └── Per-class IoU           — 클래스별 IoU                      │
│                                                                  │
│  Detection                                                       │
│  ├── mAP@0.5                 — IoU≥0.5 기준 mean Average Prec.   │
│  ├── mAP@0.5:0.95            — IoU 0.5~0.95 평균 mAP (COCO)     │
│  ├── Precision               — 예측 중 실제 객체 비율              │
│  ├── Recall                  — 실제 객체 중 예측 비율              │
│  └── Per-class AP            — 클래스별 AP                        │
│                                                                  │
│  Anomaly Detection                                               │
│  ├── AUROC                   — Area Under ROC Curve              │
│  ├── F1-Score                — Optimal threshold 기준 F1         │
│  ├── Precision / Recall      — @ optimal threshold               │
│  └── Confusion Matrix        — Normal vs Anomaly                 │
└─────────────────────────────────────────────────────────────────┘
"""

import numpy as np
from typing import Dict, List, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Classification 평가 지표
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ClassificationMetrics:
    """
    Classification 전용 메트릭 수집기

    사용법:
        meter = ClassificationMetrics(num_classes=10)
        for batch in loader:
            preds = model(images).argmax(dim=1)
            meter.update(preds.numpy(), targets.numpy())
        results = meter.compute()
    """

    def __init__(self, num_classes: int, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        # 혼동 행렬 누적 (행: 실제, 열: 예측)
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        """누적값 초기화"""
        self.confusion_matrix.fill(0)

    def update(self, preds: np.ndarray, targets: np.ndarray):
        """
        배치 결과 누적

        Args:
            preds: 예측 클래스 인덱스 (N,)
            targets: 정답 클래스 인덱스 (N,)
        """
        preds, targets = np.asarray(preds), np.asarray(targets)
        if preds.ndim != 1 or targets.shape != preds.shape:
            raise ValueError("분류 예측과 정답은 길이가 같은 1차원 배열이어야 합니다")
        for values in (preds, targets):
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                raise ValueError("분류 클래스 번호는 유한한 정수여야 합니다")
            if ((values < 0) | (values >= self.num_classes)).any():
                raise ValueError("분류 클래스 번호가 모델 범위를 벗어났습니다")
        np.add.at(self.confusion_matrix, (targets.astype(int), preds.astype(int)), 1)

    def compute(self) -> Dict:
        """
        전체 메트릭 계산

        Returns:
            {
                "accuracy": float,
                "precision_macro": float,
                "recall_macro": float,
                "f1_macro": float,
                "confusion_matrix": ndarray,
                "per_class": [{"name", "precision", "recall", "f1", "support"}, ...]
            }
        """
        cm = self.confusion_matrix
        total = cm.sum()

        # ── 전체 정확도 ──
        accuracy = cm.trace() / max(total, 1)

        # ── 클래스별 Precision / Recall / F1 ──
        # 정답 샘플이 없는 클래스는 평가할 수 없으므로 macro 평균에서 제외한다.
        # 다른 클래스 샘플을 빈 클래스로 오예측한 경우는 전체 accuracy와
        # 해당 실제 클래스의 recall/F1에 그대로 반영된다.
        per_class = []
        precisions, recalls, f1s = [], [], []

        for i in range(self.num_classes):
            tp = cm[i, i]                        # True Positive
            fp = cm[:, i].sum() - tp             # False Positive (열 합 - TP)
            fn = cm[i, :].sum() - tp             # False Negative (행 합 - TP)
            support = cm[i, :].sum()             # 실제 샘플 수

            if support > 0:
                # Precision = TP / (TP + FP)
                prec = tp / max(tp + fp, 1)
                # Recall = TP / (TP + FN)
                rec = tp / (tp + fn)
                # F1 = 2 * P * R / (P + R)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)
            else:
                prec = rec = f1 = None

            per_class.append({
                "name": self.class_names[i] if i < len(self.class_names) else f"Class {i}",
                "precision": None if prec is None else float(prec),
                "recall": None if rec is None else float(rec),
                "f1": None if f1 is None else float(f1),
                "support": int(support),
            })

        # ── Macro 평균 (클래스 균등 가중) ──
        precision_macro = float(np.mean(precisions)) if precisions else 0.0
        recall_macro = float(np.mean(recalls)) if recalls else 0.0
        f1_macro = float(np.mean(f1s)) if f1s else 0.0

        return {
            "accuracy": float(accuracy),
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "confusion_matrix": cm.copy(),
            "per_class": per_class,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Segmentation 평가 지표
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SegmentationMetrics:
    """
    Segmentation 전용 메트릭 수집기

    지표:
    - mIoU: Mean Intersection over Union (가장 중요!)
    - Dice Score: 2*|A∩B| / (|A|+|B|) — 의료 분할에서 표준
    - Pixel Accuracy: 전체 픽셀 중 정답 비율
    """

    def __init__(self, num_classes: int, class_names: List[str] = None, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        # 클래스별 TP, FP, FN 누적
        self.intersection = np.zeros(num_classes, dtype=np.int64)
        self.union = np.zeros(num_classes, dtype=np.int64)
        self.pred_area = np.zeros(num_classes, dtype=np.int64)
        self.gt_area = np.zeros(num_classes, dtype=np.int64)
        self.correct_pixels = 0
        self.total_pixels = 0

    def reset(self):
        self.intersection.fill(0)
        self.union.fill(0)
        self.pred_area.fill(0)
        self.gt_area.fill(0)
        self.correct_pixels = 0
        self.total_pixels = 0

    def update(self, preds: np.ndarray, targets: np.ndarray):
        """
        배치 결과 누적

        Args:
            preds: 예측 마스크 (N, H, W) — 클래스 인덱스
            targets: 정답 마스크 (N, H, W) — 클래스 인덱스
        """
        preds, targets = np.asarray(preds), np.asarray(targets)
        if preds.shape != targets.shape or targets.ndim < 2:
            raise ValueError("분할 예측과 정답 마스크의 크기가 같아야 합니다")
        valid = targets != self.ignore_index
        preds, targets = preds[valid], targets[valid]
        for values in (preds, targets):
            if (not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()
                    or ((values < 0) | (values >= self.num_classes)).any()):
                raise ValueError("분할 클래스 번호가 모델 범위를 벗어났습니다")
        self.correct_pixels += int((preds == targets).sum())
        self.total_pixels += targets.size
        for c in range(self.num_classes):
            pred_c, gt_c = preds == c, targets == c
            self.intersection[c] += (pred_c & gt_c).sum()
            self.union[c] += (pred_c | gt_c).sum()
            self.pred_area[c] += pred_c.sum()
            self.gt_area[c] += gt_c.sum()

    def compute(self) -> Dict:
        """
        전체 메트릭 계산

        Returns:
            {
                "pixel_accuracy": float,
                "mIoU": float,
                "dice_score": float,
                "per_class": [{"name", "iou", "dice"}, ...]
            }
        """
        pixel_accuracy = self.correct_pixels / max(self.total_pixels, 1)

        # 클래스별 IoU, Dice
        per_class = []
        ious, dices = [], []

        for c in range(self.num_classes):
            # IoU = Intersection / Union
            iou = self.intersection[c] / max(self.union[c], 1)
            # Dice = 2 * Intersection / (|Pred| + |GT|)
            dice = 2 * self.intersection[c] / max(self.pred_area[c] + self.gt_area[c], 1)

            # 정답 픽셀이 없는 클래스는 평가 대상이 아니므로 macro에서 제외한다.
            # 그 클래스로의 오예측은 해당 픽셀의 pixel accuracy와 실제 클래스
            # IoU/Dice에는 오류로 반영된다.
            present = self.gt_area[c] > 0
            if present:
                ious.append(iou)
                dices.append(dice)

            per_class.append({
                "name": self.class_names[c] if c < len(self.class_names) else f"Class {c}",
                "iou": float(iou) if present else None,
                "dice": float(dice) if present else None,
            })

        mIoU = float(np.mean(ious)) if ious else 0.0
        dice_score = float(np.mean(dices)) if dices else 0.0

        return {
            "evaluable": self.total_pixels > 0,
            "pixel_accuracy": float(pixel_accuracy),
            "mIoU": mIoU,
            "dice_score": dice_score,
            "per_class": per_class,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Detection 평가 지표
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DetectionMetrics:
    """
    Detection 전용 메트릭 수집기

    VOC 11-point AP를 여러 IoU 기준으로 계산 (전체 COCO 평가기와 다름):
    - mAP@0.5: IoU ≥ 0.5 기준 (PASCAL VOC 표준)
    - mAP@0.5:0.95: IoU 0.5~0.95 step 0.05 평균 (COCO 표준)
    """

    def __init__(self, num_classes: int, class_names: List[str] = None, ap_method="voc11"):
        if ap_method not in {"voc11", "interp101"}:
            raise ValueError("미지원 AP 계산 방식")
        self.ap_method = ap_method
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        # 예측/정답 누적 리스트
        self.all_preds = []   # [(class_id, confidence, x1, y1, x2, y2, image_id), ...]
        self.all_gts = []     # [(class_id, x1, y1, x2, y2, image_id), ...]
        self.image_count = 0
        self._tp_counts = {}

    def reset(self):
        self.all_preds.clear()
        self.all_gts.clear()
        self.image_count = 0
        self._tp_counts.clear()

    def update(self, preds: List[Dict], gts: List[Dict], image_id: int = None):
        """
        이미지 단위 결과 누적

        Args:
            preds: [{"class_id", "confidence", "bbox": [x1,y1,x2,y2]}, ...]
            gts: [{"class_id", "bbox": [x1,y1,x2,y2]}, ...]
            image_id: 이미지 고유 ID (없으면 자동 할당)
        """
        for record in list(preds) + list(gts):
            cls = record["class_id"]
            box = np.asarray(record["bbox"], dtype=float)
            if cls != int(cls) or not 0 <= cls < self.num_classes:
                raise ValueError("검출 클래스 번호가 모델 범위를 벗어났습니다")
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("검출 박스는 유한한 xyxy 좌표여야 합니다")
        if any(not np.isfinite(p["confidence"]) for p in preds):
            raise ValueError("검출 confidence가 유한하지 않습니다")
        if image_id is None:
            image_id = self.image_count
        self.image_count += 1

        for p in preds:
            self.all_preds.append((
                p["class_id"], p["confidence"],
                *p["bbox"], image_id
            ))
        for g in gts:
            self.all_gts.append((
                g["class_id"], *g["bbox"], image_id
            ))

    def _compute_iou(self, box1, box2) -> float:
        """두 박스의 IoU 계산 (x1,y1,x2,y2 형식)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - inter

        # 정규화된 작은 박스도 같은 비율을 유지하고 면적 0만 별도 처리한다.
        return inter / union if union > 0 else 0.0

    def _compute_ap(self, precisions: List[float], recalls: List[float]) -> float:
        """
        Average Precision (AP) 계산 — 11-point interpolation (VOC 스타일)

        PR 곡선 아래 면적을 11개 recall 지점에서 보간하여 계산
        """
        if not precisions or not recalls:
            return 0.0

        if self.ap_method == "interp101":
            recall = np.concatenate(([0.0], recalls, [1.0]))
            precision = np.concatenate(([1.0], precisions, [0.0]))
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            x = np.linspace(0, 1, 101)
            area = getattr(np, "trapezoid", None)
            if area is None:
                area = np.trapz
            return float(area(np.interp(x, recall, precision), x))
        # 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            # recall ≥ t 인 precision 중 최대값
            prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
            if prec_at_recall:
                ap += max(prec_at_recall)
        return ap / 11.0

    def _compute_class_ap(self, class_id: int, iou_threshold: float) -> Tuple[float, List, List]:
        """특정 클래스 + IoU 임계값에서 AP 계산"""
        # 해당 클래스 예측/정답 필터링
        class_preds = [(conf, *bbox, img_id)
                       for cls, conf, *bbox, img_id in self.all_preds
                       if cls == class_id]
        class_gts = [(*bbox, img_id)
                     for cls, *bbox, img_id in self.all_gts
                     if cls == class_id]

        if not class_gts:
            self._tp_counts[(class_id, iou_threshold)] = 0
            return 0.0, [], []

        # confidence 내림차순 정렬
        class_preds.sort(key=lambda x: x[0], reverse=True)

        # 이미지별 GT 매칭 추적
        gt_matched = {}  # {img_id: set of matched gt indices}
        gt_by_image = {}
        for idx, (*bbox, img_id) in enumerate(class_gts):
            gt_by_image.setdefault(img_id, []).append((idx, bbox))
            gt_matched.setdefault(img_id, set())

        tp_list, fp_list = [], []

        for conf, x1, y1, x2, y2, img_id in class_preds:
            pred_box = (x1, y1, x2, y2)
            best_iou = 0.0
            best_gt_idx = -1

            # 같은 이미지의 GT와 매칭
            for gt_idx, gt_bbox in gt_by_image.get(img_id, []):
                if self.ap_method == "interp101" and gt_idx in gt_matched[img_id]:
                    continue
                iou = self._compute_iou(pred_box, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx not in gt_matched.get(img_id, set()):
                # True Positive
                tp_list.append(1)
                fp_list.append(0)
                gt_matched[img_id].add(best_gt_idx)
            else:
                # False Positive
                tp_list.append(0)
                fp_list.append(1)

        # 누적 TP/FP → Precision/Recall 계산
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        total_gt = len(class_gts)

        precisions = (tp_cumsum / (tp_cumsum + fp_cumsum)).tolist()
        recalls = (tp_cumsum / total_gt).tolist()

        ap = self._compute_ap(precisions, recalls)
        self._tp_counts[(class_id, iou_threshold)] = int(sum(tp_list))
        return ap, precisions, recalls

    def compute(self) -> Dict:
        """
        전체 Detection 메트릭 계산

        Returns:
            {
                "mAP_50": float,       — mAP@IoU=0.5 (PASCAL VOC)
                "mAP_50_95": float,    — mAP@IoU=0.5:0.95 (VOC 11-point 보간)
                "precision": float,    — @IoU=0.5
                "recall": float,       — @IoU=0.5
                "per_class": [{"name", "ap_50", "ap_50_95"}, ...]
            }
        """
        # mAP@0.5
        per_class = []
        aps_50 = []
        aps_50_95 = []
        present = {int(gt[0]) for gt in self.all_gts}

        for c in range(self.num_classes):
            ap_50, precs, recs = self._compute_class_ap(c, iou_threshold=0.5)
            aps_50.append(ap_50)

            # mAP@0.5:0.95 (COCO: 10개 IoU 임계값 평균)
            aps_multi = []
            for iou_thr in np.arange(0.5, 1.0, 0.05):
                ap_at_thr, _, _ = self._compute_class_ap(c, iou_threshold=iou_thr)
                aps_multi.append(ap_at_thr)
            ap_50_95 = float(np.mean(aps_multi)) if aps_multi else 0.0
            aps_50_95.append(ap_50_95)

            per_class.append({
                "name": self.class_names[c] if c < len(self.class_names) else f"Class {c}",
                "ap_50": float(ap_50) if c in present else None,
                "ap_50_95": ap_50_95 if c in present else None,
            })

        mAP_50 = float(np.mean([aps_50[c] for c in present])) if present else 0.0
        mAP_50_95 = float(np.mean([aps_50_95[c] for c in present])) if present else 0.0

        # 전체 Precision/Recall @IoU=0.5 (모든 예측 합산)
        total_tp = 0
        total_gt = len(self.all_gts)
        # 간단히 mAP 계산 시 부산물로 추정
        for c in range(self.num_classes):
            _, precs, recs = self._compute_class_ap(c, 0.5)
            if precs and recs:
                total_tp += self._tp_counts[(c, 0.5)]

        total_preds = len(self.all_preds)
        precision = total_tp / max(total_preds, 1)
        recall = total_tp / max(total_gt, 1)

        return {
            "mAP_50": mAP_50,
            "mAP_50_95": mAP_50_95,
            "ap_interpolation": "voc_11_point",
            "evaluable": bool(present),
            "precision": float(precision),
            "recall": float(recall),
            "per_class": per_class,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Anomaly Detection 평가 지표
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AnomalyMetrics:
    """이미지 단위 이상 점수 평가. 평가 불가를 수치 0점과 구분한다."""

    def __init__(self):
        self.scores = []
        self.labels = []

    def reset(self):
        self.scores.clear()
        self.labels.clear()

    @staticmethod
    def _validate(scores, labels):
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels)
        if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
            raise ValueError("이상 점수와 정답은 길이가 같은 1차원 배열이어야 합니다")
        if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError("이상 점수는 유한값, 정답은 0 또는 1이어야 합니다")
        return scores, labels.astype(np.int64)

    def update(self, scores: np.ndarray, labels: np.ndarray):
        scores, labels = self._validate(scores, labels)
        self.scores.extend(scores.tolist())
        self.labels.extend(labels.tolist())

    @staticmethod
    def _threshold_counts(scores, labels):
        # 같은 점수 전체를 한 임계값으로 묶어 입력 순서 영향을 없앤다.
        order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
        tp = np.cumsum(sorted_labels)[ends]
        fp = ends + 1 - tp
        return sorted_scores[ends], tp, fp

    def _roc(self, scores, labels):
        _, tp, fp = self._threshold_counts(scores, labels)
        return np.r_[0.0, fp / (labels == 0).sum()], np.r_[0.0, tp / labels.sum()]

    def _compute_auroc(self, scores, labels):
        scores, labels = self._validate(scores, labels)
        if scores.size == 0 or np.unique(labels).size < 2:
            return None
        fpr, tpr = self._roc(scores, labels)
        integrate = getattr(np, "trapezoid", None)
        if integrate is None:
            integrate = np.trapz
        return float(integrate(tpr, fpr))

    def _find_optimal_threshold(self, scores, labels):
        """모든 고유 점수에서 F1을 계산한다. 판정 연산은 score >= threshold."""
        scores, labels = self._validate(scores, labels)
        if scores.size == 0 or np.unique(labels).size < 2:
            return None
        thresholds, tp, fp = self._threshold_counts(scores, labels)
        fn = labels.sum() - tp
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        # F1 동률이면 높은 임계값을 선택하는 결정적 정책.
        return float(thresholds[int(np.argmax(f1))])

    def compute(self) -> Dict:
        scores, labels = self._validate(self.scores, self.labels)
        available = scores.size > 0 and np.unique(labels).size == 2
        if not available:
            return {
                "auroc": None, "f1": None, "precision": None, "recall": None,
                "optimal_threshold": None, "evaluable": False,
                "reason": "평가 데이터 없음" if not scores.size else "정상과 불량 라벨이 모두 필요합니다",
                "sample_count": int(scores.size),
                "confusion_matrix": np.zeros((2, 2), dtype=np.int64),
                "roc_curve": {"fpr": [], "tpr": []},
            }
        threshold = self._find_optimal_threshold(scores, labels)
        preds = scores >= threshold
        positive = labels == 1
        tp = int(np.count_nonzero(preds & positive))
        fp = int(np.count_nonzero(preds & ~positive))
        fn = int(np.count_nonzero(~preds & positive))
        tn = int(np.count_nonzero(~preds & ~positive))
        fpr, tpr = self._roc(scores, labels)
        # 곡선의 시작과 끝을 보존하며 표시용 점 수를 제한한다.
        indices = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 200), dtype=int))
        return {
            "auroc": self._compute_auroc(scores, labels),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "optimal_threshold": threshold,
            "threshold_comparator": ">=", "evaluable": True, "reason": "",
            "sample_count": int(scores.size),
            "confusion_matrix": np.array([[tn, fp], [fn, tp]], dtype=np.int64),
            "roc_curve": {"fpr": fpr[indices].tolist(), "tpr": tpr[indices].tolist()},
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  유틸리티: 메트릭 팩토리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_metrics(task: str, num_classes: int = 2,
                   class_names: List[str] = None):
    """
    태스크에 맞는 메트릭 수집기 생성

    Args:
        task: "classify"| "segment"| "detect"| "anomaly"
        num_classes: 클래스 수
        class_names: 클래스 이름 리스트

    Returns:
        해당 태스크의 Metrics 인스턴스
    """
    if task == "classify":
        return ClassificationMetrics(num_classes, class_names)
    elif task == "segment":
        return SegmentationMetrics(num_classes, class_names)
    elif task == "detect":
        return DetectionMetrics(num_classes, class_names)
    elif task == "anomaly":
        return AnomalyMetrics()
    else:
        raise ValueError(f"미지원 태스크: {task}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  태스크별 주요 메트릭 이름 매핑
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# GUI 차트/카드에 표시할 메트릭 목록
TASK_METRIC_NAMES = {
    "classify": {
        "primary": "accuracy",           # Best 판단 기준
        "display": ["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        "labels": {
            "accuracy": "Accuracy",
            "f1_macro": "F1 (macro)",
            "precision_macro": "Precision",
            "recall_macro": "Recall",
        },
    },
    "segment": {
        "primary": "mIoU",
        "display": ["mIoU", "dice_score", "pixel_accuracy"],
        "labels": {
            "mIoU": "mIoU",
            "dice_score": "Dice Score",
            "pixel_accuracy": "Pixel Acc.",
        },
    },
    "obb": {
        "primary": "mAP_50_95",
        "display": ["mAP_50", "mAP_50_95", "precision", "recall"],
        "labels": {"mAP_50": "OBB mAP@0.5", "mAP_50_95": "OBB mAP@.5:.95",
                   "precision": "Precision", "recall": "Recall"},
    },
    "detect": {
        "primary": "mAP_50",
        "display": ["mAP_50", "mAP_50_95", "precision", "recall"],
        "labels": {
            "mAP_50": "mAP@0.5",
            "mAP_50_95": "mAP@.5:.95",
            "precision": "Precision",
            "recall": "Recall",
        },
    },
    "anomaly": {
        "primary": "auroc",
        "display": ["auroc", "f1", "precision", "recall"],
        "labels": {
            "auroc": "AUROC",
            "f1": "F1-Score",
            "precision": "Precision",
            "recall": "Recall",
        },
    },
}
