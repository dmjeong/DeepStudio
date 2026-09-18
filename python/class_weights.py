"""분류 학습에 실제 사용되는 표본의 분포와 손실 가중치 계산."""

import math
import operator


def class_counts(dataset, num_classes):
    """이미지를 디코딩하지 않고 Subset과 캐시된 samples의 클래스 ID를 집계."""
    if num_classes < 1:
        raise ValueError("클래스 수는 1 이상 필요")

    def labels_from(current):
        if hasattr(current, "indices") and hasattr(current, "dataset"):
            base_labels = labels_from(current.dataset)
            return [base_labels[operator.index(i)] for i in current.indices]
        samples = getattr(current, "samples", None)
        if samples is None:
            raise ValueError("학습 표본의 클래스 분포 확인 불가: samples 메타데이터 필요")
        labels = []
        for sample in samples:
            try:
                label = operator.index(sample[1])
            except (IndexError, TypeError) as exc:
                raise ValueError("학습 표본의 클래스 ID는 정수 필요") from exc
            if not 0 <= label < num_classes:
                raise ValueError(f"학습 표본의 클래스 ID 범위 오류: {label}")
            labels.append(label)
        return labels

    labels = labels_from(dataset)
    if not labels:
        raise ValueError("클래스 가중치를 계산할 학습 표본 없음")
    counts = [0] * num_classes
    for label in labels:
        counts[label] += 1
    return counts


def weights_from_counts(counts, mode):
    """표본 기준 평균 가중치 1. 배치 구성에 따라 가중치가 상쇄되지 않음."""
    if mode == "none":
        return None
    if mode not in ("balanced", "sqrt"):
        raise ValueError(f"지원하지 않는 클래스 가중치 모드: {mode}")
    if not counts or any(not math.isfinite(c) or c <= 0 or int(c) != c for c in counts):
        raise ValueError("클래스 가중치 적용에는 모든 클래스의 학습 표본이 1개 이상 필요")
    total = sum(counts)
    weights = [total / (len(counts) * count) for count in counts]
    if mode == "sqrt":
        weights = [math.sqrt(weight) for weight in weights]
    average = sum(count * weight for count, weight in zip(counts, weights)) / total
    return [weight / average for weight in weights]


def describe_class_weights(names, counts, weights, mode):
    """로그에 실제 클래스 순서와 적용값을 함께 표시."""
    values = weights if weights is not None else [1.0] * len(counts)
    details = ", ".join(
        f"{index}:{names[index]}={count}장(w={weight:.4f})"
        for index, (count, weight) in enumerate(zip(counts, values))
    )
    message = f"클래스 가중치({mode}): {details}"
    if mode != "none" and len(set(counts)) == 1:
        message += " / 클래스별 표본 수가 같아 일반 손실과 동일"
    return message
