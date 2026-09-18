"""
CustomCSP Segmentation 학습 스크립트

학습 파이프라인:
┌─────────┐    ┌──────────┐    ┌───────────────┐    ┌──────────┐
│ 이미지+  │───►│  CustomCSP  │───►│ Dice + CE     │───►│ 역전파   │
│ 마스크   │    │ Segment  │    │ Combined Loss │    │ 최적화   │
└─────────┘    └──────────┘    └───────────────┘    └──────────┘
                                                          │
                                                    ┌─────▼──────┐
                                                    │ mIoU 평가  │
                                                    │ 저장/종료   │
                                                    └────────────┘

평가 지표:
- Pixel Accuracy: 전체 픽셀 중 올바르게 분류된 비율
- mIoU (Mean Intersection over Union): 클래스별 IoU의 평균
- Dice Score: 2 * |A∩B| / (|A| + |B|)

사용법:
    python train_segmentation.py \
        --data_root ./data/segment \
        --num_classes 3 \
        --epochs 80 \
        --batch_size 4
"""

import argparse
import math
import os
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import CustomCSP
from dataset import create_segmentation_loaders
from checkpoint import make_checkpoint_metadata


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="CustomCSP Segmentation 학습"
    )
    parser.add_argument("--data_root", type=str, default="./data/segment",
                        help="데이터 루트 (images/masks 하위 포함)")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="세그멘테이션 클래스 수 (배경 포함)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="배치 크기 (세그멘테이션은 메모리 소모가 큼)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="학습률")
    parser.add_argument("--input_size", type=int, default=320,
                        help="입력 이미지 크기")
    parser.add_argument("--in_channels", type=int, default=3, choices=(1, 3),
                        help="입력 채널 수")
    parser.add_argument("--save_dir", type=str, default="./runs/segment",
                        help="결과 저장 경로")
    parser.add_argument("--exp_name", type=str, default="exp1",
                        help="실험 이름")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    return parser.parse_args()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  손실 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DiceLoss(nn.Module):
    """
    Dice 손실 함수
    - 클래스 불균형에 강건 (배경이 지배적인 세그멘테이션에 적합)
    - Dice = 2|A∩B| / (|A| + |B|)
    - Loss = 1 - Dice

    Args:
        smooth: 0으로 나누기 방지 상수
        num_classes: 클래스 수
    """
    def __init__(self, smooth: float = 1.0, num_classes: int = 2,
                 ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) 모델 출력 (로짓)
            target: (B, H, W) 정답 마스크 (클래스 인덱스)
        """
        valid = target != self.ignore_index
        if not valid.any():
            return pred.sum() * 0.0
        # 제외 픽셀은 one-hot 전에 안전한 값으로 치환한 뒤 양쪽에서 제거
        valid_pixels = valid.unsqueeze(1)
        pred_soft = F.softmax(pred, dim=1) * valid_pixels
        safe_target = target.masked_fill(~valid, 0)
        target_onehot = F.one_hot(safe_target, self.num_classes)  # (B,H,W,C)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # (B,C,H,W)
        target_onehot = target_onehot * valid_pixels

        # 클래스별 Dice 계산
        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        # 정답 픽셀이 없는 샘플은 배치 평균에서도 제외
        valid_samples = valid.flatten(1).any(dim=1)
        return 1.0 - dice[valid_samples].mean()


class CombinedSegLoss(nn.Module):
    """
    세그멘테이션 결합 손실: CrossEntropy + Dice

    ┌────────────────────────────────────────┐
    │  Combined = α × CE + (1-α) × Dice     │
    │  α = 0.5 (기본값)                      │
    └────────────────────────────────────────┘

    두 손실을 결합하면:
    - CE: 픽셀 단위 정확도 최적화
    - Dice: 영역 단위 겹침 최적화 (클래스 불균형 대응)
    """
    def __init__(self, num_classes: int = 2, alpha: float = 0.5,
                 ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, pred, target):
        if not (target != self.ignore_index).any():
            # CE의 빈 평균으로 인한 NaN 방지, 역전파 연결 유지
            return pred.sum() * 0.0
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  평가 지표
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                    num_classes: int, ignore_index: int = 255) -> dict:
    """
    세그멘테이션 평가 지표 계산

    Returns:
        pixel_acc: 픽셀 정확도
        mean_iou: Mean IoU
        class_iou: 클래스별 IoU
    """
    # 예측값: argmax
    pred_cls = pred.argmax(dim=1)  # (B, H, W)

    # Pixel Accuracy
    valid = target != ignore_index
    correct = ((pred_cls == target) & valid).sum().item()
    total = valid.sum().item()
    if total == 0:
        raise ValueError("평가 가능한 정답 픽셀 없음: 제외 라벨만 포함된 검증 배치")
    pixel_acc = correct / total

    # 클래스별 IoU 계산
    class_iou = []
    for cls in range(num_classes):
        pred_mask = (pred_cls == cls) & valid
        target_mask = (target == cls) & valid

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union == 0:
            # 해당 클래스가 존재하지 않는 경우 → NaN 제외
            iou = float('nan')
        else:
            iou = intersection / union
        class_iou.append(iou)

    # NaN 제외한 평균 IoU
    valid_ious = [iou for iou in class_iou if iou == iou]  # NaN 제외
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

    return {
        "pixel_acc": pixel_acc,
        "mean_iou": mean_iou,
        "class_iou": class_iou,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  학습 & 검증 루프
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    """한 에폭 학습"""
    if loader is None or len(loader) == 0:
        raise ValueError("학습 배치 없음: 데이터 수와 배치 크기 확인 필요")
    model.train()
    total_loss = 0.0

    for batch_idx, (images, masks) in enumerate(loader):
        # 순전파: (B, num_cls, H, W)
        outputs = model(images)

        # 출력 크기가 마스크와 다르면 리사이즈
        if outputs.shape[2:] != masks.shape[1:]:
            outputs = F.interpolate(
                outputs, size=masks.shape[1:],
                mode='bilinear', align_corners=False
            )

        loss = criterion(outputs, masks)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 5 == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {total_loss/(batch_idx+1):.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, num_classes):
    """검증 수행"""
    if loader is None or len(loader) == 0:
        raise ValueError("검증 데이터 없음: 모델 선택을 위한 검증 데이터 필요")
    model.eval()
    total_loss = 0.0
    all_metrics = {"pixel_acc": 0, "mean_iou": 0}
    count = 0

    for images, masks in loader:
        outputs = model(images)

        # 출력 리사이즈
        if outputs.shape[2:] != masks.shape[1:]:
            outputs = F.interpolate(
                outputs, size=masks.shape[1:],
                mode='bilinear', align_corners=False
            )

        loss = criterion(outputs, masks)
        total_loss += loss.item()

        # 메트릭 계산
        metrics = compute_metrics(outputs, masks, num_classes)
        all_metrics["pixel_acc"] += metrics["pixel_acc"]
        all_metrics["mean_iou"] += metrics["mean_iou"]
        count += 1

    avg_loss = total_loss / len(loader)
    avg_metrics = {
        "pixel_acc": all_metrics["pixel_acc"] / count,
        "mean_iou": all_metrics["mean_iou"] / count,
    }
    return avg_loss, avg_metrics


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("학습 에폭 수는 1 이상 필요")
    if args.num_classes < 1:
        raise ValueError("세그멘테이션 클래스 수는 1 이상 필요")

    # ── 저장 디렉토리 ────────────────────────────────
    save_path = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_path, exist_ok=True)

    print("=" * 60)
    print("🚀 CustomCSP Segmentation 학습 시작")
    print("=" * 60)

    # ── 데이터 로딩 ──────────────────────────────────
    input_size = (args.input_size, args.input_size)
    train_loader, val_loader = create_segmentation_loaders(
        data_root=args.data_root,
        input_size=input_size,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
    )
    if train_loader is None or len(train_loader) == 0:
        raise ValueError("학습 배치 없음: 데이터 수와 배치 크기 확인 필요")
    if val_loader is None or len(val_loader) == 0:
        raise ValueError("검증 데이터 없음: 모델 선택을 위한 검증 데이터 필요")
    checkpoint_metadata = make_checkpoint_metadata(
        task="segment", num_classes=args.num_classes,
        class_names=[str(index) for index in range(args.num_classes)],
        input_size=input_size, in_channels=args.in_channels,
    )

    # ── 모델 생성 ────────────────────────────────────
    model = CustomCSP(
        task="segment",
        in_channels=args.in_channels,
        num_classes=args.num_classes,
    )
    params = model.get_param_count()
    print(f"\n🔧 모델 파라미터: {params['total']:,} "
          f"({params['total_MB']:.1f} MB)")

    # ── 손실 함수 & 최적화기 ──────────────────────────
    criterion = CombinedSegLoss(num_classes=args.num_classes)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=5e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── 학습 루프 ─────────────────────────────────────
    best_miou = float("-inf")
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"학습 설정 요약")
    print(f"{'='*60}")
    print(f"  클래스 수: {args.num_classes}")
    print(f"  에폭: {args.epochs}")
    print(f"  배치: {args.batch_size}")
    print(f"  입력 크기: {input_size}")
    print(f"  손실: CE + Dice")
    print(f"{'='*60}\n")

    history = {"train_loss": [], "val_loss": [],
               "pixel_acc": [], "mean_iou": []}

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # 학습
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )

        # 검증
        val_loss, val_metrics = validate(
            model, val_loader, criterion, args.num_classes
        )
        if not all(math.isfinite(value) for value in
                   (train_loss, val_loss, val_metrics["pixel_acc"],
                    val_metrics["mean_iou"])):
            raise ValueError("학습 또는 검증 결과에 NaN/Inf 발생: 체크포인트 저장 중단")

        scheduler.step()
        elapsed = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        # 이력 기록
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["pixel_acc"].append(val_metrics["pixel_acc"])
        history["mean_iou"].append(val_metrics["mean_iou"])

        # Best 체크
        is_best = val_metrics["mean_iou"] > best_miou
        if is_best:
            best_miou = val_metrics["mean_iou"]
            patience_counter = 0
        else:
            patience_counter += 1

        # best/last 모두 단독으로 복원 가능한 구조와 전처리 정보 저장
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_miou": best_miou,
        }
        state.update(checkpoint_metadata)
        torch.save(state, os.path.join(save_path, "last.pt"))
        if is_best:
            torch.save(state, os.path.join(save_path, "best.pt"))
            print(f"  💾 Best 모델 저장 (mIoU: {best_miou:.4f})")

        # 결과 출력
        print(f"\n📊 Epoch [{epoch+1}/{args.epochs}] "
              f"({elapsed:.1f}s, lr={current_lr:.2e})")
        print(f"  Train  │ Loss: {train_loss:.4f}")
        print(f"  Val    │ Loss: {val_loss:.4f} "
              f"│ PixAcc: {val_metrics['pixel_acc']:.4f} "
              f"│ mIoU: {val_metrics['mean_iou']:.4f} "
              f"{'⭐ BEST' if is_best else ''}")

        # 조기 종료
        if patience_counter >= args.patience:
            print(f"\n⏹ 조기 종료 (patience={args.patience})")
            break

    # ── 학습 완료 ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ Segmentation 학습 완료!")
    print(f"  최고 mIoU: {best_miou:.4f}")
    print(f"  모델 저장: {save_path}/best.pt")
    print(f"{'='*60}")

    with open(os.path.join(save_path, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
