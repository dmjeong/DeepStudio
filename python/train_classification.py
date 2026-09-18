"""
CustomCSP Classification 학습 스크립트

학습 파이프라인:
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 데이터   │───►│  모델    │───►│ 손실계산 │───►│ 역전파   │
│ 로딩    │    │ 순전파   │    │ CrossEnt │    │ 최적화   │
└─────────┘    └──────────┘    └──────────┘    └──────────┘
                                                    │
                                              ┌─────▼──────┐
                                              │ 검증/저장   │
                                              │ 조기종료    │
                                              └────────────┘

사용법:
    python train_classification.py \
        --data_root ./data/classify \
        --num_classes 5 \
        --epochs 50 \
        --batch_size 8
"""

import argparse
import math
import os
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import CustomCSP
from dataset import create_classification_loaders
from checkpoint import make_checkpoint_metadata


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="CustomCSP Classification 학습"
    )
    parser.add_argument("--data_root", type=str, default="./data/classify",
                        help="학습 데이터 루트 경로")
    parser.add_argument("--num_classes", type=int, default=0,
                        help="클래스 수 (0이면 자동 감지)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="학습률")
    parser.add_argument("--input_size", type=int, default=224,
                        help="입력 이미지 크기 (정사각형)")
    parser.add_argument("--in_channels", type=int, default=3, choices=(1, 3),
                        help="입력 채널 수 (1: 그레이, 3: RGB)")
    parser.add_argument("--save_dir", type=str, default="./runs/classify",
                        help="결과 저장 경로")
    parser.add_argument("--exp_name", type=str, default="exp1",
                        help="실험 이름")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--resume", type=str, default=None,
                        help="체크포인트 경로 (학습 재개)")
    return parser.parse_args()


class EarlyStopping:
    """
    조기 종료 (Early Stopping)
    - 검증 손실이 patience 에폭 동안 개선되지 않으면 학습 중단
    """
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            # 개선됨 → 카운터 리셋
            self.best_loss = val_loss
            self.counter = 0
        else:
            # 개선 안 됨 → 카운터 증가
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(model, loader, criterion, optimizer, epoch, log_interval=10):
    """
    한 에폭 학습

    Returns:
        avg_loss: 평균 학습 손실
        accuracy: 학습 정확도
    """
    if loader is None or len(loader) == 0:
        raise ValueError("학습 배치 없음: 데이터 수와 배치 크기 확인 필요")
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 역전파 & 가중치 갱신
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 통계 업데이트
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 로그 출력
        if (batch_idx + 1) % log_interval == 0:
            avg = total_loss / (batch_idx + 1)
            acc = 100. * correct / total
            print(f"  [Epoch {epoch}] Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {avg:.4f} | Acc: {acc:.1f}%")

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion):
    """
    검증 수행

    Returns:
        avg_loss: 평균 검증 손실
        accuracy: 검증 정확도
    """
    if loader is None or len(loader) == 0:
        raise ValueError("검증 데이터 없음: 모델 선택을 위한 검증 데이터 필요")
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc,
                    class_names, save_path, is_best=False, *,
                    input_size=(224, 224), model_config=None):
    """체크포인트 저장"""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_acc": best_acc,
    }
    state.update(make_checkpoint_metadata(
        task="classify", num_classes=model.num_classes,
        class_names=class_names, input_size=input_size,
        in_channels=model.in_channels, model_config=model_config,
    ))
    # 최신 체크포인트 저장
    torch.save(state, os.path.join(save_path, "last.pt"))
    # 최적 모델 별도 저장
    if is_best:
        torch.save(state, os.path.join(save_path, "best.pt"))
        print(f"  💾 Best 모델 저장 (Acc: {best_acc:.1f}%)")


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("학습 에폭 수는 1 이상 필요")

    # ── 저장 디렉토리 생성 ──────────────────────────
    save_path = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_path, exist_ok=True)

    print("=" * 60)
    print("🚀 CustomCSP Classification 학습 시작")
    print("=" * 60)

    # ── 데이터 로딩 ──────────────────────────────────
    input_size = (args.input_size, args.input_size)
    train_loader, val_loader, class_names = create_classification_loaders(
        data_root=args.data_root,
        input_size=input_size,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
    )
    if train_loader is None or len(train_loader) == 0:
        raise ValueError("학습 배치 없음: 데이터 수와 배치 크기 확인 필요")
    if val_loader is None or len(val_loader) == 0:
        raise ValueError("검증 데이터 없음: 모델 선택을 위한 검증 데이터 필요")

    # 클래스 수 자동 감지
    num_classes = args.num_classes if args.num_classes > 0 else len(class_names)
    if num_classes < 1 or num_classes != len(class_names):
        raise ValueError(
            f"클래스 수 불일치: 설정 {num_classes}, 데이터 {len(class_names)}"
        )
    print(f"\n📋 감지된 클래스: {class_names} (총 {num_classes}개)")

    # 클래스 매핑 저장 (추론 시 필요)
    class_map = {i: name for i, name in enumerate(class_names)}
    with open(os.path.join(save_path, "class_names.json"), "w",
              encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    # ── 모델 생성 ────────────────────────────────────
    model = CustomCSP(
        task="classify",
        in_channels=args.in_channels,
        num_classes=num_classes,
    )
    params = model.get_param_count()
    print(f"\n🔧 모델 파라미터: {params['total']:,} "
          f"({params['total_MB']:.1f} MB)")

    # ── 손실 함수 & 최적화기 ──────────────────────────
    # Label Smoothing 적용 (과적합 방지)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-4
    )
    # Cosine Annealing 학습률 스케줄러
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── 학습 재개 (선택) ──────────────────────────────
    start_epoch = 0
    best_acc = float("-inf")
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        print(f"  📂 체크포인트 복원: epoch {start_epoch}, best_acc {best_acc:.1f}%")
    if start_epoch >= args.epochs:
        raise ValueError("재개할 학습 에폭 없음: --epochs 증가 필요")

    # ── 학습 루프 ─────────────────────────────────────
    early_stop = EarlyStopping(patience=args.patience)

    print(f"\n{'='*60}")
    print(f"학습 설정 요약")
    print(f"{'='*60}")
    print(f"  에폭: {args.epochs}")
    print(f"  배치: {args.batch_size}")
    print(f"  학습률: {args.lr}")
    print(f"  입력 크기: {input_size}")
    print(f"  저장 경로: {save_path}")
    print(f"{'='*60}\n")

    # 학습 이력 기록
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # 학습
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )

        # 검증
        val_loss, val_acc = validate(model, val_loader, criterion)
        if not all(math.isfinite(value) for value in
                   (train_loss, train_acc, val_loss, val_acc)):
            raise ValueError("학습 또는 검증 결과에 NaN/Inf 발생: 체크포인트 저장 중단")

        # 학습률 갱신
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 에폭 소요 시간
        elapsed = time.time() - epoch_start

        # 이력 기록
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 결과 출력
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        print(f"\n📊 Epoch [{epoch+1}/{args.epochs}] "
              f"({elapsed:.1f}s, lr={current_lr:.2e})")
        print(f"  Train  │ Loss: {train_loss:.4f} │ Acc: {train_acc:.1f}%")
        print(f"  Val    │ Loss: {val_loss:.4f} │ Acc: {val_acc:.1f}% "
              f"{'⭐ BEST' if is_best else ''}")

        # 체크포인트 저장
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_acc,
            class_names, save_path, is_best, input_size=input_size
        )

        # 조기 종료 체크
        if early_stop(val_loss):
            print(f"\n⏹ 조기 종료 (patience={args.patience})")
            break

    # ── 학습 완료 ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ 학습 완료!")
    print(f"  최고 검증 정확도: {best_acc:.1f}%")
    print(f"  모델 저장 위치: {save_path}/best.pt")
    print(f"{'='*60}")

    # 학습 이력 저장
    with open(os.path.join(save_path, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return save_path


if __name__ == "__main__":
    main()
