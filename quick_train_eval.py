"""
CustomCSP — 빠른 학습 + 실제 평가 데모

각 태스크별 실제 데이터셋으로 소규모 학습 후 추론 + Grad-CAM 시각화

┌──────────────────────────────────────────────────────────────────┐
│  태스크        │ 데이터셋           │ 학습 전략                    │
│──────────────│──────────────────│──────────────────────────────│
│  Classification│ CIFAR-10 (50K)   │ 5 epochs, lr=1e-3           │
│  Segmentation │ VOC 2012 (seg)   │ 5 epochs, lr=1e-3           │
│  Anomaly      │ CIFAR-10 정상/이상│ 5 epochs, reconstruction    │
│  Detection    │ VOC-style 합성    │ 5 epochs, lr=1e-3           │
└──────────────────────────────────────────────────────────────────┘

사용법:
    python quick_train_eval.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image, ImageDraw
import torchvision
import torchvision.transforms as T

# 프로젝트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "gui"))

from model import CustomCSP
from core.gradcam import GradCAM

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "demo_results_trained")
DATA_DIR = os.path.join(PROJECT_ROOT, "data_cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

DEVICE = "cpu"

# CIFAR-10 클래스 이름 (한국어)
CIFAR10_CLASSES = [
    "비행기", "자동차", "새", "고양이", "사슴",
    "강아지", "개구리", "말", "배", "트럭"
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. Classification — CIFAR-10
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_classification():
    """
    CIFAR-10으로 Classification 모델 빠른 학습

    학습 흐름:
    CIFAR-10 (32×32) → Resize(224) → CustomCSP(classify) → CrossEntropy
    """
    print("\n" + "=" * 60)
    print("🔍 [1/4] Classification — CIFAR-10 학습")
    print("=" * 60)

    # ── 데이터 로딩 ──
    transform_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("  📥 CIFAR-10 다운로드 중...")
    train_full = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train
    )
    val_full = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_val
    )

    # 속도를 위해 서브셋 사용 (학습 2000장, 검증 500장)
    train_subset = Subset(train_full, range(0, 2000))
    val_subset = Subset(val_full, range(0, 500))

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)

    print(f"  📊 학습: {len(train_subset)}장, 검증: {len(val_subset)}장")

    # ── 모델 생성 ──
    model = CustomCSP(task="classify", num_classes=10, in_channels=3)
    model.to(DEVICE)
    params = model.get_param_count()
    print(f"  🔧 파라미터: {params['total']:,} ({params['total_MB']:.1f} MB)")

    # ── 학습 ──
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_acc = 0
    for epoch in range(10):
        # 학습
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_loss = total_loss / len(train_loader)

        # 검증
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        best_acc = max(best_acc, val_acc)
        scheduler.step()

        print(f"  Epoch {epoch+1:2d}/10 | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% "
              f"{'⭐' if val_acc >= best_acc else ''}")

    print(f"  ✅ 최종 검증 정확도: {best_acc:.1f}%")

    return model, val_full


def eval_classification(model, val_dataset):
    """Classification 추론 + Grad-CAM 시각화 (5개 이미지)"""
    print("\n  📸 Classification 추론 + Grad-CAM...")

    # 원본(비정규화) 이미지용 변환
    transform_raw = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    transform_norm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model.eval()
    gradcam = GradCAM(model)
    results = []

    # 다양한 클래스 이미지 선택 (각 클래스에서 1개씩 5개)
    selected = []
    seen_classes = set()
    for idx in range(len(val_dataset)):
        _, label = val_dataset[idx]
        if label not in seen_classes and len(selected) < 5:
            selected.append(idx)
            seen_classes.add(label)
        if len(selected) >= 5:
            break

    for idx in selected:
        # 원본 PIL 이미지
        pil_img = val_dataset.data[idx]  # numpy (32, 32, 3)
        pil_img_pil = Image.fromarray(pil_img)
        true_label = val_dataset.targets[idx]

        # 원본 이미지 (비정규화, 224×224)
        raw_tensor = transform_raw(pil_img_pil)  # (3, 224, 224)
        raw_np = (raw_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # 정규화된 입력
        input_tensor = transform_norm(pil_img_pil).unsqueeze(0).to(DEVICE)

        # 추론
        with torch.no_grad():
            output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(1).item()
        pred_prob = probs[0, pred_class].item()

        # Top-3
        top3_probs, top3_idx = probs[0].topk(3)

        # Grad-CAM
        heatmap, overlay = gradcam.generate(input_tensor, raw_np, alpha=0.5)

        result = {
            "original": raw_np,
            "overlay": overlay,
            "true_label": true_label,
            "true_name": CIFAR10_CLASSES[true_label],
            "pred_class": pred_class,
            "pred_name": CIFAR10_CLASSES[pred_class],
            "pred_prob": pred_prob,
            "correct": pred_class == true_label,
            "top3": [(CIFAR10_CLASSES[i.item()], p.item()) for i, p in zip(top3_idx, top3_probs)],
        }
        results.append(result)

        status = "✅" if result["correct"] else "❌"
        print(f"    {status} GT={result['true_name']:4s} → 예측={result['pred_name']:4s} "
              f"({pred_prob:.1%}) | Top3: {', '.join(f'{n}({p:.0%})' for n, p in result['top3'])}")

    gradcam.release()
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. Segmentation — VOC 2012
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VOCSegDataset(Dataset):
    """
    Pascal VOC 2012 Segmentation 데이터셋 래퍼

    VOC는 21클래스 (배경 포함) → 간단히 5클래스로 리매핑
    0: 배경, 1: 사람(15), 2: 자동차(7,6,14), 3: 동물(8,12,13,17), 4: 기타
    """
    # VOC 원본 21클래스 → 5클래스 매핑
    REMAP = {
        0: 0,   # 배경 → 배경
        15: 1,  # person → 사람
        7: 2, 6: 2, 14: 2,  # car, bus, motorbike → 자동차
        8: 3, 12: 3, 13: 3, 17: 3,  # cat, dog, horse, sheep → 동물
        # 나머지 → 4 (기타)
    }
    CLASS_NAMES = ["배경", "사람", "자동차", "동물", "기타"]

    def __init__(self, voc_dataset, input_size=320):
        self.voc = voc_dataset
        self.input_size = input_size
        self.img_transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, mask = self.voc[idx]

        # 이미지 변환
        img_tensor = self.img_transform(img)

        # 마스크 리사이즈 + 리매핑
        mask_np = np.array(mask.resize(
            (self.input_size, self.input_size), Image.NEAREST
        ))
        # 255 (무시 영역) → 0 (배경)
        mask_np[mask_np == 255] = 0
        # 5클래스로 리매핑
        remapped = np.full_like(mask_np, 4)  # 기본값: 기타
        for src, dst in self.REMAP.items():
            remapped[mask_np == src] = dst

        mask_tensor = torch.from_numpy(remapped).long()
        return img_tensor, mask_tensor

    def get_raw_image(self, idx):
        """원본 RGB numpy (H, W, 3)"""
        img, _ = self.voc[idx]
        img = img.resize((self.input_size, self.input_size))
        return np.array(img)


def train_segmentation():
    """Pascal VOC 2012로 Segmentation 모델 학습"""
    print("\n" + "=" * 60)
    print("🎨 [2/4] Segmentation — Pascal VOC 2012 학습")
    print("=" * 60)

    print("  📥 VOC 2012 다운로드 중...")
    voc_train = torchvision.datasets.VOCSegmentation(
        root=DATA_DIR, year="2012", image_set="train", download=True
    )
    voc_val = torchvision.datasets.VOCSegmentation(
        root=DATA_DIR, year="2012", image_set="val", download=True
    )

    # 서브셋 (학습 500장, 검증 100장) — 속도 우선
    train_ds = VOCSegDataset(Subset(voc_train, range(0, min(500, len(voc_train)))))
    val_ds = VOCSegDataset(Subset(voc_val, range(0, min(100, len(voc_val)))))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    print(f"  📊 학습: {len(train_ds)}장, 검증: {len(val_ds)}장")

    # ── 모델 ──
    model = CustomCSP(task="segment", num_classes=5, in_channels=3)
    model.to(DEVICE)
    params = model.get_param_count()
    print(f"  🔧 파라미터: {params['total']:,} ({params['total_MB']:.1f} MB)")

    # ── 학습 ──
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)  # (B, C, H, W)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # mIoU 계산
        model.eval()
        ious = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1)  # (B, H, W)
                for c in range(5):
                    pred_c = (preds == c)
                    true_c = (masks == c)
                    inter = (pred_c & true_c).float().sum()
                    union = (pred_c | true_c).float().sum()
                    if union > 0:
                        ious.append((inter / union).item())

        miou = np.mean(ious) if ious else 0
        scheduler.step()

        print(f"  Epoch {epoch+1:2d}/10 | Loss: {train_loss:.4f} | mIoU: {miou:.3f}")

    print(f"  ✅ 학습 완료!")
    return model, voc_val


def eval_segmentation(model, voc_val):
    """Segmentation 추론 + Grad-CAM 시각화"""
    print("\n  📸 Segmentation 추론 + Grad-CAM...")

    seg_ds = VOCSegDataset(Subset(voc_val, range(0, min(50, len(voc_val)))))
    model.eval()
    gradcam = GradCAM(model)

    # 컬러 팔레트
    palette = np.array([
        [0, 0, 0],        # 배경
        [192, 64, 64],    # 사람 (빨강계)
        [64, 64, 192],    # 자동차 (파랑계)
        [64, 192, 64],    # 동물 (초록계)
        [192, 192, 64],   # 기타 (노랑계)
    ], dtype=np.uint8)

    results = []
    for idx in range(5):
        raw_img = seg_ds.get_raw_image(idx)
        input_tensor = seg_ds[idx][0].unsqueeze(0).to(DEVICE)
        gt_mask = seg_ds[idx][1].numpy()

        # 추론
        with torch.no_grad():
            output = model(input_tensor)
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

        # 컬러 마스크
        h, w = pred_mask.shape
        color_pred = np.zeros((h, w, 3), dtype=np.uint8)
        color_gt = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(5):
            color_pred[pred_mask == c] = palette[c]
            color_gt[gt_mask == c] = palette[c]

        # Grad-CAM
        heatmap, overlay = gradcam.generate(input_tensor, raw_img, alpha=0.5)

        # mIoU
        ious = []
        for c in range(5):
            inter = ((pred_mask == c) & (gt_mask == c)).sum()
            union = ((pred_mask == c) | (gt_mask == c)).sum()
            if union > 0:
                ious.append(inter / union)
        miou = np.mean(ious) if ious else 0

        results.append({
            "original": raw_img,
            "gt_mask": color_gt,
            "pred_mask": color_pred,
            "overlay": overlay,
            "miou": miou,
        })
        print(f"    이미지 {idx+1}: mIoU={miou:.3f}")

    gradcam.release()
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. Anomaly — CIFAR-10 기반 (정상: 비행기, 이상: 나머지)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AnomalyDataset(Dataset):
    """
    CIFAR-10 기반 이상 탐지 데이터셋

    정상: 클래스 0 (비행기) — 재구성 학습
    이상: 다른 클래스 — 재구성 오차 높을 것으로 기대
    """
    def __init__(self, cifar_dataset, normal_class=0):
        self.data = []
        self.labels = []  # 0: 정상, 1: 이상
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        for idx in range(len(cifar_dataset)):
            img, label = cifar_dataset[idx]
            self.data.append(img)
            self.labels.append(0 if label == normal_class else 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return self.transform(img), self.labels[idx]

    def get_raw_image(self, idx):
        img = self.data[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.resize((224, 224))
        return np.array(img)


def train_anomaly():
    """Anomaly Detection 학습 (CIFAR-10, 정상=비행기)"""
    print("\n" + "=" * 60)
    print("🔬 [3/4] Anomaly Detection — CIFAR-10 (Normal=비행기) 학습")
    print("=" * 60)

    # CIFAR-10 로딩 (이미 캐시됨)
    cifar_train = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=False)
    cifar_val = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=False)

    # 학습용: 정상(비행기) 이미지만
    normal_indices = [i for i in range(len(cifar_train)) if cifar_train.targets[i] == 0]
    train_ds = AnomalyDataset(Subset(cifar_train, normal_indices[:500]))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

    print(f"  📊 정상 학습: {len(train_ds)}장 (비행기만)")

    # ── 모델 ──
    model = CustomCSP(task="anomaly", num_classes=3, in_channels=3)
    model.to(DEVICE)
    params = model.get_param_count()
    print(f"  🔧 파라미터: {params['total']:,} ({params['total_MB']:.1f} MB)")

    # ── 학습 (재구성 손실) ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(DEVICE)
            recon = model(images)
            loss = F.mse_loss(recon, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1:2d}/10 | Recon Loss: {avg_loss:.4f}")

    print(f"  ✅ 학습 완료!")
    return model, cifar_val


def eval_anomaly(model, cifar_val):
    """Anomaly 추론 + Grad-CAM 시각화"""
    print("\n  📸 Anomaly 추론 + Grad-CAM...")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_raw = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    model.eval()
    gradcam = GradCAM(model)

    # 정상 2장(비행기) + 이상 3장(고양이, 자동차, 개구리)
    test_indices = []
    anomaly_classes = {0: 2, 3: 1, 1: 1, 6: 1}  # class_id: count
    class_counts = {}
    for idx in range(len(cifar_val)):
        label = cifar_val.targets[idx]
        if label in anomaly_classes:
            count = class_counts.get(label, 0)
            if count < anomaly_classes[label]:
                test_indices.append(idx)
                class_counts[label] = count + 1
        if len(test_indices) >= 5:
            break

    results = []
    for idx in test_indices:
        raw_img = np.array(Image.fromarray(cifar_val.data[idx]).resize((224, 224)))
        pil_img = Image.fromarray(cifar_val.data[idx])
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        true_label = cifar_val.targets[idx]
        is_normal = (true_label == 0)

        # 추론
        with torch.no_grad():
            recon = model(input_tensor)

        # 재구성 오차
        error = (input_tensor - recon).pow(2).mean(dim=1).squeeze().cpu().numpy()
        anomaly_score = float(error.mean())

        # 오차 맵 시각화
        e_norm = (error - error.min()) / (error.max() - error.min() + 1e-8)
        error_resized = GradCAM._resize_cam(e_norm, (224, 224))
        error_vis = GradCAM._jet_colormap(error_resized)

        # Grad-CAM
        heatmap, overlay = gradcam.generate(input_tensor, raw_img, alpha=0.5)

        results.append({
            "original": raw_img,
            "error_map": error_vis,
            "overlay": overlay,
            "anomaly_score": anomaly_score,
            "is_normal": is_normal,
            "true_name": CIFAR10_CLASSES[true_label],
        })

        status = "✅ 정상" if is_normal else "⚠️ 이상"
        print(f"    {CIFAR10_CLASSES[true_label]:5s} → 이상점수={anomaly_score:.4f} {status}")

    gradcam.release()
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. Detection — VOC 간이 학습
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_detection():
    """Detection은 학습 시간이 길어서 짧은 학습만 수행"""
    print("\n" + "=" * 60)
    print("📦 [4/4] Detection — 간이 학습 (3 epochs)")
    print("=" * 60)

    model = CustomCSP(task="detect", num_classes=10, in_channels=3)
    model.to(DEVICE)
    params = model.get_param_count()
    print(f"  🔧 파라미터: {params['total']:,} ({params['total_MB']:.1f} MB)")

    # CIFAR 이미지를 detection 입력으로 사용 (간이)
    transform = T.Compose([
        T.Resize((416, 416)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    cifar_val = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=False)

    # Detection은 복잡한 GT 포맷 필요 → 여기서는 간이 학습 (Objectness만)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)

    model.train()
    for epoch in range(3):
        total_loss = 0
        for i in range(0, 100, 10):  # 10 batches
            batch = []
            for j in range(i, min(i+10, 100)):
                pil_img = Image.fromarray(cifar_val.data[j])
                batch.append(transform(pil_img))
            images = torch.stack(batch).to(DEVICE)

            output = model(images)  # (B, N, 15)
            # 간이 손실: objectness를 낮추는 방향 (배경 대부분)
            obj_loss = torch.sigmoid(output[:, :, 4]).mean()
            loss = obj_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch+1}/3 | Loss: {total_loss/10:.4f}")

    print(f"  ✅ 간이 학습 완료 (Detection은 COCO 규모 학습 필요)")
    return model, cifar_val


def eval_detection(model, cifar_val):
    """Detection 추론 + Grad-CAM 시각화"""
    print("\n  📸 Detection 추론 + Grad-CAM...")

    transform = T.Compose([
        T.Resize((416, 416)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model.eval()
    gradcam = GradCAM(model)
    results = []

    for idx in range(5):
        raw_img = np.array(Image.fromarray(cifar_val.data[idx]).resize((416, 416)))
        pil_img = Image.fromarray(cifar_val.data[idx])
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        # 추론
        with torch.no_grad():
            output = model(input_tensor)  # (1, N, 15)

        preds = output[0]
        obj_scores = torch.sigmoid(preds[:, 4])
        cls_scores = torch.sigmoid(preds[:, 5:])
        max_cls, max_ids = cls_scores.max(dim=1)
        conf = obj_scores * max_cls
        top_k = min(3, len(conf))
        top_conf, top_idx = conf.topk(top_k)

        # 박스 그리기
        det_img = raw_img.copy()
        pil_det = Image.fromarray(det_img)
        draw = ImageDraw.Draw(pil_det)
        colors = [(255,0,0), (0,255,0), (0,0,255)]

        for j in range(top_k):
            i_idx = top_idx[j].item()
            cx, cy, w, h = preds[i_idx, :4].tolist()
            # 좌표 → 이미지 비율
            x1 = max(0, int((cx - w/2) * 416 / 416))
            y1 = max(0, int((cy - h/2) * 416 / 416))
            x2 = min(415, int((cx + w/2) * 416 / 416))
            y2 = min(415, int((cy + h/2) * 416 / 416))
            c = max_ids[i_idx].item()
            cls_name = CIFAR10_CLASSES[c] if c < 10 else f"cls_{c}"
            draw.rectangle([x1, y1, x2, y2], outline=colors[j % 3], width=2)
            draw.text((x1, max(0, y1-12)), f"{cls_name} {top_conf[j]:.2f}",
                      fill=colors[j % 3])

        det_array = np.array(pil_det)

        # Grad-CAM
        heatmap, overlay = gradcam.generate(input_tensor, raw_img, alpha=0.5)

        results.append({
            "original": raw_img,
            "detection": det_array,
            "overlay": overlay,
            "true_name": CIFAR10_CLASSES[cifar_val.targets[idx]],
        })
        print(f"    이미지 {idx+1} ({CIFAR10_CLASSES[cifar_val.targets[idx]]}): "
              f"top conf={top_conf[0]:.3f}")

    gradcam.release()
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  요약 이미지 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_final_summary(cls_res, seg_res, det_res, ano_res, output_dir):
    """4개 태스크 최종 요약 그리드 생성"""
    print("\n📊 최종 요약 이미지 생성 중...")

    CELL = 180
    PAD = 6
    COLS = 5  # 이미지 5장
    LABEL_W = 160
    HEADER_H = 50

    # 4개 태스크, 각 2행 (원본 + 결과/GradCAM)
    task_data = [
        ("🔍 Classification (CIFAR-10)", cls_res, ["original", "overlay"]),
        ("🎨 Segmentation (VOC 2012)", seg_res, ["original", "pred_mask", "overlay"]),
        ("📦 Detection (간이 학습)", det_res, ["original", "detection", "overlay"]),
        ("🔬 Anomaly (CIFAR-10)", ano_res, ["original", "error_map", "overlay"]),
    ]

    total_rows = sum(len(t[2]) for t in task_data)
    W = LABEL_W + COLS * (CELL + PAD) + PAD
    H = HEADER_H + total_rows * (CELL + 20 + PAD) + len(task_data) * 30 + PAD * 4

    canvas = Image.new("RGB", (W, H), (25, 28, 38))
    draw = ImageDraw.Draw(canvas)

    # 헤더
    draw.rectangle([0, 0, W, HEADER_H], fill=(35, 38, 50))
    draw.text((W // 2 - 250, 12),
              "Deep Vision Studio — Trained Model Evaluation Results",
              fill=(240, 220, 100))

    y = HEADER_H + PAD

    row_labels_map = {
        "original": "Original",
        "overlay": "Grad-CAM",
        "pred_mask": "Seg Mask",
        "detection": "Detection",
        "error_map": "Error Map",
        "gt_mask": "Ground Truth",
    }

    for task_name, results, vis_keys in task_data:
        # 태스크 제목
        draw.rectangle([0, y, W, y + 25], fill=(45, 48, 60))
        draw.text((10, y + 4), task_name, fill=(255, 200, 100))
        y += 30

        for row_key in vis_keys:
            label = row_labels_map.get(row_key, row_key)
            draw.text((8, y + CELL // 2 - 6), label, fill=(160, 165, 190))

            for col, res in enumerate(results[:COLS]):
                img_data = res.get(row_key, res["original"])
                pil_cell = Image.fromarray(img_data).resize((CELL, CELL), Image.BILINEAR)
                cx = LABEL_W + col * (CELL + PAD) + PAD
                canvas.paste(pil_cell, (cx, y))

                # 어노테이션
                if row_key == "original":
                    if "true_name" in res:
                        txt = res["true_name"]
                        if "pred_name" in res:
                            correct = res.get("correct", False)
                            mark = "✓" if correct else "✗"
                            txt = f"{mark} {res['pred_name']}({res.get('pred_prob', 0):.0%})"
                        elif "anomaly_score" in res:
                            status = "정상" if res["is_normal"] else "이상"
                            txt = f"{res['true_name']} ({status})"
                        draw.text((cx, y + CELL + 2), txt, fill=(200, 200, 220))

            y += CELL + 20 + PAD

    path = os.path.join(output_dir, "summary_trained.png")
    canvas.save(path, quality=95)
    print(f"  💾 저장: {path} ({canvas.size[0]}×{canvas.size[1]})")
    return path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  메인 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CustomCSP — 실제 데이터셋 학습 + 평가 데모                  ║")
    print("║  CIFAR-10, Pascal VOC 2012 활용                          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    start_time = time.time()

    # 1. Classification
    cls_model, cls_val = train_classification()
    cls_results = eval_classification(cls_model, cls_val)

    # 2. Segmentation
    seg_model, seg_val = train_segmentation()
    seg_results = eval_segmentation(seg_model, seg_val)

    # 3. Anomaly
    ano_model, ano_val = train_anomaly()
    ano_results = eval_anomaly(ano_model, ano_val)

    # 4. Detection
    det_model, det_val = train_detection()
    det_results = eval_detection(det_model, det_val)

    # 5. 개별 결과 저장
    for task, results in [("classify", cls_results), ("segment", seg_results),
                          ("detect", det_results), ("anomaly", ano_results)]:
        task_dir = os.path.join(OUTPUT_DIR, task)
        os.makedirs(task_dir, exist_ok=True)
        for i, res in enumerate(results):
            Image.fromarray(res["original"]).save(os.path.join(task_dir, f"img_{i+1:02d}_original.png"))
            Image.fromarray(res["overlay"]).save(os.path.join(task_dir, f"img_{i+1:02d}_gradcam.png"))

    # 6. 요약 이미지
    summary_path = create_final_summary(
        cls_results, seg_results, det_results, ano_results, OUTPUT_DIR
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"🎉 전체 완료! ({elapsed:.0f}초)")
    print(f"   결과: {OUTPUT_DIR}/")
    print(f"   요약: {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
