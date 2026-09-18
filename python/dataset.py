"""
CustomCSP 데이터셋 로더
- Classification: 폴더 기반 이미지 분류 (ImageFolder 형식)
- Segmentation: 이미지 + 마스크 쌍 로딩
- 공통 전처리 & 데이터 증강 파이프라인

디렉토리 구조:
┌───────────────────────────────────────────────────────────┐
│  Classification 데이터 구조        Segmentation 데이터 구조  │
├───────────────────────────────────────────────────────────┤
│  data/                            data/                    │
│  ├── train/                       ├── images/              │
│  │   ├── class_A/                 │   ├── train/           │
│  │   │   ├── img001.jpg           │   │   ├── img001.jpg   │
│  │   │   └── img002.jpg           │   │   └── img002.jpg   │
│  │   ├── class_B/                 │   └── val/             │
│  │   │   └── ...                  │       └── ...          │
│  │   └── class_C/                 ├── masks/               │
│  │       └── ...                  │   ├── train/           │
│  └── val/                         │   │   ├── img001.png   │
│      ├── class_A/                 │   │   └── img002.png   │
│      └── ...                      │   └── val/             │
│                                   │       └── ...          │
└───────────────────────────────────────────────────────────┘
"""

import os
import random
from typing import Tuple, List, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image
from opencv_preprocess import OpenCVResize, read_image, resize


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  데이터 증강 파이프라인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 채널별 정규화 상수 ──────────────────────────────
# 3ch (RGB): ImageNet 표준 정규화 값
IMAGENET_MEAN_3CH = [0.485, 0.456, 0.406]
IMAGENET_STD_3CH = [0.229, 0.224, 0.225]

# 1ch: 프로젝트에서 사용해 온 정규화 상수다.
# 실제 데이터의 통계 추정값이나 공식 ImageNet grayscale 통계를 뜻하지 않는다.
IMAGENET_MEAN_1CH = [0.449]
IMAGENET_STD_1CH = [0.226]


def get_normalize_params(in_channels: int = 3):
    """
    채널 수에 맞는 정규화 mean/std 반환

    Args:
        in_channels: 1 (Grayscale) 또는 3 (RGB)

    Returns:
        (mean, std) — 각각 리스트
    """
    if in_channels == 1:
        return IMAGENET_MEAN_1CH, IMAGENET_STD_1CH
    return IMAGENET_MEAN_3CH, IMAGENET_STD_3CH


def get_classification_transforms(
    input_size: Tuple[int, int] = (224, 224),
    is_train: bool = True,
    in_channels: int = 3,
    flip_prob: float = 0.5,
    rotation: float = 15.0,
    color_jitter: float = 0.2
) -> T.Compose:
    """
    Classification 전용 데이터 증강 파이프라인

    학습 시:
        OpenCV Resize → ToTensor → RandomHFlip → RandomRotation → [ColorJitter] → Normalize
    검증 시:
        Resize → ToTensor → Normalize

    Note: 1ch (Grayscale) 모드에서는 ColorJitter의 saturation/hue 제거
          (그레이스케일에는 색상 정보가 없음)
    """
    # 채널 수에 맞는 정규화 값 선택
    mean, std = get_normalize_params(in_channels)

    if is_train:
        augmentations = [
            OpenCVResize(input_size),
            T.ToTensor(),
            T.RandomHorizontalFlip(p=flip_prob),
            T.RandomRotation(degrees=rotation),
        ]

        # 1ch: brightness/contrast만 적용 (saturation/hue는 무의미)
        if in_channels == 1:
            augmentations.append(T.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
            ))
        else:
            augmentations.append(T.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter / 2  # hue는 범위가 작음
            ))

        augmentations += [
            T.Normalize(mean=mean, std=std),
        ]
        return T.Compose(augmentations)
    else:
        return T.Compose([
            OpenCVResize(input_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Transform 교체 래퍼 (random_split val 전용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _TransformOverrideSubset(Dataset):
    """
    random_split이 반환하는 Subset의 transform을 교체하는 래퍼

    ┌──────────────────────────────────────────────────────┐
    │ 문제: random_split → Subset은 원본 Dataset의 transform│
    │       을 그대로 사용 → val에 학습 증강이 적용됨        │
    │                                                      │
    │ 해결: 원본 Dataset에서 이미지+레이블을 가져온 뒤        │
    │       새 transform을 적용                             │
    └──────────────────────────────────────────────────────┘

    Args:
        subset: torch.utils.data.Subset (random_split 결과)
        transform: 교체할 새 transform (증강 없는 검증용)
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        # 상위 코드에서 .dataset.classes 등에 접근할 수 있도록 위임
        self.dataset = subset.dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        """원본 Dataset의 raw 이미지를 새 transform으로 처리"""
        # Subset.indices[idx]로 원본 인덱스 획득
        original_idx = self.subset.indices[idx]
        base_dataset = self.subset.dataset

        # 원본 픽셀을 OpenCV로 읽고 검증용 변환 적용
        img_path, label = base_dataset.samples[original_idx]
        image = read_image(img_path, base_dataset.in_channels)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @property
    def classes(self):
        return self.subset.dataset.classes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Classification 데이터셋
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ClassificationDataset(Dataset):
    """
    폴더 기반 이미지 분류 데이터셋

    폴더명이 클래스명이 됨 (ImageFolder 형식)

    Args:
        root: 데이터 루트 경로 (하위에 클래스별 폴더)
        transform: 이미지 변환 파이프라인
        in_channels: 입력 채널 수 (1=Grayscale, 3=RGB)
        extensions: 지원 이미지 확장자
    """

    # 지원 이미지 확장자
    SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self, root: str, transform=None,
                 in_channels: int = 3,
                 extensions: set = None, class_to_idx: Optional[Dict[str, int]] = None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.in_channels = in_channels
        self.extensions = extensions or self.SUPPORTED_EXT

        # 클래스 목록 & 인덱스 매핑 구축
        self.classes, self.class_to_idx = self._find_classes()
        if class_to_idx is not None:
            unknown = set(self.classes) - set(class_to_idx)
            if unknown:
                raise ValueError(f"학습에 없는 검증 클래스: {sorted(unknown)}")
            self.class_to_idx = dict(class_to_idx)
            self.classes = sorted(class_to_idx, key=class_to_idx.get)
        # 이미지 경로 & 레이블 목록
        self.samples = self._make_dataset()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"데이터가 없습니다: {root}\n"
                f"폴더 구조: root/class_name/image.jpg 형식이어야 합니다."
            )

        print(f"📂 Classification 데이터셋 로드: {root}")
        print(f"   클래스 수: {len(self.classes)}")
        print(f"   이미지 수: {len(self.samples)}")
        for cls_name, idx in self.class_to_idx.items():
            count = sum(1 for _, label in self.samples if label == idx)
            print(f"   - {cls_name}: {count}장")

    def _find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """하위 폴더명을 클래스로 인식"""
        classes = sorted([
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self) -> List[Tuple[str, int]]:
        """이미지 경로와 레이블 쌍 목록 생성"""
        samples = []
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = os.path.join(self.root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.extensions:
                    path = os.path.join(cls_dir, fname)
                    samples.append((path, cls_idx))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        # 이미지 로드 — 채널 수에 따라 RGB 또는 Grayscale 변환
        image = read_image(path, self.in_channels)
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_names(self) -> List[str]:
        """클래스 이름 목록 반환"""
        return self.classes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Segmentation 데이터셋
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SegmentationDataset(Dataset):
    """
    이미지 + 마스크 세그멘테이션 데이터셋

    마스크: 단일 채널, 픽셀값 = 클래스 인덱스 (0: 배경)

    Args:
        img_dir: 이미지 디렉토리
        mask_dir: 마스크 디렉토리
        input_size: 리사이즈 크기 (H, W)
        is_train: 학습 모드 여부
        flip_prob: 수평 뒤집기 확률
    """
    SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self, img_dir: str, mask_dir: str,
                 input_size: Tuple[int, int] = (320, 320),
                 in_channels: int = 3,
                 is_train: bool = True,
                 flip_prob: float = 0.5, num_classes: Optional[int] = None):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.in_channels = in_channels
        self.is_train = is_train
        self.flip_prob = flip_prob
        self.num_classes = num_classes

        # 채널 수에 맞는 정규화 값 선택
        self.mean, self.std = get_normalize_params(in_channels)

        # 이미지-마스크 쌍 매칭
        self.pairs = self._find_pairs()

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"이미지-마스크 쌍을 찾을 수 없습니다.\n"
                f"  이미지: {img_dir}\n"
                f"  마스크: {mask_dir}"
            )

        print(f"📂 Segmentation 데이터셋 로드")
        print(f"   이미지-마스크 쌍: {len(self.pairs)}")

    def _find_pairs(self) -> List[Tuple[str, str]]:
        from spatial_data import semantic_pairs
        return semantic_pairs(self.img_dir, self.mask_dir)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        # 이미지 로드 — 채널 수에 따라 RGB 또는 Grayscale 변환
        image = Image.fromarray(read_image(img_path, self.in_channels))
        # 팔레트 마스크는 표시 색이 아닌 인덱스를 보존한다.
        with Image.open(mask_path) as source_mask:
            mask = source_mask.copy()
        if mask.mode not in ("P", "L", "I", "I;16"):
            raise ValueError(
                f"정답 마스크는 클래스 인덱스 이미지 필요 (팔레트/L/I): {mask_path}"
            )

        if image.size != mask.size:
            raise ValueError(f"이미지/마스크 원본 크기 불일치: {img_path} {image.size}, {mask_path} {mask.size}")

        # 리사이즈 (이미지: bilinear, 마스크: nearest)
        image = Image.fromarray(resize(np.asarray(image), self.input_size))
        import cv2
        mask = Image.fromarray(cv2.resize(np.asarray(mask), tuple(reversed(self.input_size)), interpolation=cv2.INTER_NEAREST))

        # 학습 시 동기화된 데이터 증강
        if self.is_train:
            # 수평 뒤집기 (이미지와 마스크 동시)
            if random.random() < self.flip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 텐서 변환
        image = T.functional.to_tensor(image)  # (C, H, W), [0, 1]
        image = T.functional.normalize(image, self.mean, self.std)

        # 마스크: (H, W) long 텐서, 값 = 클래스 인덱스
        mask_array = np.array(mask, dtype=np.int64)
        valid = mask_array[mask_array != 255]
        if valid.size and (valid.min() < 0 or
                           (self.num_classes is not None and valid.max() >= self.num_classes)):
            raise ValueError(f"정답 마스크 클래스 범위 오류: {mask_path}")
        mask = torch.from_numpy(mask_array)

        return image, mask


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Anomaly Detection 데이터셋
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AnomalyDataset(Dataset):
    """
    이상 탐지 데이터셋 — 정상(good) 이미지만 학습

    재구성 기반 이상 탐지:
    - 학습: 정상 이미지 입력 → 동일 이미지 재구성이 목표
    - 추론: 재구성 오차가 크면 이상 판정

    폴더 구조:
        train/good/      ← 정상 이미지 (학습용)
        test/good/       ← 정상 이미지 (테스트)
        test/defect/     ← 이상 이미지 (테스트)

    Args:
        root: 데이터 루트 (train/ 또는 test/)
        input_size: 리사이즈 크기
        is_train: 학습 모드
    """
    SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self, root: str,
                 input_size: Tuple[int, int] = (224, 224),
                 in_channels: int = 3,
                 is_train: bool = True, flip_prob: float = 0.5):
        super().__init__()
        self.root = root
        self.input_size = input_size
        self.in_channels = in_channels
        self.is_train = is_train
        self.flip_prob = flip_prob

        # 채널 수에 맞는 정규화 값 선택
        self.mean, self.std = get_normalize_params(in_channels)

        # 이미지 목록 수집 (하위 폴더 재귀 탐색)
        self.samples = []
        self.labels = []  # 0=good, 1=defect
        self._scan_directory(root)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"이미지가 없습니다: {root}\n"
                f"train/good/ 또는 test/{{good,defect}}/ 구조를 확인하세요."
            )

        print(f"📂 Anomaly 데이터셋 로드: {root}")
        print(f"   이미지 수: {len(self.samples)}")
        good_count = sum(1 for l in self.labels if l == 0)
        defect_count = sum(1 for l in self.labels if l == 1)
        print(f"   Good: {good_count} | Defect: {defect_count}")

    def _scan_directory(self, root):
        """하위 폴더 재귀 탐색하여 이미지 수집"""
        for dirpath, dirnames, filenames in os.walk(root):
            # 폴더명으로 라벨 결정
            folder_name = os.path.basename(dirpath).lower()
            label = 1 if "defect" in folder_name or "ng" in folder_name else 0

            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.SUPPORTED_EXT:
                    self.samples.append(os.path.join(dirpath, fname))
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.samples[idx]
        label = self.labels[idx]

        # 이미지 로드 — 채널 수에 따라 RGB 또는 Grayscale 변환
        image = Image.fromarray(read_image(img_path, self.in_channels))
        image = Image.fromarray(resize(np.asarray(image), self.input_size))

        # 학습 시 약간의 증강 (너무 강하면 재구성이 어려워짐)
        if self.is_train:
            if random.random() < self.flip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 텐서 변환 (정규화 안함 — 재구성 타겟은 [0,1] 범위)
        image_tensor = T.functional.to_tensor(image)  # (C, H, W), [0, 1]

        # 재구성 기반: 입력 = 타겟 (정규화된 입력, 원본 타겟)
        input_normalized = T.functional.normalize(
            image_tensor.clone(), self.mean, self.std
        )

        # 학습: (normalized_input, original_image)
        # 테스트: (normalized_input, label)
        if self.is_train:
            return input_normalized, image_tensor
        else:
            return input_normalized, torch.tensor(label, dtype=torch.long)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Detection 데이터셋 (정규화 좌표 포맷)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DetectionDataset(Dataset):
    """
    정규화 좌표 포맷 객체 탐지 데이터셋

    레이블 파일 형식 (각 줄):
        class_id  x_center  y_center  width  height
        (모두 0~1 정규화 좌표)

    폴더 구조:
        images/{train,val,test}/  ← 이미지
        labels/{train,val,test}/  ← 레이블 (.txt)
        파일명 매칭: img001.jpg ↔ img001.txt

    Args:
        img_dir: 이미지 디렉토리
        label_dir: 레이블 디렉토리
        input_size: 리사이즈 크기
        num_classes: 클래스 수
        is_train: 학습 모드
        max_objects: 이미지 당 최대 객체 수
    """
    SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self, img_dir: str, label_dir: str,
                 input_size: Tuple[int, int] = (416, 416),
                 num_classes: int = 20,
                 in_channels: int = 3,
                 is_train: bool = True,
                 max_objects: int = 50, flip_prob: float = 0.5):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.is_train = is_train
        self.max_objects = max_objects
        self.flip_prob = flip_prob

        # 채널 수에 맞는 정규화 값 선택
        self.mean, self.std = get_normalize_params(in_channels)

        # 이미지-레이블 쌍 매칭
        self.samples = self._find_pairs()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"이미지-레이블 쌍을 찾을 수 없습니다.\n"
                f"  이미지: {img_dir}\n"
                f"  레이블: {label_dir}\n"
                f"  레이블 형식: class_id x_center y_center width height"
            )

        print(f"📂 Detection 데이터셋 로드")
        print(f"   이미지-레이블 쌍: {len(self.samples)}")

    def _find_pairs(self) -> List[Tuple[str, str]]:
        from pathlib import Path
        from spatial_data import image_files
        image_root = Path(self.img_dir)
        return [(str(path), str(Path(self.label_dir) / path.relative_to(image_root).with_suffix(".txt")))
                for path in image_files(image_root)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.samples[idx]

        # 이미지 로드 & 리사이즈 — 채널 수에 따라 RGB 또는 Grayscale
        image = Image.fromarray(read_image(img_path, self.in_channels))
        image = Image.fromarray(resize(np.asarray(image), self.input_size))

        # 학습 시 데이터 증강
        flip = False
        if self.is_train and random.random() < self.flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flip = True

        # 텐서 변환 + 정규화
        image = T.functional.to_tensor(image)
        image = T.functional.normalize(image, self.mean, self.std)

        from spatial_data import read_spatial_labels
        rows = read_spatial_labels(label_path, "detect", self.num_classes)
        if len(rows) > self.max_objects:
            raise ValueError(f"객체 수 {len(rows)}개가 max_objects={self.max_objects} 초과: {label_path}")
        targets = torch.zeros(self.max_objects, 5)
        if rows:
            labels = torch.tensor(rows, dtype=torch.float32)
            if flip:
                labels[:, 1] = 1.0 - labels[:, 1]
            targets[:len(rows)] = labels

        return image, targets


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  데이터 로더 팩토리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_classification_loaders(
    data_root: str,
    input_size: Tuple[int, int] = (224, 224),
    batch_size: int = 8,
    num_workers: int = 0,
    in_channels: int = 3,
    val_split: float = 0.2,
    seed: int = 0,
    **aug_kwargs
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Classification 데이터 로더 생성

    데이터 구조에 따라 두 가지 모드:
    1. train/val 하위 폴더가 있으면 → 각각 사용
    2. 없으면 → 전체를 val_split 비율로 분할

    Args:
        in_channels: 입력 채널 수 (1=Grayscale, 3=RGB)

    Returns:
        train_loader, val_loader, class_names
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        # train/val 폴더가 이미 분리되어 있는 경우
        train_transform = get_classification_transforms(
            input_size, is_train=True, in_channels=in_channels, **aug_kwargs
        )
        val_transform = get_classification_transforms(
            input_size, is_train=False, in_channels=in_channels
        )
        train_dataset = ClassificationDataset(
            train_dir, train_transform, in_channels=in_channels
        )
        val_dataset = ClassificationDataset(
            val_dir, val_transform, in_channels=in_channels,
            class_to_idx=train_dataset.class_to_idx,
        )
    else:
        # ┌──────────────────────────────────────────────────────────┐
        # │ 전체 데이터를 train/val 분할하는 경우                      │
        # │                                                          │
        # │ random_split은 Subset을 반환하므로 transform을 직접       │
        # │ 바꿀 수 없다. val에 학습 증강(flip, rotation, jitter)이   │
        # │ 적용되면 검증 지표가 비결정적이 되어 모델 선택이 불안정.    │
        # │                                                          │
        # │ 해결: _TransformOverrideSubset 래퍼로 val subset의        │
        # │       transform만 증강 없는 버전으로 교체한다.             │
        # └──────────────────────────────────────────────────────────┘
        train_transform = get_classification_transforms(
            input_size, is_train=True, in_channels=in_channels, **aug_kwargs
        )
        val_transform = get_classification_transforms(
            input_size, is_train=False, in_channels=in_channels
        )
        full_dataset = ClassificationDataset(
            train_dir if os.path.isdir(train_dir) else data_root,
            train_transform, in_channels=in_channels
        )

        # 분할
        total = len(full_dataset)
        if total < 2 or not 0 < val_split < 1:
            raise ValueError("자동 분할에는 이미지 2장 이상과 0~1 사이 검증 비율 필요")
        val_size = max(1, min(total - 1, int(total * val_split)))
        train_size = total - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        train_dataset = train_subset
        # val subset: 증강 없는 transform으로 교체
        val_dataset = _TransformOverrideSubset(val_subset, val_transform)

    class_names = (train_dataset.classes
                   if hasattr(train_dataset, 'classes')
                   else train_dataset.dataset.classes)

    # DataLoader 생성 (CPU 환경: pin_memory=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # CPU 환경에서는 False
        drop_last=False    # 작은 데이터셋도 실제 학습 배치를 유지
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, class_names


def create_segmentation_loaders(
    data_root: str,
    input_size: Tuple[int, int] = (320, 320),
    batch_size: int = 4,
    num_workers: int = 0,
    in_channels: int = 3,
    flip_prob: float = 0.5,
    num_classes: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Segmentation 데이터 로더 생성

    필요 구조:
        data_root/images/train/, data_root/images/val/
        data_root/masks/train/,  data_root/masks/val/

    Args:
        in_channels: 입력 채널 수 (1=Grayscale, 3=RGB)

    Returns:
        train_loader, val_loader
    """
    train_dataset = SegmentationDataset(
        img_dir=os.path.join(data_root, "images", "train"),
        mask_dir=os.path.join(data_root, "masks", "train"),
        input_size=input_size,
        in_channels=in_channels,
        is_train=True,
        flip_prob=flip_prob,
        num_classes=num_classes,
    )
    val_dataset = SegmentationDataset(
        img_dir=os.path.join(data_root, "images", "val"),
        mask_dir=os.path.join(data_root, "masks", "val"),
        input_size=input_size,
        in_channels=in_channels,
        is_train=False,
        num_classes=num_classes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader


def create_anomaly_loaders(
    data_root: str,
    input_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    num_workers: int = 0,
    in_channels: int = 3,
    val_split: float = 0.2,
    flip_prob: float = 0.5,
    seed: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Anomaly Detection 데이터 로더 생성

    학습: data_root/train/good/ 의 정상 이미지
    검증: 학습 데이터에서 val_split 비율만큼 분할

    Args:
        in_channels: 입력 채널 수 (1=Grayscale, 3=RGB)

    Returns:
        train_loader, val_loader
    """
    train_dir = os.path.join(data_root, "train")
    good_dir = os.path.join(train_dir, "good")
    if not os.path.isdir(good_dir):
        raise ValueError("정상 학습 이미지 경로 필요: train/good/")

    full_dataset = AnomalyDataset(
        root=good_dir,
        input_size=input_size,
        in_channels=in_channels,
        is_train=True,
        flip_prob=flip_prob,
    )

    # 학습/검증 분할
    total = len(full_dataset)
    if total < 2 or not 0 < val_split < 1:
        raise ValueError("정상 검증 분할에는 이미지 2장 이상과 0~1 사이 검증 비율 필요")
    val_size = max(1, min(total - 1, int(total * val_split)))
    train_size = total - val_size
    train_dataset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    # 반환값은 재구성 타겟 쌍을 유지하되 검증 증강만 제거한다.
    val_base = AnomalyDataset(good_dir, input_size=input_size,
                              in_channels=in_channels, is_train=True,
                              flip_prob=0.0)
    val_dataset = torch.utils.data.Subset(val_base, val_subset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def create_detection_loaders(
    data_root: str,
    input_size: Tuple[int, int] = (416, 416),
    batch_size: int = 8,
    num_workers: int = 0,
    num_classes: int = 20,
    in_channels: int = 3,
    max_objects: int = 50,
    flip_prob: float = 0.5,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Detection 데이터 로더 생성 (정규화 좌표 포맷)

    필요 구조:
        data_root/images/{train,val}/
        data_root/labels/{train,val}/

    Args:
        in_channels: 입력 채널 수 (1=Grayscale, 3=RGB)

    Returns:
        train_loader, val_loader
    """
    from spatial_data import validate_spatial_dataset
    validate_spatial_dataset(data_root, "detect", num_classes)
    train_dataset = DetectionDataset(
        img_dir=os.path.join(data_root, "images", "train"),
        label_dir=os.path.join(data_root, "labels", "train"),
        input_size=input_size,
        num_classes=num_classes,
        in_channels=in_channels,
        is_train=True,
        max_objects=max_objects,
        flip_prob=flip_prob,
    )

    val_img_dir = os.path.join(data_root, "images", "val")
    val_label_dir = os.path.join(data_root, "labels", "val")

    val_loader = None
    if os.path.isdir(val_img_dir):
        try:
            val_dataset = DetectionDataset(
                img_dir=val_img_dir,
                label_dir=val_label_dir,
                input_size=input_size,
                num_classes=num_classes,
                in_channels=in_channels,
                is_train=False,
                max_objects=max_objects,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
            )
        except RuntimeError:
            pass  # val 데이터 없으면 None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader
